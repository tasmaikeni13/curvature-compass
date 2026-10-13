"""Device correctness suite with an independently executed CPU reference."""
import argparse
from dataclasses import replace
from pathlib import Path
import tempfile
import numpy as np
from .artifacts import atomic_json, environment, source_hashes
from .config import ModelConfig, OptimizerConfig, RunConfig
from .model import Decoder
from .optim import acquire, direction
from .runtime import host_array, host_stats, initialize, mesh_and_shardings, sync


def fixture():
    model = Decoder(ModelConfig(vocab_size=127, padded_vocab_size=128, width=32, layers=2,
                                heads=4, mlp_width=64, sequence_length=16))
    parameters = model.initialize(41000)
    x = (np.arange(16 * 16).reshape(16, 16) % 120 + 1).astype(np.int32)
    y = ((x + 3) % 127).astype(np.int32)
    mask = np.ones(x.shape, dtype=np.float32); mask[-1, -9:] = 0
    vector = np.random.default_rng(41001).normal(size=len(parameters)).astype(np.float32)
    vector /= np.linalg.norm(vector)
    return model, parameters, (x, y, mask), vector


def run(output, reference, create_reference=False):
    import jax
    import jax.numpy as jnp
    from . import checkpoint
    model, parameters_host, batch_host, vector_host = fixture()
    mesh, replicated, sharded = mesh_and_shardings()
    parameters = jax.device_put(parameters_host, replicated)
    vector = jax.device_put(vector_host, replicated)
    size = batch_host[0].shape[0] // jax.process_count()
    start = size * jax.process_index()
    batch = tuple(jax.make_array_from_process_local_data(sharded, a[start:start + size], global_shape=a.shape) for a in batch_host)
    gradient_function = jax.jit(jax.value_and_grad(model.loss), in_shardings=(replicated, (sharded,) * 3), out_shardings=replicated)
    # Differentiate parameters only; integer token tangents are never needed.
    hvp_function = jax.jit(lambda p, b, v: jax.jvp(jax.grad(lambda z: model.loss(z, b)), (p,), (v,))[1],
                           in_shardings=(replicated, (sharded,) * 3, replicated), out_shardings=replicated)
    loss, gradient = gradient_function(parameters, batch)
    hvp = hvp_function(parameters, batch, vector)
    jax.block_until_ready((loss, gradient, hvp))
    if create_reference:
        if jax.default_backend() != "cpu":
            raise ValueError("Reference creation must execute independently on CPU")
        Path(reference).parent.mkdir(parents=True, exist_ok=True)
        np.savez(reference, loss=host_array(loss), gradient=host_array(gradient), hvp=host_array(hvp))
    checks = {}

    def relative(a, b):
        a, b = np.asarray(a), np.asarray(b)
        return float(np.linalg.norm(a - b) / max(np.linalg.norm(b), 1e-30))

    with np.load(reference, allow_pickle=False) as gold:
        checks["cpu_reference_gradient"] = {"error": relative(host_array(gradient), gold["gradient"]), "limit": 1e-4}
        checks["cpu_reference_hvp"] = {"error": relative(host_array(hvp), gold["hvp"]), "limit": 1e-4}
        checks["cpu_reference_loss"] = {"error": abs(float(host_array(loss)) - float(gold["loss"])), "limit": 1e-5}
    epsilon = .002
    plus = gradient_function(parameters + epsilon * vector, batch)[1]
    minus = gradient_function(parameters - epsilon * vector, batch)[1]
    checks["finite_difference_hvp"] = {"error": relative(host_array((plus - minus) / (2 * epsilon)), host_array(hvp)), "limit": .02}
    for name, options in (("sdpa", {"attention": "sdpa"}), ("remat", {"rematerialize": True}),
                          ("accumulation", {"accumulation_steps": 2}), ("bf16", {"compute_dtype": "bfloat16"})):
        # Accumulation is checked on a replicated batch here because its
        # microbatch size (8) is smaller than this pod's 16-device data mesh.
        if name == "accumulation":
            candidate_batch = tuple(jax.device_put(a, replicated) for a in batch_host)
            batch_in = (replicated,) * 3
        else:
            candidate_batch, batch_in = batch, (sharded,) * 3
        candidate = lambda p, b: model.training_loss(p, b, **options)
        candidate_grad = jax.jit(jax.value_and_grad(candidate), in_shardings=(replicated, batch_in), out_shardings=replicated)
        candidate_hvp = jax.jit(lambda p, b, v: jax.jvp(jax.grad(lambda z: candidate(z, b)), (p,), (v,))[1],
                                in_shardings=(replicated, batch_in, replicated), out_shardings=replicated)
        candidate_loss, candidate_gradient = candidate_grad(parameters, candidate_batch)
        limit = .03 if name == "bf16" else 1e-4
        checks[name + "_gradient"] = {"error": relative(host_array(candidate_gradient), host_array(gradient)), "limit": limit}
        if name != "bf16":
            value = candidate_hvp(parameters, candidate_batch, vector)
            checks[name + "_hvp"] = {"error": relative(host_array(value), host_array(hvp)), "limit": 1e-4}

    rng = np.random.default_rng(41002)
    raw = rng.normal(size=(12, 12))
    H = (raw @ raw.T + np.eye(12)).astype(np.float32)
    g_host = rng.normal(size=12).astype(np.float32)
    prior_host = np.linalg.qr(rng.normal(size=(12, 4)))[0].T.astype(np.float32)
    g, prior = jax.device_put(g_host, replicated), jax.device_put(prior_host, replicated)
    for cell in "ABCD":
        config = OptimizerConfig(configuration=cell, cap=4)
        compute = jax.jit(lambda x: acquire(lambda v: jnp.matmul(jnp.asarray(H), v, precision=jax.lax.Precision.HIGHEST), x, prior, config),
                          in_shardings=replicated, out_shardings=replicated)
        state, stats = compute(g)
        actual = direction(g, state, config)[0]
        count = int(host_array(state.count))
        Q = host_array(state.basis)[:count].T
        expected = -g_host + Q @ (Q.T @ g_host - np.linalg.solve(Q.T @ H @ Q + np.eye(count), Q.T @ g_host))
        checks[cell + "_dense_projected_inverse"] = {"error": relative(host_array(actual), expected), "limit": 1e-4}
        checks[cell + "_count_and_descent"] = {"passed": bool(host_stats(stats)["finite"] and np.dot(g_host, host_array(actual)) < 0 and (count == 4 if cell in "AB" else 1 <= count <= 4))}
    config = OptimizerConfig(configuration="D", cap=4)
    stopped, stats = jax.jit(lambda x: acquire(lambda v: v, x, jnp.eye(4), config), in_shardings=replicated, out_shardings=replicated)(jax.device_put(np.ones(4, np.float32), replicated))
    checks["actual_stopped_prefix"] = {"passed": int(host_array(stopped.count)) == 1 and bool(host_stats(stats)["stopped_by_policy"])}

    config = OptimizerConfig(configuration="D", cap=4, refresh_interval=2)
    initial = jax.device_put((jnp.zeros((4, model.parameter_count), jnp.float32), jnp.eye(4, dtype=jnp.float32), jnp.array(0, jnp.int32), jnp.array(-1, jnp.int32)), replicated)
    from .optim import TomographyState
    initial = TomographyState(*initial)

    def step(p, state, i, batch):
        gradient = jax.grad(model.loss)(p, batch)
        _, hvp = jax.linearize(jax.grad(lambda z: model.loss(z, batch)), p)
        state = jax.lax.cond(i % 2 == 0, lambda _: acquire(hvp, gradient, jnp.zeros((1, 1)), config, i)[0], lambda _: state, None)
        delta, _ = direction(gradient, state, config)
        return p + .001 * delta, state

    step_function = jax.jit(step, in_shardings=(replicated, replicated, replicated, (sharded,) * 3), out_shardings=replicated)
    p, state = parameters, initial
    for i in range(3):
        p, state = step_function(p, state, jax.device_put(np.array(i, np.int32), replicated), batch)
    interrupted, interrupted_state = step_function(parameters, initial, jax.device_put(np.array(0, np.int32), replicated), batch)
    with tempfile.TemporaryDirectory(prefix="compass-numerics-") as directory:
        checkpoint.save(directory, interrupted, interrupted_state, {"step": 1, "config_hash": "numerics", "data_hash": "fixture", "prior_hash": "none"})
        arrays, _, _ = checkpoint.load(directory, "numerics", "fixture", "none")
        resumed = jax.device_put(arrays["parameters"], replicated)
        resumed_state = jax.device_put(TomographyState(arrays["basis"], arrays["inverse"], arrays["count"], arrays["refresh_step"]), replicated)
        for i in range(1, 3):
            resumed, resumed_state = step_function(resumed, resumed_state, jax.device_put(np.array(i, np.int32), replicated), batch)
        checks["bitwise_device_checkpoint_continuation"] = {"passed": np.array_equal(host_array(p), host_array(resumed)) and all(np.array_equal(host_array(a), host_array(b)) for a, b in zip(state, resumed_state))}
    passed = all(c.get("passed", c.get("error", 1) <= c.get("limit", 0)) for c in checks.values())
    record = {"schema": 1, "status": "passed" if passed else "failed", "checks": checks,
              "environment": environment(), "source_sha256": source_hashes(),
              "scope": "Small-model distributed derivatives and recovery plus quadratic independent dense solves. Full-size numerical comparisons and workload timing are separate benchmark artifacts."}
    path = Path(output)
    atomic_json(path.with_name(path.stem + f"-worker-{jax.process_index()}.json"), record)
    print(record, flush=True)
    sync("numerics-complete")
    if not passed:
        raise ValueError("Device numerical checks failed")
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--reference", required=True, type=Path)
    parser.add_argument("--create-reference", action="store_true")
    parser.add_argument("--distributed", action="store_true")
    parser.add_argument("--coordinator")
    parser.add_argument("--processes", type=int)
    parser.add_argument("--process-id", type=int)
    args = parser.parse_args()
    initialize(args.distributed, args.coordinator, args.processes, args.process_id)
    run(args.output, args.reference, args.create_reference)


if __name__ == "__main__":
    main()
