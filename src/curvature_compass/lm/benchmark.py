"""Bottleneck and numerical benchmarks at the actual planned model/batch sizes."""
import argparse
from dataclasses import replace
import json
from pathlib import Path
import statistics
import time
import numpy as np
from .artifacts import atomic_json, environment, sha256, source_hashes
from .calibrate import load_prior
from .config import load_config
from .data import TokenBlocks, verify_data
from .model import Decoder
from .optim import clip_gradient, direction, make_gradient, make_refresh, make_update
from .runtime import global_batch, host_array, host_stats, initialize, mesh_and_shardings, sync
from .train import compile_timed

CASES = {
    "reference": dict(compute_dtype="float32", attention="reference", rematerialize=False, reuse_linearization=True),
    "bf16": dict(compute_dtype="bfloat16", attention="reference", rematerialize=False, reuse_linearization=True),
    "sdpa": dict(compute_dtype="bfloat16", attention="sdpa", rematerialize=False, reuse_linearization=True),
    "remat": dict(compute_dtype="bfloat16", attention="sdpa", rematerialize=True, reuse_linearization=True),
    "recompute": dict(compute_dtype="bfloat16", attention="sdpa", rematerialize=False, reuse_linearization=False),
}


def benchmark(config, data, prior_path, output, case="reference", repeats=3, profile=None):
    import jax
    import jax.numpy as jnp
    if repeats < 3 or repeats > 10:
        raise ValueError("Important benchmarks require 3–10 measured repetitions")
    baseline_config = replace(config, **CASES["reference"])
    candidate_config = replace(config, **CASES[case])
    decoder = Decoder(config.model)
    verify_data(data, full=False)
    data_hash = sha256(Path(data) / "manifest.json")
    frame, prior_record = load_prior(prior_path, decoder, config.optimizer.cap, data_hash)
    mesh, replicated, batch_sharding = mesh_and_shardings()
    parameters = jax.device_put(decoder.initialize(config.seed), replicated)
    prior = jax.device_put(np.asarray(frame) if not config.optimizer.adaptive_alignment else np.zeros((1, 1), np.float32), replicated)
    del frame
    reader = TokenBlocks(Path(data) / "train.bin", config.model.sequence_length, config.token_budget, config.data_seed)
    batch = global_batch(reader, 0, config.global_batch_size, batch_sharding)
    curvature = global_batch(reader, 0, config.curvature_batch_size, batch_sharding, config.curvature_sequence_length)
    step = jax.device_put(np.array(0, np.int32), replicated)
    rate = jax.device_put(np.array(config.learning_rate, np.float32), replicated)
    compilation = {}

    def functions(run):
        grad = jax.jit(make_gradient(decoder, run), in_shardings=(replicated, (batch_sharding,) * 3), out_shardings=replicated)
        refresh = jax.jit(make_refresh(decoder, run), in_shardings=(replicated, replicated, (batch_sharding,) * 3, replicated, replicated), out_shardings=replicated)
        # Fixed parameters across repetitions: no donation, no hidden training.
        update = jax.jit(make_update(decoder, run), in_shardings=(replicated, replicated, replicated, replicated), out_shardings=replicated)
        return grad, refresh, update

    baseline_grad, baseline_refresh, baseline_update = functions(baseline_config)
    baseline_grad, compilation["baseline_gradient"] = compile_timed(baseline_grad, parameters, batch)
    base_loss, base_gradient = baseline_grad(parameters, batch)
    base_gradient, _ = clip_gradient(base_gradient, config.gradient_clip)
    base_gradient.block_until_ready()
    baseline_refresh, compilation["baseline_refresh"] = compile_timed(baseline_refresh, parameters, base_gradient, curvature, prior, step)
    base_state, base_stats = baseline_refresh(parameters, base_gradient, curvature, prior, step)
    jax.block_until_ready((base_state, base_stats))
    base_direction, _ = direction(base_gradient, base_state, config.optimizer)
    baseline_update, compilation["baseline_update"] = compile_timed(baseline_update, parameters, base_gradient, base_state, rate)
    if case == "reference":
        grad, refresh, update = baseline_grad, baseline_refresh, baseline_update
    else:
        grad, refresh, update = functions(candidate_config)
        grad, compilation["candidate_gradient"] = compile_timed(grad, parameters, batch)
        loss, gradient = grad(parameters, batch)
        gradient, _ = clip_gradient(gradient, config.gradient_clip)
        gradient.block_until_ready()
        refresh, compilation["candidate_refresh"] = compile_timed(refresh, parameters, gradient, curvature, prior, step)
        state, stats = refresh(parameters, gradient, curvature, prior, step)
        jax.block_until_ready((state, stats))
        update, compilation["candidate_update"] = compile_timed(update, parameters, gradient, state, rate)

    loss, gradient = grad(parameters, batch)
    gradient, _ = clip_gradient(gradient, config.gradient_clip)
    state, stats = refresh(parameters, gradient, curvature, prior, step)
    candidate_direction, update_stats = direction(gradient, state, candidate_config.optimizer)
    diagnostics = jax.jit(lambda a, b: jnp.linalg.norm(a - b) / jnp.maximum(jnp.linalg.norm(b), 1e-30), out_shardings=replicated)
    gradient_error = float(host_array(diagnostics(gradient, base_gradient)))
    update_error = float(host_array(diagnostics(candidate_direction, base_direction)))
    loss_error = abs(float(host_array(loss)) - float(host_array(base_loss)))
    stats_host, update_host = host_stats(stats), host_stats(update_stats)
    gradient_limit = .03 if candidate_config.compute_dtype == "bfloat16" else 1e-4
    direction_limit = .05 if candidate_config.compute_dtype == "bfloat16" else 1e-4
    correct = bool(stats_host["finite"] and update_host["direction_finite_and_descent"] and gradient_error <= gradient_limit and update_error <= direction_limit)

    def measured(label):
        row = {"label": label}
        t0 = time.perf_counter()
        with jax.profiler.TraceAnnotation("training_gradient"):
            measured_loss, g = grad(parameters, batch)
            g, _ = clip_gradient(g, config.gradient_clip)
            jax.block_until_ready((measured_loss, g))
        t1 = time.perf_counter()
        with jax.profiler.TraceAnnotation("curvature_acquisition"):
            measured_state, measured_stats = refresh(parameters, g, curvature, prior, step)
            jax.block_until_ready((measured_state, measured_stats))
        t2 = time.perf_counter()
        with jax.profiler.TraceAnnotation("cached_preconditioner_update"):
            p, metrics = update(parameters, g, measured_state, rate)
            jax.block_until_ready((p, metrics))
        t3 = time.perf_counter()
        row.update(gradient_seconds=t1 - t0, curvature_seconds=t2 - t1, update_seconds=t3 - t2,
                   refresh_step_seconds=t3 - t0, hvp_count=int(host_array(measured_stats["hvp_count"])),
                   loss=float(host_array(measured_loss)))
        row["cached_step_seconds"] = row["gradient_seconds"] + row["update_seconds"]
        row["amortized_step_seconds_estimate"] = row["cached_step_seconds"] + row["curvature_seconds"] / config.optimizer.refresh_interval
        return row

    warmup = measured("warmup")
    raw = [measured(f"repeat-{i}") for i in range(repeats)]
    cached_raw = []
    for i in range(repeats):
        # Measure a genuine cached step consecutively, without acquisition
        # between gradient and update; frozen parameters keep trials matched.
        t = time.perf_counter()
        cached_loss, cached_gradient = grad(parameters, batch)
        cached_gradient, _ = clip_gradient(cached_gradient, config.gradient_clip)
        cached_parameters, cached_stats = update(parameters, cached_gradient, state, rate)
        jax.block_until_ready((cached_loss, cached_parameters, cached_stats))
        cached_raw.append({"repeat": i, "seconds": time.perf_counter() - t, "hvp_count": 0})
    trace_files, profiled = {}, None
    if profile is not None:
        with jax.profiler.trace(str(profile), create_perfetto_link=False):
            profiled = measured("profiled-not-timed")
        for path in Path(profile).rglob("*"):
            if path.is_file():
                trace_files[str(path)] = {"bytes": path.stat().st_size, "sha256": sha256(path)}
    names = ("gradient_seconds", "curvature_seconds", "update_seconds", "refresh_step_seconds", "cached_step_seconds", "amortized_step_seconds_estimate")
    medians = {key: statistics.median(r[key] for r in raw) for key in names}
    medians["cached_step_seconds"] = statistics.median(r["seconds"] for r in cached_raw)
    medians["amortized_step_seconds_estimate"] = medians["cached_step_seconds"] + medians["curvature_seconds"] / config.optimizer.refresh_interval
    record = {"schema": 1, "status": "passed" if correct else "rejected_numerical", "case": case,
              "configuration": config.optimizer.configuration, "parameter_count": decoder.parameter_count,
              "config": candidate_config.to_dict(), "control_config_hash": config.fingerprint,
              "batch_shape": list(batch[0].shape), "curvature_shape": list(curvature[0].shape),
              "warmup": warmup, "raw": raw, "medians": medians,
              "cached_raw": cached_raw, "gradient_limit": gradient_limit, "direction_limit": direction_limit,
              "gradient_relative_error": gradient_error, "direction_relative_error": update_error,
              "loss_absolute_error": loss_error, "acquisition": stats_host,
              "baseline_acquisition": host_stats(base_stats), "compilation": compilation,
              "total_hvps_executed": int(host_array(base_stats["hvp_count"])) + stats_host["hvp_count"] * (1 + int(case != "reference")) + warmup["hvp_count"] + sum(r["hvp_count"] for r in raw) + (profiled["hvp_count"] if profiled else 0),
              "total_gradient_batches_executed": 2 + int(case != "reference") + 1 + len(raw) + len(cached_raw) + int(profiled is not None),
              "device_memory_stats": [d.memory_stats() for d in jax.local_devices()],
              "data_manifest_sha256": data_hash, "prior_sha256": sha256(prior_path),
              "source_sha256": source_hashes(), "environment": environment(), "trace_files": trace_files,
              "accounting": "Every repetition measures a full refresh and a cached update at fixed parameters. Amortization is a calculated stage-cost estimate, not measured full-run throughput. Compilation, warmup and trace capture excluded from medians. All curvature products remain float32/reference attention."}
    path = Path(output)
    atomic_json(path.with_name(path.stem + f"-worker-{jax.process_index()}.json"), record)
    if jax.process_index() == 0:
        print(json.dumps({"case": case, "status": record["status"], "medians": medians,
                          "gradient_relative_error": gradient_error, "direction_relative_error": update_error}), flush=True)
    sync("benchmark-complete")
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--prior", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--case", choices=CASES, default="reference")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--profile", type=Path)
    parser.add_argument("--distributed", action="store_true")
    parser.add_argument("--coordinator")
    parser.add_argument("--processes", type=int)
    parser.add_argument("--process-id", type=int)
    args = parser.parse_args()
    initialize(args.distributed, args.coordinator, args.processes, args.process_id)
    benchmark(load_config(args.config), args.data, args.prior, args.output, args.case, args.repeats, args.profile)


if __name__ == "__main__":
    main()
