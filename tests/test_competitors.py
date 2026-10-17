from dataclasses import replace
import inspect
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from curvature_compass.lm import competitors
from curvature_compass.lm.competitors import (BaselineConfig, bartlett_diagonal, empty_baseline_state,
    make_baseline_update, make_sophia_refresh, newton_schulz)
from curvature_compass.lm.config import ModelConfig, RunConfig
from curvature_compass.lm.model import Decoder


def tiny_model():
    return Decoder(ModelConfig(vocab_size=11, padded_vocab_size=16, width=4, layers=1,
                               heads=2, mlp_width=8, sequence_length=4))


def test_adamw_matches_independent_bias_corrected_recurrence():
    model = tiny_model()
    config = RunConfig(model=model.config, optimizer=BaselineConfig(name="adamw", beta1=.7, beta2=.8),
                       weight_decay=.13)
    state = empty_baseline_state(model, config.optimizer)
    rng = np.random.default_rng(16)
    expected = model.initialize(4)
    actual = jnp.asarray(expected)
    m, v = np.zeros_like(expected), np.zeros_like(expected)
    update = jax.jit(make_baseline_update(model, config))
    for step in range(1, 5):
        g = rng.normal(size=expected.size).astype(np.float32)
        m, v = .7 * m + .3 * g, .8 * v + .2 * g * g
        oracle = (m / (1 - .7 ** step)) / (np.sqrt(v / (1 - .8 ** step)) + config.optimizer.epsilon)
        expected = expected * (1 - .003 * .13 * model.decay_mask()) - .003 * oracle
        actual, state, stats = update(actual, jnp.asarray(g), state, .003)
        np.testing.assert_allclose(actual, expected, rtol=3e-5, atol=2e-7)
        assert bool(stats["parameters_finite"]) and int(state.count) == step


def numpy_muon(matrix, steps):
    transposed = matrix.shape[0] > matrix.shape[1]
    x = (matrix.T if transposed else matrix).astype(np.float64)
    x /= np.linalg.norm(x) + 1e-7
    for _ in range(steps):
        a = x @ x.T
        x = 3.4445 * x - 4.7750 * a @ x + 2.0315 * a @ a @ x
    return x.T if transposed else x


@pytest.mark.parametrize("ns_steps", [5, 8])
def test_muon_matches_numpy_matrix_oracle_and_auxiliary_adam(ns_steps):
    model = tiny_model()
    opt = BaselineConfig(name="muon", momentum=.95, ns_steps=ns_steps, auxiliary_lr_ratio=.025)
    config = RunConfig(model=model.config, optimizer=opt, weight_decay=.1)
    initial = model.initialize(4)
    gradient = np.random.default_rng(4).normal(size=initial.size).astype(np.float32)
    update = jax.jit(make_baseline_update(model, config))
    actual, state, stats = update(jnp.asarray(initial), jnp.asarray(gradient), empty_baseline_state(model, opt), .02)
    expected = initial.copy()
    for (name, shape), (start, stop) in zip(model.layout, model.offsets):
        g, p = gradient[start:stop].reshape(shape), initial[start:stop].reshape(shape)
        if name.startswith("block.") and len(shape) == 2:
            mixed = (.05 + .95 * .05) * g
            delta = numpy_muon(mixed.T, ns_steps).T * np.sqrt(max(1, shape[1] / shape[0]))
            lr = .02
        else:
            delta, lr = g / (np.abs(g) + opt.epsilon), .02 * opt.auxiliary_lr_ratio
        expected[start:stop] = (p * (1 - lr * (.1 if len(shape) == 2 else 0)) - lr * delta).reshape(-1)
    np.testing.assert_allclose(actual, expected, rtol=4e-4, atol=2e-6)
    np.testing.assert_array_equal(newton_schulz(jnp.zeros((4, 7)), ns_steps), np.zeros((4, 7)))
    assert bool(stats["parameters_finite"])
    for (name, shape), tensor in zip(model.layout, state.tensors):
        if name.startswith("block.") and len(shape) == 2:
            assert tensor.variance.size == 0  # No unused Adam variance for Muon matrices.


def test_soap_projected_adam_and_reference_initialization():
    model = tiny_model()
    opt = BaselineConfig(name="soap", beta1=.95, beta2=.95, precondition_frequency=1)
    config = RunConfig(model=model.config, optimizer=opt, weight_decay=.1)
    rng = np.random.default_rng(5)
    initial, g1, g2 = model.initialize(3), rng.normal(size=model.parameter_count).astype(np.float32), rng.normal(size=model.parameter_count).astype(np.float32)
    update = jax.jit(make_baseline_update(model, config))
    p, state, stats = update(jnp.asarray(initial), jnp.asarray(g1), empty_baseline_state(model, opt), .001)
    np.testing.assert_array_equal(p, initial)
    assert bool(stats["soap_initialization_batch"])
    p, state, stats = update(p, jnp.asarray(g2), state, .001)
    expected = initial.copy()
    for (_, shape), (start, stop), tensor in zip(model.layout, model.offsets, state.tensors):
        first, second = g1[start:stop].reshape(shape).astype(np.float64), g2[start:stop].reshape(shape).astype(np.float64)
        if len(shape) == 2:
            _, ql = np.linalg.eigh(first @ first.T)
            _, qr = np.linalg.eigh(first.T @ first)
            ql, qr = ql[:, ::-1], qr[:, ::-1]
            projected = ql.T @ second @ qr
            delta = ql @ (projected / (np.abs(projected) + opt.epsilon / np.sqrt(1 - opt.beta2))) @ qr.T
            np.testing.assert_allclose(np.asarray(tensor.left_basis).T @ tensor.left_basis, np.eye(shape[0]), atol=2e-6)
            np.testing.assert_allclose(np.asarray(tensor.right_basis).T @ tensor.right_basis, np.eye(shape[1]), atol=2e-6)
        else:
            delta = second / (np.abs(second) + opt.epsilon / np.sqrt(1 - opt.beta2))
        expected[start:stop] = ((initial[start:stop].reshape(shape) - .001 * delta) * (1 - .001 * (.1 if len(shape) == 2 else 0))).reshape(-1)
    # Rank-deficient rectangular factors have arbitrary null-space eigenvectors;
    # test the full-rank square attention block independently of that ambiguity.
    index = next(i for i, (name, _) in enumerate(model.layout) if name in ("block.0.o_proj.weight", "block.0.attention.weight"))
    start, stop = model.offsets[index]
    np.testing.assert_allclose(np.asarray(p)[start:stop], expected[start:stop], atol=2e-6, rtol=1e-4)
    assert bool(stats["parameters_finite"]) and bool(stats["preconditioner_refreshed"])
    for tensor in state.tensors:
        assert np.all(np.asarray(tensor.variance) >= 0)


def test_soap_preserves_first_moment_under_basis_refresh_and_omits_large_axes():
    model = tiny_model()
    opt = BaselineConfig(name="soap", precondition_frequency=1, max_precondition_dimension=4)
    config = RunConfig(model=model.config, optimizer=opt)
    rng = np.random.default_rng(8)
    update = jax.jit(make_baseline_update(model, config))
    p, state = jnp.asarray(model.initialize(1)), empty_baseline_state(model, opt)
    physical = [np.zeros(shape, np.float32) for _, shape in model.layout]
    for step in range(5):
        g = rng.normal(size=model.parameter_count).astype(np.float32)
        p, state, stats = update(p, jnp.asarray(g), state, .001)
        if step:
            for i, ((_, shape), (start, stop), tensor) in enumerate(zip(model.layout, model.offsets, state.tensors)):
                physical[i] = opt.beta1 * physical[i] + (1 - opt.beta1) * g[start:stop].reshape(shape)
                recovered = np.asarray(tensor.momentum)
                if len(shape) == 2:
                    if shape[0] <= 4:
                        recovered = np.asarray(tensor.left_basis) @ recovered
                    else:
                        assert tensor.left.size == 0
                    if shape[1] <= 4:
                        recovered = recovered @ np.asarray(tensor.right_basis).T
                    else:
                        assert tensor.right.size == 0
                np.testing.assert_allclose(recovered, physical[i], atol=3e-6, rtol=3e-5)


def test_bartlett_diagonal_matches_exact_ggn_in_expectation():
    # Independent two-observation logistic model: exact GGN is mean p(1-p)*x².
    x, theta = np.array([1., 2.], np.float32), .37
    p = 1 / (1 + np.exp(-theta * x))
    labels = np.random.default_rng(9).binomial(1, p, size=(50000, 2))
    gradients = ((p - labels) * x).mean(axis=1)
    estimates = np.asarray(bartlett_diagonal(jnp.asarray(gradients, jnp.float32), 2))
    exact = np.mean(p * (1 - p) * x * x)
    assert abs(estimates.mean() / exact - 1) < .02
    # Squared supervised gradients have a different expectation/normalization.
    assert abs(float(bartlett_diagonal(jnp.asarray((p * x).mean()), 2)) / exact - 1) > .1


def test_sophia_refresh_is_real_sampled_label_gradient_and_stateless():
    model = tiny_model()
    opt = BaselineConfig(name="sophia_g", beta1=.965, beta2=.99)
    config = RunConfig(model=model.config, optimizer=opt)
    parameters = jnp.asarray(model.initialize(7))
    x = jnp.array([[1, 2, 3, 4], [4, 3, 2, 1]])
    mask = jnp.array([[1., 1., 1., 1.], [1., 1., 0., 0.]])
    state = empty_baseline_state(model, opt)
    refresh = jax.jit(make_sophia_refresh(model, config))
    a, stats = refresh(parameters, (x, x, mask), state, jnp.array(2, jnp.int32))
    b, _ = refresh(parameters, (x, x + 1, mask), state, jnp.array(2, jnp.int32))
    for first, second in zip(jax.tree.leaves(a), jax.tree.leaves(b)):
        np.testing.assert_array_equal(first, second)
    assert int(stats["sampled_label_gradient_count"]) == 1 and int(stats["hvp_count"]) == 0
    assert float(stats["ggn_predicted_tokens"]) == 6 and bool(stats["finite"])
    assert sum(float(jnp.sum(t.variance)) for t in a.tensors) > 0
    gradient = np.linspace(-.1, .1, model.parameter_count, dtype=np.float32)
    update = jax.jit(make_baseline_update(model, config))
    actual, _, _ = update(parameters, jnp.asarray(gradient), a, .001)
    h = np.concatenate([np.asarray(t.variance).reshape(-1) for t in a.tensors])
    delta = np.clip((1 - opt.beta1) * gradient / (opt.rho * h + opt.epsilon), -1, 1)
    expected = np.asarray(parameters) * (1 - .001 * config.weight_decay * model.decay_mask()) - .001 * delta
    np.testing.assert_allclose(actual, expected, atol=2e-7, rtol=3e-5)


def test_sophia_full_context_measures_every_position_and_rejects_short_sweep_context():
    from curvature_compass.lm.experiments import SweepRunConfig
    model = Decoder(replace(tiny_model().config, sequence_length=8))
    opt = BaselineConfig(name="sophia_g", epsilon=1e-15)
    config = SweepRunConfig(model=model.config, optimizer=opt, global_batch_size=2, curvature_batch_size=2,
        curvature_sequence_length=8, token_budget=40000, compute_dtype="bfloat16")
    config.validate()
    with pytest.raises(ValueError, match="full model context"):
        replace(config, curvature_sequence_length=4).validate()
    x = jnp.arange(8)[None, :]
    state, _ = jax.jit(make_sophia_refresh(model, config))(jnp.asarray(model.initialize(9)),
        (x, x, jnp.ones_like(x, dtype=jnp.float32)), empty_baseline_state(model, opt), jnp.array(0, jnp.int32))
    token_emb = next(i for i, (name, _) in enumerate(model.layout) if name == "token_embedding")
    assert np.all(np.asarray(state.tensors[token_emb].variance)[:8].sum(axis=1) > 0)
    assert sum(float(jnp.sum(t.variance)) for t in state.tensors) > 0


def test_competitors_zero_pallas_and_pure_jax_ast():
    """Assert competitors.py contains strictly zero custom Pallas or Mosaic kernels."""
    src = inspect.getsource(competitors).lower()
    assert "pallas" not in src, "competitors.py must not reference or import pallas"
    assert "mosaic" not in src, "competitors.py must not reference or import mosaic"


@pytest.mark.parametrize("opt_name", ["adamw", "muon", "soap", "sophia_g"])
def test_baseline_optimizers_execute_pure_jax_on_llama(opt_name):
    """Verify all 4 baseline optimizers step cleanly on modern LLaMA under pure JAX JIT."""
    model = Decoder(ModelConfig(vocab_size=32, padded_vocab_size=32, width=16, layers=2,
                                heads=2, mlp_width=32, sequence_length=16))
    p0 = jnp.asarray(model.initialize(42))
    tokens = jnp.ones((2, 16), dtype=jnp.int32)
    targets = jnp.ones((2, 16), dtype=jnp.int32)
    mask = jnp.ones((2, 16), dtype=jnp.float32)
    batch = (tokens, targets, mask)

    bcfg = BaselineConfig(name=opt_name, ns_steps=5, precondition_frequency=1)
    rcfg = RunConfig(model=model.config, optimizer=bcfg, weight_decay=0.1,
                     compute_dtype="float32", attention="reference")
    state = empty_baseline_state(model, bcfg)

    if opt_name == "sophia_g":
        refresh_fn = jax.jit(make_sophia_refresh(model, rcfg))
        state, rstats = refresh_fn(p0, batch, state, jnp.array(0, jnp.int32))
        assert bool(rstats["finite"])

    loss_fn = lambda p: model.loss(p, batch, compute_dtype="float32", attention="reference")
    grad = jax.grad(loss_fn)(p0)
    update_fn = jax.jit(make_baseline_update(model, rcfg))

    p1, state, ustats = update_fn(p0, grad, state, 0.001)
    assert bool(ustats["parameters_finite"])
    assert bool(ustats["optimizer_state_finite"])
    assert jnp.all(jnp.isfinite(p1))


@pytest.mark.parametrize("opt_name", ["adamw", "muon", "soap", "sophia_g"])
def test_baseline_weight_decay_filtering_on_llama_rmsnorm(opt_name):
    """Verify 0.0 weight decay on 1D RMSNorm scales and positive decay on 2D matrices."""
    model = Decoder(ModelConfig(vocab_size=32, padded_vocab_size=32, width=16, layers=2,
                                heads=2, mlp_width=32, sequence_length=16))
    p0 = jnp.asarray(model.initialize(42))
    zero_grad = jnp.zeros_like(p0)
    rate, wd = 0.01, 0.1

    bcfg = BaselineConfig(name=opt_name, ns_steps=5, precondition_frequency=1)
    rcfg = RunConfig(model=model.config, optimizer=bcfg, weight_decay=wd)
    state = empty_baseline_state(model, bcfg)
    update_fn = jax.jit(make_baseline_update(model, rcfg))

    p1, state, _ = update_fn(p0, zero_grad, state, rate)
    if opt_name == "soap":
        # SOAP skips step 0 as initialization batch; evaluate step 1
        p_eval, _, _ = update_fn(p1, zero_grad, state, rate)
    else:
        p_eval = p1

    p_dict = model.unpack(np.asarray(p_eval))
    p0_dict = model.unpack(np.asarray(p0))

    # 1D RMSNorm scale parameters: must have 0.0 weight decay
    for norm_key in ("block.0.input_norm.weight", "block.0.post_attention_norm.weight", "final_norm.weight"):
        np.testing.assert_array_equal(p_dict[norm_key], p0_dict[norm_key],
                                      err_msg=f"{opt_name} must apply 0.0 weight decay to {norm_key}")

    # 2D projection weights: must decay
    for proj_key in ("block.0.q_proj.weight", "block.0.gate_proj.weight"):
        expected = p0_dict[proj_key] * (1.0 - rate * wd)
        np.testing.assert_allclose(p_dict[proj_key], expected, rtol=1e-5)

    # 2D token embedding: verify exact decay accounting
    if opt_name == "muon":
        expected_emb = p0_dict["token_embedding"] * (1.0 - rate * bcfg.auxiliary_lr_ratio * wd)
        np.testing.assert_allclose(p_dict["token_embedding"], expected_emb, rtol=1e-6)
    else:
        expected_emb = p0_dict["token_embedding"] * (1.0 - rate * wd)
        np.testing.assert_allclose(p_dict["token_embedding"], expected_emb, rtol=1e-5)

