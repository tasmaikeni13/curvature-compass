from dataclasses import replace
import inspect
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from curvature_compass.lm import optim
from curvature_compass.lm.config import ModelConfig, RunConfig, OptimizerConfig
from curvature_compass.lm.model import Decoder
from curvature_compass.lm.optim import acquire, direction, empty_state, make_refresh, make_update, make_gradient


@pytest.mark.parametrize("cell", list("ABCD"))
def test_quadratic_matches_independent_projected_dense_inverse(cell):
    rng = np.random.default_rng(42)
    n, cap = 12, 4
    raw = rng.normal(size=(n, n))
    H = (raw @ raw.T + np.eye(n)).astype(np.float32)
    gradient = rng.normal(size=n).astype(np.float32)
    prior = np.linalg.qr(rng.normal(size=(n, cap)))[0].T.astype(np.float32)
    config = OptimizerConfig(configuration=cell, cap=cap, damping=1., tolerance=.7)
    state, stats = jax.jit(lambda g: acquire(lambda v: jnp.asarray(H) @ v, g, jnp.asarray(prior), config))(jnp.asarray(gradient))
    Q, count = np.asarray(state.basis).T[:, :int(state.count)], int(state.count)
    np.testing.assert_allclose(Q.T @ Q, np.eye(count), atol=3e-6)
    reduced = Q.T @ H @ Q + np.eye(count)
    expected = -gradient + Q @ (Q.T @ gradient - np.linalg.solve(reduced, Q.T @ gradient))
    actual, metrics = direction(jnp.asarray(gradient), state, config)
    np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=2e-5)
    assert bool(metrics["direction_finite_and_descent"])
    assert int(stats["hvp_count"]) == count
    if cell in "AB":
        assert count == cap and not bool(stats["stopped_by_policy"])


def test_adaptive_count_executes_one_query_without_hidden_suffix():
    config = OptimizerConfig(configuration="D", cap=4)
    gradient = jnp.array([1., 2., 3., 4.])
    state, stats = acquire(lambda v: v, gradient, jnp.eye(4), config)
    assert int(state.count) == 1
    assert bool(stats["stopped_by_policy"])
    assert np.count_nonzero(np.asarray(state.basis).any(axis=1)) == 1
    # Fixed count stops on an exhausted invariant subspace, not this policy.
    state, stats = acquire(lambda v: v, gradient, jnp.eye(4), replace(config, configuration="B"))
    assert int(state.count) == 1 and bool(stats["rank_exhausted"])
    assert not bool(stats["stopped_by_policy"])


def test_fixed_frames_share_prefix_and_do_not_depend_on_current_gradient():
    H = jnp.diag(jnp.array([1., 4., 10., 20.]))
    frame = jnp.eye(4)
    base = OptimizerConfig(configuration="A", cap=4)
    full, _ = acquire(lambda v: H @ v, jnp.array([1., 2., 3., 4.]), frame, base)
    stopped, _ = acquire(lambda v: H @ v, jnp.array([1., 0., 0., 0.]), frame, replace(base, configuration="C"))
    assert int(stopped.count) == 1
    np.testing.assert_array_equal(stopped.basis[:1], full.basis[:1])
    changed, _ = acquire(lambda v: H @ v, jnp.array([4., 3., 2., 1.]), frame, base)
    np.testing.assert_array_equal(changed.basis, full.basis)


def test_indefinite_clipping_descent_norm_cap_and_zero_gradient():
    H = jnp.diag(jnp.array([-50., -2., 1., 10.]))
    gradient = jnp.array([1., 2., 3., 4.])
    config = OptimizerConfig(configuration="A", cap=4, max_step_ratio=2.)
    state, stats = acquire(lambda v: H @ v, gradient, jnp.eye(4), config)
    step, metrics = direction(gradient, state, config)
    assert int(stats["clipped_eigenvalues"]) == 2
    assert float(gradient @ step) < 0
    assert np.linalg.norm(step) <= 2.00001 * np.linalg.norm(gradient)
    zero, stats = acquire(lambda v: H @ v, jnp.zeros(4), jnp.eye(4), config)
    assert int(zero.count) == 0 and int(stats["hvp_count"]) == 0
    np.testing.assert_array_equal(direction(jnp.zeros(4), zero, config)[0], np.zeros(4))


def test_nonfinite_hvp_is_detected():
    _, stats = acquire(lambda v: jnp.full_like(v, jnp.nan), jnp.ones(4), jnp.eye(4), OptimizerConfig(configuration="D", cap=4))
    assert not bool(stats["finite"])


def test_optim_zero_pallas_and_pure_jax_ast():
    """Assert optim.py contains strictly zero custom Pallas or Mosaic kernels."""
    src = inspect.getsource(optim).lower()
    assert "pallas" not in src, "optim.py must not reference or import pallas"
    assert "mosaic" not in src, "optim.py must not reference or import mosaic"


@pytest.mark.parametrize("cell", ["A", "B", "C", "D"])
def test_tomography_pure_jax_execution_on_llama(cell):
    """Verify Tomography subspace acquisition and update step on modern LLaMA."""
    model = Decoder(ModelConfig(vocab_size=32, padded_vocab_size=32, width=16, layers=2,
                                heads=2, mlp_width=32, sequence_length=16))
    p0 = jnp.asarray(model.initialize(42))
    tokens = jnp.ones((2, 16), dtype=jnp.int32)
    targets = jnp.ones((2, 16), dtype=jnp.int32)
    mask = jnp.ones((2, 16), dtype=jnp.float32)
    batch = (tokens, targets, mask)

    tcfg = RunConfig(model=model.config, optimizer=OptimizerConfig(configuration=cell, cap=2),
                     weight_decay=0.1, compute_dtype="float32", attention="reference")

    grad_fn = jax.jit(make_gradient(model, tcfg))
    refresh_fn = jax.jit(make_refresh(model, tcfg))
    update_fn = jax.jit(make_update(model, tcfg))

    loss, grad = grad_fn(p0, batch)
    assert jnp.isfinite(loss)
    assert jnp.all(jnp.isfinite(grad))

    prior = jnp.zeros((2, model.parameter_count), dtype=jnp.float32)
    state, rstats = refresh_fn(p0, grad, batch, prior, 0)
    assert bool(rstats["finite"])

    p1, ustats = update_fn(p0, grad, state, 0.001)
    assert bool(ustats["parameters_finite"])
    assert jnp.all(jnp.isfinite(p1))


def test_tomography_weight_decay_filtering_on_llama_rmsnorm():
    """Verify decay_mask zeroes out weight decay on 1D RMSNorm scales in Tomography."""
    model = Decoder(ModelConfig(vocab_size=32, padded_vocab_size=32, width=16, layers=2,
                                heads=2, mlp_width=32, sequence_length=16))
    p0 = jnp.asarray(model.initialize(42))
    zero_grad = jnp.zeros_like(p0)
    rate, wd = 0.01, 0.1

    tcfg = RunConfig(model=model.config, optimizer=OptimizerConfig(configuration="A", cap=2),
                     weight_decay=wd)
    t_state = empty_state(model.parameter_count, 2)
    update_fn = jax.jit(make_update(model, tcfg))

    p_eval, stats = update_fn(p0, zero_grad, t_state, rate)
    assert bool(stats["parameters_finite"])

    p_dict = model.unpack(np.asarray(p_eval))
    p0_dict = model.unpack(np.asarray(p0))

    # 1D RMSNorm scale parameters: strictly 0.0 weight decay
    for norm_key in ("block.0.input_norm.weight", "block.0.post_attention_norm.weight", "final_norm.weight"):
        np.testing.assert_array_equal(p_dict[norm_key], p0_dict[norm_key],
                                      err_msg=f"Tomography must apply 0.0 weight decay to {norm_key}")

    # 2D projection and embedding weights: decayed by (1 - rate * wd)
    for weight_key in ("block.0.q_proj.weight", "block.0.gate_proj.weight", "token_embedding"):
        expected = p0_dict[weight_key] * (1.0 - rate * wd)
        np.testing.assert_allclose(p_dict[weight_key], expected, rtol=1e-5)

