from dataclasses import replace
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from curvature_compass.lm.config import OptimizerConfig
from curvature_compass.lm.optim import acquire, direction


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
