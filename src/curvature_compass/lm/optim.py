"""A common damped subspace inverse with genuinely stopped HVP acquisition."""
from typing import NamedTuple
import jax
import jax.numpy as jnp
from .config import OptimizerConfig

PRECISION = jax.lax.Precision.HIGHEST


class TomographyState(NamedTuple):
    basis: jax.Array              # Rows are measured orthonormal directions.
    inverse: jax.Array            # Inactive rows/columns have zero projection.
    count: jax.Array
    refresh_step: jax.Array


def empty_state(n, cap):
    return TomographyState(jnp.zeros((cap, n), jnp.float32), jnp.eye(cap, dtype=jnp.float32),
                           jnp.array(0, jnp.int32), jnp.array(-1, jnp.int32))


def orthogonalize(vector, basis):
    for _ in range(2):
        vector = vector - jnp.matmul(jnp.matmul(basis, vector, precision=PRECISION), basis, precision=PRECISION)
    return vector


def reduced_solve(Q, Y, gradient, config):
    gram = jnp.matmul(Q, Y.T, precision=PRECISION)
    gram = .5 * (gram + gram.T) + config.damping * jnp.eye(config.cap, dtype=gradient.dtype)
    values, vectors = jnp.linalg.eigh(gram)
    safe = jnp.maximum(values, config.eigenvalue_floor * config.damping)
    inverse = jnp.matmul(vectors / safe, vectors.T, precision=PRECISION)
    rhs = jnp.matmul(Q, gradient, precision=PRECISION)
    coefficients = -jnp.matmul(inverse, rhs, precision=PRECISION)
    residual = gradient + jnp.matmul(coefficients, Y, precision=PRECISION) + config.damping * jnp.matmul(coefficients, Q, precision=PRECISION)
    return inverse, residual, jnp.sum(values < config.eigenvalue_floor * config.damping), jnp.min(values)


def acquire(hvp, gradient, prior, config: OptimizerConfig, step=0):
    """One HVP per executed loop body; C/D do not evaluate discarded suffixes.

    This is an empirical stop for a stochastic, possibly clipped operator.
    It is deliberately not reported as a Phase 1 residual certificate.
    """
    n, cap = gradient.shape[0], config.cap
    initial_norm = jnp.linalg.norm(gradient)
    Q, Y = jnp.zeros((cap, n), gradient.dtype), jnp.zeros((cap, n), gradient.dtype)
    # index, Q, Y, inverse, residual, stopped, breakdown, finite, path, clipped, minimum eigenvalue
    initial = (jnp.array(0, jnp.int32), Q, Y, jnp.eye(cap, dtype=gradient.dtype), gradient,
               initial_norm <= 1e-12, jnp.array(False), jnp.isfinite(initial_norm),
               jnp.zeros(cap, gradient.dtype), jnp.array(0, jnp.int32), jnp.array(config.damping, gradient.dtype))

    def condition(state):
        return (state[0] < cap) & ~state[5] & ~state[6] & state[7]

    def body(state):
        i, Q, Y, inverse, residual, stopped, breakdown, finite, path, clipped, minimum = state
        candidate = residual if config.adaptive_alignment else prior[i]
        candidate = orthogonalize(candidate, Q)
        norm = jnp.linalg.norm(candidate)

        def measured(_):
            q = candidate / norm
            with jax.named_scope("hessian_vector_product"):
                y = hvp(q)
            next_Q, next_Y = Q.at[i].set(q), Y.at[i].set(y)
            with jax.named_scope("direction_and_reduced_solve"):
                inv, r, clips, smallest = reduced_solve(next_Q, next_Y, gradient, config)
            ratio = jnp.linalg.norm(r) / jnp.maximum(initial_norm, 1e-30)
            stop = config.adaptive_count & (i + 1 >= config.min_count) & (ratio <= config.tolerance)
            valid = jnp.all(jnp.isfinite(y)) & jnp.all(jnp.isfinite(inv)) & jnp.isfinite(ratio)
            return (i + 1, next_Q, next_Y, inv, r, stop, breakdown, valid,
                    path.at[i].set(ratio), clips, smallest)

        def exhausted(_):
            return (i, Q, Y, inverse, residual, stopped, jnp.array(True), finite & jnp.isfinite(norm), path, clipped, minimum)

        # Float32 residuals can leave a tiny orthogonal numerical component
        # after an invariant subspace has been solved. Do not normalize that
        # rounding error into an additional information-dependent direction.
        return jax.lax.cond(norm > 1e-6 * jnp.maximum(initial_norm, 1.), measured, exhausted, operand=None)

    final = jax.lax.while_loop(condition, body, initial)
    i, Q, _, inverse, residual, stopped, breakdown, finite, path, clipped, minimum = final
    result = TomographyState(Q, inverse, i, jnp.asarray(step, jnp.int32))
    stats = {"hvp_count": i, "stopped_by_policy": stopped & (initial_norm > 1e-12),
             "rank_exhausted": breakdown, "finite": finite, "residual_ratios": path,
             "residual_ratio": jnp.linalg.norm(residual) / jnp.maximum(initial_norm, 1e-30),
             "clipped_eigenvalues": clipped, "minimum_damped_eigenvalue": minimum}
    return result, stats


def direction(gradient, state, config):
    projected = jnp.matmul(state.basis, gradient, precision=PRECISION)
    coefficients = projected / config.damping - jnp.matmul(state.inverse, projected, precision=PRECISION)
    result = -gradient / config.damping + jnp.matmul(coefficients, state.basis, precision=PRECISION)
    limit = config.max_step_ratio * jnp.linalg.norm(gradient)
    result *= jnp.minimum(1., limit / jnp.maximum(jnp.linalg.norm(result), 1e-30))
    descent = jnp.vdot(gradient, result, precision=PRECISION)
    valid = jnp.all(jnp.isfinite(result)) & ((descent < 0) | (jnp.linalg.norm(gradient) <= 1e-12))
    return result, {"direction_finite_and_descent": valid, "gradient_dot_direction": descent,
                    "direction_norm": jnp.linalg.norm(result)}


def make_refresh(decoder, run_config):
    """Return an unjitted function so callers can bind distributed shardings."""
    def refresh(parameters, gradient, batch, prior, step):
        loss = lambda p: decoder.loss(p, batch, compute_dtype="float32", attention="reference",
                                      rematerialize=run_config.rematerialize)
        grad = jax.grad(loss)
        if run_config.reuse_linearization:
            with jax.named_scope("curvature_linearization"):
                curvature_gradient, hvp = jax.linearize(grad, parameters)
        else:
            curvature_gradient = grad(parameters)
            hvp = lambda v: jax.jvp(grad, (parameters,), (v,))[1]
        state, stats = acquire(hvp, gradient, prior, run_config.optimizer, step)
        stats["curvature_gradient_norm"] = jnp.linalg.norm(curvature_gradient)
        return state, stats
    return refresh


def make_gradient(decoder, config):
    return jax.value_and_grad(lambda p, batch: decoder.training_loss(
        p, batch, config.accumulation_steps, compute_dtype=config.compute_dtype,
        attention=config.attention, rematerialize=config.rematerialize))


def make_update(decoder, config):
    decay_mask = jnp.asarray(decoder.decay_mask())
    def update(parameters, gradient, state, rate):
        delta, stats = direction(gradient, state, config.optimizer)
        parameters = parameters * (1 - rate * config.weight_decay * decay_mask) + rate * delta
        stats["parameters_finite"] = jnp.all(jnp.isfinite(parameters))
        return parameters, stats
    return update


def clip_gradient(gradient, limit):
    norm = jnp.linalg.norm(gradient)
    return gradient * jnp.minimum(1., limit / jnp.maximum(norm, 1e-30)), norm


def learning_rate(step, total_steps, config):
    if step < config.warmup_steps:
        return config.learning_rate * (step + 1) / max(1, config.warmup_steps)
    fraction = (step - config.warmup_steps) / max(1, total_steps - config.warmup_steps - 1)
    return config.learning_rate * (config.minimum_lr_ratio + (1 - config.minimum_lr_ratio) * .5 * (1 + jnp.cos(jnp.pi * min(1., fraction))))
