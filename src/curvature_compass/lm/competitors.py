"""Standard JAX/XLA AdamW, Muon, SOAP, and Sophia-G.

Algorithm references and deliberate implementation choices are registered in
analysis/phase34_protocol.md. Parameters and all optimizer algebra are float32.
Muon acts on hidden matrices in (output, input) orientation; its auxiliary
parameters use AdamW. SOAP skips its initialization batch, maintains Adam
moments in Shampoo coordinates, and uses the reference permutation/QR refresh.
"""
from dataclasses import dataclass
from typing import NamedTuple
import math
import jax
import jax.numpy as jnp

PRECISION = jax.lax.Precision.HIGHEST
BASELINES = ("adamw", "muon", "soap", "sophia_g")


@dataclass(frozen=True)
class BaselineConfig:
    name: str = "adamw"
    beta1: float = .9
    beta2: float = .95
    epsilon: float = 1e-8
    momentum: float = .95
    nesterov: bool = True
    ns_steps: int = 5
    auxiliary_lr_ratio: float = .015
    shampoo_beta: float = .95
    precondition_frequency: int = 10
    max_precondition_dimension: int = 2048
    rho: float = .04
    hessian_interval: int = 10

    def validate(self):
        if self.name not in BASELINES:
            raise ValueError("Unknown baseline optimizer")
        if any(not math.isfinite(v) for v in (self.beta1, self.beta2, self.epsilon, self.momentum,
                                             self.auxiliary_lr_ratio, self.shampoo_beta, self.rho)):
            raise ValueError("Optimizer controls must be finite")
        if not all(0 <= v < 1 for v in (self.beta1, self.beta2, self.momentum, self.shampoo_beta)):
            raise ValueError("EMA coefficients must lie in [0, 1)")
        if min(self.epsilon, self.auxiliary_lr_ratio, self.rho) <= 0:
            raise ValueError("Optimizer safeguards/rates must be positive")
        if min(self.ns_steps, self.precondition_frequency, self.max_precondition_dimension, self.hessian_interval) < 1:
            raise ValueError("Optimizer iteration counts must be positive")
        if not isinstance(self.nesterov, bool):
            raise ValueError("Nesterov must be a boolean")


class TensorState(NamedTuple):
    momentum: jax.Array
    variance: jax.Array
    left: jax.Array
    right: jax.Array
    left_basis: jax.Array
    right_basis: jax.Array


class BaselineState(NamedTuple):
    count: jax.Array
    tensors: tuple
    hessian_step: jax.Array


def hidden_matrix(name, shape):
    return name.startswith("block.") and name.endswith(".weight") and len(shape) == 2


def empty_baseline_state(decoder, config):
    tensors = []
    for name, shape in decoder.layout:
        use_muon = config.name == "muon" and hidden_matrix(name, shape)
        sizes = shape if config.name == "soap" and len(shape) == 2 else (0, 0)
        left, right = [s if s <= config.max_precondition_dimension else 0 for s in sizes]
        tensors.append(TensorState(
            jnp.zeros(shape, jnp.float32), jnp.zeros((0,) if use_muon else shape, jnp.float32),
            jnp.zeros((left, left), jnp.float32), jnp.zeros((right, right), jnp.float32),
            jnp.eye(left, dtype=jnp.float32), jnp.eye(right, dtype=jnp.float32)))
    return BaselineState(jnp.array(0, jnp.int32), tuple(tensors), jnp.array(-1, jnp.int32))


def matmul(a, b):
    return jnp.matmul(a, b, precision=PRECISION)


def newton_schulz(matrix, steps=5):
    """Muon quintic approximation; no claim of an exact polar decomposition."""
    transposed = matrix.shape[0] > matrix.shape[1]
    x = matrix.T if transposed else matrix
    x = x / (jnp.linalg.norm(x) + 1e-7)

    def body(_, x):
        gram = matmul(x, x.T)
        polynomial = -4.7750 * gram + 2.0315 * matmul(gram, gram)
        return 3.4445 * x + matmul(polynomial, x)

    x = jax.lax.fori_loop(0, steps, body, x)
    return x.T if transposed else x


def adam_direction(gradient, state, count, config):
    m = config.beta1 * state.momentum + (1 - config.beta1) * gradient
    v = config.beta2 * state.variance + (1 - config.beta2) * gradient ** 2
    result = (m / (1 - config.beta1 ** count)) / (jnp.sqrt(v / (1 - config.beta2 ** count)) + config.epsilon)
    return result, state._replace(momentum=m, variance=v)


def project(value, left, right, back=False):
    if value.ndim != 2:
        return value
    if left.shape[0]:
        value = matmul(left if back else left.T, value)
    if right.shape[0]:
        value = matmul(value, right.T if back else right)
    return value


def soap_covariances(gradient, state, config):
    left, right = state.left, state.right
    if left.shape[0]:
        left = config.shampoo_beta * left + (1 - config.shampoo_beta) * matmul(gradient, gradient.T)
    if right.shape[0]:
        right = config.shampoo_beta * right + (1 - config.shampoo_beta) * matmul(gradient.T, gradient)
    return left, right


def soap_initial(gradient, state, config):
    left, right = soap_covariances(gradient, state, config)

    def basis(covariance):
        if not covariance.shape[0]:
            return covariance
        return jnp.linalg.eigh(covariance + 1e-30 * jnp.eye(covariance.shape[0]))[1][:, ::-1]

    return jnp.zeros_like(gradient), state._replace(left=left, right=right,
                                                   left_basis=basis(left), right_basis=basis(right))


def soap_direction(gradient, state, count, config):
    g = project(gradient, state.left_basis, state.right_basis)
    m = config.beta1 * state.momentum + (1 - config.beta1) * g
    v = config.beta2 * state.variance + (1 - config.beta2) * g ** 2
    correction = jnp.sqrt(1 - config.beta2 ** count) / (1 - config.beta1 ** count)
    delta = correction * project(m / (jnp.sqrt(v) + config.epsilon), state.left_basis, state.right_basis, back=True)
    left, right = soap_covariances(gradient, state, config)

    def refresh(operand):
        m, v = operand
        physical_m = project(m, state.left_basis, state.right_basis, back=True)
        bases = []
        for axis, (covariance, old_basis) in enumerate(((left, state.left_basis), (right, state.right_basis))):
            if not covariance.shape[0]:
                bases.append(old_basis)
                continue
            order = jnp.argsort(jnp.sum(old_basis * matmul(covariance, old_basis), axis=0))[::-1]
            v = jnp.take(v, order, axis=axis)
            q, _ = jnp.linalg.qr(matmul(covariance, old_basis[:, order]))
            bases.append(q)
        return project(physical_m, *bases), v, *bases

    m, v, ql, qr = jax.lax.cond(count % config.precondition_frequency == 0, refresh,
                                lambda operand: (*operand, state.left_basis, state.right_basis), (m, v))
    return delta, TensorState(m, v, left, right, ql, qr)


def make_baseline_update(decoder, run_config):
    config = run_config.optimizer

    def update(parameters, gradient, state, rate):
        count = state.count + 1
        next_parameters, tensors, deltas = [], [], []
        saturated, elements = jnp.array(0, jnp.int32), 0
        for (name, shape), (start, stop), tensor in zip(decoder.layout, decoder.offsets, state.tensors):
            p, g = parameters[start:stop].reshape(shape), gradient[start:stop].reshape(shape)
            tensor_rate = rate
            if config.name == "muon" and hidden_matrix(name, shape):
                m = config.momentum * tensor.momentum + (1 - config.momentum) * g
                candidate = (1 - config.momentum) * g + config.momentum * m if config.nesterov else m
                # Decoder weights are (input, output); Muon's aspect scaling uses (output, input).
                delta = newton_schulz(candidate.T, config.ns_steps).T * math.sqrt(max(1, shape[1] / shape[0]))
                tensor = tensor._replace(momentum=m)
            elif config.name == "soap":
                delta, tensor = jax.lax.cond(state.count == 0,
                    lambda _: soap_initial(g, tensor, config),
                    lambda _: soap_direction(g, tensor, state.count, config), operand=None)
            elif config.name == "sophia_g":
                m = config.beta1 * tensor.momentum + (1 - config.beta1) * g
                ratio = m / (config.rho * tensor.variance + config.epsilon)
                delta = jnp.clip(ratio, -1, 1)
                saturated += jnp.sum(jnp.abs(ratio) >= 1)
                elements += math.prod(shape)
                tensor = tensor._replace(momentum=m)
            else:
                delta, tensor = adam_direction(g, tensor, count, config)
                if config.name == "muon":
                    tensor_rate *= config.auxiliary_lr_ratio
            decay = run_config.weight_decay if len(shape) >= 2 else 0.
            if config.name == "soap":
                # Reference SOAP decays after its update and skips the entire first batch.
                result = jnp.where(state.count == 0, p, (p - tensor_rate * delta) * (1 - tensor_rate * decay))
            else:
                result = p * (1 - tensor_rate * decay) - tensor_rate * delta
            next_parameters.append(result.reshape(-1))
            deltas.append(delta.reshape(-1))
            tensors.append(tensor)
        parameters = jnp.concatenate(next_parameters)
        next_state = BaselineState(count, tuple(tensors), state.hessian_step)
        finite = jnp.all(jnp.isfinite(parameters))
        for leaf in jax.tree.leaves(next_state):
            finite &= jnp.all(jnp.isfinite(leaf))
        stats = {"parameters_finite": finite, "optimizer_state_finite": finite,
                 "direction_norm": jnp.linalg.norm(jnp.concatenate(deltas)),
                 "sophia_clipped_fraction": saturated / max(elements, 1),
                 "soap_initialization_batch": (state.count == 0) & (config.name == "soap"),
                 "preconditioner_refreshed": (config.name == "soap") & ((state.count == 0) | (state.count % config.precondition_frequency == 0))}
        return parameters, next_state, stats

    return update


def bartlett_diagonal(sampled_mean_gradient, predicted_tokens):
    """N*g_sample**2 estimates the diagonal GGN of the mean token loss.

    Labels must be independently drawn from stop-gradient model probabilities.
    Squaring the globally averaged sampled-label gradient and scaling by the
    actual global token count matches the reference Sophia-G normalization.
    It is not the squared supervised gradient (an empirical-Fisher substitute).
    """
    return predicted_tokens * sampled_mean_gradient ** 2


def make_sophia_refresh(decoder, run_config):
    config = run_config.optimizer

    def refresh(parameters, batch, state, step):
        x, _, mask = batch
        key = jax.random.fold_in(jax.random.PRNGKey(run_config.seed ^ 0x534F5048), step)
        logits = decoder.logits(parameters, x, compute_dtype="float32", attention="reference", rematerialize=False)
        # Sampling is global and stateless: seed+step also reproduces it after recovery.
        labels = jax.random.categorical(key, jax.lax.stop_gradient(logits), axis=-1).astype(jnp.int32)
        sampled_gradient = jax.grad(lambda p: decoder.loss(p, (x, labels, mask),
                                   compute_dtype="float32", attention="reference", rematerialize=False))(parameters)
        diagonal = bartlett_diagonal(sampled_gradient, jnp.sum(mask))
        tensors = []
        for tensor, shape, (start, stop) in zip(state.tensors, (s for _, s in decoder.layout), decoder.offsets):
            h = config.beta2 * tensor.variance + (1 - config.beta2) * diagonal[start:stop].reshape(shape)
            tensors.append(tensor._replace(variance=h))
        state = BaselineState(state.count, tuple(tensors), step)
        return state, {"finite": jnp.all(jnp.isfinite(diagonal)), "hvp_count": jnp.array(0, jnp.int32),
                       "sampled_label_gradient_count": jnp.array(1, jnp.int32),
                       "ggn_diagonal_mean": jnp.mean(diagonal), "ggn_predicted_tokens": jnp.sum(mask)}

    return refresh
