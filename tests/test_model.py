import math
from dataclasses import replace
import jax
import jax.numpy as jnp
import numpy as np
from curvature_compass.lm.config import ModelConfig
from curvature_compass.lm.model import Decoder, apply_rope, precompute_rope_freqs, rotate_half


def small_model():
    return Decoder(ModelConfig(vocab_size=23, padded_vocab_size=32, width=16, layers=2,
                               heads=4, mlp_width=32, sequence_length=8))


def test_parameter_count_initialization_tying_and_padding():
    assert Decoder(ModelConfig()).parameter_count == 124_275_456
    model = small_model()
    p = model.initialize(1)
    assert np.array_equal(p, model.initialize(1))
    assert not np.array_equal(p, model.initialize(2))
    assert not any(name == "output_embedding" for name, _ in model.layout)
    assert not any("bias" in name for name, _ in model.layout)
    assert not any("position_embedding" in name for name, _ in model.layout)
    assert any("input_norm.weight" in name for name, _ in model.layout)
    assert any("post_attention_norm.weight" in name for name, _ in model.layout)
    assert any("final_norm.weight" in name for name, _ in model.layout)
    assert any("gate_proj.weight" in name for name, _ in model.layout)
    assert any("up_proj.weight" in name for name, _ in model.layout)
    assert any("down_proj.weight" in name for name, _ in model.layout)
    logits = model.logits(jnp.asarray(p), jnp.array([[1, 2, 3]]))
    assert np.all(np.asarray(logits)[..., 23:] < -1e8)


def test_rmsnorm_numerical_stability_and_zero_bias():
    # Verify RMSNorm scales without mean subtraction and without learnable bias
    x = jnp.array([[[1.0, 2.0, 3.0, 4.0]]], dtype=jnp.float32)
    weight = jnp.ones((4,), dtype=jnp.float32)
    variance = jnp.mean(jnp.square(x), axis=-1, keepdims=True)
    expected = x * jax.lax.rsqrt(variance + 1e-5) * weight
    # Check that centering does not occur (mean of x is 2.5, non-zero)
    assert not jnp.allclose(jnp.mean(expected, axis=-1), 0.0)
    # Check zero input stability
    zero_x = jnp.zeros((1, 1, 4), dtype=jnp.float32)
    zero_out = zero_x * jax.lax.rsqrt(jnp.mean(jnp.square(zero_x), axis=-1, keepdims=True) + 1e-5) * weight
    np.testing.assert_allclose(zero_out, 0.0)


def test_rope_equivariance_and_isometry():
    head_dim = 16
    seq_len = 32
    cos, sin = precompute_rope_freqs(head_dim, seq_len)
    
    # Isometry: RoPE preserves vector L2 norm
    q = jax.random.normal(jax.random.key(42), (1, seq_len, 1, head_dim))
    q_rot = apply_rope(q, cos, sin)
    orig_norm = jnp.linalg.norm(q, axis=-1)
    rot_norm = jnp.linalg.norm(q_rot, axis=-1)
    np.testing.assert_allclose(orig_norm, rot_norm, rtol=1e-5, atol=1e-5)
    
    # Relative position equivariance: <q_m, k_n> depends only on (m - n)
    k = jax.random.normal(jax.random.key(43), (1, seq_len, 1, head_dim))
    k_rot = apply_rope(k, cos, sin)
    # Compare inner product at (pos_m=5, pos_n=2) vs (pos_m=15, pos_n=12) with shifted inputs
    # If q is placed at 5 and k at 2, relative distance is 3
    # Same q vector at 15 and k vector at 12 should give identical dot product
    q_single = jax.random.normal(jax.random.key(44), (head_dim,))
    k_single = jax.random.normal(jax.random.key(45), (head_dim,))
    
    q_stream = jnp.zeros((1, seq_len, 1, head_dim)).at[0, 5, 0].set(q_single).at[0, 15, 0].set(q_single)
    k_stream = jnp.zeros((1, seq_len, 1, head_dim)).at[0, 2, 0].set(k_single).at[0, 12, 0].set(k_single)
    
    qr = apply_rope(q_stream, cos, sin)
    kr = apply_rope(k_stream, cos, sin)
    
    dot1 = jnp.sum(qr[0, 5, 0] * kr[0, 2, 0])
    dot2 = jnp.sum(qr[0, 15, 0] * kr[0, 12, 0])
    np.testing.assert_allclose(dot1, dot2, rtol=1e-5, atol=1e-5)


def test_swiglu_activations():
    x = jnp.array([[0.5, -1.0, 2.0]], dtype=jnp.float32)
    gate_w = jnp.ones((3, 4), dtype=jnp.float32)
    up_w = jnp.ones((3, 4), dtype=jnp.float32) * 2.0
    gate = x @ gate_w
    up = x @ up_w
    expected = jax.nn.silu(gate) * up
    assert expected.shape == (1, 4)
    # When x @ gate_w is large and negative, silu smoothly saturates to 0
    large_neg = jnp.array([[-100.0, -100.0, -100.0]], dtype=jnp.float32)
    neg_out = jax.nn.silu(large_neg @ gate_w) * (large_neg @ up_w)
    np.testing.assert_allclose(neg_out, 0.0, atol=1e-5)


def test_causal_attention_and_shifted_loss():
    model = small_model()
    p = jnp.asarray(model.initialize(3))
    first = jnp.array([[1, 2, 3, 4, 5]])
    second = first.at[0, 3:].set(jnp.array([7, 8]))
    a, b = model.logits(p, first), model.logits(p, second)
    np.testing.assert_allclose(a[:, :3], b[:, :3], atol=1e-6)
    targets = jnp.array([[2, 3, 4, 5, 6]])
    mask = jnp.array([[1., 1., 0., 0., 0.]])
    scores = np.asarray(a)[0, :2, :23]
    normalizers = np.log(np.exp(scores - scores.max(axis=-1, keepdims=True)).sum(axis=-1)) + scores.max(axis=-1)
    expected = float(np.mean(normalizers - scores[np.arange(2), [2, 3]]))
    np.testing.assert_allclose(model.loss(p, (first, targets, mask)), expected, rtol=1e-6)


def test_hvp_finite_difference_and_linearity():
    model = small_model()
    p = jnp.asarray(model.initialize(4))
    x = jnp.array([[1, 2, 3, 4], [2, 3, 4, 5]])
    batch = (x, x + 1, jnp.ones(x.shape))
    grad = jax.jit(jax.grad(lambda z: model.loss(z, batch)))
    rng = np.random.default_rng(5)
    v = jnp.asarray(rng.normal(size=len(p)).astype(np.float32))
    v /= jnp.linalg.norm(v)
    w = jnp.asarray(rng.normal(size=len(p)).astype(np.float32))
    w /= jnp.linalg.norm(w)
    hvp = jax.jit(lambda z: jax.jvp(grad, (p,), (z,))[1])
    epsilon = .002
    finite_difference = (grad(p + epsilon * v) - grad(p - epsilon * v)) / (2 * epsilon)
    actual = hvp(v)
    relative_error = np.linalg.norm(actual - finite_difference) / np.linalg.norm(actual)
    assert relative_error < .02, relative_error
    np.testing.assert_allclose(hvp(v + w), hvp(v) + hvp(w), rtol=1e-4, atol=1e-5)


def test_sdpa_rematerialization_and_accumulation_preserve_derivatives():
    model = small_model()
    p = jnp.asarray(model.initialize(6))
    x = jnp.array([[1, 2, 3, 4], [2, 3, 4, 5], [3, 4, 5, 6], [4, 5, 6, 7]])
    mask = jnp.ones(x.shape).at[-1, -2:].set(0)
    batch = (x, x + 1, mask)
    reference = lambda z: model.training_loss(z, batch)
    loss, grad = jax.value_and_grad(reference)(p)
    v = jnp.asarray(np.random.default_rng(8).normal(size=len(p)).astype(np.float32))
    v /= jnp.linalg.norm(v)
    hvp = jax.jvp(jax.grad(reference), (p,), (v,))[1]
    for options in ({"attention": "sdpa"}, {"rematerialize": True}, {"accumulation_steps": 2}):
        candidate = lambda z: model.training_loss(z, batch, **options)
        candidate_loss, candidate_grad = jax.value_and_grad(candidate)(p)
        candidate_hvp = jax.jvp(jax.grad(candidate), (p,), (v,))[1]
        np.testing.assert_allclose(candidate_loss, loss, rtol=1e-5, atol=1e-6)
        assert np.linalg.norm(candidate_grad - grad) / np.linalg.norm(grad) < 1e-4
        assert np.linalg.norm(candidate_hvp - hvp) / np.linalg.norm(hvp) < 1e-4


def test_linearization_reuse_matches_recomputed_hvps():
    model = small_model()
    p = jnp.asarray(model.initialize(7))
    x = jnp.array([[1, 2, 3, 4]])
    grad = jax.grad(lambda z: model.loss(z, (x, x + 1, jnp.ones(x.shape))))
    _, reused = jax.linearize(grad, p)
    for seed in (10, 11):
        v = jnp.asarray(np.random.default_rng(seed).normal(size=len(p)).astype(np.float32))
        np.testing.assert_allclose(reused(v), jax.jvp(grad, (p,), (v,))[1], rtol=1e-5, atol=1e-5)
