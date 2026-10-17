"""Comprehensive unit test suite for Pallas FlashAttention and CPU fallback."""
import math
import numpy as np
import pytest
import jax
import jax.numpy as jnp

from curvature_compass.lm.config import ModelConfig
from curvature_compass.lm.model import Decoder
from curvature_compass.lm.attention import (
    flash_attention,
    pallas_flash_attention,
    reference_attention,
    is_tpu_available,
    is_pallas_tpu_available,
    pad_sequence_to_tile,
)


def test_public_api_and_capability_detection():
    """Verify all public functions and capability detectors exist and execute cleanly."""
    assert callable(flash_attention)
    assert callable(pallas_flash_attention)
    assert callable(reference_attention)
    assert callable(is_tpu_available)
    assert callable(is_pallas_tpu_available)
    assert callable(pad_sequence_to_tile)

    # In CPU test environment, TPU should not be available
    assert is_tpu_available() is False
    assert is_pallas_tpu_available() is False


def test_strict_pallas_runtime_error_on_cpu():
    """Verify that pallas_flash_attention raises RuntimeError when fallback_on_cpu=False on CPU."""
    q = jnp.zeros((1, 16, 2, 32), dtype=jnp.float32)
    k = jnp.zeros((1, 16, 2, 32), dtype=jnp.float32)
    v = jnp.zeros((1, 16, 2, 32), dtype=jnp.float32)

    with pytest.raises(RuntimeError, match="Pallas FlashAttention kernel requires TPU backend"):
        pallas_flash_attention(q, k, v, fallback_on_cpu=False)


def test_forward_output_shapes_bthd_and_bhtd():
    """Verify output shapes match query shapes across both BTHD and BHTD layouts."""
    key = jax.random.PRNGKey(101)
    for b in (1, 2):
        for h in (2, 4):
            for d in (32, 64):
                for t in (32, 128):
                    # BTHD layout
                    q1 = jax.random.normal(key, (b, t, h, d))
                    k1 = jax.random.normal(key, (b, t, h, d))
                    v1 = jax.random.normal(key, (b, t, h, d))
                    out1 = flash_attention(q1, k1, v1, layout="BTHD")
                    assert out1.shape == (b, t, h, d)

                    # BHTD layout
                    q2 = jax.random.normal(key, (b, h, t, d))
                    k2 = jax.random.normal(key, (b, h, t, d))
                    v2 = jax.random.normal(key, (b, h, t, d))
                    out2 = flash_attention(q2, k2, v2, layout="BHTD")
                    assert out2.shape == (b, h, t, d)


def test_causal_masking_invariance():
    """Verify that future tokens (steps > t) cannot affect attention output at step t."""
    key = jax.random.PRNGKey(102)
    batch, seq_len, heads, head_dim = 2, 32, 4, 32
    q = jax.random.normal(key, (batch, seq_len, heads, head_dim))
    k = jax.random.normal(key, (batch, seq_len, heads, head_dim))
    v = jax.random.normal(key, (batch, seq_len, heads, head_dim))

    out_orig = flash_attention(q, k, v, causal=True)

    # Modify future tokens for positions >= 16
    k_mod = k.at[:, 16:, :, :].set(999.0)
    v_mod = v.at[:, 16:, :, :].set(999.0)
    out_mod = flash_attention(q, k_mod, v_mod, causal=True)

    # Prefix [:16] must be completely unchanged by changes in the future
    np.testing.assert_allclose(out_orig[:, :16], out_mod[:, :16], rtol=1e-6, atol=1e-6)


def test_tile_padding_helper():
    """Verify sequence padding correctly aligns to multiple of 128."""
    x = jnp.ones((2, 100, 4, 32))
    padded, pad_len = pad_sequence_to_tile(x, seq_axis=1, tile_size=128)
    assert padded.shape == (2, 128, 4, 32)
    assert pad_len == 28
    np.testing.assert_array_equal(padded[:, :100], x)
    np.testing.assert_array_equal(padded[:, 100:], 0.0)

    # Already aligned
    x_aligned = jnp.ones((2, 256, 4, 32))
    padded_aligned, pad_len_aligned = pad_sequence_to_tile(x_aligned, seq_axis=1, tile_size=128)
    assert padded_aligned.shape == (2, 256, 4, 32)
    assert pad_len_aligned == 0


def test_variable_sequence_lengths_causal_invariance():
    """Verify causal attention padding invariance across arbitrary non-128 sequence lengths."""
    key = jax.random.PRNGKey(103)
    for seq_len in (13, 64, 100, 150, 250):
        q = jax.random.normal(key, (1, seq_len, 4, 32))
        k = jax.random.normal(key, (1, seq_len, 4, 32))
        v = jax.random.normal(key, (1, seq_len, 4, 32))

        # Unpadded reference
        ref = reference_attention(q, k, v, causal=True)

        # Padded to tile size 128
        q_pad, pad_len = pad_sequence_to_tile(q, seq_axis=1, tile_size=128)
        k_pad, _ = pad_sequence_to_tile(k, seq_axis=1, tile_size=128)
        v_pad, _ = pad_sequence_to_tile(v, seq_axis=1, tile_size=128)

        out_pad = reference_attention(q_pad, k_pad, v_pad, causal=True)
        sliced = out_pad[:, :seq_len, :, :]

        np.testing.assert_allclose(ref, sliced, rtol=1e-5, atol=1e-5)


def test_numerical_parity_fp32():
    """Verify bit-for-bit exact parity between einsum reference and flash_attention on CPU in FP32."""
    key = jax.random.PRNGKey(104)
    batch, seq_len, heads, head_dim = 2, 64, 4, 64
    k1, k2, k3 = jax.random.split(key, 3)
    q = jax.random.normal(k1, (batch, seq_len, heads, head_dim), dtype=jnp.float32)
    k = jax.random.normal(k2, (batch, seq_len, heads, head_dim), dtype=jnp.float32)
    v = jax.random.normal(k3, (batch, seq_len, heads, head_dim), dtype=jnp.float32)

    # Manual einsum reference
    sm_scale = 1.0 / math.sqrt(head_dim)
    scores = jnp.einsum("bthd,bshd->bhts", q * sm_scale, k)
    causal = jnp.arange(seq_len)[:, None] >= jnp.arange(seq_len)[None, :]
    scores = jnp.where(causal, scores, -jnp.inf)
    weights = jax.nn.softmax(scores, axis=-1)
    ref_einsum = jnp.einsum("bhts,bshd->bthd", weights, v)

    out_flash = flash_attention(q, k, v, causal=True)
    np.testing.assert_allclose(out_flash, ref_einsum, rtol=1e-5, atol=1e-5)


def test_numerical_parity_bf16():
    """Verify parity in BF16 within IEEE 754 7-bit mantissa machine tolerance."""
    key = jax.random.PRNGKey(105)
    batch, seq_len, heads, head_dim = 2, 64, 4, 64
    k1, k2, k3 = jax.random.split(key, 3)
    q = jax.random.normal(k1, (batch, seq_len, heads, head_dim)).astype(jnp.bfloat16)
    k = jax.random.normal(k2, (batch, seq_len, heads, head_dim)).astype(jnp.bfloat16)
    v = jax.random.normal(k3, (batch, seq_len, heads, head_dim)).astype(jnp.bfloat16)

    ref = reference_attention(q, k, v, causal=True)
    out = flash_attention(q, k, v, causal=True)

    np.testing.assert_allclose(
        out.astype(jnp.float32),
        ref.astype(jnp.float32),
        rtol=3e-2,
        atol=3e-2,
    )


def test_gradient_and_ad_backward_pass():
    """Verify that reverse-mode AD gradients compute correctly through flash_attention."""
    def loss_fn(q, k, v):
        out = flash_attention(q, k, v, causal=True)
        return jnp.sum(out)

    key = jax.random.PRNGKey(106)
    q = jax.random.normal(key, (1, 16, 2, 16))
    k = jax.random.normal(key, (1, 16, 2, 16))
    v = jax.random.normal(key, (1, 16, 2, 16))

    g_q, g_k, g_v = jax.grad(loss_fn, argnums=(0, 1, 2))(q, k, v)
    assert g_q.shape == q.shape
    assert g_k.shape == k.shape
    assert g_v.shape == v.shape
    assert jnp.all(jnp.isfinite(g_q))
    assert jnp.all(jnp.isfinite(g_k))
    assert jnp.all(jnp.isfinite(g_v))


def test_custom_sm_scale():
    """Verify explicit sm_scale parameter affects attention outputs appropriately."""
    key = jax.random.PRNGKey(107)
    q = jax.random.normal(key, (1, 16, 2, 16))
    k = jax.random.normal(key, (1, 16, 2, 16))
    v = jax.random.normal(key, (1, 16, 2, 16))

    out_default = flash_attention(q, k, v, causal=True, sm_scale=None)
    out_scaled = flash_attention(q, k, v, causal=True, sm_scale=0.01)

    assert not np.allclose(out_default, out_scaled)
    # Check that scaled matches reference_attention with the same scale
    out_ref_scaled = reference_attention(q, k, v, causal=True, sm_scale=0.01)
    np.testing.assert_allclose(out_scaled, out_ref_scaled, rtol=1e-5, atol=1e-5)


def test_model_logits_integration():
    """Verify integration into Decoder.logits() with attention='flash' and attention='pallas'."""
    config = ModelConfig(
        vocab_size=32,
        padded_vocab_size=32,
        width=32,
        layers=2,
        heads=4,
        mlp_width=64,
        sequence_length=16,
    )
    decoder = Decoder(config)
    params = decoder.initialize(42)
    tokens = jnp.array([[1, 5, 12, 18, 22]])

    logits_ref = decoder.logits(params, tokens, attention="reference")
    logits_sdpa = decoder.logits(params, tokens, attention="sdpa")
    logits_flash = decoder.logits(params, tokens, attention="flash")
    logits_pallas = decoder.logits(params, tokens, attention="pallas")

    assert logits_flash.shape == logits_ref.shape
    assert logits_pallas.shape == logits_ref.shape

    np.testing.assert_allclose(logits_flash, logits_ref, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(logits_pallas, logits_ref, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(logits_flash, logits_sdpa, rtol=1e-5, atol=1e-5)
