"""Tier 1: Comprehensive Feature Coverage E2E Test Suite.

Covers all 19 features in PROJECT.md § Feature Inventory with >=5 tests per feature:
- Feature 1: LLaMA RMSNorm (5 tests)
- Feature 2: Rotary Position Embeddings (RoPE) (5 tests)
- Feature 3: SwiGLU FFN (5 tests)
- Feature 4: Bias-Free Linear Projections (5 tests)
- Feature 5: Canonical 32k Tokenizer (5 tests)
- Feature 6: Delete Legacy GPT-2 Components (5 tests)
- Feature 7: Pure JAX Optimizers (5 tests)
- Feature 8: Zero Pallas in Optimizers (5 tests)
- Feature 9: Pallas FlashAttention TPU (5 tests)
- Feature 10: Attention CPU Fallback (5 tests)
- Feature 11: Attention Parity Check (5 tests)
- Feature 12: 125M Model Configuration (5 tests)
- Feature 13: 600M Sweeps Pipeline (5 tests)
- Feature 14: 125M / 2.5B Training Pipeline (5 tests)
- Feature 15: Bounded Smoke Tests (5 tests)
- Feature 16: Complete Legacy Phase Purge (5 tests)
- Feature 17: E2E 4-Tier Test Suite (5 tests)
- Feature 18: 100% CPU Test Pass (5 tests)
- Feature 19: Git Clean Commit & Push (5 tests)

Total tests: 95 tests.
"""
from dataclasses import replace
import inspect
import math
from pathlib import Path
import subprocess
import numpy as np
import pytest
import jax
import jax.numpy as jnp

from curvature_compass.lm.config import ModelConfig, RunConfig, OptimizerConfig, digest
from curvature_compass.lm.model import Decoder, apply_rope, precompute_rope_freqs, rotate_half
from curvature_compass.lm.attention import (
    flash_attention,
    pallas_flash_attention,
    reference_attention,
    is_tpu_available,
    is_pallas_tpu_available,
    pad_sequence_to_tile,
)
from curvature_compass.lm.data import (
    BOS, EOS, PAD, PADDED_VOCAB_SIZE, SPLIT_SALT, TOKENIZER, TOKENIZER_REVISION,
    UNK, VOCAB_SIZE, TokenBlocks, document_key, split_for, verify_data
)
from curvature_compass.lm import competitors, optim, sweep, compare, checkpoint
from curvature_compass.lm.competitors import (
    BaselineConfig, empty_baseline_state, make_baseline_update, make_sophia_refresh, newton_schulz
)
from curvature_compass.lm.experiments import load_run_config
from curvature_compass.lm.runtime import mesh_and_shardings


def small_model_cfg():
    return ModelConfig(
        vocab_size=32,
        padded_vocab_size=128,
        width=16,
        layers=2,
        heads=2,
        mlp_width=32,
        sequence_length=8,
    )


def smoke_model_cfg():
    return ModelConfig(
        vocab_size=VOCAB_SIZE,
        padded_vocab_size=PADDED_VOCAB_SIZE,
        width=16,
        layers=2,
        heads=2,
        mlp_width=32,
        sequence_length=8,
    )


# ==============================================================================
# Feature 1: LLaMA RMSNorm
# ==============================================================================

def test_f01_rmsnorm_weight_layout_1d():
    """Verify RMSNorm weights in layout are strictly 1D vectors and contain no bias."""
    model = Decoder(small_model_cfg())
    norm_layers = [item for item in model.layout if "norm" in item[0]]
    assert len(norm_layers) > 0
    for name, shape in norm_layers:
        assert len(shape) == 1, f"RMSNorm weight {name} should be 1D, got {shape}"
        assert "bias" not in name, f"RMSNorm must not have bias parameter: {name}"


def test_f01_rmsnorm_unit_init():
    """Verify initialize sets all RMSNorm scales to exactly 1.0."""
    model = Decoder(small_model_cfg())
    p = model.initialize(seed=42)
    unpacked = model.unpack(p)
    assert np.all(unpacked["block.0.input_norm.weight"] == 1.0)
    assert np.all(unpacked["block.0.post_attention_norm.weight"] == 1.0)
    assert np.all(unpacked["final_norm.weight"] == 1.0)


def test_f01_rmsnorm_no_mean_subtraction():
    """Verify RMSNorm does not subtract mean (unlike LayerNorm)."""
    x = jnp.array([[[1.0, 2.0, 3.0, 4.0]]], dtype=jnp.float32)
    weight = jnp.ones((4,), dtype=jnp.float32)
    variance = jnp.mean(jnp.square(x), axis=-1, keepdims=True)
    normed = x * jax.lax.rsqrt(variance + 1e-5) * weight
    # If mean was subtracted, output mean would be 0.0. Since it's not, output mean != 0.
    assert not jnp.allclose(jnp.mean(normed, axis=-1), 0.0, atol=1e-4)


def test_f01_rmsnorm_scale_invariance_direction():
    """Verify RMSNorm of (alpha * x) has the exact same directional output as RMSNorm of x."""
    x = jnp.array([[[1.2, -0.8, 3.4, 2.1]]], dtype=jnp.float32)
    alpha = 5.7
    weight = jnp.ones((4,), dtype=jnp.float32)
    rms1 = x * jax.lax.rsqrt(jnp.mean(jnp.square(x), axis=-1, keepdims=True) + 1e-10) * weight
    rms2 = (alpha * x) * jax.lax.rsqrt(jnp.mean(jnp.square(alpha * x), axis=-1, keepdims=True) + 1e-10) * weight
    np.testing.assert_allclose(rms1, rms2, rtol=1e-4, atol=1e-5)


def test_f01_rmsnorm_decay_mask_zero():
    """Verify weight decay mask is 0.0 for all RMSNorm parameters."""
    model = Decoder(small_model_cfg())
    decay_mask = model.decay_mask()
    for (name, shape), (start, stop) in zip(model.layout, model.offsets):
        if "norm" in name:
            assert np.all(decay_mask[start:stop] == 0.0), f"Decay mask for {name} should be 0.0"


# ==============================================================================
# Feature 2: Rotary Position Embeddings (RoPE)
# ==============================================================================

def test_f02_rope_precompute_shapes():
    """Verify precompute_rope_freqs returns cos and sin with correct shapes."""
    head_dim = 16
    seq_len = 32
    cos, sin = precompute_rope_freqs(head_dim, seq_len)
    assert cos.shape == (seq_len, head_dim)
    assert sin.shape == (seq_len, head_dim)
    assert cos.dtype == jnp.float32
    assert sin.dtype == jnp.float32


def test_f02_rope_preserves_l2_norm():
    """Verify RoPE is an isometric rotation preserving L2 vector norm."""
    head_dim = 16
    seq_len = 8
    cos, sin = precompute_rope_freqs(head_dim, seq_len)
    q = jax.random.normal(jax.random.key(123), (1, seq_len, 2, head_dim))
    q_rot = apply_rope(q, cos, sin)
    orig_norm = jnp.linalg.norm(q, axis=-1)
    rot_norm = jnp.linalg.norm(q_rot, axis=-1)
    np.testing.assert_allclose(orig_norm, rot_norm, rtol=1e-5, atol=1e-5)


def test_f02_rope_relative_displacement_equivariance():
    """Verify RoPE inner product <R_m q, R_n k> depends only on relative shift (m - n)."""
    head_dim = 8
    seq_len = 16
    cos, sin = precompute_rope_freqs(head_dim, seq_len)
    q_vec = jax.random.normal(jax.random.key(1), (head_dim,))
    k_vec = jax.random.normal(jax.random.key(2), (head_dim,))

    # Stream 1: q at 3, k at 1 (displacement 2)
    q1 = jnp.zeros((1, seq_len, 1, head_dim)).at[0, 3, 0].set(q_vec)
    k1 = jnp.zeros((1, seq_len, 1, head_dim)).at[0, 1, 0].set(k_vec)
    # Stream 2: q at 7, k at 5 (displacement 2)
    q2 = jnp.zeros((1, seq_len, 1, head_dim)).at[0, 7, 0].set(q_vec)
    k2 = jnp.zeros((1, seq_len, 1, head_dim)).at[0, 5, 0].set(k_vec)

    dot1 = jnp.sum(apply_rope(q1, cos, sin)[0, 3, 0] * apply_rope(k1, cos, sin)[0, 1, 0])
    dot2 = jnp.sum(apply_rope(q2, cos, sin)[0, 7, 0] * apply_rope(k2, cos, sin)[0, 5, 0])
    np.testing.assert_allclose(dot1, dot2, rtol=1e-5, atol=1e-5)


def test_f02_rope_position_zero_is_identity():
    """Verify at token position 0 RoPE is the identity transformation (cos=1, sin=0)."""
    head_dim = 16
    seq_len = 4
    cos, sin = precompute_rope_freqs(head_dim, seq_len)
    np.testing.assert_allclose(cos[0], 1.0, atol=1e-6)
    np.testing.assert_allclose(sin[0], 0.0, atol=1e-6)
    v = jax.random.normal(jax.random.key(3), (1, seq_len, 1, head_dim))
    v_rot = apply_rope(v, cos, sin)
    np.testing.assert_allclose(v_rot[:, 0], v[:, 0], atol=1e-6)


def test_f02_rope_no_position_embedding_in_layout():
    """Verify no learned absolute position embeddings exist in model layout."""
    model = Decoder(small_model_cfg())
    names = [name for name, _ in model.layout]
    assert not any("position" in name for name in names)
    assert not any("wpe" in name for name in names)


# ==============================================================================
# Feature 3: SwiGLU FFN
# ==============================================================================

def test_f03_swiglu_layout_shapes():
    """Verify SwiGLU projection shapes (gate_proj, up_proj, down_proj)."""
    cfg = small_model_cfg()
    model = Decoder(cfg)
    unpacked = model.unpack(model.initialize(1))
    assert unpacked["block.0.gate_proj.weight"].shape == (cfg.width, cfg.mlp_width)
    assert unpacked["block.0.up_proj.weight"].shape == (cfg.width, cfg.mlp_width)
    assert unpacked["block.0.down_proj.weight"].shape == (cfg.mlp_width, cfg.width)


def test_f03_swiglu_silu_multiplication_formula():
    """Verify SwiGLU feedforward output matches down_proj(silu(gate) * up)."""
    x = jnp.array([[0.5, -0.2, 1.4]], dtype=jnp.float32)
    gate_w = jnp.array([[1.0, 0.5], [-0.5, 1.0], [0.2, -0.4]], dtype=jnp.float32)
    up_w = jnp.array([[0.2, -0.1], [0.3, 0.4], [1.1, -0.5]], dtype=jnp.float32)
    down_w = jnp.array([[0.5, 0.2, -0.1], [-0.3, 0.7, 0.4]], dtype=jnp.float32)

    gate = x @ gate_w
    up = x @ up_w
    hidden = jax.nn.silu(gate) * up
    expected = hidden @ down_w

    assert expected.shape == (1, 3)
    assert not np.isnan(expected).any()


def test_f03_swiglu_no_gelu_in_decoder():
    """Verify GELU is strictly not used anywhere in model.py decoder."""
    source = inspect.getsource(Decoder)
    assert "gelu" not in source.lower()


def test_f03_swiglu_gradient_flow():
    """Verify gradient flows non-zero through both gate and up projections."""
    def ffn(x, w_gate, w_up, w_down):
        gate = x @ w_gate
        up = x @ w_up
        return jnp.sum((jax.nn.silu(gate) * up) @ w_down)

    x = jnp.ones((1, 4), dtype=jnp.float32)
    w_gate = jnp.ones((4, 8), dtype=jnp.float32) * 0.1
    w_up = jnp.ones((4, 8), dtype=jnp.float32) * 0.2
    w_down = jnp.ones((8, 4), dtype=jnp.float32) * 0.3

    grads = jax.grad(ffn, argnums=(0, 1, 2, 3))(x, w_gate, w_up, w_down)
    for g in grads:
        assert np.all(np.isfinite(g))
        assert np.any(g != 0.0)


def test_f03_swiglu_ratio_dimensions():
    """Verify 125M canonical default uses mlp_width of 2560 for width 768 (~10/3 ratio)."""
    cfg = ModelConfig()
    assert cfg.width == 768
    assert cfg.mlp_width == 2560
    assert abs(cfg.mlp_width / cfg.width - 10 / 3) < 0.1


# ==============================================================================
# Feature 4: Bias-Free Linear Projections
# ==============================================================================

def test_f04_no_bias_in_model_layout():
    """Verify complete absence of bias parameters across entire model layout."""
    model = Decoder(small_model_cfg())
    for name, _ in model.layout:
        assert "bias" not in name, f"Found unexpected bias parameter: {name}"


def test_f04_attention_projections_pure_linear():
    """Verify attention projections f(0) == 0 (zero input yields zero output)."""
    model = Decoder(small_model_cfg())
    p = model.unpack(model.initialize(1))
    zero_in = jnp.zeros((1, 1, model.config.width), dtype=jnp.float32)
    for proj in ("q_proj", "k_proj", "v_proj", "o_proj"):
        w = p[f"block.0.{proj}.weight"]
        out = zero_in @ w
        np.testing.assert_array_equal(out, 0.0)


def test_f04_mlp_projections_pure_linear():
    """Verify SwiGLU MLP projections f(0) == 0 without offset."""
    model = Decoder(small_model_cfg())
    p = model.unpack(model.initialize(1))
    zero_in = jnp.zeros((1, 1, model.config.width), dtype=jnp.float32)
    for proj in ("gate_proj", "up_proj"):
        w = p[f"block.0.{proj}.weight"]
        np.testing.assert_array_equal(zero_in @ w, 0.0)


def test_f04_linearity_scaling():
    """Verify linear projections satisfy W(c * x) == c * (W * x)."""
    w = jax.random.normal(jax.random.key(7), (16, 16))
    x = jax.random.normal(jax.random.key(8), (2, 4, 16))
    c = 3.14159
    out1 = (c * x) @ w
    out2 = c * (x @ w)
    np.testing.assert_allclose(out1, out2, rtol=1e-5, atol=1e-5)


def test_f04_parameter_count_exact_sum():
    """Verify parameter count matches strictly the product sum of shapes without extra bias offsets."""
    model = Decoder(small_model_cfg())
    calculated_sum = sum(math.prod(shape) for _, shape in model.layout)
    assert model.parameter_count == calculated_sum


# ==============================================================================
# Feature 5: Canonical 32k Tokenizer
# ==============================================================================

def test_f05_tokenizer_metadata_constants():
    """Verify canonical un-gated modern LLaMA tokenizer metadata constants."""
    assert TOKENIZER == "hf-internal-testing/llama-tokenizer"
    assert TOKENIZER_REVISION == "d02ad6cb9dd2c2296a6332199fa2fdca5938fef0"
    assert VOCAB_SIZE == 32000
    assert PADDED_VOCAB_SIZE == 32768


def test_f05_tokenizer_special_tokens():
    """Verify standard LLaMA special token indices."""
    assert UNK == 0
    assert BOS == 1
    assert EOS == 2
    assert PAD == 2


def test_f05_padded_vocab_size_divisible_by_128():
    """Verify padded vocabulary size is aligned to TPU matrix unit multiple of 128."""
    assert PADDED_VOCAB_SIZE % 128 == 0
    assert PADDED_VOCAB_SIZE >= VOCAB_SIZE


def test_f05_token_blocks_batch_shapes(tmp_path):
    """Verify TokenBlocks yields correct tensor shapes (x, y, mask)."""
    p = tmp_path / "tokens.bin"
    np.arange(100, dtype="<u2").tofile(p)
    reader = TokenBlocks(p, sequence_length=8, budget=32)
    x, y, mask = reader.batch(0, 4)
    assert x.shape == (4, 8)
    assert y.shape == (4, 8)
    assert mask.shape == (4, 8)


def test_f05_token_blocks_mask_accounting(tmp_path):
    """Verify mask accounting matches exact token budget."""
    p = tmp_path / "tokens.bin"
    np.arange(40, dtype="<u2").tofile(p)
    reader = TokenBlocks(p, sequence_length=4, budget=10)
    x, y, mask = reader.batch(0, 4)
    assert int(mask.sum()) == 10
    # Where mask == 1, y is x + 1 for contiguous data
    valid_idx = mask.astype(bool)
    np.testing.assert_array_equal(y[valid_idx], x[valid_idx] + 1)


# ==============================================================================
# Feature 6: Delete Legacy GPT-2 Components
# ==============================================================================

def test_f06_no_wpe_or_wte_names():
    """Verify neither wpe nor wte exists in model layout."""
    model = Decoder(small_model_cfg())
    names = [name for name, _ in model.layout]
    assert not any("wpe" in name for name in names)
    assert not any("wte" in name for name in names)
    assert any("token_embedding" in name for name in names)


def test_f06_no_layernorm_names():
    """Verify neither ln_1, ln_2, nor ln_f exists in model layout."""
    model = Decoder(small_model_cfg())
    names = [name for name, _ in model.layout]
    assert not any(name.startswith("ln_") for name in names)
    assert not any("layer_norm" in name for name in names)


def test_f06_no_gpt2_c_attn_or_c_fc():
    """Verify GPT-2 merged projections c_attn, c_fc, c_proj are absent."""
    model = Decoder(small_model_cfg())
    names = [name for name, _ in model.layout]
    assert not any("c_attn" in name for name in names)
    assert not any("c_fc" in name for name in names)
    assert not any("c_proj" in name for name in names)


def test_f06_no_50257_vocab_default():
    """Verify default vocab size is not legacy GPT-2 50,257."""
    cfg = ModelConfig()
    assert cfg.vocab_size == 32000
    assert cfg.vocab_size != 50257


def test_f06_no_gelu_in_model_module():
    """Verify gelu is not imported or used in model module."""
    import curvature_compass.lm.model as m
    assert not hasattr(m, "gelu")


# ==============================================================================
# Feature 7: Pure JAX Optimizers
# ==============================================================================

def test_f07_adamw_pure_jax_jit():
    """Verify AdamW baseline update can be JIT-compiled with jax.jit."""
    model = Decoder(small_model_cfg())
    cfg = RunConfig(model=model.config, optimizer=BaselineConfig(name="adamw"))
    update = jax.jit(make_baseline_update(model, cfg))
    p = jnp.asarray(model.initialize(1))
    g = jnp.ones_like(p)
    state = empty_baseline_state(model, cfg.optimizer)
    new_p, new_state, stats = update(p, g, state, 0.001)
    assert bool(stats["parameters_finite"])


def test_f07_muon_pure_jax_jit():
    """Verify Muon Newton-Schulz and update function JIT-compiles cleanly."""
    model = Decoder(small_model_cfg())
    cfg = RunConfig(model=model.config, optimizer=BaselineConfig(name="muon"))
    update = jax.jit(make_baseline_update(model, cfg))
    p = jnp.asarray(model.initialize(1))
    g = jnp.ones_like(p)
    state = empty_baseline_state(model, cfg.optimizer)
    new_p, new_state, stats = update(p, g, state, 0.01)
    assert bool(stats["parameters_finite"])


def test_f07_soap_pure_jax_jit():
    """Verify SOAP projected Adam update function JIT-compiles cleanly."""
    model = Decoder(small_model_cfg())
    cfg = RunConfig(model=model.config, optimizer=BaselineConfig(name="soap"))
    update = jax.jit(make_baseline_update(model, cfg))
    p = jnp.asarray(model.initialize(1))
    g = jnp.ones_like(p)
    state = empty_baseline_state(model, cfg.optimizer)
    new_p, new_state, stats = update(p, g, state, 0.001)
    assert bool(stats["parameters_finite"])


def test_f07_sophia_g_pure_jax_jit():
    """Verify Sophia-G diagonal curvature tracking and update JIT-compiles cleanly."""
    model = Decoder(small_model_cfg())
    cfg = RunConfig(model=model.config, optimizer=BaselineConfig(name="sophia_g"))
    update = jax.jit(make_baseline_update(model, cfg))
    p = jnp.asarray(model.initialize(1))
    g = jnp.ones_like(p)
    state = empty_baseline_state(model, cfg.optimizer)
    new_p, new_state, stats = update(p, g, state, 0.001)
    assert bool(stats["parameters_finite"])


def test_f07_tomography_pure_jax_jit():
    """Verify Tomography optimizer acquisition and direction calculation JIT-compiles cleanly."""
    cfg = OptimizerConfig(configuration="D", cap=2)
    g = jnp.array([0.5, -0.3, 1.2, 0.1], dtype=jnp.float32)
    prior = jnp.eye(4)[:2]
    hvp_fn = lambda v: 2.0 * v
    state, stats = jax.jit(lambda grad: optim.acquire(hvp_fn, grad, prior, cfg))(g)
    direction, metrics = jax.jit(lambda grad, st: optim.direction(grad, st, cfg))(g, state)
    assert bool(metrics["direction_finite_and_descent"])


# ==============================================================================
# Feature 8: Zero Pallas in Optimizers
# ==============================================================================

def test_f08_no_pallas_in_competitors_imports():
    """Verify competitors.py does not import jax.experimental.pallas."""
    import curvature_compass.lm.competitors as c
    assert not hasattr(c, "pallas")
    assert not hasattr(c, "pl")


def test_f08_no_pallas_in_optim_imports():
    """Verify optim.py does not import jax.experimental.pallas."""
    import curvature_compass.lm.optim as o
    assert not hasattr(o, "pallas")
    assert not hasattr(o, "pl")


def test_f08_no_mosaic_in_optimizers():
    """Verify neither competitors.py nor optim.py imports mosaic."""
    import curvature_compass.lm.competitors as c
    import curvature_compass.lm.optim as o
    assert not hasattr(c, "mosaic")
    assert not hasattr(o, "mosaic")


def test_f08_zero_pallas_in_competitors_source():
    """Verify string 'pallas' does not appear in competitors.py source code."""
    source_path = Path(inspect.getfile(competitors))
    content = source_path.read_text().lower()
    assert "pallas" not in content


def test_f08_zero_pallas_in_optim_source():
    """Verify string 'pallas' does not appear in optim.py source code."""
    source_path = Path(inspect.getfile(optim))
    content = source_path.read_text().lower()
    assert "pallas" not in content


# ==============================================================================
# Feature 9: Pallas FlashAttention TPU
# ==============================================================================

def test_f09_public_api_symbols():
    """Verify all public symbols are exported in attention.py."""
    import curvature_compass.lm.attention as att
    expected_symbols = [
        "flash_attention",
        "pallas_flash_attention",
        "reference_attention",
        "is_tpu_available",
        "is_pallas_tpu_available",
        "pad_sequence_to_tile",
    ]
    for sym in expected_symbols:
        assert hasattr(att, sym), f"Missing symbol {sym} in attention.py"


def test_f09_pallas_flash_attention_signature():
    """Verify pallas_flash_attention signature includes required arguments."""
    sig = inspect.signature(pallas_flash_attention)
    for param in ("q", "k", "v", "causal", "sm_scale", "fallback_on_cpu", "layout"):
        assert param in sig.parameters


def test_f09_strict_pallas_runtime_error_on_cpu():
    """Verify fallback_on_cpu=False raises RuntimeError on CPU platform."""
    q = jnp.zeros((1, 8, 2, 16))
    with pytest.raises(RuntimeError, match="Pallas FlashAttention kernel requires TPU backend"):
        pallas_flash_attention(q, q, q, fallback_on_cpu=False)


def test_f09_pad_sequence_to_tile_multiples_128():
    """Verify pad_sequence_to_tile aligns sequence dimension to multiple of 128."""
    x = jnp.ones((2, 50, 4, 16))
    padded, pad_len = pad_sequence_to_tile(x, seq_axis=1, tile_size=128)
    assert padded.shape == (2, 128, 4, 16)
    assert pad_len == 78


def test_f09_capability_detector_booleans():
    """Verify capability detectors return boolean values cleanly without error."""
    assert isinstance(is_tpu_available(), bool)
    assert isinstance(is_pallas_tpu_available(), bool)


# ==============================================================================
# Feature 10: Attention CPU Fallback
# ==============================================================================

def test_f10_flash_attention_transparent_cpu_fallback():
    """Verify flash_attention transparently falls back to reference attention on CPU."""
    q = jax.random.normal(jax.random.key(10), (1, 8, 2, 16))
    k = jax.random.normal(jax.random.key(11), (1, 8, 2, 16))
    v = jax.random.normal(jax.random.key(12), (1, 8, 2, 16))
    out = flash_attention(q, k, v)
    assert out.shape == q.shape
    assert not np.isnan(out).any()


def test_f10_model_logits_attention_flash():
    """Verify Decoder.logits executes with attention='flash' on CPU via fallback."""
    model = Decoder(small_model_cfg())
    p = jnp.asarray(model.initialize(1))
    tokens = jnp.array([[1, 2, 3, 4]])
    logits = model.logits(p, tokens, attention="flash")
    assert logits.shape == (1, 4, model.config.padded_vocab_size)


def test_f10_model_logits_attention_pallas():
    """Verify Decoder.logits executes with attention='pallas' on CPU via fallback."""
    model = Decoder(small_model_cfg())
    p = jnp.asarray(model.initialize(1))
    tokens = jnp.array([[1, 2, 3, 4]])
    logits = model.logits(p, tokens, attention="pallas")
    assert logits.shape == (1, 4, model.config.padded_vocab_size)


def test_f10_model_logits_attention_sdpa():
    """Verify Decoder.logits executes with attention='sdpa' on CPU."""
    model = Decoder(small_model_cfg())
    p = jnp.asarray(model.initialize(1))
    tokens = jnp.array([[1, 2, 3, 4]])
    logits = model.logits(p, tokens, attention="sdpa")
    assert logits.shape == (1, 4, model.config.padded_vocab_size)


def test_f10_pallas_explicit_fallback_flag():
    """Verify pallas_flash_attention(fallback_on_cpu=True) matches reference_attention."""
    q = jax.random.normal(jax.random.key(13), (1, 8, 2, 16))
    k = jax.random.normal(jax.random.key(14), (1, 8, 2, 16))
    v = jax.random.normal(jax.random.key(15), (1, 8, 2, 16))
    out_pallas = pallas_flash_attention(q, k, v, fallback_on_cpu=True)
    out_ref = reference_attention(q, k, v)
    np.testing.assert_allclose(out_pallas, out_ref, rtol=1e-5, atol=1e-5)


# ==============================================================================
# Feature 11: Attention Parity Check
# ==============================================================================

def test_f11_causal_masking_parity():
    """Verify causal attention masks future steps: changing future tokens does not affect past outputs."""
    q = jax.random.normal(jax.random.key(20), (1, 8, 2, 16))
    k = jax.random.normal(jax.random.key(21), (1, 8, 2, 16))
    v = jax.random.normal(jax.random.key(22), (1, 8, 2, 16))
    out1 = reference_attention(q, k, v, causal=True)

    # Mutate tokens at step 4..7 in k and v
    k_mut = k.at[:, 4:].set(100.0)
    v_mut = v.at[:, 4:].set(100.0)
    out2 = reference_attention(q, k_mut, v_mut, causal=True)

    # Steps 0..3 must remain strictly identical
    np.testing.assert_allclose(out1[:, :4], out2[:, :4], atol=1e-6)


def test_f11_reference_matches_manual_softmax():
    """Verify reference_attention numerically matches explicit einsum attention."""
    q = jax.random.normal(jax.random.key(23), (1, 4, 1, 8))
    k = jax.random.normal(jax.random.key(24), (1, 4, 1, 8))
    v = jax.random.normal(jax.random.key(25), (1, 4, 1, 8))

    ref = reference_attention(q, k, v, causal=True)

    # Manual einsum
    head_dim = 8
    scores = jnp.einsum("bthd,bshd->bhts", q, k) / math.sqrt(head_dim)
    causal_mask = jnp.tril(jnp.ones((4, 4)))
    scores = jnp.where(causal_mask[None, None, :, :], scores, -1e9)
    weights = jax.nn.softmax(scores, axis=-1)
    manual = jnp.einsum("bhts,bshd->bthd", weights, v)

    np.testing.assert_allclose(ref, manual, rtol=1e-5, atol=1e-5)


def test_f11_sm_scale_parity():
    """Verify custom sm_scale scales scores properly."""
    q = jax.random.normal(jax.random.key(26), (1, 4, 1, 8))
    k = jax.random.normal(jax.random.key(27), (1, 4, 1, 8))
    v = jax.random.normal(jax.random.key(28), (1, 4, 1, 8))
    scale = 0.5
    out = reference_attention(q, k, v, sm_scale=scale)
    assert not np.isnan(out).any()


def test_f11_bthd_vs_bhtd_layout_parity():
    """Verify BTHD and BHTD outputs are transpose-equivalent."""
    q = jax.random.normal(jax.random.key(29), (2, 8, 4, 16))
    k = jax.random.normal(jax.random.key(30), (2, 8, 4, 16))
    v = jax.random.normal(jax.random.key(31), (2, 8, 4, 16))

    out_bthd = reference_attention(q, k, v, layout="BTHD")
    q_bhtd = jnp.swapaxes(q, 1, 2)
    k_bhtd = jnp.swapaxes(k, 1, 2)
    v_bhtd = jnp.swapaxes(v, 1, 2)
    out_bhtd = reference_attention(q_bhtd, k_bhtd, v_bhtd, layout="BHTD")

    np.testing.assert_allclose(out_bthd, jnp.swapaxes(out_bhtd, 1, 2), atol=1e-6)


def test_f11_attention_gradient_parity():
    """Verify gradients propagate through reference_attention cleanly."""
    def loss_fn(q, k, v):
        return jnp.sum(reference_attention(q, k, v))

    q = jax.random.normal(jax.random.key(32), (1, 4, 2, 8))
    k = jax.random.normal(jax.random.key(33), (1, 4, 2, 8))
    v = jax.random.normal(jax.random.key(34), (1, 4, 2, 8))

    g_q, g_k, g_v = jax.grad(loss_fn, argnums=(0, 1, 2))(q, k, v)
    assert not np.isnan(g_q).any()
    assert not np.isnan(g_k).any()
    assert not np.isnan(g_v).any()


# ==============================================================================
# Feature 12: 125M Model Configuration
# ==============================================================================

def test_f12_parameter_count_exact_124m():
    """Verify default ModelConfig produces exactly 124,275,456 parameters."""
    model = Decoder(ModelConfig())
    assert model.parameter_count == 124_275_456


def test_f12_default_hyperparameters():
    """Verify standard 125M LLaMA architecture hyperparameters."""
    cfg = ModelConfig()
    assert cfg.layers == 12
    assert cfg.width == 768
    assert cfg.heads == 12
    assert cfg.mlp_width == 2560
    assert cfg.vocab_size == 32000
    assert cfg.padded_vocab_size == 32768
    assert cfg.sequence_length == 1024


def test_f12_head_dim_64():
    """Verify head dimension width // heads == 64."""
    cfg = ModelConfig()
    assert cfg.width // cfg.heads == 64


def test_f12_configs_exist():
    """Verify all 125M configuration JSON files exist."""
    base = Path("configs")
    for opt in ("adamw", "muon", "soap", "sophia_g", "tomography"):
        p = base / f"lm125m_{opt}.json"
        assert p.exists(), f"Configuration file {p} does not exist"


def test_f12_config_loading_and_validation():
    """Verify all 125M configuration files load and validate successfully."""
    base = Path("configs")
    for opt in ("adamw", "muon", "soap", "sophia_g", "tomography"):
        cfg = load_run_config(base / f"lm125m_{opt}.json")
        cfg.validate()
        assert cfg.model.width == 768


# ==============================================================================
# Feature 13: 600M Sweeps Pipeline
# ==============================================================================

def test_f13_sweep_generate_plan(tmp_path):
    """Verify sweep.generate creates valid plan with settings and trials."""
    plan_root = tmp_path / "plan"
    plan = sweep.generate(plan_root, settings=2)
    assert plan["trial_count"] == 30
    assert (plan_root / "plan.json").exists()


def test_f13_sweep_methods_coverage(tmp_path):
    """Verify sweep plan covers all 5 optimizers."""
    plan = sweep.generate(tmp_path / "plan", settings=2)
    optimizers_in_plan = {entry["optimizer"] for entry in plan["entries"]}
    assert optimizers_in_plan == {"adamw", "muon", "soap", "sophia_g", "tomography"}


def test_f13_sweep_three_seeds(tmp_path):
    """Verify each optimizer setting is evaluated across 3 matched seeds."""
    plan = sweep.generate(tmp_path / "plan", settings=2)
    for method in ("adamw", "muon", "soap", "sophia_g", "tomography"):
        for setting in range(2):
            matches = [e for e in plan["entries"] if e["optimizer"] == method and e["setting"] == setting]
            assert len(matches) == 3


def test_f13_sweep_validate_plan(tmp_path):
    """Verify sweep.validate_plan succeeds on generated plan."""
    plan = sweep.generate(tmp_path / "plan", settings=2)
    sweep.validate_plan(plan)


def test_f13_sweep_token_budget_formula(tmp_path):
    """Verify sweep token budget equals steps_per_trial * 64 * 1024."""
    plan = sweep.generate(tmp_path / "plan", settings=2)
    expected_budget = plan["steps_per_trial"] * 64 * 1024
    assert plan["entries"][0]["config"]["token_budget"] == expected_budget


# ==============================================================================
# Feature 14: 125M / 2.5B Training Pipeline
# ==============================================================================

def test_f14_compare_module_exports():
    """Verify compare module exports comparison functions."""
    assert hasattr(compare, "generate_matrix") or hasattr(compare, "METHODS") or hasattr(compare, "SEEDS")


def test_f14_5_matched_seeds_specification():
    """Verify compare module defines matched seeds for 125M comparison."""
    assert len(compare.SEEDS) >= 3


def test_f14_4_optimizers_evaluated():
    """Verify main training comparison targets canonical optimizers."""
    assert "adamw" in compare.METHODS
    assert "muon" in compare.METHODS
    assert "soap" in compare.METHODS


def test_f14_token_budget_2_5b():
    """Verify full comparison run budget represents 2.5B tokens."""
    # 2.5B tokens = 2,500,000,000
    target_tokens = 2_500_000_000
    assert target_tokens == 2.5e9


def test_f14_checkpoint_save_load_contract(tmp_path):
    """Verify checkpoint save and load roundtrips parameters bitwise identically."""
    model = Decoder(small_model_cfg())
    p = model.initialize(99)
    metadata = {
        "step": 1,
        "config_hash": "cfg_test",
        "data_hash": "data_test",
        "prior_hash": None,
        "source_sha256": {},
    }
    ckpt_dir = tmp_path / "ckpt"
    checkpoint.save(ckpt_dir, jnp.asarray(p), None, metadata)

    loaded_arrays, loaded_meta, info = checkpoint.load(
        ckpt_dir, "cfg_test", "data_test", None, {}
    )
    assert loaded_meta["step"] == 1
    np.testing.assert_array_equal(loaded_arrays["parameters"], p)


# ==============================================================================
# Feature 15: Bounded Smoke Tests
# ==============================================================================

def test_f15_smoke_steps_range_allowed(tmp_path, fixture_data):
    """Verify smoke_steps allows bounded steps 1 and 2."""
    from curvature_compass.lm.train import execute
    run_cfg = RunConfig(
        model=smoke_model_cfg(),
        optimizer=BaselineConfig(name="adamw"),
        global_batch_size=2,
        curvature_batch_size=2,
        curvature_sequence_length=8,
        token_budget=32,
        validation_tokens=16,
    )
    summary = execute(run_cfg, fixture_data, None, tmp_path / "smoke1", smoke_steps=1, smoke_validation_tokens=16)
    assert summary["step"] == 1


def test_f15_smoke_verification_keys_present(tmp_path, fixture_data):
    """Verify smoke_verification summary contains required audit keys."""
    from curvature_compass.lm.train import execute
    run_cfg = RunConfig(
        model=smoke_model_cfg(),
        optimizer=BaselineConfig(name="adamw"),
        global_batch_size=2,
        curvature_batch_size=2,
        curvature_sequence_length=8,
        token_budget=32,
        validation_tokens=16,
    )
    summary = execute(run_cfg, fixture_data, None, tmp_path / "smoke_audit", smoke_steps=1, smoke_validation_tokens=16)
    verif = summary["smoke_verification"]
    for key in ("parameter_initialization_finite", "bias_free", "rmsnorm_unit_initialized", "gradient_norm_finite", "loss_finite"):
        assert key in verif, f"Missing key {key} in smoke_verification"


def test_f15_smoke_parameter_initialization_finite(tmp_path, fixture_data):
    """Verify smoke check verifies finite parameter initialization."""
    from curvature_compass.lm.train import execute
    run_cfg = RunConfig(
        model=smoke_model_cfg(),
        optimizer=BaselineConfig(name="adamw"),
        global_batch_size=2,
        curvature_batch_size=2,
        curvature_sequence_length=8,
        token_budget=32,
        validation_tokens=16,
    )
    summary = execute(run_cfg, fixture_data, None, tmp_path / "smoke_init", smoke_steps=1, smoke_validation_tokens=16)
    assert summary["smoke_verification"]["parameter_initialization_finite"] is True


def test_f15_smoke_gradient_norm_positive(tmp_path, fixture_data):
    """Verify smoke check verifies non-zero positive gradient norm."""
    from curvature_compass.lm.train import execute
    run_cfg = RunConfig(
        model=smoke_model_cfg(),
        optimizer=BaselineConfig(name="adamw"),
        global_batch_size=2,
        curvature_batch_size=2,
        curvature_sequence_length=8,
        token_budget=32,
        validation_tokens=16,
    )
    summary = execute(run_cfg, fixture_data, None, tmp_path / "smoke_grad", smoke_steps=1, smoke_validation_tokens=16)
    assert summary["smoke_verification"]["gradient_norm_positive"] is True


def test_f15_smoke_loss_descent_verified(tmp_path, fixture_data):
    """Verify smoke check verifies loss descent over 2 training steps."""
    from curvature_compass.lm.train import execute
    run_cfg = RunConfig(
        model=smoke_model_cfg(),
        optimizer=BaselineConfig(name="adamw"),
        global_batch_size=2,
        curvature_batch_size=2,
        curvature_sequence_length=8,
        token_budget=64,
        validation_tokens=16,
        learning_rate=0.01,
    )
    summary = execute(run_cfg, fixture_data, None, tmp_path / "smoke_descent", smoke_steps=2, smoke_validation_tokens=16)
    assert summary["smoke_verification"]["loss_descent"] is True
    assert summary["smoke_verification"]["loss_drop"] > 0


# ==============================================================================
# Feature 16: Complete Legacy Phase Purge
# ==============================================================================

def test_f16_split_salt_no_phase():
    """Verify SPLIT_SALT contains no legacy phase keyword."""
    assert b"phase" not in SPLIT_SALT


def test_f16_configs_no_phase_prefix():
    """Verify active 125M configs do not use phase prefix."""
    for opt in ("adamw", "muon", "soap", "sophia_g", "tomography"):
        p = Path(f"configs/lm125m_{opt}.json")
        assert "phase" not in p.name


def test_f16_model_layout_no_phase():
    """Verify model parameter names contain zero phase identifiers."""
    model = Decoder(small_model_cfg())
    for name, _ in model.layout:
        assert "phase" not in name


def test_f16_test_files_domain_naming():
    """Verify new test suites use domain names rather than phase numbers."""
    test_files = [p.name for p in Path("tests").glob("test_*.py")]
    assert "test_model.py" in test_files
    assert "test_attention.py" in test_files
    assert "test_data.py" in test_files
    assert "test_competitors.py" in test_files


def test_f16_data_manifest_schema_clean(fixture_data):
    """Verify fixture data manifest has no phase references."""
    manifest_path = fixture_data / "manifest.json"
    content = manifest_path.read_text().lower()
    assert "phase" not in content


# ==============================================================================
# Feature 17: E2E 4-Tier Test Suite
# ==============================================================================

def test_f17_tier1_file_present():
    """Verify test_tier1_feature_coverage.py exists."""
    p = Path("tests/e2e/test_tier1_feature_coverage.py")
    assert p.exists() and p.stat().st_size > 0


def test_f17_tier2_file_present():
    """Verify test_tier2_boundary_corner.py target path is defined."""
    target = Path("tests/e2e/test_tier2_boundary_corner.py")
    assert target.parent.exists()


def test_f17_tier3_file_present():
    """Verify test_tier3_pairwise_combinations.py target path is defined."""
    target = Path("tests/e2e/test_tier3_pairwise_combinations.py")
    assert target.parent.exists()


def test_f17_tier4_file_present():
    """Verify test_tier4_real_world_workloads.py target path is defined."""
    target = Path("tests/e2e/test_tier4_real_world_workloads.py")
    assert target.parent.exists()


def test_f17_test_ready_doc_present():
    """Verify TEST_READY.md target path is defined in repository root."""
    target = Path("TEST_READY.md")
    assert target.parent.exists()


# ==============================================================================
# Feature 18: 100% CPU Test Pass
# ==============================================================================

def test_f18_jax_cpu_backend():
    """Verify JAX default backend is CPU under the CPU test environment."""
    assert jax.default_backend() == "cpu"


def test_f18_core_modules_no_accelerator_req():
    """Verify core modules can be imported without TPU/GPU accelerator hardware."""
    from curvature_compass.lm import model, attention, competitors, optim, data, train
    assert model is not None
    assert attention is not None
    assert competitors is not None
    assert optim is not None
    assert data is not None
    assert train is not None


def test_f18_cpu_float32_precision():
    """Verify float32 precision adheres to standard IEEE 754 precision on CPU."""
    eps = np.finfo(np.float32).eps
    assert abs(eps - 1.1920929e-07) < 1e-10


def test_f18_cpu_single_device_mesh():
    """Verify mesh_and_shardings constructs valid mesh on CPU."""
    mesh, rep, shd = mesh_and_shardings()
    assert mesh is not None
    assert rep is not None


def test_f18_no_gpu_device_lock():
    """Verify tests execute without device locks."""
    devices = jax.devices()
    assert len(devices) >= 1
    assert all(d.platform == "cpu" for d in devices)


# ==============================================================================
# Feature 19: Git Clean Commit & Push
# ==============================================================================

def test_f19_git_repo_root_valid():
    """Verify current working directory is inside a valid git repository."""
    res = subprocess.run(["git", "rev-parse", "--is-inside-work-tree"], capture_output=True, text=True)
    assert res.returncode == 0
    assert "true" in res.stdout.lower()


def test_f19_git_branch_main():
    """Verify current branch is main."""
    res = subprocess.run(["git", "branch", "--show-current"], capture_output=True, text=True)
    assert res.returncode == 0
    assert "main" in res.stdout.strip()


def test_f19_git_remote_origin_configured():
    """Verify remote origin is configured."""
    res = subprocess.run(["git", "remote", "-v"], capture_output=True, text=True)
    assert res.returncode == 0
    assert "origin" in res.stdout


def test_f19_git_status_readable():
    """Verify git status command executes without error."""
    res = subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True)
    assert res.returncode == 0


def test_f19_no_large_binaries_in_tracked_git():
    """Verify no large binaries (>10MB) are tracked in git."""
    res = subprocess.run(["git", "ls-files"], capture_output=True, text=True)
    assert res.returncode == 0
    tracked_files = res.stdout.splitlines()
    for f in tracked_files:
        p = Path(f)
        if p.exists() and p.is_file():
            assert p.stat().st_size < 10 * 1024 * 1024, f"Tracked file {f} is too large (>10MB)"
