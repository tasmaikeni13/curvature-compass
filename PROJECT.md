# Project: Curvature Compass Refactoring & Publication

## Architecture
Curvature Compass is a modern JAX/XLA research framework for second-order and adaptive optimization on deep transformer architectures, targeting Google TPU Pod hardware (v4-32) with pure JAX CPU compatibility.

### Module Boundaries & Data Flow
1. **Core Decoder Model (`src/curvature_compass/lm/model.py`, `config.py`)**:
   - Modern LLaMA specifications: RMSNorm (no learnable bias), Rotary Position Embeddings (RoPE), SwiGLU FFN (`gate_proj`, `up_proj`, `down_proj` with SiLU), and 100% bias-free linear projections.
   - Padded canonical vocabulary size: 32,768 (un-gated LLaMA 32,000 vocab padded for TPU MXU alignment).
   - Tied token embeddings and output projections.
2. **Attention Subsystem (`src/curvature_compass/lm/attention.py`)**:
   - High-performance FlashAttention kernel targeting TPU v4 via `jax.experimental.pallas`.
   - Automatic fallback to reference attention (`jax.nn.dot_product_attention`) on CPU.
   - Numerical parity within floating-point tolerance between reference and Pallas.
3. **Data Pipeline (`src/curvature_compass/lm/data.py`)**:
   - Pinned tokenizer: `hf-internal-testing/llama-tokenizer` (32,000 vocab, padded to 32,768, BOS=1, EOS=2, UNK=0).
   - Sharded data streaming, text batching, deterministic splitting, and token accounting for FineWeb-Edu.
4. **Pure JAX/XLA Optimizers (`src/curvature_compass/lm/competitors.py`, `optim.py`)**:
   - AdamW, Muon, SOAP, Sophia-G, and Tomography implemented strictly in standard JAX/XLA operations. Zero custom Pallas/Mosaic kernels.
   - Compatible with bias-free LLaMA projection layouts.
5. **Training, Sweep & Pipeline Runners (`src/curvature_compass/lm/train.py`, `sweep.py`, `compare.py`)**:
   - 125M model configuration tuned for 2.5B FineWeb-Edu tokens (124,275,456 parameters).
   - 600M token per optimizer hyperparameter sweep configs and runner.
   - 125M / 2.5B tokens training pipeline comparing 4 optimizers across 5 matched seeds.
   - Fast bounded smoke test (`--smoke-steps`) verifying parameter initialization, parameter count, gradient calculation, and loss descent.
6. **E2E Testing & Verification (`tests/`, `TEST_INFRA.md`, `TEST_READY.md`)**:
   - Systematic 4-tier requirement-driven opaque-box test suite with 100% pass rate on CPU.

## Code Layout
```
src/curvature_compass/
  lm/
    model.py          # LLaMA Transformer decoder (RMSNorm, RoPE, SwiGLU, bias-free)
    attention.py      # Pallas FlashAttention TPU kernel with CPU fallback
    config.py         # Model and training configuration (125M defaults)
    data.py           # 32k LLaMA tokenizer integration and dataset batching
    optim.py          # Tomography optimizer (pure JAX/XLA)
    competitors.py    # AdamW, Muon, SOAP, Sophia-G optimizers (pure JAX/XLA)
    execution.py      # SPMD mesh and device execution helpers
    train.py          # Training runner with bounded smoke tests
    sweep.py          # 600M token per optimizer hyperparameter sweep runner
    compare.py        # 125M / 2.5B tokens multi-seed comparison pipeline
configs/
  lm125m_adamw.json   # 125M config for AdamW
  lm125m_muon.json    # 125M config for Muon
  lm125m_soap.json    # 125M config for SOAP
  lm125m_sophia_g.json# 125M config for Sophia-G
tests/
  test_model.py       # Architecture and parameter count tests
  test_attention.py   # FlashAttention parity and CPU fallback tests
  test_data.py        # Tokenizer and batching tests
  test_competitors.py # Optimizer tests
  test_train.py       # Training and smoke test tests
  e2e/                # Systematic 4-tier E2E tests
```

## Feature Inventory
| # | Feature | Description | Milestone | Source |
|---|---------|-------------|-----------|--------|
| 1 | LLaMA RMSNorm | Root-mean-square normalization without learnable bias replacing LayerNorm | M1 | ORIGINAL_REQUEST §R1 |
| 2 | Rotary Position Embeddings | RoPE applied to query and key projections, removing learned absolute position embeddings | M1 | ORIGINAL_REQUEST §R1 |
| 3 | SwiGLU FFN | Gate, up, down projections with SiLU activation replacing GELU MLP | M1 | ORIGINAL_REQUEST §R1 |
| 4 | Bias-Free Linears | Completely bias-free linear layers throughout the model | M1 | ORIGINAL_REQUEST §R1 |
| 5 | Canonical 32k Tokenizer | `hf-internal-testing/llama-tokenizer` (32,000 vocab padded to 32,768, BOS=1, EOS=2) | M1 | ORIGINAL_REQUEST §R1 |
| 6 | Delete GPT-2 Remnants | Remove all legacy GPT-2 components, shapes, and dead code | M1 | ORIGINAL_REQUEST §R1 |
| 7 | Pure JAX Optimizers | Verify all benchmark optimizers (AdamW, Muon, SOAP, Sophia-G, Tomography) are 100% pure JAX/XLA | M2 | ORIGINAL_REQUEST §R2 |
| 8 | Zero Pallas in Optimizers | Verify strictly zero Pallas kernels in optimizer code | M2 | ORIGINAL_REQUEST §R2 |
| 9 | Pallas FlashAttention TPU | Engineer high-performance FlashAttention kernel targeting TPU v4 | M2 | ORIGINAL_REQUEST §R3 |
| 10 | Attention CPU Fallback | Seamless automatic fallback to reference / SDPA attention on CPU | M2 | ORIGINAL_REQUEST §R3 |
| 11 | Attention Parity Check | Numerical parity verification between Pallas kernel and reference attention | M2 | ORIGINAL_REQUEST §R3 |
| 12 | 125M Model Configuration | 124,275,456 parameter configuration for 2.5B tokens (12 layers, width 768, heads 12, mlp 2560) | M3 | ORIGINAL_REQUEST §R4 |
| 13 | 600M Sweeps Pipeline | 600M token per optimizer hyperparameter sweep configs and runner | M3 | ORIGINAL_REQUEST §R4 |
| 14 | 125M / 2.5B Training Pipeline | 125M / 2.5B token training pipeline comparing 4 optimizers across 5 matched seeds | M3 | ORIGINAL_REQUEST §R4 |
| 15 | Bounded Smoke Tests | Fast bounded smoke tests (`--smoke-steps`) verifying init, param count, gradients, loss descent | M3 | ORIGINAL_REQUEST §R4 |
| 16 | Legacy Terminology Purge | Complete removal of legacy staging terminology references across repository | M4 | ORIGINAL_REQUEST §R5 |
| 17 | E2E 4-Tier Test Suite | Category-Partition, BVA, Pairwise, Real-World test suite and `TEST_READY.md` | M5 | ORIGINAL_REQUEST §R6 |
| 18 | 100% CPU Test Pass | All unit tests pass 100% on CPU | M6 | ORIGINAL_REQUEST §R6 |
| 19 | Git Clean Commit & Push | Clean commits and successful push to origin main | M6 | ORIGINAL_REQUEST §R6 |

## Milestones
| # | Name | Scope | Dependencies | Status |
|---|------|-------|-------------|--------|
| M1 | Modern LLaMA Architecture & Tokenizer | Features 1-6: RMSNorm, RoPE, SwiGLU, bias-free, 32k tokenizer in `model.py`, `config.py`, `data.py` | None | DONE |
| M2 | Pure JAX Optimizers & Pallas Attention | Features 7-11: `attention.py` with CPU fallback; verify pure JAX optimizers in `competitors.py`, `optim.py` | M1 | DONE |
| M3 | 125M Config, 600M Sweeps & 2.5B Pipeline | Features 12-15: 125M configs, sweep runner, 2.5B pipeline, bounded smoke tests in `train.py`, `sweep.py`, `compare.py` | M1, M2 | DONE |
| M4 | Complete Legacy Terminology Purge | Feature 16: Purge all legacy staging references across repository, remove obsolete artifacts | M3 | DONE |
| M5 | E2E Testing Suite (E2E Track) | Feature 17: Systematic 4-tier E2E testing track, publish `TEST_READY.md` | M1 | DONE |
| M6 | Final Verification & Git Push | Features 18-19: Full test suite pass, audit verification, clean git commits and push to origin main | M1, M2, M3, M4, M5 | DONE |

## Interface Contracts
### `model.py` ↔ `attention.py`
- `attention.py:flash_attention(q, k, v, causal=True, sm_scale=None) -> jnp.ndarray`
- Input shapes: `q, k, v`: `(batch, seq_len, heads, head_dim)` or `(batch, heads, seq_len, head_dim)`
- Output shape: matches query shape `(batch, seq_len, heads, head_dim)`
- Behavior: executes Pallas kernel if TPU device available; falls back automatically to `jax.nn.dot_product_attention` on CPU.

### `model.py` ↔ `competitors.py`
- All projection weights in `model.layout` are 2D arrays (`(in_features, out_features)`).
- Biases are strictly absent from `model.layout`.
- RMSNorm weights are 1D arrays (`(dim,)`) and marked with zero weight decay via `decay_mask`.
- Muon applies to 2D projection weights (`hidden_matrix(name, shape)`).
- AdamW applies to 1D RMSNorm scales and embeddings.

### `config.py` ↔ `train.py` / `sweep.py` / `compare.py`
- Default `ModelConfig`:
  - `vocab_size: int = 32000`
  - `padded_vocab_size: int = 32768`
  - `width: int = 768`
  - `layers: int = 12`
  - `heads: int = 12`
  - `mlp_width: int = 2560`
  - `sequence_length: int = 1024`
  - Parameter count: exactly 124,275,456.
