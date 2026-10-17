# E2E Test Infra: Curvature Compass

## Test Philosophy
- Opaque-box, requirement-driven testing derived from user requirements in `ORIGINAL_REQUEST.md`.
- No dependency on internal implementation details; validates public APIs, training CLI flags, parameter constraints, and numerical tolerances.
- Methodology: Category-Partition + Boundary Value Analysis (BVA) + Pairwise Combinatorial Testing + Real-World Workload Testing.

## Feature Inventory
| # | Feature | Source (Requirement) | Tier 1 (Coverage >=5) | Tier 2 (Boundary >=5) | Tier 3 (Pairwise) | Tier 4 (Real-World) |
|---|---------|----------------------|:---------------------:|:---------------------:|:-----------------:|:-------------------:|
| 1 | LLaMA RMSNorm | ORIGINAL_REQUEST §R1 | 5 | 5 | ✓ | ✓ |
| 2 | Rotary Position Embeddings (RoPE) | ORIGINAL_REQUEST §R1 | 5 | 5 | ✓ | ✓ |
| 3 | SwiGLU Feed-Forward Network | ORIGINAL_REQUEST §R1 | 5 | 5 | ✓ | ✓ |
| 4 | Bias-Free Linear Projections | ORIGINAL_REQUEST §R1 | 5 | 5 | ✓ | ✓ |
| 5 | Canonical 32k LLaMA Tokenizer | ORIGINAL_REQUEST §R1 | 5 | 5 | ✓ | ✓ |
| 6 | Delete Legacy GPT-2 Components | ORIGINAL_REQUEST §R1 | 5 | 5 | ✓ | ✓ |
| 7 | Pure JAX Optimizers (AdamW, Muon, SOAP, Sophia-G) | ORIGINAL_REQUEST §R2 | 5 | 5 | ✓ | ✓ |
| 8 | Zero Pallas in Optimizers | ORIGINAL_REQUEST §R2 | 5 | 5 | ✓ | ✓ |
| 9 | Pallas FlashAttention for TPU v4 | ORIGINAL_REQUEST §R3 | 5 | 5 | ✓ | ✓ |
| 10 | FlashAttention CPU Fallback | ORIGINAL_REQUEST §R3 | 5 | 5 | ✓ | ✓ |
| 11 | Numerical Parity (Pallas vs Reference) | ORIGINAL_REQUEST §R3 | 5 | 5 | ✓ | ✓ |
| 12 | 125M Model Configuration (124,275,456 params) | ORIGINAL_REQUEST §R4 | 5 | 5 | ✓ | ✓ |
| 13 | 600M Token Hyperparameter Sweeps | ORIGINAL_REQUEST §R4 | 5 | 5 | ✓ | ✓ |
| 14 | 125M / 2.5B Token Training Pipeline | ORIGINAL_REQUEST §R4 | 5 | 5 | ✓ | ✓ |
| 15 | Bounded Smoke Tests (`--smoke-steps`) | ORIGINAL_REQUEST §R4 | 5 | 5 | ✓ | ✓ |
| 16 | Complete Legacy Phase Purge | ORIGINAL_REQUEST §R5 | 5 | 5 | ✓ | ✓ |

## Test Architecture
- Test runner: `PYTHONPATH=src OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 JAX_PLATFORMS=cpu /home/tasma/venv/bin/python -m pytest -q`
- E2E Test directory: `tests/` and `tests/e2e/`
- Directory layout:
  - `tests/test_model.py`: Model architecture, RMSNorm, RoPE, SwiGLU, bias-free assertions, parameter counts
  - `tests/test_attention.py`: FlashAttention parity, block size limits, causal masking, CPU fallback
  - `tests/test_data.py`: Tokenizer encode/decode, 32k vocab, padding to 32768, batching, EOS/BOS tokens
  - `tests/test_competitors.py`: Pure JAX optimizer updates, weight decay on 1D RMSNorm, 2D weight updates
  - `tests/test_train.py`: Training CLI, smoke tests (`--smoke-steps`), checkpoint save/restore, loss descent
  - `tests/test_phase_purge.py`: Verification that zero phase references exist across repo files

## Coverage Thresholds
- Total identified core features $N = 16$.
- Tier 1: $\ge 5$ per feature ($16 \times 5 = 80$ test cases).
- Tier 2: $\ge 5$ boundary/corner cases per feature ($16 \times 5 = 80$ test cases).
- Tier 3: Pairwise combinations across major features ($\ge 16$ test cases).
- Tier 4: Realistic application workflows ($\ge \max(5, 16/2) = 8$ scenarios).
