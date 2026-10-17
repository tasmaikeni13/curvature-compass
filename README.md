# Curvature Compass

Curvature Compass is a modern JAX/XLA research framework for second-order and
adaptive optimization on deep transformer architectures, targeting Google Cloud
TPU Pod hardware (v4-32) with pure JAX CPU compatibility for development and
testing.

The framework benchmarks adaptive directional curvature acquisition (Hessian
tomography) alongside established first- and second-order optimizers on a modern
LLaMA-style autoregressive language model.

## Key Features

- **Modern LLaMA Architecture**: Decoder-only transformer implementing RMSNorm
  (no learnable bias), Rotary Position Embeddings (RoPE), SwiGLU feed-forward
  networks, 100% bias-free linear projections, and tied token embeddings.
- **Canonical 32k Tokenizer**: Integration with `hf-internal-testing/llama-tokenizer`
  (32,000 un-gated vocabulary padded to 32,768 for TPU matrix unit alignment,
  BOS=1, EOS=2).
- **High-Performance Attention**: Custom FlashAttention kernel implemented in
  `jax.experimental.pallas` targeting TPU v4, with automatic fallback to
  reference attention (`jax.nn.dot_product_attention`) under CPU environments.
- **Pure JAX/XLA Optimizers**: Complete implementations of AdamW, Muon (with
  auxiliary AdamW), SOAP, Sophia-G, and Tomography—strictly using standard
  JAX/XLA operations with zero custom Pallas kernels in optimizer code.
- **125M Parameter Benchmark Model**: Controlled 124,275,456-parameter
  configuration tuned for 2.5B FineWeb-Edu tokens across 5 matched seeds.
- **600M Token Hyperparameter Sweeps**: Systematic tuning pipeline allocating a
  600M FineWeb-Edu token budget per optimizer across matched initialization and
  data-order seeds.
- **Bounded Smoke Tests**: Instant verification (`--smoke-steps`) validating
  parameter initialization, parameter counts, gradient flow, and loss descent.
- **Formal Verification**: 15 machine-checked theorems in Lean 4 verifying
  curvature bounds, subspace optimality, and information-theoretic limits.

## Model Specifications

| Parameter | 125M LLaMA Model Configuration |
|---|---|
| Total Parameters | 124,275,456 (~124.28M) |
| Layers ($L$) | 12 |
| Model Dimension ($d$) | 768 |
| Attention Heads | 12 (head dimension $d_k = 64$) |
| FFN Intermediate Dimension | 2,560 (SwiGLU, ratio $\approx 3.33 \times d$) |
| Vocabulary Size | 32,000 (padded to 32,768 for TPU alignment) |
| Context Sequence Length | 1,024 |
| Positional Embedding | RoPE ($\theta = 10,000.0$) |
| Normalization | RMSNorm ($\epsilon = 10^{-5}$, zero bias) |
| Linear Projections | 100% bias-free |
| Weight Tying | Token embeddings tied to output projection logits |

## Optimizers Under Benchmark

1. **AdamW**: Standard decoupled weight decay adaptive optimizer.
2. **Muon**: Matrix update optimizer using Newton-Schulz iterations on 2D hidden
   weights, with auxiliary AdamW for 1D vectors (RMSNorm scales and embeddings).
3. **SOAP**: Second-order optimizer preconditioning gradients with empirical
   covariance eigenspaces.
4. **Sophia-G**: Second-order optimizer using stochastic diagonal Hessian
   estimates with elementwise clipping.
5. **Tomography**: Directional curvature framework adaptively selecting
   Hessian-vector product measurement vectors via Krylov subspace expansion and
   damped subspace inversion.

All optimizers are implemented strictly in standard JAX/XLA operations without
custom Pallas kernels.

## Quickstart & Verification

### Environment Setup
The project runs under Python 3.10 with JAX and standard dependencies:
```sh
pip install -e '.[lm,tpu,test]'
```

### Running the Test Suite (CPU)
Run the unit test suite on CPU:
```sh
PYTHONPATH=src OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 JAX_PLATFORMS=cpu /home/tasma/venv/bin/python -m pytest -q
```

### Bounded Smoke Tests
Verify initialization, gradient computation, parameter counts, and multi-step
loss descent without running full training:
```sh
PYTHONPATH=src JAX_PLATFORMS=cpu /home/tasma/venv/bin/python -m curvature_compass.lm.train \
  --config configs/lm125m_adamw.json --smoke-steps 3
```

## Experimental Pipelines

### 1. 600M Token Hyperparameter Sweeps
Systematic tuning across hyperparameter configurations with a 600M FineWeb-Edu
token budget per optimizer:
```sh
# Generate the deterministic sweep plan
PYTHONPATH=src JAX_PLATFORMS=cpu /home/tasma/venv/bin/python -m curvature_compass.lm.sweep plan \
  --output runs/sweep-plan
```

### 2. 125M / 2.5B Token Training Pipeline
Full comparative training of all five optimizers on 2.5 billion tokens across 5
matched seeds (`51001` through `51005`):
```sh
# Launch bounded smoke verification on a specific optimizer config
PYTHONPATH=src JAX_PLATFORMS=cpu /home/tasma/venv/bin/python -m curvature_compass.lm.train \
  --config configs/lm125m_muon.json --smoke-steps 5
```

### TPU Pod Distributed Execution
This workspace is worker 0 of a Google Cloud TPU v4-32 pod (`w0`–`w3`).
Multi-worker jobs are coordinated via `scripts/pod.py` with cooperative worker
locks:
```sh
/home/tasma/venv/bin/python scripts/pod.py --timeout 86400 --logs runs/adamw/launcher \
  curvature_compass.lm.train --config configs/lm125m_adamw.json \
  --data data/fineweb --output runs/adamw --smoke-steps 5
```

## Theoretical Foundations & Formal Verification

The mathematical theory of directional curvature estimation is documented in
[`theory.md`](theory.md). Supporting mathematical derivations and synthetic
simulations are archived in `analysis/`.

The Lean 4 formalization resides in [`formal/`](formal/):
```sh
cd formal && lake build
```
The build validates 15 proved results covering quadratic gap bounds, subspace
optimality, residual perturbation bounds, and information-theoretic scalar
blindness under Lean's foundational axioms (`propext`, `Classical.choice`,
`Quot.sound`) with zero `sorryAx`.

## Repository Structure

```
├── configs/
│   ├── lm125m_adamw.json       # 125M config for AdamW (2.5B token budget)
│   ├── lm125m_muon.json        # 125M config for Muon
│   ├── lm125m_soap.json        # 125M config for SOAP
│   ├── lm125m_sophia_g.json    # 125M config for Sophia-G
│   ├── lm125m_tomography.json  # 125M config for Tomography
│   └── smoke.json              # Small model config for fast CI checks
├── formal/
│   ├── CurvatureCompass.lean   # Lean 4 formal verification ledger
│   └── README.md               # Proof documentation and axiom scope
├── scripts/
│   ├── monitor.py              # Training progress and throughput monitor
│   ├── pod.py                  # TPU pod coordinator with worker locks
│   ├── pod_worker.py           # Worker execution wrapper
│   └── run_benchmarks.py       # Benchmark test runner
├── src/curvature_compass/
│   ├── lm/
│   │   ├── attention.py        # Pallas FlashAttention kernel & CPU fallback
│   │   ├── compare.py          # Multi-optimizer comparison runner
│   │   ├── competitors.py      # Pure JAX AdamW, Muon, SOAP, Sophia-G
│   │   ├── config.py           # ModelConfig (125M) and RunConfig
│   │   ├── data.py             # 32k LLaMA tokenizer & FineWeb-Edu pipeline
│   │   ├── execution.py        # SPMD device sharding and compilation
│   │   ├── model.py            # Modern LLaMA decoder (RMSNorm, RoPE, SwiGLU)
│   │   ├── optim.py            # Pure JAX Tomography optimizer
│   │   ├── sweep.py            # 600M token sweep planner and runner
│   │   └── train.py            # Training runner with bounded smoke tests
│   ├── core.py                 # Foundational curvature operators & CG
│   ├── report.py               # Statistical evaluation and figures
│   └── study.py                # Synthetic Monte Carlo simulation framework
└── tests/
    ├── test_attention.py       # FlashAttention numerical parity tests
    ├── test_baseline_training.py # Training loop and checkpoint recovery
    ├── test_competitors.py     # Pure JAX optimizer correctness tests
    ├── test_data.py            # Tokenizer batching and split salt tests
    ├── test_model.py           # 125M architecture and parameter count tests
    ├── test_sweep.py           # Sweep plan validation and selection tests
    └── test_train.py           # Bounded smoke and training execution tests
```
