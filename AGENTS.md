# Repository instructions

## Current scope

The repository implements Curvature Compass, a modern JAX/XLA research
framework benchmarking second-order and adaptive optimization algorithms on
decoder-only language models. The architecture follows modern LLaMA
specifications: RMSNorm without learnable bias, Rotary Position Embeddings
(RoPE), SwiGLU feed-forward networks, 100% bias-free linear layers, and tied
canonical 32,000 LLaMA vocabulary padded to 32,768 for TPU matrix unit alignment.

The benchmark compares five optimizers:
1. AdamW
2. Muon (with auxiliary AdamW)
3. SOAP
4. Sophia-G
5. Tomography (adaptive directional curvature)

The experimental pipeline comprises:
- 600M FineWeb-Edu token budget hyperparameter sweeps per optimizer.
- 125M parameter model pipeline on 2.5B FineWeb-Edu tokens comparing the
  optimizers across 5 matched initialization and document streaming seeds.
- Fast bounded smoke tests (`--smoke-steps`) verifying parameter initialization,
  parameter count (~125M), gradient calculation, and loss descent without
  launching full training runs.

CRITICAL USER DIRECTIVE: DO NOT RUN FULL SWEEPS OR FULL 2.5B TOKEN TRAINING
RUNS. ONLY RUN BOUNDED SMOKE TESTS (`--smoke-steps`) AND UNIT TESTS. Under no
circumstances should multi-hour or multi-day training jobs be launched. Verify
implementations strictly via unit tests and fast bounded smoke tests.

## Implementation

- Language-model code: `src/curvature_compass/lm/`.
- Controlled model configurations: `configs/lm125m_adamw.json`,
  `configs/lm125m_muon.json`, `configs/lm125m_soap.json`,
  `configs/lm125m_sophia_g.json`, and `configs/lm125m_tomography.json`.
- Standard JAX/XLA execution with BF16 training activations, float32
  parameters/state/curvature, highest precision for curvature and reduced
  algebra, and reused linearization.
- High-performance Pallas FlashAttention kernel targeting TPU v4
  (`src/curvature_compass/lm/attention.py`) with automatic CPU fallback to
  reference dot-product attention (`jax.nn.dot_product_attention`).
- Pure JAX/XLA optimizers: all optimizers (`src/curvature_compass/lm/optim.py`,
  `competitors.py`) are strictly implemented in standard JAX/XLA operations.
  Strictly zero custom Pallas kernels in optimizer code.
- Keep all comparison experiments strictly controlled except the optimizer
  algorithm and initialization seed. Preserve exact token accounting and
  held-out document splits (`split_for` using salt `b"curvature-compass-v1\0"`).
- Never report a stochastic clipped residual as a certified Newton bound or
  infer an optimizer winner from stationary performance measurements alone.

## Validation and operation

For numerical, architectural, or training changes, run:

```sh
PYTHONPATH=src OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 JAX_PLATFORMS=cpu /home/tasma/venv/bin/python -m pytest -q
```

To run bounded smoke tests on the 125M architecture:

```sh
PYTHONPATH=src JAX_PLATFORMS=cpu /home/tasma/venv/bin/python -m curvature_compass.lm.train \
  --config configs/lm125m_adamw.json --smoke-steps 3
```

To generate JUnit XML test reports:

```sh
PYTHONPATH=src OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 JAX_PLATFORMS=cpu /home/tasma/venv/bin/python -m pytest -q --junitxml=analysis/results/cpu_tests.xml
```

This workspace is worker 0 of a Google v4-32 pod with SSH aliases `w0`–`w3`.
Use `scripts/pod.py` for coordinated multi-host jobs; respect its cooperative
worker lock and device-owner checks. Do not interrupt other workloads. Full
training requires explicit user command; agents must strictly use bounded
smoke checks (`--smoke-steps`).

Large corpus files, PCA frames, traces, and checkpoints belong in ignored
`data/`, `artifacts/`, and `runs/`, not Git. Commit small manifests, raw timing
records, failed checks, and clear reports. Distinguish measured timings,
calculated stage sums, and theoretical estimates.
