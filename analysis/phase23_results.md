# 60M implementation and XLA decision

Delivered the preparation requested in the original phases 2/3, now combined
under Phase 2 in `prompt.md`. No full 60M training run or 1B-token comparison
was executed. The user will run those comparisons.

The decoder has **60,957,184 parameters**: 11 layers, width 512, eight heads,
MLP width 2048, tied embeddings and context 1024. A/B/C/D share model, data,
loss, safeguards and schedule; each has an independent initialization seed.
A/C use a frozen independent gradient-PCA frame; B/D acquire residual-Krylov
directions. A/B use eight HVPs per refresh, except numerical exhaustion;
C/D truly stop the compiled acquisition loop on the same residual criterion.
All cells use the same damped inverse with a full-space gradient complement.
This stochastic clipped residual is a heuristic, not a Phase 1 certificate.

The pinned FineWeb-Edu scan prepared **990,810,689 predicted training tokens**,
1,048,576 validation tokens and 524,288 calibration tokens. Exact document
hashes isolate the splits. Training visits deterministic sequence blocks
without replacement, including the masked partial block. A full-size prior
was prepared independently from 16 calibration gradients. Large data, prior
and trace files remain in ignored local directories on the pod; manifests
and their hashes are committed.

## Measured execution costs

Actual Google TPU v4-32: four workers, 16 JAX devices. Stationary benchmarks
use the full model, batch 64 × 1024, curvature prefix 16 × 256 and eight B HVPs.
Each row has three completed steady-state repetitions. Updated parameters
are discarded; these are performance checks, not training trajectories.

| Candidate | Refresh seconds | Cached stage sum seconds | Decision |
| --- | ---: | ---: | --- |
| Float32 standard XLA | 0.7549 | 0.10647 | Reference |
| Reduction/incremental algebra | 0.7434 | 0.10685 | Only ~1.5% refresh gain; implementation deleted |
| BF16 training, FP32 curvature | 0.7091 | 0.06087 | Selected standard XLA |
| BF16 XLA SDPA | 0.7080 | 0.05988 | Small incremental gain over BF16; use reference attention |
| Rematerialization | 0.7370 | 0.06786 | Slower; memory already fits |
| Recomputed JVPs | 0.7397 | 0.05975 | Acquisition slower; reuse linearization |

BF16 reduces refresh time by approximately **6.1%** and the cached stage sum
by **42.8%** against float32. Full-size relative gradient/direction errors
are 0.00550/0.00522, within the declared 0.03/0.05 limits. Curvature, parameters,
optimizer state and reduced algebra remain float32; curvature and inverse
products request highest precision. No custom kernels remain in the code.

Splash attention failed compilation first because this runtime requires a
128-aligned value head, then because Mosaic kernels could not be automatically
partitioned across the pod mesh. Its performance was not established. Both
attempts and the rejected algebra measurements remain recorded; their
implementations and switches were deleted as requested.

The cached column sums separately timed gradient and update stages. The
calculated refresh-interval amortization is an estimate, not measured full-run
throughput. Benchmark JSON retains the original measured source hashes;
final source differs through deletion of unused prototypes and final reporting.
Compilation and warmup are excluded from these medians. XLA estimates about
8.19 GB for acquisition; observed reference device peak was 8.20 GB. The raw
TPU trace remains at `artifacts/profile_reference/` locally. See
[raw/aggregated costs](results/phase23/acceleration.csv),
[figure](results/phase23/acceleration.png),
[execution selection](results/phase23/acceleration_selection.json) and
[decision/failure ledger](phase23_decisions.md).

## Verification and launch

The final CPU suite passed **31 tests**, including all four small-model
implementations, independent projected-inverse oracles, real model HVPs,
exact token accounting, bitwise resumed training and atomic checkpoint
rollback. The cleaned implementation passed **21 small-model checks on each
of all four TPU workers**, including CPU/TPU derivatives, finite differences,
BF16 comparison, true prefix stopping and bitwise device checkpoint continuation.
The preparation audit binds those checks, benchmark evidence, source,
configurations, data and prior in `results/phase23/gate.json`.

No full-size multi-step training, full-size checkpoint/validation throughput,
long-run stability or optimizer advantage is claimed. Defaults are controlled
starting settings, not tuned hyperparameters. The user's runs will establish
those outcomes. Full training requires the explicit `--full-run --gate ...`
options. See [README](../README.md) for launch and monitoring commands.
