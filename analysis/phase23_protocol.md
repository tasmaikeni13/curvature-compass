# Registered Phase 2/3 design

Registered before language-model correctness or performance outcomes. Scope:
prepare and validate phases 2 and 3; do not execute any Phase 4 1B-token run.
The user identified an existing v4-32 TPU slice. Worker 0 is this workspace;
the four existing workers are accessible through SSH. An unrelated live
`rocora` pilot was observed on all workers; do not interrupt it or claim access
to an idle TPU until its processes have exited.

## Controls and scientific transfer

Use one decoder architecture: 11 pre-normalized transformer blocks, width 512,
8 heads, MLP width 2048, learned 1024-position embeddings, tied input/output
embeddings, GELU, LayerNorm, no dropout. GPT-2's frozen 50,257-token byte BPE
is padded to 50,304 embedding rows for TPU alignment; padded output classes are
masked. Count all physical parameters. Four independent initialization seeds
31001–31004, common data-order seed 32000, and a separate prior seed 30900.
This one-seed design does not identify optimizer effects independently of seed.

Freeze dataset revision `66b78cd134ab7b0545a1a618a1cc029f3091eb20` from
`codelion/fineweb-edu-1B` and tokenizer revision
`607a30d783dfa663caf39e06633721c8d4cfcd7e` from `openai-community/gpt2`.
Split by SHA-256 of exact text plus a fixed salt: 0.5% validation, 0.5%
calibration, 99% training. Identical documents therefore cannot cross splits;
deduplicate exact text within the corpus. Preserve ordered Parquet provenance,
tokenizer bytes, EOS separators, token counts, and hashes. Pack continuous
tokens without padding except the last batch, with next-token labels. Context
may cross an EOS document boundary. Use a deterministic permutation of sequence
blocks, without replacement. Count actual predicted tokens, including EOS,
not examples or tokens processed repeatedly by curvature differentiation.

Target 1,000,000,000 predicted training tokens per configuration. If this
tokenizer yields fewer after holding out documents, report the actual count
and require an explicit revised budget rather than silently repeating data.
Global batch 64 sequences of 1024 tokens (65,536 predicted tokens), fixed
cosine schedule with 200 warmup steps, learning rate .05, minimum ratio .1,
gradient norm cap 1, decoupled weight decay .1. These are preparation defaults,
not tuned or scientifically established optimal hyperparameters. Validation
uses one frozen 1,048,576-token subset, every 100 steps and at completion.
Checkpoints every 250 steps and on interruption, with two retained generations.

Phase 1's PCA prior in dimension 16–96 cannot be reused in a 60M-parameter
space. Fit a new frozen rank-eight gradient PCA frame from 16 calibration-only
minibatches at an independently initialized, frozen model. The frame, tensor
layout and data provenance are immutable across A/C and training steps. This
transfer assumption must be tested by Phase 4, not asserted here.

All four cells measure true vector Hessian products of the same designated
training minibatch prefix at each refresh. Refresh every 16 optimizer steps,
using a common global curvature batch of 16 sequences of length 256. A/B query
eight directions, except numerical rank exhaustion/zero gradient; C/D stop
after at least one query when the measured projected damped residual norm is
at most .7 times the full training gradient norm. A/C directions come in the
immutable PCA order; B/D use the current residual, twice reorthogonalized.
Cache the frame and reduced inverse between refreshes. Log stale-step age.
Adaptation occurs on refreshes, not through an unreported change in schedule.

For indefinite neural-network curvature, symmetrize `QᵀHQ`, add common damping
λ=1, and clip reduced eigenvalues below .1λ. Phase 1's SPD residual theorem
does not apply to this noisy, clipped operator. The .7 rule is a heuristic,
never a certified Newton error bound. Supply the common full-space complement

`s = −g/λ + Q[λ⁻¹ I − (sym(QᵀHQ)+λI)_clipped⁻¹]Qᵀg`.

Pure rank-eight projection would restrict fixed cells to eight directions for
the entire run. The complement permits all cells to update all parameters and
defines the same approximate inverse model. It is an explicit neural-network
adaptation of the Phase 1 framework, not an unnoticed equivalence. Cap the step
norm at ten times the gradient norm and reject nonfinite/non-descent steps.
Do not reconstruct the dense Hessian. Preserve clipped modes and breakdowns.

## Implementation and acceptance gates

JAX/XLA, float32 parameters and optimizer state; investigate bf16 matrix
products for training, float32/highest precision for curvature. Replicate this
small model and rank-eight frame, shard minibatches over all devices of all
four hosts using a named data mesh. Initialize distributed JAX before device
discovery. Keep directional acquisition and count stopping in a compiled
device loop, without a host round trip per HVP. Compare reused linearization
with recomputed JVPs, compiled attention with an explicit reference, and
memory rematerialization with ordinary execution. Consider specialized TPU
attention only where its differentiation and end-to-end results justify it.

Numerical gates: quadratic oracle vs an independent dense solve; nested
prefix and count invariants; finite-difference HVPs; causal masking; reference
vs optimized loss/gradient/HVP; positive/descent clipped preconditioner;
full vs accumulated gradient; exact token budget; validation isolation;
bitwise checkpoint continuation on the same runtime; atomic/integrity-checked
checkpoint recovery; config and artifact binding in the Phase 4 gate.
Float32 relative gradient/HVP tolerance 1e-4 where exact comparisons are
well-conditioned, finite differences 2e-2, bf16 training gradient relative
error at most .03 and update relative error at most .05. Curvature remains
float32; any looser comparison must be explained and preserved as a failure.

Warm up compiled executables, separate compile time from steady-state time,
block on actual device completion, record at least three timing repetitions
for important candidates, and preserve raw observations. Profile an actual
60M-model workload, including eight HVPs, a normal cached step, validation and
checkpoint I/O. Report exact tested shapes and CPU/TPU topology; no extrapolated
throughput may be reported as measured full-run throughput. Report memory from
compiled executable analysis/device metrics with method and limitations.
Select accelerations by correctness plus a >=5% median end-to-end gain, or a
measured memory reduction with a justified time cost. Keep losing candidates.
Stop after the above bottlenecks have been measured and the fair common
execution path passes; do not launch an unlimited kernel search.

Full-run launch requires a machine-readable correctness/performance gate bound
to the actual source, common config, data manifest, prior, environment and
device topology. Smoke mode has a hard small-step bound and records itself as
an infrastructure check. No 1B-token training is invoked by setup, calibration,
benchmarking, testing or the default CLI.

For later Phase 4, predeclare the screening criterion: minimum final validation
cross entropy among stable completed equal-token runs; report the complete
quality/elapsed-time frontier and equal-time interpolated validation loss.
If losses differ by <.01 nat, prefer the faster run. Treat a single seed as
exploratory; do not report a statistical optimizer winner from these checks.

## Primary implementation references

* [Dataset card](https://huggingface.co/datasets/codelion/fineweb-edu-1B)
* [TPU v4 architecture/topology](https://docs.cloud.google.com/tpu/docs/v4)
* [Multi-host JAX on TPU](https://docs.cloud.google.com/tpu/docs/jax-pods)
* [JAX linearization](https://docs.jax.dev/en/latest/_autosummary/jax.linearize.html)
* [Global arrays from process-local data](https://docs.jax.dev/en/latest/_autosummary/jax.make_array_from_process_local_data.html)

Performance measurements and implementation discoveries will be recorded
separately. Do not rewrite this protocol after observing outcomes.

## Scope amendment requested by the user

The standalone kernel phase was removed from `prompt.md`; old phases 4/5/6
are now 3/4/5. Original Phase 2/3 preparation and acceleration belong to current
Phase 2. The original numbered references above remain historical.

The user clarified that only the 60M implementations and a measured kernel
comparison are requested; the user will run full training personally. Do not
run full-size training smokes or expand the kernel investigation. Small-model
A/B/C/D integration and bitwise recovery tests plus actual full-shape stationary
benchmarks are the delivery checks. Full-size validation/checkpoint I/O and
long-run optimizer behavior remain to be measured in the user's training runs.
Candidates with only 1–3% gains must be deleted; preserve their recorded outcomes.
