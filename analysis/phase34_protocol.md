# Phase 3 comparison and Phase 4 sweep implementation protocol

Implementation and bounded smoke checks only. The user will execute the 60M
comparisons and 600M sweeps. No completed comparison, optimizer winner, tuned
setting, or 600M performance measurement is asserted by this delivery.
Historical `phase23` evidence remains the 60M preparation record.

## Phase 3 screening

Keep the frozen 60,957,184-parameter A/B/C/D configs, independent initialization
seeds, common data ordering, 990,810,689 exact predicted tokens, and held-out
validation unchanged. Run one model per cell. The comparison command requires
all four complete equal-token runs and checks their configs, sources, input
hashes, logs, terminal evaluations, and all worker summaries. A numerical
failure disqualifies a run from the stable selection set even after recovery.

Select the stable cell with lowest final validation cross entropy. Among
cells less than .01 nat above that minimum, choose the lowest measured elapsed
training time (then loss, then cell name). Publish every cell's loss,
perplexity, HVP counts, optimizer/gradient/evaluation/checkpoint times, memory,
throughput, and the loss/time Pareto frontier. Report interpolation at a common
elapsed time and sampled validation threshold crossings at 8, 7, 6, 5, and 4
nats as estimates, not measurements at unobserved steps. Report the 2x2 count,
alignment, and interaction contrasts with the initialization-seed confounding
explicit. Acquisition counts, residual paths, clipping and stale ages provide
mechanism diagnostics. This is exploratory screening; it is not a statistical
or certified-Newton claim. A frozen selection hashes the inputs to the report.

## Phase 4 controls

Use one tied-embedding decoder: width 1280, 27 layers, 20 heads of width 64,
MLP width 5120, context 1024, GPT-2 vocabulary padded to 50,304. Physical
parameter count: **596,993,280**. All five optimizers share architecture, exact
training/validation stream, global batch 64, BF16 training products, float32
parameters/state/curvature, highest precision optimizer algebra, reference
attention, no rematerialization, and the same decay eligibility (matrices).
No custom kernels are introduced. All measured execution costs are retained.
The common designated curvature batch is 16 sequences with full context 1024;
shortening Sophia-G contexts would leave later position embeddings without
GGN measurements. The fixed 60M Phase 3 curvature prefix remains its registered
16 × 256. The new 600M prior is calibrated with full contexts as well.

Each trial processes exactly **2,500 batches**, or **163,840,000 predicted
tokens**. Every setting receives initialization/data-order seed pairs
(41001,42001), (41002,42002), (41003,42003), matched across optimizers and
settings. No tokens repeat within a trial. Validation uses the same frozen
1,048,576 held-out tokens, never training or calibration data. No test outcome
participates in selection. Future Phase 5 requires separately registered
evaluation and scale-transfer checks; this sweep does not certify transfer.

The plan fixes 24 settings per optimizer by default, each with three seeds:
360 trials total. The defaults are one candidate; the other settings use a
deterministic stratified search with plan seed 43000. Every optimizer receives
the same number of evaluated settings/seeds and full 2,500-step budgets.
Hyperparameter interactions are varied jointly, not one factor at a time.
The complete plan, including controls, source hashes, search spaces, selection
rule, seeds, and estimated total tokens/steps, is written before execution.
Increasing settings is an explicit new plan. Actual time and unsuccessful
attempts are recorded; equal trial counts do not imply equal tuning compute.

Search spaces (logarithmic for rates, damping, epsilon and rho; categorical
otherwise):

* Tomography: LR .005–.15, damping .1–10, tolerance .3/.5/.7/.9,
  cap 4/8, refresh 4/8/16/32, decay 0/.01/.1. The Phase 3 winning cell is
  immutable. A/C require a newly calibrated 600M frame of the maximum rank;
  the 60M prior is incompatible. B/D need no unused fixed frame. Tolerance is
  varied only for adaptive-count cells, and cap is the fixed budget for A/B.
* AdamW: LR .0001–.003, beta1 .9/.95, beta2 .95/.98/.99/.999,
  epsilon 1e-10–1e-6, decay 0/.01/.1.
* Muon: LR .003–.1, momentum .9/.95/.98, Newton–Schulz steps 5/8,
  auxiliary/main LR ratio .005–.1, auxiliary beta2 .95/.99,
  decay 0/.01/.1. Nesterov is enabled. Only hidden matrices use Muon;
  embeddings, gains and biases receive bias-corrected AdamW.
* SOAP: LR .0001–.005, beta1 .9/.95, beta2 .95/.99,
  Shampoo beta .9/.95/.99, basis refresh 5/10/20,
  maximum preconditioned dimension 1024/2048, decay 0/.01/.1.
  Larger axes are omitted, as in the reference implementation; smaller axes
  remain preconditioned. This memory control is explicitly tuned/reported.
* Sophia-G: LR .0001–.003, beta1 .9/.965/.98, beta2 .95/.99/.995,
  rho .01–.1, estimator refresh 5/10/20, decay 0/.01/.1.

Select only settings with three stable complete trials. Rank their mean final
held-out cross entropy; settings within .005 nat of the minimum are selected
by mean elapsed training time, then mean loss and setting ID. Require every
planned trial to be terminal (complete or failed) before freezing selection.
Missing/interrupted trials block selection rather than disappearing. Preserve
all seeds, standard deviation/standard error/range, failures, attempts and
quality/time frontiers. Selection artifacts contain the actual chosen configs
and hashes, not a hand-edited summary or assumed winner.

## Algorithm fidelity and memory

AdamW uses bias-corrected moments and decoupled decay. Muon's approximate
quintic Newton–Schulz transform uses the author's coefficients, momentum EMA,
Nesterov mixture and output/input aspect scaling. FP32 replaces the author's
BF16 optimizer products to meet this repository's state/algebra controls.

SOAP initializes covariance bases on its first batch and skips that update.
Subsequent gradients and Adam moments use the previous basis. After an update,
covariances incorporate that gradient; refresh uses ordered power iteration
plus QR, reorders diagonal second moments and transports first moments through
the physical space. The diagonal variance is not claimed to undergo an exact
dense covariance rotation. Decay follows the reference SOAP update ordering.

Sophia-G samples independent labels from stop-gradient model probabilities,
then differentiates their mean cross entropy in float32. With N valid global
tokens, N times the squared global sampled-label mean gradient estimates the
mean-loss GGN diagonal. It is not a squared supervised-gradient substitute.
The state stores the normalized diagonal, so the reference `rho * bs * h`
denominator becomes `rho * normalized_h`. Momentum and diagonal EMAs have no
bias correction, matching the reference. The update is coordinate clipped to
[-1,1]. Sampling keys derive from initialization seed and global step, so
checkpoints do not lose random state. Estimator passes/tokens/time are logged.

600M tomography prior and basis columns are sharded over the data mesh;
parameters are replicated. Checkpoints store each host's addressable basis
shards with indices, avoiding a full-basis device gather. Recovery requires
the same topology and checks state shapes/dtypes plus source/config/input
hashes. Large data, prior, trial logs and checkpoints remain ignored.

The 600M tomography update applies the same matrix-only decay tensor by tensor
instead of allocating a full parameter-sized constant mask. Compilation
example buffers are released before stepping. Abstract full-shape tracing
checks dimensions without claiming an executed 600M workload.

## Validation boundaries and gates

Use independent recurrence/matrix oracles, sampled-label GGN Monte Carlo,
real-LM finite smoke updates for all five methods, checkpoint continuation,
sharded checkpoint round trips, exact budgets and rejection of invalid plans,
gates, incomplete results, and smoke-result selection. Run the repository CPU
suite and regenerate the 60M preparation gate. The 60M numerical core is kept
unchanged, so its recorded TPU derivative checks remain applicable.

A Phase 4 gate is separate. It binds the complete plan, source, actual inputs,
CPU coverage and successful bounded **full-shape** smoke/recovery evidence
for every optimizer state shape on the intended runtime. Tiny smoke checks
cannot establish 600M memory, compilation or steady-state performance. Such
full-shape gates must be produced before the user's sweep; no 60M gate is
silently promoted to 600M validation. Full execution always requires explicit
`--full-run --gate`; default workflow commands generate/report artifacts only.

## Primary references inspected for the ports

* [Muon author implementation](https://github.com/KellerJordan/Muon/blob/f98f1cacc0263b04290753e32be8d498c1efc806/muon.py)
  and [author explanation](https://kellerjordan.github.io/posts/muon/).
* [SOAP author implementation](https://github.com/nikhilvyas/SOAP/blob/a1e553530fde97d0e6b307d7c82ac6d38b072340/soap.py)
  and [paper](https://arxiv.org/abs/2409.11321).
* [Sophia-G author implementation](https://github.com/Liuhong99/Sophia/blob/a7e157229b71d58cf995d32854f1be15c265b350/sophia.py)
  and [paper](https://arxiv.org/abs/2305.14342).
* [AdamW paper](https://arxiv.org/abs/1711.05101).

These are algorithmic JAX reimplementations; references describe the exact
variant being compared, not a claim of byte-for-byte framework equivalence.
