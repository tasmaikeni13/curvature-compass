# Repository instructions

## Current scope

Phase 1 research is complete. The approximately 60M A/B/C/D language-model
implementations and TPU execution comparison are prepared. Full training is
the user's next step; do not launch it as an incidental setup or validation
task. `prompt.md` now has five phases: preparation includes execution checks,
and the full 60M comparison is Phase 3. Historical `phase23` artifacts refer
to the original preparation and acceleration phases.

Phase 3 result screening and Phase 4 five-optimizer 600M sweep code are prepared
under `src/curvature_compass/lm/`, with `configs/lm600m_*.json` and the registered
`analysis/phase34_protocol.md`. Sweeps use 2,500 steps and three matched seeds
per setting. Small smoke outcomes cannot authorize a full 600M sweep; its
separate gate consumes actual full-shape smoke/recovery evidence. The user
owns the comparison, sweep, and main training runs.

## Implementation

- Language-model code: `src/curvature_compass/lm/`; controlled configs:
  `configs/lm60m_A.json` through `lm60m_D.json`.
- Ship standard JAX/XLA with BF16 training, float32 parameters/state/curvature,
  highest precision for curvature and reduced algebra, reused linearization,
  reference attention and no rematerialization.
- Custom kernel prototypes were deleted: algebra gained only about 1.5%,
  and Splash failed compilation. Preserve measurements and failed logs;
  do not reintroduce them without a requested, meaningful measured benefit.
- Keep all four experiments controlled except optimizer cell and independent
  initialization seed. Preserve exact token accounting and held-out splits.
- Never report the stochastic clipped residual as a certified Newton bound
  or infer an optimizer winner from stationary performance measurements.

## Validation and operation

For numerical or training changes, run:

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 JAX_PLATFORMS=cpu .venv/bin/python -m pytest -q --junitxml=analysis/results/phase23/cpu_tests.xml
```

Run `JAX_PLATFORMS=cpu .venv/bin/python analysis/phase23_audit.py` to regenerate
the preparation gate after appropriate evidence exists. The README documents
pod launch, monitoring, input preparation and recovery commands.

This workspace is worker 0 of a Google v4-32 pod with SSH aliases `w0`–`w3`.
Use `scripts/pod.py` for coordinated jobs; respect its cooperative worker lock
and device-owner checks. Do not interrupt other repositories' workloads.
Full training requires explicit `--full-run --gate ...`; bounded smoke checks
use `--smoke-steps`. Gate/checkpoints bind source, controls, inputs and evidence.

Large corpus, PCA frames, traces and checkpoints belong in ignored `data/`,
`artifacts/` and `runs/`, not Git. Commit small manifests, raw timing records,
failed checks and clear reports. Distinguish measured timings, calculated
stage sums, and estimates. See `analysis/phase23_results.md` and the decision
ledger for actual results and limits.
