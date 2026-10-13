# Phase 3/4 operation

The [protocol](phase34_protocol.md) fixes controls and selection rules. The
[implementation results](phase34_results.md) state what was actually tested.
All commands below are for the user's later runs. Preparation did not launch
the 60M comparisons or any 2,500-step sweep trial.

## Phase 3

Use the 60M launch, monitoring, gate and recovery commands in [README](../README.md).
After A/B/C/D complete, generate their exploratory selection:

```sh
JAX_PLATFORMS=cpu .venv/bin/python -m curvature_compass.lm.compare \
  --runs runs/A runs/B runs/C runs/D --output runs/phase3-selection.json
```

The reader checks exact tokens, training/evaluation logs, common numerical
source and input hashes, independent seeds and all worker summaries. Copy
worker 1–3 summary JSON files to worker 0 before reporting. Smoke, incomplete,
uncontrolled and numerically failed runs cannot produce a frozen winner.
The selection keeps every cell's metrics, costs, memory, residual diagnostics,
frontier, estimated threshold times and exploratory factorial contrasts.

## Register Phase 4

```sh
JAX_PLATFORMS=cpu .venv/bin/python -m curvature_compass.lm.sweep plan \
  --winner runs/phase3-selection.json --output runs/phase4-plan
JAX_PLATFORMS=cpu .venv/bin/python -m curvature_compass.lm.sweep_gate requirements \
  --plan runs/phase4-plan/plan.json
```

The default plan contains 360 trials (24 settings × five optimizers × three
seeds), 900,000 training steps and 58,982,400,000 predicted training tokens
across independent trials. Each trial visits 163,840,000 tokens without
replacement. Trial count is an equal tuning effort control, not equal compute.
`--settings` creates a different explicit plan; generated plans are immutable.

The generator rechecks the Phase 3 selection against actual raw run artifacts.
It freezes the winning cell, all 600M controls, seed pairs, sampled joint search,
source hashes and budgets. It does not assume D won. Large trial artifacts
belong under ignored `runs/`, and prepared inputs under `data/`/`artifacts/`.

Synchronize source, configs, tests, scripts and validation evidence as in
README, then copy the generated plan configs:

```sh
for host in w1 w2 w3; do
  rsync -azR runs/phase4-plan "$host:$PWD/"
done
```

## Prior for a fixed-alignment winner

A/C require a new independent rank-eight **600M** calibration frame. The 60M
frame has a different layout and is rejected. The plan's default tomography
config has rank eight. This frozen-model calibration is separate from training:

```sh
.venv/bin/python scripts/pod.py --logs runs/phase4-prior/launcher \
  curvature_compass.lm.calibrate --config runs/phase4-plan/configs/tomography-000-0.json \
  --data data/fineweb --output artifacts/lm600m_prior.npy --sequence-length 1024
for host in w1 w2 w3; do
  rsync -azR artifacts/lm600m_prior.npy artifacts/lm600m_prior.json "$host:$PWD/"
done
```

B/D and the four baselines need no unused fixed prior. Add
`--prior artifacts/lm600m_prior.npy` to tomography smoke/sweep commands for A/C.

## Separate full-shape sweep gate

Tiny tests establish optimizer/recovery correctness. They do not establish
600M compilation, memory or throughput. `sweep_gate requirements` identifies
one trial config for each distinct optimizer state shape/NS loop variant in
the plan. For each listed `TRIAL`, run a bounded full-shape smoke and recover:

```sh
.venv/bin/python scripts/pod.py --logs runs/phase4-smokes/TRIAL/launcher-first \
  curvature_compass.lm.train --config runs/phase4-plan/configs/TRIAL.json \
  --data data/fineweb --output runs/phase4-smokes/TRIAL --smoke-steps 2
.venv/bin/python scripts/pod.py --logs runs/phase4-smokes/TRIAL/launcher-resume \
  curvature_compass.lm.train --config runs/phase4-plan/configs/TRIAL.json \
  --data data/fineweb --output runs/phase4-smokes/TRIAL --smoke-steps 4 --resume
for host in w1 w2 w3; do
  scp "$host:$PWD/runs/phase4-smokes/TRIAL/summary-worker-*.json" \
    runs/phase4-smokes/TRIAL/
done
```

Include the new prior flag for A/C. Smokes use their registered 2,500-step
schedule but stop at the explicit small bound. They never become sweep trials.
The gate requires successful update, validation, checkpoint and compatible
recovery at the actual 600M shape on the intended four-worker TPU runtime.
Compilation timings and XLA memory estimates appear in each run JSON.

Create the gate from all required smoke directories and a current passing CPU
suite. Substitute the actual directory list and include `--prior` for A/C:

```sh
JAX_PLATFORMS=cpu .venv/bin/python -m curvature_compass.lm.sweep_gate create \
  --plan runs/phase4-plan/plan.json --data data/fineweb \
  --smokes runs/phase4-smokes/TRIAL1 runs/phase4-smokes/TRIAL2 \
  --output runs/phase4-gate.json
for host in w1 w2 w3; do
  rsync -azR --exclude=checkpoints runs/phase4-smokes runs/phase4-gate.json \
    analysis/results/phase23/cpu_tests.xml "$host:$PWD/"
done
```

This copies the small gate evidence, not large checkpoint shards. Every worker
must see the same gate, plan, source, controls, data and prior bytes. Changed
inputs/controls/source/runtime require fresh evidence and registration.

## Explicit user sweep launch and reporting

```sh
JAX_PLATFORMS=cpu .venv/bin/python -m curvature_compass.lm.sweep run \
  --plan runs/phase4-plan/plan.json --data data/fineweb --output runs/phase4 \
  --full-run --gate runs/phase4-gate.json
.venv/bin/python scripts/monitor.py runs/phase4/adamw-000-0 --watch
JAX_PLATFORMS=cpu .venv/bin/python -m curvature_compass.lm.sweep report \
  --plan runs/phase4-plan/plan.json --runs runs/phase4 --output runs/phase4-progress.json
```

The full-run flag and matching gate are mandatory. Trials run sequentially
through `scripts/pod.py`, respecting cooperative locks and device ownership.
Busy workers stop the launcher before training; they remain untouched.
`--trials ID ...` limits which registered trials to run. `--resume` skips
intact completed trials and continues compatible unfinished checkpoints.
Numerical/compilation failures and every launcher attempt retain their logs
and costs; an infrastructure failure stops the batch for a later retry.

The sweep coordinator collects worker 1–3 summary files after each successful
trial. For manually launched trials, collect those small records before reporting. Add
`--freeze` to `sweep report` only when every planned trial is complete or failed.
Missing/interrupted trials block freezing. Chosen settings require three
stable seeds; the report includes each seed loss, SD/SE/range, measured elapsed
costs and Pareto frontiers. Frozen artifacts contain the actual chosen configs
and evidence hashes. Transfer of those 600M settings to the eventual 125M
study remains an independent research question, with no Phase 5 outcome used
to retroactively tune them.

TPU backend process indices can differ from SSH alias order; collect the
`summary-worker-*.json` records from each host rather than assuming `w1` writes
`summary-worker-1.json`.

## Repeat the small implementation smoke

Use a fresh output directory:

```sh
.venv/bin/python scripts/pod.py --logs runs/phase4-smoke-repeat/launcher \
  curvature_compass.lm.smoke --data data/fineweb --output runs/phase4-smoke-repeat/checks \
  --evidence analysis/results/phase34/tpu_smoke_repeat
```

This runs five tiny models for three steps, including real curvature/GGN
passes, held-out validation and bitwise uninterrupted/resumed checkpoint
comparisons. All outputs label themselves as smokes and are rejected by the
comparison/sweep selector and 600M gate.
