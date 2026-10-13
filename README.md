# Curvature Compass

Research on directional curvature acquisition, with a completed Phase 1
foundation and prepared A/B/C/D language-model implementations. The common
decoder has 60,957,184 parameters and a frozen 990,810,689-token FineWeb-Edu
budget. Preparation and actual v4-32 stationary benchmarks are complete;
full training is left to the user. `prompt.md` now has five phases, with the
standalone kernel phase removed and later phases renumbered.

The current conclusions and open question are in [`theory.md`](theory.md).
The registered design is [`analysis/protocol.md`](analysis/protocol.md); primary
literature annotations and search limits are in
[`analysis/literature_review.md`](analysis/literature_review.md). Results,
including failed assumptions and independent follow-up cycles, are in
[`analysis/monte_carlo.md`](analysis/monte_carlo.md),
[`analysis/refinement_results.md`](analysis/refinement_results.md), and the
other notes under `analysis/`. Lean theorem scope and limitations are listed
in [`formal/README.md`](formal/README.md).

## 60M implementation and launch

Use **standard XLA**, BF16 training products, float32 curvature/state, reused
HVP linearization and reference attention. The reduction algebra prototype
improved refresh time only ~1.5%; it and the failed Splash attention path were
deleted. BF16 reduced measured refresh time from 0.755s to 0.709s. Read
[`analysis/phase23_results.md`](analysis/phase23_results.md) for measurements,
limits, validation and the preserved failures. No optimizer winner is inferred.

On worker 0 of the existing four-host v4-32 pod, the prepared dataset is
`data/fineweb/` and the independent rank-eight prior is
`artifacts/lm60m_prior.npy` with its adjacent JSON manifest. Both are already
copied to workers 1–3. These large artifacts are excluded from Git. Every
worker must use identical source, configs, evidence, data and prior bytes.

```sh
python -m pip install -e '.[lm,tpu,test]'
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 JAX_PLATFORMS=cpu python -m pytest -q --junitxml=analysis/results/phase23/cpu_tests.xml
JAX_PLATFORMS=cpu python analysis/phase23_audit.py
```

The audit regenerates the preparation gate after verifying the pinned data,
prior and recorded checks. Full training additionally validates that the
source, controls, inputs, evidence and runtime match that gate. Changing
experimental controls requires renewed validation. The committed
`requirements-lm-lock.txt` records the isolated LM environment.

Synchronize the finished implementation and evidence to the existing workers:

```sh
for host in w1 w2 w3; do
  rsync -azR --exclude='__pycache__' src configs scripts tests pyproject.toml \
    analysis/phase23_audit.py analysis/phase23_protocol.md \
    analysis/results/phase23 "$host:$PWD/"
done
```

When you choose to train, launch one configuration at a time, substituting
`A`, `B`, `C` or `D` consistently. This command is provided for the user; it
was **not executed** during preparation:

```sh
.venv/bin/python scripts/pod.py --timeout 86400 --logs runs/A/launcher \
  curvature_compass.lm.train --config configs/lm60m_A.json \
  --data data/fineweb --prior artifacts/lm60m_prior.npy --output runs/A \
  --full-run --gate analysis/results/phase23/gate.json
.venv/bin/python scripts/monitor.py runs/A --watch
```

Add `--resume` to the same training command to recover from the latest intact
compatible checkpoint. The launcher coordinates `w0`–`w3` via the existing
SSH aliases and refuses busy workers. Training logs token counts, real HVP
counts, gradient/acquisition/update costs, evaluations, memory and checkpoint
costs per worker. Checkpoints bind source/data/prior/config and retain two
atomic generations. SIGTERM requests a synchronized checkpoint.

For a fresh machine, recreate the prepared inputs before validation:

```sh
JAX_PLATFORMS=cpu .venv/bin/python -m curvature_compass.lm.data data/fineweb
OPENBLAS_NUM_THREADS=16 OMP_NUM_THREADS=16 JAX_PLATFORMS=cpu \
  .venv/bin/python -m curvature_compass.lm.calibrate \
  --config configs/lm60m_A.json --data data/fineweb \
  --output artifacts/lm60m_prior.npy
```

Copy the same prepared inputs to every worker. Regenerate validation evidence
and the gate for a different runtime or corpus. Small CPU fixtures and actual
TPU derivative checks are separate from full-size stationary timing; no full
60M multi-step training or full-size recovery run is claimed.

## Phase 1 reproduction

The recorded environment uses Python 3.10.12 with pinned NumPy, SciPy, and
Matplotlib in `pyproject.toml`, and Lean 4.34.1 with Mathlib v4.34.1. See
[`analysis/environment.md`](analysis/environment.md) for host details and
`requirements-lock.txt` for the Python package snapshot. The published
compressed outcomes have source and data hashes in matching manifest files.
Local literature PDFs and extracted full texts are excluded; their source
links, reading locators, retrieval hashes, and download script are retained.

From the repository root, recreate the Python environment and primary studies:

```sh
python -m venv .venv
. .venv/bin/activate
python -m pip install -e .
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m curvature_compass.checks
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m curvature_compass.study calibration --workers 8
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m curvature_compass.study evaluation --workers 8
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m curvature_compass.report
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m curvature_compass.refinements all
python analysis/audit_artifacts.py
```

`checks` verifies numerical invariants; primary calibration freezes the
candidate quartet before held-out evaluation. Refinements use new seed ranges
registered in `analysis/refinement_protocol.md`. To rebuild the proof artifact,
install Lean through elan and run `cd formal && lake build`. Regenerating study
outputs overwrites files under `analysis/results`; the checked-in records are
the outcomes analyzed in the report. The artifact audit verifies recorded raw
and source hashes, row counts, paper inventory, and Lean proof log without
rerunning any experiment; its completed coverage record is
[`analysis/completion_audit.md`](analysis/completion_audit.md).

## Reading map

* `src/curvature_compass/core.py`: operators, generators, acquisition, and CG.
* `study.py` and `report.py`: registered primary simulation and aggregation.
* `refinements.py`: separate noise, drift, nonconvex, iterative, scalar, and
  scaling follow-ups.
* `analysis/results/`: compressed raw records, manifests, summaries, and figure.
* `formal/CurvatureCompass.lean`: verified identities, bounds, and examples.
