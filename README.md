# Curvature Compass

Phase 1 research on directional curvature acquisition for optimization. This
repository studies scalar quadratic measurements and Hessian-vector products,
separates fixed/adaptive alignment from fixed/adaptive count, and compares
optimization value under explicit oracle and compute costs. Only Phase 1 of
[`prompt.md`](prompt.md) was performed; no model training or later-phase build
was started.

The current conclusions and open question are in [`theory.md`](theory.md).
The registered design is [`analysis/protocol.md`](analysis/protocol.md); primary
literature annotations and search limits are in
[`analysis/literature_review.md`](analysis/literature_review.md). Results,
including failed assumptions and independent follow-up cycles, are in
[`analysis/monte_carlo.md`](analysis/monte_carlo.md),
[`analysis/refinement_results.md`](analysis/refinement_results.md), and the
other notes under `analysis/`. Lean theorem scope and limitations are listed
in [`formal/README.md`](formal/README.md).

## Reproduction

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
