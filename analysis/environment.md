# Reproduction environment

Recorded on 2026-10-12 from the research host used for the checked-in runs.

| Component | Version/detail |
|---|---|
| OS/kernel | Linux x86_64, kernel 5.19.0-1022-gcp, glibc 2.35 |
| Python | 3.10.12 |
| NumPy / SciPy / Matplotlib | 2.2.6 / 1.15.3 / 3.10.7 |
| Lean / Mathlib | Lean 4.34.1 / Mathlib v4.34.1 |
| Host CPUs | 240 logical CPUs reported; process pool used 8 workers for primary studies |
| BLAS threads | 1 for recorded numerical runs (`OPENBLAS_NUM_THREADS=1`, `OMP_NUM_THREADS=1`) |
| Accelerator | No GPU was used; the studies are CPU-only |

Python direct dependencies are pinned in `pyproject.toml`; the resolved package
snapshot is in `requirements-lock.txt`. Lean dependencies are locked by
`formal/lake-manifest.json`. The primary and follow-up manifests record run
seeds, counts, thread count, raw-data SHA-256, and relevant source hashes. The
local literature cache contains the downloaded PDFs/text and is ignored by
version control; the checked-in retrieval record stores their hashes and
section-level reading log elsewhere in the review.
