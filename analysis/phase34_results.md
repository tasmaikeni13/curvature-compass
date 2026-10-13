# Phase 3/4 implementation delivery

Implemented the Phase 3 exploratory screening workflow and Phase 4 five-method
600M sweep infrastructure. **No 60M comparison, 2,500-step sweep trial, or
125M main run was launched.** The user owns those experiments. There is no
optimizer winner or tuned setting in this delivery.

## Implementations

The frozen A/B/C/D 60M controls and numerical core remain intact. The common
runner now supports bias-corrected AdamW, hidden-matrix Muon with auxiliary
AdamW, SOAP with projected moments and covariance/QR refresh, and sampled-label
Sophia-G with stateless random keys and correctly normalized GGN estimates.
All use standard JAX/XLA, BF16 training, float32 parameters/state/curvature,
highest precision reduced/optimizer algebra, reference attention, and no
rematerialization or custom kernels.

The five 600M templates share a **596,993,280-parameter** decoder. Registered
trials have **2,500 steps**, **163,840,000 exact predicted training tokens**,
and **three matched initialization/data-order seed pairs per setting**.
The default deterministic joint search has 24 settings per method, including
defaults, for 360 trials. The generator requires and revalidates actual Phase 3
evidence; the D tomography template is a placeholder rather than a selected
winner. Reports keep all seeds, failed trials, attempts, tuning costs,
variability and quality/time frontiers.

600M tomography prior/basis columns are sharded over the data mesh, with local
checkpoint shards and compatible recovery. Its update applies matrix-only
decay by tensor, avoiding a 596,993,280-element mask (2,387,973,120 bytes if
materialized as float32; this is a calculated allocation size, not a measured
speed or peak-memory gain). Compilation example buffers are released before
stepping. The common 600M curvature batch uses full 1024-token contexts;
Sophia-G rejects shortened sweep contexts, and a numerical test checks that
all position embeddings receive GGN estimates.

## Executed checks

The required CPU command passed **53 tests**:

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 JAX_PLATFORMS=cpu .venv/bin/python -m pytest -q --junitxml=analysis/results/phase23/cpu_tests.xml
JAX_PLATFORMS=cpu .venv/bin/python analysis/phase23_audit.py
```

Coverage includes independent optimizer recurrences and matrix oracles,
sampled-label GGN expectation/normalization, SOAP moment transport, real-LM
HVPs, all A/B/C/D cells, BF16 baseline updates, exact partial token budgets,
bitwise continuation for all baseline states, atomic checkpoint rollback,
an actual two-device CPU sharded checkpoint round trip, and rejection of
invalid controls, changed gates, incomplete results and smoke-result selection.
The 600M mask-free update is abstractly traced without allocating its full
parameters. This proves traced shapes and removal of that allocation; it does
not prove 600M execution performance or memory fit.

An actual **v4-32 pod** check ran on all four workers / 16 JAX devices, using
the real pinned prepared FineWeb data. Each optimizer used the same tiny
decoder: width 16, one layer, two heads, MLP width 32, context eight, and
**807,248 parameters**. Global batch 16 processed **128 predicted tokens per
step**. Curvature/GGN used the full eight-token context. For every method, the
check compared an uninterrupted three-step run with one step plus a resumed
continuation to step three: **384 predicted tokens per trajectory**.

| Optimizer | Finite updates/held-out validation | Bitwise checkpoint continuation | Workers passed |
| --- | --- | --- | --- |
| Tomography D | Passed, real HVP refreshes and cached update | Passed, including sharded basis | 4/4 |
| AdamW | Passed | Passed | 4/4 |
| Muon + auxiliary AdamW | Passed | Passed | 4/4 |
| SOAP | Passed, initialization skip and QR refreshes | Passed | 4/4 |
| Sophia-G | Passed, actual sampled-label GGN refreshes | Passed | 4/4 |

Final evidence is in `results/phase34/tpu_smoke-worker-0.json` through
`tpu_smoke-worker-3.json` and `tpu_smoke_final_logs/`. Every record preserves
its actual numerical source hashes, runtime, shapes, compilation timings,
XLA memory estimates and checkpoint hashes. The earlier bounded checks are
retained as `tpu_smoke_first-*`, `tpu_smoke_mask_free-*` and their logs.
The small losses/timings are infrastructure checks, not optimizer comparisons.

## Corrections retained during validation

The first checkpoint regression check exposed a NumPy-state compatibility
error, corrected before the complete passing suites. A development GGN test
passed its independent expectation check but failed an incorrect contrast:
one fixed-label squared gradient happened to nearly equal the GGN in that
particular logistic example. The counterexample now uses all-zero supervised
labels. The failed development XML is preserved in
`results/phase34/baseline_development.xml`.

The pod showed that JAX process indices differ from SSH alias order: w1/w2/w3
wrote worker 2/3/1 artifacts. Collection now reads `summary-worker-*.json` from
every host. A regression test simulates the observed ordering. No worker
workload was interrupted; all coordinated jobs acquired the cooperative lock
and passed device-owner checks before starting.

## Remaining experimental boundaries

The refreshed `results/phase23/gate.json` covers the original 60M preparation,
using the unchanged numerical core, preserved historical TPU derivative and
full-shape timing evidence, and current CPU suite/source. It does not authorize
a 600M sweep. The separate Phase 4 gate requires actual bounded **600M-shape**
smoke/recovery evidence on the intended runtime for every state shape in the
registered plan. No such full-shape gate was manufactured here.

Actual 600M compilation, memory fit, throughput and longer stability remain
unmeasured. The user will establish them before running the sweep. Hyperparameter
quality, a Phase 3 winner, three-seed sweep variability and transfer to the
future 125M study remain unestablished. Stochastic clipped residuals are never
reported as certified Newton bounds.

See [the protocol](phase34_protocol.md) and [the launch/recovery guide](phase34_launch.md).
The delivery audit verifies current source hashes, frozen 60M controls,
all 600M templates, required CPU coverage, all four workers' final evidence,
the preparation gate and the absence of full runs:

```sh
JAX_PLATFORMS=cpu .venv/bin/python analysis/phase34_audit.py
```

Its coverage and evidence hashes are in `results/phase34/delivery_audit.json`.
