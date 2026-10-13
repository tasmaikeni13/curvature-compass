# Phase 2/3 decision and failure ledger

1. The current host is worker 0 of the user's existing TPU v4-32 slice. SSH
   access, the Cloud TPU READY state and the four worker endpoints were verified.
   Another repository's pilot was running at first inspection. Its jobs are
   protected by a cooperative worker lock; use that same lock to avoid racing
   for the TPU. Never kill that workload to obtain benchmark resources.
2. The Python 3.10-compatible JAX runtime is 0.6.2 with jaxlib 0.6.2 and libtpu
   0.0.17. An isolated project environment is used for new dependencies.
3. A full pinned-source preprocessing scan read 970,397 documents, with no
   exact-text duplicates. GPT-2 BPE yielded 990,810,690 training-stream tokens
   after the registered split, plus 5,023,388 validation and 5,136,464 calibration
   tokens before subset truncation. Store 990,810,689 predicted training tokens,
   1,048,576 validation and 524,288 calibration tokens. Freeze **990,810,689** as
   the common training budget in all four configurations: 0.991 billion is
   approximately the requested billion. Do not repeat tokens or move held-out
   documents into training to reach a cosmetic exact billion. This amendment
   follows preprocessing evidence and precedes model outcome evaluation. The
   complete data manifest is `results/phase23/data_manifest.json`.
4. The architecture has 60,957,184 physical parameters, including vocabulary
   padding, and tied input/output weights. No pretrained weights are used.

Append actual correctness, profiling, and acceleration outcomes below. Retain
failed checks and rejected candidates with the explanation that resolved them.

5. The first CPU suite passed 19 checks and failed five. Four failures were a
   test assumption: a shuffled short final sequence can occur early, so two
   batches need not have exactly the nominal token count. The separate exact
   final-budget and bitwise recovery test passed. Test complete blocks for the
   cached-update check and retain the dedicated partial-block test.
6. The fifth failure exposed a genuine numerical acquisition issue: float32
   roundoff after solving an identity operator produced a spurious second
   Krylov direction. Increase the numerical rank threshold from 1e-9 to 1e-6
   times max(gradient norm, 1), while keeping policy stops distinct from
   numerical exhaustion. Preserve the failed JUnit artifact.
7. The first whole-pod probe failed to open worker 0's TPU because a separate
   `algebraic-vision` screen held all four devices, despite not taking the
   existing cooperative lock. Add an authoritative device-owner check before
   releasing workers; cancel and reap only this launcher's probe processes.
   This failure is resource contention, not a numerical validation result.

8. All 24 CPU tests then passed, including bitwise same-runtime checkpoint
   continuation, exact partial-token budgets, all four cells' real model HVPs,
   derivative equivalence, indefinite safeguards, and hash-checked rollback.
9. CPU calibration fitted rank eight from 16 independent frozen-model gradient
   snapshots at the full 60,957,184-parameter architecture. It used only the
   calibration split, seed 30900, batch 16 × 128, 16 CPU cores, and 129.31s.
   Maximum frame Gram error was 1.82e-10. The immutable frame is approximately
   1.95GB; its hash and source provenance are in the prior manifest. It is a
   preparation artifact, not a trained model or evidence of transfer quality.
10. The first actual four-host numerical run found 16 JAX TPU v4 devices, four
    per host. Reference loss/gradient/HVP agreed with CPU (gradient relative
    error 3.85e-7; HVP 4.17e-7), and device checkpoint continuation was bitwise.
    Float32 SDPA exceeded the 1e-4 derivative tolerance (gradient 4.83e-4,
    HVP 8.24e-4). Two adaptive quadratic checks also failed because default
    TPU matrix precision affected the test oracle and small inverse products.
    Make all reduced inverse/coefficient products and descent dot products
    explicitly HIGHEST precision; force float32 precision in the float32 SDPA
    path and independent quadratic test oracle. Do not relax tolerances.
    See `tpu_numerics_first_attempt.json` and
    [JAX precision control](https://docs.jax.dev/en/latest/_autosummary/jax.default_matmul_precision.html).
11. With explicit precision, all 21 whole-pod numerical checks passed on all
    four hosts. Float32 SDPA derivatives matched reference exactly in the toy
    workload; the largest quadratic inverse error was 2.54e-7. Mixed bf16 toy
    gradient error was .00505, below the .03 preregistered limit. This does not
    approve bf16 at full model size; actual 60M comparisons remain mandatory.
12. The full-size reference benchmark measured the actual 64 × 1024 gradient
    and 16 × 256 curvature workload, with all eight B queries. Three repetitions
    gave medians .0991s gradient, .6483s acquisition, .00735s cached update, and
    .7549s total refresh. Curvature was about 86% of that refresh. XLA executable
    memory estimates were 5.27GB gradient and 8.19GB acquisition; observed device
    peak was 8.20GB. Compilation (39.6/67.1/8.12s) is separate. The captured TPU
    trace includes finite reductions, basis writes, and all-reduces among its
    leading costs. Python TensorFlow trace hooks were unavailable, but TPU
    device events and an XPlane/JSON trace were produced successfully.
13. Investigate equivalent native XLA rank-eight algebra using fused fp32
    reductions and incremental Gram row/column/RHS updates. This preserves
    measured directions, the projected inverse, and stop policy; it is an
    implementation candidate, subject to dense/quadratic and full-size checks.
    Twelve CPU acquisition checks passed for both matmul and reduction paths.
14. The user requested specialized kernels only if measured speed improves;
    otherwise use XLA. The user also requested removal of the standalone
    acceleration phase from prompt.md and renumbering. The old Phase 3 has been
    removed, old 4/5/6 are now 3/4/5, and profiling/validation remains in Phase 2.
    The current task still completes the originally authorized preparation and
    validation work; it does not authorize starting the newly numbered Phase 3
    full comparison. Earlier registered files retain historical phase numbers.
    The CLI now uses --full-run rather than a phase-number-specific switch.
15. Incremental/reduction algebra passed the full-size direction comparison
    (relative error 2.17e-5), but refresh time improved only about 1.5% (.7434s
    versus .7549s) and cached time slightly worsened (.10685s versus .10647s).
    Reject it under the registered 5% speed criterion. Preserve its measured
    outcome for reproducibility; final controlled runs will use standard
    XLA matmul algebra unless later actual evidence justifies a different path.
    Do not treat a numerically correct custom implementation as an improvement.
16. BF16 training with float32 curvature passed the full-size comparison
    (gradient relative error .00550, direction .00522), with .7091s refresh
    versus .7549s fp32 and .0609s versus .1065s cached stage sum. This is an
    accepted model-execution improvement. Float32 optimizer state and explicit
    HIGHEST curvature/reduced algebra precision remain unchanged.
17. Standard XLA SDPA was effectively the same as native reference attention
    under bf16: .7080s refresh and .0599s cached stage sum. It does not separately
    meet the 5% criterion against bf16 reference attention. Rematerialization
    slowed refresh/cached costs to .7370/.0679s; current memory already fits.
    Recomputed JVPs slowed acquisition to .6799s versus .6481s for reused
    linearization. Keep reused linearization and rematerialization disabled.
18. The first Splash attempt could not compile because this pinned runtime
    requires a value head dimension divisible by 128; the controlled model has
    64-wide heads. Preserve its failure logs. Test zero-padding Q/K/V to 128
    and cropping the output, with scaling based on the original 64, so the
    architecture and attention function remain the same. Include padding costs.
19. The benchmark tool now supports direct consecutive cached-step timings.
    The recorded runs predate that addition: their cached values are sums of
    separately timed stages, not measured consecutive training steps.
20. Zero-padded Splash also failed compilation: this JAX runtime cannot
    automatically partition Mosaic kernels across the named pod mesh. No
    specialized attention speedup was established. The user requested prompt
    completion and deletion of candidates with only 1–3% gains. Delete all
    Splash and incremental/reduction implementation code, configuration
    switches and prototype tests. Preserve raw measurements and failure logs.
21. Final scope follows the user's clarification: deliver implementations and
    the kernel decision; the user will execute full 60M training personally.
    Do not require four full-size training smokes or full-size recovery trials.
    The small-model tests cover A/B/C/D and exact bitwise recovery, and the
    stationary full-size benchmarks establish the execution choice. Final
    CPU and small-model whole-pod checks cover the cleaned standard XLA code.
    Training configs use bf16/reference attention with float32 curvature,
    highest-precision reduced algebra, reused linearization, and no remat.
