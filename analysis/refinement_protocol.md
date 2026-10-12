# Second-cycle investigations, declared after primary evaluation

The primary quartet is frozen and will not be changed. Its held-out evidence
shows that residual Krylov is strong, count reduction sacrifices one-step progress,
and C underperforms an interpolated fixed-budget mixture at its mean count.
Mean instance-wise progress/HVP and ratio of aggregate progress to aggregate HVPs
differ substantially: this is a selection-objective limitation, not a new winner.

Before the following outcomes are examined, declare these follow-ups:

1. **Reference mechanisms and repeated optimization.** On independent seeds
   3000–3031, eight SPD families at dimension 48, compare the frozen quartet,
   conventional fixed/residual CG, Nyström-PCG with rank four, exact directional
   steepest descent, and Armijo L-BFGS with memory five. Use both a 128-HVP limit
   and 256 declared work units (gradient/HVP/function query each one). L-BFGS has
   no HVP limit comparison because zero-HVP methods cannot be bounded by that
   resource alone. Log HVPs, gradients, loss queries, overhead and actual loss.
   Fixed outer streams use a predeclared cyclic schedule independent of counts;
   a frozen top-k subspace cannot be a fair iterative fixed baseline.
2. **Noise and drift.** Seeds 4000–4015, dimensions 16/48, four families, coherent
   operator perturbation and bounded fresh per-query noise, amplitudes
   0/.05/.25/1.5; gradient noise 0/.1. Compare observed-residual stopping with a
   bound including declared noise radius. The bound is conditional on known
   simulator radii, not a practical estimator of neural-network uncertainty.
   Stale-curvature tests use independent rotations and compare true and observed
   residuals. Every validation HVP is separately charged.
3. **Nonconvexity.** Seeds 5000–5031, small indefinite quartic objectives. Compare
   spectral-clipped projected steps with exact small-subspace trust-region steps
   under the same acquired directions; actual-loss Armijo acceptance is charged.
   At a stationary saddle, test a two-query random-start Lanczos exploration
   against gradient-only stagnation. This is an established negative-curvature
   mechanism, not a global nonconvex convergence guarantee.
4. **Accuracy certificate and conditioning.** Seeds 6000–6031, condition numbers
   10/1000/100000. Use an optimization-error stopping criterion
   `‖r‖²/(2μ) ≤ [ε/(1−ε)] F_Q`, with target relative Newton gap `ε=.1`.
   If met, it certifies captured Newton decrease at least `.9`; distinguish target
   achievement from cap exhaustion. Compare to relative-residual `.7` and `.1`,
   preserving caps and direction rules. This is a direct residual-bound adaptation.
5. **Scalar information objective.** Seeds 7000–7063, dimensions 4/8, exact
   Gaussian matrix-prior regression with noisy scalar observations. Compare
   a preplanned entropy design to posterior-dependent inverse-step uncertainty
   design, with fixed/adaptive count as a second factor. Include a misspecified
   prior scale. Goal-oriented design is established prior art. Its inverse-step
   Jacobian and uncertainty are linearized approximations, not certified nonlinear
   posterior intervals. Also verify scalar feature rank and polarization count.
6. **Scaling.** Matrix-free ridge-plus-low-rank operators at dimensions
   64/512/4096, 24 repetitions, caps 2/8/16. Single-thread measurements include
   stream setup, orthogonalization and projected solve. These are CPU operator
   microbenchmarks; they make no language-model throughput prediction.

Preserve all observations, failures, coverage rates, seeds and run manifests.
These investigations are exploratory mechanism studies on new seeds; neither
their design nor conclusions may replace the registered primary evaluation.
Stop after this cycle and a saturation-oriented literature revisit unless an
unresolved correctness failure requires another focused investigation.
