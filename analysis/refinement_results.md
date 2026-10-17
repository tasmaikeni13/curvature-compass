# Registered follow-up results

These studies were declared in [`refinement_protocol.md`](refinement_protocol.md)
after the held-out primary evaluation. They use independent seed blocks and do
not change the frozen primary quartet. Raw compressed records, per-run source
and raw-data hashes, and record counts are in `results/{reference,iterative,
noise,drift,nonconvex,accuracy,scalar,scaling}*`. All claims below are limited
to the specified synthetic cases.

## Reference mechanisms and repeated quadratics

The Nyström-PCG reference uses the FTU23 PSD Nyström preconditioner, with sketch
HVPs charged, and independent conventional CG. Across 32 seeds and eight
dimension-48 SPD families (256 paired instances per budget/rank), mean one-step
decrease fractions were:

| HVP budget | CG rank 0 | Nyström rank 2 | Nyström rank 4 |
|---|---:|---:|---:|
| 8 | .844 | .779 | .720 |
| 16 | .973 | .960 | .948 |

The budget counts include sketch products. These settings do not favor Nyström
at equal total products; they do not rule out benefits under repeated solves,
other spectra, or an appropriately amortized setup.

The repeated study evaluates the frozen A/B/C/D methods, conventional CG,
residual-stopped CG, steepest descent, persistent CG, persistent Nyström-PCG,
and Armijo L-BFGS(5). Each method sees the same quadratic and initial error.
At the 128-HVP budget, B, D, CG8, residual CG, persistent CG and persistent
Nyström-PCG all reached numerical zero loss in the reported means; A and C left
mean relative losses .0286 and .0321. Persistent CG and PCG used about 36 and 38
HVPs on average. They exploit a fixed quadratic across outer steps and are an
important stationary-problem boundary case. Under the 256-work-unit budget
(one gradient, HVP, or function query each), B/D and CG again reached reported
zero loss; A/C left .0189/.0222, steepest descent .0071, and L-BFGS reached
numerical zero using about 163 declared work units. Roundoff-level zero is not
a general convergence result. These results show that one-step truncation is
not the whole repeated-optimization comparison and that HVP-only budgets omit
meaningful gradient and line-search work.

## Noise, drift, and certificates

The noise study has 8,192 records. It includes coherent symmetric operator
perturbations and fresh bounded per-query errors, with gradient noise, in four
families and dimensions 16/48. For the declared known error radii and `μ=1`,
the enlarged residual certificate covered every true gap checked. The
uncorrected observed-residual certificate failed on 15.0% of all records under
observed-residual stopping and 17.5% under robust stopping; this is expected
because stopping on an uncorrected statistic does not make its certificate
valid. Failure varies sharply with noise amplitude and gradient noise, so these
pooled rates are not universal error probabilities. The robust stopping rule
often spends more queries and may fail to attain the target before its cap.
Bounded noise was used; Gaussian noise has no deterministic finite radius.

In 128 controlled stale-operator cases, angles 0/.05/.25/.75 radians, the
certificate augmented by the actual operator-drift norm covered all tested
gaps. This drift norm is available only to the simulator. It is not a cheap
online estimator and says nothing about practical stale-curvature control
without a Hessian continuity bound.

## Accuracy and nonconvex behavior

In 6,144 conditioning/regime cases, the relative Newton-gap certificate met
its 10% target in 2,963 cases (48.2%), versus 3,413 (55.6%) for residual .1 and
4,335 (70.6%) for residual .7. These thresholds are not equivalent tests: the
certificate is principled when its lower-curvature bound is valid, while a
relative residual can be optimistic under ill-conditioning. All stop rules
recorded cap exhaustion as target unmet. The limited attainment rate is an
honest cost/conditioning result, not a certificate failure.

The small indefinite-quartic comparison produced 128 accepted steps for each
reconstruction method. Mean accepted loss changes were −.322 for eigenvalue
clipping and −.384 for the exact small-subspace trust-region solve; each used
the same eight acquired directions and charged actual-loss Armijo checks.
Across 32 stationary-saddle seeds, gradient-only decrease was zero, while a
two-HVP random-start Lanczos exploration found a negative direction and
decreased the constructed objective in every case. This controlled example
does not yield a general negative-curvature detection or nonconvex convergence
guarantee.

## Scalar information objective and scaling

The scalar study has 3,072 Bayesian design records plus 384 rank/polarization
records. For dimension 4/8, correct prior scale, and cap eight, mean squared
inverse-step errors in A/B/C/D order were `1.431e-4`, `5.694e-5`, `1.509e-4`,
and `8.005e-5`. The posterior-dependent, goal-oriented vector designs (B/D)
performed better in this finite linearized experiment than the preplanned
entropy design (A/C), but count adaptation did not uniformly improve error.
The selected design criterion uses a local Jacobian of `-H⁻¹g`; it is not exact
nonlinear posterior risk. Misspecified prior scales are retained in the raw
data and must constrain any practical interpretation. Rank accounting confirms
that generic scalar symmetric recovery needs `n(n+1)/2` independent features,
while vector HVP columns reveal `n` entries per query; polarization uses
quadratic scalar measurements to synthesize bilinear entries at additional
query cost.

The matrix-free dimension-64/512/4096 microbenchmark has 432 records. For its
small rank-eight operator, average complete acquisition-path times were about
.0009/.0012/.0033 seconds, while measured operator time was .0001/.0001/.0003
seconds. The reported minimum state grows from 18 KB to 1.33 MB for cap 16 at
dimension 4096. Host timing at this scale mostly reflects Python and dense
projected linear algebra; it cannot predict accelerator, distributed, or
language-model performance.

## Refinement decision

The cycle resolved key interpretation risks: repeated fixed quadratics reward
persistent Krylov methods; noisy observed residuals can invalidate certificates;
drift needs a bound; loose relative residuals do not certify energy accuracy;
and a small scalar Bayesian goal-design result is prior-art-compatible and
non-scalable. It did not establish an implementable LM method, practical noise
or drift estimator, globally reliable nonconvex strategy, or superiority over
modern training systems. The finite budget is complete after the registered
cycle and literature revisit; those unresolved items belong in subsequent
research questions, not claims of complete curvature coverage.
