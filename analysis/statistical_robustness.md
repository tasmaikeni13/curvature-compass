# Uncertainty, nonconvexity, cost, and interpretation

## Exact residuals and noisy observations are different experiments

Let the true SPD target be `K⪰μI`, the queried operator `K̃=K+E`, and the observed
gradient `g̃=g+e_g`. For any computed step `s`, define
`r̃=g̃+K̃s`, `r=g+Ks`. If a valid bound `‖E‖₂≤ε_H` and `‖e_g‖≤ε_g` is available,

`‖r‖≤‖r̃‖+ε_H‖s‖+ε_g`,
`m(s)-m(s*)≤(‖r̃‖+ε_H‖s‖+ε_g)²/(2μ)`.

The perturbation inequality is Lean verified for linear operators on a real
inner-product space; the energy bound then follows from the verified coercive
bound and quadratic-gap identity. All bounds are **conditional on a valid true
lower eigenvalue and error radii**. The matrix used by the solver need not itself
be SPD for the residual inequality, although its inverse could be unstable.
Clipping projected eigenvalues or limiting the step norm changes the update, but
the residual bound still applies to the step actually taken.

For coherent minibatch noise, `E` is a single symmetric matrix throughout the
inner solve. Holding the minibatch fixed preserves a linear symmetric operator;
it does not remove its sampling bias relative to the target. Conventional CG's
fixed-operator theory can apply to a sufficiently SPD minibatch operator, while
its target-error certificate still needs the operator mismatch term.

For fresh bounded per-query errors `Y_j=Kq_j+ξ_j`, `‖ξ_j‖≤σ`, orthonormal `Q`,
the action discrepancy on `span(Q)` is bounded by
`‖[ξ₁,…,ξ_k]‖₂≤‖[ξ₁,…,ξ_k]‖_F≤σ√k`. Thus the same bound holds with
`ε_H=σ√k` on the step's subspace. A fresh Gaussian noise law has no deterministic
finite radius: it requires a confidence bound and failure budget, not pretending
the standard deviation is an almost-sure bound. The experiment uses normalized
bounded noise specifically to test the claimed conditional certificate.

In the 8,192-row bounded-noise study, the observed-residual certificate
underestimated the true gap in 15.0% of rows when stopping on observed residual
and 17.5% when stopping on the robust rule (both certificate values are
recorded for the latter). The inflated known-radius certificate covered every
tested gap. These pooled synthetic rates depend on the declared noise
distribution and gradient-noise level; they are not calibrated probabilities
for training. Full stratification and counts are in
[`refinement_results.md`](refinement_results.md).

Independent true-HVP residual audits cost one extra HVP and are reported separately
from acquisition counts. They are evaluation checks; they are not silently fed
back into fixed-alignment policies. An adaptive validation HVP in direction `s`
would itself change the measurement-direction classification if used by C.

## Drift and reuse

For stale `K_old` and current `K_now`, replace `ε_H` by
`‖K_now-K_old‖₂`. A small step in parameters does not give a small drift bound
without a Hessian-Lipschitz or other continuity assumption. Under Lipschitz Hessian
constant `M`, one may use `ε_H≤M‖θ_now-θ_old‖`; estimating `M` has a cost and risk.
Old HVPs cannot simply be treated as fresh exact observations.

Our controlled rotations compute the discrepancy norm **for analysis**. They do
not furnish a cheap learned drift estimator. Reuse can improve amortized cost in
a stationary quadratic, as continuous CG and persistent Nyström preconditioning
illustrate. Neural-network Hessians, gradients, minibatches, and damping change;
that stationary advantage is a boundary case, not a training prediction.

## From a certificate to an optimization target

`F*=F_Q+gap`, `gap≥0`, and `gap≤B_Q=‖r_Q‖²/(2μ)`. For desired relative gap
`η∈(0,1)`, stopping when
`B_Q≤[η/(1−η)]F_Q`
ensures `F_Q/F*≥1−η`. Lean verifies the algebraic implication; the numerical
accuracy study checks the complete operator-level criterion. If the cap is
reached first, the target is not certified and must be recorded as unmet.
This criterion is stronger and more interpretable than a bare relative residual,
but often conservative for an ill-conditioned model with a coarse `μ`.

With noise, replace `B_Q` by the enlarged bound above. It can correctly refuse to
declare sufficient accuracy even when a small observed residual tempts the solver
to stop. Additional noisy measurements may increase the radius or make little
progress. Increasing count cannot solve a biased oracle problem by itself.

## Indefiniteness and actual loss

For indefinite `H`, the unconstrained quadratic may be unbounded below. The SPD
projection formula and CG guarantees cannot be used unchanged. Restricting to a
queried subspace gives `G=QᵀHQ`, but an observed positive `G` does not prove `H`
has no negative curvature outside that subspace.

Two competing safeguards are evaluated using identical directions:

* Clip `G`'s eigenvalues below a positive floor, solve, and limit step norm.
* Solve `min_{‖a‖≤Δ} (Qᵀg)ᵀa+½aᵀGa`, including the hard case in which the
  gradient is orthogonal to the minimum eigenspace. The KKT conditions are
  `(G+λI)a=-Qᵀg`, `G+λI⪰0`, `λ≥0`, and `λ(‖a‖−Δ)=0`.

Actual-loss Armijo acceptance is explicitly charged. The raw projected model's
decrease does not imply loss decrease. For `M`-Lipschitz Hessian, the model error
is bounded by `M‖s‖³/6`; this assumption and constant are not proved for an LM.
Repeated acceptance tests can dominate any saved HVPs.

At a stationary saddle, `g=0` carries no gradient-seeded exploration direction.
A gradient-only positive-definite update stagnates even with negative curvature.
On an identity minus a single negative spike, a random-start two-dimensional
Lanczos space reveals the negative mode almost surely in exact arithmetic.
The synthetic demonstration is not a guarantee that two queries detect negative
curvature in a general high-dimensional Hessian.

## Scalar Bayesian acquisition and a nonlinear goal

Let orthonormal symmetric basis matrices be `B_i`, `H=H0+Σ_i h_i B_i`,
`h~N(m,Σ)`, and `q(v)=vᵀH0v+φ(v)ᵀh+noise`, `φ_i=vᵀB_i v`.
Under independent Gaussian scalar noise with variance `σ²`, covariance reduction
is `ΣφφᵀΣ/(σ²+φᵀΣφ)`. Entropy gain is
`½log(1+φᵀΣφ/σ²)`, and a fixed prior and preplanned scalar design give a
covariance schedule independent of observation values.

For inverse step `s(H)=-H⁻¹g`, linearization around the posterior mean gives
Jacobian column `J_i=-Hbar⁻¹B_i s(Hbar)` and approximate step uncertainty
`tr(JΣJᵀ)`. A query's predicted reduction is
`‖JΣφ‖²/(σ²+φᵀΣφ)`. It targets the current gradient and may be recomputed after
observations. This is a direct goal-oriented experimental-design adaptation.
The goal is nonlinear and the Gaussian matrix prior is not PSD-supported;
posterior-mean eigenvalue safeguards do not make the uncertainty formula exact.

Adaptive directions do not invalidate correctly specified sequential Gaussian
regression merely because they are chosen from past observations: the likelihood
conditions on the declared acquisition rule and observed history. However,
misspecified noise, a gradient informative about the unknown matrix, implicit
rejection of non-SPD latent matrices, or an invalid Jacobian approximation can
invalidate calibration. Our scalar gradients are independent of the latent
matrix; misspecified prior scale is studied deliberately.

Memory is `O(d²)=O(n⁴)` for full matrix covariance, with `d=n(n+1)/2`. These
small scalar experiments establish a competing value objective and expose risk;
they are not a scalable optimizer proposal for millions of parameters.

## Computational accounting and attainable implementations

HVP count is an information budget, not a complete compute budget. At minimum
track gradient evaluations, loss evaluations, HVPs, stream setup, orthogonalization,
projected solves, state memory, and reuse. HVP costs can vary by minibatch size,
AD implementation, vectorization, precision, and communication. Count discarded
probes and independent audits. Fixed vectors can be batched more readily than
sequential information-dependent Krylov directions; serial counts miss that gain.

The transparent reference code deliberately recomputes `QᵀY` and an eigensolve at
every prefix. Its overhead is `O(nk³+k⁴)` across the full acquisition path, plus
`O(nk²)` reorthogonalization/setup and `k` HVPs. An incremental implementation
can reduce that overhead, but no unimplemented acceleration is credited. The
stored vector state is `O(nk)`. Classical short-recurrence CG needs `O(n)` state
and overhead; reorthogonalization has different stability and memory costs.

At 60M parameters, two float32 `n×8` arrays alone occupy 3.84 GB; at 125M,
8 GB. A full learned covariance basis needs `O(n²)` storage and is impossible
at those scales. The small-matrix prior winner cannot be carried over verbatim.
Defensible future A/C candidates are structured or sparse fixed streams and
externally calibrated block priors; B/D candidates are short-recurrence (P)CG
with declared SPD curvature and damping. Block structure alters the approximation
target and needs explicit validation. This is a feasibility analysis, not full
implementation or authorization to start model training.

## Statistical safeguards and ratio interpretation

Primary configurations are chosen on 24 calibration seeds and frozen before 64
held-out seeds. A root seed generates multiple related populations and policies;
bootstrap resampling uses seed clusters rather than treating all prefix outcomes
as independent trials. Family/dimension/regime summaries expose heterogeneous
effects. The many descriptive intervals do not constitute multiplicity-adjusted
significance tests or prove a universal winner.

`E[progress/count]` differs from `E[progress]/E[count]`; the former can favor
policies that solve easy instances cheaply while spending heavily on difficult
ones. In the primary C result, that difference materially changes interpretation.
We retain the registered selection objective and disclose its limitation instead
of retroactively picking a favorable metric. Mean-count matching via randomized
fixed-cap mixtures is descriptive and only interpolates inside measured support.

Repeated quadratics include both equal-HVP and declared equal-work perspectives.
A zero-HVP method such as L-BFGS cannot be fairly limited solely by HVP count.
Actual runtimes are implementation- and host-specific; small dense numerical
studies do not predict neural-network wall-clock speed, generalization, or
stochastic training stability. Such claims remain unresolved in synthetic settings.
