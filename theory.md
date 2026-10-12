# Adaptive curvature acquisition: Phase 1 findings

**Research status:** Phase 1 is complete as a bounded synthetic and formal
foundation. It establishes no empirical advantage on language models and makes
no novelty claim. The next valuable work is to test a scalable, cheaply
calibrated goal criterion on changing stochastic objectives.

For an SPD local model `m(s)=gᵀs+½sᵀHs`, a set of orthonormal measured
directions `Q` defines the common step
`s_Q=−Q(QᵀHQ)⁻¹Qᵀg`. This makes alignment and count comparable and gives exact
one-step decrease `F_Q=½gᵀQ(QᵀHQ)⁻¹Qᵀg`. With valid `H⪰μI`, the remaining model
gap is at most `‖g+Hs_Q‖²/(2μ)`. An appended direction has exact marginal gain
`(vᵀr_Q)²/(2δ)`, where `δ` is its Schur complement. The numerator is known
before querying, but the denominator is not. Residual Krylov expansion is CG,
and an exact inverse-residual direction would hide the original solve inside
the acquisition oracle.

Scalar measurements `vᵀHv` and vector products `Hv` have different information
content and cost. Generic symmetric recovery from scalar rank-one measurements
has `n(n+1)/2` unknowns; vector queries expose a full column. Neither full matrix
reconstruction nor low Frobenius error guarantees a useful inverse action for a
particular gradient. The scalar Bayesian study makes the goal explicit with a
local inverse-step uncertainty criterion, but its dense covariance is
`O(n⁴)` memory and its posterior risk is only a first-order approximation.

On the held-out synthetic population, the selected fixed PCA prior was useful
under its calibration-transfer assumption. At cap eight, the adaptive Krylov
cells made much more one-step progress, often exhausting their numerical rank
early. The selected C/D residual-.7 stop reduced HVP use; C lost progress
against a descriptive fixed-budget mixture at its mean count. The per-instance
mean of progress/HVP and the ratio of aggregate progress to aggregate HVPs
disagree enough that both are reported. Fixed and adaptive count therefore
have an economic tradeoff, not a demonstrated universal win.

Follow-up quadratics show that persistent CG can reuse a stationary operator
and finish in fewer outer-step HVPs; this advantage may disappear as model,
gradient, minibatch, or damping changes. Bounded noise and stale curvature
require explicit error radii for residual certificates. An uncorrected
observed residual under-certified the true gap in 15% of the tested noise rows;
the known-radius robust bound covered all tested rows. The radius was supplied
by the simulator. A 10% relative-gap target was reached in fewer than half the
conditioning cases before the cap. In indefinite examples, an exact small
trust-region solve improved accepted loss more than eigenvalue clipping under
the same directions, but neither establishes general nonconvex convergence.

Prior art already covers CG/inexact Newton, adaptive sketch size, Nyström
preconditioning, Bayesian curvature inference, randomized Hessian estimation,
and goal-oriented experimental design. The defensible Phase 1 output is a
carefully costed comparison framework, counterexamples to blind residual and
submodular-gain assumptions, reproducible synthetic evidence, and a small set of
Lean-checked identities and bounds. Read [the full prior-art review](analysis/literature_review.md),
[acquisition theory](analysis/acquisition_theory.md),
[statistical limits](analysis/statistical_robustness.md),
[primary results](analysis/monte_carlo.md), and
[registered refinements](analysis/refinement_results.md). The proof scope and
reproduction commands are in [formal/README.md](formal/README.md) and
[README.md](README.md).
