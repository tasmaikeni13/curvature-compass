# Iteration and critical review ledger

## Before outcome evaluation

1. Restrict foundational work to synthetic studies. Create a private repository and register the protocol
   before any calibration/evaluation outcomes (commit `03bac29`).
2. Use a shared Galerkin quadratic step so the two factors have identifiable
   meanings. Separate scalar identifiability from vector acquisition studies.
3. Prior art already includes adaptive Newton–CG, adaptive sketch sizes, Bayesian
   secant interpretation, and Nyström rank selection. Broad novelty claims are
   rejected at the outset; a renamed CG policy cannot constitute a contribution.
4. Five fixed and five adaptive alignment rules, seven common stopping rules,
   caps 2/8/16, calibration seeds 1000–1023, held-out seeds 2000–2063. Each seed
   contains all eight families, three dimensions, and four gradient regimes.
   Bootstrap clusters are root seeds. Selection criterion is progress per HVP;
   absolute progress and cost frontiers are mandatory companion outcomes.
5. Learn a fixed prior from *calibration gradients only*. This is a serious fixed
   design, with a stated transfer assumption; no current Hessian eigenvectors
   are passed to policies.
6. First invariant run demanded conventional CG and reorthogonalized Galerkin
   agree to `1e-16` relative energy. It failed at `7.09e-10` on the declared
   ill-conditioned populations. This threshold ignored three-term CG loss of
   orthogonality. Keep the two implementations independent; use `1e-7` relative
   energy tolerance and preserve the observed maximum. Do not assert bitwise
   equality or a new theoretical discrepancy.
7. The first Lean build verified the general quadratic gap/optimality and
   residual-bound proofs but failed on computability declaration and rational
   Schur denominator clearing. Failed elaborations are not verified theorems;
   fix the proof script and require a successful build without `sorryAx`.

Further entries must record actual evidence, exploratory changes, and independent
confirmation after those changes; never rewrite unfavorable hypotheses out of
the historical record.

## Outcome review and stop decision

8. The current Lean build compiles all fifteen theorems. The added perturbation
    lemma now states its operator/gradient-radius assumptions explicitly; the
    complete axiom output is in `results/lean_build.log`. No `sorryAx` remains.
9. The 64-seed held-out quartet reproduced the calibration ordering qualitatively:
    adaptive Krylov alignment drove large one-step progress, while its stopping
    policy traded progress for count. C's mean progress was below a randomized
    fixed-budget interpolation at the same mean count. Keep the fixed-quartet
    result and the selection-objective caveat; do not promote a new cell.
10. The reference Nyström-PCG experiment did not beat CG at equal total HVP
    budgets in the sampled one-step tasks. The repeated stationary quadratic
    makes persistent CG/PCG look much better because they reuse the same
    operator. This is a real boundary condition, not evidence of performance
    under changing neural-network curvature.
11. Noise results exposed an interpretation failure: an observed residual alone
    can understate target error. Use the conditional inflated certificate only
    when operator/gradient radii and a valid `μ` are available. Every robust
    bound covered the tested bounded-noise cases; do not extend this deterministic
    statement to unbounded Gaussian noise or an estimated neural-network radius.
12. The stricter relative-gap target often failed to certify before the cap.
    Preserve failures as unmet targets. The exact small trust-region safeguard
    improved accepted loss in these indefinite examples, but the study does not
    support a general nonconvex convergence claim.
13. Scalar goal-oriented Bayesian design reduced step error in the correctly
    specified small cases. Its dense covariance and local linearization prevent
    treating it as a scalable optimizer. Goal-oriented design is direct prior
    art, not a novelty claim.
14. The final artifact audit checked every raw file hash, row count, recorded
    Python source hash, all thirty retrieved primary-source records, and the
    successful fifteen-theorem Lean log. Its machine-readable output is
    `results/integrity_audit.json`.

The registered cycle is complete at the stated finite research budget. The next
unresolved questions are scalable goal-risk estimation, practical noise/drift
bounds, batched versus sequential HVP cost on accelerators, and behavior on a
changing stochastic training objective. They require subsequent experimental work and are
not presented as solved here.
