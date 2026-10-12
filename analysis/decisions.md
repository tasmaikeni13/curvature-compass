# Iteration and critical review ledger

## Before outcome evaluation

1. Restrict work to Phase 1. Create a private repository and register the protocol
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
