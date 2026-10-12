# Phase 1 protocol, registered before numerical outcomes

## Scope and stopping rule

Only mission sections 1.1–1.7 are in scope. The finite research budget is a broad
primary-source search, at least two theory/experiment refinement cycles, diverse
CPU synthetic studies, and a checked Lean foundation. Stop when each factor cell
has multiple evaluated candidates, independent evaluation of the selected rules,
explicit failures, and a defensible next-phase research question. Exhaustive
coverage of all scientific literature and global algorithmic optimality cannot
be certified; search coverage and gaps must be stated.

## Definitions

At a step, gradient `g` and an unknown symmetric curvature operator `H` are given.
Scalar oracle: `q(v)=vᵀHv`. Vector oracle: `y(v)=Hv`. A vector oracle returns `n`
coupled numbers; these are different experiments and different identifiability
models. A scalar observation computed by automatic differentiation may still
incur a full HVP internally; never count it as inherently cheaper.

For the main exact SPD experiment use the shared variational step
`s=-Q(QᵀHQ)⁻¹Qᵀg`, with orthonormal columns of `Q`. All acquisition rules use this
same reconstruction/update. The dense `H` is available to the simulator, but a
policy may only access the charged oracle and a declared eigenvalue lower bound.
Dense eigenvectors, exact solutions and objective optima are evaluation oracles.

“Fixed alignment” means a complete ordered direction stream is selected before
seeing the current gradient, Hessian, or observations. A seeded changing stream
across outer steps is fixed under this definition. “Adaptive alignment” includes
gradient seeding and dependence on current/past observations. Fixed count is a
predeclared cap, constant across outer steps, with numerical rank exhaustion
reported separately. Adaptive count is a stopping time with respect to the
observed gradient/HVP history. Prefix policies may not reorder an unseen stream.

## Candidate search

* A: coordinate, Haar orthogonal, Rademacher-QR and structured cosine streams;
  multiple budgets. Structured designs are not assumed worse than random ones.
* B: gradient-seeded Krylov/residual expansion, residual-greedy coordinates,
  gradient-first random completion; the same budgets as A.
* C: all strongest A streams with residual certificates, marginal-gain and
  cost-aware stopping; count caps identical to the B/D caps.
* D: all strongest B streams with the same stopping policies as C.

Use independent calibration seeds for candidate selection. Select A/C stream,
B/D alignment, and each stopping rule using balanced mean oracle-normalized
quadratic progress. Equal search effort per cell; disclose all configurations.
The selected quartet shares the A/C stream and B/D alignment so the factorial
effects are identifiable. Evaluate all candidates as sensitivity analyses too.
Do not retune on the held-out evaluation; any post-evaluation refinement is
explicitly exploratory and must receive a new seed block.

## Synthetic populations

SPD families: flat identity; diagonal and randomly rotated ill-conditioned;
low-rank plus ridge; clustered spectrum; Toeplitz correlation; block anisotropy;
localized needle curvature. Dimensions 16, 48, 96; several measurement budgets.
Include isotropic gradients, gradients generated from isotropic parameter errors,
and gradients concentrated in extreme eigenspaces. Use paired random instances
and seeds across policies. Preserve per-instance outcomes, not just aggregates.

Robustness experiments add coherent symmetric operator noise, inconsistent fresh
HVP noise, indefinite/quartic objectives, and curvature drift. These populations
do not inherit exact-SPD guarantees. Include a larger-dimensional matrix-free
structured case to assess scaling without claiming neural network realism.

## Outcomes and statistical analysis

Primary one-step outcome: actual quadratic decrease / exact Newton decrease,
plus decrease per charged HVP. Show full progress/count Pareto curves, not only a
ratio winner. Secondary: relative energy error, residual norm, elapsed time,
orthogonalization/solve time, HVP count, memory, numerical failures.
Residual computation from stored exact HVPs is free of additional oracle calls;
explicit validation HVPs, spectral estimates, and discarded candidates are charged.

Report paired mean effects and bootstrap 95% confidence intervals with resampling
by independent problem seed, separately by family/dimension/gradient regime.
Cells and repeated policies on a seed are correlated: do not treat them as
independent samples. Report descriptive intervals without post-hoc significance
claims or uncorrected multiple-comparison winner declarations.

Factorial effects: alignment B−A, count C−A, joint D−A, and interaction
`D−B−C+A`. Count-changing comparisons are accompanied by matching average HVP
budgets and a common total-HVP iterative experiment. Runtime is measured on this
host with single-threaded BLAS; synthetic dense timings are not a proxy for HVP
cost in a language model. Include a declared cost model sensitivity sweep.

Reference mechanisms: steepest descent, fixed-budget and residual-stopped CG,
randomized Nyström preconditioning, and a limited-memory quasi-Newton mechanism
where feasible. Never attribute CG's known optimization targeting to novel
tomography. Scalar reconstruction and polarization have their own study.

## Verification and integrity gates

Numerically verify Galerkin orthogonality, monotonic exact nested-subspace decrease,
the Schur-complement marginal-gain identity, SPD residual error bounds, scalar
polarization, scalar/vector rank separation, and equivalence of residual-Krylov
acquisition to CG in exact arithmetic. Check noise and stale-curvature failures.
Use exact low-dimensional counterexamples where possible.

Lean must compile every theorem labeled verified, with no `sorry`, added axioms,
or proof placeholders. The proof ledger distinguishes finite scalar/2D foundations
from general-dimensional results that remain informal. Check axiom dependencies.
Proof failures that expose missing assumptions must change the theory/protocol.

Keep references, primary-source reading locators, code/environment versions, run
manifests, seeds, raw data, reports, failed candidates, and final requirement audit.
No unexecuted experiment or unproved theorem is labeled complete.
