# Measurement models, identifiability, and impossibility

All statements here concern real symmetric matrices unless stated otherwise.
Proof status is explicit: detailed algebra/rank arguments below are informal
mathematical proofs; selected foundations have checked Lean counterparts.

## Three genuinely different oracles

1. Scalar directional curvature: `q(v)=vᵀHv+ε`. The linear measurement matrix is
   `vvᵀ`; one query gives one real constraint on `Sym(n)`.
2. Vector product: `y(v)=Hv+ξ`. One query gives a whole column action; its outputs
   are coupled by symmetry. AD implementations ordinarily do a gradient-scale
   operation, so `n` outputs do not mean `n` gradient-scale costs.
3. Sampled Fisher/GGN or gradient statistics: these measure a different matrix or
   average outer products. Positive semidefiniteness can follow from their
   construction but does not establish equality to the nonconvex loss Hessian.

Finite differences of gradients approximate vector actions, with truncation and
sampling error; function values along a line may approximate scalar curvature.
Directional scalars obtained by contracting a computed HVP still cost that HVP.
We do not mix any of these costs or guarantee domains.

## Scalar lower bound and a constructive design

Write `d=n(n+1)/2`. A finite exact scalar transcript yields a linear map
`A:Sym(n)→R^m`, `A(H)_i=tr(Hv_i v_iᵀ)`. Its rank is at most `m`; if `m<d`,
its kernel contains a nonzero symmetric `D`. On any SPD `H0`, sufficiently small
`t≠0` keeps `H0+tD` SPD, but leaves the transcript unchanged. Full reconstruction
of an arbitrary SPD matrix therefore requires at least `d` scalar queries.

This rank argument also applies to an adaptive deterministic policy along a
fixed transcript: `H0` and `H0+tD` induce the same first observation, next chosen
direction, next observation, and so on by induction. For randomized policies,
condition on their realized internal randomness. This is an indistinguishability
argument for exact uniform identification; it is not an information-theoretic
lower bound for every distributional optimization objective.

The lower bound is attainable without noise: query `e_i` and `e_i+e_j`, then
`H_ij=[q(e_i+e_j)-q(e_i)-q(e_j)]/2`. This uses exactly `d` scalars. Unit-normalized
directions require the corresponding scale correction. Independent homoscedastic
raw scalar noise of variance `σ²` gives off-diagonal variance `3σ²/4`; shared
diagonal observations also correlate reconstructed off-diagonal errors. Repeated
scalar observations increase cost; arbitrary orientation changes conditioning.
The 2D polarization and coordinate blind spot are Lean verified.

Recovering a projected `k×k` Hessian from scalars analogously needs `k(k+1)/2`
independent scalars in general. Treating `k` scalars as a dense `k×k` reconstruction
without a diagonal or structural assumption is invalid.

## Vector information is stronger, but not complete

For linearly independent queried columns `V` of rank `k`, change coordinates so
the first `k` basis vectors span `V`. Symmetry and `HV` determine the first block
columns and rows. The lower-right symmetric block remains free, with dimension
`(n-k)(n-k+1)/2`. Thus the number of independent constraints is
`nk-k(k-1)/2`, not `nk`. Full arbitrary-Hessian identification needs `k=n` HVPs.
The remaining free block can again be perturbed while preserving SPD.

If `u⊥span(V)`, the symmetric rank-one perturbation `t uuᵀ` is invisible because
`(t uuᵀ)V=0`. Its influence on `H⁻¹g` need not vanish. In the special case where
`H⁻¹g` already lies in the queried subspace, the Galerkin step is exactly optimal
despite a large unobserved Hessian block. Therefore full-matrix information and
update information are different tasks.

The simplest separation is 2D: querying `e1` gives scalar `a` from
`H=[[a,b],[b,c]]`, but the vector query gives `(a,b)`. Coordinate scalar queries
cannot see `b`. Even the two SPD matrices `I` and `[[1,.9],[.9,1]]` have identical
coordinate scalars and very different inverse actions. The exact 2D separation
is checked in Lean and the numerical invariant suite.

## Structural promises must be justified

Rank-one matrix-sensing theory [Kueng–Rauhut–Terstiege](literature/sources.json)
can reduce scalar budgets to order `nr` for rank-`r` recovery under its random
design, noise, and convex recovery assumptions. A Hessian with a full-rank ridge,
a long spectral tail, or negative eigenvalues is not automatically in that
model. Known-ridge subtraction is an additional assumption. Scalar measurement
count does not include the often substantial nuclear-norm recovery computation.

Vector low-rank sketches exploit range information from `HΩ`; effective dimension
`d_eff(λ)=Σ_j λ_j/(λ_j+λ)` is relevant to PSD ridge preconditioning. Neither
low-rank Hessian reconstruction nor small `d_eff` alone guarantees the gradient
has little mass in poorly reconstructed inverse directions. Small eigenvalues
receive large inverse weights. Example: the spectrally tiny direction carrying
all of `g` is the only direction that matters for the update.

## Arrangement: what is possible without current-state information?

For a prescribed proper subspace `S`, choose `H=I` and `g⊥S`, `g≠0`. A projected
update makes zero progress; a gradient-seeded direction solves the same model in
one HVP. No fixed proper subspace has a positive worst-case fraction of Newton
progress over unrestricted gradients. This does not mean every fixed design is
weak on a relevant distribution.

For `H=I`, isotropic Gaussian `g`, a rank-`k` orthogonal projector captures the
expected fraction `k/n` of squared gradient norm. For a fixed random unit gradient
the exact expected progress fraction is also `k/n`. Adaptive gradient alignment
achieves one. For an offline-known gradient covariance `Σ`, expected absolute
decrease is `½tr(P_SΣ)`. Ky Fan's variational principle gives the optimal fixed
`k`-subspace: the top covariance eigenspace. This is the rationale for the learned
prior baseline, trained only on independent calibration gradients. Its isotropic
average cannot outperform `k/n`; its distribution-specific advantage may disappear
under rotation, drift, or changed gradient distribution.

For general known SPD `H`, the optimal *distributional* fixed subspace is aligned
with the top eigenspace of `H⁻¹/²ΣH⁻¹/²` in energy coordinates. This is an oracle
characterization, not a cheap query policy for unknown `H`. Offline estimates
have their own acquisition cost and transfer error. The numerical study uses
gradient-covariance priors, not concealed current-Hessian eigenvectors.

## Low-dimensional success does not require novelty

Gradient-started Krylov spaces grow as `span(g,Hg,…,H^(k-1)g)`. Exact CG finishes
when its minimal polynomial on `g` is exhausted: one step on a scaled identity,
two on an identity plus a single spike, and at most `r+1` on rank-`r` plus ridge.
These are established explanations for an adaptive-direction advantage. Merely
renaming their queries “tomography vectors” cannot support originality.

## Open limits

We have not established a minimax optimal count/alignment policy under drifting,
biased stochastic neural-network Hessians, unknown lower eigenvalue bounds,
finite precision, and realistic HVP cost. The uniform full-reconstruction lower
bounds do not settle that problem. An honest optimization-specific lower bound
would need a declared task distribution, admissible policies, stochastic oracle,
loss/compute objective, and success criterion; its derivation remains open.
