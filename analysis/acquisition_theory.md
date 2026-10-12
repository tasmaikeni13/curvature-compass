# Optimization-targeted acquisition and the four cells

Use an exact SPD operator `K` (either the Hessian in a convex local model or a
declared SPD surrogate) and `m(s)=gᵀs+½sᵀKs`. The shift in `K=H+λI` changes the
target update; exact decrease of `m` is not automatically actual loss decrease.

## A common variational reconstruction

For acquired full-rank directions `Q`, set `G=QᵀKQ`,
`s_Q=-QG⁻¹Qᵀg`, `F_Q=-m(s_Q)=½gᵀQG⁻¹Qᵀg`. Exact HVPs determine `G` and the
full residual `r_Q=g+Ks_Q` from stored observations, without another HVP.
The normal equations give `Qᵀr_Q=0`. Storage is `O(nk)` and the practical overhead
is orthogonalization, projected matrix construction, and the small solve.

Let `s*=-K⁻¹g`. Completing the square yields
`m(s)-m(s*)=½(s-s*)ᵀK(s-s*)`. In a subspace, stationarity along the displacement
and symmetry suffice for the same energy-gap identity. Lean verifies this in
arbitrary real modules with a symmetric bilinear form, subspace optimality, and
monotonicity on nested subspaces. It does not assume the numerical solver is exact.

Since `K⪰μI`, with a *valid known* `μ>0`,
`F*-F_Q=½r_QᵀK⁻¹r_Q≤‖r_Q‖²/(2μ)`. A general inner-product-space coercive residual
bound is Lean verified. A residual norm is not necessarily monotone even though
exact nested-subspace model progress is monotone. Tiny residual needs a curvature
scale before it implies tiny energy error.

## Exact value of one extra direction

Append a direction `v` independent of `Q`. Define
`b=QᵀKv`, `d=vᵀKv`, `a=vᵀg`, and the Schur complement
`δ=d-bᵀG⁻¹b>0`. Block inversion or eliminating old coefficients gives

`F_[Q,v]-F_Q = (a-bᵀG⁻¹Qᵀg)²/(2δ) = (vᵀr_Q)²/(2δ)`.

This is an exact optimization-value metric under the SPD local model. Unlike
Frobenius reconstruction error, trace, determinant, or entropy, it explicitly
weights the current gradient. The general-dimensional block derivation is an
informal proof; Lean verifies the nonzero-denominator 2D identity and positivity
under `a>0, ac-b²>0`. Numerical checks cover random dimensions/subspaces.

Before querying `Kv`, `δ` is unknown. Evaluating this score for a bank of `p`
directions costs `p` HVPs, including discarded candidates. Selecting with an
uncharged true-Hessian score is an oracle cheat. For unit `v⊥Q`, `δ≥μ`, so
`(vᵀr_Q)²/(2μ)` upper-bounds the next gain, but does not predict it tightly.
A zero immediate numerator does not prove every later direction is useless.

If unlimited inverse information were available, `v=K⁻¹r_Q` closes the entire
remaining gap in one extra direction; computing that direction is essentially
the original inverse-action task. This formal optimum is not an implementable
one-query shortcut. Residual direction `v=r_Q/‖r_Q‖` is cheap and yields a Krylov
space, explaining why it must be compared to established CG.

## Information gain is a competing objective, not a substitute

A Gaussian matrix prior on `h=vech(H)` with covariance `Σ_h` and a scalar query
feature `φ(v)=vech(vvᵀ)` gives posterior covariance
`Σ'_h=Σ_h-Σ_hφφᵀΣ_h/(σ²+φᵀΣ_hφ)`, with mutual information
`½log(1+φᵀΣ_hφ/σ²)`. A vector query uses a block design and correlated noise;
it is not `n` independent scalar observations. Dense covariance already needs
`O(n⁴)` memory. A structured prior is necessary at scale and creates model risk.

For a noisy SPD estimate, first-order perturbation of the inverse step is
`δs≈K⁻¹(δK)K⁻¹g`. Optimization-sensitive uncertainty therefore depends on a
goal vector and conditioning, while matrix entropy does not. Prior calibration
and numerical linear-solver uncertainty are known probabilistic-numerics topics.
Selecting an entropy-maximizing direction is a defensible alternative only if
one establishes a relationship to downstream loss or an explicit estimation task.

Optimization gain need not be submodular: `K=[[1,.9],[.9,1]]`, `g=(0,1)`. Direction
`e1` alone has zero value, `e2` alone value `½`, and the pair value
`1/[2(1-.9²)]≈2.632`. The marginal value of `e2` increases after adding `e1`.
Thus a generic `(1-1/e)` information-gain guarantee cannot be imported into
optimization-targeted acquisition. Lean checks the key strict inequality; the
translation into set-function values is an explicitly documented algebraic proof.

## Candidate families and exact classifications

| Cell | Alignment candidates | Count candidates | What is adaptive |
|---|---|---|---|
| A | coordinate, Haar, Rademacher QR, cosine, offline PCA prior | constant 2, 8, 16 | nothing in the current state |
| B | residual Krylov, residual-greedy coordinate, gradient then random/cosine/coordinate | same constants | gradient seed and/or observed-residual directions |
| C | same A streams, unchanged order | residual, marginal-gain, cost stops | observed stopping time only |
| D | same B rules | same stopping alternatives | direction and stopping time |

Five alignment candidates on each side and seven count policies on each adaptive
side receive the same balanced calibration populations. The selected quartet
shares its alignment within A/C and B/D and a common count policy within C/D.
The common cap is eight; other caps are recorded sensitivity results. Numerical
invariant-subspace exhaustion is logged as effective rank, even in fixed cells.
It is distinct from a chosen adaptive count rule.

Residual stops `‖r‖≤τ‖g‖` are standard inexact-solve forcing conditions. They
certify an absolute error only with `μ`. Marginal-gain stops check a past realized
gain, not an upper bound on future gains. The cost rule stops when past gain is
below a declared cost in gradient/lower-bound units; it is heuristic expected
value reasoning, not a theorem. Their blind-prefix failure is tested explicitly.

## Best, worst, and average cases

For `K=I`, B/D gradient seeding finishes in one HVP; a fixed rank-`k` subspace has
isotropic expected progress `k/n`. A calibrated prior can be ideal if gradients
remain in an externally known low-dimensional subspace. Coordinate methods can
win on sparse gradient structure and diagonal curvature; random sketches are
rotation-robust. Dense Krylov directions can have worse storage/communication
cost than coordinate or block directions despite better progress per HVP.

For SPD condition number `κ`, the standard CG energy-norm envelope is
`‖e_k‖_K≤2[(√κ-1)/(√κ+1)]^k‖e_0‖_K`; this implies a conservative progress
lower bound `max(0,1-4ρ^(2k))`. Actual spectral clustering can be far better than
this worst-case envelope. The one-dimensional gradient optimum has worst-case
fraction at least `4κ/(κ+1)²` by the Kantorovich inequality. Both are established
numerical-linear-algebra results, not project novelty and not Lean verified here.

Full-rank reconstruction is unnecessary, but a guarantee depending only on a
small spectral *reconstruction* rank misses inverse-weighted gradient tails.
Sparse/clustered/flat examples, adversarial extreme gradients, and long-tail
ill-conditioning are all included so candidate selection cannot rely solely on
the most favorable low-rank family.

## Count is an economic decision

If measurement `j` costs `c_j`, one can maximize `F_k-λΣc_j`, progress per HVP,
expected time to a target, or a risk-adjusted certificate. These objectives can
select different counts. A retrospective `argmax_k` using unseen gains is an
evaluation oracle, not an admissible adaptive stopping rule. Without a valid
predictive model, the statement “stop when expected benefit is smaller than cost”
does not by itself specify an algorithm. Repeated low-gain prefixes can conceal
high-value coupled directions; the project retains rather than suppresses those
negative findings.
