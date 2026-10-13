# Lean proof ledger

Build with Lean 4.34.1 and Mathlib v4.34.1 pinned in `lean-toolchain`,
`lakefile.toml`, and `lake-manifest.json`:

```sh
lake build
```

[`CurvatureCompass.lean`](CurvatureCompass.lean) contains 15 proved results.
The build runs `#print axioms` for each theorem; the accepted dependencies are
Lean's standard `propext`, `Classical.choice`, and `Quot.sound`. The final
successful build log is archived at `analysis/results/lean_build.log`.

| Result | Scope and limitation |
|---|---|
| `scalar_polarization`, `coordinate_scalar_blindness`, `vector_first_column_identifies_coupling`, `scalar_same_vector_different` | Exact scalar/vector information examples over finite real vectors; not a full generic rank theorem. |
| `quadratic_gap`, `subspace_optimality`, `nested_subspace_monotone` | Arbitrary real modules with symmetric bilinear form and the stated stationarity hypotheses; applies to exact subspace minimizers. |
| `coercive_residual_bound` | General real inner-product space and coercive linear operator. |
| `schur_gain_identity`, `positive_schur` | Two-dimensional algebra with explicit nonzero/positivity hypotheses; the general block Schur derivation remains in the theory notes. |
| `negative_curvature_escape` | Explicit two-dimensional indefinite example. |
| `marginal_gains_not_submodular`, `coupled_counterexample_coercive` | Exact coupled two-coordinate counterexample and its positive-definiteness check; the interpretation as a set function is documented algebraically. |
| `residual_perturbation_bound` | General real inner-product space; conditional on supplied operator and gradient error bounds. |
| `objective_certificate` | Algebraic implication from a valid gap bound to a relative captured-decrease target. |

Classical CG convergence rates, Kantorovich bounds, generic scalar recovery
rank, full-dimensional Schur identities, trust-region convergence, and any
language-model claim are not Lean theorems in this project. Numerical checks
are evidence about sampled implementations, not formal proofs. No theorem is
accepted if the build fails or prints `sorryAx`.

The prepared language-model optimizer uses stochastic, possibly indefinite
curvature, damping and reduced eigenvalue clipping. Its residual stopping
policy is heuristic; the Phase 1 SPD certificate does not establish a Newton
error bound for those runs. No language-model theorem or empirical optimizer
advantage is claimed by these proofs.
