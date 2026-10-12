# Curvature Compass — living Phase 1 theory

Status: research in progress. Only Phase 1 of `prompt.md` is in scope.

Scalar `vᵀHv` and vector `Hv` oracles have different information and costs.
The common experimental update is the minimizer of
`gᵀs + ½sᵀHs` in an acquired subspace, for SPD `H`.
Count and alignment are separate factors under [the protocol](analysis/protocol.md).

Known methods, particularly inexact Newton/CG, already adapt HVP directions and
termination. Novelty is unresolved; the project must compare mathematical
equivalence rather than terminology. No empirical superiority or formal theorem
is claimed before the corresponding experiments and Lean builds finish.
