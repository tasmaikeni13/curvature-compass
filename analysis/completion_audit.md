# Phase 1 completion audit

This audit follows the finite budget registered in `protocol.md`. “Complete”
means the deliverable and its stated scope are present; it does not mean the
scientific literature, all possible algorithms, or general guarantees are
exhausted.

| Requirement | Evidence and boundary |
|---|---|
| 1.1 Broad primary-source review and novelty | Thirty retrieved primary PDFs with SHA-256 and extracted-text line counts; paper-by-paper formulation/oracle/objective/guarantee/overlap notes and search limits in `literature_review.md`. Search is broad but not literally exhaustive; inaccessible 2026 leads remain unresolved. No novelty claim. |
| 1.2 Mathematical analysis | `measurement_limits.md`, `acquisition_theory.md`, and `statistical_robustness.md` cover scalar/vector identifiability, the shared Galerkin objective, exact marginal gain, residual bounds, four cells, noise, drift, nonconvexity, conditioning, cost, and high-dimensional memory. General block Schur derivation and practical noise/drift estimation remain open. |
| 1.3 Statistical/numerical study | Registered 24-seed calibration and 64-seed held-out study spans eight SPD families, three dimensions, four gradient regimes, caps and candidate policies. Separate registered noise, drift, indefinite, accuracy, scalar Bayesian, iterative, reference, and matrix-free scaling cycles preserve raw records, manifests, seed ranges, and failures. All remain synthetic; uncertainty intervals are descriptive. |
| 1.4 Candidate comparison in A/B/C/D | Five fixed and five adaptive alignment candidates with seven stopping choices and three caps were assessed; 240 candidate/cap summaries retained. The calibrated quartet was frozen before held-out evaluation. Further baselines include CG, Nyström-PCG, steepest descent, persistent methods, and L-BFGS under explicit budgets. Several cost and matching conclusions are qualified in the reports. |
| 1.5 Formal foundation | Fifteen Lean 4.34.1/Mathlib v4.34.1 theorems compile. Axiom audit reports only `propext`, `Classical.choice`, and `Quot.sound`; no `sorryAx`. The ledger explicitly excludes several general numerical-analysis claims. |
| 1.6 Living theory | `theory.md` states the current formulation, evidence, prior-art boundary, limitations, and next research question; it links to supporting detail. |
| 1.7 Iteration and stop decision | `decisions.md`, `refinement_protocol.md`, and `refinement_results.md` preserve failed checks, changed tolerances, frozen selections, new-seed follow-ups, and residual uncertainties. The registered finite cycle is complete; further credible claims require scalable stochastic-objective and training-system experiments. |

The checked-in raw-data and source hashes, row counts, literature count, and Lean
log were verified by `analysis/audit_artifacts.py`; its output is
`analysis/results/integrity_audit.json`. No Phase 2 or later training or
infrastructure work is included.
