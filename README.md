# Curvature Compass

Phase 1 research on adaptive directional curvature acquisition for optimization.
The original mission is in [prompt.md](prompt.md); **only Phase 1 is authorized**.
No language model training infrastructure or training runs are part of this repository.

The investigation separates scalar quadratic measurements from Hessian-vector products,
tests fixed/adaptive count and alignment as distinct factors, and treats existing
Krylov, sketching, and quasi-Newton methods as serious prior art.

Start with [theory.md](theory.md), then the detailed investigations in `analysis/`
and checked Lean proofs in `formal/`. Reproduction instructions and final findings
will be added after the experiments and proof checks actually finish.
