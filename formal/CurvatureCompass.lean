import Mathlib

namespace CurvatureCompass

noncomputable section

def scalarCurve (a b c x y : ℝ) : ℝ := a*x^2 + 2*b*x*y + c*y^2

theorem scalar_polarization (a b c : ℝ) :
    scalarCurve a b c 1 1 - scalarCurve a b c 1 0 - scalarCurve a b c 0 1 = 2*b := by
  unfold scalarCurve
  ring

theorem coordinate_scalar_blindness (a b c x y : ℝ) :
    scalarCurve a b c x 0 = scalarCurve a 0 c x 0 ∧
    scalarCurve a b c 0 y = scalarCurve a 0 c 0 y := by
  unfold scalarCurve
  constructor <;> ring

def vectorCurve (a b c x y : ℝ) : ℝ × ℝ := (a*x+b*y, b*x+c*y)

theorem vector_first_column_identifies_coupling (a b c : ℝ) :
    (vectorCurve a b c 1 0).2 = b := by
  simp [vectorCurve]

theorem scalar_same_vector_different :
    scalarCurve 1 (1/2) 1 1 0 = scalarCurve 1 0 1 1 0 ∧
    vectorCurve 1 (1/2) 1 1 0 ≠ vectorCurve 1 0 1 1 0 := by
  norm_num [scalarCurve, vectorCurve, Prod.mk.injEq]

section Quadratic
variable {E : Type*} [AddCommGroup E] [Module ℝ E]

def quadratic (B : E →ₗ[ℝ] E →ₗ[ℝ] ℝ) (l : E →ₗ[ℝ] ℝ) (x : E) : ℝ :=
  l x + (1/2)*B x x

/-- The energy gap needs stationarity only along the compared displacement. -/
theorem quadratic_gap (B : E →ₗ[ℝ] E →ₗ[ℝ] ℝ) (l : E →ₗ[ℝ] ℝ)
    (symmetric : ∀ x y, B x y = B y x) (x y : E)
    (stationary : l (x-y) + B y (x-y) = 0) :
    quadratic B l x - quadratic B l y = (1/2)*B (x-y) (x-y) := by
  have hsym := symmetric x y
  simp only [quadratic, map_sub, LinearMap.sub_apply] at *
  linarith

theorem subspace_optimality (B : E →ₗ[ℝ] E →ₗ[ℝ] ℝ) (l : E →ₗ[ℝ] ℝ)
    (symmetric : ∀ x y, B x y = B y x)
    (nonnegative : ∀ v, 0 ≤ B v v)
    (S : Submodule ℝ E) (y : E) (hy : y ∈ S)
    (stationary : ∀ v ∈ S, l v + B y v = 0) (x : E) (hx : x ∈ S) :
    quadratic B l y ≤ quadratic B l x := by
  have hgap := quadratic_gap B l symmetric x y (stationary (x-y) (S.sub_mem hx hy))
  have hnonneg := nonnegative (x-y)
  linarith

theorem nested_subspace_monotone (B : E →ₗ[ℝ] E →ₗ[ℝ] ℝ) (l : E →ₗ[ℝ] ℝ)
    (symmetric : ∀ x y, B x y = B y x)
    (nonnegative : ∀ v, 0 ≤ B v v)
    (S T : Submodule ℝ E) (contained : S ≤ T)
    (x y : E) (hx : x ∈ S) (hy : y ∈ T)
    (stationary : ∀ v ∈ T, l v + B y v = 0) :
    quadratic B l y ≤ quadratic B l x := by
  exact subspace_optimality B l symmetric nonnegative T y hy stationary x (contained hx)

end Quadratic

section Residual
variable {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E]

/-- A coercive curvature operator bounds energy by squared residual / mu. -/
theorem coercive_residual_bound (K : E →ₗ[ℝ] E) (mu : ℝ) (hmu : 0 < mu)
    (coercive : ∀ v, mu*‖v‖^2 ≤ inner ℝ v (K v)) (e : E) :
    inner ℝ e (K e) ≤ ‖K e‖^2 / mu := by
  have hn : 0 ≤ ‖e‖ := norm_nonneg e
  have hr : 0 ≤ ‖K e‖ := norm_nonneg (K e)
  have hco := coercive e
  have hcs := real_inner_le_norm e (K e)
  have hnorm : ‖e‖ ≤ ‖K e‖ / mu := by
    apply (le_div_iff₀ hmu).2
    by_cases hz : ‖e‖ = 0
    · simp [hz, hr]
    · have hp : 0 < ‖e‖ := lt_of_le_of_ne hn (Ne.symm hz)
      by_contra h
      have hc : 0 < ‖e‖*mu - ‖K e‖ := sub_pos.mpr (lt_of_not_ge h)
      have hprod := mul_pos hp hc
      nlinarith
  calc
    inner ℝ e (K e) ≤ ‖e‖*‖K e‖ := hcs
    _ ≤ (‖K e‖ / mu)*‖K e‖ := mul_le_mul_of_nonneg_right hnorm hr
    _ = ‖K e‖^2 / mu := by ring

end Residual

/-- Exact Schur-complement gain for a two-coordinate SPD quadratic. -/
theorem schur_gain_identity (a b c u v : ℝ) (ha : a ≠ 0)
    (hdet : a*c-b^2 ≠ 0) :
    (c*u^2-2*b*u*v+a*v^2)/(2*(a*c-b^2)) - u^2/(2*a) =
      (v-b*u/a)^2/(2*(c-b^2/a)) := by
  have hschur : c-b^2/a = (a*c-b^2)/a := by field_simp
  rw [hschur]
  field_simp [ha, hdet]
  field_simp [hdet]
  ring

theorem positive_schur (a b c : ℝ) (ha : 0 < a) (hd : 0 < a*c-b^2) :
    0 < c-b^2/a := by
  have heq : c-b^2/a = (a*c-b^2)/a := by field_simp
  rw [heq]
  exact div_pos hd ha

/-- SPD is needed: this indefinite quadratic tends downwards along a direction. -/
theorem negative_curvature_escape (t : ℝ) (ht : 1 < t) :
    scalarCurve 1 2 1 t (-t) < scalarCurve 1 2 1 1 (-1) := by
  unfold scalarCurve
  nlinarith [sq_nonneg (t-1)]

/-- A concrete SPD-coupled example violates diminishing marginal returns. -/
theorem marginal_gains_not_submodular :
    (1:ℝ)/2 < 1/(2*(1-(9/10:ℝ)^2)) := by
  norm_num

theorem coupled_counterexample_coercive (x y : ℝ) :
    (1/10:ℝ)*(x^2+y^2) ≤ scalarCurve 1 (9/10) 1 x y := by
  unfold scalarCurve
  nlinarith [sq_nonneg (x+y)]

section Perturbation
variable {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E]

/-- Known operator and gradient errors enlarge a residual certificate. -/
theorem residual_perturbation_bound (K Kobs : E →ₗ[ℝ] E) (g gobs s : E)
    (epsilon eta : ℝ)
    (gradient_error : ‖g-gobs‖ ≤ eta)
    (operator_error : ∀ v, ‖K v-Kobs v‖ ≤ epsilon*‖v‖) :
    ‖g+K s‖ ≤ ‖gobs+Kobs s‖ + epsilon*‖s‖ + eta := by
  have hdiff : g+K s = (gobs+Kobs s) + ((g-gobs)+(K s-Kobs s)) := by abel
  rw [hdiff]
  calc
    ‖gobs+Kobs s + ((g-gobs)+(K s-Kobs s))‖ ≤
        ‖gobs+Kobs s‖ + ‖(g-gobs)+(K s-Kobs s)‖ := norm_add_le _ _
    _ ≤ ‖gobs+Kobs s‖ + (‖g-gobs‖+‖K s-Kobs s‖) := by
      calc
        _ = ‖g-gobs + (K s-Kobs s)‖ + ‖gobs+Kobs s‖ := by ring
        _ ≤ ‖g-gobs‖ + ‖K s-Kobs s‖ + ‖gobs+Kobs s‖ := by
          simpa [add_comm, add_left_comm, add_assoc] using
            add_le_add_right (norm_add_le (g-gobs) (K s-Kobs s)) ‖gobs+Kobs s‖
        _ = ‖gobs+Kobs s‖ + (‖g-gobs‖+‖K s-Kobs s‖) := by ring
    _ ≤ ‖gobs+Kobs s‖ + (eta+epsilon*‖s‖) :=
      by linarith [operator_error s]
    _ = ‖gobs+Kobs s‖ + epsilon*‖s‖ + eta := by ring

end Perturbation

/-- Convert a remaining-gap certificate into a captured-decrease target. -/
theorem objective_certificate (decrease gap epsilon : ℝ) (hepsilon : epsilon < 1)
    (bound : gap ≤ epsilon/(1-epsilon)*decrease) :
    (1-epsilon)*(decrease+gap) ≤ decrease := by
  have hden : 0 < 1-epsilon := by linarith
  rw [div_mul_eq_mul_div] at bound
  have hmul := (le_div_iff₀ hden).mp bound
  nlinarith

#print axioms quadratic_gap
#print axioms subspace_optimality
#print axioms nested_subspace_monotone
#print axioms coercive_residual_bound
#print axioms schur_gain_identity
#print axioms scalar_same_vector_different
#print axioms scalar_polarization
#print axioms coordinate_scalar_blindness
#print axioms vector_first_column_identifies_coupling
#print axioms positive_schur
#print axioms negative_curvature_escape
#print axioms marginal_gains_not_submodular
#print axioms coupled_counterexample_coercive
#print axioms residual_perturbation_bound
#print axioms objective_certificate

end
end CurvatureCompass
