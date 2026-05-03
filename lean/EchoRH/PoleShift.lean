/-
EchoRH/PoleShift.lean

Theorem 7.11: the pole-shift map `ρ ↦ ρ'` is trivial (i.e.
`λ'(ρ) = λ(ρ)`) for a non-real `ρ` if and only if `Re ρ = 1/2`.

The paper writes the pole-shift map directly as
  `ρ' = (1 + √(1 - 4λ')) / 2`
(with `Im ρ' > 0`).  But the load-bearing logical content does not
need that branch choice: it suffices to prove the equivalent
*spectral* statement that `λ'(ρ) = λ(ρ)`.

Concretely, by `Definition 7.10` the propagated eigenvalue is
  `λ'(ρ) = s₀(ρ) + φ⁻⁴ · s₄(ρ) · i`,
so
  `λ'(ρ) = λ(ρ)`
iff the grade-4 (imaginary) part is fixed under multiplication by
`φ⁻⁴`, iff `s₄(ρ) = 0` (since `φ⁻⁴ ≠ 1`), iff (for non-real `ρ`)
`Re ρ = 1/2`.

This file proves all four equivalences.

Reference: Tynski (2026), Theorem 7.11.
-/

import EchoRH.GradeDecomposition

namespace EchoRH

open Complex

/-! ### `λ'(ρ) = λ(ρ)` iff `s₄(ρ) = 0` -/

/-- Both `λ` and `λ'` agree on their grade-0 (real) part by
construction.  They differ only in the grade-4 (imaginary) part:
`λ` has `s₄(ρ)`, while `λ'` has `φ⁻⁴ · s₄(ρ)`. -/
theorem lambdaPrime_eq_lambda_iff_s4_eq_zero (ρ : ℂ) :
    lambdaPrime ρ = lambdaOf ρ ↔ s4 ρ = 0 := by
  -- Unfold definitions, split on (re, im).  Real parts agree by
  -- definition; the imaginary equation reduces to `φ⁻⁴ · s₄ = s₄`.
  rw [lambdaPrime, Complex.ext_iff]
  simp only [s0, true_and]
  change φ⁻⁴ * s4 ρ = s4 ρ ↔ s4 ρ = 0
  constructor
  · intro h
    -- (φ⁻⁴ - 1) · s₄ = 0; since φ⁻⁴ ≠ 1, we get s₄ = 0.
    have hsub : (φ⁻⁴ - 1) * s4 ρ = 0 := by linarith
    rcases mul_eq_zero.mp hsub with h1 | h2
    · exact absurd (by linarith : φ⁻⁴ = 1) gradeFourEigenvalue_ne_one
    · exact h2
  · intro hs4
    rw [hs4, mul_zero]

/-! ### Theorem 7.11: pole shift is trivial iff on the critical line -/

/-- **Theorem 7.11 (pole shift trivial iff critical line).**

For any non-real complex number `ρ` (i.e. `Im ρ ≠ 0` -- the
"non-trivial zero" condition `γ ≠ 0` in the paper's notation),
the pole-shift map fixes `ρ` if and only if `Re ρ = 1/2`.

We state this as the *spectral* equivalence `λ'(ρ) = λ(ρ)`, which
is in turn equivalent to `ρ' = ρ` once a square-root branch for the
inverse map `λ ↦ ρ` is chosen (cf. Definition 7.10 in the paper).
-/
theorem pole_shift_trivial_iff_critical_line {ρ : ℂ} (hγ : ρ.im ≠ 0) :
    lambdaPrime ρ = lambdaOf ρ ↔ ρ.re = 1 / 2 := by
  rw [lambdaPrime_eq_lambda_iff_s4_eq_zero, s4_eq_zero_iff]
  constructor
  · rintro (h | h)
    · exact absurd h hγ
    · exact h
  · intro h
    exact Or.inr h

end EchoRH
