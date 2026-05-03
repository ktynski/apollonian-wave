/-
EchoRH/GradeDecomposition.lean

Grade decomposition of the spectral parameter λ(ρ) = ρ(1-ρ).

For ρ = β + iγ, the Laplacian eigenvalue λ(ρ) decomposes into:
  * a *grade-0* (scalar) component  s₀(ρ) := Re(λ(ρ)) = β(1-β) + γ²
  * a *grade-4* (pseudoscalar) component  s₄(ρ) := Im(λ(ρ)) = γ(1-2β)

In the paper's `G(3,1)` framing (Section 2), `s₀` lives at grade 0
and `s₄` lives at grade 4 (the pseudoscalar).  This file gives the
simple, identification-friendly definitions; the algebraic structure
of `G(3,1)` itself is not needed for the RH argument.

The propagator `E_φ` acts on `λ` by attenuating the grade-4 part by
`φ⁻⁴`:
  λ'(ρ) := s₀(ρ) + φ⁻⁴ · s₄(ρ) · i.

Reference: Tynski (2026), Definition 7.10.
-/

import Mathlib.Analysis.SpecialFunctions.Pow.Real
import Mathlib.Data.Complex.Basic
import EchoRH.Basic

open Real goldenRatio Complex

namespace EchoRH

/-- The spectral parameter `λ(ρ) := ρ(1-ρ)`.  This is the Laplacian
eigenvalue associated to a putative zeta zero at `ρ`. -/
noncomputable def lambdaOf (ρ : ℂ) : ℂ := ρ * (1 - ρ)

/-- Grade-0 (scalar) component of `λ(ρ)`. -/
noncomputable def s0 (ρ : ℂ) : ℝ := (lambdaOf ρ).re

/-- Grade-4 (pseudoscalar) component of `λ(ρ)`. -/
noncomputable def s4 (ρ : ℂ) : ℝ := (lambdaOf ρ).im

/-- The propagated eigenvalue: the propagator `E_φ` attenuates the
pseudoscalar (grade-4) part by `φ⁻⁴` and leaves the scalar (grade-0)
part fixed. -/
noncomputable def lambdaPrime (ρ : ℂ) : ℂ :=
  ⟨s0 ρ, φ⁻⁴ * s4 ρ⟩

/-! ### Closed-form expressions for `s₀` and `s₄` in `(β, γ)` coords -/

/-- For `ρ = β + iγ`, we have `s₀(ρ) = β(1-β) + γ²`. -/
theorem s0_eq (ρ : ℂ) : s0 ρ = ρ.re * (1 - ρ.re) + ρ.im ^ 2 := by
  -- ρ(1-ρ) = (β+iγ)(1-β-iγ); real part = β(1-β) - (-γ²) = β(1-β) + γ²
  simp only [s0, lambdaOf, Complex.mul_re, Complex.sub_re, Complex.sub_im,
    Complex.one_re, Complex.one_im]
  ring

/-- For `ρ = β + iγ`, we have `s₄(ρ) = γ(1 - 2β)`. -/
theorem s4_eq (ρ : ℂ) : s4 ρ = ρ.im * (1 - 2 * ρ.re) := by
  -- ρ(1-ρ) = (β+iγ)(1-β-iγ); imaginary part = γ(1-β) - βγ = γ(1 - 2β)
  simp only [s4, lambdaOf, Complex.mul_im, Complex.sub_re, Complex.sub_im,
    Complex.one_re, Complex.one_im]
  ring

/-! ### The grade-4 piece vanishes iff on the critical line (or real) -/

/-- `s₄(ρ) = 0` exactly when `ρ` is on the real axis or on the
critical line `Re ρ = 1/2`. -/
theorem s4_eq_zero_iff (ρ : ℂ) :
    s4 ρ = 0 ↔ ρ.im = 0 ∨ ρ.re = 1 / 2 := by
  rw [s4_eq, mul_eq_zero]
  constructor
  · rintro (h | h)
    · exact Or.inl h
    · refine Or.inr ?_
      -- 1 - 2 * ρ.re = 0 ⟹ ρ.re = 1/2
      linarith
  · rintro (h | h)
    · exact Or.inl h
    · refine Or.inr ?_
      -- ρ.re = 1/2 ⟹ 1 - 2 * (1/2) = 0
      linarith

end EchoRH
