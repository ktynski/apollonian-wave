/-
EchoRH/Basic.lean

Foundational definitions for the information-acoustic RH proof.

This module sets up the golden ratio (φ), its inverse fourth power
(φ⁻⁴, the grade-4 propagator eigenvalue), and basic numeric facts
that are used throughout.

Reference: Tynski, "A Common Proof of the Riemann Hypothesis and the
Collatz Conjecture: Echo Interference in Number-Theoretic Acoustic
Spacetime" (2026), §§2-7.
-/

import Mathlib.NumberTheory.Real.GoldenRatio

open Real goldenRatio

namespace EchoRH

/-- The grade-4 propagator eigenvalue, `φ⁻⁴`.

This is the contraction factor applied by the echo propagator to the
pseudoscalar component of the Laplacian eigenvalue `λ(ρ) = ρ(1-ρ)`.
By `Theorem 2.19` of the paper (zero-defect attenuation), grade-`k`
content attenuates by `φ^(-k)` per closure cycle, with the
pseudoscalar living at grade 4 in `G(3,1)` (`Theorem 2.26`). -/
noncomputable abbrev gradeFourEigenvalue : ℝ := φ⁻¹ ^ 4

@[inherit_doc]
scoped notation "φ⁻⁴" => EchoRH.gradeFourEigenvalue

/-! ### Basic positivity and bounds for `φ⁻⁴` -/

theorem gradeFourEigenvalue_pos : 0 < φ⁻⁴ := by
  unfold gradeFourEigenvalue
  positivity

theorem gradeFourEigenvalue_lt_one : φ⁻⁴ < 1 := by
  unfold gradeFourEigenvalue
  have hφ : 1 < φ := one_lt_goldenRatio
  have hφ_pos : 0 < φ := lt_trans zero_lt_one hφ
  have hφinv_pos : 0 < φ⁻¹ := inv_pos.mpr hφ_pos
  have hφinv_lt : φ⁻¹ < 1 := by
    rw [inv_lt_one_iff₀]
    exact Or.inr hφ
  calc (φ⁻¹) ^ 4
      < 1 ^ 4 := by
        apply pow_lt_pow_left₀ hφinv_lt (le_of_lt hφinv_pos)
        decide
    _ = 1 := one_pow 4

/-- The grade-4 eigenvalue is not equal to 1.  This is the load-bearing
fact for `Theorem 7.11` (pole shift trivial iff on the critical line):
the contraction is non-trivial, so a fixed point of the contraction on
the imaginary axis must have zero imaginary part. -/
theorem gradeFourEigenvalue_ne_one : φ⁻⁴ ≠ 1 :=
  ne_of_lt gradeFourEigenvalue_lt_one

end EchoRH
