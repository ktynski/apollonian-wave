/-
EchoRH/ACBridge.lean

The analytic-continuation bridge (Lemma after Section 7.5 of the
paper):

> A graded operation that fixes the Dirichlet coefficients of
> -ζ'/ζ fixes -ζ'/ζ as a meromorphic function on ℂ, and in
> particular fixes its pole locations.

The paper's proof is one paragraph: two meromorphic functions on ℂ
that agree on the half-plane `Re s > 1` agree everywhere they are
both defined, by the identity theorem for analytic functions.

This file proves the analytic core of that statement in a form
that does **not** depend on the specific function `-ζ'/ζ`:

  `analytic_eq_of_eqOn_rightHalfPlane`: two functions analytic on
  all of `ℂ` (i.e. entire) that agree on the right half-plane are
  equal everywhere.

The wiring of this analytic statement to the *meromorphic* function
`-ζ'/ζ` and its Dirichlet expansion `Σ Λ(n) n⁻ˢ` will be done in a
separate module (`EchoRH.LogDerivativeZeta`) once the corresponding
mathlib infrastructure is wired in via the `PrimeNumberTheoremAnd`
project.

References
----------
* `Mathlib.Analysis.Analytic.IsolatedZeros` — identity theorem.
* Tynski (2026), `\\cref{lem:ac-bridge}`.
-/

import Mathlib.Data.Complex.Basic
import Mathlib.Analysis.Analytic.IsolatedZeros
import Mathlib.Analysis.Complex.Basic

namespace EchoRH

open Complex Filter Topology Set

/-- The right half-plane `{ s : ℂ | 1 < Re s }`. -/
def rightHalfPlane : Set ℂ := { s : ℂ | 1 < s.re }

theorem rightHalfPlane_isOpen : IsOpen rightHalfPlane :=
  isOpen_lt continuous_const Complex.continuous_re

theorem rightHalfPlane_nonempty : (rightHalfPlane).Nonempty :=
  ⟨2, by simp [rightHalfPlane]⟩

/-- The right half-plane is convex (a half-plane in `ℝ²`). -/
theorem rightHalfPlane_convex : Convex ℝ rightHalfPlane := by
  -- The map `ℂ → ℝ`, `z ↦ z.re`, is `ℝ`-linear; the half-plane
  -- is the preimage of the convex set `(1, ∞)` under this map.
  have h_eq : rightHalfPlane = Complex.reCLM ⁻¹' Set.Ioi 1 := rfl
  rw [h_eq]
  exact (convex_Ioi 1).linear_preimage Complex.reCLM.toLinearMap

/-- The right half-plane is preconnected. -/
theorem rightHalfPlane_isPreconnected : IsPreconnected rightHalfPlane :=
  rightHalfPlane_convex.isPreconnected

/-! ### Analytic AC bridge -/

/-- **Analytic AC bridge.**  Two functions analytic on all of `ℂ`
that agree on the right half-plane are equal everywhere.

This is the analytic core of `Lemma ac-bridge`.  The proof is a
direct application of the identity theorem for analytic functions
on the preconnected universal set `ℂ`.

The right half-plane is open and non-empty, so it contains a
neighbourhood of some point `z₀`.  On that neighbourhood `f = g`,
hence `f =ᶠ[𝓝 z₀] g`, hence (by the identity theorem) `f = g`
everywhere on the preconnected set `ℂ`. -/
theorem analytic_eq_of_eqOn_rightHalfPlane
    {f g : ℂ → ℂ}
    (hf : AnalyticOnNhd ℂ f Set.univ)
    (hg : AnalyticOnNhd ℂ g Set.univ)
    (h_eq : Set.EqOn f g rightHalfPlane) :
    f = g := by
  funext z
  -- Pick any z₀ in the right half-plane.
  obtain ⟨z₀, hz₀⟩ := rightHalfPlane_nonempty
  -- f =ᶠ[𝓝 z₀] g, because rightHalfPlane is open and z₀ is in it.
  have h_eventually : f =ᶠ[𝓝 z₀] g := by
    have h_mem : rightHalfPlane ∈ 𝓝 z₀ :=
      rightHalfPlane_isOpen.mem_nhds hz₀
    filter_upwards [h_mem] with w hw
    exact h_eq hw
  -- Use the identity theorem on the preconnected set ℂ.
  have h_eqOn : Set.EqOn f g Set.univ :=
    hf.eqOn_of_preconnected_of_eventuallyEq hg
      isPreconnected_univ
      (Set.mem_univ z₀)
      h_eventually
  exact h_eqOn (Set.mem_univ z)

end EchoRH
