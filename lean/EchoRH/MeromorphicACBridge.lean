/-
EchoRH/MeromorphicACBridge.lean

The **meromorphic** analytic-continuation bridge.

`EchoRH.ACBridge` proves the *analytic* core: two functions analytic on
all of `ℂ` that agree on the right half-plane `{Re s > 1}` are equal
everywhere.

The function `-ζ'/ζ` is *meromorphic*, not analytic, so we need a
strengthened statement that allows for poles.  Two meromorphic functions
on `ℂ` that agree on the right half-plane agree as germs of meromorphic
functions everywhere — i.e. on a punctured neighbourhood of every point
they take equal values, and consequently they have identical pole
locations and identical pole orders.

Mathematical content
--------------------
Let `h := f - g`.  Since `f, g` are meromorphic on all of `ℂ`, so is
`h`.  The set `A = {z | AnalyticAt ℂ h z}` is open, and its complement
(the singular set of `h`) is countable
(`Meromorphic.countable_compl_analyticAt`).  The complement of any
countable subset of `ℂ` is path-connected
(`Set.Countable.isPathConnected_compl_of_one_lt_rank`, applied with
`Module.rank ℝ ℂ = 2`), hence `A` is preconnected.

`h` vanishes on the open right half-plane (which sits inside `A` because
`h` is locally zero, hence locally constant, there), so by the analytic
identity theorem `h = 0` on the whole preconnected open set `A`.

Finally, every `x : ℂ` has a punctured neighbourhood lying inside `A`
(`MeromorphicAt.eventually_analyticAt`, valid because `ℂ` is complete),
so `f =ᶠ[𝓝[≠] x] g`.

The pole-order equality `meromorphicOrderAt f x = meromorphicOrderAt g x`
then follows immediately from `meromorphicOrderAt_congr`.

References
----------
* `Mathlib.Analysis.Meromorphic.Basic` — `Meromorphic`, `MeromorphicOn`,
  `MeromorphicAt.eventually_analyticAt`, `Meromorphic.countable_compl_analyticAt`.
* `Mathlib.Analysis.Meromorphic.Order` — `meromorphicOrderAt_congr`.
* `Mathlib.Analysis.Analytic.IsolatedZeros` — analytic identity theorem
  `AnalyticOnNhd.eqOn_of_preconnected_of_eventuallyEq`.
* `Mathlib.Analysis.Normed.Module.Connected` — complement of a countable
  set is path-connected when the rank is `> 1`.
* `Mathlib.LinearAlgebra.Complex.FiniteDimensional` — `Module.rank ℝ ℂ = 2`.
-/

import Mathlib.Analysis.Meromorphic.Basic
import Mathlib.Analysis.Meromorphic.Order
import Mathlib.Analysis.Normed.Module.Connected
import Mathlib.LinearAlgebra.Complex.FiniteDimensional
import EchoRH.ACBridge

namespace EchoRH

open Complex Filter Topology Set

/-! ### Helper: complement of a countable subset of `ℂ` is preconnected -/

/-- The complement of a countable subset of `ℂ` is preconnected.  This
is an immediate consequence of `Set.Countable.isPathConnected_compl_of_one_lt_rank`
applied to `ℂ` viewed as a 2-dimensional real vector space. -/
theorem isPreconnected_compl_of_countable {S : Set ℂ} (hS : S.Countable) :
    IsPreconnected Sᶜ := by
  have h_rank : (1 : Cardinal) < Module.rank ℝ ℂ := by
    rw [Complex.rank_real_complex]
    exact_mod_cast Nat.one_lt_two
  exact (hS.isPathConnected_compl_of_one_lt_rank h_rank).isConnected.isPreconnected

/-! ### Meromorphic AC bridge -/

/-- **Meromorphic AC bridge.**  Two functions meromorphic on all of `ℂ`
that agree on the right half-plane `{Re s > 1}` agree on a punctured
neighbourhood of every point.

This is the meromorphic strengthening of `analytic_eq_of_eqOn_rightHalfPlane`.
The conclusion `f =ᶠ[𝓝[≠] x] g` is the strongest pointwise statement one
can hope for: at points where `f` and `g` have poles, their *values*
need not coincide (those values are not determined by the meromorphic
germ), but on every punctured neighbourhood they agree as functions.
In particular the pole sets and orders of `f` and `g` are identical
(see `meromorphicOrderAt_eq_of_eqOn_rightHalfPlane`). -/
theorem meromorphic_eqOn_punctured_of_eqOn_rightHalfPlane
    {f g : ℂ → ℂ}
    (hf : MeromorphicOn f Set.univ)
    (hg : MeromorphicOn g Set.univ)
    (h_eq : Set.EqOn f g rightHalfPlane) :
    ∀ x : ℂ, f =ᶠ[𝓝[≠] x] g := by
  intro x
  -- Work with the difference `h := f - g`, which is meromorphic on all of `ℂ`.
  have hh_on : MeromorphicOn (f - g) Set.univ := hf.sub hg
  have hh : Meromorphic (f - g) := meromorphicOn_univ.mp hh_on
  -- The analytic locus of `h`, denoted `A`.
  set A : Set ℂ := {z | AnalyticAt ℂ (f - g) z} with hA_def
  have hAc_count : Aᶜ.Countable := hh.countable_compl_analyticAt
  -- `A` is preconnected because its complement is countable in `ℂ` (rank 2).
  have hA_preconn : IsPreconnected A := by
    have hP := isPreconnected_compl_of_countable hAc_count
    rwa [compl_compl] at hP
  -- `h` is analytic on `A`, and the constant 0 is too.
  have hh_an : AnalyticOnNhd ℂ (f - g) A := fun _ hz => hz
  have h0_an : AnalyticOnNhd ℂ (fun _ : ℂ => (0 : ℂ)) A :=
    fun _ _ => analyticAt_const
  -- The right half-plane is contained in `A`, because `h = 0` on the
  -- open right half-plane and a function locally equal to a constant
  -- is analytic.
  have hRHP_sub : rightHalfPlane ⊆ A := by
    intro z hz
    show AnalyticAt ℂ (f - g) z
    have h_local : (fun _ : ℂ => (0 : ℂ)) =ᶠ[𝓝 z] (f - g) := by
      filter_upwards [rightHalfPlane_isOpen.mem_nhds hz] with w hw
      show (0 : ℂ) = f w - g w
      rw [h_eq hw]; ring
    exact (analyticAt_const : AnalyticAt ℂ (fun _ : ℂ => (0 : ℂ)) z).congr h_local
  -- Identity theorem: `h = 0` on the whole preconnected set `A`.
  obtain ⟨z₀, hz₀⟩ := rightHalfPlane_nonempty
  have hz₀_in : z₀ ∈ A := hRHP_sub hz₀
  have h_event_zero : (f - g) =ᶠ[𝓝 z₀] (fun _ : ℂ => (0 : ℂ)) := by
    filter_upwards [rightHalfPlane_isOpen.mem_nhds hz₀] with w hw
    show f w - g w = 0
    rw [h_eq hw]; ring
  have hh_zero : Set.EqOn (f - g) (fun _ : ℂ => (0 : ℂ)) A :=
    hh_an.eqOn_of_preconnected_of_eventuallyEq h0_an hA_preconn hz₀_in h_event_zero
  -- For arbitrary `x`, a punctured neighbourhood of `x` sits inside `A`
  -- (`MeromorphicAt.eventually_analyticAt`, completeness of `ℂ`).
  have h_event : ∀ᶠ y in 𝓝[≠] x, AnalyticAt ℂ (f - g) y :=
    (hh x).eventually_analyticAt
  filter_upwards [h_event] with y hy
  have hyA : y ∈ A := hy
  have hy_zero : (f - g) y = 0 := hh_zero hyA
  show f y = g y
  have hsub : f y - g y = 0 := hy_zero
  exact sub_eq_zero.mp hsub

/-! ### Pole-order equality (the load-bearing corollary) -/

/-- **Pole-order equality for the meromorphic AC bridge.**  Two
functions meromorphic on all of `ℂ` that agree on the right half-plane
have the same meromorphic order at every point.

`meromorphicOrderAt f x : ℤ ∪ {∞}` returns:
* `∞` if `f` is locally zero around `x`,
* `n : ℤ` otherwise, where `n < 0` means a pole of order `-n`,
  `n = 0` means analytic-and-nonzero (or a removable singularity with
  nonzero limit), and `n > 0` means a zero of order `n`.

So this theorem says: `f` and `g` have *exactly* the same poles, with
exactly the same orders, and the same zeros with the same orders.  This
is the load-bearing fact behind `Lemma ac-bridge` for the RH proof. -/
theorem meromorphicOrderAt_eq_of_eqOn_rightHalfPlane
    {f g : ℂ → ℂ}
    (hf : MeromorphicOn f Set.univ)
    (hg : MeromorphicOn g Set.univ)
    (h_eq : Set.EqOn f g rightHalfPlane) :
    ∀ x : ℂ, meromorphicOrderAt f x = meromorphicOrderAt g x := by
  intro x
  exact meromorphicOrderAt_congr
    (meromorphic_eqOn_punctured_of_eqOn_rightHalfPlane hf hg h_eq x)

end EchoRH

/-! ### Kernel-check documentation

The `#print axioms` directives below force Lean's kernel to verify that
the theorems above depend only on the standard classical foundations
(`propext`, `Classical.choice`, `Quot.sound`).  Any use of `sorry` or
non-standard axioms would manifest as an extra entry. -/

#print axioms EchoRH.isPreconnected_compl_of_countable
#print axioms EchoRH.meromorphic_eqOn_punctured_of_eqOn_rightHalfPlane
#print axioms EchoRH.meromorphicOrderAt_eq_of_eqOn_rightHalfPlane
