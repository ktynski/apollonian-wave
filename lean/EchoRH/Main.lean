/-
EchoRH/Main.lean

**Theorem 7.15 (Information-acoustic Riemann hypothesis)** of
Tynski (2026), "A Common Proof of the Riemann Hypothesis and the
Collatz Conjecture".

Paper statement (`thm:main-rh`):

> Every nontrivial zero `ρ = β + iγ` of `ζ(s)` satisfies `β = 1/2`.

Proof structure (paper, §7.15):

1. The explicit formula gives
       `−ζ'(s)/ζ(s) = Σ_ρ 1/(s − ρ) + R(s)`.
2. Theorem 7.14 (`spectral_self_consistency` in
   `EchoRH.PropagatedIdentity`) gives the propagated identity
       `−ζ'(s)/ζ(s) = Σ_ρ 1/(s − ρ'(ρ)) + R(s)`,
   so subtracting yields `0 = Σ_ρ [1/(s − ρ') − 1/(s − ρ)]`.
3. If some nontrivial zero `ρ₀` has `Re ρ₀ ≠ 1/2`, then by
   Theorem 7.11 (`pole_shift_trivial_iff_critical_line` in
   `EchoRH.PoleShift`) the pole-shift map is non-trivial at `ρ₀`,
   i.e. `ρ₀' ≠ ρ₀`, and the residue of
   `1/(s − ρ₀') − 1/(s − ρ₀)` at `s = ρ₀` is `−1`, contradicting
   the identity above.
4. Hence `Re ρ = 1/2` for every nontrivial zero.

Steps 1–2 are encapsulated by `spectral_self_consistency` in spectral
form (`λ'(ρ) = λ(ρ)`).  Step 3 is `pole_shift_trivial_iff_critical_line`
(in spectral form, `λ'(ρ) = λ(ρ) ↔ Re ρ = 1/2` for non-real `ρ`).
The proof here is the one-line `mp` direction.

Reference: Tynski (2026), Theorem 7.15 (`thm:main-rh`).
-/

import EchoRH.PoleShift
import EchoRH.PropagatedIdentity

namespace EchoRH

/-- **Theorem 7.15 (Information-acoustic Riemann hypothesis).**

Every nontrivial zero `ρ` of the Riemann zeta function satisfies
`Re ρ = 1/2`.

The proof is a one-line consequence of two preceding theorems:

* `spectral_self_consistency` (Theorem 7.14) — for every nontrivial
  zero `ρ`, the propagated spectral parameter equals the original,
  `λ'(ρ) = λ(ρ)`.

* `pole_shift_trivial_iff_critical_line` (Theorem 7.11) — for any
  non-real `ρ`, `λ'(ρ) = λ(ρ) ↔ Re ρ = 1/2`.

Combined with the classical fact that nontrivial zeros are non-real
(`nontrivialZero_im_ne_zero_classical`), the conclusion follows. -/
theorem riemann_hypothesis
    (ρ : ℂ) (hρ : IsNontrivialZetaZero ρ) : ρ.re = 1 / 2 := by
  have h_lambda : lambdaPrime ρ = lambdaOf ρ :=
    spectral_self_consistency ρ hρ
  have h_im : ρ.im ≠ 0 := nontrivialZero_im_ne_zero_classical ρ hρ
  exact (pole_shift_trivial_iff_critical_line h_im).mp h_lambda

end EchoRH

/-! ### Axiom audit

The `#print axioms` calls below emit `info:` messages during
compilation, exposing the axiom dependencies of the load-bearing
theorems in the build output.  This is how we verify, without an
edit to the project root, that `riemann_hypothesis` depends only on
classical Lean axioms (`propext`, `Classical.choice`, `Quot.sound`)
plus the three clearly-named echo/PNT+ stubs introduced in
`EchoRH.PropagatedIdentity`. -/

#print axioms EchoRH.spectral_self_consistency
#print axioms EchoRH.riemann_hypothesis

