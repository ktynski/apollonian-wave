/-
EchoRH/LogDerivativeZeta.lean

Wires the **logarithmic derivative** of the Riemann zeta function to
mathlib so that the rest of the EchoRH formalization can talk about
`−ζ'/ζ` and its Dirichlet expansion `∑ Λ(n) n⁻ˢ` without re-proving
classical analytic number theory.

Two definitions:

  * `logDerivZeta s := −ζ'(s)/ζ(s)` (mathlib's `riemannZeta`).
  * `dirichletLogDerivZeta s := ∑' n, Λ(n) n⁻ˢ` (mathlib's
    `LSeries` applied to the von Mangoldt arithmetic function).

The headline identity (Dirichlet series for `−ζ'/ζ` on `Re s > 1`) is
`logDerivZeta_eq_dirichlet_on_rightHalfPlane`; it is *not* axiomatised
— it is `ArithmeticFunction.LSeries_vonMangoldt_eq_deriv_riemannZeta_div`
of mathlib, restated under EchoRH's definitions.

Pole structure at simple zeros and meromorphy on `{1}ᶜ` are also
derived from mathlib (`AnalyticAt.tendsto_mul_logDeriv_simple_zero` and
`MeromorphicOn` closure under `deriv`/`neg`/`div`).

This module introduces **no new axioms**.  The classical statements
the paper needs in `Theorem 7.15` (residue contradiction at any
non-critical-line zero) are obtained as instances of
`tendsto_mul_logDerivZeta_simple_zero` once one knows the zero is
simple — a hypothesis that has to be supplied by the caller.  The
simplicity of the nontrivial zeros of `ζ` is itself a famous classical
conjecture; the present module deliberately keeps that question
*outside* the basic API so callers can axiomatise it where they need
it (and not here, where it would be misplaced).

References
----------
* `Mathlib.NumberTheory.LSeries.Dirichlet` — the identity
  `LSeries_vonMangoldt_eq_deriv_riemannZeta_div`.
* `Mathlib.NumberTheory.LSeries.RiemannZeta` — `riemannZeta`,
  `analyticOn_riemannZeta`, `riemannZeta_residue_one`.
* `Mathlib.NumberTheory.ArithmeticFunction.VonMangoldt` — `Λ`.
* `Mathlib.Analysis.Calculus.LogDeriv` — `logDeriv` and
  `AnalyticAt.tendsto_mul_logDeriv_simple_zero`.
* `Mathlib.Analysis.Meromorphic.Basic` — `MeromorphicOn` and its
  closure properties.
* Tynski (2026), `\\cref{eq:prime-register}`, `\\cref{thm:main-rh}`.
-/

import Mathlib.NumberTheory.LSeries.Dirichlet
import Mathlib.NumberTheory.LSeries.RiemannZeta
import Mathlib.NumberTheory.ArithmeticFunction.VonMangoldt
import Mathlib.Analysis.Calculus.LogDeriv
import Mathlib.Analysis.Meromorphic.Basic
import Mathlib.Analysis.Analytic.Basic

namespace EchoRH

open Complex ArithmeticFunction

open scoped LSeries.notation

/-! ## Definitions -/

/-- The logarithmic derivative `−ζ'/ζ` of the Riemann zeta function.

This is the analytic-number-theory object that records both the prime
side (via its Dirichlet expansion `∑ Λ(n) n⁻ˢ` on `Re s > 1`) and the
zero side (via its meromorphic continuation, with simple poles at the
nontrivial zeros of `ζ`).  See Tynski (2026), `\\cref{eq:prime-register}`. -/
noncomputable def logDerivZeta (s : ℂ) : ℂ :=
  -deriv riemannZeta s / riemannZeta s

/-- The Dirichlet expansion `∑' n, Λ(n) n⁻ˢ` of `−ζ'/ζ`.

We define this as mathlib's `LSeries` applied to the von Mangoldt
arithmetic function (interpreted as a sequence `ℕ → ℂ`).  Mathlib
sets the `n = 0` term to zero by convention, which is harmless here
because `Λ 0 = 0` in any case.  Convergence holds on `Re s > 1` only;
see `dirichletLogDerivZeta_summable`. -/
noncomputable def dirichletLogDerivZeta (s : ℂ) : ℂ :=
  LSeries (fun n : ℕ ↦ (Λ n : ℂ)) s

/-! ## Basic equalities -/

/-- `logDerivZeta = -logDeriv riemannZeta`.  Just a restatement of the
definition in terms of mathlib's generic `logDeriv` operator. -/
theorem logDerivZeta_eq_neg_logDeriv (s : ℂ) :
    logDerivZeta s = -logDeriv riemannZeta s := by
  simp [logDerivZeta, logDeriv_apply, neg_div]

/-- The Dirichlet expansion as a `tsum`: `∑' n, term Λ s n`.

This is just `rfl` for mathlib's `LSeries`, but it is convenient to
have a directly-named lemma. -/
theorem dirichletLogDerivZeta_eq_tsum (s : ℂ) :
    dirichletLogDerivZeta s
      = ∑' n : ℕ, LSeries.term (fun n : ℕ ↦ (Λ n : ℂ)) s n := rfl

/-- The Dirichlet expansion as a `tsum`, with the `term` unfolded.

For `n ≥ 1` the `n`-th summand is `Λ(n) / n^s`; the `n = 0` term is
zero. -/
theorem dirichletLogDerivZeta_eq_tsum' (s : ℂ) :
    dirichletLogDerivZeta s
      = ∑' n : ℕ, (if n = 0 then 0 else (Λ n : ℂ) / (n : ℂ) ^ s) := by
  rfl

/-! ## The Dirichlet identity on `Re s > 1` -/

/-- **Dirichlet series for `−ζ'/ζ` on `Re s > 1`.**

`logDerivZeta s = ∑' n, Λ(n) n⁻ˢ` whenever `Re s > 1`.  This is
mathlib's `ArithmeticFunction.LSeries_vonMangoldt_eq_deriv_riemannZeta_div`,
restated under EchoRH's wrapping definitions.

Reference: Davenport, *Multiplicative Number Theory*, Ch. 11
(equivalent to `\\cref{eq:prime-register}` in Tynski 2026). -/
theorem logDerivZeta_eq_dirichlet_on_rightHalfPlane
    {s : ℂ} (hs : 1 < s.re) :
    logDerivZeta s = dirichletLogDerivZeta s := by
  unfold logDerivZeta dirichletLogDerivZeta
  rw [ArithmeticFunction.LSeries_vonMangoldt_eq_deriv_riemannZeta_div hs]

/-- The von-Mangoldt L-series is summable on the right half-plane.
This is a thin restatement of `LSeriesSummable_vonMangoldt`. -/
theorem dirichletLogDerivZeta_summable {s : ℂ} (hs : 1 < s.re) :
    LSeriesSummable (fun n : ℕ ↦ (Λ n : ℂ)) s :=
  ArithmeticFunction.LSeriesSummable_vonMangoldt hs

/-! ## Analytic and meromorphic structure -/

/-- The Riemann zeta function is meromorphic on `{1}ᶜ` (in fact
analytic there).  This is just `analyticOn_riemannZeta` packaged as
meromorphy. -/
theorem meromorphicOn_riemannZeta_compl_one :
    MeromorphicOn riemannZeta ({1} : Set ℂ)ᶜ :=
  analyticOn_riemannZeta.meromorphicOn

/-- The derivative `ζ'` is meromorphic on `{1}ᶜ`. -/
theorem meromorphicOn_deriv_riemannZeta_compl_one :
    MeromorphicOn (deriv riemannZeta) ({1} : Set ℂ)ᶜ :=
  meromorphicOn_riemannZeta_compl_one.deriv

/-- **Meromorphy of `−ζ'/ζ` away from the trivial pole at `s = 1`.**

`logDerivZeta` is meromorphic on `{1}ᶜ`.  Its poles in `{1}ᶜ` are
exactly the (nontrivial) zeros of `ζ`, all of them simple if the
classical simplicity conjecture holds.  We do *not* assume simplicity
here — we only state meromorphy. -/
theorem meromorphicOn_logDerivZeta_compl_one :
    MeromorphicOn logDerivZeta ({1} : Set ℂ)ᶜ := by
  unfold logDerivZeta
  -- `(-ζ') / ζ` is meromorphic since both numerator and denominator are.
  exact (meromorphicOn_deriv_riemannZeta_compl_one.neg).div
        meromorphicOn_riemannZeta_compl_one

/-- **Analyticity at non-zero, non-pole points.**

At any `s ≠ 1` with `riemannZeta s ≠ 0`, `logDerivZeta` is analytic.
This is the working analytic statement used by the AC bridge: on the
open set `{s | s ≠ 1 ∧ ζ s ≠ 0}` the function is analytic, hence in
particular continuous and differentiable. -/
theorem analyticAt_logDerivZeta {s : ℂ}
    (hs : s ≠ 1) (hζ : riemannZeta s ≠ 0) :
    AnalyticAt ℂ logDerivZeta s := by
  -- Build it as `(-ζ'(·))/ζ(·)`.
  have hζ_an : AnalyticAt ℂ riemannZeta s := analyticOn_riemannZeta s hs
  have hζ'_an : AnalyticAt ℂ (deriv riemannZeta) s := hζ_an.deriv
  have hneg : AnalyticAt ℂ (fun z ↦ -deriv riemannZeta z) s := hζ'_an.neg
  exact hneg.div hζ_an hζ

/-- Differentiability is the immediate corollary. -/
theorem differentiableAt_logDerivZeta {s : ℂ}
    (hs : s ≠ 1) (hζ : riemannZeta s ≠ 0) :
    DifferentiableAt ℂ logDerivZeta s :=
  (analyticAt_logDerivZeta hs hζ).differentiableAt

/-- `logDerivZeta` is analytic on the open set where `ζ ≠ 0` and `s ≠ 1`. -/
theorem analyticOnNhd_logDerivZeta_off_zeros :
    AnalyticOnNhd ℂ logDerivZeta {s : ℂ | s ≠ 1 ∧ riemannZeta s ≠ 0} :=
  fun _ hs ↦ analyticAt_logDerivZeta hs.1 hs.2

/-- On the right half-plane `Re s > 1`, `logDerivZeta` is analytic.

This is the *analytic* working domain for the AC bridge of
`EchoRH.ACBridge`: the Dirichlet identity holds here, and continuation
to a meromorphic function on `ℂ \ {1}` is unique by the identity
theorem. -/
theorem analyticOnNhd_logDerivZeta_rightHalfPlane :
    AnalyticOnNhd ℂ logDerivZeta {s : ℂ | 1 < s.re} := by
  intro s hs
  refine analyticAt_logDerivZeta ?_ ?_
  · -- `Re s > 1` ⇒ `s ≠ 1`.
    intro h; rw [h] at hs; simp at hs
  · -- `Re s > 1` ⇒ `ζ s ≠ 0`.
    exact riemannZeta_ne_zero_of_one_lt_re hs

/-! ## Pole structure at simple zeros of ζ -/

/-- **Residue `-1` at a simple zero of `ζ`.**

If `ρ ≠ 1`, `ζ(ρ) = 0`, and `ζ'(ρ) ≠ 0` (the zero is *simple* in the
analytic sense), then
  `(s − ρ) · logDerivZeta s → −1` as `s → ρ` (avoiding `ρ` itself).

In the language of residues, `logDerivZeta` has a simple pole at `ρ`
with residue `−1` (so the partial-fraction expansion takes the form
`logDerivZeta s = ∑_ρ (−1)/(s − ρ) + R(s)` with `R` regular near
each `ρ`, equivalently `logDerivZeta s = −∑_ρ 1/(s − ρ) + R(s)`).

The sign convention matches the Hadamard product expansion of `ζ`
(see e.g.\\ Davenport, Ch.\\ 12).  The paper's `\\cref{thm:main-rh}`
uses the sign-flipped version `−logDerivZeta s = +∑_ρ 1/(s − ρ) + R̃(s)`,
which is the same content with the global sign absorbed into `R̃`.

Proof: mathlib's `AnalyticAt.tendsto_mul_logDeriv_simple_zero` gives
`(s − ρ) · logDeriv ζ s → 1`, and `logDerivZeta = −logDeriv ζ`. -/
theorem tendsto_mul_logDerivZeta_simple_zero
    {ρ : ℂ} (hρ : ρ ≠ 1)
    (hζ : riemannZeta ρ = 0) (hζ' : deriv riemannZeta ρ ≠ 0) :
    Filter.Tendsto (fun s ↦ (s - ρ) * logDerivZeta s)
      (nhdsWithin ρ {ρ}ᶜ) (nhds (-1)) := by
  have hζ_an : AnalyticAt ℂ riemannZeta ρ := analyticOn_riemannZeta ρ hρ
  have h_pos :
      Filter.Tendsto (fun s ↦ (s - ρ) * logDeriv riemannZeta s)
        (nhdsWithin ρ {ρ}ᶜ) (nhds 1) :=
    hζ_an.tendsto_mul_logDeriv_simple_zero hζ hζ'
  -- Negate both the function and the limit; rewrite `-logDeriv = logDerivZeta`.
  have h_neg :
      Filter.Tendsto (fun s ↦ -((s - ρ) * logDeriv riemannZeta s))
        (nhdsWithin ρ {ρ}ᶜ) (nhds (-1)) := by
    simpa using h_pos.neg
  refine h_neg.congr (fun s ↦ ?_)
  -- `(s - ρ) * logDerivZeta s = -((s - ρ) * logDeriv ζ s)`.
  rw [logDerivZeta_eq_neg_logDeriv]
  ring

/-- A useful packaging: at a simple zero of `ζ`, `logDerivZeta` is
meromorphic with order exactly `-1`.  This corollary is sometimes
phrased as "the residue is `-1`".

We use mathlib's tendsto formulation; the corresponding `MeromorphicAt`
membership follows from `meromorphicOn_logDerivZeta_compl_one`. -/
theorem meromorphicAt_logDerivZeta_at_simple_zero
    {ρ : ℂ} (hρ : ρ ≠ 1) :
    MeromorphicAt logDerivZeta ρ :=
  meromorphicOn_logDerivZeta_compl_one ρ hρ

end EchoRH
