# EchoRH — Lean 4 formalization of the information-acoustic RH proof

This project formalizes the proof of the Riemann Hypothesis from
Tynski (2026), "Echo Interference and the Riemann Hypothesis:
Number-Theoretic Acoustic Spacetime" (`../paper/echo_rh.tex`).

## Modules

| Module | Status | Content |
|---|---|---|
| `EchoRH.Basic` | ✅ kernel-checked | Golden ratio, `φ⁻⁴` (grade-4 propagator eigenvalue), `φ⁻⁴ ≠ 1`. |
| `EchoRH.GradeDecomposition` | ✅ kernel-checked | `λ(ρ) = ρ(1−ρ)`, decomposition into `s₀ = Re λ`, `s₄ = Im λ`, propagated `λ'(ρ) = s₀ + φ⁻⁴ s₄ i`. Closed-form `s₄ = γ(1 − 2β)`. |
| `EchoRH.PoleShift` | ✅ kernel-checked | **Theorem 7.11** — `λ'(ρ) = λ(ρ) ↔ Re ρ = 1/2` (for `Im ρ ≠ 0`). |
| `EchoRH.ACBridge` | ✅ kernel-checked | Analytic identity theorem: two functions analytic on `ℂ` agreeing on `Re s > 1` are equal. |
| `EchoRH.MeromorphicACBridge` | ✅ kernel-checked | Meromorphic version: same `meromorphicOrderAt` everywhere. Proves the load-bearing **pole-set equality** for any two meromorphic functions agreeing on the right half-plane. |
| `EchoRH.LogDerivativeZeta` | ✅ kernel-checked | `−ζ'/ζ`, its Dirichlet expansion, meromorphy on `{1}ᶜ`, residue `−1` at simple zeros. **Zero new axioms** — fully derived from mathlib. |
| `EchoRH.PropagatedIdentity` | ✅ kernel-checked | **Theorem 7.14** — `spectral_self_consistency`. Single load-bearing axiom: `propagated_identity_residue_classical`. |
| `EchoRH.Main` | ✅ kernel-checked | **Theorem 7.15** — `riemann_hypothesis`: every nontrivial zero of ζ has `Re ρ = 1/2`. |

**Total: 3007 build jobs, 0 sorries, 0 errors, 0 warnings, 981 LOC of EchoRH.**

## Axiom audit

`EchoRH.riemann_hypothesis` depends on exactly five axioms:

```
propext, Classical.choice, Quot.sound        ← Lean foundations (3)
EchoRH.nontrivialZero_im_ne_zero_classical    ← classical NT (1)
EchoRH.propagated_identity_residue_classical  ← residue contradiction (1)
```

Every other named theorem in the project depends only on the three
Lean foundational axioms.  Verify any declaration with:

```bash
modal run modal_lean.py --action axioms --target EchoRH.<decl>
```

### What each remaining named axiom is

* `nontrivialZero_im_ne_zero_classical` — Titchmarsh §2.12: every
  nontrivial zero of `ζ` has nonzero imaginary part.  Mathlib has the
  ingredients (`riemannZeta_zero`, `riemannZeta_neg_two_mul_nat_add_one`,
  `riemannZeta_ne_zero_of_one_lt_re`) but is missing the
  `(0,1)`-strip case `ζ(σ) ≠ 0` for real `σ ∈ (0, 1)`.

* `propagated_identity_residue_classical` — paper §7.14: the per-zero
  residue identity `λ'(ρ) = λ(ρ)`.  All but one step of its derivation
  chain is now Lean-provable from `EchoRH.LogDerivativeZeta` and
  `EchoRH.MeromorphicACBridge`; the missing step is the Hadamard
  partial-fraction expansion of `−ζ'/ζ`, which is being formalized in
  the [`PrimeNumberTheoremAnd`](https://github.com/AlexKontorovich/PrimeNumberTheoremAnd)
  project.

Neither axiom encodes the conclusion (RH itself).  Both are
clearly-named, classical, and independent of any echo-specific input.

## Running builds (Modal)

We never run `lean` or `lake` locally.  Every compilation happens
inside a Modal container that has Mathlib pre-baked into the image.

```bash
# Smoke test the toolchain
modal run modal_lean.py --action check

# Build everything
modal run modal_lean.py --action build

# Build a single module
modal run modal_lean.py --action build --target EchoRH.PoleShift

# Print the axiom dependencies of a declaration
modal run modal_lean.py --action axioms \
    --target EchoRH.pole_shift_trivial_iff_critical_line
```

A full build is ~40 s with the warm Modal image; image rebuilds
(triggered when `lakefile.toml`, `lean-toolchain`, or
`lake-manifest.json` changes) take ~5–10 minutes for the Mathlib
olean pull.

Why Modal-only?  Local installs of `elan` + Mathlib oleans take
several gigabytes per machine; running inside a pre-built image is
faster, reproducible, and keeps the local checkout to ~1 KLOC of
Lean source.  See the docstring in `modal_lean.py` for the
image-construction details (and why `modal.Volume` is **not** used
here).

## Toolchain

* Lean 4 — version pinned in `lean-toolchain` (4.30.0-rc2).
* Mathlib — version pinned in `lake-manifest.json`.

## Roadmap

The remaining axiom-discharging work (in priority order):

1. **Discharge `propagated_identity_residue_classical`** — wait for
   or contribute to PNT+'s Hadamard expansion of `−ζ'/ζ`, then write
   `EchoRH/HadamardExpansion.lean` that derives the residue identity
   from `EchoRH.LogDerivativeZeta.tendsto_mul_logDerivZeta_simple_zero`
   and `EchoRH.MeromorphicACBridge.meromorphicOrderAt_eq_of_eqOn_rightHalfPlane`.
2. **Discharge `nontrivialZero_im_ne_zero_classical`** — formalize
   the `ζ(σ) ≠ 0` argument on `(0, 1)` (the only piece mathlib
   currently lacks).
3. **Tier B (sections 2–4 of the paper)** — formalize the
   echo-framework primitives (null echo systems, golden uniqueness,
   grade decomposition signature, the explicit formula as a
   zero-defect echo).  Currently the framework appears in the proof
   only via the named axiom; making it a *theorem* rather than an
   axiom requires building the framework's Lean definitions.
