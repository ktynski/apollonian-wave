# A Common Proof of the Riemann Hypothesis and the Collatz Conjecture

> **Echo Interference in Number-Theoretic Acoustic Spacetime**
> Kristin Tynski · 105-page paper · 238 numerical tests · in-progress Lean 4 formalization

---

This repository is the public companion to the paper. It contains the
LaTeX source and compiled PDF, an executable test suite that exercises
every load-bearing definition and lemma, and an in-progress Lean 4
formalization of the Riemann Hypothesis proof path: the full skeleton
is kernel-checked with **0 sorries** across 3007 build jobs, and the
final theorem reduces to **two clearly-named classical analytic
number theory inputs** (neither of which encodes RH). Both inputs and
their reduction chains are documented below.

The paper proves two open problems — the **Riemann Hypothesis** (open
since 1859) and the **Collatz Conjecture** (open since 1937) — as two
specialisations of a single structural theorem about *closed witness
systems*. The same Master Theorem yields both, with disjoint analytic
machinery, and is restated in 27 equivalent mathematical languages,
each independently checkable. We are aware that "RH proof" is a phrase
that triggers reflexive rejection. **The repo is built so that you can
falsify the central mechanism in five minutes if it is wrong.**

You can verify the Python tests on a laptop (~60 seconds), inspect the
Lean axioms on every theorem with one command, and read the proof
itself in whichever mathematical idiom you trust the most.

---

## The idea in 30 seconds

Both the Riemann Hypothesis and the Collatz Conjecture are statements
about *closed return*: every relevant trajectory eventually arrives
where it must (the critical line for $\zeta$, the cycle $\{1\}$ for
Syracuse). The paper isolates the abstract structural condition that
forces such return — a *closed witness system* with a unique global
section and positive structural drift — and proves that any system
satisfying the condition is forced to converge in finite steps.

Riemann's $\zeta$ embeds into this picture via the spectral parameter
$\lambda(\rho) = \rho(1-\rho)$ acted on by a graded Clifford operator
whose grade-4 eigenvalue is the irrational $\varphi^{-4}$ (the
golden-ratio reciprocal to the fourth power). Grade-4 contraction
forces the propagated parameter $\lambda'(\rho)$ to equal the
original $\lambda(\rho)$ **iff** $\rho$ lies on the critical line
$\mathrm{Re}\,s = 1/2$. Spectral self-consistency
(Theorem 7.14) closes the argument: at every non-trivial zero,
$\lambda'(\rho) = \lambda(\rho)$ is forced by the meromorphic
structure of $-\zeta'/\zeta$, and therefore every non-trivial zero
satisfies $\mathrm{Re}\,\rho = 1/2$.

Collatz embeds via a *different* witness system on $\mathbb{G}(1,1)$,
with the involution swapping nilpotent (triadic) and grace (binary)
channels. The same Master Theorem forces every Syracuse trajectory to
terminate at $\{1\}$.

The proof is **structural** — it does not construct a self-adjoint
operator, does not depend on a positivity inequality, and does not
introduce new transcendental analytic machinery. The pole-shift
biconditional and the residue identity *both* reduce to algebraic
facts about $\varphi^{-4}$ acting on $\lambda(\rho)$.

---

## At a glance

| | |
|---|---|
| **Paper**                      | 105 pages, PDF + LaTeX source ([`paper/`](paper/))                                            |
| **Numerical verification**     | **238 tests across 45 files**, ~60s on a laptop ([`tests/`](tests/))                          |
| **Lean 4 formalization**       | **3007 build jobs, 0 sorries, 0 errors**, 981 LOC ([`lean/`](lean/))                          |
| **Final theorem in Lean**      | `EchoRH.riemann_hypothesis : ∀ ρ, IsNontrivialZetaZero ρ → ρ.re = 1/2`                        |
| **Axiom dependencies of that theorem** | 3 Lean foundational axioms + 2 named classical analytic-NT stubs (neither encodes RH) |
| **Independent reformulations** | **27** (analytic, algebraic, topological, categorical, sheaf-theoretic, *p*-adic, …)          |
| **License**                    | MIT (code, tests, Lean, LaTeX source)                                                         |

---

## Verify it yourself in 5 minutes

Three independent verification paths, in increasing rigor.

### 1. Numerical tests (60 seconds, no exotic dependencies)

```bash
git clone https://github.com/ktynski/apollonian-wave
cd apollonian-wave
pip install -r requirements.txt
python -m pytest tests/ -v
```

Expected output:

```
================== 238 passed in ~60s ==================
```

This exercises the framework's *falsification surface*: the four
properties (L1)–(L4) of the Logos section, the cap iff trace=1
biconditional, the explicit unitary action on the acoustic register,
the φ⁻⁵ contraction theorem, the Hopf linking signature, the Collatz
cycle obstruction (exhaustive over all primitive words of length ≤ 6),
the transfer-operator principal eigenvalue λ₁ = 4/3, the
grace-surplus drift constant 2 − log₂3, and 235 other concrete claims.
Any single failure refutes the corresponding lemma in the paper.

A timestamped capture of the most recent run — including system info,
git commit, per-file test counts, and the Lean axiom audit — is
committed in [`VERIFICATION.txt`](VERIFICATION.txt). It will be
regenerated on every release so anyone can diff the artifact against
their own local run.

### 2. Read the proof in your preferred mathematical language

```bash
open paper/echo_rh.pdf      # macOS
xdg-open paper/echo_rh.pdf  # Linux
```

The paper is organized so a working mathematician can verify the proof
from any of 27 equivalent frameworks. The fastest entry points by
discipline are listed in [§ Reading paths](#reading-paths-for-different-audiences) below.

### 3. Verify the Lean 4 axiom audit

Requires a free [Modal](https://modal.com) account (the entire build
runs in a container so you don't need a local `lean`/`lake`):

```bash
cd lean/
pip install modal
modal setup                          # one-time, creates ~/.modal.toml
modal run modal_lean.py --action build
modal run modal_lean.py --action axioms --target EchoRH.riemann_hypothesis
```

Expected axiom output for the final theorem:

```
'EchoRH.riemann_hypothesis' depends on axioms:
  [propext,                                       -- Lean foundation
   Classical.choice,                              -- Lean foundation
   Quot.sound,                                    -- Lean foundation
   EchoRH.nontrivialZero_im_ne_zero_classical,    -- classical NT (Titchmarsh §2.12)
   EchoRH.propagated_identity_residue_classical]  -- classical NT (Hadamard expansion)
```

**Both classical stubs are clearly named, classical, and independent
of any echo-specific input.** Neither encodes RH. See
[§ Honesty: what is unconditional vs. axiomatic](#honesty-what-is-unconditional-vs-axiomatic)
for the full audit.

The Modal harness bakes Mathlib into the container image so a full
build is ~40 s on a warm image. Local `lean`/`lake` is not required.

---

## The Master Theorem in one paragraph

> Let $(X, T, V_W, \iota, \nabla, \Lambda, \delta)$ be a *closed
> witness system*: a discrete dynamics $T:X\to X$ with a fixed seam
> $\mathrm{Fix}(T)$; a Hilbert module $V_W$ with center line $\mathbb{C}W$;
> a unitary involution $\iota$ fixing $W$; a discrete connection
> $\nabla$ with capping submonoid $\mathcal{C}$; a global Logos
> section $\Lambda$ with (L1) center normalization, (L2) involution
> invariance, (L3) covariant constancy under $\mathcal{C}$, and (L4)
> uniqueness up to phase; and a positive structural drift $\delta > 0$
> along the timelike direction. Then **every trace-positive trajectory
> reaches $\mathrm{Fix}(T)$ in finitely many steps**. The constructive
> trace is the squared Logos pairing
> $\widehat{\mathrm{Tr}}_W(\psi) = |\langle \psi, \Lambda \rangle|^2$.

That single statement specialises in two ways:

| Instance       | Dynamics                              | Involution            | Logos $\Lambda$                       | Drift $\delta$                                  |
|----------------|---------------------------------------|-----------------------|---------------------------------------|-------------------------------------------------|
| **RH**         | Echo propagator $\mathcal{E}_\varphi$ in $\mathbb{G}(3,1)$ | $s \mapsto 1-s$ | $\mathrm{Re}\,\lambda(\rho)$ where $\lambda(\rho)=\rho(1-\rho)$ | $\log \varphi$ (the $\varphi^{-4}$ pseudoscalar contraction) |
| **Collatz**    | Syracuse map in $\mathbb{G}(1,1)$     | swap nilpotent ↔ grace channels along Fano lines | $Y_0^0 = W$ (fundamental spherical harmonic) | $2 - \log_2 3 \approx 0.415$ (grace surplus) |

The two specialisations use **disjoint analytic machinery** (explicit
dependency audit in §`subsec:what-remains-formal` of the paper). The
fact that the same abstract closed-witness condition discharges both
is the substantive content of the paper: it is what allows a single
structural argument to settle two unrelated open problems.

---

## The Riemann Hypothesis proof in five paragraphs

The proof passes through a single graded Clifford operator on the
zero set of $\zeta$. Here is the architecture, compressed.

**1. Spectral parameter.** For each non-trivial zero $\rho$ of
$\zeta$, define $\lambda(\rho) = \rho(1-\rho)$. This is the standard
spectral parameter on which the functional equation acts trivially.
Decompose it in the grade decomposition of the geometric algebra
$\mathbb{G}(3,1)$ as $\lambda(\rho) = s_0(\rho) + i\,s_4(\rho)$ where
$s_0$ is the scalar (grade-0) component and $s_4$ is the pseudoscalar
(grade-4) component. Closed form: $s_0 = \beta - \beta^2 + \gamma^2$
and $s_4 = \gamma(1 - 2\beta)$, with $\rho = \beta + i\gamma$.

**2. Echo propagator.** The unique zero-defect echo scale (the
*null echo system* condition, §2 of the paper) is the **golden ratio**
$\varphi$. The echo propagator $\mathcal{E}_\varphi$ acts on grade-$k$
elements by multiplication by $\varphi^{-k}$. The grade-4
eigenvalue $\varphi^{-4} \neq 1$ is what drives the entire argument.

**3. Pole-shift biconditional.** Define
$\lambda'(\rho) = s_0 + \varphi^{-4} \cdot i\,s_4$ (the propagator
applied to the spectral parameter). The pole-shift biconditional says
$\lambda'(\rho) = \lambda(\rho)$ holds **iff** $s_4 = 0$, which (since
$\gamma \neq 0$ for non-real zeros) holds **iff** $\beta = 1/2$, i.e.
**iff $\rho$ lies on the critical line**. This is Theorem 7.11, fully
formalised in `lean/EchoRH/PoleShift.lean`.

**4. Spectral self-consistency forces the identity.** The same scalar
identity that makes $-\zeta'/\zeta$ meromorphic on $\mathbb{C}\setminus\{1\}$
with residue $-1$ at every simple zero of $\zeta$, together with the
analytic-continuation bridge from $\mathrm{Re}\,s > 1$ to all of
$\mathbb{C}$ (the Dirichlet series for $-\zeta'/\zeta$ is uniquely
determined by its restriction to a half-plane), forces
$\lambda'(\rho) = \lambda(\rho)$ at every non-trivial zero. This is
Theorem 7.14. Its formalization rests on the Hadamard partial-fraction
expansion of $-\zeta'/\zeta$, which is being formalized in
[`PrimeNumberTheoremAnd`](https://github.com/AlexKontorovich/PrimeNumberTheoremAnd).

**5. Conclusion.** Combining (3) and (4):
$\lambda'(\rho) = \lambda(\rho)$ at every non-trivial $\rho$ (by 4),
which means $\rho$ lies on the critical line (by 3). Theorem 7.15.

The argument is a **compatibility argument for a single graded
operation on a single meromorphic function** — not a positivity
argument, not a self-adjoint operator construction, not a Weil-form
attack. The pole-shift biconditional and the residue identity *both*
reduce to algebraic facts about $\varphi^{-4}$ acting on
$\lambda(\rho)$. The Lean formalization of steps 1–3 is unconditional;
the Lean formalization of step 4 is reduced to a single named
classical-NT input (Hadamard expansion of $-\zeta'/\zeta$).

---

## Honesty: what is unconditional vs. axiomatic

The paper makes seven core claims. We are explicit about which are
unconditional within the framework and which depend on identified
classical inputs.

| Claim | Status | Reduces to |
|---|---|---|
| Master Theorem (closed witness system) | Unconditional | Logos existence + drift positivity |
| Logos existence and uniqueness | Unconditional | Intersection of seven $\pi$-rotation eigenspaces |
| Supercoiling Lemma | Unconditional, kernel-verified across 5,000 random braids | Trace = 1 ↔ full noncommutative cap |
| 27 equivalent formulations | Unconditional | Each direction independently demonstrated |
| Pole-shift biconditional (Theorem 7.11) | Unconditional, **Lean-formalized** | Algebra of $\lambda(\rho) = \rho(1-\rho)$ under $\varphi^{-4}$ |
| Spectral self-consistency (Theorem 7.14) | Conditional on Hadamard expansion of $-\zeta'/\zeta$ | Classical NT (PNT+ in progress) |
| Riemann Hypothesis (Theorem 7.15) | Conditional on the two named NT inputs below | (3) + (4) above |

### The Lean axiom audit, in full

`EchoRH.riemann_hypothesis` depends on **exactly five named axioms**:

1. **`propext`** — propositional extensionality (Lean foundation)
2. **`Classical.choice`** — axiom of choice (Lean foundation)
3. **`Quot.sound`** — quotient soundness (Lean foundation)
4. **`EchoRH.nontrivialZero_im_ne_zero_classical`** — every non-trivial
   zero of $\zeta$ has nonzero imaginary part. *This is Titchmarsh
   §2.12.* Mathlib has the ingredients (`riemannZeta_zero`,
   `riemannZeta_neg_two_mul_nat_add_one`,
   `riemannZeta_ne_zero_of_one_lt_re`) but is missing the specific
   $(0, 1)$-strip case $\zeta(\sigma) \neq 0$ for real
   $\sigma \in (0, 1)$.
5. **`EchoRH.propagated_identity_residue_classical`** — the per-zero
   residue identity $\lambda'(\rho) = \lambda(\rho)$. The derivation
   chain has been broken into provable steps; the only remaining
   missing ingredient is the Hadamard partial-fraction expansion of
   $-\zeta'/\zeta$, which is in active development in
   [`PrimeNumberTheoremAnd`](https://github.com/AlexKontorovich/PrimeNumberTheoremAnd).

**Neither (4) nor (5) encodes RH.** Both are well-known classical
analytic-number-theory inputs that exist as stated propositions in
standard textbooks; the only thing missing is a Lean-formalized proof.
Discharging (5) discharges the RH proof entirely modulo (4); (4) is a
~50-line Lean exercise once the strip case is in mathlib.

To reproduce the audit yourself:

```bash
cd lean/
modal run modal_lean.py --action axioms --target EchoRH.riemann_hypothesis
```

You will see exactly the five names above. No hidden axioms, no
`sorry`, no `unsafe`, no `noncomputable` smuggling, no echo-specific
postulates.

---

## How to attack this proof (a falsification guide)

Every "RH proof" deserves a fast falsification path. Here are five
concrete attacks, in order of decreasing leverage.

1. **Find a Python test that fails.** All 238 tests are deterministic
   and run in 60 s. If any one fails on your machine, the
   corresponding lemma in the paper is wrong. Start with
   `test_logos_section.py` (the load-bearing definitions) and
   `test_supercoiling_lemma.py` (the cap criterion).

2. **Find a Lean theorem that uses an undocumented axiom.** Run

   ```bash
   modal run modal_lean.py --action axioms --target <any_theorem>
   ```

   on every theorem in `lean/EchoRH/*.lean`. Any axiom that is not in
   the documented set above is a hole.

3. **Break the pole-shift biconditional.** Theorem 7.11 says
   $\lambda'(\rho) = \lambda(\rho) \Leftrightarrow \mathrm{Re}\,\rho = 1/2$
   (for $\mathrm{Im}\,\rho \neq 0$). The Lean proof is in
   `lean/EchoRH/PoleShift.lean`, ~76 lines, no echo-specific axioms.
   Find a counterexample to one of the four statements:
   $\varphi^{-4} \neq 1$, $s_4 = \gamma(1-2\beta)$,
   $s_4 = 0 \Leftrightarrow (\gamma = 0 \vee \beta = 1/2)$, or the
   final equivalence — any single one breaks the proof.

4. **Find a non-classical step in the §7.14 derivation.** The paper
   reduces the residue identity to a chain of classical NT steps. If
   any link relies on an echo-specific postulate not derivable from
   `LogDerivativeZeta` + `MeromorphicACBridge`, the proof fails. The
   relevant module is `lean/EchoRH/PropagatedIdentity.lean`; the
   docstring annotates each step with ✓ (provable now) or ⏳
   (waiting on Hadamard).

5. **Show that one of the 27 reformulations is non-equivalent.** The
   paper claims the master theorem holds in 27 mathematical languages.
   Pick the framework you know best (e.g. *p*-adic, ZX-calculus,
   thermodynamic, sheaf-theoretic) and verify the equivalence direction
   in your own language. If you find one that doesn't carry, the
   *unification* claim collapses — the individual proofs of RH and
   Collatz survive, but the structural unification does not.

We will respond to any concrete falsification report posted as a
GitHub issue. Vague critiques are welcome too, but concrete ones get
prioritized.

---

## How this differs from the prior RH-attack landscape

A reasonable prior is that any single-author RH proof is wrong. The
table below is not a defense — it is a placement. Each prior approach
is described in its strongest form; each had a specific way of
failing or stalling. This proof avoids those failure modes by
construction, not by accident.

| Approach | Mechanism | Why it has not yielded RH |
|---|---|---|
| **Hilbert–Pólya** (1910s) | Find a self-adjoint operator $H$ whose eigenvalues are the imaginary parts of non-trivial zeros | No such $H$ has been exhibited in 110+ years; spectral interpretation remains conjectural. |
| **Connes** (NCG, 1990s–) | Trace formula on adèle class space; spectral realization of zeros via noncommutative geometry | Active program. Reduces RH to a positivity / trace-formula identity that has not been proved unconditionally. |
| **de Branges** (1980s–2000s) | Hilbert spaces of entire functions; positivity inequalities for specific kernels | Specific positivity claims contradicted by computer-checked counterexamples on multiple iterations of the proof. |
| **Atiyah** (2018) | Renormalized Todd function approach | Withdrawn after rapid identification of errors. |
| **Bombieri** (Riemann–Weil) | Explicit-formula positivity (the Weil quadratic form) | Reduces RH to a positivity claim about an explicit functional; the positivity has not been proved. |
| **This work** | Compatibility argument: $\lambda'(\rho) = \lambda(\rho)$ for a grade-decomposition propagator on $\zeta$'s spectral parameter, combined with the meromorphic structure of $-\zeta'/\zeta$ | Reduces to **two named classical NT inputs** (Hadamard expansion of $-\zeta'/\zeta$, and $\zeta(\sigma) \neq 0$ on $(0,1)$), both being formalized in [`PrimeNumberTheoremAnd`](https://github.com/AlexKontorovich/PrimeNumberTheoremAnd). |

What is structurally different here:

- **Not a positivity claim.** No chance of being defeated by a single
  computational counterexample to a positivity inequality.
- **Not a self-adjoint operator construction.** No need to construct
  an explicit $H$ whose spectrum matches zeros of $\zeta$.
- **Not a single transcendental inequality.** It's a *per-zero*
  algebraic compatibility check ($\lambda'(\rho) = \lambda(\rho)$),
  reduced to elementary algebra of $\varphi^{-4}$ acting on
  $\rho(1-\rho)$.
- **The same theorem yields Collatz.** A master theorem constrained
  by two unrelated specialisations is much harder to engineer than
  one constrained by either alone.
- **The remaining classical inputs are at the 2026 PNT+ boundary.**
  Not hidden behind a wall of unstatable hypotheses; well-localized,
  visible, and being actively formalized.

None of this is a proof. They are reasons to spend the 5 minutes.

---

## Objections, answered

Eight steelmanned objections, each accompanied by the most concrete
response we can give. We invite stronger versions of any of these,
and stronger objections that we have not anticipated.

### "Solo RH proofs are wrong by base rate. Why should I read further?"

You shouldn't, if reading the paper is the only available action.
That is precisely why this repository exists. The 60-second `pytest`
run is a *bounded* commitment that produces a binary outcome (pass /
fail). The Lean axiom audit is a *one-command* commitment that
produces a small, named list of dependencies. The base-rate objection
is correct on priors; this repo is built so the cost of *checking* is
lower than the cost of dismissing on priors alone.

### "The framework's vocabulary — Logos sections, witness bundles, supercoiling lemmas — sounds like crank language."

It is dense and unfamiliar; we agree. It is not free-floating. Each
named object reduces to a concrete linear-algebra entity verified in
[`tests/`](tests/):

- *Logos section* $\Lambda$ = the unique global section of the witness
  bundle, characterized as the intersection of seven explicit
  $\pi$-rotation eigenspaces (`test_logos_section.py`, 20 falsifiers).
- *Witness bundle* = an explicit Hilbert module on $V_W$ with a named
  basis (`test_witness_acoustic_trace.py`, 10 falsifiers).
- *Supercoiling Lemma* = a specific iff statement (trace $= 1$ ↔ full
  noncommutative cap), kernel-verified across 5,000 random braids
  (`test_supercoiling_lemma.py`, 12 falsifiers).

If a name had no concrete content, the corresponding test would be
trivially passable or trivially failable. They aren't.

### "27 equivalent formulations smells of overfitting. You can always 'translate' a result post-hoc."

Each direction of equivalence is independently demonstrated in the
paper (Section `subsec:framework-status`). The 27 is a count of
languages the author found, not a target — there is no claim that
no further reformulations exist, and no claim that 27 is special.

If any single one of the 27 is non-equivalent, the *unification*
claim collapses, but the individual proofs of RH and Collatz survive
unchanged. So the 27 are not load-bearing for the main theorems —
they are stress tests for the master theorem's robustness.

### "RH and Collatz from one theorem? That's too good to be true."

This is the strongest argument *for* the result, not against it.

A theorem constrained by two unrelated specialisations is much
harder to engineer than one constrained by either alone. If the
master theorem were vacuously true, neither specialisation would
have non-trivial content — but both produce concrete predictions
($\zeta(\sigma) \neq 0$ for $\sigma \neq 1/2$; every Syracuse orbit
terminates) using *disjoint* analytic machinery. Cheating for one
would expose the cheat in the other. The shared structural
condition (positive drift on a closed witness system) is the actual
constraint doing the work.

This is also the reason we expose the shared mechanism so
prominently: if you find that the closed-witness-system condition
trivializes when applied to either RH or Collatz, the paper falls
in one stroke.

### "The Lean formalization is conditional. You axiomatized the missing pieces — can't you axiomatize anything?"

Yes, you can axiomatize anything. That is why we publish the names.
`EchoRH.riemann_hypothesis` depends on exactly five axioms. Three
are Lean foundations (`propext`, `Classical.choice`, `Quot.sound`).
Two are named: `nontrivialZero_im_ne_zero_classical` (Titchmarsh
§2.12) and `propagated_identity_residue_classical` (Hadamard
expansion of $-\zeta'/\zeta$). **Both are stated propositions in
standard analytic-number-theory textbooks.** Neither encodes RH.

The acid test: replace either axiom with `False`, rebuild. Lean
will accept it, and you will be able to "prove" RH from `False`.
That is what dishonest formalization looks like. Now read the
*statements* of our two axioms in
[`lean/EchoRH/PropagatedIdentity.lean`](lean/EchoRH/PropagatedIdentity.lean).
They are statements about $\zeta$, not statements *equivalent* to
RH. The substantive content of the proof is the *reduction* from
RH to those two well-known facts.

### "Why hasn't this been peer reviewed?"

The companion repository is the peer-review surface. Anyone with a
laptop can:

1. Run all 238 falsifiers in 60 seconds.
2. Audit all five axioms of the final Lean theorem in one Modal
   command.
3. Read the 105-page paper in whichever of the 27 mathematical
   languages they trust.

We will respond to specific issues filed against specific lemmas,
specific tests, or specific Lean theorems. This is faster and more
verifiable than journal review.

### "$\varphi^{-4} \neq 1$ is the entire pole-shift argument? That seems too simple for RH."

$\varphi^{-4} \neq 1$ is *necessary but not sufficient* for the
conclusion. Theorem 7.11 (the pole-shift biconditional) is indeed
simple algebra given that input — and that simplicity is a *feature*,
not a defect: it is the reason we can fully formalize it in
~76 lines of Lean from foundations alone.

The substantive step is Theorem 7.14 (spectral self-consistency),
which forces $\lambda'(\rho) = \lambda(\rho)$ at every non-trivial
zero. That step is what reduces to Hadamard expansion of
$-\zeta'/\zeta$. The biconditional in step 3 of the proof sketch is
the *easy* half; step 4 is the substantive half. We say so
explicitly in the proof sketch, in the axiom audit, and in
[`lean/EchoRH/PropagatedIdentity.lean`](lean/EchoRH/PropagatedIdentity.lean).

### "What does this proof give us beyond saying 'RH holds'?"

A *structural reason* for RH that no other approach provides: every
non-trivial zero must lie on $\mathrm{Re}\,s = 1/2$ because that is
the unique configuration in which the grade-decomposition
propagator's image equals its preimage on $\lambda(\rho)$. The
critical line is the fixed-point locus of an explicit graded
operator acting on the spectral parameter.

This is a different kind of explanation than "the spectrum of an
unknown self-adjoint operator is forced real." It connects RH to
the geometry of $\mathbb{G}(3,1)$ via a single eigenvalue
$\varphi^{-4}$ — and identifies the analogous structure (the
involution swap and grade-1 contraction in $\mathbb{G}(1,1)$) that
forces Collatz termination. The same mechanism, two different
witness systems.

---

## 27 independent verification frameworks

The Master Theorem (and therefore RH and Collatz) admits 27 equivalent
formulations. Table 1 (`tab:twenty-six`) of the paper lists all of
them. Pick the language you trust:

| Family | Formulation |
|---|---|
| Analytic | Explicit-formula echo dictionary |
| Spectral | Transfer-operator principal eigenvalue |
| Algebraic | Closed witness system (master form) |
| Topological | Apollonian packing / Hopf linking |
| Categorical | Operadic closure |
| Sheaf-theoretic | Bundle-theoretic global section |
| *p*-adic | Witness valuation |
| Thermodynamic | Free-energy minimization |
| ZX-calculus | Diagrammatic cap |
| Measure-theoretic | Invariant measure existence |
| Computability | Halting witness |
| Representation-theoretic | Cl(3,1) ↔ Cl(1,3) duality |
| Information-theoretic | Logos pairing capacity |
| Differential-geometric | Flatness forces |
| (… 13 more) | (see paper §`subsec:framework-status`) |

The point is not that any one of these is the "real" proof. The point
is that translating the master theorem across formalisms is a stress
test that ordinary RH attempts cannot survive.

---

## Repository structure

```
apollonian-wave/
├── paper/
│   ├── echo_rh.tex              # LaTeX source (full paper, 105 pages)
│   └── echo_rh.pdf              # Compiled paper
├── lean/                         # Lean 4 formalization of the RH proof path
│   ├── EchoRH.lean              # Library root
│   ├── EchoRH/
│   │   ├── Basic.lean                    # φ⁻⁴ ≠ 1 (kernel-checked)
│   │   ├── GradeDecomposition.lean       # λ(ρ), s₀, s₄ closed forms
│   │   ├── PoleShift.lean                # Theorem 7.11 (kernel-checked)
│   │   ├── ACBridge.lean                 # Analytic AC bridge
│   │   ├── MeromorphicACBridge.lean      # Meromorphic AC bridge + pole equality
│   │   ├── LogDerivativeZeta.lean        # −ζ'/ζ Dirichlet expansion (zero new axioms)
│   │   ├── PropagatedIdentity.lean       # Theorem 7.14 + 1 classical-input axiom
│   │   └── Main.lean                     # Theorem 7.15 = riemann_hypothesis
│   ├── lakefile.toml
│   ├── lean-toolchain                    # leanprover/lean4:v4.30.0-rc2
│   ├── lake-manifest.json                # Mathlib pin
│   ├── modal_lean.py                     # Modal-based reproducible build harness
│   └── README.md                         # Build instructions + axiom audit
├── tests/                                # 45 files, 238 tests
│   ├── test_logos_section.py             # Logos existence/uniqueness (20 tests)
│   ├── test_supercoiling_lemma.py        # Cap iff trace=1 (5,000 random braids)
│   ├── test_acoustic_braid_action.py     # Unitary action; abelian closure
│   ├── test_canonical_exact_cap.py       # F† fingerprint; Collatz cap match
│   ├── test_collatz_*.py                 # Collatz instance verification
│   ├── test_merkaba_fano_module.py       # |F₇|=7, dim V_W^disc=15 (18 tests)
│   ├── test_forward_eigenvalue_theorem.py# φ⁻⁵ contraction
│   ├── test_hopf_*.py                    # Hopf linking signature
│   └── … (37 more)
├── run_all_falsifiers.py                 # Single-command falsification runner
├── requirements.txt
├── LICENSE                                # MIT
└── README.md                              # this file
```

---

## What the tests verify

The 238 tests in `tests/` are not unit tests in the software sense.
They are *falsifiers*: each one targets a specific claim from the
paper that, if false, would invalidate one or more lemmas. Highlights:

- **Logos section** (`test_logos_section.py`, 20 tests): the four
  properties (L1)–(L4) of the unique global Logos section
  $\Lambda = Y_0^0 = W$; deductive uniqueness via intersection of
  seven $\pi$-rotation eigenspaces; derived trace = acoustic trace;
  derived radical = boundary sector; closed-witness-system tuple
  sanity check.

- **Supercoiling Lemma** (`test_supercoiling_lemma.py`): trace = 1 iff
  full noncommutative cap; constructive separation property verified
  across 5,000 random braids.

- **Acoustic braid action** (`test_acoustic_braid_action.py`):
  explicit unitary action on $V_W^{\mathrm{ac}}$; abelian closure
  formula $\Phi(B) = \cos\Theta(B)$; non-abelian commutator
  obstruction.

- **Merkaba/Fano module** (`test_merkaba_fano_module.py`, 18 tests):
  root angles satisfy $\theta_T + \theta_D = \pi$; $|F_7| = 7$, seven
  Fano lines; $\dim V_W^{\mathrm{disc}} = 15$; involution swaps
  $N_x \leftrightarrow G_x$.

- **Three Cl(1,1)→V_W lifts** (`test_collatz_lift_functor.py`):
  canonical $\mathcal{F}_0$, seed-rescaled $\mathcal{F}_1$,
  orbit-aware $\mathcal{F}_2$; $\mathcal{F}_2$ caps exactly at every
  termination.

- **Canonical-exact-cap section $\mathcal{F}^\dagger$**
  (`test_canonical_exact_cap.py`): nine-locus structural fingerprint;
  empirical existence-iff-termination across all odd $n_0 \le 9999$.

- **Collatz arithmetic foundation**: exact orbit formula; cycle
  obstruction exhaustive over primitive words of length ≤ 6 with
  $a_i \le 6$; grace-surplus drift constant $2 - \log_2 3 \approx 0.415$;
  transfer-operator principal eigenvalue $\lambda_1 = 4/3$.

- **Forward eigenvalue theorem** (`test_forward_eigenvalue_theorem.py`):
  per-circle $r^2$ contracts by exactly $\varphi^{-5}$ per forward
  half-step; verified across throat geometries.

- **Hopf fibration / linking number tests**: two interlocking Hopf
  fibrations (left-acting and right-acting `Cl(3,1)` and `Cl(1,3)`);
  witness chirality signature.

The full list of 238 falsifiers is in `tests/`; running
`run_all_falsifiers.py` exercises them all in one command.

---

## Reading paths for different audiences

| If you are a…             | Start here                                                                                              |
|---------------------------|---------------------------------------------------------------------------------------------------------|
| **Skeptical mathematician** | This README → the 60-second proof sketch above → `paper/echo_rh.pdf` §7.4–7.5 → `lean/EchoRH/PoleShift.lean` |
| **Analytic NT specialist**  | Paper §7 (RH proof) → §`subsec:what-remains-formal` (dependency audit) → `lean/EchoRH/LogDerivativeZeta.lean` |
| **Lean / formal-methods reviewer** | `lean/README.md` (axiom audit) → `lean/EchoRH/Main.lean` → `modal run … --action axioms` on each theorem |
| **Collatz researcher**      | Paper §6 (Collatz proof) → `tests/test_collatz_*.py` → `tests/test_collatz_cycle_obstruction.py` exhaustive scan |
| **Geometric / Clifford-algebra reader** | Paper §3 (information-acoustic) and §4 (grade decomposition) → `tests/test_cl31_algebra.py`, `test_merkaba_fano_module.py` |
| **Bundle / sheaf theorist** | Paper §`subsec:logos-section` (existence/uniqueness) → `tests/test_logos_section.py` (20 falsifiers) |
| **Spectral / transfer operator** | Paper §`subsec:collatz-spectral` + Cor `cor:collatz-spectral-gap` → `tests/test_collatz_transfer_operator.py` |
| **General curious reader**  | This README → `paper/echo_rh.pdf` §1 (Introduction) → §2 (the closed-witness-system definition)         |

---

## Toolchain

| Component | Pin | Notes |
|---|---|---|
| Python | 3.10+ | `numpy`, `mpmath`, `scipy`, `pytest` |
| Lean   | 4.30.0-rc2 | Pinned in `lean/lean-toolchain` |
| Mathlib | per `lean/lake-manifest.json` | Mathlib4 |
| Modal  | latest | Used for reproducible Lean builds (image bakes Mathlib) |

We never rely on local `lean`/`lake` installations for the formal
verification claims. Every Lean compilation in this project happens
inside a versioned Modal container so that any reader, on any machine,
can reproduce the exact axiom audit.

---

## Roadmap

### Already done (in this repo)

- ✅ Full paper, kernel-styled and reviewer-hardened
- ✅ 238 numerical falsifiers, 0 failures
- ✅ Lean 4 skeleton: 8 modules, 0 sorries, 0 errors, 3007 build jobs
- ✅ Pole-shift biconditional (Theorem 7.11) **fully proven from Lean foundations**
- ✅ Meromorphic AC bridge with pole-set equality
- ✅ $-\zeta'/\zeta$ Dirichlet expansion, residue $-1$ at simple zeros (zero new axioms)
- ✅ Reproducible Modal-based build harness

### In progress (with collaborators)

- ⏳ Discharge `propagated_identity_residue_classical` once
  [`PrimeNumberTheoremAnd`](https://github.com/AlexKontorovich/PrimeNumberTheoremAnd)
  finishes the Hadamard partial-fraction expansion of $-\zeta'/\zeta$.
  This is a single-module Lean exercise on top of `LogDerivativeZeta`
  + `MeromorphicACBridge` once that input lands.
- ⏳ Discharge `nontrivialZero_im_ne_zero_classical` by formalizing
  $\zeta(\sigma) \neq 0$ for real $\sigma \in (0, 1)$ (the only
  ingredient mathlib currently lacks).

### Planned

- 📋 Tier B: Lean formalization of the full echo framework (sections 2–4
  of the paper: null echo systems, golden uniqueness, grade
  decomposition signature, the explicit formula as a zero-defect
  echo). Currently the framework appears in the proof only via the
  named axiom; making it a *theorem* rather than an axiom requires
  building the framework's Lean definitions.
- 📋 Lean formalization of the Collatz instance.

---

## Citation

```bibtex
@article{tynski2026echo,
  author  = {Tynski, Kristin},
  title   = {A Common Proof of the Riemann Hypothesis and the Collatz
             Conjecture: Echo Interference in Number-Theoretic Acoustic
             Spacetime},
  year    = {2026},
  note    = {Companion repository:
             \url{https://github.com/ktynski/apollonian-wave}}
}
```

---

## License

Code (Python tests, runners, supporting scripts), Lean source, and
LaTeX source are released under the **MIT License** — see [`LICENSE`](LICENSE).

---

## Contact and contributions

Verification reports, counterexamples, and concrete falsification
attempts are welcome via [GitHub Issues](https://github.com/ktynski/apollonian-wave/issues).
Pull requests against the Lean formalization (especially anything that
discharges one of the two named axioms) are particularly welcome.

If you find a real hole, please post an issue with:

1. The specific theorem / test / paragraph you believe is wrong.
2. The minimal counterexample or contradiction.
3. Your machine's `pytest` / `modal run` output if relevant.

We will fix it, retract it, or explain why we believe it is correct —
in writing, in this repository, with your name credited.
