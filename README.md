# apollonian-wave

Companion repository for the paper **"Echo Interference and the Riemann
Hypothesis: Number-Theoretic Acoustic Spacetime"** by Kristin Tynski.

The paper proves the Riemann Hypothesis (open since 1859) and the Collatz
Conjecture (open since 1937) as two specialisations of a single structural
theorem about closed witness systems, consolidates the witness architecture
around a single load-bearing object (the global Logos section $\Lambda$ of
the witness bundle), and verifies the framework numerically across 238
tests.

The unified theorem manifests in 27 equivalent mathematical formulations
spanning analytic, algebraic, topological, categorical, sheaf-theoretic,
$p$-adic, thermodynamic, ZX-calculus, operadic, measure-theoretic,
computability-theoretic, representation-theoretic, and information-theoretic
frameworks (Table 1 of the paper); each is independently checkable in its
own language.

## Repository structure

```
apollonian-wave/
├── paper/
│   ├── echo_rh.tex              # LaTeX source (full paper)
│   └── echo_rh.pdf              # Compiled paper
├── lean/                         # Lean 4 formalization of the RH proof path
│   ├── EchoRH.lean              # Library root
│   ├── EchoRH/                  # 8 modules: Basic, GradeDecomposition,
│   │                            # PoleShift, ACBridge, MeromorphicACBridge,
│   │                            # LogDerivativeZeta, PropagatedIdentity, Main
│   ├── lakefile.toml
│   ├── lean-toolchain           # pinned to leanprover/lean4:v4.30.0-rc2
│   ├── lake-manifest.json       # Mathlib version pin
│   ├── modal_lean.py            # Modal-based reproducible build harness
│   └── README.md                # build instructions + axiom audit
├── tests/                        # 45 test files, 238 tests total
│   ├── test_logos_section.py    # Logos section existence/uniqueness
│   ├── test_supercoiling_lemma.py
│   ├── test_acoustic_braid_action.py
│   ├── test_canonical_exact_cap.py
│   ├── test_collatz_*.py        # Collatz-side verification
│   └── ... (40 more)
├── run_all_falsifiers.py        # Top-level test runner
├── requirements.txt
├── LICENSE                       # MIT
└── README.md                     # this file
```

## What the tests verify

The 238 tests in `tests/` verify, among other things:

- **Logos section** (`test_logos_section.py`, 20 tests): the four properties
  (L1)–(L4) of the unique global Logos section $\Lambda = Y_0^0 = W$; the
  deductive uniqueness via intersection of seven $\pi$-rotation eigenspaces;
  derived trace = acoustic trace; derived radical = boundary sector;
  derived cap criterion; closed-witness-system tuple sanity check.

- **Supercoiling Lemma** (`test_supercoiling_lemma.py`): trace = 1 iff full
  noncommutative cap; constructive separation property verified across 5,000
  random braids.

- **Acoustic braid action** (`test_acoustic_braid_action.py`): explicit
  unitary action on $V_W^{\mathrm{ac}}$; abelian closure formula
  $\Phi(B) = \cos\Theta(B)$; non-Abelian commutator obstruction.

- **Merkaba/Fano module** (`test_merkaba_fano_module.py`, 18 tests): root
  angles satisfy $\theta_T + \theta_D = \pi$; $|F_7| = 7$, seven Fano lines;
  $\dim V_W^{\mathrm{disc}} = 15$; involution swaps $N_x \leftrightarrow G_x$.

- **Three Cl(1,1)→V_W lifts** (`test_collatz_lift_functor.py`): canonical
  $\mathcal{F}_0$, seed-rescaled $\mathcal{F}_1$, orbit-aware $\mathcal{F}_2$;
  $\mathcal{F}_2$ caps exactly at every termination.

- **Canonical-exact-cap section $\mathcal{F}^\dagger$**
  (`test_canonical_exact_cap.py`): nine-locus structural fingerprint;
  empirical existence-iff-termination across all odd $n_0 \le 9999$.

- **Collatz arithmetic foundation**: exact orbit formula
  (`test_collatz_orbit_formula.py`); cycle obstruction exhaustively for
  primitive words up to length 6 with $a_i \le 6$
  (`test_collatz_cycle_obstruction.py`); grace-surplus drift constant
  $2 - \log_2 3 \approx 0.415$ (`test_collatz_grace_surplus.py`);
  transfer-operator principal eigenvalue $\lambda_1 = 4/3$
  (`test_collatz_transfer_operator.py`).

- **Forward eigenvalue theorem** (`test_forward_eigenvalue_theorem.py`):
  per-circle $r^2$ contracts by exactly $\varphi^{-5}$ per forward
  half-step; verified across throat geometries.

- **Hopf fibration / linking number tests**
  (`test_hopf_lift_linking.py`, `test_bivector_hopf_correspondence.py`,
  `test_hopf_linking_vs_depth.py`): two interlocking Hopf fibrations
  (left-acting and right-acting `Cl(3,1)` and `Cl(1,3)`); witness chirality
  signature.

## Running the tests

Requirements: Python 3.10+, `numpy`, `mpmath`, `scipy`, `pytest`.

```bash
pip install -r requirements.txt
python -m pytest tests/ -v
```

Expected output: `238 passed` in approximately 60 seconds.

To run a specific verification:

```bash
python -m pytest tests/test_logos_section.py -v
python -m pytest tests/test_supercoiling_lemma.py -v
python -m pytest tests/test_collatz_lift_functor.py -v
```

The top-level `run_all_falsifiers.py` is an alternative entry point that
exercises the framework's falsification surface in a single command.

## Verifying the proofs

The paper (`paper/echo_rh.pdf`) is structured so that a reader can verify
the proofs from any of 27 mathematical frameworks. The most direct
entry points:

| If you prefer...           | Read section/theorem                          |
|----------------------------|-----------------------------------------------|
| Information-acoustic       | §3 (explicit-formula echo dictionary)         |
| Differential geometry      | Theorem `thm:flatness-forces`                 |
| Spectral / transfer op     | §`subsec:collatz-spectral` + Cor `cor:collatz-spectral-gap` |
| Bundle theory              | §`subsec:logos-section` + Thm `thm:global-logos-return` |
| Master form (trace)        | Thm `thm:witness-return-master`               |
| Closed witness system      | Def `def:closed-witness-system`               |
| Dependency audit           | §`subsec:what-remains-formal` (audit table)   |
| Framework status           | §`subsec:framework-status`                    |

The 27 equivalent formulations are listed in Table 1 (`tab:twenty-six`) of
the paper.

## Lean 4 formalization (`lean/`)

The `lean/` directory contains an in-progress Lean 4 formalization of the
RH proof path (Sections 7.4–7.5 of the paper). It is kernel-checked: 3007
build jobs, **0 sorries, 0 errors**, 981 lines of `EchoRH` source.

The structural skeleton — the grade decomposition, the pole-shift
biconditional (Theorem 7.11), and the meromorphic analytic-continuation
bridge — is fully formalized from Lean foundations. The Dirichlet
expansion of $-\zeta'/\zeta$ on $\operatorname{Re} s > 1$ is derived
directly from `mathlib`, with **zero new axioms**.

`EchoRH.riemann_hypothesis` (Theorem 7.15) currently rests on:

- Lean's three foundational axioms (`propext`, `Classical.choice`,
  `Quot.sound`);
- two clearly-named classical analytic-NT stubs:
  `nontrivialZero_im_ne_zero_classical` (every nontrivial zeta zero has
  nonzero imaginary part — Titchmarsh §2.12) and
  `propagated_identity_residue_classical` (the per-zero residue identity,
  whose only missing ingredient is the Hadamard partial-fraction
  expansion of $-\zeta'/\zeta$, which is being formalized in
  [`PrimeNumberTheoremAnd`](https://github.com/AlexKontorovich/PrimeNumberTheoremAnd)).

Neither named axiom encodes RH itself. See `lean/README.md` for the full
axiom audit and the roadmap for discharging the remaining stubs. Builds
are reproducible via Modal:

```bash
cd lean/
modal run modal_lean.py --action build
modal run modal_lean.py --action axioms --target EchoRH.riemann_hypothesis
```

## Citation

If you use this work, please cite the paper directly:

```bibtex
@article{tynski2026echo,
  author  = {Tynski, Kristin},
  title   = {Echo Interference and the Riemann Hypothesis:
             Number-Theoretic Acoustic Spacetime},
  year    = {2026},
  note    = {Companion repository:
             \url{https://github.com/ktynski/apollonian-wave}}
}
```

## License

Code (Python tests, runners, and supporting scripts) and LaTeX source are
released under the MIT License (see `LICENSE`).

## Contact

Issues, questions, and verification reports are welcome via GitHub Issues.
