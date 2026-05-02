"""Run every apollonian-wave falsifier and print a single kill-matrix block.

Falsifiers in this directory:

  Pre-existing:
    11. merkaba_stitch_count          — Pin(3,1) representation theory of
                                         the Merkaba stitch dimension count
    12. five_witness_ops_dynamics     — Per-grade r^2-weighted contraction
                                         across forward half-steps
                                         (NOTE: this falsifier is designed
                                         to FALSIFY the multiplicative-
                                         factorization reading.  An exit
                                         code of 1 here is the expected
                                         outcome on the existing data, not
                                         a regression -- see INSIGHT
                                         falsifier 10 commentary.)
    13. five_witness_ops_algebra      — Algebraic op-grade mapping in
                                         Cl(3,1)
    14a. trefoil_lift_topology        — Literal trefoil-topology lift of
                                         the 2-generation sub-packing
                                         (FALSIFIED: natural lift is a
                                         4-component unlink)
    14b. dim_dependent_contraction    — Dimensional generalisation
                                         N(d) = d + 3 (FALSIFIED: 3D
                                         Apollonian gives phi^{-2.2},
                                         not phi^{-6})

  New (chiral contact / trefoil bookkeeping / percolation / exponent):
    15  (14). forward_eigenvalue_theorem  — Forward half-step eigenvalue
                                             from throat recursion, stable
                                             depths (algebraic promotion
                                             from conjecture to proposition)
    16  (15). trefoil_boundary            — Bookkeeping combinatorial
                                             trefoil T(2,3) signature
                                             (branching factor 3, two-return
                                             budget, signed crossings 3)
    17  (16). percolation_chirality       — Percolation threshold vs
                                             chirality ratio (critical line
                                             = pure chirality)
    18  (17). exponent_decomposition      — 5 = 2 + 3 vs Clifford reading
                                             (bookkeeping identities;
                                             strong readings already
                                             falsified by 14a/14b)

Usage:

    python3 run_all_falsifiers.py
"""

from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))


def _run(name: str, mod_path: str) -> tuple[bool, str]:
    import importlib.util

    spec = importlib.util.spec_from_file_location(name, str(HERE / mod_path))
    if spec is None or spec.loader is None:
        return False, f"could not load {mod_path}"
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
        if hasattr(module, "main"):
            module.main()
        return True, "ok"
    except SystemExit as exc:
        if exc.code in (0, None):
            return True, "ok"
        return False, f"exit code {exc.code}"
    except AssertionError as exc:
        return False, str(exc)
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}"


def main() -> int:
    print("=" * 72)
    print("APOLLONIAN-WAVE FALSIFIERS — algebra + dynamics + topology")
    print("=" * 72)

    results: list[tuple[str, bool, str]] = []
    for name, rel in [
        ("merkaba_stitch_count",        "tests/test_merkaba_stitch_count.py"),
        ("five_witness_ops_dynamics",   "tests/test_five_witness_ops_dynamics.py"),
        ("five_witness_ops_algebra",    "tests/test_five_witness_ops_algebra.py"),
        ("trefoil_lift_topology",       "tests/test_trefoil_lift_topology.py"),
        ("dim_dependent_contraction",   "tests/test_dim_dependent_contraction.py"),
        ("forward_eigenvalue_theorem",  "tests/test_forward_eigenvalue_theorem.py"),
        ("trefoil_boundary",            "tests/test_trefoil_boundary.py"),
        ("percolation_chirality",       "tests/test_percolation_chirality.py"),
        ("exponent_decomposition",      "tests/test_exponent_decomposition.py"),
    ]:
        print(f"\n--- {name} ---")
        ok, msg = _run(name, rel)
        results.append((name, ok, msg))

    print("\n" + "=" * 72)
    print("APOLLONIAN-WAVE FALSIFIER SUMMARY")
    print("=" * 72)
    for name, ok, msg in results:
        status = "PASS" if ok else "FAIL"
        # Truncate very long failure messages for the summary block.
        short = msg if len(msg) <= 60 else msg[:57] + "..."
        print(f"  {name:<30} {status}  ({short})")

    all_ok = all(ok for _, ok, _ in results)
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
