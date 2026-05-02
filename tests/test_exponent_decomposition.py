"""Falsifier 17: Decomposition of the contraction exponent 5.

Hypothesis under test:
  The forward half-step contraction exponent in lambda(T_+) = phi^{-5}
  decomposes as 5 = p + q where p is the cardinality of the closure
  budget rho^{-1} + ... + rho^{-p} = 1 and q is the branching factor
  (boundaries per interstice).  For the golden tetrad: p = 2, q = 3,
  p + q = 5.

  This is a candidate STRUCTURAL identity competing with the
  Clifford-algebraic identity 5 = (dim Cl(3,1) - dim Cl^+(3,1))/2 + 1
  = (16 - 8)/2 + 1 already in the paper (conj:phi5-relaxation).

The two readings agree on the 2+3 = 5 case and on the (16-8)/2 + 1 = 5
case, but they make DIFFERENT predictions for hypothetical generalised
systems.  This test computes both predictions across a family of
hypothetical analogs and reports the predictions side by side, so a
future numerical investigation (e.g., a Cl(4,1) Apollonian-like packing)
can falsify one or the other.

Hard test (algebraic, exact):

  (A) Both readings agree on the actual 2+3 case.
      Verify: 2 + 3 = (16 - 8)/2 + 1 = 5 = contraction exponent.

  (B) The closure budget is uniquely satisfied at the golden ratio.
      Verify: rho^{-1} + rho^{-2} = 1 has unique positive root rho = phi.

  (C) Generalised closure budgets give different roots.
      The 3-return budget rho^{-1} + rho^{-2} + rho^{-3} = 1 has root
      rho_3 = tribonacci constant approx 1.839 != phi.

  (D) The (p, q) reading proposes a structural identity:
        contraction exponent = p + q
      for any closed-recursion system with p-return budget and
      q-boundary interstices.  At the bookkeeping level, this holds
      for the actual (2, 3) case.

      The DIMENSIONAL generalisation of this reading is already
      FALSIFIED by falsifier 14b
      (test_dim_dependent_contraction.py): 3D Soddy Apollonian
      sphere packing has been measured to contract per-sphere r^2
      by phi^{-2.2}, not phi^{-6} as the dimensional reading
      N(d) = (d+1) + 2 would predict.  The literal trefoil-topology
      reading is also falsified (falsifier 14a,
      test_trefoil_lift_topology.py): the natural z=depth lift of
      the 2-generation sub-packing has all linking numbers zero,
      i.e. is a 4-component unlink, not a trefoil.

      What survives is the bookkeeping identity 5 = 2 + 3 = p + q
      for the (2, 3) case alone.  This test records the
      bookkeeping-level predictions for hypothetical analogs while
      noting that the strong dimensional and topological readings
      have been falsified by the existing tests cited above.

Pass conditions (the algebraic facts):
  - SUM_AGREES: 2 + 3 == 5 == (16 - 8)/2 + 1.
  - PHI_ROOT: rho^{-1} + rho^{-2} = 1 at rho = phi (within 1e-12).
  - TRIBONACCI: rho^{-1} + rho^{-2} + rho^{-3} = 1 has positive root
    different from phi.
  - PQ_NUMEROLOGY: the candidate p+q values across hypothetical
    (p, q) analogs are distinct integers (the reading produces
    different predictions for different parameters, so it is at
    least non-trivial).

This test is a PURE ALGEBRAIC consistency check.  No numerical Apollonian
data is involved; the verification is symbolic.  The test exists to
clearly record the two competing readings and their predictions so the
paper can present them as alternatives under future falsification.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

PHI = (1.0 + math.sqrt(5.0)) / 2.0


# ───────────────────────────────────────────────────────────────────────
# (A) Both readings agree on the (2,3) case
# ───────────────────────────────────────────────────────────────────────


def test_sum_agrees() -> None:
    """SUM_AGREES assertion: 2 + 3 == 5 == (16 - 8)/2 + 1."""
    p_plus_q = 2 + 3
    clifford = (16 - 8) // 2 + 1
    contraction_exponent = 5
    assert p_plus_q == 5
    assert clifford == 5
    assert p_plus_q == contraction_exponent
    assert clifford == contraction_exponent


# ───────────────────────────────────────────────────────────────────────
# (B) Two-return budget closes uniquely at phi
# ───────────────────────────────────────────────────────────────────────


def test_phi_root() -> None:
    """PHI_ROOT assertion: rho^{-1} + rho^{-2} = 1 at rho = phi."""
    lhs = PHI ** (-1) + PHI ** (-2)
    assert abs(lhs - 1.0) < 1e-12, (
        f"PHI_ROOT FAIL: {PHI}^-1 + {PHI}^-2 = {lhs}, expected 1.0"
    )


# ───────────────────────────────────────────────────────────────────────
# (C) Tribonacci closure: 3-return budget gives a different root
# ───────────────────────────────────────────────────────────────────────


def test_tribonacci_distinct_from_phi() -> None:
    """TRIBONACCI assertion: 3-return budget root != phi."""
    rho_3 = closure_root(3)
    assert abs(rho_3 - PHI) > 0.1, (
        f"TRIBONACCI FAIL: 3-return root {rho_3} too close to phi {PHI}"
    )
    lhs = sum(rho_3 ** (-k) for k in (1, 2, 3))
    assert abs(lhs - 1.0) < 1e-9, (
        f"TRIBONACCI FAIL: budget at rho_3 = {lhs}, expected 1.0"
    )


# ───────────────────────────────────────────────────────────────────────
# (D) Divergent predictions across hypothetical analogs
# ───────────────────────────────────────────────────────────────────────


def pq_prediction(p: int, q: int) -> int:
    """Torus-knot reading: contraction exponent = p + q.

    For (p, q) = (2, 3): exponent = 5 (matches conj:phi5-relaxation).
    For other (p, q): a falsifiable prediction for hypothetical
    higher-dimensional Apollonian-analog systems.
    """
    return p + q


def closure_root(p: int, tol: float = 1e-12, max_iter: int = 200) -> float:
    """Solve sum_{k=1..p} rho^{-k} = 1 by Newton iteration.

    Equivalently: rho^p = rho^(p-1) + rho^(p-2) + ... + 1.
    For p=1: rho = 1.  For p=2: rho = phi.  For p=3: tribonacci.
    """
    if p == 1:
        return 1.0

    def f(r: float) -> float:
        # rho^p - sum_{k=0..p-1} rho^k
        s = 0.0
        rk = 1.0
        for _ in range(p):
            s += rk
            rk *= r
        return rk - s

    def fp(r: float) -> float:
        # derivative of rho^p - (sum_{k=0..p-1} rho^k)
        return p * r ** (p - 1) - sum(k * r ** (k - 1) for k in range(1, p))

    r = 1.5
    for _ in range(max_iter):
        delta = f(r) / fp(r)
        r -= delta
        if abs(delta) < tol:
            break
    return r


def test_pq_numerology() -> None:
    """PQ_NUMEROLOGY assertion: the (p, q) reading produces a meaningful
    spread of integer predictions across hypothetical analogs.

    The reading is non-trivial: it predicts different exponents for
    different (p, q), and these predictions are at least distinct
    enough to be falsifiable.
    """
    cases = [(2, 3), (3, 3), (2, 4), (3, 4), (2, 5), (4, 5)]
    predictions = [pq_prediction(p, q) for p, q in cases]
    assert len(set(predictions)) >= 4, (
        f"PQ_NUMEROLOGY FAIL: only {len(set(predictions))} distinct "
        f"predictions, expected >= 4.  Predictions: {predictions}"
    )
    assert pq_prediction(2, 3) == 5


# ───────────────────────────────────────────────────────────────────────
# Diagnostic
# ───────────────────────────────────────────────────────────────────────


def _print_diagnostic_table() -> None:
    print("\n  Two competing readings of the contraction exponent 5:")
    print("  " + "-" * 72)
    print(f"  reading 1 (torus-knot):   p + q = 2 + 3 = 5")
    print(f"  reading 2 (Clifford):     (dim Cl(3,1) - dim Cl^+(3,1))/2 + 1")
    print(f"                          = (16 - 8)/2 + 1 = 5")
    print("  " + "-" * 72)
    print()
    print("  Closure budgets:")
    print("  " + "-" * 72)
    rho_3 = closure_root(3)
    print(f"  2-return root: rho_2 = phi = {PHI:.12f}")
    print(f"  3-return root: rho_3 = tribonacci = {rho_3:.12f}")
    print(f"  delta(rho)  : {abs(rho_3 - PHI):.6f}  (significant)")
    print("  " + "-" * 72)
    print()
    print("  Predictions of the (p, q) torus-knot reading for hypothetical")
    print("  higher-dimensional Apollonian-analog systems (bookkeeping only):")
    print("  " + "-" * 72)
    print(f"  {'(p, q)':>8} {'p + q':>8} {'closure rho':>14}")
    print("  " + "-" * 72)
    cases = [(2, 3), (3, 3), (2, 4), (3, 4), (2, 5), (4, 5)]
    for p, q in cases:
        pq = pq_prediction(p, q)
        rho = closure_root(p)
        marker = "  <- ACTUAL" if (p, q) == (2, 3) else ""
        print(f"  {f'({p},{q})':>8} {pq:>8} {rho:>14.6f}{marker}")
    print("  " + "-" * 72)
    print("  Status of the strong (p, q) reading:")
    print("    Dimensional generalisation N(d) = d + 3 is FALSIFIED by")
    print("    falsifier 14b (test_dim_dependent_contraction.py): 3D")
    print("    Apollonian gives phi^{-2.2}, not the predicted phi^{-6}.")
    print("    Literal trefoil-topology lift is FALSIFIED by falsifier")
    print("    14a (test_trefoil_lift_topology.py): natural lift is a")
    print("    4-component unlink, not a trefoil knot.")
    print("    Only the bookkeeping identity 5 = 2 + 3 for the (2,3)")
    print("    case survives.")


def main() -> None:
    print("=" * 72)
    print("FALSIFIER 17: Exponent decomposition 5 = 2 + 3 vs Clifford reading")
    print("  Algebraic comparison of two readings of the contraction exponent.")
    print("=" * 72)

    _print_diagnostic_table()

    print("\n" + "=" * 72)
    print("HARD ASSERTIONS")
    print("=" * 72)

    failed: list[tuple[str, str]] = []
    for name, fn in (
        ("SUM_AGREES         (2+3 = 5 = (16-8)/2+1)", test_sum_agrees),
        ("PHI_ROOT           (2-return root = phi)", test_phi_root),
        ("TRIBONACCI         (3-return root != phi)", test_tribonacci_distinct_from_phi),
        ("PQ_NUMEROLOGY      (>= 4 distinct (p+q) preds)", test_pq_numerology),
    ):
        try:
            fn()
            print(f"  PASS: {name}")
        except AssertionError as exc:
            print(f"  FAIL: {name}")
            print(f"        {exc}")
            failed.append((name, str(exc)))

    print("\n" + "=" * 72)
    if not failed:
        print("ALGEBRAIC FACTS VERIFIED:")
        print("  Both readings agree on the (2, 3) = 5 case at the")
        print("  bookkeeping level.  The strong (p, q) reading -- which would")
        print("  predict trefoil topology in z=depth lifts and dimensional")
        print("  scaling phi^{-(d+3)} in higher dimensions -- has already")
        print("  been falsified (see falsifiers 14a and 14b).  What")
        print("  survives is the arithmetic identity 5 = 2 + 3 for the")
        print("  actual (2, 3) case, which is bookkeeping rather than a")
        print("  derivation of the integer 5.  The Clifford reading is in")
        print("  conj:phi5-relaxation; the per-circle contraction phi^{-5}")
        print("  is rigorously linked to prop:throat via")
        print("  prop:forward-eigenvalue-throat.")
    else:
        print("ALGEBRAIC FACTS FAILED:")
        print("  An algebraic identity is wrong; the framework needs revision.")
    print("=" * 72)

    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
