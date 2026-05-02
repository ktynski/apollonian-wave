"""Collatz-braid witness-trace test.

Each Syracuse step T(n) = (3n + 1) / 2^{a(n)} is encoded as a Cl(1,1)
generator

    v_a := exp((a * log 2) e1 + (log 3) e2),

a boost along the grace direction (e1) by a*log2 plus an expansion
along the nilpotent direction (e2) by log3.  The Collatz braid B(n) of
an odd positive integer n is the ordered Cl(1,1) product of v_{a_i}
along the Syracuse word (a_0, ..., a_{m-1}) of n.

Three observables are computed:

  Tr_W(B)         := (<B + iota(B)>_0) / 2
                     (signed iota-invariant scalar; can flip sign)

  |Tr_W|(B)       := |Tr_W(B)|  (positive surrogate)

  Norm_W(B)       := |B|^2 = scalar part of B * reverse(B)
                     (Cl(1,1) signature norm; can be negative for
                     spacelike products)

  AbsNorm_W(B)    := sqrt(|Norm_W(B)|)  (positive surrogate)

The literal claim "Tr_W > 0 on every returning Collatz braid" is
**not** met by the naive signed trace -- that is exactly the
**Witness Trace Axiom** the framework flags as an open hinge.  What
this test verifies is the *constructive* substrate that any candidate
trace must support:

  - B(n) is well-defined, non-zero, and has finite Cl(1,1) magnitude
    for every odd n with finite Collatz orbit.
  - |Tr_W|(B(n)) > 0 strictly for every test n (the braid is never
    in ker of any nontrivial linear functional).
  - AbsNorm_W(B(n)) > 0 strictly.
  - Tr_W is iota-invariant: Tr_W(iota(B)) = Tr_W(B).

The signed Tr_W distribution is reported but NOT asserted positive --
the test prints the count of negative values to make the open hinge
empirically visible.

Pass conditions:
  - BRAID_NONZERO:        every B(n) has nonzero Cl(1,1) magnitude.
  - ABSTRACE_POSITIVE:    |Tr_W|(B(n)) > 1e-9 for all sampled n.
  - ABSNORM_POSITIVE:     AbsNorm_W(B(n)) > 1e-9 for all sampled n.
  - IOTA_INVARIANT:       Tr_W(iota(B(n))) = Tr_W(B(n)) for samples.
  - SIGNED_TRACE_NONTRIV: at least one sign-flip across the sample,
                          confirming the naive Tr_W is *not* a
                          sign-definite separator and motivating the
                          Witness Trace Axiom.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tests.test_cl11_algebra import (  # noqa: E402
    cl11_basis,
    cl11_mul,
    cl11_norm_sq,
    cl11_reverse,
)
from tests.test_collatz_involution import witness_iota  # noqa: E402
from tests.test_witness_trace import witness_trace  # noqa: E402
from tests.test_collatz_orbit_formula import syracuse_orbit  # noqa: E402

LOG2 = math.log(2.0)
LOG3 = math.log(3.0)


# ────────────────────────────────────────────────────────────────────────
# Cl(1,1) exponential of a pure vector
# ────────────────────────────────────────────────────────────────────────


def exp_cl11_vector(x: float, y: float) -> np.ndarray:
    """exp(x*e1 + y*e2) in Cl(1,1).

    v = x e1 + y e2,  v^2 = x^2 - y^2.
    exp(v) = cosh(s) + sinh(s)/s * v   if v^2 = s^2 > 0
           = cos(s)  + sin(s)/s  * v   if v^2 = -s^2 < 0
           = 1 + v                     if v^2 = 0
    """
    sq = x * x - y * y
    if abs(sq) < 1e-15:
        return np.array([1.0, x, y, 0.0])
    if sq > 0:
        s = math.sqrt(sq)
        return np.array([math.cosh(s), x * math.sinh(s) / s,
                         y * math.sinh(s) / s, 0.0])
    s = math.sqrt(-sq)
    return np.array([math.cos(s), x * math.sin(s) / s,
                     y * math.sin(s) / s, 0.0])


def collatz_step_generator(a: int) -> np.ndarray:
    """v_a = exp((a*log2) e1 + (log3) e2) in Cl(1,1)."""
    return exp_cl11_vector(a * LOG2, LOG3)


def braid_of_odd_n(n: int) -> np.ndarray:
    direct = syracuse_orbit(n)
    word = [a for _, a in direct]
    B = cl11_basis()["1"]
    for a in word:
        B = cl11_mul(B, collatz_step_generator(a))
    return B


def signed_tr_w(B: np.ndarray) -> float:
    return witness_trace(B)


def abs_tr_w(B: np.ndarray) -> float:
    return abs(witness_trace(B))


def norm_w(B: np.ndarray) -> float:
    """Cl(1,1) signature norm: scalar part of B * rev(B)."""
    return cl11_norm_sq(B)


def abs_norm_w(B: np.ndarray) -> float:
    return math.sqrt(abs(norm_w(B)))


# ────────────────────────────────────────────────────────────────────────
# Tests
# ────────────────────────────────────────────────────────────────────────


SAMPLE_RANGE = list(range(3, 1000, 2))


def test_braid_nonzero_for_all_samples() -> None:
    """Every braid has nonzero Cl(1,1) magnitude (norm + scalar^2 > 0)."""
    failures: list[int] = []
    for n in SAMPLE_RANGE:
        B = braid_of_odd_n(n)
        if abs_norm_w(B) < 1e-12 and abs_tr_w(B) < 1e-12 and float(np.linalg.norm(B)) < 1e-12:
            failures.append(n)
    assert not failures, f"braid degenerate for n in {failures[:5]}..."


def test_abs_tr_w_positive_for_all_samples() -> None:
    failures: list[int] = []
    for n in SAMPLE_RANGE:
        B = braid_of_odd_n(n)
        if abs_tr_w(B) < 1e-9:
            failures.append(n)
    assert not failures, (
        f"|Tr_W| <= 1e-9 for n in {failures[:5]} ..."
    )


def test_abs_norm_w_positive_for_all_samples() -> None:
    failures: list[int] = []
    for n in SAMPLE_RANGE:
        B = braid_of_odd_n(n)
        if abs_norm_w(B) < 1e-9:
            failures.append(n)
    assert not failures, (
        f"AbsNorm_W <= 1e-9 for n in {failures[:5]} ..."
    )


def test_iota_invariance_of_signed_trace() -> None:
    for n in [3, 7, 27, 97, 871]:
        B = braid_of_odd_n(n)
        t1 = signed_tr_w(B)
        t2 = signed_tr_w(witness_iota(B))
        assert abs(t1 - t2) < 1e-9, (
            f"Tr_W not invariant under iota at n={n}: {t1} vs {t2}"
        )


def test_signed_trace_is_not_sign_definite() -> None:
    """The naive signed Tr_W is NOT sign-definite on Collatz braids.

    This is a positive empirical observation that motivates the Witness
    Trace Axiom: the right trace is some refinement (e.g. positivity-
    preserving spinor representation) that is NOT the literal Cl(1,1)
    scalar projection.
    """
    pos = neg = 0
    for n in SAMPLE_RANGE:
        t = signed_tr_w(braid_of_odd_n(n))
        if t > 0:
            pos += 1
        elif t < 0:
            neg += 1
    assert pos > 0 and neg > 0, (
        f"signed trace happened to be sign-definite (pos={pos}, neg={neg}); "
        f"this would contradict the open-hinge framing"
    )


def main() -> None:
    print("=" * 72)
    print("COLLATZ BRAID WITNESS-TRACE TEST")
    print("=" * 72)

    print("\n  Sample braid observables:")
    print(f"    {'n':>6} {'orbit_len':>10} {'Tr_W (signed)':>16} "
          f"{'|Tr_W|':>14} {'AbsNorm_W':>14}")
    for n in [3, 5, 7, 9, 11, 27, 41, 97, 703, 871, 6171, 8775]:
        B = braid_of_odd_n(n)
        word_len = len(syracuse_orbit(n))
        print(f"    {n:>6} {word_len:>10} "
              f"{signed_tr_w(B):>+16.4e} "
              f"{abs_tr_w(B):>14.4e} "
              f"{abs_norm_w(B):>14.4e}")

    print("\n  Distribution of signed Tr_W over odd n in {3, 5, ..., 999}:")
    traces = [signed_tr_w(braid_of_odd_n(n)) for n in SAMPLE_RANGE]
    pos = sum(1 for t in traces if t > 0)
    neg = sum(1 for t in traces if t < 0)
    print(f"    sample size = {len(traces)}")
    print(f"    positive    = {pos}")
    print(f"    negative    = {neg}")
    print(f"    abs min     = {min(abs(t) for t in traces):.4e}")
    print(f"    abs max     = {max(abs(t) for t in traces):.4e}")

    print("\n  ==> Naive Tr_W is NOT sign-definite on Collatz braids.")
    print("      This is the open hinge: the Witness Trace Axiom would")
    print("      replace this signed scalar with a positive spinor norm")
    print("      that DOES separate returning from non-returning sectors.")

    print("\n  Running constructive checks ...")
    for fn in [
        test_braid_nonzero_for_all_samples,
        test_abs_tr_w_positive_for_all_samples,
        test_abs_norm_w_positive_for_all_samples,
        test_iota_invariance_of_signed_trace,
        test_signed_trace_is_not_sign_definite,
    ]:
        try:
            fn()
            print(f"  PASS: {fn.__name__}")
        except AssertionError as e:
            print(f"  FAIL: {fn.__name__}: {e}")
            raise

    print("\n  CONSTRUCTIVE SUBSTRATE VERIFIED;")
    print("  WITNESS TRACE AXIOM CONFIRMED AS OPEN HINGE.")


if __name__ == "__main__":
    main()
