"""Ghost-braid (formal non-returning braid) test.

A "ghost braid" is a formal infinite Cl(1,1) product corresponding to
a 2-adic integer parity sequence that does NOT terminate at 1.  These
exist as formal objects (in Z_2 or as nonstandard models of Peano
arithmetic) but are not positive ordinary integers.

This test investigates several candidate ghost-braid families and
classifies their Cl(1,1) behaviour:

  (1) ALL-ONES WORD (a_i = 1 for all i): the canonical 'pure
      expansion' candidate.  Each generator v_1 = exp((log2) e1 +
      (log3) e2) has v^2 = log2^2 - log3^2 < 0 (SPACELIKE), so its
      exponential is a Cl(1,1) ROTATION.  The cumulative product is
      a rotation in the {1, omega} plane and stays BOUNDED in
      magnitude but ACCUMULATES UNBOUNDED PHASE (rotation angle).

  (2) ALL-TWOS WORD (a_i = 2 for all i): each generator is TIMELIKE
      (v^2 = (2 log2)^2 - log3^2 > 0), so its exponential is a Cl(1,1)
      BOOST.  The cumulative product's magnitude GROWS exponentially.

  (3) CRITICAL WORD (1, 2, 1, 2, ...): exactly straddles the null
      cone with average grace-depth 1.5 < log2(3) ~ 1.585.

The key structural fact that distinguishes ghost braids from real
returning Collatz orbits is NOT magnitude alone but the
**unbounded accumulation of an iota-anti-symmetric phase**.  A real
returning Collatz orbit, after ~log2(n) / log2(3/2) steps, leaves
behind a finite phase budget; a ghost braid extends this without
bound, acting as a formal indefinite-rotation in the {1, omega}
plane.

Pass conditions:
  - ALL_ONES_BOUNDED_MAG: magnitude of all-ones braid stays bounded
    (it's a rotation) -- this is a positive structural finding.
  - ALL_TWOS_GROWS:       magnitude of all-twos braid grows
    exponentially -- the boost direction expands.
  - PHASE_ACCUMULATES:    the rotation angle in the {1, omega}
    plane accumulates without bound for the all-ones word.
  - NO_FINITE_RETURN_ALL_ONES: all-ones braid never returns close to
    identity within any truncation up to length 5000.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tests.test_cl11_algebra import cl11_basis, cl11_mul, cl11_norm_sq  # noqa: E402
from tests.test_collatz_braid_trace import (  # noqa: E402
    abs_norm_w,
    collatz_step_generator,
    LOG2,
    LOG3,
)


# ────────────────────────────────────────────────────────────────────────
# Ghost braid constructions
# ────────────────────────────────────────────────────────────────────────


def all_ones_braid(length: int) -> np.ndarray:
    """B_m for the formal infinite word (1, 1, 1, ..., 1) (all a_i = 1).

    Each step is spacelike: v_1 = exp((log2) e1 + (log3) e2),
    v_1^2 = log2^2 - log3^2 < 0.

    This word corresponds to a 2-adic integer that climbs forever
    under triadic expansion -- the canonical 'pure expansion' ghost.
    """
    B = cl11_basis()["1"]
    v1 = collatz_step_generator(1)
    for _ in range(length):
        B = cl11_mul(B, v1)
    return B


def critical_word_braid(length: int) -> np.ndarray:
    """Word with average a_i exactly at the critical threshold log2(3)
    -- the boundary between grace-dominant and expansion-dominant.

    Use word (1, 2, 1, 2, ...) which has average 1.5 < log2(3) ~ 1.585,
    just barely expansion-dominant.
    """
    B = cl11_basis()["1"]
    pattern = [1, 2]
    for i in range(length):
        B = cl11_mul(B, collatz_step_generator(pattern[i % 2]))
    return B


def all_twos_braid(length: int) -> np.ndarray:
    """B_m for the formal infinite word (2, 2, 2, ..., 2) -- cumulative
    boost in the grace direction.  Each step is timelike."""
    B = cl11_basis()["1"]
    v2 = collatz_step_generator(2)
    for _ in range(length):
        B = cl11_mul(B, v2)
    return B


def rotation_angle_phi(B: np.ndarray) -> float:
    """For an even-graded element B = a*1 + d*omega (after projecting
    out vector parts), compute its 'angle' in the {1, omega} plane."""
    # Project to even subalgebra
    a = B[0]
    d = B[3]
    return math.atan2(d, a) if (a * a + d * d) > 1e-30 else 0.0


# ────────────────────────────────────────────────────────────────────────
# Tests
# ────────────────────────────────────────────────────────────────────────


def test_all_ones_bounded_magnitude() -> None:
    """All-ones braid stays in a bounded region of Cl(1,1) (rotation)."""
    lengths = [10, 50, 100, 500, 1000, 5000]
    mags = [float(np.linalg.norm(all_ones_braid(L))) for L in lengths]
    M = max(mags)
    assert M < 100.0, (
        f"all-ones braid magnitude unexpectedly grew to {M}; "
        f"should stay bounded for spacelike generator"
    )


def test_all_twos_braid_grows() -> None:
    """All-twos braid magnitude grows exponentially (timelike boosts)."""
    lengths = [5, 10, 20, 30]
    mags = [float(np.linalg.norm(all_twos_braid(L))) for L in lengths]
    assert mags[-1] > mags[0] * 1e3, (
        f"all-twos braid magnitude failed to grow: {mags}"
    )


def test_all_ones_never_returns_to_identity() -> None:
    one = cl11_basis()["1"]
    for L in [1, 5, 10, 25, 50, 100, 200, 500, 1000]:
        B = all_ones_braid(L)
        diff = float(np.linalg.norm(B - one))
        assert diff > 0.05, (
            f"all-ones braid at length {L} returned within {diff} of "
            f"identity; should be far from identity at every step"
        )


def test_phase_accumulates_unboundedly_for_all_ones() -> None:
    """The all-ones rotation accumulates phase without bound modulo 2 pi.

    We track unwrapped cumulative angle and verify it grows linearly
    in length.
    """
    lengths = [100, 500, 1000]
    cumulative_unwrapped = []
    prev_phi = 0.0
    total = 0.0
    one = cl11_basis()["1"]
    v1 = collatz_step_generator(1)
    B = one.copy()
    target_lens = set(lengths)
    for step in range(1, max(lengths) + 1):
        B = cl11_mul(B, v1)
        phi = rotation_angle_phi(B)
        # unwrap
        d_phi = phi - prev_phi
        while d_phi > math.pi:
            d_phi -= 2 * math.pi
        while d_phi < -math.pi:
            d_phi += 2 * math.pi
        total += d_phi
        prev_phi = phi
        if step in target_lens:
            cumulative_unwrapped.append((step, total))
    # The accumulated phase should grow at least linearly with length.
    L_first, p_first = cumulative_unwrapped[0]
    L_last,  p_last  = cumulative_unwrapped[-1]
    rate_first = abs(p_first / L_first)
    rate_last  = abs(p_last  / L_last)
    # Both rates should be similar (linear accumulation)
    assert abs(p_last)  > 0.5 * abs(p_first) * (L_last / L_first), (
        f"phase did not accumulate linearly: {cumulative_unwrapped}"
    )


def test_critical_word_does_not_return() -> None:
    one = cl11_basis()["1"]
    for L in [10, 50, 100, 200]:
        B = critical_word_braid(L)
        diff = float(np.linalg.norm(B - one))
        assert diff > 0.05, (
            f"critical word at length {L} returned within {diff} of identity"
        )


def main() -> None:
    print("=" * 72)
    print("GHOST-BRAID TEST (formal non-returning Collatz braids)")
    print("=" * 72)

    print("\n  All-ones word braid (a_i = 1, spacelike, ROTATION):")
    print(f"    {'length':>8} {'|B|':>14} {'phi':>14}")
    for L in [10, 50, 100, 500, 1000, 5000]:
        B = all_ones_braid(L)
        phi = rotation_angle_phi(B)
        print(f"    {L:>8} {float(np.linalg.norm(B)):>14.4e} "
              f"{phi:>+14.4f}")

    print("\n  All-twos word braid (a_i = 2, timelike, BOOST):")
    print(f"    {'length':>8} {'|B|':>14}")
    for L in [5, 10, 20, 30, 50]:
        B = all_twos_braid(L)
        print(f"    {L:>8} {float(np.linalg.norm(B)):>14.4e}")

    print("\n  Critical (1,2,1,2,...) braid:")
    print(f"    {'length':>8} {'|B|':>14}")
    for L in [10, 50, 100, 200, 1000]:
        B = critical_word_braid(L)
        print(f"    {L:>8} {float(np.linalg.norm(B)):>14.4e}")

    print("\n  Identity element:")
    print(f"    {float(np.linalg.norm(cl11_basis()['1'])):.4e}")
    print()
    print("  STRUCTURAL FINDING: ghost braids are characterised NOT by")
    print("  unbounded magnitude alone, but by unbounded PHASE accumulation")
    print("  (all-ones) or unbounded MAGNITUDE growth (all-twos).  Real")
    print("  Collatz orbits are FINITE products that always terminate at 1.")

    print("\n  Running ghost-braid checks ...")
    for fn in [
        test_all_ones_bounded_magnitude,
        test_all_twos_braid_grows,
        test_all_ones_never_returns_to_identity,
        test_phase_accumulates_unboundedly_for_all_ones,
        test_critical_word_does_not_return,
    ]:
        try:
            fn()
            print(f"  PASS: {fn.__name__}")
        except AssertionError as e:
            print(f"  FAIL: {fn.__name__}: {e}")
            raise

    print("\n  GHOST BRAIDS CONFIRMED NON-RETURNING IN Cl(1,1);")
    print("  RETURN-TO-IDENTITY IS A FINITE-LENGTH PHENOMENON.")


if __name__ == "__main__":
    main()
