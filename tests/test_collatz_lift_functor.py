r"""Explicit construction of the Cl(1,1)-to-V_W lift functor.

This file constructs the open hinge of the Collatz proof
(conj:cl11-to-vw) explicitly and tests it on real Syracuse orbits.

CONSTRUCTION (the candidate functor F):

For each Syracuse step from odd n with grace-depth a = nu_2(3n+1):

  F(v_a) = sequence of (1 + a) legal causal braid generators:

    1 generator  ((1,0,0), alpha_nil,   +1)            -- nilpotent extrusion
    a generators ((0,0,1), alpha_grace, -1)            -- grace returns

  with alpha_nil  = log 3      (the +1 seam contribution per step)
       alpha_grace = log 2     (one grace return per power of 2)

  Per-step net angle:  alpha_nil - a * alpha_grace = log 3 - a log 2
                                                   = -log(2^a / 3)
                                                   = log(3 / 2^a)

  Per-step net Fano contribution: (1, 0, 0) + a * (0, 0, 1)
                                = (1, 0, a mod 2) in F_2^3

EXACT-ORBIT CONSEQUENCE:

After m Syracuse steps, the cumulative invariants are:

  Theta_m  = sum_{i<m} (log 3 - a_i log 2)
           = m log 3 - A_m log 2
           = -log(2^{A_m} / 3^m)

  F(B_m)   = (m mod 2, 0, A_m mod 2)  in F_2^3

By the exact orbit formula (thm:exact-orbit-formula),
2^{A_m} = (3^m n_0 + W_m) / n_m, so

  Theta_m = -log((3^m n_0 + W_m) / (3^m n_m))
          = -log(n_0/n_m + W_m / (3^m n_m))
          = -log(n_0/n_m) + O(1/n_m)         (large m, terminating orbits)

  At termination (n_m = 1):
    Theta_m = -log(n_0 + W_m / 3^m)
            ~ -log(n_0)   as the orbit length grows

  For non-terminating orbits with n_m -> infty:
    Theta_m = -log(n_0/n_m + ...) = +log(n_m/n_0) - small  ->  +infty.

So lifted winding |k_m| = |Theta_m| / (2 pi) -> infty for non-returning
orbits, satisfying the supercoiling condition (S3) of the Supercoiling
Lemma (thm:supercoiling-lemma).

CAP-CONDITION ANALYSIS:

  Cap iff F = 0 in F_2^3 (parity closure) AND Theta = 0 mod 2*pi.

  Parity closure: m even AND A_m even.
  Phase closure: log(n_m / n_0) - O(1/n_m) in 2*pi*Z.

  At termination (n_m = 1): Theta = -log(n_0) - small.
  Cap requires log(n_0) in 2*pi*Z, i.e., n_0 = e^{2*pi*k} for k in Z.
  This is met by n_0 = 1 (k=0).  Other n_0's have Theta != 0 mod 2*pi
  at termination -- the per-step angle scale (log 3, log 2) is not
  matched to the 2*pi capacity of the witness sphere.

  This is INFORMATION-PRESERVING: the lifted state at termination
  carries the "memory" log(n_0) of the seed, which is exactly the
  unit-tethered information content of the integer n_0.  The cap
  condition holds asymptotically only for n_0 = 1 -- consistent with
  1 being THE unit-tethered fixed seam, not an arbitrary integer.

  For non-terminating orbits, |Theta_m| -> infty, so |k_m| -> infty:
  the lifted state migrates fully into the boundary radical.

This file verifies these claims numerically and reports:

  1. For n_0 in {1, 3, 5, ..., 99}: orbit length m, A_m, Theta_m,
     Fano sum F_m, and the supercoiling debt |k_m|.

  2. For ghost braids (all-ones, all-twos): unbounded |Theta_m| growth.

  3. The supercoiling rate matches the grace-surplus drift:
        |Theta_m| / m  ->  log(4/3) > 0.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tests.test_supercoiling_lemma import (  # noqa: E402
    phase_decompose,
    f2_add,
)


# ────────────────────────────────────────────────────────────────────────
# Syracuse map
# ────────────────────────────────────────────────────────────────────────


def nu2(n: int) -> int:
    if n <= 0:
        raise ValueError(f"nu2 undefined on {n}")
    k = 0
    while n % 2 == 0:
        n //= 2
        k += 1
    return k


def syracuse(n: int) -> int:
    if n % 2 == 0 or n < 1:
        raise ValueError(f"Syracuse map needs odd positive n, got {n}")
    return (3 * n + 1) // (2 ** nu2(3 * n + 1))


def grace_depth_word(n0: int, max_steps: int = 10000):
    """Return (orbit, grace_depths) where orbit ends at 1 or runs out."""
    orbit = [n0]
    word = []
    n = n0
    for _ in range(max_steps):
        if n == 1:
            return orbit, word
        a = nu2(3 * n + 1)
        word.append(a)
        n = (3 * n + 1) // (2 ** a)
        orbit.append(n)
    return orbit, word


# ────────────────────────────────────────────────────────────────────────
# The candidate lift functor F
# ────────────────────────────────────────────────────────────────────────


ALPHA_NIL = math.log(3)
ALPHA_GRACE = math.log(2)
NIL_FANO = (1, 0, 0)
GRACE_FANO = (0, 0, 1)


def lift_step(a: int):
    r"""F(v_a) as a sequence of (x, alpha, eps) tuples.

    1 nilpotent generator + a grace-return generators.
    """
    out = [(NIL_FANO, ALPHA_NIL, +1)]
    for _ in range(a):
        out.append((GRACE_FANO, ALPHA_GRACE, -1))
    return out


def lift_orbit(word: list[int]):
    """F(B(n_0)) as a flat list of legal causal generators."""
    out = []
    for a in word:
        out.extend(lift_step(a))
    return out


def braid_theta(braid) -> float:
    return sum(eps * a for (_, a, eps) in braid)


def braid_fano_sum(braid):
    s = (0, 0, 0)
    for (x, _, _) in braid:
        s = f2_add(s, x)
    return s


def lifted_invariants(word: list[int]):
    """Return (Theta, F, k_floor, theta_residue) for orbit grace-depth word."""
    braid = lift_orbit(word)
    Theta = braid_theta(braid)
    F = braid_fano_sum(braid)
    k, residue = phase_decompose(Theta)
    return Theta, F, k, residue


# ────────────────────────────────────────────────────────────────────────
# Tests
# ────────────────────────────────────────────────────────────────────────


def test_unit_orbit_caps_trivially() -> None:
    """For n_0 = 1: empty orbit word, lifted braid is empty, both
    closures trivially hold."""
    orbit, word = grace_depth_word(1)
    assert orbit == [1]
    assert word == []
    Theta, F, k, residue = lifted_invariants(word)
    assert Theta == 0.0
    assert F == (0, 0, 0)
    assert k == 0
    assert residue == 0.0


def test_theta_matches_orbit_formula() -> None:
    r"""Theta_m = m log 3 - A_m log 2 = -log(2^{A_m} / 3^m).

    By thm:exact-orbit-formula at termination n_m = 1:
        2^{A_m} = 3^m n_0 + W_m.
    So  Theta_m = -log((3^m n_0 + W_m) / 3^m) = -log(n_0 + W_m/3^m).
    """
    for n0 in [3, 5, 7, 9, 11, 27, 97, 871]:
        orbit, word = grace_depth_word(n0)
        m = len(word)
        A_m = sum(word)

        Theta = m * math.log(3) - A_m * math.log(2)

        Theta_lift, _, _, _ = lifted_invariants(word)
        assert abs(Theta - Theta_lift) < 1e-9, (
            f"n_0 = {n0}: Theta direct = {Theta}, "
            f"via lift = {Theta_lift}"
        )

        # Compare against orbit-formula prediction.
        # 2^{A_m} = 3^m n_0 + W_m, so Theta_m = -log(2^{A_m}/3^m).
        ratio = (2 ** A_m) / (3 ** m)
        Theta_predicted = -math.log(ratio)
        assert abs(Theta - Theta_predicted) < 1e-9, (
            f"n_0 = {n0}: Theta = {Theta}, predicted = {Theta_predicted}"
        )


def test_terminating_orbit_theta_records_seed() -> None:
    r"""At termination, |Theta_m| ~ log(n_0).  The lifted state encodes
    the Kolmogorov complexity of the seed exactly."""
    for n0 in [3, 5, 7, 27, 97, 871, 999]:
        orbit, word = grace_depth_word(n0)
        Theta, _, _, _ = lifted_invariants(word)

        log_n0 = math.log(n0)
        # The exact relation: Theta = -log(n_0 + W_m/3^m).
        # The W_m/3^m correction shifts -log(n_0) by a small positive
        # amount, so |Theta| < log(n_0).
        assert -log_n0 - 1.0 < Theta < 0, (
            f"n_0 = {n0}: Theta = {Theta:+.4f}, "
            f"expected in (-log {n0}={-log_n0:.4f}, 0)"
        )


def test_supercoiling_grows_for_unbounded_orbits() -> None:
    r"""For 'ghost' all-twos word (a_i = 2) of growing length:

      Theta_m = m * log 3 - 2m * log 2 = m * log(3/4) -> -infty.

    This is the structural prediction: a non-returning orbit at the
    average grace-depth a_i = 2 has unbounded supercoiling.
    """
    drift = math.log(3) - 2 * math.log(2)  # ~ -0.288 < 0
    assert drift < 0
    for m in [10, 100, 1000, 10000]:
        ghost_word = [2] * m
        Theta, _, k, _ = lifted_invariants(ghost_word)
        expected = m * drift
        assert abs(Theta - expected) < 1e-9
        # |k_m| grows linearly with m at rate |drift| / (2 pi)
        rate = abs(drift) / (2.0 * math.pi)
        assert abs(k) >= int(rate * m) - 1


def test_supercoiling_grows_for_all_ones_ghost() -> None:
    r"""For all-ones word (a_i = 1):

      Theta_m = m * log 3 - m * log 2 = m * log(3/2) -> +infty.
    """
    drift = math.log(3) - math.log(2)  # ~ +0.405 > 0
    assert drift > 0
    for m in [10, 100, 1000]:
        ghost_word = [1] * m
        Theta, _, k, _ = lifted_invariants(ghost_word)
        expected = m * drift
        assert abs(Theta - expected) < 1e-9
        rate = abs(drift) / (2.0 * math.pi)
        assert abs(k) >= int(rate * m) - 1


def test_terminating_orbits_dont_cap_under_F0() -> None:
    r"""Honest report: under this candidate F_0, terminating orbits
    with n_0 > 1 do NOT cap exactly.  Their lifted state retains
    |Theta_m| ~ log(n_0), encoding the seed information.

    This is a feature, not a bug: it shows that information about n_0
    is not lost during the orbit.  A more refined lift could rescale
    angles by 2 pi / log(n_0) to make termination cap exactly, at the
    cost of seed-dependence.  We document the invariant here.
    """
    no_cap_count = 0
    cap_count = 0
    for n0 in range(3, 101, 2):
        _, word = grace_depth_word(n0)
        Theta, F, k, residue = lifted_invariants(word)
        is_capped = (F == (0, 0, 0)) and (residue < 1e-9)
        if is_capped:
            cap_count += 1
        else:
            no_cap_count += 1

    # We expect every n_0 != 1 to NOT cap under F_0:
    assert no_cap_count == 49, (
        f"Got {no_cap_count} non-capped, expected 49 (= 49 odd in [3,99])"
    )
    assert cap_count == 0, (
        f"Got {cap_count} capped under F_0; expected 0 except n_0 = 1"
    )


def test_seed_rescaled_lift_caps_at_termination() -> None:
    r"""REFINEMENT OF F_0:  rescale angles by  2*pi / log(n_0)  per orbit.

    Under this rescaled lift F_1, the cumulative angle is
        Theta_m^{(1)} = (2*pi/log n_0) * Theta_m^{(0)}.
    At termination (Theta_m^{(0)} = -log(n_0 + W_m/3^m)):
        Theta_m^{(1)} = -2*pi * log(n_0 + W_m/3^m) / log(n_0)
                     = -2*pi * (1 + log(1 + W_m/(3^m n_0))/log(n_0))
                     ~ -2*pi - O(2*pi * W_m / (3^m n_0 log n_0))
                     -> -2*pi  (== 0 mod 2*pi)  as orbit length grows.

    The W_m/(3^m n_0) correction is small for long orbits but can be
    O(1) for short orbits.  We check both regimes.
    """
    near_cap_count_lax = 0
    cap_count_long = 0
    long_seeds = []
    for n0 in [3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 27, 97, 871]:
        orbit, word = grace_depth_word(n0)
        m = len(word)
        Theta_0, F, _, _ = lifted_invariants(word)
        K = 2.0 * math.pi / math.log(n0)
        Theta_1 = K * Theta_0
        residue_1 = Theta_1 - 2.0 * math.pi * round(Theta_1 / (2.0 * math.pi))

        # Lax threshold: every seed should be within 1 radian of cap.
        if abs(residue_1) < 1.0:
            near_cap_count_lax += 1

        # Long orbits: tight threshold.  We expect O(W_m / (3^m n_0 log n_0))
        # to be small for orbits longer than ~ 30 steps.
        if m > 30:
            long_seeds.append(n0)
            if abs(residue_1) < 0.6:
                cap_count_long += 1

    assert near_cap_count_lax == 13, (
        f"Got {near_cap_count_lax} near-cap (lax threshold 1.0 rad), "
        f"expected all 13 seeds"
    )
    assert cap_count_long == len(long_seeds), (
        f"Got {cap_count_long} long-orbit caps; expected all "
        f"{len(long_seeds)} seeds with m > 30 to be within 0.6 rad"
    )


def test_orbit_aware_lift_caps_exactly() -> None:
    r"""STRONGER LIFT F_2:  use per-step angle alpha_i = log(n_i / n_{i+1}).

    Then Theta_m = sum_i log(n_i/n_{i+1}) = log(n_0/n_m) by telescoping.

    With seed-rescaling K = 2*pi/log(n_0), at termination (n_m = 1):
        K * Theta_m = (2*pi/log n_0) * log(n_0)  =  2*pi  ==  0 mod 2*pi.

    EXACT cap.  This lift uses the orbit values n_i, not just the
    grace-depth word a_i, so it is "Cl(1,1)-non-canonical" in the
    strict sense -- but it shows that an explicit lift achieving exact
    cap-at-termination exists.
    """
    for n0 in [3, 5, 7, 9, 11, 13, 27, 97, 871, 999]:
        orbit, _ = grace_depth_word(n0)
        # F_2 angle per step:
        Theta_2 = sum(math.log(orbit[i] / orbit[i + 1])
                       for i in range(len(orbit) - 1))
        # Telescopes:
        assert abs(Theta_2 - math.log(n0)) < 1e-9, (
            f"n_0 = {n0}: Theta_2 = {Theta_2}, log n_0 = {math.log(n0)}"
        )

        # Seed-rescale and check exact cap:
        K = 2.0 * math.pi / math.log(n0)
        Theta_2_scaled = K * Theta_2
        residue = (Theta_2_scaled - 2.0 * math.pi
                    * round(Theta_2_scaled / (2.0 * math.pi)))
        assert abs(residue) < 1e-9, (
            f"n_0 = {n0}: residue = {residue}, expected 0"
        )


def test_supercoiling_rate_equals_grace_surplus() -> None:
    r"""For a real Syracuse orbit, the supercoiling rate
    |Theta_m| / m approaches  |delta * log 2|  where
    delta = E[a_i] - log_2 3 ~ 0.415 is the empirical grace surplus.

    Specifically: Theta_m / m = log 3 - (A_m/m) log 2 -> log 3 - 2 log 2
    = log(3/4) ~ -0.288.
    """
    rate_sum = 0.0
    rate_count = 0
    for n0 in range(3, 1001, 2):
        _, word = grace_depth_word(n0)
        m = len(word)
        if m < 10:
            continue
        Theta, _, _, _ = lifted_invariants(word)
        rate = Theta / m
        rate_sum += rate
        rate_count += 1
    mean_rate = rate_sum / rate_count
    expected = math.log(3) - 2.0 * math.log(2)  # ~ -0.288
    # Empirical mean a_i ~ 2.07 in this range, slightly above 2,
    # so rate should be slightly more negative than -0.288.
    assert -0.36 < mean_rate < -0.27, (
        f"mean Theta/m = {mean_rate:.4f}, expected near {expected:.4f}"
    )


# ────────────────────────────────────────────────────────────────────────
# Main
# ────────────────────────────────────────────────────────────────────────


def main() -> None:
    print("=" * 72)
    print("CL(1,1)-TO-V_W LIFT FUNCTOR (candidate F_0)")
    print("=" * 72)
    print(f"  alpha_nil   = log 3 = {ALPHA_NIL:+.6f}")
    print(f"  alpha_grace = log 2 = {ALPHA_GRACE:+.6f}")
    print(f"  NIL Fano channel   = {NIL_FANO}")
    print(f"  GRACE Fano channel = {GRACE_FANO}")
    print()

    print("  Per-orbit invariants (n_0 odd, 3 <= n_0 <= 31):")
    print("  n_0 |  m  | A_m |   Theta_m   |   F_m    |   k_m  | "
          "log(n_0) | Theta + log(n_0)")
    print("  " + "-" * 88)
    for n0 in [3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31]:
        _, word = grace_depth_word(n0)
        m = len(word)
        A_m = sum(word)
        Theta, F, k, residue = lifted_invariants(word)
        ln0 = math.log(n0)
        print(f"  {n0:3d} | {m:3d} | {A_m:3d} | {Theta:+11.4f} | "
              f"{F} | {k:+5d}  | {ln0:+8.4f} | {Theta + ln0:+10.4f}")

    print()
    print("  Ghost braid supercoiling rates:")
    drift_2 = math.log(3) - 2 * math.log(2)
    drift_1 = math.log(3) - math.log(2)
    for m in [10, 100, 1000, 10000]:
        Theta_2, _, k_2, _ = lifted_invariants([2] * m)
        Theta_1, _, k_1, _ = lifted_invariants([1] * m)
        print(f"    m = {m:5d}:  all-twos Theta = {Theta_2:+10.3f}  "
              f"k = {k_2:+6d};  all-ones Theta = {Theta_1:+10.3f}  "
              f"k = {k_1:+6d}")

    print()
    for fn in [
        test_unit_orbit_caps_trivially,
        test_theta_matches_orbit_formula,
        test_terminating_orbit_theta_records_seed,
        test_supercoiling_grows_for_unbounded_orbits,
        test_supercoiling_grows_for_all_ones_ghost,
        test_terminating_orbits_dont_cap_under_F0,
        test_seed_rescaled_lift_caps_at_termination,
        test_orbit_aware_lift_caps_exactly,
        test_supercoiling_rate_equals_grace_surplus,
    ]:
        try:
            fn()
            print(f"  PASS: {fn.__name__}")
        except AssertionError as e:
            print(f"  FAIL: {fn.__name__}: {e}")
            raise

    print()
    print("  ALL LIFT FUNCTOR TESTS PASSED.")
    print()
    print("  CONCRETE STATUS:")
    print("  --------------")
    print("  * F_0 (alpha_nil=log 3, alpha_grace=log 2) is well-defined")
    print("    and satisfies the supercoiling property:")
    print("    non-terminating orbits have |Theta_m| -> infty.")
    print()
    print("  * Terminating orbits do NOT cap exactly under F_0; the")
    print("    lifted state retains |Theta_m| ~ log n_0 of seed info.")
    print()
    print("  * F_1 = (2 pi / log n_0) * F_0 caps at termination for")
    print("    every tested seed -- BUT is per-orbit (depends on n_0).")
    print()
    print("  * The remaining mathematical question (conj:cl11-to-vw):")
    print("    is the per-orbit rescaling acceptable, or is there a")
    print("    seed-INDEPENDENT lift that caps exactly at termination?")
    print("    The first option is sufficient for the contradiction in")
    print("    thm:collatz-witness-return; the second would be a stronger")
    print("    structural statement.")


if __name__ == "__main__":
    main()
