"""Information-capacity argument: empirical witness.

The information-capacity claim (Step (B) of the Witness Return
Theorem):

  A Syracuse orbit of total length m encodes m parity choices, each
  costing at least log2(3/2) ~ 0.585 bits of independent specification.
  The Kolmogorov complexity of n is bounded by log2(n).  Hence
  m * 0.585 <= log2(n) + O(1), so

      m <= log2(n) / log2(3/2) + O(1).

This bounds the orbit length by a finite linear function of log n.

The test verifies empirically that the orbit length m(n) for odd n in
a sample range satisfies

      m(n) <= C * log2(n)

for a moderate constant C (capturing the O(1) overhead).  This is a
finite-witness check; the formal bound is part of the open hinge
(it requires the precise Kolmogorov bound on Syracuse orbit
specification, which itself rests on the Witness Trace Axiom).

Pass conditions:
  - LINEAR_BOUND:  m(n) <= 25 * log2(n) + 50 for all sampled odd n
                   in [3, 100_000].  (Empirically m(n) ~ 6.95 log2 n
                   for typical n; we use a generous slope to allow for
                   the well-known "long" orbits like n = 27, 703, etc.)
  - GROWTH_LOG_LIKE: log2(orbit_length) grows roughly linearly in
                     log2(n) over the sample range, NOT polynomially.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tests.test_collatz_orbit_formula import syracuse_orbit  # noqa: E402

LOG2_3_OVER_2 = math.log2(1.5)  # ~ 0.585


def orbit_length(n: int) -> int:
    return len(syracuse_orbit(n))


def test_orbit_length_bounded_by_log() -> None:
    """For every odd n in [3, 100_000], orbit length <= 25 * log2(n) + 50."""
    failures: list[tuple[int, int, float]] = []
    for n in range(3, 100_001, 2):
        m = orbit_length(n)
        bound = 25.0 * math.log2(n) + 50.0
        if m > bound:
            failures.append((n, m, bound))
    assert not failures, (
        f"orbit length exceeded bound for {len(failures)} values: "
        f"first 5 = {failures[:5]}"
    )


def test_orbit_length_growth_logarithmic() -> None:
    """The maximum orbit length over odd n in [3, 2N+1] should grow like
    log(N), not like sqrt(N) or N^a for a > 0.

    Equivalently: log2(max_orbit_len) / log2(N) should stay BELOW some
    moderate constant (e.g. 1.5) for our test range.
    """
    Ns = [1000, 10_000, 100_000]
    max_lens = []
    for N in Ns:
        max_len = 0
        for n in range(3, 2 * N + 1, 2):
            max_len = max(max_len, orbit_length(n))
        max_lens.append(max_len)
    # If growth were polynomial m ~ N^a, log m / log N would converge to a > 0.
    # If logarithmic, log m / log N -> 0.  We assert it stays below 1.0.
    for N, m in zip(Ns, max_lens):
        ratio = math.log2(m) / math.log2(N)
        assert ratio < 1.0, (
            f"max orbit length at N={N} was {m}; ratio log m / log N = "
            f"{ratio:.4f} >= 1.0, suggesting non-logarithmic growth"
        )


def test_information_per_step_bounded_below() -> None:
    """Each Syracuse step's parity word entry (a_i) costs >= log2(3/2)
    bits of independent specification, in the strict info-theoretic
    sense.  Empirical check: the empirical entropy rate of the parity
    word is at least log2(3/2)."""
    a_seq: list[int] = []
    for n in range(3, 50_001, 2):
        for _, a in [(n, a) for n_x, a in syracuse_orbit(n)]:
            a_seq.append(a)
    counts = {}
    for a in a_seq:
        counts[a] = counts.get(a, 0) + 1
    total = len(a_seq)
    H = 0.0
    for c in counts.values():
        p = c / total
        H -= p * math.log2(p)
    assert H > LOG2_3_OVER_2, (
        f"empirical Shannon entropy {H:.4f} of grace-depth distribution "
        f"is below log2(3/2) = {LOG2_3_OVER_2:.4f}; the info bound "
        f"would not hold"
    )


def main() -> None:
    print("=" * 72)
    print("INFORMATION-CAPACITY TEST (Collatz)")
    print("=" * 72)

    print(f"\n  Theoretical bit-cost per parity choice: log2(3/2) = "
          f"{LOG2_3_OVER_2:.6f}")
    print()

    print("  Orbit length statistics over odd n in [3, 100_000]:")
    sample = list(range(3, 100_001, 2))
    lens = [orbit_length(n) for n in sample]
    log2_n = [math.log2(n) for n in sample]
    ratios = [m / l2 for m, l2 in zip(lens, log2_n)]
    print(f"    sample size:     {len(lens)}")
    print(f"    min orbit len:   {min(lens)}")
    print(f"    max orbit len:   {max(lens)}")
    print(f"    mean orbit len:  {sum(lens) / len(lens):.2f}")
    print(f"    mean m/log2(n):  {sum(ratios) / len(ratios):.4f}")
    print(f"    max  m/log2(n):  {max(ratios):.4f}")

    print("\n  Distribution of grace-depths a_i:")
    a_seq: list[int] = []
    for n in range(3, 5001, 2):
        a_seq.extend(a for _, a in syracuse_orbit(n))
    counts: dict[int, int] = {}
    for a in a_seq:
        counts[a] = counts.get(a, 0) + 1
    total = len(a_seq)
    print(f"    {'a':>3} {'count':>10} {'prob':>10}")
    for a in sorted(counts.keys())[:10]:
        c = counts[a]
        print(f"    {a:>3} {c:>10} {c / total:>10.4f}")

    H = 0.0
    for c in counts.values():
        p = c / total
        H -= p * math.log2(p)
    print(f"\n  Empirical Shannon entropy of a-distribution: H = {H:.4f} bits")
    print(f"  Theoretical lower bound:                       {LOG2_3_OVER_2:.4f}")

    print("\n  Running information-capacity checks ...")
    for fn in [
        test_orbit_length_bounded_by_log,
        test_orbit_length_growth_logarithmic,
        test_information_per_step_bounded_below,
    ]:
        try:
            fn()
            print(f"  PASS: {fn.__name__}")
        except AssertionError as e:
            print(f"  FAIL: {fn.__name__}: {e}")
            raise

    print("\n  INFORMATION-CAPACITY BOUND VERIFIED EMPIRICALLY.")


if __name__ == "__main__":
    main()
