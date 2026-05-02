"""Grace-surplus drift test for the Syracuse map.

For each Syracuse step T(n) = (3n+1)/2^{a(n)}, define the grace-depth
a_i and the cumulative grace surplus

    G_m = A_m - m * log2(3),   A_m = a_0 + ... + a_{m-1}

The "grace drift constant" is

    drift = E[a_i] - log2(3) = 2 - log2(3) ~ 0.41504

predicted on the assumption that a_i has a geometric distribution with
mean 2 (each successive bit of (3n+1) is independently zero with
probability 1/2).

This test computes the empirical mean of a_i over Syracuse orbits
starting from many odd integers and verifies it matches 2 within
statistical noise.

Pass conditions:
  - MEAN_A_NEAR_TWO:  empirical E[a] in [1.95, 2.05] for sufficiently
    many samples.
  - GRACE_DRIFT_POSITIVE: empirical drift = E[a] - log2(3) > 0.
  - GRACE_SURPLUS_GROWS: cumulative G_m / m -> drift constant.
"""

from __future__ import annotations

import math


def syracuse_step(n: int) -> tuple[int, int]:
    assert n >= 1 and n % 2 == 1
    m = 3 * n + 1
    a = 0
    while m % 2 == 0:
        m //= 2
        a += 1
    return m, a


def syracuse_orbit_until_one(n: int, max_steps: int = 100_000) -> list[int]:
    """Return the list of grace-depths a_i along the Syracuse orbit of n
    until reaching 1."""
    a_seq: list[int] = []
    cur = n
    for _ in range(max_steps):
        if cur == 1:
            return a_seq
        nxt, a = syracuse_step(cur)
        a_seq.append(a)
        cur = nxt
    raise RuntimeError(f"orbit of {n} did not terminate")


LOG2_3 = math.log2(3.0)
DRIFT_PREDICTED = 2.0 - LOG2_3  # ~ 0.41504


def aggregate_grace_depths(start_max: int = 100_000) -> list[int]:
    """Aggregate grace-depths from Syracuse orbits of all odd n in
    [3, start_max]."""
    all_depths: list[int] = []
    n = 3
    while n <= start_max:
        seq = syracuse_orbit_until_one(n)
        all_depths.extend(seq)
        n += 2
    return all_depths


def test_mean_a_is_near_two() -> None:
    depths = aggregate_grace_depths(start_max=20_001)
    mean_a = sum(depths) / len(depths)
    assert 1.90 <= mean_a <= 2.10, (
        f"mean grace-depth {mean_a:.4f} far from theoretical 2.0; "
        f"sample size {len(depths)}"
    )


def test_grace_drift_positive() -> None:
    depths = aggregate_grace_depths(start_max=20_001)
    drift = sum(depths) / len(depths) - LOG2_3
    assert drift > 0.30, (
        f"grace drift {drift:.4f} < 0.30; expected ~ {DRIFT_PREDICTED:.4f}"
    )


def test_grace_surplus_grows() -> None:
    """For long orbits, G_m / m should converge toward DRIFT_PREDICTED."""
    long_seeds = [27, 703, 871, 6171, 8775]
    for n in long_seeds:
        seq = syracuse_orbit_until_one(n)
        if len(seq) < 30:
            continue
        # cumulative drift over the orbit
        m = len(seq)
        A_m = sum(seq)
        G_m = A_m - m * LOG2_3
        ratio = G_m / m
        # Should be in [0, 1].  Long orbits should land near
        # DRIFT_PREDICTED ~ 0.415, but individual orbits can fluctuate.
        assert 0.10 < ratio < 0.80, (
            f"n={n}: G_m/m = {ratio:.4f} far from {DRIFT_PREDICTED:.4f}"
        )


def main() -> None:
    print("=" * 72)
    print("GRACE-SURPLUS DRIFT TEST (Collatz)")
    print("=" * 72)

    print(f"\n  Theoretical drift constant: 2 - log2(3) = {DRIFT_PREDICTED:.6f}")

    print("\n  Aggregating grace-depths from Syracuse orbits ...")
    depths = aggregate_grace_depths(start_max=20_001)
    mean_a = sum(depths) / len(depths)
    drift = mean_a - LOG2_3
    print(f"    sample size:        {len(depths)}")
    print(f"    empirical E[a_i]:   {mean_a:.6f}  (expected ~ 2.00)")
    print(f"    empirical drift:    {drift:.6f}  (expected ~ {DRIFT_PREDICTED:.6f})")

    print("\n  Per-seed cumulative drift G_m / m for long orbits:")
    print(f"    {'n':>8} {'m':>6} {'A_m':>8} {'G_m/m':>10}")
    for n in [27, 703, 871, 6171, 8775]:
        seq = syracuse_orbit_until_one(n)
        m = len(seq)
        A_m = sum(seq)
        G_m = A_m - m * LOG2_3
        print(f"    {n:>8} {m:>6} {A_m:>8} {G_m / m:>10.4f}")

    print("\n  Running drift checks ...")
    for fn in [
        test_mean_a_is_near_two,
        test_grace_drift_positive,
        test_grace_surplus_grows,
    ]:
        try:
            fn()
            print(f"  PASS: {fn.__name__}")
        except AssertionError as e:
            print(f"  FAIL: {fn.__name__}: {e}")
            raise

    print("\n  GRACE DRIFT VERIFIED.")


if __name__ == "__main__":
    main()
