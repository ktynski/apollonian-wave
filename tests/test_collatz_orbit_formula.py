"""Exact Collatz orbit formula test.

For odd n >= 1 the Syracuse map is

    T(n) := (3n + 1) / 2^{nu_2(3n + 1)}

After m steps with grade-depth word (a_0, a_1, ..., a_{m-1}),

    n_m = (3^m * n_0 + W_m) / 2^{A_m}

where
    A_m = a_0 + ... + a_{m-1}
    W_m = sum_{j=0}^{m-1} 3^{m-1-j} * 2^{A_j}

This identity is **rigorous** (proved by induction).  This test verifies
it numerically for several known long Collatz orbits.

Pass conditions:
  - FORMULA_MATCHES_DIRECT: For each test n, the orbit formula produces
    the same sequence as direct iteration.
  - LARGE_ORBITS_OK: Tested up to n = 27 (orbit length 41 odd steps),
    n = 97, n = 871, and a few random large odd integers.
"""

from __future__ import annotations


def syracuse_step(n: int) -> tuple[int, int]:
    """One Syracuse step: n -> (T(n), a) for odd n >= 1.

    Returns (T(n), a) where a = nu_2(3n + 1).
    """
    assert n >= 1 and n % 2 == 1, f"need odd positive n, got {n}"
    m = 3 * n + 1
    a = 0
    while m % 2 == 0:
        m //= 2
        a += 1
    return m, a


def syracuse_orbit(n: int, max_steps: int = 10_000) -> list[tuple[int, int]]:
    """Direct iteration. Returns list of (n_i, a_i) until reaching 1."""
    out: list[tuple[int, int]] = []
    cur = n
    for _ in range(max_steps):
        if cur == 1:
            return out
        nxt, a = syracuse_step(cur)
        out.append((cur, a))
        cur = nxt
    raise RuntimeError(f"orbit of {n} did not reach 1 in {max_steps} steps")


def orbit_formula(n0: int, word: list[int]) -> int:
    """Compute n_m via the closed-form orbit formula.

    n_m = (3^m * n_0 + W_m) / 2^{A_m}

    Returns n_m as an integer (the formula always gives an integer when
    `word` is the actual Syracuse word starting at n0).
    """
    m = len(word)
    A_partial = [0]
    for a in word:
        A_partial.append(A_partial[-1] + a)
    A_m = A_partial[-1]
    W_m = sum((3 ** (m - 1 - j)) * (2 ** A_partial[j]) for j in range(m))
    num = (3 ** m) * n0 + W_m
    denom = 2 ** A_m
    assert num % denom == 0, (
        f"orbit formula non-integer: num={num}, denom={denom}, "
        f"n0={n0}, word={word}"
    )
    return num // denom


def test_orbit_formula_matches_direct_for_n3() -> None:
    n0 = 3
    direct = syracuse_orbit(n0)
    word = [a for _, a in direct]
    nm = orbit_formula(n0, word)
    assert nm == 1, f"orbit formula says n_m = {nm}, expected 1"


def test_orbit_formula_matches_direct_for_n27() -> None:
    n0 = 27
    direct = syracuse_orbit(n0)
    word = [a for _, a in direct]
    nm = orbit_formula(n0, word)
    assert nm == 1, f"orbit formula says n_m = {nm}, expected 1"
    # The 27 orbit is famously long.
    assert len(word) > 30, (
        f"n=27 should have a long orbit; got only {len(word)} odd steps"
    )


def test_orbit_formula_matches_direct_for_n97() -> None:
    n0 = 97
    direct = syracuse_orbit(n0)
    word = [a for _, a in direct]
    nm = orbit_formula(n0, word)
    assert nm == 1


def test_orbit_formula_matches_direct_for_n871() -> None:
    n0 = 871
    direct = syracuse_orbit(n0)
    word = [a for _, a in direct]
    nm = orbit_formula(n0, word)
    assert nm == 1


def test_partial_orbit_formula_matches_direct() -> None:
    """Check the formula step-by-step, not just at the end."""
    n0 = 27
    direct = syracuse_orbit(n0)
    word = [a for _, a in direct]
    cur = n0
    for k in range(1, len(word) + 1):
        # closed form for partial word (a_0, ..., a_{k-1}) starting at n0
        # gives n_k.  The actual n_k after k odd steps is direct[k][0]
        # if k < len, else 1.
        expected = direct[k][0] if k < len(direct) else 1
        got = orbit_formula(n0, word[:k])
        assert got == expected, (
            f"partial orbit at k={k}: formula gives {got}, "
            f"direct iteration gives {expected}"
        )


def test_random_odd_integers() -> None:
    import random
    rng = random.Random(2026)
    for _ in range(40):
        n0 = 2 * rng.randint(1, 10**6) + 1
        try:
            direct = syracuse_orbit(n0, max_steps=5000)
        except RuntimeError:
            continue  # skip if it would take too long for the test
        word = [a for _, a in direct]
        nm = orbit_formula(n0, word)
        assert nm == 1, f"orbit formula failed at n0={n0}"


def main() -> None:
    print("=" * 72)
    print("EXACT COLLATZ ORBIT FORMULA TEST")
    print("=" * 72)

    sample = [3, 7, 27, 97, 871]
    print("\n  n      orbit length    A_m       n_m via formula")
    print("  " + "-" * 60)
    for n0 in sample:
        direct = syracuse_orbit(n0)
        word = [a for _, a in direct]
        A_m = sum(word)
        nm = orbit_formula(n0, word)
        marker = "  <- closes to 1" if nm == 1 else "  <- WRONG"
        print(f"  {n0:<6} {len(word):<14}  {A_m:<8}  {nm}{marker}")

    print("\n  Running rigorous formula checks ...")
    for fn in [
        test_orbit_formula_matches_direct_for_n3,
        test_orbit_formula_matches_direct_for_n27,
        test_orbit_formula_matches_direct_for_n97,
        test_orbit_formula_matches_direct_for_n871,
        test_partial_orbit_formula_matches_direct,
        test_random_odd_integers,
    ]:
        try:
            fn()
            print(f"  PASS: {fn.__name__}")
        except AssertionError as e:
            print(f"  FAIL: {fn.__name__}: {e}")
            raise

    print("\n  EXACT ORBIT FORMULA VERIFIED.")


if __name__ == "__main__":
    main()
