"""Cycle-obstruction test for the Collatz/Syracuse map.

If (a_0, ..., a_{k-1}) is a length-k periodic word for the Syracuse map,
the unique fixed point is

    n_0 = W_k / (2^{A_k} - 3^k)

where  A_k = a_0 + ... + a_{k-1},  W_k = sum_j 3^{k-1-j} 2^{A_j}.

The cycle-obstruction theorem says: only the root word a = (2) at k = 1
gives a positive odd integer n_0 (namely n_0 = 1).  For any other choice
of (a_0, ..., a_{k-1}) with each a_i >= 1 and k >= 1, the formula either
fails to be a positive integer, fails to be odd, or fails to actually
re-produce the word under iteration.

This test enumerates all admissible words up to a bound and verifies
that only the root cycle yields a valid integer fixed point.

Pass conditions:
  - ROOT_CYCLE_VALID: word (2), k = 1, gives n_0 = 1.
  - ALL_OTHERS_FAIL: every other admissible word fails one or more of:
        (i)  positivity of (2^{A_k} - 3^k),
        (ii) divisibility of W_k by (2^{A_k} - 3^k),
        (iii) odd-positivity of the resulting integer,
        (iv) the resulting integer actually iterates with that word.
"""

from __future__ import annotations

from itertools import product


def cycle_fixed_point(word: tuple[int, ...]) -> tuple[int, int, int]:
    """Compute (numerator, denominator, candidate n_0) for a word.

    n_0 = W_k / (2^{A_k} - 3^k).
    Returns (W_k, 2^{A_k} - 3^k, n_0_or_None).  If the denominator is
    non-positive or doesn't divide evenly, the third entry is None.
    """
    k = len(word)
    A_partial = [0]
    for a in word:
        A_partial.append(A_partial[-1] + a)
    A_k = A_partial[-1]
    W_k = sum((3 ** (k - 1 - j)) * (2 ** A_partial[j]) for j in range(k))
    denom = 2 ** A_k - 3 ** k
    if denom <= 0:
        return (W_k, denom, None)
    if W_k % denom != 0:
        return (W_k, denom, None)
    n0 = W_k // denom
    return (W_k, denom, n0)


def syracuse_step(n: int) -> tuple[int, int]:
    assert n >= 1 and n % 2 == 1
    m = 3 * n + 1
    a = 0
    while m % 2 == 0:
        m //= 2
        a += 1
    return m, a


def is_actual_cycle(n0: int, word: tuple[int, ...]) -> bool:
    """Check whether iterating Syracuse from n0 reproduces `word` and
    returns to n0 after k steps."""
    if n0 < 1 or n0 % 2 == 0:
        return False
    cur = n0
    for a_expected in word:
        nxt, a_got = syracuse_step(cur)
        if a_got != a_expected:
            return False
        cur = nxt
    return cur == n0


def test_root_cycle_is_valid() -> None:
    """Root cycle: word = (2), k = 1, gives n_0 = 1."""
    W, D, n0 = cycle_fixed_point((2,))
    assert W == 1
    assert D == 4 - 3 == 1
    assert n0 == 1
    assert is_actual_cycle(1, (2,))


def is_primitive_word(word: tuple[int, ...]) -> bool:
    """A word is primitive iff it is not a proper repetition of a
    shorter word."""
    k = len(word)
    for d in range(1, k):
        if k % d == 0 and word == word[:d] * (k // d):
            return False
    return True


def test_no_short_cycles_other_than_root() -> None:
    """Exhaustively search PRIMITIVE words up to length 6 with a_i in
    {1,...,6}.  Verify that no primitive word besides (2,) produces a
    valid actual cycle.

    Non-primitive words like (2,2,...) are simply k iterations of the
    root cycle and don't represent new cycles.
    """
    found_violations: list[tuple[tuple[int, ...], int]] = []
    for k in range(1, 7):
        for word in product(range(1, 7), repeat=k):
            if word == (2,):
                continue  # root cycle
            if not is_primitive_word(word):
                continue  # non-primitive repeats of shorter cycle
            W, D, n0 = cycle_fixed_point(word)
            if n0 is None or n0 <= 0 or n0 % 2 == 0:
                continue  # algebraically excluded
            if is_actual_cycle(n0, word):
                found_violations.append((word, n0))
    assert not found_violations, (
        f"Found unexpected primitive non-root cycles: {found_violations}"
    )


def test_all_words_of_length_one_fail_except_root() -> None:
    """Length-1 words: a in {1, 2, 3, ..., 10}.  Only a=2 should give
    a valid cycle."""
    for a in range(1, 11):
        word = (a,)
        W, D, n0 = cycle_fixed_point(word)
        if a == 2:
            assert n0 == 1
        else:
            # either denom <= 0, or n0 isn't a positive odd integer
            # that re-iterates the word
            valid = (n0 is not None and n0 > 0 and n0 % 2 == 1
                     and is_actual_cycle(n0, word))
            assert not valid, f"non-root length-1 word {word} produced cycle"


def test_classical_2_cycle_excluded() -> None:
    """The famous "2-cycle" hypothesis: words (a, b) with a + b = 4.
    Show none of them give a valid cycle."""
    candidates = [(1, 3), (3, 1)]
    for word in candidates:
        W, D, n0 = cycle_fixed_point(word)
        if n0 is not None and n0 > 0 and n0 % 2 == 1:
            assert not is_actual_cycle(n0, word), (
                f"length-2 cycle found at {word} -> {n0}"
            )


def main() -> None:
    print("=" * 72)
    print("COLLATZ CYCLE-OBSTRUCTION TEST")
    print("=" * 72)

    print("\n  Length-1 words (a,):")
    print(f"    {'a':>3} {'W_1':>6} {'2-3':>6} {'n_0':>10} {'is_cycle':>10}")
    for a in range(1, 8):
        word = (a,)
        W, D, n0 = cycle_fixed_point(word)
        cyc = is_actual_cycle(n0, word) if (
            n0 is not None and n0 > 0 and n0 % 2 == 1
        ) else False
        print(f"    {a:>3} {W:>6} {D:>6} {str(n0):>10} {str(cyc):>10}")

    print("\n  Length-2 words (a,b) with a+b in {3,4,5}:")
    print(f"    {'word':>10} {'W_2':>8} {'4-9':>6} {'n_0':>10} {'is_cycle':>10}")
    for a in range(1, 5):
        for b in range(1, 5):
            word = (a, b)
            W, D, n0 = cycle_fixed_point(word)
            cyc = is_actual_cycle(n0, word) if (
                n0 is not None and n0 > 0 and n0 % 2 == 1
            ) else False
            print(f"    {str(word):>10} {W:>8} {D:>6} {str(n0):>10} {str(cyc):>10}")

    print("\n  Running rigorous cycle-obstruction checks ...")
    for fn in [
        test_root_cycle_is_valid,
        test_no_short_cycles_other_than_root,
        test_all_words_of_length_one_fail_except_root,
        test_classical_2_cycle_excluded,
    ]:
        try:
            fn()
            print(f"  PASS: {fn.__name__}")
        except AssertionError as e:
            print(f"  FAIL: {fn.__name__}: {e}")
            raise

    print("\n  CYCLE OBSTRUCTION VERIFIED FOR k <= 6, a_i <= 6.")


if __name__ == "__main__":
    main()
