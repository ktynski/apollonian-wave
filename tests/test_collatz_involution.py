"""Witness involution tests for the Collatz Cl(1,1) extension.

The witness involution iota : Cl(1,1) -> Cl(1,1) implements the
"grace <-> nilpotent" mirror flip:

    iota(1)     =  1
    iota(e1)    =  e2
    iota(e2)    =  e1
    iota(e1 e2) = -e1 e2

extended R-linearly and respecting the algebra reversal rule.

Pass conditions:
  - INVOLUTIVE: iota^2 = id on basis and 50 random elements
  - SCALAR_FIXED: iota fixes the scalar (grade-0) part
  - VOLUME_NEGATED: iota(omega) = -omega
  - VECTOR_SWAPPED: iota(e1) = e2, iota(e2) = e1
  - LINEAR: iota(a*x + b*y) = a*iota(x) + b*iota(y)
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Reuse Cl(1,1) primitives
from tests.test_cl11_algebra import (  # noqa: E402
    cl11_basis,
    cl11_close,
    cl11_mul,
)


# ────────────────────────────────────────────────────────────────────────
# Witness involution
# ────────────────────────────────────────────────────────────────────────


def witness_iota(a: np.ndarray) -> np.ndarray:
    """Witness involution on Cl(1,1).

    Coordinate action on basis (1, e1, e2, omega):
      iota(1)     = 1
      iota(e1)    = e2
      iota(e2)    = e1
      iota(omega) = -omega
    """
    return np.array([a[0], a[2], a[1], -a[3]], dtype=np.float64)


# ────────────────────────────────────────────────────────────────────────
# Tests
# ────────────────────────────────────────────────────────────────────────


def test_iota_fixes_scalar() -> None:
    B = cl11_basis()
    assert cl11_close(witness_iota(B["1"]), B["1"])


def test_iota_swaps_e1_e2() -> None:
    B = cl11_basis()
    assert cl11_close(witness_iota(B["e1"]), B["e2"])
    assert cl11_close(witness_iota(B["e2"]), B["e1"])


def test_iota_negates_omega() -> None:
    B = cl11_basis()
    assert cl11_close(witness_iota(B["w"]), -B["w"])


def test_iota_is_involutive_on_basis() -> None:
    B = cl11_basis()
    for name, v in B.items():
        twice = witness_iota(witness_iota(v))
        assert cl11_close(twice, v), (
            f"iota^2({name}) != {name}: got {twice}"
        )


def test_iota_is_involutive_random() -> None:
    rng = np.random.default_rng(2025)
    for _ in range(50):
        a = rng.normal(size=4)
        assert cl11_close(witness_iota(witness_iota(a)), a)


def test_iota_is_linear() -> None:
    rng = np.random.default_rng(11)
    for _ in range(20):
        a = rng.normal(size=4)
        b = rng.normal(size=4)
        s, t = rng.normal(size=2)
        lhs = witness_iota(s * a + t * b)
        rhs = s * witness_iota(a) + t * witness_iota(b)
        assert cl11_close(lhs, rhs)


def test_iota_swaps_signature_directions() -> None:
    """e1^2 = +1 and e2^2 = -1; iota swaps them, so the signature 'flips'."""
    B = cl11_basis()
    e1_sq = cl11_mul(B["e1"], B["e1"])
    e2_sq = cl11_mul(B["e2"], B["e2"])
    iota_e1_sq = cl11_mul(witness_iota(B["e1"]), witness_iota(B["e1"]))
    iota_e2_sq = cl11_mul(witness_iota(B["e2"]), witness_iota(B["e2"]))
    assert cl11_close(iota_e1_sq, e2_sq), "iota(e1)^2 should equal e2^2 = -1"
    assert cl11_close(iota_e2_sq, e1_sq), "iota(e2)^2 should equal e1^2 = +1"


def main() -> None:
    print("=" * 72)
    print("WITNESS INVOLUTION TESTS (Collatz Cl(1,1))")
    print("=" * 72)

    B = cl11_basis()
    print("\n  Action of iota on basis:")
    for name in ["1", "e1", "e2", "w"]:
        print(f"    iota({name:>2}) = {witness_iota(B[name])}")

    for fn in [
        test_iota_fixes_scalar,
        test_iota_swaps_e1_e2,
        test_iota_negates_omega,
        test_iota_is_involutive_on_basis,
        test_iota_is_involutive_random,
        test_iota_is_linear,
        test_iota_swaps_signature_directions,
    ]:
        try:
            fn()
            print(f"  PASS: {fn.__name__}")
        except AssertionError as e:
            print(f"  FAIL: {fn.__name__}: {e}")
            raise

    print("\n  ALL WITNESS INVOLUTION TESTS PASSED.")


if __name__ == "__main__":
    main()
