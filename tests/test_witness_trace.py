"""Witness trace tests for the Collatz Cl(1,1) extension.

The witness trace is defined as the involution-averaged scalar
projection:

    Tr_W(x) := (1/2) <x + iota(x)>_0

For x = a*1 + b*e1 + c*e2 + d*omega:
  iota(x) = a*1 + c*e1 + b*e2 + (-d)*omega
  x + iota(x) = 2a*1 + (b+c)*e1 + (b+c)*e2 + 0*omega
  <x + iota(x)>_0 = 2a
  Tr_W(x) = a

So Tr_W is exactly the grade-0 (scalar) projection, but the
involution-averaging gives it a meaningful interpretation: it is the
component invariant under iota.

Pass conditions:
  - LINEAR: Tr_W(a*x + b*y) = a*Tr_W(x) + b*Tr_W(y)
  - UNIT: Tr_W(1) = 1
  - VECTOR_NULL: Tr_W(e1) = Tr_W(e2) = 0
  - VOLUME_NULL: Tr_W(omega) = 0
  - IOTA_INVARIANT: Tr_W(iota(x)) = Tr_W(x) for all x
  - NILPOTENT_KERNEL: V_nil := ker(Tr_W) contains e1, e2, omega and
    is closed under iota
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tests.test_cl11_algebra import cl11_basis, cl11_close  # noqa: E402
from tests.test_collatz_involution import witness_iota  # noqa: E402


# ────────────────────────────────────────────────────────────────────────
# Witness trace
# ────────────────────────────────────────────────────────────────────────


def witness_trace(x: np.ndarray) -> float:
    """Tr_W(x) = (1/2) <x + iota(x)>_0 = scalar coefficient of x."""
    return 0.5 * float((x + witness_iota(x))[0])


# ────────────────────────────────────────────────────────────────────────
# Tests
# ────────────────────────────────────────────────────────────────────────


def test_trace_unit_is_one() -> None:
    B = cl11_basis()
    assert abs(witness_trace(B["1"]) - 1.0) < 1e-12


def test_trace_e1_zero() -> None:
    B = cl11_basis()
    assert abs(witness_trace(B["e1"])) < 1e-12


def test_trace_e2_zero() -> None:
    B = cl11_basis()
    assert abs(witness_trace(B["e2"])) < 1e-12


def test_trace_omega_zero() -> None:
    B = cl11_basis()
    assert abs(witness_trace(B["w"])) < 1e-12


def test_trace_is_linear() -> None:
    rng = np.random.default_rng(99)
    for _ in range(50):
        a = rng.normal(size=4)
        b = rng.normal(size=4)
        s, t = rng.normal(size=2)
        lhs = witness_trace(s * a + t * b)
        rhs = s * witness_trace(a) + t * witness_trace(b)
        assert abs(lhs - rhs) < 1e-12


def test_trace_is_iota_invariant() -> None:
    rng = np.random.default_rng(13)
    for _ in range(50):
        a = rng.normal(size=4)
        assert abs(witness_trace(a) - witness_trace(witness_iota(a))) < 1e-12


def test_nilpotent_ideal_contains_basis_directions() -> None:
    B = cl11_basis()
    for name in ["e1", "e2", "w"]:
        assert abs(witness_trace(B[name])) < 1e-12, (
            f"{name} should be in V_nil but Tr_W({name}) = "
            f"{witness_trace(B[name])}"
        )


def test_nilpotent_ideal_closed_under_iota() -> None:
    """If Tr_W(x) = 0 then Tr_W(iota(x)) = 0 too."""
    rng = np.random.default_rng(8)
    for _ in range(50):
        a = rng.normal(size=4)
        a[0] = 0.0  # force into V_nil
        assert abs(witness_trace(a)) < 1e-12
        assert abs(witness_trace(witness_iota(a))) < 1e-12


def test_decomposition_R_plus_Vnil() -> None:
    """Every x in Cl(1,1) decomposes as x = Tr_W(x)*1 + n, n in V_nil."""
    rng = np.random.default_rng(31)
    B = cl11_basis()
    for _ in range(20):
        x = rng.normal(size=4)
        scalar = witness_trace(x)
        n = x - scalar * B["1"]
        assert abs(witness_trace(n)) < 1e-12, (
            f"Decomposition failed: residual has nonzero trace "
            f"{witness_trace(n)}"
        )
        assert cl11_close(scalar * B["1"] + n, x)


def main() -> None:
    print("=" * 72)
    print("WITNESS TRACE TESTS (Collatz Cl(1,1))")
    print("=" * 72)

    B = cl11_basis()
    print("\n  Witness trace of basis elements:")
    for name in ["1", "e1", "e2", "w"]:
        print(f"    Tr_W({name:>2}) = {witness_trace(B[name]):+.6f}")

    print("\n  Random decomposition x = Tr_W(x)*1 + nilpotent ...")
    rng = np.random.default_rng(0)
    x = rng.normal(size=4)
    s = witness_trace(x)
    n = x - s * B["1"]
    print(f"    x         = {x}")
    print(f"    Tr_W(x)*1 = {s * B['1']}")
    print(f"    nilpotent = {n}  (Tr_W = {witness_trace(n):+.2e})")

    for fn in [
        test_trace_unit_is_one,
        test_trace_e1_zero,
        test_trace_e2_zero,
        test_trace_omega_zero,
        test_trace_is_linear,
        test_trace_is_iota_invariant,
        test_nilpotent_ideal_contains_basis_directions,
        test_nilpotent_ideal_closed_under_iota,
        test_decomposition_R_plus_Vnil,
    ]:
        try:
            fn()
            print(f"  PASS: {fn.__name__}")
        except AssertionError as e:
            print(f"  FAIL: {fn.__name__}: {e}")
            raise

    print("\n  ALL WITNESS TRACE TESTS PASSED.")


if __name__ == "__main__":
    main()
