"""Cl(3,1) Clifford algebra — full geometric product in PyTorch.

Basis vectors: e1, e2, e3 (spacelike, square to +1), e4 (timelike, squares to -1).
Dimension: 2^4 = 16 basis blades, indexed by bit masks:
    bit 0 → e1,  bit 1 → e2,  bit 2 → e3,  bit 3 → e4

Blade indices:
    Grade 0:  0
    Grade 1:  1(e1), 2(e2), 4(e3), 8(e4)
    Grade 2:  3(e12), 5(e13), 9(e14), 6(e23), 10(e24), 12(e34)
    Grade 3:  7(e123), 11(e124), 13(e134), 14(e234)
    Grade 4:  15(e1234)

Geometric product of blades eI * eJ:
    result blade K = I XOR J
    sign = (-1)^(swap_count) * product_of_metric_for_common_bits
"""

from __future__ import annotations

import numpy as np
import torch

# ──────────────────────────────────────────────────────────────────────
# Sign table computation
# ──────────────────────────────────────────────────────────────────────

N_DIMS = 4
N_BLADES = 1 << N_DIMS  # 16
METRIC = (1, 1, 1, -1)  # e1²=+1, e2²=+1, e3²=+1, e4²=-1


def _popcount(n: int) -> int:
    return bin(n).count("1")


def _blade_name(idx: int) -> str:
    if idx == 0:
        return "1"
    parts = [str(k + 1) for k in range(N_DIMS) if idx & (1 << k)]
    return "e" + "".join(parts)


def _blade_product_sign(I: int, J: int) -> int:
    """Sign of eI * eJ in Cl(3,1).

    Two contributions:
    1. Reordering: for each bit j set in J, count bits in I at positions > j.
       Total swap count gives (-1)^count.
    2. Metric: each common bit k contributes METRIC[k].
    """
    swap_count = 0
    for j in range(N_DIMS):
        if not (J & (1 << j)):
            continue
        for i in range(j + 1, N_DIMS):
            if I & (1 << i):
                swap_count += 1

    sign = 1 if swap_count % 2 == 0 else -1

    common = I & J
    for k in range(N_DIMS):
        if common & (1 << k):
            sign *= METRIC[k]

    return sign


def build_sign_table() -> list[list[int]]:
    """16×16 table where sign_table[I][J] is the sign of eI * eJ."""
    return [[_blade_product_sign(I, J) for J in range(N_BLADES)]
            for I in range(N_BLADES)]


SIGN_TABLE: list[list[int]] = build_sign_table()

# For reference: blade_norm[I] = eI * eI = sign_table[I][I] * e_{I^I} = sign_table[I][I] * scalar
BLADE_NORM: list[int] = [SIGN_TABLE[I][I] for I in range(N_BLADES)]


# ──────────────────────────────────────────────────────────────────────
# PyTorch implementation
# ──────────────────────────────────────────────────────────────────────


def build_cayley_tensor(dtype: torch.dtype = torch.float64) -> torch.Tensor:
    """Build the (16, 16, 16) Cayley tensor for Cl(3,1).

    cayley[K, I, J] = sign(I,J)  if I XOR J == K, else 0.

    Usage:  result = einsum('kij,...i,...j->...k', cayley, a, b)
    """
    cayley = torch.zeros(N_BLADES, N_BLADES, N_BLADES, dtype=dtype)
    for I in range(N_BLADES):
        for J in range(N_BLADES):
            K = I ^ J
            cayley[K, I, J] = SIGN_TABLE[I][J]
    return cayley


_CAYLEY: dict[torch.dtype, torch.Tensor] = {}


def _get_cayley(dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    key = (dtype, device)
    if key not in _CAYLEY:
        _CAYLEY[key] = build_cayley_tensor(dtype).to(device)
    return _CAYLEY[key]


def clifford_product(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Geometric product of two Cl(3,1) multivectors.

    Args:
        a: (..., 16) multivector components
        b: (..., 16) multivector components

    Returns:
        (..., 16) the geometric product a * b
    """
    cayley = _get_cayley(a.dtype, a.device)
    return torch.einsum("kij,...i,...j->...k", cayley, a, b)


def clifford_scalar_product(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Scalar part of a * b, i.e. <a b>_0.

    When K=0: I XOR J = 0 ⟹ J = I.
    So <a b>_0 = Σ_I sign[I,I] * a[I] * b[I].
    """
    norms = torch.tensor(BLADE_NORM, dtype=a.dtype, device=a.device)
    return (norms * a * b).sum(dim=-1)


def grade_project(x: torch.Tensor, grade: int) -> torch.Tensor:
    """Project multivector onto a single grade."""
    mask = torch.tensor(
        [1.0 if _popcount(i) == grade else 0.0 for i in range(N_BLADES)],
        dtype=x.dtype, device=x.device,
    )
    return x * mask


def clifford_reverse(x: torch.Tensor) -> torch.Tensor:
    """Clifford reversal: reverse the order of basis vectors in each blade.

    rev(e_{i1...ik}) = (-1)^{k(k-1)/2} e_{i1...ik}.
    """
    signs = []
    for i in range(N_BLADES):
        k = _popcount(i)
        signs.append((-1) ** (k * (k - 1) // 2))
    rev_signs = torch.tensor(signs, dtype=x.dtype, device=x.device)
    return x * rev_signs


def make_blade(index: int, dtype: torch.dtype = torch.float64) -> torch.Tensor:
    """Create a unit blade multivector."""
    v = torch.zeros(N_BLADES, dtype=dtype)
    v[index] = 1.0
    return v


# ──────────────────────────────────────────────────────────────────────
# Verification
# ──────────────────────────────────────────────────────────────────────


def test_signature():
    """e1²=+1, e2²=+1, e3²=+1, e4²=-1."""
    e1, e2, e3, e4 = [make_blade(1 << k) for k in range(4)]
    one = make_blade(0)

    assert torch.allclose(clifford_product(e1, e1), one), "e1²≠+1"
    assert torch.allclose(clifford_product(e2, e2), one), "e2²≠+1"
    assert torch.allclose(clifford_product(e3, e3), one), "e3²≠+1"
    assert torch.allclose(clifford_product(e4, e4), -one), "e4²≠-1"


def test_anticommutation():
    """ei*ej = -ej*ei for i≠j."""
    blades = [make_blade(1 << k) for k in range(4)]
    for i in range(4):
        for j in range(i + 1, 4):
            ab = clifford_product(blades[i], blades[j])
            ba = clifford_product(blades[j], blades[i])
            assert torch.allclose(ab + ba, torch.zeros(N_BLADES, dtype=torch.float64)), (
                f"e{i+1}*e{j+1} + e{j+1}*e{i+1} ≠ 0"
            )


def test_e12_product():
    """e1*e2 = e12, e2*e1 = -e12."""
    e1, e2, e12 = make_blade(1), make_blade(2), make_blade(3)
    assert torch.allclose(clifford_product(e1, e2), e12), "e1*e2 ≠ e12"
    assert torch.allclose(clifford_product(e2, e1), -e12), "e2*e1 ≠ -e12"


def test_e12_squared():
    """e12² = e1*e2*e1*e2 = -1."""
    e12 = make_blade(3)
    one = make_blade(0)
    e12_sq = clifford_product(e12, e12)
    assert torch.allclose(e12_sq, -one), f"e12² ≠ -1, got {e12_sq}"


def test_pseudoscalar_squared():
    """e1234² = -1 in Cl(3,1)."""
    ps = make_blade(15)
    one = make_blade(0)
    ps_sq = clifford_product(ps, ps)
    assert torch.allclose(ps_sq, -one), f"e1234² ≠ -1, got {ps_sq}"


def test_associativity():
    """(a*b)*c == a*(b*c) for random multivectors."""
    rng = torch.Generator().manual_seed(1729)
    failures = 0
    for _ in range(200):
        a = torch.randn(N_BLADES, dtype=torch.float64, generator=rng)
        b = torch.randn(N_BLADES, dtype=torch.float64, generator=rng)
        c = torch.randn(N_BLADES, dtype=torch.float64, generator=rng)
        ab_c = clifford_product(clifford_product(a, b), c)
        a_bc = clifford_product(a, clifford_product(b, c))
        if not torch.allclose(ab_c, a_bc, atol=1e-10):
            failures += 1
    assert failures == 0, f"Associativity violated in {failures}/200 trials"


def test_reverse_involutive():
    """rev(rev(x)) == x."""
    rng = torch.Generator().manual_seed(42)
    for _ in range(50):
        x = torch.randn(N_BLADES, dtype=torch.float64, generator=rng)
        assert torch.allclose(clifford_reverse(clifford_reverse(x)), x)


def test_scalar_product_matches_full():
    """<a*b>_0 via scalar_product matches grade-0 of full product."""
    rng = torch.Generator().manual_seed(137)
    for _ in range(50):
        a = torch.randn(N_BLADES, dtype=torch.float64, generator=rng)
        b = torch.randn(N_BLADES, dtype=torch.float64, generator=rng)
        full = clifford_product(a, b)
        sp = clifford_scalar_product(a, b)
        assert torch.allclose(full[0], sp, atol=1e-12), (
            f"Scalar product mismatch: {full[0]} vs {sp}"
        )


def test_batched_product():
    """Geometric product works with batch dimensions."""
    rng = torch.Generator().manual_seed(99)
    a = torch.randn(8, 4, N_BLADES, dtype=torch.float64, generator=rng)
    b = torch.randn(8, 4, N_BLADES, dtype=torch.float64, generator=rng)
    result = clifford_product(a, b)
    assert result.shape == (8, 4, N_BLADES)

    for i in range(8):
        for j in range(4):
            single = clifford_product(a[i, j], b[i, j])
            assert torch.allclose(result[i, j], single, atol=1e-12)


def print_sign_table():
    """Print the full 16×16 sign table."""
    header = " " * 10 + "".join(f"{_blade_name(j):>8}" for j in range(N_BLADES))
    print(header)
    print(" " * 10 + "-" * (8 * N_BLADES))
    for I in range(N_BLADES):
        row_signs = "".join(f"{SIGN_TABLE[I][J]:>+3}·{_blade_name(I ^ J):<4}"
                            for J in range(N_BLADES))
        print(f"{_blade_name(I):>8} |{row_signs}")


def print_blade_norms():
    """Print eI² for each blade."""
    for I in range(N_BLADES):
        print(f"  {_blade_name(I):>6}² = {BLADE_NORM[I]:+d}")


def main():
    print("=" * 72)
    print("Cl(3,1) CLIFFORD ALGEBRA — SIGN TABLE AND PYTORCH IMPLEMENTATION")
    print("=" * 72)

    print("\nMetric signature: e1²=+1, e2²=+1, e3²=+1, e4²=-1")

    print("\n── Blade norms (eI²) ──")
    print_blade_norms()

    print("\n── Full 16×16 multiplication table ──")
    print("  (each entry is sign · result_blade, where result index = I XOR J)\n")
    print_sign_table()

    print("\n── Raw sign table as Python list ──")
    print("SIGN_TABLE = [")
    for I in range(N_BLADES):
        print(f"    {SIGN_TABLE[I]},  # {_blade_name(I)}")
    print("]")

    print("\n── Verification tests ──")
    tests = [
        test_signature,
        test_anticommutation,
        test_e12_product,
        test_e12_squared,
        test_pseudoscalar_squared,
        test_associativity,
        test_reverse_involutive,
        test_scalar_product_matches_full,
        test_batched_product,
    ]
    for fn in tests:
        try:
            fn()
            print(f"  PASS: {fn.__name__}")
        except AssertionError as e:
            print(f"  FAIL: {fn.__name__}: {e}")
            raise

    print("\n  ALL Cl(3,1) TESTS PASSED.")


if __name__ == "__main__":
    main()
