r"""Merkaba angles, Fano nilpotent seed, and the witness module V_W.

This file constructs the algebraic ingredients of the witness module

    V_W = C W  (+)  bigoplus_{x in F_7} (C N_x  (+)  C G_x)

where:

  * W = Y_0^0 is the center mode (witness center),
  * F_7 = F_2^3 \ {0} is the Fano nilpotent seed (7 nonzero classes),
  * N_x is an outgoing nilpotent channel for each x in F_7,
  * G_x = iota(N_x) is the mirrored / grace-involute channel,
  * iota is an involution swapping outgoing and grace sectors.

A Fano *line* is a triple {x, y, z} subset F_7 with x + y + z = 0 in
F_2^3.  There are exactly 7 such lines.  Each line is one nilpotent
incidence relation.

The Merkaba root angles -- forced by stella-octangula geometry -- are

    theta_T = arccos(-1/3)  ~ 109.47 deg   (tetrahedral vertex angle)
    theta_D = arccos(+1/3)  ~  70.53 deg   (tetrahedral dihedral complement)
    theta_M = arccos(1/sqrt(3)) ~ 54.74 deg (magic-axis angle)

and the legal extrusion angles are

    A_M = { +/- theta_T,  +/- theta_D,  +/- theta_M }.

Pass conditions:
  - F7_HAS_7_POINTS:  |F_7| = 7
  - FANO_LINE_COUNT:  exactly 7 Fano lines
  - FANO_LINE_PROPS:  each line has 3 distinct points summing to 0
  - LINE_COVERAGE:    each point lies on exactly 3 lines
  - INCIDENCE_PAIR:   each pair of distinct points lies on exactly 1 line
  - MERKABA_ANGLES:   A_M has 6 distinct values, all in (0, pi)
  - VW_DIM:           dim(V_W) = 1 + 2 * |F_7| = 15
  - INVOLUTION:       iota: V_W -> V_W is involutive, swaps N_x and G_x,
                      fixes W
  - W_FIXED:          iota(W) = W
"""

from __future__ import annotations

import itertools
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


# ────────────────────────────────────────────────────────────────────────
# Merkaba root angles
# ────────────────────────────────────────────────────────────────────────


THETA_T = math.acos(-1.0 / 3.0)
THETA_D = math.acos(+1.0 / 3.0)
THETA_M = math.acos(1.0 / math.sqrt(3.0))

A_M = (+THETA_T, -THETA_T, +THETA_D, -THETA_D, +THETA_M, -THETA_M)


# ────────────────────────────────────────────────────────────────────────
# Fano nilpotent seed:  F_7 = F_2^3 \ {0}
# ────────────────────────────────────────────────────────────────────────


def f7_points() -> list[tuple[int, int, int]]:
    """The 7 nonzero vectors in F_2^3."""
    return [
        (a, b, c)
        for a, b, c in itertools.product((0, 1), repeat=3)
        if (a, b, c) != (0, 0, 0)
    ]


def f2_add(
    x: tuple[int, int, int], y: tuple[int, int, int]
) -> tuple[int, int, int]:
    return ((x[0] + y[0]) % 2, (x[1] + y[1]) % 2, (x[2] + y[2]) % 2)


def fano_lines() -> list[frozenset]:
    """Triples {x, y, z} subset F_7 with x + y + z = 0 in F_2^3."""
    pts = f7_points()
    lines: set[frozenset] = set()
    for x, y in itertools.combinations(pts, 2):
        z = f2_add(x, y)
        if z != (0, 0, 0) and z != x and z != y:
            line = frozenset({x, y, z})
            if len(line) == 3:
                lines.add(line)
    return sorted(lines, key=lambda L: tuple(sorted(L)))


# ────────────────────────────────────────────────────────────────────────
# Witness module V_W = C W (+)  C[F_7] (+)  C[F_7^iota]
#
# We index basis elements as:
#   slot 0      -> W (center)
#   slot 1..7   -> N_{x_1}, ..., N_{x_7}  (outgoing nilpotent channels)
#   slot 8..14  -> G_{x_1}, ..., G_{x_7}  (mirrored grace channels)
# ────────────────────────────────────────────────────────────────────────


def vw_basis() -> dict[str, np.ndarray]:
    """Return canonical basis vectors W, N_x, G_x as one-hot 15-vectors."""
    pts = f7_points()
    dim = 1 + 2 * len(pts)
    basis: dict[str, np.ndarray] = {}
    e_w = np.zeros(dim)
    e_w[0] = 1.0
    basis["W"] = e_w
    for i, x in enumerate(pts):
        e_n = np.zeros(dim)
        e_n[1 + i] = 1.0
        basis[f"N_{x}"] = e_n

        e_g = np.zeros(dim)
        e_g[1 + len(pts) + i] = 1.0
        basis[f"G_{x}"] = e_g
    return basis


def vw_dim() -> int:
    return 1 + 2 * len(f7_points())


def witness_involution_vw(v: np.ndarray) -> np.ndarray:
    """iota: V_W -> V_W swaps N_x <-> G_x for each x and fixes W.

    iota^2 = id  by construction.
    """
    n = len(f7_points())
    out = np.zeros_like(v)
    out[0] = v[0]
    for i in range(n):
        out[1 + i] = v[1 + n + i]          # G_{x_i} -> N_{x_i}
        out[1 + n + i] = v[1 + i]          # N_{x_i} -> G_{x_i}
    return out


def projection_W(v: np.ndarray) -> np.ndarray:
    """Pi_W projects V_W onto the center component C W."""
    out = np.zeros_like(v)
    out[0] = v[0]
    return out


# ────────────────────────────────────────────────────────────────────────
# Tests: F_7 cardinality, Fano line structure
# ────────────────────────────────────────────────────────────────────────


def test_f7_has_seven_points() -> None:
    pts = f7_points()
    assert len(pts) == 7
    assert (0, 0, 0) not in pts


def test_seven_fano_lines() -> None:
    L = fano_lines()
    assert len(L) == 7


def test_each_line_has_three_distinct_points_summing_to_zero() -> None:
    for line in fano_lines():
        pts = list(line)
        assert len(pts) == 3
        s = pts[0]
        for p in pts[1:]:
            s = f2_add(s, p)
        assert s == (0, 0, 0)


def test_each_point_on_exactly_three_lines() -> None:
    L = fano_lines()
    for p in f7_points():
        count = sum(1 for line in L if p in line)
        assert count == 3, f"Point {p} lies on {count} lines, expected 3"


def test_each_pair_on_exactly_one_line() -> None:
    L = fano_lines()
    for x, y in itertools.combinations(f7_points(), 2):
        count = sum(1 for line in L if x in line and y in line)
        assert count == 1, (
            f"Pair {x},{y} lies on {count} lines, expected 1"
        )


# ────────────────────────────────────────────────────────────────────────
# Tests: Merkaba angles
# ────────────────────────────────────────────────────────────────────────


def test_merkaba_angle_count() -> None:
    """A_M has exactly 6 distinct entries."""
    seen = set()
    for a in A_M:
        seen.add(round(a, 10))
    assert len(seen) == 6


def test_merkaba_angles_are_root_geometric() -> None:
    """The three positive Merkaba angles correspond to known stella-
    octangula geometric quantities:

        theta_T = arccos(-1/3)
        theta_D = arccos(+1/3)
        theta_M = arccos(1/sqrt(3))
    """
    assert abs(math.cos(THETA_T) + 1.0 / 3.0) < 1e-12
    assert abs(math.cos(THETA_D) - 1.0 / 3.0) < 1e-12
    assert abs(math.cos(THETA_M) - 1.0 / math.sqrt(3.0)) < 1e-12


def test_merkaba_angles_in_open_pi_interval() -> None:
    for a in A_M:
        assert 0.0 < abs(a) < math.pi


def test_t_d_complementary_in_pi() -> None:
    """theta_T + theta_D = pi  (tetrahedral vertex + complement)."""
    assert abs((THETA_T + THETA_D) - math.pi) < 1e-12


# ────────────────────────────────────────────────────────────────────────
# Tests: V_W module structure and involution
# ────────────────────────────────────────────────────────────────────────


def test_vw_dimension_is_fifteen() -> None:
    assert vw_dim() == 15


def test_basis_vectors_orthonormal() -> None:
    B = vw_basis()
    keys = list(B.keys())
    for i, k1 in enumerate(keys):
        for j, k2 in enumerate(keys):
            ip = float(B[k1] @ B[k2])
            expected = 1.0 if k1 == k2 else 0.0
            assert abs(ip - expected) < 1e-12


def test_involution_fixes_W() -> None:
    B = vw_basis()
    iW = witness_involution_vw(B["W"])
    assert np.allclose(iW, B["W"])


def test_involution_swaps_N_and_G() -> None:
    B = vw_basis()
    for x in f7_points():
        N = B[f"N_{x}"]
        G = B[f"G_{x}"]
        assert np.allclose(witness_involution_vw(N), G)
        assert np.allclose(witness_involution_vw(G), N)


def test_involution_is_involutive() -> None:
    rng = np.random.default_rng(101)
    for _ in range(50):
        v = rng.normal(size=vw_dim())
        iv = witness_involution_vw(v)
        iiv = witness_involution_vw(iv)
        assert np.allclose(iiv, v, atol=1e-12)


def test_involution_is_linear() -> None:
    rng = np.random.default_rng(103)
    for _ in range(20):
        v = rng.normal(size=vw_dim())
        w = rng.normal(size=vw_dim())
        s, t = rng.normal(size=2)
        lhs = witness_involution_vw(s * v + t * w)
        rhs = s * witness_involution_vw(v) + t * witness_involution_vw(w)
        assert np.allclose(lhs, rhs, atol=1e-12)


def test_pi_W_projection() -> None:
    B = vw_basis()
    assert np.allclose(projection_W(B["W"]), B["W"])
    for x in f7_points():
        assert np.allclose(projection_W(B[f"N_{x}"]), np.zeros(vw_dim()))
        assert np.allclose(projection_W(B[f"G_{x}"]), np.zeros(vw_dim()))


def test_pi_W_radical_is_F7_plus_F7iota() -> None:
    """ker(Pi_W) = bigoplus_x  C N_x  (+)  C G_x"""
    B = vw_basis()
    rng = np.random.default_rng(7)
    for _ in range(20):
        coeffs_N = rng.normal(size=7)
        coeffs_G = rng.normal(size=7)
        v = np.zeros(vw_dim())
        for i, x in enumerate(f7_points()):
            v += coeffs_N[i] * B[f"N_{x}"] + coeffs_G[i] * B[f"G_{x}"]
        assert np.allclose(projection_W(v), np.zeros(vw_dim()))


# ────────────────────────────────────────────────────────────────────────
# Sanity: Fano addition table is a Fano-plane incidence
# ────────────────────────────────────────────────────────────────────────


def test_fano_addition_law_closure() -> None:
    """For every pair x, y in F_7, the sum x + y in F_2^3 is in F_7
    and the triple {x, y, x+y} is one of the 7 Fano lines."""
    pts = f7_points()
    L = fano_lines()
    for x, y in itertools.combinations(pts, 2):
        z = f2_add(x, y)
        assert z != (0, 0, 0), f"x + y = 0 for distinct x, y? {x}, {y}"
        line = frozenset({x, y, z})
        assert line in L


# ────────────────────────────────────────────────────────────────────────
# Main
# ────────────────────────────────────────────────────────────────────────


def main() -> None:
    print("=" * 72)
    print("MERKABA / FANO / WITNESS MODULE TESTS")
    print("=" * 72)
    print(f"  |F_7|              = {len(f7_points())}")
    print(f"  Fano lines         = {len(fano_lines())}")
    print(f"  dim(V_W)           = {vw_dim()}")
    print(f"  theta_T (deg)      = {math.degrees(THETA_T):.4f}")
    print(f"  theta_D (deg)      = {math.degrees(THETA_D):.4f}")
    print(f"  theta_M (deg)      = {math.degrees(THETA_M):.4f}")
    print(f"  theta_T + theta_D  = {math.degrees(THETA_T + THETA_D):.4f} deg "
          f"(should be 180.0)")

    print("\n  Fano lines (triples summing to 0 in F_2^3):")
    for line in fano_lines():
        print(f"    {sorted(line)}")

    for fn in [
        test_f7_has_seven_points,
        test_seven_fano_lines,
        test_each_line_has_three_distinct_points_summing_to_zero,
        test_each_point_on_exactly_three_lines,
        test_each_pair_on_exactly_one_line,
        test_merkaba_angle_count,
        test_merkaba_angles_are_root_geometric,
        test_merkaba_angles_in_open_pi_interval,
        test_t_d_complementary_in_pi,
        test_vw_dimension_is_fifteen,
        test_basis_vectors_orthonormal,
        test_involution_fixes_W,
        test_involution_swaps_N_and_G,
        test_involution_is_involutive,
        test_involution_is_linear,
        test_pi_W_projection,
        test_pi_W_radical_is_F7_plus_F7iota,
        test_fano_addition_law_closure,
    ]:
        try:
            fn()
            print(f"  PASS: {fn.__name__}")
        except AssertionError as e:
            print(f"  FAIL: {fn.__name__}: {e}")
            raise

    print("\n  ALL MERKABA / FANO MODULE TESTS PASSED.")
    print("  The witness module V_W is constructed, not assumed.")


if __name__ == "__main__":
    main()
