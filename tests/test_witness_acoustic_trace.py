r"""Acoustic / spherical-harmonic construction of the witness trace.

This module replaces the earlier "witness trace axiom" with an explicit
positive-definite construction.  The witness sphere is modelled as a
closed acoustic cavity whose state space is L^2(S^2).  In the spherical
harmonic basis,

    psi = sum_{l,m} c_{lm} Y_l^m

the *witness trace* is the squared overlap with the fundamental
center-bearing mode Y_0^0:

    \widehat{Tr}_W(psi) := |<psi, Y_0^0>|^2 = |c_{00}|^2

This is positive by construction.  Its kernel is exactly the boundary
sector  V_boundary = span_{l>=1, m} Y_l^m,  which is the *trace
radical*.

This file tests the four properties that the construction must satisfy
to qualify as the missing witness norm:

    (1) POSITIVITY:        \widehat{Tr}_W(psi) >= 0  for all psi
    (2) CENTER:            \widehat{Tr}_W(Y_0^0) = 1
    (3) BOUNDARY-NULL:     \widehat{Tr}_W(Y_l^m) = 0 for l >= 1
    (4) PHASE-INVARIANT:   \widehat{Tr}_W(e^{i alpha} psi) = \widehat{Tr}_W(psi)
    (5) DIRECT-SUM:        V_W = C Y_0^0 (+) V_boundary, with V_boundary
                           = ker(\widehat{Tr}_W).

We use a finite truncation l <= L_max to make this numerically tractable;
the construction is exact in the truncated space and convergent in the
full L^2(S^2).
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


# ────────────────────────────────────────────────────────────────────────
# Spherical harmonic basis (real) up to L_max.
#
# We use *real* spherical harmonics so coefficients are real; the
# construction is identical with complex Y_l^m but easier to test with
# real coefficients.  The center mode Y_0^0 = 1 / (2 sqrt(pi)) is the
# constant function on the sphere.
# ────────────────────────────────────────────────────────────────────────


L_MAX = 6


def _basis_index(ell: int, m: int) -> int:
    """Flat index for (l, m) with -l <= m <= l, l = 0..L_MAX."""
    if not (0 <= ell <= L_MAX):
        raise ValueError(f"l out of range: {ell}")
    if not (-ell <= m <= ell):
        raise ValueError(f"m out of range for l={ell}: {m}")
    return ell * ell + (m + ell)


def basis_dim() -> int:
    return (L_MAX + 1) ** 2


def center_index() -> int:
    return _basis_index(0, 0)


def y_lm(ell: int, m: int) -> np.ndarray:
    """Coefficient vector representing Y_l^m in the truncated basis."""
    v = np.zeros(basis_dim())
    v[_basis_index(ell, m)] = 1.0
    return v


def witness_trace_acoustic(psi: np.ndarray) -> float:
    """Witness trace = |<psi, Y_0^0>|^2 = squared center-mode coefficient."""
    c00 = float(psi[center_index()])
    return c00 * c00


def witness_projection(psi: np.ndarray) -> np.ndarray:
    """Pi_W(psi) = (<psi, Y_0^0>) Y_0^0 -- the center component."""
    out = np.zeros_like(psi)
    out[center_index()] = psi[center_index()]
    return out


def is_in_radical(psi: np.ndarray, tol: float = 1e-12) -> bool:
    return abs(psi[center_index()]) < tol


# ────────────────────────────────────────────────────────────────────────
# Tests
# ────────────────────────────────────────────────────────────────────────


def test_basis_dim() -> None:
    expected = (L_MAX + 1) ** 2
    assert basis_dim() == expected


def test_center_mode_has_unit_trace() -> None:
    psi = y_lm(0, 0)
    assert abs(witness_trace_acoustic(psi) - 1.0) < 1e-12


def test_boundary_modes_have_zero_trace() -> None:
    """Every Y_l^m for l >= 1 is in the trace radical."""
    for ell in range(1, L_MAX + 1):
        for m in range(-ell, ell + 1):
            psi = y_lm(ell, m)
            tr = witness_trace_acoustic(psi)
            assert abs(tr) < 1e-12, f"Y_{ell}^{m} has nonzero trace {tr}"
            assert is_in_radical(psi)


def test_trace_is_nonnegative() -> None:
    r"""Positivity: \widehat{Tr}_W >= 0 for all states."""
    rng = np.random.default_rng(2027)
    for _ in range(200):
        psi = rng.normal(size=basis_dim())
        assert witness_trace_acoustic(psi) >= 0.0


def test_trace_phase_invariance_for_real_states() -> None:
    """Sign flip leaves trace invariant: |<-psi, Y_0^0>|^2 = |<psi, Y_0^0>|^2."""
    rng = np.random.default_rng(7)
    for _ in range(100):
        psi = rng.normal(size=basis_dim())
        assert abs(witness_trace_acoustic(psi) - witness_trace_acoustic(-psi)) < 1e-12


def test_decomposition_center_plus_boundary() -> None:
    """V_W = C Y_0^0 (+) V_boundary, exactly."""
    rng = np.random.default_rng(11)
    for _ in range(50):
        psi = rng.normal(size=basis_dim())
        center = witness_projection(psi)
        boundary = psi - center

        assert is_in_radical(boundary)
        assert abs(witness_trace_acoustic(boundary)) < 1e-12

        c00 = psi[center_index()]
        assert abs(witness_trace_acoustic(center) - c00 * c00) < 1e-12

        recon = center + boundary
        assert np.allclose(recon, psi, atol=1e-12)


def test_radical_is_closed_under_addition() -> None:
    """If x, y in V_boundary then x + y in V_boundary."""
    rng = np.random.default_rng(13)
    for _ in range(30):
        psi1 = rng.normal(size=basis_dim())
        psi2 = rng.normal(size=basis_dim())
        psi1[center_index()] = 0.0
        psi2[center_index()] = 0.0
        s = psi1 + psi2
        assert is_in_radical(s)
        assert abs(witness_trace_acoustic(s)) < 1e-12


def test_radical_complement_separates_states() -> None:
    """Two states with the same boundary content but different center
    coefficients have different witness traces."""
    rng = np.random.default_rng(17)
    boundary = rng.normal(size=basis_dim())
    boundary[center_index()] = 0.0

    psi_a = boundary.copy()
    psi_a[center_index()] = 0.5
    psi_b = boundary.copy()
    psi_b[center_index()] = 1.0

    assert abs(witness_trace_acoustic(psi_a) - 0.25) < 1e-12
    assert abs(witness_trace_acoustic(psi_b) - 1.0) < 1e-12
    assert witness_trace_acoustic(psi_b) > witness_trace_acoustic(psi_a)


def test_unit_tethered_state_has_positive_trace() -> None:
    r"""A 'unit-tethered' state psi_n = Y_0^0 + (boundary noise) has
    \widehat{Tr}_W(psi_n) > 0 -- this is the analogue of  1 | n  for a
    positive integer Collatz seed.
    """
    rng = np.random.default_rng(19)
    for _ in range(20):
        boundary = rng.normal(size=basis_dim())
        boundary[center_index()] = 0.0
        psi_n = y_lm(0, 0) + 0.1 * boundary
        tr = witness_trace_acoustic(psi_n)
        assert tr > 0.0
        assert abs(tr - 1.0) < 1e-12


# ────────────────────────────────────────────────────────────────────────
# Sanity: orthogonality of Y_l^m
# ────────────────────────────────────────────────────────────────────────


def test_basis_orthonormal() -> None:
    """In our coefficient representation, basis vectors are orthonormal
    by construction (we use the spherical harmonic basis as the
    coordinate basis itself)."""
    indices = []
    for ell in range(L_MAX + 1):
        for m in range(-ell, ell + 1):
            indices.append((ell, m))

    for i, (l1, m1) in enumerate(indices):
        for j, (l2, m2) in enumerate(indices):
            v1 = y_lm(l1, m1)
            v2 = y_lm(l2, m2)
            ip = float(v1 @ v2)
            expected = 1.0 if (l1, m1) == (l2, m2) else 0.0
            assert abs(ip - expected) < 1e-12


# ────────────────────────────────────────────────────────────────────────
# Main
# ────────────────────────────────────────────────────────────────────────


def main() -> None:
    print("=" * 72)
    print("ACOUSTIC WITNESS TRACE TESTS (constructive, not axiomatic)")
    print("=" * 72)
    print(f"  L_max = {L_MAX}, basis_dim = {basis_dim()}")
    print(f"  Y_0^0 trace = {witness_trace_acoustic(y_lm(0, 0)):+.6f}")
    print(f"  Y_1^0 trace = {witness_trace_acoustic(y_lm(1, 0)):+.6f}  (radical)")
    print(f"  Y_3^2 trace = {witness_trace_acoustic(y_lm(3, 2)):+.6f}  (radical)")

    rng = np.random.default_rng(0)
    psi = rng.normal(size=basis_dim())
    tr = witness_trace_acoustic(psi)
    print(f"  random state trace = {tr:+.6f}  (>= 0 by construction)")

    for fn in [
        test_basis_dim,
        test_center_mode_has_unit_trace,
        test_boundary_modes_have_zero_trace,
        test_trace_is_nonnegative,
        test_trace_phase_invariance_for_real_states,
        test_decomposition_center_plus_boundary,
        test_radical_is_closed_under_addition,
        test_radical_complement_separates_states,
        test_unit_tethered_state_has_positive_trace,
        test_basis_orthonormal,
    ]:
        try:
            fn()
            print(f"  PASS: {fn.__name__}")
        except AssertionError as e:
            print(f"  FAIL: {fn.__name__}: {e}")
            raise

    print("\n  ALL ACOUSTIC TRACE TESTS PASSED.")
    print("  The witness trace is positive-by-construction.")


if __name__ == "__main__":
    main()
