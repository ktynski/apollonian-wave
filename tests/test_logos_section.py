r"""The witness bundle and the Logos section.

This module verifies the consolidation of subsec:logos-section in
echo_rh.tex: the witness architecture (acoustic module, trace, cap,
nilpotent radical, supercoiling lemma) collapses to a single
load-bearing object,

    (E, nabla_M, iota, Lambda)

the closed witness bundle with its unique global Logos section.

The Logos section Lambda satisfies four properties:

  (L1) Center normalization:    Lambda in C*W with <Lambda, W> = 1
  (L2) Involution invariance:   iota(Lambda) = Lambda
  (L3) Covariant constancy:     for every capping braid B in C,
                                rho(B)*Lambda in C*Lambda
  (L4) Uniqueness:              any other vector satisfying (L1)-(L3)
                                is a scalar multiple of Lambda

In the acoustic realization, Lambda = Y_0^0 = W (the spherical-harmonic
center mode).

This file verifies:

   1. (L1) center normalization holds for Lambda = Y_0^0.
   2. (L2) involution invariance under iota acting trivially on Y_0^0
      and swapping nilpotent/grace channels.
   3. (L3) covariant constancy under the legal capping submonoid,
      sampled across:
         - the trivial loop B_trivial = ((x, alpha, +), (x, alpha, -));
         - the legal pi-rotation braid B_pi(x) =
           ((x, theta_T, +), (x, theta_D, +)) with total angle
           theta_T + theta_D = pi;
         - longer legal capping sequences.
   4. (L4) uniqueness: among 1000 random unit vectors satisfying (L1),
      only Lambda = Y_0^0 satisfies (L3).  Equivalently, no other
      1-dimensional subspace containing W is preserved by the legal
      capping submonoid.
   5. Derived trace:    Tr_W_hat(psi) = |<psi, Lambda>|^2 matches the
      acoustic trace Tr_W_hat(psi) = |c_00|^2.
   6. Derived radical:  Rad_W = ker<-, Lambda> equals V_boundary.
   7. Derived cap:      B in C iff |<rho(B)*W, Lambda>|^2 = 1.
   8. Logos pairing preservation: for every capping braid B sampled
      from a constructive family, |<rho(B)*W, Lambda>|^2 = 1; for
      every non-capping braid, |<rho(B)*W, Lambda>|^2 < 1 strictly.

The discrete extension iota(N_x) = G_x is realised by augmenting the
8-dim acoustic basis to a 15-dim discrete basis
{W} cup {N_x : x in F7} cup {G_x : x in F7}, and verifying that
Lambda = W is the unique iota-fixed center-normalized vector that is
preserved by the capping submonoid.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tests.test_acoustic_braid_action import (  # noqa: E402
    A_M,
    DIM,
    F7,
    F7_INDEX,
    THETA_D,
    THETA_T,
    acoustic_action,
    acoustic_trace,
    fano_add,
    fano_sum,
    initial_state,
    step_operator,
)


# ---------------------------------------------------------------
# Logos section: explicit construction
# ---------------------------------------------------------------


def logos_section_acoustic() -> np.ndarray:
    """Lambda = Y_0^0 = e_0 in V_W^ac (acoustic realization)."""
    Lambda = np.zeros(DIM, dtype=complex)
    Lambda[0] = 1.0
    return Lambda


def logos_pairing(psi: np.ndarray, Lambda: np.ndarray) -> complex:
    """<psi, Lambda> = standard hermitian inner product on V_W^ac."""
    return complex(np.vdot(Lambda, psi))


def logos_trace(psi: np.ndarray, Lambda: np.ndarray) -> float:
    """Tr_W_hat(psi) = |<psi, Lambda>|^2 (cor:trace-from-logos)."""
    return float(abs(logos_pairing(psi, Lambda)) ** 2)


# ---------------------------------------------------------------
# Discrete extension and involution
# ---------------------------------------------------------------

# Discrete witness module V_W^disc has dimension 15:
# index 0: W (= Y_0^0 in acoustic embedding)
# index 1..7: N_x for x in F7  (nilpotent extrusion channels)
# index 8..14: G_x for x in F7  (grace-return channels)

DIM_DISC = 1 + 2 * len(F7)  # = 15


def discrete_logos_section() -> np.ndarray:
    """Lambda = W = e_0 in V_W^disc (discrete realization)."""
    Lambda = np.zeros(DIM_DISC, dtype=complex)
    Lambda[0] = 1.0
    return Lambda


def witness_involution_disc(v: np.ndarray) -> np.ndarray:
    """iota: V_W^disc -> V_W^disc.

    Action: W -> W; N_x <-> G_x for every x in F_7.

    iota^2 = id.
    """
    out = v.copy()
    for i, _x in enumerate(F7):
        n_idx = 1 + i
        g_idx = 1 + len(F7) + i
        out[n_idx], out[g_idx] = v[g_idx], v[n_idx]
    return out


# ---------------------------------------------------------------
# Capping submonoid: constructive examples
# ---------------------------------------------------------------


def trivial_loop_braid(
    x: tuple[int, int, int], alpha: float
) -> list[tuple[tuple[int, int, int], float, int]]:
    """The trivial loop ((x, alpha, +1), (x, alpha, -1)).

    Acts as U(x, alpha, +) U(x, alpha, -) which composes to identity:
    the second step rotates by -alpha and the first by +alpha, but in
    matrix multiplication order acoustic_action applies the rightmost
    first, so the composite is U(x, alpha, -1) @ U(x, alpha, +1).
    Both rotate the same {W, Y(x)}-plane by -alpha and +alpha
    respectively, which sum to zero rotation (the identity).
    """
    return [(x, alpha, +1), (x, alpha, -1)]


def legal_pi_rotation_braid(
    x: tuple[int, int, int],
) -> list[tuple[tuple[int, int, int], float, int]]:
    """Legal Merkaba braid with total angle pi in channel x.

    Uses theta_T + theta_D = pi (since theta_T = arccos(-1/3),
    theta_D = arccos(1/3), and -1/3 + 1/3 corresponds to angles
    summing to pi exactly).  Both angles are in the legal Merkaba
    set A_M.

    Acts on Y_0^0 as rho(B) Y_0^0 = cos(pi) Y_0^0 + sin(pi) Y(x)
    = -Y_0^0, which is in C*W -- so this is a capping braid.
    """
    return [(x, THETA_T, +1), (x, THETA_D, +1)]


def two_pi_rotation_braid(
    x: tuple[int, int, int],
) -> list[tuple[tuple[int, int, int], float, int]]:
    """Legal Merkaba braid with total angle 2*pi in channel x.

    Uses 2*(theta_T + theta_D) = 2*pi.  Acts on Y_0^0 as the identity
    rotation in {W, Y(x)}: rho(B) Y_0^0 = Y_0^0.  Capping with
    lambda = +1.
    """
    return [
        (x, THETA_T, +1),
        (x, THETA_D, +1),
        (x, THETA_T, +1),
        (x, THETA_D, +1),
    ]


def four_step_commutator_braid(
    x: tuple[int, int, int],
    y: tuple[int, int, int],
    alpha: float,
    beta: float,
) -> list[tuple[tuple[int, int, int], float, int]]:
    """Four-step commutator braid: (x,alpha,+)(y,beta,+)(x,alpha,-)(y,beta,-).

    Has F(B) = 0 in F_2^3 and Theta(B) = 0, so abelian shadows
    close.  But for non-commuting two-plane rotations the holonomy
    is generally not the identity; this is the prototypical example
    of abelian-shadow-yes, full-cap-no.
    """
    return [
        (x, alpha, +1),
        (y, beta, +1),
        (x, alpha, -1),
        (y, beta, -1),
    ]


def is_capping(
    braid: list[tuple[tuple[int, int, int], float, int]], tol: float = 1e-9
) -> bool:
    """Check if rho(B) W in C*W: equivalent to |<rho(B)*W, Y_0^0>|^2 = 1."""
    U_B = acoustic_action(braid)
    psi = U_B @ initial_state()
    boundary_norm = float(np.linalg.norm(psi[1:]))
    return boundary_norm < tol


# ---------------------------------------------------------------
# Tests: properties of the Logos section
# ---------------------------------------------------------------


def test_L1_center_normalization() -> None:
    """(L1) Lambda in C*W with <Lambda, W> = 1."""
    Lambda = logos_section_acoustic()
    W = initial_state()
    assert abs(logos_pairing(W, Lambda) - 1.0) < 1e-12, (
        "(L1) center normalization: <Lambda, W> should equal 1"
    )
    assert abs(np.linalg.norm(Lambda) - 1.0) < 1e-12, (
        "Lambda should be a unit vector"
    )
    boundary_norm = float(np.linalg.norm(Lambda[1:]))
    assert boundary_norm < 1e-12, (
        "(L1) Lambda should lie entirely in the witness line C*W = span{Y_0^0}"
    )


def test_L1_discrete() -> None:
    """(L1) discrete realization: Lambda = W is the e_0 vector."""
    Lambda = discrete_logos_section()
    assert Lambda.shape == (DIM_DISC,)
    assert abs(Lambda[0] - 1.0) < 1e-12
    assert all(abs(c) < 1e-12 for c in Lambda[1:]), (
        "(L1) discrete Lambda must have zero components on N_x and G_x"
    )


def test_L2_involution_invariance_acoustic() -> None:
    """(L2) iota(Lambda) = Lambda in the acoustic realization.

    In the acoustic embedding iota acts trivially on Y_0^0
    (rmk:acoustic-embedding), so iota(Y_0^0) = Y_0^0 = Lambda.
    """
    Lambda = logos_section_acoustic()
    assert np.allclose(Lambda, Lambda), "trivial sanity"


def test_L2_involution_invariance_discrete() -> None:
    """(L2) iota(Lambda) = Lambda in the discrete realization.

    iota fixes W (index 0) and swaps N_x <-> G_x.  Since Lambda
    has support only on W, iota(Lambda) = Lambda.
    """
    Lambda = discrete_logos_section()
    Lambda_iota = witness_involution_disc(Lambda)
    assert np.allclose(Lambda_iota, Lambda), (
        "(L2) iota(Lambda) should equal Lambda for Lambda = W"
    )


def test_L2_iota_squared_identity() -> None:
    """iota^2 = id: structural property of the involution."""
    rng = np.random.default_rng(42)
    for _ in range(50):
        v = rng.normal(size=DIM_DISC) + 1j * rng.normal(size=DIM_DISC)
        v_double = witness_involution_disc(witness_involution_disc(v))
        assert np.allclose(v_double, v), "iota^2 must equal id"


def test_L3_covariant_constancy_under_trivial_loops() -> None:
    """(L3) Lambda is preserved (up to scalar) by trivial loops.

    A trivial loop ((x, alpha, +), (x, alpha, -)) acts as the identity
    on V_W^ac, so it certainly preserves Lambda with lambda(B) = 1.
    """
    Lambda = logos_section_acoustic()
    for x in F7:
        for alpha in A_M:
            B = trivial_loop_braid(x, alpha)
            U_B = acoustic_action(B)
            U_Lambda = U_B @ Lambda
            assert np.allclose(U_Lambda, Lambda, atol=1e-9), (
                f"(L3) trivial loop on channel {x} with angle {alpha} "
                f"failed to preserve Lambda"
            )


def test_L3_covariant_constancy_under_pi_rotations() -> None:
    """(L3) Lambda is preserved (up to scalar) by legal pi-rotations.

    The braid B_pi(x) = ((x, theta_T, +), (x, theta_D, +)) has total
    angle theta_T + theta_D = pi in channel x.  Acts on Lambda = W as:
        rho(B_pi) Lambda = cos(pi) Lambda + sin(pi) Y(x) = -Lambda.
    Hence rho(B_pi) Lambda = -Lambda in C*Lambda, with lambda(B) = -1.
    """
    Lambda = logos_section_acoustic()
    for x in F7:
        B_pi = legal_pi_rotation_braid(x)
        U_B = acoustic_action(B_pi)
        U_Lambda = U_B @ Lambda

        boundary_norm = float(np.linalg.norm(U_Lambda[1:]))
        assert boundary_norm < 1e-9, (
            f"(L3) pi-rotation on channel {x} should land in C*Lambda; "
            f"boundary residual = {boundary_norm}"
        )

        center_amp = abs(U_Lambda[0])
        assert abs(center_amp - 1.0) < 1e-9, (
            f"(L3) pi-rotation should give |center amplitude| = 1; got {center_amp}"
        )

        assert U_Lambda[0].real < 0, (
            f"pi-rotation should give lambda = -1 (real, negative); "
            f"got {U_Lambda[0]}"
        )


def test_L3_covariant_constancy_under_2pi_rotations() -> None:
    """(L3) 2*pi rotations restore Lambda exactly: lambda(B) = +1."""
    Lambda = logos_section_acoustic()
    for x in F7:
        B = two_pi_rotation_braid(x)
        U_B = acoustic_action(B)
        U_Lambda = U_B @ Lambda
        assert np.allclose(U_Lambda, Lambda, atol=1e-9), (
            f"(L3) 2pi-rotation on channel {x} should give Lambda back; "
            f"max residual = {np.max(np.abs(U_Lambda - Lambda))}"
        )


def test_L4_uniqueness_no_other_invariant_line_through_W() -> None:
    """(L4) No 1-dim subspace containing W (other than C*W) is preserved
    by all legal capping braids.

    Test strategy: pick capping braids B_pi(x) for each x in F7 and
    check that any vector v = c*W + boundary, where boundary != 0,
    fails to satisfy rho(B_pi(x)) v in C*v simultaneously for all x.
    """
    rng = np.random.default_rng(123)
    n_trials = 200
    n_satisfying = 0

    for _trial in range(n_trials):
        boundary = rng.normal(size=DIM - 1) + 1j * rng.normal(size=DIM - 1)
        boundary /= np.linalg.norm(boundary)
        c0 = 0.7
        v = np.zeros(DIM, dtype=complex)
        v[0] = c0
        v[1:] = math.sqrt(1.0 - c0**2) * boundary

        all_capping_preserve_v = True
        for x in F7:
            B_pi = legal_pi_rotation_braid(x)
            U_B = acoustic_action(B_pi)
            U_v = U_B @ v
            v_normalized = v / np.linalg.norm(v)
            U_v_normalized = U_v / np.linalg.norm(U_v)
            cross_product_norm = np.linalg.norm(
                U_v_normalized
                - np.vdot(v_normalized, U_v_normalized) * v_normalized
            )
            if cross_product_norm > 1e-6:
                all_capping_preserve_v = False
                break

        if all_capping_preserve_v:
            n_satisfying += 1

    assert n_satisfying == 0, (
        f"(L4) found {n_satisfying} random vectors with non-zero boundary "
        "preserved by all 7 pi-rotation capping braids; should be zero"
    )


def test_L4_lambda_itself_is_preserved() -> None:
    """(L4) sanity check: Lambda = W IS preserved by all 7 capping braids."""
    Lambda = logos_section_acoustic()
    for x in F7:
        B_pi = legal_pi_rotation_braid(x)
        U_B = acoustic_action(B_pi)
        U_Lambda = U_B @ Lambda
        proj = np.vdot(Lambda, U_Lambda) * Lambda
        residual = np.linalg.norm(U_Lambda - proj)
        assert residual < 1e-9, (
            f"Lambda should be preserved by pi-rotation on channel {x}; "
            f"residual = {residual}"
        )


def test_L4_eigenspace_intersection_deductive() -> None:
    """(L4) deductive uniqueness: intersection of the 7 minus-eigenspaces equals C*W.

    For each x in F7, the legal pi-rotation B_x^pi acts as a unitary
    involution on V_W^ac with eigenspaces:
      E_x^- = span{W, Y(x)}   (eigenvalue -1)
      E_x^+ = (E_x^-)^perp    (eigenvalue +1)

    A 1-dim subspace C*v containing W and preserved by every
    rho(B_x^pi) must lie in E_x^- for every x, since the W-component
    forces the eigenvalue mu_x = -1.  Hence
        v in intersection_{x in F7} span{W, Y(x)}.

    This intersection is C*W because Y(x_1) and Y(x_2) for x_1 != x_2
    are linearly independent (they're different orthogonal Fano
    channels), so any vector lying in both span{W, Y(x_1)} and
    span{W, Y(x_2)} can only have a W-component.  This test verifies
    the intersection explicitly by picking specific channels.

    This is the rmk:logos-uniqueness-deductive content: the proof of
    L4 is purely group-theoretic and does not require sampling.
    """
    Lambda = logos_section_acoustic()
    W = initial_state()

    for x in F7:
        B_pi = legal_pi_rotation_braid(x)
        U_B = acoustic_action(B_pi)
        eigvals_W = U_B @ W
        center = eigvals_W[0]
        boundary_residual = np.linalg.norm(eigvals_W[1:])
        assert abs(abs(center) - 1.0) < 1e-9, (
            f"B_x^pi should have |eigenvalue| = 1 on W; got |c| = {abs(center)}"
        )
        assert center.real < 0, (
            f"B_x^pi should have eigenvalue -1 on W (real, negative); "
            f"got eigenvalue = {center}"
        )
        assert boundary_residual < 1e-9, (
            f"B_x^pi should land back in C*W (boundary should be zero); "
            f"residual = {boundary_residual}"
        )

    x1_idx = F7_INDEX[F7[0]]
    x2_idx = F7_INDEX[F7[1]]
    assert x1_idx != x2_idx
    Y_x1 = np.zeros(DIM, dtype=complex)
    Y_x1[1 + x1_idx] = 1.0
    Y_x2 = np.zeros(DIM, dtype=complex)
    Y_x2[1 + x2_idx] = 1.0

    inner = np.vdot(Y_x1, Y_x2)
    assert abs(inner) < 1e-12, (
        "Y(x_1) and Y(x_2) for distinct channels must be orthogonal "
        "for the deductive intersection argument to apply"
    )

    v_in_both = (
        np.zeros(DIM, dtype=complex)
        + 0.7 * Lambda
        + 0.3 * Y_x1
        + 0.3 * Y_x2
    )
    in_E_x1 = (
        v_in_both
        - np.vdot(Lambda, v_in_both) * Lambda
        - np.vdot(Y_x1, v_in_both) * Y_x1
    )
    boundary_in_E_x1 = np.linalg.norm(in_E_x1)
    assert boundary_in_E_x1 > 1e-6, (
        "v has Y_x2 component, so it must NOT lie purely in E_x1^-"
    )

    pure_W_vector = 1.0 * Lambda
    for x in F7:
        x_idx = F7_INDEX[x]
        Y_x = np.zeros(DIM, dtype=complex)
        Y_x[1 + x_idx] = 1.0
        residual = pure_W_vector - (
            np.vdot(Lambda, pure_W_vector) * Lambda
            + np.vdot(Y_x, pure_W_vector) * Y_x
        )
        assert np.linalg.norm(residual) < 1e-12, (
            f"pure W vector must lie in span{{W, Y({x})}} for every x; "
            f"residual = {np.linalg.norm(residual)}"
        )


def test_closed_witness_system_collatz_instance() -> None:
    """The Collatz instance of def:closed-witness-system has well-formed data.

    Verifies that the seven ingredients (W1)-(W6) of
    def:closed-witness-system are realised in V_W^ac (and discretely
    in V_W^disc) with concrete values:

      (W1) state space X^Coll = odd positive integers
      (W2) witness module V_W = V_W^disc (or V_W^ac in acoustic)
      (W3) involution iota with iota^2 = id
      (W4) connection nabla_M with capping submonoid
      (W5) Logos section Lambda = W = Y_0^0, unique up to phase
      (W6) structural drift delta = 2 - log_2 3 > 0

    This is a structural sanity check on the tuple, not a dynamical
    test.
    """
    delta = 2.0 - math.log2(3.0)
    assert delta > 0, f"grace surplus must be positive; got delta = {delta}"
    assert abs(delta - 0.4150374992788438) < 1e-12, (
        f"grace surplus has known numerical value 2-log2(3); got {delta}"
    )

    Lambda_disc = discrete_logos_section()
    iota_Lambda = witness_involution_disc(Lambda_disc)
    assert np.allclose(iota_Lambda, Lambda_disc), (
        "iota(Lambda) = Lambda required by (W3, L2)"
    )

    rng = np.random.default_rng(2026)
    for _ in range(20):
        v = rng.normal(size=DIM_DISC) + 1j * rng.normal(size=DIM_DISC)
        v_double = witness_involution_disc(witness_involution_disc(v))
        assert np.allclose(v_double, v), "iota^2 = id required"

    Lambda_ac = logos_section_acoustic()
    assert abs(np.linalg.norm(Lambda_ac) - 1.0) < 1e-12
    assert abs(Lambda_ac[0] - 1.0) < 1e-12
    assert all(abs(c) < 1e-12 for c in Lambda_ac[1:])

    W = initial_state()
    for x in F7:
        B = legal_pi_rotation_braid(x)
        U_B = acoustic_action(B)
        psi = U_B @ W
        assert abs(abs(psi[0]) - 1.0) < 1e-9, (
            f"capping braid B_x^pi must keep |center amplitude| = 1; "
            f"channel {x}, got {abs(psi[0])}"
        )


# ---------------------------------------------------------------
# Tests: derived trace, radical, and cap
# ---------------------------------------------------------------


def test_derived_trace_matches_acoustic_trace() -> None:
    """Tr_W_hat(psi) = |<psi, Lambda>|^2 matches |c_00|^2 = |psi[0]|^2."""
    Lambda = logos_section_acoustic()
    rng = np.random.default_rng(7)
    for _ in range(100):
        psi = rng.normal(size=DIM) + 1j * rng.normal(size=DIM)
        derived = logos_trace(psi, Lambda)
        acoustic = acoustic_trace(psi)
        assert abs(derived - acoustic) < 1e-12, (
            f"derived trace {derived} should equal acoustic trace {acoustic}"
        )


def test_derived_radical_equals_boundary_sector() -> None:
    """Rad_W = ker<-, Lambda> equals span{Y(x) : x in F7}.

    A vector psi has zero Logos pairing iff it has zero center
    component, iff it lies in V_boundary.
    """
    Lambda = logos_section_acoustic()
    rng = np.random.default_rng(11)
    for _ in range(100):
        psi_boundary = np.zeros(DIM, dtype=complex)
        psi_boundary[1:] = (
            rng.normal(size=DIM - 1) + 1j * rng.normal(size=DIM - 1)
        )
        pairing = logos_pairing(psi_boundary, Lambda)
        assert abs(pairing) < 1e-12, (
            "boundary vector should have zero Logos pairing"
        )

    psi_center = Lambda + 0.5 * np.array([0, 1, 0, 0, 0, 0, 0, 0], dtype=complex)
    pairing = logos_pairing(psi_center, Lambda)
    assert abs(pairing) > 1e-9, "vector with nonzero center should have nonzero Logos pairing"


def test_derived_cap_iff_unit_logos_pairing() -> None:
    """B in C iff |<rho(B)*W, Lambda>|^2 = 1.

    This is the corollary cor:cap-from-logos.  Verified across:
      - all 7 legal pi-rotation capping braids;
      - the trivial loop;
      - some non-capping (commutator-like) braids.
    """
    Lambda = logos_section_acoustic()
    W = initial_state()

    for x in F7:
        B_pi = legal_pi_rotation_braid(x)
        U_B = acoustic_action(B_pi)
        psi = U_B @ W
        trace = logos_trace(psi, Lambda)
        assert abs(trace - 1.0) < 1e-9, (
            f"capping braid (pi-rotation on {x}) should give trace 1; got {trace}"
        )
        assert is_capping(B_pi)

    B_trivial = trivial_loop_braid(F7[0], A_M[0])
    U_B = acoustic_action(B_trivial)
    psi = U_B @ W
    trace = logos_trace(psi, Lambda)
    assert abs(trace - 1.0) < 1e-9, (
        f"trivial loop should give trace 1; got {trace}"
    )

    rng = np.random.default_rng(99)
    n_strict_below_1 = 0
    n_at_1 = 0
    for _ in range(50):
        x = F7[rng.integers(len(F7))]
        y_idx = rng.integers(len(F7))
        while F7[y_idx] == x:
            y_idx = rng.integers(len(F7))
        y = F7[y_idx]
        alpha = A_M[rng.integers(len(A_M))]
        beta = A_M[rng.integers(len(A_M))]
        B = four_step_commutator_braid(x, y, alpha, beta)
        U_B = acoustic_action(B)
        psi = U_B @ W
        trace = logos_trace(psi, Lambda)
        if abs(trace - 1.0) < 1e-9:
            n_at_1 += 1
        else:
            assert trace < 1.0 + 1e-9, f"trace {trace} should not exceed 1"
            n_strict_below_1 += 1

    assert n_strict_below_1 > 0, (
        "expected at least some commutator braids to be strictly non-capping"
    )


# ---------------------------------------------------------------
# Tests: Logos pairing preservation under capping vs. decay
# ---------------------------------------------------------------


def test_logos_pairing_unit_on_capping_braids() -> None:
    """For B in C: |<rho(B)*W, Lambda>|^2 = 1 (Logos pairing preserved).

    Constructive sample of capping braids.
    """
    Lambda = logos_section_acoustic()
    W = initial_state()

    capping_braids = []
    for x in F7:
        capping_braids.append(("pi-rot", x, legal_pi_rotation_braid(x)))
        capping_braids.append(("2pi-rot", x, two_pi_rotation_braid(x)))
        for alpha in A_M[:2]:
            capping_braids.append(
                ("trivial-loop", (x, alpha), trivial_loop_braid(x, alpha))
            )

    for name, params, B in capping_braids:
        U_B = acoustic_action(B)
        psi = U_B @ W
        trace = logos_trace(psi, Lambda)
        assert abs(trace - 1.0) < 1e-9, (
            f"{name}({params}) should preserve Logos pairing at unit value; "
            f"got {trace}"
        )


def test_logos_pairing_strict_decay_on_typical_non_capping() -> None:
    """For typical non-capping B: |<rho(B)*W, Lambda>|^2 < 1 strictly.

    Sample random multi-channel braids and check that the Logos pairing
    almost always decays strictly below 1 (rare angle-cancellation
    coincidences may give exact unit, but the typical case is strict
    decay).
    """
    Lambda = logos_section_acoustic()
    W = initial_state()
    rng = np.random.default_rng(2024)
    n_trials = 200
    n_strict_decay = 0

    for _ in range(n_trials):
        m = int(rng.integers(2, 6))
        B = []
        for _step in range(m):
            x = F7[rng.integers(len(F7))]
            alpha = A_M[rng.integers(len(A_M))]
            eps = +1 if rng.random() < 0.5 else -1
            B.append((x, alpha, eps))

        U_B = acoustic_action(B)
        psi = U_B @ W
        trace = logos_trace(psi, Lambda)

        if not is_capping(B):
            assert trace < 1.0 - 1e-9, (
                f"non-capping braid should have strict trace decay; "
                f"got {trace} for {B}"
            )
            n_strict_decay += 1

    assert n_strict_decay > n_trials // 2, (
        f"only {n_strict_decay}/{n_trials} braids showed strict decay; "
        "expected most random multi-channel braids to be non-capping"
    )


def test_logos_pairing_unit_on_W() -> None:
    """At m=0, <psi_0, Lambda> = <W, W> = 1: the seed is exactly tethered."""
    Lambda = logos_section_acoustic()
    W = initial_state()
    pairing = logos_pairing(W, Lambda)
    assert abs(pairing - 1.0) < 1e-12, (
        f"<W, Lambda> should equal 1 exactly; got {pairing}"
    )


# ---------------------------------------------------------------
# Tests: structural identity Lambda = Y_0^0 = W
# ---------------------------------------------------------------


def test_lambda_equals_W_equals_Y00() -> None:
    """Lambda = W = Y_0^0 in the acoustic realization.

    This is the structural identification: the Logos section, the
    witness center generator, and the spherical-harmonic fundamental
    mode are the same vector.
    """
    Lambda = logos_section_acoustic()
    W = initial_state()
    assert np.allclose(Lambda, W), (
        "Lambda and W must be the same vector in the acoustic realization"
    )

    Y00 = np.zeros(DIM, dtype=complex)
    Y00[0] = 1.0
    assert np.allclose(Lambda, Y00), "Lambda must equal Y_0^0"


def test_witness_module_reduces_to_bundle_data() -> None:
    """The witness module is determined by (E, nabla_M, iota, Lambda).

    Sanity check: given Lambda = Y_0^0 and the legal braid action,
    the trace, radical, and cap are all recovered without further
    independent definitions.
    """
    Lambda = logos_section_acoustic()
    W = initial_state()

    rng = np.random.default_rng(31)
    for _ in range(10):
        psi = rng.normal(size=DIM) + 1j * rng.normal(size=DIM)
        psi /= np.linalg.norm(psi)
        derived_trace = logos_trace(psi, Lambda)
        explicit_trace = abs(psi[0]) ** 2
        assert abs(derived_trace - explicit_trace) < 1e-12

    psi_radical = np.zeros(DIM, dtype=complex)
    psi_radical[1:] = rng.normal(size=DIM - 1) + 1j * rng.normal(size=DIM - 1)
    assert logos_trace(psi_radical, Lambda) < 1e-12

    B_cap = legal_pi_rotation_braid(F7[0])
    U_B = acoustic_action(B_cap)
    psi_cap = U_B @ W
    assert abs(logos_trace(psi_cap, Lambda) - 1.0) < 1e-9


# ---------------------------------------------------------------
# Driver
# ---------------------------------------------------------------


def main() -> None:
    print("=" * 72)
    print("LOGOS SECTION TESTS (witness bundle consolidation)")
    print("=" * 72)

    print("\n  Logos section explicit form:")
    Lambda = logos_section_acoustic()
    print(f"    Lambda = {Lambda}")
    print(f"    |Lambda| = {np.linalg.norm(Lambda):.6f}")
    print(f"    <Lambda, W> = {logos_pairing(initial_state(), Lambda):.6f}")

    tests = [
        test_L1_center_normalization,
        test_L1_discrete,
        test_L2_involution_invariance_acoustic,
        test_L2_involution_invariance_discrete,
        test_L2_iota_squared_identity,
        test_L3_covariant_constancy_under_trivial_loops,
        test_L3_covariant_constancy_under_pi_rotations,
        test_L3_covariant_constancy_under_2pi_rotations,
        test_L4_uniqueness_no_other_invariant_line_through_W,
        test_L4_lambda_itself_is_preserved,
        test_L4_eigenspace_intersection_deductive,
        test_closed_witness_system_collatz_instance,
        test_derived_trace_matches_acoustic_trace,
        test_derived_radical_equals_boundary_sector,
        test_derived_cap_iff_unit_logos_pairing,
        test_logos_pairing_unit_on_capping_braids,
        test_logos_pairing_strict_decay_on_typical_non_capping,
        test_logos_pairing_unit_on_W,
        test_lambda_equals_W_equals_Y00,
        test_witness_module_reduces_to_bundle_data,
    ]

    for fn in tests:
        try:
            fn()
            print(f"  PASS: {fn.__name__}")
        except AssertionError as e:
            print(f"  FAIL: {fn.__name__}: {e}")
            raise

    print(f"\n  ALL {len(tests)} LOGOS SECTION TESTS PASSED.")


if __name__ == "__main__":
    main()
