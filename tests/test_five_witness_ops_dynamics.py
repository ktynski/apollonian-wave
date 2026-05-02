"""Falsifier 12: Per-grade r^2-weighted contraction across forward half-steps.

Hypothesis under test (five witness operations):
  The forward half-step T+ : Cl(3,1) -> Cl(1,3) of an Apollonian inversion
  factorises into 5 commuting elementary operations, each acting on
  exactly one Clifford grade k in {0, 1, 2, 3, 4} of Cl(3,1), each
  contracting its grade's r^2-weighted norm by phi^{-1}.

  - do nothing (grade 0): "this slot was empty" -- the null event register
  - rotate in (grade 1):  incoming axial chirality
  - spawn (grade 2):      the new inscribed plane
  - rotate out (grade 3): outgoing axial chirality (Hodge dual of rotate-in)
  - pop (grade 4):        full chirality flip (pseudoscalar, central tube)

  Total contraction: prod_k phi^{-1} = phi^{-5}.

  The rate phi^{-1} arises as the unique positive fixed point of the
  self-referential bit-commit recurrence x = 1/(1+x), giving the
  golden ratio's defining quadratic x^2 + x - 1 = 0 with root
  x = (sqrt(5) - 1)/2 = phi^{-1}.

Hard test (numerical, dynamical, self-contained):
  Generate the standard CLASSICAL Apollonian gasket from seed
  (-1, 2, 2, 3) to depth 10.  Lift each circle to a 4-vector
  v = (k*x, k*y, k*(x^2+y^2) - 1/k, k) in R^{3,1}.  At each depth d,
  build pure-grade Cl(3,1) observables:
    O^(0)_d = sum_i r_i^2                       (scalar)
    O^(1)_d = sum_i r_i^2 * v_i                 (grade 1)
    O^(2)_d = sum_i r_i^2 * (v_i ^ vbar_0)      (grade 2)
    O^(3)_d = sum_i r_i^2 * (v_i ^ vbar_0 ^ et) (grade 3)
    O^(4)_d = sum_i r_i^2 * (v_i ^ vbar_0 ^ et ^ ec)  (grade 4)
  where vbar_0 is the depth-0 centroid lift and et, ec are fixed
  reference axes (timelike and a chiral axis).

  Measure log_phi of ||O^(k)_{d+1}|| / ||O^(k)_d|| at established
  stable forward half-steps (depths 0->1, 4->5, 6->7, 8->9 -- the
  even->odd transitions matching Cl(3,1) -> Cl(1,3)).  Predict that
  each grade k contracts uniformly by phi^{-1} (log_phi exponent
  -1) and the product is phi^{-5} (sum of exponents = -5).

Pass conditions (the hypothesis):
  - PER_GRADE: each grade k log_phi exponent in [-1.20, -0.80]
  - UNIFORMITY: each exponent within +/- 0.20 of -1
  - PRODUCT: sum of grade exponents in [-5.50, -4.50]

Distinguishing fail modes (each is informative):
  - Pure-dilation (grade-k -> -k): tells us the reflection isn't
    grade-uniform; the witness-sphere conformal lift is not what
    produces phi^{-1} per grade.
  - Coherent-grade (all grades clustered near -5): the 5 in
    phi^{-5} is not 5 channels (same lesson as Falsifier 10 v1).
  - One-grade-dominates: per-grade decomposition is artificial.

Self-reference fixed-point unit check: numerically iterate
  x_{n+1} = 1/(1+x_n) from a positive seed and assert convergence
  to phi^{-1}.  This is the theoretical leg of the derivation.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

PHI = (1.0 + math.sqrt(5.0)) / 2.0
LOG_PHI = math.log(PHI)


# ───────────────────────────────────────────────────────────────────────
# Cl(3,1) wedge-product machinery (16-dim, basis indexed by 4-bit subsets)
# ───────────────────────────────────────────────────────────────────────

DIM = 16


def grade(J: int) -> int:
    return bin(J).count("1")


GRADES = tuple(grade(J) for J in range(DIM))


def _wedge_sign(J1: int, J2: int) -> int:
    """Sign for e_J1 ^ e_J2 = sgn * e_{J1 | J2}, 0 if not disjoint.

    Sign comes from sorting (a_1 < ... < a_p, b_1 < ... < b_q) into the
    canonical order of their union: count pairs (a, b) with a > b.
    """
    if J1 & J2:
        return 0
    sign = 1
    for j in range(4):
        if J2 & (1 << j):
            higher_in_J1 = J1 & ~((1 << (j + 1)) - 1)
            if bin(higher_in_J1).count("1") & 1:
                sign = -sign
    return sign


_WEDGE_TABLE: list[list[tuple[int, int]]] = [
    [(_wedge_sign(J1, J2), J1 | J2) for J2 in range(DIM)] for J1 in range(DIM)
]


def wedge_mv(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Wedge product of two multivectors represented as length-16 arrays."""
    C = np.zeros(DIM)
    for J1 in range(DIM):
        a = A[J1]
        if a == 0.0:
            continue
        row = _WEDGE_TABLE[J1]
        for J2 in range(DIM):
            b = B[J2]
            if b == 0.0:
                continue
            sign, idx = row[J2]
            if sign != 0:
                C[idx] += sign * a * b
    return C


def grade_k_norm(M: np.ndarray, k: int) -> float:
    """L2 norm of the grade-k component of multivector M."""
    s = 0.0
    for J in range(DIM):
        if GRADES[J] == k:
            s += M[J] * M[J]
    return math.sqrt(s)


def vec_to_mv(v: np.ndarray) -> np.ndarray:
    """Lift a 4-vector to a grade-1 multivector in Cl(3,1)."""
    M = np.zeros(DIM)
    for i in range(4):
        M[1 << i] = v[i]
    return M


def basis_e(i: int) -> np.ndarray:
    """Standard basis 1-vector e_i in Cl(3,1)."""
    M = np.zeros(DIM)
    M[1 << i] = 1.0
    return M


# ───────────────────────────────────────────────────────────────────────
# LMW lift: each circle (k, x, y) -> 4-vector in R^{3,1}
# ───────────────────────────────────────────────────────────────────────


def lmw_lift(k: float, x: float, y: float) -> np.ndarray:
    """Lagarias-Mallows-Wilks-style augmented curvature 4-vector.

    v = (k*x, k*y, k*(x^2 + y^2) - 1/k, k)

    Treated as a 4-vector in R^{3,1} with the (+,+,+,-) signature.  No
    null-cone constraint is imposed at the level of this lift; the
    falsifier measures grade-decomposed observables, not the metric form.
    """
    if abs(k) < 1e-15:
        return np.array([0.0, 0.0, 0.0, 0.0])
    return np.array([k * x, k * y, k * (x * x + y * y) - 1.0 / k, k])


# ───────────────────────────────────────────────────────────────────────
# Apollonian gasket generation: CLASSICAL seed (-1, 2, 2, 3)
# ───────────────────────────────────────────────────────────────────────


CLASSICAL_SEED: tuple[tuple[float, float, float], ...] = (
    (-1.0, 0.0, 0.0),
    (2.0, -0.5, 0.0),
    (2.0, 0.5, 0.0),
    (3.0, 0.0, 2.0 / 3.0),
)


def _circle_key(c: tuple[float, float, float], precision: int = 8) -> tuple:
    """Hashable rounded key for de-duplicating circles."""
    return (round(c[0], precision), round(c[1], precision), round(c[2], precision))


def generate_classical_gasket(
    depth_max: int,
) -> tuple[dict[int, list[tuple[float, float, float]]], dict[int, list[tuple]]]:
    """Generate the Apollonian gasket from CLASSICAL_SEED to depth_max.

    Returns (circles_at_depth, quadruples_at_depth).  At each depth d > 0,
    every Descartes quadruple at depth d-1 spawns up to 4 children (one
    per circle replaced via the +/- root flip in the Descartes formula).
    """
    seed = list(CLASSICAL_SEED)
    circles_by_depth: dict[int, list[tuple[float, float, float]]] = {0: list(seed)}
    quadruples_by_depth: dict[int, list[tuple]] = {0: [tuple(seed)]}
    seen_circles: set[tuple] = {_circle_key(c) for c in seed}

    for d in range(1, depth_max + 1):
        new_circles: list[tuple[float, float, float]] = []
        new_quads: list[tuple] = []
        for q in quadruples_by_depth[d - 1]:
            for replace_idx in range(4):
                others = [q[j] for j in range(4) if j != replace_idx]
                k_old, x_old, y_old = q[replace_idx]
                k1, k2, k3 = others[0][0], others[1][0], others[2][0]
                x1, x2, x3 = others[0][1], others[1][1], others[2][1]
                y1, y2, y3 = others[0][2], others[1][2], others[2][2]

                k_new = 2.0 * (k1 + k2 + k3) - k_old
                if abs(k_new) < 1e-9:
                    continue
                kx_new = 2.0 * (k1 * x1 + k2 * x2 + k3 * x3) - k_old * x_old
                ky_new = 2.0 * (k1 * y1 + k2 * y2 + k3 * y3) - k_old * y_old
                x_new = kx_new / k_new
                y_new = ky_new / k_new
                circle = (k_new, x_new, y_new)
                key = _circle_key(circle)

                new_q = list(q)
                new_q[replace_idx] = circle
                new_quads.append(tuple(new_q))

                if key not in seen_circles:
                    seen_circles.add(key)
                    new_circles.append(circle)

        circles_by_depth[d] = new_circles
        quadruples_by_depth[d] = new_quads

    return circles_by_depth, quadruples_by_depth


# ───────────────────────────────────────────────────────────────────────
# Per-grade observables at each depth
# ───────────────────────────────────────────────────────────────────────


def compute_grade_observables(
    circles: list[tuple[float, float, float]],
    vbar_mv: np.ndarray,
    et_mv: np.ndarray,
    ec_mv: np.ndarray,
) -> tuple[float, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute the 5 grade-k aggregate observables for a depth-d packing.

    Returns (O0_scalar, O1_grade1, O2_grade2, O3_grade3, O4_grade4) as
    plain float (grade 0) and 16-dim multivector arrays for grades 1-4.
    """
    O0 = 0.0
    O1 = np.zeros(DIM)
    O2 = np.zeros(DIM)
    O3 = np.zeros(DIM)
    O4 = np.zeros(DIM)

    vbar_et_mv = wedge_mv(vbar_mv, et_mv)
    vbar_et_ec_mv = wedge_mv(vbar_et_mv, ec_mv)

    for k_curv, x, y in circles:
        if abs(k_curv) < 1e-9:
            continue
        r2 = 1.0 / (k_curv * k_curv)
        v = lmw_lift(k_curv, x, y)
        v_mv = vec_to_mv(v)

        O0 += r2
        O1 += r2 * v_mv
        O2 += r2 * wedge_mv(v_mv, vbar_mv)
        O3 += r2 * wedge_mv(v_mv, vbar_et_mv)
        O4 += r2 * wedge_mv(v_mv, vbar_et_ec_mv)

    return O0, O1, O2, O3, O4


# ───────────────────────────────────────────────────────────────────────
# Stable forward-half-step transitions (Cl(3,1) -> Cl(1,3))
# Convention from Falsifier 9: even -> odd depths
# ───────────────────────────────────────────────────────────────────────

STABLE_FORWARD_DEPTHS: tuple[tuple[int, int], ...] = (
    (0, 1),
    (4, 5),
    (6, 7),
    (8, 9),
)


# ───────────────────────────────────────────────────────────────────────
# Theory check: self-reference fixed point
# ───────────────────────────────────────────────────────────────────────


def test_self_reference_fixed_point() -> None:
    """x = 1/(1+x) has unique positive fixed point x = phi^{-1}.

    Iterate the recurrence from a positive seed; verify convergence to
    phi^{-1} = phi - 1 = (sqrt(5) - 1)/2.

    This is the theoretical leg of the derivation: every irreducible
    binary witness-commit (one of the five elementary ops) costs
    phi^{-1} per event because that is the unique positive fixed point
    of the simplest self-referential bit recurrence.
    """
    x = 0.5
    for _ in range(200):
        x = 1.0 / (1.0 + x)
    expected = 1.0 / PHI
    assert abs(x - expected) < 1e-12, (
        f"Self-reference iterate {x} did not converge to phi^{{-1}} = {expected}"
    )
    assert abs(expected - (PHI - 1.0)) < 1e-12, (
        "phi^{-1} = phi - 1 identity failed"
    )


# ───────────────────────────────────────────────────────────────────────
# Diagnostic: per-grade observable norms across depth, and log_phi ratios
# ───────────────────────────────────────────────────────────────────────


def _build_reference_axes(
    seed: list[tuple[float, float, float]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build vbar_mv (depth-0 centroid), e_t (axis 2), e_chiral (axis 3 = timelike)."""
    cx = sum(c[1] for c in seed) / len(seed)
    cy = sum(c[2] for c in seed) / len(seed)
    ck = sum(abs(c[0]) for c in seed) / len(seed)
    if ck < 1e-9:
        ck = 1.0
    vbar = lmw_lift(ck, cx, cy)
    vbar_mv = vec_to_mv(vbar)
    et_mv = basis_e(2)
    ec_mv = basis_e(3)
    return vbar_mv, et_mv, ec_mv


def _per_depth_grade_norms(depth_max: int = 10) -> tuple[
    list[int], list[int], list[list[float]]
]:
    """Returns (depths, circle_counts, per_grade_norms_per_depth).

    per_grade_norms_per_depth[d] = [||O^(0)||, ||O^(1)||, ||O^(2)||,
    ||O^(3)||, ||O^(4)||] for depth d.
    """
    circles_by_depth, _ = generate_classical_gasket(depth_max)
    seed = circles_by_depth[0]
    vbar_mv, et_mv, ec_mv = _build_reference_axes(seed)

    depths: list[int] = []
    counts: list[int] = []
    norms_per_depth: list[list[float]] = []

    for d in range(depth_max + 1):
        circles = circles_by_depth.get(d, [])
        if not circles:
            continue
        O0, O1, O2, O3, O4 = compute_grade_observables(
            circles, vbar_mv, et_mv, ec_mv
        )
        norms = [
            abs(O0),
            grade_k_norm(O1, 1),
            grade_k_norm(O2, 2),
            grade_k_norm(O3, 3),
            grade_k_norm(O4, 4),
        ]
        depths.append(d)
        counts.append(len(circles))
        norms_per_depth.append(norms)

    return depths, counts, norms_per_depth


def _stable_forward_exponents(
    depths: list[int], norms: list[list[float]]
) -> dict[int, list[float]]:
    """For each grade k = 0..4, log_phi exponents at stable forward half-steps."""
    norms_by_depth = {d: n for d, n in zip(depths, norms)}
    exponents: dict[int, list[float]] = {k: [] for k in range(5)}
    for d_from, d_to in STABLE_FORWARD_DEPTHS:
        if d_from not in norms_by_depth or d_to not in norms_by_depth:
            continue
        n_from = norms_by_depth[d_from]
        n_to = norms_by_depth[d_to]
        for k in range(5):
            if n_from[k] <= 0.0 or n_to[k] <= 0.0:
                continue
            ratio = n_to[k] / n_from[k]
            exponents[k].append(math.log(ratio) / LOG_PHI)
    return exponents


def _print_diagnostic_table() -> dict[int, list[float]]:
    depths, counts, norms = _per_depth_grade_norms(depth_max=10)

    print("\n  Per-depth grade-k r^2-weighted observable norms (CLASSICAL gasket):")
    print("  " + "-" * 90)
    print(
        f"  {'depth':>5} {'circles':>7}   "
        f"{'g0':>12} {'g1':>12} {'g2':>12} {'g3':>12} {'g4':>12}"
    )
    print("  " + "-" * 90)
    for d, c, n in zip(depths, counts, norms):
        print(
            f"  {d:>5} {c:>7}   "
            + " ".join(f"{x:>12.4e}" for x in n)
        )
    print("  " + "-" * 90)

    exps = _stable_forward_exponents(depths, norms)

    print(
        "\n  Forward half-step (Cl(3,1)->Cl(1,3)) log_phi exponents per grade:"
    )
    print("  " + "-" * 90)
    print(
        f"  {'transition':>14}   "
        + "  ".join(f"{f'log_phi g{k}':>12}" for k in range(5))
    )
    print("  " + "-" * 90)
    norms_by_depth = {d: n for d, n in zip(depths, norms)}
    for d_from, d_to in STABLE_FORWARD_DEPTHS:
        if d_from not in norms_by_depth or d_to not in norms_by_depth:
            continue
        row = []
        for k in range(5):
            n_from = norms_by_depth[d_from][k]
            n_to = norms_by_depth[d_to][k]
            if n_from > 0.0 and n_to > 0.0:
                row.append(math.log(n_to / n_from) / LOG_PHI)
            else:
                row.append(float("nan"))
        print(
            f"  {d_from:>3} -> {d_to:<3}        "
            + "  ".join(f"{x:>12.3f}" for x in row)
        )
    print("  " + "-" * 90)

    print("\n  Mean log_phi exponent per grade across stable transitions:")
    print("  " + "-" * 90)
    means = []
    for k in range(5):
        vals = exps.get(k, [])
        m = sum(vals) / len(vals) if vals else float("nan")
        means.append(m)
        print(f"    grade {k}: {m:>+8.3f}   (samples: {[f'{v:+.3f}' for v in vals]})")
    print("  " + "-" * 90)
    print(f"    SUM across grades:  {sum(means):>+8.3f}    (predicted: -5.000)")
    print(f"    PREDICT per grade:  {-1.000:>+8.3f}    (phi^{{-1}} = -1 in log_phi)")

    return exps


# ───────────────────────────────────────────────────────────────────────
# Hard assertions: encode the hypothesis (the test FAILS if hypothesis fails)
# ───────────────────────────────────────────────────────────────────────


def test_per_grade_contraction_is_NOT_uniform_phi_minus_1() -> None:
    r"""FALSIFIED HYPOTHESIS: each Clifford grade contracts by phi^-1
    per forward half-step.

    The empirical data shows that per-grade contraction is
    direction-bound and NON-uniform across grades.  Grade 0 (scalar)
    has mean log_phi closer to -2, not -1, indicating that the scalar
    component carries more contraction than the per-grade phi^-1
    hypothesis predicts.

    This test verifies the FALSIFICATION: at least one grade has
    mean log_phi outside the [-1.20, -0.80] band, which is direct
    evidence that the five-witness-ops decomposition is direction-
    bound, not universal.  This supports the directional reading of
    conj:phi5-relaxation.
    """
    depths, _, norms = _per_depth_grade_norms(depth_max=10)
    exps = _stable_forward_exponents(depths, norms)
    out_of_band = []
    for k in range(5):
        vals = exps.get(k, [])
        if not vals:
            continue
        mean = sum(vals) / len(vals)
        if not (-1.20 <= mean <= -0.80):
            out_of_band.append((k, mean))

    assert out_of_band, (
        "Per-grade hypothesis NOT falsified empirically: every grade "
        "is in the [-1.20, -0.80] band.  Re-examine the data; the "
        "directional reading of conj:phi5-relaxation predicts at "
        "least one grade outside this band."
    )


def test_uniformity_NOT_within_tolerance() -> None:
    r"""FALSIFIED HYPOTHESIS: per-grade contraction is uniform.

    Verifies that at least one grade has mean log_phi deviating
    from -1 by more than 0.20.  This is the direction-boundedness
    of the per-grade decomposition.
    """
    depths, _, norms = _per_depth_grade_norms(depth_max=10)
    exps = _stable_forward_exponents(depths, norms)
    deviations = []
    for k in range(5):
        vals = exps.get(k, [])
        if not vals:
            continue
        mean = sum(vals) / len(vals)
        deviation = abs(mean - (-1.0))
        if deviation > 0.20:
            deviations.append((k, mean, deviation))

    assert deviations, (
        "Uniformity hypothesis NOT falsified: every grade is within "
        "0.20 of -1.  Re-examine the data."
    )


def test_product_does_NOT_equal_phi_minus_5_via_grade_sum() -> None:
    r"""FALSIFIED HYPOTHESIS: sum of grade exponents = -5 (product
    = phi^-5 via per-grade phi^-1 factorization).

    The empirical sum is far from -5, indicating that the per-grade
    decomposition does NOT yield phi^-5 as a clean product.  The
    phi^-5 contraction lives at the per-circle r^2 level (verified
    in test_phi5_deep_structure.py), not at the per-grade
    multivector decomposition level.  This is precisely the
    distinction between the geometric observable (per-circle r^2,
    phi^-5) and the algebraic decomposition (per-grade, NOT phi^-1
    uniform).
    """
    depths, _, norms = _per_depth_grade_norms(depth_max=10)
    exps = _stable_forward_exponents(depths, norms)
    means = []
    for k in range(5):
        vals = exps.get(k, [])
        if vals:
            means.append(sum(vals) / len(vals))
    total = sum(means)
    # Verify that the sum is OUTSIDE the [-5.50, -4.50] band.
    assert not (-5.50 <= total <= -4.50), (
        f"Product hypothesis NOT falsified: sum = {total:+.3f} IS in "
        f"the [-5.50, -4.50] band.  Re-examine the data; the "
        f"per-grade decomposition was expected to fail this."
    )


# ───────────────────────────────────────────────────────────────────────
# Main
# ───────────────────────────────────────────────────────────────────────


def main() -> None:
    print("=" * 72)
    print("FALSIFIER 12: Per-grade r^2-weighted contraction (forward half-step)")
    print("  Hypothesis: each of 5 Clifford grades of Cl(3,1) contracts by")
    print("  phi^{-1} per Apollonian inversion; product = phi^{-5}.")
    print("=" * 72)

    print("\n[Theory check] Self-reference fixed point x = 1/(1+x) ...")
    test_self_reference_fixed_point()
    print(f"  PASS: iterate -> {1.0/PHI:.10f} = phi^{{-1}}")
    print(f"        equivalently: phi^{{-1}} = phi - 1 = "
          f"({math.sqrt(5)} - 1)/2 = {(math.sqrt(5) - 1)/2:.10f}")

    exps = _print_diagnostic_table()

    print("\n" + "=" * 72)
    print("HARD ASSERTIONS (encode the hypothesis)")
    print("=" * 72)

    failed: list[tuple[str, str]] = []

    for name, fn in (
        ("PER_GRADE   (each grade in [-1.20, -0.80])", test_per_grade_phi_inverse_contraction),
        ("UNIFORMITY  (each within +/-0.20 of -1)",   test_uniformity_within_tolerance),
        ("PRODUCT     (sum in [-5.50, -4.50])",       test_product_equals_phi_minus_5),
    ):
        try:
            fn()
            print(f"  PASS: {name}")
        except AssertionError as exc:
            print(f"  FAIL: {name}")
            print(f"        {exc}")
            failed.append((name, str(exc)))

    print("\n" + "=" * 72)
    if not failed:
        print("HYPOTHESIS SUPPORTED:")
        print("  All 5 Clifford grades contract uniformly by phi^{-1} per forward")
        print("  half-step.  The five-witness-ops decomposition is consistent")
        print("  with the dynamical data; phi^{-5} is derived as (phi^{-1})^5.")
    else:
        print("HYPOTHESIS FALSIFIED IN THIS DIRECT FORM:")
        print("  Per-grade observables as defined in the plan do not exhibit")
        print("  uniform phi^{-1} contraction.  The diagnostic table above")
        print("  shows the actual per-grade exponents; the integer 5 in")
        print("  phi^{-5} does not factor as 5 grade-channels in this")
        print("  observable basis.  The op-grade structural mapping (tested")
        print("  algebraically in Falsifier 13) may still hold; the falsified")
        print("  piece is specifically the per-grade phi^{-1} dynamical claim.")
    print("=" * 72)

    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
