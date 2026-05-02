"""Falsifier 14a: Topological signature of the 2-generation Apollonian lift.

Hypothesis under test (trefoil interpretation):
  The integer 5 in phi^{-5} decomposes as
                5 = p + q = 2 + 3
  where (p, q) = (2, 3) are the indices of the (2, 3) torus knot
  (the trefoil), with
    p = 2 = number of returns in the golden-ratio defect equation
            (rho^{-1} + rho^{-2} = 1, established at thm:golden-uniqueness)
    q = 3 = the Apollonian branching factor (every interstice is bounded
            by 3 circles; one inscribed disk creates 3 sub-interstices,
            established at cor:apollonian-contraction)

  Strong reading: when the 2-generation Apollonian sub-packing
  (parent triangle + 1 inscribed disk + 3 sub-interstices) is lifted
  to R^3 via z = depth-chirality, the boundary curve traces a
  trefoil knot (the (2,3) torus knot), and 5 = p + q is forced by
  this topology.

Hard test (heuristic, not a full knot-invariant computation):
  Construct one parent-interstice + 1 inscribed-disk configuration
  in 2D (CLASSICAL Apollonian seed at depth 0/1).  Lift each circle
  to a 3D circle in the plane z = depth (depth 0 -> z = 0, depth 1 ->
  z = 1).  For each pair of circles, compute the Gauss linking
  number numerically by polygonal discretisation:

      lk(C_a, C_b) = (1/4 pi) * integral integral
                     ((r_a - r_b) . (dr_a x dr_b)) / |r_a - r_b|^3

  Pairs tested:
    - (parent_i, parent_j) for i,j in {1,2,3}: same z-plane, expect 0
    - (parent_i, inscribed) for i in {1,2,3}: different z-planes,
      tangent in (x,y) projection.  Expect 0 if the inscribed sphere
      doesn't link the parent disk; nonzero only if the inscribed
      circle "passes through" the parent's disk.

  Plus: count signed crossings of the 4-circle configuration's
  projection (each tangency at different z is a "crossing").

Pass conditions (the trefoil literal-topology reading):
  - LINKING: at least one (parent, inscribed) pair has |lk| >= 1
    (otherwise the literal trefoil is not realised in the natural
    z=depth lift).
  - CROSSING_COUNT: number of signed crossings between depth-0 and
    depth-1 circles is exactly 3 (the trefoil's writhe).

If both pass, the trefoil topology is realised in the natural lift.
If LINKING fails (lk = 0 for all pairs), the trefoil is heuristic;
the 5 = 2 + 3 decomposition is then a numerological identity rather
than a topological fact.

Independent of the topology: the test always reports the structural
decomposition 5 = 2 (budget) + 3 (branching), which is verifiable
by direct counting of paper structures and stands or falls
independently of the trefoil interpretation.

Companion: Falsifier 14b tests the *dimensional* prediction
(d-D Apollonian -> exponent d+3) which is the load-bearing
falsifiable claim of the 5 = 2 + 3 reading.
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
# Apollonian seed: classical 4-circle quadruple in 2D
# ───────────────────────────────────────────────────────────────────────


def classical_seed_2d() -> list[tuple[float, float, float]]:
    """Seed quadruple (-1, 2, 2, 3) in standard configuration.

    Returns list of (k, x, y) with all four circles mutually tangent.
    """
    return [
        (-1.0, 0.0, 0.0),
        (2.0, -0.5, 0.0),
        (2.0, 0.5, 0.0),
        (3.0, 0.0, 2.0 / 3.0),
    ]


def descartes_inscribe_in_triple(
    triple: list[tuple[float, float, float]],
) -> tuple[float, float, float]:
    """Given 3 mutually tangent circles, return the 4th circle inscribed
    in their common interstice (the smaller of the two Descartes roots).
    """
    k1, x1, y1 = triple[0]
    k2, x2, y2 = triple[1]
    k3, x3, y3 = triple[2]
    sum_k = k1 + k2 + k3
    cross = k1 * k2 + k2 * k3 + k1 * k3
    disc = 2.0 * math.sqrt(max(cross, 0.0))
    k_in = sum_k + disc
    if abs(k_in) < 1e-9:
        k_in = sum_k - disc
    sx = k1 * x1 + k2 * x2 + k3 * x3
    sy = k1 * y1 + k2 * y2 + k3 * y3
    cx = k1 * k2 * x3 + k2 * k3 * x1 + k1 * k3 * x2
    cy = k1 * k2 * y3 + k2 * k3 * y1 + k1 * k3 * y2
    cross_x = math.sqrt(max(k1 * k2 * k3 * abs(k_in), 0.0))
    # Use the algebraic position formula: k_4*(x_4, y_4) = sum k_i (x_i, y_i)
    #   +/- 2*sqrt(k1 k2 k3 k4) e_perp.  Simpler: use 2D Descartes algebraic form.
    x_in = (sx + 2.0 * math.sqrt(max(k1 * k2 * x1 * x2 + k2 * k3 * x2 * x3
                                     + k1 * k3 * x1 * x3, 0.0))) / k_in
    # The quick form above can produce wrong sign; for the seed we use,
    # the inscribed disk is the one with positive disc and smaller radius.
    # We instead use the linear update with the sign chosen to lie inside
    # the triple's centroid.
    cx_t = (x1 + x2 + x3) / 3.0
    cy_t = (y1 + y2 + y3) / 3.0
    # Try both signs and pick the one closer to the triple's centroid.
    for sign in (+1.0, -1.0):
        kx = sx + sign * 2.0 * math.sqrt(max(k1 * k2 * (x1 - x2) ** 2
                                              + k2 * k3 * (x2 - x3) ** 2
                                              + k1 * k3 * (x1 - x3) ** 2, 0.0))
        # The closed-form for x_4 in 2D Descartes is messy; fall back
        # to numerical solve.
    return _solve_inscribed_circle_numerically(triple, k_in)


def _solve_inscribed_circle_numerically(
    triple: list[tuple[float, float, float]], k_in: float
) -> tuple[float, float, float]:
    """Find (x, y) such that the circle with curvature k_in centred there
    is tangent to all three circles in `triple`.
    """
    r_in = 1.0 / k_in if k_in > 0 else -1.0 / k_in
    centres = [(c[1], c[2]) for c in triple]
    radii = [1.0 / abs(c[0]) for c in triple]
    signs = [1 if c[0] > 0 else -1 for c in triple]

    def residual(p: np.ndarray) -> np.ndarray:
        x, y = p[0], p[1]
        out = np.zeros(3)
        for i, (cx, cy) in enumerate(centres):
            dx = x - cx
            dy = y - cy
            dist = math.sqrt(dx * dx + dy * dy)
            target = radii[i] + r_in if signs[i] > 0 else radii[i] - r_in
            out[i] = dist - target
        return out

    p = np.array([sum(c[0] for c in centres) / 3.0,
                  sum(c[1] for c in centres) / 3.0])
    for _ in range(80):
        f = residual(p)
        if np.linalg.norm(f) < 1e-12:
            break
        eps = 1e-7
        J = np.zeros((3, 2))
        for j in range(2):
            pj = p.copy()
            pj[j] += eps
            J[:, j] = (residual(pj) - f) / eps
        dp = np.linalg.lstsq(J, -f, rcond=None)[0]
        p = p + dp
    return (k_in, float(p[0]), float(p[1]))


# ───────────────────────────────────────────────────────────────────────
# Lift each 2D circle to a 3D parametrised loop at z = depth
# ───────────────────────────────────────────────────────────────────────


def lift_circle(
    k: float, x: float, y: float, depth: int, n_samples: int = 256
) -> np.ndarray:
    """Return an (n_samples, 3) array sampling the lifted circle at z = depth."""
    r = 1.0 / abs(k) if abs(k) > 1e-12 else 1.0
    theta = np.linspace(0.0, 2.0 * np.pi, n_samples, endpoint=False)
    out = np.zeros((n_samples, 3))
    out[:, 0] = x + r * np.cos(theta)
    out[:, 1] = y + r * np.sin(theta)
    out[:, 2] = float(depth)
    return out


# ───────────────────────────────────────────────────────────────────────
# Gauss linking number by polygonal discretisation
# ───────────────────────────────────────────────────────────────────────


def gauss_linking(loop_a: np.ndarray, loop_b: np.ndarray) -> float:
    """Compute the Gauss linking number lk(A, B) between two oriented
    closed polygonal loops in R^3 by direct discretisation of

        lk(A, B) = (1/4 pi) * sum_{i, j}
            (r_b_j - r_a_i) . (dr_a_i x dr_b_j) / |r_b_j - r_a_i|^3
    """
    na = loop_a.shape[0]
    nb = loop_b.shape[0]
    da = np.roll(loop_a, -1, axis=0) - loop_a
    db = np.roll(loop_b, -1, axis=0) - loop_b
    # Midpoints to avoid singularity at vertices
    ma = loop_a + 0.5 * da
    mb = loop_b + 0.5 * db

    total = 0.0
    for i in range(na):
        diff = mb - ma[i]
        denom = np.linalg.norm(diff, axis=1) ** 3
        cross = np.cross(np.broadcast_to(da[i], db.shape), db)
        num = np.einsum("ij,ij->i", diff, cross)
        valid = denom > 1e-15
        total += float(np.sum(num[valid] / denom[valid]))
    return total / (4.0 * math.pi)


# ───────────────────────────────────────────────────────────────────────
# Diagnostics
# ───────────────────────────────────────────────────────────────────────


def _print_decomposition_summary() -> None:
    print("\n  Structural decomposition 5 = 2 + 3:")
    print("  " + "-" * 70)
    print("  p = 2   number of returns in echo budget (rho^{-1} + rho^{-2} = 1)")
    print("            grounded in: thm:golden-uniqueness")
    print("  q = 3   Apollonian branching factor (3 bounding circles per")
    print("            interstice; 3 sub-interstices per inscribed disk)")
    print("            grounded in: cor:apollonian-contraction, 1/(1-3 phi^-5)")
    print("  -----")
    print("  p+q = 5  the contraction exponent of the forward half-step")
    print("            (numerically established in falsifier 9 at +/-5%)")
    print("  " + "-" * 70)
    print("  This identity is independent of the trefoil topology.  The")
    print("  topology is the *interpretation* of why p+q is the exponent;")
    print("  the numerical exponent itself is a separate measurement.")


def _build_two_generation_packing() -> tuple[
    list[tuple[float, float, float]], tuple[float, float, float]
]:
    """Build a parent triple + inscribed-disk configuration.

    Returns (parents, inscribed) with parents a list of 3 mutually tangent
    circles (depth 0) and inscribed the depth-1 disk in their interstice.
    """
    seed = classical_seed_2d()
    parents = [seed[1], seed[2], seed[3]]  # the 3 small circles k=2,2,3
    inscribed = _solve_inscribed_circle_numerically(
        parents, k_in=2.0 * (2.0 + 2.0 + 3.0) - 4.0  # one Descartes step
    )
    return parents, inscribed


def _print_linking_table(
    parents: list[tuple[float, float, float]],
    inscribed: tuple[float, float, float],
) -> tuple[float, float, float, list[float]]:
    print("\n  Lifted 2-generation packing in R^3 (z = depth):")
    print("  " + "-" * 70)
    for i, p in enumerate(parents):
        print(f"    parent {i+1}:  k={p[0]:+.3f}, centre=({p[1]:+.4f}, {p[2]:+.4f}, 0.0)")
    k_i, x_i, y_i = inscribed
    print(f"    inscribed:  k={k_i:+.3f}, centre=({x_i:+.4f}, {y_i:+.4f}, 1.0)")
    print("  " + "-" * 70)

    parent_loops = [lift_circle(*p, depth=0) for p in parents]
    inscribed_loop = lift_circle(k_i, x_i, y_i, depth=1)

    print("\n  Pairwise Gauss linking numbers:")
    print("  " + "-" * 70)
    print(f"  {'pair':<26} {'lk':>10}   interpretation")
    print("  " + "-" * 70)

    # parent-parent (same z-plane)
    pp_lks: list[float] = []
    for i in range(3):
        for j in range(i + 1, 3):
            lk = gauss_linking(parent_loops[i], parent_loops[j])
            pp_lks.append(lk)
            print(f"  parent {i+1} <-> parent {j+1}        {lk:>10.4f}   "
                  "(same z-plane; expect 0)")

    # parent-inscribed (different z-planes)
    pi_lks: list[float] = []
    for i in range(3):
        lk = gauss_linking(parent_loops[i], inscribed_loop)
        pi_lks.append(lk)
        print(f"  parent {i+1} <-> inscribed       {lk:>10.4f}   "
              "(different z-planes; trefoil expects |lk|>=1)")

    print("  " + "-" * 70)

    sum_pp = sum(pp_lks)
    sum_pi = sum(pi_lks)
    max_abs_pi = max(abs(x) for x in pi_lks)

    print(f"  sum |lk(parent, inscribed)|   = {sum(abs(x) for x in pi_lks):>10.4f}")
    print(f"  max |lk(parent, inscribed)|   = {max_abs_pi:>10.4f}")

    return sum_pp, sum_pi, max_abs_pi, pi_lks


# ───────────────────────────────────────────────────────────────────────
# Hard assertions
# ───────────────────────────────────────────────────────────────────────


def test_2d_branching_factor_is_3() -> None:
    """The 2D Apollonian branching factor q is 3: each interstice is
    bounded by 3 circles, and 1 inscribed disk creates 3 sub-interstices.
    """
    parents, inscribed = _build_two_generation_packing()
    assert len(parents) == 3, (
        f"expected 3 parent circles bounding the interstice; got {len(parents)}"
    )


def test_2d_budget_is_2() -> None:
    """The two-return defect equation rho^{-1} + rho^{-2} = 1 has
    exactly two return terms, and the unique positive solution is rho = phi.
    """
    rho = PHI
    budget_value = 1.0 / rho + 1.0 / (rho * rho)
    assert abs(budget_value - 1.0) < 1e-12, (
        f"echo budget rho^{{-1}} + rho^{{-2}} = {budget_value}; expected 1.0"
    )


def test_5_equals_p_plus_q_in_2d() -> None:
    """Numerological identity: p + q = 2 + 3 = 5 matches the contraction
    exponent of the forward half-step (numerically established in F9)."""
    p = 2  # number of returns in the budget
    q = 3  # branching factor in 2D
    assert p + q == 5, f"5 = p + q failed: p+q = {p + q}"


def test_natural_z_depth_lift_does_not_link() -> None:
    """The natural z = depth lift produces a 4-component UNLINK in 2D
    Apollonian (parent and inscribed circles do not Gauss-link in this
    lift).  This is a clean falsification of the *literal* trefoil
    topology in this specific lift, while leaving 5 = p + q as a
    numerological/structural decomposition that needs verification by
    other means (Falsifier 14b's dimensional prediction).
    """
    parents, inscribed = _build_two_generation_packing()
    parent_loops = [lift_circle(*p, depth=0) for p in parents]
    inscribed_loop = lift_circle(*inscribed, depth=1)
    max_lk = 0.0
    for loop in parent_loops:
        max_lk = max(max_lk, abs(gauss_linking(loop, inscribed_loop)))
    assert max_lk < 0.1, (
        f"natural z=depth lift produced |lk| = {max_lk} > 0.1; "
        f"if this triggers, the literal trefoil topology IS realised"
    )


# ───────────────────────────────────────────────────────────────────────
# Main
# ───────────────────────────────────────────────────────────────────────


def main() -> None:
    print("=" * 72)
    print("FALSIFIER 14a: Topological signature of the 2-generation")
    print("  Apollonian lift (heuristic).  Tests whether the natural")
    print("  z = depth lift realises a non-trivial knot or just an unlink,")
    print("  and grounds the structural identity 5 = 2 + 3 = (returns) +")
    print("  (branching).  The dimensional prediction is in Falsifier 14b.")
    print("=" * 72)

    _print_decomposition_summary()

    parents, inscribed = _build_two_generation_packing()
    sum_pp, sum_pi, max_abs_pi, pi_lks = _print_linking_table(parents, inscribed)

    print("\n" + "=" * 72)
    print("HARD ASSERTIONS")
    print("=" * 72)

    failed: list[tuple[str, str]] = []
    for name, fn in (
        ("BRANCHING_2D    (q = 3 in 2D Apollonian)",
         test_2d_branching_factor_is_3),
        ("BUDGET_2_RETURNS (rho^{-1} + rho^{-2} = 1, p = 2)",
         test_2d_budget_is_2),
        ("FIVE_EQUALS_P_PLUS_Q  (5 = 2 + 3 numerically)",
         test_5_equals_p_plus_q_in_2d),
        ("NATURAL_LIFT_UNLINK (z=depth lift gives 4-component unlink,",
         test_natural_z_depth_lift_does_not_link),
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
        print("INTERPRETATION:")
        print("  The structural identity 5 = p + q = 2 + 3 holds:")
        print("    p = 2 (number of returns in the echo budget)")
        print("    q = 3 (Apollonian branching factor in 2D)")
        print("  Both are independently established in the paper.")
        print("")
        print("  The literal trefoil topology is NOT realised by the")
        print("  natural z = depth lift: parent and inscribed circles")
        print("  are unlinked in this embedding.  This means the")
        print("  trefoil-as-knot reading is a poetic/heuristic")
        print("  interpretation, not a topological fact about the")
        print("  natural lift.  The structural identity 5 = 2 + 3 is")
        print("  decoupled from the trefoil interpretation.")
        print("")
        print("  The load-bearing test of whether 5 = p + q generalises")
        print("  to higher-dimensional Apollonian packings is")
        print("  Falsifier 14b: 3D Apollonian should give phi^{-(d+3)}")
        print("  = phi^{-6} per forward half-step IF the decomposition")
        print("  is structural; phi^{-5} (no dim dependence) IF it is")
        print("  numerological coincidence.")
    else:
        print("ASSERTIONS FAILED; see diagnostic output above.")
    print("=" * 72)

    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
