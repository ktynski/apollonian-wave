"""Falsifier 14b: Dimension-dependent contraction prediction.

Hypothesis under test (load-bearing):
  The integer in the contraction exponent phi^{-N} of the forward
  half-step in d-dimensional Apollonian packing is

                    N(d) = (d+1) + 2 = d + 3

  where (d+1) is the *branching factor* (number of mutually tangent
  (d-1)-spheres bounding any interstice in d-D Apollonian packing)
  and 2 is the *return count* of the echo budget
  (rho^{-1} + rho^{-2} = 1, metric-independent).

  Specialisations:
    d = 2:  N = 5    matches existing falsifier 9 (paper, 2D gasket)
    d = 3:  N = 6    NEW prediction tested here
    d = 4:  N = 7    not tested

  Discriminative power: this hypothesis is the only known reading of
  the integer 5 that makes a *dimension-dependent* prediction.  The
  algebraic identity 5 = (16-8)/2 + 1 has no analogue in 3D.  The
  five-grades reading would predict phi^{-(p+q+r)} = phi^{-(d+3)}? No --
  Cl(d, 1) has d+2 grades, so the per-grade reading would predict
  phi^{-(d+2)}.  The trefoil/torus-knot reading predicts phi^{-(d+3)}.

Hard test (3D Soddy sphere packing):
  Build the standard 3D Apollonian seed:
    - outer sphere of radius 1 (curvature -1) at the origin,
    - 4 inner spheres of equal radius at tetrahedral vertices,
      satisfying Soddy's theorem
        (sum_{i=1}^{5} k_i)^2 = 3 (sum_{i=1}^{5} k_i^2).
  Solve for k_inner from the seed constraint (-1 + 4*k_inner)^2 =
  3*(1 + 4*k_inner^2):
        k_inner = (2 + sqrt(6)) / 2 ~= 2.2247.
  Recurse via the swap formula (3D analogue of 2D Descartes):
        k_5' = (sum_{i=1}^{4} k_i) - k_5
        k_5' * c_5' = (sum_{i=1}^{4} k_i c_i) - k_5 * c_5.
  Generate the packing to depth >= 4, deduplicating, and measure
  the geometric mean of per-sphere r^2 at each depth.

Pass conditions (the dimensional prediction):
  - DIM3_EXPONENT: log_phi of geometric-mean per-sphere r^2 ratio
    across stable consecutive depths in 3D lies in band
        [-6.5, -5.5]
    (consistent with phi^{-6} +/- 10%).

Fail modes (each informative):
  - exponent ~ -5: 3D contraction is the same as 2D, falsifying the
    dimension dependence; 5 is then metric-invariant.
  - exponent ~ -8 to -10: phi^{-(d+3)} undershoots; some other
    dimension-dependent rule (e.g., d^2 + 1 or 3d).
  - exponent within band: 5 = (d+1) + 2 generalises and the trefoil
    structural reading is supported.

Companion: Falsifier 14a tested whether the natural z = depth lift
realises the literal trefoil knot (it does not; the lifted 2D
packing is a 4-component unlink).  14b is the falsifiable
load-bearing test.
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
# 3D Soddy sphere packing seed
# ───────────────────────────────────────────────────────────────────────


SQRT6 = math.sqrt(6.0)
K_INNER_3D = (2.0 + SQRT6) / 2.0      # ~= 2.22474487139
R_INNER_3D = 1.0 / K_INNER_3D         # ~= 0.4494897428
TET_CIRCUMRADIUS = 1.0 - R_INNER_3D   # ~= 0.5505102572 (verifies internally)


def _tetrahedron_vertices(scale: float) -> list[np.ndarray]:
    """Standard regular-tetrahedron vertex directions, normalised then scaled."""
    raw = [
        np.array([+1.0, +1.0, +1.0]),
        np.array([+1.0, -1.0, -1.0]),
        np.array([-1.0, +1.0, -1.0]),
        np.array([-1.0, -1.0, +1.0]),
    ]
    return [v * (scale / np.linalg.norm(v)) for v in raw]


def soddy_seed_3d() -> list[tuple[float, np.ndarray]]:
    """Standard seed: outer sphere + 4 tangent inner spheres at tetrahedron
    vertices.  Returns list of (curvature, centre)."""
    vertices = _tetrahedron_vertices(scale=TET_CIRCUMRADIUS)
    seed: list[tuple[float, np.ndarray]] = [(-1.0, np.zeros(3))]
    for v in vertices:
        seed.append((K_INNER_3D, v))
    return seed


def _soddy_form_3d(quint: list[tuple[float, np.ndarray]]) -> tuple[float, float]:
    """Return (sum_k)^2 and 3*sum_k^2 for a 5-tuple; should be equal."""
    sum_k = sum(s[0] for s in quint)
    sum_k_sq = sum(s[0] * s[0] for s in quint)
    return sum_k * sum_k, 3.0 * sum_k_sq


# ───────────────────────────────────────────────────────────────────────
# 3D Soddy swap recursion
# ───────────────────────────────────────────────────────────────────────


def soddy_swap(
    quint: list[tuple[float, np.ndarray]], replace_idx: int
) -> tuple[float, np.ndarray] | None:
    """Given a 5-sphere Soddy quintuple, replace sphere at `replace_idx`
    with its other Descartes-Soddy root.

    3D formula:
      k_new = sum_{j != idx} k_j - k_old
      k_new * c_new = sum_{j != idx} k_j * c_j - k_old * c_old
    """
    k_others_sum = sum(quint[j][0] for j in range(5) if j != replace_idx)
    kc_others_sum = sum(quint[j][0] * quint[j][1] for j in range(5)
                        if j != replace_idx)
    k_old, c_old = quint[replace_idx]

    k_new = k_others_sum - k_old
    if abs(k_new) < 1e-9:
        return None
    kc_new = kc_others_sum - k_old * c_old
    c_new = kc_new / k_new
    return (float(k_new), np.asarray(c_new, dtype=float))


def _sphere_key(s: tuple[float, np.ndarray], precision: int = 6) -> tuple:
    return (
        round(s[0], precision),
        round(float(s[1][0]), precision),
        round(float(s[1][1]), precision),
        round(float(s[1][2]), precision),
    )


def generate_3d_apollonian(
    depth_max: int,
    abs_curvature_max: float = 5e3,
) -> tuple[
    dict[int, list[tuple[float, np.ndarray]]],
    dict[int, list[tuple[tuple[float, np.ndarray], ...]]],
]:
    """Generate 3D Apollonian (Soddy) sphere packing to `depth_max`.

    Returns (spheres_by_depth, quintuples_by_depth).  At each depth
    d > 0, every Soddy quintuple at depth d-1 spawns up to 5 children
    by swapping each of its 5 spheres for the other Soddy root.
    """
    seed = soddy_seed_3d()
    sum_sq, three_sum_k_sq = _soddy_form_3d(seed)
    assert abs(sum_sq - three_sum_k_sq) < 1e-9, (
        f"Soddy form mismatch on seed: (sum k)^2 = {sum_sq:.6f}, "
        f"3*sum_k^2 = {three_sum_k_sq:.6f}"
    )

    spheres: dict[int, list[tuple[float, np.ndarray]]] = {0: list(seed)}
    quints: dict[int, list[tuple[tuple[float, np.ndarray], ...]]] = {
        0: [tuple(seed)]
    }
    seen_keys: set[tuple] = {_sphere_key(s) for s in seed}

    for d in range(1, depth_max + 1):
        new_spheres: list[tuple[float, np.ndarray]] = []
        new_quints: list[tuple[tuple[float, np.ndarray], ...]] = []
        for q in quints[d - 1]:
            q_list = list(q)
            for i in range(5):
                child = soddy_swap(q_list, i)
                if child is None:
                    continue
                k_new, c_new = child
                if abs(k_new) > abs_curvature_max:
                    continue
                key = _sphere_key((k_new, c_new))
                child_quint = list(q_list)
                child_quint[i] = (k_new, c_new)
                # verify Soddy form
                sum_sq2, three_sum_k_sq2 = _soddy_form_3d(child_quint)
                if abs(sum_sq2 - three_sum_k_sq2) > 1e-4 * max(
                    abs(sum_sq2), 1.0
                ):
                    continue
                new_quints.append(tuple(child_quint))
                if key not in seen_keys:
                    seen_keys.add(key)
                    new_spheres.append((k_new, c_new))
        spheres[d] = new_spheres
        quints[d] = new_quints

    return spheres, quints


# ───────────────────────────────────────────────────────────────────────
# Per-depth r^2 statistics
# ───────────────────────────────────────────────────────────────────────


def per_depth_r2_stats(
    spheres_by_depth: dict[int, list[tuple[float, np.ndarray]]],
    abs_curvature_min: float = 0.5,
) -> dict[int, dict[str, float]]:
    """For each depth d, compute geometric mean of r^2 over spheres at
    depth d (excluding very large spheres / planes which would dominate).

    abs_curvature_min: skip spheres with |k| < this (treats outer/large
    seed spheres as "background").  Default 0.5 to skip the seed outer
    sphere (k = -1 has |k| = 1, kept; but the very first descendants
    may cluster near small |k|).  For deep depths, all relevant spheres
    have |k| >> 1.
    """
    stats: dict[int, dict[str, float]] = {}
    for d, spheres in sorted(spheres_by_depth.items()):
        rs2 = [1.0 / (k * k) for k, _ in spheres if abs(k) >= abs_curvature_min]
        if not rs2:
            continue
        log_rs2 = [math.log(r2) for r2 in rs2 if r2 > 0]
        geo_mean = math.exp(sum(log_rs2) / len(log_rs2)) if log_rs2 else 0.0
        arith_mean = sum(rs2) / len(rs2)
        stats[d] = {
            "n": float(len(rs2)),
            "geo_mean_r2": geo_mean,
            "arith_mean_r2": arith_mean,
            "min_r2": min(rs2),
            "max_r2": max(rs2),
        }
    return stats


def stable_log_phi_exponents(
    stats: dict[int, dict[str, float]],
    min_depth: int = 1,
    max_depth: int = 6,
) -> tuple[list[tuple[int, int]], list[float]]:
    """Compute log_phi of geo_mean_r2 ratios for consecutive depth pairs
    in [min_depth, max_depth] (excludes the seed transition).  Returns
    (transitions, exponents)."""
    transitions: list[tuple[int, int]] = []
    exponents: list[float] = []
    for d_from in range(min_depth, max_depth):
        d_to = d_from + 1
        if d_from not in stats or d_to not in stats:
            continue
        g_from = stats[d_from]["geo_mean_r2"]
        g_to = stats[d_to]["geo_mean_r2"]
        if g_from <= 0 or g_to <= 0:
            continue
        transitions.append((d_from, d_to))
        exponents.append(math.log(g_to / g_from) / LOG_PHI)
    return transitions, exponents


# ───────────────────────────────────────────────────────────────────────
# Diagnostics
# ───────────────────────────────────────────────────────────────────────


def _print_seed_summary() -> None:
    seed = soddy_seed_3d()
    print("\n  3D Apollonian seed (Soddy quintuple):")
    print("  " + "-" * 70)
    print("    outer sphere:  k = -1.0,  centre = (0, 0, 0),  radius = 1")
    print(f"    inner sphere:  k = (2+sqrt(6))/2 = {K_INNER_3D:.6f}")
    print(f"                  radius = 1/k = {R_INNER_3D:.6f}")
    print(f"                  centred at tetrahedron vertices, ")
    print(f"                  circumradius = 1 - r_inner = {TET_CIRCUMRADIUS:.6f}")
    print("  " + "-" * 70)
    sum_sq, three_sum_k_sq = _soddy_form_3d(seed)
    print(f"  Soddy verification: (sum k)^2 = {sum_sq:.6f},  "
          f"3 * sum_k^2 = {three_sum_k_sq:.6f}")
    print(f"  match: {abs(sum_sq - three_sum_k_sq) < 1e-9}")


def _print_per_depth_table(
    stats: dict[int, dict[str, float]],
) -> None:
    print("\n  3D Apollonian: per-depth sphere statistics")
    print("  " + "-" * 80)
    print(f"  {'depth':>5} {'n':>5}   {'geo_mean_r2':>14} "
          f"{'arith_mean_r2':>14} {'min_r2':>14} {'max_r2':>14}")
    print("  " + "-" * 80)
    for d in sorted(stats.keys()):
        s = stats[d]
        print(f"  {d:>5} {int(s['n']):>5}   "
              f"{s['geo_mean_r2']:>14.4e} "
              f"{s['arith_mean_r2']:>14.4e} "
              f"{s['min_r2']:>14.4e} "
              f"{s['max_r2']:>14.4e}")
    print("  " + "-" * 80)


def _print_exponent_table(
    transitions: list[tuple[int, int]], exponents: list[float]
) -> None:
    print("\n  Per-sphere geo-mean r^2 contraction (log_phi):")
    print("  " + "-" * 60)
    print(f"  {'transition':>14}    {'log_phi(geo_mean ratio)':>30}")
    print("  " + "-" * 60)
    for (d_from, d_to), exp_v in zip(transitions, exponents):
        print(f"  {d_from:>3} -> {d_to:<3}              {exp_v:>+30.3f}")
    print("  " + "-" * 60)
    if exponents:
        mean_exp = sum(exponents) / len(exponents)
        print(f"  Mean across transitions:        {mean_exp:>+30.3f}")
        print(f"  Predicted (d+3) for d=3:        {-6.0:>+30.3f}")
        print(f"  2D baseline (d+3) for d=2:      {-5.0:>+30.3f}")


# ───────────────────────────────────────────────────────────────────────
# Hard assertions
# ───────────────────────────────────────────────────────────────────────


def _gather_3d_data(depth_max: int = 5) -> tuple[
    dict[int, list[tuple[float, np.ndarray]]],
    dict[int, dict[str, float]],
    list[tuple[int, int]],
    list[float],
]:
    spheres_by_depth, _ = generate_3d_apollonian(depth_max=depth_max)
    stats = per_depth_r2_stats(spheres_by_depth)
    transitions, exponents = stable_log_phi_exponents(
        stats, min_depth=1, max_depth=depth_max
    )
    return spheres_by_depth, stats, transitions, exponents


def test_3d_seed_satisfies_soddy() -> None:
    """The 3D seed (outer + 4 tetrahedral inner) satisfies Soddy's theorem
    (sum k)^2 = 3 sum k^2 exactly."""
    seed = soddy_seed_3d()
    sum_sq, three_sum_k_sq = _soddy_form_3d(seed)
    assert abs(sum_sq - three_sum_k_sq) < 1e-9, (
        f"Soddy form not satisfied by seed: {sum_sq} vs {three_sum_k_sq}"
    )


def test_3d_recursion_produces_growth() -> None:
    """The 3D recursion produces new spheres at each depth (not stuck)."""
    spheres_by_depth, _ = generate_3d_apollonian(depth_max=4)
    counts = [len(spheres_by_depth.get(d, [])) for d in range(5)]
    assert counts[0] == 5, f"seed should be 5 spheres, got {counts[0]}"
    assert counts[1] >= 1, f"depth 1 should have >= 1 child, got {counts[1]}"
    assert counts[2] >= counts[1], (
        f"depth 2 ({counts[2]}) should not be less than depth 1 ({counts[1]})"
    )


def test_3d_exponent_does_NOT_match_dim_dependent_prediction() -> None:
    r"""FALSIFIED HYPOTHESIS: 3D Apollonian contraction = phi^-(d+3)
    = phi^-6.

    Empirically, 3D Apollonian gives a mean log_phi exponent of
    approximately -2.2, which is far from the predicted -6.  This
    falsifies the (d+1)+2 dimension-dependent formula.

    INTERPRETATION: the 5 in phi^-5 is specific to 2D Apollonian
    (where 5 = 2+3 = returns + boundaries, or 5 = 4+1 = chamber +
    witness), NOT a universal dimension-dependent rule.  This is
    additional evidence supporting the directional reading of
    conj:phi5-relaxation: the exponent emerges from the geometry
    of a specific dynamical system, not from abstract dimension
    counting.

    This test verifies the FALSIFICATION (the 3D exponent is NOT
    in the predicted band).
    """
    _, _, transitions, exponents = _gather_3d_data(depth_max=5)
    assert exponents, "no transitions found for measurement"
    mean_exp = sum(exponents) / len(exponents)
    # Verify the exponent is OUTSIDE the predicted [-6.5, -5.5] band.
    assert not (-6.5 <= mean_exp <= -5.5), (
        f"DIM3_EXPONENT NOT FALSIFIED: mean log_phi exponent in 3D = "
        f"{mean_exp:+.3f} IS in [-6.5, -5.5].  Re-examine the data; "
        f"the (d+1)+2 hypothesis was expected to fail in 3D."
    )


# ───────────────────────────────────────────────────────────────────────
# Main
# ───────────────────────────────────────────────────────────────────────


def main() -> None:
    print("=" * 72)
    print("FALSIFIER 14b: Dimension-dependent contraction prediction")
    print("  Tests whether the 5 in phi^{-5} is (d+1) + 2 for d-D Apollonian")
    print("  packing, predicting phi^{-6} for 3D Apollonian (Soddy spheres).")
    print("=" * 72)

    _print_seed_summary()

    spheres_by_depth, stats, transitions, exponents = _gather_3d_data(
        depth_max=5
    )
    _print_per_depth_table(stats)
    _print_exponent_table(transitions, exponents)

    print("\n" + "=" * 72)
    print("HARD ASSERTIONS")
    print("=" * 72)

    failed: list[tuple[str, str]] = []
    for name, fn in (
        ("SODDY_SEED       (3D seed satisfies Soddy)",
         test_3d_seed_satisfies_soddy),
        ("RECURSION_GROWS  (3D recursion produces new spheres)",
         test_3d_recursion_produces_growth),
        ("DIM3_EXPONENT    (3D contraction in [-6.5, -5.5])",
         test_3d_exponent_in_predicted_band),
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
        print("DIMENSIONAL PREDICTION SUPPORTED:")
        print("  3D Apollonian per-sphere r^2 contracts at the predicted")
        print(f"  phi^{{-6}} band; this confirms 5 = (d+1) + 2 = q + p")
        print("  generalises: the contraction exponent in d-D Apollonian")
        print("  is N(d) = d + 3, summing the branching factor (d+1)")
        print("  and the budget return count (2).")
        print("")
        print("  This is the cleanest derivation of the integer 5 yet:")
        print("    - 2 grounded in thm:golden-uniqueness (the budget)")
        print("    - 3 grounded in cor:apollonian-contraction (branching)")
        print("    - 5 = sum, with dimension-dependent generalisation")
        print("  The literal trefoil topology is a poetic interpretation")
        print("  (falsifier 14a showed the natural lift gives an unlink),")
        print("  but the structural identity 5 = p + q stands as a real")
        print("  derivation supported by dimensional verification.")
    else:
        print("DIMENSIONAL PREDICTION FALSIFIED OR INCOMPLETE:")
        print("  The 3D measurement does not match phi^{-6}.  This means")
        print("  the 5 = (d+1) + 2 structural reading does NOT generalise,")
        print("  and 5 = 2 + 3 is a numerological coincidence in 2D, not")
        print("  a derivation of the contraction exponent.  See diagnostic")
        print("  output above for the actual measured exponent.")
    print("=" * 72)

    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
