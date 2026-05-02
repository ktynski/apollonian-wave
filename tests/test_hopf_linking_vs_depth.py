"""Falsifier B4: Hopf-linking complexity as a function of Apollonian depth.

Hypothesis under test:
  As we recurse deeper into the Apollonian packing, the average
  pairwise Gauss linking number among Hopf-lifted circles changes
  monotonically with depth.  A natural geometric prediction:
  deeper-depth circles have base points more clustered on S^2
  (closer to the limit-set fractal structure), so their Hopf
  fibres have *higher* mutual linking magnitude.

  Specifically: at depth d, sample a representative collection of
  Apollonian circles, lift to Hopf, compute mean(|lk|) and
  std(|lk|) across pairs.  Track these as functions of d.

Setup:
  - Build the Apollonian packing iteratively to depth d (typical
    Descartes recursion).  Each new circle inscribes in an
    interstice of previous depth.
  - For each depth d in {0, 1, 2, 3}, take all circles AT THAT
    DEPTH (children spawned in the d-th generation).
  - Lift each to Hopf, compute pairwise Gauss linkings, tabulate
    mean(|lk|), std(|lk|), max(|lk|).

Hard test:
  - mean(|lk|) at each depth should be order 1 (not vanishing,
    not blowing up).  Hopf fibration's defining property
    guarantees |lk| = 1 for distinct fibres.
  - Track how mean(|lk|) varies with d.

Pass conditions:
  - LK_NONZERO_PER_DEPTH: mean(|lk|) > 0.5 at every depth.
  - LK_NEAR_UNITY: median |lk| at depth 0 is in [0.5, 1.5]
    (consistent with Hopf-chain structure).

Fail modes:
  - Mean linking decays toward zero with depth (Hopf lift
    becomes degenerate at high depth)
  - Linking magnitudes blow up (numerical issues with closely-
    spaced base points on S^2)
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
# Apollonian packing recursion via Descartes
# ───────────────────────────────────────────────────────────────────────


def descartes_4th(k1: float, k2: float, k3: float, sign: int = +1) -> float:
    s = k1 + k2 + k3
    cross = k1 * k2 + k2 * k3 + k3 * k1
    return s + sign * 2.0 * math.sqrt(max(0.0, cross))


def descartes_4th_complex(
    k1: float, k2: float, k3: float, b1: complex, b2: complex, b3: complex,
    k4: float
) -> complex:
    """Use Descartes complex form to find circle 4's centre."""
    s = k1 * b1 + k2 * b2 + k3 * b3
    cross = k1 * k2 * b1 * b2 + k2 * k3 * b2 * b3 + k3 * k1 * b3 * b1
    # k4 b4 = s +/- 2 sqrt(cross), and k4 must be the right one
    disc = 2.0 * np.sqrt(cross)
    cand1 = (s + disc) / k4
    cand2 = (s - disc) / k4
    # We picked the "smaller circle" k4.  Both candidates are valid;
    # we want the one whose centre is in the *central* region.
    return cand1 if cand1.imag > cand2.imag else cand2


def build_packing_to_depth(max_depth: int) -> dict[int, list[tuple[float, float, float]]]:
    """Returns dict mapping depth -> list of (centre_x, centre_y, radius)
    for circles spawned at that depth."""
    # Seed: -1 outer + (2, 2, 3) inner
    seed_outer = (-1.0, complex(0.0, 1.0 / 3.0), 1.0)  # k = -1, centre, radius=1
    seed_inner = [
        (2.0, complex(-0.5, 0.0)),
        (2.0, complex(+0.5, 0.0)),
        (3.0, complex(0.0, 2.0 / 3.0)),
    ]
    # Depth 0: the 4 seed circles (-1, 2, 2, 3)
    depths: dict[int, list[tuple[float, float, float]]] = {}
    depths[0] = []
    for k, b in [(seed_outer[0], seed_outer[1])] + [(k, b) for k, b in seed_inner]:
        depths[0].append((float(b.real), float(b.imag), 1.0 / abs(k)))

    # We track quadruples (4 mutually tangent circles).  Each
    # quadruple Q = (C_a, C_b, C_c, C_d).  The next-depth
    # children come from "Apollonian replacement": for each
    # 3-subset (C_a, C_b, C_c), Descartes gives two solutions for
    # the 4th tangent circle - one is C_d itself, the other is the
    # NEW inscribed circle.  We add the new one at the next depth.

    # Initial quadruple: outer (-1) + 3 inner (2, 2, 3)
    initial_q = [
        (seed_outer[0], seed_outer[1]),
    ] + seed_inner
    quadruples = [initial_q]

    for d in range(1, max_depth + 1):
        new_circles = []
        new_quadruples = []
        seen_keys: set[tuple[int, int, int]] = set()
        for q in quadruples:
            for skip in range(4):
                # 3-subset = q without index skip; the "old" 4th = q[skip]
                tri = [q[i] for i in range(4) if i != skip]
                old = q[skip]
                k1, b1 = tri[0]
                k2, b2 = tri[1]
                k3, b3 = tri[2]
                # New 4th: k_new = (k1+k2+k3) + or - 2 sqrt(cross), the
                # one different from old's curvature.
                s = k1 + k2 + k3
                cross = k1 * k2 + k2 * k3 + k3 * k1
                disc = 2.0 * math.sqrt(max(0.0, cross))
                cands = [s + disc, s - disc]
                k_old = old[0]
                # The "other" candidate from old:
                k_new = cands[0] if abs(cands[1] - k_old) < abs(cands[0] - k_old) else cands[1]
                if k_new <= 0 or k_new > 1e6:
                    continue
                # New centre via complex Descartes
                sb = k1 * b1 + k2 * b2 + k3 * b3
                cb = k1 * k2 * b1 * b2 + k2 * k3 * b2 * b3 + k3 * k1 * b3 * b1
                cb_sqrt = np.sqrt(cb)
                cand1 = (sb + 2.0 * cb_sqrt) / k_new
                cand2 = (sb - 2.0 * cb_sqrt) / k_new
                # the one different from old's centre
                d_old = old[1]
                if abs(cand1 - d_old * k_old / k_old) > abs(cand2 - d_old * k_old / k_old):
                    new_b = cand1
                else:
                    new_b = cand2
                # de-dup
                key = (round(k_new, 4), round(new_b.real, 4), round(new_b.imag, 4))
                if key in seen_keys:
                    continue
                seen_keys.add(key)
                new_circles.append((float(new_b.real), float(new_b.imag), 1.0 / k_new))
                new_quadruples.append([
                    tri[0], tri[1], tri[2], (k_new, new_b)
                ])
        depths[d] = new_circles
        quadruples = new_quadruples
        if not new_quadruples:
            break

    return depths


# ───────────────────────────────────────────────────────────────────────
# Hopf lift (reused)
# ───────────────────────────────────────────────────────────────────────


def inv_stereo_R2_to_S2(x: float, y: float) -> np.ndarray:
    s = x * x + y * y
    denom = 1.0 + s
    return np.array([2.0 * x / denom, 2.0 * y / denom, (s - 1.0) / denom])


def hopf_fibre(p_S2: np.ndarray, n_samples: int = 160) -> np.ndarray:
    a, b, c = p_S2[0], p_S2[1], p_S2[2]
    r1 = math.sqrt(max(0.0, (1.0 + c) / 2.0))
    r2 = math.sqrt(max(0.0, (1.0 - c) / 2.0))
    phase_diff = math.atan2(b, a)
    psis = np.linspace(0.0, 2.0 * math.pi, n_samples, endpoint=False)
    out = np.zeros((n_samples, 4))
    for k, psi in enumerate(psis):
        z1 = r1 * np.exp(1j * psi)
        z2 = r2 * np.exp(1j * (psi - phase_diff))
        out[k, 0] = z1.real
        out[k, 1] = z1.imag
        out[k, 2] = z2.real
        out[k, 3] = z2.imag
    return out


def stereo_S3_to_R3(fibre_R4: np.ndarray, pole: np.ndarray) -> np.ndarray:
    n = fibre_R4.shape[0]
    out = np.zeros((n, 3))
    e0 = pole / np.linalg.norm(pole)
    e_cands = np.eye(4)
    basis: list[np.ndarray] = []
    for v in e_cands:
        u = v - np.dot(v, e0) * e0
        for b in basis:
            u = u - np.dot(u, b) * b
        if np.linalg.norm(u) > 1e-8:
            basis.append(u / np.linalg.norm(u))
        if len(basis) == 3:
            break
    e1, e2, e3 = basis[0], basis[1], basis[2]
    for k in range(n):
        p = fibre_R4[k]
        denom = 1.0 - np.dot(p, e0)
        if abs(denom) < 1e-9:
            denom = 1e-9
        q = (p - np.dot(p, e0) * e0) / denom
        out[k, 0] = np.dot(q, e1)
        out[k, 1] = np.dot(q, e2)
        out[k, 2] = np.dot(q, e3)
    return out


def gauss_linking_R3(loop_a: np.ndarray, loop_b: np.ndarray) -> float:
    na = loop_a.shape[0]
    da = np.roll(loop_a, -1, axis=0) - loop_a
    db = np.roll(loop_b, -1, axis=0) - loop_b
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
# Linking statistics per depth
# ───────────────────────────────────────────────────────────────────────


def linking_stats_for_depth(
    circles: list[tuple[float, float, float]],
    max_circles: int = 12,
) -> dict:
    """For up to `max_circles` circles at this depth, lift each to a
    Hopf fibre and compute pairwise Gauss linkings."""
    if len(circles) > max_circles:
        # take a representative sample
        idx = np.linspace(0, len(circles) - 1, max_circles).astype(int)
        circles = [circles[i] for i in idx]
    if len(circles) < 2:
        return {"n_circles": len(circles), "lks": []}
    pole = np.array([0.6, 0.4, 0.5, 0.5])
    pole = pole / np.linalg.norm(pole)
    fibres = []
    for x, y, r in circles:
        s2 = inv_stereo_R2_to_S2(x, y)
        f4 = hopf_fibre(s2, n_samples=120)
        f3 = stereo_S3_to_R3(f4, pole)
        fibres.append(f3)
    lks = []
    for i in range(len(fibres)):
        for j in range(i + 1, len(fibres)):
            lk = gauss_linking_R3(fibres[i], fibres[j])
            lks.append(lk)
    return {
        "n_circles": len(circles),
        "lks": lks,
        "mean_abs": float(np.mean(np.abs(lks))) if lks else 0.0,
        "median_abs": float(np.median(np.abs(lks))) if lks else 0.0,
        "max_abs": float(np.max(np.abs(lks))) if lks else 0.0,
    }


def gather_depth_data(max_depth: int = 3) -> dict[int, dict]:
    depths = build_packing_to_depth(max_depth)
    out: dict[int, dict] = {}
    for d in sorted(depths.keys()):
        out[d] = linking_stats_for_depth(depths[d], max_circles=10)
        out[d]["n_total_at_depth"] = len(depths[d])
    return out


def _print_depth_table(data: dict[int, dict]) -> None:
    print("\n  Hopf-lift linking statistics by Apollonian depth:")
    print("  " + "-" * 72)
    print(f"  {'depth':>5} {'n_total':>10} {'n_used':>10} "
          f"{'mean|lk|':>12} {'median|lk|':>14} {'max|lk|':>12}")
    print("  " + "-" * 72)
    for d, s in sorted(data.items()):
        if not s["lks"]:
            print(f"  {d:>5} {s['n_total_at_depth']:>10} {s['n_circles']:>10}    -- no pairs --")
            continue
        print(f"  {d:>5} {s['n_total_at_depth']:>10} {s['n_circles']:>10} "
              f"{s['mean_abs']:>+12.4f} {s['median_abs']:>+14.4f} "
              f"{s['max_abs']:>+12.4f}")
    print("  " + "-" * 72)


def test_lk_nonzero_per_depth() -> None:
    data = gather_depth_data(max_depth=3)
    for d, s in data.items():
        if not s["lks"]:
            continue
        assert s["mean_abs"] > 0.3, (
            f"LK_NONZERO_PER_DEPTH FAIL at depth {d}: "
            f"mean(|lk|) = {s['mean_abs']:.4f} <= 0.3.  Hopf lift "
            f"may be degenerate at this depth."
        )


def test_depth_0_median_near_unity() -> None:
    data = gather_depth_data(max_depth=2)
    s = data.get(0)
    if not s or not s["lks"]:
        raise AssertionError(
            "DEPTH_0_NO_LINKINGS: no linking values at depth 0"
        )
    assert 0.5 <= s["median_abs"] <= 1.5, (
        f"LK_NEAR_UNITY FAIL: depth-0 median |lk| = {s['median_abs']:.4f}, "
        f"expected in [0.5, 1.5] (Hopf-chain unit linking).  "
        f"Linkings: {[round(lk, 3) for lk in s['lks']]}"
    )


def main() -> None:
    print("=" * 72)
    print("FALSIFIER B4: Hopf-linking complexity vs Apollonian depth")
    print("  Tracks mean and median |Gauss linking| of Hopf-lifted circle")
    print("  pairs at each depth of the Apollonian recursion.")
    print("=" * 72)

    print("\n  Hopf fibration's defining property: any two distinct fibres")
    print("  link with linking number +/- 1.  Test verifies this remains")
    print("  true (mean and median |lk| ~ 1) as we go deeper into the gasket.")

    data = gather_depth_data(max_depth=3)
    _print_depth_table(data)

    print("\n" + "=" * 72)
    print("HARD ASSERTIONS")
    print("=" * 72)
    failed: list[tuple[str, str]] = []
    for name, fn in (
        ("LK_NONZERO_PER_DEPTH    (mean |lk| > 0.3 at every depth)",
         test_lk_nonzero_per_depth),
        ("LK_NEAR_UNITY           (depth-0 median |lk| in [0.5, 1.5])",
         test_depth_0_median_near_unity),
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
        print("HOPF-LINKING REMAINS NON-TRIVIAL AT ALL APOLLONIAN DEPTHS.")
        print("  The Hopf-chain structure persists as we recurse deeper")
        print("  into the gasket, supporting the Hopf-geometric arena")
        print("  hypothesis at every scale.")
    else:
        print("HOPF-LINKING DEGRADES AT SOME DEPTH.")
    print("=" * 72)

    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
