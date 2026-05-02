"""Falsifier B2: Knot/link type of the principal closed orbit under Hopf lift.

Hypothesis under test:
  The principal length-3 closed orbit of the Apollonian IFS,
  lifted to S^3 via the standard Hopf fibration (each orbit point
  z_k -> Hopf fibre over the inverse-stereographic image of z_k),
  forms a 3-component link whose pairwise linking matches the
  combinatorial signature of a (2,3) torus knot (trefoil), i.e.
  total pairwise Gauss linking number is +/- 3.

  The original trefoil claim (from external collaborator) read as
  "the orbit traces a (2,3) torus knot in S^3."  In the natural
  Hopf lift, each ORBIT POINT lifts to a great-circle Hopf fibre,
  not to a single point.  The orbit is therefore a 3-component
  link in S^3, not a knot.  But the *total pairwise linking* of
  that link can still match the writhe / crossing-count signature
  of a (2,3) torus knot:

    sum_{i < j} lk(fibre_i, fibre_j) = +/- 3
                                = (2,3) torus-knot writhe

  This is the falsifiable numerical content of the trefoil claim
  in the Hopf arena.

Setup:
  - Reuse the principal closed orbit T_1 T_3 T_2 from A1 (Jacobian
    phi^-6 at fixed point z*).  Compute the orbit {z_0, z_1, z_2}
    with z_0 = z*, z_1 = T_2(z_0), z_2 = T_3(z_1), and
    T_1(z_2) = z_0.
  - Lift each z_k to S^3 as a Hopf fibre (great circle).
  - Stereographic project to R^3 from a generic pole.
  - Compute pairwise Gauss linking numbers and their sum.

Hard test:
  - Verify the 3 fibres are distinct (linkings well-defined).
  - Sum the 3 pairwise linkings; compare to +/- 3.

Pass conditions:
  - DISTINCT_FIBRES: each pair has |lk| > 0.5 (non-trivial linking).
  - TOTAL_LINKING_3: sum_{i<j} lk(fibre_i, fibre_j) is +/- 3 within
    +/- 0.5 (matches torus-knot writhe of trefoil T(2,3)).

Fail modes:
  - some pair has |lk| ~ 0 (orbit point coincides with another in
    Hopf-base; principal orbit doesn't visit 3 distinct points)
  - total linking is +/- 1 or +/- 2 (not the trefoil signature;
    different knot-class topology)
  - total linking is +/- 6 (some other torus knot)

This is the proper rerun of the F14a topological claim under
the Hopf hypothesis, applied to the principal closed orbit
specifically (rather than to the seed parent triple as in B1).
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

PHI = (1.0 + math.sqrt(5.0)) / 2.0


# ───────────────────────────────────────────────────────────────────────
# Apollonian IFS (reused from A1)
# ───────────────────────────────────────────────────────────────────────


CIRCLES: list[tuple[complex, float]] = [
    (complex(-0.5, 0.0), 0.5),
    (complex(+0.5, 0.0), 0.5),
    (complex(0.0, 2.0 / 3.0), 1.0 / 3.0),
]


def invert(z: complex, idx: int) -> complex:
    c, r = CIRCLES[idx]
    diff = z - c
    if abs(diff) < 1e-15:
        return complex(float("inf"), 0.0)
    return c + (r * r) / np.conj(diff)


def find_principal_orbit() -> list[complex]:
    """Find the principal length-3 closed orbit T_1 T_3 T_2 fixed point
    and return the three orbit points {z_0, z_1, z_2}."""
    word = [0, 2, 1]  # T_1 T_3 T_2 (apply T_1 first, then T_3, then T_2)
    # iterate (T_1 T_3 T_2)^2 to find fixed point of squared map
    z = complex(0.0, 0.4)
    for _ in range(4000):
        for idx in word + word:
            z = invert(z, idx)
    z0 = z
    z1 = invert(z0, word[0])
    z2 = invert(z1, word[1])
    z_back = invert(z2, word[2])
    if abs(z_back - z0) > 1e-3:
        # T_1 T_3 T_2 may swap two points; we need a length-6 fixed point,
        # but (T_1 T_3 T_2)^2 fix should give consistent orbit
        z = complex(0.05, 0.45)
        for _ in range(4000):
            for idx in (word + word):
                z = invert(z, idx)
        z0 = z
        z1 = invert(z0, word[0])
        z2 = invert(z1, word[1])
    return [z0, z1, z2]


# ───────────────────────────────────────────────────────────────────────
# Hopf lift (reused from B1)
# ───────────────────────────────────────────────────────────────────────


def inv_stereo_R2_to_S2(z: complex) -> np.ndarray:
    x, y = z.real, z.imag
    s = x * x + y * y
    denom = 1.0 + s
    return np.array([2.0 * x / denom, 2.0 * y / denom, (s - 1.0) / denom])


def hopf_fibre(p_S2: np.ndarray, n_samples: int = 256) -> np.ndarray:
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
# Diagnostics
# ───────────────────────────────────────────────────────────────────────


def gather_principal_link() -> dict:
    orbit = find_principal_orbit()
    s2_pts = [inv_stereo_R2_to_S2(z) for z in orbit]
    fibres_R4 = [hopf_fibre(p, n_samples=256) for p in s2_pts]
    pole = np.array([0.6, 0.4, 0.5, 0.5])
    pole = pole / np.linalg.norm(pole)
    fibres_R3 = [stereo_S3_to_R3(f, pole) for f in fibres_R4]
    pair_linkings = []
    for i in range(3):
        for j in range(i + 1, 3):
            lk = gauss_linking_R3(fibres_R3[i], fibres_R3[j])
            pair_linkings.append((i, j, lk))
    total = sum(lk for _, _, lk in pair_linkings)
    return {
        "orbit_2D": orbit,
        "S2_points": s2_pts,
        "pair_linkings": pair_linkings,
        "total_linking": total,
    }


def _print_diagnostic(data: dict) -> None:
    print("\n  Principal length-3 orbit T_1 T_3 T_2 on the gasket limit set:")
    print("  " + "-" * 64)
    for k, z in enumerate(data["orbit_2D"]):
        print(f"    z_{k} = {z.real:+.6f} {z.imag:+.6f}j")
    print("\n  Inverse stereographic images on S^2:")
    print("  " + "-" * 64)
    for k, p in enumerate(data["S2_points"]):
        print(f"    pi^-1(z_{k}) = ({p[0]:+.4f}, {p[1]:+.4f}, {p[2]:+.4f})")
    print("\n  Pairwise Gauss linkings of the 3 Hopf fibres in R^3:")
    print("  " + "-" * 64)
    for i, j, lk in data["pair_linkings"]:
        print(f"    lk(fibre_{i}, fibre_{j}) = {lk:+.4f}")
    print("  " + "-" * 64)
    print(f"  TOTAL = sum_{{i<j}} lk(fibre_i, fibre_j) = {data['total_linking']:+.4f}")
    print(f"  Trefoil T(2,3) writhe (signature) = +/- 3")


def test_distinct_fibres() -> None:
    data = gather_principal_link()
    for i, j, lk in data["pair_linkings"]:
        assert abs(lk) > 0.5, (
            f"DISTINCT_FIBRES FAIL: lk(fibre_{i}, fibre_{j}) = {lk:+.4f}; "
            f"|lk| <= 0.5 means the two orbit points may project to the "
            f"same Hopf base or the linking is degenerate"
        )


def test_total_linking_matches_trefoil_writhe() -> None:
    data = gather_principal_link()
    total = data["total_linking"]
    assert 2.5 <= abs(total) <= 3.5, (
        f"TOTAL_LINKING_3 FAIL: total pairwise linking = {total:+.4f}, "
        f"expected magnitude 3 (trefoil writhe).  Pairwise linkings: "
        f"{[(i, j, round(lk, 4)) for i, j, lk in data['pair_linkings']]}"
    )


def main() -> None:
    print("=" * 70)
    print("FALSIFIER B2: Principal closed-orbit knot/link type under Hopf lift")
    print("  Tests whether the 3 Hopf fibres of the principal length-3")
    print("  Apollonian orbit form a link whose total pairwise Gauss")
    print("  linking matches the (2,3) torus-knot writhe (+/- 3).")
    print("=" * 70)

    data = gather_principal_link()
    _print_diagnostic(data)

    print("\n" + "=" * 70)
    print("HARD ASSERTIONS")
    print("=" * 70)
    failed: list[tuple[str, str]] = []
    for name, fn in (
        ("DISTINCT_FIBRES        (every pair has |lk| > 0.5)",
         test_distinct_fibres),
        ("TOTAL_LINKING_3        (sum of pairwise linkings = +/- 3)",
         test_total_linking_matches_trefoil_writhe),
    ):
        try:
            fn()
            print(f"  PASS: {name}")
        except AssertionError as exc:
            print(f"  FAIL: {name}")
            print(f"        {exc}")
            failed.append((name, str(exc)))

    print("\n" + "=" * 70)
    if not failed:
        print("PRINCIPAL ORBIT'S HOPF LIFT MATCHES TREFOIL SIGNATURE.")
        print("  The 3 Hopf fibres of the principal closed orbit have")
        print("  pairwise linking summing to +/- 3, matching the writhe")
        print("  of the (2,3) torus knot.  This is the topological")
        print("  signature the trefoil interpretation predicted, realised")
        print("  in the proper Hopf arena.")
    else:
        print("HOPF-LIFTED PRINCIPAL ORBIT DOES NOT MATCH TREFOIL SIGNATURE.")
        print("  See diagnostic output for the actual linking values.")
    print("=" * 70)

    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
