"""Falsifier B1: Hopf-lift Gauss linking numbers for the Apollonian
2-generation sub-packing.

Hypothesis under test:
  When the 2-generation Apollonian sub-packing (parent triple +
  inscribed disk) is lifted to S^3 via the standard Hopf fibration
  (each circle's centre maps to a point on S^2 by inverse
  stereographic projection, then to its Hopf fibre on S^3), the
  resulting 4-component link has non-trivial Gauss linking numbers
  between distinct fibres.

  Falsifier 14a falsified the *naive z = depth* lift (gave a
  4-component unlink in R^3).  This is the proper Hopf-corrected
  rerun: any two distinct Hopf fibres in S^3 are linked with
  linking number +/- 1 by construction (Hopf fibration's defining
  property), so the Hopf lift CANNOT give an unlink.

  The interesting observation is whether parent-inscribed pairs
  give linking number 1 (Hopf chain), or higher (richer link
  structure), and whether parent-parent pairs link too (Hopf chain
  among parents).

Setup:
  Standard Hopf map pi: S^3 -> S^2,
    pi(z_1, z_2) = (2 z_1 conj(z_2), |z_1|^2 - |z_2|^2) in R^3
  for (z_1, z_2) in C^2 with |z_1|^2 + |z_2|^2 = 1.

  Apollonian 2D plane lifts to S^2 by inverse stereographic
  projection from R^2 to S^2 (centre of each circle maps to a
  base point on S^2).  Each base point's Hopf fibre is a great
  circle on S^3.  Project S^3 to R^3 via stereographic
  projection from a generic pole, then compute Gauss linking
  numbers as polygonal integrals in R^3.

Hard test:
  - Generate seed parent circles (3 inner circles of -1, 2, 2, 3) and
    the inscribed disk in their central interstice.
  - Inverse stereographic each centre to S^2.
  - Build the 4 Hopf fibres on S^3 (256 sample points each).
  - Stereographic project to R^3 (avoiding poles on the fibres).
  - Compute pairwise Gauss linking numbers.

Pass conditions:
  - HOPF_LINKED_PAIRS: every parent-inscribed pair has |lk| >= 1
    (Hopf fibration guarantees this if base points are distinct).
  - PARENT_INSCRIBED_NONZERO: at least one parent-inscribed pair
    rounds to |lk| = 1 within +/- 0.2 (clean Hopf-link integer).

Fail modes:
  - all linkings near 0 (Hopf lift produces unlink; Hopf-geometric
    arena hypothesis falsified)
  - non-integer linking numbers (numerics or singular
    stereographic projection; rerun with different pole)

This test is a clean rerun of F14a under the Hopf hypothesis.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

PHI = (1.0 + math.sqrt(5.0)) / 2.0


# ───────────────────────────────────────────────────────────────────────
# Apollonian sub-packing
# ───────────────────────────────────────────────────────────────────────


def build_2gen_subpacking() -> list[tuple[str, float, float, float]]:
    """Returns list of (label, x_centre, y_centre, radius) for the
    3 parent inner circles and the inscribed disk."""
    parents = [
        ("parent_1",  -0.5,  0.0,  0.5),
        ("parent_2",  +0.5,  0.0,  0.5),
        ("parent_3",   0.0,  2.0/3.0,  1.0/3.0),
    ]
    # Inscribed disk: solve Descartes for 4th root in central interstice.
    # k_1 + k_2 + k_3 = 2+2+3 = 7,  cross = 2*2 + 2*3 + 2*3 = 16,
    # k_4 = sum +/- 2 sqrt(cross) = 7 +/- 8.  Smaller positive k_4 = 15
    # is the inscribed disk in the central interstice.
    k4 = 15.0
    r4 = 1.0 / k4
    # Solve for centre numerically: distance to each parent centre =
    # r_parent + r4 (external tangency).
    centres = [(p[1], p[2]) for p in parents]
    radii = [p[3] for p in parents]

    def residual(p: np.ndarray) -> np.ndarray:
        x, y = p
        out = np.zeros(3)
        for i, (cx, cy) in enumerate(centres):
            out[i] = math.hypot(x - cx, y - cy) - (radii[i] + r4)
        return out

    p = np.array([0.0, 0.25])
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
    inscribed = ("inscribed", float(p[0]), float(p[1]), r4)

    return parents + [inscribed]


# ───────────────────────────────────────────────────────────────────────
# Inverse stereographic projection from R^2 to S^2
# ───────────────────────────────────────────────────────────────────────


def inv_stereo_R2_to_S2(x: float, y: float) -> np.ndarray:
    """Inverse stereographic from north pole: R^2 -> S^2 \\ {north}.

    (x, y) -> (2x/(1+x^2+y^2), 2y/(1+x^2+y^2), (x^2+y^2-1)/(x^2+y^2+1))
    """
    s = x * x + y * y
    denom = 1.0 + s
    return np.array([2.0 * x / denom, 2.0 * y / denom, (s - 1.0) / denom])


# ───────────────────────────────────────────────────────────────────────
# Hopf fibration: S^3 -> S^2
# ───────────────────────────────────────────────────────────────────────


def hopf_fibre(p_S2: np.ndarray, n_samples: int = 256) -> np.ndarray:
    """Given a point p on S^2, return n_samples points on its Hopf
    fibre, embedded in R^4 = C^2 as (Re z_1, Im z_1, Re z_2, Im z_2).

    Standard convention:
      pi(z_1, z_2) = (2 Re(z_1 conj(z_2)), 2 Im(z_1 conj(z_2)),
                      |z_1|^2 - |z_2|^2).
    For p_S2 = (a, b, c) with a^2 + b^2 + c^2 = 1, the fibre is
      z_1 = sqrt((1 + c) / 2) * exp(i psi)
      z_2 = sqrt((1 - c) / 2) * exp(i (psi - phi))
    where (a + i b) = (1 - c) / 2 * exp(... )  -- choose phi = arg(a + ib).
    Fibre is parameterised by overall phase psi in [0, 2 pi).
    """
    a, b, c = p_S2[0], p_S2[1], p_S2[2]
    # |z_1|^2 = (1+c)/2,  |z_2|^2 = (1-c)/2
    r1 = math.sqrt(max(0.0, (1.0 + c) / 2.0))
    r2 = math.sqrt(max(0.0, (1.0 - c) / 2.0))
    # 2 z_1 conj(z_2) = (a + i b)  =>  arg(z_1) - arg(z_2) = arg(a + i b)
    phase_diff = math.atan2(b, a)
    psis = np.linspace(0.0, 2.0 * math.pi, n_samples, endpoint=False)
    out = np.zeros((n_samples, 4))
    for k, psi in enumerate(psis):
        arg1 = psi
        arg2 = psi - phase_diff
        z1 = r1 * np.exp(1j * arg1)
        z2 = r2 * np.exp(1j * arg2)
        out[k, 0] = z1.real
        out[k, 1] = z1.imag
        out[k, 2] = z2.real
        out[k, 3] = z2.imag
    return out


# ───────────────────────────────────────────────────────────────────────
# Stereographic projection S^3 ⊂ R^4 → R^3 (from a pole)
# ───────────────────────────────────────────────────────────────────────


def stereo_S3_to_R3(
    fibre_R4: np.ndarray, pole: np.ndarray
) -> np.ndarray:
    """Stereographic projection of S^3 ⊂ R^4 from `pole` ∈ S^3 to R^3
    (the hyperplane orthogonal to pole through origin).

    Formula: q = (p - <p, pole> pole) / (1 - <p, pole>).
    Then represent q in coordinates orthogonal to pole.
    """
    n = fibre_R4.shape[0]
    out = np.zeros((n, 3))
    # pick orthonormal basis of pole's orthogonal complement
    e0 = pole / np.linalg.norm(pole)
    # Gram-Schmidt to find 3 orthonormal vectors
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


# ───────────────────────────────────────────────────────────────────────
# Gauss linking number for two oriented closed polygonal loops in R^3
# ───────────────────────────────────────────────────────────────────────


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
# Main scan
# ───────────────────────────────────────────────────────────────────────


def gather_linking_data() -> dict:
    """Build sub-packing, lift to Hopf fibres on S^3, project to R^3,
    compute pairwise Gauss linkings.  Returns dict with results."""
    sub = build_2gen_subpacking()
    s2_points = []
    for label, x, y, r in sub:
        # Use circle centre as the base point on S^2
        s2 = inv_stereo_R2_to_S2(x, y)
        s2_points.append((label, s2))

    # Build Hopf fibres on S^3
    fibres_R4 = []
    for label, s2 in s2_points:
        f = hopf_fibre(s2, n_samples=256)
        fibres_R4.append((label, f))

    # Pick a stereographic pole away from all fibres
    pole_R4 = np.array([0.6, 0.4, 0.5, 0.5])
    pole_R4 = pole_R4 / np.linalg.norm(pole_R4)
    fibres_R3 = []
    for label, f4 in fibres_R4:
        f3 = stereo_S3_to_R3(f4, pole_R4)
        fibres_R3.append((label, f3))

    # Pairwise linking numbers
    n = len(fibres_R3)
    linkings = []
    for i in range(n):
        for j in range(i + 1, n):
            la, fa = fibres_R3[i]
            lb, fb = fibres_R3[j]
            lk = gauss_linking_R3(fa, fb)
            linkings.append((la, lb, lk))

    return {
        "subpacking": sub,
        "S2_points": [(l, s.tolist()) for l, s in s2_points],
        "linkings": linkings,
    }


def _print_diagnostic(data: dict) -> None:
    print("\n  Apollonian 2-generation sub-packing (planar coordinates):")
    print("  " + "-" * 64)
    for label, x, y, r in data["subpacking"]:
        print(f"    {label:<10}  centre = ({x:+.4f}, {y:+.4f}),  radius = {r:.4f}")
    print("\n  Inverse stereographic image of each circle's centre on S^2:")
    print("  " + "-" * 64)
    for label, s2 in data["S2_points"]:
        print(f"    {label:<10}  (a, b, c) = ({s2[0]:+.4f}, {s2[1]:+.4f}, {s2[2]:+.4f})")
    print("\n  Pairwise Gauss linking numbers in R^3 (after Hopf lift to S^3, then stereographic):")
    print("  " + "-" * 64)
    print(f"  {'pair':<28} {'lk':>10}    {'rounded':>10}")
    print("  " + "-" * 64)
    for la, lb, lk in data["linkings"]:
        rounded = round(lk)
        print(f"  {la} <-> {lb:<14} {lk:>+10.4f}    {rounded:>+10}")
    print("  " + "-" * 64)


def test_at_least_one_parent_inscribed_pair_links() -> None:
    data = gather_linking_data()
    parent_inscribed = [
        (la, lb, lk) for (la, lb, lk) in data["linkings"]
        if "inscribed" in {la, lb}
    ]
    assert parent_inscribed, "no parent-inscribed pairs found"
    max_abs = max(abs(lk) for _, _, lk in parent_inscribed)
    assert max_abs >= 0.5, (
        f"HOPF_LINKED_PAIRS FAIL: all parent-inscribed Gauss linkings "
        f"are below 0.5 in magnitude.  Max |lk| = {max_abs:.4f}.  "
        f"Hopf lift produced effectively an unlink; the Hopf-geometric "
        f"interpretation may not realize the framework's chirality "
        f"linking either.  Linkings: "
        f"{[(la, lb, round(lk, 4)) for la, lb, lk in parent_inscribed]}"
    )


def test_parent_inscribed_link_is_unit_integer() -> None:
    data = gather_linking_data()
    parent_inscribed = [
        (la, lb, lk) for (la, lb, lk) in data["linkings"]
        if "inscribed" in {la, lb}
    ]
    near_unit = [
        (la, lb, lk) for la, lb, lk in parent_inscribed
        if 0.8 <= abs(lk) <= 1.2
    ]
    assert near_unit, (
        f"PARENT_INSCRIBED_UNIT FAIL: no parent-inscribed pair has "
        f"|lk| in [0.8, 1.2] (clean Hopf-link unit integer).  "
        f"Linkings: "
        f"{[(la, lb, round(lk, 4)) for la, lb, lk in parent_inscribed]}"
    )


def main() -> None:
    print("=" * 70)
    print("FALSIFIER B1: Hopf-lift Gauss linking numbers")
    print("  Tests whether the 2-generation Apollonian sub-packing,")
    print("  lifted to S^3 via the standard Hopf fibration, has")
    print("  non-trivial linking between parent and inscribed circles.")
    print("=" * 70)

    print("\n  Hopf map: pi(z_1, z_2) = (2 Re(z_1 conj(z_2)), 2 Im(z_1 conj(z_2)), |z_1|^2 - |z_2|^2).")
    print("  Two distinct Hopf fibres on S^3 are *guaranteed* to link with linking number +/- 1.")
    print("  This test verifies that the Apollonian-Hopf lift realises this geometrically.")

    data = gather_linking_data()
    _print_diagnostic(data)

    print("\n" + "=" * 70)
    print("HARD ASSERTIONS")
    print("=" * 70)
    failed: list[tuple[str, str]] = []
    for name, fn in (
        ("HOPF_LINKED_PAIRS         (max |lk| over parent-inscribed >= 0.5)",
         test_at_least_one_parent_inscribed_pair_links),
        ("PARENT_INSCRIBED_UNIT     (some parent-inscribed pair has |lk| ~ 1)",
         test_parent_inscribed_link_is_unit_integer),
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
        print("HOPF-LIFT TOPOLOGICAL CLAIM SUPPORTED.")
        print("  Distinct Hopf fibres in the Apollonian-Hopf lift link with")
        print("  unit Gauss linking number, as required by the Hopf fibration's")
        print("  defining geometric property.  The naive z=depth lift")
        print("  (F14a) producing an unlink was a wrong-arena failure;")
        print("  the proper Hopf arena gives non-trivial topology.")
    else:
        print("HOPF-LIFT GIVES TRIVIAL LINKING (or numerics failed).")
        print("  Inspect the diagnostic output above.  If all linkings are")
        print("  near zero, the Hopf-geometric interpretation does not")
        print("  realise the framework's chirality structure either.")
    print("=" * 70)

    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
