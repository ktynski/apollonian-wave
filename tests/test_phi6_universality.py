"""Falsifier B-COINCIDENCE: principal-cycle Jacobian = phi^-6
universality across Apollonian seeds.

Hypothesis under test:
  In 2D Apollonian packings (any Descartes-compatible curvature
  quadruple), the principal length-3 closed orbit of the inverted-
  inner-circle IFS has Jacobian phi^-6 EXACTLY, independent of the
  specific seed.  This would make phi^-6 a structural invariant of
  the Apollonian Schottky group rather than an artifact of the
  particular (-1, 2, 2, 3) seed used in A1.

  Falsification of universality (different seeds give different
  Jacobians) would mean phi^-6 is seed-specific, weakening the
  "spectral gap is phi^-6" claim from a universal theorem to a
  numerical coincidence at one configuration.

Setup:
  - Multiple primitive integral Apollonian seeds (each a quadruple
    of curvatures (k_outer, k_a, k_b, k_c) satisfying Descartes
    Q(k) = (sum k)^2 - 2 sum(k^2) = 0).
  - For each seed, compute the centres and radii of the three
    inner circles; build inversion IFS T_1, T_2, T_3.
  - Find the principal length-3 closed orbit T_1 T_3 T_2 (or its
    cyclic relabeling).  Compute Newton-iterated fixed point and
    cumulative Jacobian.
  - Tabulate log_phi(Jac) per seed.

Hard test:
  - Verify each seed satisfies Descartes.
  - Find principal loxodromic orbit for each seed.
  - log_phi(Jac) should be -6 +/- 0.05 for ALL seeds.

Pass conditions:
  - DESCARTES_HOLDS: each seed satisfies Q(k) = 0 within +/- 1e-9.
  - LOX_FOUND: each seed has at least one length-3 loxodromic orbit.
  - PHI6_UNIVERSAL: log_phi(Jac) for the principal orbit of each
    seed is in [-6.1, -5.9].

Fail modes:
  - Some seed gives log_phi != -6: phi^-6 is NOT universal; it's
    an artifact of (-1, 2, 2, 3) and possibly conformally-related
    seeds.  This would make the "spectral gap is phi^-6"
    interpretation seed-specific.
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
# Build 2D Apollonian seed from curvature quadruple
# ───────────────────────────────────────────────────────────────────────


def descartes_check(ks: tuple[float, float, float, float]) -> float:
    s = sum(ks)
    s2 = sum(k * k for k in ks)
    return abs(s * s - 2.0 * s2)


def seed_centres(
    ks: tuple[float, float, float, float],
    place_outer_at_origin: bool = True,
) -> list[complex]:
    """Given a Descartes-compatible curvature quadruple
    (k_outer, k_a, k_b, k_c), return 4 complex centres for the 4
    circles.  Convention: k_outer is the (negative-curvature)
    bounding circle.

    Strategy: the outer circle has radius 1/|k_outer| centred at
    origin.  Place the smallest inner circle on the negative-real
    axis tangent to the outer circle, then place the others using
    distance constraints.
    """
    k0, ka, kb, kc = ks
    r_o = 1.0 / abs(k0)
    r_a, r_b, r_c = 1.0 / ka, 1.0 / kb, 1.0 / kc

    # Centre 0: outer at origin
    c0 = complex(0.0, 0.0)
    # Centre A: smallest inner circle (largest k) placed on
    # negative real axis, internally tangent to outer.
    # Internal tangency: |c_a - c_0| = r_o - r_a.
    c_a = complex(-(r_o - r_a), 0.0)
    # Centre B: inner, tangent to outer (|c_b| = r_o - r_b) and
    # tangent to A (|c_b - c_a| = r_a + r_b).  Two solutions; pick
    # one in upper half-plane.
    d_o = r_o - r_b
    d_a = r_a + r_b
    # |c_b|^2 = d_o^2; |c_b - c_a|^2 = d_a^2
    # Let c_b = (x, y).  x^2 + y^2 = d_o^2, (x - c_a.real)^2 + y^2 = d_a^2
    # Subtract: (x - c_a.real)^2 - x^2 = d_a^2 - d_o^2
    #   => -2 x c_a.real + c_a.real^2 = d_a^2 - d_o^2
    #   => x = (c_a.real^2 - d_a^2 + d_o^2) / (2 c_a.real)
    x_b = (c_a.real ** 2 - d_a ** 2 + d_o ** 2) / (2.0 * c_a.real)
    y_b_sq = d_o ** 2 - x_b ** 2
    if y_b_sq < -1e-9:
        raise ValueError(f"seed inconsistent: y_b^2 = {y_b_sq}")
    y_b = math.sqrt(max(0.0, y_b_sq))
    c_b = complex(x_b, y_b)
    # Centre C: tangent to outer + A + B; choose unique interior.
    d_o = r_o - r_c
    d_a = r_a + r_c
    d_b = r_b + r_c
    # Solve via Newton from initial guess (lower half-plane, by symmetry).
    p = np.array([0.3 * c_a.real, -0.5 * y_b])

    def residual(p):
        x, y = p
        return np.array([
            math.hypot(x, y) - d_o,
            math.hypot(x - c_a.real, y - c_a.imag) - d_a,
            math.hypot(x - c_b.real, y - c_b.imag) - d_b,
        ])

    for _ in range(80):
        f = residual(p)
        if np.linalg.norm(f) < 1e-13:
            break
        eps = 1e-7
        J = np.zeros((3, 2))
        for j in range(2):
            pj = p.copy()
            pj[j] += eps
            J[:, j] = (residual(pj) - f) / eps
        dp = np.linalg.lstsq(J, -f, rcond=None)[0]
        p = p + dp
    c_c = complex(float(p[0]), float(p[1]))

    return [c0, c_a, c_b, c_c]


# ───────────────────────────────────────────────────────────────────────
# IFS and orbit search (similar to A1)
# ───────────────────────────────────────────────────────────────────────


def make_ifs(seed_ks: tuple[float, float, float, float]) -> list[tuple[complex, float]]:
    """Returns the 3 inner circles (centre, radius) used as IFS generators."""
    centres = seed_centres(seed_ks)
    inner = [(centres[i + 1], 1.0 / seed_ks[i + 1]) for i in range(3)]
    return inner


def invert(z: complex, sphere: tuple[complex, float]) -> complex:
    c, r = sphere
    diff = z - c
    if abs(diff) < 1e-15:
        return complex(float("inf"), 0.0)
    return c + (r * r) / np.conj(diff)


def jac(z: complex, sphere: tuple[complex, float]) -> float:
    c, r = sphere
    diff = z - c
    return (r * r) / (abs(diff) ** 2)


def apply_word(
    word: list[int], z: complex, ifs: list[tuple[complex, float]]
) -> tuple[complex, float]:
    cur = z
    J = 1.0
    for idx in word:
        J *= jac(cur, ifs[idx])
        cur = invert(cur, ifs[idx])
    return cur, J


def find_principal_jacobian(
    ifs: list[tuple[complex, float]],
    word: list[int] = [0, 2, 1],
    n_iter: int = 4000,
) -> tuple[float | None, complex | None]:
    """Find fixed point of `word` and return cumulative Jacobian."""
    odd = (len(word) % 2 == 1)
    iter_word = (word + word) if odd else word
    # Pick seeds inside the inner-region centroid
    centroid = sum(s[0] for s in ifs) / 3.0
    seeds = [
        centroid,
        centroid + complex(0.05, 0.0),
        centroid + complex(0.0, 0.05),
        centroid + complex(-0.05, 0.05),
    ]
    for seed in seeds:
        z = seed
        for _ in range(n_iter):
            for idx in iter_word:
                z = invert(z, ifs[idx])
        z_image, J = apply_word(word, z, ifs)
        if abs(z_image - z) > 1e-3:
            _, J2 = apply_word(word + word, z, ifs)
            if J2 <= 0 or J2 >= 1.0 - 1e-9:
                continue
            J = math.sqrt(J2)
        if 0.001 < J < 0.95:
            return J, z
    return None, None


# ───────────────────────────────────────────────────────────────────────
# Catalogue of integer Apollonian seeds
# ───────────────────────────────────────────────────────────────────────


SEEDS: list[tuple[str, tuple[float, float, float, float]]] = [
    # Standard (-1, 2, 2, 3) -- A1 reference
    ("(-1,2,2,3)",   (-1.0, 2.0, 2.0, 3.0)),
    # Other integral primitive Apollonian seeds
    ("(-2,3,6,7)",   (-2.0, 3.0, 6.0, 7.0)),
    ("(-3,5,8,8)",   (-3.0, 5.0, 8.0, 8.0)),
    ("(-6,11,14,15)", (-6.0, 11.0, 14.0, 15.0)),
    ("(-7,12,17,20)", (-7.0, 12.0, 17.0, 20.0)),
    ("(-9,18,19,22)", (-9.0, 18.0, 19.0, 22.0)),
]


# ───────────────────────────────────────────────────────────────────────
# Diagnostics and tests
# ───────────────────────────────────────────────────────────────────────


def gather_results() -> list[dict]:
    out: list[dict] = []
    for label, ks in SEEDS:
        descartes = descartes_check(ks)
        try:
            ifs = make_ifs(ks)
        except Exception as e:
            out.append({
                "seed": label, "ks": ks, "descartes": descartes,
                "ifs_error": str(e), "J": None, "log_phi": None,
            })
            continue
        J, z = find_principal_jacobian(ifs)
        out.append({
            "seed": label, "ks": ks, "descartes": descartes,
            "ifs": ifs, "J": J, "z_star": z,
            "log_phi": (math.log(J) / LOG_PHI) if J else None,
        })
    return out


def _print_results(results: list[dict]) -> None:
    print("\n  Principal length-3 cycle Jacobians across multiple seeds:")
    print("  " + "-" * 76)
    print(f"  {'seed':<18} {'Descartes |Q|':>14} {'principal |Jac|':>16} {'log_phi':>10}")
    print("  " + "-" * 76)
    for r in results:
        if r.get("ifs_error"):
            print(f"  {r['seed']:<18} {r['descartes']:>14.2e}   IFS BUILD FAILED")
            continue
        J = r["J"]
        if J is None:
            print(f"  {r['seed']:<18} {r['descartes']:>14.2e} {'no orbit':>16}")
            continue
        log_phi = r["log_phi"]
        marker = ""
        if abs(log_phi - (-6.0)) < 0.05:
            marker = "  <- phi^-6"
        elif abs(log_phi - round(log_phi)) < 0.05:
            marker = f"  <- phi^{round(log_phi):+d}"
        print(f"  {r['seed']:<18} {r['descartes']:>14.2e} "
              f"{J:>16.6e} {log_phi:>+10.4f}{marker}")
    print("  " + "-" * 76)


def test_all_seeds_satisfy_descartes() -> None:
    for label, ks in SEEDS:
        d = descartes_check(ks)
        assert d < 1e-9, f"DESCARTES FAIL for {label}: |Q(k)| = {d:.4e}"


def test_all_seeds_have_loxodromic_orbit() -> None:
    results = gather_results()
    for r in results:
        if r.get("ifs_error"):
            raise AssertionError(
                f"IFS_BUILD_FAIL for {r['seed']}: {r['ifs_error']}"
            )
        assert r["J"] is not None, (
            f"NO_LOX_ORBIT for {r['seed']}: principal length-3 cycle "
            f"didn't converge."
        )


def test_phi6_universal() -> None:
    results = gather_results()
    deviations = []
    for r in results:
        if r.get("ifs_error") or r["J"] is None:
            continue
        log_phi = r["log_phi"]
        deviations.append((r["seed"], log_phi, abs(log_phi - (-6.0))))
    failures = [(s, l, d) for s, l, d in deviations if d > 0.05]
    assert not failures, (
        f"PHI6_UNIVERSAL FAIL: {len(failures)} seeds give log_phi != -6 "
        f"within 0.05.  Deviations: {failures}.  All values: "
        f"{[(s, round(l, 4)) for s, l, _ in deviations]}"
    )


def main() -> None:
    print("=" * 76)
    print("FALSIFIER B-COINCIDENCE: principal-cycle Jacobian = phi^-6")
    print("  universality across multiple integral Apollonian seeds.")
    print("=" * 76)

    print(f"\n  Reference: A1 measured phi^-6 for the seed (-1, 2, 2, 3).")
    print(f"  Question: does this hold for ALL integer Apollonian seeds, or")
    print(f"  is phi^-6 specific to the standard (-1, 2, 2, 3) configuration?")

    results = gather_results()
    _print_results(results)

    print("\n" + "=" * 76)
    print("HARD ASSERTIONS")
    print("=" * 76)
    failed: list[tuple[str, str]] = []
    for name, fn in (
        ("DESCARTES_HOLDS    (every seed satisfies Q(k) = 0)",
         test_all_seeds_satisfy_descartes),
        ("LOX_FOUND          (every seed has length-3 loxodromic orbit)",
         test_all_seeds_have_loxodromic_orbit),
        ("PHI6_UNIVERSAL     (all seeds give log_phi = -6 +/- 0.05)",
         test_phi6_universal),
    ):
        try:
            fn()
            print(f"  PASS: {name}")
        except AssertionError as exc:
            print(f"  FAIL: {name}")
            print(f"        {exc}")
            failed.append((name, str(exc)))

    print("\n" + "=" * 76)
    if not failed:
        print("PHI^-6 IS A UNIVERSAL INVARIANT OF THE APOLLONIAN SCHOTTKY GROUP.")
        print("  The principal length-3 cycle Jacobian = phi^-6 holds across")
        print("  multiple distinct integer Apollonian seeds, supporting the")
        print("  claim that phi^-6 is the spectral gap of L_phi structurally,")
        print("  not coincidentally.")
    else:
        print("PHI^-6 IS NOT UNIVERSAL.")
        print("  The principal-cycle Jacobian depends on the specific seed.")
        print("  This weakens the 'spectral gap = phi^-6' claim from a")
        print("  universal theorem to a feature of one configuration.")
    print("=" * 76)

    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
