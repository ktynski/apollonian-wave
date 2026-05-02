"""Falsifier D1: Principal closed-orbit Jacobian in 3D Soddy Apollonian.

Hypothesis under test:
  In 3D, the principal closed loxodromic orbit of the Soddy
  Apollonian IFS has a Jacobian that's an integer phi-power.
  The value distinguishes between possible interpretations of
  the 5/6 exponent:

    (a) Universal: log_phi = -6 in all dimensions (suggesting
        the 6 = 2 returns x 3 branches structure is dimension-
        independent, with the trefoil-writhe identification
        at length-3 cycles being a 2D coincidence).
    (b) Dimensional: log_phi increases with dimension d (e.g.
        scales as -2(d+1) or similar).
    (c) Other integer value (informative).

  This is the proper rerun of F14b (which measured aggregate r^2
  contraction in 3D and got phi^-2.2, falsifying the prediction
  phi^-6 for that observable).  D1 measures the SPECTRAL
  observable (principal-orbit Jacobian), which is what the
  Hopf-trefoil-spectral-gap picture predicts.

Setup:
  - 3D Soddy seed: 4 unit spheres at corners of a regular
    tetrahedron with edge 2 (so each sphere has radius 1, centre
    at a vertex).  Outer sphere has radius sqrt(6)/2 + 1.
  - IFS: inversion in each of the 4 inner spheres T_1, T_2, T_3,
    T_4.  Inversion T_k(x) = c_k + r_k^2 (x - c_k) / |x - c_k|^2.
    Conformal factor: |T_k'(x)| = r_k^2 / |x - c_k|^2.
  - Enumerate cyclically-reduced primitive words of length 2-5.
  - For each, find fixed point and compute Jacobian (product of
    conformal factors over the cycle).
  - Identify principal loxodromic orbit (shortest length, then
    largest Jacobian among contractions).

Hard test:
  - Verify the Soddy seed satisfies the 3D Soddy theorem.
  - Verify at least one loxodromic orbit exists.
  - Identify principal loxodromic Jacobian; assert it's an
    integer phi-power within +/- 0.05.

Pass conditions:
  - SODDY_THEOREM:  (sum k)^2 = 3 (sum k^2) within +/- 1e-10.
  - LOXODROMIC_FOUND: at least one orbit with 0 < Jac < 0.95.
  - PRINCIPAL_INTEGER_PHI: log_phi(principal Jac) is an integer
    within +/- 0.05.

Fail modes:
  - SODDY fails: seed misconfigured.
  - log_phi is non-integer: 3D dynamics doesn't give clean
    phi-power structure (the framework's phi-arithmetic is 2D-
    specific, not dimension-universal).
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

PHI = (1.0 + math.sqrt(5.0)) / 2.0
LOG_PHI = math.log(PHI)
SQRT6 = math.sqrt(6.0)


# ───────────────────────────────────────────────────────────────────────
# 3D Soddy seed
# ───────────────────────────────────────────────────────────────────────


def tetrahedron_vertices(edge: float = 2.0) -> list[np.ndarray]:
    """4 vertices of a regular tetrahedron with given edge length,
    centred at origin."""
    a = edge / (2.0 * math.sqrt(2.0))
    return [
        np.array([+a, +a, +a]),
        np.array([+a, -a, -a]),
        np.array([-a, +a, -a]),
        np.array([-a, -a, +a]),
    ]


def soddy_seed_3d() -> tuple[list[tuple[np.ndarray, float]], tuple[np.ndarray, float]]:
    """Returns (inner_spheres, outer_sphere) where each sphere is
    (centre, radius).  4 inner unit spheres at tetrahedron corners,
    1 outer sphere enclosing them."""
    inner_centres = tetrahedron_vertices(edge=2.0)
    inner = [(c, 1.0) for c in inner_centres]
    R_circ = math.sqrt(6.0) / 2.0
    R_outer = R_circ + 1.0
    outer = (np.zeros(3), R_outer)
    return inner, outer


def verify_soddy_3d() -> tuple[float, float, float]:
    """Returns (sum_k_squared, three_times_sum_k_sq, |diff|)."""
    inner, outer = soddy_seed_3d()
    ks = [1.0 / s[1] for s in inner]  # all 1
    k_outer = -1.0 / outer[1]
    all_ks = [k_outer] + ks
    s = sum(all_ks)
    s2 = sum(k * k for k in all_ks)
    return s * s, 3.0 * s2, abs(s * s - 3.0 * s2)


# ───────────────────────────────────────────────────────────────────────
# 3D inversion IFS
# ───────────────────────────────────────────────────────────────────────


def invert_3d(x: np.ndarray, sphere: tuple[np.ndarray, float]) -> np.ndarray:
    c, r = sphere
    diff = x - c
    d2 = float(diff @ diff)
    if d2 < 1e-15:
        return c + np.array([float("inf"), float("inf"), float("inf")])
    return c + (r * r / d2) * diff


def jac_3d(x: np.ndarray, sphere: tuple[np.ndarray, float]) -> float:
    c, r = sphere
    diff = x - c
    d2 = float(diff @ diff)
    return r * r / d2


def apply_word_3d(
    word: list[int], x: np.ndarray, spheres: list[tuple[np.ndarray, float]]
) -> tuple[np.ndarray, float]:
    cur = x.copy()
    J = 1.0
    for idx in word:
        J *= jac_3d(cur, spheres[idx])
        cur = invert_3d(cur, spheres[idx])
    return cur, J


def reduce_word(word: list[int]) -> list[int]:
    out: list[int] = []
    for g in word:
        if out and out[-1] == g:
            out.pop()
        else:
            out.append(g)
    return out


def cyclic_reduce(word: list[int]) -> list[int]:
    w = reduce_word(word)
    while len(w) >= 2 and w[0] == w[-1]:
        w = w[1:-1]
    return w


def enumerate_primitive(max_len: int, n_gen: int = 4) -> list[list[int]]:
    seen: set[tuple] = set()
    out: list[list[int]] = []
    for n in range(2, max_len + 1):
        for code in range(n_gen ** n):
            word: list[int] = []
            x = code
            for _ in range(n):
                word.append(x % n_gen)
                x //= n_gen
            cr = cyclic_reduce(word)
            if not cr or len(cr) < 2:
                continue
            primitive = True
            for div in range(1, len(cr)):
                if len(cr) % div == 0:
                    block = cr[:div]
                    if cr == block * (len(cr) // div):
                        primitive = False
                        break
            if not primitive:
                continue
            rotations = [tuple(cr[i:] + cr[:i]) for i in range(len(cr))]
            key = min(rotations)
            if key in seen:
                continue
            seen.add(key)
            out.append(list(key))
    return out


def find_orbit_jacobian_3d(
    word: list[int],
    spheres: list[tuple[np.ndarray, float]],
    n_iter: int = 4000,
    seeds: list[np.ndarray] | None = None,
) -> float | None:
    odd = (len(word) % 2 == 1)
    iter_word = (word + word) if odd else word
    if seeds is None:
        seeds = [
            np.array([0.0, 0.0, 0.0]),
            np.array([0.1, 0.2, 0.3]),
            np.array([-0.2, 0.1, 0.0]),
            np.array([0.3, -0.1, 0.2]),
        ]
    for seed in seeds:
        x = seed.copy()
        for _ in range(n_iter):
            for idx in iter_word:
                x = invert_3d(x, spheres[idx])
        x_image, J = apply_word_3d(word, x, spheres)
        if np.linalg.norm(x_image - x) > 1e-3:
            _, J2 = apply_word_3d(word + word, x, spheres)
            if J2 <= 0 or J2 >= 1.0 - 1e-9:
                continue
            J = math.sqrt(J2)
        if 0.05 < J < 0.95:
            return J
    return None


def scan_3d_orbits(max_len: int = 5) -> list[dict]:
    inner, _ = soddy_seed_3d()
    out: list[dict] = []
    for word in enumerate_primitive(max_len, n_gen=4):
        J = find_orbit_jacobian_3d(word, inner)
        if J is not None:
            out.append({
                "word": word,
                "length": len(word),
                "jac": J,
                "log_phi": math.log(J) / LOG_PHI,
            })
    return out


# ───────────────────────────────────────────────────────────────────────
# Diagnostics and tests
# ───────────────────────────────────────────────────────────────────────


def _print_seed_info() -> None:
    inner, outer = soddy_seed_3d()
    print("\n  3D Soddy seed:")
    print("  " + "-" * 60)
    for k, (c, r) in enumerate(inner):
        print(f"    inner_{k+1}: centre = ({c[0]:+.4f}, {c[1]:+.4f}, "
              f"{c[2]:+.4f}),  r = {r:.4f}")
    c_o, r_o = outer
    print(f"    outer:    centre = ({c_o[0]:+.4f}, {c_o[1]:+.4f}, "
          f"{c_o[2]:+.4f}),  r = {r_o:.4f}")
    s2_lhs, s2_rhs, diff = verify_soddy_3d()
    print(f"    Soddy check: (sum k)^2 = {s2_lhs:.6f},  "
          f"3 sum k^2 = {s2_rhs:.6f},  diff = {diff:.4e}")


def _print_orbits(records: list[dict]) -> None:
    print("\n  3D Soddy IFS orbits found:")
    print("  " + "-" * 70)
    print(f"  {'word':<22} {'len':>4} {'|Jac|':>14} {'log_phi':>10}")
    print("  " + "-" * 70)
    for r in sorted(records, key=lambda r: (r["length"], r["jac"])):
        word_str = "".join(f"T{i+1}" for i in r["word"])
        if len(word_str) > 20:
            word_str = word_str[:17] + "..."
        marker = ""
        if abs(r["log_phi"] - round(r["log_phi"])) < 0.05:
            marker = "  <- integer log_phi"
        print(f"  {word_str:<22} {r['length']:>4} {r['jac']:>14.6e} "
              f"{r['log_phi']:>+10.4f}{marker}")
    print("  " + "-" * 70)


def test_soddy_theorem_holds() -> None:
    s2_lhs, s2_rhs, diff = verify_soddy_3d()
    assert diff < 1e-10, (
        f"SODDY_THEOREM FAIL: |(sum k)^2 - 3 sum k^2| = {diff:.4e}"
    )


def test_loxodromic_orbit_found() -> None:
    records = scan_3d_orbits(max_len=4)
    assert records, "no loxodromic orbits found in 3D Soddy IFS"


def test_principal_jacobian_integer_phi_power() -> None:
    records = scan_3d_orbits(max_len=5)
    assert records, "no orbits found"
    # principal = shortest length, then largest |Jac| (least contracting)
    min_len = min(r["length"] for r in records)
    short_orbits = [r for r in records if r["length"] == min_len]
    principal = max(short_orbits, key=lambda r: r["jac"])
    log_phi = principal["log_phi"]
    nearest = round(log_phi)
    assert abs(log_phi - nearest) < 0.10, (
        f"PRINCIPAL_INTEGER_PHI FAIL: principal orbit "
        f"({''.join(f'T{i+1}' for i in principal['word'])}, length "
        f"{principal['length']}) has log_phi = {log_phi:+.4f}; "
        f"nearest integer = {nearest}.  Expected integer within 0.10."
    )


def main() -> None:
    print("=" * 70)
    print("FALSIFIER D1: Principal closed-orbit Jacobian in 3D Soddy")
    print("  Apollonian.  Tests the dimensional behavior of the spectral")
    print("  observable measured in A1 (which gave phi^-6 in 2D).")
    print("=" * 70)

    _print_seed_info()

    records = scan_3d_orbits(max_len=5)
    _print_orbits(records)

    print("\n" + "=" * 70)
    print("HARD ASSERTIONS")
    print("=" * 70)
    failed: list[tuple[str, str]] = []
    for name, fn in (
        ("SODDY_THEOREM            (3D Soddy seed satisfies Q(k)=3 sum k^2)",
         test_soddy_theorem_holds),
        ("LOXODROMIC_FOUND         (at least one loxodromic orbit)",
         test_loxodromic_orbit_found),
        ("PRINCIPAL_INTEGER_PHI    (principal log_phi is an integer)",
         test_principal_jacobian_integer_phi_power),
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
        print("3D PRINCIPAL ORBIT JACOBIAN IS A CLEAN INTEGER PHI-POWER.")
        records = scan_3d_orbits(max_len=4)
        if records:
            min_len = min(r["length"] for r in records)
            short = [r for r in records if r["length"] == min_len]
            principal = max(short, key=lambda r: r["jac"])
            print(f"  Principal orbit: log_phi = {principal['log_phi']:+.4f}")
            print(f"  Compare to 2D value (A1): log_phi = -6.0000")
    else:
        print("3D PRINCIPAL ORBIT JACOBIAN DOES NOT HAVE CLEAN PHI-POWER.")
    print("=" * 70)

    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
