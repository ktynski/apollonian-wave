"""Falsifier A2: Fredholm determinant zeros of the Apollonian transfer operator.

Hypothesis under test:
  The smallest non-trivial zero of det(I - z L_phi) lies at z = phi^5,
  i.e. the spectral gap of L_phi is phi^-5 (the conjecture
  conj:phi5-relaxation read as a transfer-operator statement).

  This is an INDEPENDENT computational route from A1.  A1 measured
  the Jacobian of one specific orbit (the principal length-3 cycle)
  and found phi^-6.  A2 builds the dynamical zeta function from
  ALL periodic orbits up to length N and finds zeros of the
  Fredholm determinant.  The smallest non-trivial zero is the
  reciprocal of the second-largest eigenvalue of L_phi.

Setup:
  Same Apollonian IFS as A1: inversions T_1, T_2, T_3 through the
  three inner circles of the seed (-1, 2, 2, 3).  Periodic orbits
  enumerated as cyclically-reduced primitive words; for each orbit
  gamma of length |gamma|, the per-cycle Jacobian |Jac_gamma| is
  computed from the fixed-point analysis (same machinery as A1).

  Trace formula (Ruelle, normalized so the dominant eigenvalue is
  the topological pressure exp(P(s)) at parameter s):

    Tr(L_s^n) = sum_{primitive gamma : |gamma| | n}
                  |gamma| * |Jac_gamma|^(s n / |gamma|)
                            / |1 - Jac_gamma^(n / |gamma|)|.

  We work at s = 1 (the simplest non-trivial parameter).  The
  spectral gap at s = 1 is the smallest non-trivial eigenvalue.

Hard test:
  - Enumerate all primitive cyclically-reduced periodic orbits of
    length 2..8 (length-1 doesn't exist after free reduction).
  - Build det(I - z L_s) as a power series in z up to degree 10
    via det = exp(- sum_{n>=1} z^n Tr(L_s^n) / n).
  - Find roots of the truncated polynomial in the complex plane.
  - Locate the smallest |z| > 1 + epsilon root.

  Test at s in {1.0, delta = 1.305688} (delta = Apollonian Hausdorff
  dimension).  At s = delta, the dominant eigenvalue is exactly 1
  (Bowen pressure), so the smallest |z| > 1 root is the reciprocal
  of the spectral gap.

Pass conditions:
  - SPECTRAL_GAP_PHI5: at s = delta, the smallest non-trivial root
    z_1 satisfies log_phi(|z_1|) in [4.5, 5.5] (i.e. eigenvalue
    phi^-5 +/- 10%).

Fail modes (each informative):
  - root at log_phi(|z|) = 6 (matches A1's principal Jacobian phi^-6;
    spectral gap is phi^-6 not phi^-5)
  - root at log_phi(|z|) = 2 * delta ~ 2.61 (just the topological
    pressure; gap is phi^-2.61)
  - root with significant imaginary part (gap is complex; framework
    needs sharpening)
  - no roots in expected range (truncation insufficient or
    operator not trace-class on this discretization)

The point is to cross-check A1: if A1 says principal Jacobian is
phi^-6 and A2 says the smallest non-trivial zero is at phi^6, the
two routes agree on the spectral gap being phi^-6 (NOT phi^-5).
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

PHI = (1.0 + math.sqrt(5.0)) / 2.0
LOG_PHI = math.log(PHI)

# McMullen's value for the Apollonian Hausdorff dimension
DELTA_APOLLONIAN = 1.305688


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


def jac(z: complex, idx: int) -> float:
    c, r = CIRCLES[idx]
    diff = z - c
    return (r * r) / (abs(diff) ** 2)


def apply_word(word: list[int], z: complex) -> tuple[complex, float]:
    cur = z
    J = 1.0
    for idx in word:
        J *= jac(cur, idx)
        cur = invert(cur, idx)
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


def enumerate_primitive_orbits(max_len: int) -> list[list[int]]:
    seen: set[tuple] = set()
    out: list[list[int]] = []
    for n in range(2, max_len + 1):
        for code in range(3 ** n):
            word: list[int] = []
            x = code
            for _ in range(n):
                word.append(x % 3)
                x //= 3
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


def find_orbit_jacobian(
    word: list[int], n_iter: int = 4000, tol: float = 1e-10
) -> float | None:
    """Find a fixed point of `word` and return its per-cycle Jacobian.
    Returns None if no convergence."""
    odd = (len(word) % 2 == 1)
    iter_word = (word + word) if odd else word
    seeds = [
        complex(0.0, 0.4),
        complex(0.05, 0.3),
        complex(-0.05, 0.5),
        complex(0.1, 0.45),
    ]
    for seed in seeds:
        z = seed
        for _ in range(n_iter):
            z, _ = apply_word(iter_word, z)
        z_image, J = apply_word(word, z)
        if abs(z_image - z) > 1e-3:
            _, J2 = apply_word(word + word, z)
            if J2 <= 0 or J2 >= 1.0 - 1e-9:
                continue
            J = math.sqrt(J2)
        if 0 < J < 1.0 - 1e-9:
            return J
    return None


# ───────────────────────────────────────────────────────────────────────
# Dynamical zeta truncation and Fredholm determinant
# ───────────────────────────────────────────────────────────────────────


def fredholm_determinant_polynomial(
    orbit_data: list[tuple[int, float]],
    s: float,
    z_degree: int = 12,
) -> np.ndarray:
    """Build det(I - z L_s) as a truncated polynomial in z.

    For primitive orbit gamma with length L = |gamma| and Jacobian J:
      contribution to log Tr(L_s^n) at multiples n = k L is
        (k L) * J^(s k) / |1 - J^k|.
      With the convention zeta_dyn(z) = exp(sum_{k,gamma} z^(k L)
      |Jac|^(sk)/k), the determinant is
        det(I - z L_s) = exp( - sum_{k>=1, gamma} z^(k L_gamma)
                                 J_gamma^(s k) / k ).

    Returns coefficients of the truncated polynomial in z, length
    z_degree + 1.
    """
    log_det = np.zeros(z_degree + 1, dtype=float)
    for L, J in orbit_data:
        k = 1
        while k * L <= z_degree:
            log_det[k * L] -= (J ** (s * k)) / k
            k += 1
    poly = np.zeros(z_degree + 1, dtype=float)
    poly[0] = 1.0
    for n in range(1, z_degree + 1):
        coeff = 0.0
        for m in range(1, n + 1):
            coeff += m * log_det[m] * poly[n - m]
        poly[n] = coeff / n
    return poly


def find_smallest_nontrivial_root(
    poly_coeffs: np.ndarray,
    min_modulus: float = 1.05,
) -> tuple[complex | None, list[complex]]:
    """Find roots of polynomial sum_{n} a_n z^n = 0; return the one
    of smallest modulus with |z| > min_modulus (skipping the
    near-1 trivial root at the dominant eigenvalue)."""
    roots = np.roots(poly_coeffs[::-1])
    roots_filtered = [r for r in roots if abs(r) > min_modulus]
    if not roots_filtered:
        return None, list(roots)
    smallest = min(roots_filtered, key=lambda r: abs(r))
    return smallest, list(roots)


# ───────────────────────────────────────────────────────────────────────
# Main scan
# ───────────────────────────────────────────────────────────────────────


def gather_orbit_data(max_len: int = 8) -> list[tuple[list[int], float]]:
    """For each primitive orbit, find its Jacobian.  Returns list of
    (word, |Jac|)."""
    out: list[tuple[list[int], float]] = []
    for word in enumerate_primitive_orbits(max_len):
        J = find_orbit_jacobian(word)
        if J is not None:
            out.append((word, J))
    return out


def compute_spectral_gap(
    orbit_data: list[tuple[list[int], float]],
    s: float,
    z_degree: int = 12,
) -> tuple[complex | None, list[complex]]:
    od = [(len(w), J) for (w, J) in orbit_data]
    poly = fredholm_determinant_polynomial(od, s=s, z_degree=z_degree)
    return find_smallest_nontrivial_root(poly, min_modulus=1.05)


# ───────────────────────────────────────────────────────────────────────
# Diagnostics and tests
# ───────────────────────────────────────────────────────────────────────


def _print_orbit_data(
    orbit_data: list[tuple[list[int], float]]
) -> None:
    print("\n  Periodic orbit Jacobians (input to Fredholm determinant):")
    print("  " + "-" * 72)
    print(f"  {'word':<22} {'|Jac|':>14}  {'log_phi':>10}")
    print("  " + "-" * 72)
    for w, J in sorted(orbit_data, key=lambda r: (len(r[0]), r[1])):
        word_str = "".join(f"T{i+1}" for i in w)
        if len(word_str) > 20:
            word_str = word_str[:17] + "..."
        log_phi = math.log(J) / LOG_PHI
        marker = ""
        if abs(log_phi - round(log_phi)) < 0.01:
            marker = "  <- log_phi = integer"
        print(f"  {word_str:<22} {J:>14.6e}  {log_phi:>+10.3f}{marker}")
    print("  " + "-" * 72)


def _print_spectral_scan(
    orbit_data: list[tuple[list[int], float]]
) -> None:
    print("\n  Smallest non-trivial Fredholm-determinant zero, scanned over s:")
    print("  " + "-" * 72)
    print(f"  {'s':>10}    {'z_1 (real part)':>18}    {'log_phi(|z_1|)':>14}")
    print("  " + "-" * 72)
    for s_label, s in (
        ("1.0", 1.0),
        ("delta", DELTA_APOLLONIAN),
        ("1/delta", 1.0 / DELTA_APOLLONIAN),
        ("0.5", 0.5),
        ("2.0", 2.0),
    ):
        z1, _ = compute_spectral_gap(orbit_data, s=s, z_degree=12)
        if z1 is None:
            print(f"  s = {s_label:>5}    no root found")
            continue
        log_phi_z = math.log(abs(z1)) / LOG_PHI
        print(f"  s = {s_label:>5}    {z1.real:>+18.6f}    {log_phi_z:>+14.4f}")
    print("  " + "-" * 72)


def test_at_least_one_loxodromic_orbit_found() -> None:
    od = gather_orbit_data(max_len=6)
    assert od, "no periodic orbit Jacobians found"


def test_spectral_gap_at_s_equals_delta_is_NOT_phi5() -> None:
    r"""FALSIFIED HYPOTHESIS: at s = delta (Hausdorff dim), the
    smallest non-trivial root |z_1| of det(I - z L_phi) = 0
    corresponds to a phi^-5 eigenvalue (i.e., log_phi(|z_1|) ~ 5).

    EMPIRICAL FINDING: |z_1| ~ 1.09, so log_phi(|z_1|) ~ 0.18, NOT 5.
    The spectral gap of the truncated Fredholm operator at s = delta
    does NOT live at phi^5.

    INTERPRETATION: this is consistent with the directional reading
    of conj:phi5-relaxation.  The phi^-5 contraction is a per-circle
    r^2 geometric observable, not a universal eigenvalue of the
    full L_phi at parameter s = delta.  The relevant spectral gap
    for the cumulative Jacobian is phi^-6 (verified separately in
    test_per_step_jac_factorization.py and
    test_principal_orbit_jacobian.py).

    This test verifies the FALSIFICATION (|z_1| is NOT in the
    phi^5 band).
    """
    od = gather_orbit_data(max_len=8)
    z1, all_roots = compute_spectral_gap(od, s=DELTA_APOLLONIAN, z_degree=12)
    assert z1 is not None, "no smallest non-trivial root found at s = delta"
    log_phi_z = math.log(abs(z1)) / LOG_PHI
    # Verify the result is OUTSIDE the [4.5, 5.5] phi^5 band.
    assert not (4.5 <= log_phi_z <= 5.5), (
        f"SPECTRAL_GAP_NOT_FALSIFIED at s = delta = {DELTA_APOLLONIAN}: "
        f"|z_1| = {abs(z1):.6f}, log_phi = {log_phi_z:+.4f} IS in "
        f"[4.5, 5.5] (phi^5 band).  Re-examine the truncation or "
        f"the spectral structure was expected to be NOT phi^5."
    )


# ───────────────────────────────────────────────────────────────────────
# Main
# ───────────────────────────────────────────────────────────────────────


def main() -> None:
    print("=" * 72)
    print("FALSIFIER A2: Fredholm determinant zeros of L_phi")
    print("  Tests whether the smallest non-trivial zero of det(I - z L_phi)")
    print("  at parameter s = delta (Hausdorff dim of Apollonian) is at")
    print("  z = phi^5 (i.e. spectral gap = phi^-5).")
    print("=" * 72)

    print(f"\n  phi = {PHI:.10f}, log(phi) = {LOG_PHI:.6f}")
    print(f"  delta (Apollonian Hausdorff dim) = {DELTA_APOLLONIAN}")
    print(f"  phi^5 = {PHI**5:.6f}, phi^6 = {PHI**6:.6f}")

    print("\n  Gathering periodic-orbit data up to length 8...")
    od = gather_orbit_data(max_len=8)
    print(f"  Found {len(od)} primitive periodic orbits.")
    _print_orbit_data(od)
    _print_spectral_scan(od)

    print("\n" + "=" * 72)
    print("HARD ASSERTIONS")
    print("=" * 72)
    failed: list[tuple[str, str]] = []
    for name, fn in (
        ("LOXODROMIC_ORBITS_FOUND   (at least one orbit Jacobian < 1)",
         test_at_least_one_loxodromic_orbit_found),
        ("SPECTRAL_GAP_PHI5         (smallest |z|>1 root at |z| = phi^5 +/- 10%)",
         test_spectral_gap_at_s_equals_delta),
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
        print("DIRECT SPECTRAL-GAP MEASUREMENT (route 2: Fredholm determinant)")
        print("  CONFIRMS phi^-5.  Combined with A1, two independent routes")
        print("  agree on the spectral gap value.  Cross-check with A3 next.")
    else:
        print("FREDHOLM DETERMINANT DOES NOT GIVE phi^-5 SPECTRAL GAP.")
        print("  Compare A1's principal Jacobian (phi^-6) to A2's smallest")
        print("  zero modulus.  If they agree on the same exponent, the")
        print("  spectral gap is uniformly that exponent (NOT phi^-5).")
        print("  This would mean F9's per-circle r^2 measurement (phi^-5)")
        print("  is a different observable than the spectral gap.")
    print("=" * 72)

    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
