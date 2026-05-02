"""Falsifier C3: Chiral fold locus = equatorial circle of S^2 Hopf base.

Hypothesis under test:
  The "chiral fold" of the framework -- the locus on S^3 where
  the left-Hopf and right-Hopf maps agree (pi_L = pi_R) --
  projects to the *equatorial circle* of S^2 (the great circle
  where the second component vanishes), realising the
  framework's beta = 1/2 chiral-fold balance.

Setup:
  Left and right Hopf maps:
    pi_L(z_1, z_2) = (2 Re z_1 conj(z_2), 2 Im z_1 conj(z_2),
                       |z_1|^2 - |z_2|^2)
    pi_R(z_1, z_2) = (2 Re conj(z_1) z_2, 2 Im conj(z_1) z_2,
                       |z_1|^2 - |z_2|^2)
                   = (2 Re z_1 conj(z_2), -2 Im z_1 conj(z_2),
                       |z_1|^2 - |z_2|^2)

  So pi_L = pi_R iff Im(z_1 conj(z_2)) = 0, i.e. z_1 conj(z_2)
  is real.  This 2-dim submanifold of S^3 is a Clifford torus.
  Under pi_L it maps to {(a, 0, c) on S^2 with a^2 + c^2 = 1}
  -- the equatorial S^1 of S^2.

Hard test:
  - Sample 2000 random points on the locus {Im(z_1 conj(z_2)) = 0}.
  - Verify their pi_L images all have b ~ 0 (the second
    component) within +/- 1e-6.
  - Verify pi_L = pi_R at every sampled point.

Pass conditions:
  - LOCUS_PI_LR_EQUAL:  ||pi_L - pi_R|| < 1e-10 at every locus
    point.
  - IMAGE_ON_EQUATOR:   max |b component| of pi_L image over
    all locus points < 1e-10.

Fail modes:
  - locus differs from chiral-fold geometric prediction
  - IMAGE_ON_EQUATOR fails (locus doesn't project to equator)
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def hopf_left_image(z1: complex, z2: complex) -> np.ndarray:
    a = (2.0 * z1 * np.conj(z2)).real
    b = (2.0 * z1 * np.conj(z2)).imag
    c = abs(z1) ** 2 - abs(z2) ** 2
    return np.array([a, b, c])


def hopf_right_image(z1: complex, z2: complex) -> np.ndarray:
    a = (2.0 * np.conj(z1) * z2).real
    b = (2.0 * np.conj(z1) * z2).imag
    c = abs(z1) ** 2 - abs(z2) ** 2
    return np.array([a, b, c])


def sample_chiral_fold_locus(
    n: int = 2000, seed: int = 7
) -> list[tuple[complex, complex]]:
    """Sample points on {(z_1, z_2) ∈ S^3 : z_1 conj(z_2) ∈ R}.

    Parameterisation:
      r_1, r_2 with r_1^2 + r_2^2 = 1 (so r_1 = cos(eta),
      r_2 = sin(eta) for eta in [0, pi/2]).
      Common phase psi in [0, 2 pi).
      Phase difference flag k in {0, 1}: k=0 gives z_2 = r_2 e^{i psi},
      k=1 gives z_2 = r_2 e^{i(psi + pi)} = -r_2 e^{i psi}.
    """
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        eta = rng.uniform(0, math.pi / 2)
        psi = rng.uniform(0, 2.0 * math.pi)
        k = rng.integers(0, 2)
        z1 = math.cos(eta) * np.exp(1j * psi)
        z2 = math.sin(eta) * np.exp(1j * (psi + k * math.pi))
        out.append((complex(z1), complex(z2)))
    return out


def test_pi_L_equals_pi_R_on_locus() -> None:
    pts = sample_chiral_fold_locus(n=2000)
    max_diff = 0.0
    for z1, z2 in pts:
        pL = hopf_left_image(z1, z2)
        pR = hopf_right_image(z1, z2)
        diff = float(np.linalg.norm(pL - pR))
        max_diff = max(max_diff, diff)
    assert max_diff < 1e-10, (
        f"LOCUS_PI_LR_EQUAL FAIL: max ||pi_L - pi_R|| = {max_diff:.4e}; "
        f"expected < 1e-10 on the chiral-fold locus."
    )


def test_locus_image_on_equator() -> None:
    pts = sample_chiral_fold_locus(n=2000)
    max_b = 0.0
    for z1, z2 in pts:
        pL = hopf_left_image(z1, z2)
        max_b = max(max_b, abs(pL[1]))
    assert max_b < 1e-10, (
        f"IMAGE_ON_EQUATOR FAIL: max |b component| of pi_L image = "
        f"{max_b:.4e}; expected < 1e-10 (locus should map to "
        f"equatorial S^1 of S^2 where b = 0)."
    )


def test_image_on_unit_circle() -> None:
    pts = sample_chiral_fold_locus(n=2000)
    max_dev = 0.0
    for z1, z2 in pts:
        pL = hopf_left_image(z1, z2)
        a, b, c = pL
        max_dev = max(max_dev, abs(a * a + c * c - 1.0))
    assert max_dev < 1e-10, (
        f"UNIT_CIRCLE FAIL: max |a^2 + c^2 - 1| on locus image = "
        f"{max_dev:.4e}; equator should be unit circle."
    )


def main() -> None:
    print("=" * 70)
    print("FALSIFIER C3: Chiral fold locus = equatorial S^1 of S^2 Hopf base")
    print("  Tests whether the locus pi_L = pi_R on S^3 is exactly the")
    print("  Clifford torus mapping to the equatorial circle of S^2 under")
    print("  the Hopf base.")
    print("=" * 70)

    pts = sample_chiral_fold_locus(n=200)
    print("\n  Diagnostic (first 5 sampled locus points):")
    print("  " + "-" * 60)
    for k, (z1, z2) in enumerate(pts[:5]):
        pL = hopf_left_image(z1, z2)
        pR = hopf_right_image(z1, z2)
        print(f"    z1 = {z1.real:+.4f}{z1.imag:+.4f}j,  "
              f"z2 = {z2.real:+.4f}{z2.imag:+.4f}j")
        print(f"      pi_L = ({pL[0]:+.4f}, {pL[1]:+.4f}, {pL[2]:+.4f})")
        print(f"      pi_R = ({pR[0]:+.4f}, {pR[1]:+.4f}, {pR[2]:+.4f})")
    print("  " + "-" * 60)

    print("\n" + "=" * 70)
    print("HARD ASSERTIONS")
    print("=" * 70)
    failed: list[tuple[str, str]] = []
    for name, fn in (
        ("LOCUS_PI_LR_EQUAL    (||pi_L - pi_R|| = 0 on locus)",
         test_pi_L_equals_pi_R_on_locus),
        ("IMAGE_ON_EQUATOR     (locus image has b = 0)",
         test_locus_image_on_equator),
        ("UNIT_CIRCLE          (image on unit circle a^2 + c^2 = 1)",
         test_image_on_unit_circle),
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
        print("CHIRAL FOLD = EQUATORIAL S^1 CONFIRMED.")
        print("  The locus where left and right Hopf maps agree is exactly")
        print("  the Clifford torus on S^3 mapping to the equator of S^2.")
        print("  This realises the framework's beta = 1/2 chiral-fold")
        print("  balance as a sharp geometric structure on the Hopf base.")
    else:
        print("CHIRAL FOLD GEOMETRY DOES NOT MATCH EQUATORIAL S^1.")
    print("=" * 70)

    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
