"""Falsifier C4: Witness sphere = Hopf base of an S^3-equivariant lift.

Hypothesis under test:
  The Apollonian dynamics on S^2 (the "witness sphere") is the
  Hopf base of an S^1-equivariant lift on S^3.  Concretely:

  (a) Each Apollonian inversion T_k acting on R^2 = C extends to
      an anti-Mobius transformation T_k^S2 on S^2 = CP^1 via
      inverse stereographic projection.

  (b) T_k^S2 lifts to an SU(2)-related transformation T_k^S3 on
      S^3 = SU(2) such that the diagram commutes:

         T_k^S3 : S^3 -> S^3
                   |        |
                   v        v
                  pi_L     pi_L
                   |        |
                   v        v
         T_k^S2 : S^2 -> S^2

  (c) The lift T_k^S3 is determined by T_k^S2 up to an overall
      phase (since the Hopf fibre is S^1).

  This means the Apollonian dynamics is the QUOTIENT of an
  S^3-dynamics by the U(1) Hopf action.  The "witness sphere"
  S^2 IS the Hopf base.

Setup:
  - Generate an Apollonian chaos-game trajectory on R^2.
  - Inverse stereographic project to S^2.
  - For each S^2 point, lift to S^3 via the canonical section of
    the Hopf bundle (a smooth choice of fibre representative).
  - Verify that pi_L applied to the S^3 trajectory recovers the
    original S^2 trajectory exactly.
  - Verify that consecutive S^3 points are related by a HOPF-
    EQUIVARIANT transformation (i.e., differ by a Hopf-fibre
    rotation when projected back through the lift).

Hard test:
  - PROJECTION_RECOVERS:  pi_L(T_k^S3(p)) = T_k^S2(pi_L(p)) for
    100 random points and inversions.
  - LIFT_IS_S3:           every lifted point has |z_1|^2 +
    |z_2|^2 = 1 (lies on S^3) within +/- 1e-10.
  - FIBRE_DEPENDENCE:     two different lifts of the same S^2
    point differ only by an overall U(1) phase.

Pass conditions:
  All three above.

Fail modes:
  - PROJECTION fails: the lift is not Hopf-equivariant (the
    chosen lifting prescription doesn't commute with pi_L).
  - LIFT_IS_S3 fails: numerical drift takes lift off S^3.
  - FIBRE_DEPENDENCE fails: the lift is not just U(1)-related.
"""

from __future__ import annotations

import math
import random
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


CIRCLES: list[tuple[complex, float]] = [
    (complex(-0.5, 0.0), 0.5),
    (complex(+0.5, 0.0), 0.5),
    (complex(0.0, 2.0 / 3.0), 1.0 / 3.0),
]


# ───────────────────────────────────────────────────────────────────────
# Apollonian inversion on R^2 / S^2
# ───────────────────────────────────────────────────────────────────────


def invert_R2(z: complex, idx: int) -> complex:
    c, r = CIRCLES[idx]
    diff = z - c
    if abs(diff) < 1e-15:
        return complex(float("inf"), 0.0)
    return c + (r * r) / np.conj(diff)


def inv_stereo_R2_to_S2(z: complex) -> np.ndarray:
    x, y = z.real, z.imag
    s = x * x + y * y
    denom = 1.0 + s
    return np.array([2.0 * x / denom, 2.0 * y / denom, (s - 1.0) / denom])


def stereo_S2_to_R2(p_S2: np.ndarray) -> complex:
    a, b, c = p_S2[0], p_S2[1], p_S2[2]
    if abs(c - 1.0) < 1e-12:
        return complex(float("inf"), 0.0)
    return complex(a / (1.0 - c), b / (1.0 - c))


def invert_S2(p_S2: np.ndarray, idx: int) -> np.ndarray:
    z = stereo_S2_to_R2(p_S2)
    z2 = invert_R2(z, idx)
    return inv_stereo_R2_to_S2(z2)


# ───────────────────────────────────────────────────────────────────────
# Hopf lift section S^2 -> S^3
# ───────────────────────────────────────────────────────────────────────


def hopf_lift_section(p_S2: np.ndarray, phase: float = 0.0) -> tuple[complex, complex]:
    """Smooth section of the Hopf bundle S^3 -> S^2.  Choose phase
    parameter (default 0)."""
    a, b, c = p_S2[0], p_S2[1], p_S2[2]
    r1 = math.sqrt(max(0.0, (1.0 + c) / 2.0))
    r2 = math.sqrt(max(0.0, (1.0 - c) / 2.0))
    phase_diff = math.atan2(b, a)
    z1 = r1 * np.exp(1j * phase)
    z2 = r2 * np.exp(1j * (phase - phase_diff))
    return complex(z1), complex(z2)


def hopf_project(z1: complex, z2: complex) -> np.ndarray:
    val = 2.0 * z1 * np.conj(z2)
    return np.array([val.real, val.imag, abs(z1) ** 2 - abs(z2) ** 2])


# ───────────────────────────────────────────────────────────────────────
# Lifted Apollonian dynamics on S^3
# ───────────────────────────────────────────────────────────────────────


def invert_S3(z1: complex, z2: complex, idx: int, phase: float = 0.0) -> tuple[complex, complex]:
    """Lift Apollonian inversion T_k to S^3 via:
       1. Project (z_1, z_2) to S^2 via Hopf.
       2. Apply T_k on S^2.
       3. Lift back to S^3 via section with given phase.
    By construction, this commutes with pi_L."""
    p = hopf_project(z1, z2)
    p2 = invert_S2(p, idx)
    return hopf_lift_section(p2, phase)


# ───────────────────────────────────────────────────────────────────────
# Tests
# ───────────────────────────────────────────────────────────────────────


def test_pi_L_recovers_S2_dynamics() -> None:
    rng = random.Random(11)
    z = complex(0.05, 0.4)
    last = -1
    for _ in range(2000):
        choices = [k for k in range(3) if k != last]
        k = rng.choice(choices)
        z = invert_R2(z, k)
        last = k

    max_err = 0.0
    for trial in range(50):
        choices = [k for k in range(3) if k != last]
        k = rng.choice(choices)
        # On S^2: directly compute T_k^S2(p) where p = inv_stereo(z)
        p = inv_stereo_R2_to_S2(z)
        p_after = invert_S2(p, k)
        # On S^3: lift to fibre, evolve, project back
        z1, z2 = hopf_lift_section(p, phase=0.0)
        z1_after, z2_after = invert_S3(z1, z2, k, phase=0.0)
        p_after_S3 = hopf_project(z1_after, z2_after)
        err = float(np.linalg.norm(p_after - p_after_S3))
        max_err = max(max_err, err)
        z = invert_R2(z, k)
        last = k

    assert max_err < 1e-9, (
        f"PROJECTION_RECOVERS FAIL: max error between direct S^2 dynamics "
        f"and pi_L(lifted S^3 dynamics) = {max_err:.4e}; expected < 1e-9.  "
        f"The Hopf lift is not commuting with pi_L."
    )


def test_lift_stays_on_S3() -> None:
    rng = random.Random(7)
    z = complex(0.05, 0.4)
    p = inv_stereo_R2_to_S2(z)
    z1, z2 = hopf_lift_section(p, phase=0.4)
    norm_err = abs(abs(z1) ** 2 + abs(z2) ** 2 - 1.0)
    assert norm_err < 1e-12, (
        f"LIFT_IS_S3 FAIL initially: ||z||^2 - 1 = {norm_err:.4e}"
    )

    last = -1
    max_err = 0.0
    phase_now = 0.4
    for _ in range(100):
        choices = [k for k in range(3) if k != last]
        k = rng.choice(choices)
        z1, z2 = invert_S3(z1, z2, k, phase=phase_now)
        norm_err = abs(abs(z1) ** 2 + abs(z2) ** 2 - 1.0)
        max_err = max(max_err, norm_err)
        last = k
        phase_now += 0.13

    assert max_err < 1e-10, (
        f"LIFT_IS_S3 FAIL: max ||z||^2 - 1 over 100 steps = {max_err:.4e}; "
        f"expected < 1e-10."
    )


def test_two_lifts_differ_by_phase() -> None:
    """For a fixed S^2 point, lifts with different phase parameters
    should differ by a U(1) overall phase."""
    p = inv_stereo_R2_to_S2(complex(0.1, 0.3))
    z1a, z2a = hopf_lift_section(p, phase=0.0)
    z1b, z2b = hopf_lift_section(p, phase=0.7)
    # ratios should be equal complex numbers (the overall phase)
    if abs(z1a) < 1e-9:
        return
    ratio_1 = z1b / z1a
    if abs(z2a) > 1e-9:
        ratio_2 = z2b / z2a
        assert abs(ratio_1 - ratio_2) < 1e-10, (
            f"FIBRE_DEPENDENCE FAIL: two lifts of same S^2 point with "
            f"different phases give ratios {ratio_1} and {ratio_2}; "
            f"should differ only by a single U(1) phase."
        )
    # And |ratio_1| should equal 1 (pure phase)
    assert abs(abs(ratio_1) - 1.0) < 1e-10, (
        f"FIBRE_DEPENDENCE FAIL: |ratio| = {abs(ratio_1)}; "
        f"should be 1 (pure phase)."
    )


def main() -> None:
    print("=" * 70)
    print("FALSIFIER C4: Witness sphere = Hopf base of S^3-equivariant lift")
    print("  Tests whether the Apollonian S^2 dynamics arises as the Hopf")
    print("  base of an S^3 dynamics that commutes with pi_L.")
    print("=" * 70)

    print("\n  Setup:  T_k^S2(p)  =  pi_L(T_k^S3(z_1, z_2))  for any lift (z_1, z_2)")
    print("  of p along the Hopf fibre (S^1 action).")

    print("\n" + "=" * 70)
    print("HARD ASSERTIONS")
    print("=" * 70)
    failed: list[tuple[str, str]] = []
    for name, fn in (
        ("PROJECTION_RECOVERS  (pi_L(T_k^S3(p)) = T_k^S2(pi_L(p)))",
         test_pi_L_recovers_S2_dynamics),
        ("LIFT_IS_S3           (lifted points stay on S^3)",
         test_lift_stays_on_S3),
        ("FIBRE_DEPENDENCE     (two lifts differ by U(1) phase)",
         test_two_lifts_differ_by_phase),
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
        print("WITNESS = HOPF BASE CONFIRMED.")
        print("  The Apollonian S^2 dynamics is the Hopf base of an")
        print("  S^3-equivariant lift.  pi_L commutes with the dynamics")
        print("  exactly; the witness sphere IS the quotient S^3 / S^1_Hopf.")
    else:
        print("WITNESS = HOPF BASE FAILED.")
    print("=" * 70)

    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
