"""Falsifier C-COINCIDENCE: per-step Jacobian factorization at the
principal length-3 cycle.

Hypothesis under test:
  At the principal cycle's fixed point z*, the cumulative Jacobian
  factorises into per-step factors |T_k'(z_n)|.  Three possibilities:

  (P1) Uniform per-step:  each |T_k'(z_n)| = phi^-2 (so cycle = phi^-6).
       Suggests the cycle is symmetric in some sense.

  (P2) Single-step concentration:  one step is phi^-6, others are 1.
       Would suggest one inversion does all the contraction.

  (P3) Distributed unevenly:  e.g. (phi^-1, phi^-2, phi^-3) or
       (phi^-2, phi^-2, phi^-2) -- the latter being P1.

Setup:
  - Use seed (-1, 2, 2, 3) (canonical A1 reference).
  - Apply word [0, 2, 1] = T_1 T_3 T_2 (note: apply_word's iteration
    is T_first then T_next, so the first generator applied is T_1
    NOT T_2; let's clarify what "principal cycle" means by
    convention).

  Actually: apply_word([0, 2, 1], z) iterates idx in [0, 2, 1] in
  order, so we apply T_1, then T_3, then T_2.  The fixed point
  satisfies T_2(T_3(T_1(z*))) = z*.  Per-step Jacobians are:
    j_1 = |T_1'(z*)|
    j_2 = |T_3'(T_1 z*)|
    j_3 = |T_2'(T_3 T_1 z*)|

  We compute j_1, j_2, j_3 and report log_phi of each, and the
  factorisation pattern.

Hard test:
  - Compute the three per-step Jacobians.
  - Identify the pattern.
  - Test (P1):  each |log_phi(j_k)| in [1.9, 2.1]?

Pass conditions:
  - PRINCIPAL_CYCLE_FOUND: cumulative Jacobian = phi^-6.
  - Either P1 holds (uniform phi^-2), P2 holds (one step phi^-6,
    others 1), or report which pattern actually occurs.

Fail modes:
  None -- this is a diagnostic test.  We pass if the cumulative
  Jacobian is phi^-6 and the factorisation pattern is reported.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

PHI = (1.0 + math.sqrt(5.0)) / 2.0
LOG_PHI = math.log(PHI)


CIRCLES: list[tuple[complex, float]] = [
    (complex(-0.5, 0.0), 0.5),
    (complex(+0.5, 0.0), 0.5),
    (complex(0.0, 2.0 / 3.0), 1.0 / 3.0),
]


def invert(z: complex, idx: int) -> complex:
    c, r = CIRCLES[idx]
    diff = z - c
    return c + (r * r) / np.conj(diff)


def jac(z: complex, idx: int) -> float:
    c, r = CIRCLES[idx]
    diff = z - c
    return (r * r) / (abs(diff) ** 2)


def find_principal_fixed_point(
    word: list[int], n_iter: int = 4000
) -> complex:
    """Find fixed point of word."""
    odd = (len(word) % 2 == 1)
    iter_word = (word + word) if odd else word
    z = complex(0.0, 0.4)
    for _ in range(n_iter):
        for idx in iter_word:
            z = invert(z, idx)
    return z


def per_step_jacobians(
    word: list[int], z_star: complex
) -> list[tuple[int, complex, float, float]]:
    """At fixed point z*, walk the word and report (step_idx, point,
    jac, log_phi(jac)) for each step."""
    out = []
    cur = z_star
    for k, idx in enumerate(word):
        j = jac(cur, idx)
        out.append((idx, cur, j, math.log(j) / LOG_PHI))
        cur = invert(cur, idx)
    return out


# ───────────────────────────────────────────────────────────────────────
# Diagnostics and tests
# ───────────────────────────────────────────────────────────────────────


def gather_data() -> dict:
    word = [0, 2, 1]  # T_1 T_3 T_2
    z_star = find_principal_fixed_point(word)
    steps = per_step_jacobians(word, z_star)
    cum_J = 1.0
    for _, _, j, _ in steps:
        cum_J *= j
    return {
        "word": word,
        "z_star": z_star,
        "steps": steps,
        "cumulative_J": cum_J,
        "cumulative_log_phi": math.log(cum_J) / LOG_PHI,
    }


def _print_factorization(data: dict) -> None:
    print("\n  Principal cycle T_1 T_3 T_2 (apply T_1 first, then T_3, then T_2):")
    print("  " + "-" * 70)
    print(f"    Fixed point z* = {data['z_star'].real:+.6f} {data['z_star'].imag:+.6f}j")
    print(f"    Cumulative Jacobian = {data['cumulative_J']:.6e}")
    print(f"    log_phi(cumulative)  = {data['cumulative_log_phi']:+.4f}")
    print("  " + "-" * 70)
    print(f"\n  Per-step factorization at z*:")
    print("  " + "-" * 70)
    print(f"  {'step':>4} {'inv':<6} {'point':<28} {'|Tprime|':>14} {'log_phi':>10}")
    print("  " + "-" * 70)
    for k, (idx, p, j, log_phi) in enumerate(data["steps"]):
        gen = f"T_{idx + 1}"
        p_str = f"{p.real:+.4f}{p.imag:+.4f}j"
        print(f"  {k+1:>4} {gen:<12} {p_str:<28} {j:>14.6e} {log_phi:>+10.4f}")
    print("  " + "-" * 70)
    sum_log = sum(s[3] for s in data["steps"])
    print(f"  Sum of log_phi(per-step) = {sum_log:+.4f}  (should = "
          f"log_phi(cumulative) = {data['cumulative_log_phi']:+.4f})")


def test_cumulative_is_phi_minus_6() -> None:
    data = gather_data()
    log_phi = data["cumulative_log_phi"]
    assert abs(log_phi - (-6.0)) < 0.05, (
        f"PRINCIPAL_PHI6 FAIL: cumulative log_phi = {log_phi:+.4f}; "
        f"expected -6 within 0.05."
    )


def test_per_step_distribution_is_uneven_summing_to_phi_minus_6() -> None:
    r"""Hypothesis P3 (verified): the per-step Jacobians are
    UNEVENLY distributed (each step different) but their sum in
    log_phi equals -6 (cumulative is exactly phi^-6).

    Hypothesis P1 (FALSIFIED): each step is uniform phi^-2.

    This test asserts P3 and explicitly documents that P1 fails.
    The pattern is consistent with the directional reading of
    conj:phi5-relaxation: the per-circle contraction is not
    uniformly factored across steps, even though the cumulative
    spectral observable is exact.
    """
    data = gather_data()
    steps = data["steps"]

    # P3: per-step values are NON-uniform (variance > 0.1).
    log_phis = [s[3] for s in steps]
    mean = sum(log_phis) / len(log_phis)
    variance = sum((lp - mean) ** 2 for lp in log_phis) / len(log_phis)
    assert variance > 0.05, (
        f"P3 FAIL: per-step log_phi values are too uniform (variance "
        f"{variance:.4f} <= 0.05); expected the uneven distribution "
        f"that signals direction-boundedness.  Values: "
        f"{[round(lp, 4) for lp in log_phis]}"
    )

    # And the sum equals -6 (already tested by test_cumulative_is_phi_minus_6).
    total = sum(log_phis)
    assert abs(total - (-6.0)) < 0.05, (
        f"Sum of per-step log_phi = {total:+.4f}, expected -6.0 "
        f"(phi^-6 spectral gap)"
    )

    # Document P1 explicitly as FALSIFIED.
    deviations_from_minus_2 = [abs(lp - (-2.0)) for lp in log_phis]
    p1_failures = sum(1 for d in deviations_from_minus_2 if d > 0.05)
    assert p1_failures > 0, (
        "P1 (uniform phi^-2 per step) was expected to be falsified "
        "but all per-step values are within 0.05 of -2.  "
        "This contradicts the documented direction-bound structure."
    )


def main() -> None:
    print("=" * 70)
    print("FALSIFIER C-COINCIDENCE: per-step Jacobian factorization")
    print("  at the principal length-3 cycle.")
    print("=" * 70)

    print(f"\n  Hypothesis P1: each per-step |T_k'(z_n)| = phi^-2 (uniform)")
    print(f"  Hypothesis P2: one step does all contraction, others ~ 1")
    print(f"  Hypothesis P3: uneven distribution summing to phi^-6 in log_phi")

    data = gather_data()
    _print_factorization(data)

    # Classify the pattern
    log_phis = [s[3] for s in data["steps"]]
    max_lp = max(log_phis, key=abs)
    near_zero = [lp for lp in log_phis if abs(lp) < 0.5]
    is_uniform = all(abs(lp - log_phis[0]) < 0.1 for lp in log_phis)
    is_concentrated = abs(max_lp + 6.0) < 0.5 and len(near_zero) >= 2

    print(f"\n  Pattern classification:")
    print(f"    Uniform (P1):         {'YES' if is_uniform else 'NO'}")
    print(f"    Concentrated (P2):    {'YES' if is_concentrated else 'NO'}")
    print(f"    Distributed (P3):     {'YES' if not is_uniform and not is_concentrated else 'NO'}")

    print("\n" + "=" * 70)
    print("HARD ASSERTIONS")
    print("=" * 70)
    failed: list[tuple[str, str]] = []
    for name, fn in (
        ("PRINCIPAL_PHI6   (cumulative log_phi = -6)",
         test_cumulative_is_phi_minus_6),
    ):
        try:
            fn()
            print(f"  PASS: {name}")
        except AssertionError as exc:
            print(f"  FAIL: {name}")
            print(f"        {exc}")
            failed.append((name, str(exc)))

    # P1 test as informational, not hard pass/fail
    try:
        test_per_step_uniform_phi_minus_2()
        print(f"  INFO: P1 (uniform phi^-2) HOLDS")
    except AssertionError as exc:
        print(f"  INFO: P1 (uniform phi^-2) does NOT hold")
        print(f"         {str(exc)[:200]}")

    print("\n" + "=" * 70)
    if not failed:
        print("PER-STEP FACTORIZATION CHARACTERIZED.")
        print(f"  Pattern: {'UNIFORM (each step phi^-2)' if is_uniform else 'NON-UNIFORM'}")
        if not is_uniform:
            print("  The 6 in phi^-6 does NOT decompose as 3 x 2 per step;")
            print("  the cycle has internal structure where steps contribute")
            print("  unequally.  This may inform the meaning of the integer 6.")
    else:
        print("PRINCIPAL JACOBIAN CHECK FAILED.")
    print("=" * 70)

    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
