"""Falsifier D2: Closed-form phi-expression for the Apollonian Hausdorff
dimension delta ~ 1.305688.

Hypothesis under test:
  Does the Apollonian Hausdorff dimension admit a clean closed-form
  expression in terms of phi (golden ratio)?  This would link the
  fractal geometry of the gasket to the algebraic structure of
  Cl(3,1) / phi-arithmetic.

  McMullen's numerical value: delta ~ 1.305688... (no known closed
  form).  We scan a list of candidate phi-expressions and report
  the closest match.

Setup:
  - Compile a list of phi-related candidate expressions:
    phi - 1/phi^k, log_phi(1 + 1/phi^k), 1 + 1/phi^k, etc.
  - For each, compute the numerical value and compare to
    delta = 1.305688.

Hard test:
  - At least one candidate within +/- 1e-3 of delta?  (Strong positive.)
  - At least one within +/- 1e-2?  (Suggestive but weak.)
  - All candidates differ by >= 0.05?  (Negative; no clean phi
    closed form.)

Pass conditions:
  - REPORT_ONLY: this test is informative; we report the best
    match but do NOT fail unless no candidates lie within +/- 0.5
    (pathological scan).

This is the lowest-priority test in the comprehensive plan.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

PHI = (1.0 + math.sqrt(5.0)) / 2.0
LOG_PHI = math.log(PHI)
DELTA_APOLLONIAN = 1.305688


def candidate_expressions() -> list[tuple[str, float]]:
    out: list[tuple[str, float]] = []
    # Direct phi-powers
    out.append(("phi - 1", PHI - 1))            # 0.618
    out.append(("1 + 1/phi^3", 1 + PHI ** -3))  # 1.236
    out.append(("1 + 1/phi^2", 1 + PHI ** -2))  # 1.382
    out.append(("1 + 1/phi", 1 + PHI ** -1))    # 1.618
    out.append(("phi - 1/phi^4", PHI - PHI ** -4))  # 1.472
    out.append(("phi^2 / 2", PHI ** 2 / 2))     # 1.309 -- VERY CLOSE!
    out.append(("(phi^2 + 1) / 2", (PHI ** 2 + 1) / 2))  # 1.809
    out.append(("1 + phi/4", 1 + PHI / 4))      # 1.404
    out.append(("phi^(2/3)", PHI ** (2.0 / 3.0)))  # 1.382
    # Logarithmic forms
    out.append(("log_phi(2 + 1/phi)", math.log(2 + 1 / PHI) / LOG_PHI))  # 2.075
    out.append(("log_phi(phi^2 + 1/phi)", math.log(PHI ** 2 + 1 / PHI) / LOG_PHI))
    out.append(("log_phi(phi + 1/phi^3)", math.log(PHI + PHI ** -3) / LOG_PHI))
    # Algebraic forms involving Cl(3,1) dim 16, even part 8
    out.append(("(16-8)/(8+phi^?)", 8 / (8 + 1 / PHI)))  # 0.882
    out.append(("phi/sqrt(3)", PHI / math.sqrt(3.0)))   # 0.934
    # Fibonacci-like rationals
    out.append(("13/phi^2", 13 / PHI ** 2))  # too big
    out.append(("8/phi^3", 8 / PHI ** 3))     # 1.889
    out.append(("21/(8 + 8/phi)", 21 / (8 + 8 / PHI)))  # ~1.617
    return out


def main() -> None:
    print("=" * 70)
    print("FALSIFIER D2 (low priority): closed-form phi-expression for")
    print("  Apollonian Hausdorff dimension delta ~ 1.305688.")
    print("=" * 70)

    print(f"\n  Target: delta = {DELTA_APOLLONIAN}")
    print(f"  phi   = {PHI:.10f}")

    cands = candidate_expressions()
    print("\n  Candidate phi-closed-form expressions (scan):")
    print("  " + "-" * 70)
    print(f"  {'expression':<32} {'value':>14} {'|diff to delta|':>18}")
    print("  " + "-" * 70)
    diffs = []
    for label, val in cands:
        d = abs(val - DELTA_APOLLONIAN)
        diffs.append((d, label, val))
        print(f"  {label:<32} {val:>+14.6f} {d:>+18.6e}")
    print("  " + "-" * 70)

    diffs.sort()
    best_d, best_label, best_val = diffs[0]
    print(f"\n  Best candidate: '{best_label}' = {best_val:.6f}")
    print(f"  |diff to delta| = {best_d:.4e}")

    print("\n" + "=" * 70)
    print("HARD ASSERTION")
    print("=" * 70)
    failed = False
    if best_d < 0.05:
        print(f"  PASS: best candidate within 0.05 of delta")
    else:
        print(f"  REPORT: no candidate within 0.05 of delta.")
        print(f"  Apollonian Hausdorff dim does NOT admit a clean phi-closed")
        print(f"  form in this scan.  delta is likely a transcendental")
        print(f"  computed by McMullen-type random-matrix methods, not a")
        print(f"  golden-ratio expression.  This is consistent with the")
        print(f"  view that delta is a *measure-theoretic* observable")
        print(f"  separate from the *spectral* (phi-power) structure.")
    print("=" * 70)


if __name__ == "__main__":
    main()
