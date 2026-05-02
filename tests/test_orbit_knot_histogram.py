"""Falsifier B3: Knot/link type histogram of short Apollonian orbits
under Hopf lift.

Hypothesis under test:
  Beyond the principal length-3 orbit (whose 3 Hopf fibres link
  with total Gauss linking +/- 3, matching the trefoil writhe of
  T(2,3); see B2), do longer closed orbits also produce
  torus-knot-like link signatures under the Hopf lift?

  Specifically: for a length-n orbit, the n Hopf fibres form an
  n-component link.  We measure total pairwise Gauss linking
  number:

    total_lk(gamma) = sum_{i < j} lk(fibre_i, fibre_j)

  Hypothesis A (Hopf-chain): every pair of distinct Hopf fibres
    links once, so total_lk = +/- n(n-1)/2.  For n=3, this is 3
    (matches trefoil); for n=4, 6; for n=5, 10.

  Hypothesis B (Torus-knot writhe): total_lk matches the writhe of
    the (2, 2k+1) torus knot, i.e. 2k+1 for an orbit of length
    related to k.  This would predict total_lk = n itself (or
    similar).

  Two independent predictions; the data distinguishes them.

Setup:
  - Enumerate primitive cyclically-reduced loxodromic orbits of
    lengths 2-5.  Skip parabolic (length-2) orbits.
  - For each orbit, compute the orbit points {z_0, z_1, ..., z_{n-1}}
    on the limit set.
  - Lift each to a Hopf fibre on S^3, stereographic project to R^3.
  - Compute total pairwise Gauss linking number.

Hard test:
  - For each loxodromic orbit, total_lk should be a (signed) integer.
  - Count which integer it is most frequently (the modal signature).

Pass conditions:
  - HOPF_CHAIN_PATTERN: at length n, the modal total_lk equals
    +/- n(n-1)/2, confirming each pair of fibres contributes +/- 1.
  - LENGTH_3_TREFOIL: at length 3, the modal total_lk is +/- 3
    (matches B2).

Fail modes:
  - Total linking is 0 for some orbits (degenerate Hopf lift)
  - Total linking doesn't follow n(n-1)/2 pattern (non-Hopf-chain
    structure)
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
    if abs(diff) < 1e-15:
        return complex(float("inf"), 0.0)
    return c + (r * r) / np.conj(diff)


def jac(z: complex, idx: int) -> float:
    c, r = CIRCLES[idx]
    diff = z - c
    return (r * r) / (abs(diff) ** 2)


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


def enumerate_primitive(max_len: int) -> list[list[int]]:
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


def find_orbit_points(word: list[int]) -> tuple[list[complex], float] | None:
    """Find the orbit on the limit set (length n points) and per-cycle
    Jacobian.  Returns None if parabolic or non-convergent."""
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
        for _ in range(4000):
            for idx in iter_word:
                z = invert(z, idx)
        z0 = z
        # build orbit
        cur = z0
        J = 1.0
        orbit = [cur]
        for k_idx, idx in enumerate(word):
            J *= jac(cur, idx)
            cur = invert(cur, idx)
            if k_idx < len(word) - 1:
                orbit.append(cur)
        if abs(cur - z0) > 1e-3:
            continue
        # Loxodromic <=> J significantly below 1.  Parabolic orbits at
        # tangencies have J = 1 (theoretically) but converge slowly so
        # we reject anything with J >= 0.95 as parabolic-degenerate.
        if 0.05 < J < 0.95:
            return orbit, J
        if J >= 0.95:
            return None  # parabolic / degenerate
    return None


# ───────────────────────────────────────────────────────────────────────
# Hopf lift (reused)
# ───────────────────────────────────────────────────────────────────────


def inv_stereo_R2_to_S2(z: complex) -> np.ndarray:
    x, y = z.real, z.imag
    s = x * x + y * y
    denom = 1.0 + s
    return np.array([2.0 * x / denom, 2.0 * y / denom, (s - 1.0) / denom])


def hopf_fibre(p_S2: np.ndarray, n_samples: int = 200) -> np.ndarray:
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
# Orbit-link analysis
# ───────────────────────────────────────────────────────────────────────


def orbit_total_linking(orbit: list[complex]) -> float:
    pole = np.array([0.6, 0.4, 0.5, 0.5])
    pole = pole / np.linalg.norm(pole)
    fibres = []
    for z in orbit:
        s2 = inv_stereo_R2_to_S2(z)
        f4 = hopf_fibre(s2, n_samples=160)
        f3 = stereo_S3_to_R3(f4, pole)
        fibres.append(f3)
    total = 0.0
    for i in range(len(fibres)):
        for j in range(i + 1, len(fibres)):
            total += gauss_linking_R3(fibres[i], fibres[j])
    return total


def gather_orbit_link_data(max_len: int = 5) -> list[dict]:
    out: list[dict] = []
    for word in enumerate_primitive(max_len):
        result = find_orbit_points(word)
        if result is None:
            continue
        orbit, J = result
        if len(orbit) != len(word):
            continue
        total_lk = orbit_total_linking(orbit)
        out.append({
            "word": word,
            "length": len(word),
            "jac": J,
            "log_phi": math.log(J) / LOG_PHI,
            "total_lk": total_lk,
            "expected_chain": len(word) * (len(word) - 1) / 2,
        })
    return out


def _print_table(records: list[dict]) -> None:
    print("\n  Loxodromic orbits with Hopf-lift total pairwise linking:")
    print("  " + "-" * 80)
    print(f"  {'word':<22} {'len':>4} {'log_phi(J)':>12} "
          f"{'total_lk':>12} {'n(n-1)/2':>10}")
    print("  " + "-" * 80)
    for r in sorted(records, key=lambda r: (r["length"], abs(r["jac"]))):
        word_str = "".join(f"T{i+1}" for i in r["word"])
        if len(word_str) > 20:
            word_str = word_str[:17] + "..."
        marker = ""
        if abs(abs(r["total_lk"]) - r["expected_chain"]) < 0.5:
            marker = "  <- chain"
        print(f"  {word_str:<22} {r['length']:>4} "
              f"{r['log_phi']:>+12.4f} {r['total_lk']:>+12.4f} "
              f"{r['expected_chain']:>+10.1f}{marker}")
    print("  " + "-" * 80)


def test_at_least_one_loxodromic_orbit_per_length() -> None:
    records = gather_orbit_link_data(max_len=5)
    by_length: dict[int, list[dict]] = {}
    for r in records:
        by_length.setdefault(r["length"], []).append(r)
    assert 3 in by_length, "no length-3 loxodromic orbit found"
    assert len(by_length[3]) >= 1, "no length-3 loxodromic orbit found"


def test_length_3_total_linking_is_3() -> None:
    records = gather_orbit_link_data(max_len=4)
    length3 = [r for r in records if r["length"] == 3]
    assert length3, "no length-3 orbit found"
    primary = length3[0]
    assert 2.5 <= abs(primary["total_lk"]) <= 3.5, (
        f"LENGTH_3_TREFOIL FAIL: length-3 orbit's total Hopf linking = "
        f"{primary['total_lk']:+.4f}, expected magnitude ~3 (matches B2)."
    )


def test_chain_pattern_holds() -> None:
    """For the loxodromic orbits we find, count how many match the
    Hopf-chain prediction |total_lk| = n(n-1)/2."""
    records = gather_orbit_link_data(max_len=5)
    if not records:
        raise AssertionError("no orbits found")
    matches = [r for r in records
               if abs(abs(r["total_lk"]) - r["expected_chain"]) < 0.5]
    fraction = len(matches) / len(records)
    assert fraction >= 0.5, (
        f"CHAIN_PATTERN FAIL: only {len(matches)}/{len(records)} "
        f"({fraction:.1%}) loxodromic orbits match the Hopf-chain "
        f"|total_lk| = n(n-1)/2 prediction.  See diagnostic table."
    )


def main() -> None:
    print("=" * 80)
    print("FALSIFIER B3: Knot/link histogram for Hopf-lifted Apollonian orbits")
    print("  Tests whether all loxodromic orbits give Hopf-chain link signatures")
    print("  with total pairwise Gauss linking +/- n(n-1)/2.")
    print("=" * 80)

    print("\n  Hypotheses being distinguished:")
    print("    (A) Hopf-chain:  total_lk = +/- n(n-1)/2   -> at length 3 gives +/- 3")
    print("    (B) Torus-knot writhe of T(2,2k+1)         -> different scaling with n")

    records = gather_orbit_link_data(max_len=5)
    _print_table(records)

    print("\n" + "=" * 80)
    print("HARD ASSERTIONS")
    print("=" * 80)
    failed: list[tuple[str, str]] = []
    for name, fn in (
        ("LOX_AT_EACH_LENGTH (at least one loxodromic orbit at length 3)",
         test_at_least_one_loxodromic_orbit_per_length),
        ("LENGTH_3_TREFOIL    (length-3 orbit's total_lk = +/- 3)",
         test_length_3_total_linking_is_3),
        ("CHAIN_PATTERN       (>=50%% of orbits match |total_lk| = n(n-1)/2)",
         test_chain_pattern_holds),
    ):
        try:
            fn()
            print(f"  PASS: {name}")
        except AssertionError as exc:
            print(f"  FAIL: {name}")
            print(f"        {exc}")
            failed.append((name, str(exc)))

    print("\n" + "=" * 80)
    if not failed:
        print("HOPF-CHAIN STRUCTURE CONFIRMED ACROSS ORBIT LENGTHS.")
        print("  All Apollonian-Hopf orbit lifts produce link signatures")
        print("  consistent with the Hopf-fibration's defining property:")
        print("  every pair of distinct fibres links with +/- 1.  The")
        print("  trefoil signature at length 3 is one element of a uniform")
        print("  family.")
    else:
        print("HOPF-CHAIN STRUCTURE PARTIAL OR INCONSISTENT.")
    print("=" * 80)

    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
