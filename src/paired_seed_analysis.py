"""Paired multi-seed comparison: F4 vs visual_only on s-Aff-Wild2 MTL.

Usage:
    python -m src.paired_seed_analysis

Reads results/<run>/smoothing.json for visual_only and the F4 variants
at seeds {42, 0, 1, 2}, computes paired deltas (F4 - visual_only) at each
seed, and reports:
  - mean +/- s.d.
  - per-seed delta and sign count
  - paired-bootstrap 95% CI on the mean delta (10 000 resamples)
  - Wilcoxon signed-rank test
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

import numpy as np
from scipy import stats

RESULTS_ROOT = Path(__file__).resolve().parents[1] / "results"
METRIC = "P_MTL@0.5"  # smoothed composite

SEEDS = [42, 0, 1, 2, 3, 4]

VISUAL_RUNS = {
    42: "stage3_visual_only",
    0: "stage3_visual_only_seed0",
    1: "stage3_visual_only_seed1",
    2: "stage3_visual_only_seed2",
    3: "stage3_visual_only_seed3",
    4: "stage3_visual_only_seed4",
}

F4_W5_RUNS = {
    42: "stage3_f4_xattn",
    0: "stage3_f4_xattn_seed0",
    1: "stage3_f4_xattn_seed1",
    2: "stage3_f4_xattn_seed2",
    3: "stage3_f4_xattn_seed3",
    4: "stage3_f4_xattn_seed4",
}

F4_W10_RUNS = {
    42: "stage3_f4_xattn_w10",
    0: "stage3_f4_xattn_w10_seed0",
    1: "stage3_f4_xattn_w10_seed1",
    2: "stage3_f4_xattn_w10_seed2",
    3: "stage3_f4_xattn_w10_seed3",
    4: "stage3_f4_xattn_w10_seed4",
}


def load_smoothed(run_name: str) -> float:
    path = RESULTS_ROOT / run_name / "smoothing.json"
    with path.open() as f:
        data = json.load(f)
    return float(data["best"][METRIC])


def load_raw(run_name: str) -> float:
    path = RESULTS_ROOT / run_name / "smoothing.json"
    with path.open() as f:
        data = json.load(f)
    return float(data["raw"][METRIC])


def bootstrap_mean_ci(
    deltas: Sequence[float],
    n_resamples: int = 10_000,
    alpha: float = 0.05,
    rng_seed: int = 0,
) -> tuple[float, float]:
    rng = np.random.default_rng(rng_seed)
    arr = np.asarray(deltas, dtype=float)
    n = len(arr)
    boot_means = np.empty(n_resamples)
    for i in range(n_resamples):
        boot_means[i] = arr[rng.integers(0, n, size=n)].mean()
    lo = float(np.quantile(boot_means, alpha / 2))
    hi = float(np.quantile(boot_means, 1 - alpha / 2))
    return lo, hi


def report(label: str, treated: dict[int, str], control: dict[int, str]) -> None:
    print(f"\n=== {label} ===")
    print(f"{'seed':>5} {'F4':>9} {'visual':>9} {'delta':>9}")
    deltas = []
    for s in SEEDS:
        t = load_smoothed(treated[s])
        c = load_smoothed(control[s])
        d = t - c
        deltas.append(d)
        print(f"{s:>5} {t:>9.4f} {c:>9.4f} {d:>+9.4f}")

    deltas = np.array(deltas)
    mean = float(deltas.mean())
    sd = float(deltas.std(ddof=1))
    n_pos = int((deltas > 0).sum())
    lo, hi = bootstrap_mean_ci(deltas)

    # Wilcoxon signed-rank (two-sided). With n=4 the test has very low power;
    # report it for completeness alongside the bootstrap CI and sign count.
    try:
        w_stat, w_p = stats.wilcoxon(deltas, alternative="two-sided")
    except ValueError as e:
        w_stat, w_p = float("nan"), float("nan")
        print(f"  Wilcoxon failed: {e}")

    print(f"\n  mean delta:        {mean:+.4f}")
    print(f"  s.d. (n=4):        {sd:.4f}")
    print(f"  paired-bootstrap   95% CI on mean: [{lo:+.4f}, {hi:+.4f}]")
    print(f"  seeds with delta > 0: {n_pos}/{len(deltas)}")
    print(f"  Wilcoxon two-sided: W={w_stat:.3f}  p={w_p:.4f}")


def main() -> None:
    print(f"Paired analysis: metric = {METRIC}, smoothed (best (sigma, delta) per checkpoint)")
    print(f"Seeds: {SEEDS}")
    report("F4 (w=5)  vs  visual_only", F4_W5_RUNS, VISUAL_RUNS)
    report("F4 (w=10) vs  visual_only", F4_W10_RUNS, VISUAL_RUNS)


if __name__ == "__main__":
    main()
