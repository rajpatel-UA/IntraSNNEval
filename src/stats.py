#!/usr/bin/env python3
"""Canonical statistics for the paper's paired analyses. Single source of truth.

Every reported interval in the manuscript must come from here, so that different
sections cannot silently use different resampling conventions. The design is
recorded in `BOOTSTRAP_DESIGN` and emitted into every result payload, because a
future rerun that changes the resampling unit would produce a different-but-
plausible CI and nobody would notice.

Design decisions, fixed:
  * Resampling unit is the DATASET. Datasets are the generalisation units; seeds
    are within-dataset optimisation repetitions.
  * Hierarchical: each replicate samples datasets with replacement, then samples
    seeds with replacement *within* each drawn dataset. Seeds are nested.
  * The statistic is the mean over datasets of the per-dataset mean paired
    difference, so datasets are weighted equally regardless of seed count.
  * Percentile intervals (not BCa). Percentile is used because the statistic is a
    near-symmetric mean of few units and BCa's acceleration term is unstable at
    n=4 datasets.
  * All effects are in PERCENTAGE POINTS and signed as (arm - reference).
"""
from __future__ import annotations

from typing import Dict, List, Sequence

import numpy as np

N_BOOT = 10000
RNG_SEED = 20260812
ALPHA = 0.05

BOOTSTRAP_DESIGN = {
    "resampling_unit": "dataset (primary), seeds resampled within each drawn dataset",
    "hierarchy": "datasets sampled with replacement first, then seeds within dataset",
    "seeds_nested_within_datasets": True,
    "statistic": "mean over datasets of per-dataset mean paired difference",
    "n_bootstrap": N_BOOT,
    "rng_seed": RNG_SEED,
    "interval_type": "percentile",
    "alpha": ALPHA,
    "units": "percentage points (pp)",
    "sign_convention": "arm minus reference (e.g. LP/latency - LP/rate)",
}


def _stat(by_ds: Dict[str, np.ndarray]) -> float:
    return float(np.mean([np.mean(v) for v in by_ds.values()]))


def hierarchical_bootstrap(by_ds: Dict[str, Sequence[float]], n_boot: int = N_BOOT,
                           rng_seed: int = RNG_SEED, alpha: float = ALPHA) -> dict:
    """Paired differences grouped by dataset -> point estimate and percentile CI.

    by_ds maps dataset -> array of per-seed paired differences for that dataset.
    """
    keys = sorted(by_ds)
    arrs = {k: np.asarray(by_ds[k], dtype=float) for k in keys}
    est = _stat(arrs)
    rng = np.random.default_rng(rng_seed)
    reps = np.empty(n_boot)
    n_ds = len(keys)
    for b in range(n_boot):
        drawn = rng.integers(0, n_ds, size=n_ds)
        means = []
        for i in drawn:
            v = arrs[keys[i]]
            means.append(v[rng.integers(0, len(v), size=len(v))].mean())
        reps[b] = np.mean(means)
    lo, hi = np.percentile(reps, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {
        "estimate_pp": round(est, 4),
        "ci95_pp": [round(float(lo), 4), round(float(hi), 4)],
        "n_datasets": n_ds,
        "n_pairs_total": int(sum(len(v) for v in arrs.values())),
        "per_dataset_mean_pp": {k: round(float(np.mean(arrs[k])), 4) for k in keys},
        "bootstrap_p_two_sided": round(float(
            2 * min((reps <= 0).mean(), (reps >= 0).mean())), 5),
    }


def bonferroni(pvals: Sequence[float]) -> List[float]:
    m = len(pvals)
    return [min(1.0, p * m) for p in pvals]


def benjamini_hochberg(pvals: Sequence[float]) -> List[float]:
    """BH-adjusted p-values (step-up, monotone-enforced)."""
    p = np.asarray(pvals, dtype=float)
    m = len(p)
    order = np.argsort(p)
    adj = np.empty(m)
    prev = 1.0
    for rank in range(m - 1, -1, -1):
        i = order[rank]
        val = p[i] * m / (rank + 1)
        prev = min(prev, val)
        adj[i] = min(1.0, prev)
    return [float(x) for x in adj]


def paired_by_dataset(arm: Dict[str, Dict[int, float]],
                      ref: Dict[str, Dict[int, float]]) -> Dict[str, np.ndarray]:
    """{ds: {seed: value}} x2 -> {ds: array of (arm - ref) over shared seeds}."""
    out = {}
    for ds in sorted(set(arm) & set(ref)):
        seeds = sorted(set(arm[ds]) & set(ref[ds]))
        if seeds:
            out[ds] = np.array([arm[ds][s] - ref[ds][s] for s in seeds], dtype=float)
    return out
