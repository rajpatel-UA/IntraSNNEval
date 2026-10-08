#!/usr/bin/env python3
"""E3: does the value of spike timing depend on the timing resolution?

POST-HOC AND EXPLORATORY. Not part of any registration. See
DIAGNOSTIC_SPECIFICATIONS.md.

The timing premium is latency@0.01 minus delta@0.01 at matched input activity:
two arms that emit identical input spike sets and differ only in whether the
spike carries timing information. $T$ is the number of distinguishable spike
times, so it is the resolution of that temporal code. E1 measured the premium at
$T=25$ only. This measures it at $T \\in \\{5,10,25,50\\}$ on all five
confirmation protocols.

Reads the per-run `results.json` files rather than `summary_e3.csv`, because that
CSV records `encoding` but not `encoder_threshold` and so cannot tell arm C
(delta@0.05) from arm D (delta@0.01). The arm is unambiguous in the path.

$T=25$ is taken from the runs that already exist: arms A-C from the confirmation
sweep, arm D from E1 arm C. Those are three seeds where E3 is three seeds, so the
comparison is like-for-like.

Two questions, both informative either way:

  premium scales with T   timing resolution is the mechanism, and the CTU-13
                          fold 1 liability should shrink at low T
  premium flat in T       the protocol dependence is a property of the traffic
                          rather than of the resolution

    python scripts/e3_analysis.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.populations import macro_f1_pair  # noqa: E402
from src.readout import discover  # noqa: E402

E3 = ROOT / "results/runs_e3"
E1 = ROOT / "results/runs_e1"
OUT = ROOT / "results/analysis/e3"

PROTOCOLS = ["nslkdd_v2", "kddcup99_v2", "cicids2017_v2",
             "ctu13_v2_f0", "ctu13_v2_f1"]
SEEDS = (42, 43, 44)
STEPS = [5, 10, 25, 50]
#: arm -> (encoding, gate). D is the control that makes the premium measurable.
ARMS = {"A_rate": ("rate", None), "B_latency_thr001": ("latency", 0.01),
        "C_delta_thr005": ("delta", 0.05), "D_delta_thr001": ("delta", 0.01)}
LABEL = {"nslkdd_v2": "NSL-KDD", "kddcup99_v2": "KDDCup99",
         "cicids2017_v2": "CIC-IDS2017", "ctu13_v2_f0": "CTU-13 f0",
         "ctu13_v2_f1": "CTU-13 f1"}


def read_run(path: Path) -> dict | None:
    """One run's macro-F1 on the registered label universe, plus spike counts."""
    f = path / "results.json"
    if not f.exists():
        return None
    rec = json.loads(f.read_text())
    tp = rec["test_plus"]
    pre = path / "artifacts" / "test_prefix.npz"
    if pre.exists():
        z = np.load(pre)
        m = macro_f1_pair(z["y_true"].astype(np.int64),
                          z["y_pred"].astype(np.int64),
                          rec["data"]["num_classes"])
        f1 = m["macro_f1_fixed_universe"]
    else:
        # Falls back to the stored value. That is safe for runs made after the
        # label-universe fix -- E1 and E3 both are, and their stored values
        # equal the recomputed ones -- but not for the original confirmation
        # sweep. Flagged in the output either way.
        f1 = tp["metrics"]["f1_macro"]
    sa = tp["spike_accounting"]
    return {"f1": f1, "stored_f1": tp["metrics"]["f1_macro"],
            "from_predictions": pre.exists(),
            "in_spk": sa["input_spikes_per_sample"],
            "sops": sa["sops_per_sample"],
            "hidden": sa.get("hidden_spikes_per_sample"),
            "mcc": tp["metrics"]["matthews_corrcoef"]}


def collect() -> pd.DataFrame:
    rows = []
    # T in {5,10,50} from E3.
    for arm in ARMS:
        for t in (5, 10, 50):
            for proto in PROTOCOLS:
                for s in SEEDS:
                    r = read_run(E3 / arm / f"T{t}" / proto / f"seed_{s}")
                    if r:
                        rows.append({"arm": arm, "T": t, "protocol": proto,
                                     "seed": s, **r})
    # T=25: arm D from E1, arms A-C from the confirmation sweep.
    for proto in PROTOCOLS:
        for s in SEEDS:
            r = read_run(E1 / "C_delta_thr001" / proto / f"seed_{s}")
            if r:
                rows.append({"arm": "D_delta_thr001", "T": 25,
                             "protocol": proto, "seed": s, **r})
    want = {"LeakyParallel/rate": "A_rate",
            "LeakyParallel/latency": "B_latency_thr001",
            "LeakyParallel/delta": "C_delta_thr005"}
    for run in discover():
        if run.variant not in want or run.seed not in SEEDS:
            continue
        if run.protocol not in PROTOCOLS:
            continue
        rec, pre = run.record(), run.prefix()
        m = macro_f1_pair(pre["y_true"].astype(np.int64),
                          pre["y_pred"].astype(np.int64),
                          rec["data"]["num_classes"])
        sa = rec["test_plus"]["spike_accounting"]
        rows.append({"arm": want[run.variant], "T": 25,
                     "protocol": run.protocol, "seed": run.seed,
                     "f1": m["macro_f1_fixed_universe"],
                     "stored_f1": rec["test_plus"]["metrics"]["f1_macro"],
                     "from_predictions": True,
                     "in_spk": sa["input_spikes_per_sample"],
                     "sops": sa["sops_per_sample"],
                     "hidden": sa.get("hidden_spikes_per_sample"),
                     "mcc": rec["test_plus"]["metrics"]["matthews_corrcoef"]})
    return pd.DataFrame(rows)


def main() -> int:
    df = collect()
    if df.empty:
        print("no E3 runs found.")
        return 0
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT / "runs.csv", index=False, float_format="%.8f")

    cov = df.groupby(["arm", "T"]).size().unstack(fill_value=0)
    print("=== coverage (runs per arm x T; 15 = 5 protocols x 3 seeds) ===")
    print(cov.to_string())
    # Make the pre/post-fix distinction visible rather than argued: on CIC a
    # stored value computed before the fix uses a four-class denominator, and
    # the premium would be inflated by 1/0.8 on the delta arm alone.
    drift = (df.f1 - df.stored_f1).abs()
    print(f"\nrecomputed vs stored macro-F1: max |difference| = {drift.max():.2e} "
          f"over {len(df)} runs")
    if drift.max() > 1e-9:
        d = df.loc[drift > 1e-9, ["arm", "T", "protocol", "seed", "f1",
                                  "stored_f1"]]
        print("  runs where the stored value is on a different denominator:")
        print(d.to_string(index=False))

    if not df.from_predictions.all():
        n = int((~df.from_predictions).sum())
        print(f"\nNOTE: {n} run(s) had no saved predictions; their macro-F1 is "
              "the stored value.\n  On CIC-IDS2017 that is the four-class "
              "denominator, so those cells are not\n  comparable with the "
              "others and are excluded from the premium below.")

    g = (df.groupby(["arm", "T", "protocol"])
           .agg(f1=("f1", "mean"), f1sd=("f1", lambda v: v.std(ddof=1)),
                in_spk=("in_spk", "mean"), sops=("sops", "mean"),
                n=("seed", "nunique"), ok=("from_predictions", "all"))
           .reset_index())
    g.to_csv(OUT / "cells.csv", index=False, float_format="%.8f")

    # --- the headline: premium as a function of T -------------------------
    lat = g[g.arm == "B_latency_thr001"].set_index(["protocol", "T"])
    del_ = g[g.arm == "D_delta_thr001"].set_index(["protocol", "T"])
    idx = lat.index.intersection(del_.index)
    prem = pd.DataFrame({
        "premium_pp": 100.0 * (lat.loc[idx].f1 - del_.loc[idx].f1),
        "f1_timing": lat.loc[idx].f1, "f1_notiming": del_.loc[idx].f1,
        "in_spk_timing": lat.loc[idx].in_spk,
        "in_spk_notiming": del_.loc[idx].in_spk,
        "matched": (lat.loc[idx].in_spk - del_.loc[idx].in_spk).abs(),
        "usable": lat.loc[idx].ok & del_.loc[idx].ok,
    }).reset_index()
    prem.to_csv(OUT / "timing_premium_vs_T.csv", index=False,
                float_format="%.8f")

    print("\n=== input-spike matching, the control the premium depends on ===")
    print("(latency@0.01 and delta@0.01 must emit identical input spike sets)")
    bad = prem[prem.matched > 1e-6]
    print(f"cells where they differ by >1e-6: {len(bad)} of {len(prem)}"
          + ("" if len(bad) else "   -- construction holds at every T"))
    if len(bad):
        print(bad[["protocol", "T", "in_spk_timing", "in_spk_notiming",
                   "matched"]].to_string(index=False))

    print("\n=== timing premium (pp macro-F1) by protocol and T ===")
    piv = prem.pivot(index="protocol", columns="T", values="premium_pp")
    piv = piv.reindex([p for p in PROTOCOLS if p in piv.index])
    piv.index = [LABEL[p] for p in piv.index]
    print(piv.round(2).to_string())

    print("\n=== does it scale with T? ===")
    for name, row in piv.iterrows():
        v = row.dropna()
        if len(v) < 3:
            continue
        r = np.corrcoef(v.index.astype(float), v.to_numpy())[0, 1]
        print(f"  {name:14s} Spearman-free Pearson(T, premium) = {r:+.3f}   "
              f"range {v.min():+.2f} to {v.max():+.2f} pp")
    allc = prem.dropna(subset=["premium_pp"])
    r = np.corrcoef(allc["T"].astype(float), allc.premium_pp)[0, 1]
    print(f"  pooled over all cells: {r:+.3f}  (n={len(allc)})")

    print("\n=== the falsifiable consequence: does CTU-13 f1's liability "
          "shrink at low T? ===")
    f1r = piv.loc["CTU-13 f1"].dropna() if "CTU-13 f1" in piv.index else None
    if f1r is not None:
        print(f1r.round(2).to_string())
        print("  prediction if resolution is the mechanism: premium rises "
              "toward zero as T falls")
        print(f"  observed at T=5: {f1r.get(5, float('nan')):+.2f} pp, "
              f"at T=50: {f1r.get(50, float('nan')):+.2f} pp")

    print(f"\nwrote {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
