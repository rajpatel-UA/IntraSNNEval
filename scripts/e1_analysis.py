#!/usr/bin/env python3
"""E1 tables: the threshold ablation and the timing isolation.

POST-HOC AND EXPLORATORY. The design, arms and seeds are in `run_e1.py`.

Two outputs under results/analysis/:

    e1_threshold_ablation.csv   one row per (arm, protocol) for the three E1
                                arms: seed means of macro-F1 (with its
                                across-seed SD, ddof=1), MCC, DR, FAR, input
                                spikes and SOPs per sample
    e1_timing_isolation.csv     latency@0.01 against delta@0.01 (arm C). At a
                                matched gate the two emit identical input spike
                                sets, so the macro-F1 difference is the value of
                                spike timing alone (Table tab:timing)

latency@0.01 is not an E1 arm. It is LeakyParallel/latency from the
confirmation sweep, restricted to the E1 seeds 42-44 so the comparison is
paired; the five-seed mean would move the KDDCup99 premium by 0.85 pp.

Macro-F1 is the pre-registered fixed-universe metric, recomputed from each run's
saved per-sample predictions (results/analysis/AMENDMENT_cic_u2r_support.md).

Both files were first committed on 2026-08-14 from an analysis run inline, with
no script in the tree. This script was written afterwards to make them
reproducible, and regenerates both byte for byte.

    python scripts/e1_analysis.py
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

E1DIR = ROOT / "results/runs_e1"
OUT = ROOT / "results/analysis"
PROTOCOLS = ["nslkdd_v2", "kddcup99_v2", "cicids2017_v2",
             "ctu13_v2_f0", "ctu13_v2_f1"]
SEEDS = (42, 43, 44)
ARMS = ("A_latency_thr000", "B_latency_thr005", "C_delta_thr001")


def measures(rec: dict, pre: dict) -> dict:
    m = macro_f1_pair(pre["y_true"].astype(np.int64),
                      pre["y_pred"].astype(np.int64),
                      rec["data"]["num_classes"])
    tp = rec["test_plus"]
    return {"f1": m["macro_f1_fixed_universe"],
            "mcc": tp["metrics"]["matthews_corrcoef"],
            "dr": tp["binary_ids_metrics"]["detection_rate"],
            "far": tp["binary_ids_metrics"]["false_alarm_rate"],
            "in_spk": tp["spike_accounting"]["input_spikes_per_sample"],
            "sops": tp["spike_accounting"]["sops_per_sample"]}


def e1_runs() -> pd.DataFrame:
    rows = []
    for arm in ARMS:
        for proto in PROTOCOLS:
            for s in SEEDS:
                p = E1DIR / arm / proto / f"seed_{s}"
                if not (p / "results.json").exists():
                    continue
                rec = json.loads((p / "results.json").read_text())
                z = np.load(p / "artifacts" / "test_prefix.npz")
                rows.append({"arm": arm, "protocol": proto, "seed": s,
                             **measures(rec, {k: z[k] for k in z.files})})
    return pd.DataFrame(rows)


def sweep_latency() -> pd.DataFrame:
    """LeakyParallel/latency@0.01 from the confirmation sweep, E1 seeds only."""
    return pd.DataFrame([
        {"protocol": r.protocol, "seed": r.seed, **measures(r.record(), r.prefix())}
        for r in discover(protocols=PROTOCOLS, neurons=["LeakyParallel"],
                          encodings=["latency"])
        if r.seed in SEEDS])


def main() -> int:
    e1 = e1_runs()
    if e1.empty:
        print(f"no runs under {E1DIR}; nothing to analyse")
        return 0

    abl = (e1.groupby(["arm", "protocol"])
             .agg(n=("seed", "nunique"), f1=("f1", "mean"),
                  f1sd=("f1", lambda v: v.std(ddof=1)), mcc=("mcc", "mean"),
                  dr=("dr", "mean"), far=("far", "mean"),
                  in_spk=("in_spk", "mean"), sops=("sops", "mean"))
             .reset_index())
    abl.to_csv(OUT / "e1_threshold_ablation.csv", index=False,
               float_format="%.6f")

    lat = sweep_latency().groupby("protocol")[["f1", "in_spk"]].mean()
    dlt = (e1[e1.arm == "C_delta_thr001"]
           .groupby("protocol")[["f1", "in_spk"]].mean())
    ti = pd.DataFrame({"in_spk_timing": lat.in_spk, "f1_timing": lat.f1,
                       "in_spk_notiming": dlt.in_spk, "f1_notiming": dlt.f1}
                      ).reindex(PROTOCOLS)
    ti.index.name = "protocol"
    ti["in_spk_abs_diff"] = (ti.in_spk_timing - ti.in_spk_notiming).abs()
    ti["timing_premium_pp"] = 100.0 * (ti.f1_timing - ti.f1_notiming)
    ti.to_csv(OUT / "e1_timing_isolation.csv", float_format="%.10f")

    print(f"{len(e1)} E1 runs -> e1_threshold_ablation.csv ({len(abl)} rows)")
    print(ti[["f1_timing", "f1_notiming", "timing_premium_pp"]]
          .round(4).to_string())
    print(f"max input-spike difference between the matched arms: "
          f"{ti.in_spk_abs_diff.max():.1e}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
