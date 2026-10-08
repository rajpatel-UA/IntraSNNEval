#!/usr/bin/env python3
"""Decompose the timing premium into mapping and spreading (arm A2).

POST-HOC. Prediction, outcome map and stopping rule frozen in
PREREGISTRATION_ADDENDUM.md 8 before any A2 run existed.

    Delta_map    = A1 - A2   value of the amplitude-to-time mapping
    Delta_spread = A2 - A3   value of temporal distribution as such
    Delta_time   = A1 - A3 = Delta_map + Delta_spread

A1 is latency@0.01 from the confirmation sweep, A2 is the permuted-time control,
A3 is E1 arm C (delta@0.01). All three at T=25, LeakyParallel, seeds 42-44, so
the comparison is paired.

The closure identity is a harness check, not a finding: it holds by
construction, and failure means something is wrong in the pipeline rather than
in the science.

Four manipulation checks, computed on the encoders directly rather than
inferred from the runs, because they are what turn the control from "we did
something random" into an instrument.

    python scripts/a2_analysis.py
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

A2DIR = ROOT / "results/runs_a2/A2_permuted_thr001"
E1DIR = ROOT / "results/runs_e1/C_delta_thr001"
OUT = ROOT / "results/analysis/a2"
PROTOCOLS = ["nslkdd_v2", "kddcup99_v2", "cicids2017_v2",
             "ctu13_v2_f0", "ctu13_v2_f1"]
SEEDS = (42, 43, 44)
LABEL = {"nslkdd_v2": "NSL-KDD", "kddcup99_v2": "KDDCup99",
         "cicids2017_v2": "CIC-IDS2017", "ctu13_v2_f0": "CTU-13 f0",
         "ctu13_v2_f1": "CTU-13 f1"}


def read_dir(base: Path, arm: str) -> list[dict]:
    rows = []
    for proto in PROTOCOLS:
        for s in SEEDS:
            p = base / proto / f"seed_{s}"
            f = p / "results.json"
            if not f.exists():
                continue
            rec = json.loads(f.read_text())
            z = np.load(p / "artifacts" / "test_prefix.npz")
            m = macro_f1_pair(z["y_true"].astype(np.int64),
                              z["y_pred"].astype(np.int64),
                              rec["data"]["num_classes"])
            sa = rec["test_plus"]["spike_accounting"]
            rows.append({"arm": arm, "protocol": proto, "seed": s,
                         "f1": m["macro_f1_fixed_universe"],
                         "in_spk": sa["input_spikes_per_sample"],
                         "sops": sa["sops_per_sample"]})
    return rows


def collect() -> pd.DataFrame:
    rows = read_dir(A2DIR, "A2_permuted") + read_dir(E1DIR, "A3_delta")
    for r in discover():
        if (r.variant == "LeakyParallel/latency" and r.seed in SEEDS
                and r.protocol in PROTOCOLS):
            rec, pre = r.record(), r.prefix()
            m = macro_f1_pair(pre["y_true"].astype(np.int64),
                              pre["y_pred"].astype(np.int64),
                              rec["data"]["num_classes"])
            sa = rec["test_plus"]["spike_accounting"]
            rows.append({"arm": "A1_latency", "protocol": r.protocol,
                         "seed": r.seed,
                         "f1": m["macro_f1_fixed_universe"],
                         "in_spk": sa["input_spikes_per_sample"],
                         "sops": sa["sops_per_sample"]})
    return pd.DataFrame(rows)


def manipulation_checks() -> pd.DataFrame:
    """What the permutation did and did not change, measured on the encoders."""
    import torch
    from scipy.stats import spearmanr

    from src.data import get_dataset
    from src.encoding import get_encoder

    lat = get_encoder("latency", 0.01)
    per = get_encoder("permuted_latency", 0.01)
    T = 25
    out = []
    for proto in PROTOCOLS:
        b = get_dataset(proto, batch_size=4096, seed=42)
        x, _ = next(iter(b.test_loader))
        A1, A2 = lat(x, T), per(x, T)
        c1 = A1.sum(0)
        c2 = A2.sum(0)
        act = c1 > 0
        # Spike time per (row, feature); only meaningful where the feature
        # spikes at all, so every comparison below is masked by `act`.
        t1, t2 = A1.argmax(0), A2.argmax(0)
        ms_ok = all(torch.equal(torch.sort(t1[i][act[i]])[0],
                                torch.sort(t2[i][act[i]])[0])
                    for i in range(x.shape[0]))
        # rank correlation between amplitude and spike time, over active entries
        xs = x[act].numpy()
        r1 = spearmanr(xs, t1[act].numpy()).statistic
        r2 = spearmanr(xs, t2[act].numpy()).statistic
        out.append({
            "protocol": proto, "n_rows": int(x.shape[0]),
            "spike_count_cells_differing": int((c1 != c2).sum()),
            "time_multiset_identical": bool(ms_ok),
            "profile_max_abs_diff": float((A1.sum((1, 2)) - A2.sum((1, 2)))
                                          .abs().max()),
            "spearman_amp_time_A1": float(r1),
            "spearman_amp_time_A2": float(r2),
        })
    return pd.DataFrame(out)


def main() -> int:
    df = collect()
    if df.empty:
        print("no A2 runs found.")
        return 0
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT / "runs.csv", index=False, float_format="%.8f")

    cov = df.groupby(["arm", "protocol"]).seed.nunique().unstack(fill_value=0)
    print("=== coverage (seeds per arm x protocol; 3 expected) ===")
    print(cov.to_string())

    g = (df.groupby(["arm", "protocol"])
           .agg(f1=("f1", "mean"), sd=("f1", lambda v: v.std(ddof=1)),
                in_spk=("in_spk", "mean"))
           .reset_index().set_index(["protocol", "arm"]))

    rows = []
    for p in PROTOCOLS:
        if not {("A1_latency"), ("A2_permuted"), ("A3_delta")} <= set(
                g.loc[p].index if p in g.index.get_level_values(0) else []):
            continue
        a1, a2, a3 = (g.loc[(p, a), "f1"] for a in
                      ("A1_latency", "A2_permuted", "A3_delta"))
        rows.append({
            "protocol": p,
            "A1_latency": a1, "A2_permuted": a2, "A3_delta": a3,
            "d_map_pp": 100 * (a1 - a2),
            "d_spread_pp": 100 * (a2 - a3),
            "d_time_pp": 100 * (a1 - a3),
            "in_spk_A1": g.loc[(p, "A1_latency"), "in_spk"],
            "in_spk_A2": g.loc[(p, "A2_permuted"), "in_spk"],
            "in_spk_A3": g.loc[(p, "A3_delta"), "in_spk"],
            "sd_A1": g.loc[(p, "A1_latency"), "sd"],
            "sd_A2": g.loc[(p, "A2_permuted"), "sd"],
            "sd_A3": g.loc[(p, "A3_delta"), "sd"],
        })
    d = pd.DataFrame(rows)
    d["closure_err_pp"] = (d.d_map_pp + d.d_spread_pp - d.d_time_pp).abs()
    d.to_csv(OUT / "decomposition.csv", index=False, float_format="%.8f")
    # The residual is ~1e-15 and would round to zero in an 8-decimal CSV, so it
    # is persisted separately at full precision. A macro built from the CSV
    # column would report a precision the artifact no longer carries.
    (OUT / "closure.json").write_text(json.dumps({
        "max_abs_closure_error_pp": float(d.closure_err_pp.max()),
        "n_protocols": int(len(d)),
        "note": "d_time - (d_map + d_spread), exact by construction",
    }, indent=2))

    print("\n=== closure check: d_time == d_map + d_spread ===")
    print(f"max |error| = {d.closure_err_pp.max():.3e} pp over {len(d)} protocols")
    if d.closure_err_pp.max() > 1e-9:
        print("STOP: the decomposition does not close. Fix the harness before "
              "reading anything below.")
        return 1

    print("\n=== input-spike matching across all three arms ===")
    mm = (d[["in_spk_A1", "in_spk_A2", "in_spk_A3"]].max(axis=1)
          - d[["in_spk_A1", "in_spk_A2", "in_spk_A3"]].min(axis=1))
    print(f"max spread within a protocol = {mm.max():.3e} spikes per sample")

    print("\n=== decomposition (pp macro-F1) ===")
    show = d.copy()
    show["protocol"] = show.protocol.map(LABEL)
    print(show[["protocol", "A1_latency", "A2_permuted", "A3_delta",
                "d_map_pp", "d_spread_pp", "d_time_pp"]]
          .round(4).to_string(index=False))

    print("\n=== components against the published premium ===")
    # A share of d_time is not reported where |d_time| is small: the ratio
    # explodes without meaning when the two components nearly cancel, which is
    # exactly what happens on the two KDD protocols.
    for _, r in d.iterrows():
        share = (f"{100 * r.d_map_pp / r.d_time_pp:6.1f}%"
                 if abs(r.d_time_pp) >= 5.0 else "     --")
        print(f"  {LABEL[r.protocol]:14s} d_time {r.d_time_pp:+7.2f}  "
              f"d_map {r.d_map_pp:+7.2f}  d_spread {r.d_spread_pp:+7.2f}  "
              f"map share {share}")
    print("  (share withheld where |d_time| < 5 pp: the components nearly "
          "cancel there\n   and the ratio is not interpretable)")

    print("\n=== manipulation checks ===")
    mc = manipulation_checks()
    mc.to_csv(OUT / "manipulation_checks.csv", index=False,
              float_format="%.8f")
    print(mc.to_string(index=False))
    print(f"\nwrote {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
