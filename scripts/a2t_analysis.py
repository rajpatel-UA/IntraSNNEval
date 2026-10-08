#!/usr/bin/env python3
"""Is each COMPONENT of the timing premium flat in T, or only their sum?

POST-HOC. Extends PREREGISTRATION_ADDENDUM.md 8.

E3 established that Delta_time is flat in T. But Delta_time = Delta_map +
Delta_spread, and on the two KDD protocols those components are about +6 and
-6 pp, so a flat sum is equally consistent with two components that move
together and cancel. The abstract's "flat" claim rests on which reading is
right.

Arms, all at LeakyParallel, seeds 42-44:

    A1 latency@0.01   T in {5,10,50} from E3 arm B; T=25 from the sweep
    A2 permuted-time  T in {5,10,50} from runs_a2t; T=25 from runs_a2
    A3 delta@0.01     T in {5,10,50} from E3 arm D; T=25 from E1 arm C

Every cell is three seeds, so all comparisons are paired within (protocol, T).

    python scripts/a2t_analysis.py
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

OUT = ROOT / "results/analysis/a2t"
PROTOCOLS = ["nslkdd_v2", "kddcup99_v2", "cicids2017_v2",
             "ctu13_v2_f0", "ctu13_v2_f1"]
SEEDS = (42, 43, 44)
TS = [5, 10, 25, 50]
LABEL = {"nslkdd_v2": "NSL-KDD", "kddcup99_v2": "KDDCup99",
         "cicids2017_v2": "CIC-IDS2017", "ctu13_v2_f0": "CTU-13 f0",
         "ctu13_v2_f1": "CTU-13 f1"}

#: arm -> list of (path template, T values it supplies)
SOURCES = {
    "A1": [("results/runs_e3/B_latency_thr001/T{t}/{p}/seed_{s}", (5, 10, 50))],
    "A2": [("results/runs_a2t/A2_permuted_thr001/T{t}/{p}/seed_{s}", (5, 10, 50)),
           ("results/runs_a2/A2_permuted_thr001/{p}/seed_{s}", (25,))],
    "A3": [("results/runs_e3/D_delta_thr001/T{t}/{p}/seed_{s}", (5, 10, 50)),
           ("results/runs_e1/C_delta_thr001/{p}/seed_{s}", (25,))],
}


def read(path: Path):
    f = path / "results.json"
    if not f.exists():
        return None
    rec = json.loads(f.read_text())
    z = np.load(path / "artifacts" / "test_prefix.npz")
    m = macro_f1_pair(z["y_true"].astype(np.int64), z["y_pred"].astype(np.int64),
                      rec["data"]["num_classes"])
    return (m["macro_f1_fixed_universe"],
            rec["test_plus"]["spike_accounting"]["input_spikes_per_sample"])


def collect() -> pd.DataFrame:
    rows = []
    for arm, srcs in SOURCES.items():
        for tmpl, ts in srcs:
            for t in ts:
                for p in PROTOCOLS:
                    for s in SEEDS:
                        r = read(ROOT / tmpl.format(t=t, p=p, s=s))
                        if r:
                            rows.append({"arm": arm, "T": t, "protocol": p,
                                         "seed": s, "f1": r[0], "in_spk": r[1]})
    # A1 at T=25 comes from the confirmation sweep.
    for run in discover():
        if (run.variant == "LeakyParallel/latency" and run.seed in SEEDS
                and run.protocol in PROTOCOLS):
            rec, pre = run.record(), run.prefix()
            m = macro_f1_pair(pre["y_true"].astype(np.int64),
                              pre["y_pred"].astype(np.int64),
                              rec["data"]["num_classes"])
            rows.append({"arm": "A1", "T": 25, "protocol": run.protocol,
                         "seed": run.seed, "f1": m["macro_f1_fixed_universe"],
                         "in_spk": rec["test_plus"]["spike_accounting"]
                                      ["input_spikes_per_sample"]})
    return pd.DataFrame(rows)


def tie_diagnostics() -> pd.DataFrame:
    """How much room the permutation actually has, per (protocol, T).

    Two measured quantities in place of a judgement call:

      feat_per_bin    active features divided by T. When several features share
                      one distinguishable spike time, permuting among them
                      cannot change the assignment.
      identical_frac  the realised version: the share of active features whose
                      permuted time equals their original time. As it
                      approaches 1 the manipulation approaches a null operation
                      on the assignment while still destroying the code, which
                      is the regime where the permuted arm collapses rather
                      than degrades.
    """
    import torch
    from scipy.stats import spearmanr

    from src.data import get_dataset
    from src.encoding import get_encoder

    lat, per = get_encoder("latency", 0.01), get_encoder("permuted_latency", 0.01)
    out = []
    for p_ in PROTOCOLS:
        b = get_dataset(p_, batch_size=4096, seed=42)
        x, _ = next(iter(b.test_loader))
        for t in TS:
            A1, A2 = lat(x, t), per(x, t)
            act = A1.sum(0) > 0
            t1, t2 = A1.argmax(0), A2.argmax(0)
            n_act = act.sum(1).float().mean().item()
            same = ((t1 == t2) & act).sum().item() / max(act.sum().item(), 1)
            out.append({"protocol": p_, "T": t,
                        "active_per_sample": n_act,
                        "feat_per_bin": n_act / t,
                        "identical_frac": same,
                        "spearman_A1": float(spearmanr(
                            x[act].numpy(), t1[act].numpy()).statistic),
                        "spearman_A2": float(spearmanr(
                            x[act].numpy(), t2[act].numpy()).statistic)})
    return pd.DataFrame(out)


def main() -> int:
    df = collect()
    if df.empty:
        print("no runs found.")
        return 0
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT / "runs.csv", index=False, float_format="%.8f")

    print("=== coverage (seeds per arm x T; 15 = 5 protocols x 3 seeds) ===")
    print(df.groupby(["arm", "T"]).size().unstack(fill_value=0).to_string())

    g = (df.groupby(["arm", "T", "protocol"])
           .agg(f1=("f1", "mean"), sd=("f1", lambda v: v.std(ddof=1)),
                in_spk=("in_spk", "mean")).reset_index())
    g.to_csv(OUT / "cells.csv", index=False, float_format="%.8f")

    w = g.pivot_table(index=["protocol", "T"], columns="arm",
                      values=["f1", "sd", "in_spk"])
    rows = []
    for (p, t) in w.index:
        try:
            a1, a2, a3 = (w.loc[(p, t), ("f1", a)] for a in ("A1", "A2", "A3"))
        except KeyError:
            continue
        rows.append({"protocol": p, "T": t,
                     "d_map_pp": 100 * (a1 - a2),
                     "d_spread_pp": 100 * (a2 - a3),
                     "d_time_pp": 100 * (a1 - a3),
                     "in_spk_spread": float(np.ptp(
                         [w.loc[(p, t), ("in_spk", a)] for a in
                          ("A1", "A2", "A3")])),
                     "sd_A1": w.loc[(p, t), ("sd", "A1")],
                     "sd_A2": w.loc[(p, t), ("sd", "A2")],
                     "sd_A3": w.loc[(p, t), ("sd", "A3")]})
    d = pd.DataFrame(rows)
    d["se_map_pp"] = np.sqrt(d.sd_A1**2 + d.sd_A2**2) * 100 / np.sqrt(3)
    d["se_spread_pp"] = np.sqrt(d.sd_A2**2 + d.sd_A3**2) * 100 / np.sqrt(3)
    d.to_csv(OUT / "components_vs_T.csv", index=False, float_format="%.8f")

    print("\n=== input-spike matching across the three arms, every cell ===")
    print(f"max spread within a (protocol, T) cell = {d.in_spk_spread.max():.3e}"
          f" spikes per sample, over {len(d)} cells")

    for name, col in (("Delta_map", "d_map_pp"),
                      ("Delta_spread", "d_spread_pp"),
                      ("Delta_time", "d_time_pp")):
        piv = d.pivot(index="protocol", columns="T", values=col)
        piv = piv.reindex([p for p in PROTOCOLS if p in piv.index])
        piv.index = [LABEL[p] for p in piv.index]
        print(f"\n=== {name} (pp macro-F1) by protocol and T ===")
        print(piv.round(2).to_string())

    print("\n=== is each component flat in T? ===")
    print("range across T against the largest standard error in that row")
    for comp, col, se in (("map", "d_map_pp", "se_map_pp"),
                          ("spread", "d_spread_pp", "se_spread_pp")):
        print(f"\n  {comp}:")
        for p in PROTOCOLS:
            q = d[d.protocol == p]
            if len(q) < 3:
                continue
            rng = float(np.ptp(q[col]))
            mx = q[se].max()
            print(f"    {LABEL[p]:14s} range {rng:6.2f} pp   max SE {mx:5.2f}"
                  f"   ratio {rng / mx:5.2f}"
                  f"   {'flat within noise' if rng / mx < 2 else 'MOVES'}")
        r = np.corrcoef(d["T"].astype(float), d[col])[0, 1]
        print(f"    pooled Pearson(T, {comp}) = {r:+.3f}  (n={len(d)})")

    # --- component coupling across T --------------------------------------
    print("\n=== coupling: rho(Delta_map, Delta_spread) across T ===")
    print("a negative value means the two terms move against each other, so a")
    print("flat sum is a moving cancellation rather than two static terms")
    rho = {}
    for p_ in PROTOCOLS:
        q = d[d.protocol == p_].sort_values("T")
        if len(q) < 3:
            continue
        rho[p_] = float(np.corrcoef(q.d_map_pp, q.d_spread_pp)[0, 1])
        print(f"  {LABEL[p_]:14s} rho = {rho[p_]:+.2f}")
    print("  CTU-13 f1's value is driven by the degenerate low-T cells and is "
          "not interpreted")

    # --- cross-file consistency: A2T must reproduce E3's Delta_time ---------
    e3 = ROOT / "results/analysis/e3/timing_premium_vs_T.csv"
    if e3.exists():
        t3 = pd.read_csv(e3).set_index(["protocol", "T"]).premium_pp
        mine = d.set_index(["protocol", "T"]).d_time_pp
        common = t3.index.intersection(mine.index)
        err = (t3.loc[common] - mine.loc[common]).abs().max()
        print(f"\n=== cross-check against E3 (Table XVII) ===")
        print(f"max |A2T d_time - E3 premium| = {err:.3e} pp over "
              f"{len(common)} shared cells")
        # Tolerance is 1e-6, not 1e-9: both sides round-trip through CSVs
        # written at eight decimals, so values of order 10 carry about 1e-8 of
        # representation error. A tighter bound tests the file format rather
        # than the pipeline.
        if err > 1e-6:
            raise SystemExit(
                "A2T and E3 disagree on the timing premium. They read the same "
                "A1 and A3 runs, so any difference is a pipeline fault.")

    # --- degeneracy, measured rather than judged ---------------------------
    print("\n=== degeneracy diagnostics per (protocol, T) ===")
    deg = tie_diagnostics()
    deg.to_csv(OUT / "degeneracy.csv", index=False, float_format="%.8f")
    print(deg.to_string(index=False))
    print("\nfeat_per_bin = active features / T. identical_frac = share of "
          "active features\nwhose permuted time equals their original time; as "
          "it approaches 1 the\nmanipulation approaches a null operation on the "
          "assignment.")

    (OUT / "summary.json").write_text(json.dumps({
        "n_cells": int(len(d)), "n_runs_new": 45,
        "rho": rho,
        "map_moves": int(sum(
            float(np.ptp(d[d.protocol == q].d_map_pp))
            / d[d.protocol == q].se_map_pp.max() >= 2 for q in PROTOCOLS)),
        "spread_moves": int(sum(
            float(np.ptp(d[d.protocol == q].d_spread_pp))
            / d[d.protocol == q].se_spread_pp.max() >= 2 for q in PROTOCOLS)),
    }, indent=2))

    print(f"\nwrote {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
