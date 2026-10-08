#!/usr/bin/env python3
"""Protocol-level sensitivity: what survives if the protocol, not the seed, is
the unit of analysis.

The pre-registered analysis treats each (protocol x seed) pair as a block, which
gives more blocks but treats five seeds on one protocol as five independent
observations. They are not independent in the sense that matters for
generalisation: they share a dataset, a partition and a preprocessing fit. That
is the pseudo-replication concern, and this answers it directly by collapsing
each protocol to a single mean before ranking.

This is a **sensitivity analysis and does not replace anything.** The
pre-registered seed-block statistics remain primary; these numbers show what is
left when the higher-level unit is used instead. Expect wider uncertainty and
fewer blocks --- five protocols instead of twenty-five --- which is the honest
cost of the stricter assumption.

No retraining: it re-aggregates results already on disk.

    python scripts/protocol_level.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.populations import macro_f1_pair, members  # noqa: E402
from src.readout import discover  # noqa: E402

OUT = ROOT / "results/analysis/protocol_level"
PROTOCOLS = ["nslkdd_v2", "kddcup99_v2", "cicids2017_v2", "ctu13_v2_f0",
             "ctu13_v2_f1"]
#: Each axis's registered protocols. The neuron axis excludes CIC-IDS2017 (no
#: U2R test support), so it must not pick it up just because runs exist there.
AXIS_PROTOCOLS = {"encoding": PROTOCOLS,
                  "neuron": [p for p in PROTOCOLS if p != "cicids2017_v2"]}


def collect() -> pd.DataFrame:
    enc, neu = members("encoding_axis_lp"), members("neuron_axis_latency")
    rows = []
    for r in discover():
        if r.protocol not in PROTOCOLS:
            continue
        # LeakyParallel/latency is the intersection of the two axes and belongs
        # to BOTH. An exclusive if/elif silently dropped it from the neuron
        # axis, leaving an eight-neuron ranking that quietly excluded the very
        # configuration the axis exists to test.
        axes = [a for a, pop in (("encoding", enc), ("neuron", neu))
                if r.variant in pop]
        if not axes:
            continue
        rec = r.record()
        pre = r.prefix()
        m = macro_f1_pair(pre["y_true"].astype(np.int64),
                          pre["y_pred"].astype(np.int64),
                          rec["data"]["num_classes"])
        for axis in axes:
            rows.append({"axis": axis, "protocol": r.protocol,
                         "level": r.encoding if axis == "encoding" else r.neuron,
                         "seed": r.seed,
                         "macro_f1": m["macro_f1_fixed_universe"]})
    return pd.DataFrame(rows)


def analyse(df: pd.DataFrame, axis: str):
    sub = df[(df.axis == axis) & df.protocol.isin(AXIS_PROTOCOLS[axis])]
    # One mean per (protocol, level): the protocol becomes the unit.
    per_proto = (sub.groupby(["protocol", "level"]).macro_f1.mean()
                    .unstack("level").dropna())
    ranks = pd.DataFrame(
        np.apply_along_axis(lambda r: rankdata(-r, method="average"), 1,
                            per_proto.to_numpy()),
        index=per_proto.index, columns=per_proto.columns)
    summary = pd.DataFrame({
        "mean_rank_protocol_level": ranks.mean(),
        "median_rank_protocol_level": ranks.median(),
        "protocols_won": (ranks == 1).sum(),
        "n_protocols": len(per_proto),
    }).sort_values("mean_rank_protocol_level")
    wins = per_proto.idxmax(axis=1).rename("winner").to_frame()
    wins["margin_pp"] = [
        100 * (per_proto.loc[p].max() - per_proto.loc[p].drop(
            per_proto.loc[p].idxmax()).max()) for p in per_proto.index]
    return per_proto, summary, wins


def main() -> int:
    df = collect()
    if df.empty:
        print("no strict runs found.")
        return 1
    OUT.mkdir(parents=True, exist_ok=True)
    all_wins = []
    for axis, fname in (("encoding", "encoding_mean_ranks.csv"),
                        ("neuron", "neuron_mean_ranks.csv")):
        per_proto, summary, wins = analyse(df, axis)
        summary.to_csv(OUT / fname)
        per_proto.round(4).to_csv(OUT / f"{axis}_protocol_means.csv")
        wins = wins.reset_index()
        wins.insert(0, "axis", axis)
        all_wins.append(wins)

        seed_level = ROOT / f"results/analysis/v2_{axis}_axis/mean_rank.csv"
        cmp = ""
        if seed_level.exists():
            sl = pd.read_csv(seed_level, index_col=0)
            same = list(sl.index) == list(summary.index)
            cmp = ("  ordering vs pre-registered seed-block analysis: "
                   + ("UNCHANGED" if same else "CHANGED"))
        print(f"\n=== {axis} axis, protocol as unit "
              f"({summary.n_protocols.iloc[0]} protocols) ===")
        print(summary.round(3).to_string())
        if cmp:
            print(cmp)
    pd.concat(all_wins).to_csv(OUT / "protocol_wins.csv", index=False)
    print(f"\nwrote {OUT}")
    print("Sensitivity analysis only — the pre-registered seed-block statistics "
          "remain primary.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
