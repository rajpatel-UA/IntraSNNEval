#!/usr/bin/env python3
"""Two-axis confirmation analysis under v2. **Not** a v2 design-space ranking.

What this answers, and nothing more:

  ``--axis neuron``   With the encoding held at latency, where does
                      LeakyParallel rank among the nine neuron families?
  ``--axis encoding`` With the neuron held at LeakyParallel, which encoding is
                      strongest?

What it does not answer: a v2 ranking of the 27 configurations. That sweep was
not run — it costs ~198 GPU-hours — and no combination of these two axes
reconstructs it, because each holds the other factor fixed. Every output here is
labelled with its axis and its held-fixed factor so a table cannot be lifted
into the manuscript as an overall v2 rank. The complete 27-configuration
ranking is the v1 screening result and lives in `results/analysis/v1/`.

CTU-13 needs one further distinction that the caption must carry. The strict
CTU-13 *performance estimate* is the four-fold LeakyParallel/latency figure
(92.01 +/- 9.72, folds 77.0-99.9) in `results/v1_submitted/ctu13_strict_vanilla.csv`.
The CTU-13 rows produced here come from two preregistered folds and are a
**neuron-axis comparative confirmation**, not a new overall CTU-13 performance
number. They are for ranking neurons against each other under a strict
protocol; they are not to be quoted as CTU-13 performance.

    python scripts/axis_analysis.py --axis neuron
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare, rankdata

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.populations import macro_f1_pair, require_population  # noqa: E402
from src.readout import discover  # noqa: E402

OUT = ROOT / "results/analysis"
PRIMARY = "f1_macro"
CHAMPION_NEURON = "LeakyParallel"
CHAMPION_ENCODING = "latency"

#: Protocols each axis is expected to cover once the run program completes.
#: A result missing any of these is INTERIM and must not be quoted as final --
#: CIC-IDS2017 adds five paired blocks to the encoding axis, and the CTU-13
#: folds are exactly where B1 already shows the simple narrative is most likely
#: to be challenged.
EXPECTED = {
    "encoding": ["nslkdd_v2", "kddcup99_v2", "cicids2017_v2",
                 "ctu13_v2_f0", "ctu13_v2_f1"],
    "neuron": ["nslkdd_v2", "kddcup99_v2", "ctu13_v2_f0", "ctu13_v2_f1"],
}

CTU_CAVEAT = (
    "CTU-13 rows are a neuron-axis comparative confirmation on two "
    "preregistered scenario-disjoint folds. They are NOT the strict CTU-13 "
    "performance estimate; that is the four-fold LeakyParallel/latency figure "
    "92.01 +/- 9.72 (folds 77.0/91.2/99.9/99.9)."
)


def collect(axis: str) -> pd.DataFrame:
    if axis == "neuron":
        runs = discover(encodings=[CHAMPION_ENCODING])
        level = "neuron"
    else:
        runs = discover(neurons=[CHAMPION_NEURON])
        level = "encoding"
    rows = []
    for r in runs:
        if not r.protocol.endswith(("_v2", "_f0", "_f1", "_f2", "_f3",
                                    "daydisjoint")):
            continue          # v1 protocols never enter a v2 confirmation table
        rec = r.record()
        m = rec["test_plus"]["metrics"]
        # Recomputed over the fixed label universe. The stored f1_macro was
        # written before that bug was found and uses a model-dependent
        # denominator on any protocol with a zero-support class.
        pre = r.prefix()
        dual = macro_f1_pair(pre["y_true"].astype(np.int64),
                             pre["y_pred"].astype(np.int64),
                             rec["data"]["num_classes"])
        rows.append({"protocol": r.protocol, "level": getattr(r, level),
                     "neuron": r.neuron, "encoding": r.encoding,
                     "seed": r.seed,
                     PRIMARY: dual["macro_f1_fixed_universe"],
                     "macro_f1_test_present": dual["macro_f1_test_present"],
                     "macro_f1_ceiling": dual["macro_f1_ceiling"],
                     "mcc": m["matthews_corrcoef"]})
    return pd.DataFrame(rows)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--axis", choices=["neuron", "encoding"], default="neuron")
    args = ap.parse_args()

    df = collect(args.axis)
    if df.empty:
        print(f"no v2 runs on the {args.axis} axis yet.")
        return 0

    held = (f"encoding fixed at {CHAMPION_ENCODING}" if args.axis == "neuron"
            else f"neuron fixed at {CHAMPION_NEURON}")
    tag = f"v2_{args.axis}_axis"
    out = OUT / tag
    out.mkdir(parents=True, exist_ok=True)

    # Per-dataset mean +/- across-seed SD. Never pooled across datasets.
    per_ds = (df.groupby(["protocol", "level"])
                .agg(n_seeds=("seed", "nunique"),
                     f1_mean=(PRIMARY, "mean"), f1_sd=(PRIMARY, "std"),
                     f1_present_mean=("macro_f1_test_present", "mean"),
                     ceiling=("macro_f1_ceiling", "first"),
                     mcc_mean=("mcc", "mean"))
                .reset_index())
    per_ds.to_csv(out / "per_dataset.csv", index=False)

    # Mean rank *within this axis only*.
    wide = df.pivot_table(index=["protocol", "seed"], columns="level",
                          values=PRIMARY, aggfunc="mean")
    complete = wide.dropna()
    ranks = pd.DataFrame(
        np.apply_along_axis(lambda r: rankdata(-r, method="average"), 1,
                            complete.to_numpy()),
        index=complete.index, columns=complete.columns)
    mr = pd.DataFrame({"mean_rank": ranks.mean(),
                       "median_rank": ranks.median(),
                       "best_block_count": (ranks == 1).sum()}
                      ).sort_values("mean_rank")
    mr.to_csv(out / "mean_rank.csv")

    champion = CHAMPION_NEURON if args.axis == "neuron" else CHAMPION_ENCODING
    present = set(df.protocol.unique())
    missing = [p for p in EXPECTED[args.axis] if p not in present]
    status = "INTERIM" if missing else "COMPLETE"

    population_id = ("neuron_axis_latency" if args.axis == "neuron"
                     else "encoding_axis_lp")
    require_population(
        {f"{r.neuron}/{r.encoding}" for r in
         df[["neuron", "encoding"]].itertuples()}, population_id)

    summary = {
        "axis": args.axis,
        "population_id": population_id,
        "status": status,
        "missing_protocols": missing,
        "held_fixed": held,
        "scope_note": (f"Ranking is among {args.axis}s with {held}. This is a "
                       "targeted confirmation, NOT a v2 ranking of the 27 "
                       "configurations; that sweep was not run. The complete "
                       "27-configuration ranking is the v1 screening result."),
        "protocols": sorted(df.protocol.unique()),
        "n_blocks": int(len(complete)),
        "n_levels": int(complete.shape[1]),
        "champion": champion,
        "rank_method": "average",
    }
    if len(complete) and complete.shape[1] > 2:
        stat, p = friedmanchisquare(*[complete[c].to_numpy()
                                      for c in complete.columns])
        summary["friedman_chi2"] = float(stat)
        summary["friedman_p"] = float(p)
    if champion in mr.index:
        summary["champion_mean_rank"] = float(mr.loc[champion, "mean_rank"])
        summary["champion_rank_position"] = int(mr.index.get_loc(champion) + 1)
    if any(p.startswith("ctu13") for p in df.protocol.unique()):
        summary["ctu13_caveat"] = CTU_CAVEAT
    (out / "summary.json").write_text(json.dumps(summary, indent=2))

    print(f"=== v2 {args.axis}-axis confirmation [{status}] ({held}) ===")
    if missing:
        print(f"!! INTERIM — missing {', '.join(missing)}. These numbers will "
              f"change when those protocols land; do not quote them as final.")
    print(summary["scope_note"])
    print(f"\nprotocols: {', '.join(summary['protocols'])}")
    print(f"blocks: {summary['n_blocks']}  levels: {summary['n_levels']}\n")
    print(mr.round(3).to_string())
    if "champion_rank_position" in summary:
        print(f"\n{champion}: rank {summary['champion_rank_position']} of "
              f"{summary['n_levels']} on this axis "
              f"(mean rank {summary['champion_mean_rank']:.2f})")
    print("\n--- per dataset, mean +/- across-seed SD ---")
    show = per_ds.copy()
    show["macro-F1"] = (show.f1_mean.round(4).astype(str) + " ± "
                        + show.f1_sd.fillna(0).round(4).astype(str))
    print(show.pivot(index="level", columns="protocol",
                     values="macro-F1").to_string())
    if "ctu13_caveat" in summary:
        print(f"\nNOTE: {CTU_CAVEAT}")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
