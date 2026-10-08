#!/usr/bin/env python3
"""Duplicate-free sensitivity: does the conclusion survive removing memorisable rows?

$57.53\\%$ of the official KDDCup99 test set occurs verbatim in training, and
$2.68\\%$ of NSL-KDD's. The interesting question is not only how far scores fall
when those rows are excluded --- a uniform drop changes nothing --- but whether
the **relative** conclusion moves. If latency leads rate on the official test set
and stops leading on the deduplicated remainder, the encoding claim was partly a
claim about memorisation.

Reports, per protocol:

    F1_official,  F1_deduplicated,  Delta F1

per (variant, seed), and then re-runs the ranking and the latency-vs-rate paired
comparison on both, so the two conclusions can be set side by side.

Duplicates are never removed from the evaluation of record: the canonical
protocol is reported and this runs beside it as a sensitivity check.

    python scripts/dedup_analysis.py
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata, wilcoxon

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.populations import (POPULATIONS, macro_f1_pair, members,  # noqa: E402
                             require_population)
from src.readout import discover  # noqa: E402

OUT_BASE = ROOT / "results/analysis/dedup"


def collect() -> pd.DataFrame:
    """Per-run official and duplicate-free macro-F1, on a fixed label universe.

    The stored `f1_macro` is model-dependent on CIC-IDS2017. U2R has zero test
    support there, so a model that never predicts it is scored over four
    classes while a model that predicts it once is scored over five. Delta is
    the former and latency and rate are the latter, which put three rows of the
    same table on two different denominators: delta read 0.6226 where its
    fixed-universe value is 0.4981, exactly 4/5 of it.

    `f1_official` is therefore recomputed from the saved predictions on the
    registered label universe, which is authoritative.

    `f1_dedup` cannot be recomputed here: the duplicate mask is not stored in
    the artifacts, only its cardinality. It is corrected instead by the ratio
    the official pair reveals, which is exact provided the classes missing from
    the stored average score zero and the same classes are missing from the
    duplicate-free average. Both hold when the missing class has no test
    support at all, since removing test rows can neither create support for it
    nor cause it to be predicted. The factor is detected per run rather than
    assumed, and recorded in `denominator_factor` so the correction is visible
    rather than silent.
    """
    rows = []
    for r in discover(require_artifacts=False):
        m = r.record()["test_plus"]["metrics"]
        d = m.get("duplicate_free")
        if not d:
            continue
        stored = m["f1_macro"]
        try:
            pre = r.prefix()
            pair = macro_f1_pair(pre["y_true"].astype(np.int64),
                                 pre["y_pred"].astype(np.int64),
                                 r.record()["data"]["num_classes"])
            official = pair["macro_f1_fixed_universe"]
        except Exception:                                         # noqa: BLE001
            official = stored
        factor = (official / stored) if stored else 1.0
        dedup = d["metrics"]["macro_f1"] * factor
        rows.append({
            "protocol": r.protocol, "neuron": r.neuron, "encoding": r.encoding,
            "variant": r.variant, "seed": r.seed,
            "pct_excluded": d["pct_excluded"],
            "f1_official": official,
            "f1_dedup": dedup,
            "delta_pp": 100.0 * (dedup - official),
            "denominator_factor": factor,
            "denominator_corrected": abs(factor - 1.0) > 1e-9,
            "mcc_official": m["matthews_corrcoef"],
            "mcc_dedup": d["metrics"]["mcc"],
        })
    return pd.DataFrame(rows)


def rank_on(df: pd.DataFrame, col: str) -> pd.Series:
    wide = df.pivot_table(index=["protocol", "seed"], columns="variant",
                          values=col, aggfunc="mean").dropna()
    if wide.empty or wide.shape[1] < 2:
        return pd.Series(dtype=float)
    ranks = np.apply_along_axis(lambda r: rankdata(-r), 1, wide.to_numpy())
    return pd.Series(ranks.mean(axis=0), index=wide.columns).sort_values()


def paired(df: pd.DataFrame, col: str, a: str, b: str) -> dict:
    wide = df.pivot_table(index=["protocol", "seed"], columns="variant",
                          values=col, aggfunc="mean").dropna()
    if not {a, b}.issubset(wide.columns) or len(wide) < 3:
        return {}
    x, y = wide[a].to_numpy(), wide[b].to_numpy()
    d = x - y
    stat, p = (float("nan"), 1.0) if np.allclose(d, 0) else wilcoxon(x, y)
    return {"n_blocks": int(len(d)), "wins_a": int((d > 0).sum()),
            "wins_b": int((d < 0).sum()),
            "median_delta_pp": float(np.median(d) * 100),
            "wilcoxon_p": float(p)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--population", default="encoding_axis_lp",
                    choices=sorted(POPULATIONS),
                    help="ranking population; a mixed set is refused")
    ap.add_argument("--a", default="LeakyParallel/latency")
    ap.add_argument("--b", default="LeakyParallel/rate")
    args = ap.parse_args()

    df = collect()
    if not df.empty:
        # Restrict BEFORE ranking. The earlier version pooled the nine
        # latency neurons with the three LeakyParallel encodings and reported a
        # "leader" for a cross-shaped set that is not a competitive field.
        keep = members(args.population)
        df = df[df.variant.isin(keep)].copy()
        require_population(df.variant.unique(), args.population)
    if df.empty:
        print("no runs carry a duplicate-free block yet.")
        return 0
    OUT = OUT_BASE / args.population
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT / "per_seed.csv", index=False)

    agg = (df.groupby(["protocol", "variant"])
             .agg(n=("seed", "size"),
                  pct_excluded=("pct_excluded", "first"),
                  f1_official=("f1_official", "mean"),
                  f1_official_sd=("f1_official", "std"),
                  f1_dedup=("f1_dedup", "mean"),
                  f1_dedup_sd=("f1_dedup", "std"),
                  delta_pp=("delta_pp", "mean"),
                  # Carried into the summary so a corrected row is visible in
                  # the table a reader actually sees, not only per seed.
                  denominator_factor=("denominator_factor", "first"),
                  denominator_corrected=("denominator_corrected", "any"))
             .reset_index())
    agg.to_csv(OUT / "summary.csv", index=False)

    print(f"=== deduplication sensitivity — population {args.population!r} "
          f"({POPULATIONS[args.population]}) ===")
    print("macro-F1 is the pre-registered fixed-universe metric; on "
          "cicids2017_v2 it is capped at 0.80 because U2R has zero test "
          "support (see AMENDMENT).\n")
    show = agg.copy()
    show["official"] = (show.f1_official.round(4).astype(str) + " ± "
                        + show.f1_official_sd.fillna(0).round(4).astype(str))
    show["dedup"] = (show.f1_dedup.round(4).astype(str) + " ± "
                     + show.f1_dedup_sd.fillna(0).round(4).astype(str))
    print(show[["protocol", "variant", "pct_excluded", "official", "dedup",
                "delta_pp"]].to_string(index=False))

    verdict = {}
    for proto, g in df.groupby("protocol"):
        r_off, r_ded = rank_on(g, "f1_official"), rank_on(g, "f1_dedup")
        if r_off.empty or r_ded.empty:
            continue
        same_leader = r_off.index[0] == r_ded.index[0]
        order_changed = list(r_off.index) != list(r_ded.index)
        verdict[proto] = {
            "leader_official": r_off.index[0],
            "leader_dedup": r_ded.index[0],
            "leader_unchanged": bool(same_leader),
            "full_order_changed": bool(order_changed),
            "mean_rank_official": {k: round(v, 3) for k, v in r_off.items()},
            "mean_rank_dedup": {k: round(v, 3) for k, v in r_ded.items()},
            "paired_official": paired(g, "f1_official", args.a, args.b),
            "paired_dedup": paired(g, "f1_dedup", args.a, args.b),
        }
        print(f"\n--- {proto} ---")
        print(f"  leader official : {r_off.index[0]}  (mean rank {r_off.iloc[0]:.2f})")
        print(f"  leader dedup    : {r_ded.index[0]}  (mean rank {r_ded.iloc[0]:.2f})")
        print(f"  leader unchanged: {same_leader}   full order changed: {order_changed}")
        for label, key in (("official", "paired_official"), ("dedup", "paired_dedup")):
            t = verdict[proto][key]
            if t:
                print(f"  {args.a} vs {args.b} [{label}]: "
                      f"{t['wins_a']}-{t['wins_b']} blocks, "
                      f"median {t['median_delta_pp']:+.3f} pp, p={t['wilcoxon_p']:.4f}")

    (OUT / "verdict.json").write_text(json.dumps(
        {"population": args.population,
         "population_description": POPULATIONS[args.population],
         "per_protocol": verdict}, indent=2))
    print(f"\nwrote {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
