#!/usr/bin/env python3
"""Operational security evaluation from saved artifacts. Trains nothing.

Three analyses the review asked for, none needing a new run:

**Fixed-FAR operating points.** A 2% false-alarm rate is not obviously small: on
a link carrying a million benign flows a shift it is twenty thousand alerts. So
rather than reporting whatever FAR the argmax happens to produce, thresholds are
chosen on **validation** to hit 1% and 0.1% FAR, frozen, and then applied to
test. The realised test FAR is reported beside the target, because a threshold
that transfers badly is itself a finding. Choosing the threshold on test would
be selecting the operating point using the data it is reported on
(PREREGISTRATION.md §12).

**Per-class detection.** Aggregate DR hides total failure on a rare class.
Support is printed beside every metric so an F1 computed on nine samples is not
read as equivalent to one computed on hundreds of thousands.

**Duplicate-free sensitivity.** Emitted by the trainer for every v2 run via the
loader-published mask; this collects it into one table. It matters most on
KDDCup99, where 57.5% of the official test set occurs verbatim in training.

    python scripts/security_analysis.py --protocols kddcup99_v2
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.readout import attack_score, discover, threshold_at_far  # noqa: E402

OUT = ROOT / "results/analysis/security"
TARGET_FARS = [0.01, 0.001]
NORMAL = 0


def fixed_far_rows(run) -> list:
    """Validation-selected thresholds, frozen, then evaluated on test."""
    val, pre = run.val(), run.prefix()
    v_score = attack_score(val["score"], NORMAL)
    t_score = attack_score(pre["cum_out"][:, -1, :].astype(np.float64), NORMAL)
    y_val, y_te = val["y_true"].astype(np.int64), pre["y_true"].astype(np.int64)
    te_attack, te_benign = y_te != NORMAL, y_te == NORMAL

    rows = []
    for target in TARGET_FARS:
        thr = threshold_at_far(v_score, y_val, target, NORMAL)
        flag = t_score >= thr
        rows.append({
            "protocol": run.protocol, "neuron": run.neuron,
            "encoding": run.encoding, "seed": run.seed,
            "target_far": target, "threshold": thr,
            "val_far": float((v_score[y_val == NORMAL] >= thr).mean()),
            "test_far": float(flag[te_benign].mean()) if te_benign.any() else None,
            "test_dr": float(flag[te_attack].mean()) if te_attack.any() else None,
            "fp_per_10k_benign": (float(flag[te_benign].mean() * 10_000)
                                  if te_benign.any() else None),
            "n_benign_test": int(te_benign.sum()),
            "n_attack_test": int(te_attack.sum()),
        })
    return rows


def per_class_rows(run) -> list:
    from sklearn.metrics import precision_recall_fscore_support
    rec = run.record()
    names = rec["data"]["class_names"]
    pre = run.prefix()
    y, p = pre["y_true"].astype(np.int64), pre["y_pred"].astype(np.int64)
    labels = list(range(len(names)))
    pr, rc, f1, sup = precision_recall_fscore_support(
        y, p, labels=labels, zero_division=0)
    return [{"protocol": run.protocol, "neuron": run.neuron,
             "encoding": run.encoding, "seed": run.seed, "class": names[i],
             "precision": pr[i], "recall": rc[i], "f1": f1[i],
             "support": int(sup[i])} for i in labels]


def dupfree_rows(run) -> list:
    rec = run.record()
    # compute_metrics nests its context-driven breakdowns inside the metrics
    # block, not beside it.
    d = rec["test_plus"]["metrics"].get("duplicate_free")
    if not d:
        return []
    return [{"protocol": run.protocol, "neuron": run.neuron,
             "encoding": run.encoding, "seed": run.seed,
             "pct_excluded": d["pct_excluded"],
             "n_excluded": d["n_excluded"],
             "macro_f1_full": rec["test_plus"]["metrics"]["f1_macro"],
             "macro_f1_dupfree": d["metrics"]["macro_f1"],
             "delta_macro_f1_pp": d["delta_macro_f1_pp"],
             "DR_dupfree": d["metrics"]["DR"],
             "FAR_dupfree": d["metrics"]["FAR"],
             "delta_FAR_pp": d["delta_FAR_pp"]}]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--protocols", nargs="+", default=None)
    ap.add_argument("--neurons", nargs="+", default=None)
    args = ap.parse_args()

    runs = discover(protocols=args.protocols, neurons=args.neurons)
    if not runs:
        print("no runs with artifacts found yet — nothing to analyse.")
        return 0
    OUT.mkdir(parents=True, exist_ok=True)

    far, cls, dup = [], [], []
    for r in runs:
        far += fixed_far_rows(r)
        cls += per_class_rows(r)
        dup += dupfree_rows(r)

    far_df, cls_df = pd.DataFrame(far), pd.DataFrame(cls)
    far_df.to_csv(OUT / "fixed_far_per_seed.csv", index=False)
    cls_df.to_csv(OUT / "per_class_per_seed.csv", index=False)

    far_agg = (far_df.groupby(["protocol", "neuron", "encoding", "target_far"])
                     .agg(n=("seed", "size"),
                          test_dr_mean=("test_dr", "mean"),
                          test_dr_sd=("test_dr", "std"),
                          test_far_mean=("test_far", "mean"),
                          test_far_sd=("test_far", "std"),
                          val_far_mean=("val_far", "mean"),
                          fp_per_10k=("fp_per_10k_benign", "mean"))
                     .reset_index())
    far_agg.to_csv(OUT / "fixed_far.csv", index=False)

    cls_agg = (cls_df.groupby(["protocol", "neuron", "encoding", "class"])
                     .agg(n=("seed", "size"),
                          precision=("precision", "mean"),
                          recall=("recall", "mean"),
                          f1_mean=("f1", "mean"), f1_sd=("f1", "std"),
                          support=("support", "first"))
                     .reset_index())
    cls_agg.to_csv(OUT / "per_class.csv", index=False)

    # All four quantities together. Reporting realised test FAR beside the
    # validation target is what makes this hard to criticise: the test set is
    # never used to choose the threshold, so any gap between the two columns is
    # operating-point drift under distribution shift, measured rather than
    # avoided.
    print("=== fixed-FAR operating points (thresholds chosen on validation, "
          "frozen, then applied to test) ===")
    show = far_agg.copy()
    show["val target FAR"] = (show.target_far * 100).map(lambda v: f"{v:g}%")
    show["val realised FAR"] = (100 * show.val_far_mean).round(3).astype(str) + "%"
    show["test realised FAR"] = (100 * show.test_far_mean).round(3).astype(str) + "%"
    show["test DR"] = ((100 * show.test_dr_mean).round(2).astype(str) + " ± "
                       + (100 * show.test_dr_sd.fillna(0)).round(2).astype(str))
    print(show[["protocol", "neuron", "encoding", "val target FAR",
                "val realised FAR", "test realised FAR", "test DR",
                "fp_per_10k"]].to_string(index=False))

    print("\n=== per-class F1 (seed mean, support in parentheses) ===")
    piv = cls_agg.pivot_table(index=["protocol", "neuron", "encoding"],
                              columns="class", values="f1_mean")
    print(piv.round(4).to_string())
    sup = cls_agg.groupby("class").support.first()
    print("support:", {k: int(v) for k, v in sup.items()})

    if dup:
        dup_df = pd.DataFrame(dup)
        dup_df.to_csv(OUT / "duplicate_free_per_seed.csv", index=False)
        dup_agg = (dup_df.groupby(["protocol", "neuron", "encoding"])
                         .agg(pct_excluded=("pct_excluded", "first"),
                              macro_f1_full=("macro_f1_full", "mean"),
                              macro_f1_dupfree=("macro_f1_dupfree", "mean"),
                              delta_pp=("delta_macro_f1_pp", "mean"))
                         .reset_index())
        dup_agg.to_csv(OUT / "duplicate_free.csv", index=False)
        print("\n=== duplicate-free sensitivity ===")
        print(dup_agg.round(4).to_string(index=False))
    else:
        print("\n(no duplicate-free block in these runs)")

    print(f"\nwrote {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
