#!/usr/bin/env python3
"""Bidirectional class-support audit, written as a sidecar.

Gate 1 checked only classes present in test but absent from train. It therefore
missed the opposite direction, and CIC-IDS2017's order-disjoint test interval
turns out to contain **no U2R examples at all** --- so the five-class macro-F1
carries a fixed zero for that class on every CIC model.

Two design decisions here matter more than the check itself.

**The frozen manifests are not rewritten.** All 235 completed runs record the
manifest hash of the split that produced them. Regenerating those files to add a
field would change their hashes and manufacture a reproducibility failure while
fixing an auditing one. The audit is written beside them instead, carrying the
original hash so the two can always be tied together.

**Zero test support is a warning, not a gate failure.** A genuine order- or
time-respecting split can legitimately reach an interval in which some class
does not occur; that is a property of the data, not a bug to be engineered
away. The gate's job is to make it impossible to overlook, not to force the
partition to contain something it naturally does not.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.paths import RESULTS_ROOT  # noqa: E402

MANIFESTS = RESULTS_ROOT / "manifests"
SIDECAR = MANIFESTS / "support_audit"
AUDIT_VERSION = 2

KDD_CLASSES = ["normal", "dos", "probe", "r2l", "u2r"]
CTU_CLASSES = ["normal", "botnet"]
BIN_CLASSES = ["normal", "attack"]


def kdd_split(dataset: str):
    from src.data.protocols import _carve_val, _kdd_frames
    from src.data.nslkdd import _label_to_class_idx
    df = _kdd_frames(dataset)
    y = df["label"].map(_label_to_class_idx).astype(np.int64).to_numpy()
    split = df["_split"].to_numpy()
    pool = np.where(split == "train")[0]
    tr, va = _carve_val(pool, y[pool])
    return y, tr, va, np.where(split == "test")[0], KDD_CLASSES


def cic_split(binary: bool):
    from src.data.protocols import CIC_EARLY_DAYS, _carve_val, _cic_frame
    df = _cic_frame()
    y = df["y"].to_numpy(np.int64)
    if binary:
        yb = (y != 0).astype(np.int64)
        early = df["_capture"].isin(CIC_EARLY_DAYS).to_numpy()
        pool = np.where(early)[0]
        tr, va = _carve_val(pool, yb[pool])
        return yb, tr, va, np.where(~early)[0], BIN_CLASSES
    tr, va, te = [], [], []
    for _, g in df.groupby("_capture_idx", sort=True):
        pos = g.index.to_numpy()[np.argsort(g["_row_in_capture"].to_numpy())]
        n = len(pos)
        a, b = int(0.70 * n), int(0.80 * n)
        tr.append(pos[:a]); va.append(pos[a:b]); te.append(pos[b:])
    return (y, *(np.sort(np.concatenate(x)) for x in (tr, va, te)), KDD_CLASSES)


def ctu_split(fold: int):
    from src.data.ctu13_causal import load_frozen_folds
    from src.data.ctu13_conditions import FOLDS_JSON, PQ
    d = pd.read_parquet(PQ, columns=["y", "scenario"])
    f = load_frozen_folds(FOLDS_JSON)["folds"][int(fold)]
    y = d["y"].to_numpy(np.int64)
    sc = d["scenario"]
    return (y,
            np.where(sc.isin(f["train_scenarios"]))[0],
            np.where(sc.isin(f["val_scenarios"]))[0],
            np.where(sc.isin(f["test_scenarios"]))[0],
            CTU_CLASSES)


PROTOCOLS = {
    "nslkdd_v2": (lambda: kdd_split("nslkdd"), "nslkdd_v2.json"),
    "kddcup99_v2": (lambda: kdd_split("kddcup99"), "kddcup99_v2.json"),
    "cicids2017_v2": (lambda: cic_split(False), "cicids2017_v2.json"),
    "cicids2017_daydisjoint": (lambda: cic_split(True),
                               "cicids2017_daydisjoint.json"),
    **{f"ctu13_v2_f{f}": ((lambda f=f: ctu_split(f)),
                          f"ctu13_causal_scenario_f{f}.json") for f in range(4)},
}


def main() -> int:
    SIDECAR.mkdir(parents=True, exist_ok=True)
    rows, warnings = [], []
    for proto, (build, manifest_name) in PROTOCOLS.items():
        y, tr, va, te, names = build()
        sup = {p: {names[i]: int((y[idx] == i).sum()) for i in range(len(names))}
               for p, idx in (("train", tr), ("val", va), ("test", te))}
        missing_test = [c for c in names if sup["test"][c] == 0]
        missing_train = [c for c in names if sup["train"][c] == 0]
        missing_val = [c for c in names if sup["val"][c] == 0]

        mpath = MANIFESTS / manifest_name
        frozen_hash = None
        if mpath.exists():
            import hashlib
            frozen_hash = hashlib.sha256(mpath.read_bytes()).hexdigest()

        payload = {
            "protocol": proto,
            "support_audit_version": AUDIT_VERSION,
            "split_manifest": manifest_name,
            "split_manifest_hash": frozen_hash,
            "note": ("Sidecar. The frozen manifest is NOT rewritten: every "
                     "completed run records its hash, and changing it would "
                     "manufacture a reproducibility failure."),
            "label_universe": names,
            "support": sup,
            "zero_support_in_test": missing_test,
            "zero_support_in_train": missing_train,
            "zero_support_in_val": missing_val,
            "macro_f1_ceiling_fixed_universe": round(
                1.0 - len(missing_test) / len(names), 4),
        }
        (SIDECAR / f"{proto}.json").write_text(json.dumps(payload, indent=2))
        rows.append({"protocol": proto, "classes": len(names),
                     "zero_in_train": ",".join(missing_train) or "-",
                     "zero_in_val": ",".join(missing_val) or "-",
                     "zero_in_test": ",".join(missing_test) or "-",
                     "macro_f1_ceiling": payload["macro_f1_ceiling_fixed_universe"]})
        if missing_test:
            warnings.append(
                f"{proto}: classes {missing_test} have ZERO test support. "
                f"Fixed-universe macro-F1 is capped at "
                f"{payload['macro_f1_ceiling_fixed_universe']:.2f}.")

    df = pd.DataFrame(rows)
    df.to_csv(RESULTS_ROOT / "analysis/support_audit.csv", index=False)
    print(df.to_string(index=False))
    if warnings:
        print("\nWARNINGS (not gate failures — an order-respecting split may "
              "legitimately miss a class):")
        for w in warnings:
            print("  ! " + w)
    print(f"\nsidecars: {SIDECAR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
