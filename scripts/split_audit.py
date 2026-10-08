#!/usr/bin/env python3
"""Build every v2 protocol, run its audit gate, and tabulate the result.

This is Gate 1 of the execution plan: nothing trains until this table exists and
its disjointness column is clean. Building a protocol writes its manifest as a
side effect (`src/data/protocols._emit_manifest`), so the table is assembled
from the manifests rather than from a second, separately-maintained code path
that could disagree with what the loaders actually did.

The duplicate column is reported, never asserted to zero. KDDCup99 genuinely
repeats most of its test set inside training — that is the documented reason
NSL-KDD exists — and a paper that quietly omits the number is worse than one
that prints it.

    python scripts/split_audit.py                 # build all, write the table
    python scripts/split_audit.py --from-cache    # tabulate existing manifests
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data import get_dataset  # noqa: E402

PROTOCOLS = ["nslkdd_v2", "kddcup99_v2", "cicids2017_v2",
             "cicids2017_daydisjoint"] + [f"ctu13_v2_f{f}" for f in range(4)]

MANIFESTS = ROOT / "results/manifests"
OUT = ROOT / "results/analysis"

PRETTY = {
    "nslkdd_v2": ("NSL-KDD", "official split"),
    "kddcup99_v2": ("KDDCup99", "official split"),
    "cicids2017_v2": ("CIC-IDS2017", "within-capture order-disjoint"),
    "cicids2017_daydisjoint": ("CIC-IDS2017", "day-disjoint (binary)"),
}
for _f in range(4):
    PRETTY[f"ctu13_v2_f{_f}"] = ("CTU-13", f"scenario-disjoint fold {_f}")


def manifest_for(proto: str) -> Path:
    """Manifest names are keyed on (dataset, protocol) as the loader wrote them,
    which is not always the registry name — ctu13_v2_f0 is stored under its
    condition name because the condition, not the alias, is what was run."""
    direct = {
        "nslkdd_v2": "nslkdd_v2.json",
        "kddcup99_v2": "kddcup99_v2.json",
        "cicids2017_v2": "cicids2017_v2.json",
        "cicids2017_daydisjoint": "cicids2017_daydisjoint.json",
    }
    if proto in direct:
        return MANIFESTS / direct[proto]
    if proto.startswith("ctu13_v2_f"):
        return MANIFESTS / f"ctu13_causal_scenario_f{proto[-1]}.json"
    return MANIFESTS / f"{proto}.json"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--from-cache", action="store_true",
                    help="skip building; tabulate whatever manifests exist")
    args = ap.parse_args()

    rows = []
    for proto in PROTOCOLS:
        mp = manifest_for(proto)
        if not args.from_cache:
            try:
                get_dataset(proto, batch_size=4096, seed=42)
            except Exception as e:                      # noqa: BLE001
                print(f"  [{proto}] BUILD FAILED: {type(e).__name__}: {e}",
                      flush=True)
                rows.append({"protocol": proto, "status": f"FAILED: {e}"})
                continue
        if not mp.exists():
            print(f"  [{proto}] no manifest at {mp}", flush=True)
            continue
        a = json.load(open(mp))
        ds, desc = PRETTY.get(proto, (proto, ""))
        rows.append({
            "dataset": ds, "protocol": desc, "registry_name": proto,
            "n_train": a["n_train"], "n_val": a["n_val"], "n_test": a["n_test"],
            "group": a.get("group_name") or "-",
            "group_disjoint_claimed": a["group_disjoint_claimed"],
            "overlap_train_test": a["overlap_train_test"],
            "overlap_train_val": a["overlap_train_val"],
            "overlap_val_test": a["overlap_val_test"],
            "dup_test_rows_in_train": a["duplicate_test_rows_in_train"],
            "dup_test_pct": round(100 * a["duplicate_test_fraction"], 2),
            "classes_missing_from_train": a["classes_in_test_absent_from_train"],
            "scaler_fitted_on_test": a["scaler_fitted_on_test"],
            "vocabulary_fitted_on_test": a["vocabulary_fitted_on_test"],
            "gate_passes": a["passes"],
        })

    df = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT / "split_audit.csv", index=False)
    print("\n" + df.to_string(index=False))
    print(f"\nwrote {OUT / 'split_audit.csv'}")

    if "gate_passes" in df and not df["gate_passes"].fillna(False).all():
        print("\nGATE 1 NOT CLEAN — do not launch training.")
        sys.exit(1)
    print("\nGate 1 clean: no protocol violates a disjointness it claims.")


if __name__ == "__main__":
    main()
