#!/usr/bin/env python3
"""NSL-KDD detection on attack types seen in training vs types that are not.

The official KDDTest+ deliberately contains attack types absent from KDDTrain+,
which is the property that makes it a generalisation benchmark rather than a
held-out sample. Aggregate DR averages the two regimes together and hides which
one the detector is actually failing on. Splitting them costs no training: the
predictions are already saved, and the attack type per test row is recoverable
from the raw file in the frozen partition order.

The index mapping is the load-bearing part. `load_kdd_v2` selects test rows as
``np.where(split == "test")[0]``, which is ascending, and the artifact's ``idx``
is ``0..N-1`` over exactly that selection — so the k-th saved prediction belongs
to the k-th test row in file order. The script asserts the row count matches
before using it, because a silent misalignment here would produce a plausible
and completely wrong table.

    python scripts/nsl_known_unseen.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.nslkdd import ATTACK_TO_CLASS, NSLKDD_COLUMNS  # noqa: E402
from src.paths import dataset_dir  # noqa: E402
from src.readout import discover  # noqa: E402

OUT = ROOT / "results/analysis/known_unseen"
PROTOCOL = "nslkdd_v2"


def label_context():
    """Per-test-row attack type, and whether that type occurs in training."""
    d = dataset_dir("nslkdd")
    rd = lambda p: pd.read_csv(p, header=None, names=NSLKDD_COLUMNS)
    train_types = set(rd(d / "KDDTrain+.txt")["label"].astype(str))
    test = rd(d / "KDDTest+.txt")
    types = test["label"].astype(str).to_numpy()
    is_attack = types != "normal"
    known = np.array([t in train_types for t in types])
    coarse = np.array([ATTACK_TO_CLASS.get(t, "r2l") for t in types])
    return types, is_attack, known, coarse


def main() -> int:
    types, is_attack, known, coarse = label_context()
    runs = discover(protocols=[PROTOCOL])
    if not runs:
        print("no nslkdd_v2 runs with artifacts yet.")
        return 0
    OUT.mkdir(parents=True, exist_ok=True)

    n_unseen_types = sorted({t for t, k, a in zip(types, known, is_attack)
                             if a and not k})
    print(f"KDDTest+ contains {len(n_unseen_types)} attack types absent from "
          f"KDDTrain+: {', '.join(n_unseen_types)}")
    print(f"  attack rows: {int(is_attack.sum())}  "
          f"of which unseen-type: {int((is_attack & ~known).sum())} "
          f"({100*(is_attack & ~known).sum()/is_attack.sum():.1f}%)\n")

    rows = []
    for r in runs:
        pre = r.prefix()
        pred = pre["y_pred"].astype(np.int64)
        if len(pred) != len(types):
            raise SystemExit(
                f"{r.path}: {len(pred)} predictions but KDDTest+ has "
                f"{len(types)} rows — the artifact does not describe this split.")
        flagged = pred != 0                      # predicted as some attack
        for name, sel in (("known", is_attack & known),
                          ("unseen", is_attack & ~known)):
            rows.append({
                "protocol": r.protocol, "neuron": r.neuron,
                "encoding": r.encoding, "seed": r.seed,
                "regime": name, "n": int(sel.sum()),
                "DR": float(flagged[sel].mean()) if sel.any() else None,
                # Correct coarse class, not just "flagged as something"
                "correct_class": float(
                    (pred[sel] == pd.Series(coarse[sel]).map(
                        {c: i for i, c in enumerate(
                            ["normal", "dos", "probe", "r2l", "u2r"])}
                    ).to_numpy()).mean()) if sel.any() else None,
            })

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "per_seed.csv", index=False)
    agg = (df.groupby(["protocol", "neuron", "encoding", "regime"])
             .agg(n=("n", "first"), DR_mean=("DR", "mean"),
                  DR_sd=("DR", "std"),
                  class_acc_mean=("correct_class", "mean"))
             .reset_index())
    agg.to_csv(OUT / "known_unseen.csv", index=False)

    wide = agg.pivot_table(index=["neuron", "encoding"], columns="regime",
                           values="DR_mean")
    if {"known", "unseen"}.issubset(wide.columns):
        wide["gap_pp"] = 100 * (wide["known"] - wide["unseen"])
    print("=== detection rate by attack-type regime (seed mean) ===")
    print(wide.round(4).to_string())
    print(f"\nwrote {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
