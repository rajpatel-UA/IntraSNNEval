#!/usr/bin/env python3
"""Emit both macro-F1 variants for every run, with the support facts attached.

After the support audit, `macro_f1` alone is no longer an acceptable label on
CIC-IDS2017: its order-disjoint test interval contains no U2R examples, so the
pre-registered five-class metric carries a fixed zero for that class and is
capped at 0.80.

    macro_f1_fixed_universe   the PRE-REGISTERED PRIMARY metric, unchanged.
                              Drives every ranking and hypothesis test.
    macro_f1_test_present     descriptive only, added post-audit. Averages over
                              classes with nonzero test support. Never used for
                              selection or testing.

Because an absent class contributes the same zero to every model on a protocol,
the two differ by the constant n_present/n_classes and induce identical
within-protocol rankings. This script asserts that identity rather than assuming
it, since the whole argument for reporting both rests on it.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.populations import macro_f1_pair  # noqa: E402
from src.readout import discover  # noqa: E402

OUT = ROOT / "results/analysis"


def main() -> int:
    rows = []
    for r in discover():
        pre = r.prefix()
        n_cls = r.record()["data"]["num_classes"]
        m = macro_f1_pair(pre["y_true"].astype(np.int64),
                          pre["y_pred"].astype(np.int64), n_cls)
        rows.append({"protocol": r.protocol, "neuron": r.neuron,
                     "encoding": r.encoding, "variant": r.variant,
                     "seed": r.seed, **m})
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "dual_macro_f1_per_seed.csv", index=False)

    agg = (df.groupby(["protocol", "variant"])
             .agg(n=("seed", "size"),
                  fixed=("macro_f1_fixed_universe", "mean"),
                  fixed_sd=("macro_f1_fixed_universe", "std"),
                  present=("macro_f1_test_present", "mean"),
                  present_sd=("macro_f1_test_present", "std"),
                  n_present=("n_classes_present", "first"),
                  n_fixed=("n_classes_fixed", "first"),
                  ceiling=("macro_f1_ceiling", "first"))
             .reset_index())
    agg["ratio"] = agg.present / agg.fixed
    agg.to_csv(OUT / "dual_macro_f1.csv", index=False)

    # The constant-factor identity is what licenses reporting both. Assert it.
    bad = agg[(agg.ceiling < 1.0)
              & ((agg.ratio - 1.0 / agg.ceiling).abs() > 1e-9)]
    if len(bad):
        print("!! the two metrics are NOT a constant factor apart:")
        print(bad.to_string(index=False))
        return 1

    affected = agg[agg.ceiling < 1.0]
    print("=== protocols where the two metrics differ ===")
    if affected.empty:
        print("  none — every protocol has full test support.")
    else:
        show = affected.copy()
        show["fixed (primary)"] = (show.fixed.round(4).astype(str) + " ± "
                                   + show.fixed_sd.fillna(0).round(4).astype(str))
        show["present (descriptive)"] = (
            show.present.round(4).astype(str) + " ± "
            + show.present_sd.fillna(0).round(4).astype(str))
        print(show[["protocol", "variant", "n_fixed", "n_present", "ceiling",
                    "fixed (primary)", "present (descriptive)"]]
              .to_string(index=False))
        print("\nidentity verified: present = fixed x n_fixed/n_present on every "
              "affected row, so within-protocol ordering is unchanged.")
    print(f"\nwrote {OUT / 'dual_macro_f1.csv'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
