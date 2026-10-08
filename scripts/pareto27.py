#!/usr/bin/env python3
"""Quality against synaptic operations for all 27 configurations.

One row per (confirmation protocol, configuration), five-seed means, written to
results/analysis/pareto27.csv. It feeds `fig_pareto27` in `make_figures.py` and
the \\FNParetoCost* macros in `make_frozen_numbers.py`, which quote NSL-KDD
only: SOP counts scale with input dimension, so a span pooled over protocols of
different width would not be a property of the design space.

Macro-F1 is the pre-registered fixed-universe metric, recomputed from each run's
saved per-sample predictions (results/analysis/AMENDMENT_cic_u2r_support.md).
`efficiency/summary.csv` carries the stored field instead, which on CIC-IDS2017
differs for configurations that never predict U2R.

The file was first committed on 2026-08-18 from an analysis run inline. This
script was written afterwards and regenerates it byte for byte.

    python scripts/pareto27.py
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
PROTOCOLS = ["nslkdd_v2", "kddcup99_v2", "cicids2017_v2",
             "ctu13_v2_f0", "ctu13_v2_f1"]


def main() -> int:
    rows = []
    for r in discover(protocols=PROTOCOLS):
        rec, pre = r.record(), r.prefix()
        m = macro_f1_pair(pre["y_true"].astype(np.int64),
                          pre["y_pred"].astype(np.int64),
                          rec["data"]["num_classes"])
        rows.append({"protocol": r.protocol, "variant": r.variant,
                     "neuron": r.neuron, "encoding": r.encoding, "seed": r.seed,
                     "f1": m["macro_f1_fixed_universe"],
                     "sops": rec["test_plus"]["spike_accounting"]["sops_per_sample"]})
    if not rows:
        print("no confirmation runs; nothing to analyse")
        return 0

    d = (pd.DataFrame(rows)
         .groupby(["protocol", "variant", "neuron", "encoding"])
         .agg(f1=("f1", "mean"), sops=("sops", "mean"), n=("seed", "nunique"))
         .reset_index())
    d.to_csv(OUT / "pareto27.csv", index=False, float_format="%.6f")

    thin = d[d.n < 5]
    print(f"{len(rows)} runs -> pareto27.csv ({len(d)} cells"
          + (f", {len(thin)} below five seeds)" if len(thin) else ")"))
    nsl = d[d.protocol == "nslkdd_v2"]
    print(f"NSL-KDD SOP span: {nsl.sops.min():,.0f} to {nsl.sops.max():,.0f} "
          f"({nsl.sops.max() / nsl.sops.min():.0f}x)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
