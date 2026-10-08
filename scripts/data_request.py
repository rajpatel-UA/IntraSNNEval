#!/usr/bin/env python3
"""Regenerate results/analysis/data_request/: the two files that back numbers.

Both feed manuscript macros through `make_frozen_numbers.py`:

    C1_per_class.csv            per-class precision, recall and F1 (mean, and
                                SD with ddof=1) for LeakyParallel/latency on the
                                five confirmation protocols, five seeds, from the
                                saved predictions. Supplies \\FNNslRtlRec,
                                \\FNNslUtrRec and \\FNNslTestN. Classes with no
                                test support score zero (CIC-IDS2017 U2R).
    C2_delta_spikes_per_t.npy   delta's input spikes per sample at each of the
                                T=25 timesteps on the NSL-KDD test set. Supplies
                                \\FNDeltaSpkNsl and \\FNDeltaSpkAfter: on a static
                                vector delta fires only at t=1, exactly where
                                x > THETA_DELTA, which this script asserts.

Both were first committed on 2026-08-14 from analyses run inline. This script
was written afterwards and regenerates both byte for byte. C2 loads the NSL-KDD
dataset.

    python scripts/data_request.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.readout import discover  # noqa: E402

OUT = ROOT / "results/analysis/data_request"
PROTOCOLS = ["nslkdd_v2", "kddcup99_v2", "cicids2017_v2",
             "ctu13_v2_f0", "ctu13_v2_f1"]
#: Delta's default amplitude gate and the time budget of the confirmation sweep.
THETA_DELTA = 0.05
T = 25


def c1_per_class() -> pd.DataFrame:
    rows = []
    for r in discover(protocols=PROTOCOLS, neurons=["LeakyParallel"],
                      encodings=["latency"]):
        rec, pre = r.record(), r.prefix()
        names = rec["data"]["class_names"]
        p, rc, f1, s = precision_recall_fscore_support(
            pre["y_true"], pre["y_pred"], labels=list(range(len(names))),
            zero_division=0)
        rows += [{"protocol": r.protocol, "class": n, "seed": r.seed,
                  "precision": p[i], "recall": rc[i], "f1": f1[i],
                  "support": s[i]} for i, n in enumerate(names)]
    return (pd.DataFrame(rows).groupby(["protocol", "class"])
            .agg(precision=("precision", "mean"), recall=("recall", "mean"),
                 f1_mean=("f1", "mean"), f1_sd=("f1", lambda v: v.std(ddof=1)),
                 support=("support", "first"))
            .reset_index())


def c2_delta_spikes_per_t() -> np.ndarray:
    import torch
    from src.data import get_dataset
    from src.encoding import get_encoder

    b = get_dataset("nslkdd_v2", batch_size=16384, seed=42)
    x = torch.cat([X for X, _ in b.test_loader])
    s = get_encoder("delta")(x, T)                      # [T, N, F]
    n, f = x.shape
    # The claim the per-timestep profile stands for, checked on every pair
    # rather than inferred from the mean.
    if not bool((s[0].bool() == (x > THETA_DELTA)).all()) or bool(s[1:].any()):
        raise SystemExit("delta is no longer one-step thresholding on static "
                         "input; the manuscript's derivation does not hold")
    print(f"C2: delta at t=1 equals x > {THETA_DELTA} on all {n * f:,} "
          f"feature-sample pairs ({n:,} x {f}); no spikes after t=1")
    return s.numpy().astype(np.float64).sum(axis=(1, 2)) / n


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    c1_per_class().to_csv(OUT / "C1_per_class.csv", index=False,
                          float_format="%.8f")
    np.save(OUT / "C2_delta_spikes_per_t.npy", c2_delta_spikes_per_t())
    print(f"data request package -> {OUT}")
    for p in sorted(OUT.iterdir()):
        print(f"  {p.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
