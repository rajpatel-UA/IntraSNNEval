#!/usr/bin/env python3
"""Input sparsity, and the closed-form rate/latency input-spike ratio.

Both outputs are measured on each confirmation protocol's test matrix exactly as
the network receives it: train-only encoding, min-max scaled to [0, 1].

    feature_sparsity.csv        d; features exactly zero per sample, and as a
                                percent of d; and `band_per_sample`, the features
                                per sample in (0, THETA_LAT], which latency's own
                                gate removes. That column is NOT the band between
                                the two gates; appendix_data.py renames it on
                                read for that reason
    xbar_active_prediction.csv  mean value of the active features (x > 0, and
                                x > THETA_LAT); the predicted rate/latency input
                                ratio; the ratio measured from the LeakyParallel
                                runs; and the absolute percent error

The measured ratio is taken from spike_sop_decomposition.csv (five-seed means
of input spikes per sample, written at six decimals), so that script must run
first. Using the unrounded means instead moves the measured ratio by at most
4e-7 and the percent error in its third significant figure; no reported value
changes.

The prediction is an identity derived from the two encoders, not a fit: rate
emits T*x expected spikes per feature and latency one spike per feature above
its gate, so

    ratio = T * sum_{x>0} x / |{x > THETA_LAT}|

independent of dimensionality and sparsity. Sums are accumulated in float64; the
matrices are float32, and accumulating in float32 moves the prediction in the
eighth significant figure.

Both files were first committed on 2026-08-14 from an analysis run inline. This
script was written afterwards and regenerates them byte for byte. Loading a
protocol rewrites its split manifest under results/manifests/; the manifest is
deterministic, so the rewrite is byte-identical.

    python scripts/feature_sparsity.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

OUT = ROOT / "results/analysis"
PROTOCOLS = ["nslkdd_v2", "kddcup99_v2", "cicids2017_v2",
             "ctu13_v2_f0", "ctu13_v2_f1"]
#: Latency's amplitude gate and the time budget, both as in the confirmation
#: sweep (src/encoding/latency.py default, configs/*/config.yaml).
THETA_LAT = 0.01
T = 25


def test_matrix(protocol: str) -> np.ndarray:
    """Test features exactly as the network receives them."""
    from src.data import get_dataset
    b = get_dataset(protocol, batch_size=16384, seed=42)
    return np.concatenate([X.numpy() for X, _ in b.test_loader], axis=0)


def measured_ratios() -> pd.Series:
    """Rate over latency input spikes per sample, LeakyParallel, seed means."""
    d = (pd.read_csv(OUT / "spike_sop_decomposition.csv")
         .pivot(index="protocol", columns="encoding", values="input_spikes"))
    return d.rate / d.latency


def main() -> int:
    measured = measured_ratios()
    sp, xb = [], []
    for proto in PROTOCOLS:
        x = test_matrix(proto).astype(np.float64)
        d = x.shape[1]
        zero = (x == 0).sum(axis=1).mean()
        sp.append({"protocol": proto, "d": d,
                   "zero_per_sample": zero,
                   "zero_pct": 100.0 * zero / d,
                   "band_per_sample": ((x > 0) & (x <= THETA_LAT)).sum(axis=1).mean()})
        pred = T * x[x > 0].sum() / (x > THETA_LAT).sum()
        meas = measured[proto]
        xb.append({"protocol": proto, "d": d,
                   "xbar_active_gt0": x[x > 0].mean(),
                   "xbar_active_gt001": x[x > THETA_LAT].mean(),
                   "pred_ratio": pred, "measured_ratio": meas,
                   "abs_err_pct": 100.0 * abs(pred - meas) / meas})

    sp, xb = pd.DataFrame(sp), pd.DataFrame(xb)
    sp.to_csv(OUT / "feature_sparsity.csv", index=False, float_format="%.8f")
    xb.to_csv(OUT / "xbar_active_prediction.csv", index=False,
              float_format="%.8f")
    print(sp.round(3).to_string(index=False))
    print(xb[["protocol", "pred_ratio", "measured_ratio", "abs_err_pct"]]
          .round(5).to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
