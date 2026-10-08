#!/usr/bin/env python3
"""Where the spikes and synaptic operations of LeakyParallel come from.

One row per (confirmation protocol, encoding), five-seed means, written to
results/analysis/spike_sop_decomposition.csv. `make_frozen_numbers.py` reads the
input share of SOPs and the hidden share of spikes from it for the efficiency
paragraph.

SOP = input_spikes x fanout_input + hidden_spikes x fanout_hidden, the
definition every run stores in its `spike_accounting` block. Output spikes are
the readout and drive nothing downstream. The hidden fanout is the readout
width, 5 on the five-class protocols and 2 on binary CTU-13; an earlier ad-hoc
calculation assumed 5 everywhere and got the CTU-13 shares wrong.

Shares are ratios of the seed means, not means of the per-seed ratios: SOPs and
spike counts are additive, so the share of the mean is the share of the
expected operation count. Every row is a mean over seeds 42-46, never a single
seed: the `seed` column holds 44 only because the original computation averaged
every numeric column, seed included. It is kept so the committed file
regenerates unchanged, and must not be read as "measured on seed 44".

The file was first committed on 2026-08-14 from an analysis run inline. This
script was written afterwards and regenerates it byte for byte.

    python scripts/spike_sop_decomposition.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.readout import discover  # noqa: E402

OUT = ROOT / "results/analysis"
PROTOCOLS = ["nslkdd_v2", "kddcup99_v2", "cicids2017_v2",
             "ctu13_v2_f0", "ctu13_v2_f1"]
#: Largest allowed |SOP - input SOPs - hidden SOPs|; the manuscript quotes it.
SOP_TOLERANCE = 1e-6


def main() -> int:
    rows = []
    for r in discover(protocols=PROTOCOLS, neurons=["LeakyParallel"],
                      require_artifacts=False):
        rec = r.record()
        sa = rec["test_plus"]["spike_accounting"]
        rows.append({"protocol": r.protocol, "encoding": r.encoding,
                     "seed": r.seed,
                     "input_spikes": sa["input_spikes_per_sample"],
                     "hidden_spikes": sa["hidden_spikes_per_sample"],
                     "output_spikes": sa["output_spikes_per_sample"],
                     "fanout_input": sa["fanout_input"],
                     "fanout_hidden": sa["fanout_hidden"],
                     "sops": sa["sops_per_sample"],
                     "n_features": rec["data"]["num_features"]})
    if not rows:
        print("no LeakyParallel runs; nothing to analyse")
        return 0

    d = pd.DataFrame(rows).groupby(["protocol", "encoding"]).mean().reset_index()
    d["input_sops"] = d.input_spikes * d.fanout_input
    d["hidden_sops"] = d.hidden_spikes * d.fanout_hidden
    d["total_spikes"] = d.input_spikes + d.hidden_spikes + d.output_spikes
    d["hidden_share_pct"] = 100.0 * d.hidden_spikes / d.total_spikes
    d["input_share_sops_pct"] = 100.0 * d.input_sops / d.sops

    # The stored SOP must be exactly the sum of its two parts; if it is not, the
    # accounting definition changed under the runs and the shares mean nothing.
    err = (d.sops - d.input_sops - d.hidden_sops).abs().max()
    if err > SOP_TOLERANCE:
        raise SystemExit(f"stored SOPs disagree with input+hidden by {err:.3e}")

    d.to_csv(OUT / "spike_sop_decomposition.csv", index=False,
             float_format="%.6f")
    print(f"{len(rows)} runs -> spike_sop_decomposition.csv ({len(d)} rows)")
    print(d[["protocol", "encoding", "hidden_share_pct", "input_share_sops_pct"]]
          .round(1).to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
