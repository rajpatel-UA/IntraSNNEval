#!/usr/bin/env python3
"""Spike and synaptic-operation accounting, entirely inside the spiking design space.

Per PREREGISTRATION.md §10 this paper makes no SNN-vs-ANN MAC comparison. The
efficiency question it can answer is the one it is scoped to: among snnTorch
configurations, which reaches a given detection quality for the fewest synaptic
operations?

Two things the submitted paper could not report:

* **The spike count is decomposed.** Its single "spikes/sample" exceeds the
  input dimensionality, so a reader cannot tell whether it counts input,
  hidden, output, or some sum. Input, hidden and output are now separate.
* **Cost is weighted by fanout.** A spike into a 128-wide projection is not the
  same event as a spike into a 5-wide one, and a raw spike count cannot see the
  difference. SOP = input_spikes x fanout_input + hidden_spikes x fanout_hidden.

Any energy figure printed here is an **operation-based proxy**, `SOP x E_AC`,
not a measurement. The constant must be cited to its original hardware source
in the manuscript, never to a reviewer or to this script.

    python scripts/efficiency.py --protocols nslkdd_v2 kddcup99_v2
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.readout import discover  # noqa: E402

OUT = ROOT / "results/analysis/efficiency"


def rows_for(run) -> dict:
    rec = run.record()
    tp = rec["test_plus"]
    sa = tp.get("spike_accounting")
    if not sa:
        return {}
    return {
        "protocol": run.protocol, "neuron": run.neuron,
        "encoding": run.encoding, "seed": run.seed,
        "f1_macro": tp["metrics"]["f1_macro"],
        "mcc": tp["metrics"]["matthews_corrcoef"],
        "input_spikes": sa["input_spikes_per_sample"],
        "hidden_spikes": sa["hidden_spikes_per_sample"],
        "output_spikes": sa["output_spikes_per_sample"],
        "total_spikes": sa["total_spikes_per_sample"],
        "sops": sa["sops_per_sample"],
        "fanout_input": sa["fanout_input"],
        "gated": sa["gated_cell"],
        "sparsity": tp["spike_stats"]["spike_sparsity"],
        "inference_ms": tp["inference_ms_per_sample"],
        "params": rec["model_size"]["trainable_parameters"],
    }


def pareto_mask(quality: np.ndarray, cost: np.ndarray) -> np.ndarray:
    keep = np.ones(len(quality), dtype=bool)
    for i in range(len(quality)):
        dominated = ((quality >= quality[i]) & (cost <= cost[i])
                     & ((quality > quality[i]) | (cost < cost[i])))
        keep[i] = not dominated.any()
    return keep


def plot_pareto(agg: pd.DataFrame, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {"latency": "#4C72B0", "rate": "#DD8452", "delta": "#55A868"}
    protos = sorted(agg.protocol.unique())
    fig, axes = plt.subplots(1, len(protos), figsize=(4.2 * len(protos), 3.4),
                             squeeze=False)
    for ax, proto in zip(axes[0], protos):
        sub = agg[agg.protocol == proto]
        for enc, g in sub.groupby("encoding"):
            ax.scatter(g.sops_mean, g.f1_mean, s=42, color=colors.get(enc, "#888"),
                       label=enc, edgecolor="none")
            for _, r in g.iterrows():
                ax.annotate(r.neuron, (r.sops_mean, r.f1_mean), fontsize=5.5,
                            xytext=(3, 3), textcoords="offset points")
        front = sub[sub.pareto].sort_values("sops_mean")
        ax.plot(front.sops_mean, front.f1_mean, ls="--", lw=1.0, color="#C44E52",
                zorder=0, label="Pareto front")
        ax.set_xscale("log")
        ax.set_xlabel("SOPs per sample (log)")
        ax.set_ylabel("macro-F1")
        ax.set_title(proto, fontsize=9)
        ax.legend(fontsize=6.5, frameon=False)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(path.with_suffix("." + ext), dpi=300)
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--protocols", nargs="+", default=None)
    ap.add_argument("--neurons", nargs="+", default=None)
    args = ap.parse_args()

    runs = discover(protocols=args.protocols, neurons=args.neurons)
    rows = [r for r in (rows_for(x) for x in runs) if r]
    if not rows:
        print("no runs with spike accounting yet.")
        return 0
    OUT.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "per_seed.csv", index=False)

    agg = (df.groupby(["protocol", "neuron", "encoding"])
             .agg(n=("seed", "size"),
                  f1_mean=("f1_macro", "mean"), f1_sd=("f1_macro", "std"),
                  input_spikes=("input_spikes", "mean"),
                  hidden_spikes=("hidden_spikes", "mean"),
                  output_spikes=("output_spikes", "mean"),
                  total_spikes=("total_spikes", "mean"),
                  sops_mean=("sops", "mean"), sops_sd=("sops", "std"),
                  sparsity=("sparsity", "mean"),
                  inference_ms=("inference_ms", "mean"),
                  params=("params", "first"))
             .reset_index())
    agg["pareto"] = False
    for proto, g in agg.groupby("protocol"):
        agg.loc[g.index, "pareto"] = pareto_mask(g.f1_mean.to_numpy(),
                                                 g.sops_mean.to_numpy())
    agg.to_csv(OUT / "summary.csv", index=False)
    plot_pareto(agg, OUT / "pareto_sops")

    print("=== spike decomposition and SOPs per sample (seed mean) ===")
    show = agg.copy()
    show["F1"] = (show.f1_mean.round(4).astype(str) + " ± "
                  + show.f1_sd.fillna(0).round(4).astype(str))
    print(show[["protocol", "neuron", "encoding", "F1", "input_spikes",
                "hidden_spikes", "output_spikes", "sops_mean", "sparsity",
                "pareto"]].round(1).to_string(index=False))

    # Encoding-level cost ratio, the comparison the paper is scoped to make.
    print("\n=== SOPs relative to latency, same neuron and protocol ===")
    base = (agg[agg.encoding == "latency"]
            .set_index(["protocol", "neuron"]).sops_mean)
    rel = agg.join(base.rename("lat_sops"), on=["protocol", "neuron"])
    rel["sops_vs_latency"] = rel.sops_mean / rel.lat_sops
    print(rel[["protocol", "neuron", "encoding", "sops_mean", "sops_vs_latency"]]
          .round(3).to_string(index=False))
    print(f"\nwrote {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
