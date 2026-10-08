#!/usr/bin/env python3
"""Anytime-inference analysis: how fast does each encoding accumulate evidence?

This is the mechanism result the encoding claim rests on. Terminal macro-F1 for
latency and rate is close and its dataset-level bootstrap interval includes
zero, so "latency is better" cannot carry the paper on quality alone. The
defensible question is instead whether latency coding reaches near-terminal
quality at a smaller temporal budget — and that is what T95 and T99 measure.

Reads only saved artifacts; trains nothing. Emits per (protocol, variant, seed):

    F1(t), MCC(t), DR(t), cumulative output spikes(t) for t = 1..T
    T95, T99 as fixed in PREREGISTRATION.md §11

and per (protocol, variant) the seed mean and SD.

    python scripts/early_readout.py --protocols nslkdd_v2 kddcup99_v2
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, matthews_corrcoef

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.readout import (convergence_step, discover, prefix_predictions)  # noqa: E402

OUT = ROOT / "results/analysis/early_readout"
ENC_ORDER = ["latency", "rate", "delta"]


def curves_for(run) -> pd.DataFrame:
    pre = run.prefix()
    cum, y = pre["cum_out"], pre["y_true"].astype(np.int64)
    preds = prefix_predictions(cum)
    T = cum.shape[1]
    # Output spikes emitted up to t, per sample, averaged — the temporal cost of
    # deciding at t. Cumulative by construction, so no running sum needed.
    spikes_t = cum.astype(np.int64).sum(axis=2).mean(axis=0)

    rows = []
    for t in range(T):
        p = preds[:, t]
        rows.append({
            "t": t + 1,
            "f1_macro": f1_score(y, p, average="macro", zero_division=0),
            "mcc": matthews_corrcoef(y, p) if len(np.unique(p)) > 1 else 0.0,
            "DR": float(((p != 0) & (y != 0)).sum() / max((y != 0).sum(), 1)),
            "FAR": float(((p != 0) & (y == 0)).sum() / max((y == 0).sum(), 1)),
            "output_spikes_cum": float(spikes_t[t]),
        })
    df = pd.DataFrame(rows)
    df.insert(0, "seed", run.seed)
    df.insert(0, "encoding", run.encoding)
    df.insert(0, "neuron", run.neuron)
    df.insert(0, "protocol", run.protocol)
    return df


def plot_curves(curves: pd.DataFrame, protocol: str, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sub = curves[curves.protocol == protocol]
    if sub.empty:
        return
    colors = {"latency": "#4C72B0", "rate": "#DD8452", "delta": "#55A868"}
    fig, axes = plt.subplots(1, 3, figsize=(11.4, 3.3))
    for enc in ENC_ORDER:
        g = sub[sub.encoding == enc]
        if g.empty:
            continue
        # Individual seeds, not a mean ± SD band. The band would imply a
        # symmetric spread the data does not have: most latency seeds converge
        # within a few steps and one does not, and averaging that into a ribbon
        # hides exactly the caveat a reader needs.
        for _, sg in g.groupby("seed"):
            sg = sg.sort_values("t")
            axes[0].plot(sg.t, sg.f1_macro, color=colors[enc], lw=0.7, alpha=0.45)
        m = g.groupby("t")[["f1_macro", "output_spikes_cum"]].median()
        axes[0].plot(m.index, m.f1_macro, color=colors[enc], label=enc, lw=1.9)
        axes[1].plot(m.output_spikes_cum, m.f1_macro, color=colors[enc],
                     label=enc, lw=1.6)

    # Per-seed T99, so the spread is a visible data point rather than a footnote
    for i, enc in enumerate(ENC_ORDER):
        g = sub[sub.encoding == enc]
        if g.empty:
            continue
        t99 = []
        for _, sg in g.groupby("seed"):
            f1 = sg.sort_values("t").f1_macro.to_numpy()
            v = convergence_step(f1, 0.99)
            if v:
                t99.append(v)
        axes[2].scatter([i] * len(t99), t99, color=colors[enc], s=34,
                        alpha=0.8, edgecolor="none")
        if t99:
            axes[2].hlines(np.median(t99), i - 0.22, i + 0.22,
                           color=colors[enc], lw=2.0)
    axes[2].set_xticks(range(len(ENC_ORDER)))
    axes[2].set_xticklabels(ENC_ORDER, fontsize=8)
    axes[2].set_ylabel("$T_{99}$ (timesteps)")
    axes[2].set_title("per-seed convergence step", fontsize=9)

    axes[0].set_xlabel("timestep $t$")
    axes[0].set_ylabel("macro-F1 at $t$")
    axes[0].set_title(f"{protocol}: evidence accumulation (per seed)", fontsize=9)
    axes[1].set_xlabel("cumulative output spikes per sample")
    axes[1].set_ylabel("macro-F1")
    axes[1].set_title("quality vs temporal cost (median)", fontsize=9)
    for ax in axes[:2]:
        ax.legend(fontsize=7, frameon=False)
    for ax in axes:
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(path.with_suffix("." + ext), dpi=300)
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--protocols", nargs="+", default=None)
    ap.add_argument("--neurons", nargs="+", default=["LeakyParallel"])
    args = ap.parse_args()

    runs = discover(protocols=args.protocols, neurons=args.neurons)
    if not runs:
        print("no runs with artifacts found yet — nothing to analyse.")
        return 0
    OUT.mkdir(parents=True, exist_ok=True)

    curves, summary = [], []
    for r in runs:
        c = curves_for(r)
        curves.append(c)
        f1 = c.f1_macro.to_numpy()
        t95, t99 = convergence_step(f1, 0.95), convergence_step(f1, 0.99)
        summary.append({
            "protocol": r.protocol, "neuron": r.neuron, "encoding": r.encoding,
            "seed": r.seed, "T": len(f1),
            "f1_terminal": f1[-1], "f1_peak": f1.max(),
            "t_peak": int(f1.argmax() + 1),
            "T95": t95, "T99": t99,
            "spikes_at_T99": (float(c.output_spikes_cum.iloc[t99 - 1])
                              if t99 else None),
            "spikes_terminal": float(c.output_spikes_cum.iloc[-1]),
        })
        print(f"  {r.protocol:24s} {r.variant:26s} s{r.seed}  "
              f"F1(T)={f1[-1]:.4f}  T95={t95}  T99={t99}", flush=True)

    curves_df = pd.concat(curves, ignore_index=True)
    summ = pd.DataFrame(summary)
    curves_df.to_csv(OUT / "curves.csv", index=False)
    summ.to_csv(OUT / "per_seed.csv", index=False)

    agg = (summ.groupby(["protocol", "neuron", "encoding"])
               .agg(n=("seed", "size"),
                    f1_terminal_mean=("f1_terminal", "mean"),
                    f1_terminal_sd=("f1_terminal", "std"),
                    T95_mean=("T95", "mean"), T95_sd=("T95", "std"),
                    T95_median=("T95", "median"),
                    T99_mean=("T99", "mean"), T99_sd=("T99", "std"),
                    T99_median=("T99", "median"),
                    # The across-seed distribution of T99 is skewed: most seeds
                    # converge in a few steps and an occasional one does not, so
                    # the mean sits well above the typical run. Both are
                    # reported; the median is the honest headline. The T95/T99
                    # definitions themselves are frozen (PREREGISTRATION.md 11)
                    # -- this is a choice of summary statistic, not of metric.
                    T99_min=("T99", "min"), T99_max=("T99", "max"),
                    T99_q25=("T99", lambda v: v.quantile(0.25)),
                    T99_q75=("T99", lambda v: v.quantile(0.75)),
                    T95_min=("T95", "min"), T95_max=("T95", "max"),
                    spikes_at_T99_mean=("spikes_at_T99", "mean"),
                    spikes_terminal_mean=("spikes_terminal", "mean"))
               .reset_index())
    agg["spike_saving_at_T99_pct"] = 100 * (
        1 - agg.spikes_at_T99_mean / agg.spikes_terminal_mean)
    # Early convergence is only a virtue if it converges to something good.
    # Delta reaches 99% of *its own* weaker terminal score almost immediately,
    # which must not be read as "delta is better at early inference". This
    # column states each encoding's terminal quality relative to the best
    # encoding on the same protocol, so the T99 column is never read alone.
    best = agg.groupby("protocol").f1_terminal_mean.transform("max")
    agg["f1_vs_best_pp"] = 100 * (agg.f1_terminal_mean - best)
    agg.to_csv(OUT / "summary.csv", index=False)

    for proto in sorted(curves_df.protocol.unique()):
        plot_curves(curves_df, proto, OUT / f"readout_{proto}")

    print("\n=== T95 / T99 by protocol and encoding (seed mean ± SD) ===")
    show = agg.copy()
    # Median with IQR and full range. A bare median would read as "latency
    # converges in 3 steps", which one seed (T99=23) contradicts; the timing
    # advantage is large but not uniform across initialisations, and the table
    # has to show that.
    show["T95"] = (show.T95_median.round(0).astype(int).astype(str)
                   + " [" + show.T95_min.round(0).astype(int).astype(str)
                   + "-" + show.T95_max.round(0).astype(int).astype(str) + "]")
    show["T99"] = (show.T99_median.round(0).astype(int).astype(str)
                   + " [IQR " + show.T99_q25.round(1).astype(str)
                   + "-" + show.T99_q75.round(1).astype(str)
                   + ", range " + show.T99_min.round(0).astype(int).astype(str)
                   + "-" + show.T99_max.round(0).astype(int).astype(str) + "]")
    show["F1(T)"] = (show.f1_terminal_mean.round(4).astype(str) + " ± "
                     + show.f1_terminal_sd.fillna(0).round(4).astype(str))
    print(show[["protocol", "neuron", "encoding", "n", "F1(T)", "f1_vs_best_pp",
                "T95", "T99", "spike_saving_at_T99_pct"]].to_string(index=False))
    print("\nf1_vs_best_pp is terminal macro-F1 minus the best encoding's on the "
          "same protocol.\nA small T99 beside a strongly negative "
          "f1_vs_best_pp is fast convergence to a worse answer, not an "
          "early-inference advantage.")
    print(f"\nwrote {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
