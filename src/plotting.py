"""IEEE-style colorblind-safe plotting helpers for the Phase-1 SNN sweep.

The package conventions widely accepted at IEEE/ACM venues:
  * Wong (2011) 8-colour categorical palette — colorblind-safe, distinguishable
    in greyscale, used by Nature / Cell / IEEE accessibility guides.
      https://www.nature.com/articles/nmeth.1618
  * Viridis / cividis for sequential / heatmap data — perceptually uniform,
    colorblind-safe, prints well in greyscale.
  * Single-column figure width ≈ 3.5 in, double-column ≈ 7.16 in
    (IEEE Transactions / Conference templates).

Outputs are saved as **PDF** (vector, paper-ready) plus PNG (preview).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ------------------------------------------------------------------
# Style
# ------------------------------------------------------------------
WONG_PALETTE = [
    "#000000",  # black
    "#E69F00",  # orange
    "#56B4E9",  # sky blue
    "#009E73",  # bluish green
    "#F0E442",  # yellow
    "#0072B2",  # blue
    "#D55E00",  # vermillion
    "#CC79A7",  # reddish purple
]

IEEE_RC = {
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "font.family": "serif",
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linestyle": ":",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "image.cmap": "viridis",
}


def apply_ieee_style():
    mpl.rcParams.update(IEEE_RC)
    mpl.rcParams["axes.prop_cycle"] = mpl.cycler(color=WONG_PALETTE)


def save_figure(fig, out_path: Path):
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path.with_suffix(".pdf"))
    fig.savefig(out_path.with_suffix(".png"))
    plt.close(fig)


# ------------------------------------------------------------------
# Data loading
# ------------------------------------------------------------------
def load_summary(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    return df


def load_all_results(results_root: Path, dataset: str) -> List[Dict]:
    """Walk results/<dataset>/SNN_*/<encoding>/seed_*/results.json and load each.

    Backwards-compatible with the older layout (no seed_*/ subdir) — loads
    those too if present.
    """
    out = []
    for pattern in ("SNN_*/*/seed_*/results.json", "SNN_*/*/results.json"):
        for p in (results_root / dataset).glob(pattern):
            try:
                with open(p) as f:
                    out.append(json.load(f))
            except Exception as e:
                print(f"  warn: could not parse {p}: {e}")
    return out


# ------------------------------------------------------------------
# Plot 1 — Heatmap of a headline metric across (neuron, encoding)
# ------------------------------------------------------------------
def plot_metric_heatmap(df: pd.DataFrame, metric: str, out_path: Path,
                        title: Optional[str] = None,
                        cmap: str = "viridis",
                        fmt: str = "{:.3f}"):
    """One cell per (neuron, encoding) — colour = metric value, annotation = value."""
    pivot = df.pivot_table(index="neuron", columns="encoding", values=metric,
                           aggfunc="mean")
    pivot = pivot.reindex(columns=["rate", "latency", "delta"])
    pivot = pivot.sort_index()

    fig, ax = plt.subplots(figsize=(3.5, 3.4))
    im = ax.imshow(pivot.values, aspect="auto", cmap=cmap)
    ax.set_xticks(range(pivot.shape[1])); ax.set_xticklabels(pivot.columns)
    ax.set_yticks(range(pivot.shape[0])); ax.set_yticklabels(pivot.index)
    ax.set_xlabel("Encoding"); ax.set_ylabel("Neuron")
    ax.set_title(title or metric)
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            v = pivot.values[i, j]
            if not np.isnan(v):
                ax.text(j, i, fmt.format(v), ha="center", va="center",
                        color="white" if v < (pivot.values[~np.isnan(pivot.values)].mean()) else "black",
                        fontsize=7)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.grid(False)
    save_figure(fig, out_path)


# ------------------------------------------------------------------
# Plot 2 — Per-class F1 bars, grouped by encoding, faceted by neuron
# ------------------------------------------------------------------
def plot_per_class_f1(records: List[Dict], out_path: Path):
    rows = []
    for r in records:
        n = r["neuron"]; e = r["encoding"]
        per = r["test_plus"]["metrics"].get("per_class", {})
        for cls, vals in per.items():
            rows.append({"neuron": n, "encoding": e, "class": cls, "f1": vals["f1"]})
    if not rows:
        return
    df = pd.DataFrame(rows)
    # Multi-seed safety: average f1 across seeds for each (neuron, encoding, class)
    # so .loc[class, 'f1'] returns a scalar, not a Series of per-seed values.
    df = df.groupby(["neuron", "encoding", "class"], as_index=False)["f1"].mean()

    # Class list comes from the actual records so binary (normal/attack)
    # datasets like ISCXIDS2012 work alongside 5-class ones.
    classes = list(records[0]["test_plus"]["metrics"].get("class_names",
                                                            sorted(df["class"].unique())))
    encodings = ["rate", "latency", "delta"]
    neurons = sorted(df["neuron"].unique())

    fig, axes = plt.subplots(3, 3, figsize=(7.2, 6), sharey=True, sharex=True)
    axes = np.atleast_2d(axes)
    color_for_enc = {e: c for e, c in zip(encodings, WONG_PALETTE[1:4])}

    for k, n in enumerate(neurons[:9]):  # 9 neurons → 3x3 grid
        ax = axes.flat[k]
        sub = df[df["neuron"] == n]
        x = np.arange(len(classes))
        width = 0.27
        for i, enc in enumerate(encodings):
            ssub = sub[sub["encoding"] == enc].set_index("class")
            vals = [float(ssub.loc[c, "f1"]) if c in ssub.index else 0.0
                    for c in classes]
            ax.bar(x + (i - 1) * width, vals, width, label=enc,
                   color=color_for_enc[enc])
        ax.set_title(n)
        ax.set_xticks(x); ax.set_xticklabels(classes, rotation=30, ha="right")
        ax.set_ylim(0, 1.0)
    # Hide unused subplots
    for k in range(len(neurons), 9):
        axes.flat[k].axis("off")
    axes.flat[0].legend(loc="upper right", frameon=False, fontsize=7)
    fig.suptitle("Per-class F1 on KDDTest+ (5-class, by neuron × encoding)", y=1.02)
    fig.tight_layout()
    save_figure(fig, out_path)


# ------------------------------------------------------------------
# Plot 3 — Detection Rate vs False Alarm Rate scatter (the IDS ROC view)
# ------------------------------------------------------------------
def plot_dr_vs_far(df: pd.DataFrame, out_path: Path):
    fig, ax = plt.subplots(figsize=(4.0, 3.6))
    encodings = ["rate", "latency", "delta"]
    color_for_enc = {e: c for e, c in zip(encodings, WONG_PALETTE[1:4])}
    markers = {"rate": "o", "latency": "s", "delta": "^"}
    for enc in encodings:
        sub = df[df["encoding"] == enc]
        if sub.empty:
            continue
        ax.scatter(sub["false_alarm_rate"], sub["detection_rate"],
                   c=color_for_enc[enc], marker=markers[enc],
                   edgecolor="black", linewidth=0.4, s=55, label=enc)
        for _, row in sub.iterrows():
            ax.annotate(row["neuron"], (row["false_alarm_rate"], row["detection_rate"]),
                        fontsize=6, alpha=0.7, xytext=(3, 3),
                        textcoords="offset points")
    ax.set_xlabel("False Alarm Rate  (FP / (FP+TN))")
    ax.set_ylabel("Detection Rate  (TP / (TP+FN))")
    ax.set_title("Detection vs False-Alarm  —  KDDTest+ (lower-left FAR & upper DR is better)")
    ax.set_xlim(left=-0.005)
    ax.set_ylim(0, 1.05)
    ax.legend(frameon=False)
    save_figure(fig, out_path)


# ------------------------------------------------------------------
# Plot 4 — Accuracy vs Energy trade-off (Pareto plot)
# ------------------------------------------------------------------
def plot_accuracy_vs_energy(df: pd.DataFrame, out_path: Path):
    fig, ax = plt.subplots(figsize=(4.0, 3.6))
    encodings = ["rate", "latency", "delta"]
    color_for_enc = {e: c for e, c in zip(encodings, WONG_PALETTE[1:4])}
    markers = {"rate": "o", "latency": "s", "delta": "^"}
    for enc in encodings:
        sub = df[df["encoding"] == enc]
        if sub.empty:
            continue
        ax.scatter(sub["energy_spikes_per_sample"], sub["accuracy"],
                   c=color_for_enc[enc], marker=markers[enc],
                   edgecolor="black", linewidth=0.4, s=55, label=enc)
        for _, row in sub.iterrows():
            ax.annotate(row["neuron"], (row["energy_spikes_per_sample"], row["accuracy"]),
                        fontsize=6, alpha=0.7, xytext=(3, 3),
                        textcoords="offset points")
    ax.set_xscale("log")
    ax.set_xlabel("Energy proxy — spikes / sample (log scale)")
    ax.set_ylabel("Accuracy (KDDTest+)")
    ax.set_title("Accuracy vs Energy  —  upper-left is the SNN sweet spot")
    ax.legend(frameon=False)
    save_figure(fig, out_path)


# ------------------------------------------------------------------
# Plot 5 — Per-epoch training curves grid
# ------------------------------------------------------------------
def plot_training_curves(records: List[Dict], out_path: Path):
    if not records:
        return
    encodings = ["rate", "latency", "delta"]
    color_for_enc = {e: c for e, c in zip(encodings, WONG_PALETTE[1:4])}
    neurons = sorted({r["neuron"] for r in records})

    fig, axes = plt.subplots(3, 3, figsize=(7.2, 6), sharey=True, sharex=True)
    axes = np.atleast_2d(axes)
    for k, n in enumerate(neurons[:9]):
        ax = axes.flat[k]
        for r in records:
            if r["neuron"] != n:
                continue
            curve = r.get("training", {}).get("training_curve", [])
            if not curve:
                continue
            xs = [c["epoch"] for c in curve]
            ys = [c["val_loss"] for c in curve]
            ax.plot(xs, ys, marker="o", markersize=3,
                    color=color_for_enc[r["encoding"]], label=r["encoding"], linewidth=1)
        ax.set_title(n)
        ax.set_xlabel("Epoch"); ax.set_ylabel("Val loss")
    for k in range(len(neurons), 9):
        axes.flat[k].axis("off")
    axes.flat[0].legend(loc="upper right", frameon=False, fontsize=7)
    fig.suptitle("Per-epoch validation loss (neuron × encoding)", y=1.02)
    fig.tight_layout()
    save_figure(fig, out_path)


# ------------------------------------------------------------------
# Master entry
# ------------------------------------------------------------------
def generate_all(dataset: str, repo_root: Path, output_dir: Path):
    apply_ieee_style()
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_csv = repo_root / "results" / "summary.csv"
    if not summary_csv.exists():
        print(f"  warn: no {summary_csv} yet — run scripts/run_sweep.py first.")
        return

    df = load_summary(summary_csv)
    df = df[df["dataset"] == dataset].copy()
    if df.empty:
        print(f"  warn: no {dataset} rows in summary.csv")
        return
    records = load_all_results(repo_root / "results", dataset)

    print(f"  loaded {len(df)} summary rows + {len(records)} results.json files for {dataset}")

    # If multiple seeds per (neuron, encoding) exist, the heatmaps below
    # already mean-aggregate via pivot_table(aggfunc="mean"). Note the seed
    # count in the printout so the consumer knows.
    seeds_seen = sorted(df["seed"].unique().tolist()) if "seed" in df.columns else []
    if len(seeds_seen) > 1:
        print(f"  multi-seed mode: aggregating over seeds {seeds_seen} (mean)")

    # Headline heatmaps
    for metric, fname in [
        ("accuracy", "01_accuracy_heatmap"),
        ("f1_macro", "02_f1_macro_heatmap"),
        ("matthews_corrcoef", "03_mcc_heatmap"),
        ("detection_rate", "04_detection_rate_heatmap"),
        ("false_alarm_rate", "05_far_heatmap"),
        ("spike_sparsity", "06_spike_sparsity_heatmap"),
    ]:
        plot_metric_heatmap(df, metric, output_dir / fname, title=metric.replace("_", " "))

    plot_per_class_f1(records, output_dir / "07_per_class_f1")
    plot_dr_vs_far(df, output_dir / "08_dr_vs_far")
    plot_accuracy_vs_energy(df, output_dir / "09_accuracy_vs_energy")
    plot_training_curves(records, output_dir / "10_training_curves")

    # Ranked table — if multiple seeds, aggregate mean + std per combo
    if len(seeds_seen) > 1:
        agg = df.groupby(["neuron", "encoding"]).agg(
            accuracy_mean=("accuracy", "mean"), accuracy_std=("accuracy", "std"),
            f1_macro_mean=("f1_macro", "mean"), f1_macro_std=("f1_macro", "std"),
            mcc_mean=("matthews_corrcoef", "mean"), mcc_std=("matthews_corrcoef", "std"),
            detection_rate_mean=("detection_rate", "mean"),
            false_alarm_rate_mean=("false_alarm_rate", "mean"),
            energy_mean=("energy_spikes_per_sample", "mean"),
            sparsity_mean=("spike_sparsity", "mean"),
            n_seeds=("seed", "nunique"),
        ).reset_index().sort_values("f1_macro_mean", ascending=False)
        agg.to_csv(output_dir / "ranked_by_f1_macro_multiseed.csv", index=False)
        out_name = "ranked_by_f1_macro_multiseed.csv"
    else:
        rank = df.sort_values("f1_macro", ascending=False)[
            ["neuron", "encoding", "seed", "accuracy", "f1_macro",
             "matthews_corrcoef", "detection_rate", "false_alarm_rate",
             "energy_spikes_per_sample", "spike_sparsity",
             "trainable_parameters", "train_time_s"]
        ]
        rank.to_csv(output_dir / "ranked_by_f1_macro.csv", index=False)
        out_name = "ranked_by_f1_macro.csv"
    print(f"  wrote 10 figures + {out_name} → {output_dir}")
