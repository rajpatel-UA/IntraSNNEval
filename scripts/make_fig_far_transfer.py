#!/usr/bin/env python3
"""fig_far_transfer: what a validation-calibrated threshold does on test.

Each protocol contributes one arrow, from the argmax operating point to the
point reached by a threshold calibrated to a 1% false-alarm rate on validation.
The figure exists to show that the calibration does not transfer: the realised
test FAR exceeds its 1% target on every protocol, and on CTU-13 fold 1 it lands
near one half.

Read from the artifacts the tables are built from, not transcribed from the
typeset tables:

    argmax       results/analysis/strict_results.csv   (encoding axis)
    calibrated   results/analysis/security/fixed_far.csv, target_far = 0.01

Transcribing would put a fourth copy of these numbers in the repository, and a
figure that silently disagrees with its own table is worse than no figure.

    python scripts/make_fig_far_transfer.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from figstyle import ANNOT, COL_W, GRID, LATENCY, NEUTRAL_DK, RC, TEXT  # noqa: E402

A = ROOT / "results/analysis"
FIG = ROOT / "paper/figures"
SEL_NEURON, SEL_ENC = "LeakyParallel", "latency"
TARGET_FAR = 0.01

LABEL = {"nslkdd_v2": "NSL-KDD", "kddcup99_v2": "KDDCup99",
         "cicids2017_v2": "CIC-IDS2017", "ctu13_v2_f0": "CTU-13 f0",
         "ctu13_v2_f1": "CTU-13 f1"}
#: Label placement as (dx, dy) in POINTS from the argmax marker, plus its
#: horizontal alignment. Offsets in data units do not work here: the x axis is
#: logarithmic over three decades and the y range packs four protocols into its
#: top tenth, so a single multiplier lands a label on a neighbour at one end and
#: off the axis at the other.
NUDGE = {"cicids2017_v2": (+4, -11, "left"),
         "ctu13_v2_f0":   (+4, +6,  "left"),
         "kddcup99_v2":   (-6, -11, "right"),
         "ctu13_v2_f1":   (-7, +5,  "right"),
         "nslkdd_v2":     (-7, +4,  "right")}


def load() -> pd.DataFrame:
    s = pd.read_csv(A / "strict_results.csv")
    s = s[(s.variant == f"{SEL_NEURON}/{SEL_ENC}") & (s.axis == "encoding")]
    s = s.set_index("protocol")[["dr_mean", "far_mean"]]

    f = pd.read_csv(A / "security/fixed_far.csv")
    f = f[(f.neuron == SEL_NEURON) & (f.encoding == SEL_ENC)
          & (f.target_far == TARGET_FAR)]
    f = f.set_index("protocol")[["test_dr_mean", "test_far_mean",
                                 "val_far_mean"]]

    d = s.join(f, how="inner")
    if d.empty:
        raise SystemExit(
            "no protocol has both an argmax row and a calibrated row; check "
            "that security_analysis.py has been run at the 1% target.")
    return d


def main() -> int:
    d = load()
    matplotlib.rcParams.update(RC)
    fig, ax = plt.subplots(figsize=(COL_W, 2.6), layout="constrained")

    for proto, r in d.iterrows():
        ax.add_patch(FancyArrowPatch(
            (r.far_mean, r.dr_mean), (r.test_far_mean, r.test_dr_mean),
            arrowstyle="-|>", mutation_scale=8, color=NEUTRAL_DK, lw=0.9,
            shrinkA=2.5, shrinkB=2.5, zorder=2))
        ax.plot(r.far_mean, r.dr_mean, marker="o", ms=4, mfc="white",
                mec=LATENCY, mew=1.1, ls="none", zorder=3)
        ax.plot(r.test_far_mean, r.test_dr_mean, marker="o", ms=4,
                color=LATENCY, ls="none", zorder=3)
        dx, dy, ha = NUDGE.get(proto, (-5, -11, "right"))
        ax.annotate(LABEL.get(proto, proto), (r.far_mean, r.dr_mean),
                    xytext=(dx, dy), textcoords="offset points",
                    fontsize=6.5, color=ANNOT, ha=ha,
                    va="bottom" if dy > 0 else "top")

    # The target the calibration was set to. Every arrow ends to its right,
    # which is the finding.
    ax.axvline(TARGET_FAR, color=TEXT, lw=0.8, ls="--", zorder=1)
    # Along the bottom, the only band with no markers in it: the top-left is
    # taken by the two CTU-13 labels and the mid-left by the legend.
    ax.annotate(f"{TARGET_FAR:.0%} validation target", (TARGET_FAR, 0.638),
                fontsize=6.2, color=TEXT, va="bottom", ha="right",
                xytext=(-3, 0), textcoords="offset points")

    ax.legend(handles=[
        Line2D([], [], marker="o", ms=4, mfc="white", mec=LATENCY, mew=1.1,
               ls="none", label="argmax"),
        Line2D([], [], marker="o", ms=4, color=LATENCY, ls="none",
               label=f"calibrated to {TARGET_FAR:.0%} val FAR")],
        # Parked in the empty mid-left band: every marker is either above 0.90
        # or at NSL-KDD's 0.67, so nothing lives between them.
        fontsize=6.5, loc="center left", bbox_to_anchor=(0.01, 0.42),
        handletextpad=0.4, borderpad=0.2)

    ax.set_xscale("log")
    ax.set_xlim(2e-4, 1.4)
    ax.set_ylim(0.63, 1.035)
    ax.set_xlabel("test false-alarm rate (log)")
    ax.set_ylabel("detection rate")
    ax.grid(True, which="major", color=GRID, lw=0.5, zorder=0)
    FIG.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(FIG / f"fig_far_transfer.{ext}")
    plt.close(fig)

    w, h = COL_W, 2.6
    print(f"  fig_far_transfer  {w:.3f} x {h:.3f} in")
    print("\nrealised test FAR against a "
          f"{TARGET_FAR:.0%} validation target:")
    for proto, r in d.iterrows():
        print(f"  {LABEL.get(proto, proto):14s} val {r.val_far_mean:.4f} -> "
              f"test {r.test_far_mean:.4f}  "
              f"({r.test_far_mean / TARGET_FAR:5.1f}x target)   "
              f"DR {r.dr_mean:.4f} -> {r.test_dr_mean:.4f}")
    over = int((d.test_far_mean > TARGET_FAR).sum())
    print(f"\ntest FAR exceeds the target on {over} of {len(d)} protocols")
    return 0


if __name__ == "__main__":
    sys.exit(main())
