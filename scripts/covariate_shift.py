#!/usr/bin/env python3
"""E5: train-vs-test covariate shift, and how it relates to the value of timing.

POST-HOC AND EXPLORATORY. Not part of any registration.

The timing isolation (E1 arm C) established that spike timing is worth +54.5 pp
on one protocol and -9.0 pp on another. This asks what distinguishes those
conditions, with one hypothesis: timing encodes feature *magnitude* into spike
time, so a code that discards magnitude should suffer less when magnitudes move
between train and test. If so, the protocols where timing is a liability should
be the ones with the largest covariate shift.

Measured on the same train-fitted feature representation the network sees, so
the shift quantified here is the shift the SNN actually experiences.

Three measures, because each fails differently:
  KS statistic     distribution-free, per feature, insensitive to scale
  Wasserstein      magnitude-aware, interpretable because features are in [0,1]
  domain AUC       whole-vector separability; 0.5 = indistinguishable

plus threshold occupancy |P_train(x>t) - P_test(x>t)| at the delta gate, which
is the magnitude-discarded view of the same question: if continuous magnitudes
move while occupancy does not, that is the pattern the hypothesis predicts.

p-values are deliberately not reported. At these sample sizes every feature
separates significantly and the statistic is what carries information.

    python scripts/covariate_shift.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp, wasserstein_distance

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from figstyle import (COL_W, FS_ANNOT, LATENCY, MS_ORDINARY, RC,  # noqa: E402
                      TEXT, ZERO)

OUT = ROOT / "results/analysis/covariate_shift"
FIG = ROOT / "paper/figures"

PROTOCOLS = ["nslkdd_v2", "kddcup99_v2", "cicids2017_v2",
             "ctu13_v2_f0", "ctu13_v2_f1", "ctu13_v2_f2", "ctu13_v2_f3"]
DELTA_GATE = 0.05
#: Subsample cap per side. KS and Wasserstein on two million rows buy no
#: precision the conclusion depends on, and the cap keeps this CPU-only. Fixed
#: seed so the numbers are reproducible.
CAP, RNG_SEED = 50_000, 20260814


def matrices(protocol: str):
    """Train and test feature matrices exactly as the network receives them."""
    from src.data import get_dataset
    b = get_dataset(protocol, batch_size=16384, seed=42)

    def stack(loader):
        return np.concatenate([X.numpy() for X, _ in loader], axis=0)

    return stack(b.train_loader), stack(b.test_loader)


def shift_for(protocol: str, rng) -> dict:
    tr, te = matrices(protocol)
    d = tr.shape[1]
    a = tr[rng.choice(len(tr), min(CAP, len(tr)), replace=False)]
    b = te[rng.choice(len(te), min(CAP, len(te)), replace=False)]

    ks, wd, occ = [], [], []
    for j in range(d):
        x, y = a[:, j], b[:, j]
        ks.append(ks_2samp(x, y).statistic)
        wd.append(wasserstein_distance(x, y))
        occ.append(abs((x > DELTA_GATE).mean() - (y > DELTA_GATE).mean()))
    ks, wd, occ = np.array(ks), np.array(wd), np.array(occ)

    # Domain classifier: can a linear model tell train from test at all?
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import train_test_split
    X = np.vstack([a, b])
    y = np.r_[np.zeros(len(a)), np.ones(len(b))]
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=0,
                                          stratify=y)
    clf = LogisticRegression(max_iter=2000, n_jobs=-1).fit(Xtr, ytr)
    auc = roc_auc_score(yte, clf.predict_proba(Xte)[:, 1])

    per = pd.DataFrame({"protocol": protocol, "feature": np.arange(d),
                        "ks": ks, "wasserstein": wd, "occupancy_shift": occ})
    # Concentration. Two protocols can carry the same total shift with very
    # different structure: a few features moving a lot, or many moving a little.
    # A timing code maps magnitude to spike time per feature, so the two are not
    # equivalent for it, and the aggregate statistics above cannot tell them
    # apart.
    top5 = float(np.sort(wd)[::-1][:5].sum() / wd.sum()) if wd.sum() else 0.0
    n_big = int((wd > 0.1).sum())
    return {
        "protocol": protocol, "d": d,
        "ks_median": float(np.median(ks)), "ks_mean": float(ks.mean()),
        "ks_p90": float(np.percentile(ks, 90)), "ks_max": float(ks.max()),
        "w_median": float(np.median(wd)), "w_mean": float(wd.mean()),
        "w_p90": float(np.percentile(wd, 90)), "w_max": float(wd.max()),
        "occ_median": float(np.median(occ)), "occ_p90": float(np.percentile(occ, 90)),
        "domain_auc": float(auc),
        # If continuous magnitudes move more than threshold occupancy does, a
        # magnitude-discarding code sees less of the shift. Ratio > 1 is the
        # pattern the hypothesis predicts.
        "w_top5_share_pct": 100.0 * top5,
        "n_features_w_gt_0p1": n_big,
        "w_over_occ_p90": float(np.percentile(wd, 90) /
                                max(np.percentile(occ, 90), 1e-9)),
    }, per


def plot(summary: pd.DataFrame, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    matplotlib.rcParams.update(RC)
    s = summary.dropna(subset=["timing_premium_pp"])
    fig, ax = plt.subplots(figsize=(COL_W, 2.75), layout="constrained")
    ax.axhline(0, color=ZERO, lw=1.0, ls="--", zorder=1)
    # One colour: the points are protocols, not encodings, so the three-hue
    # mapping does not apply here.
    ax.scatter(s.w_p90, s.timing_premium_pp, s=MS_ORDINARY, color=LATENCY,
               zorder=3, edgecolor="white", linewidth=0.4)
    # NSL-KDD and KDDCup99 sit almost on top of each other: their shifts differ
    # by 0.008 and their timing premiums by 1.2 pp on an axis spanning 64. A
    # single offset for every point puts one label through the other, so the
    # two are pushed apart vertically and the rest keep the default.
    nudge = {"nslkdd": (4, 5), "kddcup99": (4, -9)}
    for _, r in s.iterrows():
        ax.annotate(r.label, (r.w_p90, r.timing_premium_pp),
                    fontsize=FS_ANNOT, color=TEXT,
                    xytext=nudge.get(r.label, (4, 3)),
                    textcoords="offset points")
    ax.set_xlabel("train-test shift: 90th pct. Wasserstein")
    ax.set_ylabel("value of timing (pp macro-F1)")
    # Deliberately no fitted line: five points cannot support one.
    for ext in ("pdf", "png"):
        fig.savefig(path.with_suffix("." + ext))
    plt.close(fig)


def main() -> int:
    rng = np.random.default_rng(RNG_SEED)
    OUT.mkdir(parents=True, exist_ok=True)
    rows, pers = [], []
    for proto in PROTOCOLS:
        try:
            s, per = shift_for(proto, rng)
        except Exception as e:                                   # noqa: BLE001
            print(f"  {proto}: skipped ({type(e).__name__}: {e})")
            continue
        rows.append(s); pers.append(per)
        print(f"  {proto:16s} KS med {s['ks_median']:.4f} p90 {s['ks_p90']:.4f} | "
              f"W med {s['w_median']:.4f} p90 {s['w_p90']:.4f} | "
              f"AUC {s['domain_auc']:.4f}", flush=True)

    summary = pd.DataFrame(rows)
    pd.concat(pers).to_csv(OUT / "per_feature.csv", index=False,
                           float_format="%.8f")

    # Join the timing premium where E1 measured it.
    ti = pd.read_csv(ROOT / "results/analysis/e1_timing_isolation.csv",
                     index_col=0)
    summary["timing_premium_pp"] = summary.protocol.map(ti.timing_premium_pp)
    summary["label"] = summary.protocol.str.replace("_v2", "", regex=False)
    summary.to_csv(OUT / "summary.csv", index=False, float_format="%.8f")
    plot(summary, FIG / "fig_shift_timing")

    print("\n=== shift summary, ordered by 90th-percentile Wasserstein ===")
    cols = ["protocol", "d", "ks_median", "ks_p90", "w_median", "w_p90",
            "occ_p90", "domain_auc", "timing_premium_pp"]
    print(summary.sort_values("w_p90")[cols].round(4).to_string(index=False))

    ctu = summary[summary.protocol.str.startswith("ctu13")]
    if len(ctu):
        worst = ctu.loc[ctu.w_p90.idxmax(), "protocol"]
        print(f"\nmost-shifted CTU-13 fold by 90th-pct Wasserstein: {worst}")
        print(f"most-shifted CTU-13 fold by domain AUC          : "
              f"{ctu.loc[ctu.domain_auc.idxmax(), 'protocol']}")
    have = summary.dropna(subset=["timing_premium_pp"])
    if len(have) >= 3:
        r = have[["w_p90", "timing_premium_pp"]].corr(method="spearman").iloc[0, 1]
        print(f"\nSpearman(shift, timing value) over {len(have)} protocols = "
              f"{r:+.3f}")
        print("Reported as a directional observation only: "
              f"{len(have)} points cannot support a fitted relationship.")
    print(f"\nwrote {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
