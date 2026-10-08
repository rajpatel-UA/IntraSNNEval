#!/usr/bin/env python3
"""Per-class breakdown, margin decomposition, and the well-supported subset.

No retraining: every number comes from the persisted per-sample predictions.

Three questions, in order of how much they matter:

  1. Where does the encoding margin live? macro-F1 is a flat mean over classes,
     so a 3.75 pp margin can be produced by a 67-sample class moving a few
     points. The class-wise decomposition
     ``d_macro = (1/K) sum_k (F1_k^lat - F1_k^rate)`` says which.
  2. Does the ordering survive dropping the classes with negligible support?
     If it does, the encoding conclusion is not an artefact of rare-class noise.
     If it does not, the advantage is concentrated in rare classes, which is a
     finding in its own right and connects to the unseen-attack section.
  3. Is any single class's difference significant on its own? Wilcoxon over the
     seed-blocks per class, Holm-corrected across classes.

The seed-variance arithmetic that motivates this: on NSL-KDD, U2R has 67 test
samples and an F1 standard deviation of about 0.13, so it alone contributes
0.13/5 = 2.6 pp of macro-F1 standard deviation against a reported total of
2.8 pp and a claimed margin of 3.75 pp.

    python scripts/per_class_analysis.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from sklearn.metrics import f1_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.readout import discover  # noqa: E402

OUT = ROOT / "results/analysis/per_class"
PROTOCOLS = ["nslkdd_v2", "kddcup99_v2", "cicids2017_v2",
             "ctu13_v2_f0", "ctu13_v2_f1"]
NEURON = "LeakyParallel"
#: A class is well supported when it holds at least this share of the test set.
#: Fixed before looking at any ordering, and reported alongside the counts so a
#: reader can see exactly which classes it removes.
MIN_SUPPORT_FRAC = 0.01


def collect() -> pd.DataFrame:
    rows = []
    for r in discover():
        if r.protocol not in PROTOCOLS or r.neuron != NEURON:
            continue
        rec, pre = r.record(), r.prefix()
        y, yh = pre["y_true"].astype(int), pre["y_pred"].astype(int)
        names = rec["data"].get("class_names") or []
        K = rec["data"]["num_classes"]
        per = f1_score(y, yh, labels=list(range(K)), average=None,
                       zero_division=0)
        for c in range(K):
            rows.append({"protocol": r.protocol, "encoding": r.encoding,
                         "seed": r.seed, "cls": c,
                         "name": names[c] if c < len(names) else str(c),
                         "f1": per[c], "support": int((y == c).sum()),
                         "n_test": len(y)})
    return pd.DataFrame(rows)


def main() -> int:
    df = collect()
    if df.empty:
        print("no runs found.")
        return 0
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT / "per_class_per_seed.csv", index=False, float_format="%.8f")

    cell = (df.groupby(["protocol", "encoding", "cls", "name", "support",
                        "n_test"])
              .agg(f1=("f1", "mean"), sd=("f1", lambda v: v.std(ddof=1)),
                   n=("seed", "nunique")).reset_index())
    cell["support_frac"] = cell.support / cell.n_test
    cell["well_supported"] = cell.support_frac >= MIN_SUPPORT_FRAC
    cell.to_csv(OUT / "per_class.csv", index=False, float_format="%.8f")

    print("=== 1. per-class F1 by encoding ===")
    for proto in PROTOCOLS:
        g = cell[cell.protocol == proto]
        if g.empty:
            continue
        piv = g.pivot_table(index=["cls", "name", "support", "well_supported"],
                            columns="encoding", values="f1")
        print(f"\n{proto}  (n_test={int(g.n_test.iloc[0]):,})")
        print(piv.round(4).to_string())

    print("\n\n=== 2. class-wise decomposition of the latency-rate margin ===")
    dec = []
    for proto in PROTOCOLS:
        g = cell[cell.protocol == proto]
        if g.empty or not {"latency", "rate"} <= set(g.encoding):
            continue
        p = g.pivot_table(index=["cls", "name", "support", "well_supported"],
                          columns="encoding", values="f1")
        K = len(p)
        contrib = (p["latency"] - p["rate"]) / K * 100.0
        tot = contrib.sum()
        print(f"\n{proto}: total margin {tot:+.3f} pp over K={K} classes")
        for (c, nm, sup, ws), v in contrib.items():
            share = 100 * v / tot if tot else float("nan")
            print(f"   {nm:>10s}  support {sup:>8,d}  contributes {v:+7.3f} pp"
                  f"  ({share:6.1f}% of margin){'' if ws else '   <- low support'}")
            dec.append({"protocol": proto, "cls": c, "name": nm,
                        "support": sup, "well_supported": ws,
                        "contribution_pp": v, "share_pct": share})
    pd.DataFrame(dec).to_csv(OUT / "margin_decomposition.csv", index=False,
                             float_format="%.8f")

    print("\n\n=== 3. macro-F1 over all classes vs well-supported only ===")
    sub = []
    for proto in PROTOCOLS:
        g = df[df.protocol == proto]
        if g.empty:
            continue
        ws = set(cell[(cell.protocol == proto) & cell.well_supported].cls)
        print(f"\n{proto}: well-supported classes {sorted(ws)} of "
              f"{g.cls.nunique()}")
        for enc in ("latency", "rate", "delta"):
            e = g[g.encoding == enc]
            if e.empty:
                continue
            full = e.groupby("seed").f1.mean()
            wsm = e[e.cls.isin(ws)].groupby("seed").f1.mean()
            print(f"   {enc:8s} all {full.mean():.4f} +/- {full.std(ddof=1):.4f}"
                  f"   well-supported {wsm.mean():.4f} +/- "
                  f"{wsm.std(ddof=1):.4f}")
            sub.append({"protocol": proto, "encoding": enc,
                        "macro_all": full.mean(), "sd_all": full.std(ddof=1),
                        "macro_ws": wsm.mean(), "sd_ws": wsm.std(ddof=1)})
    s = pd.DataFrame(sub)
    s.to_csv(OUT / "well_supported.csv", index=False, float_format="%.8f")

    print("\n   ordering check (latency vs rate):")
    for proto in PROTOCOLS:
        g = s[s.protocol == proto].set_index("encoding")
        if not {"latency", "rate"} <= set(g.index):
            continue
        a = 100 * (g.loc["latency", "macro_all"] - g.loc["rate", "macro_all"])
        b = 100 * (g.loc["latency", "macro_ws"] - g.loc["rate", "macro_ws"])
        print(f"   {proto:15s} all {a:+7.3f} pp   well-supported {b:+7.3f} pp"
              f"   {'PRESERVED' if np.sign(a) == np.sign(b) else 'INVERTS'}")

    print("\n\n=== 4. per-class paired Wilcoxon, Holm-corrected ===")
    for proto in PROTOCOLS:
        g = df[df.protocol == proto]
        if g.empty:
            continue
        res = []
        for c in sorted(g.cls.unique()):
            a = g[(g.encoding == "latency") & (g.cls == c)].sort_values("seed").f1
            b = g[(g.encoding == "rate") & (g.cls == c)].sort_values("seed").f1
            if len(a) != len(b) or len(a) < 2 or np.allclose(a.values, b.values):
                res.append((c, np.nan, np.nan))
                continue
            st, p = wilcoxon(a.values, b.values, zero_method="wilcox")
            res.append((c, 100 * (a.mean() - b.mean()), p))
        ps = [r[2] for r in res]
        order = np.argsort([1e9 if np.isnan(p) else p for p in ps])
        adj = [np.nan] * len(ps)
        run = 0.0
        for i, idx in enumerate(order):
            if np.isnan(ps[idx]):
                continue
            run = max(run, (len(ps) - i) * ps[idx])
            adj[idx] = min(1.0, run)
        nm = {int(r.cls): r["name"] for _, r in
              cell[cell.protocol == proto].drop_duplicates("cls").iterrows()}
        print(f"\n{proto}")
        for (c, d_pp, p), pa in zip(res, adj):
            print(f"   {nm.get(c, c):>10s}  delta {d_pp:+7.3f} pp   "
                  f"raw p {p if p == p else float('nan'):.4f}   Holm p "
                  f"{pa if pa == pa else float('nan'):.4f}")

    print(f"\nwrote {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
