#!/usr/bin/env python3
"""Test the Results section's load-bearing claim about latency's exceptions.

The encoding subsection makes a strong claim, and it is strong precisely because
it is falsifiable by a single block:

    Every block latency does not win is a block in which timing was
    independently measured to be worth nothing or worth less than nothing.

That sentence is what makes the exceptions *predicted* rather than merely
tolerated, and it is the difference between an argument that survives a
reviewer and one that does not. It was true of the three exception blocks on
the four protocols available when it was drafted. CIC-IDS2017 was not among
them, and CIC has the second-largest timing premium in the study, so it should
produce no new exceptions. Should is not a guarantee.

If CIC does produce a latency loss, the sentence must be weakened from a
universal to a tendency, and the time to discover that is before it ships. So
the claim is checked mechanically against the data rather than re-verified by
eye, and the check names every block that violates it.

The rule, stated so the audit can apply it:

    a block is an ALLOWED exception if its protocol's timing premium is
    <= TIMING_TOLERANCE_PP, i.e. timing was measured to be worth nothing or
    worth less than nothing on that protocol
    a block is a VIOLATION if latency loses it on a protocol where timing was
    measured to be clearly valuable

Exits non-zero on any violation, so it can gate a build.

    python scripts/exception_audit.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.populations import macro_f1_pair, require_population  # noqa: E402
from src.readout import discover  # noqa: E402

CONF_PROTOCOLS = ["nslkdd_v2", "kddcup99_v2", "cicids2017_v2",
                  "ctu13_v2_f0", "ctu13_v2_f1"]

#: A protocol counts as one where timing buys nothing when its measured premium
#: is at or below this. NSL-KDD sits at +1.2 pp, which is inside it; CIC sits at
#: +30.0 pp and CTU-13 fold 0 at +54.5 pp, which are far outside. The threshold
#: is not tuned: it is set at the largest premium the manuscript already
#: describes as timing buying nothing, and moving it to admit a new exception
#: after seeing one would defeat the purpose of the audit.
TIMING_TOLERANCE_PP = 2.0

OUT = ROOT / "results/analysis/exception_audit"


def blocks() -> pd.DataFrame:
    rows = []
    for r in discover():
        if r.protocol not in CONF_PROTOCOLS:
            continue
        rec, pre = r.record(), r.prefix()
        m = macro_f1_pair(pre["y_true"].astype(np.int64),
                          pre["y_pred"].astype(np.int64),
                          rec["data"]["num_classes"])
        rows.append({"protocol": r.protocol, "variant": r.variant,
                     "seed": r.seed, "f1": m["macro_f1_fixed_universe"]})
    return pd.DataFrame(rows)


def main() -> int:
    df = blocks()
    if df.empty:
        print("no confirmation runs on disk yet; nothing to audit.")
        return 0

    complete = [p for p, g in df.groupby("protocol")
                if g.variant.nunique() == 27
                and g.groupby("variant").seed.nunique().min() == 5]
    if not complete:
        print("no protocol is complete at 27 configurations x 5 seeds; "
              "nothing to audit.")
        return 0

    sub = df[df.protocol.isin(complete)]
    require_population(sub.variant.unique(), "full_27_confirmation")
    wide = sub.pivot_table(index=["protocol", "seed"], columns="variant",
                           values="f1").dropna()
    ranks = pd.DataFrame(
        np.apply_along_axis(lambda v: rankdata(-v, method="average"), 1,
                            wide.to_numpy()),
        index=wide.index, columns=wide.columns)

    ti = pd.read_csv(ROOT / "results/analysis/e1_timing_isolation.csv",
                     index_col=0)

    rows = []
    for idx in ranks.index:
        winner = ranks.loc[idx].idxmin()
        proto, seed = idx
        prem = ti.timing_premium_pp.get(proto, float("nan"))
        enc = winner.split("/")[1]
        rows.append({
            "protocol": proto, "seed": seed, "winning_variant": winner,
            "winning_encoding": enc, "timing_premium_pp": prem,
            "latency_wins": enc == "latency",
            # An exception is allowed exactly when the independent measurement
            # said timing was worthless there.
            "status": ("latency wins" if enc == "latency"
                       else "ALLOWED exception" if prem <= TIMING_TOLERANCE_PP
                       else "VIOLATION"),
        })
    au = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    au.to_csv(OUT / "blocks.csv", index=False, float_format="%.6f")

    n = len(au)
    wins = int(au.latency_wins.sum())
    allowed = au[au.status == "ALLOWED exception"]
    viol = au[au.status == "VIOLATION"]

    print(f"protocols complete: {len(complete)} of {len(CONF_PROTOCOLS)} "
          f"({', '.join(sorted(complete))})")
    print(f"blocks: {n}\n")
    print(f"latency holds rank 1 in {wins} of {n} blocks")
    print(f"allowed exceptions   : {len(allowed)}")
    print(f"VIOLATIONS           : {len(viol)}\n")

    if len(allowed):
        print("allowed exceptions (timing independently measured as worthless):")
        print(allowed[["protocol", "seed", "winning_variant",
                       "timing_premium_pp"]].to_string(index=False))
    if len(viol):
        print("\nVIOLATIONS (latency loses where timing was measured to be "
              "valuable):")
        print(viol[["protocol", "seed", "winning_variant",
                    "timing_premium_pp"]].to_string(index=False))
        print("\nThe Results claim that every block latency does not win is a "
              "block\nwhere timing was measured to be worth nothing is now "
              "FALSE. Weaken it\nfrom a universal to a tendency before the "
              "manuscript ships, and report\nthe blocks above.")
        return 1

    if len(complete) < len(CONF_PROTOCOLS):
        print(f"\nHolds on {len(complete)} protocols. Re-run when the "
              "remaining protocols land:\nthe claim is universal over blocks "
              "and a single new block can falsify it.")
    else:
        print("\nThe claim holds on all five confirmation protocols: every "
              "block latency\ndoes not win is a block where matched-input "
              "isolation independently\nmeasured timing at or below "
              f"{TIMING_TOLERANCE_PP:+.1f} pp.")
    print(f"\nwrote {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
