#!/usr/bin/env python3
"""Per-protocol strict results: the detail table the confirmation was missing.

The screening sweep had a full metric table while the strict confirmation ---
the more credible half of the study --- had only mean ranks. This closes that
gap. For every (strict protocol, configuration) it reports macro-F1, MCC,
detection rate, false-alarm rate and SOPs per sample, each as a mean over the
five seeds with the across-seed standard deviation. Dispersion is always within
a protocol; nothing here is pooled across protocols.

Both confirmation axes appear, tagged with the population they belong to, so a
reader can see that the encoding rows vary only the encoding and the neuron rows
only the neuron. They are not a joint ranking.

macro-F1 is the pre-registered fixed label universe, recomputed from the saved
per-sample predictions (see the analysis amendment: the stored field used a
model-dependent denominator on the protocol with a zero-support class).

    python scripts/strict_results.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.populations import macro_f1_pair, members  # noqa: E402
from src.readout import discover  # noqa: E402

OUT_CSV = ROOT / "results/analysis/strict_results.csv"
OUT_TEX = ROOT / "paper/tables/tab_strict_results.tex"

PROTO = {"nslkdd_v2": "NSL-KDD", "kddcup99_v2": "KDDCup99",
         "cicids2017_v2": "CIC-IDS2017", "ctu13_v2_f0": "CTU-13 f0",
         "ctu13_v2_f1": "CTU-13 f1"}
ORDER = ["nslkdd_v2", "kddcup99_v2", "cicids2017_v2", "ctu13_v2_f0",
         "ctu13_v2_f1"]
ENC_ORDER = ["latency", "rate", "delta"]

#: Structural expectations. The encoding axis runs on all five strict
#: protocols; the neuron axis was not run on CIC-IDS2017 (Tier C excluded it on
#: cost grounds), so it covers four. These are asserted rather than trusted:
#: an earlier version of the protocol-level script used exclusive if/elif
#: membership and silently produced an EIGHT-neuron ranking, because
#: LeakyParallel/latency is the intersection of the two axes and was consumed by
#: whichever branch tested first. A count assertion catches that class of bug at
#: the point it would otherwise become a table in the paper.
EXPECT = {
    "encoding": {"n_configs": 3,
                 "protocols": ["nslkdd_v2", "kddcup99_v2", "cicids2017_v2",
                               "ctu13_v2_f0", "ctu13_v2_f1"]},
    "neuron": {"n_configs": 9,
               "protocols": ["nslkdd_v2", "kddcup99_v2", "ctu13_v2_f0",
                             "ctu13_v2_f1"]},
}
N_SEEDS = 5


def restrict_to_axis_protocols(df: pd.DataFrame) -> pd.DataFrame:
    """Drop protocols an axis cannot support, and say so.

    CIC-IDS2017 carries LeakyParallel/latency (from the encoding-axis run) but
    none of the other eight neurons, because Tier C excluded it on cost. A
    one-neuron row in a nine-neuron comparison is not a comparison, so it is
    removed from the neuron block -- deliberately and out loud, rather than
    surviving as a lone bar that looks like a result.
    """
    keep = []
    for axis, exp in EXPECT.items():
        sub = df[df.axis == axis]
        dropped = sorted(set(sub.protocol.unique()) - set(exp["protocols"]))
        if dropped:
            print(f"  note: {axis} axis excludes {dropped} "
                  f"(partial coverage: that axis was not run there)")
        keep.append(sub[sub.protocol.isin(exp["protocols"])])
    return pd.concat(keep, ignore_index=True)


def assert_structure(df: pd.DataFrame) -> None:
    """Fail loudly if either axis is missing a configuration or a protocol."""
    problems = []
    for axis, exp in EXPECT.items():
        sub = df[df.axis == axis]
        missing = sorted(set(exp["protocols"]) - set(sub.protocol.unique()))
        if missing:
            problems.append(f"{axis} axis missing protocols {missing}")
        for proto in exp["protocols"]:
            block = sub[sub.protocol == proto]
            n_cfg = block.variant.nunique()
            if n_cfg != exp["n_configs"]:
                problems.append(
                    f"{axis} axis / {proto}: {n_cfg} configurations, expected "
                    f"{exp['n_configs']} ({sorted(block.variant.unique())})")
            bad_seeds = (block.groupby("variant").seed.nunique()
                         .loc[lambda x: x != N_SEEDS])
            if len(bad_seeds):
                problems.append(f"{axis} axis / {proto}: seed count != "
                                f"{N_SEEDS} for {bad_seeds.to_dict()}")
    if problems:
        raise SystemExit("STRUCTURAL CHECK FAILED:\n  " + "\n  ".join(problems))
    print("structure OK: encoding axis "
          f"{EXPECT['encoding']['n_configs']} configs x "
          f"{len(EXPECT['encoding']['protocols'])} protocols; neuron axis "
          f"{EXPECT['neuron']['n_configs']} x "
          f"{len(EXPECT['neuron']['protocols'])}; {N_SEEDS} seeds throughout")


def collect() -> pd.DataFrame:
    enc_pop, neu_pop = members("encoding_axis_lp"), members("neuron_axis_latency")
    rows = []
    for r in discover():
        if r.protocol not in PROTO:
            continue
        # Inclusive, not exclusive: LeakyParallel/latency is the intersection
        # of the two axes and must appear in both blocks.
        axes = [a for a, pop in (("encoding", enc_pop), ("neuron", neu_pop))
                if r.variant in pop]
        if not axes:
            continue
        rec = r.record()
        tp = rec["test_plus"]
        pre = r.prefix()
        dual = macro_f1_pair(pre["y_true"].astype(np.int64),
                             pre["y_pred"].astype(np.int64),
                             rec["data"]["num_classes"])
        for axis in axes:
            rows.append({
                "protocol": r.protocol, "axis": axis, "variant": r.variant,
                "neuron": r.neuron, "encoding": r.encoding, "seed": r.seed,
                "macro_f1": dual["macro_f1_fixed_universe"],
                "macro_f1_present": dual["macro_f1_test_present"],
                "mcc": tp["metrics"]["matthews_corrcoef"],
                "dr": tp["binary_ids_metrics"]["detection_rate"],
                "far": tp["binary_ids_metrics"]["false_alarm_rate"],
                "sops": tp["spike_accounting"]["sops_per_sample"],
            })
    return pd.DataFrame(rows)


def agg(df: pd.DataFrame) -> pd.DataFrame:
    g = (df.groupby(["axis", "protocol", "variant"])
           .agg(n=("seed", "size"),
                **{f"{m}_{s}": (m, s) for m in
                   ("macro_f1", "mcc", "dr", "far", "sops")
                   for s in ("mean", "std")})
           .reset_index())
    g["proto_order"] = g.protocol.map({p: i for i, p in enumerate(ORDER)})
    return g.sort_values(["axis", "proto_order", "macro_f1_mean"],
                         ascending=[True, True, False])


def cell(m, s, dp=4):
    return f"{m:.{dp}f} $\\pm$ {0.0 if pd.isna(s) else s:.{dp}f}"


def to_tex(a: pd.DataFrame) -> str:
    L = [r"\begin{table*}[t]", r"\centering",
         r"\caption{Per-protocol results under the \textbf{strict} (v2) "
         r"protocol, five seeds, mean $\pm$ across-seed standard deviation. "
         r"The upper block is the encoding axis (neuron held at "
         r"\texttt{LeakyParallel}); the lower block is the neuron axis "
         r"(encoding held at latency). Each block varies one factor, so the "
         r"table is not a joint ranking of both. Macro-F1 uses the "
         r"pre-registered fixed label universe and is therefore capped at "
         r"$0.80$ on CIC-IDS2017, where U2R has no test support; its absolute "
         r"value is not comparable across protocols, though every comparison "
         r"within a protocol is unaffected. DR and FAR are the "
         r"normal-versus-attack view. SOPs are synaptic operations per sample, "
         r"an operation count rather than measured energy.}",
         r"\label{tab:strict-results}",
         r"\resizebox{\textwidth}{!}{%",
         r"\begin{tabular}{l l c c c c r}", r"\toprule",
         r"Protocol & Configuration & Macro-F1 & MCC & DR & FAR & SOPs/sample \\"]

    for axis, title in (("encoding", "Encoding axis --- neuron fixed at "
                                     r"\texttt{LeakyParallel}"),
                        ("neuron", "Neuron axis --- encoding fixed at latency")):
        sub = a[a.axis == axis]
        if sub.empty:
            continue
        L += [r"\midrule",
              r"\multicolumn{7}{l}{\textit{" + title + r"}} \\", r"\midrule"]
        for proto in ORDER:
            block = sub[sub.protocol == proto]
            if block.empty:
                continue
            if axis == "encoding":
                block = block.set_index("variant").reindex(
                    [f"LeakyParallel/{e}" for e in ENC_ORDER]).dropna(
                    subset=["macro_f1_mean"]).reset_index()
            best = block.macro_f1_mean.max()
            for i, r in block.iterrows():
                name = r["variant"].replace("_", r"\_")
                if axis == "neuron":
                    name = name.split("/")[0]
                name = r"\texttt{" + name + "}"
                if r["macro_f1_mean"] == best:
                    name = r"\textbf{" + name + "}"
                first = PROTO[proto] if r.equals(block.iloc[0]) else ""
                L.append(f"{first} & {name} & "
                         f"{cell(r['macro_f1_mean'], r['macro_f1_std'])} & "
                         f"{cell(r['mcc_mean'], r['mcc_std'])} & "
                         f"{cell(r['dr_mean'], r['dr_std'])} & "
                         f"{cell(r['far_mean'], r['far_std'])} & "
                         f"{r['sops_mean']:,.0f} \\\\".replace(",", "{,}"))
    L += [r"\bottomrule", r"\end{tabular}}", r"\end{table*}"]
    return "\n".join(L)


def main() -> int:
    df = collect()
    if df.empty:
        print("no strict runs found.")
        return 1
    df = restrict_to_axis_protocols(df)
    assert_structure(df)
    a = agg(df)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    a.to_csv(OUT_CSV, index=False)
    OUT_TEX.parent.mkdir(parents=True, exist_ok=True)
    OUT_TEX.write_text(to_tex(a) + "\n")

    print(f"{len(df)} runs -> {len(a)} (axis, protocol, configuration) cells")
    show = a[a.axis == "encoding"].copy()
    show["macro-F1"] = [cell(m, s).replace(" $\\pm$ ", "±")
                        for m, s in zip(show.macro_f1_mean, show.macro_f1_std)]
    print("\nencoding axis:")
    print(show[["protocol", "variant", "macro-F1", "mcc_mean", "dr_mean",
                "far_mean", "sops_mean"]].round(4).to_string(index=False))
    print(f"\nwrote {OUT_CSV}\nwrote {OUT_TEX}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
