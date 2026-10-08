#!/usr/bin/env python3
"""How far does the screening protocol displace the design-space ranking?

This is the paper's own question reduced to one number. Both rankings cover the
same 27 configurations; only the evaluation protocol differs. Kendall's tau
between them, with the permutation test registered in
PREREGISTRATION_ADDENDUM.md §4 (10,000 permutations, RNG seed 20260814),
measures the displacement directly.

Runs against whatever confirmation protocols are complete, and says which those
are. A partial answer computed on four protocols is useful and clearly labelled;
a silent one computed on three would not be.

Emits:
  tau.json                 tau, permutation p, movers, protocols covered
  rank_comparison.csv      per configuration: screening rank, confirmation rank
  fig_rank_shift.{pdf,png} slopegraph, screening -> confirmation

    python scripts/protocol_shift.py
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kendalltau, rankdata

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from src.populations import macro_f1_pair, members, require_population  # noqa: E402
from src.readout import discover  # noqa: E402
from figstyle import (ANNOT, COL_W, ENC, ENC_ORDER, FS_TICK, RC,  # noqa: E402
                      TEXT)

OUT = ROOT / "results/analysis/protocol_shift"
FIG = ROOT / "paper/figures"
CONF_PROTOCOLS = ["nslkdd_v2", "kddcup99_v2", "cicids2017_v2",
                  "ctu13_v2_f0", "ctu13_v2_f1"]
N_PERM, PERM_SEED = 10_000, 20260814


def mean_ranks(wide: pd.DataFrame) -> pd.Series:
    r = np.apply_along_axis(lambda v: rankdata(-v, method="average"), 1,
                            wide.to_numpy())
    return pd.Series(r.mean(axis=0), index=wide.columns)


def screening_ranks() -> pd.Series:
    """The screening mean ranks, read from the committed artifact.

    tau is a comparison between two rank vectors, so it is wrong --- and wrong
    invisibly --- if this vector is not the one the manuscript reports. Both
    sides used to be computed independently from `sweep_540.csv` by
    `rank_analysis.py` and by this script. They agree exactly today, but they
    agree by shared convention rather than by construction: a change to
    tie-breaking or to the block definition in one would silently move tau
    without moving any number in the paper.

    So the committed artifact is authoritative and the recomputation is kept
    only as a cross-check. If they ever diverge this raises instead of
    returning a plausible tau.
    """
    d = pd.read_csv(ROOT / "results/v1_submitted/sweep_540.csv")
    d["variant"] = d.neuron + "/" + d.encoding
    require_population(d.variant.unique(), "full_27_screening")
    recomputed = mean_ranks(d.pivot_table(index=["dataset", "seed"],
                                          columns="variant", values="f1_macro"))

    published = ROOT / "results/analysis/v1/mean_rank.csv"
    if not published.exists():
        raise SystemExit(
            f"{published} is missing. Run scripts/rank_analysis.py first: tau "
            "must be computed against the screening ranks the manuscript "
            "reports, not against a private recomputation.")
    authoritative = pd.read_csv(published, index_col=0)["mean_rank"]

    if set(authoritative.index) != set(recomputed.index):
        raise SystemExit(
            "screening rank vectors disagree on membership: "
            f"{len(authoritative)} published vs {len(recomputed)} recomputed.")
    delta = (authoritative - recomputed.reindex(authoritative.index)).abs().max()
    if delta > 1e-9:
        raise SystemExit(
            f"screening ranks disagree by up to {delta:.3e} between "
            f"{published.relative_to(ROOT)} and this script's recomputation. "
            "tau would be computed against a vector the paper does not report. "
            "Reconcile rank_analysis.py and protocol_shift.py before "
            "continuing.")
    return authoritative


def confirmation_ranks():
    """Mean ranks over whatever confirmation protocols are complete.

    A protocol enters only when all 27 configurations are present at 5 seeds;
    a partially-filled protocol would rank an incomplete field.
    """
    rows = []
    for r in discover():
        if r.protocol not in CONF_PROTOCOLS:
            continue
        rec = r.record()
        pre = r.prefix()
        m = macro_f1_pair(pre["y_true"].astype(np.int64),
                          pre["y_pred"].astype(np.int64),
                          rec["data"]["num_classes"])
        rows.append({"protocol": r.protocol, "variant": r.variant,
                     "seed": r.seed, "f1": m["macro_f1_fixed_universe"]})
    df = pd.DataFrame(rows)
    if df.empty:
        return pd.Series(dtype=float), [], df
    full = [p for p, g in df.groupby("protocol")
            if g.variant.nunique() == 27 and g.groupby("variant").seed.nunique().min() == 5]
    if not full:
        return pd.Series(dtype=float), [], df
    sub = df[df.protocol.isin(full)]
    require_population(sub.variant.unique(), "full_27_confirmation")
    wide = sub.pivot_table(index=["protocol", "seed"], columns="variant",
                           values="f1").dropna()
    return mean_ranks(wide), full, df


#: Configurations named in the Results prose as the largest movers. They are
#: emphasised so the sentence and the picture point at the same lines.
HIGHLIGHT = ("SLSTM/latency", "SConv2dLSTM/delta", "Lapicque/latency")
SELECTED = "LeakyParallel/latency"


def slopegraph(cmp: pd.DataFrame, protocols, path: Path,
               tau: float | None = None, n_screening_blocks: int = 20) -> None:
    """Screening rank on the left, confirmation rank on the right.

    Positioned by rank POSITION (1..27), evenly spaced, not by the mean-rank
    value. The mean ranks cluster: eleven of the twenty-seven configurations sit
    between 10 and 16, so a value-positioned slopegraph stacks their labels on
    top of one another. Even spacing makes collisions impossible, and what a
    slopegraph is for --- who crossed whom --- is carried by the ordering.

    Encoding is carried by colour, line style and end markers together, using
    the shared mapping in figstyle.py. Delta was grey here and green on the
    Pareto plot; one constant now serves both.

    Emphasis is reserved: the selected configuration and the three movers named
    in the text are the only bold labels. Bolding every latency row, as an
    earlier version did, weights the family before the reader has judged the
    movement the figure exists to show.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    matplotlib.rcParams.update(RC)
    matplotlib.rcParams.update({
        "axes.grid": False,
        "axes.spines.left": False, "axes.spines.bottom": False,
    })
    fig, ax = plt.subplots(figsize=(COL_W, 4.95), layout="constrained")

    for _, r in cmp.iterrows():
        enc = r["variant"].split("/")[1]
        st = ENC[enc]
        sel = r["variant"] == SELECTED
        hot = r["variant"] in HIGHLIGHT
        lw = 2.0 if sel else (1.7 if hot else 0.75)
        ax.plot([0, 1], [r.pos_screening, r.pos_confirmation],
                color=st["c"], ls=st["ls"], lw=lw,
                marker=st["m"], ms=2.6 if (sel or hot) else 1.8,
                markevery=[0, 1],
                alpha=1.0 if (sel or hot) else 0.62,
                zorder=5 if sel else (4 if hot else 2),
                solid_capstyle="round")

    for side, xpos, ha, dx in (("pos_screening", 0.0, "right", -0.045),
                               ("pos_confirmation", 1.0, "left", 0.045)):
        rank_col = ("rank_screening" if side == "pos_screening"
                    else "rank_confirmation")
        for _, r in cmp.iterrows():
            sel = r["variant"] == SELECTED
            hot = r["variant"] in HIGHLIGHT
            txt = (f"{r['variant']}  {r[rank_col]:.2f}" if ha == "left"
                   else f"{r[rank_col]:.2f}  {r['variant']}")
            ax.text(xpos + dx, r[side], txt, fontsize=5.1, ha=ha, va="center",
                    color=(ENC["latency"]["c"] if sel else
                           TEXT if hot else ANNOT),
                    fontweight="bold" if (sel or hot) else "normal")

    ax.set_xlim(-1.05, 2.05)
    ax.set_ylim(len(cmp) + 0.9, 0.1)
    ax.set_xticks([0, 1])
    ax.set_xticklabels([f"screening\n({n_screening_blocks} blocks)",
                        f"confirmation\n({5 * len(protocols)} blocks)"],
                       fontsize=FS_TICK)
    ax.tick_params(axis="x", length=0, pad=4)
    ax.set_yticks([])
    ax.set_ylabel("rank position (1 = best, top)")

    if tau is not None:
        ax.text(1.0, 1.012, rf"Kendall $\tau={tau:.4f}$", transform=ax.transAxes,
                ha="right", va="bottom", fontsize=FS_TICK, color=TEXT)

    ax.legend(handles=[Line2D([], [], color=ENC[e]["c"], ls=ENC[e]["ls"],
                              marker=ENC[e]["m"], ms=2.8, lw=1.2, label=e)
                       for e in ENC_ORDER],
              loc="lower left", bbox_to_anchor=(0.0, 1.005), ncol=3,
              handlelength=1.8, columnspacing=1.0, fontsize=FS_TICK,
              borderaxespad=0.0)

    for ext in ("pdf", "png"):
        fig.savefig(path.with_suffix("." + ext))
    plt.close(fig)


def rehearsal(scr, conf, protocols, raw, out_dir: Path) -> int:
    """Pipeline rehearsal on the complete-so-far protocols. Not a result.

    Deliberately narrower than the registered run: ranking and encoding
    aggregation only. No tau, no permutation test, no figure, nothing on a path
    the manuscript reads.
    """
    print("=" * 78)
    print("PARTIAL_4PROTOCOL_REHEARSAL --- pipeline exercise, NOT a result")
    print("Kendall tau deliberately not computed: registered at 25 blocks.")
    print("Prediction frozen before this run: PARTIAL_4PROTOCOL_PREDICTION.md")
    print("=" * 78)
    if conf.empty:
        have = (raw.groupby("protocol").variant.nunique().to_dict()
                if not raw.empty else {})
        print("no confirmation protocol is complete at 27 configurations x 5 "
              f"seeds yet.\n  configurations present per protocol: {have}")
        return 0

    cmp = pd.DataFrame({"rank_screening": scr,
                        "rank_confirmation": conf}).dropna()
    cmp.index.name = "variant"
    cmp = cmp.reset_index().sort_values("rank_confirmation")
    cmp["pos_screening"] = rankdata(cmp.rank_screening, method="average")
    cmp["pos_confirmation"] = rankdata(cmp.rank_confirmation, method="average")
    cmp["movement"] = cmp.pos_screening - cmp.pos_confirmation
    cmp.to_csv(out_dir / "rank_comparison_PARTIAL.csv", index=False,
               float_format="%.6f")

    enc = encoding_aggregation(raw, protocols)
    enc.to_csv(out_dir / "encoding_aggregation_PARTIAL.csv",
               float_format="%.6f")
    missing = [p for p in CONF_PROTOCOLS if p not in protocols]
    (out_dir / "STATUS.json").write_text(json.dumps({
        "status": "PARTIAL_4PROTOCOL_REHEARSAL",
        "protocols_included": protocols, "protocols_missing": missing,
        "n_blocks": int(len(protocols) * 5),
        "kendall_tau": None,
        "tau_withheld_because": "registered at 25 blocks in "
                                "PREREGISTRATION_ADDENDUM.md §4",
        "enters_manuscript": False,
    }, indent=2))

    print(f"\nprotocols included ({len(protocols)}): {', '.join(protocols)}")
    print(f"protocols missing  ({len(missing)}): {', '.join(missing) or 'none'}")
    print(f"blocks: {len(protocols) * 5} of 25\n")
    print("top 10 by confirmation rank:")
    print(cmp.head(10)[["variant", "rank_screening", "rank_confirmation",
                        "movement"]].round(3).to_string(index=False))
    print("\nencoding aggregation (mean rank over the 27-way ranking):")
    print(enc.round(4).to_string())
    print("\nBias direction committed in advance: CIC-IDS2017 carries the "
          "largest\nlatency-over-rate margin of the five, so its absence pulls "
          "latency down here.")
    print(f"\nwrote {out_dir}  (not read by finalize.py)")
    return 0


def encoding_aggregation(raw: pd.DataFrame, protocols) -> pd.DataFrame:
    """Mean rank collapsed by encoding, over the blocks actually present.

    Ranks are formed over all 27 configurations first and then averaged within
    an encoding, so this is a summary of the same ranking rather than a separate
    9-way contest.
    """
    sub = raw[raw.protocol.isin(protocols)]
    wide = sub.pivot_table(index=["protocol", "seed"], columns="variant",
                           values="f1").dropna()
    mr = mean_ranks(wide)
    enc = pd.Series({v: v.split("/")[1] for v in mr.index})
    out = (pd.DataFrame({"mean_rank": mr, "encoding": enc})
           .groupby("encoding").mean_rank.agg(["mean", "min", "count"])
           .rename(columns={"mean": "mean_rank_over_family",
                            "min": "best_configuration_rank",
                            "count": "n_configurations"})
           .sort_values("mean_rank_over_family"))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rehearsal", action="store_true",
                    help="PARTIAL_4PROTOCOL_REHEARSAL: exercise the pipeline on "
                         "the protocols complete so far. Writes to a separate "
                         "directory, emits no figure, and computes NO tau --- "
                         "tau is registered at 25 blocks and a value on fewer "
                         "protocols has no registered status. See "
                         "PARTIAL_4PROTOCOL_PREDICTION.md.")
    args = ap.parse_args()

    scr = screening_ranks()
    conf, protocols, raw = confirmation_ranks()
    out_dir = ROOT / ("results/analysis/protocol_shift_partial4" if args.rehearsal
                      else "results/analysis/protocol_shift")
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.rehearsal:
        return rehearsal(scr, conf, protocols, raw, out_dir)
    OUT.mkdir(parents=True, exist_ok=True)

    if conf.empty:
        have = (raw.groupby("protocol").variant.nunique().to_dict()
                if not raw.empty else {})
        print("no confirmation protocol has all 27 configurations at 5 seeds yet.")
        print(f"  configurations present per protocol: {have}")
        return 0

    cmp = pd.DataFrame({"rank_screening": scr, "rank_confirmation": conf}).dropna()
    cmp.index.name = "variant"
    cmp = cmp.reset_index()
    cmp["pos_screening"] = rankdata(cmp.rank_screening, method="average")
    cmp["pos_confirmation"] = rankdata(cmp.rank_confirmation, method="average")
    cmp["movement"] = cmp.pos_screening - cmp.pos_confirmation   # + = improved
    cmp = cmp.sort_values("rank_confirmation")
    cmp.to_csv(OUT / "rank_comparison.csv", index=False, float_format="%.6f")

    tau, _ = kendalltau(cmp.rank_screening, cmp.rank_confirmation)
    rng = np.random.default_rng(PERM_SEED)
    a = cmp.rank_screening.to_numpy()
    b = cmp.rank_confirmation.to_numpy()
    null = np.array([kendalltau(a, rng.permutation(b))[0] for _ in range(N_PERM)])
    p = float((np.abs(null) >= abs(tau)).mean())

    up = cmp.nlargest(5, "movement")[["variant", "pos_screening",
                                      "pos_confirmation", "movement"]]
    dn = cmp.nsmallest(5, "movement")[["variant", "pos_screening",
                                       "pos_confirmation", "movement"]]
    payload = {
        "protocols_included": protocols,
        "n_protocols": len(protocols),
        "complete": len(protocols) == len(CONF_PROTOCOLS),
        "n_configurations": int(len(cmp)),
        "kendall_tau": float(tau),
        "permutation_p": p,
        "n_permutations": N_PERM, "rng_seed": PERM_SEED,
        "leader_screening": cmp.sort_values("rank_screening").variant.iloc[0],
        "leader_confirmation": cmp.variant.iloc[0],
        "biggest_gainers": up.to_dict("records"),
        "biggest_losers": dn.to_dict("records"),
    }
    (OUT / "tau.json").write_text(json.dumps(payload, indent=2))
    _scr_blocks = json.load(open(ROOT / "results/analysis/v1/friedman.json"))["n_blocks"]
    slopegraph(cmp, protocols, FIG / "fig_rank_shift", tau=float(tau),
               n_screening_blocks=int(_scr_blocks))

    status = "COMPLETE" if payload["complete"] else "PARTIAL"
    print(f"=== protocol shift [{status}] over {len(protocols)} protocols: "
          f"{', '.join(protocols)} ===")
    print(f"Kendall tau = {tau:.4f}  (permutation p = {p:.4f}, "
          f"{N_PERM} permutations, seed {PERM_SEED})")
    print(f"leader screening    : {payload['leader_screening']}")
    print(f"leader confirmation : {payload['leader_confirmation']}")
    print("\ntop 10 by confirmation rank:")
    print(cmp.head(10)[["variant", "rank_screening", "rank_confirmation",
                        "movement"]].round(3).to_string(index=False))
    print("\nlargest upward movement:"); print(up.to_string(index=False))
    print("largest downward movement:"); print(dn.to_string(index=False))
    print(f"\nwrote {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
