#!/usr/bin/env python3
"""Primary ranking analysis, executed exactly as PREREGISTRATION.md specifies.

Consumes a long-form sweep CSV with one row per (dataset, neuron, encoding,
seed) and emits, into ``results/analysis/<tag>/``:

  per_dataset.csv        mean ± SD **over seeds, within a dataset** — the only
                         dispersion the manuscript is allowed to report
  blocks.csv             the 20 (dataset × seed) blocks with per-block ranks
  mean_rank.csv          primary ranking statistic, ascending
  friedman.json          omnibus test + Nemenyi critical difference
  focused_tests.json     the pre-specified paired comparisons, Holm-corrected
  pareto.csv             quality vs cost, with Pareto-optimal rows flagged
  cd_diagram.{pdf,png}   critical-difference diagram
  pareto.{pdf,png}       quality-cost frontier

Nothing here chooses what to test. The comparisons are fixed in the
pre-registration; this script only carries them out.

    python scripts/rank_analysis.py --input results/v1_submitted/sweep_540.csv --tag v1
"""
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare, rankdata, studentized_range, wilcoxon

ROOT = Path(__file__).resolve().parents[1]
import sys
sys.path.insert(0, str(ROOT))
from src.populations import require_population  # noqa: E402
from src.stats import hierarchical_bootstrap  # noqa: E402

PRIMARY = "f1_macro"
SUPPORTING = "matthews_corrcoef"
CHAMPION = ("LeakyParallel", "latency")
CHALLENGER = ("LeakyParallel", "rate")     # pre-specified focused comparison
ALPHA = 0.05


def variant(df: pd.DataFrame) -> pd.Series:
    return df["neuron"] + "/" + df["encoding"]


# ---------------------------------------------------------------- ranking
def build_blocks(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    """One row per (dataset, seed) block, one column per variant, holding that
    variant's metric. Ranks are computed per block, best = 1."""
    wide = df.pivot_table(index=["dataset", "seed"], columns="variant",
                          values=metric, aggfunc="mean")
    incomplete = wide.isna().any(axis=1)
    if incomplete.any():
        missing = wide[incomplete].isna().sum(axis=1)
        raise SystemExit(
            "blocks are not complete — Friedman requires every variant in every "
            f"block.\nIncomplete blocks (variants missing):\n{missing}")
    return wide


#: Ties within a block receive the average of the ranks they span, which is
#: scipy's default and the convention the Friedman/Nemenyi machinery assumes.
#: Named explicitly and emitted into the result payload so the manuscript can
#: state it rather than leaving a reader to infer it.
RANK_METHOD = "average"


def mean_ranks(wide: pd.DataFrame) -> pd.DataFrame:
    # negate: rankdata is ascending, and higher macro-F1 is better
    ranks = pd.DataFrame(
        np.apply_along_axis(lambda r: rankdata(-r, method=RANK_METHOD), 1,
                            wide.to_numpy()),
        index=wide.index, columns=wide.columns)
    out = pd.DataFrame({
        "mean_rank": ranks.mean(),
        "median_rank": ranks.median(),
        "sd_rank": ranks.std(ddof=1),
        "best_block_count": (ranks == 1).sum(),
    }).sort_values("mean_rank")
    return out, ranks


def nemenyi_cd(k: int, n: int, alpha: float = ALPHA) -> float:
    """Critical difference for the Nemenyi test at level alpha.

    q_alpha is the studentized-range quantile divided by sqrt(2); scipy exposes
    the distribution directly, which avoids the hard-coded lookup tables that
    silently cap at k=10 in most published implementations.
    """
    q = studentized_range.ppf(1 - alpha, k, np.inf) / np.sqrt(2)
    return float(q * np.sqrt(k * (k + 1) / (6.0 * n)))


# ---------------------------------------------------------------- tests
def holm(pvals: list) -> list:
    order = np.argsort(pvals)
    m = len(pvals)
    adj = np.empty(m)
    running = 0.0
    for i, idx in enumerate(order):
        running = max(running, (m - i) * pvals[idx])
        adj[idx] = min(1.0, running)
    return adj.tolist()


#: scipy's "wilcox" drops exact zero differences before ranking, so they enter
#: neither the signed-rank sum nor the effective sample size. That is the
#: behaviour the manuscript describes, and the tie count is reported separately
#: so a reader can see how many observations were dropped.
ZERO_METHOD = "wilcox"


def paired_test(wide: pd.DataFrame, a: str, b: str) -> dict:
    x, y = wide[a].to_numpy(), wide[b].to_numpy()
    d = x - y
    n_ties = int((d == 0).sum())
    if np.allclose(d, 0):
        stat, p = float("nan"), 1.0
    else:
        stat, p = wilcoxon(x, y, zero_method=ZERO_METHOD,
                           alternative="two-sided")
    by_ds = {ds: (wide.loc[ds, a] - wide.loc[ds, b]).to_numpy() * 100.0
             for ds in wide.index.get_level_values("dataset").unique()}
    boot = hierarchical_bootstrap(by_ds)
    return {
        "a": a, "b": b, "n_blocks": int(len(d)),
        "median_delta_pp": float(np.median(d) * 100),
        "mean_delta_pp": float(np.mean(d) * 100),
        "wins_a": int((d > 0).sum()), "wins_b": int((d < 0).sum()),
        "exact_ties": n_ties,
        "n_used_by_test": int(len(d) - n_ties),
        "zero_method": ZERO_METHOD,
        "rank_method": RANK_METHOD,
        # A displayed median of +0.00 pp can mean either a genuinely tiny
        # non-zero difference or exact equality; printing both resolves it.
        "full_precision_median_delta_pp": float(np.median(d) * 100),
        "display_median_delta_pp": f"{np.median(d) * 100:+.2f}",
        "wilcoxon_stat": None if np.isnan(stat) else float(stat),
        "wilcoxon_p": float(p),
        "bootstrap_effect_pp": boot["estimate_pp"],
        "bootstrap_ci_pp": boot["ci95_pp"],
        "bootstrap_p_two_sided": boot["bootstrap_p_two_sided"],
        "ci_excludes_zero": bool(boot["ci95_pp"][0] > 0 or boot["ci95_pp"][1] < 0),
        "per_dataset_delta_pp": {ds: float(np.mean(v)) for ds, v in by_ds.items()},
    }


# ---------------------------------------------------------------- pareto
def pareto_front(df: pd.DataFrame, quality: str, cost: str) -> np.ndarray:
    """Boolean mask of rows not dominated on (higher quality, lower cost)."""
    q, c = df[quality].to_numpy(), df[cost].to_numpy()
    keep = np.ones(len(df), dtype=bool)
    for i in range(len(df)):
        dominated = (q >= q[i]) & (c <= c[i]) & ((q > q[i]) | (c < c[i]))
        keep[i] = not dominated.any()
    return keep


# ---------------------------------------------------------------- figures
def cd_diagram(mr: pd.DataFrame, cd: float, path: Path, title: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    mr = mr.sort_values("mean_rank")
    names, ranks = list(mr.index), mr["mean_rank"].to_numpy()
    n = len(names)
    fig, ax = plt.subplots(figsize=(7.2, 0.30 * n + 1.7))
    ax.barh(range(n), ranks, color="#4C72B0", height=0.62)
    ax.set_yticks(range(n))
    ax.set_yticklabels(names, fontsize=7)
    ax.invert_yaxis()
    ax.set_xlabel("mean rank across (dataset × seed) blocks — lower is better")
    ax.set_title(title, fontsize=9)
    best = ranks[0]
    ax.axvline(best, color="#C44E52", lw=1.0)
    ax.axvline(best + cd, color="#C44E52", ls="--", lw=1.0)
    ax.text(best + cd, n - 0.4, f"  CD = {cd:.2f}", color="#C44E52",
            fontsize=7, va="bottom")
    ax.text(best, n - 0.4, "best  ", color="#C44E52", fontsize=7,
            va="bottom", ha="right")
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(path.with_suffix("." + ext), dpi=300)
    plt.close(fig)


def pareto_plot(p: pd.DataFrame, quality: str, cost: str, path: Path,
                title: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5.4, 3.8))
    enc_color = {"latency": "#4C72B0", "rate": "#DD8452", "delta": "#55A868"}
    for enc, g in p.groupby("encoding"):
        ax.scatter(g[cost], g[quality], s=26, alpha=0.85,
                   color=enc_color.get(enc, "#888888"), label=enc,
                   edgecolor="none")
    front = p[p["pareto"]].sort_values(cost)
    ax.plot(front[cost], front[quality], color="#C44E52", lw=1.0, ls="--",
            zorder=0, label="Pareto front")
    for _, r in front.iterrows():
        ax.annotate(r["variant"], (r[cost], r[quality]), fontsize=6,
                    xytext=(3, 3), textcoords="offset points")
    ax.set_xlabel(cost)
    ax.set_ylabel(quality + " (mean over datasets and seeds)")
    ax.set_title(title, fontsize=9)
    ax.legend(fontsize=7, frameon=False)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(path.with_suffix("." + ext), dpi=300)
    plt.close(fig)


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="results/v1_submitted/sweep_540.csv")
    ap.add_argument("--tag", default="v1")
    ap.add_argument("--metric", default=PRIMARY)
    args = ap.parse_args()

    out = ROOT / "results" / "analysis" / args.tag
    out.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(ROOT / args.input if not Path(args.input).is_absolute()
                     else args.input)
    df["variant"] = variant(df)
    require_population(df.variant.unique(), "full_27_screening")

    # ---- per-dataset mean ± seed SD (the only permitted dispersion) ----
    metrics = [c for c in ["accuracy", "f1_macro", "matthews_corrcoef",
                           "cohen_kappa", "detection_rate", "false_alarm_rate",
                           "balanced_accuracy", "energy_spikes_per_sample",
                           "inference_ms_per_sample"] if c in df.columns]
    per_ds = (df.groupby(["dataset", "variant"])[metrics]
                .agg(["mean", "std", "count"]))
    per_ds.columns = ["_".join(c) for c in per_ds.columns]
    per_ds.reset_index().to_csv(out / "per_dataset.csv", index=False)

    # ---- blocks and mean rank ----
    wide = build_blocks(df, args.metric)
    wide.to_csv(out / "blocks.csv")
    mr, ranks = mean_ranks(wide)
    mr.to_csv(out / "mean_rank.csv")

    k, n = wide.shape[1], wide.shape[0]
    stat, p = friedmanchisquare(*[wide[c].to_numpy() for c in wide.columns])
    cd = nemenyi_cd(k, n)
    best = mr.index[0]
    not_sep = mr[mr["mean_rank"] - mr["mean_rank"].iloc[0] <= cd].index.tolist()

    friedman = {
        "population_id": "full_27_screening",
        "rank_method": RANK_METHOD,
        "tie_handling": (f"ranks: {RANK_METHOD} within block; Wilcoxon zeros: "
                         f"{ZERO_METHOD} (exact ties dropped before ranking)"),
        "metric": args.metric, "k_variants": int(k), "n_blocks": int(n),
        "friedman_chi2": float(stat), "friedman_p": float(p),
        "rejects_at_alpha": bool(p < ALPHA), "alpha": ALPHA,
        "nemenyi_critical_difference": cd,
        "best_by_mean_rank": best,
        "best_mean_rank": float(mr["mean_rank"].iloc[0]),
        "champion": "/".join(CHAMPION),
        "champion_mean_rank": float(mr.loc["/".join(CHAMPION), "mean_rank"]),
        "champion_rank_position": int(mr.index.get_loc("/".join(CHAMPION)) + 1),
        "statistically_indistinguishable_from_best": not_sep,
    }
    (out / "friedman.json").write_text(json.dumps(friedman, indent=2))

    # ---- pre-specified focused comparisons, Holm-corrected ----
    champ, chall = "/".join(CHAMPION), "/".join(CHALLENGER)
    best_non_lp = next(v for v in mr.index if not v.startswith("LeakyParallel"))
    tests = [paired_test(wide, champ, chall),
             paired_test(wide, champ, best_non_lp)]
    if best != champ:
        tests.append(paired_test(wide, champ, best))
    adj = holm([t["wilcoxon_p"] for t in tests])
    for t, a in zip(tests, adj):
        t["holm_adjusted_p"] = float(a)
        t["significant_after_holm"] = bool(a < ALPHA)
    (out / "focused_tests.json").write_text(json.dumps(
        {"alpha": ALPHA, "correction": "Holm", "family_size": len(tests),
         "tests": tests}, indent=2))

    # ---- encoding-level question, kept separate from the variant question ----
    enc_wide = df.pivot_table(index=["dataset", "seed"], columns="encoding",
                              values=args.metric, aggfunc="mean")
    enc_stat, enc_p = friedmanchisquare(*[enc_wide[c].to_numpy()
                                          for c in enc_wide.columns])
    enc_pairs = {}
    for a, b in itertools.combinations(enc_wide.columns, 2):
        enc_pairs[f"{a}_vs_{b}"] = paired_test(enc_wide, a, b)
    enc_adj = holm([v["wilcoxon_p"] for v in enc_pairs.values()])
    for (key, v), a in zip(enc_pairs.items(), enc_adj):
        v["holm_adjusted_p"] = float(a)
        v["significant_after_holm"] = bool(a < ALPHA)
    (out / "encoding_tests.json").write_text(json.dumps({
        "friedman_chi2": float(enc_stat), "friedman_p": float(enc_p),
        "mean_rank": {c: float(v) for c, v in
                      pd.DataFrame(np.apply_along_axis(
                          lambda r: rankdata(-r), 1, enc_wide.to_numpy()),
                          columns=enc_wide.columns).mean().items()},
        "pairwise": enc_pairs}, indent=2))

    # ---- Pareto over quality and cost ----
    cost_col = ("energy_spikes_per_sample" if "energy_spikes_per_sample" in df
                else "inference_ms_per_sample")
    agg = (df.groupby("variant")
             .agg(**{args.metric: (args.metric, "mean"),
                     cost_col: (cost_col, "mean")})
             .reset_index())
    agg["neuron"] = agg["variant"].str.split("/").str[0]
    agg["encoding"] = agg["variant"].str.split("/").str[1]
    agg["pareto"] = pareto_front(agg, args.metric, cost_col)
    agg = agg.merge(mr.reset_index().rename(columns={"index": "variant"}),
                    on="variant", how="left").sort_values("mean_rank")
    agg.to_csv(out / "pareto.csv", index=False)

    cd_diagram(mr, cd, out / "cd_diagram",
               f"{args.tag}: {args.metric}, {k} variants over {n} blocks")
    pareto_plot(agg, args.metric, cost_col, out / "pareto",
                f"{args.tag}: quality vs {cost_col}")

    # ---- console summary ----
    print(f"\n=== {args.tag}: {args.metric} · {k} variants × {n} blocks ===")
    print(f"Friedman chi2={stat:.2f}  p={p:.3e}  "
          f"{'REJECTS' if p < ALPHA else 'does not reject'} at {ALPHA}")
    print(f"Nemenyi CD = {cd:.3f}\n")
    print(mr.head(8).round(3).to_string())
    print(f"\nchampion {champ}: rank position "
          f"{friedman['champion_rank_position']}/{k}, "
          f"mean rank {friedman['champion_mean_rank']:.2f}")
    print(f"indistinguishable from best ({best}): {len(not_sep)} variants")
    print("\n--- focused tests (Holm-corrected) ---")
    for t in tests:
        print(f"{t['a']} vs {t['b']}: median Δ={t['median_delta_pp']:+.3f} pp, "
              f"wins {t['wins_a']}-{t['wins_b']}, p={t['wilcoxon_p']:.4f}, "
              f"holm={t['holm_adjusted_p']:.4f} "
              f"{'SIG' if t['significant_after_holm'] else 'n.s.'}, "
              f"boot {t['bootstrap_effect_pp']:+.2f} pp "
              f"[{t['bootstrap_ci_pp'][0]:+.2f}, {t['bootstrap_ci_pp'][1]:+.2f}]")
        print(f"    per-dataset Δ pp: "
              + ", ".join(f"{d}={x:+.2f}" for d, x in
                          t["per_dataset_delta_pp"].items()))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
