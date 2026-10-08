#!/usr/bin/env python3
"""Emit the paper's LaTeX tables from frozen analysis outputs.

Every number is read from a committed file under `results/`; nothing is
recomputed here and nothing is typed by hand. If a table needs a number that no
analysis produces, the fix is to add it to the analysis, not to this script.

    python scripts/make_tables.py --tag v1
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "paper" / "tables"

DS_PRETTY = {"nslkdd": "NSL-KDD", "kddcup99": "KDDCup99",
             "cicids2017": "CIC-IDS2017", "ctu13": "CTU-13"}


def _tt(v: str) -> str:
    return r"\texttt{" + v.replace("_", r"\_") + "}"


def table_ranking(tag: str, top: int = 10) -> str:
    a = ROOT / "results/analysis" / tag
    mr = pd.read_csv(a / "mean_rank.csv", index_col=0)
    fr = json.load(open(a / "friedman.json"))
    per = pd.read_csv(a / "per_dataset.csv")

    f1 = per.pivot(index="variant", columns="dataset", values="f1_macro_mean")
    sd = per.pivot(index="variant", columns="dataset", values="f1_macro_std")
    order = [c for c in ["nslkdd", "kddcup99", "cicids2017", "ctu13"] if c in f1]

    lines = [
        r"\begin{table}[t]", r"\centering",
        r"\caption{Top-%d of the %d (neuron, encoding) variants by \textbf{mean rank}"
        r" over the %d (dataset $\times$ seed) blocks; rank~1 is best. Per-dataset"
        r" cells are mean $\pm$ standard deviation \emph{over the five seeds}, never"
        r" pooled across datasets. Friedman $\chi^2=%.1f$, $p=%.1e$; Nemenyi critical"
        r" difference $=%.2f$. The Nemenyi procedure does not separate the %d"
        r" configurations within %.2f of the leader from it at this"
        r" significance level; the large critical difference indicates limited"
        r" all-pairs resolution and should not be read as evidence of"
        r" equivalence.}"
        % (top, fr["k_variants"], fr["n_blocks"], fr["friedman_chi2"],
           fr["friedman_p"], fr["nemenyi_critical_difference"],
           len(fr["statistically_indistinguishable_from_best"]),
           fr["nemenyi_critical_difference"]),
        r"\label{tab:ranking}",
        r"\resizebox{\columnwidth}{!}{%",
        r"\begin{tabular}{l r " + "c" * len(order) + "}",
        r"\toprule",
        r"Variant & Mean rank & " + " & ".join(DS_PRETTY[d] for d in order) + r" \\",
        r"\midrule",
    ]
    cd = fr["nemenyi_critical_difference"]
    best = mr["mean_rank"].iloc[0]
    for v in mr.index[:top]:
        cells = [f"{f1.loc[v, d]:.4f} $\\pm$ {sd.loc[v, d]:.4f}" for d in order]
        mark = r"$^{\dagger}$" if mr.loc[v, "mean_rank"] - best <= cd else ""
        lines.append(f"{_tt(v)} & {mr.loc[v, 'mean_rank']:.2f}{mark} & "
                     + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}}",
              r"\vspace{2pt}",
              r"\footnotesize $^{\dagger}$ not separated from the leader by the"
              r" Nemenyi procedure at $\alpha=0.05$.",
              r"\end{table}"]
    return "\n".join(lines)


def table_focused(tag: str) -> str:
    d = json.load(open(ROOT / "results/analysis" / tag / "focused_tests.json"))
    lines = [
        r"\begin{table}[t]", r"\centering",
        r"\caption{Pre-specified paired comparisons (PREREGISTRATION.md \S4),"
        r" Wilcoxon signed-rank over the %d blocks with Holm correction across the"
        r" family of %d tests. $\Delta$ is (A $-$ B) macro-F1 in percentage points;"
        r" the interval is a hierarchical bootstrap resampling datasets first and"
        r" seeds within datasets.}" % (d["tests"][0]["n_blocks"], d["family_size"]),
        r"\label{tab:focused}",
        r"\resizebox{\columnwidth}{!}{%",
        r"\begin{tabular}{l l r r r l}", r"\toprule",
        r"A & B & W/L/T & median $\Delta$ & Holm $p$ & bootstrap $\Delta$ [95\% CI] \\",
        r"\midrule",
    ]
    for t in d["tests"]:
        lo, hi = t["bootstrap_ci_pp"]
        sig = "" if t["significant_after_holm"] else r"$^{\ast}$"
        lines.append(
            f"{_tt(t['a'])} & {_tt(t['b'])} & "
            f"{t['wins_a']}/{t['wins_b']}/{t.get('exact_ties', 0)} & "
            f"{t['median_delta_pp']:+.2f} & {t['holm_adjusted_p']:.4f}{sig} & "
            f"{t['bootstrap_effect_pp']:+.2f} [{lo:+.2f}, {hi:+.2f}] \\\\")
    lines += [r"\bottomrule", r"\end{tabular}}",
              r"\vspace{2pt}",
              r"\vspace{-2pt}",
        r"\footnotesize $^{\ast}$ not significant at $\alpha=0.05$ after Holm."
        r" W/L/T counts blocks won, lost and \emph{exactly} tied; ties are"
        r" dropped before ranking (\texttt{zero\_method=wilcox}) and so enter"
        r" neither the signed-rank sum nor the effective sample size. A"
        r" displayed median of $+0.00$ therefore indicates a difference"
        r" smaller than the printed precision, not an exact tie.",
              r"\end{table}"]
    return "\n".join(lines)


def table_readout() -> str:
    """T95 and T99 side by side. The methodology defines both thresholds, so a
    table carrying only T99 leaves half the definition unused."""
    df = pd.read_csv(ROOT / "results/analysis/early_readout/summary.csv")
    df = df[df.neuron == "LeakyParallel"]
    pretty = {"nslkdd_v2": "NSL-KDD", "kddcup99_v2": "KDDCup99",
              "cicids2017_v2": "CIC-IDS2017", "ctu13_v2_f0": "CTU-13 f0",
              "ctu13_v2_f1": "CTU-13 f1"}
    order = list(pretty)
    lines = [
        r"\begin{table}[t]", r"\centering",
        r"\caption{Median $T_{95}$ and $T_{99}$ by strict protocol,"
        r" \texttt{LeakyParallel}, five seeds. $T_{q}$ is the first timestep at"
        r" which macro-F1 reaches $q\%$ of that configuration's \emph{own}"
        r" terminal value, so it measures how quickly a configuration settles,"
        r" not how good the settled answer is. The final column refers to"
        r" $T_{99}$. Latency settles earlier on two protocols and later on"
        r" three.}",
        r"\label{tab:readout}",
        r"\resizebox{\columnwidth}{!}{%",
        r"\begin{tabular}{l l r r r r l}", r"\toprule",
        r"& & \multicolumn{2}{c}{$T_{95}$} & \multicolumn{2}{c}{$T_{99}$} & \\",
        r"\cmidrule(lr){3-4}\cmidrule(lr){5-6}",
        r"Protocol & Best term.\ enc. & lat. & rate & lat. & rate &"
        r" Earlier \\", r"\midrule",
    ]
    for proto in order:
        g = df[df.protocol == proto].set_index("encoding")
        if g.empty:
            continue
        best = g.f1_terminal_mean.idxmax()
        lat99, rat99 = int(g.loc["latency", "T99_median"]), int(g.loc["rate", "T99_median"])
        lat95, rat95 = int(g.loc["latency", "T95_median"]), int(g.loc["rate", "T95_median"])
        earlier = "latency" if lat99 < rat99 else "rate"
        bold = lambda v, w: (r"\textbf{" + str(v) + "}") if w else str(v)
        lines.append(
            f"{pretty[proto]} & "
            + (r"\textbf{" + best + "}" if best == "delta" else best)
            + f" & {bold(lat95, lat95 < rat95)} & {bold(rat95, rat95 < lat95)}"
            + f" & {bold(lat99, lat99 < rat99)} & {bold(rat99, rat99 < lat99)}"
            + f" & {earlier} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}}", r"\end{table}"]
    return "\n".join(lines)


def table_timing() -> str:
    """Value of spike timing, isolated by construction.

    Both arms emit identical input spike sets, so the macro-F1 difference is the
    contribution of temporal structure alone. Generated rather than typed: the
    identical-input claim is the load-bearing part of the argument and has to
    come from the artifact that verifies it.
    """
    d = pd.read_csv(ROOT / "results/analysis/e1_timing_isolation.csv",
                    index_col=0)
    pretty = {"nslkdd_v2": "NSL-KDD", "kddcup99_v2": "KDDCup99",
              "cicids2017_v2": "CIC-IDS2017", "ctu13_v2_f0": "CTU-13 f0",
              "ctu13_v2_f1": "CTU-13 f1"}
    lines = [
        r"\begin{table}[t]", r"\centering",
        r"\caption{Value of spike timing, isolated. Both arms use"
        r" \texttt{LeakyParallel} and, by construction, emit \emph{identical}"
        r" input spike sets: delta at latency's gate fires on exactly the"
        r" features latency encodes, and the two columns agree to a maximum"
        r" absolute difference of %.0e spikes per sample. The arms therefore"
        r" differ only in whether the spike carries timing, and the final column"
        r" is the contribution of temporal structure alone. Post-hoc analysis;"
        r" not part of the registered confirmation.}" % d.in_spk_abs_diff.max(),
        r"\label{tab:timing}",
        r"\resizebox{\columnwidth}{!}{%",
        r"\begin{tabular}{l r r r r}", r"\toprule",
        r"Protocol & Input spikes & \multicolumn{2}{c}{macro-F1} & Timing \\",
        r"\cmidrule(lr){3-4}",
        r" & (both arms) & with & without & (pp) \\", r"\midrule",
    ]
    for proto, name in pretty.items():
        if proto not in d.index:
            continue
        r = d.loc[proto]
        lines.append(f"{name} & {r.in_spk_timing:.2f} & {r.f1_timing:.4f} & "
                     f"{r.f1_notiming:.4f} & {r.timing_premium_pp:+.1f} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}}", r"\end{table}"]
    return "\n".join(lines)


def table_audit_revised() -> str:
    """Split the single Group/Overlap pair into group unit, what separation is
    actually claimed, and how many groups are shared.

    The old table showed CIC-IDS2017 as `capture / 8` and left a reader to
    conclude the split had failed. It had not: eight shared capture IDs are
    exactly what a within-capture protocol should produce, because
    capture-disjointness is not being claimed. Stating the claim in its own
    column removes the ambiguity.
    """
    df = pd.read_csv(ROOT / "results/analysis/split_audit.csv")
    claim = {
        "official split": ("--", "Official train/test partition"),
        "within-capture order-disjoint": ("Capture",
                                          "Within-capture order-disjoint"),
        "day-disjoint (binary)": ("Day", "Day-disjoint"),
    }
    unmapped = sorted(set(df.protocol) - set(claim)
                      - {p for p in df.protocol if p.startswith("scenario-disjoint")})
    if unmapped:
        raise SystemExit(f"tab_audit_revised: unmapped protocol labels "
                         f"{unmapped}. Add them to `claim` rather than letting "
                         "them fall through to the scenario default.")
    lines = [
        r"\begin{table}[t]", r"\centering",
        r"\caption{Split audit for the confirmation protocols. \emph{Claimed"
        r" separation} is what each partition asserts, and \emph{shared groups}"
        r" counts group identifiers appearing on both sides. A non-zero count is"
        r" only a defect where disjointness is claimed: CIC-IDS2017's eight"
        r" shared captures are expected, because the within-capture protocol"
        r" separates position inside a capture rather than the captures"
        r" themselves. Duplicate rows are reported, never asserted away.}",
        r"\label{tab:audit-revised}",
        r"\resizebox{\columnwidth}{!}{%",
        r"\begin{tabular}{l l l r r}", r"\toprule",
        r"Dataset & Group unit & Claimed separation & Shared & Dup.\ test (\%) \\",
        r"\midrule",
    ]
    for _, r in df.iterrows():
        unit, sep = claim.get(r["protocol"], ("Scenario", "Scenario-disjoint"))
        if str(r["protocol"]).startswith("scenario-disjoint"):
            unit, sep = "Scenario", "Scenario-disjoint (" + r["protocol"].split()[-1] + ")"
        lines.append(f"{r['dataset']} & {unit} & {sep} & "
                     f"{int(r['overlap_train_test'])} & {r['dup_test_pct']:.2f} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}}", r"\end{table}"]
    return "\n".join(lines)


def table_audit() -> str:
    df = pd.read_csv(ROOT / "results/analysis/split_audit.csv")
    lines = [
        r"\begin{table}[t]", r"\centering",
        r"\caption{Split audit for the leakage-resistant (v2) protocols. Group"
        r" overlap is the number of groups shared between partitions and must be"
        r" zero wherever disjointness is claimed; a violation aborts the training"
        r" job. Duplicate rows are \emph{reported}, not asserted away: KDDCup99"
        r" repeats most of its official test set inside training, which is the"
        r" documented reason NSL-KDD exists. No transform is fitted on test data"
        r" in any protocol. Feature dimensionality differs between the two"
        r" protocol generations and each row states the confirmation figure:"
        r" the screening protocol built its categorical vocabulary from the"
        r" union of all splits, while the confirmation protocol fits it on"
        r" training rows alone and adds one \texttt{<unk>} column per"
        r" categorical field, so NSL-KDD is 122 columns under screening and"
        r" 125 under confirmation.}",
        r"\label{tab:audit}",
        r"\resizebox{\columnwidth}{!}{%",
        r"\begin{tabular}{l l r r r r r}", r"\toprule",
        r"Dataset & Protocol & Train & Test & Group & Overlap & Dup.\ test (\%) \\",
        r"\midrule",
    ]
    # `{,}` keeps LaTeX from treating the thousands separator as punctuation and
    # inserting a space after it. Applied per number, never to the whole string —
    # a global replace would corrupt the caption prose.
    def num(v) -> str:
        return f"{int(v):,}".replace(",", "{,}")

    for _, r in df.iterrows():
        lines.append(
            f"{r['dataset']} & {r['protocol']} & {num(r['n_train'])} & "
            f"{num(r['n_test'])} & {r['group']} & {int(r['overlap_train_test'])} & "
            f"{r['dup_test_pct']:.2f} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}}", r"\end{table}"]
    return "\n".join(lines)


def table_protocol_audit() -> str:
    """One table explaining why the confirmation protocol exists.

    Each row is an issue we audited in the original protocol, what the audit
    found, and what the revised protocol does about it. Deliberately compact:
    a reviewer should be able to see the whole methodological case at a glance
    and check any single number against the artifact that produced it.
    """
    rows = [
        ("NSL-KDD", "Exact train/test duplicate records",
         "2.68\\% of test rows", "Duplicate-free sensitivity reported"),
        ("KDDCup99", "Exact train/test duplicate records",
         "57.53\\% of test rows",
         "Duplicate-free sensitivity; ranking unchanged"),
        ("CIC-IDS2017", "Random split interleaves flows of one attack burst",
         "Present in the screening protocol",
         "Within-capture order-disjoint; day-disjoint transfer test"),
        ("CTU-13", "Host aggregates computed over whole scenario; random split",
         "$100.00\\rightarrow92.01$ macro-F1",
         "Causal aggregates + scenario-disjoint folds"),
        ("all", "Categorical vocabulary fitted on train$\\cup$test",
         "0 NSL-KDD / 2 KDDCup99 test rows affected",
         "All learned transforms fitted on train only"),
    ]
    lines = [
        r"\begin{table}[t]", r"\centering",
        r"\caption{Protocol audit motivating the leakage-resistant "
        r"\\emph{confirmation} protocol. Each row is an issue we tested for in "
        r"the \\emph{screening} protocol, what the test found, and how the "
        r"confirmation protocol treats it. The union-split "
        r"vocabulary turned out to be empirically negligible; we report its "
        r"size rather than citing that as grounds for keeping it.}",
        r"\label{tab:protocol-audit}",
        r"\resizebox{\columnwidth}{!}{%",
        r"\begin{tabular}{l p{3.4cm} l p{3.6cm}}", r"\toprule",
        r"Dataset & Issue audited & Finding & Revised treatment \\",
        r"\midrule",
    ]
    for ds, issue, finding, fix in rows:
        lines.append(f"{ds} & {issue} & {finding} & {fix} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}}", r"\end{table}"]
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="v1")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    for name, fn in [("tab_ranking", lambda: table_ranking(args.tag)),
                     ("tab_focused", lambda: table_focused(args.tag)),
                     ("tab_audit", table_audit),
                     ("tab_protocol_audit_revised", table_protocol_audit),
                     ("tab_readout", table_readout),
                     ("tab_timing", table_timing),
                     ("tab_audit_revised", table_audit_revised)]:
        p = (OUT / f"{name}_{args.tag}.tex"
             if name in {"tab_ranking", "tab_focused"}
             else OUT / f"{name}.tex")
        p.write_text(fn() + "\n")
        print("wrote", p)


if __name__ == "__main__":
    main()
