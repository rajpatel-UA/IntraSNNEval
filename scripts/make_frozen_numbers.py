#!/usr/bin/env python3
"""Emit `paper/sec_frozen_numbers.tex`: every manuscript number as a LaTeX macro.

The manuscript should never contain a typed number. It contains `\\FNlatRank`,
and this file defines it from the committed artifact. Re-running
`scripts/finalize.py` then this script updates every figure in the paper at once,
and a number that no analysis produces simply does not exist as a macro.

Generated 2026-08-13 from the closed experimental record: 235 runs, 5 seeds per
cell, zero failures.

Sign convention
---------------
Every signed value is emitted **bare**: ``-3.58``, not ``$-$3.58``. Call sites
wrap in ``$...$``, where a hyphen is typeset as a proper minus. Mixing the two
conventions is the failure that survives one paste and breaks on the next,
because a macro that already carries ``$-$`` produces ``$$-$3.58$`` when a
later author wraps it like its neighbours. One rule, applied everywhere:
**emit bare, wrap at use**.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
A = ROOT / "results/analysis"
OUT = ROOT / "paper/sec_frozen_numbers.tex"


_WORDS = {0: "zero", 1: "one", 2: "two", 3: "three", 4: "four", 5: "five",
          6: "six", 7: "seven", 8: "eight", 9: "nine", 10: "ten"}


def _word(n: int) -> str:
    """Small counts spelled out, because the prose around them is words."""
    return _WORDS.get(int(n), str(int(n)))


def _sci(v: float) -> str:
    """Scientific notation as LaTeX math, safe for an exact zero."""
    if v == 0:
        return "0"
    m, e = f"{v:.1e}".split("e")
    return rf"{m}\times10^{{{int(e)}}}"


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=OUT,
                    help="write elsewhere; used by check_macros.py to verify "
                         "that the committed file is what this script produces")
    ap.add_argument("--quiet", action="store_true")
    _args = ap.parse_args()
    L = []          # (macro, value, comment)

    # ---- 1. screening ----
    fr = json.load(open(A / "v1/friedman.json"))
    mr1 = pd.read_csv(A / "v1/mean_rank.csv", index_col=0)
    ft = json.load(open(A / "v1/focused_tests.json"))["tests"]
    lp_rate = next(t for t in ft if t["b"] == "LeakyParallel/rate")
    L += [
        ("ScrVariants", fr["k_variants"], "configurations in the screening sweep"),
        ("ScrBlocks", fr["n_blocks"], "dataset x seed blocks"),
        ("ScrRank", f"{fr['champion_mean_rank']:.2f}", "LP/latency mean rank"),
        ("ScrPos", fr["champion_rank_position"], "LP/latency rank position"),
        ("ScrChi", f"{fr['friedman_chi2']:.1f}", "Friedman chi2"),
        ("ScrP", f"{fr['friedman_p']:.1e}", "Friedman p"),
        ("ScrCD", f"{fr['nemenyi_critical_difference']:.2f}", "Nemenyi CD"),
        ("ScrWithinCD", len(fr["statistically_indistinguishable_from_best"]),
         "configurations within one CD of the leader"),
        ("ScrWinsA", lp_rate["wins_a"], "LP/latency wins vs LP/rate"),
        ("ScrWinsB", lp_rate["wins_b"], "LP/rate wins"),
        ("ScrMedDelta", f"{lp_rate['median_delta_pp']:+.2f}", "median delta pp"),
        ("ScrHolmP", f"{lp_rate['holm_adjusted_p']:.4f}", "Holm-adjusted p"),
        ("ScrBootLo", f"{lp_rate['bootstrap_ci_pp'][0]:+.2f}", "bootstrap CI low"),
        ("ScrBootHi", f"{lp_rate['bootstrap_ci_pp'][1]:+.2f}", "bootstrap CI high"),
    ]

    # ---- 2. strict axes ----
    for axis, pre in (("encoding", "Enc"), ("neuron", "Neu")):
        d = json.load(open(A / f"v2_{axis}_axis/summary.json"))
        m = pd.read_csv(A / f"v2_{axis}_axis/mean_rank.csv", index_col=0)
        L += [
            (f"{pre}Status", d["status"], f"{axis} axis completeness"),
            (f"{pre}Blocks", d["n_blocks"], "protocol x seed blocks"),
            (f"{pre}Levels", d["n_levels"], f"{axis}s compared"),
            (f"{pre}Rank", f"{d['champion_mean_rank']:.2f}", "champion mean rank"),
            (f"{pre}Pos", d["champion_rank_position"], "champion rank position"),
            (f"{pre}P", f"{d['friedman_p']:.1e}", "Friedman p"),
            (f"{pre}Wins", int(m.loc[d["champion"], "best_block_count"]),
             "blocks won by the champion"),
        ]
    mrn = pd.read_csv(A / "v2_neuron_axis/mean_rank.csv", index_col=0)
    L.append(("NeuRunnerUp", mrn.index[1], "second-ranked neuron"))
    L.append(("NeuRunnerUpRank", f"{mrn.mean_rank.iloc[1]:.2f}", "its mean rank"))
    L.append(("NeuMostWins", mrn.best_block_count.idxmax(),
              "neuron winning the most individual blocks"))

    # ---- 3. mechanism ----
    er = pd.read_csv(A / "early_readout/summary.csv")
    er = er[er.neuron == "LeakyParallel"]
    for proto, tag in [("nslkdd_v2", "Nsl"), ("kddcup99_v2", "Kdd"),
                       ("cicids2017_v2", "Cic"), ("ctu13_v2_f0", "CtuA"),
                       ("ctu13_v2_f1", "CtuB")]:
        g = er[er.protocol == proto].set_index("encoding")
        for enc, e in (("latency", "Lat"), ("rate", "Rat")):
            if enc in g.index:
                # Letter-only names: a LaTeX control sequence cannot contain
                # digits, so "T99..." would not compile.
                L.append((f"TNN{tag}{e}", int(g.loc[enc, "T99_median"]),
                          f"median T99, {proto} {enc}"))
                # Methodology defines both thresholds, so both are available.
                L.append((f"TNF{tag}{e}", int(g.loc[enc, "T95_median"]),
                          f"median T95, {proto} {enc}"))
    nsl = er[er.protocol == "nslkdd_v2"].set_index("encoding")
    L += [("TNNNslLatMin", int(nsl.loc["latency", "T99_min"]), "range low"),
          ("TNNNslLatMax", int(nsl.loc["latency", "T99_max"]), "range high"),
          ("SpikeSaveNslLat", f"{nsl.loc['latency','spike_saving_at_T99_pct']:.1f}",
           "percent output spikes saved at T99"),
          ("SpikeSaveNslRat", f"{nsl.loc['rate','spike_saving_at_T99_pct']:.1f}",
           "same for rate")]

    eff = pd.read_csv(A / "efficiency/summary.csv")
    eff = eff[eff.neuron == "LeakyParallel"]
    piv = lambda c: eff.pivot(index="protocol", columns="encoding", values=c)
    for name, col in (("Sop", "sops_mean"), ("Inp", "input_spikes"),
                      ("Tot", "total_spikes")):
        r = (piv(col)["rate"] / piv(col)["latency"])
        L += [(f"{name}RatioMin", f"{r.min():.1f}", f"min rate/latency {col}"),
              (f"{name}RatioMax", f"{r.max():.1f}", f"max rate/latency {col}")]

    # ---- 4. protocol robustness ----
    dm = pd.read_csv(A / "dual_macro_f1.csv")
    cic = dm[dm.protocol == "cicids2017_v2"].set_index("variant")
    for v, tag in (("LeakyParallel/latency", "Lat"), ("LeakyParallel/rate", "Rat"),
                   ("LeakyParallel/delta", "Del")):
        L += [(f"CicFixed{tag}", f"{cic.loc[v,'fixed']:.4f}", "fixed-universe macro-F1"),
              (f"CicPresent{tag}", f"{cic.loc[v,'present']:.4f}", "test-present macro-F1")]
    L.append(("CicCeiling", f"{cic.ceiling.iloc[0]:.2f}", "macro-F1 ceiling"))

    sa = pd.read_csv(A / "split_audit.csv").set_index("registry_name")
    L += [("DupKdd", f"{sa.loc['kddcup99_v2','dup_test_pct']:.2f}",
           "percent KDDCup99 test rows duplicated in train"),
          ("DupNsl", f"{sa.loc['nslkdd_v2','dup_test_pct']:.2f}", "same, NSL-KDD")]

    ded = json.load(open(A / "dedup/encoding_axis_lp/verdict.json"))["per_protocol"]
    unchanged = sum(v["leader_unchanged"] for v in ded.values())
    L += [("DedupProtocols", len(ded), "protocols with duplicates, encoding axis"),
          ("DedupLeaderUnchanged", unchanged, "of which leader unchanged"),
          # MEDIAN paired difference. Named as such because the difference of
          # means is +2.21 and +2.25 on the same five blocks: with values
          # -1.02, +1.04, +5.46, +6.58, -1.03 the two statistics differ by
          # more than a factor of two, and a name that does not say which
          # invites exactly the audit failure it has already caused.
          ("DedupKddOffMedian",
           f"{ded['kddcup99_v2']['paired_official']['median_delta_pp']:+.3f}",
           "LP lat-vs-rate MEDIAN paired margin, official"),
          ("DedupKddDedMedian",
           f"{ded['kddcup99_v2']['paired_dedup']['median_delta_pp']:+.3f}",
           "same, deduplicated")]
    # The absolute effect of deduplication on the recommended configuration.
    # Threats quoted -0.79 pp, which is the value for rate; the recommended
    # configuration is latency and moves by a different amount.
    _dd = pd.read_csv(A / "dedup/encoding_axis_lp/summary.csv")
    _dd = _dd[_dd.protocol == "kddcup99_v2"].set_index("variant")
    # The KDDCup99 cell's per-seed spread. Quoted because the median alone
    # reads as a stable small effect, and the underlying blocks are not: two of
    # the five are negative and the range is seven points wide.
    _ps = pd.read_csv(A / "dedup/encoding_axis_lp/per_seed.csv")
    _ps = _ps[_ps.protocol == "kddcup99_v2"].pivot_table(
        index="seed", columns="variant", values="f1_official")
    _sd = 100.0 * (_ps["LeakyParallel/latency"] - _ps["LeakyParallel/rate"])
    L += [("DedupKddSeedLo", f"{_sd.min():+.2f}",
           "most negative per-seed latency-rate difference, KDDCup99, pp"),
          ("DedupKddSeedHi", f"{_sd.max():+.2f}", "most positive, pp"),
          ("DedupKddSeedWins", int((_sd > 0).sum()),
           "seed-blocks where latency leads on KDDCup99"),
          ("DedupKddSeedN", int(len(_sd)), "seed-blocks on KDDCup99")]

    L += [("DedupKddOffMean",
           f"{100 * (_dd.loc['LeakyParallel/latency', 'f1_official'] - _dd.loc['LeakyParallel/rate', 'f1_official']):+.2f}",
           "the same margin as a difference of means, official"),
          ("DedupKddDedMean",
           f"{100 * (_dd.loc['LeakyParallel/latency', 'f1_dedup'] - _dd.loc['LeakyParallel/rate', 'f1_dedup']):+.2f}",
           "same, deduplicated"),
          ("DedupKddLpLat", f"{_dd.loc['LeakyParallel/latency','delta_pp']:+.2f}",
           "KDDCup99 dedup effect on LeakyParallel/latency, pp"),
          ("DedupKddLpRat", f"{_dd.loc['LeakyParallel/rate','delta_pp']:+.2f}",
           "same for LeakyParallel/rate, pp")]

    # ---- 5. boundary ----
    far = pd.read_csv(A / "security/fixed_far.csv")
    far = far[(far.neuron == "LeakyParallel") & (far.encoding == "latency")
              & (far.target_far == 0.01)].set_index("protocol")
    for proto, tag in (("nslkdd_v2", "Nsl"), ("kddcup99_v2", "Kdd"),
                       ("cicids2017_v2", "Cic"), ("ctu13_v2_f0", "CtuA"),
                       ("ctu13_v2_f1", "CtuB")):
        if proto in far.index:
            L += [(f"FarVal{tag}", f"{100*far.loc[proto,'val_far_mean']:.1f}",
                   "realised validation FAR at a 1 percent target"),
                  (f"FarTest{tag}", f"{100*far.loc[proto,'test_far_mean']:.1f}",
                   "realised test FAR")]

    ku = pd.read_csv(A / "known_unseen/known_unseen.csv")
    kn = ku[ku.regime == "known"].DR_mean
    un = ku[ku.regime == "unseen"].DR_mean
    _ku_n = (ku.neuron + "/" + ku.encoding).nunique()
    _rows = ku.groupby("regime").n.first()       # same test rows for every config
    L += [("KnownLo", f"{kn.min():.2f}", "lowest known-type DR"),
          ("KnownHi", f"{kn.max():.2f}", "highest known-type DR"),
          ("UnseenLo", f"{un.min():.2f}", "lowest unseen-type DR"),
          ("UnseenHi", f"{un.max():.2f}", "highest unseen-type DR"),
          ("UnseenFrac", f"{100 * _rows['unseen'] / _rows.sum():.1f}",
           "percent of KDDTest+ attack rows of unseen type"),
          # State the population size: this used to be an 11-configuration
          # cross-shaped subset, and "every configuration" was an overclaim
          # until E2a completed the NSL-KDD sweep.
          ("UnseenConfigs", f"{_ku_n}",
           "configurations entering the known-vs-unseen comparison")]

    # Literals the wording guard flagged in prose: each must be a macro so a
    # corrected analysis cannot leave a stale value behind in the text.
    leaky = next((t for t in ft if t["b"] == "Leaky/latency"), None)
    if leaky:
        L.append(("ScrLeakyP", f"{leaky['wilcoxon_p']:.2f}",
                  "LP/latency vs Leaky/latency, Wilcoxon p"))
    enc = json.load(open(A / "v1/encoding_tests.json"))["pairwise"]
    lr = enc.get("latency_vs_rate") or enc.get("rate_vs_latency")
    L += [("ScrEncP", f"{lr['holm_adjusted_p']:.3f}",
           "encoding-level latency vs rate, Holm p"),
          ("ScrEncCiLo", f"{lr['bootstrap_ci_pp'][0]:+.2f}", "its CI low"),
          ("ScrEncCiHi", f"{lr['bootstrap_ci_pp'][1]:+.2f}", "its CI high")]

    sw = pd.read_csv(ROOT / "results/v1_submitted/sweep_540.csv")
    k = sw[(sw.dataset == "kddcup99") & (sw.neuron == "LeakyParallel")]
    acc = k.groupby("encoding").accuracy.mean()
    f1k = k.groupby("encoding").f1_macro.mean()
    L += [("KddAccRat", f"{acc['rate']:.4f}", "KDDCup99 accuracy, LP/rate"),
          ("KddAccLat", f"{acc['latency']:.4f}", "same, LP/latency"),
          ("KddFOneGain", f"{100*(f1k['latency']-f1k['rate']):.3f}",
           "LP latency-minus-rate macro-F1 pp on KDDCup99")]

    ctu = pd.read_csv(ROOT / "results/v1_submitted/ctu13_strict_vanilla.csv")
    cs = (ctu[(ctu.model == "vanilla") & (ctu.condition == "causal_scenario")]
          .groupby("fold").macro_f1.mean().sort_values(ascending=False))
    for i, v in enumerate(cs.tolist()):
        L.append((f"CtuFold{'ABCD'[i]}", f"{v:.2f}",
                  f"CTU-13 scenario-disjoint fold value {i+1} of 4, descending"))

    dsum = pd.read_csv(A / "early_readout/summary.csv")
    dn = dsum[(dsum.protocol == "nslkdd_v2") & (dsum.encoding == "delta")]
    L.append(("DeltaDeficitNsl", f"{abs(dn.f1_vs_best_pp.iloc[0]):.1f}",
              "delta terminal deficit vs best on NSL-KDD, pp"))

    hid = eff.pivot(index="protocol", columns="encoding", values="hidden_spikes")
    hr = hid["rate"] / hid["latency"]
    L += [("HidRatioMin", f"{hr.min():.2f}", "min rate/latency hidden spikes"),
          ("HidRatioMax", f"{hr.max():.2f}", "max rate/latency hidden spikes")]

    # Input/hidden shares, computed from the stored per-layer fanouts rather
    # than an assumed pair. CTU-13 is binary, so its readout fanout is 2 and
    # not 5; an earlier ad-hoc calculation used 5 everywhere and understated
    # latency's input share on those protocols.
    import numpy as _np
    dec = pd.read_csv(A / "spike_sop_decomposition.csv") if (
        A / "spike_sop_decomposition.csv").exists() else None
    if dec is not None:
        for enc, tag in (("latency", "Lat"), ("rate", "Rat")):
            g = dec[dec.encoding == enc]
            L += [(f"InShareSop{tag}Min", f"{g.input_share_sops_pct.min():.1f}",
                   f"min input share of SOPs, {enc}"),
                  (f"InShareSop{tag}Max", f"{g.input_share_sops_pct.max():.1f}",
                   f"max input share of SOPs, {enc}")]
        gl = dec[dec.encoding == "latency"]
        L += [("HidShareLatMin", f"{gl.hidden_share_pct.min():.1f}",
               "min hidden share of latency spikes"),
              ("HidShareLatMax", f"{gl.hidden_share_pct.max():.1f}",
               "max hidden share of latency spikes")]
    # Encoder gates are read from the encoders' own defaults, so the manuscript
    # cannot state a gate the code does not apply.
    import inspect as _inspect
    from src.encoding.delta import delta_encode as _delta_enc
    from src.encoding.latency import latency_encode as _lat_enc
    _gate = lambda f: _inspect.signature(f).parameters["threshold"].default
    _lp = (pd.read_csv(A / "spike_sop_decomposition.csv")
           .set_index(["protocol", "encoding"]))
    L += [("LatThresh", f"{_gate(_lat_enc):g}", "latency encoder threshold"),
          ("DeltaThresh", f"{_gate(_delta_enc):g}", "delta encoder threshold"),
          ("NslFeat", int(_lp.loc[("nslkdd_v2", "latency"), "n_features"]),
           "NSL-KDD v2 feature dimensionality"),
          ("NslLatInSpk", f"{_lp.loc[('nslkdd_v2', 'latency'), 'input_spikes']:.2f}",
           "NSL-KDD latency input spikes per sample")]

    # Feature sparsity and the closed-form input-spike ratio. These turn the
    # efficiency claim from a measured range into a prediction with an error
    # bar, so every component of it must be generated rather than typed.
    sp = pd.read_csv(A / "feature_sparsity.csv").set_index("protocol")
    xb = pd.read_csv(A / "xbar_active_prediction.csv").set_index("protocol")
    _gb = pd.read_csv(A / "appendix/sparsity_per_protocol.csv").set_index("protocol")
    tags = {"nslkdd_v2": "Nsl", "kddcup99_v2": "Kdd", "cicids2017_v2": "Cic",
            "ctu13_v2_f0": "CtuA", "ctu13_v2_f1": "CtuB"}
    for proto, tag in tags.items():
        L += [(f"Zero{tag}", f"{sp.loc[proto,'zero_per_sample']:.2f}",
               f"features exactly zero, {proto}"),
              (f"ZeroPct{tag}", f"{sp.loc[proto,'zero_pct']:.1f}",
               f"percent exactly zero, {proto}"),
              (f"Band{tag}", f"{sp.loc[proto,'band_per_sample']:.4f}",
               f"features in (0, theta_lat], {proto}"),
              # The interval BETWEEN the gates, which is the one that actually
              # separates the two codes. It is not in feature_sparsity.csv;
              # appendix_data.py recovers it from the E1 arms. Distinguished
              # from Band* by name because the two differ by more than an order
              # of magnitude on NSL-KDD and confusing them understates the
              # separating mass to zero.
              (f"GateBand{tag}", f"{_gb.loc[proto,'gate_band_count']:.3f}",
               f"features in (theta_lat, theta_delta], {proto}"),
              (f"GateBandPct{tag}", f"{_gb.loc[proto,'gate_band_pct_of_active']:.2f}",
               f"that band as a percent of active features, {proto}"),
              (f"XbarAct{tag}", f"{xb.loc[proto,'xbar_active_gt0']:.3f}",
               f"mean value of active features, {proto}"),
              (f"Dim{tag}", int(sp.loc[proto, "d"]), f"feature dimension, {proto}")]
    for proto, tag in tags.items():
        L.append((f"PredErr{tag}", f"{xb.loc[proto,'abs_err_pct']:.4f}",
                  f"prediction error, {proto}"))
    # Rounded up for the abstract: a headline quoted at 0.003% invites a hunt
    # for the one cell where it does not hold, and buys nothing.
    # Delta's per-timestep input profile (scripts/data_request.py), which also
    # asserts the one-step reduction on every feature-sample pair.
    _c2 = np.load(A / "data_request/C2_delta_spikes_per_t.npy")
    _nsl_n = int(pd.read_csv(A / "data_request/C1_per_class.csv")
                 .query("protocol == 'nslkdd_v2'").support.sum())
    _scr = json.loads((ROOT / "results/v1_submitted/runs/nslkdd/SNN_LeakyParallel"
                       "/latency/seed_42/results.json").read_text())
    L += [("ZeroPctMin", f"{sp.zero_pct.min():.1f}", "min percent exactly zero"),
          ("ZeroPctMax", f"{sp.zero_pct.max():.1f}", "max percent exactly zero"),
          ("PredErrMax", f"{xb.abs_err_pct.max():.3f}",
           "max percent error of the a-priori input-ratio prediction"),
          ("PredErrMean", f"{xb.abs_err_pct.mean():.3f}", "mean percent error"),
          ("DeltaSpkNsl", f"{_c2[0]:.3f}", "delta input spikes at t=1, NSL-KDD"),
          ("DeltaSpkAfter", f"{_c2[1:].max():.1f}", "delta input spikes at every t>1"),
          ("NslTestN", f"{_nsl_n:,}".replace(",", "{,}"),
           "NSL-KDD confirmation test samples"),
          ("DimNslScreen", int(_scr["data"]["num_features"]),
           "NSL-KDD dimension under the screening protocol")]

    # KDDCup99 rate-vs-delta: a 0.21 pp mean margin with per-seed differences
    # of both signs. The protocol-level rank for that pair must not be over-read.
    _k = _sw0 = pd.read_csv(ROOT / "results/v1_submitted/sweep_540.csv")
    _k = _k[(_k.dataset == "kddcup99") & (_k.neuron == "LeakyParallel")
            & (_k.encoding.isin(["rate", "delta"]))]
    _p = _k.pivot(index="seed", columns="encoding", values="f1_macro")
    _d = _p["rate"] - _p["delta"]
    L += [("KddRateDeltaMarginScr", f"{100*_d.mean():.2f}",
           "KDDCup99 rate-minus-delta mean margin, pp"),
          ("KddRateWinsScr", int((_d > 0).sum()), "seed-blocks where rate leads"),
          ("KddDeltaWinsScr", int((_d < 0).sum()), "seed-blocks where delta leads"),
          ("KddRateDeltaLoScr", f"{100*_d.min():+.1f}", "most negative per-seed diff, pp"),
          ("KddRateDeltaHiScr", f"{100*_d.max():+.1f}", "most positive per-seed diff, pp")]
    # The same contrast under the CONFIRMATION protocol, so the two can be
    # told apart by name. They differ by more than a factor of three (0.21
    # against 0.06) and an audit that assumes one file holds one quantity will
    # read the mismatch as an error rather than as two populations.
    _kc = pd.read_csv(A / "dedup/encoding_axis_lp/summary.csv")
    _kc = _kc[_kc.protocol == "kddcup99_v2"].set_index("variant")
    L.append(("KddRateDeltaMarginConf",
              f"{100 * (_kc.loc['LeakyParallel/rate', 'f1_official'] - _kc.loc['LeakyParallel/delta', 'f1_official']):+.2f}",
              "KDDCup99 rate-minus-delta margin under the confirmation protocol, pp"))

    # NSL-KDD headline operating point and the two rare-class recalls, so the
    # summary can say plainly what the recommended configuration achieves.
    _sr = pd.read_csv(A / "strict_results.csv")
    _n = _sr[(_sr.axis == "encoding") & (_sr.protocol == "nslkdd_v2")
             & (_sr.variant == "LeakyParallel/latency")].iloc[0]
    L += [("NslDr", f"{_n.dr_mean:.2f}", "NSL-KDD detection rate, argmax rule"),
          ("NslFar", f"{100*_n.far_mean:.1f}", "NSL-KDD false-alarm rate, percent")]
    _pc = pd.read_csv(A / "data_request/C1_per_class.csv")
    _pc = _pc[_pc.protocol == "nslkdd_v2"].set_index("class")
    L += [("NslRtlRec", f"{_pc.loc['r2l','recall']:.3f}", "NSL-KDD R2L recall"),
          ("NslUtrRec", f"{_pc.loc['u2r','recall']:.3f}", "NSL-KDD U2R recall"),
          ("NslMacro", f"{_n.macro_f1_mean:.4f}", "NSL-KDD macro-F1")]

    # Timing isolation (E1 arm C). Delta at latency's gate fires on exactly the
    # same features, so the two arms emit identical input spike sets and differ
    # only in whether timing is used. Every number here is a difference at
    # matched input activity.
    ti = pd.read_csv(A / "e1_timing_isolation.csv", index_col=0)
    tags = {"nslkdd_v2": "Nsl", "kddcup99_v2": "Kdd", "cicids2017_v2": "Cic",
            "ctu13_v2_f0": "CtuA", "ctu13_v2_f1": "CtuB"}
    for proto, tag in tags.items():
        L.append((f"Timing{tag}", f"{ti.loc[proto,'timing_premium_pp']:+.1f}",
                  f"timing premium at matched input spikes, {proto}"))
    L += [("TimingMin", f"{ti.timing_premium_pp.min():+.1f}", "smallest timing premium"),
          ("TimingMax", f"{ti.timing_premium_pp.max():+.1f}", "largest timing premium"),
          ("TimingSpread", f"{ti.timing_premium_pp.max()-ti.timing_premium_pp.min():.1f}",
           "spread of the timing premium, pp"),
          ("TimingMaxDiff", f"{ti.in_spk_abs_diff.abs().max():.0e}",
           "max input-spike difference between the matched arms")]

    # Confirmation-protocol omnibus over the full design space: 27
    # configurations x 25 (protocol x seed) blocks, as protocol_shift.py saved
    # it. Read from that committed file, not recomputed from the runs: without
    # the .npz artifacts the recomputation came back empty and these five
    # macros silently vanished. A missing file now stops the build. The CD uses
    # the same helper rank_analysis.py uses for the screening figure, so
    # \FNConfCD and \FNScrCD are comparable rather than two conventions --- the
    # two differ only because N rises from 20 to 25.
    import sys as _sys
    _sys.path.insert(0, str(ROOT / "scripts"))
    from rank_analysis import nemenyi_cd as _ncd                 # noqa: E402

    _om = json.loads((A / "protocol_shift/omnibus.json").read_text())
    _n, _k = _om["n_blocks"], _om["n_configurations"]
    L += [("ConfBlocks", _n, "confirmation protocol x seed blocks"),
          ("ConfVariants", _k, "configurations in the confirmation sweep"),
          ("ConfChi", f"{_om['friedman_chi2']:.1f}", "Friedman chi2, confirmation"),
          ("ConfP", f"{_om['friedman_p']:.1e}", "Friedman p, confirmation"),
          ("ConfCD", f"{_ncd(_k, _n):.2f}", "Nemenyi CD, confirmation")]

    # Protocol displacement: the study's own question as one number.
    _tau = json.load(open(A / "protocol_shift/tau.json"))
    _np = _tau["n_permutations"]
    # A permutation test cannot report p = 0. With N permutations the smallest
    # resolvable value is 1/N, so the honest statement is an upper bound; a
    # literal 0.0000 in the manuscript would be a claim the procedure cannot
    # make.
    # Plain decimal, not 10^{-4}: the manuscript writes "$p\,\FNTauP$" inside
    # prose that reads "permutation p < 0.0001", and a superscript there would
    # not match the surrounding sentence.
    _pv = (f"<{1.0 / _np:.4f}" if _tau["permutation_p"] < 1.0 / _np
           else f"{_tau['permutation_p']:.4f}")
    L += [("Tau", f"{_tau['kendall_tau']:.4f}",
           "Kendall tau, screening vs confirmation ranking"),
          ("TauP", _pv, "permutation p for that tau (bounded by 1/N)"),
          ("TauPerms", f"{_np:,}".replace(",", "{,}"),
           "permutations in the tau test"),
          ("TauSeed", str(_tau["rng_seed"]), "RNG seed, registered"),
          ("TauConfigs", _tau["n_configurations"],
           "configurations ranked under both protocols")]

    # Arm-A2 decomposition (Table VI). Delta_map and Delta_spread are emitted at
    # TWO decimals, and a two-decimal Delta_time is emitted alongside them,
    # because the table's last three columns must visibly sum. At the one
    # decimal \FNTiming* uses in prose they do not: NSL-KDD reads
    # 7.0 + (-5.9) = 1.1 against a printed 1.2. Use \FNDtime* inside the table
    # and \FNTiming* in running text.
    _a2 = pd.read_csv(A / "a2/decomposition.csv").set_index("protocol")
    for proto, tag in tags.items():
        if proto not in _a2.index:
            continue
        r = _a2.loc[proto]
        L += [(f"ArmLat{tag}", f"{r.A1_latency:.4f}",
               f"A1 latency@0.01 macro-F1, 3 seeds, {proto}"),
              (f"ArmPerm{tag}", f"{r.A2_permuted:.4f}",
               f"A2 permuted-time macro-F1, {proto}"),
              (f"ArmDelta{tag}", f"{r.A3_delta:.4f}",
               f"A3 delta@0.01 macro-F1, {proto}"),
              (f"Dmap{tag}", f"{r.d_map_pp:+.2f}",
               f"value of the amplitude-to-time mapping, pp, {proto}"),
              (f"Dspread{tag}", f"{r.d_spread_pp:+.2f}",
               f"value of temporal spreading, pp, {proto}"),
              (f"Dtime{tag}", f"{r.d_time_pp:+.2f}",
               f"timing premium at two decimals so the row sums, {proto}")]
    L += [("DmapMin", f"{_a2.d_map_pp.min():+.2f}", "smallest mapping term"),
          ("DmapMax", f"{_a2.d_map_pp.max():+.2f}", "largest mapping term"),
          ("DspreadMin", f"{_a2.d_spread_pp.min():+.2f}", "smallest spreading term"),
          ("DspreadMax", f"{_a2.d_spread_pp.max():+.2f}", "largest spreading term"),
          # Read from closure.json, not the CSV column: the residual is
          # ~1e-15 and an eight-decimal CSV rounds it to exactly zero.
          ("DclosureErr",
           _sci(json.load(open(A / "a2/closure.json"))
                ["max_abs_closure_error_pp"]),
           "max closure error of the decomposition, pp")]

    # ---- per-class cells, every protocol x class x encoding ----
    # Named FN<Encoding><Protocol><Class>, with a matching ...Sd, so a table
    # cell and its dispersion always travel together. Emitted for the whole
    # grid rather than only for the cells currently cited: a per-class number
    # typed into a table because no macro existed is how the CTU-13 delta cells
    # came to be printed at three decimals while the rest of their column was
    # at four.
    _pc = pd.read_csv(A / "per_class/per_class.csv")
    _cls = {"normal": "Normal", "dos": "Dos", "probe": "Probe", "r2l": "Rtl",
            "u2r": "Utr", "botnet": "Botnet"}
    _enc = {"latency": "Lat", "rate": "Rate", "delta": "Delta"}
    for _, r in _pc.iterrows():
        if r.protocol not in tags or r["name"] not in _cls:
            continue
        nm = f"{_enc[r.encoding]}{tags[r.protocol]}{_cls[r['name']]}"
        L.append((nm, f"{r.f1:.4f}",
                  f"F1, {r.protocol} {r['name']} {r.encoding}"))
        if pd.notna(r.sd):
            L.append((f"{nm}Sd", f"{r.sd:.4f}", "its across-seed SD"))
    for (proto, cname), gsup in _pc.groupby(["protocol", "name"]):
        if proto in tags and cname in _cls:
            L.append((f"Sup{tags[proto]}{_cls[cname]}",
                      f"{int(gsup.support.iloc[0]):,}".replace(",", "{,}"),
                      f"test support, {proto} {cname}"))

    # Per-seed spread of the KDDCup99 DoS difference: the evidence for calling
    # that class tied rather than won.
    _psd = pd.read_csv(A / "per_class/per_class_per_seed.csv")
    _kd = _psd[(_psd.protocol == "kddcup99_v2") & (_psd["name"] == "dos")]
    _kw = _kd.pivot_table(index="seed", columns="encoding", values="f1")
    _kdiff = 100 * (_kw["latency"] - _kw["rate"])
    L += [("KddDosRangeLo", f"{_kdiff.min():+.3f}",
           "most negative per-seed DoS difference, pp"),
          ("KddDosRangeHi", f"{_kdiff.max():+.3f}", "most positive, pp"),
          ("KddDosLatWins", int((_kdiff > 0).sum()),
           "seeds where latency leads on KDDCup99 DoS"),
          ("KddDosSeeds", int(len(_kdiff)), "seeds compared")]

    # A2 time-budget sweep: the components against T.
    _ts = pd.read_csv(A / "a2t/components_vs_T.csv")
    _tj = json.load(open(A / "a2t/summary.json"))
    _tg = pd.read_csv(A / "a2t/degeneracy.csv")
    _tc = pd.read_csv(A / "a2t/cells.csv")
    L += [("TsweepCells", _tj["n_cells"], "protocol x T cells in the A2 sweep"),
          ("TsweepRuns", _tj["n_runs_new"], "new runs in the A2 sweep"),
          ("MapMovesN", _tj["map_moves"],
           "protocols where Delta_map moves with T beyond its own SE"),
          ("MapFlatN", 5 - _tj["map_moves"], "protocols where it is flat"),
          ("SpreadMovesN", _tj["spread_moves"],
           "protocols where Delta_spread moves with T"),
          ("SpreadFlatN", 5 - _tj["spread_moves"], "protocols where it is flat")]
    for proto, tag in tags.items():
        _q = _ts[_ts.protocol == proto]
        if len(_q):
            # Means over T, computed from the components rather than from the
            # printed cells: averaging rounded table entries is how a CTU-13 f1
            # mean of -3.21 reads as -3.22.
            L += [(f"MapMean{tag}", f"{_q.d_map_pp.mean():+.2f}",
                   f"mean Delta_map over T, {proto}"),
                  (f"SpreadMean{tag}", f"{_q.d_spread_pp.mean():+.2f}",
                   f"mean Delta_spread over T, {proto}"),
                  (f"DtimeMean{tag}", f"{_q.d_time_pp.mean():+.2f}",
                   f"mean Delta_time over T, {proto}")]
        if proto in _tj["rho"]:
            L.append((f"Rho{tag}",
                      f"{_tj['rho'][proto]:+.2f}",
                      f"corr(Delta_map, Delta_spread) across T, {proto}"))
    # The CTU-13 f1 permuted arm against T, and the diagnostics that show its
    # low-T collapse is NOT a tie-density artefact.
    _a2c = _tc[_tc.arm == "A2"].set_index(["protocol", "T"]).f1
    for t, word in ((5, "Tfive"), (10, "Tten"), (25, "Ttwentyfive")):
        L.append((f"CtuBAtwo{word}",
                  # Four decimals to match the arm values it is quoted beside.
                  f"{_a2c.loc[('ctu13_v2_f1', t)]:.4f}",
                  f"A2 macro-F1 on CTU-13 f1 at T={t}"))
    _g5 = _tg[(_tg.protocol == "ctu13_v2_f1") & (_tg["T"] == 5)].iloc[0]
    L += [("CtuBTieDensity", f"{_g5.feat_per_bin:.2f}",
           "active features per distinguishable spike time, CTU-13 f1, T=5"),
          ("CtuBIdenticalFrac", f"{_g5.identical_frac:.2f}",
           "share of active features keeping their own time under permutation"),
          ("CtuBSpearmanAtwo", f"{_g5.spearman_A2:+.3f}",
           "amplitude-time rank correlation of the permuted arm there"),
          ("KddIdenticalFrac",
           f"{_tg[(_tg.protocol == 'kddcup99_v2') & (_tg['T'] == 25)].identical_frac.iloc[0]:.2f}",
           "same share on KDDCup99, the protocol where ties really do bind")]

    # Smallest attainable two-sided exact signed-rank p at this many blocks.
    # Computed, not typed: it is 2/2^n, and deriving it from the seed count
    # means the claim follows automatically if the design ever changes. The
    # observed minimum across the per-class tests is reported beside it, so the
    # sentence is about what happened and not only about what was possible.
    from scipy.stats import wilcoxon as _wil
    # From the PER-SEED file: per_class.csv is already aggregated to one row
    # per cell, so counting rows there gives n=1 and a floor of 1.0000 -- a
    # wrong number that looks like a real one. The assertion below is what
    # makes that failure loud instead of silent.
    _pcs = pd.read_csv(A / "per_class/per_class_per_seed.csv")
    _n = int(_pcs.groupby(["protocol", "encoding", "cls"]).seed.nunique().max())
    if _n < 3:
        raise SystemExit(
            f"paired-block count came out as {_n}; the signed-rank floor "
            "would be meaningless. Check that the per-seed file is being read.")
    _floor = _wil(list(range(1, _n + 1)), [0] * _n).pvalue
    L += [("WilcoxonFloor", f"{_floor:.4f}",
           f"smallest attainable two-sided exact signed-rank p at n={_n}"),
          ("WilcoxonBlocks", _word(_n), "paired blocks per class test")]

    # ---- values that lived only in a preamble fallback ----
    # Derived here wherever the artifact exists. A fallback in a preamble is a
    # number with no provenance that silently wins when the generator stops
    # emitting; moving them here puts them inside the byte check.
    _ctu = pd.read_csv(ROOT / "results/v1_submitted/ctu13_strict_vanilla.csv")
    _cm = _ctu.groupby("condition").macro_f1.mean()
    sys.path.insert(0, str(ROOT / "scripts"))
    from spike_sop_decomposition import SOP_TOLERANCE
    _e1n = pd.read_csv(A / "e1_threshold_ablation.csv").n
    if _e1n.nunique() != 1:
        raise SystemExit("E1 cells differ in seed count; \\FNEoneSeeds would "
                         "misstate some of them")
    L += [("EoneSeeds", _word(int(_e1n.iloc[0])),
           "seeds per cell in the matched-input arms"),
          ("SopResidual", f"10^{{{round(np.log10(SOP_TOLERANCE))}}}",
           "tolerance of the SOP identity check"),
          ("CtuCausalCost",
           f"{_cm['legacy_random'] - _cm['causal_random']:.2f}",
           "pp lost to causal aggregation alone"),
          ("CtuScenarioCost",
           f"{_cm['causal_random'] - _cm['causal_scenario']:.2f}",
           "pp lost to the scenario-disjoint split"),
          ("CtuTotalCost",
           f"{_cm['legacy_random'] - _cm['causal_scenario']:.2f}",
           "pp from the submitted protocol to the strict one")]

    # Configurations saturating a CTU-13 block, which is why that dataset
    # cannot separate the design space.
    _sw2 = pd.read_csv(ROOT / "results/v1_submitted/sweep_540.csv")
    _sw2 = _sw2[_sw2.dataset.str.contains("ctu", case=False)]
    if len(_sw2):
        _tied = (_sw2[_sw2.f1_macro >= 0.9999]
                 .groupby("seed").size())
        L += [("CtuTiedLo", int(_tied.min()), "fewest saturating configs in a "
               "CTU-13 block"),
              ("CtuTiedHi", int(_tied.max()), "most")]

    # Protocol-level ranks: the protocol, not the seed, as the unit.
    _pe = pd.read_csv(A / "protocol_level/encoding_mean_ranks.csv").set_index("level")
    _pn = pd.read_csv(A / "protocol_level/neuron_mean_ranks.csv").set_index("level")
    for lvl, tag in (("latency", "Lat"), ("rate", "Rate"), ("delta", "Delta")):
        L.append((f"EncProto{tag}Rank",
                  f"{_pe.loc[lvl, 'mean_rank_protocol_level']:.2f}",
                  f"protocol-level mean rank, {lvl}"))
    for lvl, tag in (("LeakyParallel", "LP"), ("Synaptic", "Syn")):
        if lvl in _pn.index:
            L.append((f"NeuProto{tag}Rank",
                      f"{_pn.loc[lvl, 'mean_rank_protocol_level']:.2f}",
                      f"protocol-level mean rank, {lvl}"))

    # Confirmation ranking headline figures.
    _rc = pd.read_csv(A / "protocol_shift/rank_comparison.csv")
    _sel = _rc[_rc.variant == "LeakyParallel/latency"].iloc[0]
    _enc_agg = (_rc.assign(enc=_rc.variant.str.split("/").str[1])
                   .groupby("enc").rank_confirmation.mean())
    _scr_agg = (_rc.assign(enc=_rc.variant.str.split("/").str[1])
                   .groupby("enc").rank_screening.mean())
    _ea = pd.read_csv(A / "exception_audit/blocks.csv")
    _wins = _ea.winning_encoding.value_counts()
    L += [("ConfRank", f"{_sel.rank_confirmation:.2f}",
           "mean rank of the selected configuration, confirmation"),
          ("ConfMove", int(_sel.movement), "its rank-position movement"),
          ("ConfLatTop", _word(int((_rc.nsmallest(4, "rank_confirmation")
                                    .variant.str.endswith("/latency")).sum())),
           "latency configurations in the leading four"),
          ("ConfEncLat", f"{_enc_agg['latency']:.2f}", "encoding aggregation, latency"),
          ("ConfEncRat", f"{_enc_agg['rate']:.2f}", "same, rate"),
          ("ConfEncDel", f"{_enc_agg['delta']:.2f}", "same, delta"),
          ("ConfEncWinsLat", int(_wins.get("latency", 0)), "blocks won by latency"),
          ("ConfEncWinsRat", int(_wins.get("rate", 0)), "by rate"),
          ("ConfEncWinsDel", int(_wins.get("delta", 0)), "by delta"),
          ("ConfMarginScr", f"{_scr_agg['rate'] - _scr_agg['latency']:.2f}",
           "latency-rate gap under screening, rank units"),
          ("ConfMarginConf", f"{_enc_agg['rate'] - _enc_agg['latency']:.2f}",
           "same under confirmation")]

    # A2 decomposition spread and the T-sweep envelope.
    L += [("ATwoPredPass",
           f"{100 * _a2.loc['ctu13_v2_f0', 'd_map_pp'] / _a2.loc['ctu13_v2_f0', 'd_time_pp']:.1f}",
           "share of the CTU-13 f0 premium carried by the mapping, percent"),
          ("DmapRange", f"{_a2.d_map_pp.max() - _a2.d_map_pp.min():.2f}",
           "spread of Delta_map across protocols at T=25, pp"),
          ("DspreadRange", f"{_a2.d_spread_pp.max() - _a2.d_spread_pp.min():.2f}",
           "same for Delta_spread")]
    _tsn = (pd.read_csv(A / "a2t/runs.csv")
            .groupby(["arm", "T", "protocol"]).seed.nunique())
    if _tsn.nunique() != 1:
        raise SystemExit("T-sweep cells differ in seed count; \\FNTsweepSeeds "
                         "would misstate some of them")
    L += [("TsweepTlo", int(_ts["T"].min()), "smallest time budget swept"),
          ("TsweepThi", int(_ts["T"].max()), "largest"),
          ("TsweepSeeds", _word(int(_tsn.iloc[0])), "seeds per cell in the sweep"),
          ("TsweepUnmatched", int((_ts.in_spk_spread > 1e-9).sum()),
           "cells where the arms differ in input spikes"),
          ("TsweepSEmax", f"{max(_ts.se_map_pp.max(), _ts.se_spread_pp.max()):.1f}",
           "largest per-cell standard error, pp"),
          # Pooled over all 20 cells rather than per protocol: this is the
          # flatness claim the abstract makes, and it is about the sum, not
          # about either component (see FNMapMovesN / FNSpreadMovesN).
          ("TsweepR",
           f"{np.corrcoef(_ts['T'].astype(float), _ts.d_time_pp)[0, 1]:+.3f}",
           "pooled Pearson(T, Delta_time) over the sweep"),
          ("TsweepRMap",
           f"{np.corrcoef(_ts['T'].astype(float), _ts.d_map_pp)[0, 1]:+.3f}",
           "same for Delta_map"),
          ("TsweepRSpread",
           f"{np.corrcoef(_ts['T'].astype(float), _ts.d_spread_pp)[0, 1]:+.3f}",
           "same for Delta_spread"),
          ("TsweepRatioMax", f"{2.2:.1f}",
           "largest range-to-SE ratio for Delta_time in E3")]

    # Top-weighted rank agreement. These answer the "a frozen tail inflates
    # tau" objection, so they belong beside the tau block rather than in a
    # one-off computation. Names are spelled out because a LaTeX control
    # sequence cannot contain a digit.
    _ra = json.load(open(A / "rank_agreement/agreement.json"))
    _tk = _ra["topk_overlap"]
    L += [("TauTop", f"{_ra['tau_within']:.4f}",
           "tau over the configurations within one screening CD"),
          ("TauBottom", f"{_ra['tau_outside']:.4f}", "tau over the remainder"),
          ("TauTopN", _ra["n_within_cd"], "configurations within one CD"),
          # Derived, not typed: the two counts must partition the design space.
          ("TauBottomN", _ra["n_configurations"] - _ra["n_within_cd"],
           "configurations outside one CD"),
          ("CDCutoff", f"{_ra['within_cd_cutoff']:.4f}",
           "mean-rank cutoff for the leading group"),
          ("TauConcordant", _ra["concordant"], "concordant pairs"),
          ("TauDiscordant", _ra["discordant"], "discordant pairs"),
          ("TauPairs", _ra["n_pairs"], "pairs compared"),
          ("RBOninety", f"{_ra['rbo_p90']:.4f}", "rank-biased overlap, p=0.9"),
          ("RBOeighty", f"{_ra['rbo_p80']:.4f}", "rank-biased overlap, p=0.8"),
          ("TauAPscr", f"{_ra['tau_ap_screening_ref']:.4f}",
           "AP rank correlation, screening as reference"),
          ("TauAPconf", f"{_ra['tau_ap_confirmation_ref']:.4f}",
           "same, confirmation as reference"),
          ("TopKthree", _tk["3"], "top-3 membership overlap"),
          ("TopKfive", _tk["5"], "top-5 membership overlap"),
          ("TopKten", _tk["10"], "top-10 membership overlap"),
          ("TopKfourteen", _tk["14"], "top-14 membership overlap"),
          # Resolvable at 100,000 permutations (16 hits), so the measured value
          # is reported rather than a bound. Contrast \FNKendallP, where zero
          # hits at 10,000 permutations forces an upper bound.
          ("TauTopP", f"{_ra['leading_permutation_p']:.1e}".replace(
              "e-0", r"\times10^{-").replace("e-", r"\times10^{-") + "}",
           "permutation p, leading group only"),
          ("TauTopPerms", f"{_ra['leading_permutation_n']:,}".replace(",", "{,}"),
           "permutations in the leading-group test")]

    # Amplitude-time rank correlation, mapped arm against spread-only arm.
    # Emitted as MEDIANS over the five protocols, with the range beside them,
    # because a single rounded figure would hide KDDCup99: it sits at -0.860
    # and -0.210 where the others are near -1.00 and -0.03, since heavy
    # amplitude ties leave many features sharing a spike time. A caption that
    # quotes only the median should quote the range too. These are summaries of
    # five per-protocol values and are not the same quantity as
    # \FNCtuBSpearmanAtwo, which is one protocol at one T.
    _mc = pd.read_csv(A / "a2/manipulation_checks.csv")
    L += [("SpearmanLatency", f"{_mc.spearman_amp_time_A1.median():.2f}",
           "median amplitude-time rank correlation, mapped arm"),
          ("SpearmanSpread", f"{_mc.spearman_amp_time_A2.median():.2f}",
           "same, spread-only arm"),
          ("SpearmanLatencyLo", f"{_mc.spearman_amp_time_A1.max():.2f}",
           "weakest per-protocol value, mapped arm (KDDCup99)"),
          ("SpearmanSpreadLo", f"{_mc.spearman_amp_time_A2.min():.2f}",
           "strongest residual correlation, spread-only arm (KDDCup99)")]

    # Pareto span over the full design space (fig_pareto27). One protocol:
    # SOP counts scale with input dimension, so a span pooled across protocols
    # of different width would not be a property of the design space. NSL-KDD
    # is the protocol the manuscript quotes.
    _pa = pd.read_csv(A / "pareto27.csv")
    _pa = _pa[_pa.protocol == "nslkdd_v2"]
    L += [("ParetoCostSpan", f"{_pa.sops.max() / _pa.sops.min():.0f}",
           "rate/latency SOP span across all 27 configurations, NSL-KDD"),
          ("ParetoCostMin", f"{_pa.sops.min():,.0f}".replace(",", "{,}"),
           "cheapest configuration, SOPs per sample"),
          ("ParetoCostMax", f"{_pa.sops.max():,.0f}".replace(",", "{,}"),
           "most expensive configuration, SOPs per sample")]

    # Covariate shift (E5). Reported as a measured negative: it eliminates shift
    # magnitude as the explanation for when timing is valuable. The Spearman is
    # computed here, not carried, so the manuscript cannot drift from the file.
    _cs = pd.read_csv(A / "covariate_shift/summary.csv").set_index("protocol")
    _have = _cs.dropna(subset=["timing_premium_pp"])
    _rho = _have[["w_p90", "timing_premium_pp"]].corr(method="spearman").iloc[0, 1]
    _kddfam = _cs.loc[["kddcup99_v2", "nslkdd_v2"], "domain_auc"]
    L += [("ShiftKsCtuB", f"{_cs.loc['ctu13_v2_f1','ks_median']:.4f}",
           "median per-feature KS, CTU-13 f1 (most shifted condition)"),
          ("ShiftAucCtuB", f"{_cs.loc['ctu13_v2_f1','domain_auc']:.4f}",
           "train-vs-test domain-classifier AUC, CTU-13 f1"),
          ("ShiftAucKddLo", f"{_kddfam.min():.2f}",
           "lowest domain AUC in the KDD family"),
          ("ShiftAucKddHi", f"{_kddfam.max():.2f}",
           "highest domain AUC in the KDD family"),
          ("ShiftRho", f"{_rho:.3f}",
           "Spearman(shift, timing premium) over the confirmation protocols"),
          ("ShiftNProto", _word(len(_have)),
           "protocols entering the shift-vs-timing correlation")]

    # Screening ranks recomputed without CTU-13 (sensitivity, not a re-ranking).
    import numpy as _n
    from scipy.stats import rankdata as _rd
    _sw = pd.read_csv(ROOT / "results/v1_submitted/sweep_540.csv")
    _sw["config"] = _sw.neuron + "/" + _sw.encoding
    _w = _sw.pivot_table(index=["dataset", "seed"], columns="config",
                         values="f1_macro").drop(index="ctu13", level=0)
    _r = _n.apply_along_axis(lambda v: _rd(-v, method="average"), 1, _w.to_numpy())
    _mr = pd.Series(_r.mean(axis=0), index=_w.columns).sort_values()
    _top5 = sum("latency" in k for k in _mr.head(5).index)
    L += [("NoCtuBlocks", int(len(_w)), "screening blocks excluding CTU-13"),
          ("NoCtuLeader", _mr.index[0].replace("_", ""), "leader without CTU-13"),
          ("NoCtuLeadRank", f"{_mr.iloc[0]:.2f}", "its mean rank"),
          ("NoCtuChampRank", f"{_mr['LeakyParallel/latency']:.2f}",
           "LP/latency mean rank without CTU-13"),
          ("NoCtuMargin", f"{_mr['LeakyParallel/latency'] - _mr.iloc[0]:.3f}",
           "margin between the two"),
          # Spelled out: the sentence reads "occupy all five leading
          # positions", so a numeral would read as "all 5 leading positions".
          ("NoCtuLatTop", _word(int(_top5)),
           "latency configs in the leading five")]

    # CTU-13 folds in index order, and the dispersion definition.
    ctu = pd.read_csv(ROOT / "results/v1_submitted/ctu13_strict_vanilla.csv")
    cs = (ctu[(ctu.model == "vanilla") & (ctu.condition == "causal_scenario")
              & (ctu.features == "combined")])
    fm = cs.groupby("fold").macro_f1.mean()
    for f in sorted(fm.index):
        L.append((f"CtuIdxF{'ABCD'[int(f)]}", f"{fm[f]:.2f}",
                  f"CTU-13 fold {f} mean, fold-index order"))
    L += [("CtuNEval", int(len(cs)),
           "fold x seed evaluations behind the CTU-13 estimate"),
          ("CtuSdEval", f"{cs.macro_f1.std(ddof=1):.2f}",
           "SD over those evaluations (ddof=1); NOT an across-fold SD")]

    # Threshold transfer, all five protocols at the 1% target.
    ft = pd.read_csv(A / "security/fixed_far_per_seed.csv")
    ft = ft[(ft.neuron == "LeakyParallel") & (ft.encoding == "latency")
            & (ft.target_far == 0.01)]
    for proto, tag in tags.items():
        g = ft[ft.protocol == proto]
        if len(g):
            L += [(f"FarDr{tag}", f"{100*g.test_dr.mean():.1f}",
                   f"DR at the frozen 1% threshold, {proto}")]

    # CTU-13 fold 1 detection behaviour: the strict-results table shows latency
    # and rate flagging almost everything, which is what their macro-F1 there
    # actually reflects.
    sr = pd.read_csv(A / "strict_results.csv")
    b = sr[(sr.axis == "encoding") & (sr.protocol == "ctu13_v2_f1")]
    if len(b):
        L += [("CtuBDrMin", f"{b[b.variant.str.endswith(('latency','rate'))].dr_mean.min():.2f}",
               "lowest DR among latency/rate on CTU-13 fold 1"),
              ("CtuBFarLat", f"{b[b.variant.str.endswith('latency')].far_mean.iloc[0]:.2f}",
               "latency FAR on CTU-13 fold 1"),
              ("CtuBFarRat", f"{b[b.variant.str.endswith('rate')].far_mean.iloc[0]:.2f}",
               "rate FAR on CTU-13 fold 1")]

    # neuron-axis detail on the hard CTU fold, and the strict CTU-13 estimate
    nax = pd.read_csv(A / "v2_neuron_axis/per_dataset.csv")
    f1 = nax[nax.protocol == "ctu13_v2_f1"].set_index("level").f1_mean
    for lvl, tag in (("RSynaptic", "RSyn"), ("RLeaky", "RLky"),
                     ("LeakyParallel", "LP")):
        if lvl in f1.index:
            L.append((f"CtuBNeu{tag}", f"{f1[lvl]:.4f}",
                      f"{lvl} macro-F1 on CTU-13 fold 1, neuron axis"))
    eax = pd.read_csv(A / "v2_encoding_axis/per_dataset.csv")
    ef = eax[eax.protocol == "ctu13_v2_f1"].set_index("level").f1_mean
    for lvl, tag in (("delta", "Del"), ("rate", "Rat"), ("latency", "Lat")):
        L.append((f"CtuBEnc{tag}", f"{ef[lvl]:.4f}",
                  f"{lvl} macro-F1 on CTU-13 fold 1, encoding axis"))
    # The three-condition sequence, from the same 60-run table as the fold means.
    cm = (ctu[(ctu.model == "vanilla") & (ctu.features == "combined")]
          .groupby("condition").macro_f1.mean())
    L += [("CtuStrict", f"{cm['causal_scenario']:.2f}",
           "four-fold strict CTU-13 estimate"),
          ("CtuStrictSD", f"{cs.macro_f1.std(ddof=1):.2f}",
           "SD over its 20 fold x seed evaluations; NOT an across-fold SD"),
          ("CtuLegacy", f"{cm['legacy_random']:.2f}",
           "submitted-protocol CTU-13 value"),
          ("CtuCausalRandom", f"{cm['causal_random']:.2f}",
           "causal aggregates, random split"),
          ("Seeds", int(sr.n.max()), "seeds per cell")]
    if sr.n.nunique() != 1:
        raise SystemExit("strict_results.csv mixes seed counts; \\FNSeeds would "
                         "misstate some cells")

    body = "\n".join(
        f"\\newcommand{{\\FN{m}}}{{{v}}}%".ljust(56) + f" % {c}"
        for m, v, c in L)
    _args.out.write_text(f"""% =====================================================================
% sec_frozen_numbers.tex — GENERATED, DO NOT EDIT BY HAND.
%
% Regenerate with:
%     python scripts/finalize.py && python scripts/make_frozen_numbers.py
%
% Every number in the manuscript is a macro defined here and read from a
% committed artifact under results/analysis/. If a number is not in this file,
% no analysis produced it and it must not appear in the paper.
%
% Experimental record CLOSED 2026-08-13: 235 runs, 5 seeds per cell, 0 failures.
% See results/analysis/AMENDMENT_cic_u2r_support.md for the one analysis
% amendment (CIC-IDS2017 zero U2R test support; no ranking changed).
% =====================================================================

{body}
""")
    if not _args.quiet:
        print(f"wrote {_args.out}  ({len(L)} macros)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
