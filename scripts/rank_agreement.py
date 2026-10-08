#!/usr/bin/env python3
"""Top-weighted rank-agreement statistics for the screening/confirmation pair.

Kendall's tau over all 27 configurations answers "do the two protocols agree?"
but not "do they agree where it matters?". A reviewer's objection is that a
frozen tail of obviously bad configurations inflates tau. These statistics
answer it directly:

  tau restricted to the leading group   configurations within one screening
                                        Nemenyi critical difference of the
                                        leader, against the remainder
  rank-biased overlap                   weights the head; p=0.9 puts roughly
                                        86% of the weight in the top 10
  AP rank correlation                   asymmetric, so both directions are
                                        reported rather than one silently
                                        chosen
  top-k membership overlap              the plainest form of the same question
  leading-group permutation test        shuffles ONLY the leading group, so a
                                        stable tail cannot contribute

Written to JSON so `make_frozen_numbers.py` reads a committed artifact rather
than recomputing, and so the 100,000-permutation test runs once.

    python scripts/rank_agreement.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kendalltau

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

OUT = ROOT / "results/analysis/rank_agreement"
N_PERM, PERM_SEED = 100_000, 20260814


def rbo(l1, l2, p: float) -> float:
    """Rank-biased overlap. Both lists are full permutations of one set."""
    s, s1, s2 = 0.0, set(), set()
    for k in range(1, len(l1) + 1):
        s1.add(l1[k - 1]); s2.add(l2[k - 1])
        s += (p ** (k - 1)) * (len(s1 & s2) / k)
    return (1 - p) * s


def tau_ap(ref, other) -> float:
    """AP rank correlation. Asymmetric: `ref` supplies the weighting."""
    pos = {v: i for i, v in enumerate(other)}
    n = len(ref)
    acc = sum(sum(1 for j in range(i) if pos[ref[j]] < pos[ref[i]]) / i
              for i in range(1, n))
    return 2.0 / (n - 1) * acc - 1


def main() -> int:
    d = pd.read_csv(ROOT / "results/analysis/protocol_shift/rank_comparison.csv")
    fr = json.load(open(ROOT / "results/analysis/v1/friedman.json"))
    cd, best = fr["nemenyi_critical_difference"], d.rank_screening.min()
    cutoff = best + cd
    d["within"] = d.rank_screening <= cutoff

    t = lambda s: float(kendalltau(s.rank_screening, s.rank_confirmation)[0])
    a, b = d.rank_screening.to_numpy(), d.rank_confirmation.to_numpy()
    n = len(a)
    conc = disc = 0
    for i in range(n):
        for j in range(i + 1, n):
            s = np.sign(a[i] - a[j]) * np.sign(b[i] - b[j])
            conc += s > 0
            disc += s < 0

    scr = d.sort_values("rank_screening").variant.tolist()
    cnf = d.sort_values("rank_confirmation").variant.tolist()
    topk = {k: len(set(scr[:k]) & set(cnf[:k])) for k in (3, 5, 10, 14)}

    # Permutation restricted to the leading group. This is the null the
    # objection actually calls for: the tail is held fixed, so any agreement it
    # contributes is present under the null as well and cannot inflate p.
    rng = np.random.default_rng(PERM_SEED)
    w = d[d.within]
    obs = t(w)
    wa, wb = w.rank_screening.to_numpy(), w.rank_confirmation.to_numpy()
    null = np.array([kendalltau(wa, rng.permutation(wb))[0]
                     for _ in range(N_PERM)])
    p_top = float((np.abs(null) >= abs(obs)).mean())

    payload = {
        "n_configurations": n,
        "nemenyi_cd": cd, "leader_mean_rank": float(best),
        "within_cd_cutoff": float(cutoff),
        "n_within_cd": int(d.within.sum()),
        "tau_all": t(d), "tau_within": obs, "tau_outside": t(d[~d.within]),
        "concordant": int(conc), "discordant": int(disc),
        "n_pairs": n * (n - 1) // 2,
        "rbo_p90": rbo(scr, cnf, 0.9), "rbo_p80": rbo(scr, cnf, 0.8),
        "tau_ap_screening_ref": tau_ap(scr, cnf),
        "tau_ap_confirmation_ref": tau_ap(cnf, scr),
        "topk_overlap": topk,
        "leading_permutation_p": p_top,
        "leading_permutation_n": N_PERM,
        "rng_seed": PERM_SEED,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "agreement.json").write_text(json.dumps(payload, indent=2))

    print(f"configurations {n}, pairs {payload['n_pairs']}: "
          f"{conc} concordant, {disc} discordant")
    print(f"within-CD cutoff {cutoff:.4f} -> {payload['n_within_cd']} "
          "configurations")
    print(f"tau all {payload['tau_all']:+.4f} | within-CD "
          f"{payload['tau_within']:+.4f} | outside {payload['tau_outside']:+.4f}")
    print(f"RBO p=0.9 {payload['rbo_p90']:.4f}   p=0.8 {payload['rbo_p80']:.4f}")
    print(f"tau_AP screening-ref {payload['tau_ap_screening_ref']:+.4f}   "
          f"confirmation-ref {payload['tau_ap_confirmation_ref']:+.4f}")
    print(f"top-k overlap {topk}")
    print(f"leading-group permutation p = {p_top:.6f} over {N_PERM:,}")
    print(f"\nwrote {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
