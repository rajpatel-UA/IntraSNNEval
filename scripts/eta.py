#!/usr/bin/env python3
"""Projected completion of the run program. Projection only --- no recommendation.

The C2 fallback rule that used to live here has been **retired**. It read:

    IF projected completion is later than 2026-08-14 04:00
    OR more than 5 jobs have failed
    THEN reduce C2 to 3 seeds across both CTU-13 folds.

It is retired for three independent reasons, any one of which is sufficient:

  1. **The decision it recommends has already been made differently.** C2 ran to
     completion at the full 5 seeds x 2 folds (80 jobs, 0 failures) and no
     amendment was ever written. There is nothing left to reduce.
  2. **Its cutoff no longer exists.** The submission deadline moved to
     2026-08-22. A rule evaluated against 2026-08-14 04:00 fires unconditionally
     from here on and means nothing when it does.
  3. **Its action was unsafe by the time it was reachable.** `--apply` rewrote
     the seed list inside `scripts/run_program.py`, which the confirmation sweep
     is currently executing. Applying it would have edited a running program.

The projection arithmetic is kept because it is still useful. The trigger, the
`--apply` path, and the amendment writer are gone: a machine-readable
"re-run with --apply to reduce C2 to 3 seeds" is a loaded instruction, and the
only thing standing between it and a tired hand at 3am was the discipline to
re-derive why it was printed.

Retired 2026-08-15. History is in git; nothing here needs to be resurrected.

    python scripts/eta.py            # project completion
"""
from __future__ import annotations

import argparse
import datetime as dt
import math
from collections import defaultdict
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from run_program import PROGRAM, cost_h, jobs_for  # noqa: E402

#: Retired 2026-08-14 04:00 cutoff, kept only so the docstring above is
#: checkable against a value rather than a memory. Never compared to.
RETIRED_DEADLINE = dt.datetime(2026, 8, 14, 4, 0)
RULE_RETIRED_ON = "2026-08-15"

#: Every completed job logs its own wall time, so the projection is built from
#: measured runtimes rather than from `cost_h`'s a-priori table. That table
#: underestimates by 1.76x overall and by as much as 2.9x on individual phases,
#: which is enough to move a finish estimate by most of a day.
JOB_LINE = re.compile(
    r"^\[(\w+) \d+/\d+\] (ok|FAIL)\s+(\S+)/(\S+)/(\S+)/s(\d+) \(([\d.]+)m\)")

#: Failures that were diagnosed, fixed and re-run. They stay in the log because
#: the log is the record, but a bare count of them is a trap: whoever reads this
#: output during final reconciliation should not have to re-derive that the
#: failures predate the fix. Anything NOT matching an entry here is printed in
#: full, so this annotates known history without hiding anything new.
RESOLVED_FAILURES = [
    {"phase": "B1", "variant": "LeakyParallel/rate", "protocol": "nslkdd_v2",
     "count": 5, "on": "2026-08-12",
     "why": "two-pass evaluation drew fresh Bernoulli samples on the "
            "stochastic rate encoding; replaced by the single-pass "
            "authoritative _eval_prefix and re-run successfully"},
]

LOG = ROOT / "results/runs/program.log"
#: One stdout log per program invocation. All are read: the B/C tiers and the
#: E2 confirmation sweep are separate invocations, and a projection that sees
#: only one of them reports a throughput that is wrong by the ratio between
#: them. (It did, until 2026-08-15: the phase regex matched ^[BC]\d and so
#: counted every finished E2 run as outstanding.)
STDOUTS = ["results/runs/program_stdout.log", "results/runs/e2_stdout.log"]
PHASE_LINE = re.compile(r"^\[([A-Z]\d[a-z]?) ")


def measure() -> dict:
    """Project completion from measured job runtimes.

    Cost is modelled as (neuron factor) x (protocol factor): a neuron's relative
    expense is stable across protocols and a protocol's across neurons, so an
    unstarted phase on an already-seen protocol can be priced from data instead
    of from a guess.

    Throughput is measured over the active invocation only --- jobs logged after
    the most recent `program start`. Averaging from the first start would fold
    in the idle hours between invocations (there were ~24 between the B/C tiers
    and E2) and understate the rate by exactly the idle fraction.
    """
    jobs, starts, since_last = [], [], []
    for l in LOG.read_text().splitlines():
        if "program start" in l:
            starts.append(dt.datetime.strptime(
                re.search(r"(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d)", l).group(1),
                "%Y-%m-%d %H:%M:%S"))
            since_last = []
            continue
        m = JOB_LINE.match(l)
        if m:
            j = {"phase": m.group(1), "ok": m.group(2) == "ok",
                 "invocation": len(starts),
                 "protocol": m.group(3), "neuron": m.group(4),
                 "encoding": m.group(5), "seed": int(m.group(6)),
                 "h": float(m.group(7)) / 60}
            jobs.append(j)
            since_last.append(j)
    if not starts:
        raise SystemExit("no program start recorded yet")
    if not jobs:
        raise SystemExit("no completed jobs logged yet")

    done = {(j["protocol"], j["neuron"], j["encoding"], j["seed"]) for j in jobs}

    # Cost model: log h ~ mu + a[neuron] + b[protocol], fitted by alternating
    # means. A plain marginal-mean model would be confounded here, because the
    # design is badly unbalanced: CIC-IDS2017 has only ever run with
    # LeakyParallel, and the eight expensive neurons have only ever run on the
    # light protocols. Marginals would then charge CIC the average neuron and
    # the heavy neurons the average protocol, and the two errors do not cancel
    # --- it priced the remaining CIC sweep at half its likely cost. Alternating
    # means recover both factors as long as the observed cells connect the
    # levels, which LeakyParallel does by spanning both protocol groups.
    logs = [(j["neuron"], j["protocol"], math.log(max(j["h"], 1e-6)))
            for j in jobs]
    mu = sum(v for _, _, v in logs) / len(logs)
    a, b = defaultdict(float), defaultdict(float)
    for _ in range(200):
        for key, tgt, other, idx in ((0, a, b, 1), (1, b, a, 0)):
            acc = defaultdict(list)
            for n_p in logs:
                lvl = n_p[key]
                acc[lvl].append(n_p[2] - mu - other[n_p[idx]])
            for lvl, vs in acc.items():
                tgt[lvl] = sum(vs) / len(vs)

    def price(protocol, neuron):
        """Expected wall hours for one job, from the fitted model."""
        return math.exp(mu + a[neuron] + b[protocol])

    # Print the cell counts before averaging over anything. This is a habit,
    # not a check --- there is no threshold it fails at --- and it is here
    # because the same mistake has now been made three times in this project in
    # three different costumes:
    #
    #   * a 145 GPU-h estimate for E2, extrapolated from LeakyParallel without
    #     asking whether LeakyParallel was representative (it is the cheapest
    #     of the nine families);
    #   * an 88 GPU-h estimate for the remaining CIC sweep, from marginal means
    #     on a design where CIC had only ever run with LeakyParallel;
    #   * a 55-67% figure for the input share of SOPs, from a readout fanout
    #     averaged over datasets with different class counts.
    #
    # The invariant: do not average over a factor whose levels are not balanced
    # across the factor you are averaging within. The useful part is that the
    # error is invisible in the marginals and obvious in the occupancy pattern
    # --- every one of the three would have been caught by looking at which
    # cells are empty before taking the mean.
    #
    # Second clause, learned on 2026-08-15 when CIC-IDS2017 first ran with a
    # family other than LeakyParallel. The occupancy pattern tells you which
    # combinations you have no information about. It does not tell you whether
    # the model interpolating across them has the right *shape*. Additivity in
    # logs was the assumption sitting underneath the fix above, and it is
    # wrong here: CIC's cost multiplier grows with the neuron's own cost
    # (3.57x for LeakyParallel, 4.33x Lapicque, 4.59x Alpha), so there is a
    # positive interaction the model cannot express.
    #
    # Both failures compounded in the same direction, which is the part worth
    # remembering. The one cell identifying the CIC column was LeakyParallel:
    # simultaneously the cheapest family and the one with the *smallest* CIC
    # penalty. So the fitted multiplier was the minimum of the nine rather than
    # a typical one, and the E2e estimate came out at roughly half its true
    # cost. The unobserved cells were not missing at random with respect to the
    # quantity being estimated --- the same structure as the marginal-means
    # error, one level further down.
    #
    # Not fixed by fitting an interaction: there are three CIC-observed
    # families and one of them is the atypical LeakyParallel, so a slope would
    # rest on two informative points. The working number is CIC ~ 4.45 x the
    # light-protocol cost, and results are reported against actuals.
    cells = defaultdict(int)
    for j in jobs:
        cells[(j["neuron"], j["protocol"])] += 1

    remaining, per_phase = 0.0, defaultdict(float)
    for phase in PROGRAM:
        for _ph, protocol, neuron, encoding, seed, _rp in jobs_for(phase):
            if (protocol, neuron, encoding, seed) in done:
                continue
            c = price(protocol, neuron)
            remaining += c
            per_phase[phase] += c

    now = dt.datetime.now()
    t_active = max(starts)
    active_h = (now - t_active).total_seconds() / 3600
    rate = (sum(j["h"] for j in since_last) / active_h) if active_h > 0 else 0.0
    eta_h = remaining / rate if rate > 0 else float("inf")
    return {
        "t0": min(starts), "t_active": t_active, "now": now,
        "elapsed_h": (now - min(starts)).total_seconds() / 3600,
        "active_h": active_h, "jobs_done": len(jobs),
        "active_jobs": len(since_last),
        "failures": sum(not j["ok"] for j in jobs),
        "failed_jobs": [j for j in jobs if not j["ok"]],
        "active_failures": sum(not j["ok"] for j in since_last),
        "retired_h": sum(j["h"] for j in jobs), "rate": rate,
        "remaining_h": remaining, "per_phase": dict(per_phase),
        "eta_h": eta_h, "finish": now + dt.timedelta(hours=eta_h),
        "cells": dict(cells),
        "neuron_h": {k: math.exp(mu + v) for k, v in a.items()},
        "protocol_h": {k: math.exp(mu + v) for k, v in b.items()},
    }


def annotate_failures(failed) -> list[str]:
    """Explain known-and-resolved failures; surface anything unrecognised.

    A raw failure count in a status line is a trap during reconciliation: it
    looks like an open problem and costs whoever reads it a trip through the
    log to establish that it is not. Resolved incidents are named with their
    cause and their fix. Anything that does not match one is printed in full
    rather than folded into a total.
    """
    out, unexplained = [], list(failed)
    for inc in RESOLVED_FAILURES:
        hit = [j for j in unexplained
               if j["phase"] == inc["phase"]
               and j["protocol"] == inc["protocol"]
               and f"{j['neuron']}/{j['encoding']}" == inc["variant"]]
        if not hit:
            continue
        for j in hit:
            unexplained.remove(j)
        out.append(f"{len(hit)} x {inc['phase']} {inc['protocol']}/"
                   f"{inc['variant']} ({inc['on']}, RESOLVED): {inc['why']}")
    for j in unexplained:
        out.append(f"UNEXPLAINED: {j['phase']} {j['protocol']}/{j['neuron']}/"
                   f"{j['encoding']}/s{j['seed']} --- not a known incident, "
                   "investigate before relying on any table that includes it")
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cells", action="store_true",
                    help="print the neuron x protocol occupancy crosstab the "
                         "cost model is fitted on, and exit")
    ap.add_argument("--apply", action="store_true",
                    help=argparse.SUPPRESS)   # retired; hard-fails, see below
    args = ap.parse_args()

    if args.apply:
        print("REFUSED: the C2 fallback rule was retired on "
              f"{RULE_RETIRED_ON}.\n"
              "  C2 already completed at 5 seeds x 2 CTU-13 folds (80 jobs, 0 "
              "failures);\n"
              "  there is nothing to reduce, the 14 Aug cutoff it tested no "
              "longer exists\n"
              "  (deadline is 22 Aug), and applying it would rewrite the seed "
              "list inside\n"
              "  scripts/run_program.py while the confirmation sweep is "
              "executing it.\n"
              "  If a schedule cut is genuinely needed, decide it deliberately "
              "and write\n"
              "  the amendment by hand. Do not let a retired rule make it.",
              file=sys.stderr)
        return 2

    m = measure()

    if args.cells:
        neurons = sorted({n for n, _ in m["cells"]})
        protos = sorted({p for _, p in m["cells"]})
        w = max(len(n) for n in neurons)
        print(f"{'':{w}}  " + "  ".join(f"{p[:11]:>11s}" for p in protos))
        for n in neurons:
            row = [m["cells"].get((n, p), 0) for p in protos]
            print(f"{n:{w}s}  " + "  ".join(
                f"{'.':>11s}" if c == 0 else f"{c:>11d}" for c in row))
        empty = sum(1 for n in neurons for p in protos
                    if (n, p) not in m["cells"])
        print(f"\n{len(neurons)*len(protos) - empty} of "
              f"{len(neurons)*len(protos)} cells observed, {empty} empty.")
        print("Marginal means over an unbalanced design like this one are "
              "confounded:\nthe cost model is fitted as "
              "log h ~ mu + a[neuron] + b[protocol] instead.")
        return 0

    print(f"first start      {m['t0']:%a %d %b %H:%M}")
    print(f"active since     {m['t_active']:%a %d %b %H:%M}   "
          f"({m['active_h']:.1f} h)")
    print(f"jobs done        {m['jobs_done']}  ({m['active_jobs']} this "
          f"invocation)")
    print(f"failures         {m['failures']} total, "
          f"{m['active_failures']} in the active invocation")
    for line in annotate_failures(m["failed_jobs"]):
        print(f"  {line}")
    print(f"consumed         {m['retired_h']:.1f} GPU-h measured")
    print(f"throughput       {m['rate']:.2f} GPU-h per wall-hour")
    print(f"remaining        {m['remaining_h']:.1f} GPU-h projected")
    for ph, h in sorted(m["per_phase"].items()):
        print(f"  {ph:5s}         {h:6.1f} GPU-h")
    print(f"projected finish {m['finish']:%a %d %b %H:%M}")
    print("\nCosts are modelled from measured runtimes, not from cost_h's table.")
    print("Projection only. The C2 fallback rule was retired on "
          f"{RULE_RETIRED_ON}; this script\nrecommends nothing and changes "
          "nothing.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
