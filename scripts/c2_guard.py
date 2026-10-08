#!/usr/bin/env python3
"""Watchdog that evaluates the frozen C2 rule at the moment it must be decided.

The rule in `scripts/eta.py` has to be applied *before C2 begins*, otherwise
partial C2 results exist and the decision is no longer clean. It also has to
survive nobody being at the terminal at 07:00 on the 13th.

Two facts make this a separate process rather than a hook inside the driver:
the driver is already running, so an edit to its source would not be picked up;
and applying the amendment requires the driver to reload `PROGRAM` anyway. So
this watcher waits until C1 has finished, evaluates the rule, and only if it
fires does it stop the driver, amend the program and relaunch. Relaunching costs
nothing: the driver skips every run whose `results.json` and artifacts already
exist, so completed work is never repeated.

If the rule does not fire, the watcher does nothing at all and exits. It never
touches C2 on the basis of a C2 result, because it runs before any exists.

    nohup python scripts/c2_guard.py > results/runs/c2_guard.log 2>&1 &
"""
from __future__ import annotations

import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import eta  # noqa: E402
from run_program import PROGRAM, jobs_for  # noqa: E402

STDOUT = ROOT / "results/runs/program_stdout.log"
POLL_S = 300


def phase_counts() -> dict:
    if not STDOUT.exists():
        return {}
    out = {}
    for line in STDOUT.read_text().splitlines():
        m = re.match(r"^\[([BC]\d) (\d+)/(\d+)\]", line)
        if m:
            out[m.group(1)] = (int(m.group(2)), int(m.group(3)))
    return out


def driver_pids() -> list:
    r = subprocess.run(["ps", "-eo", "pid,args"], capture_output=True, text=True)
    return [int(l.split()[0]) for l in r.stdout.splitlines()
            if "scripts/run_program.py" in l and "ps -eo" not in l]


def c2_started() -> bool:
    return "C2" in phase_counts()


def c1_complete() -> bool:
    c = phase_counts().get("C1")
    return bool(c and c[0] >= c[1])


def main() -> int:
    n_c2 = len(jobs_for("C2"))
    print(f"c2_guard armed: cutoff {eta.DEADLINE:%Y-%m-%d %H:%M}, "
          f"failure budget {eta.FAILURE_BUDGET}, C2 has {n_c2} jobs at 5 seeds",
          flush=True)

    while True:
        if c2_started():
            print("C2 already started — rule window has closed, standing down.",
                  flush=True)
            return 0
        if c1_complete():
            break
        try:
            m = eta.measure()
            print(f"[{time.strftime('%H:%M')}] {m['jobs_done']} jobs, "
                  f"{m['failures']} failed, {m['rate']:.2f} GPU-h/wall-h, "
                  f"finish {m['finish']:%a %H:%M}, "
                  f"margin {(eta.DEADLINE - m['finish']).total_seconds()/3600:+.1f} h",
                  flush=True)
        except SystemExit as e:
            print(f"measure failed: {e}", flush=True)
        time.sleep(POLL_S)

    # C1 is done and C2 has not started: this is the decision point.
    m = eta.measure()
    reasons = []
    if m["finish"] > eta.DEADLINE:
        reasons.append(f"projected finish {m['finish']:%d %b %H:%M} is after "
                       f"{eta.DEADLINE:%d %b %H:%M}")
    if m["failures"] > eta.FAILURE_BUDGET:
        reasons.append(f"{m['failures']} failures exceed budget "
                       f"{eta.FAILURE_BUDGET}")

    if not reasons:
        margin = (eta.DEADLINE - m["finish"]).total_seconds() / 3600
        print(f"decision point: rule does NOT fire ({margin:.1f} h margin). "
              "C2 proceeds at 5 seeds x 2 folds.", flush=True)
        return 0

    print(f"decision point: RULE FIRES — {'; '.join(reasons)}", flush=True)
    pids = driver_pids()
    for p in pids:
        subprocess.run(["kill", "-9", str(p)])
    time.sleep(5)
    subprocess.run(["pkill", "-9", "-f", "experiments/"], capture_output=True)
    time.sleep(5)

    eta.apply_amendment(m, "; ".join(reasons))

    log = open(ROOT / "results/runs/program_stdout.log", "a")
    subprocess.Popen([sys.executable, "scripts/run_program.py"], cwd=ROOT,
                     stdout=log, stderr=subprocess.STDOUT)
    print("driver relaunched with the amended C2; completed runs are skipped.",
          flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
