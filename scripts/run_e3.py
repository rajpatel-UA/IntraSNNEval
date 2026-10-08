#!/usr/bin/env python3
"""E3: how the encodings, and the value of timing, depend on the time budget T.

POST-HOC AND EXPLORATORY. Not part of any registration.

The confirmation sweep fixes $T=25$. Two questions follow from that choice, and
only one of them is a robustness check.

The robustness check is the closed-form input-spike identity
$r = T\\sum_{x>0} x / |\\{x > \\vartheta_\\mathrm{lat}\\}|$, which predicts the
rate-over-latency input-spike ratio from the data alone. Verifying it across
$T \\in \\{5,10,50\\}$ is worth doing, but it is an identity: it will hold, and a
confirmed identity is a good check rather than a finding.

The finding, if there is one, is arm D. E1 established that spike *timing* is
worth +54.5 pp on CTU-13 fold 0 and -9.0 pp on fold 1, measured as
latency@0.01 minus delta@0.01 --- two arms that emit identical input spike sets
and differ only in whether the spike carries timing information. That premium is
defined against delta at latency's gate, so it can only be measured where that
arm exists. Arms A--C below are each encoding at its own gate; none of them can
produce it. Without arm D, E3 cannot say anything about the headline result.

With arm D, it can ask the direct follow-up: **does the value of timing depend
on timing resolution?** $T$ is the number of distinguishable spike times, so it
is the resolution of the temporal code.

Both outcomes are informative, which is why it is worth the 11 GPU-h:

  premium scales with T   timing resolution is the mechanism, and the fold-1
                          liability should shrink at low T --- checkable in the
                          same table, since fold 1 is in the grid
  premium flat in T       the protocol dependence is a property of the traffic
                          rather than of the resolution, which is a cleaner
                          statement than we can currently make

Four arms, LeakyParallel fixed, T in {5, 10, 50}:

    A  rate               no gate
    B  latency @ 0.01     own gate
    C  delta   @ 0.05     own gate
    D  delta   @ 0.01     gate matched to latency --- the timing-premium control

$T=25$ already exists: arms A--C from the confirmation sweep, arm D from E1
arm C. So each curve has a fourth point at no additional cost, and the E1 result
sits inside this grid rather than beside it.

Results land in `results/runs_e3/`, outside the tree `src.readout.discover`
walks, so no arm can leak into the 25-block encoding axis or the 20-block neuron
axis. No registered test is recomputed over the expanded grid.

    python scripts/run_e3.py --dry-run
    python scripts/run_e3.py --workers 6
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

PROTOCOLS = ["nslkdd_v2", "kddcup99_v2", "cicids2017_v2",
             "ctu13_v2_f0", "ctu13_v2_f1"]
SEEDS = [42, 43, 44]
STEPS = [5, 10, 50]
#: arm -> (encoding, gate). D is the one that makes the timing premium
#: measurable; A-C are each encoding at its own gate and cannot produce it.
ARMS = {
    "A_rate": ("rate", None),
    "B_latency_thr001": ("latency", 0.01),
    "C_delta_thr005": ("delta", 0.05),
    "D_delta_thr001": ("delta", 0.01),
}
NEURON = "LeakyParallel"
OUT = ROOT / "results/runs_e3"

#: Where the T=25 points already live, so the curves are not re-run.
EXISTING_T25 = {
    "A_rate": "confirmation sweep, LeakyParallel/rate",
    "B_latency_thr001": "confirmation sweep, LeakyParallel/latency",
    "C_delta_thr005": "confirmation sweep, LeakyParallel/delta",
    "D_delta_thr001": "results/runs_e1/C_delta_thr001",
}

RUNNER = '''import json, sys
from pathlib import Path
ROOT = Path(r"{root}")
sys.path.insert(0, str(ROOT))
from src.training import run_experiment
run_experiment(json.loads(r"""{cfg}"""))
'''


def cfg_for(arm, proto, steps, seed):
    enc, thr = ARMS[arm]
    cfg = {
        "dataset": proto, "neuron": NEURON, "encoding": enc,
        "hidden_size": 128, "num_steps": steps, "batch_size": 128,
        "epochs": 10, "lr": 0.001, "seed": seed,
        "results_dir": str(OUT / arm / f"T{steps}" / proto),
        "summary_csv": str(OUT / "summary_e3.csv"),
    }
    # get_encoder rejects a threshold for rate; do not send one.
    if thr is not None:
        cfg["encoder_threshold"] = thr
    return cfg


def done(arm, proto, steps, seed):
    return (OUT / arm / f"T{steps}" / proto / f"seed_{seed}" /
            "results.json").exists()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--arms", nargs="+", default=list(ARMS), choices=list(ARMS))
    ap.add_argument("--steps", nargs="+", type=int, default=STEPS)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    jobs = [(a, p, t, s) for a in args.arms for p in PROTOCOLS
            for t in args.steps for s in SEEDS]
    pending = [j for j in jobs if not done(*j)]
    print("E3 time-budget sweep (POST-HOC, outside the registration)")
    for a in args.arms:
        enc, thr = ARMS[a]
        gate = "none" if thr is None else f"{thr:.2f}"
        n = len([j for j in pending if j[0] == a])
        print(f"  {a:18s} {enc:8s} gate={gate:5s} {n:3d} pending   "
              f"T=25 from: {EXISTING_T25[a]}")
    print(f"total {len(jobs)} jobs, {len(pending)} pending, "
          f"{args.workers} workers -> {OUT}")
    if args.dry_run or not pending:
        return 0

    OUT.mkdir(parents=True, exist_ok=True)
    lock, state = threading.Lock(), {"n": 0, "fail": 0}
    log = open(OUT / "e3.log", "a")
    log.write(f"\n=== E3 start {time.strftime('%Y-%m-%d %H:%M:%S')} "
              f"arms={args.arms} steps={args.steps} ===\n")

    def launch(job):
        arm, proto, steps, seed = job
        cfg = json.dumps(cfg_for(arm, proto, steps, seed))
        script = RUNNER.format(root=ROOT, cfg=cfg)
        t0 = time.time()
        proc = subprocess.run([sys.executable, "-c", script], cwd=ROOT,
                              capture_output=True, text=True,
                              env=dict(os.environ, HISNN_SEED=str(seed)))
        dt = (time.time() - t0) / 60
        with lock:
            state["n"] += 1
            ok = proc.returncode == 0
            state["fail"] += not ok
            f1 = ""
            if ok:
                try:
                    r = json.loads((OUT / arm / f"T{steps}" / proto /
                                    f"seed_{seed}" / "results.json").read_text())
                    m = r["test_plus"]
                    f1 = (f" f1m={m['metrics']['f1_macro']:.4f}"
                          f" in_spk={m['spike_accounting']['input_spikes_per_sample']:.2f}"
                          f" sops={m['spike_accounting']['sops_per_sample']:.0f}")
                except Exception:
                    pass
            line = (f"[{state['n']}/{len(pending)}] {'ok  ' if ok else 'FAIL'} "
                    f"{arm}/T{steps}/{proto}/s{seed} ({dt:.1f}m){f1}")
            print(line, flush=True)
            log.write(line + "\n")
            if not ok:
                log.write(proc.stderr[-2500:] + "\n")
            log.flush()

    t0 = time.time()
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        list(as_completed([pool.submit(launch, j) for j in pending]))
    print(f"\nE3 finished in {(time.time()-t0)/60:.1f} min, "
          f"{len(pending)-state['fail']}/{len(pending)} ok")
    return 1 if state["fail"] else 0


if __name__ == "__main__":
    sys.exit(main())
