#!/usr/bin/env python3
"""Arm A2 across the time budget: is each COMPONENT flat in T, or only the sum?

POST-HOC. Extends PREREGISTRATION_ADDENDUM.md 8 to T in {5,10,50}; T=25 already
exists in results/runs_a2/.

E3 showed the published premium is flat in T. But Delta_time = Delta_map +
Delta_spread, and on the two KDD protocols those components are +6 and -6, so a
flat sum is consistent with two components that move together and cancel. This
separates the two readings. The abstract currently says the premium is flat; if
the components are not flat that claim needs qualifying, and it is better to
know before submission than after.

Five protocols x T in {5,10,50} x 3 seeds = 45 runs. Wall time is dominated by
the CIC-IDS2017 T=50 critical path, so full coverage costs little more than the
two-protocol version.

Results land in results/runs_a2t/, outside the tree src.readout.discover walks.

    python scripts/run_a2t.py --dry-run
    python scripts/run_a2t.py --workers 10
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
    "A2_permuted_thr001": ("permuted_latency", 0.01),
}
NEURON = "LeakyParallel"
OUT = ROOT / "results/runs_a2t"

#: Where the T=25 points already live, so the curves are not re-run.
EXISTING_T25 = {
    "A2_permuted_thr001": "results/runs_a2/A2_permuted_thr001",
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
        "summary_csv": str(OUT / "summary_a2t.csv"),
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
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--arms", nargs="+", default=list(ARMS), choices=list(ARMS))
    ap.add_argument("--steps", nargs="+", type=int, default=STEPS)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    jobs = [(a, p, t, s) for a in args.arms for p in PROTOCOLS
            for t in args.steps for s in SEEDS]
    pending = [j for j in jobs if not done(*j)]
    print("A2 time-budget sweep (POST-HOC, outside the registration)")
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
    log = open(OUT / "a2t.log", "a")
    log.write(f"\n=== A2T start {time.strftime('%Y-%m-%d %H:%M:%S')} "
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
    print(f"\nA2T finished in {(time.time()-t0)/60:.1f} min, "
          f"{len(pending)-state['fail']}/{len(pending)} ok")
    return 1 if state["fail"] else 0


if __name__ == "__main__":
    sys.exit(main())
