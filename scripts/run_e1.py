#!/usr/bin/env python3
"""E1: threshold-matched encoding ablation. POST-HOC AND EXPLORATORY.

Why this exists. The three encodings do not differ only in how they represent a
value; two of them apply an amplitude gate before spiking and one does not.
Latency gates at 0.01, delta at 0.05 (and on a static vector delta reduces
exactly to that gate, verified bitwise), while rate has no gate. So the
three-way comparison confounds the code with the presence and value of a
threshold, and the confound runs in the same direction as the efficiency
conclusion: on NSL-KDD only 14.58 of 125 features clear latency's gate, and
removing it would raise latency's input SOPs from 1,866 to 16,000, cutting the
measured 9.67x advantage over rate to at most 1.86x. Roughly five-sixths of the
headline ratio is gating; the remainder is timing.

Three arms, LeakyParallel fixed:

    A  latency @ 0.00   code-intrinsic gap vs rate, gate removed
    B  latency @ 0.05   timing vs no-timing at a gate matched to delta
    C  delta   @ 0.01   symmetric control, gate matched to latency

Five confirmation protocols, three seeds. CTU-13 f1 tests whether the encoding
inversion is a gate-value effect; CIC-IDS2017 anchors the SOP range.

**This is not part of the pre-registered confirmation.** Results land in
`results/runs_e1/`, deliberately outside the tree `src.readout.discover` walks,
so no arm can leak into the 25-block encoding axis or the 20-block neuron axis.
No Friedman or Wilcoxon from the registration is recomputed over the expanded
arm set: adding arms after seeing results and then re-testing would destroy the
thing the pre-registration exists to protect. The subsection reporting this is
labelled post-hoc and hypothesis-generating.

    python scripts/run_e1.py --dry-run
    python scripts/run_e1.py --workers 6
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
#: arm -> (encoding, gate). Order is the run order; C is the droppable one.
ARMS = {
    "A_latency_thr000": ("latency", 0.00),
    "B_latency_thr005": ("latency", 0.05),
    "C_delta_thr001": ("delta", 0.01),
}
NEURON = "LeakyParallel"
OUT = ROOT / "results/runs_e1"

RUNNER = '''import json, sys
from pathlib import Path
ROOT = Path(r"{root}")
sys.path.insert(0, str(ROOT))
from src.training import run_experiment
run_experiment(json.loads(r"""{cfg}"""))
'''


def cfg_for(arm, proto, seed):
    enc, thr = ARMS[arm]
    return {
        "dataset": proto, "neuron": NEURON, "encoding": enc,
        "encoder_threshold": thr,
        "hidden_size": 128, "num_steps": 25, "batch_size": 128,
        "epochs": 10, "lr": 0.001, "seed": seed,
        "results_dir": str(OUT / arm / proto),
        "summary_csv": str(OUT / "summary_e1.csv"),
    }


def done(arm, proto, seed):
    return (OUT / arm / proto / f"seed_{seed}" / "results.json").exists()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--arms", nargs="+", default=list(ARMS), choices=list(ARMS))
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    jobs = [(a, p, s) for a in args.arms for p in PROTOCOLS for s in SEEDS]
    pending = [j for j in jobs if not done(*j)]
    print(f"E1 threshold-matched ablation (POST-HOC, outside the registration)")
    for a in args.arms:
        enc, thr = ARMS[a]
        print(f"  {a:20s} {enc:8s} gate={thr:.2f}  "
              f"{len([j for j in pending if j[0] == a])} pending")
    print(f"total {len(jobs)} jobs, {len(pending)} pending, "
          f"{args.workers} workers -> {OUT}")
    if args.dry_run or not pending:
        return 0

    OUT.mkdir(parents=True, exist_ok=True)
    lock, state = threading.Lock(), {"n": 0, "fail": 0}
    log = open(OUT / "e1.log", "a")
    log.write(f"\n=== E1 start {time.strftime('%Y-%m-%d %H:%M:%S')} "
              f"arms={args.arms} ===\n")

    def launch(job):
        arm, proto, seed = job
        cfg = json.dumps(cfg_for(arm, proto, seed))
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
                    r = json.loads((OUT / arm / proto / f"seed_{seed}" /
                                    "results.json").read_text())
                    m = r["test_plus"]
                    f1 = (f" f1m={m['metrics']['f1_macro']:.4f}"
                          f" in_spk={m['spike_accounting']['input_spikes_per_sample']:.2f}"
                          f" sops={m['spike_accounting']['sops_per_sample']:.0f}")
                except Exception:
                    pass
            line = (f"[{state['n']}/{len(pending)}] {'ok  ' if ok else 'FAIL'} "
                    f"{arm}/{proto}/s{seed} ({dt:.1f}m){f1}")
            print(line, flush=True)
            log.write(line + "\n")
            if not ok:
                log.write(proc.stderr[-2500:] + "\n")
            log.flush()

    t0 = time.time()
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        list(as_completed([pool.submit(launch, j) for j in pending]))
    print(f"\nE1 finished in {(time.time()-t0)/60:.1f} min, "
          f"{len(pending)-state['fail']}/{len(pending)} ok")
    return 1 if state["fail"] else 0


if __name__ == "__main__":
    sys.exit(main())
