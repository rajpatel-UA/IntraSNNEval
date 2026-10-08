#!/usr/bin/env python3
"""CTU-13 under the three aggregation/partition conditions, LeakyParallel/latency.

The 60 runs behind results/v1_submitted/ctu13_strict_vanilla.csv and the
100.00 -> 99.69 -> 92.01 macro-F1 sequence:

    legacy_random     whole-scenario host aggregates, random flow split
                      (the submitted protocol)
    causal_random     partition-local causal aggregates, random flow split
    causal_scenario   causal aggregates, scenario-disjoint frozen folds
                      (the confirmation protocol; alias ctu13_v2_f{0..3})

  x 4 folds x seeds 42-46 = 60 runs. The random partitions are keyed on the
fold, not the seed, so every seed sees the same partition per (condition, fold).

Provenance. The original runs were produced in the parent HiSNN repository by
`scripts/run_ctu13_matrix.py` (launched 2026-08-06 in 3294c6a7, completed in
75464234), a driver that belongs to another line of work and is not part of
this tree. There were no per-cell config files: the config was built in code,
and `cfg_for` below is that construction for LeakyParallel/latency, unchanged.
Each run's results.json records the config it ran with.

This port is verified without retraining in two ways. For all 12
(condition, fold) cells, the conference loader and the parent loader produce
bit-identical train, validation and test arrays. And the confirmation sweep's
own ctu13_v2_f0/f1 LeakyParallel/latency runs, trained independently in this
tree, reproduce the causal_scenario fold 0 and 1 cells to every digit.

Results land in results/runs_ctu13_conditions/, outside the tree
`src.readout.discover` walks, so nothing here can enter a confirmation axis.
`ctu13_strict_vanilla.py --runs-dir results/runs_ctu13_conditions` tabulates
them.

    python scripts/run_ctu13_conditions.py --dry-run
    python scripts/run_ctu13_conditions.py --workers 4
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

CONDITIONS = ["legacy_random", "causal_random", "causal_scenario"]
FOLDS = [0, 1, 2, 3]
SEEDS = [42, 43, 44, 45, 46]
OUT = ROOT / "results/runs_ctu13_conditions"

RUNNER = '''import json, sys
from pathlib import Path
ROOT = Path(r"{root}")
sys.path.insert(0, str(ROOT))
from src.training import run_experiment
run_experiment(json.loads(r"""{cfg}"""))
'''


def cfg_for(condition: str, fold: int, seed: int) -> dict:
    rd = OUT / condition / f"fold{fold}"
    return {
        "dataset": f"ctu13_{condition}_f{fold}_combined",
        "neuron": "LeakyParallel", "encoding": "latency",
        "num_steps": 25, "hidden_size": 128, "batch_size": 128,
        "epochs": 10, "lr": 1e-3, "seed": seed,
        "results_dir": str(rd),
        "summary_csv": str(rd / f"seed_{seed}" / "summary.csv"),
    }


def done(condition, fold, seed):
    return (OUT / condition / f"fold{fold}" / f"seed_{seed}"
            / "results.json").exists()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--conditions", nargs="+", default=CONDITIONS,
                    choices=CONDITIONS)
    ap.add_argument("--folds", nargs="+", type=int, default=FOLDS)
    ap.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    jobs = [(c, f, s) for c in args.conditions for f in args.folds
            for s in args.seeds]
    pending = [j for j in jobs if not done(*j)]
    print("CTU-13 three-condition comparison, LeakyParallel/latency")
    for c in args.conditions:
        print(f"  {c:16s} {len([j for j in pending if j[0] == c])} pending")
    print(f"total {len(jobs)} jobs, {len(pending)} pending, "
          f"{args.workers} workers -> {OUT}")
    if args.dry_run:
        print("\nconfig for the first job:")
        print(json.dumps(cfg_for(*jobs[0]), indent=2))
        return 0
    if not pending:
        return 0

    OUT.mkdir(parents=True, exist_ok=True)
    lock, state = threading.Lock(), {"n": 0, "fail": 0}
    log = open(OUT / "ctu13_conditions.log", "a")
    log.write(f"\n=== start {time.strftime('%Y-%m-%d %H:%M:%S')} "
              f"conditions={args.conditions} folds={args.folds} ===\n")

    def launch(job):
        condition, fold, seed = job
        script = RUNNER.format(root=ROOT,
                               cfg=json.dumps(cfg_for(condition, fold, seed)))
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
                    r = json.loads((OUT / condition / f"fold{fold}"
                                    / f"seed_{seed}" / "results.json").read_text())
                    f1 = f" f1m={r['test_plus']['metrics']['f1_macro']:.4f}"
                except Exception:
                    pass
            line = (f"[{state['n']}/{len(pending)}] {'ok  ' if ok else 'FAIL'} "
                    f"{condition}/fold{fold}/s{seed} ({dt:.1f}m){f1}")
            print(line, flush=True)
            log.write(line + "\n")
            if not ok:
                log.write(proc.stderr[-2500:] + "\n")
            log.flush()

    t0 = time.time()
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        list(as_completed([pool.submit(launch, j) for j in pending]))
    print(f"\nfinished in {(time.time() - t0) / 60:.1f} min, "
          f"{len(pending) - state['fail']}/{len(pending)} ok")
    return 1 if state["fail"] else 0


if __name__ == "__main__":
    sys.exit(main())
