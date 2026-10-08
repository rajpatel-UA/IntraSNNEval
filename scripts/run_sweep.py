#!/usr/bin/env python3
"""Parallel sweep driver for one protocol.

Discovers every ``experiments/<protocol>/SNN_<Neuron>/<encoding>/run.py`` and
runs them as subprocesses, up to ``--workers`` at a time. The models are small
enough that a single H200 sits idle under one job; the useful parallelism is
across processes, bounded by CPU cores for the encoders and data loaders rather
than by GPU memory.

Idempotent: a job whose ``results.json`` already exists is skipped, so an
interrupted sweep resumes by re-running the same command. ``--force`` overrides.

    python scripts/run_sweep.py nslkdd_v2 --seeds 42 43 44 45 46 --workers 4
    python scripts/run_sweep.py ctu13_v2_f0 --only LeakyParallel --workers 2
    python scripts/run_sweep.py cicids2017_v2 --dry-run
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

NEURONS = ["Alpha", "Lapicque", "Leaky", "LeakyParallel",
           "RLeaky", "RSynaptic", "SConv2dLSTM", "SLSTM", "Synaptic"]
ENCODINGS = ["rate", "latency", "delta"]


def discover(protocol: str, only: list, encodings: list):
    base = ROOT / "experiments" / protocol
    if not base.exists():
        sys.exit(f"error: {base} does not exist")
    keep = {o.lower() for o in only}
    runs = []
    for n in NEURONS:
        if keep and n.lower() not in keep:
            continue
        for e in encodings:
            p = base / f"SNN_{n}" / e / "run.py"
            if p.exists():
                runs.append((n, e, p))
    return runs


def result_path(protocol: str, neuron: str, enc: str, seed: int) -> Path:
    return (ROOT / "results" / "runs" / protocol / f"SNN_{neuron}" / enc
            / f"seed_{seed}" / "results.json")


def status_for(protocol: str, neuron: str, enc: str, seed: int) -> str:
    rj = result_path(protocol, neuron, enc, seed)
    if not rj.exists():
        return "pending"
    try:
        m = json.load(open(rj))["test_plus"]["metrics"]
        return f"done f1m={m['f1_macro']:.3f} MCC={m['matthews_corrcoef']:.3f}"
    except Exception:
        return "done (unparseable)"


def gpu_preflight() -> None:
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index,name,utilization.gpu,memory.used,memory.total",
             "--format=csv,noheader,nounits"], text=True, timeout=5)
        print("---- GPU ----")
        print(out.strip())
    except Exception as e:
        print(f"nvidia-smi unavailable ({e}) — proceeding.")
    print()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("protocol", help="protocol name, e.g. nslkdd_v2 or ctu13_v2_f0")
    ap.add_argument("--only", nargs="+", default=[], help="restrict to these neurons")
    ap.add_argument("--encodings", nargs="+", default=ENCODINGS, choices=ENCODINGS)
    ap.add_argument("--seeds", nargs="+", type=int, default=[42])
    ap.add_argument("--workers", type=int, default=1,
                    help="concurrent subprocesses (default 1)")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    runs = discover(args.protocol, args.only, args.encodings)
    if not runs:
        sys.exit("no matching runs found")

    jobs = [(n, e, p, s) for s in args.seeds for (n, e, p) in runs]
    pending = [j for j in jobs
               if args.force or not result_path(args.protocol, j[0], j[1], j[3]).exists()]

    gpu_preflight()
    print(f"protocol : {args.protocol}")
    print(f"jobs     : {len(jobs)} total, {len(pending)} pending, "
          f"{len(jobs) - len(pending)} already done")
    print(f"workers  : {args.workers}")
    print()
    for n, e, _, s in jobs:
        print(f"  {n:14} {e:8} seed={s:<4} {status_for(args.protocol, n, e, s)}")
    print()
    if args.dry_run or not pending:
        return

    py = sys.executable
    lock = __import__("threading").Lock()
    done = {"n": 0}
    failures = []

    def launch(job):
        n, e, run_py, s = job
        env = dict(os.environ, HISNN_SEED=str(s))
        if args.force:
            env["HISNN_FORCE"] = "1"
        t0 = time.time()
        proc = subprocess.run([py, str(run_py)], cwd=ROOT, env=env,
                              capture_output=True, text=True)
        dt = time.time() - t0
        with lock:
            done["n"] += 1
            tag = "ok  " if proc.returncode == 0 else "FAIL"
            print(f"[{done['n']}/{len(pending)}] {tag} SNN_{n}/{e} seed={s} "
                  f"({dt/60:.1f} min) — {status_for(args.protocol, n, e, s)}", flush=True)
            if proc.returncode != 0:
                failures.append((n, e, s, proc.stderr.strip()[-2000:]))
        return proc.returncode

    t0 = time.time()
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        list(as_completed([pool.submit(launch, j) for j in pending]))

    print("\n" + "=" * 80)
    print(f"sweep finished in {(time.time()-t0)/60:.1f} min — "
          f"{len(pending)-len(failures)}/{len(pending)} ok, {len(failures)} failed")
    for n, e, s, err in failures:
        print(f"\nFAIL SNN_{n}/{e} seed={s}\n{err}")
    if failures:
        sys.exit(1)


if __name__ == "__main__":
    main()
