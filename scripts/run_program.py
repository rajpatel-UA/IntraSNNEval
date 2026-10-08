#!/usr/bin/env python3
"""The frozen Tier B + C1 + C2 run program.

Encodes the authorised experiment set as data rather than as a sequence of
shell invocations, so "what was run" is answerable by reading one list and
resuming is the same command again. Nothing outside `PROGRAM` runs; adding a
tier means editing it deliberately.

Phases exist for memory, not for science. CIC-IDS2017 holds ~2.8M x 78 float32
in each worker; the other protocols hold a tenth of that, so they get a wider
pool. Phase order still follows experiment priority: all of Tier B before
Tier C.

Worker sizing is measured, not assumed, and the measurement was a surprise worth
recording. Throughput on 8 identical NSL-KDD jobs:

    workers=1   561 s   0.86 jobs/min   1.00x
    workers=4   314 s   1.53 jobs/min   1.79x
    workers=8   333 s   1.44 jobs/min   1.68x   <- past the peak
    workers=20   -               -      ~2.5x aggregate, 7.9x slower per job

VRAM is irrelevant here: 20 concurrent jobs used 18 GB of the H200's 143 GB, so
the GPU has ~8x the memory headroom it needs and adding workers on that basis
makes throughput *worse*. The binding resource is GPU context scheduling — these
are thousands of tiny kernels from a T=25 Python loop, and without MPS (which
needs root here) the driver time-slices whole contexts rather than running them
concurrently. Past ~4 concurrent processes the scheduler spends more time
switching than computing.

Short jobs pay proportionally more process-startup cost, so the NSL-KDD curve
above is a pessimistic bound for the long CTU-13 and CIC-IDS2017 runs; 6 is
chosen as a small step past the measured peak for that reason.

    python scripts/run_program.py                  # run everything outstanding
    python scripts/run_program.py --dry-run        # show the plan and the ETA
    python scripts/run_program.py --phases B1 B2   # a subset
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

NEURONS = ["Alpha", "Lapicque", "Leaky", "LeakyParallel", "RLeaky",
           "RSynaptic", "SConv2dLSTM", "SLSTM", "Synaptic"]
SEEDS = [42, 43, 44, 45, 46]
ENC3 = ["rate", "latency", "delta"]

#: phase -> (description, protocols, neurons, encodings, workers)
PROGRAM = {
    # E2: the full confirmation sweep, registered in
    # PREREGISTRATION_ADDENDUM.md before any of its runs completed. Ordered
    # cheapest protocol first so the ranking becomes computable on four
    # protocols well before CIC-IDS2017 finishes.
    "E2a": ("E2 full sweep - NSL-KDD",   ["nslkdd_v2"],    NEURONS, ENC3, 6),
    "E2b": ("E2 full sweep - CTU-13 f0", ["ctu13_v2_f0"],  NEURONS, ENC3, 6),
    "E2c": ("E2 full sweep - CTU-13 f1", ["ctu13_v2_f1"],  NEURONS, ENC3, 6),
    "E2d": ("E2 full sweep - KDDCup99",  ["kddcup99_v2"],  NEURONS, ENC3, 6),
    "E2e": ("E2 full sweep - CIC-IDS2017", ["cicids2017_v2"], NEURONS, ENC3, 5),
    "B1": ("Tier B — LP x 3 encodings, light protocols",
           ["nslkdd_v2", "kddcup99_v2", "ctu13_v2_f0", "ctu13_v2_f1"],
           ["LeakyParallel"], ENC3, 6),
    "B2": ("Tier B — LP x 3 encodings, CIC-IDS2017 (memory-heavy)",
           ["cicids2017_v2"], ["LeakyParallel"], ENC3, 4),
    "C1": ("Tier C1 — 9 neurons x latency, KDD family",
           ["nslkdd_v2", "kddcup99_v2"], NEURONS, ["latency"], 6),
    "C2": ("Tier C2 — 9 neurons x latency, CTU-13 scenario-disjoint",
           ["ctu13_v2_f0", "ctu13_v2_f1"], NEURONS, ["latency"], 6),
}

#: Measured single-job GPU-hours from the v1 sweep, for the ETA only.
COST_H = {
    "nslkdd_v2": {"LeakyParallel": 0.019}, "kddcup99_v2": {"LeakyParallel": 0.133},
    "cicids2017_v2": {"LeakyParallel": 0.551}, "ctu13_v2_f0": {"LeakyParallel": 0.220},
    "ctu13_v2_f1": {"LeakyParallel": 0.220},
}
DEFAULT_COST = {"nslkdd_v2": 0.042, "kddcup99_v2": 0.202, "cicids2017_v2": 0.902,
                "ctu13_v2_f0": 0.304, "ctu13_v2_f1": 0.304}


def cost_h(protocol: str, neuron: str) -> float:
    return COST_H.get(protocol, {}).get(neuron, DEFAULT_COST.get(protocol, 0.3))


def result_json(protocol: str, neuron: str, enc: str, seed: int) -> Path:
    return (ROOT / "results/runs" / protocol / f"SNN_{neuron}" / enc
            / f"seed_{seed}" / "results.json")


def jobs_for(phase: str):
    _desc, protocols, neurons, encodings, _w = PROGRAM[phase]
    out = []
    for proto in protocols:
        for neuron in neurons:
            for enc in encodings:
                for seed in SEEDS:
                    run_py = (ROOT / "experiments" / proto / f"SNN_{neuron}"
                              / enc / "run.py")
                    if run_py.exists():
                        out.append((phase, proto, neuron, enc, seed, run_py))
    return out


def is_done(job) -> bool:
    _ph, proto, neuron, enc, seed, _p = job
    p = result_json(proto, neuron, enc, seed)
    if not p.exists():
        return False
    try:
        # An artifact-less result predates the instrumentation and must re-run,
        # otherwise the early-readout analyses would silently skip those cells.
        return (p.parent / "artifacts" / "test_prefix.npz").exists()
    except OSError:
        return False


def run_phase(phase: str, force: bool, log) -> int:
    desc, _protos, _n, _e, workers = PROGRAM[phase]
    jobs = jobs_for(phase)
    pending = jobs if force else [j for j in jobs if not is_done(j)]
    print(f"\n{'='*78}\n{phase}: {desc}\n"
          f"  {len(jobs)} jobs, {len(pending)} pending, {workers} workers, "
          f"~{sum(cost_h(j[1], j[2]) for j in pending):.1f} GPU-h\n{'='*78}",
          flush=True)
    if not pending:
        return 0

    lock = threading.Lock()
    state = {"done": 0, "fail": 0}
    t0 = time.time()

    def launch(job):
        _ph, proto, neuron, enc, seed, run_py = job
        env = dict(os.environ, HISNN_SEED=str(seed))
        if force:
            env["HISNN_FORCE"] = "1"
        s = time.time()
        proc = subprocess.run([sys.executable, str(run_py)], cwd=ROOT, env=env,
                              capture_output=True, text=True)
        dt = time.time() - s
        with lock:
            state["done"] += 1
            ok = proc.returncode == 0
            state["fail"] += not ok
            f1 = ""
            if ok:
                try:
                    r = json.loads(result_json(proto, neuron, enc, seed).read_text())
                    m = r["test_plus"]["metrics"]
                    f1 = f" f1m={m['f1_macro']:.4f} MCC={m['matthews_corrcoef']:.4f}"
                except Exception:
                    pass
            line = (f"[{phase} {state['done']}/{len(pending)}] "
                    f"{'ok  ' if ok else 'FAIL'} {proto}/{neuron}/{enc}/s{seed} "
                    f"({dt/60:.1f}m){f1}")
            print(line, flush=True)
            log.write(line + "\n")
            if not ok:
                log.write(proc.stderr[-3000:] + "\n")
            log.flush()

    with ThreadPoolExecutor(max_workers=workers) as pool:
        list(as_completed([pool.submit(launch, j) for j in pending]))

    print(f"{phase} finished in {(time.time()-t0)/60:.1f} min — "
          f"{len(pending)-state['fail']}/{len(pending)} ok", flush=True)
    return state["fail"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--phases", nargs="+", default=list(PROGRAM),
                    choices=list(PROGRAM))
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    total = pend = 0.0
    n_total = n_pend = 0
    for ph in args.phases:
        jobs = jobs_for(ph)
        todo = [j for j in jobs if args.force or not is_done(j)]
        n_total += len(jobs); n_pend += len(todo)
        total += sum(cost_h(j[1], j[2]) for j in jobs)
        pend += sum(cost_h(j[1], j[2]) for j in todo)
        print(f"{ph}: {len(jobs):4d} jobs ({len(todo):4d} pending)  "
              f"{sum(cost_h(j[1], j[2]) for j in todo):6.1f} GPU-h pending  "
              f"{PROGRAM[ph][4]:2d} workers   {PROGRAM[ph][0]}")
    print(f"\ntotal: {n_total} jobs, {n_pend} pending, {pend:.1f} GPU-h outstanding")
    if args.dry_run:
        return 0

    logp = ROOT / "results/runs/program.log"
    logp.parent.mkdir(parents=True, exist_ok=True)
    failures = 0
    with open(logp, "a") as log:
        log.write(f"\n=== program start {time.strftime('%Y-%m-%d %H:%M:%S')} "
                  f"phases={args.phases} ===\n")
        for ph in args.phases:
            failures += run_phase(ph, args.force, log)
    print(f"\nprogram complete — {failures} failed job(s). log: {logp}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
