#!/usr/bin/env python3
"""End-to-end validation of one completed run's artifacts, before the sweep.

Everything downstream — early readout, T95/T99, fixed-FAR operating points,
per-class tables, SOP accounting — reads the files this checks. A silent defect
here does not crash anything; it produces plausible numbers that are wrong. So
each property is asserted rather than eyeballed, and the sweep does not launch
until all of them hold.

    python scripts/smoke_check.py results/runs/nslkdd_v2/SNN_LeakyParallel/latency/seed_42
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

TOL = 1e-9


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir", help="a seed_<S> directory containing results.json")
    args = ap.parse_args()
    run = Path(args.run_dir)
    if not run.is_absolute():
        run = ROOT / run

    rec = json.loads((run / "results.json").read_text())
    art = run / "artifacts"
    pre = np.load(art / "test_prefix.npz")
    val = np.load(art / "val_scores.npz")

    cum, y_true, y_pred, idx = pre["cum_out"], pre["y_true"], pre["y_pred"], pre["idx"]
    tp = rec["test_plus"]
    T = rec["config"]["num_steps"]
    C = rec["data"]["num_classes"]
    checks = []

    def check(name, ok, detail=""):
        checks.append((name, bool(ok), detail))

    # 1 — prefix structure
    check("prefix shape is [N, T, C]",
          cum.ndim == 3 and cum.shape[1] == T and cum.shape[2] == C,
          f"got {cum.shape}, expected [N, {T}, {C}]")

    # 2 — terminal prefix reproduces the ordinary prediction
    # The trainer already raises if the logits differ; this re-checks the
    # *written* file, which is what every analysis will actually read.
    terminal_pred = cum[:, -1, :].astype(np.int32).argmax(axis=1)
    check("prefix at t=T reproduces the saved prediction",
          np.array_equal(terminal_pred, y_pred),
          f"{int((terminal_pred != y_pred).sum())} mismatches")

    # 3 — reconstructed macro-F1 equals the trainer's
    f1_saved = float(tp["metrics"]["f1_macro"])
    f1_recon = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    check("macro-F1 reconstructs from saved predictions",
          abs(f1_saved - f1_recon) < 1e-9,
          f"trainer {f1_saved:.12f} vs reconstructed {f1_recon:.12f}")

    # 4 — spike decomposition is internally consistent
    sa = tp.get("spike_accounting", {})
    parts = (sa.get("input_spikes_per_sample", 0)
             + sa.get("hidden_spikes_per_sample", 0)
             + sa.get("output_spikes_per_sample", 0))
    check("input + hidden + output = total spikes",
          abs(parts - sa.get("total_spikes_per_sample", -1)) < 1e-6,
          f"parts {parts:.6f} vs total {sa.get('total_spikes_per_sample')}")
    check("hidden + output matches the forward pass spike count",
          abs((sa.get("hidden_spikes_per_sample", 0)
               + sa.get("output_spikes_per_sample", 0))
              - tp["spike_stats"]["energy_spikes_per_sample"]) < 1e-3,
          "the v1 'energy_spikes_per_sample' counts hidden+output only")

    # 5 — SOPs are sensible
    sops = sa.get("sops_per_sample", -1)
    check("SOPs/sample nonnegative and at least the hidden fanout cost",
          sops >= 0 and sops >= sa.get("hidden_spikes_per_sample", 0),
          f"sops={sops}")

    # 6 — sample count matches the frozen manifest
    prov = rec.get("provenance", {})
    n_expect = prov.get("n_test_expected")
    check("test sample count equals the manifest",
          n_expect is None or len(y_true) == n_expect,
          f"saved {len(y_true)} vs manifest {n_expect}"
          + ("  (v1 protocol, no manifest)" if n_expect is None else ""))

    # 7 — the run records which frozen split produced it
    check("manifest hash recorded",
          bool(prov.get("manifest_sha256")) or prov.get("manifest") is None,
          f"protocol={prov.get('protocol')} manifest={prov.get('manifest')}")

    # 8 — no sample lost or repeated
    check("sample indices are a complete permutation, no gaps or repeats",
          len(np.unique(idx)) == len(idx) and idx.min() == 0
          and idx.max() == len(idx) - 1,
          f"n={len(idx)} unique={len(np.unique(idx))}")

    # 9 — validation scores usable for fixed-FAR threshold selection
    check("validation scores saved with matching shape",
          val["score"].ndim == 2 and val["score"].shape[1] == C
          and len(val["y_true"]) == val["score"].shape[0],
          f"score {val['score'].shape}, y {val['y_true'].shape}")

    # 10 — early readout actually varies, i.e. the prefix is not degenerate
    f1_t = [f1_score(y_true, cum[:, t, :].astype(np.int32).argmax(axis=1),
                     average="macro", zero_division=0) for t in range(T)]
    check("F1(t) is non-degenerate (varies across t)",
          max(f1_t) - min(f1_t) > 0 or T == 1,
          f"F1(1)={f1_t[0]:.4f} F1(T/2)={f1_t[T//2]:.4f} F1(T)={f1_t[-1]:.4f}")

    width = max(len(n) for n, _, _ in checks)
    print(f"\nrun: {run.relative_to(ROOT)}")
    print(f"protocol: {prov.get('protocol')}  N={len(y_true)}  T={T}  C={C}  "
          f"artifact size={(art / 'test_prefix.npz').stat().st_size / 1e6:.1f} MB\n")
    failed = 0
    for name, ok, detail in checks:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name:<{width}}  {detail}")
        failed += not ok

    f95 = next((t + 1 for t in range(T) if f1_t[t] >= 0.95 * f1_t[-1]), None)
    f99 = next((t + 1 for t in range(T) if f1_t[t] >= 0.99 * f1_t[-1]), None)
    print(f"\n  preview: T95={f95}  T99={f99}  F1(T)={f1_t[-1]:.4f}")

    if failed:
        print(f"\n{failed} check(s) FAILED — do not launch the sweep.")
        return 1
    print("\nall checks passed — instrumentation is safe to sweep on.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
