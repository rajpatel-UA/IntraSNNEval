#!/usr/bin/env python3
"""Tabulate the 60-run CTU-13 three-condition comparison.

Writes results/v1_submitted/ctu13_strict_vanilla.csv, one row per (condition,
fold, seed), from each run's results.json. `make_frozen_numbers.py` reads the
\\FNCtuLegacy, \\FNCtuCausalRandom and \\FNCtuStrict sequence, the fold means and
the 20-evaluation SD from it.

Columns are as the parent repository's `scripts/ctu13_matrix_analysis.py`
defined them, in percent: macro-F1 and MCC from the stored metrics, DR and FAR
from the binary normal-versus-botnet view, and the duplicate-free sensitivity
(macro-F1 with test rows that occur verbatim in training removed, its change
in pp, and the share of test rows removed). `features` and `model` are constant
("combined": raw flow fields plus per-source-host statistics; "vanilla": the
LeakyParallel/latency SNN). They are kept because `make_frozen_numbers.py`
filters on them and the committed file carries them.

    python scripts/ctu13_strict_vanilla.py
    python scripts/ctu13_strict_vanilla.py --runs-dir results/runs_ctu13_conditions \\
        --out results/runs_ctu13_conditions/ctu13_strict_vanilla.csv

The default `--runs-dir` holds the original runs' records. The runs themselves
are produced by `run_ctu13_conditions.py`, whose output directory is the second
form above.
"""
from __future__ import annotations

import argparse
import io
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "results/v1_submitted/ctu13_conditions"
OUT = ROOT / "results/v1_submitted/ctu13_strict_vanilla.csv"


def row(path: Path, runs_dir: Path) -> dict:
    condition, fold, seed = path.relative_to(runs_dir).parts[:3]
    m = json.loads(path.read_text())["test_plus"]
    met, b = m["metrics"], m.get("binary_ids_metrics") or {}
    dup = met.get("duplicate_free")
    return {"condition": condition, "features": "combined", "model": "vanilla",
            "fold": int(fold.removeprefix("fold")),
            "seed": int(seed.removeprefix("seed_")),
            "macro_f1": 100 * met["f1_macro"],
            "mcc": 100 * met["matthews_corrcoef"],
            "DR": 100 * b.get("detection_rate", np.nan),
            "FAR": 100 * b.get("false_alarm_rate", np.nan),
            "dupfree_macro_f1": 100 * dup["metrics"]["macro_f1"] if dup else np.nan,
            "dupfree_delta_pp": dup["delta_macro_f1_pp"] if dup else np.nan,
            "dupfree_excluded_pct": dup["pct_excluded"] if dup else np.nan}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-dir", type=Path, default=RUNS)
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()
    runs_dir = args.runs_dir if args.runs_dir.is_absolute() else ROOT / args.runs_dir
    out = args.out if args.out.is_absolute() else ROOT / args.out

    paths = sorted(runs_dir.glob("*/fold*/seed_*/results.json"))
    if not paths:
        print(f"no runs under {runs_dir}; nothing to analyse")
        return 0
    d = (pd.DataFrame([row(p, runs_dir) for p in paths])
         .sort_values(["condition", "fold", "seed"]).reset_index(drop=True))
    # The committed file was cut from a larger table after that table had been
    # written and read back once with pandas' default float parser, which is
    # fast but not round-trip exact. Repeating that one write/read reproduces
    # the file byte for byte. The values pandas reads are the same either way;
    # only the last printed digit of some cells differs.
    out.write_text(pd.read_csv(io.StringIO(d.to_csv(index=False)))
                   .to_csv(index=False))

    print(f"{len(d)} runs -> {out}")
    by = d.groupby(["condition", "fold"]).macro_f1.mean().unstack()
    print(by.round(2).to_string())
    print(d.groupby("condition").macro_f1.mean().round(2).to_string())
    return 0


if __name__ == "__main__":
    sys.exit(main())
