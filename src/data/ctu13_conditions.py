"""CTU-13 three-condition protocols as first-class datasets.

Registers the three conditions established by an offline diagnostic, so the SNN
trainer consumes them unchanged. That diagnostic was a one-off measurement of the
partition itself, not a baseline model, and no non-spiking model is trained or
reported anywhere in this study:

  legacy_random     global per-scenario aggregates + random flow split
  causal_random     partition-local causal aggregates + random flow split
  causal_scenario   causal aggregates + scenario-disjoint split (frozen manifest)

Two design points that are easy to get wrong:

1. `causal_scenario` legitimately uses the precomputed `causal__*` columns. Those
   are built from strictly earlier flows *within a scenario*, and a whole test
   scenario lies entirely inside the test partition, so no value crosses the
   boundary. `causal_random` cannot use them -- a random split cuts across
   scenarios -- so it recomputes aggregates inside each partition.

2. The random partition is keyed on the FOLD, never on the optimisation seed.
   Folds are the statistical unit; seeds measure optimisation variability within a
   fold. This mirrors the TON-IoT leakage protocol so the two are comparable.

Duplicates are NOT removed before partitioning. That would change the frozen
partitions. Instead the test rows whose feature hash also occurs in training are
flagged, and `compute_metrics` reports a duplicate-free sensitivity block
alongside the full result. The overlap is reported, never claimed to be zero.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

from src.data.nslkdd import DatasetBundle
from src.data.ctu13_causal import HOST_COLS, METADATA_COLS, load_frozen_folds

from src.paths import CONFERENCE_ROOT, dataset_dir

# The three-condition artifact is a *build product*, not raw data, so it belongs
# under results/. The dataset-dir fallback exists because the parent HiSNN
# checkout built it there first and rebuilding costs ~20 min for no gain.
PQ_LOCAL = CONFERENCE_ROOT / "results/cache/ctu13_three_condition.parquet"
PQ = PQ_LOCAL if PQ_LOCAL.exists() else dataset_dir("ctu13") / "_three_condition.parquet"
FOLDS_JSON = CONFERENCE_ROOT / "results/manifests/ctu13_scenario_folds.json"

BASE_COLS = ["Dur", "TotPkts", "TotBytes", "SrcBytes", "sTos", "dTos",
             "Sport", "Dport"]
LEG = [f"legacy__{c}" for c in HOST_COLS]
CAU = [f"causal__{c}" for c in HOST_COLS]
CONDITIONS = ("legacy_random", "causal_random", "causal_scenario")
FEATURE_SETS = ("base", "combined")


def _random_partition(n: int, fold: int) -> tuple:
    """Fold-keyed random split, identical to the 36-run diagnostic (rng 1000+fold)
    so the matrix and the diagnostic describe the same partition."""
    rng = np.random.default_rng(1000 + int(fold))
    perm = rng.permutation(n)
    nte, nva = int(0.24 * n), int(0.08 * n)
    return perm[nte + nva:], perm[nte:nte + nva], perm[:nte]


def load_ctu13_condition(condition: str, fold: int, feature_set: str = "combined",
                         batch_size: int = 128, seed: int = 42,
                         num_workers: int = 0, **_ignored) -> DatasetBundle:
    from src.data.train_only import encode_train_only
    from src.metrics import set_eval_context

    if condition not in CONDITIONS:
        raise ValueError(f"unknown condition {condition!r}; expected {CONDITIONS}")
    if feature_set not in FEATURE_SETS:
        raise ValueError(f"unknown feature_set {feature_set!r}; expected {FEATURE_SETS}")

    d = pd.read_parquet(PQ)
    fo = load_frozen_folds(FOLDS_JSON, d)      # verifies sizes, never re-solves
    f = fo["folds"][int(fold)]
    n = len(d)
    y = d["y"].to_numpy(np.int64)

    if condition == "causal_scenario":
        tr = np.where(d.scenario.isin(f["train_scenarios"]))[0]
        va = np.where(d.scenario.isin(f["val_scenarios"]))[0]
        te = np.where(d.scenario.isin(f["test_scenarios"]))[0]
        host_frame = d[CAU]
    else:
        tr, va, te = _random_partition(n, fold)
        if condition == "legacy_random":
            host_frame = d[LEG]
        else:
            from src.data.ctu13_causal import partition_local_causal as _partition_local_causal
            host_frame = _partition_local_causal(d, (tr, va, te), CAU)

    cols_note = "base" if feature_set == "base" else "base + host-context"
    feat = d[BASE_COLS] if feature_set == "base" else pd.concat(
        [d[BASE_COLS], host_frame.set_axis(CAU, axis=1)], axis=1)
    feat = feat.reset_index(drop=True)
    assert set(METADATA_COLS).isdisjoint(feat.columns), "metadata reached the features"

    # Duplicate-free sensitivity mask: test rows whose raw feature vector also
    # occurs in training. Hash-based, so it is an upper bound on true duplicates.
    h = pd.util.hash_pandas_object(feat, index=False).to_numpy()
    dupfree = ~np.asarray(pd.Index(h[te]).isin(pd.Index(h[tr])), dtype=bool)

    # Audit + manifest, so CTU-13 appears in the same Gate-1 table as the rest.
    # `scenario` is disjoint by construction under causal_scenario; the SOURCE
    # HOST is NOT, and the manifest records that explicitly. Fold 1 puts 76% of
    # its test hosts back in training, which is exactly why calling this
    # protocol "host-disjoint" anywhere in the paper would be false.
    from src.data.audit import audit_split, frame_hash
    from src.data.protocols import _emit_manifest
    _audit = audit_split(
        "ctu13", condition, feat, y, tr, va, te,
        groups=d["scenario"].to_numpy(), group_name="scenario",
        group_disjoint=(condition == "causal_scenario"), fold=int(fold))
    _hosts = d["SrcAddr"].astype(str).to_numpy()
    _te_hosts, _tr_hosts = set(_hosts[te]), set(_hosts[tr])
    _emit_manifest(_audit, {
        "feature_set": feature_set,
        "aggregation": "partition-local causal" if condition != "legacy_random"
                       else "whole-scenario (leaky)",
        "feature_hash": frame_hash(feat),
        "host_disjoint": False,
        "test_hosts_also_in_train_pct": round(
            100 * len(_te_hosts & _tr_hosts) / max(len(_te_hosts), 1), 2),
        "test_scenarios": sorted({int(v) for v in d["scenario"].to_numpy()[te]}),
        "dupfree_excluded_rows": int((~dupfree).sum()),
    })

    Xtr, Xva, Xte, meta = encode_train_only(feat, tr, va, te)

    set_eval_context(test_groups=d["scenario"].to_numpy()[te],
                     test_dupfree_mask=dupfree,
                     group_name="scenario")

    print(f"  [ctu13/{condition}/fold{fold}/{feature_set}] "
          f"rows tr={len(tr)} va={len(va)} te={len(te)} | features {meta['n_features']} "
          f"({cols_note}) | test scenarios {sorted(set(d.scenario.to_numpy()[te].tolist()))} "
          f"| dup-free excludes {int((~dupfree).sum())} "
          f"({100*(~dupfree).mean():.4f}%)", flush=True)

    def _loader(Xs, ys, shuffle):
        ds = TensorDataset(torch.from_numpy(Xs), torch.from_numpy(ys.astype(np.int64)))
        g = torch.Generator().manual_seed(seed) if shuffle else None
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                          num_workers=num_workers, generator=g, drop_last=False)

    u, c = np.unique(y[tr], return_counts=True)
    classes = ["normal", "botnet"]
    return DatasetBundle(
        train_loader=_loader(Xtr, y[tr], True),
        val_loader=_loader(Xva, y[va], False),
        test_loader=_loader(Xte, y[te], False),
        test21_loader=None,
        num_features=Xtr.shape[1],
        num_classes=2,
        class_names=classes,
        class_counts_train={classes[int(k)]: int(v) for k, v in zip(u, c)},
    )
