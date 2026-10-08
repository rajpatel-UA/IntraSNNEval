"""v2 (leakage-resistant) evaluation protocols for the four conference datasets.

The submitted paper's protocol — call it **v1** — is kept intact in the original
loaders so the published numbers stay reproducible. v2 changes exactly four
things, each in response to a specific review point:

  1. **Nothing is fitted on test.** Categorical vocabularies, log1p and MinMax
     are fitted on training rows only (`train_only.encode_train_only`). v1
     one-hot encoded against the union of all splits.
  2. **The partition is frozen independently of the model seed.** In v1,
     CIC-IDS2017 and CTU-13 passed `random_state=seed` to `train_test_split`, so
     seeds 42–46 measured partition variance *and* optimisation variance mixed
     together. In v2 the partition is a property of the protocol (and, where
     applicable, the fold); `seed` only initialises weights and shuffles batches.
  3. **Random flow-level splitting is replaced by group-respecting splitting**
     where the benchmark has a group structure: capture file for CIC-IDS2017,
     scenario for CTU-13.
  4. **Every split is audited before it is used** (`audit.audit_split`), and a
     protocol that claims group-disjointness and is not gets an exception rather
     than a plausible-looking number.

Protocol registry
-----------------
======================================  ============================================
name                                    partition
======================================  ============================================
``nslkdd_v2``                           official KDDTrain+/KDDTest+ (+ Test-21)
``kddcup99_v2``                         official 10%-corrected / corrected
``cicids2017_v2``                       within-capture chronological 70/10/20
``cicids2017_daydisjoint``              train Mon–Wed / test Thu–Fri, BINARY
``ctu13_scenario_f{0,1,2,3}``           scenario-disjoint frozen folds, causal
                                        host aggregates
======================================  ============================================

A note on CIC-IDS2017 ordering: the public Kaggle redistribution used here ships
78 features and the label, and **drops the ``Timestamp`` column**. Within a
capture file we therefore order by row position, which is CICFlowMeter's write
order and chronological up to flow-completion reordering. That is a proxy, and
the paper must call it one. The day-level grouping, which carries the actual
temporal claim, is exact — it comes from the file names.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

from src.data.audit import audit_split, frame_hash
from src.data.cicids2017 import CIC_FILES, _label_to_class_idx as _cic_label
from src.data.cicids2017 import _read_one as _cic_read
from src.data.nslkdd import (CATEGORICAL, CLASS_NAMES, DatasetBundle,
                             NSLKDD_COLUMNS, _label_to_class_idx as _kdd_label)
from src.data.kddcup99 import KDDCUP99_COLUMNS
from src.data.train_only import encode_train_only
from src.paths import RESULTS_ROOT, dataset_dir

#: Val fraction carved out of the training pool, and the fixed RNG that carves
#: it. Deliberately NOT the model seed — see module docstring point 2.
VAL_FRACTION = 0.10
PARTITION_SEED = 20260812

CACHE = RESULTS_ROOT / "cache"
MANIFEST_DIR = RESULTS_ROOT / "manifests"

#: Capture files grouped into the two halves of the day-disjoint protocol.
CIC_EARLY_DAYS = CIC_FILES[:3]      # Monday, Tuesday, Wednesday
CIC_LATE_DAYS = CIC_FILES[3:]       # Thursday ×2, Friday ×3


# --------------------------------------------------------------------------
# shared plumbing
# --------------------------------------------------------------------------
def _loader(X: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool,
            seed: int, num_workers: int = 0) -> DataLoader:
    ds = TensorDataset(torch.from_numpy(X), torch.from_numpy(y.astype(np.int64)))
    g = torch.Generator().manual_seed(seed) if shuffle else None
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                      num_workers=num_workers, generator=g, drop_last=False)


def _bundle(Xtr, ytr, Xva, yva, Xte, yte, class_names, batch_size, seed,
            num_workers=0, test21=None) -> DatasetBundle:
    u, c = np.unique(ytr, return_counts=True)
    t21 = None
    if test21 is not None:
        t21 = _loader(test21[0], test21[1], batch_size, False, seed, num_workers)
    return DatasetBundle(
        train_loader=_loader(Xtr, ytr, batch_size, True, seed, num_workers),
        val_loader=_loader(Xva, yva, batch_size, False, seed, num_workers),
        test_loader=_loader(Xte, yte, batch_size, False, seed, num_workers),
        test21_loader=t21,
        num_features=int(Xtr.shape[1]),
        num_classes=len(class_names),
        class_names=list(class_names),
        class_counts_train={class_names[int(k)]: int(v) for k, v in zip(u, c)},
    )


def _carve_val(pool: np.ndarray, y_pool: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Stratified val carve-out from a training pool, keyed on PARTITION_SEED."""
    from sklearn.model_selection import train_test_split
    tr, va = train_test_split(pool, test_size=VAL_FRACTION,
                              random_state=PARTITION_SEED, stratify=y_pool)
    return np.sort(tr), np.sort(va)


def _emit_manifest(audit, extra: dict) -> None:
    audit.extra.update(extra)
    name = f"{audit.dataset}_{audit.protocol}"
    if audit.fold is not None:
        name += f"_f{audit.fold}"
    audit.to_json(MANIFEST_DIR / f"{name}.json")


def _publish_eval_context(feat: pd.DataFrame, tr, te, groups=None,
                          group_name: str = None) -> None:
    """Hand the metric layer the duplicate mask and (optionally) test groups.

    Costs one hash pass and makes the duplicate-free sensitivity block appear in
    every result record for free. It matters most on KDDCup99, where 57.5% of
    the official test set occurs verbatim in training: without this, every
    KDDCup99 number in the paper is reported on a test set that is mostly
    memorisable, and the manuscript could not say by how much.
    """
    from src.metrics import set_eval_context
    h = pd.util.hash_pandas_object(feat, index=False).to_numpy()
    dupfree = ~np.asarray(pd.Index(h[te]).isin(pd.Index(h[tr])), dtype=bool)
    set_eval_context(test_dupfree_mask=dupfree,
                     test_groups=None if groups is None else groups[te],
                     group_name=group_name)


# --------------------------------------------------------------------------
# KDD family — official splits, train-only encoding
# --------------------------------------------------------------------------
def _kdd_frames(dataset: str):
    d = dataset_dir(dataset)
    if dataset == "nslkdd":
        cols = NSLKDD_COLUMNS
        rd = lambda p: pd.read_csv(p, header=None, names=cols).drop(columns=["difficulty"])
        parts = [("train", rd(d / "KDDTrain+.txt")), ("test", rd(d / "KDDTest+.txt"))]
        if (d / "KDDTest-21.txt").exists():
            parts.append(("test21", rd(d / "KDDTest-21.txt")))
    else:
        cols = KDDCUP99_COLUMNS
        def rd(p):
            f = pd.read_csv(p, header=None, names=cols)
            f["label"] = f["label"].str.rstrip(".")
            return f
        test = d / "corrected" / "corrected"
        if not test.exists():
            test = d / "corrected"
        parts = [("train", rd(d / "kddcup.data_10_percent_corrected")),
                 ("test", rd(test))]
    for name, f in parts:
        f["_split"] = name
    return pd.concat([f for _, f in parts], ignore_index=True)


def load_kdd_v2(dataset: str, batch_size: int = 128, seed: int = 42,
                num_workers: int = 0, **_ignored) -> DatasetBundle:
    """NSL-KDD / KDDCup99 on their official splits with train-only encoding."""
    df = _kdd_frames(dataset)
    y = df["label"].map(_kdd_label).astype(np.int64).to_numpy()

    feat = df.drop(columns=["label", "_split"])
    for c in CATEGORICAL:
        feat[c] = feat[c].astype(str)          # forces the train-only one-hot path

    split = df["_split"].to_numpy()
    pool = np.where(split == "train")[0]
    tr, va = _carve_val(pool, y[pool])
    te = np.where(split == "test")[0]
    te21 = np.where(split == "test21")[0] if (split == "test21").any() else None

    audit = audit_split(dataset, "v2", feat, y, tr, va, te, group_disjoint=False)
    _emit_manifest(audit, {"partition_seed": PARTITION_SEED,
                           "feature_hash": frame_hash(feat),
                           "official_split": True})

    Xtr, Xva, Xrest, meta = encode_train_only(
        feat, tr, va, te if te21 is None else np.concatenate([te, te21]))
    if te21 is not None:
        Xte, Xte21 = Xrest[:len(te)], Xrest[len(te):]
        t21 = (Xte21, y[te21])
    else:
        Xte, t21 = Xrest, None

    _publish_eval_context(feat, tr, te)
    print(f"  [{dataset}/v2] tr={len(tr)} va={len(va)} te={len(te)} "
          f"features={meta['n_features']} unk={meta['unknown_category_rows']}",
          flush=True)
    return _bundle(Xtr, y[tr], Xva, y[va], Xte, y[te], CLASS_NAMES,
                   batch_size, seed, num_workers, test21=t21)


# --------------------------------------------------------------------------
# CIC-IDS2017 — capture-file structure preserved
# --------------------------------------------------------------------------
def _cic_frame() -> pd.DataFrame:
    """Raw CIC-IDS2017 with the capture file retained as a group column.

    The production v1 cache stores only (X, y); the capture identity is exactly
    what the strict protocols partition on, so v2 keeps its own cache.
    """
    cache = CACHE / "cicids2017_grouped.parquet"
    if cache.exists():
        return pd.read_parquet(cache)

    d = dataset_dir("cicids2017")
    frames = []
    for i, fname in enumerate(CIC_FILES):
        f = _cic_read(d / fname)
        f["_capture"] = fname
        f["_day"] = fname.split("-")[0]
        f["_capture_idx"] = i
        f["_row_in_capture"] = np.arange(len(f), dtype=np.int64)
        frames.append(f)
    df = pd.concat(frames, ignore_index=True)
    df["y"] = df["Label"].astype(str).map(_cic_label).astype(np.int64)
    df = df.drop(columns=["Label"])
    cache.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(cache, index=False)
    return df


_CIC_META = ["_capture", "_day", "_capture_idx", "_row_in_capture", "y"]


def load_cicids2017_v2(batch_size: int = 128, seed: int = 42,
                       num_workers: int = 0, **_ignored) -> DatasetBundle:
    """Within-capture chronological 70/10/20. No flow is split away from its
    burst by shuffling, and the partition does not depend on the model seed."""
    df = _cic_frame()
    y = df["y"].to_numpy(np.int64)
    feat = df.drop(columns=_CIC_META)

    tr, va, te = [], [], []
    for _, g in df.groupby("_capture_idx", sort=True):
        pos = g.index.to_numpy()[np.argsort(g["_row_in_capture"].to_numpy())]
        n = len(pos)
        a, b = int(0.70 * n), int(0.80 * n)
        tr.append(pos[:a]); va.append(pos[a:b]); te.append(pos[b:])
    tr, va, te = (np.sort(np.concatenate(x)) for x in (tr, va, te))

    audit = audit_split("cicids2017", "v2", feat, y, tr, va, te,
                        groups=df["_capture"].to_numpy(), group_name="capture",
                        group_disjoint=False)
    _emit_manifest(audit, {"ordering": "row position within capture "
                                       "(Timestamp absent from this redistribution)",
                           "feature_hash": frame_hash(feat),
                           "fractions": [0.70, 0.10, 0.20]})

    _publish_eval_context(feat, tr, te, df["_capture"].to_numpy(), "capture")
    Xtr, Xva, Xte, meta = encode_train_only(feat, tr, va, te)
    print(f"  [cicids2017/v2] tr={len(tr)} va={len(va)} te={len(te)} "
          f"features={meta['n_features']} dup_test_in_train="
          f"{audit.duplicate_test_fraction:.4f}", flush=True)
    return _bundle(Xtr, y[tr], Xva, y[va], Xte, y[te], CLASS_NAMES,
                   batch_size, seed, num_workers)


def load_cicids2017_daydisjoint(batch_size: int = 128, seed: int = 42,
                                num_workers: int = 0, **_ignored) -> DatasetBundle:
    """Train on Mon–Wed, test on Thu–Fri. **Binary** normal-vs-attack.

    Multi-class is not available under this partition and the paper must not
    pretend otherwise: the 5-class taxonomy puts Probe (PortScan, Friday) and
    U2R (web attacks + Infiltration, Thursday) entirely in the late half and R2L
    (Patator, Heartbleed, Tuesday) entirely in the early half, so three of five
    classes would be untrainable or untestable. Collapsed to benign-vs-attack the
    partition is a clean unseen-day generalisation test, which is the question
    the review is actually asking.
    """
    df = _cic_frame()
    y_bin = (df["y"].to_numpy(np.int64) != 0).astype(np.int64)
    feat = df.drop(columns=_CIC_META)

    early = df["_capture"].isin(CIC_EARLY_DAYS).to_numpy()
    pool = np.where(early)[0]
    tr, va = _carve_val(pool, y_bin[pool])
    te = np.where(~early)[0]

    audit = audit_split("cicids2017", "daydisjoint", feat, y_bin, tr, va, te,
                        groups=df["_day"].to_numpy(), group_name="day",
                        group_disjoint=False)   # val shares days with train by design
    _emit_manifest(audit, {"train_days": sorted(set(df.loc[early, "_day"])),
                           "test_days": sorted(set(df.loc[~early, "_day"])),
                           "task": "binary normal-vs-attack",
                           "feature_hash": frame_hash(feat)})

    _publish_eval_context(feat, tr, te, df["_day"].to_numpy(), "day")
    Xtr, Xva, Xte, meta = encode_train_only(feat, tr, va, te)
    print(f"  [cicids2017/daydisjoint] tr={len(tr)} va={len(va)} te={len(te)} "
          f"features={meta['n_features']}", flush=True)
    return _bundle(Xtr, y_bin[tr], Xva, y_bin[va], Xte, y_bin[te],
                   ["normal", "attack"], batch_size, seed, num_workers)
