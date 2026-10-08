"""KDDCup99 loader (same 41-column schema as NSL-KDD).

Key differences vs NSL-KDD:
  * No `difficulty` metadata column.
  * Labels carry a trailing ".", e.g. "normal." — stripped on read.
  * Has ~78% duplicate records (the entire reason NSL-KDD exists). We keep
    duplicates here on purpose: this is the canonical KDDCup99 evaluation
    protocol used by every published baseline. If you want deduped, use the
    NSL-KDD loader.
  * Train split: ``kddcup.data_10_percent_corrected`` (494,021 rows — the
    standard subset, since the full file is ~5M rows and 700 MB).
  * Test split: ``corrected/corrected`` (311,029 rows — contains 17 novel
    attack types not seen in training, same generalisation challenge as
    NSL-KDD's KDDTest+).
  * No KDDTest-21 equivalent.

Everything else (5-class taxonomy, log1p on heavy-tailed numerics, one-hot of
categoricals against the union of train+test, 90/10 stratified train/val,
MinMax to [0, 1]) mirrors ``src/data/nslkdd.py`` so cross-dataset comparison
is one-to-one.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset

from src.data.nslkdd import (
    CATEGORICAL,
    LOG1P_COLUMNS,
    CLASS_NAMES,
    CLASS_TO_IDX,
    ATTACK_TO_CLASS,
    DatasetBundle,
    _label_to_class_idx,
)

# KDDCup99 has 41 features + 1 label — no `difficulty` column.
KDDCUP99_COLUMNS = [
    "duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes",
    "land", "wrong_fragment", "urgent", "hot", "num_failed_logins", "logged_in",
    "num_compromised", "root_shell", "su_attempted", "num_root",
    "num_file_creations", "num_shells", "num_access_files", "num_outbound_cmds",
    "is_host_login", "is_guest_login", "count", "srv_count", "serror_rate",
    "srv_serror_rate", "rerror_rate", "srv_rerror_rate", "same_srv_rate",
    "diff_srv_rate", "srv_diff_host_rate", "dst_host_count",
    "dst_host_srv_count", "dst_host_same_srv_rate", "dst_host_diff_srv_rate",
    "dst_host_same_src_port_rate", "dst_host_srv_diff_host_rate",
    "dst_host_serror_rate", "dst_host_srv_serror_rate", "dst_host_rerror_rate",
    "dst_host_srv_rerror_rate", "label",
]


def _read_split(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, header=None, names=KDDCUP99_COLUMNS)
    # KDDCup99 labels end with "." (e.g. "normal.", "smurf.") — strip it.
    df["label"] = df["label"].str.rstrip(".")
    return df


def load_kddcup99(
    data_dir: str = "dataset/kddcup99",
    batch_size: int = 128,
    val_fraction: float = 0.1,
    num_workers: int = 0,
    seed: int = 0,
) -> DatasetBundle:
    data_dir = Path(data_dir)
    train_path = data_dir / "kddcup.data_10_percent_corrected"
    test_path = data_dir / "corrected" / "corrected"
    if not test_path.exists():
        # Fallback if the user kept the gz-only layout
        test_path = data_dir / "corrected"

    train_df = _read_split(train_path)
    test_df = _read_split(test_path)

    # ---------- label remap (5-class) ----------
    train_df["y"] = train_df["label"].map(_label_to_class_idx).astype(np.int64)
    test_df["y"] = test_df["label"].map(_label_to_class_idx).astype(np.int64)

    # ---------- log1p on heavy-tailed numerics ----------
    for col in LOG1P_COLUMNS:
        train_df[col] = np.log1p(train_df[col].clip(lower=0).astype(np.float64))
        test_df[col] = np.log1p(test_df[col].clip(lower=0).astype(np.float64))

    # ---------- one-hot over union of categories ----------
    combined = pd.concat([train_df.assign(_split="train"),
                           test_df.assign(_split="test")], ignore_index=True)
    combined = pd.get_dummies(combined, columns=CATEGORICAL, dtype=np.float32)

    drop_cols = ["label", "y", "_split"]
    X_all = combined.drop(columns=drop_cols).to_numpy(dtype=np.float32)
    y_all = combined["y"].to_numpy()
    splits = combined["_split"].to_numpy()

    X_train_full = X_all[splits == "train"]
    y_train_full = y_all[splits == "train"]
    X_test = X_all[splits == "test"]
    y_test = y_all[splits == "test"]

    # ---------- stratified 90/10 train/val ----------
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full,
        test_size=val_fraction, random_state=seed, stratify=y_train_full,
    )

    # ---------- MinMax scale (fit on train, transform-and-clip val/test) ----------
    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train).astype(np.float32)
    X_val = np.clip(scaler.transform(X_val), 0.0, 1.0).astype(np.float32)
    X_test = np.clip(scaler.transform(X_test), 0.0, 1.0).astype(np.float32)

    def _loader(X, y, shuffle: bool):
        ds = TensorDataset(torch.from_numpy(X), torch.from_numpy(y.astype(np.int64)))
        g = torch.Generator().manual_seed(seed) if shuffle else None
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                          num_workers=num_workers, generator=g, drop_last=False)

    train_loader = _loader(X_train, y_train, shuffle=True)
    val_loader = _loader(X_val, y_val, shuffle=False)
    test_loader = _loader(X_test, y_test, shuffle=False)

    train_counts = dict(zip(*np.unique(y_train, return_counts=True)))
    train_counts = {CLASS_NAMES[int(k)]: int(v) for k, v in train_counts.items()}

    return DatasetBundle(
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        test21_loader=None,  # KDDCup99 has no Test-21 equivalent
        num_features=X_train.shape[1],
        num_classes=len(CLASS_NAMES),
        class_names=CLASS_NAMES,
        class_counts_train=train_counts,
    )
