"""NSL-KDD loader (Phase-1 preprocessing per notebooks/exploration/02_nslkdd_exploration.ipynb).

Steps applied (in order):
  1. Read KDDTrain+.txt, KDDTest+.txt, KDDTest-21.txt with the 43-column schema.
  2. Drop the `difficulty` metadata column.
  3. **log1p** on heavy-skewed numeric columns (byte/count features) — without
     this, MinMax scaling collapses 99 % of the mass to ~0 because src_bytes
     spans 0 → 1.4 B. The notebook explicitly flags this.
  4. One-hot encode {protocol_type, service, flag} against the **union** of
     train+test categories so the feature dimensionality matches.
  5. Map attack labels → 5-class coarse taxonomy {normal, dos, probe, r2l, u2r}.
     All 17 novel KDDTest+ attack types are explicitly placed in the dos / probe
     / r2l / u2r buckets according to standard NSL-KDD literature, so no
     "other" residual class is needed.
  6. MinMax scale to [0, 1] (test clipped to that range).
  7. Stratified 90 / 10 split of KDDTrain+ → (train, val). Validation is used
     for per-epoch monitoring; it does not leak into the final test metrics.

Returns three DataLoaders (train, val, test+, test-21) plus metadata.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset

NSLKDD_COLUMNS = [
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
    "dst_host_srv_rerror_rate", "label", "difficulty",
]
CATEGORICAL = ["protocol_type", "service", "flag"]

# Heavy-tailed numerics — log1p before MinMax to compress the long tail
LOG1P_COLUMNS = [
    "duration", "src_bytes", "dst_bytes",
    "hot", "num_failed_logins", "num_compromised", "num_root",
    "num_file_creations", "num_shells", "num_access_files",
    "count", "srv_count", "dst_host_count", "dst_host_srv_count",
]

ATTACK_TO_CLASS = {
    "normal": "normal",
    # DoS
    "back": "dos", "land": "dos", "neptune": "dos", "pod": "dos",
    "smurf": "dos", "teardrop": "dos", "apache2": "dos", "udpstorm": "dos",
    "processtable": "dos", "worm": "dos", "mailbomb": "dos",
    # Probe
    "ipsweep": "probe", "nmap": "probe", "portsweep": "probe", "satan": "probe",
    "mscan": "probe", "saint": "probe",
    # R2L
    "ftp_write": "r2l", "guess_passwd": "r2l", "imap": "r2l", "multihop": "r2l",
    "phf": "r2l", "spy": "r2l", "warezclient": "r2l", "warezmaster": "r2l",
    "sendmail": "r2l", "named": "r2l", "snmpgetattack": "r2l", "snmpguess": "r2l",
    "xlock": "r2l", "xsnoop": "r2l", "httptunnel": "r2l",
    # U2R
    "buffer_overflow": "u2r", "loadmodule": "u2r", "perl": "u2r", "rootkit": "u2r",
    "ps": "u2r", "sqlattack": "u2r", "xterm": "u2r",
}
CLASS_NAMES = ["normal", "dos", "probe", "r2l", "u2r"]
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASS_NAMES)}


def _label_to_class_idx(label: str) -> int:
    # Any label not in ATTACK_TO_CLASS falls back to r2l (rarely triggered;
    # ATTACK_TO_CLASS already enumerates every label seen in train and test).
    return CLASS_TO_IDX[ATTACK_TO_CLASS.get(label, "r2l")]


@dataclass
class DatasetBundle:
    train_loader: DataLoader
    val_loader: DataLoader
    test_loader: DataLoader            # KDDTest+ (challenging — includes difficulty=21)
    test21_loader: Optional[DataLoader]  # KDDTest-21 (standard — excludes difficulty=21)
    num_features: int
    num_classes: int
    class_names: List[str]
    class_counts_train: dict


def _read_split(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, header=None, names=NSLKDD_COLUMNS)




def load_nslkdd(
    data_dir: str = "dataset/nslkdd",
    batch_size: int = 256,
    val_fraction: float = 0.1,
    num_workers: int = 0,
    seed: int = 0,
) -> DatasetBundle:
    data_dir = Path(data_dir)

    train_df = _read_split(data_dir / "KDDTrain+.txt").drop(columns=["difficulty"])
    test_df = _read_split(data_dir / "KDDTest+.txt").drop(columns=["difficulty"])
    test21_path = data_dir / "KDDTest-21.txt"
    test21_df = _read_split(test21_path).drop(columns=["difficulty"]) if test21_path.exists() else None

    # ---------- label remap (5-class) ----------
    train_df["y"] = train_df["label"].map(_label_to_class_idx).astype(np.int64)
    test_df["y"] = test_df["label"].map(_label_to_class_idx).astype(np.int64)
    if test21_df is not None:
        test21_df["y"] = test21_df["label"].map(_label_to_class_idx).astype(np.int64)

    # ---------- log1p on heavy-tailed numerics ----------
    for col in LOG1P_COLUMNS:
        train_df[col] = np.log1p(train_df[col].clip(lower=0).astype(np.float64))
        test_df[col] = np.log1p(test_df[col].clip(lower=0).astype(np.float64))
        if test21_df is not None:
            test21_df[col] = np.log1p(test21_df[col].clip(lower=0).astype(np.float64))

    # ---------- one-hot over union of categories ----------
    frames = [train_df.assign(_split="train"),
              test_df.assign(_split="test"),
              *([] if test21_df is None else [test21_df.assign(_split="test21")])]
    combined = pd.concat(frames, ignore_index=True)
    combined = pd.get_dummies(combined, columns=CATEGORICAL, dtype=np.float32)

    drop_cols = ["label", "y", "_split"]
    X_all = combined.drop(columns=drop_cols).to_numpy(dtype=np.float32)
    y_all = combined["y"].to_numpy()
    splits = combined["_split"].to_numpy()

    X_train_full = X_all[splits == "train"]
    y_train_full = y_all[splits == "train"]
    X_test = X_all[splits == "test"]
    y_test = y_all[splits == "test"]
    X_test21 = X_all[splits == "test21"] if test21_df is not None else None
    y_test21 = y_all[splits == "test21"] if test21_df is not None else None

    # ---------- stratified train/val split ----------
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full,
        test_size=val_fraction, random_state=seed, stratify=y_train_full,
    )

    # ---------- MinMax scale (fit on train only, transform all splits) ----------
    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train).astype(np.float32)
    X_val = np.clip(scaler.transform(X_val), 0.0, 1.0).astype(np.float32)
    X_test = np.clip(scaler.transform(X_test), 0.0, 1.0).astype(np.float32)
    if X_test21 is not None:
        X_test21 = np.clip(scaler.transform(X_test21), 0.0, 1.0).astype(np.float32)

    def _loader(X, y, shuffle: bool):
        ds = TensorDataset(torch.from_numpy(X), torch.from_numpy(y.astype(np.int64)))
        g = torch.Generator().manual_seed(seed) if shuffle else None
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                          num_workers=num_workers, generator=g, drop_last=False)

    train_loader = _loader(X_train, y_train, shuffle=True)
    val_loader = _loader(X_val, y_val, shuffle=False)
    test_loader = _loader(X_test, y_test, shuffle=False)
    test21_loader = _loader(X_test21, y_test21, shuffle=False) if X_test21 is not None else None

    train_counts = dict(zip(*np.unique(y_train, return_counts=True)))
    train_counts = {CLASS_NAMES[int(k)]: int(v) for k, v in train_counts.items()}

    return DatasetBundle(
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        test21_loader=test21_loader,
        num_features=X_train.shape[1],
        num_classes=len(CLASS_NAMES),
        class_names=CLASS_NAMES,
        class_counts_train=train_counts,
    )
