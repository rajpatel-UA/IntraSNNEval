"""CIC-IDS2017 loader.

Modern (post-KDD) IDS benchmark from UNB CIC. Eight CSV files capture five
days of CICFlowMeter-extracted flow statistics (~78 numeric features per
flow). Labels span 15 attack categories which we coarse-map to the same
5-class taxonomy as NSL-KDD / KDDCup99 so cross-dataset comparison is direct.

Key differences vs KDD-family
-----------------------------
* **All-numeric features** (no protocol_type / service / flag categoricals) —
  so no one-hot. Resulting dimensionality is 78.
* **No predefined train/test split** — we hold out a stratified 30% as test
  and do the usual 90/10 train/val on the remaining 70%.
* **Data hygiene issues** known to the literature: leading whitespace on
  column names, ``Inf`` / NaN in Flow Bytes/s + Flow Packets/s when duration
  is 0, UTF-8 replacement byte (``\\xef\\xbf\\xbd``) in "Web Attack ?" labels.
  All three are handled here.

5-class mapping (15 raw labels → {normal, dos, probe, r2l, u2r})
----------------------------------------------------------------
* normal: BENIGN
* dos   : DDoS, DoS Hulk, DoS GoldenEye, DoS slowloris, DoS Slowhttptest, Bot
* probe : PortScan
* r2l   : FTP-Patator, SSH-Patator, Heartbleed
* u2r   : Web Attack (BF / SQLi / XSS), Infiltration

Bot is bucketed into dos (network-level flooding behaviour); Heartbleed into
r2l (unauthorised data access via remote vulnerability); web attacks and
Infiltration into u2r (exploitation / code execution).
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset

from src.data.nslkdd import DatasetBundle, CLASS_NAMES, CLASS_TO_IDX

CIC_FILES = [
    "Monday-WorkingHours.pcap_ISCX.csv",
    "Tuesday-WorkingHours.pcap_ISCX.csv",
    "Wednesday-workingHours.pcap_ISCX.csv",
    "Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv",
    "Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv",
    "Friday-WorkingHours-Morning.pcap_ISCX.csv",
    "Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv",
    "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv",
]

# Strip leading ASCII or UTF-8 replacement (\xef\xbf\xbd → "?") characters from labels.
_WEB_RE = re.compile(r"Web Attack[^A-Za-z]+", flags=re.UNICODE)

LABEL_TO_CLASS = {
    "BENIGN": "normal",
    "DDoS": "dos",
    "DoS Hulk": "dos",
    "DoS GoldenEye": "dos",
    "DoS slowloris": "dos",
    "DoS Slowhttptest": "dos",
    "Bot": "dos",
    "PortScan": "probe",
    "FTP-Patator": "r2l",
    "SSH-Patator": "r2l",
    "Heartbleed": "r2l",
    "Web Attack Brute Force": "u2r",
    "Web Attack XSS": "u2r",
    "Web Attack Sql Injection": "u2r",
    "Infiltration": "u2r",
}


def _normalise_label(s: str) -> str:
    s = s.strip()
    # "Web Attack \xef\xbf\xbd Brute Force" → "Web Attack Brute Force"
    if s.startswith("Web Attack"):
        s = _WEB_RE.sub("Web Attack ", s).strip()
    return s


def _label_to_class_idx(label: str) -> int:
    cls = LABEL_TO_CLASS.get(_normalise_label(label))
    if cls is None:
        # Anything we don't recognise falls into the most attack-generic bucket
        # so it doesn't get silently dropped; in practice this is empty after
        # the normalisation above on the 8 published CSVs.
        cls = "r2l"
    return CLASS_TO_IDX[cls]


def _read_one(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)
    df.columns = [c.strip() for c in df.columns]
    return df


def _build_full_arrays(data_dir: Path):
    """Heavy CSV → numpy step. Result is cached to disk; on a second call it
    loads in <1 s instead of ~6 min."""
    cache_path = data_dir / "_preprocessed_cache.npz"
    if cache_path.exists():
        z = np.load(cache_path)
        return z["X"].astype(np.float32), z["y"].astype(np.int64)

    frames = []
    for fname in CIC_FILES:
        path = data_dir / fname
        if not path.exists():
            raise FileNotFoundError(f"missing CIC-IDS2017 file: {path}")
        frames.append(_read_one(path))
    df = pd.concat(frames, ignore_index=True)

    df["Label"] = df["Label"].astype(str).map(_normalise_label)
    y = df["Label"].map(_label_to_class_idx).astype(np.int64).to_numpy()

    X_df = df.drop(columns=["Label"])
    X_df = X_df.apply(pd.to_numeric, errors="coerce")
    X_df = X_df.replace([np.inf, -np.inf], np.nan).fillna(0.0).astype(np.float32)
    X = np.log1p(np.clip(X_df.to_numpy(), 0.0, None)).astype(np.float32)

    np.savez_compressed(cache_path, X=X, y=y)
    return X, y


def load_cicids2017(
    data_dir: str = "dataset/cicids2017",
    batch_size: int = 128,
    val_fraction: float = 0.1,
    test_fraction: float = 0.3,
    subsample_fraction: float = 1.0,
    num_workers: int = 0,
    seed: int = 0,
) -> DatasetBundle:
    data_dir = Path(data_dir)
    X, y = _build_full_arrays(data_dir)

    # ---------- optional subsample (for smoke tests / time budget) ----------
    if subsample_fraction < 1.0:
        rng = np.random.RandomState(seed)
        n_keep = int(len(X) * subsample_fraction)
        idx = rng.choice(len(X), n_keep, replace=False)
        X, y = X[idx], y[idx]

    # ---------- 70/30 stratified train_pool / test ----------
    X_train_pool, X_test, y_train_pool, y_test = train_test_split(
        X, y, test_size=test_fraction, random_state=seed, stratify=y,
    )
    # ---------- 90/10 stratified train / val ----------
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_pool, y_train_pool,
        test_size=val_fraction, random_state=seed, stratify=y_train_pool,
    )

    # ---------- MinMax (fit on train, transform-and-clip the rest) ----------
    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train).astype(np.float32)
    X_val = np.clip(scaler.transform(X_val), 0.0, 1.0).astype(np.float32)
    X_test = np.clip(scaler.transform(X_test), 0.0, 1.0).astype(np.float32)

    def _loader(Xs, ys, shuffle: bool):
        ds = TensorDataset(torch.from_numpy(Xs), torch.from_numpy(ys.astype(np.int64)))
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
        test21_loader=None,
        num_features=X_train.shape[1],
        num_classes=len(CLASS_NAMES),
        class_names=CLASS_NAMES,
        class_counts_train=train_counts,
    )
