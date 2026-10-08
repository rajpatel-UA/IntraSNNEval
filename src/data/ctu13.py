"""CTU-13 loader — per-flow classification with per-source-IP context features.

Re-engineered from the dhoogla minimal-feature parquet (~78 % ceiling) to use
the original Stratosphere ``.binetflow`` CSVs (~2.5 GB, 13 scenarios). The
key change vs the dhoogla loader: every flow is enriched with **host-level
behavioural statistics** computed from all flows originating at the same
SrcAddr in the same scenario. This is the feature engineering that gets
published CTU-13 papers above 90 %:

  * Kim 2020 (RVAE, F1=0.929) groups by SrcAddr in 60 s windows;
  * McDermott 2019 (LSTM-graph, Acc=0.962) per-host graph centralities;
  * Joshi 2021 (Decision Tree, F1=0.85) per-flow stats only — our floor.

Why per-flow + host context (not per-host-window like RVAE)?
-----------------------------------------------------------
Pure per-host-window aggregation gives only ~hundreds of samples because
CTU-13 has very few active hosts (we tried 60 s windows → 283 samples,
useless). The host-level *signal* lives in those few hosts' aggregate
behaviour, so we replicate that signal onto every flow as additional
context features. Each flow becomes the unit of classification, ~800 K
samples, every one of them tagged with the source's all-scenario stats.

Pipeline
--------
1.  Read 13 .binetflow files from ``dataset/ctu13/binetflow/scenario-*.binetflow``.
2.  Drop ``Background`` rows (per canonical CTU-13 protocol).
3.  Per (scenario, SrcAddr) compute aggregate stats:
        total flows, unique DstAddrs, unique Dports, unique Sports,
        unique protos, unique states,
        sum/mean/std of (Dur, TotPkts, TotBytes, SrcBytes),
        Shannon entropy of Dport / Proto.
4.  Merge those 17 host-context columns onto every flow.
5.  One-hot proto (3), dir (5), and add the host-context numerics + per-flow
    numerics (Dur, TotPkts, TotBytes, SrcBytes, Sport, Dport, sTos, dTos).
6.  log1p the heavy-tailed columns, MinMax to [0, 1], stratified 70/30
    train_pool/test + 90/10 train/val.

Resulting feature dimensionality: ~32.

Disk cache
----------
First load reads 2.6 GB of CSVs, aggregates per source, joins back to flows,
log1p+MinMax — about 5–10 min. Cached at
``dataset/ctu13/_preprocessed_cache.npz``; subsequent loads <15 s.
"""
from __future__ import annotations

import glob
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset

from src.data.nslkdd import DatasetBundle

CLASS_NAMES = ["normal", "botnet"]
CLASS_TO_IDX = {"normal": 0, "botnet": 1}

BINETFLOW_COLS = [
    "StartTime", "Dur", "Proto", "SrcAddr", "Sport", "Dir",
    "DstAddr", "Dport", "State", "sTos", "dTos",
    "TotPkts", "TotBytes", "SrcBytes", "Label",
]

# Per-flow features kept as numerics (everything else dropped or one-hot)
FLOW_NUMERICS = ["Dur", "TotPkts", "TotBytes", "SrcBytes", "Sport", "Dport",
                  "sTos", "dTos"]
# Per-flow categoricals → one-hot
FLOW_CATS = ["Proto", "Dir"]
# Per-source-IP aggregate columns to compute (joined onto every flow)
HOST_AGG_NUMERICS = [
    "src_flow_count", "src_unique_dsts", "src_unique_dports", "src_unique_sports",
    "src_unique_protos", "src_unique_states",
    "src_total_dur", "src_mean_dur", "src_std_dur",
    "src_total_pkts", "src_mean_pkts",
    "src_total_bytes", "src_mean_bytes",
    "src_total_src_bytes", "src_mean_src_bytes",
    "src_port_entropy", "src_proto_entropy",
]
LOG1P_COLS = (
    ["Dur", "TotPkts", "TotBytes", "SrcBytes", "Sport", "Dport"]
    + ["src_flow_count", "src_unique_dsts", "src_unique_dports", "src_unique_sports",
       "src_total_dur", "src_mean_dur", "src_std_dur",
       "src_total_pkts", "src_mean_pkts",
       "src_total_bytes", "src_mean_bytes",
       "src_total_src_bytes", "src_mean_src_bytes"]
)


def _shannon_entropy(s: pd.Series) -> float:
    vc = s.value_counts(normalize=True)
    return float(-(vc * np.log2(vc.replace(0, np.nan))).fillna(0).sum())


def _aggregate_per_source(scenario_df: pd.DataFrame) -> pd.DataFrame:
    """For one scenario, compute per-SrcAddr aggregate stats."""
    g = scenario_df.groupby("SrcAddr", sort=False, observed=True)
    agg = g.agg(
        src_flow_count=("Dur", "count"),
        src_unique_dsts=("DstAddr", "nunique"),
        src_unique_dports=("Dport", "nunique"),
        src_unique_sports=("Sport", "nunique"),
        src_unique_protos=("Proto", "nunique"),
        src_unique_states=("State", "nunique"),
        src_total_dur=("Dur", "sum"),
        src_mean_dur=("Dur", "mean"),
        src_std_dur=("Dur", "std"),
        src_total_pkts=("TotPkts", "sum"),
        src_mean_pkts=("TotPkts", "mean"),
        src_total_bytes=("TotBytes", "sum"),
        src_mean_bytes=("TotBytes", "mean"),
        src_total_src_bytes=("SrcBytes", "sum"),
        src_mean_src_bytes=("SrcBytes", "mean"),
    )
    agg["src_std_dur"] = agg["src_std_dur"].fillna(0)
    # entropy features — single pass over groups
    agg["src_port_entropy"] = g["Dport"].apply(_shannon_entropy)
    agg["src_proto_entropy"] = g["Proto"].apply(_shannon_entropy)
    return agg.reset_index()


def _read_one(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, low_memory=False, on_bad_lines="skip")


def _build_full_arrays(data_dir: Path):
    """Parse 13 binetflows, attach per-source aggregates to each flow, log1p+
    one-hot. Cached as .npz; subsequent loads <15 s."""
    cache_path = data_dir / "_preprocessed_cache.npz"
    if cache_path.exists():
        z = np.load(cache_path)
        return z["X"].astype(np.float32), z["y"].astype(np.int64)

    binetflow_dir = data_dir / "binetflow"
    files = sorted(glob.glob(str(binetflow_dir / "scenario-*-*.binetflow")))
    if not files:
        raise FileNotFoundError(
            f"no .binetflow files in {binetflow_dir} — download from Stratosphere IPS"
        )

    enriched_frames = []
    for f in files:
        df = _read_one(Path(f))
        # Drop Background — keep Normal + Botnet only
        lbl = df["Label"].astype(str)
        is_bn = lbl.str.contains("Botnet", na=False)
        is_nm = lbl.str.contains("Normal", na=False)
        df = df[is_bn | is_nm].copy()
        if len(df) == 0:
            continue
        # Numeric coercion
        for c in ("Dur", "TotPkts", "TotBytes", "SrcBytes", "sTos", "dTos"):
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)
        # Port columns — coerce to int (hex/strange values become NaN → 0)
        df["Sport"] = pd.to_numeric(df["Sport"], errors="coerce").fillna(0)
        df["Dport"] = pd.to_numeric(df["Dport"], errors="coerce").fillna(0)
        # Strip whitespace from categoricals
        df["Proto"] = df["Proto"].astype(str).str.strip()
        df["Dir"] = df["Dir"].astype(str).str.strip()
        df["State"] = df["State"].astype(str).str.strip()

        # Label per flow — derive AFTER the filter, BEFORE the merge
        # (df.merge resets the index, so a pre-filter Series lookup would
        # silently misalign.)
        df["y"] = df["Label"].astype(str).str.contains("Botnet", na=False).astype(np.int64)

        # Per-scenario per-source aggregates
        host_agg = _aggregate_per_source(df)
        df = df.merge(host_agg, on="SrcAddr", how="left")
        enriched_frames.append(df)

    big = pd.concat(enriched_frames, ignore_index=True)
    print(f"  read {len(big)} non-background flows (with host-context joined)")

    y = big["y"].to_numpy().astype(np.int64)

    # Keep only the columns we want as features
    keep = FLOW_NUMERICS + FLOW_CATS + HOST_AGG_NUMERICS
    keep = [c for c in keep if c in big.columns]
    Xf = big[keep].copy()

    # One-hot the small categoricals (proto, dir) over the union
    Xf = pd.get_dummies(Xf, columns=[c for c in FLOW_CATS if c in Xf.columns],
                         dtype=np.float32)

    # log1p the heavy-tailed columns
    for c in LOG1P_COLS:
        if c in Xf.columns:
            Xf[c] = np.log1p(Xf[c].clip(lower=0).astype(np.float64))

    Xf = Xf.replace([np.inf, -np.inf], np.nan).fillna(0.0).astype(np.float32)
    X = Xf.to_numpy(dtype=np.float32)

    print(f"  enriched feature shape: {X.shape}  "
          f"(botnet={int(y.sum())}, normal={int((y==0).sum())})")

    np.savez_compressed(cache_path, X=X, y=y)
    return X, y


def load_ctu13(
    data_dir: str = "dataset/ctu13",
    batch_size: int = 128,
    val_fraction: float = 0.1,
    test_fraction: float = 0.3,
    subsample_fraction: float = 1.0,
    num_workers: int = 0,
    seed: int = 0,
) -> DatasetBundle:
    data_dir = Path(data_dir)
    X, y = _build_full_arrays(data_dir)

    if subsample_fraction < 1.0:
        rng = np.random.RandomState(seed)
        n_keep = int(len(X) * subsample_fraction)
        idx = rng.choice(len(X), n_keep, replace=False)
        X, y = X[idx], y[idx]

    X_train_pool, X_test, y_train_pool, y_test = train_test_split(
        X, y, test_size=test_fraction, random_state=seed, stratify=y,
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_pool, y_train_pool,
        test_size=val_fraction, random_state=seed, stratify=y_train_pool,
    )

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
