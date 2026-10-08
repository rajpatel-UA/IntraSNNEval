"""Train-only fitting of every learned preprocessing step.

The submitted paper one-hot encoded categorical fields "against the union of all
splits". That builds the feature vocabulary from data the model is not supposed
to have seen. The leak is weak (columns, not values) but it is real, and a
reviewer cannot distinguish a weak leak from a strong one without reading the
code — so v2 removes it rather than arguing about its size.

`encode_train_only` is the single place that fits anything. Numerics are log1p'd
and MinMax'd on the training rows only; categoricals get a training-only
vocabulary plus one explicit ``<unk>`` column each, so a category that appears
only at test time maps somewhere well-defined instead of silently expanding the
feature matrix.

Ported unchanged from the parent repository's `src/data/leakage.py`, which is
where this protocol was first validated, so the conference and journal lines
report the same transformation.
"""
from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


def encode_train_only(feat: pd.DataFrame, tr: np.ndarray, va: np.ndarray,
                      te: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict]:
    """One-hot categoricals with a TRAINING-ONLY vocabulary plus an explicit
    <unk> column per categorical, log1p the numerics, MinMax on train only."""
    obj_cols = feat.select_dtypes(include="object").columns.tolist()
    num_cols = [c for c in feat.columns if c not in obj_cols]

    Xnum = feat[num_cols].apply(pd.to_numeric, errors="coerce")
    Xnum = Xnum.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    Xnum = np.log1p(np.clip(Xnum.to_numpy(), 0.0, None)).astype(np.float32)

    blocks, vocab_sizes, n_unk = [Xnum], {}, {}
    for c in obj_cols:
        col = feat[c].astype(str)
        vocab = sorted(col.iloc[tr].unique())          # TRAIN ONLY
        vocab_sizes[c] = len(vocab)
        index = {v: i for i, v in enumerate(vocab)}
        oh = np.zeros((len(col), len(vocab) + 1), dtype=np.float32)   # +1 = <unk>
        codes = col.map(index)
        known = codes.notna().to_numpy()
        oh[np.arange(len(col))[known], codes[known].astype(int).to_numpy()] = 1.0
        oh[~known, len(vocab)] = 1.0
        n_unk[c] = {"val": int((~known)[va].sum()), "test": int((~known)[te].sum())}
        blocks.append(oh)

    X = np.concatenate(blocks, axis=1).astype(np.float32)
    scaler = MinMaxScaler().fit(X[tr])                  # TRAIN ONLY
    Xtr = scaler.transform(X[tr]).astype(np.float32)
    Xva = np.clip(scaler.transform(X[va]), 0.0, 1.0).astype(np.float32)
    Xte = np.clip(scaler.transform(X[te]), 0.0, 1.0).astype(np.float32)
    meta = {"n_numeric": len(num_cols), "n_categorical": len(obj_cols),
            "train_vocab_sizes": vocab_sizes, "unknown_category_rows": n_unk,
            "n_features": int(X.shape[1])}
    return Xtr, Xva, Xte, meta
