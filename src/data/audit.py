"""Split audit — the gate that must pass before any v2 training job starts.

Review item: *"If a split violates one of those rules, the training job should
refuse to start."* This module is that refusal. It is deliberately cheap enough
to run inside every loader rather than as a separate offline step, because an
audit you have to remember to run is an audit that eventually does not run.

Checks, per (train, val, test) index triple:

  group_overlap        |G_tr ∩ G_te|, |G_tr ∩ G_va|, |G_va ∩ G_te|  → must be 0
                       when the protocol declares itself group-disjoint
  duplicate_rows       exact feature-vector duplicates that occur in both train
                       and test. Reported always; never asserted to be zero,
                       because on KDDCup99 it genuinely is not and pretending
                       otherwise would be the dishonest option.
  class_coverage       classes present in test but absent from train, which
                       silently caps macro-F1 and must be visible in the paper
  fitted_on_test       recorded as False by construction — the flag exists so
                       the frozen manifest states it rather than implying it

`AuditFailure` is raised for the disjointness violations only. Duplicate counts
and coverage gaps are *reported*: they are properties of the benchmark, not bugs
in the split.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Dict, Optional, Sequence

import numpy as np
import pandas as pd


class AuditFailure(RuntimeError):
    """Raised when a split violates a disjointness rule it claims to satisfy."""


@dataclass
class SplitAudit:
    dataset: str
    protocol: str
    fold: Optional[int]
    n_train: int
    n_val: int
    n_test: int
    group_name: Optional[str]
    group_disjoint_claimed: bool
    overlap_train_test: int
    overlap_train_val: int
    overlap_val_test: int
    duplicate_test_rows_in_train: int
    duplicate_test_fraction: float
    train_classes: list
    test_classes: list
    classes_in_test_absent_from_train: list
    scaler_fitted_on_test: bool = False
    vocabulary_fitted_on_test: bool = False
    extra: Dict = field(default_factory=dict)

    @property
    def passes(self) -> bool:
        if not self.group_disjoint_claimed:
            return True
        return (self.overlap_train_test == 0
                and self.overlap_train_val == 0
                and self.overlap_val_test == 0)

    def to_json(self, path: Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = asdict(self)
        payload["passes"] = self.passes
        path.write_text(json.dumps(payload, indent=2, default=str))
        return path


def _row_hashes(feat: pd.DataFrame) -> np.ndarray:
    return pd.util.hash_pandas_object(feat, index=False).to_numpy()


def audit_split(dataset: str, protocol: str, feat: pd.DataFrame, y: np.ndarray,
                tr: Sequence[int], va: Sequence[int], te: Sequence[int],
                groups: Optional[np.ndarray] = None,
                group_name: Optional[str] = None,
                group_disjoint: bool = False,
                fold: Optional[int] = None,
                strict: bool = True) -> SplitAudit:
    """Audit one split. Raises AuditFailure when a declared disjointness fails."""
    tr = np.asarray(tr)
    va = np.asarray(va)
    te = np.asarray(te)

    if groups is not None:
        g_tr, g_va, g_te = set(groups[tr]), set(groups[va]), set(groups[te])
        o_tr_te = len(g_tr & g_te)
        o_tr_va = len(g_tr & g_va)
        o_va_te = len(g_va & g_te)
    else:
        o_tr_te = o_tr_va = o_va_te = 0

    h = _row_hashes(feat)
    dup_mask = np.asarray(pd.Index(h[te]).isin(pd.Index(h[tr])), dtype=bool)

    tr_cls = sorted(int(c) for c in np.unique(y[tr]))
    te_cls = sorted(int(c) for c in np.unique(y[te]))

    audit = SplitAudit(
        dataset=dataset, protocol=protocol, fold=fold,
        n_train=int(len(tr)), n_val=int(len(va)), n_test=int(len(te)),
        group_name=group_name, group_disjoint_claimed=bool(group_disjoint),
        overlap_train_test=int(o_tr_te),
        overlap_train_val=int(o_tr_va),
        overlap_val_test=int(o_va_te),
        duplicate_test_rows_in_train=int(dup_mask.sum()),
        duplicate_test_fraction=float(dup_mask.mean()) if len(te) else 0.0,
        train_classes=tr_cls, test_classes=te_cls,
        classes_in_test_absent_from_train=sorted(set(te_cls) - set(tr_cls)),
    )

    if strict and not audit.passes:
        raise AuditFailure(
            f"{dataset}/{protocol} claims {group_name}-disjoint but overlaps: "
            f"train∩test={o_tr_te}, train∩val={o_tr_va}, val∩test={o_va_te}. "
            "Refusing to train."
        )
    return audit


def frame_hash(feat: pd.DataFrame) -> str:
    """Stable content hash of a feature frame, recorded in every manifest so a
    silently regenerated cache cannot masquerade as the frozen one."""
    h = hashlib.sha256()
    h.update(",".join(map(str, feat.columns)).encode())
    h.update(_row_hashes(feat).tobytes())
    return h.hexdigest()
