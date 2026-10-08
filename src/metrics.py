"""Unified metrics module — every experiment script calls compute_metrics.

Produces:
  * Headline (single-number) metrics
      accuracy, balanced_accuracy
      f1_macro, f1_weighted
      precision_macro, recall_macro
      roc_auc_weighted (OVR, weighted, restricted to classes that actually appear)
      matthews_corrcoef
      cohen_kappa
  * Per-class breakdown
      precision_per_class, recall_per_class, f1_per_class, support_per_class
  * Confusion matrix + class names

`compute_spike_stats` summarises the spike record (energy proxy + sparsity).
`count_parameters` reports the trainable parameter count of an nn.Module.
"""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, cohen_kappa_score,
    confusion_matrix, f1_score, matthews_corrcoef,
    precision_recall_fscore_support, precision_score, recall_score,
    roc_auc_score,
)


# ---------------------------------------------------------------------------
# Evaluation context
# ---------------------------------------------------------------------------
# Side information about the TEST rows that a dataset loader knows but a trainer
# does not. Set by the loader, consumed here, so every SNN configuration gains the
# same breakdowns without a single trainer edit -- all 27 route through
# compute_metrics.
#
# Why per-group is not optional on CTU-13: fold 1 averaged a 56% false-alarm
# scenario against a 0.12% detection-rate scenario into one number. Fold means can
# combine opposite failure modes and hide both.
_EVAL_CONTEXT: Dict = {}


def set_eval_context(**kw) -> None:
    """Attach test-row side information. Keys: test_groups, test_dupfree_mask,
    group_name. Call once per run, from the dataset loader."""
    _EVAL_CONTEXT.clear()
    _EVAL_CONTEXT.update({k: v for k, v in kw.items() if v is not None})


def clear_eval_context() -> None:
    _EVAL_CONTEXT.clear()


def get_eval_context() -> Dict:
    return dict(_EVAL_CONTEXT)


def _core_block(y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
    """Compact metric block used for per-group and duplicate-free breakdowns.

    Quality (macro-F1, MCC) is computed on the labels as given, so a 5-class
    task reports a 5-class macro-F1. DR and FAR are IDS quantities and are
    always computed on the collapsed normal-vs-attack view (class 0 = normal),
    matching ``compute_binary_ids_metrics``. Binarising first matters: this
    block originally hard-coded ``labels=[0, 1]`` in the confusion matrix, which
    is right for CTU-13 and silently wrong for the 5-class KDD datasets, where
    it would have scored "dos" as the positive class and dropped probe/r2l/u2r
    from the counts entirely.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    tb = (y_true != 0).astype(int)
    pb = (y_pred != 0).astype(int)
    cm = confusion_matrix(tb, pb, labels=[0, 1])
    tn, fp, fn, tp = int(cm[0, 0]), int(cm[0, 1]), int(cm[1, 0]), int(cm[1, 1])
    multi = len(np.unique(y_true)) > 1
    return {
        "n": int(len(y_true)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "mcc": float(matthews_corrcoef(y_true, y_pred)) if multi else None,
        "DR": tp / (tp + fn) if (tp + fn) else 0.0,
        "FAR": fp / (fp + tn) if (fp + tn) else 0.0,
        "binary_confusion_matrix": cm.tolist(),
        "support_normal": tn + fp,
        "support_attack": tp + fn,
        "pred_normal": tn + fn,
        "pred_attack": fp + tp,
        "pred_attack_frac": float((fp + tp) / max(len(y_pred), 1)),
    }


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_score: Optional[np.ndarray] = None,
    class_names: Optional[List[str]] = None,
) -> Dict:
    out: Dict = {}
    out["accuracy"] = float(accuracy_score(y_true, y_pred))
    out["balanced_accuracy"] = float(balanced_accuracy_score(y_true, y_pred))
    # `labels=` is load-bearing. Without it sklearn averages over the union of
    # labels appearing in y_true or y_pred, so a model that never predicts a
    # zero-support class is scored over a SMALLER denominator than one that
    # emits a single false positive for it. On cicids2017_v2, whose
    # order-disjoint test interval contains no U2R, that gave delta a 4-class
    # denominator while latency and rate got 5 -- an incomparable metric across
    # models of the same protocol. The label universe is a property of the task,
    # not of the predictions.
    _labels = list(range(len(class_names))) if class_names else None
    out["f1_macro"] = float(f1_score(y_true, y_pred, average="macro",
                                     labels=_labels, zero_division=0))
    out["f1_weighted"] = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))
    out["precision_macro"] = float(precision_score(y_true, y_pred, average="macro", zero_division=0))
    out["recall_macro"] = float(recall_score(y_true, y_pred, average="macro", zero_division=0))
    out["matthews_corrcoef"] = float(matthews_corrcoef(y_true, y_pred))
    out["cohen_kappa"] = float(cohen_kappa_score(y_true, y_pred))

    # Per-class — aligned to class_names if given, else 0..K-1
    if class_names is not None:
        labels = list(range(len(class_names)))
        names = list(class_names)
    else:
        labels = sorted(set(np.unique(y_true).tolist()) | set(np.unique(y_pred).tolist()))
        names = [str(l) for l in labels]
    p, r, f, s = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average=None, zero_division=0)
    out["per_class"] = {
        names[i]: {
            "precision": float(p[i]),
            "recall": float(r[i]),
            "f1": float(f[i]),
            "support": int(s[i]),
        }
        for i in range(len(names))
    }
    out["class_names"] = names

    # ROC-AUC, weighted one-vs-rest. sklearn has separate signatures for
    # binary vs multi-class: binary wants y_score as a 1-D array of positive-
    # class probabilities; multi-class wants a 2-D (n, k) matrix that sums to
    # 1 per row. We softmax-normalise the score subset so the multi-class
    # branch satisfies sklearn's sum-to-1 requirement even when every output
    # spike count was zero.
    if y_score is not None:
        try:
            present = np.unique(y_true)
            if y_score.ndim == 2 and y_score.shape[1] >= 2 and present.size >= 2:
                score_subset = y_score[:, present].astype(np.float64)
                shift = score_subset - score_subset.max(axis=1, keepdims=True)
                e = np.exp(shift)
                prob = e / e.sum(axis=1, keepdims=True)
                if present.size == 2:
                    # Binary: feed only the positive-class column (last in
                    # `present`, i.e. the higher-index class).
                    out["roc_auc_weighted"] = float(
                        roc_auc_score(y_true, prob[:, 1])
                    )
                else:
                    out["roc_auc_weighted"] = float(
                        roc_auc_score(y_true, prob, multi_class="ovr",
                                      average="weighted", labels=present)
                    )
            else:
                out["roc_auc_weighted"] = None
        except Exception as e:
            out["roc_auc_weighted"] = None
            out["roc_auc_error"] = str(e)
    else:
        out["roc_auc_weighted"] = None

    out["confusion_matrix"] = confusion_matrix(y_true, y_pred, labels=labels).tolist()

    # ---- context-driven breakdowns (no-ops unless a loader set the context) ----
    # Length equality is the gate: it fires only for the evaluation whose rows the
    # context describes, so train/val calls pass through untouched.
    y_true = np.asarray(y_true); y_pred = np.asarray(y_pred)

    groups = _EVAL_CONTEXT.get("test_groups")
    if groups is not None and len(groups) == len(y_true):
        gname = _EVAL_CONTEXT.get("group_name", "group")
        out["per_group"] = {
            "group_name": gname,
            "groups": {str(g): _core_block(y_true[groups == g], y_pred[groups == g])
                       for g in sorted(np.unique(groups))},
        }

    mask = _EVAL_CONTEXT.get("test_dupfree_mask")
    if mask is not None and len(mask) == len(y_true):
        mask = np.asarray(mask, dtype=bool)
        full = _core_block(y_true, y_pred)
        kept = _core_block(y_true[mask], y_pred[mask])
        # Sensitivity check only. The overlap is not claimed to be zero, and the
        # frozen folds are NOT rebuilt to remove it.
        out["duplicate_free"] = {
            "n_excluded": int((~mask).sum()),
            "pct_excluded": float(100 * (~mask).mean()),
            "metrics": kept,
            "delta_macro_f1_pp": float(100 * (kept["macro_f1"] - full["macro_f1"])),
            "delta_FAR_pp": float(100 * (kept["FAR"] - full["FAR"])),
        }
    return out


def compute_binary_ids_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    normal_class_index: int = 0,
) -> Dict:
    """IDS-canonical binary metrics: Detection Rate, False Alarm Rate.

    Collapses the 5-class problem to (normal, attack) and reports the metrics
    that NSL-KDD literature emphasises as "most important" (per thinline72's
    community baseline and the general IDS evaluation convention).

      Detection Rate  DR  = TP / (TP + FN)   = recall on the "attack" class
                          = fraction of attacks correctly flagged.
      False Alarm Rate FAR = FP / (FP + TN)   = FPR on the "normal" class
                          = fraction of normal traffic wrongly flagged as attack.

    Also returns binary accuracy and F1 for cross-checking against the
    multi-class headline numbers, and the binary confusion matrix.
    """
    y_true_bin = (y_true != normal_class_index).astype(int)
    y_pred_bin = (y_pred != normal_class_index).astype(int)
    cm = confusion_matrix(y_true_bin, y_pred_bin, labels=[0, 1])
    tn, fp, fn, tp = int(cm[0, 0]), int(cm[0, 1]), int(cm[1, 0]), int(cm[1, 1])
    detection_rate = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    false_alarm_rate = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    binary_acc = (tp + tn) / max(tp + tn + fp + fn, 1)
    binary_precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    binary_f1 = (2 * binary_precision * detection_rate
                 / (binary_precision + detection_rate)
                 if (binary_precision + detection_rate) > 0 else 0.0)
    return {
        "binary_accuracy": float(binary_acc),
        "detection_rate": float(detection_rate),
        "false_alarm_rate": float(false_alarm_rate),
        "binary_precision": float(binary_precision),
        "binary_f1": float(binary_f1),
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "binary_confusion_matrix": cm.tolist(),
    }


def compute_spike_stats(total_spikes: float, n_samples: int,
                         total_sites: Optional[int]) -> Dict:
    """Energy proxy + sparsity from accumulated spike counts.

    total_sites is the total number of (timestep × neuron × batch_sample)
    sites that fired any spike at all *could* fire; sparsity =
    1 - (total_spikes / total_sites). If total_sites is unknown, sparsity is
    None.
    """
    out = {
        "energy_spikes_per_sample": float(total_spikes) / max(int(n_samples), 1),
        "total_spikes": float(total_spikes),
    }
    if total_sites and total_sites > 0:
        density = float(total_spikes) / float(total_sites)
        out["spike_density"] = density
        out["spike_sparsity"] = 1.0 - density
    else:
        out["spike_density"] = None
        out["spike_sparsity"] = None
    return out


def count_parameters(model: torch.nn.Module) -> Dict[str, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total_parameters": int(total), "trainable_parameters": int(trainable)}
