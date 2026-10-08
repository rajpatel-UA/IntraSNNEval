"""Per-run artifacts that make the no-retraining analyses possible.

Six manuscript analyses — early readout, fixed-FAR operating points, per-class
breakdowns, NSL-KDD known-vs-unseen, KDDCup99 duplicate-free sensitivity, and
spike/SOP accounting — need nothing beyond outputs a trained model already
produces. They only need them **saved**. This module defines exactly what to
save and nothing more; every derived quantity (AUPRC, F1(t), T95, T99, DR@1%
FAR) is computed offline in the analysis scripts, so the training loop stays
simple and a mistake there costs a script rerun rather than 17 GPU-hours.

What is written, per run, under ``<results_dir>/artifacts/``:

``test_prefix.npz``
    ``idx``      [N] int32   position in the frozen test partition
    ``y_true``   [N] int16
    ``y_pred``   [N] int16   terminal prediction (argmax at t=T)
    ``cum_out``  [N,T,C] uint8  cumulative output-layer spike counts

    ``cum_out`` is the whole early-readout analysis in one array. Slicing at t
    gives the prefix readout, and ``cum_out[:, -1, :]`` must equal the model's
    ordinary logits exactly, which is what the smoke check asserts. uint8 is
    safe because a cumulative count cannot exceed T (25 here) — asserted at
    write time rather than assumed.

``val_scores.npz``
    ``y_true``, ``score`` [Nv,C] float32 — validation logits.

    Kept because fixed-FAR thresholds must be chosen on validation and frozen
    before touching test. Saving these now is what makes that possible later
    without a second inference pass.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import numpy as np


def save_test_prefix(results_dir: Path, idx: np.ndarray, y_true: np.ndarray,
                     y_pred: np.ndarray, cum_out: np.ndarray) -> Path:
    """Persist the per-sample prefix readout. ``cum_out`` is [N, T, C]."""
    d = Path(results_dir) / "artifacts"
    d.mkdir(parents=True, exist_ok=True)

    peak = float(cum_out.max()) if cum_out.size else 0.0
    if peak > np.iinfo(np.uint8).max:
        raise ValueError(
            f"cumulative output spikes reach {peak}, which does not fit uint8. "
            "Widen the dtype in save_test_prefix rather than clipping.")
    n = len(y_true)
    if not (len(idx) == len(y_pred) == cum_out.shape[0] == n):
        raise ValueError(
            f"artifact arrays disagree on N: idx={len(idx)} y_true={n} "
            f"y_pred={len(y_pred)} cum_out={cum_out.shape[0]}")

    path = d / "test_prefix.npz"
    np.savez_compressed(
        path,
        idx=np.asarray(idx, dtype=np.int32),
        y_true=np.asarray(y_true, dtype=np.int16),
        y_pred=np.asarray(y_pred, dtype=np.int16),
        cum_out=cum_out.astype(np.uint8),
    )
    return path


def save_val_scores(results_dir: Path, y_true: np.ndarray,
                    score: np.ndarray) -> Path:
    d = Path(results_dir) / "artifacts"
    d.mkdir(parents=True, exist_ok=True)
    path = d / "val_scores.npz"
    np.savez_compressed(path,
                        y_true=np.asarray(y_true, dtype=np.int16),
                        score=np.asarray(score, dtype=np.float32))
    return path


def spike_accounting(input_spikes: float, hidden_spikes: float,
                     output_spikes: float, n_samples: int,
                     fanout: Dict) -> Dict:
    """Spike decomposition and the SOP proxy, per sample.

    The submitted paper reported one aggregate "spikes/sample" that exceeds the
    input dimensionality, so a reader cannot tell what it contains. Splitting it
    three ways removes that ambiguity, and the SOP figure is what the efficiency
    comparison should actually rest on: a spike into a wide projection costs
    more than a spike into a narrow one, and a raw spike count cannot see that.
    """
    n = max(int(n_samples), 1)
    sops = (input_spikes * fanout["fanout_input"]
            + hidden_spikes * fanout["fanout_hidden"])
    return {
        "input_spikes_per_sample": float(input_spikes) / n,
        "hidden_spikes_per_sample": float(hidden_spikes) / n,
        "output_spikes_per_sample": float(output_spikes) / n,
        "total_spikes_per_sample": float(
            input_spikes + hidden_spikes + output_spikes) / n,
        "sops_per_sample": float(sops) / n,
        "fanout_input": int(fanout["fanout_input"]),
        "fanout_hidden": int(fanout["fanout_hidden"]),
        "gated_cell": bool(fanout["gated"]),
        "sop_definition": ("input_spikes*fanout_input + "
                           "hidden_spikes*fanout_hidden; output-layer spikes "
                           "are the readout and drive no downstream synapses"),
    }


def protocol_provenance(dataset: str) -> Dict:
    """Split-manifest identity for a run, so a result can be traced to the exact
    partition that produced it. Returns empty rather than raising when a v1
    protocol has no manifest — v1 predates the manifest machinery, and that
    absence is itself the honest record."""
    import hashlib
    import json
    from src.paths import RESULTS_ROOT

    candidates = [dataset]
    if dataset.startswith("ctu13_v2_f"):
        candidates.append(f"ctu13_causal_scenario_f{dataset[-1]}")
    for name in candidates:
        p = RESULTS_ROOT / "manifests" / f"{name}.json"
        if p.exists():
            raw = p.read_bytes()
            man = json.loads(raw)
            return {
                "protocol": dataset,
                "manifest": p.name,
                "manifest_sha256": hashlib.sha256(raw).hexdigest(),
                "feature_hash": man.get("extra", {}).get("feature_hash"),
                "fold": man.get("fold"),
                "n_test_expected": man.get("n_test"),
                "group_name": man.get("group_name"),
                "group_disjoint_claimed": man.get("group_disjoint_claimed"),
            }
    return {"protocol": dataset, "manifest": None,
            "note": "v1 protocol — predates split manifests"}
