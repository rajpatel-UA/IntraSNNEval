"""Shared loading and derivation over saved run artifacts.

Every offline analysis reads the same three things — a run's `results.json`, its
`test_prefix.npz` and its `val_scores.npz` — so the discovery, indexing and
prefix arithmetic live here once rather than being reimplemented per script
with slightly different conventions.

The prefix convention, stated once: ``cum_out[i, t, c]`` is the number of output
spikes class ``c`` has emitted for sample ``i`` after ``t + 1`` timesteps. So
the readout after ``t`` steps is ``argmax_c cum_out[:, t-1, :]``, and
``cum_out[:, -1, :]`` is the terminal decision — equal, by construction, to the
logits the run reported.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterator, List, Optional

import numpy as np

from src.paths import RESULTS_ROOT

RUNS = RESULTS_ROOT / "runs"


@dataclass
class Run:
    protocol: str
    neuron: str
    encoding: str
    seed: int
    path: Path

    @property
    def variant(self) -> str:
        return f"{self.neuron}/{self.encoding}"

    def record(self) -> Dict:
        return json.loads((self.path / "results.json").read_text())

    def prefix(self) -> Dict[str, np.ndarray]:
        z = np.load(self.path / "artifacts" / "test_prefix.npz")
        return {k: z[k] for k in z.files}

    def val(self) -> Dict[str, np.ndarray]:
        z = np.load(self.path / "artifacts" / "val_scores.npz")
        return {k: z[k] for k in z.files}

    def has_artifacts(self) -> bool:
        return (self.path / "artifacts" / "test_prefix.npz").exists()


def discover(protocols: Optional[List[str]] = None,
             neurons: Optional[List[str]] = None,
             encodings: Optional[List[str]] = None,
             require_artifacts: bool = True) -> List[Run]:
    """All completed runs matching the filters, sorted for stable output."""
    out = []
    for rj in RUNS.rglob("seed_*/results.json"):
        seed_dir = rj.parent
        enc = seed_dir.parent.name
        neuron = seed_dir.parent.parent.name.removeprefix("SNN_")
        proto = seed_dir.parent.parent.parent.name
        run = Run(proto, neuron, enc, int(seed_dir.name.removeprefix("seed_")),
                  seed_dir)
        if protocols and proto not in protocols:
            continue
        if neurons and neuron not in neurons:
            continue
        if encodings and enc not in encodings:
            continue
        if require_artifacts and not run.has_artifacts():
            continue
        out.append(run)
    return sorted(out, key=lambda r: (r.protocol, r.neuron, r.encoding, r.seed))


def prefix_predictions(cum_out: np.ndarray) -> np.ndarray:
    """[N, T, C] cumulative counts -> [N, T] predicted class after each step.

    Ties are broken by lowest class index, which is ``argmax``'s behaviour and
    is what the terminal prediction already does — so early steps, where many
    samples still have an all-zero readout, resolve to class 0. That is a real
    property of the readout rule, not an artifact to correct: before any spike
    arrives the network has expressed no preference, and reporting it as the
    majority class is the honest reading of "decide now".
    """
    return cum_out.astype(np.int32).argmax(axis=2)


def convergence_step(curve: np.ndarray, fraction: float) -> Optional[int]:
    """First 1-indexed t where the curve reaches `fraction` of its terminal value.

    Returns None when the terminal value is non-positive, which happens if a run
    collapsed; a numeric answer there would be meaningless.
    """
    terminal = curve[-1]
    if terminal <= 0:
        return None
    hits = np.where(curve >= fraction * terminal)[0]
    return int(hits[0] + 1) if len(hits) else None


def attack_score(cum_slice: np.ndarray, normal_index: int = 0) -> np.ndarray:
    """Continuous attack score from an output-spike readout: 1 - P(normal).

    Softmax over the spike counts rather than a raw count difference, so the
    score is bounded and comparable across variants whose absolute firing rates
    differ by an order of magnitude.
    """
    z = cum_slice.astype(np.float64)
    z = z - z.max(axis=1, keepdims=True)
    p = np.exp(z)
    p /= p.sum(axis=1, keepdims=True)
    return 1.0 - p[:, normal_index]


def threshold_at_far(scores: np.ndarray, y_true: np.ndarray, target_far: float,
                     normal_index: int = 0) -> float:
    """Smallest threshold whose false-alarm rate on *these* rows is <= target.

    Intended to be called on validation and the result then frozen. Choosing it
    on test would be selecting the operating point using the data it is reported
    on, which is the thing the pre-registration forbids.
    """
    benign = scores[y_true == normal_index]
    if len(benign) == 0:
        return float("inf")
    # The (1 - far) quantile of benign scores is the smallest cut that leaves at
    # most `target_far` of them above it.
    return float(np.quantile(benign, 1.0 - target_far, method="higher"))
