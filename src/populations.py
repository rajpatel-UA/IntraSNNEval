"""Ranking populations, and the two metric labels CIC-IDS2017 forces on us.

Two mistakes reached the analysis before this module existed, and both were
silent rather than loud, which is why they are now enforced in code.

**Mixed populations.** A ranking is only meaningful over a set of configurations
that were all evaluated on the same footing. The deduplication analysis pooled
the nine latency-coded neurons with the three LeakyParallel encodings --- a
cross-shaped confirmation set, not a competitive field --- and dutifully
reported a "leader" for it. `require_population` makes that fail instead.

**Unlabelled macro-F1 on CIC-IDS2017.** The order-disjoint test interval
contains no U2R examples, so the pre-registered five-class macro-F1 carries a
fixed zero for that class. The value is not miscomputed; it is macro-F1 over a
fixed five-class universe, and it is capped at 0.8. But it is not comparable to
a five-class score on a dataset where all five classes occur, so it may never
again be written as a bare ``macro_f1``. See `results/analysis/AMENDMENT_*.md`.
"""
from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np

#: Named ranking populations. Members of one ranking must all belong to one.
POPULATIONS = {
    "full_27_screening": "all 9 neurons x 3 encodings, v1 screening sweep",
    "encoding_axis_lp": "LeakyParallel x {latency, rate, delta}",
    "neuron_axis_latency": "9 neuron families x latency",
    # Added by PREREGISTRATION_ADDENDUM.md, frozen before any of its runs
    # completed. Same members as the screening population but evaluated under
    # the confirmation protocols; the two must never be pooled.
    "full_27_confirmation": "all 9 neurons x 3 encodings, confirmation protocols",
}

NEURONS = ["Alpha", "Lapicque", "Leaky", "LeakyParallel", "RLeaky",
           "RSynaptic", "SConv2dLSTM", "SLSTM", "Synaptic"]
ENCODINGS = ["rate", "latency", "delta"]


class PopulationError(ValueError):
    """Raised when a ranking is attempted over a set that is not a population."""


def members(population: str) -> set:
    if population in ("full_27_screening", "full_27_confirmation"):
        return {f"{n}/{e}" for n in NEURONS for e in ENCODINGS}
    if population == "encoding_axis_lp":
        return {f"LeakyParallel/{e}" for e in ENCODINGS}
    if population == "neuron_axis_latency":
        return {f"{n}/latency" for n in NEURONS}
    raise PopulationError(f"unknown population {population!r}; "
                          f"known: {sorted(POPULATIONS)}")


def require_population(variants: Iterable[str], population: str) -> list:
    """Assert every variant belongs to `population`; return them in a stable order.

    Rejects strays rather than silently ranking a mixed field. Missing members
    are permitted --- a partial axis is legitimate and is flagged elsewhere as
    INTERIM --- but a variant from outside the population is not.
    """
    allowed = members(population)
    got = set(variants)
    stray = sorted(got - allowed)
    if stray:
        raise PopulationError(
            f"population {population!r} ({POPULATIONS[population]}) cannot "
            f"contain {stray}. Ranking a mixed set produces a leader that means "
            "nothing; split the analysis by axis instead.")
    return sorted(got)


def macro_f1_pair(y_true: np.ndarray, y_pred: np.ndarray,
                  n_classes: int) -> dict:
    """Both macro-F1 variants plus the support facts that distinguish them.

    ``macro_f1_fixed_universe`` is the pre-registered primary metric: macro-F1
    over all ``n_classes`` labels, with any class absent from test contributing
    zero. It drives every ranking and hypothesis test, unchanged.

    ``macro_f1_test_present`` averages only over classes with nonzero test
    support. It is **descriptive only** --- added after the audit, never used
    for selection or testing. Because an absent class contributes the same zero
    to every model on that protocol, the two differ by a constant factor
    (``n_present / n_classes``) and therefore induce identical within-protocol
    rankings.
    """
    from sklearn.metrics import f1_score

    labels = list(range(n_classes))
    per = f1_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
    support = np.array([(y_true == c).sum() for c in labels])
    present = support > 0
    return {
        "macro_f1_fixed_universe": float(per.mean()),
        "macro_f1_test_present": float(per[present].mean()) if present.any()
                                 else float("nan"),
        "n_classes_fixed": int(n_classes),
        "n_classes_present": int(present.sum()),
        "zero_support_classes": [int(c) for c in np.array(labels)[~present]],
        "macro_f1_ceiling": float(present.sum() / n_classes),
    }
