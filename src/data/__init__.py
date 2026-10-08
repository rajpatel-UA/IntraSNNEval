"""Dataset registry for the conference study.

Two protocol generations are registered side by side, and the name says which
one you get:

  ``<ds>``           **v1** — the protocol of the submitted paper. Frozen so the
                     published table stays reproducible. Known weaknesses:
                     union-of-splits categorical vocabulary, and (CIC-IDS2017,
                     CTU-13) a random flow-level partition keyed on the model
                     seed.
  ``<ds>_v2`` etc.   **v2** — leakage-resistant. Train-only fitting, partition
                     frozen independently of the seed, group structure respected
                     where the benchmark has one, audited before use.

Never mix generations inside one table. A v1 number and a v2 number answer
different questions and their difference is the result, not noise.
"""
from functools import partial as _partial, wraps as _wraps

from src.data.cicids2017 import load_cicids2017
from src.data.ctu13 import load_ctu13
from src.data.kddcup99 import load_kddcup99
from src.data.nslkdd import load_nslkdd
from src.data.protocols import (load_cicids2017_daydisjoint,
                                load_cicids2017_v2, load_kdd_v2)
from src.paths import dataset_dir

# ------------------------------------------------------------------ v1 (frozen)
def _v1(fn, name):
    """Resolve the dataset directory at call time, not import time, so merely
    importing the registry does not require the datasets to be present.

    ``functools.wraps`` matters here beyond cosmetics: the trainer introspects
    each loader's signature to decide which optional kwargs (``test_fraction``,
    ``subsample_fraction``) it may forward. Without ``__wrapped__`` every loader
    would look like ``(**kw)`` and those kwargs would be silently dropped.
    """
    @_wraps(fn)
    def _call(**kw):
        kw.setdefault("data_dir", str(dataset_dir(name)))
        return fn(**kw)
    return _call


DATASET_LOADERS = {
    "nslkdd": _v1(load_nslkdd, "nslkdd"),
    "kddcup99": _v1(load_kddcup99, "kddcup99"),
    "cicids2017": _v1(load_cicids2017, "cicids2017"),
    "ctu13": _v1(load_ctu13, "ctu13"),
}

# ------------------------------------------------------------------ v2 (strict)
DATASET_LOADERS.update({
    "nslkdd_v2": _partial(load_kdd_v2, dataset="nslkdd"),
    "kddcup99_v2": _partial(load_kdd_v2, dataset="kddcup99"),
    "cicids2017_v2": load_cicids2017_v2,
    "cicids2017_daydisjoint": load_cicids2017_daydisjoint,
})

# CTU-13 scenario-disjoint folds with partition-local causal host aggregates.
# Registered lazily: the three-condition artifact is a multi-GB build product and
# importing its module at registry time would pay that cost on every `import
# src.data`, including for datasets that never touch CTU-13.
_CTU13_CONDITIONS = ("legacy_random", "causal_random", "causal_scenario")


def _ctu13_condition(condition, fold, feature_set, **kw):
    from src.data.ctu13_conditions import load_ctu13_condition
    return load_ctu13_condition(condition=condition, fold=fold,
                                feature_set=feature_set, **kw)


for _c in _CTU13_CONDITIONS:
    for _f in range(4):
        for _fs in ("base", "combined"):
            DATASET_LOADERS[f"ctu13_{_c}_f{_f}_{_fs}"] = _partial(
                _ctu13_condition, condition=_c, fold=_f, feature_set=_fs)
        # `causal_scenario` with the combined feature set is the headline v2
        # CTU-13 protocol; alias it so configs do not have to spell out the
        # condition/feature-set grid.
    if _c == "causal_scenario":
        for _f in range(4):
            DATASET_LOADERS[f"ctu13_v2_f{_f}"] = DATASET_LOADERS[
                f"ctu13_causal_scenario_f{_f}_combined"]


def get_dataset(name: str, **kwargs):
    if name not in DATASET_LOADERS:
        raise ValueError(
            f"Unknown dataset {name!r}. Available: {sorted(DATASET_LOADERS)}")
    return DATASET_LOADERS[name](**kwargs)


__all__ = ["DATASET_LOADERS", "get_dataset"]
