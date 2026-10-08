"""Dataset path resolution — the single external dependency of this repository.

Everything else in this repository is self-contained. Raw datasets are the one
thing that is not vendored (they are large and separately licensed), so every
loader asks this module where they live instead of hard-coding a relative path.
See docs/DATASETS.md for sources and the expected layout.

Resolution order (first hit wins):

  1. ``$HISNN_DATASET_ROOT``          — explicit override, checked first so a
                                        reviewer can point at any location.
  2. ``<repo>/dataset``               — a directory or symlink inside this repo.
  3. ``<repo>/../dataset``            — a sibling directory next to the checkout.

If none exist we raise with the three candidates listed, because a silent
fallback to an empty directory surfaces later as an unreadable FileNotFoundError
inside a loader, several frames from the actual problem.
"""
from __future__ import annotations

import os
from pathlib import Path

CONFERENCE_ROOT = Path(__file__).resolve().parents[1]

#: Per-dataset subdirectory names under the dataset root.
DATASET_DIRS = {
    "nslkdd": "nslkdd",
    "kddcup99": "kddcup99",
    "cicids2017": "cicids2017",
    "ctu13": "ctu13",
}


def _candidates() -> list[Path]:
    env = os.environ.get("HISNN_DATASET_ROOT")
    out = []
    if env:
        out.append(Path(env).expanduser())
    out.append(CONFERENCE_ROOT / "dataset")
    out.append(CONFERENCE_ROOT.parent / "dataset")
    return out


def dataset_root() -> Path:
    for cand in _candidates():
        if cand.is_dir():
            return cand.resolve()
    raise FileNotFoundError(
        "Could not locate the dataset root. Tried:\n  "
        + "\n  ".join(str(c) for c in _candidates())
        + "\nSet $HISNN_DATASET_ROOT, or symlink `dataset/` in the repository root"
        + " at your copy. See docs/DATASETS.md."
    )


def dataset_dir(name: str) -> Path:
    """Absolute path to one dataset's raw files."""
    if name not in DATASET_DIRS:
        raise ValueError(f"Unknown dataset {name!r}. Known: {sorted(DATASET_DIRS)}")
    return dataset_root() / DATASET_DIRS[name]


#: Where run artifacts, tables and figures are written. Always inside this repo.
RESULTS_ROOT = CONFERENCE_ROOT / "results"
PAPER_ROOT = CONFERENCE_ROOT / "paper"
