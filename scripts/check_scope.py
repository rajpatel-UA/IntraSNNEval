#!/usr/bin/env python3
"""Guard the SNN-only scope boundary.

`PREREGISTRATION.md` §0 excludes conventional and hybrid model families from
this study. That boundary is easy to state and easy to erode — a helper copied
across from the parent repository, a comparison added late "just for context" —
so it is checked rather than trusted.

Two checks:

  1. No banned import is reachable from the conference tree. Scope is about what
     the code can *train*, so this looks for imports, not prose: a docstring may
     name XGBoost to explain why it is absent.
  2. `src/models` exports `SNN` and nothing else.

Exits non-zero on violation. Cheap enough to run before every commit.

    python scripts/check_scope.py
"""
from __future__ import annotations

import ast
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

#: Import roots that would mean a non-spiking or hybrid model entered the study.
BANNED_IMPORTS = {
    "xgboost", "lightgbm", "catboost",
    "sklearn.ensemble", "sklearn.linear_model", "sklearn.tree",
    "sklearn.neural_network", "sklearn.svm",
}

#: Module names from the parent repo's journal line that must not be ported here.
BANNED_MODULES = {
    "tree_to_spike", "dnn_student", "tasnn", "s_xlstm",
    "spiking_griffin", "spiking_ssm",
    "training_treespike", "training_dnn", "training_tasnn",
}

#: Narrow, reasoned exemptions. The rule this guard protects is that no non-SNN
#: model is *entered into the study as a competitor*; a classifier used as a
#: measuring instrument on the data is not an entrant. Keyed by exact path so a
#: second use of the same import elsewhere still fails, and restricted below to
#: files outside `src/`, so nothing exempted can be reachable from the models.
DIAGNOSTIC_EXEMPTIONS = {
    ("scripts/covariate_shift.py", "sklearn.linear_model"):
        "domain classifier measuring train-vs-test separability; it is never "
        "trained on labels, never evaluated on the task, and reports no "
        "detection metric",
    ("scripts/covariate_shift.py", "sklearn.model_selection"):
        "train/test split for the domain classifier above",
}

SKIP_DIRS = {"results", "dataset", "__pycache__", ".git", "paper"}


def iter_py():
    for p in ROOT.rglob("*.py"):
        if any(part in SKIP_DIRS for part in p.relative_to(ROOT).parts):
            continue
        yield p


def imported_names(tree: ast.AST):
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                yield a.name
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                yield node.module


def main() -> int:
    violations, exempted = [], []

    for path in iter_py():
        rel = path.relative_to(ROOT)
        try:
            tree = ast.parse(path.read_text())
        except SyntaxError as e:
            violations.append(f"{rel}: does not parse ({e})")
            continue
        for name in imported_names(tree):
            root = name.split(".")[0]
            if name in BANNED_IMPORTS or root in BANNED_IMPORTS:
                why = DIAGNOSTIC_EXEMPTIONS.get((rel.as_posix(), name))
                if why and rel.parts[0] != "src":
                    exempted.append(f"{rel}: {name} — {why}")
                else:
                    violations.append(
                        f"{rel}: imports {name!r} — non-SNN model family")
            if root in BANNED_MODULES or name.split(".")[-1] in BANNED_MODULES:
                violations.append(f"{rel}: imports {name!r} — journal-line model")
        if path.stem in BANNED_MODULES:
            violations.append(f"{rel}: journal-line module present in the tree")

    models_init = ROOT / "src/models/__init__.py"
    exported = set()
    if models_init.exists():
        tree = ast.parse(models_init.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                exported.update(a.asname or a.name for a in node.names)
    if exported != {"SNN"}:
        violations.append(
            f"src/models/__init__.py exports {sorted(exported)}; expected exactly ['SNN']")

    for e in exempted:
        print(f"exempt (diagnostic, not a competitor): {e}")

    if violations:
        print("SCOPE VIOLATION — this study is snnTorch-only "
              "(PREREGISTRATION.md §0):")
        for v in violations:
            print("  -", v)
        return 1

    n = sum(1 for _ in iter_py())
    print(f"scope clean: {n} Python files, no non-SNN model family reachable, "
          "src/models exports only SNN")
    return 0


if __name__ == "__main__":
    sys.exit(main())
