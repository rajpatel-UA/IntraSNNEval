#!/usr/bin/env python3
"""Enforce the manuscript's terminology rules on paper/*.tex.

The results are heterogeneous and the honest wording for them is narrow. Under
deadline pressure the natural drift is back toward the submitted paper's
vocabulary --- "consistently outperforms", "dominant", "energy savings" --- so
the banned phrases are checked rather than trusted to memory.

Each rule records why it exists, because a banned phrase with no reason attached
gets overridden by the next person in a hurry.

    python scripts/check_wording.py
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "paper"

#: (regex, why it is wrong given our actual results)
BANNED = [
    (r"statistically best",
     "14 of 27 configurations lie within one Nemenyi critical difference"),
    (r"universally superior|uniformly superior",
     "encoding preference inverts on CTU-13 fold 1"),
    (r"dominant (neuron|configuration|encoding)",
     "the neuron-axis lead is narrow and not per-block"),
    (r"consistently outperform",
     "Leaky wins more individual blocks than LeakyParallel"),
    (r"always faster|universally faster|generally enables earlier",
     "latency reaches T99 earlier on only 2 of 5 protocols"),
    (r"energy sav(ing|ings)(?!.{0,80}proxy)",
     "we measure synaptic operations, not energy; say 'operation-based proxy'"),
    (r"leakage[- ]free",
     "we removed specific leakage modes; absence of all leakage is not proven"),
    (r"generali[sz]es to unseen attacks",
     "unseen-type DR falls to 0.26-0.43 across all 27 configurations"),
    (r"operationally low (FAR|false[- ]alarm)",
     "a validation-calibrated 1% FAR reaches 48% on CTU-13 fold 1"),
    (r"\bchampion\b",
     "informal; use 'top-ranked configuration' or 'aggregate leader'"),
    (r"fastest variant",
     "an inference-speed claim the strict timings do not support"),
    # Protocol terminology: one pair of names, used everywhere.
    (r"original submission|original protocol|legacy protocol",
     "use 'screening protocol' and 'confirmation protocol' throughout"),
    (r"statistically indistinguishable",
     "Nemenyi failing to separate is not evidence of equivalence; say the "
     "procedure does not separate them at this significance level"),
    # Development scaffolding that must never survive into a build.
    (r"final figure should|final generated table should|should mark|"
     r"should add|placeholder",
     "development scaffolding left in the manuscript"),
]


#: Orderings that encode an argument rather than a preference.
#:
#: Some sequences in the Results are chains: a later finding only means anything
#: once the earlier ones have removed the alternatives. Reordering them for
#: concision, which is exactly what a page-fitting pass does, silently destroys
#: the argument while leaving every sentence individually true. The reason a
#: given order is required lives in a conversation that will not exist at
#: submission time, so it is checked here rather than left in a comment that a
#: hurried edit can step over.
#:
#: (file, [anchors in required order], why the order is load-bearing)
ORDER_LOCKS = [
    ("sec_results.tex",
     [r"\emph{Input activity}",
      r"\emph{Whole-network spike count}",
      r"\emph{Synaptic operations}"],
     "the sparsity decomposition is a chain: input activity is the only "
     "structural saving, whole-network count shows it is diluted, and synaptic "
     "operations recover it only because of where the spikes occur. The third "
     "finding is uninterpretable before the first two have removed the "
     "alternatives"),
    ("sec_results.tex",
     ["Matching the gates in both directions",
      "Distributional shift does not explain it either"],
     "the fold-1 argument eliminates the amplitude gate first and "
     "distributional shift second; the closing claim that no mechanism remains "
     "depends on both exclusions having been stated, in that order"),
    ("sec_results_v2_draft.tex",
     ["holds rank one in", "Its mean rank over the family"],
     "block wins are the primary summary and the mean rank the secondary one, "
     "because the family means are compressed by a single protocol whose "
     "leader is neither latency nor rate. Leading with the mean rank leads "
     "with the weaker statistic"),
]

#: Dash style. Em dashes are rejected in both spellings so the house style is
#: uniform and a copy-paste from a draft cannot reintroduce one.
DASHES = [("---", "LaTeX em dash; use a comma, colon, or parentheses"),
          ("\u2014", "Unicode em dash U+2014; use a comma, colon, or parentheses")]

#: Numbers that must never be typed literally — they have frozen macros.
HARDCODED = [
    (r"(?<![\d.])4\.03(?![\d])", r"\FNScrRank"),
    (r"(?<![\d.])1\.44(?![\d])", r"\FNEncRank"),
    (r"(?<![\d.])2\.70(?![\d])", r"\FNNeuRank"),
    (r"(?<![\d.])9\.28(?![\d])", r"\FNScrCD"),
    (r"92\.01", r"\FNCtuStrict"),
    # Caught only after they had been sitting in Threats: duplication rates and
    # the dedup effect were typed rather than generated, so the analysis fix
    # that moved them would not have moved the prose.
    (r"(?<![\d.])57\.53(?![\d])", r"\FNDupKdd"),
    (r"(?<![\d.])2\.68(?![\d])", r"\FNDupNsl"),
    (r"(?<![\d.])0\.79(?![\d])", r"\FNDedupKddLpLat or \FNDedupKddLpRat"),
]

SKIP = {"sec_frozen_numbers.tex", "fig_captions.tex"}

#: Draft files are exempt from the literal/placeholder checks, because their
#: purpose is to hold placeholders until the runs they wait on land. The moment
#: a draft is promoted to a build filename, the \PH{} rule below catches
#: anything still unfilled instead of letting it ship.
DRAFT_SUFFIX = "_draft.tex"

#: Numeric literals that are legitimately not experimental results.
#: Everything else in Abstract/Results/Conclusion should arrive via a macro,
#: so one stale value -- the pre-fix CIC delta figure, say -- cannot survive in
#: prose after the analysis is corrected.
ALLOWED_LITERAL = re.compile(
    r"""
      \d{4}                              # years
    | \$?T(_\{?99\}?|_\{?95\}?)         # T99 / T95 as symbols
    | [0-9]+\s*(pt|in|cm|mm|\\%)         # typographic dimensions
    | (?:\\times|\^)                     # exponents and multipliers
    """, re.VERBOSE)

#: Sections where every experimental number must come from a macro.
STRICT_FILES = {"sec_abstract.tex", "sec_results.tex", "sec_conclusion.tex"}


def _check_order_locks(paper: Path) -> list[str]:
    """Verify that argument-bearing sequences still appear in the required order."""
    out = []
    for fname, anchors, why in ORDER_LOCKS:
        f = paper / fname
        if not f.exists():
            continue
        text = f.read_text()
        pos = []
        for a in anchors:
            # Literal search: these anchors are LaTeX, and escaping them as
            # regexes is how this check broke the first time it was written.
            i = text.find(a)
            if i < 0:
                out.append(f"{fname}: anchor {a!r} has gone missing, so the "
                           f"required ordering can no longer be verified. {why}")
                pos = None
                break
            pos.append((i, a))
        if pos and pos != sorted(pos):
            order = " then ".join(a for _, a in sorted(pos))
            out.append(f"{fname}: reordered to {order}. Required order is "
                       + " then ".join(anchors) + f". {why}")
    return out


def main() -> int:
    problems = []
    files = [p for p in sorted(PAPER.glob("*.tex"))
             if p.name not in SKIP and not p.name.endswith(DRAFT_SUFFIX)]
    for path in files:
        text = path.read_text()
        # Strip comment lines: rationale in comments may quote a banned phrase.
        body = "\n".join(l for l in text.splitlines()
                         if not l.lstrip().startswith("%"))
        for pattern, why in BANNED:
            for m in re.finditer(pattern, body, flags=re.IGNORECASE):
                line = body[:m.start()].count("\n") + 1
                problems.append(f"{path.name}:~{line}  BANNED {m.group(0)!r}\n"
                                f"    why: {why}")
        if "\\PH{" in body:
            n = body.count("\\PH{")
            problems.append(
                f"{path.name}: {n} unfilled \\PH{{}} placeholder(s) in a build "
                "file. Fill them from the artifacts or keep the file named "
                "*_draft.tex until the runs land.")
        for lit, why in DASHES:
            lit = lit.encode().decode("unicode_escape")
            start = 0
            while (i := body.find(lit, start)) != -1:
                line = body[:i].count("\n") + 1
                problems.append(f"{path.name}:~{line}  DASH {lit!r} — {why}")
                start = i + len(lit)
        for pattern, macro in HARDCODED:
            for m in re.finditer(pattern, body):
                line = body[:m.start()].count("\n") + 1
                problems.append(f"{path.name}:~{line}  HARD-CODED {m.group(0)!r}"
                                f" — use {macro}")

    # Literal experimental numbers in the strict sections.
    for path in files:
        if path.name not in STRICT_FILES:
            continue
        body = "\n".join(l for l in path.read_text().splitlines()
                          if not l.lstrip().startswith("%"))
        # Blank out macro invocations and maths-mode symbols first, then look
        # for what is left: a bare decimal is almost certainly a pasted result.
        masked = re.sub(r"\\FN[A-Za-z]+", " ", body)
        masked = re.sub(r"\\(label|ref|includegraphics|texttt|begin|end)"
                        r"\{[^}]*\}", " ", masked)
        masked = re.sub(r"\\times|\\%|\\chi\^2|_\{?[0-9]+\}?", " ", masked)
        for m in re.finditer(r"(?<![\w.])\d+\.\d+(?![\w])", masked):
            val = m.group(0)
            if ALLOWED_LITERAL.fullmatch(val):
                continue
            line = masked[:m.start()].count("\n") + 1
            problems.append(
                f"{path.name}:~{line}  LITERAL RESULT {val!r} — experimental "
                "numbers in this section must come from a \\FN* macro")

    problems += [f"ORDER LOCK  {v}" for v in _check_order_locks(PAPER)]

    if problems:
        print(f"{len(problems)} wording problem(s):\n")
        for p in problems:
            print("  " + p)
        return 1
    # A lock whose file is absent is skipped, not passed; say which.
    checked = sum((PAPER / f).exists() for f, _, _ in ORDER_LOCKS)
    print(f"order locks: {checked} of {len(ORDER_LOCKS)} argument-bearing "
          f"sequences checked and intact; {len(ORDER_LOCKS) - checked} name a "
          "manuscript file not in this tree")
    print(f"wording clean across {len(files)} tex files: no banned phrasing, "
          "no hard-coded frozen number")
    return 0


if __name__ == "__main__":
    sys.exit(main())
