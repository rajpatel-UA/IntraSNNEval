# Analysis amendment — CIC-IDS2017 zero U2R test support

**Logged 2026-08-13, after all 235 runs completed and before any manuscript
number was frozen.**

This is an **analysis amendment, not a protocol amendment**. The protocol does
not change: the frozen order-disjoint split is preserved exactly, no run is
repeated, and no split manifest is rewritten.

## Defect

The Gate-1 audit checked only for classes present in test but absent from
training. It did not check the reverse. Under the CIC-IDS2017 within-capture
order-disjoint protocol, the U2R class has **zero test support**:

| class | train | val | test |
| --- | ---: | ---: | ---: |
| normal | 1,511,712 | 237,420 | 523,965 |
| dos | 346,206 | 17,556 | 18,892 |
| probe | 109,857 | 27,339 | 21,734 |
| r2l | 11,530 | 755 | 1,561 |
| **u2r** | **2,212** | **4** | **0** |

The mapped U2R events (web attacks, Infiltration) occur early in the Thursday
captures, so the final 20% of each capture contains none. Consequently the
five-class macro-F1 on this protocol is capped at **0.80**.

A second, more serious defect was found while investigating the first.
`compute_metrics` called `f1_score(..., average="macro")` **without `labels=`**,
so sklearn averaged over the union of labels appearing in `y_true` or `y_pred`.
On this protocol that produced a *model-dependent denominator*: delta, which
never predicts U2R, was scored over 4 classes while latency and rate, which each
emit at least one U2R false positive, were scored over 5. Those numbers were not
comparable to one another.

## Resolution

**The split is not changed.** Forcing U2R into the test partition would mean
altering the partition after observing its class composition, which costs more
in pre-registration credibility than it buys in interpretability. The absence is
a property of the frozen chronological structure, not an implementation error.

**The primary metric is not changed.** `macro_f1_fixed_universe` — macro-F1 over
the full five-class label universe, in which an unsupported class contributes
zero — remains the pre-registered primary metric and drives every ranking,
Friedman test and focused comparison.

**The label-universe bug is corrected at the analysis layer.** `src/metrics.py`
now passes an explicit `labels=` so future runs are correct, and the analysis
recomputes macro-F1 from the saved per-sample predictions rather than trusting
the stored field. No retraining was required: the predictions were persisted.

**A secondary descriptive metric is added.** `macro_f1_test_present` averages
only over classes with nonzero test support. It is **descriptive only** — never
used for model selection, ranking, or hypothesis testing — and exists so that
absolute CIC-IDS2017 numbers are interpretable.

## Effect on results

Because an absent class contributes the same zero to every model on the
protocol, the two metrics differ by the constant `n_present / n_classes = 4/5`
and induce **identical within-protocol rankings**. `scripts/dual_macro_f1.py`
asserts this identity rather than assuming it.

| LeakyParallel | fixed universe (primary) | test-present (descriptive) |
| --- | ---: | ---: |
| latency | 0.7842 ± 0.0030 | 0.9803 ± 0.0037 |
| rate | 0.7277 ± 0.0153 | 0.9096 ± 0.0191 |
| delta | 0.4981 ± 0.0087 | 0.6226 ± 0.0109 |

Ordering is latency > rate > delta under the stored metric, the fixed universe,
and the present-class metric alike. **No ranking, no mean rank, and no
hypothesis test changes.** The encoding-axis mean ranks (latency 1.44, rate
2.12, delta 2.44) and the neuron-axis champion mean rank (2.70) are identical
before and after the correction. The only number that moves is delta's absolute
CIC-IDS2017 value, from 0.6226 to 0.4981, because it was previously divided by a
smaller denominator than its competitors.

## Prohibition

CIC-IDS2017's fixed-universe macro-F1 must **not** be compared directly against
NSL-KDD, KDDCup99 or CTU-13 absolute macro-F1 and read as evidence that
CIC-IDS2017 is harder. Its ceiling is 0.80 by construction. Rank-based
cross-protocol statistics remain valid, because the multiplicative penalty is
common to every model within the block and cannot affect within-block ordering.

## Safeguards added

- `scripts/support_audit.py` — bidirectional class-support check across train,
  validation and test, written as a **sidecar** carrying the original manifest
  hash. Frozen manifests are not rewritten; all 235 runs record their hashes and
  changing them would manufacture a reproducibility failure while fixing an
  auditing one. Zero test support is a loud warning, not a gate failure: an
  order-respecting split may legitimately reach an interval in which a class
  does not occur.
- `src/populations.py` — named ranking populations with a guard that refuses a
  mixed field, and `macro_f1_pair`, which returns both metrics together with the
  support facts that distinguish them.
