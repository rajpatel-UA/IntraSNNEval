# Pre-registration — ranking rules for the SNN neuron × encoding study

**Frozen 2026-08-12, before any statistic in `results/analysis/` was computed.**
Nothing below may be changed after seeing an analysis output. If a rule turns out
to be wrong, the fix is a new dated section marked as an amendment, with the
reason, not an edit to the frozen text.

The point of freezing this is narrow and specific. The submitted paper names
`LeakyParallel/latency` the best configuration. Under a stricter protocol that
claim may not survive, and the temptation when it does not is to adjust the
metric, the aggregation, or the dataset set until it does. These rules remove
that freedom.

---

## 0. Scope

This is a **controlled intra-SNN design-space study implemented entirely in
snnTorch**. Every model evaluated is a spiking network built from the nine
snnTorch neuron families; the axes of variation are the neuron model and the
spike encoding.

Excluded by design, with zero runs budgeted: XGBoost, Random Forest, Logistic
Regression, conventional DNN/MLP baselines, Tree-to-Spike, DNN-Tree, TASNN,
GBDT teachers, knowledge distillation, tree-derived encodings, teacher–student
comparisons. The efficiency analysis also stays inside the spiking design space:
no SNN-vs-ANN MAC comparison (§10).

The external review recommended conventional baselines including an
architecture-matched MLP. That recommendation is **declined for this venue**,
on the record and in advance of the results, because it answers a different
question than the one this paper asks. The manuscript states the scope in the
introduction so the omission is a stated boundary, not a silent gap. This
exclusion may not be revisited on the basis of how the SNN results turn out.

## 1. Primary hypothesis

Stated neutrally. The code must not assume the champion:

> Under a common training protocol and leakage-resistant evaluation, the neuron
> model and the spike encoding significantly affect SNN-based NIDS performance,
> with the final configuration selected according to the ranking criteria fixed
> below.

The evidence may end up supporting the narrower statement that
`LeakyParallel/latency` is the top-ranked configuration. It is a permitted
conclusion, not a premise.

Note what this does **not** say. It does not say latency beats rate on terminal
macro-F1, and the paper must not be built on that. The submitted table already
shows rate is close, and on KDDCup99 rate is marginally ahead of latency for
`LeakyParallel` on **accuracy** (0.9260 vs 0.9257) — though on the primary
metric fixed in §2 latency leads there by 0.379 pp macro-F1. Much of the
apparent latency advantage in v1 came from delta being weak, which is a
separate and weaker claim (§7).

## 2. Metrics, with roles fixed in advance

| Role | Metric | Used for |
| --- | --- | --- |
| **Primary quality** | macro-F1 | the ranking, all headline claims |
| Supporting quality | MCC | corroboration; must not contradict the primary |
| Security | DR, FAR, DR@1% FAR, DR@0.1% FAR | operational framing |
| Efficiency | SOPs/sample, spikes/sample, T95, T99, inference ms | the cost axis of the Pareto analysis |

Macro-F1 is primary because every dataset here is severely imbalanced and
accuracy is inflatable by a majority-class predictor. Accuracy may appear in
tables; it may not carry a claim.

AUPRC is reported alongside ROC-AUC, because with attack prevalence below 1% on
several classes ROC-AUC is the more flattering of the two and reporting only it
would be a choice.

## 3. Experimental unit and aggregation

One **block** = one (dataset × seed) pair. Within each block all 27
configurations are ranked by macro-F1, rank 1 = best.

The primary ranking statistic is **mean rank across blocks**, not the mean of raw
macro-F1 across datasets. Raw cross-dataset means are dominated by how different
the datasets are from each other rather than by how the configurations differ,
which is the same defect that makes the submitted paper's `0.7998 ± 0.1989`
uninterpretable.

**Dispersion is always reported within a dataset**, as mean ± SD over seeds.
Pooled cross-dataset standard deviations are banned from the manuscript.

## 4. Statistical tests, specified before the data

1. **Friedman test** over all 27 configurations across blocks. If it does not
   reject, no configuration may be called better than another on quality.
2. If it rejects, **Nemenyi** post-hoc with a critical-difference diagram.
3. **Pre-specified focused comparison**, decided now and not chosen from the
   results: `LeakyParallel/latency` vs `LeakyParallel/rate`, paired Wilcoxon
   signed-rank over the blocks.
4. `LeakyParallel/latency` vs the best non-`LeakyParallel` configuration, with
   the family of focused tests corrected by **Holm**.
5. Effect size and bootstrap 95% CI for Δmacro-F1 (latency − rate), resampling
   datasets first and seeds within datasets, matching `src/stats.py`.
   α = 0.05 throughout.

The focused comparisons report win/tie/loss counts, the median paired
difference, the Holm-adjusted p, and the bootstrap interval together. Where the
Wilcoxon and the dataset-level bootstrap disagree — the block-level sign test
can reject while an interval over four datasets does not — **both are reported
and the disagreement is stated**. It is the honest description of an effect
that is consistent in direction but variable in magnitude across datasets.

## 4a. Two-axis confirmation design

A full v2 re-sweep of all 27 configurations costs ~198 GPU-hours and is out of
budget. Confirmation is therefore targeted at the two axes through the selected
configuration, fixed here so the choice cannot be made after seeing results:

- **Axis A — encoding**, neuron fixed: `LeakyParallel × {latency, rate, delta}`.
  Answers: with the neuron held constant, which encoding is strongest under the
  revised protocol?
- **Axis B — neuron**, encoding fixed: `{all nine families} × latency`.
  Answers: with latency held constant, is `LeakyParallel` still among the
  strongest neuron realisations?

Their intersection is `LeakyParallel/latency`. Axis B is **not** a second
design-space ranking and must not be reported as one; it is a confirmation that
the neuron choice survives a stricter protocol. Both axes report per-dataset
mean ± seed SD plus mean ranks within the axis.

## 5. How the result maps to manuscript wording

Decided in advance so the wording follows the evidence rather than the reverse.

| Outcome | Permitted wording |
| --- | --- |
| Lowest mean rank **and** significant advantage | "best-performing / top-ranked configuration" |
| Lowest mean rank, not significantly ahead of another variant | "top-ranked" or "co-leading configuration" |
| Comparable quality, clearly better efficiency | "strongest quality–efficiency operating point" |
| Another configuration clearly wins | report that. The champion claim is dropped. |

## 6. Protocol generations

`v1` is the submitted protocol, kept reproducible and reported as such. `v2` is
leakage-resistant: train-only fitting of every learned transform, a partition
frozen independently of the model seed, group structure respected where the
benchmark has one, and an audit that refuses to train on a split violating a
disjointness it claims. Per dataset:

| Dataset | v2 primary evaluation |
| --- | --- |
| NSL-KDD | official train/test split |
| KDDCup99 | official train/test split |
| CIC-IDS2017 | within-capture **order-disjoint** 70/10/20 (5-class); day-disjoint Mon–Wed / Thu–Fri as a binary transfer test |
| CTU-13 | scenario-disjoint frozen 4-fold, partition-local causal host aggregates |

v1's role is the **full 27-variant design-space screening**; v2's role is
**strict confirmation** along the two axes of §4a. v1 and v2 numbers are never
mixed inside one table, and the historical v1 numbers are not silently replaced.
Their difference is a result — the size of the leakage effect — and is reported
as one.

Two wording constraints follow from how these protocols were actually built:

- **CIC-IDS2017 ordering is a proxy.** The redistribution used here drops the
  `Timestamp` column, so within a capture the split orders by row position.
  Call it "within-capture order-disjoint" and state the proxy explicitly.
  Do not call it verified chronological time. The day-level grouping, which
  carries the temporal claim, is exact and comes from the file names.
- **CTU-13 is scenario-disjoint, not host-disjoint.** 76.2% of fold-1 test
  hosts also appear in training (recorded in each CTU-13 manifest). The
  scenario-disjoint result — 92.01 ± 9.72, folds 77.0 / 91.2 / 99.9 / 99.9 —
  is the **primary** CTU-13 number. The v1 100.00 ± 0.00 is retained only as a
  protocol comparison, labelled as such, with the explanation that the random
  flow split let scenario- and host-related information cross the evaluation
  boundary.

The train-only preprocessing rule is unconditional: split first, fit every
learned transform on training rows, then apply to validation and test. That
covers category vocabularies, MinMax parameters, any alternative normalisation,
and any feature-selection statistic. The audit finding that the old
union-of-splits vocabulary affected 0 NSL-KDD and 2 KDDCup99 test records is
reported as a measurement of how large the old leak was — **it is not a
justification for retaining the old preprocessing.**

## 7. Delta encoding

Delta modulation encodes temporal contrast, and the input here is a static
per-flow feature vector, so a weak delta result is at least partly a
representational mismatch rather than evidence that delta is intrinsically
inferior. Delta's weakness may be reported as a secondary observation, phrased
as a property of this formulation. **It may not be used as the primary evidence
that latency encoding is preferable.** The primary encoding comparison is
latency vs rate.

## 8. Threshold confound, stated not hidden

`SLSTM` and `SConv2dLSTM` run at firing threshold 0.1 while the other seven
families run at 1.0, because at 1.0 the gated cells emit no spikes at all and
collapse to the majority class. This means the sweep is not a pure
neuron-only ablation, and the manuscript must say so wherever it claims
otherwise. Whether it changes the ranking is an empirical question
(`scripts/threshold_sensitivity.py`); until that evidence exists the limitation
is stated, not argued away.

## 10. Efficiency accounting stays inside the spiking design space

Report, per configuration: input spikes, hidden spikes, output spikes, total
spikes, SOPs/sample, inference latency, and throughput where available, with
`SOP = Σ_{ℓ,t} s_ℓ(t) · fanout_ℓ`. Decomposing the spike count is not
cosmetic — the submitted paper's "spikes/sample" exceeds the input
dimensionality, so a reader currently cannot tell what it includes.

Any energy figure derived as `Ê = SOP × E_AC` is labelled an **operation-based
energy proxy**, never measured energy, and the constant is cited to its original
hardware source.

**No SNN-vs-ANN MAC comparison appears in this paper.** The primary efficiency
comparison is LP/latency vs LP/rate vs LP/delta — a question entirely inside
snnTorch, and the one this paper is scoped to answer.

## 11. Early readout

For `t = 1…T`, with `ŷ(t) = argmax_c Σ_{τ≤t} S_c(τ)`, report F1(t), MCC(t),
DR(t) and cumulative spike activity, and define

    T95 = min{ t : F1(t) ≥ 0.95 · F1(T) },   T99 = min{ t : F1(t) ≥ 0.99 · F1(T) }.

The question is whether latency encoding reaches near-terminal quality at a
smaller t than rate. Both thresholds are fixed here so neither can be chosen
afterwards to favour a result.

## 12. Things that are forbidden

- Selecting a subset of datasets after seeing the ranking.
- Re-running with a different seed set and reporting the better one.
- Choosing an operating threshold on test data. Fixed-FAR thresholds are
  selected on validation and then frozen.
- Reporting the consistency "Score" of the submitted Table III. It is an
  arbitrary statistic, and it was already reported inconsistently (text says
  "7 of the 8 (dataset, metric) cells", the table column says `7/12`). It is
  replaced by a Pareto analysis over quality and cost.
- Quoting a mean across CTU-13 scenario folds without the fold range beside it.
  The folds span roughly 77 to 100 macro-F1 and the mean hides that entirely.
