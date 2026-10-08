# Diagnostic specifications

Post-hoc diagnostics that were **specified, with their predictions and their
stopping rules, before the analysis was run**. They are not part of
`PREREGISTRATION.md` or `PREREGISTRATION_ADDENDUM.md`: those govern the
confirmation sweep and its registered statistics, and nothing here may be used
to revisit a registered test.

This file exists because "post-hoc" is doing two different jobs in ordinary use.
It describes *when a diagnostic entered the study* --- after the main results,
which is true of everything below. It is also read as *the prediction was formed
after the outcome was known*, which is false of everything below. The difference
matters most when a diagnostic returns null: a null result is only evidence if
the prediction preceded it, and a prediction that lives only in commit messages
is not visible to a reader. So the predictions and the stopping rules are
written here, and the git history records when.

Each entry is frozen at the timestamp given. Amendments are appended, never
edited in place.

---

## E1 --- Threshold-matched encoding ablation

**Frozen** 2026-08-13, before any arm ran. Full rationale in
`scripts/run_e1.py`.

**Question.** The three encodings differ in the code *and* in whether an
amplitude gate is applied before spiking (latency at 0.01, delta at 0.05, rate
none), so the three-way comparison confounds the two. Arms A and B vary the
gate; arm C matches delta's gate to latency's, at which point the two arms emit
**identical input spike sets** and differ only in whether the spike carries
timing information.

**Prediction.** Most of the measured efficiency advantage of latency over rate
is gating rather than timing.

**Outcome.** Confirmed for efficiency, and arm C then produced the study's
headline result: the value of timing alone spans 63.5 pp and changes sign
across protocols.

---

## E5 --- Covariate-shift diagnostic

**Frozen** 2026-08-14, before `scripts/covariate_shift.py` was run. Measured
2026-08-14 17:05, committed as `d70d57c`.

**Specification, as given.**

> Per fold, measure train-vs-test shift: per-feature two-sample KS statistics
> (report max and mean), plus a domain-classifier AUC (logistic regression
> predicting train vs test from features --- AUC 0.5 means no shift, 1.0 means
> fully separable).

**Prediction, as given.**

> Fold 1 has the largest shift and is the only fold where delta wins. If that
> holds across all four folds, you can write that magnitude-discarding codes are
> more robust under covariate shift as a *supported* claim rather than a flagged
> conjecture.

Sharpened when the E1 result landed, still before E5 ran:

> The timing premium (latency − delta@0.01) should correlate negatively with
> shift magnitude across all five protocols. Five points is thin for a
> correlation, so report it as a directional observation with the per-protocol
> values shown, not as a fitted relationship.

**Stopping rule, as given.**

> If f1 is clearly the most shifted fold and continuous magnitudes shift more
> than threshold occupancy, then you have exploratory evidence consistent with
> the hypothesis that magnitude-discarding delta coding is more robust under
> that scenario shift. Do not claim causality from four folds. **If the shift
> pattern does not support the hypothesis, leave the manuscript's current
> cautious statement unchanged.**

**Outcome: one arm confirmed, two refuted.**

| claim | result |
|---|---|
| fold 1 is the most shifted condition | **confirmed** on every measure --- KS median 0.4864, domain AUC 0.9787, against 0.72--0.75 for the KDD family |
| timing premium correlates negatively with shift | **refuted** --- Spearman = 0.000 over five protocols; the two most-shifted protocols sit at opposite extremes (+54.5 pp on fold 0, −9.0 pp on fold 1) |
| magnitudes shift more than threshold occupancy | **refuted** --- occupancy shifts *more* on 6 of 7 protocols, including fold 1 (0.349 against 0.243) |

**Action taken under the stopping rule.** The hypothesis was not supported, so
the manuscript's cautious statement was left unchanged and no mechanism was
claimed. Two facts were added to Section~IV: that fold 1 is the most shifted
condition, and that shift magnitude does **not** explain when timing is
valuable. The second is reported as a measured negative --- it eliminates the
most obvious alternative explanation for the fold-1 inversion, after E1 had
already eliminated the amplitude gate.

**Not pursued.** The shift is concentrated for CTU-13 (top-5 features hold
40--45% of total Wasserstein) and diffuse for CIC-IDS2017 (26 features above
0.1, top-5 hold 17.2%), which suggests shift *structure* rather than magnitude.
It fits four of five protocols and fails on fold 0, which is both concentrated
and the highest timing premium in the study. Five points cannot support it and
testing it properly needs a design this schedule does not have. Recorded as a
conjecture for future work, not a finding.

**Scope note.** The domain classifier is a logistic regression. It is a
measuring instrument, never trained on task labels and never evaluated on the
task, so it does not breach the SNN-only scope of `PREREGISTRATION.md` §0. The
exemption is declared by exact (path, import) pair in `scripts/check_scope.py`
and printed on every clean run.

---

## Four-protocol rehearsal of `protocol_shift.py`

**Frozen** 2026-08-15, before the rehearsal ran. Full statement in
`PARTIAL_4PROTOCOL_PREDICTION.md`, which is the authoritative version.

**Status.** `PARTIAL_4PROTOCOL_REHEARSAL` --- a pipeline exercise, not a result.
Nothing from it enters the manuscript, and Kendall's tau is **not** computed:
tau is registered at 25 blocks in `PREREGISTRATION_ADDENDUM.md` §4, and a value
on fewer protocols is a different quantity with no registered status.

**Prediction.** CIC-IDS2017 carries the largest latency-over-rate margin of the
five protocols, so its absence biases the encoding aggregation against latency.
Adding it should move latency up relative to the four-protocol figure.
