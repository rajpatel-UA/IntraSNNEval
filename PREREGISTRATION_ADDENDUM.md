# Addendum 1 — full confirmation sweep

**Frozen 2026-08-14, before any run of the full confirmation sweep completed and
before any statistic derived from it was computed.** The original
`PREREGISTRATION.md` is unchanged and remains in force; this document extends it
to a population that the original explicitly declined to evaluate on cost
grounds.

Everything below must be checkable against the commit that introduced it. If a
rule here turns out to be wrong, the fix is a further dated addendum stating
why, not an edit to this text.

---

## 1. Why this addendum exists

`PREREGISTRATION.md` §4a adopted a two-axis confirmation design because a
complete 27-configuration re-sweep was outside the budget available before the
original deadline. That constraint has lifted. 235 of the 675 cells of the full
confirmation sweep are already complete, so the remainder is 440 runs rather
than 675.

The two-axis design remains the **confirmatory** analysis, because it is the one
that was registered before any confirmation result existed. The full sweep is
reported alongside it under this addendum, and its status is stated wherever it
appears: registered in advance of its own results, but after the two-axis
results were known.

## 2. Population

All **27 configurations** (9 neuron families × 3 encodings) on all **5
confirmation protocols** (`nslkdd_v2`, `kddcup99_v2`, `cicids2017_v2`,
`ctu13_v2_f0`, `ctu13_v2_f1`) at **5 seeds** {42, 43, 44, 45, 46} = 675 cells.
Population identifier: `full_27_confirmation`, registered in
`src/populations.py` and enforced by the same guard that refuses mixed fields.

## 3. Metric, unit, and statistics — unchanged from the original

- **Primary metric:** macro-F1 over the **fixed label universe**. On
  CIC-IDS2017 this remains capped at 0.80 because U2R has no test support; the
  test-present variant stays descriptive and is never used for ranking.
- **Block:** one (protocol × seed) pair, giving 25 blocks.
- **Ranking statistic:** mean rank across blocks. Ties within a block receive
  average ranks (`rankdata(..., method="average")`).
- **Omnibus:** Friedman across the 27 configurations, with Nemenyi post-hoc
  reported **descriptively**. With 27 configurations and 25 blocks the critical
  difference is large; failure to separate is reported as limited all-pairs
  resolution and never as evidence of equivalence.
- **Cross-distribution summary:** the protocol-level collapse (one mean per
  protocol before ranking, 5 units) is the **primary** cross-distribution
  statistic, because five seeds on one protocol share a dataset, a partition and
  a preprocessing fit and are not five independent observations.
- **Intervals:** hierarchical bootstrap resampling protocols first, then seeds
  within protocols, as in `src/stats.py`.

## 4. Ranking-stability statistic, specified now

**Kendall's τ** between the screening ranking and the full confirmation ranking
over all 27 configurations, with a permutation p-value (10,000 permutations,
RNG seed 20260814). Reported with the configurations whose rank changes most in
absolute position. This is the quantity that answers the paper's own question —
how far an optimistic protocol displaces design-space conclusions — and it is
fixed here so it cannot be selected afterwards from among several possible
agreement measures.

## 5. Outcome to permitted wording

Decided before the result exists, including the branch in which the
recommendation is withdrawn. Writing that branch down is what makes the others
worth believing.

| Outcome under the full confirmation sweep | Permitted wording |
| --- | --- |
| `LeakyParallel/latency` holds rank 1 | "top-ranked under both protocols" |
| Another **latency** configuration takes rank 1 | "the leading group is stable; the leader within it is not", naming the new leader, and reporting `LeakyParallel/latency`'s position explicitly |
| A **non-latency** configuration takes rank 1 | the encoding conclusion is reported as protocol-dependent and **the recommendation is withdrawn** |

In every branch the screening result is reported unchanged alongside, and the
difference between the two rankings is the finding.

Additional constraints carried forward from the original: no dataset or protocol
may be dropped after seeing the ranking; no seed set may be re-drawn; the
primary metric may not be swapped; and pooled cross-protocol standard deviations
remain prohibited.

## 6. What this does not change

The pre-registered two-axis tests are **not** recomputed over the expanded
population. Their p-values, mean ranks and block counts stand as registered.
Adding configurations after seeing results and then re-running the registered
tests over the larger set would destroy exactly the property the registration
exists to provide.

The post-hoc analyses stay post-hoc and stay labelled: the threshold-matched
encoding ablation, the delta-equivalence derivation, the CTU-13 covariate-shift
diagnostic, and any T-sweep.

### 6.1 If the full sweep and the registered two-axis confirmation disagree

They are different populations answering overlapping questions, so a
disagreement is possible and the resolution rule is fixed here, before either
result is known.

Both are reported. The **registered two-axis confirmation is the confirmatory
result**, because it was registered before any confirmation data existed. The
**full sweep is the more complete but less strongly registered** result, for the
reason stated in §1: it was registered before its own runs but after the
two-axis results were known.

A disagreement is reported as **evidence about the targeted design**, not
resolved in either direction. Concretely: if the full sweep places a different
neuron family first while the registered Axis~B placed `LeakyParallel` first, we
report that the two-axis design's answer does not survive extension to the full
configuration space, name both leaders, and state that our evidence does not
establish which is preferable. We do not adopt whichever ranking is more
favourable, and we do not treat the larger population as automatically
superseding the registered one.

The same rule applies to the encoding axis. If the full sweep and Axis~A
disagree about the leading encoding, both are reported and the disagreement is
described rather than adjudicated.

## 7. Cost, recorded for honesty about what was affordable

Measured from the 235 completed runs rather than estimated: the remaining 440
cells cost **266.9 GPU-hours**, dominated by CIC-IDS2017 (157.3 h for 120 runs)
and by the slower neuron families (Alpha 3.00×, SConv2dLSTM 2.19× and Lapicque
1.93× the `LeakyParallel` reference). At the measured throughput of 5.17 GPU-h
per wall-hour at six-way concurrency, that is roughly 52 wall-clock hours.

---

## §8. Permuted-timing control (arm A2), frozen 2026-08-17

**Frozen before any A2 run existed.** Post-hoc and exploratory, at the same
evidence grade as E1 and E5: see `DIAGNOSTIC_SPECIFICATIONS.md`.

### Design

Three arms, LeakyParallel, $T=25$, five confirmation protocols, three seeds.

| arm | which features spike | when |
|---|---|---|
| A1 latency@0.01 | $x_i > \vartheta_\mathrm{lat}$ | $t_i=\lfloor(1-x_i)(T-1)\rfloor$ |
| A2 permuted-time | identical to A1 | random permutation of A1's own time multiset |
| A3 delta@0.01 | identical to A1 | all at $t=0$ |

$\Delta_\mathrm{map} = A1 - A2$ (value of the amplitude-to-time mapping),
$\Delta_\mathrm{spread} = A2 - A3$ (value of temporal distribution as such),
and $\Delta_\mathrm{time} = A1 - A3 = \Delta_\mathrm{map} +
\Delta_\mathrm{spread}$ exactly by construction.

A2 preserves, per sample: the set of features that spike, the total input
spike count, and the multiset of spike times. It destroys only which feature
receives which time. Uniform random times would be a weaker control, because
latency's time distribution is skewed by the amplitude distribution and
uniform sampling would also change the temporal envelope.

### Predictions, fixed now

1. **The decomposition closes.** The three measured means satisfy
   $\Delta_\mathrm{time} = \Delta_\mathrm{map} + \Delta_\mathrm{spread}$ to
   floating point. This is a harness check, not a finding; failure halts the
   analysis.
2. **CTU-13 fold 0**: $\Delta_\mathrm{map}$ is the dominant positive term,
   carrying more than half of the published $+54.5$ pp. Reasoning: fold 0 has
   25 features of which 91.4% are active per sample, so nearly every feature
   contributes an amplitude, and there is a large premium to explain.
3. **CTU-13 fold 1**: $\Delta_\mathrm{map}$ is small in magnitude
   ($|\Delta_\mathrm{map}| < 5$ pp) and the published $-9.0$ pp liability sits
   mainly in $\Delta_\mathrm{spread}$. Reasoning: a liability produced by the
   *mapping* would mean amplitude order is actively misleading there, whereas a
   liability produced by *spreading* means the network is penalised for
   receiving evidence late, which is the simpler account.
4. **Manipulation check**: rank correlation between amplitude and spike time is
   $-1.000$ for A1 by construction (modulo floor ties) and $\approx 0.000$ for
   A2, on every protocol.

### Outcome map, fixed now

- **$\Delta_\mathrm{map}$ carries most of $\Delta_\mathrm{time}$ on the
  protocols where it is large**: the published result stands and sharpens. The
  claim becomes that the amplitude-to-time code, not temporal spreading, is
  what timing buys.
- **$\Delta_\mathrm{map} \approx 0$ everywhere**: the published
  $\Delta_\mathrm{time}$ is temporal spreading, not an amplitude-to-time code.
  The headline is reframed in IV-D and the abstract as *amplitude-ordered
  temporal delivery versus simultaneous delivery*, and the title changes. No
  number is retracted; what changes is what the number is said to measure.
- **Mixed across protocols**: report the decomposition per protocol and make no
  general claim about which component matters, in the same form as the existing
  protocol-dependence result.
- **Prediction 2 or 3 fails**: recorded as a failed prediction in this file,
  with the measured values, and the account is revised rather than the
  prediction quietly dropped.

### Determinism

The permutation is a deterministic function of the sample's own feature vector
and the run seed, so it is fixed per sample across epochs and invariant to
shuffling and batch composition. A permutation redrawn each epoch would be data
augmentation and would confound the control with a regularisation effect.

### §8 outcome, recorded 2026-08-22

Measured after 15 A2 runs, zero failures. Full detail in
`results/analysis/a2/`.

| protocol | $\Delta_\mathrm{map}$ | $\Delta_\mathrm{spread}$ | $\Delta_\mathrm{time}$ |
|---|---:|---:|---:|
| NSL-KDD | +7.04 | −5.87 | +1.17 |
| KDDCup99 | +5.70 | −5.76 | −0.06 |
| CIC-IDS2017 | +16.38 | +13.62 | +29.99 |
| CTU-13 f0 | +23.92 | +30.60 | +54.52 |
| CTU-13 f1 | +3.71 | −12.73 | −9.02 |

**Prediction 1 (closure), CONFIRMED.** Maximum absolute error
$3.6\times10^{-15}$ pp over five protocols. Input spikes per sample are
identical across all three arms to $0.0$.

**Prediction 2 (CTU-13 fold 0: $\Delta_\mathrm{map}$ carries more than half of
$+54.5$ pp), FAILED.** Measured $\Delta_\mathrm{map} = +23.92$ pp, which is
43.9%. Temporal spreading carries the larger share at $+30.60$ pp. Recorded as
a failed prediction rather than revised after the fact.

**Prediction 3 (CTU-13 fold 1: $|\Delta_\mathrm{map}| < 5$ pp and the liability
sits in spreading), CONFIRMED.** $\Delta_\mathrm{map} = +3.71$ pp and
$\Delta_\mathrm{spread} = -12.73$ pp.

**Prediction 4 (manipulation), CONFIRMED with one qualification.** Zero cells
differ in input spike count, the per-sample time multiset is identical on every
row, and the per-timestep input profile is identical to $0.0$. The
amplitude-time rank correlation moves from $-0.99$ to $\approx-0.03$ on four
protocols. On KDDCup99 it moves from $-0.860$ to $-0.210$ rather than from
$-1$ to $0$: that protocol has heavy amplitude ties, so many features share a
spike time and permuting within a tied multiset cannot fully decorrelate
amplitude from time. The control is therefore weaker on KDDCup99 than
elsewhere, and its $\Delta_\mathrm{map}$ should be read as a lower bound.

**Outcome-map branch taken: none of the three cleanly.** The registered
branches asked whether $\Delta_\mathrm{map}$ carries most of
$\Delta_\mathrm{time}$. The measured pattern is different and stronger than any
branch anticipated: $\Delta_\mathrm{map}$ is **positive on all five protocols**
($+3.7$ to $+23.9$ pp) while $\Delta_\mathrm{spread}$ changes sign
($-12.7$ to $+30.6$ pp). The sign-changing character of the published
$\Delta_\mathrm{time}$ is therefore a property of temporal *spreading*, not of
the amplitude-to-time *code*. No published number is retracted; what changes is
which component the protocol dependence belongs to. Because this was not a
registered branch, it is reported as an unregistered post-hoc observation and
not as a confirmed hypothesis.
