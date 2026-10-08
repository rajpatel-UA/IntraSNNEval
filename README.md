<div align="center">

# The Value of Spike Timing

### A Leakage-Resistant Benchmark of SNN Design Choices for Network Intrusion Detection

Raj Patel, Shaswata Mitra, David Amebley, Taye Akinrele, Sayanton Dibbo, Shahram Rahimi

Department of Computer Science, The University of Alabama

*IEEE International Conference on Cognitive Machine Intelligence (CogMI), 2026*

[![arXiv](https://img.shields.io/badge/arXiv-2606.01442-b31b1b.svg)](https://arxiv.org/abs/2606.01442)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-ee4c2c.svg)](https://pytorch.org/)
[![snnTorch 0.9.4](https://img.shields.io/badge/snnTorch-0.9.4-green.svg)](https://github.com/jeshraghian/snntorch)

</div>

This repository contains the code, frozen split manifests, per-run records, and
analysis outputs behind the paper
([arXiv:2606.01442](https://arxiv.org/abs/2606.01442); the IEEE version is
forthcoming). It evaluates **9 snnTorch neuron families ×
3 spike encodings = 27 SNN configurations** on four intrusion-detection
benchmarks (NSL-KDD, KDDCup99, CIC-IDS2017, CTU-13). It then repeats the
comparison under a **leakage-resistant confirmation protocol** and separates
what latency coding contributes through spike timing from what it contributes
through temporal spreading.

## Overview

Network-flow records are static feature vectors, so the spike encoder creates
whatever temporal structure an SNN sees. This study treats the neuron model,
the spike encoding, and the evaluation protocol as separate factors:

1. **Screening.** All 27 configurations share one architecture and training
   procedure and are evaluated on one partition per dataset over five seeds
   (20 dataset–seed blocks). Configurations are ranked by mean macro-F1 rank.
2. **Leakage-resistant confirmation.** The configurations are re-evaluated with
   transforms and categorical vocabularies fitted on training rows only,
   partitions fixed independently of the model seed, capture- and
   scenario-aware separation, causal CTU-13 host aggregation, and audits of
   duplication and class support.
3. **Spike-timing controls.** Matched-input arms hold the active spike set fixed
   and change only its temporal organization. This splits the latency-coding
   effect into an *amplitude-to-time mapping* term and a *temporal spreading*
   term.
4. **Computational activity.** Input spikes, internal spikes, and
   fanout-weighted synaptic operations (SOPs) are reported separately.

All models are two-layer spiking MLPs (H = 128, T = 25, β = 0.85, α = 0.90,
arctangent surrogate). They are trained with Adam (lr 10⁻³, batch 128,
10 epochs) using seeds {42, 43, 44, 45, 46}.

## Key findings

- **Design choice.** `LeakyParallel/latency` has the lowest mean rank in both
  the screening sweep (4.03 over 20 blocks) and the complete confirmation sweep
  (5.92 over 25 blocks). It is **not** statistically separated from
  `Leaky/latency`: 14 of the 27 configurations lie within one Nemenyi critical
  difference of the leader.
- **Rank stability.** Screening and confirmation agree on 317 of 351
  configuration pairs (Kendall τ = 0.8063, permutation p < 10⁻⁴), although
  absolute performance changes under the stricter protocol.
- **Protocol effect.** On CTU-13, `LeakyParallel/latency` falls from 100.00%
  macro-F1 under a random flow split to 92.01% under causal aggregation with
  scenario-disjoint folds. Fold means range from 77.02% to 99.95%.
- **Spike timing.** At T = 25, mapping feature magnitude to spike time improves
  macro-F1 on all five confirmation protocols (+3.71 to +23.92 pp). Spreading
  the same spikes through time without that mapping ranges from −12.73 to
  +30.60 pp.
- **Activity.** Rate coding emits 9.3–19.7× more input spikes than latency
  coding but only 1.7–2.5× more total network spikes. It requires 7.2–13.0×
  more SOPs per sample. SOPs are operation counts, not energy measurements.

## Repository structure

```
.
├── src/                     library code
│   ├── data/                dataset loaders, screening and confirmation protocols, split audit
│   ├── encoding/            rate, latency, delta, and permuted-latency (timing control) encoders
│   ├── neurons/             snnTorch neuron factory (9 families)
│   ├── models/              two-layer spiking MLP
│   ├── training.py          shared training and evaluation loop
│   ├── metrics.py, stats.py metrics, Friedman/Nemenyi, Wilcoxon/Holm, bootstrap
│   └── readout.py           per-timestep readout over saved run artifacts
├── configs/                 one config.yaml per (protocol, neuron, encoding)
├── experiments/             thin run.py entry point per config
├── scripts/                 sweep drivers, analyses, table and figure generation, checks
├── results/
│   ├── manifests/           frozen split manifests for every confirmation protocol
│   ├── v1_submitted/        screening sweep (540 runs) and the 60 CTU-13 condition runs
│   ├── runs*/               per-run records (results.json) and summary CSVs
│   ├── analysis/            every statistic reported in the paper, plus PROVENANCE.json
│   └── ARTIFACTS.*          checksums of the run artifacts and the record of their archive
├── paper/                   generated figures, LaTeX tables, number macros (manuscript prose not included)
├── docs/DATASETS.md         dataset sources, expected layout, CTU-13 download script
└── *.md                     pre-registration and diagnostic specifications (see below)
```

## Installation

All reported results were produced on Linux with Python 3.12.3, PyTorch 2.12.0
(CUDA 13.0), and snnTorch 0.9.4, on one NVIDIA H200 NVL.

```bash
git clone https://github.com/rajpatel-UA/IntraSNNEval.git
cd IntraSNNEval

python -m venv .venv && source .venv/bin/activate
# Install a PyTorch build that matches your CUDA or CPU setup first, for example:
#   pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt

# Sanity check: list the registered protocols (does not need the datasets)
python -c "from src.data import DATASET_LOADERS as D; print(sorted(D))"
```

`requirements.txt` lists the direct dependencies with minimum versions.
`requirements-lock.txt` pins the exact environment the results came from. It
includes CUDA 13 packages, so it installs only on Linux with a compatible GPU
driver.

**What to expect.** Re-running the analyses on the committed run records
reproduces the reported numbers. With other library versions, the last digit
of some floating-point values can differ. Retraining on different hardware or
software gives results close to, but not bit-identical with, the reported
ones.

## Datasets

The datasets are not redistributed. Download them from their original
providers and point the code at them:

```bash
export HISNN_DATASET_ROOT=/path/to/datasets   # or: ln -s /path/to/datasets dataset
```

See **[docs/DATASETS.md](docs/DATASETS.md)** for sources, the exact file
layout, the CTU-13 download script, and how to check your copy against the
frozen manifests.

## Evaluation protocols

Protocols are selected by name from the registry in `src/data/__init__.py`.
The suffix tells you which protocol generation you get. **Do not combine the
two generations in one table.** The difference between them is a result.

| | Screening (`nslkdd`, `kddcup99`, `cicids2017`, `ctu13`) | Confirmation (`nslkdd_v2`, `kddcup99_v2`, `cicids2017_v2`, `ctu13_v2_f0`–`f3`) |
| --- | --- | --- |
| Categorical vocabulary | Fitted on all splits | Training rows only, with an `<unk>` level |
| Scaler | Training rows only | Training rows only |
| Partition | Random; keyed on the model seed for CIC-IDS2017 and CTU-13 | Frozen, with a partition seed independent of the model seed |
| Group structure | Ignored | Within-capture order-disjoint (CIC-IDS2017), scenario-disjoint folds (CTU-13) |
| CTU-13 host aggregates | Whole scenario | Causal, computed from preceding flows of the same source |
| Audit | None | Training refuses to start if a claimed separation is violated |

`cicids2017_daydisjoint` builds a day-disjoint CIC-IDS2017 partition. It is
included in the split audit, but no models were trained on it for the paper.

## Reproducing the paper

There are three levels of reproduction, from cheapest to most expensive.

### 1. Rebuild tables and figures from committed analysis outputs (no data, no GPU)

```bash
python scripts/make_tables.py --tag v1      # paper/tables/*.tex
python scripts/make_figures.py              # main paper figures in paper/figures/
python scripts/make_fig_far_transfer.py     # operating-point transfer figure (Fig. 8)
python scripts/make_frozen_numbers.py       # paper/sec_frozen_numbers.tex (every reported number as a macro)
```

None of these need the [run artifacts](#run-artifacts) or the datasets. To run
them together with the screening statistics and the consistency checks, use
`python scripts/finalize.py --groups 0 1`.

`results/analysis/PROVENANCE.json` records the SHA-256 of every analysis
artifact and manuscript source that the reported numbers come from.
`scripts/check_macros.py` checks that `paper/sec_frozen_numbers.tex` rebuilds
byte for byte and that every macro the included LaTeX uses is defined. It then
regenerates that file.

### 2. Re-run the statistical analyses (needs the run artifacts)

```bash
python scripts/rank_analysis.py --input results/v1_submitted/sweep_540.csv --tag v1   # screening ranks, Friedman/Nemenyi, focused tests
python scripts/axis_analysis.py --axis encoding                                         # confirmation, encoding axis
python scripts/axis_analysis.py --axis neuron                                           # confirmation, neuron axis
python scripts/protocol_shift.py                                                        # complete confirmation ranking, Kendall τ
python scripts/rank_agreement.py                                                        # RBO, AP correlation, top-k overlap
python scripts/finalize.py                                                              # regenerate the full evidence package
```

Only the screening analysis (`rank_analysis.py`) and `rank_agreement.py` run
from committed files alone. Every analysis of the confirmation runs reads
per-sample predictions from the per-run `artifacts/*.npz` files (see
[Run artifacts](#run-artifacts)), and reports "no runs" without them.
`finalize.py` therefore checks for the artifacts first and stops if any are
missing, because some of its steps would otherwise overwrite committed results
with incomplete versions. `--groups 0 1` limits it to the steps that do not
need them. If committed files do get overwritten, restore them
with `git checkout -- results paper`. Several steps also need the raw datasets:
`feature_sparsity.py`, `data_request.py`, `support_audit.py`, and
`nsl_known_unseen.py`.

### 3. Retrain from scratch (datasets + GPU)

```bash
# Gate: build and audit every confirmation protocol. Nothing should train until this is clean.
python scripts/split_audit.py

# Screening sweep: 27 configurations × 5 seeds on each screening protocol
for p in nslkdd kddcup99 cicids2017 ctu13; do
  python scripts/run_sweep.py "$p" --seeds 42 43 44 45 46 --workers 4
done

# Confirmation: two-axis design (B1, B2, C1, C2) and the complete 27-configuration sweep (E2a–E2e)
python scripts/run_program.py --dry-run     # show the plan and an ETA
python scripts/run_program.py

# Post hoc controls (three seeds, LeakyParallel)
python scripts/run_e1.py     # amplitude-gate controls
python scripts/run_a2.py     # spread-only (permuted timing) arm at T = 25
python scripts/run_a2t.py    # timing components at T ∈ {5, 10, 50}
python scripts/run_e3.py     # encodings across the timestep budget

# CTU-13 under three aggregation/partition conditions (60 runs, LeakyParallel/latency)
python scripts/run_ctu13_conditions.py --dry-run
python scripts/run_ctu13_conditions.py --workers 4
python scripts/ctu13_strict_vanilla.py --runs-dir results/runs_ctu13_conditions \
    --out results/runs_ctu13_conditions/ctu13_strict_vanilla.csv
```

Runs write to `results/runs/<protocol>/SNN_<Neuron>/<encoding>/seed_<S>/` and
append to `results/runs/summary.csv`. Sweeps are idempotent: re-running a
command skips completed runs and resumes the rest. Validate a single run with
`python scripts/smoke_check.py <run_dir>`.

**Compute.** The summed per-run training time was about 198 GPU-hours for the
540-run screening sweep and about 404 GPU-hours for the confirmation runs.
Runs were executed 4–6 at a time on one NVIDIA H200, which inflates per-run
wall-clock time.

## Paper-to-code map

Scripts are in `scripts/`. Results are under `results/`.

| Paper element | Scripts | Results |
| --- | --- | --- |
| Screening ranking (Fig. 4, Tables XII–XIV) | `rank_analysis.py` | `results/analysis/v1/` |
| Split and duplication audit (Tables III, VIII) | `split_audit.py` | `results/analysis/split_audit.csv`, `results/manifests/` |
| Encoding and neuron axes (Tables V, XV, Fig. 5) | `axis_analysis.py` | `results/analysis/v2_encoding_axis/`, `v2_neuron_axis/` |
| Complete confirmation and rank agreement (Table XVI, Fig. 6) | `protocol_shift.py`, `rank_agreement.py` | `results/analysis/protocol_shift/`, `rank_agreement/`, `paper/figures/fig_rank_shift.*` |
| CTU-13 protocol effect and scenario transfer (Table VI) | `run_ctu13_conditions.py`, `ctu13_strict_vanilla.py` | `results/v1_submitted/ctu13_conditions/`, `ctu13_strict_vanilla.csv` |
| Duplicate-free sensitivity (Table X) | `dedup_analysis.py` | `results/analysis/dedup/` |
| Temporal convergence, T95/T99 (Table XX, Fig. 2) | `early_readout.py` | `results/analysis/early_readout/` |
| Spikes and SOPs by layer (Tables XVIII, XXIV, Fig. 7) | `efficiency.py`, `spike_sop_decomposition.py` | `results/analysis/efficiency/`, `spike_sop_decomposition.csv` |
| Macro-F1 against SOPs, all 27 configurations (Fig. 3) | `pareto27.py`, `make_figures.py` | `results/analysis/pareto27.csv` |
| Timing decomposition (Tables VII, XXI–XXIII) | `run_a2.py`, `a2_analysis.py`, `run_a2t.py`, `a2t_analysis.py` | `results/analysis/a2/`, `a2t/` |
| Gate controls and timing isolation (Table XIX) | `run_e1.py`, `e1_analysis.py` | `results/runs_e1/`, `results/analysis/e1_threshold_ablation.csv`, `e1_timing_isolation.csv` |
| Feature sparsity and gate mass (Table XVII) | `feature_sparsity.py`, `appendix_data.py` | `results/analysis/feature_sparsity.csv`, `xbar_active_prediction.csv`, `appendix/` |
| Operating-point transfer (Table XXV, Fig. 8) | `security_analysis.py`, `make_fig_far_transfer.py` | `results/analysis/security/` |
| Known vs. unseen attack types | `nsl_known_unseen.py` | `results/analysis/known_unseen/` |
| Per-class performance (Tables XXVI, XXVII) | `per_class_analysis.py`, `data_request.py` | `results/analysis/per_class/`, `data_request/C1_per_class.csv` |
| Delta reduces to one-step thresholding (Sec. IV-D) | `data_request.py` | `results/analysis/data_request/C2_delta_spikes_per_t.npy` |
| Distribution shift (Table XXVIII, Fig. 9) | `covariate_shift.py` | `results/analysis/covariate_shift/`, `paper/figures/fig_shift_timing.*` |
| Macro-F1 label-set audit (Appendix A-C) | `dual_macro_f1.py`, `support_audit.py` | `results/analysis/dual_macro_f1.csv`, `AMENDMENT_cic_u2r_support.md` |

## Study records

The analysis plan was written down before the corresponding results and kept
in version control during the study:

| Document | Contents |
| --- | --- |
| [`PREREGISTRATION.md`](PREREGISTRATION.md) | Scope, metrics, ranking rules, and statistical tests for screening and the two-axis confirmation |
| [`PREREGISTRATION_ADDENDUM.md`](PREREGISTRATION_ADDENDUM.md) | The complete 27-configuration confirmation sweep, added after the two-axis results were known |
| [`DIAGNOSTIC_SPECIFICATIONS.md`](DIAGNOSTIC_SPECIFICATIONS.md) | Predictions and stopping rules for the post hoc diagnostics |
| [`results/analysis/AMENDMENT_cic_u2r_support.md`](results/analysis/AMENDMENT_cic_u2r_support.md) | Post-run macro-F1 label-set correction |

As the paper states, these plans were kept in local version control. The
available records, including this repository's history, do not independently
establish that they were committed before the runs.

## Run artifacts

Per-run prediction artifacts (`results/runs*/**/artifacts/*.npz`, about 2.2 GB)
and model checkpoints (`checkpoint.pt`) are **not tracked in git** because of
their size. Everything needed for level 1 above, and the `results.json` record
of every run, is tracked.

The artifacts are two files per run for 960 runs (the confirmation sweep and
the post hoc arms), 1,920 files in total:

- `artifacts/test_prefix.npz`: per-sample test predictions and the readout at
  every timestep
- `artifacts/val_scores.npz`: validation labels and scores

`results/ARTIFACTS.sha256` lists the SHA-256 of every artifact file.
`results/ARTIFACTS.json` records the archive's name, size and hash, and will
hold its download URL and DOI once it is published.

```bash
python scripts/release_artifacts.py fetch     # download, check the hash, extract
python scripts/release_artifacts.py verify    # check the files on disk against the list
python scripts/release_artifacts.py build     # rebuild the archive (identical bytes from identical files)
```

The archive is not published yet, so `fetch` currently stops with a message.

## Scope

This is a controlled **intra-SNN** design-space study. It asks which neuron and
encoding an SNN-based detector should use at a shared operating point, not
whether SNNs outperform conventional classifiers. No non-spiking baselines are
trained, and `scripts/check_scope.py` enforces that boundary. Neuron-family
results reflect one shared hyperparameter setting rather than per-family
tuning. See Section V of the paper for the full limitations.

## Citation

If you use this code or the results, please cite the paper. Until the IEEE
CogMI 2026 proceedings version is available, please cite the arXiv version:

```bibtex
@misc{patel2026spiketiming,
  title         = {The Value of Spike Timing: A Leakage-Resistant Benchmark of {SNN} Design Choices for Network Intrusion Detection},
  author        = {Patel, Raj and Mitra, Shaswata and Amebley, David and Akinrele, Taye and Dibbo, Sayanton and Rahimi, Shahram},
  year          = {2026},
  eprint        = {2606.01442},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CR},
  doi           = {10.48550/arXiv.2606.01442},
  url           = {https://arxiv.org/abs/2606.01442},
  note          = {To appear in IEEE CogMI 2026}
}
```

Please also cite the datasets you use. The references are listed in
[docs/DATASETS.md](docs/DATASETS.md).

## License

The code in this repository is released under the [MIT License](LICENSE). The
license does not cover the datasets, which remain subject to their providers'
terms.

## Acknowledgments

This work was supported by the Hewson Faculty Award (#1003-31880-214241) at
The University of Alabama and by the NVIDIA Academic Grant Program.

## Contact

Corresponding author: Sayanton Dibbo ([sdibbo@ua.edu](mailto:sdibbo@ua.edu)).
For questions about the code, please open a GitHub issue.
