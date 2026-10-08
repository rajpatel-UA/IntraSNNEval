# Datasets

The raw datasets are **not redistributed** in this repository. They are large
and each is published under its provider's own terms. Download them from the
original sources below and cite the corresponding papers.

## Dataset root

Every loader resolves the dataset root through `src/paths.py`, in this order:

1. `$HISNN_DATASET_ROOT`, if set
2. `<repo>/dataset` (a directory or a symlink)
3. `<repo>/../dataset`

```bash
export HISNN_DATASET_ROOT=/path/to/datasets
# or
ln -s /path/to/datasets dataset
```

Nothing writes to the dataset root. Derived caches go to `results/cache/`.

## Expected layout

```
<root>/
├── nslkdd/
│   ├── KDDTrain+.txt
│   ├── KDDTest+.txt
│   └── KDDTest-21.txt
├── kddcup99/
│   ├── kddcup.data_10_percent_corrected      # 494,021 rows
│   └── corrected                             # 311,029 rows (corrected/corrected also accepted)
├── cicids2017/
│   ├── Monday-WorkingHours.pcap_ISCX.csv
│   ├── Tuesday-WorkingHours.pcap_ISCX.csv
│   ├── Wednesday-workingHours.pcap_ISCX.csv
│   ├── Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv
│   ├── Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv
│   ├── Friday-WorkingHours-Morning.pcap_ISCX.csv
│   ├── Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv
│   └── Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv
└── ctu13/
    └── binetflow/
        └── scenario-<capture-id>-<capture>.binetflow   # 13 files, capture ids 42–54
```

File names must match exactly, including the original spelling of
`Wednesday-workingHours` and `Infilteration`.

## Sources

| Dataset | Source | Files used | Reference |
| --- | --- | --- | --- |
| NSL-KDD | [UNB Canadian Institute for Cybersecurity](https://www.unb.ca/cic/datasets/nsl.html) | `KDDTrain+.txt`, `KDDTest+.txt`, `KDDTest-21.txt` | Tavallaee et al., CISDA 2009 |
| KDD Cup 1999 | [UCI KDD Archive](https://kdd.ics.uci.edu/databases/kddcup99/kddcup99.html) | 10% training subset and the labelled `corrected` test set | KDD Cup 1999 |
| CIC-IDS2017 | [UNB Canadian Institute for Cybersecurity](https://www.unb.ca/cic/datasets/ids-2017.html) | The eight per-capture CSVs from `MachineLearningCSV` | Sharafaldin et al., ICISSP 2018 |
| CTU-13 | [Stratosphere Laboratory, CTU Prague](https://www.stratosphereips.org/datasets-ctu13) | Raw bidirectional NetFlow (`.binetflow`) for all 13 scenarios | García et al., Computers & Security 2014 |

### CTU-13

The CTU-13 loader needs the **raw Stratosphere `.binetflow` files**. Common
pre-processed mirrors drop `SrcAddr`, `DstAddr`, and port columns, which the
causal per-source-host aggregation requires. The raw files keep the full
15-column schema (`StartTime, Dur, Proto, SrcAddr, Sport, Dir, DstAddr, Dport,
State, sTos, dTos, TotPkts, TotBytes, SrcBytes, Label`). The total download is
about 2.5 GB.

Scenarios 1–13 correspond to `CTU-Malware-Capture-Botnet-42` through `-54`.
The loader reads files in sorted order. Keep the `scenario-<capture-id>-`
prefix so that the scenario indices match the frozen fold manifests in
`results/manifests/`.

```bash
mkdir -p "$HISNN_DATASET_ROOT/ctu13/binetflow" && cd "$HISNN_DATASET_ROOT/ctu13/binetflow"
while read sc bn cap; do
  url="https://mcfp.felk.cvut.cz/publicDatasets/CTU-Malware-Capture-Botnet-${bn}/detailed-bidirectional-flow-labels/${cap}"
  wget --continue --tries=3 --timeout=120 "$url" -O "scenario-${bn}-${cap}"
done <<'EOF'
1 42 capture20110810.binetflow
2 43 capture20110811.binetflow
3 44 capture20110812.binetflow
4 45 capture20110815.binetflow
5 46 capture20110815-2.binetflow
6 47 capture20110816.binetflow
7 48 capture20110816-2.binetflow
8 49 capture20110816-3.binetflow
9 50 capture20110817.binetflow
10 51 capture20110818.binetflow
11 52 capture20110818-2.binetflow
12 53 capture20110819.binetflow
13 54 capture20110815-3.binetflow
EOF
```

## Label mappings

NSL-KDD and KDDCup99 use the fixed five-class taxonomy (Normal, DoS, Probe,
R2L, U2R) given in Table IX of the paper. CIC-IDS2017 uses the study-specific
five-class grouping in Table XI. The KDD-style group names identify the
implemented task. They do not imply that attack categories are equivalent
across datasets. CTU-13 is evaluated as binary Normal versus Botnet, with
Background flows removed. The mappings are implemented in `src/data/`.

## Verifying a download

`python scripts/split_audit.py` builds every confirmation protocol, checks the
claimed group separation, and rewrites its split manifest in
`results/manifests/`. Afterwards, `git diff results/manifests/` should show no
changes. Matching split sizes, duplicate fractions, and `extra.feature_hash`
values confirm that your copy of the data and the preprocessing reproduce the
paper's partitions.
