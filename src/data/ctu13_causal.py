"""CTU-13 under three aggregation/partition conditions (round-2 follow-up).

The production CTU-13 loader computes per-SrcAddr host statistics over an entire
scenario and merges them onto every flow, so a test flow's features are computed
partly *from test flows*. That is a feature-level leak independent of how rows
are split, and it is the most likely explanation of the 100.00 macro-F1 ceiling.

Separating the two effects needs three conditions, not two:

  legacy    global per-scenario aggregates + random flow split   (reproduces the
                                                                  published ceiling)
  causal    online-causal aggregates       + random flow split   (isolates the
                                                                  aggregation leak)
  scenario  online-causal aggregates       + scenario-disjoint   (adds scenario shift)

    delta_aggregation = M(causal)   - M(legacy)
    delta_scenario    = M(scenario) - M(causal)

Online-causal means: within a scenario, flows are processed in StartTime order and
each flow's host features are computed from *strictly earlier* flows of the same
SrcAddr only. State is reset at every scenario boundary and never carried across
scenarios. The timestamp audit supports this: all 13 scenarios parse at 100%, are
monotonically ordered, contain no negative intervals, resolve to 1 microsecond,
and no two scenarios overlap in time.
"""
from __future__ import annotations

import glob
import math
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

HOST_COLS = ["src_flow_count", "src_unique_dsts", "src_unique_dports",
             "src_unique_sports", "src_unique_protos", "src_unique_states",
             "src_total_dur", "src_mean_dur", "src_std_dur",
             "src_total_pkts", "src_mean_pkts", "src_total_bytes",
             "src_mean_bytes", "src_total_src_bytes", "src_mean_src_bytes",
             "src_port_entropy", "src_proto_entropy"]

# Raw identifier/time columns retained in the persisted artifact so causal host
# features can be recomputed inside an arbitrary partition (condition B). The
# first build of _three_condition.parquet dropped these, which is why the
# causal_random diagnostic branch failed with KeyError: 'SrcAddr'.
#
# They are metadata, not features: they must never enter a model feature matrix.
# SrcAddr/DstAddr are host identity, and the CTU-13 label is a property of the
# host -- feeding them to a classifier would manufacture the exact shortcut this
# experiment exists to measure.
METADATA_COLS = ["SrcAddr", "DstAddr", "StartTime", "_t", "_row_pos"]


def require_raw_columns(frame: pd.DataFrame) -> None:
    """Fail loudly before causal recomputation if the artifact lacks raw columns."""
    required = set(METADATA_COLS)
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(
            f"causal_random requires raw aggregation columns; missing: {sorted(missing)}"
        )


def manifest_sha256(path) -> str:
    """Hash of a frozen manifest, to be recorded in every result file it produced."""
    import hashlib
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_frozen_folds(path, frame: pd.DataFrame = None) -> dict:
    """Load a frozen fold manifest. THE loader every consumer must use.

    Never re-solves the MIP: the solver has a time limit and no unique optimum, so
    re-solving during a training invocation can return a different feasible
    assignment and silently invalidate results computed against the old one.

    Coerces scenario ids back to int. They were written through
    json.dump(default=str), and isin(["2","4","8"]) against an int64 column matches
    zero rows -- an empty partition, not an error.

    When `frame` is given, verifies realised partition sizes against the manifest.
    Attaches `manifest_sha256` for recording into result files.
    """
    import json
    path = Path(path)
    fo = json.load(open(path))
    fo["folds"] = {int(k): {**v, **{key: [int(s) for s in v[key]]
                                    for key in ("train_scenarios", "val_scenarios",
                                                "test_scenarios")}}
                   for k, v in fo["folds"].items()}
    fo["manifest_sha256"] = manifest_sha256(path)

    if frame is not None:
        for k, f in fo["folds"].items():
            for part in ("train", "val", "test"):
                expected_rows = int(f["rows"][part])
                actual_rows = int(frame.scenario.isin(f[f"{part}_scenarios"]).sum())
                if actual_rows != expected_rows:
                    raise RuntimeError(
                        f"Frozen partition mismatch: expected {expected_rows}, "
                        f"got {actual_rows}"
                    )
    return fo


def _entropy_from_counts(counter: Dict, total: int) -> float:
    if total <= 0:
        return 0.0
    h = 0.0
    for c in counter.values():
        if c:
            p = c / total
            h -= p * math.log2(p)
    return h


def causal_host_features(df: pd.DataFrame) -> pd.DataFrame:
    """Per-flow host statistics from strictly earlier flows of the same SrcAddr.

    One chronological pass per scenario with running per-source accumulators. The
    first flow of a source has no history, so its features are zero -- that is the
    honest online value, not an imputation.
    """
    n = len(df)
    out = {c: np.zeros(n, dtype=np.float64) for c in HOST_COLS}
    cnt: Dict = defaultdict(int)
    s_dur: Dict = defaultdict(float); s_dur2: Dict = defaultdict(float)
    s_pkt: Dict = defaultdict(float); s_byt: Dict = defaultdict(float)
    s_sby: Dict = defaultdict(float)
    dsts: Dict = defaultdict(set); dports: Dict = defaultdict(set)
    sports: Dict = defaultdict(set); protos: Dict = defaultdict(set)
    states: Dict = defaultdict(set)
    dport_c: Dict = defaultdict(lambda: defaultdict(int))
    proto_c: Dict = defaultdict(lambda: defaultdict(int))

    src = df["SrcAddr"].to_numpy()
    dst = df["DstAddr"].to_numpy(); dp = df["Dport"].to_numpy()
    sp = df["Sport"].to_numpy(); pr = df["Proto"].to_numpy()
    st = df["State"].to_numpy()
    dur = df["Dur"].to_numpy(float); pkt = df["TotPkts"].to_numpy(float)
    byt = df["TotBytes"].to_numpy(float); sby = df["SrcBytes"].to_numpy(float)

    for i in range(n):
        s = src[i]
        k = cnt[s]
        if k:                                   # emit BEFORE updating -> causal
            out["src_flow_count"][i] = k
            out["src_unique_dsts"][i] = len(dsts[s])
            out["src_unique_dports"][i] = len(dports[s])
            out["src_unique_sports"][i] = len(sports[s])
            out["src_unique_protos"][i] = len(protos[s])
            out["src_unique_states"][i] = len(states[s])
            out["src_total_dur"][i] = s_dur[s]
            out["src_mean_dur"][i] = s_dur[s] / k
            var = max(s_dur2[s] / k - (s_dur[s] / k) ** 2, 0.0)
            out["src_std_dur"][i] = math.sqrt(var)
            out["src_total_pkts"][i] = s_pkt[s]
            out["src_mean_pkts"][i] = s_pkt[s] / k
            out["src_total_bytes"][i] = s_byt[s]
            out["src_mean_bytes"][i] = s_byt[s] / k
            out["src_total_src_bytes"][i] = s_sby[s]
            out["src_mean_src_bytes"][i] = s_sby[s] / k
            out["src_port_entropy"][i] = _entropy_from_counts(dport_c[s], k)
            out["src_proto_entropy"][i] = _entropy_from_counts(proto_c[s], k)
        cnt[s] = k + 1
        s_dur[s] += dur[i]; s_dur2[s] += dur[i] * dur[i]
        s_pkt[s] += pkt[i]; s_byt[s] += byt[i]; s_sby[s] += sby[i]
        dsts[s].add(dst[i]); dports[s].add(dp[i]); sports[s].add(sp[i])
        protos[s].add(pr[i]); states[s].add(st[i])
        dport_c[s][dp[i]] += 1; proto_c[s][pr[i]] += 1
    return pd.DataFrame(out, index=df.index)


def build_conditions(data_dir: Path = None, out: Path = None) -> Path:
    """Emit a parquet with BOTH aggregate variants plus scenario id and time."""
    from src.paths import CONFERENCE_ROOT, dataset_dir
    data_dir = Path(data_dir) if data_dir else dataset_dir("ctu13")
    out = Path(out) if out else CONFERENCE_ROOT / "results/cache/ctu13_three_condition.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        return out
    from src.data.ctu13 import _read_one, _aggregate_per_source

    frames = []
    for si, f in enumerate(sorted(glob.glob(str(data_dir / "binetflow" / "scenario-*-*.binetflow")))):
        df = _read_one(Path(f))
        lbl = df["Label"].astype(str)
        df = df[lbl.str.contains("Botnet", na=False) | lbl.str.contains("Normal", na=False)].copy()
        if not len(df):
            continue
        for c in ("Dur", "TotPkts", "TotBytes", "SrcBytes", "sTos", "dTos"):
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)
        df["Sport"] = pd.to_numeric(df["Sport"], errors="coerce").fillna(0)
        df["Dport"] = pd.to_numeric(df["Dport"], errors="coerce").fillna(0)
        for c in ("Proto", "Dir", "State"):
            df[c] = df[c].astype(str).str.strip()
        df["y"] = lbl.loc[df.index].str.contains("Botnet", na=False).astype(np.int64)
        df["scenario"] = si
        df["_t"] = pd.to_datetime(df["StartTime"], errors="coerce", format="mixed")
        df = df.sort_values("_t", kind="stable").reset_index(drop=True)

        legacy = df.merge(_aggregate_per_source(df), on="SrcAddr", how="left")
        legacy = legacy[HOST_COLS].reset_index(drop=True)
        legacy.columns = [f"legacy__{c}" for c in HOST_COLS]
        causal = causal_host_features(df)
        causal.columns = [f"causal__{c}" for c in HOST_COLS]

        base = df[["Dur", "TotPkts", "TotBytes", "SrcBytes", "sTos", "dTos",
                   "Sport", "Dport", "Proto", "Dir", "State", "y", "scenario",
                   # Retained for on-demand causal recomputation; see METADATA_COLS.
                   # _t is required because causal_host_features assumes
                   # chronological arrival, so a re-sort inside a random partition
                   # must sort by time and not by index.
                   "SrcAddr", "DstAddr", "StartTime", "_t"]]
        frames.append(pd.concat([base.reset_index(drop=True), legacy, causal], axis=1))
        print(f"  scenario {si}: {len(df)} flows", flush=True)

    big = pd.concat(frames, ignore_index=True)
    # Assigned once, after the concat, so it addresses the whole table. Consumers
    # scatter partition-local results back with H[g["_row_pos"]] instead of relying
    # on index alignment surviving a groupby.
    big["_row_pos"] = np.arange(len(big), dtype=np.int64)
    big.to_parquet(out)
    print(f"[ctu13] wrote {out}  rows={len(big)}  botnet={int(big.y.sum())}", flush=True)
    return out


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    build_conditions()


def partition_local_causal(d: "pd.DataFrame", partitions, cau_cols: list) -> "pd.DataFrame":
    """Recompute causal host features independently inside each partition.

    Condition B asks what the aggregates look like when a partition may only see
    its own rows, so the features cannot come from the precomputed causal__*
    columns (those were built over the whole scenario) -- they must be recomputed,
    which is why the raw identifier and time columns have to survive into the
    persisted artifact.

    Results are scattered back positionally through _row_pos. Label-based
    reindex-after-groupby is correct while the frame carries a unique contiguous
    RangeIndex, but nothing enforces that invariant, and a silent misalignment
    here would surface as a plausible metric rather than an error.
    """
    require_raw_columns(d)
    n = len(d)
    pos_all = d["_row_pos"].to_numpy()
    if not np.array_equal(pos_all, np.arange(n, dtype=pos_all.dtype)):
        raise ValueError("_row_pos must be 0..n-1 over the full table")

    H = np.full((n, len(HOST_COLS)), np.nan, dtype=np.float64)
    seen = np.zeros(n, dtype=np.int64)
    for part in partitions:
        # sort by _t, not by index: causal_host_features assumes chronological
        # arrival, and a random partition destroys the artifact's row order.
        sub = d.iloc[part].sort_values(["scenario", "_t"], kind="stable")
        for _, g in sub.groupby("scenario", sort=True):
            f = causal_host_features(g)
            pos = g["_row_pos"].to_numpy()
            H[pos] = f.to_numpy()
            seen[pos] += 1
    assert seen.min() == 1 and seen.max() == 1, "every row written exactly once"
    assert not np.isnan(H).any(), "no unwritten positions"
    return pd.DataFrame(H, columns=cau_cols, index=d.index)
