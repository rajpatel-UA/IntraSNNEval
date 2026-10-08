# Evidence package — regenerated 2026-08-13 21:20:45

Produced by `scripts/finalize.py`. Every manuscript number must come
from re-running this, not from memory.

## Coverage

```
235 runs on disk across 47 (protocol, variant) cells

protocol                 variant                      seeds  artifacts
cicids2017_v2            LeakyParallel/delta              5          5
cicids2017_v2            LeakyParallel/latency            5          5
cicids2017_v2            LeakyParallel/rate               5          5
ctu13_v2_f0              Alpha/latency                    5          5
ctu13_v2_f0              Lapicque/latency                 5          5
ctu13_v2_f0              Leaky/latency                    5          5
ctu13_v2_f0              LeakyParallel/delta              5          5
ctu13_v2_f0              LeakyParallel/latency            5          5
ctu13_v2_f0              LeakyParallel/rate               5          5
ctu13_v2_f0              RLeaky/latency                   5          5
ctu13_v2_f0              RSynaptic/latency                5          5
ctu13_v2_f0              SConv2dLSTM/latency              5          5
ctu13_v2_f0              SLSTM/latency                    5          5
ctu13_v2_f0              Synaptic/latency                 5          5
ctu13_v2_f1              Alpha/latency                    5          5
ctu13_v2_f1              Lapicque/latency                 5          5
ctu13_v2_f1              Leaky/latency                    5          5
ctu13_v2_f1              LeakyParallel/delta              5          5
ctu13_v2_f1              LeakyParallel/latency            5          5
ctu13_v2_f1              LeakyParallel/rate               5          5
ctu13_v2_f1              RLeaky/latency                   5          5
ctu13_v2_f1              RSynaptic/latency                5          5
ctu13_v2_f1              SConv2dLSTM/latency              5          5
ctu13_v2_f1              SLSTM/latency                    5          5
ctu13_v2_f1              Synaptic/latency                 5          5
kddcup99_v2              Alpha/latency                    5          5
kddcup99_v2              Lapicque/latency                 5          5
kddcup99_v2              Leaky/latency                    5          5
kddcup99_v2              LeakyParallel/delta              5          5
kddcup99_v2              LeakyParallel/latency            5          5
kddcup99_v2              LeakyParallel/rate               5          5
kddcup99_v2              RLeaky/latency                   5          5
kddcup99_v2              RSynaptic/latency                5          5
kddcup99_v2              SConv2dLSTM/latency              5          5
kddcup99_v2              SLSTM/latency                    5          5
kddcup99_v2              Synaptic/latency                 5          5
nslkdd_v2                Alpha/latency                    5          5
nslkdd_v2                Lapicque/latency                 5          5
nslkdd_v2                Leaky/latency                    5          5
nslkdd_v2                LeakyParallel/delta              5          5
nslkdd_v2                LeakyParallel/latency            5          5
nslkdd_v2                LeakyParallel/rate               5          5
nslkdd_v2                RLeaky/latency                   5          5
nslkdd_v2                RSynaptic/latency                5          5
nslkdd_v2                SConv2dLSTM/latency              5          5
nslkdd_v2                SLSTM/latency                    5          5
nslkdd_v2                Synaptic/latency                 5          5

all cells at 5 seeds
```

## Steps

- OK       screening statistics (v1, 27 configurations)
- OK       strict confirmation — Axis A, encoding (neuron fixed)
- OK       strict confirmation — Axis B, neuron (encoding fixed)
- OK       strict per-protocol metric table (tab_strict_results)
- OK       sensitivity: protocol as the unit of analysis
- OK       mechanism — early readout, T95/T99, spike saving
- OK       mechanism — spike decomposition and SOPs
- OK       protocol robustness — split audit
- OK       protocol robustness — bidirectional class-support audit
- OK       protocol robustness — dual macro-F1 (fixed universe vs test-present)
- OK       protocol robustness — dedup sensitivity, ENCODING axis
- OK       protocol robustness — dedup sensitivity, NEURON axis
- OK       boundary — fixed FAR, per class, duplicate-free
- OK       boundary — NSL-KDD known vs unseen attack types
- OK       LaTeX tables
- OK       figures at IEEE print scale
- OK       frozen manuscript numbers
- OK       guard: terminology and literal results
- OK       guard: macros defined + provenance hashes
- OK       guard: SNN-only scope
