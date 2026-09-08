# Scoped ANN admission — 2026-09-08

Each backend directory contains nine independent, fixed-count executions of
`benchmarks/record_native_ann_execution.py --fuse-elementwise --parallel-rows`
with seeds 1 through 9, followed by `benchmarks/select_native_ann_measurements.py`.
The latter reconstructs the exact package pair, binds all report identities,
invokes scoped arbitration and retires the registration after execution.

| Owning target | Median package speedup | Median-order-statistic interval | Selection |
|---|---:|---|---|
| Super-Bear / SM120 | 0.9780x | [0.9629, 1.0233] | Original retained |
| Princess-Luna / gfx1151 | 0.9972x | [0.9906, 1.0142] | Original retained |

Neither lower bound exceeds 1.02. No production promotion. These are warm
host-wall package times including transfers and the host bridge for a 3x2
ANN input; they are not kernel times, overlap evidence or broad shape claims.
NVIDIA's retained packets were remeasured after the LLVM/Tessera builds ended.
Source and recorder hashes bind the recorded tree; stale evidence refuses.
