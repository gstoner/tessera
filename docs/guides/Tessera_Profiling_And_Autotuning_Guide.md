---
status: Informative
classification: Guide
authority: Profiling and autotuning workflows; defers schedule artifact semantics to docs/spec/shape-system.md and compiler autotuner implementation
last_updated: 2026-04-28
---

# Tessera Profiling and Autotuning Guide

Performance optimization in Tessera is driven by two loops:

- **Profiler loop:** capture runtime metrics, timelines, counters, and memory
  movement to identify the bottleneck.
- **Autotuning loop:** search legal Schedule IR choices, measure or estimate
  candidates, persist the best schedule artifact, and reuse it by shape and
  target.

The current Python foundation provides `tessera.profiler` and a callable
`tessera.autotune` facade over the compiler autotuner. The lower compiler stack
already contains a Bayesian/grid GEMM autotuner with SQLite cache support and
schedule-artifact hashing.

---

## 1. Runtime Profiler

Use a profiler session to collect operation metrics:

```python
from tessera import profiler

with profiler.session() as p:
    p.record(
        "matmul",
        latency_ms=3.21,
        flops=214.5e9,
        bytes_moved=4.85e9,
        peak_tflops=233.0,
        counters={"sm_occupancy_pct": 92.0},
    )
    p.record("softmax", latency_ms=0.67, flops=2.1e9, bytes_moved=661e6)

print(p.report())
```

Example report:

```text
Op       Latency(ms)  FLOPs(G)  Bandwidth(GB/s)  Efficiency(%)
matmul   3.210        214.500   1510.903         28.7
softmax  0.670        2.100     986.567          0.0
```

For callable regions:

```python
with profiler.session() as p:
    y = p.measure("step", lambda: model(batch), flops=1.2e12, bytes_moved=80e9)
```

The CLI surface mirrors the runtime profiler:

```bash
tessera-prof my_model.py --metrics=flops,bandwidth,occupancy
tessera-prof my_model.py --trace=trace.json
```

The current `tessera-prof` implementation records a lightweight inspection
event and can emit a Chrome Trace Event JSON file. As device execution is wired
through the runtime, this command should become the stable front door for kernel
latency, FLOPs, bandwidth, occupancy, memory, collective, and launch metrics.

Profiler events carry:

- Latency in milliseconds.
- FLOPs in G.
- Derived bandwidth in GB/s.
- Optional peak-normalized efficiency.
- Hardware counters such as SM occupancy, warp divergence, L2 hit rate, and
  shared-memory bank conflicts.

---

## 2. Timeline Traces

Profiler sessions export Chrome Trace Event JSON:

```python
with profiler.session() as p:
    p.record("matmul", latency_ms=3.21)
    p.record("softmax", latency_ms=0.67)

p.timeline("trace.json")
```

Open the resulting file in Chrome trace viewers or compatible observability
systems. Future runtime integrations should attach stream, device, rank, kernel
launch ID, graph hash, and schedule hash to each trace event.

---

## 3. Cost Models

Schedule IR should make the cost model explicit:

```mlir
%1 = "tessera.schedule.tile"(%0)
       {tile_sizes = [128, 128, 32], cost_model = "roofline"}
```

The public roofline helper estimates compute-vs-memory bounds:

```python
from tessera import autotune

model = autotune.RooflineCostModel(peak_tflops=312.0, bandwidth_gbps=2000.0)
estimate = model.estimate(flops=2 * 1024**3, bytes_moved=6 * 1024**2)
print(estimate.bound, estimate.latency_ms)
```

Tessera cost models should be keyed by:

- Op kind.
- Logical shape and derived shape bindings.
- Dtype and numeric policy.
- Layout and shard map.
- Movement plan.
- Target architecture and relevant hardware features.

---

## 4. Autotuning Workflow

The public entry point tunes GEMM-like workloads:

```python
from tessera import autotune, ops

cfg = autotune(ops.matmul, shapes=(1024, 1024, 1024), max_trials=20)
print(cfg)
```

Representative result:

```text
TuningResult(tflops=..., latency_ms=..., config=TuningConfig(...))
```

Current methods:

- `method="roofline"`: synthetic analytical evaluator.
- `method="grid"` / `method="bayesian"`: select the search policy path through
  the compiler autotuner when dependencies are available.
- `method="on_device"`: accepted as the forward-compatible API for measured
  candidate execution; until runtime kernels are wired, it uses the same
  deterministic evaluator.

Autotuning should run after shape and schedule feasibility have pruned illegal
tile/layout choices. It should never benchmark candidates that violate shape
divisibility, target fragment legality, shared-memory budgets, or movement
constraints.

---

## 5. Persistent Caches

Tessera stores tuning results in SQLite:

```text
$TESSERA_CACHE_DIR/autotune/tuning_cache.db
```

If `TESSERA_CACHE_DIR` is unset, the default is:

```text
$HOME/.tessera/autotune/tuning_cache.db
```

Public helpers:

```python
best = autotune.load(ops.matmul, (1024, 1024, 1024))
key = autotune.cache_key(ops.matmul, (1024, 1024, 1024), dtype="bf16", arch="sm90")
artifact = autotune.schedule_artifact(best, op=ops.matmul, shapes=(1024, 1024, 1024))
```

Cache keys must include:

- Operation.
- Shape.
- Dtype and numeric policy.
- Architecture.
- Layout and shard map when available.
- Movement plan for kernels whose performance depends on staging/prefetch.

---

## 6. On-Device Measurements

The runtime autotuner should support online measurement:

```python
cfg = autotune(
    ops.matmul,
    shapes=(8192, 8192, 8192),
    method="on_device",
    max_trials=50,
)
```

Required behavior for the production implementation:

- Compile candidates with fixed graph/schedule hashes.
- Warm up kernels before measuring.
- Use CUDA/HIP events or equivalent device timers.
- Record p50/p95 latency and variance.
- Capture failure diagnostics for bad candidates.
- Persist both failed and successful trials for future pruning.
- Export measurements for learned surrogate cost models.

---

## 7. Advanced Profiling

Profiler events may include counters:

```python
p.record(
    "flash_attention",
    latency_ms=1.8,
    flops=900e9,
    bytes_moved=420e6,
    counters={
        "sm_occupancy_pct": 91.0,
        "warp_divergence_pct": 2.0,
        "l2_hit_pct": 78.0,
        "shared_bank_conflict_factor": 1.03,
    },
)
```

Recommended counter groups:

- Compute: tensor core active, achieved TFLOPs, eligible warps/cycle.
- Memory: HBM throughput, L2 hit rate, shared-memory replay factor.
- Control: branch efficiency, warp divergence, barrier stalls.
- Distributed: all-reduce time, overlap percentage, bytes per collective.
- Runtime: launch latency, host wait time, queue depth.

---

## 8. Compiler Integration Contract

| Component | Responsibility |
|-----------|----------------|
| Graph IR | Preserve op identity, shape, dtype, numeric policy, and graph hash |
| Schedule IR | Expose tunable knobs, movement plans, layout choices, and cost model attributes |
| Tile IR | Verify target legality and expose resource estimates |
| Runtime Profiler | Measure latency, counters, bytes, and trace events |
| Autotuner | Search legal candidates, persist measurements, emit schedule artifacts |
| Learned Surrogate | Train on autotuner measurements and predict latency/energy/memory |

Schedule artifacts should contain:

- Shape and architecture.
- Numeric policy.
- Movement plan.
- Tile knobs.
- Measured or estimated metrics.
- Stable hash.

The command foundation is in `python/tessera/cli/prof.py` and is installed as:

- `tessera-prof`

---

## 9. Best Practices

- Profile before tuning; do not tune blind.
- Always record shape, dtype, target, graph hash, and schedule hash.
- Keep autotune search spaces legal and small enough for CI smoke tests.
- Use persistent caches in development, but invalidate them when target,
  numerics policy, movement plan, or lowering changes.
- Use deterministic seeds and fixed clocks where possible for microbenchmarks.
- Export traces for any performance regression report.
