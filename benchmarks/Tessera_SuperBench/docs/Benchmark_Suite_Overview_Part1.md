
<!-- ==== MERGE_START: Tessera_Bench_Overview ==== -->
# Tessera Benchmark Suite — Overview (Part 1/2)

This document outlines a SuperBench‑style benchmark suite tailored to the Tessera Programming Model.

## Goals
- Provide **repeatable, comparable** performance & health signals across devices/backends.
- Exercise **Tessera IR levels** (Graph→Schedule→Tile→Target) at increasing fidelity.
- Validate **numerics, determinism, and correctness** alongside performance.
- Integrate into **CI and perf gates** with regressions alarms and trend tracking.

## Scope & Tiers
- **Micro**: raw compute (FLOP/s), memory BW, cache/latency, atomics/sync.
- **Kernel**: GEMM, conv2d (NHWC), FlashAttention-style, layernorm, softmax.
- **Model Fragments**: transformer block forward/backward, small CNN cells.
- **System**: host↔device BW, NVLink/PCIe probes, disk IO, CPU perf baselines.
- **Distributed**: NCCL/RCCL/oneCCL collectives latency/bandwidth, scalability.

## Signals
- Throughput (FLOP/s, bytes/s), latency, tail latency (p95/p99), utilization.
- Efficiency vs theoretical peak, roofline buckets, scaling efficiency.
- Numerical error (max/mean abs/rel), determinism deltas, reproducibility.

## Output Artifacts
- `results.json` (machine‑readable, stable schema)
- `report.html` (at‑a‑glance cards + tables)
- Optional CSV/Parquet for historical aggregation.
