---
status: Informative
classification: Informative
authority: TesseraBench documentation index
last_updated: 2026-04-30
---

# TesseraBench Documentation

TesseraBench is the official benchmarking and performance validation framework
for the Tessera programming model. It is inspired by tritonbench and provides
systematic performance evaluation across the full multi-level IR stack — from
Graph IR to Target IR — on NVIDIA, AMD, x86, and TPU backends.

All examples use the current Tessera API: `@tessera.jit`, `@tessera.kernel`,
`tessera.ops.*`, `tessera.domain.Rect`, `tessera.dist.Block/Replicated`, and
`tessera.array.from_domain`. See [`docs/CANONICAL_API.md`](../CANONICAL_API.md)
for the authoritative name list.

The Python benchmark runner suite lives in `benchmarks/` at the repo root
(`benchmark_gemm.py`, `benchmark_attention.py`, `benchmark_collective.py`,
`run_all.py`) and is part of the Phase 6 deliverables.

---

## Document Map

| Document | Topic |
|----------|-------|
| [`tesserabench_doc1.md`](tesserabench_doc1.md) | **Architecture and Design** — TesseraBench core design, `TesseraBenchCore` class, integration with the Tessera compiler pipeline |
| [`tesserabench_doc2.md`](tesserabench_doc2.md) | **Benchmark Suite Implementation** — GEMM, FlashAttention, LayerNorm, Conv2D benchmark implementations using `@tessera.jit` / `tessera.ops.*` |
| [`tesserabench_doc3.md`](tesserabench_doc3.md) | **Command Line Interface and Automation** — CLI entry points, CI/CD integration, GitHub Actions and Jenkins examples, cache management |
| [`tesserabench_doc4.md`](tesserabench_doc4.md) | **Reporting and Visualization System** — HTML/JSON report generation, interactive dashboards, statistical analysis, performance trending |
| [`tesserabench_doc5.md`](tesserabench_doc5.md) | **Distributed Benchmarking** — multi-rank MockRankGroup harness, data-parallel and tensor-parallel GEMM sweeps, collective throughput (AllReduce/ReduceScatter/AllGather), MoE Cyclic distribution, pipeline-parallel analytical model |
| [`tesserabench_doc6.md`](tesserabench_doc6.md) | **Tessera Integration and Advanced Features** — deep integration with Tessera IR inspection, NVL72 distributed benchmarking, compiler instrumentation |
| [`tesserabench_doc7.md`](tesserabench_doc7.md) | **Production Deployment and CI/CD Integration** — production stack, automated regression detection, Dockerfile, enterprise deployment |
| [`tesserabench_doc8.md`](tesserabench_doc8.md) | **Enterprise Features and Future Roadmap** — executive dashboards, multi-cloud deployment, AI-driven optimization, hardware roadmap |

---

## Quick Start

```python
from benchmarks.benchmark_gemm import run_gemm_benchmark

results = run_gemm_benchmark(
    backends=["x86"],                  # "x86", "nvidia", "rocm"
    sizes=[(512, 512, 512),
           (1024, 1024, 1024),
           (4096, 4096, 4096)],
)

for r in results:
    print(f"M={r.M} N={r.N} K={r.K}  {r.tflops:.1f} TFLOPS  {r.latency_ms:.2f} ms")
```

Run all benchmarks and produce the JSON report:

```bash
python benchmarks/run_all.py --backends x86 --output tessera_benchmarks.json
```

JSON schema fields: `backend`, `op`, `shape`, `dtype`, `latency_ms`, `tflops`,
`memory_bw_gb_s`, `device`, `tessera_version`.

---

## Related Docs

- [`docs/guides/Tessera_Profiling_And_Autotuning_Guide.md`](../guides/Tessera_Profiling_And_Autotuning_Guide.md) — roofline analysis, autotuner, persistent SQLite cache
- [`docs/guides/Tessera_QA_Reliability_Guide.md`](../guides/Tessera_QA_Reliability_Guide.md) — performance consistency and regression validation
- [`docs/tutorials/performance_tuning.md`](../tutorials/performance_tuning.md) — GPU performance tuning concepts mapped to Tessera abstractions
