
# Tessera Benchmark Suite — Overview (Part 2/2)

## CI Integration
- GitHub Actions job that runs `--smoke` subset on PRs (fast), and a broader
  `nightly` on labeled runners.
- Thresholds from YAML (`warn_if_below`, `fail_if_below`) emit GitHub summaries.

## Extensibility
- Add a new benchmark by implementing a small Python adapter (or C++ binary)
  that returns JSON with the common schema. Register it in `configs/*.yaml`.
- Prefer the shared `benchmarks.common.BenchmarkRow` and
  `telemetry_for_row(...)` helpers for compiler-backed Python benchmarks.
- Separate executable runtime measurements from artifact-only compiler rows.
  Artifact-only rows should use `runtime_status="artifact_only"` and state the
  reference timing path in `reason`.

## Security & Stability
- Runners prohibit shell injections by whitelisting commands.
- Each task has a **timeout** and **max concurrency** to avoid node exhaustion.

## Roadmap
- Promote Conv2D and FlashAttention from artifact-only to executable runtime
  rows when the compiler/runtime path is ready.
- Expand autotune beyond GEMM and emit Pareto-front artifacts.
- Add native device timer backends and vendor traces while preserving the
  `tessera.telemetry.v1` row contract.

<!-- ==== MERGE_END: Tessera_Bench_Overview ==== -->
