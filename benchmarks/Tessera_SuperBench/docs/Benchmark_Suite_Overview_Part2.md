
# Tessera Benchmark Suite — Overview (Part 2/2)

## CI Integration
- GitHub Actions job that runs `--smoke` subset on PRs (fast), and a broader
  `nightly` on labeled runners.
- Thresholds from YAML (`warn_if_below`, `fail_if_below`) emit GitHub summaries.

## Extensibility
- Add a new benchmark by implementing a small Python adapter (or C++ binary)
  that returns JSON with the common schema. Register it in `configs/*.yaml`.

## Security & Stability
- Runners prohibit shell injections by whitelisting commands.
- Each task has a **timeout** and **max concurrency** to avoid node exhaustion.

## Roadmap
- Tessera IR kernels for GEMM/conv/attention with backend mappings.
- Autotune hooks to sweep tile sizes and emit Pareto fronts.
- Perfetto/NSys trace exporter integration and roofline overlay.

<!-- ==== MERGE_END: Tessera_Bench_Overview ==== -->
