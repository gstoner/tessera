
# Tessera Benchmark Suite (SuperBench-style) — current compiler

This suite is a portable SuperBench-style harness for the current Tessera
compiler surface:

- **Executable compiler path:** GEMM+ReLU uses `@tessera.jit` through
  Graph IR -> Schedule IR -> Tile IR -> Target IR -> CPU.
- **Artifact-only compiler paths:** Conv2D and FlashAttention capture compiler
  artifacts and run NumPy timing/correctness references until native runtimes
  are promoted.
- **Shared telemetry:** each compiler-backed row emits `tessera.telemetry.v1`
  with compiler path, runtime status, artifact hashes, latency, and bottleneck
  labels.
- **Autotune artifacts:** GEMM can attach the current `tessera.autotune`
  schedule artifact through `--autotune`.
- **Portable collectives:** default configs use the `tessera.collectives` mock
  facade; CUDA/NCCL remains an opt-in hardware config.
- **System probes and reports:** GPU/NIC probes skip safely, Chrome Trace JSON
  is emitted by the runner, and `report_html.py` renders a roofline plus
  telemetry summary.

## Quick Start
```bash
# 1) Build C++ microbenches
cmake -S benches -B build && cmake --build build -j

# 2) Run the portable compiler smoke
~/venv/bin/python runner/bench_run.py --config configs/compiler_smoke.yaml --out out/compiler_smoke

# 3) Run the broader portable suite
~/venv/bin/python runner/bench_run.py --config configs/default.yaml --out out/default

# 4) Generate HTML with roofline and telemetry summary
~/venv/bin/python runner/report_html.py --results out/default/results.json --html out/default/report.html --peaks peaks/example_peaks.yaml

# Portable collective smoke via Tessera facade
~/venv/bin/python benches/distributed/collectives_torch.py --world_size 2 --backend tessera_mock --iters 10 --bytes 1048576

# Optional CUDA/NCCL hardware run
~/venv/bin/python benches/distributed/collectives_torch.py --world_size 2 --backend nccl --iters 100 --bytes 134217728

# View trace
# Open out/trace.json in https://ui.perfetto.dev or Chrome tracing.
```

## Status Labels

| Status | Meaning |
|---|---|
| `executable` | The benchmark measured a currently executable Tessera path. |
| `artifact_only` | Compiler artifacts were captured, but timing uses a reference implementation. |
| `mock` | Runtime-facing facade path is active without native hardware. |
| `backend_unavailable` | Requested native backend is not available in this environment. |

`configs/sm90_cuda.yaml` is a hardware-oriented overlay. It preserves NCCL and
large-shape attention/GEMM settings but should not be used as a CPU-only CI
gate.
