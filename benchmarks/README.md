# Tessera Benchmarks

This folder contains benchmark families at different maturity levels. The active
portable path is CPU-first and uses the current Python compiler surface where it
is available.

## Current Compiler Support

| Benchmark | Status | Compiler fit |
|---|---|---|
| `benchmark_gemm.py` | Active | Can exercise `@tessera.jit` through the current CPU `matmul -> relu` lowering path with `--use-compiler` via `run_all.py`. |
| `benchmark_attention.py` | Active proxy | Roofline model; can emit current flash-attention Graph IR/lowering diagnostics with `--use-compiler`, but executable Tile/Target lowering is not wired yet. |
| `benchmark_collective.py` | Active proxy | Alpha-beta communication model; should connect to runtime collectives once C ABI/runtime hooks are ready. |
| `run_all.py` | Active | Orchestrates GEMM, attention, and collective suites; pass `--use-compiler` for current compiler artifact checks. |
| `common/` | Active contract | Shared row schema, correctness helpers, and compiler artifact hooks used by benchmark suites. |
| `Tessera_SuperBench/` | Active harness | GEMM uses the current JIT CPU path with telemetry/autotune artifacts; FlashAttention and Conv2D emit artifact-only compiler rows while running NumPy reference timing/correctness; collectives default to the Tessera mock facade. |
| `spectral/` | Active benchmark | NumPy/PyTorch FFT/DCT/convolution benchmark with `--backend tessera-artifact` for Graph IR artifact rows. |
| `Tessera_Operator_Benchmarks/` | Active C++ harness | CPU reference timing for all registered ops, Graph IR artifact coverage for all registered ops, `tessera.telemetry.v1` JSON summaries, and explicit `backend_unavailable` Tessera-runtime status. |
| `archive/matrix_multiplication/` | Archived | Blackwell concept sketch using non-existent APIs; future Blackwell work should land as Target IR tests/runtime kernels/operator cases. |
| `DeepScholar-Bench/` | Research/speculative | Uses non-existent model APIs. Keep as research benchmark concept or move under `benchmarks/archive/` after review. |

## Quick Checks

```bash
PYTHONPATH=python python3 benchmarks/run_all.py --json-only --no-save --use-compiler --smoke
PYTHONPATH=python python3 benchmarks/Tessera_SuperBench/benches/kernel/gemm_tessera.py --m=64 --n=64 --k=64 --repeat=1
PYTHONPATH=python python3 benchmarks/Tessera_SuperBench/runner/bench_run.py --config benchmarks/Tessera_SuperBench/configs/compiler_smoke.yaml --out /tmp/tessera_superbench_smoke
cmake -S benchmarks/Tessera_Operator_Benchmarks -B /tmp/tessera_opbench_build && cmake --build /tmp/tessera_opbench_build -j
PYTHONPATH=python python3 benchmarks/Tessera_Operator_Benchmarks/scripts/opbench.py --config benchmarks/Tessera_Operator_Benchmarks/scripts/configs/quick_sweep.yaml --bin /tmp/tessera_opbench_build/opbench --out /tmp/tessera_opbench_quick
PYTHONPATH=python python3 benchmarks/spectral/spectral_bench.py --backend tessera-artifact --ops fft1d,dct2,conv1d_fft --sizes 64 --device cpu --repeats 1 --warmup 0 --outcsv /tmp/tessera_spectral.csv
```

## Refactor Direction

- Promote compiler-backed benchmark kernels into small Python modules that expose
  Graph/Schedule/Tile/Target artifacts alongside timing rows.
- Keep analytical roofline/proxy benchmarks, but label them explicitly.
- Move purely speculative benchmark concepts to `benchmarks/archive/` once they
  are no longer feeding active compiler/runtime work.
