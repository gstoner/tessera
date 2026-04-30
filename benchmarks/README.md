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
| `Tessera_SuperBench/` | Active harness | Useful suite runner; GEMM Python kernel now uses the current JIT CPU path. Other Python kernel stubs remain NumPy baselines. |
| `spectral/` | Active non-compiler benchmark | NumPy/PyTorch FFT/DCT/convolution benchmark. Relevant, but Tessera spectral compiler lowering is not connected here yet. |
| `Tessera_Operator_Benchmarks/` | C++ harness scaffold | Useful operator microbenchmark harness. Tessera runtime hooks are still TODO behind `OPBENCH_WITH_TESSERA`. |
| `matrix_multiplication/` | Needs refactor/archive | Blackwell concept sketch uses non-existent high-level APIs and should become a Target IR/architecture note or be archived. |
| `DeepScholar-Bench/` | Research/speculative | Uses non-existent model APIs. Keep as research benchmark concept or move under `benchmarks/archive/` after review. |

## Quick Checks

```bash
PYTHONPATH=python python3 benchmarks/run_all.py --json-only --no-save --use-compiler --smoke
PYTHONPATH=python python3 benchmarks/Tessera_SuperBench/benches/kernel/gemm_tessera_stub.py --m=64 --n=64 --k=64 --repeat=1
```

## Refactor Direction

- Promote compiler-backed benchmark kernels into small Python modules that expose
  Graph/Schedule/Tile/Target artifacts alongside timing rows.
- Keep analytical roofline/proxy benchmarks, but label them explicitly.
- Move purely speculative benchmark concepts to `benchmarks/archive/` once they
  are no longer feeding active compiler/runtime work.
