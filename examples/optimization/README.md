
# Tessera `src/` Code Optimization Examples

These examples demonstrate practical, measurable optimization techniques you can adapt in your `src/` tree.
Each example has a **baseline** and an **optimized** variant, with comments explaining why it’s faster.

**Contents**
- `01_loop_tiling_blocking.cpp` — Cache-blocked GEMM (baseline vs blocked + vectorized loads).
- `02_vectorization_intrinsics.cpp` — SIMD via x86 AVX2/AVX-512 intrinsics (fallback to scalar).
- `03_cache_friendly_layout.cpp` — SoA vs AoS for better cache/TLB behavior.
- `04_parallel_reduction_shared_mem.cu` — Warp-synchronous tree reduction with shared memory.
- `05_async_copy_tma_wgmma.cu` — (SM90) skeleton for `cp.async.bulk` + WGMMA epilogue (guarded by `__CUDA_ARCH__>=900`).
- `06_branchless_kernels.cu` — Replace branches with predication and math tricks.
- `07_software_prefetch.cpp` — Software prefetching & double-buffering for streaming transforms.
- `08_mlir_gemm_tiled.mlir` — MLIR example: tile + vectorize + bufferize for GEMM-like loop nest.
- `CMakeLists.txt` — Minimal build system for CPU and CUDA examples.

## Build

```bash
# CPU (GCC/Clang with -O3 -march=native recommended)
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j

# CUDA examples (need NVCC; SM target optional)
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DTESSERA_WITH_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=90
cmake --build build -j
```

## Run (examples print quick checksums/throughputs)

```bash
./build/01_loop_tiling_blocking
./build/02_vectorization_intrinsics
./build/03_cache_friendly_layout
# CUDA:
./build/04_parallel_reduction_shared_mem
./build/06_branchless_kernels
# 05_async_copy_tma_wgmma only compiles on SM90+
```

## Profiling Tips
- **CPU:** use `perf stat`, `perf record`, `likwid-perfctr`, or VTune; check IPC, LLC load misses, L1/L2 miss, cycles stalled.
- **CUDA:** use Nsight Systems (concurrency) + Nsight Compute (memory, tensor core utilization). Add NVTX around kernels/ranges.
- **Flags:** `-O3 -march=native -ffast-math` (careful with math), `-fopenmp` for simple parallel loops.
- **Data:** always size to exceed LLC to expose memory behavior; warm up once to amortize JIT and page faults.
