# Tessera GPU Backend (NVVM / Tile IR / PTX)

This is a **reference GPU backend** for Tessera targeting NVIDIA GPUs via **NVVM** (LLVM NVPTX), the MLIR **NVVM/NVGPU** dialects, and **PTX**. It contains:

- A backend interface and CUDA Driver runtime
- WMMA Tensor Core GEMM kernels (FP16→FP32, BF16→FP32)
- A lowering skeleton showing how Tessera Tile IR ops map to NVGPU/NVVM/LLVM NVPTX
- A test that runs the WMMA kernels

> **Architectures:** default `-arch=sm_90`. Override with `-DTESSERA_CUDA_ARCH=sm_80|sm_86|sm_90a|sm_102`.

## Build

```bash
mkdir build && cd build
cmake -DTESSERA_CUDA_ARCH=sm_90 -DTESSERA_BUILD_TESTS=ON ..
cmake --build . -j
./test_wmma
```

## Components

- `include/tessera/gpu/target.h`: Backend interface + C APIs
- `src/runtime/cuda_driver.{h,cpp}`: Thin CUDA Driver API wrapper
- `src/runtime/nvrtc_jit.cpp`: Optional NVRTC PTX JIT helper (guarded by `TESSERA_USE_NVRTC`)
- `src/kernels/wmma_gemm_fp16.cu`: FP16 WMMA kernel
- `src/kernels/wmma_gemm_bf16.cu`: BF16 WMMA kernel (sm80+)
- `src/lowering_nvvm_mlir.cpp`: lowering skeleton (no MLIR dep; comments and pseudo-code)
- `docs/`: IR mapping tables and backend overview



## NVIDIA Tile IR (experimental)

Enable with CMake option `-DTESSERA_ENABLE_NVTILE=ON` (default ON). This pulls in:
- `src/lowering_nvtile_mlir.cpp` (design notes / mapping scaffold)
- `src/kernels/ptx_wgmma_bf16.cu` (guarded placeholder kernel for WGMMA BF16)
- `tests/test_wgmma_tile_ir.cu` (sanity runner; does nothing on pre-Hopper)

> Integrators should replace the placeholder kernel with real **inline PTX WGMMA** (e.g., `wgmma.mma_async.*.bf16.bf16.f32`) and
> wire **TMA** descriptors & **mbarrier** sync to stage tiles.


## GPU Micro-Bench & HTML Report

Build benches:
```bash
mkdir -p build && cd build
cmake -DTESSERA_CUDA_ARCH=sm_90 -DTESSERA_BUILD_GPU_BENCH=ON ..
cmake --build . -j
```

Run & report:
```bash
./bench_gpu > ../bench/reports/gpu_results.csv
python3 ../scripts/gpu_report.py --csv ../bench/reports/gpu_results.csv --out ../bench/reports/gpu_report.html
# or: ../scripts/run_gpu_bench_and_report.sh ../bench/reports
```

CSV columns:
```
kernel,dtype,path,M,N,K,ms,tflops
```
