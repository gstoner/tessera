# Tessera NVIDIA GPU Backend (Hopper / Blackwell / NVVM / PTX)

This backend has two build modes:

- **Hardware-free compiler artifacts** with `TESSERA_BUILD_NVIDIA_BACKEND=ON` and `TESSERA_ENABLE_CUDA=OFF`. This builds `tessera-opt`, `tessera-nvidia-opt`, the `tessera_nvidia` Target IR contract dialect, and Hopper/Blackwell lowering pipelines.
- **Optional CUDA runtime validation** with `TESSERA_ENABLE_CUDA=ON`. This additionally builds CUDA kernels, NVRTC helpers, runtime launch tests, and benchmarks when a CUDA Toolkit is available.

The Python compiler frontend now routes `target="nvidia_sm90"`,
`target="nvidia_sm100"`, and related CUDA aliases through the verified
`TargetIRModule` object model before emitting inspectable `tessera_nvidia.*`
MLIR-like artifacts.

Primary profiles:
- Hopper: `SM_90` / `sm_90a` with WGMMA, TMA, and mbarrier contracts.
- Blackwell: `SM_100` / `sm_100a` and `SM_120` with TCGEN05 and TMEM contracts.

## LLVM/MLIR 21 Artifact Build

```bash
cmake -S . -B build-nvidia \
  -DCMAKE_PREFIX_PATH=/opt/homebrew/opt/llvm@21 \
  -DLLVM_DIR=/opt/homebrew/opt/llvm@21/lib/cmake/llvm \
  -DMLIR_DIR=/opt/homebrew/opt/llvm@21/lib/cmake/mlir \
  -DTESSERA_BUILD_NVIDIA_BACKEND=ON \
  -DTESSERA_ENABLE_CUDA=OFF \
  -DTESSERA_BUILD_EXAMPLES=OFF

cmake --build build-nvidia --target tessera-opt tessera-nvidia-opt
```

Available hardware-free pipelines:
- `tessera-lower-to-nvidia`
- `tessera-lower-to-hopper`
- `tessera-lower-to-blackwell`
- `lower-tile-to-nvidia`
- `lower-tessera-nvidia-to-nvvm`

## Optional CUDA Runtime Build

```bash
cmake -S . -B build-nvidia-cuda \
  -DTESSERA_BUILD_NVIDIA_BACKEND=ON \
  -DTESSERA_ENABLE_CUDA=ON \
  -DTESSERA_CUDA_ARCH=sm_90a

cmake --build build-nvidia-cuda
```

Runtime checks require a CUDA Toolkit and suitable NVIDIA hardware. They are not required for the compiler artifact gate.

## Components

- `include/tessera/gpu/target.h`: Backend interface + C APIs
- `include/tessera/gpu/IR/TesseraNVIDIADialect.td`: NVIDIA Target IR contract dialect
- `lib/Conversion/NVIDIALowering.cpp`: Tile → NVIDIA Target IR → NVVM artifact lowering
- `src/runtime/cuda_driver.{h,cpp}`: Thin CUDA Driver API wrapper
- `src/runtime/nvrtc_jit.cpp`: Optional NVRTC PTX JIT helper (guarded by `TESSERA_USE_NVRTC`)
- `src/kernels/wmma_gemm_fp16.cu`: FP16 WMMA kernel
- `src/kernels/wmma_gemm_bf16.cu`: BF16 WMMA kernel (sm80+)
- `test/nvidia/`: FileCheck contracts for Hopper and Blackwell artifact lowering

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
