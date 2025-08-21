# Tessera Unified Master Source Base (Skeleton)

This repository layout lets you:
- Build the **shared runtime** once (`tessera_runtime`), with CPU-only or CPU+CUDA backends.
- Plug in **vertical slices** (e.g., GEMM+BatchNorm) under `tessera/verticals/*` and test them against the shared runtime.
- Toggle features with standard CMake options.

## Directory layout
```
tessera/
  runtime/                # shared runtime (API, CPU/CUDA backends)
    include/tessera/runtime/api.h
    src/api_cpu.cpp
    src/api_cuda.cpp
    src/cuda/*.cu        # naive, WMMA, WGMMA illustrative kernels
  verticals/
    gemm_bn/             # vertical slice reusing the runtime
      CMakeLists.txt
      tests/test_runtime_vs.py
cmake/                   # (placeholder for toolchain/config modules)
tests/                   # top-level tests (bootstrap)
docs/
```

## Build
```bash
cmake -S . -B build -DTESSERA_ENABLE_CUDA=ON -DTESSERA_ENABLE_VERTICAL_GEMM_BN=ON
cmake --build build -j
cd build && ctest -V
```

### Selecting kernel paths at runtime
Set one of:
- `TESSERA_NAIVE=1` → naive CUDA
- `TESSERA_MMA=1` → WMMA (mma.sync)
- `TESSERA_WGMMA=1` → WGMMA (SM90+)
