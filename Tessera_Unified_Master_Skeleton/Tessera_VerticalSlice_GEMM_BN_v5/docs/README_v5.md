# Tessera Vertical Slice v5 (CUDA Path)

New in v5:
- **CUDA build path** with `TESSERA_ENABLE_CUDA=ON` builds:
  - `runtime/src/cuda/kernels.cu` (naive GEMM and BN kernels)
  - `runtime/src/api_cuda.cpp` (dispatchers for CUDA when `tensor.device >= 0`).
- **Dual CTest tests**:
  - `PythonGoldenTestCPU` (always)
  - `PythonGoldenTestCUDA` (only if CUDA enabled)

## Build (CPU-only)
```bash
cmake -S . -B build -DTESSERA_ENABLE_CUDA=OFF
cmake --build build -j
cd build && ctest -V
```

## Build (CUDA)
```bash
cmake -S . -B build -DTESSERA_ENABLE_CUDA=ON
cmake --build build -j
cd build && ctest -V
```

> Note: The CUDA tests currently treat tensor pointers as host buffers but set `device=0` to exercise the CUDA path. Replace with true device allocations and copies in a full runtime.
