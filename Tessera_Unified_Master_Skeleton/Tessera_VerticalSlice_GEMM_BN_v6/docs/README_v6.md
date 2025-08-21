# Tessera Vertical Slice v6 (True Device Allocations + Streams + MMA)

New in v6:
- **True CUDA device allocations** (`tesseraAllocDevice/tesseraFreeDevice`) and **explicit copies** (`tesseraCopyHostToDevice`/`tesseraCopyDeviceToHost`).
- **Stream API** (`tesseraStreamCreate/Destroy/Sync`) and **async ops** (`tesseraMatmulAsync`, `tesseraBatchNormAsync`).
- **Tensor Core path** via **WMMA (mma.sync)** kernel:
  - Enable with `TESSERA_MMA=1` environment variable (tests support `--mma`).
- Tests verify both **CPU** and **CUDA** execution paths with **ctest**.

## Build (CPU only)
```bash
cmake -S . -B build -DTESSERA_ENABLE_CUDA=OFF
cmake --build build -j
cd build && ctest -V
```

## Build (CUDA) and test WMMA path
```bash
cmake -S . -B build -DTESSERA_ENABLE_CUDA=ON
cmake --build build -j
cd build
# default: naive CUDA
ctest -V -R PythonGoldenTestCUDA
# prefer WMMA:
TESSERA_MMA=1 ctest -V -R PythonGoldenTestCUDA
```

> Note: WMMA kernel uses fp32→fp16 conversion for operands to demonstrate **mma.sync** usage; numerical tolerance is relaxed.
