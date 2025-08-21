# Tessera Vertical Slice v7 (SM90 WGMMA + cp.async Tiling)

New in v7:
- **SM90 WGMMA demo kernel** `gemm_wgmma_sm90.cu` (guarded by `__CUDA_ARCH__ >= 900`):
  - Illustrates a double-buffered pipeline with **cp.async**-style staged copies into shared memory
  - Shows the hook point where real **wgmma.mma_async** PTX should be used (the sample uses FMAs for clarity)
- **Runtime selection** between **naive**, **WMMA (mma.sync)**, and **WGMMA** via env vars:
  - `TESSERA_NAIVE=1` → naive
  - `TESSERA_MMA=1` → WMMA
  - `TESSERA_WGMMA=1` → WGMMA (SM90+)
- **True device allocations & copies** and **streams/async** kept from v6
- **CTest** gains a WGMMA test in addition to CPU / CUDA / WMMA

## Build (CUDA)
```bash
cmake -S . -B build -DTESSERA_ENABLE_CUDA=ON
cmake --build build -j
cd build
ctest -V -R PythonGoldenTestCUDA_Naive
ctest -V -R PythonGoldenTestCUDA_WMMA
TESSERA_WGMMA=1 ctest -V -R PythonGoldenTestCUDA_WGMMA
```

> This is a **teaching implementation**. To turn the WGMMA path into a high-performance kernel, replace the FMAs with inline PTX `wgmma.mma_async` calls and add proper smem layouts, `cp.async` groups, `commit_group`, and `wait_group` sequencing.
