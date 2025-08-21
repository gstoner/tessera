Tessera_Unified_Master_Skeleton

- Top-level CMake with options:
	- TESSERA_ENABLE_CUDA=ON|OFF
	- TESSERA_ENABLE_VERTICAL_GEMM_BN=ON|OFF
- Shared runtime under tessera/runtime/ (CPU + CUDA backends, streams/async, naïve + WMMA + WGMMA demo kernels).
- Vertical slice gemm_bn integrated under tessera/verticals/gemm_bn/ with its own tests wired into CTest.
- Env-based kernel selection at runtime:
	- TESSERA_NAIVE=1 → naïve CUDA
	- TESSERA_MMA=1 → WMMA (mma.sync)
	- TESSERA_WGMMA=1 → WGMMA (SM90+)
- Docs: docs/README_UNIFIED.md explains layout & build.

Build & test
```bash
cmake -S . -B build -DTESSERA_ENABLE_CUDA=ON -DTESSERA_ENABLE_VERTICAL_GEMM_BN=ON
cmake --build build -j
cd build && ctest -V
```