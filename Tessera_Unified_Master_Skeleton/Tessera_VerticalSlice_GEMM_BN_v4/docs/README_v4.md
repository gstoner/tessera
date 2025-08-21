# Tessera Vertical Slice v4

New in this version:
- **CMake + CTest** integration builds `libtessera_runtime.so` (CPU) and runs a **Python golden test** via `ctest`.
- The Python test (`tests/test_runtime.py`) uses **ctypes** to call `tesseraMatmul` and `tesseraBatchNorm`, and verifies against NumPy.
- Added **NVVM WGMMA** example snippet (`compiler/mlir/examples/nvvm_wgmma_example.mlir`).

## Build & Test
```bash
cmake -S . -B build
cmake --build build -j
cd build && ctest -V
```
