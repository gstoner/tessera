# Tessera TPU Backend Starter (OpenXLA)

This is a bring-up scaffold to lower Tessera Target‑IR to **StableHLO** and execute on **Google TPU** via **PJRT** (`libtpu`).

## Layout
- `docs/` – design + mapping notes.
- `src/` – MLIR passes for lowering + sharding.
- `tools/` – `tessera-tpu-opt` plugin driver.
- `runtime/` – PJRT device lister stub.
- `examples/` – tiny StableHLO examples.
- `tests/` – lit + FileCheck samples.

## Build (CMake, LLVM/MLIR + StableHLO available)
```bash
cmake -S . -B build -DTESSERA_ENABLE_TPU=ON -DMLIR_DIR=<path> -DSTABLEHLO_DIR=<path>
cmake --build build -j
```

### TPU Runtime test
On a TPU VM with `libtpu`:
```bash
PJRT_DEVICE=TPU ./build/bin/pjrt_runner --list-devices
```
