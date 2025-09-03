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


## New in v2
- `tessera-lower-attention-to-stablehlo` → lowers `tessera.flash_attn` to `stablehlo.custom_call`.
- `tessera-lower-conv-to-stablehlo` → lowers `tessera.conv2d` and `tessera.conv2d_bias_gelu` to NHWC StableHLO (+epilogue).
- `tessera-export-shardy` → emits `sdy.mesh` / `sdy.tensor_sharding` attrs.
- `pjrt_runner` can be built with PJRT C API support (compile path wired; buffer execute stub indicated).
