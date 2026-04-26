
# Tessera Rubin CPX Compiler Support (v0)

This drop includes:
- ODS for CPX Target‑IR (types, attrs, ops).
- Pass skeletons: partition, KV transport lowering, NVFP4 vectorize, video ingest fuse.
- `tessera-cpx-opt` driver, CMake + lit scaffolding.
- FileCheck tests.

## Build
```bash
cmake -S . -B build -G Ninja -DMLIR_DIR=<path-to-mlir-cmake>
cmake --build build
```

## Run tests
```bash
cd build && ctest --output-on-failure
```

## Pipelines (examples)
```bash
tessera-cpx-opt input.mlir -tessera-partition-longcontext -tessera-lower-kv-transport
tessera-cpx-opt input.mlir -tessera-vectorize-nvfp4
tessera-cpx-opt input.mlir -tessera-fuse-video-ingest
```

## Merging docs
Docs include MERGE markers as requested to re‑assemble multi‑part docs.
