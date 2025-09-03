\
# Tessera Metalium Backend (Starter)

This is a *minimal starter* for a Tessera→TT-Metalium backend that includes:

- **Dialect draft** (`TesseraMetaliumOps.td`) for `dma`, `load2d`, `store2d`, `matmul`
- **Dialect C++ stubs** (`TesseraMetaliumDialect.h/.cpp`) for registration
- **Lowering sketch** (`Lowering/TileToMetalium.cpp`) from Tessera Tile IR
- **Tiny codegen shim** (`Codegen/MetaliumCodegen.*`) that turns a module into a mock Program/Queue
- **Demo tool** (`tools/metalium-codegen-demo/`) printing JSON and queue events

> This package is intentionally header-only stubs and *does not* require MLIR
> to compile the demo. Integrate with your existing MLIR build to generate the
> op classes via `mlir-tblgen` and wire the real pass pipeline.

## Next steps

1. **Run tblgen** over `TesseraMetaliumOps.td` to generate op classes:
   ```cmake
   mlir_tablegen(TesseraMetaliumOps.h.inc -gen-op-decls ...)
   mlir_tablegen(TesseraMetaliumOps.cpp.inc -gen-op-defs ...)
   ```
2. **Include** the generated headers in `TesseraMetaliumDialect.cpp` and
   call `addOperations<...>` in `initialize()` to register ops.
3. **Implement** the real patterns in `TileToMetalium.cpp` mapping tile shapes,
   strides, and element types to Metalium DMA/Matmul.
4. **Replace** the shim with real TT-Metalium API calls (program build + queue submit).

## Layout

```
include/Tessera/Target/Metalium/
  ├─ TesseraMetaliumDialect.h
  ├─ TesseraMetaliumOps.td
  └─ MetaliumCodegen.h
lib/Target/Metalium/
  ├─ TesseraMetaliumDialect.cpp
  ├─ Lowering/TileToMetalium.cpp
  └─ Codegen/MetaliumCodegen.cpp
tools/metalium-codegen-demo/
  └─ main.cpp
CMakeLists.txt
```


## Build the pass plugin & run tests

```bash
# Configure with MLIR and lit enabled
cmake -B build -S . \
  -DTESSERA_METALIUM_WITH_MLIR=ON \
  -DTESSERA_METALIUM_BUILD_DEMO=ON \
  -DTESSERA_METALIUM_ENABLE_TESTS=ON

cmake --build build -j
```

### Use with mlir-opt
```bash
# Run the lowering via mlir-opt with the pass plugin
mlir-opt test/metalium/tile_to_dma.mlir \
  -load-pass-plugin build/libtessera_metalium_passes.so \
  -pass-pipeline="tessera-metalium" | FileCheck test/metalium/tile_to_dma.mlir
```

### Run lit tests
```bash
# If not found automatically, set LLVM_EXTERNAL_LIT to your llvm-lit path
# export LLVM_EXTERNAL_LIT=/path/to/llvm-lit
ctest --test-dir build -VV -R metalium-tests
# Or directly:
llvm-lit -sv test/metalium
```


### Optional: build the `tessera-metalium-opt` wrapper
```bash
cmake -B build -S . \
  -DTESSERA_METALIUM_WITH_MLIR=ON \
  -DTESSERA_METALIUM_BUILD_OPT=ON

cmake --build build -j
```

Then you can replace `mlir-opt -load-pass-plugin ...` with:
```bash
build/tessera-metalium-opt -pass-pipeline="tessera-metalium" test/metalium/tile_to_dma.mlir | FileCheck test/metalium/tile_to_dma.mlir
```
