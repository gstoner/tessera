
# ResolveTesseraAttrsPass

A tiny pass that rewrites symbolic placeholders in the transform dialect into concrete constants so the **GEMM meta-schedule** runs directly.

## Build
```bash
cmake -S . -B build -G Ninja \
  -DMLIR_DIR=$PWD/llvm-project/build/lib/cmake/mlir
cmake --build build -j
```

## Use
Inject concrete values either as **module attributes** or via CLI:

### 1) Module attributes
```
module attributes {tessera.BLOCK_M = 128 : i64, tessera.BLOCK_N = 128 : i64,
                   tessera.BLOCK_K = 64 : i64, tessera.num_stages = 2 : i64} {
  // ...
}
```

### 2) CLI key/values
```
build/bin/tessera-mlir-opt input.mlir \
  -tessera-resolve-attrs \
  --tessera-resolve=BLOCK_M=128 \
  --tessera-resolve=BLOCK_N=128 \
  --tessera-resolve=BLOCK_K=64  \
  --tessera-resolve=num_stages=2 \
  -pass-pipeline='builtin.module(transform-interpreter{"script-file=tessera/tools/autotune/mlir/gemm_meta_schedule.mlir"})'
```

The pass looks for:
- `transform.structured.tile` with `tile_sizes_sym` and sets `tile_sizes`.
- `transform.structured.pipeline` with `stages_sym` and sets `stages`.

Unresolved symbols are left intact (the interpreter will then error, pointing to the missing value).

## Transform script
See `tessera/tools/autotune/mlir/gemm_meta_schedule.mlir` for a version that uses resolvable placeholders.
