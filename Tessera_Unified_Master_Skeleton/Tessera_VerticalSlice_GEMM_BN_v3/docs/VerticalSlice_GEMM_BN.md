# Vertical Slice (Worked): GEMM + BatchNorm
This package shows a *worked* lowering flow for two core operators:
**GEMM** and **BatchNorm (inference)** from **Graph IR → Schedule IR → Tile IR → Target hooks**.

You get:
- Realistic **TableGen dialects** for Graph/Schedule/Tile
- **Rewrite patterns** that perform the lowerings with tile/vectorization decisions
- **Example MLIR modules** at each stage to see the transformation
- A placeholder **runtime/PTX** hook to illustrate final codegen boundaries

> These files are designed for study and adaptation. They are close to MLIR style but are not guaranteed to compile without integrating into a full MLIR build.


## Added in v3
- **CPU runtime implementations** for `tesseraMatmul` and `tesseraBatchNorm` in `runtime/src/api.cpp`.
- **NVVM lowering sketch** for `ttile.gemm_mma` in `compiler/mlir/passes/LowerTileToNVVM_GEMM_Sketch.cpp`.
- **Example NVVM MLIR** snippet in `compiler/mlir/examples/nvvm_mma_example.mlir`.

> These are sketches intended to guide a production lowering. They assume NVVM dialect types
> and attributes; wire them into your pass pipeline alongside the existing Schedule/Tile passes.
