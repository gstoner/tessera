# Vertical Slice (Worked): GEMM + BatchNorm
This package shows a *worked* lowering flow for two core operators:
**GEMM** and **BatchNorm (inference)** from **Graph IR → Schedule IR → Tile IR → Target hooks**.

You get:
- Realistic **TableGen dialects** for Graph/Schedule/Tile
- **Rewrite patterns** that perform the lowerings with tile/vectorization decisions
- **Example MLIR modules** at each stage to see the transformation
- A placeholder **runtime/PTX** hook to illustrate final codegen boundaries

> These files are designed for study and adaptation. They are close to MLIR style but are not guaranteed to compile without integrating into a full MLIR build.
