# Tessera Differentiable Entropy Regularization (v0.1)

<!-- MERGE_MARKER_START -->
This package adds a **differentiable range‑partition entropy** operator and an **attention‑row entropy regularizer** to the Tessera Programming Model.
It follows the estimator introduced in *Differentiable Entropy Regularization for Geometry and Neural Networks* (arXiv:2509.03733) and exposes it as MLIR ops with autodiff support and a CPU reference kernel.

## What’s included
- `tessera.diffentropy.range_entropy_soft` — entropy of a soft range‑induced partition for point sets (anchors or soft halfspaces).
- `tessera.diffentropy.attn_row_entropy` — row‑wise entropy penalty over attention weights.
- Autodiff (JVP + VJP) hooks and gradients.
- Target‑IR lowering stubs + CPU reference kernels.
- `lit` tests + FileCheck.
- CMake targets to build and register a `tessera-dentropy-opt` tool.

See the paper for theory and estimator definitions. Use cases: geometry preprocessors (“EntropyNet”) and Transformer attention regularization. 
<!-- MERGE_MARKER_END -->

## Build (out‑of‑tree plugin style)
```bash
cmake -S . -B build -DTESSERA_DIR=$HOME/tessera -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
# Register into mlir-opt style driver
build/bin/tessera-dentropy-opt -h
```

## Quick test
```bash
build/bin/tessera-dentropy-opt   -tessera-canonicalize -tessera-verify   -pass-pipeline='builtin.module(func.func(test-dentropy-range, test-dentropy-attn))'   test/DiffEntropy/dentropy_basic.mlir | FileCheck test/DiffEntropy/dentropy_basic.mlir
```
