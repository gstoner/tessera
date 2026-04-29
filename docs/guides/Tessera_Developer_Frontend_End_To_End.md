---
status: Informative
classification: Guide
authority: Developer frontend workflow for the first executable CPU compiler path
last_updated: 2026-04-28
---

# Tessera Developer Frontend: First End-To-End Path

This guide documents the first narrow executable compiler spine:

```text
@jit matmul -> Graph IR -> Schedule IR -> Tile IR -> Target IR -> CPU execution
```

It is intentionally small. The supported program shape is a `@tessera.jit`
function that returns exactly one `tessera.ops.matmul` or `tessera.ops.gemm`.
Unsupported functions keep the existing eager Python fallback.

## 1. Minimal Example

```python
import numpy as np
import tessera as ts

@ts.jit
def mm(A, B):
    return ts.ops.matmul(A, B)

A = np.arange(6, dtype=np.float32).reshape(2, 3)
B = np.arange(12, dtype=np.float32).reshape(3, 4)

Y = mm(A, B)
```

`Y` is computed by the CPU matmul lowering plan attached to the `JitFn`, not by
the original Python function body.

## 2. Inspecting Compiler Layers

Every supported matmul function exposes four artifacts:

```python
for artifact in mm.lowering_artifacts():
    print("==", artifact.level)
    print(artifact.text)
```

Convenience properties are also available:

```python
print(mm.graph_ir.to_mlir())
print(mm.schedule_ir)
print(mm.tile_ir)
print(mm.target_ir)
```

The artifacts are textual today:

- **Graph IR:** emitted by `GraphIRBuilder`.
- **Schedule IR:** fixed tile/layout plan for CPU matmul.
- **Tile IR:** CPU loop-nest style tile matmul.
- **Target IR:** CPU target artifact using the NumPy ABI for execution.

## 3. Current Boundaries

Supported:

- `return ts.ops.matmul(A, B)`
- `return ts.ops.gemm(A, B)`
- positional or keyword arguments
- NumPy arrays, tensor-like values with `.numpy()`, and Tessera `Tensor` values
  with `._data`

Not yet supported:

- assignment-style output buffers such as `C[:] = ts.ops.matmul(A, B)`
- fused epilogues, bias, activation, residual
- dynamic Schedule IR selection
- native C ABI CPU launch
- GPU/ROCm/TPU dispatch from this frontend path
- arbitrary Python control flow lowering

## 4. Developer Evaluation Criteria

When evaluating frontend ergonomics, use this path as the baseline:

- Can a new developer write a minimal `@jit` function without extra ceremony?
- Are lowering artifacts discoverable from the returned `JitFn`?
- Do error messages point at the unsupported construct?
- Does the CPU result match NumPy for common matmul shapes?
- Is the fallback behavior obvious when the compiler path is not used?

## 5. Next Frontend Improvements

Priority order:

1. Add explicit diagnostics when a function misses the narrow CPU path.
2. Support output-buffer matmul: `C[:] = ts.ops.matmul(A, B)`.
3. Preserve concrete shape/dtype/layout metadata in Graph IR.
4. Replace textual artifacts with MLIR objects and verifier calls.
5. Connect Target IR to the runtime C ABI CPU backend.
6. Extend the same spine to a measured CUDA matmul.
