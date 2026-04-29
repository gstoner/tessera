---
status: Informative
classification: Guide
authority: Developer frontend workflow for the first executable CPU compiler path
last_updated: 2026-04-28
---

# Tessera Developer Frontend: First End-To-End Path

This guide documents the first narrow executable compiler spine:

```text
@jit single-op function -> Graph IR -> Schedule IR -> Tile IR -> Target IR -> CPU execution
```

It is intentionally small. The supported program shape is a `@tessera.jit`
function that returns exactly one supported `tessera.ops.*` call. Unsupported
functions keep the existing eager Python fallback and now expose an explicit
diagnostic explaining why.

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

For a 256x128-based GEMM schedule, pass an explicit CPU tile:

```python
@ts.jit(cpu_tile=(256, 128, 64))
def gemm_256x128(A, B):
    return ts.ops.gemm(A, B)

print(gemm_256x128.schedule_ir)
```

The emitted Schedule IR records `tile_m = 256`, `tile_n = 128`, and
`tile_k = 64`. Execution is still CPU/NumPy today; the tile is an inspectable
compiler schedule artifact that will become the input to native CPU lowering.

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

To see whether a function used the compiler path:

```python
print(mm.uses_compiled_path)
print(mm.explain_lowering())
```

Functions defined in files usually work with no extra source plumbing. Inline
functions created from `stdin`, notebooks, or `exec(...)` may not be inspectable
by CPython. In those cases, pass the source explicitly so the AST frontend can
still build Graph IR:

```python
src = """
def mm(A, B):
    return ts.ops.gemm(A, B)
"""

ns = {}
exec(src, {"ts": ts}, ns)
mm = ts.jit(ns["mm"], source=src, cpu_tile=(256, 128, 64))

assert mm.uses_compiled_path
print(mm.explain_lowering())
```

Without `source=` or `source_path=`, these functions fall back eagerly and emit
`JIT_SOURCE_UNAVAILABLE` so developers can tell that source inspection, not the
op itself, blocked the compiler path.

## 3. Current Boundaries

Supported:

- `return ts.ops.matmul(A, B)`
- `return ts.ops.gemm(A, B)`
- `return ts.ops.relu(x)`
- `return ts.ops.sigmoid(x)`
- `return ts.ops.softmax(x)`
- `return ts.ops.sin(x)`
- `return ts.ops.adam(param, grad, moment1, moment2)`
- explicit matmul/GEMM CPU schedule tile through `@ts.jit(cpu_tile=(M, N, K))`
- explicit AST source for stdin/dynamic functions through `source=` or
  `source_path=`
- positional or keyword arguments
- NumPy arrays, tensor-like values with `.numpy()`, and Tessera `Tensor` values
  with `._data`

Not yet supported:

- assignment-style output buffers such as `C[:] = ts.ops.matmul(A, B)`
- fused epilogues, bias, activation, residual
- multi-op graphs such as `softmax(relu(x))`
- keyword capture for op attributes such as `softmax(axis=0)` or Adam `lr=...`
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

Fallback example:

```python
@ts.jit
def composite(x):
    y = ts.ops.relu(x)
    return ts.ops.softmax(y)

assert not composite.uses_compiled_path
print(composite.explain_lowering())
```

## 5. Next Frontend Improvements

Priority order:

1. Support output-buffer matmul: `C[:] = ts.ops.matmul(A, B)`.
2. Preserve keyword attributes in Graph IR for `softmax(axis=...)` and
   optimizer hyperparameters.
3. Improve notebook/REPL source capture beyond explicit `source=` handoff.
4. Compile short op chains and simple SSA values instead of only one returned
   op.
5. Preserve concrete shape/dtype/layout metadata in Graph IR.
6. Replace textual artifacts with MLIR objects and verifier calls.
7. Connect Target IR to the runtime C ABI CPU backend.
8. Extend the same spine to a measured CUDA matmul.
