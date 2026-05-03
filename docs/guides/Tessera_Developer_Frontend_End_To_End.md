---
status: Informative
classification: Guide
authority: Developer frontend workflow for the first executable CPU compiler path
last_updated: 2026-04-28
---

# Tessera Developer Frontend: First End-To-End Path

This guide documents the first reference executable compiler spine:

```text
@jit supported op graph -> Graph IR -> Schedule IR -> Tile IR -> Target IR -> CPU execution
```

It is intentionally a development/reference path. The supported program shape
is straight-line dataflow through canonical `tessera.ops.*` calls. CPU execution
uses NumPy and explicit single-rank/state stubs; it is not native GPU,
distributed, or C ABI execution.
Functions outside this supported shape keep the eager Python fallback and expose
an explicit diagnostic explaining why.

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

Every supported function exposes four artifacts:

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
- **Schedule IR:** fixed tile/layout plan for CPU execution.
- **Tile IR:** CPU loop-nest, reduction, elementwise, or layout operations.
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

## 3. Current Frontends

Tessera currently has two frontend surfaces that lower to the same Graph IR
model:

- **Python AST frontend:** `@tessera.jit` inspects Python source, extracts
  `tessera.ops.*` calls, runs constraint/effect checks, and builds Graph IR.
- **Textual Graph DSL frontend:** `tessera.compiler.frontend.parse_text` and
  `lower_text_to_graph_ir` parse canonical straight-line `module` / `func`
  source and emit `GraphIRModule`.

The old `research/sandbox_compilers/tilec` parser remains a legacy research
sample and is not the canonical Tessera frontend.

## 4. Current Boundaries

Supported:

- straight-line `tessera.ops.*` dataflow using the canonical op catalog
- `matmul`, `gemm`, `conv2d`, normalization, activation, cast/transpose,
  dropout, flash attention, collectives, fused epilogue, FFT/DCT/spectral ops,
  and KV-cache reference state ops
- explicit matmul/GEMM CPU schedule tile through `@ts.jit(cpu_tile=(M, N, K))`
- explicit AST source for stdin/dynamic functions through `source=` or
  `source_path=`
- literal keyword attributes such as `softmax(axis=0)` and optimizer params
- NumPy arrays, tensor-like values with `.numpy()`, and Tessera `Tensor` values
  with `._data`

Not yet supported:

- dynamic Schedule IR selection
- native C ABI CPU launch
- GPU/ROCm/TPU dispatch from this frontend path
- arbitrary Python control flow lowering
- full textual DSL BNF coverage: kernels, meshes, schedule/dist statements,
  control flow, barriers, and asserts are future work
- full Tensor shape/type inference and MLIR verifier-backed construction

## 5. Developer Evaluation Criteria

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

## 6. Next Frontend Improvements

Priority order:

1. Preserve concrete shape/dtype/layout metadata in Graph IR.
2. Improve notebook/REPL source capture beyond explicit `source=` handoff.
3. Extend textual DSL coverage beyond straight-line Graph IR.
4. Replace textual artifacts with MLIR objects and verifier calls.
5. Connect Target IR to the runtime C ABI CPU backend.
6. Extend the same spine to measured CUDA execution.
