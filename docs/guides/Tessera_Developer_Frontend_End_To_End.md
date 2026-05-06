---
status: Informative
classification: Guide
authority: Developer frontend workflow for the object-backed compiler artifact spine
last_updated: 2026-05-06
---

# Tessera Developer Frontend: First End-To-End Path

This guide documents the reference executable/artifact compiler spine:

```text
@jit supported op graph -> Graph IR -> Schedule IR -> Tile IR -> Target IR -> CPU execution
```

It is intentionally a development/reference path. CPU execution uses NumPy and
explicit single-rank/state stubs; it is not native GPU, distributed, or C ABI
execution. Functions outside the executable CPU subset keep the eager Python
fallback and expose an explicit diagnostic explaining why. Artifact generation
for Graph/Schedule/Tile and Apple/ROCm Target IR is object-backed and verified
before textual inspection strings are emitted.

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

The public `gemm` alias lowers to the canonical Graph IR op
`tessera.matmul`. The emitted Schedule IR records `tile_m = 256`,
`tile_n = 128`, and `tile_k = 64`. Execution is still CPU/NumPy today; the tile
is an inspectable compiler schedule artifact that will become the input to
native CPU lowering.

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

The public artifact properties are textual inspection strings, but the active
Python compiler constructs them from verified object models:

- **Graph IR:** `GraphIRModule`, `GraphIRFunction`, and verifier diagnostics
  preserve shape/dtype/layout, source spans, meshes, constants, and returns.
- **Schedule IR:** `ScheduleIRModule` lowers mesh declarations, schedule
  directives, `schedule.mesh.region`, `schedule.pipeline.region`, stages, and
  yields.
- **Tile IR:** `TileIRModule` lowers scheduled work to `tile.*`,
  `tessera.attn.*` FA-4 helpers, and `tessera.queue.*` barriers.
- **Target IR:** `TargetIRModule` lowers Tile IR into verified Apple CPU/GPU
  and ROCm hardware-free target artifacts. Metalium/NVIDIA keep their existing
  artifact renderers in this path.

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
- **Textual DSL frontend:** `tessera.compiler.frontend.parse_text` and
  `lower_text_to_graph_ir` parse canonical `module` / `func` / `kernel` source,
  mesh/type/constant declarations, schedule/dist statements, barriers/asserts,
  and structured control-flow markers, then emit `GraphIRModule`.

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
- hardware-free Target IR artifact selection with `target="rocm"`,
  `target="metalium"`, `target="apple_cpu"`, and `target="apple_gpu"`
- direct textual DSL lowering through Graph IR into Schedule/Tile/Apple/ROCm
  Target IR object models

Not yet supported:

- cost-model driven Schedule IR selection
- native C ABI CPU launch
- GPU/ROCm/Apple/Metalium runtime dispatch from this frontend path
- arbitrary Python control flow lowering in the Python AST frontend
- complete native MLIR Python binding construction for every layer
- full Tensor shape/type inference beyond the current verifier checks

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
def unsupported_control_flow(x):
    if x.sum() > 0:
        return ts.ops.relu(x)
    return x

assert not unsupported_control_flow.uses_compiled_path
print(unsupported_control_flow.explain_lowering())
```

## 6. Next Frontend Improvements

Priority order:

1. Improve notebook/REPL source capture beyond explicit `source=` handoff.
2. Extend native MLIR Python binding construction beyond the current verified
   Python object models.
3. Add cost-model driven Schedule IR selection and autotune feedback.
4. Connect Target IR to the runtime C ABI CPU backend.
5. Extend the same spine to measured CUDA/ROCm/Apple execution.

## 7. Target IR Artifact Selection

Non-CPU targets currently produce inspectable Target IR artifacts while
execution falls back to the original Python function:

```python
@ts.jit(target="apple_gpu")
def mm(A, B):
    return ts.ops.matmul(A, B)

print(mm.has_target_artifacts)
print(mm.target_ir)
print(mm.runtime_artifact().metadata["runtime_status"])
```

Current hardware-free target contracts:

- `target="rocm"` lowers Tile IR to `tessera_rocm.*` MFMA, async-copy, wait,
  elementwise, and diagnostic artifacts.
- `target="metalium"` emits `tessera_metalium.*` DMA and matmul artifacts.
- `target="apple_cpu"` lowers Tile IR to Accelerate/vecLib-style
  `tessera_apple.cpu.*` artifacts.
- `target="apple_gpu"` lowers Tile IR to Metal/MPS-style
  `tessera_apple.gpu.*` kernel and dispatch artifacts.
