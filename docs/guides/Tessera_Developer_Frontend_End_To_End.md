---
status: Informative
classification: Guide
authority: Developer frontend workflow for the object-backed compiler artifact spine
last_updated: 2026-10-24
---

# Tessera Developer Frontend: First End-To-End Path

This guide documents the reference executable/artifact compiler spine:

```text
@jit supported op graph -> Graph IR -> Schedule IR -> Tile IR -> Target IR -> CPU execution
```

It is intentionally a development/reference path. CPU execution uses NumPy and
explicit single-rank/state stubs; eligible rank-2 f32 GEMMs first try the native
CPU runtime ABI and fall back to the NumPy plan (recording
`FallbackReason.CAPABILITY_NOT_READY`) when that launch is unavailable. It is
not native GPU or distributed execution — other targets are covered in §7.
Functions outside the executable CPU subset keep the eager Python fallback
and expose an explicit diagnostic explaining why. Artifact generation for
Graph/Schedule/Tile and CPU, NVIDIA/CUDA, Apple, and ROCm Target IR is
object-backed and verified before textual inspection strings are emitted.

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
the original Python function body. `mm.last_fallback_reason` says whether the
native CPU launch was taken (`None`) or the NumPy plan ran instead.

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
- **Tile IR:** `TileIRModule` lowers scheduled work to `tile.*` ops and
  `tessera_attn.*` FA-4 helpers, with async ordering carried as explicit
  SSA-like token lineage. The old `tessera.queue.*` markers are gone from the
  Python model as well: its verifier now rejects any `tessera.queue.*` op with
  `TILE_IR_LEGACY_QUEUE` (the `tessera.queue` MLIR dialect was deleted
  2026-08-10, Decisions #29/#31).
- **Target IR:** `TargetIRModule` lowers Tile IR into verified CPU/x86,
  NVIDIA/CUDA, Apple CPU/GPU, and ROCm hardware-free target artifacts.

To see whether a function used the compiler path:

```python
print(mm.is_executable)
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

assert mm.is_executable
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
- hardware-free Target IR artifact selection with `target="cpu"`,
  `target="nvidia_sm80"`, `target="nvidia_sm90"`, `target="nvidia_sm100"`,
  `target="nvidia_sm120"`, `target="rocm"`, `target="apple_cpu"`, and
  `target="apple_gpu"`
- direct textual DSL lowering through Graph IR into Schedule/Tile/Target IR
  object models

Not yet supported:

- cost-model driven Schedule IR selection
- native C ABI CPU launch beyond the eligible rank-2 f32 GEMM fast path
- native GPU/ROCm/Apple dispatch for families without exact-device proof on
  that target (see §7)
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

assert not unsupported_control_flow.is_executable
print(unsupported_control_flow.explain_lowering())
```

A body can also be executable yet contain a call the AST front end drops — raw
NumPy or an unregistered helper. Since 2026-10-23 such a call no longer runs
the truncated graph: `JitFn.__call__` runs the original Python function and
records `FallbackReason.FRONTEND_UNLOWERED_CALL`. It raises instead under
`native_required=True` or a differentiation request (only tracer-routed
`apple_gpu` bodies are exempt). On a native target, a dropped `tessera.ops`
call is refused at decoration with `JIT_UNLOWERED_OP_NATIVE_TARGET`.

```python
@ts.jit
def mixed(x):
    y = ts.ops.relu(x)
    return np.exp(y)          # not a Graph op: dropped by the front end

mixed(np.array([-1.0, 1.0], dtype=np.float32))   # == np.exp(relu(x))
print(mixed.last_fallback_reason)                 # FRONTEND_UNLOWERED_CALL
```

## 6. Next Frontend Improvements

Historical priority order (the live compiler queue is
`docs/audit/compiler/INTEGRATED_COMPILER_PLAN.md#live-queue`; items 4 and 5 are
partially landed):

1. Improve notebook/REPL source capture beyond explicit `source=` handoff.
2. Extend native MLIR Python binding construction beyond the current verified
   Python object models.
3. Add cost-model driven Schedule IR selection and autotune feedback.
4. Connect Target IR to the runtime C ABI CPU backend.
5. Extend the same spine to measured CUDA/ROCm/Apple execution.

## 7. Target IR Artifact Selection

Non-CPU targets always produce inspectable Target IR artifacts. Whether a call
then executes natively depends on the target and op family: Apple GPU/CPU,
ROCm (gfx1151 and gfx1201, separately) and NVIDIA sm_120 run native lanes for
covered families on their own hardware, and everything else falls back to a
reference lane or refuses with a diagnostic. `runtime_status` and
`last_fallback_reason` say which happened;
`docs/audit/generated/runtime_execution_matrix.md` is the record of what is
proven.

```python
@ts.jit(target="apple_gpu")
def mm(A, B):
    return ts.ops.matmul(A, B)

print(mm.has_target_artifacts)
print(mm.target_ir)
print(mm.runtime_artifact().metadata["runtime_status"])
```

Current hardware-free target contracts:

- `target="rocm"` lowers Tile IR to `tessera_rocm.*` MFMA/WMMA, async-copy,
  wait, elementwise, and diagnostic artifacts.
- `target="apple_cpu"` lowers Tile IR to Accelerate/vecLib-style
  `tessera_apple.cpu.*` artifacts.
- `target="apple_gpu"` lowers Tile IR to Metal/MPS-style
  `tessera_apple.gpu.*` kernel and dispatch artifacts.
