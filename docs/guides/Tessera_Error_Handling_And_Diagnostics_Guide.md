---
status: Normative
classification: Normative
authority: Canonical error handling and diagnostics guidance; defers runtime status details to docs/spec/RUNTIME_ABI_SPEC.md and Python symbols to docs/spec/PYTHON_API_SPEC.md
last_updated: 2026-04-28
---

# Tessera Error Handling And Diagnostics Guide

This guide explains how Tessera reports, classifies, and helps debug errors
across the Python API, C++ API, compiler stack, runtime, distributed execution,
numerics, and autotuning. It is the CUDA-style error-handling companion for
Tessera, adapted to a multi-level IR and distributed runtime.

Use this guide with:

| Topic | Reference |
|-------|-----------|
| Python public API | `docs/spec/PYTHON_API_SPEC.md` |
| Runtime C ABI status codes | `docs/spec/RUNTIME_ABI_SPEC.md` |
| Standard operator error codes | `docs/operations/Tessera_Standard_Operations.md` |
| QA and reliability patterns | `docs/guides/Tessera_QA_Reliability_Guide.md` |
| Production replay and chaos testing | `docs/guides/Tessera_Production_Reliability_And_Chaos_Guide.md` |

## Error Model

Tessera groups failures into five categories:

| Category | Scope | Examples |
|----------|-------|----------|
| Compile-time errors | Graph IR, Schedule IR, Tile IR, Target IR, MLIR/LLVM/PTX/ROCm lowering | Invalid graph, shape mismatch, failed fusion, unsupported tile lowering, target codegen failure. |
| Launch-time errors | Kernel submission, runtime ABI validation, stream dependencies | Invalid launch shape, bad layout, stream dependency cycle, device mismatch. |
| Runtime execution errors | Device-side execution and driver/runtime calls | OOM, illegal address, misaligned access, watchdog timeout, driver failure. |
| Distributed/collective errors | Meshes, communicators, topology, rank synchronization | Communicator init failure, topology mismatch, collective desync, collective timeout. |
| Numerical/convergence errors | Numeric policy, determinism, NaN/Inf, mixed precision | NaN/Inf, loss scaling failure, nondeterministic result under strict mode. |

Autotuner and profiling failures are reported as their own codes because they
often happen while measuring a candidate implementation, not while running the
user's selected program.

## Severity Levels

| Severity | Meaning |
|----------|---------|
| `FATAL` | Operation aborted and the process should usually terminate or checkpoint/restart. |
| `ERROR` | The call failed; caller must handle the exception or status. |
| `WARNING` | Unusual condition; execution can continue, often with a fallback. |
| `INFO` | Diagnostic detail for profiling, traces, and debug runs. |
| `NOTE` | Additional context attached to another diagnostic. |

## Diagnostic Shape

Every structured Tessera diagnostic should carry:

| Field | Purpose |
|-------|---------|
| Code | Stable machine-readable enum or string such as `E_SHAPE_MISMATCH`. |
| Message | Human-readable context. |
| Where | IR level, pass/stage, op, device ID, stream, or rank context. |
| Source location | Python file/line or MLIR location when available. |
| Hints | Suggested fixes when Tessera can infer a useful next action. |

Example Python formatting:

```text
ERROR [E_OOM] matmul requested 3.2 GiB, free 1.1 GiB
  where: tile-ir, op=tessera.matmul, device=GPU:0, stream=3
  hints: reduce batch size; enable activation checkpointing; use bf16 policy
```

## Python Error Surface

The Phase 6 Python diagnostics spine is `tessera.diagnostics`.

Current implemented classes:

| Symbol | Purpose |
|--------|---------|
| `DiagnosticLevel` | Severity enum including `INFO`, `NOTE`, `WARNING`, `ERROR`, `FATAL`. |
| `TesseraErrorCode` | Stable diagnostic code enum. |
| `DiagnosticWhere` | Structured compiler/runtime location. |
| `SourceLocation` | Python or MLIR source location. |
| `TesseraDiagnostic` | One collected diagnostic. |
| `TesseraError` | Base structured compiler error. |
| `TesseraShapeError` | Shape/rank mismatch. |
| `TesseraTargetError` | Target lowering/codegen failure. |
| `TesseraTypeError` | Dtype and type policy failure. |
| `TesseraNotImplementedError` | Unimplemented lowering or backend path. |
| `ErrorReporter` | Collects warnings/errors and raises structured exceptions. |

Runtime ABI failures currently surface as `tessera.runtime.TesseraRuntimeError`.
Future Python API aliases may add more specific names such as `CompileError`,
`LaunchError`, `DistributedError`, `NumericsError`, `AutotuneError`, and
`TimeoutError`; these should map back to the same stable codes.

Example:

```python
from tessera.diagnostics import (
    DiagnosticWhere,
    ErrorReporter,
    TesseraErrorCode,
)

reporter = ErrorReporter()
reporter.error(
    "matmul shape mismatch: (M x K) times (K' x N)",
    op_name="tessera.matmul",
    code=TesseraErrorCode.SHAPE_MISMATCH,
    where=DiagnosticWhere(ir_level="graph-ir", pass_name="shape-check"),
    hints=["print tensor shapes", "add an op.assert_shape guard"],
)
reporter.raise_if_errors()
```

## C++ And Runtime Status Surface

C++ and C ABI layers should return status values or throw `TesseraException`
depending on build flags and binding layer policy. The runtime C ABI remains
the authority for low-level status codes.

```cpp
Status s = rt.submit(graph);
if (!s.ok()) {
  std::cerr << s.code() << ": " << s.message()
            << " @ " << s.where() << "\n";
}
```

Python runtime wrappers map non-success statuses to `TesseraRuntimeError` with
the failing function name and `TsrStatus`.

## Compile-Time Errors

| Code | Example message | Typical causes | Remedies |
|------|-----------------|----------------|----------|
| `E_GRAPH_INVALID` | Dangling tensor `%7` not consumed | Broken graph wiring, invalid region ownership | Run graph verifier, dump Graph IR, check producer/consumer shape contracts. |
| `E_SHAPE_MISMATCH` | `matmul (M x K) * (K' x N)` with `K != K'` | Wrong reshape, shard, transpose, or symbolic binding | Print shapes, add shape constraints, call `check_shape`. |
| `E_SCHEDULE_FUSE_FAIL` | Incompatible memory spaces for fusion | Fusion crosses conflicting staging or movement effects | Disable the specific fusion, add a schedule hint, inspect Schedule IR. |
| `E_TILE_LOWERING` | No TensorCore config for dtype/tile | Unsupported MMA shape, dtype, or accumulator policy | Change tile size or dtype, use FP32 accum, re-autotune. |
| `E_TARGET_CODEGEN` | PTX emission failed for layout | ABI mismatch, illegal vector layout, target feature mismatch | Inspect Tile/Target IR and target profile. |

Diagnostics:

```text
graph.dump_ir(level="graph|schedule|tile|target")
tessera-mlir model.py --emit=tile-ir --debug
TESSERA_DEBUG_IR=1
```

## Launch-Time Errors

| Code | Example message | Causes | Fix |
|------|-----------------|--------|-----|
| `E_LAUNCH_INVALID_SHAPE` | Kernel expects 128-divisible K | Tile requirement not met | Pad, choose a compatible tile, or re-autotune. |
| `E_LAUNCH_BAD_LAYOUT` | Expected row-major fragments | Wrong layout tag or stale schedule artifact | Convert layout or invalidate the artifact. |
| `E_LAUNCH_STREAM_BUSY` | Stream dependency cycle | Bad event/barrier ordering | Inspect graph stream dependencies and insert events. |
| `E_LAUNCH_DEVICE_MISMATCH` | Tensor on GPU:1 launched on GPU:0 | Cross-device tensor placement mismatch | Move tensors or align mesh placement. |

Diagnostics:

```text
graph.trace(model, batch).print()
tessera-prof --trace trace.json
```

## Runtime Execution Errors

| Code | Example message | Causes | Fix |
|------|-----------------|--------|-----|
| `E_OOM` | Requested X GiB, free Y GiB | Oversized batch, fragmentation, cache pressure | Reduce batch, checkpoint activations, use mixed precision, prune cache. |
| `E_ILLEGAL_ADDRESS` | Warp lane illegal address | OOB indexing, wrong stride, bad mask | Enable kernel assertions and bounds checks. |
| `E_MISALIGNED_ACCESS` | `ldmatrix` requires alignment | Shared memory layout mismatch | Align shared layout and re-run verifier. |
| `E_TIMEOUT` | Kernel watchdog timeout | Runaway loop or too-large kernel | Reduce tile, split kernel, adjust watchdog in dev only. |
| `E_DRIVER` | Backend driver failure | CUDA/HIP/driver/runtime failure | Collect crash dump and backend logs. |

Diagnostics:

```text
TESSERA_KERNEL_ASSERT=1
TESSERA_DUMP_STATE=1
tessera-prof --crash-dump dump/
```

## Distributed And Collective Errors

| Code | Example message | Causes | Fix |
|------|-----------------|--------|-----|
| `E_COMM_INIT` | Failed to create communicator | Rank/world size mismatch | Fix launcher config and environment. |
| `E_TOPOLOGY` | NVLink domain not fully connected | Mixed device type or cabling/topology mismatch | Constrain mesh or rebuild communicator. |
| `E_DESYNC` | Collective called with different shapes | Conditional branch or rank-local shape drift | Log per-rank shapes, add barriers around conditionals. |
| `E_TIMEOUT_COMM` | All-reduce timeout | Stalled peer or deadlock | Use distributed trace and rank-local logs. |

Diagnostics:

```text
dist.inspect(tensor)
dist.profile()
TESSERA_DIST_DEBUG=1
```

## Numerics And Determinism Errors

| Code | Example message | Causes | Fix |
|------|-----------------|--------|-----|
| `E_NAN_INF` | NaN detected in softmax output | Overflow, invalid input, bad scale | Use stable primitives and numeric sentinels. |
| `E_LOSS_SCALING` | FP16 underflow; scale too low | Mixed precision instability | Enable dynamic loss scaling or FP32 accum. |
| `E_NONDETERMINISTIC` | Results vary between runs | Atomics, reduction order, RNG stream drift | Enable deterministic mode and fixed schedule artifacts. |

Diagnostics:

```text
op.check_numerics(tensors)
numerics.profile("strict")
numerics.validate_cross_hardware(model, backends=["ptx", "rocm"])
```

## Autotuner And Profiling Errors

| Code | Example message | Causes | Fix |
|------|-----------------|--------|-----|
| `E_TUNE_SPACE_EMPTY` | No valid tile configs | Overconstrained search or unsupported dtype/shape | Relax search space or change dtype/tile constraints. |
| `E_TUNE_MEASURE_FAIL` | On-device run failed | Candidate kernel faulted during measurement | Inspect candidate logs and reduce search space. |
| `E_CACHE_IO` | Cannot write autotune cache | Permissions or unwritable path | Set `TESSERA_CACHE_DIR` to a writable directory. |

Diagnostics:

```text
tessera-tune --dry-run --verbose
TESSERA_CACHE_DIR=/path/to/cache
```

## Logging, Verbosity, And Environment

| Variable/API | Purpose |
|--------------|---------|
| `TESSERA_LOG_LEVEL=INFO|DEBUG|TRACE` | Controls runtime/compiler logging verbosity. |
| `tessera.logging.set_level("DEBUG")` | Python logging control when available. |
| `TESSERA_DEBUG_IR=1` | Keep intermediate IR dumps. |
| `TESSERA_KEEP_PTX=1` | Keep generated PTX/LLVM objects. |
| `TESSERA_PROF_TRACE=trace.json` | Emit Chrome trace format. |
| `TESSERA_DISABLE_FUSION=1` | Dev-only switch to isolate fusion problems. |
| `TESSERA_DISABLE_PIPELINE=1` | Dev-only switch to isolate pipeline scheduling. |
| `TESSERA_DISABLE_TENSORCORES=1` | Dev-only switch to isolate MMA/tensor-core lowering. |

## Recommended Debugging Workflow

1. Reproduce deterministically with fixed seeds, deterministic mode, and a stable schedule artifact.
2. Check shapes, dtypes, layouts, and mesh placement early.
3. Dump IR at the first failing boundary: Graph, Schedule, Tile, or Target.
4. Run with profiling and traces to see launch order, streams, and collectives.
5. Narrow to a minimal repro using small shapes and one device.
6. Validate numerics with NaN/Inf sentinels and strict numeric profiles.
7. Escalate with a repro pack: source, seed, graph hash, schedule hash, IR dumps, trace, crash dump, target, and backend versions.

## FAQ

| Question | Answer |
|----------|--------|
| OOM on attention even with BF16 | Reduce sequence length or heads, enable activation checkpointing, prune KV cache, and use FlashAttention lowering. |
| Non-deterministic loss on multi-GPU | Use deterministic mode, stable RNG streams, deterministic collectives, and schedule artifact reuse. |
| Collective timeout at step N | Suspect rank desync. Log per-rank shapes and branch decisions, then add barriers around conditional collectives. |
| Kernel crashes after tile changes | Suspect misaligned fragments or shared-memory layout. Inspect Tile IR and re-autotune. |
| Target codegen fails for fp64 TensorCores | The target likely lacks that MMA path. Use supported FP32 accumulation or change the tile/policy. |

## Stable Error Code Reference

| Code | Meaning |
|------|---------|
| `TESSERA_OK` | Success. |
| `E_GRAPH_INVALID` | Invalid Graph IR or graph wiring. |
| `E_SHAPE_MISMATCH` | Shape contract failed. |
| `E_SCHEDULE_FUSE_FAIL` | Schedule fusion failed. |
| `E_TILE_LOWERING` | Tile lowering failed. |
| `E_TARGET_CODEGEN` | Target code generation failed. |
| `E_LAUNCH_INVALID_SHAPE` | Launch shape violates kernel contract. |
| `E_LAUNCH_BAD_LAYOUT` | Launch layout violates kernel contract. |
| `E_LAUNCH_STREAM_BUSY` | Stream dependency or scheduling problem. |
| `E_LAUNCH_DEVICE_MISMATCH` | Tensor and launch device mismatch. |
| `E_OOM` | Out of memory. |
| `E_ILLEGAL_ADDRESS` | Device illegal memory access. |
| `E_MISALIGNED_ACCESS` | Alignment contract failed. |
| `E_TIMEOUT` | Execution timeout. |
| `E_DRIVER` | Driver/runtime backend failure. |
| `E_COMM_INIT` | Communicator initialization failed. |
| `E_TOPOLOGY` | Topology does not satisfy mesh/collective contract. |
| `E_DESYNC` | Distributed rank or collective desynchronization. |
| `E_TIMEOUT_COMM` | Collective or distributed operation timed out. |
| `E_NAN_INF` | NaN or Inf detected. |
| `E_LOSS_SCALING` | Loss scaling failed or mixed-precision underflow detected. |
| `E_NONDETERMINISTIC` | Determinism contract violated. |
| `E_TUNE_SPACE_EMPTY` | Autotune search space has no valid candidates. |
| `E_TUNE_MEASURE_FAIL` | Autotune measurement failed. |
| `E_CACHE_IO` | Autotune or schedule cache I/O failed. |
| `E_UNKNOWN` | Unknown or uncategorized failure. |

## Glossary

| Term | Meaning |
|------|---------|
| IR | Intermediate Representation: Graph, Schedule, Tile, or Target. |
| MMA | Matrix-multiply-accumulate hardware operation. |
| FTZ | Flush-to-zero floating-point behavior. |
| OOM | Out of memory. |
| DP/TP/PP | Data, tensor, and pipeline parallelism. |
