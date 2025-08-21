# Tessera Error Handling & Diagnostics Guide
*(CUDA-style “Error Handling” companion for Tessera)*

---

## 1. Overview

This guide explains how Tessera reports, classifies, and helps you debug errors and warnings across:
- **Front-ends**: Python API, C++ API
- **Compiler stack**: Graph IR → Schedule IR → Tile IR → Target IR (MLIR/LLVM/PTX/ROCm)
- **Runtime**: single-GPU, multi-GPU (NVLink/NVSwitch / XGMI), distributed collectives
- **Numerics**: determinism, NaN/Inf detection, mixed-precision pitfalls
- **Autotuner**: cost-model issues, on-device measurements, persistent caches

It is organized like NVIDIA CUDA’s Error Handling section, but adapted to Tessera’s multi-level IR and distributed runtime.

---

## 2. Error Model

Tessera unifies errors into **five categories**:

1. **Compile-Time Errors** — IR building or lowering fails (Graph/Schedule/Tile/Target).
2. **Launch-Time Errors** — kernel submission issues (invalid shapes, layouts, ABI mismatch).
3. **Runtime Execution Errors** — device-side faults (OOM, illegal access), driver/runtime failures.
4. **Distributed/Collective Errors** — communicator issues, topology mismatch, desync.
5. **Numerical/Convergence Errors** — NaN/Inf, divergence, nondeterminism violations.

### 2.1 Severity Levels
- **FATAL**: operation aborted; program should terminate or catch and exit.
- **ERROR**: failed call; caller must handle (exception / error code).
- **WARNING**: unusual condition; execution continues (e.g., precision fallback).
- **INFO**: diagnostics for profiling or debug runs.

### 2.2 Identifiers
Every error includes:
- **Code** (stable machine-readable enum/string)
- **Message** (human-friendly context)
- **Where** (IR level, pass/stage, device ID, stream)
- **Hints** (auto-suggested fixes when available)

Example (Python):
```text
TESSERA_RUNTIME_ERROR: OOM: requested 3.2 GiB, free 1.1 GiB
    where: tile-ir kernel matmul, device=GPU:0, stream=3
    hints: lower dtype to bf16 / reduce batch / enable activation checkpointing
```

---

## 3. Python & C++ Error Surfaces

### 3.1 Python Exceptions
- `tessera.errors.CompileError`
- `tessera.errors.LaunchError`
- `tessera.errors.RuntimeError`
- `tessera.errors.DistributedError`
- `tessera.errors.NumericsError`
- `tessera.errors.AutotuneError`
- `tessera.errors.TimeoutError`

Idioms:
```python
from tessera.errors import RuntimeError as TRuntimeError
try:
    y = model(x)               # may compile+launch+execute
except TRuntimeError as e:
    print(e.code, e.where, e.hints)
    raise
```

### 3.2 C++ Status Codes
C++ APIs return `Status` or throw `TesseraException` depending on build flags:
```cpp
Status s = rt.submit(graph);
if (!s.ok()) {
  std::cerr << s.code() << ": " << s.message() << " @ " << s.where() << "
";
}
```

---

## 4. Compile-Time Errors (Graph → Target)

| Code | Example Message | Typical Causes | Remedies |
|------|------------------|----------------|----------|
| `E_GRAPH_INVALID` | “Dangling tensor %7 not consumed” | Broken graph wiring; wrong shapes | Validate module, call `graph.check()` |
| `E_SHAPE_MISMATCH` | “matmul (M×K)·(K'×N) with K≠K'” | Wrong dims after reshape | Print shapes; add `op.assert_shape()` |
| `E_SCHEDULE_FUSE_FAIL` | “incompatible memory spaces for fusion” | Fusing ops with conflicting staging | Disable fusion or add `@schedule.hint` |
| `E_TILE_LOWERING` | “no TensorCore config for dtype=fp64 tile=48x48x48” | Unsupported MMA shape | Change tile sizes / dtype / enable FP32 accum |
| `E_TARGET_CODEGEN` | “PTX emission failed: invalid ldmatrix stride” | ABI mismatch, illegal layout | Use `op.inspect('tile')`; adjust layout |

**Diagnostics**
- `graph.dump_ir(level='graph|schedule|tile|target')`
- `tessera-mlir my_model.py --emit=tile-ir --debug`
- `TESSERA_DEBUG_IR=1` (env) to keep intermediate MLIR/LLVM artifacts

---

## 5. Launch-Time Errors

| Code | Example Message | Causes | Fix |
|------|------------------|--------|-----|
| `E_LAUNCH_INVALID_SHAPE` | “kernel expects 128-divisible K” | Tile requirements | Pad/tile to multiples; re-autotune |
| `E_LAUNCH_BAD_LAYOUT` | “expected row-major fragments” | Wrong layout tag | Convert with `op.layout_cast` |
| `E_LAUNCH_STREAM_BUSY` | “stream 2 dependency cycle” | Bad dependency or barrier | Review `graph.stream` deps; insert events |
| `E_LAUNCH_DEVICE_MISMATCH` | “tensor on GPU:1 launched on GPU:0” | Cross-device tensor | Call `x.to(device)` / align mesh |

**Diagnostics**
- `graph.trace(model, batch).print()` for dependency cycles
- `tessera-prof --trace trace.json` (Chrome trace timeline)

---

## 6. Runtime Execution Errors (Device-Side)

| Code | Example Message | Causes | Fix |
|------|------------------|--------|-----|
| `E_OOM` | “requested X GiB, free Y GiB” | Fragmentation, oversize batch | Reduce batch, use checkpointing, mixed precision |
| `E_ILLEGAL_ADDRESS` | “warp 17 lane 3 illegal addr” | OOB indexing, misaligned ldmatrix | Add bounds checks; verify tile strides |
| `E_MISALIGNED_ACCESS` | “ldmatrix requires 128-bit alignment” | Wrong shared mem layout | Align with `op.align(shared, 16/32/64)` |
| `E_TIMEOUT` | “kernel watchdog timeout” | Long-running kernel | Reduce tile size; split kernel; disable watchdog (dev only) |
| `E_DRIVER` | “CUDA driver error 700” | Underlying driver fault | Check dmesg, update driver/firmware |

**Diagnostics**
- Enable kernel assertions: `TESSERA_KERNEL_ASSERT=1`
- Collect crash repro: `tessera-prof --crash-dump dump/`
- Dump device state on fault: `TESSERA_DUMP_STATE=1`

---

## 7. Distributed/Collective Errors

| Code | Example Message | Causes | Fix |
|------|------------------|--------|-----|
| `E_COMM_INIT` | “failed to create communicator (rank mismatch)” | Env var ranks/world size wrong | Fix `RANK`, `WORLD_SIZE`, launcher config |
| `E_TOPOLOGY` | “NVLink domain not fully connected” | Mixed GPU types, cabling | Constrain mesh; rebuild communicator |
| `E_DESYNC` | “collective called with different tensor shapes across ranks” | Logic bug | Log shapes per rank; add `dist.barrier()` |
| `E_TIMEOUT_COMM` | “all-reduce timeout” | Stalled peer / deadlock | Check logs per rank; use `--trace` to find block |

**Diagnostics**
- `dist.inspect(tensor)` — show shard/placement per rank
- `dist.profile()` — comm/compute overlap timeline
- `TESSERA_DIST_DEBUG=1` — NCCL/XCCL verbose logs passthrough

---

## 8. Numerics & Determinism Errors

| Code | Example Message | Causes | Fix |
|------|------------------|--------|-----|
| `E_NAN_INF` | “NaN detected in softmax output” | Overflow/underflow | Use `op.softmax` (stable); max-subtraction |
| `E_LOSS_SCALING` | “FP16 underflow; scale too low” | Mixed precision | Enable dynamic loss scaling |
| `E_NONDETERMINISTIC` | “results vary between runs” | Atomics, reduction order | `numerics.profile('deterministic'|'strict')` |

**Diagnostics**
- `op.check_numerics(tensors)` to flag NaN/Inf
- `numerics.profile('strict')` to enforce cross-backend bitwise identity
- `numerics.validate_cross_hardware(model, backends=['ptx','rocm'])`

---

## 9. Autotuner & Profiling Errors

| Code | Example Message | Causes | Fix |
|------|------------------|--------|-----|
| `E_TUNE_SPACE_EMPTY` | “no valid tile configs” | Dtype/shape unsupported | Relax constraints; change tile sizes |
| `E_TUNE_MEASURE_FAIL` | “on-device run failed” | Kernel fault during measurement | Check kernel logs; reduce search space |
| `E_CACHE_IO` | “cannot write autotune cache” | Perms / path | Set `TESSERA_CACHE_DIR` to writable path |

**Diagnostics**
- `tessera-tune --dry-run --verbose`
- Cache location: `$HOME/.tessera/autotune/` or `TESSERA_CACHE_DIR`

---

## 10. Logging, Verbosity, and Environment

### 10.1 Verbosity
- `TESSERA_LOG_LEVEL=INFO|DEBUG|TRACE`
- Python: `tessera.logging.set_level("DEBUG")`

### 10.2 Keep Artifacts
- `TESSERA_DEBUG_IR=1` — keep IR dumps
- `TESSERA_KEEP_PTX=1` — keep PTX/LLVM objects
- `TESSERA_PROF_TRACE=trace.json` — emit Chrome trace

### 10.3 Safety Switches (Dev-Only)
- `TESSERA_DISABLE_FUSION=1`
- `TESSERA_DISABLE_PIPELINE=1`
- `TESSERA_DISABLE_TENSORCORES=1`

---

## 11. Recommended Debugging Workflow

1. **Reproduce deterministically**
   ```python
   from tessera import numerics
   numerics.profile("deterministic")
   ```
2. **Check shapes/layouts early**
   ```python
   graph.check(model)
   print(model.inspect("graph"))
   ```
3. **Dump IRs to locate lowering stage**
   ```bash
   tessera-mlir model.py --emit=schedule-ir --debug
   ```
4. **Run with profiling & traces**
   ```bash
   tessera-prof model.py --trace=trace.json --metrics=flops,bandwidth
   ```
5. **Narrow to minimal repro** (small shapes, 1 GPU)
6. **Validate numerics** (NaN/Inf, loss scaling)
7. **Escalate** with crash dumps, IR/trace, and repro script

---

## 12. Frequently Encountered Issues (FAQ)

**Q1. OOM on attention even with BF16.**  
A: Reduce sequence length or heads, enable activation checkpointing, use FlashAttention (already fused/streamed).

**Q2. Non-deterministic loss on multi-GPU.**  
A: Use `numerics.profile('deterministic')`; avoid atomic adds; rely on tree reductions.

**Q3. Collective timeout at step ~N.**  
A: Likely desync from shape mismatch. Log per-rank shapes; add `dist.barrier()` around conditional branches.

**Q4. Kernel crashes after changing tile sizes.**  
A: Misaligned `ldmatrix` strides or shared memory layout. Use `op.align(..., 32)` and re-autotune.

**Q5. PTX emission fails on fp64 TensorCores.**  
A: Not supported for that tile shape; lower to fp32 accum or change MMA tile to supported m×n×k.

---

## 13. Reference: Error Codes (Stable)

- `TESSERA_OK` — success
- `E_GRAPH_INVALID`
- `E_SHAPE_MISMATCH`
- `E_SCHEDULE_FUSE_FAIL`
- `E_TILE_LOWERING`
- `E_TARGET_CODEGEN`
- `E_LAUNCH_INVALID_SHAPE`
- `E_LAUNCH_BAD_LAYOUT`
- `E_LAUNCH_STREAM_BUSY`
- `E_LAUNCH_DEVICE_MISMATCH`
- `E_OOM`
- `E_ILLEGAL_ADDRESS`
- `E_MISALIGNED_ACCESS`
- `E_TIMEOUT`
- `E_DRIVER`
- `E_COMM_INIT`
- `E_TOPOLOGY`
- `E_DESYNC`
- `E_TIMEOUT_COMM`
- `E_NAN_INF`
- `E_LOSS_SCALING`
- `E_NONDETERMINISTIC`
- `E_TUNE_SPACE_EMPTY`
- `E_TUNE_MEASURE_FAIL`
- `E_CACHE_IO`
- `E_UNKNOWN`

---

## 14. Glossary
- **IR**: Intermediate Representation (Graph/Schedule/Tile/Target)
- **MMA**: Matrix-Multiply-Accumulate (Tensor Cores)
- **FTZ**: Flush-To-Zero
- **OOM**: Out-of-Memory
- **DP/TP/PP**: Data/Tensor/Pipeline parallelism

---

## 15. Change History
- v1.0 — Initial publication
