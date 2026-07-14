---
status: Informative
classification: Informative
last_updated: 2026-07-14
---

> **Current-state note (2026-05-20):** This is historical architecture guidance. Phase labels below are design lineage, not current support claims. For implementation status, use `docs/spec/COMPILER_REFERENCE.md`, `docs/audit/generated/support_table.md`, `docs/audit/generated/e2e_op_coverage.md`, and `docs/spec/VALIDATION_SPINE.md`.


# Tessera Compiler — Schedule IR Design
*Scope:* Graph IR → Schedule IR → Tile IR  
*Status:* Draft v0.2 (with programmer context)

> **API note:** Examples in this document using `gir.gemm` and `gir.rmsnorm_safe` reflect
> internal MLIR op names (`tessera.matmul`, `tessera.layer_norm` in the tessera dialect).
> `Region[write]` and `Region[reduce]` should be `Region["write"]` and `Region["reduce_sum"]`
> in Python source. Current JIT functions expose `schedule_ir` as a textual
> artifact emitted from the verified Python `ScheduleIRModule` object model.
> See `docs/CANONICAL_API.md` for authoritative names.

---

## 0. Goals
- Transform **Graph IR** (typed, effect-aware) into **Schedule IR** (fusion, tiling, pipelining).  
- Capture both **automatic and manual scheduling** strategies.  
- Provide a portable, legal schedule representation across accelerators.  
- Preserve legality with respect to **region privileges and effects**.  
- Expose knobs for **autotuning** (search spaces, cost models, runtime measures).

---

## 1. Role in the Compiler Pipeline

```text
Graph IR — high-level ops, effects, privileges
    │
    ▼
Schedule IR — fusion, tiling, pipelining, collective overlap
    │
    ▼
Tile IR — explicit tiles, loads/stores, shared memory, cp.async, mma
    │
    ▼
Target IR — PTX, CUDA Tile IR, LLVM
```

Schedule IR bridges semantic operations and explicit tiled kernels.

---

## 2. Core Concepts

- **Schedules**: explicit block/warp sizes, stages, vector widths, swizzles.  
- **Fusion**: merge compatible ops (e.g., GEMM + bias + norm).  
- **Pipelines**: async prefetch, double-buffering, compute/comm overlap.  
- **Collective fusion**: overlap allreduce/reduce_scatter with compute.  
- **Autotune spaces**: parameterized search ranges per schedule.

---

## 3. Schedule IR Entities

**Module / Func**
- `schedule.module`  
- `schedule.func @name` {attrs}  

**Ops** *(design-lineage names below; the concrete `schedule` dialect defined in
`src/compiler/programming_model/ir/schedule/ScheduleMeshPipelineOps.td` exposes
`mesh.define`, `mesh.region`, `pipeline.region`, `stage`, `tile`, `elementwise`,
`layout`, `warp`, `optimizer_shard`, `prefetch`, `async_copy`, `await_movement`,
`artifact`, `knob`, `yield` — there are no literal `fuse`/`split`/`autotune`/`collective_overlap` ops)*
- `schedule.fuse` : merge ops into a fused region.  
- `schedule.split` : split iteration space.  
- `schedule.tile` : assign block, warp, stage dimensions.  
- `schedule.pipeline` : mark pipelined loops with async stages.  
- `schedule.autotune` : define search space for autotuner.  
- `schedule.collective_overlap` : annotate comm/compute overlap.  

**Attributes**
- `{block=[128,128], warps=8, stages=3, vector=8, swizzle="xor"}`  
- `{autotune_space={BM=[64,128], BN=[64,96,128], BK=[32,64], warps=[4,8], stages=[2,3]}}`  

---

## 4. Privileges and Effects

- Fusions must respect **Region[write] exclusivity**.  
- Reductions (`Region[reduce]`) may be fused if operations are associative/commutative.  
- Effects (RNG, IO, collectives) define barriers — cannot reorder across them.  

---

## 5. NVIDIA Mapping

- **Blocks/warps** → CUDA blocks/warps.  
- **Stages** → async pipeline depth, lowered to `cp.async` / `tma.load`.  
- **Vector widths** → vectorized loads/stores.  
- **Collective overlap** → NCCL streams + CUDA Graph dependencies.  

---

## 6. Example Transformation

**Graph IR**
```mlir
%y = gir.gemm %x, %w
%z = gir.rmsnorm_safe %y
```

**Schedule IR**
```mlir
%s0 = schedule.fuse %y, %z
schedule.tile %s0 {block=[128,128], warps=8, stages=3, vector=8}
schedule.pipeline %s0 {stages=3}
```

---

## 7. Autotuning Support

- **Define spaces** in Schedule IR:  
```mlir
schedule.autotune %gemm {BM=[64,128], BN=[64,128], BK=[32,64], warps=[4,8], stages=[2,3]}
```

- Autotuner explores legal configurations.  
- Cost models + runtime measures guide search.  
- Persistent cache keyed by `(Graph IR hash, arch)` ensures reproducibility.

---

## 8. Diagnostics

- **Illegal tilings**: dimension not divisible.  
- **Register pressure**: warn if exceeding hardware budget.  
- **Fusion violation**: write privilege conflict.  
- **Pipeline misuse**: missing barriers.  

Diagnostics are surfaced with source locations and suggested fixes.

---

## 9. Implementation Notes

- Implement as MLIR dialect: `schedule`.  
- SSA form; ops map to `tile` dialect during lowering.  
- Autotuner integrated with runtime measurement harness.  
- LSP integration: schedule suggestions and inspection.

---

## 10. Example: GEMM Scheduling Flow

> **API note:** `@kernel.autotune(space=...)` and `@kernel.schedule(...)` below are
> design-lineage sketches — the `tessera.kernel` decorator exposes no `.autotune` or
> `.schedule` attribute. The canonical autotune surface is the top-level
> `tessera.autotune(op, shapes=..., dtype=..., ...)` function (see `docs/CANONICAL_API.md`).

**High-level source**
```python
@kernel.autotune(space=dict(BM=[128,256], BN=[128,256], BK=[64], stages=[2,3]))
def gemm(A, B, C): ...
```

**Graph IR**
```mlir
%y = gir.gemm %A, %B
```

**Schedule IR**
```mlir
schedule.autotune %y {BM=[128,256], BN=[128,256], BK=[64], stages=[2,3]}
schedule.tile %y {block=[128,128], warps=8, stages=3, vector=8}
schedule.pipeline %y {stages=3}
```

**Tile IR**
```mlir
tile.load ...
tile.mma ...
tile.store ...
```

---

## 11. For Programmers: Why Schedule IR Matters

Schedule IR is where your high-level Tensor ops (from Graph IR) are **mapped onto tiling, fusion, and pipelining strategies**.  

For programmers, this stage explains:  
- What your `@kernel.autotune` decorator expands into.  
- Which tiling/block/warp sizes the compiler chose (or you overrode).  
- How loops were pipelined with `cp.async` or collectives.  
- Why a kernel may be fast on one GPU but needs retuning on another.  

You don’t write Schedule IR directly — but you can **inspect it** to understand why your kernel performs a certain way.

---

## 12. Inspecting Schedule IR

```python
@kernel.autotune(space=dict(BM=[128,256], BN=[128,256], BK=[64], stages=[2,3]))
def gemm(A, B, C): return gemm_tile(A,B,C)

gemm.schedule_ir   # textual Schedule IR artifact (or gemm.lowering_artifacts())
```

Example output:  
```mlir
schedule.autotune %y {BM=[128,256], BN=[128,256], BK=[64], stages=[2,3]}
schedule.tile %y {block=[128,128], warps=8, stages=3, vector=8}
schedule.pipeline %y {stages=3}
```

---

## 13. Common Errors at Schedule IR Stage

- **Illegal tiling**:  
  *“Tile size 128x128 not divisible by tensor dim 96.”*  

- **Privilege violation**:  
  *“Fusion of ops %x and %y illegal due to Region[write] conflict.”*  

- **Excess register pressure**:  
  *“Tiling choice leads to 320 registers per thread (max 255). Try reducing stages or vector width.”*  

- **Pipeline misuse**:  
  *“cp.async scheduled without matching barrier.”*  

---

## 14. Programmer Workflow Example

1. You write:  
   ```python
   @kernel.autotune(space=dict(BM=[128,256], BN=[128,256], BK=[64]))
   def gemm_kernel(A,B,C): return gemm(A,B,C)
   ```

2. Inspect Schedule IR:  
   ```python
   gemm_kernel.schedule_ir
   ```
   → Shows autotune space and tiling strategy.

3. Runtime autotuner picks best config and caches it.  

4. You override:  
   ```python
   @kernel.schedule(block=(64,64), warps=4, stages=2)
   def gemm_small(...): ...
   ```

   → Inspect IR to confirm schedule.  

This lets you balance **autotune vs manual control**.

---

## 15. Summary for Programmers

- **Schedule IR** explains how Graph IR ops are tiled, fused, pipelined.  
- Use **`fn.schedule_ir`** or `fn.lowering_artifacts()` for current Schedule IR inspection.  
- Common issues: illegal tilings, privilege conflicts, register pressure.  
- Programmers can let Tessera autotune, or manually override with `@kernel.schedule`.  
- Understanding Schedule IR helps bridge **productivity → performance**.
