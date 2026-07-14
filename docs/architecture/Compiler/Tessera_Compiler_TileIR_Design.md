---
status: Informative
classification: Informative
last_updated: 2026-07-14
---

> **Current-state note (2026-05-20):** This is historical architecture guidance. Phase labels below are design lineage, not current support claims. For implementation status, use `docs/spec/COMPILER_REFERENCE.md`, `docs/audit/generated/support_table.md`, `docs/audit/generated/e2e_op_coverage.md`, and `docs/spec/VALIDATION_SPINE.md`.


# Tessera Compiler — Tile IR Design
*Scope:* Schedule IR → Tile IR → Target IR  
*Status:* Draft v0.2 (with programmer context)

> **API note:** Legacy Tile IR inspection helpers shown in older examples are
> superseded by `fn.tile_ir` and `fn.lowering_artifacts()`, which emit textual
> artifacts from the verified Python `TileIRModule` object model.

---

## 0. Goals
- Represent tiled execution explicitly in IR.  
- Capture **tile-level memory movement, synchronization, and compute ops**.  
- Serve as a **portable virtual ISA** across GPU backends.  
- Lower to backend **Target IR**: PTX, CUDA Tile IR, LLVM.  
- Provide analyzability for register pressure, bank conflicts, and roofline modeling.

---

## 1. Position in the Compiler Pipeline

```text
Graph IR (ops, effects, privileges)
    │
    ▼
Schedule IR (fusion, tiling, pipelining, collectives)
    │
    ▼
Tile IR (explicit tiles, memory ops, mma, collectives)
    │
    ▼
Target IR (PTX, CUDA Tile IR, LLVM)
```

---

## 2. Core Abstractions

- **Tile context**  
  - Tile dimensions, indices, ranges.  
  - Iteration scopes lowered from Schedule IR.  

- **Memory operations**  
  - `tile.load` / `tile.store`  
  - `tile.cp_async` (async prefetch)  
  - `tma.load` / `tma.store` (Tensor Memory Accelerator, Hopper/Blackwell).  

- **Shared memory**  
  - `tshared.alloc` with layout attributes (`swizzle="xor"`, `bank=...`).  

- **Barriers & synchronization**  
  - `tile.barrier` for group sync.  
  - Pipeline staging control.  

- **Compute intrinsics**  
  - `tile.mma` → tensor core matrix multiply.  
  - `tile.dot` → vector dot products.  
  - `softmax_online`, `tile.softmax_accumulate`.  

- **Intra-group collectives**  
  - `warp.reduce`, `warp.scan`, `warp.shuffle`.  
  - `block.reduce`, `block.broadcast`.  

---

## 3. IR Structure

- `tile.module`  
- `tile.func @name` attributes `{block, warps, stages, vector}`  
- **Types**:  
  - `tile.tensor<shape, dtype, space=reg|shared|global>`  
  - `tile.scalar<dtype>`  
- **Attributes**:  
  - `{vector=8, stages=3, swizzle="xor"}`  
  - `{space="shared"}`  

---

## 4. Memory Spaces

- **reg**: per-lane registers.  
- **shared**: per-group, low-latency scratchpad.  
- **global**: HBM-accessible tensors.  
- **mesh (meta)**: distributed tensors, tracked via attributes for NCCL insertion.  

---

## 5. NVIDIA Mapping

- `tile.cp_async` → `cp.async` PTX instruction.  
- `tma.load/store` → Tensor Memory Accelerator ops.  
- `tile.mma` → WMMA (Ampere) / WGMMA (Hopper/Blackwell).  
- `tile.barrier` → `bar.sync` or CUDA named barrier.  
- Collectives → warp shuffle PTX (`shfl.sync`), cooperative group reductions.  

---

## 6. Example Transformation

**Schedule IR**
```mlir
schedule.tile %y {block=[128,128], warps=8, stages=3, vector=8}
```

**Tile IR**
```mlir
%smA = tile.shared.alloc bf16 [128,64] {swizzle="xor"}
%smB = tile.shared.alloc bf16 [64,128] {swizzle="xor"}

%a = tile.load %A[%i,%k] {vector=8}
%b = tile.load %B[%k,%j] {vector=8}

tile.cp_async %smA <- %a
tile.cp_async %smB <- %b
tile.barrier

%c = tile.mma %smA, %smB : bf16 -> f32

tile.store %C[%i,%j], %c
```

---

## 7. Diagnostics

- **Bounds checks**: illegal tile indices.  
- **Bank conflicts**: linter for shared memory swizzles.  
- **Register pressure**: static estimator with warnings.  
- **Pipeline misuse**: async loads not paired with barriers.  

---

## 8. Implementation Notes

- Implemented as MLIR `tile` dialect.  
- SSA values represent registers or shared memory slices.  
- Attributes carry autotuner-chosen schedules (BM/BN/BK, warps, vector width).  
- Maps directly to PTX / CUDA Tile IR.  
- Compatible with AMD/Intel via lowering (XDLops / DPAS).  

---

## 9. For Programmers: Why Tile IR Matters

Tile IR is the **closest IR to what runs on the GPU**. It makes explicit:  
- How tensors are loaded from global memory into shared memory (`tile.load`, `cp_async`).  
- How shared memory tiles are used in matrix multiplies (`tile.mma`).  
- Where barriers (`tile.barrier`) are placed to synchronize groups.  
- How warps cooperate with shuffle/reduce primitives.  

Programmers use Tile IR to:  
- Understand how their kernels map onto hardware memory tiers.  
- Debug performance issues like shared memory bank conflicts.  
- Verify that the compiler generated the expected **pipelines and double-buffering**.  

---

## 10. Inspecting Tile IR

```python
gemm_kernel.tile_ir   # or gemm_kernel.lowering_artifacts()
```

Example output:  
```mlir
%smA = tile.shared.alloc bf16 [128,64] {swizzle="xor"}
%smB = tile.shared.alloc bf16 [64,128] {swizzle="xor"}

%a = tile.load %A[%i,%k] {vector=8}
%b = tile.load %B[%k,%j] {vector=8}

tile.cp_async %smA <- %a
tile.cp_async %smB <- %b
tile.barrier

%c = tile.mma %smA, %smB : bf16 -> f32

tile.store %C[%i,%j], %c
```

This shows the **explicit tile-level kernel** the compiler will lower to PTX/CTIR.  

---

## 11. Common Errors at Tile IR Stage

- **Bank conflicts**:  
  *“Shared memory access pattern leads to 4-way bank conflict.”*  

- **Register pressure**:  
  *“Tile config leads to 280 registers/thread (exceeds arch limit 255).”*  

- **Barrier misuse**:  
  *“cp.async scheduled without matching barrier at loop end.”*  

- **Vectorization issues**:  
  *“Requested vector=16 not supported for dtype bf16.”*  

---

## 12. Programmer Workflow Example

1. You write (`@kernel.autotune` is a design-lineage sketch — the canonical
   autotune surface is the top-level `tessera.autotune(op, shapes=..., ...)` function;
   `tessera.kernel` has no `.autotune` attribute):  
   ```python
   @kernel.autotune(space=dict(BM=[128],BN=[128],BK=[64],stages=[3]))
   def gemm(A,B,C): return gemm_tile(A,B,C)
   ```

2. Inspect Tile IR:  
   ```python
   gemm.tile_ir
   ```
   → Shows explicit shared memory allocation, cp.async prefetch, mma.  

3. You notice:  
   - Bank conflict warnings on shared memory.  
   - Register pressure hints.  

4. You fix:  
   - Change swizzle layout: `tshared.alloc(..., swizzle="xor")`.  
   - Reduce vector width.  

5. Re-inspect Tile IR: warnings cleared.  

This workflow shows how **Tile IR inspection helps tune performance-critical kernels**.  

---

## 13. Summary for Programmers

- Tile IR is the **last IR you’ll inspect before actual PTX/CTIR**.  
- Use **`fn.tile_ir`** or `fn.lowering_artifacts()` for current Tile IR inspection.  
- Watch for **bank conflict, register pressure, barrier errors**.  
- Tile IR connects directly to the CUDA Tile IR / PTX you’ll see at the Target IR stage.  
