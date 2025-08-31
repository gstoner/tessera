# Tessera Compiler — Target IR Design
*Scope:* Tile IR → Target IR → Binary (PTX, CUDA Tile IR, LLVM, ROCm, oneAPI)  
*Status:* Draft v0.2 (with programmer context)

---

## 0. Goals
- Provide backend-specific lowering for accelerators.  
- Support NVIDIA PTX and CUDA Tile IR, AMD ROCm/XDLops, Intel oneAPI/DPAS.  
- Ensure deterministic, reproducible codegen across vendors.  
- Preserve **numerics policies, distributions, and collectives** into final kernels.  
- Integrate with **autotuning** and artifact caching.

---

## 1. Position in Compiler Pipeline

```text
Graph IR → Schedule IR → Tile IR → Target IR → Binary/Runtime
```

Target IR is the final compiler stage before code generation and binary packaging.

---

## 2. Target IR Dialects

**NVIDIA**
- **PTX**: mature fallback backend.  
- **CUDA Tile IR (CTIR)**: preferred for Hopper/Blackwell, includes cp.async, TMA, WGMMA.  

**AMD ROCm**
- Lowers to LLVM with **XDLops** intrinsics.  
- RCCL for collectives.  

**Intel oneAPI**
- Lowers to LLVM/SPIR-V with **DPAS** intrinsics.  
- oneCCL for collectives.  

---

## 3. Interface Contracts

- All Tile IR ops must lower to supported target ops.  
- Numerics policies map to HW-supported modes (FP4/FP6/FP8 scaling, FP32 accum).  
- Collectives map to NCCL, RCCL, or oneCCL.  
- Domains/Distributions → launch dimensions + collective ops.  

---

## 4. Example: NVIDIA PTX Lowering

**Tile IR**
```mlir
%a = tile.load %A[%i,%k] {vector=8}
%b = tile.load %B[%k,%j] {vector=8}
%c = tile.mma %a, %b : bf16 -> f32
tile.store %C[%i,%j], %c
```

**PTX**
```ptx
ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%r16-%r19}, [%rdA];
ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r20-%r23}, [%rdB];
mma.sync.aligned.m16n8k16.row.col.f16.f16.f32 {%f0-%f7}, {%r16-%r19}, {%r20-%r23}, {%f0-%f7};
st.global.f32 [%rdC], {%f0-%f7};
```

---

## 5. Example: NVIDIA CUDA Tile IR (CTIR)

**Tile IR**
```mlir
tile.cp_async %smA <- %A
tile.barrier
tile.mma %smA, %smB : fp8 -> fp32
```

**CUDA Tile IR**
```ctir
cp.async.ca.shared.global [smA], [A], 128;
barrier.sync;
wgmma.mma_async.aligned.m64n128k32.f32.fp8.fp8.fp32 ...
```

---

## 6. Autotuning Integration

- Schedule IR attributes (`block`, `warp`, `vector`) flow into Target IR kernels.  
- Autotune caches store **final target binaries per arch**.  
- Fatbin packaging bundles GIR + SchedIR + TileIR + TargetIR + tuned binaries.

---

## 7. Collectives in Target IR

- Lower to runtime calls:  
  - NVIDIA → NCCL kernels.  
  - AMD → RCCL.  
  - Intel → oneCCL.  

- Collective overlap preserved with CUDA Graphs or Level Zero Graphs.

---

## 8. Debugging & Inspection

Programmers can inspect target IR:

```python
print(kernel.inspect_ir("target"))
```

Outputs:
- PTX assembly.  
- CUDA Tile IR (for Blackwell).  
- LLVM IR (AMD, Intel).  

**Perf Hints**
- Register usage, shared memory usage.  
- Occupancy estimates.  
- Bank conflict reports.

---

## 9. Implementation Notes

- Implement as MLIR backends: `ptx`, `cuda_tile`, `llvm`.  
- Integrate with vendor toolchains: NVCC, ROCm, oneAPI.  
- Artifact registry: stable hashes of IR → reproducible binaries.  
- Deterministic replay with fixed seeds + ordered reductions.

---

## 10. For Programmers: Why Target IR Matters

Target IR is the **last IR stage before binary code**. It shows you:  
- Exactly what instructions (PTX, CUDA Tile IR, LLVM IR) your kernel was lowered to.  
- How **numerics policies** (FP4/FP6/FP8 accumulations) mapped to hardware intrinsics.  
- How **collectives** were implemented (NCCL, RCCL, oneCCL).  
- Whether the kernel is **portable** across NVIDIA, AMD, Intel backends.  

You don’t usually need to modify Target IR, but you can **inspect it** to:  
- Verify that FP8 ops are accumulating in FP32.  
- Debug unexpected performance by seeing the exact PTX/TMA/WGMMA ops.  
- Check if the compiler used the right backend intrinsics.

---

## 11. Inspecting Target IR

```python
gemm_kernel.inspect_ir("target")
```

On NVIDIA (Blackwell) this might show:  
```ptx
cp.async.ca.shared.global [smA], [A], 128;
barrier.sync;
wgmma.mma_async.aligned.m64n128k32.f32.fp8.fp8.fp32 ...
st.global.f32 [%rdC], {%f0-%f7};
```

On AMD this might show LLVM with XDLops intrinsics:  
```llvm
%a = call <16 x i32> @llvm.amdgcn.mfma.f32.16x16x16.fp16(...)
```

On Intel this might show DPAS intrinsics:  
```llvm
%acc = call <16 x i32> @llvm.intel.dpas(...)
```

---

## 12. Common Errors at Target IR Stage

- **Unsupported intrinsics**:  
  *“WGMMA not available on sm80 arch; use WMMA instead.”*  

- **Numerics policy mismatch**:  
  *“fp8_e4m3 matmul lowered without FP32 accumulation — check dtype policy.”*  

- **Backend feature gaps**:  
  *“DPAS lowering not implemented for this dtype.”*  

- **Collective lowering error**:  
  *“NCCL allreduce requires uniform participation; rank 3 missing.”*  

---

## 13. Programmer Workflow Example

1. You write:  
   ```python
   @jit
   def step(X: Tensor["B","D", fp8_e4m3 @accum(fp32)],
            W: Tensor["D","K", fp8_e4m3 @accum(fp32)]):
       return gemm(X,W)
   ```

2. Inspect Target IR:  
   ```python
   step.inspect_ir("target")
   ```
   → Shows `wgmma.mma_async` with `fp8.fp8.fp32` accumulate.  

3. You notice:  
   - Accumulation was done in FP16 instead of FP32.  

4. You fix:  
   - Add explicit `@accum(fp32)` policy in type annotation.  

5. Re-inspect Target IR: accumulation now in FP32.  

This workflow ensures **numerics and backend mappings are correct** before deployment.

---

## 14. Summary for Programmers

- **Target IR** is the **final IR before GPU/accelerator execution**.  
- Use **`inspect_ir("target")`** to see PTX, CTIR, LLVM, or SPIR-V output.  
- It’s useful for debugging:  
  - Intrinsics (WGMMA/WMMA, XDLops, DPAS).  
  - Numerics policies (fp8→fp32 accum).  
  - Collective lowering (NCCL, RCCL, oneCCL).  
- Most users won’t touch Target IR, but **power users can validate hardware mappings** and **diagnose perf issues**.  
