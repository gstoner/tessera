# Tessera Compiler & Optimization Guide (v3)

## 1. Overview
This guide explains how Tessera lowers programs from Graph IR → Schedule IR → Tile IR → Target IR, and details compiler passes, optimizations, and PTX-level tuning strategies.

---

## 2. IR Lowering Pipeline

1. **Graph IR**
   - Operator algebra, autodiff, symbolic rewrites.
2. **Schedule IR**
   - Fusion, tiling, pipeline scheduling, autotuning hooks.
3. **Tile IR**
   - Blocks, warps, tensor core fragments, shared memory.
4. **Target IR**
   - LLVM IR → PTX (NVIDIA), LLVM → GCN (AMD ROCm), Level Zero (Intel).

---

## 3. Low-Level Pass Optimization

### Algebraic Simplification Passes
- Constant folding, dead code elimination.
- Fused ops: GEMM + bias + activation → tensor core kernel.

### Loop Transformations
- Loop interchange, unrolling, software pipelining.
- Mapping affine loops to warps/threads.

### Memory Passes
- Shared memory tiling & bank conflict avoidance.
- Prefetching with `cp.async` (Ampere/Hopper).
- Vectorized loads/stores: `ldmatrix`, `stmatrix`.

### Tensor Core Passes
- Pattern matching GEMMs → lowering to `mma.sync`.
- BF16/FP16 inputs with FP32 accumulation.

### Communication Passes
- Fusion of collectives (group all-reduce ops).
- Deterministic reduction ordering (tree vs ring).

### Autotuning Hooks
- Cost-model guided exploration of:
  - Tile sizes, unroll factors, pipeline depths.
- Persistent caches keyed by `(arch, dtype, shape)`.

---

## 4. MLIR Integration

Tessera adds custom dialects layered on MLIR:

- `tessera.graph` — operators, autodiff, symbolic algebra.
- `tessera.schedule` — tiling, fusion, pipelines.
- `tessera.tile` — thread/block mapping, tensor cores.

Lowering flow:

- `linalg` dialect for high-level ops (matmul, conv).
- → affine loops (loop optimizations).
- → `gpu` dialect (blocks, threads).
- → `nvvm` dialect (PTX intrinsics).

---

## 5. Example: Matmul Lowering to PTX

### MLIR (linalg dialect)
```mlir
%0 = linalg.matmul ins(%A, %B : memref<MxKxf32>, memref<KxNxf32>)
                   outs(%C : memref<MxNxf32>)
```

### MLIR (gpu dialect)
```mlir
gpu.launch blocks(%bx, %by) in (%B0, %B1)
           threads(%tx, %ty) in (%T0, %T1) {
  // Shared memory tiling + loop unrolling
}
```

### MLIR (nvvm dialect)
```mlir
%mma = nvvm.mma.sync {shape = m16n8k8, dtype = f16, acc = f32}
```

### PTX Output
```ptx
mma.sync.aligned.m16n8k8.row.col.f16.f16.f32 
  {f32reg0, f32reg1}, {f16reg0, f16reg1}, {f16reg2, f16reg3}, {f32reg0, f32reg1};
```

---

## 6. PTX-Level Optimizations

- **Warp shuffles**: `shfl.sync` for fast reductions.
- **Tensor core intrinsics**: `mma.sync`, `ldmatrix`.
- **Async copies**: `cp.async` for overlapped memory loads.
- **Inline PTX in Tessera**:
```python
op.inline_ptx("""
    { .reg .f32 r0, r1;
      shfl.sync.down.b32 r1, r0, 1, 32;
    }
""")
```

---

## 7. Summary
- Tessera’s compiler pipeline leverages MLIR for structured lowering.  
- Low-level passes optimize algebra, memory, tensor core usage, and communication.  
- PTX-level tuning provides escape hatches for maximum performance.
