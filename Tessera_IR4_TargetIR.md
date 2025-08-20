# Tessera IR Layer 4 — Target IR (LLVM → PTX/NVVM, ROCDL)
*(CUDA-style programming guide companion; normative unless stated otherwise)*

---

## 1. Scope
Target IR is the **lowering boundary** to backend toolchains:
- **NVIDIA**: MLIR → NVVM → PTX → SASS
- **AMD**: MLIR → ROCDL → GCN
- (Optional) **SPIR-V/Metal** for portability

---

## 2. Interfaces & Dialects
- **gpu / nvvm / rocdl / llvm** dialects for codegen.
- **memref** lowered to address-space-qualified pointers.
- Collectives mapped to vendor libraries (NCCL/RCCL) via runtime calls.

---

## 3. Example: Matmul Tile to NVVM
```mlir
gpu.func @matmul_kernel(%A: memref<?xf16, 1>, %B: memref<?xf16, 1>, %C: memref<?xf16, 1>) kernel {
  %bx = gpu.block_id x
  %tx = gpu.thread_id x

  // (Indexing math elided) — load fragments
  %a = nvvm.ldmatrix %A[...] : !nvvm.matrix
  %b = nvvm.ldmatrix %B[...] : !nvvm.matrix

  // Tensor Core MMA
  %acc = nvvm.mma.sync %a, %b : <m16n16k16, f32>

  // Store
  llvm.store %acc, %C[%idx] : memref<?xf16, 1>
  gpu.return
}
```

## 4. Example: ROCDL Lowering
```mlir
rocdl.func @matmul_kernel(%A: memref<?xf16, 1>, %B: memref<?xf16, 1>, %C: memref<?xf16, 1>) {
  // wave-mma ops analogous to NVVM
  %acc = rocdl.mfma %A, %B : <m16n16k16, f32>
  rocdl.return
}
```

---

## 5. Runtime ABI
- Kernel entry uses C ABI with opaque state pointer for constants.
- Calling convention MUST be stable across versions.
- Determinism flags (e.g., stable reductions) travel via kernel params.

---

## 6. Collectives
- `tsched.collective` plans are lowered to runtime stubs:
  - **NVIDIA**: NCCL (ring/tree) with deterministic ordering
  - **AMD**: RCCL
- On-wire quantization (e.g., FP8) is applied if policy allows.

---

## 7. Validation
- Verify address spaces, barrier placement, and no UB (undefined behavior).
- PTX/GCN disassembly checks (optional) for fragment utilization.
