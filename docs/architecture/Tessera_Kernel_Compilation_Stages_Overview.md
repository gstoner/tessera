---
status: Informative
classification: Informative
authority: Companion overview; defers to docs/spec/COMPILER_REFERENCE.md
last_updated: 2026-10-24
---

> **Current-state note (2026-05-20):** This is historical architecture guidance. Phase labels below are design lineage, not current support claims. For implementation status, use `docs/spec/COMPILER_REFERENCE.md`, `docs/audit/generated/support_table.md`, `docs/audit/generated/e2e_op_coverage.md`, and `docs/spec/VALIDATION_SPINE.md`. For what executes on which device, use `docs/audit/generated/runtime_execution_matrix.md` (refreshed 2026-10-24).

# Tessera Kernel Compilation Stages (Part 1: Concepts & Stage Overview)

This document introduces the compilation stages of the Tessera programming model, inspired by PyTorch’s “Triton Kernel Compilation Stages.”

## Stage 0: Source
Tessera DSL / Python API kernels — declarative tile-based compute with explicit memory ops.

## Stage 1: Frontend Typing & Verification
Parses and verifies Python API annotations, shape constraints, dtype policies, and effects. Produces **Graph IR**.

## Stage 2: Canonicalization
Normalizes loops, propagates masks, desugars epilogues. Prepares IR for optimization.

## Stage 3: Scheduling & Autotuning
Injects tunable attributes (`tile_q`, `tile_kv`, stages, swizzle) and placement structure. The Bayesian autotuner (`python/tessera/compiler/autotune_v2.py`, SQLite warm-start cache) may choose optimized configs with constraints. Produces **Schedule IR**.

## Stage 4: Lowering to Tile IR and Target IR
Converts high-level ops into explicit target dialect:
- TMA descriptors
- mbarrier ops
- Lane→fragment maps for WGMMA
- Epilogues into fused tensor ops

---
# Tessera Kernel Compilation Stages (Part 2: Worked Example & Dumps)

This section provides a worked GEMM/FlashAttention example and shows artifacts at each stage.

## Stage 5: Backend Lowering
- NVIDIA path: Graph IR → Schedule IR → Tile IR → `tessera_nvidia` Target IR → NVVM/PTX
  (exact per-arch `tessera-lower-to-nvidia-sm{90,100,120}`; sm_120 is the only
  arch with fleet hardware)
- x86: Graph IR → Schedule IR → Tile IR → `TileToX86Pass` → C ABI calls into the
  AVX-512 backend functions, each marked by a `tessera_x86.abi_call` Target op
  (AMX is a retired target)
- AMD/ROCm: verified hardware-free `tessera_rocm` MFMA/WMMA/async-copy/wait
  artifacts are implemented; native execution is proven per device (gfx1151 and
  gfx1201 separately) and remains family-specific
- Apple: `tessera_apple.cpu.*` (Accelerate) and `tessera_apple.gpu.*` (MPS / MSL /
  simdgroup / Metal 4 `matmul2d`) Target IR

## Stage 6: Codegen & Linking
Produces binaries and reflection JSON (shapes, regs, smem, features). Today NVIDIA
kernels are PTX that the driver JIT-loads (stamped with the driver's PTX ISA, not
the toolkit's), ROCm kernels are precompiled hsaco from `gpu-module-to-binary`,
Apple uses runtime-compiled or AOT `.metallib` MSL, and x86 links shared libraries.

## Stage 7: Runtime Integration
Runtime sets up descriptors, mbarriers, streams, launches grid/block. Collects metrics & profiles.

## Stage 8: Introspection & Debugging
Current JIT inspection exposes all artifact layers:
```python
fn.graph_ir.to_mlir()
fn.schedule_ir
fn.tile_ir
fn.target_ir
fn.lowering_artifacts()
```
Graph, Schedule, Tile, and CPU/x86, NVIDIA, Apple and ROCm Target IR artifacts
are emitted from verified Python object models before they are presented as
textual IR. `fn.explain_lowering()` and `fn.last_fallback_reason` report whether
a call took a compiled lane or fell back.

---

## End-to-End Example
**FlashAttention (bf16→fp32 with bias+SiLU):**
1. DSL: double-buffered TMA loads, WGMMA, fused epilogue
2. Graph IR: normalized mathematical ops and attributes
3. Scheduled: autotuned BM/BN/BK, swizzle
4. Tile IR / Target IR: explicit descriptors & fragment maps
5. PTX: cp.async + wgmma ops, epilogue math
6. Cubin: final binary
7. Runtime: configured launch with reflection metadata

---

## Appendix: Mapping (Triton vs Tessera)
| Triton | Tessera |
|--------|---------|
| `dot` | `tile.mma` / `tile.gemm` (Tile IR) |
| `cp.async` | `tile.async_copy` (Tile IR) → `tessera_nvidia.tma_async_copy` (Target IR) |
| `bar.sync` | `tile.mbarrier.*` (Tile IR) → `tessera_nvidia.mbarrier` (Target IR) |
| `mma` | `tessera_nvidia.{mma_sync,wgmma,tcgen05_mma}` / `tessera_rocm.{wmma,mfma}` (Target IR) |
