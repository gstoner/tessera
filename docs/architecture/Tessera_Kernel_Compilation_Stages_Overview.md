---
status: Informative
classification: Informative
authority: Companion overview; defers to docs/spec/COMPILER_REFERENCE.md
last_updated: 2026-04-26
---

> **Phase status note:** Unless this document explicitly says otherwise, distributed collectives (NCCL/RCCL), TPU StableHLO, Cyclic distribution, autodiff transforms, activation checkpointing, ZeRO sharding, Bayesian autotuning, the runtime Python wrapper, production deployment, and NVL72 execution are Phase 4-6 planned as defined in `docs/README.md`. Current Phase 1-3 API names are defined in `docs/CANONICAL_API.md`.

# Tessera Kernel Compilation Stages (Part 1: Concepts & Stage Overview)

This document introduces the compilation stages of the Tessera programming model, inspired by PyTorch’s “Triton Kernel Compilation Stages.”

## Stage 0: Source
Tessera DSL / Python API kernels — declarative tile-based compute with explicit memory ops.

## Stage 1: Frontend Typing & Verification
Parses and verifies Python API annotations, shape constraints, dtype policies, and effects. Produces **Graph IR**.

## Stage 2: Canonicalization
Normalizes loops, propagates masks, desugars epilogues. Prepares IR for optimization.

## Stage 3: Scheduling & Autotuning
Injects tunable attributes (`tile_q`, `tile_kv`, stages, swizzle) and placement structure. Phase 5 planned Bayesian autotuning may choose optimized configs with constraints. Produces **Schedule IR**.

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
- NVIDIA path: Graph IR → Schedule IR → Tile IR → Target IR → NVVM/PTX → cubin
- x86/AMX: Graph IR → Schedule IR → Target IR calls into AMX/AVX-512 backend functions
- AMD/ROCm: ROCm full MFMA coverage is Phase 6 planned

## Stage 6: Codegen & Linking
Produces binaries (cubin/ELF/shared lib) and reflection JSON (shapes, regs, smem, features).

## Stage 7: Runtime Integration
Runtime sets up descriptors, mbarriers, streams, launches grid/block. Collects metrics & profiles.

## Stage 8: Introspection & Debugging
Current inspection supports Graph IR:
```bash
fn.graph_ir.to_mlir()
```
Tile/Target IR inspection is Phase 4 planned.

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
| `dot` | `tessera_tile.gemm` |
| `cp.async` | `tessera_target.memcpy` |
| `bar.sync` | `tessera_target.mbarrier` |
| `mma` | `tessera_target.tensor_core` |
