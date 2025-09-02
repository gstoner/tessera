

# Tessera Kernel Compilation Stages (Part 1: Concepts & Stage Overview)

This document introduces the compilation stages of the Tessera programming model, inspired by PyTorch’s “Triton Kernel Compilation Stages.”

## Stage 0: Source
Tessera DSL / Python API kernels — declarative tile-based compute with explicit memory ops.

## Stage 1: Frontend Typing & Verification
Parses and type-checks, ensuring shape correctness, dtype policies, and safety (barriers, masks). Produces **TIR‑H** (Tessera High IR).

## Stage 2: Canonicalization
Normalizes loops, propagates masks, desugars epilogues. Prepares IR for optimization.

## Stage 3: Scheduling & Autotuning
Injects tunable attributes (`BM`, `BN`, `BK`, stages, swizzle). Searchers (Hyperband, Bayesian) pick optimal configs with constraints. Produces scheduled **TIR‑H**.

## Stage 4: Lowering to Target IR (TIR‑T)
Converts high-level ops into explicit target dialect:
- TMA descriptors
- mbarrier ops
- Lane→fragment maps for WGMMA
- Epilogues into fused tensor ops

---
# Tessera Kernel Compilation Stages (Part 2: Worked Example & Dumps)

This section provides a worked GEMM/FlashAttention example and shows artifacts at each stage.

## Stage 5: Backend Lowering
- NVIDIA path: TIR‑T → NVVM/TileIR → PTX → cubin
- x86/AMX: TIR‑T → LLVM IR with AMX/VNNI intrinsics
- AMD/ROCm: TIR‑T → ROCDL/SPIR‑V

## Stage 6: Codegen & Linking
Produces binaries (cubin/ELF/shared lib) and reflection JSON (shapes, regs, smem, features).

## Stage 7: Runtime Integration
Runtime sets up descriptors, mbarriers, streams, launches grid/block. Collects metrics & profiles.

## Stage 8: Introspection & Debugging
Dump IR at each stage with:
```bash
tessera.compile(kernel, dump=["tirh","tirt","ptx","cubin"])
```

---

## End-to-End Example
**FlashAttention (bf16→fp32 with bias+SiLU):**
1. DSL: double-buffered TMA loads, WGMMA, fused epilogue
2. TIR‑H: normalized loops, annotated tunables
3. Scheduled: autotuned BM/BN/BK, swizzle
4. TIR‑T: explicit descriptors & fragment maps
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

