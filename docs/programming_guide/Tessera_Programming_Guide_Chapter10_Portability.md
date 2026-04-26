---
status: Tutorial
classification: Tutorial
last_updated: 2026-04-26
---

> **Phase status note:** Unless this document explicitly says otherwise, distributed collectives (NCCL/RCCL), TPU StableHLO, Cyclic distribution, autodiff transforms, activation checkpointing, ZeRO sharding, Bayesian autotuning, the runtime Python wrapper, production deployment, and NVL72 execution are Phase 4-6 planned as defined in `docs/README.md`. Current Phase 1-3 API names are defined in `docs/CANONICAL_API.md`.


# Tessera Programming Guide  
## Chapter 10: Portability (Updated)

Tessera is designed for **performance portability**. The current implemented path covers x86 lowering and supported NVIDIA SM_90+ GPU lowering. Future extensions target distributed NVIDIA systems, AMD, Intel, and other accelerators. Portability is achieved through a **multi-level IR stack**, target profiles, domains/distributions, and planned mapper/runtime policy APIs.

---

### 10.1 Multi-Level IR Stack

Tessera lowers code through multiple IR layers:

- **Graph IR**: high-level autodiff, algebra, and effect representation.  
- **Schedule IR**: tiling, fusion, pipelining, layout transforms.  
- **Tile IR**: explicit tile loads/stores, shared memory, mma intrinsics.  
- **Target IR**: backend-specific lowering (PTX, CUDA Tile IR, LLVM).  

This design ensures that high-level programs remain portable, while low-level optimizations adapt per-architecture.

---

### 10.2 NVIDIA GPUs

On NVIDIA accelerators (Ampere, Hopper, Blackwell):

- **PTX backend**: traditional lowering path.  
- **CUDA Tile IR backend**: preferred on Hopper/Blackwell.  
- **WMMA/WGMMA intrinsics**: tensor core operations.  
- **cp.async / TMA**: async memory movement.  
- **NCCL**: collectives across NVLink/NVSwitch.  

#### Example: Inspecting IR
```python
# Current API — emits Graph IR as MLIR text
print(gemm_fn.graph_ir.to_mlir())

# Note: tile/target IR inspection is planned for Phase 4+
```

---

### 10.3 AMD GPUs (Future Work)

- **ROCm + XDLops**: target for tensor core equivalents.  
- **rccl**: collective backend.  
- **Unified memory (HMM)** support.  

---

### 10.4 Intel GPUs (Future Work)

- **oneAPI Level Zero + DPAS intrinsics**.  
- **oneCCL**: collective backend.  
- **SYCL-based interop** for kernel launches.  

---

### 10.5 Future Mapper API for Portability

Mapper APIs are Phase 4 planned. Future examples should be marked clearly:

```python
# Phase 4 planned sketch
# mapper.place(region, mesh)
# mapper.choose_collective(kind, size)
# mapper.choose_variant(op, arch)
```

- Intended to guide placement on systems such as NVL72.
- Intended to choose collectives and backend variants.
- Not current Phase 1-3 public API.

---

### 10.6 Domains and Distributions in Portability

Domains and distributions are portable constructs:

```python
D = tessera.domain.Rect((B, S, Dm))
dist = tessera.dist.Block(mesh_axes=("dp","tp"))
X = tessera.array.from_domain(D, dtype="bf16", distribution=dist)
```

- On current Phase 1-3 paths: represented as shard metadata and mock/test behavior.
- Phase 4 planned: NVIDIA collectives lower to NCCL/RCCL-backed runtime paths.
- Future vendor paths: AMD and Intel collective backends are planned, not current.

The same Tessera program runs across vendors.

---

### 10.7 Index Launch Portability

`index_launch` distributes kernels across mesh partitions:

```python
tessera.index_launch(axis="tp")(gemm_tile)(A.parts("tp"), B.parts("tp"), C.parts("tp"))
```

- Current tests use shard lists and mock/sequential dispatch.
- Phase 4 planned distributed runtime maps collectives to vendor backends.

---

### 10.8 NVL72 as a Portability Case Study

NVL72 demonstrates Tessera’s philosophy:

- Phase 4 planned distributed placement treats a 72-GPU NVSwitch domain as a single logical mesh.
- Phase 4 planned mapper policy co-locates tensor-parallel ranks on NVSwitch groups.
- Phase 4 planned collectives map to NCCL with SHARP reductions where available.
- CUDA Graph capture for repeated training steps is planned future runtime work.

The same code can run on **smaller NVIDIA clusters** or **future AMD/Intel systems** with no changes.

---

### 10.9 Summary

- Tessera achieves portability through **multi-level IR** and **Mapper API hooks**.  
- NVIDIA single-node compiler support exists for the documented x86 and SM_90+ GPU paths.
- NCCL-backed distributed execution is Phase 4 planned.
- AMD and Intel support is planned (ROCm/XDLops/RCCL, oneAPI/DPAS/oneCCL).
- Domains, distributions, and index launches are portable abstractions.  
- NVL72 illustrates how Tessera adapts to extreme-scale NVIDIA systems.  

Programmers write once and run anywhere, with the compiler and runtime adapting to each backend.
