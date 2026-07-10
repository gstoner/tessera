---
status: Tutorial
classification: Tutorial
last_updated: 2026-06-11
---

> **Phase status note (updated 2026-06-11):** Phases 1–7 are complete and Phase 8 (Apple M-Series CPU via Accelerate, GPU via Metal/MPS/MPSGraph/custom MSL) is operational — on Apple Silicon this is the primary single-node execution path. Autodiff (forward/reverse transforms + activation checkpointing), ZeRO-2 optimizer sharding, the Bayesian autotuner, and the runtime Python wrapper (`tessera.runtime.TesseraRuntime`) are **shipped**. Genuinely still planned: **multi-GPU / multi-rank** execution of distributed collectives (NCCL/RCCL), `Cyclic` distribution lowering, and **NVL72** rack-scale execution (single-device collectives run over in-process mock ranks today). Canonical API names: `docs/CANONICAL_API.md`; phase table: root `CLAUDE.md`.


# Tessera Programming Guide  
## Chapter 10: Portability (Updated)

Tessera is designed for **performance portability**. The **executable** paths today are x86 (AMX/AVX-512), Apple M-Series CPU (Accelerate) and GPU (Metal/MPS/MPSGraph/MSL), and a production CPU MLIR→LLVM JIT lane; NVIDIA SM_80+ and AMD ROCm **emit Target IR artifacts** today with hardware execution gated on Phase G/H. Future extensions target distributed/rack-scale NVIDIA systems, Intel, and other accelerators. Portability is achieved through a **multi-level IR stack**, target profiles, domains/distributions, and planned mapper/runtime policy APIs.

---

### 10.1 Multi-Level IR Stack

Tessera lowers code through multiple IR layers:

- **Graph IR**: high-level autodiff, algebra, and effect representation.  
- **Schedule IR**: tiling, fusion, pipelining, layout transforms.  
- **Tile IR**: explicit tile loads/stores, shared memory, mma intrinsics.  
- **Target IR**: backend-specific lowering (PTX, CUDA Tile IR, LLVM).  

This design ensures that high-level programs remain portable, while low-level optimizations adapt per-architecture.

---

### 10.2 NVIDIA GPUs (artifact today; hardware execution Phase G)

On NVIDIA accelerators (Ampere, Hopper, Blackwell), Tessera **emits Target IR
artifacts** (PTX / CUDA Tile IR) today; real-hardware execute-and-compare is
gated on Phase G hardware not present on the dev machine:

- **PTX backend**: traditional lowering path.  
- **CUDA Tile IR backend**: preferred on Hopper/Blackwell.  
- **WMMA/WGMMA intrinsics**: tensor core operations.  
- **cp.async / TMA**: async memory movement.  
- **NCCL**: collectives across NVLink/NVSwitch (multi-GPU execution Phase 4).  

#### Example: Inspecting IR
```python
# Each lowered layer is inspectable on the JIT'd function:
print(gemm_fn.graph_ir.to_mlir())
print(gemm_fn.schedule_ir.to_mlir())
print(gemm_fn.tile_ir.to_mlir())
print(gemm_fn.target_ir.to_mlir())
# The `tessera-mlir --mode=compile_artifact --symbol=NAME` CLI reads a
# JIT artifact statically, without launching tensors.
```

---

### 10.3 Apple M-Series (shipped today)

Apple Silicon is a **fully executable** single-node target — on this hardware
it is the primary path:

- `@tessera.jit(target="apple_cpu")` → Accelerate (`cblas_sgemm` + BNNS f16/bf16).  
- `@tessera.jit(target="apple_gpu")` → Metal via MPS, MPSGraph (Tier-1
  activations/norms), and custom MSL kernels, with fused chains
  (`matmul→softmax[→matmul]`, `matmul→gelu`, `matmul→rmsnorm`) and
  descriptor-driven dispatch.  
- f32 / f16 native; bf16 via host upcast on the GPU lane.  

A single-layer LLaMA-style decoder runs end-to-end on the GPU lane validated
against numpy. See [`docs/apple_backend.md`](../apple_backend.md).

---

### 10.4 AMD GPUs (artifact today; hardware execution Phase H)

- **ROCm 7.2.4 / MFMA**: Target IR + MFMA shape tables emit today
  (gfx90a / 940 / 942 / 950 / 1100); hardware execute-and-compare is Phase H.  
- **rccl**: collective backend.  
- **Unified memory (HMM)** support.  

---

### 10.5 Intel GPUs (Future Work)

- **oneAPI Level Zero + DPAS intrinsics**.  
- **oneCCL**: collective backend.  
- **SYCL-based interop** for kernel launches.  

---

### 10.6 Future Mapper API for Portability

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

### 10.7 Domains and Distributions in Portability

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

### 10.8 Index Launch Portability

`index_launch` distributes kernels across mesh partitions:

```python
tessera.index_launch(axis="tp")(gemm_tile)(A.parts("tp"), B.parts("tp"), C.parts("tp"))
```

- Current tests use shard lists and mock/sequential dispatch.
- Phase 4 planned distributed runtime maps collectives to vendor backends.

---

### 10.9 NVL72 as a Portability Case Study

NVL72 demonstrates Tessera’s philosophy:

- Phase 4 planned distributed placement treats a 72-GPU NVSwitch domain as a single logical mesh.
- Phase 4 planned mapper policy co-locates tensor-parallel ranks on NVSwitch groups.
- Phase 4 planned collectives map to NCCL with SHARP reductions where available.
- CUDA Graph capture for repeated training steps is planned future runtime work.

The same code can run on **smaller NVIDIA clusters** or **future AMD/Intel systems** with no changes.

---

### 10.10 Summary

- Tessera achieves portability through **multi-level IR** and **Mapper API hooks**.  
- Executable today: x86, Apple CPU (Accelerate), Apple GPU (Metal/MPS/MPSGraph/MSL), CPU JIT.
- NVIDIA SM_80+ and AMD ROCm emit artifacts today; hardware execution is Phase G/H.
- NCCL-backed multi-GPU distributed execution is Phase 4 planned; Intel support is planned (oneAPI/DPAS/oneCCL).
- Domains, distributions, and index launches are portable abstractions.  
- NVL72 illustrates how Tessera adapts to extreme-scale NVIDIA systems.  

Programmers write once and run anywhere, with the compiler and runtime adapting to each backend.
