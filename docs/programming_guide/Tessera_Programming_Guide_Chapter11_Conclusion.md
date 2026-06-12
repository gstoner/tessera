---
status: Tutorial
classification: Tutorial
last_updated: 2026-06-11
---

> **Phase status note (updated 2026-06-11):** Phases 1–7 are complete and Phase 8 (Apple M-Series CPU via Accelerate, GPU via Metal/MPS/MPSGraph/custom MSL) is operational — on Apple Silicon this is the primary single-node execution path. Autodiff (forward/reverse transforms + activation checkpointing), ZeRO-2 optimizer sharding, the Bayesian autotuner, and the runtime Python wrapper (`tessera.runtime.TesseraRuntime`) are **shipped**. Genuinely still planned: **multi-GPU / multi-rank** execution of distributed collectives (NCCL/RCCL), `Cyclic` distribution lowering, and **NVL72** rack-scale execution (single-device collectives run over in-process mock ranks today). Canonical API names: `docs/CANONICAL_API.md`; phase table: root `CLAUDE.md`.


# Tessera Programming Guide  
## Chapter 11: Conclusion & Putting It All Together (Updated)

This guide has introduced the Tessera programming model: a **tile-first, distributed, and numerics-aware framework** for programming modern accelerators. Tessera bridges the gap between high-level productivity and low-level performance. Today it **executes** on x86, Apple M-Series CPU/GPU, and a CPU MLIR→LLVM JIT lane; the design scales toward multi-GPU and NVL72-scale systems as those execution paths land (Phase 4 / Phase G–I).

---

### 11.1 Tessera Workflow Recap

1. **Modeling**: Write functions in Pythonic Tessera with autodiff built in.  
2. **Tiling**: Kernels are written in terms of tiles and groups, not threads.  
3. **Numerics**: Types encode precision (FP4, FP6, FP8, BF16, FP16, FP32) and accumulation policies.  
4. **Execution**: Use barriers, async pipelines, and index launches for distributed execution.  
5. **Memory**: Organize data across registers, shared, HBM, and NVLink/NVSwitch.  
6. **Domains & Distributions**: Define global iteration spaces and shard tensors declaratively.  
7. **Region Privileges**: Ensure safety and correctness in distributed execution.  
8. **Collectives**: Use high-level ops (`allreduce`, `reduce_scatter`, `all_gather`, `all_to_all`) without worrying about wiring.  
9. **Libraries & Primitives**: Build on GEMM, FFT, FlashAttention, SpMM, RNG—all autodiff-aware.  
10. **Portability**: Rely on the multi-level IR stack and Mapper API to adapt code across NVIDIA, AMD, Intel, and beyond.  

---

### 11.2 Programmer’s Checklist

- ✅ Use **tiles**, not threads.  
- ✅ Stage data through **shared memory + cp.async** (NVIDIA) or **Metal encode-sessions** (Apple GPU).  
- ✅ Declare **numerics policies** in tensor types (TF32 is a `math_mode`, not a dtype).  
- ✅ Use **safe primitives** (`softmax_safe`, `layernorm_safe`).  
- ✅ Shard tensors with **ShardSpec** and distributions (single-device today; multi-GPU Phase 4).  
- ✅ Apply **region privileges** (`read`, `write`, `reduce`) to enforce safety.  
- ✅ Scale with **index launches** across mesh axes.  
- 🔜 *Phase 4+*: CUDA Graph capture for low-overhead NVL72 runs and the **Mapper API** for placement/portability.  

---

### 11.3 NVL72 in Context (Phase 4 planned)

NVL72 exemplifies Tessera’s **design goals** for rack-scale execution — it is a
roadmap target, not an operational path today:

- Treats 72 GPUs as a single programming mesh.  
- Collectives map to NCCL/SHARP across NVSwitch.  
- Mixed precision (FP4/FP6/FP8 + FP32 accum) drives performance for LLM training.  
- Autodiff and collectives scale across dp/tp/pp meshes.  
- Mapper API ensures locality-aware scheduling.  

The intent is that the same Tessera code which runs on a single device scales to
a full NVL72 supernode with no code changes **once multi-GPU execution lands**;
today distributed collectives run over in-process mock ranks.

---

### 11.4 Looking Ahead

Tessera’s roadmap includes:  

- Full support for **AMD ROCm/XDLops** and **Intel oneAPI/DPAS** backends.  
- Richer **domain maps** for sparse/irregular workloads.  
- Advanced **mapper policies** (energy-aware, QoS, multi-job scheduling).  
- **Deployment bundles** for cloud, mobile, and edge inference.  

---

### 11.5 Final Thoughts

Tessera combines:  

- The productivity of Python and JAX.  
- The performance of CUDA and Triton.  
- The scalability of Chapel and Legion.  

By making tiles, numerics, and distributed semantics first-class, Tessera gives programmers a **single, unified model** for writing the next generation of large-scale applications.

Write once, run anywhere—from a laptop GPU to NVL72 to future exascale accelerators.

