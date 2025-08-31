# Tessera Programming Guide  
## Chapter 11: Conclusion & Putting It All Together (Updated)

This guide has introduced the Tessera programming model: a **tile-first, distributed, and numerics-aware framework** for programming modern accelerators. Tessera bridges the gap between high-level productivity and low-level performance, scaling seamlessly from a single GPU to NVL72-scale systems.

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
- ✅ Stage data through **shared memory + cp.async**.  
- ✅ Declare **numerics policies** in tensor types.  
- ✅ Use **safe primitives** (`softmax_safe`, `layernorm_safe`).  
- ✅ Shard tensors with **ShardSpec** and distributions.  
- ✅ Apply **region privileges** (`read`, `write`, `reduce`) to enforce safety.  
- ✅ Scale with **index launches** across mesh axes.  
- ✅ Capture CUDA Graphs for low-overhead NVL72 runs.  
- ✅ Use **Mapper API** for fine-tuned placement and portability.  

---

### 11.3 NVL72 in Context

NVL72 exemplifies Tessera’s goals:

- Treats 72 GPUs as a single programming mesh.  
- Collectives map to NCCL/SHARP across NVSwitch.  
- Mixed precision (FP4/FP6/FP8 + FP32 accum) drives performance for LLM training.  
- Autodiff and collectives scale automatically across dp/tp/pp meshes.  
- Mapper API ensures locality-aware scheduling.  

The same Tessera code that runs on a single GPU scales to a full NVL72 supernode with no code changes.

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

