---
status: Tutorial
classification: Tutorial
last_updated: 2026-04-26
---

> **Phase status note:** Unless this document explicitly says otherwise, distributed collectives (NCCL/RCCL), TPU StableHLO, Cyclic distribution, autodiff transforms, activation checkpointing, ZeRO sharding, Bayesian autotuning, the runtime Python wrapper, production deployment, and NVL72 execution are Phase 4-6 planned as defined in `docs/README.md`. Current Phase 1-3 API names are defined in `docs/CANONICAL_API.md`.


# Tessera Programming Guide  
## Chapter 1: Introduction & Overview (Updated)

Tessera is a new programming model for modern accelerators. The current implemented stack covers the Python frontend, x86 lowering, and supported NVIDIA SM_90+ lowering paths. Distributed training, extended autodiff, and NVL72 rack-scale execution are Phase 4-5 planned.

Unlike thread- and block-centric models (e.g., CUDA), Tessera is **tile-first**: programmers think in terms of tiles, groups, and meshes. This abstraction, combined with an expressive type system and a multi-level IR, makes Tessera code both **portable** and **high-performance**.

---

### 1.1 Why Tessera?

- **One stack for research → production**: no rewrites between Python prototyping and GPU kernel development.  
- **Tiles, not threads**: simpler reasoning about performance, memory, and synchronization.  
- **Numerics as types**: stability and performance with FP4/FP6/FP8/BF16/FP16/FP32, all declared in the type system.  
- **Autodiff-aware design**: effect-aware and collective-aware transforms are Phase 5 planned.  
- **Distributed by construction**: domains/distributions exist today; production collectives are Phase 4 planned.  
- **Portability**: NVIDIA PTX/Tile IR today, AMD/Intel future.  
- **NVL72 design target**: treating a 72-GPU NVSwitch rack as a single programming domain is Phase 4 planned.  

---

### 1.2 Key Capabilities at a Glance

| Capability | Tessera Feature | Example |
|------------|----------------|---------|
| **Execution model** | Tile → Group → Mesh hierarchy | `tile.linear_id()` |
| **IR stack** | Graph IR → Schedule IR → Tile IR → Target IR | `fn.graph_ir.to_mlir()` |
| **Numerics** | FP4/FP6/FP8/BF16/FP16/FP32 policies | `Tensor["B","D", fp8_e4m3 @accum(fp32)]` |
| **Autodiff** | Forward/reverse transforms are Phase 5 planned | Phase 5 planned |
| **Distributed tensors** | `ShardSpec`, domains, distributions | `tessera.array.from_domain(...)` |
| **Region privileges** | Read/Write/Reduce semantics for safe scheduling | `def step(W: Region["read"], Y: Region["write"])` |
| **Collectives** | Declarative DP/TP/PP/EP parallelism | Phase 4 planned |
| **Index launches** | Scale kernels across shards | `tessera.index_launch(axis="tp")(gemm_tile)` |
| **Portability** | NVIDIA PTX + CUDA Tile IR | `tile.mma → wgmma` |
| **Deployment** | Runtime ABI wiring, CUDA Graphs, AOT bundles, NVL72 scale | Phase 6 planned |

---

### 1.3 What’s New in This Specification

Since Tessera Programming Model V1, this spec now includes:  

- **Domains & Distributions**: declarative iteration spaces and tensor layouts.  
- **Region Privileges**: explicit read/write/reduce annotations for safety and overlap.  
- **Index Launches**: scalable task distribution over mesh partitions.  
- **NVL72 Appendix**: best practices for programming NVIDIA’s 72-GPU supernode.  

---

### 1.4 Teaser Example

Here is a short current example showing domains, distributions, and a compiled operation:

```python
import tessera

D = tessera.domain.Rect((4, 128, 256))
dist = tessera.dist.Block(mesh_axes=("dp", "tp"))
X = tessera.array.from_domain(D, dtype="bf16", distribution=dist)

@tessera.jit
def block(A: tessera.Tensor["M", "K"], B: tessera.Tensor["K", "N"]):
    return tessera.ops.gemm(A, B)
```

Future NVL72 examples should be marked Phase 4 planned.

---

### 1.5 Structure of This Guide

- **Ch.2 Programming Model**: tiles, groups, meshes, and distributed concepts.  
- **Ch.3 Memory Model**: registers, shared, HBM, NVLink/NVSwitch.  
- **Ch.4 Execution Model**: pipelines, async copies, barriers.  
- **Ch.5 Kernel Programming**: `@tessera.kernel`, dtype annotations, `index_launch`, tile DSL, FA-4 attention ops.  
- **Ch.6 Numerics Model**: FP4/6/8/BF16/FP16/FP32, safe ops, rounding.  
- **Ch.7 Autodiff**: Phase 5 planned transforms and current effect contracts.  
- **Ch.8 Layouts & Data Movement**: explicit layouts, async pipelines.  
- **Ch.9 Libraries & Primitives**: dense, sparse, spectral, RNG.  
- **Ch.10 Portability**: NVIDIA PTX/Tile IR, NCCL, CUTLASS interop.  
- **Ch.11 Conclusion**: programming workflow, deployment, best practices.  
- **Appendix A (NVL72)**: future-facing guide for Phase 4 planned NVL72 programming.  

---

### 1.6 Summary

Tessera provides:  
- **A unified programming model** across kernels, models, and distributed execution.  
- **Tiles, domains, and distributions** as first-class concepts.  
- **Safe parallelism** via region privileges and effect tracking.  
- **Performance portability** from current single-node paths toward planned NVL72 support.  

With Tessera, programmers use one model across current compiler paths while the distributed/runtime roadmap fills in later phases.
