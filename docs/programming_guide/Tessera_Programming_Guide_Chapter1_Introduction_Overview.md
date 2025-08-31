# Tessera Programming Guide  
## Chapter 1: Introduction & Overview (Updated)

Tessera is a new programming model for modern accelerators. It unifies **modeling, training, kernels, and distributed execution** in one stack, designed to scale from single-GPU research code to NVL72 rack-scale deployments.

Unlike thread- and block-centric models (e.g., CUDA), Tessera is **tile-first**: programmers think in terms of tiles, groups, and meshes. This abstraction, combined with an expressive type system and a multi-level IR, makes Tessera code both **portable** and **high-performance**.

---

### 1.1 Why Tessera?

- **One stack for research → production**: no rewrites between Python prototyping and GPU kernel development.  
- **Tiles, not threads**: simpler reasoning about performance, memory, and synchronization.  
- **Numerics as types**: stability and performance with FP4/FP6/FP8/BF16/FP16/FP32, all declared in the type system.  
- **Autodiff built-in**: effect-aware, collective-aware, with custom VJP/JVP support.  
- **Distributed by construction**: tensors are sharded/replicated/reduced across device meshes.  
- **Portability**: NVIDIA PTX/Tile IR today, AMD/Intel future.  
- **Flagship support for NVL72**: treat an entire 72-GPU NVSwitch rack as a single programming domain.  

---

### 1.2 Key Capabilities at a Glance

| Capability | Tessera Feature | Example |
|------------|----------------|---------|
| **Execution model** | Tile → Group → Mesh hierarchy | `tile.linear_id()` |
| **IR stack** | Graph IR → Schedule IR → Tile IR → Target IR | `inspect_ir("tile")` |
| **Numerics** | FP4/FP6/FP8/BF16/FP16/FP32 policies | `Tensor["B","D", fp8_e4m3 @accum(fp32)]` |
| **Autodiff** | Forward, reverse, custom VJP/JVP, effect-aware | `grad(loss_fn)(W)` |
| **Distributed tensors** | `ShardSpec`, domains, distributions | `ShardSpec(partition=("row",), mesh_axes=("tp",))` |
| **Region privileges** | Read/Write/Reduce semantics for safe scheduling | `def step(W: Region[read], Y: Region[write])` |
| **Collectives** | Declarative DP/TP/PP/EP parallelism | `all_gather(y_local, axis="tp")` |
| **Index launches** | Scale kernels across shards | `tessera.index_launch(axis="tp")(gemm_tile)` |
| **Portability** | NVIDIA PTX + CUDA Tile IR | `tile.mma → wgmma` |
| **Deployment** | CUDA Graphs, AOT bundles, NVL72 scale | `@jit(capture_graph=True)` |

---

### 1.3 What’s New in This Specification

Since Tessera Programming Model V1, this spec now includes:  

- **Domains & Distributions**: declarative iteration spaces and tensor layouts.  
- **Region Privileges**: explicit read/write/reduce annotations for safety and overlap.  
- **Index Launches**: scalable task distribution over mesh partitions.  
- **NVL72 Appendix**: best practices for programming NVIDIA’s 72-GPU supernode.  

---

### 1.4 Teaser Example

Here is a short example showing Tessera’s programming model in action on NVL72:  

```python
from tessera import dist

# Create a 72-GPU mesh: 4 (dp) × 9 (tp) × 2 (pp)
mesh = dist.mesh(
    devices=[f"cuda:{i}" for i in range(72)],
    axes=("dp","tp","pp"),
    shape=(4,9,2)
)

# Shard weights across tensor-parallel axis
W = dist.tensor(
    shape=(8192, 8192),
    layout=dist.ShardSpec(partition=("col",), mesh_axes=("tp",)),
    mesh=mesh,
    dtype="fp8_e4m3"
)

# Compute a distributed matmul
Y = gemm(X, W)   # compiler inserts all_gather/reduce_scatter on TP shards
```

This shows how **mesh construction**, **ShardSpec distribution**, and **numerics-as-types** come together in Tessera.

---

### 1.5 Structure of This Guide

- **Ch.2 Programming Model**: tiles, groups, meshes, and distributed concepts.  
- **Ch.3 Memory Model**: registers, shared, HBM, NVLink/NVSwitch.  
- **Ch.4 Execution Model**: pipelines, async copies, barriers.  
- **Ch.5 Error Handling**: safety, diagnostics, debugging.  
- **Ch.6 Numerics Model**: FP4/6/8/BF16/FP16/FP32, safe ops, rounding.  
- **Ch.7 Autodiff**: built-in, effect-aware, custom VJP/JVP.  
- **Ch.8 Layouts & Data Movement**: explicit layouts, async pipelines.  
- **Ch.9 Libraries & Primitives**: dense, sparse, spectral, RNG.  
- **Ch.10 Portability**: NVIDIA PTX/Tile IR, NCCL, CUTLASS interop.  
- **Ch.11 Conclusion**: programming workflow, deployment, best practices.  
- **Appendix A (NVL72)**: full guide to programming NVIDIA’s NVL72.  

---

### 1.6 Summary

Tessera provides:  
- **A unified programming model** across kernels, models, and distributed execution.  
- **Tiles, domains, and distributions** as first-class concepts.  
- **Safe parallelism** via region privileges and effect-aware autodiff.  
- **Performance portability** from single-GPU to NVL72.  

With Tessera, programmers write once and scale seamlessly, leveraging the full power of modern accelerators.

