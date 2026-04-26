---
status: Tutorial
classification: Tutorial
last_updated: 2026-04-26
---

> **Phase status note:** Unless this document explicitly says otherwise, distributed collectives (NCCL/RCCL), TPU StableHLO, Cyclic distribution, autodiff transforms, activation checkpointing, ZeRO sharding, Bayesian autotuning, the runtime Python wrapper, production deployment, and NVL72 execution are Phase 4-6 planned as defined in `docs/README.md`. Current Phase 1-3 API names are defined in `docs/CANONICAL_API.md`.


# Tessera Programming Guide  
## Chapter 8: Layouts & Data Movement (Updated)

Efficient accelerator programming requires precise control of **data layout** and **movement**. Tessera makes layouts first-class and integrates them with **domains, distributions, and region privileges**. This enables portable, optimized memory access patterns across single-GPU kernels and multi-GPU systems like NVL72.

---

### 8.1 Layouts as First-Class Objects

Layouts define how tensors are organized in memory:

- **Row-major / Column-major**  
- **Blocked / Tiled**  
- **Swizzled (xor, Morton-order)**  
- **Interleaved**  

Example:
```python
sA = tshared.alloc[f16](BM, BK, swizzle="xor")
sB = tshared.alloc[f16](BK, BN, swizzle="xor")
```

Layouts are explicit so programmers can reason about **bank conflicts, vectorization, and tiling**.

---

### 8.2 Tile Loads and Stores

Data is moved between memory tiers explicitly:

```python
Qb = tile.load(Q, rows=T.m, cols=T.d, vector=T.vector, prefetch=2)
tile.store(C, index=(i,j), value=Qb)
```

- `tile.load` supports vectorization and prefetch hints.  
- `tile.store` writes to global or shared memory.  
- Compiler lowers these ops to **ld.global, cp.async, tma.load** in CUDA Tile IR.

---

### 8.3 Async Copies and Barriers

Tessera overlaps memory movement with compute using async copies:

```python
cp_async.shared.global(sA, A_tile)
cp_async.shared.global(sB, B_tile)
tbarrier()
C += tile.mma(sA, sB, accum=f32)
```

- `cp_async` maps to **CUDA async copy** (Hopper/Blackwell).  
- `tbarrier()` ensures visibility before compute.  
- Double-buffering pipelines improve throughput.

---

### 8.4 Distributions and Layouts Together

Layouts define access patterns **within a shard**, while distributions define how data is partitioned **across the mesh**.

```python
D = tessera.domain.Rect((8192, 8192))
W = tessera.array.from_domain(
    D,
    dtype="bf16",
    distribution=tessera.dist.Block(mesh_axes=("tp",)),
)
```

- Inside each shard, programmers choose row/col/blocked layouts.  
- Across shards, distributions define partitioning and collectives.

---

### 8.5 Region Privileges for Data Movement

Privileges ensure correct use of shared buffers:

```python
@tessera.jit
def block_mm(A: tessera.Region["read"], B: tessera.Region["read"], C: tessera.Region["reduce_sum"]):
    sA = tshared.alloc[f16](BM,BK)
    sB = tshared.alloc[f16](BK,BN)
    cp_async.shared.global(sA, A)
    cp_async.shared.global(sB, B)
    tbarrier()
    C[:] += tile.mma(sA, sB, accum=f32)
```

Privileges guarantee safe reads/writes across shards and prevent races.

---

### 8.6 Example: Distributed + Swizzled GEMM

```python
@tessera.kernel(autotune={"BM": [128, 256], "BN": [128, 256], "BK": [64], "stages": [3]})
def dist_gemm(A: tessera.f16["M", "K_per_tp"], B: tessera.f16["K_per_tp", "N"], C: tessera.mut_f32["M", "N_per_tp"]):
    sA = tshared.alloc[f16](BM,BK,swizzle="xor")
    sB = tshared.alloc[f16](BK,BN,swizzle="xor")
    for k0 in range(0, K, BK):
        cp_async.shared.global(sA, A[:,k0:k0+BK])
        cp_async.shared.global(sB, B[k0:k0+BK,:])
        tbarrier()
        C += tile.mma(sA, sB, accum=f32)
```

- Sharding handled by `tessera.dist.Block(mesh_axes=("tp",))` and the resulting `ShardSpec`.  
- Layout handled via swizzled tiles.  
- Autotuner picks best BM/BN/BK/stages.  

---

### 8.7 Data Movement on NVL72

NVL72 execution is Phase 4 planned. In that phase, data movement extends across the **NVSwitch fabric**:

- Intra-shard movement: registers ↔ shared ↔ HBM.  
- Inter-shard movement: collectives over NVLink/NVSwitch.  
- Tessera will fuse collective communication with local compute where legal.  

#### Future Example: Overlapped reduce_scatter
```python
# Phase 4 planned
with tessera.overlap(comm="tp_reduce_scatter"):
    grad_W = gemm(X, Y)   # compute while reduce_scatter happens
```

---

### 8.8 Debugging Data Movement

Tools to catch inefficiencies:

- **Bank conflict linter** for shared memory.  
- **Register pressure estimator** for tile kernels.  
- **Roofline hints** (compute vs memory bound).  
- **Trace timelines**: NVTX ranges for copies/compute overlap.

---

### 8.9 Summary

- Layouts are explicit: row/col, blocked, swizzled, interleaved.  
- Data movement controlled with `tile.load/store`, `cp_async`, `tbarrier()`.  
- Distributions and layouts compose: shard across mesh, tile within shard.  
- Region privileges ensure safe buffer use.  
- Tessera autotunes layouts & movement for peak performance.  
- NVL72 inter-shard movement and NCCL/SHARP mapping are Phase 4 planned.  
- Debugging tools help detect bank conflicts, pressure, and imbalance.

This unified approach to layouts and data movement ensures that Tessera kernels scale **from registers to NVL72 fabric** efficiently and safely.
