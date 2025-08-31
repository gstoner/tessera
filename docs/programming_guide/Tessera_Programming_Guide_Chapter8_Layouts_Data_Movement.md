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
```tessera
sA = tshared.alloc[f16](BM, BK, swizzle="xor")
sB = tshared.alloc[f16](BK, BN, swizzle="xor")
```

Layouts are explicit so programmers can reason about **bank conflicts, vectorization, and tiling**.

---

### 8.2 Tile Loads and Stores

Data is moved between memory tiers explicitly:

```tessera
Qb = tile.load(Q, rows=T.m, cols=T.d, vector=T.vector, prefetch=2)
tile.store(C, index=(i,j), value=Qb)
```

- `tile.load` supports vectorization and prefetch hints.  
- `tile.store` writes to global or shared memory.  
- Compiler lowers these ops to **ld.global, cp.async, tma.load** in CUDA Tile IR.

---

### 8.3 Async Copies and Barriers

Tessera overlaps memory movement with compute using async copies:

```tessera
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
mesh = dist.mesh(devices=[f"cuda:{i}" for i in range(72)], axes=("dp","tp"), shape=(8,9))

W = dist.tensor(shape=(8192,8192),
    layout=ShardSpec(partition=("col",), mesh_axes=("tp",)),
    mesh=mesh,
    dtype="bf16")
```

- Inside each shard, programmers choose row/col/blocked layouts.  
- Across shards, distributions define partitioning and collectives.

---

### 8.5 Region Privileges for Data Movement

Privileges ensure correct use of shared buffers:

```python
@jit
def block_mm(A: Region[read], B: Region[read], C: Region[reduce_sum]):
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

```tessera
@kernel.autotune(space=dict(BM=[128,256], BN=[128,256], BK=[64], stages=[3]))
def dist_gemm(A: f16[M,K/tp], B: f16[K/tp,N], C: mut f32[M,N/tp]):
    sA = tshared.alloc[f16](BM,BK,swizzle="xor")
    sB = tshared.alloc[f16](BK,BN,swizzle="xor")
    for k0 in range(0, K, BK):
        cp_async.shared.global(sA, A[:,k0:k0+BK])
        cp_async.shared.global(sB, B[k0:k0+BK,:])
        tbarrier()
        C += tile.mma(sA, sB, accum=f32)
```

- Sharding handled via `ShardSpec` (TP axis).  
- Layout handled via swizzled tiles.  
- Autotuner picks best BM/BN/BK/stages.  

---

### 8.7 Data Movement on NVL72

On NVL72, data movement extends across the **NVSwitch fabric**:

- Intra-shard movement: registers ↔ shared ↔ HBM.  
- Inter-shard movement: collectives over NVLink/NVSwitch.  
- Tessera fuses collective communication with local compute.  

#### Example: Overlapped reduce_scatter
```python
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
- On NVL72, Tessera maps inter-shard movement to NCCL/SHARP collectives.  
- Debugging tools help detect bank conflicts, pressure, and imbalance.

This unified approach to layouts and data movement ensures that Tessera kernels scale **from registers to NVL72 fabric** efficiently and safely.
