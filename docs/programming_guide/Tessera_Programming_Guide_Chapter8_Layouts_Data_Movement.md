---
status: Tutorial
classification: Tutorial
last_updated: 2026-06-11
---

> **Phase status note (updated 2026-06-11):** Phases 1–7 are complete and Phase 8 (Apple M-Series CPU via Accelerate, GPU via Metal/MPS/MPSGraph/custom MSL) is operational — on Apple Silicon this is the primary single-node execution path. Autodiff (forward/reverse transforms + activation checkpointing), ZeRO-2 optimizer sharding, the Bayesian autotuner, and the runtime Python wrapper (`tessera.runtime.TesseraRuntime`) are **shipped**. Genuinely still planned: **multi-GPU / multi-rank** execution of distributed collectives (NCCL/RCCL), `Cyclic` distribution lowering, and **NVL72** rack-scale execution (single-device collectives run over in-process mock ranks today). Canonical API names: `docs/CANONICAL_API.md`; phase table: root `CLAUDE.md`.


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

### 8.7.1 Data Movement on Apple M-Series GPU (operational)

On Apple Silicon the data-movement primitives map onto Metal rather than
CUDA, and this lane executes today:

- **Buffers** are pooled `MTLBuffer`s acquired through RAII-hardened
  `TS_METAL_BUF_ACQUIRE` macros, so every dispatcher's early-return paths are
  release-safe by construction (locked by `test_apple_gpu_buffer_pool.py`).
- **Movement** is encoded into Metal command buffers; an encode-session /
  one-command-buffer substrate chains a multi-op program so intermediates stay
  resident on the GPU instead of round-tripping to host.
- **Graph reuse**: MPSGraph graphs are cached by `(shape-class, opcode, dtype,
  shape[, eps, weighted])` and reused across calls; custom MSL libraries are
  cached by source hash.
- **Auto-batch** coalesces compatible ops so they share one encode-session.

This is the inter-tile equivalent of the NVIDIA `cp_async`/TMA story for the
Apple backend. See [`docs/backends/apple/`](../backends/apple/).

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
