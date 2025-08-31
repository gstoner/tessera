# Tessera Programming Guide  
## Chapter 3: Memory Model (Merged Final)

Efficient GPU programming depends on understanding where data lives and how it moves. Tessera’s memory model makes this explicit: tiles operate on registers and shared memory, large tensors live in global HBM, and multi-GPU workloads use NVLink/NVSwitch. This chapter explains how these pieces fit together and how programmers control them in Tessera.

---

### 3.1 Memory Spaces

Tessera exposes the GPU’s memory hierarchy directly:

- **Registers**  
  - Private to each lane, used for scalars and accumulators.  
  - Fastest tier (~1 cycle). Compiler-managed, may spill to local memory.  

- **Shared Memory**  
  - Cooperative scratchpad for a tile group (maps to CUDA SMEM).  
  - Allocated with `tshared.alloc`, synchronized with `tbarrier()`.  
  - Best for staging tiles or cooperative reductions.  

- **Local Memory**  
  - Per-lane scratch space spilled into global DRAM.  
  - Typically avoided unless register pressure is high.  

- **Global Memory (HBM3e)**  
  - Device-wide DRAM for large tensors.  
  - High capacity, but ~200–300 cycle latency.  
  - Best used with coalesced or async copies.  

- **Constant Memory**  
  - Read-only, cached, efficient for broadcast scalars.  

- **Host Memory**  
  - CPU memory, accessible via explicit transfers or unified memory.  

**Programmer takeaway:** Registers and shared memory are for high-performance kernels; global memory is for storing large tensors; collectives move tensors between devices.  

---

### 3.2 NVIDIA GPU Memory Tiers

Tessera concepts align directly with NVIDIA’s hardware tiers:

| Memory Tier         | Tessera Concept        | Scope                      | Latency / Bandwidth            |  
|---------------------|------------------------|----------------------------|--------------------------------|  
| Registers           | Tile-local scalars     | Single lane / TensorCore   | ~1 cycle, highest BW           |  
| Shared Memory       | Tile-local buffer      | Tile group / SM            | ~20 cycles, TB/s throughput    |  
| HBM3e (Global)      | Tensor shards          | Single GPU                 | ~200–300 cycles, 8–12 TB/s     |  
| NVLink / NVSwitch   | Mesh-collective tensor | Multi-GPU domain           | ~500ns–1µs, 1–1.8 TB/s         |  

This mapping helps programmers reason about **where tensors live** and how costly it is to move them.  

---

### 3.3 Consistency & Synchronization

- **Within a tile:** instructions execute in program order.  
- **Within a group:** shared memory writes become visible after `tbarrier()`.  
- **Across groups:** visibility requires device fences or collective operations.  
- **Atomics:** supported at tile, group, device, and mesh scope.  

**Tip:** In practice, most synchronization is via `tbarrier()` inside a group, and `allreduce`/`reduce_scatter` across GPUs.  

---

### 3.4 Example: Shared Memory Reduction

```tessera
@kernel tile[(128,)] shared(16_KiB)
def sum_reduce(x: f32[n], out: mut f32[grid_groups]):
    smem = tshared.alloc
    i = tile_linear_id()
    acc = 0.0
    for k in range(i, n, group_tile_count()):
        acc += x[k]
    smem[i] = acc
    tbarrier()
    s = 64
    while s > 0:
        if i < s: smem[i] += smem[i + s]
        tbarrier()
        s //= 2
    if i == 0:
        atomic_add(out[group_linear_id()], smem[0])
```

This demonstrates using **shared memory + barriers** for a cooperative reduction.  

---

### 3.5 Sharded Tensors

Tessera expresses distributed global memory as **sharded tensors**. Each tensor has a `ShardSpec` describing:  
- **Partition dimensions** (row, col, batch, channel).  
- **Mapping to mesh axes** (dp = data parallel, tp = tensor parallel, pp = pipeline parallel).  
- **Replication or reduction** semantics.  

#### Example: Row-sharded tensor across tensor-parallel mesh
```python
from tessera import dist

X = dist.tensor(
    shape=(1024, 1024),
    layout=dist.ShardSpec(partition=("row",), mesh_axes=("tp",)),
    mesh=mesh,
    dtype="fp32"
)
```

Now each GPU in the mesh holds one row block of `X`.  

---

### 3.6 Replication & Reduction Semantics

Tessera allows programmers to annotate replication and reduction:  

- **Replication** – small weights copied onto each GPU.  
  ```python
  ShardSpec(partition=None, replicate=True)
  ```  

- **Reduction** – gradients summed across GPUs.  
  ```python
  ShardSpec(partition=("row",), reduce="sum")
  ```  

At runtime, Tessera inserts the correct **allreduce** or **reduce-scatter**.  

**Benefit for programmers:** Declare your distribution once — the compiler ensures collectives are correct and efficient.  

---

### 3.7 NVIDIA Tile IR Integration

On NVIDIA Hopper/Blackwell, Tessera lowers memory ops to CUDA Tile IR:  

- `tshared.alloc` → `tile.shared.alloc`  
- `cp_async.shared.global` → `tile.memcpy.async` or `tma.load` (bulk transfers)  
- `tbarrier()` → `tile.barrier`  

This ensures Tessera kernels leverage **async pipelines, Tensor Memory (TMEM), and TMA hardware copy engines** automatically.  

---

### 3.8 Unified Memory and Prefetching

Tessera supports unified memory for convenience, but performance-sensitive code should use:  
- **`prefetch`** – stage data ahead of use.  
- **`advise("device")`** – hint where data should live.  

These map to vendor APIs (`cudaMemPrefetchAsync`, ROCm HMM, oneAPI USM).  

---

### 3.9 Summary

- Tessera provides a **tile-aware view** of registers, shared, global (HBM), and mesh memory.  
- Programmers can shard tensors with `ShardSpec` and annotate replication/reduction.  
- **Shared memory + barriers** are the building blocks of cooperative kernels.  
- NVIDIA Tile IR integration maps Tessera ops to **cp.async, TMA, TMEM**.  
- Unified memory is supported, but **explicit placement & prefetch** yield the best performance.  
