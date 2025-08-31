# Tessera Programming Guide  
## Chapter 4: Execution Model (Updated)

The Tessera execution model defines **how kernels are launched, scheduled, and orchestrated** across tiles, groups, and meshes. It extends the CUDA-style SPMD execution model with **tile-first semantics**, **async pipelining**, and **distributed index launches** that scale to NVL72.

---

### 4.1 Tile-First SPMD Execution

Each Tessera kernel is written in **SPMD style**: every tile executes the same program, operating on its portion of the data.

```tessera
@kernel
def scale(x: f32[n], y: mut f32[n], alpha: f32):
    i = tile.linear_id()
    if i < n:
        y[i] = alpha * x[i]
```

- **`tile.linear_id()`** gives the lane index within a group.  
- The compiler maps tiles to hardware threads/warps automatically.  
- Programmers reason about **tiles, not threads**.

---

### 4.2 Groups and Synchronization

Tiles are organized into **groups**, which cooperate via shared memory and synchronization.

```tessera
@kernel tile[(128,)] shared(8_KiB)
def group_sum(x: f32[n], out: mut f32[grid_groups]):
    smem = tshared.alloc[f32](128)
    i = tile.linear_id()
    acc = 0.0
    for k in range(i, n, group_tile_count()):
        acc += x[k]
    smem[i] = acc
    tbarrier()               # ensure all writes visible
    if i == 0:
        out[group_linear_id()] = reduce_sum(smem)
```

- `tbarrier()` synchronizes tiles within a group.  
- Groups map to CUDA thread blocks under the hood.  

---

### 4.3 Async Pipelining

Tessera supports **double-buffered pipelines** that overlap data movement with compute.

```tessera
@kernel.autotune(space=dict(BM=[128,256], stages=[2,3]))
def axpy_tiled(X: f16[m], Y: mut f16[m], alpha: f16):
    s0 = tshared.alloc[f16](128)
    s1 = tshared.alloc[f16](128)
    for t in tile.range(m, step=128, prefetch=2):
        cp_async.shared.global(s1, X[t:t+128])   # prefetch next chunk
        tbarrier()
        if t > 0:
            for i in range(128):
                yv = alpha * s0[i] + tile.load(Y, index=t-128+i)
                tile.store(Y, index=t-128+i, value=yv)
        s0, s1 = s1, s0
    tbarrier()
```

The compiler lowers `cp_async` and barriers to **CUDA Tile IR** for Hopper/Blackwell.

---

### 4.4 Distributed Execution with Index Launches

On multi-GPU meshes (e.g., NVL72), Tessera uses **index launches** to execute kernels across shards.

```python
@kernel
def tp_gemm(A: f16[M,K/tp], B: f16[K/tp,N], C: mut f32[M,N/tp]): ...

# Launch kernel across 9-way tensor-parallel axis
tessera.index_launch(axis="tp")(tp_gemm)(A.parts("tp"), B.parts("tp"), C.parts("tp"))
```

- **`index_launch`** fans out kernels across partitions of a distributed tensor.  
- Collectives (`all_gather`, `reduce_scatter`) are inserted automatically.  
- On NVL72, Tessera maps this to NCCL collectives over NVSwitch with SHARP reductions.

---

### 4.5 Domains & Distributions in Execution

Execution can be expressed over **domains** with attached distributions:

```python
D = tessera.domain.Rect((B, S, Dm))
dist = tessera.dist.Block(mesh_axes=("dp","tp"))
X = tessera.array.from_domain(D, dtype="bf16", distribution=dist)

@jit
def norm_layer(X: Region[read], Y: Region[write]):
    Y[:] = rmsnorm_safe(X)
```

- The runtime partitions `D` according to `dist`.  
- Execution launches kernels on each shard automatically.  
- Region privileges (`read`, `write`) allow safe overlap.

---

### 4.6 Region Privileges in Scheduling

Privileges prevent conflicting accesses during execution:

```python
@jit
def step(W: Region[read], X: Region[read], Y: Region[reduce_sum]):
    # Compiler schedules reduce ops safely
    Y += gemm(X, W)
```

This ensures that **reductions** can be overlapped and fused without programmer-managed synchronization.

---

### 4.7 Execution on NVL72

On NVL72 (72-GPU domain):  
- **Meshes** span all 72 GPUs (e.g., dp=4 × tp=9 × pp=2).  
- Index launches distribute work across tensor-parallel and pipeline-parallel shards.  
- Collectives map to NVLink/NVSwitch with SHARP-enabled NCCL.  
- CUDA Graph capture (`@jit(capture_graph=True)`) minimizes per-launch overhead.  

#### Example: Training step graph on NVL72
```python
@jit(capture_graph=True)
def train_step(batch):
    out = model(batch)
    loss = loss_fn(out)
    return loss
```

---

### 4.8 Summary

- Tessera executes kernels in a **tile-first SPMD model**.  
- Groups cooperate via **shared memory + barriers**.  
- Async pipelining overlaps **copies + compute**.  
- **Index launches** scale execution across mesh partitions.  
- **Domains & distributions** describe global iteration spaces.  
- **Region privileges** ensure safe scheduling and fusion.  
- On NVL72, Tessera maps directly to NCCL/SHARP collectives with CUDA Graphs for low-overhead execution.
