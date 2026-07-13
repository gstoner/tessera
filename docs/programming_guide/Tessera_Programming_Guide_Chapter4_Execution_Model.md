---
status: Tutorial
classification: Tutorial
last_updated: 2026-06-11
---

> **Phase status note (updated 2026-06-11):** Phases 1–7 are complete and Phase 8 (Apple M-Series CPU via Accelerate, GPU via Metal/MPS/MPSGraph/custom MSL) is operational — on Apple Silicon this is the primary single-node execution path. Autodiff (forward/reverse transforms + activation checkpointing), ZeRO-2 optimizer sharding, the Bayesian autotuner, and the runtime Python wrapper (`tessera.runtime.TesseraRuntime`) are **shipped**. Genuinely still planned: **multi-GPU / multi-rank** execution of distributed collectives (NCCL/RCCL), `Cyclic` distribution lowering, and **NVL72** rack-scale execution (single-device collectives run over in-process mock ranks today). Canonical API names: `docs/CANONICAL_API.md`; phase table: root `CLAUDE.md`.


# Tessera Programming Guide  
## Chapter 4: Execution Model (Updated)

The Tessera execution model defines **how kernels are launched, scheduled, and orchestrated** across tiles, groups, and meshes. It extends the CUDA-style SPMD execution model with **tile-first semantics**, **async pipelining**, and **distributed index launches** that scale to NVL72.

---

### 4.1 Tile-First SPMD Execution

Each Tessera kernel is written in **SPMD style**: every tile executes the same program, operating on its portion of the data.

```python
@tessera.kernel
def scale(x: tessera.f32["n"], y: tessera.mut_f32["n"], alpha: tessera.f32):
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

```python
@tessera.kernel(tile=(128,), shared="8KiB")
def group_sum(x: tessera.f32["n"], out: tessera.mut_f32["grid_groups"]):
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

```python
@tessera.kernel(autotune={"BM": [128, 256], "stages": [2, 3]})
def axpy_tiled(X: tessera.f16["m"], Y: tessera.mut_f16["m"], alpha: tessera.f16):
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

On multi-GPU meshes (Phase 4 planned, including NVL72), Tessera uses **index launches** to execute kernels across shards.

```python
@tessera.kernel
def tp_gemm(A: tessera.f16["M", "K_per_tp"], B: tessera.f16["K_per_tp", "N"], C: tessera.mut_f32["M", "N_per_tp"]): ...

# Phase 4 planned: launch kernel across a tensor-parallel axis
tessera.index_launch(axis="tp")(tp_gemm)(A.parts("tp"), B.parts("tp"), C.parts("tp"))
```

- **`index_launch`** fans out kernels across partitions of a distributed tensor.  
- Collectives (`all_gather`, `reduce_scatter`) are Phase 4 planned for distributed lowering.  
- NVL72 execution and NCCL/SHARP mapping are Phase 4 planned.

---

### 4.5 Domains & Distributions in Execution

Execution can be expressed over **domains** with attached distributions:

```python
D = tessera.domain.Rect((B, S, Dm))
dist = tessera.dist.Block(mesh_axes=("dp","tp"))
X = tessera.array.from_domain(D, dtype="bf16", distribution=dist)

@tessera.jit
def norm_layer(X: tessera.Region["read"], Y: tessera.Region["write"]):
    Y[:] = tessera.ops.rmsnorm_safe(X)
```

- The runtime partitions `D` according to `dist`.  
- Execution launches kernels on each shard automatically.  
- Region privileges (`read`, `write`) allow safe overlap.

---

### 4.6 Region Privileges in Scheduling

Privileges prevent conflicting accesses during execution:

```python
@tessera.jit
def step(W: tessera.Region["read"], X: tessera.Region["read"], Y: tessera.Region["reduce_sum"]):
    # Compiler schedules reduce ops safely
    Y += tessera.ops.gemm(X, W)
```

This ensures that **reductions** can be overlapped and fused without programmer-managed synchronization.

---

### 4.7 Execution on NVL72

NVL72 execution is Phase 4 planned. The intended behavior is:
- **Meshes** span all 72 GPUs (e.g., dp=4 × tp=9 × pp=2).  
- Index launches distribute work across tensor-parallel and pipeline-parallel shards.  
- Collectives map to NVLink/NVSwitch with SHARP-enabled NCCL.  
- CUDA Graph capture for repeated training steps is planned future runtime work.  

#### Example: Training step graph on NVL72
```python
@tessera.jit
def train_step(batch):
    out = model(batch)
    loss = loss_fn(out)
    return loss
```

---

### 4.8 Execution on Apple M-Series GPU (operational)

On Apple Silicon, `@tessera.jit(target="apple_gpu")` is a **real execution
path** today (Phase 8.3 → 8.4.7), not a planned one. Instead of CUDA streams,
the runtime dispatches through a Metal command-buffer model:

- **Three dispatch lanes** cover the op surface: Apple's **MPS** matmul, the
  **MPSGraph** lane (Tier-1 activations/norms — `silu`, `gelu`, `softmax`,
  `layer_norm`, `rmsnorm`, `log_softmax`, … — with no `N ≤ 256` limit), and
  **custom MSL** kernels for the gaps (rope, flash-attention, fused chains).
- **Fused chains** execute as single kernels: `matmul→softmax`,
  `matmul→gelu`, `matmul→rmsnorm`, and the full attention block
  `matmul→softmax→matmul`. Pipeline ordering tries the longest fusion first
  so the most specific kernel wins.
- **Descriptor-driven dispatch**: the compiler stamps a `tessera.fusion.intent`
  the Target IR consumes, so the chosen fused kernel is the compiler's decision
  rather than a runtime re-discovery (see Ch. references and the Target IR spec).
- **Auto-batch** groups compatible ops, MSL libraries are cached by source
  hash, and MPSGraph graphs are cached by shape/opcode/dtype.

A full multi-head-attention block composes from these without a per-head loop,
and a single-layer LLaMA-style decoder (RMSNorm + attention + SwiGLU) has been
validated end-to-end against a numpy reference. See
[`docs/backends/apple/`](../backends/apple/). The Apple **CPU**
path (`target="apple_cpu"`, Accelerate) and a production CPU MLIR→LLVM **JIT**
lane are likewise executable today.

---

### 4.9 Summary

- Tessera executes kernels in a **tile-first SPMD model**.  
- Groups cooperate via **shared memory + barriers**.  
- Async pipelining overlaps **copies + compute**.  
- **Index launches** scale execution across mesh partitions.  
- **Domains & distributions** describe global iteration spaces.  
- **Region privileges** ensure safe scheduling and fusion.  
- Phase 4 planned NVL72 support maps distributed collectives to NCCL/SHARP where available; CUDA Graph capture is future runtime work.
