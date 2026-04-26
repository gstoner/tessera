---
status: Tutorial
classification: Tutorial
last_updated: 2026-04-26
---

> **Phase status note:** Unless this document explicitly says otherwise, distributed collectives (NCCL/RCCL), TPU StableHLO, Cyclic distribution, autodiff transforms, activation checkpointing, ZeRO sharding, Bayesian autotuning, the runtime Python wrapper, production deployment, and NVL72 execution are Phase 4-6 planned as defined in `docs/README.md`. Current Phase 1-3 API names are defined in `docs/CANONICAL_API.md`.


# Tessera Programming Guide
## Chapter 3: Memory Model

Efficient GPU programming depends on understanding where data lives and how it moves.
Tessera's memory model makes this explicit through a six-tier hierarchy, a staged async
copy model, explicit distribution semantics, and first-class cache management
abstractions. This chapter walks through every tier, explains how to move data between
them efficiently, and shows how the compiler lowers these abstractions to hardware.

---

### 3.1 The Six Memory Tiers

Tessera exposes the full GPU memory hierarchy, each with a distinct performance profile:

| Tier | Tessera name | Scope | Latency | Capacity |
|------|-------------|-------|---------|----------|
| **Registers** | `register` | Single lane / TensorCore | ~1 cycle | ~256 × 4 B per thread |
| **Shared Memory / LDS** | `shared` | Tile group (SM) | ~20 cycles | 48–228 KiB per SM |
| **Tensor Memory (TMEM)** | `tmem` | SM — hardware-managed | ~20 cycles | ~128 KiB per SM (SM_100+) |
| **Global Memory (HBM)** | `global` | Single GPU | ~200–300 cycles | HBM3e: 80–192 GiB |
| **Managed / Unified** | `managed` | Host and device | Migration latency | Full device capacity |
| **Host Memory** | `host` | CPU RAM | PCIe / NVLink latency | System RAM |

And a sixth space for multi-GPU workloads:

| Space | Tessera name | Scope | Latency / Bandwidth |
|-------|-------------|-------|-------------------|
| **NVLink / NVSwitch** | mesh collective | Multi-GPU domain | ~500ns–1µs, 1.8 TB/s per link |

**Programmer takeaway:** Registers and shared memory are for in-flight compute; global
memory holds persistent tensors between kernel launches; mesh collectives synchronise
distributed tensors across GPUs.

---

### 3.2 Registers

Registers are private to each hardware thread (lane) and are managed entirely by the
compiler. Programmers never allocate registers explicitly — the compiler assigns them
from tile-level scalar and accumulator values.

```tessera
# Register use is implicit: scalars and accumulators live in registers
@tessera.kernel
def scale(X: tessera.f32[..., ...], Y: tessera.mut_f32[..., ...], alpha: float):
    i = tile.linear_id()
    acc = alpha * X[i]    # 'acc' assigned to a register by the compiler
    Y[i] = acc
```

**When registers spill** to local memory (per-thread DRAM), performance degrades sharply.
Signs of spilling: reduced occupancy warnings from the autotuner, unexpectedly low
throughput. Remedies: reduce tile size, lower vector width, or split the kernel.

---

### 3.3 Shared Memory

Shared memory is an on-chip scratchpad shared by all threads in a tile group. It is
the primary mechanism for:

- Staging global memory reads ahead of compute (reduce latency exposure).
- Communicating intermediate results between threads in a group (reductions, scans).
- Buffering reuse — load a tile once, use it many times.

#### Allocation

```tessera
# Basic allocation
smem = tshared.alloc[f16](BM, BK)

# With swizzle pattern to eliminate bank conflicts
smem = tshared.alloc[f16](BM, BK, swizzle="xor")

# With explicit padding to avoid strided-access bank conflicts
smem = tshared.alloc[f16](BM, BK + 1)    # +1 column pad
```

**Swizzle patterns** reorder the physical layout within shared memory to ensure that
column-major and strided access patterns hit different banks. `"xor"` is the standard
XOR-swizzle that CUTLASS and FA-2 implementations use. For most GEMM and attention
kernels, `swizzle="xor"` eliminates 2-way and 4-way bank conflicts with no code change.

#### Synchronization

```tessera
# Required after any shared memory write before a read by a different thread
tbarrier()
```

Every write-then-read across threads requires a `tbarrier()`. Missing barriers are a
common source of incorrect results that only manifest at certain tile sizes.

#### Example: Cooperative reduction

```tessera
@tessera.kernel
def tile_sum(x: tessera.f32[..., ...], out: tessera.mut_f32[...]):
    smem = tshared.alloc[f32](128)
    i    = tile.linear_id()

    smem[i] = x[tile.group_id() * 128 + i]
    tbarrier()                     # ensure all writes landed

    # Tree reduction
    s = 64
    while s > 0:
        if i < s:
            smem[i] += smem[i + s]
        tbarrier()
        s //= 2

    if i == 0:
        atomic_add(out[tile.group_id()], smem[0])
```

---

### 3.4 Tensor Memory (TMEM) — SM_100+ / Blackwell

Tensor Memory (TMEM) is a new on-chip memory tier introduced on NVIDIA Blackwell
(SM_100). It sits at the same latency tier as shared memory but is managed directly by
the hardware's Tensor Core units, enabling the MMA pipeline to accumulate into TMEM
without register traffic.

TMEM is not programmer-allocated — the compiler inserts TMEM allocation when lowering
`tile.mma` on SM_100 targets:

```mlir
# Compiler-generated Tile IR (SM_100) — not written by programmers
%tmem_accum = tessera.tcgen05.alloc : memref<64x64xf32, tmem>
%result     = tessera.tcgen05.mma %a_tile, %b_tile, %tmem_accum
              : (tensor<64x64xbf16>, tensor<64x64xbf16>, memref<64x64xf32, tmem>)
              -> memref<64x64xf32, tmem>
tessera.tcgen05.commit %result    # flush TMEM → registers for epilogue
```

**What this means for you:** If your target is `ISA.SM_100` (Blackwell), the compiler
automatically routes accumulation through TMEM, improving MMA throughput with no kernel
changes. You only need to ensure `target=GPUTargetProfile(isa=ISA.SM_100)` is set on
`@tessera.jit`.

---

### 3.5 Global Memory (HBM) and Async Copies

Global memory is device-wide DRAM. It holds all persistent tensors between kernel
launches. The key rule is: never read from global memory with a blocking load inside a
compute loop if you can avoid it. Use async copies to overlap the movement with compute.

#### Staged async copy model

Tessera's async copy system uses an explicit **stage index** to model double- and
multi-buffering pipelines. Each stage is a small non-negative integer forming a DAG:

```tessera
# Double-buffered pipeline: stage 0 and stage 1 alternate
s0 = tshared.alloc[f16](BM, BK, swizzle="xor")   # stage 0 buffer
s1 = tshared.alloc[f16](BM, BK, swizzle="xor")   # stage 1 buffer

for t in tile.range(m, step=BK, prefetch=2):
    # Prefetch next chunk into the "loading" buffer
    cp_async.shared.global(s1, X[t:t+BK], stage=1)
    tbarrier()

    # Compute on the "ready" buffer
    if t > 0:
        for i in range(BK):
            acc += s0[i] * Y[t - BK + i]

    s0, s1 = s1, s0    # swap buffers
tbarrier()
```

At the Tile IR level this lowers to:

```mlir
tile.async_copy %src, %smem { stage = 0, vector = 16 } : (memref<*,1>, memref<*,3>) -> ()
tile.wait_async { stage = 0 }
```

The **stage verifier** checks that every `async_copy(stage=N)` is matched by a
`wait_async(stage=N)` in dominance — missing wait calls are caught at compile time.

#### Coalesced access pattern

Global memory bandwidth is maximised when consecutive threads access consecutive
addresses. A 128-byte cache line holds 32 `fp32` values or 64 `fp16` values. If thread
`i` reads `X[i]`, the access is perfectly coalesced. Strided or scattered access
patterns waste 75–94% of available bandwidth.

---

### 3.6 Distributed Tensors and Sharding

Multi-GPU workloads store tensors that are logically global but physically partitioned
across devices. Tessera expresses this with the `DistributedArray` type and `ShardSpec`
metadata.

#### The Phase 1 API: `tessera.array.from_domain`

```python
import tessera

# Step 1: define the logical shape
D = tessera.domain.Rect((4, 128, 256))       # (batch, seq, hidden)

# Step 2: choose a partition strategy
dist = tessera.dist.Block(mesh_axes=("dp", "tp"))
#   Block distributes the first N dimensions over the named axes:
#   → dim 0 (batch=4) partitioned over "dp"
#   → dim 1 (seq=128) partitioned over "tp"

# Step 3: create the array
X = tessera.array.from_domain(D, dtype="bf16", distribution=dist)

# The ShardSpec records integer dimension indices (not string names):
assert X.shard_spec.partition  == (0, 1)         # dims 0 and 1 are partitioned
assert X.shard_spec.mesh_axes  == ("dp", "tp")   # over these mesh axes
assert X.shape                 == (4, 128, 256)  # logical (global) shape
assert X.dtype                 == "bf16"
```

#### Distribution strategies

```python
# Contiguous block partition — most common
dist_block = tessera.dist.Block(mesh_axes=("dp",))

# Round-robin partition — load-balanced for MoE (Phase 4)
dist_cyclic = tessera.dist.Cyclic(mesh_axes=("tp",))    # Phase 4

# Fully replicated — small weights broadcast to all ranks
dist_repl = tessera.dist.Replicated()
W = tessera.array.from_domain(
    tessera.domain.Rect((256, 256)),
    dtype="bf16",
    distribution=dist_repl
)
# W.shard_spec.replicated == True
```

#### Accessing per-rank shards

```python
# .parts(axis) returns a list of per-rank DistributedArray slices
shards = X.parts("dp")      # list of 4 shards along dim 0 (batch)
assert shards[0].shape == (1, 128, 256)   # one batch element per rank

# Use with index_launch to dispatch a kernel to each rank's shard
tessera.index_launch(axis="dp")(my_kernel)(
    X.parts("dp"),
    Y.parts("dp"),
)
```

#### ShardSpec in IR

The compiler lowers `ShardSpec` into `tessera.shard` attributes on Graph IR tensor values:

```mlir
# ShardSpec(partition=(1,), mesh_axes=("tp",))
%W = func.arg : tensor<256x256xbf16>
     {tessera.shard = {axes = ["tp"], dims = [1]}}
```

---

### 3.7 Replication and Reduction Semantics

#### Replicated tensors

Small tensors (biases, layer norms, small weight matrices) are replicated across all
ranks so every rank has an identical copy and no communication is needed on the forward
pass.

```python
bias = tessera.array.from_domain(
    tessera.domain.Rect((256,)),
    dtype="fp32",
    distribution=tessera.dist.Replicated()
)
# bias.shard_spec.replicated == True
# bias.parts("dp") returns [bias]   (same object for all ranks)
```

#### Region privileges for reductions

Gradient accumulation is expressed via `Region["reduce_sum"]` on function parameters.
The compiler inserts the correct `reduce_scatter` or `allreduce` collective:

```python
@tessera.jit
def accumulate_grad(
    X:    tessera.Region["read"],
    W:    tessera.Region["read"],
    grad: tessera.Region["reduce_sum"],   # allreduce in backward
):
    grad[:] += tessera.ops.gemm(X, W)
```

**Privilege-to-collective mapping:**

| Region privilege | Inserted collective |
|-----------------|-------------------|
| `"read"` | none (read-only) |
| `"write"` | none (exclusive write) |
| `"reduce_sum"` | `allreduce(op="sum")` or `reduce_scatter` |
| `"reduce_max"` | `allreduce(op="max")` |
| `"reduce_min"` | `allreduce(op="min")` |

---

### 3.8 Cache Management Abstractions

For inference workloads with long contexts, Tessera provides first-class KV cache,
page table, and ring buffer abstractions. These map to `cache.*` IR ops and lower to
optimised memory management on the target.

#### KV Cache

```mlir
# Graph IR — KV cache with LRU eviction
%kv = cache.kv.create { key_dtype = f16, value_dtype = f16,
                        line = 256, evict = "lru" }
%page  = cache.page.lookup %kv, %pos : (cache.kv, i32) -> cache.page
cache.page.write %kv, %page, %k_tile, %v_tile
%k2, %v2 = cache.page.read %kv, %page
```

This translates to paged KV cache allocations at runtime: each `cache.page` is a
256-element slot in a page table. Eviction policy (`"lru"`, `"fifo"`, `"none"`) is a
compile-time attribute.

#### Page Tables

For dynamic context lengths, the page table indirection decouples logical sequence
position from physical memory location:

```mlir
%pt   = cache.pt.create  { page_size = 256, pages = 4096 }
# page_size = tokens per physical page; pages = total page capacity
```

#### Ring Buffers

Useful for streaming token generation or micro-batch pipeline staging:

```mlir
%rb   = cache.ring.create { capacity = 65536 }
cache.ring.push %rb, %item
%item = cache.ring.pop  %rb
```

**Programmer note:** These IR abstractions are emitted by `@tessera.jit` when you use
high-level cache hints (Phase 4+). For Phase 1–3, KV caching is handled by passing
explicit KV tensors to `tessera.ops.flash_attn`.

---

### 3.9 Deterministic Execution and Memory Ordering

Tessera supports a deterministic execution mode that guarantees numerically reproducible
results across runs, across ranks, and across backends. Two sources of non-determinism
are addressed: reduction ordering and RNG state.

#### Marking a function deterministic

```python
@tessera.jit(deterministic=True, seed=42)
def stable_forward(x: tessera.Tensor["B", "D"]):
    return tessera.ops.layer_norm(x)
```

This lowers to a Graph IR attribute:

```mlir
func.func @stable_forward(...) attributes { tessera.deterministic = { seed = 42 } }
```

#### Ordered reductions

In a non-deterministic region, floating-point additions can happen in any order (hardware
parallel reductions produce different rounding). In a `deterministic` region, reductions
use a canonical tree order with stable rank partitioning:

```mlir
tile.reduce %x { op = "sum", order = "tree" } : (vector<128xf32>) -> f32
```

The **verifier** rejects unordered reductions inside deterministic regions at compile time.

#### Seeded RNG

RNG ops inside a deterministic region derive their stream from a tuple:
`(func_id, mesh_rank_coords, step, user_seed)`. This ensures:

- Same seed always produces the same dropout mask on any rank.
- Different ranks get independent (non-overlapping) streams.

```mlir
rng.uniform %shape { stream = "default" } : tensor<*xf32>
```

The verifier rejects any `rng.*` op inside a deterministic region that lacks a bound
stream.

---

### 3.10 NVIDIA Tile IR Lowering

On NVIDIA Hopper and Blackwell, Tessera lowers memory operations through the Tile IR
dialect. Understanding this mapping helps when reading compiler output or debugging
performance.

| Tessera source | Tile IR op | Target hardware |
|----------------|-----------|----------------|
| `tshared.alloc[f16](M, K)` | `tile.alloc_shared : memref<MxKxf16, 3>` | SMEM |
| `tshared.alloc[f16](M, K, swizzle="xor")` | `tile.alloc_shared {swizzle="xor", bank_pad=1}` | SMEM |
| `cp_async.shared.global(smem, src, stage=0)` | `tile.async_copy %src, %smem {stage=0, vector=16}` | `cp.async` / TMA |
| `tbarrier()` | `tile.wait_async {stage=0}` + `tile.barrier` | `mbarrier.arrive/wait` |
| `tile.mma(sA, sB, accum=f32)` | `tile.mma %sA, %sB : bf16,bf16 → f32` | WGMMA (SM_90+) / WMMA |

On SM_90 (Hopper), the `AsyncCopyLoweringPass` selects between:
- **TMA path** (`cp.async.bulk.tensor`) for bulk 2D copies from global → shared.
- **`cp.async` path** for element-wise async copies (SM < 90 fallback).

On SM_100 (Blackwell), `tile.mma` additionally routes accumulators through TMEM as
described in §3.4.

#### Full swizzled GEMM kernel in Tile IR

```mlir
%sA = tile.alloc_shared : memref<128x64xbf16, 3> { swizzle = "xor", bank_pad = 1 }
%sB = tile.alloc_shared : memref<64x128xbf16, 3> { swizzle = "xor", bank_pad = 1 }

// Async prefetch
tile.async_copy %A_tile, %sA { stage = 0, vector = 16 }
tile.async_copy %B_tile, %sB { stage = 0, vector = 16 }
tile.wait_async { stage = 0 }

// MMA accumulate
%C  = tile.mma %sA, %sB : memref<128x64xbf16, 3>, memref<64x128xbf16, 3>
               -> tensor<128x128xf32>

// Store
tile.store %C, %C_global[%i, %j]
```

---

### 3.11 Unified Memory and Prefetching

Tessera supports unified memory (`managed` address space) for convenience during
prototyping. Performance-sensitive paths should use explicit placement instead.

```python
# Hint that tensor should reside on device for the next kernel
prefetch(X, target="device")

# Hint that this region won't need CPU access for a while
advise(W, policy="device")
```

These lower to vendor APIs: `cudaMemPrefetchAsync` (CUDA), `hipMemPrefetchAsync` (ROCm),
and `svmMigrateMem` (oneAPI). The compiler may insert prefetch hints automatically based
on memory access patterns inferred from the Graph IR.

---

### 3.12 Memory Decision Guide

Use this table to choose the right memory tier for a given use case:

| Use case | Recommended tier | Notes |
|----------|-----------------|-------|
| Accumulator between MMA instructions | Registers | Compiler assigns automatically |
| Staging a tile for reuse across iterations | Shared memory | `tshared.alloc` + `tbarrier()` |
| Hopper/Blackwell MMA accumulation | TMEM | Compiler inserts automatically on SM_90+/SM_100+ |
| Large weight matrices | Global (HBM) | Load with `cp_async` into shared memory |
| Distributed model weights | Sharded global | `tessera.dist.Block` + `from_domain` |
| Small replicated weights (biases) | Replicated | `tessera.dist.Replicated()` |
| Inference KV cache | `cache.kv` abstraction | Phase 4+; for Phase 1–3, pass explicit KV tensors |
| Gradient accumulation | `Region["reduce_sum"]` | Compiler inserts collective |
| Reproducible dropout / RNG | Deterministic region | `@tessera.jit(deterministic=True, seed=N)` |
| Prototyping / convenience | Managed/unified | Avoid in production — migration latency |

---

### 3.13 Summary

- Tessera exposes **six memory tiers**: registers, shared (SMEM), TMEM (SM_100+), global
  (HBM), managed, and host — each with distinct latency and capacity.
- **Shared memory + barriers** are the building blocks of cooperative in-tile computation.
  Use `swizzle="xor"` to eliminate bank conflicts without code changes.
- **Staged async copies** (`cp_async` with explicit stage indices) overlap data movement
  with compute — the verifier enforces that every stage is matched by a `wait_async`.
- **TMEM** (Blackwell) is a compiler-managed MMA accumulator space; it activates
  automatically on SM_100+ targets with no code changes.
- **Sharded tensors** use `tessera.array.from_domain` + `tessera.dist.Block/Cyclic/
  Replicated`. `ShardSpec` uses integer dimension indices, not string names.
- **Region privileges** (`"reduce_sum"`, `"reduce_max"`, `"reduce_min"`) cause the
  compiler to insert the correct collective at mesh boundaries.
- **KV cache abstractions** (`cache.kv`, `cache.pt`, `cache.ring`) are first-class IR
  constructs for inference workloads (Phase 4+).
- **Deterministic mode** (`@tessera.jit(deterministic=True, seed=N)`) guarantees
  reproducible reductions and RNG across all ranks and runs.
- On NVIDIA GPUs, Tessera maps memory ops to **`cp.async`, TMA, WGMMA, and TMEM** — the
  specific path is selected by the backend lowering passes based on `target_profile.isa`.

---

### Chapter Navigation

- **Previous:** [Chapter 2: Programming Model](Tessera_Programming_Guide_Chapter2_Programming_Model.md)
- **Next:** [Chapter 4: Execution Model](Tessera_Programming_Guide_Chapter4_Execution_Model.md)
