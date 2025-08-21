# Tessera Memory Model Specification
*(Normative — ordering, visibility, synchronization, determinism across threads, warps, GPUs, and meshes.)*

**Status:** Draft v1.0 (Normative unless marked *Informative*)

---

## 1. Introduction
This specification defines Tessera's memory model: visibility and ordering guarantees for loads/stores, synchronization primitives, consistency across threads/warps/blocks/devices, and numerical determinism in distributed environments.

---

## 2. Memory Spaces (Normative)
- **Local (private)**: Per-thread, not visible to others; lifetime = thread.
- **Shared (SM)**: Visible to threads within a block/tile; requires `barrier.block()` for visibility.
- **Global (device)**: Visible to all threads in a device; requires fences for device-wide visibility guarantees.
- **Distributed (mesh)**: Tensor shards across devices; visibility established by collectives and `dist.barrier`.
- **Host**: CPU memory; access via mapped/unified memory, or explicit transfers.

---

## 3. Consistency Model (Normative)

### 3.1 Baseline
- **Program order** preserved within a single thread.
- Cross-thread/device visibility requires synchronization.
- Absent synchronization, **data races are undefined behavior**.

### 3.2 Synchronization Scopes
- `thread` — no inter-thread visibility.
- `warp` — intra-warp operations have implicit lockstep execution, but visibility is only guaranteed through warp shuffles or barriers.
- `block` — `barrier.block()` orders shared-memory accesses.
- `device` — `fence.device()` orders global-memory visibility.
- `mesh` — `dist.barrier(mesh)` orders distributed operations across devices.

### 3.3 Ordering
- Stores followed by `barrier.block()` then loads in other threads of the same block **shall** observe the stores.
- `fence.device()` **shall** make prior global stores visible to all device threads launched after the fence and to concurrent threads that synchronize appropriately.
- `dist.barrier(mesh)` **shall** make prior global/distributed stores visible to all participants in the barrier.

---

## 4. Atomics (Normative)
Supported operations: `add`, `sub`, `min`, `max`, `and`, `or`, `xor`, `exchange`, `cas` on integer and floating types as supported by target.

### 4.1 Memory Orders
- `relaxed` — atomicity only.
- `acquire` — prevents subsequent operations from moving before the atomic.
- `release` — prevents prior operations from moving after the atomic.
- `acq_rel` — acquire+release.
- `seq_cst` — total order across the device for that atomic location/category.

### 4.2 Scopes
Atomics may specify scope: `thread | warp | block | device | mesh`. Wider scopes imply stronger visibility but higher cost.

---

## 5. Barriers & Fences (Normative)

```python
barrier.block()    # block-scoped rendezvous and visibility for shared memory
fence.device()     # device-wide visibility for prior global stores
dist.barrier(mesh) # distributed visibility across the mesh
```

- Barriers are **collective** within their scope and **shall not** be placed in divergent control paths that prevent all participants from reaching them.
- Fences are **unary** (no rendezvous), they only order the calling thread's accesses w.r.t. others.

---

## 6. Determinism & Numerical Stability (Normative)

- `numerics.profile("deterministic"|"strict")` **shall** enforce fixed reduction trees and disallow nondeterministic atomics for floating-point aggregation.
- Mixed precision **shall** follow operator-specified accumulation rules (e.g., GEMM accumulates in FP32).
- Randomness **shall** be seeded per mesh participant; stream IDs provide reproducible substreams.

---

## 7. Happens-Before (Formal)

A write **W** happens-before a read **R** if any of the following hold:
1. W precedes R in program order in the same thread.  
2. W is release and R is acquire for the same location with overlapping scope.  
3. W is ordered before a barrier/fence, and R is ordered after the same barrier/fence (within scope).  
4. There exists a chain of happens-before edges connecting W to R.

Data race definition: two conflicting accesses (at least one write) without a happens-before relation → **undefined behavior**.

---

## 8. Examples (Informative)

### 8.1 Shared Memory Visibility
```python
s = op.alloc_shared((32,), dtype="fp32")

if threadIdx == 0:
    s[0] = 42.0

barrier.block()
x = s[0]  # All threads observe 42.0
```

### 8.2 Device-Wide Publication
```python
g = op.tensor((1,), dtype="int32")  # global
g[0] = 7
fence.device()
# Other threads can observe g[0] == 7 after synchronizing appropriately
```

### 8.3 Deterministic All-Reduce
```python
y = dist.all_reduce(x, op="sum", deterministic=True)
dist.barrier(mesh)
```

### 8.4 Scoped Atomics
```python
counter = op.tensor((1,), dtype="int32")
atomic.add(counter, 1, order="acq_rel", scope="block")
barrier.block()
```

---

## 9. Verification Rules (Normative Checklist)
- No barrier divergence within scope.  
- Shared-memory uses are followed by `barrier.block()` before cross-thread reads.  
- Global publication uses `fence.device()` before cross-thread consumption.  
- Distributed communication paired with `dist.barrier(mesh)` or collective ops.  
- Atomics specify order/scope consistent with required visibility.  

---

## 10. Backend Mapping (Informative)
- CUDA/PTX: `barrier.block()` ↔ `bar.sync`; `fence.device()` ↔ `membar.gl`; atomics ↔ `atom.*`, `red.*`; warp ops ↔ `shfl.sync`.  
- ROCm: LDS barriers and DS atomics; global fences via `s_memtime` rules and cache controls.

---

## 11. Change History
- v1.0 — Initial publication.
