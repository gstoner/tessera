---
status: Normative
classification: Normative
authority: Memory ordering, visibility, synchronization, and determinism model
last_updated: 2026-04-28
---

# Tessera Memory Model Specification

This specification defines Tessera visibility and ordering guarantees for
loads, stores, synchronization primitives, atomics, asynchronous movement,
transaction barriers, and distributed mesh execution.

## 1. Memory Spaces

| Space | Visibility | Lifetime | Typical backend mapping |
|-------|------------|----------|-------------------------|
| `local` | One thread | Thread | Registers/private memory |
| `shared` | One block or cluster, depending on scope | CTA/cluster | NVIDIA SMEM, AMD LDS |
| `tmem` | Tensor-core accumulator scope | Kernel/tile | NVIDIA Blackwell+ TMEM |
| `global` | Device | Allocation | HBM/device DRAM |
| `distributed` | Mesh participants | Runtime object | Sharded tensors, NVLink/XGMI fabric |
| `host` | CPU process | Host allocation | Pageable, pinned, mapped, or unified memory |

Shared memory visibility across threads requires a block or cluster-scope
barrier. Global memory publication requires a device-scope fence or a
higher-level synchronization edge. Distributed memory visibility is established
by collectives, mesh barriers, or runtime protocol handshakes.

## 2. Baseline Consistency

- program order is preserved within a single thread
- cross-thread and cross-device visibility requires synchronization
- data races without a happens-before relation are undefined behavior
- deterministic profiles may reject legal-but-nondeterministic implementations

## 3. Synchronization Scopes

| Scope | Meaning |
|-------|---------|
| `thread` | No inter-thread visibility. |
| `warp` | Intra-warp coordination through shuffle/vote/barrier primitives. |
| `block` | CTA-wide synchronization and shared-memory visibility. |
| `cluster` | Hopper+ cluster-scope synchronization where supported. |
| `device` | Device-wide global-memory visibility with fences or kernel boundaries. |
| `mesh` | Distributed visibility across devices/ranks. |

Barriers are collective within their scope and must not be placed in divergent
control paths that prevent participants from reaching them. Fences are unary:
they order the calling thread's memory accesses but are not rendezvous points.

## 4. Barriers, Fences, And Mbarriers

Primitive model:

```text
barrier.block()
barrier.cluster()
fence.device()
dist.barrier(mesh)
mbarrier.alloc(scope, count)
mbarrier.arrive(token?)
mbarrier.arrive_expect_tx(bytes)
mbarrier.try_wait(token_or_parity)
mbarrier.arrive_and_drop()
```

Hopper forward rule: NVIDIA targets with `isa >= SM_90` support asynchronous
transaction barriers for shared-memory block or cluster scope. Tessera models
these as mbarriers. They may track both thread arrivals and asynchronous
transaction byte counts, which is required for TMA-style producer/consumer
pipelines.

Ordering rules:

- stores before `barrier.block()` are visible to same-block loads after the barrier
- `fence.device()` orders prior global stores before later synchronized readers
- `dist.barrier(mesh)` orders prior distributed writes before later mesh reads
- an mbarrier phase completes only after participating arrivals and expected
  transaction bytes are complete
- mbarrier waits must use a token or explicit phase/parity value from the same
  barrier object

## 5. Atomics

Supported atomic operations:

```text
add sub min max and or xor exchange cas
```

Integer atomics are always atomic for supported widths. Floating-point atomics
are target-dependent and are nondeterministic for reductions unless a
deterministic numeric profile provides a fixed reduction tree instead.

Memory orders:

| Order | Meaning |
|-------|---------|
| `relaxed` | Atomicity only. |
| `acquire` | Later memory operations cannot move before the atomic. |
| `release` | Earlier memory operations cannot move after the atomic. |
| `acq_rel` | Acquire and release. |
| `seq_cst` | Single total order for the location/category at the requested scope. |

Scopes:

```text
thread warp block cluster device mesh
```

Wider scopes imply stronger visibility and higher cost.

## 6. Asynchronous Movement

Asynchronous movement is a `movement` effect. Schedule IR must represent the
movement plan before Tile IR lowering. Tile IR then materializes async copies,
transaction barriers, waits, and fences.

```text
schedule.prefetch -> tile.async_copy -> tile.mbarrier.arrive_expect_tx
                  -> tile.mbarrier.try_wait -> compute
```

Verification:

- every async copy stage must have a completion edge before use
- TMA/mbarrier byte counts must match the copied region
- shared destinations must satisfy target alignment
- producer/consumer pipelines must have enough buffers for overlap
- waits must dominate consumer reads

## 7. Determinism And Numerical Stability

`numerics.profile("deterministic")` or `numerics.profile("strict")` requires:

- fixed reduction trees
- deterministic RNG stream assignment per mesh participant
- stable collective ordering
- no nondeterministic floating-point atomics for aggregation
- schedule artifact reuse or deterministic autotune search order

Mixed precision must follow operator numeric policies. For example, FP8, FP6,
FP4, and NVFP4 storage may require FP32 accumulation depending on the operator.

## 8. Happens-Before

A write `W` happens-before a read `R` if:

1. `W` precedes `R` in program order in the same thread.
2. `W` is release and `R` is acquire for the same location with overlapping scope.
3. `W` is ordered before a barrier/fence/mbarrier completion and `R` is ordered
   after the same synchronization event within scope.
4. A collective or mesh barrier defines a distributed ordering edge.
5. There is a transitive chain of happens-before edges connecting `W` to `R`.

Data race definition: two conflicting accesses, at least one of which is a
write, without a happens-before relation form a data race and have undefined
behavior.

## 9. Verification Checklist

- no barrier divergence within scope
- shared-memory cross-thread reads are ordered by barriers or mbarriers
- global publication uses device fences or kernel/collective boundaries
- distributed communication uses collectives or mesh barriers
- atomics specify order and scope
- mbarriers are only used on targets that support them
- mbarrier initialization dominates arrival and wait
- deterministic profiles do not use nondeterministic reductions

## 10. Backend Mapping

| Tessera primitive | NVIDIA mapping | AMD mapping |
|-------------------|----------------|-------------|
| `barrier.block()` | `bar.sync` / `__syncthreads()` | `s_barrier` |
| `fence.device()` | `membar.gl` / CUDA fence | global fence/cache controls |
| `mbarrier.*` | PTX `mbarrier.*`, CUDA `cuda::barrier` native handle | backend-specific async wait groups where available |
| `tile.async_copy` | `cp.async` or `cp.async.bulk.tensor` | LDS/global async copy patterns |
| warp shuffle | `shfl.sync` | wavefront permute/readlane ops |
| atomics | `atom.*`, `red.*` | flat/global/LDS atomics |
