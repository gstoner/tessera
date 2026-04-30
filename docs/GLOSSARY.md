---
status: Informative
classification: Informative
authority: Terminology reference; normative definitions defer to docs/CANONICAL_API.md and docs/spec/
last_updated: 2026-04-30
---

# Tessera Glossary

Terms are listed alphabetically. For normative definitions, follow the spec
links. This file provides plain-language explanations.

---

## @tessera.jit

The primary decorator for compiling a Python function through the Tessera
compiler pipeline. It runs the `ConstraintSolver`, infers effects, and emits
Graph IR. See [`docs/CANONICAL_API.md §Decorators`](CANONICAL_API.md).

## @tessera.kernel

Marks a tile-level kernel function dispatched by `index_launch`. Used for
per-shard computation when parallelizing over a mesh axis.

## Async Copy

A non-blocking data movement from global memory to shared memory, tracked by
a transaction barrier (`mbarrier`). Available on SM_90+ (Hopper) via TMA.
See [`docs/spec/TILE_IR.md`](spec/TILE_IR.md).

## Autotuner

A search algorithm that selects optimal tile sizes, pipeline depths, and
layout configurations for a kernel. Phase 1–2 uses Hyperband grid search;
Phase 5 upgrades to Bayesian optimization via Optuna TPE.

## Block Distribution

`tessera.dist.Block(mesh_axes=(...))` — a contiguous block partition that
splits the first N logical dimensions across the named mesh axes. Contrast
with Cyclic distribution.

## Canonicalization

The Graph IR canonicalization pass (`CanonicalizeTesseraIR`) applies four
rewrite patterns: FuseMatmulBiasGELU, FuseConvRelu, DropoutZeroSimplify, and
TransposeIntoMatmul. It runs first in both named lowering pipelines.

## Conformance Profile

One of three levels (T0, T1, T2) that define what a compliant Tessera
implementation must support. See [`docs/spec/CONFORMANCE.md`](spec/CONFORMANCE.md).

## Constraint

A structural property of a function's type signature checked at `@jit`
decoration time. Examples: `Divisible("K", 64)`, `Range("S", 1, 8192)`,
`Equal("D_in", "D_out")`. Violations raise `TesseraConstraintError`.

## CTA (Cooperative Thread Array)

CUDA terminology for a thread block. Tessera's `warps_per_cta` parameter
controls how many warps share a CTA's shared memory and synchronization scope.

## Cyclic Distribution

`tessera.dist.Cyclic(mesh_axes=(...))` — a round-robin element partition.
Used for MoE load balancing where tokens are distributed across experts with
uniform spread. Contrast with Block distribution.

## Decoration Time

When a `@tessera.jit` or `@tessera.kernel` decorator executes. The earliest
error-detection point; constraint checking and effect inference run here.

## DistributedArray

A Tessera array that carries a `ShardSpec` and a logical (global) shape.
Created with `tessera.array.from_domain(...)`. In Phase 1, backed by a numpy
array on CPU.

## Domain

`tessera.domain.Rect(dims)` — the logical iteration space (shape only). Always
kept separate from the distribution strategy.

## Effect Lattice

The ordering `pure < random < memory < io < top` used to classify side effects.
The compiler infers effects; programmers only declare `deterministic=True`.
See [`docs/CANONICAL_API.md §Effect System`](CANONICAL_API.md).

## FA-4

Flash Attention 4 — the Tessera implementation of online-softmax FlashAttention
targeting SM_90+ via WGMMA + TMA. The FA-4 Tile IR dialect adds
`ScaledDotProduct`, `OnlineSoftmax`, `LseAccumulate`, `DropoutMask`, and
`CausalMask` ops.

## Fragment

A register-resident tile of a matrix used by MMA instructions. Fragment types
carry shape, dtype, and layout constraints matching the target's matrix
accelerator (WGMMA, MFMA, AMX).

## Graph IR

The first IR layer. Represents algebraic operators, autodiff, state objects,
and distributed intent using the `tessera` MLIR dialect. Produced by `@jit`.
See [`docs/spec/GRAPH_IR_SPEC.md`](spec/GRAPH_IR_SPEC.md).

## ISA (Instruction Set Architecture)

Tessera's `ISA` enum identifies GPU compute capability: `SM_80` (A100),
`SM_90` (H100/GH200), `SM_100` (B100/GB200). WGMMA and TMA require SM_90+.

## Index Launch

`tessera.index_launch(axis="tp")(kernel)(shards...)` — dispatches a kernel
once per rank along the named mesh axis, passing per-rank shard slices.

## KV Cache

A `KVCacheType` in Graph IR for transformer key-value attention caching.
Supports rolling-window eviction and paged allocation. The Flash Attention
kernel appends to and reads from the cache.

## Layout

The physical arrangement of elements in memory (e.g., row-major, column-major,
swizzled). Tessera treats layout as first-class: operators must declare legal
input/output layouts or require an explicit `layout_cast`.

## Lowering Pipeline

A named sequence of MLIR passes that transforms Graph IR down to target code.
Two pipelines exist today: `tessera-lower-to-x86` (AMX/AVX-512) and
`tessera-lower-to-gpu` (SM_90+ NVIDIA). See
[`docs/spec/COMPILER_REFERENCE.md`](spec/COMPILER_REFERENCE.md).

## mbarrier (Transaction Barrier)

An NVIDIA SM_90+ synchronization primitive used to track TMA async copy
completion. Tessera emits `tile.mbarrier.alloc`, `arrive_expect_tx`, and
`try_wait` ops in Tile IR.

## Mesh

`mesh<axes=[...], shape=[...]>` — a process/device grid. Defines how tensors
are partitioned across devices. Mesh axes are named strings (e.g., `"dp"`,
`"tp"`, `"pp"`).

## MoE (Mixture of Experts)

A model architecture where tokens are routed to a subset of expert networks.
Tessera's Cyclic distribution supports MoE token routing via `all_to_all`
collectives.

## Numeric Policy

`NumericPolicy(storage, accum, rounding, scale, ...)` — controls dtype
promotion, rounding mode, quantization axis, and determinism for an operator.
Operators must carry legal numeric policies for the requested dtype/target pair.

## Online Softmax

The two-pass algorithm from the Flash Attention 2 paper: running max, running
sum, and correction factor applied at LSE (log-sum-exp) accumulation. Used
in FA-4 to process long sequences without materializing the full attention
matrix.

## Region

`tessera.Region["mode"]` — a type annotation on function parameters that
declares read/write/reduce intent. Not a runtime wrapper. Lowers to
`tessera.effect` attributes on Graph IR function arguments.

## Schedule IR

The second IR layer. Legalizes Graph IR into tiled, fused, pipelined kernels
with explicit movement and memory staging using the `schedule.*` MLIR dialect.

## ShardSpec

`ShardSpec(partition=(0, 1), mesh_axes=("dp", "tp"))` — metadata describing
which logical dimensions are partitioned and over which mesh axes.

## Solver Pipeline

A sequence of C++ passes for scientific computing workloads: sparse
inspection, preconditioning, RNG legalization, Newton autodiff, and implicit
lowering. Phase 5 planned.

## Target IR

The fourth and final IR layer. Lowers Tile IR to vendor-specific intrinsics:
WGMMA, TMA, mbarrier for NVIDIA; MFMA/LDS for AMD; StableHLO for TPU.

## TesseraBench

The Tessera benchmarking framework (analogous to tritonbench). Documentation
lives in [`docs/benchmarks/`](benchmarks/).

## Tile IR

The third IR layer. Binds scheduled computation to accelerator execution
primitives: blocks, warps, fragments, shared memory, transaction barriers, and
MMA instructions. See [`docs/spec/TILE_IR.md`](spec/TILE_IR.md).

## TMA (Tensor Memory Accelerator)

A hardware unit on SM_90+ that moves data between global and shared memory
asynchronously using pre-built descriptors. Avoids warp stalls during data
prefetch.

## T0 / T1 / T2

The three Tessera conformance profiles. T0 = kernel-only (Phases 1–2 today).
T1 = single-node GPU execution (Phase 3 compiler subset today; full T1 pending
Phase 6 runtime ABI). T2 = multi-node cluster (Phase 4 planned).

## Warp Specialization

A technique where warps within a CTA are assigned distinct roles (producer or
consumer) to overlap data movement with computation. `WarpSpecializationPass`
inserts role annotations and tile queue barriers in the FA-4 pipeline.

## WGMMA (Warp Group Matrix Multiply-Accumulate)

The SM_90+ warp-group-level matrix multiply instruction (`wgmma.mma_async`).
Operates on fragments stored in registers + shared memory and accumulates into
a register tile. Lower latency than WMMA for large tiles.

## ZeRO (Zero Redundancy Optimizer)

A memory-optimization strategy that partitions optimizer state across data
parallel ranks. Tessera Phase 5 implements ZeRO stage 2: momentum and variance
partitioned across the `dp` mesh axis.
