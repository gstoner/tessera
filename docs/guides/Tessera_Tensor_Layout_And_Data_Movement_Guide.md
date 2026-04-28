---
status: Normative
classification: Normative
authority: Tensor layout and explicit data movement guidance; defers memory ordering to docs/spec/MEMORY_MODEL_SPEC.md
last_updated: 2026-04-28
---

# Tessera Tensor Layout And Data Movement Guide

Tessera treats tensor layout and data movement as first-class compiler objects.
Layouts are part of tensor types and operator contracts. Movement is an explicit
effect that Schedule IR owns before Tile IR lowers it to backend instructions.

## 1. Layout Model

Every tensor has:

| Field | Meaning |
|-------|---------|
| Shape | Static or symbolic dimensions. |
| Dtype | Storage dtype such as `bf16`, `fp8_e4m3`, `fp6_e2m3`, or `nvfp4`. |
| Layout | Logical-to-physical mapping. |
| Distribution | Optional mesh placement and sharding. |
| Alignment | Required byte alignment for vector, TMA, or MMA access. |

Canonical layouts:

| Layout | Use |
|--------|-----|
| `row_major` | Dense matrices and default tensors. |
| `col_major` | Transposed GEMM operands and Fortran-style dense tensors. |
| `nhwc` / `nchw` | Image and convolution tensors. |
| `nld` / `blh` | Sequence tensors. |
| `tile(BM, BN, BK)` | Schedule/Tile IR view for MMA kernels. |
| `fragment(m, n, k, layout)` | Tensor Core fragment shape. |
| `bsr(block_m, block_n, mask)` | Block-sparse matrices. |
| `paged(page_size, order)` | KV cache and long-context state. |
| `swizzled(kind)` | Shared-memory bank-conflict avoidance. |

Operators must either accept the input layout or request an explicit
`layout_cast`. Hidden layout changes are not allowed in normative examples.

## 2. Dtype And Layout Interaction

Low-precision types are normally packed. Layout metadata must say how logical
elements map to storage lanes.

| Dtype | Layout considerations |
|-------|-----------------------|
| `fp8_e4m3`, `fp8_e5m2` | Usually vectorized and scaled by tensor or channel. |
| `fp6_e2m3`, `fp6_e3m2` | Packed lanes; alignment must be declared. |
| `fp4_e2m1`, `nvfp4` | Packed 4-bit lanes plus scale metadata. |
| `int8`, `int4` | Quantized storage with per-tensor/per-channel scale policy. |
| `bf16`, `fp16` | Common Tensor Core storage; usually FP32 accumulation. |

Rubin target note: `GPUTargetProfile(isa=ISA.SM_120)` is a Tessera placeholder
for Rubin-family target planning until NVIDIA publishes final compute
capability numbering. It exposes Tensor Core dtype support for `nvfp4`,
`fp4_e2m1`, FP8, FP6, INT8, FP16, BF16, TF32, and FP64.

## 3. Movement Effects

Movement is represented in Schedule IR before Tile IR:

```python
with prefetch(cache, into="shared", overlap="compute"):
    y = tessera.ops.flash_attn(q, k, v, cache=cache)
```

Canonical movement operations:

| Operation | Effect | Meaning |
|-----------|--------|---------|
| `schedule.prefetch` | `movement` | Request data placement before compute. |
| `schedule.layout_cast` | `movement` when materialized | Convert layout for producer/consumer compatibility. |
| `schedule.pipeline` | `movement` + schedule metadata | Overlap copies and compute. |
| `tile.async_copy` | `movement` | Backend-independent async copy. |
| `tile.mbarrier.*` | synchronization | Hopper+ transaction tracking for async movement. |
| `tile.wait_async` | synchronization | Completion edge before consumer reads. |

## 4. Hopper+ Mbarrier Movement Pattern

For NVIDIA Hopper and later, Tensor Memory Accelerator movement should use
mbarrier transaction tracking when the copy completion is consumed by another
warp role or phase.

```mlir
%bar = tile.mbarrier.alloc {count = 1, scope = "block"}
tile.async_copy %global, %shared {stage = 0, vector = 16}
%tok = tile.mbarrier.arrive_expect_tx %bar
  {bytes = 16384, semantics = "release", scope = "block"}
%ready = tile.mbarrier.try_wait %bar, %tok
```

Rules:

- mbarrier use requires `target.supports_mbarrier`
- initialization must dominate arrival and wait
- transaction byte counts must correspond to associated async movement
- producer/consumer pipelines should use separate ready/filled barriers or
  equivalent phase tracking
- consumer reads must be dominated by a successful wait

## 5. Layout Casts

`layout_cast` may be metadata-only or materialized:

| Case | Behavior |
|------|----------|
| Producer can emit requested layout | Fold into producer. |
| Consumer accepts source layout | Fold into consumer. |
| Layout changes physical order | Materialize copy and mark `movement`. |
| Distributed layout changes shard ownership | Lower through collectives or all-to-all. |

Materialized layout casts must appear in schedule artifacts and replay manifests.

## 6. KV Cache And Paged Layouts

KV cache state uses `paged` or `rolling_window` layouts:

```text
Cache["B","H","S_max","D_h", bf16] @rolling_window @layout=paged(256)
```

Movement plan:

1. append new K/V into the logical cache
2. map logical sequence positions to pages or ring slots
3. prefetch active pages into shared memory
4. use mbarrier/TMA completion when available
5. prune or evict according to the cache policy

The Graph IR object owns cache semantics. Schedule IR owns movement. Tile IR
owns the actual copy, barrier, and wait instructions.

## 7. Schedule Artifact Requirements

Any tuned layout/movement plan must include:

- logical shape and physical layout
- dtype and numeric policy
- target architecture
- alignment and vector width
- swizzle and bank-padding policy
- movement graph
- mbarrier stages and transaction byte counts where used
- tile knobs and schedule hash

## 8. Validation Checklist

- every operator declares accepted layouts
- every physical layout change is explicit
- packed dtypes declare lane packing and scale metadata
- vector and TMA accesses satisfy alignment
- shared-memory swizzles match MMA/ldmatrix access patterns
- async copies have completion edges before use
- mbarrier plans are only emitted for Hopper+ targets
- distributed layout changes lower through typed collectives
