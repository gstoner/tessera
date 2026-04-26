---
status: Informative
classification: Informative
last_updated: 2026-04-26
---

# Tessera API Reference

This reference summarizes the current public API shape. The authoritative API specification is `docs/spec/PYTHON_API_SPEC.md`; if this guide disagrees with that spec, the spec wins.

## Import Pattern

```python
import tessera
```

Use the top-level `tessera` namespace for public examples unless a spec explicitly names a submodule import.

## Decorators

| API | Status | Purpose |
|-----|--------|---------|
| `@tessera.jit` | Phase 1-3 implemented | Compile a Python function to Graph IR. |
| `@tessera.kernel` | Phase 1-3 implemented | Mark a tile-level function for `index_launch`. |

```python
@tessera.jit
def matmul_step(A: tessera.Tensor["M", "K"], B: tessera.Tensor["K", "N"]):
    return tessera.ops.gemm(A, B)
```

## Region Privileges

`Region[...]` is a type annotation only. It lowers to privilege/effect attributes in Graph IR.

```python
@tessera.jit
def update(
    X: tessera.Region["read"],
    W: tessera.Region["read"],
    Y: tessera.Region["write"],
):
    Y[:] = tessera.ops.gemm(X, W)
```

Valid modes are:

| Mode | Meaning |
|------|---------|
| `read` | Read-only access |
| `write` | Exclusive write access |
| `reduce_sum` | Parallel sum reduction |
| `reduce_max` | Parallel max reduction |
| `reduce_min` | Parallel min reduction |

## Domains, Distributions, And Arrays

Domains describe shape. Distributions describe placement. Keep them separate.

```python
D = tessera.domain.Rect((4, 128, 256))
dist = tessera.dist.Block(mesh_axes=("dp", "tp"))
X = tessera.array.from_domain(D, dtype="bf16", distribution=dist)
```

`tessera.dist.Cyclic` exists as a Phase 4 planned distribution; in Phases 1-3 it raises `NotImplementedError` when shard specs are materialized.

## Index Launch

`index_launch` dispatches a `@tessera.kernel` function over shard lists.

```python
@tessera.kernel
def tp_gemm(A: tessera.f16[..., ...], B: tessera.f16[..., ...], C: tessera.mut_f32[..., ...]):
    C[:] = tessera.ops.gemm(A, B)

tessera.index_launch(axis="tp")(tp_gemm)(
    A.parts("tp"),
    B.parts("tp"),
    C.parts("tp"),
)
```

Phase 1 uses sequential/mock execution. Production NCCL/RCCL-backed distributed execution is Phase 4 planned.

## Constraints

Use `tessera.require(...)` inside `@tessera.jit` functions.

```python
@tessera.jit(bindings={"K": 128})
def aligned_gemm(A: tessera.Tensor["M", "K"], B: tessera.Tensor["K", "N"]):
    tessera.require(tessera.constraint.Divisible("K", 64))
    return tessera.ops.gemm(A, B)
```

Implemented constraints:

| API | Meaning |
|-----|---------|
| `tessera.constraint.Divisible(dim, divisor)` | `dim % divisor == 0` |
| `tessera.constraint.Range(dim, lo, hi)` | `lo <= dim <= hi` |
| `tessera.constraint.Equal(dim_a, dim_b)` | `dim_a == dim_b` |

## Operations

Use `tessera.ops`.

| API | Status |
|-----|--------|
| `gemm`, `matmul` | Phase 1-3 implemented |
| `layer_norm`, `softmax`, `gelu`, `relu`, `transpose`, `cast` | Phase 1-3 implemented |
| `dropout` | Phase 1-3 implemented with random effect |
| `conv2d` | Phase 1 stub / Graph IR op support |
| `flash_attn` | Phase 1 naive path; Phase 3 SM_90+ FA-4 lowering path |
| `all_reduce`, `reduce_scatter`, `all_gather` | Phase 1 stubs; Phase 4 planned distributed lowering |
| `fused_epilogue` | Phase 1-3 implemented where supported by canonicalization/lowering |

## GPU Targeting

```python
from tessera.compiler.gpu_target import GPUTargetProfile, ISA
from tessera.compiler.attn_lower import FlashAttnLoweringConfig

@tessera.jit(
    target=GPUTargetProfile(isa=ISA.SM_90, warps_per_cta=4),
    attn_config=FlashAttnLoweringConfig(tile_q=64, tile_kv=64, causal=True),
)
def flash_fwd(Q, K, V):
    return tessera.ops.flash_attn(Q, K, V, causal=True)
```

## Inspection

Current public inspection supports Graph IR:

```python
print(flash_fwd.graph_ir.to_mlir())
```

Schedule IR, Tile IR, and Target IR inspection helpers are Phase 4 planned.

## Future APIs

The following areas are roadmap items and should be labeled as future examples in docs:

| Area | Phase |
|------|-------|
| NCCL/RCCL collectives and cluster execution | Phase 4 planned |
| TPU StableHLO backend | Phase 4 planned |
| Autodiff transforms and custom VJP/JVP | Phase 5 planned |
| Activation checkpointing and ZeRO sharding | Phase 5 planned |
| Bayesian autotuning | Phase 5 planned |
| Runtime Python wrapper | Phase 6 planned |
