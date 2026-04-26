---
status: Tutorial
classification: Tutorial
last_updated: 2026-04-26
---

> **Phase status note:** Distributed collectives, NVL72 execution, autodiff transforms, and checkpointing are Phase 4-5 planned unless explicitly marked otherwise. Current examples use APIs from `docs/CANONICAL_API.md`.

# Tessera Programming Guide
## Chapter 2: Programming Model

The Tessera programming model exposes a Python surface over a four-layer compiler stack:

```text
Python API -> Graph IR -> Schedule IR -> Tile IR -> Target IR
```

The current public surface centers on `@tessera.jit`, `@tessera.kernel`, region privileges, domains/distributions, `DistributedArray`, `index_launch`, and `tessera.ops`.

## 2.1 JIT Functions

Use `@tessera.jit` for graph-level computations.

```python
import tessera

@tessera.jit
def matmul_step(A: tessera.Tensor["M", "K"], B: tessera.Tensor["K", "N"]):
    tessera.require(tessera.constraint.Divisible("K", 64))
    return tessera.ops.gemm(A, B)
```

Inspect current Graph IR:

```python
print(matmul_step.graph_ir.to_mlir())
```

## 2.2 Region Privileges

Regions describe access privileges. They are type annotations, not runtime wrappers.

```python
@tessera.jit
def write_output(
    A: tessera.Region["read"],
    B: tessera.Region["read"],
    C: tessera.Region["write"],
):
    C[:] = tessera.ops.gemm(A, B)
```

Reduction privileges are also explicit:

```python
@tessera.jit
def accumulate(
    X: tessera.Region["read"],
    G: tessera.Region["reduce_sum"],
):
    G[:] += X
```

## 2.3 Domains And Distributions

Domains describe logical shape. Distributions describe placement.

```python
D = tessera.domain.Rect((4, 128, 256))
dist = tessera.dist.Block(mesh_axes=("dp", "tp"))
X = tessera.array.from_domain(D, dtype="bf16", distribution=dist)
```

`tessera.dist.Cyclic` is Phase 4 planned. Use `Block` or `Replicated` for current examples.

## 2.4 Kernel Functions And Index Launch

Use `@tessera.kernel` for tile-level kernels dispatched over shard lists.

```python
@tessera.kernel
def tp_gemm(
    A: tessera.f16[..., ...],
    B: tessera.f16[..., ...],
    C: tessera.mut_f32[..., ...],
):
    C[:] = tessera.ops.gemm(A, B)

tessera.index_launch(axis="tp")(tp_gemm)(
    A.parts("tp"),
    B.parts("tp"),
    C.parts("tp"),
)
```

Phase 1 uses sequential/mock execution for this style. Production distributed launch and collectives are Phase 4 planned.

## 2.5 Numerics

Use dtype annotations and operation-level APIs from the canonical specs:

```python
@tessera.kernel
def bf16_gemm(
    A: tessera.bf16[..., ...],
    B: tessera.bf16[..., ...],
    C: tessera.mut_f32[..., ...],
):
    C[:] = tessera.ops.gemm(A, B)
```

Advanced FP4/FP6 policy syntax and full Blackwell-specific numerics are future-facing unless specified by the normative API docs.

## 2.6 Future Example: Distributed Training On NVL72

The following concepts are Phase 4-5 planned:

```python
# Phase 4-5 planned sketch
# - 72-GPU mesh placement
# - NCCL/RCCL collectives
# - distributed autodiff
# - activation checkpointing
```

Do not use legacy distribution constructors or unmarked gradient-transform examples as current API guidance.

## 2.7 Summary

- Use `@tessera.jit` for graph-level functions.
- Use `@tessera.kernel` plus `index_launch` for tile-level distributed dispatch tests.
- Use `tessera.domain.Rect` and `tessera.dist.Block/Replicated` for current distributed-array examples.
- Use `tessera.ops` for operations.
- Mark distributed collectives, NVL72 execution, autodiff, checkpointing, and advanced autotuning as planned future work.
