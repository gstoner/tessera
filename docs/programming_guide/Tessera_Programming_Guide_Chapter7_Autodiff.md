---
status: Tutorial
classification: Tutorial
last_updated: 2026-04-26
---

> **Phase status note:** Autodiff transforms, custom VJP/JVP, activation checkpointing, and distributed gradient collectives are Phase 5 planned unless explicitly marked otherwise. Current Phase 1-3 examples should use `@tessera.jit`, `tessera.ops`, and Graph IR inspection through `fn.graph_ir.to_mlir()`.

# Tessera Programming Guide
## Chapter 7: Autodiff

Autodiff is part of Tessera's design direction, but the public autodiff transform APIs are not current Phase 1-3 APIs. This chapter therefore separates current examples from future examples.

## 7.1 Current Pattern: Write Differentiable Computations As `@tessera.jit`

Today, write the forward computation using canonical APIs:

```python
import tessera

@tessera.jit
def forward(A: tessera.Tensor["M", "K"], B: tessera.Tensor["K", "N"]):
    return tessera.ops.gemm(A, B)
```

Inspect the current Graph IR:

```python
print(forward.graph_ir.to_mlir())
```

## 7.2 Effects That Future Autodiff Must Respect

Tessera tracks effects now, and future autodiff transforms will use those effects:

| Effect | Current source | Future autodiff implication |
|--------|----------------|-----------------------------|
| `pure` | `gemm`, `softmax`, `gelu`, `relu` | Recompute-safe |
| `random` | `dropout` | Requires deterministic seed handling |
| `memory` | Region writes and reductions | Requires privilege-safe adjoints |
| `io` | Collectives | Requires matched backward collectives |

Current deterministic checking already applies to random effects:

```python
@tessera.jit(deterministic=True, seed=42)
def stable_block(x):
    return tessera.ops.dropout(x, p=0.1)
```

## 7.3 Region Privileges And Future Gradients

Region privileges are current API. Future gradient lowering will preserve these contracts.

```python
@tessera.jit
def accumulate_grad(
    X: tessera.Region["read"],
    W: tessera.Region["read"],
    G: tessera.Region["reduce_sum"],
):
    G[:] += tessera.ops.gemm(X, W)
```

## 7.4 Future Example: Reverse-Mode Gradient

The following is design intent for Phase 5, not current API:

```python
# Phase 5 planned; final API name TBD
gradient_fn = tessera.autodiff.reverse(forward)
```

When implemented, reverse-mode gradients over distributed tensors will need to insert collectives such as `reduce_scatter` and `all_gather` according to the tensor distribution.

## 7.5 Future Example: Custom VJP

Custom VJP/JVP APIs are Phase 5 planned. Keep examples clearly marked:

```python
# Phase 5 planned; final API name TBD
# tessera.autodiff.custom_rule decorator planned
# def gelu_safe(x):
#     return tessera.ops.gelu(x)
```

## 7.6 Future Example: Activation Checkpointing

Activation checkpointing and rematerialization are Phase 5 planned:

```python
# Phase 5 planned; final API name TBD
# with tessera.autodiff.rematerialize():
#     y = tessera.ops.gemm(x, W)
```

## 7.7 Summary

- Current docs should use `@tessera.jit`, `tessera.ops`, region annotations, and `fn.graph_ir.to_mlir()`.
- Autodiff transforms, custom VJP/JVP, checkpointing, and distributed gradient collectives are Phase 5 planned.
- Region privileges and effects are already part of the architecture and define the contracts future autodiff must preserve.
