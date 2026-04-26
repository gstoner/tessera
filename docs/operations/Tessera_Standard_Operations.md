---
status: Normative
classification: Normative
authority: Canonical operations guidance; defers signatures to docs/spec/PYTHON_API_SPEC.md and IR semantics to docs/spec/GRAPH_IR_SPEC.md
last_updated: 2026-04-26
---

# Tessera Standard Operations

This document is the canonical operations guide for current Tessera documentation. It describes the operations that users should call through `tessera.ops` and how those operations map into the compiler architecture.

For exact public signatures, use `docs/spec/PYTHON_API_SPEC.md`. For Graph IR verifier rules and MLIR examples, use `docs/spec/GRAPH_IR_SPEC.md`. For GPU lowering behavior, use `docs/spec/LOWERING_PIPELINE_SPEC.md` and `docs/spec/TARGET_IR_SPEC.md`.

## API Rule

Use the canonical namespace:

```python
import tessera

@tessera.jit
def step(A: tessera.Tensor["M", "K"], B: tessera.Tensor["K", "N"]):
    return tessera.ops.gemm(A, B)
```

Do not use deprecated decorator aliases or old neural-network helper namespaces in current documentation. Public examples should use `@tessera.jit`, `@tessera.kernel`, and `tessera.ops`.

## Phase Status

| Area | Status |
|------|--------|
| Python `tessera.ops` namespace | Phase 1-3 implemented |
| Graph IR emission for core math ops | Phase 1-3 implemented |
| x86 AMX/AVX-512 lowering for supported matmul paths | Phase 2 implemented |
| NVIDIA SM_90+ FlashAttention lowering path | Phase 3 implemented |
| Distributed collectives backed by NCCL/RCCL | Phase 4 planned |
| Autodiff transforms, checkpointing, ZeRO sharding, Bayesian autotuning | Phase 5 planned |
| Runtime C ABI production execution path and Python wrapper | Phase 6 planned |

## Operation Catalog

| Operation | Effect | Current behavior |
|-----------|--------|------------------|
| `tessera.ops.gemm(A, B)` | `pure` | Matrix multiply. Emits/lowers as `tessera.matmul` in Graph IR where compiled. |
| `tessera.ops.matmul(A, B)` | `pure` | Alias for `gemm`. |
| `tessera.ops.layer_norm(x, eps=1e-5)` | `pure` | Numerically stable normalization in the Python stub path. |
| `tessera.ops.softmax(x, axis=-1)` | `pure` | Numerically stable softmax in the Python stub path. |
| `tessera.ops.gelu(x)` | `pure` | GELU activation; may participate in fused epilogue canonicalization. |
| `tessera.ops.relu(x)` | `pure` | ReLU activation; may participate in canonicalization. |
| `tessera.ops.transpose(x, axes=None)` | `pure` | Transpose; may fold into `tessera.matmul` transpose attributes. |
| `tessera.ops.cast(x, dtype)` | `pure` | Element type cast. |
| `tessera.ops.dropout(x, p=0.1, training=True)` | `random` | Random effect. Requires deterministic seed when used under `@tessera.jit(deterministic=True)`. |
| `tessera.ops.conv2d(x, weight, bias=None, stride=1, padding=0)` | `pure` | Phase 1 stub; Graph IR op exists as `tessera.conv2d_nhwc`. |
| `tessera.ops.flash_attn(Q, K, V, scale=None, causal=False)` | `pure` or `random` when dropout is present | Naive Python path in Phase 1; Phase 3 lowers supported SM_90+ paths to FA-4 Tile IR. |
| `tessera.ops.all_reduce(x, op="sum")` | `io` | Phase 1 no-op stub; NCCL/RCCL implementation is Phase 4 planned. |
| `tessera.ops.reduce_scatter(x, op="sum", axis=0)` | `io` | Phase 1 no-op stub; NCCL/RCCL implementation is Phase 4 planned. |
| `tessera.ops.all_gather(x, axis=0)` | `io` | Phase 1 no-op stub; NCCL/RCCL implementation is Phase 4 planned. |
| `tessera.ops.fused_epilogue(x, bias=None, activation="linear")` | `pure` | Applies bias and activation; compiler may also create fused epilogues through canonicalization. |

## Compiler Mapping

Operations move through the canonical four-layer stack:

```text
tessera.ops call
  -> Graph IR (`tessera.*` dialect)
  -> Schedule IR (`schedule.*` dialect)
  -> Tile IR (`tile.*`, `tessera.attn.*`, `tessera.queue.*`)
  -> Target IR / backend code
```

Not every operation reaches every layer in every backend. For example, the x86 path lowers supported matrix operations through `TilingPass` and `TileToX86Pass`, while the NVIDIA GPU path lowers supported FlashAttention and matmul forms through `TileIRLoweringPass` and the NVIDIA target passes.

## Effects

Tessera treats operation effects as compiler-visible behavior:

| Effect | Meaning | Examples |
|--------|---------|----------|
| `pure` | No side effects; recompute-safe | `gemm`, `softmax`, `gelu`, `relu`, `cast` |
| `random` | Uses RNG | `dropout` |
| `memory` | Reads/writes mutable state | Region writes, reductions, copies |
| `io` | Communication or host I/O | Collectives such as `all_reduce` |

Effects are inferred by the compiler. Users should not invent operation-level effect annotations in public examples.

## Canonicalization

Current Graph IR canonicalization focuses on a small set of implemented patterns:

| Pattern | Result |
|---------|--------|
| Matmul plus bias and/or GELU | `tessera.fused_epilogue` where supported |
| Convolution plus activation | Fused `tessera.conv2d_nhwc` epilogue where supported |
| Transpose feeding matmul | Matmul transpose attributes where legal |
| FlashAttention with zero dropout | Dropout attributes simplified away |

The exact pattern set is specified in `docs/spec/GRAPH_IR_SPEC.md` and `docs/spec/COMPILER_REFERENCE.md`.

## FlashAttention

Use `tessera.ops.flash_attn(Q, K, V, scale=None, causal=False)` for the public API. The Phase 3 NVIDIA path lowers supported SM_90+ configurations through FA-4 Tile IR:

- `tessera.attn.scaled_dot_product`
- `tessera.attn.causal_mask` when `causal=True`
- `tessera.attn.dropout_mask` when dropout is present
- `tessera.attn.online_softmax`
- `tessera.attn.lse_accumulate`

Tile sizes are controlled through `FlashAttnLoweringConfig` and are recorded on the Graph IR op as attributes such as `tessera.tile_q` and `tessera.tile_kv`.

## Distributed Operations

The public collective names exist in `tessera.ops`, but production distributed lowering is not current behavior:

| Operation | Current status |
|-----------|----------------|
| `all_reduce` | Phase 1 no-op stub; Phase 4 planned NCCL/RCCL lowering |
| `reduce_scatter` | Phase 1 no-op stub; Phase 4 planned NCCL/RCCL lowering |
| `all_gather` | Phase 1 no-op stub; Phase 4 planned NCCL/RCCL lowering |

Use `MockRankGroup` for current multi-rank tests without NCCL, MPI, or CUDA hardware.

## Authoritative References

| Topic | Document |
|-------|----------|
| Public operation signatures | `docs/spec/PYTHON_API_SPEC.md` |
| Graph IR op semantics | `docs/spec/GRAPH_IR_SPEC.md` |
| Lowering pipeline | `docs/spec/LOWERING_PIPELINE_SPEC.md` |
| Target/Tile IR dialects | `docs/spec/TARGET_IR_SPEC.md` |
| Canonical public API names | `docs/CANONICAL_API.md` |
