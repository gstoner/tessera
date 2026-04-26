---
status: Tutorial
classification: Tutorial
last_updated: 2026-04-26
---

# FlashAttention In Tessera

This tutorial shows the current public way to express FlashAttention in Tessera. The public API is `tessera.ops.flash_attn`; the compiler lowers supported SM_90+ configurations through the Phase 3 FA-4 Tile IR path.

For exact API details, see `docs/spec/PYTHON_API_SPEC.md`. For the lowering pipeline and dialect details, see `docs/spec/LOWERING_PIPELINE_SPEC.md` and `docs/spec/TARGET_IR_SPEC.md`.

## Minimal FlashAttention Function

```python
import tessera

@tessera.jit
def flash_fwd(Q, K, V):
    return tessera.ops.flash_attn(Q, K, V, causal=True)
```

Inspect the generated Graph IR:

```python
print(flash_fwd.graph_ir.to_mlir())
```

Graph IR inspection is current. Tile IR and Target IR inspection helpers are Phase 4 planned.

## Targeting NVIDIA SM_90+

```python
import tessera
from tessera.compiler.gpu_target import GPUTargetProfile, ISA
from tessera.compiler.attn_lower import FlashAttnLoweringConfig

@tessera.jit(
    target=GPUTargetProfile(isa=ISA.SM_90, warps_per_cta=4),
    attn_config=FlashAttnLoweringConfig(
        tile_q=64,
        tile_kv=64,
        pipeline_stages=2,
        causal=True,
    ),
)
def flash_fwd_sm90(Q, K, V):
    return tessera.ops.flash_attn(Q, K, V, causal=True)
```

The compiler records tile sizes as Graph IR attributes and, on the GPU path, lowers supported forms through:

- `TileIRLoweringPass`
- `WarpSpecializationPass`
- `AsyncCopyLoweringPass`
- `NVWGMMALoweringPass`
- `NVTMADescriptorPass`
- `NVFlashAttnKernelEmitter`

## What The Compiler Generates

At a high level, the supported GPU path expands the single Graph IR operation into FA-4 Tile IR operations:

```text
tessera.ops.flash_attn
  -> tessera.flash_attn in Graph IR
  -> tessera.attn.scaled_dot_product
  -> tessera.attn.causal_mask        when causal=True
  -> tessera.attn.dropout_mask       when dropout is present
  -> tessera.attn.online_softmax
  -> tessera.attn.lse_accumulate
  -> tile.mma / async copy / queue synchronization
  -> NVIDIA TMA / WGMMA / mbarrier Target IR
```

## Determinism And Dropout

Dropout introduces a random effect. Use deterministic settings and a seed when random behavior is part of a compiled function.

```python
@tessera.jit(deterministic=True, seed=42)
def flash_fwd_dropout(Q, K, V):
    return tessera.ops.flash_attn(Q, K, V, causal=True)
```

The exact dropout parameters supported by `flash_attn` are defined in `docs/spec/PYTHON_API_SPEC.md`.

## Current Versus Future Examples

Current public examples should call `tessera.ops.flash_attn` rather than hand-writing tile-level FlashAttention kernels. Hand-authored warp-specialized FlashAttention kernels, custom autotune decorators, and staged Tile/Target IR inspection are future or internal compiler topics unless a canonical spec marks them public.

| Topic | Status |
|-------|--------|
| Public `tessera.ops.flash_attn` call | Phase 1-3 implemented |
| SM_90+ FA-4 lowering path | Phase 3 implemented |
| Manual public tile-kernel FlashAttention DSL | Not current public API |
| Tile/Target IR inspection helpers | Phase 4 planned |
| Bayesian autotuning for tile sizes | Phase 5 planned |
