---
status: Informative
classification: Informative
last_updated: 2026-05-11
---

# Tessera API Reference

This reference summarizes the current public API shape. The authoritative API
specification is `docs/spec/PYTHON_API_SPEC.md`; if this guide disagrees with
that spec, the spec wins. Tensor attribute and dtype vocabulary lives in
`docs/reference/tessera_tensor_attributes.md`.

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

Tensor attributes are split across logical shape (`shape`), storage dtype
(`dtype`), layout (`layout`), execution target (`target`), distribution
(`ShardSpec`), and numeric policy (`numeric_policy`). Dtype strings should use
the canonical names in `docs/reference/tessera_tensor_attributes.md`; aliases
such as `"f32"` normalize before storage.

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
| `conv2d` | Implemented — NHWC + NCHW Module forms (`tessera.nn.Conv2d` / `Conv2dNCHW`); Graph IR op + VJP/JVP registered. |
| `flash_attn` | Phase 1 naive path; Phase 3 SM_90+ FA-4 lowering path |
| `all_reduce`, `reduce_scatter`, `all_gather` | Implemented — Phase 4 distributed lowering (`GPUCollectiveInsertionPass`); NCCL/RCCL adapters wired; VJP+JVP registered for all four collectives. |
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

The compiler exposes all four IR layers as inspectable objects on the
JIT artifact:

```python
print(flash_fwd.graph_ir.to_mlir())
print(flash_fwd.schedule_ir)   # Schedule IR (mesh / pipeline)
print(flash_fwd.tile_ir)       # Tile IR (warps / TMA / async_copy)
print(flash_fwd.target_ir)     # per-target final IR
```

See `examples/compiler/ir_pipeline_tutorial/tessera_ir_pipeline_demo.py`
for a runnable walkthrough that prints all four layers for a tiny MLP.
For static inspection without launching, `tessera-mlir
--mode=compile_artifact --symbol=<name>` reads the JIT artifact directly
(see `docs/guides/Tessera_Debugging_Tools_Guide.md`).

## Roadmap (formerly "Future APIs")

These rows were authored when Phases 4–6 were planned.  The status
below reflects the post-Phase-8 reality (Apple operational; Cerebras /
Metalium / Apple backends shipped under Phase 7-8; the S-series
standalone-compiler track shipped S0–S15 + autodiff coverage).  For
the current per-component picture, read
`docs/audit/generated/support_table.md` (drift-gated) rather than this
table.

| Area | Status |
|------|--------|
| NCCL/RCCL collectives + cluster execution | Implemented (Phase 4) — `GPUCollectiveInsertionPass`, NCCL/RCCL adapters, `ChunkPlanner`, `CollectiveScheduler`. |
| TPU StableHLO backend | Implemented — `tpu_target.py` + `Tessera_TPU_Backend/` (StableHLO + Shardy export); quantized dot lit-tested. |
| Autodiff transforms + custom VJP/JVP | Implemented — `tessera.autodiff` v1 ships 241 VJPs + 236 JVPs; `tessera.custom.custom_vjp` / `custom_jvp` user-facing. |
| Activation checkpointing + ZeRO sharding | Implemented — `tessera.autodiff.rematerialize` + ZeRO stage 2 via `OptimizerShardPass`. |
| Bayesian autotuning | Implemented — `tessera.autotune` + `compiler/autotune_v2.py` (Optuna TPE + Hyperband + SQLite cache v2). |
| Runtime Python wrapper | Implemented — `tessera.runtime.TesseraRuntime` over the C ABI. |
| ROCm MFMA + RubinCPX + Cerebras + Metalium + Apple backends | Implemented (Phase 7–8). |
