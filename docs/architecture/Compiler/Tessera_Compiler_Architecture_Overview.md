---
status: Informative
classification: Informative
last_updated: 2026-04-26
---

> **Phase status note:** Unless this document explicitly says otherwise, distributed collectives (NCCL/RCCL), TPU StableHLO, Cyclic distribution, autodiff transforms, activation checkpointing, ZeRO sharding, Bayesian autotuning, the runtime Python wrapper, production deployment, and NVL72 execution are Phase 4-6 planned as defined in `docs/README.md`. Current Phase 1-3 API names are defined in `docs/CANONICAL_API.md`.


# Tessera Compiler Architecture Overview
**Version:** 2.0  
**Date:** April 26, 2026  
**Status:** Informative — narrative companion to normative spec docs  
**Audience:** Compiler engineers, kernel authors, contributors beginning Phase 4+

---

## 0. Goals

- Provide a top-level view of the **Tessera compiler pipeline** from Python surface to binary.
- Show how frontend → Graph IR → Schedule IR → Tile IR → Target IR connects at each hand-off.
- Describe the responsibilities and contracts at each stage.
- Document what is **implemented** vs. **planned** as of Phase 3 completion.
- Align with the normative spec documents in `docs/spec/`.

**Not goals:** Runtime execution scheduling, NCCL/RCCL collective protocols, autotuner internals.

---

## 1. Pipeline Overview

```text
Python surface (@tessera.jit, @tessera.kernel, Region[...], tessera.domain, index_launch)
    │
    ▼
@jit decoration time:
    ├── _ConstraintExtractor   (AST parser — extracts tessera.require(...) calls)
    ├── ConstraintSolver       (checks Divisible/Range/Equal against bindings)
    ├── EffectLattice          (infers pure/random/memory/io from ops in body)
    └── GraphIRBuilder         (emits tessera dialect MLIR text)
    │
    ▼
Graph IR  (tessera dialect — matmul, conv2d_nhwc, flash_attn, fused_epilogue, transpose, cast)
    │
    ├── EffectAnnotationPass        (annotates func.func with tessera.effect)
    ├── CanonicalizeTesseraIRPass   (4 fusion patterns — greedy fixed point)
    └── DistributionLoweringPass    (tessera.shard attrs → schedule.mesh.*)
    │
    ▼
Schedule IR  (schedule dialect — mesh.define, mesh.region, pipeline.region, stage, yield)
    │
    ├── [x86 path]  TilingPass → TileToX86Pass
    │                tessera.matmul → scf.for loops → func.call @tessera_x86_amx_gemm_bf16
    │
    └── [GPU path]  TileIRLoweringPass
                    flash_attn → tile.async_copy + tessera.attn.* + tile.mma
    │
    ▼
Tile IR  (tile.* ops + tessera.attn.* FA-4 ops + tessera.queue.* barriers)
    │
    ├── WarpSpecializationPass    (producer/consumer warp roles + queue barriers)
    ├── AsyncCopyLoweringPass     (tile.async_copy → tessera.tma.* or cp.async)
    ├── NVWGMMALoweringPass       (tile.mma → wgmma.mma_async PTX or WMMA)
    ├── NVTMADescriptorPass       (TMA descriptor hoisting + mbarrier init)
    └── NVFlashAttnKernelEmitter  (scale resolution, full mbarrier seq, launch bounds)
    │
    ▼
Target IR  (tessera.nvgpu.wgmma.*, tessera.tma.*, mbarrier ops)
    │
    └── LLVM NVPTX backend → PTX binary
```

---

## 2. Frontend: Python Surface to Graph IR

### 2.1 What the frontend is

The Tessera frontend is **pure Python** — no separate parser, no Rust layer, no domain-specific language file format. The Python interpreter runs normally; `@tessera.jit` intercepts the decorated function at decoration time through Python's standard decorator protocol.

This is a deliberate architectural decision. The Python surface is a clean API, not a separate language. See `docs/old_concepts/Rust_Frontend_Research/README.md` for the rejected alternative.

### 2.2 `@tessera.jit` decoration sequence

When Python evaluates `@tessera.jit def fn(...)`:

1. **Signature inspection.** The decorator reads `fn.__annotations__` to discover `Region[...]` and `Tensor[...]` parameters. Each `Region["read"]`, `Region["write"]`, `Region["reduce_sum"]`, etc. produces a `RegionType` object that carries `mode`, `exclusive`, and `reduces` fields.

2. **AST extraction.** `_ConstraintExtractor` (an `ast.NodeVisitor`) walks the function's AST to find `tessera.require(...)` calls and extract the constraint objects. This happens at decoration time — before the function is ever called.

3. **Constraint checking.** If `bindings=` was provided (e.g. `{"K": 128}`), `ConstraintSolver.check_all(bindings)` is called immediately. A `Divisible("K", 64)` constraint against `bindings={"K": 100}` raises `TesseraConstraintError` at decoration time, not at call time.

4. **Effect inference.** `EffectLattice` walks the function body's ops (via the same AST) and infers the function's effect level: `pure` < `random` < `memory` < `io` < `top`. If `@jit(deterministic=True)` is set and the inferred effect is `random` without a `seed`, `TesseraEffectError` is raised.

5. **Graph IR emission.** `GraphIRBuilder` emits a `module` containing a `func.func` with:
   - Argument tensors annotated with `tessera.shard` attributes (from `DistributedArray` parameters)
   - Argument tensors annotated with `tessera.effect` attributes (from `Region[...]` parameters)
   - Body ops: `tessera.matmul`, `tessera.flash_attn`, `tessera.gelu`, `tessera.cast`, etc.
   - Module-level `tessera.version = "1.0"` attribute

6. **`JitFn` wrapper.** The decorator returns a `JitFn` instance that wraps the original function and exposes `.graph_ir`, `.effect`, `.constraints`, and `.target` attributes.

### 2.3 Canonical decorator signatures

```python
# No-argument form
@tessera.jit
def fn(W: tessera.Region["read"], X: tessera.Region["read"], Y: tessera.Region["write"]):
    Y[:] = tessera.ops.gemm(X, W)

# Full keyword-argument form
@tessera.jit(
    deterministic=True,
    seed=42,
    bindings={"K": 128, "M": 512},
    target=GPUTargetProfile(isa=ISA.SM_90, warps_per_cta=4),
    attn_config=FlashAttnLoweringConfig(tile_q=64, tile_kv=64, causal=True),
)
def flash_forward(Q, K, V):
    tessera.require(tessera.constraint.Divisible("D", 64))
    return tessera.ops.flash_attn(Q, K, V, causal=True)
```

**Important:** legacy staged IR inspection helpers, `@autodiff`, `@vmap`, `@pmap`, `@scan`, `@checkpoint` are **not part of the current API**. The working inspection mechanism is `fn.graph_ir.to_mlir()`. See `docs/CANONICAL_API.md` for the complete authoritative API.

### 2.4 Region privileges

`Region[mode]` is a **type annotation only** — it does not wrap tensors at runtime. It lowers to `tessera.effect` attributes on Graph IR function arguments. The `@jit` decorator inspects annotations at decoration time to enforce privilege contracts:

| Annotation | `tessera.effect` on IR arg | Exclusive |
|-----------|---------------------------|-----------|
| `Region["read"]` | `"read"` | No |
| `Region["write"]` | `"write"` | Yes — conflict with any other write |
| `Region["reduce_sum"]` | `"reduce_sum"` | No |
| `Region["reduce_max"]` | `"reduce_max"` | No |
| `Region["reduce_min"]` | `"reduce_min"` | No |

Two `Region["write"]` parameters on overlapping tensors → `TesseraPrivilegeError` at decoration time.

### 2.5 Domain and distribution

`tessera.domain.Rect` and `tessera.dist.Block/Cyclic/Replicated` are **always separate** — shape vs. placement. `tessera.array.from_domain` creates a `DistributedArray` carrying a `ShardSpec`.

```python
D    = tessera.domain.Rect((4, 128, 256))            # shape only
dist = tessera.dist.Block(mesh_axes=("dp", "tp"))    # placement only
X    = tessera.array.from_domain(D, dtype="bf16", distribution=dist)
# X.shard_spec → ShardSpec(partition=(0, 1), mesh_axes=("dp", "tp"))
# X.parts("dp") → list of per-rank slices
```

`ShardSpec` is what `GraphIRBuilder` encodes as `tessera.shard` attributes on function arguments.

---

## 3. Graph IR

### 3.1 What Graph IR encodes

Graph IR (the `tessera` MLIR dialect) encodes the **mathematical intent** of a computation. It is backend-agnostic — the same Graph IR module can lower to x86 AMX or NVIDIA WGMMA.

Key properties:
- All ops carry `Pure` + `NoSideEffect` traits (except `tessera.copy`). The effect system is expressed as function-level attributes, not op-level side effects.
- Shapes are static where known; dynamic shapes are supported via symbolic dims.
- Distribution metadata lives in `tessera.shard` argument attributes, not in op bodies.

### 3.2 Op catalog (summary)

| Op | Purpose |
|----|---------|
| `tessera.matmul` | Matrix multiply. Supports `transposeA`/`transposeB` flags, `tile_k` hint, `TilingInterface`. |
| `tessera.conv2d_nhwc` | 2D convolution (NHWC layout). Supports fused `epilogue` attr. |
| `tessera.flash_attn` | Fused FlashAttention. Carries tile size attrs for Phase 5 autotuner. |
| `tessera.fused_epilogue` | Fused matmul + optional bias + activation. Generated by canonicalization. |
| `tessera.transpose` | Tensor transpose. Usually canonicalized away into `matmul` transpose flags. |
| `tessera.cast` | Element type cast (e.g. `bf16 → f32`). |

Full op catalog with verifier rules and MLIR examples: `docs/spec/GRAPH_IR_SPEC.md`.

### 3.3 Canonicalization patterns

Four greedy patterns run via `CanonicalizeTesseraIRPass`:

| Pattern | Fires on | Result |
|---------|----------|--------|
| `FuseMatmulBiasGELU` | `gelu(add(matmul(A,B), bias))` | Single `fused_epilogue {Gelu}` |
| `FuseConvRelu` | `relu(conv2d_nhwc(...))` | `conv2d_nhwc {epilogue=Relu}` |
| `DropoutZeroSimplify` | `flash_attn {dropout_p=0.0}` | `flash_attn` without `dropout_p` attr |
| `TransposeIntoMatmul` | `matmul(transpose(A), B)` etc. | `matmul(A, B, transposeA=true)` |

### 3.4 Effect annotation

`EffectAnnotationPass` attaches `tessera.effect` as a string attribute on each `func.func`. Inference rules (in order):

1. `flash_attn` with `dropout_p != 0.0` → `"random"`
2. `tessera.copy` op in body → `"memory"`
3. Any argument with `tessera.effect = "write"` or `"reduce_*"` → `"memory"`
4. `func.call` to external function → `"io"`
5. None of the above → `"pure"`

If `@jit(deterministic=True)` was set and the inferred effect is `"random"`, the pass signals failure.

---

## 4. Schedule IR

### 4.1 What Schedule IR encodes

Schedule IR (the `schedule` MLIR dialect) encodes **where** computation runs (mesh axis placement) and **when** (pipeline staging). It bridges Graph IR (backend-agnostic) and Tile IR (hardware-specific).

`DistributionLoweringPass` is the pass that converts Graph IR → Schedule IR by reading `tessera.shard` argument attributes and emitting `schedule.mesh.define` + `schedule.mesh.region` wrappers.

### 4.2 Key ops

| Op | Purpose |
|----|---------|
| `schedule.mesh.define` | Declares a logical device mesh (dims + axis names). |
| `schedule.mesh.region` | Wraps computation to execute across one mesh axis. |
| `schedule.pipeline.region` | Groups computation into pipeline-parallel stages (Phase 4 lowering). |
| `schedule.stage` | One stage in a pipeline region (Phase 4). |
| `schedule.yield` | Required terminator for all schedule region bodies. |

Schedule IR spec: `docs/spec/TARGET_IR_SPEC.md §2`.

---

## 5. Tile IR

### 5.1 What Tile IR encodes

Tile IR encodes **explicit tile operations** — async data movement, MMA compute, and synchronisation barriers. It is the first IR layer that knows about hardware tile sizes and warp specialization.

`TileIRLoweringPass` is the pass that converts Schedule IR → Tile IR for the GPU path by expanding `tessera.flash_attn` inside `schedule.mesh.region` bodies into FA-4 tile ops.

### 5.2 FA-4 FlashAttention op sequence

The FA-2 online softmax algorithm in Tile IR:

```
Outer loop over Q tiles:
  init running_m = -inf, running_l = 0, acc_out = 0
  Inner loop over KV tiles:
    %scores = tessera.attn.scaled_dot_product Q_tile, K_tile, scale
    [%masked = tessera.attn.causal_mask %scores ...]      ← if causal=true
    [%masked = tessera.attn.dropout_mask %masked ...]     ← if dropout_p > 0
    %new_acc, %new_m, %new_l = tessera.attn.online_softmax %masked, ...
  End inner loop
  %output, %lse = tessera.attn.lse_accumulate %acc_out, %final_m, %final_l
```

Key design decisions:
- **Online softmax is the FA-2 algorithm.** Running max + running sum with correction factor. Never a batch softmax — that OOMs on long sequences.
- **Tile sizes are stored as op attributes** (`tessera.tile_q`, `tessera.tile_kv`) so the Phase 5 autotuner can sweep them without re-emitting Graph IR.

### 5.3 Warp specialization

`WarpSpecializationPass` splits the kernel body into `tessera.schedule.warp {role="producer"}` and `role="consumer"` regions. This is a **structural separation** — the backend allocates separate register files and mbarrier slots per role.

- **Producer** warps: run `tile.async_copy` + `tile.wait_async` (TMA prefetch)
- **Consumer** warps: run `tessera.attn.*` compute ops + `tile.mma`
- `tessera.queue.push/pop` ops at the boundary express the handoff ordering

Tile IR spec: `docs/spec/TARGET_IR_SPEC.md §3–5`.

---

## 6. Target IR (NVIDIA)

### 6.1 What Target IR encodes

Target IR is fully hardware-specific. For NVIDIA SM_90+, it consists of:

- `tessera.tma.*` — TMA descriptor-based async tile copies
- `tessera.nvgpu.wgmma.mma_async` — WGMMA PTX inline asm
- `tessera.mbarrier.*` — mbarrier arrive/wait synchronisation
- `nvvm.kernel`, `nvvm.maxntidx` — CUDA kernel launch metadata

### 6.2 Key design decisions

1. **TMA descriptors are generated once per kernel, not per tile.** `NVTMADescriptorPass` hoists descriptor setup to the kernel preamble. The tile loop calls `cp.async.bulk.tensor` referencing the descriptor.

2. **WGMMA is gated behind `isa >= SM_90`.** `NVWGMMALoweringPass` falls back to legacy WMMA for SM_80/86/89.

3. **SM_90 default tile sizes: `tile_q=64, tile_kv=64, pipeline_stages=2`.** These match the WGMMA 64×64 tile granularity and provide a 2-stage software pipeline. The Phase 5 autotuner sweeps these.

4. **The `scale` sentinel in `scaled_dot_product` is resolved by `NVFlashAttnKernelEmitter`.** The value `-1.0` means "auto-compute as `1/sqrt(head_dim)`" and is replaced with the concrete float before PTX emission.

Target IR spec: `docs/spec/TARGET_IR_SPEC.md §6`.

---

## 7. Named Lowering Pipelines

### `tessera-lower-to-x86` (Phase 2)

```
EffectAnnotationPass
→ CanonicalizeTesseraIRPass
→ DistributionLoweringPass
→ TilingPass        (--tile-m, --tile-n)
→ TileToX86Pass     (--prefer-amx)
```

Output: MLIR module with `func.call @tessera_x86_amx_gemm_bf16` calls, ready for LLVM x86 backend.

### `tessera-lower-to-gpu` (Phase 3)

```
EffectAnnotationPass
→ CanonicalizeTesseraIRPass
→ DistributionLoweringPass
→ TileIRLoweringPass          (--tile-q, --tile-kv, --sm)
→ WarpSpecializationPass
→ AsyncCopyLoweringPass
→ NVWGMMALoweringPass
→ NVTMADescriptorPass
→ NVFlashAttnKernelEmitter
```

Output: MLIR module with `nvvm.kernel` functions containing WGMMA PTX intrinsics, ready for LLVM NVPTX backend.

Full pass-by-pass specification: `docs/spec/LOWERING_PIPELINE_SPEC.md`.

---

## 8. Test Suite

| Suite | Location | Covers |
|-------|----------|--------|
| Phase 1 Python tests | `tests/phase1/` | Distributed API, constraints, effects, Graph IR |
| Phase 2 Python tests | `tests/phase2/test_lowering_chain.py` | x86 lowering pipeline |
| Phase 2 lit tests | `tests/tessera-ir/phase2/` | MLIR lowering correctness |
| Phase 3 Python tests | `tests/phase3/` | GPU target, flash attn, warp specialization (56 tests) |
| Phase 3 lit tests | `tests/tessera-ir/phase3/` | Tile IR lowering, WGMMA, FA-4 |

---

## 9. Phase Completion and Roadmap

| Phase | Status | Key remaining work |
|-------|--------|-------------------|
| 1 | ✅ Complete | — |
| 2 | ✅ Complete | — |
| 3 | ✅ Complete | — |
| 4 | 🔲 Next | NCCL/RCCL adapters, `GPUCollectiveInsertionPass`, `PipelineStageInsertionPass`, TPU StableHLO backend, `Cyclic` distribution full implementation |
| 5 | 🔲 Future | `InsertRecomputePass`, `OptimizerShardPass`, `BayesianAutotuner`, sparse/RNG solvers |
| 6 | 🔲 Future | ROCm MFMA full coverage, Runtime C ABI Python wrapper, benchmark suite |

---

## 10. Authoritative References

| Topic | Document |
|-------|----------|
| Single-page API naming authority | `docs/CANONICAL_API.md` |
| IR stack, pass registry, phase status | `docs/spec/COMPILER_REFERENCE.md` |
| All Python public symbols with signatures | `docs/spec/PYTHON_API_SPEC.md` |
| Graph IR op catalog + canonicalization patterns | `docs/spec/GRAPH_IR_SPEC.md` |
| Every pass with input/output IR and invariants | `docs/spec/LOWERING_PIPELINE_SPEC.md` |
| Schedule + Attn + Queue + tile.* dialects | `docs/spec/TARGET_IR_SPEC.md` |
| Programmer system overview with what-works table | `docs/architecture/system_overview.md` |
