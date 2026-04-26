---
status: Tutorial
classification: Tutorial
last_updated: 2026-04-26
---

> **Phase status note:** Unless this document explicitly says otherwise, distributed collectives (NCCL/RCCL), TPU StableHLO, Cyclic distribution, autodiff transforms, activation checkpointing, ZeRO sharding, Bayesian autotuning, the runtime Python wrapper, production deployment, and NVL72 execution are Phase 4-6 planned as defined in `docs/README.md`. Current Phase 1-3 API names are defined in `docs/CANONICAL_API.md`.


# Goals

One stack for research → prod: modeling, kernels, distributed, deployment.
Tiles, not threads at the kernel level; autotuning is built-in.
Numerics, data movement, state, and parallelism are first-class.

# Core principles

•	Single source of truth: one language for modeling, training, serving, and kernels (no C++ sidecar).
•	Multi-level IRs: Graph IR (autodiff & algebra/MLIR), Schedule IR (fusion/tiling/pipeline/MLIR), Tile IR (blocks/warps/TensorCores/MLIR), Target IR (PTX/MLIR/LLVM).
•	Auto & manual: sane defaults with first-party autotuning, but everything overridable with explicit schedules.
•	Portability: CPU, NVIDIA/AMD/Intel GPUs, TPUs/NPUs, and WebGPU—no vendor lock-in.
•	Determinism → performance dial: reproducible by default; opt in to “fast-math/async nondet” modes.

2) The high-level modeling language

•	Pythonic surface (or identical to Python) with:
•	First-class autodiff (forward & reverse), custom VJP/JVP, checkpointing knobs.
•	Shape & dtype polymorphism (Tensor[S, D, f16] with S symbolic); static checks + runtime guards.
•	Effects for randomness, IO, and distributed collectives (so the compiler can move/merge them safely).
•	Mixed precision policies as types (e.g., fp8 in, bf16 accum, fp32 master).
•	Distributed by construction: data/tensor/pipeline/expert parallel expressed declaratively (with mesh(devs).shard(W{"d_model"})).
•	Composability: function transforms (jit, vmap, pmap, scan, checkpoint, remat, alpa_shard).
•	Graph optimizations baked in:
•	Op fusion, rematerialization, activation offload, memory planning with liveness.
•	Numerics guards (NaN/Inf sentinels, loss-scales, safe softmax/attention).
•	Interoperability: drop-in tensors for NumPy/PyTorch, ONNX import/export, custom ops bind straight to the kernel layer.

3) The kernel/scheduling language (“tiles, not threads”)

•	Tile-first SPMD DSL (a la Triton/Warp), with:
•	tile.load/store, dot/mma, softmax_online, collective intrinsics.
•	Explicit memory spaces (global/shared/register) but auto pipelining (prefetch, stages, cp.async).
•	Tensor Core/MatrixCore lowering by default; vector widths inferred, overridable.
•	Halide-style schedules you can attach or auto-discover:
•	block(BM, BN).k(BK).warps(8).stages(3).vector(8).swizzle("xor")
•	Autotuner built-in with cost models + on-device measurements, persistent caches per shape/arch.
•	Safety + perf tooling: race checker (shared/global), bank-conflict linter, register-pressure estimator, roofline hints.

4) Compiler/Runtime must-haves

•	Multi-target backends: NVIDIA (PTX/TileIR/CuTE/CuTLASS),  CPU/AMX; pluggable vendor passes.
•	Async runtime: streams/graphs, collective ops (NVIDIA NCCL,  CPU oneCCL), ZeRO-style sharding, FSDP, KV-cache manager.
•	Observability: traces, NVTX ranges, flamegraphs, tensor-level memory timelines; perf-regressions caught in CI.
•	Repro & packaging: seed discipline, graph versioning, artifact registry (fatbins + schedule cache), AOT bundles for server/mobile/web.

# Shape Debugging System
The Tessera_Shape_System_Package suggests a comprehensive approach to compile-time shape verification and runtime debugging:
Expected Features:

# Static Shape Analysis - Compile-time verification of tensor dimensions
Shape Inference - Automatic deduction of output shapes from input shapes
Symbolic Dimensions - Support for batch sizes and sequence lengths as symbols
Runtime Shape Tracking - Debugging tools to trace shape transformations
Error Diagnostics - Clear error messages when shape mismatches occur


# Core Types & Numerics

- Tensor[Dims…, DType @policy…] where policies include:
@accum(f32|bf16), @stochastic_round, @loss_scale(k), @saturate.
- Supported dtypes: fp32, tf32, bf16, fp16, fp8(e4m3|e5m2), int8/4 (+ per-channel scales).
- Safe primitives: softmax_safe, layernorm_safe, rmsnorm_safe, logsumexp_safe.
- Casting rules are typed (e.g., fp8 matmul → fp32 accumulate → policy cast).

# State Objects

- KVCache["B","H","S_max","D_h", dtype] @rolling_window @device(gpu)
- OptimizerState[param] @fp32_master @paged @checkpointable
- State ops are effects (append, prune, offload, prefetch).

# Effects & Placement

- with device(gpu[n]), prefetch(obj, into="smem", overlap="compute"): Declarative, inspectable data movement (maps to DMA/TMA, double-buffering).
- on mesh(devs) / with mesh(..., layout=...) sets placement & sharding.

# Parallelism (Algebraic)

- map, reduce, scan, all_gather, all_reduce, all_to_all, pipeline.
- pmap(fn, donate={...}) and vmap, scan are composable transforms.
- Mesh layouts: {"batch":"replicate", "seq":"shard", "params":"replicate"}.

# Kernel DSL (Tile-First)

- Intrinsics: tile.load/store, dot/mma, softmax_online, transpose, broadcast.
- Memory spaces: global/shared/register with hints (vector=8, stages=3, swizzle="xor").
- Attention helpers: mask_causal, softmax_accumulate, fused dropout option.

# Scheduling & Autotuning

- First-class Schedule IR:
.block(m=[64,128], n=[64,128], k=[64]) .warps([4,8]) .stages([2,3]) .vector([4,8])
- Phase 4 planned: canonical schedule/autotune API for metric-driven tuning and schedule-cache artifacts.
- Persistent schedule cache keyed by shapes, dtypes, arch; export/import artifacts.

# IR Stack

- Graph IR (autodiff, algebra, effects) → Schedule IR (fusion, tiling, pipelining) → Tile IR (tiles, collectives, memory ops) → Target IR (PTX/NVIDA TILE IR/LLVM/MLIR).
- All IRs are open & versioned; hashes define reproducible builds.

# Autodiff

- Built-in forward/reverse AD with custom vjp/jvp, rematerialization, and checkpointing.
- Effect-aware AD (respects state ops and collectives).

# Distribution Runtime

- Generates NCCL/oneCCL or in-network reductions automatically.
- Async graphs/streams; overlap comm/compute by default; deterministic mode available.

# Safety, Debugging, Observability

- Debug builds: bounds-checked slices, race detector (shared/global), NaN/Inf sentinels.
- Profiling: NVTX-style ranges, kernel timelines, roofline hints, autotune traces.
- “Lowering inspector” to view Graph→Schedule→Tile transformations.

# Deployment

- AOT bundles: Graph IR + tuned schedules + fatbins; per-arch variants.
- Deterministic replay (ordered reductions, fixed tie-breakers).
- ONNX import/export; PyTorch/JAX interop shims for transition.

# Example Snippets

## Modeling:
```python
@tessera.jit
def transformer_block(x, Wqkv, Wo, Wmlp_in, Wmlp_out, *, heads: int):
    h = tessera.ops.rmsnorm_safe(x)
    qkv = tessera.ops.gemm(h, Wqkv)
    q, k, v = tessera.ops.split_qkv(qkv, heads=heads)
    y = tessera.ops.flash_attn(q, k, v, causal=True)
    y = x + tessera.ops.gemm(y, Wo)
    z = tessera.ops.gelu(tessera.ops.gemm(tessera.ops.rmsnorm_safe(y), Wmlp_in))
    return y + tessera.ops.gemm(z, Wmlp_out)
```

Autodiff transforms, managed KV-cache APIs, dropout lowering, and checkpointing for this block are Phase 5 planned.

## Kernel + schedule:

```python
@tessera.jit
def attention(q, k, v):
    return tessera.ops.flash_attn(q, k, v, causal=True)

graph_ir_text = attention.graph_ir.to_mlir()
```

Custom kernel schedules and Bayesian autotuning are Phase 4 planned; current examples should prefer canonical `tessera.ops` calls unless a Tile IR lowering example is explicitly required.

# Cost-model APIs (hybrid analytic + learned) for autotune warm-starts.
Static guarantees for effect ordering across collectives (type/effect system).
Standardized artifact format for schedule caches across vendors.
Formal semantics for numerics policies (rounding, overflow, determinism).

# Add Verification Hooks:

Numerics verification decorators are Phase 5 planned:

```python
# Phase 5 planned
# @tessera.verify.numerics(stable=True, bounded=[-100, 100])
# def attention_score(q, k):
#     return tessera.ops.gemm(q, tessera.ops.transpose(k))
```

Explicit Schedule Inheritance:

Schedule inheritance is Phase 4 planned. Keep examples marked as future until the API is canonical.

Add Standard Library
Ship with pre-tuned implementations of common operations:

```python
y = tessera.ops.flash_attn(q, k, v, causal=True)
z = tessera.ops.rmsnorm_safe(y)
```
