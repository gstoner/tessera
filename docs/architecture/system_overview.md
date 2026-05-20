---
status: Informative
classification: Informative
last_updated: 2026-05-20
---

> **Current-state note (2026-05-20):** This is historical architecture guidance. Phase labels below are design lineage, not current support claims. For implementation status, use `docs/spec/COMPILER_REFERENCE.md`, `docs/audit/generated/support_table.md`, `docs/audit/generated/e2e_op_coverage.md`, and `docs/spec/VALIDATION_SPINE.md`.


# Tessera System Overview
**Version:** 2.0  
**Date:** April 26, 2026  
**Status:** Informative — narrative companion to normative spec docs in `docs/spec/`  
**Audience:** New contributors, compiler engineers, ML engineers evaluating Tessera

---

## What Tessera Is

Tessera is a **pre-alpha, tile-centric programming model and compiler** for deep learning and HPC. The design goal is to make tiles, explicit memory spaces, numerical precision, and parallelism **first-class IR objects** rather than runtime heuristics.

The Python surface (`@tessera.jit`, `Region[...]`, `tessera.domain`) provides a clean, declarative API. Below it, a four-layer MLIR IR stack provides rigorous semantics at each lowering stage. The compiler is **static and AOT** — there is no tracing JIT.

Target hardware: NVIDIA (SM90 Hopper, SM100 Blackwell), AMD ROCm, Google TPU, Cerebras WSE-3, Tenstorrent Metalium, and x86 AMX/AVX-512.

---

## Capability Status

| Area | Current status |
|------|----------------|
| **Python frontend** | Implemented: `@tessera.jit`, `@tessera.kernel`, `fn.explain()`, `from_text`, support queries, constraints, effects, and Graph IR emission. |
| **IR stack** | Graph/Schedule/Tile/Target IR are implemented and inspectable; metadata normalization and constrained-lane Graph IR views are in place. |
| **CPU / x86 lowering** | Implemented for supported paths, with reference execution and AMX/AVX-512-oriented backend hooks. |
| **NVIDIA GPU lowering** | Tile/Target artifact path implemented for supported kernels; runtime/native readiness is target- and op-gated. |
| **Apple GPU native path** | Fused native kernels exist for the audited GA/EBM and selected Visual Complex surfaces. |
| **ROCm** | MFMA/WMMA capability modeling is present, including `gfx1200`; native execution is still capability-gated. |
| **Distributed / collectives** | Lowering and adapter surfaces exist; production multi-rank execution is validation-gated. |
| **Runtime ABI** | C ABI and Python wrapper exist, with runtime smoke and sanitizer coverage. |

---

## Four-Layer IR Stack

```
Python API  (@tessera.jit, @tessera.kernel, Region[...], tessera.domain, index_launch)
     │  [GraphIRBuilder — python/tessera/compiler/graph_ir.py]
     ▼
Graph IR    (tessera dialect — matmul, conv2d_nhwc, flash_attn, fused_epilogue, cast, transpose)
     │  [EffectAnnotationPass → CanonicalizeTesseraIRPass → DistributionLoweringPass]
     ▼
Schedule IR (schedule.* dialect — mesh.define, mesh.region, pipeline.region, stage, yield)
     │  [TilingPass + TileToX86Pass]       ← x86 path
     │  [TileIRLoweringPass]               ← GPU path
     ▼
Tile IR     (tile.* ops + tessera.attn.* FA-4 ops + tessera.queue.* barriers)
     │  [WarpSpecializationPass → AsyncCopyLoweringPass → NVWGMMALoweringPass → NVTMADescriptorPass → NVFlashAttnKernelEmitter]
     ▼
Target IR   (tessera.nvgpu.wgmma.*, tessera.tma.*, mbarrier ops → LLVM NVPTX → PTX)
             [x86: func.call @tessera_x86_amx_gemm_bf16 via TileToX86Pass]
```

Each layer has its own spec document:
- Graph IR: `docs/spec/GRAPH_IR_SPEC.md`
- Lowering pipeline: `docs/spec/LOWERING_PIPELINE_SPEC.md`
- Target IR: `docs/spec/TARGET_IR_SPEC.md`

---

## What Works Today

The section names below preserve the original compiler walkthrough order but
avoid using phase labels as status claims. For current readiness, prefer the
generated support and E2E dashboards.

### Python Frontend

| Component | Status | Location |
|-----------|--------|----------|
| `@tessera.jit` decorator — ConstraintSolver, EffectLattice, GraphIRBuilder | ✅ | `python/tessera/compiler/jit.py` |
| `@tessera.kernel` decorator + `KernelFn` | ✅ | `python/tessera/distributed/launch.py` |
| `tessera.Region["read"/"write"/"reduce_sum"/"reduce_max"/"reduce_min"]` | ✅ | `python/tessera/distributed/region.py` |
| `tessera.domain.Rect` + `tessera.dist.Block/Replicated` | ✅ | `python/tessera/distributed/domain.py` |
| `tessera.array.from_domain` + `DistributedArray.parts()` | ✅ | `python/tessera/distributed/array.py` |
| `tessera.index_launch` — `IndexLauncher`, `_ShardDispatcher` | ✅ | `python/tessera/distributed/launch.py` |
| `tessera.constraint.Divisible/Range/Equal` + `ConstraintSolver` | ✅ | `python/tessera/compiler/constraints.py` |
| `Effect` enum + `EffectLattice` inference | ✅ | `python/tessera/compiler/effects.py` |
| `tessera.ops.*` — registry-backed NumPy reference ops and lowering hooks | ✅ | `python/tessera/__init__.py` |
| `MockRankGroup` — thread-based multi-rank collective testing | ✅ | `python/tessera/testing/mock_collective.py` |
| `tessera.dist.Cyclic` — cyclic `ShardSpec` construction; backend runtime behavior is target-dependent | ✅ / scaffolded runtime | `python/tessera/distributed/domain.py` |

### C++ x86 Lowering

| Component | Status | Location |
|-----------|--------|----------|
| `EffectAnnotationPass` — infers `pure/random/memory/io` on `func.func` | ✅ | `src/transforms/lib/EffectAnnotationPass.cpp` |
| `CanonicalizeTesseraIRPass` — 4 fusion patterns | ✅ | `src/transforms/lib/CanonicalizeTesseraIR.cpp` |
| `DistributionLoweringPass` — `tessera.shard` → `schedule.mesh.*` | ✅ | `src/transforms/lib/DistributionLoweringPass.cpp` |
| `TilingPass` — `tessera.matmul` → `scf.for` + `tensor.extract/insert_slice` | ✅ | `src/transforms/lib/TilingPass.cpp` |
| `TileToX86Pass` — tiled matmul → `func.call @tessera_x86_amx_gemm_bf16` | ✅ | `src/transforms/lib/TileToX86Pass.cpp` |
| `tessera-lower-to-x86` named pipeline | ✅ | registered in transforms lib |
| x86 AMX BF16 + AVX-512 GEMM kernels | ✅ | `src/compiler/codegen/tessera_x86_backend/` |

### NVIDIA GPU Backend

| Component | Status | Location |
|-----------|--------|----------|
| `GPUTargetProfile` + `ISA` enum (SM_80–SM_100) | ✅ | `python/tessera/compiler/gpu_target.py` |
| `FlashAttnLoweringConfig` (tile_q, tile_kv, pipeline_stages, causal, dropout) | ✅ | `python/tessera/compiler/attn_lower.py` |
| `TileIRLoweringPass` — `schedule.mesh.region` → `tile.*` + `tessera.attn.*` | ✅ | `src/transforms/lib/TileIRLoweringPass.cpp` |
| `WarpSpecializationPass` — producer/consumer warp roles + queue barriers | ✅ | `src/compiler/tile_opt_fa4/lib/WarpSpecializationPass.cpp` |
| `AsyncCopyLoweringPass` — `tile.async_copy` → TMA (SM_90) / `cp.async` (SM_80) | ✅ | `src/compiler/tile_opt_fa4/lib/AsyncCopyLoweringPass.cpp` |
| `NVWGMMALoweringPass` — `tile.mma` → `wgmma.mma_async` PTX (SM_90+) or WMMA | ✅ | `src/compiler/codegen/tessera_gpu_backend_NVIDIA/` |
| `NVTMADescriptorPass` — TMA descriptor hoisting + mbarrier init | ✅ | `src/compiler/codegen/tessera_gpu_backend_NVIDIA/` |
| `NVFlashAttnKernelEmitter` — scale resolution, full mbarrier seq, launch bounds | ✅ | `src/compiler/codegen/tessera_gpu_backend_NVIDIA/` |
| `tessera-lower-to-gpu` named pipeline | ✅ | registered in GPU backend |
| FA-4 Attn dialect v2.0 — `ScaledDotProduct`, `OnlineSoftmax`, `LseAccumulate`, `DropoutMask`, `CausalMask` | ✅ | `src/compiler/tile_opt_fa4/include/tessera/Dialect/Attn/Attn.td` |
| `tessera.queue` dialect — `create`, `push`, `pop` | ✅ | `src/compiler/tile_opt_fa4/dialects/tessera_queue/Queue.td` |

### Work That Is Scaffolded Or Artifact-Only

| Component | Phase |
|-----------|-------|
| Native NCCL/RCCL hardware-runtime execution for all collectives | scaffolded / mock-runtime tests |
| TPU hardware-runtime execution | lit-testable artifacts |
| Non-CPU backend native execution from Python `@jit` | artifact-only unless backend docs say otherwise |
| Production deployment guarantees | planned |

---

## System Concepts Carried Forward

The older `docs/archive/old_concepts/tessera_system_architecture.md` blueprint included several useful system concepts that remain valid as design direction, but not as implemented behavior:

| Concept | Current status |
|---------|----------------|
| Kernel metadata bundles containing tile shape, resource use, launch bounds, and backend artifacts | scaffolded / production packaging planned |
| Runtime traces, profiler events, and per-kernel metrics | implemented foundations / production diagnostics planned |
| Repro packs containing IR snapshots, launch args, binaries, logs, and environment hashes | planned |
| Distributed topology service, bucketizer, and overlap engine | scaffolded / hardware-runtime work planned |
| Runtime C ABI as the stable host/runtime contract | specified in `docs/spec/RUNTIME_ABI_SPEC.md`; Python wrapper is mock-runtime unless the C backend is built |

These concepts should be referenced as roadmap items unless a later spec marks them implemented.

---

## Component Map

```
python/tessera/
├── compiler/
│   ├── jit.py            @tessera.jit, @tessera.kernel, JitFn, KernelFn
│   ├── constraints.py    ConstraintSolver, Divisible, Range, Equal
│   ├── effects.py        Effect enum (pure/random/memory/io/top), EffectLattice
│   ├── graph_ir.py       GraphIRBuilder — Python → MLIR tessera dialect text
│   ├── gpu_target.py     GPUTargetProfile, ISA enum (SM_80–SM_100)
│   └── attn_lower.py     FlashAttnLoweringConfig, SM90_DEFAULT
│
├── distributed/
│   ├── region.py         Region["read"/"write"/"reduce_*"], RegionType
│   ├── domain.py         Rect, Block, Cyclic, Replicated
│   ├── shard.py          ShardSpec, MeshSpec
│   ├── array.py          DistributedArray, from_domain()
│   └── launch.py         index_launch, IndexLauncher, _ShardDispatcher
│
├── ops/
│   └── __init__.py       gemm, matmul, layer_norm, softmax, gelu, relu,
│                         dropout, conv2d, flash_attn, all_reduce, etc.
│
└── testing/
    └── mock_collective.py  MockRankGroup, MockRank, MockCollectiveError

src/
├── ir/
│   └── TesseraOps.td     matmul, conv2d_nhwc, flash_attn, fused_epilogue,
│                         transpose, cast — Graph IR op definitions
│
├── transforms/lib/
│   ├── CanonicalizeTesseraIR.cpp   4 fusion patterns
│   ├── EffectAnnotationPass.cpp    tessera.effect inference
│   ├── DistributionLoweringPass.cpp tessera.shard → schedule.mesh.*
│   ├── TilingPass.cpp              matmul → scf.for tiled loops
│   ├── TileToX86Pass.cpp           tiled matmul → x86 C calls
│   └── TileIRLoweringPass.cpp      flash_attn → FA-4 tile ops
│
├── programming_model/ir/schedule/
│   └── ScheduleMeshPipelineOps.td  mesh.define, mesh.region, pipeline.region,
│                                   stage, yield
│
├── tile_opt_fa4/
│   ├── dialects/tessera_attn/      Attn.td v2.0 — 7 FA-4 ops
│   ├── dialects/tessera_queue/     Queue.td — create, push, pop
│   └── lib/
│       ├── WarpSpecializationPass.cpp
│       └── AsyncCopyLoweringPass.cpp
│
├── compiler/codegen/
│   ├── tessera_x86_backend/        AMX BF16 + AVX-512 GEMM kernels (works)
│   └── tessera_gpu_backend_NVIDIA/ NVWGMMALoweringPass, NVTMADescriptorPass,
│                                   NVFlashAttnKernelEmitter
│
└── runtime/include/tessera/
    └── tessera_runtime.h           C ABI header and lifecycle surface
```

---

## Key Design Decisions (Locked)

1. **Python frontend is permanent.** No Rust frontend layer. The MLIR C++ pass pipeline handles performance-critical compilation. See `docs/archive/old_concepts/Rust_Frontend_Research/` for the rejected proposal.

2. **Static AOT compilation only.** No tracing JIT tier. ConstraintSolver runs at decoration time; effects are inferred statically. See `docs/archive/old_concepts/Tracing_JIT_Research/` for the rejected research.

3. **Region is a type annotation, not a runtime wrapper.** `Region["read"]` participates in Python's type annotation system and lowers to `tessera.effect` attrs on Graph IR func arguments. It does not wrap tensors at runtime.

4. **Domains and distributions are always separate.** `tessera.domain.Rect` (shape) and `tessera.dist.Block` (placement) are distinct objects. Never merged.

5. **ConstraintSolver runs at `@jit` decoration time.** AST parsing extracts `tessera.require(...)` calls before any IR is emitted. Errors are reported before the function is ever called.

6. **Effects are inferred, not declared.** Only `@jit(deterministic=True)` and `@jit(seed=N)` are user-declared. The `EffectLattice` infers everything else.

7. **CPU-first, then GPU.** CPU reference/x86 paths are the validation base. GPU-specific IR ops are gated by target profile and backend capability.

8. **Mock collectives use threads, not processes.** Multi-rank tests run in-process with Python threads as fake ranks where native NCCL/MPI is not required.

---

## Authoritative References

| Question | Where to look |
|----------|---------------|
| Canonical API names and syntax | `docs/CANONICAL_API.md` |
| IR stack, pass pipeline, phase status | `docs/spec/COMPILER_REFERENCE.md` |
| Python API (all symbols, signatures, errors) | `docs/spec/PYTHON_API_SPEC.md` |
| Graph IR op catalog + canonicalization | `docs/spec/GRAPH_IR_SPEC.md` |
| Every pass: input/output/invariants | `docs/spec/LOWERING_PIPELINE_SPEC.md` |
| FA-4 Attn dialect, TMA ops, WGMMA, Schedule Mesh | `docs/spec/TARGET_IR_SPEC.md` |
| Historical architecture blueprint | `docs/archive/old_concepts/tessera_system_architecture.md` |
| Programming guide | `docs/programming_guide/` |
