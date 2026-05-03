# Tessera — Claude Code Project Context

> **Phases 1–6 complete. Phase 7 (neighbors, Cerebras/Metalium, production) is next.**
> This file is the authoritative working reference. Read it before touching any code.
> Last updated: May 2026.

---

## What Tessera Is

Tessera is a **pre-alpha, tile-centric programming model and compiler** for deep learning and HPC. Tiles, explicit memory spaces, numerical precision, and parallelism are **first-class IR objects** — not runtime heuristics.

Target hardware: NVIDIA (SM90 Hopper, SM100 Blackwell), AMD ROCm, Google TPU, Cerebras WSE-3, Tenstorrent Metalium, x86 AMX/AVX512.

---

## Four-Layer IR Stack

```
Python API  (@jit, Region[...], tessera.domain, index_launch)
     │
     ▼
Graph IR    (tessera dialect — TesseraOps.td, mathematical ops, effects, shapes)
     │
     ▼
Schedule IR (schedule.* dialect — mesh regions, pipeline stages, optimizer sharding)
     │
     ▼
Tile IR     (tile_opt_fa4 — warp specialization, TMEM, async copy, KV cache)
     │
     ▼
Target IR   (per-backend: NVRubinCPX, ROCm, TPU/StableHLO, Cerebras, Metalium, x86)
```

The **x86 AMX/AVX512 backend** is the only fully wired execution path today. All GPU/accelerator backends produce IR but do not yet execute end-to-end.

---

## Phase Completion Status

| Phase | Status | Key deliverables |
|-------|--------|-----------------|
| Phase 1 | ✅ Complete | Python frontend — `@tessera.jit`, `@tessera.kernel`, `Region`, `domain`, `dist`, `DistributedArray`, `index_launch`, `ConstraintSolver`, `EffectLattice`, `GraphIRBuilder` |
| Phase 2 | ✅ Complete | C++ lowering chain — `DistributionLoweringPass`, `EffectAnnotationPass`, `TilingPass`, `TileToX86Pass`; `tessera-lower-to-x86` named pipeline |
| Phase 3 | ✅ Complete | NVIDIA GPU backend — `GPUTargetProfile`, `TileIRLoweringPass`, `WarpSpecializationPass`, `AsyncCopyLoweringPass`, `NVWGMMALoweringPass`, `NVTMADescriptorPass`, FA-4 Attn dialect |
| Phase 4 | ✅ Complete | Distributed training — Cyclic distribution, NCCL/RCCL adapters, `CollectiveInsertionPass`, `PipelineStageInsertionPass`, TPU quantized dot, `DistributedPlan`, `PipelinePlan`, MoE helpers — 127 tests |
| Phase 5 | ✅ Complete | Solver passes (11 core + 2 linalg + 3 SR), `BayesianAutotuner`, checkpoint decorator, `solver_config.py` — 176 tests |
| Phase 6 | ✅ Complete | `TesseraRuntime` Python wrapper, CUDA/HIP backends (real calls), ROCm MFMA coverage, benchmark runners, `ErrorReporter`, `ShapeInferencePass` — 170 tests |
| Phase 7 | 🟡 In progress | Neighbors dialect (halo/stencil) — passes implemented & wired into `tessera-opt` (May 2026); Cerebras WSE-3 backend, Tenstorrent Metalium backend, production hardening still pending |
| RubinCPX | ✅ Built | `tessera.target.cpx` dialect, 4 passes, `tessera-cpx-opt` driver, `TESSERA_BUILD_RUBINCPX_BACKEND` CMake option |

**Total active tests: 55+ test files in `tests/unit/`; lit tests in `tests/tessera-ir/phase2–7/`.**

---

## Key Source Locations

### Python package (`python/tessera/`)

| Module | Purpose |
|--------|---------|
| `__init__.py` | Top-level exports: `jit`, `kernel`, `Region`, `domain`, `dist`, `array`, `index_launch`, `constraint`, `ops`, `Tensor`, `f16`, `mut_f32` |
| `compiler/jit.py` | `@jit` and `@kernel` decorators; routes to x86 or GPU pipeline |
| `compiler/constraints.py` | `ConstraintSolver`: `Divisible`, `Range`, `Equal` — checked at decoration time |
| `compiler/effects.py` | `EffectLattice`: `pure < random < memory < io < top` |
| `compiler/graph_ir.py` | Python → Graph IR lowering (emits MLIR text) |
| `compiler/gpu_target.py` | `GPUTargetProfile`, `ISA` enum (SM_80–SM_100) |
| `compiler/attn_lower.py` | `FlashAttnLoweringConfig` (tile_q, tile_kv, pipeline_stages) |
| `compiler/autotune_v2.py` | `BayesianAutotuner` (Optuna TPE + Hyperband pruning, SQLite cache v2) |
| `compiler/checkpoint.py` | `@jit(checkpoint=True)` extension, `CollectiveCheckpointConfig` |
| `compiler/solver_config.py` | `SolverConfig`, `ZeROConfig`, `ResilienceConfig`, `DeploymentManifest`, `RNGStreamPlan` |
| `compiler/distributed_planner.py` | `DistributedPlan`, `LayerSpec`, dp/tp/pp assignment |
| `compiler/pipeline_planner.py` | `PipelinePlan`, 1F1B schedule builder |
| `compiler/tpu_target.py` | `TPUTargetProfile` (MXU tile=128, mesh_axes, `validate_matmul_dims`) |
| `distributed/region.py` | `Region["read"/"write"/"reduce_sum"]` type annotation |
| `distributed/domain.py` | `Rect` domain, `Block`/`Cyclic`/`Replicated` distributions |
| `distributed/shard.py` | `ShardSpec`, `MeshSpec` |
| `distributed/array.py` | `DistributedArray.from_domain()`, `.parts()` |
| `distributed/launch.py` | `index_launch()`, `@kernel` decorator |
| `distributed/moe.py` | `MoEConfig`, `route_tokens()`, `plan_all_to_all()` |
| `runtime.py` | `TesseraRuntime` — ctypes wrapper over runtime C ABI |
| `diagnostics.py` | `ErrorReporter`, `ShapeInferenceEngine`, `TesseraShapeError`/`TargetError` |
| `telemetry.py` | Shared telemetry event/report schema (profiler, autotune, benchmarks) |
| `profiler.py` | Runtime profiler facade (wraps `tools/profiler/`) |
| `autotune.py` | Public autotuning facade (wraps `compiler/autotune_v2.py`) |
| `arch.py` | Architecture helpers |
| `debug.py` | Debug utilities |
| `fault.py` | Fault tolerance primitives |
| `elastic.py` | Elastic training support |
| `server.py` | Inference server scaffolding |
| `shape.py` | `Dim`, `Layout`, `Shape`, `ShapeConstraintGraph`, `ShapeShard` |
| `testing/mock_collective.py` | Thread-based fake ranks for multi-rank tests |
| `testing/qa.py` | QA utilities |

### C++ compiler (`src/compiler/`)

| Path | Purpose |
|------|---------|
| `ir/TesseraOps.td` | Graph IR ODS — `MatmulOp`, `Conv2DNHWCOp`, `FlashAttnOp` + TilingInterface |
| `ir/TesseraTiling.cpp` | Tiling interface implementations |
| `programming_model/ir/schedule/ScheduleMeshPipelineOps.td` | Schedule IR ODS — mesh, pipeline, yield |
| `tile_opt_fa4/include/tessera/Dialect/Attn/Attn.td` | FA-4 Attn dialect ODS |
| `tile_opt_fa4/include/tessera/Dialect/Queue/Queue.td` | FA-4 Queue dialect ODS |
| `codegen/tessera_x86_backend/` | AMX BF16 + AVX512 GEMM — **works end-to-end** |
| `codegen/tessera_gpu_backend_NVIDIA/` | NVIDIA WGMMA/TMA backend (IR ready, no real execution) |
| `codegen/Tessera_ROCM_Backend/` | ROCm MFMA backend — gfx90a/gfx94x/gfx120x |
| `codegen/Tessera_RubinCPX_Backend/` | NV Rubin CPX — `tessera.target.cpx` dialect, 4 passes |
| `codegen/Tessera_TPU_Backend/` | TPU StableHLO + Shardy export |
| `codegen/Tessera_Cerebras_backend/` | Cerebras WSE-3 backend — **Phase 7** |
| `codegen/Tessera_Metalium_Backend/` | Tenstorrent Metalium backend — **Phase 7** |
| `diagnostics/ErrorReporter.cpp` | Source-attributed shape error reporting |
| `diagnostics/ShapeInferencePass.cpp` | Forward shape propagation |
| `tessera_neighbors/` | Halo/stencil neighbor exchange dialect — **Phase 7** |

### C++ transforms (`src/transforms/lib/`)

| File | Phase | Purpose |
|------|-------|---------|
| `CanonicalizeTesseraIR.cpp` | 1 | 4 Graph IR fusion patterns |
| `VerifyTesseraIR.cpp` | 1 | Module version attribute check |
| `DistributionLoweringPass.cpp` | 2 | `tessera.shard` → `schedule.mesh.define` + `schedule.mesh.region` |
| `EffectAnnotationPass.cpp` | 2 | Infers `pure/random/memory/io`; annotates `func.func` |
| `TilingPass.cpp` | 2 | `tessera.matmul` → `scf.for` M×N tile loops |
| `TileToX86Pass.cpp` | 2 | Tiled BF16 → `func.call @tessera_x86_amx_gemm_bf16` |
| `TileIRLoweringPass.cpp` | 3 | `schedule.mesh.region` → Tile IR ops |
| `GPUCollectiveInsertionPass.cpp` | 4 | `collective.reduce_scatter` at DP mesh boundaries |
| `PipelineStageInsertionPass.cpp` | 4 | 1F1B micro-batch schedule across ranks |

### C++ solvers (`src/solvers/`)

| Path | Status |
|------|--------|
| `core/passes/` — 11 solver passes | ✅ SparseInspector, SparsePrecond, SparseSolverSpecialize, RNGLegalize, RNGStreamAssign, NewtonAutodiff, TrigInit, PeriodicHalo, ParamBatchPlan, ContinuationGuard, ImplicitLower |
| `linalg/lib/Passes/` — MixedPrecision, IterativeRefinement | ✅ Implemented |
| `scaling_resilience/lib/sr/passes/` — InsertRecompute, OptimizerShard, ResilienceRestart | ✅ Implemented |
| `spectral/` | Spectral/FFT dialect — scaffold present, pass bodies need work |
| `tpp/` | Tensor Parallel Primitives — dialect scaffold present |

### C++ collectives (`src/collectives/`)

| Component | Status |
|-----------|--------|
| `CollectiveOps.td` — `AllToAllOp`, `AllReduceOp`, `ReduceScatterOp`, `AllGatherOp` | ✅ Defined |
| `Adapters.h` — `NCCLAdapter`, `RCCLAdapter` | ✅ Implemented (with mock paths) |
| `ChunkPlanner.cpp` | ✅ NVLink=512KiB, PCIe=128KiB, RDMA=256KiB |
| `CollectiveScheduler.cpp` | ✅ Credit-based scheduler |

### Runtime (`src/runtime/`)

| File | Status |
|------|--------|
| `src/tessera_runtime.cpp` | ✅ 270 lines — `tsrContextCreate`, `tsrMalloc`, `tsrMemcpy`, `tsrLaunchHostTileKernel` |
| `src/backend/cuda_backend.cpp` | ✅ Real `cudaMalloc/cudaMemcpy/cudaStream` calls |
| `src/backend/hip_backend.cpp` | ✅ Real `hipMalloc/hipMemcpy/hipStream` calls |
| `src/backend/tessera_runtime_cpu.cpp` | ✅ Real thread pool CPU backend |

### Benchmarks (`benchmarks/`)

| File | Purpose |
|------|---------|
| `benchmark_gemm.py` | M/N/K sweep — latency_ms, tflops, memory_bw |
| `benchmark_attention.py` | B/H/S/D sweep — tokens/sec, MFU |
| `benchmark_collective.py` | 2–128 ranks — bus bandwidth |
| `run_all.py` | Orchestrates all; emits `tessera_benchmarks_*.json` |
| `perf_gate.py` | Telemetry baseline gate for deterministic CPU smoke |

### Tools

| Path | Purpose |
|------|---------|
| `tools/tessera-opt/tessera-opt.cpp` | MLIR opt-style driver — all dialects + passes registered |
| `tools/profiler/` | tprof runtime, CLI, Perfetto export |
| `tools/roofline_tools/` | Roofline ingestion and HTML reports |
| `scripts/validate.sh` | CPU-only validation spine (version check + unit + runtime + benchmark smoke) |
| `scripts/check_versions.py` | CMake/Python/runtime header version drift check |

---

## Architecture Decisions — Do Not Revisit

1. **CPU-first, then GPU.** x86 AMX is the only real execution path today. All GPU ops are gated behind `target_profile.isa >= SM_90`.

2. **`Region` is a type annotation, not a runtime wrapper.** `Region["read"]` returns a `RegionType` object. It does NOT wrap tensors at runtime.

3. **Domains and distributions are always separate.** `Rect` = shape. `Block/Cyclic/Replicated` = placement. Never merge them.

4. **`ConstraintSolver` runs at decoration time.** `@jit` inspects annotations and calls `ConstraintSolver.check(signature)` before IR emission. Violations → `TesseraConstraintError`.

5. **Effects are inferred, not declared.** `EffectLattice` walks the IR. Programmers only declare `@jit(deterministic=True)` and `@jit(seed=N)` at the top level.

6. **Mock collectives use threads, not processes.** Multi-rank tests run in-process via `MockRankGroup`. No NCCL/MPI dependency in the test suite.

7. **`tessera.array` is not `numpy.ndarray`.** `DistributedArray` carries a `ShardSpec` and logical shape. Physical storage is backend-dependent; on CPU it is a numpy array.

8. **Warp role separation is structural, not advisory.** `WarpSpecializationPass` emits hard `tessera.schedule.warp {role="producer/consumer"}` boundaries. Different register files and barrier slots per role.

9. **TMA descriptors are generated once per kernel, not per tile.** `NVTMADescriptorPass` hoists descriptor setup to kernel preamble.

10. **Recompute insertion is budget-guided.** `InsertRecomputePass` uses `--memory-budget-mb` and a greedy live-set scan. Only pure ops qualify for recomputation.

11. **Bayesian autotuner warm-starts from SQLite cache.** Key = `hash(device_class + kernel_id + config)`. v2 schema adds Optuna trial IDs.

12. **Benchmark JSON schema is stable.** Fields: `backend`, `op`, `shape`, `dtype`, `latency_ms`, `tflops`, `memory_bw_gb_s`, `device`, `tessera_version`. `tools/roofline_tools/` reads this directly — do not change the schema.

13. **`TesseraShapeError` always includes Python source location.** `ErrorReporter` walks MLIR `loc` chain. Never suppress — emit `"<unknown location>"` if unavailable.

14. **MFMA shapes live in a lookup table.** `MFMAFullCoveragePass` reads `mfma_table.inc`. Do not hardcode shapes in pass logic.

15. **Canonical API.** `docs/CANONICAL_API.md` wins all naming conflicts. Decorators are `@tessera.jit` and `@tessera.kernel` — not `@tessera.function`, `@ts.kernel`, etc.

16. **ZeRO stage 2 only.** `OptimizerShardPass` partitions momentum + variance across `dp` mesh. Stage 3 (parameter sharding) is out of scope.

17. **Pipeline parallelism uses 1F1B by default.** `schedule="interleaved"` requires `micro_batches >= 2 * num_stages`.

18. **RNG streams are deterministically assigned.** `stream_id = global_seed * num_ranks + rank`. Philox counter offsets are non-overlapping for 2^128 elements.

---

## Key Design Contracts

### Region Privileges

Valid modes: `"read"`, `"write"`, `"reduce_sum"`, `"reduce_max"`, `"reduce_min"`

Two write regions on overlapping data → `TesseraConstraintError` at decoration time. `reduce_*` regions can safely overlap with `read` regions.

### Domain & Distribution

```python
D    = tessera.domain.Rect((B, S, D_model))   # shape only
dist = tessera.dist.Block(mesh_axes=("dp",))   # partition dim-0 over dp axis
X    = tessera.array.from_domain(D, dtype="bf16", distribution=dist)
# X.shard_spec → ShardSpec(partition=(0,), mesh_axes=("dp",))
# X.parts("dp") → list of per-rank shard slices
```

`Cyclic.parts("dp")` → element `i` on rank `i % dp_size`. Cyclic + Block interaction requires `all_to_all` rebalance — `distributed_planner.py` must emit this.

### TPU Constraint

TPU MXU tile is 128×128. `@jit(target=tpu)` auto-injects `Divisible("M/N/K", 128)`.

### FA-4 Tile Sizes for SM_90

Default: `tile_q=64, tile_kv=64, pipeline_stages=2`. Stored as `tessera.tile_q`/`tessera.tile_kv` attributes so the autotuner can sweep them.

### Collective Insertion Order

`GPUCollectiveInsertionPass` must run **after** `EffectAnnotationPass` — it reads `tessera.effect = "memory"` on write-region args to identify gradient tensors needing `reduce_scatter`.

---

## Phase 7 — Next Work

### Neighbors Dialect (Halo/Stencil)

`src/compiler/tessera_neighbors/` — dialect + 4 passes (HaloInfer, StencilLower, PipelineOverlap, DynamicTopology) implemented (~680 lines). Dialect and passes are registered in `tools/tessera-opt/tessera-opt.cpp` and linked via `TesseraNeighbors`.

Each pass walks the relevant `tessera.neighbors.*` ops:
- `HaloInferPass`: reads `taps` on `stencil.define`, computes per-axis max |Δ|, annotates `halo.width` on `stencil.apply` and any `halo.region`.
- `StencilLowerPass`: lowers `stencil.apply` to pack/exchange/unpack calls.
- `PipelineOverlapPass`: applies double-buffering / overlap policy.
- `DynamicTopologyPass`: handles dynamic topology updates.

Lit tests in `tests/tessera-ir/phase7/`: `neighbors_halo_infer.mlir`, `neighbors_stencil_lower.mlir`, `neighbors_pipeline_overlap.mlir`, `neighbors_dynamic_topology.mlir`, `shardy_export.mlir`.

Structural Python test: `tests/unit/test_neighbors_dialect.py` (7 passing, 1 behavioral test that runs `tessera-opt -tessera-halo-infer` once the binary is built).

**Open work:** build `tessera-opt` against MLIR 18, run lit tests, fix any pass-body bugs the tests expose.

### Cerebras WSE-3 Backend

`src/compiler/codegen/Tessera_Cerebras_backend/` — scaffold present (includes, examples, docs, partial target dialect).

Cerebras uses a fabric-routed streaming architecture with no shared memory. Tile IR must map to `cerebras.data_tile` and `cerebras.compute_tile` with explicit routing annotations.

### Tenstorrent Metalium Backend

`src/compiler/codegen/Tessera_Metalium_Backend/` — ODS in `TesseraTargetMetalium.td`, `Codegen/Lowering/Util` lib scaffold.

Metalium uses a RISC-V core grid. Tile IR maps to Metalium's op dispatch model.

### Production Hardening (ongoing)

- Spectral/FFT solver (`src/solvers/spectral/`) — dialect defined, pass bodies incomplete
- TPP solver (`src/solvers/tpp/`) — dialect defined, needs wiring
- CI expansion beyond CPU spine (once CUDA/HIP paths are deterministic)
- `scripts/validate.sh` expansion to cover Phase 4–6 test suites

---

## GPU-Only Tier — Never Implement on CPU

Gate all of these behind `target_profile.isa >= ISA.SM_90`:

- `tessera.schedule.warp` role assignments (FA-4 warp specialization)
- `tessera.tile.mma.tcgen05` (Blackwell TMEM MMA)
- `tile.async_copy` / `tile.wait_async` stage indexing
- `tessera.schedule.policy "persistent"` (persistent CTA scheduling)
- `tessera.queue.{create, push, pop}` (tile queue dialect)
- `tcgen05.mma` PTX inline asm

---

## Testing

```bash
# Python dev install (Python 3.14 venv at /Users/gregorystoner)
pip install -e ".[dev]"

# All unit tests
pytest tests/unit/ -v

# Single test file
pytest tests/unit/test_distributed_api.py -v

# Coverage
pytest tests/unit/ --cov=tessera.distributed --cov=tessera.compiler -v

# MLIR lit tests (requires tessera-opt built)
python -m lit tests/tessera-ir/ -v
python -m lit tests/tessera-ir/phase7/ -v   # Phase 7 only

# Type check
mypy python/tessera/

# CPU validation spine
bash scripts/validate.sh
```

---

## C++ Build

```bash
# CPU only (most development)
mkdir -p build && cd build
cmake .. -DTESSERA_ENABLE_CUDA=OFF -DTESSERA_CPU_ONLY=ON
make -j$(nproc)

# With CUDA (Phase 3+)
cmake .. -DTESSERA_ENABLE_CUDA=ON -DCUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda

# With ROCm
cmake .. -DTESSERA_ENABLE_HIP=ON -DHIP_ROOT_DIR=/opt/rocm

# With RubinCPX backend
cmake .. -DTESSERA_BUILD_RUBINCPX_BACKEND=ON

# Benchmarks
python benchmarks/run_all.py --backends x86 --output tessera_benchmarks.json
```

---

## Key Reference Files

| What you need | Where to look |
|---------------|--------------|
| **Authoritative API naming** | `docs/CANONICAL_API.md` |
| Graph IR op definitions | `src/compiler/ir/TesseraOps.td` |
| Graph IR canonicalizations | `src/transforms/lib/CanonicalizeTesseraIR.cpp` |
| Schedule IR op definitions | `src/compiler/programming_model/ir/schedule/ScheduleMeshPipelineOps.td` |
| FA-4 Tile IR ODS (Attn + Queue) | `src/compiler/tile_opt_fa4/include/tessera/Dialect/Attn/Attn.td`, `Queue.td` |
| Mesh + pipeline design | `src/compiler/programming_model/docs/Parallelism_Constructs_v1_1.md` |
| Memory model spec | `src/compiler/programming_model/docs/Memory_Execution_Model_v1_1.md` |
| Collective IR + runtime design | `src/collectives/include/tessera/Dialect/Collective/IR/CollectiveOps.td` |
| Collective adapter interface | `src/collectives/include/tessera/Dialect/Collective/Runtime/Adapters.h` |
| Runtime C ABI header | `src/runtime/include/tessera/tessera_runtime.h` |
| SR dialect ODS | `src/solvers/scaling_resilience/lib/sr/dialect/SROps.td` |
| Solver dialects (rng, sparse, solver) | `src/solvers/core/dialects/` |
| Neighbors dialect | `src/compiler/tessera_neighbors/include/tessera/Dialect/Neighbors/` |
| x86 backend | `src/compiler/codegen/tessera_x86_backend/` |
| NVIDIA backend | `src/compiler/codegen/tessera_gpu_backend_NVIDIA/` |
| ROCm backend | `src/compiler/codegen/Tessera_ROCM_Backend/` |
| RubinCPX backend | `src/compiler/codegen/Tessera_RubinCPX_Backend/` |
| TPU backend | `src/compiler/codegen/Tessera_TPU_Backend/` |
| Cerebras backend | `src/compiler/codegen/Tessera_Cerebras_backend/` |
| Metalium backend | `src/compiler/codegen/Tessera_Metalium_Backend/` |
| Autotuner v1 framework | `src/compiler/autotuning/tessera/tools/autotune/` |
| IR specs | `docs/spec/` (GRAPH_IR_SPEC, RUNTIME_ABI_SPEC, MEMORY_MODEL_SPEC, etc.) |
| Style guide | `tessera_style_guide.md` |
| Claude Code skill map | `skills.md` |
| Project structure | `PROJECT_STRUCTURE.md` |
| Src component index | `src/INDEX.md` |

---

## Archive — Do Not Build

`src/archive/` and `docs/archive/` are excluded from all build targets. Do not add build targets for archived material. New work lands in canonical `src/` folders only.

---

*Last updated: May 2026 — Phases 1–6 complete (525+ tests). Phase 7 is next: Neighbors dialect (halo/stencil), Cerebras WSE-3 and Tenstorrent Metalium backends, spectral/FFT and TPP solver bodies, CI expansion.*
