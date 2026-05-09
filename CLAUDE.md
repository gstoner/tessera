# Tessera — Claude Code Project Context

> **Phases 1–6 complete. Phase 7 lit-clean. Phase 8 — Apple M-Series CPU (Phase 8.2) and GPU (Phases 8.3 → 8.4.7) operational. apple_gpu ships 26 runtime symbols across 9 kernel concepts × {f32, f16, bf16}, including 4 fused chains (matmul→softmax, matmul→gelu, matmul→rmsnorm, matmul→softmax→matmul). See `docs/apple_gpu_overview.md`.**
> This file is the authoritative working reference. Read it before touching any code.
> Last updated: May 2026 (MLIR 21 / LLVM 21 build pin).

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
| Phase 7 | 🟡 In progress | Neighbors dialect (halo/stencil) wired into `tessera-opt`; Cerebras WSE-3 (487 LOC, real) and Tenstorrent Metalium (550 LOC, real) backends landed with `tessera-lower-to-metalium` pipeline alias |
| Phase 8 | 🟢 Apple operational | Hardware-free Target IR — `tessera_rocm.mfma`, `tessera_metalium.dma/matmul`, `tessera_apple.cpu/gpu.*` ODS dialects between Tile IR and hardware-specific lowering; `@jit(target="rocm"/"metalium"/"apple_cpu"/"apple_gpu")` string targets. **Apple M-Series CPU (8.2)** — `@jit(target="apple_cpu")` via Accelerate (cblas_sgemm + BNNS f16/bf16). **Apple M-Series GPU (8.3 → 8.4.7)** — `@jit(target="apple_gpu")` via MPS + custom MSL kernels: 9 kernel concepts × {f32, f16, bf16} = 26 runtime symbols; 4 fused chains (matmul→softmax, matmul→gelu, matmul→rmsnorm, matmul→softmax→matmul); threadgroup-tiled f32 matmul_softmax for N up to 8192. See `docs/apple_gpu_overview.md`. |
| RubinCPX | ✅ Built | `tessera.target.cpx` dialect, 4 passes, `tessera-cpx-opt` driver, `TESSERA_BUILD_RUBINCPX_BACKEND` CMake option |

**Total active tests: 1,938 passing in `tests/unit/` (0 failing); Apple Phase 8 lit fixtures 4/4 passing in `tests/tessera-ir/phase8/` against the in-tree `tessera-opt`.**

---

## Key Source Locations

### Python package (`python/tessera/`)

| Module | Purpose |
|--------|---------|
| `__init__.py` | Top-level exports: `jit`, `kernel`, `Region`, `domain`, `dist`, `array`, `index_launch`, `constraint`, `ops`, `Tensor`, `f16`, `mut_f32` |
| `compiler/jit.py` | `@jit` and `@kernel` decorators; routes to x86, GPU, or string-target pipeline (`"rocm"`/`"metalium"`/`"apple_cpu"`/`"apple_gpu"`). **Phase A2-followup**: `JitFn._enforce_call_time_constraints` resolves symbolic dim names against actual call shapes and re-runs `ConstraintSolver.check`; cached per-shape. |
| `compiler/op_catalog.py` | Canonical op catalog — single source of truth for op names across Graph IR / Schedule IR / Tile IR / Target IR |
| `compiler/matmul_pipeline.py` | Multi-target matmul pipeline dispatch — selects backend lowering based on `target=` argument |
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
| `diagnostics.py` | 778 LOC — `ErrorReporter`, `ShapeInferenceEngine`, `TesseraShapeError`/`TargetError`, stable diagnostic codes (`SHAPE_MISMATCH`, `TILE_LOWERING`, `TARGET_CODEGEN`, ...), source-loc tracking |
| `telemetry.py` | Shared telemetry event/report schema (profiler, autotune, benchmarks) |
| `profiler.py` | Runtime profiler facade (wraps `tools/profiler/`) |
| `autotune.py` | Public autotuning facade (wraps `compiler/autotune_v2.py`) |
| `arch.py` | Architecture helpers |
| `debug.py` | 526 LOC — full debug surface: `DebugTrace`, `GraphTrace`, `summarize_tensor`, `debug_trace`, `trace_graph`, `export_graphviz`, `debug_value`, `debug_artifact`, `debug_barrier`, `replay_capture`/`replay_manifest`/`save_replay_manifest`, **`check_grad`**, **`check_determinism`**. Documented in `docs/guides/Tessera_Debugging_Tools_Guide.md`. |
| `cli/mlir.py` | 425 LOC — `tessera-mlir` static IR inspection CLI (installed as console script). Supports `--mode=compile_artifact --symbol=name` to read a JIT artifact without launching tensors. |
| `nn/{module,layers,functional,utils}.py` | **Complete stateful `nn.*` surface — no remaining phantoms** (Tier 1 + Phases A4/B1/B3/C1/C2/D4/H1/H2) — `Module`, `Parameter`, **`Buffer`**, `Sequential`, `ModuleList`, `ModuleDict`; layers `Linear`, `RMSNorm`, `LayerNorm`, **`BatchNorm1d`**, `Embedding`, `Dropout`, `MLP`, `MultiHeadAttention`, `MultiHeadCrossAttention`, `RotaryEmbedding`, `CastedLinear`, `CastedEmbedding`, activation Modules (`SiLU`/`Sigmoid`/`GELU`/`ReLU`/`Tanh`/`Identity`), `CrossEntropyLoss`, **`KVCache`**, **`DynamicDepthwiseConv1d`**, **`Conv2d`** (NHWC) + **`Conv2dNCHW`**, **`LSTMCell`** + **`LSTM`** (state-propagation primitive `ops.lstm_cell` + `lstm_state_h`/`_c` extractors; BPTT through the v1 tape). `Module.register_buffer(name, value, persistent=True)` for non-trainable named tensors. `Module.to(dtype)` for in-place dtype migration. `nn.utils.clip_grad_norm_`. Functional API in `functional.py` decomposes through primitive `ops.*` so autodiff sees every step. |
| `cache/{__init__,handle}.py` | **Phase B2 + E KVCacheHandle** — opaque, paged KV state. `KVCacheHandle(num_heads, head_dim, max_seq, dtype, page_size, quantize_bits=None, auto_evict=False)` + `append/read/prune/evict_oldest` methods. **Phase E1/E2/E3**: optional int8 quantized storage via `quantize_bits=4/8`, sliding-window via `auto_evict=True`, `ops.quantize_kv`/`dequantize_kv` ops, `kv_cache_update` modern alias. `tessera.ops.kv_cache_*` ops dispatch on handle vs. legacy `ReferenceKVCache`. |
| `autodiff/{tape,vjp,mixed_precision,rematerialize,__init__}.py` | **Tier 2 v1 + Phase F1/F2/F3 reverse-mode autodiff** — tape-based, numpy-reference. `tape()` context manager + `reverse(fn)` decorator + `custom_rule(name)` for VJP registration. **Phase F1**: `autocast(dtype)` + `GradScaler` for mixed-precision. **Phase F2**: `rematerialize`/`checkpoint` for activation checkpointing. **Phase F3**: VJPs for `flash_attn`, `fft`/`ifft`/`rfft`/`irfft`. **Phase F4**: `AdjointInterface` ODS scaffold at `src/compiler/ir/include/Tessera/AdjointInterface.td` + `AutodiffPass.cpp` stub at `src/transforms/lib/AutodiffPass.cpp` (build integration is follow-up). 22 built-in VJPs total. Hooks into `Parameter` via a `id(numpy_buffer) → Parameter` weak-ref registry. See `docs/spec/AUTODIFF_SPEC.md`. |
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
| `codegen/Tessera_Cerebras_backend/` | Cerebras WSE-3 backend — Phase 7, ~487 LOC, real implementation |
| `codegen/Tessera_Metalium_Backend/` | Tenstorrent Metalium backend — Phase 7, ~550 LOC, real; `tessera-lower-to-metalium` pipeline alias |
| `codegen/Tessera_Apple_Backend/` | Apple M-Series backend — **CPU + GPU operational** (Phases 8.2–8.4.7). CPU: `MatmulToAppleCPU` pass + `TesseraAppleRuntime` Accelerate shim (cblas_sgemm rank-2/rank-3, BNNS f16/bf16). GPU: 9 lowering passes (Matmul/Rope/FlashAttn/Softmax/Gelu plus 4 fusion passes) + Objective-C++ runtime shim with `MetalDeviceContext` and MSL kernel cache. 26 runtime C ABI symbols. See `docs/apple_gpu_overview.md` and `docs/apple_gpu_kernel_inventory.md`. |
| `diagnostics/ErrorReporter.cpp` | Source-attributed shape error reporting |
| `diagnostics/ShapeInferencePass.cpp` | Forward shape propagation |
| `tessera_neighbors/` | Halo/stencil neighbor exchange dialect — **Phase 7** |

### C++ transforms (`src/transforms/lib/`)

| File | Phase | Purpose |
|------|-------|---------|
| `CanonicalizeTesseraIR.cpp` | 1 | 4 Graph IR fusion patterns |
| `VerifyTesseraIR.cpp` | 1 | Module version attribute check |
| `MigrateTesseraIR.cpp` | 1 | IR version migration / upgrade transforms |
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

| File / Path | Purpose |
|-------------|---------|
| `benchmark_gemm.py` (291 LOC) | M/N/K sweep — latency_ms, tflops, memory_bw |
| `benchmark_attention.py` (245 LOC) | B/H/S/D sweep — tokens/sec, MFU; causal mask option |
| `benchmark_collective.py` (225 LOC) | 2–128 ranks — bus bandwidth |
| `run_all.py` (462 LOC) | Orchestrates all; emits `tessera_benchmarks_*.json`; backend selection |
| `perf_gate.py` (73 LOC) | Telemetry baseline gate for deterministic CPU smoke |
| `compiler_support.py` | IR dispatch helper |
| `apple_gpu/benchmark_fusion.py` (186 LOC) | Phase 8.4.6 — fused vs. sequential matmul→softmax/gelu/rmsnorm; tiled large-N matmul_softmax variant. Same JSON schema as `benchmark_gemm.py`. |
| `common/` | Shared harness — `correctness.py`, `compiler_contract.py`, `artifact_schema.py` |
| `spectral/` | Spectral/FFT solver benchmarks (Phase 7 scaffold) |
| `Tessera_Operator_Benchmarks/` | Operator-level benchmark suite |
| `Tessera_SuperBench/` | Whole-model benchmark suite |
| `DeepScholar-Bench/` | DeepScholar model port |
| `baselines/cpu_smoke.json` | Recorded CPU smoke baseline for `perf_gate.py` |

### Tools

| Path | Purpose |
|------|---------|
| `tools/tessera-opt/tessera-opt.cpp` | MLIR opt-style driver — all dialects + passes registered |
| `tools/profiler/` | tprof runtime, CLI, Perfetto export |
| `tools/roofline_tools/` | Roofline ingestion + HTML reports; CLI `cli_v2.py` with `one`/`multi` modes; Nsight CSV + Perfetto JSON ingestion; comm rooflines + overlap analysis |
| `tools/CLI/Tessera_CLI_Starter_v0_1/` | CLI starter scaffold (CMakeLists + cmake + data + docs + tests + tools) |
| `tools/tessera-translate/` | Placeholder — empty as of May 2026 |
| `python/tessera/cli/mlir.py` | `tessera-mlir` console-script entry — static IR inspection; `--mode=compile_artifact` reads JIT artifacts without launching |
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

19. **Backends expose hardware-free Target IR before hardware-specific lowering.** Each backend defines an ODS dialect of abstract target ops (`tessera_rocm.mfma`, `tessera_metalium.dma/matmul`, `tessera_apple.cpu.accelerate_gemm`, `tessera_apple.gpu.metal_kernel`) that sit between Tile IR and the final hardware emission. New backends MUST follow this pattern — do not lower Tile IR directly to PTX/HIP/Metal source. The hardware-free layer is what makes backends testable in lit and what `test_target_ir_contract.py` validates.

20. **`@jit(target=...)` accepts both `GPUTargetProfile` and string aliases.** Valid string targets: `"rocm"`, `"metalium"`, `"apple_cpu"`, `"apple_gpu"`. Strings dispatch through `matmul_pipeline.py` to the matching `tessera-lower-to-{target}` pipeline. Do not invent new string aliases without adding the corresponding pipeline.

21. **Unsupported lowering must emit a stable diagnostic.** When a backend cannot lower an op (e.g., KV-cache on a target without it), emit a diagnostic that names the op and the target — never silently no-op or fall through. See the KV-cache → target lowering for the canonical pattern. **Per-target coverage matrix:** `docs/audit/kv_cache_coverage_matrix.md` (audited 2026-05-09 — Apple CPU/GPU and ROCm honor #21 with named diagnostics; NVIDIA/x86/TPU/Cerebras simply don't encounter KV-cache ops in tested paths today and need explicit handling when they light up).

22. **Doc surface is broader than IR/runtime surface — check `docs/guides/` and `docs/programming_guide/` before claiming a feature is missing.** The 11 user guides + 11-chapter programming guide describe APIs (e.g., `tessera.debug.check_grad`, `tessera.debug.check_determinism`, replay manifests, `tessera-mlir` compile-artifact mode, autodiff via Ch.7) that are fully documented and largely implemented in `python/tessera/{debug,diagnostics,cli/mlir}.py` (526 + 778 + 425 LOC) but are easy to overlook because the source-tour above doesn't make them obvious. When evaluating "do we have X", read the relevant guide first.

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

## Phase 7 — In Progress

### Neighbors Dialect (Halo/Stencil)

`src/compiler/tessera_neighbors/` — dialect + 4 passes (HaloInfer, StencilLower, PipelineOverlap, DynamicTopology) implemented (~680 lines). Dialect and passes are registered in `tools/tessera-opt/tessera-opt.cpp` and linked via `TesseraNeighbors`.

Each pass walks the relevant `tessera.neighbors.*` ops:
- `HaloInferPass`: reads `taps` on `stencil.define`, computes per-axis max |Δ|, annotates `halo.width`.
- `StencilLowerPass`: lowers `stencil.apply` to pack/exchange/unpack calls.
- `PipelineOverlapPass`: applies double-buffering / overlap policy.
- `DynamicTopologyPass`: handles dynamic topology updates.

Lit tests in `tests/tessera-ir/phase7/`. Python wiring test: `tests/unit/test_neighbors_dialect.py`.

**Open work:** build `tessera-opt` against MLIR 21, run lit tests, fix any pass-body bugs the tests expose.

### Cerebras WSE-3 Backend

`src/compiler/codegen/Tessera_Cerebras_backend/` — ~487 LOC, real implementation. Wiring into `tessera-opt` needs verification.

Cerebras uses a fabric-routed streaming architecture with no shared memory. Tile IR maps to `cerebras.data_tile` / `cerebras.compute_tile` with explicit routing annotations.

### Tenstorrent Metalium Backend

`src/compiler/codegen/Tessera_Metalium_Backend/` — ~550 LOC, real implementation. Pipeline alias `tessera-lower-to-metalium` registered.

Metalium uses a RISC-V core grid. Tile IR maps to Metalium's op dispatch model via `TesseraTargetMetalium.td` ODS.

---

## Phase 8 — In Progress

### Hardware-Free Target IR

A new abstraction layer between Tile IR and hardware-specific lowering. Each backend exposes ODS ops that are hardware-shaped but not hardware-bound:

- `tessera_rocm.mfma`, `tessera_rocm.async_copy`, `tessera_rocm.wait`
- `tessera_metalium.dma`, `tessera_metalium.matmul`
- `tessera_apple.cpu.accelerate_gemm`, `tessera_apple.gpu.metal_kernel`, `tessera_apple.gpu.dispatch`

**Why:** lit-testable backends, shared optimization passes, easier per-target pass authoring. New backends MUST follow this layering — see Architecture Decision #19.

Contract test: `tests/unit/test_target_ir_contract.py` and `tests/tessera-ir/phase8/target_ir_contracts.mlir`.

### String `target=` Aliases

`@jit(target="rocm" | "metalium" | "apple_cpu" | "apple_gpu")` — `matmul_pipeline.py` dispatches to the corresponding `tessera-lower-to-{target}` pipeline. Coexists with the existing `GPUTargetProfile` parameter form.

### Apple M-Series Backend (Phase 8.1 — Lit-testable)

`src/compiler/codegen/Tessera_Apple_Backend/` — full dialect + Tile→Apple lowering passes. Builds the `TesseraApple` static library when `-DTESSERA_BUILD_APPLE_BACKEND=ON`. Three lit fixtures under `tests/tessera-ir/phase8/`:

- `apple_dialect_roundtrip.mlir` — dialect parse + print smoke test
- `apple_cpu_lowering.mlir` — exercises `tessera-lower-to-apple_cpu`
- `apple_gpu_lowering.mlir` — exercises `tessera-lower-to-apple_gpu`

ODS sets `usePropertiesForAttributes = 0` to keep the dialect header-only against MLIR 21's properties machinery. `tessera-opt` links the backend behind a `TESSERA_HAVE_APPLE_BACKEND` guard so non-Apple builds are unaffected.

**Phase 8.2 — Apple CPU native execution (Accelerate)** — *Items #1–#4 landed.*

C++ pieces:
- Pass `MatmulToAppleCPU` (`lib/Target/Apple/Lowering/MatmulToAppleCPU.cpp`) lowers static-shape rank-2 f32 `tessera.matmul` to `func.call @tessera_apple_cpu_gemm_f32`. Pipeline alias `tessera-lower-to-apple_cpu-runtime` (parallel to the artifact-only `tessera-lower-to-apple_cpu`).
- Runtime shim `runtime/apple_cpu_runtime.cpp` (built as `TesseraAppleRuntime`, links `-framework Accelerate` on Darwin, portable reference fallback elsewhere). Exports three GEMM symbols:
  - `tessera_apple_cpu_gemm_f32` — single rank-2 f32 GEMM via `cblas_sgemm`
  - `tessera_apple_cpu_gemm_f32_batched` — rank-3 batched GEMM looping `cblas_sgemm` per batch
  - `tessera_apple_cpu_gemm_f16` — rank-2 fp16 GEMM via `BNNSMatMul` (BNNSDataLayout2DFirstMajor, native fp16) with internal cblas+fp32-conversion fallback
- Lit fixture `tests/tessera-ir/phase8/apple_cpu_runtime.mlir` covers positive (static f32) + negative (dynamic shape) paths.

Python pieces:
- `@jit(target=...)` raises `TesseraJitError` at decoration time if function source can't be inspected (REPL/heredoc) — no more silent eager-Python fallback while pretending to be a target run. `target=None` keeps the soft-warning behavior.
- `_execute_apple_cpu_accelerate_artifact` in `runtime.py` chains arbitrary supported op sequences: matmul/gemm dispatches to Accelerate, every other supported op falls through to the numpy reference path. Multi-op programs are first-class.
- `_apple_cpu_dispatch_matmul` selects between the rank-2 f32 fast-path, rank-3 batched f32 path, rank-2 fp16 (BNNS) path, or `np.matmul` fallback.
- `runtime_artifact()` metadata reports `op_count`, `accelerate_op_count`, `accelerate_ops`, and `fallback_path` for multi-op programs while preserving the original strict guards for single-matmul programs.

Tests (`tests/unit/test_apple_backend_roadmap.py` — 10 passing): single matmul, multi-op tiny decode, rank-3 batched, fp16 via BNNS, plus three runtime-shim ABI tests that compile the runtime from source and probe each exported symbol numerically.

End-to-end verified on this Mac (LLVM/MLIR 21, Accelerate active): single GEMM bitwise-matches numpy; multi-op tiny decode bitwise-matches the numpy reference; rank-3 batched GEMM bitwise-matches numpy; fp16 matmul matches an f32-converted reference at fp16 tolerance.

**Phase 8.2 follow-up — landed:**
- **bf16 GEMM** — `tessera_apple_cpu_gemm_bf16` C symbol via `BNNSDataTypeBFloat16` (macOS 12+) with a bit-shift fp32 conversion fallback. Python boundary uses `ml_dtypes.bfloat16` (registered as `[project.optional-dependencies] ml_dtypes`) — the dtype probe is a soft import, so the bf16 fast path is unavailable when `ml_dtypes` isn't installed but the rest of the runtime keeps working. Tests under `test_apple_cpu_accelerate_dispatches_bf16_matmul_via_bnns` + `test_apple_cpu_runtime_exposes_bf16_gemm_symbol` + the `_disabled_when_ml_dtypes_missing` soft-dep contract test.
- **Launch-overhead reduction** — `JitFn.runtime_artifact()` is now lazily cached on first call, and `__call__` for `apple_cpu` bypasses `runtime.launch()` via `_apple_cpu_fast_call` → `_execute_apple_cpu_accelerate_metadata` (the metadata dispatcher split out of `_execute_apple_cpu_accelerate_artifact`). The public `launch(mm.runtime_artifact(), ...)` entry stays unchanged for callers who want telemetry. **Measured speedups on Apple Silicon (M-series, Accelerate active):** 8×8×8 GEMM 459 µs → 10 µs (**46×**); 32×32×32 456 µs → 12 µs (**38×**); 128×128×128 470 µs → 19 µs (**25×**); 512×512×512 780 µs → 193 µs (**4×**). Tessera launch overhead at 512×512 is now ~1.3× numpy (was ~5×).

*Open work:*
- Apple GPU (Phase 8.3) — MPS dispatch, separate phase.

**Phase 8.3 — Apple GPU baseline via MPS — landed.** ODS ops `tessera_apple.gpu.mps_matmul` / `mps_softmax` / `mps_dispatch`. Pass `MatmulToAppleGPU`. Runtime: Objective-C++ (`.mm`) `MetalDeviceContext` wrapping `MTLDevice` + `MTLCommandQueue` + `MPSMatrixMultiplication`. No `metal-cpp` vendoring. Single rank-2 f32 matmul executes natively via MPS.

**Phase 8.4.0 — Custom MSL kernel infrastructure + rope — landed.** ODS op `tessera_apple.gpu.msl_kernel` carries MSL source as a `StringAttr`. Runtime compiles via `[device newLibraryWithSource:options:error:]`, caching by `(msl_source, entry_point)` sha256. First custom kernel: rope.

**Phase 8.4.1 — flash-attention forward — landed.** Online softmax in a single MSL kernel. fp32 accumulators throughout; head_dim ≤ 256.

**Phase 8.4.2 — softmax + gelu standalone MSL kernels — landed.**

**Phase 8.4.3 — first multi-op fusion: matmul → softmax — landed.** Fused MSL kernel materializes the (M, N) score matrix in per-thread stack array. Runtime gate becomes "recognized op-chain in envelope" rather than "single op."

**Phase 8.4.4 — fp16/bf16 matmul — landed.** Native MPSDataTypeFloat16 for fp16. fp32-conversion path inside the runtime shim for bf16 (MPS doesn't natively accept bf16 matrix descriptors as of macOS 14).

**Phase 8.4.4.1 — fp16/bf16 for simple MSL kernels (rope, softmax, gelu) — landed.** Native MSL `half` for fp16; fp32-conversion for bf16.

**Phase 8.4.4.2 — fp16/bf16 for fused matmul→softmax + flash_attn — landed.** Mixed-precision: `half` I/O + fp32 per-thread accumulators (matches production flash-attn implementations).

**Phase 8.4.5 — 3-op fusion: matmul → softmax → matmul (full attention block) — landed.** `O = softmax(A @ B) @ C` collapsed into a single MSL kernel with two stack arrays (`scores[256]` + `out[256]`). All three dtypes.

**Phase 8.4.6 — threadgroup-tiled matmul_softmax_f32 + benchmark harness — landed.** Lifts the N ≤ 256 constraint via dynamic threadgroup memory (cap N ≤ 8192). One row per threadgroup; 32 threads cooperate. Benchmark harness at `benchmarks/apple_gpu/benchmark_fusion.py`.

**Phase 8.4.7 — MLP-block fusions (matmul → gelu, matmul → rmsnorm) — landed.** Two more 2-op fusions completing the common transformer-block chains. f32 only this phase.

**Pipeline ordering:** longest fusion first (3-op → 2-op → per-op) so the most specific match wins. Detailed in `docs/apple_gpu_overview.md`.

**Out of scope:** AIR bitcode codegen (Mojo's path). Tessera uses MPS for ops Apple ships kernels for and MSL emission for the gaps; AIR codegen revisited only if a perf wall demands it.

### Production Hardening (ongoing)

- Spectral/FFT solver (`src/solvers/spectral/`) — dialect defined, pass bodies incomplete
- TPP solver (`src/solvers/tpp/`) — dialect defined, needs wiring
- CI expansion beyond CPU spine (once CUDA/HIP paths are deterministic)
- `scripts/validate.sh` expansion to cover Phase 4–8 test suites

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

# With Apple M-Series backend (Phase 8.1 — lit-testable)
cmake .. -DTESSERA_BUILD_APPLE_BACKEND=ON

# Benchmarks
python benchmarks/run_all.py --backends x86 --output tessera_benchmarks.json
```

### Canonical lowering pipelines (named pass pipelines in `tessera-opt`)

| Pipeline | Target |
|----------|--------|
| `tessera-lower-to-x86` | x86 AMX/AVX512 — Phase 2 |
| `tessera-lower-to-gpu` | NVIDIA SM_90+ WGMMA/TMA — Phase 3 |
| `tessera-lower-to-rocm` | AMD ROCm MFMA — Phase 8 |
| `tessera-lower-to-metalium` | Tenstorrent Metalium — Phase 8 |
| `tessera-lower-to-apple_cpu` | Apple Silicon CPU (Accelerate artifact) — Phase 8.1 |
| `tessera-lower-to-apple_cpu-runtime` | Apple Silicon CPU runtime (cblas_sgemm via Accelerate) — Phase 8.2 |
| `tessera-lower-to-apple_gpu` | Apple Silicon GPU (Metal artifact) — Phase 8.1 |
| `tessera-lower-to-apple_gpu-runtime` | Apple Silicon GPU runtime (MPS + custom MSL kernels). Composes (in order): matmul→softmax→matmul fusion → matmul→softmax / gelu / rmsnorm fusions → per-op (matmul mps, rope, flash_attn, softmax, gelu). Phases 8.3 → 8.4.7. |
| `tessera-cpx-pipeline` / `tessera-cpx-context-pipeline` | NV Rubin CPX (separate `tessera-cpx-opt` driver) |

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
| Apple M-Series backend (Phase 8.1) | `src/compiler/codegen/Tessera_Apple_Backend/` |
| **Apple GPU architecture story (Phase 8.3 → 8.4.7)** | `docs/apple_gpu_overview.md` |
| **Apple GPU kernel inventory** (every C ABI symbol + dtype matrix) | `docs/apple_gpu_kernel_inventory.md` |
| Apple GPU benchmark harness | `benchmarks/apple_gpu/benchmark_fusion.py` |
| Target IR contract test | `tests/unit/test_target_ir_contract.py`, `tests/tessera-ir/phase8/target_ir_contracts.mlir` |
| Autotuner v1 framework | `src/compiler/autotuning/tessera/tools/autotune/` |
| IR specs | `docs/spec/` (14 files: GRAPH_IR_SPEC, TILE_IR, TARGET_IR_SPEC, MEMORY_MODEL_SPEC, SHAPE_SYSTEM, LANGUAGE_SPEC, PYTHON_API_SPEC, RUNTIME_ABI_SPEC, COMPILER_REFERENCE, LOWERING_PIPELINE_SPEC, CONFORMANCE, CITL_ROCM_TRACE_PROFILER_SPEC, LANGUAGE_AND_IR_SPEC, **AUTODIFF_SPEC** — Tier 2 v1 design) |
| **User-facing guides** (the canonical "how to use Tessera") | `docs/guides/` — 11 guides totaling ~3,400 LOC: **Tessera_Debugging_Tools_Guide.md** (327, 6-layer debugging model + tooling per layer), **Tessera_Error_Handling_And_Diagnostics_Guide.md** (305, stable diagnostic codes), Tessera_Profiling_And_Autotuning_Guide.md (303), Tessera_Runtime_ABI_Guide.md (562), Tessera_Tensor_Layout_And_Data_Movement_Guide.md (158), Tessera_Inference_Server_Guide.md (444), Tessera_Fault_Tolerance_And_Elasticity_Guide.md (352), Tessera_Production_Reliability_And_Chaos_Guide.md (238), Tessera_QA_Reliability_Guide.md (210), Tessera_Differentiable_NAS_Guide.md (287), Tessera_Developer_Frontend_End_To_End.md (222) |
| **Programming guide** (11-chapter user manual) | `docs/programming_guide/` — Ch.1 Intro, Ch.2 Programming Model, Ch.3 Memory Model (526), Ch.4 Execution Model, Ch.5 Kernel Programming (331), Ch.6 Numerics, **Ch.7 Autodiff** (101), Ch.8 Layouts & Data Movement, Ch.9 Libraries & Primitives, Ch.10 Portability, Ch.11 Conclusion, Appendix NVL72, Tessera_Goals.md |
| Tutorials | `docs/tutorials/` — `Flash_Attention_in_Tessera.md`, `performance_tuning.md` |
| API reference | `docs/api/API_Reference_Index.md`, `docs/reference/tessera-api-reference.md`, `docs/reference/tessera_migration_guide_part{1,2}.md` |
| Getting started + glossary | `docs/GETTING_STARTED.md`, `docs/GLOSSARY.md` |
| Architecture overviews | `docs/architecture/` (system_overview.md, tessera_target_ir_usage_guide.md, Tessera_Kernel_Compilation_Stages_Overview.md), `docs/operations/Tessera_Standard_Operations.md` |
| Spec gap audits | `docs/audit/compiler_spec_gap_audit.md`, `compiler_spec_gap_matrix.md` |
| **Advanced examples capability gap** (per-example status + 10-theme tracking plan) | `docs/audit/advanced_examples_capability_gap.md` |
| **Development execution roadmap** (Phases A–I, per-task acceptance criteria, dependencies) | `docs/audit/execution_roadmap.md` |
| **NVIDIA execution audit** (Phase G1 — per-component status + 8-task punch list to first H100 BF16 GEMM) | `docs/audit/nvidia_execution_audit.md` |
| Examples (most are README/stub scaffolds) | `examples/` — `getting_started/basic_tensor_ops.py` (canonical `@tessera.jit`), `compiler/`, `advanced/` (10+ subdirs: speculative_decoding, long_context_attention, kv_cache_serving, MoE, MLA, Nemotron, Jet_Nemotron, Fast_dLLM, RLVR), `optimization/`, `integration/`. |
| Style guide | `tessera_style_guide.md` |
| Claude Code skill map | `skills.md` |
| Project structure | `PROJECT_STRUCTURE.md` |
| Src component index | `src/INDEX.md` |

---

## Archive — Do Not Build

`src/archive/` and `docs/archive/` are excluded from all build targets. Do not add build targets for archived material. New work lands in canonical `src/` folders only.

---

*Last updated: May 2026 — Phases 1–6 complete; Phase 7 lit-clean; Phase 8 Apple operational (CPU 8.2 via Accelerate; GPU 8.3 → 8.4.7 via MPS + custom MSL). Apple GPU runtime exports **26 C ABI symbols** across 9 kernel concepts × {f32, f16, bf16}: matmul (MPS), rope/softmax/gelu/flash_attn (MSL), 3 fused 2-op chains (matmul→softmax incl. tiled large-N variant, matmul→gelu, matmul→rmsnorm), and 1 fused 3-op chain (matmul→softmax→matmul, full attention block). All sharing one `MetalDeviceContext` + MSL kernel cache keyed by `(msl_source, entry_point)`. Pipelines: `tessera-lower-to-{rocm,metalium,apple_cpu,apple_cpu-runtime,apple_gpu,apple_gpu-runtime}`. Build pin: **LLVM/MLIR 21**. Test count: **2,020 unit tests passing** + **16/16 Phase 8 lit fixtures** against the in-tree `tessera-opt`. See `docs/apple_gpu_overview.md` for the full architecture story.*
