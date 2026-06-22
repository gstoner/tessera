# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

> This is the operational reference. Read it before touching code.
> **For current status, finished/open work, and what to do next, start at
> [`docs/audit/MASTER_AUDIT.md`](docs/audit/MASTER_AUDIT.md) (Decision #26).**
> Counts (entries, tests, symbols) live in `docs/audit/generated/` — do not
> trust or copy numeric snapshots written into prose anywhere, including here.
> Build pin: LLVM/MLIR 22.1.6 (Homebrew `llvm`).

---

## What Tessera Is

Tessera is a **pre-alpha, standalone, tile-centric programming model and
compiler** for deep learning and HPC. Tiles, explicit memory spaces, numerical
precision, and parallelism are **first-class IR objects** — not runtime
heuristics. "Standalone" means runtime-independent of PyTorch / JAX / Flax
(Decision #23); those are reference vocabularies only.

Target hardware: NVIDIA (SM90 Hopper, SM100 Blackwell), AMD ROCm, Google TPU,
Cerebras WSE-3, Tenstorrent Metalium, x86 AMX/AVX512, Apple M-series CPU/GPU.

**Execution reality:** the **x86 AMX/AVX512** backend and **Apple CPU
(Accelerate) + GPU (MPS/MSL/MPSGraph)** backends execute natively today. NVIDIA
and ROCm are toolchain-pinned Target IR + lit fixtures with native execution
**hardware-gated** (Phase G/H/I). Other backends produce IR/artifacts until a
hardware-gated proof row says otherwise. See
[`docs/audit/backend/BACKEND_AUDIT.md`](docs/audit/backend/BACKEND_AUDIT.md).

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
Target IR   (per-backend: NVRubinCPX, ROCm, TPU/StableHLO, Cerebras, Metalium, Apple, x86)
```

New backends MUST expose a hardware-free Target IR dialect before
hardware-specific lowering (Decision #19) — never lower Tile IR directly to
PTX/HIP/Metal source.

---

## Phase Status (high level)

| Phase | Status | Scope |
|-------|--------|-------|
| 1–6 | ✅ Complete | Python frontend → C++ lowering → NVIDIA backend IR → distributed training → solver passes/autotuner → runtime wrapper + CUDA/HIP backends |
| 7 | 🟢 Lit-verified | Neighbors (halo/stencil) dialect; Cerebras + Metalium backends; real HW gated on Phase G/H |
| 8 | 🟢 Apple operational | Hardware-free Target IR; `@jit(target="rocm"/"metalium"/"apple_cpu"/"apple_gpu")`; Apple CPU (Accelerate) + GPU (MPS + MSL + MPSGraph) execute natively |
| S-series | 🟢 In progress | Standalone-compiler track — primitive contract registry + S2–S15 Python reference surface + reasoning-model attention/RL; `backend_kernel` axis is the long-pole gate (Phase G/H/I) |
| RubinCPX | ✅ Built | `tessera.target.cpx` dialect + 4 passes + `tessera-cpx-opt` driver |

Per-phase deliverables and the open-work priority queue live in
`docs/audit/MASTER_AUDIT.md` and the theme audits.

---

## Key Source Locations

### Python package (`python/tessera/`)

| Module | Purpose |
|--------|---------|
| `__init__.py` | Top-level exports: `jit`, `kernel`, `Region`, `domain`, `dist`, `array`, `index_launch`, `constraint`, `ops`, `Tensor`, `dtype`. `train` is lazily bound via PEP 562 `__getattr__`. |
| `dtype.py` | Canonical dtype enforcement + `Dtype` typed object + promotion lattice (`canonicalize_dtype`, `result_type`, `is_canonical_dtype`, …). Canonical 15-name set; aliases normalize at API boundaries; `tf32` rejected as storage dtype (use `numeric_policy.math_mode`). See Decision #15a. |
| `compiler/jit.py` | `@jit`/`@kernel` decorators; routes to x86, GPU, or string-target pipeline. Call-time constraint re-check via `JitFn._enforce_call_time_constraints`. |
| `compiler/op_catalog.py` | Canonical op-name catalog — "what we accept today" across all IR layers. |
| `compiler/primitive_coverage.py` | **Audit truth** (Decision #24) — standalone primitive contract registry over 12 axes; consults `autodiff.vjp._VJPS`/`jvp._JVPS` so registered (V/J)VPs auto-flip to complete. Renders `docs/audit/standalone_primitive_coverage.md`. |
| `compiler/backend_manifest.py` | Per-op × per-target × per-dtype kernel manifest synthesizer; `BackendKernelEntry` + statuses `fused`/`reference`/`compileable`/`artifact_only`/`planned`. |
| `compiler/gpu_target.py` / `rocm_target.py` / `tpu_target.py` | Target profiles + feature matrices. NVIDIA pinned CUDA 13.2 U1; AMD pinned ROCm 7.2.4; TPU MXU tile 128. |
| `compiler/{constraints,effects,graph_ir}.py` | `ConstraintSolver` (decoration-time), `EffectLattice` (`pure<random<memory<io<top`), Python→Graph IR emission. |
| `compiler/{autotune_v2,attn_lower,matmul_pipeline,checkpoint,solver_config,distributed_planner,pipeline_planner}.py` | Bayesian autotuner; FA-4 lowering config; multi-target matmul dispatch; checkpoint extension; solver/ZeRO/resilience config; dp/tp/pp + 1F1B planners. |
| `compiler/evaluator.py` + `conformance_evaluator.py` + `ptx_emit.py` + `flywheel{,_autotune}.py` + `compiler_grader.py` + `attention_tasks.py` + `magellan.py` + `alphaevolve.py` | **Evaluator program** — execution-derived, rung-aware scoring engine; four oracles (vertical/horizontal/metamorphic/DESIL cross-path), conformance re-derivation, NVIDIA WGMMA PTX emission, device-keyed autotuning records, anti-cheat scored-environment search. See `docs/audit/compiler/EVALUATOR_PLAN.md` §9.5. |
| `rng.py` / `state/` / `control.py` / `sharding.py` | S4 RNG (Philox `RNGKey` + 12 samplers); S3 pytrees + 8-collection state taxonomy; S5 control flow + autodiff transforms; S6 `shard_map`/collectives + `MemoryShardSpec`. |
| `losses.py` / `rl.py` / `optim.py` / `quantization.py` | S11 21 losses; RL PPO/GRPO/CISPO; S10 9 optimizers + schedules + grad transforms; S9 int8/int4 quant + fake-quant + observers. |
| `data.py` / `aot.py` / `custom.py` / `memory.py` | S15 `Dataset` + tokenizers; S14 AOT export + compilation cache; S13 `@custom_primitive`; S7 Titans/Atlas memory primitives. |
| `nn/{module,layers,functional,utils}.py` | Complete stateful `nn.*` surface — `Module`/`Parameter`/`Buffer`, layers, attention, KV cache, conv, LSTM. `functional.py` decomposes through `ops.*` so autodiff sees every step. |
| `autodiff/{tape,vjp,jvp,mixed_precision,rematerialize}.py` | Tape-based numpy-reference reverse/forward mode; `tape()`/`reverse()`/`custom_rule()`; autocast + GradScaler + remat. See `docs/spec/AUTODIFF_SPEC.md`. |
| `cache/` | `KVCacheHandle` (paged, optional int8 quant, sliding-window) + `MemoryStateHandle` (persistent Titans/Atlas state ABI). |
| `dflash*.py` / `models/` | DFlash block-diffusion speculative decoding (rides `attn_bias` substrate; greedy spec-decode == greedy AR proven); `tessera.models` DiffusionGemma graph + native block-diffusion runtime. |
| `runtime.py` / `diagnostics.py` / `debug.py` / `cli/` | `TesseraRuntime` ctypes ABI wrapper; `ErrorReporter` + stable diagnostic codes + source-loc; full debug surface (`check_grad`, `check_determinism`, replay); `tessera-mlir`/`tessera-translate` console scripts. |
| `distributed/{region,domain,shard,array,launch,moe}.py` | `Region` annotations, `Rect`/`Block`/`Cyclic`/`Replicated`, `ShardSpec`/`MeshSpec`, `DistributedArray`, `index_launch`, MoE routing. |
| `testing/mock_collective.py` | Thread-based fake ranks for multi-rank tests (no NCCL/MPI dep). |

### C++ (`src/`)

| Path | Purpose |
|------|---------|
| `compiler/ir/TesseraOps.td` | Graph IR ODS — `MatmulOp`, `Conv2DNHWCOp`, `FlashAttnOp` (+ optional `attn_bias`), TilingInterface |
| `compiler/programming_model/ir/schedule/ScheduleMeshPipelineOps.td` | Schedule IR ODS — mesh, pipeline, yield |
| `compiler/tile_opt_fa4/include/tessera/Dialect/{Attn,Queue}/*.td` | FA-4 Tile IR dialects |
| `compiler/codegen/tessera_x86_backend/` | AMX BF16 + AVX512 GEMM — **works end-to-end** |
| `compiler/codegen/Tessera_Apple_Backend/` | Apple CPU + GPU — **operational**. CPU: `MatmulToAppleCPU` + Accelerate shim. GPU: 17-pass Tile→Apple lowering + Objective-C++ runtime (`apple_gpu_runtime.mm`) with MPS/MSL/MPSGraph lanes. |
| `compiler/codegen/{tessera_gpu_backend_NVIDIA,Tessera_ROCM_Backend,Tessera_TPU_Backend,Tessera_Cerebras_backend,Tessera_Metalium_Backend,Tessera_RubinCPX_Backend}/` | Per-target backends (IR/artifact; HW execution gated where noted) |
| `compiler/tessera_neighbors/` | Halo/stencil neighbor-exchange dialect (Phase 7) |
| `transforms/lib/*.cpp` | Pass bodies — Canonicalize/Verify/Migrate (P1), Distribution/Effect/Tiling/TileToX86 (P2), TileIRLowering/WarpSpec/AsyncCopy/WGMMA/TMA (P3), Collective/PipelineStage (P4), `AttentionFamilyPasses.cpp` (reasoning-model attention) |
| `solvers/` | Core (11 passes), linalg, scaling-resilience, spectral (6 pass bodies + `ts-spectral-opt`), TPP (7 passes + `tpp-space-time`) |
| `collectives/` | `CollectiveOps.td`, `NCCLAdapter`/`RCCLAdapter` (+ mock paths), `ChunkPlanner`, `CollectiveScheduler` |
| `runtime/src/` | `tessera_runtime.cpp` (C ABI) + CUDA/HIP/CPU backends (real calls) |

### Tools (`tools/`)

| Path | Purpose |
|------|---------|
| `tessera-opt/` | MLIR opt-style driver — all dialects + 70+ passes + named lowering pipelines. Build: `ninja -C build tessera-opt`. |
| `tessera-translate/` | C++ `tessera-translate-mlir` (MLIR↔LLVM IR / SPIR-V) + Python `tessera-translate` (StableHLO/GGUF/SafeTensors export) |
| `profiler/` / `roofline_tools/` | tprof runtime + Perfetto export; roofline ingestion + HTML reports |
| `scripts/validate.sh` / `check_versions.py` / `check_generated_docs.sh` | CPU validation spine; version-drift check; generated-doc drift gate (pre-commit) |

---

## Architecture Decisions — Do Not Revisit

1. **CPU-first, then GPU.** x86 AMX is the only real execution path on the original roadmap; GPU ops gated behind `target_profile.isa >= SM_90`. (Apple is the second native lane, Phase 8.)

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

14. **MFMA shapes live in a lookup table.** `MFMAFullCoveragePass` reads `mfma_table.inc` (generated by `scripts/generate_mfma_table.py`). Do not hardcode shapes in pass logic.

15. **Canonical API.** `docs/CANONICAL_API.md` wins all naming conflicts. Decorators are `@tessera.jit` and `@tessera.kernel` — not `@tessera.function`, `@ts.kernel`, etc.

15a. **Canonical tensor attributes & dtypes.** `docs/reference/tessera_tensor_attributes.md` is normative for the six tensor attributes (`shape`, `dtype`, `layout`, `device`/`target`, `distribution`, `numeric_policy`), the canonical dtype name set + aliases, the planned/gated dtype set, and the promotion/casting policy. Three rules that bite:
  - **Storage dtype is on the tensor; accumulator goes in `numeric_policy`** — never compress them into one dtype string. matmul/gemm/einsum/flash_attn use `storage=bf16, accum=fp32`.
  - **TF32 is not a storage dtype.** Model as `math_mode="tf32"` on `fp32` via `numeric_policy`.
  - **Planned/gated dtypes are not first-class.** Entries referencing `uint*`/`complex*`/packed `int4`/`mxfp*`/`bfp*` must declare `metadata.dtype_status = "planned_gated"`.

16. **ZeRO stage 2 only.** `OptimizerShardPass` partitions momentum + variance across `dp` mesh. Stage 3 (parameter sharding) is out of scope.

17. **Pipeline parallelism uses 1F1B by default.** `schedule="interleaved"` requires `micro_batches >= 2 * num_stages`.

18. **RNG streams are deterministically assigned.** `stream_id = global_seed * num_ranks + rank`. Philox counter offsets are non-overlapping for 2^128 elements.

19. **Backends expose hardware-free Target IR before hardware-specific lowering.** Each backend defines an ODS dialect of abstract target ops (`tessera_rocm.mfma`, `tessera_metalium.dma/matmul`, `tessera_apple.cpu.accelerate_gemm`, `tessera_apple.gpu.metal_kernel`) between Tile IR and final hardware emission. The hardware-free layer is what makes backends lit-testable; validated by `test_target_ir_contract.py`.

20. **`@jit(target=...)` accepts both `GPUTargetProfile` and string aliases.** Valid strings: `"rocm"`, `"metalium"`, `"apple_cpu"`, `"apple_gpu"`. Strings dispatch through `matmul_pipeline.py` to `tessera-lower-to-{target}`. Do not invent new string aliases without adding the corresponding pipeline.

21. **Unsupported lowering must emit a stable diagnostic.** When a backend cannot lower an op (e.g., KV-cache on a target without it), emit a diagnostic naming the op and the target — never silently no-op or fall through. See the KV-cache → target lowering for the canonical pattern.

22. **Doc surface is broader than IR/runtime surface — check `docs/guides/` and `docs/programming_guide/` before claiming a feature is missing.** APIs like `tessera.debug.check_grad`/`check_determinism`, replay manifests, and `tessera-mlir` compile-artifact mode are documented and largely implemented but easy to overlook in the source tour.

23. **Tessera is a standalone compiler — no PyTorch / JAX / Flax at runtime.** (S0, locked 2026-05-10.) Torch/aten, jax.lax/jax.numpy/flax/orbax/grain, and equivalents are reference vocabularies only. Nothing in `python/tessera/`, the C++ runtime, or any shipped artifact may import them. Same for data/tokenization (`tf.data`, `torch.utils.data`, `tiktoken`, `tokenizers`, `sentencepiece`). The single concession is *file-format compatibility* (e.g., reading SentencePiece protobufs) — the runtime consuming those bytes must be Tessera's own. Treat "the JAX way" / "torch.optim.AdamW" as vocabulary borrowing: reimplement, don't wrap.

24. **`primitive_coverage.py` is the standalone compiler's audit truth, not `op_catalog.py`.** Catalog = runtime/frontend op acceptor; coverage = audit dashboard (what each primitive must prove across 12 axes). Ship a new primitive → update *both*. The registry auto-flips (V/J)VP axes from registered `_VJPS`/`_JVPS`, and rejects duplicate planned entries. The dashboard is drift-gated.

25. **Registry `partial` ≠ compiler-complete.** Coverage is layered: Python reference, frontend, Graph IR, sharding/transpose/batching, backend manifest, runtime, benchmark proof are separate claims. A row can be useful and still `partial`. The generated dashboards are status truth; do not copy numeric snapshots into prose unless a drift gate owns the copy. When a sprint says "shipped", read the generated rows to see what is actually proven vs. `planned`/`partial`/`reference`/`artifact_only`/hardware-gated.

26. **The audit folder is the canonical "what's done / what's open" surface — follow its flow.** `docs/audit/` = one root audit + theme audits + generated dashboards + theme-local archives:
    1. **Start at `docs/audit/MASTER_AUDIT.md`** — all-up snapshot + P0/P1/P2 queue. Single entry point; do not reconstruct status by grepping.
    2. **Drill into the theme audit:** `compiler/COMPILER_AUDIT.md`, `backend/BACKEND_AUDIT.md` (+ `backend/{apple,nvidia,rocm}/`), `coverage/COVERAGE_AUDIT.md`, `domain/DOMAIN_AUDIT.md`, `roadmap/ROADMAP_AUDIT.md`.
    3. **`docs/audit/generated/` dashboards are count/status truth** (script/test-owned, drift-gated). **Never hand-edit generated docs**; regenerate via their CLI + `scripts/check_generated_docs.sh`.
    4. **`*/archive/` is provenance only** — not the current status surface.
    When you finish audit-relevant work, update the theme audit (and `MASTER_AUDIT.md` if the all-up picture shifts); let generated dashboards carry the numbers.

27. **Ground every Metal / Apple GPU API claim in a real source before declaring it possible or "blocked."** Authoritative sources, in reliability order: **(1) on-machine SDK headers** — `xcrun --show-sdk-path` → `…/System/Library/Frameworks/{Metal,MetalPerformanceShaders,MetalPerformanceShadersGraph,MetalPerformancePrimitives}.framework/Headers/`; **(2) user-provided doc dumps**; **(3) the `apple-metal-docs-urls` memory file**. **WebFetch caveat:** developer.apple.com is a JS-rendered SPA — `WebFetch` returns only the page title, not the API body — so it is NOT a reliable Metal-doc source. Anti-pattern: writing a "blocked / no API path" conclusion from absence of evidence in one source.

---

## Key Design Contracts

**Region privileges.** Modes: `"read"`, `"write"`, `"reduce_sum"`, `"reduce_max"`, `"reduce_min"`. Two write regions on overlapping data → `TesseraConstraintError` at decoration time. `reduce_*` may safely overlap with `read`.

**Domain & distribution** (always separate, Decision #3):
```python
D    = tessera.domain.Rect((B, S, D_model))    # shape only
dist = tessera.dist.Block(mesh_axes=("dp",))   # partition dim-0 over dp axis
X    = tessera.array.from_domain(D, dtype="bf16", distribution=dist)
# X.shard_spec → ShardSpec(partition=(0,), mesh_axes=("dp",)); X.parts("dp") → per-rank slices
```
`Cyclic.parts("dp")` → element `i` on rank `i % dp_size`. Cyclic + Block requires `all_to_all` rebalance (emitted by `distributed_planner.py`).

**TPU constraint.** MXU tile 128×128. `@jit(target=tpu)` auto-injects `Divisible("M/N/K", 128)`.

**FA-4 tile sizes (SM_90).** Default `tile_q=64, tile_kv=64, pipeline_stages=2`, stored as `tessera.tile_q`/`tessera.tile_kv` attrs so the autotuner can sweep them.

**Collective insertion order.** `GPUCollectiveInsertionPass` must run **after** `EffectAnnotationPass` — it reads `tessera.effect = "memory"` on write-region args to find gradient tensors needing `reduce_scatter`.

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
# One-time clone setup: activate committed git hooks (pre-push generated-doc drift gate)
bash scripts/install-git-hooks.sh

# Run the Python flow directly off the Homebrew env (no venv needed — see toolchain below)
python3 -m pytest tests/unit/ -v               # all unit tests
python3 -m pytest tests/unit/test_X.py -v      # single file
python3 -m pytest tests/unit/ -m "not slow"    # default sweep (excludes SuperBench/benchmark)
mypy python/tessera/                            # type check (ratchet baseline: 0)

# MLIR lit tests (requires tessera-opt built)
python3 -m lit tests/tessera-ir/ -v
python3 -m lit tests/tessera-ir/phase7/ -v      # one phase

bash scripts/validate.sh                         # CPU validation spine
```

Heavy SuperBench / benchmark-contract tests are marked `slow` and excluded by default.

---

## Local Toolchain (Homebrew, off-venv)

Everything needed for build / lint / typecheck / lit / unit-test is on Homebrew
on this Mac under `/opt/homebrew/bin/`: `python3` (3.14.5), `ninja`, `cmake`,
`pytest`, `mypy`, `ruff`, `black`, `isort`, `flake8`, `lit`. **LLVM/MLIR 22.1.6**
is `/opt/homebrew/opt/llvm/` (`brew install llvm`). Run the Python flow directly
with `python3 -m …` — no venv. `numpy`, `scipy`, `torch`, `transformers`,
`ml_dtypes` are installed under `/opt/homebrew/lib/python3.14/site-packages/`.

When pointing CMake at LLVM, use `/opt/homebrew/opt/llvm/lib/cmake/{llvm,mlir}`
(NOT the older `llvm@21` path that stale `build/` caches reference).

### Ubuntu 24.04 (x86 + AMD ROCm 7.2.4)

The same tree builds on Ubuntu 24.04. `bash scripts/setup_ubuntu.sh` provisions
LLVM/MLIR 22 from **apt.llvm.org** (ROCm's bundled LLVM has no MLIR), the base
build deps, and a project-local `.venv` — then `source .venv/bin/activate` and
`export PYTHONPATH=python`. CMake LLVM lives at `/usr/lib/llvm-22/lib/cmake/
{llvm,mlir}`; ROCm **7.2.4** at `/opt/rocm` (`-DTESSERA_ENABLE_HIP=ON
-DTESSERA_BUILD_ROCM_BACKEND=ON -DCMAKE_PREFIX_PATH=/opt/rocm`). ROCm kernel
execution stays hardware-gated (Phase H) until a GPU + `/dev/kfd` are present;
the build and lit fixtures need no GPU. The venv caps `numpy<2.2` (numpy ≥2.2
stubs break the `python_version=3.10` mypy ratchet). See `docs/GETTING_STARTED.md`
for the full cross-platform matrix.

---

## C++ Build

```bash
# Canonical local configure — CPU + Apple backend against Homebrew LLVM/MLIR 22 (Ninja)
cmake -S . -B build -G Ninja \
  -DLLVM_DIR=/opt/homebrew/opt/llvm/lib/cmake/llvm \
  -DMLIR_DIR=/opt/homebrew/opt/llvm/lib/cmake/mlir \
  -DTESSERA_CPU_ONLY=ON -DTESSERA_BUILD_APPLE_BACKEND=ON
ninja -C build tessera-opt        # ~150 targets, ~1-2 min cold

# Re-verify a C++ pass change end-to-end: rebuild → lit fixture + FileCheck → drift test
ninja -C build tessera-opt
./build/tools/tessera-opt/tessera-opt tests/tessera-ir/phase8/apple_gpu_lowering.mlir \
  -tessera-lower-to-apple_gpu --allow-unregistered-dialect | \
  /opt/homebrew/opt/llvm/bin/FileCheck tests/tessera-ir/phase8/apple_gpu_lowering.mlir

# Other backend toggles (additive)
cmake .. -DTESSERA_ENABLE_CUDA=ON -DCUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda   # CUDA
cmake .. -DTESSERA_ENABLE_HIP=ON  -DHIP_ROOT_DIR=/opt/rocm                  # ROCm
cmake .. -DTESSERA_BUILD_RUBINCPX_BACKEND=ON                                # RubinCPX

# Benchmarks (stable JSON schema, Decision #12)
python3 benchmarks/run_all.py --backends x86 --output tessera_benchmarks.json
```

### Canonical lowering pipelines (in `tessera-opt`)

| Pipeline | Target |
|----------|--------|
| `tessera-lower-to-x86` | x86 AMX/AVX512 (Phase 2) |
| `tessera-lower-to-gpu` | NVIDIA SM_90+ WGMMA/TMA (Phase 3); `tessera-nvidia-pipeline-{sm90,sm100,sm120}` variants |
| `tessera-lower-to-rocm` / `-metalium` | AMD ROCm MFMA / Tenstorrent Metalium |
| `tessera-lower-to-apple_cpu[-runtime]` | Apple CPU (Accelerate artifact / cblas_sgemm runtime) |
| `tessera-lower-to-apple_gpu[-runtime]` | Apple GPU (Metal artifact / MPS + MSL + MPSGraph runtime; longest-fusion-first ordering) |
| `tessera-cpx-pipeline` / `-context-pipeline` | NV Rubin CPX (separate `tessera-cpx-opt` driver) |

---

## Key Reference Files

| What you need | Where |
|---------------|-------|
| **START HERE — status + open-work queue** | `docs/audit/MASTER_AUDIT.md` (+ theme audits; `docs/audit/README.md` for the map) |
| **Generated dashboards** (count/status truth — never hand-edit) | `docs/audit/generated/` |
| Authoritative API naming | `docs/CANONICAL_API.md` |
| Canonical tensor attributes & dtypes | `docs/reference/tessera_tensor_attributes.md` |
| Apple GPU architecture + kernel inventory | `docs/apple_gpu_overview.md`, `docs/apple_gpu_kernel_inventory.md` |
| NVIDIA / ROCm / Metalium kernel inventories | `docs/nvidia_cuda13_kernel_inventory.md`, `docs/rocm_mfma_kernel_inventory.md`, `docs/metalium_kernel_inventory.md` |
| Graph IR ops / canonicalizations | `src/compiler/ir/TesseraOps.td`, `src/transforms/lib/CanonicalizeTesseraIR.cpp` |
| Schedule IR / FA-4 Tile IR ODS | `src/compiler/programming_model/ir/schedule/ScheduleMeshPipelineOps.td`, `src/compiler/tile_opt_fa4/include/tessera/Dialect/{Attn,Queue}/` |
| Runtime C ABI header | `src/runtime/include/tessera/tessera_runtime.h` |
| IR specs (14 files incl. AUTODIFF_SPEC) | `docs/spec/` |
| User guides + 11-chapter programming guide | `docs/guides/`, `docs/programming_guide/` (check before claiming a feature is missing — Decision #22) |
| Standalone primitive coverage registry / dashboard | `python/tessera/compiler/primitive_coverage.py` / `docs/audit/standalone_primitive_coverage.md` |
| Evaluator program plan | `docs/audit/compiler/EVALUATOR_PLAN.md` §9.5 |
| Target IR contract test | `tests/unit/test_target_ir_contract.py`, `tests/tessera-ir/phase8/target_ir_contracts.mlir` |
| Examples / style guide / structure | `examples/`, `tessera_style_guide.md`, `PROJECT_STRUCTURE.md`, `src/INDEX.md` |

---

## Archive — Do Not Build

`src/archive/` and `docs/archive/` are excluded from all build targets. Do not
add build targets for archived material. New work lands in canonical `src/`
folders only. The verbose pre-2026-06 narrative of this file (full sprint
changelog) is preserved at
`docs/audit/roadmap/archive/CLAUDE_MD_FULL_2026-06-13.md`.

---

## graphify

This project has a knowledge graph at `graphify-out/`.

- For codebase questions, run `graphify query "<question>"` when `graphify-out/graph.json` exists. Use `graphify path "<A>" "<B>"` for relationships and `graphify explain "<concept>"` for focused concepts.
- If `graphify-out/wiki/index.md` exists, use it for broad navigation.
- Read `graphify-out/GRAPH_REPORT.md` only for broad architecture review.
- After modifying code, run `graphify update .` to keep the graph current (AST-only, no API cost).
