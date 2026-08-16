# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

> This is the operational reference. Read it before touching code.
> **For current status, finished/open work, and what to do next, start at
> [`docs/audit/MASTER_AUDIT.md`](docs/audit/MASTER_AUDIT.md) (Decision #26).**
> Counts (entries, tests, symbols) live in `docs/audit/generated/` — do not
> trust or copy numeric snapshots written into prose anywhere, including here.
> Build pin: matched LLVM/MLIR 23.

---

## How to Read This File

**Most of what follows is a default, not a law.** Depart from a default when the
code, a test, or a device result says you should — and say so in the PR. What
makes a default worth keeping is that it encodes a reason someone already paid
for; what makes it a default is that the reason can stop applying.

Two kinds of item are **not** defaults, because departing from them silently
does damage that lands on someone else:

| Tier | What it means | Examples |
|---|---|---|
| **Default** | Depart with a recorded reason. Judgment is expected; scale the effort to the blast radius. | Which gates to run for a given change; explore via graphify before grepping; how far to read the backend queues |
| **Hard — claim integrity** | Never assert more than the evidence supports. A false claim gets acted on downstream, and the cost is paid by whoever believes it. | Device work runs on the box with that device (the sandbox has no CUDA/ROCm, and a missing device *skips* rather than errors); native execution proven separately from the `reference_cpu` lane; no parity claim for a target without *that* hardware's proof; no green report over a red lane (Decisions #19, #25, #26) |
| **Hard — destructive / not yours** | Ask first. These are irreversible or belong to the repo owner, and an agent lacks the context to judge the cost. | Deleting branches, rewriting history, discarding unrelated dirty work, force-push (see **Working Rules**) |

This taxonomy mirrors what the code already does: Decision #21a splits attributes
into *semantic keys* that fail closed on absence and *performance keys* that may
fall back with a diagnostic. Same idea, applied to the prose.

**The asymmetry is deliberate.** A rule can be a default for the repo owner and
a hard constraint for an agent working in it. You can rewrite a branch because
you know what is on it; an agent reasoning from a partial view does not, and the
cost of being wrong is not symmetric.

---

## What Tessera Is

Tessera is a **pre-alpha, standalone, tile-centric programming model and
compiler** for deep learning and HPC. Tiles, explicit memory spaces, numerical
precision, and parallelism are **first-class IR objects** — not runtime
heuristics. "Standalone" means runtime-independent of PyTorch / JAX / Flax
(Decision #23); those are reference vocabularies only.

Target hardware: NVIDIA (SM90 Hopper, SM100 Blackwell), AMD ROCm,
x86 AMX/AVX512, Apple M-series CPU/GPU.

**Execution reality (updated 2026-08-15):** the **x86 AMX/AVX512** backend and
**Apple CPU (Accelerate) + GPU (MPS/MSL/MPSGraph)** backends execute natively.
**ROCm gfx1151** (Strix Halo, RDNA 3.5) now has **broad native execution, not
merely matmul + flash-attention** — the generated execution matrix records
dozens of `native_gpu` family rows (attention family, norms/activations,
matmul-family compositions, MoE transport, SSM fwd/bwd, EBM, warp-shuffle
reduce/scan/argreduce lanes); the remaining boundary is exact-device expansion,
not a missing launcher. **NVIDIA sm_120** (RTX 5070 Ti, consumer Blackwell)
likewise executes a broad compiled family (attention fwd/bwd, conv2d, MLA
decode, SSM, bounded control flow, MoE transport, optimizer/loss backwards) —
no longer just an `mma.sync` matmul. Datacenter archs (ROCm CDNA/MI300; NVIDIA
Hopper sm_90 / datacenter sm_100) stay hardware-gated (Phase G/H). Everything
else produces IR/artifacts until a hardware-gated proof row says otherwise —
read `docs/audit/generated/runtime_execution_matrix.md` for what is actually
proven, never counts copied into prose (Decision #26). See
[`docs/audit/backend/BACKEND_AUDIT.md`](docs/audit/backend/BACKEND_AUDIT.md).

**"Executes natively" is a runtime claim, not a compiler-maturity claim — read
this before scoping Apple work (reviewed 2026-07-28).** The Apple GPU op
envelope is delivered by the *runtime*, not by generated code:
`runtime/apple_gpu_runtime.mm` is ~27k lines (≈69% of the whole Apple backend)
holding ~123 **hand-written** MSL kernels as raw-string literals, reached by
Python `runtime.launch()` → ctypes. The MLIR side is much smaller than the op
counts suggest: the value-preserving Target-IR lane executes **9 GPU op kinds**
(`runtime.py::_execute_apple_value_target_ir_gpu_artifact`), the lowering passes
emit `func.call` to a C symbol rather than generated kernels, and
`tessera_apple.gpu.msl_kernel` — the op that would carry compiler-synthesized
MSL — has **no C++ producer at all** (only Python `compiler/target_ir.py` sets
`msl_source`). Do not read "native" as "the MLIR pipeline generated it."

**But the synthesizer is not missing — it is on the other side of a seam.**
Decision #28 tier 1/2 *is* built, in Python under
[`python/tessera/compiler/emit/`](python/tessera/compiler/emit/):
`apple_msl.py` is the worked reference (matmul-epilogue plain/tiled/**coopmat
`simdgroup_matrix`**, norm-chain, pointwise-graph, pointwise-reduce, attention
incl. online-softmax, gated-matmul), alongside `nvidia_cuda.py`,
`rocm_hip.py`, `x86_llvm.py`, over the arch-agnostic regions in
`compiler/fusion_core.py` (one level up from `emit/`) and the
`KernelEmitter`/`compile_fn`/`KernelRunner` seams in `emit/kernel_emitter.py` +
`emit/kernel_cache.py`. The Workstream C handoff is archived at
`docs/audit/compiler/archive/WORKSTREAM_C_HANDOFF.md`; route current compiler
work through [`docs/audit/compiler/README.md`](docs/audit/compiler/README.md).

**So the real Apple gap is not "no emitter" — it is that the Python synthesizer
and the C++ MLIR pipeline are two disconnected compilers.** The MLIR lane
lowers to `func.call` on hand-written runtime symbols and never consults the
synthesizer; the synthesizer emits and caches source without going through
Target IR. Anyone scoping Apple work should be closing that seam, not writing a
second MSL emitter.

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
Target IR   (per-backend: NVIDIA, ROCm, Apple, x86)
```

New backends MUST expose a hardware-free Target IR dialect before
hardware-specific lowering (Decision #19) — never lower Tile IR directly to
PTX/HIP/Metal source.

---

## Phase Status (high level)

| Phase | Status | Scope |
|-------|--------|-------|
| 1–6 | ✅ Complete | Python frontend → C++ lowering → NVIDIA backend IR → distributed training → solver passes/autotuner → runtime wrapper + CUDA/HIP backends |
| 7 | 🟢 Lit-verified | Neighbors (halo/stencil) dialect; real HW gated on Phase G/H |
| 8 | 🟢 Apple operational | Hardware-free Target IR; `@jit(target="rocm"/"apple_cpu"/"apple_gpu")`; Apple CPU (Accelerate) + GPU (MPS + MSL + MPSGraph) execute natively |
| S-series | 🟢 In progress | Standalone-compiler track — primitive contract registry + S2–S15 Python reference surface + reasoning-model attention/RL; `backend_kernel` axis is the long-pole gate (Phase G/H) |
| W-series | 🟢 In progress | Compiler-contract track — W0 governance landed; **W1.1 typed Tile IR: ROCm steps 1–4 + typed performance closure landed (typed route is now the canonical gfx1151 selection; NVIDIA producers + permissive-branch deletion open)**; W2.1 `GraphDataflowAnalysis` and W2.2 IR-derived effects **closed** (2026-08-10/11). Top active program: **E2E-REAL-6, one compiler authority** (tracer becomes sole general frontend; `_OpExtractor` retired after differential proof). See `INTEGRATED_COMPILER_PLAN.md` + MASTER_AUDIT §1 |
| RubinCPX | 📦 Archived | Retired 2026-06-08 with TPU/Metalium/Cerebras (focus = x86 + Apple + NVIDIA + ROCm); material under `archive/`, no build target |

Per-phase deliverables and the open-work priority queue live in
`docs/audit/MASTER_AUDIT.md` and the theme audits.

---

## Key Source Locations

### Python package (`python/tessera/`)

| Module | Purpose |
|--------|---------|
| `__init__.py` | Top-level exports: `jit`, `kernel`, `Region`, `domain`, `dist`, `array`, `index_launch`, `constraint`, `ops`, `Tensor`, `dtype`. `train` is lazily bound via PEP 562 `__getattr__`. |
| `dtype.py` | Canonical dtype enforcement + `Dtype` typed object + promotion lattice (`canonicalize_dtype`, `result_type`, `is_canonical_dtype`, …). Canonical 15-name set; aliases normalize at API boundaries; `tf32` rejected as storage dtype (use `numeric_policy.math_mode`). See Decision #15a. |
| `compiler/jit.py` | `@jit`/`@kernel` decorators; routes to x86, GPU, or string-target pipeline. Call-time constraint re-check via `JitFn._enforce_call_time_constraints`. **In migration (E2E-REAL-6):** family selection / package construction is moving out of `JitFn` into the tracer as the one compiler authority; the AST `_OpExtractor` is deleted only after differential execution covers each migrated family — see `MASTER_AUDIT.md` §1. |
| `compiler/op_catalog.py` | Canonical op-name catalog — "what we accept today" across all IR layers. |
| `compiler/primitive_coverage.py` | **Audit truth** (Decision #24) — standalone primitive contract registry over 12 axes; consults `autodiff.vjp._VJPS`/`jvp._JVPS` so registered (V/J)VPs auto-flip to complete. Renders `docs/audit/standalone_primitive_coverage.md`. |
| `compiler/backend_manifest.py` | Per-op × per-target × per-dtype kernel manifest synthesizer; `BackendKernelEntry` + statuses `fused`/`reference`/`compileable`/`artifact_only`/`planned`. |
| `compiler/gpu_target.py` / `rocm_target.py` | Target profiles + feature matrices. NVIDIA pinned CUDA 13.3; AMD pinned ROCm 7.2.4. |
| `compiler/{constraints,effects,graph_ir}.py` | `ConstraintSolver` (decoration-time), `EffectLattice` (`pure<random<memory<io<top`, derived from registered traced Graph IR since W2.2 — see Decision #5), Python→Graph IR emission. |
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
| `compiler/tile_opt_fa4/include/tessera/Dialect/Attn/*.td` | FA-4 Attn Tile IR dialect (the dead `Queue` dialect was deleted 2026-08-10 per Decisions #29/#31) |
| `compiler/codegen/tessera_x86_backend/` | AMX BF16 + AVX512 GEMM — **works end-to-end** |
| `compiler/codegen/Tessera_Apple_Backend/` | Apple CPU + GPU — **operational**. CPU: `MatmulToAppleCPU` + Accelerate shim. GPU: 17-pass Tile→Apple lowering + Objective-C++ runtime (`apple_gpu_runtime.mm`) with MPS/MSL/MPSGraph lanes. |
| `compiler/codegen/{tessera_gpu_backend_NVIDIA,Tessera_ROCM_Backend}/` | Per-target backends (IR/artifact; HW execution gated where noted). RubinCPX/TPU/Metalium/Cerebras backends are retired to `archive/` (2026-06-08) |
| `compiler/tessera_neighbors/` | Halo/stencil neighbor-exchange dialect (Phase 7) |
| `transforms/lib/*.cpp` | Pass bodies — Canonicalize/Verify/Migrate (P1), Distribution/Effect/Tiling/TileToX86 (P2), TileIRLowering/WarpSpec/AsyncCopy/WGMMA/TMA (P3), Collective/PipelineStage (P4), `AttentionFamilyPasses.cpp` (reasoning-model attention) |
| `solvers/` | Core (11 passes), linalg, scaling-resilience, spectral (6 pass bodies + `ts-spectral-opt`), TPP (7 passes + `tpp-space-time`) |
| `collectives/` | `CollectiveOps.td`, `NCCLAdapter`/`RCCLAdapter` (+ mock paths), `ExecRuntime` chunk submit + `TokenLimiter`; the overlap scheduler is an unimplemented draft (`src/collectives/docs/Tessera_Collectives_Overlap_Design.md` §4) |
| `runtime/src/` | `tessera_runtime.cpp` (C ABI) + CUDA/HIP/CPU backends (real calls) |

### Tools (`tools/`)

| Path | Purpose |
|------|---------|
| `tessera-opt/` | MLIR opt-style driver — all dialects + 70+ passes + named lowering pipelines. Build: `ninja -C build tessera-opt`. |
| `tessera-translate/` | C++ `tessera-translate-mlir` (MLIR↔LLVM IR / SPIR-V) + Python `tessera-translate` (StableHLO/GGUF/SafeTensors export) |
| `profiler/` / `roofline_tools/` | tprof runtime + Perfetto export; roofline ingestion + HTML reports |
| `scripts/validate.sh` / `check_versions.py` / `check_generated_docs.sh` | CPU validation spine; version-drift check; generated-doc drift gate (pre-commit) |

---

## Architecture Decisions — Stable by Default, Amend with Evidence

> Treat these as load-bearing. Do not re-open one because it looks wrong at a
> glance, and never re-litigate from first principles without reading the code
> it governs — that is churn, and it is the main source of contradictory
> direction.
>
> **Do re-open when evidence demands it:** a failing lane, a device result, or a
> code read that contradicts the decision. When you do, amend in place — a dated
> banner (`Corrected YYYY-MM-DD` / `root-caused` / `withdrawn`) or a
> letter-suffixed sub-decision (#10a, #15a, #21a, #26a are the existing ones) —
> so the next reader sees what changed and why. Never silently edit a decision
> to match new behavior; the record of the change is the point.
>
> **A decision the code no longer matches is a bug in this file, not a
> constraint on the code.** As of 2026-08-15 this section carries 13 recorded
> amendments: six `Adopted 2026-08-02` entries, four letter-suffixed
> sub-decisions, and three dated banners (`Updated 2026-08-10` on #5,
> `root-caused 2026-08-15` on #19, one `withdrawn` claim in #26a). Decision #5
> has now been amended twice — fail-open AST inference was recorded as a
> correction, then superseded when W2.2 made inference IR-derived and
> fail-closed. That is the protocol working, not a defect. It was previously
> headed "Do Not Revisit," which was never true of its own contents.

1. **CPU-first, then GPU.** x86 AMX is the only real execution path on the original roadmap; GPU ops gated behind `target_profile.isa >= SM_90`. (Apple is the second native lane, Phase 8.)

2. **`Region` is a type annotation, not a runtime wrapper.** `Region["read"]` returns a `RegionType` object. It does NOT wrap tensors at runtime.

3. **Domains and distributions are always separate.** `Rect` = shape. `Block/Cyclic/Replicated` = placement. Never merge them.

4. **`ConstraintSolver` runs at decoration time.** `@jit` inspects annotations and calls `ConstraintSolver.check(signature)` before IR emission. Violations → `TesseraConstraintError`.

5. **Effects are inferred, not declared.** Programmers only declare
   `@jit(deterministic=True)` and `@jit(seed=N)` at the top level.
   **Updated 2026-08-10 — W2.2 closed; inference is IR-derived and fails
   closed.** `EffectLattice` (`python/tessera/compiler/effects.py`) now joins
   registered effects across **traced Graph IR records**: the canonical op
   catalog emits explicit `tessera.effect_kind`, the old Python-AST
   `_EffectVisitor` is **deleted**, and unknown behavior joins to `top`
   (fail-closed) — so RNG reached through an alias, wrapper, `getattr`, or dict
   dispatch no longer slips past `@jit(deterministic=True)`. The generated
   `docs/audit/generated/effect_lattice_audit.md` dashboard tracks how many ops
   sit at the conservative fallback. Separately,
   `src/transforms/lib/EffectAnnotationPass.cpp` computes effects on the MLIR
   side — that is the one `GPUCollectiveInsertionPass` orders against.
   History of the old fail-open AST design:
   [`docs/audit/compiler/COMPILER_ARCHITECTURE_SWEEP.md`](docs/audit/compiler/COMPILER_ARCHITECTURE_SWEEP.md) §F1.

6. **Mock collectives use threads, not processes.** Multi-rank tests run in-process via `MockRankGroup`. No NCCL/MPI dependency in the test suite.

7. **`tessera.array` is not `numpy.ndarray`.** `DistributedArray` carries a `ShardSpec` and logical shape. Physical storage is backend-dependent; on CPU it is a numpy array.

8. **Warp role separation is structural, not advisory.** `WarpSpecializationPass` emits hard `tessera.schedule.warp {role="producer/consumer"}` boundaries. Different register files and barrier slots per role.

9. **TMA descriptors are generated once per kernel, not per tile.** `NVTMADescriptorPass` hoists descriptor setup to kernel preamble.

10. **Recompute insertion is budget-guided.** `InsertRecomputePass` uses `--memory-budget-mb` and a greedy live-set scan. Only pure ops qualify for recomputation.

10a. **An eligibility-marking pass ships a negative fixture.** (Adopted 2026-08-02, W0.8.) Any pass that annotates work as rematerializable, fusable, or pipelineable must gate on a demand/liveness analysis and ship **at least one lit fixture whose correct output is _no annotation_** (`CHECK-NOT`). Marking everything is not a conservative default — it is an unmeasured cost that a downstream pass will believe. Derived from `EBMCheckpointInnerLoop` marking every step of every loop rematerializable with no analysis, contradicting #10's own live-set discipline.

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

19. **Backends expose hardware-free Target IR before hardware-specific lowering.** Each backend defines an ODS dialect of abstract target ops (`tessera_rocm.mfma`, `tessera_apple.cpu.accelerate_gemm`, `tessera_apple.gpu.metal_kernel`) between Tile IR and final hardware emission. The hardware-free layer is what makes backends lit-testable; validated by `test_target_ir_contract.py`.

    **`X86-DIALECT-LOAD-CRASH-2026-08-12` was a build-flag leak, not an IR
    defect — root-caused 2026-08-15.** The dialect and its `TileType`
    registration were always correct. The x86 kernel project applied its
    detected AVX-512/AMX flags with **`add_compile_options`**, which is
    *directory* scoped, and `add_subdirectory(lib/IR)` sat below that call — so
    the hardware-free Target IR dialect was compiled `-mavx512f … -mamx-tile …`
    and the compiler emitted an AVX-512-only encoding into dialect registration
    itself. On a host with AVX-512 that runs fine; on the CI runner it does not,
    and `tessera-opt` died the first time it touched the dialect. **The tell was
    the signal: all 14 fixtures failed with SIGILL (signal 4) at one identical
    address, not SIGSEGV** — an illegal instruction is a build-configuration
    fact, and `Dialect::addType<TileType>()` was merely the first code from that
    translation unit to execute. Flags are now applied per-target to the kernel
    targets only, and `lib/IR/CMakeLists.txt` **fails configure** if any
    host-specific ISA flag is in scope.

    **Standing lesson, because it will recur: a host that has the ISA cannot
    falsify a host-portability claim.** This P0 was closed once on Zen 5 —
    correctly, for that host — and CI reopened it. Decision #19's "lit-testable
    on any host" is only evidenced by a host *without* AVX-512.

    **Separately, still open: `TileToX86Pass` loads `tessera_x86` from inside
    `runOnOperation()`** (`src/transforms/lib/TileToX86Pass.cpp:1045`, a by-name
    `getOrLoadDialect` used to avoid linking the optional backend). MLIR forbids
    loading a dialect during pass execution; on an **assertions-enabled** LLVM
    this is a hard `LLVM ERROR` that fails 12 of the x86 fixtures, and on an
    NDEBUG build (what CI uses) it is silently undefined behavior. CI cannot see
    it. Fixing it means declaring the dialect in `getDependentDialects()`, which
    couples `TesseraPasses` to the optional `TesseraX86IR` — a layering call.
    Status: [`docs/audit/backend/x86/todo.md`](docs/audit/backend/x86/todo.md) P0.

    **`tessera_x86` now exists — x86 complies (built 2026-08-02, W0.10).** It was previously the one backend with no Target IR dialect at all: `TileToX86Pass` lowered Tile IR to 21 `func::CallOp`s into a hand-written C shim, and the Python emitter named a `tessera_x86.func` op no dialect defined. **No carve-out was granted.** The dialect lives at `src/compiler/codegen/tessera_x86_backend/include/TesseraX86/IR/`, is registered in `tessera-opt`, and splits into value-carrying ops (`amx_tile_load`/`amx_tile_zero`/`amx_dpbf16ps`/`amx_dpbusd`/`amx_tile_store` over a real `!tessera_x86.tile` type, so the verifier rejects a tile dot-product whose operands never came from a tile load) and directives (`avx512_gemm_microkernel`, `pack_b_panel`, `elementwise`, `kernel`, `kv_cache_read`, `unsupported`). `abi_call` **models the C-shim boundary instead of hiding it**, so Decision #28's arbiter can tell compiler-generated work from delegated work. Fixtures: `tests/tessera-ir/phase2/x86_target_ir{,_invalid}.mlir` — including a negative case, since a dialect that only ever accepts proves nothing. **Remaining follow-on is `x86vector.*` (AVX-512) lowering only; the AMX half is optional** — per project direction (2026-08-02) AMX is expected to be superseded by the ACE matrix instructions jointly agreed by Intel and AMD, so the AMX ops stay as the IR-level contract without an `amx.*` lowering. That also removes the hardware blocker: AVX-512 runs on the primary box, while no fleet machine has AMX.

    **Decision #19 is now checked by MLIR, not by substring (W0.9).** `test_target_ir_contract.py` keeps its `in`-assertions as smoke coverage and adds a real parse + dialect-load + verifier run over every emitter and every committed golden. That gate immediately found that **no** Python-emitted Target IR was valid MLIR — undialect-prefixed module attributes, an invented `<dialect>.func` container, ops emitted with signatures their ODS rejects, and five op names (`tessera_rocm.elementwise`, `tessera_rocm.kv_cache_read`, `tessera_rocm.msa_block_sparse`, `tessera_apple.cpu.kv_cache_read`, `tessera_apple.cpu.moe_solver`) that no dialect declared. All fixed. The portable `cpu` reference lane is closed too: it emitted `tessera.cpu.<source-op>`, one name per Graph IR op, which could never be enumerated in ODS — and that name was redundant, since the CPU verifier already requires a `source` attribute. It now emits the single declared `tessera.cpu.reference` node. **No exclusions remain**: every target whose dialect the build compiles (`cpu`, `x86`, `rocm`, `apple_cpu`, `apple_gpu`) passes a real parse + verify; NVIDIA skips only because its dialect is off by default.

20. **`@jit(target=...)` accepts both `GPUTargetProfile` and string aliases.** Valid strings: `"rocm"`, `"apple_cpu"`, `"apple_gpu"`. Strings dispatch through `matmul_pipeline.py` to `tessera-lower-to-{target}`. Do not invent new string aliases without adding the corresponding pipeline.

21. **Unsupported lowering must emit a stable diagnostic.** When a backend cannot lower an op (e.g., KV-cache on a target without it), emit a diagnostic naming the op and the target — never silently no-op or fall through. See the KV-cache → target lowering for the canonical pattern.

21a. **Semantic keys never default.** (Adopted 2026-08-02, W0.8.) An attribute that selects *semantics* fails **closed** on absence — emit a diagnostic and stop; it may never be silently defaulted. An attribute that selects *performance* may fall back, but must say so with a diagnostic. Semantic keys: `manifold`, `algebra`, `math_mode`, `rounding_mode`, `distribution`, `dtype`. Performance keys: tile sizes, stage depth, `auto_batch`, checkpoint budget. Derived from `EBMCanonicalize` defaulting a missing `manifold` to `"euclidean"` — a first-order-correct Euclidean fallback converges and reports a **wrong** result rather than an error. Corollaries: no `operand_types[0]` shape fallback, and no unvalidated `StrAttr` where an `EnumAttr` states the legal set.

22. **Doc surface is broader than IR/runtime surface — check `docs/guides/` and `docs/programming_guide/` before claiming a feature is missing.** APIs like `tessera.debug.check_grad`/`check_determinism`, replay manifests, and `tessera-mlir` compile-artifact mode are documented and largely implemented but easy to overlook in the source tour.

23. **Tessera is a standalone compiler — no PyTorch / JAX / Flax at runtime.** (S0, locked 2026-05-10.) Torch/aten, jax.lax/jax.numpy/flax/orbax/grain, and equivalents are reference vocabularies only. Nothing in `python/tessera/`, the C++ runtime, or any shipped artifact may import them. Same for data/tokenization (`tf.data`, `torch.utils.data`, `tiktoken`, `tokenizers`, `sentencepiece`). The single concession is *file-format compatibility* (e.g., reading SentencePiece protobufs) — the runtime consuming those bytes must be Tessera's own. Treat "the JAX way" / "torch.optim.AdamW" as vocabulary borrowing: reimplement, don't wrap.

24. **`primitive_coverage.py` is the standalone compiler's audit truth, not `op_catalog.py`.** Catalog = runtime/frontend op acceptor; coverage = audit dashboard (what each primitive must prove across 12 axes). Ship a new primitive → update *both*. The registry auto-flips (V/J)VP axes from registered `_VJPS`/`_JVPS`, and rejects duplicate planned entries. The dashboard is drift-gated.

25. **Registry `partial` ≠ compiler-complete.** Coverage is layered: Python reference, frontend, Graph IR, sharding/transpose/batching, backend manifest, runtime, benchmark proof are separate claims. A row can be useful and still `partial`. The generated dashboards are the **primary current-status evidence** — reconcile them against implementation, tests, and exact-device proof when they conflict or look stale; do not copy numeric snapshots into prose unless a drift gate owns the copy. When a sprint says "shipped", read the generated rows to see what is actually proven vs. `planned`/`partial`/`reference`/`artifact_only`/hardware-gated.

26. **The audit folder is the canonical "what's done / what's open" surface — follow its flow.** `docs/audit/` = one root audit + theme audits + generated dashboards + theme-local archives:
    1. **Start at `docs/audit/MASTER_AUDIT.md`** — all-up snapshot + P0/P1/P2 queue. Single entry point; do not reconstruct status by grepping.
    2. **Drill into the theme audit:** `compiler/COMPILER_AUDIT.md`, `backend/BACKEND_AUDIT.md` (+ `backend/{apple,nvidia,rocm,x86}/` per-backend todo queues), `coverage/COVERAGE_AUDIT.md`, `domain/DOMAIN_AUDIT.md`, `roadmap/ROADMAP_AUDIT.md`. For compiler work specifically, `docs/audit/compiler/README.md` states the **authority chain** (generated dashboards > COMPILER_AUDIT > INTEGRATED_COMPILER_PLAN > scoped plans > backend todos) — the integrated plan is the sole cross-domain compiler queue and wins when a scoped plan proposes a different order.
    3. **`docs/audit/generated/` dashboards are the primary count/status evidence** (script/test-owned, drift-gated). Note what the gate does and does not prove: `check_generated_docs.sh` **byte-compares each committed doc against its generator**, so it catches a stale doc, not a wrong generator model. A dashboard can be green and still overstate reality — Decision #24's registry auto-flips a (V/J)VP axis to complete on *registration of a numpy reference*, which is not a test and not device proof. Reconcile against implementation/tests/device evidence when a row looks stale or contradicts what you read in the code. **Never hand-edit generated docs**; regenerate via their CLI + `scripts/check_generated_docs.sh`.
    4. **`*/archive/` is provenance only** — not the current status surface. Historical checklists and sprint prose remain useful context for *why* something was built; they are not sufficient on their own to establish that it is done.
    When you finish audit-relevant work, update the theme audit (and `MASTER_AUDIT.md` if the all-up picture shifts); let generated dashboards carry the numbers.

26a. **There *is* an LLVM IR → AIR path to Apple GPUs; it is just not an open
    one (researched 2026-07-28, direction not yet chosen).** Mojo compiles
    Mojo → LLVM IR → **AIR bitcode** → `.metallib` via metal-cpp, requiring
    Xcode 16+ and `xcodebuild -downloadComponent MetalToolchain`. AIR *is* LLVM
    bitcode, so third parties can and do emit it. But the upstream
    [LLVM `air64` backend RFC](https://discourse.llvm.org/t/rfc-add-an-apple-metal-air-backend-target/90936)
    (May 2026) is **stalled** on reverse-engineering/legal exposure, no Apple
    participation, and an AI-disclosure policy violation — and AIR itself is
    undocumented, so any AIR emitter is black-box-derived and needs Apple's
    tools to package the container. Do not write "no LLVM path to Apple GPU"
    (the old claim) and do not treat AIR as a supported one.

    **Decision (2026-07-28): a direct AIR emitter is deferred — add the
    interface when a measured need appears.** SPIR-V → SPIRV-Cross → MSL is
    rejected outright (it cannot express `simdgroup_matrix`, so it caps the
    Apple ceiling the arbiter exists to protect). The shipped lane is
    **MSL synthesis + AOT packaging**: `emit/apple_air.py` compiles synthesized
    MSL through `xcrun metal -c` → `.air` → `xcrun metallib` and loads it with
    `newLibraryWithURL:`, entirely on supported tooling. What killed the
    urgency is the measurement (APPLE-AOT-1): AOT already captures the whole
    front-end saving — cold pipeline creation 29.7 ms → 15.2 ms — and the
    ~15.2 ms that remains is AIR → GPU-ISA, which *any* AIR-based path still
    pays. So direct AIR emission would save **the same ~15 ms and no more**.
    Its remaining case is architectural (sharing LLVM lowering with
    CUDA/ROCm/x86), not performance; revisit on that basis, not on speed.

    This is the fast-path pattern the whole fleet converges on — a precompiled
    artifact plus a content-addressed cache — but **do not read Apple as the
    leader here.** Counted 2026-07-28: ROCm's dominant lane is already
    precompiled hsaco built by `tessera-opt` itself (`convert-gpu-to-rocdl` →
    `rocdl-attach-target` → `gpu-module-to-binary`, ~601 references in
    `runtime.py`), with a smaller HIPRTC-at-load WMMA lane; NVIDIA's device code
    is NVRTC-compiled at load with no cubin lane in `runtime.py`. So ROCm is
    *ahead* of Apple on AOT, NVIDIA is closest to Apple's JIT-dominant position,
    and Apple's `.metallib` is one kernel old **and shelled out to `xcrun` from
    Python rather than produced by the compiler** — which is the real
    architectural gap, and the strongest remaining argument for an MLIR → AIR
    path. Every backend has both a JIT and a precompiled capability; an earlier
    claim here that NVIDIA/ROCm "have no JIT lane" was inferred from
    `nvrtc`/`hiprtc` being absent from two Python files and is withdrawn.
    When ROCm or CUDA reach the same AOT-vs-JIT question, reuse
    `benchmarks/apple_gpu/benchmark_aot_vs_jit.py` **including its cache
    control**: measure a never-before-compiled kernel per sample, or you will
    measure the vendor's shader cache instead of your compile strategy.

27. **Ground every Metal / Apple GPU API claim in a real source before declaring it possible or "blocked."** Authoritative sources, in reliability order: **(1) on-machine SDK headers** — `xcrun --show-sdk-path` → `…/System/Library/Frameworks/{Metal,MetalPerformanceShaders,MetalPerformanceShadersGraph,MetalPerformancePrimitives}.framework/Headers/`; **(2) user-provided doc dumps**; **(3) the `apple-metal-docs-urls` memory file**. **WebFetch caveat:** developer.apple.com is a JS-rendered SPA — `WebFetch` returns only the page title, not the API body — so it is NOT a reliable Metal-doc source. Anti-pattern: writing a "blocked / no API path" conclusion from absence of evidence in one source.

28. **The forward compiler direction is the three-tier / measured-arbiter model — leads set the ceiling, the generic framework raises the floor.** (North star, 2026-07-02.) Kernels come from three tiers: **(1)** a generic synthesizer (arch-agnostic region-IR + F4 oracle + synth→compile→cache→launch loop), **(2)** a per-arch codegen plugin (`KernelEmitter`/`TargetPlugin`: MSL / PTX / AMDGCN / C-LLVM), **(3)** hand-tuned kernels. A **measured, accuracy-budgeted arbiter** picks the fastest *in-budget* candidate per `(op, shape-bucket, dtype, target)`. **ROCm and CUDA are the lead performance targets: shared infra must never cap their ceiling** — hand-emitted `wgmma`/`mma.sync`/MFMA/WMMA stay first-class arbiter candidates, displaced only when a compiled kernel is both faster and in accuracy budget. The synthesizer/plugin interface is **symbolic-dim-aware from day one** (`static | bucket | dynamic` policy; first impls bucket-specialize) so dynamic shapes never force an API break. Full model: [`docs/audit/compiler/COMPILER_THEORY_OF_OPERATION.md`](docs/audit/compiler/COMPILER_THEORY_OF_OPERATION.md); execution: [`docs/audit/compiler/COMPILER_REFACTOR_PLAN.md`](docs/audit/compiler/COMPILER_REFACTOR_PLAN.md). These are *direction*; MASTER_AUDIT + generated dashboards stay the primary status evidence.

29. **A declaration must have a consumer.** (Adopted 2026-08-02, W0.8.) If the compiler declares metadata — an ODS type or attribute, a `primitive_coverage` axis, a coverage claim — a **named pass or code path must consume it**, or the declaration is deleted. A declaration with no consumer is worse than a missing one: it reads as a closed contract in review and in the dashboards while carrying nothing. Drift-gated by `tests/unit/test_governance_declarations.py`. Derived from seven independent instances (`manifold` reaching no backend; `MultivectorSpec.grades` ignored by `geometric_product`; a closed `batching_rule` axis whose `vmap` is a Python for-loop; a closed `shape_rule` axis whose inference is a five-case if-chain; nine declared `!tile.*` types alongside 70 `Variadic<AnyType>`; `numeric_policy` with no carrier below Graph IR; `TilingInterface` unused by `fusion_core.py`).

30. **Derive, don't ask.** (Adopted 2026-08-02, W0.8.) A pass that needs a program fact — purity, activity, liveness, fusion legality, shape equality, sharding — **queries the analysis layer**; it does not accept the fact syntactically and it does not hand-roll an eighth bespoke walker. New ad-hoc analyses are rejected in review. Told-not-derived facts are wrong at the edges and **fail open** — the canonical scar is the old AST-based `EffectLattice` letting an aliased RNG call pass `@jit(deterministic=True)` (fixed by W2.2, see Decision #5). **The analysis layer now exists:** W2.1's `GraphDataflowAnalysis` (closed 2026-08-11) runs shape/alias product lattices + liveness on MLIR `DataFlowSolver`, derives value-scoped memory dependence from registered effects/resources, and exposes reverse activity, with explicit `invalidate`/`recompute`. Query it. A fact it cannot derive still fails closed (treat unprovable as unsafe), never the permissive answer.

31. **One implementation per boundary.** (Adopted 2026-08-02, W0.8.) Each IR level boundary has exactly **one production lowering**. A second implementation is either (a) a **declared oracle** with a differential test against the production path, or (b) deleted. Same rule for frontends and AD engines. Drift-gated by `tests/unit/test_governance_declarations.py`. **Ordering caveat:** do not collapse a duplication before the surviving path can carry what the deleted one carried — that is the documented way this fails; see the W0→W1→W2→W3 ordering in `INTEGRATED_COMPILER_PLAN.md`.

32. **Information loss across a level boundary must be declared.** (Adopted 2026-08-02, W0.8.) A lowering either carries each Decision #15a attribute (`layout`, `numeric_policy`, `distribution`, …) forward, or **records a named reason it dropped it**. A boundary verifier fails on silent loss. Derived from `numeric_policy` vanishing above the MMA — the accumulator contract is stated at Graph IR and no longer exists by the time codegen picks an instruction.

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

**FA-4 tile sizes (SM_90).** Default `tile_q=64, tile_kv=64, pipeline_stages=2`, stored as `tessera.tile_q`/`tessera.tile_kv` attrs so the autotuner can sweep them.

**Collective insertion order.** `GPUCollectiveInsertionPass` must run **after** `EffectAnnotationPass` — it reads `tessera.effect = "memory"` on write-region args to find gradient tensors needing `reduce_scatter`.

---

## GPU-Only Tier — Never Implement on CPU

Gate all of these behind `target_profile.isa >= ISA.SM_90`:

- `tessera.schedule.warp` role assignments (FA-4 warp specialization)
- `tile.tcgen05.mma` (Blackwell TMEM MMA) — the mnemonic is `tcgen05.mma`, not `mma.tcgen05`; the latter spelling came from a parallel `tile` ODS deleted in W0.6 that nothing compiled
- `tile.async_copy` / `tile.wait_async` stage indexing
- `tessera.schedule.policy "persistent"` (persistent CTA scheduling)
- `tcgen05.mma` PTX inline asm

(The `tessera.queue.*` tile-queue dialect that used to sit in this list was
deleted 2026-08-10 as dead IR — Decisions #29/#31.)

---

## Working Rules

Tiered per **How to Read This File**. The hard ones are hard *for an agent*
specifically — they are irreversible, or they make a claim someone else acts on.

### Hard — ask first (destructive / not yours)

- **Never delete a branch, rewrite history, force-push, or discard unrelated
  dirty work without explicit approval.** This includes `git branch -d` after a
  merge, `rebase`, `reset --hard`, `commit --amend` on pushed work, and
  `stash`/`clean` over changes you did not make. All of these look cheap from
  inside a single session and are judged with a partial view of what they
  destroy. Ask, and say what you are about to remove.
- **Preserve unrelated dirty work.** If the tree has changes you did not make,
  leave them. Do not stash them to get a clean run; branch from where you are,
  or ask. If you must stash, say so before and restore it after.
- **Keep changes focused.** One concern per PR. Adjacent problems you notice go
  in the PR description or a follow-up, not in the diff.

### Hard — claim integrity

- **Run device work on the box that has the device.** The sandbox has **no CUDA
  and no ROCm**; the Mac has no CUDA and no ROCm either. Route by hardware, not
  by convenience:

  | Need | Box |
  |---|---|
  | ROCm / gfx1151, x86 AVX-512 | Strix Halo, Ubuntu 24.04 under **WSL2** |
  | CUDA / sm_120 | **NR2 Pro** (RTX 5070 Ti, Linux) |
  | Metal / Apple CPU + GPU | **Mac** M1 Max |

  This is a claim-integrity rule, not a convenience one, because of *how* it
  fails. A device test on a host without that device does not usually error —
  it **skips**, or falls through to the `reference_cpu` lane, and the run comes
  back green. Reporting that as evidence asserts a hardware result that was
  never produced. If the hardware is not present, the honest output is "this
  host cannot evaluate these lanes," never a pass.
- **Prove native execution separately from the `reference_cpu` lane.** A
  hardware claim needs exact-device evidence from *that* device. Read Decision
  #19's standing lesson: a host that has the ISA cannot falsify a
  host-portability claim.
- **Record sibling-backend impact without claiming parity.** Noting that a
  change should also help ROCm is useful; asserting it works there without that
  hardware's proof is not. Evidence never transfers between architectures.
- **Never report green over a red lane.** If a gate fails, say so with the
  output, name whether it is pre-existing (prove it — stash and re-run), and fix
  the underlying lane where you can rather than routing around it.

### Defaults — depart with a recorded reason

- **Host-independent work** (docs, pure numpy, Python-only contracts, lit
  fixtures that need no device) may run on whichever box is convenient — but
  **say which host produced the result.** "The tests pass" means different
  things on different boxes, and a suite run where half the lanes cannot be
  evaluated is not the same evidence as a clean run.
- **Explore via the code graph first.** `graphify query` / `codegraph_search`
  before grepping, for anything wider than a single known symbol. Refresh with
  `graphify update .` after edits. Both indexes are local, machine-specific
  caches and must stay out of commits — `graphify-out/` via the root
  `.gitignore`, the CodeGraph database (117 MB) via `.codegraph/.gitignore`.
  `.codegraph/config.json` is tracked on purpose; nothing else under those two
  directories is.
- **Cross-registry changes.** Dtypes, ops, diagnostics, targets, and passes each
  span several registries and contracts. Adding one means updating every
  affected contract *and* its focused drift test — see Decision #24 for the
  primitive case and Decision #29 for why an unconsumed declaration is worse
  than a missing one.
- **Read the backend queues before backend work.** All four:
  `docs/audit/backend/{apple,nvidia,rocm,x86}/todo.md`. Scale how far you read
  to the blast radius of the change.

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

# MLIR lit tests (requires tessera-opt built). `python3 -m lit` does NOT work —
# lit is a package, not a runnable module. Use the console script, and put the
# matched LLVM bin on PATH or every fixture fails with `FileCheck: not found`.
export PATH=/opt/homebrew/llvm-23.1.0-rc1/bin:$PATH
lit tests/tessera-ir/ -v
lit tests/tessera-ir/phase8/ -q                 # one phase

# `lit tests/tessera-ir/` IS NOT THE WHOLE LIT GATE. There is a second suite —
# `src/compiler/codegen/Tessera_ROCM_Backend/test/rocm/`, run through
# `tessera-rocm-opt`, not `tessera-opt` — and CI's "rocm compiler" lane runs it.
# `check-tessera` does NOT include it (that target is IR + Python unit only, and
# its unit half shells out to system python3, which has no pytest). A backend
# change that passes tests/tessera-ir/ can still fail CI here.
ninja -C build check-tessera-ir                  # == lit tests/tessera-ir/
ninja -C build check-tessera-rocm                # the ROCm backend suite

# The remaining lit targets — check-{clifford,ebm,spectral,tessera-collective,
# tessera-performance} — report no tests unless their backends are configured
# ON (see the EBM/Clifford toggles in the build section).

bash scripts/validate.sh                         # CPU validation spine
```

**Build all targets before pushing, not one.** `ninja -C build tessera-opt`
links `MLIROptLib`'s broad dependency set and will hide a missing link library
that the standalone `tessera-rocm-opt` — which CI builds — needs. Use
`ninja -C build`.

Heavy SuperBench / benchmark-contract tests are marked `slow` and excluded by default.

---

## Local Toolchain

**Core compiler work is driven on the Strix Halo / Ubuntu box (decided
2026-08-02)** — it is faster, has more memory, and is the only machine in the
fleet with an executing GPU lane, so compile-time contract work and its hardware
gate live together. The Mac is retained for Apple-backend work, which cannot be
retargeted. Fleet routing per work item:
[`INTEGRATED_COMPILER_PLAN.md`](docs/audit/compiler/INTEGRATED_COMPILER_PLAN.md) §6a.

### Primary — Ubuntu 24.04 on Strix Halo (x86 + AMD ROCm)

`AMD RYZEN AI MAX+ 395 w/ Radeon 8060S`, 32 threads, 62 GB RAM, Ubuntu 24.04.4
under WSL2. `bash scripts/setup_ubuntu.sh` provisions matched LLVM/MLIR 23 from
**apt.llvm.org**, the base build deps, and a project-local `.venv` — then
`source .venv/bin/activate` and `export PYTHONPATH=python`. CMake LLVM lives at
`/usr/lib/llvm-23/lib/cmake/{llvm,mlir}`; put `/usr/lib/llvm-23/bin` on `PATH`
for `FileCheck` before running lit. TheRock ROCm **7.14** lives under
`/opt/rocm/core` (→ `/opt/rocm-7.2.4/core-7.14`) —
`-DTESSERA_ENABLE_HIP=ON -DTESSERA_BUILD_ROCM_BACKEND=ON
-DCMAKE_PREFIX_PATH=/opt/rocm/core`. The venv caps `numpy<2.2` (numpy ≥2.2 stubs
break the `python_version=3.10` mypy ratchet).

Two WSL specifics that bite: the GPU node is **`/dev/dxg`, not `/dev/kfd`** — do
not test for `/dev/kfd` to decide whether ROCm can execute here; and `rocminfo`
reports `gfx1151` natively. gfx1151 kernel execution is **live on this box**, not
Phase-H gated. **`torch` is not installed** — anything importing it must
`pytest.importorskip`.

CPU note: Zen 5 has **AVX-512 but no AMX** (AMX is Intel-only). Native x86
execution proof on this box means AVX-512; the AMX device lane
(`tests/device/x86/`, `scripts/run_x86_amx_release_gate.sh`) has no hardware in
the current fleet and stays capability-gated.

### Apple only — Mac M1 Max (Homebrew, off-venv)

Use for the Apple backend and Apple lit fixtures. Everything needed for build /
lint / typecheck / lit / unit-test is on Homebrew under `/opt/homebrew/bin/`:
`python3` (3.14.6), `ninja`, `cmake`, `pytest`, `mypy`, `ruff`, `black`, `isort`,
`flake8`, `lit`. Run the Python flow directly with `python3 -m …` — no venv.
`numpy`, `scipy`, `transformers`, `ml_dtypes` are under
`/opt/homebrew/lib/python3.14/site-packages/`. `torch` is not installed there
either.

**LLVM/MLIR 23 is a manual install at `/opt/homebrew/llvm-23.1.0-rc1/`**
(`llvm-config --version` → `23.1.0git`). There is **no `llvm@23` Homebrew
formula** on that machine — `brew install llvm@23` does not produce
`/opt/homebrew/opt/llvm@23/`, and Homebrew's own `llvm` keg is 22.1.8, which the
build rejects. Point CMake at
`/opt/homebrew/llvm-23.1.0-rc1/lib/cmake/{llvm,mlir}`, and put
`/opt/homebrew/llvm-23.1.0-rc1/bin` on `PATH` for `FileCheck` before running lit.

See `docs/GETTING_STARTED.md` for the full cross-platform matrix.

---

## C++ Build

```bash
# Canonical configure — PRIMARY box (Ubuntu/Strix Halo), x86 + ROCm + solver dialects.
# TESSERA_BUILD_X86_BACKEND is OFF by default and MUST be on here: x86 and ROCm
# share this host, and this is the only fleet configuration that runs both
# fixture families in one `lit` invocation (the 08-12 load-crash hid behind its
# absence — see Decision #19).
cmake -S . -B build -G Ninja \
  -DLLVM_DIR=/usr/lib/llvm-23/lib/cmake/llvm \
  -DMLIR_DIR=/usr/lib/llvm-23/lib/cmake/mlir \
  -DTESSERA_ENABLE_HIP=ON -DTESSERA_BUILD_ROCM_BACKEND=ON \
  -DTESSERA_BUILD_X86_BACKEND=ON \
  -DCMAKE_PREFIX_PATH=/opt/rocm/core
ninja -C build tessera-opt        # 32 threads; ~1-2 min cold

# EBM / Clifford (GA) dialects are OFF by default — enable them or their passes
# and lit fixtures silently do not build. Required for the W0 compiler work.
cmake -S . -B build -G Ninja -DTESSERA_BUILD_EBM_BACKEND=ON -DTESSERA_BUILD_CLIFFORD_BACKEND=ON

# Re-verify a C++ pass change end-to-end: rebuild → lit fixture + FileCheck → drift test
export PATH=/usr/lib/llvm-23/bin:$PATH
ninja -C build tessera-opt
./build/tools/tessera-opt/tessera-opt tests/tessera-ir/phase8/ga_ebm_graph_ops.mlir \
  --allow-unregistered-dialect | FileCheck tests/tessera-ir/phase8/ga_ebm_graph_ops.mlir

# Mac (Apple backend only) — LLVM/MLIR 23 lives under Homebrew there
cmake -S . -B build -G Ninja \
  -DLLVM_DIR=/opt/homebrew/llvm-23.1.0-rc1/lib/cmake/llvm \
  -DMLIR_DIR=/opt/homebrew/llvm-23.1.0-rc1/lib/cmake/mlir \
  -DTESSERA_CPU_ONLY=ON -DTESSERA_BUILD_APPLE_BACKEND=ON

# Other backend toggles (additive)
cmake .. -DTESSERA_ENABLE_CUDA=ON -DCUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda   # CUDA

# Benchmarks (stable JSON schema, Decision #12)
python3 benchmarks/run_all.py --backends x86 --output tessera_benchmarks.json
```

### Canonical lowering pipelines (in `tessera-opt`)

| Pipeline | Target |
|----------|--------|
| `tessera-lower-to-x86` | x86 AMX/AVX512 (Phase 2) |
| `tessera-lower-to-gpu` | NVIDIA SM_90+ WGMMA/TMA (Phase 3); `tessera-nvidia-pipeline-{sm90,sm100,sm120}` variants |
| `tessera-lower-to-rocm` | AMD ROCm MFMA |
| `tessera-lower-to-apple_cpu[-runtime]` | Apple CPU (Accelerate artifact / cblas_sgemm runtime) |
| `tessera-lower-to-apple_gpu[-runtime]` | Apple GPU (Metal artifact / MPS + MSL + MPSGraph runtime; longest-fusion-first ordering) |

---

## Key Reference Files

| What you need | Where |
|---------------|-------|
| **START HERE — status + open-work queue** | `docs/audit/MASTER_AUDIT.md` (+ theme audits; `docs/audit/README.md` for the map) |
| **Forward compiler direction (north star)** — three-tier/arbiter model + coordination (Decision #28) | `docs/audit/compiler/COMPILER_THEORY_OF_OPERATION.md` (read first) + `COMPILER_REFACTOR_PLAN.md` + reassessed `OPTIMIZING_COMPILER_PLAN.md` |
| **Compiler map + authority chain** — which doc wins when plans disagree (dashboards > COMPILER_AUDIT > INTEGRATED_COMPILER_PLAN > scoped plans > backend todos) | `docs/audit/compiler/README.md` (routes all scoped plans, incl. `GAME_THEORY_PLAN.md`, `compiler_enhancement.md` (CAKE), `FORGE_ASSESSMENT.md`, `W1_1_TYPING_DESIGN.md`; the Workstream C handoff is archived under `compiler/archive/`) |
| **Generated dashboards** (primary count/status evidence — never hand-edit) | `docs/audit/generated/` |
| Authoritative API naming | `docs/CANONICAL_API.md` |
| Canonical tensor attributes & dtypes | `docs/reference/tessera_tensor_attributes.md` |
| Backend architecture + kernel guides | `docs/backends/` (Apple, x86, ROCm, NVIDIA) |
| **RDNA ISA data archive** (does-this-op-exist-on-my-target truth before emitting) | `docs/reference/isa/rdna/` — structured, regenerable extraction of AMD's RDNA3 / RDNA3.5 / RDNA4 ISA guides + Micro Engine Scheduler. Per-version instruction DB (`<ver>/instructions.json`: opcodes, pseudocode), microcode encoding bit-fields (`encodings.json`), and a cross-version opcode matrix (`cross_version/instruction_matrix.{json,md}`). **gfx1151 = RDNA3.5: WMMA F16/BF16/IU8/IU4, NO FP8/BF8 WMMA (those + sparse SWMMAC are RDNA4-only).** JSON = machine truth, MD = mirror; regenerate via `tools/build_archive.py` (no network). MES scheduler write-up at `mes/SCHEDULER_OVERVIEW.md`. Sibling index: `docs/reference/isa/PRIMARY_SOURCES{,_INDEX}.md` — AMD primary-source assessment incl. CDNA 5 (assessed only, no extracted `cdna/` archive). |
| Graph IR ops / canonicalizations | `src/compiler/ir/TesseraOps.td`, `src/transforms/lib/CanonicalizeTesseraIR.cpp` |
| Schedule IR / FA-4 Tile IR ODS | `src/compiler/programming_model/ir/schedule/ScheduleMeshPipelineOps.td`, `src/compiler/tile_opt_fa4/include/tessera/Dialect/Attn/` |
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
