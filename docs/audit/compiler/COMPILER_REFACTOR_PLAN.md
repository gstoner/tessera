---
last_updated: 2026-07-08
audit_role: plan
plan_state: landing
---

# Tessera Compiler — Refactor + Enhancement Plan

> **Paired with** [`COMPILER_THEORY_OF_OPERATION.md`](COMPILER_THEORY_OF_OPERATION.md)
> (the conceptual model — read it first). This document is the *execution plan*:
> workstreams, sequencing, and the three-system coordination.
>
> **Builds on**, does not replace:
> [`OPTIMIZING_COMPILER_PLAN.md`](OPTIMIZING_COMPILER_PLAN.md) (F0–F6 middle-end
> synthesis — this plan generalizes its keystone across backends),
> [`EVALUATOR_PLAN.md`](EVALUATOR_PLAN.md) (scoring/promotion gate),
> [`STAGE_A_EMIT_PLAN.md`](STAGE_A_EMIT_PLAN.md) (cross-vendor emit ladder), and
> [`COMPILER_AUDIT.md`](COMPILER_AUDIT.md) (current state / Still Open).

---

## 1. Purpose

Two convergent goals, one plan:

1. **Refactor (downward dedup).** x86, ROCm, and Apple each re-implement the same
   MLIR lowering spine (bufferize→ptr→`func.call`, fusion-chain matching, dtype
   routing, shape verifiers). Extract it into a shared layer.
2. **Enhancement (backward lift).** Apple has proven a Tier-1 generic
   kernel-generation framework (synth → oracle → compile → cache → launch →
   measured autotune) that ROCm/CUDA/x86 lack. Generalize it — **with a per-arch
   codegen plugin** — so every backend gets compiler-generated kernels that are
   *optimal per architecture*, while hand-tuned kernels remain first-class.

The governing constraint (see Theory §1): **ROCm and CUDA are the lead
performance targets; the generic framework raises the floor and must never cap
their ceiling.** Apple is close behind. Every step is additive, opt-in, and
regression-gated on real silicon.

---

## 2. Scope guardrails (what this plan does NOT do)

- It does **not** replace the NVIDIA `wgmma`/`mma.sync` PTX path
  (`ptx_emit.py`), the ROCm MFMA/WMMA `Generate*Kernel` bodies, or the Apple
  authored-MSL/`.mtlpackage` kernels. Those are Tier 3 candidates the arbiter
  measures.
- It does **not** route lead-backend crown-jewel GEMM/attention through the
  generic synthesizer unless it *measures* competitive on that silicon.
- It does **not** touch the `@jit` frontend (done per `OPTIMIZING_COMPILER_PLAN`).
- It does **not** hand-edit generated dashboards (Decision #26).

---

## 2a. Status at a glance (updated 2026-07-07)

Landed state per §4 phase. Inline **landed** notes in §3 carry the detail; this
table is the single skim surface. `✅` done · `🟡` partial · `⬜` not started.

| Phase | Task | Mac (`[MAC]`) | AMD (`[AMD]`) | NV (`[NV]`) |
|---|---|:--:|:--:|:--:|
| **0** | E1 golden-IR harness + determinism roundtrip | ✅ | — | — |
| **0** | E2 real-hardware perf ratchet | ✅ (shape gate) | ✅ gfx1151 (matmul+flash, PR #284) | ✅ sm_120 (RTX 5070 Ti mma.sync ladder, live-gated) |
| **0** | E3 escape-hatch test | ✅ (landed in D1 `force` + PR #298 test) | ✅ gfx1151 | ✅ sm_120 |
| **1** | A1 shared `extractPtr`/`ensureExternalDecl` | ✅ | — | — |
| **1** | A2a–c shared fusion helpers (emit/chain-walk/epilogue trio) | ✅ | — | — |
| **1** | A2 single declarative `FusionPattern` · A3 verifiers | ⬜ | — | — |
| **1** | A4 shared cost-aware MMA selector (`mma_selector.py`) | ✅ | — | — |
| **2** | B1 split `fusion.py` | ✅ | — | — |
| **2** | B2a–c `KernelEmitter`/`Runner`/`SpecPolicy` | ✅ | — | — |
| **2** | B3 F4 oracle universal (backend-agnostic, C0) | ✅ | — | — |
| **2** | B4a `kernel_cache` synth→compile→cache loop | ✅ | — | — |
| **2** | B4 real AOT `compile_fn`s (per-arch, landed via C1/C2/C3) | ✅ registered host-free | ✅ Zen5 `clang` + gfx1151 `hipcc` | ✅ sm_120 `nvcc`+`ptxas` |
| **3** | C0 backend-plugin handoff + non-Apple F4 gate | ✅ (PR #285) | — | — |
| **3** | C1 x86 plugin (`emit/x86_llvm.py`: emitter + `cc` compile + ctypes runner) | ✅ emit host-free | ✅ execute on Zen 5 | — |
| **3** | Oracle accuracy budget (`KernelRunner.accuracy_atol`, D2 seed) | ✅ | — | — |
| **3** | C1b x86 AOCL-DLP Tier-3 candidate (opt-in) | 🟡 candidate wired host-free | 🟡 generic Tier-1 proven on Zen; AOCL lane library-gated | — |
| **4** | C2 NVIDIA generic synth → CUDA (`emit/nvidia_cuda.py`: emitter + `nvcc` + runner) | ✅ emit host-free | — | ✅ sm_120 FusedRegion (RTX 5070 Ti) |
| **4** | C2 tail — mma.sync PTX → shipped launch bridge (`tessera_nvidia_ptx_launch`) | ✅ host-free (g++/nvcc) | — | ✅ sm_120 m16n8k16 (RTX 5070 Ti) |
| **4** | C2 tail — broaden shapes/dtypes (general `mma.sync` bf16/f16 GEMM + NVFP4 emit) + arbiter Tier-2 wiring | ✅ | — | ✅ sm_120 |
| **4** | C2 tail — `wgmma` sm_90a emit | ⬜ | — | ⬜ (needs Hopper sm_90a) |
| **5** | C3 ROCm generic synth → HIP (`emit/rocm_hip.py`: emitter + `hipcc` + runner) | ✅ emit host-free | ✅ gfx1151 FusedRegion | — |
| **5** | C3 tail — drive WMMA/MFMA `Generate*` passes through the loop | ✅ WMMA host-free | ✅ gfx1151 fused WMMA F4-gated (MFMA=CDNA-gated) | — |
| **3.5** | ROCm shipped-kernel → F4 gate (flash-attn, f16 budget) + shared scalar body | ✅ | ✅ gfx1151 attn | — |
| **6** | D1 candidate registry + F4-gate + tier-priority arbiter + `force` (E3) | ✅ (`emit/candidate.py`) | ✅ gfx1151 enumerate+select | — |
| **6** | D2 measured autotune loop · D3 fallback log | ✅ (`emit/autotune.py` + arbiter log) | ✅ gfx1151 (fused_region measure+cache+corpus) | ✅ sm_120 (matmul measure+cache) |

**Gate reality (softens §4/§9.2):** "Phase 0 gates everything" holds only for the
*lead-execution* proofs. The Mac-side E1 gate is green and gfx1151 E2 is recorded,
so `[MAC]` + `[AMD]` work (A1, B1–B4a, C0, C1 authoring) has correctly proceeded.
**A live sm_120 box is now present** (RTX 5070 Ti, CUDA 13.3) — the `[NV]` proofs
are **no longer gated on a remote box**: the C2 generic-CUDA `FusedRegion` lane is
hardware-verified on it (`emit/nvidia_cuda.py`; F4-gated + arbiter-selected on-GPU,
`test_nvidia_plugin.py` live gates), and the **E2 sm_120 perf-ratchet baseline is
recorded** (`benchmarks/baselines/nvidia_sm120_hot_paths.json` — the shipped mma.sync
GEMM ladder; `test_nvidia_perf_ratchet.py` live-gates it). The **C2-tail launch bridge
is also landed + proven** (`tessera_nvidia_ptx_launch` driver-JITs + launches the
emitted `mma.sync` PTX on-GPU). What remains `[NV]`-open is **broadening** that emit
lane (shapes/dtypes, `wgmma` sm_90a, sm_100) and its Tier-2 arbiter wiring. Do not read
§4's hard-gate phrasing as blocking Mac/AMD authoring once E1
is green.

---

## 3. Workstreams

Each workstream lists tasks with an owning system tag `[MAC] [AMD] [NV]` (see §7
for the routing matrix). `[MAC]` = host-free, done on the dev Mac.

### Workstream A — Shared lowering layer (`tessera_common`) · downward dedup

Pure mechanical dedup; zero behavior change; golden-IR-gated.

- **A1 · shared `extractPtr`/`ensureExternalDecl`** `[MAC]` — **landed 2026-07-02.**
  Hoisted the byte-identical bufferize→ptr→`func.call` C-ABI helpers into
  `src/compiler/mlir/include/Tessera/Common/Lowering.h` (`tessera::common`);
  `TileToX86Pass.cpp` and Apple's `LoweringUtils.h` (~18 call sites) now `using`-
  forward to it — zero call-site changes, lit byte-identical (x86 3/3, Apple 4/4).
  **Scope correction:** the original plan listed `TileToROCM`, but ROCm does **not**
  use this pattern — `TileToROCM` rewrites `tile.mma`→`tessera_rocm.mfma`/`wmma`
  ops directly (op-rewriting, not a runtime C-ABI call), so it is out of scope for
  this helper. A `lowerToRuntimeCall(op, ABI, symbol)` façade over the full
  matmul-lowering boilerplate is the follow-on step.
- **A2 · Declarative fusion matcher** `[MAC]` —
  `FusionPattern{opChain, rankConstraints, dtypes, dimCaps, symbol}` + one generic
  `RewritePattern`, replacing the 12+ `*FusionToAppleGPU.cpp` per-chain passes and
  the ROCm dispatch-match shell. **ROCm `Generate*Kernel` bodies stay** — only the
  match/dispatch shell is shared.
  **Shared-helper half landed 2026-07-04 (A2a–c, PRs #275–#277):** the byte-identical
  logic across all 7 `*FusionToAppleGPU.cpp` passes is now hoisted into shared
  headers — `emitFusionCall` (`Tessera/Common/Lowering.h`), the chain-walk +
  Decision-#21 descriptor hook (`FusionChainUtils.h`: `fusionDescriptorDriven`,
  `walkChainProducer`), and the data-driven epilogue body collapsing the
  Softmax/Gelu/RMSNorm trio (`EpilogueFusion.h`: `lowerMatmulEpilogueFusion`), all
  lit byte-identical. **Still open:** the *single* declarative `FusionPattern` table +
  one generic `RewritePattern` that would retire the 7 separate pass files entirely
  (the passes still exist, they just share bodies now).
- **A3 · Declarative shape/constraint verifiers** `[MAC]` — ⬜ **open.** Replace the
  hand-written `verify*()` (7 today) in the 1219-line `TileToApple.cpp` with a
  declarative spec.
- **A4 · Promote ROCm's `MmaDescriptor` + `M×N//lanes` footprint model** `[MAC]`
  to the shared MMA selector, parameterized by per-arch lane count + shape table.
  The one place a *lead* abstraction lifts upward: Apple/x86/NVIDIA gain a
  cost-aware MMA selector they lack. **Landed 2026-07-08** —
  `compiler/mma_selector.py`: an arch-neutral `MmaIsa` record (`lane_count` +
  shape table + dtype→K) + the promoted footprint (`accumulator_regs_per_lane` =
  `M*N // lane_count`, delegating to `rocm_target.mfma_accumulator_regs` so the
  arithmetic stays single-sourced) + `rank_shapes_by_footprint` cost model +
  `select_mma → MmaSelection`. **Lead-safe by construction:** the ROCm ISA is
  *built from* the live `rocm_target`/`rocm_mma` tables (feature gates — no FP8
  WMMA on gfx1151, no fp32 WMMA on RDNA — inherited via the reference selector,
  never re-encoded), and `test_mma_selector.py` gates that `rank_shapes_by_footprint`
  on a ROCm ISA is byte-identical to ROCm's own `rank_mfma_shapes_by_footprint`.
  The lift: grounded ISA records for NVIDIA (`mma.sync` m16n8k{8,16,32,64} warp-32),
  Apple (`simdgroup_matrix<8,8>` simdgroup-32), x86 AMX (tile-register — honestly
  `cooperative=False`/`lane_count=None`, per-lane footprint N/A). No emit path
  changed (hardware-free selector, Decision #19). **Follow-on (own PR — touches
  drift-gated dashboards):** wire `get_isa`/`select_mma` as `backend_manifest`'s
  cross-target MMA-metadata source (today ROCm-only via `_rocm_mma_descriptor_for`)
  and as the D1 arbiter's `cost_model`.

**Lead safety:** A1–A3 are Apple/x86-facing; ROCm adopts only the match/verifier
shell with byte-identical emit (golden-IR gated). No NVIDIA emit change.

### Workstream B — Generalize the synthesizer · the keystone

Turn Apple's MSL-welded synthesizer into a target-agnostic framework. Generalizes
`OPTIMIZING_COMPILER_PLAN` F2 (MSL synthesis) → F6 (one middle-end, many
backends).

- **B1 · Split `fusion.py`** `[MAC]` — **landed 2026-07-04.** arch-agnostic half
  (`FusedRegion`, `EpilogueOp`/`ReductionOp` semantics, `discover_*`,
  `should_fuse_*`/`*_cost`, `verify_synthesized_*`) → `compiler/fusion_core.py`
  (65 symbols); MSL emit + runtime dispatch + measured autotune loop
  (`synthesize_*_msl`, `run_*`, `_synth_*_symbol`, corpus) →
  `compiler/emit/apple_msl.py` (74 symbols); `compiler/fusion.py` is now a thin
  re-export facade (`X as X` idiom) so all ~20 importers are untouched. Pure
  relocation, no behavior change (full unit suite delta = 0 vs main; the ~181
  ROCm `--generate-*-kernel` failures are a pre-existing `tessera-opt`
  pass-registration carve-out, unrelated). **The one seam:** the F4 oracles reach
  the Apple runner via a lazy `_apple_msl()` bridge (marked for B2 to replace with
  the injected `KernelEmitter` runner) — keeps `core → emit` acyclic with no
  behavior change.
- **B2 · `KernelEmitter` plugin protocol** `[MAC]` — `EpilogueOp.msl` field
  becomes `EpilogueOp.emit(target)`; `KernelEmitter.emit(region, target, spec) →
  KernelSource`. Apple MSL is the reference impl (relocated, not rewritten).
  Sequenced as three increments: **B2a** (protocol + `emit(target)` vocab +
  `AppleMSLEmitter` wrapper) — **landed 2026-07-04:** `emit/kernel_emitter.py`
  defines `SpecPolicy(static|bucket|dynamic)`, `KernelSource`, the `KernelEmitter`
  ABC + registry (`emit_kernel(region, target, spec)`); `EpilogueOp`/`ReductionOp`
  `.msl`→`.emit(target)` (unknown target raises, Decision #21); `AppleMSLEmitter`
  wraps the `synthesize_*_msl` bodies byte-identically. **B2b** — **landed 2026-07-04:** `KernelRunner`
  ABC + runner registry (`register_runner`/`get_runner`/`active_runner`) in
  `emit/kernel_emitter.py`; `AppleMSLRunner` delegates to the `run_*` functions
  and self-registers; `fusion_core`'s 4 oracle bridge wrappers now dispatch to the
  registered active runner (lazy-register fallback preserves direct-import safety)
  instead of a hard `import apple_msl`. **B2c** — **landed 2026-07-04:**
  `FusedRegion.dim_names` carries the Graph-IR symbolic dims; `bucket_key(dims,
  spec, dim_names)` computes the shape-specialization key per policy (STATIC =
  exact, BUCKET = next-pow-2 per dim matching `_shape_bucket`, DYNAMIC = symbolic
  identity); `emit`/`emit_kernel` thread a concrete `dims` and record it in
  `KernelSource.shape_key` (metadata, not codegen — source is dims-invariant). The
  `dynamic` emitter stays stubbed behind the `EmitError` added in B2a's review
  round; the full guarded runtime-dim emitter is Workstream W2.
  **Symbolic-dim-aware from day one (dynamic-shapes decision, 2026-07-02):** the
  `region` carries symbolic dims (from Graph-IR `dim_names`), and `spec` is the
  **specialization policy** `static | bucket | dynamic`. First impls emit
  `bucket` (compile per shape-bucket — seq-len / batch / KV-len — dispatched by
  runtime shape); the interface is designed so a later `dynamic` (runtime-arg +
  guards) emitter drops in **without an API break**. This is the "pull it
  forward" — the *interface* is dynamic-ready now; the *implementation* starts at
  bucket. See Theory §8 W2.
- **B3 · F4 oracle as universal correctness gate** `[MAC]` — **landed
  2026-07-04:** the four `verify_synthesized_*` oracles gain an explicit
  `runner: KernelRunner | None = None` — the same numpy-reference oracle now gates
  *any* backend's synthesized kernel (`verify_synthesized_region(region,
  runner=<backend>)`), not just Apple; `None` resolves the registered active
  runner. The now-redundant B2b core bridge wrappers (`run_fused_region` etc. in
  `fusion_core`) are removed — the oracles dispatch straight through
  `(runner or _runner()).run_*`. This is what makes compiled kernels *safe to
  prefer*.
- **B4 · Generic synth→compile→cache→launch loop** `[MAC]` for the loop; compile
  fn is per-arch — extract from `apple_gpu_runtime.mm` (`newLibraryWithSource` +
  sha256 `cache_key`); plugin supplies `metallib`/`ptxas`/`hipcc`/`clang`.
  **B4a landed 2026-07-04:** `compiler/emit/kernel_cache.py` — the arch-neutral
  driver: `cache_key(source, dtype, target)` (sha256 over the `source + '\x1f' +
  entry` join the runtime already uses, extended with `spec`/`shape_key`/`dtype`/
  `target` so bucket/dtype variants stay distinct); a `register_compiler(target,
  fn)`/`get_compiler` plugin seam + `CompileError`; a content-addressed
  `KernelCache` (hit/miss stats); and `build(region, target, spec, dims, dtype)` =
  emit → key → cache-or-compile. Apple registers a **deferred** compiler
  (compile-on-launch — Metal compiles inside `run_*`, cached in the runtime), so
  the loop dedups and keys without duplicating work. **Launch** stays the B3
  `KernelRunner`. **The real ahead-of-time `compile_fn`s landed in Workstream C**
  (2026-07): `emit/x86_llvm.py` (`clang`/`cc`), `emit/nvidia_cuda.py` (`nvcc`, +
  `ptxas` for the mma.sync PTX bridge), and `emit/rocm_hip.py` (`hipcc`) each
  `register_compiler` their toolchain into the B4a loop.

**Lead safety:** B targets the *fusable-DAG middle ground* (epilogues, pointwise
chains, small attention). Crown-jewel GEMM stays Tier 2/3.

### Workstream C — Per-arch codegen plugin interface + the missing lead lanes

> **Picking this up on the Strix Halo / NR2 Pro box?** Start at
> [`WORKSTREAM_C_HANDOFF.md`](WORKSTREAM_C_HANDOFF.md) — the build recipe for the
> three plugin seams (emitter / compile_fn / runner) the merged Workstream B
> framework calls into, with a copy-paste skeleton, the F4-verification recipe,
> and the per-backend task cards (C1 x86 · C2 NVIDIA · C3 ROCm).

- **C1 · Per-arch plugin = three registered seams (NOT one `TargetPlugin`
  struct)** `[MAC]` author → `[AMD]` execute on Zen 5. **Interface reconciled
  2026-07-06:** the plan originally sketched a single `TargetPlugin` object with
  seven fields; what B2/B4a + the C0 handoff actually shipped is **three separate
  registries** a backend self-registers into on import (mirroring
  `emit/apple_msl.py`) — this is the real, tested seam, and there is no bundled
  struct. A backend adds one module `emit/<target>.py` implementing:
  1. **`KernelEmitter`** (`register_emitter`) — `emit(region, spec, dtype, dims)
     → KernelSource`. This is the original `emit_kernel` field.
  2. **`compile_fn`** (`register_compiler`) — `source → artifact` (x86: `clang
     -O3 -mavx512f -mavx512bf16 -shared` → `.so`). The original `compile_fn` field.
  3. **`KernelRunner`** (`register_runner`, `default=False`) — `run_*(region,
     *inputs) → (out, execution_tag)`; a real tag (`"x86_native"`) gets F4-gated,
     a `REFERENCE_EXECUTIONS` tag declines.

  The original seven fields map onto shipped reality as: **`emit_kernel`** =
  `KernelEmitter`; **`compile_fn`** = `register_compiler`; **`spec_policy`** =
  the `SpecPolicy(static|bucket|dynamic)` a `KernelEmitter` accepts +
  `bucket_key`'s strategy (this is what replaces the hard `"requires static
  shapes"` gate in `TileToX86Pass` / `MatmulToAppleCPU` — a policy, not a
  per-backend hardcode). The remaining four were speculative and are **not**
  first-class plugin fields: **`shape_table`** + **`cost_model`** live in the
  shared MMA selector (A4 `MmaDescriptor`) and the arbiter (D1), keyed per
  `(op, shape-bucket, dtype, target)` — not on the emitter; **`intrinsic_set`**
  is a `compile_fn` build-flag detail (x86 = `-mavx512f -mavx512bf16`, **never
  `-mavx*` AMX** on this AVX-512-only fleet), not a declared field;
  **`async_model`** is a no-op for the synchronous CPU/x86 lane and is deferred
  to the GPU emit lanes (C2/C3) that actually need an async-token model — do not
  add it to the x86 plugin. **DoD splits by system:** `emit` is pure/host-free
  (`[MAC]`: mypy + ruff + emitter unit tests); the clang compile + `ctypes`
  launch + F4 execute-compare require the Zen 5 box (`[AMD]`).
- **C1b · x86 Tier-3 candidate: AOCL-DLP** `[AMD]`, opt-in, **separated from
  C1** — register **AOCL-DLP** ([amd/aocl-dlp](https://github.com/amd/aocl-dlp))
  as a hand-tuned candidate the D1 arbiter measures, NOT part of the core plugin.
  AMD's BLIS-family DL primitives (low-precision GEMM/batch GEMM incl. INT4/FP16,
  pre/post-ops matching `fused_epilogue`, symmetric quant, OpenMP); AVX512-based
  (fits the Zen 5 box, no AMX), fills the x86 backend's OpenMP-threading +
  INT4/FP16 gaps, opt-in behind a build flag (a BLAS-family library like
  Accelerate — Decision #23-clean, behind the hardware-free Target IR). The
  arbiter selects it only where it measures faster than the generic kernels on
  Zen; **check its license before it becomes a shipped/linked lane.**
  **Landed 2026-07-07 (arbiter-facing seam, host-free):** `emit/x86_aocl_dlp.py`
  registers `X86AoclDlpCandidate` (Tier-3) alongside the new `X86GenericCCandidate`
  (Tier-1) under `target="x86"`. `available()` probes `$TESSERA_AOCL_DLP_LIB` +
  `$TESSERA_AOCL_DLP_SGEMM`; absent here, so it is arbiter-visible but never
  mis-selects (arbitration falls to the generic C lane, proven on Zen 5). **Still
  open (needs a licensed aocl-dlp install):** bind the concrete GEMM post-op ctypes
  ABI against real headers (deliberately *not* guessed — `_aocl_dlp_gemm` declines
  until wired), run the license review, then F4-gate + measure on Zen.
- **C2 · NVIDIA emit lanes** `[MAC]` authoring → `[NV]` proof. Two lanes, mirroring
  the ROCm split:
  **Generic CUDA lane — LANDED + hardware-proven 2026-07-07.** `emit/nvidia_cuda.py`
  is a **full three-seam plugin** (parallel to `rocm_hip`/`x86_llvm`):
  `NvidiaCudaEmitter` turns a `FusedRegion` into CUDA source (a `__global__`
  one-thread-per-row kernel + a host-pointer C-ABI wrapper doing H2D/launch/D2H),
  reusing the *same* scalar body as x86/ROCm (`_fused_scalar_body.row_compute_body`)
  so all three stay locked to the one `fusion_core` reference; `_nvidia_cuda_compile_fn`
  compiles it with `nvcc -arch=sm_120a -O3 --shared` → `.so`; `NvidiaCudaRunner`
  dlopens + launches (`"nvidia_cuda"`), else declines to the reference. Registered as
  the Tier-1 `NvidiaGenericCudaCandidate`. **Live-proven on sm_120** (RTX 5070 Ti,
  CUDA 13.3): the generic `FusedRegion` family (relu/bias-gelu/silu/softmax/rmsnorm/
  layer_norm/prologue) compiles, runs on-GPU, matches numpy (f32), passes the same
  universal F4 oracle, and the D1 arbiter selects it (`test_nvidia_plugin.py` live
  gates). Same NULL-buffer guard as x86/ROCm.
  **C2 tail — launch bridge LANDED + hardware-proven 2026-07-07.** The long pole the
  plan flagged (§9.1(2)) is built: `runtime/cuda/tessera_nvidia_ptx_launch.{cpp,h}` is
  the shipped NVIDIA counterpart to Apple's `apple_gpu_runtime.mm` — it driver-JITs
  Tessera's *emitted* `ptx_emit.py` PTX (`cuModuleLoadDataEx`, **cached by kernel
  name**) and launches it (`cuLaunchKernel`), exposing both the direct
  `tessera_nvidia_ptx_register`/`_invoke` C-ABI (dlopen-able standalone; the seam is a
  **weak** `tsrRegisterGpuLauncher` ref so the `.so` loads without the core runtime)
  and the `tsrGpuLauncherFn` registered via `tessera_nvidia_register_ptx_launcher`.
  CMake target added alongside the shipped GEMM lib. This **promoted the throwaway
  inline launcher** in `test_conformance_execute_compare_nvidia.py` into shipped code;
  that test now compiles the shipped source and drives it through the real
  `tsrLaunchKernel` seam. **Live-proven on sm_120** (RTX 5070 Ti): the emitted
  `mma.sync m16n8k16` bf16 GEMM runs on-GPU and matches the numpy oracle (max err
  ~2.4e-7), the module cache reuses, and an unregistered kernel declines honestly.
  **Emit breadth landed 2026-07-07:** `emit_mma_sync_gemm_ptx(dtype=bf16|f16)` now
  emits a general **aligned-M/N/K** kernel (K-loop + grid-tiled over 16x8 tiles,
  runtime M/N/K params); the bridge grows a general-GEMM ABI (`invokeMmaGemm16`).
  ptxas-assembled + execute-compared on sm_120 through the `tsrLaunchKernel` seam
  (32x16x32 multi-tile/multi-K) and a numpy sweep (~1e-7 rel). **NVFP4 block-scale**
  (`emit_nvfp4_block_scale_mma_ptx`, `mma.sync…m16n8k64…kind::mxf4nvf4.block_scale`)
  is productized to **emit + ptxas-assemble** on sm_120a (unit-scale data path proven
  in `spikes/`); on-device execution + non-unit-scale numerics stay gated on the
  PTX-ISA scale-distribution spec.
  **Still open:** **ragged (unaligned) M/N/K** (need boundary predication), **tf32**
  (m16n8k8) / **fp8** (m16n8k32) fragment layouts; the **`wgmma` sm_90a** path is a
  documented *instruction-encoding skeleton* (`emit_wgmma_matmul_ptx` — its own header
  says "NOT a complete assemblable kernel: needs smem matrix descriptors + TMA/cp.async
  + the full accumulator operand list"), so completing it is a real Hopper WGMMA-kernel
  build (assemble-only here; execution needs Hopper) — **not** a bug fix; and sm_100
  tcgen05.
  **B1 landed 2026-07-07 — the emit lane is now a first-class arbiter candidate.**
  A bare-matmul op-kind (`candidate.OP_MATMUL` + `fusion_core.MatmulRegion` +
  `verify_synthesized_matmul`) unblocks the D1 arbiter for plain GEMM. Two NVIDIA
  matmul candidates register under `(nvidia, matmul)`: the **shipped** mma.sync GEMM
  (`NvidiaMmaGemmShippedCandidate`, Tier-3 via `runtime._nvidia_mma_gemm_2d`) and the
  **compiler-emitted** `ptx_emit` GEMM (`NvidiaMmaGemmEmittedCandidate`, Tier-2 via the
  launch bridge, `runtime._nvidia_ptx_gemm_2d`). Both F4-gated by the universal oracle;
  tier-priority picks the shipped lane by default (lead-safe, Decision #28), the E3
  `force` hatch selects the emitted lane. Live-proven on sm_120 (bf16/f16 ×
  16x8x16/32x16x32/64x64x64). So NVIDIA now has its Tier-3 *hand-tuned* GEMM candidate
  (the pure-GEMM shipped kernel, previously only reachable via the jit `nvidia_mma`
  executor) alongside the Tier-2 emitted lane — D2's measured loop is the follow-on
  that lets Tier-2 win where faster.
- **C3 · ROCm generic synth → HIP** — **generic lane LANDED 2026-07-06**, `[MAC]`
  author → `[AMD]` proof. `emit/rocm_hip.py` is now a **full three-seam plugin**
  (parallel to x86): `RocmHipEmitter` turns a `FusedRegion` into HIP source (a
  one-thread-per-row `__global__` kernel + a host-pointer C-ABI wrapper doing
  H2D/launch/D2H), `_rocm_hip_compile_fn` compiles it with `hipcc
  --offload-arch=<gfx>` → `.so`, and `RocmHipRunner.run_fused_region` dlopens +
  launches on gfx1151 (`"rocm_hip"`), F4-gated. The per-row scalar body is
  **shared with the x86 C lane** (`emit/_fused_scalar_body.py`) so both stay
  locked to the one `fusion_core` reference. Same NULL-buffer guard as x86.
  **C3 tail (WMMA landed 2026-07-07):** the gfx1151 WMMA `generate-wmma-gemm-kernel`
  `Generate*` pass is now a **Tier-3 D1 candidate** (`RocmWmmaGemmCandidate` in
  `emit/rocm_hip.py`) driven through the shared loop: `runtime._rocm_wmma_fused_2d`
  runs the compiler-generated fused GEMM+bias+{relu,gelu,silu} on matrix cores
  (f16 storage / f32 accum), gated by the *same* universal F4 oracle as the generic
  lane and registered alongside it under `target="rocm"`. The generic scalar HIP
  kernel stays the Tier-1 middle-ground candidate; the flash-attn lane is the Tier-3
  attention candidate. Default (tier-priority) arbitration picks WMMA where it
  applies and falls to the generic lane for a reduction it cannot fuse; the E3
  escape hatch forces either. Live-proven on gfx1151 (`test_rocm_plugin.py` §4).
  **Still open:** CDNA **MFMA** `Generate*` passes as candidates (analytical-only
  until CDNA silicon — gfx1151 is RDNA3.5/WMMA, no MFMA); D2 measured selection
  replaces tier-priority with real on-device latency per shape-bucket.
- **C3-precursor (landed 2026-07-06): ROCm runner → F4 gate + oracle accuracy
  budget** `[MAC]` author → `[AMD]` proof. Ahead of the full emit pipeline, the
  *shipped* gfx1151 kernels are now wired into the universal F4 oracle:
  `emit/rocm_hip.py` registers a **runner-only** plugin (no emitter/`compile_fn`
  — ROCm's kernels are shipped, not synthesized) whose `run_fused_attention`
  runs the compiled FA-2 lane on-device (tag `"rocm_hip"`) and is gated against
  the numpy reference; other region kinds decline to the reference (honest — the
  fused-epilogue GPU kernel is C3 proper). Because those kernels are **f16
  storage**, this required the **accuracy-budget** seed (plan D2): a
  `KernelRunner.accuracy_atol` the oracle widens its tolerance to, so f16
  rounding (~2.5e-3 on the probes) is not misread as a miscompile while an O(1)
  bug still is. Apple/x86 (f32/exact) declare no budget → unchanged. This is the
  cross-backend differential-equivalence superpower (Theory §7.5) applied to the
  lead's shipped kernels, and the first concrete slice of the accuracy-budgeted
  arbiter. Proven live: `tests/unit/test_rocm_plugin.py` gates gfx1151 attention
  across scale/causal on-device.

### Workstream D — Candidate arbitration + measured autotune

- **D1 · Candidate registry** `[MAC]` — **core landed 2026-07-07**
  (`emit/candidate.py`). A `Candidate` ABC (`tier`/`target`/`op`/`available()`/
  `applies_to()`/`run()`) + a registry keyed per `(target, op)` enumerates
  `{synthesized (Tier 1), emitted (Tier 2), hand_tuned (Tier 3)}`. `arbitrate()`
  filters by applicability + availability, F4-gates each candidate through the
  *same* universal oracle (a `KernelRunner` adapter reuses
  `verify_synthesized_*`), then selects by **tier priority** (crown-jewel first —
  lead-safe, Decision #28) with a pluggable `measure` hook (the D2 seam) and a
  `force` escape hatch (E3). Wired for ROCm (generic HIP / WMMA / flash) and x86
  (generic C / AOCL-DLP). **Shape-bucket key landed via D2 (2026-07-08):**
  `measured_arbitrate` threads `bucket_key(dims, BUCKET)` + dtype + device tag into
  the cache key, so nearby shapes share a measured verdict while distinct buckets
  re-measure (D1's `arbitrate` stays a pure per-region selection — the cache is
  D2's). **Still open:** generalizing Apple's `select_variant` + `best_record` into it.
- **D2 · Measured autotune loop** — **core landed 2026-07-07** (`emit/autotune.py`).
  `measured_arbitrate()` layers on the D1 arbiter's `measure` seam: it F4-gates the
  candidates, times each survivor on-device (`measure_latency`, median of N after
  warmup), and caches the fastest in a `MeasureCache` keyed by `(device, target, op,
  shape-bucket, dtype)` — **measure-at-first-miss** (a re-query hits the cache, no
  re-timing). Lead-safety holds: only in-budget F4-passing candidates are timed, so a
  faster-but-wrong kernel can't win. **Live on sm_120** (RTX 5070 Ti): times the
  shipped vs emitted GEMM lanes and caches the winner per bucket
  (`test_nvidia_plugin.py`); `_nvidia_device_name()` supplies the `sm_<cc>` device
  tag. **`[AMD]` gfx1151 wiring + the fleet corpus landed 2026-07-08:**
  `_device_id` is a probe table (`_nvidia_device_name` / new `_rocm_device_name` →
  `gfx1151`, gated behind the live-execution probe so a GPU-less host never keys a
  gfx verdict); `measured_arbitrate` times the ROCm `fused_region` lanes (generic
  HIP vs WMMA) on-device and caches per shape-bucket. **`MeasureCache.to_dict` is
  now the committed fleet corpus** — JSON-safe, self-describing keys, `save_corpus`/
  `load_corpus` + once-only lazy warm-start of the default cache
  (`benchmarks/baselines/autotune_corpus.json`, keyed by device so a gfx1151 row is
  inert on a CDNA/NVIDIA box). Proven live on gfx1151 (`test_rocm_measured_autotune.py`):
  the measured loop overrides tier-priority *both ways* — the Tier-1 generic lane
  wins the 64³ bucket (WMMA launch overhead dominates) while WMMA wins from 256³ up
  (~19× at 1024³). **Still open:** CDNA/sm_90/sm_100 analytical-roofline +
  `MmaDescriptor` fallback until silicon.
- **D3 · Fallback log everywhere** — **landed 2026-07-07.** The arbiter records every
  dispatch as `(target, op, selected, tag)` (`candidate._note_arbiter_dispatch`, wired
  into `run_arbitrated` + `run_measured_arbitrated`); `arbiter_dispatch_histogram()`
  answers **"did the compiled path win, silently degrade, or was there no candidate?"**
  per `(target, op)` — a selection that ran but returned a reference tag is the silent
  degrade (the arbiter-layer analog of `runtime.dispatch_fallback_log`). Proven both
  host-free and live (the emitted lane forced on a ragged shape it can't run logs a
  `degraded`).

### Workstream E — Regression guardrails (continuous, not last)

Operationalizes Theory rule #3. **Built in Phase 0, before any refactor lands.**

- **E1 · Host-free golden-IR diff** `[MAC]` — snapshot ROCm/NVIDIA/Apple/x86
  emitted Target IR for a fixture set; any A/B/C change that perturbs a lead's IR
  fails on the Mac. Extend the existing `apple_runtime_ops.inc` drift-gate
  pattern. **Two determinism prerequisites (§9.1(3), verified):** IR is
  deterministic today, but before trusting a byte-exact diff — **(a)** add a
  canonical key-sort in `target_ir.py::_format_attr_dict` so attr order is
  construction-path-independent, and **(b)** add a determinism roundtrip test
  (`tessera-opt` twice → byte-identical). Keep fixtures on ordered `CHECK`, not
  `CHECK-DAG`.
- **E2 · Real-hardware perf ratchet** — `[AMD]` gfx1151 + `[NV]` sm_120 hot-path
  latency floors recorded as committed JSON (`rocm_*_hot_paths.json`,
  `nvidia_*_hot_paths.json`), mirroring `apple_gpu_hot_paths.json` +
  `perf_gate --ratchet`. No merge regresses a lead.
- **E3 · Escape-hatch test** `[MAC]` per backend — **landed 2026-07-07:** the
  arbiter's `force=<name>` path (`emit/candidate.py`) forces a named candidate and
  raises `ArbiterError` if it is absent; PR #298 added the escape-hatch test that
  asserts a hand-tuned Tier-3 kernel *can* be forced and *does* win. Proves Tier 3
  is never orphaned.

---

## 4. Sequencing

| Phase | Work | Proof system | Why this order |
|---|---|---|---|
| **0** | E1 golden-IR + E2 ratchet baselines | `[MAC]` + `[AMD]` + `[NV]` | Tripwire before touching the leads |
| **1** | A1–A4 shared lowering | `[MAC]` (x86→Apple→ROCm) | Mechanical, zero-behavior, IR-gated |
| **2** | B1–B3 split synthesizer + universal oracle | `[MAC]` (Apple relocate) | De-risk keystone where it already works |
| **3** | B4 + C1 generic loop + plugin interface | `[MAC]` → `[AMD]` (x86 on Zen 5) | x86 clang plugin = cheapest 2nd impl |
| **4** | C2 NVIDIA emit pipeline | `[MAC]` author → `[NV]` prove | New lead lane vs shipped `.so` |
| **5** | C3 ROCm emit pipeline | `[MAC]` author → `[AMD]` prove | Symmetric; reuses async-token model |
| **6** | D1–D3 arbitration + measured autotune | `[AMD]` + `[NV]` live | Ties it together; compiled-vs-handtuned becomes measured |

Phases 1 and 2 are parallel-safe (disjoint files). Phase 0 gates everything on
the leads.

---

## 5. Definition of done (per phase)

A phase is done when: (a) its `[MAC]` host-free gate is green (lit + unit + mypy +
golden-IR), (b) any `[AMD]`/`[NV]` proof is a **committed recorded artifact**
(execute-compare fixture + perf ratchet), and (c) the lead-backend ratchets show
**neutral-or-better** measured latency. No phase promotes a conformance/manifest
cell without the Evaluator's fixture-backed proof (`EVALUATOR_PLAN`, Decision #24
truth in `primitive_coverage.py`).

---

## 6. Risks + mitigations

| Risk | Mitigation |
|---|---|
| Shared abstraction cripples a lead | Theory rule #1 + E1/E2: lead opts out per op; IR/perf gated |
| Synthesizer split regresses Apple | B1–B3 pure relocation, oracle-gated, no new codegen; existing differential harness proves it |
| NVIDIA/ROCm emit pipelines are the big new build | Additive lanes; shipped-symbol path stays until compiled lane ≥ parity (arbiter decides) |
| Silicon boxes become a bottleneck | Mac-first routing (§7): only execute-compare + perf ratchet require a box |
| AMX fast-path unvalidatable | Known fleet gap; AVX-512 validates on Zen 5; AMX stays hardware-gated + flagged |

---

## 7. Three-system coordination

The fleet, exclusive capabilities, and the host-free-first invariant are
specified in **Theory §6**. This section is the operational routing.

### 7.1 System roles (summary)

- **Mac** — authoring hub + host-free CI (lit, mypy, unit, IR, **golden-IR
  generation**) + Apple execute. Default home for all `[MAC]` tasks.
- **Strix Halo** `[AMD]` — ROCm gfx1151 execute-and-compare, x86 AVX-512 native
  execute (Zen 5), AMDGCN codegen, measured autotune (RDNA).
- **NR2 Pro** `[NV]` — `ptxas` assemble, CUDA sm_120 execute-and-compare, sm_120
  measured autotune. (Its 265F CPU is a CUDA *host*, not an x86-backend target —
  never build x86 with `-mavx512*` here.)

### 7.2 Task routing matrix

| Task class | Mac | Strix Halo | NR2 Pro |
|---|:--:|:--:|:--:|
| Author IR / passes / plugin code | ✅ primary | — | — |
| Host-free CI (lit, mypy, unit, golden-IR) | ✅ primary | ✅ mirror | ✅ mirror |
| Apple MSL compile + execute | ✅ only | — | — |
| x86 AVX-512 native execute | — | ✅ only (Zen 5) | ✗ (no AVX-512) |
| x86 AMX execute | ✗ | ✗ (no AMX) | ✗ | *(hardware gap)* |
| ROCm gfx1151 execute + AMDGCN | — | ✅ only | — |
| CUDA sm_120 execute + `ptxas` | — | — | ✅ only |
| Measured autotune (per target) | Apple only | RDNA only | sm_120 only |

### 7.3 The sync loop

```
author on MAC ─▶ push branch ─▶ per-system CI lane runs:
   MAC:  host-free gate (lit/mypy/unit/golden-IR)   ── blocks merge
   AMD:  ROCm + AVX-512 execute-compare + ratchet    ── records artifact
   NV:   ptxas + CUDA execute-compare + ratchet       ── records artifact
      └─▶ commit recorded proofs back to branch ─▶ merge when all green
```

**Contract:** silicon proofs (`execute_compare_fixture`, `*_hot_paths.json`) are
committed artifacts. Once committed, the Mac's host-free gate asserts their
*shape* (fixture exists, ratchet not regressed) between silicon runs — so a
Mac-authored change stays honest about the leads without a GPU present.

### 7.4 Per-system setup pins (from `docs/GETTING_STARTED.md`)

- **Mac:** Homebrew LLVM/MLIR **22.1.6** at `/opt/homebrew/opt/llvm`; off-venv
  `python3` 3.14.5.
- **Strix Halo:** Ubuntu 24.04 + `scripts/setup_ubuntu.sh` (LLVM/MLIR 22 from
  apt.llvm.org — ROCm's bundled LLVM has no MLIR); ROCm **7.2.4** at `/opt/rocm`;
  `-DTESSERA_ENABLE_HIP=ON -DTESSERA_BUILD_ROCM_BACKEND=ON`; `.venv` numpy<2.2.
  gfx1151 = RDNA 3.5, WMMA 16×16×16, **no FP8 WMMA**.
- **NR2 Pro:** CUDA **13.3** (PTX ISA 9.3); target `sm_120a` (FP4
  `mma.sync.block_scale`); smem 100 KB/SM. `-DTESSERA_ENABLE_CUDA=ON`.

### 7.5 Two fleet superpowers to exploit (not just coordinate)

Three silicon systems are a capability, not only a logistics problem:

- **Cross-backend differential equivalence** — run the same Graph IR on Apple +
  ROCm + NVIDIA and cross-compare. A miscompile that happens to agree with numpy
  on one backend rarely agrees on all three, so the fleet *is* a correctness
  engine. Generalize the existing Apple differential generator across the fleet
  (folds into Workstream D / the F4 oracle; see Theory §8 W-superpowers).
- **Fleet-shared autotune cache** — the measured autotune corpus
  (`device+shape-bucket → best candidate + accuracy margin`) is a committed, shared
  artifact, so a config proven on one box warm-starts the others and survives
  across runs (extends Decision #11's SQLite warm-start to the §7.3 sync
  contract). **Wired into D2 (2026-07-08):** `MeasureCache.to_dict`/`save_corpus`/
  `load_corpus` + once-only lazy warm-start of the default cache, keyed by device
  tag (`benchmarks/baselines/autotune_corpus.json`; gfx1151 rows committed). The
  cross-backend differential half stays folded into the F4 oracle (C3-precursor).

---

## 8. Beyond this plan — the world-class dimensions (tracked, not on the critical path)

Workstreams A–E make the **kernel spine** (generation + selection) world-class
across the fleet. A world-class *deep-learning* compiler needs more; the
dimensions are enumerated and rationalized in
[`COMPILER_THEORY_OF_OPERATION.md`](COMPILER_THEORY_OF_OPERATION.md) §8 (W1–W8).
They graduate into workstreams **after** the spine is proven, in this rough
priority (highest DL leverage first):

- **F · Low-precision numerics (W1)** — build the tolerance derivation + the
  end-to-end accuracy guard behind the accuracy-budget mechanism (C3-precursor
  `KernelRunner.accuracy_atol` + D2). *Highest priority:
  it is the precision frontier and it is already half-specified.* Proof: `[AMD]`
  RDNA4/CDNA fp8, `[NV]` sm_120 fp4.
- **G · Dynamic shapes (W2) — partly pulled into the spine (decision 2026-07-02).**
  The *interface* half is now a **Workstream B/C/D design constraint**, not
  deferred: the `KernelEmitter`/`TargetPlugin` API carries symbolic dims + a
  `static | bucket | dynamic` specialization policy, the arbiter keys on
  shape-bucket, and first impls emit `bucket` (covers LLM serving's variable
  seq-len / KV-len via bucketing). **The `dynamic` emitter landed for the generic
  lanes (2026-07-08).** The generic scalar kernels (x86 C / ROCm HIP / NVIDIA CUDA)
  already take M/N/K as runtime args with in-kernel bounds guards, so their source
  is dims-invariant — `SpecPolicy.DYNAMIC` now emits (the `EmitError` stubs are
  gone) with a symbolic-identity `shape_key`, collapsing the arbiter/AOT cache to
  **one entry across all shapes** (no per-bucket recompile for variable-shape
  serving) vs BUCKET's one-compile-per-bucket. Proven: host-free (DYNAMIC source ==
  BUCKET source, cache key shape-independent, across all three lanes) + live
  gfx1151 (one DYNAMIC compile serves 64³/256×128×96/512³, all F4-correct, one
  cache entry — `test_dynamic_shape_emit.py`). The bucket-specialized tensor-core
  lanes (WMMA / `mma.sync`) legitimately stay BUCKET. **Still open:** a `dynamic`
  path for the *shape-specialized* lanes (Apple MSL materialized-score kernels,
  tiled tensor-core) that bake dims into codegen. Mostly `[MAC]`.
- **H · Memory planning + layout (W3+W4)** — global buffer assignment/reuse +
  wire `LayoutAssignmentPass` to a backend consumer + transpose elimination.
  Unblocks the deferred attention-dispatch orientation bug. Mostly `[MAC]`.
  **Recon + first slices (2026-07-08):** transpose elimination is already
  substantially done — `transpose(transpose(x)) → x` (`TransposeOp::getCanonicalizationPatterns`,
  TesseraOps.cpp) + transpose-into-matmul flag folding (`TransposeIntoMatmul`,
  CanonicalizeTesseraIR.cpp) both land + are lit-gated, and attention orientation
  was closed by M2 (`AttentionRegion.q_transposed/k_transposed`). **The
  backend-consumer slice for plain GEMM landed:** `MatmulRegion` now carries
  `transpose_a/transpose_b` + `_natural` and its `reference` orients per the flags
  (mirroring `AttentionRegion`), so the arbiter's GEMM candidates resolve operand
  orientation from the transpose contract — **not value shapes** — closing the
  `OPTIMIZING_COMPILER_PLAN` §6 "ambiguous when M==K==N" note for GEMM; the NVIDIA
  `mma.sync`/PTX matmul candidates orient via `_natural` before launch.
  Host-free-gated (`test_fusion_matmul_orientation.py`, incl. the square-ambiguity
  centerpiece); live GEMM proof is NV-gated. **Global buffer assignment/reuse
  landed (2026-07-08):** `TileBufferReusePass` (`--tessera-tile-buffer-reuse`)
  assigns disjoint-live-range `tile.alloc_shared`/`tile.tmem.alloc` buffers of
  identical memref type to shared reuse groups (`tile.buffer_group`) via a
  left-edge interval coloring over per-buffer live ranges, and records the static
  footprint saved (`tile.buffer_reuse.bytes_before/after/groups`). Correct by
  construction (only non-overlapping ranges alias); the assignment half of
  shared-memory planning, paired with `TileBarrierReuseLegalityPass` as verifier —
  the same two-sided pattern as `LayoutAssignmentPass`↔`LayoutLegalityPass`. v1
  output is IR metadata a shared-memory-aware backend consumes; lit-gated
  (`tests/tessera-ir/phase3/tile_buffer_reuse.mlir`). **`tile.buffer_group` now has
  a consumer (2026-07-08):** `TileBufferArenaPass` (`--tessera-tile-buffer-arena`)
  realizes the reuse plan into a concrete per-space arena — `tile.smem_offset` /
  `tile.tmem_offset` on each alloc (same-group buffers share an offset = the
  promised aliasing) + `tile.{smem,tmem}_arena_bytes` on the func, the exact form a
  shared-memory backend emits (`__shared__ char arena[N]; buf = arena + offset`).
  SMEM/TMEM get separate arenas. lit-gated
  (`tests/tessera-ir/phase3/tile_buffer_arena.mlir`). **Still open:** the
  `LayoutAssignmentPass` `tessera.layout` attrs are still unconsumed; a HIP/PTX
  emitter that actually allocates from the arena offsets; transpose-through-pointwise.
- **I · Training-graph + distributed optimization (W5+W6)** — apply the middle-end
  to backward graphs; promote comm/compute overlap from runtime machinery to a
  scheduled pass. Needs multi-rank (mock-collective today).
- **J · Absolute roofline attainment (W7)** — make `% of peak` (not "beats
  per-op") the hot-path success bar; add attainment targets to the E2 ratchets.
  **First slice landed (2026-07-08):** `benchmarks/roofline.py` — a grounded
  per-device peak table (`rocm:gfx1151` = 29.7 TF fp32 / 59.4 TF fp16 / 256 GB/s,
  each with a `source` string deriving it from `rocminfo` CU/SIMD/clock, Decision
  #27), FLOP/byte models (matmul 2·MNK, flash_attn 4·B·H·S²·D), and
  `achieved_tflops`/`pct_peak`/`evaluate_attainment`. The committed gfx1151 ratchet
  rows now carry `pct_peak` + `achieved_tflops` + an `attainment_floor` (=
  `pct_peak / margin`, symmetric with the latency cap), computed from the existing
  medians (no re-timing); `perf_gate --attainment` gates a row that regresses below
  its floor. **Honest scope:** the ratchet median is end-to-end wall-clock
  (H2D/launch/D2H + compile), so `pct_peak` is an end-to-end attainment (a lower
  bound on kernel efficiency) — the current gfx1151 lanes sit at ~0.3–2.9%, so the
  metric's immediate value is exposing the headroom and giving it a ratchet.
  Host-free-gated (`test_roofline_attainment.py`, 12). **Still open:** kernel-
  isolated attainment (strip host overhead), `[NV]` sm_120 + Apple peak rows,
  attainment floors that ratchet *upward* as the lanes optimize.
- **K · Long-tail op codegen (W8) — reassessed 2026-07-08.** W8's *specific*
  premise — a ~125-op numpy-only tail needing generic elementwise/reduction/
  scatter/gather synthesis — is stale: those families are native
  (`generated/e2e_op_coverage.md` = 280 complete / 6 runnable_reference / 0
  artifact_only; gather/scatter*/argmax/cumsum/sort/where/softmax are `fused`/
  `compiled`/`hardware_verified`) after the warp-shuffle lanes (#152–#157 +
  follow-ons). **But per-target native coverage is NOT complete** — the E2E rollup
  is cross-target and collapses per-target detail; the status truth is
  `s_series_status.md` **Backend Proof By Target** (Decision #26 — read the
  dashboard, not this prose). Open native-codegen work there: the **EBM domain
  family** (`ebm_energy` / `ebm_sphere_langevin_{sample,step}` /
  `ebm_bivector_langevin_sample` / `ebm_partition_{ais,exact,monte_carlo}` /
  `ebm_decode_init` — x86 `reference`, ROCm `planned`), ROCm `gemm`
  (`artifact_only`), and the gated NVIDIA/Apple sets. **Closing it is under way
  (2026-07-08):** `ebm_bivector_langevin_step` now runs a native affine-Langevin
  kernel on **both** x86 (AVX-512 `tessera_x86_ebm_affine_langevin_f32`) and ROCm
  (`generate-rocm-ebm-affine-langevin-kernel`, gfx1151) — the manifold step is the
  affine combo `y − η·grad + s·noise` on grade-projected coeffs with host-drawn
  noise as an input (mirrors the Apple bridge). both `ebm_bivector_langevin_step`
  **and** `ebm_sphere_langevin_step` now run native on both backends (x86 9→7
  `reference`, ROCm 10→8 open). Sphere reuses the *same* affine kernel — host
  tangent-projection + affine core + host normalize (retract), no dedicated kernel
  (like bivector's host grade-projection). The affine kernel is reusable infra.
  **Honest-count corrections (same day):** `ebm_partition_{monte_carlo,ais}` are
  host-orchestrated samplers over a *user* `energy_fn` with no kernel on any backend
  (Apple GPU is `reference` too) — reclassified `planned`→`reference` so the
  Backend-Proof tally stops counting them open (ROCm open 8→6). And `gemm` — an
  arity-2 alias of `matmul` emitting the same `tessera_rocm_wmma_gemm_f16` symbol —
  now resolves to `matmul`'s manifest on every target (`hardware_verified` on ROCm/
  sm_120, `fused` on x86/Apple), a reporting-gap fix, not a kernel (ROCm open 6→5).
  **Still per-target open:** the `_sample` chains (host loops — no on-device chain
  kernel), `ebm_energy` / `ebm_partition_exact` / `ebm_decode_init` / `ebm_ebt_tiny`
  (Apple has `fused` kernels → a ROCm kernel is genuinely achievable — real work).
  Separately, the **6 collective/MoE-transport** ops at `reference` need real
  NCCL/RCCL (Workstream I / W6), not op synthesis. So: no *generic-synthesis* gap
  remains, but real per-target + domain codegen does — do not read this as "op
  coverage done."

These are the road to world-class; A–E are the foundation they stand on.

---

## 9. First concrete step

**Phase 0, E1 golden-IR harness `[MAC]`.** Before any refactor: build the
host-free Target-IR snapshot + diff for all four backends over a fixed fixture
set, so the lead-backend regression tripwire exists first. Then capture the E2
ratchet baselines on the two silicon boxes (`[AMD]` gfx1151, `[NV]` sm_120). Only
after Phase 0 is green does Workstream A begin.

### 9.1 Seam verification results (2026-07-02, source-verified)

The three seams the plan rests on were verified against source before any Phase 0
code. Verdicts:

1. **`fusion.py` split boundary (B1) — CLEAN. Proceed.** `FusedRegion` references
   epilogue/reduction ops by **name** (string), not by `EpilogueOp`/`ReductionOp`
   objects, so the arch-agnostic core (`FusedRegion` semantics, `discover_*`,
   `fusion_cost`, `verify_synthesized_region`) carries **no** Metal state; the
   caps (`SYNTH_MAX_N`=1024, `SYNTH_MAX_D`=256, `SYNTH_MAX_N_TILED`=8192) are
   memory-model numbers, not Metal register counts; the F4 oracle is a numpy
   reference compare. Only the `.msl` fields + `synthesize_*_msl` + the
   `run_fused_region` dispatch glue move to the emitter. The plan's "F1/F3/F4/F5
   lift unchanged" holds.

2. **`ptx_emit.py` pipeline (C2) — PROBLEM. Bridge is net-new, not a wiring job.**
   The emit functions (`emit_wgmma_matmul_ptx`, `emit_mma_sync_matmul_ptx`,
   validators, `ptxas_assemble`) are a clean callable API, but **coverage is
   bf16-only** (wgmma M=64/K=16/N∈{64,128,256}; mma.sync a single `m16n8k16`
   tile) and — critically — **the sm_120 matmul that executes today runs via the
   shipped `libtessera_nvidia_gemm.so` symbol, NOT the emit path.** There is no
   serialize→`ptxas`→CUBIN→`tsrRegisterGpuLauncher` bridge. So **C2 must build the
   NVIDIA compile-and-launch bridge (the counterpart to Apple's
   `apple_gpu_runtime.mm` metal_runtime path) + an artifact/kernel cache** — that
   is the bulk of C2, ahead of broadening shapes/dtypes. *(Plan updated: see
   Workstream C2.)*

3. **Golden-IR determinism (E1) — CLEAN WITH CAVEATS. Two defensive tasks.**
   Emitted IR is deterministic today (insertion-order dicts, list iteration; the
   only `set`s in `target_ir.py` feed decision logic, never output; fixtures use
   ordered `CHECK`, not `CHECK-DAG`). Before relying on a byte-exact golden diff,
   add: **(a)** a canonical key-sort in `target_ir.py::_format_attr_dict` (~:1705)
   so attr order is construction-path-independent, and **(b)** a determinism
   roundtrip test (run `tessera-opt` twice, assert byte-identical) — both `[MAC]`,
   folded into E1.

### 9.2 First concrete step

Given the above, Phase 0 order is: **E1 golden-IR harness** (with the 9.1(3)
defensive sort + roundtrip test) `[MAC]` first, then E2 ratchet baselines on the
silicon boxes. Workstream A (mechanical dedup) can start in parallel since B1 is
clean. **C2's launch-bridge scope (9.1(2)) is now the long pole of the lead-lane
work** — sequence it accordingly.
