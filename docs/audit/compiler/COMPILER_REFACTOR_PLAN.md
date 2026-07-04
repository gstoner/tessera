---
last_updated: 2026-07-02
audit_role: plan
plan_state: open
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
- **A3 · Declarative shape/constraint verifiers** `[MAC]` — replace the 6
  hand-written `verify*()` in the 57 KB `TileToApple.cpp`.
- **A4 · Promote ROCm's `MmaDescriptor` + `M×N//lanes` footprint model** `[MAC]`
  to the shared MMA selector, parameterized by per-arch lane count + shape table.
  The one place a *lead* abstraction lifts upward: Apple/x86/NVIDIA gain a
  cost-aware MMA selector they lack.

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
  wraps the `synthesize_*_msl` bodies byte-identically. **B2b** — replace B1's
  oracle lazy `_apple_msl()` bridge with an injected runner registry (Apple
  self-registers). **B2c** — carry symbolic dims on regions + wire `SpecPolicy`
  so the `requires static shapes` gate becomes a policy; stub the `dynamic`
  emitter behind a clear diagnostic.
  **Symbolic-dim-aware from day one (dynamic-shapes decision, 2026-07-02):** the
  `region` carries symbolic dims (from Graph-IR `dim_names`), and `spec` is the
  **specialization policy** `static | bucket | dynamic`. First impls emit
  `bucket` (compile per shape-bucket — seq-len / batch / KV-len — dispatched by
  runtime shape); the interface is designed so a later `dynamic` (runtime-arg +
  guards) emitter drops in **without an API break**. This is the "pull it
  forward" — the *interface* is dynamic-ready now; the *implementation* starts at
  bucket. See Theory §8 W2.
- **B3 · F4 oracle as universal correctness gate** `[MAC]` — already
  numpy-reference and arch-independent; wire every backend's synthesized kernel
  through the same `verify_synthesized_*` before trust. This is what makes
  compiled kernels *safe to prefer*.
- **B4 · Generic synth→compile→cache→launch loop** `[MAC]` for the loop; compile
  fn is per-arch — extract from `apple_gpu_runtime.mm` (`newLibraryWithSource` +
  sha256 `cache_key`); plugin supplies `metallib`/`ptxas`/`hipcc`/`clang`.

**Lead safety:** B targets the *fusable-DAG middle ground* (epilogues, pointwise
chains, small attention). Crown-jewel GEMM stays Tier 2/3.

### Workstream C — Per-arch codegen plugin interface + the missing lead lanes

- **C1 · `TargetPlugin` interface** `[MAC]` — `{emit_kernel, shape_table,
  cost_model, intrinsic_set, async_model, compile_fn, spec_policy}`. Apple + x86
  are the first two reference impls (simplest to validate host-free / on Zen 5).
  `spec_policy` declares which specialization modes the plugin supports (`static |
  bucket | dynamic`) + its bucketing strategy — so the static-shape gate now in
  the lowering (`"requires static shapes"` in `TileToX86Pass` / `MatmulToAppleCPU`)
  is replaced by a *policy*, not re-hardcoded per backend. The **x86 plugin's
  Tier-3 candidate set** should register **AOCL-DLP** ([amd/aocl-dlp](https://github.com/amd/aocl-dlp))
  for the Zen family — AMD's BLIS-family DL primitives (low-precision GEMM/batch
  GEMM incl. INT4/FP16, pre/post-ops matching `fused_epilogue`, symmetric quant,
  OpenMP). It's AVX512-based (fits the Zen 5 fleet box, which has no AMX), fills
  the x86 backend's OpenMP-threading + INT4/FP16 gaps, and is opt-in behind a
  build flag (a BLAS-family library like Accelerate — Decision #23-clean, kept
  behind the hardware-free Target IR). The arbiter (D1) selects it only where it
  measures faster than the generic kernels on Zen; check its license before it
  becomes a shipped/linked lane.
- **C2 · NVIDIA in-process emit pipeline** `[MAC]` authoring → `[NV]` proof —
  `tessera-opt --tessera-emit-nvidia`: Tile IR → `ptx_emit.py` (keep sm_120
  `mma.sync`; extend `wgmma` for sm_90a; stub sm_100 tcgen05) → serialize →
  launch bridge. **New lead capability**, not a refactor: NVIDIA gains an
  in-process compiled lane alongside the shipped `.so`. **Scope correction
  (§9.1(2), source-verified):** the bulk of C2 is the **net-new serialize→`ptxas`
  →CUBIN→`tsrRegisterGpuLauncher` bridge + kernel/artifact cache** (the NVIDIA
  counterpart to Apple's `apple_gpu_runtime.mm`). `ptx_emit.py` is a clean but
  **bf16-only, few-shape** emitter, and today's executing sm_120 matmul runs via
  the shipped `libtessera_nvidia_gemm.so`, **not** the emit path — so the bridge
  is the long pole, ahead of broadening shapes/dtypes.
- **C3 · ROCm in-process emit pipeline** `[MAC]` authoring → `[AMD]` proof —
  `--tessera-emit-rocm`: drives the existing gfx1151 WMMA + CDNA MFMA `Generate*`
  passes through the shared loop into the launch bridge; reuses the async-token
  SSA model in `ROCMWaveLdsPipeline`.

### Workstream D — Candidate arbitration + measured autotune

- **D1 · Candidate registry** `[MAC]` — per `(op, shape-bucket, dtype, target)`
  enumerate `{synthesized, tier2_emitted, hand_tuned_1..n}`; generalize Apple's
  `select_variant` + `best_record`. The key is **shape-bucket, not exact shape**
  (dynamic-shapes decision) so one tuned kernel serves a bucket of runtime shapes.
- **D2 · Measured autotune loop** — `[AMD]` on gfx1151, `[NV]` on sm_120 run
  live; CDNA/sm_90/sm_100 fall back to analytical roofline + `MmaDescriptor` cost
  model until silicon. Measure-at-first-miss + cache keyed by
  `device+shape-bucket+accuracy-margin`.
- **D3 · Fallback log everywhere** `[MAC]` — generalize
  `dispatch_fallback_log`/`fallback_histogram` so "did the compiled path win or
  silently degrade?" is answerable per backend.

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
- **E3 · Escape-hatch test** `[MAC]` per backend — assert a hand-tuned kernel
  *can* be forced and *does* win when the cost model says so. Proves Tier 3 is
  never orphaned.

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
| Shared abstraction crips a lead | Theory rule #1 + E1/E2: lead opts out per op; IR/perf gated |
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
  contract). Wire into D1/D2.

### 7.4 Per-system setup pins (from `docs/GETTING_STARTED.md`)

- **Mac:** Homebrew LLVM/MLIR **22.1.6** at `/opt/homebrew/opt/llvm`; off-venv
  `python3` 3.14.5.
- **Strix Halo:** Ubuntu 24.04 + `scripts/setup_ubuntu.sh` (LLVM/MLIR 22 from
  apt.llvm.org — ROCm's bundled LLVM has no MLIR); ROCm **7.2.4** at `/opt/rocm`;
  `-DTESSERA_ENABLE_HIP=ON -DTESSERA_BUILD_ROCM_BACKEND=ON`; `.venv` numpy<2.2.
  gfx1151 = RDNA 3.5, WMMA 16×16×16, **no FP8 WMMA**.
- **NR2 Pro:** CUDA **13.3** (PTX ISA 9.3); target `sm_120a` (FP4
  `mma.sync.block_scale`); smem 100 KB/SM. `-DTESSERA_ENABLE_CUDA=ON`.

---

## 8. Beyond this plan — the world-class dimensions (tracked, not on the critical path)

Workstreams A–E make the **kernel spine** (generation + selection) world-class
across the fleet. A world-class *deep-learning* compiler needs more; the
dimensions are enumerated and rationalized in
[`COMPILER_THEORY_OF_OPERATION.md`](COMPILER_THEORY_OF_OPERATION.md) §8 (W1–W8).
They graduate into workstreams **after** the spine is proven, in this rough
priority (highest DL leverage first):

- **F · Low-precision numerics (W1)** — build the tolerance derivation + the
  end-to-end accuracy guard behind the §4.1 budget mechanism. *Highest priority:
  it is the precision frontier and it is already half-specified.* Proof: `[AMD]`
  RDNA4/CDNA fp8, `[NV]` sm_120 fp4.
- **G · Dynamic shapes (W2) — partly pulled into the spine (decision 2026-07-02).**
  The *interface* half is now a **Workstream B/C/D design constraint**, not
  deferred: the `KernelEmitter`/`TargetPlugin` API carries symbolic dims + a
  `static | bucket | dynamic` specialization policy, the arbiter keys on
  shape-bucket, and first impls emit `bucket` (covers LLM serving's variable
  seq-len / KV-len via bucketing). **What remains as G** is only the full
  `dynamic` emitter (runtime-arg dims + bounds guards) for shapes that don't
  bucket well — it drops into the day-one API without a break. Mostly `[MAC]`.
- **H · Memory planning + layout (W3+W4)** — global buffer assignment/reuse +
  wire `LayoutAssignmentPass` to a backend consumer + transpose elimination.
  Unblocks the deferred attention-dispatch orientation bug. Mostly `[MAC]`.
- **I · Training-graph + distributed optimization (W5+W6)** — apply the middle-end
  to backward graphs; promote comm/compute overlap from runtime machinery to a
  scheduled pass. Needs multi-rank (mock-collective today).
- **J · Absolute roofline attainment (W7)** — make `% of peak` (not "beats
  per-op") the hot-path success bar; add attainment targets to the E2 ratchets.
- **K · Long-tail op codegen (W8)** — generic elementwise/reduction/scatter/gather
  synthesis to close the ~125 numpy-only ops the residency planner only *routes*.

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
