---
last_updated: 2026-07-02
audit_role: plan
plan_state: open
---

# Tessera Compiler — Theory of Operation

> **Paired with** [`COMPILER_REFACTOR_PLAN.md`](COMPILER_REFACTOR_PLAN.md) (the
> execution plan). This document is the durable *conceptual model*: how the
> pieces fit, what the invariants are, and why the shape is what it is. The plan
> changes as work lands; this document changes only when the architecture does.
>
> **Builds on**, does not replace:
> [`OPTIMIZING_COMPILER_PLAN.md`](OPTIMIZING_COMPILER_PLAN.md) (F0–F6 middle-end
> synthesis), [`EVALUATOR_PLAN.md`](EVALUATOR_PLAN.md) (the scoring engine that
> gates promotion), [`COMPILER_AUDIT.md`](COMPILER_AUDIT.md) (current state), and
> [`STAGE_A_EMIT_PLAN.md`](STAGE_A_EMIT_PLAN.md) (cross-vendor emit grounding).

---

## 0. The one-sentence model

Tessera compiles a hardware-free op graph into a **kernel candidate set** per
`(op, shape, dtype, target)`, and a **measured arbiter** picks the fastest
candidate — where candidates come from three tiers: a **generic synthesizer**, a
**per-arch codegen plugin**, and a **hand-tuned kernel library**.

Everything else in this document is the elaboration of that sentence.

---

## 1. Governing principle — the leads set the ceiling, the framework raises the floor

ROCm (AMD) and CUDA (NVIDIA) are the **lead performance targets**. Apple is
close behind. x86 is the reference floor. The entire architecture is arranged so
that shared/generic infrastructure **can never cap a lead backend's ceiling**:

1. **No lowest-common-denominator.** A shared abstraction that cannot express
   `wgmma` / `tcgen05` / MFMA / SWMMAC-sparse / async-pipelining is *wrong*. The
   lead backend keeps its hand-emitted path; the abstraction grows to fit, or the
   lead opts out at that op. The generic path is a floor, not a mold.
2. **Hand-tuned kernels are first-class, permanently.** They are the
   highest-priority *candidate* the arbiter measures — never deprecated code. The
   compiler wins only where it measures at least as fast.
3. **Regression-gated on real silicon.** Every change ships with a host-free
   golden-IR diff and a real-hardware perf ratchet on the lead backends. If
   ROCm/CUDA emitted IR or measured latency regresses, the change is blocked
   until proven neutral-or-better.

These three rules are the *why* behind every structural choice below.

---

## 2. The four-layer IR stack (context)

Unchanged from `CLAUDE.md`; repeated here as the substrate the tiers sit on:

```
Python API  (@jit, Region[...], domain, index_launch)
   │
Graph IR    (tessera dialect — mathematical ops, effects, shapes)
   │
Schedule IR (mesh regions, pipeline stages, optimizer sharding)
   │
Tile IR     (warp/simdgroup specialization, TMEM/LDS, async copy, KV cache)
   │
Target IR   (per-backend, hardware-free dialect: tessera_rocm.*, tessera_apple.*, …)
   │
Hardware emit (MSL / PTX / AMDGCN / C-LLVM)  ← Tier 2 plugin
```

The **hardware-free Target IR** (Decision #19) is what makes every backend
lit-testable *and* what makes the golden-IR regression tripwire host-free (§6).

---

## 3. The three-tier kernel model

This is the core of the theory. For any op or fused region, on any target, a
kernel comes from exactly one of three tiers, chosen by the arbiter (§4):

### Tier 1 — Generic framework (arch-agnostic)

Lives in `compiler/fusion_core.py` + `tessera_common/` (proposed) + the runtime
loop. Arch-independent by construction:

- **Region IR** — `FusedRegion` / `EpilogueOp` / `ReductionOp` *semantics*
  (currently in `fusion.py`; the MSL strings are Tier 2, see §5).
- **Fusion-DAG discovery** — `discover_*` / `should_fuse_*`: which SSA subgraphs
  become one kernel. Pure graph analysis; no hardware knowledge.
- **F4 correctness oracle** — `verify_synthesized_*`: a reference compare that
  gates *any* synthesized kernel on *any* target before it is trusted, **within a
  per-op numerical budget** (see §4.1). Exact for integer / fp32-accumulate paths;
  **tolerance-aware for low precision** (fp16/bf16/fp8/fp4/MX) — a bf16 or fp8
  kernel is *not* bit-equal to the fp32 reference, so the oracle checks the
  accuracy budget the op's `numeric_policy` declares, not bit-equality. This is
  what makes generated kernels safe to prefer over reference.
- **Synth → compile → cache → launch loop** — hash the kernel source (sha256
  `cache_key`), compile once, cache by hash, launch. The loop is generic; the
  *compiler invoked* is the Tier 2 plugin.
- **Measured autotune** — measure-at-first-miss, cache keyed by
  `device+shape-bucket` (`select_variant`, `best_record`), so one tuned kernel
  serves a bucket of runtime shapes (dynamic-shapes decision, §8 W2).
- **Residency planner** — whole-program on-device vs host decision, with an
  enumerable "what cannot stay" residual (`per_op_metal` gate generalized).
- **Fallback log** — `dispatch_fallback_log` / `fallback_histogram`: honest
  runtime record of what degraded to reference and why.

Only Apple has proven Tier 1 end-to-end today. The plan's backward-lift makes it
shared.

### Tier 2 — Per-arch codegen plugin (one per backend)

The `TargetPlugin` contract (§5). Each backend supplies:

| Plugin element | Apple | NVIDIA | ROCm | x86 |
|---|---|---|---|---|
| Emitter | MSL source | PTX (`wgmma`/`mma.sync`/tcgen05) | AMDGCN (`rocdl.wmma`/`mfma`) | C / LLVM |
| Compile fn | `metallib` (`newLibraryWithSource`) | `ptxas` | `hipcc`/`clang` AMDGPU | host `clang` |
| Shape table | simdgroup tiles | wgmma m/n/k | `_MFMA/_WMMA_VARIANTS` | AMX/AVX512 tiles |
| Cost model | measured (M1+) | roofline + measured (sm_120) | `MmaDescriptor` footprint + measured (gfx1151) | roofline |
| Async/sync model | threadgroup | TMA + mbarrier | LDS + waitcnt (async-token SSA) | — |
| Intrinsic set | `simdgroup_matrix` | `mma`/`cp.async` | `V_WMMA_*`/`ds_*` | `_tile_*`/`_mm512_*` |

The leads (NVIDIA/ROCm) supply the *deepest* Tier 2 plugins. This is where their
performance ceiling lives and where the plan adds capability (the missing
in-process NVIDIA/ROCm emit pipelines) **without** routing them through Tier 1's
synthesizer for the crown-jewel GEMM/attention kernels.

### Tier 3 — Hand-tuned kernel library (per arch, permanent)

- Apple: 168 authored MSL kernels in `apple_gpu_runtime.mm`, `.mtlpackage` (PK).
- NVIDIA: shipped `libtessera_nvidia_gemm.so`, future cuBLAS-class kernels.
- ROCm: shipped `libtessera_rocm_{gemm,flash_attn}.so`, rocWMMA/AITER-derived.
- x86: 40+ AMX/AVX-512 kernels; **AOCL-DLP** (AMD's BLIS-family DL primitives —
  low-precision GEMM + batch + pre/post-ops, OpenMP) as an opt-in Zen candidate,
  the CPU analog of cuBLAS/rocWMMA (see Refactor Plan C1).

Tier 3 is not legacy. It is the reason rule #2 exists: the arbiter treats a
hand-tuned kernel as the highest-priority candidate and only lets a compiled one
win on measured latency.

---

## 4. The arbiter — how a kernel is chosen

For each `(op, shape, dtype, target)`:

1. **Enumerate candidates** across tiers: `{synthesized?, tier2_emitted?,
   hand_tuned_1..n?}` (some may be absent for a given key).
2. **Gate for correctness within the accuracy budget** — every
   synthesized/emitted candidate must pass the Tier 1 F4 oracle *against the op's
   declared numerical budget* (§4.1). A candidate outside budget is dropped and
   the fallback log records it.
3. **Score the surviving (in-budget) candidates** — on a system with real silicon
   for `target`, measure latency (measure-at-first-miss, then cache). Without
   silicon, score by the Tier 2 cost model (roofline / `MmaDescriptor` footprint).
4. **Select the fastest in-budget candidate + cache** it keyed by
   `device+shape-bucket` (§8 W2 — one tuned kernel serves a bucket of shapes).
5. **Record** — the winning tier + measured latency + the accuracy margin into
   the autotune record and perf ratchet.

The arbiter's objective is therefore **fastest *within* the accuracy budget** —
never fastest-overall. This is the deep-learning-specific rule that lets a lower-
precision kernel (fp8/fp4/MX) win *only* when it both measures faster and stays in
budget; it is what keeps "measured arbiter picks fastest" from silently trading
model quality for speed.

### 4.1 The accuracy budget

Low precision is the DL performance frontier (fp8/fp4/MX on Blackwell sm_120 and
CDNA/RDNA4), so accuracy is a first-class arbiter dimension, not an afterthought:

- **Per-op budget** — declared via the op's `numeric_policy` (storage/accum,
  `math_mode`; Decision #15a). The F4 oracle compares against a wide-precision
  reference to a tolerance derived from that policy, not to bit-equality.
- **End-to-end guard** — a per-kernel budget can pass while error *accumulates*
  across a deep graph. A model-level accuracy check (tie into the Evaluator's
  metamorphic oracle, `EVALUATOR_PLAN`) validates the whole compiled program, so
  no chain of individually-in-budget kernels drifts the model out of budget.
- **Recorded margin** — the arbiter records each winner's accuracy margin, so a
  later toolchain/kernel change that erodes it is visible (the numerical analogue
  of the perf ratchet).

**Consequence for the leads:** on the ROCm and NVIDIA boxes, a hand-tuned MFMA or
`wgmma` kernel that is fastest is *automatically* chosen. The generic path can
only displace it by measuring faster on that silicon. There is no way for a
refactor to silently downgrade a lead — the arbiter is the enforcement mechanism
for rule #2, and the perf ratchet (§6) is the CI enforcement for rule #3.

---

## 5. The plugin contract — where generic ends and arch-specific begins

The single most important seam. Today `fusion.py` welds the arch-agnostic region
model to Metal string emission (the `EpilogueOp.msl` field, the
`synthesize_*_msl` functions). The theory draws the line here:

```
GENERIC (Tier 1)                         │  PER-ARCH (Tier 2 plugin)
─────────────────────────────────────────┼──────────────────────────────────
FusedRegion / EpilogueOp semantics        │  EpilogueOp.emit(target) → source
discover_* / should_fuse_*                 │  KernelEmitter.emit(region, target)
verify_synthesized_* (F4 oracle, numpy)    │  compile_fn(source) → binary
select_variant / best_record (loop)        │  shape_table / cost_model
residency planner / fallback log           │  intrinsic_set / async_model
```

**The rule for drawing the line:** if a step needs to know *how the hardware
multiplies matrices or moves memory*, it is Tier 2. If it only needs to know
*what the math is and whether two kernels agree*, it is Tier 1.

A backend "joins" by implementing `TargetPlugin`. A backend "opts out" of the
synthesizer for a given op simply by having its Tier 2 plugin (or Tier 3 library)
provide a candidate the arbiter prefers — no special-casing required.

**Shape is part of the contract (dynamic-shapes decision, 2026-07-02).** The
region carries symbolic dims and the emitter takes a **specialization policy**
`static | bucket | dynamic`; `TargetPlugin` declares which modes it supports. The
interface is symbolic-dim-aware from day one even though the first implementations
emit `bucket` (compile per shape-bucket, dispatch by runtime shape). This keeps
the static-shape gate a *policy*, not a hardcoded `requires static shapes` in each
backend's lowering — and lets a full `dynamic` emitter arrive later without an API
break. See §8 W2.

---

## 6. The multi-system reality — three machines, one compiler

Some tasks **can only run on specific silicon**. The architecture accounts for
this with a hub-and-spoke model: the Mac authors and runs everything hardware-
free; the two Linux boxes run the silicon-only gates and commit recorded results
back.

### 6.1 The three systems and their exclusive capabilities

| System | Hardware | Toolchain | Can ONLY be done here |
|---|---|---|---|
| **Mac** (dev/authoring hub) | Apple M1 Max (Apple7) GPU + CPU | Homebrew LLVM/MLIR 23, macOS, off-venv `python3` 3.14.5 | Apple MSL/`metallib` compile + execute; Metal 4 / MPSGraph; **all hardware-free work** (lit, mypy, unit tests, IR, **golden-IR generation**) |
| **Strix Halo** (AMD box) | Ryzen AI Max+ 395 — Zen 5 (AVX-512 VNNI/BF16, **no AMX**) + Radeon 8060S **gfx1151** RDNA 3.5 | Ubuntu 24.04 + ROCm 7.2.4, LLVM/MLIR 23 (apt.llvm.org), `.venv` numpy<2.2 | ROCm gfx1151 WMMA **execute-and-compare**; **x86 AVX-512 native execute** (Zen 5); AMDGCN codegen via open LLVM AMDGPU; measured autotune on gfx1151 |
| **NR2 Pro** (NVIDIA box) | RTX 5070 Ti **sm_120** Blackwell; Core Ultra 7 265F (**no AVX-512, no AMX**) | Linux + CUDA 13.3, `ptxas`, PTX ISA 9.3 | `ptxas` assemble (rung 3); CUDA **execute-and-compare** (rung 7); sm_120 `mma.sync` / FP4; measured autotune on sm_120 |

**Known hardware gap:** Intel **AMX** has no home in this fleet (Intel-Xeon-only;
Zen 5 has AVX-512 but not AMX; the 265F has neither). The x86 **AVX-512** path
validates natively on the Zen 5 box; the AMX fast-path stays hardware-gated until
AMX silicon exists. Do **not** build the x86 backend with `-mavx512*` on the 265F
(it would SIGILL — Arrow Lake fused AVX-512 off).

### 6.2 The coordination invariant — host-free first, silicon confirms

The load-bearing trick: **the regression tripwire for the leads is host-free.**
Because Target IR is hardware-free (Decision #19), the Mac can emit and diff
ROCm/NVIDIA Target IR *without* the GPUs. So:

1. **Author on the Mac.** Every refactor passes the host-free gate — lit, unit,
   mypy, and the **golden-IR diff** for all four backends — before it touches
   silicon. A change that perturbs a lead's emitted IR fails here, on the Mac,
   with no GPU in the room.
2. **Confirm on silicon.** Only two things *require* a GPU box: (a) real
   **execute-and-compare** (numerical proof) and (b) the **measured perf
   ratchet**. These run on the box that owns the target, and their outputs are
   **recorded artifacts committed back to the repo** (`rocm_*_hot_paths.json`,
   `nvidia_*_hot_paths.json`, execute-compare fixtures) — the same pattern as the
   existing `apple_gpu_hot_paths.json`.
3. **The recorded artifact is the cross-system contract.** Once committed, the
   Mac's host-free gate can assert the *shape* of the recorded proof (fixture
   exists, ratchet not regressed) even though it cannot regenerate it. This is
   how a Mac-authored refactor stays honest about the leads between silicon runs.

### 6.3 Work routing

```
                    ┌──────────────────────────────┐
                    │            MAC (hub)          │
                    │  authoring · hardware-free CI │
                    │  golden-IR gen · Apple execute│
                    └───────────────┬──────────────┘
                    push branch     │     pull recorded proofs
              ┌─────────────────────┼─────────────────────┐
              ▼                                            ▼
   ┌────────────────────┐                     ┌────────────────────┐
   │   STRIX HALO box   │                     │    NR2 PRO box      │
   │ ROCm gfx1151 exec  │                     │ CUDA sm_120 exec    │
   │ x86 AVX-512 exec   │                     │ ptxas assemble      │
   │ AMDGCN codegen     │                     │ sm_120 measured AT  │
   │ measured AT (RDNA) │                     │                     │
   └────────────────────┘                     └────────────────────┘
```

A task is tagged with the system(s) that can run it (the plan's routing matrix,
`COMPILER_REFACTOR_PLAN.md` §7). The default is **Mac-first**: if a task *can* be
done host-free, it is, and the silicon boxes are reserved for the two things only
they can do.

---

## 7. Invariants (what must always hold)

1. **Every synthesized/emitted kernel passes the F4 oracle within its accuracy
   budget before it is trusted** (§4.1). No exceptions; out-of-budget → drop
   candidate + log fallback.
2. **A lead backend's crown-jewel kernels never route through a generic path that
   is measurably slower or less accurate.** The arbiter enforces this per shape —
   fastest *within the accuracy budget*, never fastest-overall.
3. **Target IR is hardware-free.** Anything requiring a driver/compiler lives in
   the Tier 2 plugin, never in Graph/Schedule/Tile/Target IR.
4. **Host-free before silicon.** No refactor lands on a lead backend without
   passing the Mac's golden-IR gate first.
5. **Silicon proofs are recorded artifacts.** Execute-and-compare fixtures and
   perf ratchets are committed, not transient — the cross-system contract.
6. **Unsupported lowering emits a stable diagnostic** (Decision #21) — never a
   silent no-op or silent numpy fallback (the fallback log makes silent
   degradation visible).
7. **The generated dashboards remain count truth** (Decision #25/#26). This plan
   and its progress never copy counts into prose.

---

## 8. World-class scope register — what these plans do *not* yet cover

The three-tier + arbiter model covers **kernel generation and selection**. A
world-class *deep-learning* optimizing compiler needs more than fast kernels. The
dimensions below are **deliberately out of the current plan's critical path** but
in scope for the compiler's endgame; they are named here so they are tracked, not
silently missing. Each will graduate into its own workstream (Refactor Plan §9)
when the kernel spine is proven across the fleet.

| # | Dimension | Why world-class DL needs it | Current state / seam |
|---|---|---|---|
| **W1** | **Low-precision numerics** (fp8/fp4/MX accuracy budgets, per-op + end-to-end) | The performance frontier is precision, not just fusion; correctness must be accuracy-aware | Partially folded in as §4.1 — the *budget mechanism* is now first-class; the *tolerance derivation + model-level guard* are unbuilt (tie to `numeric_policy` + Evaluator metamorphic oracle) |
| **W2** | **Dynamic shapes** (symbolic dims, bucketed specialization, guards) | LLM serving is variable seq-len + growing KV cache; static-only is a non-starter for inference | **First production routes live (2026-07-22):** generic emitters have shape-independent dynamic identities and x86 executes guarded rank-2 matmul, contiguous last-axis reduction/softmax, runtime-sequence attention, and growing KV-cache movement through runtime-dimension ABIs. Rank/contraction/extent/side-buffer/ABI guards run before native entry; one attention artifact spans unequal query/key lengths and one KV artifact spans cache capacities. **Remaining:** architecture-owned dynamic policies for Apple MSL/tensor-core lanes that still specialize by bucket. |
| **W3** | **Memory planning** (buffer reuse / liveness / allocation, global rematerialization) | Peak-memory is often the binding constraint in training; fusion alone doesn't plan buffers | `InsertRecomputePass` (budget-guided) + `checkpoint.py` exist but aren't a global buffer-assignment pass in the executed path |
| **W4** | **Layout & data-movement optimization** (layout propagation, transpose elimination, packing) as a *wired* pass | A large share of real DL latency is data movement, not FLOPs | `LayoutAssignmentPass` propagates an agreed layout through pointwise chains and into last-axis reduction, inserts provenance-bearing casts, and preserves packed-storage attributes. The generic x86 emitter consumes executable row/column-major binding contracts; Apple now consumes row-major/BHSD/NHWC Graph casts at its runtime boundary, and NVIDIA carries row/column-major/BHSD/NHWC bindings into Tile staging copies. Graph cast insertion remains deliberately opt-in while the consumer set broadens. **Remaining:** finish the complete matmul → epilogue → reduction route, broaden packed-storage rewrite coverage, then enable assignment only in pipelines whose entire producer/consumer envelope has a materializer. |
| **W5** | **Training / backward-graph optimization** (backward fusion, optimizer-step fusion, remat as global opt) | The compiler must optimize the *whole* training step, not just forward inference | Native Graph-IR adjoints now cover matmul, tanh/sigmoid, collectives, same-shape add/multiply, static broadcast inversion, kind-aware sum/mean reductions, GELU/SiLU/ReLU, softmax, and unary/affine RMSNorm/LayerNorm. ReLU uses a registered scalar comparison mask; integer comparisons carry explicit signedness. Normalization uses rank-reduced statistics plus runtime-dimension-carrying broadcast-in-dimension, emits `dx`/`dgamma`/`dbeta`, and lowers for static and dynamic shapes through linalg. Runtime-shaped affine forward and recompute-all backward execution are exact-device proven through public JIT/runtime seams on gfx1151 and AVX-512. Dynamic broadcast and dynamic mean retain Python-VJP fallbacks; max/min retain placeholders until tie semantics are specified. **Remaining:** Apple/NVIDIA normalization-backward materializers, backward epilogue and optimizer-step fusion, and global rematerialization. |
| **W6** | **Distributed optimization** (sharding propagation, collective placement, comm/compute overlap scheduling) | Frontier training is multi-GPU; overlap scheduling is a top differentiator | Schedule IR has mesh/ZeRO/1F1B + MegaMoE overlap exists, but as runtime machinery, not a middle-end optimization pass; multi-rank is still mock-collective |
| **W7** | **Absolute performance truth** (roofline attainment as the success metric) | "Beats per-op dispatch / beats MPS" is *relative*; world-class means *% of peak* | Flywheel + roofline tooling exist; the plans' success bar is relative — should add an absolute attainment target per hot path |
| **W8** | **Long-tail op codegen** (generic elementwise/reduction/scatter/gather synthesis) | **Premise stale (reassessed 2026-07-08).** The named families are native — `generated/e2e_op_coverage.md` = 280 complete / 6 runnable_reference / 0 artifact_only; gather/scatter*/argmax/cumsum/sort/where/softmax are `fused`/`device_verified_jit`/`device_verified_abi` after the warp-shuffle lanes. The old "~125 numpy-only" was registry-conservative prose (Decision #26). | **No *generic-synthesis* gap, but per-target native coverage ≠ complete** — the E2E rollup is cross-target; read `s_series_status.md` **Backend Proof By Target**. Open there: the **EBM domain family** (x86 `reference`, ROCm `planned`), ROCm `gemm` (`artifact_only`), gated NVIDIA/Apple. Plus the **6 collective/MoE** refs (need NCCL/RCCL → W6) + `not_applicable` structural/view/host ops. |

**Two fleet superpowers these dimensions unlock (cheap, high-value):**

- **Cross-backend differential equivalence.** With three silicon systems, the same
  Graph IR can execute on Apple + ROCm + NVIDIA and cross-compare — a correctness
  engine far stronger than any single-backend oracle (a miscompile that agrees
  with numpy on one backend rarely agrees on all three). This generalizes the
  existing Apple differential generator across the fleet.
- **Shared / persistent autotune cache across the fleet.** The measured autotune
  corpus (`device+shape-bucket → best candidate + accuracy margin`) should be a committed,
  fleet-shared artifact — so a config proven on one box warm-starts the others and
  survives across runs (extends Decision #11's SQLite warm-start to the 3-system
  contract in §6).

---

## 9. Glossary

- **Tier 1 / 2 / 3** — generic framework / per-arch plugin / hand-tuned library
  (§3).
- **Arbiter** — the measured candidate selector; picks the fastest candidate
  *within the accuracy budget* (§4).
- **F4 oracle** — `verify_synthesized_*`, the reference-equivalence gate, checked
  within a per-op accuracy budget (§4.1) — exact for int/fp32-accum,
  tolerance-aware for low precision.
- **Accuracy budget** — the per-op (and end-to-end) numerical tolerance the
  arbiter selects within, derived from `numeric_policy` (§4.1).
- **Golden-IR gate** — host-free Target-IR snapshot diff; the lead-backend
  regression tripwire runnable on the Mac.
- **Perf ratchet** — recorded real-hardware latency floor; regression blocks
  merge.
- **Host-free** — runnable without any GPU / vendor driver (lit, mypy, unit, IR,
  golden-IR).
- **Recorded artifact** — a committed proof (fixture / ratchet JSON) that is the
  cross-system contract between the Mac and a silicon box.
