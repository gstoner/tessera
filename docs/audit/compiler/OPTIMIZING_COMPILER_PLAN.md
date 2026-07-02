---
last_updated: 2026-07-02
audit_role: plan
plan_state: open
---

# Tessera Optimizing-Compiler Plan — from op-library to world-class

> Status: proposed (2026-06-14).  Companion to `EVALUATOR_PLAN.md` (the scoring
> engine this plan is gated by) and `COMPILER_AUDIT.md` (current state).
> Scope: the execution middle-end.  The `@jit` frontend (Graph IR emission,
> multi-output ops, scalar-attr lowering) is done; this plan is about what
> *consumes* those graphs.

> ## Reassessment — 2026-07-02 (read before §1)
>
> Two of this document's original assumptions have been overtaken by events.
> **F0–F5 have landed on Apple** (§6), so §1's "op library + dispatcher, not an
> optimizing compiler" describes the *pre-F0 starting point*, not today — the
> middle-end synthesizer exists and is the Apple production path. The live edge
> of this plan is **F6 (the backend lift)**, which is where specific backends get
> built. F6 has been **rewritten below** because its central premise died:
>
> 1. **"This Mac can't provide a CUDA/ROCm runner, so silicon validation is
>    deferred" is dead.** The **Strix Halo** box (Radeon 8060S **gfx1151**, ROCm
>    7.2.4) and the **NR2 Pro** box (RTX 5070 Ti **sm_120**, CUDA 13.3) now exist
>    and *execute* — ROCm gfx1151 runs a compiler-generated matmul + flash-attn
>    family; NVIDIA sm_120 runs its first `mma.sync` matmul. F6 is no longer
>    "design + assemble-on-CI, silicon deferred."
> 2. **"Lift the design, swap the F2 emitter, keep F1/F3/F4/F5" is too simple and
>    partly wrong as a governing rule.** ROCm and CUDA are the **lead performance
>    targets** — their crown-jewel GEMM/attention is hand-emitted `wgmma` /
>    `mma.sync` / MFMA / WMMA and must **not** be forced through the generic
>    synthesizer. The correct model is the **three-tier / measured-arbiter** model
>    in [`COMPILER_THEORY_OF_OPERATION.md`](COMPILER_THEORY_OF_OPERATION.md): the
>    synthesizer generalizes for the *fusable-DAG middle ground*; the leads keep
>    hand-tuned kernels as first-class **arbiter candidates**. Also, the
>    synthesizer is **not** cleanly liftable today — `fusion.py` welds the region
>    model to MSL string emission (the `.msl` fields); the split into a
>    `KernelEmitter` plugin is prerequisite work, tracked as Workstream B/C in
>    [`COMPILER_REFACTOR_PLAN.md`](COMPILER_REFACTOR_PLAN.md).
>
> §1–§5 below are preserved as the original design record (still accurate for the
> Apple middle-end). The reassessed F6 supersedes the original F6.

## 1. The honest current state

Tessera today is an **op library + dispatcher**, not an optimizing compiler:

* Fusion is **enumerated**.  Each fused chain is a hand-written MLIR pass
  (`MatmulSoftmaxFusionToAppleGPU.cpp`, `SwigluFusion`, `MLAFusion`,
  `NativeSparseAttn`, `LightningAttn`, …) that **pattern-matches an SSA chain and
  replaces it with a `func.call` into a prebuilt runtime shim**.
* Kernels are **hand-authored**.  There are **168 hand-written MSL kernels** in
  `apple_gpu_runtime.mm`.  Adding a fusion = write a new MSL kernel + a new
  pattern-match pass + a new driver chain-string.
* Anything outside the ~8-pattern catalog falls back to **per-op dispatch** —
  one kernel launch per op (the decomposition overhead `dlop_longtail_core`
  measures).

The frontend now feeds the middle-end arbitrary op graphs it cannot exploit.
That is the bottleneck this plan closes.

## 2. What "world-class" means (the five pillars)

A world-class optimizing compiler is judged on five axes.  Tessera already has
three of the supporting systems; the gap is the middle-end itself.

| Pillar | What it means | Tessera has | Tessera lacks |
|---|---|---|---|
| **Generality** | *any* graph fuses, not a catalog | frontend op graphs | general region discovery |
| **Synthesis** | the compiler *writes* the kernel | 168 hand-written kernels | a kernel synthesizer |
| **Profitability** | cost-model decides what/when to fuse | flywheel (device-keyed latency) + roofline | a fusion cost model |
| **Correctness** | every optimization is proven, not trusted | the evaluator oracles (horizontal/metamorphic/DESIL) | the gate wired into codegen |
| **Portability + self-improvement** | one middle-end → many backends; it tunes itself | magellan/alphaevolve/autotune_v2; MLIR/LLVM CPU lane | the synthesizer lifted into MLIR |

**The keystone is Synthesis.**  Without it, "general fusion" still needs a
hand-written kernel per region — the catalog just moves.  Synthesis is what turns
"168 kernels + 8 passes" into "1 synthesizer," and it is the capability that
makes Tessera a compiler rather than a library.

## 3. The plan — seven phases (F0–F6)

Each phase is **correctness-gated by the evaluator** (a synthesized/fused kernel
is trusted only when the horizontal-equivalence oracle agrees `fused ≡ unfused`)
and **measured by `dlop_longtail_core`** (dispatch-count + latency).  Apple GPU is
the proving ground; F6 lifts the proven design to MLIR/LLVM for NVIDIA/AMD.

### F0 — Fusion-region IR (the substrate)
A first-class `tessera.fused_region` op (or region attribute) that captures a
maximal producer-consumer subgraph as one schedulable unit — operands in,
results out, the inner op-DAG as its body.  Everything downstream operates on
regions, not ad-hoc SSA-chain matching.
- **Seed:** the Decision #19 Target-IR fusion descriptor already in `main` (7
  fusion passes emit it) becomes the region's metadata.
- **Accept:** a region round-trips through `tessera-opt`; a lit fixture shows
  `matmul → softmax` captured as one `fused_region` with the inner DAG intact.

### F1 — General region discovery (replace the catalog)
One `GeneralFusionPass` that grows maximal fusable regions by fusability rules,
not named patterns:
- pointwise always fuses into its consumer;
- reduction fuses as a consumer epilogue;
- matmul/conv is a region *root* with a pointwise/reduction epilogue;
- respect the real constraints already encoded per-pattern (single-use of the
  intermediate — no recompute explosion; static shapes; legal layouts).
- **Accept:** the pass reproduces all ~8 hand-written fusions *plus* novel ones
  (e.g. `matmul → bias → gelu → dropout`) the catalog never had; the old
  per-pattern passes are deleted, not added to.

### F2 — MSL kernel synthesis (the keystone)
A code generator: `fused_region` (matmul + pointwise + reduction DAG) → emitted
MSL source.  A tiled matmul/loop template with the region's inner ops **inlined
as the epilogue**, replacing hand-writing.  Stage by family:
- F2a: matmul → pointwise-epilogue (generalize the 7 existing — *one* synthesizer
  covers gelu/rmsnorm/bias/silu/any pointwise chain);
- F2b: reduction epilogues (softmax/rmsnorm *synthesized*, not hand-written);
- F2c: attention (matmul → softmax → matmul) as a synthesized online-softmax
  kernel, retiring the bespoke flash-attn variants. **(Apple-scoped — see the
  2026-07-02 reassessment banner.)** "Retiring the flash-attn variants" means the
  *Apple* MSL catalog only; on the lead backends (ROCm/CUDA) hand-emitted
  flash-attn (MFMA/WMMA, `mma.sync`) stays a first-class **arbiter candidate**
  (F6c), never retired. Even on Apple this is partially deferred — runtime
  attention dispatch is not yet wired (the transposed-K orientation issue in the
  §6 "Deferred" note).
- **Accept:** a synthesized kernel for a region *not* in the catalog passes the
  horizontal-equivalence oracle at rung 8 and beats per-op dispatch in
  `dlop_longtail_core`; the hand-written-kernel count starts *decreasing*.

### F3 — Fusion cost model + scheduling
Profitability: *which* regions to fuse and *how* to schedule them (tile sizes,
threadgroup config, when fusion hurts — e.g. recompute vs reuse, register
pressure).  Plug in the flywheel (measured per-chip latency) + an analytical
roofline; fuse only when the model predicts a win.
- **Accept:** the compiler *declines* a fusion the cost model says is slower
  (register-spilling region), proven by a measured row where fused > unfused
  latency and the compiler chose unfused.

### F4 — Correctness gating wired into codegen
Every synthesized kernel is auto-validated by the horizontal oracle (`fused ≡
unfused` on the same backend) *before* it may execute — a synthesized kernel that
diverges is **rejected**, never silently used.  This is the Sakana/anti-silent-
fallback discipline applied to codegen: synthesis you can't prove, you don't run.
- **Accept:** a deliberately-wrong synthesizer mutation is caught and rejected by
  the gate (the codegen analogue of the reward-hack-rejection test).

### F5 — Autotuning the synthesizer
`magellan`/`alphaevolve`/`autotune_v2` search the synthesis knobs (tile shapes,
fusion boundaries, vectorization, unroll), **gated behind the F3 cost model + F4
oracle** (the existing "perf behind correctness" invariant).  The flywheel corpus
accumulates `(region-shape → best-synthesis)` and distills to an O(1) decision.
- **Accept:** an autotuned synthesized kernel beats the hand-written one it
  replaced on this chip, recorded in the flywheel corpus.

### F6 — The lift: one middle-end, many backends *(rewritten 2026-07-02)*

**This is where specific backends get built**, so it carries the governing rule
from [`COMPILER_THEORY_OF_OPERATION.md`](COMPILER_THEORY_OF_OPERATION.md): the
leads (ROCm/CUDA) set the ceiling; the generic middle-end raises the floor and
must never cap them. F6 is no longer a single "lift + swap emitter" step — it is
the three-tier build-out, and it is **no longer hardware-deferred** (real gfx1151
+ sm_120 silicon exists).

**What F0–F5 give F6:** a proven, arch-agnostic *design* — region discovery (F1),
a cost model (F3), a codegen oracle (F4), and an autotuner (F5). What they do
**not** give is a portable *synthesizer*: F2's output is MSL, welded into
`fusion.py`. So F6 splits into three moves, executed as Workstreams **B/C/D** of
[`COMPILER_REFACTOR_PLAN.md`](COMPILER_REFACTOR_PLAN.md) (B = portable
synthesizer, C = per-arch plugins, D = the measured arbiter that keeps the leads
safe):

- **F6a — make the synthesizer portable (prerequisite).** Split `fusion.py` into
  the arch-agnostic core (`FusedRegion`/`EpilogueOp` semantics, F1 discovery, F4
  oracle) and a `KernelEmitter` plugin (`emit(region, target)`). Apple MSL becomes
  the reference plugin. F1/F3/F4/F5 lift *unchanged*; only F2's emission is
  per-arch. *(This is the corrected form of the original "swap the F2 emitter"
  claim — the swap point has to be built first.)*

- **F6b — per-arch codegen plugins for the fusable middle ground.** Each backend
  supplies a `TargetPlugin` (emitter + shape table + cost model + intrinsic set +
  async model + compile fn). The synthesizer then covers *epilogues, pointwise
  chains, and small attention* on NVIDIA (PTX), ROCm (AMDGCN/ROCDL), and x86
  (C/LLVM) — the same generality it has on Apple. NVIDIA and ROCm supply the
  *deepest* plugins because they own the richest hardware. **Note the oracle must
  grow up here:** F4's Apple proving ground was fp32/f16 bit-close, but the leads'
  frontier is fp8/fp4/MX — so the codegen oracle becomes **accuracy-budgeted**
  (exact for int/fp32-accum, tolerance-aware for low precision), per
  [`COMPILER_THEORY_OF_OPERATION.md`](COMPILER_THEORY_OF_OPERATION.md) §4.1. A
  bit-exact gate would reject every low-precision kernel the leads exist to run.

- **F6c — leave the crown jewels to the leads' hand-emitted paths, arbitrated.**
  GEMM and large-attention on the leads stay hand-emitted (`wgmma`/`mma.sync`/
  tcgen05, MFMA/WMMA — the real `ptx_emit.py` + ROCm `Generate*Kernel` paths) and
  enter the **measured arbiter** as Tier-3 candidates. The synthesizer competes
  only where it *measures* competitive on that silicon. This is the line the
  original F6 lacked, and it is the reason a shared middle-end cannot regress a
  lead.

**Hardware status (superseding the original "deferred"):**
- **ROCm gfx1151 (Strix Halo):** executes a compiler-generated matmul + flash-attn
  family today. F6b/F6c land against real silicon; measured autotune (F5) runs
  live. CDNA (MI300) stays hardware-gated (no box).
- **NVIDIA sm_120 (NR2 Pro):** first `mma.sync` matmul hardware-verified; `ptxas`
  (rung 3) + execute-compare (rung 7) are now reachable. The missing piece is the
  **in-process `--tessera-emit-nvidia` pipeline** (Refactor Plan C2), not a runner.
  sm_90/sm_100 stay gated (separate emit paths, no datacenter box).
- **x86:** the AVX-512 plugin validates natively on the Strix Halo Zen 5 CPU; the
  AMX fast-path stays hardware-gated (no AMX silicon in the fleet).

**Coordination:** F6 work is authored host-free on the Mac (golden-IR gated) and
proven on the box that owns the target — the three-system model in
[`COMPILER_THEORY_OF_OPERATION.md`](COMPILER_THEORY_OF_OPERATION.md) §6 and the
routing matrix in [`COMPILER_REFACTOR_PLAN.md`](COMPILER_REFACTOR_PLAN.md) §7.

## 4. Sequencing + why this order

F0→F1→F2 is the critical path (substrate → discovery → synthesis) and delivers
the headline result: a graph that isn't in the catalog *compiles to a fused
kernel*.  F3/F4 make it safe and smart; F5 makes it self-improving; F6 makes it
portable.  Each phase is independently shippable and leaves `main` green —
the same promotion-ladder discipline (reference → artifact → native, oracle-
gated) used throughout the codebase.

**Anti-goals (scoped 2026-07-02):** on **Apple**, do not add another hand-written
fusion pass or MSL kernel after F2 lands — every new fusion goes through the
synthesizer, or it is a regression in the thing this plan exists to fix. This
anti-goal is about the *default path*, and it is **Apple-scoped**: on the **lead
backends (ROCm/CUDA)**, hand-tuned kernels remain first-class — they enter the
measured arbiter as Tier-3 candidates (Theory §3–4) and the synthesizer displaces
them only when it measures at least as fast on that silicon. "No new hand-written
kernel" never means "delete a faster hand-tuned lead kernel." The invariant is:
*the synthesizer is the default for the fusable middle ground; hand-tuned kernels
are candidates, not the only path — and never silently the slower one.*

## 5. First concrete step
F0 + F1a + F2a as one vertical slice: a `fused_region` op, a `GeneralFusionPass`
that captures `matmul → any-pointwise-epilogue`, and a synthesizer that emits the
MSL for it — validated by the horizontal oracle and measured by
`dlop_longtail_core`.  If that one family generalizes (one synthesizer replacing
gelu/rmsnorm/bias/silu hand-written kernels), the same shape extends to reduction
and attention, and the catalog starts shrinking instead of growing.

## 6. Landed status (Apple proving ground)

All code in `python/tessera/compiler/fusion.py`; guards in
`tests/unit/test_fusion_synthesis.py` (57 tests, Metal-gated rows
hardware-verified on Apple Silicon); runtime wiring in
`runtime._apple_gpu_try_synthesized_fusion`.

| Phase | Status | What landed |
|-------|--------|-------------|
| F0 | ✅ | `FusedRegion` (matmul root + pointwise chain + optional reduction); `AttentionRegion`. |
| F1a | ✅ | `discover_fusable_regions` (matmul→pointwise→reduction, single-use intermediate) + `discover_attention_regions` (matmul→softmax→matmul); the pointwise/reduction discovery is **wired into the runtime hot path**. |
| F2a | ✅ | `synthesize_matmul_epilogue_msl` — one synthesizer for any pointwise-epilogue chain; reproduces hand-written `matmul_gelu` bit-close. |
| F2b | ✅ | reduction epilogues (rmsnorm/softmax) synthesized; reproduces hand-written `matmul_rmsnorm`. |
| F2c | ✅ | `synthesize_attention_msl` + `tessera_apple_gpu_synth_attention_f32` C ABI — fused `O = softmax(scale·Q·Kᵀ)·V` (causal/non-causal), oracle-verified vs numpy attention on Metal. |
| F3 | ✅ | `fusion_cost`/`attention_cost` + `should_fuse_*` — analytical profitability (stack-fit hard gate, dispatch + DRAM-traffic savings); the runtime declines an over-cap fusion so large-N matmul keeps its tiled/MPS path. |
| F4 | ✅ | `verify_synthesized_region` codegen-gated oracle — a synthesized kernel runs only after matching the unfused reference on a probe (cached per region-class). A deliberately-broken synthesizer is **rejected** (anti-cheat test, Darwin). |
| F5 | ✅ | `autotune_matmul_epilogue` — measures synthesis variants (`broadcast`/`dot` matmul schedules) on Metal, **gated behind F3 cost + F4 oracle** (`_pick_best_variant` never picks a fast-but-wrong variant — the Sakana invariant), distilling `(region-class, shape-bucket) → best variant` into an O(1) corpus consumed by the runtime. |
| F2b-tiled | ✅ | `synthesize_matmul_epilogue_msl_tiled` + `tessera_apple_gpu_synth_matmul_epilogue_tiled_f32` C ABI — threadgroup-tiled large-N synthesis (one row/threadgroup, 32 cooperating threads, dynamic threadgroup memory, tree reduce), lifting the fused envelope from N≤1024 to **N≤8192**. Mirrors the proven `matmul_softmax_tiled_f32` structure; `run_fused_region` picks stack→tiled→reference; F3 cost cap grows to the tiled bound. Reproduces hand-written `matmul_softmax_tiled_f32` bit-close. |

### Catalog retirement (landed) — 4 f32 kernels retired

The kernels the synthesizer *retires*, end-to-end across every emission and
dispatch surface (one generic symbol replaces each per-kernel one):

- **matmul_gelu_f32 + matmul_rmsnorm_f32** (pointwise + reduction epilogue).
- **matmul_softmax_f32 + matmul_softmax_tiled_f32** (after F2b-tiled lifted the
  large-N gap that previously kept softmax out of scope).

Each retirement touches: the **Tile→Apple C++ pass** (emits
`@tessera_apple_gpu_synth_matmul_epilogue_f32` + a `tessera.fusion.epilogue`
region descriptor on a uniform `(A,B,O,M,N,K)` signature) + **driver.py** +
**target_ir.py** (embeds the *synthesized* source — single source of truth, the
hand-written MSL-source constants for gelu/rmsnorm deleted) + **runtime/
`_apple_gpu_backend` dispatch** (routes f32 through `run_fused_region`) + the
**public C ABI symbol** deleted from the `.mm` + stub. The *internal* helpers and
the f16/bf16 MSL sources stay — the native f16/bf16 kernels reuse them via fp32
conversion. Lit fixtures + roadmap + target-IR contract + ABI-floor audits
(f32 families 3→2) updated; `runtime_abi` regenerated.

**f16/bf16 retirement (landed):** with half-precision synthesis in place (`half`
I/O + fp32 accumulators; bf16 host-converts), the **8** per-kernel f16/bf16
symbols — `matmul_{gelu,rmsnorm,softmax}_{f16,bf16}` + `matmul_softmax_tiled_{f16,bf16}`
— are deleted across the C ABI (.mm + stub), symbol table, freshness getattrs,
dispatch, the C++ softmax pass (now synth for **all** dtypes), `target_ir.py`, and
the lit/roadmap/audit surfaces. The whole `matmul_{gelu,rmsnorm,softmax}` catalog
family is gone; one synthesized symbol set (`synth_matmul_epilogue{,_tiled,_f16}`)
covers it. **Count-down complete** for the matmul-epilogue catalog.

### F2d — cooperative-matrix synthesis (landed) + v2 (reduction)

The scalar synthesized matmul runs as fp32 FMA in the general ALU (~12 GF/s, ~no
f16 speedup — the matrix units are never touched).  **F2d** emits a
`simdgroup_matrix` matmul (f16 multiply / fp32 accumulate) with the pointwise
epilogue fused after; `run_fused_region` prefers it for pointwise f16/f32 regions.
Measured on M1 Max (matmul→gelu): ~13 → **1289 GF/s** f16 (~98×), **1.76×** f16/f32
(the matrix-unit win the scalar kernel couldn't reach), epilogue still fused.

**v1.1 — tile upgrade (landed):** double-buffering was a wash on Apple (no
`cp.async`).  The real lever is the **64×64 register-blocked tile** (8 simdgroups,
256 threads, `acc[4][2]` = 8 accumulators per simdgroup — the `mtl4_matmul_sg_fast`
structure), selected by shape (`coopmat_tile_for`: 64 for wide+deep matmuls, 32
otherwise).  Measured on M1 Max (matmul→gelu f16): at **2048×1024×512** — the one
shape where MPS-compose previously edged coopmat 1.37× — the 64×64 kernel is
**1.42× over 32×32 (1127 → 1600 GF/s) and now beats MPS-compose 1.18×**.  At
1024×1024×1024: 1808 GF/s (1.15× over MPS).  For small/narrow shapes the 32×32
kernel ties and both crush MPS (1.9× at 512³).  So **coopmat now beats MPS-compose
across the board for pointwise** (1.15–1.96×) — the gap is closed and reversed.

**v2 (landed) — softmax/rmsnorm coopmat reduction:**
`synthesize_matmul_reduction_coopmat_msl` + `…_reduce_coopmat` C ABI: one
threadgroup computes a BM-row block × the full N via simdgroup MMA into
threadgroup memory, then reduces each row in the same kernel.  Oracle-gated
(softmax/rmsnorm, f16/f32, N ≤ 512, N % 8 == 0).  Measured **7.2×** over the
scalar reduction kernel — but ~90 GF/s.  **v2.1** (32-col blocks, 4 accumulators)
confirmed the bottleneck is *not* A-restaging but the structural limit (full row
in one threadgroup → BM=8, one simdgroup, serial row reduce).

**The measurement that settled it (the real perf win):** MPS-matmul + MPSGraph-
reduce (compose) is **4–9.5×** faster than coopmat-reduce and ~55× faster than
the scalar fused kernel — Apple's optimized GEMM crushes both for the matmul.
And the production `@jit` matmul→softmax/rmsnorm at a matmul-heavy shape was on
the *scalar* path (measured **85 ms** at 2048×1024×256).  So:
- **Reductions route to MPS-compose** when the matmul is non-trivial
  (`_APPLE_REDUCE_COMPOSE_MIN_FLOP`, gated in the softmax/rmsnorm dispatchers).
  Production `@jit` measured **85 → 2.75 ms (31×)** and **167 → 3.76 ms (44×)**.
  Tiny matmuls keep the single fused dispatch.
- **Pointwise stays coopmat** — measured competitive (1.8× over compose at 512³,
  ~tie at 256-wide, only 1.37× behind MPS-compose at the very largest), and it
  keeps the epilogue fused.
- `run_fused_region_coopmat_reduce` stays as a proven, measured alternative but
  is *not* the production path (compose wins for matmul-heavy; the scalar fused
  kernel handles sub-gate tiny shapes).

**F2d-v2.2 cooperative-reduction — measured, NOT pursued (structurally dominated):**
The hypothesis was that replacing the v2.1 serial row reduce (each of BM=8 rows
reduced by one thread) with a cooperative multi-simdgroup reduction tree would
let the single fused kernel beat compose.  Two measurements killed it:
- **fused-reduce vs compose** (f16 softmax): compose wins **2.0–6.5×**
  (512³ 4.33→1.21 ms; 2048×1024×256 10.29→1.58 ms).
- **fused-reduce vs pointwise-coopmat on the _identical matmul_**: the reduce
  kernel is **6–12× slower** than the 64×64 pointwise coopmat kernel.
That second ratio is the verdict: the reduction *phase* is already free — the
6–12× is lost in the **matmul**, because fusing the reduction forces BM=8 / one
simdgroup so the full N-wide row stays resident in threadgroup memory to be
reduced (`Cs[BM·MAXN]` ≈ 16 KB at BM=8, MAXN=512; BM=16 would blow the 32 KB
budget).  A cooperative reduction tree optimizes the free phase and **cannot
relax the BM=8 constraint that starves the matmul** — so v2.2 is structurally
dominated by compose.  Production already routes matmul-heavy reductions to
compose; v2.2 is closed, not deferred.  (Reproduce: the inline reduce/pointwise/
compose harness; same shape sweep.)

**Deferred (honest):**
- **F2d v1.1 tile upgrade** — 64×64 register-blocked tiles for v1 pointwise
  (landed; the per-shape 32/64 autotuner now picks the tile).  v2.2 cooperative
  row reduction is **closed above, not deferred** (measured-dominated).
- **Free the dead internal helpers** — the C++ `dispatch_matmul_*_msl{,_f16}` /
  `reference_matmul_*` / `*_via_fp32` helpers + their MSL-source constants are now
  unreferenced (the public symbols that called them are gone). They linger as
  `-Wunused-function` warnings (build still clean — no `-Werror`); a mechanical
  follow-up deletes them.
- **Runtime attention dispatch** — `discover_attention_regions` is a tested pure
  function, but is *not* dispatched from `_apple_gpu_try_synthesized_fusion`: the
  score matmul feeds the *transposed* K (operand shape `(D, Nk)`) while
  `run_fused_attention` expects `(Nk, D)`; resolving that orientation needs
  layout-aware operand info from a Graph-IR pass, not value shapes (ambiguous when
  `D == Nk`).  Wiring it from raw values would risk a silent transpose error.
- **F6 lift** to MLIR/LLVM for NVIDIA/ROCm — hardware-gated.
