# Tessera Optimizing-Compiler Plan — from op-library to world-class

> Status: proposed (2026-06-14).  Companion to `EVALUATOR_PLAN.md` (the scoring
> engine this plan is gated by) and `COMPILER_AUDIT.md` (current state).
> Scope: the execution middle-end.  The `@jit` frontend (Graph IR emission,
> multi-output ops, scalar-attr lowering) is done; this plan is about what
> *consumes* those graphs.

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
  kernel, retiring the bespoke flash-attn variants.
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

### F6 — The lift: one middle-end, many backends
With general fusion + synthesis proven on Apple, lift the **region model + the
synthesizer's IR-level structure** into the MLIR/LLVM production lane
(`tessera_jit`, today a CPU Phase-0 JIT).  The same `fused_region` retargets to
NVIDIA (PTX/WGMMA — currently rung 2.5, PTX *emitted*) and ROCm (MFMA) by
swapping the F2 backend emitter, not the F1 discovery or F3/F4/F5 machinery.
Apple proves the design; MLIR/LLVM scales it.  *(Hardware-gated: NVIDIA/ROCm
execution needs a real CUDA/ROCm runner this Mac can't provide — F6 is design +
emission + assemble-on-CI, with silicon validation deferred.)*

## 4. Sequencing + why this order

F0→F1→F2 is the critical path (substrate → discovery → synthesis) and delivers
the headline result: a graph that isn't in the catalog *compiles to a fused
kernel*.  F3/F4 make it safe and smart; F5 makes it self-improving; F6 makes it
portable.  Each phase is independently shippable and leaves `main` green —
the same promotion-ladder discipline (reference → artifact → native, oracle-
gated) used throughout the codebase.

**Anti-goals:** do not add another hand-written fusion pass or MSL kernel after
F2 lands — every new fusion must go through the synthesizer, or it is a
regression in the thing this plan exists to fix.

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

**Deferred (honest):**
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
