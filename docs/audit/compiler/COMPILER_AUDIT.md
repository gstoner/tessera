# Compiler Audit

This document consolidates the compiler audit material that previously lived in
multiple root audit documents and compiler archive files.

> **Latest deep pass:** [DEEP_COMPILER_AUDIT_2026_06_10.md](DEEP_COMPILER_AUDIT_2026_06_10.md)
> — source-backed audit of frontend/IR/manifest/runtime-ABI/Apple-envelope/
> benchmark coverage. Records the "generated drift clean vs semantic gap open"
> split, fixes the bench-axis staleness + the grouped_gemm/moe_swiglu_block
> manifest blind spot, and carries a prioritized gap table for the rest.
>
> **Code-level companion:** [CODE_AUDIT_2026_06_10.md](CODE_AUDIT_2026_06_10.md)
> — refactoring / per-IR-level optimization correctness / glass jaws. Headline:
> a verified `TransposeIntoMatmul` flag-composition miscompile (fixed, commit
> `acb5c6f`), missing fusion use-guards (fixed), NSA gating-semantics hazard
> (guarded), silent autodiff chain breaks (diagnosed), no upstream
> canonicalizer/CSE in named pipelines (fixed), `TESSERA_STRICT_DISPATCH`
> against silent numpy fallbacks, and runtime consumption of `fusion_groups`.
> Two earlier agent claims refuted.
>
> **Evaluator program — substantially shipped (2026-06-12):** [EVALUATOR_PLAN.md](EVALUATOR_PLAN.md)
> (see its §9.5 "what has landed"). A generative, execution-derived,
> **backend-rung-aware** Evaluator that *derives* conformance/benchmark/autotune
> surfaces from one honest scoring engine (closing the "registry models reality"
> gap). **Landed:** the 8-rung verdict engine + provenance gate
> (`evaluator.py`); four oracles — vertical, horizontal/PolyJuice, metamorphic,
> and DESIL cross-path (`cross_path_equivalence`); conformance corroboration
> (`conformance_evaluator.py`); the autotuning flywheel + per-chip calibration +
> autotune_v2 bridge (`flywheel.py`, `flywheel_autotune.py`); and the scored
> environment — a TensorBench-style grader (`compiler_grader.py`), LongCA
> structure-keyed attention (`attention_tasks.py`), and Magellan/AlphaEvolve
> gated search (`magellan.py`, `alphaevolve.py`, with reward-hack rejection
> proven). **Open (hardware-gated):** NVIDIA/ROCm sit truthfully at rung 1–2.5
> (WGMMA PTX *emitted* via `ptx_emit.py`; rung-3 `ptxas` + complete kernel +
> silicon need a Linux/CUDA runner). Research-backed
> (DESIL/PolyJuice/Mirage/TensorBench/Magellan/AlphaEvolve/BaCO/TLP).

## Library → Optimizing Compiler: front-to-back closure plan (2026-06-15)

A front-to-back audit of every IR level framed by one question: *where is Tessera
still a library/dispatcher, and what does each layer need to become an optimizing
compiler?* This section is the strategic spine; the per-item status lives in
**Still Open** / **Next Work** below (cross-referenced, not duplicated).

### The central finding — two disconnected worlds, one half-closed seam

The executed path and the C++ MLIR optimizer are largely **disconnected**:

- `tessera-opt`'s optimized IR is run for **validation only** — its stdout is
  hashed for provenance and discarded (`driver.py` `_try_validate_with_tessera_opt`,
  ~`:955/:970`). Execution dispatches off the **in-memory Python `GraphIRModule`**
  (`jit.py` `recognize(self.graph_ir)`), so the C++ fusion/canonicalize/CSE passes
  do not reach runtime.
- Consequence: fusion logic exists **twice** — real C++ rewrites the executor
  ignores (`SwigluFusion`/`MLAFusion`) *and* the Python path's own derivation.
  The seam is **half-closed already**: `canonical_compile._derive_fusion_groups`
  carries `fusion_groups`, the executor reads them (`runtime.py` ~`:2343`), and
  `stamp_fusion_intents` stamps `tessera.fusion.intent` for 4 chains (see the
  "Fusion intent is too late" item). But it is **advisory** — every dispatch
  branch is still `if fused_kernel=="X" OR _structural_rematcher(ops)`. Closing
  the seam = promote advisory → **authoritative** and delete the re-matchers.

Two facts make the transition cheaper than it looks:

1. **The seam mechanism half-exists** (above) — Phase 0 finishes it, it doesn't
   invent it.
2. **The real K-reduction GEMM loop already exists and executes** — in the
   `tessera_jit`/linalg lane (`tessera.matmul → linalg.matmul → linalg-to-loops`,
   `TesseraToLinalgPass.cpp` ~`:369`). The Tile-IR hand-rolled nest (`TilingPass.cpp`)
   stopped at M/N blocking with no K loop. **Converge the lanes** rather than
   finish the hand-roll. This is the template: drive Schedule/Tile IR through the
   already-executing linalg→loops spine; reserve hand-emitted Tile IR for the
   GPU-only ops (wgmma/async/mbarrier) linalg can't express.

### Per-IR scorecard (what's real vs. dispatcher)

| IR level | Real today | Dispatcher / stub | Primary gap to close |
|---|---|---|---|
| **Python `@jit`** | Decoration-time constraint + effect analysis; honest fallback gating (won't let eager Python masquerade as compiled). | Effect/constraint analysis is single-function, AST-only. No IR-optimization step between emission and execution. | Component-aware multi-op metadata (mostly landed); carry fusion/strategy as authoritative. |
| **Graph IR** | 132 ops, 107 real verifiers; 5 canon patterns; real fusion passes (SwiGLU/MLA/NSA). 101/109 ops are `[Pure]` (CSE/DCE-eligible *today*). | **Zero `hasFolder`/`hasCanonicalizer`** on any op. No constant folding / general DCE-CSE on the executed path. `LayoutLegalityPass` is verify-only (no assignment). ~5 passes are attribute-stamp-only. | Folders/canonicalizers; effect interfaces on the 8 non-pure ops; `LayoutAssignmentPass`. |
| **Schedule IR** | DistributionLowering (real structural wiring + escaping-value fix); collective *insertion*. | Pipeline insertion is annotation-only (no real 1F1B order). OptimizerShard is pure attrs. No collective overlap (`ChunkPlanner`/`CollectiveScheduler` exist but are never invoked). | Real 1F1B ordering; wire the collective planners. |
| **Tile IR (FA-4)** | `TilingPass` emits real `scf.for` M/N nests; `NVTMADescriptorPass` is a genuine hoist/dedup. | M/N blocking only — **no K-reduction loop**, fixed 16×16. **Flash-attn is straight-line** (no `scf.for` over KV; online-softmax is whole-tensor ops). WarpSpec emits no queues/mbarriers. WGMMA → hardcoded `m64n64k16`. | Converge GEMM to linalg K-loop; streaming flash-attn loop; autotuner→IR. |
| **Autotuner** | `BayesianAutotuner` tunes `{tile_m/n/k, num_warps, num_stages}`. | Scores via `_mock_latency` (roofline); `on_device` returns "unmeasured". **Output reaches no lowering pass** (read path exists at `TileIRLoweringPass.cpp` ~`:111`; write path absent). `flywheel` measurement lane not in compile path. | Write-path (stamp tile attrs); measured-latency scoring on Apple/CPU. |
| **Target IR / runtime** | x86 AMX real end-to-end. Apple GPU executes via MPS/MPSGraph + ~30 hand-written fused MSL. `fusion.py` synthesizer is **real runtime MSL codegen** for matmul-epilogue regions. `tessera_jit` is a real MLIR→LLVM CPU JIT (~17 op classes). | Apple lane is a name→lane→ctypes **dispatcher** (`apple_gpu_envelope.py`; longest-chain cascade in `runtime.py` ~`:2471`). ~87% of ops execute via the numpy reference interpreter. NVIDIA/ROCm emit artifacts only. | Close the seam; generalize the synthesizer; grow `tessera_jit` to default CPU, then retarget its spine to GPU. |

### The phased plan

`HF` = hardware-free (lands on this Mac); `HG` = hardware-gated. Phase 0 is the
keystone; once it lands, Phases 1/2/4 largely parallelize. Everything through
Phase 4 is HF; only GPU launch + silicon-perf is gated.

- **Phase 0 — Close the seam (keystone, HF).** Finish the half-built carry-intent
  mechanism. **(a) Landed (2026-06-15)** — each `known_chain` fusion group now
  carries a `dispatch` roles sub-dict (`a`/`b`/`c`/`x`/`wg`/`wu`/`wd`/`out` +
  scalar `eps`), resolved once from Graph-IR operand order in
  `canonical_compile._chain_dispatch_roles` — killing the value-shape guessing the
  re-matchers do inline. Strictly additive (a group carries no `dispatch` when
  roles don't resolve, so the executor path is unchanged); JSON round-trips into
  `fn.runtime_artifact().metadata`. Guard: `tests/unit/test_fusion_dispatch_roles.py`
  (5). **(b) Landed (2026-06-15)** — `_execute_apple_gpu_mps_metadata` now resolves
  a whole-program authoritative plan (`_apple_gpu_resolve_authoritative_plan`,
  reading both `fusion_groups` and the `canonical_fusion_groups` the `@jit`
  artifact actually stamps) and dispatches off the carried roles via
  `_APPLE_GPU_FUSION_DISPATCH` — no per-invoke re-matching, no value-shape
  guessing. Falls through to the structural cascade only when roles don't resolve
  (legacy safety). This surfaced a latent gap: the executor read a bare
  `fusion_groups` key the real artifact never set (it sets
  `canonical_fusion_groups`), so the re-matchers — not the carried intent — were
  what actually fired in production; the authoritative path closes that.
  **(c) Landed (2026-06-15)** — proved authoritative ≡ re-matcher (horizontal
  oracle, `tests/unit/test_fusion_authoritative_dispatch.py`, 12) then **deleted**
  the four `_apple_gpu_metadata_is_*_chain` re-matchers. Closed the one subsumption
  gap first (`matmul→rmsnorm_safe` is now a known_chain so authoritative dispatch
  covers it). The `fused_kernel == "X"` branches remain only for bare-`fusion_groups`
  metadata (hand-built / pre-`dispatch` legacy); truly-legacy no-metadata artifacts
  now run correctly per-op instead of via a re-discovered fuse. Full apple_gpu +
  canonical + fusion sweep: 2255 passed / 0 failed. **Seam closed — one fusion
  recognizer (the compiler), carried across to the executor.** Extends **Still
  Open → "Fusion intent is too late"** and **Next Work #3**.
- **Phase 1 — Make carried IR worth carrying (Graph-IR quality, HF, parallel).**
  **First increment landed (2026-06-15), observable end-to-end on the executed CPU
  JIT lane.** The tessera_jit pipeline had **no canonicalizer**; added
  `createCanonicalizerPass()` + `createCSEPass()` to `pm1` *before* `TesseraToLinalg`
  (`tools/tessera-jit/tessera_jit.cpp`), so Tessera per-op folders now bite on the
  executed path. Shipped the first two folders: `TransposeOp::getCanonicalizationPatterns`
  (`transpose(transpose(x)) → x`, a no-perm transpose is its own inverse) and
  `CastOp::fold` (identity `cast(x): T→T → x`, only when no `numeric_policy`), via
  `hasCanonicalizer`/`hasFolder` in `TesseraOps.td` + bodies in `TesseraOps.cpp`.
  Proven end-to-end: `@jit(target="cpu")` `transpose(transpose(x))` folds to
  `return %arg0` in the JIT trace and returns `x` exactly. Also registered the
  upstream `canonicalize`/`cse` passes in `tessera-opt` (`registerTransformsPasses`)
  so folders are lit-inspectable. Guards: lit `tests/tessera-ir/phase2/graph_ir_folders.mlir`
  (folders + negative cases + **DCE** of a dead pure op + **CSE** of duplicate
  matmuls — the shared-QKV-projection pattern) + `tests/unit/test_native_cpu_jit.py`.
  **CSE + DCE verified firing end-to-end on the executed CPU JIT path** (duplicate
  `matmul` → 1; dead `gelu` → eliminated; confirmed in the JIT trace) — these, not
  the rare algebraic folds, are the high-value Phase 1 wins, and they are now live.
  **Identity folders landed (2026-06-16):** `AddOp`/`SubOp`/`MulOp`/`DivOp` now
  have `hasFolder` + `fold()` bodies in `TesseraOps.cpp` — `x+0`/`0+x`/`x-0`/`x*1`/
  `1*x`/`x/1` collapse to the surviving operand when the other is a constant splat
  of the scalar identity (type-equality-guarded; no-signed-zeros, matching the
  fast-math GEMM model). Guard: 7 new cases (folds + negatives) in
  `tests/tessera-ir/phase2/graph_ir_folders.mlir` (18 total FileCheck'd). `matmul·I`
  deferred (needs identity-matrix recognition; never appears in real graphs).
  **Effect-interface item assessed + closed:** the genuinely non-pure ops
  (`dropout`=random, the `all_reduce`/`reduce_scatter`/`all_gather` collectives,
  `kv_cache_*` writes, the `adam`/`adamw`/`momentum`/… optimizer in-place updates)
  are **already conservatively sound** under MLIR's unknown-effects model (no `Pure`
  ⇒ never CSE'd, never DCE'd); the FFT/Clifford families that *look* non-pure
  actually inherit `[Pure]` from their base classes. Adding explicit
  `MemoryEffectsOpInterface` yields no practical CSE/DCE win (writes neither CSE
  nor DCE in MLIR's model) and risks subtle reordering bugs — so the current
  treatment is the right one. *Remaining Phase 1 (lower priority):* migrate the 5
  `CanonicalizeTesseraIR` patterns to per-op hooks (only those whose output the CPU
  JIT can lower — `fused_epilogue`/`conv` ones would break it), `LayoutAssignmentPass`
  v1 (layout doesn't reach the layout-agnostic rank-2 CPU JIT lane — low value until
  a layout-sensitive backend executes).
  *Flash-attn streaming is NOT a CPU-lane item* — the CPU JIT is a rank-2 simple-op
  lane; flash-attn (rank-4, batched) belongs to the Apple GPU work, where the
  streaming online-softmax kernel already exists as hand-written MSL
  (`kFlashAttnF32Source`). Extends **Still Open → "Layout and binding contracts"**.
- **Phase 2 — Real codegen in the executed path (linalg spine, HF).** **GEMM-lane
  convergence achieved + proven on the executed CPU JIT lane (2026-06-15).** Phase 4
  built the CPU JIT matmul on `linalg.matmul`, so the executed GEMM already lowers
  `tessera.matmul → linalg.fill + linalg.matmul → ConvertLinalgToLoops` into a real
  **M×N×K** loop nest with the K-reduction inner loop (`scf.for` over K + `mulf`/`addf`
  accumulate) — verified in the JIT trace and guarded by an exactly-representable
  GEMM equivalence test. The hand-rolled Tile-IR `TilingPass` (M/N blocking, no
  K-loop) is the separate **x86/validation** lane, not the executed path; converging
  *it* to linalg is the remaining, lower-priority item. *Remaining Phase 2:*
  flash-attn streaming (wrap the attn ops in `scf.for` over KV with `(m,l,acc)`
  iter_args — `OnlineSoftmaxOp` ODS is already iter_args-shaped; only the loop
  wrapper + `kv_offset` threading is missing); generalize the `fusion.py` synthesizer
  (`elementwise_only` + `norm_chain` region kinds; grow `EPILOGUE_OPS`) and displace
  the numpy interpreter lane-by-lane, **elementwise first**, Evaluator-gated — never
  displacing a working MPSGraph call.
- **Phase 3 — Close the optimizing loop (HF on Apple/CPU). ✅ landed (2026-06-16).**
  The synthesizer had a measured-latency, correctness-gated variant autotuner
  (`autotune_matmul_epilogue` — times each MSL variant on-device, gates each
  against the numpy reference, populates `_AUTOTUNE_CORPUS`) that was never
  auto-invoked, so `best_variant_for` always returned the static default. Closed
  the loop in `fusion.py`: `autotune_enabled()` (reads `TESSERA_AUTOTUNE`) +
  `select_variant(region, M, N, K, *, autotune=None)` — on a corpus miss with
  autotune on it measures + caches the measured-best variant, else it's an O(1)
  lookup. Wired into `runtime.py::_apple_gpu_try_synthesized_fusion` (replacing
  `best_variant_for`), so the executed Apple GPU lane runs the measured-best
  kernel. Latency is real (synthesizer dispatch timing); the roofline mock stays
  the honest NVIDIA/ROCm fallback. Guard: `tests/unit/test_autotune_loop.py` (5).
  *Deferred:* the `apply_to_op()` tile-attribute write-path for the Schedule/Tile
  IR autotuner (NVIDIA/tile-IR configs have no executable Apple kernel — HG).
- **Phase 4 — Grow `tessera_jit` toward default CPU (HF), then GPU spine.**
  **Brought forward (2026-06-15) — the keystone landed: the tessera_jit MLIR→LLVM
  lane is now the executed CPU path** for the covered f32 op set, so the C++ IR
  optimizations finally reach execution (closing the remaining seam for the CPU
  lane). `@jit(target="cpu")` now translates the executed `GraphIRModule` op-list
  into a whole-graph `GraphFn` (`_jit_boundary.run_graph_ops`) and runs it through
  `tessera_jit` (`tessera-to-linalg → one-shot-bufferize → linalg-to-loops → LLVM`,
  optLevel=2) **before** the numpy reference interpreter (`JitFn._try_tessera_jit_call`,
  tried in the CPU `__call__` branch). Covered set = `_JIT_GRAPH_OPS` (matmul,
  add/sub/mul/div, relu/sigmoid/tanh/silu/gelu, softmax, rmsnorm, layer_norm,
  transpose, select, masked_fill); anything else / non-f32 / unsupported rank falls
  back to numpy (correctness preserved — a fallback handles "couldn't run", never
  "ran wrong"). `TESSERA_DISABLE_CPU_JIT` is the kill-switch. Proof-of-execution via
  `_jit_boundary.invocation_count` (a silent numpy fallback can't masquerade).
  Guard: `tests/unit/test_native_cpu_jit.py` (per-op numpy equivalence + counter +
  fallbacks); 1929-test CPU/jit/ops sweep green. **Dtype breadth landed
  (2026-06-15), grounded in Apple M1 Max hardware:** the lane now routes **f32**,
  **f16** (native NEON, ARMv8.2-A FP16), and **bf16** (correct but emulated via f32
  in-kernel — M1 predates ARMv8.6 BFloat16), per-arg dtype detection in
  `_try_tessera_jit_call` (mixed dtypes → numpy fallback). matmul/reductions
  accumulate in f32 then truncate to storage (`TesseraToLinalgPass`, ABI §12.5 —
  already in the C++). Required adding `f16` to the `_jit_boundary` C-ABI dtype
  table (raw 16-bit at the boundary, like bf16) and making `_jit_unary` elem-aware
  (was f32-only while `_jit_binary` already used `_resolve_elem`). **f64 wired into
  the lane (2026-06-16)** — three contained table entries (`_elem_for` in `jit.py`,
  `_DTYPE_TABLE` + `_ELEM_TO_NP` in `_jit_boundary.py`; the C++ `TesseraToLinalgPass`
  and the whole tessera_jit LLVM pipeline were already f64-clean — `isa<FloatType>`
  includes f64 and the low-precision-→f32 accumulate rule never fires, so f64
  accumulates in f64 throughout). This is the **exact-precision lane** for
  gradient-checking / numerical validation (verified ~1.8e-15 GEMM error vs f32's
  ~1e-6). A lone rank-2 f64 GEMM still takes the numpy reference (the Accelerate
  `native_cpu` fast path is f32-only and numpy f64 matmul is already exact f64);
  multi-op f64 programs route through real f64 codegen. Guards:
  `test_f64_runs_through_jit_at_exact_precision` + `test_f64_gemm_is_exact_over_k`.
  **matmul perf — measured + diagnosed (2026-06-16).** The tessera_jit
  `linalg→loops→LLVM` GEMM runs at **~2.2 GFLOP/s** (256³/512³), **~50–110× off**
  numpy/Accelerate's 100–240 GFLOP/s — the `ConvertLinalgToLoops` body is naive
  scalar, un-tiled. Two cheap optimizer levers were tried and **measured
  insufficient**: (a) a host-detected `TargetMachine` into the transformer (was
  `nullptr` → no NEON cost model) + `-O3`; (b) stamping `fastmath<fast>` on the
  float arith ops after linalg→loops (a float reduction won't auto-vectorize
  without `reassoc`). Neither moved the GEMM (LLVM's loop vectorizer won't crack
  the reduction from this IR shape). **Both changes are kept** — they're correct
  (target-aware codegen; `fast` matches Tessera's documented fast-math GEMM
  contract) and prerequisites for vectorization — but the **real lever is an MLIR
  `linalg→vector` tiling+vectorization pipeline** (register-tile the matmul →
  `linalg::vectorize` → `vector→LLVM`), a focused multi-step effort.
  **`linalg→vector` — two attempts (2026-06-16), root cause narrowed, reverted.**
  Built the full gated lane (`TESSERA_JIT_VECTORIZE`): tile each `linalg.matmul`
  on tensors (pre-bufferize) so vectorize emits a `vector.contract` with a
  **register** (not memory) accumulator → `linalg::vectorize` →
  `populateVectorContractLoweringPatterns` (OuterProduct) + transfer lowering →
  `ConvertVectorToLLVM`; all libs wired, compiled + linked clean. **Diagnosis
  (this attempt went much deeper):** (1) **MLIR-22 tiling is NOT broken** —
  `scf::tileUsingSCF` tiles the *identical* `linalg.matmul<64x64>` perfectly via
  `mlir-opt --transform-interpreter` (`transform.structured.tile_using_for
  tile_sizes [8,16,16]` → clean `scf.for` nest + `extract_slice` + inner matmul
  with a tensor iter_arg accumulator). (2) The JIT's *direct* `scf::tileUsingSCF`
  call null-derefs (`EXC_BAD_ACCESS address=0x2c`, inside `tileUsingSCF`). (3)
  Added the registrations the transform interpreter implies —
  `linalg::registerTilingInterfaceExternalModels`,
  `tensor::registerTilingInterfaceExternalModels`, the `affine` dialect,
  `arith`/`tensor` `registerValueBoundsOpInterfaceExternalModels` — **necessary
  but NOT sufficient**: a clean build with all of them still null-derefs. So the
  blocker is the **call mechanism**, not registrations: the transform
  interpreter's wrapper (its `TransformRewriter` + state) does setup the bare
  `IRRewriter` direct call lacks. **Concrete next step (clear, lower-risk):**
  drive the tiling via the **transform interpreter** (the proven path) instead of
  the direct C++ API — parse the `transform.structured.tile_using_for` sequence +
  `transform.structured.vectorize`, run `transform::applyTransforms`. (Register
  the transform dialect + linalg/transform extensions; mind the archive-vs-target
  link rule in `CMakeLists.txt` to avoid the MLIR-aggregate-dylib coexistence
  segfault.) Reverted the direct-call experiment (a segfaulting lane shouldn't
  ship even gated). Scope honesty:
  even done well it won't match hand-tuned Accelerate BLAS, and the **single-GEMM
  hot path already routes to Accelerate** (`_native_cpu_fast_call`); this affects
  multi-op programs that contain a GEMM. AMX is only reachable via Accelerate
  (BLAS/BNNS), so the AMX fast path stays the apple_cpu lane. *Next on this thread:*
  the `linalg→vector` GEMM pipeline (tiling-crash isolation first) → then swap the
  pipeline bottom `linalg→loops→LLVM` for `linalg→gpu→NVVM/ROCDL` (**emission HF**,
  the `tsrRegisterGpuLauncher` → `cuLaunchKernel`/`hipLaunchKernel` wiring HG).
- **Phase 5 — Schedule + pipelining (mixed).** Double-buffering structure (HF) /
  async overlap (HG); real 1F1B ordering (HF); collective↔compute overlap via the
  unused `ChunkPlanner`/`CollectiveScheduler` (plan HF, measurement HG); GPU MMA
  register accumulator (HG).

**Dependency spine:** Phase 0 is the keystone and is small because the mechanism
half-exists. Phases 1, 2, 4 parallelize after it. Through Phase 4 is entirely
hardware-free on this Mac; only the GPU launch + silicon-perf items are gated.
Detailed per-layer evidence (file:line) was captured in the 2026-06-15 deep-dive
agents and feeds the Still Open / Next Work items below.

**Status (2026-06-15):** Phase 0 (seam) **closed**; C++ Target IR consume-side
**reviewed + parity-guarded**; **Phase 4's keystone brought forward and landed** —
the tessera_jit MLIR→LLVM lane is the executed CPU path for the covered f32 op set.
This re-prioritizes Phases 1–2: with the C++ codegen lane now *executing*, the C++
Graph-IR folders/canonicalizers + effect interfaces (Phase 1) and the linalg
GEMM-convergence + flash-attn streaming (Phase 2) now **reach execution** through
the CPU JIT lane rather than only the discarded validation pipeline — so they are
worth building next, with measurable end-to-end impact. The immediate Phase-4
follow-ons (dtype breadth, wider `TesseraToLinalgPass` coverage, tiling before
LLVM) compound directly on this lane.

## Autodiff v1 tape — gaps closed (2026-06-13)

Surfaced while building the agent-native MoE training stack (`tessera.train`,
GRPO post-training). The `CODE_AUDIT_2026_06_10.md` already *diagnosed* "silent
autodiff chain breaks"; this pass found the **root cause and fixed it**, plus two
adjacent ergonomic gaps. All additive; full `tests/unit -m "not slow"` green.

- **Scalar/0-d tape-link break (root cause, fixed).** `autodiff/tape.py::_describe`
  keyed `np.generic` (scalar) inputs on `id(np.asarray(arg))` — a *fresh* array —
  while producers record outputs by `id(output)`. Any reduction-to-scalar feeding
  a later op (i.e. essentially every loss expression: `mul(reduce(...), k)`,
  `exp(reduce(...))`) silently severed the gradient chain (grad came back `None`).
  Fix: key on `id(arg)`. This is the concrete mechanism behind the previously
  "diagnosed" silent breaks.
- **`reduce(op=...)` was sum-only (fixed).** The op advertised an `op=` parameter
  but raised for anything but `"sum"` in both forward and VJP. Added `"mean"`
  (forward + `vjp_reduce`, axis/keepdims-correct). Max/min still route to
  `ops.amax`/`ops.amin` by design.
- **`clip` bound aliases (fixed).** `ops.clip` accepted only `min_val`/`max_val`;
  added `min`/`max` aliases coalesced in **both** the forward and `vjp_clip`, so
  PPO-style clipping is one tape-safe call (its bounds ride in kwargs, avoiding
  the scalar-operand detach below).

### Follow-on closures (2026-06-14): F1, F2, G1, G2

- **F1 + F2 (fixed — shared root cause).** `_make_wrapper`/`_describe` dropped
  *python-scalar* positional operands from the tape, so `ops.minimum(t, 1.2)`
  raised in backward (VJP missing `y`) and `ops.mul(scalar_tensor, -3.0)`
  returned grad as if the factor were `1`. Fix: `_describe` records python
  `int`/`float` (not `bool`) as **non-differentiable literal inputs**
  (`InputDesc.is_literal=True`), and `Tape.backward` tolerates a VJP that omits
  cotangents for *trailing literal* operands (pads `None`) — preserving the
  strict per-array count check that catches genuine VJP-author bugs. Verified
  safe across the full `tests/unit -m "not slow"` suite.
- **G1 (clarified + closed).** The earlier "no tape-traceable gather" claim was
  wrong: `ops.gather` already exists, is tape-wrapped, and scatter-adds
  correctly. The real gap was that `nn.Embedding`/models used raw numpy
  indexing. Added `ops.embedding(table, ids)` (gather + scatter-add VJP) and a
  fully-traceable LM proving the embedding table trains.
- **G2 (new op).** `ops.top_k` had no VJP for routing. Added
  `ops.top_k_routing(logits, *, k)` → full-width sparse-softmax gate (zero off
  the top-k) with a VJP that routes gradient to the selected logits via the
  sparse softmax jacobian (numerically verified vs central difference). This is
  the missing primitive for a **differentiable hard top-k MoE** — proven
  end-to-end in `tessera.train.models.TracedHardMoELM` (embedding + router +
  experts + head all train via `adamw_step`).

Guards: `tests/unit/test_autodiff_tape_fixes.py` (E1/E2/E3/F1/F2),
`tests/unit/test_train_hard_moe.py` (G1/G2). New ops registered as numpy
references (no OP_SPECS requirement — 11 registry-only refs already exist).

### Compute-sparse MoE dispatch (2026-06-14)

The differentiable hard-MoE above used a *dense soft-combine* (every expert
evaluated on every token, off-top-k contributions zeroed). Closed the deferred
follow-on: `tessera.train.engine.moe.sparse_moe_dispatch` does **real per-expert
routing** — each expert runs only on its routed tokens via `ops.gather` →
expert FFN → `ops.scatter_add`. Expert work drops from O(N·E) to O(N·k) while
the result is *numerically identical* to the dense combine (proven, atol=1e-5),
and the whole path stays tape-traceable (gradients reach embedding, router, and
experts; the data-dependent token-index sets are the non-differentiable runtime
part). Exposed as `TracedHardMoELM.logits(ids, dispatch="sparse")`. Guards in
`tests/unit/test_train_hard_moe.py` (parity, grad-flow, end-to-end training).

## Finished

- **Canonical driver:** `canonical_compile` and `CompileResult` are the common
  contract for compilation results.
- **Runtime handoff:** `@jit` and `runtime.launch()` consume canonical compile
  metadata rather than inventing a second path.
- **Capability gates:** legality, codegen, toolchain, link, runtime ABI,
  hardware smoke, and numerical gates report named failure axes.
- **Conformance matrix:** op-target proof is rendered in
  `../op_target_conformance.md` and drift-gated.
- **Schedule to Tile metadata:** mesh, layout, placement, artifacts, and related
  metadata survive lowering.
- **C++ pass honesty:** `LowerScheduleToTargetPass` stopped pretending to be an
  implemented lowering pass.
- **Tile to Apple parity:** C++ Apple status tags match the Python/runtime Apple
  envelope.
- **Dynamic control flow:** unsupported dynamic control flow now gets explicit
  diagnostics and fallback behavior.
- **Frontend bugs:** AugAssign sub/div lowering, ROCm sub-arch gates, and
  Darwin arm64 platform checks were fixed.
- **Compiler correctness tests:** pass-order and oracle fixtures cover string
  parsing, MLIR pass order, halo execution, CorrDiff IR visibility, spectral,
  linear attention, and Apple runtime pipeline order.
- **CSV-canonical generated dashboards + sprint regen (2026-06-04).**
  `runtime_abi` and `verifier_coverage` now emit a machine-readable CSV
  (`docs/audit/generated/*.csv`, stable-sorted, byte-diffable) as the
  drift-gated artifact, with the human Markdown demoted to a non-byte-gated
  companion. Both are wired into `scripts/check_generated_docs.sh`, which gained
  a `--write` mode so `scripts/check_generated_docs.sh --write` regenerates
  every registered doc at sprint end. This retired the byte-exact-markdown
  drift gates that had been reddening CI (`runtime_abi.md` was stale 234 vs 241
  symbols). The four Apple CPU+GPU state docs were also consolidated into the
  single reference `docs/apple_backend.md`.

## Still Open

- **Program identity — component-op vectors + gating landed (2026-06-02);
  component-aware metadata landed (2026-06-07).** `CompileResult` carries
  ``component_ops`` (the whole-program distinct op vocabulary),
  ``program_executable`` (gated component-by-component, not just the primary
  op), and ``component_blockers`` ((op, failing-gate) pairs). **`effects` /
  `shape_envelope` / `layout_contracts` / `fusion_groups` now reach the
  user-facing `fn.runtime_artifact().metadata`** — derived in
  `canonical_compile._derive_*`, factored into
  `CompileResult.descriptive_metadata()`, and merged additively through
  `JitFn._build_runtime_artifact` (previously discarded on the `@jit` path —
  every key was absent for real jitted functions). `fusion_groups` recognizes
  the cross-family chains the Apple GPU runtime actually fuses
  (`matmul→softmax[→matmul]`, `matmul→gelu`, `matmul→rmsnorm`), not just
  same-family adjacency. Locked by `tests/unit/test_canonical_component_ops.py`
  + `tests/unit/test_canonical_metadata_jit.py`. **Graph outputs landed
  2026-06-11** (`canonical_outputs` + populated `return_values`/`result_types`;
  see Next Work #1). **Runtime consumption of `fusion_groups` landed
  2026-06-10** (see next item).
- **Fusion intent is too late — runtime half closed (2026-06-10).** The
  apple_gpu executor now consults `fusion_groups` known_chain metadata before
  the structural re-matchers (which remain as legacy-artifact fallback);
  locked by `tests/unit/test_strict_dispatch.py` (short-circuit + legacy-path
  tests). **SwiGLU is now derived too** (`_match_swiglu_at` handles the DAG —
  gate/up share %x — inside the known-chain scan) and consumed by the
  executor. **Target IR descriptor consume/emit — landed 2026-06-11.** All 7
  Apple fusion passes now *emit* a first-class fusion descriptor on the fused
  call (`tessera.fusion.kernel` + `tessera.fusion.source`): the 4 chain passes
  (matmul→softmax→matmul / matmul→softmax / matmul→gelu / matmul→rmsnorm) also
  *consume* an upstream `tessera.fusion.intent` (source `"descriptor"` vs
  `"rediscovered"`, with a Decision-#21 warning on descriptor/IR disagreement);
  the 3 composite passes (swiglu / mla_decode / native_sparse_attn) emit
  `source = "composite_op"` (the pre-fused op *is* the descriptor). The Python
  emit-half `canonical_compile.stamp_fusion_intents(module)` stamps the intent on
  each recognized chain's terminal op from the canonical `_KNOWN_FUSION_CHAINS`,
  so the frontend produces descriptor-annotated IR. Lit:
  `tests/tessera-ir/phase8/apple_gpu_fusion_descriptor.mlir`; Python:
  `tests/unit/test_apple_fusion_descriptor.py` + `test_fusion_intent_emitter.py`
  (incl. an emit↔consume contract guard). **Auto-wired 2026-06-11:**
  `driver.compile_graph_module` calls `stamp_fusion_intents(module)` before
  rendering the Graph IR for Apple targets (gated to `apple_gpu`/`apple_cpu`;
  the descriptor is backend-agnostic so it extends when other backends consume
  it), so every Apple compile now produces descriptor-annotated Graph IR that
  the Target IR passes consume. The intent is stamped into the op's MLIR `attrs`
  (not `kwargs`, which are the op's real call arguments in the reference/runtime
  path). Loop closed end-to-end.
- **Layout and binding contracts are uneven.** Graph/Schedule/Tile/Target IR
  need stronger dtype, layout, aliasing, and buffer-binding contracts.
  **Layout slice extended 2026-06-11:** `LayoutLegalityPass` was matmul-only; its
  producer/consumer accept-set rule now also covers `tessera.conv2d_nhwc` (nhwc
  on the data operand #0; the filter is a separate weight layout) and
  `tessera.flash_attn` (bhsd on Q/K/V #0..2), per-operand-scoped so it only
  checks the operands that carry each contract. matmul stays verbatim (the V4a
  diagnostic + `matmulAcceptSet()` are pinned by existing tests). Lit:
  `tests/tessera-ir/phase2/layout_conv_flashattn_accept_set.mlir`; Python:
  `tests/unit/test_layout_legality_extended.py`. Still open: dtype / aliasing /
  buffer-binding contracts, and wiring `LayoutLegalityPass` into the named
  pipelines (it's still registered standalone). **Phase 1** of the closure plan
  adds the missing *assignment* half — `LayoutAssignmentPass` (seed kernel layouts
  → propagate through pointwise → insert `cast{layout}`), with the legality pass
  reused as its verifier. Phase 1 also covers the Graph-IR `hasFolder`/
  `hasCanonicalizer` gap (zero today) and effect interfaces on the 8 non-pure ops
  to make generic CSE/DCE sound.
- **Complete claims need fixtures.** A completed backend claim should resolve to
  an explicit compare fixture, `hardware_verified` row, or packaged validation.
- **Compiler specs can still drift.** Generated dashboards must remain the
  source of counts; prose docs should link, not duplicate snapshots.
- **Generated-doc regeneration + drift gating — registry landed (2026-06-04),
  family-collapse consolidation still open.** The fragmentation finding (two
  parallel gate scripts + piecemeal unit gates + inconsistent generator CLIs)
  has been mostly addressed: `python/tessera/compiler/generated_docs.py` is now
  the single registry of all 21 fully-generated dashboards; `check_generated_docs.sh`
  and `release_gate.py` both delegate to it (the second entry point's per-doc
  drift gates were folded into one fleet-wide `generated_docs_drift`); a unified
  `--write` regenerates the whole fleet; and the fleet drift test
  `tests/unit/test_generated_docs_registry.py` includes an orphan guard so a new
  dashboard must register. The registry immediately caught 3 silently-stale
  dashboards (`test_coverage_by_op`, `test_coverage_classification`,
  `effect_lattice_audit`). **CSV-canonical data-shaped tail closed 2026-06-11 —
  12 dashboards now CSV-canonical:** `runtime_abi`, `verifier_coverage`,
  `support_table`, `op_target_conformance`, `runtime_execution_matrix`,
  `test_coverage`, `tsol_coverage`, `effect_lattice_audit`, `surface_status`, and
  the **3 target maps** (`apple_target_map` + `nvidia_sm90_target_map` +
  `rocm_target_map`, added via `apple_target_map.render_csv` /
  `gpu_target_map.render_csv(target)`, wired into the registry so the CSV is the
  drift-gated artifact). The remaining markdown-only docs are narrative rollups
  (`e2e_op_coverage`, `s_series_status`, `s_series_accelerator_proof`,
  `docs_freshness`), not row tables. *Still open (deliberately deferred):* the
  **aggressive content consolidation** (collapse the 3 target maps → 1
  multi-target doc; the `e2e_op_coverage` + `s_series_status` rollups into their
  primaries) — Next Work #6 reassessed these as low-value churn (per-platform
  maps are cross-referenced by the per-platform audit docs; the rollups are
  distinct MASTER_AUDIT truth views).
- **Code-level audit closeout (2026-06-10).** The
  [CODE_AUDIT_2026_06_10.md](CODE_AUDIT_2026_06_10.md) "Closeout status" section
  drove every remaining code-level finding to done / refuted / accepted /
  tracked-deferred. Done: 1e zero-`TRACE_DEFERRED` corpus guard, `_APPLE_GPU_*`
  table-creep enforcer, binary/rowop strict-dispatch funnel coverage, the
  `LoweringUtils.h` dedup across 18 Apple passes, bf16-probe (already cached).
  Explicitly tracked-deferred (with rationale): C-ABI int return code, Target-IR
  C++ fusion-descriptor consumption, Schedule/Tile IR autotuner/LICM (hardware-
  gated), `forbid-ops` pipeline wiring, and the `jit.py` decorator extraction.

## Next Work

1. ~~Add `component_ops`, `fusion_groups`, `shape_envelope`, `effects`, and
   `layout_contracts` to canonical compile metadata.~~ **Landed** —
   `component_ops` (2026-06-02) + `effects` / `shape_envelope` /
   `layout_contracts` / `fusion_groups` (2026-06-07), all reaching the
   user-facing `fn.runtime_artifact().metadata`. **Graph outputs landed
   2026-06-11** — `CompileResult.outputs` / `canonical_outputs`
   (`tessera.compile.outputs.v1`: each returned value + producer op + type /
   shape / dtype / layout), backed by populating `GraphIRFunction.return_values`
   + `result_types` from the jit AST `return` (the AST path previously emitted a
   value-less `return`, so outputs/`shape_envelope.returns` were empty). Locked
   by `tests/unit/test_canonical_outputs.py`; full IR/lit/canonical sweep green.
   Remaining: runtime *consumption* of `fusion_groups` (Next Work #3 / "fusion
   intent too late").
2. ~~Gate whole programs and component ops separately.~~ **Landed 2026-06-02**
   — `program_executable` + `component_blockers` gate the whole program
   component-by-component alongside the primary-op `executable` answer.
3. ~~Make Target IR emit backend descriptors rather than embedding/rediscovering
   large Apple-specific fusion/runtime decisions.~~ **Landed as Phase 0
   (2026-06-15)** — the apple_gpu executor is now authoritative over carried
   fusion roles (`dispatch` on each `known_chain` group) and the four structural
   re-matchers are deleted. Fusion is recognized once (the compiler) and carried
   across the seam to the executor; the executor no longer re-discovers it. See
   the Phase 0a/0b/0c entries in the front-to-back closure plan above.
   **C++ Target IR consume-side reviewed + parity-guarded (2026-06-15).** Unlike
   the Python executor (whose re-matchers were pure duplication and were deleted),
   the C++ Apple fusion passes *must* walk the def-use graph to collect operand
   `Value`s for codegen — that walk is intrinsic, not deletable. The chain passes
   already consume `tessera.fusion.intent` (source `"descriptor"` vs
   `"rediscovered"`, with a Decision-#21 mismatch warning) and the 2-op chains in
   fact lower through the *generic* `synth_matmul_epilogue` synthesizer (F2b), not
   per-pattern hand-kernels. Added a **producer-covers-consumer parity guard**
   (`tests/unit/test_apple_fusion_parity.py`, the C++ analogue of the 0c oracle):
   every producer-stamped chain lowers `source="descriptor"`, never
   `rediscovered`. This caught + fixed a real cross-representation drift Phase 0c
   introduced — `matmul→rmsnorm_safe` is a distinct producer kernel but the C++
   reads a single `"matmul_rmsnorm"` intent for both variants, so
   `stamp_fusion_intents` now maps it via `_FUSION_INTENT_NAME`. Note the C++ Apple
   passes run in lit/validation, **not on the execution path** (that is the Python
   runtime, closed in 0a–0c); their value today is IR auditability + keeping the
   two fusion recognizers in sync for when real codegen eventually routes through
   compiled IR (Phase 4). *Next on this thread:* extend descriptors to NVIDIA/ROCm
   when those backends light up.
4. Require fixture-backed numerical proof before conformance cells become
   complete.
5. Update specs to point at dashboards and this audit instead of old root audit
   documents.
6. **Unify generated-doc regeneration + drift into one contract — landed
   2026-06-04.** `tessera.compiler.generated_docs` is the single registry
   consumed by both `check_generated_docs.sh` and `release_gate.py` (the latter's
   per-doc drift gates folded into one fleet-wide `generated_docs_drift`), with a
   fleet `--write`/`--check`, an orphan-guard test
   (`tests/unit/test_generated_docs_registry.py`), and a `--list` view.
   - **9 dashboards CSV-canonical:** `runtime_abi`, `verifier_coverage`,
     `support_table`, `op_target_conformance`, `runtime_execution_matrix`,
     `tsol_coverage`, `effect_lattice_audit`, plus the merged `test_coverage` and
     consolidated `surface_status`.
   - **Content consolidation done (genuinely-duplicative docs):** the 5
     surface-status docs + `operator_benchmarks_coverage` → one `surface_status`
     (6→1); `test_coverage_by_op` + `test_coverage_classification` → one
     `test_coverage` (2→1). Registry count 24 → 15.
   - **Deliberately not consolidated (reassessed):** the 3 target maps stay
     per-platform — they are *not* duplicative (per-target capability matrices),
     have heterogeneous schemas, and are cross-referenced by 8 per-platform audit
     docs (`backend/{apple,nvidia,rocm}/`); collapsing them would fight the
     per-platform audit structure for a 3→1 saving. The `e2e_op_coverage` /
     `s_series_status` rollups likewise stay standalone — they are distinct
     MASTER_AUDIT truth views, and the registry already removed the duplication
     that mattered (one regen/drift contract). Folding them is available if
     desired but is low-value churn now.

## Source Material Consolidated

- `archive/compiler_apple_backend_end_to_end_audit_2026_06_02.md`
- `archive/compiler_correctness_testing_audit.md`
- `archive/compiler_improvement_milestone_plan_2026_05_18.md`
- `archive/compiler_layer_gap_remediation.md`
- `archive/compiler_spec_gap_audit.md`
- `archive/docs/audit/compiler/COMPILER_AUDIT.md`

