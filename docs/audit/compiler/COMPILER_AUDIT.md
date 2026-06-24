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
| **Python `@jit`** | Decoration-time constraint + effect analysis; honest fallback gating (won't let eager Python masquerade as compiled). | Effect/constraint analysis is single-function, AST-only. A general IR-optimization step (folders/effects) between emission and execution is still thin. | Component-aware multi-op metadata **landed** (carried to the `@jit` artifact); fusion dispatch is **authoritative** (Phase 0 seam closed). Remaining: effect interfaces + broader folding. |
| **Graph IR** | 132 ops, 107 real verifiers; 5 canon patterns; real fusion passes (SwiGLU/MLA/NSA). 101/109 ops are `[Pure]` (CSE/DCE-eligible *today*). | **Folders/canonicalizers landed (2026-06-22):** `add`/`sub`/`mul`/`div`/`cast`/`reshape` folders + `matmul`/`transpose`/`reshape` canonicalizers (8 ops — `reshape` carries an identity fold + a `reshape(reshape(x))` chain-collapse, both guarding the optional `dim_names` symbolic-dim annotations), wired into the `tessera_jit` CPU `canonicalize→cse` pipeline (`graph_ir_folders.mlir`); `LayoutAssignmentPass` landed (seed→propagate→insert `cast{layout}`, `test_layout_assignment.py`). **Per-op effect interfaces landed (2026-06-22):** all 23 non-pure ops carry an explicit `MemoryEffectsOpInterface` — deterministic value ops (`adam`/`adamw`/`momentum`/`adafactor`/`lion`, `arch.ste_one_hot`/`weighted_sum`/`switch`/`mixed`) are `[Pure]`; random (`dropout`/`arch.gumbel_softmax`/`arch.hard_concrete`), stateful (`kv_cache.*`/`ring.create`/`arch.parameter`), collective (`all_reduce`/`reduce_scatter`/`all_gather`) and MoE-transport ops carry `MemWrite`/`MemRead`, so generic CSE/DCE is sound and precise across them (`graph_ir_op_effects.mlir`). `LayoutAssignmentPass` is now **wired into the named x86/GPU/CUDA-13 pipelines behind the opt-in `assign-layouts` option (2026-06-22)** — default off (the inserted `cast{layout}` markers have no backend consumer yet, so the executing path is byte-identical; proven by `layout_assignment_pipeline.mlir`), on when a layout-sensitive backend lands. **Phase 1 closed (2026-06-22)** — effect interfaces, opt-in LayoutAssignment wiring, and reshape folder coverage all landed. ~5 passes are attribute-stamp-only. | Add folders opportunistically as new algebraic identities surface; ~5 attribute-stamp-only passes could gain real bodies. |
| **Schedule IR** | DistributionLowering (real structural wiring + escaping-value fix); collective *insertion*. **Real pipeline partitioning + 1F1B proof landed (2026-06-23)** — `PipelineStagePartition` does a cost-balanced, program-order-monotonic partition (emits `tessera.pp_stage`, no longer external-tag-only), the insertion pass does the genuine send/recv SSA rewire, and `PipelineScheduleLegality` proves the 1F1B schedule (`PP_MICRO_BATCHES_TOO_FEW` per Decision #17, `PP_EMPTY_STAGE`, send/recv pairing, and `PP_UNROUTED_CROSS_STAGE_VALUE` = value-rewrite completeness). Chained as `tessera-pipeline` (partition → insert → legality); lit `tests/tessera-ir/phase4/pipeline_{partition,schedule_legality}.mlir`. | Still annotation-level: the explicit warmup/steady/cooldown *step order* isn't emitted (the proof verifies the structural 1F1B contract, not an emitted step sequence); OptimizerShard is pure attrs; no collective overlap (`ChunkPlanner`/`CollectiveScheduler` never invoked). | Emit the explicit 1F1B step schedule; wire the collective planners. |
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
  treatment is the right one. **Graph-IR folder tail closed (2026-06-17):** of the
  5 `CanonicalizeTesseraIR` patterns, only 2 were CPU-JIT-lowerable. `TransposeIntoMatmul`
  is now also a per-op hook — `MatmulOp::getCanonicalizationPatterns` (the exact
  proven XOR flag-composition: `transpose(Aᵀ)=A`) — so the transpose→flag fold fires
  under the generic `--canonicalize` the tessera_jit CPU lane runs, reaching the
  executed path (proven by `tests/tessera-ir/phase2/graph_ir_folders.mlir` +
  `test_native_cpu_jit.py::test_transpose_into_matmul_folds_on_executed_path`). The
  original stays in `CanonicalizeTesseraIR` for the custom-pass pipelines
  (zero-regression). `EraseIdentityCast` was already covered by `CastOp::fold`. The
  remaining 3 (`FuseMatmulBiasGELU`/`FuseConvRelu`/`DropoutZeroSimplify`) are
  deliberately NOT migrated — they emit `fused_epilogue`/`conv2d_nhwc`/`flash_attn`
  the rank-2 CPU JIT can't lower. **Layout-cast guard landed (2026-06-17):** the
  latent finding that `EraseIdentityCast` (in tessera-canonicalize) and
  `CastOp::fold` (generic --canonicalize) erased a same-type `cast{layout}` before
  the legality check / codegen saw it is **fixed** — both now skip a same-type
  cast carrying a `tessera.layout` attribute (a layout-change marker, not dead
  weight), while plain identity casts still fold. This is the prerequisite for
  `LayoutAssignmentPass` (which inserts same-type `cast{layout}` markers). Guard:
  the `@layout_cast_survives` case in `graph_ir_folders.mlir`.
  **LayoutAssignmentPass v1 landed (2026-06-17):** the assignment half of the
  layout contract (`src/transforms/lib/LayoutAssignmentPass.cpp`,
  `--tessera-layout-assignment`) — (1) seed kernel-producer layouts
  (matmul/batched_gemm→row_major, flash_attn→bhsd, conv2d_nhwc→nhwc), (2) propagate
  through single-result pointwise ops to a fixpoint, (3) insert
  `tessera.cast{tessera.layout=…}` markers at consumer accept-set boundaries (the
  same-type markers the 2026-06-17 cast-fold guard preserves). Paired with
  LayoutLegalityPass as its verifier — `tests/tessera-ir/phase2/layout_assignment.mlir`
  proves the assignment output runs clean through `--tessera-layout-legality`
  (assign + verify). Guards: that lit fixture + `tests/unit/test_layout_assignment.py`.
  *Honest scope:* v1 assignments are IR metadata — no backend consumes them yet
  (the rank-2 CPU JIT is layout-agnostic; Apple GPU is hand-MSL), so this is an
  IR-completeness milestone; when a layout-sensitive backend lands it reads these
  attrs to pick kernels and the cast markers become real memory reorders.
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
  wrapper + `kv_offset` threading is missing — **but note the executed Apple GPU
  path already streams** via the hand-written MSL `flash_attn_f32` online-softmax
  kernel, so this gap is the C++ Tile-IR validation lane + the NVIDIA emitter, not
  Apple execution).
  **Synthesizer generalization — A→B→C→D landed (2026-06-17).** **(A, keystone)**
  `fusion.py` gained `verify_synthesized_pointwise` — the F4 codegen oracle the
  pointwise-DAG path was *missing* (it was the only synthesizer region kind with
  no correctness gate; region/gated/attention all had one). The apple_gpu executor
  now gates the pointwise dispatch branch on it, so a divergent synthesizer falls
  back to the per-op MPSGraph lane instead of being trusted. **(B, measure —
  corrected the plan)** new `compiler/apple_gpu_coverage.py` + guard classifies
  every catalog op against the authoritative lane table: of **302 ops, 177 have a
  GPU lane, 125 are numpy-only, and 0 of those are elementwise/pointwise** — i.e.
  single-op elementwise displacement is *already complete*; the numpy tail is
  layout/indexing/quantize/linalg/spectral/complex. This refuted the original
  Phase-C assumption ("displace elementwise single-ops"). **(C, guided by B)** the
  real lever is enlarging fusable *DAGs*: added `sqrt`/`rsqrt`/`log`/`log1p`/
  `expm1`/`reciprocal`/`softplus` to `POINTWISE_OPS` (they already had single-op
  lanes, so DAGs containing them used to bail at those nodes — now they fuse into
  one kernel, a dispatch-count win), each auto-gated by the (A) oracle
  (`equal_nan`-aware for the domain-restricted ops). **(D, lock)** fused-DAG cases
  added to the differential harness (`_diff_lane.numeric_cases`).
  **Close-out follow-ups landed (2026-06-17):** **(C1 tail)** `maximum`/`minimum`/
  `sign` added to `POINTWISE_OPS`. **(C2)** closed by decision — `EPILOGUE_OPS` is
  deliberately *not* grown beyond the hot matmul-epilogue activations
  (bias/relu/gelu/silu/sigmoid/tanh); rarer activations ride the general
  pointwise-DAG path as a separate on-GPU dispatch, so further in-matmul-epilogue
  entries would be speculative (rationale in the `EPILOGUE_OPS` docstring).
  **(B1 runtime half)** `apple_gpu_coverage.fallback_histogram(run_fn)` runs a
  model under `@jit(apple_gpu)` and reports the failure-class fallbacks
  (shape/dtype/Metal-failure reasons + frequency) from
  `runtime.dispatch_fallback_log` — the runtime complement to the static no-lane
  worklist. **(D2)** the real no-silent-rot regression lock landed: a
  representative pre-norm decoder-MLP block (rmsnorm→matmul→silu→matmul→residual)
  runs on apple_gpu and asserts an **empty** fallback histogram (Darwin-gated);
  a kernel that quietly degrades to numpy trips it. **Parameterized-unary
  follow-up landed (2026-06-17):** `softcap` (the Gemma logit soft-cap
  `cap*tanh(x/cap)`) was the one genuinely numpy-only *real-valued* elementwise
  op. It carries a scalar `cap`, so it rides a GPU **compose** lane (div-by-scalar
  → tanh unary → mul-by-scalar — the clamp/where pattern, no dedicated kernel, no
  `.mm` change) rather than a pointwise-vocab entry. Made a first-class runtime op
  (`_APPLE_GPU_SOFTCAP_OPS` in the master envelope set + `"softcap"` lane +
  handler), which required regenerating the `apple_runtime_ops.inc` X-macro the
  C++ Tile→Apple pass `#include`s and rebuilding `tessera-opt` (the C++/Python
  single-source enforcer + `.inc` drift gate both pass). `cap` is a config literal
  in the jitted source in practice; closure-captured scalars are an unresolved SSA
  ref the apple_gpu metadata path doesn't fold (a known frontend limit, not
  softcap-specific) and the handler fails loudly rather than silently wrong.
  `clamp`/`clip`/`where` were already on GPU compose lanes. Guards:
  `tests/unit/test_apple_gpu_softcap.py`. Remaining: no parameterized-elementwise
  numpy-lane ops left — the displacement worklist's real-valued elementwise tail
  is closed. Guards (prior phases):
  `tests/unit/test_fusion_pointwise_oracle.py`, `test_apple_gpu_coverage.py`,
  `test_fusion_pointwise_vocab_phase_c.py`,
  `test_apple_gpu_displacement_regression.py`.
  **Non-elementwise tail — investigated + categorized (2026-06-17).** The naive
  "displace the 124 numpy-only ops" framing is mostly wrong (the same lesson as
  the elementwise + P2 findings). Investigation: `optim.adam` runs host-side on
  pytrees of numpy (a training-loop utility, never emitted as a single `@jit`
  graph op), and `matmul→transpose→gelu` *demotes to `artifact_only`* because a
  structural op mid-program isn't a recognized chain. So
  `apple_gpu_coverage.disposition_for` now classifies the numpy-only tail:
  **51 `real_gap_structural`** (layout/indexing/state/dropout/position-encoding —
  the genuine target: ops that demote an otherwise-GPU program off
  `metal_runtime`), **50 `hard_kernel`** (quantize packed-FP4/6/8, sparse,
  spectral, stencil, linalg, complex-elementwise, sort/einsum), **8 `host_utility`**
  (optimizers + RNG — no GPU gap), **6 `distributed`** (collectives + MoE
  transport), **9 `unclassified`** (per-op judgment). Guard:
  `test_apple_gpu_coverage.py::test_displacement_disposition_classifies_the_real_gap`.
  *The real displacement target is the 51 structural ops, not 124.*
  **First structural displacement landed — transpose (2026-06-17).**
  `tessera.transpose` now runs on a real MPSGraph kernel
  (`transposeTensor:permutation:`, SDK-header-grounded per Decision #27): N-D
  permute, value-preserving, f32 native + f16/bf16 on the 2-byte raw path, host
  fallback for non-Darwin / GPU-miss. New `.mm` `mpsg_run_transpose` +
  `tessera_apple_gpu_mpsgraph_transpose_{f32,f16}` symbols + stub parity;
  first-class runtime op (`_APPLE_GPU_TRANSPOSE_OPS` → `"transpose"` lane +
  `_apple_gpu_dispatch_transpose`); `.inc` regenerated + `tessera-opt` rebuilt
  (C++ enforcer + drift gate pass). A single-op `@jit(apple_gpu)` transpose now
  reports `execution_kind="native_gpu"` / driver `execution_mode="metal_runtime"`
  (was `fallback_eager`). Guards:
  `tests/unit/test_apple_gpu_transpose.py` (7: 2D/3D/4D + explicit permute, f16,
  jit, no-fallback-on-Metal).
  **General residency gate landed — `per_op_metal` (2026-06-17).**
  `_apple_gpu_chain_kind` now returns `"per_op_metal"` for any multi-op program
  where *every* op has a GPU lane (`lane_for(op) is not None`), checked LAST so the
  named fused chains still win. This closes the transpose-mid-program caveat:
  `matmul→transpose→gelu` (and `matmul→add→transpose→silu`) now run `native_gpu` /
  `metal_runtime` per-op (each op on its lane; the fusion prepass still fuses
  sub-chains) instead of demoting the whole program to `artifact_only`. Conservative
  by construction — a program with any non-lane op returns `None` (stays
  `artifact_only`); per-op handlers still fall back individually (recorded), so the
  program claim stays honest. Guards: `tests/unit/test_apple_gpu_per_op_metal.py`
  (recognizer accepts all-GPU-capable; named fusion still wins; non-GPU op stays
  conservative; mixed program runs `native_gpu` + no-fallback-on-Metal). Updated
  the two Phase-8.4 "multi-op = artifact_only" roadmap gate tests to the new
  contract (all-GPU-capable → `metal_runtime` + numpy-proven; non-lane op →
  artifact_only). *Representation gap (tracked):* the runtime *contract* is
  correctly `metal_runtime` (metadata + verified execution), but the `.target_ir`
  artifact-projection string still uses the per-op-contract / `metal_artifact`
  format for multi-op programs — routing per_op_metal through the runtime-pipeline
  target-IR text is a cosmetic follow-on, orthogonal to the (correct) residency
  claim. **Gather landed (2026-06-17) — second data-mover.** `tessera.gather` now
  runs on a real MPSGraph kernel (`gatherWithUpdatesTensor:axis:0`, header-grounded):
  embedding / attention-index row gather of a 2D table by int32 indices (v1
  envelope: axis-0 + 2D table; other axes / N-D tables fall back to `np.take`).
  Negative indices are normalized before the GPU call so the Metal path matches
  numpy. New `.mm` `mpsg_run_gather` + `tessera_apple_gpu_mpsgraph_gather_{f32,f16}`
  + host fallback + stub parity; first-class runtime op (`_APPLE_GPU_GATHER_OPS` →
  `"gather"` lane). It immediately compounds on the residency gate — an embedding
  lookup mid-program (`gather→matmul`) now runs `native_gpu` instead of demoting.
  Guards: `tests/unit/test_apple_gpu_gather.py` (handler vs numpy over 1D/N-D
  indices, negative indices, f16, jit, no-fallback-on-Metal).
  **Concat landed (2026-06-17) — third data-mover + a frontend-gap fix.**
  `tessera.cat` now runs on a real MPSGraph kernel (`concatTensors:dimension:`,
  header-grounded): the KV-cache-append data-mover — two operands stacked along
  one axis, flattened to an `(outer, axis, inner)` view so *any* rank/axis is one
  GPU concat along dim 1; value-preserving, f32 native + f16/bf16 on the 2-byte
  raw path. >2 operands or mixed dtypes fall back to `np.concatenate` inside the
  dispatcher. New `.mm` `mpsg_run_concat` + `tessera_apple_gpu_mpsgraph_concat_{f32,f16}`
  + host fallback + stub parity; first-class runtime op (`_APPLE_GPU_CONCAT_OPS` →
  `"concat"` lane + `_apple_gpu_dispatch_concat`). Unlike transpose/gather, cat
  was blocked in **both** frontend builders before it could reach a kernel: the
  AST `GraphIRBuilder` and the abstract-interp tracer each rejected a *list* of
  tensor operands (`cat([a, b], axis)` → empty body → `_trace_deferred` /
  "non-Tracer positional operand"), and the op-catalog declared cat/stack as
  fixed arity-1. Fixed generally (also unblocks `stack`): both builders now expand
  a list/tuple of defined tensor values into flat operands, cat/stack arity widened
  to variadic (1–64), and `_execute_op` re-packs the flattened operands for the CPU
  plan (`np.concatenate`/`np.stack`). A single-op `@jit(apple_gpu)` cat now reports
  `execution_kind="native_gpu"`; `matmul→cat` compounds on the per_op_metal gate
  (a KV append mid-program stays GPU-resident). Guards:
  `tests/unit/test_apple_gpu_concat.py` (handler vs numpy over axis 0/1/-1 + rank-3
  seq-axis + f16 + >2-operand fallback, jit native_gpu, matmul→cat per_op_metal,
  no-fallback-on-Metal).
  **Slice landed (2026-06-17) — fourth data-mover + the mirror frontend fix.**
  `tessera.slice` now runs on a real MPSGraph kernel (`sliceTensor:starts:ends:strides:`,
  header-grounded per Decision #27): the StableHLO dynamic-slice / KV-window data-
  mover — a static per-axis window `x[starts[i] : starts[i]+sizes[i]]` (stride 1)
  over an N-D input; `ends[i] = starts[i]+sizes[i]`, value-preserving, f32 native +
  f16/bf16 on the 2-byte raw path. Rank mismatch or out-of-bounds window falls back
  to numpy. New `.mm` `mpsg_run_slice` + `tessera_apple_gpu_mpsgraph_slice_{f32,f16}`
  + host fallback + stub parity; first-class runtime op (`_APPLE_GPU_SLICE_OPS` →
  `"slice"` lane + `_apple_gpu_dispatch_slice`). The frontend fix is the **mirror**
  of cat's: slice's two trailing positional args are index/size *lists of ints*
  (not tensors), so the AST `GraphIRBuilder` must bind them as **attributes**
  (`_POSITIONAL_ATTR_PARAMS["tessera.slice"] = ("start_indices","slice_sizes")`)
  rather than flatten them into operands — otherwise they dropped as `"%?"`
  operands and the op never reached a kernel (cat flattened a list-of-tensors *into*
  operands; slice binds a list-of-ints *out* of operands). A single-op
  `@jit(apple_gpu)` slice now reports `execution_kind="native_gpu"`; `matmul→slice`
  compounds on the per_op_metal gate (windowing a matmul output stays GPU-resident).
  Guards: `tests/unit/test_apple_gpu_slice.py` (handler vs numpy over 2D windows +
  rank-3 + f16 + out-of-bounds fallback, jit native_gpu, matmul→slice per_op_metal,
  no-fallback-on-Metal). *Still open:* the `norm_chain` broadening (bare norms
  already run on the MPSGraph rowop lane — no numpy there to displace, so
  deliberately deferred) — all Evaluator-gated, never displacing a working MPSGraph
  call. **The four structural data-movers (transpose, gather, concat, slice) are now
  all GPU-resident**, so the common KV-cache / embedding / reshape-window glue
  between matmuls no longer demotes a program off Metal.
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
  **`linalg→vector` GEMM lane ✅ LANDED (gated, 2026-06-16) — ~13× over scalar.**
  After two direct-`scf::tileUsingSCF` attempts null-derefed, the **transform
  interpreter** is the working path (it tiles the identical op cleanly under
  `mlir-opt`). The lane (`tools/tessera-jit/tessera_jit.cpp`, opt-in via
  `TESSERA_JIT_VECTORIZE`): run a parsed `transform.named_sequence`
  (`tile_using_for [8,16,16]` → `vectorize_children_and_apply_patterns`) via
  `transform::applyTransformNamedSequence` on the tensor-level IR before
  bufferization (so the K-reduction accumulates in a **register** iter_arg, not
  the memref accumulator that blocked LLVM's vectorizer); then post-bufferize lower
  the vectors (`reduction_to_contract` → contract `OuterProduct` → broadcast /
  shape_cast; **NOT** transfer→`vector.load`, which strands the strided-subview
  load) + `ExpandStridedMetadata` FIRST + `ConvertVectorToSCF` + `ConvertVectorToLLVM`
  + `UBToLLVM` (vectorize emits `ub.poison`); load MLIR's `libmlir_c_runner_utils`
  via `ExecutionEngineOptions.sharedLibPaths` so the DPS-copy `memrefCopy` symbol
  resolves. **Required registrations** (the hard-won set): `TilingInterface` on
  linalg+tensor, the linalg transform-dialect extension, the `vector`/`ub` dialects
  + vector bufferization models. **Result:** matmul programs with all dims ≤ 2048
  (`TESSERA_JIT_VECTORIZE_MAXDIM`, default raised 256→2048 on 2026-06-16) vectorize
  at **~40-46 GFLOP/s** (512³–1024³, ~30 at 128³) — ~13-20× the 2.3 GFLOP/s scalar
  — correct vs numpy; larger programs stay on the scalar JIT lane.
  **Large-N hardened (2026-06-16):** the earlier large-N failure was a *compile-time*
  explosion, not a crash — `vectorize_children` over-vectorized the **untiled**
  elementwise epilogue into a giant `vector<MxN>` that LLVM unrolled into M·N scalar
  ops. The transform now also tiles the 2D elementwise/fill/generic ops (`[8,16]`)
  before vectorizing the func, bounding every vector by the tile sizes; `MAXDIM` is
  now a compile-time safety valve (many tiles ⇒ long-but-finite compile), not a
  crash clamp. Default path (lane off) byte-identical; 25 CPU-JIT tests green incl.
  the gated-lane guard (now pins `MAXDIM=128` to exercise the scalar fallback).
  *Follow-ons:* tune tile sizes. Scope honesty: won't match hand-tuned Accelerate
  BLAS, and the
  **single-GEMM hot path already routes to Accelerate** (`_native_cpu_fast_call`);
  this lane targets multi-op programs that contain a small/medium GEMM.
  **GPU-emission spine landed (2026-06-17, HF).** `tessera-opt` now lowers a
  tessera kernel through `linalg → empty-tensor-to-alloc → one-shot-bufferize →
  convert-linalg-to-parallel-loops → gpu-map-parallel-loops →
  convert-parallel-loops-to-gpu → gpu-kernel-outlining →
  gpu.module(lower-affine, convert-gpu-to-nvvm)`, exposed as the
  `--tessera-emit-nvvm` pipeline. A `tessera.add` emits real NVVM — an outlined
  `gpu.module` with an `llvm.func` kernel (`nvvm.kernel`) reading
  `nvvm.read.ptx.sreg.ctaid.x` etc. Required registering the GPU dialect + the
  bufferization external models + the conversion passes in `tessera-opt` and
  linking the MLIR GPU/NVVM libs (Homebrew LLVM 22 ships `nvptx64` + the libs).
  Guards: `tests/tessera-ir/phase8/gpu_emit_nvvm.mlir` +
  `tests/unit/test_gpu_emit_nvvm.py`. **EMISSION ONLY** — the kernel is produced
  for inspection/codegen; the host `gpu.launch_func` stub remains and GPU launch
  (`tsrRegisterGpuLauncher` → `cuLaunchKernel`/`hipLaunchKernel`) is hardware-gated.
  **ROCDL emission landed (2026-06-17):** `--tessera-emit-rocdl` is the AMD twin
  of the NVVM lane (identical spine, `gpu.module(convert-gpu-to-rocdl)`); a
  `tessera.add` emits real ROCDL (`rocdl.kernel` + `rocdl.workgroup.id.x` + AMD
  data layout). Guard: the ROCDL RUN line in `gpu_emit_nvvm.mlir` +
  `test_gpu_emit_nvvm.py`. **PTX attempted + deferred (2026-06-17):** wired
  `nvvm-attach-target{chip=sm_90}` + `gpu-module-to-binary{format=isa}` (with
  NVPTX target init + the LLVM-IR translation interfaces) as `--tessera-emit-ptx`,
  but it **segfaults inside `mlir::gpu::transformGpuModulesToBinaries`** (the NVVM
  target serialization) on this macOS / Homebrew LLVM 22 build — likely a
  libdevice/toolkit lookup or an LLVM-22 serialization quirk even for `format=isa`.
  Reverted (won't ship a crashing pipeline); the NVVM/ROCDL emission is the proven
  layer. *Next on this thread:* debug the `gpu-module-to-binary` serialization
  (target options / toolkit path) — or chain `tessera-emit-nvvm` → isolate the
  `gpu.module` → `tessera-translate-mlir --mlir-to-llvmir` → `llc -mtriple=nvptx64`
  for PTX text; plus matmul/reduction GPU kernels (beyond elementwise) and the
  gated launch wiring.
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
  `tests/unit/test_layout_legality_extended.py`. **Pipeline wiring landed
  (2026-06-17):** `LayoutLegalityPass` now runs inside `tessera-lower-to-x86`,
  `tessera-lower-to-gpu`, and `buildCUDA13Pipeline` (the nvidia-pipeline aliases)
  — early, after distribution lowering and before `SymbolicDimEqualityPass`, so
  unknown-layout / producer-consumer-mismatch / scale-without-layout violations
  surface with the other structural diagnostics during real lowering (was
  standalone `--tessera-layout-legality`). Proven firing end-to-end by
  `tests/tessera-ir/phase2/layout_legality_in_pipeline.mlir` (x86) +
  `tests/unit/test_layout_legality_pipeline_wiring.py` (all three builders,
  before-symdim ordering). **dtype / aliasing / buffer-binding contracts landed
  2026-06-19:** `IRContractLegalityPass` (`--tessera-ir-contracts`,
  `src/transforms/lib/IRContractLegalityPass.cpp`) is LayoutLegalityPass's sibling
  — one `ModuleOp` walk, 7 stable-coded rules across three families: **dtype**
  (numeric_policy storage/accum coupling, `DTYPE_LEGALITY_TF32_AS_STORAGE`,
  `DTYPE_LEGALITY_LOWP_WITHOUT_WIDE_ACCUM`, `DTYPE_LEGALITY_UNKNOWN_STORAGE` —
  enforces Decision #15a: storage≠accum, TF32 is a math_mode not a storage dtype);
  **aliasing** (`tessera.inplace` requires an in-range `tessera.aliases` —
  `ALIAS_LEGALITY_MISSING_ALIASES` / `_OPERAND_OOB`); **buffer-binding**
  (`tessera.buffer_role` accept-set + no conflicting role per `tessera.binding` —
  `BUFFER_BINDING_UNKNOWN_ROLE` / `_CONFLICT`). Lit:
  `tests/tessera-ir/phase2/ir_contract_legality.mlir` (13 cases); Python:
  `tests/unit/test_ir_contract_legality.py` (12). **Wired into all three named
  lowering pipelines** (`tessera-lower-to-x86`, `tessera-lower-to-gpu`,
  `buildCUDA13Pipeline`) right after `LayoutLegalityPass`, so the contracts fire
  during real lowering — full tessera-ir lit sweep 148 PASS / 19 UNSUPPORTED /
  0 FAIL confirms no existing fixture violates them. The earlier-open
  **Phase 1** of the closure plan
  added the missing *assignment* half — `LayoutAssignmentPass` (seed kernel layouts
  → propagate through pointwise → insert `cast{layout}`), with the legality pass
  reused as its verifier. **Landed 2026-06-22** (`test_layout_assignment.py` +
  `layout_assignment.mlir`); still **not wired into the named x86/GPU pipelines**
  (it mutates IR, so wiring is gated on a layout-sensitive backend consuming the
  attrs). The Graph-IR `hasFolder`/`hasCanonicalizer` gap is closing —
  8 ops now carry folders/canonicalizers (the arithmetic/cast set plus
  `reshape`: identity fold + `reshape(reshape(x))` chain-collapse) wired into
  the `tessera_jit` CPU `canonicalize→cse` pipeline (`graph_ir_folders.mlir`);
  **per-op effect
  interfaces landed 2026-06-22** — all 23 non-pure ops carry an explicit
  `MemoryEffectsOpInterface` (`[Pure]` for the deterministic optimizer/arch
  value ops, `MemWrite`/`MemRead` for random/stateful/collective/MoE-transport
  ops), so generic CSE merges/removes the pure ones and preserves the effectful
  ones (`graph_ir_op_effects.mlir`). `LayoutAssignmentPass` is now **wired into
  the named x86/GPU/CUDA-13 pipelines behind the opt-in `assign-layouts` option**
  (2026-06-22, default off so the executing path is byte-identical;
  `layout_assignment_pipeline.mlir` + `test_layout_legality_pipeline_wiring.py`).
  Folder coverage was broadened to `reshape` the same day, so **Phase 1 is
  closed**; further folders land opportunistically as new algebraic identities
  surface.
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

### External input — TIRx / "Modern GPU Programming for MLSys" review (2026-06-23)

Reviewed the CMU/mlc.ai book *Modern GPU Programming for MLSys*
(https://mlc.ai/modern-gpu-programming-for-mlsys/, TIRx DSL — a TVM-TIR-derived
Blackwell-gen Tile-IR/FA-4 stack). It is a parallel-universe analog of our Tile
IR + FA-4 dialects (TMA, tcgen05, TMEM, mbarriers, warp specialization,
clusters) and commits to several design choices we have not. Candidate work,
**not yet started** — captured here so the Tile-IR/FA-4 thread (Per-IR scorecard
row "Tile IR (FA-4)") can pull from it. Cross-refs noted; reference memory
`reference_mlsys_gpu_book`.

- **C1 — Layout algebra vs. our flat `tessera.layout` string (HF, foundational).**
  Today `tessera.layout` is a **string enum** (`row_major`/`col_major`/`bhsd`/
  `nhwc`/`nchw` — `LayoutAssignmentPass.cpp` `producerLayout`/`consumerAcceptSet`)
  and Tile IR carries only a coarse "optional swizzle" flag on `smem.alloc`
  (`TileMemoryOps.td`). TIRx models layout as a compositional object:
  `S[(shape):(strides)]` shape–stride pairs whose strides carry **named hardware
  axes** (`@laneid/@reg/@warpid/@TLane/@TCol/@m/@gpuid`), with **replication**
  `R[n:stride]` and **swizzle** as a *separate* non-affine `ComposeLayout(swizzle,
  tile)` — never folded into the stride map. This is the abstraction the FA-4
  warp-spec lowering (`WarpSpecializationPass`, `WGMMA`/`TMA`/`TileToX86`) is
  missing — it would unify per-backend layout logic and give the autotuner a real
  object to sweep (tile/lane/swizzle) instead of hardcoded `m64n64k16`. The
  `@gpuid` axis means the *same* algebra spans intra-warp placement and our
  mesh-level `ShardSpec` (Decision #3) at a different scope. **Increment that
  fits today:** add a structured `TileLayoutAttr` (shard/replica/offset triples
  + an explicit `SwizzleAttr` composition) to Tile IR ODS, keep the Graph-IR
  string enum as the coarse producer/consumer contract, and lower the string →
  structured attr at the Schedule→Tile boundary. Extends **Still Open → "Layout
  and binding contracts are uneven"**.
  **v1 LANDED (2026-06-23).** Added first-class `#tile.layout` / `#tile.swizzle`
  attributes to the canonical Tile dialect (`src/compiler/ir/.../TileOps.td` +
  `TileDialect.cpp`): `#tile.layout<shard = [extents] : [strides] on [axes],
  replica = [..] : [..] on [..], offset = N (, swizzle = #tile.swizzle<..>)>` —
  the book's `S ⊕ R ⊕ O` with swizzle held as a *separate* attribute (never
  folded into the affine map). Hand-written parser/printer (the default
  ArrayRefParameter parser rejects the empty `replica = []` common case);
  `genVerifyDecl` enforces parallel-array lengths, positive extents, and a known
  hardware-axis accept-set (`m/tlane/tcol/laneid/warpid/reg/…/gpuid_x/y`) with
  stable codes `TILE_LAYOUT_{RANK_MISMATCH,NONPOSITIVE_EXTENT,UNKNOWN_AXIS}`.
  Lit: `tests/tessera-ir/phase2/tile_layout_attr.mlir` (round-trip incl. a
  TMEM replicated-scale `R[..]` + swizzle case + 3 verifier negatives). *Still
  open:* attaching the attr to the buffer/fragment ops, the string-enum →
  structured-attr lowering at Schedule→Tile, a `.apply()` forward-mapping, and
  WGMMA/TMA consumers reading it.

- **C2 — Barriers as a layout-reuse correctness property, not scheduling (HF→HG).**
  TIRx's central inversion: in FA-4 one `128×512` TMEM allocation is aliased as
  an fp32 view (S/O) *and* an fp16 view (P at 2× column density); the barriers
  exist because each region is **reused** strictly after its prior consumer
  finishes. So barrier requirements should be *generated* by an aliasing/reuse
  analysis over TMEM/SMEM buffers, not emitted alongside `tessera.schedule.warp`
  boundaries. Reinforces Decision #8 (warp roles structural) by making barrier
  slots a function of buffer-reuse decisions. Targets `WarpSpecializationPass` +
  the Queue dialect (currently "WarpSpec emits no queues/mbarriers" per the
  scorecard).
  **v1 LANDED (2026-06-23) as a legality pass.** `TileBarrierReuseLegalityPass`
  (`--tessera-tile-barrier-reuse-legality`, `src/transforms/lib/`, sibling to
  `LayoutLegalityPass`): for a buffer (keyed by `tile.buffer`), two `tile.access
  = "write"` ops whose `#tile.layout` storage-axis (`m/tlane/tcol`) footprints
  *overlap* with no intervening barrier op (name contains `mbarrier`/`wait_async`
  /`barrier`, or a `tile.barrier` attr) emit `TILE_BARRIER_REUSE_MISSING_BARRIER`
  + a note at the prior write. Footprint = `[offset, offset + Σ(extent-1)|stride|]`
  over storage-axis shard dims; a pure register/lane fragment has no storage
  footprint and never aliases. Lit: `tile_barrier_reuse_legality.mlir` — the
  canonical FA-4 fp32/fp16 TMEM-aliasing race (flagged), the same pair with a
  barrier between (clean), disjoint double-buffer offsets (clean), and a
  register-only fragment (clean). This is the **acceptance gate** for C3: once
  WarpSpec emits real typed barriers + buffer reuse, this pass going green on the
  FA-4 fixture is the correctness check. *Still open:* it consumes a convention
  today (`tile.buffer`/`tile.access`); wiring it to real `alloc_shared`/`tmem.alloc`
  SSA buffers + WarpSpec output is the C2↔C3 join.

- **C3 — Typed barrier domains + a `PipelineState` SSA value (HF→HG).** Three
  barrier primitives with distinct completion semantics — `TMABar`
  (byte-count/engine-signaled), `TCGen05Bar` (MMA-completion), `MBarrier`
  (thread-arrived) — and a `PipelineState` that auto-tracks `(stage, phase-bit)`
  with producer initialized `phase=1` / consumer `phase=0` (the packaged fix for
  the classic off-by-one ring deadlock). Our mbarriers are currently generic.
  Targets the `AsyncCopy`/pipeline lowering + Queue dialect; pairs with C5.
  **v1 LANDED (2026-06-23).** Two Tile-dialect attributes —
  `#tile.barrier<kind = tma|tcgen05|mbarrier, expect = N>` (the three completion
  semantics) and `#tile.pipeline_state<depth, stage, phase, role>` — with
  `genVerifyDecl` bounds (`TILE_BARRIER_{UNKNOWN_KIND,NEGATIVE_EXPECT}`,
  `TILE_PIPELINE_{BAD_DEPTH,STAGE_OOB,BAD_PHASE,BAD_ROLE}`). Plus the cross-op
  `TilePipelineLegalityPass` (`--tessera-tile-pipeline-legality`): the initial
  producer-role op of a pipeline (keyed by `tile.pipeline`) must carry `phase=1`
  and the initial consumer `phase=0` (`TILE_PIPELINE_PHASE_ASYMMETRY` — the
  off-by-one deadlock fix), and all ops on one `tile.barrier_id` must agree on
  `kind` (`TILE_PIPELINE_BARRIER_KIND_MISMATCH` + note). Lit:
  `tile_pipeline_attrs.mlir` (round-trip + 6 verifier negatives),
  `tile_pipeline_legality.mlir` (well-formed pipeline clean; producer-phase-0 and
  mixed-kind flagged). *Still open:* `PipelineState` as a threaded SSA value (not
  just an annotation) and WarpSpec emitting these — the C3↔C6 join.

- **C4 — Separate *compute*-legalize from *storage*-legalize (HF).** TIRx runs
  `BF16/FP8 ComputeLegalize` (rewrite math to f32-upcast form) early and
  `…StorageLegalize` (packing) terminally — two passes. This is exactly our
  storage-dtype-vs-accumulator split (Decision #15a, enforced statically by
  `IRContractLegalityPass`) operationalized as *pass ordering*: `numeric_policy.
  accum=fp32` becomes a compute-legalize rewrite, low-precision storage packing a
  terminal pass. Gives the `numeric_policy` contract a concrete lowering home on
  the executed lane. Lowest-risk item; closest to landing.
  **v1 LANDED (2026-06-23).** Two ordered rewrite passes (`DtypeLegalizePass.cpp`):
  `--tessera-compute-legalize` (early) stamps `numeric_policy.accum` on any op
  whose `storage` is reduced-precision and lacks an accumulator — `fp32` for
  float storages, `int32` for `int4`/`int8`; `--tessera-storage-legalize`
  (terminal) stamps `tessera.storage_packed` + `tessera.storage_container` on
  sub-byte / block-scaled storage (`fp4`/`nvfp4`/`fp6`/`int4`). Both idempotent,
  additive, and reusing `IRContractLegalityPass`'s dtype sets. Lit:
  `dtype_legalize_split.mlir` — bf16→accum=fp32, int8→accum=int32, fp4→accum
  +packed-int8-container, fp32 untouched, already-has-accum idempotent; the
  3rd RUN composes `--tessera-ir-contracts` after the split to prove the
  legalized IR is contract-legal (the assign-then-verify pairing).
  **Part 1 — real consumer LANDED (2026-06-23).** `StoragePackConsume`
  (`--tessera-storage-pack-consume`) is the first real consumer of the packing
  markers (previously inert): it reads `tessera.storage_packed` /
  `storage_container` + `numeric_policy.storage` and emits a concrete
  `tessera.storage_pack = {logical, container, factor}` descriptor —
  `factor = container_bits / storage_bits` (fp4/nvfp4/int4 → 2 per int8, fp6 →
  1) — the form a backend's packed load/store reads; bad widths emit
  `DTYPE_PACK_BAD_WIDTHS`. HF Target-IR step (Decision #19). Lit:
  `storage_pack_consume.mlir`. **The real consumer ships on AMD today, not a
  future NVIDIA emitter** — `GenerateWMMAGemmKernel` already nibble-packs 16 int4
  into `vector<2xi32>` (iu4 ABI) and bitcasts int8 to `vector<4xi32>` (iu8). *Real
  integration task (coordinated with the ROCm backend, not a blind rewire):*
  reconcile `tessera.storage_pack {factor}` with the ROCm WMMA `pack` model so
  one packing contract feeds both AMD (shipping) and NVIDIA (future), then flip
  `legalize-dtypes` opt-in → default per target.

- **C5 — Independent per-stream pipeline depths (HF plan / HG perf).** FA-4 runs
  three *independent* rings (Q depth 2, KV depth 3, TMEM depth 2), not one global
  `pipeline_stages`. Our FA-4 config exposes a single `pipeline_stages=2` knob
  (`attn_lower.py`); attention wants per-ring depths the autotuner sweeps
  separately. Also: persistent kernel + **L2-aware tile scheduler** ordering and
  **cluster cross-CTA SMEM views** (`map_shared_rank`/`remote_view`/`cta_mask`) —
  both GPU-only-tier, model when SM90/SM100 execution ungates (Phase G/H).
  **HF scaffold LANDED (2026-06-23).** The hardware-free half — the IR vocabulary
  + the autotuner sweep surface: (1) `#tile.pipeline_depths<q, kv, tmem>`
  Tile-dialect attribute (verifier `TILE_PIPELINE_DEPTHS_NONPOSITIVE`, each ring
  >= 1; lit round-trip + negative in `tile_pipeline_attrs.mlir`); (2)
  `FlashAttnLoweringConfig` gains `q_depth`/`kv_depth`/`tmem_depth` (book defaults
  2/3/2, validated), emitted as `tessera.q_depth/kv_depth/tmem_depth` i32 attrs
  alongside the legacy `pipeline_stages` (which still drives `lds_bytes`, so the
  executing path is byte-identical), plus a `ring_depth_search_space()` that
  enumerates the per-ring sweep candidates (default first). Guard:
  `tests/unit/test_attn_ring_depths.py` (8). **Execution stays gated:** *scoring*
  a candidate needs on-device SM_90/SM_100 latency (Phase G/H) — the surface
  enumerates, it does not measure; persistent/L2/cluster scheduling are likewise
  HG. *Still open (HG):* the measured per-ring sweep, WarpSpec stamping
  `#tile.pipeline_depths` from the config, and the kernel consuming per-ring depths.

- **C6 — A warp-spec diagnostics pass (HF, tooling).** The book's "Debugging
  Warp-Specialized Kernels" appendix is a ready-made spec for a `tessera-opt`
  verification pass: a roles/storage/handoff/lifetime worksheet with checkable
  invariants — *arrival-count == init-count*, *producer/consumer initial phases
  differ*, *no `cta_sync()`/`next_tile()` inside a divergent warpgroup branch*,
  *`fence.proxy_async` before TMA store*, *`commit_group()`+`wait_group(0)` before
  storage reuse*, *`cta_sync()` before writeback dealloc*. These are statically
  checkable on warp-specialized Tile IR and would catch deadlocks/races at
  compile time instead of as device hangs. Natural sibling to
  `IRContractLegalityPass`/`LayoutLegalityPass`. Depends on C2/C3 landing the
  typed barriers + reuse model first. Detailed mapping in the 2026-06-23 review
  notes (this session).
  **v1 LANDED (2026-06-23).** `WarpSpecLegalityPass` (`--tessera-warpspec-legality`,
  `src/transforms/lib/`) checks the four *structural* invariants that complement
  C3's phase asymmetry: `WARPSPEC_INIT_UNDER_GUARD` (a barrier init must run at
  CTA top level, not inside a `tile.warp_role` region), `WARPSPEC_COLLECTIVE_IN_
  DIVERGENT_BRANCH` (cta_sync / cluster_sync / next_tile not inside a warp-role
  region), `WARPSPEC_LOOP_COUNT_DISAGREE` (ops sharing a `tile.pipeline` must
  agree on `tile.trip_count` — the "MMA does K_TILES-1" signature, + note), and
  `WARPSPEC_MISSING_VISIBILITY_FENCE` (a TMA store needs a prior
  fence.proxy_async / commit_group in its block). Convention-driven (warp-role
  region = any ancestor carrying `tile.warp_role`/`tile.warp_guard`/`tile.wg_id`;
  op classes by marker attr or name substring), so it runs on the value lane and
  unregistered husks alike. Lit: `tile_warpspec_legality.mlir` (well-formed
  kernel clean + one negative per invariant). *Still open* (need lifetime
  modeling — the C2↔C6 join): `arrival-count == init-count` and
  cta_sync-before-writeback-dealloc (use-after-free).

**Suggested order:** C4 (cheapest, validates #15a) → C1 (`TileLayoutAttr`,
foundational) → C2+C3 (reuse model + typed barriers/`PipelineState`, mutually
enabling) → C6 (diagnostics, needs C2/C3) → C5 (HG perf). Not to port: TIRx's
TVM plumbing passes (`FlattenBuffer`/`MakePackedAPI`/`LowerWarpMemory`) — MLIR
handles those differently.

**Status (2026-06-23): C1–C4 + C6 v1 LANDED** — the structured `#tile.layout`/
`#tile.swizzle` algebra (C1), the `TileBarrierReuseLegalityPass` reuse-as-
correctness rule (C2), the typed `#tile.barrier` + `#tile.pipeline_state`
attributes and `TilePipelineLegalityPass` (C3), the compute/storage legalize
split (C4), and the `WarpSpecLegalityPass` structural diagnostics (C6) all build
into `tessera-opt` and are lit-green (full `tests/tessera-ir/` sweep 160 passed /
19 unsupported / 0 failed). All five are hardware-free and attribute/convention-
driven (and now wired into the named GPU pipelines + fed by real WarpSpec
markers — see the "Join + pipeline wiring" block below). Together C2 (reuse),
C3 (typed barriers + phase asymmetry), and C6
(structural invariants) are the **deadlock-freedom gate** for the FA-4 warp-spec
lowering.

**Join + pipeline wiring LANDED (2026-06-23).** Two follow-ons closed the gap
between "standalone convention-checkers" and "live lowering gates":
1. **WarpSpec emits the markers.** `WarpSpecializationPass`
   (`src/compiler/tile_opt_fa4/lib/`) now stamps `tile.warp_role` +
   `tile.pipeline` + the typed `#tile.pipeline_state` (producer `phase=1`,
   consumer `phase=0`, `depth=2`) on the producer/consumer `schedule.warp` ops
   it creates — one `warpspec.N` pipeline id per region. So C3/C6 verify *real
   lowering output*, not a hand-written convention. Guard:
   `tests/tessera-ir/phase3/warpspec_emits_markers.mlir` (markers emitted +
   output flows clean through C3+C6).
2. **Wired into the named pipelines.** `tessera-lower-to-gpu` and the four
   `tessera-nvidia-pipeline*` aliases now run `TilePipelineLegality` (C3) +
   `WarpSpecLegality` (C6) + `TileBarrierReuseLegality` (C2) **always-on**
   immediately after `WarpSpecialization` (verified by `--mlir-print-ir-after-all`
   showing the four passes in sequence; full `tests/tessera-ir/` sweep
   **158 passed / 19 unsupported / 0 failed** — the gates pass on every existing
   GPU-pipeline fixture incl. `flash_attn_full`). C4's compute-legalize (before
   `IRContractLegality`) + storage-legalize (terminal) are wired into all three
   pipelines (x86/gpu/CUDA13) behind a new **opt-in `legalize-dtypes` option**
   (default off → byte-identical executing path, mirroring `assign-layouts`;
   confirmed scheduled only when on).

**Buffer-marker emission LANDED (2026-06-23) — C1/C2 markers now on real output.**
`WarpSpecializationPass` also stamps the staged-buffer writes it moves into the
warp regions: each `tile.async_copy` gets `tile.access="write"` +
`tile.buffer="warpspec.N.smem.K"` + a row-major `#tile.layout` on the linear `m`
axis (distinct buffer per copy), and each `tile.mma` gets a TMEM accumulator
buffer (`warpspec.N.tmem.acc.K`) with a `#tile.layout` on the `tlane`/`tcol`
axes. So **C2 (`TileBarrierReuseLegality`) now runs live on real lowering output**
— clean on well-formed lowering (distinct buffers don't alias), and it still
fires `TILE_BARRIER_REUSE_MISSING_BARRIER` on a genuine same-buffer overlap.
Guard: `tests/tessera-ir/phase3/warpspec_buffer_markers.mlir` (markers on
async_copy + mma; C2 clean) + `flash_attn_full` lowers clean through the gate.
*Robustness fix surfaced here:* `TileLayoutAttr::get` runs the `genVerifyDecl`
verifier and **fatal-errors** on an invalid layout, so the stamper skips the
`#tile.layout` (buffer identity only) when a tile has dynamic / placeholder
(`kDynamic`/-1) extents — caught via the flash-attn dynamic-shape path.

**`#tile.barrier` emission + C6 arrival-count LANDED (2026-06-23).**
`NVTMADescriptorPass` now stamps a typed `#tile.barrier<kind="tma", expect=
expect_tx>` + a per-slot `tile.barrier_id="mbar.N"` on **both** the
`tile.tma.setup_descriptor` (init site — declares the expected transaction byte
count) and the `tile.tma.copy_async` (arrive site) for each mbarrier slot, so the
init and arrive of one slot carry the same `(kind, expect, id)`. New C6 rule
`WARPSPEC_ARRIVAL_COUNT_MISMATCH` (`WarpSpecLegalityPass`): per `tile.barrier_id`,
all `#tile.barrier` `expect` values must agree (init count == arrival count) —
else the wait never releases. C3's existing per-id kind-consistency check now
also runs live on these. **The barrier checks need to run *after*
NVTMADescriptor**, so the GPU + CUDA13 pipelines run a *second*
`TilePipelineLegality (C3) + WarpSpecLegality (C6)` placement right after
NVTMADescriptor (the first placement, after WarpSpecialization, still gates the
warp-structure + buffer markers). Verified end-to-end: `flash_attn_full` lowers
through the full `tessera-lower-to-gpu` reaching **both** gate placements, emits
6 consistent `#tile.barrier` markers, exits clean. Guards:
`tests/tessera-ir/phase3/nvtma_barrier_emission.mlir` (emission on setup +
copy_async; output passes C3+C6) + the `arrival_count_mismatch` negative in
`tile_warpspec_legality.mlir`.

**C6 use-after-free LANDED (2026-06-23) — C6 now fully closed (all 7 invariants).**
`WarpSpecializationPass` emits a **writeback-dealloc epilogue** before each
specialized region's terminator: a `tile.cta_sync` followed by a
`tile.buffer_free {tile.buffer=…, tile.access="free"}` for every buffer the
region allocated (the `smem.K` + `tmem.acc.K` it stamped). New C6 rule
`WARPSPEC_USE_AFTER_FREE` (`WarpSpecLegalityPass`, block-local like the fence
check): a buffer free needs a prior `cta_sync` in its block, else a warp may
still be reading the buffer during writeback. Correct lowering is clean (the
epilogue's `cta_sync` precedes the frees); the negative fires on a free with no
preceding sync. Verified: `flash_attn_full` still lowers clean through the full
`tessera-lower-to-gpu` with the epilogue ops flowing downstream. Guards: the
dealloc-epilogue CHECK in `warpspec_buffer_markers.mlir` + the `use_after_free`
negative in `tile_warpspec_legality.mlir`. **All seven appendix invariants
(init-placement, collective-in-branch, loop-count, visibility-fence,
phase-asymmetry [C3], arrival-count, use-after-free) are now checked, and the
full C1–C3/C6 marker vocabulary — `#tile.layout`, `tile.buffer`/`access`,
`#tile.pipeline_state`, `#tile.barrier`, and buffer-free lifetimes — is emitted
by real lowering passes and gated in-pipeline.**

**C5 HF scaffold LANDED (2026-06-23).** The hardware-free half of C5 — the
`#tile.pipeline_depths<q, kv, tmem>` IR attribute + `FlashAttnLoweringConfig`'s
per-ring depths/emission/`ring_depth_search_space()` sweep surface — is in
(`test_attn_ring_depths.py`). **Every TIRx-review item (C1–C6) now has its
hardware-free portion landed, lit/unit-green, and (for C1–C4/C6) wired into the
named GPU pipelines fed by real lowering markers.** What remains is strictly
**hardware-gated** (Phase G/H, SM90/SM100 silicon): the measured per-ring depth
sweep, persistent/L2-aware tile scheduling + cluster cross-CTA SMEM views (C5),
WarpSpec stamping `#tile.pipeline_depths`, and the kernels that consume the
per-ring depths. There is no remaining hardware-free TIRx-review work.

**Registry sync + typed-contract hardening LANDED (2026-06-23).** Two follow-ups
after the C1–C6 feature: (1) the Python meta-registries now reflect the new
surface — `dialects_manifest` registers the `tile` dialect; `diagnostic_codes`
registers the 23 new MLIR codes (TILE_LAYOUT_* / TILE_BARRIER_* / TILE_PIPELINE_*
/ TILE_PIPELINE_DEPTHS_NONPOSITIVE / TILE_BUFFER_REF_* / WARPSPEC_*);
`pass_metadata` adds the 5 new passes (compute/storage-legalize + the C2/C3/C6
gates) with their codes/dialects/required-attrs; `pipeline_registry` reflects the
two gate placements + the `legalize-dtypes` option in the GPU/nvidia pipelines.
All drift gates green (108 registry tests, 17 generated docs in sync). (2) The
first **typed contract** strengthening of the marker conventions: the loose
`tile.buffer` + `tile.access` string pair is replaced by a typed
`#tile.buffer_ref<name, space, access>` attribute whose `space` (smem/tmem/gmem/
reg) and `access` (read/write/free) are closed, verifier-checked sets
(`TILE_BUFFER_REF_{EMPTY_NAME,BAD_SPACE,BAD_ACCESS}`). WarpSpec emits it on staged
writes + the dealloc epilogue; C2/C6 read the typed handle; flash_attn lowers
clean. *Next (the SSA half):* promote buffer/barrier *identity* from a string
name to an SSA `!tile.buffer` handle produced by a `tile.alloc` op and consumed
by `tile.dealloc` (def-use lifetimes instead of name matching) — a TypeDef +
op-pair refactor that lets C2/C6 track real values, scoped as a focused follow-on.

**Backend parity — ROCm is first-class, not second-class to CUDA (2026-06-23).**
The C1–C6 IR contracts were initially NVIDIA-shaped (the typed vocabularies only
spoke CUDA). Corrected: the contracts now name AMD hardware natively, neither
backend privileged — `#tile.layout` axes add `lds` (AMD shared) + `waveid` (AMD
wave) alongside `m`/`warpid`; `#tile.barrier` kinds add `s_barrier` (workgroup
arrival) + `waitcnt` (async vmcnt/lgkmcnt) alongside tma/tcgen05/mbarrier;
`#tile.buffer_ref` space adds `lds`; and C2's storage-aliasing treats `lds` as a
memory axis, so **LDS reuse-without-barrier is caught exactly like SMEM/TMEM**
(`tile_{layout_attr,pipeline_attrs,barrier_reuse_legality}.mlir` carry the AMD
cases). Reality check that drove this: the ROCm WMMA lane is the *more active
execution path* (the #87–90 commits ship real hsaco + int4/int8 WMMA + flash-attn
fwd/bwd on gfx1151), so it earns first-class treatment in the shared contracts.
*Deliberately NOT done:* bolting the NVIDIA pass-chain / legality gates onto
`tessera-lower-to-rocm` — that lane is a different (direct WMMA kernel-gen)
architecture the backend team actively owns; the gates apply there only once/if
ROCm grows a warp-specialized Tile-IR path, and wiring them is a coordinated
change, not a unilateral one.

## Next Work

> **Open items: #4 (fixture-backed numerical proof before conformance cells go
> complete) and #5 (point specs at dashboards/this audit, not old root audits).**
> Items #1, #2, #3, and #6 have **landed** — they are kept below (struck through)
> for provenance, not as pending work.

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
   documents. **Verified 2026-06-19:** the only specs that link an audit
   (`TARGET_IR_SPEC.md`, `AUTODIFF_SPEC.md`) already use the current theme-audit
   path `docs/audit/coverage/COVERAGE_AUDIT.md` — no stale root-audit references
   remain. Adding generated-dashboard pointers to the remaining specs is optional
   polish, not a correctness gap.
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

