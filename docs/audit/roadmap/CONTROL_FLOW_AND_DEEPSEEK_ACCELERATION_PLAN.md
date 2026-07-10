---
last_updated: 2026-06-30
audit_role: plan
plan_state: open
---

# Control-Flow and DeepSeek Acceleration Plan

## Goal

Move `tessera.control.{scan, fori_loop, while_loop, cond}` from Python-reference
host loops into compiler-visible Graph IR and backend-executable CUDA/ROCm
lowerings. The first production target is speculative decode, especially the
Gumiho and DSpark-style draft/verify loops, where a host-dispatched loop pays a
GPU launch and sync on every iteration.

Apple GPU remains a research lane for this plan. The existing Phase-G Apple work
proves important runtime ideas, but the portable closure path should start with
CUDA and ROCm because both backends can own explicit kernels, device-side
control flow, and target-specific fused decode primitives.

## Scope

This plan covers four connected tracks:

1. First-class control-flow IR for bounded and predicate-driven loops.
2. CUDA/ROCm lowering for device-side control flow and speculative decode.
3. DSpark/Gumiho draft-block serving primitives.
4. DeepSeek-family fused kernels: MLA, sparse attention, MoE packing, grouped
   GEMM, dequant-GEMM, and FP8/INT4 numeric policy.

Out of scope for the first CUDA/ROCm closure:

- Full Apple MPSGraph/MSL parity beyond the current Phase-G research ladder.
- Distributed multi-node serving.
- Full DeepEP-style expert-parallel transport. This plan prepares the compiler
  and runtime contracts, but transport is a later standard-runtime layer.

## Current Baseline

- `python/tessera/control.py` owns reference `scan`, `fori_loop`, `while_loop`,
  and `cond`. These can interact with tracing, but they are not first-class
  executable Graph IR control-flow ops.
- `docs/audit/backend/apple/archive/apple_gpu_control_flow_lowering.md` documents the Apple Phase-G ladder:
  bounded scan, serial draft, while-generate, and speculative accept slices.
- `python/tessera/speculative.py`, `python/tessera/stdlib/hybrid.py`, and the
  Gumiho tests already provide speculative-decode reference semantics.
- `python/tessera/stdlib/{attention,moe,quant}.py` already define reference
  contracts for MLA, DSA/NSA-style sparse attention, MoE routing, grouped GEMM,
  and dequantized low-precision weight paths.
- `src/transforms/include/Tessera/Transforms/Passes.h` already reserves fusion
  pass slots for MLA decode and native sparse attention.
- CUDA/ROCm have artifact and kernel-inventory work, but the control-flow and
  DeepSeek-family fused kernels are not yet a coherent execution ladder.

## Milestone Ladder

| ID | Title | Main owner | Output |
|---|---|---|---|
| CF0 | Contract inventory | Python/compiler docs | Shape/effect/control-flow envelope and runnable reference tests |
| CF1 | Graph IR control-flow ops | Graph IR | `tessera.control.{scan,for,while,cond}` ops with verifiers |
| CF2 | Schedule/Tile control-flow lowering | Compiler | Static-shape loop bodies lower to Schedule/Tile regions |
| CF3 | CUDA device-control kernels | CUDA backend/runtime | Single-launch scan/for/while/cond proofs |
| CF4 | ROCm device-control kernels | ROCm backend/runtime | HIP equivalents for CF3 |
| SD1 | Speculative decode primitive set | Stdlib/runtime | Draft proposal, verify, accept, commit/rollback contracts |
| DS1 | DSpark block draft reference | Stdlib/models | Anchor-block attention, Markov head, confidence head |
| DS2 | Fused draft-block forward | Compiler + CUDA/ROCm | One backend primitive for DSpark/Gumiho draft block |
| DK1 | MLA fused decode | Compiler + CUDA/ROCm | Absorbed MLA decode kernel contract and executable proof |
| DK2 | NSA/DSA sparse attention | Compiler + CUDA/ROCm | Native block-sparse decode/prefill kernels |
| DK3 | MoE dispatch and grouped GEMM | Compiler + CUDA/ROCm | Expert pack/dispatch/combine plus grouped GEMM kernels |
| DK4 | Quantized dequant-GEMM | Compiler + CUDA/ROCm | Packed INT4/FP8 no-materialization dequant-GEMM |
| DK5 | Integration gate | End-to-end | Speculative DeepSeek/DSpark-style scaled decode uses fused paths |

## CF0 - Contract Inventory

Close the semantic envelope before emitting backend code.

Deliverables:

- Document supported loop forms:
  - `scan`: static trip count, static carry shapes, optional stacked outputs.
  - `fori_loop`: static or symbolic bounded trip count, static carry shapes.
  - `while_loop`: bounded maximum plus dynamic predicate, static carry shapes.
  - `cond`: scalar predicate, same-typed branch results.
- Define effects:
  - no Python side effects inside device-lowered bodies;
  - RNG only through explicit state operands;
  - cache mutation only through typed cache handles;
  - speculative rollback via cursor/state handles, not implicit Python state.
- Add negative tests for dynamic shapes, data-dependent allocation, host object
  capture, and branch type mismatch.

Acceptance:

- Reference tests prove current Python behavior remains unchanged.
- Trace/JIT tests produce a clear "unsupported control flow" diagnostic until
  CF1 lands, never a silent host-loop fallback inside an executable backend
  claim.

## CF1 - Graph IR Control-Flow Ops

Add first-class Graph IR ops rather than lowering Python loops directly to a
backend-specific runtime symbol.

Proposed ops:

- `tessera.control.scan`
- `tessera.control.fori_loop`
- `tessera.control.while_loop`
- `tessera.control.cond`
- Optional helper: `tessera.control.yield`

Verifier rules:

- Carry/result arity and dtypes match across iterations and branches.
- Loop-carried tensor ranks and static dimensions are preserved.
- `while_loop` carries an explicit `max_steps` or bounded trip guard for backend
  codegen.
- Region body effects are declared and checked.
- Cache handles are linear or explicitly fork/commit/rollback typed.

Acceptance:

- MLIR lit fixtures cover positive and negative verifier cases.
- Python tracing emits these ops for simple control functions.
- Existing Python eager path remains the reference fallback when not under JIT.

## CF2 - Schedule/Tile Lowering

Lower Graph IR control-flow ops into Schedule/Tile region constructs that CUDA
and ROCm can share.

Lowering rules:

- `scan` and static `fori_loop` become static trip loops with explicit carry
  buffers and optional output accumulation.
- `while_loop` becomes a bounded loop with a device-side predicate and an
  output `actual_steps`.
- `cond` becomes predicated branch regions or select-like lowering when both
  branches are pure elementwise regions.
- Body lowering reuses existing matmul, attention, reduction, and gather/scatter
  lowerings; unsupported body ops fail at compile time.

Acceptance:

- Hardware-free Target IR fixtures for CUDA and ROCm show loop structure,
  region operands, and launch metadata.
- Tests assert there is one backend launch for the loop wrapper, not one launch
  per iteration.

## CF3 - CUDA Support

CUDA should be the first fully executable backend for this plan.

Implementation slices:

1. Runtime ABI:
   - register control-flow kernels through the existing CUDA runtime bridge;
   - expose launch metadata for loop trip count, carry sizes, and output spans.
2. Proof kernels:
   - static scan over a small recurrent cell;
   - bounded `fori_loop` with tensor carry;
   - bounded `while_loop` with EOS-style early stop;
   - scalar `cond` with matching tensor branches.
3. Speculative accept:
   - CUDA kernel for fixed-capacity multi-path accept/select;
   - match the existing Apple `spec_accept` semantics, but do not depend on
     Apple-specific control-flow choices.

Acceptance:

- CUDA tests compare against numpy/Python reference for each primitive.
- Provenance says CUDA runtime execution, not numpy fallback.
- Artifact-only fixtures are promoted to executable where CI has CUDA; otherwise
  they skip cleanly and retain compile-only coverage.

## CF4 - ROCm Support

ROCm should mirror CUDA at the contract level, with HIP implementation details
and AMD target profiles driving dtype/wave choices.

Implementation slices:

1. Runtime ABI:
   - HIP launch bridge for control-flow kernels;
   - per-arch capability checks using the existing ROCm target profile.
2. Proof kernels:
   - HIP scan, for, while, and cond kernels with the same reference tests as
     CUDA.
3. Wave policy:
   - keep the first control-flow kernels scalar/control dominated and wave-size
     conservative;
   - only specialize wave32/wave64 when the body kernel needs MFMA/WMMA.

Acceptance:

- ROCm lit fixtures verify Target IR.
- HIP execute-and-compare tests run when ROCm is available and skip cleanly
  otherwise.
- Diagnostics distinguish "unsupported arch/dtype" from "no ROCm host".

## SD1 - Speculative Decode Primitive Set

Make speculative decode a compiler/runtime contract rather than only a Python
or example-level loop.

Primitives:

- `draft_propose`: produces candidate tokens and optional draft probabilities.
- `target_verify`: runs target model over current token plus draft prefix.
- `spec_accept`: computes accepted prefix length and bonus/residual token.
- `cache_commit`: advances KV/SSM/cache handles by accepted prefix.
- `cache_rollback`: rewinds rejected draft state.

Compiler visibility:

- `spec_accept` should be a small standalone op first.
- `cache_commit`/`cache_rollback` need typed effects so loops can be lowered
  without hidden Python mutation.
- `target_verify` remains a composed model call, but its input/output contract
  should be explicit enough for batching and fusion.

Acceptance:

- Existing `tessera.speculative` tests become the reference oracle.
- CUDA and ROCm each provide one native `spec_accept` proof.
- Gumiho serial draft can execute as one backend loop once CF3/CF4 are present.

### Status

- **SD1-1 (spec_accept standalone op + ROCm proof) — done, ROCm/gfx1151.**
  `spec_accept` is now a first-class Graph IR op (`Tessera_SpecAcceptOp`, `[Pure]`,
  ODS + verifier): `draft P×D i32`, `target P×(D+1) i32` →
  `tensor<3xi32> [accepted_path_idx, accepted_prefix_length, bonus_token]`. The
  **greedy** (argmax-match) longest-prefix form is the small standalone op the
  plan asks for first; `GenerateROCMSpecAcceptKernel`
  (`--generate-rocm-spec-accept-kernel`) lowers it to one cooperative-workgroup
  `gpu.func` (thread/path match-length into LDS, barrier, thread-0 argmax + bonus)
  and executes bit-exact on gfx1151 vs the `tessera.speculative` greedy reference
  (`_ref_spec_accept`), proven by `tests/unit/test_rocm_spec_accept_exec.py`
  (`op_catalog` + the auto-generated coverage row updated per Decision #24). The
  CUDA native `spec_accept` proof stays artifact-only here (no NVIDIA HW); the
  distribution-preserving **Leviathan / rejection-sampling** form (per-token
  `min(1, p_t/p_q)` acceptance + residual bonus, with RNG) is **SD1-2**.
- **SD1-2 (Leviathan rejection-sampling spec_accept) — done, ROCm/gfx1151.** New
  op `Tessera_SpecAcceptSampleOp` (`[Pure]`, ODS + verifier): the
  distribution-preserving linear-chain verify. Per position accept
  `accept_u[i]·p_draft ≤ p_target`; on first reject draw a corrected token from the
  residual `normalize(relu(target−draft))`, on full accept a bonus from
  `target_probs`'s extra row. **RNG is explicit** — `accept_u`/`resid_u` are op
  operands (CF0 contract), and the categorical draw is **CDF inversion** (not
  numpy `rng.choice`), so the op is a deterministic, device-bit-exact function.
  `GenerateROCMSpecAcceptSampleKernel`
  (`--generate-rocm-spec-accept-sample-kernel`) lowers it to one single-thread
  `gpu.func` (serial accept loop + CDF-inversion categorical), proven bit-exact on
  gfx1151 by `tests/unit/test_rocm_spec_accept_sample_exec.py` (random
  rejection-sampling + full-accept). `op_catalog`/`tessera.ops`
  registration + `PYTHON_API_SPEC` updated.
- **SD1-3 (cache_commit/cache_rollback typed effects) — done.** New Graph IR ops
  `Tessera_CacheCommitOp` (`cache.commit`) + `Tessera_CacheRollbackOp`
  (`cache.rollback`), both `MemoryEffects<[MemWrite]>` (NOT `[Pure]`) — the
  `EffectLattice` classifies a region using them as **`state`**, not pure, so a
  speculative loop that mutates the cache is compiler-visible. `cache.commit`
  advances the KV/SSM cursor to keep the accepted prefix (the IR form of
  `speculative.advance_kv`); `cache.rollback` rewinds rejected drafts
  (`SSMStateHandle.rollback` / `KVCacheHandle.trim`, the IR form of `advance_ssm`).
  The handle is threaded value-to-value (`cache → updated`) so the mutation is
  linear. `op_catalog` (`effect="state"`), `tessera.ops` registration, `effects.py`
  (`Effect.state`), `PYTHON_API_SPEC`, and the `_NO_LANE_BARE` residual updated;
  proven by `tests/unit/test_cache_effect_ops.py` (EffectLattice → `state`; the
  cursor refs match `advance_kv`/`advance_ssm` on real KV + SSM handles). This makes
  CF0's "cache mutation only through typed cache handles" + "speculative rollback
  via cursor/state handles" an **IR-visible typed effect**. No device kernel
  (cursor ops).
- **Tree (multi-path) rejection-sampling — done, ROCm/gfx1151.** New op
  `Tessera_SpecAcceptTreeSampleOp` (`[Pure]`) — the device form of
  `speculative.batch_verify`. Over `P` draft paths each `D` deep, accept the draft
  token at `(p,i)` iff `accept_u[p,i] ≤ exp(target_log_probs[p,i] −
  draft_log_probs[p,i])`; pick the longest-accepted-prefix path (first wins ties)
  → `[accepted_path_idx, accepted_prefix_length]`. Explicit-`u` (deterministic);
  `GenerateROCMSpecAcceptTreeSampleKernel`
  (`--generate-rocm-spec-accept-tree-sample-kernel`) lowers it to one
  cooperative-workgroup `gpu.func` (thread/path exp-accept length + argmax), proven
  bit-exact on gfx1151 by `tests/unit/test_rocm_spec_accept_tree_sample_exec.py`.
  Generalizes the linear `spec_accept_sample` to a tree and the greedy
  `spec_accept` to the stochastic rule.
- **Gumiho serial-draft loop — done (reference + composition).**
  `speculative.gumiho_serial_draft` composes the SD1 primitive chain — draft →
  `target_verify` → `spec_accept` → `cache_commit` — into ONE bounded decode loop
  (the kind that lowers to one `control_scan` device dispatch, CF4e, rather than
  one launch per token). The greedy invariant is proven in
  `tests/unit/test_gumiho_serial_draft.py`: for ANY draft model the emitted
  sequence is identical to plain autoregressive decode with the target
  (`autoregressive_decode`) — speculation changes only the number of target calls,
  never the output. The fully-device-executed loop (with the draft/target *models*
  as device kernels) is the DS/DK-track integration.
- **SD1-4 (target_verify I/O contract) — done.** New Graph IR op
  `Tessera_TargetVerifyOp` (`target_verify`, `[Pure]`, ODS + verifier): the
  input/output contract for the target-scoring step. Given the verified-position
  context `tokens` (current committed token + `D` draft tokens, `S = D+1`
  positions) and the composed target model's raw `logits` (`S × V`), it emits the
  contract-shaped per-position target log-probs `S × V` (`log_softmax` over the
  vocab) — exactly what `spec_accept`/`spec_accept_sample` consume (the verify
  output for a `D`-token draft has the bonus row built in). It's a **composed model
  call**, not a fused kernel: the op pins the `(S × V)` batching shapes so the
  target scoring is explicit enough to batch/fuse, but the target model is
  external; a fused target-verify kernel is a later DK-track concern. `op_catalog`
  (reusing the `acceptance_verification` category), `tessera.ops` registration
  (`log_softmax` reference), `PYTHON_API_SPEC`, and `_NO_LANE_BARE` updated; proven
  by `tests/unit/test_target_verify_contract.py` (log-softmax contract +
  feeds-spec_accept_sample shape) + the verifier lit fixture.
- **SD1-4+ (open):** fully-device-executed Gumiho loop with the draft/target
  models as device kernels (DS/DK-track integration).

## DS1 - DSpark Library Contract

Add DSpark as a library/model-family contract before attempting fused kernels.

Current increment (2026-06-30):

- **DS1 reference library — landed.** `tessera.stdlib.dspark` now owns the
  NumPy oracle for anchor sampling, anchor candidate masks, anchor-block
  attention masks, vanilla draft-block forward, confidence-prefix selection,
  proposal selection, and CE + L1/probability/confidence losses.
- **Tests — landed.** `tests/unit/test_stdlib_dspark.py` pins the DSpark shape
  contract that DS2 CUDA/ROCm fused draft-block kernels must match:
  `[B, anchors, block_size, vocab]` logits, `[B, anchors, block_size]`
  confidence logits, selected draft tokens, and proposal rows consumable by the
  SD1 `spec_accept` path.

Library pieces:

- anchor sampling and anchor candidate masks;
- anchor-block attention mask;
- DSpark draft block model surface;
- Markov head variants: vanilla first, gated/RNN later;
- confidence head and thresholded prefix selection;
- CE plus L1/probability-matching loss against aligned target logits.

Compiler-visible pieces:

- `dspark_anchor_block_attention` as a block-mask attention lowering pattern;
- `dspark_confident_prefix` as a bounded prefix-select op;
- target-cache hidden-state reads as ordinary tensor operands, not implicit
  external state.

Acceptance:

- DSpark reference tests match DeepSpec-style shape contracts:
  `[B, anchors, block_size, vocab]` logits, target ids, masks, confidence logits.
- DSpark serving proposal plugs into `tessera.speculative` verification.

## ROCm Execution Order

This is the dependency order for the ROCm buildout. CUDA can lead when a kernel
design is easier to validate there, but every step below has a ROCm acceptance
gate before it is considered closed.

1. **DS1 — reference oracle.** Keep `tessera.stdlib.dspark` as the source of
   truth for DSpark proposal numerics and shapes.
2. **DS2 — fused draft-block forward.** Lower the DSpark block mask + vanilla
   draft head to a static-shape ROCm kernel and compare against DS1.
3. **DK4 — dequant-GEMM.** Promote packed codes + scales to CUDA/ROCm fused
   dequant-into-GEMM, because MoE experts and model weights consume it.
4. **DK3 — MoE dispatch/combine/grouped GEMM.** Use DK4 for quantized expert
   compute; prove deterministic dispatch/combine and uneven group sizes.
5. **DK1 — MLA fused decode.** Reuse the existing MLA fusion slot and add the
   ROCm absorbed-latent decode proof against `stdlib.attention.mla_decode_step`.
6. **DK2 — NSA/DSA/MSA sparse attention.** Lower selected-block layouts to ROCm
   decode kernels and preserve dense-equivalence tests.
7. **DK5 — scaled end-to-end speculative decode gate.** Compose DS2 + target
   verify + spec_accept + cache commit/rollback, selecting DK1/DK2/DK3/DK4 lanes
   when the model config asks for them.

## DS2 - Fused Draft-Block Forward

This is the first DSpark performance payoff.

Current increment (2026-06-30):

- **DS2 native ROCm generator + runtime ABI — landed.** The
  `rocm_dspark_draft_block_compiled` execution-matrix row now reaches
  `runtime.launch()` and attempts the compiler-generated
  `generate-rocm-dspark-draft-block-kernel` HSACO path first. On ROCm hardware
  it reports `execution_kind=native_gpu`; without `tessera-opt`/ROCm hardware it
  falls back to the DS1 oracle and reports `reference_cpu`.
- **DS2 tests — landed.** `tests/unit/test_rocm_dspark_draft_block_compiled.py`
  proves runtime output matches `tessera.stdlib.dspark.draft_block_forward`,
  includes a codegen-lowering smoke for the generated ROCm pass, and carries a
  hardware-gated native ROCm correctness proof. `test_stdlib_dspark_perf.py`
  carries the DS1/DS2 CPU perf budgets.
- **DS2 native ROCm perf proof — hardware-gated.** The first native fused
  draft-block kernel is intentionally shape-generic and serializes each
  `(batch, anchor)` chain inside one GPU thread. The remaining promotion is the
  cooperative H/V reduction tuning and benchmark proof against the reference
  launch baseline on target ROCm hardware.

Kernel contract:

- Inputs:
  - current token/prompt slice;
  - target hidden-state context slices;
  - anchor positions;
  - draft model weights;
  - optional Markov/confidence head weights.
- Outputs:
  - draft tokens;
  - draft probabilities or logits;
  - confidence logits;
  - optional proposal length after thresholding.

Lowering:

- Start with static `block_size` and static `num_anchors`.
- Lower anchor-block attention as block-sparse attention over context plus draft
  tokens.
- Fuse Markov bias and confidence projection after the draft hidden states.
- Keep target verification as a separate target-model call until the draft block
  path is stable.

Acceptance:

- CUDA and ROCm each provide a scaled fused draft-block proof.
- Fused output matches the DSpark reference proposal path.
- Perf ratchet beats host-loop/per-op dispatch on a representative small block.

## DK1 - MLA Fused Decode

Promote existing MLA compiler visibility into executable CUDA/ROCm kernels.

Current increment (2026-06-30):

- **DK1 ROCm absorbed-latent decode proof — landed.** The existing
  `rocm_exotic_attn_compiled` MLA fusion slot now accepts
  `tessera.mla_decode_step`, appends compressed latent/rope cache state like
  `stdlib.attention.mla_decode_step`, and routes the attention body through the
  generated `generate-rocm-mla-absorb-decode-kernel` path on ROCm hardware.
- **DK1 fallback provenance — landed.** Without a usable ROCm runtime the same
  artifact falls back to `stdlib.attention.mla_decode_step` and reports
  `execution_kind=reference_cpu`; hardware runs report `native_gpu`.
- **DK1 remaining promotion.** The first ROCm kernel is scalar-per-output and
  proves the ABI/numerics without materializing per-head K/V. The open
  performance work is cooperative tiling over sequence/latent/value dimensions
  and the CUDA sibling proof.

Work:

- Reuse the MLA fusion pass slot to rewrite explicit latent K/V materialization
  into `mla_decode_fused`.
- Support absorbed latent-cache decode first:
  - compressed latent cache;
  - shared RoPE key slice;
  - no materialized full K/V.
- Add dtype policy for f32/f16/bf16 first; FP8 KV cache can follow once numeric
  policy is closed.

Acceptance:

- CUDA and ROCm lit fixtures show fused op selection.
- Runtime tests compare against `stdlib.attention.mla_decode_step`.
- Decode chunk and token-by-token outputs match within dtype tolerance.

## DK2 - DeepSeek Sparse Attention / NSA-Style Lowering

Unify the existing DSA/NSA/MSA sparse-attention contracts into target-specific
block-sparse kernels.

Current increment (2026-06-30):

- **DK2 ROCm selected-block kernel proof — landed.** A new
  `rocm_sparse_attn_compiled` lane lowers explicit selected KV-block layouts to
  the generated `generate-rocm-block-sparse-attn-kernel` path. The ABI consumes
  `B,Hkv,Sq,top_k` block ids, query positions, Q/K/V, and block metadata, then
  computes exact GQA softmax attention over only the selected blocks.
- **MSA/DSA selected-layout coverage — landed.** MSA can pass
  `selected_block_ids` directly or let the runtime use the stdlib selector; DSA
  normalizes its selected keep mask into the same block-id worklist. The
  artifact falls back to stdlib references without ROCm hardware and reports
  `reference_cpu`; hardware runs report `native_gpu`.
- **DK2 first performance promotion — landed.** The lane now has a row-tiled
  selected-block attention directive (`block_sparse_attention_tiled`) and a
  GPU-resident top-k selector directive (`block_sparse_topk_select`). Runtime
  `selection_strategy="auto"` uses the GPU selector on ROCm hardware and avoids
  hardware probes on CPU-only hosts; `attention_strategy="tiled"` selects the
  row-tiled kernel for value widths that fit one workgroup.
- **DK2 remaining promotion.** Open performance work is deeper cooperative
  sparse-flash tiling that shares score/softmax work across value lanes, plus
  CUDA parity.

Work:

- Define one selected-block layout contract per family:
  - DSA/NSA decode;
  - MSA KV-outer prefill/decode where applicable.
- Preserve dense-equivalence oracle when selected blocks cover the full KV span.
- Lower top-k selected blocks to a fixed worklist; dynamic selection can remain
  a pre-kernel step until the block-sparse kernel is stable.

Acceptance:

- CUDA native KV-outer sparse attention closes the current artifact-only MSA
  gap.
- ROCm has matching Target IR and a first executable decode proof.
- Dense-equivalence and restricted-top-k tests pass.

## DK3 - MoE Dispatch, Combine, and Grouped GEMM

Make MoE movement and grouped expert compute compiler-visible and backend
executable.

Current increment (2026-06-30):

- **DK3 ROCm transport ABI + perf baseline - landed.** The
  `rocm_moe_transport_compiled` execution-matrix row now reaches
  `runtime.launch()` for `moe_dispatch`, `moe_combine`, and `grouped_swiglu`,
  consuming the stdlib `DispatchPlan` through the same ABI the HIP transport
  kernels will use. It reports `execution_kind=reference_cpu` until native
  gather/scatter transport kernels are promoted.
- **DK3 tests - landed.** `tests/unit/test_rocm_moe_transport_compiled.py`
  proves dispatch permutation, weighted combine, uneven grouped SwiGLU, and a
  runtime-launch perf baseline against `tessera.stdlib.moe`.
- **DK3 native ROCm transport - open.** Existing `rocm_moe_compiled` remains the
  native top-1 compute lane; the remaining DK3 promotion is HIP dispatch,
  combine, and grouped expert transport that can flip
  `rocm_moe_transport_compiled` from `reference_cpu` to `native_gpu`.

Work:

- Promote routing plan fields to an IR/runtime contract:
  - expert ids;
  - token permutation;
  - group sizes;
  - capacity/drop metadata;
  - combine weights.
- Lower `moe_dispatch` and `moe_combine` to CUDA/ROCm gather/scatter kernels.
- Lower expert compute through grouped GEMM.
- Fuse grouped SwiGLU where shape and dtype allow.

Acceptance:

- Dispatch/combine round-trip matches `stdlib.moe`.
- Grouped GEMM handles uneven group sizes.
- Capacity drop and deterministic ordering are tested.

## DK4 - Dequant-GEMM and Numeric Policy

Close the quantized matrix path for CUDA/ROCm.

Current increment (2026-06-30):

- **DK4 native ROCm generator + runtime ABI — landed.** The
  `rocm_dequant_gemm_compiled` execution-matrix row now reaches
  `runtime.launch()` for `dequant_matmul` and `dequant_grouped_gemm`, consuming
  packed int4/int8 codes + per-group scales through the same ABI a fused kernel
  will use. On ROCm hardware it launches
  `generate-rocm-dequant-gemm-kernel` and reports `execution_kind=native_gpu`;
  without `tessera-opt`/ROCm hardware it falls back to the packed-weight oracle
  and reports `reference_cpu`.
- **DK4 tests — landed.** `tests/unit/test_rocm_dequant_gemm_compiled.py` proves
  single INT4 dequant-GEMM, grouped INT8 expert GEMM, and a runtime-launch perf
  baseline against `tessera.stdlib.quant`; it also includes codegen lowering
  coverage and a hardware-gated native ROCm correctness proof.
- **DK4 native ROCm perf proof — hardware-gated.** The first native fused kernel
  avoids full fp32 weight materialization but is scalar-per-output. The remaining
  promotion is cooperative tiled GEMM, grouped launch batching, and benchmark
  proof against the reference launch baseline on target ROCm hardware.

Work:

- Treat packed codes and scales as first-class operands.
- Do not materialize full fp32 weights in the fused path.
- Define target-specific FP8 semantics:
  - NVIDIA: CUDA/SM capability keyed FP8/NVFP4 policy.
  - ROCm: gfx942 FNUZ vs gfx950 OCP and no-FP8 cases.
- Support INT4 packed-nibble weights with explicit group scales.
- Reuse grouped GEMM metadata for quantized expert compute.

Acceptance:

- `dequant_matmul` and `dequant_grouped_gemm` lower to CUDA/ROCm fused kernels.
- Tests prove parity against reference dequant-then-matmul.
- Unsupported dtype/arch combinations fail loudly.

## DK5 - End-to-End Integration Gate

The integration target is a scaled speculative decode step that exercises the
new stack without requiring full frontier-model hardware.

Current increment (2026-06-30):

- **DK5 scaled gate — landed.** `tessera.speculative.
  dk5_scaled_speculative_decode_gate` composes DS2 draft-block generation,
  target verification, greedy `spec_accept` semantics, and cache
  commit/rollback through the existing cursor trim contract.
- **Lane selection provenance — landed.** `dk5_select_lanes` inspects a model
  config/dataclass/mapping and records the requested ROCm lanes:
  `rocm_exotic_attn_compiled` for DK1 MLA, `rocm_sparse_attn_compiled` for DK2
  sparse attention, `rocm_moe_transport_compiled` for DK3 MoE transport, and
  `rocm_dequant_gemm_compiled` for DK4 quantized weights.
- **DK5 tests — landed.** `tests/unit/test_dk5_scaled_speculative_gate.py`
  forces a partial DSpark acceptance, appends speculative KV entries, proves
  rollback trims rejected draft state, and checks the selected DK lanes.

Gate:

1. DSpark/Gumiho-style draft block proposes up to `block_size` tokens.
2. Target verify runs as a composed target-model call.
3. Native `spec_accept` picks the accepted prefix.
4. Cache commit/rollback updates state.
5. MLA/sparse attention/MoE/quantized grouped GEMM lanes are selected when the
   model config asks for them.

Acceptance:

- CUDA and ROCm each have a scaled execute-and-compare gate.
- The result matches autoregressive reference generation.
- Provenance confirms native backend execution for every claimed fused path.
- Perf ratchets track launch count, accepted tokens per proposal, and per-token
  latency versus the Python host-loop baseline.

## Suggested PR Sequence

1. CF0 docs and diagnostics.
2. CF1 Graph IR ops plus verifier lit tests.
3. CF2 hardware-free Schedule/Tile/Target lowering fixtures.
4. CF3 CUDA proof kernels for scan/for/while/cond.
5. CF4 ROCm proof kernels for scan/for/while/cond.
6. SD1 `spec_accept` op and CUDA/ROCm runtime kernels.
7. DS1 DSpark reference library and tests.
8. DS2 fused draft-block forward on CUDA, then ROCm.
9. DK4 dequant-GEMM CUDA/ROCm, because MoE and model weights depend on it.
10. DK3 MoE dispatch/combine/grouped GEMM.
11. DK1 MLA fused decode.
12. DK2 NSA/DSA/MSA sparse attention.
13. DK5 scaled end-to-end speculative decode gate.

## Risks and Mitigations

| Risk | Mitigation |
|---|---|
| Control-flow ops become too general to lower | Start with static carry shapes, bounded loops, pure bodies, and typed cache effects |
| Backend claims silently fall back to Python | Add provenance assertions and anti-fallback tests for every executable gate |
| DSpark pulls in training complexity before serving value | Land serving proposal/reference first; keep target-cache training as a later library layer |
| CUDA and ROCm diverge semantically | Share Graph/Schedule contracts and use identical numpy reference tests |
| FP8 semantics drift by target | Keep arch-keyed numeric policy tests and fail unsupported paths loudly |
| Apple work distracts from portable closure | Treat Apple as research/follow-up until CUDA/ROCm contracts are stable |

## Decision Points

- Whether `scan`/`fori_loop`/`while_loop` should be Graph dialect ops or reuse
  upstream `scf` directly after tracing. Recommended: Tessera Graph ops first,
  lower to `scf`/backend constructs later, because cache effects and model-level
  provenance need Tessera attributes.
- Whether DSpark should live under `tessera.stdlib.dspark` or
  `tessera.models.dspark`. Recommended: start in `stdlib.dspark` for reusable
  blocks, then add model wrappers once the reference contract is stable.
- Whether top-k/block selection for sparse attention is fused immediately.
  Recommended: keep selection separate until block-sparse kernels pass dense
  equivalence and decode parity.
