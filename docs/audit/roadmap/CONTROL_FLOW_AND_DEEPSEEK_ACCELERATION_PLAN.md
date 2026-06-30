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
- `docs/apple_gpu_control_flow_lowering.md` documents the Apple Phase-G ladder:
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
- **SD1-4+ (open):** `target_verify` I/O contract; fully-device-executed Gumiho
  loop with the draft/target models as device kernels (DS/DK-track integration).

## DS1 - DSpark Library Contract

Add DSpark as a library/model-family contract before attempting fused kernels.

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

## DS2 - Fused Draft-Block Forward

This is the first DSpark performance payoff.

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
