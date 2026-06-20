# Contract-Consuming Pass Plan

> Status: active (started 2026-06-19). Turns the 8-item KV/attention/quant/TP/multimodal
> audit into a sequenced engineering plan. Theme: close the recurring meta-gap where a
> typed **contract** exists (handle fields, cost scores, sharding specs, multimodal nodes)
> but **no compiler pass consumes it as an obligation**.
>
> This is a roadmap doc, not a generated dashboard. Counts/status truth still live in
> `docs/audit/generated/`. When a workstream lands, update its row here and flip the
> `contract_consumer` axis in `primitive_coverage.py` (Phase 0).

## The unifying diagnosis

The recurring gap is **not** "the substrate is missing." In most items the substrate —
and often the consuming machinery — already exists; they aren't joined by a typed
contract a pass is obligated to consume. Three precise shapes:

1. **Two substrates, no shared contract** (#1/#6): contiguous `KVCacheHandle`
   (`page_size` is metadata-only per its own docstring, `cache/handle.py:8`) and the
   paged `TieredKVCache.stage/gather/evict` ABI (`cache/tiered.py:209`) both exist;
   nothing unifies them or lets an attention op consume either.
2. **A scorer used as a gate, not a selector** (#3): `FusionCost.score`
   (`compiler/fusion.py:2358`) can rank, but attention lowering branches on a hard
   `Nk <= SYNTH_MAX_N` threshold (`compiler/fusion.py:2107`) and never compares variant
   scores.
3. **A producer missing for an existing consumer** (#4): the backend already consumes
   packed-int4/W8A8 operands (`runtime.py:2914`); what's missing is the *pass that
   produces calibrated operands* (SmoothQuant migration).

And #5: the adjoint collective pass already exists
(`src/transforms/lib/AdjointCollectiveInsertionPass.cpp`); the real gaps are the
**automatic `nn.Linear` rewrite** and **numeric cross-rank gradient equivalence tests**.

Every workstream follows one pattern:

> **Contract (typed object) → Consuming Pass (rewrites/selects) → Oracle (proves it).**

## Dashboard closeout (2026-06-19) — front-to-back hardening

A sweep to clear the small concrete dashboard gaps ahead of AMD/NVIDIA bring-up:

- **2 partial ops → complete** (`e2e_op_coverage`): `clifford_norm_squared` and
  `ebm_energy_quadratic` both had shipped Apple-GPU MSL kernels but missing
  *manifest registration* — added `clifford_norm_squared` to `_CLIFFORD_FUSION_OPS`
  + `_CLIFFORD_APPLE_GPU_FUSED`, and `ebm_energy_quadratic` to `_EBM_PRIMITIVES` +
  `_EBM_APPLE_GPU_FUSED` (same fused kernel as the `ebm_energy` alias). graph_ir
  auto-resolved to `not_applicable`.
- **2 thinly-tested ops cleared** (`test_coverage`): direct numerical tests for
  `kv_cache_prune` + `memory_index_score`. Rippled into a classifier fix —
  `classify_op` now returns `directly_tested` for any op with >1 real reference
  (was mislabeling 263 well-tested ops by category default); the actionable
  `needs_direct_test` bucket is now **0**.
- **conv2d / kv_cache_read** (`op_target_conformance`): conv2d computes on Metal
  via its direct dispatcher (tested) and `apple_gpu_kv_cache_read` lands a genuine
  device-resident (DeviceTensor) read with a provenance gate. The conformance
  "complete" flip was **deliberately not taken** — the `@jit→launch` path for both
  is `unimplemented`, and the conformance Evaluator correctly refuses rung-7
  corroboration; forcing it would be a provenance overclaim. The launch wiring is
  the real conformance closer (tracked follow-on).

The recurring lesson: most "partial" dashboard cells were **registration gaps over
shipped kernels**, not missing capability — and the honesty guards (conformance
Evaluator, envelope-dispatch lane check) correctly blocked the one case
(conv2d/kv_cache_read) where the kernel exists but the launch path doesn't.

## IR contract legality (2026-06-19) — dtype / aliasing / buffer-binding

Closes COMPILER_AUDIT's "Layout and binding contracts are uneven" (layout was
already done). New C++ `IRContractLegalityPass`
(`src/transforms/lib/IRContractLegalityPass.cpp`, `--tessera-ir-contracts`) is
LayoutLegalityPass's sibling — one `ModuleOp` walk, 7 stable-coded rules:

- **dtype** (enforces Decision #15a): `DTYPE_LEGALITY_TF32_AS_STORAGE` (TF32 is a
  math_mode, not storage), `DTYPE_LEGALITY_LOWP_WITHOUT_WIDE_ACCUM` (fp8/fp6/fp4/
  nvfp4/int4/int8 storage must declare a wider accum — storage≠accum),
  `DTYPE_LEGALITY_UNKNOWN_STORAGE`.
- **aliasing**: `tessera.inplace=true` requires an in-range `tessera.aliases`
  (`ALIAS_LEGALITY_MISSING_ALIASES` / `_OPERAND_OOB`).
- **buffer-binding**: `tessera.buffer_role` accept-set + no conflicting role per
  `tessera.binding` (`BUFFER_BINDING_UNKNOWN_ROLE` / `_CONFLICT`).

**Wired into all three named pipelines** (x86 / gpu / CUDA13) after
LayoutLegalityPass, so it fires during real lowering. Lit
`tests/tessera-ir/phase2/ir_contract_legality.mlir` (13 cases) + 12 Python guards;
full tessera-ir lit sweep 148 PASS / 0 FAIL. With layout (done earlier), the
Graph/Schedule/Tile/Target IR contract surface is now closed front-to-back.

## AMD / NVIDIA emission readiness (2026-06-19) — toward bring-up

The single biggest "hardware-gated" bucket every dashboard named is the
NVIDIA/ROCm backend-kernel axis. With the Strix Halo (gfx1151) + Blackwell
(sm_120) boxes coming online, the hardware-free ceiling for each was pushed and
**scored in the Evaluator**:

- **AMD — rung 4 (ASSEMBLES), on this Mac.** `rocdl_emit` already emits
  `llvm.amdgcn.wmma` IR for gfx1100 / **gfx1151** / RDNA4 / gfx1250 + a full
  K-loop GEMM; the LLVM 22 AMDGPU backend means `llc -mcpu=gfx1151` lowers it to
  real `v_wmma_f32_16x16x16_f16 …` AMDGCN **here, no GPU**. New
  `evaluator.rocm_emission_verdict` scores this at rung 4 (parallel to
  `nvidia_emission_verdict`), provenance-honest (never claims execution / rungs
  6–7 without silicon). 86 rocdl tests + 5 new verdict tests.
- **NVIDIA — rung 3 (EMITS_ASM_TEXT) ceiling, here.** `ptx_emit` emits valid
  WGMMA PTX text + structural validation; `ptxas` (rung 4 → SASS) and execution
  (rungs 6–7) need the Linux/CUDA Blackwell box — the genuinely hardware-gated
  remainder the plan leaves open.

The asymmetry is honest: AMD assembles on-host because the AMDGPU backend ships
in LLVM; NVIDIA's SASS assembler does not. Both are at their true hardware-free
max. Remaining = rungs 6–7 on real silicon (multi-GPU / collectives / PeerSync
/ DMA-Buf / NVIDIA low-level), as scoped.

## Phase 0 — Make "is the contract consumed?" an audited axis

> **Status (2026-06-19): LANDED.** `compiler/contract_consumers.py` — one row per
> workstream contract, `status` **probed live** (the probe imports the consumer;
> `live` vs `declared`). Registered as generated doc `contract_consumers`
> (`docs/audit/generated/contract_consumers.{md,csv}`), drift-gated. A/B report
> `live`; C/D/E/F `declared` and flip automatically as their passes land. 8 guard
> tests in `test_contract_consumers.py`.

Without this we re-accumulate orphaned contracts.

- Add `contract_consumer: live | declared | none` to `primitive_coverage.py` metadata,
  populated from a `(contract_type → consuming_pass)` registry. A contract with fields
  but no pass that reads them reports `declared` (today's KV `page_size`) vs `live`.
- Generate `docs/audit/generated/contract_consumers.md`, drift-gated via
  `scripts/check_generated_docs.sh` (Decision #26).
- Each workstream flips its row `declared → live` only when its oracle passes.

## Workstreams

### A — Unifying KV ABI + paged-attention consumer (#1, keystone)

> **Status (2026-06-19): core LANDED.** `cache/paged_kv.py` (`PagedKVState` +
> `PageTier`/`KVKind` + adapters for `KVCacheHandle`/`TieredKVCache` +
> `paged_attention`); `ops.paged_attention` + `flash_attn(kv_state=)`;
> `evaluator.paged_kv_equivalence` differential oracle (residency-schedule
> invariant). 17 ABI tests + 648 attention regression tests green; mypy clean.
> **Follow-on (task #8): LANDED (2026-06-19, Metal 4.0 path).**
> `paged_attention(backend="apple_gpu")` runs the gathered KV through the shipped
> fused matmul→softmax→matmul Metal kernel per head with honest provenance
> (`metal_runtime` only if every head fired, else `reference`).
> `evaluator.paged_kv_native_equivalence` is the native rung — earned only when
> Metal actually ran AND agrees with the numpy reference (provenance-gated, so a
> silent fallback stays `inconclusive`). LATENT (MLA expand via `latent_paged_kv`)
> and QUANTIZED_TAIL (`quantized_tail_paged_kv`, hot fp window + int8 cold tail)
> kinds route through the same consumer. 7 tests in `test_paged_kv_native.py`
> (native parity, provenance gate, latent ≡ expanded-full, quantized-tail ≈ fp);
> mypy clean. The FP8-native perf rung waits on Metal 4.1 / macOS 27.0 — a
> throughput upgrade, not a correctness gate.

**Contract** `python/tessera/cache/paged_kv.py` — `PagedKVState` protocol implemented by
*both* substrates (adapter, not rewrite):
- `page_table` (block tables; degenerate single block for `KVCacheHandle`)
- `tier(page) -> {resident, host, offload}` (`TieredKVCache` already tracks this)
- `quant -> NumericPolicy | None` (reuse `quantize_bits`)
- `sharing -> {block-shared seq ids}` (new; prefix sharing)
- `trim/restore` (already on all handles, commit 48c3db3)
- Heterogeneous kinds (ShadowKV #6): full / latent (MLA) / low-rank / quantized-tail,
  all sharing the protocol.

**Consuming op + pass**: `ops.flash_attn` gains `kv_state=`; new
`PagedAttentionLoweringPass` reads the contract and inserts **prefetch → gather →
dequant** stages driven by `tier`/`quant`.

**Oracle**: `evaluator.py` `cross_path_equivalence` — paged == contiguous for the same
logical sequence, across f32/f16 and contiguous/tiered/quantized-tail/MLA-latent.

**Done when**: `paged_attention` row = `contract_consumer=live`; all kinds pass the
metamorphic oracle on Apple GPU.

### B — Prefill and decode as different compiled programs (#2)

> **Status (2026-06-19): LANDED.** `compiler/phase_specialization.py` (`Phase`,
> `SLO`, `SchedulePolicy.for_phase` — prefill=bulk_throughput/materialize vs
> decode=low_latency/resident-pinned; `CacheHandoff` carrying a `PagedKVState`;
> `specialize` + `PhaseSpecializedProgram`; `verify_phase_split` oracle).
> `@jit(phase=, slo=)` attaches the policy. 16 tests green (oracle across
> seeds/lengths proves prefill▸decode ≡ monolithic forward); mypy clean.

Depends on A. `@jit(phase="prefill"|"decode", slo=...)` + a `CacheHandoff` ABI that *is*
a `PagedKVState`. `PhaseSpecializationPass` emits two programs: prefill = throughput
schedule, decode = latency schedule. Oracle: `prefill ▸ decode_loop == forward`.

### C — Promote IO cost model from gate to selector (#3)

> **Status (2026-06-19): LANDED.** `fusion.select_attention_lowering` +
> `attention_lowering_costs` score materialized/online/reference by off-chip
> bytes (the FA currency); `paged_stage_bytes` feeds page-staging cost from a
> Workstream-A PagedKVState. The hard `Nk <= SYNTH_MAX_N` branch in
> `run_fused_attention` now routes through the selector (behavior-preserving).
> 29 tests (cost-monotonicity, feasibility invariant, crossover-at-cap,
> staging-bytes, numerical preservation); 920 fusion/attention tests green.

Contract exists (`FusionCost` + Schedule IR `bytes_moved`/`flops`,
`compiler/schedule_ir.py:292`). Add `select_attention_lowering()` scoring every
candidate `{materialized, online, tiled, paged-gather, reference}` by total bytes
(incl. cache reads, page gathers, offload transfers from A, command-buffer syncs),
picking min feasible. Replace the hard branch at `compiler/fusion.py:2107`. Oracle:
cost-monotonicity + feasibility invariant.

### D — SmoothQuant activation-scale migration pass (#4)

> **Status (2026-06-19): LANDED.** `compiler/smoothquant.py` —
> `migrate_activation_scale` folds per-channel activation scale into weights
> (`s_j = max|X|^α/max|W|^(1-α)`) and emits int8 W8A8 operands;
> `smoothquant_matmul` runs the direct-consume int8×int8→int32 path;
> `verify_w8a8` oracle proves fp parity + the **anti-fallback invariant**
> (operands stay int8). 8 tests incl. exact-fp factorization + beats-naive on
> outliers; mypy clean.

Direct-consume already works; the gap is the producer. `ActivationScaleMigrationPass`
folds per-channel activation scale into weights (from `CalibrationObserver`) and emits
calibrated W8A8/int4 operands consumed by the existing direct-consume kernel. Oracle:
W8A8 parity vs fp16 + **anti-fallback assertion** (provenance proves no dequant-then-GEMM).

### E — Megatron TP rewrite + cross-rank gradient equivalence (#5)

> **Status (2026-06-19): LANDED.** `compiler/tensor_parallel.py` —
> `rewrite_linear(W, TPSpec)` auto-rewrites a plain linear into column/row/
> sequence parallel with the correct collectives (E1); `ParallelLinear`
> forward/backward run over `MockRankGroup` threads. `verify_tp_gradient_
> equivalence` is the E2 oracle — sharded dX/dW recombined equal single-rank
> gradients to ~1e-9. 20 tests across 3 modes × world sizes 2/4 + rectangular;
> mypy clean. Sequence-parallel (the declared-not-consumed `cyclic`) is now wired.

Adjoint collective insertion exists. **E1**: `TensorParallelRewritePass` lowers plain
`nn.Linear` → col/row-parallel + collectives from a `TPSpec`; wire sequence parallelism
(`cyclic`, declared-not-consumed). **E2** (the precise missing oracle): numeric
cross-rank gradient equivalence via `MockRankGroup` — sharded grad == single-rank grad
for col/row/seq-parallel.

### F — Named multimodal walks + encoder-free ops (#7/#8)

> **Status (2026-06-19): LANDED.** `compiler/model_walk.py` — `ModelWalk` +
> `partition_walks` split the MiniMax-M3 graph into named entry points
> (vision_prefill / video_prefill / splice); `walks_reconstruct_graph` proves the
> partition is lossless. First-class encoder-free ops (`patch_embed`,
> `coordinate_lookup`, `audio_frame_projection`, `splice_embeddings`) + an
> executable `EncoderFreeVLM` whose named walks recompose into the monolith,
> proven by `verify_walk_parity`. 13 tests; mypy clean.

MiniMax-M3 graph builder + JEPA tests exist. `ModelWalk` contract: named entry points
(`vision_prefill`, `text_decode`, `image_gen`) over existing nodes, each a separately
compiled program sharing cache state via A+B. `WalkPartitionPass` splits the graph; flip
`vision_execution_supported` (native patch path) and add first-class **audio frame
projection** + **coordinate/position-lookup** ops. Oracle: per-walk parity vs full
forward.

## Sequencing

```
Phase 0 ──► everything
           ┌─► B ──┐
A ─────────┤        ├─► F
           └─► C ──┘
D  ── independent (parallel)
E  ── independent (parallel)
```

Critical path: 0 → A → {B, C} → F. D and E run parallel off Phase 0. C can begin on
compute/materialization bytes before A, gaining page-gather/offload vocabulary when A
lands.

**Locally provable now** (Apple GPU + `MockRankGroup` + evaluator oracles): A, B, C, D, F
fully; E's rewrite + numeric equivalence via mock ranks. **Hardware-gated** (Phase
G/H/I): only real multi-GPU TP throughput and NVIDIA/ROCm kernel rungs — every pass's
*correctness* is provable here.

Every workstream lands its oracle into `evaluator.py` so "the pass consumes the contract
correctly" is a derived, drift-gated fact, not prose.
