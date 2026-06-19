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

## Phase 0 — Make "is the contract consumed?" an audited axis

Without this we re-accumulate orphaned contracts.

- Add `contract_consumer: live | declared | none` to `primitive_coverage.py` metadata,
  populated from a `(contract_type → consuming_pass)` registry. A contract with fields
  but no pass that reads them reports `declared` (today's KV `page_size`) vs `live`.
- Generate `docs/audit/generated/contract_consumers.md`, drift-gated via
  `scripts/check_generated_docs.sh` (Decision #26).
- Each workstream flips its row `declared → live` only when its oracle passes.

## Workstreams

### A — Unifying KV ABI + paged-attention consumer (#1, keystone)

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

Depends on A. `@jit(phase="prefill"|"decode", slo=...)` + a `CacheHandoff` ABI that *is*
a `PagedKVState`. `PhaseSpecializationPass` emits two programs: prefill = throughput
schedule, decode = latency schedule. Oracle: `prefill ▸ decode_loop == forward`.

### C — Promote IO cost model from gate to selector (#3)

Contract exists (`FusionCost` + Schedule IR `bytes_moved`/`flops`,
`compiler/schedule_ir.py:292`). Add `select_attention_lowering()` scoring every
candidate `{materialized, online, tiled, paged-gather, reference}` by total bytes
(incl. cache reads, page gathers, offload transfers from A, command-buffer syncs),
picking min feasible. Replace the hard branch at `compiler/fusion.py:2107`. Oracle:
cost-monotonicity + feasibility invariant.

### D — SmoothQuant activation-scale migration pass (#4)

Direct-consume already works; the gap is the producer. `ActivationScaleMigrationPass`
folds per-channel activation scale into weights (from `CalibrationObserver`) and emits
calibrated W8A8/int4 operands consumed by the existing direct-consume kernel. Oracle:
W8A8 parity vs fp16 + **anti-fallback assertion** (provenance proves no dequant-then-GEMM).

### E — Megatron TP rewrite + cross-rank gradient equivalence (#5)

Adjoint collective insertion exists. **E1**: `TensorParallelRewritePass` lowers plain
`nn.Linear` → col/row-parallel + collectives from a `TPSpec`; wire sequence parallelism
(`cyclic`, declared-not-consumed). **E2** (the precise missing oracle): numeric
cross-rank gradient equivalence via `MockRankGroup` — sharded grad == single-rank grad
for col/row/seq-parallel.

### F — Named multimodal walks + encoder-free ops (#7/#8)

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
