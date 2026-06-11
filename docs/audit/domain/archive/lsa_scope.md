# LSA (Lookahead Sparse Attention) — scope lock

> Status: **experimental, inference-only, v1.** Provenance/archive doc (per
> Decision #26 `*/archive/` is provenance, not the live status surface). The
> live status surface is `docs/audit/domain/DOMAIN_AUDIT.md` and the generated
> coverage dashboards.

This note records the scope decisions that bound the LSA op family before any
IR work, in the spirit of the S0 scope lock. It exists so a later reader can see
what was deliberately *in* v1 versus deferred, and why.

## What LSA is

Two new standalone primitives layered on the existing attention surface — **not**
a replacement for the `flash_attn` kernel:

- `memory_index_select` — sigmoid-threshold selection over compressed historical
  block keys. Union across indexer layers; empty-selection fallback to the
  query's own block. Reference: `python/tessera/lsa.py`.
- `lookahead_sparse_attention` — composite attention *policy*: each query attends
  over the union of its causal local window and the tokens of the selected
  historical blocks. Explicit composition of local-window + selected-block
  attention; reference: `python/tessera/lsa.py`.

The closest existing anchor is `deepseek_sparse_attention` (host-mediated
data-dependent selection + GPU dense attention); LSA reuses that lane shape.

## Decisions (D1–D5)

| ID | Decision | Rationale |
|----|----------|-----------|
| **D1** | Ship as `lookahead_sparse_attention` + `memory_index_select`. **No "FlashMemory" branding** in code / catalog / audit until the KV-tiering substrate exists. | Avoids a Decision #25 "registry claims more than the runtime proves" drift. The name must not imply memory-hierarchy behavior that v1 does not ship. |
| **D2** | The op is **pure per call**. `tau` / `threshold` / `window_size` / `block_size` are attributes; one forward call performs exactly one selection. The every-`tau` lookahead *cadence* is owned by the caller's decode loop. | Matches the stateless-per-call grain of every existing attention op; keeps autodiff and conformance tractable. |
| **D3** | `memory_index_select` is a **new primitive** — sigmoid-threshold boolean retrieval with union-across-layers. It does **not** reuse `memory_read` (top-k + softmax). | The genuinely novel piece. `memory_read` cannot express threshold retrieval, so a new op is required (not a wrapper). |
| **D4** | v1 selection is **host-mediated + data-dependent**, identical to the `deepseek_sparse_attention` Apple-GPU lane. **No CPU cold-pool ↔ GPU-resident KV tiering** — deferred. | The tiering substrate (Phase E KV paging, real `schedule.prefetch` overlap) does not exist. v1 stays on proven rails. |
| **D5** | `tau=64` / `threshold=0.5` are **chosen test fixtures**, not a reproduced external result. No paper-equivalence claim is written into any audit doc. Status stays `planned` for `backend_kernel` until real hardware kernels exist. | Decision #25 / #27 — ground claims in executed oracle equivalence, not citations. |

## Deferred gaps — all closed 2026-06-11

These were tracked gaps (Decision #21/#25 — named, not silent omissions). All
four have since landed; the strikethrough entries record what closed them.

- ~~**CPU cold-pool ↔ GPU-resident KV tiering**~~ — **LANDED 2026-06-11**
  (`python/tessera/cache/tiered.py`). `TieredKVCache` holds every KV page in a
  host cold pool and a bounded set of pages in device-resident `DeviceTensor`
  buffers; `stage`/`evict`/`gather` are the host↔device staging ABI (real
  cold→resident copies, LRU eviction, byte accounting).
  `lookahead_attention_tiered` drives staging from the `memory_index_select`
  output and is numerically identical to the non-tiered oracle while being
  independent of `resident_capacity`. Guards: `tests/unit/test_lsa_tiered_kv_cache.py`
  (12). This is the piece that earns the "FlashMemory" tiering story; the op is
  no longer *just* lookahead-periodic sparse attention.
- ~~**Real `schedule.prefetch` overlap**~~ — **LANDED 2026-06-11.** The
  `tpp-async-prefetch` pass (`src/solvers/tpp/lib/Passes/AsyncPrefetch.cpp`) is no
  longer a no-op: it software-pipelines `schedule.prefetch` ops — rotating
  double-buffer stages + dependency-safe hoist of overlap-policy prefetches above
  preceding compute. An `into="host"` / `overlap="none"` prefetch (how LSA's
  cold-pool staging is *recorded*) is annotated but never overlapped/hoisted — no
  overlap semantics claimed, per this gap's contract. Lit:
  `src/solvers/tpp/test/TPP/async_prefetch_overlap.mlir`; Python:
  `tests/unit/test_lsa_prefetch_overlap.py`. **Follow-on LANDED 2026-06-11** — the
  LSA Graph→Schedule lowering now exists: `tessera-lookahead-sparse-prefetch`
  (`AttentionFamilyPasses.cpp`) emits a `schedule.prefetch{into="host",
  overlap="none"}` for each `tessera.lookahead_sparse_attention` op and rewires
  the op to consume it (a true dataflow edge), making the cold-pool KV staging a
  first-class IR value. Running it then `-tpp-async-prefetch` is the end-to-end
  Graph→Schedule→overlap flow; the host prefetch is recorded without an overlap
  claim (matching the synchronous Gap-1 runtime — a backend that stages
  asynchronously flips `overlap="compute"` and the real pass software-pipelines
  it). Lit: `tests/tessera-ir/phase8/lsa_prefetch_lowering.mlir`.
- ~~**Indexer-key training**~~ — **LANDED 2026-06-11** (the differentiable
  surface; the training *loop* still lives in user code, as intended).
  `memory_index_score` is the indexer's scoring head — `sigmoid(q·kᵀ·scale)` with
  a closed-form VJP+JVP (finite-difference verified), so the indexer keys are
  trainable. `memory_index_select_ste` is hard-forward / straight-through
  (sigmoid) backward for training under hard selection. The hard
  `memory_index_select` stays non-differentiable for inference.
  `tests/unit/test_lsa_indexer_training.py` (6) includes a gradient-descent loop
  that trains the indexer to select a target block.
- ~~**Fused GPU LSA kernel**~~ — **LANDED 2026-06-11.**
  `tessera_apple_gpu_lookahead_sparse_attn_f32` (MSL kernel +
  host-reference fallback in `apple_gpu_runtime.mm` / `_stub.cpp`) collapses the
  host-select bmm + mask-add + softmax + bmm (4 GPU dispatches) into ONE MSL
  dispatch: per (head, query) it computes `softmax(scale·Q·Kᵀ + mask)·V` over the
  padded footprint (cap T ≤ 256, else the multi-dispatch path). The runtime LSA
  lane prefers it; matches the oracle (~5e-7) and the multi-dispatch fallback.
  Guards: `tests/unit/test_lsa_fused_gpu_kernel.py`.

## Conformance posture

The numpy oracle in `python/tessera/lsa.py` *is* the contract. The Graph IR op,
the autodiff rules, and the Apple-GPU runtime lane are each validated against it
at fp32 tolerance. No "production MLIR/LLVM" or external-equivalence status is
claimed until oracle equivalence **and** executed codegen both exist.
