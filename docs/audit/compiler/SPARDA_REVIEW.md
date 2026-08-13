---
last_updated: 2026-08-12
audit_role: plan
plan_state: open
---

# SparDA — Review, Mathematical Verification, and Tessera Extraction

> **Routing:** start at [`README.md`](README.md). This document owns the SparDA
> source review and the extraction decisions that follow from it; global
> ordering lives only in
> [`INTEGRATED_COMPILER_PLAN.md`](INTEGRATED_COMPILER_PLAN.md).
> `MASTER_AUDIT.md` + generated dashboards stay status truth (Decision #26);
> this is a provenance + build-sequence document, not a status claim.

**Status:** review (2026-08-12). **Sources — three, all read:**

1. **Paper** — *SparDA: Sparse Decoupled Attention for Efficient Long-Context
   LLM Inference*, Fu, Xiao, Dong, Han, Villa (NVIDIA / MIT), arXiv 2606.04511,
   submitted 2026-06-03. Read via arXiv abs + HTML.
2. **Code** — [`NVlabs/SparDA`](https://github.com/NVlabs/SparDA) @ `c8e235b`
   ("Initial release"), deep inspection of kernels, tests, benchmarks, training.
3. **Third-party review** — zhongzhuzhou.org technical review, 2026-06-24.

**Verification:** every mathematical claim reused below is machine-checked by
`tests/unit/test_sparda_contracts.py` (pure numpy, no new deps) against the
**code's** semantics, not the paper's prose — the two differ in places recorded
in §1.4.

**Bottom line for Tessera:** SparDA needs essentially **no new math ops**. Its
demands are a streaming compressed-key cache, an index-producing top-k with set
difference, and a cross-layer data-dependent host↔device prefetch edge — i.e.
it is a **scheduling and memory-space problem**, which is the layer Tessera
already makes first-class. The reusable extractions are §3; the explicit
non-extractions are §5.

---

## Part I — What SparDA Is

### I.1 Problem and mechanism

Long-context inference with the KV cache offloaded to CPU DRAM. Two costs grow
badly: block **selection** is O(T²) in prefill and dominates per-step decode
cost, and PCIe fetch latency sits on the critical path because the blocks layer
`l+1` needs are only known once layer `l+1`'s query exists.

SparDA adds a **Forecast projection** `F_l` — one head per GQA group, <0.5%
added params, backbone frozen — trained to predict which KV blocks **layer
`l+1`** will attend to:

```
(Q_l, K_l, V_l, F_l) = φ_l(X_l)                                        (3)
B_{l+1} = B_init ∪ B_local ∪ f_top(F_l · K̃_{l+1}ᵀ, k)                  (4)
O_{l+1} = Attn(Q_{l+1}, K_{l+1}[B_{l+1}], V_{l+1}[B_{l+1}])            (5)
```

That buys two things: a **softmax-free indexer** (one head instead of G
softmax-normalized head scores) and a **one-layer lookahead** that turns block
selection into a schedulable signal, so CPU→GPU prefetch overlaps layer `l`'s
compute.

Substrate is InfLLM-V2 block-sparse attention: block 64 tokens, compressed keys
mean-pooled with kernel 32 / stride 16, block sets `B_init ∪ B_local ∪ B_topk`.

### I.2 Provenance — what NVIDIA actually wrote

The repo is an **NVIDIA-curated integration**, not a from-scratch system:
OpenBMB's InfLLM-V2 CUDA kernels (Apache-2.0) + THUNLP's NOSA offload engine
(MIT) + a vendored FlashAttention-2 fork (BSD-3), with the Forecast decoupling
added on top. Nearly every file carries a "copied/adapted from" header. Read
performance claims with that split in mind: the **kernel** engineering is
largely inherited; the **decoupling** is the contribution.

### I.3 Reported results

| Metric | Value | Character |
|---|---|---|
| Prefill (128K, batch 4) | **1.25×** vs offloaded sparse | like-for-like |
| Decode (128K, batch 8) | **1.69×** vs offloaded sparse | like-for-like |
| Selection cost | **2.50×** cheaper at 128K | component measurement |
| "Effective throughput" | **5.3×** | **capacity unlock, not speedup** |
| Accuracy | ≥ sparse baseline everywhere; **+6.5** NOSA reasoning | see §2.3 |

### I.4 Paper-vs-code divergences (found by inspection)

1. **Forecast is not fused into QKV in the training/HF code** — it is a separate
   `nn.Linear`. Fusion into one `qkvqf` GEMM happens only in the NOSI deployment
   engine. The paper's "fourth output of the QKV projection" is the deployment
   form.
2. **The inference indexer is not just softmax-free, it is scale-free** — a raw
   `bmm` with no `1/√d`.
3. **The KL rest-bucket is finer than described** — implemented over `k·4+1`
   compressed-key sub-buckets (4 per block) with init/local blocks *excluded*,
   not the paper's block-level `(k+1)`.
4. **Prefill prefetch is a plain side-stream offload**, not the persistent
   kernel; the persistent UVA kernel is decode-only.
5. **A bitonic top-k CUDA kernel ships but is unused** — the model path calls
   `torch.topk` everywhere.

---

## Part II — Mathematical Verification

### II.1 Verified sound (checked against code, machine-checked where marked ✅)

- ✅ **Pooling index arithmetic is exact.** The Python remap to
  (kernel 5, stride 4, padding 1) over the compressed axis is *precisely* the
  set of compressed keys whose 32/16 windows overlap a 64-token block — block
  `b` ↔ compressed indices `[4b−1, 4b+3]`. Verified over 200 blocks.
- ✅ **The legacy stage-1 causal rule is strictly causal.** `col < (t−16+1)/16`
  with **C-truncating** division is equivalent to "the compressed key's entire
  32-token window lies in the past", via `⌊(t−31)/16⌋ + 1 = ⌊(t−15)/16⌋`.
  Verified exhaustively for `t < 4096`. **Reimplementation trap:** Python's
  floor division gives a different answer for small `t`; the test pins this.
- ✅ **Softmax elision is rank-equivalent.** Training matches a softmax
  distribution; inference takes top-k of raw logits. Softmax is strictly
  monotone per row, so `top-k(softmax(s/√d)) == top-k(s)`. 500 random trials.
- ✅ **Decode compressed-key lag is covered, with a closed form.** With
  full-window-only compression the uncovered tail is exactly
  `(seqlen − kernel) mod stride`, hence bounded by **`stride − 1` = 15**
  tokens — far inside the forced local window (1024–2048).
  *Derivation:* with `n = ⌊(L−K)/S⌋ + 1` full windows the last covered token is
  `(n−1)S + K − 1`, so the gap is `L − K − S⌊(L−K)/S⌋ = (L−K) mod S`.
  **Two looser readings are wrong** and the test pins both: `kernel−1 = 31`
  ignores that a window forms every `stride` tokens, and `kernel−1+stride = 47`
  double-counts — a bound that holds but is never attained. An earlier draft of
  this review asserted 47; the tight test caught it, which is the argument for
  executable contracts over prose in the first place.
- **Stage 1 is an exact two-pass online softmax** — fp32 stats, max
  subtraction, exp2 trick; pass 1 computes stats with no output write, a quad
  allreduce forms the denominator, pass 2 replays QK and emits normalized
  probabilities, group-summed by a warp butterfly.
- **Stage 2 is bit-exact dense attention restricted to the selected block
  set.** Selection is the *only* approximation in the system.
- **Permuted-slot attention is sound** — RoPE is pre-applied to K and softmax
  is permutation-invariant.

### II.2 Approximate by design (visible in code, honestly)

- Compressed-key windows straddle block boundaries, leaking ≤31 **future**
  tokens into *selection scores* — never into attention output.
- The fine (2,1) → coarse (32,16) KL target regrid is a **deliberately
  non-exact** pooling heuristic; the code documents that exact alignment was
  tried and performed worse.
- In the decoupled steady state there is **no misprediction correction**: the
  prefetched set *is* the attended set. Forecast misses are a silent accuracy
  cost, not a staleness bug. Only layer 0 runs a blocking correction path.

### II.3 Two causal conventions, never reconciled — the design lesson

The codebase carries **two different causal rules** for compressed-grid scores:

| Path | Rule | Future leakage |
|---|---|---|
| Legacy stage 1 | `col < (t−15)/16` | none (whole window past) |
| Decoupled q_future | `16j ≤ q_pos` | up to 31 tokens |

Both are defensible; neither is named as a choice anywhere. **This is the
strongest single design input for Tessera:** the causal convention of a
compressed-grid scoring op is a *semantic key* and must fail closed
(Decision #21a) — `window_fully_past | window_start_past`, no default. Two
conventions selected by which code path you happen to be in is exactly the
condition that decision exists to prevent.

### II.4 No theory at all

The central premise — that layer `l`'s hidden state linearly predicts layer
`l+1`'s block-attention pattern — has **no error bound, no recall guarantee,
and no characterization of failure modes**. The paper reports **no selection
precision/recall metric anywhere**, and never quantifies misprediction rate or
tail latency. Notably the code *has* the instrumentation
(`record_true_miss_stats`, hit/true-miss/wasted-reload counters) — it was built
and never reported.

### II.5 Claims to handle with care

- **5.3× is a capacity unlock**, not a like-for-like speedup: SparDA at batch
  128 vs a baseline that OOMs there. Honest in the abstract's phrasing;
  secondary reporting will inflate it.
- **"No accuracy loss" conflates two changes** — a genuinely better *learned
  selector* and the system change. The +6.5 NOSA reasoning gain is the more
  interesting result and the paper undersells it.
- **Generality is asserted, not shown** — one substrate family (InfLLM-V2 both
  times), one scale (8B), one interconnect regime (PCIe). NVLink-class hosts
  would change the offload calculus entirely.

### II.6 Failure modes all fail *open* — the negative-space lesson

Every defect found by inspection fails open rather than closed:

| Site | Behavior |
|---|---|
| Stage-1 wrapper | blanket `NaN → 0` sanitization; would mask real kernel NaNs |
| `diff_offload` | silently drops a block when slot assignment misses; attention proceeds on a divergent set |
| `bwdIterator` bit-scan | `1ULL << 64` when `q_bit_pos + target ≡ 63 (mod 64)` — C++ UB, only accidentally correct because PTX `shl.b64` clamps. The **forward** iterator guards this case; the backward one does not |
| Event choreography | one shared event, correctness leans on unstated snapshot + host-order semantics |
| `num_k_heads = 2` | hardcoded in the blockmask setup |

The hot path is mature (CUDA graphs, event discipline, zero-sync device-side
counters); the edges are research-grade. This makes SparDA a useful **negative
validation** of Decisions #13/#21a/#30 — these are precisely the failures those
rules exist to prevent.

---

## Part III — Extractions for Tessera

### III.1 Core compiler

1. **Stats-emitting attention is doubly confirmed.** SparDA's stage 1 is
   "attention-scores-with-stats + group-reduce" — the same substrate as the
   `flash_attn_stats` / `softmax_merge` pair in
   [`BLOCK_ATTNRES_ROCM_PLAN.md`](BLOCK_ATTNRES_ROCM_PLAN.md) §III.1. Two
   independent 2026 systems reduce to it.
   **Design consequence:** they are *one primitive family with two result
   modes* — `(o, m, ℓ)` for value-weighted consumers (AttnRes, ring/context
   parallel), and `(m, ℓ)` + replay for probability emission (block scoring, KL
   teachers). Model the second as an attribute on the same op, not a second op
   (Decision #31: one implementation per boundary).
2. **Bitmask-driven block-sparse iteration.** A uint64 mask operand plus a
   `max_no_larger` bit-scan makes the KV loop *leap* over unselected tiles — no
   index list, no gather buffer. Complementary to the index-list approach in
   the existing DSA lane; a Tile IR representation choice the arbiter can pick
   per shape (Decision #28).
   **Caveat found by inspection:** the leap is only in the steady-state loop.
   The masking-phase loop still loads V tiles for skipped blocks, so boundary
   tiles pay bandwidth even when masked. Any cost model must capture both
   regimes.
3. **GQA-fold-to-rows layout transform.** Folding the 16-head group into the
   sequence dimension makes the per-(kv-head, token) mask tile-uniform and turns
   the cross-head reduce into an intra-tile row reduction — a layout rewrite
   expressible with the Decision #15a `layout` attribute.
4. **Cross-layer prefetch as a Schedule IR edge.** An effect-carrying
   `prefetch(blocks, layer+1)` whose dependence edge skips one layer body, with
   a wait at consumption — structurally `tile.async_copy` / `wait_async`
   staging at host↔device granularity with *data-dependent* transfer sets.
   The overlap-feasibility arithmetic (bytes-per-layer ÷ PCIe BW vs compute
   window) is a **static legality check per (batch, context, k, block) bucket**
   — the analytical model the third-party review faults SparDA for lacking
   (its CTA budgets are hand-tuned for H100 only).
5. **Memory-space precedents.** Pinned host DRAM as a first-class KV tier is a
   concrete two-tier use case. The **reserved-tail-slot** discipline is worth
   copying: a fixed tail slot holds the growing partial block, is protected
   from eviction, and flushes D2H every 64 steps.

### III.2 Cache-engine state machine (for a `KVCacheHandle` extension)

Extracted with the invariants made **explicit** — in SparDA every one of these
is emergent and asserted nowhere:

| Element | Semantics | Invariant to assert |
|---|---|---|
| Device cache | exactly `topk` block slots; `block_map[H,B,topk]` slot→CPU block | per-row ids unique |
| Tail slot | fixed index `topk−1`, host-tracked `tail_len` | always writable; maps to current partial block |
| Append | token written at `tail·64 + tail_len` | precedes any eviction this step |
| Delta-fetch | k-th new miss → k-th evictable slot | `#new-misses == #evictable`; tail never evictable |
| Tail flush | D2H at CPU offset `seq−64`; `block_map[tail] += 1` | `seq ≡ 0 (mod 64)` |

The eviction-safety chain is elegant once stated: forced-local keeps the
partial block in every selection → its id always hits → the tail slot is never
evictable → appends never race with loads. **Every link is emergent**; the only
backstop is a post-sync tail overwrite that fails by silently dropping a block.

Limitations not to inherit: batch-uniform `cache_lens` (no ragged decode), a
host-side tail-flush branch their own TODO admits CUDA Graphs cannot capture,
and wholesale duplication of the update method to dodge one Python branch.

### III.3 TSOL / stdlib

Cheapest landing spot is **extending the existing DSA lane** in
`python/tessera/stdlib/attention.py` — SparDA's indexer is lightning-indexer
shaped, which `dsa_block_index` already models. New pure-numpy pieces:

- `compress_keys(K, kernel, stride)` — strided-window mean pool with
  **full-window-only** ragged semantics as an explicit contract, plus the
  incremental ring-buffer update for decode.
- Softmax-free forecast scoring + forced init/local injection (`+inf` into
  pooled scores) feeding the existing `dsa_select_blocks`.
- `block_set_diff` for delta-fetch (previous vs new index sets).

Oracles come nearly free: select-all ≡ dense (already the DSA DESIL cross-path),
plus two new metamorphic invariants SparDA's math hands us — **monotone-
transform invariance** of top-k and **slot-permutation invariance** of output.

### III.4 Benchmarks

- **Counter set** worth adopting for any offload/prefetch work: hit rate,
  true-miss, wasted-reload, bytes — as *device-side* accumulators read only at
  report time (zero-sync profiling). Metadata alongside the stable Decision #12
  schema; **not** a schema change.
- **Conformance pattern:** sha256 input fingerprints + generated-token dumps
  per run for cross-backend output equivalence. Cheap; fits the F4 /
  conformance-evaluator machinery.
- **Controlled-miss-ratio copy benchmarking** — the offload analog of the
  AOT-vs-JIT cache-control lesson (Decision #26a): control the cache state or
  you measure the wrong thing.

### III.5 Tests

Adopt: the metamorphic invariants above; ragged-tail contract tests (seqlen not
a multiple of kernel/stride/block); per-batch `cache_lens` variation; TP
artifact-equality across simulated ranks (matches the Decision #6 thread-based
mock-rank style).

Do **not** inherit SparDA's gaps: print-heavy scripts with 2e-2 bf16 tolerances
instead of asserting tests, and **zero automated coverage of the async
pipeline** — the paper's headline systems contribution has no test at all.
Nothing covers prefetch-miss behavior, empty top-k, slot recycling, or the
persistent copy kernel.

---

## Part IV — Phasing

| Phase | Deliverable | Where |
|---|---|---|
| 0 *(this PR)* | This review + `tests/unit/test_sparda_contracts.py` (4 machine-checked contracts) | any box |
| 1 | `compress_keys` + `block_set_diff` + forecast scoring in the stdlib DSA lane; monotone/permutation metamorphic oracles | any box |
| 2 | `causal_convention` as a fail-closed semantic key on compressed-grid scoring (Decision #21a) | any box |
| 3 | Bitmask block-sparse iteration as a Tile IR alternative to index-gather; arbiter selects per shape | Strix Halo |
| 4 | Cross-layer prefetch edge + static overlap-feasibility check, folded into effect-aware overlap | Strix Halo |

Phases 0–2 are pure Python and land the whole mathematical contract; 3–4 are
where it earns performance, on the fleet's only executing non-Apple GPU lane.

---

## Part V — Explicit non-extractions

Recorded so nobody re-litigates them:

- **The persistent UVA Triton copy kernel** — torch/Triton-coupled and
  substrate-specific; the *pattern* (persistent CTA pool, runtime task count,
  CTA budget as SM-contention control) is captured in §III.1.4, the code is not
  reusable under Decision #23.
- **The InfiniGen comparison path** — a baseline reimplementation with no
  Tessera consumer.
- **NOSA's CIS lane** (learned per-token importance as an additive pre-softmax
  logit bias) — rides `attn_bias`, which Tessera already has; the learned-score
  half is a model change, not a compiler capability.
- **Vendored HELMET / RULER / LongBench suites** — evaluation harnesses, out of
  scope.
- **The whole PyTorch/HF/flashinfer runtime surface** — reimplement or ignore
  (Decision #23); nothing here is wrappable.
