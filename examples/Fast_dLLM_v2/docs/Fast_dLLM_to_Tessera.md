# MERGE_START
# Fast‑dLLM v2 → Tessera Programming Model Mapping

**Scope.** Map NVLabs **Fast‑dLLM** (block‑wise **approximate KV Cache** + **confidence‑aware parallel decoding**) into Tessera IR stack and runtime.

## 1) Concepts → Tessera IR

### Graph IR (model & control)
- `tessera.graph.dlmm_step(seed, t, state) -> state'` — one diffusion decoding step (bidirectional attention model call).
- `tessera.graph.kv_cache.block_init(n_blocks, d, policy)` — allocate block‑wise KV cache (approximate).
- `tessera.graph.kv_cache.block_update(state, block_id)` — write/update approximate block contents.
- `tessera.graph.parallel_decode{K}` — fork K speculative branches with shared read‑only cache views.
- `tessera.graph.validate_and_merge(policy)` — confidence‑aware validator; commit longest prefix that passes.

### Schedule IR (where/when)
- Tile‑time blocking: `B_tok` (tokens per block), `H` heads, `D` head_dim, `S` steps.
- Placement: map branch `k` to device stream/queue with **QOS token** from `tessera.schedule.qos.acquire`.
- Dependencies:
  - `cache.block_update` **happens‑before** any `branch[k].attn_read(block)` via `tessera.schedule.dep` edges.
  - `validate_and_merge` waits on `branch[k].done` **and** `attn_stats[k]` reductions.

### Tile IR (compute kernels)
- `tessera.tile.attn_bidir(q, k, v) -> o` — bidirectional attention core (supports block window).
- `tessera.tile.kv_block_pack(k, v) -> k_blk, v_blk` — pack/quantize to approximate block, stride‑aware.
- `tessera.tile.kv_block_read(idx) -> k_blk, v_blk` — fast path for reuse.
- `tessera.tile.confidence_stats(o, logits) -> s` — per‑token confidence (entropy/top‑p margin).
- `tessera.tile.prefix_lcp(tokens[k]) -> lcp_len` — longest common prefix across branches.

> Reuse your FlashMLA kernels for `attn_bidir` with windowed causal masks; add a (small) branch to support **bidirectional** diffusion attention masks and block‑window reads.

## 2) Fast‑dLLM specifics

### 2.1 Block‑wise approximate KV Cache
- Partition sequence into blocks of `B_tok` (paper uses block‑wise cache tailored to diffusion LLMs).
- Approximation: keep low‑rank / quantized summaries per block; retain boundary stripes in full precision.
- Tessera ops:
  - `kv_block_pack`: choose `(dtype_q, dtype_kv)` and optional rank‑r projection; emit metadata `{blk_id, t0, t1, scales}`.
  - `kv_block_read`: returns dequantized views + boundary stripes for current window.

### 2.2 Confidence‑aware parallel decoding
- Spawn K branches, compute per‑step confidence stats, **commit** only tokens up to the **validated prefix**.
- Tessera runtime primitive:
  - `tessera.runtime.decode_policy.confidence{tau, window}` returns a validated prefix length using entropy/top‑p and cross‑branch agreement.
  - `validate_and_merge` commits prefix and **re-bases** surviving branches (share cache pages, COW semantics).

## 3) Runtime & memory

- **Cache arena** (device): ring of KV blocks with ref‑counts; COW on divergence.
- **Allocator keys**: `(layer, head, blk_id)` → page range; per‑page compression `(QBlocks, FP16 stripes)`.
- **Streams**: `ComputeQ` per branch; `IOQ` for kv_pack/writeback; `ReduceQ` for confidence & prefix LCP.
- **Overlap**: schedule `kv_block_pack` of step `t` while step `t+1` reads prior blocks; use `qos.acquire` tokens to cap K.

## 4) Lowering plan

1. Graph → Schedule:
   - Inline `dlmm_step` into per‑layer tiles; create `dep(cache_update → attn_read)`.
   - Expand `parallel_decode{K}` into `branch[k]` regions with cloned subgraphs and shared cache handles.
2. Schedule → Tile:
   - Lower attn to `attn_bidir` with block masks and `kv_block_*` intrinsics.
   - Insert `confidence_stats` + `prefix_lcp` reductions after every Δ tokens.
3. Tile → Target IR:
   - NVIDIA: use WGMMA/FlashMLA tiles for Q·Kᵀ and windowed KV fetch; vectorized de/quant pack.
   - AMD: map to MFMA tiles; LDS‑resident boundary stripes for quality.
   - CPU: AVX2/AMX fallback tiles (smaller `B_tok`, no parallel K>2).
4. Passes:
   - `-tessera-kv-cache-blockify`
   - `-tessera-parallel-decode-expand`
   - `-tessera-attn-bidir-windowing`
   - `-tessera-conf-validate-merge`
   - `-tessera-cow-cache-dedup`

## 5) Interfaces & attrs (sketch)

```
#cache = #tessera.cache<
  blocks = 64, B_tok = 16, approx = {quant=fp8_e4m3, stripes=fp16, rank=0}
>

%h = "tessera.graph.kv_cache.block_init"() {cache = #cache} : () -> !tessera.cache.handle
%st' = "tessera.graph.parallel_decode"(%st, %h) {K = 4} : (!state, !tessera.cache.handle) -> !tessera.branches
%ok  = "tessera.graph.validate_and_merge"(%st') {tau = 0.75, window = 8} : (!tessera.branches) -> !state
```

## 6) FileCheck test ideas

- **Blockify** inserts `kv_block_pack/read` around attention with correct `(t0,t1)` windows.
- **Parallel decode** creates `branch[0..K-1]` regions and joins with `lcp_len` scalar in SSA.
- **Merge** erases committed prefix and updates cache ref‑counts.

## 7) Milestones

- **M0**: single‑branch, block‑cache only; parity with paper’s speedups at K=1.
- **M1**: K=2 with entropy‑threshold validator; correctness on LLaDA eval set.
- **M2**: K=4 + COW cache; throughput & latency dashboards (roofline + Perfetto).
- **M3**: ROCm/AVX2 targets; YAML peaks feed into roofline lines.

# MERGE_END
