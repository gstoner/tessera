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

## Explicitly deferred (named, not dropped)

Per Decision #21/#25 these are tracked gaps, not silent omissions:

- **CPU cold-pool ↔ GPU-resident KV tiering** — the part that would earn a
  "FlashMemory" name. Needs Phase E KVCache paging (`page_size` is a recorded but
  unused skeleton today) + a host↔device KV staging ABI.
- **Real `schedule.prefetch` overlap** — the op verifies but its lowering
  (`src/solvers/tpp/lib/Passes/AsyncPrefetch.cpp`) is a no-op. v1 may *record*
  prefetch intent as metadata but must not claim overlap semantics.
- **Indexer-key training** — v1 is inference-only; the indexer learning loop
  stays outside the compiler.
- **Fused GPU LSA kernel** — revisit only if the host-select round-trip is a
  measured perf wall.

## Conformance posture

The numpy oracle in `python/tessera/lsa.py` *is* the contract. The Graph IR op,
the autodiff rules, and the Apple-GPU runtime lane are each validated against it
at fp32 tolerance. No "production MLIR/LLVM" or external-equivalence status is
claimed until oracle equivalence **and** executed codegen both exist.
