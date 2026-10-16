---
last_updated: 2026-10-16
audit_role: reference
---

# Native NVFP4 bounded source program

Owner: ROCM-NVFP4-INGEST-1 / FRONTEND-IR-MEDIUM-1 / E2E-REAL-6.
Sync: NVFP4-SOURCE-BOUNDED-20261016.

Status: native export and member projection implemented; runtime owner and
public execution remain pending. This document does not certify v4 execution.

## Version meanings

| Native NVFP4 schema | Producer program | Row policy |
|---|---|---|
|v1|lossy requantize, storage, product|static|
|v2|lossy requantize, storage, product|bounded|
|v3|source-preserving expansion, exact K16 FP8 product|static|
|v4|source-preserving expansion, exact K16 FP8 product|bounded|

These are `tessera.native.nvfp4_program` versions. Other program namespaces
retain their independent schema meanings.

## Native frame

The existing `tessera.native.nvfp4_m_bound = CAP : i64` request annotates a
static traced activation M. Native MLIR checks positive M<=CAP and checked
byte capacities, retains `original_graph_ir`, then projects capacity types.
Source expansion and scaled-product policies and SSA edges remain unchanged.
The compiler exports both verified members. Python does not reconstruct them.

Five distinct root roles may be reordered. Activation and activation-scale
slots are named explicitly; weight roots and both private producer results
remain independent of active M.

| Role | Capacity shape | Row bytes |
|---|---|---|
|activation|CAP x K, E4M3|K|
|activation scales|CAP x K/16, FP32|4*K/16|
|returned product|CAP x N, FP32 or BF16|N*output_bytes|
|private expanded weights|N x K, E4M3|invariant|
|private scale plane|K/16 x N, FP32|invariant|

`row_frame` records `input_slots`, `output_slot`, `row_strides_bytes` in the
activation/scale/output order and `consumer_step`. Buffer ownership and write/
read lifetimes retain the ordinary native SSA plan. `active_m` is the trace
extent; `m_bound` is the capacity. Their integer values do not replace the
actual runtime row count in the consumer's M scalar and grid.

## Remaining runtime acceptance gates

- Native image projection must support arbitrary positive active M, including
  when CAP is aligned to a physical LDS tile. A capacity-derived `whole_m`
  assumption cannot authorize unguarded active-row accesses.
- The generic multi-output native owner must validate active row counts, byte
  spans and geometry before buffer access, retain one capacity allocation, and
  update row-dependent scalar/grid bindings atomically.
- Activation-only updates preserve imported weights and invalidate dependent
  outputs. Row changes invalidate any captured consumer geometry and stale
  output generations. Existing context, stream and allocation lifetimes apply.
- Original JIT, prepared ownership and compiler-free replay must execute the
  same native members across shrink/grow frames and reordered roots. Static
  v1/v2/v3 compatibility remains independently tested.
- Exact gfx1201 numerical comparisons must cover changed roots and both output
  storages. Separate producer, consumer/program and completed-host timing
  scopes are required before performance claims. Original model/checkpoint
  quality acceptance remains a separate gate.

## Current proof boundary

Thirteen native export/projection tests cover two capacities, both result
storages, argument permutations and invalid bound types/values/overflow. The
pre-change compiler rejects all thirteen at its static-frame boundary. Rebuilt
core/ROCm tools pass 351 host cases; three legacy runtime cases require the
owning ROCm host. The fresh main-based provider independently passes 50 static
source/ownership/schema tests on gfx1201. V4 has no device execution proof yet.
