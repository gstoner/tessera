---
last_updated: 2026-10-16
audit_role: reference
---

# Native NVFP4 bounded source program

Owner: ROCM-NVFP4-INGEST-1 / FRONTEND-IR-MEDIUM-1 / E2E-REAL-6.
Sync: NVFP4-SOURCE-BOUNDED-20261016.

Status: native export, arbitrary-active-M image projection and prepared/public
execution merged in PR960 at 6480571c0f. Owning gfx1201 validation
passes the named bounded envelopes below; broader integration and performance
gates remain open.

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

## Runtime ownership contract

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

The focused native export/projection and diagnostic/pass gates pass 375 cases.
On RX 9070 XT/gfx1201, 59 bounded/static source, public replay and native
partition tests pass with matching rebuilt compiler tools and the native
HIP owner. Fifteen bounded cases cover FP32/BF16 output, root permutations,
M1/19/33/64 shrink/grow, full-root re-import, activation-only weight reuse,
stale output rejection, retained output independence and eight malformed
native row frames. The capacity-256, N4096, K16 LDS package executes active
M19/1/256/127 with whole_m=false. Device buffer allocation count and byte
capacity remain unchanged during row rebinding.

The native owner checks the named A/B/SA/SB/output plus M/N/K ABI, including
one-byte activation elements, four-byte activation scales and two/four-byte
outputs. The additive row ABI does not alter the previous static struct
layouts or v1/v2/v3 schema meanings. Image cache identity includes the
arbitrary-M projection policy; static routes retain their existing instruction
semantics.

The device test module is
tests/device/rocm/test_nvfp4_source_bounded_rows.py.
Receipts and the exact timing recorder are preserved under
benchmarks/baselines/nvfp4_source_bounded_20261016; matching binaries remain
under ~/scratch/nvfp4-source-bounded-20261016 on the owning WSL host. A separate fresh-process bounded replay passes with compiler paths disabled
and numerical helpers forbidden. Nine correctness-gated timing rows cover
FP32/BF16 global capacities and FP32 capacity-256 LDS. Each has seven member,
program-event and completed prepared-host windows of at least 10 ms. Member
graph-dispatch medians are approximately 3-4 us for expansion and 6-12 us for
the consumer; completed prepared invoke/read medians are 0.219-0.416 ms.
These scopes differ and establish no speedup or pure-kernel latency.
A further owning-device compatibility gate passes 226 legacy bounded NVFP4,
FP8 W8A8, MXFP8, exact MXFP4 and generic scaled-owner cases. Adjacent host gates
pass 544 cases with 22 hardware-dependent skips. Generated-document, ABI and registry final gates pass 401 cases.
PR960 required CI and review passed before merge: 21,996 unit passes with 9,958
skips and 1,265 production compiler-route passes with zero skips. No full-model quality
acceptance, broad performance closure or gfx1151 FP8 execution is inferred.

## Representative original checkpoint follow-through

The packet's original_checkpoint/ directory retains native prepared,
ordinary JIT and portable replay for the original 24576x4096 combined gate/up
weights and validated captured activations. One capacity-256 owner handles
M19/1/256/200/127/19 with fixed allocations. Public/portable M1/19/200/256/1
replay forbids compiler and eager helpers. All arithmetic bounds pass.
M256 relative RMS versus the decoded FP8 projection is 2.568e-7, while the
original BF16 projection difference remains 6.2286%. Full MLP/model-quality
acceptance is still open.

PR960's first unit lane exposed an unnamed recorder stem and an incomplete
host-only readback fixture. The repaired fixture consumes the production span
helper and adds pinned/pageable active-prefix read coverage; 23 focused tests
pass. Native runtime and arithmetic source hashes remain unchanged.
