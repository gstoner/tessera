# Per-object readers and incremental marking — 2026-09-11

Owners: W4-PRODUCT-1 / W2.4a / DISPATCH-BREAKER.
Sync: `HEAP-INCREMENTAL-2026-09-11`.

Independent CUDA SM120 (Super-Bear, RTX 5070) and ROCm gfx1151
(Princess-Luna, Radeon 8060S) runs pass the same recorded workload.
Both hosts run WSL. No performance or measured hardware-overlap claim.

## Implemented envelope

`ResidentIncrementalPool` owns the pool, per-slot pin counters and all admitted
reader completions. Native pin validation checks slot/generation/lifecycle before
exposing only that object's immutable payload bytes. Admission currently waits
for a device status result; it is not fully asynchronous. A retired generation
rejects new admission even while an earlier reader still holds it.

Metadata updates remain serialized through the existing stream epoch. Grey/black
marks have a real producer and consumer: validated graph publication shades
new targets before storing references; bounded `mark_step` scans grey nodes.
Final retirement verifies no grey work and closed black edges/root reachability
before retiring white objects. It may execute during an admitted payload-reader
scope because the payload is immutable and pins prevent reuse. This is
retirement concurrent with payload readers, not racing metadata writers.

Reclamation frees only retired, unpinned slots. Closed scopes retain pins until
their recorded device event completes. Polling an unfinished reader does not
enqueue a wait that blocks unrelated metadata work. Unpin itself currently waits
for its native status; uncertain decrements quarantine the pool rather than
retrying a potentially executed decrement. Close is synchronous. A refusal
before decrement submission remains retryable.

The v2 `tessera.heap_protocol` is serialized and validated at exact native replay;
its `exclusive_metadata_epoch` and `exclusive_metadata` final-remark fields do
not claim device atomics. Pin/mark/reclaim arguments come from the tensor
manifest. Existing v1 combined-collector behavior remains available separately.

## Evidence

- The two-object model enumerates 6,690 states / 64,536 transitions, including
  cycles, root/edge changes, payload readers, pending/uncertain completion,
  marking and retirement. It reports counterexamples when shading or pin
  protection is removed. Metadata actions are linearized; this is not a
  hardware memory-model proof or arbitrary concurrent collector proof.
- Native publication from a black node to a white target creates grey work.
  Final retirement refuses incomplete work without changing the object states.
- An existing reader survives logical retirement of its object. Another,
  unpinned slot is reclaimed and reused while that reader scope remains active.
- Retired and stale-generation acquisitions refuse; proven reader completion
  permits reuse with an incremented generation and new payload.
- A following rootless cycle collects conservatively retained objects.
- Host fault injection verifies eventless reader retention, uncertain-unpin
  quarantine, and retry after a pre-submission metadata-scope refusal.

Device copies are small and do not force prolonged physical overlap. The test
proves admission and lifetime behavior, not a bandwidth or latency improvement.

## Limits and reproduction

One metadata writer; at most 256 outstanding host reader scopes; bounded fixed
slots and strong references. Payload writes, allocation during an active mark
cycle, weak/finalizer semantics, moving objects, raw pointer escape and racing
metadata writers are refused or outside the interface contract. Whole-pool
inspection still excludes metadata writes. Low-level kernel bindings require
their declared ownership preconditions; they are not arbitrary-pointer safety.

Run `benchmarks/record_incremental_object_heap.py --backend nvidia|rocm --compiler
<owning compiler> --output <packet.json>` with the owning environment script in
host WSL. Packets contain exact bindings and compiler/recorder hashes;
`source-hashes.json` pins participating source. Old evidence directories remain
historical and unchanged. No selector ledger is installed.
