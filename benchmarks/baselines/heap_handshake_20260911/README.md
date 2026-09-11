# Native heap handshake and asynchronous lifecycle proof

`record_heap_handshake.py` runs independently on CUDA SM120 and ROCm gfx1151.
Packets identify compiler, recorder and replay-bound native artifacts. The source
hash manifest identifies the shared implementation tested on each host.

Checks cover private asynchronous admission/unpin, grey allocation publication,
polled finalization, and context-owning off-thread teardown. Native graph and
retirement callers use a shared aligned i64 gate and separate statuses. A held
gate returns busy (3) without spinning; two-stream submissions complete with
valid success/incomplete/busy results and release the gate. No overlap is measured.

The finite publication/retirement model finds an invalid rooted-dead object when
the gate is omitted. It assumes atomic linearization; weak-memory scope and
general heap correctness are not proved by enumeration. Schema 3 lowers through
LLVM system-scope acquire/release compare-exchange and release exchange.

The resident pool still uses stream epochs. Allocation, mark steps, pin/reclaim
and external metadata access must join the gate before general concurrent use.
Finalization receipts forbid further mutations until checked. Teardown polling
does not wait, but its worker may block in driver calls; four retained worker
slots cap admission. Unknown frees/unloads are not retried. Isolated recovery
of those retained pools remains open. No performance promotion or latency claim.

Validation: 77 focused WSL tests, including native replay and failure injection;
separate NVIDIA and ROCm device recorders. Apple and x86 execution are not inferred.
