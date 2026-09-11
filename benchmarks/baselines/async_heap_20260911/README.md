# Asynchronous heap receipt and marking-allocation correctness

CUDA SM120 (Super-Bear) and ROCm gfx1151 (Princess-Luna) independently ran
`benchmarks/record_async_heap_admission.py` against their native compiler.
`nvidia.json` and `rocm.json` bind compiler, recorder and native artifact identities;
`source-hashes.json` identifies the shared implementation used in this wave.

Both pass eight admission/read/unpin cycles with two reusable private receipts,
cancellation cleanup, stale-generation refusal without pool poisoning, and
allocation during marking. Final retirement refuses until the new grey root is
scanned. Copies preserve the expected payload. Host tests additionally inject
pending completions, pre-submission refusal and uncertain status-copy failure.

`begin_read_object` returns a request; poll it before entering the payload scope.
Poll `poll_object_readers` after closing scopes to drive deferred unpin. Receipt
memory remains retained until completion; unknown decrements quarantine the pool.
Preallocate receipts with `prepare_readers` to avoid allocation in admission.
The new receipt path adds no explicit event/stream/context synchronization;
legacy synchronous readers should not be mixed into a no-wait cleanup interval.
The recorder forbids the shared synchronous status helper during receipt checks;
fake-driver tests also assert absence of synchronization calls in that path.

The two-writer model finds a split validate/write reservation counterexample.
Its atomic reservation variant passes its finite state exploration; this is a
model assumption, not native atomic lowering or weak-memory proof. GPU metadata
writers remain serialized by stream epochs. Mark begin/finalization, preparation
and pool teardown retain synchronous boundaries. Driver call latency is unbounded.
No measured overlap, kernel timing, bare-metal calibration or performance
promotion is claimed. Apple needs its own MSL binding; x86 model results do not
constitute GPU execution evidence.
