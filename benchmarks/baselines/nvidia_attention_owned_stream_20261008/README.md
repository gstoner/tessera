# SM120 attention owner stream ordering

Owner: E2E-REAL-6 / AD-RESIDUAL-EVAL-1.
Synchronization key: ATTENTION-OWNED-STREAM-2026-10-08.

Exact-device execution uses Super-Bear's RTX 5070, SM120. The JSON records GPU identity, compiler/runtime hashes, source hashes and package ancestry.

The synchronous saved-O/LSE owner now uses a private nonblocking CUDA stream, orders declared producer streams with event waits, queues private copies on that stream, and synchronizes that stream before returning. Shape, allocation extent and context checks remain. JVP directions are synchronized before handing off to the separate native JVP binding. Allocation/free remain synchronous CUDA operations; this does not establish asynchronous return or absence of implicit allocator synchronization.

Validation:
- 114 unit/protocol/public multi-result VJP checks passed.
- 2 device tests passed with pending seed writes on a different producer stream, numerical gradient comparison and actual event/wait assertions.
- 28 public/prepared native JVP checks passed; four existing multithreaded-fork deprecation warnings remain.

Matched A/B:
- The preserved context-wide control is context_control.py.
- Recorder: benchmarks/nvidia/benchmark_attention_owned_stream_ab.py.
- Two counterbalanced rounds, 24 profiles per arm, 96 profile executions and 48 matched comparisons.
- Both arms use identical actual traced serialized packages, checkpoint identities and compiler/runtime binaries. Package pinning avoids cold/warm metadata changing program digests.
- Three samples per metric; resident device windows include driver enqueue gaps. Public capture/backward/pair wall times include their documented allocations/copies/launches. These are not isolated kernel measurements.
- Median control/candidate ratios: capture 0.968099, backward 0.938151, pair 0.966780. Candidate is about 3.3%, 6.6%, and 3.4% slower respectively. No performance promotion.
- Initial unpinned recording passed numerics but failed its identity gate because cold/warm compile metadata differed; it remains in host scratch and is not the matched packet.

Remaining: attribute driver/allocator overhead, tune while preserving producer ordering, broader dynamic/composed AD and asynchronous ownership. NVIDIA proof is not Apple, x86, gfx1151 or gfx1201 execution proof.
