---
last_updated: 2026-09-11
audit_role: reference
---

# Heap publication, readers and reclamation: architecture exploration

This records the design and the bounded stream-ordered implementation below.
It is not an implemented concurrent collector or a second status queue. Sequencing belongs to [W4-PRODUCT-1](INTEGRATED_COMPILER_PLAN.md#w4-product-1),
with [W2.4a](INTEGRATED_COMPILER_PLAN.md#w24a) owning reader lifetimes and
[DISPATCH-BREAKER](INTEGRATED_COMPILER_PLAN.md#dispatch-breaker) owning uncertainty.
See the [compiler map](README.md). Sync key: `HEAP-BARRIERS-2026-09-11`.

## Decision and first envelope

Use nonmoving, generation-checked storage. First formalize the existing exclusive
path, then implement reader-protected retirement and barrier-aware graph updates.
Retain a short exclusive final remark/retirement phase until a separate protocol
proves concurrent mark termination and the retirement/publication race safe.
An immutable copied snapshot remains a supported alternative, especially where
its copy cost is lower than mutation-barrier or retained-generation cost.

The first concurrent prototype is one device, bounded slots and strong edges,
immutable payload after publication, declared consumer streams, one graph writer,
and a collector. Exclude weak references, finalizers, resurrection, moving objects,
raw pointer escape, concurrent payload writes and cross-device/system-memory access.
A generation check protects identity; it does not protect a payload against reuse
between that check and its final load. A completion event protects queued work;
it does not itself preserve collector reachability.

No new dialect or public operation is authorized by this review. Any required
carrier must land with a real producer, verifier, lowering and executable consumer.
Do not revive the deleted `tessera.queue` dialect for naming convenience.

## Inventory from the current working tree

| Mechanism | Source and present guarantee | Missing boundary |
|---|---|---|
| Stream epoch | [native_stream_epoch.py](../../../python/tessera/compiler/native_stream_epoch.py): `write` excludes open host reader scopes and waits recorded device readers | No per-object reader/publication protocol; writers remain exclusive |
| Reader lease | [native_reader_retirement.py](../../../python/tessera/compiler/native_reader_retirement.py): acquire waits producer; scope exit records consumer completion | Raw exported pointers cannot outlive their declared lease; lexical exit is not device completion |
| GPU heap | [gpu_heap_collection.py](../../../python/tessera/compiler/gpu_heap_collection.py): bounded one-thread kernels, ordinary LLVM loads/stores, generation checks, lifecycle 0/1/2 | No concurrent allocator, atomic publication or mutation barrier |
| Pool owner | [resident_object_pool.py](../../../python/tessera/compiler/resident_object_pool.py): allocation, graph replacement and sweep use exclusive epoch; marker uses private copies | `set_graph` is bulk replacement, not an instrumented edge-update operation |
| Incremental sweep | Same emitter: whole dead cohort becomes lifecycle 2 before selected ranges become 0 | Retirement currently relies on exclusion; the states alone do not implement reader epochs |
| Snapshot readers | [resident_pool_snapshot.py](../../../python/tessera/compiler/resident_pool_snapshot.py): writer-ordered copies, separate reader-owned buffers | No same-storage concurrency proof; close remains synchronous |
| Tile lifetime legality | [TileBarrierReuseLegalityPass.cpp](../../../src/transforms/lib/TileBarrierReuseLegalityPass.cpp): allocation/SSA identity and completing dependencies gate reuse | Dynamic object handles, reachability and heap readers are not covered by Tile allocation identity |
| Typed waits | [TileOps.cpp](../../../src/compiler/ir/TileOps.cpp): typed async/mbarrier dependencies; empty waits refuse | A block/pipeline completion token cannot be reinterpreted as device-wide heap reclamation |

Existing packets in [dag_snapshot_20260911](../../../benchmarks/baselines/dag_snapshot_20260911/README.md)
cover separate snapshot storage and SSD DAGs, not the proposed barriers.

## Semantic obligations

| Boundary | Required invariant | Candidate implementation and refusal |
|---|---|---|
| Reservation/publication | No reachable handle before payload, edges, generation and length are initialized | Private reservation, then atomic release publication of a versioned descriptor; failed initialization remains unreachable |
| Root/edge update | A published edge cannot make a live target invisible to this collection | Barrier transaction validates/pins target, records required marking work, then publishes the edge; overflow must stop admission or conservatively retain, never drop work |
| Reader admission | A reader is registered before it can load a reclaimable descriptor | Launch-scoped epoch pin before enqueue, acquire publication, validate generation; per-object hazard protocol only in a later envelope |
| Retirement | No new root, edge or reader can acquire a retired generation | Close admission and linearize retirement against publishers; first implementation uses exclusive final remark |
| Reclamation | All readers that could observe the generation have completed | Recorded stream completions and retirement epoch jointly authorize reuse; a passed generation comparison is insufficient |
| Cross-queue order | Consumer follows the matching publication; reuse follows every admitted consumer | Existing events at launch boundaries; preserve device/context, owner and generation identity |
| Failure | Unknown completion cannot authorize destruction or epoch advance | Quarantine storage, event and module together; explicit completion or confirmed isolated-owner teardown, not timeout-as-completion |

Reservation is a proposed private state, not a reinterpretation of current
lifecycle 2 (retired). Specify the versioned ABI before adding states. Never
change the existing `(slots, 3)` state layout without updating manifests and
consumers. Generation/epoch exhaustion refuses admission before wraparound.

Epoch pinning starts before descriptor acquisition and remains live through the
last device access, even after the host scope closes. Final retirement must close
new admission and account for every older epoch. Newer epochs may use live objects
but must reject retired handles. Initially a pin protects the entire eligible
pool generation; finer granularity is an optimization with its own proof.
If retaining too many generations exceeds a budget, apply backpressure before
submission; do not reclaim a pinned generation to make room.

## Collector choice and unresolved algorithm questions

Prefer an incremental-update prototype over adding a deletion log to the current
snapshot scheme: the new edge producer can record the target before publishing
it, and a final exclusive remark still provides a conservative safety boundary.
This is a recommendation to model, not a claim that a pre-store mark alone is a
correct concurrent algorithm. Scanning, dirtying, edge replacement and retirement
must share an explicit transaction/linearization protocol. Multiword `(slot,
generation)` edges cannot be read torn; choose an atomic packed handle or a
versioned immutable descriptor and prove the target-specific representation.

Compare three implementations against the same graph traces:

- Existing snapshot plus exclusive remark: correctness baseline, copy/remark cost.
- Incremental update plus final remark and epoch reclamation: candidate; measure
  dirty-work amplification and retained bytes as well as pause time.
- Snapshot-at-the-beginning with old-edge logging: alternative if deletion-heavy
  workloads justify it; requires complete root/edge deletion logging, new-object
  policy, bounded log overflow handling and termination proof.

Do not start with moving GC or per-load read barriers. Nonmoving immutable payloads
allow a launch-level reader pin to cover many loads. If arbitrary dynamic pointer
escape is required, revisit hazards/read barriers and the API rather than assuming
current Python scopes cover it. Concurrent payload mutation needs a separate
race-free field/versioning protocol even when memory reclamation is correct.

An acquire/release pair needs a matching communication event and suitable scope;
a fence alone neither publishes a handle nor supplies reclamation. LLVM scope is
target-dependent. Use [LLVM atomic semantics](https://llvm.org/docs/Atomics.html)
and the [LLVM language reference](https://llvm.org/docs/LangRef.html#atomic-memory-ordering-constraints)
when defining lowering. Check [CUDA memory scopes](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/cuda-cpp-memory-model.html)
and [AMD atomic allocation/scope support](https://rocm.docs.amd.com/en/docs-7.2.4/reference/gpu-atomics-operation.html)
against the installed toolchain and actual allocation kind. Device scope must not
be silently promoted to host/peer coherence. These manuals constrain the design;
they do not validate our emitter.

## MLIR/LLVM integration boundary

The serialized artifact must carry pool identity, protocol version, handle layout,
address space, participant scope, supported effects, generation/epoch operands and
completion lineage. Runtime bindings derive those fields from the artifact.
Python ownership objects enforce admission but must not invent absent semantics.

Prototype explicit effectful runtime calls using existing MLIR/LLVM machinery,
then decide whether a dedicated operation is necessary for analysis/optimization.
Do not mark publication, reader acquisition, edge update or retirement pure.
Optimizers must preserve ordering and cannot hoist a payload load before admission,
CSE acquisitions across releases, sink publication ahead of initialization, or
coalesce storage across a pending reader. Any new stable diagnostics or registered
pass require the normal registry/metadata gates in the implementation PR.

Reuse Tile tokens for their existing completion domains only. Connect heap lifetime
legality to an explicit owner/generation proof; do not overload a shared-memory
barrier with global-heap or process-lifetime meaning. Kernel launch dependencies
can lower to existing events without forcing a GPU atomic for every host action.
Same-kernel concurrent participants require a separately checked atomic protocol.

## Engineering sequence and acceptance

These are scoped deliverables under the owners above, not independent global priorities.

1. **Executable protocol model (W4-PRODUCT-1, W2.4a).** Enumerate bounded
   reserve/initialize/publish/acquire/update/retire/complete/reuse interleavings.
   Model reader admission racing retirement, edge insertion racing scan, dead
   cycles across batches, generation exhaustion, log exhaustion and failed
   completion recording. Require no reachable freed node, torn handle or
   premature reuse. First model the exclusive baseline, then the candidate.
2. **Consumed artifact contract (W4-PRODUCT-1).** Version descriptor/handle/epoch
   layout; emit real publication and edge-update calls; serialize effects and
   lineage. Parse/verify and assertion-enabled optimization tests must reject
   mismatched pools/generations, unsupported scopes and missing dependencies.
3. **Reader-protected runtime slice (W2.4a, DISPATCH-BREAKER).** Implement
   launch-scoped epoch admission and deferred reuse with a real pool consumer.
   Keep final remark exclusive. Test enqueue/copy/event/free/unload failures,
   retryability, retained-byte limits and fresh-owner recovery boundaries.
4. **Barrier-aware marking (W4-PRODUCT-1).** Add the bounded graph-update
   transaction and dirty-work consumer. Compare retained sets against an
   independent stop-the-world reachability oracle. Do not admit bulk mutable
   graph replacement on the concurrent route without equivalent instrumentation.
5. **Owning-device proof, then performance.** CUDA and HIP independently run
   adversarial multi-stream schedules and inspect emitted atomic order/scope.
   Verify forward progress without assuming inter-block scheduling fairness;
   avoid spinning on work that may not be resident. Profile equal-work exclusive,
   snapshot and candidate runs: kernel/launch/copy costs, retained bytes, barrier
   traffic, collection pauses and throughput. Promotion requires eligible
   process-separated evidence. Successful WSL correctness is not bare-metal
   performance evidence.

Only after these gates consider concurrent final retirement, multiple writers,
variable-size allocation and extension-object tracing. Each extension changes
an invariant and needs a new model/device case, not just a wider numeric bound.

## Backend disposition

| Backend | Next architecture-specific obligation |
|---|---|
| NVIDIA | Verify global-memory atomic representation/scope and CUDA event ordering on the owning GPU; block barriers are insufficient for cross-block reclamation |
| ROCm | Verify gfx1151 ISA archive and LLVM atomic lowering, HIP event semantics and allocation coherence; gfx12 or Instinct evidence is separate |
| Apple | Follow-up: independent MSL atomic and command-buffer ownership design; no CUDA/HIP pool binding or device proof transfers |
| x86 | Follow-up: native host allocator threading/epoch contract and LLVM atomic proof; CPU model tests do not establish GPU execution |

## Bounded implementation follow-through (2026-09-11)

The first loop implements the conservative baseline across all five steps:

- `heap_protocol_model.py` explores 103 states / 256 transitions for one slot,
  two readers and two generations; a deliberately missing reclamation barrier
  has a counterexample. Graph reachability is abstracted, not generally proven.
- `heap_barrier_contract.py` emits and consumes `tessera.heap_protocol` alongside
  the tensor manifest. Kernel binding checks the protocol and exact native
  replay. Schema 1 explicitly requires exclusive stream epochs and publication
  at kernel completion; it does not claim atomic device publication.
- `ResidentObjectPool.retire_unreachable` and `reclaim_retired` separate logical
  retirement from reuse while retaining every reader dependency. Open scopes
  refuse; closed scopes enqueue event waits; eventless failure stays retryable
  after explicit completion. The entire pool remains the protection granularity.
  `pool.read` is a retained-buffer inspection lease and may inspect lifecycle-2
  records; it is not a live-object handle acquisition. Such inspection readers
  also delay reuse. A per-object API must reject new retired-handle acquisition.
- The resident `set_graph` producer now validates the complete candidate graph
  before any root/edge writes. Nonzero status refuses without partial mutation;
  callers must inspect status before treating the requested graph as installed.
  It remains an exclusive bulk update, not an incremental marking write barrier.
- CUDA SM120 and ROCm gfx1151 independently pass the recorder. The split route
  is slower in the five-sample diagnostic comparison; retain the existing fast
  path. No selector promotion or concurrent sweeping claim follows.

[Packets and methodology](../../../benchmarks/baselines/heap_barriers_20260911/README.md)
record limitations and exact artifact identities. This completes a bounded
baseline through the sequence, not its unrestricted extensions. Next: model
edge publication/scan interleavings, implement consumed dirty-work transactions,
and prove finer reader epochs before removing exclusion. Moving objects,
concurrent payload mutation and weak/finalizer semantics remain outside v1.


## Per-object and incremental follow-through (2026-09-11)

`resident_incremental_pool.py` now implements a separate v2 owner: native
per-object pin validation before payload exposure, bounded tricolor marking,
shade-before-publication graph updates, and final retirement while admitted
immutable-payload readers remain active. Reclamation checks per-slot pins;
unrelated slots can be reused before another object's reader closes.

The metadata writer remains exclusive. The new completion boundary is
**retirement concurrent with admitted payload readers**, not concurrent graph
mutation/retirement. Admission and unpin status checks currently synchronize;
close is synchronous. Unknown decrements quarantine rather than double-decrement.
Allocation during marking refuses, and the owner caps outstanding reader scopes.

The two-object model explores 6,690 states / 64,536 transitions and detects
missing marking/reclamation barriers. CUDA/HIP independently pass the native
recorder, including stale admission and incomplete-mark refusal. See
[packets and limitations](../../../benchmarks/baselines/incremental_object_heap_20260911/README.md).
No performance promotion or measured physical overlap follows.

Remaining: asynchronous checked admission and unpin, allocation barriers during
marking, larger/cooperative marking, and multiple metadata writers with an
atomic publication/retirement handshake. Those changes need additional model,
artifact, owning-device and performance gates; the earlier conservative route
remains available.


## Asynchronous receipts and marking allocation (2026-09-11)

The optional `begin_read_object` / `poll_object_readers` path uses a bounded pool
of private device statuses and pinned-host receipts. Admission exposes no payload
until its event and status are proven; cancellation still waits for a successful
pin to be unpinned. Receipts recycle only after failed admission or completed
unpin. A possibly executed decrement is never retried. Pre-submission metadata
exclusion remains retryable. Legacy `read_object` remains synchronous, and mixing
legacy readers into cleanup can retain that synchronous boundary.

`allocate_marked` is now a replay-checked schema-2 producer. It publishes new roots
as grey before live-state publication, so final retirement refuses until that work
is consumed. Allocation no longer universally refuses during marking.

CUDA SM120 and ROCm gfx1151 independently pass these contracts; see
[receipts and allocation evidence](../../../benchmarks/baselines/async_heap_20260911/README.md).
The two-writer reservation model exhibits the split validate/write race. Atomic
try-reservation removes that bounded counterexample, but no native CAS, memory
scope or concurrent retirement handshake is implemented. Metadata remains
serialized. Next: consumed atomic reservation/publication contracts and retirement
interleavings, followed by owning-device proof. Cooperative marking, arbitrary
heap mutation, asynchronous finalization/teardown and performance remain open.
No measured overlap or performance promotion. Sync: `HEAP-ASYNC-2026-09-11`.


## Native handshake and deferred destruction (2026-09-11)

Schema 3 adds replay-checked `atomic_graph_incremental` and
`atomic_retire_marked` producers. Both use one aligned zero-initialized i64 gate:
LLVM system-scope acquire/release compare-exchange admits the entire validation
and mutation transaction; release exchange publishes completion. A failed CAS
returns status 3 without spinning or mutating graph/lifecycle state. Each caller
owns a separate status buffer. Retirement and graph validation share the gate,
so no writer validates a live object while an admitted retire transaction kills
it. Retired handles remain invalid; the gate does not permit resurrection.

This is a native transaction foundation, not a replacement for the resident
pool's epoch ordering. All competing metadata operations must share the gate
before admitting them concurrently. Pin/reclaim, allocation, mark-step and
external metadata access still need integrated gate ownership. The model checks
one publication/retirement race and finds a missing-gate counterexample; it does
not establish general weak-memory or arbitrary-heap correctness. RDNA3.5's ISA
archive includes `GLOBAL_ATOMIC_CMPSWAP_B64`; CUDA and gfx1151 compile and execute
their own native artifacts.

`finish_mark_async` returns a private receipt and prevents further mutations
until polling establishes success or incomplete marking. `close_async` transfers
destruction to a bounded, context-owning worker. Its poll is nonblocking; driver
wait/free/unload may still stall. Four process-wide worker slots bound admission;
unknown failures retain the pool and slot and cannot retry a possibly executed
free. Ordinary `close` remains synchronous. Close reader scopes and resolve any
pending finalization before asynchronous teardown admission.

[Independent CUDA/HIP packets](../../../benchmarks/baselines/heap_handshake_20260911/README.md)
cover held-gate refusal, submissions on two streams, finalization and teardown.
No measured overlap, bounded driver latency or performance promotion follows.
Next: integrate gate participation across all metadata users, finer transactions,
cooperative marking and isolated recovery of retained teardown failures.
Sync: `HEAP-HANDSHAKE-2026-09-11`.


## Gated metadata owner and isolated recovery (2026-09-11)

`ResidentGatedPool` admits eleven gate-backed producers: allocation (ordinary and
marking), graph publication (checked and incremental), mark begin/step/retirement,
pin/unpin, reclaim and copied inspection. An owner-injected gate argument prevents
callers from accidentally choosing different gates through this API. Stream epochs
remain in place for ordering, shared status ownership and lifetime retention.
The legacy `ResidentIncrementalPool` remains a separate supported implementation;
this change does not silently convert its callers to concurrent execution.

Live metadata reads, legacy snapshots and object imports are refused by the gated
owner. `inspect_metadata` copies under the gate and returns checked host arrays;
payload access uses generation-checked pins. Constructor initialization occurs
before publication. Private raw driver access is outside the admitted API. Migrating
legacy snapshot/import callers requires an explicit gated producer, not an exception
to gate participation.

`IsolatedHeapPool` hosts the gated owner in a spawned CUDA/HIP process. Its bounded
host-only command envelope is allocate/inspect/mark/close; no pointer crosses IPC.
Input errors are rejected before sending. Timeouts or failures poison the owner;
`recover_async` requests process teardown and `poll_recovery` releases channels and
worker capacity only after confirmed death. Eight process slots bound admission,
and uncertain termination retains the slot. A close receipt alone is not death
proof. No driver free/unload is retried in the parent. Death proves resource-owner
termination, not device health; automatic replacement and health admission remain
separate gates. In-process `PoolTeardown` cannot recover a hung driver by this API.

[CUDA/HIP evidence](../../../benchmarks/baselines/gated_heap_20260911/README.md)
includes a real heap worker deliberately stopped before close, timeout retention,
and confirmed process teardown. This is injected process stalling, not a reproduced
driver hang. No measured overlap or performance promotion.
Sync: `HEAP-GATED-ISOLATION-2026-09-11`.
