# Gated metadata and isolated teardown recovery

`record_gated_heap.py` executes independently on CUDA SM120 and ROCm gfx1151.
Each packet records native binding/protocol, compiler and recorder identities.
`source-hashes.json` binds the implementation used by both owning-device runs.

The opt-in `ResidentGatedPool` routes all admitted metadata operations through
its owned gate. Eleven native producers have replay tests; device execution
covers allocation, graph publication, mark phases, pin/unpin, reclaim and copied
inspection. Live metadata pointers, snapshot and object-import shortcuts refuse.
Epoch ordering remains; the existing incremental owner is not silently migrated.

`IsolatedHeapPool` uses a spawned device-zero worker and bounded host-only
allocate/inspect/mark/close messages. Invalid input never enters IPC. Normal close
and a stopped-worker timeout both require confirmed process death before channel
and worker capacity release. Eight retained slots bound process admission.
The stopped-worker fault is injected with SIGSTOP; it is not a real driver hang.
Death does not prove device health or authorize automatic replacement promotion.
In-process stalled teardown threads cannot be recovered through this boundary.

No measured overlap, performance promotion, arbitrary command transport or
cross-architecture evidence transfer is claimed. Legacy caller migration and
health-checked replacement remain follow-ups.
