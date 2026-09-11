# Bounded heap publication and reclamation — 2026-09-11

Owners: W4-PRODUCT-1 / W2.4a / DISPATCH-BREAKER.
Sync: `HEAP-BARRIERS-2026-09-11`.

The recorder runs separately on CUDA SM120 (RTX 5070, Super-Bear) and ROCm
gfx1151 (Radeon 8060S, Princess-Luna). Both are WSL hosts. These packets are
correctness and diagnostic measurements, never selector evidence.

## What is demonstrated

- A one-slot, two-reader, two-generation model visits 103 states / 256 transitions
  without an unsafe lifetime. Disabling the reclamation dependency yields a
  counterexample. This abstracts reachability as a root bit; it is not a proof
  of general concurrent graph marking or a hardware weak-memory model.
- Serialized v1 protocol and exact replay bind whole-kernel publication,
  exclusive stream epochs, generation layout and all-reader reclamation.
- Stale edges refuse the complete graph update without changing roots/edges.
- A newly published root between snapshot marking and final remark preserves
  its cycle. Final remark remains exclusive.
- Retired slots cannot be allocated or resurrected. Reader scopes on two
  streams queue payload copies before reclamation/reallocation; both copies
  retain the old bytes. All allocations occur before this sequence; there is
  no explicit host wait between reader enqueue and reuse submission.
- Unit fault injection covers failed reader-event recording and retry after
  explicit completion. This is not a physical driver-hang experiment.

The copy case does not force long-running simultaneous readers. It validates
multi-stream submission/completion semantics together with the model and
injected pending-reader tests; it is not a measured overlap claim.

## Equal-work diagnostic comparison

Each variant starts with the same 16-slot graph (32-byte payload capacity,
one edge per slot); exactly two nodes remain reachable. Both routes are
compiled/warmed before timing. Five samples alternate ordering in one process.
The baseline combines retirement/reclamation; the second route splits them
across two kernels. Reset/upload/compilation and oracle reads are outside the
interval. Native event intervals include host submission gaps and are **not
isolated kernel durations**. Host completion timing includes submission and
waiting. No counters/clocks calibration or independent-process confidence bound
is present. Exact measured bindings are in `artifacts.measurement_*`.

| Backend | Variant | Median completion wall ms | Median submission stream interval ms |
|---|---|---:|---:|
| nvidia | exclusive_collect | 0.3089 | 0.2386 |
| nvidia | split_retire_reclaim | 0.3649 | 0.3567 |
| rocm | exclusive_collect | 0.3675 | 0.3290 |
| rocm | split_retire_reclaim | 0.6750 | 0.6358 |

The split route costs more in this experiment. Retain the existing fast path;
use explicit retirement only when its lifetime boundary is needed. This does
not establish performance on a larger graph, concurrent marker, or bare metal.

## Reproduction

Run `benchmarks/record_heap_barriers.py --backend nvidia|rocm --compiler <owning
compiler> --output <packet.json>` in the host WSL environment, with the owning
backend environment script loaded. Packets include compiler/recorder hashes;
`source-hashes.json` pins the participating sources. No ledger is installed.

Open: per-object reader admission, incremental-update dirty work, concurrent
final retirement, atomics/scope lowering, physical overlap and eligible
per-process measurements. Apple and x86 require their own native bindings.
