# Effect-aware source CFG and asynchronous status composition

Recorded 2026-09-08; uncommitted continuation after PR #736.
Owners: W4-PRODUCT-1 and W2.4a. Sync key:
`CFG-STATUS-COMPOSITION-2026-09-08`.

## Device composition

`compose-nvidia.json` (RTX 5070 / SM120) and `compose-rocm.json`
(Princess-Luna / gfx1151) independently exercise this truth table:

| Capture status | Upstream derivative status | Consumer exposure |
| --- | --- | --- |
| Success | Success | Success; derivative equals 24 in every element |
| Failure | Success | Named product-guard failure |
| Success | Failure | Named product-guard failure |
| Failure | Failure | Named product-guard failure |

Failure injection is a device-to-device status write on the owning stream,
after capture and before the consumer. This tests propagation and suppression;
it is not a device fault/recovery test. Native backward receives two distinct
readonly status allocations. It checks both before body effects. The recorder
forbids host capture checks and explicit context waits in the exercised path;
final result inspection is an explicit host boundary. Scoped upstream readers
remain live until the consumer is enqueued, and both frames retire with no
remaining owned buffers. Packet fields include package identities, compiler and
source fingerprints. Unexpected runtime errors are not counted as guard failures.

No performance or overlap claim is made. These packets do not prove Apple, RDNA4,
CDNA, arbitrary fan-in, external-reader closure or exceptional cleanup.

## Source CFG

`compose-source.json` records ten native LLVM CPU executions on Princess-Luna:
nested branches/early returns, single-carry native while, and bounded mixed local
state with break/continue and an ordered assertion. Expanded iterations merge
state before constructing the next iteration; a regression test rejects
exponential continuation growth. Assertion and exhaustion tests invoke compiled
code in separate processes and require SIGABRT after invocation begins. The
current LLVM host ABI does not return structured assertion diagnostics.

The envelope is opt-in, tensor-valued initialized loop state, at most 16 expanded
iterations and 256 statement/region visits. Single-carry while retains native
SCF. Unknown Python calls are rejected even on dead paths; arbitrary mutation,
loop-return payloads, exceptions, alias contracts and general JIT integration
remain open. This is not arbitrary/effectful Python closure.

Focused validation: 59 source/persistent/checked-status tests passed on the native
JIT host; 286 assertions-enabled compiler, registry and audit tests passed (two
owning-JIT tests skipped there). Ruff and the zero-error mypy ratchet passed.

The final frozen-tree WSL unit run passed **18,542 tests**, with 2,231 skipped
and 870 slow tests deselected. All 30 generated-document checks passed.
Source-inspection tests must run against unchanged files so loaded function
line numbers continue to identify the on-disk source.

Recorded by `benchmarks/record_async_status_composition.py` (`compose-{nvidia,rocm}.json`).
