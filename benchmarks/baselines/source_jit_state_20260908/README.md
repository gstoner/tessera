# Source JIT, exception transport and device state

Recorded 2026-09-08; uncommitted continuation after PR #736.
Owners: W4-PRODUCT-1 / W2.4a. Sync key: `SOURCE-JIT-STATE-2026-09-08`.

## CPU source and public JIT

`tests/unit/test_source_error_transport.py` exercises native LLVM execution of
explicit builtin exception transport, writes before an exception and finally,
strided/reversed state views, overlap refusal, keyword binding and bounded JIT
module ownership. The opt-in `jit(source_control_flow=True, ...)` owner retains
at most four native shape/alias specializations and closes evicted modules.
Use it as a context manager or explicitly close it. Incompatible target, AD,
source-string and batching options refuse rather than using an eager fallback.

Error-result specs supply static floating result shapes. Native code returns
AssertionError/RuntimeError/ValueError codes as an explicit completion result;
the CPU boundary commits preceding declared state writes and then raises the
class with a generic message. This does not preserve original messages or
transport dynamic exception objects, implicit numerical errors or bound exhaustion.
The latter retains its existing assertion boundary.

View admission conservatively proves non-overlapping elements and preserves
exact aliases. Partial overlaps, overlapping/broadcast views and arbitrary
Python objects remain unsupported. Copyback requires exclusive external ownership.

## Owning-device results

`fanin-eight-rocm.json` records all 256 combinations of capture plus seven
upstream statuses on Princess-Luna / gfx1151. Only all-success exposes the
expected derivative. Every failed combination produces the named guard failure;
all frames retire with no remaining owned buffers. Fault injection is an ordered
device status write, not a hardware fault. Final inspection is a host boundary;
composition forbids host capture checks and explicit context synchronization.

`source-state-rocm.json` records two native state steps: input 2 becomes state 4
and value 16; consuming that state produces state 8 and value 64. Original input
storage remains 2. The compiler lowers the serialized source-state results through
capacity/shape projection; no fabricated AD identity is used. Returned frames own
immutable state generations. Admission is restricted to one declared state
input until multi-input alias projection reaches the binding. This is not arbitrary in-place GPU mutation, a
parallel schedule, exception transport on GPU or performance promotion.

Super-Bear repeatedly timed out over SSH. No NVIDIA eight-status or source-state
packet was recorded, and synchronizing its new source remains pending. Apple
needs an MSL consumer and separate device evidence. No architecture proof transfers.

## Validation

292 final focused native/compiler/registry tests and 25 audit/governance tests passed
in host WSL. Ruff, the zero-error mypy ratchet and all 30 generated-document checks
passed. The frozen-tree full suite passed **18,577 tests**, with 2,231 skipped
and 870 slow tests deselected in 643 seconds. The final single-state GPU
admission restriction and two regressions were added afterward; the 292-test
focused suite and device-state recorder were rerun against that final tree.

Assertions-enabled validation of the new source-state admission remains a
follow-up: the assertions-enabled Super-Bear host was unavailable. The native
compiler build and contract tests here ran on Princess-Luna.

Recorded by `benchmarks/record_source_state_gpu.py` (`source-state-rocm.json`)
and `benchmarks/record_status_fanin.py` (`fanin-eight-rocm.json`).
