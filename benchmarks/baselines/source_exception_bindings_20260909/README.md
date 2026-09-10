# GPU exception-class bindings and scoped retirement

Owners: W4-PRODUCT-1 / AD-RESIDUAL-EVAL-1.
Sync key: SOURCE-EXCEPTION-BINDINGS-2026-09-09.

Independent SM120 and gfx1151 packets use
`benchmarks/record_source_exception_bindings_gpu.py` (19 cases per host).
A custom DomainError is reconstructed on the host after native GPU failure,
retaining its constructor-set attribute and cause/context identity. Repeated
failed polls invoke no additional constructor and submit no backward. Scoped
cases retire through event-ordered polling; source/compiler hashes bind each
packet. Earlier forward-only signed-view cases remain regression coverage.

Host failure-injection tests verify directory cleanup retry only after successful
unload and context exit, without repeating driver work or losing the admission
slot. Host traceback tests also preserve actual constructor/completion frames
across repeated failed polls without growing the traceback chain. These injected failures are not physical device fault/recovery proof.

No arbitrary exception heap, full CPython frame, handled constructor-side-effect,
unbounded driver cancellation, Apple support or performance promotion is claimed.
