# Declared object state and owned GPU mutation

Uncommitted engineering evidence for W4-PRODUCT-1 / W2.4a /
AD-RESIDUAL-EVAL-1, sync key SOURCE-OBJECT-OWNERSHIP-2026-09-09.

`nvidia.json` and `rocm.json` are independently measured correctness packets
for SM120 and gfx1151. Each records three synchronous updates into the same
privately owned state allocation, checked result values, active-reader refusal
and expired-reader refusal. Fresh output storage plus device-to-device copyback
implements the update; this is not a fused in-place kernel or performance result.
Each packet binds source hashes, compiler hash and recorder hash.

`nvidia_eight_status.json` closes the previously unavailable NVIDIA measurement:
all 256 combinations of eight incoming statuses executed on SM120. It does not
transfer evidence to another NVIDIA architecture or to Metal.

Focused CPU source and registry validation: 278 tests passed on Super-Bear with
the native x86 JIT. Static exception payloads, declared object fields, read-only
overlapping snapshots, overlapping-write refusal and explicit pure tensor VJP
are covered. Package mypy completed with no errors. Apple execution remains open.

## API scope

For declared fields, use `jit(source_control_flow=True,
source_mutable=(0,), source_fields=((0, ('x', 'y')),))(fn)` with an exact
`dict` or `SimpleNamespace`. The callable may update `state.x[:]` or
`state['x'][:]`; replacing the field itself is not admitted. Read-only inputs
may overlap, but any invocation declaring mutable state retains strict overlap
checks. Callers must exclude concurrent external host mutation.

For a pure tensor function, `program.vjp(x, cotangents=(seed,))` returns
`(primal_results, input_gradients)` using exported native products. Cotangents
are explicit; no claim is made for automatic effectful `grad` integration.

`OwnedSourceGPUState(program, input)` executes the initial state transition.
Each `step()` returns an independently owned result frame that the caller closes.
The owner must stay alive through all scoped reads and must itself be closed.
Readers must not retain escaped raw pointers beyond their scope. Device copy
or synchronization failure poisons the owner and retains the pending result
until safe close can complete.

## Final validation

Princess-Luna host WSL unit suite: **18,565 passed, 2,255 skipped,
870 deselected**, 591.24 seconds. Native-JIT execution is covered separately by
the 278-test focused run with explicit compiler/library configuration.
All 30 generated-document checks and 25 audit/governance tests passed;
Ruff passed and package mypy reported no errors. NVIDIA device validation used
LLVM 23.1.1, optimized with assertions enabled. These are correctness and
contract checks; none establishes performance promotion.
