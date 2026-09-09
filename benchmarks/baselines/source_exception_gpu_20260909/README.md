# Mapped views, adjoints and checked GPU exceptions

Owners: W4-PRODUCT-1 / AD-RESIDUAL-EVAL-1 / W2.4a.
Sync key: SOURCE-MAPPED-EXCEPTION-2026-09-09.

Independent SM120 and gfx1151 packets each cover eleven cases: synchronous and
asynchronous negative/multidimensional source-view execution; successful and
failed static/dynamic exception completion; and a native mapped backward
product. Static cause and context share identity. Failed asynchronous frames
expose no results and repeated polling retains the same exception object.
Inputs remain unchanged. Single-element dynamic f32 exception payloads are
read back only after completion and shape/status checks.

384 focused native source/ownership, mathematical, diagnostic, pass metadata,
and audit/governance tests passed on Super-Bear with assertions-enabled LLVM 23.
Package mypy (516 files) and touched-file Ruff passed. The native CPU regressions
include finite-difference overlapping-view adjoints and a Block AttnRes full-rank
mixing-matrix counterexample. Device packets fingerprint the compiler, recorder
and relevant sources. They do not establish model workload or performance
promotion. No full unit suite was run for this increment.

Mapped views are injective, static, rank preserving and bounded to 256 elements;
positive rank-one slices keep the existing direct-slice path. Private copies and
singleton-slice expansion are correctness mechanisms, not a scalable cooperative
schedule. Dynamic/large maps, new loop context slots, retained dynamic context
payloads, full Python traceback frames and exception AD remain open. Native
raise locations are notes, not fabricated Python frames. Owned in-place GPU
mutation remains exception-free; error cleanup may synchronize.

The Block AttnRes plan routes this evidence to state-view/lifetime oracles.
Multi-query stats descriptors, composed backward/checkpoint products, native
CUDA depth-attention packaging and per-target cooperative kernel timing remain
separate acceptance work.
