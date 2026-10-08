# Public saved O/LSE attention VJP and private residual ownership

Owner NVIDIA-LSE-1 / AD-RESIDUAL-EVAL-1 / FRONTEND-IR-MEDIUM-1 / E2E-REAL-6.
Synchronization key NVIDIA-JIT-MULTIRESULT-OWNER-2026-10-07.
Publication pending.

Ordinary Python JIT reverse attention returns the saved (output, LSE) tuple
through native paired AD, Graph/Schedule/Tile/Target lowering and checked
SM120 packages. The synchronous resident owner exposes read-only tuple views,
accepts both output and row-LSE cotangents, and privately copies seeds before
launch. It preserves the one-result backward API for legacy pairs. Seeded
complete and compact gradient contracts are validated before CUDA loading.

RTX 5070 numerical proof: 72 public cases covering three rectangular/batched
GQA shapes, causal/noncausal masks, optional broadcast bias, complete/compact
outputs and output-only/LSE-only/mixed seeds. Portable JSON round-trip replay
executes after compiler paths are disabled. Private Q/K/V/bias survives caller
mutation, repeated reverse calls retain prior results, undersized seed
allocations are rejected without allocating more frame buffers, and borrowed
views reject use after close. Independent FP64 oracle includes bias reduction
and reordered requested gradient roles. See device-tests.txt.

Owner/checkpoint/registry and legacy public/prepared/compact regression gates:
505 passed, 10 skipped (regressions.txt). Audit-document lifecycle: 11 passed
(docs-tests.txt). Changed Python Ruff and resident-owner Mypy checks pass
(lint.txt). Allocation-failure tests cover both one- and two-seed rollback.
These focused gates do not establish full-suite or sibling physical closure.

Recorder: python -m benchmarks.nvidia.record_jit_multiresult_owner.
rtx5070.json records 24 mixed-seed public program rows, five samples per
capture/backward/pair/device-forward/device-backward lane: 600 timing windows.
The resident event lanes execute the same public packages and numerically
check their outputs. Capture includes private input copies, module loading
and forward; backward includes seed/gradient allocation and synchronization.
Pair excludes oracle/download and frame close. Event windows include driver
gaps and are separate from host wall time. The interrupted overlapping trial
was discarded; this packet was recorded after device tests completed.

Median ranges: capture 1.835775–2.651811 ms; backward 0.192631–0.274132 ms;
pair 2.019846–2.657180 ms; forward event 0.007568–0.014717 ms;
backward event 0.008328–0.040712 ms. No speedup or route-promotion claim.
Source files and matching compiler/runtime binaries are fingerprinted.

Remaining: dynamic/nested/composed multi-result AD, asynchronous residual
ownership, wider batching/layout contracts, sibling exact-device consumers,
and delivery/full-suite gates. This static synchronous proof does not close
the five-slice objective or transfer to other architectures.
