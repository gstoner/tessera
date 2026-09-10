# Scoped source VJP retirement

Owners: W4-PRODUCT-1 / AD-RESIDUAL-EVAL-1.
Sync key: SOURCE-GATHER-SCOPED-2026-09-09.

Independent CUDA SM120 and ROCm gfx1151 packets run 19 cases each using
`benchmarks/record_source_scoped_ad_gpu.py`. They cover the existing signed
nested forward views, retained exception causes, synchronous checked VJP and
scoped asynchronous VJP. Successful derivatives are read through leases;
failed forwards never enqueue backward. Async cases use retire/poll, not close,
including failure completion. Inputs and cotangents remain owned until teardown.

The packets fingerprint their source tree and compiler. They establish
correctness of the bounded single-input source VJP binding, not arbitrary
exception object support, cross-queue writer ownership, performance or overlap.
Device module unload is deferred; failure quarantine remains a separate boundary.
Apple proof is absent. Native gather transposes have CPU numerical tests,
including signed/nested/empty maps and repeated-index accumulation; the GPU
slice cases in these packets are forward-only and do not prove gather AD.
