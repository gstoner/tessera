# Native paged read → softmax edge

Owner E2E-REAL-6; sibling FRONTEND-IR-MEDIUM-1.
Synchronization key ROCM-PAGED-SOFTMAX-EDGE-2026-10-06.

## Implemented route

Two typed frontend JIT packages retain their independent native
Graph → Schedule → Tile → ROCm Target → LLVM → HSACO chains.
The native C++ owner binds paged-read output directly to f32 last-axis softmax,
owns the intermediate and final allocation, retains both image leases and
orders both kernels on one private stream. One completion publishes a checked
generation. No intermediate upload/download or Python GPU argument construction
occurs on a warm edge invocation. Separate producer and consumer event windows
are returned. GPU arithmetic and physical schedules are unchanged.

```python
read = tessera.jit(target="rocm_gfx1201", native_required=True)(paged_read)
softmax = tessera.jit(target="rocm_gfx1201", native_required=True)(normalize)
with read.prepare_native_paged_softmax(softmax, pages, table) as owner:
    receipt = tessera.runtime.launch(
        read.runtime_artifact(), {"resident_movement": owner, "download": False})
    assert receipt["ok"]
    output = receipt["output"].to_host()
    owner.upload((new_pages, new_table))
```

This is an explicit static package edge. A generic composed Graph still needs
compiler-owned partitioning/bufferization and native lifetime planning; this API
does not claim that closure. Consumer admission checks exact architecture,
f32 storage, unchanged output shape, last-axis semantics, static guards, scalar
ABI, workgroup policy, accurate exponent and FTZ policy before native ownership.

## Frontend gap fixed

The device loop first exposed that ordinary row-softmax JIT was artifact-only.
Canonical descriptor dispatch now compiles and launches static f32 softmax.
gfx1151 gets its explicit operation capability; gfx1201 reuses the existing
exact-device softmax capability and adds scheduled-package auto-selection.
The operation's BF16 contract remains rejected; target-wide BF16 storage does
not establish operation support. The numeric-policy test now checks explicit
operation dtype boundaries as well as derived positive support.

Six standalone rank-1/2/3 rows prove public native execution on the owning GPUs,
with one image/entry per architecture reused across those shapes. Six paired
small/large/full paged cases match an independent float64 oracle with maximum
absolute error below 4e-8. Rebound source/table, retained original host outputs,
stale generations, compiler-forbidden warm calls and zero warm image reloads
pass. Eight prior native resident movement cases pass again after the change.

## Measurements

Milliseconds, medians of nine alternating trials per arm.

| Architecture | Case | Public host pair | Common resident + download | Resident only | Common/host pair |
| --- | --- | ---: | ---: | ---: | ---: |
| gfx1151 | small | 1.0859 | 0.3180 | 0.1360 | 0.2928 |
| gfx1151 | large | 1.1723 | 0.3322 | 0.1364 | 0.2834 |
| gfx1151 | full | 9.3341 | 3.4513 | 1.1206 | 0.3698 |
| gfx1201 | small | 1.1074 | 0.3430 | 0.1641 | 0.3098 |
| gfx1201 | large | 1.2189 | 0.3800 | 0.1869 | 0.3118 |
| gfx1201 | full | 7.0530 | 2.2956 | 0.6402 | 0.3255 |

The resident arms exclude initial input upload and preparation. The common arm
includes the final host download; resident-only verification downloads outside
its wall window. The public pair includes the intermediate host round trip and
both input uploads. Ratios therefore measure ownership, dispatch and transfer
savings, not faster GPU arithmetic. Separate HIP stage event windows may include
host feed gaps; they are not isolated instruction times. No graph capture or
persistent GPU scheduling is added by this product.

## Validation and evidence

512 shared WSL tests passed, 13 hardware/environment skips. Controlled native
tests prove exactly two launches without a host copy, four owned allocations,
two retained image leases, shape/capacity refusal, consumer failure invalidating
the old output, and release through context clear. Five Python descriptor tests
cover architecture, extent, row count, numerical policy and layout mismatches.
Two new C ABI exports are registered through the owning runtime ABI generator.

Packets retain live GPU/PCI identity, compiler and native runtime binary
SHA256, source fingerprints, independent errors, raw wall/stage samples,
exact entries/image digests, both adjacent IR chains and both HSACOs.
Architecture-specific source differences are retained by scoped transfer.
history-before-capability-cleanup/ contains earlier source history; final
packet.json is current proof. movement-regression/ is the current generic-owner
regression, while the previous owner packet is historical after this change.
contracts.txt records the shared gate results.

## Still open

Generic composed/dynamic/layout edges, arbitrary borrowed tensors, asynchronous
streams, softmax AD and other ROCm math metadata consumers remain open.
Higher derivatives, NVIDIA producer breadth, quantized-format performance
gates and the full five-slice program remain separate obligations. Apple,
NVIDIA and x86 require independent physical consumers; HIP execution and its
measurements establish no sibling parity.
