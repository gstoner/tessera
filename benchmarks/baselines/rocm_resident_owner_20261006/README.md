# Native resident ROCm movement owner

Owner: E2E-REAL-6. Sibling: FRONTEND-IR-MEDIUM-1.
Synchronization key: ROCM-RESIDENT-MOVEMENT-OWNER-2026-10-06.

## Implemented and proved

The Python frontend retains typed native Graph → Schedule → Tile → ROCm Target
→ LLVM → HSACO lineage. Five new C ABI exports put immutable image and argument
binding, three private device buffers, module lease, stream/events, completion,
generation checks and release in C++. No GPU kernel or physical schedule changes.

Public preparation and the checked common runtime use the same sealed artifact:

```python
with jitted.prepare_native_movement(*host_arrays) as owner:
    artifact = jitted.runtime_artifact()
    receipt = tessera.runtime.launch(
        artifact, {"resident_movement": owner, "download": False})
    assert receipt["ok"]
    output = receipt["output"].to_host()
    owner.upload(replacement_arrays)  # declared frontend argument order
```

An opaque result must be read before the next invocation. A materialized host
array remains independent after rebinding. Fork, wrong context, mismatched
artifact, unsupported storage, index bounds and external streams are checked.
Failed completion retains resources until a safe retry; context clear retires
native owners before image teardown. Close before destroying a HIP context.

Five Radeon 8060S gfx1151 and three RX 9070 XT gfx1201 cases pass bit-exact
nonfinite/signed-zero movement, rebinding, retained original host outputs,
invalid-index rejection and stale-generation rejection. Token gather is admitted
only on gfx1151. Full paged read proves one static logical-page count exceeding
physical pages, not general layouts. All warm compiler subprocesses are forbidden;
image load/unload counts remain unchanged in each warm arm.

## Measurements

Milliseconds, medians of nine alternating trials per arm.

| Architecture | Case | Public host JIT | Common resident + download | Resident without download | Common/public |
| --- | --- | ---: | ---: | ---: | ---: |
| gfx1151 | paged small | 0.5108 | 0.2899 | 0.1069 | 0.5675 |
| gfx1151 | paged large | 0.5287 | 0.2904 | 0.1070 | 0.5493 |
| gfx1151 | full large | 3.7906 | 2.3755 | 0.4813 | 0.6267 |
| gfx1151 | dispatched small | 0.5034 | 0.2799 | 0.1065 | 0.5560 |
| gfx1151 | dispatched large | 1.2064 | 0.9536 | 0.1200 | 0.7905 |
| gfx1201 | paged small | 0.5237 | 0.2582 | 0.0967 | 0.4930 |
| gfx1201 | paged large | 0.6080 | 0.2689 | 0.1108 | 0.4423 |
| gfx1201 | full large | 3.0158 | 1.7805 | 0.2233 | 0.5904 |

The common resident arm excludes repeated input upload but includes output
download and descriptor validation. Public host JIT includes both transfers.
Resident-only output verification downloads outside the measured wall window.
This demonstrates reuse and runtime overhead savings, not faster GPU arithmetic.
Compilation, initial upload and public preparation are outside warm timings.

The packets retain separate native one-launch HIP event windows; these can
include device/host feed gaps and must not be compared as isolated instruction
times or with the previous 256-node graph windows. This product uses a module
launch per invocation; graph capture remains a separate benchmark result.

## Validation and provenance

Focused host WSL gates: 394 passed, 13 hardware/environment skips, including
controlled native failure/lifetime tests, registry/diagnostic/pass metadata,
runtime ABI/header, shared execution contracts and public movement frontend.
Five exports are registered through the runtime ABI owning generator.
Controlled tests prove persistent allocation, independently retained image
ownership, failed completion recovery, wrong-context refusal and context clear.
The native runtime was rebuilt separately on each owning ROCm host.

Each architecture packet includes live GPU identity/PCI address, compiler and
native runtime binary SHA256, exact entry/image digest, all five adjacent IR
snapshots, HSACOs, source fingerprints and raw samples. Architecture-specific
jit.py/runtime.py differences are retained; scoped additions were transferred.
history-before-public-factory/ is earlier recorder evidence, not final proof.
contracts.txt is the focused gate log.

## Remaining work

Borrowed device tensors, compiler-owned producer/consumer chaining, dynamic or
general paged layouts, asynchronous stream admission, movement AD, distributed
transport and wider five-slice closure remain open. The next meaningful increment
is one native resident movement-to-consumer edge with explicit capacity/lifetime
checks, numerical proof and separate timings. Apple/NVIDIA/x86 need independent
physical consumers; no HIP result establishes sibling execution. FP8/MXFP8/MXFP4
and wider W8A8/MXFP4 performance gates remain independent.

## Native intermediate follow-on

[Paged read softmax edge](../rocm_paged_softmax_edge_20261006/README.md) adds an
owned compiled consumer and reruns all eight generic movement cases. This
earlier owner packet retains historical runtime/JIT source fingerprints;
movement-regression/ in the new packet is current-source proof.
