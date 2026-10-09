# gfx1201 bounded NVFP4 ingest/product rows

Owner ROCM-NVFP4-INGEST-1 / E2E-REAL-6 / FRONTEND-IR-MEDIUM-1.
Sync GFX1201-BOUNDED-NVFP4-ROWS-20261009. Depends on PR913.

The final packet records active HIP device RX 9070 XT, ordinal 0, UUID bytes
32386439653765666266326566373136 (HIP's 16-byte identity). Source, recorder,
core/target compiler and both native runtime provider identities are separate.
Imported source/recorder/compiler hashes match the PR worktree.
Recorded by benchmarks/rocm/record_bounded_nvfp4_rows.py.

## Numerical and lifecycle proof

Eight new exact-device tests cover ordinary JIT, portable artifact replay,
input-role permutations, changed weights, shrink/grow/shrink, graph retirement,
activation-only rebinding and independent retained outputs. Compiler invocation,
eager frontend work and warm tracing are forbidden. Same-owner sessions retain
eleven allocations and constant capacity bytes. Thirty-six existing static
resident cases pass separately. Sixty-one native/frontend/lifetime host cases
pass, with three owning-image skips; native failure injection checks quarantine
on failed graph retirement and legacy static cache compatibility.

The twelve timing rows use capacities (M,N,K) = (257,32,64), (513,80,256),
(256,64,1024), with active M = 17, capacity, 1, 200. Producer packed codes,
exponents, statistics, fragment and scale plane are checked around timing.
Consumer/public outputs match the independent decoded post-ingest oracle with
max absolute error zero. This does not prove original BF16 checkpoint quality.

## Timing domains

Seven independent HIP event windows per row, 128 repetitions per window, measure
ingest (conversion plus storage), consumer, and the complete three-stage program.
These event intervals include native repeated dispatch and any dispatch gaps;
they are not a hardware-counter certificate of isolated kernel instructions.
Medians across the rows:

| Domain | Range |
| --- | --- |
| Native ingest events | 90.826–146.997 us |
| Native consumer events | 18.614–25.411 us |
| Native three-stage events | 116.891–171.390 us |
| Warm ordinary public call | 2.405–9.350 ms |
| Explicit capacity compilation | 3.471–3.619 s |

The public domain includes checks/packing, all-input uploads, native execution,
synchronization and readback. Values change on every recorded call and compiler
processes are forbidden. Compilation is measured separately in the recorder's
process; it is not a cold-process startup certificate. Stage medians are not
added to infer program latency. This characterizes latency without a paired
comparison, selector promotion, or speedup claim.

## Reproduce

Use matching native core/ROCm compilers and the HIP movement/image providers.
Verify the live device is gfx1201, enable TESSERA_GFX1201_DEVICE_PROOF=1, and bind
TESSERA_OPT, TESSERA_ROCM_OPT, TESSERA_ROCM_NATIVE_MOVEMENT_LIB and
TESSERA_ROCM_NATIVE_IMAGE_LIB. From the owning worktree:

```sh
python -m pytest -q tests/device/rocm/test_nvfp4_bounded_rows.py
python benchmarks/rocm/record_bounded_nvfp4_rows.py --output benchmarks/baselines/gfx1201_bounded_nvfp4_rows_20261009/gfx1201.json
```

Run the recorder after tests complete. It rejects concurrent pytest/Graphify jobs.
Device and host gate receipts are retained beside the packet. The full unit
summary records four failures before delivery repairs; nine targeted ownership/
routing checks pass after repair. Two generic scaled-product batching/transpose
closure gates remain open.

The owning environment warned that its pytest timeout configuration was unknown;
the device receipts do not establish plugin-enforced test timeouts.

Whole-model quality, generic layouts/batching/transpose, broader ingest policies
and ROCm W8A8/MXFP4 optimization remain separate obligations. No sibling backend
physical evidence is transferred.
