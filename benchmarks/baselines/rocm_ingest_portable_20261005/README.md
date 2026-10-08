# Portable resident NVFP4 program

Owner ROCM-NVFP4-INGEST-1; sibling ROCM-MXFP4-W4A8-1.
Sync ROCM-INGEST-PORTABLE-2026-10-05.

## Implemented contract

NVFP4ResidentProgram.to_json/from_json and to_dict/from_dict retain three
compiler-owned stages: native conversion, lossless packed storage and packed
matmul. Each stage retains Graph, Schedule, Tile and Target IR, checked native
HSACO and its launch descriptor. Replay retains the existing one-stream,
eleven-private-allocation lifetime; no intermediate weight readback/reupload
is required for execution.

The v1 schema admits only the named static gfx1201 contract. Restore checks
ordered stages, shape, image/descriptor digests, canonical Graph semantics,
physical storage and explicit numerical policies before HIP allocation.
Export and restore own their nested metadata. JSON duplicate fields and
nonfinite constants are rejected. Digests establish content integrity, not
origin authentication; portable images come from a trusted compiler source.

Twenty-one focused gfx1201 tests cover numerical/bitwise replay, new-process
execution with tessera-opt unavailable, altered stage order/geometry/images,
recomputed metadata digests, caller mutation, and failure-safe allocation
lifetime. Compiler and host conversion helpers are disabled during replay
checks. The three synthetic shapes are complete, ragged-M, and short-K
envelopes. [Device tests](resident-tests.txt); [shared gates](shared-tests.txt).

## Exact-device benchmark

[Checkpoint](checkpoint.json) and [synthetic rows](synthetic.json) identify the
RX 9070 XT / gfx1201, compiler hash, source hashes, component image digests and
input hashes. Correctness is checked before and after event/graph timing and
after every portable replay. Restore timing is JSON decode plus all contract
validation; full replay wall additionally includes module loads, allocations,
uploads, kernels, output readback and cleanup. Resident graph windows include
GPU graph dispatch; host-submitted event windows include host submission gaps.
Neither is presented as isolated kernel time.

| M,N,K | Restore/validate ms | Restore + full replay wall ms | Resident combined graph ms |
| --- | ---: | ---: | ---: |
| 128,32,256 | 2.013 | 13.573 | 0.174 |
| 257,80,1024 | 1.965 | 16.995 | 0.226 |
| 256,64,64 | 2.033 | 15.370 | 0.200 |
| 256,24576,4096 | 2.275 | 172.077 | 41.553 |

For pinned gate/up M256/N24576/K4096, package compilation is
3972.7 ms and serialization is
281346 bytes. Separate graph medians are
converter 40.786 ms, storage
0.341 ms and consumer
0.353 ms. Conversion remains the
dominant full-chain device cost. Resident-weight reuse wall is
5.825 ms. Full/reuse/control walls
have different ownership scopes and are not speedup denominators.

## Independent FP8, MXFP8, MXFP4 controls

All controls use the same original seeded FP32 activation source and BF16
checkpoint weights; activation SHA equality is asserted in the recorder.
These are reproducible format measurements, not model-activation quality.

| Format arm | Device execution + dispatch ms | Output source relative RMS % |
| --- | ---: | ---: |
| fp8_k128_n128 | 0.548 | 3.686 |
| fp8_k32_n1_control | 0.808 | 3.388 |
| mxfp8_k32_n1 | 0.829 | 3.751 |
| mxfp4_folded | 0.391 | 12.049 |
| mxfp4_folded_native_packed | 0.386 | 12.049 |

The ingested folded chain has 15.177%
output source relative RMS error and 14.980%
folded weight error. Arithmetic versus the declared folded oracle has
0.03109025 maximum absolute error and zero per-element
bound violations. This separates numerical implementation correctness from
quantization quality. Model-quality acceptance and default promotion remain open.

## Validation and remaining work

- 21 exact gfx1201 focused resident/portable tests pass, including three fresh-process replays.
- 519 shared artifact/operator/dtype/diagnostic/pass gates pass; 18 hardware tests skip on the NVIDIA host.
- One existing NumPy array-shape deprecation warning remains in the artifact suite.
- The post-maintenance RTX 5070 LayerNorm-to-matmul JIT/replay lane passed 20 tests.
- 11 audit-document tests pass; 32 generated documents are in sync; compiler-plan, scoped Ruff and diff checks pass.
- All four backend queues assess shared ownership and architecture-specific physical follow-ups.
- Graphify query/update is unavailable in WSL; no fresh graph claim is made.

Ordinary composed frontend JIT, general frontend/AD, dynamic packing/layouts,
broader cache/producer families, whole-model quality and sibling conversion
execution remain open. The original five-slice objective remains active.
No general uint8, format selector or default strategy is promoted.
