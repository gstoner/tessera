# Native packed MXFP4 compiler materializer — gfx1201

Owner: ROCM-MXFP4-W4A8-1; sibling ROCM-NVFP4-INGEST-1.
Sync: ROCM-PACKED-NATIVE-2026-10-03. No selector/default promotion.

## Compiler and runtime closure

The public packed-folded package builder now materializes its image from
Graph → Schedule → Tile → ROCm Target → native MLIR/LLVM lowering.
The legacy hand-emitted HIP builder is retained for historical diagnostic
experiments and is not called by this route. Device tests replace it with a
throwing sentinel. This is the named static M>64, N16, K64 packed-folded profile;
it is not arbitrary producer/ingest conversion closure.

Native B staging decodes fragment-order E2M1 bytes into typed E4M3 LDS using
the declared nearest-even folding policy. K32 block scales and the trailing
row-reference plane are preserved. Row-reference scaling remains after
full-K accumulation. Exponent code zero remains the named legacy zero-block
policy, distinct from standard MXFP8 E8M0 semantics.

A dynamic-offset reinterpret view replaces an unlowered subview at the
LLVM boundary. The checked launch validates native ownership, expanded memref
arguments, static dimensions and geometry before marshalling 25 memref
arguments plus M/N/K. Payload hashes guard weights and scale-plane identity.

## Exact-device tests

RX 9070 XT / live gfx1201, matching rebuilt LLVM23 compiler on Tajasaurus.
Five numerical cases cover ragged rows/columns, all E2M1 codes, scale deltas
0..14, zero-block scales, checked/resident launches and poisoned output stores.
Results match the independently decoded BF16 oracle bitwise.
Mutation cases reject wrong scale payloads, native ownership, argument layout,
decode policy and image dimensions before launch.

See device-all-codes.txt; native-regressions.txt has 88 adjacent host tests.
Super-Bear WSL host-drift.txt has 304 diagnostic/pass/checkpoint-input gates;
the separate packed host lane has 16 passes. Hardware claims belong only
to gfx1201. Graphify is unavailable on these scratch hosts.

## Native decode A/B and format gates

gfx1201.json records repeated per-nibble integer rounding. The native shared
magnitude-table change hoists this work per staged scale group;
gfx1201-shared-table.json records the tuned compiler. No schedule/scale
contract changed. Each run contains FP8 production, matched FP8 control,
standard MXFP8, expanded folded MXFP4 and native packed folded MXFP4.
All controls retain identical image payload hashes between these runs.

| M,N,K | Expanded MXFP4 device us | Packed baseline us | Packed shared-table us | Tuned packed / expanded | Tuned packed checked ms |
| --- | ---: | ---: | ---: | ---: | ---: |
| 200,256,128 | 16.963 | 20.514 | 18.404 | 1.085 | 3.895 |
| 256,1024,1024 | 21.235 | 27.067 | 23.010 | 1.084 | 6.905 |
| 256,4096,5120 | 77.075 | 118.803 | 89.369 | 1.160 | 13.386 |

Three alternating/rotating graph windows per arm pass the independent
f64 decoded-operand arithmetic bound before and after timing. Device timing
includes GPU graph dispatch; checked wall timing includes staging, transfers,
module loading and synchronization. It is not isolated ISA phase attribution.
Cross-run changes have matching unchanged control images but are not a single
interleaved dual-compiler capture.

The shared table reduces packed device time by 10.3%, 15.0% and 24.8% on
these rows. Packed weights reduce total physical input bytes by 25.8%,
37.4% and 44.1%, but device time remains 8.4–16.0% behind expanded weights.
Large-case checked wall time is 13.386 ms versus 17.275 ms expanded; small
wall measurements are noisy and do not establish a universal advantage.

## Remaining requirements

Pinned real gate/up evaluation is recorded separately in checkpoint.json:
eighteen format/preparation/shape arms passed on the live gfx1201 device.
Weights are the pinned Qwen layer-0 gate/up tensors (N=24576,K=4096);
activations remain seeded synthetic f32, not captured model activations.

| M=256 arm | Device us | Checked ms | Source-output relative RMS |
| --- | ---: | ---: | ---: |
| FP8 K128/N128 | 549.482 | 14.438 | 3.686% |
| MXFP8 K32/N1 | 827.894 | 16.206 | 3.751% |
| MXFP4 max-abs expanded | 383.994 | 64.886 | 12.049% |
| MXFP4 max-abs native packed | 388.713 | 55.915 | 12.049% |
| NVFP4-ingested MXFP4 expanded | 391.812 | 98.138 | 15.177% |
| NVFP4-ingested MXFP4 native packed | 391.446 | 57.197 | 15.177% |
| Direct joint-SSE MXFP4 expanded | 390.127 | 98.347 | 11.584% |
| Direct joint-SSE MXFP4 native packed | 391.481 | 55.841 | 11.584% |

Packed preserves each preparation's decoded output quality; it does not
repair ingest's additional quantization error. M128 packed device cost is
5.2–11.9% above its expanded counterpart. These are workload-specific results,
not a universal dispatch rule. Wall timings differ materially even between
expanded preparations and remain host-staging measurements.
Preparation/compile measurements include independent folded oracle preparation
and the counterfactual expanded package; they are not isolated compiler latency.

Host NVFP4 conversion still needs a policy-gated native MLIR conversion
operation; source-activation/whole-model quality is not proved.
Persistent/deeper staging, wider format/layout contracts and M256 Radiance
cost attribution remain open. FP8, MXFP8 and MXFP4 require independent
numerical/quality/performance evidence before any final strategy decision.
