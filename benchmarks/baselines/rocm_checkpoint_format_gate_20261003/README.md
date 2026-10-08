# Pinned gate/up FP8, MXFP8 and MXFP4 decision gate

Owner ROCM-NVFP4-INGEST-1; sibling ROCm route/performance work.
Sync ROCM-CHECKPOINT-FORMAT-GATE-2026-10-03.

## Exact-device result

Tajasaurus WSL, AMD Radeon RX 9070 XT, live gfx1201. Both M=128 and M=256
use the same pinned Qwen3 layer-0 gate/up source weights, N=24576, K=4096.
Six arms per shape passed checked native launch and resident replay against
independently decoded float64 operands before and after timing. All twelve
arms have zero elementwise arithmetic-bound violations. Weight quantization,
activation quantization and kernel arithmetic are separate comparisons.
Activations are seeded synthetic f32, not captured model activations.

The bounded row-batch FP4 search preserves signed midpoint nearest/even
decisions while avoiding a whole-model N*K*8 distance allocation. A real
transposed checkpoint also exposed noncontiguous FP32/E8M0 scale planes:
the runtime refused them before launch. Explicit contiguous materialization
now preserves quantized bytes/decoded values for C- and F-contiguous sources.
33 focused host tests passed on gfx1201.

## M=256 measured comparison

| Arm | Weight relative RMS vs BF16 | Native output relative RMS vs BF16 | Resident device/dispatch median (us) | Checked end-to-end median (ms) |
| --- | ---: | ---: | ---: | ---: |
| fp8_k128_n128 | 2.647% | 3.686% | 544.956 | 14.380 |
| fp8_k32_n1_control | 2.391% | 3.388% | 806.761 | 17.951 |
| mxfp8_k32_n1 | 2.656% | 3.751% | 831.134 | 17.787 |
| mxfp4_folded | 11.780% | 12.049% | 381.118 | 98.203 |
| mxfp4_nvfp4_ingested | 14.980% | 15.177% | 390.248 | 99.186 |
| mxfp4_direct_joint_sse | 11.300% | 11.584% | 388.028 | 100.604 |

[Full packet](gfx1201.json) records both shapes, pinned source revisions and
tensor hashes, independent projection global scales/row boundaries, compiler
and source SHA256s, actual geometry, native provenance, image/Tile/Target
digests, WMMA ISA evidence and balanced three-window graph measurements.

Resident timing includes device execution and dispatch; host staging,
transfers, validation, module handling and synchronization belong to checked
end-to-end measurements. The formats use distinct scale granularities and
physical schedules, so their differences do not isolate format cost.

All three folded MXFP4 weight payloads are lossless in their E4M3 expansion
for this particular checkpoint. The direct joint-SSE source has 11.300% weight
relative RMS error and the NVFP4-ingested source 14.980%; that gap precedes
folding. Conversion took about 14.619 seconds of **host**
preprocessing (the field is host_ingest_ms), not a native conversion kernel.
FP8 and MXFP8 weight errors are about 2.65% and 2.66% for the production
granularities. No selector or global format default is promoted.

## Pipeline and remaining work

FP8 and MXFP8 execute compiler-owned Graph/Schedule/Tile/ROCm native images.
Folded MXFP4 uses its existing Graph/Schedule/Tile/Target contract and native
MLIR typed-LDS materializer with checked ABI. Its physical weights are
**expanded E4M3 bytes**, not packed four-bit storage. The named folded profile
reserves exponent byte zero as a zero block; MXFP8 uses standard E8M0.
These distinct contracts must remain explicit in a native ingest operation.

The inspected packed folded public compiler helper still uses a hand-emitted
HIP materializer after Graph/Target carrier verification. It is not accepted
as native compiler closure and was not added as a substitute benchmark arm.
Native packed consumer migration, a policy-gated MLIR conversion op,
captured source activations/whole-model quality, host movement attribution,
wider W8A8 and general route/cache families remain open. This packet is a
mandatory FP8/MXFP8/MXFP4 evidence increment, not completion of all five slices.
NVIDIA, Apple and x86 cannot inherit this gfx1201 proof.

## Final validation

85 passed in 1.02s. Audit/diagnostic/pass drift gates: 303 passed in 1.82s. Compiler-plan, scoped Ruff, diff and generated-document regeneration checks passed. Graphify was unavailable on WSL (command not found), so there is no fresh graph claim.
