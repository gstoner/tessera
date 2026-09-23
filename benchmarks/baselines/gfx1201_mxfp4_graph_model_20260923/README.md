# GFX1201 model-owned MXFP4 graph and producer ablation — 2026-09-23

Owner `ROCM-MXFP4-W4A8-1 / IKF-1`; sync
`GFX1201-MXFP4-GRAPH-MODEL-PRODUCER-2026-09-23`. This packet was
recorded on Tajasarus's selected RX 9070 XT at merged-main revision
`bf422db9` plus the individually hashed source files in `evidence.json`.
The selected compiler target was pinned to gfx1201; the selected ROCm
toolkit and loaded HIP library were both ROCm 10.0 / HIP 7.15.

The model-owned boundary uses separate ROCm FP32 and BF16 allocations.
The three-kernel graph borrows their stable pointers and refuses model
free while captured. Exact-device tests exercise full BF16 output against
the folded oracle, both block and wave FP8 producers, repeated replay
with allocator churn, ragged/lossy shapes, and the exact-M pool. Host-free
failure tests cover failed capture, mismatched shape/device, allocation
failure, and uncertain asynchronous completion. The recorder ran all 18
focused cases with no failure or skip. An uncertain completion quarantines
the model buffers rather than claiming that they are safe to free.

The wave-shuffle reduction is an opt-in producer experiment. It preserves
the exact activation bytes and output but has no convincing whole-graph
gain on the two measured prefill shapes:

| M×N×K | Block producer event | Wave producer event | Block graph event | Wave graph event |
| --- | ---: | ---: | ---: | ---: |
| 256×5120×8704 | 20.83 µs | 23.23 µs | 197.26 µs | 196.73 µs |
| 1024×17408×5120 | 40.59 µs | 38.33 µs | 1350.84 µs | 1349.12 µs |

These are medians of two block/wave/wave/block rounds. Each round has
seven HIP-event samples of twenty launches, the same packed GEMM HSACO,
shape, logical inputs, and device. The block producer remains the default:
the stage-only improvement does not meet the declared whole-graph margin.
On the wide shape, even the block graph remains 1350.84 µs versus
1262.52 µs for direct triple launches (7.0% slower). Host enqueue is
lower, but that does not excuse the device-time regression.

The admission receipt therefore refuses automatic selection for the
wide-shape device-time regression, absence of a high-level model frontend
route, and lack of matched Radiance parity. The model-owned buffer API is
an opt-in ROCm boundary, not the Metal `DeviceTensor` ABI or a public
frontend. The exact K32 numerical route remains the correctness oracle.
Next: integrate a real model caller, investigate the wide-shape graph
scheduling overhead and GEMM/producer fusion, then remeasure Radiance
before considering selector admission.
