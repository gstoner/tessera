# GFX1201 packed MXFP4 resident HIP lifecycle — 2026-09-23

Owner `ROCM-MXFP4-W4A8-1` / `IKF-1`; synchronization key
`GFX1201-MXFP4-RESIDENT-HIP-2026-09-23`. The packet was recorded on
Tajasarus (selected AMD Radeon RX 9070 XT, gfx1201) from merged main
`5fc724c2` plus the two source files bound by SHA-256 in `evidence.json`.
The manual executor was not automatically selected.

The same compiled HSACO and packed/scale payload were used for each shape's
host-array and resident measurements. Full BF16 output agreed between routes,
and sampled cells agreed with the independent exact K32 oracle. Separate
device tests proved ragged N48/N80, K64/K128 and lossless/lossy cases.

| M×N×K | Legacy host-call median | Resident host-call median | Resident kernel HIP-event median |
| --- | ---: | ---: | ---: |
| 256×5120×8704 | 19.46 ms | 2.25 ms | 203 µs |
| 1024×17408×5120 | 54.71 ms | 6.64 ms | 1077 µs |

Seven host-wall samples were measured for each route. The resident session
loads one module, allocates five device buffers, and uploads fragment-order B
and scale plane once. Each later `run_host` call still transfers A/As and
output, so this is a lifecycle comparison, not an assertion that data movement
has disappeared. NumPy host memory is pageable: `hipMemcpyAsync` may block on
the host. Kernel-event timing enqueues ten resident launches per sample and
divides the elapsed device time; it excludes host transfers and must not be
compared directly with host-wall timing or the older matched Radiance packet.

Next proof is device-owned A/As/output with explicit stream/lifetime ownership
and HIP graph capture/replay. Only then should a same-stream, same-data timing
comparison with Radiance and a public selection decision be considered.
