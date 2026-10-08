# Native checkpoint Graph ancestry

Owner NVIDIA-LSE-1 / E2E-REAL-6; sync
NVIDIA-CHECKPOINT-GRAPH-LINEAGE-2026-10-07. Publication pending.

Saved forward/backward checkpoint packaging replays retained Graph IR through
the production Graph-to-Schedule pass before native Schedule-to-Tile replay.
The Graph source is never reconstructed in Python. Missing Graph source or
valid but conflicting causal/scale policies are rejected before Target compile.
Saved and recompute descriptors expose the native Target digest, and the
recorder checks it against the packaged native image. No numerical algorithms,
native buffers, kernel schedules, pass registrations or checked ABI change.

Validation on Super-Bear host WSL with live RTX 5070/SM120 and CUDA 13.3:
80 focused tests pass with six gated skips; 115 device/contract tests pass;
122 paired and broadcast-bias tests pass. These lanes overlap and are not
a combined unique-test count. Coordinator native LLVM/MLIR build passes.

Recorder: benchmarks/nvidia/record_lse_checkpoint.py.
Final run: --adaptive-window-ms 20 with six explicit shapes:
1x2x1x3x4x4x3, 1x2x1x8x8x8x8, 1x4x2x15x17x16x12,
2x4x2x5x7x8x6, 1x4x2x127x131x64x64, 1x4x2x256x256x64x64.
Plain and --bias variants retain all profiles. Five windows use independently
calibrated event/wall counts, capped at 100 launches and 20 warmups. Actual
counts and pilot latencies are explicit per arm in schema v5. Fixed-count
behavior remains available without --adaptive-window-ms.
Progress messages occur outside timing windows. Output poisoning/readback
and independent checks remain after every event and wall window.

Final host-free contract/window lane: 95 pass, six gated skips. Mypy checks
all three changed compiler/recorder files with zero errors; Ruff is clean.
Both final packets are verified: 48 arms, 240 event windows and 240 wall
windows pass independent FP64 oracle checks. Every arm retains verified
Graph/Schedule/Tile/Target ancestry. Source files, both compiler binaries and
the native launch library match the recorded SHA256 hashes; see
packet-verification.txt. Event counts range from 1 to 100, wall counts from
1 to 20, and actual counts are recorded per arm. This is calibrated sampling,
not a fixed-count comparison or selector promotion.

The earlier uninstrumented fixed-count run was stopped without a packet.
Source and diagnostic logs remain in ignored scratch; they establish no
timing claim. Profiler diagnostics remain separate from final measurements.

Explicit LSE cotangents, general dynamic/composed AD, full-suite scaled-matmul
batching/transpose closure and publication remain open. No sibling device
proof is inferred.

## Larger backward profiles (CUDA-event median, milliseconds)

| Shape B,Hq,Hkv,Sq,Sk,D,Dv | Plain recompute | Plain saved | Bias recompute | Bias saved |
| --- | ---: | ---: | ---: | ---: |
| 1,4,2,127,131,64,64 | 308.020 | 2.408 | 319.472 | 2.405 |
| 1,4,2,256,256,64,64 | 1925.763 | 8.821 | 2023.076 | 8.871 |

These are separate backward kernels, not paired-program speedups. Saved
execution needs the forward O/LSE residual and its lifetime/storage cost.
Recompute backward's repeated row normalization dominates these long shapes;
native residual selection is a concrete compiler optimization follow-up.
Forward event and full checked host-wall timing are separate in the packets.
Tiny cases have about 1.1–1.3 ms host walls despite 11–25 us saved/recompute
event time, so host admission/marshalling remains a separate opportunity.
