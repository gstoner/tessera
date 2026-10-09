# NVFP4 winner-code attribution on gfx1201

Owner: ROCM-NVFP4-INGEST-1. Sync: ROCM-NVFP4-WINNER-CODES-20261009.
Recorder: benchmarks/rocm/record_nvfp4_winner_codes.py.
Control source: PR914 a9bec72ad0f48f8a7522b77c238ff789ebc6457c.
Both candidate patches apply to that unchanged control. Production lowering
was restored after measurement; neither candidate is promoted.

The ordinary frontend NVFP4 converter -> folded MXFP4 storage -> FP8 activation
matmul program traverses native Graph/Schedule/Tile/ROCm/LLVM/HSACO. Both arms use
the same HIP runtime, operands, storage image and consumer image. Python supplies
the independent oracle and measurement orchestration.

Recompute-codes retains only the winning exponent then computes final codes.
Packed-winner retains four words rather than 32 scalar codes across selection.
Both preserve candidate SSE/reduction order, midpoint/exponent ties, negative
codes and zero codes when no candidate is selected.

Each candidate passed 20 host WSL native lowering and 55 owning-device checks:
raw scale bytes, signed/zero codes, extreme projection globals, public/portable
packages, changing bounded rows and allocation ownership. The paired packets
also prove packed codes, exponents, f64 statistics, stored buffers and output
bitwise equal, with independent numerical checks before/after timing.

## Matched captured timings

Ratios are control median / candidate median; below 1 means candidate is slower.
Seven AB/BA rounds, 128 repetitions per event window; direct samples are retained
separately. Converter and full-program windows are measured independently.

| Candidate | M,N,K | Converter ratio | Full native program ratio |
| --- | --- | --- | --- |
| recompute-codes | [256, 64, 1024] | 0.9794 | 0.9844 |
| recompute-codes | [256, 512, 1024] | 0.7700 | 0.8079 |
| recompute-codes | [256, 1024, 4096] | 1.3265 | 1.0041 |
| packed-winner | [256, 64, 1024] | 0.9663 | 0.9706 |
| packed-winner | [256, 512, 1024] | 0.9538 | 0.9622 |
| packed-winner | [256, 1024, 4096] | 0.9588 | 0.9529 |

Recomputation reduced VGPR metadata 256 -> 212 and private-segment metadata
180 -> 0 bytes. Packed words kept 256 VGPRs and reduced private-segment metadata
180 -> 124 bytes. Neither improved every profile; packed words regressed all
three. Resource metadata is not a counter/occupancy certificate. The larger
recomputation converter improved but its full-program ratio was about 1.004,
insufficient to establish meaningful program benefit. Stage windows cannot be
summed or treated as one simultaneous clock state. Default policy is unchanged.
Scratch elimination alone is insufficient for promotion; no unique instruction
bottleneck or short/long strategy closure is claimed.

The source-age warning is expected for the deliberately retained control build.
Tool/component/input hashes, runtime identity and live RX 9070 XT HIP UUID are
recorded separately. These are explicit NVFP4 conversion/folded-row approximate
consumer contracts, with no original-BF16/model-quality or sibling-device claim.

Reproduce from the control revision and one applied candidate patch, matching
LLVM/MLIR 23.1.1 full core/ROCm tools and the same PR914 leaf-ready HIP provider.
Set image/movement provider variables and invoke:

    python benchmarks/rocm/record_nvfp4_winner_codes.py --control /scratch/control-build --candidate /scratch/candidate-build --output /scratch/comparison.json

Remaining: attribute quantization/SSE cost and short/long dispatch, then require
correctness and matched full-program benefit before changing policy. NVIDIA
producer/attention, generic frontend/AD batching/transpose, wider ROCm routes,
MXFP4 A-fetch/LDS and W8A8 obligations remain open.
