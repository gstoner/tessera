# gfx1201 format and scheduling decision gates — 2026-10-03

Owner: ROCM-MXFP4-W4A8-1; FP8 comparator owner: ROCM-FP8-BLOCKSCALE-1.
Synchronization key: ROCM-FORMAT-STRATEGIES-2026-10-03.

No final schedule promotion is justified by this packet.

## Required format coverage

| Format | Current evidence | Remaining decision gate |
| --- | --- | --- |
| FP8 E4M3 W8A8 | 99 exact-device tests passed; five shapes checked against an independent blockscale oracle and measured against AITER | Broader short/long and ragged coverage, matched output and scale semantics |
| MXFP8 | Native textual Graph/Schedule/Tile/Target/HSACO has 12 exact gfx1201 cases; canonical bundled storage remains unpromoted | Checked package/runtime ABI, cache identity, kernel and end-to-end timing |
| MXFP4 | Explicit folded physical package, with 38 exact-device checks for ordered BF16 LDS staging | Retune across short/long envelopes and compare quality and performance under its explicit approximate contract |

The canonical dtype census is not a census of explicit physical packages. In particular, its MXFP4 unsupported entry does not erase the folded package evidence. FP8 performance does not substitute for native MXFP8 proof. The folded row-reference encoding maps code zero to zero; standard E8M0 semantics must be established separately for MXFP8.

## FP8 baseline

RX 9070 XT, live gfx1201, selected device 0. BF16-output production package versus AITER; values below are device-time ratios, smaller is faster. This is a fresh baseline of the existing FP8 route, not an improvement caused by the folded epilogue experiments.

| M × N × K | Tessera / AITER |
| --- | --- |
| 200 × 8192 × 1024 | 0.9513 |
| 200 × 2048 × 2048 | 0.9346 |
| 200 × 4096 × 1536 | 0.9307 |
| 256 × 4096 × 1024 | 0.8628 |
| 256 × 8192 × 5120 | 0.7808 |

Source, compiler, image and AITER fingerprints, numerical errors, paired device windows and separate public-runtime times are in fp8-key-shapes.json. The 99-case suite log is fp8-device-tests.txt. No evidence here transfers to gfx1151 or NVIDIA.

## Ordered C-through-LDS experiments

The prior wave-private prototype used wave barriers without explicit memory fences. Those numerical results did not prove LDS ordering. The ordered variants in this packet put wavefront release/acquire fences around both wave barriers per row stripe; exported LLVM IR preserves eight release and eight acquire fences.

BF16 LDS staging casts only after scale application and cold-path recovery. It preserves BF16 bits through LDS and the final store. Both ordered variants passed 38 exact-device cases, use 25,600 LDS bytes, have four static workgroup barrier pairs and no scratch spills. BF16 staging versus ordered f32 staging changes measured time by less than 0.8% on these four shapes.

BF16 staging versus original direct stores is slower by approximately 5.6%, 4.1%, and 0.8% on three shapes, and faster by 1.7% on the largest-N shape. This does not establish a short/long dispatch threshold.

The isolated A/B prologue overlap emits exactly the same native instruction fingerprint as the reference. Its paired timing ratios range from 0.991 to 1.018. This identical-code control shows that a roughly 1–2% timing change alone is insufficient for promotion.

Patches are experimental, not default route changes. Persistent scheduling has not been implemented or measured in this packet. A future native policy must carry verified worker geometry, lifetime/barrier rules and package identity through Schedule/Tile/Target IR.

Follow-up evidence: [standard E8M0 Tile consumer](../rocm_e8m0_tile_scale_20261003/README.md)
passes all-code, signed/zero, extreme-scale and ragged-edge checks. This is a
native Tile consumer foundation and does not close the MXFP8 Graph/package gate.

[Native MXFP8 compiler integration](../rocm_mxfp8_schedule_20261003/README.md)
now proves nonuniform multi-group accumulation and both KN/NK layouts through
HSACO. Its raw diagnostic HIP launcher does not close checked runtime packaging.
