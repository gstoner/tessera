# Native pinned scaled-program staging candidate

Owner FRONTEND-IR-MEDIUM-1 / E2E-REAL-6 / ROCM-FP8-BLOCKSCALE-1.
Owning gfx1201 / RX 9070 XT. Native C++ owns reusable pinned input snapshots
and a pinned output staging buffer. Completion precedes overwriting or freeing
these buffers. Pinned memory is counted in the bounded idle-owner budget, and
transfer mode participates in the exact image/ABI ownership key.

Thirteen public JIT device cases pass, including independent block-scale
numerics, finite differences, changed inputs, active owner isolation, stale
handles/generations, cache cleanup and pinned/pageable cache separation.

Fresh-process off/on public warm medians, milliseconds:
| M/N/K | Pageable | Pinned |
| --- | --- | --- |
| 17/129/256 | 1.84467 | 1.68869 |
| 200/129/1536 | 6.22276 | 2.28880 |
| 33/257/2048 | 4.01668 | 2.31133 |

Native four-member event windows remain essentially unchanged (approximately
0.055, 0.251 and 0.313 ms); those include native enqueue gaps, not isolated
kernel measurements. The larger prepared update/invoke/two-readback costs
fall from 5.023/3.442 ms to 1.124/1.098 ms. The gains are host staging/transfer
attribution for these named scale-JVP cases, not new numerical kernel claims.

Candidate is opt-in with TESSERA_ROCM_PROGRAM_PINNED=1. Default remains
pageable. MXFP8/MXFP4, other program envelopes and sibling runtime validation
remain follow-ups before wider promotion. Generic batching/transpose and the
full five-slice completion gate remain open.
Recorder: benchmarks/rocm/benchmark_public_scaled_jvp.py --transposed-rhs.
