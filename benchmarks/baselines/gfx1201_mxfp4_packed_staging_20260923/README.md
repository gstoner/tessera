# GFX1201 packed-fragment producer staging ablation

Owner ROCM-MXFP4-W4A8-1 / IKF-1; sync
`GFX1201-PACKED-STAGING-ABLATION-2026-09-23`.

Run `benchmarks/rocm/benchmark_gfx1201_mxfp4_packed_folded.py` with
`--include-batched` to interleave the exact K32, expanded folded, packed
table, packed integer, B-only batching, A-only batching, A+B batching,
paired-K16 scale reuse, and pinned Radiance routes. Each shape must pass
the independent sampled exact oracle and bitwise BF16 agreement before HIP
event timing. Package/model-load work is outside the timed kernel. The
packet binds each selected-symbol ISA digest to its timed HSACO payload.

These are opt-in schedule ablations, not automatic selector candidates.
The versioned Graph physical layout and native ABI are unchanged. The
baseline v1 packet remains an immutable historical control; the v2 packet
records the current revision and every ablation on the exact gfx1201 host.

The clean packet comes from `2c1bacd5ad51f98015ae7eb816a51ca3118c6f78`
on Tajasarus RX 9070 XT/gfx1201. LLVM 23.1.1 and HIP 7.15.26333 were
probed before the run. Median kernel milliseconds (11 interleaved trials,
12 iterations each):

| Route | 256×5120×8704 | 1024×17408×5120 |
|---|---:|---:|
| Expanded folded | 0.1728 | 1.1314 |
| Packed integer control | 0.2420 | 1.4167 |
| Packed B-only batched | 0.2326 | 1.3990 |
| Packed A-only batched | 0.2402 | 1.4385 |
| Packed A+B batched | 0.2340 | 1.4145 |
| Packed paired-K16 scale reuse | 0.2297 | 1.4305 |
| Radiance | 0.1383 | 0.8912 |

B-only batching improves the packed control by 3.9% and 1.3%, but remains
1.68× and 1.57× Radiance and 1.35× and 1.24× Tessera expanded folded.
The selected packed B-only ISA has 87 static `s_wait_loadcnt` instructions
versus 88 for control, 118 versus 117 VGPRs, 32 FP8 WMMA instructions,
25,600 LDS bytes, and no spills. A-only batching does not reduce the
static wait count and loses on wide N. Paired-scale reuse also loses on
wide N; the compiler emits the same four `global_load_u8` instructions as
B-only, so source-level hoisting did not reduce that selected ISA count.
Static instruction counts are not dynamic stall or DRAM measurements.
No variant merits automatic selector admission; the next experiment should
attack per-word decode dependencies without expanding all B weights.
