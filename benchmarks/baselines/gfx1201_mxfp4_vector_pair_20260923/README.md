# GFX1201 packed-fragment paired-lane load ablation

Owner `ROCM-MXFP4-W4A8-1 / IKF-1`; sync
`GFX1201-PACKED-VECTOR-PAIR-2026-09-23`. The opt-in producer maps each
thread to two adjacent fragment-order lane words, issuing one 64-bit packed
weight load and paired 16-bit scale/reference loads. It does not change the
versioned physical ABI, approximate folding policy, default route, or selector.

Tajasarus RX 9070 XT/gfx1201 ran the two prefill shapes against the pinned
Radiance binary (`dfdfa383`). The benchmark supplied identical logical
quantized inputs, required an independent sampled exact FP32-dequantized
reference, and required bitwise BF16 agreement with Tessera's exact K32 route
before timing. Eleven interleaved HIP-event trials of twelve iterations each
produced these kernel medians (microseconds):

An additional five targeted exact-device tests passed for ragged N48/N80,
lossy folds, all E2M1 codes, exponent deltas, and reserved E8M0 zero blocks.

| M×N×K | Prior packed permute | Paired lane | Radiance | Paired / Radiance |
| --- | ---: | ---: | ---: | ---: |
| 256×5120×8704 | 186.4 | 174.1 | 137.6 | 1.27× |
| 1024×17408×5120 | 1160.8 | 1185.9 | 895.8 | 1.32× |

The selected timed ISA has one `global_load_b64` instead of two packed
`global_load_b32` sites, and paired scale reads appear as two
`global_load_u16` sites. Static `s_wait_loadcnt` remains 75; VGPR use rises
from 110 to 119, with 25,600 LDS bytes and no spills. The first shape gains
6.6% against the current packed-permute control; the wide shape regresses
2.2%. These static counts do not establish dynamic stalls or DRAM traffic.
The ablation remains manual and cannot justify automatic selection.

The full [packet](evidence.json) binds each timed Tessera HSACO payload to its
selected-symbol ISA digest, compiler and generator hashes, samples, and output
hash. Its `source_revision` is the merged-main base `96ce7f58`; the two
experiment sources were copied into an isolated Tajasarus worktree, and their
separate SHA-256 fields identify the actual tested versions. The Radiance
binary hash matches the prior pinned comparator packet. The distinct graph
admission receipt still refuses missing frontend and Radiance parity proof.

Next: inspect the selected A-stage address arithmetic and register lifetimes
against the pinned Radiance ISA, then test one base-address or occupancy change
at a time. Any candidate must win both shapes without weakening the exact
oracle or the source/HSACO binding.
