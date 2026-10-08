# Two-panel typed fragment lookahead on gfx1201

Owner ROCM-MXFP4-W4A8-1; sync ROCM-FOLDED-DEEP-FRAGMENTS-2026-10-02.

The candidate buffers current/next K16 fragments and distributes later LDS
reads between row-first WMMA chains with native scheduling groups. It differs
from the prior serialized K16 grouping. 89 focused tests pass, including
ragged rows/columns and exceptional scaling. Ordered full-K accumulation,
runtime ABI, and LDS lifetime barriers are preserved. Native HSACO instructions
match the LLVM diagnostic stream; virtual pressure falls to 154, physical
allocation stays 177 VGPRs, LDS 25,600 bytes, no spills.

Actual ISA: six ds_load_2addr_b64 and twelve ds_load_b64, versus twelve
ds_load_2addr_b64 in the reference. Both have 32 WMMA instructions and three
static split-barrier signal/wait pairs. The LDS access count covers the same
data; changed combining/distribution is not fewer operand bytes.

Seven alternating exact RX 9070 XT graph-window trials validate all poisoned
resident outputs before and after timing. Windows include GPU dispatch and
markers; public launch wall timing is separate. No profiler counter,
occupancy, speedup promotion, or Radiance comparison follows.

| M x N x K | Candidate us | Reference us | Ratio |
|---|---:|---:|---:|
| 256 x 4096 x 1024 | 23.853 | 23.060 | 1.0344 |
| 256 x 4096 x 2048 | 36.129 | 35.328 | 1.0227 |
| 256 x 4096 x 5120 | 77.130 | 77.317 | 0.9976 |
| 256 x 8192 x 5120 | 145.350 | 150.773 | 0.9640 |

The 3.6% N8192 improvement is paired with 2-3% short-K regressions.
Preserve candidate.patch for retuning; the default source is restored.
Exact-per-K32, broader shape/layout/model quality and performance closure
remain open. No sibling backend or gfx1151 physical proof follows.
