# gfx1201 MXFP4 M=256 per-column decomposition, 2026-09-28

Owner `ROCM-MXFP4-W4A8-1`. Tajasarus RX 9070 XT (`gfx1201`, WSL2), pinned
Radiance revision `dfdfa3832922c9a4253133f09c1f5c0d39748fc7` and binary
SHA-256 `c9f91bc8...` (the same binary as the 2026-09-27 N scan). The
[`nscan_decomposition.json`](nscan_decomposition.json) packet records two
independent processes, seven alternating device-clock/HIP-event trials per
shape, one rotating input copy, K=5120 and N=8192/12288/17408. The exact
K32 route passes an independent sampled reference; every timed engine's BF16
output is bitwise equal to it. The source tree is dirty and this packet is
marked diagnostic, so it does not promote a route.

The recorder's selected schedule is ablated one key at a time. The marginal
time from N=8192 to N=17408 is in nanoseconds per added output column:

| Engine | Process 1 | Process 2 |
| --- | ---: | ---: |
| Radiance | 13.60 | 13.43 |
| Tessera selected | 18.09 | 17.99 |
| Selected without vector epilogue | 17.69 | 18.10 |
| Selected without next-slab prefetch | 18.82 | 18.62 |
| Selected without grouped raster | 17.93 | 17.83 |

The vector epilogue and prefetch both lower absolute latency, but removing
either leaves the per-column gap to Radiance. Grouped raster has negligible
effect at this single M row block. This narrows the remaining cost to the
unchanged core loop or launch geometry; it does not distinguish A-tile
restaging, LDS fragment traffic, instruction issue or their interaction.
No hardware counters are available on this WSL2 host. The earlier
[residency scan](../gfx1201_mxfp4_one_row_block_20260927/README.md)
already ruled out packed weight bytes as the dominant cause. Exact K32 remains
the default and the folded route remains opt-in.
