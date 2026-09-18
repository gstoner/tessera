# The typed Tile route's f16 GEMM against the lanes it competes with (2026-09-18)

Recorded by `benchmarks/rocm/record_typed_route_gap.py` on Tajasarus
(gfx1201, RX 9070 XT) and Princess-Luna (gfx1151, Strix Halo), both ROCm 10.0
/ HIP 7.15 under WSL2. Sync `GFX1201-PARITY-2026-09-17`, the "typed-route
performance gap" item.

Per shape, in fresh processes (three runs, medians of interleaved rounds after
a clock ramp): the production scheduled package's Tile IR re-panelled to
`typed:16x16`, `typed:32x64` and `typed:64x64`, each at K unroll 1, 2 and 4
(`typed:<panel>:k<n>`); the same Tile IR through the LDS-staged multi-wave
body at 2x2 and 4x2 waves (`typed-lds:<panel>:<WMxWN>`); on gfx1151 the
directive lane's production kernel (`directive:<panel>`); on both chips the
shipped hand-written HIP GEMM (`shipped:register|lds|pipe`, timed by its own
bench entry). Host wall clock over 50 launches with a synchronize; the HIP
event timer on these WSL2 `/dev/dxg` hosts returns garbage intervals, and
neither box exposes `/dev/kfd`, so there are no counters. A variant the
pipeline refuses is recorded with its refusal rather than dropped.

**What the packet decided.** gfx1201's f16/bf16 macro tile is now
shape-selected in `PMPasses.cpp` (mirrored by
`scheduled_matmul.rocm_gfx1201_panel`): the 2x4 register panel for a static,
fully tiled problem at 1024 and above (2.1x at 1024^3, 3.3x at 2048^3), the
1x1 otherwise (a wash at 512^3, slower on ragged shapes). Before this packet
the panel was fixed at 1x1 and the 2x4 body was refused on gfx12 by the
gfx1151 performance-closure stamp.

**What the first cut of the packet got wrong, kept as a record.** The first
run varied a `staging` option and read a 3.4x gap between `register` and
`lds`. The two kernels were byte-identical (the option reaches only the
generator's canonical scf.for body, which the typed route never enters); the
gap was the first variant timed per shape paying the clock ramp. The recorder
now compiles and loads everything before timing, ramps the clocks, and
interleaves rounds; the gfx1151 package's `physical_route` label, which had
claimed an LDS route, now names the register panel it builds.

**What the second cut decided (2026-09-18, same day).** Three things, in
the order they were measured.

*The LDS-staged body loses.* It is correct on device on both chips with
multi-wave workgroups, and slower than the register body everywhere: 0.39x to
0.65x on gfx1201, and on gfx1151 a narrow 1024^3 win that the K unroll then
beat. Staging B transposed through shared memory does remove the strided
gather, but the barriers cost more than the stride does. It ships as a
packaging option, selected nowhere.

*The panel axis is exhausted.* On gfx1201 64x64 peaks at 64.1 TFLOP/s; 64x128
falls to 15.6 and 128x128 to 7.4. That is a VGPR cliff, not a trend, so there
is no larger tile to reach for.

*The K unroll is the lever.* TFLOP/s at k = 1 / 2 / 4 -- gfx1201 1024^3
42.1 / 56.0 / 61.5, 2048^3 43.3 / 86.4 / 74.1, 4096^3 65.5 / 92.5 / 77.4;
gfx1151 1024^3 11.7 / 16.9 / 11.8 (directive lane 12.2), 2048^3
17.8 / 19.2 / 17.2 (directive lane 22.1). gfx1201 takes 4 below 2048 and 2
from 2048 up; gfx1151 takes 2 only in the [1024, 2048) band where it also
takes the 4x4 panel, and above that the directive lane still leads so the
knob stays off. Each chip's rule comes from its own sweep.

**The denominator these numbers are missing until you look it up.** AMD's
published dense RDNA4 WMMA peak on gfx1201 is 191 TFLOP/s for f16/bf16, so
the best row here (92.5) is 48% of it. The table for every storage, and what
follows from fp8 and int4 having higher ceilings than f16, is in
`docs/backends/rocm/wmma-fragment-layout.md`.

`promotion.performance_eligible` is false on every row by construction: a
selection input where the gap dwarfs run-to-run spread, never a promotion.
