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

*The LDS-staged body never wins a selection, and how it loses differs by
chip.* It is correct on device on both chips with multi-wave workgroups. On
gfx1201 it is 0.23x-0.83x of the register body at the same panel, at every
panel and wave count here. On gfx1151 it *does* beat the register body at the
same panel -- up to 1.26x, and mostly at the 1x1 panel, where the register
body is weakest -- but it never beats the best register configuration at a
shape: its best 1024^3 row is 12.57 against the selected 4x4 k=2 body's 15.75.
So staging B transposed really does remove the strided gather; the barriers
just cost more than the stride does wherever the register body is already
competent. It ships as a packaging option, selected nowhere.

*The panel axis is exhausted.* gfx1201's 4096^3 single-slab row climbs
14.5 -> 57.2 -> 65.2 TFLOP/s across 1x1, 2x4 and 4x4 and stops there. An
exploratory sweep beyond these panels (64x128, 128x128) fell off a VGPR cliff
rather than continuing, which is why the packet's panel list ends at 64x64.

*The K unroll is the lever.* TFLOP/s at k = 1 / 2 / 4, on the panel each chip
selects -- gfx1201 1024^3 46.4 / 58.7 / 57.5, 2048^3 51.6 / 88.2 / 73.7,
4096^3 65.2 / 90.3 / 77.8; gfx1151 1024^3 10.8 / 15.8 / 11.0 (directive lane
11.6), 2048^3 21.2 / 20.3 / 9.8 (directive lane 23.3). Both chips take 2, and
only where they also take the larger panel; above 2048 on gfx1151 the
directive lane still leads so the knob stays off. Each chip's rule comes from
its own sweep.

**A withdrawn row, kept as a record.** An earlier cut of this packet took k=4
below 2048 on gfx1201 on the strength of one row: 1024^3 reading 61.5 for k=4
against 56.0 for k=2. The re-record reversed it to 58.7 against 57.5. Two runs
disagreeing on the sign of a 2% margin means the margin is noise, so the rule
now takes k=2 wherever gfx1201 takes the 4x4 panel -- the result that
reproduces, and one branch fewer.

**The denominator these numbers are missing until you look it up.** AMD's
published dense RDNA4 WMMA peak on gfx1201 is 191 TFLOP/s for f16/bf16, so
the best row here (90.3) is 47% of it. The table for every storage, and what
follows from fp8 and int4 having higher ceilings than f16, is in
`docs/backends/rocm/wmma-fragment-layout.md`.

`promotion.performance_eligible` is false on every row by construction: a
selection input where the gap dwarfs run-to-run spread, never a promotion.
