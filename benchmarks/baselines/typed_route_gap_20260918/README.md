# The typed Tile route's f16 GEMM against the lanes it competes with (2026-09-18)

Recorded by `benchmarks/rocm/record_typed_route_gap.py` on Tajasarus
(gfx1201, RX 9070 XT) and Princess-Luna (gfx1151, Strix Halo), both ROCm 10.0
/ HIP 7.15 under WSL2. Sync `GFX1201-PARITY-2026-09-17`, the "typed-route
performance gap" item.

Per shape, in fresh processes (three runs, medians of interleaved rounds after
a clock ramp): the production scheduled package's Tile IR re-panelled to
`typed:16x16`, `typed:32x64` and `typed:64x64`; on gfx1151 the directive
lane's production kernel (`directive:<panel>`); on both chips the shipped
hand-written HIP GEMM (`shipped:register|lds|pipe`, timed by its own bench
entry). Host wall clock over 50 launches with a synchronize; the HIP event
timer on these WSL2 `/dev/dxg` hosts returns garbage intervals, and neither
box exposes `/dev/kfd`, so there are no counters.

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

`promotion.performance_eligible` is false on every row by construction: a
selection input where the gap dwarfs run-to-run spread, never a promotion.
