# The typed route's K unroll by storage (2026-09-19)

Recorded by `benchmarks/rocm/record_typed_route_gap.py --dtype <storage>` on
Tajasarus (gfx1201, RX 9070 XT) and Princess-Luna (gfx1151, Strix Halo), both
ROCm 10.0 / HIP 7.15 under WSL2. Sync `GFX1201-PARITY-2026-09-17`.

One packet per (chip, storage), on the panel that chip's rule selects for it:
the 4x4 register panel on gfx1201, the 2x4 its integer branch ships on
gfx1151. K unroll 1, 2 and 4 per shape, two runs of 30 iterations each in
fresh processes, medians of interleaved rounds after a clock ramp. Host wall
clock: the HIP event timer returns garbage intervals on these WSL2 `/dev/dxg`
hosts, and neither box exposes `/dev/kfd`, so there are no counters.
`promotion.performance_eligible` is false on every row by construction.

Each storage brings its own reference and error budget, and the integer rows
are **exact** against an i32 reference at every K — an integer product has no
rounding, so an integer row with any error at all would be a wrong kernel
rather than a tolerance question.

**What the packets decided.** `scheduled_matmul.rocm_k_unroll` takes the
storage now, because the answer depends on it:

| chip | storage | 1024³ | 2048³ | 4096³ | rule |
|---|---|---|---|---|---|
| gfx1201 | fp16 | 46.4 / 58.7 / 57.5 | 51.6 / 88.2 / 73.7 | 65.2 / 90.3 / 77.8 | 2 |
| gfx1201 | fp8_e4m3 | 64.8 / 76.1 / 55.0 | 76.6 / 86.6 / 115.1 | 68.5 / 77.3 / 128.5 | 2, then 4 |
| gfx1201 | int8 | 63.3 / 73.5 / 54.5 | 76.1 / 85.9 / 115.2 | 68.5 / 77.3 / 129.1 | 2, then 4 |
| gfx1201 | int4 | 54.1 / 53.5 / 51.2 | 77.5 / 65.4 / 96.0 | 68.6 / 61.5 / 106.8 | 1, then 4 |
| gfx1151 | int8 | 9.6 / 17.0 / 15.2 | 21.0 / 22.6 / 14.3 | — | 2 in band |
| gfx1151 | int4 | 9.6 / 15.0 / 20.9 | 14.7 / 17.8 / 27.3 | — | 4 |

(TFLOP/s at k = 1 / 2 / 4. gfx1201's fp16 row is from the 2026-09-18 packet.)

**The mechanism, not a curve fit.** A fragment load is 8 elements per lane
whatever the storage, so fp16 saturates the 128-bit interface, fp8 and int8
use 64 bits, and int4 uses 32 (AMD's RDNA4 WMMA guide, part 2). A narrower
operand needs more slabs in flight to keep that path busy. **fp8 and int8
landing on the same rule at every shape is the check** — they are unrelated
storages of equal width.

**Margins inside spread are not taken.** gfx1151 int8 at 2048³ gains 7% from
k=2 and keeps k=1; gfx1201 int4 at 1024³ spans 6% across all three and keeps
k=1. Both are inside the run-to-run variation these hosts show.

**What this is a workaround for.** A deeper unroll issues *more* loads to
reach the bandwidth. AMD's extended-K technique instead fuses two WMMAs so a
single load fetches 16 elements and fills 128 bits, bit-identically, and for
int4 the hardware already has that shape as `V_WMMA_I32_16X16X32_IU4`.
Tessera cannot emit it while `materializeMma` pins `kBlocks = 1`, so these
numbers are the best the K=16 fragment can do, not the ceiling.
