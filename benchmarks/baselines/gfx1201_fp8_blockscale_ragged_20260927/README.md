# ROCM-FP8-BLOCKSCALE-1 — gfx1201 W8A8: CU count from one authority, ragged M at the whole-M tile, short-K negatives, 2026-09-27

Sync `FOUNDATION-BATCH-2-2026-09-27` (follow-ups of `GFX1201-PERF-2026-09-27`,
PR #872). Host **Tajasarus** (Radeon RX 9070 XT, gfx1201, Ubuntu 26.04 under
WSL2, ROCm 10.0 / HIP 7.15). Worktree `~/programming/tessera-w-gaps`, branch
`claude/foundation-batch-2-gfx1201-gaps`. Two trees from the same source:
`build/` (matched LLVM/MLIR 23.1.1, NDEBUG; every timing) and
`build-assertions/` (assertions-ON LLVM/MLIR 23.1.1; tests and lit).

| File | Clean source commit | What |
|---|---|---|
| `ragged.json` | `2c9b1726` | ragged M (and five whole-M neighbours): production at this commit, production at the **previous compiler** (`@prev`), the previous ragged tile (128x64) forced under this compiler, AITER. 30 rows, 5 windows |
| `compare.json` | `2c9b1726` | the 54 (M, N, K) rows of PR #872's `compare.json`: production now vs `@prev` vs AITER, 5 windows |
| `isa.json`, `isa_previous.json` | `2c9b1726` | static ISA census of the selected kernels, this compiler and the previous one |
| `knobs_short_k.json` | `b6c39449` | item (c): grouped raster (4/8/16), 16-wave grids (128x128, 256x128, 128x256) at the six short-K / N=1024 shapes AITER still leads, plus four controls, 7 windows |
| `knobs_pipelining.json` | `b6c39449` | item (c): register-staged next slab (stage K 128 and 64), double-buffered LDS at stage K 64 (128x128 and 128x64), 5 windows |
| `identity_*.json` | final commit | every production kernel of `ragged.json` / `compare.json` recompiled at the final commit (see "Identity") |

**The previous compiler** is `tessera-opt` built from `12842e59` (the umbrella
base, i.e. PR #872's final code), sha256 `41f0d744…`; its arms are marked
`@prev` and were packaged without this tree's panel projection
(`panel_projection_checked: false` -- its rule differs from this tree's
oracle by design). AITER is `_gemm_a8w8_blockscale_kernel` from
`~/programming/aiter`, compiled **unmodified** by Triton 3.8.0 with that
checkout's tuned gfx1201 JSON (the same harness and checkout as PR #872).

**Timing:** the compiler-built device-clock marker (`llvm.readsteadycounter`,
100 MHz), HIP events and host wall per window; all arms warmed together, then
windows paired and interleaved. Every window in `ragged.json` and
`compare.json` is >= 7.04 ms; device clock and HIP event agree within 2.2% in
every window; every arm is `admissible`. Reported value: median per-launch
device-clock time. Every Tessera kernel is checked against the fp64 oracle
before timing (max relative error 1.4e-7).

## (a) The CU count has one authority

`selectFp8W8A8BlockScalePanel` hard-coded `kComputeUnits = 64`. It now reads
`measuredComputeUnits(arch)` (PMPasses.cpp), which mirrors the new
`rocm_target.compute_units(arch)` -- derived from the measured
`_DISPATCH_SLOTS` WGP table (CU = 2 x WGP on RDNA), not restated. Two guards:

* `tests/unit/test_rocm_fp8_blockscale.py::test_cpp_compute_units_mirror_rocm_target`
  parses the C++ table and requires it equal, entry for entry, to every part
  the Python table has measured (gfx1201 = 64, gfx1151 = 40).
* `lower_blockscale` checks the native Schedule's panel (staging, macro tile,
  warps) against `blockscale_panel_oracle`, the Python oracle of the rule, and
  refuses on divergence ("oracle disagrees with the native Schedule") -- the
  same differential guard split-K has. A 100-point grid (M 64..2048, N
  1024..24576, both weight layouts) agrees; a forged oracle is refused.

An arch without a measured count keeps the register panel with a registered
`ROCM_FP8_BLOCKSCALE_LDS_NOT_APPLIED` warning -- never a guessed denominator.
(The W8A8 derivation admits gfx1201 alone today, so that path is defensive.)

## (b) Ragged M: the masked edge was the cost, not the 128-row tile

PR #872 left ragged M at 1.075x AITER (geomean over 20 points), 1.37-1.56x at
N = 24576, K = 1536, and suggested a 64-row tile. Measured first
(exploration, not recorded here): 64x64/4-wave and 64x128/8-wave tiles moved
the ragged geomean only from 1.07 to 1.04 vs AITER, and not at all at
N = 24576 -- while a **ragged M = 1000 ran 1.25x slower than the whole
M = 1024 on the identical grid** (8 row blocks either way). The ISA says why:
the ragged 128x128 kernel took 251 VGPRs (whole 238) and the ragged 128x64
233 (whole 187). With the masked store removed (a diagnostic build) both
dropped to the whole-M counts, so the **bounded store** is the whole cost; the
row clamp in the staging copy is free.

**Root cause.** The block-scale join (`tile.fragment_scaled_accumulate`) asks
the accumulator map for each element's absolute row inside the K loop; LICM
hoists those per-element rows above the loop, and GVN hands them to the
bounded store, which was written against absolute rows -- one 64-bit value per
element held live across the whole K loop, only for the epilogue.

**Fix (TileToROCM, `materializeFragmentStore`).** A bounded store now tests
element `i`'s row as the constant `i * rowStep` against the lane's room
(`bound - origin - rowBase`, one value per lane per fragment) and addresses it
from the lane's row base plus a constant multiple of the leading dimension --
the same elements, predicates and addresses, in index arithmetic that cannot
wrap. The column keeps its absolute form (writing it in the lane form too made
the ragged-N 128x128 body 256 VGPRs plus spills). The accumulator map is
factored into one affine `accumulatorLaneMap` (base + i x step) that the
per-element forms and the store share; unbounded stores are built op for op as
before.

| kernel (`isa.json` / `isa_previous.json`) | previous VGPR | now VGPR | spills | instructions prev -> now |
|---|---:|---:|---:|---:|
| 1000x24576x1536, 128x128 | 251 | **240** | 0 | 2400 -> 2456 |
| 1000x24576x1536, 128x64 | 233 | **189** | 0 | 1419 -> 1431 |
| 256x4000x256 (ragged N), 128x128 | 251 | **240** | 0 | 2400 -> 2463 |
| whole-M 1024x24576x1536, 128x128 | 238 | 238 | 0 | byte-identical |
| register panel 32x32 (1024x4096x1024) | 231 | 231 | 0 | 2979 -> 2888 |

240 and 238 are the same allocation (a wave per SIMD more than 251's). Broader
static census (28 generic f16/bf16/fp8 gfx1201 matmul kernels, register and
LDS bodies, ragged and whole): VGPRs equal or lower everywhere; the spilling
1024^3 register panels spill less (f16 135 -> 125, bf16 143 -> 128, fp8
90 -> 74). Static counts, not timed.

**Rule change.** With the edge no longer costing a wave, ragged M follows the
whole-M rule: 128x128 whenever that tiling gives >= 64 workgroups, else
128x64 when that covers the CUs.

### Result (`ragged.json`)

| | rows | production / AITER geomean |
|---|---:|---:|
| ragged M, this compiler | 26 | **0.965** |
| ragged M, previous compiler | 26 | 1.086 |

Where the rule changes the selection (24 rows), production / the old 128x64
tile under this compiler is geomean **0.901 (0.83-1.08)**; it loses at
**200x8192x1024 (1.081)** and 600x8192x1024 (1.008) -- recorded, not tuned
around (a two-point K = 1024 carve-out would be fitting noise to a rule). The
whole-M neighbours (1024 rows) are byte-identical and time within 0.5%.

| M | N | K | production route | production us | previous route | previous us | 128x64 now us | AITER us | prod/prev | prod/AITER | prev/AITER |
|---:|---:|---:|---|---:|---|---:|---:|---:|---:|---:|---:|
| 200 | 8192 | 1024 | lds_128x128 | 45.8 | lds_128x64 | 42.6 | 42.3 | 44.4 | 1.076 | 1.031 | 0.959 |
| 300 | 8192 | 1024 | lds_128x128 | 57.5 | lds_128x64 | 60.8 | 59.7 | 56.9 | 0.946 | 1.010 | 1.068 |
| 600 | 8192 | 1024 | lds_128x128 | 97.6 | lds_128x64 | 99.3 | 96.8 | 96.4 | 0.983 | 1.012 | 1.030 |
| 1000 | 8192 | 1024 | lds_128x128 | 148.8 | lds_128x64 | 157.7 | 151.4 | 153.9 | 0.944 | 0.967 | 1.024 |
| 1024 | 8192 | 1024 | lds_128x128 | 131.5 | lds_128x128 | 130.9 | 152.2 | 155.7 | 1.004 | 0.844 | 0.840 |
| 1500 | 8192 | 1024 | lds_128x128 | 210.2 | lds_128x64 | 239.4 | 227.8 | 295.8 | 0.878 | 0.711 | 0.809 |
| 200 | 4096 | 7168 | lds_128x128 | 106.8 | lds_128x64 | 132.1 | 126.5 | 134.9 | 0.808 | 0.792 | 0.979 |
| 300 | 4096 | 7168 | lds_128x128 | 153.8 | lds_128x64 | 187.0 | 184.0 | 184.5 | 0.823 | 0.834 | 1.014 |
| 600 | 4096 | 7168 | lds_128x128 | 257.4 | lds_128x64 | 313.6 | 307.4 | 287.9 | 0.821 | 0.894 | 1.089 |
| 1000 | 4096 | 7168 | lds_128x128 | 406.6 | lds_128x64 | 500.9 | 489.3 | 460.8 | 0.812 | 0.882 | 1.087 |
| 1024 | 4096 | 7168 | lds_128x128 | 402.1 | lds_128x128 | 401.5 | 490.0 | 461.4 | 1.001 | 0.871 | 0.870 |
| 1500 | 4096 | 7168 | lds_128x128 | 605.1 | lds_128x64 | 750.6 | 730.6 | 691.8 | 0.806 | 0.875 | 1.085 |
| 200 | 24576 | 1536 | lds_128x128 | 142.6 | lds_128x64 | 170.7 | 164.0 | 147.4 | 0.835 | 0.967 | 1.158 |
| 300 | 24576 | 1536 | lds_128x128 | 214.3 | lds_128x64 | 256.8 | 246.6 | 160.4 | 0.835 | 1.336 | 1.601 |
| 600 | 24576 | 1536 | lds_128x128 | 361.0 | lds_128x64 | 437.8 | 419.9 | 312.2 | 0.825 | 1.157 | 1.402 |
| 1000 | 24576 | 1536 | lds_128x128 | 578.6 | lds_128x64 | 693.5 | 672.0 | 504.9 | 0.834 | 1.146 | 1.374 |
| 1024 | 24576 | 1536 | lds_128x128 | 552.5 | lds_128x128 | 553.2 | 667.7 | 504.8 | 0.999 | 1.094 | 1.096 |
| 1500 | 24576 | 1536 | lds_128x128 | 865.3 | lds_128x64 | 1042.3 | 1006.9 | 755.9 | 0.830 | 1.145 | 1.379 |
| 200 | 2048 | 2048 | lds_128x64 | 24.2 | lds_128x64 | 22.8 | 24.3 | 24.0 | 1.061 | 1.011 | 0.953 |
| 300 | 2048 | 2048 | lds_128x64 | 32.4 | lds_128x64 | 31.7 | 32.6 | 35.1 | 1.022 | 0.924 | 0.905 |
| 600 | 2048 | 2048 | lds_128x128 | 48.6 | lds_128x64 | 49.4 | 50.6 | 52.4 | 0.984 | 0.928 | 0.943 |
| 1000 | 2048 | 2048 | lds_128x128 | 75.9 | lds_128x64 | 76.9 | 76.8 | 77.5 | 0.987 | 0.980 | 0.993 |
| 1024 | 2048 | 2048 | lds_128x128 | 71.5 | lds_128x128 | 71.5 | 74.5 | 77.4 | 0.999 | 0.923 | 0.923 |
| 1500 | 2048 | 2048 | lds_128x128 | 100.1 | lds_128x64 | 115.4 | 112.0 | 109.0 | 0.868 | 0.919 | 1.059 |
| 300 | 4096 | 1024 | lds_128x128 | 30.8 | lds_128x64 | 33.3 | 32.9 | 29.2 | 0.924 | 1.052 | 1.138 |
| 1000 | 3072 | 1536 | lds_128x128 | 79.8 | lds_128x64 | 90.2 | 87.8 | 71.4 | 0.884 | 1.117 | 1.264 |
| 1000 | 6144 | 2048 | lds_128x128 | 189.0 | lds_128x64 | 228.5 | 221.3 | 196.0 | 0.827 | 0.964 | 1.166 |
| 1000 | 7168 | 2048 | lds_128x128 | 219.8 | lds_128x64 | 265.8 | 256.1 | 251.6 | 0.827 | 0.874 | 1.056 |
| 1000 | 1024 | 4096 | lds_128x128 | 68.7 | lds_128x64 | 79.7 | 75.3 | 82.8 | 0.862 | 0.830 | 0.963 |
| 600 | 4000 | 2048 | lds_128x128 | 85.4 | lds_128x64 | 96.4 | 94.3 | 89.9 | 0.886 | 0.950 | 1.072 |

**Measured negative.** At the two rows whose selection did not change and whose
kernel did (the store), 200x2048x2048 and 300x2048x2048 at 128x64, this
compiler is 1.06x / 1.02x the previous one (22.8 -> 24.2 us; the exploration
run measured the same 1.06x). These are one-wave-of-workgroups kernels where
the epilogue is not hidden; it is recorded as a small regression, not
explained. A 64-row LDS tile (64x64/4 waves, 64x128/8 waves) was measured in
exploration and is not selected: once the edge is free, 128x128 is faster at
every ragged point where it covers the CUs.

The ragged rows AITER still leads are the K = 1536 family (N = 24576: 1.15x;
1000x3072x1536: 1.12x) and 300x24576 (1.34x) -- the same short-K gap as whole
M, item (c).

### Whole M unchanged (`compare.json`)

| M bucket | rows | now / AITER geomean | previous / AITER | now / previous | now faster than AITER |
|---|---:|---:|---:|---:|---:|
| M <= 64 | 24 | 0.647 (0.36-1.08) | 0.648 | 0.994 (0.92-1.06) | 22 / 24 |
| M = 256 | 9 | 0.905 (0.80-0.99) | 0.901 | 1.004 (0.99-1.02) | 9 / 9 |
| M >= 1024 | 18 | 0.912 (0.68-1.08) | 0.911 | 1.001 (0.99-1.01) | 12 / 18 |

All 26 LDS-body production kernels are byte-identical to the previous
compiler's. The 28 register-panel kernels changed (their store is always
bounded; 2979 -> 2888 instructions, same VGPRs) and time at 0.92-1.06x of the
previous ones, geomean 0.995 -- the run-to-run spread PR #872 recorded for
these short kernels (0.98-1.05).

## (c) Short K / N = 1024: still open, the tried levers all measured negative

The six M >= 1024 shapes AITER leads are unchanged (all whole-M 128x128):
2048x1024x4096 1.04x, 1024x3072x1536 1.06x, 2048x3072x1536 1.05x,
2048x6144x2048 1.09x, 1024x24576x1536 1.08x, 2048x24576x1536 1.08x.

| lever (`knobs_*.json`, time / production) | range at the six shapes | verdict |
|---|---|---|
| grouped-M raster, group 4 / 8 / 16 | 0.98-1.01 / 0.97-1.00 / 0.98-1.05 | ~1-3% at best, not converging (as in PR #872) |
| 16 waves: 128x128 (32x32 waves), 256x128, 128x256 | 1.09-1.23 / 1.06-1.17 / 1.14-1.22 | negative |
| register-staged next slab, stage K 128 / 64 (5 of the 6 shapes) | 1.06-1.19 / 1.07-1.21 | negative |
| double-buffered LDS at stage K 64 (fits 64 KiB): 128x128 / 128x64 (5 of the 6) | 1.39-1.46 / 1.22-1.34 | negative |

The ISA shows where the time is not hidden: per scale group the body issues
its slab's `global_load_b128`s, meets the barrier, then waits on them before
the LDS store -- the global latency is covered only by other resident
workgroups -- and 14 of the 20 per-group scale loads are scheduled after the
WMMA chain with their waits right behind it. Moving that latency behind the
WMMAs (a register-staged slab, a second LDS buffer) is exactly what the table
measures as slower, so the obvious fixes are closed; split-K is not the lever
here (1024x3072 already has 192 workgroups). Open; no counters on WSL2 to
attribute it further.

## Identity

`identity_ragged.json` / `identity_compare.json`: every production kernel of
the two packets recompiled at `e2ec4529` and compared byte for byte with the
timed HSACO: **30/30 and 54/54 byte-identical**. The commits after `2c9b1726`
change the gfx11 element map's op order, a diagnostic registration, a test
expectation, the MXFP4 recorder and comments -- none of them the gfx1201 W8A8
kernels.

## Tests

- W8A8 + MXFP4 device and host (`tests/device/rocm/test_fp8_blockscale_w8a8.py`,
  `tests/unit/test_rocm_fp8_blockscale.py`,
  `tests/unit/test_rocm_pipeline_cache_key.py`,
  `tests/device/rocm/test_mxfp4_folded_prefill.py`,
  `tests/device/rocm/test_mxfp4_w4a8_exact.py`): **270/270 on both trees** at
  `e2ec4529` (`device_tests_gfx1201.txt`). New device rows: ragged M,
  ragged N and both at the Schedule-selected 128x128, bit for bit against the
  register panel and the fp64 oracle; a ragged kernel stays within 8 VGPRs of
  its whole-M twin and does not spill (three (N, K)).
- `lit tests/tessera-ir` 459 passed / 66 unsupported, `check-tessera-rocm`
  82/82, on both trees at `e2ec4529` (`lit_gfx1201.txt`).

## Not claimed

- Kernel device-clock time on one WSL2 host; not a model-level result.
- No profiler attribution (no `/dev/kfd`); the register reading is static.
- No gfx1151 evidence: the W8A8 contract is gfx1201-only. The TileToROCM
  bounded store is shared by gfx1151's typed stores; its lit fixtures pass,
  but no gfx1151 kernel was timed or census-checked here.
- The 28-kernel generic census is static; those kernels were not timed.

## Reproduce

```bash
# env: TESSERA_OPT at the worktree's build/, TESSERA_ROCM_CHIP=gfx1201,
# TESSERA_GFX1201_DEVICE_PROOF=1; AITER side: a venv with triton==3.8.0.
flock /tmp/tessera-timing.lock ~/wbs-aiter-venv/bin/python \
  benchmarks/rocm/benchmark_gfx1201_fp8_blockscale.py --compiler <final tessera-opt> \
  --alt-compiler prev=<tessera-opt at 12842e59> --aiter-root ~/programming/aiter \
  --shape M,N,K ... --no-production --with-aiter --sweep prod:nk --sweep prod:nk@prev \
  --sweep lds:128x64:8:1:-1:-1:-1:nk --windows 5 --output /tmp/ragged.json
TESSERA_LLVM_BIN=<llvm 23 bin> python benchmarks/rocm/inspect_gfx1201_fp8_blockscale_isa.py \
  --variant M,N,K:prod ... --output /tmp/isa.json          # --other-compiler for @prev
python benchmarks/rocm/check_gfx1201_fp8_blockscale_identity.py \
  --packet benchmarks/baselines/gfx1201_fp8_blockscale_ragged_20260927/ragged.json \
  --arm tessera_nk --output /tmp/identity.json
```
