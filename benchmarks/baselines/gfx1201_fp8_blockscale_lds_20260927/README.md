# ROCM-FP8-BLOCKSCALE-1 — gfx1201 W8A8 large-M LDS body and bf16 store vs AITER, 2026-09-27

Sync `GFX1201-PERF-2026-09-27`. Host **Tajasarus** (Radeon RX 9070 XT, gfx1201,
Ubuntu 26.04 under WSL2, kernel `6.18.33.2-microsoft-standard-WSL2`, ROCm 10.0 /
HIP 7.15). Worktree `~/programming/tessera-w-perf` on branch
`claude/gfx1201-perf-w8a8-mxfp4`. Two trees from the same source: `build/`
(matched LLVM/MLIR 23.1.1, NDEBUG; every timing) and `build-assertions/`
(assertions-ON LLVM/MLIR 23.1.1, `-fno-rtti -UNDEBUG`; tests and lit).

| File | Clean source commit | What |
|---|---|---|
| `compare.json` | `79a641b1` | Tessera production `[N, K]` f32 and bf16, the previous production route, and AITER, 54 (M, N, K) rows, 10 windows |
| `identity_compare.json` | `70650bb2` | all 108 Tessera production kernels of `compare.json` recompiled at the final compiler: **108/108 byte-identical HSACOs** |
| `rule.json` | `a1f4e5e7` | the Schedule rule: production vs forced register panel vs both LDS tilings, 32 rows, 5 windows |
| `identity_rule.json` | `70650bb2` | 30/32 production kernels byte-identical at the final compiler; the 2 that differ are the ragged-M rows the later rule change moved (re-measured in `ragged.json`) |
| `ragged.json` | `ed8d6da8` | ragged M: production (final rule), both LDS tilings, both tilings under the pre-clamp compiler, AITER, 20 rows, 5 windows |
| `knobs.json` | `29926fcb` | LDS-body performance keys at four shapes, 5 windows, one arm under the first-cut compiler |
| `isa.json` | `70650bb2` | static ISA census of the selected kernels (not a runtime measurement) |
| `device_tests_gfx1201.txt`, `lit_gfx1201.txt` | `70650bb2` | device + host tests and lit on both trees |

Changes between the timed commits and the final one are either benchmark-only
or change only ragged-M code (the rule and the copy clamp); the two identity
files are what show the timed kernels are the final route's.

## What was measured

- **Tessera** -- the production route: `tessera.scaled_matmul` (e4m3 A and
  weight `[N, K]`, fp32 scales, `scale_layout.block = [128, 128]`) compiled
  Graph -> Schedule -> Tile -> Target -> HSACO by
  `tessera.compiler.rocm_fp8_blockscale`. The Schedule picks the body: the
  one-wave register panel, or (new) the LDS-staged multi-wave body. Output f32
  (`tessera_nk`) or bf16 (`tessera_nk+bf16`, new: the Graph result type).
- **Before** -- `tessera_nk@prev`: the same production route compiled by
  `/tmp/wperf-final/tessera-opt-prev` (commit `3747b40f`, whose Schedule rule
  and register body are the ones this work started from), in the same process.
- **AITER** -- `_gemm_a8w8_blockscale_kernel` from `~/programming/aiter` at
  `f0966b0a` (kernel source sha256 `f4824524…`), compiled **unmodified** by
  Triton 3.8.0 for gfx1201 with that checkout's tuned gfx1201 JSON. Weight
  `[N, K]`, bf16 output. Split-K buckets (`NUM_KSPLIT > 1`) need AITER's reduce
  kernel, which the harness does not drive: those 3 rows are **not measured**.
- Every kernel is checked against an fp64 oracle before timing: max relative
  error 1.3e-7 for the f32 stores, 1.6e-3 for the bf16 stores (Tessera and
  AITER alike).
- **Timing source: the compiler-built device-clock marker**
  (`--tessera-device-clock-span`, `llvm.readsteadycounter` at
  `hipDeviceAttributeWallClockRate` = 100000 kHz), HIP events and host wall as
  per-window cross-checks. All arms warmed together, then windows **paired and
  interleaved**. In every packet every window is >= 5 ms (shortest 6.23 ms, in
  `compare.json`), device clock and HIP event agree within 3.95% in every
  window, and every arm is `admissible`. Reported value: median per-launch device-clock time.

## What changed

1. **The LDS-staged multi-wave body** (`emitTypedLdsBlockScaleBody`). Eight
   waves of 32 rows share one staged K slab of A `[wgM][128]` and of the
   `[N, K]` weight `[wgN][128]` (16-byte rows of padding; 16-byte global loads
   and `ds_store_b128`; fragments read from LDS). Each wave keeps the register
   body's semantics exactly: every 128-wide scale group starts a zero partial,
   walks its eight instruction panels in ascending K and joins the
   accumulator once through `tile.fragment_scaled_accumulate`. Barriers fence
   LDS only. Rows past M / N are clamped, not zeroed (they reach no stored
   element).
2. **The Schedule rule** (`selectFp8W8A8BlockScalePanel`, `[N, K]` only, M >=
   128): 128x128 (waves 4x2 of 32x64) when that tiling gives >= 64
   workgroups -- the RX 9070 XT's CU count -- and M is a whole number of 128-row
   blocks; otherwise 128x64 (waves 4x2 of 32x32) when that covers the CUs;
   otherwise the register panel. `staging`, `warps` and `pipeline_depth` are
   stated on `schedule.matmul` (digested when set), on the Tile carrier and on
   the Target directive, which is what the package binds its 256-thread
   workgroup from.
3. **A bf16 store epilogue.** The W8A8 contract now admits a bf16 Graph
   result: the fp32 accumulator is rounded once, to nearest-even, by the typed
   store (`arith.truncf`), under distinct package ABIs
   (`...e4m3_e4m3_f32_bf16.wmma_exact.v1`).

## Result: large M now beats AITER on most shapes

Tessera `[N, K]` time / AITER time (< 1 = Tessera faster), 51 measured rows:

| M bucket | rows | f32 geomean | bf16 geomean | before geomean | f32 faster |
|---|---:|---:|---:|---:|---:|
| M <= 64 | 24 | 0.653 (0.37-1.11) | 0.655 | 0.653 | 22 / 24 |
| M = 256 | 9 | **0.899** (0.80-1.01) | 0.892 | 1.075 | 8 / 9 |
| M >= 1024 | 18 | **0.911** (0.68-1.08) | 0.907 | 1.291 | 12 / 18 |

At the 26 rows where the rule selects the LDS body, new / before is geomean
0.739 (0.46-0.94). At the 28 register-panel rows the two compilers emit
byte-identical HSACOs, and their new / before ratio (0.98-1.05) is this
packet's run-to-run spread for short kernels.

**The bf16 store buys nothing measurable**: bf16 / f32 geomean 0.998
(0.95-1.10) over all 54 rows. The output bytes the f32 store writes were not
what separated us from AITER; the comparison with bf16 on both sides is now
matched-output and says the same thing.

Where AITER still wins at M >= 1024 (6 of 18, 1.04-1.08x) the K is short
(1536-2048, or 4096 with N = 1024): K = 1536-2048 is 12-16 scale groups, so the
per-workgroup prologue, epilogue and 16 joins weigh more.

### Rows

| M | N | K | Tessera route (Schedule) | AITER config | AITER us | f32 us | f32/AITER | bf16 us | bf16/AITER | before us | before/AITER |
|---:|---:|---:|---|---|---:|---:|---:|---:|---:|---:|---:|
| 16 | 4096 | 1024 | register_1x2_k128 | N=4096-K=1024:M_LEQ_16 | not measured (split-K bucket) | 13.3 | – | 13.5 | – | 13.2 | – |
| 32 | 4096 | 1024 | register_1x2_k128 | N=4096-K=1024:M_LEQ_32 | 12.6 | 14.0 | 1.11 | 13.5 | 1.07 | 14.0 | 1.11 |
| 64 | 4096 | 1024 | register_2x2_k128 | N=4096-K=1024:M_LEQ_1024 | 16.7 | 13.1 | 0.78 | 12.8 | 0.77 | 13.1 | 0.79 |
| 256 | 4096 | 1024 | lds_128x128_w8_d1_k128 | N=4096-K=1024:M_LEQ_1024 | 24.9 | 22.3 | 0.89 | 21.4 | 0.86 | 24.8 | 0.99 |
| 1024 | 4096 | 1024 | lds_128x128_w8_d1_k128 | N=4096-K=1024:M_LEQ_1024 | 84.2 | 66.5 | 0.79 | 66.3 | 0.79 | 83.8 | 1.00 |
| 2048 | 4096 | 1024 | lds_128x128_w8_d1_k128 | N=4096-K=1024:M_LEQ_2048 | 154.6 | 128.1 | 0.83 | 127.6 | 0.83 | 158.1 | 1.02 |
| 16 | 2048 | 2048 | register_1x2_k128 | N=2048-K=2048:M_LEQ_16 | 13.5 | 9.1 | 0.67 | 9.2 | 0.68 | 9.1 | 0.67 |
| 32 | 2048 | 2048 | register_1x2_k128 | N=2048-K=2048:M_LEQ_32 | 13.6 | 9.3 | 0.68 | 9.3 | 0.68 | 9.2 | 0.68 |
| 64 | 2048 | 2048 | register_1x2_k128 | N=2048-K=2048:M_LEQ_128 | 13.7 | 10.1 | 0.74 | 10.2 | 0.75 | 10.1 | 0.74 |
| 256 | 2048 | 2048 | lds_128x64_w8_d1_k128 | N=2048-K=2048:M_LEQ_256 | 23.8 | 24.0 | 1.01 | 23.7 | 1.00 | 25.6 | 1.07 |
| 1024 | 2048 | 2048 | lds_128x128_w8_d1_k128 | N=2048-K=2048:M_LEQ_1024 | 77.6 | 70.8 | 0.91 | 72.9 | 0.94 | 77.5 | 1.00 |
| 2048 | 2048 | 2048 | lds_128x128_w8_d1_k128 | N=2048-K=2048:M_LEQ_2048 | 145.9 | 123.1 | 0.84 | 123.6 | 0.85 | 148.8 | 1.02 |
| 16 | 6144 | 2048 | register_1x2_k128 | N=6144-K=2048:M_LEQ_16 | 28.6 | 17.4 | 0.61 | 17.3 | 0.61 | 17.5 | 0.61 |
| 32 | 6144 | 2048 | register_1x2_k128 | N=6144-K=2048:M_LEQ_32 | 29.7 | 16.2 | 0.55 | 17.8 | 0.60 | 16.3 | 0.55 |
| 64 | 6144 | 2048 | register_2x2_k128 | N=6144-K=2048:M_LEQ_128 | 35.2 | 21.8 | 0.62 | 22.1 | 0.63 | 22.2 | 0.63 |
| 256 | 6144 | 2048 | lds_128x128_w8_d1_k128 | N=6144-K=2048:M_LEQ_256 | 63.2 | 53.3 | 0.84 | 52.8 | 0.83 | 66.6 | 1.05 |
| 1024 | 6144 | 2048 | lds_128x128_w8_d1_k128 | N=6144-K=2048:M_LEQ_1024 | 198.7 | 182.1 | 0.92 | 182.7 | 0.92 | 247.6 | 1.25 |
| 2048 | 6144 | 2048 | lds_128x128_w8_d1_k128 | N=6144-K=2048:M_LEQ_2048 | 336.0 | 358.5 | 1.07 | 362.3 | 1.08 | 492.2 | 1.47 |
| 16 | 3072 | 1536 | register_1x2_k128 | N=3072-K=1536:M_LEQ_16 | 11.7 | 7.7 | 0.65 | 7.8 | 0.66 | 7.7 | 0.66 |
| 32 | 3072 | 1536 | register_1x2_k128 | N=3072-K=1536:M_LEQ_32 | 12.2 | 8.0 | 0.66 | 8.1 | 0.67 | 8.0 | 0.65 |
| 64 | 3072 | 1536 | register_1x2_k128 | N=3072-K=1536:M_LEQ_64 | 12.3 | 11.0 | 0.89 | 10.8 | 0.87 | 10.9 | 0.88 |
| 256 | 3072 | 1536 | lds_128x64_w8_d1_k128 | N=3072-K=1536:M_LEQ_256 | 29.4 | 25.8 | 0.88 | 25.3 | 0.86 | 27.5 | 0.94 |
| 1024 | 3072 | 1536 | lds_128x128_w8_d1_k128 | N=3072-K=1536:any | 71.4 | 74.9 | 1.05 | 75.4 | 1.06 | 92.8 | 1.30 |
| 2048 | 3072 | 1536 | lds_128x128_w8_d1_k128 | N=3072-K=1536:any | 132.9 | 140.0 | 1.05 | 141.1 | 1.06 | 188.9 | 1.42 |
| 16 | 1024 | 4096 | register_1x2_k128 | N=1024-K=4096:M_LEQ_16 | not measured (split-K bucket) | 16.0 | – | 15.8 | – | 15.1 | – |
| 32 | 1024 | 4096 | register_1x2_k128 | N=1024-K=4096:M_LEQ_32 | not measured (split-K bucket) | 17.2 | – | 16.9 | – | 17.4 | – |
| 64 | 1024 | 4096 | register_1x2_k128 | N=1024-K=4096:M_LEQ_1024 | 26.9 | 17.6 | 0.66 | 17.8 | 0.66 | 17.6 | 0.66 |
| 256 | 1024 | 4096 | register_2x2_k128 | N=1024-K=4096:M_LEQ_1024 | 31.3 | 30.1 | 0.96 | 30.3 | 0.97 | 30.1 | 0.96 |
| 1024 | 1024 | 4096 | lds_128x128_w8_d1_k128 | N=1024-K=4096:M_LEQ_1024 | 75.7 | 67.3 | 0.89 | 65.7 | 0.87 | 101.7 | 1.34 |
| 2048 | 1024 | 4096 | lds_128x128_w8_d1_k128 | N=1024-K=4096:M_LEQ_2048 | 125.9 | 132.1 | 1.05 | 133.8 | 1.06 | 198.2 | 1.57 |
| 16 | 4096 | 7168 | register_1x2_k128 | N=4096-K=7168:M_LEQ_16 | 67.1 | 24.8 | 0.37 | 25.1 | 0.37 | 24.2 | 0.36 |
| 32 | 4096 | 7168 | register_1x2_k128 | N=4096-K=7168:M_LEQ_32 | 67.0 | 27.0 | 0.40 | 28.2 | 0.42 | 26.7 | 0.40 |
| 64 | 4096 | 7168 | register_2x2_k128 | N=4096-K=7168:M_LEQ_64 | 72.6 | 37.7 | 0.52 | 37.9 | 0.52 | 37.3 | 0.51 |
| 256 | 4096 | 7168 | lds_128x128_w8_d1_k128 | N=4096-K=7168:M_LEQ_256 | 137.5 | 109.8 | 0.80 | 109.1 | 0.79 | 163.2 | 1.19 |
| 1024 | 4096 | 7168 | lds_128x128_w8_d1_k128 | N=4096-K=7168:any | 455.8 | 407.5 | 0.89 | 402.8 | 0.88 | 886.0 | 1.94 |
| 2048 | 4096 | 7168 | lds_128x128_w8_d1_k128 | N=4096-K=7168:any | 922.5 | 816.6 | 0.89 | 791.5 | 0.86 | 1653.3 | 1.79 |
| 16 | 7168 | 2048 | register_1x2_k128 | N=7168-K=2048:M_LEQ_16 | 32.7 | 19.4 | 0.60 | 19.3 | 0.59 | 19.5 | 0.60 |
| 32 | 7168 | 2048 | register_1x2_k128 | N=7168-K=2048:M_LEQ_32 | 31.0 | 19.6 | 0.63 | 20.1 | 0.65 | 19.7 | 0.63 |
| 64 | 7168 | 2048 | register_2x2_k128 | N=7168-K=2048:M_LEQ_64 | 32.1 | 25.3 | 0.79 | 25.5 | 0.80 | 25.8 | 0.80 |
| 256 | 7168 | 2048 | lds_128x128_w8_d1_k128 | N=7168-K=2048:M_LEQ_256 | 72.6 | 65.5 | 0.90 | 66.7 | 0.92 | 84.6 | 1.16 |
| 1024 | 7168 | 2048 | lds_128x128_w8_d1_k128 | N=7168-K=2048:any | 254.6 | 211.5 | 0.83 | 211.2 | 0.83 | 322.0 | 1.26 |
| 2048 | 7168 | 2048 | lds_128x128_w8_d1_k128 | N=7168-K=2048:any | 501.4 | 422.2 | 0.84 | 417.7 | 0.83 | 640.9 | 1.28 |
| 16 | 8192 | 1024 | register_1x2_k128 | N=8192-K=1024:M_LEQ_16 | 20.0 | 21.4 | 1.07 | 21.5 | 1.08 | 21.5 | 1.08 |
| 32 | 8192 | 1024 | register_2x2_k128 | N=8192-K=1024:M_LEQ_32 | 18.7 | 18.6 | 0.99 | 18.6 | 0.99 | 18.6 | 0.99 |
| 64 | 8192 | 1024 | register_2x2_k128 | N=8192-K=1024:M_LEQ_64 | 25.4 | 19.3 | 0.76 | 18.8 | 0.74 | 19.3 | 0.76 |
| 256 | 8192 | 1024 | lds_128x128_w8_d1_k128 | N=8192-K=1024:M_LEQ_256 | 46.8 | 42.5 | 0.91 | 42.1 | 0.90 | 48.4 | 1.04 |
| 1024 | 8192 | 1024 | lds_128x128_w8_d1_k128 | N=8192-K=1024:M_LEQ_1024 | 157.3 | 132.6 | 0.84 | 130.3 | 0.83 | 171.5 | 1.09 |
| 2048 | 8192 | 1024 | lds_128x128_w8_d1_k128 | N=8192-K=1024:M_LEQ_2048 | 391.0 | 267.8 | 0.68 | 254.6 | 0.65 | 333.7 | 0.85 |
| 16 | 24576 | 1536 | register_1x2_k128 | N=24576-K=1536:M_LEQ_16 | 75.0 | 31.5 | 0.42 | 30.7 | 0.41 | 31.3 | 0.42 |
| 32 | 24576 | 1536 | register_2x2_k128 | N=24576-K=1536:M_LEQ_32 | 76.4 | 33.0 | 0.43 | 32.0 | 0.42 | 33.1 | 0.43 |
| 64 | 24576 | 1536 | register_2x2_k128 | N=24576-K=1536:M_LEQ_64 | 79.5 | 56.0 | 0.70 | 54.1 | 0.68 | 56.0 | 0.70 |
| 256 | 24576 | 1536 | lds_128x128_w8_d1_k128 | N=24576-K=1536:M_LEQ_256 | 154.1 | 141.0 | 0.91 | 141.2 | 0.92 | 203.0 | 1.32 |
| 1024 | 24576 | 1536 | lds_128x128_w8_d1_k128 | N=24576-K=1536:any | 514.9 | 552.9 | 1.07 | 551.5 | 1.07 | 814.6 | 1.58 |
| 2048 | 24576 | 1536 | lds_128x128_w8_d1_k128 | N=24576-K=1536:any | 1021.4 | 1102.6 | 1.08 | 1096.1 | 1.07 | 1654.8 | 1.62 |

## The Schedule rule (`rule.json`)

Production (`prod`), the register panel forced (`32x32`), and both LDS tilings
forced, `[N, K]` f32. At every whole-M point (30) the production route is
within 1.5% of the fastest Tessera arm. Where the rule selects the LDS body (29
points, 27 of them whole-M) it runs at 0.45-0.97x of the register panel. The two ragged rows were timed under the
earlier rule (128x128 at 1000x4096x2048 and 200x8192x1024) -- the evidence for
changing it is `ragged.json` below.

| N | K | M | production route | production us | register 32x32 us | LDS 128x128 us | LDS 128x64 us | AITER us | prod/register | prod/best Tessera | prod/AITER |
|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1024 | 4096 | 128 | register_1x2_k128 | 20.3 | 22.6 | 43.7 | 30.6 | 28.0 | 0.90 | 1.00 | 0.72 |
| 1024 | 4096 | 256 | register_2x2_k128 | 30.7 | 30.8 | 43.9 | 32.8 | 30.9 | 1.00 | 1.00 | 1.00 |
| 1024 | 4096 | 512 | lds_128x64_w8_d1_k128 | 45.1 | 50.3 | 52.1 | 45.3 | 40.3 | 0.90 | 1.00 | 1.12 |
| 1024 | 4096 | 1024 | lds_128x128_w8_d1_k128 | 66.9 | 101.9 | 66.9 | 74.4 | 79.6 | 0.66 | 1.00 | 0.84 |
| 1024 | 4096 | 2048 | lds_128x128_w8_d1_k128 | 131.4 | 199.5 | 130.2 | 144.4 | 126.4 | 0.66 | 1.01 | 1.04 |
| 2048 | 2048 | 128 | register_2x2_k128 | 15.2 | 15.0 | 22.2 | 16.1 | 16.2 | 1.01 | 1.01 | 0.94 |
| 2048 | 2048 | 256 | lds_128x64_w8_d1_k128 | 23.7 | 25.9 | 25.2 | 23.6 | 23.7 | 0.91 | 1.01 | 1.00 |
| 2048 | 2048 | 512 | lds_128x128_w8_d1_k128 | 36.2 | 42.7 | 36.3 | 40.7 | 40.4 | 0.85 | 1.00 | 0.89 |
| 2048 | 2048 | 1024 | lds_128x128_w8_d1_k128 | 71.9 | 77.4 | 71.8 | 75.0 | 77.0 | 0.93 | 1.00 | 0.93 |
| 2048 | 2048 | 2048 | lds_128x128_w8_d1_k128 | 123.2 | 150.0 | 122.9 | 147.5 | 143.3 | 0.82 | 1.00 | 0.86 |
| 4096 | 1024 | 128 | lds_128x64_w8_d1_k128 | 15.5 | 17.9 | 18.0 | 15.5 | 19.1 | 0.87 | 1.00 | 0.81 |
| 4096 | 1024 | 256 | lds_128x128_w8_d1_k128 | 22.4 | 24.8 | 22.6 | 23.7 | 24.8 | 0.90 | 1.00 | 0.90 |
| 4096 | 1024 | 512 | lds_128x128_w8_d1_k128 | 41.2 | 44.1 | 41.3 | 40.9 | 44.3 | 0.93 | 1.01 | 0.93 |
| 4096 | 1024 | 1024 | lds_128x128_w8_d1_k128 | 67.4 | 84.8 | 67.4 | 76.8 | 85.2 | 0.80 | 1.00 | 0.79 |
| 4096 | 1024 | 2048 | lds_128x128_w8_d1_k128 | 129.2 | 160.0 | 129.2 | 150.6 | 151.4 | 0.81 | 1.00 | 0.85 |
| 4096 | 2048 | 1000 | lds_128x128_w8_d1_k128 | 150.7 | 293.2 | 151.5 | 144.1 | 144.0 | 0.51 | 1.05 | 1.05 |
| 4096 | 7168 | 128 | lds_128x64_w8_d1_k128 | 72.0 | 74.5 | 76.7 | 71.0 | 75.8 | 0.97 | 1.02 | 0.95 |
| 4096 | 7168 | 256 | lds_128x128_w8_d1_k128 | 109.6 | 163.4 | 110.0 | 129.9 | 136.2 | 0.67 | 1.00 | 0.80 |
| 4096 | 7168 | 512 | lds_128x128_w8_d1_k128 | 225.4 | 426.7 | 223.7 | 238.8 | 255.4 | 0.53 | 1.01 | 0.88 |
| 4096 | 7168 | 1024 | lds_128x128_w8_d1_k128 | 401.3 | 885.4 | 395.4 | 487.6 | 465.1 | 0.45 | 1.01 | 0.86 |
| 4096 | 7168 | 2048 | lds_128x128_w8_d1_k128 | 803.6 | 1438.4 | 801.7 | 991.0 | 939.6 | 0.56 | 1.00 | 0.86 |
| 6144 | 2048 | 128 | lds_128x64_w8_d1_k128 | 33.1 | 36.2 | 34.7 | 33.0 | 37.3 | 0.91 | 1.00 | 0.89 |
| 6144 | 2048 | 256 | lds_128x128_w8_d1_k128 | 52.8 | 66.8 | 52.9 | 59.2 | 62.4 | 0.79 | 1.00 | 0.85 |
| 6144 | 2048 | 512 | lds_128x128_w8_d1_k128 | 95.9 | 126.7 | 95.6 | 113.3 | 116.3 | 0.76 | 1.00 | 0.82 |
| 6144 | 2048 | 1024 | lds_128x128_w8_d1_k128 | 184.4 | 247.7 | 185.2 | 221.0 | 195.4 | 0.74 | 1.00 | 0.94 |
| 6144 | 2048 | 2048 | lds_128x128_w8_d1_k128 | 362.2 | 493.7 | 368.7 | 439.1 | 326.7 | 0.73 | 1.00 | 1.11 |
| 8192 | 1024 | 200 | lds_128x128_w8_d1_k128 | 49.7 | 110.2 | 49.6 | 41.7 | 44.0 | 0.45 | 1.19 | 1.13 |
| 24576 | 1536 | 128 | lds_128x128_w8_d1_k128 | 75.2 | 104.6 | 74.6 | 87.0 | 103.5 | 0.72 | 1.01 | 0.73 |
| 24576 | 1536 | 256 | lds_128x128_w8_d1_k128 | 141.4 | 202.1 | 141.1 | 169.0 | 153.7 | 0.70 | 1.00 | 0.92 |
| 24576 | 1536 | 512 | lds_128x128_w8_d1_k128 | 280.6 | 408.0 | 281.1 | 337.7 | 253.7 | 0.69 | 1.00 | 1.11 |
| 24576 | 1536 | 1024 | lds_128x128_w8_d1_k128 | 554.7 | 813.8 | 555.4 | 667.3 | 506.5 | 0.68 | 1.00 | 1.10 |
| 24576 | 1536 | 2048 | lds_128x128_w8_d1_k128 | 1101.0 | 1652.0 | 1100.4 | 1323.7 | 1011.8 | 0.67 | 1.00 | 1.09 |

## Ragged M (`ragged.json`)

Every ragged M takes 128x64 under the final rule. 128x64 is 0.71-0.97x of
128x128 at 19 of 20 points; the exception, 200x4096x7168, is 1.03x here
(1.10x in the exploration run). The cause is registers, not arithmetic: the
128x128 body's masked edge takes 251 VGPRs against 238 whole (`isa.json`),
one wave per SIMD fewer. Clamping the out-of-range copy rows instead of
zero-filling them is neutral (clamp / zero-fill geomean 0.991, 0.94-1.01).

Ragged M is also where AITER keeps its largest lead: production / AITER
geomean 1.075 over the 20 points, and 1.37-1.56x at N = 24576, K = 1536 with
M in 300..1500, where a 64-row AITER tile wastes fewer rows than our 128-row
one and our masked edge costs occupancy. Open.

| M | N | K | production route | production us | LDS 128x128 us | LDS 128x64 us | 128x64/128x128 | clamp/zero-fill (128x128, 128x64) | AITER us | production/AITER |
|---:|---:|---:|---|---:|---:|---:|---:|---|---:|---:|
| 200 | 8192 | 1024 | lds_128x64_w8_d1_k128 | 41.7 | 50.8 | 42.0 | 0.83 | 0.985, 1.001 | 43.8 | 0.95 |
| 300 | 8192 | 1024 | lds_128x64_w8_d1_k128 | 59.8 | 71.5 | 59.1 | 0.83 | 0.992, 0.995 | 56.4 | 1.06 |
| 600 | 8192 | 1024 | lds_128x64_w8_d1_k128 | 100.4 | 116.7 | 98.6 | 0.85 | 0.995, 1.002 | 94.7 | 1.06 |
| 1000 | 8192 | 1024 | lds_128x64_w8_d1_k128 | 154.9 | 181.6 | 154.5 | 0.85 | 0.996, 0.997 | 149.2 | 1.04 |
| 1500 | 8192 | 1024 | lds_128x64_w8_d1_k128 | 236.7 | 267.1 | 228.7 | 0.86 | 0.997, 0.996 | 291.3 | 0.81 |
| 200 | 4096 | 7168 | lds_128x64_w8_d1_k128 | 131.5 | 126.5 | 129.7 | 1.03 | 0.998, 0.939 | 134.2 | 0.98 |
| 300 | 4096 | 7168 | lds_128x64_w8_d1_k128 | 179.6 | 231.1 | 176.3 | 0.76 | 0.972, 1.003 | 178.8 | 1.00 |
| 600 | 4096 | 7168 | lds_128x64_w8_d1_k128 | 307.7 | 367.8 | 293.9 | 0.80 | 0.997, 0.994 | 289.5 | 1.06 |
| 1000 | 4096 | 7168 | lds_128x64_w8_d1_k128 | 498.4 | 508.6 | 483.1 | 0.95 | 0.991, 0.991 | 463.3 | 1.08 |
| 1500 | 4096 | 7168 | lds_128x64_w8_d1_k128 | 745.1 | 760.9 | 728.2 | 0.96 | 0.983, 0.991 | 696.4 | 1.07 |
| 200 | 24576 | 1536 | lds_128x64_w8_d1_k128 | 166.7 | 178.6 | 164.6 | 0.92 | 0.987, 0.984 | 147.9 | 1.13 |
| 300 | 24576 | 1536 | lds_128x64_w8_d1_k128 | 250.1 | 268.5 | 244.6 | 0.91 | 0.993, 1.001 | 160.2 | 1.56 |
| 600 | 24576 | 1536 | lds_128x64_w8_d1_k128 | 431.8 | 453.5 | 416.7 | 0.92 | 0.986, 0.993 | 313.5 | 1.38 |
| 1000 | 24576 | 1536 | lds_128x64_w8_d1_k128 | 696.0 | 716.4 | 674.5 | 0.94 | 0.986, 0.987 | 504.8 | 1.38 |
| 1500 | 24576 | 1536 | lds_128x64_w8_d1_k128 | 1041.5 | 1059.4 | 1023.1 | 0.97 | 0.989, 0.989 | 759.1 | 1.37 |
| 200 | 2048 | 2048 | lds_128x64_w8_d1_k128 | 22.8 | 30.7 | 22.5 | 0.73 | 1.007, 0.992 | 24.6 | 0.93 |
| 300 | 2048 | 2048 | lds_128x64_w8_d1_k128 | 31.8 | 37.7 | 31.7 | 0.84 | 0.983, 1.004 | 35.0 | 0.91 |
| 600 | 2048 | 2048 | lds_128x64_w8_d1_k128 | 49.2 | 68.6 | 48.6 | 0.71 | 0.993, 0.988 | 51.4 | 0.96 |
| 1000 | 2048 | 2048 | lds_128x64_w8_d1_k128 | 76.8 | 82.7 | 76.8 | 0.93 | 0.982, 1.003 | 77.2 | 0.99 |
| 1500 | 2048 | 2048 | lds_128x64_w8_d1_k128 | 114.9 | 119.9 | 112.4 | 0.94 | 0.989, 0.991 | 107.9 | 1.07 |

## LDS-body performance keys (`knobs.json`)

Time / the selected 128x128 single-buffered body (columns: shapes).

| arm | 1024x4096x7168 | 2048x6144x2048 | 1024x24576x1536 | 512x4096x1024 |
|---|---:|---:|---:|---:|
| tessera_nk_lds128x128_w8_d1_s64_p-1_f-1 | 1.12 | 1.11 | 1.11 | 1.10 |
| tessera_nk_lds128x64_w8_d1_s-1_p-1_f-1 | 1.19 | 1.17 | 1.17 | 1.01 |
| tessera_nk_lds128x64_w8_d1_s-1_p-1_f1 | 1.12 | 1.11 | 1.12 | 1.00 |
| tessera_nk_lds128x128_w8_d1_s-1_p-1_f-1@prev | 1.01 | 0.99 | 1.00 | 0.98 |
| tessera_nk_lds64x128_w4_d1_s-1_p-1_f-1 | 1.24 | 1.23 | 1.26 | 1.02 |
| tessera_nk_lds128x128_w8_d1_s-1_p32_f-1 | 1.22 | 1.20 | 1.20 | 1.26 |
| tessera_nk_lds256x128_w16_d1_s-1_p-1_f-1 | 1.06 | 1.08 | 1.13 | 0.96 |
| tessera_nk_lds128x64_w8_d2_s-1_p-1_f-1 | 1.41 | 1.47 | 1.48 | 1.22 |
| tessera_nk | 1.01 | 1.00 | 1.00 | 1.00 |
| aiter | 1.16 | 0.93 | 0.93 | 1.08 |
| tessera_nk_lds128x128_w8_d1_s-1_p-1_f-1 | 1.00 | 1.00 | 1.00 | 1.00 |
| tessera_nk_lds128x128_w8_d1_s-1_p0_f-1 | 4.48 | 4.34 | 4.16 | 3.43 |
| tessera_nk_lds128x128_w8_d1_s-1_p-1_f-1_gm8 | 0.99 | 0.95 | 0.98 | 0.96 |
| tessera_nk_lds64x64_w4_d1_s-1_p-1_f-1 | 1.15 | 1.13 | 1.12 | 0.98 |
| tessera_nk_lds128x128_w8_d1_s-1_p-1_f1 | 1.19 | 1.24 | 1.23 | 1.03 |

Measured negative, not selected: double-buffered LDS (128x64: 1.22-1.48x; the
128x128 form needs 72 KiB and is refused), the register-staged next slab
(1.00-1.24x, and at 128x128 it spills), a 64-byte slab (1.10-1.12x), row padding
0 (3.4-4.5x, bank conflicts) and 32 (1.20-1.26x), 64x64/4 waves, 64x128/4
waves, 256x128/16 waves (0.96-1.26x; faster only at 512x4096x1024, by at most
4%). Neutral, not selected: grouped raster (`gm8`: 0.95-0.99x here,
0.98-1.01x in an earlier exploration -- not a converging win) and the first-cut compiler's full-fence barriers with the
slab loads after the barrier (`@prev`: 0.98-1.01x; the final body fences LDS
only and issues the loads first).

## ISA (`isa.json`, static counts)

| kernel | VGPR | LDS bytes | `v_wmma_..._fp8_fp8` | `global_load_b128` | `ds_store_b128` | `ds_load_2addr_b64` | barriers | `global_inv` |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| register 32x32 (1024x4096x1024) | 231 | 0 | 40 | 0 | 0 | 0 | 0 | 0 |
| LDS 128x128 (1024x4096x7168) | 238 | 36864 | 64 | 8 | 8 | 24 | 2 | 0 |
| LDS 128x128, bf16 store | 238 | 36864 | 64 | 8 | 8 | 24 | 2 | 0 |
| LDS 128x64 (128x4096x1024) | 187 | 27648 | 32 | 6 | 6 | 16 | 2 | 0 |
| LDS 128x64, ragged (1000x8192x1024) | 233 | 27648 | 32 | 6 | 6 | 16 | 2 | 0 |
| LDS 128x128, ragged (1000x8192x1024) | 251 | 36864 | 64 | 8 | 8 | 24 | 2 | 0 |

No kernel spills or uses scratch. The register panel's 40 static WMMAs include
its edge path. With the default `gpu.barrier` the body carried a
`global_inv` after every barrier (an L0 invalidate for the workgroup-scope
acquire); the LDS-only memfence removes it.

## Tests (`device_tests_gfx1201.txt`, `lit_gfx1201.txt`)

- W8A8 device + host (`tests/device/rocm/test_fp8_blockscale_w8a8.py`,
  `tests/unit/test_rocm_fp8_blockscale.py`,
  `tests/unit/test_rocm_pipeline_cache_key.py`): **147 passed on both trees**.
  New device rows: Schedule-selected LDS shapes (128x128 and 128x64, ragged M
  and N, f32 and bf16) against the fp64 oracle and **bit for bit against the
  register panel**; every LDS performance key (register prefetch, double
  buffer, 64-byte slab, padding 0/32, 4/16 waves) computing the same bits;
  the bf16 store equal to the fp32 result rounded to nearest-even (register
  and LDS bodies); over-budget LDS and the `[K, N]` weight refused by name.
- MXFP4 device files + folded host tests: 145 passed on both trees.
- `lit tests/tessera-ir`: 455 passed / 66 unsupported on both trees (new
  fixture `phase2/e2e_fp8_blockscale_lds_rocm_target.mlir`).
  `check-tessera-rocm`: 82/82 on both trees.
- A first assertions-tree run reported 5 W8A8 failures and 1 lit failure
  against a `tessera-opt` two commits stale (the generator's staleness warning
  fired); after rebuilding that tree the recorded run above is green.

## Not claimed

- **Not a model-level result.** Kernel device-clock time on one WSL2 host.
- **The `[K, N]` weight is unchanged** (register panel only; the LDS body
  refuses it by name).
- **No profiler attribution** (no `/dev/kfd`); the register-pressure reading of
  ragged M is from static VGPR counts.
- **No gfx1151, NVIDIA or Apple evidence.** The contract is gfx1201-only, and
  the rule's 64-CU constant is this part's.

## Reproduce

```bash
# Tessera env: ~/programming/w-perf-env.sh (TESSERA_OPT at the worktree's
# build/, TESSERA_ROCM_CHIP=gfx1201, TESSERA_GFX1201_DEVICE_PROOF=1, matched
# LLVM 23). AITER side: a venv with triton==3.8.0, numpy, ml_dtypes (no torch).
flock /tmp/tessera-timing.lock ~/wbs-aiter-venv/bin/python \
  benchmarks/rocm/benchmark_gfx1201_fp8_blockscale.py \
  --compiler build/tools/tessera-opt/tessera-opt --shape M,N,K ... \
  --with-aiter --no-production --bf16 --sweep prod:nk@prev \
  --alt-compiler prev=<previous tessera-opt> --windows 10 --output /tmp/compare.json
# rule / ragged / knobs: --sweep prod:nk, 32x32:1:-1:nk (register panel),
# lds:MMxMN:W:D:S:P:F:nk[:gmG][@alias] (LDS body keys), --windows 5.
python benchmarks/rocm/check_gfx1201_fp8_blockscale_identity.py \
  --packet <packet>.json --arm tessera_nk --arm tessera_nk+bf16 --output /tmp/id.json
python benchmarks/rocm/inspect_gfx1201_fp8_blockscale_isa.py --variant M,N,K:prod ... \
  --output /tmp/isa.json
```
