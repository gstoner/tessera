# ROCM-FP8-BLOCKSCALE-1 — gfx1201 block-scaled FP8 W8A8 vs AITER, 2026-09-27

Sync `GFX1201-LANES-2026-09-27`. Host **Tajasarus** (Radeon RX 9070 XT,
gfx1201, Ubuntu 26.04 under WSL2, kernel `6.18.33.2-microsoft-standard-WSL2`,
ROCm 10.0 / HIP 7.15). Worktree `~/programming/tessera-w-blockscale` on branch
`claude/gfx1201-lanes-blockscale`. Two trees built from the same source:
`build/` (matched LLVM/MLIR 23.1.1, NDEBUG) and `build-assertions/`
(assertions-ON LLVM/MLIR 23.1.1, `-fno-rtti -UNDEBUG`).

| File | Source commit | What |
|---|---|---|
| `compare.json` | `38a095cb` (clean) | Tessera vs AITER, 54 (M, N, K) rows |
| `sweep.json` | `38a095cb` (clean) | Tessera-only panel / group-step / layout sweep behind the Schedule rule |
| `device_tests_gfx1201.txt` | `ab2f1a7f` (clean) | device + host-free tests, both trees |
| `lit_gfx1201.txt` | `ab2f1a7f` (clean) | `lit tests/tessera-ir` and `check-tessera-rocm`, both trees |

`ab2f1a7f..38a095cb` changes only `benchmarks/rocm/benchmark_gfx1201_fp8_blockscale.py`
(window warm-up and re-sizing), so the compiler that produced the tested
kernels is the one that was timed. The commit that adds this packet also
rewrites two C++ comments (the panel rule and the `scale-group-panels`
default) to cite these numbers; it changes no code.

## What was measured

- **Tessera** -- the production route: `tessera.scaled_matmul` (e4m3 A/B, fp32
  scales, `scale_layout.block = [128, 128]`) compiled Graph -> Schedule -> Tile
  -> Target -> HSACO by `tessera.compiler.rocm_fp8_blockscale`, register panel
  and loop from the Schedule's rule, generator defaults. Both named weight
  layouts: `tessera_nk` (weight [N, K], `transposeB`) and `tessera_kn` (B [K, N]).
  f32 output.
- **AITER** -- `_gemm_a8w8_blockscale_kernel` from the local checkout
  `~/programming/aiter` at `f0966b0a` (kernel source sha256 `f4824524…`),
  compiled **unmodified** ahead of time by Triton 3.8.0 for gfx1201 with that
  checkout's tuned gfx1201 JSON (`configs/gfx1201/triton/gemm/gemm_a8w8_blockscale/`,
  the (N, K) file and M bucket, else `DEFAULT.json`). The harness reproduces
  only the host wrapper's grid and Triton's argument specialization; torch is
  not installed on the box, so the framework-dependent config helper is not
  imported. Weight [N, K] (`b = w.T`), bf16 output -- AITER's production form.
  Split-K buckets (`NUM_KSPLIT > 1`) need AITER's reduce kernel, which the
  harness does not drive: those 3 rows are **not measured**, not substituted.
- Every kernel is checked against an fp64 oracle of the block-scaled math
  before timing: max relative error (to the magnitude of the summed terms)
  1.3e-7 for Tessera (f32 store), 1.6e-3 for AITER (bf16 store).
- **Timing source: the compiler-built device-clock marker**
  (`--tessera-device-clock-span`, `llvm.readsteadycounter` at
  `hipDeviceAttributeWallClockRate` = 100000 kHz), with HIP events and host wall
  as per-window cross-checks. All arms are warmed together, then 10 windows
  per arm are run **paired and interleaved** (ABC, CBA, ...). Every window is
  >= 6.19 ms (floor 5 ms); device clock vs HIP event agree within 1.9% in every
  window; every arm is `admissible` on the first attempt. Reported value:
  median per-launch device-clock time.

## Result: we win small M, lose large M

Tessera `[N, K]` time / AITER time (< 1 = Tessera faster), 51 measured rows:

| M bucket | rows | geomean | range | Tessera faster |
|---|---:|---:|---|---:|
| M <= 64 | 24 | **0.65** | 0.36 - 1.13 | 22 / 24 |
| M = 256 | 9 | **1.09** | 0.96 - 1.31 | 2 / 9 |
| M >= 1024 | 18 | **1.31** | 0.86 - 1.94 | 2 / 18 |

The `[K, N]` layout loses almost everywhere (geomeans 1.14 / 2.16 / 2.67): its
B fragment is a strided per-element gather, and nothing on this chip's typed
route widens it for 8-bit storage (`TR_B64` is a different permutation,
wmma-fragment-layout.md). A W8A8 checkpoint ships `[N, K]`; `[K, N]` is kept
because it is the Graph op's untransposed spelling, not because it is fast.

Where AITER wins big (M >= 1024 at K >= 4096, up to 1.94x) its tuned configs
use 64x128 / 128x64 **LDS-staged** multi-warp tiles; Tessera's W8A8 body is the
one-wave register panel, capped at 32x32 (231 VGPRs) by the doubled
accumulator live set (below). Where Tessera wins big (M <= 64, up to 2.8x at 16x4096x7168), AITER's
small-M buckets are 16-32 row tiles with K-loop-bound tuning that leave the
GPU under-occupied. Neither side was tuned for this comparison.

### Rows

| M | N | K | AITER config | AITER us | Tessera [N,K] us | nk/AITER | Tessera [K,N] us | kn/AITER |
|---:|---:|---:|---|---:|---:|---:|---:|---:|
| 16 | 4096 | 1024 | N=4096-K=1024:M_LEQ_16 | not measured (split-K bucket) | 12.9 | – | 17.5 | – |
| 32 | 4096 | 1024 | N=4096-K=1024:M_LEQ_32 | 12.3 | 13.8 | 1.13 | 18.1 | 1.48 |
| 64 | 4096 | 1024 | N=4096-K=1024:M_LEQ_1024 | 16.7 | 13.0 | 0.78 | 17.6 | 1.05 |
| 256 | 4096 | 1024 | N=4096-K=1024:M_LEQ_1024 | 25.0 | 25.1 | 1.01 | 46.8 | 1.87 |
| 1024 | 4096 | 1024 | N=4096-K=1024:M_LEQ_1024 | 83.9 | 83.4 | 0.99 | 177.7 | 2.12 |
| 2048 | 4096 | 1024 | N=4096-K=1024:M_LEQ_2048 | 153.5 | 158.9 | 1.04 | 331.5 | 2.16 |
| 16 | 2048 | 2048 | N=2048-K=2048:M_LEQ_16 | 13.5 | 9.2 | 0.68 | 24.1 | 1.79 |
| 32 | 2048 | 2048 | N=2048-K=2048:M_LEQ_32 | 13.6 | 9.3 | 0.68 | 24.4 | 1.80 |
| 64 | 2048 | 2048 | N=2048-K=2048:M_LEQ_128 | 13.6 | 10.1 | 0.74 | 25.7 | 1.88 |
| 256 | 2048 | 2048 | N=2048-K=2048:M_LEQ_256 | 24.5 | 26.4 | 1.08 | 51.0 | 2.09 |
| 1024 | 2048 | 2048 | N=2048-K=2048:M_LEQ_1024 | 78.5 | 79.9 | 1.02 | 175.2 | 2.23 |
| 2048 | 2048 | 2048 | N=2048-K=2048:M_LEQ_2048 | 146.4 | 150.4 | 1.03 | 340.7 | 2.33 |
| 16 | 6144 | 2048 | N=6144-K=2048:M_LEQ_16 | 28.6 | 14.9 | 0.52 | 23.4 | 0.82 |
| 32 | 6144 | 2048 | N=6144-K=2048:M_LEQ_32 | 29.9 | 16.5 | 0.55 | 25.7 | 0.86 |
| 64 | 6144 | 2048 | N=6144-K=2048:M_LEQ_128 | 35.4 | 22.5 | 0.63 | 32.1 | 0.91 |
| 256 | 6144 | 2048 | N=6144-K=2048:M_LEQ_256 | 64.5 | 67.7 | 1.05 | 137.5 | 2.13 |
| 1024 | 6144 | 2048 | N=6144-K=2048:M_LEQ_1024 | 199.7 | 248.1 | 1.24 | 511.9 | 2.56 |
| 2048 | 6144 | 2048 | N=6144-K=2048:M_LEQ_2048 | 337.6 | 492.7 | 1.46 | 1031.3 | 3.05 |
| 16 | 3072 | 1536 | N=3072-K=1536:M_LEQ_16 | 11.7 | 7.7 | 0.66 | 17.9 | 1.53 |
| 32 | 3072 | 1536 | N=3072-K=1536:M_LEQ_32 | 12.2 | 8.0 | 0.65 | 18.5 | 1.51 |
| 64 | 3072 | 1536 | N=3072-K=1536:M_LEQ_64 | 12.4 | 10.9 | 0.88 | 19.9 | 1.61 |
| 256 | 3072 | 1536 | N=3072-K=1536:M_LEQ_256 | 29.2 | 28.0 | 0.96 | 57.6 | 1.97 |
| 1024 | 3072 | 1536 | N=3072-K=1536:any | 71.4 | 93.5 | 1.31 | 210.3 | 2.95 |
| 2048 | 3072 | 1536 | N=3072-K=1536:any | 132.7 | 188.6 | 1.42 | 413.6 | 3.12 |
| 16 | 1024 | 4096 | N=1024-K=4096:M_LEQ_16 | not measured (split-K bucket) | 14.3 | – | 47.2 | – |
| 32 | 1024 | 4096 | N=1024-K=4096:M_LEQ_32 | not measured (split-K bucket) | 17.5 | – | 52.7 | – |
| 64 | 1024 | 4096 | N=1024-K=4096:M_LEQ_1024 | 27.2 | 17.5 | 0.64 | 52.9 | 1.94 |
| 256 | 1024 | 4096 | N=1024-K=4096:M_LEQ_1024 | 30.7 | 30.2 | 0.98 | 58.9 | 1.92 |
| 1024 | 1024 | 4096 | N=1024-K=4096:M_LEQ_1024 | 75.3 | 101.1 | 1.34 | 218.3 | 2.90 |
| 2048 | 1024 | 4096 | N=1024-K=4096:M_LEQ_2048 | 124.4 | 198.2 | 1.59 | 390.9 | 3.14 |
| 16 | 4096 | 7168 | N=4096-K=7168:M_LEQ_16 | 67.1 | 24.3 | 0.36 | 76.7 | 1.14 |
| 32 | 4096 | 7168 | N=4096-K=7168:M_LEQ_32 | 67.0 | 26.4 | 0.39 | 80.5 | 1.20 |
| 64 | 4096 | 7168 | N=4096-K=7168:M_LEQ_64 | 72.1 | 36.6 | 0.51 | 89.3 | 1.24 |
| 256 | 4096 | 7168 | N=4096-K=7168:M_LEQ_256 | 136.1 | 163.0 | 1.20 | 379.1 | 2.79 |
| 1024 | 4096 | 7168 | N=4096-K=7168:any | 464.6 | 886.3 | 1.91 | 1884.0 | 4.06 |
| 2048 | 4096 | 7168 | N=4096-K=7168:any | 931.8 | 1803.3 | 1.94 | 3132.1 | 3.36 |
| 16 | 7168 | 2048 | N=7168-K=2048:M_LEQ_16 | 32.9 | 17.7 | 0.54 | 23.8 | 0.72 |
| 32 | 7168 | 2048 | N=7168-K=2048:M_LEQ_32 | 31.2 | 20.4 | 0.65 | 31.2 | 1.00 |
| 64 | 7168 | 2048 | N=7168-K=2048:M_LEQ_64 | 32.1 | 26.3 | 0.82 | 38.0 | 1.18 |
| 256 | 7168 | 2048 | N=7168-K=2048:M_LEQ_256 | 73.3 | 85.5 | 1.17 | 169.1 | 2.31 |
| 1024 | 7168 | 2048 | N=7168-K=2048:any | 252.4 | 351.0 | 1.39 | 598.1 | 2.37 |
| 2048 | 7168 | 2048 | N=7168-K=2048:any | 498.6 | 699.2 | 1.40 | 1208.0 | 2.42 |
| 16 | 8192 | 1024 | N=8192-K=1024:M_LEQ_16 | 20.3 | 21.5 | 1.06 | 20.0 | 0.99 |
| 32 | 8192 | 1024 | N=8192-K=1024:M_LEQ_32 | 18.9 | 18.6 | 0.98 | 18.5 | 0.98 |
| 64 | 8192 | 1024 | N=8192-K=1024:M_LEQ_64 | 25.6 | 19.3 | 0.76 | 21.9 | 0.86 |
| 256 | 8192 | 1024 | N=8192-K=1024:M_LEQ_256 | 47.1 | 49.9 | 1.06 | 89.7 | 1.91 |
| 1024 | 8192 | 1024 | N=8192-K=1024:M_LEQ_1024 | 157.8 | 170.7 | 1.08 | 351.0 | 2.22 |
| 2048 | 8192 | 1024 | N=8192-K=1024:M_LEQ_2048 | 393.1 | 336.6 | 0.86 | 687.4 | 1.75 |
| 16 | 24576 | 1536 | N=24576-K=1536:M_LEQ_16 | 75.1 | 31.0 | 0.41 | 38.9 | 0.52 |
| 32 | 24576 | 1536 | N=24576-K=1536:M_LEQ_32 | 76.6 | 32.8 | 0.43 | 53.6 | 0.70 |
| 64 | 24576 | 1536 | N=24576-K=1536:M_LEQ_64 | 79.6 | 55.8 | 0.70 | 101.5 | 1.27 |
| 256 | 24576 | 1536 | N=24576-K=1536:M_LEQ_256 | 155.4 | 203.8 | 1.31 | 405.7 | 2.61 |
| 1024 | 24576 | 1536 | N=24576-K=1536:any | 516.7 | 817.0 | 1.58 | 1649.2 | 3.19 |
| 2048 | 24576 | 1536 | N=24576-K=1536:any | 1023.8 | 1658.6 | 1.62 | 3301.8 | 3.22 |

## Not claimed

- **Not a matched-output comparison.** Tessera stores f32 (its contract's
  plain accumulator store); AITER stores bf16. At M >= 1024 Tessera writes 2x
  AITER's output bytes. That is in AITER's favour, and it is stated rather
  than corrected away.
- **The `[K, N]` rows are not a claim about AITER on `[K, N]`.** AITER was run
  only in its own production layout.
- **Not a model-level result.** The reported 25% / 63% decode gains for AITER
  on an R9700 are end-to-end numbers; nothing here reproduces or contradicts
  them.
- **No profiler attribution.** Tajasarus has no `/dev/kfd`; the explanations
  above for where each side wins are readings of the two kernels' structure,
  not counter measurements.
- **No gfx1151, NVIDIA or Apple evidence.** The contract is gfx1201-only.

## The Schedule's panel rule (`sweep.json`)

Tessera-only, same protocol, 5 windows per arm, `[N, K]` unless named:

- 32x32 wins or comes within 2% at every fully tiled shape with >= 256 tiles
  at that panel except 4096^3 (64x6144x2048: 24.6 us; 256x4096x1024 25.3 vs
  25.0 at 32x64; 1024x4096x1024 87.0; 2048^3 152.0). 64-wide panels lose -- 64x64 by 2-20x -- because the isolated
  group partial doubles the live accumulators. From the HSACO notes at
  1024x4096x1024 (`[N, K]`, default group step): 16x32 189 VGPRs, 32x32 231,
  both unspilled; 32x64 256 + 10 spilled, 64x32 256 + 54, 64x64 256 + 337.
- Below 256 tiles the half-height 16x32 doubles the grid (32x4096x1024 13.7
  vs 15.6; 16x4096x1024 10.7 vs 73.1); at 64x2048x2048 16x64 was 6% ahead of
  the rule's 16x32 (15.1 vs 16.0).
- A ragged M under a 32-row panel sends a whole tile row down the masked edge
  path: M=48 146.1 us at 32x32 vs 25.9 at 16x32, M=100 158.0 vs 37.5.
- `scale-group-panels` in {0, 1, 2, 4} spreads within 4% at every M that is a
  whole 32 (no value wins consistently); 2 is kept. On ragged-M shapes the
  32x32 panel's edge path is much faster at 1 (16x4096x1024: 40.8 vs 73.1 us),
  but the rule never picks 32 rows there. `k_unroll=2` (two whole groups per iteration) on
  the 32x32 panel loses 1.1-1.7x at every M that is a whole 32 (it is only
  competitive on ragged-M shapes, where the rule does not pick 32 rows).
- **Open, not taken:** 64x32 is 13% faster than 32x32 at 4096^3 (1402 vs 1618
  us) and 6% slower at 2048^3. One shape is not a rule.

The 16x16 arm had windows under the floor in the earlier `ab2f1a7f` sweep; in
this packet every arm of both files is admissible.

## Tests (`device_tests_gfx1201.txt`, `lit_gfx1201.txt`)

- `tests/device/rocm/test_fp8_blockscale_w8a8.py` + `tests/unit/test_rocm_fp8_blockscale.py`:
  **52 passed** with `build/`; the device file again with the
  assertions-ON `tessera-opt`: **23 passed**. Every device row compiles the
  Graph op, asserts `v_wmma_f32_16x16x16_fp8_fp8` in the disassembly, counts
  the generator's per-group zeros / joins / MMAs and the `[N, K]` column-major
  views, launches through `runtime.launch` (`native_gpu`) and compares with the
  fp64 oracle -- bit-equal on the exact inputs. A separate row checks the
  result is far from a single-rescale kernel's.
- `lit tests/tessera-ir`: 453 passed / 66 unsupported, identical on both trees.
  `check-tessera-rocm`: 82/82 on both trees.

## Reproduce

```bash
# Tessera env: ~/programming/tessera venv + scripts/_rocm_env.sh; TESSERA_OPT
# and TESSERA_ROCM_OPT at the worktree's build/, TESSERA_ROCM_CHIP=gfx1201,
# TESSERA_GFX1201_DEVICE_PROOF=1, TESSERA_LLVM_BIN at the matched LLVM 23.
# AITER side: a python venv with triton==3.8.0, numpy, ml_dtypes (no torch).
flock /tmp/tessera-timing.lock ~/wbs-aiter-venv/bin/python \
  benchmarks/rocm/benchmark_gfx1201_fp8_blockscale.py \
  --compiler build/tools/tessera-opt/tessera-opt \
  --shape M,N,K ...  --windows 10 --output /tmp/compare.json
# sweep: add --sweep PMxPN:U:G:L (panel, groups per iteration, panels per
# group step (-1 = default), layout kn|nk); no AITER arm.
```
