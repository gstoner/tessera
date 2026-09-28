# GFX1201 folded MXFP4 at one row block (M = 256): the packed-bytes hypothesis, tested

Owner `ROCM-MXFP4-W4A8-1`; sync `FOUNDATION-BATCH-2-2026-09-27` (the open item
of `GFX1201-PERF-2026-09-27`: M = 256 still 1.05-1.23x behind Radiance,
unexplained; Radiance reads packed E2M1 weights, half the bytes of our
expanded E4M3 ones). Host **Tajasarus** (RX 9070 XT, `gfx1201`, Ubuntu 26.04
under WSL2, ROCm 10.0 / HIP 7.15, LLVM/MLIR 23.1.1), worktree
`~/programming/tessera-w-gaps`, tree `build/` (NDEBUG) for timing. Pinned
Radiance `dfdfa383…` (binary `c9f91bc8…`), fragment-order weights,
`RADIANCE_MXFP4_WPERM=1`, `RADIANCE_MXFP4_DECODE_MAX_M=64` (Radiance runs its
folded BM256/TN2 kernel -- the same tile as Tessera's -- at every shape here).

| File | Clean source commit | What |
|---|---|---|
| `mxfp4_nscan.json` | `2c9b1726` | M = 256, K = 5120, N = 4096..24576; selected folded, V1, the two opt-in **packed-E2M1** candidates, Radiance; three rotating input copies per engine |
| `mxfp4_nscan_copies1.json` | `e2ec4529` | the same N scan with **one** input copy per engine (operands may stay in the last-level cache), no packed arms |
| `mxfp4_small.json` | `2c9b1726` | PR #872's one-row-block band (M = 128 / 256) with the packed candidates |

Every packet: `benchmarks/rocm/record_gfx1201_mxfp4_folded_load_schedule.py`
(`--shapes nscan|small`, `--packed`, `--copies`, all new here), three
independent processes with alternating engine order, seven trials, matched
lossless-fold inputs. Before timing, the exact K32 route matches an
independent sampled reference and **every engine's BF16 output -- the packed
candidates included -- is bitwise equal to exact K32**. Timing: the
compiler-built device-clock marker (`llvm.readsteadycounter`, 100 MHz),
witnessed by HIP events in the same window, agreeing within 2.8% in every
window. Windows are sized for 6 ms from a warm-up estimate; the shortest
recorded are 4.72 ms (`mxfp4_nscan.json`), 4.41 ms (`copies1`) and 4.95 ms
(`small`) -- a few windows under the 5 ms target, as in the previous packets.
No profiler counters (no `/dev/kfd`). The compiler under test is this
branch's; the folded MXFP4 route is untouched by it: the selected and V1
kernels of `mxfp4_small.json` have the same selected-symbol
instruction-stream SHA-256 as PR #872's `mxfp4_small_final.json` at all four
shapes (`fbf530ea…` whole row block, `d7df4355…` partial, `113a5371…` V1;
the HSACO bytes themselves are not deterministic across hipcc runs).

## Why an N scan, and why copies

The packed-bytes hypothesis predicts a gap that depends on where the weight
comes from. The expanded E4M3 weight is N x K bytes, the packed one half of
that; the RX 9070 XT has a 64 MiB last-level cache. With three rotating copies
(the recorder's default) each launch reads operands the previous launch did
not; with one copy, a weight small enough stays cached across launches. If the
weight bytes were the gap, it should shrink when the weight is cache-resident
and grow with N once only the expanded weight spills.

## Results (per-process median device-clock ms; ratio to Radiance in parentheses)

Three copies (`mxfp4_nscan.json`):

| M×N×K | selected | V1 | packed permute | packed permute + A offset32 | Radiance |
|---|---:|---:|---:|---:|---:|
| 256×4096×5120 | 0.0722-0.0730 (1.03-1.07) | 0.0876-0.0883 (1.26-1.28) | 0.0924-0.0929 (1.33-1.35) | 0.0913-0.0930 (1.31-1.35) | 0.0682-0.0700 |
| 256×8192×5120 | 0.1426-0.1438 (1.15-1.17) | 0.1671-0.1681 (1.35-1.36) | 0.1489-0.1491 (1.20-1.21) | 0.1517-0.1522 (1.22-1.24) | 0.1226-0.1243 |
| 256×12288×5120 | 0.2130-0.2139 (1.19-1.23) | 0.2475-0.2498 (1.40-1.42) | 0.2558-0.2568 (1.44-1.47) | 0.2552-0.2579 (1.44-1.48) | 0.1745-0.1783 |
| 256×16384×5120 | 0.2764-0.2786 (1.21-1.22) | 0.2974-0.2999 (1.30-1.31) | 0.3067-0.3102 (1.35-1.36) | 0.3078-0.3127 (1.36-1.37) | 0.2266-0.2290 |
| 256×17408×5120 | 0.3009-0.3045 (1.21-1.21) | 0.3188-0.3221 (1.28-1.29) | 0.3181-0.3200 (1.27-1.28) | 0.3184-0.3215 (1.27-1.29) | 0.2481-0.2525 |
| 256×20480×5120 | 0.3405-0.3420 (1.20-1.22) | 0.3570-0.3602 (1.27-1.28) | 0.3624-0.3677 (1.29-1.31) | 0.3645-0.3678 (1.29-1.31) | 0.2787-0.2844 |
| 256×24576×5120 | 0.4057-0.4082 (1.21-1.25) | 0.4245-0.4254 (1.27-1.30) | 0.4528-0.4543 (1.35-1.39) | 0.4513-0.4537 (1.35-1.39) | 0.3274-0.3354 |

One copy (`mxfp4_nscan_copies1.json`):

| M×N×K | selected | V1 | Radiance |
|---|---:|---:|---:|
| 256×4096×5120 | 0.0670-0.0678 (**0.98-1.01**) | 0.0800-0.0804 (1.18-1.20) | 0.0668-0.0681 |
| 256×8192×5120 | 0.1265-0.1289 (1.12-1.15) | 0.1351-0.1362 (1.20-1.22) | 0.1112-0.1131 |
| 256×12288×5120 | 0.1880-0.1905 (1.11-1.15) | 0.2196-0.2211 (1.31-1.33) | 0.1651-0.1692 |
| 256×16384×5120 | 0.2557-0.2572 (1.18-1.21) | 0.2815-0.2879 (1.30-1.33) | 0.2128-0.2168 |
| 256×17408×5120 | 0.2817-0.2910 (1.20-1.26) | 0.2956-0.3066 (1.27-1.32) | 0.2318-0.2342 |
| 256×20480×5120 | 0.3208-0.3333 (1.20-1.25) | 0.3357-0.3428 (1.24-1.28) | 0.2668-0.2696 |
| 256×24576×5120 | 0.3902-0.3981 (1.19-1.24) | 0.4039-0.4071 (1.24-1.26) | 0.3219-0.3285 |

The one-row-block band (`mxfp4_small.json`) reproduces PR #872: selected /
Radiance 0.74-0.77 at 128×5120×8704, 1.04-1.07 at 128×17408×5120, 1.06-1.08
at 256×5120×8704, 1.20-1.22 at 256×17408×5120; the packed candidates are
1.25-1.31x Radiance at all four.

## What it says

1. **Weight residency explains the small-N part of the gap and no more.** At
   N = 4096 (expanded weight 21 MB) one copy brings the selected schedule to
   parity (0.98-1.01x, from 1.03-1.07x). But at N = 8192-12288, where one copy
   keeps **both** weights cache-resident (expanded 42-63 MB), 1.11-1.15x
   remains.
2. **The gap is a per-output-column cost that does not care where the weight
   comes from.** A linear fit over N >= 8192 gives Tessera **16.0 ns per
   column** (three copies) / 16.7 (one copy) against Radiance's **12.7 / 12.8**
   -- a 1.26-1.31x marginal cost in both cache regimes. Reading half the weight
   bytes would change the DRAM share of that cost, and the DRAM share is what
   the copies experiment moves; it barely moves the slope.
3. **Packed E2M1 as we can currently decode it does not help.** Both opt-in
   packed candidates (the batched `v_perm_b32` decode, with and without the
   32-bit A offsets) are bitwise exact and read half the weight bytes, and are
   1.20-1.48x Radiance -- slower than the expanded selected schedule at every
   N. They also lack the selected schedule's load-schedule keys (register-staged
   next slab, grouped raster, vector epilogue), so this is not a matched test of
   "packed vs expanded on the same schedule"; against V1 (the same era of
   schedule) packed wins at N = 8192 (0.89x), ties at 17408 (1.00x) and loses at
   the other five N (1.02-1.07x).

**Verdict (measured):** the packed-bytes hypothesis is **not supported** as
the explanation of the M = 256 gap -- the gap is dominated by a per-column cost
that persists with the weight cache-resident. What that per-column cost is
(A-tile restaging per 64-column block, LDS fragment traffic, the epilogue, or
issue) is not attributed: no counters on this host. Exact K32 stays the default
and the oracle; folded stays opt-in; no selector change.

## Not claimed

- Kernel device-clock time on one WSL2 host; not a serving result (Radiance's
  activation quantization and dispatch are outside the timed launch).
- "Cache-resident" is inferred from sizes (64 MiB last-level cache, one input
  copy), not measured -- no counters.
- The packed candidates are manual, receipt-bound HIP packages, not a selected
  route; nothing here promotes them.
