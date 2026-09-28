# GFX1201 folded MXFP4 prefill: the one-row-block band (M ≤ 256)

Owner `ROCM-MXFP4-W4A8-1`; sync `GFX1201-PERF-2026-09-27`. Host **Tajasarus**
(AMD Radeon RX 9070 XT, `gfx1201`, Ubuntu 26.04 under WSL2, ROCm 10.0 /
HIP 7.15, LLVM/MLIR 23.1.1), worktree `~/programming/tessera-w-perf`, tree
`build/` (NDEBUG) for timing. Pinned Radiance `dfdfa383…` (binary
`c9f91bc8…`), fragment-order weights, `RADIANCE_MXFP4_WPERM=1`,
`RADIANCE_MXFP4_DECODE_MAX_M=64` (the recorder's setting, so Radiance runs its
folded BM256/TN2 kernel at every shape here -- the same tile as Tessera's).

| File | Clean source commit | What |
|---|---|---|
| `mxfp4_small.json` | `29926fcb` | exploration: `sgpr_base`, `sched0`, `uncond` diagnostic edits on the selected schedule, M ≤ 256 |
| `mxfp4_small2.json` | `8fcc4eb3` | exploration: `skip_idle_waves`, `wave_epilogue` and both, M ≤ 256 |
| `mxfp4_small_final.json` | `a1f4e5e7` | the promoted `row_guard` key, M ≤ 256, with the single-key decomposition |
| `mxfp4_all_final.json` | `a1f4e5e7` | the promoted schedule on all ten recorded shapes |

Every packet: `benchmarks/rocm/record_gfx1201_mxfp4_folded_load_schedule.py`,
three independent processes (alternating engine order), seven trials, matched
lossless-fold inputs. Before any timing, the exact K32 route matches an
independent sampled reference and **every engine's BF16 output is bitwise
equal to exact K32** (diagnostic engines included). Timing source: the
compiler-built device-clock marker (`--tessera-device-clock-span`,
`llvm.readsteadycounter`, 100 MHz), witnessed by HIP events in the same
window; every window agrees within 2.4%. Windows are sized for 6 ms from a
warm-up estimate; the shortest recorded is 5.34 ms in `mxfp4_small_final.json`
and **4.84 ms** in `mxfp4_all_final.json` (one sweep window under the 5 ms
target, as in the previous packet). No profiler counters (no `/dev/kfd`).

## Why the gap was where it was

At M ≤ 192 the BM256 tile has whole 64-row waves with no row below M: at
M = 128 waves 2 and 3 of every workgroup multiply the clamped last row and
their results are never stored. The earlier probes had already shown the gap
was not operand traffic; the two probes recorded in `mxfp4_small.json` show it
was not the per-lane 64-bit staging address arithmetic either (`sgpr_base`:
wave-uniform base plus 32-bit offsets, 0.99–1.00×) nor the K16 scheduling
barrier (`sched0`, mask 0 instead of 6: 1.00–1.01×). `uncond` (unguarded K16
steps) re-measured slower again (1.00–1.06×). `mxfp4_small2.json` then
measured the wasted-wave hypothesis directly:

| probe on the selected schedule | 128×5120×8704 | 128×17408×5120 | 256×5120×8704 | 256×17408×5120 |
|---|---:|---:|---:|---:|
| `skip_idle_waves` | 0.710–0.714 | 0.921–0.922 | 0.994–1.010 | 0.993–1.000 |
| `wave_epilogue` | 0.966–0.971 | 0.967–0.978 | 1.005–1.011 | 0.998–1.002 |
| both | 0.676–0.681 | 0.843–0.851 | 1.010–1.016 | 0.999–1.005 |

(time / selected, per process.) At M = 256 there is no idle wave and neither
edit moves anything. The per-wave vector epilogue matters only once idle
waves are skipped, because only then do the two live waves of a short tile
take the fast path (the CTA-level test fails whenever M < m0 + 256).

## What changed

A fifth performance key on the folded load schedule, **`row_guard`**
(`cta` | `wave`), emitted by Tile→ROCm on `tessera_rocm.scaled_wmma_gemm` and
required by the folded materializer (a missing or undeclared value is
refused). `wave`: a wave whose 64 rows all lie at or past M issues no WMMAs
and returns before the epilogue -- it still stages its share of every slab
and meets every barrier (the branch is wave-uniform) -- and the
complete-tile vector epilogue tests the wave's own 64×32 block. Tile→ROCm
selects `wave` when M is not a whole number of BM256 row blocks and `cta`
otherwise, so every whole-row-block shape keeps its kernel byte for byte
(`selected_minus_row_guard` at M = 128 has the same instruction-stream
SHA-256, `fbf530ea…`, as the selected kernel at M = 256). The tile, operand
bytes, WMMA order and each stored element's arithmetic are unchanged; exact
K32 stays the default and the oracle; folded stays opt-in.

## Results (per-process median device-clock ms)

`mxfp4_small_final.json` (the promoted key, with its decomposition):

| M×N×K | row_guard | V1 | selected | Radiance | selected/V1 | selected/Radiance |
|---|---|---:|---:|---:|---:|---:|
| 128×5120×8704 | wave | 0.1649–0.1658 | 0.1085–0.1087 | 0.1440–0.1447 | 0.654–0.659 | **0.750–0.755** |
| 128×17408×5120 | wave | 0.2915–0.2942 | 0.2408–0.2426 | 0.2253–0.2319 | 0.818–0.832 | **1.038–1.070** |
| 256×5120×8704 | cta | 0.1732–0.1742 | 0.1611–0.1627 | 0.1521–0.1529 | 0.930–0.939 | 1.053–1.068 |
| 256×17408×5120 | cta | 0.3200–0.3207 | 0.3049–0.3054 | 0.2481–0.2526 | 0.952–0.953 | 1.207–1.230 |

Leave-one-out at M = 128 (5120 / 17408): removing `row_guard` costs +49% /
+15% (0.1620–0.1629 / 0.2769–0.2845 ms); adding it alone to V1 gives
0.1238–0.1252 / 0.2868–0.2878 ms.

`mxfp4_all_final.json` (all ten shapes, the promoted schedule):

| M×N×K | row_guard | V1 | selected | Radiance | selected/V1 | selected/Radiance |
|---|---|---:|---:|---:|---:|---:|
| 128×5120×8704 | wave | 0.1634–0.1645 | 0.1024–0.1053 | 0.1366–0.1387 | 0.623–0.643 | 0.738–0.771 |
| 128×17408×5120 | wave | 0.2844–0.2965 | 0.2472–0.2495 | 0.2218–0.2227 | 0.842–0.870 | 1.112–1.120 |
| 256×5120×8704 | cta | 0.1630–0.1643 | 0.1544–0.1549 | 0.1460–0.1468 | 0.942–0.947 | 1.054–1.061 |
| 256×17408×5120 | cta | 0.3072–0.3160 | 0.2871–0.2988 | 0.2423–0.2449 | 0.909–0.967 | 1.172–1.233 |
| 512×5120×8704 | cta | 0.2760–0.2766 | 0.2411–0.2431 | 0.2298–0.2352 | 0.872–0.879 | 1.025–1.058 |
| 512×17408×5120 | cta | 0.5555–0.5611 | 0.4690–0.4763 | 0.4453–0.4569 | 0.836–0.857 | 1.027–1.065 |
| 1024×5120×8704 | cta | 0.5202–0.5235 | 0.4144–0.4166 | 0.4362–0.4445 | 0.795–0.797 | 0.932–0.955 |
| 1024×17408×5120 | cta | 1.1705–1.1761 | 0.9030–0.9191 | 0.8982–0.9195 | 0.772–0.781 | 0.982–1.020 |
| 2048×5120×8704 | cta | 1.0571–1.0803 | 0.8408–0.8694 | 0.9759–0.9833 | 0.795–0.805 | 0.862–0.884 |
| 2048×17408×5120 | cta | 2.2594–2.3104 | 1.8121–1.8364 | 1.9347–1.9591 | 0.795–0.802 | 0.936–0.939 |

M = 128 is now 0.74–0.77× Radiance at N = 5120 and 1.04–1.12× at N = 17408
across the two packets (it was 1.10–1.11× and 1.24–1.27×). Whole row blocks
are unchanged and within this packet's run-to-run spread of the previous one.

## ISA (selected symbol, process 0)

| | V1 | selected, whole row block (`cta`) | selected, partial (`wave`) |
|---|---:|---:|---:|
| `v_wmma_f32_16x16x16_fp8_fp8` | 32 | 32 | 32 |
| `s_barrier_signal` / `s_barrier_wait` | 2 / 2 | 2 / 2 | 2 / 2 |
| instructions | 4195 | 6954 | 7183 |
| VGPR / SGPR / LDS | 109 / 54 / 25600 | 123 / 54 / 25600 | 125 / 54 / 25600 |
| scratch / spills | 0 / none | 0 / none | 0 / none |

## Device proof

`tests/device/rocm/test_mxfp4_folded_prefill.py` adds three partial-row-block
shapes (100×80×128: waves 2–3 idle and wave 1 partial; 192×64×64: wave 3
idle, waves 0–2 on the per-wave vector epilogue; 65×48×64: one live row),
each lossless and lossy, bitwise against the independent FP32-partial oracle
for V1, the compiled route (which selects `wave`) and every single-key
schedule including `row_guard = wave` alone. See
`../gfx1201_fp8_blockscale_lds_20260927/device_tests_gfx1201.txt` for the
MXFP4 device-file totals on both trees.

## Open

M = 256 (one full row block) stays 1.05–1.23× behind Radiance; nothing here
moved it, and without profiler counters it is not attributed. Radiance reads
packed E2M1 weights (half of the expanded E4M3 bytes) and upconverts in LDS;
whether that matters at one row block is untested. The rule is fitted on
gfx1201 only; gfx1151 has no folded route.
