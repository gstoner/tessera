# GFX1201 folded MXFP4 prefill: load schedule carried in Target IR

Owner `ROCM-MXFP4-W4A8-1`; sync `GFX1201-LANES-2026-09-27`. Host
**Tajasarus** (AMD Radeon RX 9070 XT, `gfx1201`, WSL2, ROCm 10.0 / HIP 7.15,
LLVM/MLIR 23.1.1), clean source `61f5fd19` (recorded in both packets with
`worktree_dirty: false` and per-file SHA-256). Pinned Radiance
`dfdfa383…` (binary `c9f91bc8…`), fragment-order weights, `RADIANCE_MXFP4_WPERM=1`.

## What changed

The opt-in folded BM256/BN64/BK64, TM4/TN2 kernel (`approx_bm256_tm4.v1`) now
takes a `FoldedPrefillSchedule` of four **performance keys**. Tile→ROCm
lowering emits them on `tessera_rocm.scaled_wmma_gemm` and the folded
materializer consumes them (a missing or undeclared value is refused):

| key | selected value | effect |
|---|---|---|
| `raster_group_m` | 4 | 1-D grid visiting 4 row blocks per weight column tile |
| `workgroup_mode` | `cu` if M spans ≥ 2 BM256 row blocks, else `wgp` | `-mcumode` |
| `staging_prefetch` | `register_next_slab` | next K64 A/B slab requested into registers before the current slab's WMMAs |
| `epilogue_schedule` | `complete_tile_vector_scales` | complete, 16-byte-aligned tiles fetch 32 activation scales as 8 vector loads ahead of use; all other launches run the predicated per-element epilogue |

None changes the tile, operand bytes, per-lane WMMA order or any element's
epilogue arithmetic. Direct packaging keeps the original schedule (`V1`),
whose emitted HIP source is byte-identical to the relabel-era generator
(`v1_source_identity.json`), so every sealed packet's kernel is unchanged.
Exact K32 remains the default and oracle; folded stays opt-in; there is no
automatic folded selection.

## Method

`benchmarks/rocm/record_gfx1201_mxfp4_folded_load_schedule.py`, three
independent processes (alternating engine order), matched lossless-fold
inputs. Before timing: the exact K32 route matches an independent sampled
FP32-dequantized reference, and **every engine's BF16 output is bitwise equal
to exact K32**. Timing source: the compiler-built device-clock marker
(`--tessera-device-clock-span`, `llvm.readsteadycounter`, 100 MHz) bracketing
windows sized for 6 ms (shortest recorded: 5.33 ms production, 4.94 ms sweep —
the recorder sizes windows from a warm-up estimate, so one sweep window fell
just under its 5 ms target), witnessed by HIP events in the same window.
Every window's device clock agrees with its HIP event within 2.3% (band 5%);
marker-bracketed versus plain event medians differ by at most 2.8%
(production) and 4.3% (sweep). No profiler counters
(no `/dev/kfd`); nothing here is a DRAM or phase measurement.

## Results (per-process median device-clock ms; three processes)

Production shapes (`production.json`, 11 trials, with decomposition):

| M×N×K | V1 | selected (compiled route) | Radiance | selected/V1 | selected/Radiance |
|---|---:|---:|---:|---:|---:|
| 256×5120×8704 | 0.1731–0.1737 | 0.1619–0.1627 | 0.1518–0.1531 | 0.932–0.940 | 1.062–1.068 |
| 1024×17408×5120 | 1.148–1.158 | 0.906–0.924 | 0.915–0.923 | 0.789–0.798 | 0.982–1.010 |

Eight-shape sweep (`sweep.json`, 7 trials), selected/V1 and selected/Radiance
ranges over processes:

| M×N×K | selected/V1 | selected/Radiance |
|---|---:|---:|
| 128×5120×8704 | 0.969–0.980 | 1.102–1.110 |
| 128×17408×5120 | 0.926–0.949 | 1.237–1.266 |
| 256×17408×5120 | 0.916–0.953 | 1.185–1.239 |
| 512×5120×8704 | 0.876–0.884 | 1.064–1.068 |
| 512×17408×5120 | 0.817–0.854 | 1.020–1.061 |
| 1024×5120×8704 | 0.786–0.804 | 0.945–0.973 |
| 2048×5120×8704 | 0.797–0.803 | 0.861–0.882 |
| 2048×17408×5120 | 0.794–0.799 | 0.936–0.940 |

The selected schedule is faster than V1 on all ten shapes and reaches or
passes Radiance from M = 1024; one-row-block shapes (M ≤ 256) remain
1.06–1.27× behind, worst at N = 17408.

### Decomposition (process 0; the three processes agree within 3%)

| engine | 256×5120×8704 | 1024×17408×5120 |
|---|---:|---:|
| V1 | 0.1737 | 1.1539 |
| V1 + raster | 0.1735 | 1.1084 |
| V1 + CU mode | (not selected at one row block) | 1.0703 |
| V1 + prefetch | 0.1683 | 1.1358 |
| V1 + vector epilogue | 0.1659 | 1.1505 |
| selected − raster | 0.1618 | 0.9776 |
| selected − CU mode | — | 1.1208 |
| selected − prefetch | 0.1669 | 0.9595 |
| selected − vector epilogue | 0.1695 | 0.9598 |
| selected | 0.1619 | 0.9199 |

At one row block the raster is a no-op (as designed) and the gain is the
epilogue plus prefetch. At four row blocks CU mode is the largest single
lever and the keys compound: removing CU mode costs 22%, any other key 4–6%.
Why CU mode helps is **not attributed** (no counters); the one-row-block
regression that motivated the M rule was measured in the exploration below.

## ISA (selected symbol of each timed HSACO)

| | V1 | selected |
|---|---:|---:|
| `v_wmma_f32_16x16x16_fp8_fp8` | 32 | 32 |
| `s_barrier_signal` / `s_barrier_wait` | 2 / 2 | 2 / 2 |
| `global_load_b128` (static sites) | 5 | 18 |
| `ds_load_*` / `ds_store_b128` | 16 / 5 | 16 / 5 |
| `s_wait_loadcnt` | 71 | 76 |
| VGPR / SGPR / LDS | 109 / 54 / 25600 | 123 / 54 / 25600 |
| scratch / spills | 0 / none | 0 / none |

The WMMA count and barrier structure are unchanged; the extra b128 sites are
the register prefetch and the vector-scale epilogue. 123 VGPRs round to the
same 10-wave LDS-bound residency as 109.

## Exploration and measured-negative results (HIP events, diagnostic)

Recorded during exploration with `ablate_gfx1201_folded_load_schedule.py` on
the same host and inputs, bitwise-equal outputs, HIP-event medians of 11
interleaved trials; these are diagnostic, not the promotion evidence above.

* **Hot-stream probes** (output-changing, `diagnostic_bound`): serving A, B or
  both from a cache-hot column made the one-row-block shape only 3.6%/3.6%/4.7%
  faster (164.6 → 156.8 µs, Radiance 141 µs) and the wide shape 8–10% faster.
  The small-shape gap is therefore not operand traffic.
* **Unconditional K16 steps** lost 7%/2% (175.1 vs 163.7 µs; 1180 vs 1154 µs),
  confirming the earlier refusal.
* **LDS fragment double buffering** (one-step look-ahead, split barrier; its
  ISA matches Radiance's main-loop shape) lost 6%/1.5% on V1; on top of the
  selected schedule it gained ≤ 1.6% at 150 VGPRs — not selected.
* **LDS-only barrier fences** were neutral (±2%).
* **CU mode alone** lost 3% at 256×5120 and 6–10% at 256×17408, and won
  11–24% from M = 512 — hence the row-block rule.
* **Raster groups 2/4/8** were equivalent (±0.5%); 4 is selected.

## Device proof (Tajasarus, this branch)

`tests/device/rocm/test_mxfp4_folded_prefill.py` 26/26 with both the normal and
the assertions-ON `tessera-opt`, adding: bitwise preservation on nonuniform,
ragged, multi-row-block, multi-slab, lossless **and lossy** folds against an
independent bitwise FP32-partial oracle for V1, the compiled route and every
single-key schedule; the extreme-scale fallback inside the vector epilogue;
and a misaligned activation-scale pointer that must take the predicated path.
All MXFP4 device files: 114 passed. IR lit 451 passed / 66 unsupported and
`check-tessera-rocm` 82/82 on both trees.

## Limits

Kernel device-clock time on one WSL2 host, not model latency or DRAM
traffic. The M rule for CU mode is fitted to these ten shapes on this part
and does not transfer to gfx1151. Radiance may choose a different internal
variant at some shapes (e.g. TN4 at M ≥ 2048); it is compared as shipped.
