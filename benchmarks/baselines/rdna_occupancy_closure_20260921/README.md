# RDNA occupancy: closure evidence — 2026-09-21

Sync `RDNA-OCCUPANCY-GRANULE-2026-09-20`. Closes the three items left open by
PR #789. Device work on **Princess-Luna** (gfx1151, RDNA 3.5, ROCm 10.0).

## 1. Wave slots per SIMD — closed, three independent sources

PR #789 left this resting on one source: the probe's occupancy plateau. Neither
the RDNA4, RDNA3.5 nor CDNA5 manual states a wave-slot count. Two further
sources now agree, and neither is the compiler remark:

| source | evidence | independent of |
|---|---|---|
| HSA runtime (`rocminfo` agent properties) | gfx1151 and gfx1201 both: `Max Waves Per CU: 32`, `SIMDs per CU: 2` → **16/SIMD** | the compiler entirely |
| LLVM `AMDGPUBaseInfo.cpp::getMaxWavesPerEU` | `hasGFX10_3Insts(STI) ? 16 : 20` → **16** for gfx11/gfx12 | HSA and the remark |
| `-Rpass-analysis=kernel-resource-usage` plateau | **16** | — |

`rocminfo` independently corroborates two other PR #789 changes: `SIMDs per CU: 2`
confirms `_SIMDS_PER_CU`, and `Max Waves Per CU: 32` confirms the `_MAX_WAVES`
16→32 unit correction. Upstream LLVM 23.1.1 and the ROCm fork carry byte-identical
`getMaxWavesPerEU`.

**CDNA checked too.** `getMaxWavesPerEU` returns 8 when `isGFX90A`, and a
compiler probe confirms gfx90a, gfx942 and gfx950 all report **8 waves/SIMD** —
so `isGFX90A` covers the whole CDNA2/3/4 family and the derived `32 / 4 SIMDs =
8` is right. No discrepancy.

## 2. The G6 kernel at ≤120 VGPRs — measured, and the answer is NO

PR #789 found the two-wave D=128 attention kernel at 121 VGPRs, one register
above the measured 12-wave rung (which ends at 120). Nobody had checked whether
those two waves were worth what they cost. They are not.

**The rung is reachable, and free of spills.** Stamping ROCDL's `waves_per_eu`
attribute (an env-gated scaffold, since deleted):

| request | VGPRs | occupancy | spills | scratch |
|---:|---:|---:|---:|---:|
| none | 121 | 10 waves/SIMD | 0 | 0 |
| 11 or 12 | **113** | **12 waves/SIMD** | **0** | **0** |
| 13+ | 96 | 16 waves/SIMD | 9 | 40 B |

The model predicted the boundary at 120; the allocator landed at 113, under it.

**And it is 2.2x slower.** Nine interleaved trials per point, `one_wave` as a
fixed control (the scaffold was gated to the two-wave path so the control's
register count could not move):

| shape | causal | schedule | VGPRs | baseline | candidate | delta |
|---|---|---|---|---:|---:|---:|
| 1x16x1009x128 | yes | one_wave *(control)* | 218 → 218 | 0.7344 ms | 0.7202 ms | +2.0% |
| 1x16x1009x128 | yes | **two_wave** | **121 → 113** | **0.7343 ms** | **1.6195 ms** | **−54.7%** |
| 1x16x1024x128 | no | one_wave *(control)* | 218 → 218 | 1.2770 ms | 1.2765 ms | +0.0% |
| 1x16x1024x128 | no | **two_wave** | **121 → 113** | **1.3640 ms** | **3.1649 ms** | **−56.9%** |

Correctness held throughout (the harness asserts max abs error ≤ 3e-3).

**Mechanism, from the disassembly** — not inferred:

| metric | 121 VGPR | 113 VGPR | delta |
|---|---:|---:|---:|
| `s_waitcnt` | 33 | **45** | **+36%** |
| SALU | 412 | 444 | +32 |
| VALU | 690 | 686 | −4 |
| `ds_` (LDS) | 58 | 58 | — |
| `global_`/`buffer_` | 26 | 26 | — |
| `v_wmma` | 2 | 2 | — |
| `scratch_` | 0 | 0 | — |

Identical memory traffic, identical WMMA count, no scratch — and 36% more
waits. Freeing 8 registers required shortening live ranges, which moves loads
closer to their uses and destroys instruction-level latency hiding. On a kernel
already bound by unhidden load latency, trading ILP for two more waves loses,
and loses by a lot.

## 3. Occupancy binding in tile selection — rejected on this evidence

Item 2 *is* the exact-device proof item 1 was gated on, and it came back
negative. On the one kernel where the rung could be crossed in isolation,
**occupancy and performance move in opposite directions**: +20% occupancy,
−55% throughput. A selector that maximised occupancy would have chosen the
2.2x-slower kernel here.

This is consistent with what the repo already recorded from a different
direction — `matmul_opt_ladder.py`: *"Wins despite 67%→17% occupancy —
arithmetic intensity beats occupancy."*

So `RankedTileCandidate.occupancy_waves_per_simd` stays **reported, never
scored**, and that is now a measured position rather than caution.

## What this does not show

One kernel family, one arch (gfx1151), one rung crossing. It does **not** show
occupancy never matters — it shows occupancy is not *sufficient* to rank, and
that a maximising selector is wrong at least sometimes. A kernel that is
occupancy-bound rather than latency-bound could well go the other way; nothing
here has measured one.

The scaffold that produced the 113-VGPR build is deleted (Decision #29): its
hypothesis is answered. Reproducing means re-adding a `rocdl.waves_per_eu`
stamp to the flash-attention generator.
