# gfx1201 VGPR file, allocation granule and wave slots — 2026-09-20

Sync `RDNA-OCCUPANCY-GRANULE-2026-09-20`. Compile-time probe, run on
**Tajasarus** (RX 9070 XT, gfx1201, WSL2, ROCm 10.0 / HIP 7.15, AMD clang
23.0.0git). Produced by `scripts/probe_rdna_vgpr_granule.py`.

## Result

| constant | measured | provenance |
|---|---|---|
| VGPR file per SIMD | **1536** | unique solution over 12 observations |
| wave32 allocation granule | **24** | unique solution; matches RDNA4 ISA 3.3.2.1 |
| wave slots per SIMD | **16** | unique solution |

The probe does not read these from any tool — no tool reports them. It compiles
twelve kernels at varying register pressure, reads the `(VGPRs, Occupancy
[waves/SIMD])` pair from the compiler's own `-Rpass-analysis=kernel-resource-usage`
remark, and solves for the `(file, granule, wave_slots)` triple that explains
every observation. Exactly one triple survives.

## Observations

| VGPRs | occupancy (waves/SIMD) |
|---:|---:|
| 14 | 16 |
| 24 | 16 |
| 31 | 16 |
| 39 | 16 |
| 47 | 16 |
| 62 | 16 |
| 78 | 16 |
| 111 | 12 |
| 132 | 10 |
| 164 | 9 |
| 196 | 7 |
| 228 | 6 |

`tessera.compiler.rocm_occupancy.estimate_occupancy` reproduces **12/12** of
these, with no parameter fitted to them — the model was written from the ISA
before the probe ran.

## Verified against the RDNA4 ISA PDF (2026-09-20)

Checked against AMD's "RDNA4" Instruction Set Architecture manual (707 pp.),
not the extracted archive. Note the archive's `sections/*.md` headers cite
**PDF-physical** page numbers, not the document's own footer numbering — the
two differ by 10.

| claim | ISA | verdict |
|---|---|---|
| granule 24 for a 1536-VGPR-per-SIMD device | 3.3.2.1, verbatim | **confirmed** |
| per-wave cap 256 VGPRs | 3.3.2.1; Table 4 `V0-V255` | **confirmed** (not pinned by the probe — see below) |
| 4 SIMD32s per WGP | 2.3, "any of the 4 SIMD32s" | **confirmed** (3rd source) |
| LDS 128 KiB/WGP as two 64 KiB halves, one per CU | 3.3.5 | **confirmed** |
| 64 KiB max LDS per work-group | 2.3, 3.3.5, Table 4 | **confirmed** |
| 128 B per wave32 VGPR | Table 4, "32 bits per work-item x 32" | **confirmed** |
| wave slots per SIMD = 16 | **not stated anywhere in the ISA** | probe-measured only |

Three ceilings the ISA supplied that the model was missing, now implemented:

* **LDS allocates in 1 KiB blocks** (3.3.5). Dividing the pool by an unrounded
  request overstates residency — a 26000 B request occupies 26624 B, so four
  work-groups fit on a WGP, not the five naive division gives.
* **1024 work-items per work-group** (2.3) — now refused rather than clamped.
* **32 work-groups per WGP** (2.3) — applied, but measured to be *exactly
  coincident* with the wave-slot ceiling on this part (4 x 16 = 64 waves ÷ 2
  waves minimum = 32), so it cannot change an answer here. Marked as
  non-binding at the site and asserted so by test.

## Independent corroboration — AMD matrix instruction calculator

`ROCm/amd_matrix_instruction_calculator` (on Tajasarus) is a second,
non-overlapping source: pure Python, no device, AMD-authored. It does not know
the VGPR file size, granule or wave-slot count, so it cannot confirm those
directly — but its per-instruction "Execution statistics" block pins a constant
this model depends on, and settles a separate architectural claim.

**SIMDs per WGP = 4**, from `FLOPs/WGP/cycle ÷ (FLOPs ÷ execution cycles)`:

| instruction | FLOPs | cycles | per-SIMD | per-WGP | ⇒ SIMDs/WGP |
|---|---:|---:|---:|---:|---:|
| `v_wmma_f32_16x16x16_f16` | 8192 | 16 | 512 | 2048 | **4** |
| `v_wmma_f16_16x16x16_f16` | 8192 | 16 | 512 | 2048 | **4** |
| `v_wmma_f32_16x16x16_fp8_fp8` | 8192 | 8 | 1024 | 4096 | **4** |
| `v_swmmac_f32_16x16x32_f16` | 16384 | 16 | 1024 | 4096 | **4** |

That is `rocm_occupancy._SIMDS_PER_WGP`, and it closes the byte arithmetic for
the VGPR file: 1536 VGPRs/SIMD x 128 B (32 lanes x 4 B) = 192 KiB/SIMD, x4 =
**768 KiB per WGP** — not per CU, as previously recorded.

**Matrix ops cannot co-execute with VALU on RDNA4.** `Can co-execute with
VALU: False` for all 22 gfx1201 matrix instructions, and all 6 on RDNA3, while
CDNA1/2/3 report True for 16/20, 25/27 and 33/40. Not an occupancy fact, but
it belongs with this evidence: on RDNA4 matrix latency is hidden by occupancy
alone, because there is no co-issue mechanism to hide it with. It also
independently voids any WMMA-plus-elementwise dual-issue scheme, which the
closed 17-entry `V_DUAL_*` opcode table (ISA 16.11) already forbids by
encoding.

## What this settles

1. **The ISA rule holds on gfx1201.** Granule is 24, as ISA 3.3.2.1 prescribes
   for a 1536-register file. The alternative (16) is refuted: it predicts 13
   waves at 111 VGPRs where the device reports 12.
2. **The ROCm queue's recorded "121 VGPRs → 12 waves/SIMD" is wrong.** The
   measured rung boundary is 120: 111 VGPRs gives 12, and 132 gives 10. A
   121-VGPR kernel gets **10** waves/SIMD and is one register above the
   12-wave rung.
3. **Wave slots are per SIMD, not per CU.** 16/SIMD means 32/CU, so
   `rocm_target._MAX_WAVES` (16, commented "Maximum waves per CU") has either
   the wrong number or the wrong unit. Not changed here — it bounds a
   user-supplied profile field, and relaxing a validator is a separate change.

## Not proven here

This is a **compile-time** probe: it reads the compiler's resource model, not a
running kernel's achieved occupancy. It says what the hardware allocates, not
what a workload attains. No kernel was launched and no latency was measured.
Nothing here transfers to **gfx1151**, which remains unmeasured and stays in
`VGPR_GRANULE_CONTESTED` — run the same probe with `--arch gfx1151` on
Princess-Luna.

## Reproduce

    source ~/.config/tessera/env.sh && source scripts/_rocm_env.sh
    python3 scripts/probe_rdna_vgpr_granule.py --arch gfx1201 \
        --output benchmarks/baselines/gfx1201_vgpr_granule_20260920/probe.json
