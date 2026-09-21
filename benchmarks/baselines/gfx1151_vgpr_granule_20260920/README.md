# gfx1151 VGPR file, allocation granule and wave slots — 2026-09-20

Sync `RDNA-OCCUPANCY-GRANULE-2026-09-20`. Compile-time probe, run on
**Princess-Luna** (Radeon 8060S / Strix Halo, gfx1151, RDNA 3.5, WSL2,
ROCm 10.0). Produced by `scripts/probe_rdna_vgpr_granule.py`.

## Result — identical to gfx1201, and measured rather than inherited

| constant | measured | 
|---|---|
| VGPR file per SIMD | **1536** |
| wave32 allocation granule | **24** |
| wave slots per SIMD | **16** |

The two ROCm parts agree. **That agreement is the result, not the
assumption** — `ROCM_AUDIT.md` is explicit that proof never transfers between
gfx1151 and gfx1201, so each was probed on its own silicon. Had this run
differed, the shared `rocm_occupancy` model would have needed per-arch
branching.

## Observations

| VGPRs | occupancy (waves/SIMD) |
|---:|---:|
| 14 | 16 |
| 25 | 16 |
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

Pinned by `test_model_reproduces_the_gfx1151_device_measurement`.

## RDNA3.5 ISA verification

Checked against AMD's "RDNA3.5" Instruction Set Architecture manual (644 pp.),
not the extracted archive. The clauses the model depends on are **identical in
wording** to RDNA4's:

| claim | RDNA3.5 ISA | RDNA4 ISA | verdict |
|---|---|---|---|
| granule 24 for a 1536-VGPR/SIMD device | 3.3.2.1 | 3.3.2.1 | same sentence |
| per-wave cap 256 | 3.3.2.1 | 3.3.2.1 | same |
| LDS allocated in 1024-byte blocks | 3.3.4 | 3.3.5 | same |
| LDS = two 64 KiB blocks, CU0 at 0–65535, CU1 at 65536–131071 | 3.3.4 | 3.3.5 | same |
| 4 SIMD32s per WGP | 2.3 | 2.3 | same |
| 32 work-groups / 1024 work-items per WGP | 2.3 | 2.3 | same |
| single-wave work-groups exempt from the 32 limit | 2.3 | 2.3 | same |
| wave slots per SIMD | **not stated** | **not stated** | probe-measured only |

So the arch-independent parts of `rocm_occupancy` are ISA-verified for both
parts rather than assumed to generalise from one.

## Not proven here

Compile-time only: it reads the compiler's resource model, not a running
kernel's achieved occupancy. No kernel was launched, no latency measured, no
performance claim.

## Reproduce

    source scripts/_rocm_env.sh
    python3 scripts/probe_rdna_vgpr_granule.py --arch gfx1151 \
        --output benchmarks/baselines/gfx1151_vgpr_granule_20260920/probe.json
