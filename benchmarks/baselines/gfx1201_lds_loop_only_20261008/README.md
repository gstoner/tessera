# gfx1201 LDS loop-only K2048 diagnostic

Owner ROCM-FP8-BLOCKSCALE-1 / E2E-REAL-6.
Sync GFX1201-LDS-LOOP-ONLY-2026-10-08.

The isolated native candidate clones only the uniform scale-group loop for K2048, sharing the LDS prologue and masked epilogue. The generic fallback, reduction order, private storage and barriers remain. Primary compiler source and selector are unchanged. candidate.patch is the exact source delta; candidate compiler SHA256 50e3d9c4159aca0db66adf40888280f5804fe872bc1ea7eb7b4c1104506623a2; matched control 6075bf1450770b22635f9a22959638e4845287030d716d37c513cd7a8db51eb0.

Eight owning gfx1201 image-reuse/final-prefetch tests pass. Sixteen FP8 K128, FP8 K32, MXFP8 and explicitly approximate folded MXFP4 rows pass their numerical bounds before/after each timing window.

## Recorder repair

The original single slow calibration could yield later windows below the requested minimum. Its packet is retained as inadmissible historical evidence, not a performance finding. The recorder now calibrates both arms repeatedly, admits a complete paired series only when every device window meets the requested floor, and retains any rejected whole series before doubling the shared launch count. Three meaningful duration-admission tests and 21 affected recorder regression tests pass.

Final duration-admitted packet uses 21 alternating resident graph windows per arm and a 6 ms minimum. All admitted windows exceed 7.39 ms. Native device clock timing includes graph dispatch and has a separate HIP-event cross-check; this is not isolated kernel or AITER comparison. timing_quality.json identifies two unchanged-image rows whose device/event disagreement exceeds 5%; those rows remain diagnostic and cannot support a precision performance claim.

## Result

| K128 MNK | Paired candidate/control ratio | Control/candidate registers |
| --- | ---: | ---: |
| [200, 8192, 1024] | 0.983517 | 192/223 |
| [200, 2048, 2048] | 0.982179 | 192/223 |
| [197, 1056, 1536] | 1.000049 | 256/256 |
| [256, 1024, 2304] | 0.992569 | 256/256 |

The two M200 K128 images use 223 registers instead of 192, with unchanged 27648-byte static LDS, zero local spill bytes and reported two-block occupancy. Apparent 1.6–1.8% paired gains are comparable to unchanged-image control variation. No candidate promotion or structural speedup is justified. Register/fallback rows and all other formats retain identical images. Wider W8A8 short/ragged coverage, vendor comparison and MXFP4 M256 Radiance attribution remain open.

Build failures/repairs and both raw timing runs are retained separately. This packet proves its source/binary snapshot, not later compiler changes. gfx1151 lacks this RDNA4 FP8 WMMA recipe; NVIDIA, Apple and x86 get no physical schedule or execution claim from this experiment.
