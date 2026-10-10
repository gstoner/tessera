# Widened gfx1201 short-K four-format comparison

Owner ROCM-FP8-BLOCKSCALE-1 / E2E-REAL-6.
Sync GFX1201-SHORT-K-WIDE-2026-10-08.

Owning RX 9070 XT/gfx1201, GPU-28d9e7efbf2ef716. The packet records compiler
and source fingerprints, shared operand hashes, native image identities,
21 alternating paired windows, and before/after correctness checks.
Candidate/reference compiler SHAs are the earlier native short-K experiment;
this is not proof of subsequent shared/NVIDIA compiler changes.

Ratio is candidate/reference resident HIP graph device execution plus dispatch.
It excludes host transfers and is not isolated kernel time or AITER comparison.
Four shapes by four formats pass. Byte-identical controls are preserved.
Folded MXFP4 reports its approximation policy and zero folding-output error
for these chosen operands; that does not establish general exact MXFP4 semantics.

| M/N/K | Format | Paired ratio | Identical image | Native route |
| --- | --- | ---: | --- | --- |
| [200, 8192, 1024] | fp8_k128_n128 | 0.999655 | True | gfx1201_lds_wmma_blockscale_nk_128x64_w8_d1_k128 |
| [200, 8192, 1024] | fp8_k32_n1 | 1.000717 | True | gfx1201_lds_wmma_blockscale_nk_128x64_w8_d1_k32 |
| [200, 8192, 1024] | mxfp8_k32_n1 | 1.014358 | True | gfx1201_lds_wmma_blockscale_nk_128x128_w8_d1_k32 |
| [200, 8192, 1024] | mxfp4_folded | 1.018586 | True |  |
| [200, 2048, 2048] | fp8_k128_n128 | 1.008321 | True | gfx1201_lds_wmma_blockscale_nk_128x64_w8_d1_k128 |
| [200, 2048, 2048] | fp8_k32_n1 | 0.999248 | True | gfx1201_lds_wmma_blockscale_nk_128x64_w8_d1_k32 |
| [200, 2048, 2048] | mxfp8_k32_n1 | 0.993210 | True | gfx1201_lds_wmma_blockscale_nk_128x64_w8_d1_k64 |
| [200, 2048, 2048] | mxfp4_folded | 1.012373 | True |  |
| [197, 1056, 2048] | fp8_k128_n128 | 0.091979 | False | gfx1201_register_wmma_blockscale_nk_1x2_k128 |
| [197, 1056, 2048] | fp8_k32_n1 | 1.001710 | True | gfx1201_register_wmma_blockscale_nk_1x2_k32 |
| [197, 1056, 2048] | mxfp8_k32_n1 | 1.000273 | True | gfx1201_register_wmma_blockscale_nk_1x1_k32 |
| [197, 1056, 2048] | mxfp4_folded | 0.997501 | True |  |
| [256, 1024, 2304] | fp8_k128_n128 | 0.439026 | False | gfx1201_register_wmma_blockscale_nk_2x2_k128 |
| [256, 1024, 2304] | fp8_k32_n1 | 1.003389 | True | gfx1201_register_wmma_blockscale_nk_2x2_k32 |
| [256, 1024, 2304] | mxfp8_k32_n1 | 0.999178 | True | gfx1201_register_wmma_blockscale_nk_1x1_k32 |
| [256, 1024, 2304] | mxfp4_folded | 1.002916 | True |  |

14 of 16 image pairs are byte-identical. Both M=200 K128 rows select
the unchanged LDS route and remain near parity. This compiler change does not
fix those historical AITER gaps. Ragged M=197/N=1056/K=2048 uses the changed
register route and measures candidate/reference 0.091979.
The K=2304 fallback register case measures 0.439026 despite not selecting a
constant-K arm; resource/instruction attribution remains required.
K32 FP8, MXFP8 and folded MXFP4 controls do not establish a gain.
No selector promotion, sibling architecture proof or broad performance closure.
