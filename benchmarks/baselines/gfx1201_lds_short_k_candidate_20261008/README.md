# Isolated gfx1201 LDS short-K candidate

Owner ROCM-FP8-BLOCKSCALE-1 / E2E-REAL-6.
Sync GFX1201-LDS-SHORT-K-2026-10-08.

The candidate extends native uniform K=1024/1536/2048 dispatch specialization to the K128/fp32-scale runtime LDS generator. It was built in isolated scratch; the authoritative compiler source and selector are unchanged.

Owning RX 9070 XT/gfx1201: eight existing LDS image-reuse/final-prefetch tests pass. The matched recorder evaluates four shapes across FP8 K128, FP8 K32, MXFP8 and folded MXFP4, validating shared operands before/after 21 interleaved resident HIP-graph windows. Device graph timing includes dispatch; it is neither isolated kernel time nor an AITER comparison.

- M200 N8192 K1024 K128 candidate/reference paired ratio: 1.178661 (regression).
- M200 N2048 K2048 K128: 0.940124 (improvement).
- Ragged M197 N1056 K1536 K128: 0.997247.
- Fallback M256 N1024 K2304 K128: 0.993950.

The candidate is unpromoted because its first named M200 case regresses. Further resource/ISA attribution and a revised schedule are needed; no broad short-K performance closure. Folded MXFP4 remains its explicitly approximate physical policy.

Recorder: benchmarks/rocm/record_gfx1201_interleaved_compiler_formats.py.
Raw matched samples, image identities and numerical checks: packet.json.

## Live resource attribution

A fresh eight-row run retains both named M200 shapes and all four formats. The recorder queries hipFuncGetAttribute and module occupancy on the actual loaded function at its real 256-thread block and zero dynamic LDS, and records distinct compiler-source snapshots.

K128 control/candidate registers per thread are 192/241. Static LDS stays 27,648 bytes; local bytes per thread stays zero; reported active blocks per multiprocessor stays two. The K1024 paired regression reproduces at 1.181637; K2048 reproduces at 0.933179. Six FP8 K32/MXFP8/folded MXFP4 controls retain matching resource counts and near-parity paired timing.

Register pressure increased; these measurements do not isolate causal instruction stalls or prove an occupancy drop. Further ISA/code-size and narrower specialization investigation is needed before changing a production selector. Raw results: resources_packet.json.

## K2048-only revision

The isolated native revision specializes only K2048 in the runtime LDS path; the register-route specializations stay unchanged. The original runtime body handles every other K. Uniform branching preserves workgroup barriers and the checked ABI. The primary compiler binary is unchanged.

Owning gfx1201 gate: eight image-reuse/final-prefetch checks pass. Sixteen four-format rows verify numerical bounds before and after every interleaved timing window.

| Named K128 shape | Paired candidate/reference ratio |
| --- | --- |
| M200 N8192 K1024 | 1.025554 |
| M200 N2048 K2048 | 0.934908 |
| M197 N1056 K1536 | 0.967551 |
| M256 N1024 K2304 | 0.995457 |

M200 LDS registers decline from the previous candidate's 241 to 209, versus 192 for control. Static LDS, zero local memory and reported two-block occupancy remain unchanged. Ragged/fallback rows use the register route and retain matching resources; their timing variations are not a generator gain claim.

The K2048 gain remains, but K1024 still measures 2.6% slower. This revision stays isolated; no selector promotion or architecture-wide closure. Raw samples and source/compiler hashes: k2048_only_packet.json. Reviewable native source change: k2048_only_candidate.patch.
