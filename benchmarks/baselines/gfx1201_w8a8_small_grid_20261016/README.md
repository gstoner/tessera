# gfx1201 W8A8 small-grid Schedule follow-through

Owner: ROCM-FP8-BLOCKSCALE-1 / E2E-REAL-6.
Sync: GFX1201-W8A8-SHORT-RAGGED-20261016.

## Compiler boundary

The native PM pass selects the physical recipe from a typed Graph scaled
matmul. Schedule -> Tile -> ROCm Target -> GPU/ROCDL/LLVM -> HSACO and the
checked runtime ABI remain the executable route. Python's panel oracle is a
differential check, not a selector or kernel constructor. No public operation,
dtype, diagnostic, pass or ABI is added.

The candidate selects 64x64, four-wave, single-buffer LDS tiles only for:

- gfx1201 NK W8A8 E4M3 operands and FP32 K128/N128 scale blocks;
- 64 < M < 128, excluding M divisible by 32;
- N1024..3072, divisible by128;
- K1024..2048, with the existing whole-scale-group admission.

The old selector remains outside that envelope. In particular, aligned M96,
wide N, larger M, KN storage and other scale contracts keep their prior policy.
MXFP8/MXFP4 and gfx1151 physical selectors are unchanged. No sibling execution
or performance result transfers from RX9070XT.

## Evidence and timing protocol

The owning host is Tajasaurus, RX9070XT/gfx1201. LLVM/MLIR23.1.1 tools were
built on Super-Bear and transferred with hashes; this is not a local Tajasaurus
compiler build. ROCm10.0 native image provider and isolated Triton3.8.0 AITER
reference identities are retained under diagnostics/. The frozen baseline
compiler manifest identifies source0c25a75; post-merge-equivalence.json binds
its source tree to merged main e8a782d. Diagnostic packets retain their actual
source labels, including unpromoted prototype checkpoints.

All benchmark arms must pass an independent FP64 numerical comparison before
paired, interleaved device-clock windows. Each window is cross-checked with
HIP events. Public completed host timings include staging/transfers and are
reported separately. F32 and BF16 are separate output contracts; compare AITER
only with the matching BF16 arm. No hardware counters are available. Resource
records describe exact timed image hashes; occupancy is a HIP API model and
barrier instruction sites are static, not executed counts.

439 focused compiler/registry and 110 adjacent MX/ISA/fragment/Target tests passed. The first owning-device run
passed 341 and failed four stale fixture geometry assertions; those expectations
were corrected without relaxing numerical or bitwise register-route comparisons.
The final owning-device gate passed 345, zero skips, including FP8, MXFP8 and
exact/folded MXFP4 controls. Compiler mtime warnings are retained; donor and
execution hashes match and source-tree guards pass. The compiler source was
checkpoint 2e9ad4dd66; fixture repair 508c8dad65 has identical `src` tree.

`automatic-route-ab.json` records ten profiles with all 50 timing arms admissible.
The six selected profiles improve matched BF16 device medians 1.056-3.694x.
Examples (microseconds, same-process interleaved windows):

| M,N,K | Baseline BF16 | New automatic BF16 | Baseline/new | New/AITER time |
|---|---:|---:|---:|---:|
|65,1024,1024|30.38|9.54|3.18x|1.101|
|95,2048,2048|57.45|15.55|3.69x|0.940|
|97,1024,2048|17.09|14.41|1.19x|1.192|
|127,3072,1536|17.14|16.08|1.07x|0.978|
|127,3072,1280|15.39|14.57|1.06x|1.003|
|127,2048,1536|15.50|13.74|1.13x|1.099|

The four excluded profiles retain byte-identical F32 and BF16 images. Their
small timing differences characterize the same machine code, not a policy gain
or regression. The large-M K1536 control remains about 3.5% slower than AITER;
several selected narrow-N rows also remain slower than AITER. No global closure
or default promotion outside the stated envelope follows.

The selected image has 206 VGPR, 18 KiB static LDS,zero private segment and modeled
three active 128-thread blocks, compared with the M127 control's 192 VGPR, 27 KiB
and two 256-thread blocks. This is a different geometry, not a register-count
reduction claim. Static LDS/WMMA counts and exact image hashes are retained in
`diagnostics/compact/small-grid-isa.json`. Host-call samples remain in each raw
row's `end_to_end`; device improvements are not public-call speedup claims.

## Engineering decisions

- Partial-row batching preserved K order but kept 236 VGPR/36 KiB LDS and had no
  repeatable speedup. Do not promote it.
- A real LDS boundary reduced registers to 209 but increased static LDS read
  instructions from 24 to 40 and regressed timings 27-39%. Do not promote it.
- Unpadded K128 raised modeled block residency 1->2 but ran about 4x slower.
- K64 staging and double buffering also lost in the measured controls.
- Compact XOR LDS improved the unpadded control but used 256 VGPR, 28 bytes private
  storage and 48 single-address LDS reads rather than 24 paired reads; it still
  lost to production. Do not promote it.
- Smaller geometries lost at large M. Four-wave 64x64 won at small ragged M,
  but aligned M96/N2048 and wider-N controls refuted broad promotion. The
  native candidate therefore keeps those controls unchanged.

Raw diagnostic packets, resource metadata and manifests are in diagnostics/.
The three rejected native source experiments are reproducible patches against
main e8a782d under nonwinning/. Their source changes are not in the candidate's
native generator. The final implementation changes the native Schedule rule,
its differential oracle and exact-device fixture expectations.

## Still open

This is a bounded W8A8 schedule improvement, not closure of the original five
compiler slices. Large-M K1536, remaining narrow-N AITER gaps, wider W8A8
coverage, MXFP4 attribution/optimization, broader ROCm routes and the other
named programs retain their own obligations. Source compilation, device proof
and timing claims remain distinct.
