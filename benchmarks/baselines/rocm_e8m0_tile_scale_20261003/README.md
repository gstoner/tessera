# Standard E8M0 Tile scale consumer — gfx1201

Owner ROCM-FP8-BLOCKSCALE-1; sync ROCM-E8M0-TILE-2026-10-03.

This is a native Tile consumer foundation, not a complete MXFP8 frontend package.
The fixture computes a K32 partial with two native FP8 WMMA instructions,
then consumes both scale buffers as raw E8M0 bytes. Code 0 means 2^-127;
code 255 propagates NaN. Neither behavior comes from the folded row-reference
ABI. The scaled partial is evaluated in f64, rounded once to f32, then added
to the f32 accumulator. No host scale expansion occurs.

Twelve RX 9070 XT/gfx1201 numerical cases cover signed and zero partials,
all 256 codes, reciprocal extreme scale pairs, finite subnormal outputs,
infinite outputs and NaN masks. M=255/256 and N=15/16 test ragged/full edges.
The independent oracle uses ml_dtypes E8M0 decoding into f64. numerical.json
records compiler, source and native image fingerprints and per-case counts.

The fixture is authored Tile MLIR with runtime leading dimensions. It is
isolated test infrastructure, not a Python backend or alternate production route.
An initial fixture retained static N=16 address expressions while testing N=8;
that invalid fixture was corrected to runtime-leading-dimension tile views.
Its failed result was not a decoder or production-route regression.

Remaining: Graph/Schedule derivation, named physical contract, package and
runtime ABI admission, nonuniform multi-group accumulation and performance
benchmarks. No canonical MXFP8 dtype or sibling architecture is promoted.
FP8 and MXFP4 results are separate evaluation gates. No timing claim is made.

Validation: 433 tests passed in the combined exact-device/registry run, including
99 existing FP8 W8A8 device cases and the 12 E8M0 cases. Three focused MLIR
FileCheck/verifier commands also passed with the matching compiler build.
