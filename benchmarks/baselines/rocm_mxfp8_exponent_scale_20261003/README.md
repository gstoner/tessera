# gfx1201 MXFP8 exponent-scale consumer

Owner ROCM-FP8-BLOCKSCALE-1; sync ROCM-MXFP8-EXPONENT-SCALE-2026-10-03.

## Retained change

The standard E8M0 Tile consumer now composes signed exponents and calls
upstream LLVM ldexp.f32. Finite codes have exponent code-127, including code
zero (-127); the combined exponent is lhs_code + rhs_code - 254. Code 255
explicitly propagates NaN. The result has one f32 rounding before the unchanged
f32 accumulator join, equivalent to evaluating the partial and both decoded
scales in f64. Finite f32 partials have at most 24 significand bits and the
combined exponent range is -254..254; the wide reference cannot lose bits or
overflow/underflow in f64 before the final f32 rounding.

The RDNA4 archive records V_LDEXP_F32. LLVM 23's LoadExpOp is used directly;
LLVM owns instruction selection. This introduces no Tessera operation,
dtype, pass, package ABI, runtime buffer or Python kernel constructor.
The ODS wording now permits an equivalent native implementation and pass
metadata includes its LLVM output. The wide_scale ABI retains its mathematical
wide-range semantics; it does not require a hardware f64 implementation.

## Numerical proof

The diagnostic Tile fixture carries arbitrary f32 vectors through the existing
typed-fragment bridge and scale consumer. It covers every 65,536 E8M0 code pair
for 116 partial bit patterns (explicit f32 boundaries and 96 seeded random
patterns), with +0, -0, one-plus-ulp and negative-max-finite accumulators.
That is 30,408,704 checked results per compiler, including signed zero,
subnormal/normal boundaries, finite overflow, infinities and NaNs. Finite and
infinite results compare by raw f32 bits; NaNs compare by classification.
Both the preserved f64 compiler and the candidate pass the independent
ml_dtypes/f64 oracle. This is generic Tile consumer proof, not a new Graph
producer. The separate native MXFP8 Graph/package cases preserve their route.

The owning gfx1201 regression suite includes checked MXFP8 packages, raw Tile
and Graph cases, FP8 W8A8, folded MXFP4, image identity, diagnostic/pass metadata
and op/dtype drift gates. The final run passed 575 tests (final-tests.txt); validation.json binds the final source and compiler fingerprints.
The E8M0 FileCheck requires LLVM ldexp and forbids f64 in that consumer.
The refreshed NVIDIA compiler passed 86 existing matmul/attention device rows;
this establishes existing-route parity, not NVIDIA MXFP8 execution.

## A/B evidence

The original f64 compiler was copied before editing its lowering. Reference
packets record its binary hash and saved source override; its stale-source
warning is expected and retained. The small exponent_scale.patch reconstructs
the original consumer from the active source. Reference and candidate have
identical Tile/Target IR digests, ABI, grid and workgroup. FP8 image payloads
are byte-identical across both compilers.

Two independent process pairs reverse reference/candidate order. Every arm
passes bitwise BF16 validation before timing and after graph replay. Five
alternating windows per arm are at least 20 ms; compiler-owned device clock
markers and HIP-event witnesses must agree within 5%. Device times include
GPU graph dispatch. Checked runtime end-to-end staging/transfers/completion
is recorded separately; compilation and graph construction are excluded.

| M/N/K | B storage | f64 reference (us) | ldexp candidate (us) | Candidate/reference | Repeat |
| --- | --- | ---: | ---: | ---: | ---: |
| 17/19/64 | kn | 10.339 | 9.658 | 0.9341 | 0.9327 |
| 17/19/64 | nk | 8.228 | 7.469 | 0.9077 | 0.9094 |
| 200/256/128 | kn | 18.420 | 14.731 | 0.7997 | 0.8550 |
| 200/256/128 | nk | 10.803 | 8.447 | 0.7819 | 0.8031 |
| 256/512/1024 | kn | 71.654 | 20.714 | 0.2891 | 0.2879 |
| 256/512/1024 | nk | 67.250 | 14.551 | 0.2164 | 0.2182 |

For the NK long-K image, 34 f32-to-f64 conversions, 32 f64 multiplies and
16 f64-to-f32 conversions are replaced by 16 native v_ldexp_f32 instructions.
These are static ISA counts, not dynamic phase timings. The long-K seed
improves approximately 3.5-4.6x and the tested short-K rows improve too.
Source, compiler, native payload, descriptor and ISA hashes are recorded.

## Remaining evaluation

This retains a numerically equivalent scale consumer. It selects no new
physical schedule, persistent strategy or global frontend dtype capability.
Broader short/long shapes and a matched MXFP4 comparison under its distinct
quantization/folded contract remain necessary. FP8, MXFP8 and MXFP4 are all
required before making those choices. Gfx1151 E8M0 exact-device parity and
native Apple/x86/NVIDIA consumers remain separate obligations.

Reproduce with the matching gfx1201 compiler/environment:

    python benchmarks/rocm/benchmark_gfx1201_mxfp8_package.py --compiler "$TESSERA_OPT" --llvm-bin "$TESSERA_LLVM_BIN" --output PATH/candidate.json

For the original reference, select the preserved compiler with TESSERA_OPT
and pass --reference-source pointing at its saved TileToROCM.cpp.
