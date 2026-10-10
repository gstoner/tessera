# Native short-K dispatch on gfx1201

Owner ROCM-FP8-BLOCKSCALE-1 / E2E-REAL-6.
Sync GFX1201-NATIVE-SHORT-K-2026-10-08.

The C++ WMMA generator specializes the existing projected K128/fp32-scale
register body for K=1024,1536,2048. Uniform runtime dispatch selects a cloned
native MLIR body with constant K; other K values retain the original general
body. This keeps a single image and checked ABI. It does not author a Python
kernel or change numerical policy. Static images, K32 FP8/MXFP8, folded MXFP4
and the LDS-staged body are outside this transformation.

## Source and compiler identity

Candidate compiler SHA256:
736644c0ecc6d36ad3530aa617c8c66a3c2d0401dab5b6a1b674d2e39688dbf0.
Reference:
65126f8c651e5ecc0473239e311e9e1e81a812b5a59b348b0b75d1c9d982e4b2.
The candidate was built on Super-Bear from matching source with LLVM/MLIR
23.1.1 and copied to Tajasarus; binary hashes agree. reference_generator.cpp
preserves the actual earlier generator. Raw packets bind candidate source and
both compiler binaries. The source-copy mtime is newer than the binary mtime,
triggering the preserved freshness warning despite equal transferred binary
hashes. The reference intentionally implements the prior generator.

## Numerical and regression gates

RX 9070 XT, live architecture gfx1201 on Tajasarus WSL.
The new test_register_short_k_dispatch_and_fallback_reuse_image covers
K=768,1024,1536,2048,2304, ragged M=17/N=19, nk/kn orientations and f32/bf16.
Three parameterized tests pass: each five-K family reuses one image and entry,
matches an independent fp64 oracle and its static compiler control exactly.
423 WSL host blockscale/diagnostic/pass registry tests pass.
These gates do not replace full-unit batching/transpose closure.

## Paired timing

Every row validates before timing and after every window. There are 21
alternating candidate/reference windows, target 40 ms, sharing identical
resident pointers. The metric is HIP graph device execution plus dispatch,
not an isolated kernel, host-transfer or end-to-end measurement. No selector
promotion or AITER/Radiance closure is claimed.

From the three-format packet, M200/N1024/K1536 K128 FP8 falls from about
549.9 us to 35.4 us per launch window (paired candidate/reference 0.06439).
M197/N1056/K1536 gives 0.06128; M256/N1024/K1024 gives 0.51328.
The four-format repeat gives 0.51511 for the latter.

K32 FP8, MXFP8 and folded MXFP4 have byte-identical candidate/reference images
in both four-format shapes. Their paired ratios range approximately 0.998
to 1.037, showing identical-image timing variability, not codegen gains.
M256/N8192/K1536 K128 FP8 also has a byte-identical LDS image (ratio 0.99683).

The folded MXFP4 control explicitly permits folding approximation, reports
folding output error separately, and checks the compiler output against the
decoded physical folded operand oracle. It stores expanded E4M3 weight bytes;
it is not proof for every packed MXFP4 route or original checkpoint quality.

## Open

Broader K/layout coverage, unchanged LDS scheduling, MXFP4 M256 Radiance cost
attribution, wider W8A8 performance closure, architecture-specific sibling
regressions, final native PR delivery and the full-unit closure gate remain
open. gfx1151 cannot consume the RDNA4 FP8 WMMA contract. NVIDIA, Apple and
x86 receive no execution or performance claim from this gfx1201 experiment.
