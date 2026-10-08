# Native SM120 cooperative normalization candidate

Owner: W1.1; sibling FRONTEND-IR-MEDIUM-1. Sync: NVIDIA-COOPERATIVE-NORM-2026-10-05.

## Architecture

Typed Graph → content-addressed Schedule norm decision → Tile norm launch → native MLIR LLVM/NVVM → PTX image → checked CUDA pointer/scalar ABI.

The explicit cooperative_128 candidate assigns one 128-thread CTA per row. Threads visit columns with stride 128 and combine partial sums through 512 bytes of shared memory. LayerNorm retains centered variance rather than subtracting squared means. Every lane participates on ragged rows; the row guard is CTA-uniform. A final read-completion barrier protects scratch reuse. RMSNorm has 9 regular block barriers; LayerNorm has 18. No split/async barrier is used.

Schedule/Tile and the native entry symbol encode the decision. Host, resident and timing launchers all use one CTA per row for the candidate. Tile verification requires a string schedule and sm_120 architecture. Packaging detects a stale NVIDIA tool that produces no cooperative barriers. The serial default and pointer/scalar ABI remain unchanged.

## Exact-device evidence

NVIDIA GeForce RTX 5070, GPU-cba12639-821a-7a10-4cd3-f918f9c0a545, 610.88, 12.0

- 98 NVIDIA device tests pass: serial/cooperative fp16/BF16/fp32 norms, short/ragged/long rows, centered variance, constants, policy integrity, foreign-architecture/attribute rejection, ordinary LHS/RHS JIT and portable replay.
- 424 shared tests pass; 17 target/compiler-dependent cases skip.
- 18 gfx1201 native RMSNorm/epilogue regressions pass with the rebuilt shared compiler.
- Apple/x86 cooperative execution is not established; the candidate requires SM120. gfx1151 proof is not inferred.

## Paired measurements

All 36 cases validate both schedules against an independent float64 oracle before timing. Schedule order alternates over five trials. Resident CUDA event windows include dispatch. Checked host wall includes module loading, allocation, uploads, launch, readback and cleanup; it is reported separately.

| Storage | Norm | M/K | Serial event ms | Cooperative event ms | Ratio |
| --- | --- | --- | --- | --- | --- |
| fp16 | rmsnorm | 1/32 | 0.009371 | 0.009367 | 1.00x |
| fp16 | rmsnorm | 17/35 | 0.009944 | 0.009616 | 1.03x |
| fp16 | rmsnorm | 129/257 | 0.045040 | 0.010056 | 4.48x |
| fp16 | rmsnorm | 128/1024 | 0.207666 | 0.008748 | 23.74x |
| fp16 | rmsnorm | 256/4096 | 0.936679 | 0.011715 | 79.96x |
| fp16 | rmsnorm | 2/4097 | 0.176121 | 0.009156 | 19.23x |
| fp16 | layernorm | 1/32 | 0.008959 | 0.008673 | 1.03x |
| fp16 | layernorm | 17/35 | 0.008959 | 0.008739 | 1.03x |
| fp16 | layernorm | 129/257 | 0.048313 | 0.008144 | 5.93x |
| fp16 | layernorm | 128/1024 | 0.255091 | 0.008998 | 28.35x |
| fp16 | layernorm | 256/4096 | 1.124792 | 0.011171 | 100.69x |
| fp16 | layernorm | 2/4097 | 0.195004 | 0.010037 | 19.43x |

The examined launch mapping explains the candidate direction: serial row walking uses strided neighboring-thread accesses and very few CTAs for these row counts; the candidate exposes coalesced columns and many row CTAs. This is source/launch attribution, not a hardware-counter claim. Resource records retain register, static shared-memory and spill counts.

## Scope and remaining obligations

This is an explicit native Schedule candidate, not default-strategy or universal W1.1 closure. Short and long envelopes remain separate. Automatic selection, general/dynamic frontend producer graphs and composed AD require further integration/evidence. FP8, MXFP8 and MXFP4 retain independent arithmetic/quality/performance gates before any final strategy decision. The full five-slice goal remains active.

Four representative arms retain Graph, Schedule, Tile, Target and portable runtime images. timings.json retains both compiler and CUDA bridge fingerprints, source hashes, native image identities, barriers/resources, correctness and event/wall samples. Historic packets retain their original fingerprints.
