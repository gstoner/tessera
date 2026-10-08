# gfx1201 LDS W8A8 runtime-K image reuse

Owner ROCM-FP8-BLOCKSCALE-1 / E2E-REAL-6-ROCM-MATMUL-CACHE.
Sync ROCM-W8A8-LDS-RUNTIME-K-2026-10-02.

## Compiler contract

Native MLIR projects LDS W8A8 K to a checked runtime value alongside M/N,
retaining whole/partial panel classes, scale semantics, staging and raster
keys. The scale-group loop bound and final-prefetch clamp use runtime K.
The positive whole-group ABI contract is exposed to LLVM with an assumption;
without it the fictitious zero-trip loop edge caused 15 spilled VGPRs in
ragged kernels. The corrected implementation passes existing no-spill gates.

Graph -> Schedule -> Tile -> verified ROCm Target -> typed fragments ->
ROCDL/LLVM -> HSACO remains the native route. Each package retains static
shape guards and capacity checks. Runtime K must be positive and divisible
by the scale group. Edge classes and physical schedules remain image keys.
The diagnostic static-image control uses the same native compiler consumer.

## Validation

Tajasaurus, AMD Radeon RX 9070 XT, live gfx1201, matching rebuilt compiler.
253 native/device/package tests passed; six additional final-prefetch tests
passed at stage widths 64/128 and prefetch modes 0/1/2. Five runtime K values
128/384/1536/2048/3072 reuse one image per output contract and match the fp64
oracle and independent static-K controls. Existing ragged no-spill tests pass.
Super-Bear WSL registry/package/cache gates: 441 passed, 22 hardware skips.

## Matched benchmark

Nine paired/interleaved device-clock windows, minimum 8 ms, HIP event and
host-clock cross-checks. Oracle checks precede timing and follow public launches.
Source and compiler fingerprints are recorded; source hashes verified against
the authoritative checkout. All twelve timing arms are admissible.

| M,N,K | Output | First package ms | Repeated package ms | Kernel us | Runtime/static kernel ratio | Public launch ms |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 200,8192,1536 | f32 | 257.984 | 14.091 | 52.928 | 1.00805 | 6.008 |
| 200,8192,1536 | bf16 | 198.230 | 13.829 | 53.532 | 1.00490 | 5.649 |
| 200,8192,2048 | f32 | 41.476 | 13.874 | 68.577 | 1.01475 | 6.550 |
| 200,8192,2048 | bf16 | 41.622 | 14.077 | 69.486 | 1.02589 | 7.290 |
| 200,8192,3072 | f32 | 42.373 | 13.888 | 97.686 | 1.00870 | 6.887 |
| 200,8192,3072 | bf16 | 42.302 | 14.093 | 97.351 | 1.00155 | 7.993 |

First image compilation is cold; other K shapes hit the same image.
Repeated package construction is about 14 ms. Kernel ratios range from
1.00155 to 1.02589: measured runtime-K overhead is 0.16–2.59%, not a kernel
speedup. Public launch includes staging/transfers/completion and excludes
compile; frontend and package costs are recorded separately. No AITER
comparison or performance closure is claimed.

## Remaining scope

Reduce the measured runtime-K overhead; MXFP4/folded image keys and wider W8A8
short/ragged-K performance remain open. This packet does not establish model
quality, NVIDIA producer/attention closure or sibling backend execution.
gfx1151 is RDNA3.5 without FP8 WMMA. Apple/NVIDIA/x86 physical routes are
unchanged; shared runtime checks are restricted to ROCm W8A8.
