# gfx1201 W8A8 native Target consumer

Owner ROCM-FP8-BLOCKSCALE-1 / E2E-REAL-6-ROCM-MATMUL-CACHE.
Sync ROCM-W8A8-TARGET-CONSUMER-2026-10-02.

This is revision-bound evidence for the preceding native Target-consumer
snapshot. Its register-image/cache status is superseded by the
[register runtime-shape packet](../rocm_w8a8_register_cache_20261002/README.md);
the recorded compiler/source hashes and measurements remain unchanged.

Production packaging now lowers Graph -> Schedule -> Tile -> checked ROCm
Target IR, then compiles that Target IR to the native image. The C++ consumer
preserves named KN/NK scale contracts, exact-per-block numerical policy,
fp32 scales, fp32/bf16 output, panel and LDS configuration. Conflicting
Target semantics are rejected. No Python kernel constructor was introduced.

## Validation

On Tajasaurus (AMD Radeon RX 9070 XT, live gfx1201), 199 device/package
tests passed. Twelve new numerical cases compare independent Tile-input
images and Target-input images with the fp64 oracle, covering KN/NK,
fp32/bf16 output, ragged M/N and K=1536. Three negative native Target cases
reject conflicting policy, combine order and scale-group size. Host WSL
package/diagnostic/pass gates passed: 415 tests.

The packet records the active device, architecture, compiler/image/source
hashes, seven interleaved kernel windows with HIP-event/host cross-checks,
and seven separate public-runtime launch samples after five warmups.
Correctness precedes timing and is checked after every public launch.
Public-launch time excludes compilation and artifact construction, and
includes staging, transfers, dispatch and completion.

| M,N,K | Kernel median (us) | Public-launch median (ms) |
| --- | ---: | ---: |
| 17,128,256 | 11.877 | 0.948 |
| 200,2048,1536 | 19.394 | 3.796 |
| 200,8192,1024 | 40.854 | 5.972 |

All rows use the production NK/bf16-output arm. No AITER comparison,
speedup claim or selector promotion is made. The large timing-domain gap
is not evidence of kernel regression.

## Remaining work and sibling assessment

Shape-independent W8A8 image keys remain open. The current LDS body uses
static grid and scale-group information; dimensions must not be removed
until native lowering and runtime guards prove safe reuse. This packet is
the operational Target-consumer prerequisite, not cache closure.

gfx1151 (RDNA3.5) does not support FP8 WMMA; no gfx1151 execution claim.
Apple, NVIDIA and x86 have no changed physical consumer or ABI in this
ROCm-specific slice. Their existing obligations remain open.
