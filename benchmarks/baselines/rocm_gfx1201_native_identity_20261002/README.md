# GFX1201 native MLIR kernel identity and dynamic/fused image reuse

Owner: E2E-REAL-6-ROCM-MATMUL-CACHE / E2E-REAL-6-ROCM-CACHE-KEYS.
Sync: ROCM-NATIVE-IDENTITY-2026-10-02.

## Compiler and execution evidence

The production driver invokes tessera-rocm-project-kernel-identity after
TileToROCM and compiles the resulting Target IR through the native ROCm
executable pipeline. Python no longer transforms Target IR to derive an image.
The pass retains every module and physical directive attribute. It validates
the closed scaffold envelope before removing host signatures, constants and
allocations. For dynamic matmul it permits only constant-axis tensor.dim reads
of entry arguments. Matmul Schedule ancestry remains bound to the per-package
descriptor and replay check; the image identity excludes that provenance hash.

The generic Target generator previously used a trailing bias, while the
scheduled launcher uses A/B/bias/output/M/N/K. Fused numerical validation
exposed this mismatch. TileToROCM now records portable_abi and output storage
explicitly; native generation consumes that ABI field, and projection preserves
it in the image key. The legacy direct directive keeps its established default
ABI. This is an explicit native contract change, not a Python argument fix.

All four fp16 timing envelopes queried the AMD Radeon RX 9070 XT and verified
live/configured gfx1201. Each envelope has six shapes, one image and symbol,
one cold construction followed by five warm image-cache hits, distinct
shape-bound descriptors, five numerical warmups and eleven checked samples.
FP16 and BF16 exact-device tests additionally cover two capacity envelopes,
bias+ReLU, and smaller ragged dynamic runtime shapes. The negative native
compiler tests cover missing/malformed ancestry, unaudited operations and
multiple directives. Bias, activation, storage, module attributes and physical
tile changes remain separate identities.

[Validation transcript](validation-tests.txt): 338 passed, three owning-gfx1151
cases skipped. All 17 native MLIR projection tests ran. Eight numerical
dynamic/fused cases ran on gfx1201. Host WSL package-driver checks: 45 passed,
four compiler/device-specific skips; operation/attribute/pipeline totality:
84 passed; audit-document lifecycle: 11 passed. Derived documents were regenerated
with the owning script and git diff --check passed. Graphify is unavailable
in this authoritative WSL checkout, so graph freshness is not claimed.
Test-only text projection is a cache-driver double, not compiler
evidence.

## Separate timing domains

| Packet | HIP-event median range (us) | End-to-end median range (ms) | Maximum kernel sample CV |
| --- | --- | --- | --- |
| [dynamic](dynamic.json) | 15.92–24.68 | 2.676–5.837 | 6.83% |
| [dynamic_bias_relu](dynamic_bias_relu.json) | 15.96–24.48 | 2.861–6.769 | 2.16% |
| [static](static.json) | 8.76–24.64 | 2.220–5.831 | 28.49% |
| [static_bias_relu](static_bias_relu.json) | 17.92–25.60 | 2.652–6.553 | 1.29% |

HIP events bracket the kernel on the default stream. End-to-end samples include
the instrumented event synchronization, per-call HIP module loading, allocation,
transfers, descriptor checks and host dispatch. Schedule and package times are
recorded separately. These are diagnostic execution measurements; the rows are
not matched speedup experiments, and selector promotion is false.

All timing packets bind compiler SHA256 and exact current packaging/runtime/
Target-generation source hashes. validation.json binds the raw packets and
test transcript. The benchmark was rerun after synchronizing shared Python
source hashes from the authoritative scratch checkout.

## Remaining obligations

- Exact gfx1151 dynamic/fused reuse and numerical proof require Princess-Luna.
- Per-call module/transfer overhead remains open; image-cache reuse does not
  establish runtime module-cache reuse.
- Split-K, K-unroll greater than one, LDS staging, and wider storage/layout
  schedules retain their separate compilation envelopes.
- Apple, NVIDIA and x86 have different target generation and ABI contracts;
  no gfx1201 physical proof or timings transfer to them.
- General W1.1 producer migration, wider attention envelopes and broader
  NVFP4-ingest quality remain open in the five-slice program.
