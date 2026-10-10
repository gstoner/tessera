# gfx1201 LDS W8A8 M/N image reuse

Owner ROCM-FP8-BLOCKSCALE-1 / E2E-REAL-6-ROCM-MATMUL-CACHE.
Sync ROCM-W8A8-LDS-MN-IMAGE-CACHE-2026-10-02.

## Native contract and safety

Native MLIR projects a verified LDS W8A8 Target directive to runtime_mn:
m/n=0, positive static k, and explicit whole_m/whole_n panel-divisibility
classes. The typed LDS producer uses runtime M/N for raster geometry and
retains static K staging. Edge classes select the existing masked/unmasked
stores. Scale layout, output storage, panel/wave geometry, pipeline depth,
raster order/group and physical compiler options remain image keys.

Production packaging still follows Graph -> Schedule -> Tile -> checked
ROCm Target -> typed native fragments -> ROCDL/LLVM -> HSACO. Each descriptor
retains its original static shape guards, Schedule provenance and capacity
checks. The runtime additionally validates static K and the LDS edge class
before acquiring/launching the image. A mismatched class is rejected.
The diagnostic project_image_identity=False control compiles the same checked
Target with its static dimensions; it introduces no alternate kernel emitter.

## Exact-device validation

Tajasaurus, AMD Radeon RX 9070 XT, selected architecture gfx1201:
249 native projection/device/package tests passed. For f32 and bf16 output,
(200,8192,1536), (328,8192,1536), (456,10240,1536) reuse one 128x128,
eight-wave image per storage contract with separate shape guards and Schedule
hashes. K, whole-M and whole-N changes cause cold distinct images and still
pass numerical execution. Grouped-M, grouped-N and column-major diagnostic
rasters match the static control and fp64 oracle. Existing register/LDS
coverage passed. Host WSL package/diagnostic/pass tests: 415 passed; shared
cache/lifetime tests: 31 passed, 33 hardware/capability skips.

For the default row-major route, the projected and static HSACO .text sections
are byte-identical for all tested shapes and both outputs. The benchmark
records and checks those hashes independently of image/symbol identity.
This is code-preservation evidence, not a kernel optimization claim.

## Matched measurements

Nine paired/interleaved device-clock windows per arm, at least 8 ms each,
with HIP event and host-clock cross-checks. The same input values, physical
schedule and output storage are held across projected/static arms.
The fp64 oracle is checked before timing and after every public launch.

First and repeated package time excludes frontend Graph/Schedule/Tile
construction, recorded separately. Kernel time excludes host transfers.
Public launch includes staging, transfers, dispatch and completion and
excludes compilation. No timing domains are combined.

| M,N,K | Output | First package ms | Repeated package ms | Kernel us | Projected/static kernel ratio | Public launch ms |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 200,8192,1536 | f32 | 258.625 | 14.370 | 52.596 | 1.00098 | 5.921 |
| 200,8192,1536 | bf16 | 197.388 | 14.289 | 53.329 | 1.00117 | 5.548 |
| 328,8192,1536 | f32 | 42.247 | 13.801 | 75.566 | 0.99994 | 6.575 |
| 328,8192,1536 | bf16 | 42.021 | 13.648 | 76.237 | 1.00110 | 6.110 |
| 456,10240,1536 | f32 | 42.239 | 13.951 | 122.852 | 1.00093 | 8.014 |
| 456,10240,1536 | bf16 | 42.314 | 13.535 | 124.068 | 1.00261 | 7.011 |

The first shape is cold per output contract; subsequent M/N shapes are warm
image hits. Repeated package construction reuses the verified Target product.
Kernel median ratios are within 0.3% of unity, consistent with identical code;
no AITER comparison, isolated kernel speedup or selector promotion is claimed.

## Remaining and sibling scope

K remains an explicit image dependency of LDS scale-group staging. Full
runtime-K LDS reuse, MXFP4/folded scaled image keys and broader W8A8 short/ragged-K
performance obligations remain open. New edge classes correctly use distinct
images rather than pretending different store masks share one physical kernel.

gfx1151 has no RDNA4 FP8 WMMA; no gfx1151 execution claim. Apple, NVIDIA and
x86 have no changed physical consumers or runtime ABI; the new runtime checks
are confined to ROCm W8A8 descriptors. Existing sibling obligations remain open.

Revision-bound static-K evidence above; runtime-K follow-up is recorded in
[the newer packet](../rocm_w8a8_lds_runtime_k_cache_20261002/README.md).
