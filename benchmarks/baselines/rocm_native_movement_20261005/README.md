# Native ROCm movement host orchestration

Owner E2E-REAL-6; sibling FRONTEND-IR-MEDIUM-1.
Synchronization key ROCM-NATIVE-MOVEMENT-2026-10-05.

## Architecture and lifetime

Python supplies typed Graph inputs and checked package metadata. Existing native
Graph → Schedule → Tile → ROCm Target → LLVM lowering produces the same HSACO
and checked f32/i32 buffer ABI used by all three benchmark arms. The new C++
host service performs HIP image leasing, allocation, transfer and launch;
it contains no GPU kernel or numerical implementation.

The synchronous default-stream service keeps three reusable staging buffers
per device/context/architecture, bounded to 128 MiB retained capacity. Larger
requests are one-shot. Native validation checks byte counts, integer overflow,
host alignment, physical page/token bounds, architecture and launch capacity.
Completion precedes buffer/image release. Failed completion quarantines the
arena and holds its image lease until explicit recovery. Forked children are
refused before HIP calls. Explicit current-context clear is required before
device reset/context destruction; no HIP calls occur at static destruction.

TESSERA_ROCM_NATIVE_MOVEMENT=0 selects the existing checked compiler-image
host launcher. TESSERA_ROCM_MOVEMENT_STAGING_REUSE=0 independently disables
native staging reuse. An explicit missing movement library is rejected.
Other families and explicit-stream launches retain their existing path.

## Exact-device numerical and timing evidence

Princess-Luna: AMD Radeon 8060S, gfx1151, PCI 0000:c5:00.0.
Tajasaurus: AMD Radeon RX 9070 XT, gfx1201, PCI 0000:03:00.0.
All nine rows pass independent NumPy oracles bit-exactly before and after timing.
Paged shapes are [physical_pages,page_size,heads,dim,start,tokens];
MoE shapes are [tokens,slots,hidden].

Nine rotating-order trials use ten checked launches per arm, with policy-switch
warmups outside timing. Image caching is enabled in every arm: warm image
loads/function lookups are zero. Native pooled windows also have zero allocation
or free calls and three buffer reuses per launch. Separate preloaded resident
HIP-event windows measure 64 dispatches per sample and include dispatch effects;
these are not isolated kernel timings or a kernel-speedup claim.

| Architecture | Family | Shape | Python host ms | Native unpooled ms | Native pooled ms | Python / pooled | Resident dispatch ms |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| gfx1151 | paged_kv | [4, 4, 3, 8, 1, 5] | 0.5768 | 0.5187 | 0.5094 | 1.132× | 0.00306 |
| gfx1151 | paged_kv | [32, 16, 4, 64, 3, 31] | 1.3652 | 1.3021 | 1.2957 | 1.054× | 0.00301 |
| gfx1151 | paged_kv | [64, 32, 8, 128, 7, 249] | 3.2772 | 3.1822 | 2.5377 | 1.291× | 0.01099 |
| gfx1151 | moe_dispatch | [7, 9, 13] | 0.5860 | 0.5319 | 0.5246 | 1.117× | 0.00257 |
| gfx1151 | moe_dispatch | [64, 128, 256] | 1.3050 | 1.2428 | 1.2357 | 1.056× | 0.00281 |
| gfx1151 | moe_dispatch | [512, 768, 1024] | 3.5047 | 3.4407 | 2.3230 | 1.509× | 0.01568 |
| gfx1201 | paged_kv | [4, 4, 3, 8, 1, 5] | 0.5769 | 0.5237 | 0.5280 | 1.093× | 0.00418 |
| gfx1201 | paged_kv | [32, 16, 4, 64, 3, 31] | 1.6038 | 1.5709 | 1.5174 | 1.057× | 0.00616 |
| gfx1201 | paged_kv | [64, 32, 8, 128, 7, 249] | 3.4843 | 3.4243 | 2.7344 | 1.274× | 0.02075 |

Host reductions cover the checked compiled package launcher, not the historical
retained HIP production transport. No route-retirement or universal performance
promotion is claimed. Compilation/package wall cost remains separate in JSON.
The generated image is unchanged between host arms.

## Validation and reproducibility

- 88 focused native artifact, runtime ABI, binding, image and movement lifetime
  tests pass on Super-Bear WSL; 75 existing ROCm host tests pass, 47 device skips.
- Eleven focused binding/lifetime and paged-KV device tests pass on gfx1201.
- Seven binding/lifetime tests pass on gfx1151, followed by all six device rows.
- Controlled native HIP tests cover reuse/growth, malformed requests, alignment,
  allocation/copy/free/completion failures, quarantine recovery, context/device
  isolation, parallel serialization and fork refusal.
- Each architecture packet fingerprints current sources, compiler and native
  image/movement libraries and retains Graph/Tile/Target/HSACO snapshots.
  Source hashes match the authoritative checkout.

The first gfx1151 recorder failed because the lean artifact driver omitted
the semantic Graph dialect. A fresh source build with
TESSERA_FORCE_FULL_COMPILER_DRIVER=ON corrected the configuration. The initial
failure and corrected configure/build logs are preserved under gfx1151/.

## Open work

General paged-KV layouts, resident/asynchronous movement APIs, retained-route
performance comparisons and route retirement remain open. The 128 MiB limit
is a staging retention bound, not an application memory budget. Python package
validation and compilation overhead remain follow-up work.

Apple, NVIDIA and x86 physical changes are not applicable: this service uses
HIP-only context/image ownership. No sibling device performance transfers.
Broader frontend/AD, W1.1, quantized formats and five-slice closure remain open.
