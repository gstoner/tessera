# GFX1201 MXFP8 long-K selection boundary

Owner ROCM-FP8-BLOCKSCALE-1 / E2E-REAL-6. Sync ROCM-MXFP8-LONG-K-2026-10-06.

Live Tajasaurus gfx1201/RX9070XT: 16 shape profiles, six arms each, forward and reverse arm order in fresh processes; 192 arm validations have zero numerical-bound violations. FP8, MXFP8 and explicit approximate MXFP4 controls use matched source operands. MXFP8 K32/K64 share quantized operands and exact K32 E8M0 semantics. Paired input/source/compiler identities and all ISA digests were verified after transfer.

Device windows measure resident graph execution plus dispatch; each packet separately retains allocating end-to-end wall samples. Ratios below compare the forced LDS K64 candidate against the actual automatic MXFP8 route, not against forced LDS K32 alone.

| M,N,K | Forward K64/auto | Reverse K64/auto |
| --- | ---: | ---: |
| [128, 1024, 3072] | 1.079 | 1.091 |
| [256, 1024, 4096] | 0.456 | 0.458 |
| [256, 4096, 3072] | 0.990 | 1.023 |
| [200, 1024, 4096] | 0.243 | 0.238 |
| [256, 512, 3072] | 1.116 | 1.120 |
| [512, 512, 4096] | 0.487 | 0.490 |
| [300, 1024, 3072] | 0.379 | 0.383 |
| [256, 2048, 4096] | 0.788 | 0.689 |
| [128, 1024, 4096] | 1.097 | 1.113 |
| [128, 2048, 3584] | 0.994 | 0.993 |
| [256, 1024, 2560] | 0.858 | 0.883 |
| [256, 1024, 5120] | 0.531 | 0.547 |
| [300, 1024, 3584] | 0.414 | 0.417 |
| [400, 512, 3584] | 0.940 | 0.945 |
| [512, 1024, 2560] | 0.743 | 0.806 |
| [200, 2048, 5120] | 0.784 | 0.860 |

The small grids lose and the wider panel is neutral; only the larger narrow-grid long-K cases support a selector trial. Static ISA records two split-barrier signal/wait pairs per K32 or K64 slab loop body; K64 processes two independently scaled K32 groups per iteration. This is an ISA witness, not a hardware-counter attribution. No native selector/default was changed for these measurements. Native occupancy-aware trial, matching build, numerical replay, automatic-route timing and drift gates remain required.

Owning source/compiler hashes belong to the Tajasaurus scratch checkout and do not identify the evolving NVIDIA aggregate. No sibling physical proof or model-quality acceptance is inferred.

## Native occupancy-aware selector trial

The native MLIR Schedule pass selects LDS K64 for measured transposed-RHS MXFP8 profiles with M=200..512, N=512..2048, K=2560..5120 divisible by 64, at least half a compute-unit grid of 128x64 panels, and fewer than a full compute-unit grid of 128x128 panels. Existing explicit policies and wider-panel choices remain intact. Scale groups remain independent K32 groups; Python validates the resulting Schedule contract and does not choose the physical recipe.

Paired fresh-process replay covers six profiles and seven arms: **84 numerical validations**, zero forward-bound violations, matching paired inputs/source/compiler identities and verified ISA digests. Eligible automatic images match forced K64 images; the small-grid control matches seed and the wide-panel control matches LDS K32.

| M,N,K | Forward auto/seed | Reverse auto/seed |
| --- | ---: | ---: |
| [200, 1024, 4096] | 0.244 | 0.246 |
| [256, 1024, 2560] | 0.860 | 0.869 |
| [256, 1024, 5120] | 0.534 | 0.544 |
| [400, 512, 3584] | 0.950 | 0.956 |
| [128, 1024, 4096] | 0.999 | 0.995 |
| [256, 4096, 3072] | 0.386 | 0.388 |

The wide-panel auto/seed ratio describes the pre-existing LDS K32 route and is not a gain from this change. Device windows include graph execution and GPU dispatch; allocating end-to-end timings are retained separately in both JSON packets. The recorder field selector_promotion=false means the recorder does not promote recipes; this unpublished compiler trial changes automatic selection only within the stated envelope.

Exact gfx1201 replay passes 31 tests, including ragged M/N, K divisible by 64 but not 128, f32/bf16 outputs and independent K32 scale groups. Host WSL native contract gates pass 115 tests; registry/audit gates pass 333 tests. Receipts retain their original owning source/tool identities. General compiler closure, MXFP4 M=256 attribution and wider W8A8 coverage remain open.

The full native envelope fixture file was synchronized to Tajasaurus and passed **43 tests** against the owning compiler after the paired replay. Receipts: native-envelope-contract-tests.log, native-long-k-device-tests.log, wsl-native-contract-tests.log and wsl-registry-audit-tests.log.
