---
last_updated: 2026-10-15
---

# Native gfx1201 folded fragment read scheduling

Owner ROCM-MXFP4-W4A8-1 / ROCM-FP8-BLOCKSCALE-1.
Synchronization key GFX1201-FOLDED-FRAGMENT-GROUPS-20261015.

The implementation follows Graph -> Schedule -> Tile -> ROCm Target ->
ROCDL/LLVM -> HSACO and the existing checked package ABI. Python reports the
native physical decision; it does not select or generate the kernel.

The broad N4096 candidate was rejected by repeated regressions. The narrower
N>=8192/K>=1024 full-panel policy retains original images outside its band.
See docs/spec/ir/gfx1201_folded_fragment_schedule.md for the complete contract.

Production validation and source-scoped timing receipts are being assembled.
Exact-device evidence is RX 9070 XT/gfx1201 only. This is an explicit approximate
folded route, not default MXFP4 promotion or closure of the Radiance attribution
gap. Host launch, ordinary device windows and captured graph dispatch are
different measurements; WSL exposes no hardware counters.


## Validation before the frozen timing run

The normal CMake core and ROCm optimizer builds pass with LLVM/MLIR 23.1.1.
Host WSL focused Target/frontend/diagnostic/pass gates pass 378 checks.
On the owning RX 9070 XT/gfx1201, 40 folded numerical device tests pass,
including selected N8192/K1024 under fixed and runtime-K images; 221 adjacent
FP8/MXFP8 device checks pass. The production compiler reproduces all 14
selection/image checks, four invalid-policy rejections and two image-reuse
classes. All ten excluded images are byte-identical to the immutable baseline;
all 14 production images are also identical to the corresponding narrowed
prototype images. Wrong-type attributes are now rejected by ODS before lowering.

The recorded prototype runs retain their original tool/source identity.
The broad candidate regresses N4096/K1024 by 4.35 percent in both orders and
also regresses N4096/K2048 and K3072. The narrowed selected cases improve
1.5-4.1 percent in the first process and 1.9-3.7 percent in reverse order.
The excluded N4096/K2048 image is identical; its reversed timing varies by
1.8 percent, characterizing measurement noise rather than a code regression.

The recorder is benchmarks/rocm/record_gfx1201_folded_native_package.py.
Compare graph_rows engines native_mlir and native_reference_control; the
recorder's printed native/control device ratio is versus HIP, not the native
baseline. Reverse the ratio when compiler roles are reversed. All numerical
checks precede timing and are repeated after poison/capture windows.
No fresh Radiance or AMD AITER comparison is claimed.

A frozen-source production benchmark and the final publication gates remain.
