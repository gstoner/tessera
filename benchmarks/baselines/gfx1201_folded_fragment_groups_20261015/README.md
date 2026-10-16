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
N>=8192 divisible4096/K>=1024 full-M256 policy retains original images outside its band.
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


## Regular-grid correction

The first production-source candidate 58da1a9fe remained too broad. Its frozen
eight-case run passed 452 focused and 40 device checks, but N8256/K1088 regressed
2.4-4.0 percent. Further independent N/K trials show improvement with odd K64
slab counts at N8192/12288; short-K irregular N widths regress, and even a
long-K N8704/K4096 case regresses. Thus K alignment or a universal long-K rule
does not explain or fix this band. counterexample-decision.json reports native
A/B ratios with compiler roles normalized; compressed raw packets retain all
source/tool/device fingerprints. The retained old compiler is intentionally
immutable; its stale-source warning is expected for this controlled A/B.

One shared native predicate now owns selection and static Target admission:
M256 multiples, N>=8192 divisible4096, K>=1024 divisible64. All other legal
shapes continue through the existing native compiled route. This is empirical
schedule policy for the measured RX9070XT; no cache/occupancy cause is asserted
without counters. Performance on other gfx1201 boards is unmeasured.

The rebuilt regular-grid implementation passes 384 focused registry/contract
checks and 44 numerical device tests, including both N8256/K1088 (fallback)
and N12288/K1088 (grouped). Strengthened IR checks require 32 MMAs in both
images and positive unchanged workgroup barrier counts. Its 19 selection/image
checks preserve all 14 excluded images, including the observed counterexamples.
Twelve FP8/MXFP8 KN/NK controls have identical Tile, Target, native image, entry
and ABI between the final compiler and frozen baseline. The earlier 221-device
format gate retains its original source scope.

Raw historical/probe JSON is stored as deterministic gzip to avoid repeating
large package/resource records in review diffs. compressed-evidence.json binds
both uncompressed and stored hashes. Python gzip.open(path, "rt") reads it.
The final frozen regular-grid timing and publication gates remain open.
