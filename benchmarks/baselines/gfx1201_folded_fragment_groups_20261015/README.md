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


## Final frozen production gate

Frozen 3b2fa3b70146fb04c597116230ded25bf6863321 remains clean and unchanged
through 458 focused regression checks, 44 owning-device numerical checks,
19 selection/image checks with four negative admissions and two reuse classes,
12 FP8/MXFP8 image comparisons, and eight-shape forward/reverse timings.
regular-grid-frozen-validation.json records source/tool hashes, versions,
actual GPU, exact commands and terminal results. Historical receipts above
retain their original scope.

Ratios below are candidate / immutable native baseline for captured device
graph dispatch, normalized when compiler roles reverse. They are not the
recorder's separate native/HIP ratio, pure hardware instruction time, ordinary
host-dispatch windows or completed public launch time. All those raw timing
domains remain separate in the compressed recorder packets.

| M,N,K | Grouping selected | Forward ratio | Reverse ratio |
| --- | --- | ---: | ---: |
| 256,8192,1024 | yes | 0.9493 | 0.9786 |
| 256,12288,1088 | yes | 0.9877 | 1.0047 |
| 768,8192,1536 | yes | 0.9629 | 0.9607 |
| 1024,16384,5120 | yes | 0.9622 | 0.9588 |
| 256,4096,2048 | no; identical image | 0.9865 | 1.0063 |
| 256,8256,1088 | no; identical image | 0.9974 | 1.0056 |
| 256,8704,4096 | no; identical image | 0.9909 | 1.0199 |
| 256,10240,1024 | no; identical image | 1.0112 | 0.9944 |

Three selected rows improve 2.1-5.1 percent in both orders; N12288/K1088 is
within identical-image control variation and has no stable gain claim.
Identical-image controls vary by up to about two percent. The policy is an
empirical choice inside the explicit approximate folded route, not a
per-shape speed guarantee or automatic logical-MXFP4 promotion.

## Timed-image ISA attribution

The exported candidate and baseline images are hash-matched to the actual
timing packet before disassembly. The single WMMA-containing backward branch
region has 95 versus 100 static instructions and 7 versus 11 DS waits.
Both retain 32 WMMAs, 12 LDS loads and two barrier signal/wait pairs in that
region. Both use 177 VGPRs, 29 SGPRs, 25,600 LDS bytes and no spills.
The full symbol retains three barrier pairs (prologue plus loop body).
This is static selected-image attribution, not dynamic weighting, a hardware
counter measurement or a priced barrier/stall explanation.

regular-grid-isa-census.json binds symbol/image/resource evidence; compressed
assembly retains the instruction stream. The raw compiler executable hashes
are recorded separately from common toolchain/dependency fingerprints.

The remaining delivery gates are synchronization with the current coordinating
NVIDIA PR/main, broad regression and publication CI. Wider MXFP4/Radiance
attribution, other ROCm families and sibling exact-device parity remain open.


## Fresh M256 long-K column curve

A further clean-source run on proof commit 64c54aa0e verifies that compiler,
implementation sources and tools match frozen 3b2fa3b70, then measures the
original M256/K5120, N4096/8192/16384 curve in both compiler/shape orders.
The grouped N8192/16384 medians improve 2.58-3.31 percent forward and
2.66-3.27 percent in reverse. The N8192->16384 marginal cost is 16.129 versus
16.817 ns/column forward and 16.040 versus 16.699 ns/column reverse.

The unchanged N4096 control varies by 0.7-3.6 percent in this extra run.
That larger control variation limits confidence in small individual gains;
it is retained rather than represented as code improvement. This is the
current native-only curve, not a fresh Radiance comparison. A captured
independent-reference comparison remains a follow-up. m256-slope-decision.json
normalizes compiler roles and binds the two compressed raw packets.


## Optional independent reference recorder

The existing native-package recorder now accepts --radiance-module and
--radiance-revision. The pinned reference preflight passed independent-oracle
and full native/reference output parity, explicit-stream capture node counts
and all-three-copy poison checks for bracketed/plain graphs. A diagnostic
four-engine CLI smoke passed under the uncommitted recorder; its timings are
not promoted as evidence. Thirty host-free argument/census/documentation
checks and Ruff pass.

For a frozen owning-device run, set the native optimizer/LLVM paths and
RADIANCE_MXFP4_WPERM=1, then invoke the same recorder with:
--hip-graph-windows --radiance-module PATH_TO_INDEPENDENT_MODULE
--radiance-revision DECLARED_SOURCE_REVISION.
The actual module digest and explicit tuning inputs are recorded separately
from that declaration. A reference row has an empty public-API sample list
and null public-API median. The recorder writes each timing domain separately;
use native_over_radiance_graph_ratio for the captured comparison.
