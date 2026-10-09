---
last_updated: 2026-10-08
audit_role: reference
---

# Integrated compiler engineering log



### 2026-10-05 — native normalization accuracy closure

Owner: [W1.1](INTEGRATED_COMPILER_PLAN.md#w11)
PRs: pending; sync NVIDIA-NORM-ACCURACY-2026-10-05.

Outcome: MLIR square-root/division and compensated serial FP32 sums fix
the named BF16 K4096/K8192 numerical gaps without changing tolerance.
A finite guard preserves IEEE non-finite propagation. 136 RTX 5070
device tests pass; 48 current numerical arms and 40 ordinary composed
program cases pass. Frozen original compiler has three failing arms.
Graph/Schedule/Tile and consumer PTX/Target IR are unchanged.
Matched event/wall timing and barrier/register/shared/spill records
remain separate; all four backend queues are assessed.

Remaining: General producer/dynamic/AD integration and wider attention/
ROCm route/performance obligations. Attention JVP still constructs GPU
MLIR in Python and requires native Schedule/Tile migration.
FP8/MXFP8/MXFP4 and sibling IEEE/accuracy gates remain independent.

Evidence: benchmarks/baselines/nvidia_norm_accuracy_20261005/README.md,
accuracy.json, timing.json, program-timing.json, device-tests.txt,
nonfinite-counterexample.txt, ir-attribution.json, packet-integrity.txt.
<!-- entry-fields:end -->


### 2026-10-05 — native SM120 norm selection under evaluation

Owner: [W1.1](INTEGRATED_COMPILER_PLAN.md#w11)
PRs: pending; sync NVIDIA-NORM-NATIVE-SELECTION-2026-10-05.

Outcome: Native Graph-to-Schedule chooses the norm row schedule, Python
reads the serialized policy, and explicit overrides preserve attribution.
126 exact RTX 5070 tests, 424 shared tests (17 skipped), 18 gfx1201
regressions and 36 composed paired benchmark cases pass. At fp16
M128/K1024, RMSNorm producer event dispatch changes 0.207394 -> 0.008716 ms;
LayerNorm changes 0.254577 -> 0.009190 ms. Consumer images are identical.
Checked full wall time does not improve consistently.

Remaining: BF16 K4096 composed oracle tolerance fails for both schedules;
retained stage attribution isolates storage rounding from consumer error.
No accepted timing claim for that envelope and no tolerance relaxation.
General producers/dynamic/AD and FP8/MXFP8/MXFP4 gates remain open.
All four sibling queues assessed; no architecture proof transfer.

Evidence: benchmarks/baselines/nvidia_norm_native_selection_20261005/README.md,
timing.json, bf16-long-attribution.json, benchmark.txt, device-tests.txt,
shared-tests.txt, gfx1201-tests.txt.
<!-- entry-fields:end -->


### 2026-10-05 — native cooperative normalization candidate

Owner: [W1.1](INTEGRATED_COMPILER_PLAN.md#w11)

PRs: pending; sync NVIDIA-COOPERATIVE-NORM-2026-10-05.

Outcome: Explicit content-addressed serial/cooperative_128 Schedule/Tile
decision, native SM120 LLVM/NVVM materialization and matched CUDA launch
geometry. Centered variance and scratch-read completion are preserved.
Native barriers distinguish the cooperative image from stale-tool serial
materialization. Paired fp16/BF16/fp32 event and checked wall measurements
retain correctness, source/image/tool fingerprints and resource counts.

Remaining: Default strategy selection, general/dynamic frontend and
composed AD. FP8/MXFP8/MXFP4 evaluation gates remain independent. Sibling
cooperative schedules require their own physical implementation and proof;
all four queues are assessed.

Evidence: benchmarks/baselines/nvidia_cooperative_norm_20261005/README.md,
timings.json, device-tests.txt, shared-tests.txt, gfx1201-tests.txt.
<!-- entry-fields:end -->


### 2026-10-05 — ordinary NVIDIA LHS producer dispatch

Owner: [W1.1](INTEGRATED_COMPILER_PLAN.md#w11)
PRs: pending; sync NVIDIA-LHS-JIT-PROGRAM-2026-10-05.

Outcome: Complete Graph verification precedes native producer/consumer
Schedule/Tile/Target/PTX packaging. Static fp16/BF16 RMSNorm, LayerNorm and
last-axis softmax execute through ordinary @jit with checked epilogues,
private CUDA lifetime, cache reuse and portable runtime replay. Attribute
ordering is canonical across JSON serialization. Fifty-three exact SM120
tests and 512 shared tests pass (49 compiler/target-dependent skips).
Sixteen role-binding tests pass on the matching gfx1201 compiler.

Remaining: General producer graphs, dynamic frontend, composed AD and
sibling physical routes. Three-format evaluation gates are unchanged.
All four backend queues assessed; ROCm residual and Apple/x86 fused gates
remain explicit. Timing windows include device dispatch and are separate
from full host execution.

Evidence: benchmarks/baselines/nvidia_lhs_jit_program_20261005/README.md,
timings.json, device-tests.txt, shared-tests.txt and rocm-shared-roles.txt.
<!-- entry-fields:end -->



### 2026-10-05 — exact native power-of-two candidate normalization

Owner: [ROCM-NVFP4-INGEST-1](INTEGRATED_COMPILER_PLAN.md#rocm-nvfp4-ingest-1)

PRs: pending; sync ROCM-INGEST-RECIPROCAL-2026-10-05.

Outcome: Native FP64 division by candidate powers of two becomes multiplication
by exact normal reciprocal powers. The nine-candidate search, reduction order
and tie rules are preserved. Forty-seven gfx1201 tests pass, including accepted
exponent limits. Paired Graph packages retain bitwise-identical converted bytes,
exponents, f64 statistics and final outputs. Pinned converter/combined graph
times improve 40.217/40.859 -> 18.521/18.955 ms. Checked full host execution
improves 168.012 -> 145.158 ms. Static native divisions fall 289 -> 1 and scratch
bytes per work-item 920 -> 180. Benchmark order alternates baseline/candidate;
the baseline compiler is deliberately frozen with its source archived.

Remaining: Register spills, general AD/dynamic/layout/model quality and the
broader five-slice objective. Three-format gates remain independent. All four
queues assessed; this ROCm physical leaf changes no sibling IR/ABI/registry.
No dtype/default promotion.

Evidence: benchmarks/baselines/rocm_ingest_reciprocal_20261005/README.md,
paired.json, normalization-counts.json, isa-attribution.json, device-tests.txt,
checkpoint-format-controls.json and shared-tests.txt.
<!-- entry-fields:end -->


### 2026-10-05 — ordinary frontend resident checkpoint program

Owner: [ROCM-NVFP4-INGEST-1](INTEGRATED_COMPILER_PLAN.md#rocm-nvfp4-ingest-1)

PRs: pending; sync ROCM-INGEST-JIT-PROGRAM-2026-10-05.

Outcome: Ordinary @jit captures converter/storage/scaled-matmul SSA without
CPU arithmetic. Catalog-owned result typing, full native Graph verification,
three native Schedule/Tile/Target/LLVM packages and checked common runtime
preserve policy, argument order and private HIP lifetime. Fresh-process
RuntimeArtifact replay runs without the compiler. Thirty gfx1201 tests,
882 shared gates, 76 selected RTX 5070 regressions and exact target
Schedule/Tile/Target checks pass.
Pinned M256 and matched FP8/MXFP8/MXFP4 measurements are recorded separately
from compile, host launch/replay and device graph timing.

Remaining: General frontend/AD, dynamic/layout/image-key and model-quality
envelopes; sibling physical conversion/storage execution. Unsupported semantic
metadata needs an explicit native contract. No dtype/default promotion.
All four queues assessed; the broader five-slice goal remains active.

Evidence: benchmarks/baselines/rocm_ingest_jit_program_20261005/README.md,
jit.json, checkpoint-format-controls.json, device-tests.txt, shared-tests.txt,
nvidia-regression.txt and native-fixture.txt.
<!-- entry-fields:end -->


### 2026-10-05 — portable owned native checkpoint program

Owner: [ROCM-NVFP4-INGEST-1](INTEGRATED_COMPILER_PLAN.md#rocm-nvfp4-ingest-1)

PRs: pending; sync ROCM-INGEST-PORTABLE-2026-10-05.

Outcome: Versioned serialization retains Graph/Schedule/Tile/Target IR, native
images and exact descriptors for converter, lossless storage and packed matmul.
Restore validates stage order, dimensions, semantic policy and image integrity
before GPU allocation. Export/restore copy nested metadata. Fresh-process replay
runs on gfx1201 with tessera-opt unavailable. Twenty-one device tests, 519
shared gates (18 skips), 11 audit checks and 32 generated-document checks pass.
The packet separates restore/validation wall, full replay wall and resident
event/graph windows. Matching-source FP8/MXFP8/MXFP4 controls remain mandatory.

Remaining: Ordinary composed JIT, general frontend/AD integration,
dynamic/layout/image-key envelopes, whole-model quality and sibling physical
execution. Serialization checks integrity, not compiler-origin authentication.
No default or general dtype promotion. All four queues assessed.

Evidence: benchmarks/baselines/rocm_ingest_portable_20261005/README.md,
checkpoint.json, synthetic.json, resident-tests.txt and shared-tests.txt.
<!-- entry-fields:end -->


### 2026-10-05 — resident native checkpoint conversion through packed matmul

Owner: [ROCM-NVFP4-INGEST-1](INTEGRATED_COMPILER_PLAN.md#rocm-nvfp4-ingest-1)

PRs: pending; sync ROCM-INGEST-RESIDENT-2026-10-05.

Outcome: Three native Graph/Schedule/Tile/Target/LLVM packages share one owned
HIP stream and eleven distinct allocations. Converted weights remain resident;
activation updates reuse weights/images. Shape-only consumer packaging issues
no fabricated content/loss certificates. Private lifetime, upload-failure and
completion-failure checks pass. Sixty focused gfx1201 tests and 473 shared
regressions (nine skipped exact-device cases) pass. Pinned gate/up M256/N24576/
K4096 matches independent conversion/storage/arithmetic oracles. Balanced graph
conversion/storage/matmul/combined medians: 40.724/0.340/0.344/41.475 ms.
Matched-source FP8/MXFP8/MXFP4 controls pass; ingested folded output relative
RMS error is 15.177%, so model quality/default acceptance remains open.

Remaining: Ordinary composed JIT, portable program serialization, general
frontend/AD integration, dynamic/layout/image-key envelopes and whole-model
quality. All four queues are assessed; sibling physical execution is unproved.
No format selector or general dtype is promoted.

Evidence: benchmarks/baselines/rocm_ingest_resident_20261005/README.md,
checkpoint.json, synthetic.json, tests.txt and shared-regression.txt.
<!-- entry-fields:end -->



### 2026-10-05 — native lossless MXFP4 storage bridge

Owner: [ROCM-NVFP4-INGEST-1](INTEGRATED_COMPILER_PLAN.md#rocm-nvfp4-ingest-1)

PRs: pending; sync ROCM-INGEST-STORAGE-2026-10-05.

Outcome: Typed two-result Graph traces into hashed Schedule/Tile storage
ownership, ROCm Target integer permutation/unsigned maximum, LLVM and checked
HSACO. Ordinary N16/K64 gfx1201 JIT, common runtime, cached image and portable
replay pass bitwise parity. Matching ROCm/NVIDIA builds pass; 40 gfx1201
focused tests, 692 shared drift tests (one skip), 11 NVIDIA packaging and 39
RTX 5070 device JIT/replay regressions pass. Three separate wall/event rows
pass; event windows include dispatch gaps.

Remaining: Owned resident converter/bridge/packed-consumer handoff and
combined timing; dynamic/layout/AD envelopes and model-quality acceptance.
All four queues are assessed. FP8/MXFP8/MXFP4 remain required independent
gates; no selector/default or sibling execution is promoted.

Evidence: benchmarks/baselines/rocm_ingest_storage_bridge_20261005/README.md,
gfx1201.json, tests.txt, drift.txt, nvidia-regression.txt, nvidia-device.txt.
<!-- entry-fields:end -->


### 2026-10-05 — canonical NVFP4 JIT and common descriptor runtime

Owner: [ROCM-NVFP4-INGEST-1](INTEGRATED_COMPILER_PLAN.md#rocm-nvfp4-ingest-1)

PRs: pending; sync ROCM-INGEST-RUNTIME-2026-10-05.

Outcome: Catalog inference avoids host conversion during tracing; literal
closure attributes and separate multi-result types reach canonical Graph.
Native Schedule/Tile/Target/LLVM owns conversion. Ordinary static gfx1201 JIT,
image-cache reuse, common checked runtime and serialized replay pass numerical
comparison. Exact-architecture manifest registration preserves the canonical
gates. Three synthetic JIT/event benchmark rows pass on RX 9070 XT.

Remaining: Resident converter-to-consumer integration, combined timing, general
packing/dynamic/AD envelopes and model-quality acceptance. All four queues are
assessed. FP8/MXFP8/MXFP4 gates remain required; no default promotion.

Evidence: benchmarks/baselines/rocm_ingest_runtime_20261005/README.md,
gfx1201.json, jit-tests.txt, drift-regression.txt and drift-final.txt.
<!-- entry-fields:end -->



### 2026-10-05 — pinned native checkpoint conversion and format consumers

Owner: [ROCM-NVFP4-INGEST-1](INTEGRATED_COMPILER_PLAN.md#rocm-nvfp4-ingest-1)

PRs: pending; sync ROCM-CHECKPOINT-NATIVE-INGEST-2026-10-05.

Outcome: Catalog-inferred three-result Graph ownership reaches native conversion
on pinned Qwen gate/up weights. Codes/exponents match bitwise and bounded f64
loss checks pass. Eighteen FP8/MXFP8/MXFP4 format/shape arms pass on RX 9070 XT.
The unsigned byte spelling is ui8 while dtype admission remains planned/gated;
sibling capability queries explicitly refuse nonexistent conversion routes.

Remaining: Host readback/upload still bridges converter and consumer. Ordinary
JIT, resident edge, measured combined timing and model-quality acceptance remain
open. No format selector or default is promoted. All four queues are assessed.

Evidence: benchmarks/baselines/rocm_checkpoint_native_ingest_20261005/README.md,
gfx1201.json, benchmark.txt and package.txt.
<!-- entry-fields:end -->



### 2026-10-05 — NVFP4 semantic ownership and checked host package

Owner: [ROCM-NVFP4-INGEST-1](INTEGRATED_COMPILER_PLAN.md#rocm-nvfp4-ingest-1)

PRs: pending; sync ROCM-GRAPH-INGEST-2026-10-05.

Outcome: Three-result Graph semantics lower through a content-addressed Schedule
record, six-memref Tile ownership and native ROCm Target materialization.
The explicit package validates IR/image/descriptor lineage and privately owns
device outputs. Twelve package tests and three synthetic conversion benchmark
rows pass on RX 9070 XT; 322 focused registry gates pass in Super-Bear WSL.
All four backend queues retain architecture-specific follow-ups.

Remaining: Ordinary JIT dispatch, full semantic AD/conformance registration,
pinned checkpoint native conversion-plus-consumer integration and separate
consumer/end-to-end timings remain open. FP8, MXFP8 and MXFP4 remain mandatory
independent correctness/quality/performance gates. No default promotion.

Evidence: benchmarks/baselines/rocm_graph_ingest_20261003/README.md,
package.txt, package-drift.txt and gfx1201-package.json.
<!-- entry-fields:end -->





### 2026-10-01 — canonical SM120 producer route versus legacy Tile patterns

Owner: [W1.1](INTEGRATED_COMPILER_PLAN.md#w11)

PRs: pending; sync NVIDIA-W1.1-CANONICAL-ROUTE-2026-10-01.

Outcome: The registered SM120 pipeline ordering was traced in
src/transforms/lib/Passes.cpp. For sm=120 it runs PMV11 verification,
GraphToSchedule, and ScheduleToTile before generic Tile lowering. The
canonical Graph route therefore replaces Graph matmul/add producers before
the legacy LowerMatmulToTileMMA and LowerKReductionAddToTileMMA patterns can
match. The exact-device RMSNorm/softmax-to-matmul packages prove typed
tile.view/fragment_pack/MMA through NVVM and PTX; this is production-route
evidence.

Remaining: the two generic patterns remain tensor-valued on direct pre-schedule
Tile input and have no bufferization/lifetime conversion. They are a distinct
legacy-path obligation; do not infer its parity from the canonical pipeline
test. No shared IR, ABI, or runtime contract changed, so sibling backends are
not applicable.



Evidence: Registered SM120 pass ordering and exact-device RMSNorm/softmax-to-matmul packets in benchmarks/baselines/nvidia_sm120_w1_typed_tensor_edge_20261001/README.md.
<!-- entry-fields:end -->



### 2026-10-01 — gfx1201 NVFP4 larger-shape exact-device

Owner: [ROCM-NVFP4-INGEST-1](INTEGRATED_COMPILER_PLAN.md#rocm-nvfp4-ingest-1)

PRs: pending; sync ROCM-NVFP4-INGEST-1-SHAPE-2026-10-01.

Outcome: The gfx1201 recorder now generates shape-matched Graph fixture types
and accepts M/N/K parameters. On Tajasaurus RX 9070 XT, synthetic M/N/K=64/96/128
passed Graph -> Schedule -> Tile -> ROCm Target packaging and exact native output
checks, including after every host end-to-end launch. It converted 48 gate and
48 up rows, with a 6x4 grid. The native image digest matched the prior 17x19x64
packet. Relative RMS was 0.0930/0.0717 for the jointly requantized gate/up
weights versus 0.3568/0.4186 for the preserve-code scale baseline. Ingest was
26.50 ms, package construction 240.36 ms, HIP-event kernel median 4.71 us,
and runtime.launch median 2.44 ms; startup outliers make host timing
diagnostic. The old default shape also passed through the new CLI.

Remaining: there is no original BF16 checkpoint on the device host. The synthetic
error comparison does not close model quality, direct BF16-to-MXFP4 comparison,
or model-sized ingestion. No shared Graph, ABI, or runtime contract changed;
Apple, NVIDIA, and x86 are not applicable to this gfx1201 recorder slice.

[Packets and limitations](../../../benchmarks/baselines/rocm_nvfp4_ingest_20261001/README.md).

Evidence: [NVFP4 shape packet and limitations](../../../benchmarks/baselines/rocm_nvfp4_ingest_20261001/README.md).
<!-- entry-fields:end -->



### 2026-10-01 — five compiler-slice rechecks

Owner: [E2E-REAL-6](INTEGRATED_COMPILER_PLAN.md#e2e-real-6)

PRs: pending; sync COMPILER-NEXT-FIVE-2026-10-01.

Outcome: On Super-Bear sm_120, the K=64 W1.1 resident RMSNorm -> matmul case
again passed numerical and typed-fragment checks with a four-step accumulator;
its 21-sample producer/consumer timings were noisy. Saved-LSE forward passed
eight full/causal, fp16/fp32, regular/ragged oracle rows, and the native
checkpoint suite passed saved/recompute gradients. Backward device-event
median was 1.0871 ms at 0.013% CV; host E2E was unstable. On Tajasaurus gfx1201,
NVFP4 ingest passed ten tests and its 17x19x64 package reproduced the same
HSACO and exact outputs (4.31 us kernel median; 2.96 ms ingest; 236.59 ms
package). The bounded matmul cache key passed 31 tests and three shapes reused
one image with errors below 3.6e-7. All new packets are linked from the
backend plans and evidence READMEs. These are exact-device functional and
attribution results; no cross-target schedule reuse or selector promotion is
claimed.

Remaining: the generic W1.1 tensor-valued producer constructors still need actual
bufferization/lifetime and typed loop-carried fragment conversion; saved-LSE
shape coverage and NVFP4 model-scale/source-BF16 validation remain incomplete;
gfx1201 cache reuse is still bounded to the three static register shapes.
No shared Graph, ABI, or target contract changed, so sibling backend outcomes
are not applicable to these evidence-only reruns.


Evidence: Architecture-specific device receipts and numerical comparisons are recorded in the NVIDIA and ROCm backend queues; timing remains diagnostic.
<!-- entry-fields:end -->



Revision-bound provenance, not a sequencing or capability-status authority.
Historical “Next” lists are preserved after routing their obligations to the
[live plan](INTEGRATED_COMPILER_PLAN.md#routing-index). Entries may describe
uncommitted work; a date or test count does not establish merge or promotion.
`last_updated` records the latest append or substantive correction.

For new entries use a date-first heading and the five fields below. Name one
primary Owner; end the five-field block with `<!-- entry-fields:end -->` and
link additional owners in the body. Update that owner record in
the same PR. Evidence corrections are explicit; do not rewrite old results as
current proof. Current priorities live only in the plan.





### 2026-10-01 — gfx1201 NVFP4 projection ingest through scheduled W4A8

Owner: [ROCM-NVFP4-INGEST-1](INTEGRATED_COMPILER_PLAN.md#rocm-nvfp4-ingest-1)

PRs: pending; sync ROCM-NVFP4-INGEST-1-2026-10-01.

Outcome: Added an explicit NVFP4 checkpoint projection converter for packed E2M1
weights with E4M3 K16 scales and per-projection global scales. It preserves the
E2M1 payload, selects E8M0 K32 scales by decoded-weight SSE between the
no-clip exponent and one binade finer, and emits relative RMS/SQNR metadata
for the single declared scale-requantization loss. Gate/up projection metadata
and row boundaries remain separate. A converted synthetic gate/up pair passes
Graph→Schedule→Tile→gfx1201 Target IR packaging and launches the existing
W4A8 WMMA ABI on Tajasaurus.

Remaining: Run the route with a real NVFP4 checkpoint and compare the ingest
against both its source BF16 weights and a direct BF16-to-MXFP4 conversion.
Expand beyond the fixed 17x19x64 diagnostic package before drawing quality or
performance conclusions.

Evidence: Super-Bear WSL passed 9 NVFP4 ingest unit tests and lint. Tajasaurus
was built from this source snapshot with the exact LLVM/MLIR 23.1.1 pin; the
owning-device test passed and checked the result against the decoded ingested
MXFP4 weights. Its diagnostic 17x19x64 packet records 234.2 ms Graph/Schedule/
Tile/target package construction, 4.33 us resident HIP-event median, and
2.48 ms runtime.launch median. First launch samples were warm-up outliers;
the packet is not a throughput claim. The source format to destination
format conversion itself measured 0.368 / 0.422 relative RMS (8.69 / 7.48 dB)
on these synthetic projection scales, confirming that checkpoint quality
needs a real distribution. The fresh Tajasarus recheck reproduces the same image digest at 4.704 us
HIP-event and 2.553 ms end-to-end medians; the latter contains two warm-up
outliers. [Packets and limitations](../../../benchmarks/baselines/rocm_nvfp4_ingest_20261001/README.md)
and [benchmark source](../../../benchmarks/rocm/benchmark_rocm_nvfp4_ingest_schedule.py).
Correction/update (2026-10-01): the initial implementation compared the
no-clip exponent with one binade finer. The converter now evaluates the two
neighboring powers of two around the code-energy-weighted mean scale, which
minimizes the decoded-weight SSE for that K32 block. A skewed-code regression
passes on Super-Bear and Tajasaurus; the exact gfx1201 package retains the same
image digest. On the existing synthetic packet, up-projection relative RMS
moves from 0.42246 to 0.42102 and gate remains 0.36791. This modest result does
not settle model quality. Updated packet:
[weighted-exponent gfx1201 rerun](../../../benchmarks/baselines/rocm_nvfp4_ingest_20261001/gfx1201_nvfp4_ingest_weighted_exponent_20261001.json).
No shared IR, operation, dtype, or runtime ABI was
changed. Cross-backend assessment: NVIDIA has no affected lowering or ABI;
Apple and x86 are not applicable to the ROCm-specific E2M1/E4M3-to-MXFP4
conversion. Exact-device proof is gfx1201 only.
<!-- entry-fields:end -->

### 2026-10-01 — gfx1201 W8A8 M=200 short-K and ragged-K remeasurement

Owner: [E2E-REAL-6](INTEGRATED_COMPILER_PLAN.md#e2e-real-6)

PRs: Uncommitted; original synchronization labels are preserved below.

Outcome: A clean source-matched gfx1201 package run paired production Tessera NK

Remaining: AITER-relative short/ragged-K coverage remains open. The expanded

Evidence: Tajasaurus rebuilt tessera-opt from clean commit

<!-- entry-fields:end -->

Recorded detail (preserved; metadata formatting correction):


Owner: [ROCM-FP8-BLOCKSCALE-1](../backend/rocm/todo.md)

PRs: pending; sync GFX1201-W8A8-M200-RAGGED-2026-10-01.

Outcome: A clean source-matched gfx1201 package run paired production Tessera NK
with AITER on the two previously regressed M=200 shapes and one K=1536 ragged
shape. Both M=200 rows now beat AITER; the K=1536 row remains about 2% slower.

Remaining: AITER-relative short/ragged-K coverage remains open. The expanded
Tessera-only layout sweep below has no AITER arm because the Tajasaurus Python
environment lacks Triton. No selector promotion follows.

Evidence: Tajasaurus rebuilt tessera-opt from clean commit
58b848ccbc7682db03d3b1e350a5421ded56984d. Seven device-clock windows of at
least 6 ms, HIP-event cross-checks, and pre-timing numerical checks passed.
M=200x8192x1024: 38.82 us vs 42.67 us AITER; M=200x2048x2048: 20.46 us vs
23.70 us; 1024x3072x1536: 71.40 us vs 69.97 us. Full per-arm provenance and
errors are in the [paired AITER packet](../../../benchmarks/baselines/gfx1201_w8a8_next_five_20261001/README.md).
A follow-up eight-shape Tessera-only sweep covers M=127/200/255/1024, N=2048–
8192, and K=1024/1536/2048. Both layouts passed the fp64 oracle; NK beat the KN
attribution arm on all rows. This expands route evidence but contains no AITER
arm, so it makes no new AITER-relative claim. No shared Graph, Schedule, ABI,
or runtime contract changed; Apple, NVIDIA, and x86 have no consumer of the
gfx1201-only LDS/WMMA schedule.


### 2026-10-01 — NVIDIA attention saved-LSE forward and backward proof

Owner: [E2E-REAL-6](INTEGRATED_COMPILER_PLAN.md#e2e-real-6)

PRs: Uncommitted; original synchronization labels are preserved below.

Outcome: The sm_120 native checkpoint path verifies saved and recomputed LSE

Remaining: Broaden realistic attention shapes and comparative baselines. W1.1

Evidence: Super-Bear RTX 5070 exact-device checkpoint tests passed 2/2. Focused

<!-- entry-fields:end -->

Recorded detail (preserved; metadata formatting correction):


Owner: [E2E-REAL-6](../backend/nvidia/todo.md)

PRs: pending; sync NVIDIA-ATTENTION-LSE-BACKWARD-2026-10-01.

Outcome: The sm_120 native checkpoint path verifies saved and recomputed LSE
forward outputs and gradients against an independent oracle. A separate
correctness-gated Graph-to-native benchmark covers full and causal attention,
fp16/fp32 storage, and regular/ragged extents.

Remaining: Broaden realistic attention shapes and comparative baselines. W1.1
tensor-valued producer migration remains independently open; this checkpoint
route does not close its two C++ producer sites.

Evidence: Super-Bear RTX 5070 exact-device checkpoint tests passed 2/2. Focused
Schedule, consumer, benchmark, and backward contract tests passed 63 with 6 skips. Eight
benchmark rows passed an independent fp64 reference with maximum absolute
error 5.96e-8. Device medians were 25.21–32.44 us; all met 3% device stability,
while one end-to-end row was 3.12%. The benchmark was generated from commit
58b848ccbc7682db03d3b1e350a5421ded56984d with a dirty benchmark-oracle change.
[Packet](../../../benchmarks/baselines/nvidia_attention_lse_e2e_20261001/README.md).
No shared Graph op, ABI, or runtime contract changed; Apple, ROCm, and x86 have
no consumer of this NVIDIA attention package.


### 2026-10-01 — W1.1 SM120 producer seam rechecked

Owner: [W1.1](INTEGRATED_COMPILER_PLAN.md#w11)

PRs: Uncommitted; original synchronization labels are preserved below.

Outcome: The current producer census still finds two generic tensor-valued

Remaining: Design and implement one bounded matmul producer with explicit

Evidence: Rebuilt scratch tessera-opt on Super-Bear emitted TMA tensor results

<!-- entry-fields:end -->

Recorded detail (preserved; metadata formatting correction):


Owner: [W1.1](../backend/nvidia/todo.md)

PRs: pending; sync NVIDIA-W1.1-PRODUCER-SEAM-2026-10-01.

Outcome: The current producer census still finds two generic tensor-valued
matmul lowering sites: LowerMatmulToTileMMA and
LowerKReductionAddToTileMMA. The current sm_120 typed-fragment lowering
requires pointer-backed tile.view inputs, fragment_pack A/B, fragment_zero C,
and fragment_unpack/store for a materialized output. Its accepted form does
not carry a reduction-loop accumulator. The gap includes storage lifetime and
loop-region signatures; replacing tensor types with fragment types is not a
valid migration.

Remaining: Design and implement one bounded matmul producer with explicit
bufferization/lifetime and loop-carried typed accumulator semantics, then
prove output against an independent oracle on Super-Bear before widening.
This source trace does not count as a producer migration or exact numerical
proof.

Evidence: Rebuilt scratch tessera-opt on Super-Bear emitted TMA tensor results
and a tensor-valued tile.mma for tests/tessera-ir/phase2/full_pipeline.mlir.
The NVIDIA target lowering requires typed fragment_pack inputs and refuses
that tensor-valued form. Direct typed-fragment tests continue to pass, but do
not exercise either generic producer. The relevant source is
src/transforms/lib/TileIRLoweringPass.cpp and
src/compiler/codegen/tessera_gpu_backend_NVIDIA/lib/Conversion/NVIDIALowering.cpp.
No shared IR or ABI was changed; Apple, ROCm, and x86 have no NVIDIA fragment
consumer.


### 2026-10-01 — W1.1 bounded typed SM120 producer-to-matmul edge

Owner: [W1.1](INTEGRATED_COMPILER_PLAN.md#w11)

PRs: Uncommitted; original synchronization labels are preserved below.

Outcome: Static fp16 RMSNorm -> matmul cases at K=16 and K=64 now prove a

Remaining: This is one bounded static edge. The generic

Evidence: Super-Bear RTX 5070 (sm_120a) passed both numerical oracles

<!-- entry-fields:end -->

Recorded detail (preserved; metadata formatting correction):


Owner: [W1.1](../backend/nvidia/todo.md#w11-sm120-typed-producer-edge-2026-10-01)

PRs: pending; sync NVIDIA-W1.1-TYPED-PRODUCER-EDGE-2026-10-01.

Outcome: Static fp16 RMSNorm -> matmul cases at K=16 and K=64 now prove a
real scheduled producer-to-consumer tensor edge on Super-Bear. The K=64 case
carries its typed fragment accumulator through four reduction iterations. The
consumer Tile materializes
pointer-backed views and typed fragments, carries a typed accumulator through
`tile.mma`, then unpacks and stores the result. Target IR and PTX show native
SM120 MMA. The producer's caller-owned intermediate survives through consumer
completion and the resident route reuses its device allocation and stream.

Remaining: This is one bounded static edge. The generic
`LowerMatmulToTileMMA` and `LowerKReductionAddToTileMMA` constructors remain
tensor-valued and open under W1.1; generic bufferization, loop-carried
accumulators in those legacy patterns, and wider storage/layout/shape
envelopes remain unproved. The loop proof here belongs to the scheduled SM120
path only.

Evidence: Super-Bear RTX 5070 (sm_120a) passed both numerical oracles
(max absolute errors 0 and 1.4305e-6). A separate public `from_text` exact-
device test at K=64 requests fp32 output and asserts the fragment and loop-
carried accumulator markers before execution. Exact route assertions check
`tile.view -> tile.fragment_pack -> tile.fragment_zero -> tile.mma ->
tile.fragment_unpack -> tile.store -> nvvm.mma.sync ->`
`mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32`. Three-batch resident CUDA-event timings are recorded for each shape; the
variation and small sample count make them diagnostic only. The packet records source revision
58b848ccbc7682db03d3b1e350a5421ded56984d and a dirty worktree. No selector or
performance promotion. No shared Graph operation or ABI changed; Apple, ROCm,
and x86 have no SM120 typed-fragment consumer.


### 2026-09-28 — gfx1151 paged-KV native Schedule and shape-free image

Owner: [E2E-REAL-6](INTEGRATED_COMPILER_PLAN.md#e2e-real-6)

PRs: pending; sync `E2E-REAL-6-GFX1151-PAGED-2026-09-28`.

Outcome: The admitted bounded physical-page Graph read uses the native paged-KV
Schedule and Tile producer. ROCm packaging replays both boundaries, compiles a
shape-free Target directive, and binds the exported HSACO symbol to the launch
descriptor. Shape-only Graph changes reuse the image.

Remaining: General KV layouts and throughput, gfx1151 `moe_dispatch`, sm_120
paged-KV regression, the other E2E-REAL-6 family migrations and Apple device
execution.

Evidence: Princess-Luna WSL rebuilt `tessera-opt` and passed 13 focused paged
tests, including four exact gfx1151 permuted-page oracle launches, replay
refusal and cross-shape cache reuse. NVIDIA's shared Schedule consumer is
covered by host-free regression; no new sm_120 device claim. The
[host-wall timing packet](../../../benchmarks/baselines/e2e_real6_gfx1151_paged_20260928/README.md)
records 94–103 ms warm packages and 2.16–2.26 ms launches, with no kernel
throughput attribution.
<!-- entry-fields:end -->

### 2026-09-28 — Apple scaled RoPE Graph division and x86 trunc migration

Owner: [E2E-REAL-6](INTEGRATED_COMPILER_PLAN.md#e2e-real-6)

PRs: pending; sync `E2E-REAL-6-APPLE-X86-2026-09-28`.

Outcome: Static f32 `tessera.trunc` now follows replayed x86 Graph→Schedule→Tile
and executes as a native AVX-512 image. Apple `theta / scale` from the shared
`ntk_rope` rewrite lowers to a checked MPSGraph division before the rope call;
the f32 rope compiler call now requires a successful Metal status.

Remaining: Apple `@jit` execution of `target_verify` and `ntk_rope`, the Philox
Langevin Apple Graph lane, exact Mac device proof for the new calls, and the
other E2E-REAL-6 x86/ROCm/NVIDIA families.

Evidence: Princess-Luna WSL rebuilt `tessera-opt`, passed 47 focused x86
tests with three exact-CPU native launches, 21 Apple IR lit fixtures and 233
Apple/pass-metadata checks. The [Zen 5 timing packet](../../../benchmarks/baselines/e2e_real6_trunc_20260928/README.md)
records about 100 ms packaging and 0.72 ms end-to-end native launch medians;
there is no performance promotion and no Apple Metal execution claim.

<!-- entry-fields:end -->

### 2026-09-04 — Engineering follow-through

Owner: [W2.4a](INTEGRATED_COMPILER_PLAN.md#w24a)

PRs: #721, #722, #723

Outcome: Original recorded scope and validation follow below.

Remaining: Historical obligations below route through the Owner; they do not set current priority.

Evidence: Original source, tests and packet references preserved below.

<!-- entry-fields:end -->

### 2026-09-21 — The 90 gfx1201 hardware-dependent skips become one zero-skip owning-device gate

Owner: [COMPILER-DEVEX-1](INTEGRATED_COMPILER_PLAN.md#compiler-devex-1)
PRs: branch `codex/gfx1201-hardware-skip-closure`.
Sync: `GFX1201-SCHEDULED-SKIP-CLOSURE-2026-09-21`.

Outcome: the complete scheduled-package fixture now has a machine-readable
closure contract. Its inventory is 90 exact-device/native-compiler cases and
five adjacent host-contract cases. Tajasarus (RX 9070 XT, `gfx1201`) passes
**95/95 with zero skips** across unary, dense/fused/BF16/FP8/integer/mixed-FP8
matmul, attention forward/backward and ownership, dynamic reuse/retirement,
paged KV, selected panels, K32 int4, transpose loads, and macro-K traversal.

The first green run was rejected because it found a `tessera-opt` older than
172 generator sources. After rebuilding the assertions-enabled LLVM/MLIR
23.1.1 compiler from merged PR #801, the same suite passed again. The recorder
now makes that evidence rule executable: wrong live/compiler architecture,
stale sources, family-count drift, or any failed/error/skipped row aborts the
packet. Ordinary CI retains the skip gates because it cannot inherit an RX
9070 XT result.

Remaining: preserve this gate as the scheduled ABI set evolves. `gfx1200` and
R9700-specific performance still require their own exact-device evidence.

Evidence: [closure packet](../../../benchmarks/baselines/gfx1201_scheduled_closure_20260921/README.md),
`benchmarks/rocm/record_gfx1201_scheduled_closure.py`,
`python/tessera/compiler/rocm_exact_device_proofs.py`, and
`tests/unit/test_rocm_gfx1201_closure_evidence.py`.

<!-- entry-fields:end -->

Merged snapshots: [PR #721](https://github.com/gstoner/tessera/pull/721),
[PR #722](https://github.com/gstoner/tessera/pull/722), and
[PR #723](https://github.com/gstoner/tessera/pull/723).
The following work remains independently gated; publishing the snapshot does
not close native/compiler evidence gaps.

| Owner | Current action and acceptance |
|---|---|
| APPLE-DISPATCH-WEDGE-1 / telemetry | Scoped telemetry capture restores prior process state, including nested/error paths; two device-clock tests adopted it. An opt-in pytest order tracer records transitions after teardown without resetting them. The original process latch and downstream MoE failures still require the triggering order and exact Metal reproduction. |
| MSW-9 | Program-pair native evaluator adapter and separate reference composition/identity law family implemented. Graph IR fragment inventory and the actual fusion consumer remain open. |
| FRONTEND-IR-MEDIUM-1 | PR #721 contains native pre-bucket CSE and narrow source-loop recognition in rank/prune mode. Native instantiation, arbiter consumption, tracer/MLIR raising and attention remain open. |
| APPLE-DISPATCH-WEDGE-1 | Metal 4 direct wait now stamps timeout kind/message. Branch-specific M1 Max E2E re-seals were recorded in commits `3e0ccb21` (#721) and `9773db8f` (#722); this is separate from timeout fault injection and telemetry-order reproduction. Lowp MoE remains opt-in without a measured ledger row; cooperative rewrite remains open. |
| DISPATCH-BREAKER cross-backend | CUDA and HIP waits remain unbounded. A replacement must poison the owning context on timeout and retain every outstanding buffer/module/event; adding a polling deadline followed by current synchronous cleanup would still hang or free live storage. First slices should target one owning bridge each, with injected not-ready/error cases before GPU proof. |
| IKF-1 | Bind `INTRA_KERNEL_FEEDBACK_PLAN.md` here. P0 extends the existing D2 gfx1151 timing probe; P2 waits for green clocks and consumes Schedule-Object region identity. P1 host schema/math is independent. Shared D2 cache insertion/loading and dispatch admission now reject L2/L3 or malformed instrumentation levels. No IR instrumentation is authorized by a scalar clock sample alone. |
| Toolchain evidence | Assertion-enabled LLVM remains an explicit fleet proof gap. Do not interpret release-build negative tests as assertion-enabled contract falsification. The sandbox/Metal mechanism remains unproven; no claim of its root cause is made. |
| Apple cross-run decision | Default remains mean ± t·sd/√n. The eight-process M1 Max cohort and predeclared synthetic 1.5x/3.0x slowdowns produced no policy disagreements across 24 decisions. Retain the mean default; the median remains opt-in. See `benchmarks/baselines/apple7_cross_run_policy_20260904/summary.json`. This comparison installs no ledger and does not establish robustness across workloads or physical stalls. |
| Generated coverage | Decision #26 now assigns coverage evidence to revision-bound CI artifacts: source commit/tree digest, output hashes, run and attempt. The owning generator and semantic tests remain required; missing/expired evidence must be regenerated from the cited revision. |

All numerical/backend promotion remains architecture-owned. The immediate
reproduction priority is telemetry attribution; a scoped cleanup proves a
state leak is fixed but does not establish the root cause of the reported
order-dependent failure.

#### Next acceptance steps after #723

This sequence refines the engineering follow-through above; it does not replace
the broader central queue or reopen its completed rows.

| Order | Owner and next bounded deliverable | Acceptance before expanding scope |
|---:|---|---|
| 1 | APPLE-DISPATCH-WEDGE-1: capture the ordered failing session with `scripts.trace_apple_telemetry`, then minimize the prefix that changes capture state or breaks downstream MoE. | Fresh-runtime reproduction, first transition attributed after teardown, and the minimized order passes after a cause-specific fix. A test passing alone or another scoped state cleanup is not closure. |
| 2 | FRONTEND-IR-MEDIUM-1: instantiate one optimized matmul recipe for two valid shape buckets through the native compiler. `ParametricRecipe.rank_buckets` currently emits only `BucketRank`. | Both instances bind the same recipe/compiler digest and complete shape witnesses; invalid witnesses fail before lowering; execute against the original oracle on the owning backend. Then connect the resulting native artifacts to arbiter admission. Broader attention recognition follows that working path. |
| 3 | MSW-9: enumerate the Graph IR envelope for affine composition and ReLU identity extension, then wire one eligible program pair into a fusion candidate's evaluator gate. | `program_pair_equivalence` must see native provenance on both sides; wrong-shape, unsupported-policy and reference-only cases refuse admission. Reference `ann_identity_checks` is the oracle, not production proof. |
| 4 | DISPATCH-BREAKER: implement one CUDA or HIP bridge's bounded wait with an explicit poisoned-context/resource-retention state. | Inject not-ready, error and timeout before exact-device tests; prove cleanup cannot free live storage or enter another unbounded wait. Migrate sibling bridges only after the owning state machine is sound. |
| 5 | IKF-P1 host schema/math can proceed independently; IKF-P0 continues cross-CU, monotonicity and read-cost checks on gfx1151. | Validate synthetic slot/clock edge cases for P1; retain WSL regression-only limits on P0. P2/P3 instrumentation waits for the specified clock evidence and Schedule-Object identity. |

The Apple policy experiment is complete for its declared cohort: retain the
existing default. Lowp MoE ledger admission/cooperative kernel work and an
assertions-enabled LLVM remain separately owned follow-ups; neither is closed
by a policy comparison or a fleet visibility probe.

### 2026-09-04 — Archive follow-through

Owner: [W4-PRODUCT-1](INTEGRATED_COMPILER_PLAN.md#w4-product-1)

PRs: Not specified in this historical entry; no merge inferred.

Outcome: Original recorded scope and validation follow below.

Remaining: Historical obligations below route through the Owner; they do not set current priority.

Evidence: Original source, tests and packet references preserved below.

<!-- entry-fields:end -->

The superseded code-review snapshot and typing inventory are archived with
[owning-audit summaries](COMPILER_AUDIT.md#archive-reconciliation--2026-09-04).
No new implementation queue is created. Surviving actions remain:

- **W1.1 / foundation F2:** migrate the two tensor-valued MMA constructors in
  `TileIRLoweringPass` and obtain NVIDIA Target/device proof before deleting
  the checked tensor-value lane. The old bare-fragment permissive branch is
  already gone; do not recreate that closed task.
- **DISPATCH-BREAKER / NVIDIA `P3-SOURCE-ONLY-2026-08-30`:** inject allocation
  failure in flash-backward cleanup and prove exactly-once release of acquired
  resources. Recorded normal-path device tests do not establish this case.
- **`P2-REVIEW-SHARED-PASSES-2026-08-29`:** preserve the existing backend
  queue obligations and their exact-host acceptance. Reconcile later receipts
  per row before claiming closure; the P3 closure note is not blanket P2 proof.

Older architecture reviews and overlapping scoped plans remain live during
reconciliation. Foundation F1 (NVIDIA scheduled matmul Graph re-entry) remains
the first code migration; archiving neither replaces nor completes it.

### 2026-09-04 — Foundation F1 implementation

Owner: [W5.5](INTEGRATED_COMPILER_PLAN.md#w55)

PRs: Not specified in this historical entry; no merge inferred.

Outcome: Original recorded scope and validation follow below.

Remaining: Historical obligations below route through the Owner; they do not set current priority.

Evidence: Original source, tests and packet references preserved below.

<!-- entry-fields:end -->

`nvidia_native.package_scheduled_matmul(artifact)` now constructs the descriptor
from the scheduled launch contract and compiles only its Tile IR. The Graph
argument, dynamic Graph clone and discarded base-image build are removed.
Schedule fields and Tile entry ABI must agree with descriptor metadata before
compilation. The driver records the real adjacent scheduled ancestry.

This is the first bounded migration, not closure of E2E-REAL-3 or all native
packaging: F2 still owns the remaining Graph package roots and typed producers.
Validation and exact-device scope are recorded in the NVIDIA queue under
`IR-NATIVE-FOUNDATION-1`. Physical kernel optimization is outside this cut.

### 2026-09-05 — Foundation F2 unary driver migration

Owner: [E2E-REAL-6](INTEGRATED_COMPILER_PLAN.md#e2e-real-6)

PRs: Not specified in this historical entry; no merge inferred.

Outcome: Original recorded scope and validation follow below.

Remaining: Historical obligations below route through the Owner; they do not set current priority.

Evidence: Original source, tests and packet references preserved below.

<!-- entry-fields:end -->

The first E2E-REAL-5 NVIDIA subset now traverses native Graph → Schedule → Tile
before packaging: static f32 last-axis softmax and serial rank-reducing
sum/mean/max on arbitrary axes. The native passes own the launch wrapper;
Python validates/binds the artifact and does not synthesize a replacement body.
The established SM120 approximate-exp policy and runtime scalar ABI are retained.

Exact-device old/new numerical and ABI tests, independent Schedule replay,
policy tampering, and extra-work rejection cover this subset. A discovered
legacy canonical-reduce kind bug is fixed alongside the differential tests.
This closes the default-driver migration for that envelope, not F2: direct
Graph clients, narrow dtypes, min/keepdims/cooperative reductions and other
backends' remaining package families retain explicit retirement obligations.
See all four backend queues under `IR-NATIVE-FOUNDATION-1` for evidence boundaries.

### 2026-09-05 — F2 direct unary entry points

Owner: [E2E-REAL-6](INTEGRATED_COMPILER_PLAN.md#e2e-real-6)

PRs: Not specified in this historical entry; no merge inferred.

Outcome: Original recorded scope and validation follow below.

Remaining: Historical obligations below route through the Owner; they do not set current priority.

Evidence: Original source, tests and packet references preserved below.

<!-- entry-fields:end -->

The migrated NVIDIA static f32 envelope now shares one native scheduling path
between driver and direct package clients. Supported requests require the
native scheduling compiler; they do not silently reconstruct Tile IR when it
is absent. Narrow dtypes and min/keepdims/cooperative reductions remain explicit
unmigrated envelopes, with the old implementation retained privately until their
own comparisons pass. Direct-client migration is closed for f32 softmax and
serial sum/mean/max; full constructor retirement remains open.

### 2026-09-05 — Next ten unary foundation actions

Owner: [W4-PRODUCT-1](INTEGRATED_COMPILER_PLAN.md#w4-product-1)

PRs: Not specified in this historical entry; no merge inferred.

Outcome: Original recorded scope and validation follow below.

Remaining: Historical obligations below route through the Owner; they do not set current priority.

Evidence: Original source, tests and packet references preserved below.

<!-- entry-fields:end -->

These are bounded cuts under F2 / E2E-REAL-5, in dependency order. Completion
requires native replay, correct ABI and owning RTX numerical evidence; constructor
retirement also requires all production callers to migrate.

| Action | Deliverable | State |
|---|---|---|
| F2-U1 | f16 scheduled softmax | implemented; RTX parity passed |
| F2-U2 | bf16 scheduled softmax | implemented; RTX parity passed |
| F2-U3 | scheduled min reduction | implemented; validated |
| F2-U4 | keepdims reduction carrier and scheduling | implemented; validated |
| F2-U5 | f16 input / f32 output reduction | implemented; validated |
| F2-U6 | bf16 input / f32 output reduction | implemented; validated |
| F2-U7 | explicit cooperative_128 reduction scheduling | implemented; validated |
| F2-U8 | retire production Graph softmax constructor | implemented; validated |
| F2-U9 | retire production Graph reduction constructor | implemented; validated |
| F2-U10 | close caller inventory, native replay and negative-policy gates | implemented; validated |

F2-U1–U10 implementation evidence: 184 RTX 5070 execute-and-compare cases
cover f32/f16/bf16 softmax and all 144 combinations of reduction storage, kind,
axis, keepdims and serial/cooperative policy. New packages match the retained
test-only baseline and NumPy. Another 24 cooperative cases cover 257-element
contiguous/strided axes; no performance promotion is inferred. The old
production unary constructors are removed. Native Schedule replay, policy
mutation refusal, missing-compiler refusal, one Tile compilation and explicit
unsupported-adjoint failure are regression gated. This closes the ten bounded
unary actions, not all of F2: norm, attention, packed matmul and sibling direct
package migrations remain open in the survey inventory.

### 2026-09-05 — F2 norm and forward-attention contracts

Owner: [E2E-REAL-6](INTEGRATED_COMPILER_PLAN.md#e2e-real-6)

PRs: Not specified in this historical entry; no merge inferred.

Outcome: Original recorded scope and validation follow below.

Remaining: Historical obligations below route through the Owner; they do not set current priority.

Evidence: Original source, tests and packet references preserved below.

<!-- entry-fields:end -->

Owning item: E2E-REAL-5; synchronization key `IR-NATIVE-FOUNDATION-1`.
Implemented native SM120 unweighted RMSNorm/RMSNormSafe/LayerNorm scheduling and
forward attention packaging for f16/bf16/f32 storage. Native replay validates
the Tile artifact before one target compilation; the driver records real adjacent
Graph/Schedule/Tile lineage. The two Graph constructors are now test-only baselines.

Attention forward hashes now preserve exact f32 policy bits on all architectures.
The initial cut refused NVIDIA short-query causal/window masks; F2-A2 below
removes that restriction after aligning both physical kernels.
No new backward or saved-LSE support is implied. Numerical parity, not performance
promotion, is the acceptance criterion for this migration.

Next actions, in dependency order:

1. F2-A2: align NVIDIA ragged forward/backward masks and audit backward float hashes.
2. F2-A3: native paired saved-LSE migration implemented in the bounded f32 SM120 lane; integrate residual-policy selection separately.
3. F2-P1: signed INT4 migration implemented; extend scale-layout projections for NVFP4/MX.
4. F2-S1: bounded paged read migration implemented; replay-SSM and MoE state/workspace contracts remain.
5. F2-C1: retire sibling direct package constructors using their existing scheduled
   consumers only where storage, policy and ABI envelopes agree.

The updated survey lists every remaining NVIDIA constructor family and retains
all five backend package entry points in the broader census.

### 2026-09-05 — F2 aligned masks and package contracts

Owner: [E2E-REAL-6](INTEGRATED_COMPILER_PLAN.md#e2e-real-6)

PRs: Not specified in this historical entry; no merge inferred.

Outcome: Original recorded scope and validation follow below.

Remaining: Historical obligations below route through the Owner; they do not set current priority.

Evidence: Original source, tests and packet references preserved below.

<!-- entry-fields:end -->

Owning item: E2E-REAL-5; synchronization key `IR-NATIVE-FOUNDATION-1`.

- **F2-A2 implemented:** NVIDIA forward/backward Tile kernels use
  `q + max(Sk - Sq, 0)` for causal and window masks, including saved-LSE variants.
  Short-query native scheduling is admitted. Backward Schedule identity now
  encodes exact f32 scale/softcap/dropout bits; regenerate older serialized
  backward artifacts. This changes no pointer ABI or physical schedule.
- **F2-A3 contract step implemented; native migration open:** explicit paired
  packaging checks shape, physical f32 policy, Q/K/V and saved-LSE bindings before
  compiling either half. Both descriptors carry a checkpoint digest. Unsupported
  score-modifier aliases and reordered gradient returns are refused. The current
  Graph forward op has a single native result; the historical Python saved-LSE
  API has two. Next add a native multi-result checkpoint producer and a consuming
  Schedule contract, then retire both Python Tile constructors together. A digest
  does not establish that a runtime LSE buffer came from the specified inputs.
- **F2-P1 reviewed, native migration open:** packed physical fields now reject
  lossy/noninteger values and negative offsets before target compilation. Next
  start with unscaled signed INT4 A/B/i32 output: make native storage packing,
  packing axis, container strides and output roles part of the Schedule identity;
  replay the serialized artifact and compare odd-K output on SM120. NVFP4/MX
  require scale-layout projections before admission. No packed route was promoted.
- **F2-S1 reviewed, native migration open:** paged-KV bounds no longer coerce
  strings/floats/bools, and unrelated function returns are refused. Shared replay
  geometry/span validation rejects noninteger values and overflowing allocation
  sizes. Next native paged read must retain table bounds and logical start/end;
  replay decode/flush must share an effects/lifetime contract; MoE must retain
  multi-entry capacities and workspace ownership. These are separate bounded
  producer migrations, not another Python wrapper around Graph reconstruction.

Validation includes 18 RTX 5070 attention cases, three forward-save/backward-load
pairs (both rectangular directions and unmasked), and 13 packed/paged/replay
regressions on the rebuilt NVIDIA compiler. This is correctness evidence, not
latency evidence. Princess-Luna runs shared contracts and the existing ROCm
backward lane (three gfx1151 device cases passed). The focused contract, registry
and dtype suite passed 510 tests with seven explicit skips; 11 audit tests, three
native IR fixtures, Ruff, the zero-error mypy ratchet and all 30 generated-doc
gates passed. Apple owning-host evidence is still separate.

### 2026-09-05 — Deleted functionality reassessment

Owner: [W3.3](INTEGRATED_COMPILER_PLAN.md#w33)

PRs: Not specified in this historical entry; no merge inferred.

Outcome: Original recorded scope and validation follow below.

Remaining: Historical obligations below route through the Owner; they do not set current priority.

Evidence: Original source, tests and packet references preserved below.

<!-- entry-fields:end -->

Owner: `IR-NATIVE-FOUNDATION-1`; this section routes reassessment into existing
work items rather than creating a parallel implementation queue. Status: open
architecture review, not authorization to restore deleted implementations or
claim new backend support.

Decisions #29/#31 remove unsupported production declarations and competing
implementations. They do not establish that the underlying semantics are
unhelpful. A bounded experiment may be retained with an owner, producer,
consumer and exit criteria, explicitly outside production capability claims.
The queue stays-deleted gate continues to prevent accidental restoration; change
it only with an accepted replacement contract and executable positive/negative
fixtures. No gate is weakened by this planning update.

| Reassessment | Existing owner and priority | Producer, consumer and native boundary | Acceptance evidence |
|---|---|---|---|
| Queue/pipeline ownership semantics | W2.4a / CAKE / SO-2, with W5.2 scheduling; next bounded spike | One native staged GEMM or attention producer; existing Tile pipeline/token, role and barrier legality consumers; architecture-owned NVVM/ROCDL lowering. First determine whether current operations already express capacity, acquire/publish/consume/release, slot generation and safe buffer reuse. A separate dialect requires a demonstrated representational gap. | Parse/serialize/replay; reject premature reuse, wrong slot/phase, missing release and illegal effects; execute against the current baseline. Measure device latency, barrier count, shared memory, registers and occupancy on NVIDIA and ROCm separately. No gain is presumed from adding vocabulary. |
| Native residual-policy consumption | W5.1 / AD D5 / foundation F3; existing high-priority work | Existing demand/retained-residual analysis and selected SAVE/RECOMPUTE/HYBRID plan feed the C++ region adjoint and native package. Do not re-enable the annotation-only EBM pass as a substitute. | Same forward/gradient results and stochastic identity; effectful replay refused; selected policy matches executed save/replay operations. Measure peak retained memory, complete forward/replay/backward work and device latency for each owning family. |
| Deleted-verifier invariant inventory | W2.4 legality consolidation; bounded correctness audit | Extract useful invariants from deleted ScheduleOps.cpp/Target helpers; map each to current registered ODS/type verification, IRContractLegality or TileDataflowLegality. Canonical producers and their actual lowered output are the inputs. Restore only a missing, still-valid invariant in its current authority. | Disposition for every reviewed invariant: covered, obsolete, intentionally extensible or missing. Missing checks require minimal malformed-IR rejection and valid-production acceptance fixtures. Check compiler cost if an analysis becomes nonlocal; no runtime speedup claim is required. |
| Native sharding / Shardy integration | W5.4 / W5.4-RESHARD-1 / foundation F3; architecture evaluation | Current typed placement lattice, mesh and explicit reshard SSA feed either native Tessera analysis or an evaluated typed Shardy bridge, then existing Schedule/Tile collectives and backend transport. Assess independently of TPU support. | Preserve partial reductions, local shapes, subgroup identity and nested-region constraints through round trips. Reject unknown metadata rather than silently treating it as replicated. Compare against existing mock-mesh semantics; require owning-host multi-rank packets before native transport or performance claims. Account for compile cost, communication bytes and device latency. |
| Neighbors/halo pipeline composition | Existing PDE/distributed work, W5.4 and COMP-SCHED-OVERLAP-1 | Existing halo inference, stencil materialization and transport passes produce native dataflow; overlap and target/runtime consumers must carry true-use dependencies through pack/exchange/unpack/compute. Deleted plugin pipeline names are not the deliverable. | One serialized end-to-end stencil workload, correct boundaries and deterministic exchanges; negative dependency/lifetime tests. Compare sequential and overlapped device/transport timelines on the actual multi-rank backend before claiming hidden communication. |
| StableHLO interoperability | Foundation F1/F3; deferred until a concrete external consumer is named | A bounded canonical Graph export/import boundary, with an identified framework/compiler or differential oracle as consumer. It is not a replacement for the native backend spine and does not reactivate TPU. | Preserve shapes, dtype, numerical policy and effects; explicitly refuse unsupported operations; round-trip and numerical checks with that consumer. Evaluate maintenance/compile cost; no performance benefit is assumed. |

Disposition ledger from the source/history survey:

- Queue MLIR implementation: remains deleted; evaluate ownership semantics above.
- EBM checkpoint pass: remains experimental and registered, removed only from
  the default pipeline. Its useful demand-aware behavior belongs to W5.1.
- Duplicate attention ODS: remains deleted. Canonical `tessera_attn.lse.save`
  and `lse.load` still exist; F2-A3 must evaluate those before defining another
  checkpoint vocabulary. Their existence alone does not close native producer
  or packaging gaps.
- Duplicate Tile/Schedule dialects, private registration scaffolds, permissive
  bare fragments and synchronization attribute escape hatches: keep removed.
  Preserve useful invariants through current typed authorities.
- Old TPP and scaling/resilience directories: implementations remain under
  `src/solvers/tpp` and `src/solvers/scaling_resilience`; these are relocation/
  consolidation cases, not lost capabilities. Likewise, deleting the collective
  generated-type implementation file is not deletion of async collective semantics.
- TPU, Metalium, Cerebras and Rubin CPX retirement remains a target-scope decision.
  Reuse portable ideas only through a named consumer; backend reactivation needs
  its own requirement, implementation owner and exact-device validation.

Backend assessment: NVIDIA and ROCm own separate physical pipeline/transport
experiments. Apple needs its own Metal synchronization and residency mapping;
x86 needs CPU ownership/effect validation, with explicit no-async behavior where
appropriate. Shared IR proofs do not transfer physical schedules or measurements.
All four backend queues link this reassessment; implementation remains ordered
by the existing owners above and the active F2 saved-LSE/packed/stateful sequence.

### 2026-09-05 — F2 unary artifact replay correction

Owner: [E2E-REAL-6](INTEGRATED_COMPILER_PLAN.md#e2e-real-6)

PRs: #726

Outcome: Original recorded scope and validation follow below.

Remaining: Historical obligations below route through the Owner; they do not set current priority.

Evidence: Original source, tests and packet references preserved below.

<!-- entry-fields:end -->

E2E-REAL-5 / `IR-NATIVE-FOUNDATION-1`, review follow-up to PR #726: native
Schedule-to-Tile replay is now mandatory in the shared NVIDIA unary package
consumer, covering softmax, serial/cooperative reductions and norm. Matching
attributes, entry ABI and a retained schedule hash cannot prove that serialized
Tile operands still implement the recorded schedule. Regressions mutate source/
destination operands without changing those markers and require rejection before
target compilation; missing production `tessera-opt` also fails closed. This
corrects the earlier F2-U10 replay-coverage claim rather than opening a new queue.

### 2026-09-05 — Five-action native ownership loop

Owner: [W4-PRODUCT-1](INTEGRATED_COMPILER_PLAN.md#w4-product-1)

PRs: Not specified in this historical entry; no merge inferred.

Outcome: Original recorded scope and validation follow below.

Remaining: Historical obligations below route through the Owner; they do not set current priority.

Evidence: Original source, tests and packet references preserved below.

<!-- entry-fields:end -->

Owner: E2E-REAL-5 / W2.4 / W2.4a; synchronization key `IR-NATIVE-FOUNDATION-1`.
This increment supersedes the F2-A3/P1/S1 migration status above for the bounded
families listed here. It does not close all packed or stateful execution.

1. **F2-A3 implemented:** registered `tessera_attn.checkpoint_forward` yields
   output and row LSE; `checkpoint_backward` consumes dO/Q/K/V/LSE and yields
   dQ/dK/dV. Existing `lse.save/load` only transfer an already computed value, so
   they cannot substitute for the fused multi-result producer. Native
   `schedule.attention_checkpoint` retains shapes, binding roles, exact f32 scale,
   causal alignment and hash. Packaging replays Schedule-to-Tile and never
   reconstructs Graph. The old forward and saved-backward constructors are
   retired from production. Recompute backward remains a separate migration.
2. **W2.4 audit:** the deleted file's useful prefetch memory-space allowlist was
   missing in registered verification; restored with seven valid spaces and
   three invalid-input tests. Dispositions of the reviewed checks are below.
3. **W2.4a spike:** existing pipeline depth/stage/phase, roles, allocation roots
   and completion tokens express the bounded ownership vocabulary. A decorated
   `mbarrier.try_wait` falsely cleared reuse hazards; a native negative fixture
   reproduced it before the fix. Only registered completing operations now
   release that local hazard. No queue dialect is restored. **Still open:** the
   checker clears pending hazards broadly at waits and walks regions in order;
   allocation-specific release, control-flow joins and consumer-completion
   lifetimes need relational analysis. Do not call this full queue ownership.
4. **F2-P1 signed INT4 implemented:** builtin i4 Graph tensors enter the existing
   native matmul Schedule. C++ derives signed low-nibble-first byte packing,
   A-axis 1/B-axis 0, physical shapes, container strides, zero offsets and absent
   scales in typed packed views. Serialized Schedule replay rejects edited Tile
   operands before compilation. The fixed runtime entry ABI is preserved.
   NVFP4/MX scale layouts remain open; no performance promotion occurred.
5. **F2-S1 bounded paged read implemented:** registered physical `tessera.paged_kv_read`
   verifies static f32 pages, i32 table, logical interval and output shape.
   Native Schedule binds read-only borrowing, runtime physical-index validation,
   layout and names; the package consumes the replayed artifact. Runtime still
   validates actual table contents. Stateful mutation, replay decode/flush and
   MoE workspace/lifetime contracts remain separate open cuts.

The canonical driver retains adjacent native Graph/Schedule/Tile lineage for all
three package migrations. Descriptor identity is not proof that an external LSE
buffer contains values from the paired forward input. Python remains the
frontend, descriptor validation and oracle; native C++ owns these Tile programs.

#### Deleted-verifier disposition inventory

History inspected: `e34a59de^` ScheduleOps.cpp and Target/TesseraTargetIR.cpp.
The old files were unbuilt; comments claiming invariants were not enforcement.

| Historical checks | Disposition and current authority |
|---|---|
| Mesh dimensions, axis strings/count; mesh body/symbol; pipeline schedule/microbatches; stage devices | Covered by registered Schedule ODS and ScheduleDialect.cpp. Current checks additionally enforce result/yield roles. |
| Prefetch destination and overlap | Missing destination allowlist restored in ScheduleDialect.cpp; overlap/type preservation already covered. |
| Schedule async-copy spaces differ, nonnegative stage; await arity | Covered by Schedule ODS/verifiers. Historical prose claimed known-space validation but the implementation only checked presence/difference. Extending the space vocabulary is a separate contract decision. |
| Artifact hash/arch; knob name/choices/logit cardinality | Covered by registered Schedule verification; not a second verifier implementation. |
| Cache kv/pt/ring construction and page-lookup arity | Historical cache scaffold has no active native package producer. Its handle API is obsolete for these tensor package boundaries; do not resurrect it solely for arity checks. New paged-read ODS binds its actual tensor interface. |
| Tile alloc-shared memref operand; required async-copy/wait stage and two memrefs | Obsolete historical signature. Native tile.alloc returns an SSA buffer; copy/wait use typed tokens. Optional stage nonnegativity already lives in TileOps.cpp. Requiring historical operands would reject real producers. |
| Mbarrier count/scope, positive arrive bytes, try-wait arity | Active init/arrive/wait ODS and TileOps.cpp cover slots/phase bits/bytes/token shapes; TileDataflowLegality owns pairing and slot agreement. Historical alloc/scope spelling is not the current API. PMV11 compatibility checks must not become a competing native authority. |
| Atomic order/scope and divergent barrier | Old generic spellings retain PMV11 compatibility checks; current target effects and relational WarpSpec legality own executable semantics. No new atomic support implied. |
| Reduction op/order strings and unknown-op warning | Obsolete generic reduction helper. Registered reduction kernels have closed operation/storage contracts; the old advisory unknown-op warning is not a useful correctness gate. |
| Target warp_config positive count/power-of-two warning; smem_layout attr presence | Obsolete standalone target helper spellings. Current typed schedule, layout and backend launch/resource contracts are authoritative; a power-of-two warning cannot replace architecture limits. |
| Target TMA cp_size positive | Historical descriptor signature obsolete; current descriptor expected-byte and arrival agreement is checked by Tile types/ops and TileDataflowLegality. The old comment about global source memory was never enforced there. |

Validation in this loop: RTX 5070 passed nine native package device cases
(three checkpoint pairs, two INT4 cases and four paged intervals/invalid tables).
Princess-Luna gfx1151 passed 39 existing WMMA compiler/runtime tests, including
native SSA/LDS structure. These are correctness checks. There is no new
cross-backend latency/occupancy comparison, and the ownership spike's performance
exit criterion remains open. Apple/x86 physical migrations and owning-host proof
remain separate. Focused registry, native fixture, lint and generated-doc results
are recorded with the implementation rather than inferred from these device runs.

Final focused validation passed **438 tests**, including registry, dtype, artifact
replay, driver lineage and audit lifecycle checks. The separate canonical/consumer
cohort passed 90 tests with 12 explicit host/tool skips. Both native dataflow/reuse
fixtures passed, including strict registered parsing for the changed reuse
fixture. Ruff and the zero-error mypy ratchet passed. The ROCm SSA/LDS compiler
probe preserved allocation/token ownership through target lowering at one, two
and four stages; it measures host compiler cost only, not device latency.

### 2026-09-05 — Allocation lifetime analysis and device regression comparison

Owner: [W2.4a](INTEGRATED_COMPILER_PLAN.md#w24a)

PRs: Not specified in this historical entry; no merge inferred.

Outcome: Original recorded scope and validation follow below.

Remaining: Historical obligations below route through the Owner; they do not set current priority.

Evidence: Original source, tests and packet references preserved below.

<!-- entry-fields:end -->

Owner W2.4a / CAKE / SO-2; sync `IR-NATIVE-FOUNDATION-1`.
This increment supersedes the blanket-release and preorder-walk limitations in
the five-action loop for the bounded registered `!tile.buffer` vocabulary.

- Track all pending storage footprints and async accesses per allocation. A
  typed wait retires only its direct SSA copy producer; a keyed legacy wait
  retires matching copies. An arrival-only mbarrier wait does not prove buffer
  completion. Preserve the declared keyless legacy wait-all envelope.
- Thread barriers retire synchronous write hazards, not outstanding DMA.
  Check explicit copy destinations, source borrows, premature deallocation,
  use-after-free, and overlapping writes, including earlier disjoint writes.
- Merge `scf.if` branches and CFG predecessors by union of may-live facts.
  Iterate `scf.for` and CFG backedges to a finite fixed point; statically empty
  and single-trip loops do not invent backedge accesses. Follow supported
  allocation aliases across structured results and loop carries without
  recursive cycles. Preserve signed-stride footprint bounds.
- Unknown allocation origins, unsupported buffer-access operations and region
  forms fail closed. Loop-carried async tokens cannot release by static producer
  identity alone: dynamic slot/generation correlation remains open. Function/CFG
  buffer arguments need an alias/ownership contract before admission. This is
  not a universal interprocedural lifetime proof or a new queue dialect.

`benchmarks/record_allocation_lifetime_comparison.py` compares baseline and
candidate compilers on each owning GPU, preserves timing domains, validates
outputs and requires identical native images. It measures existing native attention/GEMM
routes, not a new queue schedule. This verifier does not optimize physical
schedules; selector promotion and occupancy gains are not implied. The older
memref reuse-group planner and backend consumption of its allocation groups
remain a separate integration task.

ROCm exact-host results remain in `benchmarks/baselines/allocation_lifetime_rocm.json`.
The NVIDIA allocation-lifetime comparison is withdrawn: the harness set the core
compiler variable while native lowering consumed a different compiler variable.
Its recorded binary hashes did not identify the native lowerers actually used.
The NVIDIA JSON is now an explicit withdrawal record; new measurements are required.

| Host / workload | Before → after median | Domain / proof |
|---|---|---|
| Princess-Luna gfx1151, 512³ f16 direct GEMM | 0.02278 → 0.02217 ms | Synchronized host wall time, not HIP events; relative error 2.28e-6. |
| Princess-Luna, same GEMM via Tile route | 0.02265 → 0.02234 ms | Same clock/oracle and identical HSACO. |

The ROCm comparison is regression evidence for existing executable paths, not
device proof of a new queue protocol. No CUDA timing or resource comparison from
the withdrawn packets is retained as compiler-change evidence.

Validation: 161 native fixtures passed (four feature-gated exclusions), including
real WarpSpecialization buffer markers and 22 positive/negative lifetime cases.
The inherited-access rule prevents role-local waits from clearing other roles.

### 2026-09-05 — Dynamic completion generations and memref arena proof

Owner: [W2.4a](INTEGRATED_COMPILER_PLAN.md#w24a)

PRs: Not specified in this historical entry; no merge inferred.

Outcome: Original recorded scope and validation follow below.

Remaining: Historical obligations below route through the Owner; they do not set current priority.

Evidence: Original source, tests and packet references preserved below.

<!-- entry-fields:end -->

Owner W2.4a / CAKE / SO-2; sync `IR-NATIVE-FOUNDATION-1`; **landing**.
This section supersedes the preceding increment's open direct-token, CFG-alias
and memref-planner boundaries within the following native envelope.

1. **Dynamic completion:** pending accesses carry SSA completion names. Structured
   branch results, loop initial/carry/backedge/exit edges and CFG forwarding rename
   those names. Backedges discard expired local names before a new dynamic copy
   executes; an unforwarded outstanding generation remains pending. Static
   producer reachability never releases an access. `tile.wait_async` and
   `tile.dealloc` share a declared-origin resolver with the lifetime verifier.
2. **Alias contracts:** allocation roots flow through supported structured and CFG
   identity edges. The memref planner follows cast and view interfaces transitively,
   retaining the entire backing allocation for unknown/subview offsets. Unknown
   users and non-linear control flow prevent reuse. Unannotated external Tile
   buffer ownership remains unproven; this does not add interprocedural noalias.
3. **Physical consumer:** `TileBufferReusePass` and `TileBufferArenaPass` use one
   native memref lifetime service. Missing/wrong waits retain in-flight copies
   through exit; stage and barrier keys both match. Synchronous loads/stores
   require a thread rendezvous before storage reassignment. The arena consumer
   rejects forged overlapping groups and pre-existing/escaping/nonidentity aliases,
   then propagates workgroup address space through supported view/cast chains.
   Static byte-size and arena-offset arithmetic are checked before materialization.

Native fixtures cover a legal rolling token pipeline, dropped generations,
branch-result and CFG token forwarding, opaque origins, transitive alias uses,
missing/wrong/conditional waits, physical arena alias types and forged reuse.
The real arena consumer is exercised; parser acceptance alone is not the evidence.

**Remaining:** path-sensitive memref group coalescing across branches/loops;
interprocedural ownership/escape summaries; offset-disjoint subview optimization;
physical ring slot/phase protocols and architecture-owned performance promotion.
The existing native GEMM/attention comparison remains a regression workload, not
measurement of a new ring schedule. Each backend's queue records its own device
boundary and follow-up; no cross-architecture performance transfer.


Validation for this increment: **167 native fixtures passed** (four feature-gated
exclusions), **252 focused unit/registry/audit tests passed**, Ruff passed and
mypy remained at zero errors. The ROCm owning-host packet
`benchmarks/baselines/token_memref_rocm.json` remains valid.
`benchmarks/baselines/token_memref_nvidia.json` is withdrawn for the same native
compiler-selection defect as the earlier NVIDIA comparison. No CUDA before/after
image or timing claim survives from those two records. The separate synchronous
ring and asynchronous GEMM experiments below are unaffected.

### 2026-09-05 — Structured-path coalescing, private borrowing and device ring experiment

Owner: [W2.4a](INTEGRATED_COMPILER_PLAN.md#w24a)

PRs: Not specified in this historical entry; no merge inferred.

Outcome: Original recorded scope and validation follow below.

Remaining: Historical obligations below route through the Owner; they do not set current priority.

Evidence: Original source, tests and packet references preserved below.

<!-- entry-fields:end -->

Owner W2.4a / CAKE / SO-2; sync `IR-NATIVE-FOUNDATION-1`; **landing**.

**Native compiler increment.** The shared memref lifetime service now proves
completion on every `scf.if` path, coalesces opposite arms only with a derived
workgroup-uniform predicate, and permits loop-local reuse only when accesses
complete before the backedge. Uniformity follows registered constants, GPU
block/grid dimensions and deterministic arithmetic; thread IDs and opaque
arguments do not establish it. The planner checks every member of a reuse group,
and the physical arena consumer independently uses the same proof.

Private direct calls can borrow memrefs when their bodies and transitive callees
prove no escape, free, returned alias or outstanding asynchronous access. External
symbols, recursion and opaque consumers remain unproven. The arena admits these
calls only with an existing workgroup-space ABI; it does not change a helper's
signature behind other callers. Native positive and negative fixtures exercise
both planning and physical `memref.view` materialization.

**Measured ring protocol.** `benchmarks/record_device_ring_protocol.py` emits a
native MLIR GPU kernel and lowers it through LLVM to NVPTX or AMDGPU. Direct
streaming and depths 1/2/4 implement identical neighbor-lane math. Shared slots
carry full generation tags, a publish barrier and a release barrier. Fill/drain
and wrap tests cover short and uneven trip counts through 257 rounds; an
intentionally stale generation must fail the output oracle. This is a standalone
cooperative protocol experiment, not a product Tile ring producer or proof of
asynchronous transfer/compute overlap.

Five unprofiled trials, 20 repetitions, 64 workgroups of 128 threads and 257
rounds produced the following medians. The fixed small grid is not an occupancy
saturation study. Each backend was compiled and measured on its own host.

| Backend / clock | Direct | Depth 1 | Depth 2 | Depth 4 |
|---|---:|---:|---:|---:|
| RTX 5070 / CUDA events | 0.037504 ms | 0.048861 ms | 0.049038 ms | 0.049131 ms |
| gfx1151 / HIP events | 0.078530 ms | 0.132742 ms | 0.120325 ms | 0.129554 ms |

Packets: `benchmarks/baselines/ring_protocol_{nvidia,rocm}.json`. Both numerical
and stale-generation negative controls passed. CUDA registers were 26 for direct
and 16 for the rings; API-reported active-block capacity stayed at 12 per SM.
ROCm reported 11/10/13/11 registers and capacity 16 blocks per compute unit.
Shared storage was 0/520/1040/2080 bytes; no local-memory allocation was reported.
API capacity is theoretical, not measured active occupancy.

**CUDA profiler evidence.** Isolated Nsight Compute direct and depth-2 captures
are exported as `ring_protocol_nvidia_ncu_{direct,depth2}.csv` in the same baseline
directory. The ring adds barrier stalls (2.017 stalled-barrier warps per issue
active versus zero); achieved active occupancy stays about 11.1% on this grid.
Profiler replay times are not the unprofiled event medians above. Nsight Systems
exports `ring_protocol_nvidia_nsys_{kernels,api}.csv`: the ring kernel median is
48.048 microseconds across four launches; context creation dominates the short
process API trace and must not be attributed to kernel execution. Reproduce
isolated profiling with `--profile-depth 0` or `2` after the normal run has emitted
its binaries under `--artifacts`; profile-only runs never write timing packets.

**Disposition.** Rings are correct but slower for this streaming experiment on
both devices. Do not promote a selector candidate or restore a queue dialect on
these results. Next: connect a real asynchronous producer/consumer to the proven
release protocol, then measure representative compute intensity and saturated
grids. General CFG coalescing, symbolic kernel-argument uniformity, recursive or
external ownership summaries and private-helper ABI specialization remain open.

### 2026-09-05 — Real asynchronous GEMM comparison

Owner: [TPROF-NATIVE-1](INTEGRATED_COMPILER_PLAN.md#tprof-native-1)

PRs: Not specified in this historical entry; no merge inferred.

Outcome: Original recorded scope and validation follow below.

Remaining: Historical obligations below route through the Owner; they do not set current priority.

Evidence: Original source, tests and packet references preserved below.

<!-- entry-fields:end -->

W2.4a / CAKE / SO-2, `IR-NATIVE-FOUNDATION-1`: the next experiment now uses the
production SM120 macro-CTA F16 GEMM producer. Its next-panel `cp.async` overlaps
current-panel MMA; a benchmark-only immediate-wait control preserves the native
instruction body, two slots and launch geometry. See
[method, reproduction and evidence](../../../benchmarks/nvidia/ASYNC_PRODUCER_CONSUMER.md).

On RTX 5070, unprofiled resident CUDA-event medians improve 6.1% / 2.9% / 2.3%
for 512 / 1024 / 2048 square GEMMs. Six shapes pass the numerical oracle. Both
images have 56 registers/thread, 4 KiB static shared memory and zero spills.
Native SASS confirms MMA before the deferred wait; Nsight Compute reports fewer
long-scoreboard stalls. These are single-process observations, not promotion or
cross-run confidence evidence. Compute Sanitizer cannot initialize the WSL WDDM
debugger interface; synchronization sanitizer proof remains open.

This closes the useful-compute asynchronous benchmark gap on NVIDIA, but does
not connect the generic allocation/release-token pass to the macro kernel.
That integration, sanitizer validation and independent timing repetitions remain
open. ROCm needs a target-specific async producer; Apple and x86 have no physical
schedule parity claim. The previous synchronous ring remains a distinct negative
experiment and must not be relabeled as asynchronous.

### 2026-09-05 — Symbolic kernel uniformity for memref reuse

Owner: [W4-PRODUCT-1](INTEGRATED_COMPILER_PLAN.md#w4-product-1)

PRs: Not specified in this historical entry; no merge inferred.

Outcome: Original recorded scope and validation follow below.

Remaining: Historical obligations below route through the Owner; they do not set current priority.

Evidence: Original source, tests and packet references preserved below.

<!-- entry-fields:end -->

Owner W2.4a / CAKE / SO-2; sync `IR-NATIVE-FOUNDATION-1`; **landing**.
Reuse assignment and static arena materialization now also visit registered
`gpu.func` bodies. Scalar entry arguments of actual GPU kernels provide a launch
uniformity proof; induction values are uniform when all three `scf.for` controls
are uniform. Arithmetic may propagate that proof. Ordinary function arguments,
GPU helper arguments, thread IDs and unproven loop-carried values remain unknown.
A `gpu.kernel` attribute on `func.func` does not override this boundary.

Native `gpu.barrier` releases synchronous memref accesses but never pending DMA.
The static arena consumer creates the shared global inside the owning GPU module
and replaces descriptors with address-space-3 views. The dynamic GPU extension
below supplies the native launch-size consumer; the existing dynamic `func.func`
path keeps its region-local allocation behavior.

The native fixture checks symbolic loops, induction-dependent exclusive arms,
real memref loads/stores and GPU barriers, divergent controls, missing release,
helper and forged-kernel annotations, and dynamic storage. These uniformity
fixtures establish compiler legality/materialization; the device experiment below
separately establishes its bounded dynamic-storage workload.
The remaining symbolic boundary is loop-carried scalar and general CFG uniformity.
Generic release-token integration with the production async macro kernel,
external ownership summaries and architecture-owned execution remain open.

### 2026-09-05 — Native dynamic GPU storage and owning-device proof

Owner: [W2.4a](INTEGRATED_COMPILER_PLAN.md#w24a)

PRs: Not specified in this historical entry; no merge inferred.

Outcome: Original recorded scope and validation follow below.

Remaining: Historical obligations below route through the Owner; they do not set current priority.

Evidence: Original source, tests and packet references preserved below.

<!-- entry-fields:end -->

Owner W2.4a / CAKE / SO-2; sync `IR-NATIVE-FOUNDATION-1`; **landing**.
`TileBufferArena` now materializes entry-block runtime-sized GPU scratch through
`gpu.dynamic_shared_memory`, with workgroup-address-space memref views. Its exact
layout expression also produces a native `func.func` sizing companion, referenced
by `tile.dynamic_shared_size`. Local `gpu.launch_func` sites call that companion
and use its i32 byte count; an explicit count must agree. No Python or generated
CUDA/HIP expression is the authority for those launch sizes.

Admission is deliberately bounded: actual registered GPU kernels, identity-layout
scalar-element memrefs, and a size expression derived from kernel arguments or
their memref dimensions using constants/add/multiply/max/positive-constant divide.
Every host intermediate must lie in `[0, INT32_MAX]`; the host index must be
64-bit so the checked arithmetic itself cannot overflow. The driver registers
upstream DLTI so a supplied data layout is interpreted. Existing independent
dynamic shared allocation, nested markers, GPU helpers and device-local size
provenance fail closed. This does not infer a device's physical shared-memory
capacity; the native runtime still rejects launches exceeding that capacity.

The [recorder](../../../benchmarks/record_dynamic_gpu_storage.py) compiles the
emitted host companion through LLVM into a native shared library, obtains the
launch count from that function, and separately lowers the emitted GPU module
to NVPTX or ROCDL. Negative and oversized extents must abort in fresh native
processes. Both owning-device packets cover four runtime widths, cross-lane
scratch reads, publish/release barriers, 17 iterations and 256 blocks, against
exact output oracles and independently compiled static arenas. Device-event
samples remain experiment evidence, with no selector promotion or speedup claim.

**Follow-on scope (bounded implementation below):** bind the companion and kernel
fingerprints together in a production native package; admit path-dependent/nested dynamic storage only with
a host-evaluable lifetime/size envelope; establish dynamic-size reuse equivalence
before coalescing unknown-size buffers; extend symbolic loop-carried/CFG uniformity;
and integrate release tokens with an actual asynchronous producer. Apple MSL
threadgroup arguments need their own materializer/ABI, and x86 retains its host
allocation route. See the [experiment report](../../../benchmarks/DYNAMIC_GPU_STORAGE.md)
for owning-host evidence and timing limits.

### 2026-09-05 — Bound native packages, nested dynamic lifetimes and NVGPU completion

Owner: [W4-PRODUCT-1](INTEGRATED_COMPILER_PLAN.md#w4-product-1)

PRs: Not specified in this historical entry; no merge inferred.

Outcome: Original recorded scope and validation follow below.

Remaining: Historical obligations below route through the Owner; they do not set current priority.

Evidence: Original source, tests and packet references preserved below.

<!-- entry-fields:end -->

Owner W2.4a / CAKE / SO-2; sync `IR-NATIVE-FOUNDATION-1`; **landing**.
The [native storage package API](../../../python/tessera/compiler/native_gpu_storage.py)
now builds a single immutable pair from compiler-emitted IR. The binding digest
covers the GPU image, host sizing library, entry symbols, argument ABI, target,
compiler fingerprints and arena IR. Serialization requires the caller's pinned
digest on reload; mutations are rejected before native loading. Synchronous
launches use the loaded companion and the same argument tuple as the loaded
kernel. Invalid extents return `-1` from native sizing and raise before dispatch;
`gpu.launch_func` retains a checked assertion for this failure. This supersedes
the earlier companion's process-abort behavior.

The implemented runtime boundary is a raw device-pointer/index ABI on the
caller's current CUDA/HIP context. It does not allocate tensors, infer logical
shape guards, own scheduling, select a route, or regenerate lower-level IR from
Graph. Callers retain responsibility for pointer sizes and kernel launch geometry.
The loader owns target-image compatibility; this first build envelope is sm_120
and gfx1151, with x86 native host companions. It is not automatic JIT/arbiter
integration or a new general tensor operation.

Uniform `scf.if`/`scf.for` regions now admit dynamic markers when every nested
lifetime completes before region exit/backedge. Launch-derived size arithmetic
is hoisted without hoisting payload operations. Disjoint same-type dynamic GPU
buffers can share a group whose capacity is the maximum member size; unrelated
groups retain separate offsets. The envelope reserves even untaken branch sizes
conservatively. Iteration-dependent sizes, divergent control and escaping or
incomplete lifetimes remain rejected.

Native `nvgpu.device_async_copy` tokens now participate in the allocation proof.
Only the copy's direct commit-group token, a full drain (absent/zero numGroups),
and a following GPU barrier establish completion/publication. Missing, partial
and unrelated-group waits fail closed for nested arena reuse. This admits a real
NVGPU producer through NVVM `cp.async` into runtime-sized shared storage; it does
not claim overlap with useful compute or migration of the separate macro-GEMM
schedule. Loop-carried async token forwarding remains outside this direct-group
proof.

Owning-device evidence is in the [package report](../../../benchmarks/NATIVE_GPU_STORAGE_PACKAGE.md):
serialized/reloaded nested packages pass four exact cases on gfx1151 and RTX 5070;
RTX 5070 additionally passes four async-copy cases and native protocol-negative
checks. Native sizing rejects oversized requests before GPU dispatch. No timing
or sanitizer claim is made by this increment.

**Next:** typed tensor/package descriptors and JIT/arbiter consumer wiring;
path-dependent size selection and iteration-varying bounded envelopes; async
token forwarding and integration into the production macro-GEMM schedule; and
ROCm's own physical asynchronous producer. Apple MSL dynamic threadgroup argument
binding remains separate. Keep existing compiler-comparison tombstones withdrawn.

### 2026-09-05 — Explicit tensor/JIT and native producer integration

Owner: [W4-PRODUCT-1](INTEGRATED_COMPILER_PLAN.md#w4-product-1)

PRs: Not specified in this historical entry; no merge inferred.

Outcome: Original recorded scope and validation follow below.

Remaining: Historical obligations below route through the Owner; they do not set current priority.

Evidence: Original source, tests and packet references preserved below.

<!-- entry-fields:end -->

W2.4a / CAKE / SO-2, synchronization `IR-NATIVE-FOUNDATION-1`.
The [native tensor/producer report](../../../benchmarks/NATIVE_TENSOR_PRODUCERS.md)
supersedes the preceding increment's tensor and direct-token limitations:
explicit tensor/index descriptors now bind native storage packages directly to
`JitFn`, preserving frontend checks while bypassing Graph regeneration. Native
allocation extent/device checks precede dispatch; Python equivalence is explicitly
caller-declared. Automatic lowering/arbiter selection and paired AD remain open.

NVGPU completion follows unanimous branch forwarding and identity loop carries;
changing backedges remain rejected by generic lifetime analysis. SM120 production
macro-GEMM emits typed copy/group/wait tokens and lowers through NVGPU-to-NVVM;
its static two-panel ownership is not inferred by the dynamic arena proof.

Exact-device validation: four tensor/JIT cases each on RTX 5070 and gfx1151;
six macro-GEMM shapes in both deferred/immediate-wait variants on RTX 5070.
ROCm uses an ISA-supported register-prefetch producer. Immediate VMEM drains at
several emitted barriers prevent an overlap claim; direct async global-to-LDS
is gfx1250-only. Next: ROCm barrier/scheduling ablation, automatic descriptor and
arbiter production, and changing-generation lifetime proofs. No performance
policy promotion or Apple runtime parity is implied.

### 2026-09-05 — ABI manifests, paired-program consumption and streams

Owner: [W2.4a](INTEGRATED_COMPILER_PLAN.md#w24a)

PRs: Not specified in this historical entry; no merge inferred.

Outcome: Original recorded scope and validation follow below.

Remaining: Historical obligations below route through the Owner; they do not set current priority.

Evidence: Original source, tests and packet references preserved below.

<!-- entry-fields:end -->

W2.4a / CAKE / SO-2; `IR-NATIVE-FOUNDATION-1`.
The [integration matrix](../../../benchmarks/NATIVE_STORAGE_INTEGRATION.md)
records implementation separately from native and device proof.

- Compiler-preserved ABI manifests now generate tensor descriptors and optional
  Tier-2 arbiter candidates. The existing numerical oracle still gates selection;
  general Schedule producers must supply their ABI manifests and operation oracle.
- An existing compiler `NativeJVPArtifact` can bind pinned storage children with
  full ABI preflight. This is a host-tested consumer, not automatic derivative
  production or device-proven AD. The production AD planner remains the next
  producer integration; reverse mode stays unsupported by this adapter.
- Caller-owned CUDA/HIP streams use event dependencies and retained allocation
  owners; four exact generated/JIT/arbiter/stream cases pass on each owning GPU.
  Conflicting submissions through one binding are ordered. External writes still
  require published producer streams; no concurrency performance claim.
- Generic token provenance now handles loop-external generation replacement
  without dropping zero-trip seeds. Fresh loop-issued and rotating generations
  remain rejected; inter-iteration coalescing is not inferred from static origins.
- ROCm has a measured immediate-wait ablation: three exact workloads, seven
  alternating HIP-event samples. The control is 2–4.5% slower in this run, but
  added instructions and scheduling changes prevent an overlap/promotion claim.
- Apple MSL uses a shared slot declaration/preflight contract for its existing
  tiled runtime ABI, including static scratch. Generic dynamic-arena materialization,
  an Apple-native companion and exact Metal validation remain open. x86 continues
  to supply host sizing; GPU results do not establish CPU kernel performance.

**Next producer work:** Schedule-authored manifests and operation-specific arbiter
oracles; storage children from real compiler AD output; rotating-generation
ownership; independent ROCm profiling; Apple generic dynamic-arena materialization.

### 2026-09-06 — Generated AD children, ownership recurrence and Apple arenas

Owner: [W4-PRODUCT-1](INTEGRATED_COMPILER_PLAN.md#w4-product-1)

PRs: Not specified in this historical entry; no merge inferred.

Outcome: Original recorded scope and validation follow below.

Remaining: Historical obligations below route through the Owner; they do not set current priority.

Evidence: Original source, tests and packet references preserved below.

<!-- entry-fields:end -->

W2.4a / CAKE / SO-2; **IR-NATIVE-FOUNDATION-1**. The
[follow-up report](../../../benchmarks/NATIVE_STORAGE_FOLLOWUP.md) records the
bounded implementation and owning-host evidence, superseding the corresponding
blanket gaps in the preceding section.

- The actual C++ forward-AD SSA product can generate a native storage child for
  equal-shape rank-one f32 add/mul programs. Four exact primal/tangent cases pass
  on each GPU owner. General family/Schedule integration and reverse AD remain open.
- A fixed allocation's seed/iteration/final-drain token recurrence proves reuse
  after changing generations. Native adversarial tests reject stale or missing
  completion; five CUDA cases include zero trips and generation-varying input.
  Dynamic slot selection, permuted aliases and general CFG ownership remain open.
- The native arena pass emits bounded MSL with a dynamic threadgroup argument and
  a checked, 16-byte-aligned native sizing companion. Six M1 Max cases pass using
  native arm64 sizing. Production Metal package/JIT binding and broader operations
  remain open; this is a materialization/device probe, not route promotion.
- Five independent ROCm processes now compare an instruction-identical no-wait
  control: 348 instructions, only five wait thresholds differ, matching resources.
  Larger workloads favor deferred waits in 5/5 runs; the small case remains mixed.
  The WSL profiler exposes no PMC metrics/device trace, so stall attribution remains
  open. No selector threshold or incumbent policy changed.

### 2026-09-06 — Expanded AD and dynamic aliases / resident Apple packages

Owner: [W4-PRODUCT-1](INTEGRATED_COMPILER_PLAN.md#w4-product-1)

PRs: Not specified in this historical entry; no merge inferred.

Outcome: Original recorded scope and validation follow below.

Remaining: Historical obligations below route through the Owner; they do not set current priority.

Evidence: Original source, tests and packet references preserved below.

<!-- entry-fields:end -->

W2.4a / CAKE / SO-2; **IR-NATIVE-FOUNDATION-1**. The expanded section of the
[follow-up report](../../../benchmarks/NATIVE_STORAGE_FOLLOWUP.md) supersedes the
corresponding gaps above:

- AD adds subtraction, stop-gradient and compiler-owned splat constants. Forward
  tangent mapping now respects requested `wrt` order. Sixteen exact cases pass on
  each CUDA/ROCm owner. Transcendental, reduction/attention and reverse families
  still require family-owned lowering and validation.
- Dynamic memref aliases support a bijective two-slot swap with full copy
  completion/publication and collective release before every backedge. Seven
  CUDA cases cover zero/odd/even trips. Pending tokens across swaps, N-slot rings,
  nested permutations and general CFG ownership remain open.
- Apple has an immutable shader/native-companion package and bounded synchronous
  binding to resident Metal tensors; six M1 Max cases pass after serialization.
  This is an explicit raw ABI with caller-owned extents and geometry. Typed JIT
  descriptors, arbiter registration and cross-binding stream ownership remain open.
- ROCm hardware-counter attribution is blocked by the installed gfx1151 WSL
  profiler: no PMC/SPM counters or PC-sampling agents. Structured capability
  evidence distinguishes unavailable counters from zero-valued observations.

### 2026-09-06 — Nonlinear forward products, pending swaps and typed Apple JIT

Owner: [W2.4a](INTEGRATED_COMPILER_PLAN.md#w24a)

PRs: Not specified in this historical entry; no merge inferred.

Outcome: Original recorded scope and validation follow below.

Remaining: Historical obligations below route through the Owner; they do not set current priority.

Evidence: Original source, tests and packet references preserved below.

<!-- entry-fields:end -->

W2.4a / CAKE / SO-2; **IR-NATIVE-FOUNDATION-1**. The newest section of the
[follow-up report](../../../benchmarks/NATIVE_STORAGE_FOLLOWUP.md) records:

- Native sigmoid/tanh/composed forward children, using overflow-safe tails and
  cancellation-safe small-input tanh. Nine numerical cases pass on each CUDA/ROCm
  owner, including signed zero and nonfinite inputs. Derivatives stay in C++ AD.
- Coupled pending-token/two-slot recurrences, with exact seed/backedge/final-drain
  checks. Prefetch may precede current-slot consumption. Seven CUDA cases cover
  zero/odd/even trips; no speedup is claimed from correctness evidence.
- Shared manifest validation with an Apple-specific resident tensor adapter and
  explicit JIT binding. Six M1 Max cases cover keyword calls, lazy rebinding and
  invalid shapes/scalars. Requested AD fails closed until Apple has a paired ABI.

Next: reduction/attention and reverse AD families; generalized N-slot/nested
recurrences; Apple paired AD and cross-queue ownership; supported ROCm counter
capture tied to actual dispatch images. Automatic family/arbiter selection still
requires semantic oracles. x86 retains host-companion evidence only, without
inherited GPU compute/AD proof.

### 2026-09-06 — N-slot ownership, reduction/reverse pairs and Metal queues

Owner: [W2.4a](INTEGRATED_COMPILER_PLAN.md#w24a)

PRs: Not specified in this historical entry; no merge inferred.

Outcome: Original recorded scope and validation follow below.

Remaining: Historical obligations below route through the Owner; they do not set current priority.

Evidence: Original source, tests and packet references preserved below.

<!-- entry-fields:end -->

W2.4a / CAKE / SO-2; **IR-NATIVE-FOUNDATION-1**. This section supersedes the
bounded status in the preceding loop; [device packets and scope](../../../benchmarks/NATIVE_STORAGE_FOLLOWUP.md)
remain the evidence authority.

- Released bijective N-slot rings and uniformly nested two-slot pending ownership
  now participate in reuse and arena revalidation. CUDA proves 3/4/8-slot rings
  and nested pending swaps, including zero/odd/even inner trips.
- Compiler-generated sum/mean reduction JVPs and residual-free single-input
  elementwise reverse pairs share a typed, serialized primal/derivative ABI.
  CUDA, ROCm and Metal each pass 15 cases. Reverse inputs include an explicit
  output cotangent; the compiler's backward SSA owns the derivative.
- Apple explicit JIT binding accepts paired packages. Fresh retained shared-event
  fences order producer blits before consumer AD on separate queues; 12 Metal
  generations pass. This is correctness evidence, with no selector promotion or
  overlap/performance claim.
- Frontend scalar tensors retain rank zero; traced and AST reduce(op=...) emit
  the canonical kind attribute. This closes actual native parse failures.

Next ordered contracts:
1. General N-slot *pending* ownership: a bijection alone is insufficient. Track
   token-to-allocation generation mappings through every nested backedge, seed,
   zero-trip path and final drain; reject incomplete or stale mappings.
2. Reduction VJP: separate primal and cotangent shapes in the physical ABI,
   implement scalar-to-vector broadcast from compiler backward SSA, and prove
   both output extents. Current reverse native children require equal shapes.
3. Attention AD: Tessera_FlashAttnOp lacks TangentInterface and AdjointInterface.
   Start with dense deterministic Q/K/V and paired saved-LSE/mask contracts,
   then causal/bias variants. Keep dropout RNG and cache effects explicit;
   never infer support from the existing hand-written backward packages.
   Use row/tile cooperative schedules rather than extending lane scalarization
   into quadratic materialized score matrices. Validate primal, JVP and VJP
   against independent oracles on each owning backend before arbiter enrollment.
4. General reverse residual/tape ABI and automatic family/arbiter generation.
5. Cross-queue resource ownership beyond explicit Metal fences, CUDA/HIP event
   parity, and measured overlap with hardware attribution where available.
   x86 remains the host companion; no GPU execution proof transfers to it.

### 2026-09-06 — Pending rings, mixed-shape reverse and attention products

Owner: [W2.4a](INTEGRATED_COMPILER_PLAN.md#w24a)

PRs: Not specified in this historical entry; no merge inferred.

Outcome: Original recorded scope and validation follow below.

Remaining: Historical obligations below route through the Owner; they do not set current priority.

Evidence: Original source, tests and packet references preserved below.

<!-- entry-fields:end -->

W2.4a / CAKE / SO-2; **IR-NATIVE-FOUNDATION-1**. The
[latest follow-up evidence](../../../benchmarks/NATIVE_STORAGE_FOLLOWUP.md)
supersedes the previous next-action list:

1. N-slot pending ownership is implemented for a single outstanding copy and
   a proven bijective slot/token destination mapping, including uniform nesting.
   CUDA validates 3/4/8 slots. Multiple outstanding generations and general CFG
   joins still need a set of generation-specific ownership facts.
2. Sum/mean reduction VJP and independent primal/cotangent tensor widths are
   implemented and measured for correctness on CUDA, ROCm and Metal.
3. Dense-f32 attention reverse and V-only forward interfaces are implemented
   using existing registered checkpoint operations; LSE is explicit and recomputed
   under one causal/scale policy. This is compiler-IR evidence, not device AD
   closure. Next: schedule compound products, persist forward LSE with generation
   identity, then add Q/K forward products and bias/dropout/cache policies.
4. Straight-line SAVE residuals now feed the fused native backward from forward
   SSA; multiple public primal outputs cannot masquerade as saved residuals.
   External persistent tapes and nested control-flow/state tapes remain open.
5. Five-process Metal queue measurements establish overlapping command timestamp
   intervals for matched independent work, with exact numerical results. They
   do not establish simultaneous instruction issue or a selector-worthy speedup.
   CUDA/HIP event parity and backend-specific counter attribution remain open.

Keep compiler builds separate from tests: pin the validated executable before
starting a suite, record its digest, and avoid re-linking under running tests.

### 2026-09-06 — Outstanding cohorts and saved attention LSE

Owner: [W2.4a](INTEGRATED_COMPILER_PLAN.md#w24a)

PRs: Not specified in this historical entry; no merge inferred.

Outcome: Original recorded scope and validation follow below.

Remaining: Historical obligations below route through the Owner; they do not set current priority.

Evidence: Original source, tests and packet references preserved below.

<!-- entry-fields:end -->

W2.4a / CAKE / SO-2; **IR-NATIVE-FOUNDATION-1**. This supersedes the
single-pending and recomputed-LSE boundaries above. See the
[implementation, measurements and next architecture boundaries](../../../benchmarks/NATIVE_STORAGE_FOLLOWUP.md).

- Multiple matched pending cohorts now carry distinct generation ownership
  facts. RTX5070 validates 2/3/4 outstanding copies under uniform nesting;
  arbitrary CFG joins and head-only FIFO waits remain open.
- Paired attention forward now returns LSE and backward consumes that explicit
  residual. This closes compiler SSA persistence, not device-owned tape storage.
- CUDA/HIP matched queue experiments have exact numerical and event evidence;
  CUDA additionally has Nsight kernel intervals and an isolated counter sample.
  ROCm hardware attribution remains open, and no selection policy changes.
- Next ordered work: owned persistent tape frames and split native products;
  nested invocation/state ownership; cooperative Q/K JVP lowering with its native
  consumer; attention package/LSE generation binding; production queue and
  fresh-process hardware attribution. Do not register an unsupported attention
  product merely to advertise an interface.

MSW status rechecked: MSW-5/6/7/8 are implemented in the reference lane;
36 focused tests, both tutorials and all 15 ANN reference-spike checks pass.
MSW-9's design spike and native program-pair evaluator adapter are implemented;
fragment/parameter extraction and the fusion candidate consumer remain open.
These statuses do not imply native backend closure for the math examples.

### 2026-09-06 — Persistent snapshots and generated checkpoint packages

Owner: [W4-PRODUCT-1](INTEGRATED_COMPILER_PLAN.md#w4-product-1)

PRs: Not specified in this historical entry; no merge inferred.

Outcome: Original recorded scope and validation follow below.

Remaining: Historical obligations below route through the Owner; they do not set current priority.

Evidence: Original source, tests and packet references preserved below.

<!-- entry-fields:end -->

W2.4a / CAKE / SO-2 / MSW-9; **IR-NATIVE-FOUNDATION-1**.

- CUDA/HIP reverse packages now expose `NativeStoragePair.capture`: owned
  snapshots survive source mutation, repeated backward results have distinct
  allocations, and nested invocation frames release with their parent. Twelve
  cases per GPU validate square/tanh/sum/mean. This is synchronous recomputation
  from snapshots, not higher-order AD or a lowering of nested control-flow tapes.
- Paired AD can export an isolated forward/backward attention checkpoint with
  canonical physical argument order and full paired-IR provenance. It feeds the
  existing native Schedule/Tile and NVIDIA package paths without Graph rebuilding.
  Six RTX5070 cases validate forward, LSE and Q/K/V VJP for causal/noncausal and
  unequal-length inputs. The host-buffer bridge persists LSE between calls;
  resident attention tape ownership and Q/K JVP remain open.
- MSW-9 has bounded immutable fragment extraction and an explicit composition
  candidate gate. Automatic fusion discovery, executable-to-fragment identity
  and backend promotion remain open; reference evidence never supplies native
  provenance.
- Incoming native Ubuntu26.04 RX9070XT profiling uses **gfx1201**, not gfx1200.
  The [commissioning plan](../backend/rocm/NATIVE_RDNA4_COMMISSIONING.md) includes
  a read-only probe, assertions-enabled compiler build, rocprofv3 and Systems
  Profiler capture. Hardware/counter validation awaits the actual host.

Next: split native forward/backward packages to avoid recomputation; integrate
saved-state control-flow frames; implement cooperative attention Q/K JVP with a
registered native consumer; bind resident LSE generation ownership; connect ANN
inventories to automatic fusion candidates; commission gfx1201 counters.

Q/K JVP algorithm spike: `tools/attention_jvp_spike.py` maintains rescaled
normalization/value/directional-score moments per row. Seventeen reference
cases cover Q-only, K-only, combined directions, block-size variation, extreme
logits and empty masks. It avoids full score materialization. Native tangent
registration is deliberately unchanged until the cooperative consumer exists.
The saved-LSE variant should consume the same generation's output and LSE,
compute `sum(P*dS*V + P*dV) - O*sum(P*dS)`, and share the canonical mask policy.

### 2026-09-06 — native ANN composition and resident attention generation

Owner: [W2.4a](INTEGRATED_COMPILER_PLAN.md#w24a)

PRs: Not specified in this historical entry; no merge inferred.

Outcome: Original recorded scope and validation follow below.

Remaining: Historical obligations below route through the Owner; they do not set current priority.

Evidence: Original source, tests and packet references preserved below.

<!-- entry-fields:end -->

Owner: W2.4a / CAKE / SO-2; MSW-9. Sync: **IR-NATIVE-FOUNDATION-1**.

Implemented this increment:

- `AttentionCheckpointPair.capture(q,k,v)` binds the existing native forward and
  backward images directly to CUDA allocations. The frame copies Q/K/V into
  private device storage and retains the exact forward-produced natural-log LSE.
  Repeated backward calls use that generation and return independently owned
  results. Dtype, strides, full allocation bounds, CUDA context, image/descriptor
  ABI, shape guards and pair policy are checked. Close invalidates views and
  unloads images; failed backward allocations do not reclaim earlier results.
  Six RTX 5070 cases cover both causal modes, unequal sequence lengths, caller
  input mutation and repeated backward. This is synchronous correctness evidence,
  not stream overlap, kernel speedup, or sibling-backend execution proof.
- `tessera-canonicalize=ann-reassociate=true` now automatically recognizes a
  single-use two-affine chain in native MLIR and folds constant weights and
  constant matrix biases with deterministic fp32 arithmetic. No replacement
  GraphIRModule or hand-written backend source is generated. Runtime parameters,
  intervening activations, transposition, policy overrides, nonfinite folding and
  excessive folding work refuse. The option defaults off because association
  changes rounding. This establishes a native MSW-9 fusion consumer for frozen
  inference constants, not automatic JIT/arbiter promotion or trainable fusion.

The next architecture work remains open, with these concrete boundaries:

1. **General nested control-flow tapes:** replace NativeStorageJVP's hard-coded
   two-output scalarizer with a split forward/backward product ABI. The ABI must
   carry every residual's dtype, full shape, ownership and indexing; nested loop
   state requires an iteration path, valid extent and branch selection, not just
   a stack of invocation frames. Preserve zero-trip and untaken-region behavior.
   Lower compiler-produced tensor tape reads/writes before accepting a new family;
   never infer a backward from an independently replayed control path. Establish
   allocation-size overflow checks and fail-before-launch capacity checks.
2. **Native Q/K JVP:** retain the paired O/LSE generation and lower
   `sum(P*dS*V + P*dV) - O*sum(P*dS)`, with
   `dS = scale*(dQ*K + Q*dK)` and `P = exp(S-LSE)`. The row consumer must use the
   exact `end_aligned_v1` mask and grouped-head mapping of forward. A registered
   tangent producer must ship with a real bounded-storage native consumer and
   exact-device finite-difference cases. The streaming Python spike remains a
   numerical oracle; V-only is still the public native tangent interface.
3. **Resident concurrency and siblings:** use completion-owned allocation
   generations before adding stream submission; private allocations alone do
   not establish concurrent lifetime safety. Add HIP/Metal ownership on those
   hosts independently. x86 needs its own host tape execution path.
4. **ANN promotion:** connect frozen parameter materialization and explicit
   reassociation policy to pipeline selection, then require native original/fused
   program-pair comparison and performance evidence before arbiter promotion.
   The earlier Python fragment view is an inventory, not executable identity.

Evidence: [loop8 implementation and validation](../../../benchmarks/NATIVE_STORAGE_FOLLOWUP.md).

### 2026-09-06 — typed nested exports and resident score tangents

Owner: [W4-PRODUCT-1](INTEGRATED_COMPILER_PLAN.md#w4-product-1)

PRs: Not specified in this historical entry; no merge inferred.

Outcome: Original recorded scope and validation follow below.

Remaining: Historical obligations below route through the Owner; they do not set current priority.

Evidence: Original source, tests and packet references preserved below.

<!-- entry-fields:end -->

Owners **W2.4a / CAKE / SO-2**, **MSW-9**; synchronization key
**IR-NATIVE-FOUNDATION-1**. This section supersedes the loop8 next-step status.

- `tessera-autodiff-paired=export-product=forward|backward` exports native
  products with a compiler-produced, full-type residual ABI and common paired
  lineage. Saved if/while products and nested SAVE loops retain native regions.
  Nested pullbacks now receive their cloned region residuals; statically empty
  positive-step loops are removed before tape sizing. This closes the export
  boundary, not general device tape lowering: nested inner residuals may be
  reconstructed from the saved outer state. General persistent device tapes
  still need allocation ownership, tensor tape lowering, dynamic iteration-path
  indexing, capacity checks and backend execution proof.
- Explicit CUDA resident `prepare_jvp`/`jvp` products now implement Q, K and V
  directions through native GPU MLIR, shared-memory reduction and the saved
  forward O/LSE generation. The bounded consumer uses grouped heads and the
  same end-aligned causal mask. The automatic FlashAttnOp TangentInterface is
  still V-only; wiring this consumer into automatic AD remains open. This
  consumer recomputes scores for each output column; it is correctness evidence,
  not a hand-tuned performance candidate.
- Resident backward launch sizing now covers the concatenated dQ/dK/dV range.
  The 129-key case exposed the old maximum-of-ranges grid under-launch.
- Arbiter exact-cache reuse and incumbent retention now require admissible
  selection evidence, including explicit eligibility and timing separation.
  Malformed or ineligible cached records trigger a fresh comparison. This is a
  shared admission prerequisite; no ANN candidate was promoted. MSW-9 still
  needs frozen-parameter native candidate registration, policy-bound identity,
  original/fused backend comparison and measured selection evidence.

See [loop9 evidence and backend boundaries](../../../benchmarks/NATIVE_STORAGE_FOLLOWUP.md).

### 2026-09-06 — control-flow contract correction and automatic score JVP

Owner: [W4-PRODUCT-1](INTEGRATED_COMPILER_PLAN.md#w4-product-1)

PRs: Not specified in this historical entry; no merge inferred.

Outcome: Original recorded scope and validation follow below.

Remaining: Historical obligations below route through the Owner; they do not set current priority.

Evidence: Original source, tests and packet references preserved below.

<!-- entry-fields:end -->

Owners W4 / W2.4a / CAKE / SO-2; sync **IR-NATIVE-FOUNDATION-1**.
`docs/spec/CONTROL_FLOW_CONTRACT.md` now distinguishes frontend acceptance,
portable control-to-SCF lowering, native packaging and owning-device proof.
The previous Apple-only sentence contradicted ROCm support; the table also
omitted cooperative ROCm paths, bounded native x86 state-machine proof and narrow handwritten CUDA control helpers, and confused incomplete direct CUDA graph packaging
with absence of native CUDA control flow. Historical CF0–CF4 narrative is
replaced by the active ownership links; this changes no backend execution state.

The FlashAttnOp TangentInterface now emits a registered internal
`checkpoint_jvp` for Q/K directions, with a same-producer O/LSE verifier and
explicit inactive tangent slots. `export-attention-jvp` accepts one isolated
attention function with direct argument/return mapping. The resident CUDA
consumer lowers this native contract and binds its product digest into package
identity. Eight CUDA cases / 32 directions pass analytic and finite-difference
oracles. Generic JIT compositions and sibling backend consumers remain open.

Persistent snapshot allocations now check byte overflow before allocation;
partial backward allocation failure rolls back only the attempted outputs.
This closes a lifetime defect, not general persistent nested tensor tapes.
The next tape implementation must replace NativeStorageJVP's combined
forward/recomputed-backward scalarizer with two independently callable native
products: bufferize full typed residual outputs, preserve iteration-path and
valid-extent indexing, then retain those allocations until all backward users
complete. Start with static nested SAVE loops; reject dynamic capacities until
native size derivation and fail-before-launch checks exist. A host-only arena
or another residual inventory would not close this item.

Evidence: [loop10](../../../benchmarks/NATIVE_STORAGE_FOLLOWUP.md).

### 2026-09-06 — split persistent tensor tapes and JIT-owned Q/K programs

Owner: [W4-PRODUCT-1](INTEGRATED_COMPILER_PLAN.md#w4-product-1)

PRs: Not specified in this historical entry; no merge inferred.

Outcome: Original recorded scope and validation follow below.

Remaining: Historical obligations below route through the Owner; they do not set current priority.

Evidence: Original source, tests and packet references preserved below.

<!-- entry-fields:end -->

Owners **W4 / W2.4a / CAKE / SO-2**; sync **IR-NATIVE-FOUNDATION-1**.
The first physical split-tape consumer now runs fresh native forward and backward
products independently. Upstream one-shot bufferization preserves full f32 tensor
residual shapes; readonly product inputs prevent in-place mutation of retained
snapshots. The new `tessera-native-tape-to-gpu` pass materializes bounded static
for/if bodies with per-iteration-path temporary slots. Backend-owned allocation
addressing is explicit: generic allocas for NVVM and private address space 5 for
AMDGPU. Repeated backward calls consume retained forward residuals and produce
independent outputs. The owning frame retains all allocations until explicit
close. These synchronous, serial GPU entries establish correctness, not a
parallel schedule or performance improvement.

The public JIT object now exposes `compile_persistent_device_tape` for reverse
requests and `compile_native_attention_jvp` for isolated SM120 forward attention
requests. The latter traces the function, runs native AD, binds the same resident
forward O/LSE generation and accepts tangent arguments in `wrt` order. Dense
three-tensor, dropout-free attention now emits its required head dimension and
precise pure effect; cache, dropout and unrecognized forms remain conservative.

Remaining: dynamic/mixed-type residual capacities (including saved predicates),
persistent while/general control tapes, concurrent backward retirement, parallel
physical tape schedules, composed automatic attention AD and sibling native
attention consumers. Static f32 slots are limited to 1024 elements each and
logical temporary storage to 4096 bytes. Native for/if acceptance does not imply
that every frontend control-flow or saved-state form fits this envelope.
See [loop11 evidence](../../../benchmarks/NATIVE_STORAGE_FOLLOWUP.md).

### 2026-09-06 — domain and autodiff documentation consolidation

Owner: [AD-HIGHER-1](INTEGRATED_COMPILER_PLAN.md#ad-higher-1)

PRs: Not specified in this historical entry; no merge inferred.

Outcome: Original recorded scope and validation follow below.

Remaining: Historical obligations below route through the Owner; they do not set current priority.

Evidence: Original source, tests and packet references preserved below.

<!-- entry-fields:end -->

The scoped [AD execution plan](AUTODIFF_EXECUTION_PLAN.md) now owns remaining
P0–P6/A–B/D/AD-NEXTGEN acceptance gates. The three prior documents are archived
as superseded designs, with small routing files preserving source links. Their
uncompleted work is transferred, not marked complete. W4/W5.1/W6 sequencing and
IR-NATIVE-FOUNDATION-1 remain the active owners.

The [domain audit](../domain/DOMAIN_AUDIT.md) and
[GA/EBM review](../domain/GA_EBM_ARCHITECTURE_REVIEW.md) correct stale manifold,
grade-consumer and CPU-performance claims. The next domain sequence is W3.6
batched GA/native rotor consumption; W3.5/W4 typed energy/gradient programs;
W5.1/W2.4a resident state and measured policy; W6.4 native finite-algebra lowering.
No independent domain emitter or rematerialization stack is commissioned.

[Target IR review](TARGET_IR_REVIEW.md) now recognizes the x86 dialect, semantic
string constraints and mixed native/compatibility ownership. Standalone loose
matrix contracts and spelling-only smoke tests remain real scoped gaps.
`CORE_SUBSTRATE_VIEW.md` was reviewed without modification; the
[compiler audit](COMPILER_AUDIT.md#2026-09-06-substrate-review-and-ad-consolidation)
records its outdated status/ownership passages. This update changes no target
execution status and adds no new device/performance claim.

### 2026-09-06 — Functional-analysis contracts — consolidated ownership

Owner: [RIEMANNIAN-OT](INTEGRATED_COMPILER_PLAN.md#riemannian-ot)

PRs: Not specified in this historical entry; no merge inferred.

Outcome: Original recorded scope and validation follow below.

Remaining: Historical obligations below route through the Owner; they do not set current priority.

Evidence: Original source, tests and packet references preserved below.

<!-- entry-fields:end -->

Updated 2026-09-06. This section replaces the independent FA-1–FA-7 execution
sequence. The [archived design](archive/FUNCTIONAL_ANALYSIS_TSOL_PLAN.md)
retains the mathematics and original IDs. These are remaining tasks, not new
support claims. Numerical-policy transport is an existing foundation;
composable error bounds are an additional analysis obligation.

| Item and owner | Remaining work and dependencies | Acceptance gate |
|---|---|---|
| **FA-1 — numerical legality / evaluator and arbiter** | First deliver one bounded reduction or fused-epilogue consumer. Extend NUMPOL-CARRIER-1 semantics with explicit norm, admissible domain, shape/reduction length and absolute error budget; preserve analysis facts across native MLIR boundaries. No independent Python production lowering stack or unused coverage axis. | A real candidate is accepted/refused using the budget. Composition checks perturbed intermediate-domain containment and uses justified downstream Lipschitz constants. Reduction bounds enforce their accumulation-algorithm, Ku < 1 and under/overflow assumptions. Unknown bounds fail closed for budget-based promotion. Oracle evidence, analytic bounds and device measurements stay distinct; norm-to-tolerance conversions are explicit. |
| **FA-2 — AD-LAW / AD-CLOSEOUT-1** | Reuse the implemented adjoint and canonical-forward laws; carry only public debug adapter, norm-aware tolerance and derivative-coverage evidence integration into AUTODIFF_EXECUTION_PLAN.md. Does not wait on invention of FA-1 or a new harness. | Planted incorrect and matched-incorrect derivative pairs remain detected; coverage changes cite the actual law result and evidence tier. Public adapter tests exercise the existing engine. Amend the coverage contract explicitly before tightening automatic status transitions. |
| **FA-3 — spectral family / numerical legality** | Audit existing spectral laws, then fill normalization, window/domain and multiplier-bound gaps. Error-budget consumption depends on FA-1; independent oracle coverage does not. | FFT normalization and adjoints agree; STFT/ISTFT round trips declare window/overlap assumptions. A spectral transformation consumes a justified multiplier bound, with negative legality cases and native-boundary preservation. Existing adjoint tests alone do not close this consumer. |
| **FA-4 — sequence-mixer stability** | Sequence-mixer plan owns recurrence/discretization-specific certificates and eventual op wiring. No new generic certificate registry without a recurrence consumer. | Domain and timestep assumptions are checked; nonnormal transient amplification and discretization-specific stability are covered. Reference oracle and native-device execution remain separate evidence. |
| **FA-5 — functional-calculus admission, deferred** | Require a named workload and a specific missing operation before reopening broad admission. Reuse spectral/solver owners; not a prerequisite for FA-1/2/3/6. | An admission proposal names semantics, domain, derivative rules, native producer/consumer and measurable benefit. An abstract common interface is insufficient. |
| **FA-6 — approximation legality / arbiter, consumer-gated** | After FA-1, attach norm-specific truncation bounds to an actual low-rank substitution candidate. Preserve the distinction between exact factorization and approximation. | Candidate selection respects the caller's budget and composition domain; over-budget substitutions reject. Sampled estimates cannot masquerade as certified upper bounds, and no automatic tolerance relaxation is allowed. |
| **FA-7 — PDE/forms, deferred** | Reopen under PDE_STENCIL_CAPABILITY_PLAN.md only when a solver or rewrite needs coercivity. | Concrete discrete operator, boundary/domain hypotheses and a certificate-consuming solver or transformation. |

Order: FA-1's bounded consumer first; FA-3/FA-6 numerical promotion builds on
it. Existing AD and spectral law improvements proceed independently. FA-4 is
sequenced by its domain owner; FA-5/FA-7 are explicitly deferred. Introduce
registry/schema changes together with their first consumer and focused drift
gates. CUDA, ROCm, Apple and x86 each require their own lowering-preservation
and exact-device evidence before hardware promotion; this consolidation changes
no backend support state or physical schedule.

### 2026-09-06 — F0 census correction and current next steps

Owner: [COMPILER-DEVEX-1](INTEGRATED_COMPILER_PLAN.md#compiler-devex-1)

PRs: #721, #732

Outcome: Original recorded scope and validation follow below.

Remaining: Historical obligations below route through the Owner; they do not set current priority.

Evidence: Original source, tests and packet references preserved below.

<!-- entry-fields:end -->

This update supersedes older summary counts, not their historical evidence.
The frontend-authority dashboard currently has nine missing NVIDIA family rows
and Apple normalization; older eight-row paragraphs are historical. Future
status reports should read the generated rows rather than duplicate this total.
A declared certification path is not an emitted exact-device certificate.

The package census now includes Apple GPU alongside Apple CPU and the three
other native package modules. Unannotated, `Any` and raw-IR inputs no longer
count automatically as compiled artifact consumers. Computed classifier returns
are exposed explicitly; Apple primitive membership remains unclassified rather
than being inferred from incomplete literals. Typed artifact inputs still need
semantic consumer/replay verification before migration closure. Missing source
modules fail the census instead of shrinking its denominator.

F0 remains landing: resolve computed Apple family membership from the live
producer, make family-to-compiled admission target/envelope aware, then join
actual driver/plugin call paths and artifact lineage. This bounded census fix
adds no backend execution support. After the relevant family's census is sound,
F2 migrates its remaining Graph constructor and F3 instantiates one optimized
matmul recipe for two witnessed buckets. PR #721's pre-bucket optimization and
narrow source-loop recognition are implemented foundations, not unstarted work.

Preserve PR #732's bounded tapes, resident Q/K JVP and ANN extraction/native
constant composition; their generalization and measured promotion remain open.
The recomputed-tape scratch-retirement follow-up is implemented with repeated
backward and failure-path allocation regressions. The active AD plan and FA-series
consolidation own the remaining native product and numerical-legality work.

### 2026-09-06 — F0 Apple family domains and native unary ancestry

Owner: [E2E-REAL-6F](INTEGRATED_COMPILER_PLAN.md#e2e-real-6f)

PRs: Not specified in this historical entry; no merge inferred.

Outcome: Original recorded scope and validation follow below.

Remaining: Historical obligations below route through the Owner; they do not set current priority.

Evidence: Original source, tests and packet references preserved below.

<!-- entry-fields:end -->

The census resolves the known Apple CPU computed return from the producer's
value/low-precision symbol tables and Apple GPU `value_*` returns from the
ready-descriptor table minus dedicated-family exclusions. Unknown expressions
remain unresolved. These are family-domain candidates, not proof that every
shape/policy is admitted. Apple primitive-membership joins remain explicitly
unverified; resolving names does not authorize coverage promotion.

A family-to-scheduled mapping now requires the corresponding consumer to exist
in that target's package module. In particular, Apple CPU matmul does not inherit
Apple GPU or x86 scheduled coverage. This is a necessary structural gate;
full driver/plugin control-flow and envelope joins remain F0 work.

Apple scheduled softmax/reduction now replays the retained Schedule through
native `--tessera-schedule-to-tile`, requires exact equality with the supplied
Tile product, and checks the native module's apple7/apple_gpu identity before
runtime lookup. A changed operand pair or relabelled x86 parent is refused even
when the previous structural validator accepts it. Direct packaging therefore
requires the production `tessera-opt`; synthetic descriptor tests explicitly
stub replay and do not count as compiler proof. Native positive/negative replay
runs on the WSL compiler host, not as new Metal execution evidence.

Next: verify descriptor-field projection and extend the same ancestry obligation
to remaining Apple matmul/attention consumers before retiring their Graph
constructors. Reuse each native parent and preserve library-vs-generated-body
provenance. No other backend's runtime, capability or promotion state changes.

### 2026-09-06 — Descriptor projection and seven-program continuation

Owner: [LAYOUT-ALG-1](INTEGRATED_COMPILER_PLAN.md#layout-alg-1)

PRs: #732

Outcome: Original recorded scope and validation follow below.

Remaining: Historical obligations below route through the Owner; they do not set current priority.

Evidence: Original source, tests and packet references preserved below.

<!-- entry-fields:end -->

Owning key: **F0 / IR-NATIVE-FOUNDATION-1**. Apple GPU unary and plain
rank-two matmul packages now check descriptor geometry, tensor types, supported
layout/arithmetic policy and tile decisions against native-printed Schedule IR.
Schedule-to-Tile replay now also gates matmul and forward/backward attention
before runtime lookup. Unary buffer names remain explicit host binding aliases;
they are not claimed as serialized SSA provenance. The projection reader accepts
the current bounded native printer grammar and rejects unsupported forms.

This closes a bounded package-boundary defect, not canonical ownership as a
whole. Attention descriptor projection, dynamic/epilogue matmul projection,
complete shape-policy route joins and remaining Graph constructors stay open.
Compiler replay evidence is separate from Metal execution and performance.
Princess-Luna's LLVM 23.1.1 reports assertion mode `OFF`; its passing tests do
not satisfy the assertions-enabled MLIR gate.

Continue in this dependency order, preserving each program's existing owner:

1. **Canonical ownership / descriptor projection:** finish attention and dynamic
   descriptor fields, then migrate the next Graph constructor with independent
   target ancestry and owning-device comparison. Retain library/generated-body
   provenance rather than equating a typed wrapper with compiled execution.
2. **General AD execution:** dynamic/mixed/while persistent tapes and asynchronous
   retirement, followed by composed attention, batching, sparse derivatives and
   native higher-order products. Bounded PR #732 products remain bounded.
3. **Optimization integration:** native optimized-recipe instantiation and native
   ANN evaluation before selector-grade measured promotion.
4. **Numerical legality:** carry FA-1–FA-7 error-budget composition into concrete
   spectral/approximation consumers; metadata alone cannot establish legality.
5. **Runtime resilience:** bounded CUDA/HIP waits must define context poisoning
   and retain resources until completion is known, including timeout tests.
6. **Evidence infrastructure:** obtain an assertions-enabled native compiler,
   finish route/envelope inventories and reconcile authored summaries against
   revision-bound generated evidence. Device proof remains architecture-owned.

These remain active programs; this implementation does not close their broader
execution or performance gates.

### 2026-09-07 — Attention projection and static softmax migration

Owner: [W4-PRODUCT-1](INTEGRATED_COMPILER_PLAN.md#w4-product-1)

PRs: Not specified in this historical entry; no merge inferred.

Outcome: Original recorded scope and validation follow below.

Remaining: Historical obligations below route through the Owner; they do not set current priority.

Evidence: Original source, tests and packet references preserved below.

<!-- entry-fields:end -->

**F0/F2 / IR-NATIVE-FOUNDATION-1:** Apple forward/backward attention packaging
now projects rank-four tensor signatures, dimensions, storage, numeric modifiers,
bias presence, LSE policy and scheduling fields from native-printed Schedule IR
after exact Schedule-to-Tile replay. Host buffer aliases remain explicit and
must be distinct. Backward's public function name is a source alias; its native
entry is synthesized and the library symbol is fixed by the verified storage ABI.
Floating attributes are compared at the native f32 precision.

Static f32 `package_softmax(GraphIRModule)` now acts as a compatibility frontend:
it invokes the shared native lowering and artifact consumer instead of building
a descriptor and placeholder Tile text from Graph fields. The f16/bf16 branch
remains historical. This is a bounded constructor migration, not deletion of
the Graph entry API or proof that every package is IR-owned. The scheduled
consumer now owns f32 provenance even when reached through the legacy entry.
Missing native compiler support fails rather than reconstructing a package.

WSL native compiler and descriptor tests validate the boundary; no new Metal
execution/performance evidence is claimed. A follow-up probe found that Apple
f16/bf16 backward's forward companion is rejected by the current native
Graph-to-Schedule pass despite the runtime variant table. That producer gap
must close before claiming low-precision paired package support. Remaining:
full operand/region semantic projection, low-precision softmax migration and
owning-device differential proof before removing historical constructors.

### 2026-09-07 — Low-precision Apple native closure slice

Owner: [W4-PRODUCT-1](INTEGRATED_COMPILER_PLAN.md#w4-product-1)

PRs: Not specified in this historical entry; no merge inferred.

Outcome: Original recorded scope and validation follow below.

Remaining: Historical obligations below route through the Owner; they do not set current priority.

Evidence: Original source, tests and packet references preserved below.

<!-- entry-fields:end -->

**F0/F2 / IR-NATIVE-FOUNDATION-1:** static f16/bf16 softmax now follows the same
native Schedule/Tile consumer as f32, with storage-derived ABI, alignment and
buffer types. The historical static softmax descriptor constructor is removed;
the public Graph entry remains a compatibility frontend to native lowering.
Apple low-precision attention's forward recompute companion is now admitted by
Graph-to-Schedule when its function carries the recompute checkpoint contract.
Standalone low-precision forward runtime packaging is still refused.

The new native compiler passed WSL projection tests. Exact-device differential
checks on the M1 Max passed for f16 and bf16 softmax and causal GQA backward,
including all three gradients, using a freshly compiled runtime and requiring
`native_gpu` for every launch. See
[the bounded evidence packet](../../../benchmarks/apple_gpu/lowp_native_differential.json)
and `tests/unit/test_apple_lowp_native_contract.py` for shapes, tolerances,
source/compiler/runtime hashes and reproduction assertions. Native compilation
ran on Princess-Luna over SSH; execution and numerical proof belong only to the
Mac. This is correctness evidence, not timing or selector promotion.

Remaining: biased/noncausal and broader-shape low-precision device coverage,
full semantic operand projection, and subsequent Graph-owned families. The
assertions-enabled LLVM gate remains open.

### 2026-09-07 — Broader Apple coverage and measured runtime promotion

Owner: [W4-PRODUCT-1](INTEGRATED_COMPILER_PLAN.md#w4-product-1)

PRs: Not specified in this historical entry; no merge inferred.

Outcome: Original recorded scope and validation follow below.

Remaining: Historical obligations below route through the Owner; they do not set current priority.

Evidence: Original source, tests and packet references preserved below.

<!-- entry-fields:end -->

**F0/F2 / APPLE-ATTN-BWD-1 / IR-NATIVE-FOUNDATION-1:** six additional M1 Max
package differential cases passed: MHA/GQA/MQA, B=1/2, Sq=7/9/17/19,
Sk=7/19/33/65, D=16/32/64/128, causal/noncausal, fp16/bf16 unbiased and fp32
biased. All dQ/dK/dV outputs were checked against a float64 analytic VJP and
every launch required native GPU placement. Low-precision compiled bias remains
explicitly refused: native IR specifies fp32 bias while the runtime variant
requires storage-typed bias. Silently quantizing it would change the program.

Five independent, paired/interleaved runtime benchmark processes (seven trials,
ten repetitions) validated all three routes on the existing six-case closure
matrix. The strict median-order-statistic gate promoted `split_reduced` over
`serial_recompute` for six exact shape/dtype keys in both timing domains.
Across the recorded per-run medians, device speedups were 5.25–22.55x and
host-input/output end-to-end speedups were 4.26–16.72x; these ranges describe
these fixtures only. Atomic was measured but was not the selected winner.

[Raw reports and coverage](../../../benchmarks/baselines/apple_backward_20260907)
and the [strict family ledger](../../../benchmarks/baselines/apple7_attention_backward_strict_v2_route_ledger.json)
retain distinct timing domains and exact context. The backward route-policy
resolver now defaults to this dedicated ledger, preserving explicit global and
per-call overrides. Live validation accepted 12 rows, refused a changed runtime
fingerprint, and retained serial when the split workspace exceeded the cap.
This promotes the measured **runtime-route policy**; compiled artifacts still
bind their declared fixed two-way split and these reports do not establish
package-subgraph performance. Biased BF16 direct-runtime evidence uses BF16
bias and is not evidence for compiled fp32-bias inputs.

Remaining: mixed-storage bias ABI/projection, package-level promotion evidence,
unmeasured shape envelopes, and assertions-enabled compiler validation.

### 2026-09-07 — Math audit / foundation reconciliation

Owner: [W6.4](INTEGRATED_COMPILER_PLAN.md#w64)

PRs: Not specified in this historical entry; no merge inferred.

Outcome: Original recorded scope and validation follow below.

Remaining: Historical obligations below route through the Owner; they do not set current priority.

Evidence: Original source, tests and packet references preserved below.

<!-- entry-fields:end -->

This is a source-and-ownership review, not a fresh reproduction of the cited
papers or every historical numerical result. Math semantics and reference laws
feed one native compiler; they do not justify parallel Python lowering stacks.
Keep four evidence tiers distinct: reference law, serialized native contract,
executed target artifact, and measured candidate admission.

| Document | Disposition | Current integration / remaining gate |
|---|---|---|
| MLIR_NATIVE_FOUNDATION_SURVEY | Keep as architectural reference; this section supersedes dated census/absence claims. | F0 inventory plus F2 descriptor/ancestry migration; static Apple softmax now delegates through native IR for f32/f16/bf16. General program-body compilation remains distinct from library delegation. |
| MATH_SOURCE_WORKSTREAM | Archive the original proposal; live routing note retains MSW IDs. | MSW-1–8 bounded reference work stays landed. MSW-9 uses ANN/F3 for automatic native fragment discovery, executable identity and measured admission. Native higher-order products use the AD execution plan. |
| MATRIX_CALCULUS_REVIEW | Keep mathematical reference, label its historical findings. | MC1/2/5/8 remediation is not a new backlog. MC3 metric/manifold descent uses geometry + native AD; MC4 native Kronecker/vec rewrite uses F3 and a no-materialization gate; MC6 native higher-order composition uses AD-CLOSEOUT-1 and AD-WEIL-1; MC7/9 counting/law fixes stay landed and are reused as gates. Recheck deferred matrix operations before adding new public ops. |
| RIEMANNIAN_OT_PLAN | Keep scoped acceptance-workload plan. | Existing ga.manifold Euclidean/Sphere/SOn and metric reference methods contradict a blanket “whole layer missing” claim. R0/R1 require explicit native manifold/domain carriers and consumers; R2 reuses native AD/residual authority, R3–R5 are downstream workloads. No fresh OT numerical/backend proof is inferred. |
| SEQUENCE_MIXER_ENGINEERING_PLAN | Keep scoped implementation plan. | W1/W2 reference/state interfaces precede W3/W4 native recurrence descent, then W5/W8 target forward/backward ownership. W6 precision and W7 admission use NUMPOL/F3/FA-4. The common tiled SSD program remains distinct from ReplaySSM's bounded ABI. |
| SEQUENCE_MIXER_THEORY | Keep reference; dated capability table is not current status. | Transition/state/reassociation semantics constrain native IR. Check each transition family and current producer, not just the existence of scalar delta_rule or a runtime kernel. |
| PDE_STENCIL_CAPABILITY_PLAN | Keep scoped contract/consumer plan. | Python pde_operator classification and diagonal-diffusion certificates, coordinate fields and stencil materialization are bounded. PDE-STENCIL-FOUNDATION-1 / FA-7 require explicit spacing, coefficients, boundary/domain and discretization assumptions consumed by native passes and target code. |
| FORGE_ASSESSMENT | Archive proposal; preserve mathematical evidence and live routing note. | W1/W2 → LAYOUT-ALG-1/Schedule memory authority; W3/W4/W7/W8 → F3 stateful fusion; W5 → NUMPOL/FA-1; W6 → DIST-NATIVE-1. No new residency registry or unconditional “exact candidate” promotion. |

#### Dependency order and exit tests

1. **F0/F2 native ownership:** every selected family serializes ABI, layout,
   numeric policy, state and provenance; package fields are checked projections.
   Wrong dtype, operand ancestry, alias or policy rejects before launch.
2. **Native AD and numerical legality:** wire manifold/coordinate/recurrence
   consumers into existing AD and NUMPOL/FA-1 authority. State norm, domain,
   discretization and error-budget assumptions; unsupported cases refuse.
3. **F3 math optimizations:** native Kronecker/vec, ANN and stateful epilogues
   require original/transformed execution, alias/effect and no-materialization
   checks, then candidate identity binding. Reference identities remain oracles.
4. **Target workloads:** sequence mixers/SSD, PDE/OT and optimizer state updates
   consume that foundation. Paired package-level performance evidence is owned
   per architecture; direct-runtime measurements do not promote a compiled
   package or transfer to a sibling target.

The two archived proposals are consolidated, not completed programs. Remaining
work above stays active under existing IDs; the archive is not a second queue.

### 2026-09-07 — Mixed-storage bias and native-package admission

Owner: [W2.4a](INTEGRATED_COMPILER_PLAN.md#w24a)

PRs: Not specified in this historical entry; no merge inferred.

Outcome: Original recorded scope and validation follow below.

Remaining: Historical obligations below route through the Owner; they do not set current priority.

Evidence: Original source, tests and packet references preserved below.

<!-- entry-fields:end -->

**F2 / APPLE-ATTN-BWD-1 / IR-NATIVE-FOUNDATION-1:** dedicated f16/bf16 + fp32-bias
status ABIs now preserve the native bias tensor without narrowing. MSL bias
loads and buffer allocation widths are independent of Q/K/V/dO storage; existing
storage-typed ABI signatures are unchanged. Python binding registration, native
launcher dispatch, descriptor types/alignment and non-Metal stubs agree.
Eight broader M1 Max package VJP cases passed, including fp16/bf16 biased cases.
This supersedes the earlier mixed-storage refusal as an implementation gap.

Five independent paired reports compare complete native-package dispatch with
the identical direct split-kernel ABI, excluding compilation from warm timings
and reporting it separately. The strict `package_subgraph` ledger retains the
direct incumbent: f16 package calls were slower; bf16 evidence was mixed and
failed its lower-bound gate. No package promotion is made. These are native
image/LaunchDescriptor packages, not `.mtlpackage` ML execution, and no device
clock claim is derived from the complete-call timer. See
[package reports and ledger](../../../benchmarks/baselines/apple_native_backward_package_20260907).
The instability reason no longer incorrectly says a mixed-win candidate won
every run; the admission thresholds themselves are unchanged.

Remaining: reduce measured package overhead with identity-preserving caching,
then remeasure before admission; extend mixed-bias shape/policy coverage. The
new runtime changes the exact fingerprint. Five fresh owning-device runtime
reports admit split_reduced for six shapes in both timing domains; live loading,
context mismatch refusal and workspace fallback passed. See
[refreshed runtime evidence](../../../benchmarks/baselines/apple_backward_mixed_runtime_20260907).
Fresh M1 Max fleet measurement is now sealed against committed source
`80504c8384e61f157a5fb7e772f3d6b2c22da56c`: matmul and softmax prove Metal
placement at fixture and timing shapes, with device-event and end-to-end rows.
The initial 15-sample/50-iteration attempt failed the unchanged 4% stability
gate; 21 samples with 200 amortized iterations passed. This is a fresh
measurement, not a fingerprint-only update. See
[the sealed fleet packet](../evidence/e2e_spine/apple_gpu/apple7/manifest.json).

### 2026-09-07 — PR #733 contract corrections

Owner: [W4-PRODUCT-1](INTEGRATED_COMPILER_PLAN.md#w4-product-1)

PRs: Not specified in this historical entry; no merge inferred.

Outcome: Original recorded scope and validation follow below.

Remaining: Historical obligations below route through the Owner; they do not set current priority.

Evidence: Original source, tests and packet references preserved below.

<!-- entry-fields:end -->

F2 / APPLE-ATTN-BWD-1: the initial low-precision softmax placement claim used
void ABIs and could not distinguish CPU fallback. Native packages now require
v2 status ABIs that return success only for Metal; failed or missing status
refuses. Ten fresh M1 Max differential cases passed using this boundary.
Low-precision attention recompute is admitted only with reciprocal primal/VJP
symbol links and verified matching types, zero/explicit bias and numerical
policy. A checkpoint attribute alone is insufficient. Shared graph producers
emit the relationship; standalone low-precision Apple forward remains refused.
The Apple tests use the central hardware capability inventory. Five fresh runtime reports retain the six-key admission in both timing domains.
The fresh fleet packet is sealed against e73cdfcee39c8265cdc329b407b05c834c5a5a70;
matmul and softmax passed placement and timing checks at the unchanged 4% gate.

### 2026-09-07 — Capability-plan reconciliation

Owner: [W4-PRODUCT-1](INTEGRATED_COMPILER_PLAN.md#w4-product-1)

PRs: Not specified in this historical entry; no merge inferred.

Outcome: Original recorded scope and validation follow below.

Remaining: Historical obligations below route through the Owner; they do not set current priority.

Evidence: Original source, tests and packet references preserved below.

<!-- entry-fields:end -->

This source/evidence reconciliation retains existing IDs and creates no parallel
compiler queue. It follows the F0–F4 foundation order above. Reference laws,
serialized native contracts, exact-target execution and measured admission
remain separate. The review's 111 passing focused reference/evidence tests
(one skipped) do not renew device measurements or external-paper surveys.

| Scoped document / residual | Existing owner | Required implementation and exit gate |
|---|---|---|
| CORE_SUBSTRATE_VIEW S1–S9 | W2.4a / W5.2 / F0–F4 / NUMPOL / LAYOUT-ALG-1 / AD | Keep one map to actual consumers. Retire the old blanket absence/unowned claims; derive alias/effect/lifetime facts and preserve serialized policy/provenance. The archived P0–P5 sequence is historical. |
| DIFFERENTIABLE_PROGRAMMING C1–C6/T1–T3/R1–R2 | AUTODIFF_EXECUTION_PLAN / AD-CLOSEOUT-1 / AD-HIGHER / AD-WEIL / NUMPOL | Preserve linear-transpose, effect, reference-loss and bounded solver foundations. Extend native products, checkpoint-plan execution, constrained/matrix-free solves and named estimators with appropriate laws, certificates and target evidence. These labels are provenance aliases, not new work IDs. |
| BLOCK-ATTNRES-1 phases 6–7 and broader envelopes | BLOCK-ATTNRES-ROCM-2026-08-12 / F2/F3 / W2.4a / AD / W5.2 | Consume the existing depth-statistics/merge contract. Prove native query hoisting and block-state retention/recompute legality; broaden storage/shapes with explicit numeric policy. Sibling packages need independent ancestry and device correctness; selector admission needs valid target clocks and matched baselines. TP uses DIST-NATIVE-1. |
| EGGROLL W2 reverse/breadth | EGGROLL-ES-LOWRANK-2026-08-09 / F2/F4 / AD / NUMPOL | Use shared linear transposition for the fixed-key correction, preserving member/RNG identity. Rank>1, quantized accumulation and other targets need explicit contracts and native comparison; rank-1 fp32 x86/gfx1151 correctness is already recorded. |
| EGGROLL W3/W4 | F3 / W5.2 / DIST-NATIVE-1 | Preserve optimizer-state order, aliases and numeric policy; prove no forbidden materialization, then measure the complete update. Mock scalar-gather reconstruction does not close native transport or multi-rank performance. |
| GAME G1b/G5 | REF-TIER-PHYS-2026-08-16 / LAYOUT-ALG-1 / F2/F3 | Preserve the shared coalition carrier. Replace FFT-only tiling/sharding only after layout and FFT bit-identity gates pass; require per-target package evidence. Coalition kernels do not imply equilibrium-solver support. |
| GAME G2–G4/G6 | AD execution / W4 / F4 / DIST-NATIVE-1 | Build a named certified solver or game workload using existing segmented reduction, scans and explicit RNG. Check constrained derivative assumptions, game-oracle correctness and real batching. Distributed/sampled variants require their own transport or estimator certificates. |

Workload selection is downstream of the required foundation slice, not a reason
to recreate it. Keep Block AttnRes, EGGROLL and game theory as scoped landing
plans; keep substrate and differentiable-programming documents as references.
Historical status/sequence excerpts live in `archive/` and carry no active queue.
The four backend plans retain architecture-specific validation obligations.

### 2026-09-07 — Status, native GELU and recipe instantiation

Owner: [DIST-NATIVE-1](INTEGRATED_COMPILER_PLAN.md#dist-native-1)

PRs: Not specified in this historical entry; no merge inferred.

Outcome: Original recorded scope and validation follow below.

Remaining: Historical obligations below route through the Owner; they do not set current priority.

Evidence: Original source, tests and packet references preserved below.

<!-- entry-fields:end -->

**F0/F2/F3 / IR-NATIVE-FOUNDATION-1:** this increment closes the native status
boundary for descriptor f32 softmax and f32/f16/bf16 GELU, including their dynamic
compatibility descriptors. ABI v2 uses native-only status symbols; missing or
failed dispatch refuses. The native Apple softmax/GELU lowering also consumes
status with `cf.assert`; it no longer drops the result of a fallible submission.
The legacy void APIs retain their explicit compatibility fallback. Positive i32
shape limits are checked before the runtime boundary.

Static GELU packaging now accepts a retained native artifact, replays its parent,
and projects its tensor contract from native-printed IR. Unsupported policies,
extra operations, edited output IR and wrong target identity refuse before
runtime access. This uses the existing Apple native library lowering rather
than adding a duplicate Schedule dialect operation. Dynamic GELU now delegates through the same native parent/replay artifact path.
Native dimension positivity and i32 element-count guards precede allocation.
CUDA floating matmul (including bounded dynamic and fused epilogues) and ROCm
plain/dynamic matmul now verify descriptor projections against native Schedule
and replayed Tile IR. Static axes retain equality guards even in a partly
dynamic CUDA package. Graph text remains historical provenance, not an input
to these package verifiers.

F3 now has explicit native instantiation of a straight-line matmul recipe using
`tessera-symdim-equality`'s opt-in dimension witness. Two buckets preserve one
optimized parent identity and produce distinct concrete IR identities. This
has no execution/promotion bit: broader shape-transfer rules, native ANN pair
execution and artifact-bound arbiter admission remain next consumers.

Package profiling identified buffer-contract preparation overhead. Only immutable
NumPy dtype spelling is cached. Mutable nested descriptor metadata invalidates
the cached-identity premise, so descriptor digests are recomputed. Every launch
still validates live buffers/scalars. Five new M1 Max complete-package/direct-ABI
comparisons do **not** justify promotion; see the
[reports](../../../benchmarks/baselines/apple_package_status_20260907/README.md).

Remaining dependency order:

1. Dynamic GELU and CUDA/ROCm floating matmul descriptor projection are implemented
   for their existing envelopes, with native replay and metadata-corruption
   regressions. Broader GELU shapes and ROCm fused matmul are not admitted;
   neither package replay nor differential execution establishes promotion.
2. Extend recipe shape transfers to the chosen ANN fragment; bind original and
   transformed executable artifacts, numerical policy and measured admission to
   each concrete instance. Matmul instantiation alone does not close ANN/F3.
3. Static mixed f32/f64 slots and proven counted whiles now have a native
   persistent consumer. Continue integer/predicate and dynamic slots, genuinely
   data-dependent while termination and completion-owned asynchronous retirement.
   Selected bounded SAVE/HYBRID/recompute-all plans execute on CUDA/HIP; automatic
   checkpoint-plan selection and performance promotion remain open.
4. FA-1 now has a bounded frozen-affine absolute-error consumer in native ANN
   admission. Extend beyond this fixed f32/linf envelope: serialized numerical
   carriers, general reductions, spectral/approximation consumers and justified
   domain composition. Sampled agreement alone remains insufficient.

Cross-cutting: all three current LLVM installations report assertions OFF.
A separate assertions-enabled LLVM/MLIR build remains required. The generated
route inventory must distinguish compatibility frontend adapters from
Graph-owned packaging, and a native library call from general program-body
compilation. The fleet packet
must be resealed after committing the changed runtime source, before publishing.

### 2026-09-07 — Dynamic package projection validation

Owner: [E2E-REAL-6](INTEGRATED_COMPILER_PLAN.md#e2e-real-6)

PRs: Not specified in this historical entry; no merge inferred.

Outcome: Original recorded scope and validation follow below.

Remaining: Historical obligations below route through the Owner; they do not set current priority.

Evidence: Original source, tests and packet references preserved below.

<!-- entry-fields:end -->

F2 / IR-NATIVE-FOUNDATION-1: the dynamic GELU native producer handles dimensions
in IR; its compatibility frontend no longer constructs a separate descriptor.
Argument-local dimension names survive packaging. Matmul validation projects
shape bounds, individual dynamic axes, storage, accumulation, epilogues and
entry/tile identity. A modified Tile program refuses before compilation.

Focused WSL compiler/registry checks: 371 passed, 26 skipped for capabilities;
RTX 5070 scheduled-matmul device suite: 14 passed; M1 Max selected native
softmax/GELU suite: 6 passed; gfx1151 scheduled matmul: 3 passed. These are correctness results, not performance
promotion packets. Shared changes do not establish x86 or other GPU parity.

The next implementation boundary is F3's native ANN program-pair executable
adapter and artifact-bound admission. In parallel with that dependency chain,
AD-RESIDUAL-EVAL-1 still requires typed saved slots and bounded while retirement
before general checkpoint execution, and FA-1 still requires a real analytic
consumer. None of those is closed by this F2 increment.

### 2026-09-07 — Native ANN admission and persistent checkpoint execution

Owner: [W4-PRODUCT-1](INTEGRATED_COMPILER_PLAN.md#w4-product-1)

PRs: Not specified in this historical entry; no merge inferred.

Outcome: Original recorded scope and validation follow below.

Remaining: Historical obligations below route through the Owner; they do not set current priority.

Evidence: Original source, tests and packet references preserved below.

<!-- entry-fields:end -->

Owners: **F3 / FA-1 / AD-RESIDUAL-EVAL-1 / W2.4a / IR-NATIVE-FOUNDATION-1**.

`native_ann.py` prepares one native two-affine-layer to one-layer rewrite and
replays its native producer before admission. The evaluator uses the existing
MLIR/LLVM JIT on x86, with no Graph reconstruction or reference fallback. Both
programs are checked against an independent rational-arithmetic affine oracle;
matching wrong native outputs refuse. `register_native_ann` registers original
and rewritten programs in the existing arbiter under its internal `ann_affine`
family. The field is bound to the program pair, input domain, budget and probe
snapshots. An over-budget rewrite is excluded even before timing. Equal-tier
selection retains the original; actual calls recheck the input domain.

The analytic contract is deliberately bounded: static rank-two f32 tensors with
dimensions at most 64, frozen normal-or-zero parameters, explicit reassociation
permission, finite inputs with `||X||inf <= R`, and absolute output linf difference from the original program.
Round-to-nearest is checked on the calling thread; other rounding modes refuse.
Exact rational analysis includes folded parameter error, each matrix's induced
column-sum norm, at most 2K arithmetic roundings per dot plus bias rounding, and
additive minimum-normal allowances for underflow/flush-to-zero. `2Ku < 1` and a
conservative intermediate-overflow exclusion are mandatory. Downstream domain
bounds include preceding rounding errors. This is numerical eligibility, not a
performance packet, automatic JIT routing, or GPU ANN proof. Broader recipe
instantiation, nonlinear ANN fragments, numerical-carrier integration and
measured promotion remain open.

Persistent storage now projects f32/f64 widths through compiler private arrays,
ABI manifests, owned allocations and backward outputs. Replay-loop capacity is
derived from bounded SSA arithmetic/selects over enclosing induction variables;
loaded bounds and unchecked maximum annotations do not constitute proofs. Every
nested allocation reserves a distinct slot for its maximum iteration path.
The optional paired-AD `normalize-counted-while` converts only pure zero-origin,
unit-step, constant-bound whiles with sufficient declared capacity to native
for loops, retaining SAVE/HYBRID policy. Other while forms remain refused by the
physical tape consumer.

The owning-device recorder exercises mixed storage, nested SAVE, HYBRID,
recompute-all and counted-while SAVE, with repeated backward calls and unchanged
residuals. See `benchmarks/baselines/tape_checkpoint_20260907/`. Physical retained
bytes are recorded separately from private temporary capacity. These results
do not establish latency, overlap, general mixed-state while execution, or
checkpoint autotuning. Apple still needs MSL-owned typed tape materialization;
x86 ANN execution does not supply CUDA/HIP ANN proof.

### 2026-09-07 — Data-dependent native tapes and nonlinear GPU ANN

Owner: [W4-PRODUCT-1](INTEGRATED_COMPILER_PLAN.md#w4-product-1)

PRs: Not specified in this historical entry; no merge inferred.

Outcome: Original recorded scope and validation follow below.

Remaining: Historical obligations below route through the Owner; they do not set current priority.

Evidence: Original source, tests and packet references preserved below.

<!-- entry-fields:end -->

Owners: **F3 / FA-1 / AD-RESIDUAL-EVAL-1 / W2.4a / IR-NATIVE-FOUNDATION-1**.

The next physical increment preserves MLIR ownership throughout:

- `normalize-data-while` admits pure, single-block whiles with a zero-origin,
  unit-step index counter and an actual signed `counter < constant` conjunct.
  It freezes the complete carried state after a data-dependent exit, then uses
  the existing generic counted-loop checkpoint machinery. A `max_iters`
  annotation by itself is insufficient. Capacity remains at most 1024 steps;
  physical temporary storage remains at most 4096 logical bytes.
- `box-product-scalars` projects logical index/predicate residuals, including
  checkpoint tensors, into i64/i8 tensor storage in the native export. It does
  not change the differentiation domain: discrete region yields no longer
  receive cotangent seeds. Rank-zero LLVM memref descriptors now omit empty
  dimension/stride arrays. Dynamic tensor extents still refuse at this physical
  boundary; static integer checkpoint storage is not dynamic allocation proof.
- Persistent backward can submit distinct generations on caller-owned streams.
  Completion polling retires event owners, while each result advertises its
  producer stream. Explicit generation release waits for device completion and
  frees only that generation. Fully asynchronous allocation retirement requires
  tracking every downstream reader; no such general closure is claimed.
- The native frozen-affine rewrite can retain a terminal ReLU. ReLU is
  nonexpansive in the absolute infinity norm, so the existing analytic bound
  remains valid. Internal nonlinear activations cannot be commuted through an
  affine composition. CUDA/HIP original and transformed programs now consume
  native bufferized IR, preserve nonsplat constants, and replay source-to-device
  arena ancestry before binding. This is a bounded serial physical baseline,
  not a tuned tensor-core/WMMA ANN schedule.
- Nine independent processes compare each original/rewrite package, including
  the host buffer bridge, using randomized paired samples. Raw measurements are
  rechecked before applying the existing exact median order-statistic interval.
  Nine runs give a second-order bound, so one extreme run cannot alone set
  the lower endpoint. The lower bound must exceed a 2% margin. Package wall time is not a kernel
  clock, and numerical eligibility does not register a production GPU winner.

Six tape cases pass on each GPU. Nine independent ANN runs per target refuse
promotion: CUDA median 1.00447× (lower 0.98833×), ROCm median 1.00243×
(lower 0.99360×), against the 1.02× threshold. Evidence is in
[the native tape/ANN packet](../../../benchmarks/baselines/native_tape_ann_20260907/README.md).
The previous bounded checkpoint packet remains historical evidence for its own
source fingerprints. Neither GPU transfers proof to Metal or x86 tape execution.

Next concrete boundaries remain: shape-varying residual allocation with native
extent guards; broader while/CFG recovery beyond the proven counter envelope;
reader-complete asynchronous allocation retirement; GPU arbiter registration
and tuned ANN schedules; reduction/spectral/approximation error-budget consumers;
and measured promotion only where the independent bound passes. Assertions-enabled
MLIR validation remains missing on the fleet.

### 2026-09-07 — Shape-varying host tapes and GPU ANN arbitration

Owner: [W4-PRODUCT-1](INTEGRATED_COMPILER_PLAN.md#w4-product-1)

PRs: #734

Outcome: Original recorded scope and validation follow below.

Remaining: Historical obligations below route through the Owner; they do not set current priority.

Evidence: Original source, tests and packet references preserved below.

<!-- entry-fields:end -->

Owners: **F3 / FA-1 / AD-RESIDUAL-EVAL-1 / W2.4a / IR-NATIVE-FOUNDATION-1**.

Native Tessera-to-Linalg binary lowering now derives dynamic output extents
from the operand and checks unresolved operand/result equality before identity
indexing. Dynamic `zeros_like` adjoints use logical primal extents. The host JIT
accepts signature-checked i64/i8 residual buffers without widening its floating
high-level math API. Three shrinking-loop cases (input widths 4, 8, 16) execute
native x86 forward and repeated backward products with saved shape tapes and
unchanged persistent payloads. The host JIT now runs upstream ownership-based
deallocation after DPS result copies, including loop-carried temporary allocations.
Native extent violations currently lower through `cf.assert` to process abort,
not a recoverable Python exception. This is bounded shape-varying **host execution**;
CUDA/HIP persistent slot allocation still rejects dynamic extents.

Counter-bound recovery also recognizes the frontend's false-else short-circuit
`scf.if`, and equivalent `arith.select`. An arbitrary else value cannot prove
capacity. Arbitrary source CFG recovery, unbounded termination, dynamic device
slot allocation and generalized checkpoint selection remain open.

Terminal absolute value joins ReLU as a nonexpansive consumer of the affine
absolute-error bound. GPU candidates bind the logical domain and numerical
budget to both native artifact digests and the owning architecture. Scoped
registration retains the original under a zero rewrite budget and unregisters
its candidates only after successful binding close. Both GPUs execute all four
ReLU/absolute-value budget cases. The optional upstream elementwise-fusion
pipeline is serialized and replayed as part of artifact ancestry.

Nine fresh independent processes per GPU still refuse performance promotion:
CUDA median 0.99809x, lower bound 0.97773x; ROCm median 0.99728x, lower bound
0.98430x. Neither clears the 1.02x threshold. These are warm package wall times,
including transfers, for the small serial schedule; they neither measure kernel
clocks nor establish a tuned tensor-core/WMMA schedule. See
[raw reports and scope](../../../benchmarks/baselines/shape_tape_ann_20260907/README.md).

Fully asynchronous reclamation remains open: producer-event completion cannot
prove completion of readers of exported views. The next implementation needs
reader leases that prevent new acquisitions after retirement, stream-ordered
allocator/free support, and failure retention until all recorded readers finish.
The existing context barrier is retained. Further numerical consumers need
operator-specific induced-norm/error propagation; absolute value does not prove
reduction, spectral or approximation legality. Assertions-enabled MLIR validation
and Metal-owned tape storage remain required follow-ups.

**PR #734 review closure (F3):** `register_native_ann` returns a
`NativeANNRegistration`; use its `.region` inside a context manager or call
`.close()` when retiring a model/bucket. Close removes exact candidate instances,
invalidates retained candidates and releases owned probe references. Same-name
replacement ownership and registration rollback are tested without native tools.
Native-only tests now check compiler/JIT availability before preparing IR;
clean CI does not claim native execution. No new physical promotion is made.

### 2026-09-07 — bounded while recovery and row-parallel ANN

Owner: [NUMPOL-CARRIER-1](INTEGRATED_COMPILER_PLAN.md#numpol-carrier-1)

PRs: Not specified in this historical entry; no merge inferred.

Outcome: Original recorded scope and validation follow below.

Remaining: Historical obligations below route through the Owner; they do not set current priority.

Evidence: Original source, tests and packet references preserved below.

<!-- entry-fields:end -->

Owners: **W2.4a / AD-RESIDUAL-EVAL-1 / F3 / FA-1**. This increment remains
landing; it is not general CFG, GPU dynamic-storage, or reclamation closure.

- Data-dependent while normalization proves the trip capacity for nonnegative
  constant starts and positive constant strides (each at most 1024), with signed
  `<` or `<=` guards. It retains the original carried counter and refuses an
  insufficient declared maximum. The checkpoint envelope still requires a
  proven capacity of 2 through 1024. Both owning GPUs validate early exits and
  repeated reverse evaluations for a counter starting at 2 with stride 2.
- The native x86 JIT executes shape-varying while residuals for widths
  3, 4, 5, 8 and 16, covering zero through three iterations and two reverse
  evaluations without changing saved state. This connects existing logical
  shape tapes with while recovery; it supplies no GPU dynamic-allocation proof.
- A terminal fp32 row sum is an ANN numerical consumer. Its bound multiplies
  input error by the row width and adds a reduction gamma bound plus the
  existing subnormal allowance. Folded-parameter error receives the same
  width factor. Native execution and binding project the rank-one result;
  nonlinear-plus-reduction and other reduction axes remain outside this slice.
- Optional `parallel-ann-rows` checks actual load/store row independence for
  batches 2 through 64 before assigning one row to each GPU thread. Cross-row
  reads, input writes, unknown memory operations/aliases, outer loop-carried
  values and unmatched outer loops refuse.
  The launch shape and pipeline choice are serialized and replayed. Private
  temporary capacity remains conservative; this is not a tensor-core schedule.
  Both SM120 and gfx1151 validate ReLU, absolute value and row-sum admission,
  including zero-budget rewrite exclusion and scoped registry cleanup.

Next ordered dependencies:

1. GPU shape-varying slots need an allocation-capacity proof separate from the
   logical shape, plus bounds-checked residual reads in backward. Never infer
   GPU support from the host memref descriptor or replace dynamic extents with
   their maximum in logical copies.
2. Extend recovery to typed source CFG edges only with dominance, condition
   effects and carried-state proofs. Arbitrary multi-block/while recovery is
   still open; the new constant-stride envelope does not establish it.
3. Reader-complete asynchronous retirement needs scoped reader acquisition,
   prohibition of new readers after retirement, completion events on every
   reader stream and allocator-owned stream-ordered frees. Existing raw exported
   views must retain context synchronization. Event/allocator failures must
   retain owners, including partially queued frees. Producer completion alone
   cannot authorize reclamation. No reclamation implementation changed here.
4. Extend numerical consumers through explicit induced norms and intermediate
   overflow checks; reduction support does not establish spectral legality.
5. Compare serial and row schedules for the same native program, then introduce
   tiled target-owned schedules. Rewrite-versus-original package timings under
   the row schedule do not prove that row parallelism beats serial execution.
   Performance promotion remains disabled pending selector-grade evidence.

Validation and raw device packets are recorded under
`benchmarks/baselines/row_ann_while_20260907/`. LLVM assertions and architecture-
owned Apple tape storage remain required follow-ups.

### 2026-09-07 — dynamic temporary capacity and tracked reader retirement

Owner: [W4-PRODUCT-1](INTEGRATED_COMPILER_PLAN.md#w4-product-1)

PRs: Not specified in this historical entry; no merge inferred.

Outcome: Original recorded scope and validation follow below.

Remaining: Historical obligations below route through the Owner; they do not set current priority.

Evidence: Original source, tests and packet references preserved below.

<!-- entry-fields:end -->

Owners: **W2.4a / AD-RESIDUAL-EVAL-1 / IR-NATIVE-FOUNDATION-1**. Landing.

`NativeTapeToGPU` now reserves each internal dynamic allocation from an SSA
interval proof, retaining its actual logical extents in the memref view and
copy loops. Bounded iteration paths still own separate capacity-sized slots,
within the 4096-byte reservation limit. Dynamic copies require identical shape
SSA on source and destination allocations; loaded extents, dynamic external
arguments and unknown dynamic aliases refuse. Both SM120 and gfx1151 execute
logical widths 1/2/3 in distinct three-element slots, without writing unused
capacity. This closes a bounded internal-storage slice, not the exported
dynamic-shape tape ABI or arbitrary allocation envelopes.

Bounded native CFG recovery accepts typed `cf.switch` case/default edges in
addition to branches. It validates edge ABIs and preserves selected successor
state via structured conditionals. Native x86 forward/reverse tests cover two
cases and the default; the existing positive replay bound and pure-body rules
remain. General Python source-CFG recovery and effectful/unbounded CFGs remain
open. Native edge handling is not evidence that the frontend recovers all source
control flow.

`frame.backward_async(stream, seed, tracked=True)` uses native stream-ordered
pool allocations. The resulting generation lends views through `read(stream)`
scopes. Every scope closes with a recorded reader event; `retire(stream)` forbids
new readers and orders frees after the producer and all readers. `poll()` retires
owners after completion. Native tensor submissions reject a borrowed view on a
different stream. A borrowed view may only be submitted on its declared stream
before scope exit; caching/exporting its raw address bypasses this ownership API
and is outside the contract.

Healthy tracked retirement uses no context wait. Event-record failures retain a
stream completion fallback; partial frees are never resubmitted. Failure paths
may require explicit blocking wait/close. The parent frame's unrestricted
primal/residual exports and legacy derivative views retain context-synchronized
close. This is reader-complete asynchronous derivative retirement, not arbitrary
external-pointer reclamation. Tests cover two readers, scope invalidation,
retirement exclusion, failed event records and partially queued frees. Both GPU
hosts validate four exit paths, two readers and a third retirement stream while
forbidding a context barrier in the tracked region.

Evidence: `benchmarks/baselines/dynamic_readers_20260907/`. No throughput,
overlap, allocator-latency bound or cross-architecture performance claim is made.
Remaining priorities are dynamic external descriptor/status projection and saved
logical-shape validation, general source-CFG recovery, and adoption of scoped
reader ownership by composed consumers. Apple needs its own MSL allocation and
completion binding; x86 native CFG execution supplies no CUDA/HIP physical proof.


**Allocator-failure boundary:** a failed free quarantines the frame and retains
its owners for device teardown; normal close must not retry an ambiguously freed
pointer. Completion-event failures alone retain the explicit stream-wait recovery
path. Quarantine is intentionally not reclaimed by garbage collection.


**Verified CFG device blocker:** exported multiway products retain
`cf.assert ... "bounded native CFG exhausted max_steps"`. The GPU packager
currently refuses this guard. A native device status/termination consumer is
required before those products become executable; removing the assertion is not
an admissible migration. The dynamic temporary device packets validate storage
lowering separately from this still-host-only multiway AD product.

Pool/event ownership follows the ordering contracts in the
[CUDA stream-ordered allocator guide](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/stream-ordered-memory-allocation.html)
and [HIP stream-ordered allocator guide](https://rocmdocs.amd.com/projects/HIP/en/develop/how-to/hip_runtime_api/memory_management/stream_ordered_allocator.html).
The measured native cases include zero logical extents (widths 0/1/2), while
retaining positive physical capacity.


Cross-block SSA values are also owned state: dominating operation results and
foreign block arguments used by a successor are mapped to distinct state slots,
not left pointing into the erased CFG region. Native forward/reverse regressions
cover direct shared definitions across all switch cases and the default. Dynamic
saved values still require explicit shape envelopes for their new state slots.


ROCm already has a separate per-element state-machine status consumer in
`GenerateROCMStateMachineKernel.cpp`, covered by the irreducible-CFG execution
lane. The pending work is integrating/extending that status contract for the
persistent-product route, plus a CUDA counterpart; it is not a claim that ROCm
has no bounded CFG device execution. Preserve each route's shape and control
scope when reusing the status mechanism.

### 2026-09-07 — Checked persistent products and composed readers

Owner: [W2.4a](INTEGRATED_COMPILER_PLAN.md#w24a)

PRs: Not specified in this historical entry; no merge inferred.

Outcome: Original recorded scope and validation follow below.

Remaining: Historical obligations below route through the Owner; they do not set current priority.

Evidence: Original source, tests and packet references preserved below.

<!-- entry-fields:end -->

Owner: **W2.4a / AD-RESIDUAL-EVAL-1**; synchronization key:
**IR-NATIVE-FOUNDATION-1**. This increment remains a bounded slice.

- Native function-body CFGs can enter the same bounded recovery as imported
  `scf.execute_region` CFGs. Function arguments remain external SSA values;
  returns become region yields. Positive step bounds, identity, purity and
  supported typed edge/state requirements remain enforced. This is native MLIR
  source recovery, not arbitrary Python syntax or effectful CFG recovery.
- `tessera-native-tape-to-gpu`'s opt-in `status-buffer` appends a one-element
  i64 status pointer (`tessera.autodiff.gpu_status = "guard-v1"`). It initializes
  success, replaces top-level assertions with guarded suffixes, and reports 1
  on a failed guard. Subsequent memory operations do not execute on failure.
  Nested assertions refuse; the serial CUDA/HIP product boundary is unchanged.
- `materialize_persistent_tape(..., checked_status=True)` projects and verifies
  that physical status ABI on both products. Capture and synchronous backward
  consume status before exposing outputs. Status-enabled asynchronous backward
  deliberately refuses pending an asynchronous checked-result protocol.
- Tracked generations provide `submit_to` for native tensor consumers and
  `backward_into` for persistent reverse composition. Both delimit reader
  ownership across validation/enqueue, including exceptions. Synchronous tensor
  calls reject borrowed views because they cannot honor the declared stream.

**Exported shape-varying ABI remains open.** Internal dynamic temporaries and
static capacity/shape-tape residuals are not a runtime-sized public result ABI.
The next implementation must return logical extents alongside capacity-backed
result storage, with dtype/rank/capacity projected from serialized compiler IR.
Caller-supplied extents must not stand in for device-computed output shapes.
Validate each loaded extent before reconstructing a view or replaying a slice;
carry status through both shape validation and CFG exhaustion. A failed product
must not expose data or reusable shape metadata. Forward-owned shape residuals
must remain immutable across repeated reverse calls. Tests must cover zero
logical extents, exact capacity, overflow, mismatched cotangent shape and failure
before any out-of-bounds access. The current GPU external-slot parser continues
to refuse dynamic slots rather than fabricate these contracts.

General source CFG recovery additionally needs frontend-produced typed edges and
SSA merge values for break/continue/early returns, effect-aware replay and
shape-envelope propagation. Do not infer those from native multi-block support.
For asynchronous checked outputs, status must gate every downstream reader or
complete a checked host ticket before exposure; merely recording a producer
event proves completion, not success. General allocator/reclamation performance
and assertions-enabled MLIR validation remain open.

Owning-device packets for this increment are in
`benchmarks/baselines/product_status_20260907/`: RTX 5070/SM120 and Radeon
8060S/gfx1151 each pass six cases covering switch/repeated reverse, imported
function CFG, exhaustion refusal and scoped native consumer composition.
Checked status is synchronous; the stream composition case uses ordinary static
products. No timing, overlap or performance promotion is inferred.

### 2026-09-07 — Compiler plan role migration

Owner: [COMPILER-DEVEX-1](INTEGRATED_COMPILER_PLAN.md#compiler-devex-1)

PRs: Uncommitted documentation and governance increment.

Outcome: One grouped live queue, ID-only anchors, a reference log and archived wave provenance replace competing delivery-order sections. Detailed FA gates remain in their scoped reference. Capability and device support are unchanged.

Remaining: Follow the live queue; assertions-enabled compiler validation and the existing tool-execution gates remain open. Historical numerical/device claims have not been remeasured by this move.

Evidence: `scripts/check_compiler_plan.py`, `tests/unit/test_compiler_plan_routing.py` and existing audit/governance tests; 42 focused tests on Princess-Luna and eight routing tests on Super-Bear, plus the 30-document generated drift gate. No performance claim.

<!-- entry-fields:end -->

The routing index preserves successors and archived dispositions without copying
support statuses. CI checks structured references and requires an owner-record
or disposition update when adding an entry; the pre-push hook performs the
structural check. Current-priority links were separated from historical evidence
links, including the earlier math/capability reconciliation anchors.


### 2026-09-08 — Native unary ancestry and direct x86 migration

Owner: [E2E-REAL-6F](INTEGRATED_COMPILER_PLAN.md#e2e-real-6f)

PRs: Uncommitted follow-through after #735.

Outcome: x86 unary packages replay the native Schedule parent, verify native target attributes and project descriptor shapes, scalar extents and arithmetic policy before compilation; Apple reuses the bounded projection verifier.

Remaining: Other family ancestry and full route/envelope census; assertions-enabled LLVM/MLIR validation remains unavailable on the probed fleet builds.

Evidence: tests/unit/test_x86_unary_migration.py; tests/unit/test_scheduled_kernel_consumers.py; llvm-config --assertion-mode reports OFF on Princess-Luna and Super-Bear.

<!-- entry-fields:end -->

Forged dimensions/scalars, altered Tile dataflow and relabelled foreign parents
refuse before compilation. Host aliases remain explicit binding aliases rather
than claims about native SSA names. No device evidence transfers between targets.

### 2026-09-08 — Direct x86 unary caller migration

Owner: [E2E-REAL-6](INTEGRATED_COMPILER_PLAN.md#e2e-real-6)

PRs: Uncommitted follow-through after #735.

Outcome: Direct AVX-512 softmax and rank-reducing sum/mean/max route through the existing native Schedule/Tile producer and checked consumer.

Remaining: Baseline x86 and keepdims reductions retain their Graph constructors; other census families and frontend retirement are unchanged.

Evidence: tests/unit/test_x86_unary_migration.py direct runtime execution on Princess-Luna and native ancestry rejection tests.

<!-- entry-fields:end -->

The lower-level C ABI and native library implementation are unchanged. This is
package-authority migration, not new vectorization or a performance promotion.
The census still counts Graph-typed compatibility entry points; a migrated
branch does not erase its function's remaining fallback surface.

### 2026-09-08 — Checked asynchronous derivative tickets

Owner: [AD-RESIDUAL-EVAL-1](INTEGRATED_COMPILER_PLAN.md#ad-residual-eval-1)

PRs: Uncommitted follow-through after #735.

Outcome: Static checked GPU products enqueue backward asynchronously with an independent status allocation per generation; successful wait/poll is mandatory before exposing outputs.

Remaining: Exported logical shapes, loaded extents, nested guards, fully asynchronous allocation/reclamation and device-gated tracked readers remain open under AD-RESIDUAL-EVAL-1 and W2.4a.

Evidence: benchmarks/baselines/checked_derivatives_20260908/; tests/unit/test_checked_derivative_ticket.py; tests/unit/test_native_product_status.py.

<!-- entry-fields:end -->

[W2.4a](INTEGRATED_COMPILER_PLAN.md#w24a) retains scoped-reader ownership.
Failed status never exposes output and can still drain/release safely; frame
close drains failed submissions without treating status failure as a reason to
leak allocations. Unrestricted successful exports keep their context-completion
release barrier. Two owning-device generations and explicitly injected nonzero
status pass on SM120 and gfx1151; no overlap or performance claim is made.


### 2026-09-08 — Native keepdims reduction

Owner: [E2E-REAL-6](INTEGRATED_COMPILER_PLAN.md#e2e-real-6)

PRs: Uncommitted follow-through after #735.

Outcome: AVX-512 keepdims sum/mean/max now project output shapes and policy from native Schedule/Tile; descriptor keepdims forgery refuses before compilation.

Remaining: Baseline x86 and remaining census families still retain Graph-owned paths. No performance promotion.

Evidence: tests/unit/test_x86_unary_migration.py; direct native execution on Princess-Luna.

<!-- entry-fields:end -->


### 2026-09-08 — Logical results and nested device guards

Owner: [AD-RESIDUAL-EVAL-1](INTEGRATED_COMPILER_PLAN.md#ad-residual-eval-1)

PRs: Uncommitted follow-through after #735.

Outcome: Native rank-one capacity-buffer programs return checked device-written logical lengths. Nested for/if guards suppress the enclosing suffix and later loop iterations, including scalar loop-carried exits.

Remaining: This is an explicit buffer-program producer, not automatic dynamic tensor/AD result generation. Multidimensional logical shapes, arbitrary while regions and asynchronous public-result reclamation remain open.

Evidence: benchmarks/baselines/deep_native_20260908/; tests/unit/test_native_public_results.py; tests/unit/test_native_product_status.py.

<!-- entry-fields:end -->


### 2026-09-08 — Device-gated derivative readers

Owner: [W2.4a](INTEGRATED_COMPILER_PLAN.md#w24a)

PRs: Uncommitted follow-through after #735.

Outcome: CheckedDerivativeSubmission.backward_into enqueues a dependent backward with an event wait and native incoming-status gate before body effects; parent status need not return to the host first.

Remaining: Release still uses a context completion barrier, and ordinary checked outputs still require successful host-status consumption. This does not close fully asynchronous reader-aware reclamation or general device graphs.

Evidence: benchmarks/record_deep_native_contracts.py on SM120 and gfx1151; injected upstream failure refuses child output; no overlap claim.

<!-- entry-fields:end -->


### 2026-09-08 — Scoped measured ANN admission

Owner: [MSW-9](INTEGRATED_COMPILER_PLAN.md#msw-9)

PRs: Uncommitted follow-through after #735.

Outcome: Scoped native ANN registration binds nine independent reports to exact package images, source/recorder identity, target, domain and analytic budget before oracle-verified arbitration. Closing the registration retires its candidates.

Remaining: Package timing includes the host bridge and is not kernel timing. Broader nonlinear programs, tuning and global production promotion remain open.

Evidence: tests/unit/test_native_ann_measurement_owner.py; benchmarks/baselines/deep_ann_20260908/; benchmarks/select_native_ann_measurements.py.

<!-- entry-fields:end -->


### 2026-09-08 — Assertions-enabled native compiler

Owner: [COMPILER-DEVEX-1](INTEGRATED_COMPILER_PLAN.md#compiler-devex-1)

PRs: Uncommitted follow-through after #735.

Outcome: Pinned LLVM/MLIR 23.1.1 assertions-ON build and isolated Tessera consumer run on Super-Bear. An executable LLVM assertion probe aborts as expected. The new lane exposed and fixed NativeTapeToGPUPass's missing Tile dependent-dialect registration; the corrected compiler passes 257 focused checks plus 76 composed tape/ANN/storage/reader checks. The latter retain release downstream LLVM tools and host JIT.

Remaining: Full pass-corpus and installed-driver assertions lanes; other hosts continue using release toolchains. Upstream no-RTTI and Tessera compile flags must match. No GPU execution/performance claim follows from compiler validation.

Evidence: scripts/build_assertions_llvm.sh; scripts/probe_llvm_assertions.py; benchmarks/baselines/assertions_llvm_20260908/; tests/unit/test_native_public_results.py; tests/unit/test_native_product_status.py; native unary, pass metadata and diagnostic registry checks. Five target-tool tests skipped in the isolated build.

<!-- entry-fields:end -->


### 2026-09-08 — Baseline x86 unary ownership

Owner: [E2E-REAL-6](INTEGRATED_COMPILER_PLAN.md#e2e-real-6)

PRs: Uncommitted follow-through after #735.

Outcome: Baseline softmax/reduction now project from native Schedule/Tile into their own scalar C ABI and image, with no AVX-512 feature requirement. Direct runtime tests cover both baseline families.

Remaining: Matmul/attention, cohort and breadth routes remain Graph-owned; frontend retirement is unchanged.

Evidence: tests/unit/test_x86_unary_migration.py; tests/unit/test_x86_e2e_spine.py on Princess-Luna.

<!-- entry-fields:end -->


### 2026-09-08 — Automatic AD public result generation

Owner: [AD-RESIDUAL-EVAL-1](INTEGRATED_COMPILER_PLAN.md#ad-residual-eval-1)

PRs: Uncommitted follow-through after #735.

Outcome: NativeTapeToGPUPass generates a checked capacity copy and logical-length sidecar from a rank-one exported AD result. Integer select alternatives extend SSA allocation bounds. Python invokes native export/bufferization rather than synthesizing result-copy math.

Remaining: The dynamic forward slice uses static external inputs and one rank-one result; dynamic GPU backward inputs and saved multi-result/multidimensional products remain open.

Evidence: tests/unit/test_automatic_ad_public_results.py; benchmarks/baselines/automatic_ad_retirement_20260908/.

<!-- entry-fields:end -->


### 2026-09-08 — Checked generation asynchronous reclamation

Owner: [W2.4a](INTEGRATED_COMPILER_PLAN.md#w24a)

PRs: Uncommitted follow-through after #735.

Outcome: Checked derivative data and status use stream-ordered pool allocation/free. Device-gated backward readers register completion before retirement; optional generic scoped reads require explicit successful host status. Successful polling cleans up events without a redundant wait, and failed destruction retains a monotonic completion proof for safe retry.

Remaining: Whole-frame capture/close and exceptional teardown remain synchronous. Uncertain free submissions quarantine ownership. This is not general external-reader tracking or measured overlap.

Evidence: tests/unit/test_native_reader_retirement.py; tests/unit/test_native_gpu_streams.py; benchmarks/record_checked_retirement.py on SM120 and gfx1151.

<!-- entry-fields:end -->


### 2026-09-08 — Independently tuned ANN rewrite

Owner: [MSW-9](INTEGRATED_COMPILER_PLAN.md#msw-9)

PRs: Uncommitted follow-through after #735.

Outcome: Native source replay now allows a fused row-parallel transformed candidate against an unchanged serial incumbent. Nine independent 16x8 runs yield scoped SM120 selection (median 1.0753x; interval [1.0374,1.0883]); gfx1151 retains the incumbent (median 0.9997x; interval [0.9657,1.0170]).

Remaining: Evidence is warm host-wall package timing for this frozen affine/ReLU workload, not kernel timing, arbitrary nonlinear closure or global promotion. Registrations retire after use.

Evidence: benchmarks/baselines/tuned_ann_20260908/; benchmarks/record_native_ann_execution.py; benchmarks/select_native_ann_measurements.py.

<!-- entry-fields:end -->


### 2026-09-08 — Runtime-shaped AD products

Owner: [AD-RESIDUAL-EVAL-1](INTEGRATED_COMPILER_PLAN.md#ad-residual-eval-1)

PRs: Uncommitted follow-through after #735.

Outcome: Native export/bufferization now binds dynamic GPU backward inputs with checked physical capacities and per-axis shape sidecars. Multiple rank-one through rank-four floating outputs use independent flat storage; native guards validate each extent and total volume. GPU result submission exposes no logical view until event completion and successful status/shape consumption.

Remaining: Exact-device proof covers rank-one lengths 0/2/4, dynamic rank-two shapes, malformed/mismatched inputs and two matrix outputs. General saved mixed products, arbitrary layouts/aliases and automatic Python capture remain open.

Evidence: benchmarks/baselines/runtime_shape_frames_20260908/; tests/unit/test_automatic_ad_public_results.py; tests/unit/test_native_public_results.py.

<!-- entry-fields:end -->


### 2026-09-08 — Scoped whole-frame retirement

Owner: [W2.4a](INTEGRATED_COMPILER_PLAN.md#w24a)

PRs: Uncommitted follow-through after #735.

Outcome: Opt-in scoped capture uses pool storage and reader leases for primals/residuals. Frame retirement enqueues every generation and frame free after registered readers; successful polling releases idle modules without an explicit context wait. Active readers refuse retirement and uncertain frees retain quarantine.

Remaining: Scoped persistent frames still use static tensor shapes; runtime-shaped public-result frames retain synchronous close. Capture and exceptional cleanup remain synchronous. Module unload has no bounded-latency claim. Existing unrestricted views still require synchronous close; general external reader adoption and measured overlap remain open.

Evidence: benchmarks/baselines/runtime_shape_frames_20260908/; tests/unit/test_native_reader_retirement.py; tests/unit/test_native_gpu_streams.py.

<!-- entry-fields:end -->


### 2026-09-08 — x86 f32 matmul native ownership

Owner: [E2E-REAL-6](INTEGRATED_COMPILER_PLAN.md#e2e-real-6)

PRs: Uncommitted follow-through after #735.

Outcome: Direct f32 matmul now enters native Graph-to-Schedule-to-Tile lowering. Packaging replays the Schedule parent and validates projected fields before target compilation; descriptor shape forgery refuses.

Remaining: Non-f32 matmul, attention, cohort and breadth constructors remain scoped F2 work. This changes ownership, not the runtime kernel or performance promotion.

Evidence: tests/unit/test_x86_unary_migration.py; tests/unit/test_x86_e2e_spine.py.

<!-- entry-fields:end -->


### 2026-09-08 — Broader ANN workload evidence

Owner: [MSW-9](INTEGRATED_COMPILER_PLAN.md#msw-9)

PRs: Uncommitted follow-through after #735.

Outcome: The measurement workload generator and scoped replay admit terminal absolute value alongside ReLU. Fresh 8x4 absolute-value and 32x4 ReLU comparisons retain independently replayed incumbent/rewrite schedules, analytic domains and finite error budgets. Nine quiet runs per workload and target retain the incumbent in all four comparisons; no lower confidence bound clears the 2% margin.

Remaining: Use the revision-bound packets for selection outcomes. Host-wall package timing is not kernel timing or global promotion. Nonexpansive terminal consumers do not authorize moving nonlinearities across affine composition; arbitrary nonlinear graphs remain open.

Evidence: benchmarks/baselines/broad_ann_20260908/; benchmarks/record_native_ann_execution.py; benchmarks/select_native_ann_measurements.py.

<!-- entry-fields:end -->


### 2026-09-08 — Python CFG boundary review

Owner: [W4-PRODUCT-1](INTEGRATED_COMPILER_PLAN.md#w4-product-1)

PRs: Uncommitted follow-through after #735.

Outcome: Source review confirms TraceRef.__bool__ rejects raw value-dependent Python if/while; explicit tessera.control regions and imported native bounded CFGs are the current consumed paths. Nested native status guards now also protect dynamic input shapes and multiple output shape copies.

Remaining: Arbitrary Python source CFG remains open: tracer-owned typed merge values, break/continue/early return, effect legality and source-location-preserving recovery require a frontend producer. Legacy AST markers are not execution proof.

Evidence: python/tessera/compiler/trace.py; python/tessera/compiler/structured_cfg.py; tests/unit/test_trace_f4.py; tests/unit/test_native_public_results.py; docs/audit/compiler/AUTODIFF_EXECUTION_PLAN.md.

<!-- entry-fields:end -->

### 2026-09-08 — Source CFG and asynchronous ownership

Owner: [W4-PRODUCT-1](INTEGRATED_COMPILER_PLAN.md#w4-product-1)

PRs: Uncommitted engineering continuation after #736.

Outcome: Opt-in pure Python branch/early-return and single-carry bounded while recovery feeds typed tracer SSA and native SCF. Scoped dynamic public frames and asynchronous persistent capture use checked status, reader leases and event-ordered retirement. Module unload admission is bounded and polling does not wait for the driver. Dominating exact-product guards bound joint temporary volume. x86 forward attention consumes scheduled artifacts; terminal-square ANN has an analytic error consumer.

Remaining: Effectful/arbitrary Python CFG, multiple loop carries, automatic general JIT wiring, multi-status asynchronous composition, exceptional cleanup, driver cancellation, remaining Graph families and broader nonlinear kernels remain open. Bounded worker admission is not a bound on driver unload latency. GPU correctness is not performance promotion.

Evidence: `benchmarks/baselines/source_async_foundation_20260908/`; focused source, ownership, joint-volume, ancestry and nonlinear tests. NVIDIA's nine-run square comparison retained the incumbent. See the evidence README for independent host outcomes and measured scope.

<!-- entry-fields:end -->

Additional owners: [AD-RESIDUAL-EVAL-1](INTEGRATED_COMPILER_PLAN.md#ad-residual-eval-1),
[W2.4a](INTEGRATED_COMPILER_PLAN.md#w24a),
[E2E-REAL-6](INTEGRATED_COMPILER_PLAN.md#e2e-real-6), and
[MSW-9](INTEGRATED_COMPILER_PLAN.md#msw-9).
Synchronization key: `SOURCE-ASYNC-FOUNDATION-2026-09-08`.

The source producer rejects effect-only calls, mutation, break/continue and
implicit numeric tensor truth. Native exhaustion asserts rather than silently
truncating. Logical dynamic shapes retain independent sidecars even when an
exact dominating product guard permits a smaller flat physical allocation.

Successful asynchronous retirement covers data, shape and status storage plus
registered readers. Generic capture readers require successful `poll_capture`;
the single incoming-status ABI cannot replace an unchecked capture status with
an external dependency. Unload workers retain their context and admission slot
on failure or stall. Explicit retirement/polling is the asynchronous API;
exceptional cleanup and unrestricted legacy views remain synchronous.

x86 attention now replays Schedule-to-Tile ancestry and projects native fields
through a shared contract. The extended route supplies the runtime's symmetric
`window` field. Non-f32 matmul, backward attention/cohorts and generic Graph
families are not retired by this migration. Square ANN propagates folded
parameter error through the nonlinear tail; route admission still requires
independent exact-device measurement and successful retirement of scoped candidates.

### 2026-09-08 — Effect-aware CFG and status composition

Owner: [W4-PRODUCT-1](INTEGRATED_COMPILER_PLAN.md#w4-product-1)

PRs: Uncommitted continuation after #736.

Outcome: Bounded break/continue and multi-variable source loops merge iteration state, native assertions preserve effects, and dead-path unmodelled calls refuse. Two serialized native status inputs preserve capture plus upstream failure through checked asynchronous composition. The CAKE enhancement review now maps lessons to current consumers and owners.

Remaining: Loop-return payloads, external mutation/alias contracts, arbitrary Python CFG, wider status joins, exceptional recovery and general JIT integration. This bounded expansion is not general source recovery; host assertion failure still uses the LLVM abort ABI.

Evidence: `benchmarks/baselines/cfg_status_composition_20260908/`, source/compiler negative tests and independent SM120/gfx1151 four-case status truth tables.

<!-- entry-fields:end -->

Additional owner: [W2.4a](INTEGRATED_COMPILER_PLAN.md#w24a).
Sync key: `CFG-STATUS-COMPOSITION-2026-09-08`.
The first expansion prototype grew exponentially; per-iteration merges make
successive iterations linear in this fixed body. Native multi-result conditional
traces now retain the full inferred result types. Distinct status storage avoids
relaxing the native package's alias contract. `compiler_enhancement.md` preserves
its August assessment as historical provenance and corrects unsupported current
claims about phase readiness, pruning regret and fixed economic thresholds.

### 2026-09-08 — Completion state and status fan-in

Owner: [W4-PRODUCT-1](INTEGRATED_COMPILER_PLAN.md#w4-product-1)

PRs: Uncommitted continuation after #736.

Outcome: Bounded nested loops carry tuple return payloads and statically resolved builtin exception completions. Explicit CPU state slots preserve exact full-tensor input aliases through serialized state results and post-completion copyback. Incoming checked statuses support counts one through eight, retaining reader leases for every prerequisite.

Remaining: Arbitrary objects, partial/strided mutable views, mutable return aliases, changing loop-state types, uncaught exception transport, general JIT integration, GPU mutation and heterogeneous/unbounded asynchronous effect joins. Eight-status device execution is not claimed by the four-status packet.

Evidence: `benchmarks/baselines/completion_state_fanin_20260908/`; native CPU differential tests and independent CUDA SM120/ROCm gfx1151 sixteen-case status truth tables.

<!-- entry-fields:end -->

Additional owner: [W2.4a](INTEGRATED_COMPILER_PLAN.md#w24a).
Sync key: `COMPLETION-STATE-FANIN-2026-09-08`.
The source producer handles explicit ValueError, RuntimeError and AssertionError
without creating Python exception objects. Finally continuations preserve or
override the pending completion. Native assertion exhaustion still uses the host
abort ABI. State execution requires exclusive host ownership; copyback is not an
atomic transaction against concurrent external readers. Device status-only
prerequisites add gates and reader lifetimes, not extra cotangent operands.

### 2026-09-08 — Source JIT error transport and device state

Owner: [W4-PRODUCT-1](INTEGRATED_COMPILER_PLAN.md#w4-product-1)

PRs: Uncommitted continuation after #736.

Outcome: CPU source state accepts non-overlapping strided views and exact aliases. Explicit floating result specs transport builtin exception classes, preserving prior writes. Opt-in public JIT owns at most four native shape/alias specializations. ROCm executes two immutable source-state generations and all 256 eight-status combinations.

Remaining: Arbitrary object/overlapping-view mutation, implicit/dynamic exception transport and original messages, broader JIT/AD integration, in-place device state mutation and NVIDIA eight-status/device-state proof. Super-Bear SSH was unavailable for this increment; Apple needs a separate MSL consumer.

Evidence: `benchmarks/baselines/source_jit_state_20260908/`, focused native CPU tests and independent gfx1151 packets. No performance promotion.

<!-- entry-fields:end -->

Additional owner: [W2.4a](INTEGRATED_COMPILER_PLAN.md#w24a).
Synchronization key: `SOURCE-JIT-STATE-2026-09-08`.
The sequence mixer theory now distinguishes exact algebra, numerical legality,
state ownership and physical proof. NoPE alone does not justify MQA conversion;
scalar SSD is not generic DPLR, and zero/underflowing decay invalidates division.

### 2026-09-09 — Declared object state and owned GPU mutation

Owner: [W4-PRODUCT-1](INTEGRATED_COMPILER_PLAN.md#w4-product-1)

PRs: Uncommitted continuation after #736.

Outcome: Declared dict/SimpleNamespace tensor fields project to native arguments and full-slice state writes. Read-only overlapping arrays use snapshots. Static builtin exception args, inherited/tuple handlers and bare re-raise, including pending exceptions in finally, survive native completion transport. Pure tensor source JIT exposes explicit compiler-exported CPU VJP. Exclusive owned GPU state reuses its allocation after synchronous checked computation; scoped readers prevent writes.

Remaining: Custom accessors and object identity/rebinding; overlapping writes; implicit/dynamic exception objects, chaining and traceback semantics; automatic effectful JIT/AD; asynchronous in-place writes and external borrowed storage. The GPU implementation computes into fresh output storage then copies back: this is allocation identity and correctness evidence, not a fused mutation kernel or performance promotion.

Evidence: `benchmarks/baselines/source_object_ownership_20260909/`, focused native CPU regressions and independent SM120/gfx1151 recorder packets.

<!-- entry-fields:end -->

Additional owners: [W2.4a](INTEGRATED_COMPILER_PLAN.md#w24a), [AD-RESIDUAL-EVAL-1](INTEGRATED_COMPILER_PLAN.md#ad-residual-eval-1).
Synchronization key: `SOURCE-OBJECT-OWNERSHIP-2026-09-09`.
Writable overlapping views require one shared backing-storage SSA root, typed view maps and ordered updates. Independent input snapshots cannot preserve interleaved alias writes, so those writes still refuse before execution. VJP derivatives are per formal tensor input; this does not differentiate mutable object state.

### 2026-09-09 — Mixed aliases and asynchronous state

Owner: [W4-PRODUCT-1](INTEGRATED_COMPILER_PLAN.md#w4-product-1)

PRs: Uncommitted follow-through after #737.

Outcome: Read-only partial overlaps remain admissible beside disjoint mutable state at compilation, cache selection and execution. Plain custom instances project dictionary fields without custom accessors. Explicit CPU VJP differentiates functional results plus declared next-state outputs without copyback. Single-stream GPU submit/poll validates computation before event-ordered copyback and excludes readers until completion; failures poison the owner and retain pending storage.

Remaining: Arbitrary accessors/object rebinding, overlapping writes with common backing-storage SSA, dynamic/implicit exception payloads and chaining, object/exception AD, aliased state VJP, multi-writer mutation and asynchronous teardown of failed updates. Close may synchronize; no full exception or asynchronous lifecycle closure is claimed.

Evidence: `benchmarks/baselines/mixed_alias_async_state_20260909/`; native source regressions, injected pending/failure ownership tests and independent SM120/gfx1151 asynchronous state packets. No performance promotion.

<!-- entry-fields:end -->

Additional owners: [W2.4a](INTEGRATED_COMPILER_PLAN.md#w24a), [AD-RESIDUAL-EVAL-1](INTEGRATED_COMPILER_PLAN.md#ad-residual-eval-1).
Synchronization key: `MIXED-ALIAS-ASYNC-STATE-2026-09-09`.

### 2026-09-09 — Writable views and exception value transport

Owner: [W4-PRODUCT-1](INTEGRATED_COMPILER_PLAN.md#w4-product-1)

PRs: Uncommitted continuation after #737.

Outcome: Contiguous rank-one writable aliases share an explicit containing-input SSA root; standard tensor.extract_slice/insert_slice preserve ordered writes and subsequent reads. Runtime revalidates view offsets and cache identity includes the view map. Single-element f32 exception-value payloads cross native completion, loop exits and finally; nested re-raise restores the outer payload. Explicit native VJP projects declared object fields and differentiates public/next-state values without mutation.

Remaining: Writable views without a supplied containing input, noncontiguous/multidimensional write maps, aliased state adjoints, dynamic strings/heterogeneous exception values, cause/context/traceback transport and implicit operation errors. Mutable exception aliases refuse rather than silently changing finally semantics. Exception AD and GPU view/exception execution remain open.

Evidence: `benchmarks/baselines/source_views_exception_20260909/`; native CPU execution regressions and independent SM120/gfx1151 regression packets for the existing single-state asynchronous consumer. Device packets do not prove writable view or exception execution. No performance promotion.

<!-- entry-fields:end -->

Additional owners: [AD-RESIDUAL-EVAL-1](INTEGRATED_COMPILER_PLAN.md#ad-residual-eval-1), [W2.4a](INTEGRATED_COMPILER_PLAN.md#w24a).
Synchronization key: `SOURCE-VIEW-EXCEPTION-2026-09-09`.

### 2026-09-09 — Strided roots static causes and device execution

Owner: [W4-PRODUCT-1](INTEGRATED_COMPILER_PLAN.md#w4-product-1)

PRs: Uncommitted continuation after #737.

Outcome: Positive-stride rank-one views and local static slices project onto an explicit contiguous root. Exact-alias state VJP accumulates at that root and returns zero for duplicate alias arguments. Explicit literal builtin causes and from-None suppression cross the native completion boundary. GPU bufferization copies before writes; the native ownership verifier resolves bounded known subview/cast chains to private allocations or output roots and continues rejecting input-rooted writes.

Remaining: Negative/multidimensional views, nontrivial slice adjoints, general caught exception identity/context/tracebacks, dynamic causes, exception AD and GPU exception transport. Copy-before-write can increase temporary storage and does not establish performance eligibility. Local one-root GPU slices do not admit arbitrary external aliased device pointers.

Evidence: `benchmarks/baselines/strided_source_gpu_20260909/`; focused source/ownership tests, compiler rebuilds, and independently measured strided asynchronous state updates. No performance promotion.

<!-- entry-fields:end -->

Additional owners: [AD-RESIDUAL-EVAL-1](INTEGRATED_COMPILER_PLAN.md#ad-residual-eval-1), [W2.4a](INTEGRATED_COMPILER_PLAN.md#w24a).
Synchronization key: `STRIDED-SOURCE-GPU-2026-09-09`.

### 2026-09-09 — Mapped adjoints and GPU exception completion

Owner: [W4-PRODUCT-1](INTEGRATED_COMPILER_PLAN.md#w4-product-1)

PRs: Uncommitted continuation after #737.

Outcome: Bounded injective negative/multidimensional views lower through standard tensor slices. Native slice adjoints accumulate overlapping reads at the containing root and mask overwritten destination gradients. Static caught bindings retain identity through re-raise; explicit causes and implicit contexts decode as an exception graph, with logical native source notes. SM120 and gfx1151 independently execute mapped forward/backward and checked synchronous/asynchronous static/dynamic exception cases. Failed public frames remain unreadable and repeated polls retain the same error object. Block AttnRes now routes these foundations into native package, lifetime and performance gates; its incorrect full-matrix-rank liveness claim is removed.

Remaining: Runtime-sized/rank-changing or large mapped views, noninjective writes, general object mutation, loop exception-context slots, retained dynamic context payloads, real Python traceback frames and exception AD. Owned in-place GPU mutation remains exception-free. Multi-state device aliases, tuned cooperative kernels and performance promotion are separate.

Evidence: `benchmarks/baselines/source_exception_gpu_20260909/`; native source/ownership regressions, finite-difference view adjoints, Block AttnRes rank counterexample, independent device packets.

<!-- entry-fields:end -->

Additional owners: [AD-RESIDUAL-EVAL-1](INTEGRATED_COMPILER_PLAN.md#ad-residual-eval-1), [W2.4a](INTEGRATED_COMPILER_PLAN.md#w24a), [Block AttnRes](BLOCK_ATTNRES_ROCM_PLAN.md).
Synchronization key: `SOURCE-MAPPED-EXCEPTION-2026-09-09`.

### 2026-09-09 — Runtime maps context slots and cooperative experiment

Owner: [W4-PRODUCT-1](INTEGRATED_COMPILER_PLAN.md#w4-product-1)

PRs: Uncommitted continuation after #737.

Outcome: Positive rectangular source maps use compact native slices; CPU VJP no longer inherits a GPU slot cap. Per-site dynamic exception payload slots retain cause/context values; CPU VJP validates forward status before backward and zero-seeds completion metadata. Runtime-shaped native slice products execute across shapes on SM120/gfx1151, with exact cotangent-shape assertions, nonnegative unsigned-division allocation bounds and dominating equality proofs for logical copies. Block AttnRes has an opt-in cooperative width reduction, distinct cache/compiler identity and serialized pipeline policy; measured operation-total results do not establish promotion.

Remaining: Automatic runtime Python slice capture, large/general negative maps, generation-indexed loop context storage, full CPython frames/tracebacks and product-aware GPU exception AD. Cooperative Block AttnRes needs device-clock attribution, broader numerical/shape coverage and repeatable package wins; CUDA depth-attention packaging remains separate.

Evidence: `benchmarks/baselines/runtime_source_maps_20260909/`; CPU source/AD, artifact and registry tests; independent twenty-case GPU packets; default/cooperative gfx1151 package measurements.

<!-- entry-fields:end -->

Additional owners: [AD-RESIDUAL-EVAL-1](INTEGRATED_COMPILER_PLAN.md#ad-residual-eval-1), [W2.4a](INTEGRATED_COMPILER_PLAN.md#w24a), [Block AttnRes](BLOCK_ATTNRES_ROCM_PLAN.md).
Synchronization key: `RUNTIME-SOURCE-MAPS-2026-09-09`.


### 2026-09-09 — Source index generations and checked GPU VJP

Owner: [W4-PRODUCT-1](INTEGRATED_COMPILER_PLAN.md#w4-product-1)

PRs: Uncommitted continuation after #737.

Outcome: Python source slices lower runtime int64 tensor bounds and positive steps to native arithmetic and dynamic slice results. Read-only view roots retain their SSA value across local rebinding. Bounded expanded loops carry one completion code and distinct per-generation exception payload slots; nested cause/context chains no longer depend on a frozen pre-loop edge table. Native public results preserve scalar i8/i64 residual types. A paired GPU VJP owns device input snapshots, checks forward exceptions before backward and zero-seeds completion metadata. Repeated failed-frame polling preserves exception identity without accumulating old host traceback chains.

Remaining: Python integer/index protocols, negative runtime strides, nested/runtime-shaped source roots and dynamic CPU JIT result allocation; cross-iteration arbitrary exception objects and unbounded storage; full CPython traceback frames/locals; asynchronous exception-aware AD. Block AttnRes promotion is unchanged.

Evidence: `benchmarks/baselines/source_generation_ad_20260909/`; native CPU and independent SM120/gfx1151 slice, generation and checked-VJP cases. No performance promotion.

<!-- entry-fields:end -->

Additional owners: [AD-RESIDUAL-EVAL-1](INTEGRATED_COMPILER_PLAN.md#ad-residual-eval-1), [W2.4a](INTEGRATED_COMPILER_PLAN.md#w24a).
Synchronization key: `SOURCE-GENERATION-AD-2026-09-09`.


### 2026-09-09 — Signed nested views and asynchronous source VJP

Owner: [W4-PRODUCT-1](INTEGRATED_COMPILER_PLAN.md#w4-product-1)

PRs: Uncommitted continuation after #737.

Outcome: Signed runtime slice maps compose nested rank-preserving views against an immutable root and lower through tensor.generate, including INT64_MIN steps and empty results. The CPU JIT uses compiler-projected capacity/shape sidecars to allocate multiple multidimensional dynamic outputs; Python integer bounds remain runtime inputs. Identity memref arguments receive Python and C static-extent checks. Bounded loop-carried exception references retain their original generation payload and explicit cause. Same-stream GPU snapshots and checked asynchronous VJP stage backward only after a successful forward, retaining matching residuals until completion.

Remaining: Runtime-shaped original roots, custom index protocols, runtime gather adjoints, arbitrary custom exception heaps and unbounded generations; full CPython frames with live locals/closures and instruction identity; fully asynchronous retirement and cross-queue ownership. CPU automatic capacity is bounded to 1024 elements. Device source VJP remains single-input without projected state aliases/object fields. No performance promotion.

Evidence: `benchmarks/baselines/source_nested_async_20260909/`; 19 independently measured cases each on SM120 and gfx1151, host source/ABI tests and assertions-enabled compiler validation.

<!-- entry-fields:end -->

Additional owners: [AD-RESIDUAL-EVAL-1](INTEGRATED_COMPILER_PLAN.md#ad-residual-eval-1), [W2.4a](INTEGRATED_COMPILER_PLAN.md#w24a).
Synchronization key: `SOURCE-NESTED-ASYNC-2026-09-09`.


### 2026-09-09 — Gather transposes and scoped source retirement

Owner: [W4-PRODUCT-1](INTEGRATED_COMPILER_PLAN.md#w4-product-1)

PRs: Uncommitted continuation after #738.

Outcome: Index-only tensor.generate gathers transpose to serial nested scatter-add loops with exact seed-shape assertions, preserving repeated-index accumulation. CPU source JIT retains explicitly captured custom exception class bindings and invokes their constructors only on failed completion. Scoped source VJP records backward as a reader of forward snapshots/residuals; retirement enqueues frees after closed reader scopes and queries completion without a context wait on the healthy path.

Remaining: Arbitrary exception object heaps/attributes in native execution, GPU custom-class bindings, nonlinear generator adjoints, source-level dynamic VJP output allocation, fully asynchronous module unloading/failure recovery and full CPython frames. Native source notes remain distinct from interpreter frames and locals. No performance promotion.

Evidence: Native CPU gather/source regressions; independent 19-case SM120/gfx1151 packets in `benchmarks/baselines/source_scoped_ad_20260909/`.

<!-- entry-fields:end -->

Additional owner: [AD-RESIDUAL-EVAL-1](INTEGRATED_COMPILER_PLAN.md#ad-residual-eval-1).
Synchronization key: `SOURCE-GATHER-SCOPED-2026-09-09`.


### 2026-09-09 — Exception class bindings and unload cleanup recovery

Owner: [W4-PRODUCT-1](INTEGRATED_COMPILER_PLAN.md#w4-product-1)

PRs: Uncommitted continuation after #738.

Outcome: Expanded raise sites now carry occurrence/generation identities even for identical literal payloads. GPU source VJP accepts explicitly owned host exception-class bindings and validates missing bindings before compilation; failed completion constructs the host exception once, including repeated polling. Actual host constructor/completion traceback frames survive repeated polls without accumulating polling frames. Module retirement separates confirmed unload/context exit from filesystem cleanup. Only post-unload cleanup failures admit off-thread retry through native tensor, public-result and source-product owners; uncertain driver outcomes retain their quarantine and admission slot.

Remaining: Unbounded exception heaps, arbitrary object-field mutation/arguments in native execution, custom constructor effects inside handled source paths, true CPython frame reconstruction, cancellation of stalled driver unload and recovery from uncertain free/unload/context failures. No performance promotion.

Evidence: `benchmarks/baselines/source_exception_bindings_20260909/`; CPU identity and failure-injection tests plus independent SM120/gfx1151 custom-class completion and scoped-retirement packets.

<!-- entry-fields:end -->

Additional owner: [AD-RESIDUAL-EVAL-1](INTEGRATED_COMPILER_PLAN.md#ad-residual-eval-1).
Synchronization key: `SOURCE-EXCEPTION-BINDINGS-2026-09-09`.

### 2026-09-09 — Architecture sweep and failure-boundary reconciliation

Owner: [FRONTEND-IR-MEDIUM-1](INTEGRATED_COMPILER_PLAN.md#frontend-ir-medium-1)

PRs: Uncommitted engineering increment.

Outcome: Reconciled the historical sweep into live owner routes; structural shape compatibility replaces string identity and pipeline validation rejects missing stages. Selected exception graphs validate before host constructors; unload failures survive a second context-exit failure with ownership retained.

Remaining: General nonlinear constraint solving, automatic distributed placement, arbitrary exception heaps, native CPython frames and uncertain driver recovery remain open. No new device or performance promotion.

Evidence: Super-Bear host WSL: 228 passed, 1 skipped across source/shape/distributed/lifetime/audit regressions; Ruff passed and mypy checked 305 files. Princess-Luna host WSL: 18 shared regression/fault-injection cases passed independently. This is not new GPU execution evidence.

<!-- entry-fields:end -->

Additional owners: [DIST-NATIVE-1](INTEGRATED_COMPILER_PLAN.md#dist-native-1),
[W4-PRODUCT-1](INTEGRATED_COMPILER_PLAN.md#w4-product-1),
[W2.4a](INTEGRATED_COMPILER_PLAN.md#w24a).
Sync: `ARCH-SWEEP-FAILURE-2026-09-09`.
See [reconciliation](COMPILER_ARCHITECTURE_SWEEP.md#current-reconciliation-2026-09-09).
Previous source-exception packets retain their original fingerprints; they are
historical evidence and are not re-labelled as validation of this changed tree.

### 2026-09-09 — Indexed exception completion and one-shot unload

Owner: [W4-PRODUCT-1](INTEGRATED_COMPILER_PLAN.md#w4-product-1)

PRs: Uncommitted engineering increment.

Outcome: Source IR emits indexed exception objects, runtime validates and materializes shared/cyclic identities iteratively, and synchronous unknown unload/sync outcomes retain owners and refuse repeat driver actions.

Remaining: Arbitrary runtime native heap allocation, full CPython deoptimization/frame materialization and isolation-based recovery remain open. Current source generations remain bounded; no performance promotion.

Evidence: Super-Bear host WSL: 201 passed, 1 skipped; Princess-Luna: 18 heap/retirement tests passed. Ruff and mypy (306 files) pass. `benchmarks/baselines/source_exception_heap_20260909/` records 19 cases each on SM120 and gfx1151; this is completion-carrier correctness, not native heap allocation or destructive driver recovery.

<!-- entry-fields:end -->

Additional owner: [W2.4a](INTEGRATED_COMPILER_PLAN.md#w24a).
Sync: `SOURCE-HEAP-RETIRE-2026-09-09`.

### 2026-09-09 — Completion reclamation and broader assertions corpus

Owner: [COMPILER-DEVEX-1](INTEGRATED_COMPILER_PLAN.md#compiler-devex-1)

PRs: Uncommitted engineering increment.

Outcome: Broader lit execution found an undeclared Tile dialect dependency in AutodiffForwardPass; declaring it fixes the assertions-build abort. Fifteen x86/Apple fixtures now declare the optional backend they consume, so omitted components report unsupported rather than misleading compiler failures. Schedule-to-Tile declares its NVIDIA Target dialect dependency instead of loading it during pass execution. A hardware-free all-target build can explicitly retain the full compiler driver; ROCm translation registration follows the serialization libraries actually linked. The resulting assertions build passes every active fixture, and the opt-in CI lit lane requests the same portable matrix. Successful public completion retirement drops cached exception/traceback roots without mutating caller-owned errors. Unknown synchronous frees retain frames and refuse retry.

Remaining: Structural assertions coverage is closed. Add installed-driver smoke; keep native Apple, NVIDIA and ROCm device correctness and performance under their owning evidence gates. General native heap allocation, complete CPython frame state and isolation-based driver recovery remain open.

Evidence: 428 passed, 1 skipped in host WSL unit gates. Super-Bear assertions-enabled LLVM/MLIR 23.1.1 hardware-free all-target lane: 474 passed, 0 unsupported and 0 failed with CUDA/HIP runtime integration disabled. `scripts/check_lit_fleet_union.py` reports 474/474 active fixtures covered with no unexpected results. `benchmarks/baselines/source_completion_retirement_20260909/` retains the reports and exact compiler/source hashes. Ruff and mypy (306 files) pass.

<!-- entry-fields:end -->

Additional owners: [E2E-REAL-6F](INTEGRATED_COMPILER_PLAN.md#e2e-real-6f),
[W4-PRODUCT-1](INTEGRATED_COMPILER_PLAN.md#w4-product-1), [W2.4a](INTEGRATED_COMPILER_PLAN.md#w24a).
Sync: `COMPLETION-CORPUS-2026-09-09`.
F0 reconciliation: 72 package functions = 45 Graph/bootstrap inputs + 14 typed
scheduled inputs + 13 raw/unclassified inputs. This is an input inventory, not
artifact-consumption or device proof. Native unary family mappings still require
consumer-by-consumer reconciliation before any Graph constructor deletion.

### 2026-09-09 — Four canonicalization legality improvements

Owner: [W5.5](INTEGRATED_COMPILER_PLAN.md#w55)

PRs: Uncommitted engineering increment.

Outcome: Compose actual transpose permutations and eliminate only identities; fold only rank-two matrix swaps into matmul while preserving bias/residual operands; retain identity casts carrying policy/metadata; refuse epilogue fusion when the replacement cannot carry source attributes. Direct rank-two transpose lowering honors explicit identity permutations. Failed host exception materialization drops unpublished object roots while retaining constructor failure tracebacks.

Remaining: These are legality and data-movement improvements, not tuned GPU candidate promotion. Arbitrary native heap allocation/collection, full native CPython frames and isolation-based driver recovery remain open. The core assertions corpus has zero failures; owning-backend matrix coverage remains open.

Evidence: `canonicalize_permutation_policy.mlir`, existing fusion/transpose fixtures, `test_native_canonical_permutation.py` and `test_source_exception_heap.py`. Super-Bear: 236 focused unit/registry checks passed; Princess-Luna: 13 semantic/heap checks and three MLIR fixtures passed independently. CPU execution distinguishes an identity permutation from a matrix swap on non-symmetric square data. Broader source/operation/dtype checks: 179 passed, 1 skipped. The final all-target assertions corpus passes all 474 active fixtures.

<!-- entry-fields:end -->

Additional owner: [W4-PRODUCT-1](INTEGRATED_COMPILER_PLAN.md#w4-product-1).
Sync: `CANONICAL-LEGALITY-2026-09-09`.
The generic and custom canonicalizers share transpose interpretation; unknown
metadata refuses simplification instead of silently losing an obligation.
Non-inverse permutations of equal-sized axes no longer cancel by type equality.
### 2026-09-09 — Native exception arena and isolation recovery boundary

Owner: [W4-PRODUCT-1](INTEGRATED_COMPILER_PLAN.md#w4-product-1)

PRs: Uncommitted engineering increment.

Outcome: Native exception allocation/collection and process-isolation recovery foundations implemented; typed native frame records cross the checked exception boundary.

Remaining: Exact CPython frames require interpreter deoptimization; F2 non-f32 matmul/cohort/breadth migrations and the registered W5.2f Schedule/Tile SSD producer remain open.

Evidence: Super-Bear passes 37 focused runtime/audit checks; Princess-Luna passes 26 focused runtime checks.

<!-- entry-fields:end -->

Additional owners: [E2E-REAL-6](INTEGRATED_COMPILER_PLAN.md#e2e-real-6),
[W5.2f](INTEGRATED_COMPILER_PLAN.md#w52f).

A C-compatible exception arena now exposes node and
payload pointers, grows without invalidating live handles, and collects
unrooted object graphs including cycles. Transported exceptions retain a typed
native deoptimization-frame record (file, line, function and instruction
slot); no synthetic Python traceback is claimed because CPython provides no
supported API for constructing an exact frame with native locals. A bounded
driver-isolation lease now permits recovery after an uncertain outcome only by
proving death of the owning process/context, and refuses recovery of a healthy
context. CUDA and ROCm host suites each pass the focused 15-case contract set.

The F2 x86 automatic package selector now routes backward attention through
the existing content-addressed Schedule/Tile artifact and saved-LSE descriptor;
its compiled consumer is no longer reachable only by a manual artifact call.
The remaining census identifies non-f32 matmul, cohort and breadth constructors.
The W5.2f source audit
found backend ReplaySSM kernels but no shared Schedule SSD producer or
Schedule-to-Tile consumer; the first correct SSD increment must add that typed
IR boundary before any backend package is promoted.


### 2026-09-10 — Installed drivers and owning-device measurements

Owner: [COMPILER-DEVEX-1](INTEGRATED_COMPILER_PLAN.md#compiler-devex-1)

PRs: Uncommitted engineering increment.

Outcome: The compiler-tools install component ships both compiler drivers and the layout shared library. Relative loader paths support prefix relocation. A smoke check copies the installed prefix, removes loader overrides, runs both drivers outside the checkout and translates MLIR to LLVM IR. The opt-in CI lane now consumes both the lit-union and installed-driver checks. Native Metal GELU correctness and attention-backward package timing are measured on M1 Max; native ANN correctness and scoped measurement admission run independently on CUDA SM120 and ROCm gfx1151.

Remaining: CI execution of this revision is pending publication. Measurements cover bounded workloads and complete-call timing; they do not close general backend execution, device-clock attribution or production performance promotion.

Evidence: `benchmarks/baselines/installed_device_gates_20260910/`, `scripts/check_installed_compiler.py`; installed-prefix smoke and all 474 assertions lit fixtures pass on Super-Bear. Two native Metal static/dynamic GELU tests pass on the Mac. Per-device measurement reports retain numerical checks and admission outcomes.

<!-- entry-fields:end -->

Additional owner: [EVIDENCE-PACKET-1](INTEGRATED_COMPILER_PLAN.md#evidence-packet-1).
Sync: `INSTALLED-DEVICE-GATES-2026-09-10`.


### 2026-09-10 — Exception arena reader and payload lifetimes

Owner: [W4-PRODUCT-1](INTEGRATED_COMPILER_PLAN.md#w4-product-1)

PRs: Uncommitted continuation after #739.

Outcome: Exported exception ABI views now pin storage through explicit reader leases. Allocation and collection refuse while any lease remains. Partial collection coalesces and reuses dead payload ranges without moving live payloads; repeated transient allocations alongside permanent roots remain bounded.

Remaining: Native arena producer integration, arbitrary object heaps and real CPython frame reconstruction. A caller must retain its ABI lease through native completion; raw addresses must not escape that lease.

Evidence: `test_native_exception_arena.py` pointer/owner retention and repeated partial-collection tests on Super-Bear and Princess-Luna. Addresses are host pointers, not a GPU heap claim.

<!-- entry-fields:end -->

### 2026-09-10 — Isolated native ANN worker and recovery

Owner: [W2.4a](INTEGRATED_COMPILER_PLAN.md#w24a)

PRs: Uncommitted continuation after #739.

Outcome: Native ANN packages can bind in spawned CUDA/HIP workers with worker-owned contexts. One outstanding request carries only copied host tensors. Failed or timed-out completion exposes no result and quarantines ownership; explicit recovery confirms process death before replacement. Error paths exit without destructor-driven driver retries.

Remaining: General package families, heterogeneous dynamic frames, external device readers, concurrent writers and actual uncertain-driver fault testing. Worker IPC is opt-in and is not a performance candidate.

Evidence: `benchmarks/baselines/ownership_census_profile_20260910/`; independent SM120/gfx1151 numerical execution, stopped-idle-worker timeout, confirmed teardown and replacement. SIGSTOP injection is a process/transport fault, not a GPU driver failure.

<!-- entry-fields:end -->

### 2026-09-10 — x86 BF16 scheduled package migration

Owner: [E2E-REAL-6](INTEGRATED_COMPILER_PLAN.md#e2e-real-6)

PRs: Uncommitted continuation after #739.

Outcome: x86 BF16 inputs with f32 accumulation/output now traverse the canonical Schedule/Tile producer. Native packaging projects dtype, byte widths, shape guards, CPU feature requirements and schedule identity, and rejects altered projection/Tile evidence before lowering.

Remaining: uint8/int8 and fp64 matmul, cohort/breadth constructors and target-specific promotion. Existing BF16 runtime kernels remain the physical consumer; no new speedup is claimed.

Evidence: `test_scheduled_matmul_consumers.py` BF16 producer, descriptor/replay tampering and owning-CPU differential cases; exact results recorded in the wave evidence directory.

<!-- entry-fields:end -->

### 2026-09-10 — Route census and device profile attribution

Owner: [E2E-REAL-6F](INTEGRATED_COMPILER_PLAN.md#e2e-real-6f)

PRs: Uncommitted continuation after #739.

Outcome: A reproducible direct-call census keeps Graph-input wrappers separate from native scheduled consumers. The broader 32x8 terminal-square ANN workload is measured independently with Nsight Systems/Compute and rocprofv3.

Remaining: The 64x8 workload exceeds the current 4096-byte temporary bound. Row-parallel lowering still allocates full matrix temporaries per thread; per-row storage projection needs its own ownership proof. CUDA has one-block underutilization and substantial host launch/transfer overhead. ROCm WSL supplies HIP API traces but no kernel/copy timeline in this run; no counter/device-time parity or performance promotion follows.

Evidence: `benchmarks/baselines/ownership_census_profile_20260910/`. Profiler-instrumented timings are diagnostic, not independent promotion measurements.

<!-- entry-fields:end -->

Additional owners: [MSW-9](INTEGRATED_COMPILER_PLAN.md#msw-9), [TPROF-NATIVE-1](INTEGRATED_COMPILER_PLAN.md#tprof-native-1).
Sync for this wave: `OWNERSHIP-CENSUS-PROFILE-2026-09-10`.

### 2026-09-10 — Asynchronous isolation teardown

Owner: [W2.4a](INTEGRATED_COMPILER_PLAN.md#w24a)

PRs: Uncommitted continuation after #739.

Outcome: A bounded off-thread process teardown ticket retains dependent owners and admission slots on uncertain termination. ANN workers and declared process-owned module retirements consume it; host cleanup follows confirmed death. Nonfinite recovery deadlines are refused.

Remaining: Real driver-hang recovery, fresh device-health admission, external device readers and in-process hung unloads are not closed. A failed ticket deliberately retains its slot; operator-level recovery of an unkillable process remains necessary. Module filesystem cleanup during completion polling is still synchronous.

Evidence: focused isolation/ANN/module ownership tests and independently measured SM120/gfx1151 stopped-worker replacement in `benchmarks/baselines/async_isolation_20260910/`. No GPU hang was injected and no performance promotion follows.

<!-- entry-fields:end -->

### 2026-09-10 — Device health admission and external readers

Owner: [W2.4a](INTEGRATED_COMPILER_PLAN.md#w24a)

PRs: Uncommitted continuation after #739.

Outcome: Fresh isolated ANN workers must execute deterministic bounded numerical probes against the analytic oracle before announcing readiness. Replacement requires confirmed failed-worker teardown and repeats admission. A stuck health probe is bounded by the parent deadline. Multi-stream read scopes record every declared consumer, including exception paths. Eventless dependencies no longer silently synchronize the host; completion needs an explicit wait or isolated recovery.

Remaining: These probes establish workload/device-zero readiness at admission, not global driver health. Real wedged-driver recovery, unkillable kernel-mode processes, arbitrary device selection and unsupervised raw-pointer consumers remain open. External consumers must enqueue reads on declared streams and not retain pointers outside the scope. No automatic GPU reset or performance promotion is introduced.

Evidence: `benchmarks/baselines/health_reader_admission_20260910/` records independent SM120/gfx1151 health/replacement and two-stream native copy consumers across four tape generations. Host tests cover numerical rejection, a hung probe, partial reader failures and retained ownership.

<!-- entry-fields:end -->

### 2026-09-10 — Recovery retry and caller-error boundaries

Owner: [W2.4a](INTEGRATED_COMPILER_PLAN.md#w24a)

PRs: #740 review follow-up.

Outcome: Missing dependency events refuse before retirement commits state, preserving retry after explicit completion. Invalid ANN shape/domain inputs refuse in the parent without poisoning a healthy worker. Leaving a scope with pending work poisons and tears down the worker; uncertain cleanup retains ownership and preserves the caller's original exception.

Remaining: Actual wedged-driver recovery and unrestricted external-reader lifetimes remain open; these fixes establish host lifecycle behavior.

Evidence: focused reader-retirement and isolated-ANN regressions cover retry after a missing event, repeated invalid inputs followed by successful execution, pending context exit and failed cleanup with exception preservation.

<!-- entry-fields:end -->

### 2026-09-10 — Expanded route callers and f64 ownership

Owner: [E2E-REAL-6](INTEGRATED_COMPILER_PLAN.md#e2e-real-6)

PRs: Uncommitted continuation after #740.

Outcome: FP64 x86 matmul gains a native Schedule producer/verifier, complete storage/accumulation/output projection and the existing native fp64 consumer. Floating Graph constructor branches are removed; mixed-signedness VNNI remains. The caller census now includes x86 breadth, import aliases and local helper-to-emitter paths.

Remaining: F0 certificate reconciliation and dynamic call resolution, mixed uint8/int8 matmul, cohort/elementwise/breadth migration. Tracks 3–10 retain their prior gates; no new driver-hang recovery, heap producer, CPython deoptimization, AD/F3/SSD closure or performance promotion follows from this increment.

Evidence: `benchmarks/baselines/f64_route_ownership_20260910/`; fp64 compiler fixture and owning-CPU projection/numerical tests in `test_scheduled_matmul_consumers.py`. Registry and ABI tests preserve compiler-free coverage.

<!-- entry-fields:end -->

Additional owner: [E2E-REAL-6F](INTEGRATED_COMPILER_PLAN.md#e2e-real-6f).

### 2026-09-10 — Dynamic reader fanout and row-private ANN

Owner: [W2.4a](INTEGRATED_COMPILER_PLAN.md#w24a)

PRs: Uncommitted continuation after #740.

Outcome: Checked dynamic public results and paired source-VJP frames expose multi-stream scoped readers. Host regressions preserve both products on external failure and close every acquired reader. The existing native row-independence proof now authorizes compacting entry-owned mutable ANN temporaries to one row per thread, preserving complete constants and nested generation slots. Internal SSA proof sets, not user attributes, authorize compaction.

Remaining: General package isolation, real wedged-driver recovery, native heap producers, CPython deoptimization, general heterogeneous saved-product generation and relational alias proofs, attention raising and shared tiled SSD remain open. This increment does not substitute host reader tests for owning-device frame proof. ROCm device counters and broader/multi-block tuning remain open.

Evidence: 64x8 square ANN numerical oracles pass independently on SM120/gfx1151; single-run rewrite speedups about 0.991x/0.994x retain incumbents. CUDA Nsight separates kernel, transfer and API costs. See `benchmarks/baselines/ann_row_private_20260910/`. No promotion or direct old/new optimization speedup is claimed.

<!-- entry-fields:end -->

Additional owners: [MSW-9](INTEGRATED_COMPILER_PLAN.md#msw-9), [TPROF-NATIVE-1](INTEGRATED_COMPILER_PLAN.md#tprof-native-1).

### 2026-09-10 — SSD recurrence and retryable completion

Owner: [W5.2f](INTEGRATED_COMPILER_PLAN.md#w52f)

PRs: Uncommitted continuation after #740.

Outcome: A native registered internal `schedule.ssd` and Python producer carry X, multiplicative decay, B/C, immutable initial carry, Y, final carry and chunk-end checkpoints. Schedule-to-Tile lowers the verified static f32 recurrence to structured tensor loops; native CPU tests cover partial chunks and input preservation. Artifact validation replays the incoming Schedule with the recorded compiler identity. Failed asynchronous recovery tickets can reconcile late process death without retrying termination; concurrent polls release owners/slots once. Paired VJP retirement resumes only unsubmitted children. Arena edge/root APIs permit actual cycles and preserve leased storage. Exception completion restores interpreter-owned fields without user assignment hooks.

Remaining: SSD public/frontend integration, tiled/cooperative kernels, checkpoint adjoints, ReplaySSM comparison and target packages; native exception allocation producers, general CPython deoptimization/frames, general saved products and attention raising remain open. Process exit proves resource isolation teardown, not driver health or recovery from a wedged driver. No new GPU or performance promotion claim.

Evidence: `tests/unit/test_scheduled_ssd.py`, `tests/tessera-ir/phase2/ssd_schedule.mlir`, `tests/unit/test_native_driver_isolation.py`, `tests/unit/test_native_public_results.py`, `tests/unit/test_native_exception_arena.py`, `tests/unit/test_source_exception_heap.py`. Assertions-enabled host compiler and CPU JIT; backend evidence boundaries remain independent.

<!-- entry-fields:end -->

Additional owners: [W2.4a](INTEGRATED_COMPILER_PLAN.md#w24a), [W4-PRODUCT-1](INTEGRATED_COMPILER_PLAN.md#w4-product-1), [AD-RESIDUAL-EVAL-1](INTEGRATED_COMPILER_PLAN.md#ad-residual-eval-1).

### 2026-09-10 — Native heap, attention recipes and SSD device proof

Owner: [W5.2f](INTEGRATED_COMPILER_PLAN.md#w52f)

PRs: Uncommitted continuation after #740.

Outcome: A bounded native C++ exception heap owns allocation, explicit roots, cause/context cycle collection and payload-hole reuse through generation-checked handles and copy-out reads. An exact dense f32 attention loop recognizer retains its source oracle and produces a native symbolic recipe; two buckets lower through Schedule and Tile with head-width witness validation. Serial replay-bound SSD packages execute on CUDA SM120 and ROCm gfx1151. Device tests exposed initial-carry mutation during bufferization; immutable input ownership is now projected before bufferization. The standalone runtime shared build also exposed duplicate disabled CUDA/HIP factories in the CPU translation unit; their owning backend files now supply them exclusively.

Remaining: Automatic source-IR heap producers, GPU heap allocation, general exception objects and native CPython deoptimization records/frames remain open. Attention needs broader patterns and exact-device candidate admission. SSD needs cooperative kernels, public frontend and checkpoint AD integration, ReplaySSM comparison and measured promotion. No performance promotion is claimed.

Evidence: `tests/unit/test_native_exception_producer.py`, `tests/unit/test_attention_loop_idiom.py`, `tests/unit/test_native_ssd.py`, the native runtime ABI smoke and `benchmarks/baselines/native_heap_attention_ssd_20260910/`. SSD records cover chunks 1, 2 and 5 on each owning device and verify all five inputs remain unchanged.

<!-- entry-fields:end -->

Additional owners: [W4-PRODUCT-1](INTEGRATED_COMPILER_PLAN.md#w4-product-1), [FRONTEND-IR-MEDIUM-1](INTEGRATED_COMPILER_PLAN.md#frontend-ir-medium-1).

### 2026-09-10 — Heap IR, raised attention and cooperative SSD

Owner: [W5.2f](INTEGRATED_COMPILER_PLAN.md#w52f)

PRs: Uncommitted continuation after #741.

Outcome: Static source exception tables emit checked native allocation/root/edge calls through MLIR/LLVM, with a native source-state decoder consumer and explicit generated-library lifetime. Native attention buckets project their Schedule descriptor and bind NVIDIA execution without Graph reconstruction. Cooperative SSD uses block-owned head/value columns, lane-owned state, shared-memory barriers and an ordered leader reduction. CUDA and ROCm correctness covers immutable inputs, outputs, final carry and chunk checkpoints; profiling separates resident event windows, checked-call preparation and API/kernel traces. The assertions build caught a missing Tessera dialect dependency in Schedule-to-Tile, now declared.

Remaining: Runtime-valued/iteration-site exception allocation, GPU heaps and full CPython frames; broader attention patterns, sibling-device bindings and arbiter promotion; SSD public integration, checkpoint AD, more efficient reduction/tiling, ReplaySSM comparison and selector-grade measurements. ROCProfiler exposed HIP API records but no GPU kernel/copy trace records on this host, so hardware-counter attribution remains open. No promotion claim.

Evidence: `tests/unit/test_native_exception_producer.py`, `tests/unit/test_attention_loop_idiom.py`, `tests/unit/test_native_ssd.py`, `benchmarks/record_ssd_gpu.py` and `benchmarks/baselines/heap_attention_cooperative_ssd_20260910/`. Backend queue historical increments are marked superseded rather than left as competing current obligations.

<!-- entry-fields:end -->

Additional owners: [W4-PRODUCT-1](INTEGRATED_COMPILER_PLAN.md#w4-product-1), [FRONTEND-IR-MEDIUM-1](INTEGRATED_COMPILER_PLAN.md#frontend-ir-medium-1), [TPROF-NATIVE-1](INTEGRATED_COMPILER_PLAN.md#tprof-native-1).

### 2026-09-10 — Dynamic heap payloads, checkpoint AD and paired measurements

Owner: [W5.2f](INTEGRATED_COMPILER_PLAN.md#w52f)

PRs: Uncommitted continuation after #741.

Outcome: Runtime numeric exception payloads are copied through native pointer/size operands into a fresh bounded heap; changing shapes and failed allocations preserve ownership and retryability. The native source-state exception decoder consumes the same path. Raised attention now projects x86 and Apple parents, with executable x86 proof on Princess-Luna and no Graph reconstruction. The SSD native CPU checkpoint VJP accepts cotangents for output, final carry and checkpoints, produces all five input gradients, and recomputes from chunk boundaries. Its scoped forward/VJP owner keeps checkpoints private to the corresponding forward call. Nine fixed independent process pairs on each GPU compare exact serial/cooperative artifacts in alternating order.

Remaining: In-kernel/iteration-site heap allocation, GPU exception heaps and general objects; broader attention recognition, Apple owning-device validation and ROCm low-precision binding; automatic public mixer AD, GPU checkpoint backward, persistent/asynchronous checkpoint ownership and ReplaySSM comparison. Paired resident event windows include submission gaps; they do not supply calibrated device clocks or selector admission. No route promotion.

Evidence: `tests/unit/test_native_exception_producer.py`, `tests/unit/test_attention_loop_idiom.py`, `tests/unit/test_scheduled_ssd.py`, `tests/unit/test_ssd_comparison.py`, `benchmarks/compare_ssd_variants.py`, and `benchmarks/baselines/ssd_paired_process_20260910/`. Numerical VJP checks use finite differences of every input with nonzero seeds for every result, including partial chunks.

<!-- entry-fields:end -->

Additional owners: [W4-PRODUCT-1](INTEGRATED_COMPILER_PLAN.md#w4-product-1), [FRONTEND-IR-MEDIUM-1](INTEGRATED_COMPILER_PLAN.md#frontend-ir-medium-1), [TPROF-NATIVE-1](INTEGRATED_COMPILER_PLAN.md#tprof-native-1).

### 2026-09-10 — GPU payload frames, mixer AD and artifact selection

Owner: [W5.2f](INTEGRATED_COMPILER_PLAN.md#w52f)

PRs: Uncommitted continuation after #741.

Outcome: Native GPU exception-payload producers consume serialized numeric-site metadata and allocate bounded frames transactionally, publishing generation/offset/length records only on success. CUDA and ROCm prove overflow/negative-length refusal, unchanged storage on failure, empty payloads, subsequent success and stale-generation decoder rejection. The native SSD checkpoint VJP now packages for both GPUs, consumes actual cooperative-forward checkpoints and matches finite differences for all five inputs with nonzero output/carry/checkpoint cotangents. SSDCheckpointProgram also connects native CPU Y to automatic host-tape grad with private forward snapshots. Attention recognition accepts one positive f32-exact post-dot literal scale while retaining full AST matching; two CUDA buckets execute scaled/unscaled variants. Explicit measured SSD binding replays both packages, recomputes paired bounds, binds calibration to exact images/durations, and rebuilds the existing ROCm policy instead of trusting eligibility flags.

Remaining: GPU frames are preallocated single-writer numeric storage with caller-supplied generations, not arbitrary allocation or a concurrent object collector. Automatic source throw-site integration and reader-aware frame reclamation remain open. Automatic GPU-resident mixer AD, tuned backward/cooperative checkpoint kernels, higher-order products and public mixer family wiring remain open. Attention masks/GQA/broader recognition and sibling exact-device scaled proof remain open. The actual CUDA and ROCm selector calls retain incumbents: CUDA lacks a typed native calibration adapter, and ROCm lacks per-process eligible calibration (the current policy also requires bare metal). No promotion.

Evidence: `tests/unit/test_gpu_exception_heap.py`, `tests/unit/test_scheduled_ssd.py`, `tests/unit/test_native_ssd.py`, `tests/unit/test_attention_loop_idiom.py`, `tests/unit/test_ssd_comparison.py`, `benchmarks/record_gpu_heap_ssd_ad.py`, `benchmarks/check_ssd_admission.py`, and `benchmarks/baselines/gpu_heap_ssd_ad_20260910/`. Production package identities still match the prior nine-pair measurements.

<!-- entry-fields:end -->

Additional owners: [W4-PRODUCT-1](INTEGRATED_COMPILER_PLAN.md#w4-product-1), [FRONTEND-IR-MEDIUM-1](INTEGRATED_COMPILER_PLAN.md#frontend-ir-medium-1), [TPROF-NATIVE-1](INTEGRATED_COMPILER_PLAN.md#tprof-native-1).

### 2026-09-10 — Reusable GPU pools, resident AD and window calibration

Owner: [W5.2f](INTEGRATED_COMPILER_PLAN.md#w52f)

PRs: Uncommitted continuation after #741.

Outcome: Bounded native GPU pools allocate fixed payload slots and run stop-the-world mark/sweep with runtime root/edge graphs, generation checks, partial reuse and unrooted cycle collection. CUDA SM120 and ROCm gfx1151 execute the same semantic regressions with independent packages. ResidentSSDProgram generates paired forward/VJP packages, owns device snapshots and checkpoints, differentiates Y into all five resident gradients, and closes scoped buffers after synchronous completion. CUDA attention recognition now proves explicit GQA indexing and end-aligned causal masks for both longer-Q and longer-K buckets; native recipe instantiation enforces positive divisible head counts. CUDA SSD admission rebuilds a per-process Nsight/event-window calibration instead of requiring an absent adapter.

Remaining: Pools are bounded, fixed-width, single-writer numeric storage with two outgoing edges and quiescent readers; concurrent collection, automatic throw-site integration and arbitrary object/payload allocation remain open. Resident AD is synchronous first-order family integration, not general public tape composition, external-reader retirement or tuned GPU backward. General additive/padding masks, broader attention idioms and sibling device admission remain open. The measured CUDA calibration agrees within 2.26% but has 1.957x profiler overhead, dirty source and WSL execution, so promotion refuses. Eighteen eligible per-process calibrations and renewed exact-artifact comparisons remain required; prior packets retain their original compiler identities. ROCm bare-metal dispatch/counter calibration remains open.

Evidence: `tests/unit/test_gpu_heap_collection.py`, `tests/unit/test_attention_loop_idiom.py`, `tests/unit/test_cuda_window_calibration.py`, `benchmarks/record_pool_resident_ssd.py`, `benchmarks/calibrate_ssd_cuda.py`, and `benchmarks/baselines/pool_resident_gqa_calibration_20260910/`.

<!-- entry-fields:end -->

Additional owners: [W4-PRODUCT-1](INTEGRATED_COMPILER_PLAN.md#w4-product-1), [FRONTEND-IR-MEDIUM-1](INTEGRATED_COMPILER_PLAN.md#frontend-ir-medium-1), [TPROF-NATIVE-1](INTEGRATED_COMPILER_PLAN.md#tprof-native-1).

### 2026-09-10 — Stream-owned object graphs, asynchronous AD and window legality

Owner: [W5.2f](INTEGRATED_COMPILER_PLAN.md#w52f)

PRs: Uncommitted continuation after #741.

Outcome: ResidentObjectPool stores bounded opaque byte records with configurable generation-checked reference edges. NativeStreamEpoch permits scoped readers on multiple streams and serializes graph updates/allocation/collection after their events. Open reader scopes refuse mutation; failed event recording retains an eventless dependency until explicit completion. CUDA/HIP validate cyclic graphs, fourth-edge reachability, partial collection, byte payload preservation and generation reuse. SSD asynchronous VJPs now compose through projected scoped gradients and stream-ordered derivative retirement on both devices. Capture still snapshots synchronously. Windowed GQA recognizes bounded asymmetric slices and carries a mandatory Presburger nonempty-row condition through native recipe instantiation. SSD packaging no longer imports the 1024-element persistent-tape shape parser; its checked f32 storage bound is 64 MiB per buffer, validated with a 512x2x32x8 cooperative workload on both GPUs. CUDA calibration now binds process nonces and the Nsight PID, and a nine-pair collector preflights clean/bare-metal eligibility.

Remaining: Collection is reader-coordinated stop-the-world mutation, not simultaneous mutator/collector execution or arbitrary CPython graph discovery. Variable-size allocation, automatic exception producer integration and general object semantics remain open. AD is an explicit first-order SSD family API; general public tape integration, asynchronous capture/whole-frame teardown, higher-order products and optimized backward remain open. Additive/padding/general masks and sibling-device window admission remain open. The new CUDA capture passes clock agreement (0.51%) and overhead (0.24%) gates but remains dirty/WSL evidence. Eighteen eligible independent calibrations plus renewed exact-artifact comparison are still required for promotion; ROCm needs bare-metal profiler evidence. No promotion claimed.

Evidence: `tests/unit/test_native_stream_epoch.py`, `tests/unit/test_gpu_heap_collection.py`, `tests/unit/test_attention_loop_idiom.py`, `tests/unit/test_cuda_window_calibration.py`, `benchmarks/record_async_pool_ad.py`, `benchmarks/record_ssd_calibrated_pairs.py`, and `benchmarks/baselines/async_objects_windows_20260910/`.

<!-- entry-fields:end -->

Additional owners: [W4-PRODUCT-1](INTEGRATED_COMPILER_PLAN.md#w4-product-1), [FRONTEND-IR-MEDIUM-1](INTEGRATED_COMPILER_PLAN.md#frontend-ir-medium-1), [TPROF-NATIVE-1](INTEGRATED_COMPILER_PLAN.md#tprof-native-1).

### 2026-09-10 — Snapshot marking, public VJP and additive bias

Owner: [W5.2f](INTEGRATED_COMPILER_PLAN.md#w52f)

PRs: Uncommitted continuation after #741.

Outcome: Private GPU graph snapshots permit marking on a separate stream while active graph updates continue. Final seeded remark/sweep is exclusive and retains snapshot survivors conservatively; malformed seeds fail closed. Hook-free discovery handles exact builtin containers and plain instance dictionaries with cycles/aliases. Public vjp dispatches an explicitly owned resident SSD program; asynchronous capture retains inputs through copy completion, backward composes scoped gradients, and whole-frame retirement orders all forward/derivative readers before frees. Failed event dependencies keep retirement retryable. Full-shape finite additive attention bias now survives recognition, native recipe instantiation, Schedule projection and NVIDIA execution.

Remaining: Concurrent sweeping, lock-free producers, arbitrary extension heaps, concurrent Python discovery, variable-size storage and automatic exception-site fusion remain open. Public AD integration is first-order SSD protocol dispatch, not arbitrary traced GPU programs or higher-order AD. Program/module close may synchronize. General Boolean/padding/broadcast masks and fully masked-row semantics need further contracts. CUDA/HIP tests run on owning GPUs under WSL: they do not establish physical overlap or eligible performance. Clean bare-metal per-process evidence remains required; no promotion.

Evidence: `tests/unit/test_object_discovery.py`, `tests/unit/test_resident_ssd_ownership.py`, `tests/unit/test_gpu_heap_collection.py`, `tests/unit/test_attention_loop_idiom.py`, `benchmarks/record_snapshot_public_ad.py`, and `benchmarks/baselines/snapshot_public_ad_20260910/`.

<!-- entry-fields:end -->

Additional owners: [W4-PRODUCT-1](INTEGRATED_COMPILER_PLAN.md#w4-product-1), [FRONTEND-IR-MEDIUM-1](INTEGRATED_COMPILER_PLAN.md#frontend-ir-medium-1), [TPROF-NATIVE-1](INTEGRATED_COMPILER_PLAN.md#tprof-native-1).

### 2026-09-10 — Benchmark compiler alignment

Owner: [EVIDENCE-PACKET-1](INTEGRATED_COMPILER_PLAN.md#evidence-packet-1)

PRs: Uncommitted continuation after #741.

Outcome: Reviewed math, linalg, energy/Clifford compositions, eight AD scripts and the operator harness against their actual callers. Consolidated navigation in benchmarks/COMPILER_ALIGNMENT.md; registered the missing math/autodiff manifest entries with syntax-only compile_only checks. New physical-math results require observed native execution and finite correctly shaped outputs; linalg enforces residual bounds and emits clean JSON stdout. GA/EBM rows now label unattributed library execution instead of claiming CPU reference placement despite optional Apple fast paths. New math and selected AD packets do not inherit performance eligibility from target names or historical packet paths. Historical evidence is unchanged; no executable suite is archived because each retains consumers or a distinct oracle.

Remaining: Shared evidence adapters, exact-package math ancestry, per-call GA/EBM route attribution, target-aware operator harness adapters, and public frontend counterparts for direct-IR AD probes. No fresh performance measurements or promotion. Clean bare-metal exact-artifact comparisons remain required.

Evidence: `benchmarks/COMPILER_ALIGNMENT.md`, `tests/unit/test_physical_math_evidence.py`, `tests/unit/test_benchmark_surface_repair.py`, `tests/unit/test_clifford_core_benchmark.py`, `tests/unit/test_solver_ift_evidence.py`, and `tests/unit/test_operator_benchmarks_contract.py`.

<!-- entry-fields:end -->

Additional owner: [TPROF-NATIVE-1](INTEGRATED_COMPILER_PLAN.md#tprof-native-1).

### 2026-09-10 — Extended benchmark alignment

Owner: [EVIDENCE-PACKET-1](INTEGRATED_COMPILER_PLAN.md#evidence-packet-1)

PRs: Uncommitted continuation after #741.

Outcome: Extended the benchmark alignment review across autodiff, DLOP, SuperBench, DeepScholar, grid-AI, lattice reasoning, visual-complex, RL and E2E spine. Retired the synthetic attention timer without removing compatibility wrappers. DLOP dispatch counts are explicitly static estimates; lattice Apple-call durations now populate host-wall rather than kernel timing. Grid/visual compositions report unattributed library execution. Every timed Apple policy submission now requires native execution and finite output. Four missing manifest entries now have syntax-only CI checks. E2E recorders, historical packets and distinct reference oracles remain intact.

Remaining: Profiler-backed dispatch receipts, native SuperBench package adapters, automatic public AD counterparts and full model/distributed execution proof. No device packet was re-sealed, no hardware speedup measured and no performance promoted.

Evidence: `benchmarks/COMPILER_ALIGNMENT.md`, `tests/unit/test_benchmark_alignment_extended.py`, `tests/unit/test_rl_policy_loss_benchmark.py`, and the affected core/schema/orchestration tests.

<!-- entry-fields:end -->

Additional owner: [TPROF-NATIVE-1](INTEGRATED_COMPILER_PLAN.md#tprof-native-1).

### 2026-09-10 — Native benchmark adapters

Owner: [EVIDENCE-PACKET-1](INTEGRATED_COMPILER_PLAN.md#evidence-packet-1)

PRs: Uncommitted continuation after #741.

Outcome: Added native CUDA/HIP SuperBench ANN adapters and a separate DLOP observed-dispatch lane sharing checked package execution. Three ANN workloads pass on RTX 5070 and gfx1151; original and transformed variants each produce one accepted driver launch per measured call. Added public resident SSD VJP comparisons for three shapes, two cotangents and all five inputs against independent float64 central differences; maximum gradient error is below 1.2e-8 on both hosts.

Remaining: Driver API receipts are not profiler kernel records. Broader GEMM/attention adapters, native mappings for the original DLOP catalog, general public frontend AD and clean bare-metal performance promotion remain open. Instrumented WSL host-wall timings are diagnostic only; no promotion or dispatch-reduction claim.

Evidence: `benchmarks/baselines/native_benchmark_adapters_20260910/`, `benchmarks/native_ann_adapter.py`, `benchmarks/autodiff/benchmark_public_ssd.py`, `tests/unit/test_native_benchmark_receipts.py`.

<!-- entry-fields:end -->

Additional owners: [TPROF-NATIVE-1](INTEGRATED_COMPILER_PLAN.md#tprof-native-1), [W5.2f](INTEGRATED_COMPILER_PLAN.md#w52f).

### 2026-09-10 — Broader adapters and kernel attribution

Owner: [EVIDENCE-PACKET-1](INTEGRATED_COMPILER_PLAN.md#evidence-packet-1)

PRs: Uncommitted continuation after #741.

Outcome: Expanded native SuperBench configs to baseline/larger ANN and serial/cooperative SSD; all four execute correctly on SM120 and gfx1151. Preserved separate host-call and device-event timing domains. A dedicated CUDA SSD capture has 701 profiler kernels matched one-to-one to successful owning-process launch API records, tied to the packet's artifact/image identity. Attribution tests reject wrong processes, failed launches, wrong kernel names and missing/duplicate correlations.

Remaining: General mixed-artifact range attribution, native GEMM/attention adapters and clean performance promotion. ROCm's fresh kernel/copy-enabled trace still provides only HIP API and agent records on WSL, so owning-device kernel attribution remains open. No timing or execution evidence transfers to Apple/x86.

Evidence: `benchmarks/baselines/broader_adapters_20260910/`, `benchmarks/attribute_ssd_cuda.py`, `tests/unit/test_ssd_profiler_attribution.py`, and both native SuperBench suite results.

<!-- entry-fields:end -->

Additional owners: [TPROF-NATIVE-1](INTEGRATED_COMPILER_PLAN.md#tprof-native-1), [W5.2f](INTEGRATED_COMPILER_PLAN.md#w52f).

### 2026-09-10 — Matrix adapters and mixed attribution

Owner: [EVIDENCE-PACKET-1](INTEGRATED_COMPILER_PLAN.md#evidence-packet-1)

PRs: Uncommitted continuation after #741.

Outcome: Added scheduled native GEMM/attention adapters with descriptor-projected input layout and independent NumPy oracles. CUDA executes both. Mixed profiler ranges bind run, artifact and native image identities and correlate successful launch APIs to kernels. Tests reject wrong image identity, missing/duplicate correlations, overlapping ranges, failed APIs and escaping completion.

Remaining: ROCm GEMM/attention both abort at LLVM GPUFuncOpLowering's duplicate DictionaryAttr assertion with the installed assertions-enabled compiler; no ROCm execution result claimed. Asynchronous/multi-thread attribution, broader matrix envelopes and clean performance promotion remain open. Native ROCm config retains failing rows rather than silently demoting.

Evidence: `benchmarks/baselines/matrix_mixed_20260910/`, `benchmarks/native_matrix_adapter.py`, `benchmarks/attribute_mixed_cuda.py`, `tests/unit/test_mixed_profiler_attribution.py`.

<!-- entry-fields:end -->

Additional owner: [TPROF-NATIVE-1](INTEGRATED_COMPILER_PLAN.md#tprof-native-1).


### 2026-09-11 — Program retirement and declared-slot discovery

Owner: [W5.2f](INTEGRATED_COMPILER_PLAN.md#w52f)

PRs: Uncommitted continuation after #742; preserves the SSD numerical-admission repair.

Outcome: Resident SSD now exposes program-level retire_async/poll_close: reader preflight precedes frame frees, partially queued retirement can resume, new captures refuse, and modules unload off-thread only after frame completion. CUDA SM120 and ROCm gfx1151 validate two public-VJP frames while context synchronization is forbidden on the program path. Opt-in declared-slot discovery preserves inherited aliases/cycles through native descriptors and refuses custom descriptors, undeclared dictionaries and opaque layouts; a slotted self-cycle survives GPU collection on both hosts.

Remaining: Arbitrary traced GPU tapes, extension heaps, variable-size device payloads, concurrent sweeping, asynchronous owner adoption and general masks remain open. A mutation publication barrier plus generation/reader-aware reclamation must precede concurrent sweep. Module workers bound admission and polling, not the driver's own latency or failure recovery.

Evidence: `benchmarks/baselines/program_retirement_20260911/`, `benchmarks/record_program_retirement.py`, `tests/unit/test_resident_ssd_ownership.py`, `tests/unit/test_object_discovery.py`; 54 focused host tests, Ruff and zero-error mypy on WSL.

<!-- entry-fields:end -->

Additional owner: [W4-PRODUCT-1](INTEGRATED_COMPILER_PLAN.md#w4-product-1).

The requested ten-track list is reconciled to existing owners, not a new status registry:

- F0/F2 retain caller/envelope certificate joins and the remaining mixed-integer/cohort/breadth constructors. Scheduled non-f32 GEMM already exists; ROCm matrix native lowering currently exposes an assertions-enabled LLVM failure. No constructor is removed by this increment.
- Isolation recovery already launches bounded native ANN packages; actual driver-hang/device-health recovery remains separate from stopped-worker teardown proof. Scoped dynamic/external readers exist; general heterogeneous capture and failure ownership still need work.
- Native exception allocation/root producers and bounded cycle collectors exist. Automatic throw-site integration, arbitrary heaps, concurrent reclamation and complete CPython deoptimization/frame reconstruction remain open.
- Persistent SSD AD/checkpoints and native attention recipes/buckets exist. General traced effectful AD, heterogeneous/aliased products, Boolean/padding/broadcast masks and fully masked-row behavior remain owner-specific acceptance gates.
- Larger ANN/SSD measurements and CUDA mixed-artifact attribution exist; they do not grant clean promotion or ROCm hardware-counter evidence. The shared registered Schedule SSD family is implemented; next work is frontend integration, tuning and selector-grade proof, not starting another SSD operation.


### 2026-09-11 — Incremental sweeping and traced SSD composition

Owner: [W4-PRODUCT-1](INTEGRATED_COMPILER_PLAN.md#w4-product-1)

PRs: This change set (`codex/gated-heaps-resident-ad`); includes the slotted-module identity correction.

Outcome: Incremental GPU collection retires a complete unreachable cohort before range reclamation, preserving generation-safe reuse and avoiding dangling dead-cycle edges between batches. Explicit exact-type extension extractors copy schema-tagged bytes/referents under caller-owned quiescence. Linear ResidentSSDTrace composition builds its call graph before GPU submission, generates reverse traversal and projects scoped gradients back to all public inputs. NVIDIA additive-mask admission now accepts negative infinity but refuses fully masked rows after causal/window intersection.

Remaining: Sweep kernels still exclude writers/readers; simultaneous sweeping needs atomic publication and read barriers. Extension callbacks are explicitly trusted, not automatic arbitrary-heap discovery. GPU tracing is host orchestration of independently replayed packages; canonical composition IR/fusion remains open. It admits uniquely consumed linear SSD calls only: branches, accumulation, reused inputs, effects and additional families remain open. Boolean/broadcast masks and defined empty-row outputs need compiler/ABI work. No promotion or Apple/x86 device proof is claimed.

Evidence: `benchmarks/baselines/incremental_heap_trace_20260911/`, `benchmarks/record_incremental_heap_trace.py`, `tests/unit/test_resident_trace.py`, `tests/unit/test_resident_pool_sweep.py`, `tests/unit/test_object_discovery.py`, `tests/unit/test_attention_loop_idiom.py`.

<!-- entry-fields:end -->

Additional owners: [W5.2f](INTEGRATED_COMPILER_PLAN.md#w52f), [FRONTEND-IR-MEDIUM-1](INTEGRATED_COMPILER_PLAN.md#frontend-ir-medium-1).

The ten-track reconciliation above remains the ownership map. This increment
advances the heap, persistent-product/AD and attention tracks; it does not close
F0/F2 census/migrations, isolated driver recovery, CPython reconstruction or
performance admission. The shared tiled-SSD family already exists.


### 2026-09-11 — Resident DAG accumulation and snapshot readers

Owner: [W5.2f](INTEGRATED_COMPILER_PLAN.md#w52f)

PRs: This change set (`codex/gated-heaps-resident-ad`), extending the preceding incremental sweep work.

Outcome: Traced SSD DAGs now admit fan-out and repeated public inputs across calls; replay-validated native f32 addition accumulates every cotangent path in a fixed order. Addition outputs and modules share frame/program retirement. CUDA/HIP validate a three-call DAG against independent float64 finite differences. Immutable pool snapshots hold copied epochs for readers while the live pool sweeps; parent close retains all snapshots through reader completion. Discovery adds exact general-key maps, sets, frozensets and bytearrays, and refuses undeclared builtin-subclass payloads and custom metaclasses before invoking user hooks. CUDA validates irregular additive masks composed with ragged GQA, causal and window masks in both Q/K size directions.

Remaining: Snapshot isolation does not permit racing arbitrary readers/writers over the same storage. Unrestricted sweep still needs publication/read barriers and generation-safe reclamation. Arbitrary extension discovery still requires a declared trusted layout; opaque native storage is not guessed. Tracing remains host orchestration of native SSD packages, not canonical whole-program IR, effectful control-flow AD, general operation families, or same-call alias admission. Boolean/broadcast masks and defined empty-row outputs remain open. No promotion.

Evidence: `benchmarks/baselines/dag_snapshot_20260911/`, `benchmarks/record_dag_snapshot.py`, `tests/unit/test_resident_gradient_sum.py`, `tests/unit/test_resident_trace.py`, `tests/unit/test_resident_pool_snapshot.py`, `tests/unit/test_object_discovery.py`, `tests/unit/test_attention_loop_idiom.py`.

<!-- entry-fields:end -->

Additional owners: [W4-PRODUCT-1](INTEGRATED_COMPILER_PLAN.md#w4-product-1), [FRONTEND-IR-MEDIUM-1](INTEGRATED_COMPILER_PLAN.md#frontend-ir-medium-1).

### 2026-09-11 — Heap publication and reader barrier exploration

Owner: [W4-PRODUCT-1](INTEGRATED_COMPILER_PLAN.md#w4-product-1)

PRs: This change set (`codex/gated-heaps-resident-ad`).

Outcome: Inventoried exclusive epochs, immutable snapshots, ordinary heap loads/stores and Tile completion domains; selected a staged nonmoving design for modeling.

Remaining: Executable interleaving model, consumed artifact contract, epoch retirement, barrier-aware updates and independent device/performance gates remain unimplemented.

Evidence: [Source-linked architecture review](HEAP_BARRIER_ARCHITECTURE_REVIEW.md); source inspection only, no new device result.

<!-- entry-fields:end -->

Reader ownership routes to [W2.4a](INTEGRATED_COMPILER_PLAN.md#w24a);
uncertain completion routes to [DISPATCH-BREAKER](INTEGRATED_COMPILER_PLAN.md#dispatch-breaker).
Sync: `HEAP-BARRIERS-2026-09-11`. Final remark remains exclusive; memory ordering,
reachability and reclamation are independent obligations.

### 2026-09-11 — Bounded heap protocol and device comparison

Owner: [W4-PRODUCT-1](INTEGRATED_COMPILER_PLAN.md#w4-product-1)

PRs: This change set (`codex/gated-heaps-resident-ad`).

Outcome: Added a bounded interleaving model, consumed v1 artifact protocol, split retirement/reclamation and validated graph publication. CUDA/HIP independently execute the new boundaries.

Remaining: Per-object epochs, dirty-work barriers, concurrent final retirement, atomic scope lowering and selector-grade performance remain open. Final remark and whole-pool writes remain exclusive.

Evidence: [Independent packets and comparison](../../../benchmarks/baselines/heap_barriers_20260911/README.md); 69 focused host-WSL tests passed, Ruff passed and mypy ratchet remains zero.

<!-- entry-fields:end -->

[W2.4a](INTEGRATED_COMPILER_PLAN.md#w24a) owns reader-protected reuse;
[DISPATCH-BREAKER](INTEGRATED_COMPILER_PLAN.md#dispatch-breaker) owns failure
retention. The model finds a counterexample when reclamation dependencies are
removed. Device packets check stale publication, newly rooted cycles and
multi-stream payload copies before reuse. Five single-process samples show
higher cost for split retirement; no promotion. Sync: `HEAP-BARRIERS-2026-09-11`.

### 2026-09-11 — Per-object readers and incremental marking

Owner: [W4-PRODUCT-1](INTEGRATED_COMPILER_PLAN.md#w4-product-1)

PRs: This change set (`codex/gated-heaps-resident-ad`).

Outcome: V2 native generation-checked pins, consumed tricolor marking barriers and final retirement during immutable payload-reader scopes execute on CUDA SM120 and ROCm gfx1151. Unrelated unpinned slots reuse while another reader remains admitted.

Remaining: Metadata writers are still exclusive. Asynchronous admission/unpin, allocation during marking, cooperative marking and an atomic multi-writer handshake remain open; no measured overlap or promotion.

Evidence: [Independent packets and model](../../../benchmarks/baselines/incremental_object_heap_20260911/README.md); model 6,690 states / 64,536 transitions; 77 focused host-WSL tests passed.

<!-- entry-fields:end -->

[W2.4a](INTEGRATED_COMPILER_PLAN.md#w24a) owns pin/completion lifetime;
[DISPATCH-BREAKER](INTEGRATED_COMPILER_PLAN.md#dispatch-breaker) owns uncertainty.
Closed reader scopes remain pinned until completion; unknown decrements retain
the pool and cannot be retried. Metadata-scope refusal before submission stays
retryable. Sync: `HEAP-INCREMENTAL-2026-09-11`.


### 2026-09-11 — Asynchronous heap receipts and marking allocation

Owner: [W4-PRODUCT-1](INTEGRATED_COMPILER_PLAN.md#w4-product-1)

PRs: This change set (`codex/gated-heaps-resident-ad`).

Outcome: Optional private polled admission/unpin receipts and grey allocation publication execute independently on CUDA SM120 and ROCm gfx1151. Cancelled admission releases its pin after completion; stale admission does not poison the pool.

Remaining: Metadata writers remain serialized. The reservation model exposes the split validate/write race; native atomic reservation, retirement handshake, cooperative marking and asynchronous finalization/teardown remain open. No measured overlap or promotion.

Evidence: [Device packets and limitations](../../../benchmarks/baselines/async_heap_20260911/README.md).

<!-- entry-fields:end -->

[W2.4a](INTEGRATED_COMPILER_PLAN.md#w24a) owns receipt and reader lifetime;
[DISPATCH-BREAKER](INTEGRATED_COMPILER_PLAN.md#dispatch-breaker) owns retention
after unknown copy/decrement outcomes. Sync: `HEAP-ASYNC-2026-09-11`.


### 2026-09-11 — Native heap handshake and deferred destruction

Owner: [W4-PRODUCT-1](INTEGRATED_COMPILER_PLAN.md#w4-product-1)

PRs: This change set (`codex/gated-heaps-resident-ad`).

Outcome: Schema-3 native graph/retirement try-lock kernels, private polled mark finalization and bounded context-owning teardown workers pass independent CUDA SM120 and ROCm gfx1151 checks.

Remaining: All metadata users must join the gate before removing resident-pool epoch ordering. Cooperative marking, finer transactions and isolated recovery after uncertain teardown remain open. Worker driver calls may block; no measured overlap or promotion.

Evidence: [Device packets, model and limitations](../../../benchmarks/baselines/heap_handshake_20260911/README.md); 77 focused host-WSL tests passed.

<!-- entry-fields:end -->

[W2.4a](INTEGRATED_COMPILER_PLAN.md#w24a) owns reader and receipt lifetimes;
[DISPATCH-BREAKER](INTEGRATED_COMPILER_PLAN.md#dispatch-breaker) owns retained
teardown failures. Sync: `HEAP-HANDSHAKE-2026-09-11`.


### 2026-09-11 — Gated metadata owner and isolated recovery

Owner: [W4-PRODUCT-1](INTEGRATED_COMPILER_PLAN.md#w4-product-1)

PRs: This change set (`codex/gated-heaps-resident-ad`).

Outcome: Opt-in gated owner routes all admitted metadata operations through its owned gate and exposes copied inspection. Spawned CUDA/HIP heap workers support death-confirmed recovery with bounded retained capacity.

Remaining: Legacy snapshot/import migration, broader isolated commands, health-checked replacement, actual driver-hang validation and epoch relaxation remain open. The existing incremental owner is not silently migrated. No measured overlap or promotion.

Evidence: [Independent CUDA/HIP packets and limitations](../../../benchmarks/baselines/gated_heap_20260911/README.md).

<!-- entry-fields:end -->

[W2.4a](INTEGRATED_COMPILER_PLAN.md#w24a) owns pinned readers and copied metadata;
[DISPATCH-BREAKER](INTEGRATED_COMPILER_PLAN.md#dispatch-breaker) owns confirmed
process death and retained failures. Sync: `HEAP-GATED-ISOLATION-2026-09-11`.

### 2026-09-11 — Mixed integer ownership and dtype arithmetic

Owner: [E2E-REAL-6](INTEGRATED_COMPILER_PLAN.md#e2e-real-6)

PRs: Uncommitted engineering increment after #744.

Outcome: Mixed x86 u8/s8 matmul now consumes a replay-verified Schedule/Tile artifact. Scalar reference accumulation uses defined modulo arithmetic. CUDA and HIP each execute 24 basic scalar/vector dtype rows with instruction witnesses.

Remaining: Cohort/breadth constructors; FP8 generic conversion legalization; packed/scaled, complex, bool and Apple arithmetic probes; broader layout and numerical consumers; selector-grade performance evidence.

Evidence: [dtype arithmetic and matrix packets](../../../benchmarks/baselines/dtype_arithmetic_20260911/README.md), `test_scheduled_matmul_consumers.py`, `test_x86_integer_wraparound.py`, `test_dtype_arithmetic_probe.py`.

<!-- entry-fields:end -->

Related owners: [E2E-REAL-6F](INTEGRATED_COMPILER_PLAN.md#e2e-real-6f),
[NUMPOL-CARRIER-1](INTEGRATED_COMPILER_PLAN.md#numpol-carrier-1), and
[LAYOUT-ALG-1](INTEGRATED_COMPILER_PLAN.md#layout-alg-1). The F0 caller census
separates lexical scopes, parameter/rebinding shadowing and relative package
imports. The dtype inventory retains all 26 canonical/planned storage names;
missing layered declarations do not mean no operation-specific implementation.

### 2026-09-11 — FP8 conversions and numerical boundaries

Owner: [NUMPOL-CARRIER-1](INTEGRATED_COMPILER_PLAN.md#numpol-carrier-1)

PRs: Uncommitted engineering increment after #744.

Outcome: Byte-backed scalar/vector FP8 conversions legalize to integer/f32 operations. Exhaustive CUDA/HIP arithmetic, bool logic and bounded complex probes pass. NVIDIA packed/scaled probes exercise nonzero codes and both axes. ROCm packed producers use inherent kernel properties and compiler serialization no longer depends on host runtime linkage.

Remaining: Apple strict subnormal arithmetic fails three comparisons; broader storage, numerical/layout consumers, matrix admission and clean performance evidence remain open.

Evidence: [independent dtype follow-through](../../../benchmarks/baselines/dtype_followthrough_20260911/README.md), `test_lowp_conversions.py`, `test_dtype_arithmetic_probe.py`, `test_packed_dtype_probe.py`.

<!-- entry-fields:end -->

### 2026-09-12 — Reconciliation and wave start

Owner: [NUMPOL-CARRIER-1](INTEGRATED_COMPILER_PLAN.md#numpol-carrier-1)

PRs: Uncommitted follow-through after #745.

Outcome: MASTER_AUDIT now routes sequencing to the live plan and distinguishes implemented SSD, MPI, checkpoint and assertions-enabled toolchain slices from their open boundaries. CUDA serialization now accepts an explicit toolkit path; the arithmetic recorder requires it for NVIDIA and fingerprints ptxas and the disassembler. All 34 scalar/vector rows pass on Super-Bear with CUDA SDK 13.4.1. Snapshot retirement closes admission and polls recorded readers without implicitly synchronizing or freeing.

Remaining: Apple gradual-underflow policy closure, the selected x86 absolute migration and serialized broadcast attention masks are scoped, not implemented in this increment. Snapshot frees and module unload remain synchronous; no new owning-device snapshot-retirement or performance proof is claimed.

Evidence: [CUDA arithmetic packet](../../../benchmarks/baselines/cuda1341_20260912/README.md); WSL tests cover toolkit selection, audit governance, reader admission, eventless retention and snapshot recovery.

<!-- entry-fields:end -->

Additional owners: [E2E-REAL-6](INTEGRATED_COMPILER_PLAN.md#e2e-real-6),
[FRONTEND-IR-MEDIUM-1](INTEGRATED_COMPILER_PLAN.md#frontend-ir-medium-1) and
[W2.4a](INTEGRATED_COMPILER_PLAN.md#w24a). The live records hold acceptance gates.

### 2026-09-12 — Native numerical and packaging slices

Owner: [NUMPOL-CARRIER-1](INTEGRATED_COMPILER_PLAN.md#numpol-carrier-1)

PRs: Uncommitted continuation after #745.

Outcome: Explicit Apple arena gradual/FTZ policy now consumes integer-significand f32 add/sub/mul/div. Both policies pass M1 Max checks over 133,376 boundary/random input pairs per operation. x86 absolute now uses a native Schedule record and Tile producer with replay-derived descriptors and owning-Zen-5 exceptional/ragged proof. Batch/head broadcast attention masks reach native SM120 indexing; the versioned host ABI copies the physical bias extent, and two ragged causal/GQA/window cases pass.

Remaining: Wider Apple floating operations/dtypes/vectors and optimized policy performance; other Graph-owned elementwise/cohort/breadth families; query/key broadcast, Boolean/padding masks and sibling attention backends. No general AD or performance closure is implied.

Evidence: [native slice packets](../../../benchmarks/baselines/native_slices_20260912/README.md), [Apple numerical contract](../../spec/APPLE_ARENA_NUMERICAL_POLICY.md), focused native/compiler and registry checks, and the new absolute lit fixture.

<!-- entry-fields:end -->

Additional owners: [E2E-REAL-6](INTEGRATED_COMPILER_PLAN.md#e2e-real-6) and
[FRONTEND-IR-MEDIUM-1](INTEGRATED_COMPILER_PLAN.md#frontend-ir-medium-1).
The first native-hardware FTZ experiment failed 28 multiply/divide comparisons;
FTZ admission was corrected to use IEEE rounding before output flushing. The
passing packet is from the corrected implementation, not a tolerance change.

### 2026-09-13 — gfx1201 foundation and assertions

Owner: [COMPILER-DEVEX-1](INTEGRATED_COMPILER_PLAN.md#compiler-devex-1)

PRs: Uncommitted engineering increment.

Outcome: Assertions-enabled LLVM/MLIR and Tessera tools on Tajasarus; exact-chip storage binding, corrected RDNA4 fragment store and gfx12 HIPRTC GEMM/forward-attention specialization.

Remaining: General scheduled/Graph route and AD architecture admission, compiler-owned ragged/K-loop matrix proof, LDS/pipelined variants and native-Linux performance attribution. No constructor retirement or performance promotion.

Evidence: [gfx1201 commissioning](../../../benchmarks/baselines/gfx1201_foundation_20260913/README.md); [ROCm owner](../backend/rocm/todo.md#gfx1201-commissioning--2026-09-13).

<!-- entry-fields:end -->

### 2026-09-13 — gfx1201 scheduled integration and fleet validation

Owner: [COMPILER-DEVEX-1](INTEGRATED_COMPILER_PLAN.md#compiler-devex-1)

PRs: Uncommitted engineering increment.

Outcome: Replay-derived gfx1201 f32 unary packages execute; standalone FP16/BF16 backward passes ten cases on each of gfx1201 and gfx1151. Five generators use the LLVM 23 inherent kernel property. LDS/pipelined runtime variants pass correctness but lose the three-shape diagnostic comparison. Super-Bear and Princess-Luna retain clean main checkouts and working rebuilt compilers; candidate gfx1151 tests use an isolated compiler binary.

Remaining: General scheduled matrix/attention and paired public AD, resident saved-LSE, broader masks, emitted-kernel attribution, tuned schedules and native-Linux performance promotion. No Graph constructor retirement or cross-architecture performance transfer.

Evidence: [integration and fleet packet](../../../benchmarks/baselines/gfx1201_integration_20260913/README.md); [ROCm owner](../backend/rocm/todo.md#gfx1201-scheduled-integration--2026-09-13).

<!-- entry-fields:end -->

### 2026-09-13 — Attention ancestry and resident LSE proof

Owner: [E2E-REAL-6](INTEGRATED_COMPILER_PLAN.md#e2e-real-6)

PRs: Uncommitted continuation.

Outcome: ROCm scheduled attention forward/backward now replay native parents and project the actual f32-output ABI. gfx1201 forward and backward compose using resident O/LSE; baseline/resident/optional half-load cases pass independently of the host-output oracle. The gfx1151 native composition remains validated. Static half-load evidence shows higher VGPR usage, not a promotion win.

Remaining: General gfx1201 matrix/attention package admission; automatic paired public AD, mixed-precision cotangent conversion and resident tape ownership; broader masks and runtime kernel attribution. WSL /dev/kfd blocks profiler capability enumeration. No Graph constructor deletion or performance promotion.

Evidence: [paired attention and loading packet](../../../benchmarks/baselines/gfx1201_attention_pair_20260913/README.md).

<!-- entry-fields:end -->

### 2026-09-13 — RDNA4 WMMA operand and accumulator audit

Owner: [NUMPOL-CARRIER-1](INTEGRATED_COMPILER_PLAN.md#numpol-carrier-1)

PRs: Uncommitted continuation.

Outcome: ISA-total RDNA operand signatures; all eleven dense gfx1201 WMMA forms lower with correct A/B provenance, signedness and accumulator storage. The 76-case device corpus includes finite range/subnormals and nonfinite classification; native low-precision accumulation has a distinct rounding oracle. Existing f32/i32 exact checks are preserved.

Remaining: Sparse SWMMAC metadata/packing and native producers, general matrix/Graph packaging, scaled-format consumers, error-budget integration and performance admission. gfx1200 has no inherited device proof; gfx1151 does not acquire FP8 or K32 forms.

Evidence: [WMMA dtype packet](../../../benchmarks/baselines/gfx1201_wmma_dtypes_20260913/README.md).

<!-- entry-fields:end -->


### 2026-09-13 — GFX1201 scheduled package ancestry

Owner: [E2E-REAL-6](INTEGRATED_COMPILER_PLAN.md#e2e-real-6)

PRs: Uncommitted continuation.

Outcome: Static f16 matmul and f16/bf16 forward attention have exact gfx1201 driver/Schedule/Tile/package lineage and twenty device/contract checks. Fixed architecture loss in the direct attention adapter, missing half-wave K partitions in precomputed addresses, and cold-start family registration.

Remaining: General dynamic/epilogue matrix envelopes, automatic public paired AD, resident LSE and reusable asynchronous tapes, backward package admission, sparse SWMMAC storage/index producers. rocprofv3 records API activity but zero dispatch/code-object rows; runtime kernel attribution and clean performance evidence remain open.

Evidence: [scheduled package packet](../../../benchmarks/baselines/gfx1201_scheduled_packages_20260913/README.md).

<!-- entry-fields:end -->


### 2026-09-13 — GFX1201 public attention AD

Owner: [E2E-REAL-6](INTEGRATED_COMPILER_PLAN.md#e2e-real-6)

PRs: Uncommitted continuation.

Outcome: gfx1201 scheduled backward programs retain architecture through direct Tile lowering and native packaging. Public fp16/bf16 GQA Q/K/V native_backward selects its owning architecture, executes the exact package and records artifact-bound physical certificates. The gfx1201 policy remains recompute-only; saved-LSE and unknown architecture requests fail closed. Device tests cover paired public AD and explicit bias/window/softcap/dropout programs.

Remaining: Reusable HIP resident-LSE tapes and asynchronous readers/teardown, sparse SWMMAC packing/index/native producers, broader matrix envelopes and runtime kernel attribution. No performance promotion or sibling device proof.

Evidence: [public AD packet](../../../benchmarks/baselines/gfx1201_public_attention_ad_20260913/README.md).

<!-- entry-fields:end -->


### 2026-09-13 — GFX1201 resident ownership and sparse packing

Owner: [E2E-REAL-6](INTEGRATED_COMPILER_PLAN.md#e2e-real-6)

PRs: Uncommitted continuation.

Outcome: Reusable HIP recompute workspace snapshots inputs, owns queued host calls through retirement and returns independent host results. fp16/bf16 device checks and host failure/interleaving tests pass. Dynamic f16 matmul reuses one image across three runtime shapes and refuses over-bound calls. Immutable fp16/bf16 2:4 packing/index bytes pass six native LLVM SWMMAC probe comparisons with exact disassembly checks. Compiler subprocess profiler injection is excluded after reproducing a linker crash.

Remaining: Saved-LSE admission remains closed; GPU-stream overlap, external device readers, isolated recovery, broader matrix dtypes/epilogues and production sparse Schedule/Tile producers remain open. Raw one-process host timing is diagnostic only. Profiler has API events but no kernel dispatch/code-object/counter rows; no performance promotion.

Evidence: [resident/sparse packet](../../../benchmarks/baselines/gfx1201_resident_sparse_20260913/README.md).

<!-- entry-fields:end -->


### 2026-09-13 — GFX1201 streams readers and sparse Target IR

Owner: [E2E-REAL-6](INTEGRATED_COMPILER_PLAN.md#e2e-real-6)

PRs: Uncommitted continuation.

Outcome: Resident attention uses private nonblocking HIP streams. Explicit same-device read-only output leases order publication, external reads, reuse and retirement; release failures retain retryable ownership. Two gfx1201 owners have intersecting ordered-program event windows in four of five trials with correct gradients. Registered internal f16/bf16 `tessera_rocm.swmmac` validates fragments and lowers through the production Target pass; six MLIR-to-HSACO numerical comparisons are exact. This is an internal physical primitive, not a new public Graph operation, dtype, batching or AD rule.

Remaining: Saved-LSE admission stays closed. General public sparse Schedule/Tile packaging, additional sparse forms, generic reader adoption, isolated uncertain-teardown recovery, calibrated clocks and per-kernel/counter attribution remain open. Event-window intersection is not proof of simultaneous individual kernels. No performance promotion or sibling-device proof.

Evidence: [stream/sparse IR packet](../../../benchmarks/baselines/gfx1201_stream_sparse_ir_20260913/README.md).

<!-- entry-fields:end -->


### 2026-09-13 — Saved LSE and sparse Schedule handoff

Owner: [E2E-REAL-6](INTEGRATED_COMPILER_PLAN.md#e2e-real-6)

PRs: Uncommitted continuation.

Outcome: Explicit gfx1201 saved-LSE retains forward O/LSE in the resident owner's immutable frame across repeated cotangents and external-reader retirement. Auto remains recompute. Internal packed f16/bf16 Schedule and Tile sparse MMA producers verify and lower to Target SWMMAC; six exact native comparisons pass. Valid HIP timing samples are distinguished from independently calibrated selector evidence.

Remaining: Public logical sparse Graph/package integration and additional sparse forms, generic reader adoption, uncertain teardown recovery and calibrated per-kernel attribution remain open. Fresh rocprofv3 records API regions but zero dispatch/code-object/symbol/counter samples. No performance promotion or sibling device proof.

Evidence: [saved/sparse Schedule packet](../../../benchmarks/baselines/gfx1201_saved_sparse_schedule_20260913/README.md).

<!-- entry-fields:end -->


### 2026-09-13 — Logical sparse ownership and floor migration

Owner: [E2E-REAL-6](INTEGRATED_COMPILER_PLAN.md#e2e-real-6)

PRs: Uncommitted follow-through after #747.

Outcome: Logical sparse Schedule packing and tiled accumulation execute on gfx1201; asynchronous attention reader release retains retryability; isolated ANN replacement preserves its explicit device ordinal; x86 floor bypasses the historical Graph-owned Tile emitter through replayed native contracts.

Remaining: Public sparse Graph/package binding, additional sparse dtypes, general public attention composition and mixed cotangents, in-process uncertain teardown isolation, remaining cohort/elementwise/breadth constructors, and calibrated performance admission.

Evidence: [Compiler and owning-device packet](../../../benchmarks/baselines/sparse_ownership_migration_20260913/README.md), native artifact/ownership regressions and refreshed lexical route census.

<!-- entry-fields:end -->

The census counts Graph-annotated entry points, not remaining physical constructors
or device certificates. Floor is a migrated branch inside a function that retains
other Graph-owned routes. Sparse execution consumes ordinary matrix storage, but
its validity output is only checked by the device proof harness; no public runtime
admission is asserted. Explicit device ordinal tests use host IPC and do not prove
recovery from a hardware hang or recovery of resident attention allocations.

### 2026-09-14 — Sparse runtime and isolated attention

Owner: [E2E-REAL-6](INTEGRATED_COMPILER_PLAN.md#e2e-real-6)

PRs: Uncommitted follow-through after #747.

Outcome: Explicit logical sparse compilation/runtime binding validates all device validity words before output exposure and confirms worker teardown. Matching f16/bf16 accumulation lowers through Schedule/Tile/Target to RDNA4 SWMMAC; eight bounded device cases cover two shapes and both accumulator policies. Resident attention composes pending host futures and lossless fp32/fp64 cotangents. The isolated owner confirms process death before replacement and repeats a workload-scoped zero-VJP check. The new zero-first saved-LSE test exposed a missing forward carrier in direct Tile lowering; the fix preserves it only for a reciprocal saved backward companion. Static f32 x86 ceil now projects and replays the native unary contract.

Remaining: Automatic sparse Graph/JIT admission, integer/FP8 sparse packing, general mixed-precision AD, external device pointers across isolation, recovery of an actually hung driver, cohort/breadth and remaining elementwise migrations. No calibrated timing or performance promotion. Zero VJP is a bounded liveness/numerical check, not a global device-health certificate.

Evidence: [Owning-device and host validation](../../../benchmarks/baselines/sparse_runtime_20260914/README.md).

<!-- entry-fields:end -->

Correction to earlier saved-LSE evidence: reciprocal source attributes and successful reused random-cotangent tests were insufficient to prove the direct forward ABI. The new constant cotangent after a zero-first launch exposed uninitialized checkpoint storage. Both checkpoint modes and repeated reuse are now tested against the oracle with the corrected producer.

### 2026-09-14 — Sparse capture, byte formats and scan

Owner: [E2E-REAL-6](INTEGRATED_COMPILER_PLAN.md#e2e-real-6)

PRs: Uncommitted follow-through after #747.

Outcome: Opt-in JIT tracing captures one matmul under an explicit checked 2:4 precondition. Native Graph verification precedes the existing sparse Schedule producer; output storage conversion is emitted on the GPU. Signed i8/i32 and same-format E4M3FN/E5M2/f32 now lower through Schedule/Tile/Target with six gfx1201 instruction/numerical cases. Public attention reverse capture and execution apply exact wider-cotangent checks. Isolated attention admission now compares a nonzero VJP against the shared oracle as well as checking zero VJP. Trailing-axis f32 cumsum moves out of the x86 cohort emitter into replay-verified native Schedule/Tile contracts.

Remaining: Automatic sparse dispatch/arbiter selection, native Graph-to-sparse recipe lowering, mixed FP8/unsigned/INT4 packing, arbitrary public AD composition, actual driver-hang recovery, and remaining cohort/breadth routes. The explicit sparse adapter is frontend code; it does not establish complete MLIR ownership of sparse source lowering. Forced worker death is not a hung-driver or GPU-reset experiment.

Evidence: [Validation packet](../../../benchmarks/baselines/sparse_capture_20260914/README.md). No performance promotion.

<!-- entry-fields:end -->

### 2026-09-14 — Mixed sparse operand contracts

Owner: [E2E-REAL-6](INTEGRATED_COMPILER_PLAN.md#e2e-real-6)

PRs: Uncommitted follow-through after #747.

Outcome: Sparse packages carry an independently typed B operand in their replay identity and input preflight. Schedule/Tile/Target accept both mixed FP8 orders; integer operands carry independent signedness flags consumed by LLVM intrinsic lowering. Noninteger signedness overrides are rejected. No automatic sparsification or performance admission is introduced.

Remaining: Native Graph-to-sparse producer and automatic selection; INT4 packing; arbitrary public AD; actual driver-hang recovery; remaining cohort/breadth migrations. Worker-death proof remains distinct from driver recovery, and replacement numerical checks remain workload-scoped.

Evidence: `tests/unit/test_rocm_sparse_byte_formats.py` owns emitted-instruction and gfx1201 numerical checks, including unsigned values above 127, all sparse index pairs and invalid-pattern refusal. The assertions-enabled gfx1201 build passed 407 focused device/registry tests and 33 additional sparse-contract/audit tests. [Validation packet](../../../benchmarks/baselines/sparse_mixed_20260914/README.md).

<!-- entry-fields:end -->

### 2026-09-14 — INT4 packing and sparse AD boundary

Owner: [E2E-REAL-6](INTEGRATED_COMPILER_PLAN.md#e2e-real-6)

PRs: Uncommitted follow-through after #747.

Outcome: K=32 sparse INT4 retains byte-addressable logical operands with independently signed range contracts. Schedule/Tile/Target carry integer_bits; native lowering packs i32 words and uses the scalar-A/vector-B LLVM intrinsic ABI. Host and compiled GPU guards reject out-of-range values before exposing output. Four gfx1201 cases cover signed/unsigned pairs, sparse indices, emitted instructions and numerical agreement. The AD scoped plan now explicitly requires differentiation of logical matmul before physical packing, preserving derivatives at zero-valued entries.

Remaining: Automatic sparse selection/native Graph lowering, K=64 and other packed storage envelopes, arbitrary public AD and composed derivative ownership. The explicit sparse API remains forward-only; this is not general AD closure or performance promotion.

Evidence: `tests/unit/test_rocm_sparse_byte_formats.py::test_int4_logical_packing_device`; assertions-enabled LLVM on gfx1201. Sub-byte vector truncation and a vector-valued scalar intrinsic operand were rejected during development; final packing uses explicit i32 operations.

<!-- entry-fields:end -->

### 2026-09-14 — Native sparse Graph and logical AD

Owner: [E2E-REAL-6](INTEGRATED_COMPILER_PLAN.md#e2e-real-6)

PRs: Uncommitted follow-through after #747.

Outcome: Declared checked-2:4 half matmul now lowers from Graph to a GPU Schedule kernel in C++, including logical packing, indices, accumulator/output conversion and validity status. JIT projects shape/storage from emitted attributes; it does not invoke the Python recipe producer. The pass declares every emitted dialect and rejects unconsumed module/function/argument/op policies. AD-configured JIT parents retain their logical native_backward path. ROCm matmul adjoints now follow operand order, sum repeated-operand contributions and identify the selected device rather than device zero/gfx1151 by assumption.

Remaining: Automatic default/arbiter sparse-versus-dense selection, larger/nonisolated Graph envelopes, arbitrary public AD composition and higher-order/control-flow closure. The selected forward remains an explicit checked contract; no uncalibrated promotion is introduced. Backward matmul still uses the existing composed runtime GEMM route, not a new general region AD package.

Evidence: `tests/unit/test_sparse_capture.py`, `tests/unit/test_native_matmul_operand_ad.py` and `tests/unit/test_autodiff_rocm_matmul_composed.py`; final results recorded with this wave. No performance promotion.

<!-- entry-fields:end -->

### 2026-09-14 — Native automatic sparse selection

Owner: [E2E-REAL-6](INTEGRATED_COMPILER_PLAN.md#e2e-real-6)

PRs: Uncommitted follow-through after #747.

Outcome: JitFn.compile_sparse_auto lowers isolated half matmul through the native Graph producer. Each K tile uses a wave-wide 2:4 agreement: eligible tiles execute SWMMAC, others execute native dense accumulation. One image accepts changing density without pruning or CPU fallback. The original logical JIT source still owns native_backward. The emitted selection policy is checked against its source policy during package validation.

Remaining: Default-dispatch/performance promotion, broader shapes and composed native regions, arbitrary AD/control-flow/higher-order closure. This explicit automatic policy does not replace incumbents globally and claims no speedup. General AD remains separate from sparse selection.

Evidence: `tests/unit/test_sparse_capture.py::test_native_automatic_sparse_dense_selection` checks dense, sparse, one-invalid-lane mixed tiles and density changes with one artifact on gfx1201; fp16 also checks logical backward. f16 and bf16 cases use the owning device.

<!-- entry-fields:end -->


### 2026-09-14 — Native composed HVP execution

Owner: [W4-PRODUCT-1](INTEGRATED_COMPILER_PLAN.md#w4-product-1)

PRs: Current follow-through after #747.

Outcome: Adds static CPU HVP execution through compiler-produced derivatives and signature-owned output allocation. Tracer nested regions use SCF; scalar extraction tangents, passive predicates and ownership-clone lowering repair nested branch/loop products.

Remaining: Effectful CFG, dynamic outputs, higher derivative orders and general asynchronous HVP frames remain open in [AD execution](AUTODIFF_EXECUTION_PLAN.md). No performance promotion.

Evidence: Native CPU HVP tests on Princess-Luna; implementation and exact-device evidence remain separate.

<!-- entry-fields:end -->


### 2026-09-14 — Saved-product HVP and GPU export

Owner: [W4-PRODUCT-1](INTEGRATED_COMPILER_PLAN.md#w4-product-1)

PRs: Current follow-through after #747.

Outcome: Continuous saved products carry captured tangents into backward JVPs. Typed export feeds bufferized GPU tape lowering. Unsupported effectful while refuses rather than using an empty AST candidate.

Remaining: Effectful CFG, dynamic outputs, higher derivative orders and general asynchronous HVP frames remain open in [AD execution](AUTODIFF_EXECUTION_PLAN.md). No performance promotion.

Evidence: Native CPU nested SAVE curvature and gfx1201 composed/counting-loop HVP tests.

<!-- entry-fields:end -->


### 2026-09-14 — HVP product identity and CUDA execution

Owner: [W4-PRODUCT-1](INTEGRATED_COMPILER_PLAN.md#w4-product-1)

PRs: Current follow-through after #747.

Outcome: Export accepts only products generated by the pass invocation. Vector/matrix composed and counted-loop HVPs execute independently on SM120 and gfx1201.

Remaining: Effectful CFG, dynamic outputs, higher derivative orders and general asynchronous HVP frames remain open in [AD execution](AUTODIFF_EXECUTION_PLAN.md). No performance promotion.

Evidence: `tests/unit/test_native_hvp_execution.py`: 7 passed, 10 skipped per owning GPU host, including four device cases each.

<!-- entry-fields:end -->


### 2026-09-15 — Apple Metal 4 matmul2d family on the canonical GEMM

Owner: [E2E-REAL-6](INTEGRATED_COMPILER_PLAN.md#e2e-real-6)

PRs: Uncommitted follow-through after #751 (APPLE-MATMUL2D-1).

Outcome: The first Apple family enters F2. `tessera_apple.gpu.tensor_view` (a strided rank-2 MTLTensor view over a real f16/bf16/f32/f8E4M3FN/f8E5M2/f4E2M1FN element type) and `tessera_apple.gpu.matmul2d` (MetalPerformancePrimitives matmul2d, operand pair verified against the header table incl. f16 × {f8E4M3FN, f8E5M2, f4E2M1FN}, fp32 accumulator required, K/M/N agreement) are declared with verifiers; `tessera-apple-canonical-gemm-matmul2d` re-forms the shared TilingPass reduction into them and refuses packed 8/4-bit operands whose row stride is off Apple's 128-byte quantum; `tessera-apple-matmul2d-to-call` lowers to a value-producing `gpu.kernel_call` on the runtime's `mtl4_matmul2d_{f16,bf16,lowp}` symbols with the view ABI (inner/outer/stride/byte_offset per operand, storage pair, format code, accumulator) as attributes; the value-lane dispatcher `mtl4_matmul2d` consumes those attributes and re-derives nothing. Both passes are registered standalone and are not in the default `tessera-lower-to-apple_gpu` pipeline; the incumbent MPS/Accelerate route is unchanged (APPLE-TILE-2 rule).

Remaining: Ragged M is zero-padded by the shared tiling pass before the view is taken (the IR states it; the runtime's own ragged support is not yet used); nonzero view origins and padded strides are verified in IR but not yet projected through the dispatcher; default-pipeline admission needs a paired corpus (measured 2026-09-14: FP8 at 0.77–0.93× fp16, fused epilogue not faster); the Python `apple_native` GEMM packager is bypassed only for this canonical-reduction route and is not deleted; the fused bias/act epilogue and the coopmat MSL emitter remain outside this op family.

Evidence: `tests/tessera-ir/phase8/apple_matmul2d{,_invalid,_lowering_invalid}.mlir` (positive, six verifier negatives, one lowering refusal); `tests/unit/test_apple_matmul2d_lane.py` — host-free lowering rows for seven operand pairs and three refusals, plus seven owning-Mac execution rows (M1 Max, macOS 27.0) comparing the lowered call's value-lane result against a float64 oracle built from the exact quantized bytes; `docs/audit/generated/target_ir_membership.md` scores both ops as requiring their contracts. No performance or promotion claim.

<!-- entry-fields:end -->

### 2026-09-15 — matmul2d route: ragged tails, view origins, fused epilogue, paired admission

Owner: [E2E-REAL-6](INTEGRATED_COMPILER_PLAN.md#e2e-real-6)

PRs: #753 (same branch as APPLE-MATMUL2D-1; follow-through commits).

Outcome: The four items the first slice left open are closed on the compiled route. (1) Ragged M/N: `tessera-apple-canonical-gemm-matmul2d` looks through the shared TilingPass zero-pad (`tensor.insert_slice` into a zero constant) and binds the ORIGINAL operand at its true extents; the product is the true [M, N], the trailing slice disappears, and the op states the runtime contract it relies on (`tessera_apple.ragged_tail`, MPP's partial-tile store); the padded form is kept only when the true extents are not a legal MTLTensor view. (2) Nonzero origins and padded strides reach the dispatcher: a static unit-stride `tensor.extract_slice` operand is bound as a view INTO its parent (byte offset, parent row stride, nothing copied, misaligned FP8 origins refused with the verifier's wording); the runtime gained one strided-view entry for every pair (`tessera_apple_gpu_mtl4_matmul2d_view[_epilogue]`, pair codes 0–5 low precision, 10 f16, 11 bf16, 16-bit kernels added to the MSL 4.1 source), the call lowering emits that one symbol with `tessera_apple.pair`, and `runtime._dispatch_gpu_mtl4_matmul2d` passes element origins and strides through after checking the storage row equals the stated stride. The driver's value-call extractor never captured dialect-prefixed keys (`\w+` took only the suffix), so the view ABI had not actually reached the value lane through the driver before; keys now capture whole with the bare suffix kept for older consumers, and the GPU value executor admits the op kinds it used to reject before its own branch. (3) The fused epilogue is an op: `tessera_apple.gpu.matmul2d_epilogue` (optional rank-1 f32 [N] bias, `act` in {none, relu, gelu, silu}; contract: bias added to the fp32 accumulator, activation in fp32, one store; an op fusing nothing is refused) produced by `tessera-apple-matmul2d-fuse-epilogue` from the tracer's broadcast-add + activation chain (single-use only; Tessera's gelu IS the tanh form the kernel evaluates) and lowered to the `_view_epilogue` symbol with the bias as a third operand. (4) Paired admission measured: two independent 20-rep interleaved processes on the M1 Max against what production dispatches today. f16 retained at every shape (0.53–0.96×); bf16 within ±10% of the runtime's own contiguous bf16 entry (admitted only at 512/1024 square, split at ragged 1000, retained at 2048/MLP/decode); the low-precision pairs lose at every GEMM shape (0.72–0.88× fp16 on identical values) and win only at M == 1 decode (1.6–1.7×, lb ≥ 1.40 both runs) with a one-time ~28 ms pack of a 4096² weight. Decision: the default `tessera-lower-to-apple_gpu` pipeline admits the family for the 8/4-bit storage pairs only (`admit=lowp`), because those pairs had no executable route — the pipeline emitted an MPSGraph `matmul_contract` claim for an FP8 GEMM that MPSGraph cannot run — while f16 and bf16 keep the APPLE-TILE-2 incumbent. The bf16 ≤ 1024 and FP8-decode wins are arbiter-bucket candidates (Decision #28), not pipeline admissions.

Remaining: An arbiter bucket entry for bf16 ≤ 1024 square and weight-only FP8 decode (needs the toolchain-versioned cache key of Decision #11); the JIT front door for 8/4-bit storage dtypes (the pipeline now lowers them, `@jit` tracing of FP8 tensors is a separate gate); the bias-operand matmul form (`tessera.matmul` with `bias = "..."`) is dropped by the shared TilingPass before any backend sees it (pre-existing, sibling-visible — the fusion recognizes the broadcast-add form that survives); amortised or fused operand packing before any FP8 arbiter candidacy; the `apple_native` GEMM packager is still not deleted.

Evidence: `tests/tessera-ir/phase8/apple_matmul2d.mlir` (ragged look-through, sub-block origins, epilogue fusion, no fusion with two consumers), `apple_matmul2d_invalid.mlir` (nine verifier negatives), `apple_matmul2d_lowering_invalid.mlir` (two lowering refusals), `apple_matmul2d_pipeline.mlir` (default-pipeline admission: f16/bf16 on the incumbent, FP8/FP4 on the view call, closed admit set); `tests/unit/test_apple_matmul2d_lane.py` — 21 host-free rows and 19 owning-Mac rows (macOS 27.0, M1 Max): seven pairs at true ragged M = 100, ragged N = 100 f16 at a 200-byte row stride, four sub-block origin rows through the dispatcher, four fused-epilogue rows against the decomposed oracle, one `runtime.launch` end-to-end through the value artifact, two refusals; corpus packet `benchmarks/baselines/apple_matmul2d_route_corpus_20260915/` (two processes, README records the decision); `docs/audit/generated/target_ir_membership.md` scores the epilogue op as requiring its contracts. Full lit 442 passed / 40 unsupported on the Mac; the ROCm backend lit suite cannot run on this host. No general speedup claimed.

<!-- entry-fields:end -->

### 2026-09-15 — shared TilingPass preserves the matmul bias/residual epilogue

Owner: [E2E-REAL-6](INTEGRATED_COMPILER_PLAN.md#e2e-real-6)

PRs: follow-up to #753 (sync `TILING-MATMUL-EPILOGUE-2026-09-15`).

Outcome: The shared `tessera-tiling` matmul pattern accepted the tracer's keyword-operand form (`tessera.matmul %a, %b, %bias {bias = "..."}` / `residual`) but copied only the two product operands into the inner tile op while keeping the marker attrs, so the pass failed its own verifier and, had it not, the epilogue would have been silently dropped (Decision #32). The inner K step is now the plain product with the markers stripped, and the epilogue is re-applied once on the logical [M, N] result as ordinary Graph IR — bias broadcast per output column then added, residual added after — the same form `nn.functional.linear` already emits, so every backend's canonical-GEMM recognizer sees the unchanged nest and the Apple `matmul2d_epilogue` fusion consumes the bias-operand matmul end to end.

Remaining: No backend executes a multi-op value program, so a biased matmul still runs the epilogue outside the kernel except on Apple's fused route; the arbiter bucket, the `@jit` front door for 8/4-bit storage tensors and the strict route ledger re-seal stay open from the previous entry.

Evidence: `tests/tessera-ir/phase2/tiling_matmul_epilogue.mlir` (bias, bias + residual on ragged M, residual alone; inner op carries no marker), `tests/tessera-ir/phase8/apple_matmul2d_bias_operand.mlir` (bias-operand matmul → `gpu.matmul2d_epilogue`); full lit 445 passed / 40 unsupported on the Mac (host-free fixtures); registry, tiling and lane unit gates green. No device or performance claim.

<!-- entry-fields:end -->

### 2026-09-15 — the @jit front door for 8/4-bit storage tensors on Apple GPU

Owner: [E2E-REAL-6](INTEGRATED_COMPILER_PLAN.md#e2e-real-6)

PRs: follow-up to #753 (sync `LOWP-FRONT-DOOR-2026-09-15`).

Outcome: Probing `@jit(target="apple_gpu")` with FP8 / FP4 operands found a hollow green, not a missing feature: the tracer named every numpy dtype it did not know "f32", so an FP8 program traced as an f32 program; the Graph IR spellings of the low-precision types were unparseable (`xxf8E4M3FN`, `!tessera.fp4_e2m1` — a type no dialect defines); and the MPS matmul dispatcher, finding no lane for the dtype, computed the product on the host with numpy while the artifact reported `native_gpu` (the result matched a float64 oracle exactly). Fixed end to end: the tracer names `float8_e4m3fn` / `float8_e5m2` / `float4_e2m1fn` by their canonical storage dtypes; Graph IR spells them as the MLIR 23 builtins `f8E4M3FN` / `f8E5M2` / `f4E2M1FN` (reverse table too); a matmul over storage-only dtypes types its result fp32 (Decision #15a, matching `_promote_two`); the dispatcher routes every low-precision pair — fp8×fp8, fp4×fp4 and the weight-only f16×{fp8, fp4} — through the strided-view MPP matmul2d lane the compiled route also uses, and a dtype with no lane now goes through the fallback funnel (strict dispatch raises) instead of standing in silently; the apple_gpu matmul capability and manifest rows declare the three dtypes. Two shared-frontend fixes rode along: `to_graph_ir_module` verifies legality against the target the caller compiles for instead of always the CPU table, and the jit's Graph IR renders verify against the jit's own target — before this the tracer authority silently lost every GPU-only-dtype program to the AST fallback.

Remaining: The compiled (value-mode) route for a traced FP8 program is proven host-free through the default pipeline (`op_kind = "mtl4_matmul2d"`); the `@jit` default mode still dispatches through the MPS envelope, which now reaches the same Metal kernel — moving the default mode onto the compiled artifact is E2E-REAL-6's general migration, not this item. The arbiter bucket for bf16 ≤ 1024 stays open; the FP8-decode "bucket" is a quantization-policy decision (a different program), not an arbiter choice, and is recorded as such. The bias-operand matmul form dropped by the shared TilingPass is addressed separately (sync `TILING-MATMUL-EPILOGUE-2026-09-15`).

Evidence: `tests/unit/test_apple_jit_lowp_front_door.py` — 13 host-free rows (tracer naming, builtin spellings with round trip, fp32 matmul result, a traced FP8 program lowering to the matmul2d view call through the default pipeline) and 7 owning-Mac rows (five pairs through `@jit` on the Metal view lane against an exact-bytes float64 oracle, f16 unchanged, strict-dispatch refusal of a lane-less dtype); jit, frontend-authority, Apple value-lane, manifest and capability suites 507 passed on the Mac; `dtype_flow`, `apple_target_map` and `support_table` regenerated. No performance claim.

<!-- entry-fields:end -->

### 2026-09-15 — bf16 matmul2d arbiter bucket: measured, retained

Owner: [E2E-REAL-6](INTEGRATED_COMPILER_PLAN.md#e2e-real-6)

PRs: follow-up to #753 and #755 (sync `APPLE-MATMUL2D-BUCKET-2026-09-15`).

Outcome: The bf16 ≤ 1024 "win" the paired admission corpus reported (5–12% median, view entry over the production bf16 route) is now an arbiter bucket in the strict route ledger — `retune_matmul2d_bf16` at 512, 1024 and 2048 square, the runtime's contiguous MPP `matmul2d` entry against the strided-view entry the compiled route dispatches, five independent runs of nine interleaved trials each — and the production bf16 route (`_mtl4_route_matmul2d_bf16`) consults that ledger per exact shape and dispatches the view entry only when it is promoted. On the first seal **every row retained the contiguous incumbent** (six rows, both timing domains): on device time the two entries are within ±3% (the same kernel), and end to end the view entry is 6–14% slower at 512 and 20–42% slower at 1024. The corpus's earlier win was not the kernel: its incumbent went through the production wrapper (a bf16 result cast and its buffer path) while its candidate did not, so the comparison measured wrapper overhead. The bucket exists so a later runtime change can promote it with evidence; nothing is promoted today.

Remaining: The view-entry wrapper zero-fills its fp32 output on the host, which is most of its end-to-end penalty at 1024; that is a wrapper cost to remove before re-measuring, not a kernel finding. The FP8-decode "bucket" from the corpus is a quantization-policy decision (a different program), recorded as such and not an arbiter row.

Evidence: `benchmarks/baselines/apple_strict_route_ledger.json` (24 decisions, 6 ineligible, macOS 27.0 / SDK 27.0, zero routes moved among the previous 18) and `apple7_legacy_retune_multi_run.json`; `tests/unit/test_apple_legacy_retune_benchmark.py` (14 passed on the Mac incl. the live-host admission row and the host-free consumer test that the bf16 route dispatches the view entry only on a promotion). No performance claim.

<!-- entry-fields:end -->

### 2026-09-15 — matmul epilogue markers from both frontends, activation after the reduction

Owner: [E2E-REAL-6](INTEGRATED_COMPILER_PLAN.md#e2e-real-6)

PRs: follow-up to #754 (review), sync `MATMUL-EPILOGUE-MARKERS-2026-09-15`.

Outcome: The #754 rewrite was unreachable from Python: `ops.matmul(a, w, bias=b)` traced as a three-operand `tessera.matmul` with no `bias` marker (the tracer moves keyword tensors into the operand list and drops the keyword), so `MatmulOp::verify` rejected it before tiling. The shared `apply_presence_flags` — one implementation both frontends call, held to parity by the differential certificate — now also emits the string markers the C++ verifier reads (`bias = "bias"`, `residual = "residual"`) for the bound epilogue operands of `tessera.matmul`. The `activation` attribute was left on every inner K step, where a consumer honoring it would apply it per partial product; the tiling pass now strips it and materializes `tessera.<act>` once on the completed product, between the bias and the residual (the public `gemm` contract), and leaves the nest untouched for an unknown activation name rather than guessing.

Remaining: Outside Apple's fused `gpu.matmul2d_epilogue` no backend executes the re-applied epilogue inside the kernel; the value lane still executes single-call programs only.

Evidence: `tests/unit/test_tiling_matmul_epilogue.py` (tracer and AST markers, contract-order tiling of a traced matmul through tessera-opt, host-free); `tests/tessera-ir/phase2/tiling_matmul_epilogue.mlir` (bias + gelu + residual order, activation alone, unknown activation left alone); full lit and the frontend/trace/jit suites green on the Mac. No device or performance claim.

<!-- entry-fields:end -->

### 2026-09-15 — query/key-axis broadcast attention masks reach native SM120 indexing

Owner: [FRONTEND-IR-MEDIUM-1](INTEGRATED_COMPILER_PLAN.md#frontend-ir-medium-1)

PRs: continuation of the 2026-09-12 batch/head increment (sync `ATTN-QK-BROADCAST-2026-09-15`).

Outcome: A raised attention bias may now broadcast on any of its four axes: a key-padding row (`[1,1,1,K]`) and a per-(batch, head, query) column (`[B,Hq,Q,1]`) pass source recognition (`loop_idioms` accepts extent 1 and index 0 on the query and key axes), symbolic binding (`SymbolicDimEqualityPass`), Schedule selection and `bias_shape` carriage (`PMPasses`), Tile lowering that keeps the physical block (`TileIRLoweringPass` slices `[1,tkv]` / `[tq,1]` blocks and `ScoreBiasOp` verifies each axis as 1 or the scores extent), and SM120 kernel indexing (`NVIDIALowering` reads index 0 with stride 1 on every broadcast axis). The f32 broadcast host ABI is versioned to v3 and carries all four physical extents (`BiasB`, `BiasH`, `BiasQ`, `BiasK`); the launcher validates each against 1 or the logical extent and copies only the physical storage. Empty-row refusal now judges rows on the logical broadcast view, so a fully masked key-padding row is refused before launch. Six B=2 ragged causal/GQA/window buckets (batch/head, key-padding and per-query forms at Q/K = 3/5 and 5/3) match the numpy oracle on the owning RTX 5070 within 6e-8. ROCm's canonical streaming attention refuses a broadcast `score_bias` block instead of indexing it at the scores' shape.

Remaining: Boolean/padding masks as a first-class operand (today a padding mask is an additive `-inf` row), sibling-target consumers (Apple, ROCm, x86 raised attention have no broadcast proof; ROCm fails closed), f16/bf16 storage for broadcast bias, and any performance admission. The 2026-09-12 batch/head evidence was re-recorded here under the v3 ABI; the v2 packets are historical.

Evidence: [attention broadcast packets](../../../benchmarks/baselines/attention_broadcast_20260915/README.md) (six device rows, source fingerprints), `tests/unit/test_attention_broadcast.py` (recognition, physical-shape guards, tampered-schedule refusal, device rows behind `TESSERA_TEST_RAISED_ATTENTION=1`), `tests/tessera-ir/phase3/flash_attn_broadcast_bias_tile_lowering.mlir` (host-free Tile lowering of both forms), `ninja -C build-nvidia-cuda check-tessera-nvidia` 61/61 on Super-Bear.

<!-- entry-fields:end -->

### 2026-09-15 — probed admission and health-checked replacement of isolated heap workers

Owner: [W4-PRODUCT-1](INTEGRATED_COMPILER_PLAN.md#w4-product-1)

PRs: continuation of the 2026-09-11 gated owner (sync `HEAP-REPLACEMENT-HEALTH-2026-09-15`).

Outcome: A spawned `IsolatedHeapPool` worker is admitted only after its own in-process device probe verifies the admitted producers on the ordinal it will own (int8 allocation → one live slot of the payload width → bitwise readback through a generation-checked pin → empty graph published, marked and reclaimed); the ready message carries the probe tag and anything else is a failed admission, torn down with confirmed death. `replacement()` refuses until the predecessor's death is confirmed and then admits a fresh worker whose own probe is the health evidence. Admission has its own bound because it compiles the metadata kernels in-process. Both owning devices (RTX 5070, gfx1151) pass all nine recorder proofs. Prerequisite repair: since 100a2980 the storage packager expanded low-precision conversions while six validators (heap pool, SSD, ANN, exception heap, public result, gradient sum) replayed a hand-copied pipeline, so every such package was refused on device with "disagrees with native replay"; the pipeline is now spelled once (`native_gpu_storage.ARENA_PIPELINE`) with a drift test.

Remaining: Legacy snapshot/import callers still need an explicit gated producer; the stopped-worker fault is injected, not a reproduced driver hang; the probe proves the selected workload on one ordinal now, not global driver health; automatic replacement policy, broader isolated commands and epoch relaxation stay open. No measured overlap or promotion.

Evidence: [CUDA/HIP replacement packets](../../../benchmarks/baselines/gated_heap_replacement_20260915/README.md) (nine proofs each, source fingerprints), `tests/unit/test_gated_heap.py` (admission refusals and the probe's three failure modes, host-free), `tests/unit/test_arena_replay_pipeline.py`.

<!-- entry-fields:end -->

### 2026-09-16 — batched geometric products execute through the MLIR/LLVM backbone

Owner: [W6.4](INTEGRATED_COMPILER_PLAN.md#w64)

PRs: domain-support stream, first slice (sync `GA-NATIVE-BATCHED-2026-09-16`); also carries the `FLEET-REDS-2026-09-15` NVIDIA/x86 sibling entries Codex asked for on #761.

Outcome: `ExpandProductTable` lowers any static `[..., dim]` operand pair — the compile-time Cayley table emitted once inside an scf.for nest over the leading axes, result tensor as iter_arg, grade pruning unchanged, dynamic extents refused. `libtessera_jit` registers the Clifford dialect and runs GradeFusion + ExpandProductTable ahead of tessera→linalg, so `tessera_clifford.geo_product` compiles and executes through one-shot bufferization and LLVM; `_jit_boundary.jit_clifford_geo_product` and `package_clifford_geo_product_cpu` + `runtime.launch` are the consumers, and the execution matrix carries the `cpu` row the domain proof ladder now counts. Parity with the standalone GA reference for rank 1/2/3, grade-restricted products equal the projection with untouched coefficients zero, non-commutativity preserved, no fallback; verified on the M1 Max, Princess-Luna (Zen 5) and Super-Bear (Zen 2), with the Clifford lit suite 17/17 on all three. Princess-Luna and Super-Bear now configure the Clifford backend ON — until now only the Mac could build the domain dialects.

Remaining: rotor sandwich fold and the remaining Clifford ops have no lowering behind the JIT; ragged batches (static extents only); the GPU package route through the arena pipeline; the acceptance's separate dispatch/allocation/memory/kernel-time measurements (no performance claim); the Python-emitted x86/ROCm/Apple GA kernels stay the device lanes until displaced by measured evidence. EBM's traceable quadratic energy loop is the next domain slice.

Evidence: `tests/unit/test_clifford_jit_native.py` (three CPU hosts), `src/solvers/clifford/test/ir/passes/expand_batched.mlir` + `expand_rejects_dynamic.mlir`, `docs/audit/generated/domain_proof_ladder.md` (`cpu=1` under GA), `docs/audit/generated/runtime_execution_matrix.md`.

<!-- entry-fields:end -->

### 2026-09-16 — the Clifford product family executes behind the MLIR/LLVM JIT

Owner: [W6.4](INTEGRATED_COMPILER_PLAN.md#w64)

PRs: domain-support stream, second slice, stacked on the batched-product slice (sync `GA-NATIVE-FAMILY-2026-09-16`).

Outcome: One compile-time table now carries every linear/bilinear op the standalone GA reference defines: `ExpandProductTable` lowers wedge (disjoint blades only), left contraction (result grade = grade(b) − grade(a)), inner and norm (one scalar per multivector, typed `[..., 1]`, norm clipped before sqrt), reverse / grade involution / conjugate (per-grade signs), Hodge star (`reverse(a)·I` as a signed blade permutation), standalone grade projection, and rotor sandwich expanded to `gp(gp(R,x), reverse(R))` behind an `expand-rotor-sandwich` pass option — the fused marker still survives the standalone pipeline for backends with a sandwich kernel, and the JIT enables the expansion. All batched through the shared loop nest. `jit_clifford_op` / `package_clifford_cpu` are the consumers; the execution-matrix row is renamed `cpu` / `cpu_clifford_llvm_jit` (op family `clifford`). Nine ops × single/batched/rank-3 shapes match the reference on the M1 Max, Princess-Luna and Super-Bear; grade projection keeps only the listed grades; a unit rotor preserves the vector norm; Clifford lit 19/19 on all three.

Remaining: `exp`/`log` (series with branch handling) and the field ops (ext_deriv, codiff, vec_deriv, integral) have no native lowering; ragged batches; the GPU package route through the arena pipeline; the acceptance's separate overhead/traffic/kernel-time measurements (no performance claim); the Python-emitted device kernels stay the x86/ROCm/Apple GA lanes until measured against this path.

Evidence: `tests/unit/test_clifford_jit_native.py` (three CPU hosts), `src/solvers/clifford/test/ir/passes/expand_family.mlir` + `expand_family_rejects.mlir`, `docs/audit/generated/runtime_execution_matrix.md`.

<!-- entry-fields:end -->

### 2026-09-16 — the Clifford family reaches ROCm and sm_120 through the arena pipeline

Owner: [W6.4](INTEGRATED_COMPILER_PLAN.md#w64)

PRs: domain-support stream, third slice, on the family branch (sync `GA-NATIVE-GPU-2026-09-16`).

Outcome: A domain op now reaches a GPU package without a Python-emitted kernel. `native_clifford_gpu.py` writes only the kernel skeleton (one thread per multivector: loads, a rank-1 `tessera_clifford` op on tensors, stores); `ts-clifford-opt` — which now registers gpu/llvm/memref to parse kernels — expands it through the same GradeFusion + ExpandProductTable lowering the CPU JIT runs; the arena pipeline's canonicalization folds the tensors away, leaving scalar arithmetic the compiler emitted (64 products for the full Cl(3,0) product, 24 under a grade-2 restriction, counted in the arena IR); `build_native_gpu_storage` packages it and the native storage binding launches it. `runtime.launch` rows `rocm` / `rocm_clifford_native_compiled` and `nvidia_sm120` / `nvidia_clifford_native_compiled` share one executor; `package_clifford_native` builds the artifacts. Ten ops × single/batched/rank-3 shapes match the standalone GA reference on gfx1151 (Princess-Luna), gfx1201 (Tajasarus, Clifford backend now ON there too) and sm_120 (Super-Bear), 49/49 tests each; the domain proof ladder's GA row gains `nvidia_sm120=1` and a second `rocm` row from the matrix.

Remaining: the Python-emitted `rocm_clifford_compiled` / `x86_clifford_compiled` / Apple kernels are untouched and remain the device lanes until this route is measured against them (per-call host transfers here are correctness-only); `exp`/`log` and the field ops; ragged batches; an Apple package route (the Apple arena lane is MSL, not this pipeline); the acceptance's separate overhead/traffic/kernel-time measurements. EBM's traceable quadratic energy loop is the next domain slice.

Evidence: [three device packets](../../../benchmarks/baselines/clifford_native_gpu_20260916/README.md) with source fingerprints, `tests/unit/test_clifford_native_gpu.py` (host-free expansion/pruning half on the Mac; device half on the three boxes), `benchmarks/record_clifford_native_gpu.py`, `docs/audit/generated/domain_proof_ladder.md`.

<!-- entry-fields:end -->

### 2026-09-16 — the EBM quadratic energy loop executes through the MLIR/LLVM backbone

Owner: [W4-PRODUCT-1](INTEGRATED_COMPILER_PLAN.md#w4-product-1)

PRs: domain-support stream, fourth slice (sync `EBM-NATIVE-QUADRATIC-2026-09-16`); co-owner [AD-SOLVER-IFT-1](INTEGRATED_COMPILER_PLAN.md#ad-solver-ift-1).

Outcome: The GA/EBM review's first acceptance clause for "an energy is a typed program" is met on the CPU lane. `E(y, x) = 0.5·Σ(x − y)²` is a Graph IR function marked for reverse-mode; inside `libtessera_jit` the paired autodiff pass derives `@E__bwd`, the EBM dialect's first lowering pass (`tessera-ebm-lower-langevin`) turns `energy` / `inner_step` / `langevin_step` into arith + linalg over that gradient with Philox-4x32-10 / Box-Muller noise generated in a `linalg.generic`, and the K-step `scf.for` compiles as one function — no per-step host gradient or noise transfers. `langevin_step` gained a variadic `captures` operand for the energy's context. The declared RNG policy is stated in the pass and mirrored bit-for-bit by `native_langevin.reference_langevin_loop`; forward energy and the T = 0 step match the independent formulas; fixed-key samples match for 1, 5 and 12 steps; `runtime.launch` row `cpu` / `cpu_ebm_langevin_llvm_jit`. Verified on the M1 Max, Princess-Luna (Zen 5) and Super-Bear (Zen 2); EBM lit 14/14 on the Mac, both Zen hosts and Tajasarus under its assertions LLVM. Three compiler gaps found and closed on the way: `tessera.sub` had no adjoint (reverse-mode stopped on any subtraction); `tessera.unsqueeze` / `tessera.broadcast`, which the sum-reduce adjoint emits, had no linalg lowering (no reduce-sum gradient could reach the JIT); and the JIT's DPS out-param rewrite did not follow intra-module call sites.

Remaining: a GPU package for the loop (the tensor-level gradient needs the tile pipeline, not the arena skeleton); nonlinear and manifold energies (sphere / bivector integrators fail closed); opaque-callback energies keep the reported reference path; no performance measurement. The engineering plan for all of these — with the measured stop of each existing device route on this loop (the serial native-tape route rejects it for exactly two reasons; the cooperative route has no EBM Schedule op) — is [EBM_NATIVE_LOOP_ARCHITECTURE.md](../domain/EBM_NATIVE_LOOP_ARCHITECTURE.md). Pre-existing and unrelated: every bf16 JIT test fails on Super-Bear (Zen 2, no AVX512-BF16) with unresolved `_mlir_ciface_*` symbols — bisected by building the JIT from main's sources on that box; bf16 JIT proof belongs on the Zen 5 hosts.

Evidence: `tests/unit/test_ebm_native_langevin.py` (three CPU hosts), `src/solvers/ebm/test/ir/passes/lower_langevin_quadratic.mlir` + `lower_langevin_rejects.mlir`, `tests/tessera-ir/phase2_autodiff/autodiff_paired_sub.mlir`, `docs/audit/generated/domain_proof_ladder.md` (`cpu=1` under EBM).

<!-- entry-fields:end -->

### 2026-09-16 — the EBM Langevin loop runs as one cooperative kernel on gfx1151, gfx1201 and sm_120

Owner: [W4-PRODUCT-1](INTEGRATED_COMPILER_PLAN.md#w4-product-1)

PRs: domain-support stream, fifth slice, on the EBM branch (sync `EBM-NATIVE-GPU-2026-09-16`); co-owner [AD-SOLVER-IFT-1](INTEGRATED_COMPILER_PLAN.md#ad-solver-ift-1).

Outcome: The tensor-level gradient is lowered *inside* a device kernel, through the compiler alone. A new core pass, `tessera-row-program-to-gpu` (`RowProgramToGPUPass.cpp`), consumes the loop after paired autodiff → EBM lowering → `tessera-to-linalg` → inlining and emits one `gpu.func`: one block per row, one lane per feature (≤ 1024), the K-step `scf.for` carried in registers, Philox-4x32-10 / Box–Muller inside the loop, ordered shared-memory reductions (sequential over the feature index, so reduced quantities are bit-exact with a host f32 fold, not tolerance-based), guarded `!llvm.ptr<1>` loads/stores and the arena marker — then `build_native_gpu_storage` packages it and the native tensor binding launches it. `tessera-opt` now registers the EBM and Clifford dialects and passes when built with them, so the whole chain is one driver invocation (T1). Rows `rocm` / `rocm_ebm_langevin_native_compiled` and `nvidia_sm120` / `nvidia_ebm_langevin_native_compiled`; `package_ebm_langevin_native`; `compiler/native_row_program.py` is the packaging seam any `[rows, features]` domain program reuses. Measured on gfx1151 (Princess-Luna), gfx1201 (Tajasarus, assertions-ON driver) and sm_120 (Super-Bear): one launch per loop, bit-exact with the declared policy for K ∈ {1,4,5,8,12}, F ∈ {5,…,1024}, T ∈ {0,…,0.7} — worst abs error 0 in every packet row, the device f64 `log`/`cos` included; the row-normalization reduction program bit-exact with the sequential fold. The scoped G2 Tile contract was not needed: the lowered linalg loop already *is* the scalar body, and one pass over upstream dialects maps it (Decision #31 — no second scalar emitter); G1 serial residency stays scoped only as the fallback outside the envelope. The assertions-enabled driver falsified three promises every NDEBUG driver ran green through — `--inline` needs the LLVM dialect's promised inliner interface (now registered in `tessera-opt`), the pass emitted `tile.alloc_shared` without declaring the tile dialect, and its `tessera.*` provenance attributes load the Tessera dialect an op-free input never loaded (both now declared) — Decision #19's standing lesson, third instance. sm_120 exposed a fourth defect: `math.sqrt` on the NVVM route lowers to libdevice's `__nv_sqrtf`, whose precise branch is gated on a reflect flag MLIR never sets, so the kernel ran `MUFU.SQRT` and one row of the reduction proof was 1 ulp off; the emitter now calls libdevice's rounding-explicit `__nv_fsqrt_rn` on NVIDIA (`convert-gpu-to-nvvm` outlaws the LLVM math intrinsics) and pins `llvm.intr.sqrt` on ROCm and every other libdevice f32 path on that route is recorded as unmeasured.

Remaining: no performance claim — the Python-emitted `rocm_ebm_langevin_compiled` / `x86_ebm_langevin_compiled` / Apple kernels remain the lanes until the dispatch/allocation/traffic/kernel-time comparison is recorded with `route` and `latency_source`; rows wider than 1024 features, data-dependent control flow and cross-row-coupled energies fail closed; nonlinear (N1) and manifold (M1 sphere, M2 bivector) energies are the next slices and are row programs the same emitter maps; no Apple package route (the Apple arena lane is MSL). Super-Bear's `build-nvidia-cuda/` tree was configured without the EBM/Clifford backends and is now reconfigured with both ON (the runtime resolves that tree's driver).

Evidence: [three device packets](../../../benchmarks/baselines/ebm_langevin_native_gpu_20260916/README.md) with source fingerprints, `tests/unit/test_ebm_native_langevin_gpu.py` (host-free chain/replay/envelope half on the Mac; device half on the three boxes), `benchmarks/record_ebm_langevin_native_gpu.py`, `tests/tessera-ir/phase2_autodiff/row_program_to_gpu_langevin.mlir`, `tests/tessera-ir/phase_f5/row_program_to_gpu_reduce.mlir`, `docs/audit/generated/domain_proof_ladder.md` (EBM row gains the two device columns from the matrix).

<!-- entry-fields:end -->

### 2026-09-16 — nonlinear energies and the sphere integrator reach the same kernel

Owner: [W4-PRODUCT-1](INTEGRATED_COMPILER_PLAN.md#w4-product-1)

PRs: domain-support stream, sixth slice (sync `EBM-NONLINEAR-MANIFOLD-2026-09-16`); co-owner [AD-SOLVER-IFT-1](INTEGRATED_COMPILER_PLAN.md#ad-solver-ift-1).

Outcome: The EBM acceptance's remaining two clauses close on the CPU lane and on all three GPUs. **N1:** the integrator is energy-agnostic, so admitting an energy is a differentiation question — `native_langevin` now carries quadratic, Huber and softplus, each with its own numpy oracle, and the one gap this found was `softplus`, whose adjoint was a `custom_adjoint_call` placeholder (a host VJP, i.e. the per-step transfer the acceptance forbids). It gained a native adjoint, `dy · sigmoid(x)`, the stable form the architecture doc specified, plus a stable linalg lowering `max(x,0) + log1p(exp(-|x|))`; no kernel on any energy now contains a placeholder. **M1:** `manifold = "sphere"` has a native integrator — per row the gradient and the noise are projected to the tangent plane and the step is retracted by normalization, the dot products and norms being sequential f32 row sums. Both singularities fail closed per Decision #21a instead of being repaired silently: `langevin_step` gained an optional per-row i32 status word (bit 0 entry ‖x‖ ≠ 1, bit 1 retraction underflow with the previous state kept) that the loop ORs across steps and returns, and the JIT boundary admits i32 as a storage dtype beside its existing i64/i8 residual types. All six energy × manifold programs compile through one `tessera-opt` invocation and replay; the sphere kernels carry (state, key, key, status) in registers with four ordered row reductions per step, so the device reproduces the host's sequential fold rather than merely matching within a tolerance. Verified on the Mac (CPU lane 35/35), gfx1151, gfx1201 (assertions-ON driver) and sm_120 (device 32/32 each).

Three defects in the row-program emitter surfaced doing it, all fail-open or unsafe: the pass could fail with **no diagnostic at all**; a failed slot lookup returned an empty slot that every caller then **indexed out of bounds**; and a reshape copied a slot *reference* into the same `DenseMap`, so the insertion could rehash and the reshaped row arrived empty — the actual cause of the quadratic-sphere refusal. The pass now refuses to fail without naming an op, every refusal names the value and its producer, and multi-result `linalg.generic` is admitted (the Huber backward yields dstate and dcontext from one body). Separately, registering the EBM and Clifford dialects in `tessera-opt` (T1, last slice) had turned three `tests/tessera-ir/phase7` fixtures red on main: they predated their dialects and named placeholder ops with f32 attributes the real ops reject, parsing only under `--allow-unregistered-dialect`, which stops covering an op once its dialect is registered. CI does not run that suite. All three now use the real spellings and verify with no escape flag; rotor construction and an annealing schedule are genuinely absent and are named as gaps rather than faked (Decision #29).

Remaining: `manifold = "bivector"` still refuses (it needs the Clifford `grade` op inside the EBM lowering and a build dependency); no performance measurement, so the Python-emitted kernels remain their lanes; the device packets for this slice are the test suite rather than a recorder packet.

Evidence: `tests/unit/test_ebm_native_langevin.py` (35 on the Mac, Princess-Luna and Super-Bear), `tests/unit/test_ebm_native_langevin_gpu.py` (32 on each of gfx1151, gfx1201, sm_120), `src/solvers/ebm/test/ir/passes/lower_langevin_sphere.mlir`, `tests/tessera-ir/phase_f5/row_program_sphere_status.mlir`, `tests/tessera-ir/phase7/` (19/19).

<!-- entry-fields:end -->

### 2026-09-16 — the bivector integrator and the overhead measurement close the EBM stream

Owner: [W4-PRODUCT-1](INTEGRATED_COMPILER_PLAN.md#w4-product-1)

PRs: domain-support stream, seventh slice (sync `EBM-BIVECTOR-OVERHEAD-2026-09-16`); co-owner [AD-SOLVER-IFT-1](INTEGRATED_COMPILER_PLAN.md#ad-solver-ift-1).

Outcome: **M2 and the measurement.** `manifold = "bivector"` projects the gradient and the noise onto a grade, takes the Euclidean affine step and projects once more to clear float leakage — the reference's own structure — using `tessera_clifford.grade` rather than a second grade projection in the EBM pass (Decision #31). TesseraEBM therefore depends on TesseraClifford for this integrator only, gated on that backend: built without it the integrator fails closed naming the missing option instead of silently taking a Euclidean step. `grade` and `algebra` are semantic keys and are never defaulted; the entry grade is reported per row, never repaired. The state stays exactly in the subspace over a 100-step chain, and grade 1 is admitted beside grade 2, so the integrator is parameterized rather than a hard-wired so(3).

Getting it onto the device took two compiler changes. The **Clifford expansion** gained an elementwise path for DIAGONAL blade maps (grade, reverse, grade involution, conjugation): the per-multivector extract/insert loop nest expressed a diagonal scale, which was both slower and opaque to any consumer that maps the trailing axis to lanes. It is a select against per-blade keep/negate masks, **not** a multiply by a {0, ±1} mask, because `0 * NaN` is NaN while this map *drops* a blade — the drop stays exact and the family's "sign and permutation maps, never multiplies" contract holds; rank 1 keeps the `from_elements` path the Clifford GPU skeleton route folds to scalars. The **row-program emitter** learned per-feature constant tables: a rank-1 constant read along the feature axis is a compile-time table the lane indexes, emitted as a private constant global. That carries the grade masks into a cooperative kernel and generalizes to any per-feature weight vector. Separately, the EBM lowering now **fails when one of its ops survives the pass**: a pattern that refuses an op emits a diagnostic and declines to rewrite, and the greedy driver still reported success, so the pass exited 0 with an error printed and an unlowered op in the output.

**The overhead measurement** the GA/EBM review asks for, at temperature 0 where both routes compute the same iterated descent, with the agreement checked before any timing is kept and both dispatched at the same wrapper depth. The native route is **flat in K** — about 1.1 ms whether the loop runs 1 step or 32, and whether the state is 32 or 16384 elements — because the whole loop is one launch and one host round trip; the Python-emitted route costs about 1.8 ms **per step**, because its kernel takes `(y, grad)` and the loop cannot live inside it. On gfx1151 that is 1.6x at K = 1 and 53x at K = 32. It is a dispatch result, not a kernel-speed claim: wall clock only, since neither WSL2 ROCm box exposes `/dev/kfd` and `rocprofv3` returns no records. **No lane is promoted.** The more useful finding is that on two of the three devices the cooperative kernel is the *only* compiled Langevin lane — gfx1201 has no promoted Python-emitted EBM family and sm_120 never had one — so only gfx1151 can run the comparison at all.

Remaining: promotion still waits on kernel-time attribution, which no WSL2 ROCm box can produce, and on bare-metal calibration for the NVIDIA rows; `exp`/`log` of multivectors (rotor sampling on the group) stays closed; opaque-callback energies keep the reference path; no Apple package route.

Evidence: [three overhead packets](../../../benchmarks/baselines/ebm_langevin_overhead_20260916/README.md) with source fingerprints, `tests/unit/test_ebm_native_langevin.py` (44 on the Mac, Princess-Luna and Super-Bear), `tests/unit/test_ebm_native_langevin_gpu.py` (46 on each of gfx1151, gfx1201, sm_120), `src/solvers/ebm/test/ir/passes/lower_langevin_bivector.mlir` (+ its refusals), EBM lit 16/16 and Clifford lit 19/19 on the Mac and under Tajasarus's assertions driver.

<!-- entry-fields:end -->

### 2026-09-16 — the math a kernel is allowed to contain, and four closed domain gaps

Owner: [W4-PRODUCT-1](INTEGRATED_COMPILER_PLAN.md#w4-product-1)

PRs: domain-support stream, eighth slice (sync `EBM-GA-GAPCLOSE-2026-09-16`); co-owner [AD-SOLVER-IFT-1](INTEGRATED_COMPILER_PLAN.md#ad-solver-ift-1).

Outcome: **The math audit found a silent wrong answer, and closing it changed what a row-program kernel may contain.** `math.sqrt` had been pinned to libdevice's rounding-explicit `__nv_fsqrt_rn` after one sm_120 row came back 1 ulp off; every *other* `math.*` op on both device routes was subject to the same vendor default and passed through unmeasured. The emitter now carries a **closed admission table**, each op declaring why its result can be trusted — `rounding_explicit`, `bit_exact`, or `measured` — and refuses anything else by name, pointing at the recorder. The declaration has a consumer: each kernel carries `tessera.row_program.math`, the recorder refuses a kernel whose declaration disagrees with the table, and a unit test holds the Python mirror against the C++ table. Measured at 16384 points per input domain on all three devices: `sqrt` and `absf` exact everywhere (the sm_120 zero **is** the pin working), `cos` 1 ulp, `exp` 2 ulp on both RDNA parts and 3 on sm_120, `log` 3 ulp. These bounds do not weaken the earlier EBM packets, which recorded zero error for *their* inputs; this is the wider statement those rows never made.

The first run of that sweep produced the finding. **`math.tanh` returned zero for every input on gfx1151**, because it lowers to `__ocml_tanh_f32` and the packager's binary serialization shipped a kernel whose entire body was one `s_endpgm`: the launch succeeded, wrote nothing, and the caller read back a plausible-looking answer. The identical module serialized with `format=isa` contains the correct 40-instruction implementation, so the body is lost in the *binary* path, not the lowering. Nothing in the packager noticed and nothing about it is specific to tanh, so `build_native_gpu_storage` now disassembles the image it is about to ship and refuses one whose kernel contains no store — sound rather than heuristic, since every kernel in this ABI writes at least one output buffer. The guard covers the AMDGPU image only; the NVIDIA cubin needs `nvdisasm`, not a matched-LLVM tool, so an equivalent silent loss on the NVVM route would still ship. `math.log1p` left the table beside tanh: the device computes `log(1 + x)`, so `log1p(-1e-6)` returned -1.0132795e-06 against the host's -1.0000005e-06, and accuracy near zero is the only reason log1p exists.

**Four domain gaps closed.** `exp` and `log` were declared with no lowering, so rotor sampling happened on the Lie algebra with the reference doing the group step; both now lower to their closed forms on Cl(3, 0), and `rotor_from_axis` lands with them (its angle is an attribute, so both transcendentals fold and the kernel is one reciprocal and a scale). The power series is deliberately not emitted — 24 geometric products is ~1500 unrolled mul-adds — so `exp` admits an operand it can *prove* is a pure bivector; the reference picks its branch from the value, and guessing would make the two disagree on the same input. **Ragged batches**: leading axes may be dynamic, the bound comes from `tensor.dim`, and one cached module serves every batch length, with the shape agreement every bilinear op requires emitted as a `cf.assert` per dynamic axis rather than assumed. The geometric product's own static-only copy of the loop nest is now the shared one. **An annealing schedule**: `langevin_step` takes its temperature either as the constant attribute or as a runtime operand, exactly one, so a cooling chain is one loop and one cooperative kernel with the temperature carried in registers — as a carried multiply, not `powf` of the index, which is also the only form the math admission table admits. A ratio of 1.0 reproduces the constant chain bit for bit on both lanes. **An opaque energy adjoint** is now refused by name: documented as refused from the first slice and never checked, it previously surfaced as a JIT pipeline error naming nothing.

Two fail-open exit statuses closed: the Clifford expansion printed its refusals and exited 0 (the same defect the EBM lowering was fixed for), and the row-program emitter's no-diagnostic fallback fired on top of a real refusal.

**Four reds that were on main and that CI cannot see, found by sweeping and fixed rather than labelled.** (a) A *product* request — native VJP storage child, typed export, checkpoint — on an already-paired module exited 0 having emitted no product: the per-function pairing guard added two slices earlier left an existing `@f__bwd` alone, which is right for a plain re-pairing run and wrong for a request that must derive something from a pairing it is about to perform. It refuses now. (b) `test_f16_cond_native_selects_branch` had been recorded as a macOS 27 MPSGraph regression, "f16 cond off 10%". It is ours: `mpsg_build_branch` built every authored-branch op at the *boundary* dtype, so an f16 branch computed internals in f16, against the ABI's own f32-internal policy — which the authored-package path in the same file states in its comment. In an f16 cond branch `matmul` is accurate to 2e-3 and `sigmoid` to 3e-4, while `silu(x @ w)`, just those composed, was **34% off**; an explicit `mul(m, sigmoid(m))` reproduced it exactly and the bf16 boundary, which already upcast, was fine. Casting once at the branch edges fixes it. Attributing a numeric red to the vendor before bisecting our own lanes is what kept this open. (c) Changing the runtime invalidated the strict route ledger and the e2e packet, both **re-measured, not re-stamped** — all 24 ledger decisions and 6 ineligible rows reproduce with identical key sets, routes and incumbents. (d) Five tests hard-coding `nvidia`/`sm_120` failed inside NVVM serialization on a host with no CUDA toolchain instead of skipping; `test_native_ssd` guarded its `rocm` lane and left `nvidia` bare. `require_native_storage_lane` is the guard, and `native_storage_target`'s own docstring had already recorded this class on 2026-09-15.

Remaining: **26 further pre-existing failures on Princess-Luna**, bisected against main built from its own sources on that box and spun out as their own work — 12 are `complex64` on `tessera.istft` failing the ROCm capability check, and the rest span the profiler callback trace, the runtime artifact ABI, the ROCm backend lit suite and four native-storage files; the Clifford **field ops** (`ext_deriv`, `codiff`, `vec_deriv`, `integral`) still have no lowering; promotion still waits on kernel-time attribution no WSL2 ROCm box can produce; no Apple package route for the loop or the products; the row-program emitter's 1024-feature ceiling still needs a second reduction level, which would change the declared reduction order and so requires re-deriving the reference; G1 (serial native-tape residency) stays deliberately unbuilt, scoped only for programs outside the row-program envelope.

Evidence: [three math-precision packets](../../../benchmarks/baselines/row_program_math_precision_20260916/README.md) with source fingerprints; `tests/unit/test_row_program_math_admission.py` (16), `tests/unit/test_clifford_exp_log_native.py` (27), `tests/unit/test_ebm_native_langevin.py` (56), `tests/unit/test_ebm_native_langevin_gpu.py` (46 + 30 device); `src/solvers/clifford/test/ir/passes/expand_exp_log_rotor.mlir`, `expand_ragged_batch.mlir` and their refusals; `src/solvers/ebm/test/ir/passes/lower_langevin_annealed.mlir`, `src/solvers/ebm/test/ir/langevin_temperature_rejects.mlir`, `tests/tessera-ir/phase2_autodiff/langevin_opaque_adjoint_rejects.mlir`; lit 491/491 on the Mac and, in **both** trees, on Tajasarus including its assertions-enabled driver; EBM 18/18 and Clifford 22/22 on the Mac, Princess-Luna and both Tajasarus trees; 302 unit on Princess-Luna, 149 on Super-Bear, 65 on Tajasarus (68 skipped: that box has no `libtessera_jit`, so every CPU-lane case skips there). Collecting the Tajasarus figures exposed that `check-ebm` / `check-clifford` print nothing and exit 0 on that host — the documented venv-only-`lit` trap — so those numbers come from running lit directly; recorded in the ROCm queue.

<!-- entry-fields:end -->

### 2026-09-17 — the ROCm host's red zone, and four causes that were not what the errors said

Owner: [COMPILER-DEVEX-1](INTEGRATED_COMPILER_PLAN.md#compiler-devex-1)

PRs: ROCm-host red zone (sync `ROCM-HOST-RED-ZONE-2026-09-17`); co-owner [W4-PRODUCT-1](INTEGRATED_COMPILER_PLAN.md#w4-product-1).

Outcome: **A full unit sweep on Princess-Luna reported 31 failures, all present on main and none visible to CI** — its unit lane has neither a CUDA nor a ROCm toolchain, so every one of these tests skips there. Five were test-gating bugs fixed earlier; **23 of the remaining 26 are fixed and verified on the owning box**, and not one cause was what its error message suggested.

**Twelve looked like a missing capability and were a routing bug of ours.** Every spectral test failed with "dtype 'complex64' is not supported for tessera.istft on rocm". The capability registry was right: `rocm_gfx1151` declares `tessera.stft`/`tessera.istft` and no other ROCm entry does, because `GenerateROCMSpectralBackwardKernel.cpp` is gfx1151-only and says so in its own diagnostics. But the rest of the stack already treats a `rocm` request as a gfx1151 compilation — `native_vjp_plugins` registers its ROCm consumers under the family name and resolves them to gfx1151 internally, `jit.py` emits Graph IR for `rocm_gfx1151` when the request said `rocm`. Only the legality gate asked about the generic name, and so refused at trace time a capability the lane would then have provided. It now routes the question to the chip the request compiles for — and **routes without widening**: `supports_op("rocm", "tessera.istft")` is still False, so the generic name still inherits no proof anywhere a dashboard or audit row reads it. The `stft` half of each pair had been passing only because the check reads the *operand* dtype and stft takes a real input, which is what made this read as a complex-storage problem.

**Six were a link requirement nobody published.** `libtessera_runtime.a` carries the HIP backend objects whenever the build enabled them; a harness compiled with its own command line gets none of the CMake target's usage requirements, so it failed with a page of `undefined reference to hipMalloc` — a missing library reported as a broken ABI. The build now writes `tessera_runtime.consumer-link.txt` beside the archive, `tests/_support/runtime_link.py` reads it, and a drift test holds writer and readers together; that test caught a link line I had missed while writing it.

**One failed `check-tessera-rocm` for the whole repository** — this backend's only automated fixture coverage, which no PR check runs. A Python-driven data file inside the lit suite was Unresolved ("Test has no RUN line"). It carried `UNSUPPORTED: true` meant to prevent exactly that; on LLVM 23's lit a test with no RUN line is Unresolved *before* the marker is consulted. Moved to `tests/fixtures/` beside the x86 twin, and `test_lit_fixture_keywords.py` now fails any lit fixture with no RUN line. 68/68.

**Four were the reverse-mode gate refusing what it should not.** `arith.select` had no transpose, so reverse mode over any lowered branch — a recovered switch edge, a Huber kink, a masked manifold step — stopped dead; it is linear in its two value operands, so the taken branch receives the cotangent and the other receives zero. `arith.index_cast` was refused as non-differentiable when index arithmetic is not a differentiable variable at all; that is now a property of the *types* rather than a list of op names that would need extending every time a shape computation used one more integer op. A first cut of that predicate omitted `complex<f32>` and silently dropped six FFT transposes — recorded, because dropping a gradient is the failure the rule exists to prevent.

Remaining: **`AUTODIFF-SHAPE-WHILE-FORWARD-2026-09-17`** — three shape-varying `scf.while` tests that compiled once the gate was fixed and then crashed inside JIT-compiled code. **The mechanism recorded here on 2026-09-17 ("a shrinking dynamic iter_arg does not survive bufferization") was wrong, and is corrected in the ROCm queue under the same key:** the exported forward product still carried the `tessera.autodiff = "reverse"` *request* marker, so the JIT's unconditional paired pass differentiated it again and re-materialized its tapes, and the invoke wrote past the ABI the caller had read. Ten hand-written modules established it; the envelope-carry change written against the wrong mechanism was reverted. Separately, the x86 JIT AD lane those tests exercise has **no host in any automated check**, so defects in it surface only in a manual sweep.

**The benchmark surface, reviewed the same way (2026-09-17).** `docs/benchmarks/` held eight "TesseraBench" documents describing a `tesserabench` package, CLI, CI/regression/reporting/production modules and an enterprise roadmap, and its README called them "the official benchmarking and performance validation framework". Checked against the tree: 23 of the 26 classes and 13 of the 14 module paths they name do not exist, nor the package or the command. They are now `archive/benchmarks/tesserabench_docs/` with that check recorded, beside the Blackwell sketch archived for the same reason, and `docs/benchmarks/README.md` is an index of what runs. Running `benchmarks/README.md`'s own quick checks found `run_all.py --json-only` printing the human summary and no JSON (it now prints the report), and the three SuperBench kernels reporting `execution_kind: unknown` for lanes that ran — the axis was never set. `artifact_schema.infer_execution_kind` is now the one rule for the proxy lanes (CPU JIT and the roofline/mock models are *reference*; Apple GPU JIT is native; nothing that did not execute is either), used by `run_all.py` and the three kernels; it is a closed table that refuses the ~35 family-named routes the device recorders emit, because whether `rocm_norm_compiled` ran natively is known only to the launch result, not to the path's name (Decision #30). Every `run_all.py` row now carries `route` and `latency_source` (Decision #12, amended). Sealed evidence: 31 top-level baseline files (four more are read only by a derived name) and four packet directories are cited by nothing outside `benchmarks/baselines/`, and ten packet directories have no manifest or README; `tests/unit/test_benchmark_baselines_are_cited.py` freezes those as a ratchet (one `git grep -F`, ~10 s) so the next uncited seal fails on the CPU lane. The recorders got the same treatment: 50 of the 319 Python files under `benchmarks/` are named by no other tracked file — 21 are `benchmarks/nvidia/record_*` whose products the NVIDIA queue cites by date but never by the recorder that made them — and `test_benchmark_recorders_are_named.py` freezes those (a tokenised single pass, about two seconds). The README's own quick-check commands all run to completion on a CPU-only host. Not decided here, because it is the owner's: whether those 31 files, 14 directories and 50 recorders are pruned, cited or indexed. The surface dashboard's `benchmarks/baselines` entry pointed at a JSON file as its "runnable" entry point; it now names `perf_gate.py`, the thing that runs.

Evidence: `tests/unit/test_runtime_link_requirements.py`, `tests/unit/test_lit_fixture_keywords.py`; `check-tessera-rocm` 68/68 and `tests/unit/test_autodiff_spectral_target_binding.py` 40/40 on Princess-Luna, whose full sweep is **19455 passed / 0 failed**; the Mac at **18380 passed / 0 failed**; the bisect against `4898812c` built from its own sources in worktrees on Princess-Luna and Super-Bear.

<!-- entry-fields:end -->

### 2026-09-17 — the gfx1201 tail: two boxes swept to their real defects, and the benchmark surface reviewed

Owner: [COMPILER-DEVEX-1](INTEGRATED_COMPILER_PLAN.md#compiler-devex-1)

PRs: ROCm-host red zone follow-ups (sync `ROCM-HOST-RED-ZONE-FOLLOWUPS-2026-09-17`); co-owner [W4-PRODUCT-1](INTEGRATED_COMPILER_PLAN.md#w4-product-1).

Outcome: **The three items the previous entry left open are closed or reduced to named, verified defects, and the two red zones it found are swept.** The shape-varying `scf.while` crash was not the mechanism recorded; the exported forward product still carried the `tessera.autodiff = "reverse"` *request* marker, so the JIT's unconditional paired pass differentiated it again and re-materialized its tapes — ten hand-written modules through the JIT in subprocesses established it, the envelope-carry change written against the wrong story was reverted, and products now carry their role. Tajasarus's ~1500 gfx1151-family failures were one gating class in a dozen wordings; the fail-closed refusals now become skips at the report hook, only when they name the host's own arch. Super-Bear's 48 were four gating bugs and one real device failure.

**Four assertions-only compiler defects in one day, all invisible on every NDEBUG driver.** 81 sites in 72 ROCm generators (and the NVIDIA Philox generator) stamped `gpu.kernel` by raw attribute name on top of LLVM 23's inherent property, so single-invocation pipelines aborted with `DictionaryAttr element names must be unique` — the NDEBUG build shows it as `attributes {gpu.kernel, gpu.kernel, rocdl.kernel}`, and a text round-trip hides it, which is why upstream `mlir-opt` on the printed IR never saw it. The paired pass's `emit-storage-child` and this backend's `KernelABIPass` each loaded a dialect from inside `runOnOperation` (tile, llvm). All fixed; `rocm_generated_kernel_stamps_gpu_kernel_once.mlir` catches the first class on NDEBUG hosts too.

**Owed, verified on gfx1201 and not diagnosed past the reason** (the ROCm queue lists each): `fused_epilogue_launch_execute` produces **wrong numbers** rather than refusing — a gfx11 WMMA kernel run through a path that never consults the arch guard — and 8 more rows fail with "matmul requires exactly two operands"; `int8 @ int8` routes to the WMMA f16/bf16 executor; the KU reference kernel fails to launch (rc=2); the state-machine and geo-Langevin lanes; one inline HIP source that no longer compiles under HIP 7.15; an x86 shared image missing from that box's tree. On Super-Bear the sm_120 Lion stop-sign lane returns `rc=3` from the PTX launcher — a CUDA memory-API failure — **on `main` as well**, bisected with both trees built fresh in a worktree there. That bisect found the `power_retention` example could not build in a clean CUDA-configured tree at all (four layers: header path, a hand-written dialect beside the generated one, a missing `GET_OP_CLASSES`, a removed `PassRegistration` constructor — and beneath them a CUDA kernel that does not compile, now `EXCLUDE_FROM_ALL`).

**The benchmark surface** (`docs/benchmarks`, `benchmarks/`) reviewed the same way is recorded in the previous entry's addendum: the TesseraBench sketch archived with its check, `run_all.py --json-only` producing JSON with `route` and `latency_source`, one execution-kind rule, and ratchets for uncited baselines (31 files, 14 directories) and unnamed recorders (50).

| Host | Before | After |
|---|---|---|
| Princess-Luna (gfx1151) | 26 failed (3 crashing) | **19490 passed, 0 failed**; `check-tessera-rocm` 68/68; lit 493/493 |
| Tajasarus (gfx1201, assertions LLVM) | 1578 failed | **50 failed, 16247 passed** — all in the owed list; `check-tessera-rocm` 68/68 on the assertions driver; lit 493/493 in both trees |
| Super-Bear (sm_120) | 48 failed | **1 failed, 16091 passed** — the Lion lane, pre-existing on `main` (bisected, both trees built fresh); one sweep also reported 8 in `test_automatic_ad_public_results.py` that reproduced neither alone, in alphabetical order, nor in a second full sweep; lit 493/493 |
| Mac (M1 Max) | 1 failed | **18400 passed, 0 failed**; lit 493/493; mypy, ruff, doc and plan gates clean |

Remaining: the owed gfx1201 list above; the Lion `rc=3` on sm_120 (first step: make `tessera_nvidia_ptx_invoke_v2` say which CUDA call failed); the `power_retention` scaffold's kernel; `test_dynamic_shape_emit` passes alone and failed once in a full Tajasarus sweep (order-dependent, not chased); and whether the frozen orphan baselines and recorders are pruned, cited or indexed — the owner's call.

Evidence: `tests/unit/test_rocm_compiled_family_gate.py`, `tests/tessera-ir/phase2_autodiff/autodiff_paired_export_product_is_not_a_request.mlir`, `tests/tessera-ir/phase3/rocm_generated_kernel_stamps_gpu_kernel_once.mlir`, `tests/unit/test_benchmark_baselines_are_cited.py`, `tests/unit/test_benchmark_recorders_are_named.py`; the bisects built `main` from its own sources in fresh worktrees on Princess-Luna and Super-Bear.

<!-- entry-fields:end -->

### 2026-09-17 — the engineering loops: every owed item from the gfx1201 tail worked to its disposition

Owner: [COMPILER-DEVEX-1](INTEGRATED_COMPILER_PLAN.md#compiler-devex-1)

PRs: ROCm-host red zone engineering loops (sync `ROCM-HOST-RED-ZONE-FOLLOWUPS-2026-09-17`); co-owner [W4-PRODUCT-1](INTEGRATED_COMPILER_PLAN.md#w4-product-1).

Outcome: **Every item the previous entry left owed is closed, promoted, reduced to an upstream reproducer, or retired — and none of the seven gfx1201 defects was what its failure said.** The fused-epilogue "wrong numbers" were the bare matmul: the compiled 16x16x16 lane refuses on RDNA4 (correctly) and the launch path fell back to the hand-written oracle, which drops the activation kwarg and knows two operands. A fallback now runs only when it computes the same program, and the oracle refuses what it does not implement; those rows skip on gfx1201 naming the arch. The KU reference rung was missing the RDNA4 fragment specialization its rung-1 sibling had (8/8 pass on gfx1201 now). Two scalar families — the state machine and the affine Langevin core — are promoted on gfx1201 on the evidence their tests produced (9/9), and the samplers' numpy fallback no longer promotes a float32 state to float64. The WMMA compare harness selects its fragment layout per device pass. Tajasarus's `build/` now builds the x86 backend, so its Zen 5 lanes have their shared image.

**The sm_120 Lion `rc=3` was the toolchain pin, and the launcher now says so.** Every CUDA driver call in the PTX bridge records its name and `CUresult` (`tessera_nvidia_ptx_last_error`); the first instrumented run read `cuModuleLoadDataEx: CUDA_ERROR_UNSUPPORTED_PTX_VERSION ... Unsupported .version 9.4; current version is '9.3'`. The 2026-09-15 bump pinned nvcc 13.4's PTX ISA as *the* ISA while driver 610.88 implements the CUDA 13.3 driver API (`cuDriverGetVersion` 13030) and JIT-compiles at most 9.3. The driver is now pinned and derived separately (`gpu_target.driver_jit_ptx_isa`), the emitter stamps it, and the runtime's one registration point re-stamps nvcc's PTX to it. The lane passes; Super-Bear's full sweep is **0 failed**.

**The first assertions-ON NVIDIA driver in the fleet (`build-assertions-nvidia` on Tajasarus) found three more defects no NDEBUG driver could:** the Philox generator creating `math` ops without declaring the dialect, the Tile→NVIDIA lowering loading `tile` from inside `runOnOperation`, and — a new class — the Philox normal kernel emitting `sin`/`cos` in whichever order the *host C++ compiler* evaluated two nested `create` calls (unspecified in C++; the fixture's order held under one compiler and not the other). All three fixed; the driver now registers the NVVM lowering and the ConvertToLLVM extensions so the single-invocation stamp fixture can run there: **62/62**, and the Philox kernel carries `gpu.kernel` exactly once through `convert-gpu-to-nvvm`.

**The upstream SCEV assertion is reduced to 31 lines.** An assertions-ON `opt`/`llvm-reduce` built from LLVM 23.1.1 sources on Tajasarus reproduces `SCEVDivision::divide` inside `LoopInterchangePass` on the rank-2 dynamic backward's `product` kernel and shrinks it to a two-loop nest storing through a generic pointer cast from address space 5 (`tests/fixtures/llvm23_loop_interchange_scev_division_gfx1151.ll`); `-enable-loopinterchange=0` passes, an NDEBUG `opt` compiles it in silence, and the real kernel executes correctly on gfx1151 (`runtime_shape_frames_20260908/rocm_gfx1151_revalidation_20260917.json`). The rank-2 case skips on an assertions host with that reason; it is LLVM's, not ours.

**`power_retention` is retired** to `archive/examples/advanced/` (op already canonical, kernel never compiled, torch stub), and **the two benchmark orphan ratchets hold at an empty floor**: 6 baselines, 4 packet directories and 11 recorders removed, 25 baselines shown to be read by a derived name, 10 packets given a README naming their recorder and citing record, 41 recorders named.

| Host | Result |
|---|---|
| Princess-Luna (gfx1151) | **19491 passed, 0 failed**; `check-tessera-rocm` 68/68; lit 493/493; shape-frames recorder 11/11 |
| Tajasarus (gfx1201, assertions LLVM) | **17538 passed, 0 failed** (was 50); lit 493/493 in both trees; `check-tessera-rocm` 68/68 on the assertions driver; `check-tessera-nvidia` **62/62** on the new assertions NVIDIA driver; the last failure was the thread-sticky `hipGetLastError` read (ROCm queue item 8), found by ordered bisect |
| Super-Bear (sm_120) | **16093 passed, 0 failed** (was 1: Lion); NVIDIA lit both trees after the driver change, see the NVIDIA queue |
| Mac (M1 Max) | **18402 passed, 0 failed** (four sweep artifacts re-run clean: three `inspect.getsource` reads of a file being edited, one Apple ledger hashed under the lit PATH's clang); ruff, mypy, doc and plan gates clean |

Remaining: gfx1201 parity is now a scoped program (`GFX1201-PARITY-2026-09-17` in the ROCm queue: 7 of 63 families promoted; the matmul family's fused epilogue and integer storage go through the typed Tile route, which already has the RDNA4 fragment family, not a second generator); the LLVM issue for the SCEV reproducer is unfiled (owner's call); the `gfx1151`-pinned attention-backward test skips on gfx1201 by design.

Evidence: `tests/unit/test_rocm_compiled_family_gate.py`, `tests/unit/test_target_toolchain_pins.py`, `src/compiler/codegen/tessera_gpu_backend_NVIDIA/test/nvidia/philox_stamps_gpu_kernel_once.mlir`, `tests/fixtures/llvm23_loop_interchange_scev_division_gfx1151.ll`, `benchmarks/baselines/runtime_shape_frames_20260908/rocm_gfx1151_revalidation_20260917.json`; the two backend queues under `ROCM-HOST-RED-ZONE-FOLLOWUPS-2026-09-17`.

<!-- entry-fields:end -->

### 2026-09-17 — GFX1201-PARITY slice 1: the fused matmul epilogue on the typed Tile→ROCm route

Owner: [COMPILER-DEVEX-1](INTEGRATED_COMPILER_PLAN.md#compiler-devex-1)

PRs: GFX1201-PARITY slice 1 (sync `GFX1201-PARITY-2026-09-17`, ROCm queue); co-owner [W4-PRODUCT-1](INTEGRATED_COMPILER_PLAN.md#w4-product-1).

Outcome: **The matmul family's fused bias/activation epilogue runs on gfx1201 through the typed Tile route, and gfx1151 gets the same implementation — one epilogue, applied after the fragment family has resolved each element's row and column.** `tile.store` carries a `tile.epilogue` attribute and a trailing bias operand; `TileToROCM::materializeFragmentStore` applies `bias[col]` and the activation inside both the unmasked and the guarded stores, so `rdna3_wmma` (gfx1151, replicated rows) and `rdna4_wmma` (RDNA4, half-wave rows) share the arithmetic and differ only in the layout the family already owned. The typed generator passes the epilogue on the store instead of refusing it; Schedule→Tile's ROCm branch carries the bias pointer; `scheduled_matmul` and `package_scheduled_matmul` admit the fused contract on both ROCm targets; and the runtime's `rocm_compiled` lane sends a non-gfx11 f16 GEMM through Graph IR → `--tessera-graph-to-schedule` → Tile → native package → `launch`, so the fused-epilogue launch/execute lane on gfx1201 runs an RDNA4 kernel rather than the gfx11 one (the red-zone entry's owed item 1: 8 lowering errors and 6 wrong answers, now 14/14).

**What masked it.** Graph→Schedule admitted bias/activation only on sm_120. Every one of the 39 slice rows on both boxes failed there, one level above the new consumer, before the device was reached; the first read of the failing lit fixture ("the generator emitted the untyped body") was also wrong — the typed body was emitted on both archs and the CHECK named `gfx11_wmma` where the lowering stamps `rdna3_wmma`. Both corrected in the second commit. Residual add stays NVIDIA-owned.

| Host | Slice files | Lit |
|---|---|---|
| Tajasarus (gfx1201, assertions LLVM) | **151 passed, 0 failed, 50 skipped** (was 29 failed): `test_rocm_gfx1201_scheduled` fused epilogue 15/15, `test_rocm_fused_epilogue_launch_execute` 14/14 | `check-tessera-rocm` 69/69 in `build` and `build-assertions`; `tests/tessera-ir` 493/493 both trees |
| Princess-Luna (gfx1151) | **140 passed, 0 failed, 61 skipped** (was 10 failed): `test_gfx1151_scheduled_matmul_executes_fused_epilogue` 10/10 | `check-tessera-rocm` 69/69; `tests/tessera-ir` 493/493 |

Full sweeps on the branch at `aef83fc1`:

| Host | Full `-m "not slow"` sweep at `aef83fc1` |
|---|---|
| Princess-Luna (gfx1151) | **19500 passed, 1 failed, 2584 skipped**; `check-tessera-rocm` 69/69; `tests/tessera-ir` 493/493 |
| Tajasarus (gfx1201, assertions LLVM) | **17566 passed, 1 failed, 4518 skipped**; `check-tessera-rocm` 69/69 in both trees; `tests/tessera-ir` 493/493 both trees |
| Super-Bear (sm_120) | **16092 passed, 1 failed, 5992 skipped**; both trees built; NVIDIA lit both trees and `tests/tessera-ir` clean |
| Mac (M1 Max) | **18406 passed, 1 failed, 3678 skipped**; `tests/tessera-ir` 493/493; mypy clean on the touched Python; doc, plan and generated-doc gates clean |

The one failure on every host is the same test, `test_diagnostic_code_registry.py::test_every_cpp_code_is_registered`: the slice emits two new codes (`TILE_STORE_EPILOGUE_BIAS`, `ROCM_FRAGMENT_STORE_EPILOGUE`) that were not in the registry. Registered in `b65c8916`; the registry file passes on all four hosts at that commit (40/40).

Remaining: slice 1b — int8/int4 storage on the typed route (the typed generator still refuses int4 packing and a non-accumulator output dtype; the int4 rows of `test_rocm_compiled_launch_execute.py` skip on gfx1201 as unpromoted); the `rocm_wmma_gemm` arbiter candidate still builds the legacy gfx11 directive kernel and declines on gfx12 — that generator's gfx11 gate is now the documented boundary; slices 2–5 of the program as listed in the ROCm queue.

Evidence: `src/compiler/codegen/Tessera_ROCM_Backend/test/rocm/typed_matmul_fused_epilogue_store.mlir`, `tests/unit/test_rocm_gfx1201_scheduled.py`, `tests/unit/test_scheduled_matmul_consumers.py`, `tests/unit/test_rocm_fused_epilogue_launch_execute.py`, `docs/audit/backend/rocm/todo.md` §`GFX1201-PARITY-2026-09-17` slice 1.

<!-- entry-fields:end -->

### 2026-09-17 — GFX1201-PARITY slice 2: twenty-four scalar and row-program families promoted on measurement

Owner: [COMPILER-DEVEX-1](INTEGRATED_COMPILER_PLAN.md#compiler-devex-1)

PRs: GFX1201-PARITY slice 2 (sync `GFX1201-PARITY-2026-09-17`, ROCm queue); co-owner [W4-PRODUCT-1](INTEGRATED_COMPILER_PLAN.md#w4-product-1).

Outcome: **gfx1201 goes from 7 to 31 promoted families, and the promotion is a measurement, not a list.** Both promotion tables (Python `promoted_families`, the C++ `tessera-rocm-executable` gate, now a shared `StringRef` array) gained the 24 families with no WMMA fragment in them. A per-test `pytest -v` sweep on Tajasarus at `main` was diffed against the same sweep with the candidates promoted: **972 tests skip → pass, zero kernel failures**, from `test_rocm_unary_compiled` (273) and `test_rocm_norm_compiled` (109) down through 50 files. Every family that was promoted has its own tests passing on the RX 9070 XT; the queue entry lists the blocks.

**The eight that failed were a record, not a result.** The optimizer VJP plugin lanes stamp `rocm_gfx1151` as their evidence target and name their consumer for that chip, so the certificate validator reported `runtime_unattested` on gfx1201 after the numerics had already matched. That is the validator doing its job (Decision #26: a claim about gfx1151 does not become gfx1201 evidence by running there). Those eight tests are now pinned explicitly to gfx1151 with a helper whose skip reason names both archs and the owning item; slice 2b makes `rocm_gfx1201` a first-class target name on the stateful/optimizer lanes.

| Host | Full `-m "not slow"` sweep at `8251b9e5` |
|---|---|
| Tajasarus (gfx1201, assertions LLVM) | **18538 passed, 3539 skipped** (was 17566 / 4519 at `main`); 8 failed = the new pin helper raising `NameError` instead of skipping (fixed in `58fb7dcd`, re-run: 35 passed, 8 skipped with the named reason); `check-tessera-rocm` 69/69 in both trees; `tests/tessera-ir` 493/493 both trees |
| Princess-Luna (gfx1151) | **19501 passed, 0 failed, 2584 skipped**; `check-tessera-rocm` 69/69; `tests/tessera-ir` 493/493 |
| Super-Bear (sm_120) | **16093 passed, 0 failed, 5992 skipped**; both trees built |
| Mac (M1 Max) | **18406 passed, 0 failed, 3678 skipped**; `tests/tessera-ir` 493/493; doc, plan and generated-doc gates clean |

Remaining: slice 2b (the `rocm_gfx1151` target-name key in `stateful_training.py`, `native_vjp_plugins.py`, `jit.py`, `gpu_target_map.py`); slice 1b (int8/int4 storage on the typed route); slices 3–5 as listed in the ROCm queue.

Evidence: `tests/unit/test_rocm_compiled_family_gate.py` (pins the 31), `tests/_support/rocm_build.require_rocm_host_arch`, the per-file diff in the ROCm queue's slice-2 entry.

<!-- entry-fields:end -->

### 2026-09-17 — the engineering loops: gfx1201 to 62 of 63 families, the first RDNA4-only consumer, and the sm_120 dispatch that never read its own corpus

Owner: [COMPILER-DEVEX-1](INTEGRATED_COMPILER_PLAN.md#compiler-devex-1)

PRs: engineering loops on both GPU backends (sync `GFX1201-PARITY-2026-09-17` slices 3-5 and the sm_120 items in the NVIDIA queue); co-owner [W4-PRODUCT-1](INTEGRATED_COMPILER_PLAN.md#w4-product-1).

Outcome: **gfx1201 promotes from 31 to 62 of 63 families, OCP FP8 matmul is the first family gfx1151 cannot have, and the sm_120 arbiter reads its device-timed corpus at dispatch.** The attention tail (slice 3) was scalar kernels behind chip-named contracts plus one real port: the linear-attention generator now switches to RDNA4's 8-element half-wave fragments like the flash-attention generator, and the canonical attention-backward adapter admits the chip the backward generator already handled. The spectral, solver, EBM and f32-matmul families (slice 4) had arch-neutral generators under a layer of `gfx1151` literals — five Tile→ROCm adapters, the Schedule verifiers, the Schedule→Tile selectors, the `rocm` alias in three target maps, a prebuilt spectral image and a solver executor's own chip check — every one now names the chip the artifact carries. Slice 5 gives the audited FP8 WMMA forms a kernel-shaped consumer through the typed route, end to end from `tessera.matmul` over `f8E4M3FN` operands to a launched package. The `rocm_wmma_gemm` arbiter candidate reaches gfx12 through the same route. On NVIDIA: the production dispatch consults the device-timed verdict before the wall-clock one, the native-storage packager gains the cubin-side store check the ROCm image already had, and the exact sm_120 HVP execution the queue owed is recorded as a packet beside a gfx1201 twin.

**Corrections to the plan's premises, recorded rather than worked.** Four sm_120 items were already closed on `main` (the emitted GEMM's device timer, the Clifford lane, the EBM sphere/nonlinear lanes, the rounding-explicit sqrt); the queue entry names each. Three stay owed with their first step named: the NVFP4 emitted kernel's launcher entry and fragment-order packer, the isolated CUDA attention tape, and the EBM sphere front door on CUDA. On ROCm, `paged_kv` is the one family not promoted (its device tests sit behind the gfx11 flash-attention directive lane); slice 2b closed along the way — every ROCm VJP lane stamps the chip it launches on, so the optimizer, spectral, sequence-mixer and SSM-backward certificates attest gfx1201 — and public Graph admission for the 2:4 sparse stack is scoped in the ROCm queue as the five layers it needs (Graph op, Schedule contract, packager, launch ABI, VJP) rather than begun.

| Host | Full `-m "not slow"` sweep | Lit |
|---|---|---|
| Tajasarus (gfx1201, assertions LLVM) | **18810 passed, 0 failed, 3291 skipped** at `ecb086c5` (17566 / 4519 before slice 2; 18538 / 3539 after it) | `check-tessera-rocm` 72/72 in `build` and `build-assertions`; `tests/tessera-ir` 493/493 both trees |
| Princess-Luna (gfx1151) | **19504 passed, 0 failed, 2597 skipped** at `e461e368`; the eleven files touched after it 109 passed at `ecb086c5` | `check-tessera-rocm` 72/72; `tests/tessera-ir` 493/493 |
| Super-Bear (sm_120) | **16096 passed, 0 failed, 6005 skipped** at `e461e368`; the host-free files touched after it 86 passed at `ecb086c5` | NVIDIA lit clean in `build-nvidia-cuda`; `tests/tessera-ir` clean; both trees built |
| Mac (M1 Max) | **18410 passed, 0 failed, 3691 skipped** at `e461e368`; the touched files 67 passed at `ecb086c5` | `tests/tessera-ir` 493/493; doc, plan, generated-doc and registry gates clean |

The commits after `e461e368` are Python and test changes only (the attestation's expected chip, the gfx1201 capability rows, three tests deriving the chip), so the three earlier sweeps stand for the compiled trees; Tajasarus, the host they change, was swept again in full.

Remaining: `paged_kv` on gfx1201; the typed route's performance gap against the directive lane (raster order, macro tile, LDS staging, int storage); public sparse admission; the three owed sm_120 items above. `tests/unit/test_solver_ift_evidence.py::test_new_dense_krylov_run_does_not_inherit_old_performance` fails on a clean `main` worktree on the Mac independent of this branch (the benchmark stub returns text where the recorder decodes bytes) and is not touched here.

Evidence: `docs/audit/backend/rocm/todo.md` §`GFX1201-PARITY-2026-09-17` slices 3-5, `docs/audit/backend/nvidia/todo.md` §"sm_120 engineering loops", `src/compiler/codegen/Tessera_ROCM_Backend/test/rocm/typed_matmul_fp8_gfx1201.mlir`, `gfx1201_tile_depth_attention_kernel.mlir`, `gfx1201_tile_attention_backward_wmma_adapter.mlir`, `tests/unit/test_rocm_gfx1201_scheduled.py`, `tests/unit/test_rocm_compiled_family_gate.py` (pins the 62), `tests/unit/test_arbiter_autotune.py`, `benchmarks/baselines/native_hvp_20260917/`.

<!-- entry-fields:end -->

### 2026-09-18 — the owed loops: gfx1201 at every family, integer and bf16 storage on the typed route, the measured typed-route gap, public 2:4 sparse admission, and the three sm_120 items

Owner: [COMPILER-DEVEX-1](INTEGRATED_COMPILER_PLAN.md#compiler-devex-1)

PRs: the owed items of sync `GFX1201-PARITY-2026-09-17` on both GPU backends (branch `claude/gfx1201-sm120-owed-loops`, after the five merged branches were pruned locally, on origin and on the three boxes and every checkout synced to `main`); co-owner [W4-PRODUCT-1](INTEGRATED_COMPILER_PLAN.md#w4-product-1).

Outcome: **gfx1201 has every family, the typed ROCm matmul route carries int8, int4 and bf16 storage on both chips, the typed route's performance gap is measured on both chips and acted on, the 2:4 sparse stack has a public admission through the production boundaries, and the three owed sm_120 items are closed on Super-Bear.** `paged_kv` was never a generator problem: the runtime built the directive flash-attention kernel without an `arch` stamp, so the family's gate probed a gfx11 kernel on gfx12; naming the chip opened the whole directive flash-attention lane on gfx1201 along with the family. Slice 1b was four admission checks between a fragment materializer that already packed integers for both chips and a runtime that already knew the dtypes. The gap measurement found that the `staging` knob the queue named does nothing on the typed route (byte-identical kernels, a false `multiwave_lds` label since corrected) and that its own first packet had timed a clock ramp; the corrected packet selects gfx1201's 2x4 panel for fully tiled problems at 1024 and above (2.1x at 1024³, 3.3x at 2048³) and shows gfx1151's typed 2x4 at parity with the directive lane. The raster contract now reaches the generator from the typed route (selection still row-major, pending counters). Sparse admission is the five layers the queue scoped — `jit.package_sparse_2to4`, a replayed Schedule artifact, a packager through an RDNA4-only family, a launch ABI that consumes every validity word and refuses a non-2:4 tile, and AD left on the logical function. On NVIDIA: the emitted NVFP4 warp tile has a launcher entry, a per-lane packer and an exact packet in five scale modes; the isolated attention tape is backend-parametric with a CUDA twin proven through death, recovery and replacement; the EBM sphere step runs as one cooperative sm_120 kernel behind its front door.

| Host | Full `-m "not slow"` sweep | Lit |
|---|---|---|
| Tajasarus (gfx1201, assertions LLVM) | **18955 passed, 4 failed, 3230 skipped** at `a5474704`; the four were host gates this branch tripped everywhere (staging-arena body slice, pipeline registry, gfx1151 dtype tuple, device-marker location), fixed at `a9a158e0` where the touched files 250 passed | `check-tessera-rocm` 74/74 in `build` and `build-assertions` (two new fixtures: `typed_matmul_int_storage.mlir`, `typed_matmul_raster_order.mlir`); `tests/tessera-ir` 431 passed / 62 unsupported / 0 failed in both trees |
| Princess-Luna (gfx1151) | **19545 passed, 4 failed, 2640 skipped** at `a5474704` (the same four; 195 passed at `a9a158e0`) | `check-tessera-rocm` 74/74; `tests/tessera-ir` 489 passed / 4 unsupported / 0 failed |
| Super-Bear (sm_120) | **16148 passed, 8 failed, 6033 skipped** at `a5474704`: the same four gates plus the four sm_120 HVP rows, which read `TESSERA_OPT`/`TESSERA_LLVM_BIN` from the environment and raised a bare KeyError under the device gate alone — they discover the tools now and pass by discovery (13 passed at `f93dfabe`); the gates 80 passed at `a9a158e0` | both trees built; the NVFP4, isolated-attention and sphere device rows pass at the head |
| Mac (M1 Max) | **18453 passed, 0 failed, 3736 skipped** at `a9a158e0` (18449 / 4 at `00cd822b`, the same four gates) | `tests/tessera-ir` 452 passed / 41 unsupported; doc, plan, generated-doc and registry gates clean. `tests/MEMORY_AND_PERFORMANCE.md`'s full-count table is stale on `main` independent of this branch (the tree collects 22974 on `main`, outside the doc's ±15% band; this branch adds 87), which its drift gate reports only after a full local run |

Remaining: an LDS-staged typed body (the real work behind the "LDS staging" the queue named); gfx1151's 4x4 typed panel at 1024³ (10% behind the directive lane there); raster-order selection needs counters (ROCM-RASTER-1B); the packed-int4 input route on the typed path; `auto_2to4` device rows through the public package; the NVFP4 tile's general-shape dispatch before it can be an arbiter candidate. Performance promotion on both GPU boxes still needs a counter-capable native-Linux host.

Evidence: `docs/audit/backend/rocm/todo.md` §"The owed items, worked — 2026-09-18", `docs/audit/backend/nvidia/todo.md` §"The three owed sm_120 items, worked — 2026-09-18", `benchmarks/baselines/typed_route_gap_20260918/`, `benchmarks/baselines/nvfp4_emitted_20260918/`, `tests/unit/test_scheduled_sparse.py`, `tests/unit/test_nvidia_nvfp4_emitted.py`, `tests/unit/test_isolated_cuda_attention.py`, `tests/unit/test_cuda_ebm_geo_langevin_compiled.py`, `tests/unit/test_rocm_compiled_family_gate.py` (pins every family on gfx1201 and the RDNA4-only one).

<!-- entry-fields:end -->

### 2026-09-18 — the typed route's real levers: an LDS body that loses, the K unroll that wins, general-shape NVFP4, and the RDNA4 ceilings

Owner: [COMPILER-DEVEX-1](INTEGRATED_COMPILER_PLAN.md#compiler-devex-1)

PRs: branch `claude/typed-route-lds-nvfp4-dispatch`, sync `GFX1201-PARITY-2026-09-17`; co-owner [W4-PRODUCT-1](INTEGRATED_COMPILER_PLAN.md#w4-product-1).

Outcome: **the four owed ROCm typed-route items and the NVFP4 dispatch item are closed, one of them as a negative result, and the RDNA4 throughput ceilings now say what to do next.** The LDS-staged typed body is real — WM x WN waves per workgroup with both operands staged per K slab and B transposed so fragment packs become contiguous vector loads — and it is **correct on device on both chips at multi-wave workgroups and selected nowhere**. The precise shape of that loss is the interesting part: on gfx1201 it is 0.23x-0.83x of the register body at the same panel everywhere measured, while on gfx1151 it *does* beat the register body at the same panel (up to 1.26x, mostly at the 1x1 panel where that body is weakest) and still never beats the best register configuration at any shape — 12.57 against the selected body's 15.75 at 1024³. It ships as a packaging option carried in the descriptor. Per-shape panel selection landed on both chips and the panel axis is exhausted: gfx1201's 4096³ single-slab row climbs 14.5 → 57.2 → 65.2 TFLOP/s across the three panels and stops there, and an exploratory sweep past them fell off a VGPR cliff. The lever that did move is the **K unroll** — the body is memory-latency bound, so issuing the next slab's loads under the current slab's MMAs is worth 1.4x-1.7x on gfx1201 (4096³: 65.2 → 90.3 TFLOP/s) and puts gfx1151's typed route ahead of the directive lane in the [1024, 2048) band (15.8 vs 11.6). Both chips take 2, and only where they take the larger panel; an earlier cut took 4 below 2048 on gfx1201 and is withdrawn, because it rested on one row whose sign the re-record reversed inside a 2% margin. Each chip's unroll comes from its own sweep; neither is evidence for the other. On NVIDIA the emitted NVFP4 kernel became general-shape with a launcher entry, a structural PTX validator and an arbiter registration through the additive `register_op_kind` seam, so it is an eligible candidate rather than a fixed warp tile — registered alongside the shipped path, which is visible as *untimed* rather than absent, per the recorded sm_120 lesson that a biased corpus hides the fastest kernel.

Two wrong-memory guards and a contract page came out of the same work. A `tile.view` claiming the `lds` space over a default-space buffer was accepted and would have lowered to global loads reading the wrong memory; the attribute and the memref address space must now agree. The gfx11 2x4 contract stamp is no longer applied to bodies that are not that shape, which had made every K-unrolled 2x4 row on gfx1151 unmeasurable — and I had briefly written that silence into a docstring as a negative result before the data showed k2 winning. `docs/backends/rocm/wmma-fragment-layout.md` is the normative page for the four mappings whose failure mode is a few silently wrong tiles: the intra-wave lane contract, the column-distributed gfx12 accumulator (`VGPR[lane][j] = C[(lane/16)*8 + j][lane%16]`, against RDNA3's row `2j + lane/16`), K-major operands, and the undocumented double-K int4 nibble order. The accumulator mapping was independently confirmed on a second stack (Radeon AI PRO R9700, ROCm 7.14 nightly) at 256/256 elements, the row-major misreading matching only the 16 diagonal elements.

**The ceilings reorder the queue.** RDNA4 dense WMMA on gfx1201 is 191 TFLOP/s for fp16/bf16, 383 for fp8, 383 TOP/s for int8, 766 TOP/s for int4, each doubling under 2:4 structured sparsity, and **no FP4 form exists**. So the typed f16 GEMM's best row is 47% of its ceiling with the panel axis already spent, and — measured on Tajasarus — the storages with the *highest* ceilings are on the *smallest* tile: `lower_scheduled_matmul` consults the panel rule only in its f16/bf16 branch, so an fp8 matmul packages as `gfx1201_register_wmma_1x1` where its f16 twin gets `4x4`. That is a selection omission rather than a generator limit (the fp8 Tile IR compiles at a 64x64 panel), and it is the next loop's top ROCm item. Tessera's FP4 refusal on this arch is now pinned by a test, because the common workaround elsewhere — dequantize to fp16 before the MMA — returns the fp16 ceiling under an FP4 label with nothing red anywhere.

A driver-skew defect surfaced underneath the NVIDIA work and was strictly larger than the item that found it: **every NVRTC-compiled kernel on The-Super-Bear was dead.** NVRTC 13.4 emits `.version 9.4`; the loaded driver reports API version 13030 (CUDA 13.3) and its JIT accepts 9.3, so `cuModuleLoadData` returned 222 for both `compute_120` and `compute_120a` while the runtime reported it as "requires an sm_120a-capable CUDA device/toolchain" — an accurate-sounding sentence about the device for a pure toolkit/driver skew. `compileKernel` now retries with a progressively lowered `.version`. Same trap as the 09-15 pin bump, second lane.

| Host | Sweep | Lit |
|---|---|---|
| Tajasarus (gfx1201, assertions LLVM) | **18998 passed, 0 failed, 3253 skipped** at `3319590b`; targeted re-run 180 passed at `10bd09f7` | `check-tessera-rocm` 75/75 in `build` and `build-assertions` (new fixture `typed_matmul_lds_staged.mlir`); `tests/tessera-ir` 431 passed / 62 unsupported / 0 failed |
| Princess-Luna (gfx1151) | **19584 passed, 1 failed, 2666 skipped** at `3319590b` — the one was this branch's own: the selected-panel device row pinned its route name as a literal, so the K unroll landing in the same band read as a failure; it derives the route from `rocm_k_unroll` now and 134 passed at `10bd09f7` | `check-tessera-rocm` 75/75; `tests/tessera-ir` 489 passed / 4 unsupported / 0 failed |
| Super-Bear (sm_120) | **16201 passed, 7 failed, 6043 skipped** at `3319590b`; the seven are **pre-existing on this host** — clean `main` fails the identical seven when a compiler is on `PATH` — and are skips again at `10bd09f7`, where 90 passed and 0 failed | 47 NVFP4 tests pass with zero skips; both trees built |
| Mac (M1 Max) | **18475 passed, 0 failed** at `3319590b` | `tests/tessera-ir` clean; doc, plan, generated-doc and registry gates clean |

The Super-Bear seven are worth naming as a class, because the gate said the opposite of the truth: `require_rocm_hsaco_toolkit` mirrored the runtime detector, which accepts any root holding an `ld.lld` — and a plain LLVM install has one. With `/usr/lib/llvm-23/bin` on `PATH`, a CUDA box with no ROCm at all admitted eight ROCm packaging tests. The ROCDL target links device bitcode too, so the gate now requires `<root>/amdgcn/bitcode/ocml.bc`: present under both ROCm roots on Princess-Luna, absent everywhere on Super-Bear. A host that cannot evaluate a lane must skip it, not fail it.

Remaining: the panel and K unroll on the fp8 and integer branches (with a device row per dtype, which needs the gap recorder's timing harness to accept a non-f16 storage — `rt.launch` is host-bound enough to read 7.5 TFLOP/s for a kernel the recorder times at 65.0); raster-order selection, still blocked on counters neither WSL2 ROCm box can produce; the packed-int4 input route, now the highest-ceiling item on gfx1201; NVFP4 corpus rows on the owning box before any promotion; and the remaining 2x to the f16 ceiling, for which neither lever pulled this loop is the answer.

Evidence: `docs/audit/backend/rocm/todo.md` §"The owed typed-route items, worked — 2026-09-18", `docs/audit/backend/nvidia/todo.md` §"General-shape NVFP4 dispatch, and the driver JIT ceiling that hid under it — 2026-09-18", `docs/backends/rocm/wmma-fragment-layout.md`, `benchmarks/baselines/typed_route_gap_20260918/`, `tests/unit/test_scheduled_matmul_consumers.py`, `tests/unit/test_rocm_gfx1201_scheduled.py`, `tests/unit/test_nvidia_nvfp4_emitted.py`, `tests/unit/test_rdna4_wmma_dtype_contract.py`.
<!-- entry-fields:end -->

### 2026-09-19 — the low-precision panel, and a VGPR reading that is not yet evidence

Owner: [COMPILER-DEVEX-1](INTEGRATED_COMPILER_PLAN.md#compiler-devex-1)

PRs: branch `claude/lowp-panel-selection`, sync `GFX1201-PARITY-2026-09-17`; co-owner [W4-PRODUCT-1](INTEGRATED_COMPILER_PLAN.md#w4-product-1).

Outcome: **gfx1201's macro tile now applies to every storage the chip admits, which is worth 3.3x-4.5x on fp8 and the integer pair, and the gap recorder can measure a storage at all.** The recorder was the stated gate and it is done: `--dtype` selects fp16, bf16, fp8_e4m3, fp8_e5m2, int8 or int4, each with its own reference and budget rather than f16's, because an integer product is exact in i32 and a float tolerance there would hide the packing defects the int4 nibble order can produce. Measured on Tajasarus, 1x1 → 4x4: fp8_e4m3 15.80 → 58.33 at 1024³ and 20.23 → 78.01 at 2048³, int8 15.96 → 58.36 and 19.70 → 76.55, int4 14.95 → 49.85 and 16.96 → 76.44. The generator already emitted that panel correctly for all three, so this was a **selection omission and never a codegen limit**: the rule was derived on f16 and applied in the f16/bf16 branch alone, leaving fp8 (2x f16's RDNA4 ceiling) and int4 (4x) on the smallest tile the chip has. The predicate is hoisted once in `getInferredMatmulSchedule` and shared by all three branches, the Python row mirrors it, and the artifact projection caught the change mid-flight before either box ran. One device row per storage at 1024³ on the owning device, the integer ones asserted exactly rather than with a tolerance; the pre-existing fp8 rows reach 65 in any extent and so could never have caught this.

**The more important result was settled the same day, against AMD's own document.** The 2026-09-18 packet recorded the panel axis as exhausted because 4x8 and 8x8 fell off a VGPR cliff. Compile-only measurement showed every panel compiling to `vgpr_count` 256 with the **shipped** 4x4 panel already spilling 126 registers (1433 at 4x8, 4247 at 8x8), and the ROCm generators set no occupancy attribute anywhere. An external RDNA4 FP8 recipe attributes exactly that to occupancy policy — one wave per SIMD gets the full register file — so a `waves-per-eu` option stamped `rocdl.waves_per_eu`, and it changed nothing at 0, 1 or 2 even though the attribute demonstrably reached the `llvm.func`. The RDNA4 ISA says why: §3.3.2.1, *"VGPRs are allocated in blocks of 16 for wave32 or 8 for wave64, and a shader may have up to 256 VGPRs"*, and dynamic VGPR mode (§3.3.3) caps at the same 256 with a 32-VGPR block size, `S_ALLOC_VGPR` returning `SCC=0` above it. The 768 KiB file is per CU and shared across wave slots. **256 is the hardware ceiling, the original "panel axis is exhausted" reading was right, and CDNA's 512-VGPR recipe does not transfer.** The knob is deleted rather than left as an unconsumed declaration, and the panel docstring now cites the ISA. Our own extracted archive could not have answered this — it holds section titles with no VGPR-allocation text — which is why the first pass recorded the question instead of guessing; the owner's PDFs on Princess-Luna (`~/AMD_GPU_ISA_DOCS`, RDNA4 / RDNA3.5 / CDNA5) are the source when the archive cannot answer. What survives as work is the 126-VGPR spill at the *shipped* panel: a real cost whose only remaining lever is fewer live registers, since the ceiling cannot be raised.

A second lead, `ROCM-GLOBAL-LOAD-TR-1`, came from the same external material and was confirmed against our MLIR 23 rather than taken on: `amdgpu.global_transpose_load` wraps RDNA4's `global_load_tr` on gfx1200+, valid at (8 bits, 8 elements) and (16 bits, 8 elements), so it covers f16/bf16 and fp8/int8 on gfx1201 and excludes int4. That is the documented remedy for the standing asymmetry where A takes one `vector.load` and B scalarizes into 16 guarded loads, and it is upstream, so no intrinsic hand-rolling. It also retires an avenue: the **LDS** transpose family is gfx950/CDNA4 only, so an LDS-staged body on either of our chips must always pay a software transpose — which is consistent with the 2026-09-18 finding that it never wins a selection, and means the fix for B is global-to-register, not shared memory.

| Host | Targeted suites at `32f005d1` | Notes |
|---|---|---|
| Tajasarus (gfx1201) | **161 passed, 0 failed**; the four new low-precision device rows pass on the device | both trees rebuilt |
| Princess-Luna (gfx1151) | **112 passed, 0 failed** | first run reported 31 failures from a build that silently did nothing — the script called `.venv/bin/ninja`, which does not exist there, and grepped for `error:`, which that failure does not print |

Remaining: the low-precision K unroll on gfx1201, whose sweep was contaminated by a build running concurrently with the timing and is discarded rather than recorded; the gfx1151 integer K-unroll rule, which has a clean sweep but no device row; the 126-VGPR spill at the shipped 4x4 panel, whose only lever is now fewer live registers; `ROCM-GLOBAL-LOAD-TR-1`; raster-order selection, still blocked on counters neither WSL2 ROCm box can produce.

Evidence: `docs/audit/backend/rocm/todo.md` §"The low-precision panel, and an occupancy reading that is not yet evidence — 2026-09-19", `tests/unit/test_rocm_gfx1201_scheduled.py`, `benchmarks/rocm/record_typed_route_gap.py`.
<!-- entry-fields:end -->

### 2026-09-19 — the K unroll follows the storage's load width, and three AMD sources confirm the contract

Owner: [COMPILER-DEVEX-1](INTEGRATED_COMPILER_PLAN.md#compiler-devex-1)

PRs: branch `claude/lowp-k-unroll`, sync `GFX1201-PARITY-2026-09-17`; co-owner [W4-PRODUCT-1](INTEGRATED_COMPILER_PLAN.md#w4-product-1).

Outcome: **`rocm_k_unroll` depends on the storage, not just the chip, and the reason is load width rather than a fitted curve.** A fragment load is 8 elements per lane whatever the storage, so fp16 saturates the 128-bit interface, fp8 and int8 use 64 bits and int4 uses 32 — a narrower operand needs more slabs in flight to keep the path busy. Measured on the panel each chip selects, fp8 goes 76.6 → 115.1 TFLOP/s at 2048³ and 68.5 → **128.5** at 4096³ by taking k=4, int8 tracks it within 1% at every shape, and int4 on gfx1151 goes 14.7 → 27.3 at 2048³. **fp8 and int8 agreeing at every shape is the check on the mechanism**, since they are unrelated storages of equal width. The rule is width-derived on gfx1201 and enumerated on gfx1151, because that chip has no fp8 WMMA and falling through on width would hand a storage it cannot execute the rule measured for int8; an unlisted storage takes the single-slab loop on either chip. Two gains were declined for sitting inside the run-to-run spread: gfx1151 int8 at 2048³ (7%) and gfx1201 int4 at 1024³ (6% across all three). Device rows assert the route's `_k{n}` suffix against the rule, so a rule change and its evidence move together. fp8's best row is now 128.5 against f16's 90.3, on a 383 TFLOP/s fp8 ceiling.

**Three AMD sources were read against the contract page and all three confirm it.** The gpuopen RDNA4 WMMA guide (parts 1-3), a community RDNA4 guide, and the rocWMMA docs. The column-distributed accumulator mapping now has three independent statements, each including the warning this repo already gives that a symmetric probe cannot tell the two readings apart; the int4 signedness and clamp contract is confirmed by the builtin's signature carrying `neg_a`, `neg_b` and `clamp` explicitly; and "both operands K-major, 8 contiguous elements, 128-bit vectorized loads" is what the page already says. Nothing changed — which for a contract whose failure mode is a few silently wrong tiles is the result worth having.

What they add is three recorded items. **`ROCM-EXTENDED-K-1`**: the deeper unroll is a workaround for an instruction we cannot emit, since AMD's extended-K technique fuses two WMMAs so one load fetches 16 elements bit-identically, and for int4 the hardware already has that shape as `V_WMMA_I32_16X16X32_IU4` — which is **in our own enumerated table and unreachable**, because `materializeMma` pins `kBlocks = 1`. That is a Decision #29 gap and the principled version of int4's measured k=4 win. **`ROCM-GLOBAL-LOAD-TR-1` gains a second candidate**: besides `global_load_tr`, AMD's identity-matrix WMMA transposes in registers with no special load, their answer to CUDA's `ldmatrix.trans`. And **rocWMMA does not list gfx1151**, so it is not an available Tier-3 delegate on Princess-Luna though it would be on Tajasarus — an asymmetry Decision #28 tiering had not recorded.

A GEMM layout audit prompted by the same material **found a silent-wrong-tiles path, and its first pass missed it.** Searched with substrings the audit reported clean, and most of that holds: the attention family branches correctly on `rdna4 ? e + 8*half : 2*e + half`, the typed route resolves row and column from the fragment family, and the two control-for-WMMA generators hardcode the RDNA3 formula but **fail closed** on gfx1201 because `llvm.amdgcn.wmma.f32.16x16x16.f16` is a gfx11 intrinsic gfx12 cannot select. Re-run structurally with `ast-grep` (added to the fleet this day), the complementary question — which accumulator math carries *no* arch branch — found `emitCanonicalLdsBody` storing at row `2*e + lhi` with no guard on its dispatch, confirmed on Tajasarus to accept `arch=gfx1201` and emit a kernel with zero errors. It emits `tessera_rocm.wmma`, the **arch-resolving** Target IR op, so unlike the control generators it has nothing to fail on: it would lower to the RDNA4 instruction and scatter the result to RDNA3 rows. It refuses by name now (`ROCM_CANONICAL_LDS_ARCH_UNSUPPORTED`, registered, two fixtures). The lesson is about the audit rather than the tool — "which sites lack a guard" is not answerable by searching for the guard, and reporting clean from that search put a false claim in the queue for several hours.

| Host | Targeted suites at `19aa67e5` | Notes |
|---|---|---|
| Tajasarus (gfx1201) | **175 passed, 0 failed** | both trees rebuilt before any timing |
| Princess-Luna (gfx1151) | **122 passed, 0 failed** | |
| Mac (M1 Max) | **18487 passed, 0 failed**, full `-m "not slow"` sweep | doc, plan and registry gates clean |

Remaining: `ROCM-EXTENDED-K-1`, now the highest-ceiling ROCm item; `ROCM-GLOBAL-LOAD-TR-1` with its two candidates; the 126-VGPR spill at the shipped 4x4 panel, whose only lever is fewer live registers; raster-order selection, still blocked on counters neither WSL2 ROCm box can produce.

Evidence: `docs/audit/backend/rocm/todo.md` §"The K unroll follows the storage, and three AMD sources that reframe it — 2026-09-19", `benchmarks/baselines/lowp_k_unroll_20260919/`, `docs/backends/rocm/wmma-fragment-layout.md` §§6-7, `tests/unit/test_scheduled_matmul_consumers.py`, `tests/unit/test_rocm_gfx1201_scheduled.py`.
<!-- entry-fields:end -->

### 2026-09-19 — ROCM-EXTENDED-K-1: the double-K int4 instruction is reachable, and it loses to the unroll

Owner: [COMPILER-DEVEX-1](INTEGRATED_COMPILER_PLAN.md#compiler-devex-1)

PRs: branch `claude/rocm-extended-k`, sync `GFX1201-PARITY-2026-09-17`; co-owner [W4-PRODUCT-1](INTEGRATED_COMPILER_PLAN.md#w4-product-1).

Outcome: **RDNA4's native `V_WMMA_I32_16X16X32_IU4` is emitted by the typed route for the first time, it is exact, and it is slower than the K unroll it was expected to replace.** The item was opened on an assumption worth stating because it was half right: a fragment load is 8 elements per lane whatever the storage, so int4 moves 32 of 128 bits at K=16, the unroll reaches the bandwidth by issuing *more* loads, and the double-K instruction fetches 16 elements in one. The first two clauses hold; the conclusion does not. Measured on Tajasarus in TOP/s, shipped k16-unroll-4 against k32-unroll-1: 47.5 vs 42.8 at 1024³, 98.5 vs 91.5 at 2048³, 106.9 vs 100.8 at 4096³, every row exact against the i32 reference with `v_wmma_i32_16x16x32_iu4` confirmed in the disassembly. Unrolling the double-K form collapses it to 34.0 and 20.8, which is register pressure. Both forms move the same bytes per lane; the unroll keeps two **independent** MMAs in flight while the double-K is one dependent instruction, and on a memory-latency-bound body the instruction-level parallelism beats the instruction density. Load width explains which storages want a deeper unroll; it does not imply the widest instruction wins. So it ships as a **capability, not a selection** — `rocm_k_unroll` is unchanged and a device row pins the instruction and the exact result.

Five independent gates pinned K to 16, and the one the queue named was a red herring: `materializeMma`'s `kBlocks = 1` is not the lever, the descriptor's K is. The real five were the generator's descriptor gate, the typed body's four K-width constants, the emitted fragment types, the Target matmul contract check, and the view layout. Two layers needed nothing: `resolveFragmentLayout` already selects the K=32 instruction and the nibble packer already emits two words in the documented order.

One gate was an architectural defect in its own right. The generator attached a single `tile.layout` of `{16, 16}` to three different tiles — the A view `{M, K}`, the B view `{K, N}`, and the accumulator `{M, N}` — which coincide only at K=16. Each is now built from what it describes, byte-identical at K=16. And the arch check stays in one place: the generator does not know the target, so it admits the shape and the lowering adjudicates, verified by a K=32 int4 fragment aimed at gfx1151 refusing with `ROCM_FRAGMENT_ILLEGAL_ARCH_DESCRIPTOR`.

| Host | Suites at `dbe37ab5` | Lit |
|---|---|---|
| Tajasarus (gfx1201) | **175 passed, 0 failed**; the double-K rows pass on the device | ROCm 75/75 in both trees |
| Princess-Luna (gfx1151) | **141 passed, 0 failed** | ROCm 75/75 |
| Mac (M1 Max) | **18487 passed, 0 failed**, full `-m "not slow"` | doc, plan and registry gates clean |

Remaining: `ROCM-GLOBAL-LOAD-TR-1` with its two candidates; the 126-VGPR spill at the shipped 4x4 panel, whose only lever is fewer live registers; raster-order selection, still blocked on counters; the sparse twin `V_SWMMAC_I32_16X16X64_IU4`, now reachable by the same route and unmeasured.

Evidence: `docs/audit/backend/rocm/todo.md` §"ROCM-EXTENDED-K-1 closed: the instruction exists, works, and loses — 2026-09-19", `tests/unit/test_rocm_gfx1201_scheduled.py`, `python/tessera/compiler/scheduled_matmul.py`.
<!-- entry-fields:end -->

### 2026-09-19 — ROCM-GLOBAL-LOAD-TR-1: the strided B gather becomes one instruction

Owner: [COMPILER-DEVEX-1](INTEGRATED_COMPILER_PLAN.md#compiler-devex-1)

PRs: branch `claude/rocm-b-transpose`, sync `GFX1201-PARITY-2026-09-17`; co-owner [W4-PRODUCT-1](INTEGRATED_COMPILER_PLAN.md#w4-product-1).

Outcome: **the A/B asymmetry is closed for f16 and bf16.** B is stored row-major `[K][N]` and each lane wants a column, so the B fragment was eight guarded scalar loads where A took a single `vector.load`. RDNA4's `GLOBAL_LOAD_TR_B128` reads a 16x16 tile and transposes it into the registers, and the typed route emits it now — four per K slab at the 4x4 panel, confirmed in the disassembly. Worth, at the 4x4 panel with K unroll 2: 1024³ **65.9 against 54.2** TFLOP/s (+21.6%), 1536³ 79.2/74.3, 2048³ 75.4/80.5, 2560³ 94.3/89.5, 3072³ 92.8/89.4, 4096³ **93.9/89.4** (+5.0%) — five of six shapes.

The ISA does not state the permutation, so it was measured rather than guessed, with `B[r][c] = r*16 + c` because a symmetric probe cannot tell a row reading from a column one. The wave does an 8x8 transpose inside each group of 8 lanes, `received(L, j) = R(8*(L/8) + j)[L % 8]`, and solving that for the fragment gives the per-lane address, verified at 256/256 elements with the arbitrary addressing failing the same check as a control.

Two results kept because they are the ones that could mislead later. **2048³ is a genuine reversal**, −6.3%, reproducing at five runs, so it is not sampling noise; a leading dimension of exactly 2048 is the obvious aliasing suspect but no counters exist on either WSL2 ROCm box to confirm it, so the instruction stays on everywhere rather than being special-cased around one unexplained point. And **8-bit storages are excluded by measurement**: `TR_B64` is a different permutation, and enabling it on the strength of the matching per-lane width produced wrong results on device for every 8-bit storage — 16 failing rows across fp8 and both integer widths, with f16 and bf16 untouched, which is also what localised it.

Infrastructure that outlives the item: the `amdgpu` dialect is registered in both drivers and declared in `TileToROCM`'s `getDependentDialects`, `convert-gpu-to-rocdl` lowers the op to `rocdl.global.load.tr.b128` with no additional pass, and the generic kernel argument is cast to the global address space the op requires.

| Host | Suites | Lit (lane, not suite) |
|---|---|---|
| Tajasarus (gfx1201) | **177 passed, 0 failed**, plus five rows pinning which storages take the instruction | 433/495, 62 unsupported; `check-tessera-rocm` 75/75 |
| Princess-Luna (gfx1151) | 168 passed, 103 skipped — the regression lane; gfx1151 has no `GLOBAL_LOAD_TR`, so this is *no evidence* for the instruction and is not offered as any | 491/495, 4 unsupported; `check-tessera-rocm` 75/75 |
| Mac (M1 Max) | 129 passed, 162 skipped — host-portability only; the ROCm backend is not configured there, so a green Mac build says nothing about this C++ | 452/495, 43 unsupported |

Counts recorded 2026-09-19 at the branch head. No single box configures every
backend, so the suite result is the fleet union (`scripts/check_lit_fleet_union.py`);
the rows above are lane results.

Remaining: the 8-bit `TR_B64` mapping, measured the same way, which would extend this to fp8 and int8; the 2048³ anomaly, which needs counters; AMD's identity-matrix in-register transpose as the int4 fallback; the 126-VGPR spill at the shipped panel; `ROCM-MIXED-FP8-1`; raster-order selection.

Evidence: `docs/audit/backend/rocm/todo.md` §"ROCM-GLOBAL-LOAD-TR-1 closed: the B gather becomes one instruction — 2026-09-19", `docs/backends/rocm/wmma-fragment-layout.md` §8, `tests/unit/test_rocm_gfx1201_scheduled.py`.

<!-- entry-fields:end -->

### 2026-09-19 — ROCM-MIXED-FP8-1: the mixed OCP FP8 pairs execute, and two gates that were not checking what they claimed

Owner: [COMPILER-DEVEX-1](INTEGRATED_COMPILER_PLAN.md#compiler-devex-1)

PRs: branch `claude/rocm-b-transpose`, sync `GFX1201-PARITY-2026-09-17`; co-owner [W4-PRODUCT-1](INTEGRATED_COMPILER_PLAN.md#w4-product-1).

Outcome: **all four OCP FP8 pairings execute natively on gfx1201, each selecting its own instruction.** `fp8_fp8` and `bf8_bf8` already did; `FP8_BF8` and `BF8_FP8` were declared by `wmma_dtype_forms` and emitted by nothing — Decision #29's unconsumed declaration. Measured: `fp8_fp8` rel 0.00e+00, `fp8_bf8` 2.60e-08, `bf8_fp8` 2.59e-08, `bf8_bf8` 5.21e-08, the mixed figures being accumulation order alone (an fp8 product is exact in f32).

The recorded blocker — "the Graph matmul contract requires `a_dtype == b_dtype`" — was true and was not the bottom. The single-storage assumption ran the **whole depth of the stack**, and every layer refused rather than computing wrong numbers, which is why this was a reachability gap and never a correctness bug: `MatmulSchedule` carried one `storage` so the descriptor wrote the same name to both slots; the generated kernel typed both operand buffers from A; the fragment types inherited A's element; the generator gate, the Target matmul gate and the packager each required `a == b`; the launch ABI was keyed on A alone with no id naming a pair; the bindings gave B A's dtype; and the submit path validated B against A's. Eight successive refusals, each diagnosed from the device.

**Two gates were then found not to be checking what their names claimed, and both had been green since they were written.**

*The instruction claim.* Six device rows assert which matrix instruction was emitted, each with its own copy of finding `llvm-objdump` and three policies for not finding it — one of which dropped the assertion and still passed, in a row named `..._select_their_instruction_...`. They now share `tests/_support/rocm_isa.py`, where a missing disassembler is a failure, because the caller has already asserted it is the owning device. Two claims were weaker than their names: `test_gfx1201_double_k_int4_emits_its_instruction_and_is_exact` asserted only numbers, and **the numbers cannot distinguish the form** — the double-K int4 WMMA and two K=16 WMMAs compute the identical exact integer product, so the sole owner of RDNA4's double-K capability would have passed on a generator that quietly kept K=16. And every row required a mnemonic while forbidding none, so none could see a *swap*: finding `fp8_bf8` does not establish that `bf8_fp8` is absent. Falsified on device — asking the `e4m3 x e5m2` row for the mirror now fails naming what was actually emitted (`{'v_wmma_f32_16x16x16_fp8_bf8': 4}`).

*The reachability list.* It checked one direction. Two of its own entries were self-fulfilling: the helper answered "unreachable" for the reduced-precision accumulators from the same hardwired rule that had written them, without asking the route. Probed, the f16 entry named a C++ diagnostic that never fires on that path, and the bf16 entry was proved by a `KeyError` in the test fixture's own output-dtype map — evidence about the fixture, not the compiler. Entries now carry the marker the actual refusal must contain, every answer comes from the route, and a form still refused *for a different reason than recorded* fails. All three directions were falsified before being recorded.

Supporting that marker, the typed route's fall-through refusal named neither the dtypes nor the target and was identical for a reduced-precision accumulator, an unsupported storage and a target with no GEMM for the pair; it now names all three (Decision #21). Registering that code exposed the same defect class once more: the **Python diagnostic scanner is a prefix allowlist**, so 16 code-shaped diagnostics raised from Python were invisible to their own registry gate — the failure its own `GRAPH_IR_` note records happening before. It now matches the code *shape*, as the C++ scan always has; the 15 pre-existing codes are a shrink-only ratchet (`DIAG-PY-BACKLOG-1`), not an allowlist.

Route confirmed rather than assumed: Tile IR (`tile.matmul_kernel`/`tile.mma_desc`/`tile.epilogue`) → `tessera_rocm.wmma_gemm` with zero unconsumed `tile.*` → upstream `convert-gpu-to-rocdl` → `gpu-module-to-binary` → ISA. `_compile_native_tile_ir` hard-fails without `tessera-opt`, so there is no Python packager to degrade into.

| Host | Suites | Lit (lane, not suite) |
|---|---|---|
| Tajasarus (gfx1201) | **95 passed** (scheduled + reachability) and **46 passed** (sparse), 0 failed | 433/495, 62 unsupported; `check-tessera-rocm` 75/75 |
| Princess-Luna (gfx1151) | 168 passed, 103 skipped — regression lane only; RDNA3.5 has no FP8 WMMA at all, so it is no evidence for this item | 491/495, 4 unsupported; `check-tessera-rocm` 75/75 |
| Mac (M1 Max) | 129 passed, 162 skipped — host portability; the ROCm backend is not configured there | 452/495, 43 unsupported |

Remaining: the 8-bit `TR_B64` mapping; the 2048³ transpose-load reversal, which needs counters neither WSL2 ROCm box can produce; AMD's identity-matrix in-register transpose as the int4 fallback; implementing the register-pressure reduction now that `ROCM-VGPR-PRESSURE-1` is scoped; raster-order selection; the sparse `V_SWMMAC_I32_16X16X64_IU4`, reachable and unmeasured; and `DIAG-PY-BACKLOG-1`.

Evidence: `docs/audit/backend/rocm/todo.md` §"ROCM-MIXED-FP8-1 closed: the pair was a Schedule field, and two gates were not checking their own claim — 2026-09-19", `tests/unit/test_rocm_gfx1201_scheduled.py`, `tests/unit/test_rocm_wmma_form_reachability.py`, `tests/_support/rocm_isa.py`.

<!-- entry-fields:end -->

### 2026-09-20 — Examples expose the lost MoE route name

Owner: [E2E-REAL-6](INTEGRATED_COMPILER_PLAN.md#e2e-real-6)

PRs: [#792](https://github.com/gstoner/tessera/pull/792)

Outcome: The S8 Qwen3-MoE compiler example became an executable Apple CPU
oracle and exposed a silent wrong-answer path: Graph IR retained the third
route tensor, but runtime artifact metadata dropped whether that tail meant
`scores` or `route`, so Apple CPU/GPU execution selected round-robin routing.
Both frontends now record the declared optional names in `kwargs["extras"]`;
all foundation-target artifacts retain `route`, and the portable Apple CPU
launch matches the public-operation oracle.

Remaining: NVIDIA, ROCm, x86, and Apple GPU native execution remain owned by
their architecture-specific fixtures and exact-device evidence; this shared
fix does not transfer Apple results or promote a selector.

Evidence: Focused compiler-example, frontend, examples-audit, generated-doc,
and audit-plan gates; sync `EXAMPLE-MOE-OPTIONAL-BINDING-2026-09-20`.

<!-- entry-fields:end -->

### 2026-09-21 — Exact gfx1201 MXFP4 W4A8 reaches FP8 WMMA

Owner: [ROCM-MXFP4-W4A8-1](INTEGRATED_COMPILER_PLAN.md#rocm-mxfp4-w4a8-1)

PRs: [#801](https://github.com/gstoner/tessera/pull/801); sync
`ROCM-MXFP4-PHYSICAL-CONTRACT-2026-09-21`.

Outcome: The dedicated gfx1201 native package now has two proved ABIs. The
scalar exact-per-K32 package is the executable oracle; the production selector
defaults to a wave32 route which batches independent fragment loads, converts
packed E2M1 exactly to E4M3, issues two
`v_wmma_f32_16x16x16_fp8_fp8` operations per scale group, scales that local
FP32 partial, and then joins the running accumulator. Tajasarus proved both
ABIs bit-exact after BF16 rounding at ragged `17x19x64` and `32x32x128`.
Review follow-through aligns Schedule macro K to each complete scale group
(`scale_k=128` now carries `block_k=128`) and preserves reserved E8M0 code zero
as a zero block through folded-row conversion and its losslessness check.

Follow-through: The generic `tessera.scaled_matmul` path now owns a
content-addressed Schedule record, a first-class Tile scaled-partial carrier,
and a gfx1201 Target directive. The packed `rocm_mxfp4_w4a8_exact_v1` form
binds that directive to the proved WMMA ABI while the logical W8A8 form remains
distinct. Folded-row conversion requires explicit approximate-policy opt-in
and reports quantified loss.

Remaining: The in-pipeline binary materializer, tuned decode/prefill selection,
and counter-capable comparison remain open; no gfx1200 claim follows from this
gfx1201 proof.

Evidence: [recorded HSACO/ISA/resource packet](../../../benchmarks/baselines/gfx1201_mxfp4_w4a8_20260921/README.md),
`tests/device/rocm/test_mxfp4_w4a8_exact.py`, and
`tests/unit/test_rocm_mxfp4_native.py`.

<!-- entry-fields:end -->

### 2026-09-22 — Exact MXFP4 K-step and fragment-prefill loop

Owner: [ROCM-MXFP4-W4A8-1](INTEGRATED_COMPILER_PLAN.md#rocm-mxfp4-w4a8-1)

PRs: branch `codex/gfx1201-mxfp4-kstep-prefill`.

Outcome: a portable isolated-scale-group contract lowers to a selective AMD
VMEM/WMMA scheduling boundary; padded fragment-word LDS prefill is exact after
an explicit producer drain, and the former wide-N corruption passes 10/10
repeats. The full gfx1201 MXFP4 device file passes 16/16. Alternating matched
timing retains HSACO ISA/resources and measures Tessera at 0.0490/0.0885 ms
decode and 0.5174/4.3722 ms prefill.

Remaining: Radiance is 1.33x/1.02x faster on decode and 3.44x/4.81x on
prefill. The prefill gap requires a separately opted-in approximate
row-reference ABI and a BM256/TM4 multi-output-wave tile; the exact route must
remain the oracle. Two-stage, streaming-cache, and sub-percent tuning results
are not promoted.

Evidence: [Tajasarus K-step/prefill packet](../../../benchmarks/baselines/gfx1201_mxfp4_kstep_prefill_20260922/README.md), with alternating HIP-event samples, independent FP32 oracle, pinned Radiance/libr4d binaries, and per-HSACO ISA/resource fields.

<!-- entry-fields:end -->

### 2026-09-22 — Folded MXFP4 BM256/TM4 prefill on gfx1201

Owner: [ROCM-MXFP4-W4A8-1](INTEGRATED_COMPILER_PLAN.md#rocm-mxfp4-w4a8-1)

PRs: branch `codex/gfx1201-mxfp4-folded-prefill`.

Outcome: an explicitly approximate load-time E4M3 fold reaches a distinct
gfx1201 launch ABI. The descriptor records fold loss and binds the converted
weight and row-reference payloads by SHA256; the runtime refuses mismatched
payloads. BM256/TM4 prefill stages padded A/B tiles and emits 32 FP8 WMMAs.
Tajasarus proves deliberately inexact underflow and ragged N. Matched
lossless-fold timing improves over exact K32 by 2.85x/3.55x and trails pinned
Radiance by 1.25x/1.38x on the two production shapes.

Remaining: expanded E4M3 weight traffic and A/B staging account for an
unquantified part of the gap. The exact K32 route stays the default oracle;
there is no selector-default or gfx1200 promotion.

Evidence: [folded prefill packet](../../../benchmarks/baselines/gfx1201_mxfp4_folded_prefill_20260922/README.md),
`tests/device/rocm/test_mxfp4_folded_prefill.py`.

<!-- entry-fields:end -->

### 2026-09-22 — Folded MXFP4 scale repair and IKF diagnostic

Owner: [ROCM-MXFP4-W4A8-1](INTEGRATED_COMPILER_PLAN.md#rocm-mxfp4-w4a8-1)

PRs: branch `codex/gfx1201-folded-ikf-scale`.

Outcome: the folded gfx1201 epilogue now combines row-reference and
activation scales before applying the FP32 accumulator, with an FP64 rare
path for overflowing or underflowing scale products. Tajasarus passes six
device tests, including cancellation, zero-partial, and finite-after-overflow
regressions. Refreshed exact/folded/Radiance
outputs agree on lossless inputs. A compile-time-only same-CTA phase probe
preserves output but fails structural perturbation: 64 versus 32 FP8 WMMAs
and 24 versus four barriers after synchronizing every phase boundary. Both
diagnostic packets refuse attribution.

Remaining: IKF-P0 cross-CU/read-cost validation and an ISA-preserving
measurement path. No phase fractions feed cost models, selector promotion,
or exact gfx1200 claims; exact K32 remains the correctness oracle.

Evidence: [matched folded packet](../../../benchmarks/baselines/gfx1201_mxfp4_folded_prefill_20260922/README.md),
[phase refusal packet](../../../benchmarks/baselines/gfx1201_folded_phase_diagnostic_20260922/README.md).

<!-- entry-fields:end -->

### 2026-09-22 — GFX1201 external phase-profiler preflight

Owner: [ROCM-MXFP4-W4A8-1](INTEGRATED_COMPILER_PLAN.md#rocm-mxfp4-w4a8-1)

PRs: branch `codex/gfx1201-folded-carrier-clock`.

Outcome: an exact-device, source-bound preflight checks the external
`rocprofv3` PC-sampling path without changing the folded kernel. Tajasarus
reports the RX 9070 XT but has no `/dev/kfd`; the packet refuses sample
collection, clock validation, phase attribution, and promotion. A separate
kernel-trace attempt produced no trace artifact.

Outcome also includes a distinct folded physical Graph→Schedule→Tile→Target
carrier: the full-K row-reference partial, K64 stage, BM256/TM4 geometry,
pointer ABI and approximate policy survive the hash-bound materializer. An
isolated assertions-enabled Tajasarus compiler passed the new FileCheck
fixture; a `65x48x64` package launched and the folded device file passed 7/7.

Remaining: restore a profiler-capable gfx1201 environment and validate
production-image PC sampling plus cross-CU clocks. Rerun the carrier proof
from a clean final revision, then widen its shape coverage and wire the
high-level frontend. Exact K32 remains the default oracle.

Evidence: [profiler refusal packet](../../../benchmarks/baselines/gfx1201_phase_profiler_preflight_20260922/README.md),
`tests/unit/test_gfx1201_phase_profiler_preflight.py`.

<!-- entry-fields:end -->

### 2026-09-22 — GFX1201 folded frontend and layout-verified prefill

Owner: [ROCM-MXFP4-W4A8-1](INTEGRATED_COMPILER_PLAN.md#rocm-mxfp4-w4a8-1)

PRs: branch `codex/gfx1201-folded-frontend-proof`

Outcome: a typed physical frontend authors `tessera.scaled_matmul` Graph IR
from A, token scales, and load-time folded weights, then binds the selected
Schedule/Tile/Target, ABI, and HSACO to a route receipt. Tajasarus passes
11/11 folded device cases, including both production prefill shapes and one
deliberately lossy approximate-oracle case. The original Radiance packet
did not record its WPERM mode despite fragment-order input and is now
explicitly refused for selector use. A layout-verified v2 matched run leaves
the folded route 1.28×/1.40× behind pinned Radiance. A static selected-symbol
ISA census and requested-byte model identify doubled B demand but do not
establish measured DRAM or stage fractions. A one-load B non-temporal
ablation emits the intended ISA modifier, preserves full BF16 output,
and regresses 20%/104% against the matched baseline, so it is unselected.

Remaining: exact K32 stays default. Test A-staging reuse/load scheduling on
matched inputs; extend nonuniform numerical envelopes; restore a usable
gfx1201 profiler API and close IKF-P0 cross-CU/read-cost gates before L2/L3
attribution or selector training. General public logical-op integration
remains outside this bounded physical frontend.

Evidence: [frontend and census packets](../../../benchmarks/baselines/gfx1201_mxfp4_folded_frontend_20260922/README.md),
the historical [layout-unverified packet](../../../benchmarks/baselines/gfx1201_mxfp4_folded_prefill_20260922/README.md),
`tests/device/rocm/test_mxfp4_folded_prefill.py`, and
`tests/unit/test_gfx1201_folded_staging_census.py`.

<!-- entry-fields:end -->

### 2026-09-22 — GFX1201 MXFP4 receipt and lowering coverage follow-on

Owner: [ROCM-MXFP4-W4A8-1](INTEGRATED_COMPILER_PLAN.md#rocm-mxfp4-w4a8-1)

PRs: branch `codex/gfx1201-mxfp4-lowering-coverage`

Outcome: `hsaco_sha256` now binds the emitted HSACO bytes, while
`artifact_image_digest` names the composite image identity. Tajasarus passes
13/13 folded device cases, adding ragged and multi-K-step frontend lowering;
the generic exact selector refuses folded layout without opt-in. A clean
layout-matched rerun leaves folded/Radiance at 1.28×/1.40×. The older B-cache
packet's receipt metadata was repaired from its retained payload hash without
changing its losing timing. The current-main scheduled closure independently
passes 95/95 with zero skips on rebuilt LLVM/MLIR 23.1.1 at `89b2f2fd`.

Remaining: exact K32 remains default; no automatic folded selector or
profiler-based phase attribution is promoted. Continue controlled A/B staging
work and nonuniform numerical envelopes on the owning device.

Evidence: [refreshed frontend packet](../../../benchmarks/baselines/gfx1201_mxfp4_folded_frontend_20260922/README.md),
`tests/device/rocm/test_mxfp4_folded_prefill.py`, and
`tests/unit/test_gfx1201_folded_staging_census.py`.

<!-- entry-fields:end -->

### 2026-09-26 — ROCM-SPLIT-K-1: ordered cross-workgroup split-K lands on gfx1201

Owner: [ROCM-SPLIT-K-1](INTEGRATED_COMPILER_PLAN.md#rocm-split-k-1)

PRs: branches `claude/amd-x86-alpha-lanes-splitk` (merged into `claude/amd-x86-alpha-lanes` as cecabc5d) and `claude/amd-x86-alpha-lanes-splitk-fixes` (pre-PR review fixes).

Outcome: cross-workgroup split-K with an **ordered** reduction on the gfx1201 typed route, f16/bf16. One decider -- `selectGfx1201SplitK` in Graph->Schedule -- with `rocm_tiling.select_split_k` as its declared oracle, compared on every package (Decision #31). `split_k` + `split_k_reduction="ordered"` are a semantic pair carried through `schedule.matmul` (and its digest, only when S>1), `tile.matmul_kernel` and `tessera_rocm.wmma_gemm`, every verifier failing closed. The generator emits a partial (grid.z = slice, fp32 workspace, no epilogue) and an ordered reduce that applies the epilogue once; the descriptor declares the workspace in its typed field. Router gate 16x256x2048 on Tajasarus, paired and interleaved, host wall clock: S=2 is 2.05x (fp16) / 2.01x (bf16) over the unsplit kernel of the same Tile IR, both launches counted.

Remaining: per-shape slice rule (the measurement-only sweep had S=4/S=8 faster still on this shape); fp8/int split; device-clock timing witness; gfx1151 (never split; no evidence).

Evidence: `benchmarks/baselines/rocm_split_k_20260926/` (timing packet, README, device-test logs with host and commit).

<!-- entry-fields:end -->

Review fixes (2026-09-26). (1) The S*M*N*4 scratch was only in untyped provenance -- a Decision #32 under-declaration; it is now `LaunchDescriptor.workspace` (256-aligned, launch lifetime, uninitialized because every element is written by exactly one slice), and the launcher allocates from it and refuses a provenance disagreement. **Not moved to `ROCMNativeProgram`:** that type is consumed only by the attention-backward launcher and would change `package_scheduled_matmul`'s return type for every caller (runtime `RuntimeArtifact`, the canonical GEMM benchmark, the gap recorder) for one extra entry; the reduce entry stays a declared second image entry point with its own ABI id. (2) The artifact now states the split the C++ Schedule wrote (`schedule_split_k`), so an oracle defect reports as oracle-vs-authority. (3) `k_unroll` is a performance key and yields: a derived unroll that does not divide the slice falls back to 1 and is recorded; a pinned one is refused. (8) `ROCM_SPLIT_K_NOT_APPLIED` is a registered warning and is emitted as one, and only when K >= 512 is misaligned: firing on every 16x256x256 decode GEMM was noise about a split that was never on offer. The "three idle SIMDs" explanation of S=4/8 is a hypothesis, not a measurement (no counters on WSL2).

### 2026-09-27 — ROCM-SPLIT-K-1: device-clock slice sweep; measured 256-workgroup target

Owner: [ROCM-SPLIT-K-1](INTEGRATED_COMPILER_PLAN.md#rocm-split-k-1)

PRs: branch `claude/gfx1201-lanes-splitk` (sub-branch of `claude/gfx1201-lanes`, sync `GFX1201-LANES-2026-09-27`).

Outcome: the split-K timing is now on the admitted ROCm device-clock route. `benchmarks/rocm/record_split_k_sweep.py` times each image between two `--tessera-device-clock-span` markers, and `build_rocm_profiler_packet` derives admission for every variant and run. The sweep covers 16 skinny shapes (router gates, MoE expert/decode GEMMs, M<=64) x {f16, bf16} x S in {1,2,4,8,16,32}, on Tajasarus. The slice rule changed in `selectGfx1201SplitK` and `rocm_tiling.select_split_k` together. The old trigger was tiles < 32 WGPs with `S = ceil(32/tiles)`. The new one splits when `2 x tiles <= 256` and takes the largest power-of-two `S <= min(32, 256/tiles)` whose slices are whole 32-wide K blocks of >= 256. It was chosen because every selection it makes was measured positive in both storages, not because it hits each shape's peak. Examples: router 16x256x2048 S=2 -> 8 (2.18x -> 3.40x fp16); 16x256x7168 S=2 -> 16 (2.78x -> 7.45x); 16x768x2048 1 -> 4 (2.43x); 64x512x2048 1 -> 2 (1.82x). These are 20 ms-window device-clock medians over 3 fresh processes x 9 interleaved rounds.

Remaining: fp8/int8 split (the generator's split gate would admit an fp8 f32-accumulate contract, but there is no device proof; int8 needs an i32 workspace). gfx1151 is unmeasured (never split). The S=32 cap and the 256-per-slice guard are conservative, not optima. A per-call workspace pool remains open (`runtime.launch` allocates per call; excluded from timing). M > 64 and dynamic shapes are unmeasured.

Evidence: `benchmarks/baselines/rocm_split_k_20260927/` (README with the full table, `sweep_20ms/`, `admission_50ms/`, packets, device-test log).

<!-- entry-fields:end -->

Details.
- **Negative side of the rule (measured).**
  - 32x1536x4096 (192 tiles) is neutral at S=2 and loses from S=8 (0.87-0.90x).
  - 16x2048x768 loses at S=8 (tiles x S = 1024).
  - 64x512x2048 loses at S=32 (tiles x S = 4096).
  - 16x256x256 loses at every S (128-wide slices, 0.89-0.92x), while 32x128x512
    gains at two 256-wide slices (1.16-1.18x). So the formerly unmeasured
    256-per-slice guard is now measured at its boundary. Narrower slices were
    positive at larger K, so it stays conservative there.
- **Admission.** Every packet takes the `device_clock_witness` route, and no
  packet has a window refusal. The shortest windows are 12.2 ms (20 ms sweep)
  and 20.8 ms (50 ms re-run).
  - Some rows fail the two-sided marker-overhead gate
    (`INSTRUMENTATION_CHANGED_THE_KERNEL` / `_OVERHEAD_EXCEEDED`), sporadically.
    A 50 ms re-run of the 7 affected shapes gives each selection a sweep with
    both sides admitted 3/3, **except 16x2048x768 S=2**. It gains 1.12-1.24x
    and wins in 25-27 of 27 rounds, but its packets are admitted in only 1-2 of
    3 runs in both sweeps.
- **Absolute times depend on window length.** In the 50 ms windows, long
  unsplit kernels ran 25-35% slower per launch (a lower clock state is one
  untested hypothesis; there are no counters on WSL2). Ratios are compared only
  within one sweep, and the conservative 20 ms sweep decides.
- **Determinism.** Every variant was checked bit-identical across two launches,
  with the output poisoned in between. Relative error vs f64 is <= 1.3e-5.
- **Sibling outcomes (`GFX1201-LANES-2026-09-27`).**
  - gfx1151: not applicable by rule. `_SPLIT_K_TARGET_WORKGROUPS` has no
    gfx1151 entry, so the ranking keeps its pre-sweep, unmeasured trigger and
    nothing splits. That remains correct in the sense that nothing unproven is
    emitted. Whether split-K pays on gfx1151 is unmeasured; gfx1201 evidence
    does not transfer.
  - NVIDIA / Apple / x86: not applicable. The rule is gfx1201-only, and
    unsplit schedule digests are unchanged.

### 2026-09-27 — GFX1201 folded MXFP4 load schedule in Target IR

Owner: [ROCM-MXFP4-W4A8-1](INTEGRATED_COMPILER_PLAN.md#rocm-mxfp4-w4a8-1)

PRs: branch `claude/gfx1201-lanes-mxfp4` (into the consolidated `claude/gfx1201-lanes` PR; sync `GFX1201-LANES-2026-09-27`).

Outcome: the opt-in folded BM256/TM4 prefill carries a four-key physical load schedule through Tile→Target: grouped M-major raster, register-staged next-slab prefetch, complete-tile vector-scale epilogue, and CU mode at two or more row blocks. The folded materializer consumes it and refuses a missing or undeclared value. Output is bitwise equal to exact K32 on matched inputs and to an independent oracle on nonuniform, ragged and lossy device cases. On Tajasarus, device-clock timing witnessed by HIP events across three processes gives 0.93–0.94×/0.79–0.80× the original schedule on the production shapes, 1.06–1.07×/0.98–1.01× pinned Radiance. Every one of ten shapes improves; Radiance is matched or passed from M=1024. Unconditional K16 steps and LDS fragment double-buffering lost; LDS-only barrier fences were neutral.

Remaining: one-row-block shapes (M≤256) stay 1.06–1.27× behind Radiance and the hot-stream probes say that gap is not operand traffic. The CU-mode mechanism is unattributed (no counters on WSL2). The M rule is fitted on gfx1201 only. Exact K32 stays default; no automatic folded selection.

Evidence: [load-schedule packet](../../../benchmarks/baselines/gfx1201_mxfp4_prefill_20260927/README.md), `tests/device/rocm/test_mxfp4_folded_prefill.py`, `tests/unit/test_rocm_mxfp4_folded_schedule.py`, `tests/tessera-ir/phase2/e2e_folded_mxfp4_rocm_load_schedule.mlir`.

<!-- entry-fields:end -->

### 2026-09-27 — ROCM-FP8-BLOCKSCALE-1: logical W8A8 block scaling binds on the typed gfx1201 route

Owner: [ROCM-FP8-BLOCKSCALE-1](INTEGRATED_COMPILER_PLAN.md#rocm-fp8-blockscale-1)

PRs: sub-branch `claude/gfx1201-lanes-blockscale` of the consolidated `claude/gfx1201-lanes` PR (sync `GFX1201-LANES-2026-09-27`).

Outcome: the logical W8A8 block-scale directive is bound on the typed gfx1201 route. Graph->Schedule derives `rocm_fp8_w8a8_blockscale{,_nk}_v1` from an fp32-scale e4m3 `tessera.scaled_matmul` (refusing a nonconforming one as `ROCM_FP8_BLOCKSCALE_CONTRACT`), `generate-wmma-gemm-kernel` emits the isolated scale-group body joined through the new `tile.fragment_scaled_accumulate`, TileToROCM binds the directive's `package_abi`, and `rocm_fp8_blockscale.py` binds the HSACO to a launch descriptor after checking every Target field. Device-proven on Tajasarus against an fp64 oracle (23 rows, both trees). Against AITER's unmodified Triton kernel with its tuned gfx1201 configs, device clock, paired: Tessera `[N, K]` / AITER geomean 0.65 at M <= 64, 1.09 at M = 256, 1.31 at M >= 1024.

Remaining: large-M performance (LDS-staged or multi-wave W8A8 body), bf16/f16 store epilogue, AITER split-K buckets unmeasured, `[K, N]` B gather.

Evidence: `benchmarks/baselines/gfx1201_fp8_blockscale_20260927/` (comparison + sweep JSON, device-test and lit logs, README naming host, commits, trees and timing source).

<!-- entry-fields:end -->

Why the W8A8 panel is not the unscaled one: each group's partial is a second live accumulator per fragment, so the unscaled 4x4 panel would carry 32 fragments -- 256 VGPRs before any operand. Measured at 1024x4096x1024: 32x32 231 VGPRs unspilled, 64x32 256 + 54 spilled, 64x64 256 + 337. A first cut that issued a whole K128 group (eight panels) straight-line spilled even at 32x32; the inner panel loop fixed that without changing what a group computes. The `[N, K]` weight is what made the route competitive: its B fragment is one K-contiguous vector load where `[K, N]` is a strided per-element gather (1.8-3x slower here).

Two measurement corrections before the recorded packet: the harness first read `hipDeviceAttributeWallClockRate` by parsing the HIP header and got the wrong enumerator (1 kHz); it now compiles a probe against the header, as `record_ssd_gpu.py` does. And windows sized on one cold probe came in at 1.5-4.6 ms, under the 5 ms admission floor; the harness now warms every arm together and re-runs any paired set whose windows fall short (every arm of the recorded packet cleared on the first attempt).

### 2026-09-27 — Every arbiter candidate identifies the code it runs

Owner: [W5.2](INTEGRATED_COMPILER_PLAN.md#w52)

PRs: branch `claude/autotune-emitted-identity` (Codex review P2 on PR #859).
Sync: `AUTOTUNE-EMITTED-IDENTITY-2026-09-27`.

Outcome: `Candidate.requires_artifact_identity()` was `tier == HAND_TUNED`, so
a SYNTHESIZED/EMITTED lane with no identity was served on the pin-based
toolchain identity alone, and a changed emitter kept a verdict measured for
its old kernel. Now every live candidate must match a stamped identity or the
verdict misses; the arbiter consults no opt-out. `compiler/emitted_code_identity.py`
(`tessera.emitted_source.v1`) identifies Python-emitted source (text +
`kernel_cache.cache_key` + the compile flags the pin does not fix, read from the
same flag list the compile step uses), PTX, checked-in sources plus the host
compiler's version, and Python/numpy lanes; `tessera-opt` images keep
`kernel_code.v2`. gfx1151 `fused_region` rows re-recorded on Princess-Luna
(winners unchanged; served, and missed after an emitter change with pins
unchanged). The 96 sm_120 registry rows were re-recorded on The-Super-Bear
(clean worktree at `1a737129`, fresh `build/` + `build-nvidia-cuda/`, under the
timing lock): every timed candidate stamped, 15 rows served by production
lookup with inferred dims (the 14 admissible before plus one f16 attention
row), all 96 miss when an NVIDIA emitter changes with the pins unchanged; 15
winners changed, none of them served (inadmissible before and after). An
independent review fixed three gaps: the finalizer merged two runs that timed
different code (now refused), an empty identity `{}` matched like a real one
(now a miss), and composed spectral lanes did not cover the inner FFT lane
they fall through to; new tests require each emitted lane's identity to equal
the digest of the exact source and command line its compiler receives.

Remaining: `AUTOTUNE-GATED-INFER-DIMS` (`_infer_dims` has no gated rule; no
gated row admissible today); `AUTOTUNE-KERNEL-IDENTITY-PAGED-KV`; host launch
code in runtime libraries behind tessera-opt images stays outside
`kernel_code.v2`.

Evidence: `benchmarks/baselines/autotune_corpus_rerecord_20260927/`,
`benchmarks/baselines/autotune_corpus_rerecord_sm120_20260927/`,
`tests/unit/test_autotune_emitted_identity.py`,
`tests/unit/test_autotune_toolchain_key.py`.

<!-- entry-fields:end -->

Cache coherence (Codex review P2 on PR #861, same day). The identity named the
code a lane *would* compile, but several runtime caches held compiled code
under keys that did not change with it (`_mma_{fused,attn,gated}_*fn_cache`
by storage/epilogue/raster, `_resident_ops_artifact` once per process, the
PTX-registered set by entry name, libraries loaded once and digested later,
checked-in CPU libraries compiled once while the identity re-read the file),
so an in-process code change could time one kernel under another's stamp.
Now every such cache keys on the code (`kernel_cache.cache_key` + the compile
line; `register_compiler(build_line=)` folds arch/compiler into
`kernel_cache.build`'s store key), or the stamp is taken from the loaded
artifact (registered PTX text; `toolchain_identity.load_library` pins; the
bytes a checked-in compile read, now incl. quoted local headers);
`measured_arbitrate` leaves unstamped a candidate whose identity moved during
the race; `source_identity`/`composite_identity` refuse field collisions.
Full per-cache inventory: `docs/audit/backend/nvidia/todo.md` (same key).
Verified: Mac sweep 20698 passed / 3840 skipped / 0 failed; sm_120 device gate
1122 passed / 1 skipped both passes, serve check byte-identical (96/96 match,
15 served, 96/96 miss), real-device recompile probes on sm_120 and gfx1151;
gfx1151 serve check 8/8 served, 8/8 miss. No identity of unchanged code moved.
Open: `AUTOTUNE-KERNEL-IDENTITY-MEMO` (ROCm queue). Tests:
`tests/unit/test_autotune_identity_cache_coherence.py`.

Follow-ups before merge (same day, same key). **`AUTOTUNE-KERNEL-IDENTITY-MEMO`
closed:** `kernel_code_identity.compiler_kernel_identity` memoized a
`tessera-opt` image's identity by selectors + `tessera-opt` digest while the
hsaco caches key on the directive text, so an in-process directive-generator
change launched a new image under the old identity; it now consults the
launch's own build path on every lookup and reuses the memo only for a
byte-identical image (cost moves to the served-verdict check: Mac
`rocm_wmma_gemm` 4.9 -> 24.3 us, `rocm_flash_attn` 1.9 -> 11.4 us per lookup).
**Hot path:** keying caches by source had made every emitted-lane launch
re-run its Python emitter (~10 us); `emit/source_memo.py` memoizes the source
against the emitter function object and every global it reaches by name (a
patch, reload or patched helper re-emits; env-reading, stateful or opted-out
emitters and mutable arguments are never memoized; numbers keyed with their
type), and launch and identity read the same memoized object. Mac per-call
lookup: `_mma_fused_fn` 10.60 -> 2.42 us, `_mma_attn_fn` 9.43 -> 1.78,
`_mma_gated_fn` 10.52 -> 2.32, `_resident_ops_lib` 3.66 -> 1.38, generic
`kernel_cache.build` 10.50 -> 4.19, `nvidia_mma_fused.artifact_identity`
16.91 -> 3.51. Verified at `946530a8`: Mac sweep 20794 passed / 3840 skipped /
0 failed; sm_120 device gate 1122 passed / 1 skipped both passes, serve check
byte-identical (96/96 match, 15 served, 96/96 miss), real-device memo probe;
gfx1151 real-device directive-change probe (fresh image, identity names it,
correct results), serve check 8/8 served / 8/8 miss. No identity of unchanged
code moved. Tests: `tests/unit/test_autotune_identity_memo_coherence.py`
(the directive cases fail on `dedae4b0`). Detail: ROCm and NVIDIA queues.

### 2026-09-27 — Spectral image survives a stale HIP error; streaming STFT names the chip that ran

Owner: [TSOL-POLICY-PHYS-1](INTEGRATED_COMPILER_PLAN.md#tsol-policy-phys-1)

PRs: branch `claude/spectral-stale-hip-error` (sync `SPECTRAL-STALE-HIP-ERROR-2026-09-27`).

Outcome: the ROCm spectral image no longer fails a correct launch because of an earlier, unrelated HIP failure on the thread. HIP's last-error slot is per thread and sticky, and the image's post-launch `hipGetLastError()` checks read it; each exported host-pointer entry that performs device work now discards errors older than the call once, on entry, and never between launches, so grouped checks still see every launch of the call. The streaming STFT's `target="rocm"` architecture is read from the loaded image's stamp instead of the constant `gfx1151`, and fails closed without a ready TSOL profile. Device-proven with a primed slot on gfx1201 (Tajasarus) and gfx1151 (Princess-Luna): the pre-fix image fails with the probe's `rc=246`, the fixed image passes. Extended on the same branch to the sibling hand-written hooks: the sm_120 `libtessera_nvidia_fft.so` (spectral policy + FFT) and `libtessera_nvidia_rng.so` entries, the `tessera_gpu_backend` mbarrier/TMA smoke entries, and the gfx1151 timing probe `collect_rocm_timing_probe` follow the same once-first rule; reproduced and fixed on the RTX 5070 (pre-fix libraries fail every checked path with their own launch-check code) and on gfx1151 (a primed slot voided a valid timing sample).

Remaining: the emitted CUDA templates in `emit/nvidia_cuda.py` keep the unguarded pattern; now that #861 has merged (an emitted lane's autotune identity is the digest of its emitted source), apply the rule there together with an sm_120 corpus re-record (NVIDIA queue). `test_tma_smoke` fails on the RTX 5070 before and after this change (`cuTensorMapEncodeTiled: invalid argument`, pre-existing, not investigated).

Evidence: `tests/unit/test_rocm_spectral_stale_hip_error.py`, `tests/device/nvidia/test_spectral_stale_cuda_error.py`, `tests/unit/test_spectral_streaming.py`, ROCm and NVIDIA queue entries `SPECTRAL-STALE-HIP-ERROR-2026-09-27` (probe output, sweep source, per-entry classification, per-host counts).

<!-- entry-fields:end -->

sm_120 extension, verification (The-Super-Bear RTX 5070, own worktree and `build-nvidia-cuda`, loaded `.so` paths checked; code commit `31209ae8`): the new regression test 9 failed against the pre-fix libraries (`rc=3` host/device FFT and Philox, `rc=292` DCT, `rc=306` STFT and streaming STFT, `rc=365` STFT JVP) and 9 passed fixed; `cudaGetLastError` call sites in the images 23→37 (fft) and 4→8 (rng), matching the 14 + 4 entry clears. NVIDIA spectral/FFT/RNG device files 102 passed; `tests/unit -m "not slow" -k "spectral or stft or fft or dct or rng or philox or dropout"` 707 passed / 221 skipped. Release gate device layer (`scripts/run_nvidia_release_gate.sh --layer device` at `31209ae8`, reports `~/gate-reports/stale-nv-31209ae8` on that box): both passes 1132 tests, 1130 passed, 2 skipped (NCCL not installed; `libtessera_runtime.a` not built), 0 failed, the 9 new tests included; `status=success`. The timing probe on Princess-Luna (gfx1151): pre-fix primed → `HIP instrumented launch failed`, fixed primed → valid sample; Tajasarus (gfx1201) builds it but fails closed before any launch (`requires exact gfx1151`), so it cannot evaluate that lane.


### 2026-09-27 — Evidence governance gates: reason vocabularies, ODS consumers, corpus eligibility

Owner: [X86-EVIDENCE-VOCAB-1](INTEGRATED_COMPILER_PLAN.md#x86-evidence-vocab-1)

PRs: branch `claude/evidence-governance-gates`.
Sync: `EVIDENCE-GOVERNANCE-GATES-2026-09-27`.

Outcome: Three host-free governance gates. **Reason vocabularies:** the x86
packet's eleven promotion-ineligibility tags (re-counted from the producer: still
eleven) and four sibling families with the same shape -- the ROCm profiler
packet, the NVIDIA device-clock packet, the x86 PMU event map and the
calibration corpus -- are each declared once through
`evidence_reasons.ReasonVocabulary`, with a meaning per tag; validators refuse
an undeclared tag, and the event-map validator, which only type-checked its
reasons, now re-derives them. Registered diagnostics a packet also carries are
borrowed by `pass_origin`, never redeclared; no tag was added to
`diagnostic_codes.py`. **ODS consumers:** the 2026-09-20 gate parsed 286 of 623
op records, passed fixture-only ops and matched bare substrings; the rebuilt
scan (`ods_consumer_audit.py`, cross-checked once by hand against
`llvm-tblgen --dump-json`, refusing constructs it cannot read) waives 84 ops at
landing -- 38 fixture-only, 46 unreferenced -- each with a reason; none meets
#29a, none was deleted. The pre-PR review found three fail-open holes (a
`using mlir::func::FuncOp` vouching for `tessera_nvidia.func`, prose in a
`reason=` string, an op's own dialect arity table) that had passed seven ops;
each is closed and pinned by a synthetic test.
Seven `tessera.neighbors.*` names are declared by two records. **Calibration
corpus (EVIDENCE-PACKET-1 slice):** `apply_corpus` read `selector_eligible` with a
default of `True` and never read `ineligibility_reasons`; it and
`load_pruning_corpus` now refuse missing, contradicted or undeclared eligibility
evidence by three registered codes, and the committed 2026-08-15 corpus still
reads and still cannot promote.

Remaining: the 77 waived ops need consume-or-delete decisions per dialect
owner, and the `tessera.neighbors` double declaration needs one authority;
EVIDENCE-PACKET-1's shared evidence envelope, GA/EBM route receipts, public-
frontend AD pairing, asynchronous attribution and clean performance admission
stay open (plan gate text). `profiler_cuda_window` reasons are prose, not tags,
and were not changed.

Evidence: `tests/unit/test_x86_evidence_vocabulary.py`,
`tests/unit/test_ods_op_has_consumer.py`,
`tests/unit/test_calibration_corpus_eligibility.py`,
`python/tessera/compiler/evidence_reasons.py`,
`python/tessera/compiler/ods_consumer_audit.py`; host: Mac (M1 Max, macOS 27),
no device lane involved.

<!-- entry-fields:end -->

Additional owners: [GOV-ODS-CONSUMER-1](INTEGRATED_COMPILER_PLAN.md#gov-ods-consumer-1),
[EVIDENCE-PACKET-1](INTEGRATED_COMPILER_PLAN.md#evidence-packet-1).

### 2026-09-27 — sm_120 autotune follow-ups

Owner: [W5.2](INTEGRATED_COMPILER_PLAN.md#w52)

PRs: branch `claude/sm120-autotune-followups`.
Sync: `SM120-AUTOTUNE-FOLLOWUPS-2026-09-27`.

Outcome: three follow-ups from the emitted-identity re-record landed together,
since each invalidates or extends the same sm_120 rows.

1. The emitted CUDA templates (`emit/nvidia_cuda.py`) now follow the
   stale-error rule of `SPECTRAL-STALE-HIP-ERROR-2026-09-27`: in each source
   that reads the last-error slot, every exported device-work entry clears it
   once, first. The raced lanes (generic fused, scalar attention, gated,
   pointwise, mma.sync fused/attention/gated, host and timer entries) also
   check each launch group through the slot, as the ROCm generic lane does. On
   this box `cudaDeviceSynchronize` returned success after an
   invalid-configuration launch, so before this change such a launch was
   reported, and could be timed, as a kernel.
2. `nvidia_generic_cuda`, `nvidia_flash_attn` and `nvidia_gated` have a
   CUDA-event `_device_ms` timer.
3. `_infer_dims` has a `gated_matmul` rule.

The 96 sm_120 registry rows were re-recorded on The-Super-Bear (clean worktree
at `31a58bd4`, fresh trees, recorder + finalizer under the timing lock). No
row now races an untimed candidate (was 20), and 38 rows are
selector-eligible (was 31). 13 rows are served (was 15): the two dropped rows
keep their winner, but their re-measured margin fell under the noise. All 96
miss after an emitter change with pins unchanged. 10 winners changed, none of
them a served row. The gfx1151 and serving rows are byte-identical.

Remaining: the 20 formerly partial rows are still not served. 18 have winners
with no route-resource fingerprint (`selector_eligible: false`), and 2 are
unseparated. The sync-only emitted sources keep unchecked launches (listed in
the NVIDIA queue).

Evidence: `benchmarks/baselines/autotune_corpus_rerecord_sm120_followups_20260927/`,
`tests/unit/test_nvidia_emitted_stale_error_rule.py`,
`tests/unit/test_autotune_gated_infer_dims.py`,
`tests/device/nvidia/test_emitted_stale_cuda_error.py`.

<!-- entry-fields:end -->

Why the device proof primes through a private symbol. Each emitted `.so` links
cudart statically, and its cudart symbols are local, so the slot these entries
read belongs to that library alone. Priming the process's shared `libcudart`
would not touch it. The test therefore resolves the library's own local
`cudaSetDevice` / `cudaPeekAtLastError` from its symbol table and load base.
Each lane runs a negative control first: with the clears stripped, the primed
slot must fail the lane. The two mma.sync attention entries are the recorded
exception. A successful `cudaFuncSetAttribute` resets the slot (measured on
sm_120), which masks them; they clear anyway, because the rule must not rest on
undocumented behaviour.

### 2026-09-27 — ROCM-FP8-BLOCKSCALE-1: an LDS-staged multi-wave W8A8 body beats AITER at large M

Owner: [ROCM-FP8-BLOCKSCALE-1](INTEGRATED_COMPILER_PLAN.md#rocm-fp8-blockscale-1)

PRs: branch `claude/gfx1201-perf-w8a8-mxfp4` (sync `GFX1201-PERF-2026-09-27`).

Outcome: the W8A8 block-scale contract gets a second physical body on the typed route: eight waves of 32 rows share one LDS-staged K slab of A and the `[N, K]` weight, with the register body's isolated scale-group semantics unchanged (zero partial per group, one `tile.fragment_scaled_accumulate` join), bit-identical to the register panel on device. Graph→Schedule selects it (`staging = "lds"` on `schedule.matmul`, in the digest when set): 128x128 once that tiling gives >= 64 workgroups on a whole-128 M, else 128x64 when that covers the 64 CUs, else the register panel; Target IR states `staging`/`warps`/`pipeline_depth`, from which the package binds its 256-thread workgroup. The contract also admits a bf16 Graph result (one RNE rounding in the typed store, distinct package ABIs). Against unmodified AITER `gemm_a8w8_blockscale`, device clock, paired: `[N, K]` / AITER geomean **0.91 at M >= 1024** (was 1.29; 12/18 faster), **0.90 at M = 256** (was 1.08), 0.65 at M <= 64 (unchanged). The bf16 store is timing-neutral (bf16/f32 geomean 0.998).

Remaining: 6 of 18 M >= 1024 shapes stay 1.04-1.08x behind AITER (K <= 2048, or N = 1024); ragged M is 1.075x AITER geomean over 20 points, 1.37-1.56x at N = 24576, K = 1536 (a 128-row tile and a masked edge that costs 13 VGPRs); `[K, N]` stays on the register panel; AITER's split-K buckets remain unmeasured.

Evidence: [LDS-body packet](../../../benchmarks/baselines/gfx1201_fp8_blockscale_lds_20260927/README.md), `tests/device/rocm/test_fp8_blockscale_w8a8.py`, `tests/unit/test_rocm_fp8_blockscale.py`, `tests/tessera-ir/phase2/e2e_fp8_blockscale_lds_rocm_target.mlir`.

<!-- entry-fields:end -->

What did not survive measurement (all in `knobs.json` / `ragged.json`, bit-identical results): double-buffered LDS (1.22-1.48x; 128x128 needs 72 KiB and is refused), a register-staged next slab (1.00-1.24x; spills at 128x128), a 64-byte slab (1.10-1.12x), 0 or 32 bytes of row padding (3.4-4.5x / 1.20-1.26x), 4- and 16-wave grids, grouped raster (neutral), and zero-filling the rows past M (clamping is 0.99x). The full-fence `gpu.barrier` put a `global_inv` after every wait; the body now fences LDS only, which measured neutral. A bf16 output was expected to close the K <= 2048 gap by halving the store bytes; it did not, so the remaining gap there is not the output traffic.

Correction, same day: the first assertions-tree run of the final code reported 5 W8A8 failures and a lit failure. The `tessera-opt` it used was two commits stale (the generator's staleness warning fired); rebuilt, the recorded run is green on both trees.

### 2026-09-27 — ROCM-MXFP4-W4A8-1: a per-wave M guard closes most of the one-row-block gap

Owner: [ROCM-MXFP4-W4A8-1](INTEGRATED_COMPILER_PLAN.md#rocm-mxfp4-w4a8-1)

PRs: branch `claude/gfx1201-perf-w8a8-mxfp4` (sync `GFX1201-PERF-2026-09-27`).

Outcome: the folded prefill's load schedule gains a fifth Target-IR performance key, `row_guard` (`cta` | `wave`), required by the folded materializer. `wave` skips the WMMAs and the epilogue of a wave whose 64 rows all lie past M (it still stages and meets every barrier) and tests the vector epilogue's completeness per wave; Tile→ROCm selects it only when M is not a whole number of BM256 row blocks, so whole-row-block kernels are byte-identical. Output bitwise equal to exact K32. On Tajasarus, device clock witnessed by HIP events, three processes: M = 128 goes 0.62-0.66x of the original schedule and **0.74-0.77x Radiance at N = 5120** (was 1.10-1.11x), **1.04-1.12x at N = 17408** (was 1.24-1.27x). Diagnostic probes ruled out the 64-bit staging address arithmetic (`sgpr_base`, 0.99-1.00x) and the K16 scheduling barrier (`sched0`, 1.00-1.01x) as the gap.

Remaining: M = 256 (one full row block, no idle wave) stays 1.05-1.23x behind Radiance and is unattributed (no counters on WSL2); Radiance's packed E2M1 weights read half the bytes of the expanded E4M3 ones, untested as the cause. Exact K32 stays default; folded stays opt-in.

Evidence: [one-row-block packet](../../../benchmarks/baselines/gfx1201_mxfp4_small_m_20260927/README.md), `tests/device/rocm/test_mxfp4_folded_prefill.py`, `tests/unit/test_rocm_mxfp4_folded_schedule.py`, `tests/tessera-ir/phase2/e2e_folded_mxfp4_rocm_load_schedule.mlir`.

<!-- entry-fields:end -->

### 2026-09-27 — The x86 f32 GEMM packs B itself: alignment no longer sets its speed

Owner: [EVIDENCE-PACKET-1](INTEGRATED_COMPILER_PLAN.md#evidence-packet-1)

PRs: branch `claude/x86-gemm-align`.
Sync: `X86-GEMM-ALIGN-2026-09-27` (closes `X86-GEMM-ALIGN-1`).

Outcome: `tessera_x86_avx512_gemm_f32` read B with one unaligned 64-byte load per
FMA, so any caller whose B was not 64-byte aligned (numpy guarantees 16) ran it
~1.5x slower at 256³ (`X86-MATMUL-BIMODAL-1`). The recorder had been aligned; the
production path had not.

What the kernel does now:
- **Two paths.** The packed path copies blocks of up to 8 strips × 512 rows of B into
  the kernel's own 64-byte-aligned, L2-resident panel and runs eight accumulators
  over it; the sum continues through C between K blocks. The direct path runs the
  same loop over B in place.
- **Measured rule:** direct iff M == 1, or M ≤ 4 with B ≤ 1 MiB. It comes from a
  paired TSC-witness crossover on Princess-Luna (M ∈ 1..16, B from 16 KiB to 4 MiB,
  B%64 ∈ {0,16}). It replaced "packed for every M > 1" after Codex review showed that
  rule regressed aligned small-M calls.
- **Unrolled strip loops.** Measuring the rule also exposed that GCC 15 `-O2` left
  the strip loops rolled with the accumulators on the stack; they are now fully
  unrolled.
- **Overlap.** An overlapping C (with A or B) is detected at entry and computed
  through scratch.
- **Why the kernel, not `runtime.launch`:** the matmul-family lane and
  `TileToX86Pass`'s `func.call` reach the symbol directly.

Numerics: results are bitwise identical to the pre-fix kernel on both paths at every
alignment. The oracle is declared in `test_gemm_f32.cpp` and checked with `memcmp`
at every 4-byte B offset; a K-block mutation fails it. The path rule and overlap
cases are pinned by tests.

Measurement: a paired interleaved before/after probe (both builds in one process,
TSC witness, timing lock, production package asserted to embed the timed library).
Princess-Luna ran 17 shapes and Tajasarus 7:
- On the packed path, the best-process alignment effect fell from 1.14–3.34x to
  ≤ 1.04x.
- The direct path keeps 1.07–1.43x.
- No shape is slower than before at any alignment: 0.04–0.77x above 10 µs, and
  0.87–1.05 on the ~4.5 µs ctypes floor.
- 256³ went to 0.42x of before when aligned and 0.28x when misaligned.

Packets: both AVX-512 E2E packets were re-recorded twice at `86ec9d31`. Matmul 256³
went from ~0.73 / ~0.70 ms to ~0.30 / ~0.29 ms. Princess-Luna attention is +6%
(recording drift, unchanged source); everything else is within 4%.

Remaining: the direct path's residual alignment effect; the rule is measured on
Princess-Luna only; `_tiled` and the bf16 / f64 / u8s8 GEMMs are unchanged (the
latter's sensitivity is unmeasured).

Evidence: `benchmarks/baselines/x86_gemm_align_20260927/`,
`docs/audit/evidence/e2e_spine/x86/x86_64_avx512_{strix_halo,granite_ridge}/`,
`src/compiler/codegen/tessera_x86_backend/tests/test_gemm_f32.cpp`,
`tests/unit/test_x86_matmul_family_compiled.py`.

<!-- entry-fields:end -->

### 2026-09-27 — E2E-REAL-6: gfx1151 softmax and reduction retire their Graph-owned constructors

Owner: [E2E-REAL-6](INTEGRATED_COMPILER_PLAN.md#e2e-real-6)

PRs: branch `claude/e2e-real-6-rocm-unary`.
Sync: `E2E-REAL-6-rocm-unary-2026-09-27`.

Outcome: `rocm_native.package_softmax` / `package_reduction` no longer read the
Python Graph object to author Tile IR text. Both lower through
`scheduled_kernel.lower_scheduled_kernel(target="rocm_gfx1151")` (tessera-opt
Graph -> Schedule -> Tile), and `package_scheduled_kernel` replays the Schedule
record and projects the descriptor from native IR. The Schedule contract now
carries the envelope the retired constructors served on gfx1151: f16/f32
softmax (including `softmax_safe`, canonicalized to the one semantic it is),
f16/bf16/f32 sum/mean/max with f32 output, and keepdims
(`PMPasses.cpp::getSemanticKernelSchedule`, Python `_graph_contract`,
`native_unary_contract.verify_unary_projection`). The ROCm consumer selects the
storage-keyed ABI and carries `nan_mode` into the descriptor (#21a/#32).
gfx1201 keeps its proved f32 rank-reducing envelope in all three layers.
The retired constructors are frozen in `tests/_support/rocm_unary_baseline.py`
as the declared oracle Decision #31(a) allows; nothing in `python/` calls them.

Why this family: MASTER_AUDIT §1 already named ROCm softmax/reduction first in
the bootstrap absorption order; the compiled consumer existed for both chips
(f32 since 2026-08-05); the Graph constructor was still the production caller
for every narrow-storage or keepdims request; and both routes compile through
the same `_compile_tile_ir`/`_compile_reduction_tile_ir` into images that can be
run side by side. Paged-KV and MoE have no ROCm Schedule producer yet, and the
NVIDIA gap families need new Schedule contracts, so neither is a one-PR move.

Found on the way: the retired constructor derived the combiner from the op
*name*, so `tessera.reduce {kind = "max"}` (what `ts.reduce(x, op="max")`
traces to) was packaged and executed as a **sum**. The compiled route reads
`kind`; `min`/`prod` are refused instead of summed. Kept verbatim in the
baseline as evidence, with a device test. The same review listed every other
case the retired contract admitted and the compiled one refuses, each on
purpose and each pinned by a test: an integer `keepdims`, a combiner-less
`tessera.reduce` (summed before), a reduction `schedule` hint on softmax, and a
softmax whose declared output shape differs from its input (the retired route
guarded the output with the input shape). `nan_mode` in the descriptor is now
read from the replayed Tile op, not written as a literal. A knock-on: the native
JVP reduce child (`native_jvp_plugins._scheduled_reduce_step`) now accepts
keepdims on gfx1151 through the same contract -- same device-proven kernel, not
separately device-tested through the JVP entry.

Measured cost (Princess-Luna, not a promotion): the HSACO is shape-invariant on
both routes, but the compiled route's cache key is its Tile text, which binds
the shape (through the Schedule digest and constants) and the Graph function
name, so each new shape or function name is a cold compile (~370 ms) where the
retired route hit its cache (~186 ms). The f32 envelope has had this since 2026-08-05.
`benchmarks/rocm/measure_rocm_unary_route_cache.py` reproduces it.

Remaining: E2E-REAL-6 still owns ROCm paged-KV, MoE dispatch and forward
attention's Graph-owned constructors, the NVIDIA and Apple gap families, the
x86 cohort/elementwise/breadth constructors and the frontend (`_OpExtractor`).
Follow-ups this cut exposed: key the ROCm image cache on the shape-free kernel;
gfx1201 narrow/keepdims unary rows (device proof on Tajasarus); NVIDIA and x86
classify `softmax_safe` as a native softmax their scheduled packagers refuse;
`numeric_policy` keyword arguments are still ignored by every target's unary
contract (pre-existing, unchanged).

Evidence: `tests/unit/test_rocm_unary_migration.py` -- host-free differential
over the retired envelope (28 softmax + 150 reduction cases: ABI, buffers,
scalars, shape guards, geometry, semantic provenance, Tile-kernel attributes),
refusal parity and the gfx1201 boundary; on Princess-Luna (gfx1151,
`TESSERA_ROCM_E2E_DEVICE_TEST=1`) all 375 pass with no skips, including 179
device rows in which the retired and compiled images agree bit-for-bit and
match an oracle. On Tajasarus (gfx1201, assertions-ON LLVM/MLIR 23.1.1) the
host-free rows and every gfx1201 scheduled device row pass (310 passed; the
skips are the gfx1151 device rows and Darwin-only rows). Route census:
`scripts/record_package_route_census.py` differs from main only in the ROCm
unary rows; `bootstrap_prune_gap.md` moves `rocm_gfx1151` softmax/reduction from
gap to generic.

<!-- entry-fields:end -->

### 2026-09-27 — EVIDENCE-PACKET-1: shared evidence envelope, GA and EBM route receipts

Owner: [EVIDENCE-PACKET-1](INTEGRATED_COMPILER_PLAN.md#evidence-packet-1)

PRs: branch `claude/evidence-packet-1-envelope`.
Sync: `EVIDENCE-PACKET-1-2026-09-27`.

Outcome: two slices. **Shared envelope** (`compiler/evidence_envelope.py`):
`read_evidence_packet` dispatches on `schema` to one of three registered
families (x86 Zen 5 profiler v1/v2, ROCm gfx1151/gfx1201 profiler, NVIDIA
sm_120 device clock) and runs the family validator unchanged. It then projects
the packet onto one envelope: artifact identity (image, ISA and semantic
digests, the timing sample's digests, x86 per-row images), compiler identity
where a family records it, timing domain, clock validity, execution
environment, source commit and worktree state, sample ids, route, eligibility
and refusal causes. A missing or malformed field is refused, never defaulted
(`EVIDENCE_ENVELOPE_INCOMPLETE`). The envelope also enforces invariants no
family may waive (`EVIDENCE_ENVELOPE_CONTRADICTED`): a packet is eligible
exactly when it names no refusal cause (the x86 benchmark's own retain/reject
verdict counts as a cause), and an eligible packet has a valid clock, a clean
tree, a sample id, and a measured image its timing sample names. The last
rule was checked only on the ROCm device-clock route before; now it applies
on every route. An unregistered schema refuses (`EVIDENCE_ENVELOPE_SCHEMA_UNKNOWN`).

Two fail-open derivations were closed at the family, not only in the
envelope. `profiler_rocm_evidence` read `if source.get("worktree_dirty")`, and
`profiler_x86_evidence` read `environment.get("virtualized"/"wsl"/"worktree_dirty")`,
so a packet that omitted a field (or stored the string `"false"`) derived no
`SOURCE_WORKTREE_DIRTY` / `VIRTUALIZED_HOST` / `WSL_CLOCK_DOMAIN`. Both now
require bools. NVIDIA already refused (`is not False`). The SSD admission loops
(`ssd_performance`, ROCm and NVIDIA device-clock routes) read calibrations
through the envelope, pinned to their family. A drift test refuses any
production module that calls a family validator directly. Committed evidence:
188 packets found under `benchmarks/`; 186 read. Those are 36 NVIDIA, 75
gfx1151, 73 gfx1201 and 2 x86 packets; 149 are promotable. The 37 retained are
36 gfx1201 packets (`INSTRUMENTATION_OVERHEAD_EXCEEDED`) and the 2026-08-06 x86
packet (`VIRTUALIZED_HOST`, ..., `benchmark_verdict=retain`). The 2 refused
are the pre-existing `DEVICE_CLOCK_WINDOW_TOO_SHORT` pair. No committed packet
changed state.

**GA/EBM route receipts** (`tessera/_route_receipts.py`): each of the 32
`_try_<target>_*` native-lane helpers in `tessera.ga` and `tessera.ebm` is
`@native_attempt` (target from its name: `apple_gpu_runtime`, `x86_avx512`,
`rocm`, `cuda`), and each of the 29 public primitives that reaches one is
`@public_route`. Inside `capture_route_receipts()` every public call leaves a
receipt naming its route. A native dispatch outside any receipt frame (an
orphan), or a capture with no receipts, makes the span `unattributed`
(`ROUTE_RECEIPT_ORPHAN_DISPATCH`, `ROUTE_RECEIPT_EMPTY`). `clifford_core`,
`energy_core` and `visual_complex_core` rows now carry `route` and
`route_receipts` over the timed span, and derive `device` from them. The
jit_bridge trace had covered only the Apple manifest lane. A receipt run on
Princess-Luna shows why that mattered: `ebm.energy_quadratic` and
`partition_exact_from_energies` ran on the **x86 AVX-512** kernel, which the
old trace would have reported as no native dispatch.

Device receipts, all at clean `eed48b9b`
(`benchmarks/baselines/ga_ebm_route_receipts_20260927/`):

- **Mac M1 Max:** the GA primitives ran on the Apple GPU runtime.
  `geometric_product` split between that runtime and the reference, and
  `rotor_sandwich` is `mixed`. EBM `langevin_step` and the partition ran on
  the Apple GPU; `energy_quadratic` ran on the reference.
- **Princess-Luna and Tajasarus (Zen 5):** EBM energy and partition ran on
  x86 AVX-512. `langevin_step` and all GA ran on the reference.
- **The-Super-Bear (Zen 2):** everything ran on the reference.
- **No host:** no composition reached a ROCm or CUDA GPU lane. The x86 lane
  precedes ROCm in those primitives, and the native Langevin/Clifford GPU
  routes are separate entry points these suites do not call.

Receipts attribute routes. They are not timings, and every row stays
non-promotable. The pre-PR full sweep caught one defect in the first
labelling. The labels were dotted (`tessera.ebm.inner_step`), and the ODS
consumer audit read them as compiler consumers of six EBM ODS ops. Labels are
now `ebm:inner_step`, dotted labels are refused, and the receipts were
re-recorded. The Codex review of PR #869 found two more defects, both now fixed with tests. First, the CLI filtered out a top-level packet whose schema is unregistered, then exited 0 having read nothing; it now refuses with `EVIDENCE_ENVELOPE_SCHEMA_UNKNOWN`, and so does an input that contains no packet at all. Second, nested receipt captures were closed with `list.remove`, which matches by dataclass equality, so an inner capture could remove an equal outer one; captures now close by identity, in stack order. Each backend outcome is recorded in `docs/audit/backend/{apple,nvidia,rocm,x86}/todo.md` under the sync key.

Remaining: math probes' manufactured `RuntimeArtifact.metadata`, paired
public-frontend runs for the direct-IR AD probes, asynchronous/multi-thread
attribution, DLOP profiler receipts and clean performance admission (plan
record). Also open: the GPU families record no compiler build identity, and
the CUDA activity-window calibration, the calibration corpus and E2E-spine
packets are not yet envelope families. The image-binding rule means a future
bare-metal x86 packet on the profiler route can promote only if its rows also
carry timing witnesses that name their images. No such packet exists, since
every fleet host is WSL2.

Evidence: `tests/unit/test_evidence_envelope.py`,
`tests/unit/test_route_receipts.py`, `tests/unit/test_ssd_comparison.py`,
`benchmarks/baselines/ga_ebm_route_receipts_20260927/`; hosts: Mac (M1 Max,
macOS 27) for the envelope and the unit gates, and one receipt record each from
Mac, Princess-Luna, Tajasarus and The-Super-Bear.

<!-- entry-fields:end -->

### 2026-09-27 — Autotune launch integrity

Owner: [W5.2](INTEGRATED_COMPILER_PLAN.md#w52)

PRs: branch `claude/autotune-launch-integrity`.
Sync: `AUTOTUNE-LAUNCH-INTEGRITY-2026-09-27`.

Outcome: four items that each move the same corpus rows landed together.

1. `NVIDIA-EMITTED-UNCHECKED-LAUNCH`: every emitted CUDA source (55 sync-only
   entries and two inline sources, now emitters) and the three sync-only HIP
   entries (paged-KV gather, direct paged attention, ReplaySSM `su`) read the
   last-error slot after each launch group and consume every
   allocation/copy/memset/event status. The host-independent gates reject a
   sync-only entry. With every launch made invalid, the sync-only judgment
   reported success on sm_120 (17/17 lanes), gfx1151 and gfx1201 (3/3 each);
   the fixed entries fail.
2. `AUTOTUNE-SM120-ROUTE-RESOURCES`: the 17 routes without Nsight route
   resources were captured one route per report; 91 registry rows are
   selector-eligible (was 38).
3. The shipped `libtessera_nvidia_gemm` was not byte-reproducible because its
   DT_RUNPATH recorded how CMake discovered CUDA; it now builds without one and
   four builds across two worktrees and both configures are identical.
4. `AUTOTUNE-KERNEL-IDENTITY-PAGED-KV`: `autotune.RouteIdentity` gives the
   non-registry rows the registry's Decision #11 contract; both paged-KV warm
   starts refuse a changed route.

Re-records: 108 sm_120 rows on The-Super-Bear (20 registry rows served, was
13; all 108 miss after an emitter change with pins unchanged) and the 8 gfx1151
paged-KV rows on Princess-Luna (winners unchanged, 3 served in two build
trees). No row was backfilled.

Remaining: 15 of the 20 formerly partial sm_120 rows are unseparated and 1 is
unstable; an `ncu`-only exit abort of processes holding a generic-lane library
is recorded, not root-caused.

Evidence: `benchmarks/baselines/autotune_corpus_rerecord_sm120_launch_integrity_20260927/`,
`benchmarks/baselines/autotune_corpus_rerecord_gfx1151_paged_kv_20260927/`,
`tests/unit/test_nvidia_emitted_stale_error_rule.py`,
`tests/unit/test_rocm_emitted_launch_rule.py`,
`tests/unit/test_autotune_route_identity.py`,
`tests/device/nvidia/test_emitted_unchecked_launch.py`,
`tests/device/rocm/test_emitted_unchecked_launch_hip.py`.

<!-- entry-fields:end -->

Additional owners: the NVIDIA and ROCm backend queues (same sync key).

### 2026-09-27 — Small correctness gaps: sm_120 TMA smoke, a lit runner that runs, one neighbors authority

Owner: [GOV-ODS-CONSUMER-1](INTEGRATED_COMPILER_PLAN.md#gov-ods-consumer-1)

PRs: branch `claude/small-correctness-gaps`.
Sync: `SMALL-CORRECTNESS-GAPS-2026-09-27`.

Outcome: Three small defects, each root-caused on the host that shows it.
**TMA smoke (sm_120):** two stacked defects. Driver 610.88 rejects
`cuTensorMapEncodeTiled` when a rank-1 map's `globalStrides` is `nullptr`
(zero meaningful entries; the contents are ignored -- 128, 0, 7 and 2^41 all
encode), and once that was fixed the launch faulted because the by-value
`CUtensorMap` lacked `__grid_constant__`, so nvcc copied it to local memory
and TMA was handed a local address (visible in the PTX; the fault goes away
with exactly that change). The smoke now passes on the RTX 5070.
**Lit runner:** on Tajasarus every lit selector picked an LLVM-prefix wrapper
that cannot `import lit`, so `check-tessera-ir` ran zero fixtures;
`cmake/TesseraLit.cmake` now selects, for every suite, the first candidate
whose `--version` runs, warns per rejected candidate, fails configure for
`tests/` and fails (not skips) the backend check targets when none works.
Two resolver ratchets that failed only when `TESSERA_BUILD_DIR` was exported
now clear every selector the resolver reads, gated against the resolver's
source. **Neighbors:** the seven `tessera.neighbors.*` names declared three
times (core `TesseraOps.td`, an unbuilt `tessera_neighbors.td`, a hand-written
C++ dialect) now have one authority, `TesseraOps.td` -- the only one the parser
ever reached, since `tessera.neighbors.x` resolves by its first segment (a
taps/coeffs mismatch parsed clean on the pre-change `tessera-opt`, showing the
hand-written verifier never ran). The semantics only the dead copies stated
moved into the core verifier: stencil.define tap/coefficient well-formedness
(before, checked only inside `-tessera-stencil-lower`) and neighbor.read's
required `delta`. The duplicate ratchet's baseline is empty and a new gate
rejects hand-written C++ ops that shadow an ODS name.

Remaining: `test_tma_smoke` has no ctest/pytest wrapper, so no lane runs it.
The halo.exchange `!tessera.neighbors.halo`-only operand check of the deleted
dialect was not ported: `-tessera-halo-mesh-integration` legitimately builds
exchanges over tensors. Array-of-integer stencil taps, which
`-tessera-halo-infer` read but `-tessera-stencil-loop-materialize` silently
skipped, are now rejected at parse (one unit-test input moved to dense taps).
The two broken lit wrappers remain in Tajasarus's toolchain prefixes (reported,
not used). The 77 waived ops' consume-or-delete decisions stay open.

Evidence: The-Super-Bear (RTX 5070, CUDA 13.4.59 / driver 610.88, own worktree,
device runs under `flock /tmp/tessera-timing.lock`): probe matrix and PTX in
the [NVIDIA queue](../backend/nvidia/todo.md); pre-fix binary fails, fixed
`test_tma_smoke` (CMake-built, `sm_120a`) passes 5/5. Tajasarus (own worktree,
`build-wb`, configured like `build/`, under `env.sh`): before, `check-tessera-ir`
/ `check-ebm` / `check-tessera-rocm` die with `ModuleNotFoundError`; after,
`check-tessera-ir` 520 discovered / 454 passed / 66 unsupported,
`check-tessera-rocm` 82/82, ebm 18, clifford 22, spectral 11; resolver ratchets
17/17 under `env.sh`, with `TESSERA_BUILD_DIR` and under `env -i`
([ROCm queue](../backend/rocm/todo.md)). Neighbors: the new negative fixture
fails all nine expected diagnostics on the pre-change binary and passes after;
Mac lit 520 / 471 passed / 49 unsupported; Tajasarus as above, and the same
counts (454 / 66, `check-tessera-rocm` 82/82) from an assertions-ON LLVM 23.1.1
tree built from this branch; Super-Bear reconfigured through the validator
(selects `/usr/lib/llvm-23/bin/lit`), `check-tessera-nvidia` 62/62. Mac full
unit sweep 21306 passed / 3821 skipped / 0 failed (the first sweep caught
`NEIGHBORS_TOPOLOGY_UNKNOWN_KIND` losing its only C++ occurrence; the registry
scan now reads the `.td` constraint that emits it);
`tests/unit/test_ods_op_has_consumer.py`, `test_neighbors_*.py`,
`test_tessera_opt_build.py`, `test_test_suite_architecture.py`.

<!-- entry-fields:end -->

### 2026-09-27 — Latent defects from the ODS triage: TMEM lowering, TMEM planning, solver matching, ZeRO config; dashboards stop over-claiming

Owner: [GOV-ODS-CONSUMER-1](INTEGRATED_COMPILER_PLAN.md#gov-ods-consumer-1)

PRs: branch `claude/tile-latent-defects`.
Sync: `TILE-LATENT-DEFECTS-2026-09-27`.

Outcome: the defects the ODS connection triage recorded in passing are fixed,
each with a fixture that fails on the unfixed code. All are IR-level; TMEM is
datacenter sm_100, which no fleet box has, so nothing here claims execution.

1. `LowerTileToNVIDIA` maps `tile.tmem.allocate/load/store` by op identity.
   Anything else under `tile.tmem.` (including the unregistered legacy
   `tile.tmem.alloc`) fails with `NVIDIA_TMEM_UNKNOWN_OP`; it used to become a
   `tmem_store` contract. The `!tile.tmem` handle lowers to the i32 TMEM
   address, and load results are replaced; before, every op was erased with
   live uses, which aborted the assertions-ON driver ("operation destroyed but
   still has uses"). A handle feeding an unlowered op (`tile.tcgen05.mma`)
   fails with `NVIDIA_TMEM_HANDLE_UNLOWERED`.
2. `LowerNVIDIAToNVVM` refuses (`NVIDIA_MARKER_RESULT_USED`) a void-marker
   contract whose result is used outside the contract family, instead of
   `dropAllUses` leaving a null operand.
3. `TileBufferReuse` / `TileBufferArena` / `TileMemrefLifetime.h` matched the
   unregistered `"tile.tmem.alloc"` marker nothing produces, so no real TMEM
   allocation was planned. They match `tile.tmem.allocate` (`isa<>`), size and
   align it from the op, and never coalesce it: Tile IR carries no TMEM
   completion fact, so no TMEM lifetime is provably disjoint (#30; #10a
   negatives in `tile_buffer_reuse.mlir` / `tile_buffer_arena_tmem_invalid.mlir`).
4. The linalg solver passes matched `contains("solve")` (every
   `tessera_solver.*` op, through the dialect prefix) and `contains("lu")` /
   `contains("factor")` (`gelu`, `relu`, `silu`, `adafactor`). They match
   exact ops now (`linalg_solver_op_identity.mlir`); the four solver ops they
   consume left the ODS waiver (ceiling 84 → 79 with `tile.tmem.store`).
5. `ZeROConfig.to_ir_attr()` emits `tessera_sr.zero_config`, but
   `OptimizerShardPass` read `tessera.num_dp_ranks` / `tessera.dp_axis`,
   which nothing produces, and sharded with its defaults (1 rank, axis "dp").
   It reads the emitted dictionary now and treats stage/axis/rank count as
   semantic keys (#21a): `SR_ZERO_CONFIG_{MISSING,MALFORMED,CONFLICT}`,
   including a count that disagrees with the `tessera.distributed_plan` mesh.

Dashboards (Decision #25/#26), each regenerated through its generator:
`ntk_rope` Tile `fused` → `partial` and Target `device_verified_abi` →
`reference` (the `ntk_rope → rope` audit alias rested on a rewrite that does
not exist); the three AttnRes ops' `lowering_rule` `complete` → `partial`
(registered Graph ops, no lowering); the SM120 differentiation dashboard's
four promoted rows cite fixture-only Target ops, so their Target-IR column is
open and their status is runtime-promoted; `GRAPH_IR_SPEC.md` no longer calls
`cache.page_lookup`, `ring.create` and the DNAS ops "scaffolded lowering".

Remaining: `tile.tcgen05.mma` has no NVIDIA lowering, so a TMEM handle that
feeds it cannot lower; the NVVM stage emits void markers for TMEM contracts;
`OptimizerShardPass` still selects optimizer ops by substring
(`contains("optimizer"/"adam"/…)`, the `schedule.optimizer_shard` WIRE row);
the linalg precision/refinement annotations still have no consumer (an
attribute-level #29 gap); the WIRE slices that would make the corrected rows
green again (ntk_rope canonicalization, sm_120 Target producers) are open.

Evidence: fixtures under `src/compiler/codegen/tessera_gpu_backend_NVIDIA/test/nvidia/tmem_*.mlir`,
`nvidia_marker_result_used.mlir`, `tests/tessera-ir/phase3/tile_buffer_*`,
`tests/tessera-ir/phase5/{linalg_solver_op_identity,optimizer_shard_zero_config*}.mlir`;
before/after on Tajasarus's assertions-ON LLVM/MLIR 23.1.1 recorded in the
NVIDIA queue entry and the PR.

<!-- entry-fields:end -->

### 2026-09-27 — ROCm scheduled softmax/reduction images keyed on a shape-free kernel identity

Owner: [E2E-REAL-6](INTEGRATED_COMPILER_PLAN.md#e2e-real-6)

PRs: branch `claude/foundation-batch-2-rocm-cache-key` (umbrella `claude/foundation-batch-2`).
Sync: `FOUNDATION-BATCH-2-2026-09-27`.

Outcome: the follow-up the ROCm unary cut recorded as open is closed. The
compiled route's image was keyed on its Tile text, which binds the shape
(through the Schedule digest and launch constants) and the Graph function name,
so every new shape or caller name recompiled an identical HSACO.
`package_scheduled_kernel` now compiles softmax and reduction (gfx1151 and
gfx1201) through `rocm_native._compile_shape_free_tile_ir`: `tessera-opt` still
consumes the Tile IR per request (Tile -> Target; Python never translates Tile
to Target), the Target IR is projected onto its one directive with the `name`
replaced by a symbol derived from the rest of the module, and the HSACO is
compiled from that projection at `input=directive` and cached by it through
the single ROCm cache authority (`_native_cache_key`). The key therefore holds
exactly what the binary is compiled from -- every directive attribute, the
arch, the pipeline config, the device libraries and the compiler binary --
rather than an audited list of fields. Host scaffolding (function signature,
buffer casts, `arith.constant` launch extents) is the only thing dropped; any
other op fails closed, and a family joins `_SHAPE_FREE_DIRECTIVES` only after
its generator is read and measured (a generator that bakes an extent must
carry it as a directive attribute first). The descriptor's `entry_symbol` is
read from the Target IR that produced the image; the Graph symbol is
provenance (`graph_symbol`). Decision #11: `kernel_code_identity` digests the
image's decoded instruction stream and symbols, so it reads what runs; images
of one identity are now byte-identical across shapes, and the `tessera-opt`
digest in every ROCm compile key is memoized on the binary's stat signature
(content-hashed again on any rebuild).

Measured (Princess-Luna gfx1151, `benchmarks/rocm/measure_rocm_unary_route_cache.py`;
compile cost only, not a runtime claim): before, each new shape cost 374–387 ms
(softmax f16) / 376–387 ms (reduce bf16), all cold. After, the first shape is
cold at ~200 ms and each further shape, and a different Graph symbol, is a
warm hit at 98–102 ms with the same HSACO and `image_digest`. The ~100 ms left
is per-shape lowering + ancestry replay + Tile -> Target (five `tessera-opt`
runs) and the device-library probe. The digest memo takes an exact warm hit
from ~186–208 ms to 16–20 ms (visible on the retired-route rows). Evidence the
projection is binary-faithful: on Princess-Luna the reduction HSACO compiled
from the extracted directive has the same `llvm-objdump -d` stream and `.kd`
descriptor as the one compiled from the full Tile module, and all retired-vs-
compiled bitwise device rows pass launching the canonical symbol.

Remaining: other ROCm families keep the Tile-text key (each needs the same
generator read and measurement before joining); the ~100 ms per-shape floor is
lowering/replay subprocesses, not codegen; NVIDIA's scheduled unary image key
was not measured (follow-up candidate, no parity claim).

Evidence: `tests/unit/test_rocm_shape_free_cache_key.py` (host-free: projection
independent of shape and Graph symbol; each directive attribute, the header,
the arch and a rebuilt compiler miss; two shapes and two symbols share one
compile; unaudited Target IR, a second directive and a nameless directive fail
closed; exact-device: two shapes + two symbols -> one image, both launches
correct, a storage/kind change -> a new image). Princess-Luna
(`TESSERA_ROCM_E2E_DEVICE_TEST=1`): 500 passed / 101 skipped (gfx1201 gates,
Darwin) over the five ROCm unary/scheduled files, 185 exact-device rows; ROCm
subset 4578 passed / 465 skipped / 0 failed; `check-tessera-ir` 521 + 4
unsupported, `check-tessera-rocm` 82/82. Tajasarus (gfx1201, assertions-ON
LLVM/MLIR 23.1.1): the same five files with `TESSERA_ROCM_CHIP=gfx1201 TESSERA_GFX1201_DEVICE_PROOF=1` -- 384 passed, 217 skipped (gfx1151 device gates, Darwin), including both new two-shape/two-symbol reuse rows and both `test_gfx1201_scheduled_package_executes` rows through the `input=directive` compile; ROCm subset 4615 passed / 418 skipped / 10 failed under `-n 8`, all ten in `test_rocm_sparse_{runtime,byte_formats}.py` ("sparse worker teardown is unconfirmed" -- worker-process contention on a GPU shared with another job), and those two files pass 34/34 run serially; `check-tessera-ir` 459 passed / 66 unsupported, `check-tessera-rocm` 82/82. Mac: full `tests/unit` sweep (`-m "not slow"`, Apple + x86 + EBM + Clifford build) 21675 passed / 4017 skipped / 0 failed; mypy clean; `check_compiler_plan.py` and generated-doc drift clean.

<!-- entry-fields:end -->

### 2026-09-27 — ROCM-FP8-BLOCKSCALE-1: ragged M stops paying for its masked edge

Owner: [ROCM-FP8-BLOCKSCALE-1](INTEGRATED_COMPILER_PLAN.md#rocm-fp8-blockscale-1)

PRs: branch `claude/foundation-batch-2-gfx1201-gaps` (sync `FOUNDATION-BATCH-2-2026-09-27`; follow-ups of PR #872).

Outcome: (a) the W8A8 LDS rule's CU count comes from one authority: `measuredComputeUnits` (PMPasses.cpp) mirrors the new `rocm_target.compute_units` (2 x the measured `_DISPATCH_SLOTS` WGPs), a unit test compares the tables entry for entry, `lower_blockscale` refuses a Schedule whose panel the Python oracle `blockscale_panel_oracle` does not reproduce, and an unmeasured arch keeps the register panel with a registered `ROCM_FP8_BLOCKSCALE_LDS_NOT_APPLIED` warning. (b) The ragged-M gap was the bounded store, not the 128-row tile: a ragged M = 1000 ran 1.25x slower than M = 1024 on the same grid because the block-scale join's per-element rows, hoisted by LICM above the K loop, were handed by GVN to a store written against absolute rows (251 vs 238 VGPRs at 128x128, 233 vs 187 at 128x64). `materializeFragmentStore` now tests each element's row as a constant against the lane's room and addresses it from the lane's row base (same elements, predicates and addresses; the column keeps its absolute form, which measured better): every ragged 128x128 variant is 240 VGPRs, no spills. Ragged M then follows the whole-M rule (128x128 when it gives >= 64 workgroups). Device clock, paired, vs unmodified AITER: ragged-M geomean **0.965** (was 1.086 at the previous compiler, 26 points); where the selection changed, 0.90x of the old tile (0.83-1.08). Whole-M LDS kernels byte-identical; the 54-row comparison is unchanged (0.647 / 0.905 / 0.912 by M bucket). (c) The short-K / N = 1024 whole-M gap (6 of 18 shapes at 1.04-1.09x AITER) is still open; grouped raster (~1-3%, not converging), 16-wave grids, a register-staged next slab and double-buffered LDS at stage K 64 all measured negative.

Remaining: short K (K <= 2048, or N = 1024) at 1.04-1.09x AITER, also the K = 1536 ragged rows (1.12-1.34x); 200x8192x1024 loses 8% under the new rule and 200x2048x2048 is 6% slower from the store change (both recorded, not tuned around); the shared bounded store's effect on gfx1151 kernels is untimed (static census on gfx1201 only).

Evidence: [ragged/short-K packet](../../../benchmarks/baselines/gfx1201_fp8_blockscale_ragged_20260927/README.md), `tests/unit/test_rocm_fp8_blockscale.py`, `tests/device/rocm/test_fp8_blockscale_w8a8.py`, `tests/tessera-ir/phase2/e2e_fp8_blockscale_lds_rocm_target.mlir`.

<!-- entry-fields:end -->

Found on the way: the first attempt wrote the column test in the per-lane form too; that made the ragged-N 128x128 body 256 VGPRs plus spills (from 251), so only the row is rewritten. A generator-side attempt (stating a whole dimension's bound as the fragment edge so it folds) was built, measured unnecessary once the row form landed, and removed. A 64-row LDS tile (64x64/4 waves, 64x128/8 waves) moved the ragged geomean only 1.07 -> 1.04 before the store fix and is not selected.

### 2026-09-27 — ROCM-MXFP4-W4A8-1: the one-row-block gap is not the weight bytes

Owner: [ROCM-MXFP4-W4A8-1](INTEGRATED_COMPILER_PLAN.md#rocm-mxfp4-w4a8-1)

PRs: branch `claude/foundation-batch-2-gfx1201-gaps` (sync `FOUNDATION-BATCH-2-2026-09-27`).

Outcome: the M = 256 gap to Radiance (1.05-1.23x) was tested against the hypothesis that Radiance's packed E2M1 weights (half the bytes) explain it. An N scan at M = 256, K = 5120 (N 4096..24576) with three rotating input copies and with one (operands cache-resident where they fit the 64 MiB last-level cache), device clock witnessed by HIP events, three processes, all engines bitwise equal to exact K32: residency closes the gap at N = 4096 (1.03-1.07x -> 0.98-1.01x), but with both weights resident at N = 8192-12288 the gap stays 1.11-1.15x, and the marginal cost per output column is 16.0-16.7 ns for Tessera against 12.7-12.8 for Radiance in both regimes. The two opt-in packed-E2M1 candidates (bitwise exact, half the weight bytes) are 1.20-1.48x Radiance, slower than the expanded selected schedule at every N. Verdict: not the weight bytes; the per-column cost is unattributed (no counters on WSL2). Exact K32 stays default, folded opt-in, no selector change.

Remaining: the M = 256 per-column cost (A restaging per 64-column block, LDS fragment traffic, epilogue or issue -- unmeasurable here); a packed decode on the selected load schedule is untested (the candidates lack its keys).

Evidence: [one-row-block packet](../../../benchmarks/baselines/gfx1201_mxfp4_one_row_block_20260927/README.md), `benchmarks/rocm/record_gfx1201_mxfp4_folded_load_schedule.py` (`--shapes nscan`, `--packed`, `--copies`).

<!-- entry-fields:end -->

### 2026-09-27 — CI lanes that reported success having tested nothing

Owner: [COMPILER-DEVEX-1](INTEGRATED_COMPILER_PLAN.md#compiler-devex-1)

PRs: branch `claude/foundation-batch-2-ci-lanes` (umbrella `claude/foundation-batch-2`).

Outcome: The hosted `lit` and `rocm-serialize` lanes gated on the exact
LLVM/MLIR 23.1.1 pin; apt.llvm.org now serves 23.1.2, so both printed
`::warning … skipping`, set `mlir=false`, gated every later step off it and
reported **success having configured, built and tested nothing** (push run
36347063229 on main). The `sanitizer` lane installed no LLVM/MLIR at all; its
last real run (35609056270, 2026-09-21) failed at `find_package(MLIR)`.
Owner decision (sync `FOUNDATION-BATCH-2-2026-09-27`): hosted CI accepts any
23.1.x patch, records the exact version, and **fails** when none is available;
the fleet keeps its exact pin. Implemented as: `TESSERA_LLVM_PIN_MODE`
(`exact` default, `minor` passed only by `validate.yml`) in
`cmake/TesseraToolchainPins.cmake` — `minor` relaxes only the patch, still
rejects a mixed LLVM/MLIR pair and other series, writes
`tessera_llvm_pin.txt`, and refuses an MLIR that reports no version;
`scripts/ci_resolve_llvm.sh` resolves the prefix (llvm-config, mlir-opt,
CMake packages, optional ld.lld), writes version to `$GITHUB_OUTPUT`, the job
summary and a `ci-toolchain/*.json` manifest, or fails with `::error`; every
MLIR lane uploads `ci-toolchain/` on success and failure; apt install
failures fail the step (the `|| echo ::warning` fallbacks are gone); the
pytest proof steps now run through `scripts/ci_require_executed.py`, which
fails an all-skipped or unexpectedly-skipped run (their tests `skipif` a
missing tool, so a broken build was otherwise a second green no-op); the
sanitizer lane installs LLVM/MLIR 23 + `libclang-rt-23-dev` and passes the
pin mode through `run_sanitizers.sh`. Audit of the other workflows: no other
step-output skip gate; `pylint.yml` (`--exit-zero`) and
`profiler-native-proofs.yml` (`--allow-unavailable`, no device on hosted
runners) still succeed without proving anything, deliberately, and are now an
explicit allow-list in the gate. Sibling backends: ROCm — `rocm-serialize`
proves hsaco emission again once the lane runs; Apple/NVIDIA/x86 — not
applicable (no hosted lane builds for them beyond the portable `lit` build).

Remaining: Only a real Actions run proves the lanes now pass on
ubuntu-latest with apt's 23.1.2 — the local simulation cannot see apt, the
runner image, or a configure/build that 23.1.2 might break (MLIR API drift
within 23.1 is possible; if it happens the lane now fails loudly, which is the
intended outcome). The sanitizer lane's Linux TSAN path (clang-23 +
`libclang-rt-23-dev`, non-PIE) has not run since it was added. Whether a
label-triggered `profiler-native-proofs` lane should fail when its provider is
unavailable is an owner call.

Evidence: `tests/unit/test_ci_workflow.py` (`TestNoSilentToolchainSkip`,
`TestFleetPinStaysExact`, `TestResolverBehaviour`, `TestCMakePinModes`,
`TestRequireExecuted`) — fails 11 tests against the pre-change `validate.yml`
and passes on this branch (Mac, macOS 27, LLVM/MLIR 23.1.1). Local simulation
(Mac): the resolver against faked prefixes accepts 23.1.1 (`exact`) and
23.1.2 (`series`, recorded) and fails 23.2.0, 24.1.0, absent, mixed
LLVM/MLIR, missing CMake packages and missing required ld.lld;
`tessera_pin_llvm` under `cmake -P` rejects 23.1.2 in `exact` mode and
accepts only matched 23.1.x in `minor`; the real Homebrew 23.1.1 keg resolves
`exact`; `run_sanitizers.sh asan ubsan` with `TESSERA_LLVM_PIN_MODE=minor`
configured, built and ran both smoke binaries clean.

<!-- entry-fields:end -->

### 2026-09-27 — ODS WIRE slices 1 and 4: target_verify / ntk_rope reach their canonical consumers; the Philox Langevin step gets a producer

Owner: [GOV-ODS-CONSUMER-1](INTEGRATED_COMPILER_PLAN.md#gov-ods-consumer-1)

PRs: branch `claude/foundation-batch-2-wire-a` (umbrella `claude/foundation-batch-2`).
Sync: `ODS-WIRE-1-4-2026-09-27`.

Outcome: Three waived ODS ops now have a producer and a consumer, and the
waiver ceiling drops 79 -> 76 ([triage rows](ODS_OP_CONNECTION_TRIAGE.md#tessera-target-verify)).
**Slice 1.** `src/transforms/include/Tessera/Transforms/CompositeDecomposition.h`
holds one name-matched pattern source: `tessera.target_verify(tokens, logits)`
-> `tessera.softmax(logits){axis = rank-1}` (the verifier already pinned S, so
`tokens` carries nothing further -- the named #32 reason), and
`tessera.ntk_rope(x, theta){s}` -> `tessera.rope(x, tessera.div(theta,
arith.constant splat(s)))`, with no division at `s = 1.0`; a scaled theta that
is not a static floating tensor fails closed with
`TESSERA_NTK_ROPE_THETA_UNREWRITABLE` (registered), and any composite left
after the rewrite is an error, not a silent no-op. Routes: `tessera-canonicalize`
(so every pipeline built on `addGraphIRPreLoweringPasses`: `-x86`, `-gpu`,
`-nvidia-sm{90,100,120}`); new standalone `tessera-decompose-composite-ops`
first in `tessera-lower-to-apple_gpu-runtime` (header-only, so `TesseraApple`
gains an include path, not a link), in the Apple `-full` reasoning prologue, and
in libtessera_jit stage 1a. ROCm is not a route: its pipelines consume Tile /
directive carriers and have no Graph softmax or rope consumer. `target_verify`
joins `_JIT_GRAPH_OPS`; `GraphFn` gained a declared i32 *index operand* (only
`target_verify` operand 0 may take one -- any other use refuses the graph, so
`@jit` falls back rather than computing on integer bits), and `@jit` passes an
int32 argument through as that operand. `Canon` now declares `arith` as a
dependent dialect (the rewrite builds `arith.constant`).
The `ntk_rope -> rope` dashboard alias stays withheld, departing from the
recorded "re-add when the rewrite lands": rope's x86 / ROCm device rows are
Python runtime executors keyed on the literal op, which no C++ rewrite feeds,
and on the Apple -runtime route a scaled `ntk_rope` leaves `tessera.div` with no
Graph consumer (`composite_decomposition_apple_gpu.mlir` pins it), so borrowing
rope's device-verified cells would over-claim. `target_verify` rows unchanged.
**Slice 4.** The MERGE alternative (onto `tessera_ebm.langevin_step`) was read
and rejected: that solver op takes an `energy_fn` whose gradient the compiler
derives, draws `sqrt(2 eta T)` noise and advances its key, so it cannot carry a
precomputed-gradient step over a caller-owned (seed, counter) stream -- a Graph
producer for the Philox op is a different capability, not a second authority
(#31). Producer: catalog `OpSpec("ebm_langevin_step_philox", ..., 4, 4,
effect=random, stochastic_identity=seed_counter)`, `ops.ebm_langevin_step_philox`
(`_ebm_ops.py`), `graph_ir._KEYWORD_ATTR_PARAMS` (`eta`, `temperature`),
`primitive_coverage`, PYTHON_API_SPEC. Consumer: `runtime._EBM_LANGEVIN_OPS =
("tessera.ebm.langevin_step_philox",)`; seed (1 x i64, split low word first) and
counter (4 x i64, each < 2^32 -- refused, never truncated) come from operands;
`eta` / `temperature` are required (#21a) and `noise_scale` defaults to
`sqrt(2 eta T)`, now stated in the ODS description. The executors used to accept
the 3-operand host-noise `tessera.ebm.langevin_step` and ignore its noise
operand; that name is now refused there, and its x86 / ROCm manifest credit
(which cited these Philox tests) moved to the Philox op, so
`ebm_langevin_step` shows x86 `reference` / ROCm `planned` -- the honest state.
The Apple Philox MSL row now cites `tests/unit/test_philox_runtime.py`
(execute-compare on Metal); the Graph op still has no Apple lane.

Remaining: Apple *execution* of `target_verify` / `ntk_rope` (their `@jit
(target="apple_gpu")` capability stays `artifact_only`); a route that executes
rope and the `theta / s` division (then re-add the alias and drop the two
`_KNOWN_OPEN_SINGLE_GPU` rows); an Apple Graph lane for the Philox op; gfx1201
proof of the repointed ROCm executor (not evaluated; no proof transfers).

Evidence: Mac (macOS 27, Homebrew LLVM/MLIR 23.1.1 NDEBUG): lit
`composite_decomposition{,_invalid,_apple_gpu,_x86}.mlir` pass; full `lit
tests/tessera-ir/` 479 passed / 50 unsupported / 0 failed;
`tests/unit/test_composite_decomposition.py` executes `target_verify` through
libtessera_jit (invocation counter +1) against numpy and the Python reference;
`test_ebm_langevin_philox_op.py` (host-free executor mapping and refusals);
full unit sweep 21792 passed / 4024 skipped / 1 failed, the one failure
(`test_op_arity_contract`, the new op's keyword attributes) fixed and re-run
green in the same session. Princess-Luna (Zen 5 AVX-512 + gfx1151, apt LLVM
23.1 NDEBUG, `~/wk-wirea`): `test_{x86,rocm}_ebm_langevin_compiled.py` (incl.
the traced-op launch), the kernel-level `*_langevin_philox_compiled` tests,
`test_ebm_langevin_philox_op.py`, `test_composite_decomposition.py`,
`test_native_cpu_jit.py` -- 61 passed, 0 skipped; lit 462 passed / 67
unsupported / 0 failed; `check-tessera-rocm` 82/82; full unit sweep 22709
passed / 3107 skipped / 1 failed (the same arity test, pre-fix).
Tajasarus (assertions-ON LLVM/MLIR 23.1.1, `llvm-config --assertion-mode` ON,
`~/wk-wirea/build-assertions`, `tessera-opt` only): the four new fixtures (the
Apple one unsupported there) and `ga_ebm_graph_ops{,_invalid}.mlir` pass; full
lit 462 passed / 67 unsupported / 0 failed. libtessera_jit is not built on that
box (no libffi), so the JIT-lane registration was not run under assertions.

<!-- entry-fields:end -->

### 2026-09-28 — ODS wiring slices 2 and 3: `tessera.istft_jvp` gets its Schedule consumer; `cache.commit/rollback` lower through an x86 handle ABI

Owner: [GOV-ODS-CONSUMER-1](INTEGRATED_COMPILER_PLAN.md#gov-ods-consumer-1)

PRs: branch `claude/foundation-batch-2-wire-b` (umbrella `claude/foundation-batch-2`).
Sync: `ODS-WIRE-B-2026-09-28` (x86, ROCm, NVIDIA, Apple todos).

Outcome: **Slice 2.** `GraphToSchedulePass` (`PMPasses.cpp`,
`scheduleIstftJvps`) consumes the `tessera.istft_jvp` that
`ISTFTOp::buildTangent` produces under `--tessera-autodiff-forward`: exact
profile only (Zen 5 AVX-512, gfx1151, gfx1201, sm120; else
`SPECTRAL_JVP_SCHEDULE_REFUSED`), every default resolved and written back on
the op (#32), tangent activity read from the IR -- a zero-splat tangent is
inactive (#30) -- the overlap-add geometry checked against the static tangent
type, and one hashed `schedule.jvp_contract` plus a matching
`schedule.artifact` (`family=spectral_jvp`). The native JVP plugin now builds
the ISTFT package from that contract
(`native_jvp_plugins.istft_jvp_contract_from_paired_ir` →
`istft_jvp_spectral_arguments` → `lower_scheduled_spectral`, which stays the
one spectral-program authority; the package records `graph_schedule_artifact`).
**The #31 dual authority is collapsed to one production path plus a declared
oracle**: the old source-kwargs derivation (`_source_kwargs_spectral_arguments`)
is re-derived for every ISTFT package, which is refused unless both lower to
the identical scheduled program; it is not deleted (ordering caveat).
Departure from the triage text, recorded: the arm does not emit
`schedule.spectral_program` itself, because that op's identity is the
ScheduleObject digest `scheduled_spectral.py` mints -- minting it in C++ too
would be a second spectral-program authority. Found while wiring: the x86 and
ROCm window-product symbols (`tessera_x86_istft_jvp_f32`,
`ts_istft_jvp_plan_hostptr_batch_amd`) take no n_fft/center/length and write
`(frames-1)*hop+window` samples into the cropped output buffer; those
geometries are now refused before launch. **Slice 3.** `TileToX86Pass`
(`LowerKVCacheCursorToX86`) lowers both cursor ops to a handle ABI in
`kv_cache_f32.cpp` -- `tessera_x86_kv_cache_{commit,rollback}_f32(handle*,
i64) -> handle*` over `struct tessera_x86_kv_cache_f32_handle` (truncate in
place, zero the dropped rows, same handle back or NULL untouched) -- and
threads the result, so commit → rollback becomes a call chain on the handle
pointer; a constant negative count is refused at compile time
(`X86_KV_CACHE_CURSOR_REFUSED`), and every call is followed by a NULL check
and a `cf.assert` naming that code, so a dynamic rejection traps instead of
threading NULL into later cache ops (review fix before merge). Runtime: `runtime.x86_kv_cache_cursor`
(refuses quantized/latent/SSM/non-f32/non-contiguous handles,
`X86_KV_CACHE_HANDLE_REFUSED`) and the bufferized form in
`x86_kv_cache_compiled` (`current_seq` and the count are never-defaulted
kwargs). The three ops leave the ODS consumer waiver (ceiling 79 → 76).

Remaining: none on sm_120 -- the ISTFT JVP proof ran on Super-Bear's RTX 5070
2026-09-28 (see the NVIDIA queue entry `ODS-WIRE-B-2026-09-28`). A reduced-precision ISTFT window is refused: the
frontend types the result f32 while the native packages emit window storage,
and the two must agree before it is admitted. Window-only ISTFT activity is
rejected upstream by the forward transform (pre-existing, unchanged:
reproduced on Princess-Luna's `build/` at `4e12e5d7b`, an ancestor of the
umbrella, and this branch touches no autodiff or TangentInterface source).
The `!tessera.kv_cache` function-boundary type conversion (the lowering keeps
the type behind a cast at the boundary), backend-manifest rows for
`cache_commit`/`cache_rollback`, and the SSM ring rewind stay open. Retire
the kwargs oracle once the differential test covers what it covers.

Evidence: Mac (macOS 27, LLVM/MLIR 23.1.1 NDEBUG): lit
`phase_f4/spectral_jvp_istft_schedule{,_invalid}.mlir`,
`phase2/x86_kv_cache_cursor_{abi,invalid}.mlir`; full `lit tests/tessera-ir/`
479 passed / 50 unsupported / 0 failed; `test_istft_jvp_ir_contract.py`
host-free differential (5 geometries × 2 activity sets × 4 profiles) and
`test_x86_kv_cache_cursor.py` host-free rows pass. Princess-Luna (Zen 5 +
gfx1151, `~/wk-wireb` at `ef16ce164`, canonical x86+ROCm build): the four
focused files 94 passed / 1 skipped (a gfx1201-only row), including x86 KV
commit/rollback/chain/rejection and the bufferized lane bit-exact against
`tessera.ops.cache_commit`/`cache_rollback`, and the IR-built ISTFT product vs
centred difference on x86 and gfx1151; `check-tessera-ir` 463 passed / 66
unsupported; `check-tessera-rocm` 81 passed / 1 unsupported. Tajasarus
(Zen 5 + gfx1201, assertions-ON LLVM/MLIR 23.1.1, `~/wk-wireb` at
`befbd4fd3`, `TESSERA_ROCM_CHIP=gfx1201 TESSERA_GFX1201_DEVICE_PROOF=1`):
the eight touched/adjacent fixtures pass under assertions; full lit 463
passed / 66 unsupported; `check-tessera-rocm` 81 passed / 1 unsupported; the
four focused files 92 passed / 5 skipped (all five gfx1151-only rows), with
the x86 KV rows and the IR-built ISTFT product on x86 and gfx1201 executing.

<!-- entry-fields:end -->

### 2026-09-28 — E2E-REAL-6: x86 softmax and reduction retire their Graph-owned admission and constructors

Owner: [E2E-REAL-6](INTEGRATED_COMPILER_PLAN.md#e2e-real-6)

PRs: branch `claude/foundation-batch-2-e2e-x86-unary` (umbrella `claude/foundation-batch-2`).
Sync: `E2E-REAL-6-x86-unary-2026-09-28`.

Outcome: the x86 twin of the ROCm unary cut. Since 2026-09-08 x86
`package_softmax` / `package_reduction` already lowered through
`lower_scheduled_kernel`, but admission (`supports_softmax` /
`supports_reduction`, hence `supports_native_package`) still read the Python
Graph object through `_softmax_contract` / `_reduction_contract`, the
`emit_softmax_tile_ir` / `emit_reduce_tile_ir` constructors were still exported
from production, and the packagers reached the lowering through an indirection
the bootstrap audit could not prove. Now admission is
`scheduled_kernel.supports_scheduled_kernel(target="x86")`, each packager hands
`lower_scheduled_kernel(..., architecture=zen5-avx512 | x86_64_base)` straight to
`package_scheduled_kernel`, and the retired contracts, constructors and the
pre-2026-09-08 Graph-owned packagers are frozen in
`tests/_support/x86_unary_baseline.py`, the declared oracle #31(a) allows.

Envelope. Retired: f32 in and out; `softmax` and `softmax_safe` (last axis,
shape-preserving); `sum` / `mean` / `max` / `amax` over the last axis (either
spelling), keepdims true or false; any positive static rank; both images.
Compiled: the same. The one point the scheduled contract had lost is
`softmax_safe` -- the Graph contract kept selecting it for native packaging and
the scheduled packager then refused it (the follow-up the ROCm cut recorded) --
and it is now admitted for x86 through the existing canonicalization.
Refused on purpose, each pinned by a test: an integer `keepdims` (coerced with
`bool` before), a non-serial reduction `schedule` hint and a `schedule` hint on
softmax (both silently ignored before). The reduction descriptor now carries
`nan_mode` read from the replayed Tile op and fails closed on anything but
`propagate`, which both reduce kernels implement (#21a, #32).

Image identity (the shape-free-key question). x86 compiles nothing per
package -- the payload is the prebuilt shared object, the same bytes for every
shape -- so there is no compile cache to key. But `image_digest` binds the
Target IR digest, which on the scheduled route carries the launch constants and
Graph symbol, and `runtime._load_x86_native_image` loaded one copy per digest
and never released it. Measured on Princess-Luna
(`benchmarks/x86/measure_x86_unary_route_cost.py`, 8 shapes per family, timing
lock): retired 1 digest, compiled 8; 16 copies of the 382 KiB object loaded;
each new shape's first launch ~0.4 ms slower. A digest miss now resolves through
an architecture + payload-sha256 map, and the same run loads one object (first
launch 2.23-2.25 ms vs 1.98-2.03 ms retired; warm 0.78-0.85 ms on both; runtime
claim, Princess-Luna). Package cost is 96-105 ms compiled vs 29-37 ms retired
per call (compile-cost claim, Princess-Luna), all lowering and replay
subprocesses; that route has been production since 2026-09-08, so a package
cache is recorded as a follow-up rather than widened into this cut.

Remaining: E2E-REAL-6 still owns x86 cohort/elementwise/breadth constructors,
ROCm paged-KV / MoE / forward attention, the NVIDIA and Apple gap families and
the frontend. NVIDIA still classifies `softmax_safe` as a native softmax its
scheduled packager refuses (needs sm_120 rows). `numeric_policy` keyword
arguments are still ignored by every target's unary contract (pre-existing).
The compiled x86 package costs ~3x the retired one per call (subprocess-bound).

Evidence: `tests/unit/test_x86_unary_differential.py` -- 288 host-free rows
(descriptor, ABI, buffers, scalars, shape guards, geometry, shared provenance
and Tile-kernel attributes identical, both images), refusal parity, pinned
intentional refusals, a fail-closed `nan_mode` row, and 298 device rows (288
bitwise retired-vs-compiled over every envelope point and both images, NaN
propagation, one loaded object across shapes). Princess-Luna (Zen 5 AVX-512):
that file + `test_x86_unary_migration.py` + `test_x86_e2e_spine.py` 662 passed
/ 0 skipped; `-k x86 -m "not slow"` 2461 passed / 1 skipped (umbrella head 1858
/ 1, the same skip); `check-tessera-ir` 462 passed / 67 unsupported. Tajasarus
(Zen 5, assertions-ON LLVM/MLIR 23.1.1 tree): the same three files 662 passed /
0 skipped; `-k x86` 2435 passed / 27 skipped / 0 failed; the four x86 unary
Schedule/Tile lit fixtures pass under assertions (no C++ changed).
`bootstrap_prune_gap.md`: x86 softmax/reduction gap -> generic; alpha
scoreboard `schedule_tile` 18 -> 24 (ratchet baseline tightened).

<!-- entry-fields:end -->

### 2026-09-28 — native MoE, cache, Apple Philox and sm_120 paged-KV follow-ups

Owner: [E2E-REAL-6](INTEGRATED_COMPILER_PLAN.md#e2e-real-6)

PRs: pending; sync `E2E-REAL-6-NATIVE-FOLLOWUPS-2026-09-28`.

Outcome: gfx1151 MoE direct token gather now follows a registered Graph subtype,
native Schedule and Tile replay, a shape-free ROCm Target directive and a
checked image entry. Exact Graph x86 `trunc` repeats reuse the native package.
Apple has a bounded Philox Langevin Graph lowering to a status-bearing Metal
ABI; the scaled RoPE Metal calls passed direct C ABI numerical checks.

Remaining: Apple `@jit` execution still reports `artifact_only` for `ntk_rope`
and `target_verify`; the Philox pass and direct ABI test do not prove a public
JIT route. Broader paged KV and the other x86/NVIDIA families, ROCm cache keys,
per-envelope certificates and device-only paged-KV kernel timing remain open.

Evidence: Princess-Luna WSL rebuilt `tessera-opt` and passed 333 focused MoE,
registry and drift tests plus both native MoE compiler fixture stages; the
device cases include S<T and S>T. Its x86 cache packet records 96.85–104.18 ms
cold and 0.313–0.336 ms repeat medians. Super-Bear sm_120 passed four paged-KV
remap/invalid-page device rows and 25 scheduled packed-state tests. The Mac
rebuilt its runtime/compiler, passed the Philox lit fixture and direct Metal
Philox test, and measured scaled RoPE C ABI maximum absolute error 4.77e-7.
WSL HIP events remain invalid on this fleet, so the paged-KV packet contains
host launch wall time only. See the [x86 packet](../../../benchmarks/baselines/e2e_real6_x86_trunc_cache_20260928/README.md).
<!-- entry-fields:end -->

### 2026-09-28 — E2E-REAL-6: x86 elementwise lowers through a native Schedule contract; cohort-2 and breadth follow except ALiBi and batched linalg; x86 package compile cache

Owner: [E2E-REAL-6](INTEGRATED_COMPILER_PLAN.md#e2e-real-6)

PRs: branch `codex/x86-batch3-dedup` (post-#875).
Sync: `E2E-REAL-6-x86-kernel-2026-09-28`.


Outcome: `x86_native.package_elementwise` and `package_cohort2` and
`x86_breadth.package_graph_breadth` no longer read the Python Graph object to
decide admission or author Tile IR. Admission is
`scheduled_kernel.supports_scheduled_kernel(target="x86")` (host-free,
`native_x86_kernel.admit`), and every moved packager hands
`lower_scheduled_kernel(module, target="x86")` straight to
`package_scheduled_kernel`, which projects the descriptor from the contract
serialized in replayed Tile IR (`native_x86_kernel.project`). The native owner
is a new table-driven contract, `src/compiler/programming_model/lib/NativeX86Kernel.h`
(Graph -> content-addressed `schedule.artifact` record -> the same
`tile.{elementwise,argreduce,scan,rope,x86_abi}_kernel` launch the retired
constructors authored), opt-in per module by `tessera.launch_bindings`;
`absolute`/`floor`/`ceil`/`trunc`/`cumsum` keep `NativeAbsolute.h`, and x86 row
normalization now rides the existing `schedule.norm` semantic kernel (Zen 5
admitted beside sm_120 in `getSemanticKernelSchedule` and the `NormOp`
verifier). The retired `_elementwise_contract` / `_cohort2_contract` /
`graph_breadth_contract`, the `emit_elementwise_tile_ir` /
`emit_cohort2_tile_ir` constructors and the packagers are frozen verbatim in
`tests/_support/x86_kernel_baseline.py`, the declared oracle #31(a) allows;
`scheduled_absolute`'s packagers are the oracle for its five kinds.

What had to land first: 45 new Graph ODS ops; `trunc` was already registered by #875. Most of the x86 vocabulary
(`sqrt`, `exp`, `isnan`, `logical_*`, `bitwise_*`, `where`, `argmax`,
`cumprod`, `gather`, `loss.log_cosh`, ...) was a catalog spelling the Graph
dialect never declared, so `tessera-opt` could not parse it and no native route
could own it. They are now registered with real verifiers (tail of
`TesseraOps.td`; consumer: the x86 contract), as the tensor/tensor comparisons
were before them. Catalog aliases the retired tables accepted (`subtract`,
`equal`, `power`, `swiglu`, `mse_loss`, ...) are spelled as the ODS op before
emission.

Envelope, before -> after. Elementwise: 71 op spellings (unary 10, binary 12,
predicate 3, compare 12, logical 4, bitwise 5, transcendental 21,
binary_math 4) plus `where`, any positive static rank, AVX-512 image only ->
the same. Cohort-2: argmax/argmin (last axis or flattened, keepdims),
cumsum/cumprod/cummax/cummin (last axis), rmsnorm/rmsnorm_safe/layer_norm
(eps), rope, ALiBi -> all but ALiBi. Breadth: gather, the 10 pointwise-loss
spellings with `reduction="none"`, rank-2/3 cholesky and tri_solve -> all but
rank-3. Refused on purpose, each pinned by a test (15): unknown keywords on
elementwise/loss/gather (ignored before), comparison `signedness`, an integer
`keepdims`, a flattened rank >= 2 scan, a norm `numeric_policy` or non-last
`axis` (ignored before), `lower=False` cholesky and `trans`/`unit_diag`
tri_solve (silently computed the default before), a zero huber `delta`,
rank-1 rope (Graph verifier `LEGALITY_ROPE_RANK`), flattened keepdims argmax
over rank >= 2 (NumPy keeps every axis), and a repeated operand (`add(x, x)`,
which the retired route admitted and then could not package). Corrected, also
pinned: the retired flattened argmax described a rank-1 operand the caller
never passes, and the runtime refused that descriptor for any rank >= 2
operand; the compiled route describes the real operand and executes. The norm
`epsilon=` keyword is honoured (the retired x86 norm read only `eps` and
silently used 1e-5). A rank-1 `cumsum(axis=None)` now packages (the retired
route emitted `axis = none`).

Still gaps, and exactly why. `cohort2` keeps ALiBi on a narrowed retained
constructor (`x86_native._alibi_contract`): its Graph operand list is not
decodable by position (the catalog admits 0-2 optional operands with no
presence flags, `test_op_arity_contract.py::_UNDECODABLE_OPERAND_LISTS`) and
the ODS `tessera.alibi` declares no slopes operand at all, so the x86 slopes
form is not Graph IR. `breadth` keeps rank-3 (batched) cholesky / tri_solve on
`x86_breadth.batched_linalg_contract`: the ODS verifiers are rank-2 only
("batched rank-3 is a follow-on", pinned by
`apple_cholesky_graph_ir_invalid.mlir`). Both rows stay `gap` on the dashboard
by the audit's all-paths rule.

Package cache. `x86_compile_cache` memoizes each `tessera-opt` run on (the
compiler binary's SHA-256 -- `rocm_native._tool_digest`, stat-memoized --, the
pass option, the complete source text), the `--version` probe on the binary
digest, and the shared object on its stat signature. The source text is the
MLIR the compiler receives, so Graph op identity, attributes, shapes, dtypes,
target/arch and bindings are all in the key; a changed compiler or input misses,
a failing run is not cached, and the verified #875 unary artifact caches remain keyed by complete artifact and
toolchain identity; new x86 kernel descriptors, projections and replay
comparisons are rebuilt every call (a forged artifact still fails). Used by every x86 lowering, replay and Tile ->
Target run. Compile-cost claim, Princess-Luna, load < 2, timing lock
(`benchmarks/baselines/x86_package_cache_20260928/`): compiled route cold
53.65-82.10 ms (3 compiler runs), warm 0.29-0.95 ms (0 runs), retired
cold 25.44-26.86 ms except absolute at 82.37 ms; all relevant #875 and
compiler-run caches were cleared between cold samples. The first call in a
process took 147.16 ms including binary digest work.

Remaining: E2E-REAL-6 still owns x86 ALiBi (needs a decodable Graph operand
list and an ODS slopes operand) and batched linalg (needs rank-3 Graph ODS),
ROCm paged-KV / MoE / forward attention, the NVIDIA and Apple gap families and
the frontend. `scheduled_absolute` survives only as its five kinds' oracle.

Evidence: `tests/unit/test_x86_kernel_differential.py` -- 501 host-free rows
(elementwise 288, cohort-2 158, breadth 55: descriptor, ABI, buffers, scalars,
shape guards, geometry and shared provenance identical; the Tile launch op's
attributes identical; and the `tessera-x86-executable` Target IR identical --
the C-ABI call and kind constant, or for norm the symbol and the f32 epsilon),
15 pinned intentional refusals, 13 refusal-parity rows, the retained-constructor
rows, the flattened-argmax correction, forged-contract rows, a drift test that
the native table and the Python admission own the same ops and that the native
breadth ABI rows match `X86_BREADTH_ABIS`, and 501 device rows (bitwise
retired-vs-compiled on two seeds over every envelope point).
`tests/unit/test_x86_compile_cache.py` (11 rows: exact-repeat hit, op / shape /
kind / attribute / compiler / shared-object misses, failures uncached, forged
artifact under a warm cache). On the deduplicated Princess-Luna Zen 5 worktree, the native differential
file passed 1,049 cases, including bitwise retired-versus-compiled execution
on two seeds per admitted envelope point; focused cache, cohort, breadth and
#875 unary regression files passed 122 tests. The two new Graph/Schedule/Tile
lit fixtures passed against the freshly built full Graph compiler. The
registry/audit drift gates passed 317 tests and the exact ODS count gate
passed 748. Broader suites and sibling-host proof remain to be refreshed.
`bootstrap_prune_gap.md`: x86 `elementwise` gap -> generic
(`cohort2`, `breadth` stay gap for the two reasons above); Tile-constructing
bootstrap packagers 10 -> 8; `verifier_coverage` 245 -> 291 real.

<!-- entry-fields:end -->

### 2026-09-28 — ROCM-FP8-BLOCKSCALE-1: grouped LDS reads and uniform scale join on gfx1201

Owner: [ROCM-FP8-BLOCKSCALE-1](INTEGRATED_COMPILER_PLAN.md#rocm-fp8-blockscale-1)

PRs: branch `codex/rocm-perf-batch3-dedup`; sync `FOUNDATION-BATCH-3-DEDUP-2026-09-28`.

Outcome: The 128x64 FP8 LDS body groups fragment reads before WMMA under a
register-budget rule and the block-scale join loads one uniform weight scale
per fragment with a block-zero fallback for masked columns. Both M=200
regression rows improve versus rebuilt #875 and
AITER; the selected K=1536 row is near AITER.

Remaining: Four paired rows do not close the short-K or ragged-K envelope.
The sibling backends have no consumer of this gfx1201 physical schedule.

Evidence: Tajasarus gfx1201, full HIP compiler from `0a2d7fa7a`, 57 exact
W8A8 device tests and two lit fixtures passed. `w8a8_paired.json` records
source-matched #875/AITER comparisons with all twelve timing arms admitted
by device-clock and HIP witnesses. [Packet](../../../benchmarks/baselines/gfx1201_batch3_dedup_20260928/README.md).

<!-- entry-fields:end -->

### 2026-09-28 — ROCM-MXFP4-W4A8-1: M256 A-restaging attribution

Owner: [ROCM-MXFP4-W4A8-1](INTEGRATED_COMPILER_PLAN.md#rocm-mxfp4-w4a8-1)

PRs: branch `codex/rocm-perf-batch3-dedup`; sync `FOUNDATION-BATCH-3-DEDUP-2026-09-28`.

Outcome: Controlled M=256 output-changing probes show A fetch plus LDS
restaging together cost more than either isolated half at N=8192/12288.
The source control varies by 2-4%; no traffic counters are available.

Remaining: Which A-stage component Radiance implements more cheaply is
unattributed. The selected folded route remains opt-in; no performance
promotion or selector change follows from diagnostic probes.

Evidence: Tajasarus gfx1201, 32 folded numerical device tests passed;
`mxfp4_slope.json` has three processes and seven trials for each of three
N values against pinned Radiance `dfdfa383`. [Packet](../../../benchmarks/baselines/gfx1201_batch3_dedup_20260928/README.md).

<!-- entry-fields:end -->

### 2026-09-28 — batched x86 linalg and ROCm attention image identity

Owner: [E2E-REAL-6](INTEGRATED_COMPILER_PLAN.md#e2e-real-6)

PRs: pending; sync `COMPILER-NEXT-SLICES-2026-09-28`.

Outcome: Rank-3 x86 Cholesky/triangular solve use native Graph/Schedule/Tile;
the retained Python constructor is removed. ROCm forward attention reuses
images across runtime-only extents while retaining replay and launch guards.

Remaining: Apple native package execution, x86 ALiBi, general paged layouts,
NVIDIA LSE/backward/quantized routes, ROCm matmul cache identity and physical
W8A8/MXFP4 follow-ups. Movement and sm_120 timing gates did not justify promotion.

Evidence: 1,112 x86 differential/breadth/registry checks; gfx1151 attention
47 passed/5 skipped, gfx1201 29 passed/1 skipped; 21 gfx1151 movement checks;
25 sm_120 packed/state replay checks; two fresh direct Apple Metal ABI checks.
[Measured packet](../../../benchmarks/baselines/compiler_next_slices_20260928/README.md)
keeps compile cost, device timing and full-call time separate. Apple @jit
still records artifact-only provenance. All four backend queues were assessed.
<!-- entry-fields:end -->

### 2026-09-29 — evidence consumers, NVIDIA fragments, and public residuals

Owner: [EVIDENCE-PACKET-1](INTEGRATED_COMPILER_PLAN.md#evidence-packet-1)

Additional owners: [W1.1](INTEGRATED_COMPILER_PLAN.md#w11),
[FRONTEND-IR-MEDIUM-1](INTEGRATED_COMPILER_PLAN.md#frontend-ir-medium-1),
[AD-RESIDUAL-EVAL-1](INTEGRATED_COMPILER_PLAN.md#ad-residual-eval-1).

PRs: pending; sync `COMPILER-EVIDENCE-FRAGMENT-RESIDUAL-2026-09-29`.

Outcome: Three gfx1151 physical-math sum rows now reload a serialized native
Graph/Schedule/Tile/Target package and require artifact, image, and descriptor
identity in launch receipts. The sm_120 lowering refuses tensor-valued MMA
producers before an async token becomes a data operand. A coupled public
tracer residual executes forward and repeated backward through native tape
packages on gfx1151 and sm_120 with analytic gradient checks and saved-input
mutation isolation.

Remaining: Eighteen ROCm math rows retain metadata-only probes; no selector
promotion. The two NVIDIA producers still require explicit typed fragment and
accumulator materialization. Broader frontend masks, layouts, aliases, and
solver wiring remain open. Apple and gfx1201 have no new device proof.

Evidence: Princess-Luna passed 35 focused audit/frontend/math tests and the
21-row physical-math run; Super-Bear passed 13 NVIDIA fragment tests and the
source-matched coupled residual run. The math and residual timings are
synchronized host calls, not device-kernel measurements.
[Packet](../../../benchmarks/baselines/compiler_evidence_fragment_residual_20260929/README.md).

<!-- entry-fields:end -->

### 2026-09-29 — GitHub LLVM/MLIR patch pin corrected

Owner: [COMPILER-DEVEX-1](INTEGRATED_COMPILER_PLAN.md#compiler-devex-1)

PRs: [#883](https://github.com/gstoner/tessera/pull/883); sync `CI-LLVM-EXACT-2026-09-29`.

Outcome: Hosted lit, ROCm serialization, and sanitizer lanes install a
SHA256-checked official LLVM/MLIR 23.1.1 release instead of rolling
apt.llvm.org MLIR 23.1.2. The resolver and CMake both enforce the exact
fleet pin. The release's ICU70 runtime is isolated beside the toolchain,
and Tessera matches its no-RTTI LLVM ABI.

Remaining: Exact-device backend execution and performance proof remains owned
by each device host. The hosted compiler lanes are host-free and do not replace
those results.

Evidence: GitHub main lit artifact from run 36514973125 recorded LLVM and
MLIR 23.1.2 against fleet pin 23.1.1. The verified official release archive
has SHA256 `832aeb58d105de1cabc7b982dd2c65de0610f7377df48ae8fc2dd8e97420a15c`.
Princess-Luna WSL configured Tessera with exact 23.1.1 from that archive;
the automatic no-RTTI CMake configuration built `tessera-rocm-opt` and
three ROCm HSACO serialization tests passed with the pinned tools. The
hosted lit lane exposed a stale gfx1201 scale-load fixture; its expected
second masked-column scale-zero load passes FileCheck on host LLVM 23.1.1.
CI and audit drift tests passed 72 cases on WSL. GitHub PR run
36571380840 recorded exact 23.1.1 LLVM/MLIR, 542/542 lit passes and successful
ROCm HSACO serialization, alongside required lint, unit, audit and fan-in
checks. The earlier full dispatch run 36568476381 passed ASAN, TSAN and UBSAN
with the same pinned compiler bundle.
<!-- entry-fields:end -->

### 2026-09-29 — x86 ALiBi explicit slopes reaches native Schedule and Tile

Owner: [E2E-REAL-6](INTEGRATED_COMPILER_PLAN.md#e2e-real-6)

PRs: [#885](https://github.com/gstoner/tessera/pull/885); sync E2E-REAL-6-ALIBI-2026-09-29.

Outcome: Graph ALiBi declares and verifies its optional f32 slopes operand.
The x86 explicit-slopes envelope lowers through a content-addressed native
Schedule record to Tile and a replay-projected AVX-512 launch descriptor.
The remaining Python ALiBi constructor has moved out of production.

Remaining: Apple JIT execution, ROCm paged layouts and matmul image keys,
NVIDIA attention/quantized migrations, and the route census. Sibling ALiBi
operand consumers need their own target proof.

Evidence: Princess-Luna WSL rebuilt with LLVM/MLIR 23.1.1 and the x86
backend, passed 1,088 x86 differential/cohort/position tests and 38 operator
registry/arity tests. Four numerical benchmark rows agree with NumPy and
record diagnostic package and host-wall launch timing only.
[Packet](../../../benchmarks/baselines/x86_alibi_native_20260929/README.md).
<!-- entry-fields:end -->

### 2026-09-29 — traced ALiBi result shape reaches native x86 execution

Owner: [E2E-REAL-6](INTEGRATED_COMPILER_PLAN.md#e2e-real-6)

PRs: [#886](https://github.com/gstoner/tessera/pull/886); sync E2E-REAL-6-ALIBI-SHAPE-2026-09-29.

Outcome: The catalog names an ALiBi-specific shape rule: optional slopes[H] and static num_heads/seq_len produce f32 [H,S,S]. A public @jit trace now passes the Graph verifier, native x86 Schedule/Tile packaging and checked AVX-512 launch.

Remaining: Apple, gfx1151/gfx1201 ROCm and sm_120 ALiBi consumers still require their own physical parity. The other E2E-REAL-6 family migrations remain open.

Evidence: Host WSL traced-frontend and exact Zen 5 numerical test compares the native output with NumPy. Focused shape-rule, operator, and generated-doc gates run with this slice; no new kernel-time measurement is claimed.
<!-- entry-fields:end -->

### 2026-09-29 — gfx1151 bounded matmul image identity

Owner: [E2E-REAL-6](INTEGRATED_COMPILER_PLAN.md#e2e-real-6)

PRs: [#887](https://github.com/gstoner/tessera/pull/887); sync E2E-REAL-6-ROCM-MATMUL-IMAGE-2026-09-29.

Outcome: The static unfused unsplit gfx1151 f16/bf16 register-matmul route projects one replayed Target directive, excludes only shape-bound wrapper text and its checked Schedule digest, and compiles the image from that exact directive. Three distinct static shapes reuse one image while retaining separate Schedule and Tile artifacts. The ROCm-specific packet schema is registered as a syntax-only benchmark surface.

Remaining: gfx1201, fused, split-K, dynamic and LDS matmul image identities; Schedule/Tile replay package cost; clean device-kernel timing and any selector decision. Apple, x86 and NVIDIA have no consumer of this HSACO key.

Evidence: Clean-source Princess-Luna gfx1151 with rebuilt LLVM/MLIR 23.1.1 passed 34 focused shape-free tests (one other-device skip). A 31-sample-per-shape packet records numerical error below 9e-8, one image across three shapes, and three cold compilations under the historical Tile-text control. Timings are WSL host-wall diagnostics. Post-review, the raw benchmark launcher resolves the package descriptor entry; exact gfx1151 aligned 64x64x64 and ragged 65x67x31 cases pass NumPy, while the small-shape throughput gate remains rejected. [Packet](../../../benchmarks/baselines/gfx1151_matmul_shape_key_20260929/README.md).
<!-- entry-fields:end -->

### 2026-09-30 — fp32 matmul epilogues and resident-edge fused-input refusal

Owner: [E2E-REAL-6](INTEGRATED_COMPILER_PLAN.md#e2e-real-6)

PRs: [#890](https://github.com/gstoner/tessera/pull/890); sync MATMUL-EPILOGUE-RESIDENT-EDGE-2026-09-30.

Outcome: The eager matmul reference now retains fp32 accumulation through explicit and mapping epilogues and casts only the final output when fp16 is requested. The SM120 RMSNorm-to-matmul resident package rejects fused bias, residual, and activation at both Schedule artifact admission and descriptor validation because its current edge ABI does not carry those operands.

Remaining: the consumer ABI needs an explicit extension before fused resident epilogues are admitted. Rounding-sensitive native parity remains follow-up work for Apple, x86, gfx1151 and gfx1201. NVIDIA attention and quantized routes and the other E2E-REAL-6 families remain open. No performance claim or route promotion.

Evidence: Super-Bear (RTX 5070, sm_120) passed the 20-test `test_nvidia_tensor_program.py` suite, including exact-device resident execution. Seven eager matmul/dynamic-M projection checks passed; 11 unrelated rows were deselected. The test uses a bias/ReLU value chosen so early fp16 rounding flips the activation result. The gfx1201 packaging test could not run on Super-Bear because its selected `tessera-opt` lacks the ROCm executable pass; rerun the owning gfx1201 suite on Tajasarus.
<!-- entry-fields:end -->


### 2026-09-30 — NVIDIA bounded dynamic-M resident RMSNorm-to-matmul

Owner: [E2E-REAL-6](INTEGRATED_COMPILER_PLAN.md#e2e-real-6)

PRs: [#890](https://github.com/gstoner/tessera/pull/890); sync SM120-RMSNORM-MATMUL-DYNAMIC-M-2026-09-30.

Outcome: A public Graph RMSNorm -> matmul pair now projects one bounded M extent through Schedule, Tile and native packaging on sm_120. fp16 and bf16 producers and consumers accept active row prefixes within the package bound, reuse the same intermediate allocation and image, and preserve same-stream completion. The existing admission checks still refuse fused epilogues because their operands are absent from this resident ABI. This slice also advances W1.1's NVIDIA producer-to-matmul edge proof.

Remaining: dynamic M and N cannot be combined in one package; only row-major fp16/bf16 is proved. Tajasaurus RX 9070 XT (gfx1201) rebuilt tessera-opt from this PR head and passed the owning resident suite 18/18, revalidating the shared bounded-M transformation. More NVIDIA tensor-valued producers remain under W1.1. CUDA-event timing variation is too high for a performance claim or route promotion.

Evidence: Super-Bear RTX 5070 (sm_120) passed the focused NVIDIA tensor-program and fp16 epilogue regression selection (25 passed), plus Ruff. A clean-source 31-sample packet checks active M=64 and 128 under bound 128; producer/consumer max errors are 0/2.38e-6 and 9.77e-4/2.38e-6. Same-allocation and stable-image checks pass. Stage medians are 11.76/12.74 us at M=64 and 16.42/13.58 us at M=128, but CV ranges from 9.7% to 25.8%; these are diagnostics only. The Tajasaurus 18/18 rerun establishes current gfx1201 correctness, with no new gfx1201 timing claim. [Packet](../../../benchmarks/baselines/sm120_rmsnorm_matmul_edge_20260930/README.md).
<!-- entry-fields:end -->

### 2026-09-30 — bounded dynamic-K resident RMSNorm-to-matmul

Owner: [E2E-REAL-6](INTEGRATED_COMPILER_PLAN.md#e2e-real-6)

PRs: [#891](https://github.com/gstoner/tessera/pull/891); sync `E2E-REAL-6-RESIDENT-DYNAMIC-K-2026-09-30`.

Outcome: The resident producer → consumer package contract admits an active K prefix within a fixed package bound while preserving package reuse, producer-to-consumer buffer residency, and checked runtime leading dimensions. The gfx1201 public Graph → Schedule → Tile route is numerically proved on Tajasaurus for fp16 and bf16 storage at K=128/192/256 under bound 256 (20/20 focused fp16 tests plus a correctness-gated BF16 device packet). The NVIDIA route is separately proved on Super-Bear sm_120 from public `from_text` RMSNorm/matmul traces for fp16 and bf16 at K=7/11/16 (4/4 focused tests), with an additional fp16 benchmark at M=128, K-bound=256, N=256. Both packets check the stable resident allocation and separately record producer/consumer device-event timing. Timing variability remains too high for a speedup claim or route promotion. Apple/x86 plans record architecture-specific non-applicability; no sibling execution claims are inferred.

Remaining: dynamic K combined with M/N; broader dtype/layout coverage; and follow-on producer migrations under W1.1.

Evidence: [gfx1201 dynamic-K method and packet](../../../benchmarks/baselines/gfx1201_resident_dynamic_k_20260930/README.md); [gfx1201 BF16 packet](../../../benchmarks/baselines/gfx1201_resident_dynamic_k_20260930/dynamic_k_bf16.json); [sm_120 packet](../../../benchmarks/baselines/sm120_rmsnorm_matmul_edge_20260930/dynamic_k_sm120.json).

<!-- entry-fields:end -->

### 2026-09-30 — padded host-view ingress on the resident edge

Owner: [E2E-REAL-6](INTEGRATED_COMPILER_PLAN.md#e2e-real-6)

PRs: [#891](https://github.com/gstoner/tessera/pull/891); sync `E2E-REAL-6-RESIDENT-STRIDED-INGRESS-2026-09-30`.

Outcome: The paired resident RMSNorm → matmul route accepts padded and sliced
host views, normalizing the source to compact row-major storage and RHS to
compact column-major storage before device upload. CUDA staging allocations
remain owned until a successful stream synchronization. This keeps device
leading dimensions aligned with the checked ABI and makes host packing explicit.
The gfx1201 path already normalized host views and now has exact-device
regressions that exercise them.

Remaining: physical noncompact device layouts are outside this ingress contract.
Combined dynamic extents and broader producer/layout coverage remain open.
Apple and x86 have no consumer of this CUDA resident ABI; their execution
obligations remain independent. No performance claim or route promotion.

Evidence: Tajasaurus RX 9070 XT (gfx1201) passed 20/20 resident tests, including
padded source/RHS bounded-K fp16 and bf16 cases. Super-Bear RTX 5070 (sm_120)
passed 28/28 tensor-program tests, including both dtypes, padded views, numerical
oracle checks, stable images, and same-allocation producer/consumer execution.
A host-free fake-runtime regression verifies CUDA upload staging survives failed
synchronization and is released only after successful synchronization. Device
event timings keep producer and consumer separate and exclude host packing and
upload; the gfx1201 probe showed high variation at K=128/256, while sm_120
consumer K=128 also varied sharply. These results are diagnostic only. [Packets](../../../benchmarks/baselines/resident_strided_ingress_20260930/README.md).
<!-- entry-fields:end -->

### 2026-09-30 — paired bounded dynamic M+K resident RMSNorm-to-matmul

Owner: [E2E-REAL-6](INTEGRATED_COMPILER_PLAN.md#e2e-real-6)

PRs: [#891](https://github.com/gstoner/tessera/pull/891); sync `E2E-REAL-6-RESIDENT-DYNAMIC-MK-2026-09-30`.

Outcome: A shared Graph projection represents bounded dynamic M and K together,
and both gfx1201 and sm_120 resident RMSNorm → matmul packages consume those
bounds through Schedule, Tile, and checked native launch descriptors. Dynamic N
remains separate. Padded/sliced host views normalize to the compact device ABI;
the CUDA session retains upload staging through successful synchronization.

Remaining: Apple and x86 need independent resident package/runtime consumers
before claiming parity. Dynamic N combined with M/K, broader storage/layout
envelopes, and remaining W1.1 producers are still open. No performance or
selector promotion.

Evidence: the gfx1201 suite passed 22/22 and the sm_120 tensor-program suite
passed 30/30, including fp16/bf16 numerical parity, package reuse, and resident
buffer checks. Five host-free Graph projection tests passed. Exact-device
benchmark packets check three active (M,K) pairs on each target and record
separate producer/consumer events; multiple CVs are high, so timings support
attribution only. [Packets](../../../benchmarks/baselines/resident_dynamic_mk_20260930/README.md).
<!-- entry-fields:end -->

### 2026-09-30 — paired bounded dynamic M/N/K resident RMSNorm-to-matmul

Owner: [E2E-REAL-6](INTEGRATED_COMPILER_PLAN.md#e2e-real-6)

PRs: [#891](https://github.com/gstoner/tessera/pull/891); sync E2E-REAL-6-RESIDENT-DYNAMIC-MNK-2026-09-30.

Outcome: The shared Graph projection composes bounded M, N, and K, and both
native resident consumers carry the projected extents through Schedule, Tile,
and checked launch descriptors. Exact-device tests passed for fp16 and bf16 on
Tajasaurus gfx1201 and Super-Bear sm_120. Each route reuses package images and
the producer intermediate while accepting padded host views.

Remaining: Apple and x86 have no resident package/runtime consumer for this
edge. Wider physical device layouts and further W1.1 producer migrations remain
separate work. No selector or performance promotion.

Evidence: Host-free bounded-axis projection tests cover M/N, N/K, and M/N/K.
The gfx1201 focused joint-M/N/K exact-device test passed 2/2 dtype rows. Its
clean-source fp16 and bf16 packets each record 100 separate HIP-event samples
per stage at active M/N/K=(64,128,128),(96,192,192),(128,256,256). The sm_120
focused test passed 2/2 dtype rows; its clean-source fp16 and bf16 packets
each record 31 separate CUDA-event samples per stage at
(256,256,128),(384,384,192),(512,512,256). Correctness, stable image identity,
and same-allocation checks passed. gfx1201 event variation remains high,
especially for bf16; these measurements support stage attribution only.
[Packets](../../../benchmarks/baselines/resident_dynamic_mnk_20260930/README.md).
<!-- entry-fields:end -->

## NVIDIA-NVFP4-SCHEDULE-2026-09: scheduled NVFP4 native package

The SM120 NVFP4 scaled-matmul contract now traverses Graph IR, verified
Schedule IR, replayed Tile IR, NVIDIA Target IR and PTX. The Graph dialect
now registers its NVFP4 type and the scaled-matmul verifier checks static
rank-2 operands, f32 output, K16 UE4M3 scale vectors, and the named physical
contract. Exact Super-Bear RTX 5070 execution passes 16x8x64, 33x19x129, and
ragged 7x5x31 against decoded NV_E2M1/UE4M3 NumPy (max absolute error 0.0).
The packet reports CUDA-event kernel and runtime.launch end-to-end timing
separately. Micro-shape timings are diagnostic; no selector promotion.
Apple, ROCm, and x86 sibling plans record the need for explicit target
rejection or independent backend lowering. Exact packet:
benchmarks/baselines/nvidia_sm120_nvfp4_scheduled_20260930/.


### 2026-10-01 — W1.1: SM120 softmax producer reaches typed-fragment matmul

Owner: [W1.1](INTEGRATED_COMPILER_PLAN.md#w11)

PRs: Uncommitted; original synchronization labels are preserved below.

Outcome: The checked resident tensor-to-matmul package now admits a shape-preserving fp16/bf16 Schedule softmax producer in addition to RMSNorm. Runtime scalar binding follows descriptor metadata; the CUDA resident dispatcher and launch bridge have an explicit f16/bf16 softmax path that launches on the existing stream directly from device pointers. Public from_text Graph packages flow through Schedule and Tile; the consumer's tile.view feeds typed fragments and SM120 MMA.

Remaining: exact-device static numerical tests cover fp16 and bf16 at M/K/N=16/64/8; the original timing packet covers fp16 only. Its 15-sample CUDA-event packet reports 17.24 us producer median (5.09% CV), 16.21 us consumer median (5.05% CV), and 712.15 ms resident execute-through-sync including uploads and launch overhead. A bounded dynamic-K test now reuses the same packages at active K=7, 11, and 16 under bound 16 for both dtypes. Its additional fp16 K=7 packet reports 7.14/9.99 us producer/consumer medians, but CV is 24.5%/83.9%, so the measurements are diagnostic only. Generic tensor-valued LowerMatmulToTileMMA / LowerKReductionAddToTileMMA migration remains open. No performance promotion.

Evidence: exact RTX 5070 tests check public frontend compilation, typed-fragment consumer IR, resident allocation separation, and independent numerical agreement. The complete `test_nvidia_tensor_program.py` suite passes 37/37 on Super-Bear with the CUDA 13.4 libraries selected. [Static packet](../../../benchmarks/baselines/nvidia_sm120_w1_typed_tensor_edge_20261001/softmax_matmul.json); [bounded dynamic-K packet](../../../benchmarks/baselines/nvidia_sm120_w1_typed_tensor_edge_20261001/softmax_matmul_dynamic_k16_active7.json).

<!-- entry-fields:end -->

Recorded detail (preserved; metadata formatting correction):


Owner: [W1.1](INTEGRATED_COMPILER_PLAN.md#w11); sync W11-SM120-SOFTMAX-MATMUL-EDGE-2026-10-01.

Outcome: The checked resident tensor-to-matmul package now admits a shape-preserving fp16/bf16 Schedule softmax producer in addition to RMSNorm. Runtime scalar binding follows descriptor metadata; the CUDA resident dispatcher and launch bridge have an explicit f16/bf16 softmax path that launches on the existing stream directly from device pointers. Public from_text Graph packages flow through Schedule and Tile; the consumer's tile.view feeds typed fragments and SM120 MMA.

Remaining: exact-device static numerical tests cover fp16 and bf16 at M/K/N=16/64/8; the original timing packet covers fp16 only. Its 15-sample CUDA-event packet reports 17.24 us producer median (5.09% CV), 16.21 us consumer median (5.05% CV), and 712.15 ms resident execute-through-sync including uploads and launch overhead. A bounded dynamic-K test now reuses the same packages at active K=7, 11, and 16 under bound 16 for both dtypes. Its additional fp16 K=7 packet reports 7.14/9.99 us producer/consumer medians, but CV is 24.5%/83.9%, so the measurements are diagnostic only. Generic tensor-valued LowerMatmulToTileMMA / LowerKReductionAddToTileMMA migration remains open. No performance promotion.

Evidence: exact RTX 5070 tests check public frontend compilation, typed-fragment consumer IR, resident allocation separation, and independent numerical agreement. The complete `test_nvidia_tensor_program.py` suite passes 37/37 on Super-Bear with the CUDA 13.4 libraries selected. [Static packet](../../../benchmarks/baselines/nvidia_sm120_w1_typed_tensor_edge_20261001/softmax_matmul.json); [bounded dynamic-K packet](../../../benchmarks/baselines/nvidia_sm120_w1_typed_tensor_edge_20261001/softmax_matmul_dynamic_k16_active7.json).


### 2026-10-01 — NVIDIA sm_120 saved-LSE attention backward timing

Owner: [E2E-REAL-6](INTEGRATED_COMPILER_PLAN.md#e2e-real-6)

PRs: Uncommitted; original synchronization labels are preserved below.

Outcome: The compiler-owned saved-LSE forward package produced O and row LSE;

Remaining: The recorded limitations below remain in force; this schema correction changes no completion or support state.

Evidence: Original architecture-specific receipts, packet links and timing qualifications are preserved below.

<!-- entry-fields:end -->

Recorded detail (preserved; metadata formatting correction):


Owner: E2E-REAL-6 / NVIDIA-ATTENTION-LSE-BACKWARD; sync
NVIDIA-ATTENTION-LSE-BACKWARD-2026-10-01.

Outcome: The compiler-owned saved-LSE forward package produced O and row LSE;
the saved-LSE backward package consumed that checkpoint and ran natively on
the RTX 5070. Saved-LSE and recompute gradients both matched an independent
causal-attention oracle (maximum absolute gradient error 1.49e-7). A second
benchmark records device-event and host-array end-to-end distributions
separately. Device-event median was 1.089 ms at 0.07% CV. End-to-end CV was
48%, including a 6.66 ms outlier, and is not suitable for a performance claim.
The current bridge times backward with persistent scratch allocations but does
not yet expose the explicit resident-stream API used by the producer-to-matmul
edge. No selector promotion. [Packet](../../../benchmarks/baselines/nvidia_attention_lse_e2e_20261001/backward_recheck_20261001.json).



### 2026-10-01 — resident saved-LSE attention backward on sm_120

Owner: [E2E-REAL-6](INTEGRATED_COMPILER_PLAN.md#e2e-real-6)

PRs: Uncommitted; original synchronization labels are preserved below.

Outcome: The NVIDIA launch bridge now accepts caller-owned device pointers and

Remaining: The recorded limitations below remain in force; this schema correction changes no completion or support state.

Evidence: Original architecture-specific receipts, packet links and timing qualifications are preserved below.

<!-- entry-fields:end -->

Recorded detail (preserved; metadata formatting correction):


Owner: E2E-REAL-6 / NVIDIA-ATTENTION-LSE-BACKWARD; sync
NVIDIA-ATTENTION-LSE-BACKWARD-2026-10-01.

Outcome: The NVIDIA launch bridge now accepts caller-owned device pointers and
a caller CUDA stream for the existing saved-LSE attention-backward descriptor.
The exact RTX 5070 packet ran saved-LSE forward and backward natively on one
stream; resident dQ/dK/dV matched the independent oracle with maximum absolute
error 1.49e-7. CUDA-event backward median was 1.088 ms (0.02% CV) across nine
samples of 20 launches. Host-array end-to-end median was 2.292 ms (4.03% CV).
These micro-shape measurements support timing-domain attribution, not a speedup
or selector promotion. Broader attention shapes, dtypes, and fused-bias routes
remain open. [Packet](../../../benchmarks/baselines/nvidia_attention_lse_e2e_20261001/backward_resident_20261001.json).





### 2026-10-01 — W1.1 named SM120 Graph-to-Schedule-to-Tile integration

Owner: [W1.1](INTEGRATED_COMPILER_PLAN.md#w11)

PRs: Uncommitted; original synchronization labels are preserved below.

Outcome: W1.1 named SM120 Graph-to-Schedule-to-Tile integration: implementation and evidence are preserved in the recorded detail below.

Remaining: The recorded limitations below remain in force; this schema correction changes no completion or support state.

Evidence: Original architecture-specific receipts, packet links and timing qualifications are preserved below.

<!-- entry-fields:end -->

Recorded detail (preserved; metadata formatting correction):


Owner: W1.1 / NVIDIA fragment-producer closure; sync
`W1.1-SM120-GRAPH-SCHEDULE-TILE-PIPELINE-2026-10-01`.

The registered `--tessera-nvidia-pipeline-sm120` now runs PM verification,
Graph-to-Schedule, and Schedule-to-Tile before residual generic lowering.
Its MLIR fixture confirms a Graph matmul reaches pointer-backed Tile views,
typed fragment packs, `tile.mma`, and fragment unpack. The new pass linkage
requires `TesseraPasses` to link the programming-model library. The metadata
verifier now validates module-level drop declarations across erased child
scopes, with a regression fixture preserving stale-declaration rejection.

Exact-device validation on Super-Bear RTX 5070 ran an fp16 RMSNorm-to-matmul
package at M/K/N=64/128/128. Both producer and consumer executed as
`native_gpu`, matched independent numerical references (max errors 0 and
4.77e-6), and reused the caller-owned intermediate. Compiler evidence
included Schedule and Tile digests, `nvvm.mma.sync`, and PTX
`mma.sync.aligned.m16n8k16`. Five-sample CUDA-event timing CV was too high
for promotion; this packet supports correctness and stage attribution only.
The remaining generic tensor-valued constructors and other producer families
stay open. Apple, ROCm, and x86 plans record sibling outcomes as not applicable
for this SM120 target route.

[Packet](../../../benchmarks/baselines/nvidia_sm120_integrated_pipeline_20261001/README.md).



### 2026-10-01 — gfx1201 scheduled attention shape-free image cache

Owner: [E2E-REAL-6](INTEGRATED_COMPILER_PLAN.md#e2e-real-6)

PRs: Uncommitted; original synchronization labels are preserved below.

Outcome: gfx1201 scheduled attention shape-free image cache: implementation and evidence are preserved in the recorded detail below.

Remaining: The recorded limitations below remain in force; this schema correction changes no completion or support state.

Evidence: Original architecture-specific receipts, packet links and timing qualifications are preserved below.

<!-- entry-fields:end -->

Recorded detail (preserved; metadata formatting correction):


Owner: E2E-REAL-6 / ROCm cache-key follow-up; sync
`E2E-REAL-6-GFX1201-SCHEDULED-ATTENTION-IMAGE-CACHE-2026-10-01`.

On Tajasaurus RX 9070 XT (gfx1201), the exact-device recorder compiled one
HSACO for four query lengths (17, 23, 31, 65). All launches were native and
matched the independent streaming-attention reference with max absolute error
at most 6.51e-5. Image identity remained constant while schedule digests and
shape guards varied. The gated unit test also checks a changed bias contract
causes a cache miss. Cold package time was 316 ms; warm first-use packages
were 50.5–52.8 ms and repeated package medians were 37.1–38.3 ms. These are
host compile/package timings; the recorder did not measure device kernel time.
Scheduled matmul image identity remains open outside the gfx1151 static
f16/bf16 register-route envelope. [Packet](../../../benchmarks/baselines/rocm_scheduled_attention_cache_20261001/README.md).



### 2026-10-01 — scheduled attention image reuse on gfx1151 and gfx1201

Owner: [E2E-REAL-6](INTEGRATED_COMPILER_PLAN.md#e2e-real-6)

PRs: Uncommitted; original synchronization labels are preserved below.

Outcome: scheduled attention image reuse on gfx1151 and gfx1201: implementation and evidence are preserved in the recorded detail below.

Remaining: The recorded limitations below remain in force; this schema correction changes no completion or support state.

Evidence: Original architecture-specific receipts, packet links and timing qualifications are preserved below.

<!-- entry-fields:end -->

Recorded detail (preserved; metadata formatting correction):


Owner: E2E-REAL-6 / ROCm cache-key follow-up; sync
`E2E-REAL-6-ROCM-SCHEDULED-ATTENTION-CACHE-2026-10-01`.

Princess-Luna gfx1151 and Tajasaurus gfx1201 each packaged four scheduled
attention query lengths (17, 23, 31, 65) with one HSACO compilation per
architecture. Native launches matched the independent streaming-attention
oracle with maximum absolute error 6.51e-5. Each architecture retained its own
image digest across the four shapes, while Schedule digests and descriptor
guards remained shape-specific. The gfx1201 gated test also verifies a
changed-bias cache miss.

Cold package costs were 557.6 ms (gfx1151) and 316.2 ms (gfx1201); warm
first-use package costs were 70.7–72.7 ms and 50.5–52.8 ms. Repeated package
medians were 54.2–56.2 ms and 37.1–38.3 ms. These are host compilation and
package timings, not GPU kernel timings. The recorder, compiler, schedule,
runtime, and test helper sources were hash-matched across the two owning
hosts. Scheduled matmul shape-independent image identity remains open beyond
the narrow gfx1151 static f16/bf16 register route.
[Packets](../../../benchmarks/baselines/rocm_scheduled_attention_cache_20261001/README.md).



### 2026-10-01 — gfx1201 scheduled matmul cache boundary experiment

Owner: [E2E-REAL-6](INTEGRATED_COMPILER_PLAN.md#e2e-real-6)

PRs: Uncommitted; original synchronization labels are preserved below.

Outcome: Tested whether the static, unfused f16 scheduled matmul image could

Remaining: The recorded limitations below remain in force; this schema correction changes no completion or support state.

Evidence: Original architecture-specific receipts, packet links and timing qualifications are preserved below.

<!-- entry-fields:end -->

Recorded detail (preserved; metadata formatting correction):


Owner: [E2E-REAL-6](../backend/rocm/todo.md); sync
`ROCM-MATMUL-CACHE-GFX1201-2026-10-01`.

Outcome: Tested whether the static, unfused f16 scheduled matmul image could
reuse the existing Directive-input compilation route already measured for
gfx1151. On the exact gfx1201 host, LLVM 23.1.1 aborted during HSACO
serialization with `Cannot select: llvm.amdgcn.wmma.f32.16x16x16.f16`.
The normal Tile-input gfx1201 scheduled package still passed the existing
three-shape numerical test. The experiment was removed from the package
predicate and does not claim gfx1201 shape-independent matmul image reuse.

Next: trace the differing Target-to-binary lowering and target-feature setup,
then rerun image reuse and numerical parity on gfx1201 before widening the
route. The owning ROCm plan records this boundary. No other backend contract
changed.


### 2026-10-01 — gfx1201 scheduled matmul image reuse

Owner: [E2E-REAL-6](INTEGRATED_COMPILER_PLAN.md#e2e-real-6)

PRs: Uncommitted; original synchronization labels are preserved below.

Outcome: gfx1201 scheduled matmul image reuse: implementation and evidence are preserved in the recorded detail below.

Remaining: The recorded limitations below remain in force; this schema correction changes no completion or support state.

Evidence: Original architecture-specific receipts, packet links and timing qualifications are preserved below.

<!-- entry-fields:end -->

Recorded detail (preserved; metadata formatting correction):


Owner: E2E-REAL-6 / ROCm cache-key follow-up; sync
`E2E-REAL-6-GFX1201-SCHEDULED-MATMUL-IMAGE-CACHE-2026-10-01`.

The earlier Directive-input gfx1201 experiment exposed that the matmul family
plugin selected the legacy direct emitter (`viaTile=false`), which reached an
unsupported legacy WMMA intrinsic. The registered generator now selects the
typed Tile route for gfx1201 Directive input as well as explicit Tile input.
The package cache predicate admits the validated gfx1201 static fp16/bf16
register envelope. On Tajasaurus, three f16 shapes reused one HSACO and entry
symbol with shape-specific Schedule digests and guards; exact native launches
matched NumPy within `3.58e-7`. The exact-device cache test passed for fp16 and
bf16. The packet reports package construction, HIP-event kernel time, and
end-to-end time separately. Micro-shape timing is diagnostic only and is not
promotion evidence. Dynamic, split-K, fused-epilogue, LDS, wider-shape, and
gfx1151 cases remain open.

[Packet](../../../benchmarks/baselines/rocm_gfx1201_matmul_shape_key_20261001/README.md).


### 2026-10-01 — gfx1201 NVFP4 joint code/scale ingest requantization

Owner: [ROCM-NVFP4-INGEST-1](INTEGRATED_COMPILER_PLAN.md#rocm-nvfp4-ingest-1)

PRs: Uncommitted; original synchronization labels are preserved below.

Outcome: gfx1201 NVFP4 joint code/scale ingest requantization: implementation and evidence are preserved in the recorded detail below.

Remaining: The recorded limitations below remain in force; this schema correction changes no completion or support state.

Evidence: Original architecture-specific receipts, packet links and timing qualifications are preserved below.

<!-- entry-fields:end -->

Recorded detail (preserved; metadata formatting correction):


Owner: ROCM-NVFP4-INGEST-1; sync
ROCM-NVFP4-INGEST-1-JOINT-REQUANTIZATION-2026-10-01.

The ingest converter previously copied source E2M1 codes and changed only the
two E4M3/K16 scales to one E8M0/K32 scale. It now jointly selects signed
destination E2M1 codes and a K32 E8M0 exponent by decoded-weight SSE over a
bounded candidate window around the weighted-scale seed. The numerical policy
declares both scale and code requantization. Eleven focused host tests pass,
including lower-SSE behavior than the preserve-code baseline.

On Tajasarus RX 9070 XT (gfx1201), the exact native Graph -> Schedule -> Tile ->
Target package passed numerical checks before and after timing. For the
deterministic synthetic 17x19x64 gate/up case, relative RMS improved from
0.36791/0.42102 (preserved source codes) to 0.09064/0.06701; the image digest
remained unchanged. Ingest time was 3.44 ms; package construction 108.95 ms;
persistent HIP-event median 4.54 us; runtime.launch median 2.45 ms. The packet
distinguishes conversion and execution costs and records dirty-source
provenance.

No original BF16 checkpoint was present in the available scratch stores.
Comparison against source BF16 and direct BF16-to-MXFP4 remains open. The
small synthetic conversion time does not project to model-sized checkpoints,
and bounded exponent search is not claimed globally optimal. No shared IR,
runtime ABI, or dtype registry changed. Apple, NVIDIA, and x86 assessments are
not applicable because this conversion and its package are ROCm-specific.

[Packet and limitations](../../../benchmarks/baselines/rocm_nvfp4_ingest_20261001/README.md).


### 2026-10-01 — five-slice exact-device recheck

Owner: [W1.1](INTEGRATED_COMPILER_PLAN.md#w11)

PRs: Uncommitted; original synchronization labels are preserved below.

Outcome: five-slice exact-device recheck: implementation and evidence are preserved in the recorded detail below.

Remaining: The recorded limitations below remain in force; this schema correction changes no completion or support state.

Evidence: Original architecture-specific receipts, packet links and timing qualifications are preserved below.

<!-- entry-fields:end -->

Recorded detail (preserved; metadata formatting correction):


Owner: W1.1, NVIDIA saved-LSE attention, ROCm cache-key validation.

On Super-Bear RTX 5070, the five focused softmax-to-matmul producer-edge
checks passed, including bounded dynamic K. A 21-sample recheck with 1,000
launches per sample matched the independent softmax and matmul oracles (max
absolute errors 0 and 1.49e-8). Timing remained noisy (producer/consumer CV
36.8%/37.0%), so it is diagnostic only. The saved-LSE checkpoint suite passed
both forward/backward cases. The benchmark README records the recheck and
unstable timing result.

The Tajasaurus WSL session currently exposes no gfx1201 agent: `rocminfo`
reports only the CPU, `/dev/kfd` is absent, and ROCm reports its driver
uninitialized. The host-independent cache-key unit tests pass (26 passed); six
ROCm device cases were skipped by the explicit no-device gate. The existing
Tajasaurus exact-device cache packet remains the prior evidence; this session
adds no new gfx1201 execution or timing claim. Generic NVIDIA tensor-valued
Tile constructors and the BF16-checkpoint NVFP4 comparison remain open.


### 2026-10-01 — gfx1201 cache-key exact-device recheck

Owner: [E2E-REAL-6](INTEGRATED_COMPILER_PLAN.md#e2e-real-6)

PRs: Uncommitted; original synchronization labels are preserved below.

Outcome: gfx1201 cache-key exact-device recheck: implementation and evidence are preserved in the recorded detail below.

Remaining: The recorded limitations below remain in force; this schema correction changes no completion or support state.

Evidence: Original architecture-specific receipts, packet links and timing qualifications are preserved below.

<!-- entry-fields:end -->

Recorded detail (preserved; metadata formatting correction):


Owner: E2E-REAL-6 / ROCm scheduled matmul image cache; sync
`E2E-REAL-6-GFX1201-SCHEDULED-MATMUL-IMAGE-CACHE-2026-10-01`.

Tajasaurus exact-device tests passed 31 cases, including gfx1201 fp16/bf16
shape reuse; the lone skipped row was gfx1151-only. The refreshed packet
recompiled three static shapes to one HSACO digest and entry symbol, retained
shape-specific Schedule digests/guards, and passed independent NumPy checks
(max absolute errors 1.19e-7, 3.58e-7, 1.79e-7). It records compiler SHA-256,
separate Schedule/package/HIP-event/end-to-end timings, and dirty-source
provenance. The tiny-shape timing is diagnostic only; no route promotion.
[Packet](../../../benchmarks/baselines/rocm_gfx1201_matmul_shape_key_20261001/gfx1201_recheck_20261001.json).


### 2026-10-01 — gfx1201 NVFP4 ingest exact-device recheck

Owner: [ROCM-NVFP4-INGEST-1](INTEGRATED_COMPILER_PLAN.md#rocm-nvfp4-ingest-1)

PRs: Uncommitted; original synchronization labels are preserved below.

Outcome: gfx1201 NVFP4 ingest exact-device recheck: implementation and evidence are preserved in the recorded detail below.

Remaining: The recorded limitations below remain in force; this schema correction changes no completion or support state.

Evidence: Original architecture-specific receipts, packet links and timing qualifications are preserved below.

<!-- entry-fields:end -->

Recorded detail (preserved; metadata formatting correction):


Owner ROCM-NVFP4-INGEST-1; sync
`ROCM-NVFP4-INGEST-1-2026-10-01`.

Tajasaurus reran the deterministic synthetic gate/up case with the current
branch-built compiler. The new converter again improved relative RMS from
0.36791/0.42102 to 0.09064/0.06701, matched the independent decoded-weight
package oracle, and produced the same HSACO digest. Ingest measured 2.97 ms,
packaging 237.25 ms, persistent HIP events 4.40 us, and `runtime.launch` 2.46
ms median. Two host E2E samples were near 10 ms; the packet remains diagnostic.
No source-BF16 checkpoint was available, so model-checkpoint comparison stays
open. [Packet](../../../benchmarks/baselines/rocm_nvfp4_ingest_20261001/gfx1201_nvfp4_joint_recheck_20261001.json).



### 2026-10-01 — gfx1201 NVFP4 ragged-K exact-device

Owner: [ROCM-NVFP4-INGEST-1](INTEGRATED_COMPILER_PLAN.md#rocm-nvfp4-ingest-1)

PRs: Uncommitted; original synchronization labels are preserved below.

Outcome: gfx1201 NVFP4 ragged-K exact-device: implementation and evidence are preserved in the recorded detail below.

Remaining: The recorded limitations below remain in force; this schema correction changes no completion or support state.

Evidence: Original architecture-specific receipts, packet links and timing qualifications are preserved below.

<!-- entry-fields:end -->

Recorded detail (preserved; metadata formatting correction):


Owner: ROCM-NVFP4-INGEST-1; sync `ROCM-NVFP4-INGEST-1-2026-10-01`.

Tajasaurus RX 9070 XT passed the synthetic M/N/K=200/2048/1536 gate/up case
through Graph -> Schedule -> Tile -> gfx1201 Target packaging. Output equality
passed before timing and after every host launch sample. The package used a
128x2 grid and 256-thread workgroup. Synthetic gate/up relative RMS was
0.09396/0.07200, versus 0.35583/0.41749 for preserving source codes and
requantizing only scales.

CPU ingest took 6.703 s; native package construction took 217.69 ms;
persistent HIP-event median was 54.10 us; host `runtime.launch` median was
6.17 ms with two initial samples near 13 ms. Event samples ranged 47.68–56.99
us. These synthetic timings are diagnostic and do not support promotion. The
run establishes a larger ragged-K package envelope, while model-checkpoint
quality and source-BF16 comparison remain open. No shared IR or ABI changed;
Apple, NVIDIA, and x86 have no corresponding ROCm NVFP4 conversion path.
[Packet](../../../benchmarks/baselines/rocm_nvfp4_ingest_20261001/gfx1201_nvfp4_m200n2048k1536_20261001.json).



### 2026-10-01 — W1.1 SM120 LayerNorm producer edge

Owner: [W1.1](INTEGRATED_COMPILER_PLAN.md#w11)

PRs: Uncommitted; original synchronization labels are preserved below.

Outcome: W1.1 SM120 LayerNorm producer edge: implementation and evidence are preserved in the recorded detail below.

Remaining: The recorded limitations below remain in force; this schema correction changes no completion or support state.

Evidence: Original architecture-specific receipts, packet links and timing qualifications are preserved below.

<!-- entry-fields:end -->

Recorded detail (preserved; metadata formatting correction):


Owner: W1.1 / NVIDIA fragment-producer closure; sync
W1.1-SM120-LAYERNORM-MATMUL-EDGE-2026-10-01.

The resident tensor-program contract now accepts affine-free, shape-preserving
last-axis LayerNorm through its existing scheduled norm package ABI. Public
from_text LayerNorm and matmul graphs passed exact RTX 5070 execution for
fp16 and bf16. The compiler route preserves kind=layernorm in the
scheduled norm Tile kernel and emits the consumer tile.view and typed
fragment pack. Both producer and consumer receipts report native_gpu.

The fp16 M/K/N=256/256/256 packet passed independent references with maximum
absolute errors 9.77e-4 (LayerNorm) and 1.53e-5 (matmul). Producer and consumer
ran on one resident intermediate allocation; seven samples of 500 resident
launches measured 71.64 us and 10.06 us median, with CV 0.008% and 1.68%.
The packages report 22 and 40 registers per thread with zero spill bytes.
These are stage-attribution measurements only; no selector promotion.

This changes NVIDIA package admission and tests only. It adds no shared Graph
op, Schedule schema, runtime ABI, or sibling-backend physical schedule.
The two generic legacy tensor-valued Tile MMA constructors remain open.
[Packet and producer history](../../../benchmarks/baselines/nvidia_sm120_w1_typed_tensor_edge_20261001/README.md).



### 2026-10-01 — gfx1201 NVFP4 vectorized ragged-K ingest

Owner: [ROCM-NVFP4-INGEST-1](INTEGRATED_COMPILER_PLAN.md#rocm-nvfp4-ingest-1)

PRs: Uncommitted; original synchronization labels are preserved below.

Outcome: gfx1201 NVFP4 vectorized ragged-K ingest: implementation and evidence are preserved in the recorded detail below.

Remaining: The recorded limitations below remain in force; this schema correction changes no completion or support state.

Evidence: Original architecture-specific receipts, packet links and timing qualifications are preserved below.

<!-- entry-fields:end -->

Recorded detail (preserved; metadata formatting correction):


Owner: ROCM-NVFP4-INGEST-1; sync
ROCM-NVFP4-INGEST-1-2026-10-01.

The NVFP4 -> MXFP4 host converter now evaluates batches of K32 blocks with
vectorized NumPy arrays instead of Python dispatch for each row and block.
The focused gfx1201 ingest suite passed 11 tests, including exact comparison
against a scalar reference for packed codes, scale exponents, and error
metadata.

On Tajasaurus, the same synthetic M/N/K=200/2048/1536 ragged-K case passed
numerical checks before timing and after every launch sample. CPU ingest fell
from 6702.96 ms to 487.31 ms (13.76x). Gate/up relative RMS was unchanged at
0.09396468/0.07199587, and the preserve-code baseline remained
0.35582517/0.41749312. Target IR, Tile IR, and HSACO digests were identical to
the scalar run. Package construction was 238.30 ms; persistent HIP-event
median was 53.51 us; host runtime.launch median was 6.13 ms. The GPU timings
are attribution only; this change makes no kernel-performance claim.

No shared IR, dtype, ABI, or sibling-backend contract changed. Real checkpoint
quality and source-BF16 comparison remain open.
[Packet and comparison](../../../benchmarks/baselines/rocm_nvfp4_ingest_20261001/gfx1201_nvfp4_m200n2048k1536_vectorized_20261001.json).


### 2026-10-02 — NVIDIA saved-LSE float64 correctness-gated benchmark

Owner: [E2E-REAL-6](INTEGRATED_COMPILER_PLAN.md#e2e-real-6)

PRs: Uncommitted; original synchronization labels are preserved below.

Outcome: NVIDIA saved-LSE float64 correctness-gated benchmark: implementation and evidence are preserved in the recorded detail below.

Remaining: The recorded limitations below remain in force; this schema correction changes no completion or support state.

Evidence: Original architecture-specific receipts, packet links and timing qualifications are preserved below.

<!-- entry-fields:end -->

Recorded detail (preserved; metadata formatting correction):


Owner: E2E-REAL-6 / NVIDIA attention LSE; sync NVIDIA-ATTENTION-LSE-ORACLE-PACKET-2026-10-02.

The exact-device saved-LSE checkpoint suite passed 2/2 on the RTX 5070. The
recorder previously smoke-launched its larger timing shapes without comparing
outputs or gradients. It now gates each saved and recompute timing row on
forward output, saved row-LSE, and dq/dk/dv checks against an independent
float64 scalar oracle. Three shapes passed, including ragged Sq/Sk=15/17;
maximum output, LSE, and gradient errors were 3.10e-8, 2.74e-7, and 9.53e-8.
The packet now records the actual GPU UUID and nvidia_sm120 target. Saved
forward/backward artifacts carry complete Graph/Schedule/Tile digests; the
recompute controls expose only partial provenance and are labeled separately.
Timing variation remains diagnostic; no performance promotion follows.

This changes NVIDIA benchmark evidence and its recorder only. No shared Graph,
Schedule, ABI, or sibling-backend physical schedule changed.
[Packet](../../../benchmarks/baselines/nvidia_attention_lse_e2e_20261001/saved_lse_recheck_20261002.json).


### 2026-10-01 — gfx1201 scheduled-attention cache source-fingerprinted recheck

Owner: [E2E-REAL-6](INTEGRATED_COMPILER_PLAN.md#e2e-real-6)

PRs: Uncommitted; original synchronization labels are preserved below.

Outcome: gfx1201 scheduled-attention cache source-fingerprinted recheck: implementation and evidence are preserved in the recorded detail below.

Remaining: The recorded limitations below remain in force; this schema correction changes no completion or support state.

Evidence: Original architecture-specific receipts, packet links and timing qualifications are preserved below.

<!-- entry-fields:end -->

Recorded detail (preserved; metadata formatting correction):


Owner: E2E-REAL-6 / ROCm scheduled attention; sync ROCM-SCHEDULED-ATTENTION-CACHE-RECHECK-2026-10-01.

On Tajasaurus, exact gfx1201 cache tests passed and four scheduled-attention
shapes reused one HSACO image with matching native numerical outputs. The
packet records the compiler revision and five source SHA-256 values. It
identifies the Tajasaurus runtime snapshot explicitly; that runtime differs
from Super-Bear because of NVIDIA-scoped edits, so this packet does not claim
byte-identical shared runtime sources. Host compile/package costs are recorded
separately from GPU execution, and no GPU speedup claim follows.

This changes no shared IR, ABI, or sibling physical schedule.
[Packet](../../../benchmarks/baselines/rocm_scheduled_attention_cache_20261001/gfx1201_recheck_20261001.json).


### 2026-10-02 — NVIDIA W1.1 typed-matmul exact-device route

Owner: [W1.1](INTEGRATED_COMPILER_PLAN.md#w11)

PRs: Uncommitted; original synchronization labels are preserved below.

Outcome: NVIDIA W1.1 typed-matmul exact-device route: implementation and evidence are preserved in the recorded detail below.

Remaining: The recorded limitations below remain in force; this schema correction changes no completion or support state.

Evidence: Original architecture-specific receipts, packet links and timing qualifications are preserved below.

<!-- entry-fields:end -->

Recorded detail (preserved; metadata formatting correction):


Owner: W1.1 / NVIDIA fragment-producer closure; sync NVIDIA-W1.1-TYPED-MATMUL-EDGE-2026-10-02.

A five-shape exact RTX 5070 typed-fragment test was present but uncollected
because its test function began with an underscore. Renaming it made pytest
exercise all five shapes. The test confirms pointer-backed tile.view,
typed fragment_pack/tile.mma, NVVM MMA lowering, and exact equality between
descriptor provenance and Graph/Schedule/ScheduleIR/Tile artifacts. All native
outputs matched the fp32 reference (maximum absolute error 4.77e-7). The
correctness-gated packet measures CUDA-event and end-to-end timings separately;
several rows are noisy and remain diagnostic.

NVIDIA scheduled-matmul descriptors now include graph_ir_digest and
schedule_ir_digest, which the artifact already possessed but the descriptor
omitted. No physical schedule, shared operation, dtype, or ABI changed.
[Packet](../../../benchmarks/baselines/nvidia_sm120_w1_typed_tensor_edge_20261001/typed_m16n8_20261002.json).



### 2026-10-02 — gfx1201 scheduled matmul image reuse recheck

Owner: [E2E-REAL-6](INTEGRATED_COMPILER_PLAN.md#e2e-real-6)

PRs: Uncommitted; original synchronization labels are preserved below.

Outcome: gfx1201 scheduled matmul image reuse recheck: implementation and evidence are preserved in the recorded detail below.

Remaining: The recorded limitations below remain in force; this schema correction changes no completion or support state.

Evidence: Original architecture-specific receipts, packet links and timing qualifications are preserved below.

<!-- entry-fields:end -->

Recorded detail (preserved; metadata formatting correction):


Owner: E2E-REAL-6 / ROCm matmul cache; sync ROCM-GFX1201-MATMUL-CACHE-RECHECK-2026-10-02.

Tajasaurus live and configured gfx1201 passed three static f16 Graph/Schedule/
Tile package launches. The shapes reused one HSACO and entry while preserving
shape guards and schedule digests; max absolute error against fp32 NumPy was
below 3.6e-7. HIP-event medians (8.64/14.60/15.36 us) are separated from
end-to-end medians (2.20/2.37/2.45 ms) and Schedule/package construction.
The 39.6 ms cold outlier makes timing diagnostic. This closes only the bounded
static register, k_unroll=1, split_k=1, unfused envelope; broader cache keys
and dynamic/split/fused/LDS routes remain open. No sibling backend or shared
IR contract changed.

[Packet](../../../benchmarks/baselines/rocm_gfx1201_matmul_shape_key_20261002/gfx1201_recheck_20261002.json).



### 2026-10-02 — gfx1201 scheduled matmul representative-size extension

Owner: [E2E-REAL-6](INTEGRATED_COMPILER_PLAN.md#e2e-real-6)

PRs: Uncommitted; original synchronization labels are preserved below.

Outcome: gfx1201 scheduled matmul representative-size extension: implementation and evidence are preserved in the recorded detail below.

Remaining: The recorded limitations below remain in force; this schema correction changes no completion or support state.

Evidence: Original architecture-specific receipts, packet links and timing qualifications are preserved below.

<!-- entry-fields:end -->

Recorded detail (preserved; metadata formatting correction):


Owner: E2E-REAL-6 / ROCm matmul cache; sync ROCM-GFX1201-MATMUL-CACHE-RECHECK-2026-10-02.

The source-matched recorder now checks six static f16 sizes from 16^3 through
512^3. Tajasaurus live/configured gfx1201 reused one HSACO and entry for every
shape, with a cold compile followed by five warm cache hits. All outputs
matched an independent fp32 NumPy product; maximum absolute error was 1.10e-5.
The packet separates seven-sample HIP-event kernel timings (8.60–28.76 us)
from end-to-end timings (2.20–6.51 ms), and records Schedule/package costs.
No performance promotion follows. Dynamic, split-K, fused, LDS-staged, and
other dtype/layout cache routes remain open.

[Packet](../../../benchmarks/baselines/rocm_gfx1201_matmul_shape_key_20261002/gfx1201_representative_recheck_final_20261002.json).



### 2026-10-02 — saved-LSE larger-shape SM120 execution and backward scaling

Owner: [E2E-REAL-6](INTEGRATED_COMPILER_PLAN.md#e2e-real-6)

PRs: Uncommitted; original synchronization labels are preserved below.

Outcome: saved-LSE larger-shape SM120 execution and backward scaling: implementation and evidence are preserved in the recorded detail below.

Remaining: The recorded limitations below remain in force; this schema correction changes no completion or support state.

Evidence: Original architecture-specific receipts, packet links and timing qualifications are preserved below.

<!-- entry-fields:end -->

Recorded detail (preserved; metadata formatting correction):


Owner: E2E-REAL-6 / NVIDIA attention backward; sync NVIDIA-LSE-BACKWARD-SM120-SCALE-2026-10-02.

The exact RTX 5070 recorder now covers two added shapes: ragged
B/Hq/Hkv/Sq/Sk/D/Dv=1/4/2/127/131/64/64 and regular
1/4/2/256/256/64/64. Saved/recompute forward outputs, saved row-LSE, and all
saved/recompute gradients passed the independent float64 oracle. At 256,
maximum output/LSE/gradient errors were 5.22e-8/1.15e-6/7.06e-7. Saved
packages include complete Graph/Schedule/Tile provenance.

The measurements reveal the unresolved performance limit: 256x256 saved and
recompute backward event medians are 1221.5 and 1927.4 ms, while forward is
2.843 and 2.826 ms. At 127x131, backward is 189.3/308.8 ms. End-to-end timing
tracks device time at these sizes. The current SM120 materializer contains
scalar dot-product reductions inside per-key scans, a concrete lead for
tiled-kernel work. This is correct-route evidence, not a performance closure
or selector promotion.

[Source-fingerprinted exact-device packet](../../../benchmarks/baselines/nvidia_attention_lse_e2e_20261001/saved_lse_extended_recheck_20261002.json).


### 2026-10-02 — saved-LSE delta experiment and paged-KV recheck

Owner: [E2E-REAL-6](INTEGRATED_COMPILER_PLAN.md#e2e-real-6)

PRs: Uncommitted; original synchronization labels are preserved below.

Outcome: saved-LSE delta experiment and paged-KV recheck: implementation and evidence are preserved in the recorded detail below.

Remaining: The recorded limitations below remain in force; this schema correction changes no completion or support state.

Evidence: Original architecture-specific receipts, packet links and timing qualifications are preserved below.

<!-- entry-fields:end -->

Recorded detail (preserved; metadata formatting correction):


Owner: E2E-REAL-6 / NVIDIA attention backward and paged-KV; sync
NVIDIA-LSE-BACKWARD-SM120-SCALE-2026-10-02.

A direct saved-output `dO·O` delta experiment was rejected after exact RTX 5070
runs disagreed with the independent gradient oracle and recompute path (maximum
absolute error 2.80e-3). The extra saved-output operand and ABI changes were
removed. The existing five-input saved-LSE backward route passed 48 focused
unit/device tests. A fresh correctness-gated five-shape timing packet confirms
saved/recompute backward medians of 188.55/306.50 ms at 127x131 and
1218.60/1926.13 ms at 256x256; the serialized scalar backward remains a clear
tiling/performance action.

The paged-KV Tile-direct versus staged CUDA recheck passed all three
permuted-page correctness cases. With only three samples and ten device-event
repetitions, every row failed the 4% stability gate, so the packet does not
change the higher-sample retain-existing disposition or selector.

[Saved-LSE five-shape packet](../../../benchmarks/baselines/nvidia_attention_lse_e2e_20261001/saved_lse_abi_recheck_20261002.json).
[Paged-KV diagnostic packet](../../../benchmarks/baselines/nvidia_sm120_paged_kv_recheck_20261002/paged_kv_recheck.json).


## NVIDIA attention forward timing refresh — 2026-10-02

Super-Bear RTX 5070 (sm_120) reran all eight full/causal, fp16/fp32,
regular/ragged Graph-to-native forward cases in two interleaved cohorts. All
passed the independent fp64 oracle before timing, with maximum absolute error
5.96e-8. Device-event medians were 25.3–32.4 us and all eight met the 3%
stability policy; six of eight end-to-end rows met it. The two unstable rows
were non-causal fp16. This is exact-device route and correctness evidence,
without comparative speedup or selector promotion.

Packet: `benchmarks/baselines/nvidia_attention_lse_e2e_20261001/attention_lse_followup_20261002.json`.


## gfx1201 NVFP4 ingest scale-search audit — 2026-10-02

The bounded joint E2M1/E8M0 search was compared with all 254 finite E8M0
exponents on 16,384 deterministic randomized K32 blocks using positive
representable E4M3 source scales and per-block powers-of-two global scaling.
No block missed the exhaustive minimum decoded-weight SSE. A 128-block version
is now retained in `tests/unit/test_rocm_nvfp4_ingest.py`; it passes in WSL.
This tests a synthetic host-side envelope only. The original BF16 checkpoint
and exact gfx1201 rerun of this new unit change remain open.

Evidence report: `benchmarks/baselines/rocm_nvfp4_ingest_20261001/README.md`.


## SM120 typed producer-to-matmul refresh — 2026-10-02

The five-shape typed matmul recorder passed its independent fp32 oracle on the
RTX 5070 (sm_120a), with maximum error 4.77e-7 and full Graph/Schedule/Tile
lineage. A separate resident RMSNorm -> matmul case carried typed fragments
and a fragment accumulator through four K iterations, reused the same
caller-owned intermediate on one stream, and passed with errors 0 and
1.43e-6. Producer/consumer resident CUDA-event medians were 10.42/11.45 us
with 7.4%/8.0% CV; timings are diagnostic. The generic tensor-valued C++
constructors remain open.

Packets: `benchmarks/baselines/nvidia_sm120_w1_typed_tensor_edge_20261001/typed_m16n8_followup_20261002.json` and `benchmarks/baselines/nvidia_sm120_w1_typed_tensor_edge_20261001/rmsnorm_matmul_followup_20261002.json`.



## NVIDIA saved-output row-delta checkpoint — 2026-10-02

Owner: E2E-REAL-6 / NVIDIA attention backward; sync
NVIDIA-ATTENTION-LSE-BACKWARD-2026-10-01.

Correction to the earlier 2026-10-02 rejection entry above: the reported
2.80e-3 gradient mismatch was caused by the PTX host bridge misclassifying
the new nine-buffer saved-output ABI as bias+LSE. It allocated the wrong byte
count for O and shifted output pointers. The failure did not disprove the
dO·O row-delta identity. The native entry now carries an explicit
backward_lse_output name; host, resident, and event launchers recognize
that role, allocate/copy O at its full shape, and place LSE and gradient
buffers at their declared ordinals.

On Super-Bear RTX 5070 (sm_120a), saved-output/LSE backward matches both the
independent float64 oracle and recompute at [1,4,2,16,16,32,32] and
[1,4,2,128,128,64,64]. At shape 16, maximum dq/dk/dv errors were
2.79e-9/4.66e-9/1.49e-7; medians were 0.0804 ms device and 1.304 ms E2E.
At shape 128, errors were 5.59e-9/7.45e-9/2.09e-7; medians were 2.378 ms
device and 4.079 ms E2E, versus the prior same-shape saved-LSE/dV-elided
183.179/185.364 ms. The 77.0x kernel and 45.4x E2E reductions are exact-device
comparisons on the RTX 5070; selector/default route remains unchanged.
Broader shape/dtype/checkpoint envelopes and the compiler-plan route census
remain open.

Packets:
- benchmarks/baselines/nvidia_attention_lse_e2e_20261001/attention_backward_saved_output_recheck_20261002.json
- benchmarks/baselines/nvidia_attention_lse_e2e_20261001/attention_backward_saved_output_shape128_20261002.json



## NVIDIA saved-output host-bridge capacity recheck — 2026-10-02

Owner E2E-REAL-6 / NVIDIA attention backward; sync
NVIDIA-ATTENTION-LSE-BACKWARD-2026-10-01. The resident and CUDA-event PTX
launchers now allocate ten pointer slots, matching the accepted maximum
saved-O + bias + row-LSE ABI; output argument slots remain sized for ten
buffers plus seven dimensions. The rebuilt RTX 5070 small-shape route passed
forward, saved/recompute, and independent-oracle checks. Five resident event
samples (100 launches each) measured 0.08015 ms median at 0.075% CV. End-to-end
median was 1.419 ms at 53.3% CV due to a host outlier. This confirms the
rebuilt launcher and event path for the saved-output ABI but makes no speedup
claim. The ten-buffer bias combination has capacity coverage in code but still
needs a dedicated exact-device test before that sub-envelope is claimed.

Packet: benchmarks/baselines/nvidia_attention_lse_e2e_20261001/attention_backward_saved_output_bridge_refresh_20261002.json



## 2026-10-02 SM120 typed matmul 16-panel exact-device refresh

Owner W1.1; cross-backend sync
`W1.1-SM120-FOUR-PANEL-BENCHMARK-2026-10-02`. The typed-matmul benchmark
schema advances to v2 with an explicit K-panel count and requires a Tile loop
for multi-panel K. On Super-Bear RTX 5070 (sm_120), all six Graph -> Schedule
-> Tile -> NVIDIA Target IR -> PTX rows passed their fp32 oracle before timing.
The new 64x256x64 case has 16 K panels, max absolute error 2.86e-6, 8.30 us
event median (7.36% CV), and 0.278 ms E2E median (4.34% CV). Timings remain
route attribution only. The pre-schedule generic tensor-valued constructors
remain open; this increment validates the canonical producer route.

Sibling assessment: ROCm requires its own architecture and route evidence;
Apple and x86 are not applicable because no shared IR, ABI, numeric policy,
or runtime contract changed. No physical schedule or performance evidence is
transferred.

Packet: benchmarks/baselines/nvidia_sm120_w1_typed_tensor_edge_20261001/typed_matmul_four_panel_refresh_20261002.json


## 2026-10-02 gfx1201 scheduled matmul warmed cache benchmark

Owner E2E-REAL-6-ROCM-MATMUL-CACHE; sync
`E2E-REAL-6-GFX1201-MATMUL-CACHE-WARMED-2026-10-02`. The exact Tajasaurus
gfx1201 run passed the full cache-key test file (31 passed, one gfx1151 case
skipped on the gfx1201 host). Six static f16 shapes from 16^3 to 512^3 reused
one HSACO and entry with distinct shape guards. Every output matched the fp32
NumPy oracle (maximum absolute error 1.10e-5). Five correctness-checked warmup
launches were excluded from seven timed samples per shape. HIP-event medians
were 8.64–24.96 us with 0.74–1.45% CV; end-to-end medians were 2.14–5.85 ms.
This is exact-device correctness and route attribution, not selector promotion.
Dynamic, split-K, fused, LDS-staged, and broader dtype/layout cache identity
remain open. Princess-Luna could not be reached during this pass, so gfx1151
cache parity remains unverified.

Packet: `benchmarks/baselines/rocm_gfx1201_shape_key_20261002/matmul_cache_reuse_warmed.json`



## 2026-10-02 gfx1151 shape-free cache exact-device parity

Owner E2E-REAL-6-ROCM-CACHE-KEYS; sync
`E2E-REAL-6-ROCM-CACHE-GFX1151-PARITY-2026-10-02`. Princess-Luna exact
gfx1151 passed softmax, reduction, and scheduled-attention cache reuse tests
(3 passed). Each test confirmed one native image and entry across changed
runtime shapes and graph symbols, distinct per-package shape guards, native
GPU execution, and numerical agreement with its independent reference. The
packet fingerprints the Python sources and compiler executable. This does not
close other ROCm families/layouts or claim performance; timings were not
collected. gfx1201 scheduled-matmul timings are recorded separately.

Packet: `benchmarks/baselines/rocm_gfx1151_cache_20261002/exact_device_recheck.json`



## 2026-10-02 SM120 canonical producer route regression guard

Owner W1.1 / NVIDIA fragment-producer closure; sync
`W1.1-SM120-CANONICAL-PRODUCER-GUARD-2026-10-02`. The pipeline alias fixture
now checks the actual boundary: pointer-backed `tile.view` values feed typed
`tile.fragment_pack`, the MMA consumes typed fragments and accumulator, and
the route contains neither residual `tessera.matmul` nor generic
`tile.async_copy`. The built Super-Bear compiler passed FileCheck for the
SM120 prefix. The four-panel exact-device benchmark remains the numerical and
timing evidence. This guards the registered canonical route but does not
migrate the two direct legacy Tile constructors; that W1.1 work remains open.

Sibling outcome: Apple, ROCm, and x86 are not applicable because the fixture
exercises only the NVIDIA SM120 pipeline and changes no shared IR or ABI.

Fixture: `tests/tessera-ir/phase3/cuda13/nvidia_pipeline_alias.mlir`

## 2026-10-02 — saved-LSE backward larger ragged shape recheck

Owner E2E-REAL-6 / NVIDIA attention backward; sync
`NVIDIA-LSE-BACKWARD-SM120-SCALE-2026-10-02`. With saved output enabled,
the exact RTX 5070 (sm_120) 1x4x2x127x131x64x64 case passed saved/recompute
and independent fp64-oracle checks. Maximum output/LSE/gradient errors were
4.29e-8 / 8.75e-7 / 2.53e-7. Saved backward measured 2.382 ms CUDA-event
median (0.54% CV) and 4.012 ms end-to-end median (25.09% CV); recompute was
307.918 ms / 310.458 ms. Device time confirms saved output removes most of the
previous repeated row-delta work, while end-to-end variability remains high.
This is route evidence only; selector default remains recompute. Tiled row
statistics and the bias-plus-saved-output/LSE envelope remain open.

Packet: `benchmarks/baselines/nvidia_attention_lse_e2e_20261001/attention_lse_127x131_saved_output_20261002.json`





## 2026-10-02 — Qwen3-8B q-projection exact gfx1201 comparison

Owner: [ROCM-NVFP4-INGEST-1](INTEGRATED_COMPILER_PLAN.md#rocm-nvfp4-ingest-1); sync
`ROCM-NVFP4-INGEST-1-QWEN3-QPROJ-2026-10-02`.

Outcome: Tajasaurus RX 9070 XT (gfx1201) fetched only the named q_proj weight,
scales, global scale, and corresponding BF16 tensor ranges from pinned Qwen3-8B
revisions; 42,991,620 tensor bytes were read and no model shard was saved. The
shipped NVFP4 projection is 9.50% relative RMS from its pinned BF16 source;
a direct BF16-to-MXFP4 bounded-search baseline is 11.22%. NVFP4-to-MXFP4
conversion is 14.99% from BF16, 11.29% from shipped NVFP4, and 18.03% from the
direct BF16-to-MXFP4 result. This is material additional conversion error and
blocks default-route promotion pending an explicit accepted quality envelope
and broader projections.
Both ingested NVFP4 weights and direct BF16-to-MXFP4 weights traversed the same
Graph -> Schedule -> Tile -> gfx1201 Target IR package and executed natively;
each output matched its decoded-weight reference with maximum absolute error
0.0. Ingest took 2555.53 ms, package construction 145.28 ms, and source range
fetch 6402.95 ms. Five correctness-checked E2E warmups preceded timing:
persistent HIP-event median 31.29 us; `runtime.launch` median 6.281 ms at 0.61% CV. The
activation is synthetic FP8; one projection does not establish whole-model
quality or throughput.

Sibling assessment: Apple has no matching MXFP4 scale-plane consumer; NVIDIA's
NVFP4 package is a different physical contract and its evidence does not
transfer; x86 has no ROCm storage or instruction route. No shared IR, ABI, dtype,
or runtime contract changed in this evidence-only slice.

Packet: `benchmarks/baselines/rocm_nvfp4_ingest_20261001/gfx1201_qwen3_8b_q_proj_20261002.json`.
Benchmark: `benchmarks/rocm/benchmark_rocm_nvfp4_checkpoint.py`.



## 2026-10-02 — saved-LSE forward/backward full ragged recheck

Owner: E2E-REAL-6 / NVIDIA attention; sync
`NVIDIA-LSE-FORWARD-BACKWARD-127X131-2026-10-02`. On the RTX 5070 (sm_120),
the 1x4x2x127x131x64x64 saved and recompute packages passed independent fp64
forward, row-LSE, and gradient checks before timing. Maximum output/LSE/gradient
errors were 4.29e-8 / 8.75e-7 / 2.54e-7. Saved forward measured 790.67 us CUDA
event and 2.045 ms end-to-end median; recompute measured 784.92 us and 4.369 ms.
Saved backward measured 2.387 ms device-event and 6.350 ms end-to-end; recompute
measured 307.780 ms and 315.367 ms. Only saved rows have complete
Graph->Schedule->Tile->Target provenance. End-to-end samples are a short
three-sample diagnostic; selector default remains recompute. Broader dtype and
bias-plus-saved-output/LSE envelopes remain open.

Packet: `benchmarks/baselines/nvidia_attention_lse_e2e_20261001/attention_lse_127x131_full_recheck_20261002.json`.



## 2026-10-02 — W1.1 legacy producer census recheck

Owner: W1.1 / NVIDIA fragment-producer closure; sync
`W1.1-SM120-LEGACY-PRODUCER-CENSUS-2026-10-02`. The source census still finds
two generic `tile.mma` constructors in `TileIRLoweringPass.cpp`. On the active
Super-Bear build, `phase2/tiling.mlir` passes its standalone generic Tile
FileCheck, while `phase3/cuda13/nvidia_pipeline_alias.mlir` passes the registered
SM120 pipeline fixture requiring pointer-backed `tile.view`, typed
`fragment_pack`, MMA, and no generic async-copy producer. The pipeline runs
Graph-to-Schedule and Schedule-to-Tile before generic Tile lowering, so these
constructors are bypassed for the registered Graph matmul path. The standalone
generic form has no demonstrated supported native producer ABI and fails the
sm_120 typed-fragment contract; bufferization/lifetime migration is therefore
not justified by the current census. W1.1 remains open for a named, supported
legacy producer route or explicit retirement; this fixture pair does not claim
that route executes.

Sibling outcome: Apple and ROCm have separate fragment producers and schedules;
x86 has no NVIDIA fragment ABI. No shared contract changed and no sibling
physical evidence is transferred.

Fixtures: `tests/tessera-ir/phase2/tiling.mlir`;
`tests/tessera-ir/phase3/cuda13/nvidia_pipeline_alias.mlir`.



## 2026-10-02 — gfx1201 shape-free matmul cache exact-device rebuild

Owner E2E-REAL-6-ROCM-MATMUL-CACHE; sync
`E2E-REAL-6-GFX1201-MATMUL-CACHE-WARMED-2026-10-02`. The first explicit
gfx1201 test run selected a stale compiler from an older scratch snapshot; its
two scheduled-matmul cases crashed during AMDGPU instruction selection and are
not counted as product failures or route evidence. Rebuilt `tessera-opt` from
the active Tajasaurus checkout into `.build-gfx1201-current`; its SHA-256
`1e9b32318ce4a15b8466dcb79a2d55b504c0b6cce5f563abf853abc10bf9b397` matches
the exact compiler recorded in the warmed six-shape cache packet. The full
`test_rocm_shape_free_cache_key.py` then passed 31 tests with one expected
gfx1151-only skip on Tajasaurus. This confirms the current build for the tested
softmax/reduction/attention and scheduled matmul cache envelopes; other ROCm
families and layouts remain open. The packet's warmed six-shape measurements
remain diagnostic and are not selector promotion.

Packet: `benchmarks/baselines/rocm_gfx1201_shape_key_20261002/matmul_cache_reuse_warmed.json`.



### 2026-10-02 — SM120 legacy tensor MMA route guard

Owner: [W1.1](INTEGRATED_COMPILER_PLAN.md#w11)

PRs: Uncommitted; original synchronization labels are preserved below.

Outcome: SM120 legacy tensor MMA route guard: implementation and evidence are preserved in the recorded detail below.

Remaining: The recorded limitations below remain in force; this schema correction changes no completion or support state.

Evidence: Original architecture-specific receipts, packet links and timing qualifications are preserved below.

<!-- entry-fields:end -->

Recorded detail (preserved; metadata formatting correction):


Owner: W1.1 / NVIDIA fragment-producer closure; sync `NVIDIA-W1.1-SM120-LEGACY-ROUTE-GUARD-2026-10-02`.

`TileIRLoweringPass` no longer registers the generic tensor-valued `LowerMatmulToTileMMA` and `LowerKReductionAddToTileMMA` rewrites for sm_120+. If direct Tile input retains `tessera.matmul`, the pass reports that the operation must enter the registered Graph-to-Schedule-to-Tile pipeline. Earlier SM targets retain their existing generic lowering. The phase2 fixture checks the SM120 refusal and the phase3 fixture checks the positive typed-fragment pipeline. The exact RTX 5070 device wrapper passed all 14 cases, including five public typed scheduled-matmul shapes; focused host-free producer and admission coverage passed 15 tests. Diagnostic/pass registries and audit-doc tests passed 287 tests.

This retires the unsupported competing direct-Tile SM120 producer route and protects the canonical pointer-backed `tile.view` -> typed-fragment -> MMA path. It does not migrate arbitrary tensor lifetimes or close the broader W1.1 producer census. Apple and ROCm retain independent producer schedules; x86 has no NVIDIA fragment ABI. No shared ABI changed.


### 2026-10-02 — SM120 static ragged-K typed-fragment producer

Owner: [W1.1](INTEGRATED_COMPILER_PLAN.md#w11)

PRs: Uncommitted; original synchronization labels are preserved below.

Outcome: SM120 static ragged-K typed-fragment producer: implementation and evidence are preserved in the recorded detail below.

Remaining: The recorded limitations below remain in force; this schema correction changes no completion or support state.

Evidence: Original architecture-specific receipts, packet links and timing qualifications are preserved below.

<!-- entry-fields:end -->

Recorded detail (preserved; metadata formatting correction):


Owner: W1.1 / NVIDIA fragment-producer closure; sync `NVIDIA-W1.1-STATIC-RAGGED-K-2026-10-02`.

Schedule-to-Tile now selects the typed SM120 fragment producer for positive static K not divisible by 16 when M is divisible by 16 and N by 8. The A/B `tile.view` operations carry logical row/column bounds while retaining static leading dimensions, allowing `materializeSm120Mma16Pack` to safely zero-fill the final K panel. Exact Super-Bear RTX 5070 (sm_120) numerical checks pass for fp16 and bf16 at M/K/N=48/67/16. Host-free structure checks confirm two bounded views with leading dimension 67; the broader exact-device test passes seven shape/dtype cases.

The fp16 correctness-gated benchmark separates device events and end-to-end launch time. Its current K=67 row measures 15.75 us event median at 33.6% CV and 0.363 ms E2E at 3.8% CV in the 11-sample stability run. Repeated packets varied materially; no performance conclusion or selector change follows. The typed producer scope remains aligned M/N, no fused epilogue; the generic direct-Tile constructors and full W1.1 census remain open. This uses the existing shared bounded `tile.view` contract, so Apple, ROCm, and x86 physical schedules are not affected and no sibling parity is inferred.

Packet: `benchmarks/baselines/nvidia_sm120_w1_typed_tensor_edge_20261001/typed_matmul_ragged_k67_stability_20261002.json`.



## W1.1 static M/N/K tails through typed fragments — 2026-10-02

Owner W1.1; synchronization key `NVIDIA-W1.1-STATIC-MNK-TAILS-2026-10-02`. The canonical
SM120 Graph -> Schedule -> Tile producer now admits every positive static M/N/K
in its existing fp16/bf16, fp32 accumulation/output, unfused envelope. Input
`tile.view` operations carry logical M/N/K bounds when any dimension has a
tail. NVIDIA lowers the existing six-operand static bounded `tile.store`
form with per-element M/N predicates; exact static leading dimensions remain.
Canonical typed packages have no tensor-valued MMA operands.

Large ragged-K shapes above the macro crossover previously inherited a
`_macro_kernel` symbol from workload alone. Their new typed producer uses a
16x8 output tile, so keeping that name would incorrectly launch the 32x32 CTA
grid. The compiler now names the actual producer geometry. Packaging projects
the actual native Tile entry; the duplicate Python entry-name and macro-policy
implementations are retired. Sixteen compiler admission/entry tests cover both
storage types and both sides of the crossover.

The full exact RTX 5070 scheduled device module passed 32 cases. This includes
17 typed numerical shape/dtype cases through M/K/N=257/513/257, six fp16/bf16
resident canary cases, and the existing macro/dynamic/epilogue cases. Canaries
prove input and output guard preservation; the tests synchronize before closing
borrowed views and verify owned-buffer cleanup. Seven compiler structural
checks and 48 resident-program/frontend checks also pass.

Shared launch validation now accepts opposite contiguous row/column labels only
when at most one dimension can vary; the physical addresses are then identical.
Genuinely different orders and strided mismatches still fail. NVIDIA-owned
buffers normalize scalar dtype classes to NumPy dtype objects to retain bf16
metadata at the checked CUDA ABI. Shared contract, diagnostic/pass, dtype and
audit gates passed 488 tests. Apple/ROCm/x86 have contract-only coverage and
require their own native singleton execution proof.

The twelve-row packet has eleven samples, 1000 CUDA-event repetitions, 20
end-to-end repetitions and 20 warmups. Source fingerprints match the measured
checkout. The new rows are:

| M/K/N | Max absolute error | Event median (us) | Event CV | E2E median (ms) | E2E CV |
| --- | --- | --- | --- | --- | --- |
| 1/1/1 | 0 | 10.93 | 36.3% | 0.298 | 92.3% |
| 17/19/23 | 1.19e-07 | 15.79 | 31.7% | 0.365 | 2.2% |
| 31/33/9 | 1.79e-07 | 15.01 | 48.3% | 0.300 | 50.2% |
| 48/67/17 | 4.77e-07 | 10.01 | 36.3% | 0.294 | 52.2% |
| 257/513/257 | 7.15e-06 | 16.57 | 7.6% | 0.526 | 36.5% |

Timing variance is high; no speedup, selection or promotion claim follows.
Compute Sanitizer 13.3 could not initialize the WSL/WDDM debugger interface and
reported unsupported-device errors. Its earlier application rerun passed 21
cases, but there is no memcheck proof. Resident numerical/canary evidence is
separate.

This closes the static-tail extension of the canonical typed producer.
The two generic direct-Tile constructors, fused epilogues and broader arbitrary
tensor-lifetime envelopes remain W1.1 obligations.

[Packet](../../../benchmarks/baselines/nvidia_sm120_w1_typed_tensor_edge_20261001/typed_matmul_static_mnk_tails_20261002.json).
[Validation and compiler fingerprints](../../../benchmarks/baselines/nvidia_sm120_w1_typed_tensor_edge_20261001/static_mnk_validation_20261002.json).
[Final test transcript](../../../benchmarks/baselines/nvidia_sm120_w1_typed_tensor_edge_20261001/static_mnk_validation_20261002.txt).


### 2026-10-02 — native ROCm identity and dynamic/fused cache reuse

Owner: [E2E-REAL-6](INTEGRATED_COMPILER_PLAN.md#e2e-real-6)

PRs: Uncommitted; original synchronization labels are preserved below.

Outcome: native ROCm identity and dynamic/fused cache reuse: implementation and evidence are preserved in the recorded detail below.

Remaining: The recorded limitations below remain in force; this schema correction changes no completion or support state.

Evidence: Original architecture-specific receipts, packet links and timing qualifications are preserved below.

<!-- entry-fields:end -->

Recorded detail (preserved; metadata formatting correction):


Owner E2E-REAL-6-ROCM-MATMUL-CACHE / E2E-REAL-6-ROCM-CACHE-KEYS;
sync ROCM-NATIVE-IDENTITY-2026-10-02. The compiler now validates and projects
audited Target directives onto physical image identity. Production Python text
rewriting is retired. Dynamic tensor.dim scaffolding is restricted to constant
axes of entry arguments. Fused exact-device validation exposed the legacy Target
trailing-bias ABI versus scheduled bias-before-output mismatch; native Target
lowering now carries portable_abi and output storage, and generation honors them.

Current-source gfx1201 validation: 338 passed, three owning-gfx1151 skips,
including seventeen real MLIR projection cases and eight fp16/bf16 numerical
static/dynamic/fused cases. Four six-shape packets each reuse one image and entry
after one cold compile, with distinct shape guards and independent numerical
checks before/during timing. HIP-event and instrumented runtime.launch samples
are separated; no selector promotion or matched speedup claim.
See [the evidence report](../../../benchmarks/baselines/rocm_gfx1201_native_identity_20261002/README.md).
Runtime module-loading overhead and exact gfx1151 dynamic/fused proof remain
open. Apple/NVIDIA/x86 target ABI parity is not applicable; their physical
evidence and open compiler programs are not inferred from this AMD result.


### 2026-10-02 — native HIP image leases and independent gfx1151/gfx1201 attribution

Owner: [E2E-REAL-6](INTEGRATED_COMPILER_PLAN.md#e2e-real-6)

PRs: Uncommitted; original synchronization labels are preserved below.

Outcome: native HIP image leases and independent gfx1151/gfx1201 attribution: implementation and evidence are preserved in the recorded detail below.

Remaining: The recorded limitations below remain in force; this schema correction changes no completion or support state.

Evidence: Original architecture-specific receipts, packet links and timing qualifications are preserved below.

<!-- entry-fields:end -->

Recorded detail (preserved; metadata formatting correction):


Owner E2E-REAL-6-ROCM-MATMUL-CACHE / E2E-REAL-6-ROCM-CACHE-KEYS;
sync ROCM-NATIVE-MODULE-CACHE-2026-10-02. Princess-Luna's relocated build was
restored with explicit LLVM/MLIR 23 paths and the full semantic driver. Its
gfx1151 dynamic/fused native image proof is now separate from gfx1201 evidence.

A new C++ HIP image service owns bounded device/context/process-scoped module
and function handles. Opaque leases span checked launch completion; explicit
context clearing rejects live leases. The Python driver binds this native
service and preserves per-call descriptor validation. Native controlled-HIP
tests cover context/device isolation, concurrency, bounded eviction, error
cleanup and inherited-process ownership; hot-device tests prove invalid
capacities do not touch a cached module.

Both devices passed 61 focused checks (three sibling-owned skips each).
gfx1201 split-K: 59 passed, including real secondary-entry numerical checks.
Retained direct paged-KV/MoE package cache checks executed on the real ROCm
compiler (two passes, artifact evidence only). Host checked-ABI/routing suite:
1041 passed, 25 environment-specific skips. The NVIDIA-only compiler correctly
reports native ROCm packaging unavailable instead of inferring capability from
an executable path; required native checks ran on the ROCm compiler.

Matched six-shape fp16 timing envelopes hold physical schedules, inputs and
compiled image digests identical: 96 per-call module loads versus one native
load and 95 hits. Sampled instrumented E2E control/cached median ratios are
1.16–3.93x on gfx1151 and 1.39–4.54x on gfx1201. HIP events are separate,
uncontrolled-clock launch brackets; no isolated kernel speedup is claimed.
See [gfx1151 evidence](../../../benchmarks/baselines/rocm_gfx1151_native_module_cache_20261002/README.md)
and [gfx1201 evidence](../../../benchmarks/baselines/rocm_gfx1201_native_module_cache_20261002/README.md).
Per-call allocation/transfers and wider image-key envelopes remain open.
External context destruction/device reset requires explicit clearing; no
automatic reset or general post-fork HIP execution support is claimed.
Apple/NVIDIA/x86 retain their own native context and module contracts.


## 2026-10-02 — NVIDIA bias saved-checkpoint native integration

Owner E2E-REAL-6; sync NVIDIA-BIAS-SAVED-CHECKPOINT-2026-10-02. The internal
checkpoint Graph tensor operations accept an exact-shape f32 bias operand.
Native Graph-to-Schedule and Schedule-to-Tile validate and retain its ordered
binding. Distinct forward/backward bias ABIs bind the existing CUDA kernel's
six/ten-buffer envelopes. The canonical driver carries bias role metadata;
the resident forward bridge now accepts six buffers. No Python kernel emitter
or alternate backend path was introduced.

Paired capture includes bias in semantic identity and private device ownership.
A preexisting tape defect dropped saved output from backward; the current
nine-buffer plain and ten-buffer bias ordering is now enforced. Exact RTX 5070
checks cover independent fp64 output/LSE/gradient oracles, host and resident
launches, repeated backward, and caller input mutation. The three-shape
correctness-gated packet retains CUDA-event and E2E timing domains and source/
compiler/runtime-library fingerprints. Automatic bias AD/JVP and broader
producer migration remain open; no architecture parity or selector promotion.

All four backend plans assess the shared optional operand. Non-NVIDIA native
checkpoint bias execution requires owning-backend follow-up.


## 2026-10-02 — W1.1 positive legacy SM120 matmul integration

Owner W1.1; sync W1.1-SM120-LEGACY-SCHEDULE-2026-10-02. The legacy Tile entry now composes
registered native Graph-to-Schedule/Schedule-to-Tile passes for standalone SM120
matmul. Original frontend symbols are normalized by MLIR SymbolTable, preserving
symbol users; explicit target/architecture guards prevent foreign delegation.
The native replay product is preserved before unrelated legacy tensor folding.
No duplicated Python lowering or backend emitter was added.

Exact RTX 5070 numerical checks cover fp16/bf16, static tails, multi-panel K, and
two-shape bounded dynamic bias/ReLU/residual reuse with padded-view canaries.
Host-free replay comparisons cover fused/static/dynamic output-storage cases.
SM80 and SM90 FileCheck fixtures still pass, without an older-GPU claim.
The correctness-gated twelve-shape packet binds native compiler/source/library
hashes and retains separate event/E2E domains; no selector promotion.

All four queues assess the shared pass change. Canonical K-step conversion and
broader tensor lifetime integration remain open; the full W1.1 goal is active.



### 2026-10-02 — native SM120 K-reduction Schedule ownership

Owner: [W1.1](INTEGRATED_COMPILER_PLAN.md#w11)

PRs: Uncommitted; original synchronization labels are preserved below.

Outcome: native SM120 K-reduction Schedule ownership: implementation and evidence are preserved in the recorded detail below.

Remaining: The recorded limitations below remain in force; this schema correction changes no completion or support state.

Evidence: Original architecture-specific receipts, packet links and timing qualifications are preserved below.

<!-- entry-fields:end -->

Recorded detail (preserved; metadata formatting correction):


Owner W1.1; sync W1.1-SM120-NATIVE-K-SCHEDULE-2026-10-02.

For an explicit nvidia_sm120/sm_120 Graph matmul, generic tiling now emits the
registered native Schedule contract. Schedule-to-Tile owns the typed fragment
K loop and accumulator lineage. The direct Tile entry consumes an existing
Schedule without generating it again. MLIR SymbolTable normalization is shared
by both entry points before Schedule hashing; no Python emitter or Target bypass
was introduced. Generic tensor tile-size overrides are explicitly gated because
the SM120 native profile owns physical instruction geometry.

Exact RTX 5070 fp16/bf16 static-tail and multi-panel numerical checks exercise
both entries. Bounded dynamic fused bias/ReLU/residual shapes and padded output
canaries also exercise both entries. The twelve-shape fp16 packet compiles the
actual tiling-produced Tile product, proves byte-identical canonical Schedule
and Tile replay, and records separate CUDA-event and E2E timing with compiler,
runtime and source fingerprints. Timing variance prevents performance promotion.

The ordinary SM120 frontend tiling path no longer creates tensor-valued
canonical K-step MMA producers. Already-authored generic tensor K-step graphs,
arbitrary producer bufferization/lifetimes, native Schedule tuning knobs, and
older-target physical migrations remain open. Older-target FileCheck results
are compiler artifact evidence, not exact-device parity.



Validation for W1.1-SM120-NATIVE-K-SCHEDULE-2026-10-02: 508 passed,
50 environment/capability skips, zero failures. Generic T16/T32 tiling,
fused epilogue tiling and SM80/SM90 Tile FileCheck fixtures passed. Packet
source, compiler and native-launch-library hashes match the measured checkout.


### 2026-10-02 — resident producer edge with complete native epilogue

Owner: [W1.1](INTEGRATED_COMPILER_PLAN.md#w11)

PRs: Uncommitted; original synchronization labels are preserved below.

Outcome: resident producer edge with complete native epilogue: implementation and evidence are preserved in the recorded detail below.

Remaining: The recorded limitations below remain in force; this schema correction changes no completion or support state.

Evidence: Original architecture-specific receipts, packet links and timing qualifications are preserved below.

<!-- entry-fields:end -->

Recorded detail (preserved; metadata formatting correction):


Owner W1.1; sync W1.1-SM120-RESIDENT-EPILOGUE-2026-10-02.

The checked SM120 RMSNorm/LayerNorm/softmax-to-matmul tensor edge now carries
native fused bias, activation and residual operands. The Graph, native
Schedule/Tile, Target IR and image remain compiler-owned; Python sequences
compiled packages and manages allocations. The resident CUDA bridge consumes
the complete A/B/bias?/residual?/D pointer order followed by M/N/K and optional
leading dimensions. It validates buffer counts and span limits and queues the
consumer on the producer stream without host intermediate transfers.

Host execute and resident execute accept named fp32 bias/residual operands,
validate active shapes, reject missing/extraneous operands and preserve
allocation lifetime through stream completion. Program validation binds
epilogue semantics to the compiled entry and checks operand shapes/storage.
The shared bounded-axis Graph projection preserves epilogue arguments and
caller-owned Graph state, with dynamic M/N residual and N bias dimensions.
Numerics check the ordered fp32 epilogue before final fp16 output conversion.

Exact RTX 5070 evidence covers fp16/bf16, fp16/fp32 output and static/dynamic-M
reuse. This is a named producer edge, not general tensor bufferization or
arbitrary lifetime graphs. Native bias derivatives, broader AD integration,
older-target migrations and general producer graphs remain open.



Boundary for W1.1-SM120-RESIDENT-EPILOGUE-2026-10-02: fused consumers
currently use the native tile.matmul_kernel carrier. Their explicit typed
fragment epilogue migration remains open; resident execution does not close
that producer census.

Validation for W1.1-SM120-RESIDENT-EPILOGUE-2026-10-02: 625 passed,
49 environment/capability skips, zero failures. Eight isolated exact-device
benchmark rows passed before timing and after device replay. Packet source,
compiler and native-launch-library hashes match the measured checkout.


### 2026-10-02 — explicit typed fragment epilogue producer

Owner: [W1.1](INTEGRATED_COMPILER_PLAN.md#w11)

PRs: Uncommitted; original synchronization labels are preserved below.

Outcome: explicit typed fragment epilogue producer: implementation and evidence are preserved in the recorded detail below.

Remaining: The recorded limitations below remain in force; this schema correction changes no completion or support state.

Evidence: Original architecture-specific receipts, packet links and timing qualifications are preserved below.

<!-- entry-fields:end -->

Recorded detail (preserved; metadata formatting correction):


Owner W1.1; sync W1.1-SM120-TYPED-EPILOGUE-2026-10-02.

Native Schedule-to-Tile now emits typed views, packs, a loop-carried fp32
fragment accumulator, unpack and store for positive f16/bf16 SM120 matmuls
with bias/activation/residual and f16/f32 output. The epilogue is on tile.store:
bias then activation then residual in fp32, followed by final output conversion.
Auxiliary loads are inside the output M/N guard. The explicit producer uses a
16x8 one-warp grid; native symbol/descriptor geometry remains matched. This
replaces the deferred tile.matmul_kernel carrier for this named envelope.

The existing shared store verifier now discounts ordered bias/residual pointers
from address arity and checks the required residual order contract. Schedule
pass metadata and the existing bias diagnostic guidance describe this
extension. ROCm explicitly rejects a residual store pending its own consumer;
its earlier bias/activation contract is retained. No Python Tile constructor,
new operation or Target bypass was added.

General producer bufferization/lifetime graphs and already-authored generic
tensor K-step graphs remain open. This does not close W1.1 for other storage
types or older targets, and it does not claim sibling exact-device parity.



Validation for W1.1-SM120-TYPED-EPILOGUE-2026-10-02: 688 passed,
49 environment/capability skips, zero failures. Twenty-four correctness-gated
RTX 5070 rows cover fp16/bf16 input, fp16/fp32 output, static/dynamic M, and
bounds M/K/N=16/16/8, 17/67/23 and 256/256/512. The packet binds compiler,
Target compiler, native launch library and source hashes to this checkout.
Generic tiling, SM80/SM90 and the gfx1151 shared-store verifier fixture pass as
compiler evidence only. Timings are diagnostic; there is no selector promotion
or speedup claim. Fused macro-CTA reuse optimization remains a measured follow-up.

## ROCM-K-UNROLL-IMAGE-CACHE-2026-10-02

Native ROCm image reuse now retains K-unroll configuration in Tile and directive compilation keys. Exact gfx1151/gfx1201 fp16/bf16 plain and bias+ReLU numerical tests passed, with ragged M/N/K and distinct images for different recipes. Independent six-shape static/fused packets separate kernel and end-to-end time. LDS/split-K/scaled image keys remain open; no performance promotion claim.

## ROCM-LDS-IMAGE-CACHE-2026-10-02

Owner E2E-REAL-6-ROCM-MATMUL-CACHE. Native MLIR identity projection now supports the portable LDS matmul image path while preserving staging and wave geometry in both compilation keys. Exact gfx1151/gfx1201 fp16/bf16, ragged shapes, bias+ReLU and changed-wave numerical checks passed. Nonportable LDS directives remain rejected. Independent static/dynamic fused packets separate kernel and end-to-end time. Split-K/scaled keys remain open.

## ROCM-SPLIT-PARTITION-IMAGE-2026-10-02

ROCm Target now preserves macro-K panel count and static split partition K. Native image projection retains those physical dependencies while dropping M/N launch scaffolding. Exact gfx1201 fp16/bf16 numerical and ordered-reduction tests passed; M/N siblings reuse an image and a different K partition misses. gfx1151 unsplit parity passed. Earlier image-key packets that dropped macro-K remain historical. Separate partial/reduction event and uninstrumented end-to-end packets are recorded; scaled image keys remain open.

### 2026-10-02 — gfx1201 W8A8 native Target consumer

Owner: [ROCM-FP8-BLOCKSCALE-1](INTEGRATED_COMPILER_PLAN.md#rocm-fp8-blockscale-1)

PRs: Uncommitted; sync ROCM-W8A8-TARGET-CONSUMER-2026-10-02.

Outcome: Production Graph/Schedule/Tile W8A8 packages now compile checked native
ROCm Target IR, preserving exact scale policy, KN/NK layout and physical
schedule. Tajasaurus gfx1201 passed 199 device/package checks; the final
15 Target-vs-Tile cases passed again after provenance recording. Super-Bear
host WSL passed 415 package/registry checks. Three correctness-gated rows
record kernel and separate public-runtime launch timings with active GPU,
compiler and matching source fingerprints. All four backend queues assess
this ROCm-specific change. No sibling physical proof, AITER speedup or
selector promotion is claimed.

Remaining: Shape-independent scaled image identity remains open; static LDS
grid and K-group dependencies must be preserved or generalized with proof.
Broader short/ragged-K performance obligations remain open.

Evidence: [Exact gfx1201 packet, timing domains and limitations](../../../benchmarks/baselines/rocm_w8a8_target_consumer_20261002/README.md).

<!-- entry-fields:end -->

### 2026-10-02 — gfx1201 register W8A8 runtime-shape image identity

Owner: [ROCM-FP8-BLOCKSCALE-1](INTEGRATED_COMPILER_PLAN.md#rocm-fp8-blockscale-1)

PRs: Uncommitted; sync ROCM-W8A8-REGISTER-IMAGE-CACHE-2026-10-02.

Outcome: Native MLIR projects verified one-wave global W8A8 Target directives
to an explicit runtime-shape contract. The native typed generator uses runtime
M/N/K, while checked descriptors retain shape guards and Schedule ancestry.
KN/NK and fp32/bf16 each reuse one image across three distinct M/N/K shapes;
changing scale_n misses. The scaled Target handoff now preserves raster
order/group, fixing an omission exposed by the physical-key audit. The
compiler/config-bound Target product cache avoids repeating native subprocesses
for the same program. Current gfx1201 native/device/package checks: 240 passed.
Host WSL package/diagnostic/pass gates: 415 passed. Twelve benchmark arms record
independent numerical checks and separate frontend/package/kernel/launch times.

Remaining: LDS grid/edge/K-stage dimensions and MXFP4/folded scaled image keys
remain open, as do broader W8A8 short/ragged-K performance obligations. No
AITER speedup, selector promotion or sibling exact-device claim.

Evidence: [Exact gfx1201 packet and validation](../../../benchmarks/baselines/rocm_w8a8_register_cache_20261002/README.md).

<!-- entry-fields:end -->

### 2026-10-02 — gfx1201 LDS W8A8 runtime-M/N image identity

Owner: [ROCM-FP8-BLOCKSCALE-1](INTEGRATED_COMPILER_PLAN.md#rocm-fp8-blockscale-1)

PRs: Uncommitted; sync ROCM-W8A8-LDS-MN-IMAGE-CACHE-2026-10-02.

Outcome: Native MLIR projects verified LDS W8A8 M/N to runtime dimensions with
explicit whole/partial panel edge classes and positive static K. Typed native
raster geometry uses runtime M/N; the checked launch validates K/class before
acquiring the image. fp32/bf16 each reuse one eight-wave 128x128 image across
three M/N shapes, while K and edge-class changes produce distinct cold images.
Default row-major machine text is byte-identical to the static control;
grouped/column raster numerical parity passes. gfx1201 native/device/package
checks: 249 passed. Host WSL registry/package checks: 415 passed; shared
cache/lifetime: 31 passed, 33 environment/capability skips. Twelve matched
benchmark arms separate frontend, package, kernel and public-launch timings.
Kernel median ratios are within 0.3% of unity; no AITER speedup or promotion.

Remaining: Full runtime-K LDS reuse, MXFP4/folded scaled image identity and
broader W8A8 performance obligations remain open. New edge classes remain
physical image keys. No sibling exact-device parity is inferred.

Evidence: [Exact gfx1201 code-preservation and timing packet](../../../benchmarks/baselines/rocm_w8a8_lds_mn_cache_20261002/README.md).

<!-- entry-fields:end -->

### 2026-10-02 — gfx1201 LDS W8A8 runtime-K image identity

Owner: [ROCM-FP8-BLOCKSCALE-1](INTEGRATED_COMPILER_PLAN.md#rocm-fp8-blockscale-1)

PRs: Uncommitted; sync ROCM-W8A8-LDS-RUNTIME-K-2026-10-02.

Outcome: Native MLIR projects LDS W8A8 K to a checked runtime value alongside M/N,
retaining whole/partial panel classes, scale semantics, staging and raster
keys. The scale-group loop bound and final-prefetch clamp use runtime K.
The positive whole-group ABI contract is exposed to LLVM with an assumption;
without it the fictitious zero-trip loop edge caused 15 spilled VGPRs in
ragged kernels. The corrected implementation passes existing no-spill gates.
Exact gfx1201 gates: 253 passed plus six prefetch boundary checks.
Host WSL registry/cache/package: 441 passed, 22 hardware skips.
Five K values share one image per output. Twelve paired benchmark arms
measure 0.16–2.59% runtime-K kernel overhead with separate package/public timings.

Remaining: Reduce runtime-K overhead, MXFP4/folded image keys and wider
W8A8 performance closure. No AITER speedup or sibling execution is claimed.

Evidence: [Runtime-K correctness and timing packet](../../../benchmarks/baselines/rocm_w8a8_lds_runtime_k_cache_20261002/README.md).

<!-- entry-fields:end -->

### 2026-10-02 — gfx1201 native folded full-K scale epilogue

Owner: [ROCM-MXFP4-W4A8-1](INTEGRATED_COMPILER_PLAN.md#rocm-mxfp4-w4a8-1)

PRs: Uncommitted; sync ROCM-FOLDED-NATIVE-EPILOGUE-2026-10-02.

Outcome: Internal verified Tile folded-scale operation has a native ROCm
fragment conversion. Typed e4m3 full-K MMA, folded scale and bf16 store execute
on gfx1201; all six normal/overflow/underflow/zero/ragged cases match an
independent fp64 oracle bitwise after bf16 rounding. Combined native verifier,
new device and W8A8 regression gates: 110 passed. Host operation/dtype-attribute,
diagnostic/pass checks: 318 passed. No Python HIP shader produces this fixture.

Remaining: The folded Graph/Target package still uses Python-emitted HIP.
Connect its native LDS producer, checked ABI and selected schedule, then prove
package numerics, image-key reuse and production device/end-to-end timings.
Sibling native consumers and exact-device parity are not established.

Evidence: [Native folded epilogue proof](../../../benchmarks/baselines/rocm_folded_native_epilogue_20261002/README.md).

<!-- entry-fields:end -->

### 2026-10-02 — gfx1201 folded Graph native package and matched timing

Owner: [ROCM-MXFP4-W4A8-1](INTEGRATED_COMPILER_PLAN.md#rocm-mxfp4-w4a8-1)

PRs: Uncommitted; sync ROCM-FOLDED-NATIVE-PACKAGE-2026-10-02.

Outcome: Authored Graph/Schedule/Tile folded Target generates native
LDS typed-view/fragment/WMMA producer and full-K numerical epilogue, followed
by ROCDL/LLVM image generation. Runtime distinguishes expanded memref ABI
from HIP control raw pointers and rejects mismatches before device probing.
Five random/ragged package cases match fp64 and HIP bitwise. Native Target
validates numerical policy, physical schedule and conflicting pass options.
Both existing benchmark launchers honor the image ABI. Host WSL folded
frontend/schedule/registry gates: 320 passed. Combined native Target/Tile/ABI
adapter and folded device gate: 72 passed. Production M256/K5120
N4096/8192/16384 passes full HIP parity and independently sampled reference.
Seven paired device-clock/event trials measure 2.8–8.2% native overhead;
177 versus 123 VGPRs, both 25.6 KiB LDS, no spills. Public wall time is separate.

Remaining: Register/staging/barrier optimization, shape-independent folded
image keys, wider scale/layout and checkpoint/model quality proof. Resource
differences are candidate causes, not profiler attribution. No Radiance or
performance promotion claim. gfx1151 cannot emit FP8 WMMA; Apple/NVIDIA/x86
require a consumer only if the internal operation is admitted.

Evidence: [Native folded package proof and paired timing](../../../benchmarks/baselines/rocm_folded_native_package_20261002/README.md).

<!-- entry-fields:end -->

### 2026-10-02 — gfx1201 folded native scheduling attribution experiments

Owner: [ROCM-MXFP4-W4A8-1](INTEGRATED_COMPILER_PLAN.md#rocm-mxfp4-w4a8-1)

PRs: Uncommitted; sync ROCM-FOLDED-NATIVE-OPT-2026-10-02.

Outcome: Three correctness-gated compiler experiments tested complete-panel
bounds, K16 issue boundaries and fragment retirement. The emitter already
derives static whole-panel proofs; redundant setup gave identical machine
instructions and was removed. K16 boundaries changed the stream but retained
177 VGPRs. Fragment-store boundaries gave 176 versus HIP 123, no spills, and
2.6–7.1% native/HIP overhead in a separate seven-trial packet. Neither scheduling
candidate resolved register pressure or established meaningful improvement;
both were removed from active code. Candidate patch and exact-device packets
remain reproducible evidence. Compiler/device gates: 58 passed for the first
two candidates, 67 for retirement. Three strengthened package cases separately
prove finite overflow recovery, nonzero underflow recovery and zero partials
against a wide oracle and matched HIP. The timing recorder now adapts short
marker windows while retaining rejected spans and normalizing actual counts.
Host registry/ABI gates: 299 passed. Restored native schedule and strengthened
package recovery gate: 67 passed, no warnings; audit/navigation/registry: 311 passed.

Remaining: Compiler register-liveness attribution, shape-independent folded
image keys, broader layouts/scales and checkpoint quality. Peak resource counts
do not locate the live register pressure. No profiler or Radiance claim.
Sibling physical schedules are not applicable; shared numerical semantics
are unchanged, and no sibling execution parity is implied.

Evidence: [Fragment retirement and numerical recovery](../../../benchmarks/baselines/rocm_folded_native_fragment_retirement_20261002/README.md), [K16 boundary](../../../benchmarks/baselines/rocm_folded_native_fragment_boundary_20261002/README.md), [Complete panels](../../../benchmarks/baselines/rocm_folded_native_complete_panels_20261002/README.md).

<!-- entry-fields:end -->

### 2026-10-02 — gfx1201 folded native runtime-M/N image reuse

Owner: [ROCM-MXFP4-W4A8-1](INTEGRATED_COMPILER_PLAN.md#rocm-mxfp4-w4a8-1)

PRs: Uncommitted; sync ROCM-FOLDED-RUNTIME-MN-2026-10-02.

Outcome: Native generation and image projection share one strict folded Target
validator. Projection removes launch-only M/N and Schedule ancestry from image
identity while retaining K, whole/partial classes and every physical/module
key. Descriptors retain original Schedule/Tile/authored Target/payload hashes,
exact buffer guards and geometry; runtime checks image policy/classes/fixed K
and launch geometry before HIP. Three random/ragged shapes share one image
with distinct ancestry/payloads; K and physical changes miss. Combined native
and device gate: 84 passed; physical-key/frontend: 42; geometry: 3. Shared
native cache gates on the owning device: 130 passed, 9 owning-envelope skips.
Host WSL registry/frontend/payload/ABI: 307 passed. M256/K5120 N4096/8192/16384
uses one production image: first package 324 ms, cross-N cache hits 90–123 ms.
Paired runtime/static device ratios 0.9920–1.0314; no spills. Public launch cost
is separate. All outputs match static native, HIP and sampled independent oracle.

Remaining: Runtime K, exact per-K32 native migration, register/column-cost
attribution, wider scale/layout and checkpoint quality. Image reuse does not
remove frontend/payload-binding or public transfer/module overhead. No GPU
speedup or sibling execution claim; gfx1151 cannot emit FP8 WMMA.

Evidence: [Folded native runtime-M/N image reuse](../../../benchmarks/baselines/rocm_folded_native_runtime_mn_20261002/README.md).

<!-- entry-fields:end -->

### 2026-10-02 — gfx1201 folded native runtime-K image reuse

Owner: [ROCM-MXFP4-W4A8-1](INTEGRATED_COMPILER_PLAN.md#rocm-mxfp4-w4a8-1)

PRs: Uncommitted; sync ROCM-FOLDED-RUNTIME-K-2026-10-02.

Outcome: The native folded Target projector and generator admit an explicit
runtime full-K extent while preserving K64 physical stages, full-K f32
accumulation and one row-reference epilogue. The native frontend defaults to
runtime M/N/K image identity. Checked ABI validates positive K64 multiples,
buffer capacity/payload/storage, panel classes and launch geometry before
device access. Exact gfx1201 numerical/cache/ABI gate: 91 passed after default
promotion; host registry/folded guards: 329; shared image/cache families:
72 passed, 9 skipped. Physical-key isolation: 10 passed; final audit/registry gates: 311 passed. Four production cases reuse one HSACO; paired native
runtime/static ratios are 0.9957–1.0185. Native/HIP ratios are 1.0224–1.2148,
with native 177 versus HIP 123 VGPRs and no spills. All four sibling queues
assess the scoped ABI/Target change; gfx1151 FP8 WMMA is not applicable.

Remaining: Native register/column-cost attribution, exact per-K32 native
migration and wider layout/model-quality proof. Shape-independent folded
image reuse does not close these obligations or the larger program.

Evidence: [Runtime-K packet](../../../benchmarks/baselines/rocm_folded_native_runtime_k_20261002/README.md), exact-device tests and separate public/device timings.
<!-- entry-fields:end -->

### 2026-10-02 — gfx1201 native folded instruction-bound liveness

Owner: [ROCM-MXFP4-W4A8-1](INTEGRATED_COMPILER_PLAN.md#rocm-mxfp4-w4a8-1)

PRs: Uncommitted; sync ROCM-FOLDED-LIVENESS-2026-10-02.

Outcome: The new native pressure recorder translates only compiler-emitted
LLVM/ROCDL, runs target-aware LLVM optimization and compares diagnostic selected
instructions/resources against actual HSACO. Both candidate and restored
streams match exactly; restored instructions also match the prior runtime-K
packet. Restored virtual pressure peaks at 165 VGPRs in a K-loop WMMA:
64 accumulator, 48 LDS fragment, 20 prefetch and 25 address/state VGPRs are
listed live, with an 8-register instruction-point difference. Physical
allocation remains 177 VGPRs/29 SGPRs. A final-stage peel removed unused terminal
prefetch/drain but increased virtual peak to 187 and physical allocation to 180;
four timing cases established no consistent improvement. The candidate was
removed, its patch/evidence retained, and the restored compiler rebuilt.
Candidate/restored compiler/device gates each passed 96 overlapping tests.
All sibling queues assess this physical gfx1201 probe as not applicable.

Remaining: Reduce LDS-fragment/address overlap with a native schedule change
and exact-device timing; exact per-K32 migration, wider layout/model proof.
Virtual liveness is not a hardware counter or final physical allocation.

Evidence: [Instruction-bound report and rejected terminal peel](../../../benchmarks/baselines/rocm_folded_terminal_prefetch_20261002/README.md).
<!-- entry-fields:end -->

### 2026-10-02 — gfx1201 folded native cold-scale branch likelihood

Owner: [ROCM-MXFP4-W4A8-1](INTEGRATED_COMPILER_PLAN.md#rocm-mxfp4-w4a8-1)

PRs: Uncommitted; sync ROCM-FOLDED-COLD-BRANCH-2026-10-02.

Outcome: Two native candidates were tested with exact-instruction-bound LLVM
reports and independently compiled interleaved native reference images.
K16 DS-read/WMMA grouping reduced virtual peak from 165 to 154 but retained
177 physical VGPRs and regressed two N4096 cases by 3.4–4.4%; it was removed.
A regular-scale llvm.expect hint preserves ordered FP64 recovery and reaches
optimized native LLVM as expected 2000:1 weights. The hint changes instructions
with unchanged 177 VGPRs/no spills. Compiler/device/verifier gate: 101 passed;
native hint-survival: 1 passed. Initial and repeated measurements agree on
short-K improvement; the nine-trial repeat yields native/reference 0.9112
and 0.9424 for K1024/2048, and 0.9940–1.0051 for three long-K cases.
Native/HIP remains 1.0232–1.1134. All output hashes match and independent sampled
oracles pass before timing; public wall time is separate. The hint is retained,
and all four queues assess sibling outcomes.

Remaining: Device windows include dispatch gaps and individual samples remain
noisy; prove pure kernel/host dispatch separation, wider/ragged envelopes,
exact per-K32 native migration and model quality. No hardware counter, isolated
epilogue duration, Radiance comparison or sibling parity claim follows.

Evidence: [Branch-hint repeat packet](../../../benchmarks/baselines/rocm_folded_cold_branch_20261002/README.md), [removed panel grouping](../../../benchmarks/baselines/rocm_folded_panel_group_20261002/README.md).
<!-- entry-fields:end -->


### 2026-10-02 — gfx1201 folded checked package HIP graph timing

Owner: [ROCM-MXFP4-W4A8-1](INTEGRATED_COMPILER_PLAN.md#rocm-mxfp4-w4a8-1)

PRs: Uncommitted; sync ROCM-FOLDED-GRAPH-WINDOWS-2026-10-02.

Outcome: Checked native and frozen HIP package images now have explicit-stream
benchmark graph capture/replay with node census, separate build cost, ordered
clock/event witnesses and dependency lifetime cleanup. Both marked/unmarked
graphs overwrite all three poisoned outputs before timing. Seven alternating
trials on RX 9070 XT gfx1201 leave a native/HIP graph gap of 2.2–10.4% across
four M256 shapes. Ordinary-versus-graph medians differ only modestly and in
mixed directions, so repeated host dispatch is not the dominant gap here.
No runtime ABI, public graph route or sibling execution claim changed.

Remaining: Native instruction scheduling/epilogue investigation, exact per-K32
migration, wider short/ragged-K, layouts and model-quality proof remain open.
Graph windows include GPU graph dispatch and markers; no profiler isolation
or hardware counters are implied. Graphify update remains unavailable in WSL.

Evidence: benchmarks/baselines/rocm_folded_graph_windows_20261002/README.md
records exact-device identity, bitwise/oracle checks, source/image fingerprints,
capture node census and seven-trial ordinary/graph windows.
<!-- entry-fields:end -->


### 2026-10-02 — gfx1201 folded typed-fragment read-barrier experiment

Owner: [ROCM-MXFP4-W4A8-1](INTEGRATED_COMPILER_PLAN.md#rocm-mxfp4-w4a8-1)

PRs: Uncommitted; sync ROCM-FOLDED-READ-BARRIER-2026-10-02.

Outcome: A typed K64 fragment capture moved the LDS read-completion barrier
before the WMMA chain while retaining every barrier outside the wave guard.
64 compiler/frontend/verifier and 38 exact-device package tests passed.
Native instruction-bound LLVM pressure and seven-trial graph replay against
the preserved hinted native compiler showed unchanged 165 virtual/177
physical VGPRs and a roughly 1.8% long-K N4096 regression. Other rows were
near-flat. The candidate was removed; its patch and exact-device receipts
remain for attribution history. No shared IR/ABI or sibling runtime changed.

Remaining: Native live-range/code generation, exact per-K32 migration,
wider short/ragged-K, layout and model-quality proof remain open. The
previously proved producer and native LLVM scale hint remain active.
Graph dispatch/markers remain within graph timing; no hardware counters or
isolated kernel-time attribution is claimed.

Evidence: benchmarks/baselines/rocm_folded_read_barrier_20261002/README.md,
candidate.patch, compiler-tests.txt, device-tests.txt, gfx1201.json and
pressure/pressure.json preserve the measured rejection and numerical proof.
<!-- entry-fields:end -->


### 2026-10-02 — SM120 canonical tensor reduction native migration

Owner: [W1.1](INTEGRATED_COMPILER_PLAN.md#w11)

PRs: Uncommitted; sync NVIDIA-W1.1-CANONICAL-TENSOR-REPLAY-2026-10-02.

Outcome: Explicit SM120 generic tensor M/N/K reductions recover their Graph
matmul only after complete function equivalence to a verified generic tiling
replay, including accumulator, padding, bounds, pipeline and epilogue. Native
Schedule/Tile then materializes checked pointer-backed views and typed
fragments, with byte-identical Tile parity to the direct Graph route. 65
producer tests, including exact RTX 5070 fp16/BF16 packages and tampered
lineage cases, pass. Twelve queried-device benchmark rows retain independent
oracle checks plus separate compilation, resident CUDA-event C++ launch-window
and full host package wall timings. The historical tensor constructors remain
only on older SM targets; they are not SM120 producers.

Remaining: Arbitrary tensor producer graphs, noncanonical reductions/initial
accumulators, dynamic generic loop recovery, additional activation device proof
and older architectures remain open. The event timing includes potential
driver dispatch gaps and establishes no isolated instruction-time or speedup
claim. No shared op/dtype or runtime ABI changes; sibling native pipelines
remain architecture-owned.

Evidence: benchmarks/baselines/nvidia_sm120_canonical_tensor_replay_20261002/README.md,
rtx5070.json, producer-tests.txt and lit.txt; tests/unit/test_sm120_legacy_scheduled_producer.py
proves numerical packages and full-replay refusals.
<!-- entry-fields:end -->

### 2026-10-02 — registered SM120 canonical tensor pipeline closure

Owner: [W1.1](INTEGRATED_COMPILER_PLAN.md#w11)

PRs: Uncommitted; sync NVIDIA-W1.1-REGISTERED-TENSOR-2026-10-02.

Outcome: Early whole-function tiling replay recovers canonical tensor reductions
before registered SM120 Graph prepasses. This fixes inner-K16 scheduling that
discarded the complete shape and fused operands. Twelve RTX 5070 fp16/bf16,
plain and bias/ReLU/residual rows have full-pipeline PTX identical to the
numerically executed checked Schedule package. 414 focused gates passed.

Remaining: arbitrary/noncanonical producers, dynamic generic loop recovery and
other activation hardware proof remain open. Shared Graph-to-Schedule guards
isolated canonical K steps; Apple, ROCm and x86 need their own recovery before
admitting those loops. No sibling exact-device parity or universal W1.1 closure.

Evidence: [Registered tensor pipeline packet](../../../benchmarks/baselines/nvidia_sm120_registered_tensor_pipeline_20261002/README.md).
<!-- entry-fields:end -->

### 2026-10-02 — JIT reverse attention native execution

Owner: [E2E-REAL-6](INTEGRATED_COMPILER_PLAN.md#e2e-real-6)

PRs: Uncommitted; sync NVIDIA-JIT-ATTENTION-VJP-2026-10-02.

Outcome: The reverse-mode JIT trace now packages compiler-generated attention
forward/backward through native paired AD and verified Schedule/Tile. Requested
Q/K/V selection/order survives the physical full-gradient product. Integration
fixes correct O/LSE pair slot comparison, independent V width in canonical
Graph shape inference, storage dtype preservation and eager grouped-query
mapping. Eighteen RTX 5070 rows pass independent f64 output/LSE/gradient checks,
private saved-state mutation/repeated-backward proof and separate timing scopes.
404 focused tests passed.

Remaining: automatic bias derivatives/JVP, composed AD graphs and broader
dtype/layout policies remain open. Shared frontend/AD changes require owning
Apple/ROCm/x86 consumer proof; SM120 physical schedules are not sibling parity.

Evidence: [JIT reverse attention packet](../../../benchmarks/baselines/nvidia_jit_attention_vjp_20261002/README.md).
<!-- entry-fields:end -->

### 2026-10-02 — optional attention bias-gradient compiler foundation

Owner: [E2E-REAL-6](INTEGRATED_COMPILER_PLAN.md#e2e-real-6)

PRs: pending; sync NVIDIA-ATTENTION-BIAS-GRADIENT-2026-10-02.

Outcome: The internal checkpoint backward dialect accepts an optional fourth
result only with an exact-shaped f32 bias operand. Native Schedule replay
preserves all four result roles and hashes the expanded result contract.
Tile carries eleven pointers and seven dimensions for the bias-gradient form;
NVIDIA lowering assigns each bias element one deterministic writer, using
saved O/LSE and omitting the extra Q.K scale from its derivative. Existing
three-gradient checkpoint packages are unchanged.

Evidence: Both tessera-opt and tessera-nvidia-opt were rebuilt on Super-Bear
WSL from current source with LLVM/MLIR 23.1.1. The focused compiler, checkpoint,
paired package, JIT interface, pass metadata, diagnostic and audit tests pass
362/362. New tests verify Schedule/Tile/PTX propagation and reject a missing
bias operand, mismatched bias extent and wrong gradient dtype.

Remaining: This is compiler artifact proof. Automatic AD export, checked
runtime ABI, exact RTX 5070 numerical execution and separate device/end-to-end
timings for the fourth result remain open. Apple, ROCm and x86 have shared
dialect follow-up required and no native checkpoint admission or device proof
for this result. All four backend queues record the same synchronization key.
<!-- entry-fields:end -->

### 2026-10-02 — checked attention bias-gradient package execution

Owner: [E2E-REAL-6](INTEGRATED_COMPILER_PLAN.md#e2e-real-6)

PRs: pending; sync NVIDIA-ATTENTION-BIAS-GRADIENT-2026-10-02.

Outcome: The fourth checkpoint gradient projects a distinct checked ABI with
eleven buffers and seven dimensions. Its entry symbol identifies the
bias-gradient contract; host, resident and timing bridge paths reject
mismatched buffer counts and include all four gradient element counts in
launch geometry. Overflow checks precede launch. Existing three-gradient
checkpoint packages retain their ABI.

Evidence: Super-Bear RTX 5070 runs six full/causal rows across GQA, ragged
sequence pairs, batch two and independent value width. Forward O/LSE and
all four gradients match the independent float64 oracle, with maximum
absolute gradient error 1.43987783e-07. Host/resident outputs are NaN-poisoned.
Both matching compiler and CUDA bridge were rebuilt. 400 focused compiler,
checkpoint, launch, registry, metadata, diagnostic and audit tests pass.
The packet records separate resident CUDA-event windows and host end-to-end
samples, exact GPU UUID/driver and source/compiler/bridge/image fingerprints.
[Packet](../../../benchmarks/baselines/nvidia_checkpoint_bias_gradient_20261002/README.md).

Remaining: automatic paired-AD bias export, JIT requested gradient ordering,
private resident tape bias-gradient ownership, and general composed AD.
Apple, ROCm and x86 require their own native checkpoint contracts and owning
device proofs. All four queues share this synchronization key. No speedup,
default-dispatch promotion or sibling parity claim follows.
<!-- entry-fields:end -->

### 2026-10-02 — public JIT bias gradients with private checkpoints

Owner: [E2E-REAL-6](INTEGRATED_COMPILER_PLAN.md#e2e-real-6)

PRs: pending; sync NVIDIA-ATTENTION-BIAS-GRADIENT-2026-10-02.

Outcome: The public traced additive bias is a typed Graph operand with dense
pure semantics. Native reverse AD admits exact-shaped static f32 score bias,
returns its fourth gradient and saves forward O/LSE. Native checkpoint export
derives canonical bias/result roles and retains selected/reordered gradient
lineage. The resident tape validates the eleven-buffer contract, owns a private
bias copy and includes the fourth output range in launch geometry.

Evidence: Super-Bear RTX 5070 passed eighteen public JIT reverse rows: full/
causal, grouped query, ragged sequences, batch two, independent value width,
bias-only and reordered gradient requests. Caller Q/K/V/bias mutation,
repeated backward, changed cotangents and closed-frame refusal are checked.
Maximum absolute gradient error against the independent float64 oracle is
1.62360434e-07. 441 focused AD, checkpoint, tape, op/dtype registry, metadata,
diagnostic and audit tests pass, including Q-only biased exact-device activity,
saved O/LSE matching and failure-time allocation rollback. Both compiler
tools were rebuilt; source and compiler fingerprints accompany the packet.
Capture and backward wall timings remain separate from the prior explicit
checkpoint CUDA-event windows. No speedup or default-dispatch claim follows.
[Packet](../../../benchmarks/baselines/nvidia_jit_attention_bias_vjp_20261002/README.md).

Remaining: broadcast-bias reductions, bias JVP/higher derivatives, wider
dtype/layout, general composed graphs and sibling native checkpoint consumers.
All four backend queues assess the shared AD contract; no NVIDIA device or
physical schedule evidence transfers to Apple, ROCm or x86.
<!-- entry-fields:end -->

### 2026-10-02 — native attention frontend argument and gradient roles

Owner: [E2E-REAL-6](INTEGRATED_COMPILER_PLAN.md#e2e-real-6)

PRs: pending; sync NVIDIA-ATTENTION-ARGUMENT-ORDER-2026-10-02.

Outcome: The old automatic wrapper used frontend argument indices as physical
gradient indices and captured inputs in canonical Q/K/V/bias order. Native
paired checkpoint export now derives the frontend input permutation from
typed block arguments. Schedule hashes that map; replay and artifact
projection check it. The VJP wrapper reorders capture inputs and maps requested
cotangents through this compiler-owned contract. Keyword bias uses its
original frontend position.

Evidence: Super-Bear RTX 5070 passes 28 canonical/permuted biased and unbiased
frontend cases across full/causal, GQA, batch two, ragged sequence pairs and
independent value width. Expected input and gradient maps are independent
of compiler metadata. Caller mutation and repeated backward preserve the
private generation. Maximum gradient error against float64 is 2.55680632e-07.
452 focused tests pass, including Schedule mutation, invalid permutations,
capture arity, keyword bias and owning-device numerical checks. Both native
compiler tools were rebuilt. Capture/backward wall timings have their own
samples; no kernel-time or speedup claim follows.
[Packet](../../../benchmarks/baselines/nvidia_attention_argument_order_20261002/README.md).

Remaining: aliased/composed producer inputs, broadcast-bias reduction,
permuted JVP/higher derivatives, broader dtype/layout and sibling native
checkpoint consumers. All four backend queues assess this shared contract.
<!-- entry-fields:end -->

### 2026-10-02 — gfx1201 fragment buffering and C transpose retunes

Owner: [ROCM-MXFP4-W4A8-1](INTEGRATED_COMPILER_PLAN.md#rocm-mxfp4-w4a8-1)

PRs: pending; sync ROCM-FOLDED-RETUNES-2026-10-02.

Outcome: Column-first consumption and row-first delayed A/B/half-A loads did
not improve the measured envelope. Two-panel typed fragment buffering spreads
LDS reads between WMMA chains, retaining three static split-barrier pairs;
N8192/K5120 improves 3.6% with 2-3% short-K regressions. C transposition reuses
the A byte slab with typed views after its lifetime ends. Wave-private gather
uses native wave barriers: 145 physical VGPRs, 25,600 LDS bytes, no spills,
four static workgroup signal/wait pairs versus twelve in the collective
prototype and three in the reference. N8192 improves 1.3%, short K regresses.
Native ISA matches the separate LLVM register-pressure diagnostic.

Evidence: Matching LLVM/MLIR 23.1.1 tools on Tajasaurus RX 9070 XT/gfx1201.
89 focused checks pass for each consumption/prefetch/deep-buffer candidate.
All 38 exact-device C-transpose checks pass after the byte-view repair;
three structural tests still expect original global Tile stores. These were
not weakened. Seven alternating graph-window trials poison and validate all
three resident copies before timing and check outputs again after timing.
Public runtime-launch wall samples are separate. No isolated-kernel,
occupancy, Radiance, speedup promotion, or sibling parity claim follows.
[Deep fragments](../../../benchmarks/baselines/rocm_folded_deep_fragments_20261002/README.md).
[C transpose](../../../benchmarks/baselines/rocm_folded_c_lds_transpose_20261002/README.md).

Remaining: Experimental patches/compiler snapshots are preserved and active
source is restored. BF16 LDS staging after scale recovery, prologue A/B
overlap, and verified persistent Schedule/Tile worker assignment are next.
Persistent folded matmul is not implemented. Exact-per-K32 migration, broad
shape/layout/model quality and native performance closure remain open.
Apple/NVIDIA/x86 have no changed retained contract; gfx1151 FP8 WMMA is not
supported. All four queues record architecture-specific outcomes.
<!-- entry-fields:end -->


### 2026-10-03 — standard E8M0 Tile consumer foundation

Owner: [ROCM-FP8-BLOCKSCALE-1](INTEGRATED_COMPILER_PLAN.md#rocm-fp8-blockscale-1)

PRs: pending; sync ROCM-E8M0-TILE-2026-10-03.

Outcome: tile.fragment_scaled_accumulate distinguishes fp32 scale buffers
from standard E8M0 raw-byte buffers through an explicit scale_format.
The ROCm lowering decodes code 0 as 2^-127 and code 255 as NaN, scales
each isolated f32 partial with both decoded scales in f64, rounds it once
to f32, then joins the running f32 accumulator. This avoids intermediate
scale-product overflow and underflow and does not reuse the folded ABI's
code-zero-to-zero interpretation. A native K32 two-WMMA Tile fixture on
Tajasaurus RX 9070 XT passes twelve signed, zero, all-code and ragged-edge checks.
The matching gfx1201 build passed 433 combined device/registry tests and three
focused MLIR verifier/FileCheck commands.

Evidence: [Native Tile consumer packet](../../../benchmarks/baselines/rocm_e8m0_tile_scale_20261003/README.md).
[Three-format decision gates](../../../benchmarks/baselines/rocm_folded_bf16_lds_20261003/README.md).
Verifier negative cases cover mismatched buffer types and unknown scale formats.
Existing FP8 uniform-scale lowering remains covered.

Remaining: MXFP8 Graph/Schedule derivation, named physical package contract,
checked runtime ABI and exact-device performance are open. This is not an
end-to-end MXFP8 package or a canonical dtype promotion. All four backend
queues record sibling follow-ups. The older unfenced wave-private C-through-LDS
prototype did not prove memory ordering; only the ordered follow-up's evidence
may support that claim. Persistent schedules and a general short/long selector
remain unimplemented, and FP8/MXFP8/MXFP4 remain mandatory decision gates.
<!-- entry-fields:end -->


### 2026-10-03 — MXFP8 native Graph and Schedule integration

Owner: [ROCM-FP8-BLOCKSCALE-1](INTEGRATED_COMPILER_PLAN.md#rocm-fp8-blockscale-1)

PRs: pending; sync ROCM-MXFP8-SCHEDULE-2026-10-03.

Outcome: A typed textual frontend states FP8 E4M3 payloads and raw E8M0 bytes.
Graph->Schedule validates K32 groups, per-column scales, exact numerical mode
and buffer extents/storage, then derives separate KN/NK physical contracts.
Verified Schedule/Tile/Target passes carry the format into native typed
fragments, ROCDL/LLVM and HSACO; the package ABI names wide-scale evaluation
and cannot be confused with fp32-scale W8A8. Twelve gfx1201 native cases
prove two/four K32 groups, ragged M/N, KN/NK layouts and f32/bf16 stores.
Eight malformed-contract cases and four compiler boundary checks pass.
MXFP8 admission also requires the caller to explicitly state fp32 accumulation;
a missing or conflicting accumulator policy is rejected before derivation.
The integrated ROCm/device/registry run passed 608 tests, with one unavailable
legacy runtime test skipped. Shared registry gates passed 322 tests on
Super-Bear; the integrated NVIDIA build passed 86 existing device cases.
Newer NVIDIA producer/tail/epilogue changes were preserved when merging the
shared pass. The initial NVIDIA check overlapped linking; its nine
PermissionError cases are excluded, and the complete post-build run passed.

Evidence: [Native MXFP8 compiler packet and reproduction](../../../benchmarks/baselines/rocm_mxfp8_schedule_20261003/README.md).

Remaining: checked production descriptor/runtime binding, shape/image cache
identity and separate device/E2E timings. Execution uses a raw diagnostic HIP
launcher; native image proof does not establish checked-runtime completion.
The one-wave seed is numerical integration, not a promoted performance rule.
Canonical bundled MXFP8 storage and sibling native consumers remain open;
FP8/MXFP8/MXFP4 evaluation still precedes a persistent or short/long decision.
<!-- entry-fields:end -->


### 2026-10-03 — MXFP8 checked package and image identity

Owner: [ROCM-FP8-BLOCKSCALE-1](INTEGRATED_COMPILER_PLAN.md#rocm-fp8-blockscale-1)

PRs: pending; sync ROCM-MXFP8-PACKAGE-2026-10-03.

Outcome: The native Graph -> Schedule -> Tile -> Target -> ROCDL/LLVM -> HSACO
route now binds checked E8M0 K32 descriptors to runtime.launch on gfx1201.
The native image-identity pass projects M/N/K while retaining the 16x16
one-wave seed, raw uint8 scales, per-column layout and distinct wide-scale ABI.
Checked descriptor validation rejects conflicting scale/numerical/geometry
policies before HIP access. KN/NK, f32/BF16, ragged M/N, multiple K32 groups,
NaN/subnormal scales and cross-shape image reuse are covered on RX 9070 XT.
FP8 and MXFP8 image keys remain distinct. No canonical MXFP8 storage or
sibling execution capability is promoted.

Remaining: The one-wave MXFP8 schedule is a functional baseline. Its scale
consumer and physical schedule need performance attribution/optimization;
matched MXFP4 evaluation and broader short/long coverage are required before
any selector promotion or short/long/persistent decision. Current graph
windows measure device execution plus GPU dispatch, separately from checked
runtime end-to-end staging/transfers/synchronization; they are not isolated
instruction-phase measurements. See the packet for completed validation and
timing receipts. NVIDIA existing-route parity is rechecked on RTX 5070;
Apple/x86 native MXFP8 consumers remain follow-up required.

Evidence: [Checked MXFP8 packet](../../../benchmarks/baselines/rocm_mxfp8_checked_package_20261003/README.md).
<!-- entry-fields:end -->


### 2026-10-03 — MXFP8 native exponent scale consumer

Owner: [ROCM-FP8-BLOCKSCALE-1](INTEGRATED_COMPILER_PLAN.md#rocm-fp8-blockscale-1)

PRs: pending; sync ROCM-MXFP8-EXPONENT-SCALE-2026-10-03.

Outcome: The standard E8M0 Tile consumer uses upstream LLVM ldexp.f32 with
combined signed exponents and explicit code-255 NaN propagation. Code zero
remains 2^-127. Results equal the f64 reference with one f32 rounding before
the ordered accumulator join. The shared ODS contract states equivalence
rather than requiring a particular evaluation width. The checked package ABI,
native Schedule geometry and lifetime contract are unchanged. On RX 9070 XT,
575 ROCm device/registry/package/cache tests passed, including every scale-code
pair and 116 arbitrary f32 bit patterns with four accumulators (30,408,704 results), existing FP8 and
folded MXFP4 numerics. The exponent FileCheck passed with f64 forbidden.
Two process pairs reverse reference/candidate order, preserve compiler/source/
image/ISA identities and prove identical Tile/Target/geometry/ABI contracts.
The long-K MXFP8 seed improves 3.5-4.6x; all six measured short/long layout rows
improve. The FP8 reference images remain byte-identical. RTX 5070 retained
86 existing scheduled matmul/attention numerical cases after the ODS rebuild.

Remaining: Broader shapes, matched MXFP4 physical-route/quality evaluation,
gfx1151 E8M0 consumer parity and sibling native consumers remain open.
This is a semantic-preserving scale-consumer improvement, not a short/long,
persistent or global selector promotion. Device timing includes GPU graph
dispatch and is separate from checked host end-to-end staging/transfers;
static ISA removal does not isolate dynamic phase cost.

Evidence: [Native exponent-scale packet](../../../benchmarks/baselines/rocm_mxfp8_exponent_scale_20261003/README.md).
<!-- entry-fields:end -->

### 2026-10-03 — gfx1201 three-format evaluation and partial LDS copies

Owner: [ROCM-FP8-BLOCKSCALE-1](INTEGRATED_COMPILER_PLAN.md#rocm-fp8-blockscale-1)

PRs: pending; sync ROCM-THREE-FORMATS-2026-10-03.

Outcome: A matched-input exact gfx1201 sweep now evaluates compiler-owned FP8,
standard MXFP8 and opt-in folded MXFP4 packages. Seven shapes and four arms
include both M200 regression shapes, K1536 ragged M and M256 long K. Independent
f64 references separate source quantization, FP4 folding and native arithmetic
error. Device graph timing is witnessed by HIP events and checked host E2E is
reported separately. Two processes pass all numerical gates. The sweep found
a native FP8 K32 gap: the 128x64 eight-wave LDS body's RHS vectors did not
divide among all threads. The generator now masks the final incomplete copy
round; inactive threads do not access global/LDS memory and every thread still
meets workgroup barriers. Whole copy rounds retain their prior emitted IR.
ROCM-MXFP4-W4A8-1 shares this evaluation; no dtype or physical selector is promoted.
Focused host/device proof passed 523 tests with no skips, including 48 partial
copy cases across f32/BF16 and prefetch modes 0/1/2. Native typed-IR FileCheck
and target lowering passed. Reference/candidate FP8, MXFP8 and folded MXFP4
whole-copy images and Tile/Target digests are byte-identical.

Remaining: MXFP8's one-wave seed needs an E8M0 LDS Schedule/Tile consumer for wide
grids. K16 uses explicit panel key 1; the auto default's one-panel policy is
still open. The M256 Radiance per-column gap requires a slope/phase experiment,
and model quality, rotating-copy/cache characterization and persistent scheduling
remain open. Apple, NVIDIA and x86 physical implementations are not applicable:
no shared dialect/ABI/runtime change, and ROCm results do not establish parity.

Evidence: [Three-format packet](../../../benchmarks/baselines/rocm_three_formats_20261003/README.md).
<!-- entry-fields:end -->

### 2026-10-03 — gfx1201 MXFP8 LDS schedule and guarded image reuse

Owner: [ROCM-FP8-BLOCKSCALE-1](INTEGRATED_COMPILER_PLAN.md#rocm-fp8-blockscale-1)

PRs: pending; sync ROCM-MXFP8-LDS-2026-10-03.

Outcome: Native Graph/Schedule/Tile/Target/LLVM lowering now supports eight-wave
NK E8M0 K32 LDS packages on gfx1201. Automatic wide-grid selection requires
K >= 1024; short K and small grids retain the unchanged one-wave seed. Explicit
frontend performance intent is checked and resolved natively without Python
Tile generation. Image keys retain physical recipe/edge classes and exclude
resolved intent; shape-specific package guards remain enforced. Device tests
prove guarded rejection and separately authored smaller-K packages reuse one
native image. Two paired nine-shape evaluations retain FP8/MXFP8/MXFP4 quality
gates and show 1.94–3.00x MXFP8 device improvement on five changed wide grids.
FP8/control/folded and seed MXFP8 images are unchanged. 496 tests and native
Schedule/generator/LLVM FileCheck pass on RX 9070 XT. All four queues assess
shared verifier, metadata, benchmark and runtime contracts separately.

Remaining: Multi-group K64 staging, broader shapes/layouts, source-model quality,
M256 Radiance cost attribution and persistent scheduling remain open. Sibling
native E8M0 consumers are not proved here. Device execution/dispatch timing is
separate from checked host end-to-end timing. No global format or dtype promotion.

Evidence: [MXFP8 LDS packet](../../../benchmarks/baselines/rocm_mxfp8_lds_20261003/README.md).
<!-- entry-fields:end -->

### 2026-10-03 — real gfx1201 gate up checkpoint ingest and recorder repair

Owner: [ROCM-NVFP4-INGEST-1](INTEGRATED_COMPILER_PLAN.md#rocm-nvfp4-ingest-1)

PRs: pending; sync ROCM-NVFP4-INGEST-1-QWEN3-GATE-UP-2026-10-03.

Outcome: Both real layer-0 Qwen3 gate/up projections now execute as one
16x24576x4096 native packed-MXFP4 Graph/Schedule/Tile/Target package on RX9070XT.
Pinned tensor/index/scale hashes, both source globals and row boundaries bind
the merged source. Ingested and direct-BF16 MXFP4 outputs have zero error against
decoded references. Weight relRMS versus BF16 is 9.50% for shipped NVFP4,
14.97% after ingest and 11.30% for direct MXFP4. The real globals are equal;
unequal-scale proof remains the independent synthetic 0.5/2.0 device case.
Device median is 0.16075 ms; checked host E2E is 9.53975 ms, separately from
14.818 s host ingest. No speedup or whole-model accuracy claim.

The old combined fixture mislabeled f32 scales as E8M0 and was correctly refused
by the current compiler. Packed ingest now has a dedicated native fixture; the
combined fixture now tests raw-byte standard E8M0 through WMMA and LLVM ldexp.
Recorder output agreement is checked before any timing events, with injected
bad-output cleanup proof. Temporary quality/quantizer storage is row bounded.
All four backend queues assess the benchmark/fixture change separately.

Remaining: Native conversion-op integration, whole-model/source-activation
quality, packing variants and FP8/MXFP8 checkpoint comparisons remain open.
Ingest is host checkpoint preprocessing with measured conversion policy;
only its destination operands enter the native Graph compiler route.

Evidence: [Real gate/up packet](../../../benchmarks/baselines/rocm_nvfp4_gate_up_20261003/README.md).
<!-- entry-fields:end -->

### 2026-10-03 — SM120 broadcast checkpoint native arithmetic core

Owner: [E2E-REAL-6](INTEGRATED_COMPILER_PLAN.md#e2e-real-6)

PRs: pending; sync NVIDIA-BROADCAST-CHECKPOINT-CORE-2026-10-03.

Outcome: Paired AD and the checkpoint Graph verifier preserve rank-four physical
bias shapes and cotangents. Native Schedule hashes the shape and deterministic
B/Hq/Q/K reduction policy; Tile/NVIDIA LLVM lowering broadcasts bias reads and
assigns one physical gradient owner without atomics or dense temporary gradients.
Twelve exact RTX5070 raw-resident rows pass independent f64 O/LSE and Q/K/V/bias
gradient comparisons, including every broadcast axis, combined axes, GQA and
full/causal masks. Maximum gradient error is below 1.1e-7; 32 poison words after
each physical gradient remain untouched. Native frontend/export tests found and
fixed shape projection incorrectly matching bias_shape as the logical shape.
The shared Tile verifier now accounts for the existing bias-gradient pointer.

Remaining: Checked broadcast package/tape ABI integration is required before
production admission: host copy sizes, geometry, pairing identity, shape guards,
private saved state, output allocations, stream/lifetime and public repeated
backward checks. The current package gate prevents using a dense-copy descriptor.
This is raw native arithmetic/capacity proof, not completed public JIT dispatch.
Sibling physical consumers, dynamic/lower-rank bias, JVP and higher derivatives
remain open. Device event windows include dispatch gaps and claim no speedup.

Evidence: [Broadcast core packet](../../../benchmarks/baselines/nvidia_broadcast_checkpoint_core_20261003/README.md).
<!-- entry-fields:end -->

### 2026-10-03 — Broadcast checkpoint pairing identity

Owner: [E2E-REAL-6](INTEGRATED_COMPILER_PLAN.md#e2e-real-6)

PRs: pending; sync NVIDIA-BROADCAST-CHECKPOINT-CORE-2026-10-03.

Outcome: The saved-state digest now binds physical broadcast extents and the deterministic B/Hq/Q/K reduction policy. Full-shaped bias retains its existing v1 identity. Generated producer/consumer pairs with differing physical storage are rejected before package compilation.

Remaining: The checked package ABI and private tape capture remain unfinished; the temporary package gate remains in place.

Evidence: 51 focused broadcast/resident/JVP tests pass in Super-Bear WSL; this is contract proof and does not claim new production execution. Existing exact-device routes passed 86 cases. Sibling plans retain follow-up-required status because the native broadcast checkpoint recipe is SM120-only.

<!-- entry-fields:end -->

### 2026-10-03 — Broadcast checkpoint checked packages and private tape

Owner: [E2E-REAL-6](INTEGRATED_COMPILER_PLAN.md#e2e-real-6)

PRs: pending; sync NVIDIA-BROADCAST-CHECKPOINT-PACKAGE-2026-10-03.

Outcome: Distinct broadcast checkpoint descriptors bind physical bias extents and shape guards. CUDA host copies and event paths size physical storage, and backward geometry covers the physical gradient. Resident capture privately owns physical bias and saved Q/K/V/O/LSE, with matching allocations on repeated backward. Generated public reverse AD and explicitly authored saved-LSE Graph pairs preserve shape and pairing identity. The temporary broadcast package gate has been replaced by the integrated checked route. RTX 5070 numerical recording covers every bias axis, combined reductions, GQA, full/causal masking and both sequence orderings; caller mutation and changed cotangents preserve the captured generation. Native kernels keep their seven logical scalar ABI; four additional descriptor scalars validate host storage. Sibling queues record follow-up required.

Remaining: Broadcast JVP/higher derivatives, dynamic/lower-rank bias, pruning unrequested cotangents and sibling native consumers remain open. Event windows include dispatch gaps and are separate from checked host/public capture wall timings. FP8/MXFP8/MXFP4 remain mandatory gates before any format or scheduling promotion; this f32 attention increment makes none.

Evidence: [Broadcast package packet](../../../benchmarks/baselines/nvidia_broadcast_checkpoint_package_20261003/README.md).
<!-- entry-fields:end -->

### 2026-10-03 — SM120 row-major RHS typed fragment gather

Owner: [W1.1](INTEGRATED_COMPILER_PLAN.md#w11)

PRs: pending; sync NVIDIA-ROW-MAJOR-B-CORE-2026-10-03.

Outcome: The native NVIDIA B fragment materializer consumes explicit transpose intent on row-major fp16/BF16 views. Paired K elements gather at physical pitch; bounded views retain safe-address loads and zero masking, while complete unbounded views gather both elements separately. Incompatible transpose/storage requests refuse before PTX. Exact RTX 5070 raw-Tile proof covers 16 dtype/pitch/bounds cases and multi-panel carried accumulators. Source census confirms both historical tensor-MMA constructors are registered only below SM120; canonical SM120 recovery remains native Graph/Schedule/Tile.

Remaining: The raw physical probe intentionally removes the inherited Schedule hash. Native Schedule/profile identity, checked descriptor storage, host ingress, and a named resident RHS producer-to-matmul edge still require integration. This is not a completed production route or a default strategy. FP8/MXFP8/MXFP4 remain separate required evaluation gates. Older-device/generic producer, arbitrary graph and noncanonical accumulator obligations remain open.

Evidence: [Row-major B core packet](../../../benchmarks/baselines/nvidia_row_major_b_core_20261003/README.md).
<!-- entry-fields:end -->

### 2026-10-03 — SM120 native row-major RHS Schedule and resident producer

Owner: [W1.1](INTEGRATED_COMPILER_PLAN.md#w11)

PRs: pending; sync NVIDIA-ROW-MAJOR-B-SCHEDULE-2026-10-03.

Outcome: Native Graph RHS storage selection is hashed and replayed through Schedule into typed B views and transposed fragments. Static unfused half-storage row-major RHS packages use distinct checked ABIs. Twenty host/resident matched row/column cases and eight resident RMSNorm RHS producer/consumer cases passed on RTX 5070. The edge owns its stream and allocations through result close, with no intermediate host transfer. Independent normalization and consumer oracles precede timing; producer and consumer event dispatch windows are recorded separately.

Remaining: Dynamic/fused RHS edges, arbitrary producers, isolated kernel attribution and FP8/MXFP8/MXFP4 strategy evaluation. This introduces no format/default promotion. HIP, Metal and x86 physical parity require their own architecture proof; all four queues assess the shared contract.

Evidence: benchmarks/baselines/nvidia_row_major_b_schedule_20261003/README.md, packet.json and rhs-packet.json.
<!-- entry-fields:end -->

### 2026-10-03 — SM120 traced RHS producer package integration

Owner: [W1.1](INTEGRATED_COMPILER_PLAN.md#w11)

PRs: pending; sync NVIDIA-RHS-JIT-2026-10-03.

Outcome: An explicit public JIT compile API verifies the complete traced RMSNorm RHS-to-matmul Graph, partitions existing semantic operations without backend resynthesis, checks source CFG identity and recovers CFG metadata for both native partitions. Frontend argument order is preserved. Eight RTX 5070 cases validate normalization, stored-edge matmul, owned resident lifetime and separate producer/consumer event dispatch windows.

Remaining: Automatic native JIT dispatch, general/dynamic/fused producer graphs and composed autodiff. FP8, MXFP8 and MXFP4 remain required strategy gates. CUDA evidence does not establish HIP, Metal or x86 execution; all four queues assess the shared frontend API.

Evidence: benchmarks/baselines/nvidia_row_major_b_schedule_20261003/jit-packet.json and README.md.
<!-- entry-fields:end -->

### 2026-10-03 — ordinary RHS JIT and portable native execution

Owner: [W1.1](INTEGRATED_COMPILER_PLAN.md#w11)

PRs: pending; sync NVIDIA-RHS-JIT-DISPATCH-2026-10-03.

Outcome: Ordinary static primal FP16/BF16 RMSNorm RHS-to-matmul calls now execute
checked native producer/consumer packages. The complete Graph is verified in
MLIR before partitioning. Serialized runtime artifacts preserve Graph lineage,
frontend argument order, native image/descriptor hashes, owned stream and
private buffer lifetime. Ten exact RTX 5070 benchmark rows passed ordinary and
portable replay; 60 passed, 8 skipped in 10.14s. Separate wall and producer/consumer
event dispatch windows are recorded, without a speedup claim.

Remaining: dynamic/general graphs, gamma, fused epilogues and composed AD.
FP8, MXFP8 and MXFP4 remain mandatory separate correctness/performance gates
before strategy/default promotion. Shared runtime component receipts and
stream ownership are assessed in all four queues; sibling physical execution
remains follow-up required with no transferred device evidence.

Evidence: [JIT dispatch packet](../../../benchmarks/baselines/nvidia_rhs_jit_dispatch_20261003/README.md).
<!-- entry-fields:end -->

### 2026-10-03 — LayerNorm RHS native producer integration

Owner: [W1.1](INTEGRATED_COMPILER_PLAN.md#w11)

PRs: pending; sync NVIDIA-LAYERNORM-RHS-2026-10-03.

Outcome: A second real tensor producer now reaches typed matmul B through the
native Graph/Schedule/Tile/PTX package contract. Ordinary JIT and serialized
replay pass ten exact RTX 5070 FP16/BF16 complete/ragged rows, with independent
centered-variance and matmul oracles. The new normalization manifest binds
operation kind and retains legacy RMSNorm-only schema validation. Default
epsilon, large-offset/constant rows, cached native execution, private lifetime
and semantic mutation-before-allocation checks pass. 104 passed, 8 skipped in 40.15s.
Producer/consumer event dispatch windows and end-to-end wall times are separate.

Remaining: affine normalization, dynamic/general producers and composed AD.
FP8, MXFP8 and MXFP4 remain required independent correctness/performance gates
before strategy selection. All four backend queues assess the shared JIT and
manifest changes; physical parity outside sm_120 requires owning-device proof.

Evidence: [LayerNorm producer packet](../../../benchmarks/baselines/nvidia_layernorm_rhs_20261003/README.md).
<!-- entry-fields:end -->

### 2026-10-03 — pinned gfx1201 checkpoint format decision gate

Owner: [ROCM-NVFP4-INGEST-1](INTEGRATED_COMPILER_PLAN.md#rocm-nvfp4-ingest-1)

PRs: pending; sync ROCM-CHECKPOINT-FORMAT-GATE-2026-10-03.

Outcome: Twelve native checked/resident arms pass on RX 9070 XT for the pinned
Qwen3 layer-0 gate/up weights at M=128/256,N=24576,K=4096. FP8, MXFP8 and
three folded MXFP4 source preparations retain separate weight/source-output
quality and kernel arithmetic checks. Bounded FP4 row batches preserve signed
midpoint decisions; explicit contiguous scale planes fix a transposed-source
runtime refusal. 33 focused host tests passed. At M=256, production FP8 and
NVFP4-ingested folded MXFP4 output errors are 3.686% and 15.177%; device
execution/dispatch medians are 545us and 390us, while checked end-to-end
medians are 14.38ms and 99.19ms. Folding is lossless here; the quality gap
precedes folding. Distinct physical schedules/granularities are explicit.

Remaining: native packed consumer migration and policy-gated MLIR conversion,
captured source activations/whole-model quality and host movement attribution.
Expanded folded E4M3 storage is not packed MXFP4 evidence. The inspected
packed helper still materializes hand-emitted HIP after carrier verification;
it was excluded from native compiler closure. Standard MXFP8 E8M0 and the
named folded MXFP4 zero-block scale contract must remain distinct. All four
backend queues assess this shared benchmark schema/input-preparation change;
no sibling device evidence or default strategy promotion is inferred.

Evidence: [Checkpoint format packet](../../../benchmarks/baselines/rocm_checkpoint_format_gate_20261003/README.md).
<!-- entry-fields:end -->


### 2026-10-03 — native packed MXFP4 materialization and decode A/B

Owner: [ROCM-MXFP4-W4A8-1](INTEGRATED_COMPILER_PLAN.md#rocm-mxfp4-w4a8-1)
PRs: pending; sync ROCM-PACKED-NATIVE-2026-10-03.
Outcome: The public packed-folded builder now uses Graph/Schedule/Tile/Target
native MLIR/LLVM image materialization. Native E2M1 decode preserves nearest-even
folding, zero-block semantics and post-full-K row scaling. Exact gfx1201 checked
and resident cases agree bitwise with an independent oracle; malformed runtime
contracts are refused. Shared native magnitude tables reduce packed device time
10–25% across three rows, with unchanged FP8/MXFP8/expanded control images.
Packed device time still trails expanded by 8–16% on the three synthetic rows.
All eighteen pinned gate/up format/preparation rows also pass; M256 packed
device cost is within about 1% of expanded, with NVFP4-ingested checked wall
time 57.197 ms versus 98.138 ms expanded. Format quality remains separate;
no default promotion.
Remaining: native ingest conversion, model/source-activation quality, broader
packing/shape envelopes and performance attribution. Sibling architectures
require their own physical implementation and exact-device evidence.
Evidence: benchmarks/baselines/rocm_packed_native_20261003/README.md.
<!-- entry-fields:end -->


### 2026-10-03 — native gfx1201 NVFP4 ingest physical leaf

Owner: [ROCM-NVFP4-INGEST-1](INTEGRATED_COMPILER_PLAN.md#rocm-nvfp4-ingest-1)
PRs: pending; sync ROCM-NATIVE-INGEST-LEAF-2026-10-03.
Outcome: The registered Target conversion materializes native GPU MLIR/ROCDL/LLVM
joint-SSE ingest with projection-global operands, packed codes/E8M0 exponents
and measured f64 signal/error. Seventeen contract/numerical tests pass on the
owning gfx1201 host; three synthetic resident timing rows are recorded.
The largest conversion measures 7.038 ms resident event time; CPU oracle
wall time is a separate scope. Other architectures are not physically proved.
Remaining: Graph/Schedule/Tile integration, checked package ABI, public dispatch,
pinned checkpoint conversion/consumer evidence and broad five-slice closure.
Evidence: benchmarks/baselines/rocm_native_ingest_20261003/README.md.
<!-- entry-fields:end -->

### 2026-10-05 — native saved-LSE JVP Schedule migration

Owner: [FRONTEND-IR-MEDIUM-1](INTEGRATED_COMPILER_PLAN.md#frontend-ir-medium-1)
PRs: pending; sync NVIDIA-JVP-NATIVE-SCHEDULE-2026-10-05.

Outcome: Native C++ Graph-to-Schedule and Schedule-to-Tile replace the Python
GPU arithmetic constructor and inactive-tangent string rewriting. Paired
O/LSE SSA lineage, shapes/policy, argument roles and active tangent slots
are replay-sealed. Native Tile shared buffers feed the checked arena and
LLVM/NVVM package. The nine-pointer ABI is unchanged.
298 focused tests and 98 runtime/arena regressions pass. Ten ordinary
JIT cases agree with independent finite differences. Direct and automatic
paths each pass eight saved-state cases with four tangent modes per case.
CUDA-event dispatch windows and checked allocating JVP host wall time
are separate; actual launch resources and tool/source hashes are retained.
Matching gfx1201 shared rebuild passes 18 native norm/epilogue regressions.

Remaining: General composed/dynamic/bias/dropout/value-only AD and sibling
architecture-owned JVP routes. FP8/MXFP8/MXFP4 gates remain independent.
All four backend queues are assessed; no cross-device proof transfer.

Evidence: benchmarks/baselines/nvidia_jvp_native_schedule_20261005/README.md,
jit.json, direct.json, automatic.json, artifacts/, unit.txt, runtime.txt.
<!-- entry-fields:end -->

### 2026-10-05 — native compiler orchestration overhead reduction

Owner: [FRONTEND-IR-MEDIUM-1](INTEGRATED_COMPILER_PLAN.md#frontend-ir-medium-1)
PRs: pending; sync NATIVE-COMPILE-ORCHESTRATION-2026-10-05.

Outcome: One native pass manager retains verified Graph/Schedule/Tile SSA.
Bounded stable-file exact SHA-256 reuse removes repeated tool reads/hashes.
Same-size timestamp-restored edits, atomic replacement, symlink changes and
read-time rebuilds invalidate or reject identities. Image validation is
unchanged. 321 focused and 142 shared tests pass (one skipped); ten RTX 5070
JIT oracle cases and 30 gfx1201 ingest/package cases pass.
Balanced fresh-package A/B arms retain identical arena/image/ABI bytes.
Cold identities and warm identity reuse are timed separately.

Remaining: Native in-process compiler sessions, image-validation overhead,
general frontend/AD integration and broader five-slice closure. Quantized
FP8/MXFP8/MXFP4 gates remain independent. No sibling performance transfer.

Evidence: benchmarks/baselines/native_compile_orchestration_20261005/README.md,
profile.json, package-profile.json, package-final.json, cprofile.txt,
unit.txt, shared.txt, jit.json, gfx1201-tests.txt.
<!-- entry-fields:end -->

### 2026-10-05 — native saved-LSE value-only JVP integration

Owner: [AD-RESIDUAL-EVAL-1](INTEGRATED_COMPILER_PLAN.md#ad-residual-eval-1)
PRs: pending; sync NVIDIA-VALUE-JVP-2026-10-05.

Outcome: Native AD export binds a value-only linear attention product to its
primal saved O/LSE generation. Native Schedule/Tile omits V/O reads and the
unused shared moment reduction. General TangentInterface remains unchanged;
public V-only JVP is executable without Python GPU construction.
341 focused tests, twelve ordinary JIT oracle cases and 24 fixed-QK
linearity cases pass on RTX 5070. Finite/Inf/NaN primal V does not contaminate
the derivative; maximum error is 3.11e-8. Shared storage is 512 bytes and
dispatch/wall timing remains separate. After correcting sibling source/build
drift and rebuilding the complete compiler, 40 shared AD/JVP tests and 18
existing gfx1201 norm/epilogue device regressions pass; HIP JVP remains open.

Remaining: General composed/dynamic/bias/dropout/higher AD and sibling
architecture-owned execution. Quantized FP8/MXFP8/MXFP4 gates remain open.

Evidence: benchmarks/baselines/nvidia_value_only_jvp_20261005/README.md,
unit.txt, jit.json, linearity.json, artifacts/.
<!-- entry-fields:end -->


### 2026-10-05 — native requested attention gradient pruning

Owner: [AD-RESIDUAL-EVAL-1](INTEGRATED_COMPILER_PLAN.md#ad-residual-eval-1)
PRs: pending; sync NVIDIA-VJP-ACTIVITY-2026-10-05.

Outcome: Native checkpoint export validates requested frontend input indices
and maps them through actual forward SSA argument roles. Schedule hashes
seal gradient activity; native Tile/Target emits only requested gradient
arithmetic and zero-fills the complete ABI's inactive outputs. Public reverse
execution checks that native activity matches requested result ordering.
431 focused tests and matched RTX 5070 numerical/dispatch evidence are retained.
Shared matching gfx1201 compiler tests and norm/epilogue regressions pass.

Remaining: Compact gradient allocation/launch ABIs; general composed, dynamic
and higher AD; architecture-owned sibling execution. Quantized-format gates
and full five-slice producer/ROCm route closure remain open.

Evidence: benchmarks/baselines/nvidia_vjp_activity_20261005/README.md,
unit-final.txt, packet.json, device.txt, artifacts/, vjp-activity-shared-20261005.txt,
vjp-activity-gfx1201-20261005.txt.
<!-- entry-fields:end -->

### 2026-10-05 — native ROCm movement host orchestration

Owner: [E2E-REAL-6](INTEGRATED_COMPILER_PLAN.md#e2e-real-6)
PRs: pending; sync ROCM-NATIVE-MOVEMENT-2026-10-05.

Outcome: A native C++ host service leases compiler-generated movement images,
validates checked memref extents and reuses context-owned staging buffers.
Completion, quarantine recovery and explicit context teardown protect buffer
and image lifetimes. Python remains the frontend and thin ABI binding.
Nine independent bit-exact device rows pass across gfx1151/gfx1201.
Balanced checked host-wall reductions range 5–34%; resident event dispatch
windows are recorded separately. Native artifact/runtime ABI/lifetime gates
pass. The initial gfx1151 lean-driver configuration failure is retained;
the corrected full semantic compiler build passes the complete route.

Remaining: General paged layouts, asynchronous/resident movement, retained
production-route performance admission and retirement. Apple, NVIDIA and x86
physical changes are not applicable to this HIP-only host service. Broader
frontend/AD, W1.1 and full five-slice closure remain open.

Evidence: benchmarks/baselines/rocm_native_movement_20261005/README.md,
contracts.txt, shared.txt, gfx1151/gfx1151.json, gfx1201/gfx1201.json,
architecture-specific build/device logs and artifacts/.
<!-- entry-fields:end -->

### 2026-10-05 — canonical native ROCm movement spine

Owner: [E2E-REAL-6](INTEGRATED_COMPILER_PLAN.md#e2e-real-6)
PRs: pending; sync ROCM-MOVEMENT-SPINE-2026-10-05.

Outcome: The compiler driver retains the native movement Schedule and binds
packaging to the original typed Graph/target. Adjacent Graph/Schedule/Tile/
Target/backend hashes prove the complete package spine. Canonical gate names
resolve dotted caches through the op catalog. Exact capabilities, numerical
fixtures and descriptor execution rows agree. gfx1201 bounded paged reads
default to native packaging; explicit opt-out remains respected.
Nine exact-device canonical rows pass independent bitwise oracles and the
existing retained-route 10% host non-regression gate. Initial registry and
generated/lifecycle failures are retained with their corrected evidence.

Remaining: Public Python tensor-call shape/eager semantics, general layouts,
asynchronous/resident movement and resident-kernel admission. Historical
DispatchPlan transports/general retained helpers remain separate. Apple,
NVIDIA and x86 have shared canonical-name parity tests and no physical HIP
movement change. Full five-slice compiler closure remains open.

Evidence: benchmarks/baselines/rocm_movement_admission_20261005/README.md,
gfx1151/gfx1151.json, gfx1201/gfx1201.json, architecture-specific spine logs,
registry.txt, generated.txt and artifacts/.
<!-- entry-fields:end -->

### 2026-10-05 — public ROCm movement frontend and native JIT

Owner: [E2E-REAL-6](INTEGRATED_COMPILER_PLAN.md#e2e-real-6)
PRs: pending; sync ROCM-PUBLIC-MOVEMENT-2026-10-05.

Outcome: Physical-page tensor reads and token-gather MoE use concrete public
reference and catalog shape contracts. Tracing preserves i32 indices and static
cache bounds without eager arithmetic. Native Schedule accepts reordered entry
arguments while retaining distinct semantic roles. Ordinary ROCm JIT binds
canonical compiler descriptors and reuses runtime artifacts by specialization.
Owning gfx1151/gfx1201 tests prove bit-exact outputs, repeated/reordered inputs,
nonfinite bit patterns and adjacent native artifact chains. Host registry and
native compiler tests cover sibling shared-contract parity.

Remaining: gfx1151 public paged-call overhead misses the retained-route 10%
gate; prebound descriptor gains cannot stand in for full JIT admission.
General layouts, asynchronous/resident APIs, kernel-only timing and transport
retirement remain open. Apple/x86 physical package admission and SM120 ordinary
movement JIT/exact-device proof remain follow-up required. Full five-slice
compiler closure remains open.

Evidence: benchmarks/baselines/rocm_public_movement_20261005/README.md,
gfx1151/gfx1151.json, gfx1201/gfx1201.json, architecture-specific jit logs,
registry.txt, shared.txt and artifacts/.
<!-- entry-fields:end -->

### 2026-10-05 — prepared native ROCm movement calls

Owner: [E2E-REAL-6](INTEGRATED_COMPILER_PLAN.md#e2e-real-6)
PRs: pending; sync ROCM-PREPARED-MOVEMENT-2026-10-05.

Outcome: Prepared C++ host calls copy the verified compiler image/static ABI.
Warm ordinary JIT performs no Graph serialization and retains checked metadata,
index bounds, context ownership and completion. Copied image/shape and close
during invocation are proved. All ten owning gfx1151/gfx1201 rows retain their
native image digests, pass bitwise oracles and satisfy retained-route 10% host
non-regression. Native counters prove zero warm staging allocations.
MoE reference JVP/VJP and opaque DispatchPlan tape replay now implement actual
gather/scatter-add, with finite-difference, adjoint and public eager AD proof.
No new GPU constructor or compiler bypass is introduced.

Remaining: General layouts, asynchronous/resident movement, native movement AD,
kernel-only timing and distributed transport retirement. HIP preparation is not
applicable to Apple/x86/CUDA physical execution; shared reference AD/Tape parity
is host-validated. Native persistent compiler sessions and the full five-slice
compiler objective remain open.

Evidence: benchmarks/baselines/rocm_prepared_movement_20261005/README.md,
gfx1151/gfx1151.json, gfx1201/gfx1201.json, architecture-specific build/JIT logs,
contracts.txt, reference-ad.txt, laws.txt, image-parity.txt and artifacts/.
<!-- entry-fields:end -->

### 2026-10-06 — native compact requested attention gradients

Owner: [AD-RESIDUAL-EVAL-1](INTEGRATED_COMPILER_PLAN.md#ad-residual-eval-1)
PRs: pending; sync NVIDIA-COMPACT-GRADIENTS-2026-10-06.

Outcome: Native paired AD keeps complete logical results while Schedule seals
requested physical outputs, launch layout and 64/128-thread geometry. Verified
Tile and NVIDIA Target lowering omit inactive storage. Checked host/resident
C++ launchers execute the compact ABI without a Python GPU body constructor.
Forty RTX 5070 numerical cases across five arms pass private capture, actual
caller mutation, repeated cotangents and result-order checks. Long-row V-only
gradient storage drops 7,736 to 3,096 bytes. Preserved logical launch ranges
recover the named packed K-only submission gap; no universal/default schedule
promotion follows. Event windows include submission gaps and are separate from
checked allocating backward wall time. Matching gfx1201 shared compiler tests
and existing norm/epilogue regressions pass.

Remaining: General composed/dynamic/higher AD, asynchronous ownership, sibling
native compact output consumers, kernel-isolated attribution and independent
FP8/MXFP8/MXFP4 gates. Apple/x86 shared registry parity is host-tested; CUDA
physical ABI/timing is not applicable. Full five-slice closure remains open.

Evidence: benchmarks/baselines/nvidia_compact_gradients_20261005/README.md,
packet.json, comparison.json, artifacts/, build-threads.txt, contracts-final.txt,
device-threads-final.txt and .validation-compact-threads/.
<!-- entry-fields:end -->

### 2026-10-06 — native attention JVP frontend argument order

Owner: [AD-RESIDUAL-EVAL-1](INTEGRATED_COMPILER_PLAN.md#ad-residual-eval-1)
PRs: pending; sync NVIDIA-JVP-ARGUMENT-ORDER-2026-10-06.

Outcome: Native forward export verifies distinct direct frontend Q/K/V roles.
Paired AD/Schedule/Tile retain native activity and argument mapping. Automatic
capture and tangent submission project frontend indices into physical roles
and check activity agreement. Seventy-two owning RTX 5070 finite-difference
cases cover all six input orders, short noncausal/long causal profiles and six
wrt orders, with actual caller mutation, scaled repeated directions and retained
outputs. Native image bytes match across every semantic permutation group.
Event/checked wall samples are separate characterization. Matching gfx1201
shared compiler tests and existing norm/epilogue device regressions pass.

Remaining: General composed/dynamic/bias/dropout/higher attention AD, aliases
and architecture-owned physical tangent consumers. Apple/x86 shared contracts
are host-tested; SM120 physical execution is not applicable. Full five-slice
closure and independent quantized-format programs remain open.

Evidence: benchmarks/baselines/nvidia_jvp_argument_order_20261006/README.md,
packet.json, image-parity.json, artifacts/, contracts.txt, metadata.txt,
device.txt, build.txt and .validation-jvp-order/.
<!-- entry-fields:end -->

### 2026-10-06 — portable native attention JVP program

Owner: [AD-RESIDUAL-EVAL-1](INTEGRATED_COMPILER_PLAN.md#ad-residual-eval-1)
PRs: pending; sync NVIDIA-JVP-PORTABLE-2026-10-06.

Outcome: Canonical JSON pins the native checkpoint pair, tangent image/sizer,
roles and manifests. Before CUDA allocation, capture validates image/descriptor
integrity, frontend mapping, physical activity, shared generation and complete
tensor/scalar/grid/scratch ABI. Seventy-two restored RTX 5070 numerical cases
and three fresh-process replays with all compiler subprocesses forbidden pass.
No Graph reconstruction, Python GPU body or physical ABI change is introduced.
Matching gfx1201 host tests prove shared contracts, without HIP replay claims.

Remaining: Source audit confirms flash attention has no generic native-JVP
plugin owner; explicit compile/frame proof does not close ordinary dispatch.
Canonical public native-JVP family/runtime integration, retirement
of unused reverse compilation/loading for forward products, native prepared
validation/launch owners, general composed/dynamic/bias/higher AD, asynchronous
ownership and sibling physical consumers. Apple/x86 shared contracts are
host-tested; CUDA physical replay is not applicable. Full five-slice closure
and independent quantized-format programs remain open.

Evidence: benchmarks/baselines/nvidia_jvp_portable_20261006/README.md,
packet.json, artifacts/, *-replay.json, *-replay.txt, contracts.txt and
.validation-jvp-portable/shared.txt.
Keyword-binding follow-on: pinned frontend parameter names survive serialization; positional, keyword and mixed calls bind before physical role projection. All 72 restored RTX 5070 cases and three compiler-forbidden fresh-process replays pass. Current focused keyword/native-JVP/diagnostic/pass gate: 339 tests; audit gate: 11 tests. Parameter-name shadowing found by replay is fixed. Public native-JVP dispatch and unused reverse-image retirement remain open.

<!-- entry-fields:end -->

### 2026-10-06 — Public native attention JVP

Owner: [AD-RESIDUAL-EVAL-1](INTEGRATED_COMPILER_PLAN.md#ad-residual-eval-1)
PRs: pending; sync NVIDIA-PUBLIC-ATTENTION-JVP-2026-10-06.

Outcome: Ordinary native_jvp now uses the canonical family planner and common
runtime execution matrix for a pinned native saved-LSE attention product.
The specialized tracer Graph, native paired AD and existing Schedule/Tile/
NVVM/LLVM images remain authoritative. Host storage/activity and private
generation ownership are checked before CUDA allocation. The public sweep
found and fixed AST cache collisions between different captured causal policies;
captured literal environments now participate in cache identity and differential
parity is retained. Target declarations name actual implemented consumers.
Seventy-two public RTX5070 oracle cases, four device tests, three externally
pinned compiler-free runtime replays and 444 host tests pass. Warm wall median
14.6047ms is characterization, not isolated kernel time or a speedup claim.

Remaining: General composed/dynamic/alias/bias/dropout/higher attention AD,
generic VJP dispatch, unused reverse image compilation/loading retirement and
native prepared validation/launch ownership. Apple/x86/ROCm shared contracts
are host-tested; CUDA images are not applicable physical sibling evidence.
Their architecture-owned tangent consumers remain follow-up required. Full
five-slice closure and independent quantized-format programs remain open.

Evidence: benchmarks/baselines/nvidia_public_attention_jvp_20261006/README.md,
packet.json, artifacts/, *-replay.json, contracts.txt and device-tests.txt.
<!-- entry-fields:end -->

### 2026-10-06 — Native prepared attention JVP

Owner: [AD-RESIDUAL-EVAL-1](INTEGRATED_COMPILER_PLAN.md#ad-residual-eval-1)
PRs: pending; sync NVIDIA-PREPARED-ATTENTION-JVP-2026-10-06.

Outcome: Four private native C ABI exports retain forward/tangent modules,
compiler sizing, aligned storage, frontend/tangent binding and event timing.
The verified Graph/paired AD/Schedule/Tile/NVVM/LLVM images are unchanged.
PID-before-lock, unique context identity, extent/closed-handle and thread-object
registration checks guard synchronous ownership. Seventy-two matched RTX5070
A/B cases, 72 public cases, 12 device tests, 443 host tests and three externally
pinned compiler-free fresh-process replays pass. Matched wall ratio 0.116622
measures adapter overhead; separate CUDA event windows remain separate.

Remaining: Unused reverse compilation/serialization, generic native VJP,
composed/dynamic/bias/dropout/higher AD and asynchronous ownership remain open.
Apple/ROCm/x86 shared host contracts are validated; this CUDA owner is not
applicable physical sibling evidence. Their native tangent consumers require
follow-up. Full five-slice closure remains open.

Evidence: benchmarks/baselines/nvidia_prepared_attention_jvp_20261006/README.md,
packet.json, public_packet.json, artifacts/, *-replay.json, contracts.txt,
device-tests.txt.
<!-- entry-fields:end -->

### 2026-10-06 — Forward-only native attention JVP

Owner: [AD-RESIDUAL-EVAL-1](INTEGRATED_COMPILER_PLAN.md#ad-residual-eval-1)
PRs: pending; sync NVIDIA-FORWARD-ATTENTION-JVP-2026-10-06.

Outcome: Public native JVP packages a compiler-owned forward checkpoint and
tangent without compiling/serializing the unused reverse executable. Distinct
forward products use pinned v2 serialization; historical paired v1 and VJP
consumers remain intact. Native forward descriptors own frontend mapping;
Graph/AD/Schedule/Tile/NVVM/LLVM remains the image authority. Forward-only
resident frames refuse backward access before buffers. Seventy-two public
RTX5070 oracle cases, four matched compile cases, six compiler-free fresh-process
replays, seven device tests including compact backward regression and 532 host
tests pass. Matched median compile ratio 0.828158; native images are equal.

Remaining: Native reverse intermediate construction, generic public VJP,
composed/dynamic/bias/dropout/higher AD and asynchronous ownership remain open.
Apple/ROCm/x86 shared host contracts are validated; CUDA images are not applicable
physical sibling evidence. Architecture-owned native consumers require follow-up.
Full five-slice closure remains open.

Evidence: benchmarks/baselines/nvidia_forward_attention_jvp_20261006/README.md,
compile-packet.json, public-packet.json, artifacts/, *-replay.json,
*-capture-replay.json, contracts.txt, reverse-guards.txt, device-tests.txt.
<!-- entry-fields:end -->

### 2026-10-06 — Captured native ROCm movement measurements

Owner: [E2E-REAL-6](INTEGRATED_COMPILER_PLAN.md#e2e-real-6)
PRs: pending; sync ROCM-CAPTURED-MOVEMENT-2026-10-06.

Outcome: Resident HIP capture uses exact native JIT descriptor symbols with
verified Graph/Schedule/Tile/Target/LLVM lineage. Five gfx1151 and three gfx1201
cases prove bitwise nonfinite movement, changed resident source/index contents
and restoration. Nine alternating 256-node trials separate event windows from
submission/completion wall samples. Small-case capture gains differ by target;
large full reads are unchanged within 1%. The named full read has more logical
than physical pages. Shared WSL gates pass 30 tests with 13 hardware skips.

Remaining: Native persistent/resident movement packaging, general layouts,
asynchronous lifetime/stream contracts, movement AD, distributed transport and
broader performance obligations remain open. Benchmark capture is not ordinary
runtime admission. Apple/NVIDIA/x86 HIP capture is not applicable physical proof.
Quantized-format gates and full five-slice closure remain independent and open.

Evidence: benchmarks/baselines/rocm_captured_movement_20261006/README.md,
gfx1151/packet.json, gfx1201/packet.json, adjacent IR/HSACOs and contracts.txt.
<!-- entry-fields:end -->

### 2026-10-06 — Native resident ROCm movement owner

Owner: [E2E-REAL-6](INTEGRATED_COMPILER_PLAN.md#e2e-real-6)
PRs: pending; sync ROCM-RESIDENT-MOVEMENT-OWNER-2026-10-06.

Outcome: Five registered C ABI exports move static movement storage, image lease,
argument binding, owned stream/events, completion and generation lifetime into
C++. Public JIT preparation and checked common launch retain native MLIR/LLVM
lineage. Five gfx1151 and three gfx1201 cases pass bit-exact nonfinite movement,
rebind/retained host outputs, bounds and stale-generation checks. Nine alternating
warm wall trials forbid compiler subprocesses and image reloads. Common resident
download/public host ratios are 0.4423–0.7905, excluding repeat upload. GPU bodies
are unchanged. 394 WSL tests pass with 13 hardware/environment skips.

Remaining: General resident producer/consumer edges, layouts/dynamic shapes,
asynchronous ownership, movement AD, distributed transport and full five-slice
closure remain open. HIP graph capture remains benchmark-only. Sibling host
contracts are validated; ROCm images are not Apple/NVIDIA/x86 physical evidence.

Evidence: benchmarks/baselines/rocm_resident_owner_20261006/README.md,
gfx1151/packet.json, gfx1201/packet.json, adjacent IR/HSACOs and contracts.txt.
<!-- entry-fields:end -->

### 2026-10-06 — Native paged read softmax edge

Owner: [E2E-REAL-6](INTEGRATED_COMPILER_PLAN.md#e2e-real-6)
PRs: pending; sync ROCM-PAGED-SOFTMAX-EDGE-2026-10-06.

Outcome: Canonical static f32 softmax JIT executes on gfx1151/gfx1201. Two
typed frontend packages bind through C++ owned intermediate/final allocations,
two image leases and one completion; two new runtime exports are registered.
Six public unary, six edge and eight movement regression rows pass. Maximum
float64 oracle error is below 4e-8. Nine alternating common resident download
wall ratios are 0.2834–0.3698, excluding repeat upload. GPU bodies are unchanged.
512 shared WSL tests pass with 13 skips; explicit per-op dtype boundaries are
checked rather than inferred from target-wide BF16 storage.

Remaining: Generic composed Graph partitioning/bufferization/lifetime planning,
dynamic/layout/async edges, borrowed tensors, softmax AD and remaining ROCm math
consumers stay open. The full five-slice objective remains open. Apple/NVIDIA/x86
shared contracts are validated; HIP image/measurement proof is not applicable.

Evidence: benchmarks/baselines/rocm_paged_softmax_edge_20261006/README.md,
gfx1151/packet.json, gfx1201/packet.json, both adjacent IR chains/HSACOs,
movement-regression/ and contracts.txt.
<!-- entry-fields:end -->

### 2026-10-06 — Public native attention VJP

Owner: [AD-RESIDUAL-EVAL-1](INTEGRATED_COMPILER_PLAN.md#ad-residual-eval-1)
PRs: pending; sync NVIDIA-PUBLIC-ATTENTION-VJP-2026-10-06.

Outcome: Generic native_backward selects the canonical SM120 saved-LSE family.
Certified tracer Graph passes through native paired AD, Schedule, Tile and
Target packages. Portable product identity and private checkpoint lifetime
preserve requested physical gradients. Eighty-eight RTX 5070 FP64 oracle cases,
changed cotangents and retained prior outputs pass; three fresh-process common
runtime replays forbid compilers/processes. Maximum absolute error is 1.991e-8.
Warm host-call median is 11.3600 ms; no isolated kernel or speedup claim.
An instrumented profile identifies repeated product restoration as a measured
next native prepared-service boundary. No new dtype/op/pass/diagnostic/C ABI.

Remaining: Native prepared reverse ownership, general composed/dynamic/layout/
dropout/higher AD and sibling physical consumers remain open. All four backend
plans assess target-scoped declarations and runtime schema; SM120 evidence is
not transferred. FP8/MXFP8/MXFP4 and full five-slice closure remain independent.

Evidence: benchmarks/baselines/nvidia_public_attention_vjp_20261006/README.md,
packet.json, compiler-identity.json, artifacts/, replay-*.json and warm-profile.json.
<!-- entry-fields:end -->

### 2026-10-06 — Native prepared attention VJP

Owner: [AD-RESIDUAL-EVAL-1](INTEGRATED_COMPILER_PLAN.md#ad-residual-eval-1)
PRs: pending; sync NVIDIA-PREPARED-ATTENTION-VJP-2026-10-06.

Outcome: Four C ABI exports extend existing attention native ownership to
compact saved-LSE reverse products, requested gradients and optional bias.
A bounded planning/registration layer imports verified products once; C++
retains modules, arena, stream, events and the synchronous saved generation.
88 public and 88 matched RTX5070 cases pass independent FP64 gradients; all
88 native program pins are unchanged. Nine alternating common-runtime trials
record a median prepared/unprepared wall ratio of 0.0700637, including transfers.
CUDA forward/backward windows are separate. Three compiler-free fresh-process
replays and extent/closed/context/PID/lock guards pass. JVP regressions pass.
Runtime build context is LLVM/MLIR 23.1.1 and CUDA SDK 13.4.59.

Remaining: General Graph composition/bufferization, dynamic/layout/resident/
async/dropout/higher AD and sibling physical consumers remain open. All four
backend plans assess the shared ABI; no GPU schedule/default promotion or
sibling device proof follows. Full five-slice closure remains open.

Evidence: benchmarks/baselines/nvidia_prepared_attention_vjp_20261006/README.md,
public.json, matched.json, toolchain.json, artifacts/, replay-*.json and device-tests.txt.
<!-- entry-fields:end -->

### 2026-10-06 — Native score-bias attention JVP

Owner: [AD-RESIDUAL-EVAL-1](INTEGRATED_COMPILER_PLAN.md#ad-residual-eval-1)
PRs: pending; sync NVIDIA-BIAS-JVP-2026-10-06.

Outcome: Native Graph/TangentInterface export and verified Schedule/Tile carry
rank-four full/broadcast bias and explicit dbias through LLVM/NVVM. Complete
forward-generation identity includes physical bias/reduction policy. Portable
programs check roles/manifests/pins. One declared C ABI export extends existing
native prepared ownership; old unbiased products/exports are retained.
36 resident and 36 ordinary public/matched RTX5070 cases pass independent
fp64 analytic JVP/central differences, mutation and repeated-direction checks.
Maximum error 3.970e-8. Median per-case prepared/replay host ratio 0.134276
includes transfers; forward/tangent events are separate. Three compiler-free
fresh-process replays and 38 device/guard regressions pass. Runtime build:
LLVM/MLIR 23.1.1, CUDA SDK 13.4.59, RTX5070 driver 610.88.

Remaining: General composed/dynamic/layout/aliasing/resident/async/dropout/
higher AD and wider dtypes remain open. All four queues assess shared contracts.
Sibling physical consumers need their own exact-device proof. FP8/MXFP8/MXFP4
and full five-slice closure remain independent obligations.

Evidence: benchmarks/baselines/nvidia_bias_jvp_20261006/README.md,
packet.json, public-packet.json, artifacts/, public-artifacts/, replay-*.json,
host-tests.txt and device-tests.txt.
<!-- entry-fields:end -->


### 2026-10-06 — explicit native MXFP8 K64 staging candidate

Owner: [ROCM-FP8-BLOCKSCALE-1](INTEGRATED_COMPILER_PLAN.md#rocm-fp8-blockscale-1)
Related owners: ROCM-FP8-BLOCKSCALE-1
PRs: pending; sync ROCM-MXFP8-K64-2026-10-06.

Outcome: MLIR Schedule admits an explicit K64 LDS recipe with separate semantic
K32 scale joins. Native generator stages once per slab; runtime checks whole
slabs and physical recipe metadata. All 39 initial focused checks and 354
regressions pass. A scratch-spilling 128x128 candidate was rejected; the narrower
128x64 candidate passed matched FP8/MXFP8/MXFP4 numerical and timing checks.
Automatic selection remains unchanged. All four backend plans are assessed.

Remaining: Independent repetitions and wider envelopes precede selector changes.
General persistence/prologue/epilogue and broader format/performance closure
remain open; no sibling schedule or exact-device proof transfers.

Evidence: benchmarks/baselines/rocm_mxfp8_k64_20261006/README.md
<!-- entry-fields:end -->


### 2026-10-06 — bounded gfx1201 MXFP8 K64 selector

Owner: [ROCM-FP8-BLOCKSCALE-1](INTEGRATED_COMPILER_PLAN.md#rocm-fp8-blockscale-1)
Related owners: ROCM-FP8-BLOCKSCALE-1
PRs: pending; sync ROCM-MXFP8-K64-2026-10-06.

Outcome: Two independent/reversed seven-shape six-arm processes retain native
K64 staging only for the pre-existing 128x64 LDS envelope at whole-slab K1024–
2048. Measured reductions are 10–21%; 128x128 controls lose and long K ties,
so both retain K32. 125 exact-device/contract checks pass, including intermediate
K, special scales, ragged rows and checked image reuse. All four queues assessed.
Python carries no kernel construction or automatic physical selector.

Remaining: Persistence/deep buffering, general cache/layout and M256 MXFP4
attribution remain open. This is not a global FP8/MXFP8/MXFP4 format decision.

Evidence: benchmarks/baselines/rocm_mxfp8_k64_20261006/README.md
<!-- entry-fields:end -->


### 2026-10-06 — SM120 canonical input lineage integration

Owner: [W1.1](INTEGRATED_COMPILER_PLAN.md#w11)
Related owners: W1.1
PRs: pending; sync NVIDIA-W11-OPERAND-LINEAGE-2026-10-06.

Outcome: Native tensor recovery follows actual entry roles through verified
canonical storage wrappers and retains full tiling replay equivalence. Fused
bias/residual roles are independent of frontend signature order. NVIDIA
package validation reads retained Graph roles and checks native canonical ABI.
12 FP16/BF16 plain/fused RTX5070 benchmark rows prove numerical agreement,
full registered-pipeline PTX parity, resident and host timings. Maximum error
1.55e-6; 373 producer/diagnostic/pass-metadata regressions pass. All four queues
assessed. No Python kernel construction or recovery Graph is introduced.

Remaining: General composed producers/attention/AD, dynamic reconstruction,
noncanonical accumulators and wider dtype/layout envelopes remain open.

Evidence: benchmarks/baselines/nvidia_operand_lineage_20261006/README.md
<!-- entry-fields:end -->

### 2026-10-06 — SM120 ordinary matmul canonical package execution

Owner: [W1.1](INTEGRATED_COMPILER_PLAN.md#w11)
Related owners: W1.1 / FRONTEND-IR-MEDIUM-1
PRs: pending; sync NVIDIA-W11-PUBLIC-PACKAGE-2026-10-06.

Outcome: Static public host-array matmul executes the checked canonical native
Graph/Schedule/Tile package, including permuted arguments, compact C/F RHS,
FP16/BF16 storage, fused bias/ReLU/residual and final f16/f32 output. Native
Schedule, typed view/fragment lowering and explicit row-RHS ABI variants carry
physical semantics. Twelve RTX5070 tests prove numerical and serialized parity,
cached no-eager/no-recompile execution and changed-value rebinding. Focused
native/JIT/registry regression: 591 passed, 53 explicitly skipped.
All four backend queues assessed.

Remaining: Python tracing/allocation/launch overhead still needs a prepared
native public-call contract. General composed AD/attention and dynamic matmul
are outside this static envelope; no universal integration or speedup claim.

Evidence: benchmarks/baselines/nvidia_operand_lineage_20261006/README.md
<!-- entry-fields:end -->

### 2026-10-06 — SM120 native prepared matmul ownership

Owner: [W1.1](INTEGRATED_COMPILER_PLAN.md#w11)
Related owners: W1.1 / FRONTEND-IR-MEDIUM-1
PRs: pending; sync NVIDIA-PREPARED-MATMUL-2026-10-06.

Outcome: Checked static FP16/BF16 Schedule/Tile/LLVM images now have native
module/context/stream ownership and context-shared synchronous scratch.
Warm public calls preserve sealed frontend signatures without trace,
compile, eager evaluation or portable descriptor restoration. Native host-view
guards cover physical storage, singleton pitch, lifetime and changed values.
75 RTX5070 device/guard tests and 651 regressions pass (66 skipped).
Four independent matched A/B packets retain identical images and separate
event/wall scopes. Median per-case warm public wall ratios are
0.405027/0.395491.
All four backend plans assessed; no kernel/format strategy promotion.

Remaining: General A layouts, composed producer/AD, dynamic/resident/async
ownership and sibling physical consumers remain open. The full five-slice
goal remains active.

Evidence: benchmarks/baselines/nvidia_prepared_matmul_20261006/README.md
<!-- entry-fields:end -->

### 2026-10-06 — SM120 producer edge RHS layout integration

Owner: [W1.1](INTEGRATED_COMPILER_PLAN.md#w11)
Related owners: W1.1 / FRONTEND-IR-MEDIUM-1
PRs: pending; sync NVIDIA-LHS-RHS-LAYOUT-2026-10-06.

Outcome: Ordinary named LHS RMSNorm/LayerNorm/softmax matmul records compact
C/F RHS storage facts on a copied Graph and retains native Schedule/Tile
row/column recipes through the resident edge, cache and portable replay.
70 RTX5070 device tests prove numerical and lifecycle/layout guards,
including four fresh compiler-disabled portable replays. 676 focused host
regressions pass (66 skipped). Four independent 24-row timing packets retain
matched inputs, device/compiler identity and source hashes.
All four backend plans assessed; sibling physical parity remains follow-up.

Remaining: General producer composition, dynamic row-RHS, A layouts,
composed AD and native resident/asynchronous ownership remain open.
The full five-slice goal remains active.

Evidence: benchmarks/baselines/nvidia_lhs_rhs_layout_20261006/README.md
<!-- entry-fields:end -->

### 2026-10-06 — SM120 native producer-to-matmul ownership

Owner: [W1.1](INTEGRATED_COMPILER_PLAN.md#w11)
Related owners: W1.1 / FRONTEND-IR-MEDIUM-1
PRs: pending; sync NVIDIA-PREPARED-LHS-2026-10-06.

Outcome: Verified static FP16/BF16 norm/softmax and matmul packages now have
native module, context, stream, pinned staging and private intermediate
ownership. All transfers/kernels share one stream; submission failures retire
before scratch reuse. A benchmark-discovered large first-frame ordering gap
is corrected and covered by changed-value and standalone-matmul checks.
921 focused checks pass (67 skipped), including 182 RTX5070 device cases.
Eight identical-image A/B packets retain separate event/wall timing and
show about 89% lower warm public-call overhead. All four plans assessed.

Remaining: General producer composition, AD, dynamic layouts, asynchronous/
resident and common portable native ownership, wider formats and sibling
physical consumers remain open. The full five-slice goal remains active.

Evidence: benchmarks/baselines/nvidia_prepared_lhs_20261006/README.md
<!-- entry-fields:end -->

### 2026-10-06 — SM120 portable tensor replay ownership

Owner: [W1.1](INTEGRATED_COMPILER_PLAN.md#w11)
Related owners: [W1.1](INTEGRATED_COMPILER_PLAN.md#w11); FRONTEND-IR-MEDIUM-1.
PRs: pending; sync NVIDIA-PORTABLE-LHS-2026-10-06.

Outcome: named static FP16/BF16 RMSNorm/LayerNorm/softmax -> matmul portable
replay retains admitted type-strict manifests and bounded per-context native
two-module owners. Parent Graph/target/input ABI guards precede context access.
One cache lock leases the complete synchronous frame through native completion;
eviction/clear retire owners, and fork refusal precedes locks/CUDA.
498 focused checks pass, including 142 RTX5070 device cases. Matched replay
wall A/B records and independent producer/consumer dispatch windows preserve
physical image, ABI, source/compiler/runtime identities. Median per-case warm
portable wall ratios are 0.163194–0.168429 (approximately 83–84% lower replay
overhead); this is not kernel time. All four plans assessed.

Remaining: general composed producer/AD, dynamic layouts, asynchronous/resident
ownership, wider formats and sibling exact-device consumers. Full five-slice
closure remains open; no physical kernel or format promotion.

Evidence: benchmarks/baselines/nvidia_portable_lhs_owner_20261006/README.md.
<!-- entry-fields:end -->

### 2026-10-06 — bounded SM120 frontend tensor programs

Owner: [W1.1](INTEGRATED_COMPILER_PLAN.md#w11)
Related owners: [W1.1](INTEGRATED_COMPILER_PLAN.md#w11); FRONTEND-IR-MEDIUM-1.
PRs: pending; sync NVIDIA-DYNAMIC-LHS-2026-10-06.

Outcome: explicit frontend compilation connects all seven bounded M/N/K axis
subsets to the existing native producer/strided-matmul contract. Full retained
Graph types and shape bounds verify; semantic epilogue role markers project
to actual argument names. Native Schedule permits explicit column-major RHS
for dynamic SM120 storage. Independent numerical, unchanged-caller, portable,
permuted/fresh-process and pre-allocation guard evidence is linked below.
766 focused checks pass (49 skipped), including 202 device cases; positive/
negative native IR FileCheck fixtures pass.
Compile, replay wall and producer/consumer event windows remain separate.
All four plans assessed; no sibling physical or performance proof transfers.

Remaining: ordinary JIT bounded-shape selection, dynamic row-RHS/native
prepared owners, general composition/AD, asynchronous ownership and wider
formats. Full five-slice closure remains open.

Evidence: benchmarks/baselines/nvidia_dynamic_lhs_frontend_20261006/README.md.
<!-- entry-fields:end -->

### 2026-10-06 — native bounded SM120 tensor ownership

Owner: [W1.1](INTEGRATED_COMPILER_PLAN.md#w11)
Related owners: [W1.1](INTEGRATED_COMPILER_PLAN.md#w11); FRONTEND-IR-MEDIUM-1.
PRs: pending; sync NVIDIA-DYNAMIC-LHS-OWNER-2026-10-06.

Outcome: native two-module owners now retain immutable exact/bounded M/N/K
capacities, reflect the dynamic pointer/scalar parameter ABI and derive checked
active frames/leading dimensions per invocation. Typed portable admission reuses
one context-specific handle across shapes. Shared pinned/device scratch is leased
through completion; failed consumers retire real queued producers without writing
caller output. 688 focused checks pass, including 332 RTX5070 device cases.
Matched identical-image warm wall and separate producer/consumer dispatch
packets are linked below: median per-case prepared/control warm wall ratios
are 0.120738/0.120296, approximately 88% lower host-call overhead; no kernel
speedup is claimed. All four sibling plans assessed.

Remaining: ordinary JIT bounded-shape selection, dynamic row-RHS, general
composition/AD, asynchronous/resident owners and wider formats. Full five-slice
closure remains open; no physical kernel or format promotion.

Evidence: benchmarks/baselines/nvidia_dynamic_lhs_owner_20261006/README.md.
<!-- entry-fields:end -->

### 2026-10-06 — ordinary bounded SM120 tensor JIT

Owner: [W1.1](INTEGRATED_COMPILER_PLAN.md#w11)
Related owners: [W1.1](INTEGRATED_COMPILER_PLAN.md#w11) / FRONTEND-IR-MEDIUM-1.
PRs: pending; sync NVIDIA-BOUNDED-LHS-JIT-2026-10-06.

Outcome: Immutable public bounds and source/live-code certification connect
ordinary straight-line norm/softmax -> matmul calls to verified bounded
Graph/Schedule/Tile packages and native dynamic ownership. Original Graph
contracts are verified before projection. 547 checks pass, including 211
RTX5070 device cases. Four matched 48-case packets retain identical package
identities; median warm ordinary wall ratios are 0.0765821/0.0761356.
Cold compile and separate component event timing remain recorded.
All four backend queues assessed; no physical parity transfer.

Remaining: General composition/control flow/bufferization/AD, dynamic row-RHS,
resident/asynchronous ownership, wider formats and full five-slice closure.

Evidence: benchmarks/baselines/nvidia_bounded_lhs_jit_20261006/README.md,
control.json, prepared.json, prepared_reverse.json, control_reverse.json,
analysis.json and regression-tests.txt.
<!-- entry-fields:end -->

### 2026-10-06 — bounded SM120 row-major RHS integration

Owner: [W1.1](INTEGRATED_COMPILER_PLAN.md#w11)
Related owners: [W1.1](INTEGRATED_COMPILER_PLAN.md#w11) / FRONTEND-IR-MEDIUM-1.
PRs: pending; sync NVIDIA-DYNAMIC-ROW-RHS-2026-10-06.

Outcome: Native Schedule dynamic composed indices and tile.view pitches
feed typed B fragment gathers. Explicit row strided ABIs, native span/pitch
binding and symbol/storage checks carry the compiled images to execution.
Public bounded storage requests preserve authored Graph facts and keep the
column-major default. Resident CAI pitches are checked before launch.
781 checks pass (406 RTX5070 cases, four unrelated environment skips);
28 frontend contract checks pass. Eight 48-case matched packets pass.
Row/column median ordinary wall ratios are 1.06658/1.03469; consumer event
ratios 1.09243/1.14834. This is layout coverage; row gather tuning remains.
All four backend queues assessed with no physical parity transfer.

Remaining: General composition/bufferization/control-flow/AD, resident/async
ownership, wider formats and full five-slice closure. No strategy promotion.

Evidence: benchmarks/baselines/nvidia_dynamic_row_lhs_20261006/README.md,
analysis.json, eight matched packets, regression-tests.txt,
frontend-contract-tests.txt and build.txt.
<!-- entry-fields:end -->
### 2026-10-06 — packed gfx1201 image identity and runtime proof

Owner: [ROCM-NVFP4-INGEST-1](INTEGRATED_COMPILER_PLAN.md#rocm-nvfp4-ingest-1)
Related owners: ROCM-NVFP4-INGEST-1 / ROCM-MXFP4-W4A8-1.
PRs: pending; sync ROCM-PACKED-IMAGE-IDENTITY-2026-10-06.

Outcome: Native packed Target projection admits runtime M/N with fixed K and
whole/partial tile classes. Optional packaging retains authored and compiled
Target digests separately; checked HIP admission rejects mismatched classes,
K, policies and geometry. 37 exact gfx1201 compiler/device/resident regressions
pass. Two static shape pairs reuse native image bytes; six matched benchmark
cases pass independent bitwise BF16 oracles before and after timing.

Follow-through: 52 compiler/device/frontend/resident regressions pass.
The three-stage resident and portable program retain static authored Target
ancestry while image identity binds projected Target. Compiler-free fresh
replay and static-image compatibility pass. Three ordinary-JIT/resident
benchmark shapes retain independent conversion/storage/output checks and
separate converter/storage/consumer/combined device and checked wall timings.

Remaining: General dynamic Graph/packing/layout integration, native runtime
ownership below Python and full five-slice closure. No performance promotion
or sibling physical proof.

Evidence: benchmarks/baselines/rocm_packed_image_identity_20261006/.
<!-- entry-fields:end -->





### 2026-10-06 — native gfx1201 NVFP4 program ownership

Owner: [ROCM-NVFP4-INGEST-1](INTEGRATED_COMPILER_PLAN.md#rocm-nvfp4-ingest-1)
Related owners: ROCM-NVFP4-INGEST-1 / E2E-REAL-6 / FRONTEND-IR-MEDIUM-1.
PRs: pending; sync ROCM-NVFP4-NATIVE-OWNER-2026-10-06.

Outcome: Five C ABI exports move private stream, 11 allocations, three image
leases, snapshots, argument arrays, readiness/generation, repeated launches,
events/readback and close below Python. Existing MLIR/LLVM images are unchanged.
47 native/device/frontend/resident/failure tests pass; 21 ABI gates pass with
one environment skip. Exact-device three-case matched packet retains independent
conversion, lossless storage and folded-output correctness. Allocating checked
walls characterize 0.507-0.566 ratios; warm reuse remains mixed.

Remaining: Ordinary native runtime/session admission, native graph replay,
warm-path attribution/retuning, general dynamic packing and full five-slice
closure. No default/GPU performance promotion or sibling hardware inference.

Evidence: benchmarks/baselines/rocm_packed_image_identity_20261006/native-owner*.
<!-- entry-fields:end -->


### 2026-10-06 — native gfx1201 NVFP4 graph and ordinary JIT integration

Owner: [ROCM-NVFP4-INGEST-1](INTEGRATED_COMPILER_PLAN.md#rocm-nvfp4-ingest-1)
Related owners: ROCM-NVFP4-INGEST-1 / E2E-REAL-6 / FRONTEND-IR-MEDIUM-1.
PRs: pending; sync ROCM-NVFP4-NATIVE-GRAPH-JIT-2026-10-06.
Outcome: Ordinary traced JIT and common-runtime portable replay execute
through the native C++ owner; fresh process replay has no compiler dependency.
The sixth graph ABI export retains bounded capture/instantiate/replay ownership
and safe failed-close retry. Exact gfx1201 integration: 58 passed; shared
ABI/audit: 32 passed, one skipped. Six JIT benchmark cases and three matched
Python/native/native-graph cases pass independent numerical checks.
Direct native submission remains ordinary; graph measurements do not justify
promotion. Allocating, warm resident and device event timings remain separate.
Remaining: General dynamic packing/layout/AD consumers, warm-call tuning and
full five-slice closure. Sibling physical parity requires exact-device evidence.
Evidence: benchmarks/baselines/rocm_packed_image_identity_20261006/README.md.
<!-- entry-fields:end -->


### 2026-10-06 — ordinary SM120 attention native frontend integration

Owner: [E2E-REAL-6](INTEGRATED_COMPILER_PLAN.md#e2e-real-6)
Related owners: E2E-REAL-6 / NVIDIA-LSE-1 / FRONTEND-IR-MEDIUM-1.
PRs: pending; sync NVIDIA-ORDINARY-ATTENTION-2026-10-06.
Outcome: Ordinary primal f32 attention selects canonical native compilation
and checked descriptor launch. Native Schedule validates external argument
roles, supporting Q/K/V and full-bias permutations; shape projection follows
retained Graph SSA. Exact RTX5070: 28 device checks and 36 independent-oracle
benchmark rows pass. Shared attention/backward gates: 53 passed, 18 skipped.
Public allocating wall and resident CUDA event windows remain distinct.
Existing saved-LSE forward/JVP/VJP contracts remain separate and unchanged.
Remaining: General composition, dynamic/wider dtype/broadcast primal bias,
higher AD, asynchronous ownership and full five-slice closure.
Evidence: benchmarks/baselines/nvidia_ordinary_attention_20261006/README.md.
<!-- entry-fields:end -->


### 2026-10-06 — direct authored SM120 saved-LSE Graph import

Owner: [E2E-REAL-6](INTEGRATED_COMPILER_PLAN.md#e2e-real-6)
Related owners: NVIDIA-LSE-1 / FRONTEND-IR-MEDIUM-1 / E2E-REAL-6.
PRs: pending; sync NVIDIA-SAVED-GRAPH-2026-10-06.
Outcome: Native MLIR import consumes original saved-LSE Graph operands,
types and validated policy. Python no longer reconstructs checkpoint Graphs.
Optional forward LSE and backward Graph ODS declarations verify shape relations;
native Schedule binds roles independently of function argument order.
Matching build and 56 checkpoint contracts pass. Exact RTX5070: 45 device
checks pass, including forward O/LSE and backward gradient independent oracles,
permuted roles, bias and lifetime. Metadata/diagnostic/dialect gates: 330 pass.
Forward/backward device medians are 26.725/80.041 us; respective complete
host-array launch walls 1.023/1.362 ms. Timing domains are distinct; no promotion.
Remaining: general composition, dynamic shapes, wider storage, broader AD,
sibling physical parity and full five-slice closure.
Evidence: benchmarks/baselines/nvidia_ordinary_attention_20261006/README.md.
<!-- entry-fields:end -->


### 2026-10-06 — saved Graph policy compatibility and verifier evidence

Owner: [E2E-REAL-6](INTEGRATED_COMPILER_PLAN.md#e2e-real-6)
Related owners: NVIDIA-LSE-1 / FRONTEND-IR-MEDIUM-1 / E2E-REAL-6.
PRs: pending; sync NVIDIA-SAVED-GRAPH-2026-10-06.
Outcome: Native checkpoint import accepts neutral integer numerical spellings.
Typed dropout serialization and recompute floating literals preserve numerical
intent. Backward ODS admission covers both saved and recompute shapes.
Qualified NVFP4/storage C++ verifiers now count as implemented in the coverage
scanner; no compiler verifier was removed. Matching build; 164 host contracts
(one skip),
27 exact RTX5070 numerical/lifetime cases, 57 verifier/dialect and 22 focused
route/benchmark/generated-coverage checks pass. Fresh seven-sample forward and
backward packet records live SM/UUID and compiler/runtime hashes.
Remaining: recompute Python Tile-constructor migration, general composition,
dynamic shapes, wider storage, sibling physical proof and full five-slice scope.
Evidence: benchmarks/baselines/nvidia_ordinary_attention_20261006/README.md.
<!-- entry-fields:end -->


### 2026-10-06 — original recompute Graph to native Schedule/Tile

Owner: [E2E-REAL-6](INTEGRATED_COMPILER_PLAN.md#e2e-real-6)
Related owners: NVIDIA-LSE-1 / FRONTEND-IR-MEDIUM-1 / E2E-REAL-6.
PRs: pending; sync NVIDIA-RECOMPUTE-GRAPH-2026-10-06.
Outcome: Canonical recompute backward retires its Python Tile construction.
Native policy hashing/replay retains SSA roles, storage, window, softcap and
seeded dropout. Matching build; 407 host/metadata checks and 54 exact RTX5070
numerical/lifetime cases pass. Device validation exposed a missing storage
field in the launch symbol; corrected native symbols preserve the existing
two/four-byte transfer ABI, with regression coverage.
Mixed legal keep/drop masks and signed-int64 seed extrema pass independent
device oracles. NVIDIA LCG offset construction preserves low-32-bit semantics
without signed overflow. The fresh packet separates forward, saved backward
and recompute device/wall timings and records current binary hashes; no promotion.
Shared audit/op/dtype gates: 41 pass; full generated-doc refresh succeeds.
Remaining: general composition, dynamic shapes, broader bias/AD, recompute
performance optimization and sibling physical proof.
Evidence: benchmarks/baselines/nvidia_ordinary_attention_20261006/README.md.
<!-- entry-fields:end -->

### 2026-10-06 — gfx1201 native NVFP4 allocation reuse

Owner: [ROCM-NVFP4-INGEST-1](INTEGRATED_COMPILER_PLAN.md#rocm-nvfp4-ingest-1)
Related owners: ROCM-NVFP4-INGEST-1 / FRONTEND-IR-MEDIUM-1 / E2E-REAL-6.
PRs: pending; sync ROCM-NVFP4-ALLOCATION-REUSE-2026-10-06.
Outcome: Native full-input rebinding and bounded idle ownership drive ordinary
JIT and portable execution below Python. Fresh handle tokens, complete-image
identity, HIP context/device checks, full uploads and derived-state invalidation
preserve lifetime and mutation semantics. Matching runtime; 35 exact-gfx1201/
ownership/fault-injection checks and 37 host ABI/frontend checks pass (22 skips).
Three-shape public cached/control wall ratios 0.546/0.577/0.552 with identical
images and independent oracles; separate HIP events remain diagnostic.
Remaining: general packing/layout/dynamic/AD, gfx1151 physical proof and full
five-slice closure. No kernel speedup or sibling parity claim.
Evidence: benchmarks/baselines/rocm_nvfp4_allocation_reuse_20261006/README.md.
<!-- entry-fields:end -->

### 2026-10-06 — checked gfx1201 static program retention

Owner: [ROCM-NVFP4-INGEST-1](INTEGRATED_COMPILER_PLAN.md#rocm-nvfp4-ingest-1)
Related owners: ROCM-NVFP4-INGEST-1 / FRONTEND-IR-MEDIUM-1 / E2E-REAL-6.
PRs: pending; sync ROCM-NVFP4-STATIC-PROGRAM-2026-10-06.
Outcome: Profiling identified repeat manifest reconstruction. A bounded exact
typed snapshot cache retains validated static products under the common runtime.
Caller mutation, parent Graph/argument identity, bool/int and list/tuple
distinctions, signed zero, LRU bounds and process guards are tested.
Owning-host suite: 47 pass (37 hardware, nine metadata and one native fault
injection). Shared diagnostic/pass/ABI/frontend gates: 332 pass.
Two fresh-process balanced packets reproduce 39–49% warm public-JIT wall
reduction against reconstruction with identical images and native allocation
reuse in both arms. Native five-input uploads/lifetime checks remain per call.
Remaining: broader composition/dynamic/layout/AD, math metadata consumers and
W8A8/MXFP4/kernel performance closure; no sibling physical parity.
Evidence: benchmarks/baselines/rocm_nvfp4_allocation_reuse_20261006/README.md.
<!-- entry-fields:end -->

### 2026-10-06 — native ROCm math Schedule foundation

Owner: [E2E-REAL-6](INTEGRATED_COMPILER_PLAN.md#e2e-real-6)
Related owners: E2E-REAL-6 / FRONTEND-IR-MEDIUM-1.
PRs: pending; sync ROCM-MATH-NATIVE-SCHEDULE-2026-10-06.
Outcome: Six f32 math operations retain original Graph SSA roles in a
native replay-checked Schedule/Tile contract; owning ROCm Target lowering
checks policy/operands/architecture and generates native executable images.
Matching builds on Princess-Luna/gfx1151 and Tajasaurus/gfx1201; 65 checks
per host, including 12 IEEE numerical cases. Each device has 24 oracle-
checked rows with separate resident HIP-event and allocating host walls.
Shared diagnostic/pass/op/dtype gates: 358 pass, 17 owning-Target skips.
Remaining: portable/JIT native ABI projection, f16/bf16-to-f32 inputs,
18 recorder metadata rows, general composition/layout/dynamic/AD.
Evidence: benchmarks/baselines/rocm_native_math_20261006/README.md.
<!-- entry-fields:end -->

### 2026-10-06 — ordinary JIT and portable native ROCm math

Owner: [E2E-REAL-6](INTEGRATED_COMPILER_PLAN.md#e2e-real-6)
Related owners: E2E-REAL-6 / FRONTEND-IR-MEDIUM-1.
PRs: pending; sync ROCM-MATH-NATIVE-PACKAGE-2026-10-06.
Outcome: Six isolated f32 original Graph operations reach native verified
Schedule/Tile, shape-independent Target images, checked packages and ordinary
JIT. Persisted native role metadata binds reversed binary operands correctly;
default-valued Target attributes follow ODS semantics. Separate capabilities,
numerical fixtures and execution rows name the proved static envelope.
48 checks pass on each exact ROCm architecture; two 24-row numerical packets
separate resident device events, warm JIT wall and portable-launch wall.
Remaining: f16/bf16 inputs with f32 outputs, physical-math metadata recorder
retirement, general composition/layout/dynamic/AD and host launch overhead.
Evidence: benchmarks/baselines/rocm_native_math_20261006/README.md.
<!-- entry-fields:end -->

### 2026-10-06 — exact ROCm widening and math recorder retirement

Owner: [E2E-REAL-6](INTEGRATED_COMPILER_PLAN.md#e2e-real-6)
Related owners: E2E-REAL-6 / FRONTEND-IR-MEDIUM-1 / EVIDENCE-PACKET-1.
PRs: pending; sync ROCM-MATH-WIDENING-2026-10-06.
Outcome: Original f16/bf16-to-f32 Graph casts plus one math consumer reach
native replay-checked Schedule/Tile and mixed-storage Target/LLVM packages.
No weakening of the Graph math result dtype rules. ROCm-owned narrow Tile
admission preserves sibling f32 gates. Optional Target output_dtype preserves
legacy same-storage semantics; new checked descriptors name six narrow image
ABIs and explicit f32 output. Positional cast dtype is traced as an attribute.
Matching builds and 212 checks pass on each ROCm chip; independent 72-row
packets record exact GPU/IR/image/compiler identity and three timing domains.
gfx1151 physical recorder: 21 native-package rows, zero metadata probes,
36 owning tests with seven x86 skips. Legacy metadata construction retired.
Remaining: general composition/broadcast/dynamic/AD, standalone narrow-output
math, native buffer reuse/host overhead and quantized performance programs.
Evidence: benchmarks/baselines/rocm_native_math_20261006/README.md.
<!-- entry-fields:end -->


### 2026-10-06 — ROCm math host-call attribution

Owner: [E2E-REAL-6](INTEGRATED_COMPILER_PLAN.md#e2e-real-6)
Related owners: E2E-REAL-6 / FRONTEND-IR-MEDIUM-1.
PRs: pending; sync ROCM-MATH-LAUNCH-ATTRIBUTION-2026-10-06.
Outcome: Both owning GPUs pass 18 changed-input numerical/call-count cases.
Instrumented allocation/free costs explain a bounded small-shape opportunity;
large-shape host copies dominate. Uninstrumented host walls are separate.
No runtime, kernel, ABI or schedule changes; no speedup claim.
Remaining: implement checked native allocation reuse and resident composed
Graph edges, with lifetime/stream guards and matched A/B timing.
Evidence: benchmarks/baselines/rocm_native_math_20261006/README.md.
<!-- entry-fields:end -->



### 2026-10-06 — native ROCm math allocation ownership

Owner: [E2E-REAL-6](INTEGRATED_COMPILER_PLAN.md#e2e-real-6)
Related owners: E2E-REAL-6 / FRONTEND-IR-MEDIUM-1.
PRs: pending; sync ROCM-MATH-NATIVE-STAGING-2026-10-06.
Outcome: Native checked capacity reuse for existing f32 and exact widening
math images. Every input is uploaded; output spans and device/context/PID
are checked. Fault/lifetime harness passes; 99 owning numerical/staging
checks per chip and nine final ownership checks per chip pass. Generated
runtime ABI inventory includes the new service entry. Shared gates 356 pass,
12 owning skips; op/dtype/movement binding gates 36 pass. Matched 18-row
three-arm packets check oracle, same images and allocation counters.
Remaining: resident multi-operation Graph edges to avoid intermediate
transfers, general layouts/dynamics/AD and wider quantized performance.
Evidence: benchmarks/baselines/rocm_native_math_20261006/README.md.
<!-- entry-fields:end -->


### 2026-10-06 — native package contract parsing and type cleanup

Owner: [W1.1](INTEGRATED_COMPILER_PLAN.md#w11)
Related owners: E2E-REAL-6 / FRONTEND-IR-MEDIUM-1 / W1.1.
PRs: pending; sync NATIVE-CONTRACT-TYPE-GATES-2026-10-06.
Outcome: NVFP4 ingest and lossless MXFP4 storage parsers reject missing native
metadata fields with a contract ValueError before descriptor construction.
Return types preserve their exact descriptor argument order. Paged-KV and
MoE lowering recheck admitted Graph contracts before unpacking. Attention
metadata validates sequence containers and positive physical bias dimensions;
variable-length role tuples retain their actual ABI shape. Numerical kernels,
image ABIs, storage policies and physical schedules are unchanged.
Validation: Super-Bear WSL focused contract suite 49 passed, 25 owning
device/compiler skips; changed-module Ruff passes. Mypy falls from 105 to
80 errors with the baseline still zero; the type gate remains failing.
Remaining: fix remaining type errors, reconcile broad unit environment and
source failures, regenerate derived evidence/docs and obtain owning ROCm
package replay checks for these parser changes before publication.
Evidence: tests/unit/test_native_packed_contract_fields.py;
tests/unit/test_rocm_mxfp4_storage_native.py;
tests/unit/test_rocm_nvfp4_ingest_package.py;
tests/unit/test_resident_attention.py.
<!-- entry-fields:end -->

### 2026-10-06 — native contract type gate restored

Owner: [W1.1](INTEGRATED_COMPILER_PLAN.md#w11)
Related owners: E2E-REAL-6 / FRONTEND-IR-MEDIUM-1 / W1.1.
PRs: pending; sync NATIVE-CONTRACT-TYPE-GATES-2026-10-06.
Outcome: Mypy validates all 625 Python source files with zero errors and the
baseline remains zero. Bounded frontend weak ownership and optional native
libraries are checked before dispatch. Cache declarations use actual native
program/owner types. Attention policy booleans and integer sequences retain
strict types; compact checkpoint options are passed through typed arguments.
Resident descriptors, ROCm folded schedules and math families validate their
metadata before use. Movement and attention contracts retain distinct tuple
types. Resident movement verifies every compiler stage before lineage checks.
No numerical kernel, native image ABI or physical schedule changes.
Validation: Super-Bear WSL combined frontend/portable/attention/movement/
MXFP8/dynamic projection suite 125 passed. Full Python Ruff passes. Owning
RTX 5070 bounded dynamic M/K and M/N/K numerical cases pass for FP16/BF16;
the metadata plus owning subset passed 11 tests, 96 deselected. Complete
NVIDIA tensor-program and attention-program regression files pass 107 tests
with the matching compiler and owning RTX 5070. Regenerated bootstrap, alpha,
compiler-progress, domain-proof and freshness views; focused audit gates
59 passed.
Remaining: broader unit environment/source failures, derived evidence drift,
owning ROCm replay checks and publication. No universal backend or AD closure.
Evidence: tests/unit/test_bounded_lhs_frontend_contract.py;
tests/unit/test_nvidia_portable_lhs_seal.py;
tests/unit/test_native_attention_program.py;
tests/unit/test_nvidia_tensor_program.py.
<!-- entry-fields:end -->



### 2026-10-06 — owning contract replay and portable drift repair

Owner: [E2E-REAL-6](INTEGRATED_COMPILER_PLAN.md#e2e-real-6)
PRs: pending; sync COMPILER-CONTRACT-REVALIDATION-2026-10-06.
Outcome: Native metadata/type repairs replay on all owning GPU hosts:
RTX 5070 producer/attention 107 pass; gfx1201 ingest/storage 50 pass;
gfx1201 math/widening/movement 106 pass with two sibling skips; gfx1151
107 pass with one sibling skip. Log records now use canonical linked owners
and each plan Latest resolves to its final owned append. Required packed-op
keywords are classified as attributes; the ODS inventory pins the eight
added operations. API specification names all three public packed operations.
Historical shape-key and host-regression receipts have explicit documentation.
Checkpoint mocks carry the current typed compact options; native AD mutation
tests preserve semantic lineage independently of SSA spelling.
Remaining: Portable integration gate is not yet green. Before repair it had
33 failures, 18271 passes and 9019 skips. General scaled-matmul batching/
transpose integration, remaining dashboard/evidence drift and publication
remain open. Device placement is repaired: six hardware-marked functions
now live in the NVIDIA device root, preserving 60 numerical cases. Owning
RTX 5070 replay plus two existing placement guards pass all 62 checks;
focused Ruff passes. No broader batching/transpose closure is claimed. No kernel/ABI/schedule or promotion.
Evidence: benchmarks/baselines/compiler_contract_revalidation_20261006/README.md;
348 registry/spec/diagnostic/metadata checks, 1011 native AD/claim checks,
zero-error mypy and full Python Ruff pass.
Follow-through: Generic scaled Graph transpose typing now agrees with MLIR
free-result verification. A direct scheduled NVFP4 replay exposed and repaired
SM120 logical nvfp4 admission overwritten by the shared uint8 ingest override.
Matching compiler rebuild passes; 540 focused registry/package gates and 34
capability gates pass. Three scheduled shapes are oracle-exact; four RTX 5070
native package tests pass. Packet nvidia_nvfp4_post_verifier.json separates
CUDA-event and end-to-end samples with source/compiler hashes. Manifest names
the bounded descriptor route; no generic batching/AD/sibling promotion.
<!-- entry-fields:end -->


### 2026-10-06 — native shared-RHS NVFP4 batching

Owner: [W1.1](INTEGRATED_COMPILER_PLAN.md#w11)
Related owners: E2E-REAL-6 / FRONTEND-IR-MEDIUM-1.
PRs: pending; sync NVIDIA-NVFP4-SHARED-RHS-BATCH-2026-10-06.
Outcome: Named static rank-three logical A/scales/output with shared rank-two
RHS/scales remain in Graph IR; native C++ Schedule flattens rows without
copying and native Tile/Target/LLVM/PTX performs one GPU launch. Shape guards
retain separate batch/row axes. Scale policy remains exact K16 UE4M3/fp32.
Driver routes the five-buffer ABI and reuses its existing Schedule, verified
against original Graph and native Tile replay. Equal-flat-row logical Graph
rebinding is rejected. Existing rank-two authored scaled packages also execute.
Validation: Matching core/NVIDIA target builds; 16 host/compiler/device tests
(five owning numerical cases) and 560 focused registry/package checks pass.
Zero-error mypy. Three RTX 5070 batch benchmark rows are oracle-exact; separate
native-batch and serial event/wall timings have five samples, 50 repetitions,
ten warmups, compiler/source hashes and actual device report. No default change.
Remaining: Generic vmap, independent-RHS/dynamic batches, wider storage,
transpose/AD products, sibling hardware proof, final integration gates and
publication. Native lowering exists; batching remains partial.
Evidence: benchmarks/baselines/compiler_contract_revalidation_20261006/README.md;
nvidia_nvfp4_shared_rhs_batch.json; tests/device/nvidia/test_nvfp4_shared_rhs_batch.py.
Follow-through regression: 287 pass, 49 sibling/compiler skips; all registered
derived views regenerate. Closure replay: 18 pass, two failures retain generic
scaled batching/linear-transpose obligations. Full Python Ruff passes; Graphify
CLI unavailable on owning WSL host. Final full-suite and publication remain open.
<!-- entry-fields:end -->


### 2026-10-06 — native independent-RHS NVFP4 batching

Owner: [W1.1](INTEGRATED_COMPILER_PLAN.md#w11)
Related owners: E2E-REAL-6 / FRONTEND-IR-MEDIUM-1.
PRs: pending; sync NVIDIA-NVFP4-INDEPENDENT-RHS-2026-10-06.
Outcome: Typed Python source emits named static rank-three logical A/B/scales
and output. Native Schedule preserves batch ownership in a batch-sensitive
digest, native Tile/Target/LLVM/PTX uses whole M16 row tiles per batch, and a
checked ten-argument ABI carries exact rows/count. Launch and event timing
reject equal-flat-M rebinding before native copies. Legacy shared/rank-two
packages replay without a Python production batch loop.
Validation: Matching core/NVIDIA builds; 43 package/host/device checks, 561
shared registry/ABI regressions; zero-error mypy, lint. Three RTX 5070 batch
benchmark rows are oracle-exact, with separate event and wall intervals.
Remaining: General vmap/dynamic batching, wider storage, transpose/AD, sibling
physical proof, full integration and publication. No selector/default change.
Evidence: benchmarks/baselines/compiler_contract_revalidation_20261006/README.md;
nvidia_nvfp4_independent_rhs_batch.json;
tests/device/nvidia/test_nvfp4_independent_rhs_native.py.
<!-- entry-fields:end -->


### 2026-10-06 — shared source types and GFX1151 softmax parity

Owner: [W1.1](INTEGRATED_COMPILER_PLAN.md#w11)
Related owners: E2E-REAL-6 / FRONTEND-IR-MEDIUM-1 / ROCM-E2E-1.
PRs: pending; sync COMPILER-SHARED-REPLAY-2026-10-06 and ROCM-FP16-SOFTMAX-PARITY-2026-10-06.
Outcome: Native source comparison projection updates structured inferred types
with printed i1 results before explicit mask conversion. Existing gfx1151
FP16 softmax Graph admission is restored without changing its native schedule.
Validation: Matching ROCm core rebuilds; gfx1201 217 tests and gfx1151 169 tests
pass. Source/checkpoint replay 75 passed, two skipped; shared identity/audit/
diagnostic/pass gates 342 passed; zero-error source mypy and lint. GFX1151
softmax differential 67 passed. Eight numerical benchmark rows pass with
separate HIP-event and end-to-end timing; wall gains are host overhead only.
Remaining: Generic scaled batching/transpose/AD closure, full integration,
publication and sibling physical coverage. Graphify unavailable on owning WSL.
Evidence: benchmarks/baselines/compiler_contract_revalidation_20261006/README.md;
rocm_gfx1151_softmax_parity.json and owning shared replay/build identity receipts.
<!-- entry-fields:end -->


### 2026-10-06 — native named NVFP4 policy preservation

Owner: [W1.1](INTEGRATED_COMPILER_PLAN.md#w11)
Related owners: E2E-REAL-6 / FRONTEND-IR-MEDIUM-1.
PRs: pending; sync NVFP4-NAMED-POLICY-PARITY-2026-10-06.
Outcome: Direct authored MLIR cannot add numeric or scale-layout policy fields
that would be dropped by native scheduling. Graph verification and Schedule
derivation require the same exact named dictionaries as Python packaging.
The valid K16/FP32 computation and both batch ABIs retain their contracts.
Validation: Three positive-control reproductions exposed policy loss before
the fix. Matching LLVM/MLIR 23.1.1 assertions all-target build passed. Native
policy and RTX 5070 NVFP4 checks pass in a 459-test replay; its single AMD
toolkit setup failure is resolved by a real scratch SDK subset, followed by
ten passing host-only ROCm packaging tests. Final combined replay: 460 passed,
16 skipped, including native exact-policy and owning NVFP4 batch checks.
Remaining: Generic batching/transpose/AD closure, aggregate integration and
publication. No new kernel timing, selector promotion or sibling execution.
Evidence: benchmarks/baselines/compiler_contract_revalidation_20261006/README.md;
nvfp4_named_policy_before.txt; alltarget_named_policy_build_identity.txt.
<!-- entry-fields:end -->

### 2026-10-06 — ordinary logical NVFP4 JIT binding

Owner: [W1.1](INTEGRATED_COMPILER_PLAN.md#w11)
Related owners: FRONTEND-IR-MEDIUM-1; E2E-REAL-6.
PRs: pending; sync NVIDIA-NVFP4-LOGICAL-JIT-2026-10-06.
Outcome: logical NVFP4 metadata over caller-owned packed host buffers,
  preserving native Graph/Schedule/Tile/Target computation and existing ABIs.
Validation: 18 owning host/device checks; 460 regression checks; 115 manifest/
  tracing checks with 23 skips; zero type errors in three changed sources.
Benchmark: six RTX 5070 rows match an independent fp64 oracle exactly;
  cold/warm host call and resident native event timing domains remain separate.
Siblings: physical binding follow-up required for Apple/ROCm/x86; no physical
  proof transfer or generic transform/AD promotion.
Remaining: general batching/transpose/AD, final integration and publication.
Evidence: benchmarks/baselines/compiler_contract_revalidation_20261006/README.md.
<!-- entry-fields:end -->

### 2026-10-06 — public native NVFP4 batch transformation

Owner: [W1.1](INTEGRATED_COMPILER_PLAN.md#w11)
Related owners: FRONTEND-IR-MEDIUM-1; E2E-REAL-6.
PRs: pending; sync NVIDIA-NVFP4-NATIVE-VMAP-2026-10-06.
Outcome: public vmap projects logical leading axes into existing named batch
  Graph/Schedule/Tile packages, with one native launch and separate JIT ownership.
Validation: 367 host/device/transform/registry checks passed; two rank-two
  non-batch parameterizations skipped. General closure tests remain enforced.
Siblings: Apple/ROCm/x86 physical follow-up required; no proof transfer.
Remaining: general vmap/compositions/dynamic batches, transpose/AD, full
  integration and publication.
Evidence: benchmarks/baselines/compiler_contract_revalidation_20261006/README.md.
<!-- entry-fields:end -->

### 2026-10-06 — symbolic native NVFP4 vmap contracts

Owner: [W1.1](INTEGRATED_COMPILER_PLAN.md#w11)
Related owners: FRONTEND-IR-MEDIUM-1; E2E-REAL-6.
PRs: pending; sync NVIDIA-NVFP4-VMAP-SYMBOLIC-2026-10-06.
Outcome: symbolic dtype-unresolved Graph annotations defer packed byte shape
  inference; mapped scalar constraints retain M/N/K/S bindings and no-map
  vmap semantics. Batch axes use a collision-free symbol.
Validation: 81 shape/constraint checks; 22 owning device/host checks with two
  non-batch skips. RTX 5070 equal-flat-row geometries retain distinct package
  identities and reuse the original specialization on return.
Siblings: Apple/ROCm/x86 physical follow-up remains; no device proof transfer.
Remaining: general/composed/dynamic batching, transpose/AD, aggregate gates
  and publication. No primitive closure gate or selector changed.
Evidence: benchmarks/baselines/compiler_contract_revalidation_20261006/README.md.
<!-- entry-fields:end -->

### 2026-10-06 — native NVFP4 operand orientation

Owner: [W1.1](INTEGRATED_COMPILER_PLAN.md#w11)
Related owners: FRONTEND-IR-MEDIUM-1; E2E-REAL-6.
PRs: pending; sync NVIDIA-NVFP4-NATIVE-ORIENTATION-2026-10-06.
Outcome: native Graph/Schedule/Tile carries logical A/B orientation for codes
  and K16 scales through PTX and checked physical shapes; existing ABIs remain.
Validation: matching LLVM/MLIR 23.1.1 core/NVIDIA builds passed; 23 new owning
  numerical cases passed. Metadata sealing and broader replay follow through.
Siblings: Apple/ROCm/x86 physical scope unchanged; no device proof transfer.
Remaining: transposed shared-batch A ABI, dynamic/composed transforms,
  linear-transpose AD products, aggregate gates and publication.
Evidence: benchmarks/baselines/compiler_contract_revalidation_20261006/README.md.
Standalone Schedule/Tile orientation verification: 67 focused tests passed, including 23 RTX 5070 cases; registry/audit gates: 333 passed. Enabled orientation flags require the named physical contract before target lowering.

<!-- entry-fields:end -->

### 2026-10-06 — gfx1201 packed vector-scale epilogue

Owner: [ROCM-MXFP4-W4A8-1](INTEGRATED_COMPILER_PLAN.md#rocm-mxfp4-w4a8-1)
Coordination: ROCM-MXFP4-W4A8-1 / ROCM-NVFP4-INGEST-1.
PRs: pending; sync GFX1201-PACKED-VECTOR-SCALES-2026-10-06.
Outcome: Native static M256 long-K packed epilogue uses guarded vector
activation-scale loads. 72 numerical checks and 344 selector/registry/pass
checks pass. Sixty matched three-format arms preserve control image hashes.
Measured gain is small; forward control drift is recorded explicitly.
Remaining: intermediate shape characterization, Radiance cost attribution,
staging/persistent strategies, AD closure and aggregate publication.
Evidence: benchmarks/baselines/gfx1201_packed_vector_scales_20261006/README.md.
<!-- entry-fields:end -->

### 2026-10-06 — ordinary forward native lineage certification

Owner: [E2E-REAL-6](INTEGRATED_COMPILER_PLAN.md#e2e-real-6)
Coordination: NVIDIA-LSE-1 / E2E-REAL-6.
PRs: pending; sync NVIDIA-FORWARD-LINEAGE-2026-10-06.

Outcome: Forward packages export real Graph/Schedule/Target digests after
native Graph -> Schedule -> Tile replay. Valid divergent policy fails before
target compilation. 69 tests pass, 6 sibling-environment skips; 46 owning
RTX 5070 cases retain lineage through serialization and execution. Mypy
zero errors and Ruff clean. All 24 regenerated checkpoint benchmark arms
have complete ancestry and per-window oracle checks. All backends assessed.

Remaining: Explicit LSE cotangents, general composed/dynamic routes,
scaled-matmul batching/transpose closure, ROCm performance and publication.

Evidence: benchmarks/baselines/nvidia_checkpoint_event_readback_20261006/README.md,
forward-lineage-tests.txt, plain-rtx5070.json, bias-rtx5070.json.
<!-- entry-fields:end -->

### 2026-10-06 — native attention event output readback

Owner: [E2E-REAL-6](INTEGRATED_COMPILER_PLAN.md#e2e-real-6)
Coordination: NVIDIA-LSE-1 / E2E-REAL-6.
PRs: pending; sync NVIDIA-EVENT-READBACK-2026-10-06.

Outcome: Native forward/backward profilers return final timed outputs after
the stop event. Poisoned output buffers prevent stale smoke results from
passing per-window oracle checks. Native build passed; 63 device/contract
checks and 292 registry checks pass, mypy zero errors, Ruff clean. Twenty-four
isolated RTX 5070 arms retain 120 event and 120 wall-window correctness
records with matching source/tool hashes. All four backend queues assessed.

Remaining: Explicit LSE-cotangent native AD, wider composed/dynamic routes,
general scaled-matmul closure gates and publication. Recompute backward
already has native Graph/Schedule/Tile lineage; ordinary comparator ancestry
is recorded independently. No selector promotion or sibling device claim.

Evidence: benchmarks/baselines/nvidia_checkpoint_event_readback_20261006/README.md,
plain-rtx5070.json, bias-rtx5070.json, device-and-contract-tests.txt.
<!-- entry-fields:end -->

### 2026-10-06 — saved-LSE tuple SSA bindings

Owner: [FRONTEND-IR-MEDIUM-1](INTEGRATED_COMPILER_PLAN.md#frontend-ir-medium-1)
Coordination: NVIDIA-LSE-1 / FRONTEND-IR-MEDIUM-1 / E2E-REAL-6.
PRs: pending; sync NVIDIA-SAVED-TUPLE-SSA-2026-10-06.

Outcome: Local tuple aliases preserve all Graph results through copying,
destructuring and constant-index selection. Rebinding clears stale tuple
bindings; scalar aliases reference defined SSA values. 122 tests pass,
1 skips; six new RTX 5070 cases and eighteen correctness-gated benchmark
rows cover bound/copied/destructured O/LSE outputs. Mypy zero errors and
focused Ruff clean. All four backend plans assessed.

Remaining: General nested/dynamic tuples, composed attention, native
explicit-LSE cotangents and general scaled-matmul batching/transpose closure.

Evidence: benchmarks/baselines/nvidia_ordinary_attention_20261006/README.md,
public-saved-lse-tuple-aliases-rtx5070.json, saved-tuple-alias-tests.txt.
<!-- entry-fields:end -->

### 2026-10-06 — public saved-LSE compiler integration

Owner: [FRONTEND-IR-MEDIUM-1](INTEGRATED_COMPILER_PLAN.md#frontend-ir-medium-1)
Coordination: NVIDIA-LSE-1 / FRONTEND-IR-MEDIUM-1 / E2E-REAL-6.
PRs: pending; sync NVIDIA-PUBLIC-SAVED-LSE-2026-10-06.

Outcome: Optional public O/LSE tuple typing and catalog-only production
attention tracing reach native Graph/Schedule/Tile checkpoint packages.
Original Graph import preserves operand roles and broadcast-bias extents.
Explicit differential certification retains reference evaluation. 685 tests
pass, 7 skip; mypy has zero errors and focused Ruff passes. Eighteen queried
RTX 5070 rows validate independent FP64 O/LSE oracles before separate public
wall and resident event windows. All four backend plans are assessed.

Remaining: Native explicit LSE-cotangent differentiation, general dynamic/
composed attention, textual tuple aliases, full closure gates and publication.
Reference AD proof does not certify native auxiliary-output differentiation.

Evidence: benchmarks/baselines/nvidia_ordinary_attention_20261006/README.md,
public-saved-lse-rtx5070.json, public-saved-lse-regression.txt,
public-saved-lse-mypy.txt, public-saved-lse-lint.txt.
<!-- entry-fields:end -->

### 2026-10-06 — host compiler separation and recorder inventory

Owner: [E2E-REAL-6](INTEGRATED_COMPILER_PLAN.md#e2e-real-6)
Coordination: E2E-REAL-6 / FRONTEND-IR-MEDIUM-1.

PRs: pending; coordinated compiler-contract validation aggregate.

Outcome: repaired validation environment selection and named the two new NVFP4 recorders. Host LLVM precedes the relocated AMD SDK for unqualified C/C++ oracle compilation; AMD generation stays explicitly pinned.

Validation: full non-slow unit run: 25 failed, three errors, 21567 passed, 7506 skipped and 874 deselected. Focused repaired lanes: 340 passed. General scaled batching/linear-transpose closure remains open; tests and coverage states are unchanged.

Sibling assessment: all four backend queues record host validation scope; no owning GPU or CPU execution proof transfers from these host-oracle repairs.

Remaining: Full-unit failures/errors, generic scaled-matmul batching and
linear-transpose AD closure, and publication.

Evidence: benchmarks/baselines/compiler_contract_revalidation_20261006/README.md.

<!-- entry-fields:end -->

### 2026-10-06 — gfx1201 native MXFP8 long-K selector trial

Owner: [ROCM-FP8-BLOCKSCALE-1](INTEGRATED_COMPILER_PLAN.md#rocm-fp8-blockscale-1)
Coordination: ROCM-FP8-BLOCKSCALE-1 / E2E-REAL-6.
PRs: pending; sync ROCM-MXFP8-LONG-K-2026-10-06.
Outcome: native MLIR Schedule selects LDS K64 in a measured narrow-panel long-K occupancy envelope; independent K32 scale semantics and checked runtime ABI remain unchanged. Explicit seed policy validation now decodes the omitted ODS global default.
Validation: 84 paired gfx1201 benchmark arm validations with zero numerical-bound violations, verified paired identities and ISA hashes; 31 owning device tests; 115 native contract tests; 333 registry/audit gates.
Siblings: Apple/NVIDIA/x86 physical selection unchanged; no exact-device proof transfer. All four queues updated.
Remaining: aggregate integration and publication, broader W8A8 coverage and MXFP4 M=256 attribution. General batching and linear-transpose closure remain open.
Evidence: benchmarks/baselines/rocm_mxfp8_long_k_20261006/README.md.
<!-- entry-fields:end -->

### 2026-10-06 — native NVFP4 shared-RHS transposed A

Owner: [W1.1](INTEGRATED_COMPILER_PLAN.md#w11)
Coordination: W1.1 / FRONTEND-IR-MEDIUM-1 / E2E-REAL-6.
PRs: pending; sync NVIDIA-NVFP4-SHARED-TRANSPOSE-2026-10-06.
Outcome: frontend/Graph admission, sealed Schedule batch intent, Tile policy and native per-batch A/output addressing support shared rank-two B with transposed rank-three A. Existing ten-argument rows/batches ABI is retained.
Validation: matching core/NVIDIA builds; 133 focused tests including 30 RTX 5070 cases; 333 registry/audit checks; 24 matched orientation rows with current source/tool hashes and separate resident events/public wall timing.
Siblings: all four plans assessed; no sibling physical evidence transferred.
Remaining: general/dynamic/composed batching, AD linear-transpose, higher AD/residual integration, aggregate gates and publication.
Evidence: benchmarks/baselines/compiler_contract_revalidation_20261006/README.md.
<!-- entry-fields:end -->

## NVIDIA-PAIRED-COST-2026-10-07

Expanded existing JIT saved attention VJP recorder to full capture/backward wall timing. RTX 5070 passes 18 rows and 54 paired numerical windows; caller mutation proves private residual generation. Respects pruned gradient activity and frame-owned borrowed views. No production selector promotion. All four backend plans assessed. Evidence: benchmarks/baselines/nvidia_paired_attention_cost_20261007/README.md.

## NVIDIA-MATCHED-PAIR-2026-10-07

Matched checked host-buffer saved/recompute forward/backward pairs pass 60 poisoned numerical windows on RTX 5070. Both-stage native ancestry and exact current hashes are verified. Native paired AD already retains O/LSE; misleading recorder default metadata corrected. No production selector change. All sibling plans assessed. Evidence: benchmarks/baselines/nvidia_matched_checkpoint_pair_20261007/README.md.

## NVIDIA-LSE-COTANGENT-LEAF-2026-10-07

Native SM120 saved backward includes P*dLSE in Q/K/bias, preserving dV. Shared Tile boolean/storage/pointer contract and explicit ROCm refusal prevent sibling misexecution. Matching builds and 41 checks pass on RTX 5070; 36 diagnostic rows / 180 poisoned event windows pass. Public Graph/AD/package integration remains open. Recorder: benchmarks/nvidia/record_lse_cotangent_leaf.py. Evidence: benchmarks/baselines/nvidia_lse_cotangent_leaf_20261007/README.md.

## NVIDIA-LSE-COTANGENT-SCHEDULE-2026-10-07

Explicit Graph dLSE operands now survive native Graph/Schedule/Tile with sealed/replayed roles. Matching compilers/runtime and 36 new Graph-generated numerical cases pass on RTX 5070; 36 scheduled diagnostic rows / 180 poisoned event windows pass. Shared Graph/dialect and Schedule metadata changes assessed across four backend queues. Existing bridge refuses seeded symbols until checked ABI integration. Public package, multi-result AD and residual owner remain open. Recorder: benchmarks/nvidia/record_lse_cotangent_leaf.py --scheduled. Evidence: benchmarks/baselines/nvidia_lse_cotangent_schedule_20261007/README.md.

## NVIDIA-LSE-COTANGENT-BRIDGE-2026-10-07

Native seeded host/resident/event bridges pass independent numerical checks on RTX 5070 (81 tests in the overlapping device lane). No Python production lowering added. All four backend queues assessed. Descriptor/common-runtime/AD integration remains open. Evidence: benchmarks/baselines/nvidia_lse_cotangent_bridge_20261007/README.md.

## NVIDIA-LSE-COTANGENT-PACKAGE-2026-10-07

Native Graph/Schedule/Tile saved-LSE row-seed consumers now project a checked f32 package and common runtime registration, serialized compiler-free replay, host/resident/event numerical checks and pre-CUDA invalid shape/stride/scalar/alias rejection. RTX 5070: 36 package cases plus eight negative cases and 322 registry checks pass; 36 recorder rows / 360 checked timing windows bind current source and compiler/runtime hashes. All four backend queues assessed. Compact/bias-gradient package, automatic multi-result AD and residual ownership remain open. Recorder: benchmarks/nvidia/record_lse_cotangent_package.py. Evidence: benchmarks/baselines/nvidia_lse_cotangent_package_20261007/README.md.

## NVIDIA-LSE-COTANGENT-GRADIENT-ROLES-2026-10-07

Seeded native checkpoint packages now cover all compact activity masks, packed/logical layouts, 64/128 threads and complete physical bias gradients. C++ entry parsing and pointer roles preserve the saved-LSE seed; numerical metadata binds checkpoint identity. RTX 5070 final seeded gate: 119 passed; checkpoint/registry regressions: 496; legacy compact: 39. Shared timing engine records 66 source-bound rows / 660 poisoned numerical windows. All four queues assessed. Public multi-result AD generation, private residual owner and dynamic/composed integration remain open. Evidence: benchmarks/baselines/nvidia_lse_cotangent_gradient_roles_20261007/README.md.

## NVIDIA-MULTIRESULT-ATTENTION-AD-2026-10-07

Native reverse AD now maps both semantic FlashAttention O/LSE results and cotangents through saved checkpoints to the seeded package; paired replacement preserves both SSA results. Physical metadata validation excludes sibling lineage. RTX 5070: 72 generated-AD numerical cases; checkpoint/registry gates 491 passed; recorder 144 producer/consumer rows / 1440 checked windows binds source and compiler/runtime hashes. All four queues assessed. Public JIT/private residual owner and dynamic/composed/batching integration remain open. Evidence: benchmarks/baselines/nvidia_multiresult_attention_ad_20261007/README.md.

### 2026-10-07 — native shared-LHS NVFP4 batch integration

Owner: [W1.1](INTEGRATED_COMPILER_PLAN.md#w11)
PRs: pending; sync NVIDIA-NVFP4-SHARED-LHS-2026-10-07.
Outcome: Public leading-axis vmap maps RHS and RHS scales while sharing A and
A scales. Typed Graph verification, sealed Schedule intent, Tile batch policy
and NVIDIA native lowering use one checked rows/batches launch without
Python replication or per-member launch loops. All four backend plans assess
the shared contract; this named physical profile remains SM120-only.
Validation: Matching final core/NVIDIA build and 89 frontend/runtime/native/
device checks pass on RTX 5070. Registry/manifest checks passed 406 tests;
mypy reports zero errors without baseline changes. All 32 generated documents
are in sync; 22 audit/citation/routing regressions pass and compiler-plan
ownership/navigation checks pass.
Outcome detail: Final-source packet verifies 32 numerical orientation rows,
eight shared-LHS rows, and all recorded source/tool hashes.
Remaining: Publication; dynamic/nested
maps, arbitrary producer composition and generic scaled-matmul AD. The fresh
generic batching/transpose gate remains two failures and four passes; its
coverage status and assertions are retained.
Recorder: benchmarks/nvidia/record_nvfp4_transpose.py.
Evidence: benchmarks/baselines/nvidia_shared_lhs_batch_20261007/README.md.
<!-- entry-fields:end -->

### 2026-10-07 — checkpoint Graph ancestry

Owner: [E2E-REAL-6](INTEGRATED_COMPILER_PLAN.md#e2e-real-6)
Coordination: NVIDIA-LSE-1 / E2E-REAL-6.
PRs: pending; sync NVIDIA-CHECKPOINT-GRAPH-LINEAGE-2026-10-07.
Outcome: Saved checkpoint packaging verifies original native Graph replay
before exposing its digest. Saved/recompute descriptors name matching Target
digests. Missing or conflicting source is rejected before target compilation.
115 device/contract, 80 focused (six skips) and 122 paired/broadcast checks pass;
counts overlap. No numerical or native ABI changes.
Recorder: static checks clean; 95 window/contract checks pass (six skips).
Forty-eight exact RTX 5070 arms verify all 480 event/wall oracle windows and
current source/core/NVIDIA compiler/native runtime hashes. Adaptive counts
are explicit; long recompute backward exposes a native residual-selection
optimization opportunity, with no selector promotion.
Remaining: long-shape residual selection, explicit LSE cotangents,
general AD closure and publication.
Evidence: benchmarks/baselines/nvidia_checkpoint_graph_lineage_20261007/README.md.
<!-- entry-fields:end -->

### 2026-10-07 — current-source gfx1151 native math and movement

Owner: [E2E-REAL-6](INTEGRATED_COMPILER_PLAN.md#e2e-real-6)
Coordination: FRONTEND-IR-MEDIUM-1 / ROCM-E2E-2.
PRs: pending; sync GFX1151-CURRENT-SOURCE-2026-10-07.
Outcome: Matching current-source LLVM/MLIR 23.1.1 core/ROCm builds and native
HIP ownership execute on Princess-Luna gfx1151. 237 tests pass; one foreign
gfx1201 movement case skips. Seventy-two f32/f16/bf16 math rows verify JIT/
portable numerical parity and shape-independent image reuse. Five native
resident rows verify 180 bit-exact warm windows, 90 native single-kernel
events, stale-generation/index rejection and repeated logical page mappings.
All four backend queues assess scope without transferring physical proof.
Remaining: Paired paged-KV resident launch-window timing still fails the 10%
non-regression gate; enqueue gaps/startup outliers are distinguished from
native single-kernel events. Broader layouts/dynamic/composed/AD routes,
other-family revalidation, full unit gates and publication remain open.
Evidence: benchmarks/baselines/gfx1151_current_source_revalidation_20261007/README.md.
<!-- entry-fields:end -->

### 2026-10-07 — fresh compiler unit gate

Owner: [FRONTEND-IR-MEDIUM-1](INTEGRATED_COMPILER_PLAN.md#frontend-ir-medium-1)
Coordination: E2E-REAL-6; COMPILER-FULL-UNIT-2026-10-07.
PRs: pending.
Outcome: Frozen-source Super-Bear WSL non-slow unit gate reports 21,694
passed, 7,503 skipped, 874 deselected and three failures. Generic scaled_matmul
batching and transpose closure are the two implementation failures; the
third is the new packet's tracked-document citation, corrected separately.
All four backend queues cite scope without transferring physical proof.
Remaining: Implement generic native scaled-product batching/AD contracts;
retain the closure assertions and partial/planned status until proved.
The isolated ROCm address candidate is outside this unit snapshot.
Evidence: benchmarks/baselines/compiler_full_unit_20261007/README.md.
<!-- entry-fields:end -->


### 2026-10-07 — native paged-KV flat-token indexing

Owner: [E2E-REAL-6](INTEGRATED_COMPILER_PLAN.md#e2e-real-6)
PRs: pending; sync ROCM-PAGED-KV-FLAT-INDEX-2026-10-07.

Outcome: The native ROCm generator indexes compact tokens without separate
head/feature division. gfx1151/gfx1201 pass bit-exact same-allocation A/B
windows across six shapes, including repeated page mappings and odd extents.
Focused gates pass 339 and 336 cases respectively. HIP-event launch-window
gains exceed identical-image control noise; windows include Python enqueue
gaps. Coordinated compiler rebuild and 331 shared tests pass; 13 device-gated
tests skip on SM120. All four queues assessed.

Remaining: General layouts, dynamic/composed consumers, isolated kernel/public
timing closure, wider family performance and generic scaled_matmul batching/
linear-transpose integration. No selector promotion or sibling proof transfer.

Evidence: benchmarks/baselines/rocm_paged_kv_flat_index_20261007/README.md,
gfx1151/ab.json, gfx1151/control.json, gfx1201/ab.json, gfx1201/control.json.
<!-- entry-fields:end -->


### 2026-10-07 — native compiler regression repair

Owner: [W1.1](INTEGRATED_COMPILER_PLAN.md#w11)
PRs: pending; sync COMPILER-NATIVE-LANES-2026-10-07.

Outcome: Eight native regression failures are repaired: current route/scale
fixtures, owning SDK propagation, and valid high-level kernel intent before
LLVM ABI materialization. Core lane passes 485 cases, with 66 unsupported
feature cases; both backend lanes pass 153 cases. Registry/inventory gates
pass 374 tests. Fresh RTX 5070 ordinary JIT NVFP4 and saved-LSE attention
forward/backward numerical/ownership checks pass 159 cases. All four queues
are assessed; ROCm serialization is artifact evidence, not physical proof.

Remaining: Generic scaled_matmul batching/linear-transpose AD, wider dynamic/
nested/composed routes, full-unit/fleet-union gates and PR publication.
SM90 high-level kernel intent still requires executable materialization and
owning-device proof; no new speedup or sibling parity claim.

Evidence: benchmarks/baselines/compiler_native_lane_repair_20261007/README.md,
core-final.json, backends-after.json, registry-tests.txt, device-tests.txt.
<!-- entry-fields:end -->


### 2026-10-07 — native scaled-product JVP foundation

Owner: [FRONTEND-IR-MEDIUM-1](INTEGRATED_COMPILER_PLAN.md#frontend-ir-medium-1)
PRs: pending.
Outcome: native exact-per-block f32 TangentInterface preserves scale groups; four native Graph/Schedule fixtures and 314 focused drift tests pass.
Remaining: repeated Schedule artifact ownership/lifetime, transposed FP8 admission, native sum/program integration, owning-device numerical and timing proof, generic batching/transpose closure.
Evidence: benchmarks/baselines/scaled_matmul_native_jvp_20261007/README.md.
<!-- entry-fields:end -->

### 2026-10-07 — repeated-product Schedule artifact binding

Owner: [FRONTEND-IR-MEDIUM-1](INTEGRATED_COMPILER_PLAN.md#frontend-ir-medium-1)
PRs: pending; sync SCALED-MATMUL-ARTIFACT-BINDING-2026-10-07.
Outcome: Native matmul instance bindings preserve content hashes and validate Graph/Schedule/artifact ownership. Scale-seed JVP reaches Tile/ROCm Target. Core native fixtures pass 489 cases with 66 unsupported.
Remaining: Native multi-product sum/symbol/ABI ownership, allocation lifetime, transposed FP8 admission and owning AD numerical/timing proof; generic full-unit closure remains open.
Evidence: benchmarks/baselines/scaled_matmul_artifact_binding_20261007/README.md.
<!-- entry-fields:end -->

### 2026-10-07 — native scaled-product program export

Owner: [FRONTEND-IR-MEDIUM-1](INTEGRATED_COMPILER_PLAN.md#frontend-ir-medium-1)
PRs: pending; sync SCALED-MATMUL-NATIVE-PROGRAM-2026-10-07.
Outcome: Compiler-owned paired program export retains actual products/sum, typed buffer bindings and SSA lifetimes. Primal/tangent storage metadata is preserved. Combined native fixtures pass 643 with 66 unsupported; focused AD/registry tests pass 347 with 16 skips.
Remaining: Native member image/projection, sum execution, physical allocation/stream/completion ownership, owning AD numerical/timing proof and generic full-unit closure.
Evidence: benchmarks/baselines/scaled_matmul_native_program_20261007/README.md.
<!-- entry-fields:end -->

### 2026-10-07 — native scaled-product member images

Owner: [FRONTEND-IR-MEDIUM-1](INTEGRATED_COMPILER_PLAN.md#frontend-ir-medium-1)
PRs: pending; sync SCALED-MATMUL-NATIVE-MEMBERS-2026-10-07.
Outcome: Native member projection preserves the full program witness, exact operations and buffer roles. All three FP8 products and the sum compile to gfx1201 HSACO; combined native lanes pass 644 with 66 unsupported and focused AD/registry tests pass 347 with 16 skips.
Remaining: Checked program image/ABI projection, native private-buffer and completion ownership, owning paired numerical/timing proof, and generic full-unit closure.
Evidence: benchmarks/baselines/scaled_matmul_native_members_20261007/README.md.
<!-- entry-fields:end -->

### 2026-10-07 — native scaled-product HIP ownership

Owner: [FRONTEND-IR-MEDIUM-1](INTEGRATED_COMPILER_PLAN.md#frontend-ir-medium-1)
PRs: pending; sync SCALED-MATMUL-NATIVE-OWNER-2026-10-07.
Outcome: A native HIP owner retains four compiler images and checked SSA buffers/lifetimes, submits the full paired sequence in C++, and completes before readback/release. Owning RX 9070 XT numerics, changed-input reuse, finite differences, rejection/cleanup checks and separate device/host timing windows pass.
Remaining: Automatic compiler/package image/ABI plan projection into the owner, ordinary public JIT AD execution, sibling physical integration and generic batching/transpose/full-unit closure.
Evidence: benchmarks/baselines/scaled_matmul_native_owner_20261007/README.md.
<!-- entry-fields:end -->

### 2026-10-07 — compiler-projected scaled-product native packages

Owner: [FRONTEND-IR-MEDIUM-1](INTEGRATED_COMPILER_PLAN.md#frontend-ir-medium-1)
PRs: pending; sync SCALED-MATMUL-NATIVE-PACKAGE-2026-10-07.
Outcome: Native SSA manifest binds differentiated Graph witnesses and storage/lifetimes; backend generators project entry symbols, scalar ABI and physical geometry. Package/registry corruption gates pass 305 tests. Owning gfx1201 replay passes numerics, changed-input reuse, finite differences and separate native device/host timings with compiler subprocesses disabled.
Remaining: Ordinary public JIT AD capture, transposed FP8 Schedule admission, general batching/linear transpose, sibling physical integration and full-unit closure.
Evidence: benchmarks/baselines/scaled_matmul_native_package_20261007/README.md.
<!-- entry-fields:end -->

### 2026-10-07 — canonical typed FP8 primal JIT

Owner: [FRONTEND-IR-MEDIUM-1](INTEGRATED_COMPILER_PLAN.md#frontend-ir-medium-1)
PRs: pending; sync ROCM-TYPED-SCALED-PRIMAL-2026-10-07.

Outcome: Ordinary gfx1201 JIT packages the original typed FP8 scaled Graph
through native Schedule/Tile/Target lowering. Checked descriptor argument and
shape-guard names follow actual SSA operand roles; the caller Graph is preserved.
Four primal cases cover KN/NK storage and M17/N19/K256, M200/N129/K1536.
Independent float64 numerics and compiler-free changed-input reuse pass.
Seventeen owning device regression cases and 337 shared unit/registry checks pass.
Public-call and same-image native event timings are recorded separately.

Remaining: MXFP8 unsigned-byte frontend integration, transpose-left, general
batching/composed AD and full-unit closure. Public primal staging costs remain
larger than native event windows. All four backend queues assessed; no sibling
physical support is inferred from gfx1201 evidence.

Evidence: benchmarks/baselines/rocm_typed_scaled_primal_20261007/README.md.
<!-- entry-fields:end -->

### 2026-10-07 — native primal transfer and bounded idle eviction

Owner: [FRONTEND-IR-MEDIUM-1](INTEGRATED_COMPILER_PLAN.md#frontend-ir-medium-1)
PRs: pending; sync ROCM-PRIMAL-TRANSFER-EVICTION-2026-10-07.
Outcome: Native gfx1201 moderate-size pinned staging and completed-owner LRU
eviction preserve four-owner/128 MiB bounds, quarantine and active ownership.
24 owning public primal/JVP tests and 296 registry gates pass. Eight paired
public rows record numerics and compiler-free reuse; large automatic costs
are 1.22-1.74 ms versus pageable 4.17-4.72 ms, with separate controls.
Remaining: Generic batching/transpose, eager E8M0 differential evaluation,
sibling owning proof, wider performance programs and full-suite/PR delivery.
Evidence: benchmarks/baselines/rocm_primal_transfer_attribution_20261007/README.md.
<!-- entry-fields:end -->

### 2026-10-07 — typed E8M0 eager and native parity

Owner: [FRONTEND-IR-MEDIUM-1](INTEGRATED_COMPILER_PLAN.md#frontend-ir-medium-1)
PRs: pending; sync E8M0-EAGER-PARITY-2026-10-07.
Outcome: Explicit E8M0 [1,32] typed eager reference decodes byte scales and
preserves smallest/NaN codes. 474 shared checks and 25 gfx1201 public
primal/JVP tests pass, including independent eager/native numerical comparison.
No native schedule/image/ABI changes; all four backend queues assessed.
Remaining: Generic batching/linear transpose, encoded-scale derivative policy,
composed/dynamic AD, broader performance closure and full-suite/PR delivery.
Evidence: benchmarks/baselines/rocm_e8m0_eager_parity_20261007/README.md.
<!-- entry-fields:end -->

### 2026-10-07 — native typed shared-RHS batches and scale JVP

Owner: [FRONTEND-IR-MEDIUM-1](INTEGRATED_COMPILER_PLAN.md#frontend-ir-medium-1)
PRs: pending; sync ROCM-SHARED-SCALED-BATCH-2026-10-07.
Outcome: Native Graph preserves batch ranks, Schedule flattens B*M without
replication, Tile binds original pointers, and Target/program ownership retain
exact logical/physical capacities and completion. 39 gfx1201 device tests,
541 shared checks, 222 target checks (one skip) and 494 native fixtures
(66 unsupported) pass. Twelve correctness-gated rows separate public,
prepared host and native sequence event timings. All four queues assessed.
Remaining: Independent/shared-LHS/dynamic/nested batching, linear-transpose
and general composed AD, sibling physical parity, full-unit and PR delivery.
Evidence: benchmarks/baselines/rocm_shared_scaled_batch_20261007/README.md.
<!-- entry-fields:end -->

### 2026-10-07 — native independent-RHS and shared-LHS scaled batches

Owner: [FRONTEND-IR-MEDIUM-1](INTEGRATED_COMPILER_PLAN.md#frontend-ir-medium-1)
PRs: pending; sync ROCM-INDEPENDENT-SCALED-BATCH-2026-10-07.
Outcome: Native named static typed FP8/MXFP8 independent-RHS/shared-LHS batches retain
logical matrix/scale storage and per-batch M through Schedule, Tile and Target.
Native grid-z memref views offset matrices, scales and output; shared LHS
storage is reused without Python replication or batch launch loops.
67 gfx1201 tests cover primal/JVP numerics, changed per-batch scales, finite
differences and warm compiler-free execution. The wider LDS shape initially
failed because image projection was incorrectly gated; the repair preserves
checked logical capacity while permitting projected runtime image dimensions.
Twenty-four correctness-gated primal rows separate public, prepared host and
native sequence-event costs.
Remaining: Dynamic/nested batching, transpose/composed AD, sibling physical parity,
generic full-unit closure and PR delivery remain open.
Evidence: benchmarks/baselines/rocm_independent_scaled_batch_20261007/README.md.
All four backend queues assessed.
<!-- entry-fields:end -->

### 2026-10-07 — immutable native scaled-program host binding

Owner: [FRONTEND-IR-MEDIUM-1](INTEGRATED_COMPILER_PLAN.md#frontend-ir-medium-1)
PRs: pending; sync ROCM-NATIVE-PLAN-BINDING-2026-10-07.
Outcome: Complete immutable contents key bounded checked decode and readonly
native ABI bindings, retaining native lifetime/capacity/context/completion checks.
68 gfx1201 default device tests and 386 shared gates pass. Six image-identical
paired rows measure 7-13% lower primal and 15-20% lower scale-JVP public cost.
All four backend queues assessed.
Remaining: MXFP4/NVFP4 and sibling owner attribution, dynamic/nested/composed
integration, general transpose AD, broader performance, full-unit and PR delivery.
Evidence: benchmarks/baselines/rocm_native_plan_binding_20261007/README.md.
<!-- entry-fields:end -->

### 2026-10-07 — public typed scaled vmap native integration

Owner: [FRONTEND-IR-MEDIUM-1](INTEGRATED_COMPILER_PLAN.md#frontend-ir-medium-1)
PRs: pending; sync ROCM-PUBLIC-TYPED-VMAP-2026-10-07.

Outcome: Public direct scalar FP8/MXFP8 vmap projects leading shared-RHS,
independent-RHS and shared-LHS batch intent into Graph IR and the existing
native Schedule/Tile/Target/image ABI. Scalar owners remain unchanged.
25 frontend regression tests, 362 shared drift/audit checks and 12 owning
gfx1201 numerical/warm-call tests pass. Twelve timing rows separately
record public wall, prepared host wall and native sequence events.
Public medians are 0.77-1.02 ms; no comparative speedup claim is made.
All four backend plans assess the shared frontend change.

Remaining: Dynamic/nested/nonleading/composed AD and linear transpose,
generic batching closure, sibling physical parity, full suite and PR delivery.

Evidence: benchmarks/baselines/rocm_public_typed_vmap_20261007/README.md,
timings.json, source-tools.json, frontend-tests.log, shared-tests.log,
device-tests.log.
<!-- entry-fields:end -->

### 2026-10-07 — wider public typed scaled vmap proof

Owner: [FRONTEND-IR-MEDIUM-1](INTEGRATED_COMPILER_PLAN.md#frontend-ir-medium-1)
PRs: pending; sync ROCM-PUBLIC-TYPED-VMAP-2026-10-07.

Outcome: 36 owning gfx1201 cases and 36 separate timing rows cover static
leading maps, including ragged M200 and wider LDS staging. Every recorded
native program has one step. Original narrow evidence is retained.

Remaining: Generic batching/transpose closure, dynamic/nested/composed AD,
sibling physical consumers, full-suite validation and PR delivery.

Evidence: benchmarks/baselines/rocm_public_typed_vmap_20261007/README.md,
timings-wide.json, source-tools-wide.json, device-tests-wide.log.
<!-- entry-fields:end -->

### 2026-10-07 — mapped native scale JVP integration

Owner: [FRONTEND-IR-MEDIUM-1](INTEGRATED_COMPILER_PLAN.md#frontend-ir-medium-1)
PRs: pending; sync ROCM-PUBLIC-MAPPED-JVP-2026-10-07.

Outcome: Public leading typed FP8 maps preserve native FP32 scale-JVP intent.
Projected Graph/reference certificates compare an eager scalar-map oracle;
native AD and checked packages own execution. 72 gfx1201 primal/JVP cases,
503 shared dtype/frontend/registry gates and 12 separate timing rows pass.
Explicit gated byte candidate specialization preserves the source declaration.
All four backend plans assess the shared owner/semantic changes.

Remaining: General dynamic/nested/nonleading/composed AD and transpose,
sibling physical consumers, full-suite validation and PR delivery.

Evidence: benchmarks/baselines/rocm_public_mapped_jvp_20261007/README.md,
timings.json, source-tools.json, device-tests.log, shared-tests.log.
<!-- entry-fields:end -->

### 2026-10-07 — native JVP source constraint parity

Owner: [FRONTEND-IR-MEDIUM-1](INTEGRATED_COMPILER_PLAN.md#frontend-ir-medium-1)
PRs: pending; sync ROCM-PUBLIC-MAPPED-JVP-2026-10-07.

Outcome: Public native JVP checks source shape constraints before capture or
compilation. Three mapped policy counterexamples reject an invalid M bound
before frontend work. 44 focused and 72 shared native-JVP/attention checks
pass (16 gated skips); 59 gfx1201 scalar/mapped JVP cases pass.
Twelve refreshed timing rows retain identical native images.
All four backend plans assess shared shape-constraint enforcement.

Remaining: Generic batching/transpose and AD closure, sibling exact-device
constraint parity, final full-suite validation and PR delivery.

Evidence: benchmarks/baselines/rocm_public_mapped_jvp_20261007/README.md,
constraint-tests.log, native-jvp-regressions.log, device-tests-constraints.log,
timings-constraints.json, source-tools-constraints.json.
<!-- entry-fields:end -->

### 2026-10-07 — five-slice unit contract repairs

Owner: [FRONTEND-IR-MEDIUM-1](INTEGRATED_COMPILER_PLAN.md#frontend-ir-medium-1)
PRs: pending; sync COMPILER-UNIT-CONTRACT-REPAIR-2026-10-07.

Outcome: A diagnostic full unit run found seven failures. Native scaled tangent
forward/proof registration, recorder consumers and ROCm early family admission
are repaired. A new exact image target/architecture guard rejects mislabeled
native primal programs before preparation. 110 focused checks pass with 90
gated skips; 330 drift/audit checks pass. gfx1201 revalidation passes 72
combined cases and 36 later primal cases. All four backend plans are assessed.

Remaining: The two genuine generic batching/transpose closure failures remain
unchanged. A frozen final full-suite gate, broader owning proof and curated PR
delivery remain required.

Evidence: benchmarks/baselines/compiler_unit_contract_repairs_20261007/README.md,
full-unit.log, focused-tests.log, drift-tests.log, device-tests.log,
device-image-guard-tests.log, device-source-tools.json.
<!-- entry-fields:end -->

### 2026-10-07 — native real-checkpoint ingest integration

Owner: [ROCM-NVFP4-INGEST-1](INTEGRATED_COMPILER_PLAN.md#rocm-nvfp4-ingest-1)
PRs: pending; sync ROCM-NATIVE-CHECKPOINT-2026-10-07.

Outcome: Typed Graph/Schedule/Tile/ROCm Target/LLVM conversion executes pinned
Qwen q_proj and gate/up bytes on gfx1201. Packed output is bitwise reference
equivalent; independent f64 block statistics pass before/after timing and
native consumer outputs match decoded weights exactly. Separate conversion/
consumer event and wall domains are retained; 359 focused tests pass, seven skip.
All four backend queues are assessed.

Remaining: Whole-model/source-activation quality, generic resident composition,
dynamic/layout/AD, shared full-suite closure and curated aggregate delivery.
No numerical-policy or selector promotion.

Evidence: benchmarks/baselines/rocm_native_checkpoint_20261007/README.md.
<!-- entry-fields:end -->

### 2026-10-07 — native multidimensional scaled batches

Owner: [FRONTEND-IR-MEDIUM-1](INTEGRATED_COMPILER_PLAN.md#frontend-ir-medium-1)
PRs: pending; sync ROCM-MULTIDIMENSIONAL-SCALED-BATCH-2026-10-07.

Outcome: Rank-four typed FP8/MXFP8 batch storage and scale-JVP execute through
ordinary Graph/native Schedule/Tile/Target/LLVM packages and checked lifetime/
completion ownership on gfx1201. 102 device tests, 24 timing rows, 381 shared
and 48 target/ABI checks pass; native fixtures pass 649 with 66 unsupported.
Nine exact RTX 5070 NVFP4 cases pass shared-Schedule regression. Both frontend
and native verifier reject equal-product permuted prefixes. All backend queues
are assessed; logical shapes survive native physical flattening.

Remaining: Nested/dynamic/nonleading map projection, generic linear transpose/
storage AD, sibling rank-four physical routes, full-unit closure and PR delivery.

Evidence: benchmarks/baselines/rocm_multidimensional_scaled_batch_20261007/README.md.
<!-- entry-fields:end -->

### 2026-10-07 — nested public typed map integration

Owner: [FRONTEND-IR-MEDIUM-1](INTEGRATED_COMPILER_PLAN.md#frontend-ir-medium-1)
PRs: pending; sync ROCM-NESTED-TYPED-VMAP-2026-10-07.

Outcome: Two matching leading public maps retain scalar capture, logical
prefixes, source bounds and independent owners through gfx1201 native primal/
scale-JVP programs. 102 device cases, 24 separate primal timing rows and 415
shared tests pass, with 16 gated skips. Existing single leading SM120 public
maps pass 11 exact RTX 5070 regressions. All backend queues are assessed.

Remaining: Mixed/dynamic/nonleading/deeper maps, generic linear transpose/
storage AD, sibling nested routes, generic full-unit closure and PR delivery.

Evidence: benchmarks/baselines/rocm_nested_typed_vmap_20261007/README.md.
<!-- entry-fields:end -->


### 2026-10-07 — typed scaled-product native transpose foundation

Owner: [FRONTEND-IR-MEDIUM-1](INTEGRATED_COMPILER_PLAN.md#frontend-ir-medium-1)
PRs: pending; sync SCALED-PRODUCT-TRANSPOSE-2026-10-07.

Outcome: Native LinearTransposeInterface constructs static E4M3FN/f32-scale
adjoints as structured tensor/scf reductions preserving original K/N groups,
mapped prefixes, shared batch reductions and ragged limits. Three native
fixtures pass paired/in-place construction plus tangent/transpose regressions.
338 focused registry/oracle gates pass. Twelve every-coordinate finite-
difference checks and 12 gfx1201 native-JVP duality cases pass. Warm native
JVP reuse forbids compiler/reference calls; this is not native VJP evidence.
All four backend plans retain their distinct physical boundaries.

Remaining: Native Schedule/Tile reduction members, immutable program export,
ABI/lifetime/completion integration, public reverse AD, owning gfx1201 scale
gradient numerics and separate event/program/public benchmarks. Generic
transpose closure and sibling physical execution remain open.

Evidence: benchmarks/baselines/scaled_product_transpose_foundation_20261007/README.md,
identity.json, native-tests.json, registry-tests.txt, gfx1201-duality.txt.
<!-- entry-fields:end -->

### 2026-10-07 — native scale-transpose program export

Owner: [FRONTEND-IR-MEDIUM-1](INTEGRATED_COMPILER_PLAN.md#frontend-ir-medium-1)
PRs: pending; sync SCALED-TRANSPOSE-PROGRAM-2026-10-07.

Outcome: Native paired export clones actual generated tensor/SCF regions,
derives captured-input order from SSA, preserves frontend permutations and
requested gradient order, and records byte storage and buffer lifetimes.
55 native export/package regressions, all 75 native AD fixtures and 11 owning
RTX 5070 existing NVFP4 public-map cases pass. Pass metadata/specification
and all four backend plans are updated. No production Python math or IR
reconstruction is introduced.

Remaining: Native Schedule/Tile reduction members, image/ABI packaging,
HIP ownership integration, public reverse AD, owning gfx1201 gradient
numerics and separate kernel/program/public timings. This is native program
export, not physical VJP execution, generic closure or sibling promotion.

Evidence: benchmarks/baselines/scaled_transpose_program_export_20261007/README.md,
identity.json, native-package-tests.txt, native-fixtures.json, sm120-regression.txt.
<!-- entry-fields:end -->

### 2026-10-07 — public native scale transpose

Owner: [FRONTEND-IR-MEDIUM-1](INTEGRATED_COMPILER_PLAN.md#frontend-ir-medium-1)
PRs: pending; sync SCALED-PUBLIC-TRANSPOSE-2026-10-07.

Outcome: Public reverse capture retains exact ROCm target metadata and binds
frontend arguments positionally through the native scaled-product plugin.
72 owning gfx1201 numerical/replay cases, 397 shared drift gates and 11
existing RTX 5070 public-map regressions pass. 48 public timing rows are
recorded separately from earlier native launch/prepared windows. Native
Graph/Schedule/Tile/Target/LLVM images and HIP lifetime ownership execute
without Python numerical differentiation or launch-sequence construction.

Remaining: Dynamic/nonleading/deeper/composed/storage AD, serial reduction
performance, sibling reverse execution and generic/full-unit/publication gates.

Evidence: benchmarks/baselines/rocm_public_scaled_vjp_20261007/README.md,
identity.json, scaled-vjp-public-device-final-20261007.log,
scaled-vjp-public-timing-20261007.json, shared-tests.txt, sm120-regression.txt.
<!-- entry-fields:end -->

### 2026-10-07 — native scale transpose wave candidate

Owner: [FRONTEND-IR-MEDIUM-1](INTEGRATED_COMPILER_PLAN.md#frontend-ir-medium-1)
PRs: pending; sync SCALED-TRANSPOSE-WAVE-2026-10-07.

Outcome: Native Schedule admits an explicit 32-lane additive outer reduction
with checked single-use accumulator lineage, original K-group arithmetic,
sealed Tile/Target body and codegen-owned algorithm/geometry ABI. 372 focused
export/package/image/registry gates and 24 same-compiler paired gfx1201
numerical/timing rows pass. Native event ratios span 1.752–18.379 with median
7.597. Python constructs no arithmetic or native launch sequence.

Remaining: Serial remains default; wider regimes, repeated controls, public
candidate timing, selector policy and generic/sibling/full-unit/publication
closure remain open. FP32-scale adjoints do not imply discrete scale AD.

Evidence: benchmarks/baselines/rocm_scaled_vjp_wave_20261007/README.md,
identity.json, scaled-vjp-wave-paired-20261007.json, native-registry-tests.txt.
<!-- entry-fields:end -->

### 2026-10-08 — non-leading typed scaled map integration

Owner: [FRONTEND-IR-MEDIUM-1](INTEGRATED_COMPILER_PLAN.md#frontend-ir-medium-1)
PRs: pending; sync SCALED-MAP-AXIS-INTEGRATION-20261008.

Outcome: Alias-only nested input-axis projection retains scalar bounds.
Native host packing checks positive storage spans and preserves strict compact
descriptor admission. Native Graph/Schedule/Tile/Target/LLVM continues to own
primal/JVP/VJP arithmetic. Reverse results restore original axes after native
unbroadcast, rather than reshaping a permuted adjoint. Exact gfx1201 execution
and separate completed-public/native-program-event timings are recorded;
all four backend queues are assessed.

Remaining: Generic scaled-product closure, dynamic maps, nonzero output axes,
wider storage derivatives, NVIDIA packed-axis integration and sibling native
parity remain open. No primitive coverage promotion or weakened closure gate.

Evidence: benchmarks/baselines/scaled_map_axes_20261008/README.md, device.json,
owning-tests.txt, host-tests.txt, mypy.txt, native-build.txt.
<!-- entry-fields:end -->

### 2026-10-08 — native scaled JVP frame preparation

Owner: [FRONTEND-IR-MEDIUM-1](INTEGRATED_COMPILER_PLAN.md#frontend-ir-medium-1)
PRs: #902; sync SCALED-MAP-AXIS-INTEGRATION-20261008.

Outcome: Checked native C++ byte packing replaces Python compact-copy frame
preparation for the admitted gfx1201 scaled JVP program. Pre-certification
backing checks reject unsafe views; row-block copies retain the checked ABI.
368 host, 36 gfx1201 and four RTX5070 attention JVP regression cases pass.
Matched same-package control records a 2.3–4.5% remaining public-call cost;
the original large-case 42% gap is reduced without changing GPU arithmetic.

Remaining: Native host preparation cost, generic batching/transpose closure,
dynamic/nonzero output axes, wider storage AD and sibling execution integration.
No primitive coverage promotion or weakened zero-open gate.

Evidence: benchmarks/baselines/native_scaled_jvp_frame_20261008/README.md,
delivery.json, scaled-jvp-frame-paired-final.json, host-tests.txt, sm120-tests.txt.
<!-- entry-fields:end -->

### 2026-10-08 — mapped result-axis semantic foundation

Owner: [FRONTEND-IR-MEDIUM-1](INTEGRATED_COMPILER_PLAN.md#frontend-ir-medium-1)
PRs: [#903](https://github.com/gstoner/tessera/pull/903), dependent on #902;
sync MAP-RESULT-AXES-FOUNDATION-20261008.

Outcome: Exact Graph transpose axes replace multiset-only verification.
Keyword/positional/negative frontend axes carry canonical permutations.
Native reverse AD inverts declared axes; static general-rank Linalg lowering
carries activity/pure facts. Symbolic constraints and SSA dimension names follow
the declared axes; native shape inference shares that decoder. 53 host cases include actual native CPU execution
and changed-input replay; 381 final axis/symbolic/diagnostic/pass cases pass.
451 focused registry/frontend/shape gates pass.
Diagnostic CPU timing records five profiles; compilation and execution are
separate. Existing gfx1201 and RTX5070 routes are checked independently.

Remaining: Native result-permutation program membership, Schedule/Tile
materialization, ownership/ABI and exact GPU timing, public nonzero out_axes,
dynamic/wider storage/layout and generic scaled-product closure. No primitive
coverage promotion or weakened closure gate.

Evidence: benchmarks/baselines/mapped_result_axes_foundation_20261008/README.md,
source-tests.txt, registry-tests.txt, cpu.json, identity.json.
<!-- entry-fields:end -->
