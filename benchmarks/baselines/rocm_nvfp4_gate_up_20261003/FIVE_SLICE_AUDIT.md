# Current five-slice acceptance audit

Objective: NVFP4 ingest, NVIDIA W1.1 producers, saved-LSE attention forward,
attention backward, and ROCm route/performance closure.
The broader goal remains active; current evidence proves named envelopes.

Acceptance clarification (2026-10-06): attention JVP/VJP source integration in
the capture packets uses the public compile/frame APIs. Earlier references to
ordinary JIT attention AD mean a traced JIT source feeding those explicit APIs;
they do not prove dispatch through the generic `native_jvp` family runtime.
That historical source audit preceded the named public route. The 20261006 public attention JVP packet now proves canonical native_jvp family/common-runtime dispatch for static distinct direct fp32 Q/K/V on SM120. The public/prepared reverse packets now prove canonical native_backward family/common-runtime dispatch and synchronous native ownership for static f32 direct Q/K/V and supported score bias on SM120. General composed/dynamic/higher AD, bias JVP and sibling physical consumers remain acceptance obligations.
Ordinary primal norm/matmul and ROCm ingest/movement call evidence is separate.

| Slice | Inspected authoritative evidence | Proven increment | Still open |
| --- | --- | --- | --- |
| gfx1201 NVFP4 ingest | gate-up.json, final-tests.txt; rocm_nvfp4_ingest.py; packed-MXFP4 Graph fixture | Real pinned layer-0 gate/up executes; direct-BF16 and ingested outputs agree with decoded weights; separate globals/row boundaries and declared loss are recorded | Graph/Schedule/Tile converter and explicit checked host package are synthetic-device-proven in rocm_graph_ingest_20261003; pinned native conversion and host-mediated consumer integration are proved in rocm_checkpoint_native_ingest_20261005; ordinary static converter JIT/common-runtime replay is proved in rocm_ingest_runtime_20261005; lossless N16/K64 Graph/Schedule/Tile storage bridge JIT and bitwise replay are proved in rocm_ingest_storage_bridge_20261005; named static resident converter/storage/packed-consumer edge and combined timing are proved in rocm_ingest_resident_20261005; portable resident serialization and fresh-process replay are proved in rocm_ingest_portable_20261005; ordinary static composed JIT/common-runtime execution and fresh-process replay are proved in rocm_ingest_jit_program_20261005; general frontend-AD integration, dynamic packing/layout variants and source-activation/whole-model quality remain open |
| NVIDIA W1.1 | TileIRLoweringPass.cpp recovery and sm<120 registration gates; nvidia_sm120_canonical_tensor_replay_20261002 packet | Whole-function replay recovers static canonical tensor reductions into Graph; Schedule/Tile owns views/fragments; fp16/BF16 plain/fused exact-device proof | Arbitrary producer graphs, noncanonical accumulators, dynamic generic reconstruction and older-device proof |
| Saved-LSE forward | test_lse_checkpoint_native.py; nvidia_jit_attention_vjp_20261002 and argument-order packets | Verified native O/LSE checkpoint products; independent output/LSE oracle and private saved state | General composed/aliasing producer graphs and wider dtype/layout envelopes |
| Attention backward | same device test and bias/argument-order packets; current NVIDIA queue | Saved/recompute Q/K/V and exact-shaped bias gradients agree with independent oracle; public compile API maps frontend argument order and requested gradients. The 20261003 broadcast package packet adds checked rank-four physical dBias reduction and private public capture; the 20261006 public/prepared packets add ordinary native_backward dispatch and native synchronous reverse ownership | Broadcast JVP/higher derivatives, dynamic/lower-rank bias, unused-cotangent pruning and general AD integration |
| ROCm route/performance | rocm_mxfp8_lds_20261003 validation/comparison; three-format and W8A8 packets; current ROCm queue | Native shape-guarded image reuse; matching FP8/MXFP8/MXFP4 gates; measured wide-grid LDS gain | Multi-group staging, M256 Radiance cost attribution, wider W8A8 envelopes, general cache/layout families and host movement overhead |

The current source registration was inspected directly: both historical tensor
MMA rewrite patterns remain only under smVersion < 120. Static SM120 recovery
requires a canonical K-step and verified whole-function replay; this does not
establish arbitrary tensor-producer closure. The saved-LSE device test explicitly
compares forward O/LSE and saved/recompute gradients against its independent
oracle and requires native_gpu for resident launches. Historical packet source
and compiler hashes retain their original revision; they are not relabeled as
fresh measurements from the newest compiler.

The previous turn's matching RTX 5070 build passed 86 existing matmul/attention
cases; the latest gfx1201 loop passed 623 adjacent tests plus the separate real
ingest evidence here. Those are envelope-specific checks, not a claim that all
five program families are closed. Native conversion ownership and generic
producer/AD envelopes need implementation/evidence before full completion.
FP8, MXFP8 and MXFP4 remain required before a global strategy decision.

Follow-on evidence: [SM120 broadcast checked packages and private tape](../nvidia_broadcast_checkpoint_package_20261003/README.md) adds exact-device physical storage, copy-size and public capture proof. This does not close the broader five-slice goal.

Follow-on W1.1 evidence: [native row-major RHS Schedule and RMSNorm producer](../nvidia_row_major_b_schedule_20261003/README.md) proves 20 checked layout package cases and eight resident RHS producer cases on RTX 5070. This closes the named static RHS edge increment; arbitrary/dynamic producer and format strategy obligations remain open.

Frontend follow-on: the explicit JIT compile_native_rhs_matmul API now traces the composed RMSNorm RHS-to-matmul source, verifies/rebuilds partition CFGs, preserves argument order and passes eight RTX 5070 numerical resident cases (nvidia_row_major_b_schedule_20261003/jit-packet.json). Automatic native dispatch, composed AD and general producers remain open.

## Ordinary RHS JIT runtime increment

[2026-10-03 ordinary RHS calls and portable native replay](../nvidia_rhs_jit_dispatch_20261003/README.md):
ten RTX 5070 FP16/BF16 benchmark rows, checked component receipts, owned
stream/private buffers and complete native Graph verification. This closes
the named static primal dispatch/replay gap; general graphs and composed AD
remain open. FP8, MXFP8 and MXFP4 remain mandatory independent evaluation
gates before a final/default strategy decision.

## LayerNorm RHS producer increment

[LayerNorm RHS native integration](../nvidia_layernorm_rhs_20261003/README.md)
extends ordinary JIT and portable replay to a second shape-preserving native
producer on RTX 5070, with centered-variance numerical proof, private buffers
and separate event/wall timing. General producers and composed AD remain open;
FP8, MXFP8 and MXFP4 evaluation remains required before strategy promotion.

## Pinned format decision gate

[Real gate/up FP8/MXFP8/folded MXFP4 comparison](../rocm_checkpoint_format_gate_20261003/README.md)
adds twelve exact gfx1201 native format/shape arms and fixes transposed input
scale-plane materialization. Quantization quality and resident/host timing
remain separate. Native packed MXFP4 and policy-gated MLIR ingest conversion
remain open; expanded folded weights are not packed native closure.


## Native packed consumer increment

[Native gfx1201 packed MXFP4](../rocm_packed_native_20261003/README.md)
replaces the public packed consumer's hand-emitted HIP materializer with
Graph/Schedule/Tile/Target native MLIR/LLVM lowering and proves checked/resident
execution. This closes the named packed-consumer materialization gap, while
native ingest conversion, general producer/AD and broader ROCm route/performance
requirements remain open. All three low-precision format gates are preserved.


## Native ingest physical leaf increment

[Native joint-SSE conversion](../rocm_native_ingest_20261003/README.md)
now materializes packed destination codes/scales and measured loss in native
GPU MLIR/LLVM. Seven exact gfx1201 numerical cases and three synthetic timing
rows are proved. Graph/Schedule/Tile integration and the public checked package
remain required; the original five-slice objective is not complete.


## Canonical converter JIT/runtime increment

[Canonical gfx1201 converter JIT](../rocm_ingest_runtime_20261005/README.md)
proves no-oracle frontend inference, complete Graph/Schedule/Tile/native package
handoff, common checked execution, cached image reuse and portable replay.
The static synthetic checkpoint envelope has separate cold/warm wall and
resident event timing. This closes ordinary converter JIT dispatch, while the
resident consumer edge, broader producers/AD and full five-slice objective remain
open. No global format or numerical strategy decision is made.

## Portable resident program increment

[Portable native converter/storage/matmul replay](../rocm_ingest_portable_20261005/README.md)
retains all three stages and proves replay without compiler access in a fresh
process. This closes the named serialization gap; ordinary composed JIT,
general frontend/AD, dynamic/layout and whole-model quality remain open.
The original five-slice objective remains active.

## Ordinary composed ingest JIT increment

[Named frontend conversion/storage/scaled-matmul chain](../rocm_ingest_jit_program_20261005/README.md)
proves static primal ordinary calls, cached images, reordered argument roles,
complete Graph verification and portable common-runtime replay on gfx1201.
The three stages retain native Schedule/Tile/Target/LLVM ancestry and owned
resident buffers. General producers, AD, dynamic/layout and model-quality
acceptance remain open; the original five-slice objective remains active.
Matched-source FP8/MXFP8/MXFP4 controls remain required.

## Native converter normalization increment

[Paired exact-power-of-two normalization](../rocm_ingest_reciprocal_20261005/README.md)
removes candidate FP64 divisions with bitwise-identical conversion and output.
Pinned M256 converter/combined resident graph improve by 2.171x/2.156x.
This is physical converter performance proof; it does not close general
frontend/AD, model-quality or the broader route/performance obligations.
FP8/MXFP8/MXFP4 remain independent gates and the full five-slice goal stays active.

## NVIDIA ordinary LHS frontend increment (2026-10-05)


## NVIDIA-LHS-JIT-PROGRAM-2026-10-05: ordinary frontend producer-to-matmul execution

Owner W1.1; sibling FRONTEND-IR-MEDIUM-1; sync NVIDIA-LHS-JIT-PROGRAM-2026-10-05.
Ordinary primal @jit verifies the complete typed Graph, retains caller semantics, and partitions static fp16/BF16 RMSNorm, LayerNorm or last-axis softmax LHS edges into existing native Graph/Schedule/Tile/Target/PTX packages. Native tile.view/fragments, private intermediate allocation, checked producer/consumer ABIs, one owned CUDA stream and completion govern execution. Bias, activation, residual and final output dtype remain native consumer contracts; serialized replay validates Graph, argument roles and image lineage before allocation.
The shared Schedule helper resolves unbound tracer bias/residual role markers from typed SSA operand positions, preserving explicit named bindings and existing target gates. No new GPU semantic source templates, dtype, operation, pass or diagnostic.

Parity validated on RTX 5070 / sm_120: 53 LHS/RHS tests cover three producers, fp16/BF16, padded inputs, native epilogues, reordered arguments, unchanged caller Graph, cached execution, portable replay and fresh-process replay without a compiler. Twenty-four matched cases record correctness before separate producer/consumer CUDA-event dispatch windows and cold/warm/replay host wall time. General producer graphs, dynamic frontend and composed AD remain follow-up required.
This proves the named static primal edge, not universal W1.1 or the complete five-slice objective. Evidence: ../nvidia_lhs_jit_program_20261005/README.md.

## Native cooperative norm candidate (2026-10-05)


## NVIDIA-COOPERATIVE-NORM-2026-10-05: native row schedule candidate

Owner W1.1; sibling FRONTEND-IR-MEDIUM-1; sync NVIDIA-COOPERATIVE-NORM-2026-10-05.
Native Schedule/Tile carries an explicit serial|cooperative_128 norm decision, included in the content hash. The SM120 cooperative materializer uses one 128-thread CTA per row, coalesced strided-column loads, shared partial reductions, centered LayerNorm variance and a scratch-reuse barrier. The pointer/scalar ABI is unchanged; host, resident and benchmark launch geometry recognizes the cooperative entry. Tile verification requires a string schedule and the owning sm_120 architecture. Packaging detects a selected NVIDIA tool that fails to materialize the cooperative schedule.
The explicit candidate preserves serial as the default. FP8, MXFP8 and MXFP4 remain independent correctness/quality/performance gates before any final strategy/default decision.

Exact RTX 5070 evidence compares serial/cooperative RMSNorm and LayerNorm across fp16/BF16/fp32, short/ragged/long rows, constant rows and large-offset centered variance. The matched native tools and CUDA bridge are fingerprinted; regular barrier/resource counts and separate resident event/checked host windows are recorded. Ordinary producer/JIT regressions remain required. General producer graphs, automatic strategy selection, dynamic frontend and composed AD remain open.
Evidence: ../nvidia_cooperative_norm_20261005/README.md. The broader five-slice objective remains active.


## NVIDIA-NORM-NATIVE-SELECTION-2026-10-05: native selection under evaluation

Owner W1.1; sibling FRONTEND-IR-MEDIUM-1; sync NVIDIA-NORM-NATIVE-SELECTION-2026-10-05.
C++ Graph-to-Schedule selects cooperative_128 for SM120 norm columns >= 256 and serial for shorter rows; explicit policies override selection. Python reads the native decision. Shared Schedule/Tile hashes and portable producer provenance retain the policy. No dtype, op, pass, diagnostic or quantized-format policy changes.

Parity validated in the bounded RTX 5070 envelope: 126 device tests and 36 composed paired cases, with identical consumer images. Shared tests: 424 passed, 17 skipped. Producer event time improves at K1024 but checked wall time does not consistently improve. Follow-up required: BF16 K4096 composed numerical tolerance fails for both serial and cooperative; retained attribution separates stored normalization rounding from consumer error. General producer/AD/dynamic and FP8/MXFP8/MXFP4 gates remain open.
Parity validated for unchanged serial gfx1201 behavior: 18 norm/epilogue tests pass after matching shared rebuild. Not applicable to ROCm physical scheduling: the new automatic decision is SM120-specific. gfx1151 proof and a separately designed HIP cooperative norm remain follow-up required.

[Evidence](../nvidia_norm_native_selection_20261005/README.md).


## NVIDIA-NORM-ACCURACY-2026-10-05: native FP32 numerical closure

Owner W1.1; sibling FRONTEND-IR-MEDIUM-1; sync NVIDIA-NORM-ACCURACY-2026-10-05.
The SM120 native Target materializer uses MLIR square-root/division and compensated serial FP32 sums, guarded for IEEE Inf/NaN propagation. Graph/Schedule/Tile snapshots and the pointer/scalar ABI are unchanged; no dtype/op/pass/diagnostic registration changes.

Parity validated on RTX 5070: 136 device tests, 48 current numerical arms passing the original tolerance, and 40 ordinary composed program cases. Three original frozen-compiler arms fail; the named BF16 K4096 gap is fixed for both schedules. Consumer PTX/Target IR is unchanged; source/tool/image fingerprints, barriers, resources and separate event/wall timing are retained. General producers/dynamic/AD and broader five-slice closure remain follow-up required. Attention JVP still has a Python GPU MLIR body constructor and needs native Schedule/Tile migration. FP8/MXFP8/MXFP4 remain independent gates.

[Evidence](../nvidia_norm_accuracy_20261005/README.md). Full five-slice objective remains active.

Native AD follow-on: [saved-LSE JVP native Schedule/Tile](../nvidia_jvp_native_schedule_20261005/README.md) retires the bounded SM120 Python GPU body constructor. Ten ordinary JIT cases and direct/automatic saved-state cases pass independent oracles, with separate dispatch/wall windows. General composed/dynamic/bias/dropout/value-only AD and sibling routes remain open.

Compiler overhead follow-on: [native pass-manager orchestration and exact tool identity reuse](../native_compile_orchestration_20261005/README.md) reduces fresh-package wall cost while retaining identical arena/image bytes and all native validation. General native persistent compilation, AD and backend closure remain open.

AD residual follow-on: [native saved-LSE value-only JVP](../nvidia_value_only_jvp_20261005/README.md) admits public active V alone, with native dependency pruning and 512-byte shared storage. RTX 5070 ordinary JIT and finite/Inf/NaN primal-V linearity proof is retained. General composed/dynamic/higher AD and sibling physical consumers remain open.


Attention-backward follow-on: [native requested-gradient activity](../nvidia_vjp_activity_20261005/README.md) moves isolated public gradient selection into native AD/Schedule/Tile/Target computation pruning. The complete ABI still allocates and zero-fills inactive gradient outputs; compact gradient allocation, composed/dynamic/higher AD and sibling consumers remain open. Exact RTX 5070 numerical and balanced dispatch evidence is retained; full five-slice closure remains unproven.

Native movement follow-on: [HIP host orchestration and context-owned staging](../rocm_native_movement_20261005/README.md) preserves compiled Graph/Schedule/Tile images while replacing per-launch Python HIP orchestration. Nine exact-device paged-KV/MoE rows pass on gfx1151/gfx1201 with zero warm pooled allocations and separate host/event windows. General layouts, asynchronous movement and retained production-route performance admission remain open; the full five-slice objective remains active.

Canonical movement follow-on: [native ROCm compiler spine and retained-route admission](../rocm_movement_admission_20261005/README.md) retains adjacent native Graph/Schedule/Tile/Target/backend artifacts and reconciles exact capability/manifest/execution rows. Nine gfx1151/gfx1201 canonical results are bit-exact and pass the retained-route 10% host non-regression gate. Public tensor eager/shape contracts, general layouts, asynchronous movement and resident-kernel timing/retirement remain open; full five-slice closure is unproven.

Public movement follow-on: [ordinary ROCm movement frontend and native JIT](../rocm_public_movement_20261005/README.md) proves tensor shapes, i32 index tracing, reordered native entry roles and cached artifact binding on owning gfx1151/gfx1201. MoE JIT improves retained calls; gfx1151 paged JIT still misses the retained-route 10% gate. General layouts/residency and full five-slice closure remain open.

Prepared-call follow-on: [native ROCm movement binding](../rocm_prepared_movement_20261005/README.md) removes warm Graph serialization, with native metadata/context/lifetime guards. All ten owning ordinary-JIT envelopes pass the retained-route 10% host gate with unchanged images and zero warm allocations. Reference MoE AD and DispatchPlan replay are corrected and mathematically tested; native movement AD, general layouts/residency and full five-slice closure remain open.

Compact backward follow-on: [native requested physical gradients](../nvidia_compact_gradients_20261005/README.md) preserves the complete logical AD product while narrowing the static SM120 physical output ABI. Forty owning RTX 5070 cases compare complete/packed/logical and 64/128-thread arms; checked host/resident execution, capture lifetime, repeated cotangents and actual allocation reduction pass. Preserved logical launch ranges recover the named long-row K-only gap; timings remain submission/event versus checked backward wall. General AD, sibling physical compact consumers, quantized gates and full five-slice closure remain open.

Frontend JVP follow-on: [native attention argument-order integration](../nvidia_jvp_argument_order_20261006/README.md) admits all six distinct Q/K/V frontend permutations through native AD/Schedule/Tile, private capture and physical tangent roles. Seventy-two RTX 5070 finite-difference cases pass while native image bytes remain invariant for equivalent semantic groups. General composed/dynamic/bias/higher AD and full five-slice closure remain open.

Portable JVP follow-on: [pinned native attention program](../nvidia_jvp_portable_20261006/README.md) validates complete image/manifest/role/generation contracts before capture. Seventy-two restored RTX 5070 cases and three compiler-forbidden fresh-process replays pass. Public native-JVP family/runtime dispatch and unused reverse-image retirement remain required; general AD and full five-slice closure remain open.

Public attention JVP increment: [NVIDIA-PUBLIC-ATTENTION-JVP-2026-10-06](../nvidia_public_attention_jvp_20261006/README.md) proves 72 public API cases and compiler-free parent replay. It does not close the larger five-slice objective or sibling physical consumers.

## Native prepared attention JVP follow-on

[Evidence](../nvidia_prepared_attention_jvp_20261006/README.md): 72 matched RTX5070 A/B cases and 72 public cases pass; four private C ABI exports move synchronous module/storage/role ownership into C++. The median matched wall ratio is 0.116622. Twelve device tests, 443 host tests and three fresh-process replays pass. No GPU algorithm changes or sibling physical proof. Unused reverse compilation, generalized native VJP and the complete five-slice program remain open.

## Forward-only native attention JVP follow-on

[Evidence](../nvidia_forward_attention_jvp_20261006/README.md): unused reverse executable compilation/serialization is retired from public JVP via a native forward checkpoint and pinned v2 product. Native reverse intermediate construction remains. Seventy-two public oracle cases, six compiler-free fresh-process replays, seven device tests including compact backward regression and 532 host tests pass. Four matched compile profiles record a median ratio of 0.828158; images are unchanged. Generic native VJP/composed/dynamic/higher AD, sibling physical consumers and full five-slice closure remain open.

## Captured native ROCm movement attribution

[Evidence](../rocm_captured_movement_20261006/README.md): five gfx1151 and three gfx1201 owning cases retain full native JIT lineage, exact symbols and bitwise resident replay including changed indices/source bits. Nine alternating 256-node graph/driver-loop trials separate resident events from host wall time. Small gains differ by architecture; full read is unchanged within 1%. This proves measured benchmark ownership and a named LP>P static read, not ordinary captured-runtime integration or general layout/AD closure. The full five-slice goal remains active.

## Native resident ROCm movement owner

[Evidence](../rocm_resident_owner_20261006/README.md): five gfx1151 and three
gfx1201 owning cases execute through public preparation and common runtime,
with native C++ storage/image/argument/completion ownership. Bit-exact rebinding,
retained host outputs, stale generations and bounds are proved. Warm common
download wall ratios are 0.4423–0.7905 against host-input JIT; repeat upload is
excluded. 394 host tests pass, 13 hardware/environment skips. GPU bodies are
unchanged. General resident producer/consumer integration, dynamic/layout/async
contracts, AD and full five-slice closure remain open.

## Native paged read softmax edge

[Evidence](../rocm_paged_softmax_edge_20261006/README.md): two typed native
frontend packages execute through a C++ owned intermediate on gfx1151/gfx1201.
Six standalone public softmax, six edge and eight movement regression rows
pass; max float64 oracle error below 4e-8. 512 shared tests pass, 13 skips.
Nine alternating common resident/download wall ratios are 0.2834–0.3698,
excluding repeat upload; GPU bodies are unchanged. Generic composed Graph
bufferization/lifetime planning, dynamic/layout/async contracts, AD and full
five-slice closure remain open.

## Public native attention VJP increment

[Evidence](../nvidia_public_attention_vjp_20261006/README.md): 88 owning RTX5070
public native_backward cases and three compiler-free fresh-process replays
pass. Native AD/Graph/Schedule/Tile checkpoint products preserve requested
gradient order, Q/K/V permutations and full/broadcast bias. Maximum independent
FP64 gradient error is 1.991e-8. Static canonical reverse-family integration is
proved; general composed/dynamic/dropout/higher AD, native prepared reverse
ownership and sibling physical consumers remain open. Warm public timings and
an instrumented host profile identify repeated portable-product restoration
as the next measured ownership boundary. The full five-slice goal stays open.

## Native prepared reverse ownership

[Evidence](../nvidia_prepared_attention_vjp_20261006/README.md): existing native
attention ownership now binds compact reverse gradients and full/broadcast
bias through four C ABI exports. 88 public and 88 matched RTX5070 cases pass;
native program pins are unchanged. Median common-runtime wall ratio 0.0700637
includes uploads/downloads and preserves separate forward/backward event
timing. Three fresh-process compiler-free replays and lifecycle/context/fork
guards pass. Generic composition/bufferization, dynamic/layout/resident/async/
dropout/higher AD and sibling physical consumers remain open. Historical
packets retain historical source fingerprints. Full five-slice closure remains
open; this is native host-runtime ownership progress, not GPU strategy promotion.


## Static score-bias forward AD integration

[Native score-bias JVP](../nvidia_bias_jvp_20261006/README.md) adds 36 resident
and 36 ordinary public/matched SM120 cases through native AD/Schedule/Tile/
LLVM/NVVM, explicit full/broadcast dbias and native prepared ownership. Three
compiler-free fresh-process replays and 38 device/guard regressions pass.
This resolves the named static bias-JVP gap; historical references above
retain their earlier evidence scopes. General composed/dynamic/layout/async/
higher AD and sibling physical consumers remain open. No full five-slice
closure or format/default promotion follows.


## 2026-10-06 native MXFP8 K64 staging increment

ROCM-FP8-BLOCKSCALE-1 explicit candidate: native Graph/Schedule/Tile/Target
K64 LDS slab with independent semantic K32 scale joins, whole-slab runtime guard,
and runtime-K image reuse. Final 40 checks and 354 regressions pass.
Narrow 128x64 candidate eliminates the first wide-panel scratch spill regression;
three-format matched numerical/event/host packet is retained at
../rocm_mxfp8_k64_20261006/README.md. Automatic strategy selection is unchanged.
Independent repetitions and wider shapes remain open. This increment does not
close the five-slice program or the M256 MXFP4/Radiance attribution gap.


## 2026-10-06 bounded multi-group MXFP8 selector

Independent forward/reverse packets retain K64 only for existing 128x64 LDS
panels at whole-slab K1024–2048; 128x128, long-K and global profiles stay K32.
125 exact-device/contract checks pass. The named multi-group staging gate is
proved in rocm_mxfp8_k64_20261006; general cache/layout/persistence and M256
MXFP4 cost attribution remain open. The full five-slice goal remains active.


## 2026-10-06 W1.1 frontend argument-lineage increment

Native canonical tensor recovery now preserves actual LHS/RHS input lineage
and fused bias/residual roles across frontend permutations, while requiring
complete tiling replay equivalence. Twelve RTX5070 rows prove registered
pipeline PTX parity and numerical/resident/host execution; 373 regression
checks pass. See ../nvidia_operand_lineage_20261006/README.md.
General producer/composed attention/AD, dynamic/layout and full five-slice
requirements remain open; this increment does not claim their closure.

## 2026-10-06 W1.1 ordinary public-call increment

Static SM120 FP16/BF16 matmul now enters the canonical checked native package
on ordinary @jit calls. Compact C/F RHS, fused bias/ReLU/residual and final
f16/f32 output are proved through native Schedule/Tile and explicit row-RHS
ABI variants. Twelve owning-device tests, 591 affected regressions (53 skipped),
and scoped public/resident timings are recorded in
../nvidia_operand_lineage_20261006/README.md. Python warm-call overhead remains;
prepared native dispatch is the next performance gate. Full five-slice and
general composed/dynamic/AD closure remain open.

## 2026-10-06 W1.1 native prepared matmul ownership

[Native prepared ownership](../nvidia_prepared_matmul_20261006/README.md)
moves static SM120 host binding and synchronous scratch/module/context lifetime
below Python. 75 exact-device tests, 651 regressions (66 skipped) and four
identical-image A/B packets prove this envelope. Warm calls reuse eight sealed
shape/dtype/layout handles without frontend tracing or portable restoration;
scratch is shared only through a synchronous native lease in one verified
context. General column-major A, composed/dynamic/AD, resident/async and sibling
owners remain open; the full five-slice objective is not complete.

## 2026-10-06 W1.1 producer RHS layout integration

Named static RMSNorm/LayerNorm/softmax -> matmul preserves compact C/F RHS
storage through ordinary JIT, native Schedule/Tile, checked ABI, resident
execution and portable replay. 70 RTX5070 device tests cover storage, fused
epilogue/output, cached layout switches and pre-allocation drift refusal, including four compiler-disabled fresh replays.
See ../nvidia_lhs_rhs_layout_20261006/README.md for separate producer/consumer
event and public-call timings. General producer/AD, dynamic row-RHS, A
layouts and asynchronous/native resident ownership remain open.
The full five-slice objective remains active.

## 2026-10-06 W1.1 native two-package ownership

Verified static norm/softmax -> matmul keeps its native Schedule/Tile/LLVM
images while C++ owns modules, one stream, pinned/device staging and private
intermediate lifetime. Large first-frame ordering and failed-consumer
retirement have exact-device proof. 921 checks pass (67 skipped), including
182 RTX5070 cases; eight matched packets show about 89% lower warm public
overhead. See ../nvidia_prepared_lhs_20261006/README.md.
General composition/AD, dynamic/resident/asynchronous and common portable
native ownership remain open. The full five-slice objective remains active.

## 2026-10-06 W1.1 portable native ownership

The named static FP16/BF16 norm/softmax -> matmul portable artifact now reuses
bounded native two-module owners by admitted semantic digest and unique live
CUDA context identity. Parent seals/ABI, two-context reuse, concurrency,
eviction/clear/rebind and pre-lock fork refusal have numerical/lifetime proof.
498 focused checks pass, including 142 RTX5070 device cases. Eight matched
24-case packets retain identical native images and show median warm replay
wall ratios of 0.163194–0.168429, approximately 83–84% lower overhead.
See ../nvidia_portable_lhs_owner_20261006/README.md.
This supersedes the common portable ownership gap only for this static
host-array envelope. General composition/AD, dynamic/resident/asynchronous,
wider formats and sibling owners remain open. Full five-slice closure remains
unproven and the objective remains active.

## 2026-10-06 W1.1 bounded frontend tensor programs

Explicit traced compilation and portable replay connect all seven bounded
M/N/K axis subsets to native normalization/softmax -> strided matmul packages.
FP16/BF16, fused epilogues, complete/ragged/single-element active extents,
caller Graph preservation and capacity guards have owning RTX5070 evidence.
See ../nvidia_dynamic_lhs_frontend_20261006/README.md for validation and separate
compile/replay/producer/consumer timing. Ordinary JIT shape-cache coalescing,
dynamic row-RHS/native prepared owners, general composition/AD and sibling
consumers remain open. Full five-slice completion remains unproven.

## 2026-10-06 W1.1 bounded native ownership

The named bounded normalization/softmax -> matmul envelope now owns native
modules, stream and intermediate/pinned/device staging across active shapes.
688 focused checks pass, including 332 RTX5070 cases. Private parameter ABI,
immutable capacities, no-growth/concurrent/failure retirement and portable
context reuse are proved. See ../nvidia_dynamic_lhs_owner_20261006/README.md
for matched wall/dispatch timing. Ordinary JIT bounded selection, dynamic
row-RHS, general composition/AD, asynchronous/resident and sibling consumers
remain open. Full five-slice completion remains unproven.


## 2026-10-06 ordinary bounded SM120 tensor JIT

Ordinary straight-line normalization/softmax -> matmul now reuses verified
bounded programs and native ownership across active shapes. 547 checks pass,
including 211 RTX5070 device cases. Four matched 48-case packets retain
identical images and warm wall ratios 0.0765821/0.0761356. See
benchmarks/baselines/nvidia_bounded_lhs_jit_20261006/README.md.
General composition/control-flow/AD, dynamic row-RHS, resident/asynchronous
execution and sibling physical consumers remain open. Full five-slice closure
remains unproven; the objective stays active.


## 2026-10-06 bounded SM120 row-major RHS integration

Native dynamic composed row indices, typed B gathers and explicit strided
row ABI now execute through ordinary bounded JIT/prepared/portable routes.
781 expanded checks pass (406 device cases, four unrelated skips); 28
frontend checks pass. Eight matched 48-case packets pass. Row consumer
event time is 9–15% higher than column storage in this envelope; column
packing remains the bounded default. See
benchmarks/baselines/nvidia_dynamic_row_lhs_20261006/README.md.
General composition/bufferization/control-flow/AD/resident/async, quantized
strategy gates and sibling physical consumers remain open. Full five-slice
completion remains unproven; the objective remains active.
