---
audit_role: plan
plan_state: landing
owner: NVIDIA backend
target: nvidia_sm120
last_updated: 2026-10-08
---



## ROCM-LINKED-TOOL-IDENTITY-2026-10-07

Owner FRONTEND-IR-MEDIUM-1 / E2E-REAL-6.
Shared contract: bounded ROCm compiler metadata reuse and image-cache identity
now include loaded ELF dependencies and loader search-path changes. Static
ELF and non-ELF executable content semantics are preserved. Runtime ABI,
Graph/Schedule/Tile semantics and physical schedules are unchanged.
Focused host validation passes 372 checks with four hardware skips.
gfx1201 owning-device validation and existing RMSNorm package A/B receipts
are recorded in benchmarks/baselines/rocm_linked_tool_identity_pr_20261007/.
This is a cache-contract slice; wider five-slice compiler closure remains open.

Not applicable: CUDA image identity and SM120 schedules are unchanged.

## NATIVE-JVP-SOURCE-CONSTRAINTS-2026-10-07

Owner FRONTEND-IR-MEDIUM-1 / AD-RESIDUAL-EVAL-1.
Shared public runtime contract: native_jvp checks source shape constraints
before tracing, compiling or preparing a backend. The existing solver and
constraint error semantics are reused. Positional/keyword x86, ROCm and
SM120 selectors have host-free early-rejection coverage.
nvidia assessment: shared source-bound parity is validated; numerical/device
execution is not applicable to this admission-only change.
No image, kernel, dtype, operation or ABI changes are introduced.

## Current actions — five-slice integration

Current as of 2026-10-08. Required route: frontend -> typed Graph MLIR -> verified AD/optimization -> Schedule -> Tile -> Target -> native image and checked ABI.

W1.1: arbitrary producer composition, accumulator lineage, wider resident layouts and asynchronous lifetime. E2E-REAL-6: dynamic/nested/composed attention AD, asynchronous residual ownership and broader quantized envelopes. Existing static saved-LSE/native backward and twelve canonical producer rows have owning RTX 5070 proof.

Named gfx1201 leading/scalar primal, scale JVP/VJP, transposed A and partial K32 groups are proved. Actual-image packaging and bounded version-query reuse are proved in recorded envelopes; they are not pending builds.

General dynamic/nonleading/composed maps, other scale widths/layouts, storage derivatives and generic closure remain open. Delivery requires reviewable PRs, final generated-doc/Graphify gates and a fresh full-suite result. The last full suite still has two generic batching/transpose closure failures.

Evidence ledger: ../../compiler/FIVE_SLICE_STATUS_20261007.md. Superseded active notes: archive/compiler_slice_integration_checkpoints_20261007.md.



## Composed scaled-product public JVP — 2026-10-08

Owner FRONTEND-IR-MEDIUM-1 / AD-RESIDUAL-EVAL-1 / E2E-REAL-6.
Sync GFX1201-COMPOSED-SCALED-JVP-2026-10-08.

Two frontend scaled products plus add now execute through public native_jvp
and the existing verified native AD/Schedule/Tile/HIP program. The full Graph
participates in cache identity. Five focused host tests, 357 contract/registry
gates and 29 gfx1201 owning tests pass, including selected/reordered scale
seeds, changed-input warm reuse and retained outputs. Two correctness-gated
profiles record separate ten-member native event medians 0.057127/0.087523 ms
and public host medians 2.748519/2.978129 ms; no speedup claim.
Composed reverse AD, generic batching/transpose, dynamic/nonleading/storage
derivatives and focused delivery remain open.
Evidence: benchmarks/baselines/gfx1201_composed_scaled_jvp_20261008/README.md.

Follow-up required for nvidia-owned composed scale-JVP execution. gfx1201 HIP images and timing do not establish sibling physical parity; target admission is unchanged.


## Composed native scale reverse integration — 2026-10-08

Owner FRONTEND-IR-MEDIUM-1 / AD-RESIDUAL-EVAL-1 / E2E-REAL-6.
Sync GFX1201-COMPOSED-SCALED-VJP-2026-10-08.

Role-aware native export and checked ABI bind the complete frontend product/sum
Graph to scale adjoints. Shared LHS scales preserve both contributions through
native reduction and sum members, including private/returned lifetimes.
Matching compiler build, 419 host gates, 86 gfx1201 owning checks, eight wave
owning checks and seven recorder guard tests pass. Four serial and four wave
profiles pass independent float64 numerics; native-program and public host
timings are recorded separately. Long native medians are 7.67–7.70 ms serial
and 0.264–0.271 ms wave. This is characterization, not counterbalanced A/B or
default promotion; serial remains the default.
General batching/transpose, dynamic/nonleading/storage AD, wider format/shape
performance, model quality and focused publication remain open.
Evidence: benchmarks/baselines/gfx1201_composed_scaled_vjp_20261008/README.md.

Follow-up required for nvidia-owned composed scale-adjoint execution. Shared AD export, program validation and frontend routing are assessed; gfx1201 reductions/sums and HIP timing provide no sibling physical proof.

## Linked compiler-library identity — validated

Owner FRONTEND-IR-MEDIUM-1 / E2E-REAL-6.
Sync ROCM-LINKED-TOOL-IDENTITY-2026-10-07.
ELF dependency content now participates in compiler image-cache identity;
version metadata invalidates on dependency/search-path changes. 95 WSL host
checks, 307 shared drift gates and 40 gfx1201 scalar numerical cases pass.
All 16 package A/B profiles retain matching fingerprints and 7-to-5
subprocess reuse. Actual warm compiler identity binds eight loaded libraries
without subprocesses. These are package metadata results, not kernel gains.
Evidence: benchmarks/baselines/rocm_linked_tool_identity_20261007/README.md.

Not applicable to CUDA image identity; no SM120 runtime or schedule changes.


## Focused cache delivery — PR #894 ready

ROCM-LINKED-TOOL-IDENTITY-2026-10-07 is published separately at
https://github.com/gstoner/tessera/pull/894, branch
codex/rocm-linked-tool-identity, head a73212f5820850a3e9842374f6b2fecfc06394ca.
Live status checked 2026-10-08: open, ready, mergeable; all required CI
checks are green. The matching-source x86-enabled isolated full WSL unit
lane passes 20,419 tests, with 7,387 skips and 874 deselections.
This focused PR publishes compiler cache identity work, not the accumulated
native five-slice branch. Its green result does not establish aggregate
batching/transpose closure or later generator/device proof.
Remaining native five-slice delivery is open.

## NVFP4 native whole-Graph public integration — owning proof

Owner ROCM-NVFP4-INGEST-1 / FRONTEND-IR-MEDIUM-1 / E2E-REAL-6.
Sync NVFP4-NATIVE-PROGRAM-2026-10-08.
Public packaging now consumes compiler-authored actual member Graphs and
portable v2 replay checks eleven buffer capacities/read-write lifetimes.
No Python Graph constructors or legacy author helpers are used in the proved
public route. Final gfx1201 lane: 68 mixed checks; six JIT/portable benchmark
profiles pass independent numerics with separate stage device-event, graph
dispatch and end-to-end timing. Shared native lane: 97 passes/three skips;
RTX 5070 producer/saved-LSE sibling lane: 72 passes.
Model-quality acceptance, broader dynamic/layout/storage-AD envelopes,
performance optimization and focused native PR delivery remain open.
Evidence: benchmarks/baselines/rocm_nvfp4_whole_graph_partition_20261007/README.md.

Owning RTX 5070 shared-export producer/saved-LSE parity validated; no NVIDIA ingest promotion.

## Dated synchronization receipts

Sections below record their source snapshots. Earlier pending statements may be superseded by later proof and the current actions above.

## INDEPENDENT-SCALED-PRIMAL-2026-10-07

Owner FRONTEND-IR-MEDIUM-1 / AD-RESIDUAL-EVAL-1 / ROCM-FP8-BLOCKSCALE-1.
Synchronization key INDEPENDENT-SCALE-BATCH-2026-10-07.
Native Schedule identity, Tile/Target metadata, independent GPU plane views
and checked member ABI now retain all four operand prefixes. Static plane
types remain in image identity. 465 native/registry regressions and 82 shared
gates pass. 69 gfx1201 primal/scale-JVP programs pass independent numerics,
changed-input replay, stale generation refusal and separate event/host timing.
The final policy guard passes 74 native tests and all 69 corrected owning
replays. Native event timings use the ABI-provided per-invocation average.
Public independent primal/JVP, transposed A, partial K scale groups, dynamic/
nonleading/composed/storage AD, larger regimes and generic/delivery closure
remain open. No speedup or sibling scale-execution claim is made.
Existing owning RTX 5070 NVFP4 JIT/map regression parity passes 14 tests with two rank-two/no-batch skips. Independent packed-scale maps and scale AD remain follow-up required.
Evidence: benchmarks/baselines/rocm_independent_scaled_primal_20261007/README.md.


## PUBLIC-INDEPENDENT-SCALE-REVERSE-2026-10-07

Owner FRONTEND-IR-MEDIUM-1 / AD-RESIDUAL-EVAL-1.
Synchronization key INDEPENDENT-SCALE-BATCH-2026-10-07.
Public leading maps now project independent matrix/scale prefixes into native
Graph paired AD, Schedule/Tile structured reductions and checked gfx1201 HIP
programs. 584 integrated host gates and 261 owning gfx1201 cases pass.
All 180 public/native baseline rows pass changed-input compiler-free replay
and independent gradient comparison (maximum error 9.1122e-8). Public wall time,
prepared host time and native launch windows remain separate; no speedup claim.
Independent-map primal/JVP, dynamic/nonleading/composed and storage AD, wider
regimes, generic batching/transpose/full-unit closure and delivery remain open.
SM120 independent packed-scale maps and scale AD are follow-up required; existing NVFP4 regression evidence covers its admitted profile.
Evidence: benchmarks/baselines/rocm_public_independent_scale_reverse_20261007/README.md.


## INDEPENDENT-SCALE-BATCH-2026-10-07

Owner FRONTEND-IR-MEDIUM-1 / AD-RESIDUAL-EVAL-1.
Independent numerical broadcast conformance passes 417 host checks.
Native Graph verification and scale transpose now retain each operand's
right-aligned prefix, including shared/singleton scale reduction axes.
131 candidate native export/Schedule/Tile/image tests and 382 shared registry/
SM120 owning regressions pass. All 68 gfx1201 serialized reverse packages
pass independent numerics, changed-input replay and separate native/host
timings. Matching aggregate CMake rebuild, 589 focused gates and a fresh
68-case gfx1201 canonical rerun pass; all final source/image identities bind.
Existing owning SM120 NVFP4 regressions pass; independent packed-scale
batch addressing and scale AD remain follow-up required.
Public static leading-map reverse integration is now proved in the newer
public packet above. Independent-map primal/JVP, dynamic/nonleading/composed
integration, wider performance, generic closure and publication remain open.
Evidence: benchmarks/baselines/scaled_independent_batch_native_reverse_20261007/README.md.


## NATIVE-PADDED-RHS-2026-10-07

Owner W1.1 / E2E-REAL-6.
Native resident submission now projects RHS pitch into the existing dynamic
strided consumer ABI and validates the full physical allocation span.
The matching aggregate CMake runtime passes 581 focused tests, including
padded/compact resident ownership, changed-value/shape replay, and registry
gates. All 48 canonical same-image A/B rows pass pre/post independent
numerics; median control/native wall ratio is 2.440, with a worst 3.62%
regression. No kernel-only or selector
claim. SM120 padded row/column RHS and offset views have owning device proof.
Padded source/output/residual, general AD/composition, asynchronous lifetime,
generic closure and delivery remain open. Generated documents are in sync
and the post-integration Graphify refresh completed successfully.
Evidence: benchmarks/baselines/nvidia_padded_resident_owner_20261007/README.md.



## NATIVE-RESIDENT-NVIDIA-2026-10-07

Owner W1.1 / E2E-REAL-6.
Compiler-owned producer/matmul images now share native C++ submission for
host-staged and resident execution. The resident ABI checks device allocation
capacity, active shape/pitch/type, stream/context and output/edge aliasing;
synchronous completion retires live uses before caller buffers may be released.
Immutable receipt hashes move to preparation; each invocation retains the
semantic package snapshot check and returns independent receipt copies.
57 new owning RTX 5070 cases and 137 existing native-owner regressions pass.
48 alternating same-image timing rows pass pre/post independent numerics.
Median control/native resident wall ratio is 2.518; one row is 2.4% slower.
The measurements include ABI/submission/completion and concurrent full-unit
activity, not isolated-kernel gains. Initial regressions and profile are retained.
SM120 has exact-device candidate numerics, native lifetime/context checks and paired same-image timing. General composition, AD, padded pitches and asynchronous ownership remain open.
Generic scaled-matmul batching/transpose closure and delivery remain open.
Evidence: benchmarks/baselines/nvidia_native_resident_owner_20261007/README.md.



## DEEP-LEADING-MAPS-2026-10-07

Owner FRONTEND-IR-MEDIUM-1 / AD-RESIDUAL-EVAL-1.
Static typed gfx1201 map prefixes retain arbitrary positive matching depth;
minimum-rank capability and native adjoint construction are aligned. Reverse
source bounds precede capture on all targets. 15 host ordering controls,
474 shared registry/policy checks and 89 native transpose regressions pass.
SM120: 72 attention/VJP and 11 owning NVFP4 map regressions pass; nested packed maps remain independently gated.
18 paired public/native timing rows pass independent numerics; serial remains
default. Generic/dynamic/mixed/nonleading/storage-AD and delivery remain open.
Evidence: benchmarks/baselines/rocm_deep_leading_map_20261007/README.md.


## NVFP4-SHORT-M-2026-10-07

Owner ROCM-NVFP4-INGEST-1 / E2E-REAL-6.
Compiler-owned packed folded admission now supports positive M through the
Graph/Schedule/Tile/Target/native image and checked native owner contract.
24 host package and 26 owning gfx1201 short/ragged cases pass; 12 timing
shapes pass pre/post numerical checks. Existing device gates pass 35 cases
and the fresh-process replay passes after an explicit environment repair.
Not applicable to the SM120 physical schedule; existing NVIDIA NVFP4 batching proof is independent.
General layout/AD/model-quality/performance and full-unit/publication closure
remain open. Evidence: benchmarks/baselines/rocm_nvfp4_short_m_20261007/README.md.


## COMPILER-INTEGRATION-DRIFT-2026-10-07

Owner FRONTEND-IR-MEDIUM-1 / AD-RESIDUAL-EVAL-1.
Native adjoint and two structured-reduction inventory rows now match source;
family-specific evidence prevents architecture alias inheritance. 780 focused
tests pass. Two generic batching/transpose full-unit failures remain open.
SM120 existing map proof is unchanged; scale-transpose physical parity remains follow-up required.
Evidence: benchmarks/baselines/compiler_drift_repair_20261007/README.md.


## SCALED-REVERSE-RAGGED-2026-10-07

Owner FRONTEND-IR-MEDIUM-1 / AD-RESIDUAL-EVAL-1.
Logical FP32 scale-adjoint admission now preserves ragged final K groups
without broadening primal WMMA scheduling. 73 frontend/contract tests and
361 shared/native gates pass; missing tail scales reject before capture.
96 gfx1201 ragged/long public A/B rows pass numerics and warm replay with
median paired wall-time ratios 4.043/10.321. Serial remains default.
Parity validated for 11 existing RTX 5070 NVFP4 map cases; native scale reverse remains follow-up required.
General AD, dynamic/mixed maps, generic closure and publication remain open.
Evidence: benchmarks/baselines/rocm_public_scaled_vjp_regimes_20261007/README.md.


## SCALED-TRANSPOSE-COMPENSATED-2026-10-07

Public experimental schedule selection now binds cache identity and receipts.
Both schedules pass 72 gfx1201 public tests; 48 alternating public timing rows
pass numerics and compiler-free warm reuse (median paired ratio 3.934).
362 focused shared/native gates pass. These are named gfx1201 results, not
sibling physical proof or generic AD closure. Serial remains default.

Owner FRONTEND-IR-MEDIUM-1 / AD-RESIDUAL-EVAL-1.
Native compensated FP32 outer reductions repair six long-shape failures per
serial/wave arm without relaxing bounds. 72 tiny/ragged/long gfx1201 paired
cases pass; serial remains default. Native lowering gates pass 23 tests and
shared metadata/diagnostic/audit gates pass 307. Wider public regimes,
selector policy and generic/full-unit/publication closure remain open.
Follow-up required for SM120 native scale-adjoint images/ABI and owning-device numerics.
Evidence: benchmarks/baselines/rocm_scaled_vjp_compensated_20261007/README.md.


## SCALED-TRANSPOSE-WAVE-2026-10-07

Owner FRONTEND-IR-MEDIUM-1 / AD-RESIDUAL-EVAL-1.
Native Schedule/Tile carries an opt-in 32-lane additive outer reduction,
retaining K-group arithmetic and binding algorithm/width to image ABI.
372 shared/native gates and 24 paired gfx1201 numerical/timing rows pass.
Serial remains default; wider/public/selector closure remains open.
Follow-up required for nvidia native scale-adjoint schedule/ABI and owning-device proof; its physical schedule is unchanged.
Evidence: benchmarks/baselines/rocm_scaled_vjp_wave_20261007/README.md.

## SCALED-PUBLIC-TRANSPOSE-2026-10-07

Owner FRONTEND-IR-MEDIUM-1 / AD-RESIDUAL-EVAL-1.
Public scale-only reverse @jit now uses compiler-owned native packages.
72 gfx1201 scalar/one-map/two-map cases pass numerics and compiler-free reuse.
397 shared drift gates pass; 48 public timing rows retain separate wall-time
labels. Dynamic/composed/storage AD and serial-reduction optimization remain open.
11 existing RTX 5070 NVFP4 map regressions pass. NVIDIA native scale reverse remains follow-up required.
Evidence: benchmarks/baselines/rocm_public_scaled_vjp_20261007/README.md.

## SCALED-TRANSPOSE-PROGRAM-2026-10-07

Owner FRONTEND-IR-MEDIUM-1 / AD-RESIDUAL-EVAL-1.
Native paired AD outlines actual scale-adjoint regions with faithful captures,
requested output order, frontend permutations and SSA buffer lifetimes.
55 native export/package tests and 75 native AD fixtures pass.
Existing public NVFP4 maps pass 11 owning RTX 5070 cases with the matching compiler. Native NVIDIA transpose members and physical proof remain follow-up required.
Evidence: benchmarks/baselines/scaled_transpose_program_export_20261007/README.md.


Ongoing native reduction integration: sealed Tile/ROCm region carriers now
provide MLIR symbol-table ownership, and GPU ABI verification uses typed
function properties after serialization. Matching image/package validation
and gfx1201 physical reverse proof are pending; this supplies no sibling
execution evidence. See the five-slice ledger's Schedule/Tile integration entry.

The subsequent native carrier repair passes 15 lowering/image checks and
55 export regressions. 24 gfx1201 reverse packages pass numerical and lifetime
checks with separate native/host timing. Public JIT reverse remains open.
Follow-up required for nvidia native transpose images/ABI and owning-device proof.
Evidence: benchmarks/baselines/rocm_native_scaled_vjp_20261007/README.md.

## SCALED-PRODUCT-TRANSPOSE-2026-10-07

Owner FRONTEND-IR-MEDIUM-1 / AD-RESIDUAL-EVAL-1.
Native structured scale-adjoint construction preserves original K/N groups
and reduces shared batch axes. Twelve independent finite-difference cases
and 12 owning gfx1201 native-JVP duality cases establish the numerical contract.
Follow-up required for native scale transpose execution on SM120; no CUDA image/ABI or owning-device reverse proof is supplied.
Evidence: benchmarks/baselines/scaled_product_transpose_foundation_20261007/README.md.


## ROCM-NESTED-TYPED-VMAP-2026-10-07

Owner FRONTEND-IR-MEDIUM-1 / AD-RESIDUAL-EVAL-1 / E2E-REAL-6.
Two matching public leading maps now descend through one compiler-owned typed
batch primal. Source bounds, distinct batch symbols, independent owners and
gated storage metadata survive projection; native FP32 scale-JVP also executes.
415 shared tests pass with 16 gated skips. Evidence:
benchmarks/baselines/rocm_nested_typed_vmap_20261007/README.md.
Existing single leading public SM120 NVFP4 maps pass 11 exact RTX 5070 cases. Nested typed gfx1201 map policy is not admitted for NVIDIA; physical follow-up required before promotion.


## ROCM-MULTIDIMENSIONAL-SCALED-BATCH-2026-10-07

Owner FRONTEND-IR-MEDIUM-1 / AD-RESIDUAL-EVAL-1 / E2E-REAL-6.
Native static typed two-axis batch prefixes survive Graph/Schedule/Tile/Target/
LLVM and checked lifetime/ABI projection. Equal products do not authorize
different prefix tuples. 381 focused and 48 target/ABI tests pass; native
fixtures pass 649 with 66 unsupported. Evidence:
benchmarks/baselines/rocm_multidimensional_scaled_batch_20261007/README.md.
Existing SM120 rank-two/rank-three NVFP4 shared-Schedule parity passes nine RTX 5070 cases. Rank-four typed FP8/MXFP8 admission is not promoted for NVIDIA; its physical follow-up remains open.


## ROCM-NATIVE-CHECKPOINT-2026-10-07

Owner ROCM-NVFP4-INGEST-1.
The real-checkpoint recorder now executes native Graph/Schedule/Tile/Target/
LLVM ingest before the native consumer. gfx1201 q_proj and merged gate/up
codes/exponents match the reference bitwise, independent f64 block statistics
pass before/after timing, and consumer decoded-weight outputs are exact.
359 focused host WSL tests pass with seven gated skips.
Evidence: benchmarks/baselines/rocm_native_checkpoint_20261007/README.md.
Not applicable to SM120 execution: this is a gfx1201 recorder integration. NVIDIA source NVFP4 semantics are retained; no NVIDIA image or schedule changes.


## COMPILER-UNIT-CONTRACT-REPAIR-2026-10-07

Owner FRONTEND-IR-MEDIUM-1 / AD-RESIDUAL-EVAL-1 / E2E-REAL-6.
Native scaled tangent forward/proof registration and named recorder consumers
are reconciled. ROCm optional metadata no longer precedes its intended family
refusal. 110 focused checks pass with 90 gated skips.
Shared derivative/proof and recorder metadata assessment complete. nvidia physical scaled AD remains a separate owning-device follow-up.
Generic batching/transpose closure and final full-suite/PR delivery remain open.
Evidence: benchmarks/baselines/compiler_unit_contract_repairs_20261007/README.md.


## ROCM-PUBLIC-MAPPED-JVP-2026-10-07

Owner FRONTEND-IR-MEDIUM-1 / AD-RESIDUAL-EVAL-1 / E2E-REAL-6.
Mapped owners retain admitted FP32 scale-JVP intent. Signature-cached eager
map/reference certificates remain diagnostic; native AD and packages own
execution. 72 gfx1201 primal/JVP tests, 503 shared gates and 12 timing rows pass.
Explicit planned-gated uint8 candidate storage is preserved without inferring
byte opt-in from arrays.
Public native JVP now checks source shape constraints before capture/compile.
44 focused and 72 shared native-JVP/attention checks pass (16 gated skips);
59 owning gfx1201 JVP cases pass. Sibling device constraint parity requires
its own follow-up evidence.
Shared frontend owner/intent assessment complete; existing NVFP4 map regressions pass. SM120 typed FP8 AD requires independent owning-device implementation/proof.
General dynamic/nested/composed AD, transpose closure and PR delivery remain open.
Evidence: benchmarks/baselines/rocm_public_mapped_jvp_20261007/README.md.


## ROCM-PUBLIC-TYPED-VMAP-2026-10-07

Owner FRONTEND-IR-MEDIUM-1 / E2E-REAL-6 / ROCM-FP8-BLOCKSCALE-1.
Shared frontend map intent projects direct typed scalar products into native
leading batches; scalar owners remain independent. 25 host frontend tests and
36 gfx1201 numerical/warm-call tests and 36 timing rows pass, including ragged M=200 and wider LDS execution.
Existing SM120 named NVFP4 frontend regressions pass. Typed FP8 owning-device parity requires a separate follow-up; ROCm physical schedules do not transfer.
Dynamic/nested/composed AD, generic closure and aggregate PR delivery remain open.
Evidence: benchmarks/baselines/rocm_public_typed_vmap_20261007/README.md.

## ROCM-NATIVE-PLAN-BINDING-2026-10-07

Owner FRONTEND-IR-MEDIUM-1 / E2E-REAL-6 / ROCM-FP8-BLOCKSCALE-1.
Complete immutable manifests/packages key bounded checked decoding and readonly
native ABI bindings. Native capacities/lifetimes/context checks and execution
ownership remain in C++; mutable metadata and active owners stay independent.
68 gfx1201 default tests and 386 shared checks pass. Six image-identical paired
rows show 7-13% lower primal and 15-20% lower scale-JVP public host cost.
Not applicable to the SM120 runtime: it uses a separate owner/binding class. Shared contract assessment complete; no NVIDIA timing claim.
Dynamic/nested/composed AD, broader performance and aggregate closure remain open.
Evidence: benchmarks/baselines/rocm_native_plan_binding_20261007/README.md.

## ROCM-INDEPENDENT-SCALED-BATCH-2026-10-07

Owner FRONTEND-IR-MEDIUM-1 / E2E-REAL-6 / ROCM-FP8-BLOCKSCALE-1.
Shared contract: typed independent-RHS/shared-LHS Graph ranks, per-batch
Schedule rows, Tile/Target z-plane policy and native program capacities.
67 gfx1201 primal/JVP cases pass, including wider LDS numerics and changed
single-batch scales; 24 primal rows retain separate host/native event windows.
Shared semantic/package checks validated; SM120 NVFP4 batch policy remains distinct. Follow-up required for typed FP8 owning-device parity.
Dynamic/nested batching, transpose/composed AD, generic closure and PR delivery remain open.
Evidence: benchmarks/baselines/rocm_independent_scaled_batch_20261007/README.md.

## ROCM-BATCH-OFFSET-FOUNDATION-2026-10-07

Owner FRONTEND-IR-MEDIUM-1 / E2E-REAL-6 / ROCM-FP8-BLOCKSCALE-1.
Native gfx1201 compiler fixture proves dynamic memref offsets survive
unbounded vector and ragged guarded scalar E4M3 fragment loads and stores
through Target-to-ROCDL. No shared semantic/ABI or selector changes.
Not applicable to this ROCm-only fixture; existing SM120 batch contracts are unchanged.
Independent-RHS/shared-LHS integration and owning numerical/timing proof remain open.
Evidence: benchmarks/baselines/rocm_batch_offset_foundation_20261007/README.md.

## ROCM-SHARED-SCALED-BATCH-2026-10-07

Owner FRONTEND-IR-MEDIUM-1 / E2E-REAL-6 / ROCM-FP8-BLOCKSCALE-1.
Shared contract: static typed E4M3 shared-RHS Graph ranks, native B*M Schedule
projection, pointer-backed Tile binding and native program policy/lifetime ABI.
541 shared and 222 target registry checks pass; one target case skips.
Shared Graph/program batch metadata and registry gates validated. The existing named NVFP4 route is distinct; no SM120 E4M3 shared-row execution or physical schedule is introduced. Follow-up required for owning typed FP8 parity.
Generic batching/linear transpose, broader programs and full-suite/PR closure remain open.
Evidence: benchmarks/baselines/rocm_shared_scaled_batch_20261007/README.md.


## E8M0-EAGER-PARITY-2026-10-07

Owner FRONTEND-IR-MEDIUM-1 / E2E-REAL-6.
Shared explicit E8M0 byte decoding now matches the declared [1,32] semantic
contract in the typed eager reference. Code 0 and NaN code 255 are preserved;
no production lowering, native ABI or selector changes. 474 shared gates pass.
Shared eager semantic parity validated by host-free numerical tests. Follow-up required for nvidia physical typed scaled consumers; gfx1201 proof does not establish sibling execution.
Generic batching/linear transpose and aggregate/full-suite closure remain open.
Evidence: benchmarks/baselines/rocm_e8m0_eager_parity_20261007/README.md.


## ROCM-PRIMAL-TRANSFER-EVICTION-2026-10-07

Owner FRONTEND-IR-MEDIUM-1 / E2E-REAL-6 / ROCM-FP8-BLOCKSCALE-1.
Native bounded idle LRU eviction waits for completion, preserves quarantine
and keeps active owners untouched. Automatic gfx1201 moderate-size pinned
staging is bound to cache identity; explicit overrides remain.
Not applicable to nvidia physical cache implementation: this change is native HIP idle ownership and gfx1201 staging policy. Shared lifetime/budget contracts assessed; sibling execution requires independent proof.
Public wall-clock improvement does not establish isolated kernel speedup.
Generic closure and aggregate publication remain open.
Evidence: benchmarks/baselines/rocm_primal_transfer_attribution_20261007/README.md.



## PRIMAL-PROFILE-POLICY-2026-10-07

Owner FRONTEND-IR-MEDIUM-1 / E2E-REAL-6 / ROCM-FP8-BLOCKSCALE-1.
Not applicable to nvidia physical policy: Target selection is gfx1201-specific. Shared program-kind/ABI metadata assessed; sibling exact-device evidence remains independent.
Evidence: benchmarks/baselines/rocm_primal_profile_policy_20261007/README.md.


## PRIMAL-IMAGE-PROJECTION-2026-10-07

Owner FRONTEND-IR-MEDIUM-1 / E2E-REAL-6 / ROCM-FP8-BLOCKSCALE-1.
Not applicable to nvidia physical image identity: projection and recovered launch ABI are ROCm-specific. Shared SSA/lifetime contracts assessed; sibling execution proof remains independent.
Evidence: benchmarks/baselines/rocm_primal_image_projection_20261007/README.md.


## NATIVE-PRIMAL-OWNER-2026-10-07

Owner FRONTEND-IR-MEDIUM-1 / E2E-REAL-6 / ROCM-FP8-BLOCKSCALE-1.
Shared SSA primal/paired ownership contracts assessed; physical execution follow-up required on nvidia. HIP proof does not establish sibling parity.
Evidence: benchmarks/baselines/rocm_native_primal_owner_20261007/README.md.


## MXFP8-PUBLIC-PRIMAL-2026-10-07

Owner FRONTEND-IR-MEDIUM-1 / E2E-REAL-6 / ROCM-FP8-BLOCKSCALE-1.
Shared explicit gated Tensor/Graph metadata assessed; physical parity follow-up required on nvidia. HIP evidence does not establish sibling execution.
Evidence: benchmarks/baselines/rocm_mxfp8_public_primal_20261007/README.md.


## ROCM-TYPED-SCALED-PRIMAL-2026-10-07

Owner FRONTEND-IR-MEDIUM-1 / E2E-REAL-6 / ROCM-FP8-BLOCKSCALE-1.
Not applicable to nvidia physical admission: primal hook selects gfx1201 FP8 images only. Shared original-Graph and SSA binding contracts assessed; sibling execution needs independent proof.
Evidence: benchmarks/baselines/rocm_typed_scaled_primal_20261007/README.md.

## THREE-FORMAT-NATIVE-STAGING-2026-10-07

Owner FRONTEND-IR-MEDIUM-1 / E2E-REAL-6 / ROCM-FP8-BLOCKSCALE-1.
Not applicable to nvidia physical proof: native HIP staging uses existing gfx1201 images; shared memory/ABI lifetime assessed, sibling execution remains independent.
Evidence: benchmarks/baselines/scaled_program_three_format_staging_20261007/README.md.

## SCALED-MATMUL-NATIVE-PINNED-2026-10-07

Owner FRONTEND-IR-MEDIUM-1 / E2E-REAL-6 / ROCM-FP8-BLOCKSCALE-1.
Shared runtime contract: completion-before-staging-reuse/free, bounded memory accounting and transfer-mode cache identity. Not applicable to nvidia physical execution: implementation is native HIP staging. Shared lifetime and memory-budget contracts assessed; owning sibling staging proof remains independent.
Evidence: benchmarks/baselines/scaled_matmul_native_pinned_20261007/README.md.

## SCALED-MATMUL-PUBLIC-ORIENTATION-2026-10-07

Owner FRONTEND-IR-MEDIUM-1 / E2E-REAL-6 / ROCM-FP8-BLOCKSCALE-1.
Shared Graph transpose and scale-JVP contracts assessed. Follow-up required: HIP RHS-orientation/JVP proof does not establish nvidia scaled-program parity.
Evidence: benchmarks/baselines/scaled_matmul_public_orientation_20261007/README.md.

## SCALED-MATMUL-NATIVE-REUSE-2026-10-07

Owner FRONTEND-IR-MEDIUM-1 / E2E-REAL-6.
Shared runtime contract: bounded idle ownership keyed by exact images and ABI, fresh handles, preserved generations and process/context/device identity; poison quarantine and explicit cache cleanup.
Not applicable to nvidia physical execution: reuse implementation is HIP-specific. Shared image/ABI ownership assessed; native sibling owner reuse requires independent follow-up.
Evidence: benchmarks/baselines/scaled_matmul_native_reuse_20261007/README.md.

## SCALED-MATMUL-PUBLIC-JVP-2026-10-07

Owner FRONTEND-IR-MEDIUM-1 / E2E-REAL-6.
Shared frontend contract: explicit differential reference evaluation, typed FP8 capture and native paired program packaging.
Follow-up required: gfx1201 HIP proof does not establish nvidia physical scaled-program execution.
Evidence: benchmarks/baselines/scaled_matmul_public_jvp_20261007/README.md.

## SCALED-MATMUL-NATIVE-PACKAGE-2026-10-07

Owner FRONTEND-IR-MEDIUM-1 / E2E-REAL-6.
Native compiler exports machine-readable program/member ABI bound to actual
Graph/SSA witnesses, typed buffers/lifetimes and backend launch geometry.
Package/registry corruption gates pass 305 tests.
Native common SSA manifest preserves Graph witnesses and ownership. HIP member/package replay does not establish SM120 scaled-program physical parity; CUDA projection/runtime follow-up required.
Evidence: benchmarks/baselines/scaled_matmul_native_package_20261007/README.md.


## SCALED-MATMUL-NATIVE-OWNER-2026-10-07

Owner FRONTEND-IR-MEDIUM-1 / E2E-REAL-6.
Native HIP C ABI checks SSA prefix bindings, exact lifetimes, buffer sizes,
private scratch and returned generation ownership. Images and allocations
remain owned through stream completion; repeated sequence launch runs in C++.
Shared SSA lifetimes/ownership assessed. This HIP owner and gfx1201 numerical/timing packet do not provide SM120 paired scaled-program integration or physical proof; CUDA projection/ownership follow-up remains required.
Evidence: benchmarks/baselines/scaled_matmul_native_owner_20261007/README.md.


## SCALED-MATMUL-NATIVE-MEMBERS-2026-10-07

Owner FRONTEND-IR-MEDIUM-1 / E2E-REAL-6.
Shared contract: native selected-member projection retains the full SSA
program witness, typed buffer roles and semantic/argument attributes.
Combined native lanes pass 644, with 66 unsupported; focused AD/registry
tests pass 347, with 16 skips. The three products and sum compile as
gfx1201 native HSACO members, without Python IR reconstruction.
NVIDIA follow-up: common native export/selection checks pass; this FP8 gfx1201 image receipt does not establish SM120 scaled-program lowering or physical proof.
Generic batching/transpose and full-unit closure remain open.
Evidence: benchmarks/baselines/scaled_matmul_native_members_20261007/README.md.

## SCALED-MATMUL-NATIVE-PROGRAM-2026-10-07

Owner FRONTEND-IR-MEDIUM-1 / E2E-REAL-6.
Shared contract: native paired SSA export outlines actual products/sum and
records typed buffer IDs, logical sizes, argument metadata and read/write
lifetimes. Native forward AD preserves primal argument attributes and tangent
layout/shard/dimension metadata. Combined native fixtures pass 643 cases,
with 66 unsupported; focused AD/registry tests pass 347, with 16 skips.
Follow-up required: native member image/program projection, sum execution,
physical allocations, stream/completion ownership and owning numerical/timing
proof. This shared artifact evidence does not establish physical parity for
this backend. NVIDIA follow-up: existing SM120 packages are tested separately; this export is not yet a scaled AD CUDA program/ABI and does not transfer ROCm physical schedules. Fresh RTX 5070 regression proof passes 159 existing NVFP4/attention tests after the argument metadata change. Generic batching/transpose and full unit closure remain open.
Evidence: benchmarks/baselines/scaled_matmul_native_program_20261007/README.md.

## SCALED-MATMUL-ARTIFACT-BINDING-2026-10-07

Owner FRONTEND-IR-MEDIUM-1 / E2E-REAL-6.
Shared contract: per-instance Graph/Schedule/artifact binding keeps content
hash reuse distinct from SSA product ownership. Forged binding checks pass.
The native core lane passes 489 fixtures; 66 feature cases are unsupported.
Follow-up required: shared binding validation reaches existing native core fixtures; multi-product kernel/ABI ownership still needs integration. Fresh RTX 5070 regression proof passes 159 NVFP4/attention tests with rebuilt tools; it does not establish scaled AD execution.
No AD device timing, general batching/transpose or full-unit closure claim.
Evidence: benchmarks/baselines/scaled_matmul_artifact_binding_20261007/README.md.

## SCALED-MATMUL-NATIVE-JVP-2026-10-07

Owner FRONTEND-IR-MEDIUM-1 / E2E-REAL-6 / ROCM-FP8-BLOCKSCALE-1.
Shared contract: native Graph TangentInterface for exact_per_block f32
scaled products preserves block policy and transpose flags; active encoded
storage remains refused. Four focused native fixtures pass on Super-Bear WSL.
Graph/Schedule artifact evidence only; no new owning-device execution proof.
Follow-up required: repeated Schedule artifact identity/lifetime, transposed
FP8 Schedule admission, native sum/program integration and numerical/timing
proof. Generic batching/transpose closure and the full unit gate stay open.
Sibling physical parity is not established for this backend.
Evidence: benchmarks/baselines/scaled_matmul_native_jvp_20261007/README.md.

## COMPILER-NATIVE-LANES-2026-10-07

Owner W1.1 / E2E-REAL-6 / ROCM-FP8-BLOCKSCALE-1.
Shared contracts: native fixture admission, SDK environment propagation and
high-level kernel metadata ownership. No dtype, ABI, scale policy or physical
schedule changes. SM120 parity validated by 159 fresh RTX 5070 JIT numerical tests. The SM90 marker repair proves high-level IR validity only; older-SM executable promotion and exact-device proof remain follow-up required.

Core native fixtures pass 485 cases with 66 unsupported feature cases.
All 153 NVIDIA/ROCm backend fixtures and 374 registry/inventory checks pass.
Generic scaled_matmul batching/transpose and broader five-slice scopes
remain open; no full-unit, fleet-union or universal compiler closure claim.
Additional SM120 W1.1 owning proof passes 84 bounded row-major producer
cases through JIT/prepared/portable routes; no generic dynamic AD closure.
Evidence: benchmarks/baselines/compiler_native_lane_repair_20261007/README.md.



## ROCM-PAGED-KV-FLAT-INDEX-2026-10-07

Owner ROCM-E2E-2 / E2E-REAL-6.
Shared contract: compact f32/i32 storage, checked seven-scalar ABI and
256-thread geometry remain unchanged. Flat-token indexing eliminates redundant
head/feature division. Not applicable to nvidia physical proof: only the ROCm generator changes; no sibling ABI, package or physical schedule changes.

Focused owning gates pass 339 cases on gfx1151 and 336 on gfx1201.
Same-allocation HIP-event launch-window A/B and identical-image controls
remain distinct from isolated kernel/public timing. General layouts,
dynamic/composed consumers and wider performance obligations remain open.
No selector promotion or sibling proof transfer.
Recorder: benchmarks/rocm/record_paged_kv_index_ab.py.
Evidence: benchmarks/baselines/rocm_paged_kv_flat_index_20261007/README.md.


## COMPILER-FULL-UNIT-2026-10-07

Owner E2E-REAL-6 / FRONTEND-IR-MEDIUM-1.
Shared contract assessment: fresh unchanged non-slow unit gate on Super-Bear
WSL with matching native tools. Result: 21,694 passed, 7,503 skipped, 874
deselected, three failed. The two implementation gates are generic
scaled_matmul batching and linear-transpose closure. Their assertions and
partial/planned coverage states remain unchanged. The third failure was this
unit packet's missing tracked-document citation, now recorded here.
Host/unit coverage and foreign skipped lanes do not prove physical execution
for this backend; architecture-specific receipts remain separate.
Evidence: benchmarks/baselines/compiler_full_unit_20261007/README.md.


## GFX1151-CURRENT-SOURCE-2026-10-07

Owner E2E-REAL-6 / FRONTEND-IR-MEDIUM-1 / ROCM-E2E-2.
Shared contract: current source synchronization and matching LLVM/MLIR 23.1.1
core/ROCm build; existing typed Graph/Schedule/Tile/Target, checked native
image ABI and C++ prepared/resident ownership contracts are unchanged.
Receipt scope is compact static math/movement, not all compiler families.

Not applicable to SM120 physical proof: the receipt executes gfx1151
math/movement and native HIP ownership only. Shared typed compiler source is
synchronized and rebuilt; Radeon schedules, HIP timings and lifetime checks
do not establish new RTX 5070 execution. Existing NVIDIA dynamic/composed
producer and multi-result AD obligations remain open.

Recorders: benchmarks/rocm/benchmark_native_math_package.py,
benchmarks/rocm/benchmark_rocm_e2e_movement.py,
benchmarks/rocm/benchmark_resident_movement.py.
Evidence: benchmarks/baselines/gfx1151_current_source_revalidation_20261007/README.md.



## NVIDIA-NVFP4-SHARED-LHS-2026-10-07

Owner W1.1 / FRONTEND-IR-MEDIUM-1 / E2E-REAL-6.
Shared contract: named static NVFP4 shared-LHS batch policy, Graph scale and
orientation verification, sealed Schedule intent, Tile batch policy, checked
ten-argument rows/batches ABI. Public vmap maps RHS/scales with
in_axes=(None,0,None,0), reusing A/scales without Python replication or launch
loops. Existing scalar/shared-RHS/independent modes retain their contracts.

RTX 5070 parity validated: 89 frontend/runtime/native/device tests and
406 registry/manifest checks; 32 matched orientation rows (eight shared-LHS)
have zero measured error for the chosen operands. Five event/public timing
samples per row, with separate domains and no speedup claim. Dynamic/nested
batching, general linear-transpose AD and composed producer envelopes remain
open. Generic scaled_matmul batching/transpose closure gates remain open.
Recorder: benchmarks/nvidia/record_nvfp4_transpose.py.
Evidence: benchmarks/baselines/nvidia_shared_lhs_batch_20261007/README.md.
Parent synchronization key: NVIDIA-NVFP4-NATIVE-ORIENTATION-2026-10-06.


## GFX1201-PROJECTION-ATTRIBUTION-2026-10-07

Owner ROCM-FP8-BLOCKSCALE-1 / E2E-REAL-6. Native K128 projected
register lowering preserves checked positive dimensions and whole scale-group
K facts through LLVM assumptions. K32 FP8/MXFP8 retain byte-identical original
images. No dtype, numerical policy, scale semantics or runtime ABI change.
The MXFP8 descriptor error message is clarified; the identity fixture now
covers admitted K64 slabs with distinct keys and partial-slab refusal.

Not applicable to SM120 physical lowering: the LLVM transform is in
the ROCm register generator and this descriptor wording is ROCm-specific.
The coordinated core compiler rebuild passes; this does not establish new
RTX 5070 execution/timing parity. Existing attention/producer obligations
remain open.
Recorder: benchmarks/rocm/record_gfx1201_interleaved_compiler_formats.py.
Evidence: benchmarks/baselines/gfx1201_m200_k1536_attribution_20261007/README.md.


## GFX1201-CURRENT-SOURCE-2026-10-07

Owner ROCM-NVFP4-INGEST-1 / E2E-REAL-6. Current shared compiler/Python
source is synchronized; matching core and ROCm compiler builds are pinned.
Existing semantic, numeric-policy and physical ABI contracts are unchanged.

Not applicable to nvidia physical proof: this coordinating source-sync receipt executes gfx1201 native packages only. Shared Graph/AD/Tile source is assessed; no sibling schedule, ABI execution or timing evidence transfers.
Evidence: benchmarks/baselines/gfx1201_current_source_revalidation_20261007/README.md.



## NVIDIA-JIT-MULTIRESULT-OWNER-2026-10-07

Owner NVIDIA-LSE-1 / AD-RESIDUAL-EVAL-1 / FRONTEND-IR-MEDIUM-1 / E2E-REAL-6.
Shared contract: saved O/LSE result tuple, two seeds, read-only private residual
views, complete/compact seeded physical roles, synchronous seed lifetime.

RTX 5070 parity validated: 72 public JIT cases and 24 matched package/owner timing rows (600 windows). Compiler-free replay, caller mutation, repeated gradients and allocation/lifetime rejection are proved.
Dynamic/composed AD and asynchronous ownership remain open.
Recorder: benchmarks/nvidia/record_jit_multiresult_owner.py.
Evidence: benchmarks/baselines/nvidia_jit_multiresult_owner_20261007/README.md.



## NVIDIA-MULTIRESULT-ATTENTION-AD-2026-10-07

Owner NVIDIA-LSE-1 / AD-RESIDUAL-EVAL-1 / FRONTEND-IR-MEDIUM-1 /
E2E-REAL-6. Native reverse AD maps semantic O/LSE results and both seeds
through saved paired checkpoints into checked native packages. Physical
metadata checks exclude embedded sibling lineage.

RTX 5070 parity validated: 72 native generated-AD cases; 144 separate producer/consumer rows and 1440 numerically checked timing windows. Follow-up required: private residual owner, public JIT multi-result AD and broader composition/dynamic/batching.
Recorder: benchmarks/nvidia/record_multiresult_attention_ad.py.
Evidence: benchmarks/baselines/nvidia_multiresult_attention_ad_20261007/README.md.


## NVIDIA-LSE-COTANGENT-GRADIENT-ROLES-2026-10-07

Owner NVIDIA-LSE-1 / AD-RESIDUAL-EVAL-1 / E2E-REAL-6. Native seeded
checkpoint packages carry compact physical outputs or complete bias
gradients with sealed role/launch symbols and numerical identity checks.

RTX 5070 parity validated: all 15 compact activity masks, both layouts/thread counts and physical bias gradients pass; 66 rows / 660 checked timing windows. Follow-up required: automatic multi-result AD, residual ownership, public JIT and dynamic/composed integration.
Recorder: benchmarks/nvidia/record_lse_cotangent_gradient_roles.py.
Evidence: benchmarks/baselines/nvidia_lse_cotangent_gradient_roles_20261007/README.md.


## NVIDIA-LSE-COTANGENT-PACKAGE-2026-10-07

Owner NVIDIA-LSE-1 / AD-RESIDUAL-EVAL-1 / E2E-REAL-6. Native seeded
Graph/Schedule/Tile packages project checked row-seed buffer roles and
serialized replay. The shared runtime ABI is registered with validated
shapes, strides, scalars, aliasing and producer-stream ordering.

RTX 5070 parity validated: 36 full-gradient seeded package/replay cases; 180 event and 180 end-to-end numerical windows. Follow-up required: compact/bias-gradient envelopes, multi-result AD and residual-owner integration.
Recorder: benchmarks/nvidia/record_lse_cotangent_package.py.
Evidence: benchmarks/baselines/nvidia_lse_cotangent_package_20261007/README.md.


## NVIDIA-LSE-COTANGENT-BRIDGE-2026-10-07

Owner NVIDIA-LSE-1 / AD-RESIDUAL-EVAL-1 / E2E-REAL-6. Native bridge
indexes the explicit saved-LSE seed before outputs.

RTX 5070 parity validated for full Q/K/V seeded host, resident and event launches. Follow-up required: checked package descriptor, portable replay, compact/bias-gradient envelopes and multi-result AD.
Evidence: benchmarks/baselines/nvidia_lse_cotangent_bridge_20261007/README.md.

## NVIDIA-LSE-COTANGENT-SCHEDULE-2026-10-07

Owner NVIDIA-LSE-1 / AD-RESIDUAL-EVAL-1 / E2E-REAL-6. Explicit typed
Graph dLSE operands are verified, sealed in native Schedule, replay checked
and lowered to the SM120 Tile consumer. Caller Graphs are preserved. The
legacy bridge rejects new symbols until a checked seeded ABI is integrated.

RTX 5070 parity validated: 36 new Graph-generated numerical cases and 36 recorder rows / 180 event windows. Follow-up required: checked seeded package ABI, portable/common runtime, multi-result AD and residual-owner integration.
Recorder: benchmarks/nvidia/record_lse_cotangent_leaf.py --scheduled.
Evidence: benchmarks/baselines/nvidia_lse_cotangent_schedule_20261007/README.md.


## NVIDIA-LSE-COTANGENT-LEAF-2026-10-07

Owner NVIDIA-LSE-1 / AD-RESIDUAL-EVAL-1 / E2E-REAL-6. Shared Tile
attention backward declares an optional LSE cotangent pointer and verifies
its saved f32 deterministic-direct contract. Native SM120 materialization
includes P*dLSE in Q/K/bias derivatives; V derivatives remain unchanged.

RTX 5070 parity validated: matching build, 41 numerical/contract checks and 36 diagnostic rows with 180 poisoned event windows pass. Follow-up required: Graph/Schedule/package/multi-result AD integration and clean performance characterization. This leaf is not end-to-end closure.
Recorder: benchmarks/nvidia/record_lse_cotangent_leaf.py.
Evidence: benchmarks/baselines/nvidia_lse_cotangent_leaf_20261007/README.md.


## NVIDIA-MATCHED-PAIR-2026-10-07

Owner NVIDIA-LSE-1 / E2E-REAL-6 / AD-RESIDUAL-EVAL-1. Matched
saved/recompute two-package calls alternate lane order and validate poisoned
outputs after each forward/backward window. Native paired AD already retains
O/LSE; the standalone recorder now describes this accurately. No production
policy, ABI or numerical algorithm changes.

RTX 5070 parity validated: plain/bias, three shapes, 12 paired lanes and 60 poisoned numerical windows pass. Saved native paired ownership is already implemented; explicit LSE cotangents, dynamic/composed AD and publication remain open.
Evidence: benchmarks/baselines/nvidia_matched_checkpoint_pair_20261007/README.md.


## NVIDIA-PAIRED-COST-2026-10-07

Owner NVIDIA-LSE-1 / E2E-REAL-6 / AD-RESIDUAL-EVAL-1. The existing
JIT paired attention recorder measures capture plus backward and validates
borrowed results before frame release. Native gradient activity is respected.
No production numerical policy, ABI or physical schedule changes.

RTX 5070 parity validated: 18 expanded rows, 54 paired windows and caller-mutation residual checks pass. Native selection, complete matched recompute comparison, explicit LSE cotangents and dynamic/composed AD remain open.
Evidence: benchmarks/baselines/nvidia_paired_attention_cost_20261007/README.md.


## NVIDIA-CHECKPOINT-GRAPH-LINEAGE-2026-10-07

Owner NVIDIA-LSE-1 / E2E-REAL-6. Saved-checkpoint packaging verifies
retained Graph-to-Schedule ancestry before Schedule-to-Tile replay. Four-stage
digests include the matching native Target; no Python Graph reconstruction,
numerical algorithms, buffer roles, native ABI or schedules change.

RTX 5070 parity validated: 115 device/contract checks; host-free ancestry and paired/broadcast lanes pass 80 (six skips) and 122 tests. Counts overlap. Recorder static checks are clean and the 95-pass window/contract gate (six skips) passes. 48 benchmark arms and all 480 event/wall oracle windows pass with current source/compiler/runtime hashes. Large recompute backward is 308–2023 ms versus 2.4–8.9 ms saved; native long-shape residual selection remains a measured follow-up; explicit LSE cotangents, wider AD and publication remain open.
Evidence: benchmarks/baselines/nvidia_checkpoint_graph_lineage_20261007/README.md.


## GFX1201-PACKED-VECTOR-SCALES-2026-10-06

Owner ROCM-MXFP4-W4A8-1 / ROCM-NVFP4-INGEST-1. Native packed
materialization selects guarded vector activation-scale loads for static
M256/N>=1024/K>=1024; runtime/short/ragged seeds retain their images.
Numerical policy, packed decode, full-K scaling and checked ABI are unchanged.

Not applicable to nvidia physical lowering: this changes the gfx1201 packed materializer only. No shared ABI/op/dtype/pass or numerical policy changes; no sibling performance/device proof.
Evidence: benchmarks/baselines/gfx1201_packed_vector_scales_20261006/README.md.


## NVIDIA-FORWARD-LINEAGE-2026-10-06

Owner NVIDIA-LSE-1 / E2E-REAL-6. Forward descriptors export Graph,
Schedule and Target digests after native Graph/Schedule/Tile replay.
RTX 5070 parity validated: 46 serialized/native forward cases within the
69-pass focused gate. All 24 saved/recompute benchmark arms retain complete
ancestry and independent per-window numerical proof.
General composed/dynamic routes, explicit LSE cotangents and publication remain open.
Evidence: benchmarks/baselines/nvidia_checkpoint_event_readback_20261006/README.md.


## NVIDIA-EVENT-READBACK-2026-10-06

Owner NVIDIA-LSE-1 / E2E-REAL-6. Native profiler readback follows the
stop event; the recorder poisons outputs and validates every timed window.
RTX 5070 parity validated: 63 tests, eight poisoned-output cases and
24 saved/recompute forward/backward arms. Every event and wall window
passes independent FP64 checks. Device and public-wall scopes are separate.
Explicit LSE cotangents, wider AD and publication remain open.
Evidence: benchmarks/baselines/nvidia_checkpoint_event_readback_20261006/README.md.


## NVIDIA-SAVED-TUPLE-SSA-2026-10-06

Owner NVIDIA-LSE-1 / FRONTEND-IR-MEDIUM-1 / E2E-REAL-6.
Tuple assignment/copy/destructuring preserves all SSA results; rebinding
clears only the local binding. Native Graph/Schedule/Tile arithmetic and
checked runtime ABI are unchanged.
RTX 5070 parity validated: six new native cases and eighteen independent
FP64 O/LSE benchmark rows. Resident and public timing scopes remain separate.
Nested/dynamic tuples, composed attention and native LSE gradients remain open.
Evidence: benchmarks/baselines/nvidia_ordinary_attention_20261006/README.md.


## NVIDIA-PUBLIC-SAVED-LSE-2026-10-06

Owner NVIDIA-LSE-1 / FRONTEND-IR-MEDIUM-1 / E2E-REAL-6.
Public saved-LSE tuples retain f32 auxiliary types; production attention
tracing uses catalog shapes. Native checkpoint import preserves caller Graph
SSA roles and broadcast-bias extents.
RTX 5070 exact-device parity validated: 18 benchmark rows and the
685-pass regression run include public forward and existing saved/recompute
checkpoint consumers. Native explicit-LSE cotangent differentiation remains open.
General dynamic/composed attention, closure gates and publication remain open.
Evidence: benchmarks/baselines/nvidia_ordinary_attention_20261006/README.md.



## NVIDIA-NVFP4-NATIVE-ORIENTATION-2026-10-06

Owner W1.1 / FRONTEND-IR-MEDIUM-1 / E2E-REAL-6. Logical transpose intent
flows through native Graph verification, sealed Schedule identity, Tile code
and K16 scale accessors, PTX and checked packed-buffer shape guards. No
Python transpose/expansion in the production path; existing eight/ten-argument
launch contracts remain unchanged. Standalone native Schedule/Tile verifiers check boolean orientation types and require the named NVFP4 contract for enabled flags before target lowering.
RTX 5070 numerical parity validated for rank-two A/B orientations, independent-RHS batches, shared-RHS B orientation and public vmap. Native Graph rejects mismatched scale orientations; sealed Schedule and Tile orientation flags are checked.
Static transposed shared-batch A is integrated on RTX 5070 under
NVIDIA-NVFP4-SHARED-TRANSPOSE-2026-10-06 (see entry below); sibling physical
support is not inferred. General/dynamic batching, linear-transpose AD and
publication remain open. Evidence: benchmarks/baselines/compiler_contract_revalidation_20261006/README.md.


## NVIDIA-NVFP4-VMAP-SYMBOLIC-2026-10-06

Owner W1.1 / FRONTEND-IR-MEDIUM-1 / E2E-REAL-6. Symbolic dtype-unresolved
Graph annotations defer byte-layout inference until concrete tracing. Vmap
lifts scalar constraint dimension names onto mapped logical axes with a fresh
batch symbol, preserving scalar-owner metadata and no-map API semantics.
RTX 5070 typed execution validated: B=3/M=7 and B=7/M=3 keep distinct Schedule/package identities despite the same flattened M=21. Returning to the first geometry reuses its package; symbolic violations refuse before compilation.
General/dynamic batching, transpose/AD and publication remain open. Evidence:
benchmarks/baselines/compiler_contract_revalidation_20261006/README.md.


## NVIDIA-NVFP4-NATIVE-VMAP-2026-10-06

Owner W1.1 / FRONTEND-IR-MEDIUM-1 / E2E-REAL-6. Shared contract: leading
batch axes project logical Graph operand/result types and existing named
shared/independent RHS batching intent. Native Schedule/Tile owns geometry
and arithmetic; no Python member slicing, stacking or production launch loop.
RTX 5070 parity validated: public vmap shared/independent static batches use one checked native launch; scalar-owner/cache isolation and refreshed-input numerical checks pass.
General vmap/compositions/dynamic batches, transpose/AD and publication remain
open. Primitive closure gates remain enforced. Evidence:
benchmarks/baselines/compiler_contract_revalidation_20261006/README.md.


## NVIDIA-NVFP4-LOGICAL-JIT-2026-10-06

Owner W1.1 / FRONTEND-IR-MEDIUM-1 / E2E-REAL-6. Shared contract: explicit
logical NVFP4 dimensions over checked caller-owned compact byte storage.
Existing Graph/Schedule/Tile and runtime ABIs remain unchanged. RTX 5070 ordinary JIT numerical/cache checks pass for rank-two, shared-RHS and independent-RHS static NVFP4. Six oracle-exact rows retain separate public and native-event timings.
General batching/transpose/AD and publication remain open; no physical selector
promotion. Evidence: benchmarks/baselines/compiler_contract_revalidation_20261006/README.md.



## COMPILER-HOST-SEPARATION-2026-10-06

Owner E2E-REAL-6 / FRONTEND-IR-MEDIUM-1. The full non-slow unit replay
reported 25 failures and three errors. The relocated AMD compiler lacks host
compiler-rt builtins; complete host LLVM now handles unqualified oracle C/C++
builds, while AMD device generation remains explicitly pinned. Two new NVFP4
recorders are named in the benchmark inventory. All repaired lanes replayed:
340 passed. General scaled batching and linear-transpose closure remain open;
no gate or coverage status changed. No new owning physical evidence is claimed
for this backend by host-oracle repairs. Evidence:
benchmarks/baselines/compiler_contract_revalidation_20261006/README.md.

## COMPILER-INTEGRATION-DIAGNOSIS-2026-10-06

Owner E2E-REAL-6 / W1.1 / ROCM-FP8-BLOCKSCALE-1. Full host integration
completed with 21524 passes, 5862 skips and five failures. Three diagnosed
fixture/tool setup failures now pass within a 72-test focused replay (seven
skips). Generic scaled batching/transpose closure remains open; no closure
gate was weakened. Not applicable to sibling physical lowering: ROCm fixture and validation-tool setup changes only; no new device claim.
Evidence: benchmarks/baselines/compiler_contract_revalidation_20261006/README.md.



## NVFP4-NAMED-POLICY-PARITY-2026-10-06

Owner W1.1 / E2E-REAL-6 / FRONTEND-IR-MEDIUM-1. Shared contract: native
Graph/Schedule validates the exact NVFP4 numeric and scale-layout fields
already required by Python packaging. Three direct-MLIR positive-control
reproductions exposed extra fields being dropped by native import.
NVIDIA native rebuild passed; exact-policy and NVFP4 execution checks pass
within the final 460-test replay (16 skips). The profile keeps its
existing K16/FP32 arithmetic and batch ABI; no performance promotion.
Evidence: benchmarks/baselines/compiler_contract_revalidation_20261006/README.md.
Generic batching/transpose/AD and publication remain open.
Sync NVFP4-NAMED-POLICY-PARITY-2026-10-06.

## ROCM-FP16-SOFTMAX-PARITY-2026-10-06

Owner E2E-REAL-6 / ROCM-E2E-1. Restore existing FP16 softmax Graph admission
with native Schedule/Tile and f32 accumulation. No BF16/layout/AD promotion.
Not applicable to nvidia physical lowering: this changes only the
gfx1151 last-axis FP16 capability. No new sibling device evidence.
Evidence: benchmarks/baselines/compiler_contract_revalidation_20261006/README.md.
Sync ROCM-FP16-SOFTMAX-PARITY-2026-10-06.

## COMPILER-SHARED-REPLAY-2026-10-06: owning parity and source types

Owner E2E-REAL-6 / W1.1 / FRONTEND-IR-MEDIUM-1.
Shared contract: source comparison projection keeps structured result types
consistent with native i1 emission and the public f32 mask conversion.
Native source/checkpoint regression replay: 75 passed, two skipped.
RTX 5070 batch evidence remains owning proof. Shared ROCm existing-envelope
replay passed on both owning hosts; final broad integration remains open.
Broad integration, generic scaled batching/transpose and publication remain
open. Evidence: benchmarks/baselines/compiler_contract_revalidation_20261006/README.md.
Sync COMPILER-SHARED-REPLAY-2026-10-06.

## NVIDIA-NVFP4-INDEPENDENT-RHS-2026-10-06: checked native batch package

Owner W1.1; related E2E-REAL-6 / FRONTEND-IR-MEDIUM-1.
Shared contracts: scaled Graph batching policy, native batch-sensitive Schedule
digest, rank-three RHS/scales/output, batch-owned Tile row geometry, and a
ten-argument NVIDIA package ABI. Exact compiled batch scalars are checked in
launch and event timing. No Python production batch loop or default promotion.

RTX 5070 parity validated: 43 package/host/device checks and 561 shared
regression gates pass; three independent-RHS rows are oracle-exact.

General vmap/dynamic batches, wider storage, transpose/AD, final integration
and publication remain open. Benchmark event/wall domains stay separate.
Evidence: benchmarks/baselines/compiler_contract_revalidation_20261006/README.md.
Sync NVIDIA-NVFP4-INDEPENDENT-RHS-2026-10-06.


## NVIDIA-NVFP4-SHARED-RHS-BATCH-2026-10-06: native row batching

Owner W1.1; related E2E-REAL-6 / FRONTEND-IR-MEDIUM-1.
Shared contract: optional scaled Graph batching policy, logical rank-three
A/scales/output, native static row flattening, rank-preserving ABI guards,
and reuse of the driver-produced Schedule. No Python production batch loop,
new diagnostic, operation, pass, kernel ABI, or default selector.

Parity validated on RTX 5070: five owning numerical tests plus host/compiler checks pass (16 total). Three small batch benchmark rows are oracle-exact; kernel-event intervals and end-to-end wall intervals remain separate. Independent-RHS/dynamic batches, generic vmap, transpose/AD and other storage remain follow-up required.

Evidence: benchmarks/baselines/compiler_contract_revalidation_20261006/README.md.
Sync NVIDIA-NVFP4-SHARED-RHS-BATCH-2026-10-06.

Independent-RHS native layer: explicit batch scalars, batch-owned M16 tiles
and per-batch offsets compile and pass RTX 5070 numerical replay (12 tests).
Tile/LLVM/PTX proof only; Graph/Schedule/package integration and benchmark
remain pending. Sibling physical follow-up required, with no ABI/selector
activation for this backend beyond the owning NVIDIA native test path.

Typed Python-source follow-through: unsigned MLIR annotations round-trip via
canonical dtype names. Public scaled source reaches native Schedule/Tile/PTX;
174 source/device/dtype checks pass, with three oracle-exact RTX 5070 benchmark
rows and separate event/wall timing. Sibling physical proof remains follow-up
required; unsigned spelling repair changes no target dtype admission.

Public keyword follow-through: scaled_matmul accepts the declared transpose
and batching keywords; the packed-folded eager reference checks its supported
profile explicitly. Frontend/dashboard gates pass. General native transforms
and sibling physical execution remain open; no coverage promotion.

Native verification follow-through: Graph and Schedule NVFP4 scale ceiling
division avoids signed overflow at INT64_MAX K. Direct MLIR boundary tests
cover maximum K, overflowing batch-row products and zero dimensions. Matching
WSL core/NVIDIA rebuilds pass; 16 native/host checks and 297 NVIDIA device/registry
checks pass. No physical schedule, ABI or sibling execution claim changes.

## COMPILER-CONTRACT-REVALIDATION-2026-10-06: owning replay and drift repair

Owner E2E-REAL-6 / W1.1 / FRONTEND-IR-MEDIUM-1.
Shared contracts: strict native metadata, frontend signature/lifetime guards,
packed-operation keyword classification and canonical plan/log ownership.
No new kernel, ABI or physical schedule in this follow-through.

Parity validated for the existing RTX 5070 producer/attention envelope:
107 pass with the matching compiler. Native AD alias/return-lineage gates
also pass. General producer composition/layout/storage/AD remain open.

Portable integration remains failing pending remaining source/documentation
repairs. Zero-error mypy, full Python Ruff and focused registry gates pass.
Evidence: benchmarks/baselines/compiler_contract_revalidation_20261006/README.md.

NVIDIA device-test placement follow-through: six hardware-marked functions
moved to the owning device root, retaining all numerical assertions. RTX 5070
replay and placement guards pass 62 checks; no kernel, ABI or schedule change.
Parity validated for the existing NVIDIA envelope only. General native scaled-matmul transforms and final integration/publication gates remain open.




Shared scaled-matmul follow-through: generic logical transpose shape inference
and MLIR free-result dimensions now agree. Native batching/AD remain open.
SM120 normalized NVFP4 admission is repaired without changing sibling dtype
admission or physical schedules. Matching compiler rebuild, 540 focused gates,
34 capability gates and three oracle-exact scheduled NVFP4 benchmark rows pass.
RTX 5070 native package replay: four pass; CUDA-event/end-to-end timings are separate, with no speedup claim.

## NATIVE-CONTRACT-TYPE-GATES-2026-10-06: checked native metadata

Owner E2E-REAL-6 / FRONTEND-IR-MEDIUM-1 / W1.1.
No new operation, dtype, pass, diagnostic code or image ABI. Missing packed
metadata fields produce a contract ValueError; admitted Graph contracts and
attention sequence metadata are checked before projection.

Host metadata regression gates pass on Super-Bear WSL. No CUDA kernel or
image changes; broader resident composition and exact-device envelopes
remain open.

Validation: WSL focused suite 49 passed, 25 owning skips; module lint passes.
Mypy now passes all 625 source files at zero errors with baseline zero;
full Python Ruff passes. Follow-through frontend/portable/attention/movement/
MXFP8/dynamic projection tests: 125 passed in WSL. Publication and broader
integration gates remain open.



## ROCM-MATH-NATIVE-STAGING-2026-10-06: checked capacity reuse

Owner E2E-REAL-6 / FRONTEND-IR-MEDIUM-1. A native synchronous service
launches the existing compiler-owned unary/binary/scan images. It shares the
movement runtime's capacity arena and image leases; no kernels or numerical
algorithms are added. Original descriptors still check typed aliases, shapes,
roles and policy. Native validation checks source widths, exact byte products,
output/input overlap, alignment, dimensions, grid capacity, live chip,
device/context and PID before use. Every invocation uploads all inputs.
A failed completion retains a quarantined arena and lease until explicit
successful clear. Successful completion makes a pre-launch global sync
redundant. Retention is capped at 128 MiB per device/context arena; clear
before context teardown remains required. Older libraries retain the existing
launcher. TESSERA_ROCM_NATIVE_MATH=0 selects its baseline, and
TESSERA_ROCM_MATH_STAGING_REUSE=0 controls native allocation reuse alone.

Not applicable to nvidia physical execution: only the checked ROCm math descriptor branch calls this HIP service. Shared runtime binding changes have host gates; no sibling ABI, image, schedule, arithmetic or performance claim. Owning resident composition and AD follow-ups remain required.

Evidence: benchmarks/baselines/rocm_native_math_20261006/README.md.



## ROCM-MATH-LAUNCH-ATTRIBUTION-2026-10-06: measured host costs

Owner E2E-REAL-6 / FRONTEND-IR-MEDIUM-1. A checked portable-package
recorder attributes warm HIP calls separately from uninstrumented host walls.
All inputs change at the same addresses on every call; output is compared
with an independent promoted-f32 oracle before and after timing. Three input
storages, unary/binary/scan and two shapes give 18 cases per owning GPU.
Image acquisition/release and allocation/copy/launch call counts are checked.
Wrapper timings include instrumentation overhead and are diagnostic only.

Not applicable to nvidia physical execution: this diagnostic recorder measures the ROCm host-array package launcher only. No shared IR, runtime ABI, registry or physical schedule changes. Existing owning integration and performance obligations remain open; no HIP timing transfers.

Evidence: benchmarks/baselines/rocm_native_math_20261006/README.md.



## ROCM-MATH-WIDENING-2026-10-06: exact widening and math recorder migration

Owner E2E-REAL-6 / FRONTEND-IR-MEDIUM-1 / EVIDENCE-PACKET-1.
Explicit original Graph casts widen same-storage f16/bf16 entry tensors to
f32 before sqrt/exp/add/div/cumsum/cummax. Replay-sealed native Schedule
preserves those casts; native Tile folds exact widening into input loads
without an intermediate tensor allocation. Graph math type-preservation
verifiers are unchanged. Narrow Tile storage requires the ROCm ownership
contract; sibling Tile consumers retain f32 admission. Native Target
output_dtype is optional for legacy same-storage directives and explicit f32
for these checked math products. Six added image ABIs distinguish narrow
input storage from f32 output. Typed descriptors retain byte widths, aliases,
shape guards and roles; native image identity retains both storage types.
The tracer binds positional cast dtype as a static attribute. Capabilities,
manifest fixtures, execution metadata and pass summaries assess this envelope.

Not applicable to nvidia physical execution: exact widening Tile admission
requires a ROCm-owned sealed contract. Existing nvidia math storage/physical
schedules are unchanged. Shared Graph tracing/Tile/registry changes have host
gates; owning composition/storage/AD parity remains follow-up required.
No ROCm kernel, image, numerical or timing proof transfers.

Evidence: benchmarks/baselines/rocm_native_math_20261006/README.md.



## ROCM-MATH-NATIVE-PACKAGE-2026-10-06: ordinary JIT and portable native math

Owner E2E-REAL-6 / FRONTEND-IR-MEDIUM-1. Original isolated static f32
Graph -> verified native Schedule/Tile -> ROCm Target -> ROCDL/LLVM -> HSACO
now reaches ordinary @jit and portable checked packages. Python binds the
frontend/native contract; it does not construct physical kernels. Three
explicit f32 image ABIs cover unary, binary and inclusive last-axis scans.
Native identity removes validated static shape/SSA metadata while preserving
physical kind, storage, architecture and pipeline identity. Descriptor roles
preserve noncommutative reversed operands; each invocation checks exact
dimensions, complete aliases, compact storage and independent output.
Capabilities, numerical fixtures and a math-specific execution row are updated.

Not applicable to nvidia physical execution: this native math product and
its ABI projection are ROCm-owned. Shared registry/runtime changes require
host drift gates; architecture-owned composition/storage/AD remain follow-up
required. No ROCm image, physical schedule or device timing transfers.

Evidence: benchmarks/baselines/rocm_native_math_20261006/README.md.



## ROCM-MATH-NATIVE-SCHEDULE-2026-10-06: native math consumers — bounded proof

Owner E2E-REAL-6 / FRONTEND-IR-MEDIUM-1. Original Graph sqrt/exp/add/div/
cumsum/cummax -> replay-sealed native Schedule -> typed Tile -> ROCm
Target generators. SSA roles, policy, shape and architecture are checked.
No new operation/dtype/pass/image ABI; existing pass metadata updated.

Not applicable to nvidia physical execution: the new recipe is
ROCm-owned and f32-only. Existing shared Tile operations and passes
are reused; their diagnostic/pass/op/dtype host gates pass. Owning
backend composition/storage/AD follow-ups remain required. No ROCm
physical schedule, execution or timing proof transfers.

Evidence: benchmarks/baselines/rocm_native_math_20261006/README.md.


## ROCM-NVFP4-STATIC-PROGRAM-2026-10-06: checked warm contract retention

Owner ROCM-NVFP4-INGEST-1 / FRONTEND-IR-MEDIUM-1 / E2E-REAL-6.
Shared runtime change: retain at most 24 fully validated compiled NVFP4 program
contracts, keyed by detached exact typed metadata plus parent Graph/arguments.
List/tuple, bool/int and signed-zero distinctions are preserved; changed
contracts revalidate. Parsing uses the same detached snapshot as the key,
preventing caller mutation from changing an admitted product. Process guard
precedes the cache lock. No new op/dtype/target/pass/diagnostic/image ABI;
native input validation, all five uploads and native lifetime checks remain.

Not applicable to nvidia physical execution: this runtime consumer is the
gfx1201 NVFP4 compiled program. Shared host gates assess callback isolation,
diagnostics and ABI/frontend compatibility. Follow-up required for owning
architecture contract retention and exact-device proof; no HIP timing or
physical schedule transfers.

Evidence: benchmarks/baselines/rocm_nvfp4_allocation_reuse_20261006/README.md.


## ROCM-NVFP4-ALLOCATION-REUSE-2026-10-06: native checked idle ownership

Owner ROCM-NVFP4-INGEST-1 / FRONTEND-IR-MEDIUM-1 / E2E-REAL-6.
Shared contracts: additive native full-input update, checked prepare/release and
cache-clear C ABI; ordinary JIT/portable receipts include native allocation
cache hit metadata. Kernel image ABIs, numerical policy, Graph/Schedule/Tile
recipes, operations, dtypes, targets and passes are unchanged. The runtime ABI
audit generator records the additive exports.

Not applicable to nvidia physical execution: allocation retention and full
rebinding are confined to the native gfx1201 NVFP4 program. Shared runtime ABI
inventory and frontend receipt metadata are assessed by host drift gates.
Follow-up required for architecture-owned native residency/cache designs and
exact-device evidence; gfx1201 images, lifetime proof and timing do not transfer.

Evidence: benchmarks/baselines/rocm_nvfp4_allocation_reuse_20261006/README.md.


## NVIDIA-RECOMPUTE-GRAPH-2026-10-06: native recompute backward construction

Owner NVIDIA-LSE-1 / FRONTEND-IR-MEDIUM-1 / E2E-REAL-6. Shared contract:
original recompute backward Graph -> replay-sealed native Schedule -> native
Tile launch, with SSA role bindings, f16/bf16/f32 storage and complete supported
window/softcap/dropout policy. Canonical packaging no longer calls the Python
Tile constructor. The legacy emitter remains an explicit diagnostic/control.
No new operation, dtype, pass or ABI; existing pass metadata is updated.

Matching compiler build succeeds. 407 native contract, checkpoint,
diagnostic and pass metadata tests pass. Exact RTX5070: 54 numerical and
lifetime cases pass, including three storage types, full bias, reordered
arguments and changed-value replay. Native symbols retain storage width for
the existing host/resident transfer ABI; a unit regression guards this field.
A first device run exposed the missing symbol field and host-memory overrun;
the corrected build passes the same device scope.

Mixed legal keep/drop masks and signed-int64 seed extremes have exact-device
oracle proof. Target lowering explicitly reduces the seed modulo 32 bits before
adding the LCG offset, avoiding host signed overflow while preserving semantics.
The fresh seven-sample packet records forward, saved backward and recompute
CUDA-event and host-array timing separately. No selector promotion.

General composition, dynamic shapes, broadcast recompute bias, higher AD and
asynchronous ownership remain open.

Evidence: benchmarks/baselines/nvidia_ordinary_attention_20261006/README.md.


## NVIDIA-SAVED-GRAPH-2026-10-06: original checkpoint Graph admission — bounded proof

Owner NVIDIA-LSE-1 / FRONTEND-IR-MEDIUM-1 / E2E-REAL-6. Shared contract:
saved-LSE forward and backward Graphs retain authored SSA operands and policy
through native Graph-to-Schedule import; Python only copies frontend metadata
and decodes the native physical contract. The caller-owned Graph is not mutated.
The Graph dialect now declares optional forward row LSE and the existing
backward Graph spelling; native verification checks checkpoint shape relations.
This changes dialect/serialization surfaces and requires focused registry,
dialect and generated-dashboard gates before publishing.

Matching compiler build succeeds. Original-Graph policy/permutation/immutability
and checkpoint contracts: 56 pass. Exact RTX5070: 45 forward/backward, lifetime,
ordinary frontend and AD device checks pass. Seven-sample packet separates
forward/backward CUDA event windows from host-array complete launch walls.
Metadata/diagnostics/dialect gates: 330 pass. General composition, dynamic
shapes, wider storage and full five-slice closure remain open.
Evidence: benchmarks/baselines/nvidia_ordinary_attention_20261006/README.md.



Policy compatibility follow-through: 164 host Graph/checkpoint/registry checks
(one environment skip),
57 verifier/dialect checks and 22 route/benchmark/generated-coverage regressions
pass. Qualified NVFP4/storage verifier definitions are now counted by the
coverage scanner. Backward ODS verification preserves recompute admission;
neutral numerical attributes retain their meaning.
Exact RTX5070: 27 saved/recompute numerical and lifetime cases pass with
integer scale/default modifiers, argument permutations, plain/full bias and
three shapes. Current timing packet records both forward/backward domains.
At this earlier checkpoint recompute Tile construction was still open; the native migration above records the subsequent bounded closure.

## NVIDIA-ORDINARY-ATTENTION-2026-10-06: ordinary primal frontend execution

Owner E2E-REAL-6 / NVIDIA-LSE-1 / FRONTEND-IR-MEDIUM-1. Shared contracts:
retained Graph SSA operand-role projection, native distinct-argument admission
and ordinary checked-descriptor JIT dispatch. No new operation, dtype, ABI,
target, pass or stable diagnostic. Existing saved-LSE/JVP/VJP contracts remain.

Exact RTX5070 parity: 28 device cases and 36 oracle-gated ordinary/resident
benchmark rows pass. Static f32 full/causal, GQA, batch two, Q/K/V permutations,
full-bias permutations, changed values and portable descriptor replay execute
through canonical native Graph/Schedule/Tile/Target/LLVM. Shared regressions:
53 passed, 18 skipped. General composition, dynamic/wider dtype/broadcast
ordinary bias, higher AD and asynchronous ownership remain open.

[Evidence](../../../../benchmarks/baselines/nvidia_ordinary_attention_20261006/README.md).


## NVIDIA-W11-CENSUS-2026-10-06: current positive producer reconciliation

Owner W1.1 / FRONTEND-IR-MEDIUM-1. No shared operation, dtype, ABI, target,
pass or physical schedule changes.

Current source confirms the two historical tensor-MMA constructors register
only below SM120. Positive supported SM120 producers delegate/reconstruct
verified Graph -> native Schedule/Tile with pointer-backed views and typed
fragments. Current exact RTX5070 proof: 127 producer tests and twelve independent
oracle benchmark rows pass. Arbitrary tensor composition, noncanonical
accumulators, generic dynamic recovery and older-SM physical proof remain open.

[Evidence](../../../../benchmarks/baselines/nvidia_w11_census_20261006/README.md).


## ROCM-NVFP4-ARTIFACT-CACHE-2026-10-06: ordinary warm manifest retention

Owner ROCM-NVFP4-INGEST-1 / FRONTEND-IR-MEDIUM-1 / E2E-REAL-6.
Shared contract: bounded 24-entry compiler-product retention keyed by verified
Graph specialization; common-runtime manifest and native ABI validation remain.
Inspection artifacts and per-call input values are independent. No new ABI,
operation, dtype, target or pass.

Not applicable to nvidia physical execution: retention is scoped to gfx1201
NVFP4 JIT. Shared JIT source receives host regression gates; sibling architecture
compiler-product retention and exact-device evidence require owning follow-up.
No ROCm image, physical schedule or timing proof transfers.

[Evidence](../../../../benchmarks/baselines/rocm_packed_image_identity_20261006/README.md).


## ROCM-NVFP4-NATIVE-GRAPH-JIT-2026-10-06: native ordinary execution and graph replay

Owner ROCM-NVFP4-INGEST-1 / E2E-REAL-6 / FRONTEND-IR-MEDIUM-1.
Shared contracts: sixth native graph C ABI export; ordinary traced JIT and
common-runtime portable replay now bind the native C++ NVFP4 owner. Runtime
library discovery tolerates an absent compiler; portable deployment supplies
the matching native runtime independently. Existing compiler-owned Graph,
Schedule, Tile, Target and HSACO images retain their numerical semantics.

Shared ABI/audit gates: 32 passed, one environment skip. Not applicable
to nvidia physical execution: this runtime owns gfx1201 HIP resources and
NVFP4 ingest/storage/consumer images. Backend-owned lifecycle and execution
parity require architecture-specific follow-up; no ROCm timing or image proof
transfers to nvidia.

[Evidence](../../../../benchmarks/baselines/rocm_packed_image_identity_20261006/README.md).



## ROCM-NVFP4-NATIVE-OWNER-2026-10-06: native resident lifecycle

Owner ROCM-NVFP4-INGEST-1 / E2E-REAL-6 / FRONTEND-IR-MEDIUM-1.
Shared contracts: five registered C ABI exports prepare/update/invoke/read/close;
an explicit native_session adapter consumes existing compiler-owned stages.
No kernel, physical schedule, dtype, operation, target or pass is introduced.

Shared ABI inventory regenerated and host gates validated. Not applicable to
nvidia physical execution: this owner binds gfx1201 NVFP4 conversion/storage/
packed images and HIP private-stream resources. Existing nvidia runtime ABI
and physical routes are retained. Architecture-owned lifecycle parity requires
separate exact-device follow-up; no ROCm image, schedule or timing transfers.

[Evidence](../../../../benchmarks/baselines/rocm_packed_image_identity_20261006/README.md).



## ROCM-PACKED-IMAGE-IDENTITY-2026-10-06: packed-consumer image reuse

Owner ROCM-NVFP4-INGEST-1 / ROCM-MXFP4-W4A8-1.
Shared contracts: optional packed runtime-M/N image projection, fixed-K
whole/partial tile-class admission and separate authored/image Target digests.
Per-shape Graph/Schedule/Tile certificates and checked ABI guards remain static.
Projected consumer lineage also persists through the resident portable product;
compiler-free replay validates all retained authored/projected attributes.
No new operation, dtype, pass, diagnostic or ABI identifier.

Not applicable to nvidia physical execution: packed E2M1/E8M0 folded
BM256/BN64 images and HIP admission are gfx1201-specific. Existing nvidia
routes remain unchanged. Architecture-owned image identity and exact-device
parity require separate follow-up; no ROCm schedule or timing is transferred.

[Evidence](../../../../benchmarks/baselines/rocm_packed_image_identity_20261006/README.md).



## NVIDIA-DYNAMIC-ROW-RHS-2026-10-06: bounded physical RHS integration

Owner W1.1 / FRONTEND-IR-MEDIUM-1. Shared contracts: explicit bounded
rhs_storage_order, conflict-preserving Graph projection, two row-major
strided ABI identifiers, native symbol/storage matching and resident
shape/pitch validation. No new dtype/op/target/pass/stable diagnostic
or C ABI export. The bounded default remains column-major.

Parity validated on RTX5070: 781 regression checks pass (406 device
cases, four unrelated environment skips), plus 28 frontend contract checks.
All 384 matched row/column/control/prepared cases pass. Row ordinary wall
ratios are 1.06658/1.03469 and consumer event ratios 1.09243/1.14834 against
column storage. Explicit row support is numerical/layout coverage; gather
tuning and FP8/MXFP8/MXFP4 strategy gates remain open. General composition/
bufferization/control-flow/AD/resident/async and sibling closure remain open.

[Evidence](../../../../benchmarks/baselines/nvidia_dynamic_row_lhs_20261006/README.md).



## NVIDIA-BOUNDED-LHS-JIT-2026-10-06: ordinary bounded tensor reuse

Owner W1.1 / FRONTEND-IR-MEDIUM-1. Shared contracts: public immutable
shape_bounds, source/live-code certification, original Graph verification
before capacity projection, bounded role specialization and context-owned
24-entry lifecycle. No new dtype/op/target/pass/stable diagnostic or C ABI.

Parity validated on RTX5070: 547 regression checks pass, including 211
device cases. Four matched 48-case packets retain identical images and show
ordinary warm wall ratios 0.0765821/0.0761356. Separate component events do
not establish GPU algorithm gains. General composition/control-flow/AD,
dynamic row-RHS and resident/asynchronous consumers remain follow-up required.

[Evidence](../../../../benchmarks/baselines/nvidia_bounded_lhs_jit_20261006/README.md).



## NVIDIA-BIAS-JVP-2026-10-06: native bias tangent integration

Owner AD-RESIDUAL-EVAL-1; siblings E2E-REAL-6 / FRONTEND-IR-MEDIUM-1.
Shared internal checkpoint JVP, native AD export, Schedule sealing,
saved-generation identity, portable validation and one declared C ABI export
carry full/broadcast score bias and explicit dbias.
Parity validated on RTX5070: 36 resident and 36 public/matched cases,
three compiler-free replays and 38 device/guard regressions. Median per-case
prepared/replay host ratio 0.134276 includes transfers; separate forward/
tangent events do not establish GPU algorithm gains.
General composed/dynamic/layout/async/higher AD and wider dtypes remain open.
[Evidence](../../../../benchmarks/baselines/nvidia_bias_jvp_20261006/README.md).


## NVIDIA-PREPARED-ATTENTION-VJP-2026-10-06: native reverse runtime ownership

Owner AD-RESIDUAL-EVAL-1; siblings E2E-REAL-6 / FRONTEND-IR-MEDIUM-1.
Shared contracts: four registered C ABI exports, immutable reverse pin import,
bounded thread/process registrations, native requested-gradient binding and
saved-generation completion. No new operation/dtype/pass/diagnostic.
Parity validated on RTX5070: 88 public and 88 matched reverse cases, three compiler-free replays and native lifetime/context/fork guards. All native program pins are unchanged. Matched wall ratio median 0.0700637; separate forward/backward events do not establish GPU algorithm gains.
General composed/dynamic/layout/resident/async/higher AD remains open.
[Evidence](../../../../benchmarks/baselines/nvidia_prepared_attention_vjp_20261006/README.md).



## NVIDIA-PUBLIC-ATTENTION-VJP-2026-10-06: public native reverse integration

Owner AD-RESIDUAL-EVAL-1; siblings E2E-REAL-6 / FRONTEND-IR-MEDIUM-1.
Shared contracts: target-scoped Graph/Schedule family declarations, optional
persisted reverse runtime product, pinned checkpoint identity and checked ABI.
Parity validated on RTX 5070: 88 public reverse oracle cases and three compiler-free fresh-process replays. Static f32 canonical flash_attn, causal/grouped heads and full/broadcast bias are proved. Native prepared reverse ownership and general composed/dynamic/higher AD require follow-up.
No new operation, dtype, pass, diagnostic or C ABI is introduced.
[Evidence](../../../../benchmarks/baselines/nvidia_public_attention_vjp_20261006/README.md).



## ROCM-PAGED-SOFTMAX-EDGE-2026-10-06: native intermediate consumer

Owner E2E-REAL-6; sibling FRONTEND-IR-MEDIUM-1. Shared contracts changed:
public canonical f32 softmax descriptor dispatch, explicit per-op dtype
admission, static paired package binding, two registered C ABI exports and
generation/completion lifetime. 512 WSL tests pass, 13 environment skips.
Shared host dtype/binding/ABI contracts validated. HIP intermediate ownership
and the owning gfx1151/gfx1201 images are not applicable nvidia physical proof.
Backend-owned native composed consumers and exact-device execution remain
follow-up required. No ROCm physical schedule or timing is transferred.
[Evidence](../../../../benchmarks/baselines/rocm_paged_softmax_edge_20261006/README.md).


## ROCM-RESIDENT-MOVEMENT-OWNER-2026-10-06: native movement lifecycle

Owner E2E-REAL-6; sibling FRONTEND-IR-MEDIUM-1. Shared contracts: sealed package
matching, public preparation, common launch binding, five registered C ABI
exports and generation/completion ownership. Focused WSL gates: 394 pass,
13 hardware/environment skips.
Shared host binding/ABI contracts validated. HIP private storage/stream ownership
and the gfx1151/gfx1201 images are not applicable to nvidia physical execution.
Architecture-owned resident consumers and exact-device parity require follow-up;
no ROCm schedule or measurement is transferred.
[Evidence](../../../../benchmarks/baselines/rocm_resident_owner_20261006/README.md).


## ROCM-CAPTURED-MOVEMENT-2026-10-06: resident dispatch attribution

Owner E2E-REAL-6; sibling FRONTEND-IR-MEDIUM-1. Native Graph/Schedule/Tile/Target/LLVM images and checked descriptor symbols drive resident HIP graph replay; adjacent compiler lineage is verified. The benchmark stream helper retains default behavior.

HIP capture is not applicable to nvidia physical execution. This changes benchmark stream binding/timing evidence, with no shared runtime ABI or selector change. Its architecture-owned resident routes require independent proof.

Shared WSL gates: 30 pass, 13 hardware/environment skips. Benchmark capture is not ordinary runtime admission. Native resident ownership, general layouts, asynchronous movement, AD and full five-slice closure remain follow-up required.
[Evidence](../../../../benchmarks/baselines/rocm_captured_movement_20261006/README.md).

## NVIDIA-FORWARD-ATTENTION-JVP-2026-10-06: forward-only native product

Owner AD-RESIDUAL-EVAL-1; siblings FRONTEND-IR-MEDIUM-1 / W1.1. Public JVP packages a native forward checkpoint plus tangent, omitting reverse executable compilation/serialization. Pinned v2 carries no backward image; historical v1 and VJP paired products remain valid. Native Graph/AD/Schedule/Tile/NVVM/LLVM owns the images. Resident forward frames gate reverse calls before buffer access.

Parity validated on RTX5070: 72 public cases, four matched compile cases, six compiler-free fresh-process replays and seven device tests including compact backward regression. Median compile ratio 0.828158; no GPU algorithm change.

532 host tests pass. Generic VJP dispatch, composed/dynamic/higher AD, reverse intermediate construction and full five-slice closure remain open.
[Evidence](../../../../benchmarks/baselines/nvidia_forward_attention_jvp_20261006/README.md).

## NVIDIA-PREPARED-ATTENTION-JVP-2026-10-06: native runtime ownership

Owner AD-RESIDUAL-EVAL-1; siblings FRONTEND-IR-MEDIUM-1 / W1.1. Four private native C ABI exports retain CUDA modules, compiler sizing, aligned storage, role binding and timing events below the Python frontend. Graph/paired AD/Schedule/Tile/NVVM/LLVM images remain authoritative.

Parity validated on RTX 5070: 72 matched A/B cases, 72 public cases, 12 device tests and three fresh-process replays. Matched wall ratio 0.116622 is adapter overhead evidence; GPU algorithms are unchanged.

443 focused host tests pass. Unused reverse compilation, generic VJP/composed/dynamic/higher AD and full five-slice closure remain follow-up required.
[Evidence](../../../../benchmarks/baselines/nvidia_prepared_attention_jvp_20261006/README.md).

## NVIDIA-PUBLIC-ATTENTION-JVP-2026-10-06: canonical family/runtime route

Owner AD-RESIDUAL-EVAL-1; siblings FRONTEND-IR-MEDIUM-1 / W1.1; synchronization key NVIDIA-PUBLIC-ATTENTION-JVP-2026-10-06.
Public native_jvp consumes the actual specialized tracer Graph and native paired AD through the canonical family registry. The content-addressed parent carries the pinned native saved-LSE program; its registered runtime consumer owns uploads/capture/download lifetime. No Python GPU body or derivative recipe is added. Shared cache identity now includes captured literal environments, fixing causal/noncausal closure collisions; the frontend differential gate remains mandatory. Target declarations name implemented consumers rather than requiring fictional sibling routes.

Parity validated: 72 public RTX5070 oracle cases, four owning-device tests, three externally pinned common-runtime replays, and 444 host tests pass. Warm wall characterization remains separate from device-event evidence. Generic VJP dispatch, composition/dynamic/bias/higher AD and native prepared ownership remain follow-up required.

[Evidence](../../../../benchmarks/baselines/nvidia_public_attention_jvp_20261006/README.md). Full five-slice closure remains open.


## NVIDIA-JVP-PORTABLE-2026-10-06: pinned native product replay

Owner AD-RESIDUAL-EVAL-1; siblings FRONTEND-IR-MEDIUM-1 / W1.1; synchronization key NVIDIA-JVP-PORTABLE-2026-10-06.
Canonical program JSON pins the native checkpoint pair, tangent image/sizer, role mapping, frontend parameter names and manifests. Capture binds positional and keyword inputs to the recorded frontend signature, then projects native Q/K/V roles. Mapping/activity/generation and pointer/scalar/shape/grid contracts are validated before CUDA allocation. No Graph reconstruction, GPU body constructor, new op/dtype/target/pass/stable diagnostic or physical ABI is introduced.

Parity validated on RTX 5070: 72 restored-program finite-difference/mutation/repeated-direction cases and three fresh-process primal/tangent replays with all compiler subprocesses forbidden. The portable product is ready for the existing native-JVP family boundary; public dispatch integration and forward-only retirement of the unused reverse image remain follow-up required. No performance/default promotion is claimed. General composed/dynamic/bias/higher AD remains open.

[Evidence](../../../../benchmarks/baselines/nvidia_jvp_portable_20261006/README.md). Full five-slice closure remains open.

## NVIDIA-JVP-ARGUMENT-ORDER-2026-10-06: native frontend permutation integration

Owner AD-RESIDUAL-EVAL-1; siblings FRONTEND-IR-MEDIUM-1 / W1.1; synchronization key NVIDIA-JVP-ARGUMENT-ORDER-2026-10-06.
Native forward AD verifies distinct direct Q/K/V argument permutations and retains the paired O/LSE product and sealed activity/role contract. Automatic capture and requested tangent submission follow independently verified native frontend roles; native JVP activity must agree. Repeated/aliased roles remain outside this envelope. No new GPU body, op/dtype/target/pass/stable diagnostic or physical ABI is introduced.

Parity validated on RTX 5070: 72 independent finite-difference cases across all six argument permutations, short noncausal/long causal profiles, six wrt orders, actual caller mutation, repeated scaled directions and retained/closed frames. Twelve semantic groups retain identical native image bytes across permutations. Separate event/checked wall samples are characterization, not speedup or throughput. General composed/dynamic/bias/dropout/higher AD remains follow-up required.

[Evidence](../../../../benchmarks/baselines/nvidia_jvp_argument_order_20261006/README.md). Full five-slice closure remains open.

## NVIDIA-COMPACT-GRADIENTS-2026-10-06: native requested-output ABI

Owner AD-RESIDUAL-EVAL-1; siblings FRONTEND-IR-MEDIUM-1 / W1.1 / E2E-REAL-6; synchronization key NVIDIA-COMPACT-GRADIENTS-2026-10-06.
Native paired AD retains the complete logical Graph result product and seals requested physical output roles, launch layout and 64/128-thread geometry into Schedule/Tile. Target lowering removes inactive stores; the checked compact ABI allocates only requested gradients. Host/resident C++ submission validates the projected roles and symbol. No Python GPU body, new op/dtype/target/pass or stable diagnostic is introduced.

Parity validated on owning RTX 5070: 40 independent numerical cases across five arms, private capture after actual caller mutation, repeated cotangents/result ordering, checked host/resident execution and 18 device regressions. Long-row V-only retained gradient storage drops 7,736 to 3,096 bytes. Preserved logical launch ranges recover the named packed K-only submission gap; 64-thread blocks are explicit alternatives. Event windows and checked allocating wall time remain separate. Thirty-six finite logical-128 rows have worst prepared-event/wall candidate ratios 1.0435/1.0368, but no universal/default promotion follows. General composed/dynamic/higher AD and independent FP8/MXFP8/MXFP4 gates remain follow-up required.

[Evidence](../../../../benchmarks/baselines/nvidia_compact_gradients_20261005/README.md). The full five-slice compiler objective remains open.

## ROCM-PREPARED-MOVEMENT-2026-10-05: prepared native call binding

Owner E2E-REAL-6; sibling FRONTEND-IR-MEDIUM-1 / AD-RESIDUAL-EVAL-1; synchronization key ROCM-PREPARED-MOVEMENT-2026-10-05.
The canonical Graph/Schedule/Tile/Target/native image and checked ABI precede native preparation. C++ owns copied image/static ABI and validates host storage metadata, index bounds and context on invocation. Warm typed JIT calls compare a sealed Graph snapshot without serializing Graph IR. Close during invocation preserves image/ABI lifetime; explicit close/rebind and stale-handle/fork/context refusal are proved. Execution receipts are cleared at call entry and emitted after native completion. Three host C ABI exports are regenerated; no new op/dtype/target/pass/diagnostic or Python GPU body.

Reference MoE AD is reconciled with gather semantics: gather tangents, repeated-slot scatter-add cotangents and preserved opaque DispatchPlan tape slots. Independent finite differences, adjoint laws and public eager AD prove those reference contracts; compiled movement AD remains follow-up required.

Shared reference AD/Tape and native GPU storage regressions are parity validated by host tests. HIP context/image preparation is not applicable to CUDA; SM120 physical call binding/performance is unchanged. Architecture-owned producer/attention/quantized obligations remain open.

[Prepared call evidence](../../../../benchmarks/baselines/rocm_prepared_movement_20261005/README.md).


## ROCM-PUBLIC-MOVEMENT-2026-10-05: public tensor movement

Owner E2E-REAL-6; sibling FRONTEND-IR-MEDIUM-1; synchronization key ROCM-PUBLIC-MOVEMENT-2026-10-05.
Public physical-page reads and explicit token-of-slot MoE now have concrete eager and catalog shape contracts. Tracing retains static cache bounds and i32 indices without running the movement reference. Native Schedule accepts both distinct entry arguments in either order; semantic operand/binding roles remain sealed. The ordinary static ROCm JIT uses canonical compiler packages and checked descriptor scalars/outputs, with one runtime artifact per compiled specialization. Cache handles retain their two-result interface. No new op/dtype/target/pass/diagnostic or Python GPU body.

Shared native paged Schedule argument-order change is covered by native compiler host tests. Follow-up required for SM120 ordinary movement JIT admission and exact-device parity/performance; the ROCm JIT adapter is not an NVIDIA physical route. Existing producer/attention/quantized obligations remain open.

[Public JIT evidence](../../../../benchmarks/baselines/rocm_public_movement_20261005/README.md).


## ROCM-MOVEMENT-SPINE-2026-10-05: canonical native movement

Owner E2E-REAL-6; sibling FRONTEND-IR-MEDIUM-1; synchronization key ROCM-MOVEMENT-SPINE-2026-10-05.
The main compiler now retains the native movement Schedule artifact and binds packaging to the exact typed caller Graph/target. Adjacent Graph/Schedule/Tile/Target/backend digests and native producer metadata replace the previous bundle gap. Canonical primary/component gate names use the existing op catalog's dotted-cache normalization. Exact movement capabilities, manifest/numerical fixtures and checked descriptor execution rows agree; gfx1151 is no longer wholly unimplemented. Default gfx1201 static paged-read admission respects explicit opt-out. No new op/dtype/target/pass/diagnostic or Python GPU body.

Shared canonical gate-name normalization is parity validated by canonical host tests. Not applicable to NVIDIA physical scheduling: new movement admission, exact capability/manifest rows and the execution adapter are HIP-only. Existing SM120 producer/attention/quantized work remains open; no ROCm timings transfer.

[Canonical movement evidence](../../../../benchmarks/baselines/rocm_movement_admission_20261005/README.md).


## ROCM-NATIVE-MOVEMENT-2026-10-05: native host staging

Owner E2E-REAL-6; sibling FRONTEND-IR-MEDIUM-1; synchronization key ROCM-NATIVE-MOVEMENT-2026-10-05.
The checked Graph/Schedule/Tile/Target/LLVM movement images and f32/i32 kernel ABI are unchanged. A C++ HIP host service replaces per-launch Python allocation/copy/launch orchestration for synchronous default-stream packaged paged KV and MoE gathers. Context/device/architecture-owned buffers reuse capacity with a 128 MiB retention cap; completion precedes release, failures quarantine buffers and image leases, and explicit clear precedes context teardown. New host runtime exports are recorded by the owning runtime ABI generator. No new op, dtype, target, pass or diagnostic.

Not applicable to NVIDIA physical execution: the new host service uses HIP image leases, HIP contexts and existing ROCm memref ABIs. CUDA stream/buffer contracts and compiler images are unchanged. No ROCm timings are SM120 proof; existing NVIDIA producer/attention and quantized follow-ups remain open.

[Native movement evidence](../../../../benchmarks/baselines/rocm_native_movement_20261005/README.md).


## NVIDIA-VJP-ACTIVITY-2026-10-05: native requested-gradient pruning

Owner AD-RESIDUAL-EVAL-1; sibling FRONTEND-IR-MEDIUM-1; synchronization key NVIDIA-VJP-ACTIVITY-2026-10-05.
The optional native paired-AD checkpoint export validates the frontend wrt indices and maps them through verified Q/K/V/bias SSA argument roles. Schedule hashes seal gradient activity and zero_fill_v1 semantics; native Tile verification and SM120 Target lowering omit inactive gradient arithmetic and deterministically zero-fill those complete physical ABI outputs. The public isolated reverse program enables this native option and checks its requested result mapping. The default complete checkpoint export remains unchanged. No new operation, dtype, target, pass or diagnostic; Python carries the typed request and reads native package metadata.

Parity validated on RTX 5070 / sm_120: 431 focused tests pass, including native request/role mapping and tamper refusal, existing biased/permuted device routes, pass metadata and diagnostics. Matched finite numerical and alternating dispatch benchmarks cover Q-only, K-only, V-only, reordered mixed and full-gradient requests. Nonfinite primal-V and full/broadcast physical bias zero-fill checks are retained in the packet. Complete ABI output allocation and launch ranges remain unchanged; general composed/dynamic/higher AD and compact gradient ABIs remain follow-up required.

[Gradient activity evidence](../../../../benchmarks/baselines/nvidia_vjp_activity_20261005/README.md).


## NVIDIA-VALUE-JVP-2026-10-05: saved-LSE value-only AD integration

Owner AD-RESIDUAL-EVAL-1; sibling FRONTEND-IR-MEDIUM-1; synchronization key NVIDIA-VALUE-JVP-2026-10-05.
The native isolated attention JVP export binds the general linear checkpoint_forward(Q,K,dV) product to the primal Q/K/V O/LSE generation. General TangentInterface output remains unchanged. Native Schedule seals the value-only active roles and linear algorithm; native Tile lowering omits primal V/O reads and the unused shared moment reduction. The nine-pointer ABI is unchanged; its native sizing companion returns 512 shared bytes for this algorithm. Public compile_native_attention_jvp now accepts V-only activity. No new operation/dtype/target/pass/diagnostic and no Python GPU semantics.

Parity validated on RTX 5070 / sm_120: 341 focused tests; twelve ordinary JIT finite-difference/forward oracle cases; 24 V-only fixed-QK linearity cases with finite, Inf and NaN primal V, causal/noncausal and ragged/unequal query/key lengths. Maximum linearity error is 3.11e-8. Value-only kernels use 512 shared bytes, nine barriers, 29/38 registers and zero reported local bytes in the measured Sk5/129 envelopes. Dispatch and checked wall times remain separate. Broader composed/dynamic/bias/dropout/higher AD and quantized-format gates remain follow-up required.

[Value-only JVP evidence](../../../../benchmarks/baselines/nvidia_value_only_jvp_20261005/README.md).


## NATIVE-COMPILE-ORCHESTRATION-2026-10-05: measured compiler overhead reduction

Owner FRONTEND-IR-MEDIUM-1; sibling E2E-REAL-6; synchronization key NATIVE-COMPILE-ORCHESTRATION-2026-10-05.
SM120 saved-LSE JVP Graph-to-Schedule-to-Tile runs in one native MLIR pass manager, retaining typed SSA between verified passes instead of Python launching two processes and reparsing Schedule text. Native GPU packaging reuses exact binary SHA-256 only for the same stable file identity (device, inode, size, mtime and ctime), with bounded process memory and read-time rebuild checks. Hash bytes and image/arena/ABI content are unchanged. Python remains the frontend and thin process/package wrapper; native image validation still runs on every package. No new operation, dtype, target, pass or diagnostic.

Parity validated on RTX 5070 / sm_120: ten ordinary JIT oracle cases pass. Thirty balanced fresh-package A/B arms retain byte-identical image/arena output, cold-identity and warm-identity timings separately. 321 focused tests plus 142 shared identity/runtime tests pass (one unrelated skip). Complete-package medians improve about 12–14% with cold identities and 23% with stable identity reuse. General native in-process compiler sessions, image validation overhead and broader five-slice work remain follow-up required.

[Compiler overhead evidence](../../../../benchmarks/baselines/native_compile_orchestration_20261005/README.md).


## NVIDIA-JVP-NATIVE-SCHEDULE-2026-10-05: native saved-LSE tangent product

Owner FRONTEND-IR-MEDIUM-1 / AD residual integration; sibling W1.1; synchronization key NVIDIA-JVP-NATIVE-SCHEDULE-2026-10-05.
The registered paired checkpoint JVP now passes through native C++ Graph-to-Schedule and Schedule-to-Tile. The replay hash seals static f32 shapes, scale, causal policy, active tangent slots, primal/tangent argument roles and the SM120 cooperative algorithm. Native MLIR builds the arithmetic, shared buffers, barriers and tensor ABI; Python supplies only a typed Graph frontend or the actual native AD export. The nine-pointer ABI and checked arena/sizing companion remain unchanged. Unrelated functions and differing residual generations cannot be discarded during this isolated export.

Parity validated in the bounded RTX 5070 / sm_120 envelope: ten ordinary JIT Q/K/QKV cases pass independent finite differences, including reordered tangent inputs and ragged Sk=129. Five preloaded-kernel CUDA-event trials and checked allocating JVP wall windows are recorded separately. General composed AD, dynamic attention, bias/dropout and value-only native JVP integration remain follow-up required; no quantized-format/default strategy promotion.

[Native JVP evidence](../../../../benchmarks/baselines/nvidia_jvp_native_schedule_20261005/README.md).


## NVIDIA-NORM-ACCURACY-2026-10-05: native FP32 numerical closure

Owner W1.1; sibling FRONTEND-IR-MEDIUM-1; sync NVIDIA-NORM-ACCURACY-2026-10-05.
The SM120 native Target materializer uses MLIR square-root/division and compensated serial FP32 sums, guarded for IEEE Inf/NaN propagation. Graph/Schedule/Tile snapshots and the pointer/scalar ABI are unchanged; no dtype/op/pass/diagnostic registration changes.

Parity validated on RTX 5070: 136 device tests, 48 current numerical arms passing the original tolerance, and 40 ordinary composed program cases. Three original frozen-compiler arms fail; the named BF16 K4096 gap is fixed for both schedules. Consumer PTX/Target IR is unchanged; source/tool/image fingerprints, barriers, resources and separate event/wall timing are retained. General producers/dynamic/AD and broader five-slice closure remain follow-up required. Attention JVP still has a Python GPU MLIR body constructor and needs native Schedule/Tile migration. FP8/MXFP8/MXFP4 remain independent gates.

[Accuracy evidence](../../../../benchmarks/baselines/nvidia_norm_accuracy_20261005/README.md).


## NVIDIA-NORM-NATIVE-SELECTION-2026-10-05: native selection under evaluation

Owner W1.1; sibling FRONTEND-IR-MEDIUM-1; sync NVIDIA-NORM-NATIVE-SELECTION-2026-10-05.
C++ Graph-to-Schedule selects cooperative_128 for SM120 norm columns >= 256 and serial for shorter rows; explicit policies override selection. Python reads the native decision. Shared Schedule/Tile hashes and portable producer provenance retain the policy. No dtype, op, pass, diagnostic or quantized-format policy changes.

Parity validated in the bounded RTX 5070 envelope: 126 device tests and 36 composed paired cases, with identical consumer images. Shared tests: 424 passed, 17 skipped. Producer event time improves at K1024 but checked wall time does not consistently improve. Follow-up required: BF16 K4096 composed numerical tolerance fails for both serial and cooperative; retained attribution separates stored normalization rounding from consumer error. General producer/AD/dynamic and FP8/MXFP8/MXFP4 gates remain open.

[Selection evidence](../../../../benchmarks/baselines/nvidia_norm_native_selection_20261005/README.md).


## NVIDIA-COOPERATIVE-NORM-2026-10-05: native row schedule candidate

Owner W1.1; sibling FRONTEND-IR-MEDIUM-1; sync NVIDIA-COOPERATIVE-NORM-2026-10-05.
Native Schedule/Tile carries an explicit serial|cooperative_128 norm decision, included in the content hash. The SM120 cooperative materializer uses one 128-thread CTA per row, coalesced strided-column loads, shared partial reductions, centered LayerNorm variance and a scratch-reuse barrier. The pointer/scalar ABI is unchanged; host, resident and benchmark launch geometry recognizes the cooperative entry. Tile verification requires a string schedule and the owning sm_120 architecture. Packaging detects a selected NVIDIA tool that fails to materialize the cooperative schedule.
The explicit candidate preserves serial as the default. FP8, MXFP8 and MXFP4 remain independent correctness/quality/performance gates before any final strategy/default decision.

Exact RTX 5070 evidence compares serial/cooperative RMSNorm and LayerNorm across fp16/BF16/fp32, short/ragged/long rows, constant rows and large-offset centered variance. The matched native tools and CUDA bridge are fingerprinted; regular barrier/resource counts and separate resident event/checked host windows are recorded. Ordinary producer/JIT regressions remain required. General producer graphs, automatic strategy selection, dynamic frontend and composed AD remain open.

[Matched native schedule evidence](../../../../benchmarks/baselines/nvidia_cooperative_norm_20261005/README.md).



## NVIDIA-LHS-JIT-PROGRAM-2026-10-05: ordinary frontend producer-to-matmul execution

Owner W1.1; sibling FRONTEND-IR-MEDIUM-1; sync NVIDIA-LHS-JIT-PROGRAM-2026-10-05.
Ordinary primal @jit verifies the complete typed Graph, retains caller semantics, and partitions static fp16/BF16 RMSNorm, LayerNorm or last-axis softmax LHS edges into existing native Graph/Schedule/Tile/Target/PTX packages. Native tile.view/fragments, private intermediate allocation, checked producer/consumer ABIs, one owned CUDA stream and completion govern execution. Bias, activation, residual and final output dtype remain native consumer contracts; serialized replay validates Graph, argument roles and image lineage before allocation.
The shared Schedule helper resolves unbound tracer bias/residual role markers from typed SSA operand positions, preserving explicit named bindings and existing target gates. No new GPU semantic source templates, dtype, operation, pass or diagnostic.

Parity validated on RTX 5070 / sm_120: 53 LHS/RHS tests cover three producers, fp16/BF16, padded inputs, native epilogues, reordered arguments, unchanged caller Graph, cached execution, portable replay and fresh-process replay without a compiler. Twenty-four matched cases record correctness before separate producer/consumer CUDA-event dispatch windows and cold/warm/replay host wall time. General producer graphs, dynamic frontend and composed AD remain follow-up required.
FP8, MXFP8 and MXFP4 retain independent correctness/quality/performance gates; no default-format promotion.

[Frontend program evidence](../../../../benchmarks/baselines/nvidia_lhs_jit_program_20261005/README.md).



## ROCM-INGEST-RECIPROCAL-2026-10-05: exact native candidate normalization

Owner ROCM-NVFP4-INGEST-1; sibling ROCM-MXFP4-W4A8-1; sync ROCM-INGEST-RECIPROCAL-2026-10-05.
The native gfx1201 converter replaces FP64 candidate normalization by power-of-two division with multiplication by the exact reciprocal. Both factors are normal f64 powers of two; reduction order, midpoint rules, candidate search and tie-breaking are preserved. No Graph/Schedule/Tile, ABI, diagnostic, dtype or format policy changes.
Not applicable to SM120 physical execution: this edit is confined to the ROCm native converter, built only with that backend. Shared IR/ABI/registries and CUDA materialization are unchanged. An SM120 ingest optimization and exact-device proof remain follow-up required.
FP8, MXFP8 and MXFP4 remain independent correctness/quality/performance gates; no selector/default promotion.

[Paired converter packet](../../../../benchmarks/baselines/rocm_ingest_reciprocal_20261005/README.md).


## ROCM-INGEST-JIT-PROGRAM-2026-10-05: ordinary composed frontend execution

Owner ROCM-NVFP4-INGEST-1; sibling ROCM-MXFP4-W4A8-1; sync ROCM-INGEST-JIT-PROGRAM-2026-10-05.
Ordinary Python @jit now traces the named conversion/storage/scaled-matmul chain without eager arithmetic. Catalog inference supplies the BF16 MxN result. The complete typed Graph is verified before native Graph/Schedule/Tile/Target/LLVM stage packaging; SSA, numerical policy, argument roles and private buffer lifetime remain checked. Common runtime execution and fresh-process serialized replay retain all three native packages.
The contract is static and primal: M>64, N16/K64, gfx1201 packed folded storage. Unsupported argument effect/layout/sharding/model and function metadata require an explicit native contract. General AD, dynamic/layout envelopes and model-quality acceptance remain follow-up required. FP8, MXFP8 and MXFP4 remain independent evaluation gates; no default or general uint8 promotion.
Follow-up required for NVIDIA physical converter/storage parity: this program owns HIP resources and gfx12 storage. Shared Graph/catalog/runtime and SM120 regression are assessed on Super-Bear; the ROCm result does not establish SM120 ingest execution.

[Frontend native program packet](../../../../benchmarks/baselines/rocm_ingest_jit_program_20261005/README.md).


## ROCM-INGEST-PORTABLE-2026-10-05: portable resident native program

Owner ROCM-NVFP4-INGEST-1; sibling ROCM-MXFP4-W4A8-1; sync ROCM-INGEST-PORTABLE-2026-10-05.
Versioned JSON retains each native stage's Graph/Schedule/Tile/Target IR, HSACO and checked descriptor. Restore validates stage order, shape, semantic policy and ownership before GPU allocation; fresh-process replay needs no compiler. Nested metadata has independent ownership. Content digests check integrity, not origin authentication.
Not applicable to NVIDIA physical execution: replay owns HIP resources and gfx1201 HSACO. Shared native image/descriptor regression is validated on Super-Bear; an equivalent SM120 converter/storage program still requires implementation and exact-device evidence.
Ordinary composed JIT, general frontend/AD, dynamic/layout envelopes and model-quality acceptance remain follow-up required. FP8, MXFP8 and MXFP4 remain independent evaluation gates; no selector/default or general uint8 promotion.

[Portable resident packet](../../../../benchmarks/baselines/rocm_ingest_portable_20261005/README.md).

## ROCM-INGEST-RESIDENT-2026-10-05: owned converter/storage/matmul chain

Owner ROCM-NVFP4-INGEST-1; sibling ROCM-MXFP4-W4A8-1; sync ROCM-INGEST-RESIDENT-2026-10-05.
Not applicable to NVIDIA physical execution: this fixed program uses gfx1201 HSACO, packed gfx12 byte layout and HIP stream ownership. Shared ABI ancestry/operator/dtype/diagnostic/pass regressions pass on Super-Bear; conversion/layout/lifetime parity and exact SM120 proof remain follow-up required.
Shape-only consumer packaging preserves typed Graph/Schedule/Tile/native ownership without fabricated host weight hashes. Host-payload ancestry/content checks remain enforced per producer.
Composed JIT/portable program/general frontend-AD/dynamic-layout integration and model quality remain follow-up required. Matched-source FP8, MXFP8 and MXFP4 controls pass arithmetic checks; ingested folded output error is 15.177% relative RMS. No default/dtype/selector promotion.

[Resident native packet](../../../../benchmarks/baselines/rocm_ingest_resident_20261005/README.md).



## ROCM-INGEST-STORAGE-2026-10-05: lossless compiler storage bridge

Owner ROCM-NVFP4-INGEST-1; sync ROCM-INGEST-STORAGE-2026-10-05.
Follow-up required for NVIDIA storage/conversion physical parity. The matching rebuilt compiler passes 11 packaging tests and 39 exact RTX 5070 ordinary JIT/portable replay regressions; this is shared-contract regression evidence, not gfx1201 conversion parity.
The named checkpoint-to-fragment byte operation adds no quantization. General uint8 stays planned/gated and sibling capabilities explicitly remain unsupported.
Resident converter/bridge/consumer lifetime, combined timing and model quality remain follow-up required. FP8, MXFP8 and MXFP4 remain independent mandatory correctness/quality/performance gates; no default promotion.

[Storage bridge packet](../../../../benchmarks/baselines/rocm_ingest_storage_bridge_20261005/README.md).



## ROCM-INGEST-RUNTIME-2026-10-05: canonical JIT conversion

Owner ROCM-NVFP4-INGEST-1; sync ROCM-INGEST-RUNTIME-2026-10-05.
Follow-up required for NVIDIA conversion implementation/exact-device proof. Shared literal-attribute capture, multi-result frontend typing and descriptor runtime changes are assessed with RTX 5070 regression; no gfx1201 conversion proof transfers to SM120.
The exact gfx1201 manifest/capability describes only the named static checkpoint operation; general uint8 remains planned/gated and sibling conversion states remain unsupported.
FP8, MXFP8 and MXFP4 remain mandatory independent correctness/quality/performance gates; no default promotion.

[Canonical JIT/runtime packet](../../../../benchmarks/baselines/rocm_ingest_runtime_20261005/README.md).



## ROCM-CHECKPOINT-NATIVE-INGEST-2026-10-05: native pinned conversion

Owner ROCM-NVFP4-INGEST-1; sync ROCM-CHECKPOINT-NATIVE-INGEST-2026-10-05.
Follow-up required for nvidia conversion implementation/exact-device proof. Shared Graph byte spelling is ui8 while uint8 remains planned/gated; the named checkpoint operation explicitly has no sibling execution capability. This gfx1201 packet does not prove nvidia numerical or performance parity.
FP8, MXFP8 and MXFP4 remain mandatory independent correctness/quality/performance gates; no default promotion.

[Checkpoint native conversion packet](../../../../benchmarks/baselines/rocm_checkpoint_native_ingest_20261005/README.md).



## ROCM-GRAPH-INGEST-2026-10-05: semantic conversion ownership

Owner ROCM-NVFP4-INGEST-1; sync ROCM-GRAPH-INGEST-2026-10-05.
Follow-up required for nvidia semantic registration/AD/conformance and checked conversion ABI assessment; the physical converter is exact-gfx1201 only. No sibling execution parity is inferred.
The shared three-result Graph contract retains explicit lossy policy and projection globals; Schedule replay binds policy, layout and private output ownership.
FP8, MXFP8 and MXFP4 remain mandatory independent evaluation gates.

[Graph ownership checkpoint](../../../../benchmarks/baselines/rocm_graph_ingest_20261003/README.md).



## ROCM-NATIVE-INGEST-LEAF-2026-10-03: native joint-SSE conversion leaf

Owner ROCM-NVFP4-INGEST-1; sync ROCM-NATIVE-INGEST-LEAF-2026-10-03.
Not applicable to nvidia execution: this physical converter is exact-gfx1201 only. The future target-neutral loss/scale/boundary contract needs sibling assessment during Graph/Schedule/Tile integration. No nvidia conversion or numerical/performance parity is inferred.
FP8, MXFP8 and MXFP4 remain independent mandatory evaluation gates.

[Native ingest leaf packet](../../../../benchmarks/baselines/rocm_native_ingest_20261003/README.md).


## ROCM-PACKED-NATIVE-2026-10-03: compiler-owned packed MXFP4

Owner ROCM-MXFP4-W4A8-1; sibling ROCM-NVFP4-INGEST-1. Sync ROCM-PACKED-NATIVE-2026-10-03.
Not applicable to nvidia physical code generation: this materializer is gated to gfx1201. Shared ROCm runtime admission validates static image geometry and expanded memref ABI; nvidia requires its own packing/scale/producer and exact-device proof. No sibling parity is inferred.
FP8, MXFP8 and MXFP4 remain mandatory independent evaluation gates before final/default decisions.

[Native packed packet](../../../../benchmarks/baselines/rocm_packed_native_20261003/README.md).


## ROCM-CHECKPOINT-FORMAT-GATE-2026-10-03: pinned format evaluation

Owner ROCM-NVFP4-INGEST-1; sync ROCM-CHECKPOINT-FORMAT-GATE-2026-10-03.
Not applicable to nvidia physical execution: this recorder selects exact gfx1201 packages and the pinned ROCm gate/up consumer. Shared benchmark helpers now accept actual source arrays and bounded FP4 batches with layout-preserving scale-plane materialization. nvidia needs its own format/quantization/geometry and exact-device comparison before any strategy selection; no sibling parity follows.
No format default is promoted. The folded MXFP4 arm has expanded E4M3 physical weights and a named legacy zero-block scale policy; it is not packed native compiler closure.

[Format packet](../../../../benchmarks/baselines/rocm_checkpoint_format_gate_20261003/README.md).


## NVIDIA-LAYERNORM-RHS-2026-10-03: second normalization producer

Owner W1.1 / FRONTEND-IR-MEDIUM-1; sync NVIDIA-LAYERNORM-RHS-2026-10-03.
Parity validated on RTX 5070: ten FP16/BF16 LayerNorm RHS-to-matmul ordinary JIT and portable replay benchmark cases. Stable centered variance, default epsilon, argument order, padded host compaction, private lifetime and cache reuse are proved. The manifest binds normalization kind to the native producer; legacy RMSNorm-only schemas retain their contract.
Affine/dynamic/general producer and AD integration remain open. FP8, MXFP8 and MXFP4 are mandatory correctness/performance gates before final/default strategy selection.

[LayerNorm RHS packet](../../../../benchmarks/baselines/nvidia_layernorm_rhs_20261003/README.md).


## NVIDIA-RHS-JIT-DISPATCH-2026-10-03: ordinary calls and portable execution

Owner W1.1 / FRONTEND-IR-MEDIUM-1; sync NVIDIA-RHS-JIT-DISPATCH-2026-10-03.
Parity validated on RTX 5070 for ten ordinary JIT and serialized runtime replay cases. Complete native Graph verification, checked two-package execution, argument lineage, owned stream/lifetime, cache reuse and failure-before-allocation tests passed. General graphs, dynamic shapes and composed AD remain follow-up required.
FP8, MXFP8 and MXFP4 remain mandatory correctness/performance gates before a final or default strategy decision.

[Runtime packet](../../../../benchmarks/baselines/nvidia_rhs_jit_dispatch_20261003/README.md).


## NVIDIA-RHS-JIT-2026-10-03: traced producer graph integration

Owner W1.1 / FRONTEND-IR-MEDIUM-1; sync NVIDIA-RHS-JIT-2026-10-03.
Parity validated on RTX 5070: the explicit JIT compile API traces one RMSNorm RHS-to-matmul Graph, verifies it, checks/rebuilds structured CFG metadata across native package partitions and preserves frontend argument order. Eight numerical resident cases passed; automatic JIT dispatch, composed autodiff and general graphs remain follow-up required. FP8, MXFP8 and MXFP4 remain mandatory before any default strategy decision.

[Frontend packet](../../../../benchmarks/baselines/nvidia_row_major_b_schedule_20261003/README.md).


## NVIDIA-ROW-MAJOR-B-SCHEDULE-2026-10-03: checked RHS storage selection

Owner W1.1; sync NVIDIA-ROW-MAJOR-B-SCHEDULE-2026-10-03.
Parity validated on RTX 5070 for static unfused fp16/BF16, fp32-output Graph-to-Schedule-to-Tile packages with row-major RHS. Hash replay, checked compact-layout rejection, host copies and resident launches passed 20 matched row/column cases. The named static RMSNorm RHS producer now composes with this consumer without an intermediate host transfer; eight fp16/BF16 numerical cases prove same-stream ownership, closed-buffer rejection and separate producer/consumer event windows. Dynamic/fused widening, arbitrary producer graphs and isolated kernel-only attribution remain follow-up required. FP8, MXFP8 and MXFP4 are required evaluation gates before any strategy/default promotion.

[Checked package packet](../../../../benchmarks/baselines/nvidia_row_major_b_schedule_20261003/README.md).


## NVIDIA-ROW-MAJOR-B-CORE-2026-10-03: RHS tensor storage materialization

Owner W1.1; sync NVIDIA-ROW-MAJOR-B-CORE-2026-10-03.
Parity validated for the native physical loader on RTX 5070: explicitly transposed row-major fp16/BF16 B views gather K elements at their actual pitch, including bounded ragged edges and unbounded complete fragments. Typed accumulator carries remain intact. This is raw Tile lowering evidence; native Schedule/profile, checked package and resident RHS producer integration remain follow-up required. No default strategy or low-precision promotion is made.

[Physical loader packet](../../../../benchmarks/baselines/nvidia_row_major_b_core_20261003/README.md).


## NVIDIA-BROADCAST-CHECKPOINT-PACKAGE-2026-10-03: checked saved-state integration

Owner E2E-REAL-6 / AD-HIGHER-1; sync NVIDIA-BROADCAST-CHECKPOINT-PACKAGE-2026-10-03.
Parity validated on RTX 5070: distinct saved-LSE broadcast ABIs carry four checked physical bias extents while native kernels retain seven logical dimensions. Forward copies, backward copies/geometry, physical dBias guards, private tape allocations and pairing identity now preserve rank-four broadcasts. Checked host packages and public JIT reverse execute across every broadcast axis, combined reductions, GQA, full/causal masking and both Sq/Sk orderings. Public capture survives caller mutation, repeated/changed cotangents and rejects use after close. Higher-order/JVP broadcast, lower-rank/dynamic bias and pruning unrequested cotangents remain follow-up required; no throughput or format promotion claim.

[Package and public JIT packet](../../../../benchmarks/baselines/nvidia_broadcast_checkpoint_package_20261003/README.md).


## NVIDIA-BROADCAST-CHECKPOINT-CORE-2026-10-03: physical bias derivative

Owner E2E-REAL-6 / AD-HIGHER-1; sync NVIDIA-BROADCAST-CHECKPOINT-CORE-2026-10-03.
Native arithmetic parity validated on RTX 5070 for 12 rank-four bias broadcasts, including every axis, combined reductions, GQA and full/causal masking. Paired AD exports physical-shaped dBias; native Schedule hashes shape and fixed B/Hq/Q/K reduction policy. Tile and NVIDIA LLVM lowering use one physical owner without atomics or dense temporaries. Checked package/tape ABI remains follow-up required: host copy capacities, shape guards, private capture and backward allocations must carry physical extents. Pairing identity now binds physical extents and reduction policy, with mismatched pairs rejected before compilation. The temporary package gate prevents using the old dense-copy descriptor. This is raw native arithmetic proof, not completed public JIT execution.

[Native core packet](../../../../benchmarks/baselines/nvidia_broadcast_checkpoint_core_20261003/README.md).


## ROCM-NVFP4-INGEST-1-QWEN3-GATE-UP-2026-10-03: real merged source proof

Owner ROCM-NVFP4-INGEST-1; sync ROCM-NVFP4-INGEST-1-QWEN3-GATE-UP-2026-10-03.
Not applicable to CUDA physical execution: only gfx1201 checkpoint recording and ROCm format fixtures changed. The recorder's new merged-source fields and correctness-before-timing gate do not change CUDA ABI or native AD. Follow-up required for an architecture-owned real-checkpoint ingest/quality comparison. Existing RTX 5070 W1.1/saved-LSE/backward evidence remains separate; AMD schedules and timings do not establish CUDA parity.

[Packet](../../../../benchmarks/baselines/rocm_nvfp4_gate_up_20261003/README.md).


## ROCM-MXFP8-LDS-2026-10-03: native wide-grid schedule and guarded image reuse

Owner ROCM-FP8-BLOCKSCALE-1; sibling ROCM-MXFP4-W4A8-1; sync ROCM-MXFP8-LDS-2026-10-03.
Follow-up required for a CUDA E8M0 consumer and matched three-format evaluation. Shared Schedule verification admits only named gfx1201 profiles; the native scheduling-intent attribute resolves inside the ROCm branch and HIP runtime admission changes do not add a CUDA ABI. Existing-route parity validated: 86 scheduled matmul/attention device cases pass on RTX 5070 after the matching shared-pass rebuild, with six existing oracle warnings; 345 host registry/package gates also pass. AMD schedules and speedups do not establish NVIDIA format support.

[Exact-device packet](../../../../benchmarks/baselines/rocm_mxfp8_lds_20261003/README.md).


## ROCM-THREE-FORMATS-2026-10-03: required format evaluation and partial LDS copies

Owner ROCM-FP8-BLOCKSCALE-1 and ROCM-MXFP4-W4A8-1; sync ROCM-THREE-FORMATS-2026-10-03.
Not applicable to CUDA sm_120 physical execution: the repair is confined to the ROCm FP8/folded LDS copy generator and the recorder binds gfx1201-only packages. No shared dialect, ABI, runtime or capability contract changes. The benchmark protocol now separates source quantization, folding approximation and native arithmetic error plus device/E2E time. Follow-up required for a matched CUDA sm_120 three-format evaluation on its owning hardware; RX 9070 XT results do not establish sibling parity.

[Exact-device packet](../../../../benchmarks/baselines/rocm_three_formats_20261003/README.md).



## ROCM-MXFP8-EXPONENT-SCALE-2026-10-03: standard E8M0 scale consumer

Owner ROCM-FP8-BLOCKSCALE-1; sync ROCM-MXFP8-EXPONENT-SCALE-2026-10-03.
Follow-up required for a native E8M0 consumer: the shared Tile wording now
permits native scaling equivalent to the f64 reference with one f32 rounding.
The implementation change is confined to ROCm lowering; no NVIDIA ABI or
Schedule changed. Existing RTX 5070 scheduled matmul/attention regressions
passed 86 cases after the shared ODS rebuild. This is existing-route parity,
not NVIDIA MXFP8/E8M0 support; AMD ldexp timings do not transfer to CUDA.

[Exact-device A/B packet](../../../../benchmarks/baselines/rocm_mxfp8_exponent_scale_20261003/README.md).


## ROCM-MXFP8-PACKAGE-2026-10-03: checked runtime and image identity

Owner ROCM-FP8-BLOCKSCALE-1; sync ROCM-MXFP8-PACKAGE-2026-10-03.
Not applicable to NVIDIA physical execution: the package profile, native
identity projection and HIP byte-scale launcher admission are gfx1201-specific.
Shared runtime.py edits are confined to ROCm registration/submission; no CUDA
ABI or schedule changed. Existing RTX 5070 matmul/attention regressions are
rechecked separately. MXFP8 execution on NVIDIA remains follow-up required;
AMD schedules and timings do not establish CUDA parity.

[Checked package packet](../../../../benchmarks/baselines/rocm_mxfp8_checked_package_20261003/README.md).


## ROCM-MXFP8-SCHEDULE-2026-10-03: native compiler integration

Owner ROCM-FP8-BLOCKSCALE-1; sync ROCM-MXFP8-SCHEDULE-2026-10-03.
Existing-route parity validated: the integrated shared pass preserves newer
SM120 producer/tail/epilogue changes; 86 scheduled matmul/attention device
cases passed on Super-Bear RTX 5070 with CUDA 13.3 after a complete matching
build. The gfx1201 MXFP8 K32 physical contract and wide-scale ABI do not add a
CUDA consumer. Follow-up required for MXFP8 native admission or explicit
target refusal, capability classification and exact-device format evaluation.


## ROCM-E8M0-TILE-2026-10-03: standard E8M0 consumer foundation

Owner ROCM-FP8-BLOCKSCALE-1; sync ROCM-E8M0-TILE-2026-10-03.
Follow-up required: the shared Tile fragment scale operation admits explicit
standard E8M0 raw-byte scales, with f64 scaling of an isolated f32 partial
before its f32 accumulator join. NVIDIA has no consumer of this extension
proved here. Existing sm_120 quantized packages and W1.1 evidence do not imply
its support. Assess a native consumer or explicit target refusal separately;
no CUDA ABI, schedule, or execution claim is changed by gfx1201 evidence.


## ROCM-FOLDED-RETUNES-2026-10-02: gfx1201 performance experiment assessment

Owner ROCM-MXFP4-W4A8-1; sync ROCM-FOLDED-RETUNES-2026-10-02.
Not applicable: CUDA has no affected retained IR, memory ABI, or execution route.
ROCm fragment/LDS/prefetch/C-transpose changes are revision-bound experimental
patches, removed from active source. Exact gfx1201 numerical/timing evidence
does not establish nvidia parity. Proposed persistent tile scheduling still
needs an explicit verified contract and owning-target validation.
[ROCm packet](../../../../benchmarks/baselines/rocm_folded_c_lds_transpose_20261002/README.md).


## `W1.1-SM120-TYPED-PRODUCER-EDGE-2026-10-01`: bounded typed producer proof

Owner W1.1; sync `NVIDIA-W1.1-TYPED-PRODUCER-EDGE-2026-10-01`.

Parity validated on Super-Bear RTX 5070 (sm_120a) for static public
RMSNorm -> matmul edges at fp16 [16,16] x [16,8] and [16,64] x [64,8]. The
four-K case proves the scheduled consumer carries a typed accumulator through
four loop iterations. Consumer Tile IR carries `tile.view`, typed
`fragment_pack` A/B, `fragment_zero`, `tile.mma`, `fragment_unpack`, and
`tile.store`; NVIDIA Target IR contains
`nvvm.mma.sync`, and PTX contains
`mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32`. A correctness-gated
benchmark asserts these markers and records the Schedule/Tile digests.
Producer and consumer are separate native packages. The resident chain reuses
one intermediate allocation on the same stream; max absolute errors are 0
for RMSNorm and 1.4305e-6 for matmul across both shapes. Three-batch CUDA-event
timings are diagnostic only and support no performance claim. See
[packet](../../../../benchmarks/baselines/nvidia_sm120_w1_typed_tensor_edge_20261001/README.md).

A follow-up admits scheduled softmax as a resident producer alongside RMSNorm.
The public from_text route is numerically validated for fp16 and bf16; bounded
K reuses one package at active K=7, 11, and 16 under a K=16 bound. The
correctness test covers both dtypes; the benchmark packet measures fp16 active
K=7 only and reports separate producer/consumer medians of 7.14/9.99 us with
24.5%/83.9% CV. These noisy timings are diagnostic only. See the linked packet
for image digests and full samples.

This proves bounded resident package reuse, not generic tensor lifetime
conversion. The two generic tensor-valued constructors `LowerMatmulToTileMMA`
and `LowerKReductionAddToTileMMA` in `TileIRLoweringPass.cpp` remain open under
W1.1. No shared IR or ABI changed. Apple, ROCm, and x86 do not consume the
SM120-specific fragment lowering.

## `NVIDIA-NVFP4-SCHEDULE-2026-09`: scheduled NVFP4 package - parity validated

Owner E2E-REAL-6; sync `NVIDIA-NVFP4-SCHEDULE-2026-09`.
Super-Bear RTX 5070 (sm_120), built from this scratch checkout with LLVM/MLIR
23.1.1 and CUDA 13.4.59: the NVFP4 scaled-matmul package now flows through
Graph IR -> Schedule IR -> Tile IR -> NVIDIA Target IR -> PTX. Exact-device
numerics pass for 16x8x64, 33x19x129, and ragged 7x5x31 against decoded
NV_E2M1/UE4M3 NumPy. The correctness-gated packet records separate CUDA-event
and end-to-end timings in the packet. End-to-end includes host
binding/staging and synchronization. Repeated micro-shape runs show noticeable
variance, so the measurements are diagnostic only; no selector or performance
promotion follows. See
[packet](../../../../benchmarks/baselines/nvidia_sm120_nvfp4_scheduled_20260930/README.md).

Apple, ROCm, and x86 require follow-up to verify unsupported-target refusal or
add their own lowering; the shared Graph NVFP4 type/serializer does not confer
physical schedule or execution parity. NVIDIA evidence does not transfer.

## `E2E-REAL-6-GFX1201-BF16-NORM-MATMUL-2026-09`: sibling assessment

Owner E2E-REAL-6; sync `E2E-REAL-6-GFX1201-BF16-NORM-MATMUL-2026-09`.
Not applicable to the NVIDIA implementation: this adds an architecture-scoped ROCm RMSNorm ABI and gfx1201 Schedule/Tile admission; no CUDA lowering, SM120 ABI, or NVIDIA runtime dispatch changed. The existing SM120 bf16 resident tensor edge remains a separate CUDA implementation. The focused NVIDIA tensor-program suite reran 18/18 on Super-Bear RTX 5070 (sm_120) with its CUDA 13.4 GEMM/PTX libraries selected explicitly; this is a regression check, not parity inferred from ROCm timings or physical schedules.

## `E2E-REAL-6-GFX1201-NORM-MATMUL-2026-09`: paired package edge - parity validated

Owner E2E-REAL-6; sync `E2E-REAL-6-GFX1201-NORM-MATMUL-2026-09`.
Parity validated on exact devices: Super-Bear RTX 5070 (sm_120) and Tajasaurus RX 9070 XT (gfx1201) each ran compiler-owned RMSNorm -> matmul packages with a resident intermediate, ordered same-stream execution, numerical checks, and stable allocation lifetime. Both public `from_text` routes now carry `matmul(output_dtype="fp32")` into Graph IR and compile through native Schedule/Tile packages with fp16 inputs, fp32 accumulation, and fp32 output. The new explicit-output exact-device test passes on each owner; Super-Bear's tensor-program file passes 19/19. NVIDIA additionally retains fp16/bf16 input/output coverage and bounded dynamic-N reuse (N=7 and N=16 with static M/K). The gfx1201 correctness-gated 128x256x256 public-route packet records separate 100-trial HIP-event medians of 11.20 us for RMSNorm and 14.40 us for matmul; timings varied across reruns and are diagnostic only, with no promotion. The combined gfx1201 host-free/exact-device compiler, frontend, registry, and audit lane passed 456 tests before two added direct API cases; those cases separately passed with the public frontend checks. The paired targets use distinct schedules and binaries; no physical schedule is transferred. The refreshed Super-Bear packet `benchmarks/baselines/sm120_rmsnorm_matmul_edge_20260930/` records correctness-gated static-N and bounded dynamic-N runs: producer/consumer medians 54.3/9.9 us and 54.1/8.9 us respectively; short-run variance is diagnostic and no selector promotion is made. The expanded focused cross-contract lane passed 357 tests with 19 skips, including active SM120 device rows. gfx1201 bounded dynamic-M proof is architecture-specific and does not transfer to sm_120; NVIDIA's own bounded dynamic-N resident route remains validated, while equivalent dynamic-M reuse remains follow-up required. Apple and x86 owning-device parity for the shared output dtype request remains follow-up required.

## `SM120-DYNAMIC-MATMUL-EDGE-2026-09-29`: bounded resident package reuse

Owner E2E-REAL-6; sync `SM120-DYNAMIC-MATMUL-EDGE-2026-09-29`.
Exact Super-Bear RTX 5070 proof: two canonical bounded-dynamic Schedule/Tile
matmul packages were reused for producer/consumer chains at 17x19x13x11 and
9x5x7x6, with the intermediate held in a caller-owned CUDA allocation and
both packages submitted on one explicit stream. Both stages matched NumPy
after synchronization. The resident launch now accepts the six-scalar
M/N/K/LDA/LDB/LDD ABI, and the CUDA buffer wrapper preserves strided contract
labels while retaining row/column storage order. Seven CUDA-event batches of
80 invocations measured 386.07 us producer and 387.41 us consumer median per
launch; these stream intervals include Python enqueue gaps and are not isolated
kernel timings or promotion evidence. Apple, ROCm, and x86 are not applicable:
the changed resident ABI, PTX bridge, and buffer wrapper are NVIDIA CUDA
specific; no shared Graph/Schedule operation or physical schedule changed.
The earlier W1.1 source trace cited `tests/tessera-ir/phase2/full_pipeline.mlir`,
which is an x86 fixture; that trace did not establish the SM120 path and is
superseded by `test_sm120_graph_matmul_k_loop_carries_typed_fragment_accumulator`.
A direct four-panel Graph matmul through `tessera-nvidia-pipeline-sm120` emits
Schedule-owned `tile.view` A/B operands, typed A/B `fragment_pack`, a
`fragment_zero` accumulator carried by `scf.for`, typed `tile.mma` consuming
the carried value, and one `fragment_unpack`; it emits neither generic
`tile.async_copy` nor `tile.tma.copy_async`. This proves the canonical SM120
Graph matmul producer and multi-panel accumulator route. The generic
`LowerMatmulToTileMMA` and `LowerKReductionAddToTileMMA` constructors remain
for legacy generic pipelines; their use by a separate SM120 producer route is
not established. The typed Schedule route is covered by exact-device resident
producer/consumer evidence above. No cross-backend or selector claim follows.

# NVIDIA compiler test-suite evaluation and rearchitecture

## NVIDIA-ATTENTION-LSE-BACKWARD-2026-10-01: native forward, saved LSE, and backward proof — landing

Owner E2E-REAL-6; sync NVIDIA-ATTENTION-LSE-BACKWARD-2026-10-01.
Super-Bear RTX 5070 (sm_120) passes the dedicated paired-checkpoint device test
for saved and recomputed LSE forward outputs and backward gradients against an
independent oracle. The correctness-gated Graph-to-native forward benchmark
covers full/causal attention, fp16/fp32 storage, regular and ragged extents.
The fresh recheck passes all eight outputs against an independent fp64
reference; max absolute error is 5.96e-8. Its device medians span 25.18–33.01 us;
5/8 rows meet the 3% device stability policy, and 6/8 meet the end-to-end
policy. The earlier packet had lower timing variance, so neither run supports a
speedup or selector claim. Both packets and the exact-device checkpoint test
are summarized in the
[evidence report](../../../../benchmarks/baselines/nvidia_attention_lse_e2e_20261001/README.md).
The producer migration under W1.1 remains open; this consumer route does not
close the two generic tensor-valued C++ tile.mma producers.


## E2E-REAL-6-ALIBI-SHAPE-2026-09-29: traced ALiBi result type

Owner E2E-REAL-6; sync E2E-REAL-6-ALIBI-SHAPE-2026-09-29.
Follow-up required: the shared Graph shape contract now yields f32 [H,S,S] for a traced explicit-slopes call. No sm_120 typed Tile/Target or exact-device proof follows from the x86 route.

## E2E-REAL-6-ROCM-MATMUL-IMAGE-2026-09-29: bounded image identity

Owner E2E-REAL-6; sync E2E-REAL-6-ROCM-MATMUL-IMAGE-2026-09-29.
Not applicable to sm_120 images: this change keys only gfx1151 HSACO matmul directives and adds a ROCm-specific diagnostic packet. No shared Graph IR, ABI, dtype, PTX route, or NVIDIA fragment producer changed.

[Packet](../../../../benchmarks/baselines/gfx1151_matmul_shape_key_20260929/README.md).


## E2E-REAL-6-ALIBI-2026-09-29: shared Graph operand — follow-up required

Owner E2E-REAL-6; sync E2E-REAL-6-ALIBI-2026-09-29.
Graph ALiBi now declares an optional rank-one f32 slopes operand and checks
its explicit-output shape. The sm_120 backend has no new typed Tile/Target
consumer or exact-device result for this form. The attention and fragment
producer obligations are unchanged; x86 proof does not transfer.

## `CI-LLVM-EXACT-2026-09-29`: hosted compiler toolchain pin

Owner compiler foundation F0; sync `CI-LLVM-EXACT-2026-09-29`.
Portable NVIDIA MLIR lit coverage now uses exact LLVM/MLIR 23.1.1, matching Super-Bear's fleet pin. It does not replace sm_120 device proof or close the two tensor-valued fragment producers.

[Compiler log](../../compiler/INTEGRATED_COMPILER_LOG.md).

## `COMPILER-MATH-RESIDUAL-NEXT-2026-09-29`: split residual ABI

Owners AD-RESIDUAL-EVAL-1 / EVIDENCE-PACKET-1 / W1.1.
Shared CUDA/HIP persistent tape validation now binds residual-source identity
and exported slots across both products. Exact-device nested SAVE tape recording passes on Super-Bear sm_120 at
widths 4, 8, and 16, including repeated backward and mutation controls.
The two tensor-valued `tile.mma` producers remain open under W1.1. The gfx1151
math measurements are not NVIDIA evidence.

[Evidence](../../../../benchmarks/baselines/compiler_math_residual_next_20260929/README.md).


## `COMPILER-EVIDENCE-FRAGMENT-RESIDUAL-2026-09-29`: evidence, fragments and residuals

Owners EVIDENCE-PACKET-1 / W1.1 / FRONTEND-IR-MEDIUM-1 / AD-RESIDUAL-EVAL-1.
Super-Bear sm_120: the legacy tensor MMA now refuses before an invalid async-token target op is formed; the typed accumulator fixture retains FileCheck. The two tensor-to-fragment producers remain open. The public coupled residual passes native tape capture, mutation isolation and analytic repeated backward. ROCm physical-math packaging is not applicable to sm_120.
No selector or performance promotion.

[Evidence](../../../../benchmarks/baselines/compiler_evidence_fragment_residual_20260929/README.md).

## `FRONTEND-RESIDUAL-FRAGMENT-2026-09-28`: traced residual and typed-loop proof

Owners FRONTEND-IR-MEDIUM-1 / AD-RESIDUAL-EVAL-1 / W1.1.
Parity validated on Super-Bear sm_120: public traced cubic-residual native
AD products preserve input snapshots across mutation and repeated backward.
Four typed accumulator-loop rows (zero through four K panels) pass NumPy;
the existing nine fragment tests and FileCheck pass. The two tensor-valued
TileIRLoweringPass producers remain follow-up required.
No selector or performance promotion.

[Evidence](../../../../benchmarks/baselines/frontend_residual_fragments_20260928/README.md).

## `EVIDENCE-MATH-PACKAGES-2026-09-28`: serialized x86 consumers

Owner EVIDENCE-PACKET-1; sync `EVIDENCE-MATH-PACKAGES-2026-09-28`.
Not applicable to NVIDIA code generation: this slice changes the x86 math
benchmark consumer only. No sm_120 fragment producer or device ABI changed.
NVIDIA fragment-producer closure remains the next architecture slice.
No promotion eligibility is granted.

[Evidence](../../../../benchmarks/baselines/evidence_math_packages_20260928/README.md).

## `COMPILER-NEXT-SLICES-2026-09-28`: implementation and exact-device loops

Owner E2E-REAL-6; shared synchronization key `COMPILER-NEXT-SLICES-2026-09-28`.

Super-Bear RTX 5070 sm_120 rebuilt merged main and passed 25 packed/state
Schedule replay tests. Paged-KV numerical checks passed for all three recorder
envelopes in two runs. Increasing to 41 samples / 300 device repetitions did
not stabilize every row; only canonical 2048-token timing met the gate in
both runs. Raw unstable results are retained; no selector change.
Attention LSE/backward and nvfp4/int4/mx migrations remain open. Shared rank-3
linalg Graph verification requires NVIDIA physical-consumer follow-up; no
rank-3 device proof. ROCm attention code-identity changes are not applicable
to NVIDIA code generation.

[Evidence packet](../../../../benchmarks/baselines/compiler_next_slices_20260928/README.md).

## `E2E-REAL-6-x86-kernel-2026-09-28`: x86 native Schedule contract — sibling outcome — follow-up required

Owner E2E-REAL-6 ([x86 queue](../x86/todo.md)); sync
`E2E-REAL-6-x86-kernel-2026-09-28`. The 45 new Graph ODS declarations
have target-neutral verifiers; `schedule.norm` additionally admits Zen 5
without changing the sm_120 branch. Princess-Luna host drift gates passed.
No Super-Bear exact-device run has been made for this branch, so sm_120
parser and execution parity remain a follow-up.

## `E2E-REAL-6-GFX1151-PAGED-2026-09-28`: sibling outcome — follow-up required

The shared native paged-KV Schedule producer now also admits `rocm_gfx1151`;
its existing sm_120 branch and `nvvm.kernel` marking are unchanged. The ROCm
image-symbol and shape-free cache fix is backend-specific. The shared
serialized-Schedule consumer passed host-free regression in Princess-Luna
WSL; sm_120 exact-device evidence is still needed before a parity claim.

## `E2E-REAL-6-APPLE-X86-2026-09-28`: sibling outcome — follow-up required

The shared `tessera.trunc` MLIR declaration/shape verifier adds no sm_120
Schedule, PTX or runtime ABI. Apple's checked Graph division and f32 rope
entry are Metal-only, and x86 trunc's AVX-512 policy does not transfer.
Attention with LSE/backward, paged KV and quantized matmul E2E-REAL-6
migrations still need their own native Schedule/Tile and Super-Bear proof.

## `GFX1201-W8A8-M200-SHORTK-2026-09-28`: sibling outcome — not applicable

The native Schedule change selects a gfx1201-only FP8 W8A8 LDS panel at a
bounded M/N/K envelope. NVIDIA sm_120 has no ROCm LDS or WMMA consumer for
that physical contract; its Schedule, Target IR and runtime ABI are unchanged.
The Tajasarus performance result is not sm_120 evidence.

## `GFX1201-W8A8-RAGGED97-2026-09-28`: sibling outcome — not applicable

The new Schedule panel rule is gated to gfx1201 FP8 W8A8 with NK weights and
only changes ROCm LDS tiling. NVIDIA sm_120 Schedule selection, fragments,
numerical policy and runtime ABI are unchanged. No sm_120 performance or
device parity is inferred from the Tajasarus packet.

## `E2E-REAL-6-SM120-SOFTMAX-SAFE-2026-09-28`: scheduled admission and device rows — landing

Owner E2E-REAL-6. `tessera.softmax_safe` is now admitted to the existing
sm_120 stable row-softmax Schedule/Tile consumer for shape-preserving,
last-axis fp16/bf16/fp32 inputs and matching output storage. The scheduled
producer canonicalizes it to `tessera.softmax`; the compiled PTX and ABI
remain the existing softmax family. Focused admission and exact-device
parity/oracle rows were added for all three storages, including driver
packaging. Super-Bear's RTX 5070 (sm_120) passed 40/40 focused scheduled
unit rows and 14/14 focused device softmax rows, including the new safe
route. This is exact-device functional proof; no new performance claim is
made. Shared scheduled-kernel contract
assessment: Apple, ROCm and x86 outcomes are recorded in their plans under
this sync key.

## `CI-LIT-EBM-CLIFFORD-2026-09-28`: hosted lit lane covers the EBM / Clifford fixtures — sibling outcome — parity validated (host-free IR only)

Owner: CI toolchain lanes (PR #874, follow-up to #873 item 1). The hosted `lit` lane now configures `TESSERA_BUILD_{EBM,CLIFFORD}_BACKEND=ON`, so the six fixtures that `REQUIRES: tessera-ebm` / `tessera-clifford` run there and the fleet-union gate passes on one lane: dispatched run 36418280808, LLVM/MLIR 23.1.2 under the CI 23.1.x tolerance (fleet pin 23.1.1), 533/533 passed, `uncovered: []`. Every one of the six already passed on every fleet box, which configures both backends ON; this adds hosted-runner coverage and makes no device claim. `phase2_autodiff/row_program_to_gpu_langevin.mlir` drives `tessera-row-program-to-gpu=backend=nvidia`, so the NVIDIA row-program emitter's IR now also runs on a hosted runner. It is IR/FileCheck only: no sm_120 execution is implied, and the RTX 5070 Langevin device rows are unchanged.

## `ODS-WIRE-1-4-2026-09-27`: sibling outcome — not applicable

Owner GOV-ODS-CONSUMER-1 (ODS triage WIRE slices 1 and 4). The `target_verify -> softmax` / `ntk_rope -> rope(x, theta/s)` rewrite runs inside `tessera-canonicalize`, so every `tessera-lower-to-{gpu,nvidia-sm90,-sm100,-sm120}` pipeline now normalizes the two composites; those pipelines have no Graph softmax or rope consumer (the sm_120 families lower through their scheduled contracts), so nothing downstream changes and nothing was run on Super-Bear. No CUDA Langevin Philox executor exists (the CUDA EBM lane is the geo Langevin family).

## `ODS-WIRE-B-2026-09-28`: the ISTFT forward product builds from compiler IR — sibling outcome: parity validated (sm_120, 2026-09-28)

The native JVP plugin now builds every ISTFT package, including
`nvidia_sm120`, from the hashed contract `GraphToSchedulePass` binds to
`tessera.istft_jvp` (triage row `tessera-istft-jvp`). The host-free
differential covers the sm120 profile (contract and oracle lower to the same
scheduled program). **Device proof, 2026-09-28, The-Super-Bear RTX 5070**
(umbrella `claude/foundation-batch-2` @ `e4ba9496`, `_nvidia_env.sh` sourced,
worktree `build/` sm_120 + `build-nvidia-cuda/` sm_120a):
`tests/device/nvidia/test_spectral_jvp.py` 13 passed / 0 skipped and
`test_spectral_stale_cuda_error.py` 9 passed / 0 skipped, identical to the
pre-change base `6008a942`; `tests/unit/test_istft_jvp_ir_contract.py` 51
passed / 4 skipped (AVX-512 and gfx1151/gfx1201 gates only). The ISTFT
centered-difference row's inputs re-run through `native_jvp` reported
`execution_kind=native_gpu`, `evidence_target=nvidia_sm120`,
`compiler_path=nvidia_sm120_jvp_compiled`, and the package was built by one
call to `istft_jvp_contract_from_paired_ir(target=nvidia_sm120)` (schema
`tessera.spectral_jvp.v1`) -- the IR contract, not the kwargs oracle.
`test_native_jvp_compiled.py` has no sm120 row (all 16 skips are x86/ROCm
gates), so it is no evidence either way. The KV-cache cursor half of the key
is x86-only.


## `E2E-REAL-6-rocm-unary-2026-09-27`: sibling outcome — follow-up required (softmax_safe admission)

ROCm moved its gfx1151 softmax/reduction packagers onto the native Schedule
contract (ROCm queue, same key). NVIDIA's unary family already made that move
on 2026-09-05; no NVIDIA code or packet changed, and `supports_scheduled_kernel`
for `nvidia_sm120` is unchanged (host-free tests pass on the Mac). Found while
checking: `nvidia_native.native_package_kind` classifies `tessera.softmax_safe`
as `softmax`, but the scheduled contract refuses it, so a `softmax_safe` module
defaults to `package_native` and then fails to package instead of taking another
route. ROCm now canonicalizes `softmax_safe` to `softmax` in
`lower_scheduled_kernel`, gated to gfx1151; admitting it for sm_120 needs
sm_120 device rows on The-Super-Bear.
## `AUTOTUNE-LAUNCH-INTEGRITY-2026-09-27`: checked emitted launches, route resources, reproducible shipped GEMM, route identities; sm_120 rows re-recorded

Four items, one change, because each one moves the same sm_120 corpus rows and
one re-record covers them. Evidence and commands:
[`benchmarks/baselines/autotune_corpus_rerecord_sm120_launch_integrity_20260927/`](../../../../benchmarks/baselines/autotune_corpus_rerecord_sm120_launch_integrity_20260927/README.md).

**1. `NVIDIA-EMITTED-UNCHECKED-LAUNCH` — closed.** All 55 sync-only entries,
and the two inline sources (now the emitters `_synthesize_relu_bias_cuda` /
`_synthesize_fused_epilogue_cuda`), read the last-error slot after each launch
group, clear it once on entry, and consume every allocation / copy / memset /
event status; the raced lanes' unchecked H2D/D2H copies are checked too. The
ReplaySSM ring runtime launches through the driver and checks each
`CUresult`, so it is the one `status-checked` source. The host-independent
gate (`tests/unit/test_nvidia_emitted_stale_error_rule.py` +
`tests/_support/nvidia_emitted_sources.py`) no longer accepts `sync-only`: it
parses every host function and requires a slot read after the last launch, a
read before a timer's timing boundary, no `cudaFuncSetAttribute` between a
launch and its read, and every status in `STATUS_CALLS` consumed; no kernel
source may live outside an emitter. Device proof on The-Super-Bear
(`tests/device/nvidia/test_emitted_unchecked_launch.py`, 17 lanes: flash
fwd/bwd, softmax, norm, reduce + timer, linear attention, MoE, paged-KV read +
timer, relu-bias, fused/gated epilogue + timer, conv2d, rope, control-for):
with every launch made invalid (grid `dim3(0u)`) the old sync-only judgment
returned success on all 17 (the defect, observed), the shipped entries raise on
all 17, and the unmodified lanes run; 17 + the 19
`test_emitted_stale_cuda_error.py` cases passed. All 64 distinct rendered
sources compile with the lane's own nvcc line. ROCm half: see the ROCm queue.

**2. `AUTOTUNE-SM120-ROUTE-RESOURCES` — closed.** The route-resource manifest
attests Nsight Compute launch facts per route (registers, shared memory,
theoretical/achieved occupancy, spill requests; `parse_ncu_resources.py`); the
finalizer admits a stable winner only when its route has an entry. The 17
missing routes (native tf32/fp8 fused/attention/gated, composed fp8, and the
scalar `nvidia_flash_attn` / `nvidia_gated`) were captured by the same method,
one route per `ncu --set full` report so same-named tf32/fp8 builds stay apart
(`profile_route_resources.py` brackets one timed invocation with
`cuProfilerStart/Stop`; `build_test5_resource_manifest.py --route`). After the
re-record 91 registry rows are selector-eligible (was 38). Of the 20 formerly
partial rows **4 are now served** (attention fp8_e5m2 device 128x128, gated tf32
and fp8_e5m2 device rows); 15 stay out as unseparated (device-event noise
33–332% between repeats) and 1 because the two runs disagreed. None is blocked
by missing resources any more.

**3. Shipped GEMM byte-reproducibility — closed.** Root cause measured on
Super-Bear: tree paths never mattered (two worktrees, same configure, identical
bytes); the CUDA discovery path did. Linking the imported `CUDA::nvrtc` wrote
the toolkit directory CMake found into DT_RUNPATH (`/usr/local/cuda/lib64` with
`CUDAToolkit_ROOT=/usr/local/cuda` — reproducing the old `a00c9040…` exactly —
vs `/usr/local/cuda-13.4/targets/x86_64-linux/lib`); `.text`/`.rodata`/`.data`
were identical, only `.dynstr`/`.dynamic`/build-id differed. The library now
builds with `SKIP_BUILD_RPATH` (every loader preloads the driver and NVRTC);
four builds across two worktrees and both configures are byte-identical
(`198b85b3…`). Identity scheme unchanged; the committed rows serve with the
library built in another worktree under the other configure (96/96 match, 20
served).

**4. `AUTOTUNE-KERNEL-IDENTITY-PAGED-KV` — closed (sm_120 half).** Non-registry
rows get the registry contract (`autotune.RouteIdentity`, `route_identities`,
`route_record_matches`, fail closed). sm_120: the paged-attention routes are the
resident-stage source plus the entries each launches, and
`_paged_attention_corpus_winner` refuses a row whose live identities differ;
`conv2d` rows stamp `direct`/`shared` (resident stages) and `im2col_tf32`
(resident stages + the shipped GEMM tf32 device entry) — no production path
reads them yet, stated; `ssm_replay_decode` rows stamp the `async_ring` (ring
runtime source + the `tessera-nvidia-opt` decode/flush images) — no reader
either. The ring runtime and the paged HIP artifacts are cached by source
content now.

**Re-record (The-Super-Bear, own worktree at `fcfa3677`, fresh `build/` +
`build-nvidia-cuda/`, everything under the timing lock).** Two recorder runs
and the finalizer (rc 0, identities agreed), the serving recorder, then the
order restored: 108 sm_120 rows changed, no key lost or added, every timed
candidate of all 124 rows stamped. **Served registry rows: 20 (was 13)**; the
two `matmul` end-to-end 2048³ shipped-GEMM rows dropped out (winner unchanged,
now unseparated). Route rows match 12/12; the paged-KV warm start serves the
128-token device row. Every one of the 108 rows misses under some single
emitter perturbation with pins unchanged, and all 108 with all perturbed. 11
winners changed, none served. Reproducibility: 92/92 strict records admitted.

**NVIDIA release gate, device layer, at `71e1e81e`** (same box and worktree,
under the timing lock, `TESSERA_NVIDIA_REPORT_DIR=~/gate-reports/a-71e1e81ea`):
both device-correctness passes **1167 passed, 1 skipped, 0 failed** (junit:
1168 tests, 0 failures, 0 errors, 1 skipped each), `status=success`. The skip
is NCCL not installed (multi-rank topology lane not evaluable here). The 17
new `test_emitted_unchecked_launch.py` cases are included. After the gate's
re-configure the bridge (`1bcb4967…`) and `build/`'s GEMM (`198b85b3…`) are
unchanged.

Open, found here: the `ncu`-only exit abort of a process holding a
generic-lane library (recorded in the evidence README, not root-caused);
route-resource entries are keyed by route, not by code identity or storage.
## `SMALL-CORRECTNESS-GAPS-2026-09-27`: `test_tma_smoke` root-caused and passing on sm_120

The TMA smoke (`src/compiler/codegen/tessera_gpu_backend_NVIDIA/`
`src/kernels/tma_smoke.cu`, one rank-1 f32 box of 32) had **two** defects; the
first hid the second.

1. **Encode: `globalStrides == nullptr`.** For a rank-1 map `globalStrides`
   has `tensorRank - 1 == 0` entries and the CUDA 13.4 `cuda.h` comment states
   no requirement on the pointer, but driver 610.88 (`cuDriverGetVersion`
   13030) returns `CUDA_ERROR_INVALID_VALUE` for a null pointer and ignores the
   contents. Probe matrix on the RTX 5070 (every other documented
   precondition held: `CUtensorMap` 64-byte aligned (alignof 128), global
   address 256-byte aligned, `boxDim[0] * 4 = 128` a multiple of 16, rank 1,
   interleave/swizzle NONE, `elementStrides` 1): rank-1 with `nullptr` fails
   for f32 dim 32 / box 32, f32 dim 1024 / box 32 and u8 dim 128 / box 128,
   also through `cudaGetDriverEntryPointByVersion(..., 12000)`; rank-1 with a
   non-null array succeeds whether it holds 128, 0, 7 or 2^41; rank-2 with a
   real stride succeeds. Fix: pass `{globalDim[0] * sizeof(float)}`.
2. **Launch: the descriptor was read from local memory.** With the encode
   fixed, the launch failed with `an illegal memory access`. The kernel took
   `CUtensorMap` by value without `__grid_constant__`; the PTX shows nvcc
   copying the parameter into `__local_depot0` (eight `st.local.v2.b64`) and
   handing `cp.async.bulk.tensor` the generic address of that copy, while PTX
   requires the tensor-map operand in `.param`, `.const` or `.global`. With
   `const __grid_constant__` the PTX has no local depot, the instruction
   addresses the parameter directly, and the smoke passes. `compute-sanitizer`
   cannot attach under WSL2 ("Failed to initialize WDDM debugger interface"),
   so the fault's cause rests on the PTX difference plus the fault
   disappearing with exactly that change, not on a sanitizer report.

Also added `fence.proxy.async.shared::cta` after `mbarrier.init` (the CUDA
programming guide's TMA pattern). It is not shown necessary here: the smoke
passed 6/6 with `__grid_constant__` alone.

Evidence (The-Super-Bear, RTX 5070, CUDA 13.4.59 / driver 610.88, own
worktree, `flock /tmp/tessera-timing.lock` around every device run): the
unmodified `build-nvidia-cuda` binary fails (`cuTensorMapEncodeTiled: invalid
argument`); strides-only fails (`illegal memory access`); the fixed source
passes 3/3 at `-arch=sm_120a` and 3/3 at `sm_120` standalone, and the
CMake-built `test_tma_smoke` (fresh tree, `TESSERA_CUDA_ARCH=sm_120a`) passes
5/5 and compares all 32 values. `test_tma_smoke` is a standalone executable
with no ctest/pytest wrapper, so no automated lane runs it; that is unchanged.

Sibling outcome: ROCm not applicable (no TMA; `cuTensorMap*` is CUDA-only);
Apple not applicable; x86 not applicable.

## `TILE-LATENT-DEFECTS-2026-09-27`: Tile TMEM lowering fails closed and wires its results; the sm_120 dashboard stops citing fixture-only Target ops

Owner: GOV-ODS-CONSUMER-1 (the ODS connection triage found these in passing).
IR/lowering evidence only: TMEM is datacenter sm_100, which no fleet box has,
so nothing here is an execution claim. Verified under the assertions-ON
`tessera-nvidia-opt` / `tessera-opt` on Tajasarus (LLVM/MLIR 23.1.1,
`--assertion-mode ON`, fresh trees at `f9023d62`): NVIDIA backend lit 68/68,
`lit tests/tessera-ir` 458 passed / 66 unsupported / 0 failed,
`check-tessera-rocm` 82/82. The same new fixtures against the unfixed
`origin/main` build (`d8da67f7`) on that box: four TMEM fixtures abort with
`LLVM ERROR: operation destroyed but still has uses`, the unknown-op fixture
exits 0 emitting `tessera_nvidia.tmem_store` for both ops, and
`nvidia_marker_result_used.mlir` fails with `null operand found`.

**Fixed in `LowerTileToNVIDIA` (`NVIDIALowering.cpp`, `lowerTmemOp`).**

- The branch matched `starts_with("tile.tmem.")` and defaulted anything that
  was not alloc/load to a `tessera_nvidia.tmem_store` contract, so a new or
  misspelled op silently became a store. The three registered ops
  (`tile.tmem.allocate` / `load` / `store`, `TileOps.td`) now map by op
  identity; anything else under the prefix fails with
  `NVIDIA_TMEM_UNKNOWN_OP` (Decision #21). The unregistered legacy spelling
  `tile.tmem.alloc` is no longer accepted as an alias.
- Every op was erased without replacing its results. On the unfixed
  assertions-ON driver, every new fixture with a used handle or load result
  aborts with `LLVM ERROR: operation destroyed but still has uses`. Now the
  `!tile.tmem` handle lowers to the i32 tensor-memory address
  (`tessera_nvidia.tmem_alloc ... -> i32`, the `[taddr]` that `tcgen05.ld/st`
  take), `index` operands are widened to i64 (neither type is an NVIDIA target
  value), load results are replaced, and the allocation is erased only once it
  has no users.
- A handle consumed by an op this pass does not lower (today
  `tile.tcgen05.mma`) is refused with `NVIDIA_TMEM_HANDLE_UNLOWERED` rather
  than erased under a live use.

**Fixed in `LowerNVIDIAToNVVM`.** A contract with no value-producing NVVM
lowering becomes a void marker, and the pass called `dropAllUses()`. A result
used outside the contract family left its user with a null operand (unfixed:
`error: null operand found` on `func.return`). It now fails with
`NVIDIA_MARKER_RESULT_USED`. Uses by other contracts, or by ops nested inside
one, are erased with them as before.

Fixtures (`src/compiler/codegen/tessera_gpu_backend_NVIDIA/test/nvidia/`):
`tmem_tile_to_nvidia.mlir` (used load result and handle),
`tmem_unknown_op_rejected.mlir` (legacy spelling + invented op),
`tmem_handle_unlowered.mlir` (`tcgen05.mma` consumer),
`tmem_to_nvvm_contract.mlir` / `tmem_to_nvvm_result_used.mlir`, and
`nvidia_marker_result_used.mlir` (the NVVM stage alone).

**Dashboard correction (`SM120_DIFFERENTIATION_DASHBOARD.md`).** The four
promoted rows' “Typed IR + verifier” cells cited
`sm120_differentiation_target_ir.mlir` / `tessera_nvidia.fpquant`. No compiler
path produces `mma_fused`, `mma_attention` or `fpquant`; the lanes execute
through Python candidates that bypass Target IR. The cells now say
fixture-only, and the statuses now read **runtime-promoted … Target-IR column
open** (the dashboard's own rule requires all six columns). Runtime,
provenance and benchmark evidence is unchanged.
`test_nvidia_sm120_promotion_gate.py` asserts the honest state.

Open (not in this change):

- `tile.tcgen05.mma` has no NVIDIA lowering, so a TMEM handle that feeds it
  cannot lower yet. It is the next sm_100 slice, and it needs no hardware for
  its IR half.
- The NVVM stage lowers TMEM contracts to void markers only. A real
  `tcgen05.alloc/ld/st` emission is sm_100 work and is hardware-gated for
  execution.
- WIRE slice 7 of the ODS triage: producers for `mma_fused` /
  `mma_attention` / `fpquant` so the dashboard's Target-IR column can close.

## `SPECTRAL-STALE-HIP-ERROR-2026-09-27`: sibling outcome — hand-written hooks fixed on sm_120; emitted templates follow-up required

**Update 2026-09-27 (same branch, PR #862) — reproduced and fixed on the RTX
5070 for the hand-written hooks.** Both hook libraries link the shared
`libcudart.so.13` (`nm -D`: `cudaGetLastError@libcudart.so.13`, undefined), so
the slot is the process's per-thread one and any runtime caller on the thread
can leave an error there. With `cudaSetDevice(97)` (returns 101, proven unread
by `cudaPeekAtLastError`) before each call, the pre-fix libraries fail every
checked path: host C2C inverse `rc=3`, device-pointer C2C inverse / C2R `rc=3`,
DCT `rc=292`, STFT `rc=306`, streaming STFT `rc=306`, STFT JVP `rc=365`,
Philox `rc=3`.

Rule, as on ROCm: each exported entry that does device work calls
`clearStaleCudaError()` once, first thing, and never between a launch and its
check (several checks are grouped: `istft_backward_broadcast_layout_f32` reads
the slot once after three launches). Sticky device-fault errors are not reset
by `cudaGetLastError()` and still fail the next call.

| Library / file | Entry | Clears | Reason when not |
|---|---|---|---|
| `tessera_nvidia_spectral.cu` | `dct_policy_layout_f32`, `stft_policy_broadcast_layout_f32`, `stft_jvp_broadcast_layout_f32`, `istft_policy_broadcast_layout_f32`, `istft_jvp_broadcast_layout_f32`, `stft_backward_broadcast_layout_f32`, `istft_backward_broadcast_layout_f32`, `spectral_conv_f32` | yes (8) | — |
| | `spectral_package_abi`, `spectral_arch` | no | metadata; `spectral_arch` returns its own calls' statuses and never reads the slot |
| | the seven `*_storage` wrappers, `streaming_stft_broadcast_layout_f32` | no | host packing, then a call to a clearing `_f32` entry with no CUDA call before it |
| `tessera_nvidia_fft.cu` | `execute_{c2c,r2c,c2r}_f32` (host pointers) | yes (3) | — (`r2c` reads no slot today; cleared by the rule so a later launch check cannot inherit a stale error) |
| | `execute_{c2c,r2c,c2r}_device_f32` (device pointers) | yes (3) | differs from ROCm on purpose: nothing in this library composes them after unchecked launches of its own; their callers are ctypes, which read every status directly. A future C++ composer would own the slot and take the clear |
| | `package_abi`, `current_device`, `plan_create_{c2c,r2c,c2r}_f32`, `plan_destroy`, `workspace_alloc`, `workspace_free` | no | metadata / plan and workspace lifecycle: each returns its own call's status and never reads the slot |
| `tessera_nvidia_rng.cu` | `philox_{uniform,uniform_range,normal,dropout}_f32` | yes (4) | — |
| `src/kernels/{mbarrier,tma}_smoke.cu` | `tessera_mbarrier_smoke`, `tessera_tma_smoke` | yes (2) | — (standalone smoke executables; same rule) |

Autotune identity: neither library feeds one. No candidate's
`delegate_identity`/`loaded_library_identity` binds `libtessera_nvidia_fft` or
`libtessera_nvidia_rng` (the only NVIDIA delegate digest is
`libtessera_nvidia_gemm`, untouched), and no committed packet pins their
digest, so rebuilding them unserves no corpus row.

Evidence: `tests/device/nvidia/test_spectral_stale_cuda_error.py` wraps every
checked entry to prime the slot before each call and requires each public
path to succeed and to have reached the entries it names — 9/9 fail on the
pre-fix libraries, 9/9 pass fixed (The-Super-Bear, own worktree and
`build-nvidia-cuda`, loaded `.so` paths verified). Counts and the release-gate
report are in the compiler log entry for this sync key.

**Still open — emitted CUDA templates (follow-up required).** After #861
merges, apply the entry-clear rule to the emitted CUDA templates in
`python/tessera/compiler/emit/nvidia_cuda.py` (no entry clear before its
post-launch `cudaGetLastError` checks) together with an sm_120 corpus
re-record. Not done here because #861 makes an emitted lane's autotune
identity the digest of its emitted source: editing the templates would change
every NVIDIA emitted-lane identity and silently unserve the sm_120 corpus rows
just re-recorded. (#861 merged while this branch was open; the follow-up is now
unblocked but still owes the re-record.)
`benchmarks/nvidia/record_shared_arena_rematerialization.py` embeds one more
such source (benchmark harness, same follow-up).

**Separate, pre-existing, not investigated:** `test_tma_smoke` fails on the RTX
5070 with `cuTensorMapEncodeTiled: invalid argument` both before and after this
change (the encode is a driver-API call ahead of any launch, so the entry clear
cannot affect it). Owed: root-cause the descriptor (rank-1 f32, box 32).
**Resolved 2026-09-27** (`SMALL-CORRECTNESS-GAPS-2026-09-27`, top of this
queue): a null rank-1 `globalStrides` the driver rejects, then a by-value
descriptor read from local memory.

The ROCm spectral image checked its launches with `hipGetLastError()`, whose
per-thread slot is sticky: an unrelated failed HIP call earlier on the thread
(the probe used `hipSetDevice(97)`) made the next correct launch return
failure — `rc=246` from the streaming STFT in the gfx1201 full sweep. The fix
discards errors older than the call once, at the entry of every exported
device-work function, never between launches ([ROCm queue](../rocm/todo.md)).

**Follow-up required here, unproven on sm_120.** The CUDA runtime's
`cudaGetLastError()` has the same read-and-reset contract for non-sticky
errors, and the hand-written hooks follow the same pattern with no entry clear:
`runtime/cuda/tessera_nvidia_spectral.cu` (21 post-launch reads),
`tessera_nvidia_fft.cu` (4), `tessera_nvidia_rng.cu` (2), plus the emitted
entries in `python/tessera/compiler/emit/nvidia_cuda.py` (no `(void)
cudaGetLastError()` anywhere). Owed: reproduce on The-Super-Bear with a primed
slot (`cudaSetDevice` on a missing ordinal, then a streaming STFT), apply the
same per-entry rule, and keep grouped checks intact. Not changed in this PR,
because no sm_120 run backs it. The NVIDIA streaming-STFT label is already
derived (`tessera_nvidia_spectral_arch() == 120`).

## `SM120-AUTOTUNE-FOLLOWUPS-2026-09-27`: emitted-CUDA stale-error rule, scalar-lane device timers, gated dims; sm_120 rows re-recorded

This section closes three items left open by the emitted-identity re-record
below. They are in one change because each one invalidates or extends the
same 96 sm_120 registry rows, so one re-record covers all three. Evidence and
commands:
[`benchmarks/baselines/autotune_corpus_rerecord_sm120_followups_20260927/`](../../../../benchmarks/baselines/autotune_corpus_rerecord_sm120_followups_20260927/README.md).

**1. Emitted CUDA: the stale-error rule (the emitted half of
`SPECTRAL-STALE-HIP-ERROR-2026-09-27`).** The hand-written hooks were fixed
under that key. The templates in `emit/nvidia_cuda.py` were held out of that
change because editing them changes emitted-source identities. Measured on
The-Super-Bear with a standalone probe (CUDA 13.4 / driver 610.88):

- `cudaDeviceSynchronize` and `cudaMemcpy` neither return nor reset a stale
  error.
- After an invalid-configuration launch, `cudaDeviceSynchronize` returns
  **success** and the error sits only in the slot.
- A successful `cudaFuncSetAttribute` **resets** the slot. This is
  undocumented.
- Every emitted `.so` links cudart statically (its `cudaGetLastError` is a
  local `t` symbol), so each library owns a private slot. A stale error there
  comes from an earlier call into the same library: a refused `cudaMalloc` on
  an early-return path, or a launch nobody checked.

Rule as applied:

- In a source that reads the slot, every exported entry that does device work
  clears it once, as its first statement, and nowhere else.
- Entries that make no call do not clear.
- Sources that never read the slot are unchanged.
- The arbiter-raced lanes now also check each launch group through the slot,
  as the ROCm generic lane does, so a launch that never ran can no longer be
  reported, timed or served as a kernel. That was a fail-open before this
  change.

`emit/rocm_hip.py` already clears (verified): every entry that reads the slot
clears it (generic, bench, replay `de`/`fu`/`bl`/`as`). The ROCm
`su`/`paged_kv`/`paged_attention` entries launch without reading it, which is
the same unchecked-launch gap recorded below (ROCm sibling outcome, follow-up).

| Emitted source | Exported entries | Verdict |
|---|---|---|
| `fused`, `attention`, `gated` | host + new `_device_ms` | reads; clears first; each launch group checked |
| `pointwise` | host | reads; clears first; launch checked |
| `mma_fused`, `mma_attn_lowp`, `mma_gated` | host + `_device_ms` | reads; clears first; each launch group checked |
| `mma_attn_16` | host | reads; clears first; launch checked (also masked by its `cudaFuncSetAttribute`) |
| `resident_ops` | 11 `resident_*` device-pointer stages | reads (each checks its own launch); clears first. Callers are Python ctypes and read every status, and no stage runs after an unchecked launch in this library |
| `flash_fwd_multiwarp` (w4/w8) | host, `_timed` | the `_timed` entry reads; both entries clear first (the host launch stays unchecked) |
| `binary`, `solver_ift` | 1 each | reads; clears first |
| `solver_children` | `unary`/`compare`/`where`/`diagonal_cg` | helpers read; each exported entry clears first |
| `ssm_replay_device` | `dp`, `ps` | no call: no clear |
| `ssm_replay_device` | 12 queue entries | no slot read: unchanged |
| `flash_fwd{,_f16}`, `mla_fused`, `flash_bwd{,_f16}` (4+1), `linear_attn` (4 sources), `softmax{,_f16}` (2+2), `norm`, `reduce` (2), `fpquant`, `local_collective`, `optimizer`, `dequant_grouped`, `moe` (4), `deltanet`, `ssm`, `ssm_replay_decode`, `paged_kv_read` (2), `gated_epilogue` (2 per activation), `conv2d_nhwc`, `posenc` (2), `control_flow` (4) | as listed | no slot read: unchanged; **launches unchecked** (open, below) |

Gate: `tests/unit/test_nvidia_emitted_stale_error_rule.py` (host-independent).
It renders every emitter, and every `_synthesize_*` must be listed in
`tests/_support/nvidia_emitted_sources.py`, so a new emitter cannot skip the
gate. It also checks the two inline sources in run functions.

Device proof: `tests/device/nvidia/test_emitted_stale_cuda_error.py`, 19
passed on The-Super-Bear. It primes each library's own private slot through
its local `cudaSetDevice` on a missing ordinal (checked with its
`cudaPeekAtLastError`), for 18 lanes (generic, flash, gated, pointwise, mma
fused/attn/gated host and timers, resident stages, binary, solver ift/unary,
multiwarp timed). It also checks the realistic trigger: a refused 2 TiB
allocation followed by an ordinary launch. Each lane first runs a **negative
control** with the clears stripped from its emitter. That control failed
under the prime for 16 of 18 lanes, as it must. The two mma.sync attention
entries are masked by the `cudaFuncSetAttribute` reset, and the test records
that exception.

**2. `AUTOTUNE-GATED-INFER-DIMS` — closed.** `_infer_dims` maps `A (M,K), Wg
(K,H), Wu (K,H)` to `(M, H, K)`, which is the recorder's order. Malformed
operand sets stay shape-anonymous. `tests/unit/test_autotune_gated_infer_dims.py`
covers two cases: operands built from every committed gated row's workload
shape land in that row's own bucket, and `corpus_winner` without dims serves a
stamped gated row. On the box, the inferred-dims and explicit-dims answers
agree on all 96 rows. No gated row is admissible (see below).

**3. Scalar lanes get a device timer — closed.** `nvidia_generic_cuda`,
`nvidia_flash_attn` and `nvidia_gated` have a `_device_ms` entry in the same
source and artifact that `run` launches, with the same launch configuration
and guards. The timer is CUDA events with operands resident, the method every
other NVIDIA lane's `measure_device_latency` uses. They are timed in device
rows now, not recorded `unmeasured`. The "every live candidate timed" rule is
unchanged. As before, these are selection hints: CUDA events alone never
qualify a promotion (`WSL-TIMING-ADMISSION-2026-09-26`).

**Re-record (The-Super-Bear, clean worktree at `31a58bd4`, fresh `build/` +
`build-nvidia-cuda/`, both recorder runs, the finalizer and the checks under
`flock /tmp/tessera-timing.lock`).**

- The recorder runs and finalize returned rc 0, the two runs agreed on every
  identity, and no key was lost or added. 98 rows changed; the 16 gfx1151 rows
  and the serving rows are byte-identical.
- 0 timed candidates are unstamped. **0 registry rows carry an `unmeasured`
  candidate (was 20).** 38 rows are selector-eligible (was 31); 39/39 are
  strictly admitted.
- **Served with inferred dims: 13 (was 15).** The winners of the served rows
  are unchanged. Two rows dropped out: `matmul` f16 device 256³
  (`nvidia_mma_gemm_emitted`, margin 5.8% vs 9.6% noise) and `attention` f16
  end-to-end 128x128x64x64 (`nvidia_mma_attn`, 19.5% vs 16.4%). Both keep
  their winner and are now unseparated. They were not re-raced to get them
  back.
- All 96 miss under the emitter perturbations with pins unchanged. Every row
  misses under at least one single perturbation, and with all perturbed 96/96
  miss and 0 are served.
- **10 winners changed**, none of them a served row (table in the evidence
  README).
- The PTX bridge is byte-identical to the one the previous re-record stamped.
  The shipped GEMM library is a different build of unchanged source
  (`b2191291…` vs `a00c9040…` in the previous recording tree), so its rows are
  bound to this build.

**NVIDIA release gate, device layer, at `4a275453`** (same box and worktree,
under the timing lock,
`TESSERA_NVIDIA_REPORT_DIR=~/gate-reports/sm120-fu-4a275453`): both device-correctness passes **1141 passed, 1 skipped, 0 failed** (junit `device-correctness-{1,2}.xml`: 1142 tests, 0 failures, 0 errors, 1 skipped each), `status=success`. The skip is NCCL not installed, so the multi-rank topology lane cannot be evaluated here. The 19 new `test_emitted_stale_cuda_error.py` cases are included; the previous gate ran 1122. A follow-up probe on this box showed that a successful `cudaEventRecord`, `cudaEventSynchronize` or `cudaEventElapsedTime` leaves a launch error in the slot, so the timers' read after the end event still sees a failed timed launch.

Open, found here:

- **`AUTOTUNE-SM120-ROUTE-RESOURCES`** (**closed 2026-09-27**, `AUTOTUNE-LAUNCH-INTEGRITY-2026-09-27` at the top of this file): the 20 formerly partial-field rows now
  race every live candidate, and the scalar lanes lost all 20 by 3–750x. They
  are still unserved. 18 have winners (native `nvidia_mma_{attn,fused,gated}_{tf32,fp8_*}`)
  with no entry in `nvidia_sm120_test5_route_resources.json`, so the finalizer
  marks them `selector_eligible: false`. The other 2 are unseparated. Owed:
  resource fingerprints for the native low-precision lanes, and more device
  repeats where a verdict is unseparated.
- **`NVIDIA-EMITTED-UNCHECKED-LAUNCH`** (**closed 2026-09-27**, `AUTOTUNE-LAUNCH-INTEGRITY-2026-09-27`): the sync-only emitted sources in the
  table above judge their launches by `cudaDeviceSynchronize` only. It
  returned success after an invalid-configuration launch on this box, so a
  launch that never ran reports `rc 1`. To fix it, add a post-launch slot read
  to each entry and apply the same clear-first rule, with device proof. Their
  unchecked H2D copies (`cudaMemcpy` status ignored) are the same class of
  gap. In the raced lanes, the new post-launch read now also catches a failed
  copy, because a failing `cudaMemcpy` writes the slot. ROCm has the same gap in
  `su`/`paged_kv`/`paged_attention`.
- The two mma.sync attention entries depend, in practice, on an undocumented
  reset (`cudaFuncSetAttribute`) that masks a stale error. They clear anyway,
  so nothing depends on the reset. Only the negative control cannot be shown
  for them.

## `AUTOTUNE-EMITTED-IDENTITY-2026-09-27`: every sm_120 candidate carries a code identity; sm_120 registry rows re-recorded

Codex review P2 on PR #859: a SYNTHESIZED/EMITTED candidate was served on the
CUDA/PTX/driver/LLVM pins alone, so a changed emitter kept a verdict measured
for its old kernel. Owner decision: every candidate of every tier carries a
workload-specific code identity or the verdict misses (no opt-out). The
mechanism landed on the Mac, host-independent; the sm_120 re-record below ran
on The-Super-Bear the same day.

NVIDIA identities (`compiler/emitted_code_identity.py`, `tessera.emitted_source.v1`):

| Candidates | Identity |
|---|---|
| `nvidia_generic_cuda`, `nvidia_flash_attn`, `nvidia_gated`, `nvidia_pointwise` | emitted CUDA for the region (`build(..., dims=None)`, dims-invariant) + `kernel_cache.cache_key` + `nvcc -arch=<arch> -O3 --shared -Xcompiler -fPIC -lcuda` (the flag list `_nvidia_cuda_compile_fn` uses; nvcc by name, version is the pin) |
| `nvidia_mma_fused_*`, `nvidia_mma_gated_*` | the `_synthesize_mma_*` source `run` and the device timer compile at the default raster the arbiter dispatches, + nvcc flags |
| `nvidia_mma_attn_*` | composite: the mma.sync source **and** the scalar flash source it hands large/sharp workloads to (a data-dependent branch, so both are covered) |
| `nvidia_mma_{fused,attn,gated}_composed_*` | composite: shipped `libtessera_nvidia_gemm` by content + the `_device` entry bound (identified as the shipped delegate is) and the emitted resident-stage CUDA |
| `nvidia_mma_gemm_emitted`, `nvidia_nvfp4_gemm_emitted` | composite: `ptx_emit` PTX (full-line comments/blank lines dropped) + the PTX launch bridge `libtessera_nvidia_ptx_launch` by content (it registers the PTX and computes the grid) |
| `nvidia_tile_matmul_{direct,shared}` | composite: the PTX `tessera-nvidia-opt` -> mlir-opt -> mlir-translate -> llc generates for the schedule/dtype (`_nvidia_tile_matmul_ptx`) + the bridge; a host without the tools misses |
| `nvidia_mma_gemm_shipped`, `nvidia_nvfp4_gemm_shipped` | unchanged (delegate library) |

The bridge and GEMM library identities are content digests, so a rebuild of
either misses every row that raced a lane using it (a false miss at worst).

**Consequence for the committed corpus (history -- superseded by the re-record
below).** The 96 `nvidia:sm_120` registry rows
(`fused_region` / `attention` / `gated_matmul` / `matmul`) stamped only
`nvidia_mma_gemm_shipped`, so under the new rule **none was servable**: the 14
that were admissible dispatch hints before (5 `fused_region` end-to-end ->
`nvidia_mma_fused`; 7 `matmul` device -> `nvidia_mma_gemm_emitted`, the
1.5-1.7x emitted-PTX win; 2 `matmul` end-to-end -> the shipped delegate) now
miss, and dispatch falls back to lead-safe tier priority until re-recorded.
**They were deliberately not backfilled** -- a backfilled identity would claim
the recorded runs used code nobody verified. The 12 non-registry rows
(`paged_kv_decode`, `ssm_replay_decode`, `conv2d`) are read by their own
consumers and are unaffected.

### `AUTOTUNE-EMITTED-IDENTITY-SM120-RERECORD` — closed 2026-09-27 (The-Super-Bear)

**All 96 sm_120 registry rows re-recorded, every timed candidate stamped; 15
served by production lookup; every row misses when an NVIDIA emitter changes
with the pins unchanged.** The-Super-Bear (RTX 5070, WSL2, CUDA 13.4 / driver
610.88), fresh clean worktree at `1a737129` with its own `build/` and
`build-nvidia-cuda/` configured from scratch and fully built, both recorder
runs and the checks under `flock /tmp/tessera-timing.lock`, no other GPU
process. Evidence and commands:
`benchmarks/baselines/autotune_corpus_rerecord_sm120_20260927/` (README).

- `record_autotune_corpus.py` (the shape lists of `AUTOTUNE-TOOLCHAIN-KEY-2026-09-26`)
  twice, then `finalize_test5_corpus.py`: neither run refused (0 unstamped
  timed candidates), the two runs agreed on toolchain and every identity at
  every key (the finalizer now refuses otherwise), no key lost or added, 98
  rows changed (96 registry + 2 `conv2d`), the 16 `rocm:gfx1151` rows and 10
  serving rows byte-identical. 31 registry rows selector-eligible (was 38),
  all 31 admitted strictly by `record_autotune_reproducibility.py`.
- **Served (fresh process, `corpus_winner` with inferred dims): 15**, exactly
  the admissible rows -- 7 `matmul` device -> `nvidia_mma_gemm_emitted` (the
  emitted-PTX win is back in production dispatch), 2 `matmul` end-to-end 2048
  -> `nvidia_mma_gemm_shipped`, 5 `fused_region` f16 end-to-end ->
  `nvidia_mma_fused`, and 1 new, `attention` f16 end-to-end 128x128x64x64 ->
  `nvidia_mma_attn`. Perturbing each emitter in turn (CUDA synthesizers,
  resident stages, `ptx_emit` GEMM PTX, `tessera-nvidia-opt` Tile PTX) with
  the toolchain digest asserted unchanged makes every row miss under at least
  one; with all perturbed, 96/96 miss and 0 are served.
- **15 winners changed**, all at small or ragged shapes (64³, 128x256x64, the
  127x259x63 bucket, bf16 256³ end-to-end; three composed-vs-native attention /
  gated rows) whose verdicts are inadmissible (unseparated or
  finalizer-ineligible) in both the old and new corpus, so no served winner
  changed. Table in the evidence README.
- The PTX bridge and `build/`'s shipped GEMM library are byte-identical to
  another tree's build of the same source, and the bridge again after a
  from-scratch rebuild of `build-nvidia-cuda/` with the release gate's
  configure arguments; the check re-run in that rebuilt tree gives the same
  96/96 match and 15 served. These content digests are reproducible, not
  per-build.
- **NVIDIA release gate, device layer, at `7df492f9`** (same box, same
  worktree, under the timing lock): both device-correctness passes **1122
  passed, 1 skipped, 0 failed** (the skip: NCCL not installed, multi-rank
  topology lane not evaluable here); `status=success`.

Open, found by the re-record (not caused by it):

- **`AUTOTUNE-GATED-INFER-DIMS`** (**closed 2026-09-27**, `SM120-AUTOTUNE-FOLLOWUPS-2026-09-27` above) -- `autotune._infer_dims` had no
  `gated_matmul` rule, so the 12 gated rows (keyed on the recorder's explicit
  `(M, H, K)`) cannot be found by ordinary `run_arbitrated` dispatch. None is
  admissible today, so no served count changes; add the rule
  (`A (M,K), Wg (K,H)` -> `(M, H, K)`) with a test before a gated row becomes
  admissible.
- (**Timer added 2026-09-27**, `SM120-AUTOTUNE-FOLLOWUPS-2026-09-27` above: no row races an untimed candidate now; the rows remain unserved for other reasons, see `AUTOTUNE-SM120-ROUTE-RESOURCES`.) 20 device rows race a scalar lane with no device timer
  (`nvidia_flash_attn`, `nvidia_gated`, `nvidia_generic_cuda`), recorded
  `unmeasured`; production refuses them as partial-field races
  (`_record_raced_the_live_field`), and since those lanes are never timed they
  are never stamped either. Pre-existing; a device timer for the scalar lanes
  is what would make those rows servable.

**Review of the mechanism (2026-09-27, independent, Mac).** Fixed on the branch
with tests: the finalizer merged two runs that timed different code (it took
the winner from both runs but the stamps from the second) and now refuses a
pair whose toolchain digest or identities differ; an empty identity `{}` was
stamped and matched like a real one and is now a miss everywhere; the composed
spectral lanes covered only the first inner FFT lane although `_inner_fft`
falls through to the next on a run-time decline (latent: one FFT lane per
target today). New tests run each emitted lane's real `run` with the compiler
intercepted and require its identity to equal the digest of the exact source
and command line the compiler received (NVIDIA generic / mma fused incl. fp8 /
gated / both arms of mma attention, ROCm generic HIP, x86 generic C): no
mismatch was found. Not changed, with reasons: production dispatch now
recomputes every live candidate's identity per lookup (re-emits source and
hashes it; `python_code_identity` calls `inspect.getsource`) -- correct, and
cheap next to a kernel launch, but unmeasured; nvcc/hipcc appear by name
with their version taken from the pin, so a box whose `nvcc` drifts off the
pin without the pin moving is not caught here (the family identity's stated
limit); host-side Python around a kernel (`_composed_operand`, layout
materialization) is outside every identity (stated in the module docstring).

### Cache coherence — Codex review P2 on PR #861 (2026-09-27)

**Finding (confirmed in code).** `NvidiaMmaFusedCandidate.artifact_identity()`
hashed the emitter's *current* output while `run` launched a function from
`_mma_fused_fn_cache`, keyed by storage/epilogue/raster alone. After an
in-process emitter change a re-measurement timed the OLD binary under the NEW
source's identity; after a restart the genuinely new binary matched that stamp
and reused a latency measured for other code. The attention, gated and
resident-stage caches, the PTX registration set, the once-per-process library
loads and the checked-in CPU lanes had the same shape. **Fix: every runtime
cache behind an identified candidate is keyed by the code it holds, or the
stamp is taken from the artifact actually loaded.** Identities of unchanged
code are unchanged.

| Lane(s) | Cache | Old key | New key / why safe |
|---|---|---|---|
| `nvidia_mma_fused_*` (+ device timer) | `_mma_fused_fn_cache`, `_mma_fused_device_fn_cache` | (storage, bias, act, raster, group) | (`cache_key(source, dtype=storage)` + nvcc path/flags, symbol); one artifact per source in `_EMITTED_ARTIFACTS`, shared by host entry and timer |
| `nvidia_mma_attn_*` | `_mma_attn_fn_cache`, `_mma_attn_device_fn_cache` | storage | same |
| `nvidia_mma_gated_*` | `_mma_gated_fn_cache`, `_mma_gated_device_fn_cache` | (storage, act, raster, group) | same |
| `nvidia_mma_*_composed_*` stages | `_resident_ops_artifact` | none (once per process) | `_EMITTED_ARTIFACTS[source key]`; key memoized by source text for the per-stage launch path |
| composed lanes' GEMM, `nvidia_mma_gemm_shipped`, `nvidia_nvfp4_gemm_shipped` | `runtime._nvidia_gemm_runtime` | file digest at identity time (path re-located) | pinned at load (`toolchain_identity.load_library`); identity of the *loaded* path; a rebuild after the load raises `LibraryChangedSinceLoad` (miss) |
| `nvidia_generic_cuda`, `nvidia_flash_attn`, `nvidia_gated`, `nvidia_pointwise`, mma-attn scalar arm | `kernel_cache` + `_LIB_CACHE` | `cache_key` (source-safe; no arch/nvcc) | store key = `cache_key` + `(nvcc, -arch=…, flags)` via `register_compiler(build_line=)`; `_LIB_CACHE` is per compiled temp path |
| `nvidia_mma_gemm_emitted`, `nvidia_nvfp4_gemm_emitted` | `runtime._nvidia_ptx_registered` (set) + bridge module cache | entry name | dict entry -> PTX the bridge holds (written only on successful registration); identity reads it -- the lane never re-registers, so the stamp names what runs |
| `nvidia_tile_matmul_*` | `_nvidia_tile_ptx_cache` + registration | (schedule, dtype) | already safe (identity reads the same cached PTX); now also the registered text |
| PTX bridge (every PTX lane) | `runtime._nvidia_ptx_launch_lib` | file digest at identity time | pinned at load |
| ROCm / x86 / CPU / ANN / native storage | see the ROCm and x86 queues | | `rocm_generic_hip`, `x86_generic_c`: build line in the store key; `rocm_stockham`, `x86_aocl_dlp`, `libtessera_jit`: pinned at load; `cpu_stockham`, `cpu_stencil_grad`: stamped with the bytes compiled (+ quoted local headers); native storage / ANN GPU: identity is the immutable package `binding_digest` (already safe) |

**Arbiter.** `measured_arbitrate` now computes each live candidate's identity
before the race as well as after, and leaves unstamped (so unservable) any
candidate whose identity moved during it: samples that straddle two kernels
describe neither. The stamp is still computed after the race, from the same
source-producing functions the run used, which with content-keyed caches is
the code the last timed run executed.

Tests: `tests/unit/test_autotune_identity_cache_coherence.py` (Mac, compiler /
`dlopen` / PTX bridge intercepted) -- each lane runs, its emitter is patched,
it runs again, and the test requires one fresh compile of exactly the new text,
a stamp equal to the digest of what that compile received, and no recompile
for unchanged source; the Codex cases fail on `042ef54f` with "a changed
emitter must compile fresh".

Device re-check at `3d4bc6a8` (The-Super-Bear, `~/programming/tessera-eid`,
`build/` and `build-nvidia-cuda/` `ninja` no-op -- no C++ changed --
`_nvidia_env.sh`, under the timing lock): **release gate `--layer device`:
both device-correctness passes 1122 passed, 1 skipped (NCCL not installed), 0
failed, `status=success`** (reports `~/gate-reports/eid-3d4bc6a8/` on that
box); the sm_120 serve/miss check is **byte-identical** to the committed
`sm120_serve_check_after_rebuild.txt` apart from host/date -- 96/96 identities
match, 15 served, 96/96 miss with every emitter perturbed -- so the fix moved
no identity; and a real-device probe ran `nvidia_mma_fused` and
`nvidia_generic_cuda` twice (one nvcc compile each), perturbed the emitter,
ran again: one recompile of exactly the new text, still `nvidia_cuda`, stamp
equal to the digest of what nvcc received both times.

Also fixed (orchestrator review N1): `source_identity`'s `extra` fields could
overwrite a core field (`source_sha256`, `build`, ...) and
`composite_identity` could flatten two parts onto one field (`("a.b","c")` vs
`("a","b.c")`); both now raise `EmittedIdentityUnavailable` (a miss).
`python_code_identity` takes no caller fields. No existing identity collided.

`AUTOTUNE-KERNEL-IDENTITY-MEMO` (ROCm queue) -- **closed 2026-09-27**: the
`tessera-opt`-image identity memo is now reused only for a byte-identical
image returned by the launch's own build path, so a Python directive-generator
change within one process re-identifies (Mac tests + gfx1151 real-device
probe; ROCm queue). No NVIDIA lane uses that memo.

### Hot path: no per-launch re-synthesis (follow-up, 2026-09-27)

**Finding.** Keying the emitted lanes' caches by their source (above) made
every `mma.sync` fused/attn/gated launch, every resident-stage launch of the
composed lanes, and every generic-lane `kernel_cache.build` re-run its Python
emitter and hash the text just to find the key -- ~10 us per call on the Mac,
the order of an sm_120 kernel (2-20 us), so it landed in production dispatch
and biased every end-to-end race.

**Fix.** `python/tessera/compiler/emit/source_memo.py`: the source is
memoized against the emitter *function object*, looked up by name at call
time, **and every global it reaches by name** (functions followed into their
own modules, classes, constants, `<tessera module>.<fn>` and function-local
`from tessera... import` references, `self.<method>` on the emitter class).
Replacing any of them -- `monkeypatch`, assignment, `importlib.reload` --
misses and re-emits; an unchanged emitter costs a dict lookup plus an identity
check of its bindings. Never memoized: an emitter whose reachable code reads
the environment (`environ`/`getenv`), a stateful emitter object, a class
marked `source_memo_safe = False` (`AppleAIREmitter` delegates through an
object), and any mutable argument; numbers in the key are tagged with their
type (`1`, `1.0`, `True` emit different C). A rebinding *during* an emit is
not stored. Applied to `_mma_{fused,attn,gated}_source`, `_resident_ops_source`
and `emit_kernel` (all generic lanes: NVIDIA/ROCm/x86; Apple's emitter reads
the environment and is not memoized). `artifact_identity` reads the same
memoized object as the launch, so identity and launch still read one text --
and a change the walk cannot see (a method on the region, in-place mutation of
a module-level table) leaves *both* on the old text, which is what runs, so it
cannot stamp one kernel with another's name. The remaining per-call work is
memoized by object: the nvcc cache line (on the three env vars it reads, from
`os.environ`'s own store, and the helper objects that build it),
`_emitted_key`, `kernel_cache.cache_key`/`store_key`, and the CUDA source
identity.

**Measured (Mac, compile and `dlopen` intercepted, best of 5 x 20000 calls):**
| Per-call path | before (`dedae4b0`) | after |
|---|---|---|
| `_mma_fused_fn(bias, relu, f16)` (host entry lookup) | 10.60 us | 2.42 us |
| `_mma_fused_device_fn` (device-timer entry) | 10.67 us | 2.48 us |
| `_mma_attn_fn(fp8_e4m3)` | 9.43 us | 1.78 us |
| `_mma_gated_fn(bf16, silu)` | 10.52 us | 2.32 us |
| `_resident_ops_lib()` (per composed stage launch) | 3.66 us | 1.38 us |
| `kernel_cache.build` (generic NVIDIA fused lane) | 10.50 us | 4.19 us |
| `nvidia_mma_fused.artifact_identity` (served-verdict check) | 16.91 us | 3.51 us |
| `nvidia_generic_cuda.artifact_identity` | 10.82 us | 5.40 us |

What remains per call is the memo's binding check (~1 us for the ~15 bindings
an mma emitter reaches), the env-store reads, and the symbol/argtypes lookup.
Mac M1 Max host timings; the sm_120 host (Zen 2) differs in absolute terms.

**Tests.** `tests/unit/test_autotune_identity_memo_coherence.py`: an
unchanged emitter runs once across repeated launches (fused/attn/gated/
resident, `_mma_fused_fn` end to end with one compile, generic `emit_kernel`);
a patched emitter, a patched helper, and `emit` patched on the class re-emit;
env-reading / stateful / opted-out / mutable-argument cases are called every
time; `1`/`1.0`/`True` are not aliased. The existing coherence tests
(`test_autotune_identity_cache_coherence.py`, which patch the emitter) pass
unchanged.

**Device evidence at `946530a8` (The-Super-Bear, RTX 5070, `~/programming/tessera-eid`,
`_nvidia_env.sh`, under the timing lock).** Release gate `--layer device`: **both
device-correctness passes 1122 passed, 1 skipped (NCCL not installed), 0 failed**
(`device-correctness-{1,2}.xml`: 1123 tests, 0 failures, 0 errors, 1 skipped;
`status=success`; reports `~/gate-reports/eid-946530a8/` on that box). The
sm_120 serve/miss check is **byte-identical** to the committed
`sm120_serve_check_after_rebuild.txt` (only its trailing `rc=0` line differs):
96/96 identities match, 15 served, 96/96 miss with every emitter perturbed --
the memo moved no identity, and every `setattr` perturbation and its restore
were seen through it. A real-device probe (`probe_source_memo.txt`, same dir):
`nvidia_mma_fused` run 20 times synthesized and compiled once; a patched
emitter compiled exactly the new text, ran correctly on device and was stamped
with the digest of what nvcc received; restoring the emitter reused the
original artifact and stamp with no compile; `nvidia_generic_cuda` (through
`kernel_cache.build`) compiled once for 10 runs and recompiled exactly the
patched text. `_mma_fused_fn` lookup on that host: 3.0 us per call.

## `FOUNDATION-BATCH-2-2026-09-27`: gfx1201 W8A8 CU authority, ragged-M store, MXFP4 one-row-block test — not applicable

The changes are gfx1201 W8A8 selection (`selectFp8W8A8BlockScalePanel` and
its Python oracle, now reading `rocm_target.compute_units`), the ROCm Tile
consumer's bounded fragment store (`TileToROCM.cpp`, ROCm-only), and MXFP4
recorder options. NVIDIA schedules no `tessera.scaled_matmul` W8A8 contract
and lowers no ROCm Tile store, so nothing here changed for NVIDIA.

## `GFX1201-PERF-2026-09-27`: gfx1201 W8A8 LDS body, bf16 store, folded MXFP4 row guard — not applicable

ROCM-FP8-BLOCKSCALE-1 / ROCM-MXFP4-W4A8-1 on gfx1201 ([W8A8 packet](../../../../benchmarks/baselines/gfx1201_fp8_blockscale_lds_20260927/README.md),
[MXFP4 packet](../../../../benchmarks/baselines/gfx1201_mxfp4_small_m_20260927/README.md)).
One shared-dialect change: `schedule.matmul` gains `staging` (default
`global`, printed and digested only when set; the verifier admits `lds` only
for the gfx1201 `[N, K]` W8A8 contract), so every sm_120 schedule and digest
is unchanged. The bf16 store narrowing lives in the ROCm fragment-store
materializer only. sm_120 still has no W8A8 or folded MXFP4 route; nothing
here transfers.

## `GFX1201-LANES-2026-09-27` (ROCM-MXFP4-W4A8-1 folded load schedule): sibling outcome — not applicable

The gfx1201 folded MXFP4 prefill gained a Target-IR-carried load schedule
(raster, register prefetch, vector-scale epilogue, CU mode by row blocks;
[packet](../../../../benchmarks/baselines/gfx1201_mxfp4_prefill_20260927/README.md)).
Not applicable here: sm_120 has no folded MXFP4 package; the new keys exist only on the gfx1201 `tessera_rocm.scaled_wmma_gemm` folded contract, and the CU/WGP rule is an RDNA workgroup-mode choice with no CUDA analogue. No shared contract changed.

## `GFX1201-LANES-2026-09-27`: W8A8 block-scaled FP8 — follow-up only if sm_120 wants it

Owner ROCM-FP8-BLOCKSCALE-1 (gfx1201). The logical `tessera.scaled_matmul`
over e4m3 with fp32 scales now derives `rocm_fp8_w8a8_blockscale{,_nk}_v1` at
Graph->Schedule, and only on `rocm`/`gfx1201`: on `nvidia_sm120` the same op
still has no schedule and is refused, exactly as before. Two shared pieces are
portable and have **no NVVM consumer**: the Tile op
`tile.fragment_scaled_accumulate` (the architecture consumer owns which
(row, col) each accumulator register holds) and the `scale_n` field on
`schedule.matmul` (stated, and digested, only when set -- every existing
schedule digest is unchanged). An sm_120 W8A8 route would need its own
consumer for that op (or the native block-scale MMA where the scale format
allows it) and its own device proof; no gfx1201 number transfers.

## `NVIDIA-PREPR-REVIEW-2026-09-26`: review fixes to the device-layer and marker work

Branch `claude/timing-foundation-nvidia-fixes`. Host-free fixes checked on the
Mac; device results from The-Super-Bear (RTX 5070, WSL2).

- **P0, wrong result (pre-existing, CPU): every `tessera.reduce` ran as a
  sum.** The frontends canonicalize `ops.reduce(op=...)` into the Graph op's
  `kind` and drop `op`; `runtime._execute_runtime_cpu_op` and
  `matmul_pipeline._execute_op` read `kwargs.get("op", "sum")`, so
  `reduce(op="max", axis=1)` on `[[1,5],[2,3]]` returned `[6, 5]` under
  `@tessera.jit` (reproduced on the Mac). New `compiler/reduction_kind.py` is
  the one reader: `kind` (legacy `op` only when `kind` is absent; the two
  disagreeing is refused), `E_REDUCTION_KIND_MISSING` when neither is present
  (#21a), exactly the ODS `ReductionKindAttr` set (sum/max/min/mean, pinned
  against `TesseraOps.td` by a test), else `E_REDUCTION_KIND_UNSUPPORTED`.
  The same stale read was found and fixed in the Apple GPU reduce lane
  (`tessera.reduce` always ran its sum code), in the Python Schedule/Tile
  metadata (the schedule dropped the kwargs and Tile IR stated `op = "sum"`
  for every reduce -- a #32 loss), in `nvidia_native._reduction_contract`
  (kind-less defaulted to sum) and in the `tessera.ops.reduce` reference
  (supported sum/mean only). Checked and not affected: `scheduled_kernel`
  (already fails closed), native JVP plugins (refuse), collectives (their
  `op` is a different key), autodiff rules (fed the eager `op` by the tape).
- **P1-a: device-clock admission compared medians only.** A packet rebuilt at
  10 launches per window (per-window errors to 18%, median 3.5%) was
  admitted; the ROCm route had the same gap. Every window is now checked
  (`profiler_timing.device_clock_window_refusals`): stored windows present and
  the source of the stored medians, each window at least the target's
  measured minimum (sm_120 1 ms from the probe; gfx1151/gfx1201 5 ms, about 3x
  the lengths where disagreement was seen), each within 5%. The NVIDIA packet
  applies it always; the ROCm packet on the current window protocol (legacy
  packets stay history, refused by name at admission). The committed sm_120,
  gfx1151-interleaved and gfx1201 admission packets still validate (shortest
  windows 10.0 / 13.4 / 15.8 ms, worst window 0.9 / 1.1 / 1.2%); two
  short-window history packets (gfx1201 `superseded/second_6e6904dc_short_window`,
  gfx1151 `diagnostics/launches_probe`) now derive the refusal, as intended.
  Tests rebuild the reviewer's forgery on both routes.
- **Other review items:** witness refusals keep their own codes
  (`DEVICE_CLOCK_WITNESS_MISSING` is not `..._DISAGREES`); the SASS check
  requires exactly one 64-bit MIN and one MAX; SSD admission requires one
  device UUID across the 18 calibrations (`DEVICE_CLOCK_PART_MISMATCH`);
  `relative_spread` returns None for one sample (it returned 0.0, "no
  noise") and a verdict then does not exist; the serving recorder warms both
  routes up before sampling. Ten reason codes registered.
- **Admission keying: bound to the part, not the compute capability.**
  `NVIDIA_DEVICE_CLOCK_VALIDATED_PARTS = {"sm_120": {"NVIDIA GeForce RTX
  5070"}}`; any other part refuses with `DEVICE_CLOCK_PART_UNVALIDATED`. Why
  the part and not the UUID: the window-length validation (the span/event
  offset and so the 1 ms minimum) is a property of the part and its driver,
  measured on this RTX 5070; cc 12.0 spans the consumer Blackwell line, so it
  is too coarse, while a second card of the same model shares the counter
  behaviour and every packet re-checks per-window agreement anyway, so a
  UUID pin would add no evidence. A 5070 Ti / 5080 / 5090 needs its own probe.
- **P1-b wording:** the `nan_mode` group is a provenance-parity fix under
  #32, not a compiler defect, and under #29 a declaration whose only
  consumer is a test (corrected below and in MASTER_AUDIT).
- **SSD README:** the cooperative variance is within-process and
  time-ordered (pairs 5 and 7 run slow, then fast), a clock-ramp or warm-up
  hypothesis, not a finding.

## `NVIDIA-DEVICE-LAYER-88-2026-09-26`: the 88 pre-existing device-layer failures, root-caused

The-Super-Bear (RTX 5070, WSL2), `scripts/run_nvidia_release_gate.sh --layer
device`, fresh worktree with `build/` and `build-nvidia-cuda/` both fully
built. **Before** (branch at `048fac95`, = `claude/timing-foundation`): 1035
passed / 88 failed, the identical failure set to `main` `b5da0a4e`.
**After** (`aa6fac65`): `device-correctness-1` 1122 passed / 1 skipped /
0 failed, and `device-correctness-2` (never reached before) also 1122 / 1 / 0. Re-run at the
final branch head `49a3d4b1` (after the timing-foundation merges and the sm_120
corpus re-record): both passes again 1122 passed / 1 skipped / 0 failed.
**Corrected 2026-09-26 (pre-PR review):** no group was a wrong kernel result.
The `nan_mode` group was a **provenance-parity fix recorded under Decision
#32**, not a compiler defect: the kernel's NaN behaviour was never affected
(the policy was validated in Schedule and Tile IR and honoured by the
kernel), only the descriptor stopped repeating it. Under Decision #29 that
provenance field is a declaration whose **only consumer is a test** (no
runtime path reads `provenance["nan_mode"]`); it is kept for #32 parity and
counted as such, not as a behaviour fix. The rest were tests asserting
behaviour the compiler had correctly moved past. (The review did find a real
wrong-result bug nearby, outside this set: the CPU executors ran every
`tessera.reduce` as a sum -- `NVIDIA-PREPR-REVIEW-2026-09-26` below.)

| Group | n | Cause | Fix |
|---|---|---|---|
| `KeyError: 'nan_mode'` (reduction mean/max/min/amax/amin) | 45 | **Provenance parity (Decision #32), not a behaviour defect.** `package_scheduled_kernel` validated `nan_mode = "propagate"` in Schedule and Tile IR, then left it out of the launch descriptor's provenance; the retired Python constructor used to carry it. Kernel NaN behaviour unaffected. Only a device test reads the field (#29: a declaration whose only consumer is a test). | Carry the validated policy into provenance for reductions. |
| "requires a supported native scheduled reduction" (`sum`) | 21 | **Stale test.** The test built `tessera.reduce` with no `kind`; ODS makes `kind` a required `ReductionKindAttr`, and Decision #21a forbids defaulting it. The pre-Schedule constructor read kind-less `tessera.reduce` as sum (fail-open); the scheduled contract refuses it. | Test passes `kind="sum"`; a unit test pins that a kind-less reduce fails closed on nvidia/rocm/x86/apple. |
| `9.999999747378752e-06 == 1e-05` (norm epsilon) | 18 | **Stale test.** The descriptor carries the f32 epsilon the kernel uses and whose bits the packager checks against Schedule and Tile IR; 1e-5 is not representable in f32. | Compare against `float(np.float32(1e-5))`, not a looser tolerance. |
| LSE saved-checkpoint forward, max abs 0.136 | 1 | **Stale oracle, not wrong numerics.** Tessera's causal mask is bottom-right aligned (`tessera.ops.flash_attn`: `triu(k=1+max(Sk-Sq,0))`); the test's oracle masked top-left. At Sq=3 < Sk=4 they differ; the failing device values equal the bottom-right reference (first row to 4e-8). | Oracle uses bottom-right alignment: it matches `tessera.ops.flash_attn` to 6e-8 and its own finite-difference dq to 6e-6 (Mac), and the device forward/backward now pass the test's 3e-5/4e-5 tolerances. |
| NCCL topology message | 1 | **Host cannot evaluate.** No `libnccl` on Super-Bear, so the probe refuses on libraries before device enumeration. | Assert the executor also refuses, then skip ("host cannot evaluate"), never pass. |
| `test_live_nvidia_emitted_ragged_degrade_is_logged` | 1 | **Stale test.** Since `982f5225` the emitted GEMM predicates ragged M/N and runs them; odd K is refused before selection by `applies_to_inputs`, so a forced dispatch can no longer degrade silently. | Renamed `..._ragged_served_odd_k_refused`: M=24 runs natively (`won`), odd K raises `ArbiterError`. Degrade bookkeeping stays unit-covered. |
| `test_the_delegate_wins_on_device_only_at_small_shapes` | 1 | **Test asserted a ranking, not Decision #28.** The emitted PTX lane gained a device timer, joined the race and measured fastest at 512³ (0.0285 ms vs shipped 0.0455). | Replaced by `test_the_measured_arbiter_selects_the_fastest_in_budget_candidate`: at 512³ and 2048³ the device-timed arbiter races a complete field incl. the delegate, its verdict is the minimum of its own measurements, and the winner meets the delegate's declared `tolerance`/`tolerance_rel`. `_NO_DEVICE_TIMER` is now empty. |

The one skip in the after-run is the NCCL lane. Unit additions: a kind-less
reduce is refused on every scheduled target.

## `NVIDIA-GLOBALTIMER-MARKER-2026-09-26`: the `%globaltimer` device-clock witness, validated and admitted

Sync key (follows `DEVICE-CLOCK-MARKER-2026-09-26`). **Validated on The-Super-Bear
(RTX 5070, cc 12.0, WSL2, driver 610.88) and admitted for `nvidia_sm120` only.**
Evidence: [`benchmarks/baselines/sm120_ssd_calibrated_pairs_20260926/`](../../../../benchmarks/baselines/sm120_ssd_calibrated_pairs_20260926/README.md).

- **Builds, two reads into one span.** `build_device_clock_marker(backend='nvidia',
  chip='sm_120')` goes through `tessera-opt` -> NVVM -> `gpu-module-to-binary`; its
  SASS is two `CS2R Rn, SR_GLOBALTIMERLO` and two `REDG.E.MIN/MAX.64.STRONG.SYS`,
  and the builder refuses an image without them (`cuobjdump --dump-sass`).
- **Resolution: exactly 32 ns** per `%globaltimer` step (65,536 back-to-back
  reads x3, ~9.8 ns per read). Not coarse on WSL2.
- **Agreement with CUDA events:** a roughly fixed ~10-16 us per-window offset
  (up to ~46 us). Every window of about 1 ms or longer agreed within 5% (worst
  3.2%); every configuration of ~0.36 ms or less had a window outside it. The
  window length is therefore a recorded, admission-checked parameter.
- **SSD packet** (`32,2,16,4` chunk 8, nine interleaved pairs, 7 x 1000-launch
  windows, source `de66d702` clean): the production selector **admits
  cooperative**, lower bound **4.33x** (median 7.48x, pairs 4.27-8.29x), all 18
  calibrations eligible (device clock 0.03-0.29% below the event;
  bracketed/plain 0.956-1.013), max abs error 0; `replay.json` reproduces it.
  Not claimed: the cooperative variant's 12.0-23.2 us/launch per-process spread
  is unexplained, and its lowest ratio (0.956) sits near the two-sided band edge.
- **Contracts changed:** `native_device_clock.VALIDATED_MARKER_TARGETS` adds
  `('nvidia','sm_120')`; `profiler_timing` gains `NVIDIA_CLOCK_SLOTS` for
  `nvidia_sm120` (`cuda_event_ns` witnesses `device_wall_clock_ns`; witnesses
  stay intersected with the target's own slots); new
  `profiler_nvidia_evidence` packet (witness agreement enforced in every
  environment, verdicts re-derived by the validator, queried identity
  required); `ssd_performance` admits sm_120 on these packets and refuses a mix
  with Nsight windows; `record_ssd_gpu.py` / `record_ssd_rocm_calibrated_pairs.py
  --backend nvidia`. No functional C++ change was needed; the pass comment
  records the validation.
- **Open:** the Nsight activity-window packet on WSL2 (a separate route);
  event-only recorders (`phase5_ingest`, `record_packed_storage_foundation`)
  do not use the marker yet.
- **Tightened 2026-09-26 (`NVIDIA-PREPR-REVIEW-2026-09-26`):** admission now
  checks every window (at least 1 ms, each within 5%), is bound to the
  validated part (RTX 5070) rather than cc 12.0, and needs one device UUID
  across an SSD comparison. The window-length validation came from this RTX
  5070 only; a different cc 12.0 part needs its own probe.

## `AUTOTUNE-TOOLCHAIN-KEY-2026-09-26`: sm_120 and gfx1151 corpus rows re-recorded

**sm_120 re-recorded 2026-09-26 on The-Super-Bear** (RTX 5070, WSL2, CUDA 13.4 /
driver 610.88), clean worktree at `4e699f7e`, with the commands below; logs in
`benchmarks/baselines/autotune_corpus_rerecord_20260926/sm120_*`.

- **Checks before commit:** the 16 `rocm:gfx1151` rows are unchanged (and
  textually untouched: the finalized file was re-ordered to the prior record
  order, content verbatim); sm_120 keys 97 -> 108, none lost, the 11 additions
  being the predicted composed / serving `end_to_end` keys; every sm_120 row
  carries one `toolchain_digest`; all 28 rows that race a shipped delegate
  carry `delegate_identities`; `MeasureCache().stale_records()` is empty (124
  served).
- **Strict admission:** `record_autotune_reproducibility.py` considered and
  admitted 38 selector-eligible sm_120 rows (was 20), no stale resource
  fingerprint, so the route-resources manifest did not need regenerating.
  21 winners changed against the pre-schema rows (listed by `git diff`); 23 of
  108 sm_120 rows are admissible dispatch hints under `record_is_admissible`,
  the rest are unseparated or finalizer-ineligible and are kept, not served.
- **The paged-attention warm start is served again (first re-record; superseded by the post-review re-record below).** The review made it apply
  `record_is_admissible`, so `benchmark_serving.py` now measures fused and staged
  **interleaved** over 20 reps, keeps every rep, and writes a separation
  verdict (as `record_paged_kv_corpus.py` does). Served winners, device timing:
  128 tokens `fused` (margin 88%, noise 8%), 512 `fused` (52% vs 12%), 2048
  `staged` (45% vs 17%); all separated. **Pre-schema winners were fused /
  staged / staged, so 512 tokens flipped.** Cause: the staged route's device
  latency rose from 0.25 ms (pre-schema row) to 0.72-0.90 ms at 128 and 512
  tokens (four runs, two without a corpus update), while fused is
  unchanged (0.104 / 0.42 ms). **Not investigated** -- recorded as an open
  question, not a regression claim, because the pre-schema row's toolchain
  and runtime-library build are unknown (RUNTIME-LIB-OPT-1 changed those
  libraries' `-O` level since). **A possible confounder of the recorder
  itself:** the interleaving runs staged right after fused on even reps (and
  first on odd ones), so staged may pay for state fused leaves behind
  (caches, clocks, allocator); the pre-schema row timed each route in its own
  block. Not separated from the other causes.
- **Serving rows re-recorded 2026-09-26 after the review** (warm-up added;
  one-sample spreads no longer read as zero noise), The-Super-Bear, clean
  worktree at `9111b1b6`, only the 10 serving rows replaced. Device timing:
  128 tokens fused 0.104 vs staged 1.235 ms, **separated** (margin 92%, noise
  28%); 512 fused 0.418 vs staged 1.063, **not separated** (61% vs 38%);
  2048 staged 1.239 vs fused 1.666, **not separated** (26% vs 25%). So the
  warm start now serves **128 tokens only (fused)**; 512 and 2048 fall back to
  the lead-safe default. Staged's per-rep spread is large (25-38%) and its
  median moved again (0.73-0.92 ms in the first re-record, 1.06-1.24 ms now),
  which is what denies the verdicts -- the staged route's timing is the open
  question, not the arbiter. `paged_kv_decode [1,8,128,64] end_to_end` **now
  separates** (fused 3.83 vs staged 29.05 ms, margin 87% vs noise 27%); before
  the warm-up its noise was 17,520% from the 3031 ms first-call sample.
- **Test fix:** `test_committed_corpus_has_sm120_matmul_comparisons` required the
  emitted GEMM in every end_to_end matmul row; it cannot serve odd K and is
  removed from the 127x259x63 race before timing, so that row must not list it.

Decisions #11/#12 landed host-independently on the Mac (MASTER_AUDIT action
item 3). **Follow-up required on Super-Bear; no sm_120 row has been
re-recorded** (history; done above). The arbiter corpus (`benchmarks/baselines/autotune_corpus.json`,
written as v4 since the gfx1151 re-record) keys every verdict on the toolchain
identity (`compiler/toolchain_identity.py`: CUDA 13.4 / PTX 9.4 / driver
610.88 / driver-JIT PTX 9.3 / LLVM 23.1.1 pins) and, for both Tier-3
candidates that bind it (`nvidia_mma_gemm_shipped`, `nvidia_nvfp4_gemm_shipped`),
the content digest of `libtessera_nvidia_gemm` plus the bound entry. EMITTED and
SYNTHESIZED candidates carry no artifact identity (pins only). All
97 committed `nvidia:sm_120` rows predate that key, so they load as stale and
select nothing -- including the paged-attention serving warm start
(`emit/nvidia_cuda.py::_paged_attention_corpus_winner` returns `None`; since the
2026-09-26 review it also applies `record_is_admissible`, like its ROCm twin, so
`benchmark_serving.py --update-corpus` rows -- one latency per mode, hence no
separation verdict -- are refused as unproven rankings until that recorder
records spreads). The 16
`rocm:gfx1151` rows were re-recorded on Princess-Luna on 2026-09-26 (ROCm plan,
same key); the sm_120 rows were left byte-identical in content.

**Recorder fixes landed host-free (Mac), so the re-record cannot destroy
evidence:**

- `record_autotune_corpus.py` always loads the corpus now. Before, a run without
  `--warm-start` started empty and `save_corpus` deleted every other device's
  rows (it would have deleted the re-recorded gfx1151 rows); with
  `--warm-start` it stamped the nvcc evidence block onto every row in `_store`,
  which after Decision #11 includes current gfx1151 rows on any host. It now
  evicts only its own current sm_120 rows (`matmul`, `fused_region`,
  `attention`, `gated_matmul`, `conv2d`) unless `--warm-start`, stamps only rows
  this run measured, refuses to write if another device loses rows, and names
  owned rows left stale. Test: `tests/unit/test_nvidia_autotune_corpus_recorder.py`.
- `finalize_test5_corpus.py` wrote `"version": 3` and, for an unstable pair,
  copied the fresh evidence -- now carrying today's `toolchain_digest` -- onto
  the prior committed row. With a stale prior row that serves a pre-key
  measurement as current. It now replaces a prior row from a different (or no)
  toolchain with the fresh selector-ineligible one, and writes
  `CORPUS_VERSION`. Tests: `tests/unit/test_nvidia_test5_corpus_finalize.py`.

**The 97 keys and who writes them.** 92 are `record_autotune_corpus.py` keys
followed by `finalize_test5_corpus.py` (two runs; the stability/resource
evidence `stable_runs`, `run_winners`, `selector_eligible`,
`resource_fingerprints` comes from the finalizer, not the recorder):

- `matmul`, `float16` and `bfloat16`, both timings (28): 64³, 256³, 512³, 1024³,
  2048³, 128x256x64, 127x259x63.
- `fused_region` bias+gelu: `f16` end_to_end at 64³, 256³, 128x512x256,
  127x259x63, 128x256x256 (5); `f32`/`fp8_e4m3`/`fp8_e5m2` both timings at
  64³, 256³, 128x512x256, 127x259x63 (24), plus device-only at 128x256x256 (3).
- `attention` causal: `f16` end_to_end at 128x128x64x64, 64x512x64x64,
  64x256x64x64 (3); `f32`/`fp8_e4m3`/`fp8_e5m2` both timings at
  128x128x64x64, 64x512x64x64 (12), plus device-only at 64x256x64x64 (3).
- `gated_matmul` silu, `f32`/`fp8_e4m3`/`fp8_e5m2`, both timings, 64x256x256
  and 128x512x512 (12).
- `conv2d` f32 device, 1x32x32x32x3x3x64 and 1x64x64x64x3x3x64 (2).

The remaining 5 are `benchmark_serving.py --update-corpus` keys, device timing:
`paged_kv_decode` f32 buckets `[1,8,128,64]`, `[1,8,512,64]`, `[1,8,2048,64]`
and `ssm_replay_decode` f32 `[1,128,64]`, `[1,256,128]`.

Adding 128x256x256 / 64x256x64x64 to the recorder's shape lists covers the 8
keys the defaults miss; the current recorder races composed dtypes in both
timings, so it also writes 6 composed `end_to_end` rows at those two shapes,
and the serving recorder writes 5 `end_to_end` rows beside its device rows --
11 keys the committed corpus does not have today. Those are additive, not a
loss.

**Commands (Super-Bear, `ssh bear`; a fresh worktree of the branch with its
own `build/` and `build-nvidia-cuda/` both fully built -- see the stale
`build-nvidia-cuda` trap; every timing run under the shared lock):**

```bash
source .venv/bin/activate && source scripts/_nvidia_env.sh
export PYTHONPATH=python
SHAPES=(--fused-shapes 64x64x64 256x256x256 128x512x256 127x259x63 128x256x256
        --attention-shapes 128x128x64x64 64x512x64x64 64x256x64x64)
# Two independent fresh runs into scratch corpora (absent file => empty cache).
for run in 1 2; do
  rm -f /tmp/sm120_run$run.json
  TESSERA_AUTOTUNE_CORPUS=/tmp/sm120_run$run.json \
    flock /tmp/tessera-timing.lock \
    python benchmarks/nvidia/record_autotune_corpus.py "${SHAPES[@]}"
done
# Merge: replaces a stale row with the fresh one; stable pairs become eligible.
python benchmarks/nvidia/finalize_test5_corpus.py \
  --base benchmarks/baselines/autotune_corpus.json \
  --first /tmp/sm120_run1.json --second /tmp/sm120_run2.json \
  --resources benchmarks/baselines/nvidia_sm120_test5_route_resources.json \
  --output benchmarks/baselines/autotune_corpus.json
# Serving rows (loads the committed corpus, replaces its 5 stale device rows).
flock /tmp/tessera-timing.lock python benchmarks/nvidia/benchmark_serving.py \
  --update-corpus --output /tmp/sm120_serving.json
# Strict admission of every selector-eligible row.
python benchmarks/nvidia/record_autotune_reproducibility.py
```

Before committing, diff rows and evidence against the previous corpus: 16
`rocm:gfx1151` rows unchanged, no sm_120 key lost, every sm_120 row carrying
`evidence.toolchain_digest`, and
`MeasureCache().stale_records()` empty (or naming exactly the keys the run
could not re-race). `resource_fingerprints` in
`nvidia_sm120_test5_route_resources.json` are from the earlier TEST-5 run; if
`record_autotune_reproducibility.py` refuses a row as a stale resource
fingerprint, regenerate that manifest first rather than dropping the row.

**Kernel-code identity (2026-09-26, ROCm plan same key): follow-up required,
not implemented for NVIDIA.** Compiler-generated candidates are now keyed on
the digest of the normalized instruction stream of the image they would run
for the workload (`compiler/kernel_code_identity.py`,
`Candidate.artifact_identity(region, *inputs)`), replacing the `tessera-opt`
binary digest that differed on every build; proven on gfx1151 for
`rocm_wmma_gemm` (rows served in a build tree that did not record them). The
mechanism is generic -- a candidate overrides `artifact_identity` and returns
`compiler_kernel_identity(key, build_image, isa=...)`, and an EMITTED candidate
sets `requires_artifact_identity()` to `True` so an uncomputable digest misses
-- but only the AMDGPU HSACO normalization exists. The NVIDIA EMITTED lanes
(`nvidia_tile_matmul_*` and the other emitted PTX candidates) carry no artifact
identity today and rely on the pins alone; adopting it needs a PTX or
SASS-level normalization (e.g. `cuobjdump -sass`/`nvdisasm` of the loaded cubin
with addresses and encodings stripped, or the PTX text minus its version
header) and a Super-Bear proof that two build trees yield one digest. Not
trivial, so not done here. The two shipped Tier-3 delegates
(`nvidia_mma_gemm_shipped`, `nvidia_nvfp4_gemm_shipped`) keep their library
digest. No sm_120 row was touched.

**Review fixes (2026-09-26, ROCm plan same key) that an NVIDIA adopter
inherits:** the identity is now `tessera.kernel_code.v2` -- it fails closed on
any undecodable word, digests non-code data sections, and names its
disassembler; a PTX/SASS normalization must meet the same bar.
`measured_arbitrate` now infers dims with `_infer_dims` when none are given, so
recorders that pass none land on the key `corpus_winner` looks up. **Dims
mismatch found in the sweep, not fixed (NVIDIA-owned rows):** the sm_120
`gated_matmul` (`dims=(m, h, k)`) and `conv2d` rows are recorded with explicit
dims, but `_infer_dims` returns `None` for those ops, so ordinary
`run_arbitrated` dispatch never consults them -- they select only for a caller
that passes dims. Fix by teaching `_infer_dims` those ops (one authority) and
re-recording; matmul, fused_region, attention and the serving paged-attention
rows agree with their lookups.

## `NVIDIA-LANE-B-1`: routes that skip Schedule IR or bypass it from Python — 2026-09-26

Sync `LANE-B-SWEEP-2026-09-26` (ROCm owns the pattern: its Lane B was retired
the same day, `docs/audit/backend/rocm/ROCM_LANE_MAP.md` §"Decision — Lane B
is retired"). **Follow-up required; nothing changed on NVIDIA and nothing is
device-proven here** (Super-Bear offline). Found by a read-only audit and
re-checked against the tree at `dc071ae4`. The canonical route is
`scheduled_matmul.lower_scheduled_matmul` (Graph → Schedule → Tile, replay
checked) → `nvidia_native.package_scheduled_matmul`.

Each item is a second authority for a boundary the scheduled route already
owns (Decision #31), or a route that skips Schedule IR, so its schedule never
enters the replayed contract:

1. **Graph → Tile C++ pipelines (the Lane B pattern).**
   `tessera-lower-to-gpu` and `tessera-nvidia-pipeline[-sm90/-sm100/-sm120]`
   (`src/transforms/lib/Passes.cpp`, `addCUDA13PipelineForSM` and the
   `tessera-lower-to-gpu` builder) run `createTileIRLoweringPass` straight on
   Graph IR. `LowerMatmulToTileMMA` rewrites `tessera.matmul` to `tile.mma`,
   taking tile M/N from the attention options `tile-q`/`tile-kv`.
   - Production use: only `driver._try_validate_with_tessera_opt`, which runs
     it on every NVIDIA compile and discards the output.
   - Other appearances: lit (`nvidia_pipeline_alias.mlir`,
   `tile_ir_lowering.mlir`) and as a `pipeline_name` label in several
   modules.
   - To do: retire the matmul part, or point the validation at the scheduled
     route. A validation step that exercises a route nothing ships validates
     the wrong compiler.
2. **`@jit(target="nvidia_sm120")` matmul runs `nvidia_mma`.**
   `JitFn._uses_nvidia_mma_default` → `runtime._execute_nvidia_mma_artifact`
   → the hand-written NVRTC kernel in `libtessera_nvidia_gemm.so`. No
   declared Target IR op sits on this path, so the scheduled route is not the
   production `@jit` path for fp16/bf16 matmul on sm_120.
   - Nothing previously recorded this.
   - To do: route `@jit` through the scheduled package, or put `nvidia_mma`
     behind a declared, arbitrated Target IR op (Decision #28 Tier 3).
     Either way, one authority decides.
3. **`package_matmul` / `emit_matmul_tile_ir`.** Python writes the Tile IR,
   skipping Graph and Schedule. The schedule comes from the `nvidia_schedule`
   option.
   - It is the fallback when `tessera-opt` is absent, and the only route for
     fp64/tf32/fp8/int8.
   - Already recorded above (§ the Decision #31 follow-on to "declare or
     retire" this fallback). This entry adds the dtype coverage: retiring it
     first needs scheduled contracts for those dtypes.
   - Consequence for the dashboards: `bootstrap_prune_gap` marks
     `nvidia_sm120,matmul` **compiled**, which overstates it while these
     dtypes have no other route.
4. **nvfp4 / mx matmul** (`package_nvfp4_matmul`, `package_mx_matmul`): Python
   Tile IR with no scheduled contract. They are the only route, not a
   duplicate, but they skip Schedule IR.
5. **Bench-only arbiter candidates.**
   - `NvidiaMmaGemmEmittedCandidate` emits Python PTX via
     `ptx_emit.emit_mma_sync_gemm_ptx`.
   - `NvidiaTileMatmulCandidate` writes a Python `tile.matmul_kernel` string
     copied from the `emit_matmul_tile_ir` template.
   - Neither is production today; if either is ever promoted, it must enter
     as a declared Target IR candidate.
6. **Broken benchmark (verified by reading the code, not run):**
   `benchmarks/nvidia/benchmark_scheduled_macro_matmul.py` calls
   `package_scheduled_matmul(module, scheduled, pipeline_name=...)`. The
   function takes one artifact, so this raises `TypeError`. **Fixed and run
   2026-09-26 on The-Super-Bear (RTX 5070):** the unfixed call reproduced the
   `TypeError` at 16x32x8; the fixed call (`package_scheduled_matmul(scheduled,
   pipeline_name=...)`) ran to completion. Smoke run only, not a timing claim.

**Branch regression check for this PR (The-Super-Bear, 2026-09-26).** Fresh
worktrees of `main` (b5da0a4e) and the branch (4e810487), each built all
targets in both `build` and `build-nvidia-cuda` configurations; the branch's
NVIDIA runtime libraries compile at `-O2` (RUNTIME-LIB-OPT-1), `main`'s with
no `-O`. Release gate: cpu 923 vs 927 passed; compiler 1 vs 1 passed; device
1035 passed / 88 failed on **both**, with identical failure sets; NVIDIA unit
selection 959 vs 959 passed. **No branch-only failure.** The 88 device
failures are pre-existing on `main` on this box (84 in
`test_e2e_spine_native.py`: `KeyError: 'nan_mode'`, "requires a supported
native scheduled reduction", a 9e-06 vs 1e-05 tolerance literal) and are
owed their own investigation (**root-caused 2026-09-26:
`NVIDIA-DEVICE-LAYER-88-2026-09-26` below; one provenance-parity fix under
#32, the rest stale tests**). The `compiler` layer's lit half
(`check-tessera-nvidia`) passed 62/62 on both trees; its pytest half selects a
single test on both, so that half checks very little. The gate exits after the
first device pass fails, so `device-correctness-2` never ran on either tree.
Re-checked at the later branch head 70d1c65e, since shared Python
(`scheduled_matmul` split-K parsing) changed after 4e810487: the scheduled
NVIDIA tests (`-k "scheduled and (nvidia or sm120)"`) pass 301/301 and the
fixed benchmark's 13 rows are numerically green.

Other families follow the same Python-Tile-string pattern
(`emit_softmax/reduce/norm/paged_attention/...` in `nvidia_native.py`). They
are in scope for the same sweep, once the matmul order is settled.


## Device-clock markers — 2026-09-26

Sync `DEVICE-CLOCK-MARKER-2026-09-26` (follows `WSL-TIMING-ADMISSION-2026-09-26`). **Shared contracts changed:**

- New `tessera-opt` pass `--tessera-device-clock-span{backend=rocm|nvidia}` (`src/transforms/lib/DeviceClockSpanPass.cpp`) stamps a kernel's span with the target's constant-rate device clock (ROCm `llvm.readsteadycounter` → `s_sendmsg_rtn_b64 MSG_RTN_GET_REALTIME`; NVIDIA `%globaltimer`) into an appended span buffer. It refuses kernels whose return is not the single final terminator, an alloca after work, double instrumentation, and an unnamed backend.
- Its consumer is `tessera.compiler.native_device_clock`: an **empty marker kernel** launched before and after a timing window, so the span covers the window while the measured image stays byte-identical. Stamping the measured kernel directly was measured to change its codegen (gfx1151 serial SSD: 2512 → 924 instructions stamped at the block start, 2.4× faster; ~1230 even after the entry allocas) and was abandoned.
- `profiler_rocm_evidence`: the instrumentation gate is now **two-sided** — an instrumented measurement faster than the clean one by more than the band blocks as `INSTRUMENTATION_CHANGED_THE_KERNEL`.
- `target_perf.apply_corpus`: every selector-eligible corpus carries its raw measurements; the environment is derived from them (a WSL measurement cannot be relabelled bare metal), each overlay must equal its raw record's `results`, the raw architecture must match the device target, and a WSL witness sample counts only if its `artifact_digests` name the computed raw-measurement digest (reviews of #855 and this branch). Binding is by digest, not yet by comparing sample clocks with the measured metric — owed with the first WSL corpus producer.
- `benchmarks/check_ssd_admission.py` now replays calibrations carried in the comparison.

**NVIDIA outcome: follow-up required.** The pass already emits the NVIDIA form (`%globaltimer` reads, native `atom.min/max.u64`, `bar.sync` — checked by lowering to sm_120 PTX on the Mac), which is the missing **non-profiler** NVIDIA witness. The marker builder refuses NVIDIA until it is validated on Super-Bear: build the marker there, record an SSD calibrated-pairs packet, and add the NVIDIA packet route. No NVIDIA evidence is claimed.

**Closed 2026-09-26 by `NVIDIA-GLOBALTIMER-MARKER-2026-09-26` (below): validated on The-Super-Bear and admitted for sm_120 only.**

## WSL timing admission — 2026-09-26

Sync `WSL-TIMING-ADMISSION-2026-09-26` (owner direction, [MASTER_AUDIT](../../MASTER_AUDIT.md#consolidated-action-list-2026-09-25), 2026-09-25). **Shared timing contract changed.** Missing `/dev/kfd` or bare metal no longer blocks promotion; the independent-witness method does:

- `profiler_timing`: on WSL, promotion is carried only by a kernel-side clock **of the sample's own target** (`promotion_clock_slots`: `device_wall_clock_ns` for ROCm, `tsc_cycles` for x86, none for NVIDIA). Its admissible witnesses are fixed per clock — HIP event or profiler activity for the device clock, `CLOCK_MONOTONIC_RAW` for the TSC, **never host wall** — at least one must be valid in the same sample, and **every** valid one it names must agree within 5% (`|witness − clock| / witness`, the providers' band). A TSC must carry a frequency from an independent source (`cpuid_leaf_0x15` or a separate calibration interval). Environments are matched exactly; an unrecognized one carries no promotion.
- `target_perf.apply_corpus`: a WSL calibration corpus is selector authority only when `timing_witness.samples` carries at least two admissible WSL timing samples per calibrated device, of that device's target and distinct by clock content — derived evidence, not a declared method.
- The ROCm profiler packet gains a derived `admission_route` (`device_clock_witness`); on it, environment and profiler reasons become `diagnostic_gaps`, the witness sample must name the calibrated image, and the validator re-derives reasons, gaps, route and eligibility from the packet's own inputs. SSD admission on gfx1151 consumes it. (An x86 `tsc_witness` packet route was drafted and **withdrawn**: the probe's `clock_agreement_valid` compares steady_clock with monotonic-raw, not the TSC, so it proved nothing about the TSC.)
- `profiler_cuda_window` drops "bare metal required"; its activity-window / event 5% agreement and 5% overhead gates decide. This is an explicit exception recorded in MASTER_AUDIT and **kept by owner decision (2026-09-26)**: the Nsight activity window is profiler-derived, and its validity on WSL2 is unverified until a Super-Bear packet is recorded.
- Recorders that time with events or host wall only now stamp `kernel_clock_witness_required` instead of a bare-metal reason; committed historical packets are unchanged.

This supersedes the WSL half of `GFX1151-CALIB-BAREMETAL-2026-08-16`.

**NVIDIA outcome: follow-up required.** `profiler_timing` has no NVIDIA kernel-side slot, so sm_120 timing samples cannot promote through it until the non-profiler `%globaltimer` witness lands (DEVICE-CLOCK-DISCIPLINE, owed on Super-Bear); CUDA events alone do not qualify and were not loosened. The CUDA activity-window calibration no longer refuses WSL2 by environment — **whether Nsight on Super-Bear's WSL2 yields valid activity windows is unverified**; recording an SSD calibrated-pairs packet there (`benchmarks/record_ssd_calibrated_pairs.py`, which now accepts WSL2) is the test. `phase5_ingest` and `record_packed_storage_foundation` now state the witness as the blocker.

**Updated 2026-09-26 (`NVIDIA-GLOBALTIMER-MARKER-2026-09-26`):** the `%globaltimer` witness landed and validated, so `promotion_clock_slots("nvidia_sm120")` is now `{device_wall_clock_ns}`, witnessed by `cuda_event_ns` (other compute capabilities still have none). The Nsight activity-window packet on WSL2 is still unrecorded, and `phase5_ingest` / `record_packed_storage_foundation` still time with events only, so their `kernel_clock_witness_required` stamp stays true for them.


## gfx1201 native spectral JVP — sibling outcome — 2026-09-25

Sync `ROCM-SPECTRAL-JVP-GFX1201-2026-09-25` (#850). **Not applicable.** The
shared `NativeJVPArtifact` admission is now `native_jvp.architecture_admits`,
but sm120 remains single-architecture with no per-chip entry. The shared unit
test asserts that sm120 still rejects other architectures.

## Spectral benchmarks: `cold_ms` is the first call — 2026-09-25

Owner `NVIDIA-FFT-WORKSPACE-1`; sync `SPECTRAL-BENCH-COLD-2026-09-25` (#849).
**Changed here.** `benchmarks/spectral/benchmark_nvidia_spectral.py` used to
invoke each case once, untimed, for its correctness check, and then time
`cold_ms`. So `cold_ms` was a second call. It now times the first invocation,
and that result is also what the correctness check reads. The FFT runtime
library, package ABI, spectral arch and device tag (`cuInit`) are probed only
after every case is timed, so no case's first call finds them already loaded.
The device-resident lane still loads the library before it sets up plans,
because it calls the plan ABI directly; its `cold_ms` is the first execute on
a ready plan.

**Validation (Super-Bear, RTX 5070, one full run):** all 24 rows `ok`. The
first case, `fft_c2c_1x1024`, reports 326.5 ms cold (library load, `cuInit`
and plan creation) vs 0.21 ms warm. Later forward cases are 1.2–9.3 ms cold.

**Follow-up required:** the committed `nvidia_spectral_20260925` packets carry
second-call `cold_ms`. Their README says so; a cold figure needs a new
recording, not a relabel. Warm medians are unaffected.

## CUDA spectral: centered-only reflect, cuFFT reverse identities — 2026-09-25

Owner `NVIDIA-FFT-WORKSPACE-1`; sync `ROCM-SPECTRAL-FFT-POLICY-2026-09-25`.
This closes both CUDA follow-ups recorded from the ROCm change (#844).

1. **Reflect only centered frames.** `tessera_nvidia_spectral.cu` reflected a
   non-centered frame past the signal whenever `padMode == 1`. The six affected
   sites were the four frame kernels (forward real/complex, JVP real/complex),
   the `dx` gather's candidate count, and the window reduction.
   `tessera.ops.stft` and `vjp._VJPS["stft"]` zero-fill that frame. Every site
   is now gated on `center && padMode == 1`, inside the kernels, so all
   callers are covered. The `dx` gather's three single-bounce candidates are
   complete only under that gate (a centered reflect requires
   `samples > n_fft/2`).
2. **Algorithm identities.** The CUDA reverse packages were labelled
   `direct_stored_bin_sm120_v1`, `normalized_overlap_add_direct_dft_sm120_v1`
   and, for full spectra, the target-neutral `full_complex_direct_dft_v1`.
   Since #842 they run cuFFT (C2R/R2C one-sided, C2C full), so they are now
   `cufft_stored_bin_sm120_v1` and `normalized_overlap_add_cufft_sm120_v1` for
   both layouts.

**Validation (The-Super-Bear, RTX 5070, both `build/` and `build-nvidia-cuda/`
rebuilt).** The new
`test_noncentered_reflect_frame_is_zero_filled_forward_tangent_and_reverse`
covers a 5-sample signal with `n_fft=16`, `hop=3`, `center=False`,
`pad_mode="reflect"` and a full spectrum. It checks the forward launch against
a zero-filled FFT, the JVP tangent against the linearization, and the reverse
against the reference VJP.

- On `main` (`e6df2191`) it fails at the forward assertion (48/48 elements,
  e.g. 5.98 vs 0.61).
- At `d6b0b25a` it passes, and the FFT/spectral device set is **145 passed,
  4 skipped** (x86 and gfx1151 packages absent).
- On the Mac, the host-free spectral contract tests (53 passed) and `mypy` are
  clean.

**Sibling backends:** ROCm fixed in #844; x86 has its own entry
(`ROCM-SPECTRAL-FFT-POLICY-2026-09-25`); Apple not applicable.

## Runtime libraries built at -O0 in empty-build-type trees — 2026-09-25

Owner `RUNTIME-LIB-OPT-1` (defined in the x86 queue, where the full inventory
lives); sync `RUNTIME-LIB-OPT-1-2026-09-25`.

**Applied 2026-09-26.** The `-O2` runtime-library helper and the
`runtime_library_build.json` record landed. Status and the owed re-measurement
live in the x86 queue entry. Parity is not claimed on this backend: its rows
need re-recording on its own box before this backend's numbers change.
Raw evidence and reproduction scripts: `benchmarks/baselines/runtime_lib_opt_20260925/`.

**Finding (NVIDIA).** On The-Super-Bear, `build/`, `build-nvidia-cuda/` (the
tree the runtime loads), `build-nvidia/` and `build-nv/` are all empty.
`libtessera_nvidia_{fft,gemm,rng}.so` compile with no `-O` flag, so their
**host** code is `-O0`. Device code is unaffected. nvcc hands `ptxas` no `-O`
(its default is `-O3`), and the SASS for `tessera_nvidia_spectral.cu` is
identical with and without `-O3` (9,048 instructions both). Only the host
object changes (16,918 → 19,844 x86 instructions).

Measured effect: a host `-O3` rebuild roughly halved the host-bound
STFT/ISTFT autodiff rows before #842's code fixes
(`benchmarks/baselines/nvidia_spectral_20260925/README.md`). Every NVIDIA
host-side latency recorded from these trees carries `-O0` host code.
`build-assertions/` is `RelWithDebInfo`, per the documented recipe.

**Proposal (not applied; owner decision because it moves baselines):**

1. **Per-target optimization for the runtime libraries only.** Add a
   `tessera_runtime_library_optimization(<target>)` helper under `cmake/`. It
   acts only when `NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES` and
   adds `-O2` for C/CXX/OBJCXX, `-Xcompiler=-O2` for CUDA host code (device
   code is already optimized) and `-O2` for HIP (host and device). It must not
   define `NDEBUG`. Apply it to these CMake targets (target names, not
   output names): `tessera_nvidia_{fft,gemm,rng,ptx_launch}`,
   `TesseraSpectralHIP` (output `libtessera_spectral_rocm.so`),
   `TesseraAppleRuntime`/`TesseraAppleRuntimeShared`,
   `tessera_x86_elementwise`, `tessera_x86_base` and `tessera_jit`
   (`tools/tessera-jit`, also built without `-O` on the Mac and on
   Princess-Luna).
2. **Why not a top-level `RelWithDebInfo` default.** It defines `NDEBUG` for
   every Tessera translation unit. That switches off the MLIR/LLVM header
   assertions (`cast<>`, interface-promise checks) that today run inside
   Tessera's compiler code on every empty-build-type tree, including the three
   boxes whose LLVM is NDEBUG. The runtime library directories contain no
   `assert()`, so optimizing only them costs no checks.
3. **Stamp the level into evidence.** Have each runtime library export its
   compile flags, and have benchmark rows record them beside `route`
   (Decisions #11/#12). A latency from an `-O0` library is not comparable to
   one from `-O3`.
4. **Re-measure, never re-stamp.** Once this lands, every packet whose
   host-side time came from an unoptimized library is stale. Those packets
   must be re-recorded, not relabelled. Until then, cross-box comparisons are
   invalid where one box is `-O0` and the other `Release`: Princess-Luna vs
   Tajasarus x86, and gfx1151 vs gfx1201 composite rows.

**Intentional?** No evidence of it:

- CI, `scripts/build.sh` (default `Release`) and the documented assertions
  recipe (`RelWithDebInfo`, `COMPILER_REFACTOR_PLAN.md`) all set a type. The
  empty trees trace to the canonical configure commands in CLAUDE.md and
  `GETTING_STARTED.md`, which omit `-DCMAKE_BUILD_TYPE`.
- Assertions do not depend on it. In Tajasarus `build-assertions/` (`Release`)
  the assertions LLVM's trailing `-UNDEBUG` follows `-O3 -DNDEBUG`, so
  `NDEBUG` stays undefined.
- The only empty-by-choice-looking tree is Tajasarus
  `build-assertions-nvidia/` (empty + `-UNDEBUG`). It builds the
  `tessera-nvidia-opt` driver, not runtime libraries.

Unverified side note: `src/compiler/autotuning/CMakeLists.txt` sets
`CMAKE_BUILD_TYPE Release` as a directory-scoped normal variable. Its effect on
a single-config generator was not checked.

**Inventory** (`CMakeCache.txt` plus the runtime libraries'
`ninja -t commands`, 2026-09-25):

| Box | Tree | Build type |
|---|---|---|
| Mac | `build` | empty |
| Princess-Luna | `build` | empty |
| Tajasarus | `build` | Release |
| Tajasarus | `build-assertions` | Release |
| Tajasarus | `build-assertions-nvidia` | empty |
| Super-Bear | `build`, `build-nvidia-cuda`, `build-nvidia`, `build-nv` | empty |
| Super-Bear | `build-assertions` | RelWithDebInfo |

Every runtime library in an empty tree compiles with no `-O` flag.

## ROCm spectral FFT policy paths — sibling outcome — 2026-09-25

Sync `ROCM-SPECTRAL-FFT-POLICY-2026-09-25`; owner `NVIDIA-FFT-WORKSPACE-1`.
Two follow-ups are required. Neither was run on sm_120 in the ROCm change.

1. **Unconditional reflect.** The ROCm work found that a non-centered frame
   past the signal under `pad_mode="reflect"` must be zero-filled
   (`tessera.ops.stft` and the reference VJP define it so), but the kernels
   reflected it. `tessera_nvidia_spectral.cu` has the same unconditional
   `padMode == 1` test in its frame and reverse kernels (the
   `(source < 0 || source >= samples) && padMode == 1` sites, the reflect
   candidate count, and the window reduction's `!present && padMode == 1`).
   Nothing canonicalizes `pad_mode` when `center=False`, so the case is
   reachable. It needs an sm_120 repro and fix.

   One consequence is specific to CUDA. The `dx` gather tries three
   single-bounce candidates `{s, -s, 2*samples-2-s}`. A non-centered frame
   over a signal shorter than half of `n_fft` reflects more than once, so the
   reverse is also inconsistent with CUDA's own forward there. Limiting reflect
   to centered framing (where `samples > n_fft/2` is enforced) makes the three
   candidates complete.
2. **Algorithm identities.** `_algorithm_identity` still labels the CUDA
   reverse packages `direct_stored_bin_sm120_v1` and
   `normalized_overlap_add_direct_dft_sm120_v1`. #842 made both FFT-based, so
   the labels misdescribe what runs.


## CUDA spectral: measured, then deepened — 2026-09-25

Owner `NVIDIA-FFT-WORKSPACE-1`; sync `NVIDIA-SPECTRAL-DEEPEN-2026-09-25`.
Every change was chosen from a measured hotspot (`nsys`, `ncu`, cProfile) and
re-measured. Packet and attribution:
`benchmarks/baselines/nvidia_spectral_20260925/` (before at `a58a09db`,
after at `41e008db`, The-Super-Bear RTX 5070, WSL2 wall-clock, no promotion).
**Landed:**
- DCT-II/III are FFT-based (Makhoul), replacing an O(N²) fp64 direct kernel:
  11.95 → 0.34 ms at 64×1024.
- A per-device cuFFT plan LRU and scratch pool serve STFT/ISTFT, their JVPs
  and backward, and conv.
- FFT host entry reuses per-plan staging.
- New device-pointer entry points `tessera_nvidia_fft_execute_{c2c,r2c,c2r}_device_f32`,
  which are async and take a stream (ROCm parity), run at 0.04–0.07 ms.
- `spectral_conv` is one native batched R2C/multiply/C2R call
  (`tessera_nvidia_spectral_conv_f32`): 6.58 → 0.62 ms at 262144×1025.
- STFT/ISTFT backward is FFT-based instead of direct DFT: STFT VJP
  1658 → 1.1 ms, ISTFT VJP 77 → 2.1 ms.
- JVPs are pooled: STFT 12.6 → 1.7 ms, ISTFT 21.3 → 1.9 ms.
- Host staging is skipped for compact layouts.
- Window-gradient reductions run one block per element with a
  deterministic tree sum.

**Validation:** FFT/spectral device set **143 passed, 4 skipped** (x86 and
gfx1151 packages absent) on The-Super-Bear at `41e008db`.

**Still losing to NumPy:** small shapes through the host-buffer entry point
(`fft_c2c_1x1024`, `dct2_64x1024`, `spectral_conv_16384x257`). That is
~0.2 ms of copies and synchronization, not kernels. `spectral_filter` stays a
host complex multiply on purpose.

**Open (recorded, not in this change):**
- The build trees configure with an empty `CMAKE_BUILD_TYPE`, so runtime
  libraries compile host code at `-O0`. Measured only on Super-Bear's two
  trees; a host `-O3` rebuild halved the pre-fix autodiff rows.
- The scheduled contracts still name `sm120_cufft_workspace_v2`
  (`scheduled_fft.py` kernel family) and a nonexistent
  `tessera_nvidia_spectral_filter_f32` (`scheduled_spectral.py` native
  entry). Both are hashed into the schedule digest, so renaming them is a
  digest change with its own evidence re-recording.
- The STFT/ISTFT JVP and backward still take host buffers. A device-resident
  AD entry is the next step for the small-shape rows.

**Sibling backends:** ROCm follow-up required (below, in its queue); Apple
and x86 not applicable (their queues state why).

**Corrected 2026-09-25 (Codex review on #842).** The compact-layout fast path
indexed a caller's strides before any check, so a null shape or stride
descriptor through the C ABI segfaulted where `packHostLayout` had returned
the entry point's layout status. Every f32 layout entry point now validates
its data descriptors first, and the three ISTFT storage wrappers, which size
buffers from the shape early, validate rank, extents and `outputSamples`. A
subprocess probe over all eleven entry points died with SIGSEGV before the fix
and passes after (The-Super-Bear, FFT/spectral device set 144 passed,
4 skipped).


## ROCm executable-pipeline follow-ups — 2026-09-24

Sync `ROCM-EXEC-PIPELINE-2026-09-24`; owner `NVIDIA-FFT-WORKSPACE-1` (the
cuFFT plan/workspace contract, sync `NVIDIA-FFT-WORKSPACE-1-2026-08-22`
below). **Follow-up required (spectral plan cache) — fixed by the entry
directly below.** #835 fixed ROCm
spectral plan caches that were keyed without the image/device that created
the plan. The CUDA equivalent has the same shape: `_nvidia_fft_plans`
(`python/tessera/runtime.py`, used by `_nvidia_fft_c2c_rows` and
`_nvidia_fft_real_rows` for r2c/c2r) keys
cuFFT plans and their workspaces by `(kind, batch, length)` only, while a cuFFT
plan belongs to the CUDA context current at creation. A process that switches
its CUDA device would reuse another device's plan. Not exercisable on the
fleet (one NVIDIA GPU per box). **Not applicable (rest):** no NVIDIA GPU lowering runs
`convert-vector-to-llvm` (checked: the pipelines lower through
`convert-gpu-to-nvvm`), and `ROCM_CANONICAL_LDS_KNOB_UNSUPPORTED` and the LDS
knob forwarding are confined to the ROCm WMMA generator.

## CUDA FFT plans bound to their device — 2026-09-25

Owner `NVIDIA-FFT-WORKSPACE-1`; sync `ROCM-EXEC-PIPELINE-2026-09-24`. Closes
the follow-up above. `libtessera_nvidia_fft` moves to
`tessera.nvidia.cuda_fft_workspace.v3`: each cuFFT plan records the CUDA device
current at creation, execute refuses without running when the current device
differs, and `tessera_nvidia_fft_current_device` reports the device
through the library's own CUDA runtime. `_nvidia_fft_plans` keys on
`(device, kind, batch, length)` with the device read through that library;
a v2 library is refused as unloadable. **Shared contracts:** the NVIDIA FFT
C ABI only (versioned); no IR, op, dtype or numerical-policy change.
**Validation (The-Super-Bear, RTX 5070 CC 12.0, CUDA 13.4.59, driver
610.88):** the FFT/spectral device set (`test_fft_workspace`,
`test_spectral_{autodiff,jvp,policy}`, `test_native_vjp_execution_certificates`
plus the spectral/JVP/capability unit files) went from 121 passed at `main`
to 125 passed with the fix (the new device-key test and three host-only
device-switch tests), 4 skipped in both (x86 and gfx1151 packages, not on this
box). **Missing evidence:** a real mid-process device switch -- one NVIDIA GPU
per fleet box; the host-only fake-library tests cover it and fail on the old
key. **Host note:** both Super-Bear build trees were re-pointed from CUDA
13.3.73 to 13.4.59 because the enforced toolkit pin refused 13.3
(`build-nvidia-cuda/` had been configured against `/usr/local/cuda-13.3`;
`build/` had a stale cached compiler identity); caches backed up as
`CMakeCache.txt.pre-cuda134-20260925`. **Sibling backends:** ROCm fixed in
#835; Apple and x86 not applicable (no per-device FFT plan cache).

**Corrected 2026-09-25 (Codex review on #840).** v3 returned **3** for the
device refusal, but 3 was already the library's generic "a CUDA/cuFFT call
failed during execution" status, so the runtime reported every such failure as
a foreign-device plan. v4 gives the refusal its own status, **4**; 3 keeps its
meaning and its generic message, and a host-only test pins that distinction.
A failed `cudaGetDevice` inside execute is also status 3, not 4 (Codex review
on #841); an `LD_PRELOAD` shim over the library's dynamic `cudaGetDevice`
drives both branches on the RTX 5070 -- which also exercises the
device-mismatch refusal on hardware, unreachable on a one-GPU box otherwise.

## ROCm review fixes: spectral composite + Quark W4A4 — 2026-09-24

Sync `ROCM-REVIEW-SPECTRAL-QUARK-2026-09-24`. NVIDIA parity is not applicable: no CUDA spectral, MXFP4 or GEMM code changes. Follow-up worth checking separately: whether the CUDA spectral loader has the same only-one-chip prebuilt preference; not evaluated here.

## ROCm Gumiho native step — 2026-09-23

Owner COMPILER-DEVEX-1 / DK1; sync `ROCM-GUMIHO-NATIVE-STEP-2026-09-23`.
The example backend contract changes shared model composition only. No CUDA
Gumiho native package or exact-device result follows; NVIDIA parity is a
separate follow-up if this example is added to its backend queue.


## ROCm spectral target feedback and MLA example — 2026-09-23

Owner COMPILER-DEVEX-1 / DK1; sync
`ROCM-SPECTRAL-MLA-EXAMPLE-2026-09-23`. ROCm source-target selection and an
opt-in HIP MLA decode-step example change no SM120 FFT or MLA package, CUDA
ABI, or NVIDIA selector. CUDA parity is not applicable; no ROCm device result
transfers to the NVIDIA MLA lane.

## ROCm spectral live-architecture fix — 2026-09-23

Owner COMPILER-DEVEX-1 / ROCM-FFT-PREBUILT; sync
`GFX1201-SPECTRAL-LIVE-ARCH-2026-09-23`. The numerical availability probe
and exact live-architecture source selection are ROCm candidate behavior.
They change no cuFFT package, CUDA ABI, SM120 schedule, or NVIDIA execution;
CUDA parity is not applicable and its selector is unchanged.

## Quark W4A4 independent projection probe — 2026-09-23

Owner ROCM-MXFP4-W4A8-1 / IKF-1; sync
`GFX1201-QUARK-INDEPENDENT-W4A4-2026-09-23`. The pinned independent
MXFP4 projection-slice arithmetic can inform a future CUDA oracle, but the
manual E2M1/E8M0 ABI and BF16 proof belong to gfx1201. They do not imply
NVFP4 byte compatibility, an SM120 W4A4 kernel, or NVIDIA device proof;
CUDA selection is unchanged.

## Quark byte-oracle preflight — 2026-09-23

Owner ROCM-MXFP4-W4A8-1 / IKF-1; sync
`GFX1201-QUARK-BYTE-ORACLE-2026-09-23`. The host-only candidate
checkpoint decode is not an NVFP4 converter, CUDA W4A4 activation
carrier, or SM120 numerical proof. NVIDIA selection remains unchanged.

## GPT-OSS-20B ROCm full-expanded capacity packet — 2026-09-23

Owner ROCM-MXFP4-W4A8-1 / IKF-1; sync
`GFX1201-GPT-OSS-20B-MXFP4-CAPACITY-2026-09-23`. The pinned checkpoint
inventory does not establish an NVFP4 layout, CUDA device budget, or
NVIDIA exact-device proof. Tajasarus's physical-capacity refusal is
architecture-specific; NVIDIA selection remains unchanged.

## GFX1201 MXFP4 prefill sweep schema — 2026-09-23

Owner ROCM-MXFP4-W4A8-1 / IKF-1; sync
`GFX1201-MXFP4-PREFILL-SWEEP-2026-09-23`. Not applicable to CUDA: the
RDNA4 TN2/TN4 schedules, HIP-event/HSACO packet, and gfx1201 free-memory
snapshots do not establish an NVFP4/PTX selector or a loaded-model budget.
No NVIDIA exact-device result transfers.

## Quark MXFP4 checkpoint metadata and W4A4 refusal — 2026-09-23

Owner ROCM-MXFP4-W4A8-1 / IKF-1; sync
`GFX1201-QUARK-MXFP4-CHECKPOINT-2026-09-23`. The new Quark assessment field
and Quark source-layout refusal do not specify an NVFP4 packing contract,
CUDA activation producer, or NVIDIA execution route. No SM120 proof or
selector state transfers from the Qwen/GLM metadata assessment.

## GFX1201 folded-prefill experiments and runtime guard — 2026-09-23

Owner ROCM-MXFP4-W4A8-1 / IKF-1; sync
`GFX1201-MXFP4-PREFILL-EXPERIMENTS-2026-09-23`. Not applicable to CUDA:
the manual HIP safe-scale ABI, RDNA4 BN128 schedule, HSACO timing schema,
and ROCm launch-time certificate guard do not define an NVFP4/PTX route.
No NVIDIA selector or exact-device proof transfers from Tajasarus.

## GFX1201 bounded A-offset benchmark schema — 2026-09-23

Owner ROCM-MXFP4-W4A8-1 / IKF-1; sync
`GFX1201-PACKED-A-OFFSET32-2026-09-23`. Not applicable to CUDA: the
RDNA4 A-producer arithmetic and v6 HIP-event/HSACO ISA census define no
PTX/NVFP4 route. Shared ABI, numerical policy, and NVIDIA selection
remain unchanged; no SM120 proof transfers from Tajasarus.

## GFX1201 A-base benchmark and provenance correction — 2026-09-23

Owner ROCM-MXFP4-W4A8-1 / IKF-1; sync
`GFX1201-PACKED-A-BASE-2026-09-23`. Not applicable to CUDA: the A-stage HIP
address ablation and v5 HSACO/HIP-event packet define no PTX/NVFP4 route.
The vector-pair sync-key correction is ROCm artifact provenance only; shared
numeric policy, NVIDIA selection, and SM120 proof are unchanged.

## GFX1201 paired-lane MXFP4 benchmark schema — 2026-09-23

Owner ROCM-MXFP4-W4A8-1 / IKF-1; sync
`GFX1201-PACKED-VECTOR-PAIR-2026-09-23`. Not applicable to CUDA: the
RDNA4 fragment-word load and v4 HIP-event/HSACO benchmark schema do not
define an NVFP4/PTX schedule. Shared numerical policy and NVIDIA selection
are unchanged; no SM120 proof transfers from Tajasarus.

## GFX1201 model-owned MXFP4 graph and producer ablation — 2026-09-23

Owner ROCM-MXFP4-W4A8-1 / IKF-1; sync
`GFX1201-MXFP4-GRAPH-MODEL-PRODUCER-2026-09-23`. Not applicable to CUDA:
the exclusive HIP buffer lease, error quarantine, RDNA4 producer ablation,
and benchmark schema do not define a CUDA graph or NVFP4 schedule. The
shared numeric policy and NVIDIA execution matrix are unchanged; no
NVIDIA exact-device proof transfers from Tajasarus.

## GFX1201 three-kernel MXFP4 graph pipeline — 2026-09-23

Owner ROCM-MXFP4-W4A8-1 / IKF-1; sync
`GFX1201-MXFP4-GRAPH-PIPELINE-2026-09-23`. Not applicable to CUDA:
the HIP OCP E4M3 producer, packed RDNA4 GEMM, BF16 consumer, and exact-M
HIP graph pool do not establish a CUDA graph or NVFP4 schedule. The
benchmark schema is AMD-specific; no NVIDIA device result transfers.

## GFX1201 device-owned MXFP4 HIP graph — 2026-09-23

Owner ROCM-MXFP4-W4A8-1 / IKF-1; sync
`GFX1201-MXFP4-DEVICE-GRAPH-2026-09-23`. Not applicable to CUDA:
the capture APIs, RDNA4 packed HSACO, and benchmark packet are HIP-only.
The existing NVIDIA staging/runtime contract is unchanged; no CUDA graph
or device proof transfers from Tajasarus.

## GFX1201 resident MXFP4 HIP lifecycle — 2026-09-23

Owner ROCM-MXFP4-W4A8-1 / IKF-1; sync
`GFX1201-MXFP4-RESIDENT-HIP-2026-09-23`. Not applicable to CUDA
execution: this manual HIP module/stream/buffer lifecycle and its benchmark
schema are gfx1201-specific. The NVIDIA staging arena is a separate runtime
contract; no CUDA MXFP4 selection or exact-device timing is implied.

## GFX1201 packed-word permute decode — 2026-09-23

Owner ROCM-MXFP4-W4A8-1; sync `GFX1201-PACKED-PERMUTE-DECODE-2026-09-23`.
Not applicable to CUDA execution: the RDNA4 permute instruction and AMD
fragment-order HSACO are not an NVFP4/PTX schedule. The matched benchmark v3
schema is gfx1201-only and supplies no NVIDIA performance or selector proof.

## GFX1201 packed producer staging ablation — 2026-09-23

Owner ROCM-MXFP4-W4A8-1; sync `GFX1201-PACKED-STAGING-ABLATION-2026-09-23`.
Not applicable to CUDA execution: the opt-in HIP producer ordering, RDNA4
fragment layout, and HSACO ISA census do not define an NVFP4/PTX schedule
or NVIDIA performance result.

## GFX1201 packed-folded decode proof — 2026-09-23

Owner ROCM-MXFP4-W4A8-1; sync `GFX1201-PACKED-FOLDED-DECODE-2026-09-23`.
Not applicable to CUDA execution: the manual HIP launcher and RDNA4 packed
fragment decode do not define an NVFP4/PTX schedule. The AMD-only benchmark
schema and Radiance timing do not constitute NVIDIA evidence or selector
admission.

## GFX1201 packed-folded physical ABI — 2026-09-22

Owner ROCM-MXFP4-W4A8-1; sync `GFX1201-PACKED-FOLDED-ABI-2026-09-22`.
Not applicable to CUDA execution: RDNA4 fragment order and the AMD Target
ABI do not define NVFP4/PTX packing. Shared Graph verification admits only
the explicit gfx1201 physical contract; no NVIDIA selector or proof changes.

## GFX1201 folded K64 staging codegen — 2026-09-22

Owner ROCM-MXFP4-W4A8-1; sync `GFX1201-FOLDED-K64-STAGING-2026-09-22`.
Not applicable to CUDA execution: the new K64 specialization changes only
the gfx1201 HIP vector-copy path and AMD benchmark packet. No PTX/NVFP4
carrier, numerical policy, or NVIDIA device proof changes.
The timed-HSACO ISA evidence schema does not apply to cubin/PTX or Nsight
measurements.


## GFX1201 MXFP4 receipt and device coverage — 2026-09-22

Owner ROCM-MXFP4-W4A8-1; sync `GFX1201-MXFP4-RECEIPT-COVERAGE-2026-09-22`.
Not applicable to CUDA: the corrected HSACO-byte receipt and RDNA4 folded
carrier have no NVFP4/PTX consumer. Pinned Radiance timing and gfx1201
lowering/launch tests are not NVIDIA evidence or selector input.


## GFX1201 folded frontend and matched-census correction — 2026-09-22

Owner ROCM-MXFP4-W4A8-1; sync `GFX1201-FOLDED-FRONTEND-2026-09-22`.
Not applicable to CUDA execution: the new typed authoring path and receipt
bind a gfx1201-only folded E4M3/E8M0 ABI and HSACO. Radiance's
fragment-order WPERM gate and static AMD ISA census do not change NVFP4,
PTX, Nsight, or the CUDA selector. No timing or device proof transfers.
The losing AMD B non-temporal load ablation is not a CUDA cache-policy result.

## GFX1201 external phase-probe preflight — 2026-09-22

Owners IKF-1 / ROCM-MXFP4-W4A8-1; sync
`GFX1201-FOLDED-PROFILER-CARRIER-2026-09-22`. Not applicable to CUDA:
Tajasarus lacks `/dev/kfd`, so its rocprofv3 PC-sampling preflight refuses
without phase attribution. The packet has no Nsight/CUPTI consumer, changes
no NVIDIA clock schema or selector, and provides no transferable CUDA proof.
The shared Graph/Schedule/Tile carrier now names a distinct folded full-K
partial, but the E4M3 `[N,K]` payload, E8M0 row reference, BM256/TM4
producer and Target ABI are gfx1201-only. No CUDA package consumes it.

## GFX1201 folded scale fix and phase-probe refusal — 2026-09-22

Owners ROCM-MXFP4-W4A8-1 / IKF-1; sync
`ROCM-MXFP4-IKF-DIAGNOSTIC-2026-09-22`. Not applicable to CUDA execution.
The HIP fold epilogue now combines two FP32 scales before accumulation and
uses a rare FP64 path for extreme scale products; neither is a CUDA ABI change.
The synchronized same-CTA wall-clock trace is AMD-only and fails its structural
perturbation gate. NVIDIA's `%globaltimer` IKF lane and exact-device proof
remain separate; the packet is not CUDA cost-model training data. Super-Bear
has Nsight Systems, Nsight Compute, and CUPTI for its later uninstrumented
span/counter/PC-sample comparisons, with CUDA provenance kept separate from
Tajasarus gfx1201 evidence. The proposed Graph→Schedule→Tile→Target feedback
lineage remains a shared plan only; no CUDA carrier or selector changes here.

## GFX1201 folded MXFP4 ABI and benchmark schema — 2026-09-22

Owner ROCM-MXFP4-W4A8-1; sync `ROCM-MXFP4-FOLDED-PREFILL-2026-09-22`.
Not applicable to CUDA execution. The load-time E8M0 row-reference fold and
BM256/TM4 HIP schedule use a distinct RDNA4 ABI; NVIDIA's UE8M0/NVFP4
physical contract remains independently owned. The added matched-timing
packet changes no CUDA selector or benchmark consumer, and no gfx1201 proof
transfers.

## GFX1201 MXFP4 K-step and benchmark-evidence schema — 2026-09-22

Owner ROCM-MXFP4-W4A8-1; sync
`ROCM-MXFP4-KSTEP-PREFILL-2026-09-22`. Not applicable to CUDA execution. The
portable isolated-scale-group attributes are shared IR vocabulary, but this PR
adds only an AMD lowering and gfx1201 native package. Alternating timing order
and retained HSACO ISA/resource fields change the ROCm evidence schema only;
no PTX ABI, schedule, numerical policy, selector, or owning-host claim changes.

## GFX1201 MXFP4 versioned layout ABI — 2026-09-22

Owner ROCM-MXFP4-W4A8-1; sync
`ROCM-MXFP4-FRAGMENT-ABI-2026-09-22`. Not applicable to CUDA physical
execution. The gfx12 lane-word permutation and ABI are incompatible with
NVIDIA's NVFP4 block-scale layout; the HIP graph-capture proof changes no CUDA
launcher or PTX schedule. NVIDIA layout/version work remains separately owned,
and no gfx1201 timing or selector decision transfers.

## Shared scaled-partial carrier and MXFP4 policy — 2026-09-21

Owner ROCM-FP8-BLOCKSCALE-1 / ROCM-MXFP4-W4A8-1; sync
`ROCM-MXFP4-SCALED-PARTIAL-CARRIER-2026-09-21`. The shared Graph/Schedule/Tile
carrier can now state isolated scale-group partials and quantified approximate
fold loss. NVIDIA follow-up is required only if its distinct UE8M0/NVFP4
physical form adopts this carrier; the packed E2M1/E8M0 K32 ABI, Target op,
and exact-device proof remain gfx1201-only. The MXFP4 evidence packet's added
generic-materializer row is not applicable to CUDA and changes no shared
benchmark semantics.

## GFX1201 scheduled-proof provenance schema — 2026-09-21

Owner ROCM-2; sync `GFX1201-SCHEDULED-SKIP-CLOSURE-2026-09-21`. Not
applicable to CUDA execution. The evidence extension binds a gfx1201 run to
one selected ROCm toolkit and its loaded HIP runtime library. It changes no
PTX, CUDA launch ABI, dtype, numerical policy, or NVIDIA device row, and no
ROCm proof transfers to NVIDIA.

## GFX1201 exact MXFP4 execution — 2026-09-21

Owner ROCM-MXFP4-W4A8-1; sync
`ROCM-MXFP4-PHYSICAL-CONTRACT-2026-09-21`. Not applicable to CUDA physical
execution. This route consumes OCP E8M0-per-32 MXFP4 through RDNA4 FP8 WMMA;
it does not alter NVIDIA's distinct NVFP4 E4M3-per-16 scale ABI or block-scale
MMA. No PTX, CUDA launch ABI, dtype, policy, or device claim changes, and the
gfx1201 proof does not transfer.

## GFX1201 dtype and exact-executor closure — 2026-09-21

Owner ROCM-2 / NUMPOL-CARRIER-1; sync
`GFX1201-DTYPE-EXACT-EXECUTOR-2026-09-21`. Not applicable to CUDA physical
execution: the exact-executor guard and dense dtype-state split are ROCm-only,
and the shared inventory only attaches existing gfx1201 proof metadata. No
NVIDIA IR, ABI, dtype, numerical, runtime, or device claim changes.

## ROCm MXFP4 physical-contract sibling assessment — 2026-09-21

Owner ROCM-MXFP4-W4A8-1; sync
`ROCM-MXFP4-PHYSICAL-CONTRACT-2026-09-21`. The host reference models OCP MXFP4
with E8M0-per-32 weights for a gfx1201 W4A8 route; it does not change NVIDIA's
distinct NVFP4 E4M3-per-16 scale ABI, PTX block-scale MMA, canonical dtypes, or
CUDA execution. No CUDA parity or device proof is inferred.

## GFX12 public projection and D=128 load batching — 2026-09-21

Owner ROCM-2 / ROCM-4; sync
`GFX12-PUBLIC-LINEAR-LOAD-BATCH-2026-09-21`. The shared capability/execution
registries gain only exact `rocm_gfx1201` rows, and load batching changes only
the ROCm generator. CUDA IR, ABI, runtime, dtype, and numerical contracts are
unchanged. No CUDA proof is inferred, and RX 9070 XT evidence does not promote
`gfx1200`.

## GFX12 exact-target alias correction — 2026-09-21

Owner ROCM-2 / ROCM-4; sync `GFX12-EXACT-TARGET-SUPPORT-2026-09-21`. Shared
target normalization now maps RX 9070 products to `rocm_gfx1201`, RX 9060/RX
9050 products to `rocm_gfx1200`, and refuses ambiguous RDNA4 family aliases.
Not applicable to CUDA physical execution: no NVIDIA IR, ABI, dtype, numerical,
or runtime contract changed, and no ROCm proof transfers to this backend. The
ROCm-only post-launch selected-HIP-device attestation fix does not affect CUDA
attestation. The shared dtype-flow audit gained only the missing exact
`rocm_gfx1201` ISA row; NVIDIA rows and generated states are unchanged.

## Compiler-example MoE optional binding — 2026-09-20

Owner E2E-REAL-6; sync `EXAMPLE-MOE-OPTIONAL-BINDING-2026-09-20`. Shared
Graph/runtime artifact metadata now names the optional `scores` / `route`
operands of `tessera.moe` instead of leaving the existing NVIDIA consumer to
interpret an unlabeled tensor tail. Host-free artifact tests cover the NVIDIA
target. No CUDA launch, numerical, or performance claim is added; accelerator
runtime qualification remains NVIDIA-owned and requires the designated host.

## Logical sparse matrices, reader release and floor migration — 2026-09-13

Owner E2E-REAL-6 / ROCM-2; sync `LOGICAL-SPARSE-OWNERSHIP-FLOOR-2026-09-13`. Shared floor Graph registration preserves the public unary shape/type contract; its new physical Schedule producer is x86-only. Logical sparse packing/accumulation is gfx1201-only and establishes no NVIDIA sparse layout or execution proof. Isolated ANN now carries an explicit CUDA device ordinal into worker context selection and replacement; host IPC tests verify ordinal preservation, confirmed death and fresh health-probe admission. Exact selected-device CUDA execution/recovery remains a follow-up. HIP attention reader futures do not establish CUDA ownership parity.

Evidence: [bounded compiler/device packet](../../../../benchmarks/baselines/sparse_ownership_migration_20260913/README.md).

## Selected HIP device follow-through — 2026-09-13

Owner ROCM-2; sync `HIP-SELECTED-DEVICE-2026-09-13`. Shared runtime change assessed: selected-device discovery and GEMM specialization are HIP-only. No nvidia ABI, device execution or numerical policy change; no sibling execution proof is inferred.

## PR 747 review follow-through — 2026-09-13

Owner ROCM-2; sync `GFX1201-SAVED-SPARSE-SCHEDULE-2026-09-13`. Not applicable to this backend runtime: the correction is confined to HIP attention module identity and selected-device compilation. No sibling device proof or ABI change is claimed. The CPU CI test double now accepts and verifies the architecture passed to ROCm device-library selection.

## Saved LSE and sparse Schedule handoff — 2026-09-13

Sync: `GFX1201-SAVED-SPARSE-SCHEDULE-2026-09-13`; owner E2E-REAL-6 / ROCM-2. Not applicable to CUDA physical execution: gfx1201 sparse fragment layouts and saved-LSE evidence do not establish sparse MMA or CUDA ownership proof. Follow-up required for any shared reader/calibration adoption.

Evidence: [saved/sparse Schedule packet](../../../../benchmarks/baselines/gfx1201_saved_sparse_schedule_20260913/README.md).

## GFX1201 streams, readers and sparse Target IR — 2026-09-13

Sync: `GFX1201-STREAM-SPARSE-IR-2026-09-13`; owner E2E-REAL-6 / ROCM-2. CUDA requires its own stream/reader device proof and sparse MMA storage/index contracts; HIP/SWMMAC evidence does not transfer. Shared-contract assessment: no public operation or dtype was added; the sparse op is an internal ROCm Target primitive. Sibling device validation is not claimed. Follow-up required for any adoption of the external-reader lifetime interface.

Evidence: [stream/sparse IR packet](../../../../benchmarks/baselines/gfx1201_stream_sparse_ir_20260913/README.md).

## GFX1201 resident ownership and sparse packing — 2026-09-13

Sync: `GFX1201-RESIDENT-SPARSE-2026-09-13`; owner E2E-REAL-6 / ROCM-2. Shared runtime ownership and bounded-dynamic Schedule admission assessed. ROCm worker ownership and SWMMAC register layout are not applicable to nvidia; its device ownership and numerical proof remain independent. No sibling parity or performance closure is claimed.

Evidence: [resident/sparse packet](../../../../benchmarks/baselines/gfx1201_resident_sparse_20260913/README.md).

## GFX1201 public attention AD — 2026-09-13

Sync: `GFX1201-PUBLIC-AD-2026-09-13`; owner E2E-REAL-6 / ROCM-2. Shared attention package/certificate identity changes assessed. ROCm WMMA and HIP backward admission are not applicable to this backend; its paired AD, tape ownership and physical proof remain independent. No sibling device or performance closure is claimed.

Evidence: [public AD packet](../../../../benchmarks/baselines/gfx1201_public_attention_ad_20260913/README.md).

## GFX1201 scheduled package integration — 2026-09-13

Sync: `GFX1201-PACKAGES-2026-09-13`; owner E2E-REAL-6 / ROCM-2. Shared driver/Tile contracts assessed. The new gfx1201 profiles and ROCm launch ABIs are not applicable to this backend; its numerical, physical scheduling and runtime evidence remain independent. No sibling performance or device closure is claimed.

Evidence: [package packet](../../../../benchmarks/baselines/gfx1201_scheduled_packages_20260913/README.md).

## RDNA4 WMMA datatype audit — 2026-09-13

Sync `GFX1201-WMMA-DTYPES-2026-09-13`.

ROCm's operand-pair inventory and compact accumulator map do not change
SM120 MMA/WMMA packing or NVVM selection. Unsigned byte/nibble modifiers are
ROCm Tile consumers, not new public unsigned dtypes or CUDA admissions.
No NVIDIA execution or performance proof is inferred from the RDNA4 corpus.

Evidence: [dtype packet](../../../../benchmarks/baselines/gfx1201_wmma_dtypes_20260913/README.md).

## Attention pairing and loading experiment — 2026-09-13

Sync `GFX1201-ATTENTION-PAIR-2026-09-13`.

Shared attention projection preserves NVIDIA's existing result contract.
The new ROCm replay checks and gfx1201 native composition do not change NVVM
or resident CUDA tape behavior. Follow-up required for any shared public AD
change; no CUDA performance evidence is inherited from these experiments.

Evidence: [paired attention packet](../../../../benchmarks/baselines/gfx1201_attention_pair_20260913/README.md).

## gfx1201 integration sibling check — 2026-09-13

Sync `GFX1201-INTEGRATION-2026-09-13` (F0/F2/EVIDENCE-PACKET-1).
Super-Bear's main checkout remains clean at ca0e0c7263a443ea62974dffb7a33e6f786cff66.
Its assertions-enabled compiler rebuild and 253 focused tests pass. All 34
scalar/vector arithmetic probes execute on SM120 with CUDA SDK 13.4.1.
RDNA4 fragment mapping and HIP launch admission are not applicable to NVVM;
this independently verifies the existing main compiler, not the uncommitted
ROCm increment or NVIDIA performance promotion.

Evidence: [integration packet](../../../../benchmarks/baselines/gfx1201_integration_20260913/README.md).

## gfx1201 sibling assessment — 2026-09-13

Cross-backend sync `GFX1201-FOUNDATION-2026-09-13` (ROCM-2 / F0 / F2).
The native storage package target check now admits the exact ROCm gfx1201 pair
and still rejects incompatible backend/chip pairs, including NVIDIA/gfx1201.
NVIDIA remains SM120 in this package envelope. RDNA4 accumulator and HIPRTC
changes are not applicable to NVVM/PTX. Follow-up: preserve CUDA 13.4.1 owning-
device validation; no NVIDIA performance or execution evidence is transferred.

## Host-assumption cleanup sibling assessment — 2026-09-20

Cross-backend sync `HOST-ASSUMPTION-CLEANUP-2026-09-20` (PR #785, follow-on to #784). Shared surfaces
changed: `scripts/_rocm_env.sh` (arch detection), `cmake/TesseraToolchainPins.cmake`
+ top-level `CMakeLists.txt` (LLVM/CUDA/HIP pins made EXACT and wired for the
first time), `scripts/bump_toolchain_pins.py` (new), `scripts/install_test_deps.sh`
(interpreter default), `tests/tessera-ir/lit.cfg.py` (new `tessera-clifford`
feature), and `tests/unit/test_foreign_target_host_claims.py` (new drift gate
over every backend's tests). No IR, ABI, dtype/op registration, diagnostic
code, or numerical-policy change on any backend.

**NVIDIA outcome: follow-up required (two pre-existing findings).**
`tessera_pin_cuda_toolkit()` is wired behind `TESSERA_ENABLE_CUDA` and verified
both directions on The-Super-Bear: accept reporting `pinned CUDA Toolkit
13.4.59`, reject at a forced 13.9. The pin is now EXACT rather than a floor,
and the sm_120 Lion lane is the argument recorded in the code — nvcc 13.4 emits
PTX 9.4 while driver 610.88 JITs only <= 9.3, so "a newer toolkit is fine" cost
a week of opaque `rc=3`.

Two pre-existing issues found on that box and **not fixed here**:

1. `build-nvidia-cuda/` (configured 2026-09-17) caches
   `CMAKE_CUDA_COMPILER=/usr/local/cuda-13.3/bin/nvcc` while the fleet pin is
   13.4 and `/usr/local/cuda → 13.4`. That tree is what the NVIDIA native
   runtime loads `tessera-nvidia-opt` and the PTX launcher from, so results
   from it are CUDA 13.3 results recorded under a 13.4 pin (Decision #11). The
   new pin refuses that configure until it is reconfigured or the pin moves
   deliberately.
2. `CMAKE_CUDA_ARCHITECTURES=75` in the same tree — sm_75 is Turing, the box is
   sm_120. Flagged, not asserted: what it affects in an NVRTC-at-load lane is
   not established.

The-Super-Bear went down for maintenance during this work, so neither is
re-verified and no new sm_120 device proof is claimed.
## Current integrated-plan handoff

[The integrated compiler plan](../../compiler/INTEGRATED_COMPILER_PLAN.md) owns
sequencing; [the backend audit map](../README.md) defines document authority.
This queue owns architecture-specific execution and evidence. Start with the
[2026-09-05 native checkpoint, packed/state boundaries and ownership audit](#native-checkpoint-packedstate-boundaries-and-ownership-audit--2026-09-05)
entry below (sync `IR-NATIVE-FOUNDATION-1`, E2E-REAL-5 / W2.4 / W2.4a).

That entry assesses the bounded saved-LSE, INT4 and paged-read migrations and
ownership spike. Allocation-scoped release, control-flow lifetime, remaining
Graph-owned constructors and architecture-specific performance proof stay open.
Earlier synchronization notes retain their original scope and date; statements
such as “no follow-up owed” apply to that increment, not the whole backend.


## Cross-backend sync `FRONTEND-DTYPE-BOUNDARY-2026-09-03`

A **shared Graph IR diagnostic boundary and dtype annotation contract** landed
in [PR #706](https://github.com/gstoner/tessera/pull/706), so all four backends
are assessed here per the integrated plan's PR rule 4.

Three shared changes, none backend-specific:

1. **`GRAPH_IR_UNRESOLVED_ELEMENT_TYPE`** (`graph_ir.unresolved_element_type_diagnostics`)
   — a tensor with no element type renders `tensor<...x?>`, which MLIR rejects.
   The Apple value lane now consults this preflight *before* rendering, so the
   recorded reason names the argument and the missing semantic key (Decision
   #21a) instead of the parser's symptom. Renders are byte-unchanged.
2. **Tracer `loc`** — every traced op carries the user's call site, emitted as
   repo-relative `loc("file":line:col)` in the canonical (parser-bound) render
   only. Decision #13; the paren/golden render is byte-identical.
3. **Dtype annotations** — `Tensor["M","K","bf16"]` binds a trailing dtype
   instead of reading it as a third dim name; `tessera.bf16["M","K"]` keeps its
   `dim_names` and renders symbolic dims as `?`. `tf32` and the planned/gated
   set (`uint*`, `complex*`, `mxfp*`) are refused **by name** rather than
   demoted to a dimension (#15a/#21a).

Verification for all three was on the **Mac**, host-independent lanes only. No
device claim is made or transferred by this entry; the outcomes below are
contract assessments, not exact-device results (Decision #26).

**NVIDIA outcome: not applicable as written; the shared rules still bind.**
No NVIDIA pipeline consults the new preflight — it gates only the Apple value
lane — and the sm_120 packagers construct their own Tile IR rather than
re-rendering decoration-time Graph IR, so nothing here changes NVIDIA lowering
or numerics. Two shared facts do apply the next time this queue touches the
frontend: an unresolved element type is now a *named* refusal (so a future
NVIDIA parser-bound route should call the preflight rather than surface a
parse error), and traced Graph IR carries `loc`, which is available for
NVIDIA diagnostics for free. **No follow-up owed; no evidence transferred.**

## Cross-backend sync `DEVICE-CLOCK-DISCIPLINE-2026-08-31`

A **shared runtime timing contract** now decides which clock a device latency
may be read from, so all four backends are assessed here per AGENTS.md.

`runtime._select_rocm_latency_ms` ranks up to three clocks for one timed loop:

1. **`wall_clock64` (in-kernel)** — a device-side counter at a constant,
   queryable rate (`hipDeviceAttributeWallClockRate`; 100 MHz / 10 ns ticks on
   gfx1151). The only one that is both kernel-only *and* independent of the
   host event API. Unlike `clock()`, its rate does not move with DVFS.
2. **HIP events**, accepted only inside a two-sided band against the host wall
   clock.
3. **The host wall clock**, which includes launch overhead and can therefore
   only make a kernel look slower. A benchmark must not be able to flatter
   itself.

**Measured on gfx1151, 20 launches of the generic fused kernel — all three
agree to four significant figures:**

| shape | wall | event | `wall_clock64` | device/event |
|---|---|---|---|---|
| 256³ | 82.6946 ms | 82.5909 ms | 82.5600 ms | 1.000 |
| 512³ | 498.0912 ms | 497.9570 ms | 497.8904 ms | 1.000 |

The ordering `wall > event > device` is exactly right: wall includes launch
overhead, the event brackets the stream, `wall_clock64` measures the kernel
span. This is a mutual validation with an **independent witness**, not the
weaker "the event agrees with the host clock".

**Two rules that came out of this and generalize beyond ROCm.**

* **`hipEventSynchronize` is mandatory; `hipDeviceSynchronize` is not the way
  to get it.** Launches are async, so without an event (or stream) sync the
  wall clock times the *enqueue*, producing a catastrophically small number
  that then drags the acceptance band down with it. A device-wide barrier does
  work, but halts every stream — it is now kept strictly as the fallback for a
  host whose event API is unusable.
* **Never time on the default stream.** Stream 0 implicitly serialises against
  every other stream, so a measurement taken while other GPU work is in flight
  is distorted by it. The generated bench entry creates a dedicated
  `hipStreamNonBlocking` stream and synchronises *that*.

**NVIDIA outcome: follow-up required — the same discipline is not yet applied
here, and one rule is measurably violated.**

The NVIDIA timers landed before this contract existed and use CUDA events
without an independent witness:

* `_nvidia_mma_gemm_device_latency` records events, synchronises, and divides —
  no wall-clock cross-check and no band. It has not been observed lying, but
  neither had HIP events before they were measured doing so.
* Both NVIDIA timers record on the **default stream** (`cuEventRecord(ev, 0)`
  and the launch bridge's `cuLaunchKernel(..., 0, ...)`), which implicitly
  serialises against every other stream.

CUDA has the direct analogue of the in-kernel clock (`%%globaltimer` /
`clock64()`), so the three-clock model transfers in shape. **It does not
transfer as evidence:** whether sm_120's event clock is trustworthy is an open
question here, and the gfx1151 agreement says nothing about it. Owed on
Super-Bear.

## Cross-backend sync `AUTOTUNE-RACED-FIELD-SYNC-2026-08-30`

PR (this branch) changes a **shared measurement contract**: an autotune
`MeasureRecord` must now declare which applicable candidates it did *not* race
(`unmeasured`), and `corpus_winner` refuses a verdict whose race was smaller
than the one the live registry would hold. All four backends read this corpus,
so all four are assessed here per AGENTS.md.

**The defect, measured in the committed corpus.** Every device-timed row was
missing exactly the candidates that had no `measure_device_latency`: matmul
raced 2 of 4, attention 5 of 6, fused_region 6 of 10, gated_matmul 6 of 7.
`_measure` scored an untimeable candidate `float("inf")`, so it lost silently,
and the record stored a `winner` with nothing to say the field had been
reduced. The verdicts read as "the compiled kernel is faster"; they meant "the
compiled kernel was the only one that could be timed". End-to-end rows are
unaffected — `measure_latency` just calls `run()`, so they raced the full field.

**Why it matters more than bookkeeping (sm_120, f16, device-resident):** with
all four NVIDIA matmul candidates raced for the first time, the
**compiler-emitted PTX lane wins at every shape** — 0.0095 / 0.0291 / 0.1930 /
1.4719 ms at 256/512/1024/2048³ against the hand-tuned delegate's 0.0155 /
0.0431 / 0.3202 / 2.4509 ms, i.e. **1.5–1.7× faster**. That candidate had been
excluded from every device measurement ever recorded. A biased corpus did not
merely mis-rank; it hid the fastest kernel in the registry.

**NVIDIA outcome: parity validated, on device (RTX 5070 / sm_120).** This
backend owns both the defect and the fix. `tileLaunchConfig` now carries the
block-index convention as a flag (`columnMajorGrid`, the same one
`invokeMmaGemm16` already took), so `tessera_mma_gemm_f16` no longer returns
rc=5 and the emitted lane has a device timer. The grid was verified
empirically, not by construction: at 2048×128×256 and 128×2048×256 the two
orientations time identically (0.0162 / 0.0161 ms, ~8.3 TFLOP/s), whereas a
transposed grid would put most blocks fully out of bounds and finish much
faster.

**Consequence for `NVIDIA-TIER-PRIORITY-IS-WRONG-AT-SCALE-2026-08-30`, which
understated the case.** That entry compared the delegate against the Tile lanes
only and concluded the compiled route wins at 1024³+ by 2.3–16%. With the
emitted lane in the race the margin is 1.5–1.7× at *every* shape including
256³, where the entry had the delegate winning. Read the table above, not that
one, for the ranking.

**Follow-up owned here — CLOSED 2026-09-01, and this paragraph's own
assessment was wrong twice.** See
`APPLIES-TO-SHAPE-BLIND-2026-09-01` below. What it got right: `applies_to(region)`
is shape-blind, and the emitted lane is aligned-only. What it got wrong:

* *"applicable-but-unmeasurable … every ragged device verdict is refused …
  safe and honest."* True of the **device** path only, where
  `measure_device_latency` already returned `None` for a ragged shape. On the
  **end-to-end** path `_measure` timed `run`, and `run` on a ragged shape
  returns `region.reference(...)` — so the row recorded numpy's latency under
  the kernel's name. Not unmeasurable: **mis-measured**.
* *"selection falls back to tier priority"* — stated as the safe outcome, and
  it is the unsafe one. Tier priority picks the aligned-only lane, which then
  declines to numpy while a lower-tier lane that could serve the shape goes
  untried.

Fixing it also did **not** require the shared-signature change predicted here.
An additive `applies_to_inputs(region, *inputs)` with a `True` default left all
17 existing `applies_to` implementations untouched.

## Cross-backend sync `DELTA-OPERAND-ABI-SYNC-2026-08-30`

PR #653 changes a **shared Graph IR ABI**: the delta-rule family
(`gated_deltanet`, `kimi_delta_attention`, `modified_delta_attention`) now
declares its optional tensor operands in `graph_ir._KEYWORD_OPERANDS` as
`(gate, beta, decay)` and emits `has_gate`/`has_beta`/`has_decay` presence
flags from both frontends. All four backends consume this ABI, so all four are
assessed here per AGENTS.md.

**What was wrong.** Undeclared, the AST emitter appended keyword operands
*sorted by name*, so `gated_deltanet(q, k, v, gate=g, beta=b, decay=d)` emitted
them as `(beta, decay, gate)`. Order alone would not have been enough either:
with `[Q, K, V, %x]` the lone optional sits at index 3 whichever slot it fills.

**Load-bearing fact for every backend: no producer had ever set these flags.**
`has_gate`/`has_beta`/`has_decay` were read by four executors and written by
none. The compiled ROCm, NVIDIA and x86 deltanet lanes all compute
`need = 3 + has_gate + has_beta + has_decay` and raise when that disagrees with
the operand count — so with the flags absent they accepted **only** the
three-operand form and raised on any traced call carrying `beta`/`decay`. This
PR is therefore what makes those lanes reachable with optionals at all. That is
a behaviour change on three backends and each owes its own exact-device proof;
the Apple result does not transfer to any of them.

**NVIDIA outcome: follow-up required — exact-device proof owed on sm_120.**

Two NVIDIA consumers change behaviour and neither has been run on hardware for
this PR:

* `_execute_nvidia_deltanet_compiled` now receives the presence flags it always
  parsed for, so the compiled sm_120 lane becomes reachable with
  `gate`/`beta`/`decay` for the first time.
* `native_vjp_plugins._nvidia_sm120_deltanet_backward` **stops guessing.** It
  previously derived presence from operand *variable names* and, when no name
  matched, fell back to "trailing operands fill the slots in declaration
  order" — which mis-binds whenever a caller names its locals anything but
  `beta`/`decay`. `gated_deltanet(q, k, v, beta=b, decay=d)` presents `b` and
  `d`, so the fallback fired and bound beta into the gate slot: the VJP
  differentiated a different recurrence and returned gradients that were wrong
  but finite. It now raises rather than infer.

Owed on Super-Bear: forward parity for the compiled lane with each optional
subset, and a gradient check for the backward plugin. Neither transfers from
the Apple result.

## `NVIDIA-DELEGATE-CONTRACT-2026-08-30` — the fast-path boundary is real; NVIDIA goes first

**Enabling step for the bootstrap prune, and it had to land before any
deletion.**

*Corrected 2026-08-30, by measuring rather than assuming.* This section first
said "the 19 NVIDIA bootstrap packagers contain legitimate fast paths — vendor
library entries, hand-tuned kernels, inline PTX". **They contain none.**
`nvidia_native.py` has zero references to NVRTC, cuBLAS/cuDNN/CUTLASS, any
`.so`, or raw device source; 13 of its 19 bootstrap packagers construct Tile
IR and compile it through `tessera-opt`. NVIDIA's real delegation surface is
`ptx_emit.py`, `emit/nvidia_cuda.py` and `runtime.py` — different files.

Across all four backends the same holds: **24 of 34 bootstrap packagers are
IR-constructing, 1 delegates** (`bootstrap_prune_gap.md`). So the prune is
overwhelmingly an *absorption* job — moving Graph → Schedule → Tile into the
compiled route — not a delegation-migration job. The boundary below was still
the right thing to land first, but for the delegation surface that actually
exists, not for these packagers.

NVIDIA was
chosen over ROCm because it has both the largest gap (19 of 34 bootstrap
packagers) and **working profiling tools**, which matters more than gap size:
Decision #28's arbiter is *measured*, so a delegation boundary on a target
that cannot be profiled is bookkeeping rather than a candidate.

**What `tessera_nvidia.kernel_call` was.** A summary line and nothing else.
It inherited `TesseraNVIDIA_Op`'s shared `attr-dict`, so `callee` — the single
fact naming *what is delegated to* — rode as an unvalidated discardable
attribute. An emitter could name any symbol, or none, and still verify. The
dialect header says why it existed: Python emitters "may add
`tessera_nvidia.kernel_call`", and registering it "keeps the emitted surface
parseable". It was a parse-compatibility stub **for the bootstrap packagers
being pruned** — Decision #29's anti-pattern exactly.

**Both pathways are now declared, as two ops rather than one with a mode.**

| Op | Delegate is | Required contract |
|---|---|---|
| `kernel_call` | a named CUDA kernel or host C-ABI symbol | `callee`, `arch`, `binding` ∈ {`cuda_kernel`,`c_abi`}, `provenance` ∈ {`vendor_library`,`handwritten_kernel`}, `accuracy` |
| `inline_ptx` | PTX text embedded in the artifact | `ptx`, `constraints`, `arch`, `accuracy`, optional `has_side_effects` |

They are separate ops because the delegate differs in kind: one is a binding
resolved at link/launch time, the other is text carried in the artifact. An
empty `callee` is an unresolved-symbol error; an empty `ptx` body is a
*silently successful no-op*. One op with a mode attribute would need a
verifier that decides which half of its own attributes to trust — the shape
that lets a malformed candidate through.

**The attributes are the arbiter's inputs, which is what "real" means here.**
`accuracy` is the budget half of "fastest *in-budget* candidate": a delegate
claiming `tolerance_bounded` must state `tolerance`, and `reference_exact`
must not carry one, because two contradictory claims leave a reader unable to
tell which is honoured. It is a semantic key and never defaults (#21a).
`provenance` is what lets the arbiter tell delegated from compiler-generated
work when scoring; `binding` separates two pathways whose launch costs and
failure modes differ.

*Evidence (The-Super-Bear, full driver):* `tessera-nvidia-opt` builds clean;
the positive fixture parses both ops with full attributes; the new negative
fixture `nvidia_delegate_contract_invalid.mlir` rejects **7 cases** —
empty callee, bounded-without-a-bound, exact-carrying-a-tolerance,
non-positive tolerance, unknown `binding`, empty constraints, empty ptx.
NVIDIA lit suite **60/60**.

*Arbiter integration landed:* `DelegatedCandidate`
(`emit/delegate_contract.py`) derives tier from `provenance` and the F4 budget
from `accuracy`/`tolerance`/`tolerance_rel`, so a delegate cannot claim in
Python a budget it did not declare in IR. The ROCm equivalent is still owed.

### Gaps found by stress-testing this design (2026-08-30)

Two were live defects in the contract as first shipped and are **fixed**:

* **Determinism was undeclarable.** Tessera guarantees
  `@jit(deterministic=True)`, and a split-K delegate accumulating with atomics
  is not reproducible run to run — the arbiter could have selected one inside
  a deterministic region. `determinism` is now a required enum. Same shape as
  the Decision #5 scar: a guarantee defeated through a path nobody checked.
* **The accuracy claim was absolute-only** while `Candidate` already carried
  both atol *and* rtol. An absolute bound is meaningless without the result's
  magnitude — 1e-6 is vacuous at 1e6 and unsatisfiable at 1e-9 — so a delegate
  whose real claim is relative had to overclaim. `tolerance_rel` added;
  either or both now satisfy a bounded claim.

Open, ordered by whether the design is *wrong* versus merely incomplete:

1. **Per-op accuracy budgets do not compose (mathematical, unsound as stated).**
   Five delegates each within 1e-3 do not give an end-to-end result within
   1e-3; propagation depends on conditioning. A graph can be assembled
   entirely from in-budget candidates and land outside any budget with nothing
   detecting it. Needs a graph-level check, or the composition claim must be
   withdrawn.
2. ~~Fusion foreclosure is not costed.~~ **Closed 2026-08-30 — and the bias
   was worse than first described.** `arbitrate()` picks by **tier** by
   default, and `Tier.HAND_TUNED` is the highest, so a delegate won
   *outright, before anything was measured*; on the measured path it won
   because the latency excluded the work it displaced. Both paths preferred
   delegates on exactly the graphs where fusion is the win.

   Fixed structurally rather than with a penalty. A delegate now declares
   `covers` (`root_only` | `whole_region`), and `DelegatedCandidate.applies_to`
   **declines a region it implements only part of**. A penalty would have been
   a guess at foregone DRAM traffic that then had to outweigh a tier bonus;
   "this candidate does not serve this region" is a fact the delegate
   declared. If the delegate-plus-separate-epilogue plan really is faster,
   that is a comparison of *plans* and does not belong as a peer candidate.
   Whole-region hand-tuned kernels still compete, so the #28 governing rule
   (never cap the leads) is preserved.
3. **`kernel_call` does not verify operands against the callee ABI.** The op
   requires `constraints` for inline PTX on the argument that unstated
   constraints become silent miscompiles — and then leaves the symbol path
   unchecked. Inconsistent; `tessera_x86.abi_call` has the same hole.
4. **No delegate versioning.** cuBLAS 12 and 13 differ numerically and in
   performance; with no version or ABI hash, a cached measurement from one
   applies to the other. That is the stale-baseline failure this queue already
   recorded once for the Krylov ratchet.
5. **Accuracy uses a vocabulary parallel to `numeric_policy`.** Decision #15a
   puts accumulator contracts there; `reference_exact` cannot even be honoured
   for float reductions, where result depends on accumulation order. This is a
   Decision #32 information-loss issue inside the op meant to prevent them.
6. **`has_side_effects` is one bit.** Reads, writes and barriers have
   different legality; one bit forces treating any side-effecting asm as a
   full barrier, which costs real performance. MLIR has `MemoryEffects`.
7. **Shape-bucket boundaries are undefined.** The sm_120 macro-CTA threshold
   is one number (67,108,864 FLOPs) measured once under WSL, and this file
   already says it is not global selector authority. Coarse buckets apply an
   M=4096 measurement at M=17.
8. **No measurement statistic or hysteresis.** "Fastest" by mean, median or
   min, over how many reps? Without a minimum effect size the arbiter thrashes
   inside noise.
9. **The arbiter is on the wrong side of the prune (architectural).** It lives
   in Python. Pruning the Python backend path while keeping a Python arbiter
   keeps the seam. Probable resolution: arbitration is legitimately *outside*
   the IR pipeline because it requires execution, like PGO — but then the
   contract must be stated: IR declares candidates, an orchestrator measures
   and selects, selection is recorded back as an attribute.
10. **`tessera_rocm.mfma` vs `rocdl.mfma`** — measure whether the Target IR op
    carries a contract ROCDL cannot, per Decision #19's amended membership
    test. If it mirrors, it is Decision #31 duplication.

**Sequencing rule for the Apple/x86 operator expansion.** Add each op only
when its producer and its consumer land with it. Apple needs ~12 ops and x86
~8; landing the families ahead of the passes that produce them manufactures
exactly the unconsumed-declaration anti-pattern (Decision #29) this contract
work exists to remove. One op proven end-to-end beats twelve declared.

**And when x86's `avx512_gemm_microkernel` is decomposed into primitives, the
microkernel must survive as a Tier-3 candidate rather than being replaced.**
If it is hand-scheduled, decomposing it and hoping LLVM re-schedules is the
"generic IR caps the ceiling" trap applied within x86 — the arbiter should
decide, not the refactor.

Cross-backend sync `AVX512-MARKER-AND-AMX-CONSUMER-2026-08-30` — **shared
marker vocabulary and conftest boundary changed; per-backend outcome below.**
`hardware_avx512` joins `policy.MARKERS`, the PR marker expression and its
four verbatim copies, `pyproject.toml`, and the device-accounting families.
`conftest` now consumes `hardware_avx512` and `hardware_amx` centrally,
matching the existing `hardware_nvidia` / `hardware_apple_gpu` boundaries.

*NVIDIA outcome: parity validated — no behaviour change, and the reason
matters.* The `hardware_nvidia` arm of `pytest_runtest_setup` is checked
**before** the two new arms and returns early, so no NVIDIA lane can be
diverted by them; no test under `tests/device/nvidia/` carries
`hardware_avx512` or `hardware_amx`. The new PR-expression term
(`not hardware_avx512`) deselects nothing here for the same reason.

One fleet fact worth recording, because it is the opposite of the intuition:
**The-Super-Bear has no AVX-512.** Its Threadripper 3970X (Zen 2) reports no
`avx512f`, so despite building the x86 backend (`TESSERA_BUILD_X86_BACKEND=ON`)
that box probes `avx512=False` and its x86 lanes skip honestly. Princess-Luna
(Zen 5) is the only AVX-512 host in the fleet, which is also why
`hardware_amx` must never be used to mean "x86 hardware" — see the standing
section in `docs/audit/backend/x86/todo.md`.

## Cross-backend sync `DELEGATE-CONTRACT-SYNC-2026-08-30`

PR #652 changed two **shared** runtime contracts, so all four backends are
assessed here per AGENTS.md:

1. `Candidate.accuracy_budget(region)` — a new hook on the shared arbiter
   base class. `candidate._as_runner()` now resolves the F4 oracle's budget
   through it instead of reading `accuracy_atol` off the class.
2. `DelegatedCandidate` gained a `name` override and a per-dtype contract
   *family* (`variants`), so one delegate may bind a different callee per
   storage dtype and still derive tier and budget from declared IR.

**Measured blast radius (37 registered candidates: nvidia 32, rocm 3, x86 2,
apple 0): exactly one overrides `accuracy_budget`** — `nvidia_mma_gemm_shipped`.
Every other candidate inherits the base implementation, which returns
`(self.accuracy_atol, self.accuracy_rtol)`: the same two values the arbiter
previously read directly, at the same call site. That equivalence is static,
not a measurement, so no sibling backend owes a device re-proof for change (1).

**NVIDIA outcome: parity validated, on device.** The delegate is this PR's
subject. `tests/device/nvidia/test_shipped_gemm_delegate.py` — 14 passed on
sm_120 (RTX 5070): declared contract, both dtype callees executing, the
declared budget holding across K=32..4096, and device-resident latency for the
delegate and both compiled Tile lanes.

**Follow-up owned here:** `nvidia_mma_gemm_emitted` still has no device timer
(two block-index conventions, see
`NVIDIA-TIER-PRIORITY-IS-WRONG-AT-SCALE-2026-08-30`), and shape-bucketed
measured selection is not yet wired into the `OP_MATMUL` path.

## `NVIDIA-TIER-PRIORITY-IS-WRONG-AT-SCALE-2026-08-30` — measured, not argued

**The first thing the delegate's device timer produced, and it contradicts the
arbiter's default.** Decision #28 displaces a hand-tuned kernel when a compiled
one measures **faster and in accuracy budget**. On sm_120 (RTX 5070), f16,
square, device-resident CUDA-event timing, spreads of 0.000–0.008 ms across
repeats:

| shape | `nvidia_mma_gemm_shipped` (T3) | `nvidia_tile_matmul_shared` (T2) | faster | max\|err\| |
|---|---|---|---|---|
| 512³ | **0.043 ms** | 0.059 ms | delegate, by 37% | both 2.48e-05 |
| 1024³ | 0.320 ms | **0.312 ms** | compiled, by 2.3% | both 6.10e-05 |
| 2048³ | 2.448 ms | **2.051 ms** | compiled, by 16.2% | both 1.54e-04 |

The error columns are **equal at every shape**, so the in-budget half is
satisfied outright. The displacement condition therefore holds at 1024³ and
above — and `arbitrate()` still returns the delegate, because tier priority is
the default and D2's measured loop is not wired into this path.

Two things follow, and neither was visible before:

* **The compiled Tessera kernel beats the hand-tuned one at scale.** That is a
  result about the compiler, not about the arbiter.
* **The crossover is shape-dependent**, which is the concrete argument for
  shape-bucketed measured selection rather than a single global winner. A
  flat "measurement beats tier" switch would regress 512³ by 37%.

**Do not read this as "delete the delegate."** It wins by 37% at 512³, and
Decision #28's lead-safety exists precisely so a crown-jewel lane is displaced
per shape by evidence rather than wholesale by policy.

**Why it was invisible until now.** End-to-end wall time ranks the two the
*other* way — 9.4 ms vs 33.1 ms at 2048³ — because it is host-dominated: the
Tile lane spends 2.99 ms on device inside 34.0 ms of wall time, and the two
lanes do not share a host path, so e2e compares numpy conversions. The Tier-3
lane had no device timer at all, so the honest comparison could not be made.
Pinned by `tests/device/nvidia/test_shipped_gemm_delegate.py`.

**Open follow-ups.**
1. Wire shape-bucketed measured selection into the `OP_MATMUL` NVIDIA path so
   the 1024³+ crossover is acted on. The `measure` hook and the autotune
   corpus already exist; nothing calls them for this bucket.
2. `nvidia_mma_gemm_emitted` still has no device timer. The NVIDIA backend
   carries **two block-index conventions**: `ptx_emit` and the shipped AOT
   kernel map x→M, y→N, while `NVIDIALowering.cpp` and the launch bridge's
   `benchmarkTileGemm16` map x→N, y→M. Driving the emitted kernel through the
   harness returns rc=5, and registering its geometry would launch a
   transposed grid (at 512×512: rows to 1024, columns only to 256 — half the
   output unwritten, with a plausible-looking latency). Unify the convention,
   or give the harness an explicit axis-order field. A unit test pins the
   current mapping so "fixing" one side fails loudly.

## `SM120-BUILD-CONFIG-RESOLVED-2026-08-30` — there was no trade; use CUDA=ON

**Superseded the "fleet-config decision" framing below: configuring the NVIDIA
backend on Super-Bear costs nothing, because that box has CUDA.** The lean
driver that carves out core/x86/Apple registration is gated on

```cmake
(TESSERA_BUILD_NVIDIA_BACKEND AND NOT TESSERA_ENABLE_CUDA)
```

— it only fires for a **CUDA-less** NVIDIA build. `tools/tessera-opt/CMakeLists.txt`
says so directly: "A backend built against a real toolchain (CUDA or HIP) IS a
full build and gets the core dialect." Measured: configuring
`build-nvidia-cuda/` with `-DTESSERA_ENABLE_CUDA=ON
-DTESSERA_BUILD_NVIDIA_BACKEND=ON -DTESSERA_BUILD_X86_BACKEND=ON` yields
`tessera-opt --tessera-build-info` → **`build profile: full`, features
`… core-tessera-ir … nvidia-backend … x86-target-ir`** — all three at once.
The existing `build/` was left untouched; point tests at the new tree with
`TESSERA_BUILD_DIR=build-nvidia-cuda`.

## `SM120-STAGING-ROUTING-DIAGNOSED-2026-08-30` — the filed description was wrong on every count

With the full driver the four tests get far enough to say what is actually
happening, and **it is not "4 stale shared-staging assertions".** They fail at
**three different assertions**, and only one of them is about staging:

| storage | shape | `tile.matmul_kernel` in Tile IR | `ab_stage` in Target IR | PTX entry | fails at |
|---|---|---|---|---|---|
| f16 | (16,8,16) | ✗ | ✗ | `nvidia_sm120_scheduled_matmul_…_kernel` | L495 `tile.matmul_kernel` |
| f16 | (37,29,23) | ✓ | ✗ | `nvidia_sm120_scheduled_matmul_…_kernel` | L500 entry name |
| bf16 | (16,8,16) | ✗ | ✗ | `nvidia_sm120_scheduled_matmul_…_kernel` | L560 `tile.matmul_kernel` |
| bf16 | (37,29,23) | ✓ | ✗ | `nvidia_sm120_scheduled_matmul_…_kernel` | L564 `ab_stage_bf16` |

**Root cause: `nvidia_schedule="shared"` is silently dropped on the
scheduled-matmul route.** All four requests take
`package_scheduled_matmul`, whose entry is named by
`_SM120_SCHEDULED_MATMUL_PREFIX`; `package_native`'s `kind` dispatch forwards
`nvidia_schedule` **only** on the fall-through `package_matmul` branch. The
tests request `shared` and then assert `package_matmul`'s artifacts (entry
`tessera_tile_matmul_shared_*`, the `__tessera_sm120_ab_stage_*` global,
`tile.matmul_kernel`), so they are describing a route the compiler no longer
sends them down.

Two things this settles, and one it does not:

* The earlier source-read holds: `emit_matmul_tile_ir(schedule="shared")` does
  emit `warps = 4 : i64, staging = "shared"`, and `NVIDIALowering.cpp:1228`
  consumes it with no shape guard and no silent fallback. The Python and C++
  halves of the *shared* route are fine. Nothing is quietly re-routing small
  shapes away from staging.
* Within the scheduled route there **is** a shape split — (16,8,16) has no
  `tile.matmul_kernel` at all while (37,29,23) does — matching the
  `sm120_scheduled_typed_16x8_mn` vs `macro_cta_32x32_mn` policy selection.
* **Resolved 2026-08-30 (project direction): the compiled route is
  authoritative and `nvidia_schedule` does NOT select it.** The Tessera
  foundation is core MLIR/LLVM → Tile IR → codegen; hand-written NVIDIA/CUDA
  kernels are not what should fall out of a compile. `driver.py:526` already
  encodes this — the scheduled route is taken whenever `tessera-opt` is
  available, and `package_matmul` is the fallback for when it is not. So
  `nvidia_schedule` steers only that fallback.

  What was wrong was the silence, and that is fixed. `driver.py` now emits
  **`SCHEDULE_KEY_NOT_HONORED_ON_COMPILED_ROUTE`** (registered in
  `diagnostic_codes.py`, severity `warning`) when a fallback-only key is
  supplied on the compiled route. `"auto"` is deliberately not reported: it
  means "you choose", which is what the compiled route does, and warning on it
  would train people to ignore the diagnostic.

  The four tests now assert the **contract** rather than a spelling — entry
  carries the `nvidia_sm120_scheduled_matmul_` prefix, the launch policy is one
  of the two scheduled policies, and the k contract goes through the existing
  producer-aware `_assert_canonical_k_loop` rather than pinning
  `canonical_k_loop` (which the typed-16x8 route legitimately does not emit).
  Two new tests cover the diagnostic itself, in both directions. Measured on
  sm_120 with the full driver: `test_e2e_spine_native.py` **304 passed, 0
  failed** (was 4 failed at three different assertions).

  The diagnostic paid for itself immediately: it surfaced 16 warnings per run
  from `test_canonical_sm120_k_loop_shape_matrix`, which was still supplying
  `nvidia_schedule="shared"` on the compiled route. That key is now dropped
  there. One site remains by design —
  `benchmarks/e2e_spine/record_sm120_packet.py:256` passes `"direct"`, which is
  inert on any host with `tessera-opt` but **does** change the fallback's
  choice on a host without it (`auto` resolves to `shared` for fp16/bf16), so
  removing it would be a silent behaviour change on that path. Left alone; the
  diagnostic will tell whoever runs it.

* **Follow-on, not done here: Decision #31 on this boundary.** Two packagers
  now serve one IR-level boundary — the compiled scheduled route and the
  templated `package_matmul`. Decision #31 allows exactly one production
  lowering per boundary; a second must be either a **declared oracle with a
  differential test** or deleted. Given the direction above, the fallback is
  the one to declare or retire. Decision #31's own ordering caveat applies —
  do not collapse it before the compiled route demonstrably carries what it
  carried — so this wants a scoped plan with a coverage comparison
  (which `native_package_kind` families reach which packager, and what happens
  on a host with no `tessera-opt`), not a drive-by deletion.

## `SM120-BASELINE-IS-BUILD-DEPENDENT-2026-08-30` — read before trusting any sm_120 suite count

**The Super-Bear device-suite baseline of "5 failed / 844 passed" is only
valid for a build with `TESSERA_BUILD_NVIDIA_BACKEND=ON`. That box is
currently configured OFF, and the identical commit then reports 34 failed /
815 passed.** Measured 2026-08-30, both numbers on `main`, same box, same
GPU, differing only in that cmake flag.

The mechanism is not subtle once seen: without the NVIDIA backend the
`tessera-opt` in `build/` never registers the NVIDIA Target IR dialect, so
every scheduled-matmul lane dies at the `--tessera-schedule-to-tile`
boundary with `SM120 scheduled matmul requires the registered NVIDIA Target
IR dialect`. Thirteen tests in `test_scheduled_matmul_consumers.py` fail in
2.5 s having touched no GPU at all. Nothing in the suite output says "your
build cannot evaluate these lanes" — it reads as a code regression, and a
control run on `main` is the only thing that tells the two apart.

**Three configurations, and do not collapse them** (a README edit did, and
review caught it). `TESSERA_BUILD_NVIDIA_BACKEND=ON` **always** builds the
hardware-free Target IR spine — `src/compiler/codegen/tessera_gpu_backend_NVIDIA/CMakeLists.txt`
says so in its header, and `tools/tessera-opt/CMakeLists.txt` links
`TesseraNVIDIAIR`/`TesseraNVIDIAConversion` under `if(TARGET
TesseraNVIDIAConversion)`, which is not gated on leanness.

| Config | NVIDIA Target IR | Core spine | Scheduled lanes |
|---|---|---|---|
| backend ON + `ENABLE_CUDA=ON` | registered | linked + registered | run |
| backend ON + CUDA off (**lean**) | **registered** | linked, **not registered** | unavailable — missing core *registration*, not the dialect |
| backend OFF (what this box has) | not built | linked + registered | fail with `requires the registered NVIDIA Target IR dialect` |

The middle row is the **supported host-free artifact configuration** that
Decision #19's hardware-free Target IR exists to enable; `_tessera_opt_lean_permitted`
lists `nvidia-backend` explicitly. Saying a CUDA-less NVIDIA build "never
registers the dialect" erases it and contradicts that contract — the symptom
above belongs to the third row only.

**Consequence for the open staging item.** The four
`__tessera_sm120_ab_stage_bf16` assertions were recorded as "stale
shared-staging assertions — a routing question". **That characterisation
does not currently reproduce and should not be acted on until it is
re-measured.** On this box the four tests
(`test_canonical_sm120_{bf16,request}_*`, two shapes each) never reach the
staging assertion: they fail 1.4 s in, at the same missing-dialect boundary.
Whatever was seen last session was seen on a differently configured build.

What *is* established about that item, from source rather than from a run:
the Python side is correct end to end. `emit_matmul_tile_ir(schedule="shared")`
emits `warps = 4 : i64, staging = "shared"` into the Tile IR, and
`NVIDIALowering.cpp:1228` reads that attribute with **no shape guard and no
silent fallback** — if `staging == "shared"` arrives, the buffer is
materialised or the pass hard-errors. So the routing question, if one
survives re-measurement, is about which op the attribute reaches, not about
small shapes being quietly re-routed. The scheduled path is the suspect:
`package_native`'s `kind` dispatch forwards `nvidia_schedule` only on the
fall-through `package_matmul` branch, and `package_scheduled_matmul` selects
`sm120_scheduled_macro_cta_32x32_mn` vs `sm120_scheduled_typed_16x8_mn` from
the entry name without consulting the requested schedule. If a caller's
`nvidia_schedule="shared"` is being dropped there, Decision #21a applies —
a performance key may fall back, but not silently.

**Corrected 2026-08-30 — see `SM120-BUILD-CONFIG-RESOLVED` above.** This
section originally said turning the NVIDIA backend on would carve out
core/x86/Apple registration, making it a fleet-configuration trade. That is
true only for a **CUDA-less** NVIDIA build; Super-Bear has CUDA, so
`-DTESSERA_ENABLE_CUDA=ON -DTESSERA_BUILD_NVIDIA_BACKEND=ON` gives a full
driver with all three registered and there is no trade to make. The
build-dependence of the suite count, which is the point of this section,
still stands.

Cross-backend sync `HOLLOW-GREEN-GATES-2026-08-30` — **shared test infra
changed; per-backend outcome below.**
A pytest session ledger (`tests/_support/device_accounting.py`) now tallies
executed-vs-skipped per hardware family and **fails the session** when a
family skipped everything on a host that plausibly has the device. It exists
because `pytest tests/device/nvidia/` on this box once reported 454 passed /
395 skipped / exit 0 while running zero GPU work.

*NVIDIA outcome: parity validated, with follow-up.* Eight files under
`tests/device/nvidia/` carried no hardware marker
(`test_{fft_workspace,optimizer_reverse,philox_jvp,plugin,rng_compiled,spectral_autodiff,spectral_jvp,spectral_policy}.py`);
they were invisible to both the PR-lane deselection and the new ledger, and
they *failed* rather than skipped on a non-CUDA host. All eight are marked,
and `conftest` now consumes `hardware_nvidia` centrally, matching the
existing `hardware_apple_gpu` boundary — verified on the Mac, where those
four optimizer tests changed from hard failures to honest skips. Follow-up:
the device suite count above cannot be re-baselined until the build-config
question in this section is settled.

Cross-backend sync `ADAFACTOR-BIAS-CORRECTION-2026-08-30` — **shared numerical
policy changed; per-backend outcome below.**
`optim.adafactor_decay` makes the Adafactor second-moment decay step-dependent
(`b2_t = b2*(1 - b2^(t-1))/(1 - b2^t)`), removing an early-step update
inflation of 1/sqrt(1 - b2^t) — 31.6x at step 1, 10.0x at step 10, 1.26x at
step 1000 for the default beta2. The correction is applied HOST-SIDE as a
scalar decay, so **no kernel ABI moves**: every physical kernel already takes
`beta2` as a scalar and receives the effective value instead of the nominal
one. The flat op gained an optional `step` kwarg matching the `adam`/`adamw`
ABI beside it.

Two contract details a backend owner needs to know. `state["v"]` now carries
the DEBIASED estimate rather than the raw EMA, so the state dict grew a
`v_representation` marker and a state without one is migrated on load rather
than misread. And an absent `step` is NOT treated as step 1 — `decay(b2, 1)`
is exactly 0, so defaulting would have made a stateful caller that never
passes one discard its own moments; such a caller keeps the legacy
uncorrected decay.

*NVIDIA outcome: parity validated 2026-08-30 (was follow-up required).*
`sm120_adafactor_*` receives the effective decay through the existing scalar;
`tests/device/nvidia/test_optimizer_reverse.py` was migrated to pass `step`.
**The owed exact-device run is done.** On The-Super-Bear (RTX 5070, sm_120,
CUDA 13.3, `scripts/_nvidia_env.sh` sourced) that file is **4 passed, 0
skipped**, including
`test_sm120_adafactor_full_and_factored_exact_certificates`, which compares
the device reverse package against `get_vjp("adafactor")` for both the full
and factored topologies. Zero skips is the load-bearing half of that
sentence: the same file reported a clean *skip* on this host for as long as
the driver shim was off `PATH`.

Cross-backend sync `P3-DEVICE-VERIFIED-2026-08-30` — **the two NVIDIA rows
owed by `P3-SOURCE-ONLY` are now measured, and one of them was a regression.**

* `emit/nvidia_solver_krylov.py` `tsr_matvec` — the warp-per-row rewrite was
  shipped on a reasoned access-pattern claim. Measured on an RTX 5070
  (sm_120), medians of 9 reps, device_event: **dense_cg 0.44-0.63x (a
  REGRESSION of up to 2.3x)** and **dense_gmres 1.22-1.56x (a win)**. The
  coalescing argument was correct and still lost, because a COOPERATIVE
  launch caps the grid at what stays resident, so warp-per-row also buys 32x
  fewer rows in flight. The solvers no longer share one matvec:
  `tsr_matvec_scalar` for CG, `tsr_matvec_warp` for GMRES, with the table in
  the source. Re-measured after the split: CG back to 1.00-1.06x of scalar,
  GMRES keeps 1.24-1.56x. `benchmarks/baselines/nvidia_sm120_solver_krylov_performance.json`
  was recorded with the OLD matvec and **passed throughout the regression** —
  re-recorded at 15 reps / 5 warmup, and the ratchet now measures reality.
* `emit/nvidia_cuda.py` flash-backward cleanup — the 20 Krylov/solver device
  tests and the flash-backward route tests pass on sm_120. An induced
  allocation failure is still not exercised; that remains the honest gap.

Also closed here: the `rc=5` invoke failure (the runtime dispatches scheduled
sm_120 matmuls by NAME PREFIX while the compiler named the kernel after the
caller's Graph function) and the sm_120 packager reading matmul epilogue
edges from `op.kwargs` when the verifier requires operands. Device suite:
**81 failed -> 5 failed / 844 passed.** The 5 remaining are 4 stale
shared-staging assertions (`__tessera_sm120_ab_stage_bf16`, pre-existing and
a routing question, not a test-editing one) and NCCL not being installed.

Cross-backend sync `P3-SOURCE-ONLY-2026-08-30` — **two rows are fixed in
source and have never run on a GPU; they are this queue's to close.**
The P3 batch changed two NVIDIA emitters with no CUDA host available:

* `emit/nvidia_cuda.py` flash-backward — `TSR_ATOMIC_ENTRY` and the f16
  wrapper now free through a `goto fail` block, and the atomic entry CHECKS
  its H2D copies, which it previously fired unchecked (a failed upload
  yielded a confidently wrong gradient). Needs a real run, and ideally an
  induced allocation failure confirming the cleanup frees exactly what was
  allocated.
* `emit/nvidia_solver_krylov.py` `tsr_matvec` — now a warp per row with
  lane-strided columns and a `__shfl_down_sync` butterfly. The claim made is
  structural only: at a fixed inner iteration a load's 32 lanes touch 32
  consecutive elements of one row rather than 32 rows `n*sizeof(T)` apart,
  so transactions per load drop from 32 to 4 for f32. **No speedup was
  claimed and none is known.** Two consequences to check on device: the
  per-row summation order changed, so CG/GMRES convergence needs
  re-confirming and results are no longer bit-identical to a sequential sum;
  and `tests/performance/nvidia/test_solver_krylov_ratchet.py` compares
  against `benchmarks/baselines/nvidia_sm120_solver_krylov_performance.json`,
  recorded with the OLD matvec — that baseline will report a false result
  until re-recorded. Whether to enlarge the launch geometry now that a warp
  owns a row (`useful = ceil(n/256)` was sized for one thread per row) is an
  open measured question, deliberately left alone.

Evidence that does exist: the generated text was asserted on, and both
sources parse clean under `clang++ -std=c++17 -fsyntax-only -Wall` with CUDA
stubs — a harness confirmed to reject a `goto`-crosses-initialization, so the
clean parse means something. It is not device evidence.

Cross-backend sync `P2-REVIEW-SHARED-PASSES-2026-08-29` — **15 shared MLIR
passes changed; only the Mac's fixture set could be run.**
The P2 code-review batch touched passes every backend lowers through:
`TesseraToLinalgPass` (rejection checks moved before IR creation),
`SymbolicDimEqualityPass` (transposeA/B in the contract + flow rules, and
malformed `dim_bindings`/`dim_sizes` now fail closed),
`AdjointCollectiveInsertionPass` (cotangent-array bounds),
`AutodiffPairedPass` (dynamic while state refused; erase re-checks use_empty),
`RegionAdjointInterface` (O(1) dense-checkpoint slot),
`ActivationRematerializationPass` (difference-array peak),
`WarpSpecLegalityPass` (transitive staged-data provenance),
`TileBufferArenaPass` (non-scalar element types),
`IRContractLegalityPass` (narrowing-accum restricted to same-domain pairs),
`MaterializeControlPayloadPass` (shared body-stub conflict),
`InsertRecomputePass` (real live-set), `LegalizeSpaceTime` + the CPU stencil
hook (orders 6 and 8 implemented; unimplemented orders refused), and
`AsyncPrefetch` (memory-write dependence).
Evidence produced: `lit tests/tessera-ir/` **437 discovered, 396 passed, 41
unsupported, 0 failed** on the Mac (M1 Max, brew LLVM/MLIR 23.1.0, assertions
OFF), plus per-finding reproductions with controls. **Not evidence for this
backend's own fixtures.**

*What this queue must run on sm_120.* The split-route flash backward gained a
`tsr_flash_bwd_stats` kernel so per-`(b,qh,m)` softmax statistics are computed
once instead of once per KV split; `tsr_flash_bwd_dq` was restructured n-outer
into a `aq[D]` accumulator. Host-executed through a launch-emulation shim the
split route now matches the untouched atomic route to <=1.9e-7 across causal /
sliding-window / bias / logit-cap, with a negative control at 2.8e-2 — but no
line of it has run on a GPU. **Re-measure
`measure_flash_attention_backward_device`**: the route arbiter was choosing
between these two routes using the inflated split number. Also owed: the
decayed `run_linear_attention_variant{,_backward}` F4 tolerance under the new
Horner recurrence (forward folds decay into the accumulator, keeping the
summation ascending; backward uses a descending running product because the
factor feeds per-key atomics), and one `run_optimizer_f32` call per valid kind
to confirm the new `kind<0||kind>5` rc-2 guard left the happy path alone.

Cross-backend sync `LINUX-BASELINE-2604-LLVM231-2026-08-29` — **not applicable to SM120; the CUDA host is already on 26.04.**
The Linux baseline moves to **Ubuntu 26.04 LTS** and the compiler-backbone pin
tightens from "LLVM/MLIR 23.x" to **23.1.x exactly**; `scripts/setup_ubuntu.sh`
now FAILS on any other Ubuntu release rather than warning. `CLAUDE.md`'s host
record moved in the same change, because leaving it at 24.04 pointed this
project's own instructions at a bootstrap command that exits immediately.

Measured on the migrated box (`Princess-Luna`): Ubuntu 26.04.1, LLVM/MLIR
23.1.0 (assertions OFF), ROCm 10 series (HIP 7.15), repo at
`~/programming/tessera`, ssh on the default port.
*NVIDIA outcome.* No NVIDIA impact. The CUDA host (The-Super-Bear) was
already Ubuntu 26.04 with LLVM/MLIR 23.1.0, so the tightened pin and the new
`setup_ubuntu.sh` gate match it as-is; the branch built and ran there with
`TESSERA_ENABLE_CUDA=ON` (unit failure set identical to main, `lit` 429/429).
Unrelated pre-existing snag worth carrying: `TESSERA_ENABLE_CUDA=ON` fails to
configure because `examples/advanced/power_retention/src/extension` has no
CMakeLists.txt — use `-DTESSERA_BUILD_EXAMPLES=OFF` until that is repaired.
Cross-backend sync `SM120-REGRESSION-VALIDATION-2026-08-29` — **branch
validated on the RTX 5070; no regressions, and no P0 was ever owed here.**

Built and run on The-Super-Bear (RTX 5070 cc 12.0 / sm_120, Ubuntu 26.04,
CUDA 13.3, LLVM/MLIR 23.1.0 assertions OFF) with `TESSERA_ENABLE_CUDA=ON`,
from clean worktrees at the branch and at `f65f9b3b`:

* unit suite **13503 passed / 53 failed / 5414 skipped**, and the failure set is
  **byte-identical to main's on the same box (53 = 53)** — no regressions, none
  fixed. `lit tests/tessera-ir/` **429/429**.
* Confirm before reading the numbers: NO P0 from the 2026-08-29 review touches
  NVIDIA code. The CUDA item is a P1 (`nvidia_cuda.py:1137` shared reduction
  scratch reused without a barrier).
* That P1 remains **only partially proven**. The fixed pattern is correct and
  deterministic over 300 runs on sm_120, but the race was NOT reproduced, and
  `compute-sanitizer --tool racecheck` **cannot initialize under WSL2**
  ("Failed to initialize WDDM debugger interface") — its summary reports its own
  init failure identically for both variants, so it is not evidence either way.
  Nsight Compute (`ncu`) and Nsight Systems (`nsys`) ARE installed here and are
  the obvious next instrument.
* Build note, pre-existing on main: `TESSERA_ENABLE_CUDA=ON` fails to configure
  because `examples/advanced/power_retention/src/extension` has no
  CMakeLists.txt. Work around with `-DTESSERA_BUILD_EXAMPLES=OFF`; the example
  tree needs repair independently.

Cross-backend sync `SHARED-CONTRACTS-P1-REVIEW-2026-08-29` — **assessed; SM120 execution unchanged, one device proof attempted.**
PR #638 changes four SHARED contracts, so each backend records its own
outcome rather than letting the queues drift:
1. **Float `ne` is now UNORDERED** (`TesseraToLinalgPass`). `arith.cmpf one`
   is false when either operand is NaN; IEEE-754 and numpy define `!=` as the
   negation of `==`, so `NaN != NaN` is true and `x != x` — the idiomatic NaN
   test — silently never fired. eq/lt/le/gt/ge stay ordered.
2. **Control-flow predicate forms** (`LowerControlFlowToSCFPass`): boolean and
   signless-integer conditions lower instead of crashing; explicitly
   signed/unsigned integer predicates are refused, because `arith.cmpi`
   requires signless operands and cannot express them at all.
3. **Symbolic-dim while results** (`SymbolicDimEqualityPass`) are seeded from
   the condition's forwarded values, not the init/yield position.
4. **Recompute purity is derived** (`InsertRecomputePass`): an op with no
   effect attribute must be provably memory-effect-free, so an RNG draw or an
   opaque call is no longer marked recomputable.
*NVIDIA outcome.* All four are host-free contract changes; no PTX, cubin,
SM120 selector, or numeric policy moves. The emitted-CUDA barrier fix that
rides in this PR (`nvidia_cuda.py`: `tsr_sum` returning `s[0]` with no barrier
before the next call's `s[t]=v`, and the softmax max broadcast) WAS exercised
on the RTX 5070 (cc 12.0) this cycle: the fixed pattern computes correctly and
deterministically over 300 runs. **The race itself was NOT reproduced** —
`compute-sanitizer --tool racecheck` cannot initialize under WSL2 ("Failed to
initialize WDDM debugger interface"), so its output is not evidence, and 300
direct runs of the pre-fix pattern produced identical results. The defect
stands as a CUDA memory-model argument (an unsynchronized read of `s[0]`
against a write of `s[t]`), not a measured failure. A racecheck-capable host
would settle it.

Cross-backend sync `FOUNDATION-LLVM231-REVIEW-P0-2026-08-29` — **no NVIDIA P0
in this batch; foundation actions apply, and four sm_120-gated lanes plus one
confirmed CUDA-emitter P1 are owed by this box.**

*Foundation (all backends).* The LLVM/MLIR major pin is unchanged at **23**;
the Mac moved from a manual pre-release `23.1.0git` prefix to Homebrew's
production `llvm` keg **23.1.0** (old prefix deleted). This box stays on
apt.llvm.org `/usr/lib/llvm-23`. **No fleet box has an assertions-enabled LLVM
any more**, so an MLIR promise/contract claim can currently be falsified
nowhere (Decision #19) — relevant before recording any "no longer reproduces"
here. Python/deps: the `numpy<2.2` cap **can be dropped** — `pyproject.toml`
now skips numpy/scipy stubs (`follow_imports = "skip"` **plus**
`follow_imports_for_stubs = true`), keeping `python_version = "3.10"` while
making the mypy ratchet independent of the installed numpy (the fleet spans
three versions). Also: `check-{clifford,ebm,spectral}` no longer hardcode
`llvm-lit`, and driver discovery in `tests/_support/compiler_tool.py`,
`compiler/driver.py` and `_jit_boundary.py` now requires a candidate binary to
**start**, not merely exist.

*Actions on this box.*
1. Sweep for build trees pinned to a removed toolchain:
   `grep -l llvm-23.1.0-rc1 build*/CMakeCache.txt`, then `ldd` the drivers.
   On the Mac, four stranded trees produced ~86 unit failures whose messages
   were all about a missing dylib rather than about any code under test.
2. This host is x86_64, so the new `CMAKE_SYSTEM_PROCESSOR` gate in
   `src/CMakeLists.txt` must still select the native x86 kernels — confirm
   `cmake` prints no "x86 native kernels skipped" line (see the x86 todo).
3. Run the four sm_120-gated lanes that cannot be evaluated on the Mac and
   currently fail there as hardware-absent:
   `tests/unit/test_scheduled_matmul_consumers.py` (3) and
   `tests/unit/test_nvidia_compiler_artifacts.py::test_sm120_tile_fragment_lowers_to_real_nvvm_mma`.

*The confirmed CUDA-emitter finding this box owns (P1, unfixed).*
`python/tessera/compiler/emit/nvidia_cuda.py:1137` — several emitted
block-reduction kernels read the reduction result from `scratch[0]` and then
rewrite the same shared buffer for the next phase **with no
`__syncthreads()` between**, a data race under the CUDA memory model. The
reported instance is `run_row_norm(x, 'layer_norm', eps)` at K=4096 with a
256-thread block (8 warps): after `tsr_sum` returns, warp 0 can re-enter the
scratch write while slower warps are still reading. This is the same defect
class fixed in the HIP paged-attention kernel this cycle (see the ROCm todo),
which is corroborating but **not** transferable evidence — it needs an sm_120
run. Two adjacent confirmed P2s in the same emitter: `:622` split-reduced
backward recomputes softmax statistics `Sk` times, and `:879` recomputes the
decay product `O(S)` per key for `O(S^3)` total.

Cross-backend sync `IKF-INTRA-KERNEL-CONTRACT-2026-08-27` — **follow-up
required at IKF-P6; SM120 execution unchanged.** The IKF-1 intra-kernel
measurement plan (`docs/audit/compiler/INTRA_KERNEL_FEEDBACK_PLAN.md`,
PR #634) defines a shared artifact schema, Tile IR trace ops, and a runtime
buffer contract, proven first on ROCm gfx1151. The NVIDIA lowering (planned
P6) maps the constant-rate clock rule to `%globaltimer`; the slot index's
wave-in-role coordinate must be re-derived from SM120's own role structure
(no wgmma/tcgen05 — consumer-Blackwell schedules differ from the sm90 plans
the key was designed against). No gfx1151 timing or schedule evidence
transfers; promotion requires an exact-SM120 clock-validation packet
mirroring IKF-P0 before any NVIDIA lowering lands.

Cross-backend sync `X86-AVX512-IMAGE-ADMISSION-2026-08-27` — **shared runtime
load safety repaired; CUDA execution unchanged.** The AVX2 Threadripper that
hosts the RTX 5070 Ti now rejects Tessera's monolithic AVX-512 CPU image through
the canonical complete-feature authority before `ctypes.CDLL`; the image itself
also performs no AVX-512 work in ELF initialization. This prevents unrelated
CUDA/solver collections from dying with SIGILL while preserving fail-closed
x86 execution. No PTX, cubin, SM120 selector, CUDA numeric policy, or RTX
certificate changes or inherits evidence from this host-side x86 repair.
The independently rebuilt Zen 5 image passes its complete 79-case FFT/solver
packet; that CPU evidence likewise transfers no SM120 execution claim.

Cross-backend sync `CUDA-SOLVER-KRYLOV-SCALE-2026-08-27` — **arbitrary dense
operator Arnoldi/GMRES, non-diagonal CG, explicit low-precision solver matmul,
and multi-CTA performance ratchets exact-SM120 closed.** A content-addressed
dense-operator v2 package represents any finite-dimensional linear map as a
row-major matrix and launches one cooperative CUDA grid for the complete
solve. Restarted GMRES retains its basis, Hessenberg matrix, Givens rotations,
work vectors, dot partials, and convergence state on device; twice-modified
Gram-Schmidt limits loss of orthogonality and an fp32 recomputed `b-Ax`, never
the Givens estimate alone, establishes convergence. The CG route admits an
authored SPD promise, checks positive curvature on device, periodically
replaces the recursive residual with the true residual, and rejects indefinite
operators. Dot and norm reductions use deterministic two-level CTA partials
with a cooperative grid barrier; the exact RTX 5070 packet exercises 2-3 CTAs
and f32/f16/bf16 storage. The live performance packet covers orders 513, 1025,
and 2049, scales reduction geometry 3->5->9 CTAs, and ratchets both complete
host calls and CUDA-event kernel time. Separately, solver residual matmul now
requires `{storage=f16|bf16, accum=fp32, math_mode=ieee}` before selecting the
native SM120 `mma.sync` route; missing or contradictory storage fails closed,
while f32 retains the scalar IEEE route and never substitutes TF32. Krylov
matrix-vector products intentionally convert their declared storage to fp32
FMA; the tensor-core claim belongs only to the registered rank-2 matmul child.
Open boundaries are compiler-fused matrix-free child callbacks, sparse/
structured operator encodings, preconditioners, and NCU-guided matvec tuning;
none is implied by the dense package.

Cross-backend sync `CUDA-SOLVER-FAMILY-2026-08-27` — **typed residual children,
broader storage policy, and dedicated device-resident CG exact-SM120 closed.**
Compiler-emitted CUDA children now execute sqrt/reciprocal/exp/log/tanh/
sigmoid/sin/cos, sum/mean/max/min, eq/ne/lt/le/gt/ge, predicate `where`, and
rank-2 matmul products. Matmul uses the resident scalar IEEE-f32 route only
when the residual product preserves an explicit `math_mode="ieee"`; missing
mode or non-fp32 accumulation fails closed and never selects TF32. Residual,
JVP, and VJP SSA replay consumes all of these target-owned children, including
pure predicate recomputation and exact reduction duals. f16/bf16 storage is
admitted with explicit package-boundary widening and fp32 Krylov arithmetic;
leaf unary/comparison/where/reduction and the dedicated CG input ABI also
execute true two-byte storage. A distinct content-addressed positive-diagonal
SPD CG package retains solution, residual, direction, matvec, dot reductions,
and convergence state in device memory for one complete CUDA launch. The
f32 packet converges in 17 iterations with max solution/equation error below
`3.6e-7`; f16/bf16 device oracles also pass. General residual GMRES remains
host-orchestrated, and device-resident arbitrary-operator Arnoldi/GMRES,
multi-CTA reductions, non-diagonal CG operators, and performance promotion
remain open.

Cross-backend sync `CUDA-SOLVER-IFT-PILOT-2026-08-27` — **diagonal-sqrt and
the first general matrix-free CUDA solver envelope exact-SM120 closed.**
The shared content-addressed IFT contract now admits `nvidia_sm120`/`sm120`
and preserves its residual, matrix-free solve, JVP/VJP mode, and parameter
product lineage into one Tile artifact. An NVIDIA-owned compiler-emitted CUDA
package executes all three phases on the RTX 5070 under CUDA 13.3. The
reproducible 30-sample packet records a maximum absolute error below `4.8e-7`,
complete-backward timing, stale-lineage rejection, and correctness-only
promotion. Compiler-generated affine residual/JVP/VJP SSA now replays through
the registered CUDA binary carrier, and a digest-bound parent executes
restarted GMRES with true-residual checks. Its separate 20-sample packet has
zero numerical error for both product directions and binds all five child
digests. This is a binary-f32 envelope: unary, reduction, comparison/where,
matmul, fully device-resident Krylov state, CG-specific execution, and broader
dtype policy remain open and fail closed.

Cross-backend sync `CUDA-BINARY-SPECTRAL-JVP-2026-08-27` — **compound spectral
JVP dispatch hole and the first general CUDA binary-math family exact-device
closed.** The public compound plugin previously declared NVIDIA ownership for
`spectral_filter` and `spectral_conv`, but two active tangents selected the
ROCm binary executor while carrying an NVIDIA artifact. `nvidia_binary_compiled`
now owns matching-shape add/sub/mul/div/pow/max/min/mod/floor-div with
f32/f16/bf16 storage and fp32 evaluation; NaN propagation, signed-zero min/max,
and floor-quotient semantics are explicit. Filter and convolution JVP tangent
terms consume that CUDA add route, while logical complex64 filter storage is
recorded as interleaved fp32. Exact RTX 5070 CC 12.0 / CUDA 13.3 execution
passes the full binary dtype matrix, public filter/convolution bilinear laws,
and public f16/bf16 STFT JVP oracles. Unsupported shape, dtype, operation, or
target still fails before launch. This is correctness closure, not a selector
or performance promotion.

Cross-backend sync `CUDA-SPECTRAL-JVP-NUMPOL-2026-08-26` — **remaining SM120
spectral admission and numeric-policy rows exact-device closed.** Public
`native_jvp` now constructs a content-addressed STFT/ISTFT child whose Graph,
Schedule, Tile, and CUDA identities are distinct and digest-bound. Production
Schedule→Tile admits the `nvidia_sm120`/`sm120` cuFFT policy rather than routing
through a sibling profile. f16/bf16 DCT, STFT, ISTFT, JVP, and VJP storage use
explicit two-byte ABIs with fp32 framing, cuFFT accumulation, overlap-add, and
window-gradient reduction. The RTX 5070 Ti packet passes 8/8 independent
forward, centered-difference, low-precision parity, and JVP/VJP adjoint checks;
unsupported or stale policy still fails closed. This closes the three
follow-ups in `TSOL-CUDA-POLICY-V1` below without transferring ROCm/x86/Metal
evidence.

Cross-backend sync `CUDA-OPTIMIZER-VJP-2026-08-26` — **SM120 optimizer reverse
execution exact-device closed for the ordered package.** The shared
non-reexecuting state-lineage carrier now admits NVIDIA as a physical owner and
stamps `nvidia_sm120`/`sm_120` through Schedule→Tile. CUDA-owned PTX packages
execute SGD, Momentum, Nesterov, Adam, AdamW, and full/factored Adafactor;
factored reductions have one deterministic owner and every output is a fresh
no-alias write. The RTX 5070 Ti packet passes all seven numerical variants and
requires runtime-origin `sm_120` attestations in content-addressed execution
certificates. This is correctness-first f32 execution, not a performance
promotion or a transfer of gfx1151/AVX-512 schedules.
The consolidated all-family packet now observes exact equality for all nine
registered NVIDIA VJP families, including `optimizer_vjp` and
`adafactor_vjp`; omitting either new family fails the packet.

Cross-backend sync `TSOL-CUDA-POLICY-V1-2026-08-26` — **SM120 f32 physical
forward, inverse, reverse, DCT, and streaming rows exact-device proven.** The
new `tessera.nvidia.spectral_policy.v1` ABI owns DCT-I/II/III/IV, STFT/ISTFT
device framing, cuFFT execution, normalization, deterministic overlap-add,
analytic signal/spectrum and broadcast-window adjoints, and causal streaming
state transitions. Exact RTX 5070/SM120 packets cover arbitrary axes, true
element strides, `n_fft >= window`, centered constant/reflect padding,
explicit inverse cropping, one-sided/full spectra, trailing batch-window
broadcast, all FFT normalization modes, DCT types, streaming artifact/parent
lineage, independent forward/VJP oracles, and the native forward/adjoint
inner-product law. NVIDIA transform capability now records complex64 as
logical interleaved-fp32 storage instead of rejecting the shipped ISTFT Graph
path. At this synchronization point the public content-addressed JVP child,
Schedule→Tile admission, and f16/bf16 storage were still open; the later
`CUDA-SPECTRAL-JVP-NUMPOL` row above closes them with exact-device evidence.
This synchronization supersedes the CUDA-physical-open clauses in
`TSOL-POLICY-PHYS-1-8C8G`, `TSOL-POLICY-PHYS-1-8B`,
`TSOL-POLICY-PHYS-1-8A`, and `AD-TSOL-STFT-GFX1151` below; those entries remain
as the history of the shared carrier landing; their subsequent closure is
recorded by the newer synchronization key.

Cross-backend sync `TSOL-POLICY-PHYS-1-8C8G-2026-08-26` — **shared spectral
carrier assessed; CUDA implementation remains open.** Schedule→Tile now binds
the runtime-stride ABI, independent transform/window lengths, and full versus
one-sided spectrum policy; streaming state has digest-chained lineage. The x86
numerical packet supplies no SM120 evidence. CUDA still needs its own
true-stride forward/adjoint package, broader/full transforms, broadcasting,
streaming package, and exact-device oracle packet. The independent AVX-512 and
gfx1151 forward/streaming packets and their artifact-bound state certificates
transfer no SM120 schedule or execution evidence. The later exact-gfx1151
expanded reverse/VJP packet likewise transfers no CUDA adjoint implementation;
SM120 retains every architecture-owned reverse row named above.

Cross-backend sync `TSOL-POLICY-PHYS-1-8B-2026-08-26` — **shared logical-axis
contract assessed; CUDA implementation remains open.** Centered STFT and
centered/cropped ISTFT now preserve arbitrary normalized logical axes and
`outer`/`inner` indexing through Schedule→Tile, while non-C-contiguous storage
still fails closed. The AVX-512 and exact-gfx1151 packets supply no SM120
execution evidence; CUDA needs its own architecture-owned forward/adjoint
package and oracle packet. Full spectrum, broader lengths, broadcasting,
streaming, and true stride support remain open.

Cross-backend sync `TSOL-POLICY-PHYS-1-8A-2026-08-26` — **shared policy
carrier assessed; CUDA implementation remains open.** Center, pad mode, crop,
and explicit ISTFT length are digest-bound through Schedule→Tile. The AVX-512
and exact-gfx1151 centered/cropped packets supply no SM120 execution evidence.
CUDA still needs an architecture-owned forward/adjoint package and exact-device
oracle packet before any policy row can be promoted.

Cross-backend sync `AD-TSOL-STFT-GFX1151-2026-08-26` — **shared carrier
assessed; CUDA implementation and evidence remain open.** gfx1151 n=16/n=18
forward/adjoint numerics and certificates do not transfer to SM120. CUDA still
needs an architecture-owned STFT/ISTFT package and exact-device oracle packet.
The structured spectral `numeric_policy` now survives Schedule→Tile, closing
the generalized carrier ceiling, but SM120 `math_mode` consumption and device
proof remain NVIDIA-owned Order 3b follow-ups.

Cross-backend sync `E2E-REAL-6F-EXACT-CERT-2026-08-26` — **initial seven-family
SM120 certificate packet exact-device closed, subsequently expanded to nine
under `CUDA-OPTIMIZER-VJP`.** The target-owned single-process
packet executes binary loss, class loss, Lion, normalization, regression loss,
sequence mixer, and spectral backward on the RTX 5070 and requires
runtime-origin `sm_120` attestations plus exact family-set equality. The
spectral row now also exercises STFT and ISTFT analytic adjoints. A CUDA-mode
string without that attestation remains `runtime_unattested` and cannot close
a row; no x86 or gfx1151 result was transferred.

Cross-backend sync `BOUNDED-GATE-RELAXATION-2026-08-26` — **shared
control-scan normalization assessed; CUDA physical outcome not applicable.**
The paired reverse pass now admits the statically bounded symbol-body scan and
keeps payload/dynamic/malformed forms closed. The f16-accumulate WMMA consumer
is exact-gfx1151 and transfers no Tensor Core claim. The AVX-512 packed-C2R and
gfx1151 direct-DFT STFT/ISTFT packages each transfer no CUDA spectral claim.
The shared
family-plugin certificate carrier is portable, but the new factored Adafactor
certificate is x86 evidence, not an SM120 optimizer execution. The bounded MPI slice now consumes all five
explicit Schedule→Tile collective SSA forms and has exact two-process x86 host
evidence, including artifact/communicator/subgroup binding. That evidence does
not transfer to NCCL: process-rank ownership and an sm_120 multi-rank packet
remain NVIDIA follow-ups; no MPI or mock result satisfies them.
NVIDIA's duplicated composed-layout materializer was also aligned with the
shared CuTe rule by retaining the slowest mixed-radix quotient. The lowering
fixture passes in an isolated hardware-free NVIDIA compiler build on this box,
but there is no RTX exact-device run, so CUDA numerical parity remains
follow-up required rather than promoted.

Cross-backend sync `W4-EFFECTS-1-E5-2026-08-25` — **one physical family carrying an admissible effect, end to end; NVIDIA outcome: not applicable — no row claimed.** E5's physical acceptance was scoped to x86 + gfx1151 and executed there. No sm_120 evidence exists or is implied; an NVIDIA row would need its own exact-device replay on NR2 Pro, since no result transfers between architectures.


Cross-backend sync `W4-EFFECTS-1-E4-2026-08-25` — **ordered-collective
recorded products (identity only); NVIDIA outcome: not applicable today, inherited on adoption.** The product
binds communicator, issue order, reduction algorithm and topology; the
verifier rejects a permuted order and a changed tree under an identical
order. Order evidence comes from the deterministic mock-mesh executor.
When an NVIDIA collective family adopts recorded products it inherits the same requirement, and NCCL deterministic-algorithm selection becomes the gating evidence for any result claim.


Cross-backend sync `W4-EFFECTS-1-E3-2026-08-25` — **shared state-lineage
identity change; NVIDIA outcome: not applicable today, inherited on
adoption.** The lineage is host-side package identity, not target codegen; no
sm_120 artifact changes. Worth knowing when NVIDIA stateful packages adopt
recorded products: the dtype field is now real, so a bf16 or fp8 optimizer
state gets its own identity rather than aliasing the f32 one.


Cross-backend sync `W4-EFFECTS-1-E2-2026-08-25` — **shared autodiff gate
change (AutodiffPairedPass); NVIDIA outcome: not applicable, no behaviour
change.** The pass is target-neutral and the change is a diagnostic split
over a fail-closed check, so no sm_120 artifact or numerical result moves and
no NVIDIA-owned surface needs revalidation. What NVIDIA inherits when a
stochastic family is admitted on its lane: the same product requirement, and
its own exact-device replay evidence.


Cross-backend sync `W4-EFFECTS-1-2026-08-25` — **UPDATED 2026-08-25 (slice E1
landed): shared recorded-product carrier + verifier implemented in Python;
NVIDIA outcome: not applicable at this slice, follow-up on adoption.** E1
introduces no target-owned surface and E5's physical acceptance stays scoped
to x86 + gfx1151, so no sm_120 row is claimed or implied. Obligations on
adoption are unchanged: exact-device replay evidence of its own, and NCCL
deterministic-algorithm selection, since the carrier requires the reduction
tree to be bound before bit-identity may be claimed.

Cross-backend sync `SPECTRAL-PAYLOAD-CHAIN-2026-08-25` — **shared
Schedule->Tile spectral identity contract + pipeline carrier ordering; NVIDIA
outcome: not applicable today, inherited on adoption.** No NVIDIA Target
consumer exists for the scheduled spectral program (its dialect is off by
default in this build and the spectral physical lanes are x86/gfx1151), so no
sm_120 artifact or numerical result changes. When an NVIDIA spectral consumer
lands it inherits the same required preimages; the pipeline-carrier ordering
fix is target-neutral and applies unchanged.


Cross-backend sync `SCHEDULE-AUTHORITY-RESHARD-2026-08-24` — **shared SO-3 and W5.4 parity validated; no CUDA physical change.** Pipeline and compound-spectral lowering now consume one digest-bound Schedule Object, inferred producer edges, roles, and resource evidence without scalar reconstruction. Placement emits exact mesh-sized local-shard/collective SSA and all movement forms execute on the deterministic mock mesh. The carrier and verifier changes are shared; Zen 5/gfx1151 spectral evidence and mock transport transfer no CUDA schedule, NCCL proof, or RTX claim. `NUMPOL-CARRIER-1` owns the generalized S5 carrier; SM120 consumption remains a later architecture-owned assessment.


Cross-backend sync `SO3-INFER-EDGES-2026-08-24` — **shared W2.1/W5.2e
dependence-inference semantics + MegaMoE R3 producer; NVIDIA outcome: not
applicable today, inherited when NVIDIA adopts MoE plans.** The change is
host-side schedule ANALYSIS (Python R3 composition), not target codegen: no
NVIDIA-owned surface compiles through it and no sm_120 artifact or numerical
result changes. When an NVIDIA MoE transport lane adopts the overlap plan it
inherits the corrected inference unchanged; the exact-device evidence rules
are unaffected (no ROCm/x86 analysis result transfers a NVIDIA claim).


Cross-backend sync `NUMPOL-CARRIER-1-2026-08-24` — **shared Schedule→Tile
`numeric_policy` carrier contract (integrated-plan queue row 3b); NVIDIA
outcome: follow-up required, sequenced behind W1.1.** Newly owned row,
nothing implemented yet. NVIDIA will consume the same carrier, but its typed
fragment producers are still open under W1.1 — the carrier work here should
follow that, not race it, or the two will collide at the same seam. CAKE
(#32's original derivation) is an NVIDIA-facing consumer, so the barrier and
TCGen05 fragment paths are the ones to check first. sm_120 exact-device
evidence required before any numerical claim; no ROCm/x86 result transfers.


Cross-backend sync `LAYOUT-ALG-APPLE-PHYSICAL-2026-08-24` — **shared ABI
assessed; no CUDA physical change.** Apple now exports the existing C++ rank-2
plan through the native layout ABI for its MSL emitters and owns fresh M1 Max
proof. NVIDIA continues to consume the same header authority in CUDA; Metal
source templates, simdgroup scheduling, and Apple evidence transfer no CUDA
schedule, PTX, or RTX claim.

Cross-backend sync `LAYOUT-ALG-L5-X86-2026-08-24` — **shared admissibility
parity validated; no CUDA physical change.** The x86 consumer follows the same
canonical dynamic-leaf order and mixed-radix mathematics already proven by the
SM120 consumer. CPU assertions and AVX2/AVX-512 evidence transfer no CUDA
schedule, PTX, or RTX claim.

Cross-backend sync `LAYOUT-ALG-L4-X86-2026-08-24` — **shared rank-2 authority
parity validated; no CUDA physical change.** NVIDIA already consumes the same
C++ coordinate mapping in its proven SM120 matmul paths. AVX-512 host admission,
loop structure, intrinsics, performance, and Ryzen numerical evidence transfer
no CUDA schedule or RTX claim.

Cross-backend sync `LAYOUT-ALG-L3-L5-DYNAMIC-2026-08-24` — **L3/SO-4,
mixed-radix/static tuple materialization, and dynamic macro-CTA execution are
closed for the stated NVIDIA subset.** The native factorization/residency ABI
and Schedule Object v2 proof are shared contracts. CUDA now consumes the shared
rank-2 index authority, mixed-radix basis maps, and static tuple codomain
products. Both the corrected narrow dynamic route (`17x19 @ 19x13`, padded
`29/31/23`) and the alignment-safe scalar-shared macro route
(`257x127 @ 127x259`, padded `139/137/269`) match FP32 oracles on RTX 5070. The
post-correctness macro NCU row is 17.536 us, 40 registers/thread, 82.85% L2
sector hit rate, and 13.89% active warps. Dynamic/non-separable tuple codomains,
explicit pointer-offset alignment metadata, and bare-metal selector authority
remain open. Apple/x86 rank-2 index-template migration is closed by
architecture-owned evidence.

Cross-backend sync `DYNAMIC-COMPOSED-SM120-2026-08-24` — **bounded dynamic
Graph matmul, arbitrary leading dimensions, and scalar-affine nested
materialization are exact-device proven.** A dynamic rank-2 NVIDIA matmul is
admitted only with static `shape_bounds=[M,N,K]`. Its canonical scheduled Tile
producer carries runtime `M/N/K/LDA/LDB/LDD`, bounded `tile.view` operands, and
dynamic outer shape/stride leaves through `tile.materialize_composed_layout`;
the CUDA target re-runs the shared C++ affine proof before emitting address
arithmetic. The versioned strided f16/bf16 descriptor validates A `(LDA,1)`, B
`(1,LDB)`, and D `(LDD,1)` element strides and the launch bridge allocates and
copies their physical spans with overflow checks. RTX 5070 execution for
`17x19 @ 19x13`, `LDA=29`, `LDB=31`, `LDD=23` matches the independent FP32
oracle, including M/N bounds and the final K tail. The proof found and fixed a
bounded B-fragment defect where the second packed lane advanced by `LDB`
instead of contiguous K. The subsequent synchronization record closes
mixed-radix/static tuple materialization and the alignment-safe dynamic macro
route. Dynamic/non-separable tuple codomains and the bare-metal selector gate
remain open.

Cross-backend sync `SCHEDULED-MATMUL-TAIL-EPILOGUE-LDS-2026-08-24` — **SM120
static K-tail and scheduled epilogue ABI closed; dynamic stride proof and
bare-metal policy remain open.** The canonical package now distinguishes
16-byte-row-aligned partial K panels from arbitrary-alignment K tails. The
former uses `cp.async` source-size zero fill; the latter uses masked scalar
shared staging. RTX 5070 cases `257x513 @ 513x257` and
`257x520 @ 520x257` match the FP32 oracle. Graph matmul now carries optional
fp32 bias and residual as ordered SSA operands, while Schedule/Tile preserve
ReLU/GELU/SiLU and f16 reduced-output policy. The widened CUDA descriptor and
runtime launch path execute bias→ReLU→residual with an f16 store on RTX 5070.
The current host is WSL, so neither
these correctness results nor the retained NCU observations can authorize a
global selector change; the bare-metal packet is still required.
After correctness, Nsight Compute on the aligned `257x520x257` tail reports
scheduled/direct duration about `15.3/23.2 us`, `56/35` registers, `4.10 KiB/0`
static shared memory, and `14.0%/23.3%` achieved occupancy. Logical input
redundancy is `10.09x/26.19x`. Reports:
`/tmp/tessera-sm120-k520-{scheduled,direct}.ncu-rep`. These are diagnostic
profiler observations, not selector authority.

Cross-backend sync `SM120-MACRO-CTA-2026-08-24` — **NVIDIA-owned async
shared-panel contract implemented and exact-device proven; global selector
unchanged.** Canonical f16/bf16-to-f32 scheduled matmul now emits the registered
`tessera_nvidia.macro_cta_matmul` Target IR boundary. Its exact contract is one
`32x32` CTA tile, four warps with `quadrant_2x2_two_n_tiles` ownership,
`m16n8k16` MMA, two 2 KiB shared A/B slots, and `cp.async` commit/wait plus CTA
barriers. The 128 threads cooperatively transfer one 16-byte vector each.
Out-of-range M/N vectors use `cp.async` source-size zero, which zero-fills the
shared panel without dereferencing an invalid address; K remains a positive
multiple of 16. Exact RTX 5070 proof includes aligned and ragged FP16 plus
ragged BF16 (`257x512 @ 512x257`).

The retained nine-sample, 1000-repetition WSL CUDA-event packet admits only
cases at or above 67,108,864 FLOPs. Its three eligible scheduled/direct ratios
are 0.721 (`256x512x256`), 0.453 (`512x256x512`), and 0.429 (`512^3`); every
eligible route has CoV below 3% and all numerical rows are green. Smaller cases
remain on the typed fallback because repeated packets exposed launch-scale
variance through 33.6M FLOPs. The route threshold is pruning-only WSL evidence,
not global `target_perf` authority; a bare-metal packet is still required for
global selector promotion.

Post-correctness Nsight Compute at `256^3` records 8.22 us scheduled versus
9.57 us direct, 48/35 registers per thread, 4 KiB/0 static shared memory, no
spills, and L1/TEX throughput pressure of about 29%/80%. Achieved occupancy is
lower (11.1%/21.6%), but panel reuse reduces load pressure enough to win. The
reports are `/tmp/tessera-sm120-async-{scheduled,direct}-256.ncu-rep`; the
durations are diagnostic profiler observations, not selector timings.

Still open for this family: dynamic/non-separable tuple-codomain
materialization, explicit pointer-offset alignment metadata, and a bare-metal
selector packet. Static separable tuple products and the alignment-safe dynamic
macro-CTA specialization are closed by the synchronization record at the top
of this plan. Bounded dynamic extents and arbitrary runtime leading dimensions are closed by
`DYNAMIC-COMPOSED-SM120-2026-08-24`; static K tails and the bounded scheduled
epilogue ABI are closed by the synchronization record above.

Cross-backend sync `SM120-SCHEDULED-LICM-2026-08-24` — **superseded for the
macro route by `SM120-MACRO-CTA-2026-08-24`; narrow typed fallback retained.** The
native pipeline now applies LICM before SCF destruction, hoisting invariant lane
and composed-layout address terms out of the typed K loop. Five RTX 5070
numerical cases pass. Final v2-packet scheduled/direct CUDA-event ratios were
1.002 (`128x128x128`), 0.929 (`128x256x64`), and 0.901 (`256x256x256`). The
first is tied and the rectangular scheduled sample contains an outlier, so one
session does not define a safe selector boundary. On the largest case, separate NCU
observations were 9.088/10.528 us, 303104/634368 DRAM bytes,
187392/510976 executed instructions, and 40/35 registers per thread. The
benchmark's logical traffic model exposes a 24x input-reuse gap at 256 cubed.
A repeated largest-case ratio was 0.914, but scheduled/direct sample CoV was
3.89%/1.76%, failing the low-variance promotion requirement.
The target-owned macro-CTA follow-up described by this earlier checkpoint is
now implemented above. This LICM record remains the evidence for the narrow
typed fallback and changes no sibling physical schedule.

Cross-backend sync `SM120-BLOCK-COORDINATE-2026-08-24` — **NVIDIA-owned
coordinate boundary and macro numerical closure landed; selector unchanged.**
The registered pure `tessera_nvidia.block_coordinate` op returns typed i64 row
and column bases and verifies the sole contract `sm_120/16x8/column_major_xy`.
Only NVIDIA lowering interprets it as `ctaid.y*16, ctaid.x*8`; the canonical
scheduled producer consumes those SSA values in its two proven composed-layout
materializations and typed K-loop MMA. RTX 5070 exact-device cases
`16x32@32x8`, `32x32@32x16`, and `48x64@64x24` pass with max absolute error
`<=5.97e-7`. Seven-sample CUDA-event scheduled/direct ratios were 1.003, 1.025,
and 1.005, so no macro selector promotion is justified. Separate Nsight
resource captures for the largest case recorded scheduled/direct 44/36
registers per thread, 1024/1024 shared-memory bytes, and equal 2.08% active-warp
occupancy. Reports are retained at `/tmp/tessera-sm120-{scheduled,direct}-macro.ncu-rep`.


Cross-backend sync `CUTE-LAYOUT-MATERIALIZE-1-2026-08-23` — **SM120 static
affine view-address bridge landed; physical layout selection remains open.**
`tile.materialize_composed_layout` now takes one i64 coordinate per outer mode,
rechecks the recursive carrier through `tessera_layout_coalesce_v1`, and admits
only the static scalar-basis subset. SM120 lowers its canonical offset into the
existing `tile.view{tile.linear_base}` consumer, preserving ordinary bounds and
the selected fragment contract. It does not select a new register/shared-memory
layout. Nested or dynamic-residue carriers fail closed. This is compiler
lowering evidence plus exact RTX 5070 numerical proof for the static f16
m16n8k16 row-major-A/column-major-B subset: nonzero A-row and B-column origins
reached `tile.view → fragment_pack → mma.sync` and matched NumPy. It remains no
layout-performance claim. Dynamic and non-affine carriers, other dtypes, and
ROCm physical consumption remain follow-up required.

Cross-backend sync `ROCM-CI-HSACO-SERIALIZE-2026-08-23` — **ROCm-owned host-free
CI serialization lane; NVIDIA outcome: follow-up available, not required.**
The transferable idea is the technique, not the artifact: device-code
*serialization* is host work, so a GPU-less runner can prove the lane still
emits an object. NVIDIA's analogue would be a cubin/fatbin emission proof, but
it is **not** a drop-in — `runtime.py`'s NVIDIA device code is NVRTC-compiled at
load with no cubin lane (Decision #26a), and CUDA codegen needs the CUDA
toolkit rather than a stock `lld`. No PTX, cubin, sm_120 schedule, or
exact-device evidence transfers from gfx1151; nothing in the NVIDIA queue
changes.

Cross-backend sync `CI-BACKEND-CAPABILITY-SKIP-2026-08-23` — **Apple-owned
pytest capability gate; NVIDIA outcome: not applicable / no exposure measured.**
Measured 2026-08-23 on a host with `TESSERA_BUILD_NVIDIA_BACKEND:BOOL=OFF`:
`pytest -k nvidia -m "not slow"` reports **496 passed, 14 skipped, 0 failed**, so
the NVIDIA fixtures do not fail-instead-of-skip when their backend is absent and
need no equivalent guard. No sm_120 artifact, cubin, or exact-device evidence is
involved.


Cross-backend sync `NVIDIA-AOT-PACKAGE-V1-HARDEN-2026-08-22` — **NVIDIA-owned
runtime package hardened and exact-device validated on SuperBear.** The f16
SM120 peer now ships both versioned fatbin and cubin images plus a generated
package manifest. Each image embeds artifact-version, physical-ABI-version,
and canonical-source SHA metadata; the loader verifies those device globals
before admitting the kernel, binds the selected format into its cache key, and
uses identical-source NVRTC for missing, corrupt, incompatible, or stale
images. Forced fatbin, forced cubin, stale-image fallback, missing-image
fallback, and cache-key separation pass on RTX 5070 CC 12.0. This adds only
NVIDIA C-ABI inspection symbols; no shared IR, operation, dtype, numerical
policy, or sibling physical package changes.

Cross-backend sync `NVIDIA-FFT-WORKSPACE-1-2026-08-22` — **canonical CUDA
FFT/workspace ABI and first C2C consumers exact-device validated on SuperBear.**
`libtessera_nvidia_fft.so` exports the versioned
`tessera.nvidia.cuda_fft_workspace.v2` contract (superseded and extended by the
sync record below): reusable opaque cuFFT plans,
automatic allocation disabled, exact workspace-byte reporting, explicit
caller-owned device workspace allocation/free, and normalized inverse
execution. `nvidia_fft_compiled` consumes the contract for complex64-logical /
interleaved-f32 physical `fft` and `ifft`, including arbitrary positive length,
nonleading axes, and explicit pad/truncate `n`. SuperBear RTX 5070 (CC 12.0,
CUDA 13.3) exact-device proof is 7/7 across radix lengths 4/16, mixed length
100, prime length 257, forward/inverse comparison with NumPy, undersized
workspace rejection, and plan/workspace reuse. The initially deferred real,
compound, and autodiff consumers are promoted in
`NVIDIA-SPECTRAL-PHILOX-JVP-2026-08-22` below.

Cross-backend sync `NVIDIA-RNG-PHILOX-CORE-2026-08-21` — **typed NVIDIA
stateless compiler/runtime package exact-device validated on SuperBear.**
`tessera_nvidia.philox` is a registered Target IR directive with
closed `uniform_core`, `uniform_range`, `normal`, and `dropout` modes.
`--generate-nvidia-philox-kernel` consumes
the explicit `(seed_lo, seed_hi, counter_lo, counter_hi)` ABI and emits a
Philox4x32-10 `gpu.func` whose threads own disjoint 128-bit counter blocks. The
NVIDIA lit suite proves directive typing, fail-closed mode rejection, constants,
launch-index construction, four bounds-checked core stores, range scaling,
Box–Muller normal transforms, and dropout replay (51/51 host-free tests). This
is paired with the shipped `libtessera_nvidia_rng.so` four-symbol ABI and the
registered `nvidia_rng_compiled` executor/manifest row. SuperBear RTX 5070
(CC 12.0, CUDA 13.3) proof is 10/10: uniform core and range are bit-exact for
zero/ragged/large counts and explicit keys/counters, normal is tolerance- and
statistics-bounded, dropout replays the exact mask, and determinism/counter
separation hold. Compiler-JVP integration remains required.

Cross-backend sync `IEEE-MINMAX-CONTRACT-2026-08-23` — **NVIDIA
outcome: assessed, exact-device tie probe required (NR2 Pro).** The
fleet-wide IEEE-754-2019 ±0-tie contract for tessera.maximum/minimum
(rocm plan owns the key). Survey of the NVIDIA emitters: reductions and
elementwise max/min emit NaN-propagating `arith.maximumf` (no
numpy-emulating tie wrapper exists here, unlike the old ROCm binary
kernel), and `maxnumf`/`fmaxf` appear only in the Philox uniform floor
(`GenerateNVIDIAPhiloxKernel.cpp`, `tessera_nvidia_rng.cu`), whose
input cannot be NaN — the accepted pattern. So NVIDIA is expected
IEEE-conformant by construction, but the ±0 tie behavior of the
maximumf lowering on sm_120 is a hardware claim: run signed-zero tie +
NaN probes on the RTX 5070 Ti (mirror
`test_binary_max_min_signed_zero_ties_are_ieee_ordered`) before
recording parity. No evidence transfers from gfx1151 or the AVX-512
host.


Cross-backend sync `JIT-MATH-AUDIT-FIXES-2026-08-23` — **NVIDIA
outcome: assessed, no defective pattern found; adafactor_vjp NaN
follow-up open (NR2 Pro).** The NaN-laundering eps-floor defect fixed
on ROCm/x86 (rocm plan owns the key) was searched for in the NVIDIA
backend: no `maxnumf(statistic, eps)` floors exist in the C++ emitters
(the only maxnum is the Philox floor, input cannot be NaN). The sm_120
`adafactor_vjp`/`optimizer_vjp` training families route through
`tessera_nvidia.training_kernel`; when their update bodies are next
touched, add the NaN-gradient reference test on the device (mirror
`test_adafactor_factored_nan_gradient_propagates_like_reference`). The
softmax running-max maxnumf optimization is ROCm-kernel-only; NVIDIA's
softmax/attention paths were not changed.


Cross-backend sync `JIT-ELEMENTWISE-LINALG-2026-08-21` — **shared
`tessera_jit` pipeline change; NVIDIA outcome: not applicable.**
`tessera_jit` is the host-CPU JIT lane; the NVIDIA backend's device
paths (NVRTC / tessera-opt pipelines) do not consume it, and NR2 Pro's
host-CPU lane is the x86 follow-up recorded in the x86 plan. No
NVIDIA-owned surface compiles through the changed pipeline; the module-scope
correction and residual legality gate therefore change no CUDA IR, kernel,
runtime ABI, or exact-device claim.


Cross-backend sync `JIT-VECTORIZE-UNGATED-2026-08-23` /
`JIT-CACHE-BLOCK-2026-08-23` / `JIT-MATH-AUDIT-2026-08-23` — **shared
`tessera_jit` boundary/pipeline changes; NVIDIA outcome: not
applicable.** The x86 plan owns these keys. `tessera_jit` is the
host-CPU JIT lane; the NVIDIA device paths (NVRTC / tessera-opt
pipelines) do not consume it, and NR2 Pro's host-CPU lane is the same
x86 lane those keys validate — though on a different x86
microarchitecture, so if NR2 Pro's host lane is ever cited as
evidence, rerun the signature-guard + totality + vectorize packet
there rather than transferring the Strix Halo result. No CUDA IR,
kernel, runtime ABI, or exact-device claim changes.

Cross-backend sync `AD-DATUM-POLYGAMMA-2026-08-21` — **autodiff reference
numerical policy, wave 3; NVIDIA outcome: follow-up required (sm_120).**
Same contract change as the rocm entry. Expected parity-neutral for the
same reason as the previous two keys (both sides of every parity
comparison read the same updated reference; dtype preserved; lgamma/
digamma primals mirror the canonical forwards bit-for-bit). Run the
CUDA-marked autodiff/loss parity tests on NR2 Pro and record here — one
NR2 Pro session can now close all three open keys (this one,
AD-RETIRE-1-POINTWISE-2026-08-20, AD-RETIRE-2-2026-08-20).
Supplemental SuperBear evidence (2026-08-22): the complete CUDA training /
loss-autodiff package lane passed 55/55 on RTX 5070 CC 12.0. This is useful
independent SM120 parity evidence but does not close the NR2 Pro-owned row.


Cross-backend sync `AD-RETIRE-2-2026-08-20` — **autodiff reference numerical
policy, wave 2; NVIDIA outcome: follow-up required (sm_120).** Same contract
change as the rocm entry. Expected parity-neutral for the same reason as
`AD-RETIRE-1-POINTWISE-2026-08-20` (both sides of every parity comparison
read the same updated reference; dtype preserved); run the CUDA-marked
autodiff/loss parity tests on NR2 Pro and record here. Note this key AND the
still-open AD-RETIRE-1 key can be closed by one NR2 Pro session.
The same 55/55 SuperBear supplemental run above is green; NR2 Pro confirmation
remains required by the owning-host declaration.


Cross-backend sync `AD-RETIRE-1-POINTWISE-2026-08-20` — **autodiff reference
numerical policy; NVIDIA outcome: follow-up required (sm_120).** PR #600
retires the ODE-family pointwise hand rules behind the `DerivativeContract`
datum (dtype-preserving reference rules; unified log/sqrt boundary guard —
see the rocm entry for the full contract statement). Expected impact on
sm_120: parity-neutral — the CUDA loss/optimizer backward lanes compare
against the same updated reference on both sides — but per the
no-evidence-transfer rule that expectation is not a claim: run the
CUDA-marked autodiff/loss parity tests on NR2 Pro and record the outcome
under this sync key. Boundary inputs below 1e-12 are outside every sampled
parity envelope; the dtype change moves the reference TOWARD the fp32 native
lanes, not away.
The same 55/55 SuperBear supplemental run above is green; NR2 Pro confirmation
remains required by the owning-host declaration.


Cross-backend sync `APPLE-RUNTIME-SINGLE-IMAGE-2026-08-19` — **Apple runtime
loading; NVIDIA outcome: not applicable.** The single-image slice fixes duplicate loading of the Apple GPU runtime.
`_apple_gpu_dispatch._prebuilt_candidate` decided whether a prebuilt dylib was
current by `ctypes.CDLL`-ing it and probing sentinel symbols. Loading is not a
read-only probe: it registers the runtime's Objective-C classes process-wide,
and skipping the candidate afterwards does not unregister them (the ObjC
runtime pins an image that has defined classes). A stale candidate therefore
stayed resident and the from-source dylib compiled next registered the same
classes again -- two images of one runtime, each with its own copy of every
file-static, including the thread_local last-error channel that
`_apple_gpu_run_checked` reads to decide whether a kernel failed. Staleness is
now read from the file's symbol table via `nm`, so a stale candidate is never
loaded; an undecidable probe (no `nm`) keeps the previous load-and-probe
behaviour rather than rejecting a library it cannot fault.
NVIDIA impact: none. The loader is Apple-specific (`_apple_gpu_dispatch.py`,
`libTesseraAppleRuntime` / `libtessera_apple_gpu_runtime`) and no NVIDIA path
reaches it. No sm_120 retest required and no device evidence is produced or
claimed.


Cross-backend sync `APPLE-STUB-BINARY-OPCODES-2026-08-19` — **shared runtime
contract; NVIDIA outcome: not applicable.** The portable-stub opcode slice fixes a silent wrong-answer class in the Apple
GPU elementwise-binary lane. `apple_gpu_runtime_stub.cpp` — compiled on every
NON-Darwin host so the C symbol exists — implemented opcodes 0-8 and its
`default:` arm assigned `out[i] = x`, so `mod`(9), `floor_div`(10), the six
comparisons(11-16), and the logical/bitwise ops(17-22) returned the LEFT operand
instead of computing. Because the symbol exists,
`_apple_gpu_dispatch_mpsgraph_binary` takes the kernel branch rather than its
numpy fallback, so those values came back as if computed, with no diagnostic
(Decision #21). Fixed by implementing opcodes 9-22 to match
`mpsg_binary_node` and the declared host reference
`runtime._apple_gpu_binary_numpy`, and by rejecting an unknown opcode through
the stub's last-error channel (new kind 3) so it routes to the host fallback
instead of returning a plausible buffer.
NVIDIA impact: none. `tessera_apple_gpu_mpsgraph_binary_f32` is referenced only
by Apple files (`runtime.py`, `_apple_gpu_backend.py`, `_apple_gpu_dispatch.py`,
`apple_exact_device_proofs.py`, the two Apple runtime TUs, and
`SiluMulToAppleGPU.cpp`); no NVIDIA lowering or runtime path reaches it. No
sm_120 retest required and no device evidence is produced or claimed.
Cross-backend sync `ZERO-FUNCTION-CANDIDATE-2026-08-19` — **shared frontend ABI
and diagnostics; NVIDIA outcome: not applicable today; parity by construction
for future migrations.** The zero-function-candidate slice (PR #590) changes `JitFn`'s call ABI recovery
and adds one diagnostic code. A `@jit` function whose AST lowering produced no
function raised a bare `IndexError` from `_establish_tracer_authority`, and the
same absence left `_call_arg_names`/`_constraint_ir_args` empty — silently
mis-binding keyword calls and skipping call-time constraint re-checking. The ABI
is now derived from the Python signature via the shared
`graph_ir.ir_args_from_signature` (Decisions #30/#31), and the apple_gpu tracer
lane lifts foreign interpreter exceptions into the new registered
`JIT_APPLE_GPU_TRACE_FAILED` code (Decision #21). `TesseraTraceError` passes
through unwrapped.
NVIDIA impact: none today. The diagnostic is apple_gpu-scoped, and the
trace-defer route that reaches it is gated on `target == "apple_gpu"`. The ABI
recovery is target-independent and strictly widens what was previously an empty
tuple, so no NVIDIA behaviour changes. No sm_120 retest required and no device
evidence is produced or claimed. Future sm_120 frontends inherit the corrected
keyword binding and constraint re-check.


Cross-backend sync `SCALAR-SIDE-ORDERING-2026-08-19` — **shared Graph IR
runtime contract; NVIDIA outcome: not applicable today.** The `scalar_side` slice (PR #589) makes the Graph IR lifted-scalar form carry
operand order. `graph_ir._OpExtractor._try_map_binop` lifts a literal out of
either side of a `BinOp` into the `scalar` attribute and records the side; until
now no code in `python/`, `src/`, or `tools/` read that record (Decision #29), so
`2.0 - x` and `x - 2.0` emitted indistinguishable IR and any consumer binding
`scalar` as the right operand computed `x - 2.0` for both — sign-flipped for
`sub`, reciprocal for `div`, with no diagnostic. Shared contract changed: a lone
`scalar` means the RIGHT operand, `scalar_side="left"` requests the mirrored
binding, and any other value is rejected rather than guessed (Decision #21).
NVIDIA impact: none. No NVIDIA lowering or runtime path consumes the `scalar`
kwarg — an exhaustive sweep for `get("scalar"`/`["scalar"]`/`get("other"` across
`python/tessera/` finds consumers only in `runtime._apple_gpu_dispatch_mpsgraph_binary`,
`runtime._execute_runtime_cpu_op`, and `matmul_pipeline._execute_op`, none of them
NVIDIA-specific. No sm_120 retest required and no device evidence is produced or
claimed. If an NVIDIA elementwise lane later accepts the lifted-scalar form, it
inherits this contract and must honor `scalar_side` for its non-commutative
opcodes.


Cross-backend sync `AD-LAW-SERIES-2026-08-19` — **shared reference rules and
test infrastructure; Nvidia outcome: not applicable today; parity by construction for future migrations.** The AD-LAW series (PR #588)
closes the swallowed-kwarg class registry-wide and adds the AD-WEIL-1 algebra
substrate plus all six executable laws. Reference-rule changes in this slice:
`stft`/`spectral_conv` JVPs **deleted** in favour of derivation from the
forward (both are bilinear, so every configuration key is honored by
construction); `jvp_istft` rewritten to honor axis/center/length/onesided/norm
while preserving its window quotient; `dequantize_nvfp4` fixed to accept the
per-block scale array its canonical forward takes (both modes previously
crashed); `jvp_lgamma`/`jvp_digamma` replaced (a dead zero-returning stub and
an identity placeholder); `jvp_cast` fixed for canonical dtype strings; the
shared polygamma helpers given reflection formulas (previously an O(n) loop
that hung on valid negative input — a live defect in the REVERSE path too).
NVIDIA impact: no native binding consumes these reference rules yet, so nothing retests. Future sm_120 backward family migrations inherit the law-checked oracle lane and the E2E-REAL-6 Law-3 gate. The previously recorded open spectral/quantize swallow findings are
therefore CLOSED; `_OPEN_FORWARD_KEY_SWALLOWS` (42 entries from the tape
positional-routing scan) remains the open set.


Cross-backend sync `AD-LAW-1-SHARED-ORACLE-2026-08-18` — **shared test
infrastructure; NVIDIA outcome: not applicable today, parity by
construction for future migrations; no sm_120 evidence changed.** AD-LAW-1
(PR #584) adds law oracles (adjoint + canonical-forward chain) over the
shared numpy reference JVP/VJP registries and the byte-gated
`autodiff_law_audit` dashboard. NVIDIA has no native binding to the two
fixed reference rules (`jvp_rmsnorm` eps default, `jvp_clamp` kwarg names),
so nothing retests; future sm_120 backward family migrations inherit the
law-checked oracle lane and the E2E-REAL-6 Law-3 gate (#583). Open shared
follow-up: the pinned swallowed-kwarg findings in `test_autodiff_laws.py`.
*Triage update, same key (AD-LAW-1b):* reference JVP fixes landed for
`clip` alias deafness, `add`/`mul` unary-`scalar`, and fft/ifft/rfft/irfft
`norm` handling; five entries benign-classified; the open set is now the
stft/istft/spectral_conv and quantize families only. *Spec-growth update, same key (AD-LAW-1c):* law coverage roughly doubled (109 adjoint / 87 chain rows green incl. attention, spectral-complex, structural, and loss families); two more silent reference JVP defects found and fixed — `lgamma` (derivative was a dead stub returning 0) and `digamma` (whole JVP was an identity placeholder) — plus `jvp_cast` crashing on canonical dtype strings. Forward-mode reference oracles for those three ops changed; reverse-mode VJPs are untouched, so no backend backward package is affected. Reflection formulas added to the shared polygamma helpers (`_digamma_positive`/`_trigamma_positive`): the upward recurrence advanced by 1 per step, so a valid input like -1e9+0.5 spun for ~10^9 iterations — a live defect in the **reverse** path too, since `vjp_lgamma`/`vjp_digamma` already call these. Now O(1) on the whole real line, exact against the canonical forward, poles -> nan.

Cross-backend sync `W4-DYNAMIC-EFFECT-NONLINEAR-CFG-2026-08-18` — **shared
contract parity; CUDA follow-up required.** Dynamic saved-slot data/shape
tapes, exact polynomial witness guards, and variadic branch CFG state are
target-neutral contracts. Effectful replay remains fail closed except for
compiler-owned extent assertions. NVIDIA has no native region-product consumer
or SM120 evidence and inherits no x86/gfx1151 proof.

Cross-backend sync `W2.4-E2E6-SYMBOLIC-2026-08-18` — **shared legality and
frontend authority landed; CUDA evidence unchanged.** Production pipeline,
WarpSpec, barrier-reuse, and derived Tile dataflow relations now run through
one staged `TileDataflowLegalityPass`; legacy CLI names are wrappers. Pure
static annotations abstract-trace before AST compatibility capture. The
existing SM120 Lion and causal DeltaNet launchers are now selected by explicit
family plugins instead of `JitFn` target dispatch. This is an ownership move,
not new device evidence; CUDA state-lineage package unification and an exact
SM120 packet remain open.

Cross-backend sync `W1-W3-AUTHORITY-CLOSEOUT-2026-08-18` — **shared contracts
landed; CUDA producer proof remains open.** CUDA-math kinds and TCGen05 operands
are ODS-typed, bare Tile fragments are rejected, and WarpSpec no longer trusts
legacy ancestor markers. Existing SM120 loss packages are reached through
explicit VJP plugins. The final tensor-valued MMA producers, barrier-at-birth
emission, SM120 Target proof, and Lion/DeltaNet shared-lineage proof remain
NVIDIA-owned; host-free verification transfers no device evidence.

Cross-backend sync `W4-PRODUCT-1-RESIDUAL-CONTRACT-2026-08-17` — **shared
contract parity validated; CUDA physical follow-up required.** SAVE/HYBRID
selection now has a digest-bound Graph→SCF carrier. Shared paired AD consumes
dynamic branch-local residual extents, bounded SAVE and sparse HYBRID `while`
state tapes, and bounded-dynamic counted-loop tapes. The target-neutral
compiler now classifies source-CFG SCCs and structurizes bounded pure reducible
or irreducible native CFGs as a typed program-counter state machine while
preserving CFG/Presburger identity. Nested canonical structured bodies and
mixed control/tensor state are admitted. Saved dynamic slots require total
data/shape-tape envelopes; unbounded, unsupported-region, and unrecorded
effectful forms remain fail closed. SM120
still needs its own region-product correctness/performance packet and inherits
no sibling proof.

Cross-backend sync `E2E-REAL-6F-OPTIMIZER-VJP-2026-08-17` — **shared
optimizer lineage validated; NVIDIA not applicable for the bounded physical
set.** The new `schedule.optimizer_vjp` → `tile.training_kernel` authority
does not declare CUDA consumers because no matching native reverse package
was proved in this slice. SM120 remains fail closed and inherits no AVX-512 or
gfx1151 evidence.

Cross-backend sync `E2E-REAL-6E-STATEFUL-VJP-2026-08-17` — **shared
Adafactor/sequence-mixer authority validated; NVIDIA target follow-up
required.** Factored/full Adafactor and causal DeltaNet backward now have a
non-reexecuting Graph→Schedule→Tile package for x86/gfx1151. The existing
SM120 DeltaNet path is now plugin-owned but remains a compatibility package;
NVIDIA has no
Adafactor Target owner. Neither inherits sibling numerical evidence; each must
adopt the shared lineage carrier and produce CUDA exact-device proof.

Cross-backend sync `E2E-REAL-6D-LION-VJP-2026-08-17` — **shared flat Lion
Graph and non-reexecuting proof parity validated; CUDA plugin follow-up
required.** The existing SM120 PTX Lion VJP remains numerically valid, but it
is now selected by the family plugin but does not yet consume the shared
`schedule.lion_vjp` state-lineage artifact. A CUDA follow-up must extend that
typed package to SM120 and rerun its owning-device packet. No x86/gfx1151 evidence
transfers.

Cross-backend sync `E2E-REAL-6C-ATTENTION-VJP-2026-08-17` — **shared
rank-4 authority and bottom-right ragged-causal semantics validated; CUDA
plugin follow-up required.** The shared family registry now owns
`flash_attn`/GQA/MQA reverse through tracer Graph →
`schedule.attention_backward` → `tile.attention_backward_kernel`. NVIDIA’s
existing SM120 package is unchanged and inherits no x86/gfx1151 evidence. A
CUDA plugin consumer and independent SM120 numerical packet are still required.
The rank-3 `multi_head_attention` wrapper remains outside this bounded rank-4
migration, and active dropout needs keyed, non-reexecuting replay proof.

Cross-backend sync `E2E-REAL-6B-SPECTRAL-VJP-2026-08-17` — **shared tracer
and package-contract parity validated; CUDA physical follow-up required.**
Concrete AST specialization now resolves the same shape-derived spectral
identity as tracing, and compound spectral reverse products have one declared
Graph/Schedule/Tile/Target family-plugin boundary. NVIDIA is deliberately not
a Target consumer in this slice: no PTX package, CUDA runtime path, or SM120
evidence was added. A future CUDA consumer must own its package and independent
device proof.

Cross-backend sync `GFX1151-CALIB-BAREMETAL-2026-08-16` — **shared calibration
authority parity validated; no CUDA evidence transfers.** `target_perf` now
rejects explicitly provisional and WSL-hosted corpora from its measured
selector registry while exposing a non-mutating pruning reader. *(Superseded 2026-09-26 by `WSL-TIMING-ADMISSION-2026-09-26`, top of this file: a WSL corpus is now admitted when it carries admissible kernel-clock witness samples.)* NVIDIA code,
SM120 selectors, and CUDA packets are unchanged. A future SM120 corpus still
requires its own CUDA-event/CUPTI-correlated evidence.

Cross-backend sync `REF-TIER-PHYS-2026-08-16` — **shared Schedule contract
received; CUDA physical follow-up required.** Batched tridiagonal solve and
the four coalition zeta/Mobius transforms now have content-addressed
Schedule→Tile carriers. The coalition transforms share one parameterized
Yates butterfly rather than four emitters. NVIDIA gains no PTX consumer or
SM120 claim in this slice; a future lane must select an architecture-owned
parallel solver/butterfly schedule and provide independent device evidence.
The shared Schedule Object now snapshots nested resource metadata before
digesting, and dynamic rearrange/GQA-fold inference preserves ranked `?`
dimensions; parity is validated without transferring a physical schedule.

Cross-backend sync `LAYOUT-SCHEDULE-OBJECT-2026-08-16` — **shared carrier
parity validated; physical producer follow-up required.** The native layout ABI
and GQA fold transfer no CUDA layout or raster decision. SO-1 now owns the
content-addressed action/edge/role/residency value; SO-2 registers symbolic
producer/consumer roles on Tile mbarriers and proves the Hopper split with the
same rule as AMD ping-pong. Shared loop-carried role provenance and role-bearing
pipeline state are implemented. Barrier-at-birth now lands on the canonical
typed producer path: WarpSpecialization creates the role-bearing barrier beside
the roles, AsyncCopyLowering binds that exact SSA value to copies and waits,
and NVTMA assigns region-local slots without synthesizing a function-global
barrier. Nested streaming and flattened single-device paths are both covered;
multi-region isolation is regression-tested. Deletion of the remaining legacy
WarpSpec ancestry assumptions and exact-device SM120 proof remain open. No
raster output or selector evidence changed.

Cross-backend sync `ATTN-BWD-ARCH-2026-08-16` — **no NVIDIA result transfers.**
ROCm's canonical split backward program was re-audited and x86 gained a
deterministic parallel implementation. The SM120 architecture-owned backward
package and performance evidence remain independent.

Cross-backend sync `X86-PASS-DIALECT-DEPENDENCY-2026-08-16` — **parity
validated; no NVIDIA physical follow-up.** The shared pass library now models
the optional hardware-free x86 Target dialect as a declared MLIR pass
dependency and fails closed when it is absent. No NVIDIA dialect, pipeline,
CUDA ABI, schedule, selector, or SM evidence changed.
The same closeout removes a private permissive `schedule` dialect from the
shared transform library; all transforms now consume the canonical ODS
`TesseraScheduleIR` authority. NVIDIA receives the build-parity fix only.

Cross-backend sync `PDE-EXACT-CONTRACT-2026-08-14` — **shared semantic parity
validated; CUDA physical follow-up required.** Exact-rational PDE
classification and the first fail-closed diffusion stability certificate are
backend-neutral. No PTX stencil/boundary/halo package or evidence is claimed.

Cross-backend sync `DIST-SHARD-HVP-2026-08-14` — **shared compiler parity;
native NVIDIA follow-up required.** Reshard plans now materialize bulk
collectives as Graph→Schedule→Tile SSA with subgroup/region identity and
deterministic all-to-all matching rounds. Exact forward-over-reverse HVP now
exists as a compiler Graph product. NVIDIA receives those shared contracts but
adds no NCCL launch binding or SM120 HVP package in this slice; native
multi-rank and product evidence remain independent gates.

Cross-backend sync `E2E-REAL-6-NATIVE-VJP-2026-08-14` — **normalization parity
validated; no new SM120 evidence claimed.** The existing SM120 normalization
backward launch is now owned by the shared native-VJP family registry with
explicit Graph/Schedule/Tile/Target declarations instead of package
construction inside `JitFn`. Runtime semantics and the existing SM120 evidence
row are unchanged. Other NVIDIA backward families remain compatibility paths
until migrated independently.

Cross-backend sync `AMD-ISA-DTYPE-2026-08-14` — **parity assessed; no NVIDIA
physical change required.** The new selector is AMD-specific and changes no
Graph dtype spelling, public operation, Schedule contract, NVVM/PTX ABI, or
SM120 capability. AMD FP8/BF8/FP4/MX and sparse-WMMA evidence cannot be used as
CUDA evidence. The shared `OP-DTYPE-FLOW-1` generator now audits SM120 by
operator and storage dtype; target-wide derived legality remains `legal_only`
unless an NVIDIA manifest owns the physical kernel.

Cross-backend sync `CI-LIT-BACKEND-DIALECTS-2026-08-12` — **not applicable to
NVIDIA, by existing architecture.** The `Validate / lit` lane was dead from
2026-08-11 to 2026-08-12 (pytest collection aborted on a missing `ml_dtypes`,
fixed in #554); the first green-collection run failed 27 of 367 fixtures on
unregistered `tessera_x86` / `tessera_rocm` dialects. No NVIDIA fixture is in
the failure set and this PR changes nothing for NVIDIA: the NVIDIA dialect is
off by default and its fixtures run under the separate `tessera-nvidia-opt`
driver (`%tnv`), not the `tessera-opt` binary this lane builds — the same
separation `test_target_ir_contract.py` already records as NVIDIA's documented
skip under Decision #19.

Architecture-specific reason this stays not-applicable rather than
follow-up-required: adding `TESSERA_BUILD_NVIDIA_BACKEND=ON` to the shared lane
would be actively harmful. With `ENABLE_CUDA` off — the only option on a
CUDA-less runner — `tools/tessera-opt/CMakeLists.txt:95-96` forces
`TESSERA_OPT_LEAN_ARTIFACT_DRIVER`, dropping core `TesseraIR`/`TesseraPasses`
and the x86 Target IR. That is the same trap documented for ROCm under this
sync key; NVIDIA's separate driver is the existing, correct answer. Revisit
only if a CUDA-capable runner joins CI, which would also be the point at which
sm_120 exact-device evidence becomes schedulable.
Cross-backend sync `CI-LIT-DEPS-2026-08-12` — **parity validated; no physical
follow-up required.** PR 554 made the shared opt-in MLIR lit lane install the
workflow-owned Python dependency set before `lit`/FileCheck collection. This is
backend-neutral test infrastructure, changes no compiler/runtime contract, and
requires no CUDA package or exact-device evidence.

Cross-backend sync `PDE-STENCIL-FOUNDATION-1-2026-08-12` — **shared semantic
parity validated; CUDA physical follow-up required.** Explicit coefficients,
scheme/order, and per-axis spacing are compiler requirements. The absent
NVIDIA stencil/halo/boundary symbols are artifact-only rather than callable
Target records. No SM120 package or CUDA packet was added.

Cross-backend sync `BLOCK-ATTNRES-ROCM-2026-08-12` — **follow-up required.**
The shared Block AttnRes plan establishes portable balanced-partition,
epsilon-qualified numeric, VJP, and softmax-merge oracle contracts. This PR
adds Phase-1 stats/merge/finalize references, the Phase-2 stdlib recurrence,
Phase-3 typed Graph/VJP/JVP contracts, and the Phase-4 content-addressed
Schedule→Tile artifact, but no SM120 package or CUDA evidence. NVIDIA must
later provide its own
stats-attention/merge Target consumer and exact-device packet; the small depth
shapes do not by themselves justify tensor-core lowering, and ROCm proof does
not transfer.

Cross-backend sync `MODEL-FUSED-PHYS-1-2026-08-12` — **shared MiniMax MSA
package lineage landed; SM120 consumption remains follow-up required.** x86
and gfx1151 now consume exact digest-bound MSA artifacts without Graph
redispatch. No CUDA package or SM120 evidence was added, and those schedules do
not transfer. NVIDIA must bind the same parent-digest contract to its canonical
NVVM/PTX package; DeepSeek MLA/DSA remain independently open.

Cross-backend sync `MODEL-WEIGHT-PHYS-1-2026-08-12` — **shared physical-byte
weight ABI landed; SM120 FP8/INT4 consumption remains follow-up required.** The
carrier preserves genuine checkpoint bytes and separate fp32 scales under a
content digest and forbids full-weight materialization. No CUDA package or
SM120 packet was added, and gfx1151 INT4 evidence does not transfer. NVIDIA
must bind this ABI to its architecture-owned FP8/INT4 package and exact-device
performance proof.

Cross-backend sync `W4-PRESBURGER-SHARD-2026-08-12` — **shared analysis
contracts landed; SM120 consumption remains follow-up required.** Graph IR now
carries typed integer-affine plus exact modular/divisibility constraints into
the C++ Presburger consumer. The shared sharding layer has a fail-closed
replicated/tiled/partial-reduction fixed point and explicit reshard planner;
lowered `control_scan` owns shared recompute-all JVP/VJP products. This adds no
CUDA region product, reshard lowering, NCCL packet, or SM120 evidence; other
architecture results do not transfer.

Cross-backend sync `W4-CFG-RESIDUAL-W5.2G-2026-08-14` — **shared compiler
carriers and scalable scheduler landed; CUDA follow-up required.** The
tracer-owned structured CFG, block-wide Presburger identity, and executable
SAVE/HYBRID residual ABI change shared lineage only. The action-DAG model now
uses deterministic critical-path/list scheduling with safe lower-bound pruning
and a small-DAG exhaustive oracle. No SM120 region product, physical producer
wiring, calibrated packet, or selection claim was added; architecture evidence
does not transfer.

Cross-backend sync `E2E-AUTH-DAG-2026-08-12` — **shared native-product v3 and
automatic dependency consumption landed; SM120 remains follow-up required.**
Reduction and normalization now have truthful Schedule/Tile product carriers,
native JVP/VJP paths require cached tracer/AST differential proof for pure
programs, and Graph-derived R3 candidates consume compiler-generated edges.
No CUDA product child, physical dependency consumer, or SM120 packet was
added; x86/gfx1151 evidence does not transfer.

Cross-backend sync `E2E-AUTH-DAG-2026-08-11` — **shared frontend authority and
automatic dependence-edge contracts landed; SM120 remains follow-up
required.** Pure straight-line tensor signatures now cache tracer-owned Graph
IR and can be differentially certified against the retained AST candidate.
Native-JVP plugins declare their Graph/Schedule/Tile/Target disposition and own
package construction; compatibility gaps are explicit. W2.1 facts now generate conservative Tile action-DAG
edges with reason and analysis digests. This adds no CUDA family package,
edge-consuming physical pipeline, selector promotion, or SM120 evidence.

Cross-backend sync `AD-SOLVER-ISTFT-PHYSICAL-2026-08-11` — **shared product
contracts landed; SM120 consumption remains open.** Graph IR now represents
the exact ISTFT spectrum/window product, and the general solver parent binds
residual plus solution/parameter JVP/VJP children under restarted GMRES with a
true-residual gate. The shared compiler now derives those five children for
typed pointwise, sum/mean, rank-2 matmul, bounded-dynamic/mixed-storage,
distinct parameter-space, and statically counted-region residual Graphs.
Pure scalar `if`/bounded-`while` predicates now have explicit compare/select
replay in the shared child contract. The AVX-512 and gfx1151 packages and
packets do not transfer to CUDA. NVIDIA still needs PTX children and independent SM120 correctness and
performance evidence.

Cross-backend sync `E2E-REAL-6-JVP-SOLVER-2026-08-11` — **shared frontend and
family-plugin boundary landed; SM120 remains follow-up required.** Native
forward-product specialization now originates in tracer-produced canonical
Graph IR, and explicit family plugins own reduction, normalization, FFT, and
compound-spectral planning outside `JitFn`. General solver contracts bind exact
residual/JVP/VJP identities and can execute matrix-free reference products
without finite differences. This adds no CUDA family plugin, native package,
or SM120 evidence; the AST lane remains for unmigrated families.

Cross-backend sync `AD-FWD-DIST-3-2026-08-11` — **shared exact JVP,
structured-region products, and typed point-to-point transport landed; SM120
evidence remains open.** Public JVP/jacfwd no longer substitutes finite
differences. Compiler forward mode carries primal/tangent state through bounded
SCF. `collective_permute` reaches the existing one-process/multi-device NCCL
launcher as grouped send/receive with an explicit peer map. SM120 still needs
architecture-owned JVP packages, subgroup communicators, and exact multi-GPU
correctness/performance packets.

Cross-backend sync `W4-SOLVER-REGION-2026-08-11` — **shared bounded-region
adjoints and general matrix-free solver policy landed; CUDA consumption is
follow-up required.** Portable tracing now emits bounded SCF, and the paired
compiler differentiates effect-safe single-block `if`, counted `for`, and
canonical bounded `while` with implicit captures. General residual execution
uses restarted GMRES/CG policy in shared IR. This adds no SM120 region executor,
solver package, checkpoint packet, or performance claim.

Cross-backend sync `COMP-GRAPH-DATAFLOW-W2.1-2026-08-11` — **shared
analysis substrate landed; CUDA behavior and evidence are unchanged.** Graph
IR now has one fail-closed, invalidatable shape/alias/liveness/memory-
dependence/activity analysis with C++ and Python query surfaces. Reverse AD and
await sinking consume it. This is target-independent legality infrastructure;
it transfers no SM120 schedule or performance evidence. Region-aware clients
and native CUDA overlap proof remain separately owned.

Cross-backend sync `AD-FWD-FAMILY-2-2026-08-11` — **shared affine,
compound-spectral, solver-product, and native-collective contracts landed;
SM120 consumption remains open.** Compound spectral Graph operations now own
direct tangent interfaces, including an exact ISTFT window-product carrier, and solver
artifacts distinguish JVP/non-transposed from VJP/transposed solves. The
multi-rank product accepts only a live NCCL hardware adapter, but no SM120
correctness/performance packet is claimed by this shared-contract slice.

Cross-backend sync `AD-FWD-NATIVE-1-2026-08-11` — **shared native-product
lineage landed; SM120 consumption is follow-up required.** The parent artifact
schema binds paired-JVP IR to immutable ordered child packages and detects
child substitution. Only x86/AVX-512 and ROCm/gfx1151 are executable in this
slice; no schedule or evidence transfers to CUDA. SM120 needs family-owned
product packages and exact-device numerical/performance packets before a
native JVP matrix row may be added.

Cross-backend sync `COMP-EFFECTS-W2.2-2026-08-10` — **shared registered-effect
analysis closed; no CUDA evidence claim.** Canonical Graph records now carry
effect, alias, mutation, and stochastic identity; Python and C++ consume the
same fail-closed facts and internal calls reach a fixed point. Await sinking
uses that shared query. This changes scheduling legality only; SM120 still owns
native overlap execution and exact-device correctness/performance packets.

Cross-backend sync `COMP-SCHED-OVERLAP-1-R4-2026-08-10` — **shared functional
MegaMoE plan consumption landed; NCCL/SM120 evidence remain open.** The
content-addressed plan binds chunk slices, per-expert capacity, two-live-frame
workspace limits, true-use dependencies, ordered collectives, and deterministic
combine order. R3 only prunes complete measured plan records; scalar CUDA-event
latency selects. Mock multi-rank execution transfers no performance claim.
SM120 still needs native NCCL stream/event binding and exact-device packets.

Cross-backend sync `COMP-SCHED-OVERLAP-1-R3-2026-08-10` — **shared prune-only
Tile action-DAG model landed; no CUDA selection claim.** R3 validates explicit
dependencies and calibration identity, uses deterministic critical-path/list
scheduling, and composes compute/memory/communication lanes with queue
serialization. Exact small DAGs and proven lower-bound losers may be pruned;
every estimate is promotion-ineligible and scalar
measured latency remains authoritative. SM120 needs its own calibrated vectors
and exact-device packet before using this analysis; R4 production consumption
does not transfer from another backend.

Cross-backend sync `COMP-SCHED-OVERLAP-1-R2-2026-08-10` — **shared measured
resource-vector schema landed; CUDA evidence remains architecture-owned.**
Successful measured autotune rows may record compute time, dtype-correct bytes
moved, communication bytes, queue/resource identity, timing provenance, and
the measured-candidate digest. Analytical rows cannot claim the vector, and
scalar measured latency remains selector authority. No gfx1151 or x86 timing,
queue, or resource identity transfers to SM120; CUDA event/activity providers
must populate their own provenance before R3 composition analysis can use it.

Cross-backend sync `COMP-SCHED-OVERLAP-1-R1-2026-08-10` — **shared explicit
async lineage and fail-closed await sinking landed; CUDA consumption remains
architecture-owned.** Python Schedule→Tile no longer emits internal
`tessera.queue.*` compatibility markers: async copies produce named tokens and
waits consume them. Collective awaits move only across operations proven
memory-effect-free; mutation, RNG, aliases/casts, regions, and ordered
collectives are barriers. Existing typed CUDA token lowering is unchanged;
SM120 needs independent executable overlap and exact-device proof.

Cross-backend sync `AD-STOCHASTIC-RNG-1-2026-08-10` — **shared stochastic JVP
contract available; CUDA core consumption started.** Explicit key/counter Graph ops,
estimator provenance, dropout replay, fixed-key EGGROLL JVP, and derivative
proof obligations are target-independent. The typed NVIDIA generator and its
four native modes now consume the explicit key/counter ABI, but x86 and gfx1151
runtime evidence still does not transfer to CUDA; SM120 needs compiler-JVP
packaging, runtime launch, and exact-device proof.

Cross-backend sync `AD-FWD-PRODUCT-2-2026-08-10` — **public JVP ABI landed;
CUDA execution remains follow-up required.** Forward/JVP requests now carry
mode-neutral provenance and stable `wrt_indices`, and the compiler emits only
requested tangent terms. Tanh/sigmoid add direct CPU-oracle proof. No PTX
package, selector, or SM120 evidence transfers; native JVP remains fail-closed.

Cross-backend sync `AD-FWD-CORE-1-2026-08-09` — **shared compiler JVP
foundation landed; NVIDIA physical consumption remains architecture-owned.**
The Graph dialect now exposes compiler-owned tangent rules and a paired
`--tessera-autodiff-forward` function contract. Matmul/mul has independent CPU
IR numerical proof, while unsupported active operations and regions fail
closed. The generated ledger distinguishes compiler `ir_tangent` evidence from
Python JVP registration. This changes no PTX package or SM120 evidence; NVIDIA
must lower and prove any native JVP package independently.

Cross-backend sync `X86-TYPED-FAMILY-PLUGIN-2026-08-09` — **shared schema
parity assessed; no CUDA physical change.** x86 now validates a closed Tile
family and registered Target marker before selecting its prebuilt AVX-512
image. NVIDIA may reuse the schema and fail-closed family discipline, but no
x86 ABI call, schedule, package, or Zen 5 evidence transfers to SM90/SM120.
The x86 backward-family allowance is narrowly one explicit forward-recompute
companion, not a general multi-carrier escape hatch. The canonical NVVM/PTX
image boundary remains NVIDIA-owned.

Cross-backend sync `EGGROLL-ES-LOWRANK-2026-08-09` — **the shared Graph,
Schedule, Tile, lineage, member-RNG-v1, and fp32 numeric-policy contract has
landed; NVIDIA physical consumption is follow-up required.** gfx1151 owns the
first GPU exact-device rank-1 proof and Zen 5 now owns an independent AVX-512
fp32 package; neither is a portable SM schedule. NVIDIA remains a lead
performance target: an SM-owned SGMV / `mma.sync` implementation and numerical
packet are open. The `s32` lane maps to native int8→s32 tensor cores and remains
a separate EGG expansion. W4 scalar-gather/member reconstruction passes
mock-mesh proof; native NCCL multi-rank execution and a target packet remain
open. Contract:
`docs/audit/compiler/EGGROLL_SUPPORT_PLAN.md`.

Cross-backend sync `COLLECTIVE-RCCL-ADVANCED-LANES-2026-08-09` — **shared
artifact discrimination adopted; AMD transports not applicable.** Advanced
collective artifacts now distinguish Copy Engine, GIN/RMA, and gfx1250 DDA and
bind target architecture plus selector evidence where required. These RCCL
lanes do not transfer to SM120. NVIDIA device-initiated communication remains
an architecture-owned NCCL follow-up with its own public API, legality, and
exact-device packet; it must not reuse an AMD transport or selector claim.
The shared Target dialect now registers explicit window-lifecycle and
put/signal/wait operations, but those records are RCCL GIN-gated and do not
imply NCCL device-initiated support on SM120.
The native multi-process harness binds only the RCCL GIN ABI; its launcher
metadata and evidence schema may be reused, but no AMD operation or result
transfers to NVIDIA's separately gated NCCL device-initiated lane.

Cross-backend sync `COLLECTIVE-NATIVE-FOUNDATION-2026-08-09` — **host adapter
and artifact contract landing; SM120 evidence open.** The C++ NCCL adapter now
executes all-reduce, reduce-scatter, all-gather, and grouped send/receive
all-to-all from an explicit communicator and CUDA stream instead of compiling
to successful no-ops. The shared Target artifact binds initiation,
registration, ordering, capture compatibility, backend/source identity, and a
capability digest. Shared communicator-property discovery, move-only symmetric
window ownership, and runtime-digest rejection are available, but still need a
CUDA-enabled build and SM120 packet. No AMD LSA, Copy Engine, GIN, or DDA claim
transfers.

Cross-backend sync `COLLECTIVE-ASYNC-UNIFY-2026-08-09` — **shared software
contract closed; SM120 NCCL evidence open.** The legacy unregistered
`tessera.collective.*` markers are gone from active producers and fixtures.
Forward and adjoint passes emit registered futures, await their payloads, and
rewire SSA uses. Runtime topology validation now fails closed for unknown or
unsupported subgroup mesh axes, and native v1 forbids implicit non-fp32
conversion. Exact multi-GPU NCCL correctness/performance remains required; no
PTX selector or device claim changes.

Cross-backend sync `DIST-SHARD-ALIAS-1-2026-08-09` — **shared alias mapping
available; SM120 evidence open.** Five public reduction/broadcast aliases now
resolve to the registered all-reduce/all-gather transport; three sharding
entries remain compile-time placement/region contracts. `collective_permute`
correctly remains a distinct point-to-point gap rather than being mislabeled
as all-to-all. CUDA frontend capture and exact multi-GPU NCCL proof remain
architecture-owned; no SM120 claim transfers from portable execution.

Cross-backend sync `AD-SOLVER-RESIDUAL-EVAL-2026-08-08` — **bounded x86/ROCm
pilot landed; CUDA follow-up required.** The shared IFT chain now has a
content-addressed Schedule→Tile physical contract, and counted-region treeverse
can execute checkpoint replay before a row becomes eligible. SM120 has no
consumer for the diagonal-sqrt pilot, no general iterative solver, and no
complete-backward packet. This PR changes no PTX package, selector, policy, or
device claim.

Cross-backend sync `AD-CORE-EFFECT-CONTROL-COLLECTIVE-2026-08-08` — **shared
Graph/Tile/portable-Target contracts available; CUDA follow-up required.** Compiler activity,
effects, `stop_gradient`, stochastic rejection, and fail-closed region
adjoints are target-independent. The four collectives now lower into one
content-addressed asynchronous Target queue and execute through the shared
runtime-adapter ABI. SM120 still needs exact multi-GPU NCCL execution and a
device packet. No PTX selector, native performance, or device evidence is
claimed or transferred.

Cross-backend sync `GRAPH-VERIFY-SIGNED-1-2026-08-08` — **shared legality
parity validated; no CUDA physical claim.** Graph and canonical-attention
integer bounds now consume signed `IntegerAttr` values, preventing MLIR 23
unsigned accessors from accepting negative schedules, seeds, cache windows, or
control bounds. Direct negative IR cases cover both dialects. No PTX ABI,
SM120 schedule, selector, package, or exact-device evidence changes.

Cross-backend sync `AD-TSOL-SPECTRAL-1-2026-08-08` — **shared Graph contract
available; CUDA follow-up required.** Compiler spectral identity and the
FFT/RFFT/DCT transpose rules are target-independent and CPU-oracle proven.
SM120 still needs its own Schedule→Tile/native compound-backward package and
exact-device evidence; no CUDA support or performance claim follows from the
x86/gfx1151 carrier.

Cross-backend sync `AD-TSOL-SPECTRAL-NATIVE-2026-08-09` — **SM120 follow-up
still required.** The bounded spectral-filter/convolution consumers and proof
land only for AVX-512 and gfx1151. No PTX image, schedule, correctness, or
performance evidence transfers to NVIDIA; its compound-backward path remains
fail-closed until an architecture-owned package lands.

Cross-backend sync `AD-CORE-LINEAR-1-2026-08-08` — **shared Graph-IR follow-up
available; no CUDA physical claim.** Compiler-owned linear transposition now
covers structural views, broadcast, and operand-wise matmul in both autodiff
passes with CPU numerical proof. SM120 backward packaging remains
architecture-owned; no CUDA image, selector, schedule, or device evidence is
transferred by this shared interface.

Cross-backend sync `COMPILER-DASHBOARD-PROOF-TRUTH-2026-08-08` — **SM90 and
SM120 proof separated; no CUDA physical change.** SM90 compile/artifact rows
are no longer added to SM120 runtime counts, and only exact-device statuses
close a hardware op×target grain. No package, selector, or device evidence is
changed.

Cross-backend sync `X86-BUILD-ARTIFACT-DISCOVERY-2026-08-08` — **shared
fail-closed selection assessed; no CUDA package changes.** The x86 runtime and
native packager now honor `TESSERA_BUILD_DIR` and reject a missing selected
tree rather than loading a stale default image. NVIDIA keeps its own CUDA tool
and image discovery, and inherits no AVX-512 implementation or timing result.

Cross-backend sync `STANDALONE-COVERAGE-TRUTH-2026-08-08` — **registry truth
adopted; no CUDA execution claim changes.** The standalone dashboard now
generates its counts, compiler-layer rollup, exact-target manifest summary,
and open queues from the live registries. It explicitly separates aggregate
best-available evidence from per-target support. SM120 still owns every CUDA
physical and benchmark follow-up in its manifest; x86 and gfx1151 TSOL or
Adafactor evidence does not transfer.

Cross-backend sync `TSOL-NATIVE-REAL-FFT-2026-08-08` — **shared artifact
follow-up required; no CUDA schedule transfers.** The target-neutral FFT
contract now binds logical/physical length, Hermitian layout, and packed-real
versus full-complex policy. Only x86 and gfx1151 own physical N/2 consumers and
evidence. SM120 must implement and measure its own real-transform package
before selecting the packed policy; it inherits no AVX-512 or RDNA schedule.

Cross-backend sync `ROCM-BUILD-ARTIFACT-DISCOVERY-2026-08-07` — **parity
validated; no CUDA physical change.** Shared compiler-test discovery now
accepts fail-closed `TESSERA_BUILD_DIR` selection while retaining explicit
`TESSERA_OPT` precedence. The migrated runtime-library and backend-tool users
are ROCm-owned; SM120 packages, schedules, evidence, and selectors are
unchanged.

Cross-backend sync `AUTODIFF-RELAXATION-1-2026-08-07` — **shared
Python-reference contract; CUDA physical follow-up required.** `sparsemax`,
`entmax15`, `soft_top_k`, `gumbel_softmax`, and `perturbed_argmax` now have
storage-preserving reference semantics and autodiff rules, but no CUDA lowering
or SM120 evidence. They remain explicitly reference-only until a CUDA-owned
physical package is selected and proven.

Cross-backend sync `MATH-PHYSICAL-2-2026-08-06` — **shared dtype contract
assessed; CUDA follow-up required.** Physical binary math packages now require
matching input storage dtypes. The Zen 5 scan selector and gfx1151 HIP module
cache are architecture-owned and transfer no PTX schedule, dtype, or
performance claim. NVIDIA must run the reduced-storage and difficult-domain
corpus on its canonical CUDA math packages before claiming parity.

Cross-backend sync `TSOL-CONTRACT-GENERALIZE-2026-08-06` — **shared semantic
contract adopted; physical consumer remains follow-up.** Bounded dynamic
dimensions, arbitrary axes, storage policy, and normalization are now explicit
before an exact TSOL specialization is emitted. Zen 5 and gfx1151 now consume
that wider contract, but their ABI and evidence do not transfer. NVIDIA still lacks the
prerequisite promoted FFT package, so Schedule→Tile lowering rejects CUDA
physical consumption and records no dtype, numerical, or performance claim.
The architecture-owned sequence remains: close canonical CUDA FFT, define an
SM120 workspace/residency ABI, implement the compound package, then gather
exact-device evidence.
Cross-backend sync `TPROF-MULTICLOCK-2026-08-06` — **shared evidence schema is
executable on ROCm/x86; CUDA adoption remains follow-up.** The ROCm plan now records each
clock independently and forbids fallback relabeling. CUDA should adopt the same
provenance, validity, calibration, instrumentation, and verdict fields for host
wall, CUDA events, an architecture-qualified device clock, and CUPTI activity.
HIP `wall_clock64()`, `rtg_hsa_dispatch`, ROCprofiler, and gfx1151 evidence do
not transfer to SM120. CUDA clock choice and CUPTI/SM120 promotion remain
architecture-owned exact-device work.
The native-evidence extension adds content-digested provider captures,
clean-versus-instrumented image/resource comparisons, and exact-machine event
maps. These are shared evidence concepts only: ROCprofiler, RTG, Linux perf,
IBS, gfx1151, and Zen 5 records transfer no CUDA result. NVIDIA must populate
the corresponding fields from CUPTI activity/metrics/PC sampling on SM120.

Cross-backend sync `TSOL-ROCM-E2E-1-2026-08-06` — **shared ODS vocabulary
adopted; CUDA physical execution remains follow-up.** The target-neutral
`schedule.spectral_program` and `tile.spectral_program_kernel` contract is
registered in production. NVIDIA still lacks a promoted canonical FFT package,
so it cannot consume the compound artifact or inherit ROCm/x86 evidence. A
future CUDA implementation must first close its FFT package gap, then bind its
own workspace/residency policy and SM120 device evidence.

Cross-backend sync `TSOL-GFX1151-FUSED-BATCH-2026-08-08` — **not applicable to
CUDA execution.** The content-addressed FFT vocabulary now carries gfx1151's
batched fused-LDS residency explicitly, but the HIP image dependency, AMD LDS
kernel, and WSL timing evidence establish no CUDA package or SM120 selector.
NVIDIA's architecture-owned FFT/TSOL follow-up is unchanged.

Cross-backend sync `TSOL-SPECTRAL-POLICY-2026-08-08` — **shared DCT and
streaming policy adopted; CUDA physical follow-up unchanged.** DCT-I/II/III/IV
now carry distinct API, autodiff, Graph, Schedule, and Tile identities. The
target-neutral causal chunked-STFT state binds its policy digest and overlap
lineage, while centred streaming fails closed pending explicit lookahead.
NVIDIA still lacks the prerequisite CUDA FFT/compound package and inherits no
x86/gfx1151 physical or performance evidence. The length-one convolution and
one-sample STFT/ISTFT physical boundary repairs transfer no CUDA implementation
claim.

Cross-backend sync `ROCM-MATH-EVIDENCE-2026-08-06` — **not applicable to
NVIDIA codegen.** Centered Welford and the scalar boundary fixes alter ROCm C++
generators plus shared host-side atan2 quadrant semantics; no PTX, CUDA ABI,
math mode, or NVIDIA capability changed. NVIDIA must evaluate the same domains
on its own canonical math/reduction lanes.

Cross-backend sync `ROCM-FFT-PREBUILT-2026-08-05` — **not applicable; NVIDIA
still has no promoted canonical FFT package.** The ROCm opaque plan ABI and HIP
allocation policy are not transferred. A future CUDA package must define and
measure its own artifact-bound plan/workspace contract on NVIDIA hardware.

Cross-backend sync `FFT-PERF-2-2026-08-05` — **not applicable to the unproven
CUDA lane; follow-up remains required.** The new cached Bluestein, Rader,
mixed-radix AVX-512 codelets, and rejected Bailey candidate are x86-owned.
They do not establish SM120 code generation or evidence. NVIDIA's existing
mixed-radix/Bluestein and exact-CUDA gaps remain unchanged.

Cross-backend sync `FFT-PERF-FOUNDATION-2026-08-05` — **follow-up required;
radix-17 source changed without CUDA evidence.** The shared planner now admits
radix 17 and the CUDA generic-stage private array was widened accordingly, but
this Ubuntu/gfx1151 host cannot compile or execute the SM120 lane. NVIDIA must
compile and compare radix-17 direct execution against its prior rejection path
on CUDA before claiming parity. The expanded Schedule→Tile FFT identity remains
outside NVIDIA until its Bluestein gap closes.

Cross-backend sync `E2E-REAL-FFT-2026-08-05` — **follow-up required; no support claim.**
ROCm corrected its public FFT authority to the proven Stockham/Bluestein
package and identified `schedule.fft`→launch Tile as the remaining shared
boundary. That shared content-addressed contract is now implemented and
consumed by x86/gfx1151, while NVIDIA remains deliberately outside its target
set. NVIDIA must join it only after its existing mixed-radix
hook gains Bluestein and passes exact CUDA/SM120 evidence; ROCm evidence does
not promote this lane.

Cross-backend sync `FFT-MIXED-RADIX-BLUESTEIN-2026-08-03` — **follow-up required — mixed-radix only, no Bluestein, unverified.**
Tessera's own FFT (Stockham, `TargetHooks/`) extends from powers of two to
every length: a generic radix-r stage for the odd small primes and Bluestein
for the rest. Shared contracts changed, so all four backends are affected:

* **Planning is now one implementation** (`TargetHooks/Common/FFTPlan.h`).
  CPU, AMD and NVIDIA each carried their own `while (n%4) ... while (n%2)`
  driver loop, and all three silently returned a HALF-FINISHED transform for
  any other N while reporting success. `LegalizeSpectral::pickRadixSequence`
  was a fourth copy, factoring over radices 7/5/3/4/2 and pushing a residual
  prime as a "stage" of that radix -- a stage nothing could execute.
* **Compiler routing was wrong independently of the kernels.**
  `LowerToTargetIR::stageSymbolFor` mapped every radix other than 4 to
  `ts_stockham_r2_*`, so a static N = 12 = 4x3 emitted a radix-2 call for a
  radix-3 stage. The runtime driver was correct; the compiler path was not, and
  direct driver tests could not see the difference.
* **New C ABI surface:** `ts_stockham_rn_<backend>(in, out, N, L, r, sign)`
  (note the extra radix argument, which r4/r2 do not take), plus
  `tessera.target_ir.stage_radices` carrying it, and a
  `tessera.target_ir.bluestein` marker routing those lengths to the driver.

NVIDIA gets the shared plan and the generic radix-r stage, so every
mixed-radix length is routed and emitted correctly. It does NOT get Bluestein.

There is no CUDA toolchain on the development box, so ~60 lines of device code
written for it could not be compiled, let alone checked against a reference.
Shipping unverifiable device code is the same unproven-claim pattern the silent
truncation was an instance of, so the driver DECLINES instead:
`ts_fft_supported_nvidia(N)` answers the question and the driver returns
without writing `d_out` rather than truncating.

**Nothing in this change has been compiled for NVIDIA.** The generic radix
kernel is a mechanical mirror of the gfx1151-verified AMD one, which lowers but
does not remove the risk. First task on the CUDA box: compile the `.cu`, run
the mixed-radix sizes against numpy, then implement and verify Bluestein.
Registering the lane as a `spectral_fft` arbiter candidate is a separate,
still-open item.


Cross-backend sync `SHAPE-RULE-REGISTRY-2026-08-03` — **follow-up required - scale operands changed, and NVIDIA has no FFT lane.**
PR #493 closed the Graph IR shape-rule registry: **303 declared / 6 deliberately
undeclared / 0 unexamined**, with the `MAX_UNCLASSIFIED` ratchet dropped 106 -> 0.
Shared contracts changed; all four backends are affected equally at the
reference level:

* **Result contracts.** Multi-result ops now emit every SSA result
  (`kv_cache.read -> (K, V)`, `top_k`, `qr`/`svd`/`lu`/`nonzero`), and tuple
  destructuring (`v, i = ...`) lowers. The emitter previously called the
  single-result `_infer_result_type`, so a declared multi-result contract
  stopped at Graph IR.
* **Stateful handles.** `!tessera.kv_cache` is now reachable from Python; the
  emitter had been printing `tensor<*x?>` for a type the ODS has always
  declared.
* **dtype policy.** An integer input to a float-producing op promotes to the
  declared `COMPUTE_FLOAT_DTYPE` (fp32) instead of NumPy's width-derived float
  (`cos(int8) -> f16`, `cos(int32) -> f64`); index/count results use a declared
  `INDEX_DTYPE`; complex is a LOGICAL dtype carried in an interleaved real pair,
  not a storage format.
* **Diagnostics.** The whole `GRAPH_IR_*` family (17 codes) is registered - the
  drift gate's scanner did not know the prefix, so it reported green while the
  family accumulated unregistered.

**This is the Python reference lane, not generated device code.** The NVFP4/MX matmul lane carried `scale_a`/`scale_b` as ATTRIBUTES holding SSA
names; they are now real operands 2 and 3. This corrects Graph IR toward what
NVIDIA already declared everywhere else: the ABI is
`tessera.nvidia.nvfp4.a_b_scale_a_scale_b_d_m_n_k.v1` and the kernel is
`tile.matmul_kernel %a, %b, %scale_a, %scale_b, %d`, so Tile IR modelled the
scales as operands and only Graph IR demoted them. `nvidia_native.py`'s
packagers and its `requests_`/`supports_` predicates were updated; `bias` was
never affected (x86, ROCm and the unscaled NVIDIA lanes all append it as an
operand). Parity validated at the Python packaging level; **device evidence is
missing** - no exact-device run confirms the packaged buffer order end-to-end on
sm_120.

Recorded plainly: **complex FFT is REJECTED on nvidia_sm120**, because no NVIDIA
target declares an `fft` capability entry and zero NVIDIA source files mention
FFT at all. An earlier cut synthesised capability entries for absent ops and made
the sm_90 dashboard assert `fp8_e4m3` and `int8` FFT kernels - nine
`artifact_only` rows for a backend with no FFT. That was backed out. A target
with no `fft` entry is stating it has no `fft`.


Cross-backend sync `SUBBYTE-STORAGE-PATH-2026-08-03` — **follow-up required; NVIDIA is the target where this matters most.**
The quantize family is now correctly declared as MULTI-RESULT `(codes, scale)`,
and `quantize_nvfp4` has its own rule because its scale is per-BLOCK (one per 16
elements along the last axis) rather than per-tensor — the micro-scaled form
Blackwell implements. A shared per-tensor rule would have misstated the format
for exactly the architecture that motivates it.
**The open gap is a backend-path one, not a shape-rule one.** The reference
returns codes as **f32** — fake-quant. `fp8_e4m3`, `fp8_e5m2`, `fp4_e2m1` and
`nvfp4` are canonical dtypes and the Graph IR type system can carry them, so
nothing in the compiler prevents real sub-byte storage; no lowering produces it.
NVIDIA owns this first: consumer/datacenter Blackwell has native FP8 and NVFP4,
so it is the one target where "the backend upcasts anyway" is NOT the answer.
The Target IR must carry fp8/fp4 as real storage into the mma path.

Cross-backend sync `REDUCED-PRECISION-COMPUTE-2026-08-03` — **follow-up required, reference-level only.**
The shared reduced-precision policy changed: ops whose declared rule preserves
storage dtype now upcast reduced-precision inputs to f32, compute, and store
back. This repaired six ops whose INTERNAL arithmetic left fp16 range while
their answers fit easily — including `flash_attn` and `mla_decode`, both hot
SM120 paths, which previously returned float64 for f32 AND bf16 inputs.
**This is the Python reference lane, not generated CUDA.** The same hazard
class applies to NVIDIA kernels — a QK^T contraction overflowing fp16 before the
softmax rescales — and nothing here proves the generated kernels handle it.
NVIDIA owns verifying the accumulate-in-f32 contract on device; the reference
now states what the kernels must match.

Cross-backend sync `TILE-MMA-DATA-OPERANDS-2026-08-03` — **parity validated, and the prior NOT-VALIDATED status is now CLOSED.**
`MMAOp::verify()` now counts DATA operands, so the typed `tile.mma` fragment
form and the warp-spec `!tile.async_token` edge can coexist. NVIDIA needed the
same correction and had not received it: `NVIDIALowering.cpp` compared raw
`op->getNumOperands()` against 3/5 and then indexed raw operands, so the exact
typed-plus-token form this change unblocks would have hit `emitError` +
`signalPassFailure` during SM120 lowering — and operand 3 would have been the
token rather than an NVFP4 scale. Both the count and the indexing now use
`tessera::tile::dataOperands`.
**Verified by building it**, not by inference: ROCm and NVIDIA cannot both
register in one `tessera-opt`, so a second tree (`build-nvidia`,
`-DTESSERA_BUILD_NVIDIA_BACKEND=ON`) was configured and built. With
`tessera_nvidia` actually registered, the W0.9 parse/verify gate **passes for
sm90 / sm100 / sm120, plain and probe-annotated** — it had been SKIPPING every
run since PR #490. The contract test now discovers either build, so NVIDIA is
no longer silently unmeasured.

Cross-backend sync `TARGET-IR-CONFORMANCE-2026-08-02` — **NVIDIA host-free
conformance validated (2026-08-21); exact-device execution remains separate.**
W0.9 added a real parse + dialect-load + verifier gate over every Target-IR
emitter, and it found that no Python-emitted Target IR was valid MLIR
(undialect-prefixed module attributes, an invented `<dialect>.func` container,
ops emitted with signatures their ODS rejects, and several undeclared op names).
Those defects were fixed and verified for `cpu`, `x86`, `rocm`, and `apple`.
The test harness now discovers this workspace's NVIDIA-enabled
`build-nvidia-cuda/` compiler rather than silently skipping it. Its real MLIR
parse/load/ODS lane passes for sm90, sm100, and sm120, including probe-annotated
multi-op IR and committed NVIDIA goldens. `tessera_nvidia.profiler_probe` and
the Python wrapper/call surface are registered. The former unrestricted
`AnyType` envelope is now a Target-value union: tensor/memref data, scalar or
vector fragments, and LLVM pointer/aggregate ABI values. A negative NVIDIA lit
fixture proves `!tile.async_token` cannot enter an MMA Target IR op. Validation:
NVIDIA lit 48/48 and `test_target_ir_contract.py` 28 passed; the 10 skips are
only Apple/ROCm dialects absent from this build. This is compiler conformance,
not SM120 execution evidence.

Cross-backend sync `CORE-ATTENTION-TRAINING-X86-2026-07-30` — **follow-up
required, no NVIDIA contract change.** X86 adopted the shared rank-4 forward
and tensor backward loops and closed its optimizer adjoints. No Zen 5 ABI,
schedule, LSE policy, or timing transfers to SM120. NVIDIA retains direct
shared-loop forward/backward consumption, architecture-owned LSE selection,
and its remaining backward materializers.

## NVIDIA-SPINE-1: make the completed SM120 package the canonical default

Cross-backend sync `EXECUTION-SPINE-2026-07-29` — **landing.** The SM120
Graph/Schedule/Tile → NVVM/PTX image and launch-descriptor path was already
complete under NVIDIA-E2E-1/-2, but `canonical_compile()` did not select it by
default. NVIDIA now owns one `native_package_kind` / `package_native` entry
point; the shared driver no longer duplicates the vendor family-dispatch table,
and eligible static modules auto-promote when the complete SM120 toolchain is
available. Explicit `package_native = false` remains a stable opt-out.

Host-free focused validation covers canonical selection, typed package
production, runtime projection, and the native artifact contract. Existing
RTX 5070 Ti evidence remains the exact-device proof for the unchanged lowering,
PTX, ABI, and schedules; this slice changes selection authority, not emitted
code. The ROCm, Apple, and x86 selector reconciliations subsequently landed
under the same synchronization key. Apple keeps its explicit Value Target-IR
compatibility/probe route outside descriptor promotion. X86 has since separated canonical MLIR/native target
`x86` from its `x86_c` source candidate. NVIDIA already has that separation;
no PTX, ABI, schedule, or exact-device evidence changes.

APPLE-RASTER-1 subsequently consumed the shared map in emitted MSL and retained
row-major after mixed Apple7 timing. This is Apple-specific evidence, not an
NVIDIA selector or measurement result.

## NVIDIA-CALIB-1: supply the sm_120 corpus to the hardware-free score calibration

Cross-backend sync `COSTMODEL-CALIB-2026-07-29` — **superseded by the terminal
ROCm home-architecture rejection.** No NVIDIA arbiter-score adoption work remains.

The independent Zen 5 hierarchical T1 packet now also rejects T1 for latency
ranking (median rho -0.4062, 0/3 winner matches). Its x86 cache hierarchy,
bandwidths, candidates, and verdict do not transfer to SM120; NVIDIA's own
descriptor-complete correlation packet remains the only valid local decision.

**Correction that created this item.** `APPLE_AUDIT.md` originally scoped this
calibration to Apple alone, on the stated grounds that ROCm and NVIDIA kernels
"cannot be measured". That was false for NVIDIA: this backend already has a
committed, **consumed**, device-keyed `nvidia:sm_120` autotune corpus covering
64/256/512/1024/2048 square buckets plus fused GEMM and causal attention,
generated by `benchmarks/nvidia/record_autotune_corpus.py`. Excluding it would
have discarded the deepest per-shape latency evidence in the fleet.

**Historical calibration result and current subject.** The original two
static, device-free scores from
[`../../compiler/AMD_KERNEL_COMPILER_SURVEY.md`](../../compiler/AMD_KERNEL_COMPILER_SURVEY.md)
§3.7–3.8 — a step-distance locality histogram and an N-way bank-conflict
analyzer — were not retained after the step-distance line failed on its ROCm
home architecture. Do not tune or transplant that score. The current subject is
the shared T1 GEMM model: symbolic tile identities, capacity-bounded LRU reuse,
cache-derived DRAM traffic, and explicit target compute/bandwidth inputs
([`TILESIGHT_ASSESSMENT.md`](../../compiler/TILESIGHT_ASSESSMENT.md) §3).

**NVIDIA's role: shape depth.** The committed corpus already varies the shape
axis within GEMM and attention at fixed op kind, which is exactly the axis Apple
cannot supply and the one a cache/reuse score most needs to be tested against —
locality changes with shape at constant op. Apple supplies op breadth
(`APPLE-CALIB-1`); ROCm supplies a second, independent architecture
(`ROCM-CALIB-1`). Fitting on any one of the three reproduces the single-arch
overfit the assessment records for NeuSight.

**Note on translating the retired bank metric.** It was derived for AMD LDS with a known
bank count and a wave64 4-phase access pattern (survey §5.1). CUDA shared memory
is 32-bank and warp-synchronous; the *method* transfers but every constant must
be re-derived for sm_120 before a conflict number here means anything. Do not
port AMD constants.

**Fleet outcome (2026-07-29).** ROCM-CALIB-1 tested the metric where it
originated and reproduced 0/6 committed gfx1151 winners (median rho -0.1381, 0%
positive). The agreed home-architecture failure rule ends this latency-ranking
line without coefficient or target retuning. NVIDIA therefore owes no sm_120
promotion analysis for this score; CUDA-specific bank diagnostics remain a
separate future model and must not inherit AMD constants.

**Live-input update (2026-07-30).** The owning RTX 5070 Ti now reports through
CUDA 13.3 `cudaDeviceGetAttribute`: 70 SMs, 2.497 GHz core, 14.001 GHz memory,
a 256-bit bus, and `cudaDevAttrL2CacheSize = 50,331,648` bytes (48 MiB). The
registry consumes the measured L2 capacity; the observed memory clock and bus
corroborate the existing 896 GB/s peak-bandwidth derivation.

**Counter rerun (2026-07-30).** Nsight Compute 2026.2.1 is now installed on
the owning WSL host. A resource-only capture of the existing 512³ f16
production-route launcher retained `/tmp/nvidia-calib-sm120-2026-07-30.ncu-rep`
(`sha256:e0d0a1e2650dbfc1da921f72ebf0f36c27a1ec1165b278c51905336d02d6c79d`).
For the two Tile implementations, `tessera_tile_matmul_direct_f16` reported
97.65% L2-sector hit rate and 1,273,856 DRAM bytes, while
`tessera_tile_matmul_shared_f16` reported 91.24% and 1,101,312 bytes. This
confirms that CUDA counter collection works and that the two schedules have
distinct cache/traffic behavior. The profiler's replay duration is explicitly
not timing evidence, and these are different schedule shapes rather than a
controlled candidate-rank packet; neither value changes selection or supplies
a T1 correlation verdict.

**Correlation verdict: not identifiable from the committed corpus.** The
corpus retains latency keyed by named implementation but does not serialize the
candidate Tile M/N/K/raster descriptor needed to replay T1 for each competitor.
It therefore cannot supply an honest within-shape rank correlation, even with
the now-measured cache input. Retain T1 solely as a legal pruning estimator;
measured latency remains selection authority. A future CUDA corpus revision
must persist candidate schedule descriptors and profiler counter availability
before this item can produce a correlation verdict. Do not fabricate a rank or
revive the rejected AMD step-distance metric.

## NVIDIA-RASTER-1: consume the shared block-rasterization contract

Cross-backend sync `RASTER-CONTRACT-2026-07-28` — **follow-up required, owning
host NR2 Pro (RTX 5070 Ti, sm_120).**

**Shared contract changed.** Schedule IR gained two attrs and two knobs —
`raster_order` (`row_major` | `column_major` | `grouped_m` | `grouped_n`) and
`raster_group` — carried on `schedule.tile` and `schedule.knob`, mirrored by
`TuningConfig.raster_order`/`raster_group` and persisted in the SQLite tuning
cache. The order is a *permutation of block ids onto the tile grid*, defined in
the arch-neutral `compiler/tile_rasterization.py` with a `remap()` reference, an
`emit_c()` snippet valid identically under CUDA and HIP, and `is_bijection()` as
a total hardware-free oracle. Rationale and the 35%→72% L2 figure that motivated
it: [`compiler/TILESIGHT_ASSESSMENT.md`](../../compiler/TILESIGHT_ASSESSMENT.md)
§3.2.

**Implementation landed (2026-07-30); selection remains open.** The SM120
`mma.sync` fused-GEMM and gated-matmul emitters now consume `raster_order` and
`raster_group`. `row_major` retains their established 2-D launch and direct
`blockIdx.x` / `blockIdx.y` coordinate arithmetic. Non-default orders flatten
the same block count to one dimension and inject the shared `emit_c()` mapping,
including ragged final panels. The compiled-artifact cache key includes both
knobs, so a swizzled binary can never alias a row-major one. Focused host-free
tests cover source selection and the shared permutation oracle; exact RTX 5070
Ti execute-and-compare covered grouped-M fused GEMM and grouped-N gated matmul
on ragged dimensions. No selector change is implied.

**Why it did not land in the contract PR.** Changing a hardware-verified
`mma.sync` kernel without sm_120 silicon to measure the result would be an
unverified edit to a proven path for no demonstrable gain. `row_major` is the
default and reproduces the existing index arithmetic exactly, so today's emitted
code is byte-identical.

**Validation performed (host-free).** `tests/unit/test_tile_rasterization.py`
proves the permutation property over ragged grids and **compiles the emitted C
with host clang, running it against the Python reference for every block id**
under `-Wall -Wshadow -Werror`. That covers the arithmetic and the emission's
scoping, on any host.

**Remaining exact-device evidence.** Whether a swizzle moves sm_120 latency, and
at which `raster_group`, for the GEMM shape buckets in the perf ratchet. This
needs a committed repeated-median CUDA-event matrix plus `ncu` L2 hit-rate
deltas on the NR2 Pro. Until then the axis is **carried, not selected**. T1 can
score the order symbolically, but it has not earned an sm_120 raster retain
verdict and cannot promote a choice.

**SuperBear timing packet (2026-08-22): row-major retained.** Seven repeated
CUDA-event medians across square, rectangular, and ragged buckets swept
row-major, column-major, grouped-M, and grouped-N at groups 2/4/8. The three
per-shape winners disagreed (column-major, grouped-N/2, grouped-N/2) and
improved over row-major by 0.57%, 3.42%, and 2.54%; only one bucket crossed the
recorded 3% promotion floor. The committed packet is
[`nvidia_sm120_superbear_raster_2026_08_22.json`](../../../../benchmarks/baselines/nvidia_sm120_superbear_raster_2026_08_22.json).
Nsight Compute 2026.2.1 captured the exact 512x512x512 kernel: row-major was
96.65% L2 / 1,305,600 DRAM bytes, column-major 96.81% / 1,298,944,
grouped-M/8 94.08% / 1,054,720, and grouped-N/8 95.85% / 1,054,976. The lower
grouped traffic did not produce a stable all-shape timing winner, so the
selector remains row-major. This closes SuperBear's measured timing + counter
decision but leaves the NR2 Pro packet open under the owning-host rule.

## NVIDIA-AOT-1: decide whether NVRTC needs a precompiled peer — complete

Cross-backend sync `APPLE-AOT-METALLIB-2026-07-28` — **follow-up required**.
Apple added `apple_gpu_air`, a precompiled-artifact lane behind the shared
`register_compiler(target, compile_fn)` seam, measured against its compile-on-
launch lane (cold pipeline creation 29.7 ms -> 15.2 ms, ~1.95x; host-wall
timing on Apple M1 Max, not device-event evidence). NVIDIA is the backend
closest to Apple's position, not a distant one: its device code is NVRTC-
compiled at load (`nvrtc_jit.cpp`; runtime.py describes the mma.sync lane as
NVRTC-compiled for the device arch) and `runtime.py` has no cubin/fatbin
precompiled lane. So the AOT-vs-JIT question is genuinely open here. An earlier
version of this note said CUDA had 'nothing to catch up on' because
`emit/nvidia_cuda.py` contains no nvrtc reference — that was inferred from
absence of evidence in one file and is withdrawn. Follow-up: decide whether
SM120 wants a precompiled artifact lane, and if it is measured, reuse
benchmarks/apple_gpu/benchmark_aot_vs_jit.py *with its cache control* (a never-
before-compiled kernel per sample) — the driver's own cache is what made the
first Apple number 13x too good. No shared IR, ABI, dtype/op registration, or
numerical contract changed.

**Decision (2026-07-30): a precompiled peer is warranted.** The exact SM120
probe [`benchmark_aot_vs_jit.cu`](../../../../benchmarks/nvidia/benchmark_aot_vs_jit.cu)
uses a unique CUDA source and entry symbol for every sample, so neither NVRTC
nor the driver can serve a prior module. Both lanes load, launch, and verify the
same device result. On the RTX 5070 Ti (CUDA 13.3, driver 610.62), seven-sample
medians were **18.266 ms** for NVRTC compile + module load + launch and
**0.867 ms** for a precompiled cubin load + launch: **17.399 ms** saved per cold
request. Offline cubin construction was **173.004 ms**, amortizing after about
**10 cold launches**. The retained packet is
[`nvidia_sm120_aot_vs_jit_2026_07_30.json`](../../../../benchmarks/baselines/nvidia_sm120_aot_vs_jit_2026_07_30.json).

This closes the decision, not productization: a follow-on must add a versioned
SM120 cubin/fatbin artifact to the native package/runtime seam, preserve the
current NVRTC fallback for unsupported or stale artifacts, and execute-compare
the production matmul ABI before any selector promotion.

**Productization landed (2026-07-30).** `libtessera_nvidia_gemm.so` now ships
the first package-owned precompiled peer: the versioned
`tessera_nvidia_mma_f16_sm120_v1.cubin` beside the runtime image. Its canonical
`.cu` input is also generated into the library as the NVRTC fallback source, so
the AOT and JIT lanes have one kernel body, entry symbol, and physical
`A:u16[M,K], B:u16[K,N], D:f32[M,N], M,N,K` ABI. The loader admits the cubin
only on exact CC 12.0 and CUDA-driver >= 13000; absent, corrupt, incompatible,
or explicitly disabled artifacts use NVRTC. `TESSERA_NVIDIA_AOT_MODE=require`
is a strict deployment check and never falls back silently. Fresh-process RTX
5070 Ti execution proves forced-AOT, forced-NVRTC, and missing-artifact
fallback are numerically equivalent on ragged `17x31x9` f16 GEMM; the version
and canonical-source SHA are queried from the shipped C ABI. This is NVIDIA
runtime packaging only: no shared IR/ABI, selector, Apple, or ROCm contract
changed, so sibling backend plan changes are not applicable.

**Product hardening (2026-08-22).** The native package now carries both
`tessera_nvidia_mma_f16_sm120_v1.fatbin` and `.cubin` plus a generated manifest.
Both images embed artifact/ABI versions and the canonical-source SHA. Admission
reads and compares those globals before resolving `gemm`; a loadable but stale
image therefore reports `nvrtc_stale` and takes NVRTC in auto mode, while
`require` remains fail-closed. Format and source identity are part of the
runtime cache key. Exact SuperBear tests execute forced fatbin and cubin,
distinguish their keys, mutate an embedded SHA in an otherwise loadable image,
and prove the stale route numerically equals the fallback.

Cross-backend sync `TESSERA-OPT-CAPABILITY-SKIP-2026-07-27` moves the last 43
self-resolving test files onto the shared `tests/_support/compiler_tool.py`
driver contract, adds `--pass-pipeline=` inner-pass capability checking, and
folds `CompilerToolchain` onto one resolver and one capability check. NVIDIA is
**not applicable** for an architecture-specific reason: the NVIDIA lit and
compiler lanes drive the *separate* `tessera-nvidia-opt` binary through
`TESSERA_NVIDIA_OPT` / `CompilerToolchain.require_nvidia_opt` (the `%tnv`
substitution), which this resolver does not govern and which this change leaves
byte-for-byte untouched — `require_nvidia_opt` keeps its own `_tool_path`
lookup and its own skip. No CUDA registration, PTX or SM120 schedule, runtime
ABI, selector, or device evidence changed, and **no exact-device evidence is
claimed or required**. Should the NVIDIA lane later want the same
build-capability skip behaviour for `tessera-nvidia-opt`, that is a separate
follow-up owned by this plan, not a debt created here.

Cross-backend sync `ROCM-BF16-ATTENTION-2026-07-27` validates that the shared
BF16 attention carrier and canonical forward/backward loop contracts can be
consumed by a second physical backend. ROCm now has exact ragged-GQA
bias+softcap+causal-window+dropout forward proof and deterministic five-entry
backward proof on gfx1151, with dedicated resident BF16 timing ratchets. This
is parity validation at the shared semantic boundary only. AMD BF16 WMMA,
LDS scheduling, HSACO packaging, HIP workspace and launch ABI, numerical
evidence, and timing do not transfer to CUDA; NVIDIA retains its own SM120
BF16 package and exact-device evidence requirements.

Cross-backend sync `TESSERA-OPT-BUILD-CAPABILITY-2026-07-27` is **closed**.
The shared lit resolver now accepts `TESSERA_OPT_BIN`, `TESSERA_OPT_PATH`, and
`TESSERA_OPT_CPP` after the canonical `TESSERA_OPT` override, and the validation
script forwards its selected binary through that contract. Exact gfx1151
verification proves the full ROCm driver, legitimate lean ROCm artifact
driver, conflict rejection, both named streaming-attention fixtures, the
seven-fixture filter, and the complete 50-test ROCm backend lit suite. This is
shared test/build infrastructure only; no CUDA registration, PTX schedule,
runtime ABI, device evidence, or selector changes.

Cross-backend sync `LSE-CHECKPOINT-CONTRACT-2026-07-27` lands the real shared
checkpoint vocabulary: explicit memref source/destination, SSA row offset,
identity, memory space, lifetime scope, cache policy, and read/write effects.
Default forward lowering no longer emits a destination-less save. ROCm
validates saved versus recompute on gfx1151 and retains the provisional
128+ policy, but the newer dual-clock packet is explicitly fail-closed on WSL:
HIP events are positive yet non-transferable, and FP16 at 256 is not a stable
saved winner. Bare-metal gfx1151 confirmation remains required. NVIDIA is
**follow-up required**: consume the same shared contract, measure its own CUDA
forward-store/backward-load package, and retain or replace its zero-workspace
policy using exact SM120 evidence. AMD WMMA, HSACO size, threshold, and WSL
host-wall results do not transfer.

Cross-backend sync
`ROCM-ATTENTION-SHARED-BACKWARD-CONSUMER-2026-07-26` makes ROCm gfx1151 the
first direct physical consumer of the shared tensor-valued attention backward
phase loops. NVIDIA remains **follow-up required** to validate the same
dQ/split-dK/dV/fixed-reduction contract and map it to a CUDA-owned package.
The AMD WMMA schedule, five-entry HSACO, HIP launch workspace, gradient
evidence, and host-wall timing do not transfer. No shared IR or NVIDIA
capability state changed in this ROCm-owned closure.

Cross-backend sync `CORE-ATTENTION-TENSOR-LOOPS-MODIFIERS-2026-07-26`
materializes the deterministic split/reduced backward contract as tensor-valued
shared `scf.for` bodies with explicit dQ ownership, split dK/dV workspace
tensors, and ascending reduction. Registered shared score-bias and softcap
operations now preserve `softcap(scale*QK^T + bias)` inside the forward
KV-block recurrence, including rank-4 per-head bias. NVIDIA is **follow-up
required** to consume these phase operations through its SM120 package and
direct forward schedule. AMD HIP ABI code, HSACO, exact-device gradients, and
resident timing do not transfer.

Cross-backend sync `CORE-ATTENTION-BACKWARD-CONTRACT-2026-07-26` adds verified
split count, launch-owned workspace, block-loop metadata, ascending reduction
order, and canonical `softcap(scale*QK^T + bias)` semantics to the shared
carrier/oracle. NVIDIA is **follow-up required** to consume this form through
its SM120 schedule and validate dropout replay; AMD code and evidence do not
transfer.

## Cross-backend sync `E2E-REAL-5C-STATE-LINEAGE-2026-08-05`

The shared training spine now defines content-addressed logical-buffer lineage,
mutation identity, and typed Schedule→Tile contracts for Lion VJP,
factored/full Adafactor VJP, and sequence-mixer backward. **NVIDIA outcome:
follow-up required.** Existing SM120 packages remain NVIDIA-owned and do not
yet consume these exact artifacts. CUDA evidence and CUDA-owned buffer bindings
remain required.

Cross-backend sync `ROCM-E2E-ATTENTION-BACKWARD-2026-07-26` is not applicable
to NVIDIA physical execution. It adds a ROCm-owned five-entry HSACO and
gfx1151 split/reduced launch workspace without changing the shared launch
descriptor schema or canonical backward loop. AMD WMMA kernels, workspace
topology, exact-device gradients, timings, and selector state do not transfer.

The ROCm optimized-attention feature follow-up under
`ROCM-E2E-ATTENTION-CARRIERS-2026-07-26` adds AMD-only deterministic dropout
replay and combined bias+softcap consumption to the gfx1151 WMMA schedule,
plus a host-wall resident performance ratchet. It changes no shared carrier,
ABI, NVIDIA Target IR, CUDA schedule, capability, or selector. NVIDIA parity
at the semantic carrier remains validated independently; AMD counter code,
HSACO evidence, and WSL timing do not transfer.

Cross-backend sync `SSA-STATEFUL-TRANSPORT-2026-07-26` removes the last active
shared and ROCm `#tile.buffer_ref` compatibility readers after their fixtures
migrate to `!tile.buffer`; the deprecated attribute is parser-only. NVIDIA was
already SSA-only, so its SMEM/TMEM schedule and evidence are unchanged. The
shared ReplaySSM lifecycle schema now keys Apple and ROCm resident ABIs while
preserving session-private ring ownership, flush/rollback, ordered submission,
and drain-before-release. MoE metadata now owns launch-lifetime workspace and
can bind a canonical NCCL/RCCL rank/device fingerprint. NVIDIA consumes the
same local descriptor as before; no CUDA schedule, selector, or timing changes.

Cross-backend sync `ROCM-E2E-ATTENTION-CARRIERS-2026-07-26` lands an
AMD-owned consumer, native HSACO package, descriptor, and exact gfx1151 proof
for the already-shared `tile.attention_kernel` contract, plus a direct
correctness consumer for `tile.attention_backward_kernel`. NVIDIA parity at the
shared semantic carrier remains validated by its existing SM120 forward and
backward packages. ROCm's wave32 WMMA descriptor, LDS allocation, HIP ABI,
resource counts, timings, selector boundary, and direct scalar recurrence do
not transfer to CUDA. The ROCm v2 benchmark's operation-total and resident
synchronized HIP host-wall domains do not replace CUDA-event or CUDA
end-to-end evidence. No NVIDIA plan state or exact-device claim changes.

Cross-backend sync `ROCM-SSA-LDS-PIPELINE-2026-07-26` lands the AMD consumer of
the already-shared `!tile.buffer`, `!tile.async_token`, and
`!tile.pipeline_state` ownership vocabulary. It changes no shared operation,
type, verifier, ABI, or NVIDIA lowering. NVIDIA parity is therefore validated
at the existing SSA contract: WarpSpecialization continues to own SMEM/TMEM,
TMA/mbarrier, and architecture-specific pipeline mechanics. AMD LDS layouts,
waitcnt/s_barrier semantics, gfx1151 evidence, compiler timings, and selectors
do not transfer to CUDA or SM120; no NVIDIA follow-up is required.

Cross-backend sync `PACKED-LEGALIZE-CAPABILITY-2026-07-26` expands terminal
storage legalization without making sub-byte storage global. For `nvidia_sm120`,
the pass now proves the complete operation-specific consumer before stamping a
physical pack: packed load to ordinary store (explicit unpack/format
conversion), matching unscaled packed-load/store round trips, and packed
matmul whose A/B MMA descriptor agrees with the logical storage format.
Orphan or mixed-use packed loads, descriptor disagreement, arbitrary
operations, and the public shape-preserving Graph quantize/dequantize ABI stay
logical. The standalone empty-target transform remains available for explicit
IR inspection; named pipelines use the capability decision. Apple and ROCm
retain architecture-owned physical schedules; their evidence is not inferred
from SM120. The deprecated `#tile.buffer_ref` attribute is parser-only and no
active pass consumes it. No selector or timing disposition changes.

Cross-backend sync `CORE-STREAMING-ATTN-2026-07-26` replaces the shared
rank-2 FlashAttention whole-KV lowering with an explicit KV-block `scf.for`.
The loop carries the FP32 output accumulator, running maximum, normalization
sum, producer and consumer `!tile.pipeline_state` values, and an absolute
boundary offset. Each block consumes K and V through typed async tokens; the
online update now takes V explicitly, while causal/window/ragged masking and
counter-based dropout consume the loop offset rather than replaying block zero.
NVIDIA TMA descriptor hoisting traces each block slice to its kernel argument
and retains typed coordinates plus logical source extents, enabling
out-of-bounds zero fill for the ragged tail. WarpSpecialization no longer emits
name-based `#tile.buffer_ref` or annotation-only `#tile.pipeline_state`
metadata, and Schedule→Tile consumes structured per-operand `#tile.layout`
directly. The SM90 structural pipeline is lit-green. Follow-up sync
`CORE-STREAMING-ATTN-RANK4-ROCM-2026-07-26` adds shared rank-4 batch/head
distribution and proves a direct ROCm consumer. A direct NVIDIA
Target-IR/runtime consumer of the shared loop remains open; gfx1151 LDS/WMMA
schedules, resources, wall timing, and selector evidence do not transfer.
Existing launch-level attention images and selectors are unchanged.

Cross-backend sync `CORE-GEMM-KLOOP-2026-07-25` is **landing**, owned by
NVIDIA under the `NVIDIA-E2E-2` continuation. The shared compiler now forms a
target-neutral M/N/K `scf.for` nest with FP32/INT32 loop-carried accumulation,
zero-pad ragged guards, structured copy layouts, asynchronous SSA
dependencies, and threaded `!tile.pipeline_state`. The SM120 launch-level
FP16/BF16/TF32 images serialize the same `tile_m/tile_n/tile_k` contract and
reject descriptor disagreement before NVIDIA materialization. Two exact RTX
5070 Ti SM120 runs each pass all 12 FP16/BF16/TF32 square, rectangular,
ragged-K, and fully fragment-misaligned rows with FP32 accumulation,
matmul→bias→activation→residual ordering retained by the shared epilogue
contract, warm image/descriptor identity, and numerical comparison. The
checked-in 12-row repeated-median packet discards the first complete call,
amortizes 1,000 resident launches and 50 complete calls, and retains 31
interleaved observations per cohort in both timing domains. Every row meets the
4% two-run WSL gate (maximum device-event delta 1.39%, maximum end-to-end delta
3.47%); shared FP16/BF16 uses 42 registers and 10 active blocks/SM, direct TF32
uses 38 registers and 24 active blocks/SM, and all rows retain zero local
memory and zero spills. INT8 and packed formats remain follow-on after the
ordinary loop is stable. WSL timing is selector-ineligible and no selector
changes in this slice.

Cross-backend sync `ROCM-CORE-GEMM-KLOOP-2026-07-27` is **parity validated**
for NVIDIA. The only shared edit preserves the existing canonical
ragged-zero-fill guarantee across `tessera.matmul` → `tile.mma`; NVIDIA's
already-proven SM120 consumer and twelve-row packet are unchanged. AMD LDS,
wait/barrier, WMMA, HSACO resource, and gfx1151 wall-clock evidence do not
transfer. No NVIDIA route, capability, execution state, or selector changes.

Cross-backend sync `COMPILER-LIT-BACKEND-GATING-2026-07-24`: retired eleven
never-runnable CUDA13 pseudo-IR fixtures whose undefined `tessera_opt_built`
feature masked stale CLI options and unregistered operations. Core named
pipeline aliases now run in the ordinary LLVM lit lane; typed
Tile→NVIDIA→NVVM contracts remain owned by the NVIDIA backend lit suite.
`validate_nvcc_compile.py` now labels and runs its handwritten instruction
catalog strictly as a CUDA-toolchain probe, not evidence of Tessera emission.
The two integrated core+NVIDIA control-flow fixtures retain their precise
backend gate and still require the CUDA-enabled build on the NVIDIA host.

Cross-backend sync `COMPILER-PYTEST-PLATFORM-SKIPS-2026-07-24`: shared
compiler-owner markers now report foreign compiler proofs as skipped with the
required Apple, CUDA, ROCm, X86, or AVX512 system and a per-system count. This
is test-harness observability only; NVIDIA compiler ownership, CUDA evidence,
and selector state are unchanged.

This is the execution plan for evaluating, repairing, and then restructuring
the CUDA compiler tests on the NVIDIA box. It complements
[`NVIDIA_AUDIT.md`](NVIDIA_AUDIT.md); it does not reopen completed sm_120 feature
work unless a test exposes a real defect.

Baseline state on the NVIDIA box (2026-07-15, commit `ecf9483f`):

- The repository collects **264 exact-device CUDA tests** under
  `pytest -m hardware_nvidia`: **246 correctness** cases and **18 measured
  performance** cases. Eight of the correctness cases also require external
  compiler tools.
- The required CPU PR lane now excludes hardware and measured-performance
  states while retaining host-free CUDA emit, selector, validation, rejection,
  registry, and source-contract tests.
- Live CUDA tests carry `hardware_nvidia`; measured tests additionally carry
  `performance`; compiler/toolchain crossings carry `compiler_tool`.
- The WSL NVIDIA host is an RTX 5070 Ti (UUID
  `GPU-5072cda5-509a-008c-93c8-dc06e105f307`, CC 12.0), driver 610.62, CUDA
  13.3.73, LLVM/MLIR 23, and Python 3.14.4. At collection it was idle;
  observed graphics clock/power were 375 MHz / 23.18 W with a 300 W limit.
- The compiler-artifact lit suite passed **19/19**. The exact-device
  non-performance lane passed twice: **224 selected, 0 failed, 0 errored, 1
  Apple-only skip** (54.783 s and 53.658 s). This covers the required
  execute/compare layer, but it does not constitute performance evidence.
- The serial measured lane passed **18 CUDA tests** (plus the same unrelated
  Apple-only collection skip; JUnit: `/tmp/nvidia-performance-ecf9483f.xml`).
  The production hot-path ratchet, device-resident event timing, convolution
  routes, and Tile/autotune selection were included. Verbose `ptxas` and
  `cuobjdump --dump-resource-usage` for the compiler-produced Tile kernels
  report zero spills: 36 registers and no shared/local memory for direct,
  GELU, and SiLU; 42 registers and 2 KiB static shared memory from `ptxas`
  (3 KiB in the cubin resource table) for shared-staging f32/bf16/bias-ReLU.
  Nsight Compute 2026.2.1 profiled the f16 device-resident GEMM proof: the
  19x13x29 case launches four one-warp blocks (2x2 grid), uses 40
  registers/thread and no kernel shared memory, and reports 50% theoretical
  versus 2.08% achieved occupancy. The low achieved value is expected for this
  deliberately tiny ABI fixture: four blocks cannot occupy all 70 SMs. It is
  launch-shape evidence, not a production-throughput claim.
- Nsight captures for the production hot-path, convolution-route, and Tile
  schedule tests are retained under `/tmp/nvidia-*-ncu.ncu-rep`. Hot-path
  GEMM/fused/attention use 40 registers/thread and no shared memory (the
  attention kernel has 22.92% theoretical occupancy); fp quantization uses 18
  registers/thread, no shared memory, and 74.36% achieved occupancy. The
  direct/shared convolution routes use 40/30 registers and 0/32 bytes static
  shared memory, respectively. The selected Tile candidates report direct:
  36 registers, 0 shared, 50% theoretical occupancy; shared: 42 registers,
  2.05 KiB shared, 83.33% theoretical occupancy. The measured achieved
  occupancies are shape-dependent and low for the micro-grid fixtures, so they
  are resource evidence rather than selector-retuning evidence.
- The hot-path ratchet intentionally fails when run under Nsight Compute:
  profiler replay raises the wall-clock samples above its uninstrumented
  repeated-median caps. Its ordinary serial run remains the timing proof; the
  Nsight run is resource-only and must never update or relax the ratchet.
- The serving benchmark completed 20-repetition device-event and end-to-end
  rows for ReplaySSM `1x128x64`/`1x256x128` and fused/staged paged-KV decode
  at 128/512/2048 tokens (`/tmp/nvidia-sm120-serving-test5.json`). A temporary
  candidate corpus records both timing domains for rectangular `128x256x64`
  and ragged `127x259x63` f16 GEMM: end-to-end selects shared Tile (0.506 and
  0.490 ms), while device timing selects direct Tile (0.00657 and 0.00703 ms).
  No committed selector or timing cap changed.
- Nsight reduction coverage passed 49 native cases: f32/f16 kernels use 28
  registers and 1.02 KiB static shared memory with 100% theoretical occupancy.
  MoE transport passed three native cases: gather/combine use 20 registers and
  no shared memory; grouped GEMM uses 40 registers and no shared memory. The
  two live MoE transport tests were missing `hardware_nvidia` and are now in
  the canonical exact-device collection; its host-only rejection test remains
  unmarked. Their repeated-median rows and resource evidence are now committed
  under the TEST-5 baselines described below.
- NVIDIA-TEST-4 now has a shared storage/accumulation tolerance contract for
  f32, f16, bf16, TF32, FP8, int8, and NVFP4 semantics. The shipped MMA and
  compiler-produced Tile proofs consume that contract; exact integer/NVFP4
  contracts remain bit-exact. The reduction matrix now also proves f16/f32
  non-finite propagation and rejects empty, rank-invalid, unsupported-storage,
  and unknown-operation contracts before launch. Two WSL exact-device runs
  recorded 243 tests with zero failures/errors (one expected Apple-only skip).
- NVIDIA-TEST-5 now productizes repeated-median reduction and MoE transport
  rows in `record_reduction_transport_baseline.py`: every route records both
  end-to-end and CUDA-event timing through the production generated kernel.
  Two 20-sample sm_120 runs were recorded, the committed ratchet baseline
  covers reduction sum/mean/max plus MoE dispatch/combine/grouped-GEMM, and
  the first expanded serial performance lane passed 19 tests (one expected
  skip). The wider corpus and parsed resource evidence have since landed.
- The TEST-5 D2 corpus now includes measured square `512x512x512`, rectangular
  `128x256x64`, and ragged `127x259x63` f16/bf16 GEMM rows in both timing
  domains, alongside fused GELU, forward attention, gated MLP, and convolution
  routes. Two 20-sample WSL runs were taken before retaining the second corpus;
  the initial end-to-end winners varied between runs, so no selector was
  promoted from that evidence.
  Serving was likewise refreshed from two 20-sample runs for ReplaySSM and
  fused/staged paged-KV at 128/512/2048 tokens, retaining device-event and
  end-to-end medians separately.
- **NVIDIA-TEST-5 is closed (2026-07-16).** Two fresh high-sample sweeps
  (50 end-to-end repetitions after 10 warmups; 200 device-event repetitions
  after 20 warmups) converge for all 20 retained D2 rows under the declared 3%
  noise policy. Every row is selector-eligible only because both runs share a
  near-winner consensus and the selected route has a committed resource
  fingerprint. Backward attention adds regular `1x8x128x64` and ragged
  `1x8x257x64` dual-domain ratchets. Parsed Nsight evidence records registers,
  static/dynamic shared memory, theoretical/achieved occupancy, and explicit
  local-load/store spill counters for GEMM/Tile, fused and forward/backward
  attention, convolution, reductions, MoE transport, paged-KV, and ReplaySSM.
  The backward VJP uses 48 registers and measurable local-memory traffic; this
  is retained evidence, not hidden by a zero-spill claim. All other selected
  rows in the resource manifest recorded zero local spill traffic. The final
  serial performance lane passed 20 tests with one expected Apple-only skip.
- NVIDIA-TEST-6 has begun with `tests/_support/nvidia.py` (with a retained
  `tests/unit/_nvidia_testutil.py` compatibility import): it centralizes
  CUDA-toolchain, MMA-runtime, and bare CUDA-host probes without conflating
  their skip semantics, and supplies a common native-provenance assertion.
  The MoE transport, reductions, paged-KV, and ReplaySSM families migrated in
  the first batch; 70 focused tests passed and the canonical device collection
  remains 243 nodes. This is NVIDIA-only test infrastructure; Apple and ROCm
  plan states are unaffected.
- Cross-backend sync `LLVM23-NVIDIA-2026-07-16`: NVIDIA exact-device parity is
  now validated on the RTX 5070 Ti after the shared LLVM/MLIR 23 migration.
  A clean `sm_120a` build required the MLIR bytecode interface include and the
  `NVVM::Barrier0Op` to `NVVM::BarrierOp` API migration. NVIDIA lit passes
  19/19, two stable collections contain the same 268 nodes, the host-free
  compiler-artifact proof passes, exact-device correctness passes 248/248
  twice, TEST-4/TEST-6 focused gates pass 190/190, and the isolated TEST-5
  lane passes 20/20. Explicit Tile tool paths now take precedence over stale
  build-tree binaries. ROCm receives only the LLVM 23 lit-shell compatibility
  update; Apple has no affected physical schedule or runtime contract.
- **NVIDIA-TEST-7 is closed as local WSL release ownership; GitHub runners are
  intentionally not used.** The release command exposes independent `cpu`,
  `compiler`, `device`, and `performance` layers, rejects overlapping runs with
  a host lock, writes a fail-closed status record, retains timestamped machine,
  JUnit, and baseline bundles, and keeps performance serial. The finalized
  all-layer invocation passed 410 host-free/shared-registry tests (one explicit
  skip), 20/20 lit, 1/1 compiler artifact, 268/268 correctness twice, and 20/20
  performance. Its retained bundle is
  `artifacts/nvidia-release/20260717T003224Z-18866bbb/all/`.
- The second batch removed the same local MMA-runtime probe from norm, softmax,
  matmul-ReLU, matmul-softmax, compiled KV-cache, forward/backward Flash
  Attention, and convolution tests. Their 89 focused exact-device tests passed
  on the RTX 5070 Ti. Specialized compiler and Tile availability probes remain
  local until their stronger capability contracts can be preserved explicitly.
- The third batch migrated control flow, DeltaNet, dequant GEMM, FP quant,
  local collectives, optimizers, positional encoding, and SSM to the shared
  MMA-runtime probe. It also classified their live CUDA tests with
  `hardware_nvidia` while leaving host-only negative tests unmarked. The 32
  focused tests passed; collection increased from 243 to 264 exact-device
  nodes (246 correctness, 18 performance).
- The next helper-deduplication batch replaced private ordinary MMA-runtime
  probes in linear attention, MLA decode, and sparse attention with the shared
  capability-specific helper. The E3 hand-tuned GEMM proof now uses the shared
  MMA-plus-PTX-launch predicate; Tile tool/runtime checks remain local because
  they prove a stronger compiler-path capability.
- The second physical relocation split the mixed NVIDIA MMA launch file into
  two host-free execution-matrix contracts and five exact-device launch/JIT
  proofs under `tests/device/nvidia/`. The mapped cohort passed 21 focused
  tests, 19/19 compiler lit plus its compiler pytest contract, exact-device
  correctness twice (246 passed, one Apple-only skip each), and serial
  performance (18 passed, one Apple-only skip).
- The third physical relocation moved the two device-only DSA sparse-attention
  proofs to `tests/device/nvidia/test_sparse_attention.py`. Its node map,
  focused execute/compare run, compiler artifact lane, two exact-device runs
  (246 passed, one Apple-only skip each), and serial performance lane (18
  passed, one Apple-only skip) all passed without changing the 264-node
  NVIDIA marker topology.
- NVIDIA compiler-artifact selection no longer relies on the
  `test_nvidia_*.py` filename pattern: the `compiler_nvidia` marker owns the
  CUDA artifact lane and its release-gate selection. `NvidiaDeviceSession`
  now frees all tracked buffers and destroys its stream even after a
  synchronization failure, and destroys a successfully-created timing event
  if its partner event cannot be created. Host-free fault-injection tests pass
  (2/2); the marker artifact lane passed and the real stream/event ABI fixture
  passed 15/15 on the RTX 5070 Ti.
- **NVIDIA-TEST-6 is complete (2026-07-16).** The closure audit found no
  remaining ordinary private MMA/PTX probe
  implementation: plugin and hot-path-ratchet compatibility names now delegate
  to shared predicates, while the Tile probe remains intentionally specialized.
  Running the hot-path ratchet immediately after the broad plugin matrix
  exceeded two f16 caps (512³ and 1024³); two isolated serial reruns both
  passed. The disposition is test-state contamination outside the canonical
  isolated performance lane, not a tolerance change or performance regression.
  An AST ratchet now rejects any future exact-device test under `tests/unit`.
  The final topology collects 333 NVIDIA nodes; compiler artifacts pass 20/20,
  exact-device correctness passes 313/313 twice, and the serial measured lane
  passes 20/20. New backward-attention and epilogue families landed directly
  in `tests/device/nvidia`, and the backward nodes are recorded in the
  executable post-migration map.
- The control-flow cohort is now accepted: source/rejection contracts remain
  host-free under `tests/unit`, while the bounded-control and runtime-binding
  execute/compare proofs moved to `tests/device/nvidia/test_control_flow.py`.
  Its mapped nodes passed focused validation, 19/19 compiler lit plus the
  NVIDIA compiler marker lane, exact-device correctness twice (246 passed,
  one Apple-only skip each), and serial performance (18 passed, one skip).
- The first device run exposed a product defect in Tile GELU: NVPTX could not
  select LLVM's `ftanh`; after its arithmetic lowering, SiLU exposed the same
  issue for `fexp`. Both now lower through a bounded Pade tanh expression, so
  no unsupported transcendental libcall is emitted. Focused Tile and related
  CUDA correctness tests passed **38/38** after the fix.

## Completion definition

This plan reaches `closed` only when all of the following are true on the
NVIDIA box:

1. Host-free CPU PR tests, NVIDIA compiler-artifact tests, CUDA device
   correctness tests, and CUDA performance tests run as separate commands with
   separate reports.
2. Every exact-device test proves `native_gpu` provenance and compares against
   the same numerical oracle used by the CPU/ROCm paths; no fallback earns a
   pass.
3. The full non-performance CUDA device matrix passes twice from a clean build.
4. Performance tests run serially after warmup and commit repeated-median
   kernel-only and end-to-end evidence. Timing under xdist is forbidden.
5. Tool, dtype, diagnostic, op, target, execution-state, and generated-doc
   registries remain green.
6. Duplicate/source-scan tests are removed only after an equal or stronger
   semantic, FileCheck, object/SASS, or execute/compare proof replaces them.
7. The NVIDIA release gate owns this lane and preserves logs plus machine
   identity for each proof run.

## NVIDIA-box preflight

Record this before interpreting any failure:

```bash
nvidia-smi --query-gpu=name,uuid,compute_cap,driver_version,memory.total \
  --format=csv,noheader
nvcc --version
ptxas --version
python3 --version
git rev-parse HEAD
```

Required target is RTX 5070 Ti / compute capability 12.0. NVFP4/block-scale
tests compile the architecture-specific `sm_120a` target. Record driver, CUDA
toolkit, LLVM/MLIR, Python, GPU UUID, clocks/power mode, and whether another
process is using the device.

### Install LLVM/MLIR 23 on Ubuntu

Use the repository bootstrap on Ubuntu 24.04; it installs one matched LLVM,
Clang, LLD, MLIR, and Polly 23 toolchain from apt.llvm.org:

```bash
bash scripts/setup_ubuntu.sh
source .venv/bin/activate
```

For a toolchain-only manual installation, use a dedicated versioned source
file rather than replacing the distribution LLVM packages:

```bash
sudo install -d -m 0755 /etc/apt/keyrings
wget -qO- https://apt.llvm.org/llvm-snapshot.gpg.key \
  | sudo gpg --dearmor --yes -o /etc/apt/keyrings/apt.llvm.org.gpg

. /etc/os-release
LLVM_SUITE="llvm-toolchain-${VERSION_CODENAME}-23"
if ! wget -q --spider \
  "https://apt.llvm.org/${VERSION_CODENAME}/dists/${LLVM_SUITE}/Release"; then
  LLVM_SUITE="llvm-toolchain-${VERSION_CODENAME}"
fi
echo "deb [signed-by=/etc/apt/keyrings/apt.llvm.org.gpg] https://apt.llvm.org/${VERSION_CODENAME}/ ${LLVM_SUITE} main" \
  | sudo tee /etc/apt/sources.list.d/llvm-23.list >/dev/null
sudo apt-get update
sudo apt-get install -y \
  clang-23 lld-23 llvm-23 llvm-23-dev llvm-23-tools \
  mlir-23-tools libmlir-23-dev libpolly-23-dev

export LLVM_ROOT=/usr/lib/llvm-23
export PATH="$LLVM_ROOT/bin:$PATH"
export CMAKE_PREFIX_PATH="$LLVM_ROOT${CMAKE_PREFIX_PATH:+:$CMAKE_PREFIX_PATH}"

llvm-config --version
mlir-opt --version
mlir-tblgen --version
FileCheck --version
```

All four commands must report major version 23. Remove or disable any stale
pre-23 toolchain source selection from the build environment; keeping
multiple apt repositories installed is acceptable, but Tessera's CMake cache,
compiler executables, MLIR tools, and CMake package directories must all resolve
to `/usr/lib/llvm-23`.

Build the compiler and CUDA runtime from a clean NVIDIA build directory:

```bash
cmake -S . -B build-nvidia-cuda -G Ninja \
  -DCMAKE_C_COMPILER=/usr/lib/llvm-23/bin/clang \
  -DCMAKE_CXX_COMPILER=/usr/lib/llvm-23/bin/clang++ \
  -DLLVM_DIR=/usr/lib/llvm-23/lib/cmake/llvm \
  -DMLIR_DIR=/usr/lib/llvm-23/lib/cmake/mlir \
  -DTESSERA_BUILD_NVIDIA_BACKEND=ON \
  -DTESSERA_ENABLE_CUDA=ON \
  -DTESSERA_CUDA_ARCH=sm_120a \
  -DTESSERA_BUILD_EXAMPLES=OFF
ninja -C build-nvidia-cuda tessera-opt tessera-nvidia-opt \
  tessera_nvidia_gemm tessera_runtime
```

Export explicit tool paths rather than relying on a previous build:

```bash
export TESSERA_OPT="$PWD/build-nvidia-cuda/tools/tessera-opt/tessera-opt"
export MLIR_OPT=/usr/lib/llvm-23/bin/mlir-opt
export PYTHONPATH="$PWD/python:$PWD"
```

Adjust `TESSERA_OPT` to the actual Ninja output reported by the build if the
generator places it under `build-nvidia-cuda/tools/tessera-opt/` differently.

The 2026-07-16 shared compiler migration raises the project floor to matched
LLVM/MLIR 23 and updates portable Tile/NVIDIA TableGen plus greedy-rewrite
compatibility. The shared sources compile in the LLVM/MLIR 23 ROCm build, and
NVIDIA exact-device parity is now validated independently on the `sm_120`
host. No CUDA execution status was inferred or promoted from the ROCm run.

## Ordered work

| Order | ID | Work | Engineering action | Completion gate |
|---:|---|---|---|---|
| 1 | NVIDIA-CALIB-1 | Validate T1 reuse/cache pruning against the committed sm_120 corpus | Supply evidence-backed sm_120 bandwidth/cache inputs, then compute per-family and per-shape rank correlations without reviving the rejected step-distance score. | The analysis records model version, corpus identity, rank correlations, and a retain/reject verdict; no new device run is required. |
| 2 | NVIDIA-RASTER-1 | Consume and measure the shared raster contract | Wire the emitter with row-major identity preserved, then sweep only on sm_120 with device timing and `ncu` L2 evidence. | Exact-device correctness remains unchanged and a measured raster-order/group decision is recorded without hardware-free arbitrary selection. |
| 3 | NVIDIA-AOT-1 | Decide whether the NVRTC lane needs a precompiled peer | Reuse the Apple AOT/JIT harness and its never-before-compiled-kernel cache control; first decide whether the expected cold-start use case justifies implementation. | A documented not-applicable/retain decision or a measured precompiled candidate with equivalent numerics and explicit offline-build amortization. |
| 4 | NVIDIA-TEST-1 | Establish a reproducible baseline | Run collection and each proof layer separately; save JUnit, skip reasons, duration report, machine identity, and the exact commit. Classify every failure as product defect, test defect, environment defect, or stale claim. | Two collections return the same node set; no unknown markers; every skip has an explicit unavailable capability. |
| 5 | NVIDIA-TEST-2 | Compiler-artifact layer | Run `check-tessera-nvidia` plus CUDA pytest files carrying `compiler_tool`; migrate private tool probes to `compiler_toolchain`; split artifact assertions from the eight tests that currently continue into device execution; replace large textual snapshots with named diagnostics, FileCheck, or focused IR/object invariants. | Clean build passes without a GPU; missing-tool simulation skip-cleans; no compiler test invokes a nonexistent path. |
| 6 | NVIDIA-TEST-3 | Exact-device correctness | Run `hardware_nvidia and not performance`; group failures by GEMM/Tile, attention, reductions/norms, control flow, KV/ReplaySSM, collectives, and ABI/conformance. Require native provenance and execute/compare. | Entire correctness matrix passes twice; fallback-injection negatives fail to earn native proof. |
| 7 | NVIDIA-TEST-4 | Numerical policy | Centralize dtype/op tolerances from accumulation/storage behavior. Add ragged, rectangular, boundary, non-finite, misalignment, and invalid-contract cases where absent. | f16/bf16/tf32/FP8/int8/NVFP4 cases use documented tolerances; no default zero-`atol` checks near zero. |
| 8 | NVIDIA-TEST-5 | Measured performance | Run `hardware_nvidia and performance` serially. Warm up compilation and caches; use repeated medians; measure kernel-only and end-to-end separately; record registers, shared memory, occupancy, spills, and selected route. | Stable baselines cover square/rectangular/ragged GEMM, fused epilogues, attention, paged KV, ReplaySSM, reductions, and transport. Each ratchet identifies the selected implementation. |
| 9 | NVIDIA-TEST-6 | Refactor and deduplicate | Move mature families toward `tests/compiler/`, `tests/device/nvidia/`, `tests/integration/`, and `tests/performance/nvidia/`. Consolidate repeated CUDA availability, compilation, launch, oracle, and cleanup code. | No central filename allowlist; no duplicated private CUDA probe/loader; process trees and device allocations clean up on failure. |
| 10 | NVIDIA-TEST-7 | Local release ownership | Own the NVIDIA-box release gate locally in WSL with a host concurrency lock and retained artifacts; GitHub runners are intentionally not used. Keep two-run device correctness required for NVIDIA promotion and performance serial. | A clean branch run reports NVIDIA host-free/shared registries, compiler artifact, device correctness, and performance independently and retains the fail-closed evidence bundle. |
| 11 | NVIDIA-LSE-1 | Consume and measure the real shared LSE checkpoint on CUDA | **Landing, SM120 P0 complete.** Compiler-owned f32 paired physical ABIs now carry `Q/K/V -> O,row_lse` and `dO/Q/K/V,row_lse -> dQ/dK/dV` through Tile operands, descriptors, bridge allocation/copies/argument order, runtime validation, and the native-package route. Exact RTX 5070 Ti proof covers oracle equality, saved-vs-recompute forward/backward equality, and malformed rank rejection. | At `[1,2,1,3,4,4,3]`, saved lowers backward event time (0.01716 vs 0.02777 ms) but the paired save/load e2e median loses (1.56149 vs 1.26335 ms); NCU/resource packet is retained in `benchmarks/baselines/nvidia_sm120_lse_checkpoint_2026_07_30.json`. Retain recompute default; repeat representative sequence/shape sweeps before promotion. |
| 12 | NVIDIA-E2E-1 | Canonical SM120 compiler spine | Under sync `E2E-SPINE-2026-07-18`, compose Graph/Schedule/Tile lowering with `LowerTileToNVIDIA(sm=120)`, NVVM/PTX/native-image packaging, and the existing register/invoke launch bridge. Prove f16 and NVFP4 first, including non-origin scale tiles and general-shape dispatch. | One canonical driver request returns a typed image artifact plus launch descriptor, registers and launches on `sm_120`, compares numerically, and retains compiler/ABI/device/resource evidence without a selector change. |
| 13 | NVIDIA-E2E-2 | Per-SM and operation breadth | Replace shared-alias/hardcoded target behavior with architecture-specific pipelines, then move supported CUDA families through the same typed image/launch seam. | Every enabled SM/family has the four-layer proof on its exact device or an explicit unsupported/planned terminal state; `sm_90` and `sm_100` are never inferred from `sm_120`. |

### High-risk NVIDIA-TEST-6 migration

**NVIDIA-TEST-6-HIGH — Relocate mature CUDA families without breaking the
proof contract.** Move mature compiler, device, integration, and performance
families toward `tests/compiler/`, `tests/device/nvidia/`,
`tests/integration/`, and `tests/performance/nvidia/`. This is high risk because
pytest node IDs, import roots, marker collection, CI selection, and retained
JUnit history can all change even if individual assertions still pass.

Before accepting the migration, record an old-to-new node map, preserve every
`hardware_nvidia`/`performance`/`compiler_tool` classification, prove that the
old paths have no duplicate collection, and run the host-free, artifact,
exact-device, and serial-performance layers. Do not combine this migration with
backend behavior, tolerance, or selector changes.

**Pilot evidence (2026-07-15, `landing`).** MoE transport is the first
relocated family. Its two native CUDA execute/compare nodes now live in
`tests/device/nvidia/test_moe_transport.py`; its host-free invalid-partition
contract remains in `tests/unit/test_nvidia_moe_transport_contract.py`. The
checked-in old-to-new map is `tests/device/nvidia/node_migrations.json`, and
`tests/unit/test_nvidia_test_location_migration.py` prevents restoration of
the old file or duplicate destinations. The second cohort applies the same
contract to the former mixed `test_nvidia_launch_execute.py`: two host-free
execution-matrix nodes remain under `tests/unit/`, and five native launch/JIT
nodes move to `tests/device/nvidia/test_launch_execute.py`. The combined roots
collect exactly **264** `hardware_nvidia` nodes (246 correctness, 18
performance). The two device-only DSA sparse-attention nodes are also mapped
to `tests/device/nvidia/test_sparse_attention.py`. Every relocated node
preserves its `hardware_nvidia` classification; none gained `performance` or
`compiler_tool` classification.

The compiler-artifact proof passed (19/19 lit and 1 compiler-tool pytest
contract), exact-device correctness passed twice (246 passed, 1 unrelated
Apple-only skip, zero failures/errors on each run), and the serial performance
lane passed (18 passed, 1 unrelated Apple-only skip). The second cohort
repeated those artifact, two-run correctness, and serial-performance proofs;
its executable node-map and retained host-free contracts passed (4/4). The
complete host-free PR command is **not an
NVIDIA-host acceptance gate** when it exercises Apple/ROCm compiler passes:
this WSL checkout's generic `build/` is intentionally NVIDIA-only, so 274
foreign-backend compiler tests cannot run here. This is not a relocation
failure or an NVIDIA-TEST-6-HIGH blocker. `APPLE-CI-2` and `ROCM-TEST-1` own
validation of their respective host-free compiler configurations on the correct
backend hosts; the NVIDIA host retains the focused host-free migration guard
plus its artifact and exact-device proof layers.

**Completion evidence (2026-07-16, `complete`).** The executable map now covers
**286** relocated node IDs. Mature execute/compare families are collected from
`tests/device/nvidia/`; paged-KV, ReplaySSM, and the MMA bridge are in
`tests/integration/`; and hot-path, Conv2D, MMA-symbol, and plugin timing
proofs are in `tests/performance/nvidia/`. The mixed plugin implementation is
shared through a non-discovered support module, while its 20 host-free
contracts, 53 native nodes, and 8 measured nodes are collected only from their
respective unit/device/performance entry points. The only remaining
`hardware_nvidia` references under `tests/unit/` are release-gate and
marker-policy structural assertions.

The final migrated plugin cohort passed its focused mapping/architecture guard
(93 tests), compiler artifacts (19/19 lit; one compiler pytest pass and one
hardware-excluded skip), exact-device correctness twice (246 passed and one
Apple-only skip each), and serial performance (18 passed and one skip).
Relocating Conv2D exposed an order-dependent product defect: automatic f32
dispatch admitted the explicit `im2col_tf32` candidate under a looser internal
tolerance. Automatic dispatch now selects only f32-accurate direct/shared
routes; explicitly requested TF32 performance coverage remains intact. The
post-fix exact-device matrix passed twice (246 passed and one skip each), and
the serial performance lane passed (18 passed and one skip). This is a product
correctness fix with retained before/after numerical evidence, not a tolerance
relaxation.

The final static audit finds no `hardware_nvidia` test function under
`tests/unit`; structural marker/release assertions remain host-free. The
expanded map contains 292 relocations plus 23 post-migration nodes. The final
four-layer proof is 20/20 compiler lit, 313/313 exact-device correctness twice,
and 20/20 serial performance on the RTX 5070 Ti. This closes the migration;
future native tests must land directly in device, integration, or performance
roots and satisfy the same AST/node-map ratchets.

## Canonical commands on the NVIDIA box

```bash
# 0. State/collection contract (currently 334 nodes)
python3 -m pytest tests/unit tests/device/nvidia tests/performance/nvidia tests/integration \
  -m hardware_nvidia --collect-only -q --no-header

# 1. Host-free PR contract, including CUDA emit/validation/rejection tests
python3 scripts/run_unit_tests.py --timeout=180 -q

# 2. Compiler artifacts without claiming device execution
ninja -C build-nvidia-cuda check-tessera-nvidia
python3 -m pytest tests/unit tests/device/nvidia tests/integration \
  -m "compiler_nvidia and not hardware_nvidia" -q --durations=50 \
  --junitxml=/tmp/nvidia-compiler-tool.xml

# 3. Exact-device correctness; run twice from the same clean build
python3 -m pytest tests/unit tests/device/nvidia tests/integration \
  -m "hardware_nvidia and not performance" -q --durations=100 \
  --junitxml=/tmp/nvidia-device-correctness.xml

# 4. Measured lane: serial only
python3 -m pytest tests/unit tests/device/nvidia tests/performance/nvidia tests/integration \
  -m "hardware_nvidia and performance" -q -n 0 --durations=0 \
  --junitxml=/tmp/nvidia-performance.xml
```

Do not use `-x` for the first baseline: the complete failure topology is needed
to design the migration. After triage, use focused files for the edit loop and
rerun the complete layer before marking an item complete.

## Failure triage contract

For every failure, record:

- node id, proof layer, target/dtype/shape, seed, selected route, and native
  provenance;
- whether it reproduces alone, serially, and on the second clean run;
- compiler stdout/stderr and named diagnostic code;
- numerical maximum absolute/relative error and first failing index;
- kernel-only versus end-to-end latency for performance cases;
- register/shared-memory/occupancy/spill evidence when a kernel changes;
- disposition: fix product, fix test state, replace weak test, merge duplicate,
  or document an exact environment blocker.

Never relax a tolerance or timing cap solely to make the lane green. Recompute
it from dtype semantics or a stable repeated-median baseline, and retain the
before/after evidence.

## Initial family matrix

| Family | Representative coverage | Required follow-up on NVIDIA box |
|---|---|---|
| Tile/GEMM | compiler-generated SM120 fragments, shipped MMA symbols, ragged/grid GEMM, f16/bf16/tf32/FP8/int8, NVFP4 OMMA | Verify exact SASS/instruction family, lane maps, ragged stores, allocation cleanup, and kernel/device timing separation. |
| Fusion | bias, ReLU, GELU, SiLU, gated SwiGLU, matmul-softmax | Cross-check epilogue order and dtype accumulation against CUDA and ROCm shared oracles. |
| Attention | MHA/GQA/MQA, backward, sparse/DSA, window/bias/softcap | Separate compiler artifact from live execution; cover global decode positions and non-finite policy. |
| Reductions/norms | sum/mean/min/max, softmax, RMSNorm/LayerNorm | Validate non-power-of-two/ragged widths, NaN policy, large-offset variance, and dtype tolerances. |
| Stateful serving | paged KV and ReplaySSM async ring | Long decode, flush, rollback, rejection/backpressure, remapped pages, native provenance, and leak-free teardown. |
| Control/collectives | bounded for/if/while/scan and single-device collectives | Validate one-launch ABI, bad-shape rejection before launch, and explicit multi-rank deferral. |
| Performance | GEMM routes, convolution routes, device timing, hot-path ratchet | Run isolated and serial; record winner, resource evidence, kernel-only, and end-to-end rows. |

## ROCm-derived CUDA parity work

The completed ROCm work raised the proof standard for several features that
already exist on CUDA. These are CUDA audits and measured retunes, not literal
ports of AMD schedules. Share logical fixtures, ABI contracts, numerical
oracles, benchmark schemas, and decision rules across backends; keep physical
fragments and schedules architecture-owned.

In particular, an RDNA wave is not a CUDA warp, LDS is not evidence about
shared-memory behavior, VGPR pressure does not predict the CUDA register file,
and WMMA/MFMA winners do not select `mma.sync` or OMMA winners. Every production
selector change below requires fresh `sm_120a` measurements on the NVIDIA box.

| Order | ID | ROCm lesson and CUDA work | Current CUDA state | Completion gate |
|---:|---|---|---|---|
| 1 | NVIDIA-PARITY-TILE | Re-run the same logical portable-Tile fixture through the NVIDIA architecture-owned fragment selector. Cover direct/shared schedules, grid and ragged edges, supported f16/bf16/tf32/FP8/int8/NVFP4 forms, and bias/ReLU/GELU/SiLU epilogues. Add a CUDA fragment resource record containing registers, shared memory, occupancy, spills, and the selected SASS instruction family. | Compiler-generated SM120 fragments, layout oracles, and direct/shared execution tests exist. | Fixtures never author physical fragments; pack/execute/unpack/store matches the shared oracle; emitted instructions and resource rows match the selected `sm_120a` contract. |
| 2 | NVIDIA-PARITY-GEMM-RATCHET | Extend the hot-path recorder into a repeated-median schedule matrix covering square, rectangular, ragged, dtype, and fused-epilogue cases. Record kernel/device-event and end-to-end time separately, then capture registers, shared memory, occupancy, and spills before changing the production tile selector. | `record_hot_path_baseline.py` provides a useful but narrow latency ratchet. | A committed device-keyed baseline identifies every candidate and winner; two stable runs agree within the declared noise policy; no selector change lands without before/after resource evidence. |
| 3 | NVIDIA-PARITY-LEGACY-RETUNE | Re-evaluate older f32/tf32 GEMM, grouped GEMM, grouped SwiGLU, KV movement, and MoE transport now that the compiler and fragment selection are stronger. Compare compiled, shipped, and staged/direct candidates without conflating launch/transfer cost with kernel time. | Individual CUDA paths and hot-path rows exist, but there is no ROCm-equivalent wide retune corpus. | All candidates match one oracle; kernel-only and end-to-end winners are recorded independently; grouped and transport rows include launch collapse and achieved-bandwidth evidence. |
| 4 | NVIDIA-PARITY-ATTN-FWD | Apply the G6-B methodology to CUDA forward attention: evaluate occupancy-aware multi-warp CTA schedules with online softmax at D=128, plus ragged, causal/window, bias, softcap, and MHA/GQA/MQA cases. Do not assume ROCm's two-wave shape is the CUDA winner. | Compiled CUDA forward-attention paths and exact-device tests exist. | Candidate schedules match the shared oracle; traffic and resource evidence explain the winner; the selected route wins repeated-median kernel timing without regressing end-to-end timing. |
| 5 | NVIDIA-PARITY-ATTN-BWD | Apply the G6-C methodology to dK/dV backward. Measure the existing path against atomic and split-workspace/reduction candidates, including deterministic behavior and workspace limits. | Compiled CUDA backward attention is covered, but has not been re-ratcheted against the split/reduced design space. | Forward-derived gradients pass the shared tolerance matrix; determinism and workspace caps are explicit; resource, kernel-only, and end-to-end rows select the production route. |
| 6 | NVIDIA-PARITY-PAGED-KV | Re-prove the stable paged-KV ABI with non-identity/permuted pages, remaps, causal offsets, and boundary lengths. Compare direct resident page-table attention with staged/gather-to-FA using the same oracle and retain both timing domains. | Direct fused and staged paged-attention candidates plus an SM120 serving baseline already exist. | Every candidate consumes the same ABI and matches the same permuted-page oracle; device-event and end-to-end rows may choose different winners and the cache keys preserve that distinction. |
| 7 | NVIDIA-PARITY-REPLAY | Re-run CUDA ReplaySSM against the closure matrix exposed by ROCm: long decode, flush, rollback, speculative rejection, block submit, ordered async ring, backpressure, and teardown. Expand B/D/N/M shapes and record state traffic as well as latency. | CUDA is the reference persistent ReplaySSM implementation and has serving rows, but needs the wider proof and benchmark matrix. | All transitions match `SSMStateHandle`; rejected work cannot mutate committed state; ring ordering and cleanup survive stress; traffic plus kernel/end-to-end latency are committed. |
| 8 | NVIDIA-PARITY-EPILOGUE | Make the common Tile epilogue contract explicit for bias, ReLU, GELU, and SiLU. Check accumulator precision, operation order, optional bias/residual guards, ragged stores, and all supported storage dtypes against shared CUDA/ROCm fixtures. | CUDA emits fused epilogues and plugin tests cover representative forms. | One backend-neutral oracle drives both backends; every supported fusion executes natively; unsupported dtype/op pairs reject with registered diagnostic codes rather than silently de-fusing. |
| 9 | NVIDIA-PARITY-AUTOTUNE | Align CUDA and ROCm corpus schemas around device-keyed candidates, timing domain, compiler/resource fingerprint, cold/warm compile state, and cache behavior. Promote a winner only after the relevant correctness and schedule ratchets pass. | CUDA has autotune and serving corpus writers, but their evidence must be reconciled with the newer ROCm records. | Corpus validation rejects stale devices, compilers, resources, and timing domains; cold/warm behavior is reproducible; selector decisions cite a retained measurement row. |
| 10 | NVIDIA-PARITY-TRANSPORT | Close KV-movement and MoE-transport parity with direct/staged routes, ragged/grouped loads, bandwidth attainment, and launch-amortization measurements. Feed any winner into the legacy retune only after ABI and correctness closure. | CUDA transport operations exist but lack one consolidated exact-device performance proof. | Byte counts and achieved bandwidth are auditable; kernel-only and end-to-end winners are separate; awkward sizes and grouped routes match their reference without leaks or hidden host staging. |

### CUDA parity execution record

The parity queue uses ROCm's logical coverage and proof methodology, not its
physical schedules. CUDA owns warp/register packing, `HMMA`/`QMMA`/`IMMA`/OMMA
selection, shared-memory staging, barriers, occupancy limits, and every selector
winner. An AMD wave shape, LDS strategy, or VGPR result is never a CUDA default.

- **NVIDIA-PARITY-TILE — complete on sm_120a.** The architecture-owned SM120 fragment
  selector now describes f16 (f16/f32 accumulation), bf16, TF32, FP8 E4M3/E5M2,
  int8, and block-scaled NVFP4 separately. C++ lowering consumes the descriptor
  for physical input packing and per-lane register-count validation. The exact
  compiler path passes the shared numerical oracle for f16/bf16/TF32/FP8/int8,
  direct/shared grids, ragged edges, and bias/ReLU/GELU/SiLU. The reproducible
  13-row `nvidia_sm120_tile_fragment_resources.json` record retains cubin hashes,
  registers, shared memory, theoretical occupancy, spills, and observed SASS:
  `HMMA` for f16/bf16/TF32, `QMMA` for FP8, `IMMA` for int8, and block-scaled
  `OMMA` for NVFP4. Portable typed Tile now carries NVFP4's two logical UE4M3
  scale tiles. C++ consumes nibble-packed logical A/B storage, materializes the
  backend-owned scale selectors, emits real block-scaled inline PTX, assembles
  for `sm_120a`, and passes the non-uniform-scale numerical oracle without
  fixture-authored physical fragments. The resource row now comes from that
  typed compiler artifact rather than the original CUDA spike.
- **NVIDIA-PARITY-GEMM-RATCHET — measured complete; no promotion.** The
  device-keyed `nvidia_sm120_gemm_schedule_matrix.json` contains 34 exact-case
  rows spanning square, rectangular, ragged, f16/bf16, and bias plus
  none/ReLU/GELU/SiLU epilogues. Every row has two stable repeated-median runs,
  separate CUDA-event and rotated-interleaved end-to-end timing, complete
  per-candidate resource fingerprints, and an explicit 3% noise policy. The
  smallest ragged rows require 50 untimed device warmups to remove clock-ramp
  drift. The record intentionally leaves the production selector unchanged.
  CUDA 13.3's renamed local/shared spill-request metrics are normalized alongside
  the legacy Nsight metrics, and the synthesized fused fallback now has retained
  production-sized Nsight evidence.
- **NVIDIA-PARITY-LEGACY-RETUNE — stable complete; no selector promotion.** The
  device-keyed `nvidia_sm120_legacy_retune.json` now compares compiled exact-f32
  and shipped TF32 GEMM on square/ragged rows, one grouped GEMM launch against
  the retained per-expert decomposition, and a new grouped SwiGLU route whose
  four launches are independent of expert count against the legacy `4E` route.
  All candidates use one f32 oracle and retain separate event/end-to-end rows,
  byte and achieved-bandwidth accounting, launch counts, and linked resources.
  SwiGLU rows retain both the grouped-GEMM cubin fingerprint and the exact
  generated SiLU-gate registers, occupancy, and spill record.
  The final corpus uses production-scale 512-square and 509x773x257 ragged
  GEMM, 1024x384x256x5 grouped GEMM, and 512x256x384x8 grouped SwiGLU rows.
  Exact-route resident warmup occurs after allocation in the timed session,
  and two disjoint interleaved cohorts retain every device and end-to-end batch.
  All eight rows pass 3% (maximum device/end-to-end deltas 2.47%/0.77%). TF32
  and the launch-collapsed grouped routes win both retained runs, but this
  evidence intentionally leaves selectors unchanged.
- **NVIDIA-PTX-STAGING-ARENA — landed in the legacy-retune bridge slice
  (2026-09-02).** The synchronous host-buffer PTX GEMM bridge now maps regular
  and fused GEMM operands into aligned slices of one retained primary-context
  allocation, growing only when a larger request arrives. If retaining the old
  arena makes a growth request OOM, the bridge releases that idle region and
  retries once, so a call that fits alone is not rejected for transient peak
  residency. On The-Super-Bear
  (RTX 5070 / sm_120, CUDA 13.3), the emitted f16 512-cube GEMM bridge median
  fell from **0.9666 ms** to **0.3967 ms** (2.44x) over matched warm calls;
  CUDA-event kernel time was unchanged within noise (**0.0274 → 0.0272 ms**).
  The shipped bridge integration still executes and matches the oracle. This is
  a NVIDIA-private launch-bridge change: Apple, ROCm, and x86 have no shared
  ABI, registry, or physical-schedule change. It does not promote a selector.
- **NVIDIA-PARITY-ATTN-FWD — stable complete; CUDA 4-warp candidate leads kernel time.**
  CUDA-owned 4- and 8-warp CTA candidates now cover D=128 MHA, causal sequence
  1009, ragged GQA windowing, and MQA bias+softcap. Each warp owns one query,
  uses warp shuffles for QK, and keeps distributed online-softmax/PV state;
  this is not ROCm's two-wave LDS schedule. All rows match the shared oracle
  within `7e-8`. Both candidates use 56 registers with zero spills; modeled
  occupancy is 75% for four warps and 66.67% for eight. Four warps win CUDA-event
  timing on both retained runs for every case. Ten disjoint, sample-interleaved
  end-to-end batches remove run-order aliasing without sharing observations;
  all eight rows now pass 3% with maximum device/end-to-end deltas of
  0.22%/1.84%. Small end-to-end rows do not have unanimous winner consensus,
  so production selection remains unchanged.
- **NVIDIA-PARITY-ATTN-BWD — measured complete; atomic retained.** The atomic
  incumbent and deterministic two-part split/workspace/fixed-order-reduction
  candidate share one forward-derived oracle across D64 MHA, causal D128 MHA,
  and ragged windowed GQA. The split route is bitwise repeatable, rejects
  unsupported f16 storage, and enforces an exact one-extra-dK+dV f32 workspace
  cap (524,288 bytes on MHA rows; 134,144 on ragged GQA). All six candidate
  rows pass 3% with maximum device/end-to-end deltas of 0.36%/1.67%. Resources
  retain atomic 48-register/83.33%-occupancy and split dQ 48-register,
  dK/dV 56-register/75%-occupancy, and reduction 12-register/100%-occupancy
  fingerprints plus spill evidence. Atomic wins both timing domains in every
  case by a large margin, so `selector_changed` remains false.
- **NVIDIA-PARITY-PAGED-KV — correctness and timing complete; no promotion.** Both fused
  and staged routes now pass the same permuted-page oracle at lengths 1, 3, 4,
  5, 7, 8, 9, and 13, including non-monotonic logical indices and global causal
  offsets. The 13-row transport corpus covers 127/128/129/511-token boundaries
  with separate device/end-to-end keys, byte formulas, resources, and no
  selector change. Repeated event batches now remain inside one warmed resident
  session; all eight fused/staged rows pass the 3% two-run policy, with maximum
  device and end-to-end deltas of 1.89% and 1.85% respectively.
- **NVIDIA-PARITY-REPLAY — canonical state contract and correctness complete;
  timing characterization retained.** Exact-device
  tests cover long decode across flushes, rollback, speculative rejection,
  block submit, reset, ordered ring backpressure, rejected-submit immutability,
  and teardown over wider B/D/N shapes. The 10-row replay corpus spans five
  geometries and 16/64 tokens with traffic, resources, and both timing domains.
  Each runtime handle now carries the shared `tessera.replayssm.state.v1`
  descriptor: exact persistent device and pinned-host byte formulas, session
  lifetime with preserved initialization, ordered stream/event slot ownership,
  consumer-wait-before-release, and teardown draining. Span checks reject before
  CUDA submission. The CPU oracle is outside the end-to-end interval; each
  retained run has disjoint four-route batch medians with recorded out-of-band
  clock conditioning. All errors remain below `1.5e-8`. Under the WSL 4%
  foundation policy 5/10 refreshed rows satisfy both domains; the remaining
  small/multi-batch rows range from 4.05% to 8.75%. No selector decision consumes
  these unstable rows.
- **NVIDIA-PARITY-EPILOGUE — execution matrix complete.** `FusedRegion` is the
  backend-neutral bias/activation/residual/order oracle and now emits registered
  `E_FUSED_EPILOGUE_*` diagnostics for unsupported dtype/op/order and missing
  operands. The exact-device matrix now executes all 43 supported combinations
  over f32/f16/bf16/FP8 E4M3/FP8 E5M2, optional bias, no activation or
  ReLU/GELU/SiLU, and f32 residual-after-activation ordering. Accumulation is
  f32; low-precision residual, activation-before-bias, repeated activation,
  and unsupported dtype/op pairs reject with the registered diagnostics.
- **NVIDIA-PARITY-AUTOTUNE — strict admission complete; no promotion.** Corpus
  admission can require exact device, timing domain, compiler fingerprint,
  resource fingerprints, compile state, and cache state. The committed
  reproducibility record admits all 20 selector-eligible NVIDIA rows, rejects
  stale device/timing/compiler/resource mutations, and reproduces one kernel
  cache key across two cold builds and warm hits (about 0.05 ms warm lookup).
- **NVIDIA-PARITY-TRANSPORT — correctness, evidence, and timing complete.**
  The consolidated 13-row paged-KV/MoE/grouped corpus retains auditable traffic
  formulas, achieved bandwidth, launch-amortization keys, exact resources, and
  independent timing domains. MoE dispatch/combine now consume one canonical
  `tessera.moe_transport.v1` int32/fp32 descriptor with stable expert grouping,
  capacity/drop semantics, and dispatch-before-compute-before-combine ordering;
  grouped GEMM consumes canonical ragged sizes/offsets and retains empty experts.
  Local-device scope is explicit; multi-rank collective execution remains a
  separate backend/runtime item. Maximum oracle error is below `3e-7`; all 13
  rows pass the WSL 4% foundation policy. MoE CUDA-event samples retain one
  native allocation set across repeated batches, and the tiny routes use 101
  medians per run. No selector or legacy-retune winner is promoted.

- **NVIDIA-SM120-LOWP-PRODUCTIZATION — complete (2026-07-18).** The shipped
  CUDA ABI adds general-shape block-scaled NVFP4: packed E2M1 A/B, raw UE4M3
  scale views, M16/N8 grid dispatch, K64 accumulation, ragged zero fill, and
  pre-launch shape/view rejection. Fixed 16x8x64, multi-tile 33x19x129, and
  sub-tile 7x5x31 non-uniform-scale cases match the exact NVFP4 oracle on the
  RTX 5070 Ti. Native one-kernel TF32 and FP8 E4M3/E5M2 fused-epilogue,
  QK-softmax-PV attention, and gated routes now coexist with the composed
  candidates. Two fresh runs use 20 end-to-end medians and 100 CUDA-event
  repetitions per route. The cross-domain 3% gate promotes 11 of 18 retained
  shape/dtype rows; long attention and disagreement rows remain unpromoted.
  The linked 12-row cubin record reports 40-register fused/attention kernels,
  47–48-register gated kernels, 8/32 KiB attention dynamic shared memory,
  shape-dependent 22.92%/6.25% modeled attention occupancy, zero compiler spill
  storage, and the expected TF32 HMMA / FP8 QMMA SASS. Evidence:
  `nvidia_sm120_low_precision_native_{routes,resources}.json`.
- **Audit-document reconciliation — complete (2026-07-18).** This plan,
  `NVIDIA_AUDIT.md`, and `sm120-kernel-guide.md` now agree that mature SM120
  fragments lower for real, general NVFP4 dispatch is executable, and native
  TF32/FP8 transformer candidates exist. The plan remains `landing` only for
  unrelated architecture-specific follow-ons such as sm_90 WGMMA and sm_100
  tcgen05 exact-device proof; landed SM120 work is no longer described as open.

Cross-backend sync `NVFP4-TILE-SCALES-2026-07-16` changes the shared typed Tile
operand contract only. NVIDIA supplies exact-device materialization evidence;
Apple and ROCm do not inherit its physical schedule and record their outcomes
in their own plans.

Cross-backend sync `PR420-REVIEW-2026-07-17` corrects the NVIDIA-owned NVFP4
scale materializer to apply both `tile.view` origins using the declared
row-major A-scale and column-major B-scale layouts. A live `sm_120a` fixture
selects nonzero A-row and B-column scale tiles and matches the NumPy oracle;
the NVIDIA compiler lit suite passes 21/21. The SM120 Target IR selector also
accepts canonical `fp16` as the existing f16 fragment contract. This is a
correctness/dispatch repair only: no physical fragment, resource record,
timing row, or production selector changes. The same sync makes Ubuntu LLVM
repository setup install its probe prerequisites before first use; sibling
backend outcomes are recorded in their plans.

Cross-backend sync `NVIDIA-SM120-LOWP-2026-07-18` is NVIDIA-owned. It changes no
shared dtype spelling, portable Tile scale layout, backend-neutral epilogue
order, or generic autotune schema. Apple has no enabled NVFP4 cooperative-matrix
route and ROCm gfx1151 has no FP8/FP4 WMMA instruction; neither inherits CUDA
packing, HMMA/QMMA/OMMA schedules, resource values, timings, or selector rows.

Cross-backend sync `E2E-SPINE-2026-07-18`: NVIDIA owns **NVIDIA-E2E-1** and
**NVIDIA-E2E-2**. Shared code owns only the image/launch schemas and canonical
orchestration; NVIDIA retains PTX/SASS generation, physical fragments, launch
geometry, resources, and route selection. Existing NVRTC, shipped-library, and
PTX-register/invoke paths remain valid candidates while the typed spine lands.
Host-free IR/object evidence cannot promote an SM or selector, and exact-device
proof for `sm_90`, `sm_100`, and `sm_120` remains architecture-specific. The
completed E2E-SPINE-0 foundation records SM80 as lacking an exact registered
pipeline and SM100/SM120 as shared-builder aliases; it also corrects the Python
pass inventory to match that builder without changing CUDA runtime selection.
E2E-SPINE-1 adds the portable image/descriptor and rejection contract only;
PTX/cubin contents, warp schedules, launch geometry policy, resources, and CUDA
selectors remain NVIDIA-owned and unchanged until NVIDIA-E2E-1.
E2E-SPINE-2 completes the shared typed carriers, stage ledger, cache join, and
descriptor-first exact-target launcher registry. It registers no CUDA hook and
does not reinterpret `nvidia_mma` or any shipped/NVRTC candidate; NVIDIA-E2E-1
still owns PTX packaging, `sm_120` registration/submission, numerical proof,
resources, cleanup, and the first Level-C row.

NVIDIA-E2E-1 is **complete**. The f16 slice makes an explicit canonical
driver request own the typed `tile.matmul_kernel`, runs the production
`LowerTileToNVIDIA(sm=120)` and NVVM/LLVM/PTX pipeline, validates the image with
`ptxas`, and returns the shared native-image plus exact A/B/D/M/N/K descriptor.
The descriptor registers and launches through the shipped PTX bridge on the RTX
5070 Ti; aligned `16x8x16` and ragged `37x29x23` rows match the f32 NumPy oracle.
The image retains compiler/toolchain fingerprints, cold/warm state, and ptxas
register/shared-memory/spill fields. This slice changes no production selector.
The same driver now selects a CUDA-owned general-shape NVFP4 descriptor with
packed E2M1 A/B, logical UE4M3 `scale_a`/`scale_b`, f32 output, and M/N/K. The
typed lowering owns M16/N8 origins, K64 accumulation, ragged zero fill, scale
word materialization, and guarded stores before LLVM 23 emits `sm_120a` PTX.
Exact RTX 5070 Ti rows `16x8x64`, `33x19x129`, and `7x5x31` match the block-scale
oracle; the multi-tile row uses nonuniform row/column scales to prove non-origin
scale views. Missing/malformed scales, wrong scale storage, bias, and malformed
launch shapes reject before CUDA submission. Both f16 and NVFP4 retain stable
cold/warm image identity and ptxas register/shared-memory/spill evidence. The
shared Tile verifier change is limited to the explicit eight-operand NVFP4
launch ABI; it transfers no CUDA schedule, layout, resources, or selector.

NVIDIA-E2E-2 is **closed for the available SM120 host**, with the unavailable
multi-GPU, SM90, and SM100 boundaries assigned the deferred terminal states
below. Its first dependency slice replaces the former
shared SM90 alias with exact SM90/SM100/SM120 Graph→Tile builders and registered
Tile→`tessera_nvidia`→NVVM producers. The exact target now reaches Tile IR,
the control-flow guard, async-copy lowering, and the target producer without
being rewritten to SM90. Hopper alone consumes the proven WGMMA and Hopper
FlashAttention markers; SM100 and SM120 retain target-tagged typed carriers for
architecture-owned lowering. Straight-line async copies mint typed completion
tokens, the matching wait retires them, and matrix consumers preserve those
edges through TMA lowering. Host WSL FileCheck proves the three distinct IR
routes; native SM90 and SM100 remain unsupported-by-evidence until exact-device
runs exist. No selector changes. That breadth statement described the first
landing slice and is now superseded by the implementation record below: SM120
canonical execution covers the complete matmul dtype matrix plus softmax,
reductions, fused epilogues, attention, paged-KV, ReplaySSM, and local MoE.

The next NVIDIA-E2E-2 family slice now gives static f16/f32 last-axis softmax a
canonical Level-C path. `tile.softmax_kernel` carries source/destination,
flattened Rows/K, `storage="f16"|"f32"`, `accum="f32"`, and `axis=-1`; the SM120
materializer emits a stable max-shifted row loop and target-native `nvvm.ex2`
instead of an unavailable NVPTX `fexp` libcall. LLVM 23 emits and ptxas
validates `sm_120a` PTX, while the typed descriptor registers and launches it
through the shipped CUDA-driver bridge. Exact RTX 5070 Ti proof covers shapes
`1x16`, `8x64`, `4x300`, and `2x3x48`, extreme logits, malformed output shape
rejection, stable cold/warm image identity, and resource/spill fields for both
storage types. f16 loads extend before the max/sum/normalization loops and
truncate only at output storage. This is a correctness-first 128-thread,
one-thread-per-row candidate. The existing cooperative CUDA-C route remains
selected. All four final canonical/production rows are stable in both timing
domains, and production wins both domains for the production-sized and ragged
cases.

The following NVIDIA-E2E-2 dtype-totality slice centralizes consumer-Blackwell
storage, math-mode, scalar/vector, Tensor Core, compiler, and runtime states in
`nvidia_dtype_contract.py`. Every canonical float storage type now has an
explicit row. CUDA 13.3 compile proof covers scalar/vector forms for fp64,
fp32, fp16, bf16, FP8 E4M3/E5M2, FP6 E2M3/E3M2, and packed FP4; TF32 remains
strictly an fp32 `math_mode`, never storage. Tensor Core Target IR/PTX rows now
cover the required TF32, bf16, fp16, FP8, FP6, FP4, and int8 families. The
canonical descriptor lane now executes BF16, explicit fp32-storage TF32 math,
FP8 E4M3/E5M2, and INT8 with int32 accumulation. FP64 m8n8k4 DMMA now owns a
distinct Tile lane map and f64 descriptor/bridge ABI; aligned and ragged RTX
5070 Ti rows match the f64 oracle with masked tails.
FP6 E2M3/E3M2 now assemble as `kind::mxf8f6f4`, m16n8k32,
UE8M0/`scale_vec::1X`; OCP/MXFP4 assembles as `kind::mxf4`, m16n8k64,
UE8M0/`scale_vec::2X`. Compiler-owned packed-memory Tile materializers,
five-buffer descriptors, CUDA-driver launch ABIs, and aligned/ragged numerical
proof now cover both FP6 encodings and MXFP4. In particular,
`fp4_e2m1` does not alias NVFP4: MXFP4's UE8M0 scale contract cannot reuse
NVFP4's UE4M3/`scale_vec::4X` scale words.
The shared MMA selector now requires explicit `math_mode="tf32"` for fp32 and
retains distinct `nvfp4` and `fp4_e2m1` K64 identities. No selector promotion
or production route changes.

The canonical dtype execution matrix records two disjoint, sample-interleaved
runs for square and ragged fp64/fp16/bf16/TF32/FP8/FP6/MXFP4/INT8 routes, with separate
CUDA-event and allocation/copy-inclusive timing, cold/warm image identity, and
ptxas register/shared-memory/spill fields. The retained 20-row collection
changes no selector. The final 31-sample, 10,000-device-launch and
50-end-to-end-launch run has 19/20 rows stable in both timing domains. The only
terminal miss is TF32 `256x256x256`: its device cohorts remain bimodal at 7.02%
while end-to-end is stable at 0.59%. That row is explicitly non-promoting; the
existing selector is retained rather than hiding the exact-device result.

The broader-family NVIDIA-E2E-2 reduction slice now carries
`tile.reduce_kernel(X,O,Outer,AxisExtent,Inner)` and an SM120-owned v2
materializer/descriptor ABI for f16/f32 sum, mean, and NaN-propagating max.
Normalized arbitrary axes and keepdims shape contracts execute through both a
single-owner serial schedule and a 128-thread cooperative shared-memory
candidate. Exact RTX 5070 Ti proof covers axes 0/1/2, keepdims on/off,
rectangular/ragged rank-3 inputs, f32 accumulation, non-finite values,
image/resource retention, and 42 numerical rows. The earlier last-axis record
remains historical evidence; the new comparative record applies the WSL 4%
foundation policy in both timing domains and changes no selector.

The canonical epilogue slice carries f16/bf16/TF32/FP8 E4M3/E5M2 bias,
ReLU/GELU/SiLU, optional f32
residual, and the explicit `matmul -> bias -> activation -> residual` order in
the Tile kernel plus launch descriptor. The CUDA materializer consumes distinct
bias/residual buffers and rejects unsupported dtype/order/shape contracts
instead of silently dropping epilogue semantics. The original 32 f16/bf16 rows
and the 48-case TF32/FP8 matrix pass exact-device execution. The comparative
record measures canonical single-kernel images against the existing production
composed routes, retaining both timing domains, cold/warm state,
image/resource fingerprints, spills, and raw disjoint cohorts. Production
selectors remain unchanged unless both domains select the same stable winner.

The first canonical attention slice adds a shared typed
`tile.attention_kernel(Q,K,V,O,B,Hq,Hkv,Sq,Sk,D,Dv)` carrier with explicit
f16/f32 storage, f32 accumulation/output, positive scale, and causal semantics.
The SM120 correctness-first materializer and four-buffer descriptor launch
through the shipped PTX bridge; exact RTX 5070 Ti proof passes 8/8
MHA/MQA, rectangular/ragged, causal/non-causal cases with zero spills. The
entry symbol includes the scale/causal semantic digest so incompatible images
cannot alias in the driver cache. Bias, window, softcap, dropout, and backward
are completed below. The retained eight-row
two-cohort baseline records CUDA-event and allocation/copy-inclusive timings,
cold/warm image identity, resources, and raw samples. A higher-amortization
rerun now has 8/8 rows within 3% in both domains. It remains historical
evidence; the final comparison below owns the production disposition.

The forward carrier now also owns optional dense f32 bias, signed left/right
window bounds, arithmetic softcap, and deterministic `lcg32_counter_v1`
dropout. These semantics participate in the image digest and descriptor
provenance. An exact-device advanced row proves causal+window+bias+softcap,
bitwise dropout replay, and malformed-bias rejection; the earlier 8-row
MHA/MQA matrix remains green. The f32 backward reference now also crosses the
compiler-owned seam through `tile.attention_backward_kernel` and a seven/eight-
buffer native descriptor. It assigns one dQ/dK/dV element to one thread,
performs fixed-order single-owner dK/dV reduction, requires
`deterministic=true`, and declares zero workspace. The exact-device GQA row
proves causal+window+bias+softcap derivatives, bitwise replay, descriptor-shape
rejection, and agreement with the shared Pade-softcap oracle. The final semantic
slice below adds matching f16 storage and dropout-mask replay.

The refreshed backward candidate matrix passes 6/6 exact-device oracle,
determinism, and workspace cases. All six atomic/split rows are stable in both
timing domains. Atomic wins both domains for MHA D64, causal MHA D128, and
ragged GQA; split/reduced remains the bitwise-repeatable option with one extra
dK+dV f32 workspace (134,144--524,288 bytes in the retained shapes). Production
already selects atomic, so the evidence retains that selector. The canonical
deterministic reference carrier is now landed; production selection continues
to be governed by the stable atomic/split corpus rather than the intentionally
serial reference materializer.

The paged-KV landing slice adds `tile.paged_kv_read_kernel` and a compiler-owned
f32-pages/i32-table direct descriptor ABI. Four exact-device boundary ranges,
two non-identity physical-page permutations, remap/reuse, and invalid-table
rejection pass. The existing 12-case fused/staged suite also remains green,
including causal offsets and page boundaries. The committed
`nvidia_sm120_e2e_spine_paged_kv.json` corpus compares canonical Tile-direct
against legacy CUDA staged gather at 128, 512-ragged, and 2048-ragged tokens.
It retains two repeated medians in both timing domains, cold/warm image and
cache state, registers, shared memory, occupancy, spills, and resource
fingerprints. This WSL foundation lane uses a 4% repeatability policy because
its graphics clocks are host-managed. All six candidate rows are accepted; the
legacy 2048 device-event row uses an explicit five-basis-point WSL margin at
4.02%, and margin-accepted rows are selector-ineligible. Timing-domain winners
also disagree at 512/2048, so the selector
remains unchanged. The SM120 foundation disposition is closed as retain-existing;
a future native-Linux controlled-host promotion attempt is a separate
hardware-environment follow-up, not an open migration dependency.

The stateful/MoE image slice adds compiler-owned Tile→NVIDIA→PTX packages for
ReplaySSM decode/flush and local f16/bf16/f32 MoE dispatch/combine/ragged
grouped GEMM.
The resident Replay handle no longer embeds those device kernels in its CUDA
host bridge: it loads the compiler-produced PTX functions while retaining the
session-persistent allocations, asynchronous ring, events, and ordering
contract. Compiler-owned MoE candidates launch through the generic descriptor
submission path. Exact RTX 5070 Ti tests cover
dispatch/combine numerical order,
zero-sized expert groups, ragged grouped GEMM, Replay transitions, persistent
workspace metadata, image identity, and resource retention.

The final comparative record contains 14 strict-stable rows for cooperative
softmax, four-warp forward attention, and local MoE dispatch/combine/grouped
GEMM. Every row retains two CUDA-event and allocation/copy-inclusive cohorts,
the discarded first lifecycle launch, 100-launch end-to-end amortization,
per-candidate clock conditioning, cold/warm image state, and exact resource
fingerprints. Production softmax and attention win both domains. MoE does not
produce cross-domain consensus across all three routes, so the existing MoE
selector is retained. ReplaySSM's higher-amortization 10-row matrix is now
10/10 stable in both timing domains. No selector changes.

The collective follow-on adds an explicit content-addressed rank/device
topology and a one-process/multiple-device NCCL executor for all-reduce,
all-gather, reduce-scatter, and grouped send/receive all-to-all. This host
exposes one CUDA device, so it proves deterministic topology/rejection and
records the two-device request as unavailable; it cannot supply the required
two-or-more-GPU numerical, topology, resource, or timing evidence. RCCL and
Apple mappings remain architecture-owned follow-ups. No collective or MoE
selector changes.

The remaining-dtype/reduction performance corpus uses production-sized square
and ragged TF32/FP8 fused epilogues plus f16/f32 arbitrary-axis reductions.
For every candidate it records first-use compilation/cache fill separately,
discards the first launch, and amortizes each device-event and end-to-end sample
over the next ten launches. Two disjoint time-interleaved 100-sample cohorts
retain raw samples, cold/warm or first/second-use state, image/resource
fingerprints, registers, shared memory, and spill fields. All 30 rows are
accepted under the WSL foundation rule: 29 pass the strict 4% gate and the
production fp16-mean reduction end-to-end row is explicitly margin-accepted at
4.099% under the user-approved 4.15% rounding bound. That row is
selector-ineligible. Seven strict rows have cross-domain winner consensus, but
the record changes no selector because stable consensus alone does not establish
a promotion policy or required material benefit.

The final SM120 semantic slice removes the remaining execution limitations.
The deterministic attention VJP now accepts matching f16 or f32 dO/Q/K/V and
gradient storage, accumulates in f32, and replays the forward
`lcg32_counter_v1` dropout mask from the semantic seed without a saved-mask
workspace. A compiler-owned `tile.paged_attention_kernel` consumes Q, K/V
pages, the i32 remap table, i64 logical token indices, and an explicit causal
offset in one fused descriptor; the offset is never inferred from allocation
capacity. MoE dispatch, deterministic combine, and ragged grouped GEMM now
accept f16, bf16, or f32 storage with int32 metadata, f32 combine weights, and
f32 grouped accumulation. Exact RTX 5070 Ti tests prove numerical agreement,
dropout bitwise replay, page remapping/causal boundaries, malformed metadata
rejection, and low-precision MoE execution. This shared Tile-carrier extension
transfers no CUDA schedule: Apple is not applicable because it owns a separate
resident paged-attention ABI and mature low-precision dispatch paths; ROCm
requires architecture-owned lowering before claiming these carrier variants.

Two hardware boundaries now have formal **deferred terminal** states for this
work item:

- exact two-or-more-GPU NCCL topology, numerical, resource, and timing proof is
  deferred because the available SM120 WSL host exposes one GPU;
- exact SM90 Hopper and SM100 datacenter-Blackwell Level-C evidence is deferred
  because neither exact target is available. Their compile-only Level-B
  artifacts do not inherit SM120 execution evidence.

Deferred hardware terminals do not authorize selector changes and do not hide
missing evidence. A future hardware follow-up must reopen its own exact-device
item under synchronization key `E2E-SPINE-2026-07-18`.

E2E-SPINE-3 is a shared-contract follow-up under the same synchronization key.
It is applicable to NVIDIA only as a family-granular evidence envelope around
the existing SM120 results: fixture identity, Level-C provenance, cold/warm
cache identity, benchmark metadata, and hash-sealed release-packet validation.
It changes no CUDA schedule, ABI, dtype capability, or selector. SM90, SM100,
and exact multi-GPU rows remain explicit hardware-deferred terminals and may
not inherit the SM120 packet.

The E2E-SPINE-3 exact-host recorder now packages all eight bounded SM120
families through compiler-owned image/descriptor seams: matmul, softmax,
reduction, fused epilogue, attention, paged-KV, ReplaySSM, and MoE. The six
formerly pending family rows add shared differential fixtures where needed,
prove cold/warm image and descriptor identity, retain selected route plus ptxas
resource fingerprints, and record independently conditioned repeated-median
device-event and allocation/copy-inclusive end-to-end rows. The hash-sealed
WSL RTX 5070 Ti packet is checked in against landed source commit
`9da32b78c37fc3bebf3f69d575e7b1eb4013a399`; all 16 timing rows pass the
unchanged 4% stability gate. Family-granular recording prevents one noisy
family from invalidating already-stable evidence while the final manifest
still seals one `(nvidia_sm120, sm_120a)` packet. No CUDA selector changed.

The LLVM-stage device-library follow-on makes CUDA `libdevice` an explicit
compiler dependency rather than accidental driver behavior. Native-image
identity now retains logical device-library name, content digest, and link mode
without serializing host paths. The SM120 packager fingerprints
`nvvm/libdevice/libdevice.10.bc` and uses `llvm-link --only-needed` whenever
translated LLVM IR retains an unresolved `__nv_*` call. A real `__nv_sinf`
fixture links through CUDA 13.3 libdevice, lowers with LLVM 23 `llc`, and
assembles with `ptxas -arch=sm_120a`. Intrinsic-only kernels retain an empty
linked-library set, while the available libdevice digest still participates in
the toolchain/cache fingerprint. This changes no runtime selector.

The CUDA floating-point follow-on separates three semantic routes: IEEE
arithmetic operators, function-specific CUDA libdevice calls, and explicit PTX
approximations. The shared softmax envelope now carries
`exp_mode="approx_exp2"` and `ftz=false`; SM120 accepts only that proven mode
and lowers it to `ex2.approx.f32`. The contract records PTX's full-range 2-ULP
bound, requires a nonzero near-zero comparison budget, and versions native
cache identity independently of `-O3`. It does not reuse the `__expf` accuracy
table for a different instruction and does not enable global fast math.
The semantic authority is NVIDIA's
[floating-point computation appendix](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/mathematical-functions.html);
instruction-specific accuracy comes from the
[PTX `ex2` specification](https://docs.nvidia.com/cuda/parallel-thread-execution/#floating-point-instructions-ex2).

The CUDA Math API scalar/integer follow-on records representative integer math,
bit, packed-dot, numeric/bit-cast, and 2x16/4x8 packed-SIMD families. A CUDA
13.3 `nvcc -arch=sm_120a` fixture proves the documented symbols compile, while
this original synchronization point kept every Tessera Target-IR/runtime state
`planned`. `NVIDIA-PACKED-MATH-2026-07-25` first promoted a bounded subset; the
structured continuation recorded below now closes the listed bit, bit-cast,
and packed-SIMD families through typed Target IR and 27 exact-device cases.
The shared rounding vocabulary now represents CUDA's four conversion suffixes
RN/RD/RU/RZ exactly; nearest-away and stochastic modes cannot silently map to a
CUDA cast. Undefined signed-min absolute value, out-of-range float-to-integer
conversion, funnel-shift wrap/clamp, signedness, lane width, and saturation are
retained as contract boundaries. The later internal Tile route adds no public
Graph op or selector. Sources: [CUDA Math API](https://docs.nvidia.com/cuda/cuda-math-api/index.html),
[integer intrinsics](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__INTRINSIC__INT.html),
[integer math](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__INT.html),
[casts](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__INTRINSIC__CAST.html),
and [packed SIMD](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__INTRINSIC__SIMD.html).

The PTX 9.3 truth audit now separates CUDA C++ storage spelling from physical
PTX typing. Fundamental storage rows are fp64 `.f64`, fp32 `.f32`, fp16 `.f16`,
and int8 `.s8`; BF16, TF32, FP8, FP6, FP4, and NVFP4 are alternate instruction
formats carried in same-width bit registers. Tensor fragments explicitly name
`.f64` or packed `.b32` operands. This corrects BF16 scalar/vector status to
`conversion_only` and prevents CUDA header types from implying fundamental PTX
types. PTX operand compatibility never performs automatic numeric conversion.
Direct `ptxas -arch=sm_120a` proof assembles the fundamental register surface
and rejects `bf16`, `tf32`, `e4m3`, and `u8x4` as register declarations.

Cross-backend sync `ROCM-E2E1-SOFTMAX-2026-07-19` is ROCm-owned. It adapts the
shared `tile.softmax_kernel` envelope to `tessera_rocm.softmax`, packages
HSACO, and submits through an exact gfx1151 HIP descriptor hook. NVIDIA's
`tessera_nvidia` lowering, `ex2.approx.f32` math contract, PTX ABI, SM120
schedule, resource/timing evidence, and selectors are unchanged. No AMD
wave/LDS or OCML behavior transfers to CUDA. ROCm's subsequent use of the
shared device-library record for its driver-selected OCML/OCKL/OCLC set is
parity validated at the schema boundary and requires no CUDA record or cache
change.

Cross-backend sync `ROCM-DTYPE-TOTALITY-2026-07-19` is ROCm-owned and not
applicable to NVIDIA target state. It adds no canonical dtype or alias and does
not change the SM120 PTX storage/Tensor Core contract, fragment ABI, runtime
readiness, or selector; it only prevents RDNA3.5 ISA formats from being
conflated with Tessera gfx1151 execution support.

Cross-backend sync `ROCM-DTYPE1-CLOSE-2026-07-21` promotes signed `int4` and
alias `i4` into the shared canonical/Graph-IR vocabulary and adds signedness to
the shared packed-storage descriptor. NVIDIA parity is validated at that
logical contract; NVFP4 and NVIDIA packed-weight/Tensor Core ABIs remain
distinct and backend-owned. No PTX capability, fragment ABI, runtime route, or
selector is promoted by the gfx1151 proof, and unsigned packed-4 remains
unregistered.

Cross-backend sync `E2E-FROZEN-IDENTITY-CACHE-2026-07-19`: ROCM-E2E-1 memoizes
deterministic hashes for frozen runtime artifacts, native images, and launch
descriptors. Serialized identity values and required launch validation are
unchanged, so CUDA schema parity is validated; no NVIDIA ABI, schedule,
runtime route, performance claim, or selector changes.

Cross-backend sync `ROCM-E2E2-REDUCE-2026-07-19` is ROCm-owned. It consumes the
already-shared `tile.reduce_kernel` carrier and widens its portable verifier to
admit bf16. At that synchronization point NVIDIA's backend-specific
materializer still accepted only f16/f32, so the ROCm five-argument HSACO ABI
transferred no CUDA claim. That historical NVIDIA boundary is superseded by
`NVIDIA-BF16-CANONICAL-BREADTH-2026-07-25`, which owns its independent
`Outer/AxisExtent/Inner` PTX ABI, serial/cooperative-128 lowering, resources,
and exact-SM120 evidence without inheriting the ROCm schedule or selector.

Cross-backend sync `ROCM-E2E2-PAGED-KV-2026-07-19` is ROCm-owned. It consumes
the existing shared paged-KV carrier without changing its verifier or public op
schema. NVIDIA's existing direct PTX mapping remains parity validated; no ROCm
gather schedule, HSACO ABI, page-table validation evidence, timing, readiness,
or selector state transfers to CUDA.

Cross-backend sync `ROCM-E2E2-MOE-DISPATCH-2026-07-19` is ROCm-owned. It
consumes the existing shared MoE dispatch carrier and public operation without
changing their verifier or dtype registry. NVIDIA's typed PTX mapping remains
parity validated at the carrier boundary; no AMD gather schedule, HSACO ABI,
gfx1151 evidence, timing, readiness, or selector state transfers to CUDA.

The accompanying PTX memory contract records CTA/cluster/GPU/system scopes and
relaxed/acquire/release/acq_rel atomic semantics. Vector and packed memory
accesses are sets of scalar accesses in unspecified element order, not one
atomic unit; mixed-size races fall outside the model; `red` does not form an
acquire pattern; texture/`ld.global.nc` accesses are excluded; ordered CUDA
submission does not establish intra-kernel memory order. Sources:
[types and state spaces](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#state-spaces-types-and-variables),
[instruction operands](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#instruction-operands),
and [memory consistency](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#memory-consistency-model).

The first focused CUDA parity proof on the NVIDIA box is:

```bash
python3 -m pytest -q \
  tests/device/nvidia/test_tile_fragment_compiler_path.py \
  tests/unit/test_nvidia_fragment_layout.py \
  tests/integration/test_nvidia_paged_kv_native.py \
  tests/integration/test_nvidia_replay_ssm.py \
  tests/device/nvidia/test_flash_attention.py \
  tests/device/nvidia/test_flash_attention_backward.py

python3 benchmarks/nvidia/benchmark_serving.py \
  --shapes 1x128x64 1x256x128 \
  --tokens 64 --chunk 4 --slots 4 \
  --kv-tokens 128 512 2048 --heads 8 --dim 64 --page-size 16 \
  --reps 20 --output /tmp/nvidia-sm120-serving.json

python3 benchmarks/nvidia/record_hot_path_baseline.py --reps 20 --margin 2.0
```

The focused pytest command is a correctness loop, not a substitute for the
marker-separated full CUDA lanes above. Benchmark outputs under `/tmp` are
review artifacts only; update committed baselines or autotune corpora only
after two stable runs and an explicit before/after review.

## Next update

Cross-backend sync `X86-E2E1-NATIVE-CPU-2026-07-19` classifies shared native
descriptor results for host x86 targets as `native_cpu` with CPU-wall timing.
CUDA remains `native_gpu` with its existing event and end-to-end timing domains;
no PTX ABI, SM schedule, device evidence, readiness, or selector state transfers.
The x86 pilot consumes existing Tile softmax/reduction carriers without changing
their shared dtype or operation registration.

Cross-backend sync `X86-E2E1-BREADTH-2026-07-19` consumes the existing shared
matmul and attention carriers for f32 AVX-512 descriptors. NVIDIA inherits no
x86 ABI, vector schedule, host timing, readiness, or selector state. SM120
GQA/dropout and dtype breadth remain governed by NVIDIA-owned Target IR and
exact-device evidence; x86's narrower descriptor contract changes no CUDA row.

Cross-backend sync `E2E-SPINE-2026-07-18` records the 2026-07-20 scoped x86
selector retirement: eligible static X86-E2E-1 modules now use their canonical
descriptor by default. NVIDIA parity is not applicable; no NVIDIA pipeline,
PTX ABI, schedule, capability, or selector changes. X86-E2E-2 subsequently
closed the remaining inventory and reassessed NVIDIA at each shared-contract
boundary.

Cross-backend sync `X86-E2E2-ELEMENTWISE-2026-07-20` adds the internal shared
`tile.elementwise_kernel` semantic carrier for f32 unary/binary and f32-to-bool
predicate requests. NVIDIA parity is assessed at the carrier boundary only;
the AVX-512 ABI, CPU schedule/timing, 16K binary selector threshold, and exact
x86 evidence transfer no PTX implementation or CUDA selector claim. Existing
NVIDIA elementwise target and execution rows are unchanged.

Cross-backend sync `X86-E2E2-TYPED-LOGIC-2026-07-20` widens that internal
carrier with compare, logical, and bitwise semantics plus explicit f32/i8/i32
physical storage. The capability repair is x86-owned bool/int32 truth for
already-shipped AVX-512 ABIs. NVIDIA inherits no C ABI, null-operand convention,
32K selector threshold, CPU timing, PTX implementation, or CUDA selector
claim; NVIDIA target and execution rows remain unchanged.

Cross-backend sync `X86-E2E2-FLAT-FOLLOWON-2026-07-20` extends the shared
elementwise carrier with where, transcendental, and binary-math semantics.
NVIDIA parity is assessed at the carrier boundary only; AVX-512 approximations,
C ABIs, CPU-wall thresholds, exact-host evidence, PTX routes, and CUDA selectors
do not transfer. Existing NVIDIA rows remain unchanged.

Cross-backend sync `X86-E2E2-DTYPE-2026-07-20` adds an x86-only datatype/CPUID
contract and BF16, VNNI U8/S8, and FP64 descriptor ABIs. NVIDIA already owns
independent dtype, MMA, accumulator, PTX, and runtime contracts; no CUDA target,
execution, or selector row changes.

Cross-backend sync `ATTN-DIALECT-MLIR23-2026-07-20` corrects the internal MLIR
attention dialect namespace from the nested `tessera.attn` spelling to the
MLIR-23-compatible `tessera_attn` spelling. Public Graph IR operation names,
attention semantics, NVIDIA target capabilities, PTX ABIs, schedules, and
selector state are unchanged; NVIDIA parity is validated by the shared
attention lit coverage.

Cross-backend sync `LLVM23-BACKBONE-2026-07-20` makes LLVM/MLIR 23.x the sole
accepted compiler build environment. Top-level and standalone CMake entry
points reject every other major and mixed installations; NVIDIA uses the
versioned apt LLVM 23 packages alongside CUDA 13. NVIDIA target semantics, PTX
ABIs, and selectors are unchanged, and the LLVM 23 compiler/lit build validates
host-free parity; exact-device claims remain NVIDIA-owned.

The collection contract, compiler-artifact, exact-device correctness, and
serial measured lanes now have recorded baselines. NVIDIA-TEST-5 and
NVIDIA-TEST-6 are closed; the requested attention, epilogue, and legacy-retune
parity records are stable without a selector promotion. Keep `plan_state:
landing` while unrelated implementation or exact-device follow-ons remain.
Move this plan to the NVIDIA archive only after every completion gate is met.

Consumer plan `SEQUENCE-MIXER-2026-07-17`: the compiler-direction Sequence Mixer
track ([`../../compiler/SEQUENCE_MIXER_ENGINEERING_PLAN.md`](../../compiler/SEQUENCE_MIXER_ENGINEERING_PLAN.md))
consumes the NVIDIA families as a **lead performance target** (Decision #28 — its
`wgmma`/`mma.sync` candidates set the ceiling and are never capped by the shared
mixer framework). It adds candidates under existing families, opening no new
NVIDIA-TEST item: channel-wise KDA/GDN decode → **NVIDIA-TEST-3/-5 KV/ReplaySSM**;
`sliding_window`/full mixer fwd + backward → **attention** (split/reduced dK/dV,
G6-C-style); chunkwise-scan inner GEMMs → **GEMM/Tile** (`wgmma` sm_90 / `mma.sync`
sm_120, preferably via the NVIDIA Tile IR lowering target); NVFP4/MXFP8 mixer GEMMs
→ **NVIDIA-TEST-4** numerical policy (this is the executing FP4 lane — sm_120
`mma.sync`, not `tcgen05`). Inherits the TEST-3 native-provenance / TEST-5
kernel-vs-E2E evidence contract unchanged. Direction pointer only; no NVIDIA gate,
route, or exact-device claim changes here.

Cross-backend sync `X86-E2E2-COHORT2-2026-07-20` adds shared typed Tile
carriers for argreduce, inclusive scan, unweighted row normalization,
interleaved-pair RoPE, and ALiBi. NVIDIA parity is assessed at the semantic
carrier boundary only. AVX-512 ABIs, CPU schedules, Ryzen timing, and route
disposition transfer no PTX/CUDA implementation, device evidence, or selector.

Cross-backend sync `X86-E2E2-BREADTH-2026-07-20` adds an explicitly x86-owned
`tile.x86_abi_kernel` and cohort-3/4 C-ABI registry. It changes no portable
semantic Tile carrier, PTX/CUDA ABI, NVIDIA schedule, dtype capability,
execution row, or selector. NVIDIA parity is therefore not applicable.
X86-E2E-2 is now closed with measured x86-only selector thresholds; this does
not change the NVIDIA not-applicable disposition or transfer device proof.

Cross-backend sync `LLVM23-LOCAL-CLEANUP-2026-07-20` hardens the LLVM 23 and
Linux TSAN host environment and repairs an Apple-only capability row. NVIDIA
parity is not applicable: no CUDA dtype, PTX ABI, schedule, execution row,
selector, or exact-device evidence changes.

Cross-backend sync `ROCM-E2E-SPINE3-TEST1-2026-07-21` adds shared paged-KV and
MoE fixture identities to the E2E-SPINE-3 corpus. NVIDIA fixture-schema parity
is validated, but the gfx1151 HSACO, HIP ABI, resources, timing, and
exact-device evidence do not transfer to CUDA. The ROCm-owned compiler lane
explicitly excludes `compiler_nvidia`; no NVIDIA capability, test ownership,
schedule, execution row, or selector changes.

Cross-backend sync `CORE-COMPILER-1-2026-07-22` closes shared Graph/Neighbors
verifier gaps and records the shared `sm_120` MMA selection in NVIDIA manifest
rows. Equal-tier candidates may use its analytical accumulator footprint only
after route-tier precedence. This is parity validated at the host-free
compiler/manifest boundary; it changes no PTX instruction schedule, automatic
selector promotion, CUDA ABI, or exact-device evidence.

Cross-backend sync `CORE-COMPILER-2-2026-07-22` makes compute dtype
legalization the default in NVIDIA named pipelines. Terminal storage
legalization remains intentionally opt-in for the generic value-level CUDA
route because it has no block-scale operand ABI. The later
`NVIDIA-BF16-CANONICAL-BREADTH-2026-07-25` continuation closes the physical
consumer gap for the scale-bearing NVFP4/MXFP4/FP6 launch envelopes and wires
StoragePackConsume after opt-in terminal legalization; at that point generic
INT4 and a sub-byte default remained unsupported. The executable row-major layout
materializer and guarded
dynamic launch are x86-only and transfer no PTX schedule, CUDA ABI, bucket
policy, selector, or exact-device evidence.

Cross-backend sync `CORE-COMPILER-NEXT-2026-07-22` tightens shared Graph layout
propagation through agreed-layout pointwise chains and last-axis reductions,
preserves packed-storage attributes, and records source-layout provenance on
inserted casts. At that synchronization point the architecture-owned
Graph-cast materializer was open; it landed in the later NVIDIA continuation
recorded below. The pass stays opt-in and transfers no PTX layout, schedule,
selector, or device proof. The x86 dynamic last-axis reduction guard
is not applicable to bucketed tensor-core routes. Shared add/multiply/static-
broadcast adjoints change Graph IR only; no CUDA backward runtime or exact-
device promotion is claimed.

Cross-backend sync `CORE-COMPILER-FOLLOWON-2026-07-22` adds shared kind-aware
sum/mean, GELU/SiLU, and softmax Graph adjoints with host CPU oracle proof.
Dynamic mean, max/min, ReLU, and normalization remain explicit fallbacks for
the documented Graph-contract reasons. Guarded dynamic softmax, attention, and
growing KV-cache execution are x86-only and are not applicable to bucketed
tensor-core routes; no CUDA ABI, schedule, selector, backward runtime, or
exact-device claim transfers. NVIDIA's architecture-owned Graph-cast consumer
is host-validated: after shared legality it accepts row/column-major/BHSD/NHWC,
removes the Graph marker, and carries the binding into `tile.async_copy`.
This changes staging metadata only and claims no PTX schedule or device proof.

Cross-backend sync `CORE-COMPILER-ADJOINTS-2026-07-22` registers shared
tensor-to-i1 comparison contracts plus internal scalar-threshold,
rank-reduced normalization-statistics, and explicit broadcast-in-dimension
Graph carriers. ReLU and unweighted RMSNorm/LayerNorm paired adjoints are
static/dynamic Graph-native and CPU-IR oracle-proven; the static shared path
lowers through linalg. This shared sync added no PTX/CUDA execution; the
architecture-owned affine backward ABI, runtime binding, and exact-SM120 proof
land in `CUDA-TRAINING-MEMORY-FOUNDATION-2026-07-24` below. It does not imply a
tensor-core schedule or selector promotion.

Cross-backend sync `CORE-COMPILER-NORM-AFFINE-2026-07-22` makes integer
comparison signedness explicit in shared Graph IR and adds dynamic-dimension
carriers plus channel-affine RMSNorm/LayerNorm adjoints. The NVIDIA dynamic
affine materializer and backward runtime were still open at this sync and are
closed by the architecture-owned continuation below. The gfx1151 HSACO and
AVX-512 ABIs, schedules, and timings did not transfer to CUDA/PTX; no selector
promotion was inferred from sibling evidence.

Cross-backend sync `CORE-COMPILER-NORM-BWD-DETERMINISM-2026-07-22` changes only
the ROCm architecture-owned backward schedule and temporary-buffer ABI. The
shared affine adjoint and f32 accumulation contract are unchanged. The
CUDA/PTX backward materializer and exact-device proof were supplied later by
the NVIDIA-owned continuation below; the gfx1151 two-kernel schedule, bitwise
evidence, and timing did not transfer.

Cross-backend sync `CORE-COMPILER-NORM-BWD-2026-07-22` adds family-specific
RMSNorm/LayerNorm backward execution rows and public JIT binding for ROCm and
x86. The then-open NVIDIA execution row is now closed by the CUDA/PTX
continuation below. Neither the gfx1151 HSACO ABI nor the AVX-512 ABI,
schedule, timing, or device evidence transferred; the NVIDIA implementation
retains its own descriptor and exact-device proof.

Cross-backend sync `CORE-COMPILER-LAYOUT-AUTODIFF-MEMORY-2026-07-23` completes
the shared transpose/packed epilogue/reduction layout envelope and adds native
guarded-dynamic broadcast, runtime-extent mean, and equal-share max/min Graph
adjoints. NVIDIA parity is host-validated through the shared linalg contract.
All NVIDIA backend variants now execute Tile buffer reuse and materialize one
address-space-3 shared-memory arena with typed planned-offset views before
Tile-to-NVIDIA/NVVM lowering. Function-budgeted liveness-aware
rematerialization also runs in the shared production post-autodiff pipeline.
Exact CUDA/PTX shared-allocation assembly and occupancy were open at this
shared sync; the static and dynamic CUDA evidence is recorded in the NVIDIA
closeouts below. No selector promotion is implied.

Cross-backend sync `CORE-COMPILER-TRAINING-SPINE-2026-07-23` registers
`tessera.loss.mse` and its paired backward carrier as verifier-checked shared
Graph IR, with dynamic none/sum/mean Linalg lowering and FP32 compute for
FP16/BF16 storage. Shape-preserving MSE participates in shared layout
propagation, and post-autodiff rematerialization now distinguishes saved
forward activations from backward temporaries. NVIDIA parity is host-validated
at the shared IR boundary. The gfx1151 HIP composition/module cache and
AVX-512 execution do not transfer to CUDA/PTX. The NVIDIA-owned compiled MSE
backward launch and exact-device evidence land in the continuation below; no
tensor-core training schedule or selector promotion is claimed.

Cross-backend sync `CORE-COMPILER-DEEPENING-2026-07-23` adds shared
runtime-sized address-space-3 arena planning and a benchmark-fed
rematerialization cost contract. NVIDIA retains opt-in Graph layout assignment
through its existing materializer. The new MSE backward launch and numerical
proof are ROCm gfx1151-only. The architecture-owned CUDA VJP and dynamic
shared-allocation/occupancy proof land in the later NVIDIA closeouts; selector
evidence remains explicitly non-promoting on WSL.

Cross-backend sync `CORE-COMPILER-TRAINING-BREADTH-2026-07-23` adds shared
Graph-native MAE, Huber, SmoothL1, and SGD adjoints with dynamic Linalg and CPU
oracle proof. The architecture-owned CUDA/PTX backward materialization and
exact-device evidence land in the continuation below. The gfx1151 generated
HIP kernel, AVX-512 C ABI, boundary timing, caches, and selector state did not
transfer.

Cross-backend sync `CORE-COMPILER-TRAINING-SERIES-2026-07-23` adds shared
Graph-native stable BCE-with-logits, class-index/label-smoothed cross entropy,
KL/JS, explicit Momentum/Nesterov state, and explicit Adam/AdamW moment-state
adjoints. Dynamic shared Linalg contracts are live for BCE, Momentum/Nesterov,
and Adam/AdamW. The NVIDIA CUDA/PTX materializers and exact-device evidence
land in the continuation below, including KL/JS and FP16/BF16 storage. The
gfx1151 and AVX-512 loss and optimizer ABIs did not transfer.

Cross-backend sync `CORE-COMPILER-TRAINING-FUSION-2026-07-23` adds shared
single-use loss-backward to SGD/AdamW fusion carriers and one-loop dynamic
Linalg lowering for MSE, MAE, Huber, SmoothL1, and BCE-with-logits. This sync
validated only the shared Graph/Linalg contract; the architecture-owned
CUDA/PTX fused materializer and exact-device evidence land below. gfx1151 HIP
and AVX-512 ABIs, cache identities, timings, and selector decisions did not
transfer.

Cross-backend sync `CORE-COMPILER-MEMORY-LAYOUT-CLOSEOUT-2026-07-23` replaces
the shared static address-space-3 alloca with a workgroup global and supports
dominance-scoped dynamic arena cohorts. At that point exact CUDA assembly,
resource, occupancy, and performance evidence were open and not inferred from
gfx1151; the NVIDIA-owned static/dynamic closeouts below now provide them.
No NVIDIA selector or default policy changes.

Cross-backend sync `CORE-COMPILER-HONEST-BOUNDARIES-2026-07-23` broadens the
shared measured-rematerialization schema to exact consumer chains and
64/128/192 matmul shapes with ReLU/GELU/SiLU. The later NVIDIA packets provide
CUDA measurements; native-Linux policy selection remains deferred. ROCm dynamic
normalization epilogues, HIP launch-sized LDS materialization, and packed IU4
WMMA are architecture-owned and transfer no PTX ABI, shared-memory allocation,
packed consumer, performance, or selector claim. The existing NVIDIA
architecture-owned layout consumer remains unchanged.

Cross-backend sync `CORE-COMPILER-HONEST-BOUNDARIES-2-2026-07-24` extends the
shared rematerialization corpus schema with softmax, RMSNorm, and MSE producer
families plus measured workload-budget decisions. The later NVIDIA packets
provide CUDA measurements; native-Linux policy selection remains deferred.
ROCm's packed
multi-arena LDS ABI, GELU normalization epilogue, and terminal-pack
dequant-GEMM consumer are architecture-owned; no PTX shared-memory ABI, packed
consumer, timing, selector, or support claim transfers. CUDA path-max and
general serialized launch expressions are closed below; physical packed
consumers remain governed by their own dtype rows.

Cross-backend sync `CORE-COMPILER-HONEST-BOUNDARIES-3-2026-07-24` extends the
shared rematerialization evidence schema to a measured four-layer workload with
softmax, RMSNorm, MSE, Huber, SmoothL1, and BCE instances. CUDA measurements
now land in the NVIDIA continuation below; controlled native-Linux policy
selection remains deferred. ROCm's
branch-path dynamic-LDS expression, binary normalization epilogues, and packed
elementwise/sparse/cache ABIs are architecture-owned; no PTX shared-memory
expression, packed ABI, timing, selector, or support claim transfers.

Cross-backend sync `CORE-COMPILER-CFG-MEMORY-BUDGETS-2026-07-24` adds a shared
model/device-derived rematerialization budget contract with explicit override
precedence and bounded dynamic parameters. The exact CUDA-context capacity,
free-memory cap, reserve policy, parameter bounds, and measured packet now land
in the NVIDIA continuation below. ROCm's alias-aware nested/loop LDS slots and 40,208-byte
gfx1151 packet are architecture-owned; no PTX shared-memory expression,
occupancy, execution, or selector claim transfers.

NVIDIA-owned closeout `E2E-SPINE3-SM120-MEMORY-2026-07-24` completes the
static NVPTX boundary named by `CORE-COMPILER-MEMORY-LAYOUT-CLOSEOUT`.
Tile lowering turns three logical 512-byte allocations with disjoint lifetimes
into one 1,024-byte address-space-3 arena at offsets `[0, 512, 0]`, reducing
the unreused 1,536-byte plan. LLVM 23 emits an exact 1,024-byte NVPTX shared
declaration; `ptxas` reports the declaration and the retained executable route
has matching tool-reported shared-resource accounting. Exact SM120 execution
compares that retained route with a register-rematerialized expression:
both return 42, have complete no-spill resource records and 100% theoretical
occupancy, and pass the 4% two-run gate in device-event and end-to-end domains.
The evidence is intentionally selector-ineligible. Dynamic/path-expression
arenas and model-level CUDA rematerialization policy remain separate follow-ups.

NVIDIA-owned continuation `CUDA-TRAINING-MEMORY-FOUNDATION-2026-07-24`
closes the named SM120 execution gaps for dynamic-affine RMSNorm/LayerNorm
backward; MSE, MAE, Huber, SmoothL1, stable BCE-with-logits, KL/JS, and
class-index/label-smoothed cross-entropy backward; deterministic general
broadcast-gradient reduction; SGD, Momentum/Nesterov, Adam, and AdamW updates;
and fused loss-backward plus SGD/AdamW. FP32, FP16, and BF16 storage preserve
their physical dtype while every arithmetic/reduction path accumulates in FP32. The
architecture-owned path generates CUDA, compiles an immutable PTX image,
binds a launch descriptor, and executes through the shipped CUDA-driver
bridge. Public runtime artifacts and `@jit(...).native_backward()` use the
same descriptor seam. Exact SM120 tests cover dynamic/ragged extents,
transition boundaries, extreme logits, none/sum/mean cotangents, optimizer
state, fused-versus-composed numerics, invalid labels, cache identity, and
live resources.

The same continuation completes CFG-forwarded and locally computed dynamic
shared-memory sizing. A post-LLVM NVIDIA pass replaces runtime-sized
address-space-3 byte arenas with slices of an external NVPTX shared symbol,
colors mutually exclusive lifetimes into one slot, keeps simultaneously live
arenas in distinct aligned slots, and serializes checked constant, argument,
add, multiply, cast, select, and path-max launch expressions. The v2 CUDA
launch ABI evaluates the descriptor expression and passes the resolved byte
count to `cuLaunchKernel`. Exact SM120 execution proves the original
12,289/32,001-byte branch paths and a CFG-forwarded
`max(4096*4+17, 12289)` expression with a 16,416-byte allocation, rejects
undersized descriptors, and retains driver-JIT register/static/local-memory/
occupancy evidence.

The native bridge also reports total/free bytes from the same retained CUDA
context. The compiler policy caps usable capacity by current free memory,
marks static or explicitly bounded dynamic Graph IR model parameters, and
stamps reserve, gradient-copy, optimizer-state, and persistent-state inputs
consumed by the shared activation-rematerialization pass.

The checked-in 26-row repeated-median packet discards the first/JIT launch,
uses 1,000 device-event repetitions and 20 end-to-end iterations, and retains
cold/warm image identity plus artifact and resource fingerprints. Fourteen
rows meet the 4% two-run gate in both timing domains on this WSL2 host; twelve
retain explicit unstable dispositions, with the two tiny dynamic-shared probes
remaining especially host-noisy. Fused MSE+SGD and MSE+AdamW are stable and
beat their composed references in both timing domains, but their references
are not stable and WSL is not the controlled native-Linux promotion host.
Therefore no production selector changes. Rerunning this exact packet on
controlled native Linux is the remaining performance/selector boundary.

NVIDIA-owned closeout `NVIDIA-BF16-CANONICAL-BREADTH-2026-07-25` extends the
compiler-owned SM120 Tile-image seam from FP16/FP32 to BF16 input storage with
FP32 accumulation and output for reduction, stable row softmax, and attention
forward/backward. The reduction contract covers sum, mean, max/min and
amax/amin aliases, arbitrary static axes, keepdims true/false, ragged extents,
NaN propagation, and both serial and cooperative-128 physical schedules.
Every BF16 descriptor carries a distinct typed ABI and two-byte input binding;
the CUDA-driver bridge retains four-byte FP32 result bindings. The canonical
Graph verifier, target capability, backend manifest, execution matrix, Tile
verifier, NVIDIA lowering, runtime registration, and launch bridge agree on
that boundary.

Exact RTX 5070 Ti SM120 execution proves 44 focused BF16 softmax, reduction,
and attention cases through MLIR to NVVM, immutable PTX image, descriptor, and
CUDA-driver launch. Compiler lit proves BF16 extension into FP32 arithmetic,
serial/cooperative reduction assembly, min lowering, and attention
forward/backward materialization. The checked-in 12-row reduction packet
compares both schedules across all six public kind spellings, discards the
first launch, amortizes 500 resident launches and 10 complete calls, and
retains two disjoint repeated-median cohorts, cold/warm image and entry-symbol
identity, numerical error, ptxas registers/shared-memory/spills, and live
driver local-memory/occupancy. WSL timing remains an explicit non-promotion
result: only 3/12 candidates meet the 4% gate in both timing domains and no
candidate has stable cross-domain winner consensus. Production selectors are
therefore unchanged; controlled native-Linux comparison remains required
before any schedule promotion.

The continuation of `NVIDIA-BF16-CANONICAL-BREADTH-2026-07-25` closes the
remaining normalization envelope. Compiler-owned immutable PTX images now
execute unweighted RMSNorm/RMSNorm-safe and LayerNorm with f32, f16, or bf16
storage, f32 row statistics/accumulation, immutable nonnegative epsilon, and
same-storage output. Eighteen exact SM120 cases cover ragged/non-power-of-two
rows, multiple ranks, both normalization kinds, all three storage types,
resource/no-spill inspection, warm image/cache/descriptor identity, and FP32
oracles. Dynamic affine BF16 backward remains on the previously proven
training descriptor, so forward and backward now share the BF16 storage
boundary without claiming an affine forward fusion in this unweighted kernel.

The same synchronization point lands NVIDIA's first physical consumer of the
shared terminal packing descriptor. Canonical NVFP4, MXFP4, FP6-E2M3, and
FP6-E3M2 launch images carry `tessera.storage_pack`; NVIDIA lowering requires
the logical format, int8 container, packing factor (2 for four-bit, 1 for
six-bit), and format-defined signedness to agree with the selected
scale-bearing fragment ABI before generating packed byte/nibble loads.
Descriptor drift rejects in compiler lit, while exact general-shape/ragged
SM120 tests traverse the descriptor-driven loaders. Opt-in terminal storage
legalization now runs `StoragePackConsume` in the NVIDIA named pipeline.
The later capability expansion admits only the proven value-level decode,
unscaled round-trip, and matching packed-matmul def-use paths. It does not make
terminal packing the default for arbitrary FP4/FP6 values or for Graph-level
quantize/dequantize operations.

NVIDIA-owned continuation `NVIDIA-PACKED-MATH-2026-07-25` adds the missing
canonical signed-INT4 consumer. Its compiler-owned Tile image and typed CUDA
launch ABI accept only an int8 container, factor two, signed two's-complement
nibbles, low logical index in the low nibble, i32 accumulation/output, and no
scale or fused-epilogue operands. The correctness-first general-shape schedule
decodes packed A rows and packed B columns, guards ragged M/N/K edges, and
rejects container, factor, signedness, scale/operand, and epilogue disagreement
before PTX materialization. Exact aligned and ragged RTX 5070 Ti execution
matches an integer oracle, retains zero ptxas spills, and proves cold/warm
image and descriptor identity. NVFP4/MXFP4/FP6 keep their distinct
scale-bearing fragment semantics; no physical schedule is transferred between
those formats or from ROCm.

The 2026-07-25 continuation replaces the dictionary-only pack record with
portable `#tile.packed_format`, `#tile.scale_layout`, and
`#tile.packed_view` attributes plus generic `tile.packed_load`/
`tile.packed_store`. The contract records logical bit width independently of
the int8 container (notably FP6 factor one), signedness/encoding/lane order,
packing axis/strides, alignment/offset, and an explicit scale operand/layout.
SM120 lowering decodes signed INT4, FP4/NVFP4, and both FP6 encodings, applies
origin-aware scale indexing for non-origin views, guards ragged bounds, and
supports unscaled packed round trips. Descriptor, axis, scale, alignment, and
store disagreement fail closed. Terminal marking is capability-filtered by
target + operation + format + available consumer; unsupported operations stay
logical rather than inheriting launch-envelope support.

The generic value-level consumer now also owns a CUDA runtime ABI rather than
ending at compiler lit. Exact RTX 5070 Ti execution covers non-origin/ragged
NVFP4 with explicit UE4M3 scale binding, signed INT4, and FP6-E2M3/FP6-E3M2
with explicit UE8M0 scale binding. All four cases match their format oracles,
retain zero ptxas spill bytes, and reproduce cold/warm image plus launch
descriptor identity. Source/scale byte extents are launch scalars checked
against the serialized physical view; the bridge rejects negative origins,
nonpositive extents, overflow, or buffer/descriptor disagreement. Generic
packed stores retain host-free compiler round-trip proof; no unsupported
format is default-enabled by this runtime addition.

The checked-in SM120 packed-storage packet measures the production-sized
`4097x4099` ragged view for signed INT4, NVFP4, FP6-E2M3, and FP6-E3M2. Each
device observation amortizes ten resident launches, each cohort retains seven
device-event observations and ten end-to-end submissions, and the first
allocation/copy/JIT-inclusive submission is discarded. Every row reproduces
its compiler image, launch descriptor, cache fingerprint, and 18-register,
zero-local-memory resource record. On this WSL host all four rows meet the 4%
two-run gate in both timing domains: the device-event deltas are 0.62%, 0.57%,
0.32%, and 0.08%, while the end-to-end deltas are 0.27%, 0.07%, 3.46%, and
0.07%. WSL evidence remains selector-ineligible, so every row records an
explicit retain disposition. No selector or terminal-legalization default
changes from these timings.

CUDA Math now crosses a registered `tessera_nvidia.cuda_math_kernel` seam
instead of extending an open five-pointer string dispatcher. Its closed
verifier covers scalar `brev`, `prmt`, `clz`, `ffs`, `popc`, numeric and
bit-preserving f32/i32 casts, packed 2x16 and 4x8 signed/unsigned wrapping or
saturating arithmetic, byte absolute difference, and both lane-mask and
predicate-bit comparisons. All 27 exact RTX 5070 Ti cases launch, match
bit-exact/numerical oracles, retain zero spills, and reproduce warm image and
descriptor identity. No production selector changes.

The structural continuation registers `!tile.buffer`, `tile.alloc`,
`tile.dealloc`, `!tile.pipeline_state`, `tile.pipeline_init`, and
`tile.pipeline_advance`. It now also registers SSA TMA descriptor, mbarrier,
mbarrier-token, TMEM, and TCGen05 types/operations. WarpSpecialization allocates
SMEM/TMEM handles in the parent region, threads them into staged copies and
consumers, threads producer/consumer pipeline states, and deallocates only
after CTA synchronization. AsyncCopy lowering emits registered TMA descriptor
and copy operations; descriptor deduplication assigns slots, creates one SSA
mbarrier, binds it to every copy, and threads copy completion tokens into the
typed wait. FlashAttention emits typed arrive/try-wait token chains.
Barrier-reuse legality passes on this real WarpSpec output. NVIDIA no longer
emits or consumes name-based `#tile.buffer_ref`; the shared reader remains only
for Apple/ROCm migration fixtures. Annotation-only `#tile.pipeline_state`
compatibility metadata is rejected, and the structured Schedule→Tile layout is
consumed directly. TCGen05/TMEM has host-free structural and SM120 fail-closed
proof only; exact execution remains SM100-owned and cannot be inferred from
consumer Blackwell.

Cross-backend sync `ROCM-TRAINING-MEMORY-FUSION-2026-07-27` adds ROCm-owned
Adam/AdamW and KL/JS physical backward execution plus a ROCm normalization
softcap epilogue; none of those HIP kernels, gfx1151 timings, or selector
evidence transfers to CUDA. The shared change is the target-neutral,
serializable dynamic-local-memory expression field on `LaunchDescriptor`.
NVIDIA's existing SM120 add/multiply/path-max/alignment probe now consumes
that field and retains its CUDA-owned PTX, launch-v2, resource, and exact-device
evidence. No NVIDIA execution row or selector changes.

Cross-backend sync `ROCM-LION-BACKWARD-2026-07-27` adds only the ROCm-owned
physical consumer of the already-shared Lion stop-sign VJP policy and extends
the gfx1151 operation-total benchmark packet. HIP code objects, AMD launch ABI,
and WSL timings do not transfer to CUDA. NVIDIA remains follow-up required for
an architecture-owned compiled Lion backward materializer; no SM120
capability, execution row, PTX schedule, or selector changes.

**NVIDIA follow-on update (2026-07-30):** the CUDA-owned Lion materializer now
has landed as `nvidia_lion_bwd_compiled`: an SM120 PTX package with an explicit
eight-buffer ABI (`p/g/m`, two output cotangents, and `dp/dg/dm`), CUDA bridge
layout, runtime executor, execution-matrix row, f32 oracle, and RTX 5070 Ti
device validation. It implements the shared stop-sign VJP without importing an
AMD code object or schedule. This is a correctness-first 128-thread package;
no optimizer selector changed and a timing packet remains a separate SM120
measurement task.

Cross-backend sync `CORE-SCHEDULE-1F1B-MATERIALIZE-2026-07-27` emits a shared
unique-clock warmup/steady/cooldown dependency order after pipeline legality.
At this synchronization point CUDA runtime consumption and collective overlap
remained NVIDIA-owned follow-up; the immediately following
`CORE-COMPILER-RUNTIME-CLOSEOUT-2026-07-27` record supersedes that structural
gap with the shared runtime consumer. The carrier itself changes no SM120
capability, PTX schedule, selector, or exact-device claim.

Cross-backend sync `CORE-COMPILER-RUNTIME-CLOSEOUT-2026-07-27` supplies a shared
runtime consumer for emitted 1F1B steps with independent collective transport,
and makes measured schedule records alter the actual Schedule/Tile M/N/K,
warp-count, and stage attributes after target/evidence validation. NVIDIA's
named pipelines now default layout assignment on because the architecture-owned
Graph-cast materializer immediately consumes the markers; focused structural
proof covers ordering. A real CUDA multi-rank transport packet and measured
SM120 selector application remain NVIDIA-owned exact-device follow-ups.

The same sync replaces DeltaNet-family finite-difference reverse mode with an
analytic carried-state recurrence and explicit reverse-token scheduling.
Directional-derivative fixtures validate shared semantics only; CUDA backward
packaging and device scheduling remain follow-up required. ROCm factored
Adafactor HSACO, HIP capacity query, gfx1151 numerics, and WSL timing do not
transfer to CUDA.

Cross-backend sync `CORE-PRODUCTION-EVIDENCE-2026-07-27` serializes collective
descriptors on emitted pipeline steps and binds OptimizerShard ownership
transitions to the shared runtime. NCCL remains the CUDA-native executor, but
this ROCm-host continuation contains no real multi-rank CUDA packet and changes
no SM120 selector. The gfx1151 Adafactor adjoint and two-entry DeltaNet
reverse-chunk HSACO (later superseded by the five-entry AMD package) are AMD
schedules and do not transfer. CUDA sequence-mixer
backward packaging and a refreshed measured selector packet remain
NVIDIA-owned exact-device follow-ups.

Cross-backend sync `CORE-SEQUENCE-MIXER-PHYSICAL-BACKWARD-2026-07-28` adds the
exact modified-Delta normalization VJP to physical ROCm and AVX-512 backward
paths and proves an affine parallel chunk-composition algorithm for
`erase=false`. This changes shared algorithm evidence, not CUDA execution. The
five-entry gfx1151 HSACO, AMD workgroup schedule, and WSL resident timings do
not transfer to SM120. CUDA sequence-mixer backward packaging, nonlinear/erase
chunk scheduling, and a refreshed exact-CUDA-host selector packet remain open.

**CUDA DeltaNet reverse architecture (2026-07-30):** NVIDIA owns a separate
four-stage package: (1) fp32 state checkpoints per `(batch, head)` trajectory;
(2) an erase-free affine chunk-summary/prefix path; (3) a serial nonlinear or
`erase=true` checkpoint-fill path; and (4) reverse-token gradients with unique
`(batch, head)` ownership for Q/K/V/gate/beta/decay. The modified-normalization
derivative is part of stage 4, not a reuse of an AMD or AVX implementation.
The current CUDA forward recurrence is intentionally unchanged. Do not add a
`nvidia_deltanet_bwd_compiled` execution row or selector candidate until this
CUDA package has exact f32 numerical tests plus SM120 device-event and
end-to-end timing cohorts; ROCm/x86 packets are non-evidence for that decision.

**Implementation update (2026-07-31):** the versioned v2 CUDA package now
keeps its fixed 13-buffer/10-scalar ABI while executing the validated bounded f32
`Dqk,Dv <= 8` reverse recurrence. It has CUDA-owned analytic gate, beta, and
decay derivatives, followed by serial replay for `erase` and modified-update
normalization; direct-package and public JIT exact-device tests compare all six
gradients against the shared oracle. This is correctness evidence only: no
execution-matrix promotion or selector choice has been made until the required
SM120 timing and counter packet exists. Apple, ROCm, and x86 are **not
applicable to this physical package**; their differently scheduled packages
and evidence remain sibling follow-ups rather than CUDA proof.

**Timing/NCU packet (2026-07-31):**
[`nvidia_sm120_deltanet_backward_2026_07_31.json`](../../../../benchmarks/baselines/nvidia_sm120_deltanet_backward_2026_07_31.json)
records two repeated CUDA-event and end-to-end cohorts at `[B,H,S,Dqk,Dv] =
[1,2,16,8,8]`: plain `0.64318 / 1.53015 ms`, affine `0.76717 / 1.54674 ms`,
and erase+modified serial-fill `0.92778 / 1.69752 ms`. All variants match the
oracle (maximum errors `7.45e-09`, `1.86e-09`, and `1.35e-04`). The packet
also records live resources (255 registers/thread, 1328 B local/thread, two
active blocks/SM) and two NCU L2/DRAM cohorts. Only serial-fill counters were
stable (94.77/94.70% L2 and 779,776/715,008 DRAM B); plain and affine counter
cohorts diverged materially, so they are evidence of collection instability,
not a schedule result. The packet changes no selector: there is one
correctness-first implementation and no controlled alternative to promote.

## Cross-backend sync `TILE-FRAGMENT-TYPE-PARAM-2026-08-03` — `!tile.fragment` parameterized (W1.1 step 1)

Shared Tile IR type changed: `!tile.fragment` gained `(m, n, k, elem, acc, role, layout, family)` and a domain verifier. **No behaviour changes in this PR** — the bare `!tile.fragment` still parses AND still prints bare, so every existing producer and fixture is unaffected. All 7 C++ `FragmentType` uses are `isa<>` checks, so there were no construction sites to migrate.

**Outcome: follow-up required.** 8 files under this backend reference `FragmentType` / `!tile.fragment`. Two obligations, both already scoped in [`W1_1_TYPING_DESIGN.md`](../../compiler/W1_1_TYPING_DESIGN.md):

* **Step 2b (blocking for the motivating GEMM).** `NVIDIALowering.cpp` requires the accumulator's direct defining op to be `FragmentZeroOp` (3 sites). A K-loop accumulator is an `scf.for` iter-arg with no defining op, so a typed K-loop will verify and still fail to lower until this is block-argument-aware. Needs a per-backend *lowering* fixture, not just a verifier one.
* **Step 3.** `GenerateWMMA*`-equivalent producers migrate to the typed form one PR at a time.

`family` is a type parameter partly because `NVIDIALowering.cpp` gates on it before matching (m, n, k, dtype) to an `mma.sync` variant — leaving it on the attribute would have let a fragment packed for one family feed an op selecting another.

No exact-device evidence in this PR; none required, since no generated code changed.

## Cross-backend sync `TILE-FRAGMENT-KLOOP-ACCUM-2026-08-03` — typed `tile.mma` K-loop (W1.1 step 2)

Shared Tile IR contract changed: `MMAOp::verify()` (and the `fragment_pack` / `fragment_zero` producers) now read the operand contract from the fragment TYPE when it is parameterized, falling back to producer-chasing for the bare form. `#tile.mma_desc` is optional on the typed path and cross-checked when present. **The canonical K-loop now verifies.** No lowering changed in this PR, and no existing IR is affected — the bare form keeps its old path.

**Outcome: follow-up required — and larger than previously recorded.**

Same finding as ROCm under this key. `NVIDIALowering.cpp` synthesizes zero constants for the accumulator (`operands.append(4, zero)` for f32, and the f16/s32 equivalents) and never reads `mmaData[2]` as a value. So step 2b is accumulator threading plus an `scf.for` region-signature conversion, not a relaxed check — and relaxing it alone would emit a silently wrong GEMM.

No sm_120 device evidence in this PR; no generated code changed. When 2b lands, its gate needs numerics on real hardware for the same reason ROCm's does.

## Cross-backend sync `NVWGMMA-ACCUMULATOR-GUARD-2026-08-03` — WGMMA accumulator drop (W1.1 step 2b guard)

A `tile.mma` carrying an accumulator was lowered by `NVWGMMALoweringPass` to a **two-operand** WGMMA call: the accumulator was discarded, the shape hardcoded `m64n64k16`, and the dtype inferred through `dyn_cast<ShapedType>` (which a `!tile.fragment` is not, so it defaulted to bf16) — with **rc=0 and no diagnostic**. A K-loop recomputed A×B from nothing each step and returned the last partial product.

Measured on merged main, this was **not** specific to the typed fragment form: a legacy bare `tile.mma(A, B, C)` — what `LowerKReductionAddToTileMMA` emits for the canonical K-step — was dropped identically. **No fixture in the tree covered either case**, which is how it survived. The guard therefore keys on *has an accumulator*, not *is typed*.

**Outcome: follow-up required — this backend owned the defect.**

`NVWGMMALoweringPass` now refuses such an mma with `NVWGMMA_ACCUMULATOR_DROPPED` and calls `signalPassFailure()`. The check lives in the PASS BODY, not the pattern: a pattern that emits an error and returns `failure()` only declines to match, so the diagnostic printed while the tool still exited 0 and the pipeline continued — an error that does not fail compilation is a warning in disguise, the same fail-open shape as the bug.

The two-operand lane is untouched: `nvwgmma_lowering.mlir`'s emitted call is byte-identical before and after (diffed, not assumed).

**Follow-up:** W1.1 step 2b threads the accumulator for real, which needs an `scf.for` region-signature conversion. This guard is replaced by lowering then, not relaxed. No sm_120 device evidence here; no working codegen changed, only a refusal added.

## Cross-backend sync `ROCM-COMPILED-STRICT-DISPATCH-2026-08-04` — compiled-lane failures stop masquerading

Runtime dispatch contract changed. A compiled-ROCm **failure** (tessera-opt ran and serialized no kernel, or emitted a non-ELF blob) now routes through the existing `_note_dispatch_fallback` funnel, so `TESSERA_STRICT_DISPATCH=1` raises instead of degrading. **Envelope limits** (no libamdhip64, hipInit failed, tessera-opt not built, dtype/rank/arch out of range) are unchanged and still degrade silently — making those raise would break strict runs on every CPU-only host.

Measured before the fix: a deliberately broken pass pipeline returned `ok=True, compiler_path="rocm_compiled", execution_kind="native_gpu"` with correct numbers. Strict-mode suite results are identical before and after (18 fail both ways, all pre-existing), so this adds no new failures.

**Outcome: not applicable — architecture-specific reason.** The changed sites are all `_RocmCompiledUnavailable` raise points inside ROCm compiled-lane hsaco builders. NVIDIA's compiled lanes do not raise that exception and are untouched.

**Follow-up worth recording, not created here:** NVIDIA has no equivalent failure/envelope split on its own compiled paths, so the same masking may exist there. Establishing that needs an sm_120 host, which this box is not — asserting it either way from here would be the guesswork this thread has been eliminating.

## Cross-backend sync `ROCM-PIPELINE-TILE-LOWERING-2026-08-04` — the compiled pipeline can lower `tile.mma`

Both ROCm compiled pipelines (plain and canonical) now run `lower-tile-to-rocm{arch=<chip>}` after `generate-wmma-gemm-kernel`. Verified byte-identical hsaco with and without the pass on the default path, so the production lane is unchanged.

**Outcome: follow-up required — recorded, not fixed here.** The equivalent NVIDIA seam is worse, not merely missing: `NVWGMMALoweringPass` lowered a `tile.mma` carrying an accumulator to a two-operand call and dropped it (`NVWGMMA-ACCUMULATOR-GUARD-2026-08-03`). That is guarded to fail closed; threading it for real is W1.1 step 2b on this backend and needs an sm_120 host for the numeric gate that ROCm just got.

## Cross-backend sync `TILE-VIEW-BOUNDED-CONTRACT-2026-08-04` — bounded `tile.view` is a shared contract

`ViewOp::verify` now defines the pointer-backed operand contract: exactly 3 `(base, rowOrigin, colOrigin)` or 5 with `(rowBound, colBound)`. It previously accepted any count >= 3, so a 4-operand view was legal and meaningless and the bounded form's validity was decided by whichever backend looked.

**Outcome: follow-up required — refuses the bounded form, and the refusal is UNVERIFIED on this box.**

`NVIDIALowering`'s fragment materializer emits an unguarded load, so ignoring `(rowBound, colBound)` would read past the edge of a ragged matrix. It now emits `NVFRAGMENT_BOUNDED_VIEW_UNSUPPORTED` naming op and target (Decision #21) instead of folding the case into the generic arity message, which would have read as "malformed IR" for IR that is well-formed and merely unsupported here.

**Explicitly not verified:** the NVIDIA dialect is off by default in this build, and neither `--tessera-lower-to-gpu` nor `--tessera-nvidia-pipeline-sm120` reached the materializer with a bounded view on this host — no diagnostic, no error. The code path is written and compiles; it has not been executed. Verifying it needs a build with `-DTESSERA_ENABLE_CUDA=ON` and, for the numeric half, an sm_120 host.

Until this materializer grows masking, a portable producer must emit the 3-operand form; only ROCm can consume the bounded one.

## Cross-backend sync `TILE-VIEW-LINEAR-BASE-2026-08-05` — should `tile.view` carry a precomputed linear base?

ROCm W1.1 step 3 (`W1_1_TYPING_DESIGN.md` §4.7) established that isolated
fragment address derivation could not express the direct lane's shared row
offset. Measurement selected an optional precomputed `linear_base` operand on
`tile.view`; logical row/column origins remain present for bounds.

**ROCm has implemented and measured the shared answer.** `tile.view` now accepts
an optional precomputed `linear_base`, and the ROCm producer uses it for A-base
hoisting and sibling-B address sharing. At 2048^3 the final rebuilt ratio is
0.711x (15.45/21.74 TFLOP/s), still insufficient for promotion. The
remaining ROCm evidence points to fragmented load scheduling and excess waits.

**Outcome for NVIDIA: FOLLOW-UP REQUIRED.** This backend consumes `tile.view`
and `tile.fragment_pack` (8 files each), so it must consume the optional base
form correctly or fail closed when a producer selects it. No exact-device
evidence from an NVIDIA host is claimed. This is also the owner of W1.1's final two untyped C++ `tile.mma`
construction sites: both are tensor-valued producers in
`TileIRLoweringPass`. They must migrate together with NVIDIA's typed
tensor-to-fragment materializer and accumulator-threading gate; ROCm/x86
hardware cannot validate that physical decision.

## Cross-backend sync `TILE-DYNAMIC-LEADING-DIM-2026-08-04` — generic typed fragment addresses

Shared `tile.view` / `tile.store` can now carry an SSA leading dimension when
`#tile.memory_layout` states zero. **Outcome for NVIDIA: FOLLOW-UP REQUIRED.**
The sm_120 fragment materializer fails closed on this valid shared form and
still requires a static leading dimension. No CUDA-enabled build or sm_120
device evidence is claimed from this ROCm/x86 host.

## Cross-backend sync `E2E-REAL-LINEAGE-SCHEDULE-2026-08-05`

Shared compiler orchestration now records explicit artifact ancestry and
production `tessera-opt` registers the generated Schedule dialect. **NVIDIA
outcome: follow-up required.** SM120 packaging still consumes `GraphIRModule`
and synthesizes NVIDIA-owned Tile, so its lineage truthfully records the Graph
fork and remains incomplete. This change does not migrate the two untyped
`tile.mma` producers, accumulator threading, bounded/dynamic views, PTX, or a
physical schedule; no SM120 evidence is claimed. Those remain NVIDIA-owned
gates after the shared x86/ROCm vertical slice establishes the consumer API.

## Cross-backend sync `E2E-REAL-SCHEDULED-MATMUL-2026-08-05`

Shared Graph→Schedule→launch-Tile lowering is now real for the initial x86-f32
and ROCm-f16/f32 matmul instances. **NVIDIA outcome: follow-up required, not
validated by this slice.** SM120 is deliberately outside the first bounded
dtype/descriptor selector and therefore fails closed rather than inheriting
another architecture's schedule. NVIDIA packaging still synthesizes its Tile
program from Graph IR. Its later consumer must accept the canonical scheduled
artifact only after the final two untyped `tile.mma` producers, accumulator
threading, and dynamic/bounded view gates pass in a CUDA-enabled SM120 lane.

## Cross-backend sync `E2E-REAL-PHYSICAL-CONSUMERS-2026-08-05`

The shared package boundary is now concrete: a validated
`ScheduledMatmulArtifact` carries exact Graph, Schedule, and launch-Tile text
plus content identities into x86 and ROCm physical consumers. **NVIDIA outcome:
follow-up required.** No CUDA code changed and no SM120 evidence is claimed.
The NVIDIA consumer remains blocked on its two untyped `tile.mma` producers,
accumulator threading, dynamic/bounded view support, and CUDA-enabled SM120
validation before it can adopt this boundary without recreating Graph intent.

## Cross-backend sync `E2E-REAL-PERFORMANCE-2026-08-05`

Schedule/Tile matmul now distinguishes instruction-tile shape from an
architecture-owned macro tile and carries that value through artifact identity
and launch provenance. **NVIDIA outcome: follow-up required.** This generic
contract is applicable, but no CUDA lowering, SM120 schedule, PTX, selector, or
device evidence changed. NVIDIA must choose its own macro tile after its two
untyped producers, accumulator threading, and bounded/dynamic view gates land;
the gfx1151 32x64 decision does not transfer.

## Cross-backend sync `E2E-REAL-SEMANTIC-KERNELS-2026-08-05`

The shared spine now has content-addressed `schedule.softmax` and
`schedule.reduce` SSA edges and atomically lowers the bounded canonical f32
contracts to launch-level Tile artifacts. **NVIDIA outcome: follow-up required
on a CUDA-enabled SM120 host.** Existing NVIDIA physical Tile softmax/reduction
lowering is unchanged, but SM120 packaging still synthesizes its Tile program
from `GraphIRModule`; no PTX, cubin, descriptor, schedule, selector, or device
claim changed. Its consumer must accept the exact scheduled artifact and run
the established SM120 numerical/performance gates. x86/gfx1151 schedules and
evidence do not transfer. Canonical Graph reduction currently excludes
mixed-output and keepdims forms; widening that shared contract is separate
from adopting this first f32 boundary.

## Cross-backend sync `E2E-REAL-ATTENTION-2026-08-05`

The shared spine now defines a content-addressed `schedule.attention` edge and
one launch-level Tile artifact for the bounded x86/gfx1151 instances.
**NVIDIA outcome: follow-up required on a CUDA-enabled SM120 host.** Existing
SM120 forward packaging still synthesizes an NVIDIA-owned Tile program from
Graph IR; no PTX, cubin, LSE policy, schedule, selector, or device evidence
changes here. NVIDIA must define its own schedule/LSE instance, consume the
exact artifact, and run its forward numerical/performance gates. x86 and
gfx1151 policy or evidence does not transfer.

## Cross-backend sync `E2E-REAL-ATTENTION-BACKWARD-2026-08-05`

The shared spine now defines a content-addressed three-result
`schedule.attention_backward` program carrying dQ, split-dK/dV, fixed reduction,
workspace, and LSE checkpoint identity. **NVIDIA outcome: follow-up required on
a CUDA-enabled SM120 host.** Existing SM120 backward packaging remains
NVIDIA-owned Graph-to-Tile synthesis. NVIDIA must define its own LSE identity,
consume this exact artifact, and validate MHA/GQA/MQA plus modifier/ragged
coverage before claiming parity; x86/gfx1151 schedules and evidence do not
transfer.

## Cross-backend sync `ROCM-TYPED-EXECUTABLE-PIPELINE-2026-08-07`

The shared orchestration direction now has a concrete typed configuration:
family, input artifact level, output artifact level, architecture, Tile
producer, Target-IR consumer, and backend code generator. **NVIDIA outcome:
follow-up required.** CUDA must define its own SM-specific family plugins around
the canonical Schedule/Tile artifact and NVVM/PTX/cubin code generator; this
change does not transfer gfx1151 scheduling or AMD wait semantics and supplies
no CUDA-enabled or SM120 evidence. The existing Graph-owned synthesis lane is
unchanged. NVIDIA accepts the shared strict-boundary policy (no surviving Tile
or Target IR, undefined result, or contract-marker symbol), but enforcement in
the NVVM/PTX/cubin pipeline remains CUDA-owned follow-up.
ROCm has now retired its final generic runtime pass-name helper. **NVIDIA
outcome: follow-up required:** CUDA family plugins must likewise expose a
closed semantic registry rather than an arbitrary pass option. No PTX/cubin,
SM schedule, selector, or device evidence changes here.

## Cross-backend sync `TSOL-PACKED-FUSION-2026-08-08`

The shared `schedule.spectral_program` contract now hashes packed-real fusion
topology and N/2 child identity. **NVIDIA outcome: follow-up required on a
CUDA-enabled host.** No NVVM/PTX/cubin consumer changed. NVIDIA must select its
own real-transform plan and carry the exact v5 artifact through its physical
package; Zen 5 and gfx1151 schedules, workspaces, and evidence do not transfer.

## Cross-backend sync `TILE-SYNC-RECONCILE-2026-08-10`

`tile.async_copy`/`tile.wait_async` now have one declared contract (ODS dual
form, `TileOps.td`): typed `!tile.async_token` SSA is production; legacy
grouping keys are the declared envelope, optional and conservative on absence.
New shared diagnostic `TILE_ASYNC_STAGE_NEGATIVE`. **NVIDIA outcome: parity
validated at the core-IR level.** The typed-token SSA edge is the NV warp-spec
production model (`TileIRLoweringPass::emitAsyncCopy`,
`tessera-warpspec-legality`); the previously contradictory required-stage
verifier was the one rejecting production NV Tile IR, and
`phase2/pm_verify_async_token.mlir` (red at baseline) now passes. No
NV-device-lane (`tessera-nvidia-opt`) fixtures changed.

## TILE-SYNC-TYPED-2026-08-15 — shared Tile sync ABI assessment (PR #566)

**Follow-up required.** The retyped family is NVIDIA's vocabulary:
`tile.mbarrier.wait` gains an optional `!tile.mbarrier_token` segment (the
operand-segment ABI is now barrier/token/dependencies), `tile.tma.copy_async`
and the wait grow fail-closed gates-on-nothing verifiers, the keyless legacy
wait is an explicit `tile.retire_all` marker (stamped by AsyncCopyLowering,
resolved — or preserved when no completion tokens exist — by
NVTMADescriptorPass), and `--tessera-tile-dataflow-legality` runs in every
NVIDIA pipeline after the post-NVTMA legality blocks. Host-free evidence:
full lit 324/0 including the pipeline-alias and NVTMA fixtures. **Open on
this queue:** sm_120-host revalidation of the NVTMA pipelines and any
SM90/Hopper device proof (Phase G/H). The barrier-at-birth compiler restructure
is host-free parity validated by the complete Phase 3 IR lane (24 supported
tests passed, 7 unsupported), including the named NVIDIA pipeline, streaming
FlashAttention, tokenless retire-all compatibility, and distinct barriers with
local slot zero across two schedule regions. Exact-device evidence is still
required before closing the hardware rows.

## REF-TIER-OPS-2026-08-15 — reference-tier op registration assessment (PR #568)

PR #568 registered ten new public operations through the canonical op catalog
and the primitive coverage registry — `tridiagonal_solve` (Thomas recurrence,
PDE plan §III.1 / TSOL-A1) and the nine-op coalition-lattice family
(`game_subset_zeta`, `game_subset_mobius`, `game_superset_zeta`,
`game_superset_mobius`, `game_coalition_marginal`, `game_semivalue`,
`game_boltzmann_value`, `game_coalition_excess`, `game_mex`). Op registration
is a shared contract, so this queue records the outcome per AGENTS.md
"Cross-backend work coordination"; PR #568 itself landed without these records.

**Follow-up required — no NVIDIA lane exists for any of the ten.** No
`tessera-nvidia-pipeline-{sm90,sm100,sm120}` stage, PTX emitter, or backend
manifest row consumes either family; the declared tier is the Python
reference. GAME_THEORY_PLAN.md G5 names the per-target arbiter registration
(SubsetZetaRegion beside SpectralFFTRegion, `boltzmann_value` on the
online-softmax emitter) as the NVIDIA-side entry point for the lattice family;
the solver's entry point is the PDE op-set admission, not G5. No sm_120 host
revalidation was run for this registration and no device evidence is claimed —
nothing here changes generated code.

## APPLE-SCHEDULED-REDUCE-NAN-2026-08-16 — shared reduce NaN semantics (PR #571)

**Shared contract changed; assess before relying on extrema reductions.** The
synthesizer's reduce vocabulary (`compiler/fusion_core.py::_PW_REDUCE_KINDS`)
emitted `max(acc, v)` / `min(acc, v)` for `amax`/`amin`. Metal's `max`/`min` are
IEEE maxNum/minNum-style and **suppress** a NaN operand, so the emitted kernel
disagreed with the table's own numpy reference (`a.max(-1)`, which propagates)
and with the `nan_mode = "propagate"` the reduce Schedule artifact declares.
With the `-INFINITY` seed an all-NaN row reduced to **`-inf`** — missing data
silently becoming a finite extreme. The accumulators now propagate explicitly.

**NVIDIA outcome: not applicable — no consumer.** `_PW_REDUCE_KINDS` supplies MSL
accumulate expressions consumed only by `compiler/emit/apple_msl.py`; SM120
reduction lowering is untouched. No CUDA schedule, PTX path, ABI, or exact-device
claim changes. The NaN-propagation contract is worth noting for any future
`emit/nvidia_cuda.py` reduce emitter: CUDA's `fmaxf` suppresses NaN the same way,
so a naive port would reintroduce the same divergence from the declared
`nan_mode = "propagate"`.

Also recorded for coordination: PR #571 admits Apple GPU into the shared
scheduled reduce contract (`scheduled_kernel.py`, last axis only) and closes
APPLE-DEVICE-EVENT-1 by giving the Apple MPSGraph BMM route an owned command
buffer. Both are Apple-guarded — the shared `scheduled_kernel` gate adds an
`apple_gpu` branch beside the existing x86/ROCm ones and changes neither — and
the runtime edit is in `apple_gpu_runtime.mm`, which no sibling links.

## APPLE-ATTN-BWD-PERF-1-2026-08-16 — backward row-prepass assessment (PR pending)

**Outcome: not applicable to this architecture — no shared contract changed.**
Apple's attention-backward dK/dV split kernel became key-parallel by finally
implementing the `row_prepass` stage that
`attention_contract.plan_attention_backward_workspace` **already declares**
(`row_lse`, `row_delta`, consumed by `dkdv_split` and `dq`). The change is
confined to Apple-private MSL in `apple_gpu_runtime.mm` and its dispatch; no
shared IR, Schedule contract, workspace plan, dtype, ABI, or sibling schedule
was modified. The declared schedule is unchanged — still `split_count = 2` with
ascending `reduction_order = (0, 1)`; only Apple's thread mapping changed, which
is architecture-owned.

**Worth reading anyway, because the shape of the bug is portable.** The declared
workspace stage existed and had no consumer, so every kernel recomputed the row
statistics inline. That is not merely duplicated work: because the statistics
reduce over the whole key axis, an inline-recompute kernel must own an entire
query stream, which capped the dK/dV split at one thread per (partial, KV batch)
— 4 threads at `B1 Hq4/Hkv2 S64`. Implementing the declared prepass raised it to
`2 * kv_outer * Sk` and moved backward from 195 ms to 6.0 ms (~32x), while
keeping single-owner determinism. Any backend whose backward recomputes row
statistics inside its dK/dV kernel should check whether it has the same
structural cap before attributing slowness to memory or instruction mix.

## APPLE-NORM-VJP-1-2026-08-16 — Apple native-VJP registration assessment

**Outcome: not applicable — no shared contract changed.** Apple registered as a
Target consumer in the existing native-VJP normalization plugin and added its own
Metal MSL VJP kernels. The shared registry schema, the `_parse_compiled_norm_backward`
contract, the Graph/Schedule contracts, and every sibling executor are unchanged;
the new code is Apple-private (`apple_gpu_runtime.mm`, one runtime executor, two
`execution_matrix` rows behind one new Apple executor id). This target's own
normalization VJP rows, evidence and selectors are untouched.

Recorded because the *scoping* lesson is portable: the item was scoped as
"registry wiring", but the target had no normalization-backward ABI at all, so
the entry would have been an unconsumed declaration (Decision #29). Any backend
asked to join this boundary should confirm it owns an executable VJP before
declaring a consumer.

## APPLE-NORM-VJP-2-2026-08-16 — reduced-precision normalization VJP assessment

**Outcome: not applicable — no shared contract changed.** Apple's normalization
VJP gained f16/bf16 storage: four new Apple-private C ABI exports
(`tessera_apple_gpu_{rmsnorm,layer_norm}_bwd_{f16,bf16}`), their MSL, the
non-Darwin stub returning 0, and Apple's own ABI registry rows. The shared
native-VJP registry schema, `_parse_compiled_norm_backward`, the Graph/Schedule
contracts, the execution-matrix executor id, and every sibling executor are
unchanged. This architecture's normalization VJP dtype support, evidence and
selectors are untouched.

Two portable notes, recorded because both are cheap to get wrong:

1. **Store rounding is part of the numeric contract.** The kernel accumulates in
   f32 and stores back at the operand's own dtype, and bf16 stores use
   round-to-nearest-even. Truncating instead would bias every gradient in one
   direction — a systematic error, not noise. Any sibling adding reduced-precision
   gradients should match its own runtime's established rounding convention
   rather than defaulting to a shift.
2. **A tolerance derived from storage epsilon is what proves the accumulator.**
   Measured error is ~0.5-0.65 ulp of each format's epsilon, i.e. one rounding on
   store; storage-format accumulation would land near `sqrt(cols)` ulps. A test
   asserting against that second level catches a silently dropped f32
   accumulator, where a hand-picked loose tolerance would absorb it.

A third note is Apple-specific but the shape recurs: adding an export obliges
updating the dylib **freshness gates**, or a runtime built between two landings
exports the older names, passes the staleness check, and then fails at the call
site — which reads as a broken consumer rather than the stale build it is.

---

## Cross-backend record — MC1 matrix-function family (PR #596, 2026-08-20)

**Owning item:** MATRIX-CALCULUS-MC1 · **synchronization key:** `MC1-LINALG-FAMILY`

**Shared contracts changed.** Ten public op registrations (`det`, `logdet`,
`inv`, `solve`, `trace`, `eigh`, `kron`, `vec`, `matrix_power`, `norm`); two
new Graph IR lowering kinds (`linalg_function`, `linalg_multilinear`) with four
new shape rules (`matrix_scalar`, `vec`, `kron`, `eigh`); numeric-policy
entries for both kinds; two diagnostic codes (`E_LINALG_CONTRACT`,
`E_METRIC_CONTRACT`); the `degeneracy_policy` semantic key extended to
eigenvalue gaps and to the nuclear norm's rank condition.

**Outcome for this backend: `not applicable — no lane attempted or implied`.**

The family landed as a *derivative contract* — closed-form VJP+JVP pairs under
the AD law sweep — because that was the gap in the AD stack. No Tile IR
lowering, Target IR op, or kernel was added for any backend, and none is
implied: every one of the ten is registered `backend_kernel: partial` /
reference tier, and each is listed in the Apple no-lane golden and the
single-GPU closeout classifier as an *intentional* reference-only decision with
a stated rationale, so none of them appears on this backend's promote queue as
a phantom blocker.

**What a sibling picking this up would need to decide, per op.** The reference
implementations delegate to LAPACK through numpy, so a native lane is a
vendor-library question rather than a codegen one for `eigh`/`inv`/`solve`
(rocSOLVER / cuSOLVER / Accelerate / MKL), and a "probably never worth it"
question for `det`/`logdet`/`trace`/`matrix_power`, which reduce a small matrix
to one number. `kron` and `vec` are layout/contraction shaped and would ride
existing lanes if anything.

**Numeric policy this backend must match if it does build a lane.**
`linalg_function` and `linalg_solver` declare **f64 accumulate with f32 storage
admitted** — the same conditioning-sensitive policy `linalg_decomposition`
already carries, because `det`/`logdet`/`inv`/`matrix_power` carry a factor of
the condition number. `linalg_multilinear` (`trace`, `kron`) declares the
ordinary f32 accumulator, since neither carries one. A lane that accumulates in
storage precision would silently lose the digits these rules exist to keep.

**Degeneracy contract that travels with the rules.** `eigh`'s eigenvector
coupling `1/(w_j - w_i)` and `svd`'s `1/(s_j^2 - s_i^2)` have no limit at a
crossing; both fail closed under the declared `degeneracy_policy` rather than
emitting `inf`/`NaN`. Any native lane must reproduce that refusal, not paper
over it with an epsilon — a damped coupling returns a finite, plausible, wrong
gradient, which is the failure mode the key exists to prevent.

**Exact-device evidence: none, and none claimed.** Everything in PR #596 was
validated on the host-independent Python reference lane on an M1 Max, where no
AMD GPU, no CUDA device and no AVX-512 exist. No parity claim is made for any
backend.

**NVIDIA-specific note.** cuSOLVER covers `eigh`/`inv`/`solve` directly; the
open design question is whether they enter through Target IR as `abi_call`-style
delegated work (Decision #19's x86 precedent) so the arbiter can still tell
compiler-generated work from delegated work.


## Cross-backend sync `NVIDIA-SPECTRAL-PHILOX-JVP-2026-08-22`

**Owning work:** native spectral sequencing and Philox compiler-JVP integration.
**Outcome: landed and exact-device validated on SuperBear RTX 5070 (sm120).**

The canonical CUDA FFT/workspace ABI is now v2 and owns reusable typed C2C,
R2C, and C2R cuFFT plans with automatic allocation disabled, exact caller-owned
workspace bounds, and on-device inverse normalization. Native `rfft`/`irfft`
cover odd, even, mixed, and prime lengths. DCT-II, STFT, ISTFT, spectral
convolution, and spectral filtering consume this package; framing, windowing,
pointwise work, and overlap-add remain explicit host orchestration and are not
reported as fused CUDA kernels.

The compound spectral reverse package now has an SM120 consumer. Real spectral
convolution VJPs use CUDA R2C/C2R and match direct correlation; the complex
spectral-filter adjoint is exact-package validated. Public NVIDIA Graph IR
`complex64` storage remains planned-gated, so filter VJP is artifact-level
physical evidence rather than a public Graph execution claim.

Compiler JVP packaging now admits exact `nvidia_sm120`. Seeded dropout binds
both primal and tangent children to the identical Philox key/counter attributes,
proving mask replay rather than resampling; unseeded training JVP fails closed.
Exact-device evidence: 22 tests in the three NVIDIA fixtures.


Cross-backend sync `NUMPOL-CARRIER-1-SCHEMA-AND-REDUCTION-2026-08-25` — **the
policy gets a schema and the reduction family carries its accumulator;
NVIDIA outcome: shared contract only; no CUDA change.** Two measured defects closed, both shared,
neither architecture-specific.

*Schema.* `numeric_policy` was a bare `DictionaryAttrBase` whose ODS predicate
checked only "is a dictionary". Measured before the change: five malformed
policies were all ACCEPTED while the documented TF32-as-storage violation
correctly failed, so the pass was running and simply had nothing to say. The
sharpest was a typo — `getAs<StringAttr>("accum")` returns null for a
misspelled key exactly as for an absent one, so an op carried a policy that
looked like it stated an accumulator contract and stated none. Seven new
diagnostics now refuse unknown keys, non-string values, unknown dtypes/modes, a
math_mode that does not reduce its storage, and an accumulator NARROWER in
significand bits than its storage.

*Reduction carrier.* `{storage="bf16", accum="fp32"}` on rmsnorm / softmax /
layer_norm lowered to `arith.addf … : bf16` — the emitted code contradicted the
declared contract on the very op that performs the accumulation. Executed on
Zen 5 through `--tessera-to-linalg` → LLVM → native object: a 4096-wide softmax
row summed to **1.466** (a 47% violation of the function's defining property)
versus **1.000169** once the declared accumulator is honoured. The whole
derived chain now runs in the accumulator with a single truncation at the
result — chosen by measurement over truncating the reduced value, which is 326x
worse. With no policy the emitted IR is byte-identical, so nothing widens
without being asked. The Graph→Linalg boundary is now bracketed by the
Decision #32 record/verify pair and declares `represented_in_type` /
`re_expressed`.

The schema and the reduction carrier are target-independent Graph-level contracts. Zen 5 execution transfers no sm_120 claim.

**`math_mode` now has a consumer on the runtime dispatch, and closing it exposed an INTERNAL CONTRADICTION on this backend — NR2 Pro proof owed.** This queue already records (above) that "the shared MMA selector now requires explicit `math_mode=\"tf32\"` for fp32". `runtime.py` did not: `_NVIDIA_GEMM_SYMBOLS` mapped `"float32"` to `tessera_nvidia_mma_gemm_tf32` and the dispatch took it **unconditionally**, with a comment citing Decision #15a as the reason. So two components held opposite policies for the same fact, and the permissive one is the one that executes. #15a says "TF32 is not a storage dtype. Model as `math_mode='tf32'` on fp32 via numeric_policy" — precisely so the reduced arithmetic is a choice the program makes; the storage dtype was making it instead, and the comment made the violation read as compliance.

Measured against an fp64 reference on 64xKx64 GEMMs (median relative error): fp32 1.64e-07 → tf32 2.93e-04 at K=128 (**1783x**); 3.02e-07 → 3.01e-04 at K=1024 (998x); 3.63e-07 → 2.91e-04 at K=4096 (800x). TF32 keeps 11 significand bits against fp32's 24 and rounds the OPERANDS, so no accumulator width recovers it. A program that asked for fp32 got tf32 numbers and no diagnostic.

Selection is now the pure function `runtime._nvidia_gemm_selection`, so its contract is host-testable on a box with no CUDA: explicit `math_mode="tf32"` selects the tf32 kernel, `"ieee"` (or any mode the lane lacks) is refused with `NVIDIA_MATH_MODE_UNAVAILABLE` rather than handed tf32 numbers under an fp32 label, and narrow-storage paths are untouched. This makes the runtime agree with the selector contract this backend already adopted — it is not a new policy.

**Absent `math_mode` deliberately keeps today's TF32 behaviour**, and that is the owed decision rather than an oversight. By #21a a semantic key should fail closed on absence; changing the default would alter every existing fp32 NVIDIA program from a host that cannot execute one, which is exactly the claim the fleet rule forbids. A test pins the current behaviour so the follow-up has a fixed baseline. **Owed on NR2 Pro (RTX 5070 Ti):** execute the tf32 and f16 lanes, confirm the selection reaches the intended kernel, and decide whether absent-`math_mode` should fail closed.

Also on this backend, not device-proven here: the sm_120 typed route's `!tile.fragment` accumulator now follows `selected->accum` instead of a hardcoded `"f32"` written three times beside that unused field, and a declared accum the schedule cannot provide fails closed with `MATMUL_SCHEDULE_ACCUM_UNSUPPORTED`. Byte-identical across all 169 Graph→Schedule→Tile fixtures under a before/after control.

---

## Cross-backend sync `DELTANET-BOUNDED-VJP-2026-08-31`

**Owning item:** the bounded ("modified"/Kimi) DeltaNet reverse rule ·
**PR:** #660 · **synchronization key:** `DELTANET-BOUNDED-VJP-2026-08-31`

**Shared contract changed — the bounded VJP's correction term.** The bounded
variant scales the rank-1 update by `f = 1/(1 + n)` with `A_de = k_d·target_e`
and `n = ‖A‖_F`, so the reverse rule for `U = b·A·f` is

```
∂L/∂A_ij = b·dU_ij·f  −  b·(Σ_de dU_de·A_de)·f²·A_ij / n
```

This is a *shared numerical contract*, not a per-backend schedule: three
backends implement the same closed form independently against one reference
(`get_vjp("modified_delta_attention")`). Two of them divided that correction by
**`max(n, 1)`** instead of `n`, understating it by a factor of `n` whenever
`n < 1` — which, with L2-normalised keys, is the ordinary case rather than an
edge case. No clamp is needed or wanted: the numerator is `O(n²)` (both
`update` and `projection` scale with `A`), so the quotient vanishes as `n → 0`,
and the existing `norm > 0` select already covers `n == 0` exactly.

**Why the failure was silent in four of six gradients.** At `erase=False` the
bounded update reaches only `dk` and `dv`; `dq`, `dgate`, `dbeta`, `ddecay` do
not consume `du` and stayed exact to ~1e-9. A wrong *bound* derivative
therefore presents as two gradients off by 19–33% while the other four are
perfect — which is what ruled out precision loss and pointed at a missing term.

**Outcome for this backend: `parity validated` — the formula was already
correct, but the exact-device evidence for it was not, and that was fixed
here.** `_deltanet_backward_source` in `python/tessera/compiler/
nvidia_training.py:1115` already divided by `(norm * denom * denom)`, so CUDA
never carried the defect. Confirming that by inspection is not evidence, so it
was measured on **Super-Bear / RTX 5070 (sm_120)**, and the measurement found a
second, separate problem.

**The NVIDIA device test could not have failed.**
`test_erase_modified_deltanet_serial_fill_backward_executes_on_sm120` asserted
at `rtol=5e-3, atol=5e-3` over gradients whose full scale is 1e-3..3e-2. Two of
the six — `dgate` (max 1.07e-03) and `ddecay` (max 2.20e-03) — are *entirely
below* that atol, so those assertions would pass for any kernel output
including all zeros.

Proven by mutation rather than argued: injecting this PR's exact defect
(`fmaxf(norm, 1.0f)`) into the CUDA source left the suite **green**. A control
mutation (a gross 2x on `dupdate`) failed immediately, which rules out a stale
cubin and locates the fault in the assertion rather than the toolchain. Both
tolerances are now `rtol=1e-5, atol=1e-8`, chosen from the measured
device-vs-reference deviation of **4.7e-10 abs / 2.0e-07 rel** (f32 round-off,
~50x margin) rather than picked by feel. Re-verified on sm_120: correct kernel
**2 passed**; same injected defect now **fails on `dk` at 13.6% relative**, the
same signature ROCm and x86 showed at 19–33%.

**Standing lesson.** A sibling backend that is green on a shared numerical
contract has not been assessed until its test is shown to be capable of going
red. This is another instance of the hollow-green pattern the fleet ledger
tracks, and the first found by mutating a device kernel. The cheapest general
form of the check needs no mutation at all: compare a test's tolerance against
the *magnitude of the quantity it asserts on*: an atol above full scale is a
vacuous assertion on its face.

---

## Cross-backend sync `NVIDIA-TIMER-DRAIN-2026-08-31`

**Owning item:** the NVIDIA half of the three-clock timing discipline
(`AUTOTUNE-RACED-FIELD-SYNC-2026-08-30`'s recorded follow-up) ·
**synchronization key:** `NVIDIA-TIMER-DRAIN-2026-08-31`

**Shared contract changed — what makes a device latency believable.** Every
backend's autotune verdict rests on one, and the corpus compares them across
architectures, so the rule for accepting a device clock is shared even though
each host's clocks are not.

**The finding: the drain and the cross-check are *ordered*, and adding the
second without the first is worse than adding neither.** Measured on sm_120
(RTX 5070) with 2500 2048³ GEMMs resident on a blocking stream, timing 40
launches of a 1024³ GEMM:

| start event recorded | wall ms/rep | event ms/rep | event/wall |
|---|---|---|---|
| without a preceding drain | 63.3338 | 0.3227 | **0.005** |
| after a drain | 0.3263 | 0.3255 | **0.998** |

The event is correct in both rows. Undrained, the start event is queued behind
the contending work, so the *wall* spans that drain and the event does not —
the two clocks bracket different regions and comparing them is meaningless.
Apply the two-sided band to the undrained row and it rejects the correct
0.3227 ms event (its lower bound is 31.7 ms) and falls back to the 63.33 ms
wall: a **196× overstatement**. In review, "adds a wall cross-check" reads as
strictly safer; here it would have been strictly worse.

**Outcome for this backend: `parity validated` — defect owned and fixed here,
on device.** `_nvidia_mma_gemm_device_latency` recorded two events around its
launches and returned the elapsed value with **no validation of any kind**: no
wall clock to compare against, and no drain that would have made a comparison
mean anything. It now goes through `_nvidia_timed_launch_ms`, which drains,
takes a wall witness, and band-checks. Every sm_120 shape reports
`device_event`, as expected.

**Two assumptions this plan carried by analogy did not survive measurement.**

* *"Whether sm_120's event clock is trustworthy is an open question."* It is
  trustworthy: `event/wall` measured **0.996–0.998** idle and contended. HIP's
  lying-clock rationale does not apply, so `_accept_nvidia_event_ms` is a
  separate function from ROCm's `_accept_device_event_ms` rather than a shared
  one. The band is identical; merging them would force one host's rationale
  onto the other, and the next person to widen one would silently widen both.
* *"Never time on the default stream"* does not transfer as stated. Legacy
  stream 0 **does** serialise against a blocking stream here — the wall
  inflated **161×** and the contending stream had drained by the end of the
  region, both signatures of it — but the CUDA **event was unmoved (1.01×)**,
  because the event pair brackets only its own stream's span. Moving the timed
  launches to a dedicated `CU_STREAM_NON_BLOCKING` stream measured **8.65×
  slower** (2.7667 vs 0.3199 ms): a truthful measurement of a kernel sharing
  the GPU, and the wrong number for an autotune verdict. For isolated latency
  the serialisation is a *feature*. A dedicated stream is required for
  concurrency benchmarks — which these timers are not.

**Correction to `NVIDIA-TIER-PRIORITY-IS-WRONG-AT-SCALE-2026-08-30`: "the
compiled kernel wins at every shape" is wrong at the smallest shape.** The
delegate half of that comparison came from the undrained timer, which
over-reads most where the kernel is shortest — measured **−25.4%** at 256³
against **+0.5…+2.6%** at 512³/1024³/2048³. Re-raced with the corrected timer,
12 interleaved samples per lane at 200 reps:

| shape | delegate | emitted PTX | verdict |
|---|---|---|---|
| 256³ | 0.01300 median (sd 14.5%) | 0.01057 median (sd 39.1%) | **not separated** — 18.7% apart against a 39.1% spread |
| 384³ | 0.02078 (sd 7.8%) | 0.01365 (sd 5.5%) | emitted, 1.52× |
| 512³ | 0.04402 (sd 2.2%) | 0.02695 (sd 0.6%) | emitted, 1.63× |
| 1024³ | 0.32102 | 0.19366 | emitted, 1.66× |
| 2048³ | 2.45369 | 1.47545 | emitted, 1.66× |

So the standing claim holds from 384³ up and **256³ is a tie**: that shape sits
at the launch-overhead floor, where a lane's own run-to-run spread exceeds the
gap between lanes. The earlier 1.63× there was an undrained delegate number
plus a three-sample read. The arbiter should not record a matmul winner at 256³
without a separation check — a median difference smaller than the spread is not
a verdict, and this is the second time a matmul ranking has been wrong at a
shape nobody re-measured.

---

## Cross-backend sync `AUTOTUNE-SEPARATION-2026-08-31`

**Owning item:** the arbiter's measured-verdict contract ·
**synchronization key:** `AUTOTUNE-SEPARATION-2026-08-31`

**Shared contract changed — a `MeasureRecord` must now say whether its ranking
survives its own noise.** All four backends write into and read from this
corpus, so the rule for what counts as a verdict is shared even though the
hardware producing the numbers is not.

**The defect.** `measured_arbitrate` took one median per candidate and picked
`min`. A margin was a margin; nothing recorded how noisy the lanes were, so a
19% gap between two lanes whose own spreads were 14.5% and 39.1% was stored
exactly like a 40% gap between lanes spread 2.2% and 0.6%. Measured on sm_120
at 256³:

| lane | median | spread |
|---|---|---|
| delegate (`nvidia_mma_gemm_shipped`) | 0.01300 ms | 14.5% |
| emitted PTX | 0.01057 ms | 39.1% |

That was recorded as a clean **1.63× win**. The same *ratio* at 2048³ is real.
The ratio is not what distinguishes them — the spread is, and the record did
not keep it.

**Measured over the committed corpus, so this is not a hypothetical:** 87 rows,
of which **75 assert a ranking** (two or more candidates timed), **none**
declares a separation, and **11 of the 75 — 15% — picked a winner that beat the
runner-up by under 2%**, the tightest at **0.07%**. That is inside ordinary
end-to-end wall jitter; those eleven verdicts record which lane was luckier.

**The rule.** The margin must exceed `SEPARATION_FACTOR` (2×) the *noisier* of
the two fastest lanes. `None` — not `True` — when fewer than two candidates were
timed: a sole candidate is chosen by applicability, not by a race, so there is
no margin to defend. As with `unmeasured`, absence is not the favourable
answer, and a publisher must treat `None` and `False` alike.

**A tie never blocks dispatch** — something has to run. It blocks two things:
*claiming* one candidate is faster, and re-picking by noise on every run. An
unseparated re-race now keeps the incumbent; a separated one still displaces it.

**Outcome for this backend: `parity validated` — the defect was found here and
the sm_120 evidence for it is this backend's.** All eleven under-2% rows in the
committed corpus are NVIDIA rows, which is an artifact of NVIDIA being the only
backend with four matmul candidates racing at every bucket, not of anything
CUDA-specific.

**Consequence for the corpus: it is now under-declared, not wrong.** Every
committed row reads `separation: None` — never asked — so no verdict has to be
retracted, but the eleven tight rows should not be cited as rankings until
re-raced. Re-generating them needs sm_120, and is the follow-up this key owns.

**Two NVIDIA-specific notes for whoever re-races.**

* The 256³ matmul bucket is at the launch-overhead floor and may simply never
  separate. That is a legitimate outcome to record, not a measurement to keep
  retrying at higher `reps` until one lane wins.
* `device_repeats` (default 3) triples device-timing cost, which matters most
  here because NVIDIA races the widest field. Three samples give a usable noise
  floor but a poor spread *estimate*; where a bucket lands near the bar, raise
  it for that run rather than trusting the boundary.

---

## Cross-backend sync `APPLE-TIMER-WITNESS-2026-08-31`

**Owning item:** the Apple half of `NVIDIA-TIMER-DRAIN-2026-08-31`, recorded
there as follow-up required · **synchronization key:**
`APPLE-TIMER-WITNESS-2026-08-31`

**Shared contract changed — a device latency must be checked against a clock
that did not produce it.** ROCm and NVIDIA already did; Apple did not, and had
no host clock at all to check against.

**Two findings from the measurement, both of which changed the design.**

*The witness must bracket the same region the device clock does.* The first
version instrumented `commit_and_wait_with_timeout` only — and the lane this
workload actually takes (`metal4_mpsgraph_envelope`) does not go through it.
It reported a null wall, which the acceptance rule reads as "no witness
available" and passes the device number through unchecked. The failure was
silent: telemetry looked healthy, the band existed, and it was checking nothing.

*A witness scoped to the wrong region produces a wrong rule, not an obvious
failure.* Measured device/wall against a **Python-level** wall — which carries
numpy marshalling and array conversion no GPU interval could contain — was
**0.35–0.60**, and that argued for a one-sided band, since 0.35 fails a 0.5×
floor. Against the **runtime** witness the same dispatches run **0.568–0.937**
over 100 samples from 8² to 2048². The symmetric band was fine all along; the
one-sided version was defending against an artifact of its own denominator.

**Corrected after review (2026-08-31): the band is ONE-SIDED, and the two-sided
version was wrong twice for the same reason — generalising from one route.** A
second route family, resident batched sessions on `metal_kernel_interval`,
measures **0.037–0.101** once warm: a 25 µs kernel inside a 265 µs
submit-to-signal window. Nothing is wrong there — `kernelStartTime`/
`kernelEndTime` is kernel execution only and legitimately excludes queueing.
Across routes the honest range is **0.037–0.937**, so no wall-derived floor can
separate a small kernel from an under-reading clock. That is exactly what ROCm
already states in `_select_rocm_latency_ms`.

What survives is containment, which is exact: GPU execution is a strict subset
of commit-and-wait, so **`device <= 1.25 × wall`** (1.25 rather than 1.0
because the two are independent clocks over nested regions and the measured max
of 0.937 leaves a strict bound only 6.3%). **The under-reading direction — the
dangerous one, since an under-estimate inflates throughput and gets published —
is now explicitly unguarded**, and asserted as such in the tests so the gap is
visible rather than assumed covered. It closes against Apple's *second* device
clock, not the wall; see the follow-up below.

**Outcome for this backend: `not applicable` — NVIDIA closed its half under
`NVIDIA-TIMER-DRAIN-2026-08-31` and nothing here changes it.** CUDA already
takes a wall witness across a drained region and band-checks the event against
it; the Apple work is the sibling item that key recorded as owed.

**The one transferable lesson, and it is a scoping one.** NVIDIA's finding was
that a band without a *drain* is worse than no band. Apple's is that a band
without a **correctly scoped witness** is worse than no band — the first
version measured against a Python-level wall carrying numpy marshalling, and
concluded a one-sided band was needed when it was not. Same failure at
different ends of the same sentence: the band is only as good as the region its
witness covers, and both ways of getting that wrong produce a plausible rule
rather than a visible error. Worth checking against the remaining twelve
duplicated timing loops in `tessera_nvidia_ptx_launch.cpp` when those are
de-duplicated — each one defines its own region.

**No sm_120 evidence is claimed** — no CUDA code changed.

---

## Cross-backend sync `APPLE-DEVICE-CLOCK-2026-08-31`

**Owning item:** Apple's device clock · **synchronization key:**
`APPLE-DEVICE-CLOCK-2026-08-31`

**Shared contract changed — what makes a device clock a *measurement*.**
`APPLE-TIMER-WITNESS` added a host witness and a containment bound. This closes
the direction that bound provably cannot reach: a clock that under-reads looks
exactly like a small kernel, since both sit far below the host wall.

**The defect.** `ts_record_dispatch_gpu_elapsed` preferred
`cb.kernelStartTime`/`kernelEndTime` and treated `GPUStartTime`/`GPUEndTime` as
a fallback, on a comment asserting the first pair was "the completed
compute-kernel interval". **The SDK says the opposite by omission**:
`GPUStartTime` carries an `@abstract` — *"the host time in seconds that GPU
starts executing this command buffer"* — and `kernelStartTime` is a bare,
undocumented declaration. Measured on an M1 Max with only a kernel's loop count
varying:

| iters | `kernelS/E` | `GPUS/E` | encoder stage | host wall | kern/wall |
|---|---|---|---|---|---|
| 5,000 | 54,583 | 498,375 | 498,375 | 764,417 | 0.071 |
| 320,000 | 65,833 | 9,390,833 | 9,390,792 | 9,833,750 | **0.007** |

`kernelStartTime` is **flat across a 64× workload**. `GPUStartTime` tracks the
wall *and* agrees with an independent stage-boundary counter-sample clock **to
the nanosecond** — two mechanisms agreeing that closely is what distinguishes a
measurement from a plausible number.

**A second bug hid behind the first.** `GPUStartTime` is documented to read zero
until the GPU starts and to be readable "in command buffer completion handler".
Every dispatch path here waits on a *shared event*, which proves the GPU
finished but does not publish those properties. Simply preferring the documented
pair therefore changed nothing — it read zero and fell straight back. The
recorder now forces publication itself (`ts_gpu_interval`), so no caller can
forget.

**The generalisable finding is the check, not the property.** No bound catches
an under-reading clock. What caught this is **metamorphic**: vary the workload
and require the device clock and the host wall to move *together*. They may
diverge in magnitude — the wall carries submission overhead — but not in
direction. Under the defect that ratio was 0.32–0.40; healthy it is 0.86–1.14.

**Outcome for this backend: `parity validated` on existing sm_120 evidence — no
CUDA change, and the check this key generalises was already satisfied here.**
`NVIDIA-TIMER-DRAIN-2026-08-31` measured `event/wall` at **0.996–0.998 across
shapes**, which is a tracking result and a stronger one than Apple had: two
clocks that agree to 0.2–0.4% at every size cannot be flat while the other
moves. The re-raced matmul corpus is the same story at the workload level —
0.01076 / 0.04434 / 0.32102 / 2.45369 ms across 256³–2048³, monotonic over a
512× work range.

**Why this is worth a row anyway.** NVIDIA has exactly one device clock and no
second one to cross-check, so its whole defence is the wall witness plus that
agreement. Apple's defect is the case where a device clock is *internally*
plausible and simply not measuring — and no bound detects that. If a CUDA lane
ever gains an in-kernel `%globaltimer` stamp (the `wall_clock64` analogue ROCm
uses), the metamorphic tracking check is what should gate it, not agreement at
a single shape.

**No new sm_120 evidence is claimed** — nothing CUDA-side changed under this key.

---

## Cross-backend sync `PACKET-PROVENANCE-2026-08-31`

**Owning item:** exact-device evidence provenance ·
**synchronization key:** `PACKET-PROVENANCE-2026-08-31`

**Shared contract: a packet may not claim a commit it was not generated from.**
Every lane's recorder stamps `tested_commit` from `git rev-parse HEAD` and
**none of the four checked that HEAD is what was actually measured.** Recording
from a modified working tree therefore produces a packet whose measurements
came from edited sources while its `tested_commit` names the parent — false
provenance that then propagates into `docs/audit/generated/e2e_fleet.*` as
though it were a device result for that commit (AGENTS.md:87-90).

**Found by doing it.** The Apple packet on PR #665 was sealed from a dirty tree:
its `source_fingerprint` hashed the *edited* `apple_gpu_runtime.mm` while
`tested_commit` named the parent, whose runtime hashes to something else. It was
review that caught it, not any gate.

**Apple is the only lane where the contradiction is visible at all**, because
only its packet carries a `source_fingerprint` of a runtime source file. The
other three fingerprint measured *resources*, not sources — so a packet built
from modified kernels is internally consistent and silently wrong. **The lane
with the strongest self-check is the one that got caught; the weaker three
would not have surfaced it.**

**Outcome for this backend: `follow-up required` — same defect, unfixed.**
`record_sm120_packet.py:426` stamps `tested_commit` from `git rev-parse HEAD` with no dirtiness
check, exactly as Apple's did.

**Why it is not fixed in this PR.** The Apple guard works because that packet
declares which file it fingerprints, so the set to check is unambiguous. This
lane fingerprints measured resources rather than sources, so choosing the right
file set — plausibly the CUDA runtime sources and `ptx_emit` (which generates the kernel text at record time) — is a judgement about what this backend's
measurement actually depends on, and getting it wrong fails in the worse
direction: a too-narrow set is a guard that passes while the provenance is
false, which reads as protection and is not. That call belongs with someone
looking at this backend's build, on **Super-Bear**.

**The cheap interim** is the whole-tree form: refuse when `git status
--porcelain` is non-empty for this backend's source directory. Cruder than
Apple's and more likely to be bypassed, but it cannot be wrong in the
dangerous direction.

---

## Cross-backend sync `SPECTRAL-CONV-RANK-2026-08-31`

**Owning item:** `tessera.spectral_conv` operand contract ·
**synchronization key:** `SPECTRAL-CONV-RANK-2026-08-31`

**Shared contract enforced (not changed): `spectral_conv` takes equal ranks.**
It was already stated in two places — the host reference in `__init__.py`, which
raises on a mismatch, and the `conv_full` shape rule, which returns *unknown*
rather than deriving `n + m - 1`. **No dispatch lane stated it.** Each computes
`rfft(x) * rfft(w)` and so inherited numpy broadcasting for free, silently
admitting a rank-1 kernel against a rank-2 signal.

**The divergence was in what the lanes ADMIT, not in what they compute.** The
broadcast and rank-matched forms are **bit-identical** (max |Δ| exactly 0.0),
and both match `np.convolve(..., mode="full")` to 2.4e-07. That is precisely why
it survived: nothing was ever numerically wrong, so no accuracy check could see
it. What it produced instead was a GPU lane that accepted input the CPU
reference rejects.

**And it had already caused a hollow test.**
`test_apple_gpu_spectral::test_composites_match_host_reference` was written
against the permissive lane, passing a rank-1 kernel — so its
`assert_allclose` **never executed**: the reference raised while building the
expected value. The test had been red long enough to be repeatedly triaged as
"pre-existing".

**Fixed by stating the rule once** — `_check_spectral_conv_ranks` in
`runtime.py`, called by every dispatch lane — rather than adding a check per
lane. There are **three** dispatch implementations of this op
(`_spectral_composite`, shared by NVIDIA and ROCm; `_apple_gpu_dispatch_spectral`;
and the host reference), and a rule copied three times is a rule that will hold
in two of them.

**Outcome for this backend: `follow-up required` — the fix reaches NVIDIA's
lane but no sm_120 evidence exists for it.** `_spectral_composite` is shared by
NVIDIA and ROCm, so `_nvidia_fftexec` now refuses a rank mismatch too. That is
a **behaviour change on this backend made from a Mac**, and it is host-free
only in the sense that the check itself is pure Python — whether NVIDIA's
spectral lane has any caller passing a rank-1 kernel is not something this host
can answer.

**What is owed on Super-Bear:** run the spectral suite against the CUDA lane and
confirm nothing was relying on the broadcast form. The risk is low (every
in-repo call site outside the corrected test uses equal ranks) but it is not
zero, and a refusal that fires in production is a hard error rather than a
degraded number.

---
## Cross-backend sync `HSACO-NEGATIVE-CACHE-2026-08-31`

**Owning item:** compiled-family build caching ·
**synchronization key:** `HSACO-NEGATIVE-CACHE-2026-08-31`

**`_build_rocm_family_hsaco` cached successes and not failures.** It invokes
`tessera-opt` through `subprocess.run`, so a host that HAS the binary but
cannot serialize ROCm — any dev box without the toolkit, and this Mac — forked
a process, waited for it to fail, and discarded the answer **on every single
launch**. **74 call sites** funnel through it.

**Measured on an M1 Max**, `rt.launch` of a draft-block workload:

| | direct | launch | ratio |
|---|---|---|---|
| before | 0.1042 ms | **70.5963 ms** | **677×** |
| after | 0.1141 ms | 0.1481 ms | 1.3× |

**477× faster**, and 0.347 s of a 0.354 s profile was the subprocess.

**It was hidden by a test that ratified it.** Five perf baselines assert
`launch_ms < max(75.0, direct_ms * 4.0)`. The oracle arm is ~0.4 ms, so `max()`
always selected **75.0** and the self-calibrating comparison was dead code —
while launch sat at **94% of that limit on an idle machine**. The constant had
been sized to accept exactly the overhead it claimed to bound, and tipped over
under any load, which is how it read as five flaky tests rather than one defect.

**The floor is now 2.0 ms, chosen by the only criterion that makes a constant
worth having: it FAILS on the old code.** Verified by mutation — with the
negative cache removed, all five report `71.3 < 2.0`. The old 75.0 accepted
70.6 ms in silence.

Caching the failure is sound because the causes are host properties, fixed for
the process. A genuinely transient failure would be remembered until exit — the
right trade against re-forking per call, and why the stored value keeps the
original message. Every attempt still records a dispatch fallback:
`_rocm_compiled_failed` is re-raised through its own funnel on a cache hit, so
`TESSERA_STRICT_DISPATCH=1` still raises. Only the subprocess is skipped.

**Outcome for this backend: `follow-up required` — the same asymmetry is likely
present and unmeasured.** Nothing NVIDIA-side changed. The question this key
raises for CUDA is whether its own compile paths cache failure: NVRTC compiles
at load (`nvidia_training.py`, `ptx_emit`), and a host without a usable CUDA
toolchain would retry per call in the same shape if the negative result is
discarded.

**Worth checking on Super-Bear**, and cheap: profile one `rt.launch` of a CUDA
family on a host where the build fails, and look for `subprocess`/NVRTC in the
cumulative time. The tell is the one this key was found by — a launch whose cost
is orders of magnitude above the work it performs.

---

## Cross-backend sync `APPLE-COMPLETION-HANDLER-2026-09-01`

**Owning item:** how a device timestamp is *obtained* ·
**synchronization key:** `APPLE-COMPLETION-HANDLER-2026-09-01`

**Shared lesson: when a value is only valid after an event, take it from the
event — do not block waiting for the value.** `APPLE-DEVICE-CLOCK` established
that `GPUStartTime`/`GPUEndTime` is the clock that measures work; it published
those values by forcing them with `[cb waitUntilCompleted]`, which has **no
timeout**. Both resident-session paths invoke the recorder *after* their 30 s
shared-event wait expires, so on a hung GPU that turned a bounded failure into
a permanent hang — in code whose only job is telemetry.

The first repair gated the wait on a `completed` flag threaded through six call
sites. That contained the hang but surrendered the clock on exactly the paths
that had timed out, and put a correctness-critical boolean into six places
where it could be passed wrongly (it was: one site passed a literal `true`).

**`addCompletedHandler` removes the dilemma instead of containing it**, and is
what the SDK prescribes — `GPUStartTime` is documented as readable *"in command
buffer completion handler"*. Metal fills a slot on completion; the reader waits
on a semaphore with a **bounded** timeout. A hung buffer never fires its
handler, the wait expires, telemetry reports no device number. No hang, and the
flag is gone.

**Two traps worth carrying to any backend doing the same.**

*The callback runs on another thread.* The slot is heap-shared and deliberately
**not** `thread_local` like the other 15 telemetry globals — a `thread_local`
write from Metal's callback thread lands in storage the reader never sees.

*Removing a wait can silently re-route the fallback.* Three tile recorders have
no slot of their own (they run after `commit_and_wait_with_timeout` already
waited) and would have fallen straight through to `cb.kernelStartTime` — the
clock measured **flat across a 64× workload**. The change would have undone
`APPLE-DEVICE-CLOCK` on three paths while every test stayed green. A direct,
non-blocking property read now precedes that fallback.

**Outcome for this backend: `not applicable` — nothing changed here, and the
specific API is Apple's.** Recorded because the *reasoning* transfers and the
question is cheap to ask of CUDA: `cuLaunchHostFunc` / a stream callback, against the event-elapsed pair it uses today.

**The test worth applying, whatever the API.** Does the timestamp read path
contain a wait that has no timeout? On Apple the answer was yes, twice — once
outright, once behind a flag that six call sites had to pass correctly. Neither
was visible as a failure until a reviewer traced what happens when the shared
event times out, because the hang only occurs on a GPU that is already broken.

**If this is ever checked on Super-Bear**, the control that made the Apple answer
trustworthy is the one to reuse: measure dispatch variance with and without the
change on the same box, before attributing instability to either.

---

## Cross-backend sync `CAPABILITY-GUARDS-2026-09-01`

**Owning item:** host-capability gating in the unit suite ·
**synchronization key:** `CAPABILITY-GUARDS-2026-09-01`

**Three tests failed on a host that cannot evaluate them, instead of skipping.**
CLAUDE.md's claim-integrity rule is explicit that a host lacking a device or
toolchain must say so; these did the inverse, and one of them (`gfx1151`) had
been carried as "pre-existing" for the whole session.

| test | what it needed | what happened |
|---|---|---|
| `..._sm120_tile_fragment_lowers_to_real_nvvm_mma` | a runnable `tessera-nvidia-opt` | `dyld: Symbol not found` from a binary linked against a prior LLVM keg |
| `..._attention_package_rejects_stale_parent_and_tile_lineage` | the AVX-512 shared image | `RuntimeError: X86 native packaging requires ...` |
| `..._gfx1151_scheduled_attention_backward_packages_exact_tile_program` | AMD clang | `RuntimeError: ROCm native packaging requires AMD clang ...` |

**One family: a guard that checks PRESENCE rather than USABILITY.** And in
every case **the helper that prevents it already existed, and the caller did
not use it** — which is the part worth carrying, because writing the helper
felt like fixing the problem:

* `compiler_tool.is_runnable`, whose docstring describes this exact dyld
  failure, gated only `tessera_opt` discovery. `_tool_path` — which resolves
  `mlir-opt` *and* `tessera-nvidia-opt` — checked `is_file()` alone.
* `rocm_native.native_packaging_available`, whose docstring says checking tools
  without AMD clang "reads as a broken test on any host without ROCm rather
  than an absent toolchain", was not called by the gfx1151 test.
* `rt._x86_elementwise_available` is used by a sibling test one function away.

**The skip has to be proven narrow, not just present.** A guard that skips too
much is the hollow-green pattern wearing a fix's clothes, so each was verified
on a host that HAS the capability:

| host | result |
|---|---|
| Princess-Luna (AVX-512 + ROCm) | both tests **PASSED**, not skipped |
| Super-Bear (CUDA) | nvidia-opt test **PASSED**, not skipped |
| Princess-Luna, resolver control | **771 passed / 85 skipped before and after** — the `_tool_path` change adds zero skips where tools work |

**Outcome for this backend: `parity validated` — fixed and verified on
Super-Bear.** `_tool_path` now requires `tessera-nvidia-opt` to start, not
merely to exist. An exported selector stays final: an unrunnable
`$TESSERA_NVIDIA_OPT` returns `None` (a clean skip) rather than silently
falling back to a different binary, because running one the developer did not
ask for is how a passing run stops meaning anything.

**The failure mode is specific to a machine that has built more than one
toolchain**, which is every developer box and no CI runner — so CI would never
have caught it. That is the same asymmetry recorded under
`fleet_llvm_assertion_asymmetry`: the host that can falsify a claim is often
the one nobody runs the check on.

---
## Cross-backend sync `AUTOTUNE-SEPARATION-NVIDIA-2026-09-01`

**Owning item:** `AUTOTUNE-SEPARATION`, NVIDIA half ·
**synchronization key:** `AUTOTUNE-SEPARATION-NVIDIA-2026-09-01`

**The corpus was re-raced on sm_120 with #663's separation verdicts recorded,
and 42 of 51 freshly-raced rankings (82%) turn out to be unsupported.** The
earlier estimate — "11 rows with margins under 2%" — understated it badly,
because a margin cannot be judged without the noise beside it. That is the
whole content of #663, now measured rather than argued.

**Two recorded verdicts are retired by evidence, not by opinion.** At 512³ and
1024³ device-timed matmul, the compiler-**emitted** PTX lane wins by ~38%
against **0.15–1.86%** noise, racing the full four-candidate field. The prior
rows named a *tile* lane and pinned the 1024³ field to exactly two candidates —
which encoded the biased race #655/#662 removed: the GEMM lanes had no device
timer, `_measure` scored them `inf`, and they lost silently. "The tile lane
wins" meant "the tile lanes were the only ones that could be timed".

**`device_repeats=3` overstates the noise floor it reports.** Measured at
128×512×64 bf16: sd **48.31% / 30.74% / 19.34%** over 3 / 10 / 30 whole
measurements, with a 2.3× min–max range even at 30. The lane genuinely is ~19%
noisy, so the *unseparated* verdicts hold either way — but a recorded floor
2.5× the truth is a number someone will act on. The corpus recorder now uses
10; `measured_arbitrate` keeps 3, which is the right cost trade for runtime
selection rather than published evidence.

**An independent mechanism agrees, which is what makes this trustworthy.**
`finalize_test5_corpus` replaces a row only when **two** runs pick the same
winner. The one row it refuses — `bfloat16 [128, 256, 64]` device — is exactly
the row separation flags at margin 9.92% against 102.96% noise. Two checks
built years apart, from different premises, rejecting the same ranking.

**Outcome for this backend: `parity validated` — measured on Super-Bear
(RTX 5070 / sm_120), and the corpus is committed.**

**Two near-misses on the way, both of which would have destroyed evidence
silently.**

*Without `--warm-start`, regenerating deletes other boxes' rows.* The recorder
writes the whole cache, so a bare run dropped **all 12 `rocm:gfx1151` rows** and
13 NVIDIA rows at shapes the default flags do not cover — 25 rows of
exact-device evidence, with no error. Re-run with `--warm-start`: 0 lost.

*Recording is two runs plus a finalizer, not one run.* `stable_runs == 2` is
literally two independent measurements agreeing;
`record_autotune_corpus.py` alone produces rows with no `evidence` block at
all. A single-run corpus therefore carries **less** evidence than the one it
replaces (41 rows → 0) while looking like an update. Done properly the count
goes 41 → **84** and nothing is lost.

**Both traps share a shape worth naming: a regeneration that succeeds while
producing weaker evidence than it replaced.** Nothing failed, nothing warned;
only a before/after row-and-evidence count catches it. Any future corpus
regeneration should print that diff before committing.

**Three consumer tests were rewritten to the measured reality**, and each now
pins a property the old assertion could not express:

* the 1024³/512³ rows assert the **full four-candidate field** and that the
  winner's margin is **separated** — not merely that a particular name won;
* the stability matrix asserts **eligible ⟹ stable** rather than "every row is
  stable", because reproducibility at the launch-overhead floor is not a
  property this hardware has;
* the fused/attention rows assert a **subset** of the known candidate pair,
  because `applies_to` makes them mutually exclusive by contract (below).

All three are mutation-verified: shrinking the field, un-separating the verdict,
and offering an unstable row to the selector each fail.

**Found on the way, and it is the `applies_to` item:**
`NvidiaGenericCudaCandidate.applies_to` returns True only when the epilogue
contract *selects* it, so `nvidia_generic_cuda` and `nvidia_mma_fused` are
never competitors in one race. Such a row records `unmeasured: {}` — "nothing
was skipped" — which is true and misleading: the other candidate was not
skipped, it was **contract-excluded**, and no field distinguishes "one
candidate exists" from "a second was excluded before the race". `unmeasured`
(#655) closed the *timing* half of this; the *applicability* half is still open.

---

## Cross-backend sync `AUTOTUNE-SEPARATION-ROCM-2026-09-01`

**Owning item:** `AUTOTUNE-SEPARATION`, ROCm half, and the dispatch tightening
it unblocks · **synchronization key:** `AUTOTUNE-SEPARATION-ROCM-2026-09-01`

**All 16 gfx1151 rows now carry a verdict — zero never-asked**, where all 12
previously had `separation: None`. 15 separate cleanly (margins 34–99.9%
against 0.10–14.24% noise); the single refusal is `paged_kv_decode 8192
end_to_end` at 6.52% margin vs 5.59% noise, which is genuinely marginal.

**That is the opposite of sm_120's 82%-unsupported result, and the reason is
structural rather than a hardware difference.** ROCm races two candidates that
are far apart (generic HIP vs WMMA); NVIDIA races four that often sit within a
few percent. A backend with a *narrow* field gets clean verdicts almost for
free — which is worth knowing before reading either number as a quality signal
about the backend.

Verified the numbers come from `device_event`, not the wall-clock fallback this
backend is prone to, so the noise floor is genuine rather than inflated.

**The dispatch rule is now tightened, and it is deliberately not "reject
`None`".** `corpus_winner` refuses a row that ranks **two or more** candidates
and has no verdict. `separation_verdict` returns `None` when fewer than two
were timed — a sole candidate is chosen by *applicability*, not by a race, and
has no margin to defend. Refusing those would be a category error: 12 of the 23
remaining `None` rows are exactly that shape. `inf` is likewise not a
competitor (it marks "could not be timed"), so a row with one latency and one
`inf` is a sole-candidate row wearing a pair's clothes.

Committed corpus: **113 rows — 67 refused as dispatch hints, 34 with a
supported verdict, 12 sole-candidate.**

**Outcome for this backend: `follow-up required` — 11 sm_120 rows are now
inert.** The tightening refuses them: they rank two or more candidates and
carry no verdict, at shapes the recorder's default flags do not cover
(`attention`, `conv2d`, `ssm_replay_decode`, and a handful of matmul buckets).

**They are recoverable by widening the recorder's shape flags and re-running**,
which is a mechanical job on Super-Bear rather than a design question. Until
then those buckets fall back to lead-safe tier priority, which is the correct
degraded behaviour and not a regression — before this change they served a
ranking nothing had checked.

**Read the ROCm/NVIDIA contrast carefully.** 15/16 separated here against 19/74
there is *not* evidence that gfx1151 measurements are better. It reflects field
width: two far-apart candidates separate almost automatically, four close ones
rarely do. A backend that adds candidates should expect its separated fraction
to fall, and that is the field getting more honest, not the hardware getting
worse.

## Cross-backend sync `APPLIES-TO-SHAPE-BLIND-2026-09-01`

**Owning item:** `applies_to(region)` shape-blind (NVIDIA plan, "Follow-up
owned here", now closed) · **synchronization key:**
`APPLIES-TO-SHAPE-BLIND-2026-09-01`

**Shared contracts changed** — arbiter selection and autotune measurement, so
all four backends are assessed here per AGENTS.md:

* `Candidate.applies_to_inputs(region, *inputs)` — new, additive, defaults
  `True`. Every existing `applies_to` implementation is untouched.
* `candidate.live_candidates(region, op, target, inputs)` — one statement of
  "who is racing", replacing the copy that `arbitrate`, `measured_arbitrate`
  and `corpus_winner` each kept.
* `arbitrate(..., inputs=())` — additive keyword; omitted, selection is
  byte-for-byte what it was.
* `autotune._measure` now reads the execution tag it was already producing.

**The defect: a region carries structure, not dimensions.** `MatmulRegion` has
a dtype and transpose flags; `FusedRegion` an epilogue chain. M/N/K arrive with
the operands and are inferred separately. So `applies_to(region)` could not
express "aligned shapes only", and the F4 oracle could not cover for it either
— its probe shape is fixed (32×16×32 for matmul) and its verdict is cached
under a key with **no shape in it**. An aligned-only lane therefore declined
inside `run`, by returning the numpy reference, *after* it had already won.

**Two harms, both reproduced against the real
`NvidiaMmaGemmEmittedCandidate` before the fix:**

| | before | after |
|---|---|---|
| ragged-shape winner | `nvidia_mma_gemm_emitted`, tag `reference` | the lane that can run the shape, real tag |
| its recorded latency | `0.00525 ms` (numpy) vs a real `0.00196 ms` rival | absent from the field |

The second is the one that propagates. A fabricated latency does not sit
inert — it **ranks**. With the backstop disabled the same record comes back
`separated: True, margin 0.59, runner_up: nvidia_mma_gemm_emitted`: the
separation machinery from #663/#671 confidently certifies a 2.4× loss for a
kernel that never ran. Separation judges the numbers it is given, and this is
a second, independent mechanism for the corpus bias recorded in
`NVIDIA-TIER-PRIORITY-IS-WRONG-AT-SCALE-2026-08-30` — where a biased race hid
the fastest kernel in the registry.

**The tag was already there and nothing read it.** The D3 arbiter log has
described this since it was written: *"the arbiter selects a candidate, but
that candidate's `run` may still decline to the numpy reference at execution
time (a device error, **an unsupported shape**) — a silent degrade the tag
reveals."* Observability without a consumer, which is Decision #29 wearing a
diagnostic's clothes.

**Fix, layered deliberately.** `applies_to_inputs` fails **open** on absent or
malformed operands — the question cannot be answered there, and refusing would
disable every shape-anonymous caller, while a malformed pair is an operand
error that must still raise through `run` rather than be silently excluded
(Decision #21). The fail-**closed** backstop sits one level down: `_measure`
refuses to record a latency for a run that came back with a reference tag,
whether or not the candidate declared itself. A lane that never adopts the
hook is therefore still safe from the fabricated-measurement half.

**Host-free evidence:** `tests/unit/test_arbiter_workload_applicability.py`
(10 tests). Mutation-verified — four independent mutations, each killing only
its own tests: disabling the selection filter (3 fail), disabling the tag
backstop (1), removing the NVIDIA producer (1), removing the ROCm producer (1).

**Outcome for this backend: `follow-up required` — the producer landed, the
device proof has not.** `NvidiaMmaGemmEmittedCandidate.applies_to_inputs` now
states the aligned-only contract its own class docstring has always carried and
its `measure_device_latency` already enforced; the declaration was the only
place the three disagreed. Its 20 sibling lanes were audited and need nothing:
their run-time guards test `ndim`/contraction agreement, which is operand
*validity*, not shape *support*, and those correctly stay in `run`.

Owed from Super-Bear (RTX 5070 / sm_120): re-race a ragged bucket and confirm
(a) the emitted lane is absent from the field rather than present with a numpy
latency, and (b) the ragged winner now carries a real execution tag. Neither is
claimed here — this Mac has no CUDA, so the aligned F4 probe declines for the
wrong reason and cannot falsify the device-present case. Harm 2 above was
therefore reproduced under a simulated device (aligned path stubbed to succeed,
ragged left to decline), and is labelled as such.

**Blast radius, measured: no committed corpus row is invalidated.** Every
persisted matmul bucket is aligned (0 of 28 rows ragged) and every attention
bucket uses head_dim 64 (0 of 18 ragged), so neither producer's decline was
ever exercised during a recorded race. The defect was live in the *dispatch*
path and had not yet reached the evidence.

That is luck, and it points at the real gap: the recorders only ever race
power-of-two shapes, which is why nobody hit this and also why **the ragged
path has no measured coverage at all**. The device follow-ups above should
add a ragged bucket rather than only re-checking an aligned one — a re-race
of the existing buckets would pass identically before and after this change.

**Two follow-on findings from the same code path, both now covered.**

*The two consumers had to move together, and it is not obvious why.* Buckets
are coarse — `bucket_key` maps both `(24,12,20)` and `(32,16,32)` to
`(32,16,32)` — so a ragged workload genuinely reads the aligned workload's
corpus row, and `run_arbitrated` passes a corpus hint to `arbitrate` as
`force`, which restricts to that one name. Making `arbitrate` shape-aware
*without* `corpus_winner` therefore converts the silent degrade into an
`ArbiterError` (verified, not inferred). `corpus_winner` withholds the hint
because its own `live` set excludes the lane — that coupling is now pinned by
a regression test rather than left to be rediscovered.

*The `force` diagnostic named the wrong gate.* One message — "not available" —
covered not-registered, wrong-region and unavailable-here alike, and once a
shape axis existed it was actively wrong for the commonest case: the lane IS
available, on a host that has it, for a shape it cannot serve. It now names
which of the four gates rejected the candidate (Decision #21).

## Cross-backend sync `ROUTE-LEDGER-RULES-UNCONSUMED-2026-09-01`

**Owning item:** Apple strict route ledger re-seal · **synchronization key:**
`ROUTE-LEDGER-RULES-UNCONSUMED-2026-09-01`

Apple's `promotion_rules` block turned out to be a declaration no code read:
sealed into twelve ledgers for audit, and `status: "promote_candidate"` was
self-certifying at load. Fixed there (see the Apple plan). All four backends
are assessed because the *pattern* is what travels, not the Apple code.

**Outcome for this backend: `follow-up required` — the same gap exists here,
in a milder form.** `nvidia_sm120_legacy_retune.json` and
`nvidia_sm120_low_precision_native_routes.json` declare `noise_policy`
(0.03 / 0.04) and `selector_promotions`, and the ratchets assert those values
**equal a constant** without ever comparing a promoted row's margin against the
policy the file declares. So a promotion inside the noise band would pass every
existing check.

Credit where due: `test_nvidia_low_precision_native_routes.py` is otherwise
well built for this class — it cross-checks the summary count against the rows,
explicitly guards the vacuous-pass case (`len(promoted) == selector_promotions
> 0`), and requires timing-domain consensus plus resource fingerprints on every
promoted row. The missing piece is exactly one: **the margin is never held to
`noise_policy`.** That is a smaller job than Apple's was, and it needs a
decision about what "margin" means for these rows before it is written.
## Cross-backend sync `PROMOTION-EVIDENCE-REDERIVED-2026-09-01`

**Owning item:** the NVIDIA/ROCm half of
`ROUTE-LEDGER-RULES-UNCONSUMED-2026-09-01` ·
**synchronization key:** `PROMOTION-EVIDENCE-REDERIVED-2026-09-01`

Apple's `promotion_rules` was a declaration nothing read; its loader now
re-derives each promotion from the evidence the ledger retained. The same
question was then put to the other backends' evidence artifacts, and the
answers differ enough to be worth stating one by one.

**A near-miss that shaped the whole exercise, recorded because it would have
been convincing.** The first NVIDIA checker modelled promotion as *"the winner
beats the runner-up by more than `noise_fraction`"* — the obvious reading — and
it flagged **7 of 11 committed promotions as violations**, complete with a
tidy table. Reading the producer settled it: `finalize_low_precision_native_routes._near`
promotes a candidate that is **within** `noise_fraction` of the fastest in
every run of every domain, tie-broken by total time. Every one of those 7 is
correct under the rule that was actually applied. A plausible model of someone
else's gate, checked against committed evidence, produces confident and wrong
findings — so both checkers here mirror their producer's own predicate rather
than a reasonable-looking substitute.

**Outcome for this backend: `parity validated` — both evidence files now
re-derive, 0 mismatches.**

* `nvidia_sm120_low_precision_native_routes.json` — every recorded conclusion
  (`near_winner_consensus`, `run_winners`, `timing_domain_consensus`, `winner`,
  `selector_promoted`) is recomputed from `timings`, the only field that is
  measurement rather than verdict. **18 rows, 11 promotions, 0 mismatches.**
* `nvidia_sm120_legacy_retune.json` — `noise_policy` was asserted to *equal
  0.03* and never compared to a measurement, so a regressed recording whose two
  runs disagreed by 40% would keep `stable: true` and pass. Stability is now
  `|run0 - run1| / max(run0, run1) <= noise_policy` per domain, re-derived, plus
  cross-candidate winner consensus per case. **8 rows, 4 cases, 0 mismatches.**

Both confirm their recorders rather than accusing them. Checkers live in
`tests/_support/nvidia.py`; each is mutation-verified against forged rows
(invented winner, stripped resources, widened consensus, lying `run_winners`,
drifted runs, deleted timings).

## Cross-backend sync `MATRIX-LANE-RAGGED-SHAPES-2026-09-01`

**Owning item:** the matrix-core lanes decline ragged shapes ·
**synchronization key:** `MATRIX-LANE-RAGGED-SHAPES-2026-09-01`

**One gap, found three times while fixing other things.** Every backend's
matrix-core lane — the *fast* one — declined ragged shapes and fell back to a
much slower path:

| backend | lane | gate | fallback |
|---|---|---|---|
| NVIDIA sm_120 | emitted `mma.sync` GEMM | `M%16, N%8, K%16` | numpy |
| ROCm gfx1151 | WMMA flash-attention | `head_dim % 16` | numpy |
| Apple M1 Max | coopmat `simdgroup_matrix` reduce | `N % 8` | scalar path |

What made it worth doing now is the measurement from the corpus work: the
compiler-**emitted** PTX GEMM beats the hand-tuned delegate **1.5–1.7x at every
shape** on sm_120. The alignment gate was not costing a little — it was
excluding the fastest kernel in the registry from every ragged workload. The
`applies_to_inputs` declines added in #672 made that *honest*; they did not make
it *fast*.

Both fixes share one idea and split on which axis a dimension plays:

* **A contraction dimension is zero-padded** — exact, because a zero operand
  contributes nothing to the dot product.
* **An output dimension is store-suppressed** — zero-padding is wrong there; a
  lane past the edge would write a correct-but-zero value into someone else's
  slot.

Getting that backwards is silent corruption rather than a fault, which is why
it is stated as the rule rather than left implicit in each kernel.

**Outcome for this backend: `parity validated` — measured on Super-Bear
(RTX 5070 / sm_120).** `emit_mma_sync_gemm_ptx` now clamps its M/N load index
and suppresses the out-of-range store, so the hot K loop is byte-for-byte the
proven aligned kernel: no per-iteration predicate, no divergence, cost confined
to a few instructions outside the loop. The K remainder is a genuinely
predicated extra slab.

**25 ragged shapes x 2 dtypes, 0 failures, <= 1.3e-7 relative error**; both
dtypes assemble under `ptxas --gpu-name=sm_120a`.

**K must stay EVEN, and that is hardware, not laziness.** The first run faulted
and `compute-sanitizer` named it: `CUDA_ERROR_MISALIGNED_ADDRESS`, not an
out-of-bounds. `ld.global.b32` needs a 4-byte-aligned address while the
fragments address 2-byte elements, so `row*K + k` must be even — with K odd,
every odd row starts misaligned, in the **main loop** and not only the tail.
Every data point fits once that is known: 24x12x20 passed (K even), 16x8x17
faulted (K odd), 1x1x7 passed (row 0 only). Lifting it needs a padded row
stride (the strided ABI could already carry it) or a paired `ld.global.u16`
slow path; both are real designs and neither is a one-line change, so K%2 is
recorded as the boundary rather than claimed as working.

**A dead seam turned out to be exactly this work.** `invokeMmaGemm16` has had a
`bool ragged = false` parameter since it was written that **no caller ever
passed true**, so the guard behind it rejected every unaligned shape with rc=5.
Decision #29's unconsumed declaration, in C++.

**The benchmark path needed the same guard, separately** — it does not route
through `invokeMmaGemm16`, so a first fix placed only there let an odd-K
*measurement* fault the device, which then poisons the CUDA context for every
later launch in the process. The autotuner calls that path. The capability now
lives in `tileLaunchConfig`, the per-kernel geometry table, so both paths get
one answer; the Tile-direct and scheduled kernels stay aligned-only, which is
why it is per-entry rather than a global relaxation.

**Aligned-path regression, measured rather than asserted** (A/B against the
unmodified emitter and launcher, same box, same session):

| shape | baseline ms | ragged ms | spread |
|---|---|---|---|
| 512³ | 0.02678 | 0.02762 | 1.1–1.5% |
| 1024³ | 0.19358 | 0.19764 | 0.3–0.5% |
| 2048³ | 1.48253 | 1.49670 | 0.3–0.5% |

**1–3% at large shapes** — real, outside noise, and the price of ragged shapes
going from numpy to full speed. At 64³/256³ the first sample looked like a 1.4x
regression, but against **52–58% spread**, so not a supported comparison; a
15x3000-rep re-measure pushed the spread *worse* (85%, 73%) and the medians
crossed over. Those shapes are unresolvable on this box and the apparent 1.4x
was noise, exactly as its own spread warned.

## NVIDIA-RAGGED-TIMING-FIELD-2026-09-01: the ragged flag was doing two jobs *(fixed)*

**Review finding on #675, verified rather than accepted — and it is a
regression that PR introduced.** `tileLaunchConfig` defaulted `ragged` to false
and set it true only for the two PTX GEMM entries. But **every** entry the
invoke dispatch routes through `invokeMmaGemm16` passes `ragged=true`: the
Tile-direct, Tile-shared and scheduled-SM120 kernels have masked their
out-of-bounds loads and stores in `NVIDIALowering.cpp` since they were written.
The comment #675 added — *"the Tile-direct and scheduled kernels are still
aligned-only"* — was simply false.

**Measured A/B on Super-Bear**, same PTX, same shapes, two libraries:

| kernel | shape | merged main | fixed |
|---|---|---|---|
| `tile_matmul_direct_f16` | 256×256×256 | accepted | accepted |
| `tile_matmul_direct_f16` | 255×129×258 | **rc=5 refused** | accepted + timed |
| `tile_matmul_shared_f16` | 255×129×258 | **rc=5 refused** | accepted + timed |

A refused measurement is a candidate that returns `None` and leaves the race —
the exact corpus bias `_record_raced_the_live_field` exists to refuse, and the
one this session spent PRs #655/#662/#670 removing. #675 put a small version of
it back on the ragged path.

**Two capabilities, not one.** Even-K is a property of *this emitter's fragment
layout*, not of ragged shapes: `ld.global.b32` needs a 4-byte-aligned address
while the fragments address 2-byte elements. The Tile kernels declare
**element alignment (2)** on their masked loads
(`unsigned alignment = (f16Storage || bf16Storage) ? 2 : 4;`), so an odd
element offset is legal for them. Folding the two into one flag would have
imposed an odd-K refusal on kernels that have no such limit — trading one
regression for another. `tileLaunchConfig` now reports `ragged` (default
**true**) and `requiresEvenK` (true only for `kGemmF16`/`kGemmBf16`).

**One honest limit on the evidence.** The A/B above proves the *gate* changed,
using real PTX that JITs — a first probe used a stub that failed to compile,
and rc=3 (JIT failure) happens *before* the shape guard, which made that run
silently inconclusive rather than informative. What is **not** separately
device-proven here is that the real Tile kernels compute correctly at odd K;
that rests on the alignment-2 masked loads above **and** on the fact that
merged main's own invoke path already dispatches them at odd K. This change
makes the benchmark path agree with the invoke path rather than making a new
claim about the kernels.

Follow-up: a ragged-bucket device re-race is still owed here (recorded under
`MATRIX-LANE-RAGGED-SHAPES-2026-09-01`), and it should now include the Tile
lanes, which this fix returns to the field.

## MATRIX-LANE-RAGGED-SHAPES device evidence *(sm_120, 2026-09-01)*

The re-race owed by `MATRIX-LANE-RAGGED-SHAPES-2026-09-01`, run on Super-Bear
against `origin/main` at #676's merge. Two claims were owed and they are
different: that the emitted lane now *measures* at ragged shapes (it declined
before #675), and that the Tile lanes are still *in the field* (#676 — the
benchmark path refused them as #675 landed it).

**The field is complete at every shape: 4 timed / 0 absent.** At odd K the
emitted lane is not *absent* from the race but not *live* — `applies_to_inputs`
excludes it, which is the honest form: 3 live, 3 timed, 0 absent.

**And three of the four shapes have no supportable winner.** Nine whole-device
repeats per candidate, medians with spread beside them:

| shape | fastest | margin | noise | separated |
|---|---|---:|---:|---|
| 256³ aligned | `mma_gemm_emitted` | 14.9% | 14.6% | **no** |
| 255×129×258 ragged | `mma_gemm_emitted` | 7.6% | 17.9% | **no** |
| 100×50×70 ragged | `tile_matmul_direct` | 3.6% | 13.2% | **no** |
| 1000×999×1002 ragged | `mma_gemm_emitted` | **19.9%** | **0.1%** | **yes** |

So the only ranking this run supports is the large ragged one: at
1000×999×1002 the compiler-**emitted** PTX GEMM beats `tile_matmul_direct` by
19.9% (0.18343 vs 0.22910 ms) and the shipped delegate by 34%, at 0.1% spread.
The small shapes are launch-overhead dominated and unresolved — the same
conclusion the 64³/256³ A/B reached under `MATRIX-LANE-RAGGED-SHAPES`, reached
again by a different route.

**Why the field composition mattered, concretely.** At 100×50×70 the ordering
puts `tile_matmul_direct` first. Under #675-as-merged both Tile lanes were
refused by the benchmark path, so that shape would have raced two candidates
instead of four and recorded the emitted lane as winner with no visible
competitor. The verdict would not have been provably wrong — it is unseparated
either way — but it would have been drawn from a field missing the candidate
that happens to lead it.

**Method notes, because two of them nearly produced a false result.** The first
run reported `0 timed / 0 absent of 0 live` — an empty registry, not a device
answer: the scratch worktree lacked `libtessera_nvidia_gemm.so`, so every
candidate probed unavailable. The second run had 2 of 4 live because the Tile
lanes additionally require `tessera-nvidia-opt`, `mlir-opt`, `mlir-translate`
and `llc`; pointing `TESSERA_NVIDIA_OPT` at the box's existing build (my changes
touch no MLIR pass) completed the field. **An empty or partial field reports as
a clean-looking table**, which is exactly why the harness prints
`N timed / M absent of L live` rather than just the winner.

Run from a detached `git worktree` at `origin/main` so the box's own checkout,
branch and 16 untracked study files were never touched; the worktree was
removed afterwards and the box verified back to `verify/sep` with the same 16.

## Cross-backend sync `PRIMITIVE-ROUTE-MAP-2026-09-01`

**Owning item:** the coverage ↔ MLIR/LLVM route join
(`generated/primitive_route_map.md`) · **synchronization key:**
`PRIMITIVE-ROUTE-MAP-2026-09-01`

A shared registry and generated dashboard reporting, per primitive and per
target, whether the mainline compiler or the Python bootstrap packager serves
it. All four backends are assessed here per AGENTS.md — the first landing
(#677) updated only the NVIDIA plan, which is the omission this entry closes.

**It shipped with six false rows, and how they got there is the useful part.**
`depth_attn` was published `compiled` on NVIDIA and x86 although `driver.py`
dispatches it only under `target_kind == "rocm_gfx1151"`; `min`/`amin` were
published as ROCm and x86 reduction routes although both contracts accept only
`sum/mean/max/amax`. Both came from the same mistake: a membership that was
described as "grounded in the `tessera.*` literals each backend names" but was
actually one global tuple applied to every target, plus a compiled-route
fan-out whose comment claimed it "keeps the claim no stronger than the source"
when the source is target-aware and the fan-out made it strictly stronger.

Three guards now make that class of error mechanical rather than editorial:

* **membership is per target**, and every declared member is cross-checked
  against the `tessera.*` literals of that target's own native module (either
  the canonical Graph IR name or the public alias — `sum` is `tessera.reduce`
  in coverage and `tessera.sum` in `x86_native`, and both are real);
* **a compiled route is claimed only where `driver.py` dispatches it**
  (`COMPILED_ROUTE_TARGETS`), never fanned across targets;
* **a target that cannot be classified says so** rather than vanishing.

**Outcome for this backend: `parity validated`.** `nvidia_sm120` is fully
classified by the audit. Its rows were corrected: `depth_attn` is no longer
claimed here (rocm-only dispatch), while `min`/`amin` legitimately remain —
`nvidia_native` is the one reduction contract that names them.

## LAUNCH-OVERHEAD-BOUND-1 — cross-backend assessment (recorded 2026-09-02)

`tests/_support/launch_overhead.py` (PR #686) is shared test infrastructure: it
bounds `rt.launch` overhead against the `execution_kind` the launch reports.
A device dispatch gets a flat ceiling; every other lane keeps the
self-calibrating `max(2.0, direct_ms*4)` against the oracle arm. Review on #686
asked for a per-backend verdict, and the four are NOT the same.

**NVIDIA — MEASURED 2026-09-02; sm_120 needs its own constant, and inheriting
the ROCm one would have been hollow.** The follow-up this entry asked for is
done. Mirroring the ROCm methodology exactly, on The Super-Bear (RTX 5070,
sm_120, WSL2), using `nvidia_dequant_gemm_compiled` — the direct analogue of the
ROCm `dk4` row:

| | gfx1151 (the constant's origin) | sm_120 |
|---|---|---|
| direct oracle median | 0.078 ms | 0.072 ms |
| first launch (one-time compile) | 642 ms | **2320 ms** |
| steady median, idle | 3.343 ms (max 4.703) | **0.965 ms** (max 1.379) |
| steady median, 10 busy cores | 4.752 ms (max 5.541) | **0.951 ms** (max 1.245) |
| dominant per-launch cost | `_rocm_dev_in`, 13 H2D copies | `compiler/emit/nvidia_cuda.py`, host-side |

`execution_kind` is `native_gpu` on both, so the lane gate itself needs no
change.

**Three findings.** sm_120 dispatch is **3.5x faster** than gfx1151, so
inheriting `NATIVE_LAUNCH_CEILING_MS = 20.0` would have been 14.5x the worst
value ever observed here — a bound that could not fail, which is exactly the
hollow-gate shape the lane gate exists to prevent. Second, NVIDIA is essentially
**load-insensitive** (0.965 -> 0.951 ms under ten busy cores) where ROCm
degrades 42% (3.343 -> 4.752): the ROCm cost is host-to-device transfer that
contends for CPU, while NVIDIA's is host-side emission. Third, the one-time
compile is 3.6x *longer* on NVIDIA (2320 ms vs 642 ms), which matters for any
row that times a cold launch instead of steady state.

**The constant to use on adoption is 5.0 ms**, by the same criterion as the ROCm
one: 3.6x the worst observed (1.379 ms) while still failing the 70.6 ms
regression class — and here with **14x** margin below it, where ROCm could only
manage 3.5x. Both of the original row's criteria are satisfiable on this
backend; on ROCm they were not.

Deliberately NOT added to `tests/_support/launch_overhead.py` yet: no NVIDIA row
consumes it, and an unconsumed constant is Decision #29's case. Add it with the
first adopting row, not before.

## MSW-4A-CODIFF-SIGN-1 — cross-backend assessment (recorded 2026-09-02)

`ga.calculus.codiff` changed from the unsigned `⋆d⋆` composition to the true
codifferential `δ = (-1)^(n(k+1)+1) ⋆d⋆` (PR #688), and its `clifford_codiff`
VJP changed with it. That is a shared numerical contract, so each backend gets
an explicit verdict.

**NVIDIA — not applicable; there is no NVIDIA codiff to correct.** No
`clifford_codiff` entry exists for this backend in `backend_manifest.py` and no
CUDA kernel implements the operator; a JIT'd Clifford program on NVIDIA
executes `tessera.ga.*`, which is the signed Python path. If a native NVIDIA
codiff is ever added it must apply the sign at its ABI boundary the way the
Apple symbol now does — the exported name promises δ, and the mistake this
entry records is precisely a symbol named `codiff` returning `⋆d⋆`.

## Cross-backend sync `DISPATCH-BREAKER-RESIDENT-2026-09-03`

**Owning item:** `APPLE-DISPATCH-WEDGE-1` (Apple plan) · **synchronization
key:** `DISPATCH-BREAKER-RESIDENT-2026-09-03`

**Shared contract changed.** `runtime._apple_gpu_run_checked` gained an
optional `silent_failure_timeout_s`, and a new
`runtime._apple_gpu_device_call_checked` routes the eight device-resident
(`DeviceTensor`) Apple dispatch paths through the dispatch circuit breaker.
Shared test infrastructure changed with it: `test_apple_gpu_dispatch_breaker.py`
gained a per-dispatch AST drift gate. Nothing outside the `_apple_gpu_*`
namespace is touched, and no IR, ABI, dtype, diagnostic code or benchmark
schema changes.

**The premise that makes this Apple-shaped.** The breaker exists because Metal's
`waitUntilSignaledValue:timeoutMS:` **returns** when its deadline expires. The
caller then falls back to host, the next dispatch asks the device again, and
each one pays the full 30 s — an observed 70-minute sweep against a 4-minute
healthy run. The repeated cost, not the hang, is what a breaker cuts.

**NVIDIA — not applicable for the breaker; follow-up required for the bounded
wait.** CUDA has no timeout-bearing wait to return early from:
`cuda_backend.cpp:193` is `TSR_CUDA_CHECK(cudaStreamSynchronize(...))`, which
blocks until the device answers or the driver's own watchdog resets the
context, and reports through an error code rather than a deadline. So the
"returns, falls back, is asked again" cycle this breaker cuts cannot occur, and
porting it would guard nothing. The *inverse* gap is real and is the follow-up:
an unresponsive sm_120 device produces one unbounded block with no Tessera-side
diagnostic, which is strictly less recoverable than what Apple had. Assess a
bounded wait for the PTX bridge under this key before adopting any of this
code. No exact-device evidence is claimed on Super-Bear; none is owed, because
no NVIDIA code changed.

**Extended 2026-09-03 for the runtime-side follow-ups (#710, #711).** The
earlier note above covers the Python dispatch helpers. Three further contract
changes landed in the Apple runtime itself, and each is assessed here:

1. **A bounded wait now publishes timeout kind 1.** `ts_enc_commit_wait` used
   to print an expiry to stderr and touch nothing, so the Python accounting had
   to infer a stall from wall time. It now reports on the shared error channel,
   as the other bounded waits already did.
2. **Each bounded wait owns its event.** Both Apple wait helpers reserved
   increasing values on one context-wide `MTLSharedEvent` under a lock released
   before commit, so a later dispatch could signal first and satisfy an earlier
   waiter while its own command buffer was still running.
3. **A timed-out dispatch quarantines its pooled buffers.** A guard whose
   acquire predates a timeout drops its buffer instead of returning it to the
   shared pool, since the stalled command may still read or write it.

**NVIDIA — not applicable, for three separate architecture-specific reasons;
the existing bounded-wait follow-up is unchanged.**

1. *Nothing to publish.* CUDA still has no timeout-bearing wait to report from:
   `cuda_backend.cpp:193` is `cudaStreamSynchronize` and
   `cuda_backend.cpp:247` is `cudaEventSynchronize`, both blocking with
   error-code reporting and no deadline. A timeout kind has no source here.
   This is the same gap the follow-up above already owns, not a new one.
2. *No shared-event hazard.* `createEvent` (`cuda_backend.cpp:199`) allocates a
   fresh `cudaEvent_t` per event object and `EventSynchronize` waits on that
   object, so there is no context-wide counter for a concurrent dispatch to
   overshoot. The Apple defect came from one event plus increasing values;
   CUDA's ownership model rules it out by construction.
3. *No pool to quarantine.* The recycling buffer pool is Apple-only — it lives
   in `apple_gpu_runtime.mm` and has no counterpart under `src/runtime/`, so
   there is no allocator that could hand a stalled dispatch's memory to the
   next one.

**Validation performed:** none on device for this key; these are structural
readings of the CUDA backend, and no NVIDIA code changed. **Missing exact-device
evidence:** none required — no NVIDIA behaviour is claimed.

**Extended again 2026-09-03 — the event-less fallbacks are bounded.** Four
Apple waits fell back to an untimed `waitUntilCompleted` when `newSharedEvent`
failed. That is the case where the device is already in trouble, so the one
path taken *because* it was unhealthy was the only one that could hang
forever. They now poll `status` against a deadline and, on expiry, report
timeout kind 1 and quarantine their pooled buffers.

**NVIDIA — not applicable, and for a reason worth stating precisely.** This
change bounds a *fallback* taken when a bounded primitive is unavailable. CUDA
has no bounded primitive to fall back FROM: `cuda_backend.cpp:193` and `:247`
are `cudaStreamSynchronize` / `cudaEventSynchronize`, both untimed. So CUDA is
not in the state this fixes — it is in the state the standing follow-up
already owns, which is that its only wait is the untimed one. Porting the poll
loop would be that follow-up's work, not this one's, and would need its own
device evidence.

**Validation performed:** none on device; no NVIDIA code changed. **Missing
exact-device evidence:** none required — no NVIDIA behaviour is claimed.


## Cross-backend sync `FRONTIER-MSW-2026-09-04`

Owners: `FRONTEND-IR-MEDIUM-1`, `APPLE-DISPATCH-WEDGE-1`,
`APPLE-MOE-ROUTE-1`, and `MSW-5` through `MSW-9`.

Shared changes: pending metadata snapshots are verified before a new record and
the preceding boundary's drop declarations are retired; einsum execution and
AD/trace recording consume one alpha-normal equation; sampled reference fields
carry orthogonal coordinate contracts with registered diagnostic
`FIELD_COORDINATE_CONTRACT`. Coordinate laws extend the existing field-calculus
registry. MSW-6/MSW-8 are host-free examples; MSW-9 is a design spike, with the
native program-pair evaluator/fusion gate still open. No new ODS operation,
backend candidate, physical schedule or native metric ABI is introduced.

**nvidia outcome:** Shared-contract parity validated on host-free tests only. The NVIDIA production metadata recorder consumes and retires a pending frontier obligation before taking its own boundary snapshot. No CUDA kernel, timing route or storage contract is changed; exact-device performance/numerics are not claimed.

Validation records live in `MATH_SOURCE_WORKSTREAM.md` and the focused review
fixtures. The metadata pass was rebuilt on WSL; the positive/negative lifecycle
fixtures pass. Reference proof does not transfer exact-device status.

## Cross-backend sync `FRONTEND-RECIPE-2026-09-04`

Owner: `FRONTEND-IR-MEDIUM-1`, staged rank/prune acceptance. Shared contracts:
AST location fidelity and location-sensitive capture caching; argument-local
symbolic dimension consumption; opt-in native parametric CSE with one recipe
digest across constraint-checked buckets; strict source-loop matmul candidate
raising. Existing passes and operations are reused; execution selection is
unchanged. The validation spine now binds checkout fingerprints, and coverage
conflicts are regenerated after authored inputs are resolved.

NVIDIA outcome: follow-up required for any future recipe-based native selection. No CUDA execution or performance evidence is claimed.

## Engineering sync `EVAL-TELEMETRY-IKF-2026-09-04`

Owners: MSW-9, APPLE-DISPATCH-WEDGE-1 and IKF-1. Program-pair evaluator
comparison keeps both native provenance gates; ANN composition and ReLU
identity checks are separately registered reference laws. IKF-1 is now bound
in the integrated plan: P0 timing first, P2 after validated clocks and
Schedule-Object region identity, P1 host schema/math independently.

NVIDIA: follow-up required. No native ANN/IKF evidence transfers from WSL
logic tests. Bounded device wait must include context poisoning and retention
of outstanding resources before replacing CUDA synchronization; host API
calls that themselves block need an owning process boundary.


## Cross-backend sync `EVIDENCE-POLICY-20260904`

Decision #26 coverage snapshots move to revision-bound CI artifacts with source
commit/tree fingerprints and output hashes. The canonical renderer and semantic
coverage checks remain shared; this changes evidence delivery, not backend
capability or execution status. Host-free validation applies to this contract.
No device measurements or schedule parity are inferred.

The optional median bound is Apple-route-only and is not applicable here:
this backend does not consume the Apple route ledger or inherit its timings.

IKF-1 admission guard: the shared D2 cache and persisted-record consumer refuse
L2/L3 intra-kernel timings (`evidence.instr_level`) and malformed levels as
dispatch evidence. L0/L1 and existing pre-instrumentation records retain their
semantics. This is a host-contract check for this backend, not a device-clock
or instrumentation implementation claim.

Sync `APPLE-POLICY-COMPARE-20260904`: not applicable to CUDA route selection.
The fixed-count Apple comparison is an analysis-only benchmark harness; no
NVIDIA selector, timing policy, runtime ABI or device evidence changes.


## Compiler foundation sync `IR-NATIVE-FOUNDATION-1` — 2026-09-04

First migration: remove Graph re-entry and the base-package build from scheduled matmul, derive the ABI from verified scheduled IR, then extend semantic-kernel consumers. Follow-up required on Super-Bear for exact RTX 5070 numerics and resource/performance comparison; this survey adds no CUDA execution proof.

Sequencing and acceptance are owned by
[`INTEGRATED_COMPILER_PLAN.md`](../../compiler/INTEGRATED_COMPILER_PLAN.md#foundation-program).
Shared change in this slice: architectural migration plan only; runtime, ABI,
selector and physical schedules are unchanged. Historical routes have explicit
replacement and deletion gates, not permanent compatibility exemptions.

## Compiler archive handoff — 2026-09-04

The [August review](../../compiler/archive/CODE_REVIEW_2026-08-29.md)
and [historical typing census](../../compiler/archive/W1_1_TYPING_INVENTORY.md)
are archived. Their [reconciliation](../../compiler/COMPILER_AUDIT.md#archive-reconciliation--2026-09-04)
retains unresolved work in live owners; archival does not close this backend's
`P2-REVIEW-SHARED-PASSES-2026-08-29` proof obligations. This is a documentation
and diagnostic-specification link change; runtime parity testing is not
applicable, and no new device evidence is claimed.

Follow-up required: induced allocation-failure cleanup remains untested. The
two tensor-valued `tile.mma` construction sites remain owned by W1.1 / foundation
F2; scheduled artifact packaging (F1) remains the first migration step.

## Foundation F1 — `IR-NATIVE-FOUNDATION-1` — 2026-09-04

NVIDIA scheduled matmul packaging now consumes its scheduled artifact without a
`GraphIRModule` argument or a base-package compilation. Descriptor fields are
checked against the durable Schedule record and Tile entry signature; the driver
records adjacent Graph → Schedule → Tile → Target → PTX ancestry. Runtime ABI,
physical kernels and numerical policy are unchanged.

Owning items: E2E-REAL-3 / NVIDIA-E2E-1. Static and bounded-dynamic f16/bf16,
epilogue and reduced-output paths retain their binding contracts. Host tests
reject stale shape/output/entry/arity metadata before compilation, require one
compiler call, and prove Graph text is no longer a packaging input. Exact-device
validation uses Super-Bear RTX 5070, CUDA 13.3 and its LLVM 23 tools. This removes
the first scheduled Graph re-entry; other Graph-owned package families and
NVIDIA typed materialization remain F2 work. No performance gain is claimed.

Validation: Princess-Luna WSL passed 305 focused scheduled-consumer, NVIDIA
packaging, audit, diagnostic-registry and pass-metadata tests (11 environment
skips); Ruff passed. Super-Bear passed 52 tests from
`tests/unit/test_scheduled_matmul_consumers.py` and
`tests/device/nvidia/test_scheduled_matmul_consumers.py` (10 sibling-backend
skips), including static f16/bf16, bounded-dynamic f16/bf16, fused/reduced output
and macro-CTA tails. This used the host's existing LLVM 23 native tools and PTX
bridge: C++ sources are unchanged. Set `CUDA_HOME=/usr/local/cuda-13.3`, source
`scripts/_nvidia_env.sh`, and point `TESSERA_NVIDIA_OPT` at the host build's
`src/compiler/codegen/tessera_gpu_backend_NVIDIA/tools/tessera-nvidia-opt`.

## Foundation F2 unary slice — `IR-NATIVE-FOUNDATION-1` — 2026-09-05

Owning item: E2E-REAL-5 / foundation F2. The shared native scheduling passes now
admit SM120 f32 softmax and serial rank-reducing sum/mean/max. They emit a raw
LLVM launch wrapper with the established NVIDIA symbol/ABI, and retain the
Schedule hash in Tile IR. NVIDIA packaging consumes that artifact directly.
NVIDIA's existing `approx_exp2` policy is explicit and hashed; other architectures
retain `accurate`. The verifier rejects a policy inconsistent with its architecture.
The wrapper consumer refuses extra function work rather than erasing it.

Implemented for the default static f32 driver path, with independent Schedule
replay and descriptor/ABI checks. Narrow dtypes, min, keepdims and explicit
cooperative reductions remain on their existing routes; direct Graph package
clients also remain. Their migration and constructor deletion are still open.
The comparison exposed and fixed a separate existing defect: canonical
`tessera.reduce` ignored its `kind` in the Graph packager and always selected
sum. It now preserves sum/mean/max/min and refuses unknown kinds.

Validation runs use Super-Bear RTX 5070, CUDA 13.3 and LLVM 23, with a fresh
`tessera-opt` built from this change on Princess-Luna and existing NVIDIA lowering
tools/runtime (unchanged sources). Tests compare native scheduled and retained
Graph packages against NumPy and each other; this is numerical/ABI proof, not
performance promotion. Remaining dtype/axis-policy breadth is not closed.

Validation results: 12/12 exact RTX comparison cases passed (softmax, sum,
mean, max across `(2,3,5)`, `(7,19,257)` and `(2,3,1)`); old/new results were
identical and matched NumPy. Princess-Luna passed 331 focused compiler, audit
and registry tests with 24 environment skips, including the two enabled gfx1151
semantic-kernel tests. Three native FileCheck fixtures passed. Mypy reports zero
errors and Ruff passes. These builds are not assertions-enabled LLVM proof.

## F2 direct unary clients — `IR-NATIVE-FOUNDATION-1` — 2026-09-05

The public NVIDIA unary package APIs now enter the same native Schedule/Tile
boundary as the driver for the migrated f32 envelope. Missing `tessera-opt` is
an explicit failure for that envelope, not a return to Python kernel emission.
Private Graph constructors remain for unmigrated dtype/policy cases and retained
differential baselines; they are not new production routes.

Closed in this cut: direct `package_native`, `package_softmax`,
`package_f32_softmax` and serial f32 sum/mean/max `package_reduction` callers.
Narrow softmax/reduction, min, keepdims and cooperative schedules retain their
previous implementations. Constructor deletion remains blocked on those cases.
The RTX comparison now launches baseline, direct API and driver artifacts for
all 12 shape/operation cases; outputs are identical and agree with NumPy. No
native compiler, runtime ABI, physical schedule or numerical policy changes in
this follow-through; the prior F2 compiler build is reused.

Validation: 83 focused WSL routing, NVIDIA packaging and audit tests passed;
mypy retained zero errors and Ruff passed. The 12 RTX comparison cases include
singleton/tail widths and multiple blocks, with the direct API separately
launched alongside the driver and retained baseline. Constructor retirement is
still gated on the remaining dtype/policy envelopes, not on renaming helpers.

## F2-U1–U10 unary closure — `IR-NATIVE-FOUNDATION-1` — 2026-09-05

NVIDIA unary packaging now consumes native Schedule/Tile for f16/bf16/f32
softmax and sum/mean/max/min reductions with f32 accumulation/output, arbitrary
static axes, keepdims and serial/cooperative_128 policy. Production Graph
constructors were removed; differential baselines live only under test support.
The shared Graph reduce verifier admits narrow-to-f32 and retained dimensions;
reverse AD refuses those new envelopes explicitly. Generic Linalg lowering still
declines them. No new dtype, operation, runtime ABI or physical schedule is added.

Owning outcome: parity validated on Super-Bear RTX 5070 / CUDA 13.3. All 184
cases passed, including 144 reduction combinations and 24 cooperative
257-element contiguous/strided cases; no timing/promotion claim.
Native replay and policy refusal gates pass with the rebuilt LLVM 23 compiler.
Next F2 work: native norm/attention contracts and the remaining packed families.

Validation: 516 focused WSL unit/registry/dtype/routing/audit tests passed, with
24 explicit environment/envelope skips; the final policy/doc follow-up passed
43 tests. Seven native IR fixtures passed, including existing reverse-AD paths.
Ruff and the zero-error mypy ratchet passed; all 30 generated-document drift
gates passed. LLVM 23 `tessera-opt` was rebuilt on Princess-Luna and its matching
Linux executable used for Super-Bear native scheduling. Assertions-enabled LLVM
and new Apple/x86 hardware evidence remain outside this cut.

## F2 norm/attention — `IR-NATIVE-FOUNDATION-1` — 2026-09-05

The shared compiler adds `schedule.norm` for SM120 unweighted f16/bf16/f32 row
normalization and admits SM120 forward `schedule.attention` with explicit
recompute policy. NVIDIA direct package/driver paths consume native Tile wrappers;
old constructors are test-only baselines. Runtime ABI and physical kernels are
unchanged. Norm rejects unsupported policy overrides and epsilon that cannot be
represented as positive finite f32.

Shared finding fixed: forward attention hashes now encode exact f32 policy bits
instead of six-decimal strings. Regenerate old forward Schedule artifacts;
stale serialized hashes fail closed. Backward hash encoding remains follow-up.

Owning scope: native norm and forward attention, including bias, windows and
seeded dropout. Short-query masked attention (`Sq < Sk`) is refused: its local
query mask disagrees with canonical end-aligned ragged masking. F2-A2 must fix
forward and backward together before that envelope is admitted. Saved-LSE,
packed matmul, paged KV, replay-SSM and MoE remain Graph-owned follow-ups.

Validation: rebuilt LLVM 23 `tessera-opt`; 36 Super-Bear RTX 5070 / CUDA 13.3
comparisons passed (18 norm, 12 masked/bias attention, six GQA/dropout). Packages
match the test-only historical baseline and NumPy/canonical streaming oracles.
All-masked rows retain the existing NaNs; no zero-fill or performance promotion
is claimed. 515 focused WSL contract/registry/dtype/routing tests passed with
17 explicit skips, three native IR fixtures passed, and 11 audit tests passed.
All 30 generated-document gates passed. New exact-device Apple/x86 evidence and
assertions-enabled LLVM remain outside this cut.


### F2-A2/A3/P1/S1 package-contract synchronization — 2026-09-05

Owner: E2E-REAL-5; synchronization key `IR-NATIVE-FOUNDATION-1`.
Shared backward Schedule hashes now encode exact f32 policy bits. Regenerate
older backward Schedule artifacts; no runtime pointer ABI changes. Shared
ReplaySSM geometry and spans reject lossy integer inputs and overflowing native
workspace sizes. Native packed/stateful producers remain follow-up required.
NVIDIA parity validated: 18 attention cases, three saved-LSE producer/consumer
pairs and 13 packed/paged/replay cases on Super-Bear RTX 5070 / CUDA 13.3 with a
fresh LLVM 23 target compiler. Both Tile masks are end-aligned; short-query masks
are now admitted. Paired saved-LSE packaging validates physical policy and
bindings before compilation, but both checkpoint constructors remain Python
owned until a native multi-result producer and Schedule consumer exist. Packed
metadata and paged-KV bounds/returns are stricter. No performance promotion.


### Deleted functionality reassessment — 2026-09-05

Synchronization key: `IR-NATIVE-FOUNDATION-1`. The
[central reassessment](../../compiler/INTEGRATED_COMPILER_LOG.md#2026-09-05--deleted-functionality-reassessment)
routes pipeline ownership to W2.4a/CAKE/SO-2, verifier coverage to W2.4,
native residual policy to W5.1, and sharding/halo composition to W5.4 and
COMP-SCHED-OVERLAP-1. StableHLO interoperability is deferred pending a named
consumer. This is planning only: no dialect restoration, capability promotion,
or new hardware evidence. Existing F2 implementation ordering is unchanged.

Own the staged-kernel NVVM/PTX experiment and NVIDIA multi-rank transport evidence; no ROCm timing or physical schedule transfers.


### Scheduled unary replay review fix — 2026-09-05

Owner: E2E-REAL-5; synchronization key `IR-NATIVE-FOUNDATION-1`.
PR #726 follow-up: all NVIDIA scheduled unary packages now require exact native
Schedule-to-Tile replay before target compilation. Softmax and serial/cooperative
reductions previously checked attributes, signature and hash without proving
Tile dataflow. Altered pointer operands and a missing replay compiler now fail
before target compilation; norm retains replay through the same shared path.
No kernel, pointer ABI or runtime schedule changes; this is compiler-validation
coverage, with no new device-performance claim.


### Native checkpoint, packed/state boundaries and ownership audit — 2026-09-05

Owner: E2E-REAL-5 / W2.4 / W2.4a; sync `IR-NATIVE-FOUNDATION-1`.
The integrated plan's five-action loop records native saved-LSE, signed-INT4 and
bounded paged-read package migrations, the recovered prefetch-space check, and
the nonblocking-poll reuse-verifier fix. Shared legality is assessed on every
backend; allocation-scoped release and control-flow lifetime proof remain open.
NVIDIA owns all three package migrations. Nine RTX 5070 device cases passed;
old migrated Python Tile constructors are retired. Preserve the launch symbol
ABI: backward attention dispatch depends on its prefix. No latency improvement
or selector promotion is claimed. Follow-up required: full queue release proof,
queue performance/resource packet, scaled packing, recompute backward, replay-SSM
and MoE native ownership contracts.

### Physical paged-read mnemonic isolation — PR #728 review

Owner E2E-REAL-5 / F2-S1; sync `IR-NATIVE-FOUNDATION-1`.
The bounded native tensor producer is `tessera.paged_kv_read`, distinct from the
public `tessera.kv_cache.read(cache, start, end) -> (K, V)` contract. The public
handle form remains unchanged and its Graph ODS declaration remains open.
This shared-contract correction prevents the NVIDIA physical form from claiming
Apple, ROCm or x86 public cache semantics; no sibling runtime or ABI changes.

### Allocation lifetime analysis and device regression comparison — 2026-09-05

Owner W2.4a / CAKE / SO-2; sync `IR-NATIVE-FOUNDATION-1`.
Shared analysis now retains all pending allocation accesses, scopes direct-token
and keyed completion, joins branch/CFG paths, checks loop backedges and rejects
premature frees and use-after-free. Thread rendezvous cannot retire DMA.
Unknown origins/regions and loop-token generation remain conservatively gated;
see the integrated plan's matching section for the exact admission boundary.
NVIDIA owns CUDA-event and end-to-end native-image regression comparison on Super-Bear; no schedule or selector change.

The comparison in `benchmarks/baselines/allocation_lifetime_nvidia.json` is
withdrawn: the supplied core compiler hashes did not identify the NVIDIA lowerers
actually selected. New exact-host comparison evidence is required.


### Dynamic completion generations and memref arena proof — 2026-09-05

Owner W2.4a / CAKE / SO-2; sync `IR-NATIVE-FOUNDATION-1`.
The integrated plan's matching section supersedes the direct-token-only and
memref-planner limitations above. SSA edge renaming distinguishes dynamic
completion generations; structured/CFG allocation identity is shared by operation
and lifetime verifiers. Memref cast/view lifetimes now feed both reuse assignment
and arena preflight; forged groups fail before physical materialization, and
supported alias descriptors retain the arena address space. No runtime ABI,
selector or architecture schedule changes.

Follow-up required: SM120 lowering and measured slot/phase protocol/resource proof for a producer using the new dynamic completion edges. Existing attention image/oracle regression is recorded separately; it is not queue throughput evidence.

`benchmarks/baselines/token_memref_nvidia.json` is also withdrawn for the native
compiler-selection defect. No before/after claim is retained from this packet;
the independently recorded ring and async GEMM experiments are unaffected.


### Structured-path reuse, private borrowing and device ring experiment — 2026-09-05

Owner W2.4a / CAKE / SO-2; sync `IR-NATIVE-FOUNDATION-1`.
The integrated plan's matching section supersedes the blanket structured-region
and direct-call exclusions above. Coalescing requires all-path completion,
derived uniform branch exclusivity and release before loop backedges. Private
callee bodies establish borrowing; the arena preserves the existing helper ABI.
External/recursive ownership and general CFGs remain conservatively excluded.

RTX 5070 native MLIR ring experiment and stale-generation negative control
passed. CUDA-event direct/depth-2 medians were 0.037504/0.049038 ms. Nsight Compute
and Systems exports are linked by the integrated plan; barrier cost increased
without an occupancy-capacity gain. Follow-up required: production Tile ring
producer, actual asynchronous overlap and representative/saturated-grid tests.
No selector promotion or restored queue dialect is justified by this prototype.

**Real async follow-up (2026-09-05), W2.4a / `IR-NATIVE-FOUNDATION-1`:**
[Native GEMM benchmark](../../../../benchmarks/nvidia/ASYNC_PRODUCER_CONSUMER.md)
now measures useful next-panel `cp.async`/current-panel MMA against an immediate
wait control. RTX 5070 medians improve 6.1/2.9/2.3% at 512/1024/2048 square shapes;
all six oracle shapes pass. Matching resources and Nsight/SASS support the
scheduling interpretation. Follow-up required: generic release-token integration,
independent runs, and sanitizer validation (WDDM debugger initialization blocked).
No selector promotion. Sibling ROCm needs its own physical producer; Apple/x86
schedule parity is not applicable to this CUDA-only experiment.

**PR #729 review correction — IR-NATIVE-FOUNDATION-1:** forwarded TMA
descriptors now contribute all derivable source allocations to lifetime checks;
unknown descriptor origins fail closed. Arena diagnostics register both emitting
passes. NVIDIA's two core-compiler comparison packets are withdrawn; the runner
now explicitly selects the consumed NVIDIA lowerer and restores its environment.
Shared verifier/registry validation applies to this backend; no new device or
physical schedule proof is claimed by this correction.

**Symbolic kernel reuse — W2.4a / IR-NATIVE-FOUNDATION-1:** registered GPU
kernel scalar launch arguments and induction values can prove uniform control
for memref reuse; real GPU barriers release synchronous accesses. Static arenas
materialize in GPU modules; the dynamic extension is assessed below.
Static shared globals remain in the GPU module for NVVM lowering. Follow-up required: native producer/package integration and exact-device proof on Super-Bear; no new CUDA timing claim.


**Dynamic GPU storage — W2.4a / IR-NATIVE-FOUNDATION-1:** entry-block GPU
arenas now use native dynamic shared memory plus a checked native host sizing
companion; local gpu.launch_func byte counts are wired to that companion.
Exact-device experiment validated on Super-Bear RTX 5070: native LLVM host sizing supplies CUDA launch bytes, with exact cross-lane numerical checks for four runtime widths. Follow-up required: production package binding, nested dynamic lifetimes and asynchronous producer integration. No selector promotion or general speedup claim.
See the [shared experiment](../../../../benchmarks/DYNAMIC_GPU_STORAGE.md).


**Native package / nested lifetime / async producer — W2.4a / IR-NATIVE-FOUNDATION-1:**
Parity validated for the bounded raw native package on RTX 5070: four nested and four native async-copy cases pass after serialization/reload. Missing/partial/wrong-group waits and oversized sizing are rejected. Follow-up required: macro-GEMM schedule integration, token forwarding, tensor/JIT binding and sanitizer proof; no overlap or performance claim.
See the [package proof](../../../../benchmarks/NATIVE_GPU_STORAGE_PACKAGE.md) and integrated plan for the bounded ABI and remaining work.


**Tensor/JIT and producer integration — W2.4a / IR-NATIVE-FOUNDATION-1:**
Parity validated on RTX 5070: four exact explicit tensor/JIT cases and six macro-GEMM shapes in both deferred/immediate-wait forms. Production macro-GEMM now emits NVGPU copy/group/wait tokens before conversion. Generic arena proofs accept identity forwarding, not changing-generation macro ownership. Follow-up required: automatic descriptors/arbiter and paired AD; exploratory timings do not promote a route.
See the [integration report](../../../../benchmarks/NATIVE_TENSOR_PRODUCERS.md).


**ABI manifests / paired programs / streams — W2.4a / IR-NATIVE-FOUNDATION-1:**
Parity validated on RTX 5070 for four generated-descriptor/JIT/arbiter/two-stream cases, including loop-external token replacement. Event completion retains allocation owners and orders conflicts. Follow-up required for rotating generations, Schedule-authored manifests/oracles, paired-AD producer/device proof and concurrency measurements; no new selector promotion.
See the [integration matrix and measurement](../../../../benchmarks/NATIVE_STORAGE_INTEGRATION.md).


**Generated children / rotating ownership / Apple materialization — W2.4a / IR-NATIVE-FOUNDATION-1:**
Parity validated on RTX 5070: four compiler-generated AD primal/tangent cases and five fixed-slot rotating-generation cases, including zero trips and varying generation input. Follow-up required for dynamically selected/permuted slots, broader AD families and sanitizer validation. Apple MSL and ROCm wait timings are architecture-specific and do not change NVIDIA selection.
See the [shared follow-up and evidence](../../../../benchmarks/NATIVE_STORAGE_FOLLOWUP.md).


**Expanded AD / dynamic aliases / resident Apple binding — W2.4a / IR-NATIVE-FOUNDATION-1 (2026-09-06):**
Parity validated on RTX 5070: sixteen expanded forward-AD cases, including subtraction, stop-gradient and reversed wrt order; seven dynamic two-slot alias cases include zero/odd/even trips. Each iteration fully drains and collectively releases before swapping; pending token generations across swaps and N-slot rings require follow-up. Apple raw package binding is a sibling-only ABI; ROCm counter availability does not constrain NVIDIA Nsight profiling.
See [expanded implementation and device evidence](../../../../benchmarks/NATIVE_STORAGE_FOLLOWUP.md).


**Nonlinear AD / pending swaps / typed Apple JIT — W2.4a / IR-NATIVE-FOUNDATION-1 (2026-09-06):**
Parity validated on RTX 5070 for nine sigmoid/tanh/composed AD cases, including stable tails, small inputs, signed zero and nonfinite inputs. Seven two-slot pending-token cases validate generation/slot coupling, final drain and zero/odd/even trips. Prefetch can precede consumption; this proves correctness, not a speedup. General N-slot/nested recurrences and broader/reverse AD families require follow-up. Apple typed JIT is a sibling ABI, not a CUDA buffer adapter.
See [implementation and owning-host evidence](../../../../benchmarks/NATIVE_STORAGE_FOLLOWUP.md).


**N-slot / reduction and reverse pairs / Metal queues — W2.4a / IR-NATIVE-FOUNDATION-1 (2026-09-06):**
Parity validated on RTX5070: 15 compiler-produced reduction/nonlinear JVP and elementwise VJP cases; 3/4/8 released-slot rings and uniformly nested pending two-slot recurrence, seven zero/odd/even cases each. Follow-up required for N-slot pending generation maps, attention/reduction VJP, general reverse tapes, event ABI parity and performance comparisons. Metal shared events are not CUDA execution proof.
See [shared scope and device evidence](../../../../benchmarks/NATIVE_STORAGE_FOLLOWUP.md) and the integrated plan’s next ordered contracts.


**Pending N-slot / reduction VJP / attention products / queue intervals — W2.4a / IR-NATIVE-FOUNDATION-1 (2026-09-06):**
Parity validated on RTX5070: 21 AD pair cases including reduction VJP and seven exact cases each for uniformly nested 3/4/8-slot single-pending rings. Follow-up required for multiple outstanding generations, compound attention AD device products, persisted LSE/tapes and cross-stream overlap attribution. Metal GPU timestamps are not CUDA performance evidence.
See [shared implementation and evidence](../../../../benchmarks/NATIVE_STORAGE_FOLLOWUP.md).


**Outstanding cohorts / saved LSE / queue attribution — W2.4a / IR-NATIVE-FOUNDATION-1 (2026-09-06):**
Parity validated on RTX5070 for 21 nested multi-cohort cases (2/3/4 pending copies). Thirty matched queue pairs pass exact oracles; Nsight Systems records cross-stream kernel intersections and Nsight Compute supplies an isolated workload counter sample. Follow-up required for production submission attribution, fresh-process performance evidence, Q/K JVP and device-owned attention tapes.
See the [shared evidence and next boundaries](../../../../benchmarks/NATIVE_STORAGE_FOLLOWUP.md).


**Persistent snapshots / attention export / MSW-9 — IR-NATIVE-FOUNDATION-1 (2026-09-06):**
Twelve persistent snapshot/repeated-backward/nested-frame cases and six compiler-generated attention checkpoint package cases pass on RTX5070. Native Schedule/Tile consumes the AD export directly. Attention uses the existing host-buffer bridge; resident LSE ownership and Q/K JVP still require follow-up. MSW-9 candidate inventories do not promote a native route.
See [implementation and evidence](../../../../benchmarks/NATIVE_STORAGE_FOLLOWUP.md).


**Resident LSE / native ANN constants — W2.4a / MSW-9 / IR-NATIVE-FOUNDATION-1 (2026-09-06):**
Parity validated on RTX 5070 for six resident Q/K/V + saved-LSE generations, caller-buffer mutation, repeated backward, and close invalidation. No host tensor bridge participates in backward. Native ANN constant composition has compiler tests; exact-device ANN equivalence/performance and promotion require follow-up. Native Q/K JVP, general nested residual tapes and event-owned concurrent generations remain open.
See the integrated plan’s native ANN composition and resident attention generation section.


**Typed nested exports / resident score tangents / promotion admission — W2.4a / MSW-9 / IR-NATIVE-FOUNDATION-1 (2026-09-06):**
Parity validated on RTX 5070 for eight resident forward/backward cases and 32 Q-only, K-only, Q+K and Q+K+V JVP directions, including 129 keys. Native MLIR uses bounded shared storage and the private O/LSE generation. Backward grid sizing now covers concatenated gradient ranges. Automatic TangentInterface integration, general device tapes, concurrency and ANN performance promotion remain follow-ups.
See the integrated plan’s typed nested exports and resident score tangents section.


**Automatic score JVP / control-flow inventory — W4 / W2.4a / IR-NATIVE-FOUNDATION-1 (2026-09-06):**
Parity validated for the isolated native automatic Q/K/V attention export bound to resident CUDA O/LSE: eight cases and 32 directions on RTX 5070. This is correctness, not performance or overlap evidence. General JIT composition and persistent nested tensor tape execution remain follow-ups.


**Split tensor tapes / JIT-owned Q/K — W4 / W2.4a / IR-NATIVE-FOUNDATION-1 (2026-09-06):**
The bounded split f32 tape consumer and JIT-owned isolated Q/K program are implemented for owning-device validation on RTX 5070. NVVM retains generic alloca lowering; AMDGPU private descriptor addressing is not transferred. Follow-up required for dynamic/mixed-type tapes, parallel scheduling, asynchronous retirement and composed attention AD. No latency/overlap or arbiter promotion claim.
See [loop11 implementation and device evidence](../../../../benchmarks/NATIVE_STORAGE_FOLLOWUP.md).


**PR #732 scratch retirement — W2.4a / IR-NATIVE-FOUNDATION-1 (2026-09-06):**
Shared CUDA/HIP recomputed-tape backward now releases its temporary primal after synchronous completion; persistent captured primals and returned derivative generations remain owned. Repeated-call and launch-failure allocation tests cover host ownership; no new exact-device or performance claim.


**F0 census correction — IR-NATIVE-FOUNDATION-1 (2026-09-06):**
Shared inventory now distinguishes Graph, typed artifact and unknown/raw inputs; missing modules fail closed. Apple GPU packaging is included, with computed Apple family returns explicitly unresolved. No support or device-proof state changes. Follow-up required for target/envelope-aware producer-to-consumer lineage before retiring remaining Graph constructors.


**Apple domains / unary parent replay — F0 / IR-NATIVE-FOUNDATION-1 (2026-09-06):**
Shared census now requires a scheduled consumer in the owning target module before reporting its declared family route. Apple replay/target checks do not establish nvidia execution or performance. No runtime or support state changes; full target/envelope call-path joins remain follow-up required.


**Descriptor projection / ancestry — F0 / IR-NATIVE-FOUNDATION-1 (2026-09-06):**
Not applicable to this implementation: the changed package consumers are Apple-specific. Follow-up required for independent target descriptor/ancestry review; Apple native replay transfers no execution or performance evidence.


**Attention projection / static softmax — F0/F2 / IR-NATIVE-FOUNDATION-1 (2026-09-07):**
Not applicable to this implementation: Apple library package consumers changed; no nvidia runtime or promotion state changes. Follow-up required for independent target descriptor projection and owning-device evidence.


**Low-precision native slice — F0/F2 / IR-NATIVE-FOUNDATION-1 (2026-09-07):**
Shared Graph-to-Schedule admission changes are Apple-conditional. No nvidia ABI or scheduling changes; Apple device proof is not applicable to this target. Independent exact-device evidence remains required for promotion.


**Broader coverage / route promotion — APPLE-ATTN-BWD-1 / IR-NATIVE-FOUNDATION-1 (2026-09-07):**
Not applicable to this target: Apple-only package guard, route-policy default and M1 Max evidence. No nvidia schedule or performance promotion; independent exact-device promotion remains required.


**Mixed bias / math ownership — F2 / APPLE-ATTN-BWD-1 / IR-NATIVE-FOUNDATION-1 (2026-09-07):**
Apple-specific mixed-bias ABI has no nvidia ABI or execution change. Shared audit sequencing and truthful instability wording updated; no sibling performance evidence transferred. Follow-up required: existing native AD/numeric-policy/math consumer work under the integrated plan.


**PR #733 status/paired-companion correction — F2 / APPLE-ATTN-BWD-1 / IR-NATIVE-FOUNDATION-1 (2026-09-07):**
Shared backward Graph emission carries reciprocal primal/VJP symbol references; the new admission restriction and status ABI are Apple-only. No nvidia ABI, physical schedule or performance claim changes. Existing native compiler consumers are checked where available; exact-device nvidia evidence is not inferred from Metal tests.


**Capability-document reconciliation — IR-NATIVE-FOUNDATION-1 (2026-09-07):**
Documentation-only ownership/lifecycle cleanup: Block AttnRes, EGGROLL and game
theory remain scoped landing plans; AD residuals use the active AD plan and
shared substrate demands use existing F0–F4/NUMPOL/layout/transport owners.
NVIDIA workload consumers still need independent native package and exact-SM evidence; no physical schedule is copied from gfx1151.
No runtime, ABI or support status changes. Follow-up requirements are mapped in
[the integrated reconciliation](../../compiler/INTEGRATED_COMPILER_LOG.md#2026-09-07--capability-plan-reconciliation).


**Native status / artifact identity / recipe instances — F0/F2/F3 / IR-NATIVE-FOUNDATION-1 (2026-09-07):**
Shared recipe specialization and descriptor identity changes apply. CUDA schedules/runtime ABIs are unchanged; Metal execution and package timings do not establish SM120 parity. RTX 5070 is reachable. Native ANN execution and heterogeneous/while tape work require owning-host follow-up.
All three fleet LLVM builds currently report assertions OFF. No assertions-enabled
validation or broader route/envelope closure is claimed. See the integrated
plan’s status, native GELU and recipe instantiation section.


**Dynamic native package projection — F2 / IR-NATIVE-FOUNDATION-1 (2026-09-07):**
Floating scheduled matmul packages now verify native Schedule-to-Tile replay and project dynamic axes, epilogues and entry identity before compilation. Partly dynamic descriptors preserve equality guards on static axes. Fourteen RTX 5070 scheduled-matmul device tests pass. No performance promotion follows from these correctness tests.
ANN executable admission, mixed/while tapes and analytic error-budget consumers
remain follow-ups under F3, AD-RESIDUAL-EVAL-1 and FA-1 respectively.


**ANN admission / mixed checkpoint products — F3 / FA-1 / AD-RESIDUAL-EVAL-1 / IR-NATIVE-FOUNDATION-1 (2026-09-07):**
Five RTX 5070 cases execute mixed f32/f64, nested SAVE/HYBRID/recompute-all and proven counted-while SAVE products with repeated backward calls. NVVM private allocation handling remains architecture-owned. This is correctness/retained-byte evidence, not latency or overlap promotion. GPU ANN admission remains follow-up required.
General data-dependent while, integer/predicate and dynamic slots, asynchronous
retirement and automatic policy selection remain open. See the integrated plan
and benchmarks/baselines/tape_checkpoint_20260907/.


**Data-dependent tape / nonlinear native ANN — F3 / FA-1 / AD-RESIDUAL-EVAL-1 / IR-NATIVE-FOUNDATION-1 (2026-09-07):**
Owning RTX 5070 validation covers bounded data-dependent exits, stored predicate branches and asynchronous derivative generations. Native frozen-affine plus terminal ReLU original/rewrite packages have independent numerical and host-bridge timing evidence. Production GPU arbiter registration and tuned schedules remain follow-ups.
Shape-varying residuals, reader-complete asynchronous reclamation, broader
nonlinear/error-budget consumers and measured production promotion remain open.
See the integrated plan and benchmarks/baselines/native_tape_ann_20260907/.

The nine-run native ANN package comparison refuses performance promotion; the
original remains the incumbent. See the packet README for target-local bounds.


**Shape tapes / scoped GPU ANN — F3 / FA-1 / AD-RESIDUAL-EVAL-1 / IR-NATIVE-FOUNDATION-1 (2026-09-07):**

Parity validated for the bounded GPU ANN slice on Super-Bear RTX 5070 (sm_120): four native arbiter cases pass. Nine fused-package runs refuse promotion (median 0.99809x, lower 0.97773x against 1.02x). Shape-varying x86 execution does not transfer to CUDA; dynamic GPU residual storage, reader-complete asynchronous frees and tuned parallel ANN remain follow-ups.

Shared source, physical artifact replay, numerical-domain guards and evidence are
recorded in [the integrated plan](../../compiler/INTEGRATED_COMPILER_LOG.md#2026-09-07--shape-varying-host-tapes-and-gpu-ann-arbitration). All fleet LLVM builds still lack assertions.

The shared host JIT now retires compiler-owned temporaries through upstream
ownership-based buffer deallocation after DPS copies. Validation is x86 on
Princess-Luna; this does not replace the GPU exported-reader completion barrier
or establish asynchronous device freeing. Native extent guards currently abort
on mismatch; recoverable JIT status propagation remains a follow-up.

**PR #734 lifecycle review — F3 / IR-NATIVE-FOUNDATION-1:** CPU ANN
registration now returns a scoped owner. Explicit/context close unregisters
only its own candidates, disables retained candidate handles and releases
probe references; failed registration rolls back. This is host registry ownership; no nvidia runtime ABI or device promotion changes.
CUDA/HIP registration continues to use its separate device binding owner.


### 2026-09-07 — W2.4a / F3 / FA-1: while and row ANN integration

Sync key: **AD-RESIDUAL-EVAL-1 / IR-NATIVE-FOUNDATION-1**.
SM120 owning-device validation passes six row-parallel ANN admission cases and twelve asynchronous derivative generations across four strided while exits and two predicate paths. This is correctness evidence; no tensor-core or overlap claim.

Shared changes: proved constant-stride while capacity, terminal row-sum error
propagation and rank-changing ABI projection, and opt-in row independence with
serialized launch/pipeline replay. See the integrated compiler plan's bounded
while recovery increment and `benchmarks/baselines/row_ann_while_20260907/`.
Follow-up required: general CFG recovery, GPU shape-capacity materialization,
reader-complete stream-ordered reclamation, and measured schedule selection.
The raw-view context barrier is retained; producer-event completion is not a
proof that external readers have finished. Assertions-enabled MLIR remains open.


### 2026-09-07 — W2.4a / AD-RESIDUAL-EVAL-1: dynamic capacity and reader ownership

Sync key: **IR-NATIVE-FOUNDATION-1**. Parity validated on RTX 5070 / SM120: dynamic internal widths 1/2/3 preserve logical copy extents; four tape exits retire pool-allocated derivatives after two reader streams, with a third retirement stream and no context barrier in the tracked region.

Shared contracts: SSA-proved temporary capacity separate from logical shape,
matching dynamic-copy dimension SSA, typed bounded `cf.switch` edges, and scoped
reader completion before stream-ordered frees. Partial-free/event failures retain
ownership; legacy unrestricted exports retain the context barrier. See the
integrated compiler plan's dynamic temporary capacity increment and
`benchmarks/baselines/dynamic_readers_20260907/`.
Follow-up required: exported dynamic tape descriptors/status, loaded residual
shape validation, broader source CFG recovery and adoption by composed consumers.
Assertions-enabled MLIR and measured overlap/throughput remain open.

Free API failures quarantine the frame for device teardown and forbid normal
close/retry; event-record failures retain an explicit completion-wait recovery.

Native CFG cross-block SSA definitions now receive distinct state slots, with
host forward/reverse proof. GPU multiway products still refuse the retained
bound-exhaustion assertion pending a device status/termination consumer.


### 2026-09-07 — checked products and composed readers

Owner: **W2.4a / AD-RESIDUAL-EVAL-1**; sync key **IR-NATIVE-FOUNDATION-1**.
Parity validated on SM120 for checked switch products, exhaustion refusal and composed scoped readers. Returned logical shapes, nested guards and asynchronous checked-result exposure remain follow-ups. No schedule or performance promotion.

See the integrated plan for the exported shape-varying ABI design: returned
logical extents, validated loaded bounds, capacity ownership and status before
view exposure. Arbitrary Python CFG/effects, assertions-enabled MLIR and measured
reclamation/overlap remain open.

Exact-device packet: `benchmarks/baselines/product_status_20260907/nvidia.json`. Six correctness cases; checked-status and asynchronous composition are separate routes.


### 2026-09-08 — F0 / F2 / F4: native unary ancestry and checked tickets

Owner: **E2E-REAL-6F / E2E-REAL-6 / AD-RESIDUAL-EVAL-1 / W2.4a**; sync key **IR-NATIVE-FOUNDATION-1**.

Checked host tickets pass independent derivative generations and injected failure refusal on RTX 5070/SM120. Tracked checked readers, dynamic returned shapes and measured overlap remain open.

Current sequencing: [live compiler plan](../../compiler/INTEGRATED_COMPILER_PLAN.md).
LLVM assertions remain OFF on the two probed Linux fleet builds. Correctness
and injected-status evidence are separate from performance promotion.


### 2026-09-08 — Native result and measured-admission increment

Sync key: `DEEP-NATIVE-2026-09-08`; owners: AD-RESIDUAL-EVAL-1, W2.4a, E2E-REAL-6, MSW-9 and COMPILER-DEVEX-1 in the [live compiler plan](../../compiler/INTEGRATED_COMPILER_PLAN.md).

SM120 exact-device validation covers logical lengths 0/3/8, nested failure refusal and checked device-reader chaining. Release remains context-synchronous; no overlap claim. Scoped ANN evidence remains package timing, not kernel or global promotion.

Shared contracts: compiler-projected capacity/shape ABI, nested status propagation, incoming product status and exact-artifact scoped measurement admission. Evidence: `benchmarks/baselines/deep_native_20260908/`, `benchmarks/baselines/deep_ann_20260908/` and focused native contract tests. General dynamic tensor/AD production, broader reader adoption and tuned ANN families remain open.

COMPILER-DEVEX-1: Super-Bear now hosts an isolated assertions-ON LLVM/Tessera
compiler lane (257 focused checks). It found and fixed a missing Tile dialect
dependency in NativeTapeToGPUPass. Other hosts retain release builds; compiler
validation is shared contract evidence, not sibling device execution proof.


### 2026-09-08 — Automatic AD, checked pools and independent ANN tuning

Owners: AD-RESIDUAL-EVAL-1 / W2.4a / E2E-REAL-6 / MSW-9; sync key `AUTO-AD-POOL-ANN-2026-09-08`. See the [live compiler plan](../../compiler/INTEGRATED_COMPILER_PLAN.md).

SM120 validates generated dynamic forward lengths, capacity refusal and checked generation retirement. Scoped 16x8 ANN evidence selects the independently fused row-parallel rewrite at 1.0753x median package speedup; this does not publish a global route.

Shared changes: native result-capacity projection, monotonic completion proof during event cleanup, scoped checked readers and independent physical ANN schedules. Whole-frame asynchronous ownership, dynamic GPU backward inputs and wider tuned families remain open. Evidence: `benchmarks/baselines/automatic_ad_retirement_20260908/` and `benchmarks/baselines/tuned_ann_20260908/`.


### 2026-09-08 — Runtime shapes and scoped frame retirement

Owners: AD-RESIDUAL-EVAL-1 / W2.4a / E2E-REAL-6 / MSW-9 / W4-PRODUCT-1;
sync key `RUNTIME-SHAPES-FRAMES-2026-09-08`. Sequencing:
[live integrated plan](../../compiler/INTEGRATED_COMPILER_PLAN.md).

SM120 device proof covers runtime-shaped backward products, multiple matrix results and eight scoped frame retirements. Broader ANN measurements remain workload-specific; no global promotion.

Shared contracts: checked per-axis/volume bounds, contiguous dynamic input
views, independent output sidecars, post-completion checked exposure and scoped
frame reader retirement. Capture and exceptional cleanup remain synchronous;
module-unload latency, arbitrary Python CFG, general saved products and wider
nonlinear consumers remain open. Evidence:
`benchmarks/baselines/runtime_shape_frames_20260908/` and
`benchmarks/baselines/broad_ann_20260908/`.

### Source CFG and asynchronous ownership (2026-09-08)

Owners: `W4-PRODUCT-1`, `AD-RESIDUAL-EVAL-1`, `W2.4a`, `E2E-REAL-6`, `MSW-9`;
sync key `SOURCE-ASYNC-FOUNDATION-2026-09-08`. Sequencing and remaining gates:
[live integrated plan](../../compiler/INTEGRATED_COMPILER_PLAN.md).

Parity validated on RTX 5070/SM120 for scoped dynamic public frames, cross-stream checked readers, asynchronous capture/backward and whole-frame retirement. The recorder forbids explicit context waits in the successful asynchronous path. Joint-volume matrices include a 64-element capacity rather than the independent 64x64 upper envelope. Nine independent terminal-square ANN runs did not clear the promotion margin; the incumbent remains selected.

Shared contracts: explicit source-comparison predicates, tracer-owned SSA,
exact dominating product guards, immutable checked shape/status exposure,
reader-aware event ordering and bounded off-thread unload admission. Arbitrary
CFG and exceptional cleanup remain open; worker admission does not guarantee
driver latency. Evidence: `benchmarks/baselines/source_async_foundation_20260908/`.

### Effect-aware CFG and status composition (2026-09-08)

Owners: `W4-PRODUCT-1` / `W2.4a`; sync key
`CFG-STATUS-COMPOSITION-2026-09-08`. Sequencing:
[live plan](../../compiler/INTEGRATED_COMPILER_PLAN.md).

Parity validated on SM120: two native status arguments gate capture plus upstream failure, with four-case truth-table execution and reader-aware retirement. No overlap or performance promotion is claimed.

Shared contracts: typed multi-result conditional SSA, registered assertion effects,
explicit bounded loop expansion and serialized incoming-status count. Unknown
Python effects refuse before capture. Loop returns, external aliases, exceptions,
wider status joins and heterogeneous dynamic capture remain open. Evidence:
`benchmarks/baselines/cfg_status_composition_20260908/`.

### Completion state and status fan-in (2026-09-08)

Owners: `W4-PRODUCT-1` / `W2.4a`; sync key
`COMPLETION-STATE-FANIN-2026-09-08`. Sequencing:
[live plan](../../compiler/INTEGRATED_COMPILER_PLAN.md).

Parity validated on RTX 5070 / SM120: four incoming statuses passed all sixteen truth-table combinations with scoped upstream readers and successful frame retirement. Eight-input device proof, GPU external mutation and overlap/performance promotion remain follow-ups.

Shared contracts: typed completion payloads, explicit serialized source state,
bounded incoming-status count and reader leases for every prerequisite.
Arbitrary object/view mutation, uncaught exception transport, heterogeneous
dynamic capture and unbounded effect joins remain open. Evidence:
`benchmarks/baselines/completion_state_fanin_20260908/`.

### Source JIT and device state (2026-09-08)

Owners: `W4-PRODUCT-1` / `W2.4a`; sync key `SOURCE-JIT-STATE-2026-09-08`.
Sequencing: [live plan](../../compiler/INTEGRATED_COMPILER_PLAN.md).

Follow-up required: Super-Bear SSH timed out on repeated probes. Eight-status and source-state device proofs were not run; prior four-status proof remains historical. Source synchronization also remains pending.

Shared contracts: serialized state/error results, capacity/shape projection,
exact alias checks and bounded JIT module ownership. Arbitrary objects, partial
overlap, dynamic exception objects/messages and in-place GPU mutation remain open.
Evidence: `benchmarks/baselines/source_jit_state_20260908/`.

### Declared object state and owned GPU mutation (2026-09-09)

Owners: `W4-PRODUCT-1` / `W2.4a` / `AD-RESIDUAL-EVAL-1`.
Sync key: `SOURCE-OBJECT-OWNERSHIP-2026-09-09`.
Sequencing: [live plan](../../compiler/INTEGRATED_COMPILER_PLAN.md).

Parity validated on RTX 5070 SM120: three owned in-place state updates preserve allocation identity, and active/expired readers refuse. All 256 eight-status fan-in combinations also passed on this host. No asynchronous-write or performance claim.

Shared contracts: field projection, read-only overlapping snapshots, static
exception args and scoped exclusive GPU writes. Custom accessors, overlapping
writes, general exception semantics and automatic effectful AD remain open.
Evidence: `benchmarks/baselines/source_object_ownership_20260909/`.

### Mixed aliases and asynchronous state (2026-09-09)

Owners: `W4-PRODUCT-1` / `W2.4a` / `AD-RESIDUAL-EVAL-1`.
Sync key: `MIXED-ALIAS-ASYNC-STATE-2026-09-09`.
Sequencing: [live plan](../../compiler/INTEGRATED_COMPILER_PLAN.md).
Parity validated: SM120 asynchronous computation/copyback and pending-reader exclusion.
Shared contracts: per-pair alias admission, plain instance projection, explicit
next-state VJP and event-ordered checked copyback. Custom hooks, writable overlap,
dynamic exception transport, effectful replay and multi-writer mutation remain
open. Failure teardown can synchronize; no performance promotion.
Evidence: `benchmarks/baselines/mixed_alias_async_state_20260909/`.

### Writable source views and exception values (2026-09-09)

Owners: `W4-PRODUCT-1` / `AD-RESIDUAL-EVAL-1` / `W2.4a`.
Sync key: `SOURCE-VIEW-EXCEPTION-2026-09-09`.
Sequencing: [live plan](../../compiler/INTEGRATED_COMPILER_PLAN.md).
Existing single-state asynchronous consumer regression validated on SM120. Writable multi-input views and exception execution require new device bindings.
Shared contracts: containing-input view maps, typed completion payload outputs,
field-projected VJP and invocation/cache ancestry checks. General write maps,
exception identity/chains, exception AD and GPU view/exception lowering remain
open. No performance promotion.
Evidence: `benchmarks/baselines/source_views_exception_20260909/`.

### Strided source roots (2026-09-09)

Owners: `W4-PRODUCT-1` / `AD-RESIDUAL-EVAL-1` / `W2.4a`.
Sync key: `STRIDED-SOURCE-GPU-2026-09-09`.
Sequencing: [live plan](../../compiler/INTEGRATED_COMPILER_PLAN.md).
SM120 strided local state executes through copied private roots and asynchronous checked copyback; input-rooted view writes still reject.
Shared contracts: stride-aware view maps, exact-alias adjoint roots, static
exception cause metadata, copy-before-write and bounded native write ancestry.
Negative/multidimensional views, slice adjoints, exception identity/context and
GPU exception transport remain open. No performance promotion.
Evidence: `benchmarks/baselines/strided_source_gpu_20260909/`.

### Mapped views and GPU exception completion (2026-09-09)

Owners: `W4-PRODUCT-1` / `AD-RESIDUAL-EVAL-1` / `W2.4a`.
Sync key: `SOURCE-MAPPED-EXCEPTION-2026-09-09`.
Sequencing: [live plan](../../compiler/INTEGRATED_COMPILER_PLAN.md).
Parity validated on SM120: mapped forward/backward and synchronous/asynchronous static/dynamic exception completion, without result exposure on failure.
Shared contracts: bounded root-coordinate maps, native slice adjoints, static
exception graph identity and checked completion before public result exposure.
Mapped expansion is bounded to 256 elements; runtime-sized/general maps,
dynamic context slots, full tracebacks and multi-state device bindings remain
open. No performance promotion; private copies and teardown synchronization
remain possible. [Block AttnRes](../../compiler/BLOCK_ATTNRES_ROCM_PLAN.md)
uses these as foundation oracles, not a workload kernel/promotion claim.
Evidence: `benchmarks/baselines/source_exception_gpu_20260909/`.

### Runtime maps and context payload slots (2026-09-09)

Owners: `W4-PRODUCT-1` / `AD-RESIDUAL-EVAL-1` / `W2.4a` / `BLOCK-ATTNRES-1`.
Sync key: `RUNTIME-SOURCE-MAPS-2026-09-09`.
Sequencing: [live plan](../../compiler/INTEGRATED_COMPILER_PLAN.md).
SM120 validated twenty runtime-map/context cases, including shape-mismatched backward refusal. No CUDA Block AttnRes kernel or promotion is established.
Shared changes: compact rectangular maps, dynamic slice seed guards, exact
logical-shape equality proofs, bounded unsigned division and per-site exception
payload outputs. General runtime Python slicing, large negative maps, loop
exception generations, CPython frame semantics and GPU exception AD remain
open. Full tracebacks are not equivalent to source notes. No performance
promotion; see [Block AttnRes](../../compiler/BLOCK_ATTNRES_ROCM_PLAN.md).
Evidence: `benchmarks/baselines/runtime_source_maps_20260909/`.


### Source generations and checked VJP (2026-09-09)

Owners: `W4-PRODUCT-1` / `AD-RESIDUAL-EVAL-1` / `W2.4a`.
Sync key: `SOURCE-GENERATION-AD-2026-09-09`.
Sequencing: [live plan](../../compiler/INTEGRATED_COMPILER_PLAN.md).
SM120 owning-host cases validate runtime indices, generation contexts and synchronous checked VJP. Asynchronous VJP staging and general source shapes remain follow-ups.
Shared contracts: int64 source slice bounds, bounded per-generation exception
slots, scalar integer residuals, forward-gated VJP over snapshots and bounded
host traceback retention. Negative runtime strides, arbitrary loop exception
objects, full CPython frames and asynchronous exception AD remain open.
Block AttnRes cooperative kernel/promotion work is unchanged; no performance
promotion follows from these correctness packets.
Evidence: `benchmarks/baselines/source_generation_ad_20260909/`.


### Signed nested source maps and asynchronous VJP (2026-09-09)

Owners: `W4-PRODUCT-1` / `AD-RESIDUAL-EVAL-1` / `W2.4a`.
Sync key: `SOURCE-NESTED-ASYNC-2026-09-09`.
Sequencing: [live plan](../../compiler/INTEGRATED_COMPILER_PLAN.md).
SM120 independently passed 19 signed/nested map, checked VJP and retained-exception cases. No cross-architecture or performance promotion.
Shared contracts: signed int64 view composition, compiler-projected CPU result
capacities, bounded loop-carried exception identities and forward-gated async
VJP over retained snapshots. Runtime gather adjoints, arbitrary exception heaps,
full CPython frames, runtime-shaped source roots and fully asynchronous teardown
remain open. Block AttnRes kernel/promotion obligations are unchanged.
Evidence: `benchmarks/baselines/source_nested_async_20260909/`.


### Gather transpose and scoped source retirement (2026-09-09)

Owners: `W4-PRODUCT-1` / `AD-RESIDUAL-EVAL-1`.
Sync: `SOURCE-GATHER-SCOPED-2026-09-09`.
SM120 scoped source VJP validated independently (19 correctness cases); native gather adjoints currently have CPU proof, not a CUDA promotion.
Shared contract: index-only accumulating gather adjoints, explicit CPU exception
class ownership, scoped source VJP readers and ordered asynchronous retirement.
Arbitrary exception heaps, full CPython frames and fully asynchronous unload or
failure recovery remain open. Evidence: `benchmarks/baselines/source_scoped_ad_20260909/`.
Sequencing: [live plan](../../compiler/INTEGRATED_COMPILER_PLAN.md).


### Exception bindings and unload recovery (2026-09-09)

Owners: `W4-PRODUCT-1` / `AD-RESIDUAL-EVAL-1`.
Sync: `SOURCE-EXCEPTION-BINDINGS-2026-09-09`.
SM120 validates custom host-class reconstruction following GPU failure and scoped retirement; no performance promotion.
Shared contracts: raise occurrence identity, explicit host class bindings, cached
completion exceptions, and retry only after confirmed unload/context exit.
Arbitrary exception heaps, handled custom-constructor effects, CPython frames
and uncertain driver recovery remain open. No performance promotion.
Evidence: `benchmarks/baselines/source_exception_bindings_20260909/`.
Sequencing: [live plan](../../compiler/INTEGRATED_COMPILER_PLAN.md).

### Architecture sweep and failure boundaries (2026-09-09)

Owners: `FRONTEND-IR-MEDIUM-1`, `DIST-NATIVE-1`, `W4-PRODUCT-1`, `W2.4a`.
Sync: `ARCH-SWEEP-FAILURE-2026-09-09`.
Follow-up required: WSL fault injection is not SM120 driver-failure recovery proof.
Shared changes: structural dimension equality, contiguous pipeline-stage
validation, constructor-free exception graph preflight and preservation of both
unload/context-exit failures. Unknown driver outcomes retain owners and refuse
retry. Arbitrary exception heaps and native CPython frame reconstruction remain
open; no device reset/recovery or performance claim.
Sequencing: [live plan](../../compiler/INTEGRATED_COMPILER_PLAN.md).

### Indexed exception completion and one-shot unload (2026-09-09)

Owners: `W4-PRODUCT-1`, `W2.4a`. Sync: `SOURCE-HEAP-RETIRE-2026-09-09`.
SM120 completion-carrier correctness is measured independently; driver failure recovery remains host fault injection only.
Shared contract: indexed source exception objects, pre-constructor graph/payload
snapshot, explicit bindings, and retention of synchronous uncertain driver
outcomes. Arbitrary native heap allocation, full CPython frames and confirmed
isolation teardown/replacement remain open. No performance promotion.
Evidence: `benchmarks/baselines/source_exception_heap_20260909/`.
Sequencing: [live plan](../../compiler/INTEGRATED_COMPILER_PLAN.md).

### Completion roots and broader assertions corpus (2026-09-09)

Owners: `COMPILER-DEVEX-1`, `E2E-REAL-6F`, `W4-PRODUCT-1`, `W2.4a`.
Sync: `COMPLETION-CORPUS-2026-09-09`.
SM120 validates source completion independently; uncertain free outcomes use host fault injection, not destructive device recovery. The Super-Bear assertions-enabled all-target compiler lane passes all four NVIDIA-owned fixtures and the 474-fixture union is complete. This structural result does not replace CUDA device execution or performance proof.
Shared changes: forward AD declares Tile dependency, completed frames release
cached host exception roots, uncertain synchronous frees retain owners and
refuse retry. General native heap allocation/reclamation, full CPython frames
and isolation-based driver recovery remain open.
Evidence: `benchmarks/baselines/source_completion_retirement_20260909/`.
Sequencing: [live plan](../../compiler/INTEGRATED_COMPILER_PLAN.md).

### Four canonicalization legality improvements (2026-09-09)

Owners: `W5.5`, `W4-PRODUCT-1`. Sync: `CANONICAL-LEGALITY-2026-09-09`.
Host assertions build and CPU semantics pass; CUDA physical schedule/performance promotion is not established.
Shared contracts: actual permutation composition, matrix-swap-only matmul flags,
preserved epilogue operands, policy/metadata cast guards and conservative fusion
admission. Failed exception reconstruction releases private unpublished roots.
General native allocation/collection, full CPython frames and isolated driver
recovery remain open. Sequencing: [live plan](../../compiler/INTEGRATED_COMPILER_PLAN.md).

### Native exception arena and isolation boundary (2026-09-09)

Owners: `W4-PRODUCT-1`, `W2.4a`, `E2E-REAL-6`, `W5.2f`. Sync: `NATIVE-HEAP-ISOLATION-2026-09-09`.
Parity validated for the host contract on Super-Bear; process death is not yet
wired to the CUDA package launcher, so no poisoned-context recovery or device
heap claim is made. ReplaySSM kernels do not substitute for the missing shared Schedule SSD producer.


### Installed drivers and device measurements (2026-09-10)

Owners: `COMPILER-DEVEX-1`, `EVIDENCE-PACKET-1`. Sync: `INSTALLED-DEVICE-GATES-2026-09-10`.
Native 16x8 ANN numerical validation and independent repeated package measurements run on RTX 5070 SM120. Nine independent runs retain the incumbent: median speedup 0.999666x and lower bound 0.977299x fail the 1.02 margin; no global route is promoted. Device-clock attribution and broader workloads remain open.
Shared tooling: both compiler drivers and the layout library install through the compiler-tools component; CI consumes relocated-prefix smoke and the lit union. Evidence: `benchmarks/baselines/installed_device_gates_20260910/`.
Sequencing: [live plan](../../compiler/INTEGRATED_COMPILER_PLAN.md).


### Arena readers, BF16 migration and isolated ANN (2026-09-10)

Owners: `W4-PRODUCT-1`, `W2.4a`, `E2E-REAL-6/F`, `MSW-9`, `TPROF-NATIVE-1`.
Sync: `OWNERSHIP-CENSUS-PROFILE-2026-09-10`.
Native isolated ANN execution and stopped-worker teardown/replacement pass on SM120. Nsight Systems/Compute record 32x8 square ANN: kernel time is small relative to host costs and one-block occupancy is low. This is not driver-hang recovery or promotion proof.
Shared contracts: exported arena leases pin storage and block mutation; partial collection reuses payload holes. Isolated ANN returns copied checked host results, retains uncertain workers and requires confirmed process death before replacement. General device heaps/readers and driver-fault recovery remain open.
Evidence: `benchmarks/baselines/ownership_census_profile_20260910/`.
Sequencing: [live plan](../../compiler/INTEGRATED_COMPILER_PLAN.md).

### Asynchronous isolation teardown (2026-09-10)

Owner: W2.4a. Sync: `ASYNC-ISOLATION-2026-09-10`.

SM120 native execution and stopped-worker asynchronous teardown/replacement independently passed. This is process-fault evidence, not a wedged CUDA driver test.

Shared recovery tickets retain owners on uncertain termination; ANN and declared isolated module owners can poll process teardown. General driver-hang recovery, device-health admission and broader external-reader integration remain follow-ups. Evidence: `benchmarks/baselines/async_isolation_20260910/`.

### Health admission and external reader scopes (2026-09-10)

Owner: W2.4a. Sync: `HEALTH-READERS-2026-09-10`.

SM120 workload health probes, replacement admission and two native external copy streams passed independently. Actual wedged CUDA driver recovery remains unproven.

Fresh ANN workers execute numerical probes before admission. External readers may borrow on multiple declared streams; eventless dependency insertion refuses implicit host synchronization. Unkillable processes, driver reset/recovery and arbitrary external pointer lifetimes remain open. Evidence: `benchmarks/baselines/health_reader_admission_20260910/`.

PR #740 review follow-up (`HEALTH-READERS-2026-09-10`): eventless retirement remains retryable; parent ANN input checks preserve healthy workers; pending context exit retains or reclaims ownership without masking caller errors. Shared host lifecycle fixes; existing exact-device evidence remains scoped to its recorded revision.

### FP64 artifact ownership and route callers (2026-09-10)

Owners: E2E-REAL-6 / E2E-REAL-6F. Sync: `F64-ROUTE-OWNERSHIP-2026-09-10`.

No CUDA fp64 schedule or device evidence is transferred from x86; existing NVIDIA scheduling gates remain. Caller candidates and helper-to-emitter paths are recorded in `benchmarks/baselines/f64_route_ownership_20260910/`; they are not device certificates.

### Dynamic readers and row-private ANN (2026-09-10)

Owners: W2.4a / MSW-9. Sync: `DYNAMIC-READERS-ROW-ANN-2026-09-10`.

SM120 independently executes 64x8 square ANN after compaction. Nsight kernel/copy/API costs are recorded separately. No promotion. Checked public/paired frames now support declared multi-stream reads; general heterogeneous capture and external-pointer escape remain open. Evidence: `benchmarks/baselines/ann_row_private_20260910/`.

### Shared SSD and retryable completion (2026-09-10)

Historical increment: its remaining-work statements are superseded by **Heap IR, raised attention and cooperative SSD** below. Retained measurements keep their original scope.

Owners: W5.2f / W2.4a / W4-PRODUCT-1 / AD-RESIDUAL-EVAL-1.
Sync: `SSD-RETRY-COMPLETION-2026-09-10`.

Follow-up required: nvidia target SSD tiling, runtime packaging and exact-device execution are not supplied by the shared structured-loop baseline. Late-death reconciliation and paired AD retirement are host-tested ownership changes; they do not establish driver health. Arena cycle/root operations remain host-produced, and exception source frames remain diagnostic metadata. At that increment, native heap producers, full CPython deoptimization and attention raising remained open; see the superseding entry below and [the owning queue](../../compiler/INTEGRATED_COMPILER_PLAN.md#w52f).

### Native heap, attention recipes and SSD device proof (2026-09-10)

Historical increment: its remaining-work statements are superseded by **Heap IR, raised attention and cooperative SSD** below. Retained measurements keep their original scope.

Owners: W4-PRODUCT-1 / FRONTEND-IR-MEDIUM-1 / W5.2f.
Sync: `NATIVE-HEAP-ATTENTION-SSD-2026-09-10`.

RTX 5070 SM120 executes the replay-bound serial SSD package for chunks 1, 2 and 5 with immutable-input, output, carry and checkpoint checks. This is correctness evidence; cooperative scheduling, device timing and promotion remain open. Exact dense f32 attention recognition and native bucket instantiation are shared artifact contracts, not sibling-device certificates. Automatic heap IR producers, full CPython frames and broader attention remain follow-ups. Evidence: `benchmarks/baselines/native_heap_attention_ssd_20260910/`; [current owner](../../compiler/INTEGRATED_COMPILER_PLAN.md#w52f).

PR #741 review follow-up (`NATIVE-HEAP-ATTENTION-SSD-2026-09-10`): the GPU pass now declares `tessera.ssd.source` in both metadata inventories. Expanded copies already narrow their outer loops to the owning thread row; new NVIDIA/ROCm lowering regressions pin global/private/output indexing. Shared contract validation only; no new device performance evidence. Apple and x86 execution claims remain unchanged.

### Heap IR, raised attention and cooperative SSD (2026-09-10)

Owners: W4-PRODUCT-1 / FRONTEND-IR-MEDIUM-1 / W5.2f / TPROF-NATIVE-1.
Sync: `HEAP-IR-ATTENTION-COOPERATIVE-SSD-2026-09-10`.

Parity validated for the bounded SM120 routes: two raised dense f32 attention buckets execute through replay-bound native packages; cooperative SSD passes input/output/carry/checkpoint oracles. Nsight reports kernel and transfer/API costs separately. Broader attention and promotion remain open. Evidence: `benchmarks/baselines/heap_attention_cooperative_ssd_20260910/`. The SSD candidate remains opt-in; single-session measurements do not promote it. [Current queue](../../compiler/INTEGRATED_COMPILER_PLAN.md#w52f).

### Dynamic heap payloads, checkpoint AD and paired measurements (2026-09-10)

Owners: W4-PRODUCT-1 / FRONTEND-IR-MEDIUM-1 / W5.2f / TPROF-NATIVE-1.
Sync: `DYNAMIC-HEAP-SSD-CHECKPOINT-PAIRS-2026-09-10`.

CUDA SSD now has nine independent paired process runs with unchanged compiler/image/binding identity. Median resident-window speedup is 6.88x; the conservative population-median lower bound is 6.74x. This objective includes host submission gaps. Missing calibration and exact-artifact selector admission prevent promotion. Dynamic heap payloads and SSD checkpoint VJP are CPU contracts; GPU implementations remain follow-ups. Evidence: `benchmarks/baselines/ssd_paired_process_20260910/` and focused native tests. [Current queue](../../compiler/INTEGRATED_COMPILER_PLAN.md#w52f).

### GPU payload frames, mixer AD and artifact selection (2026-09-10)

Owners: W4-PRODUCT-1 / FRONTEND-IR-MEDIUM-1 / W5.2f / TPROF-NATIVE-1.
Sync: `GPU-HEAP-SSD-AD-ADMISSION-2026-09-10`.

Parity validated on SM120 for bounded transactional f32 payload frames and native SSD checkpoint VJP fed from cooperative forward. Scaled/unscaled attention executes two buckets without Graph reconstruction. Exact measured package binding selected the serial incumbent because native CUDA calibration admission is still missing. No promotion; general heap collection and resident automatic GPU AD remain follow-ups. Evidence: `benchmarks/baselines/gpu_heap_ssd_ad_20260910/`. [Current queue](../../compiler/INTEGRATED_COMPILER_PLAN.md#w52f).

### Reusable GPU pools, resident AD and window calibration (2026-09-10)

Owners: W4-PRODUCT-1 / FRONTEND-IR-MEDIUM-1 / W5.2f / TPROF-NATIVE-1.
Sync: `GPU-POOL-RESIDENT-AD-CALIBRATION-2026-09-10`.

Parity validated on SM120: fixed-slot mark/sweep, cyclic reachability, stale-edge refusal, reuse and resident SSD forward/VJP snapshots. Causal GQA executes Q<K and Q>K buckets. Nsight/event windows agree within 2.26%, but 1.957x trace overhead, dirty source and WSL refuse promotion. Renew exact-artifact paired measurements and obtain eligible per-process calibration. Evidence: `benchmarks/baselines/pool_resident_gqa_calibration_20260910/`. [Current queue](../../compiler/INTEGRATED_COMPILER_PLAN.md#w52f). General concurrent/object collection and public asynchronous resident AD remain open.

### Stream-owned object graphs and asynchronous AD (2026-09-10)

Owners: W4-PRODUCT-1 / FRONTEND-IR-MEDIUM-1 / W5.2f / TPROF-NATIVE-1.
Sync: `STREAM-OBJECTS-ASYNC-AD-WINDOWS-2026-09-10`.

Parity validated on SM120 for reader-ordered opaque byte graph collection, fourth-edge reachability, cyclic collection and asynchronous projected SSD VJP composition/retirement. Windowed GQA executes with mandatory nonempty-row constraints. The larger SSD case passes clock agreement and overhead gates; clean-source/bare-metal evidence and nine independently calibrated process pairs remain required. No promotion. Evidence: `benchmarks/baselines/async_objects_windows_20260910/`. [Current queue](../../compiler/INTEGRATED_COMPILER_PLAN.md#w52f).

### Snapshot marking and public asynchronous VJP (2026-09-10)

Owners: W4-PRODUCT-1 / FRONTEND-IR-MEDIUM-1 / W5.2f / TPROF-NATIVE-1.
Sync: `SNAPSHOT-PUBLIC-AD-BIAS-2026-09-10`.

RTX 5070 SM120 validates snapshot retention during mutation, discovered cyclic objects, public asynchronous SSD VJP and whole-frame retirement. Full-shape finite additive attention bias passes native differential execution. WSL correctness is not measured overlap or promotion evidence; clean bare-metal calibration remains required. Final sweep is exclusive; arbitrary heaps, general public traced AD and general masks remain open. Evidence: `benchmarks/baselines/snapshot_public_ad_20260910/`. [Current queue](../../compiler/INTEGRATED_COMPILER_PLAN.md#w52f).

### Benchmark/compiler alignment (2026-09-10)

Owner: EVIDENCE-PACKET-1 / TPROF-NATIVE-1. Sync: `BENCHMARK-COMPILER-ALIGNMENT-2026-09-10`.

Shared benchmark admission now refuses inherited dense-Krylov performance eligibility. Historical SM120 packets remain unchanged. Follow-up required: exact-image native operator adapters and clean bare-metal timing. [Shared review](../../../../benchmarks/COMPILER_ALIGNMENT.md); [sequencing](../../compiler/INTEGRATED_COMPILER_PLAN.md#evidence-packet-1). No cross-backend performance transfer or promotion.

### Extended benchmark alignment (2026-09-10)

Owner: EVIDENCE-PACKET-1 / TPROF-NATIVE-1. Sync: `BENCHMARK-ALIGNMENT-EXTENDED-2026-09-10`.

SuperBench synthetic attention timing is retired. Existing wrappers remain CPU/reference/artifact entry points; native CUDA package adapters and profiler-backed DLOP dispatch counts are follow-up work. SM120 sealed evidence is untouched. [Review and follow-ups](../../../../benchmarks/COMPILER_ALIGNMENT.md#additional-suite-review--2026-09-10).

### Native benchmark adapters (2026-09-10)

Owner: EVIDENCE-PACKET-1 / TPROF-NATIVE-1 / W5.2f. Sync: `NATIVE-BENCHMARK-ADAPTERS-2026-09-10`.

Owning-device correctness validated on RTX 5070 (SM120), WSL: three ANN workloads, one accepted launch per variant/call, six public SSD VJP comparisons against independent finite differences. Follow-up required: profiler correlation, broader workloads and bare-metal promotion. [Evidence](../../../../benchmarks/baselines/native_benchmark_adapters_20260910/README.md). Instrumented host-wall timing is diagnostic only.

### Broader adapters and kernel attribution (2026-09-10)

Owner: EVIDENCE-PACKET-1 / TPROF-NATIVE-1 / W5.2f. Sync: `BROADER-ADAPTERS-PROFILING-2026-09-10`.

Validated all four native SuperBench workloads on SM120. Nsight correlated all 701 dedicated SSD kernels to successful owning-process launch calls. Mixed-artifact attribution and clean performance admission remain open. [Evidence](../../../../benchmarks/baselines/broader_adapters_20260910/README.md). WSL diagnostics do not qualify for performance promotion.

### Matrix adapters and mixed attribution (2026-09-10)

Owner: EVIDENCE-PACKET-1 / TPROF-NATIVE-1. Sync: `MATRIX-MIXED-ATTRIBUTION-2026-09-10`.

Validated scheduled fp16 GEMM and fp32 attention with independent NumPy oracles. Mixed NVTX capture attributes sequential synchronous calls by run/artifact/image identity and CUDA API correlation. Follow-up: asynchronous/multi-thread ranges, broader shapes/dtypes/masks and clean performance evidence. [Evidence](../../../../benchmarks/baselines/matrix_mixed_20260910/README.md). No promotion.

### SSD numerical admission repair (2026-09-11)

Owner: EVIDENCE-PACKET-1 / W5.2f. Sync: `SSD-NUMERICAL-ADMISSION-2026-09-11`.

Shared SSD admission now rejects any forward/carry/checkpoint maximum absolute error above 1e-6 before speed or calibration gates. Summary-only evidence cannot reconstruct relative tolerance; larger errors need richer numerical evidence. No new device measurement or promotion.

### Program retirement and slot discovery (2026-09-11)

Owner: W4-PRODUCT-1 / W5.2f. Sync: `PROGRAM-RETIREMENT-SLOTS-2026-09-11`.

Owning-device proof passed: declared-slot self-cycle discovery/collection and two public-VJP SSD frames retiring through off-thread module unloading with context synchronization forbidden on the program path. No performance claim. Concurrent sweep, arbitrary extension heaps and general traced GPU AD remain open. [Evidence](../../../../benchmarks/baselines/program_retirement_20260911/README.md).

### 2026-09-11 — Slotted snapshot type identity

Owner: W4-PRODUCT-1. Sync: `SLOTTED-MODULE-IDENTITY-2026-09-11`.
Slotted records now validate and include the class module, matching ordinary
instances; inherited slot owners also include their module. The host snapshot producer is corrected; device collection consumes opaque payload bytes, so no device execution claim is added.

### 2026-09-11 — Incremental heap and traced SSD

Owners: W4-PRODUCT-1 / W5.2f / FRONTEND-IR-MEDIUM-1.
Sync: `INCREMENTAL-HEAP-TRACE-2026-09-11`.

CUDA SM120 validates generation reuse between sweep batches, a trusted extension cycle, and all nine gradients of two traced SSD calls. Raised full-shape negative-infinity attention masks pass differential execution; empty rows refuse.
Sweep batches still run exclusively; lifecycle 2 means logically retired, not
reusable. Arbitrary extension discovery, simultaneous sweeping, general GPU AD
and Boolean/broadcast/empty-row masks remain open. No performance promotion.
[Evidence](../../../../benchmarks/baselines/incremental_heap_trace_20260911/README.md).

### 2026-09-11 — DAG accumulation and snapshot readers

Owners: W5.2f / W4-PRODUCT-1 / FRONTEND-IR-MEDIUM-1.
Sync: `DAG-SNAPSHOT-2026-09-11`.

CUDA SM120 validates snapshot reader scopes during live-pool collection and a three-call SSD DAG with shared cotangent accumulation. Ragged GQA + causal/window + irregular additive masks pass for Q>K and K>Q.
Snapshot readers access immutable copied epochs; unrestricted same-storage
concurrent sweeping still requires barriers. General-key maps/sets/bytearrays
are discovered; undeclared builtin-subclass native payloads refuse. Canonical
composition IR, broader AD families and Boolean/broadcast masks remain open.
[Evidence](../../../../benchmarks/baselines/dag_snapshot_20260911/README.md). No performance promotion.

### 2026-09-11 — Heap barrier architecture exploration

Owners: W4-PRODUCT-1 / W2.4a / DISPATCH-BREAKER.
Sync: `HEAP-BARRIERS-2026-09-11`.

Follow-up required: CUDA global-memory atomic/scope and event-lineage proof on the owning GPU.
[Architecture review](../../compiler/HEAP_BARRIER_ARCHITECTURE_REVIEW.md)
selects a nonmoving model, reader-protected reuse and barrier-aware updates,
retaining exclusive final remark. Design only; no new execution or promotion.

### 2026-09-11 — Bounded heap protocol implementation

Owners: W4-PRODUCT-1 / W2.4a / DISPATCH-BREAKER.
Sync: `HEAP-BARRIERS-2026-09-11`.

Parity validated for the bounded stream-ordered protocol on SM120: graph rejection, new-root final remark, retired-slot refusal and two-stream copies before reuse. Five single-process WSL samples show higher split-route cost; no promotion.
[Evidence](../../../../benchmarks/baselines/heap_barriers_20260911/README.md).
The serialized protocol requires exclusive stream epochs and kernel-completion
publication. Per-object admission, dirty-work marking, concurrent final
retirement and atomic scope lowering remain follow-ups; this is not concurrent
sweeping or calibrated kernel timing.

### 2026-09-11 — Per-object readers and incremental marking

Owners: W4-PRODUCT-1 / W2.4a / DISPATCH-BREAKER.
Sync: `HEAP-INCREMENTAL-2026-09-11`.

Parity validated on SM120 for per-object payload admission, incremental graph shading, incomplete-mark refusal and retirement/reuse during a different admitted reader. Follow-up: asynchronous checked admission, cooperative marking and exact-device atomic multi-writer proof; no timing promotion.
[Evidence](../../../../benchmarks/baselines/incremental_object_heap_20260911/README.md).
Metadata remains single-writer; final retirement can coexist with admitted
immutable-payload readers. Admission/unpin and close still have synchronous
boundaries. This is not unrestricted concurrent sweeping or measured overlap.


### 2026-09-11 — Asynchronous heap receipts and marking allocation

Owners: W4-PRODUCT-1 / W2.4a / DISPATCH-BREAKER.
Sync: `HEAP-ASYNC-2026-09-11`.

Parity validated for the bounded SM120 device recorder: private receipt cycles, cancellation, stale admission and grey allocation publication. Concurrent metadata atomics and measured overlap remain unproven.
[Evidence](../../../../benchmarks/baselines/async_heap_20260911/README.md).
Private statuses remain owned until completion; uncertain unpin cannot retry.
Metadata writers remain serialized; the split-reservation model has a race.
Finalization/teardown remain synchronous. No measured overlap or promotion.


### 2026-09-11 — Native heap handshake and deferred destruction

Owners: W4-PRODUCT-1 / W2.4a / DISPATCH-BREAKER.
Sync: `HEAP-HANDSHAKE-2026-09-11`.

Parity validated on SM120 for shared-gate busy refusal, two-stream submissions, polled finalization and off-thread teardown. Full resident metadata gate integration and calibrated performance remain open.
[Evidence](../../../../benchmarks/baselines/heap_handshake_20260911/README.md).
Native graph/retirement transactions share a try-lock; busy status requires
explicit retry. Resident-pool metadata remains epoch serialized until every
user participates. Teardown is asynchronous to the caller, with bounded worker
admission and retained uncertain failures; driver latency is unbounded.
No measured overlap or performance promotion.


### 2026-09-11 — Gated metadata owner and isolated recovery

Owners: W4-PRODUCT-1 / W2.4a / DISPATCH-BREAKER.
Sync: `HEAP-GATED-ISOLATION-2026-09-11`.

Parity validated on SM120 for gated metadata, copied inspection, pinned readers and normal/stopped-worker recovery. A stopped process is not an actual CUDA driver hang; replacement health and performance remain open.
[Evidence](../../../../benchmarks/baselines/gated_heap_20260911/README.md).
The opt-in gated owner admits no live metadata or legacy snapshot/import bypass.
Epochs remain for ordering/lifetimes. Eight process slots cap isolated heap owners;
timeout and unconfirmed death retain resources. No measured overlap or promotion.


### 2026-09-11 — Snapshot recovery-close repair after #744

Owners: W2.4a / DISPATCH-BREAKER. Sync: `SNAPSHOT-RECOVERY-2026-09-11`.

Owning-device follow-through: SM120 receipt-copy fault recorder checks recovery close of parent and snapshot.
Cleanup now propagates recovery readiness through snapshot and parent epoch
waits without bypassing completion or active-reader checks. Ordinary poisoned
reads still refuse. [Evidence](../../../../benchmarks/baselines/snapshot_recovery_20260911/README.md).
No performance promotion or actual driver-hang recovery claim.

## Dtype and ownership reconciliation — 2026-09-11

Owners: E2E-REAL-6F / E2E-REAL-6 / NUMPOL-CARRIER-1 / LAYOUT-ALG-1.
Sync: `DTYPE-CODEGEN-2026-09-11`.

The route census now separates lexical scopes; the dtype inventory projects existing
contracts across all canonical and planned storage names without adding support states.
Follow-up required: join SM120 scalar/vector, Tensor Core and block-scale declarations to emitted PTX and exact-operation packets; conversion/storage is not native arithmetic.
No new execution or performance promotion is claimed.

### Dtype arithmetic execution follow-through — 2026-09-11

Sync: `DTYPE-CODEGEN-2026-09-11`; owners E2E-REAL-6 / NUMPOL-CARRIER-1 / LAYOUT-ALG-1.

Parity validated for 24 SM120 basic scalar/vector arithmetic rows, including division and special values, with exact-image SASS witnesses. Four generic FP8 conversion rows fail LLVM translation; packed/scaled matrix routes remain separately gated.
Evidence: [independent packets](../../../../benchmarks/baselines/dtype_arithmetic_20260911/README.md).
No performance promotion; remaining packing, policy and layout consumers stay open.

### FP8 conversion and numerical boundaries — 2026-09-11

Sync: `DTYPE-CODEGEN-2026-09-11`; owners NUMPOL-CARRIER-1 / LAYOUT-ALG-1.

SM120: four FP8 scalar/vector cases pass exhaustive 65,536 input pairs each. Bool logic and bounded complex components pass. Ten INT4/FP4/NVFP4/FP6 decode cases pass with both packing axes, varying scales, offsets and padding. This is software conversion/generic decoding, not Tensor Core or performance promotion.
Evidence: [follow-through packets](../../../../benchmarks/baselines/dtype_followthrough_20260911/README.md). No performance promotion.

## Reconciliation and wave start — 2026-09-12

Sync `NUMPOL-CARRIER-1` / `E2E-REAL-6` / `FRONTEND-IR-MEDIUM-1` / `W2.4a`.

Fresh CUDA SDK 13.4.1 / nvcc 13.4.59 / driver 610.88 on RTX 5070 passes all 34 arithmetic rows with explicit MLIR toolkit selection and matching cuobjdump. Old 13.3 packets remain historical. Serialized broadcast attention and owning-device snapshot retirement require follow-up; WSL timings do not promote candidates.

See the [integrated wave record](../../compiler/INTEGRATED_COMPILER_LOG.md#2026-09-12--reconciliation-and-wave-start).

## Native numerical and packaging slices — 2026-09-12

Sync `NUMPOL-CARRIER-1` / `E2E-REAL-6` / `FRONTEND-IR-MEDIUM-1`.

Parity validated on owning SM120 with CUDA 13.4.1: two B=2 ragged causal/GQA/window attention buckets use compact batch/head broadcast bias and match the oracle. The v2 f32 host ABI carries BiasB/BiasH and copies only physical bias storage; kernel logical dimensions remain unchanged. Follow-up required: query/key broadcast, Boolean/padding masks, wider storage and clean performance admission. The Apple-specific denormal carrier fails closed in generic CUDA storage.

Evidence: [native slice packets](../../../../benchmarks/baselines/native_slices_20260912/README.md). No performance promotion.

## Sparse runtime and isolated attention — 2026-09-14

Owner E2E-REAL-6; sync `SPARSE-RESIDENT-2026-09-14`.

Follow-up required for CUDA resident attention composition and isolated recovery. Shared process-boundary refactoring retains existing isolated ANN semantics; host tests are not CUDA recovery proof. RDNA4 sparse formats and HIP execution evidence do not transfer to NVIDIA. Registered ceil Graph contract does not change NVIDIA packaging.

Evidence: [bounded validation](../../../../benchmarks/baselines/sparse_runtime_20260914/README.md). No performance promotion; general Graph sparse capture, additional packing formats and actual driver-hang recovery remain open.

## Sparse capture, byte formats and scan — 2026-09-14

Owner E2E-REAL-6; sync `SPARSE-CAPTURE-2026-09-14`.

Follow-up required for CUDA public AD composition and isolated admission. RDNA4 sparse byte packing and HIP execution do not confer NVIDIA sparse support; shared cumsum registration changes no NVIDIA package route.

[Evidence](../../../../benchmarks/baselines/sparse_capture_20260914/README.md). Actual driver-hang recovery and calibrated promotion remain open.

### 2026-09-14 — Mixed sparse operands

Sync key: `SPARSE-MIXED-2026-09-14`; owner: [E2E-REAL-6](../../compiler/INTEGRATED_COMPILER_PLAN.md#e2e-real-6).

Sparse package identity now includes B storage. Schedule/Tile/Target preserve independent integer signs and mixed FP8 types through native lowering. Not applicable to nvidia physical lowering: these SWMMAC contracts are gfx1201-only. No sibling device or performance evidence transfers. Automatic selection/native Graph lowering, INT4, general AD and actual driver-hang recovery remain separate work.

### 2026-09-14 — INT4 logical packing

Sync `SPARSE-INT4-2026-09-14`, owner E2E-REAL-6. Shared sparse IR carries a verified 4/8-bit integer interpretation; byte storage is range-checked and native lowering packs nibbles. Not applicable to nvidia execution: this physical lowering targets gfx1201 only. No sibling support or device proof is inferred. Logical AD must precede packing; automatic sparse selection/native Graph lowering and general AD remain open.

### 2026-09-14 — Native sparse Graph and logical AD

Sync `SPARSE-GRAPH-AD-2026-09-14`; owner E2E-REAL-6. C++ now owns declared checked-2:4 half Graph lowering and the emitted ABI. AD remains on logical matrices, with operand-order/repeated-edge correctness fixes. Not applicable to nvidia physical execution: no sparse consumer changes on this backend. Shared AD/IR contracts confer no sibling device proof.

### 2026-09-14 — Automatic native sparse selection

Sync `SPARSE-AUTO-2026-09-14`; owner E2E-REAL-6. The explicit auto_2to4 Graph policy selects per K tile using wave-uniform agreement, with a native dense branch. Not applicable to nvidia execution: only gfx1201 emits this policy; no sibling schedule is transferred. Default promotion, broader regions and arbitrary AD remain open; no measured speedup claimed.


### Native composed HVP execution — 2026-09-14

Sync: `AD-HVP-2026-09-14`; owner: AD-HIGHER-1 / W4-PRODUCT-1.
Shared changes: tracer-to-SCF counted regions, scalar extraction tangents, passive
comparison predicates, signature-owned CPU outputs and ownership-clone lowering.
Follow-up required: CUDA higher-order package binding and exact SM120 execution remain open.


### Saved-product HVP and GPU export — 2026-09-14

Sync: `AD-HVP-SAVE-2026-09-14`; owner AD-HIGHER-1 / W4-PRODUCT-1.
Shared compiler captures continuous residual tangents; typed export reuses GPU
tape lowering without rebuilding derivative formulas in Python.
Bounded CUDA execution is now covered by the AD-HVP-DEVICE follow-through below; broader products remain open.

Shared follow-through: native GPU tape copies require bounded extents even when
source and destination use identical shape SSA. Constant-bounded views remain
admitted; unbounded loaded sizes refuse. This changes CUDA/HIP admission; Apple
and x86 do not consume that GPU pass.


### HVP product identity and device breadth — 2026-09-14

Sync `AD-HVP-DEVICE-2026-09-14`; owner AD-HIGHER-1 / W4-PRODUCT-1.
Parity validated on Super-Bear RTX 5070 / SM120: composed cubic and counted-loop
HVPs execute for rank-one and rank-two inputs, checking gradients and curvature
at numerical zero as well as nonzero inputs. The test verifies the current
CUDA context architecture and retires that context after buffers and bindings.
`test_native_hvp_execution.py`: 7 passed, 10 skipped; four passing cases are
CUDA device executions. WSL2, CUDA compiler 13.4.59, assertions LLVM 23.1.1;
Ubuntu clang 23.1.1 compiles the host sizing helper because the assertions
bundle lacks clang. The isolated source build uses LLVM's matching -fno-rtti.
Super-Bear main remains clean at origin/main; validation used /tmp/tessera-ad-wave.
Export now selects products generated by the pass, not arbitrary __hvp symbols.
Effectful CFG, dynamic outputs, higher orders and asynchronous HVP frames remain
open. No performance promotion or Metal/gfx1151 execution is inferred.


### Apple OS/toolchain validation — 2026-09-14

Sync `APPLE-METAL41-20260914`; owner IR-NATIVE-FOUNDATION-1.
Not applicable to nvidia execution: Apple-only language probes and fresh
fp16/bf16 validation do not change shared dtype registration, runtime ABI or
this target's admission. No sibling correctness or performance proof transfers.
See [Apple follow-through](../apple/todo.md#macos-27--metal-41-low-precision-validation--2026-09-14).

SDK27 bridge follow-through (`APPLE-METAL41-20260914`): Apple enum/extent/scale-plane fixes and execution admission assessed. Not applicable to nvidia code generation or runtime ABI; no sibling dtype, schedule or execution proof changes.

Packed numeric binding (`APPLE-METAL41-20260914`): explicit Apple-only status ABI and Python numerical entry added. Not applicable to nvidia execution; shared ABI inventory updated, no dtype or operation promotion. Owning-device numerical evidence does not transfer.

### Fleet-portable LLVM tools and ROCm toolkit detection — 2026-09-15

Sync `HOST-SWEEP-2026-09-15`; owner COMPILER-DEVEX-1. Shared change: the native
lanes resolved `mlir-opt` / `mlir-translate` / `llc` / `llvm-link` from a
hard-coded `/usr/lib/llvm-23/bin`, so every native package test was a false
failure on the Mac. `python/tessera/compiler/llvm_tools.py` now resolves the
matched LLVM 23 companion (`TESSERA_LLVM_BIN`, canonical apt/Homebrew prefixes
verified by `llvm-config`, then `PATH`), and the shared subprocess choke point
in `native_gpu_storage._run` maps a caller-supplied non-existent companion path
to it. Separately, `runtime._rocm_toolkit_root` accepted any `ld.lld` on `PATH`
as a ROCm root (Homebrew's `lld` keg on the Mac); it now requires a ROCm marker
(device-library bitcode, `.info/version`, `rocminfo`/`hipcc`/`amdgpu-arch`), and
`tests/_support/rocm_build.require_rocm_hsaco_toolkit` gates `backend='rocm'`
package tests on it. ROCm-lowering tests gate on the registered passes through
`compiler_tool.require_tessera_opt(...)` rather than on driver presence.
Owning outcome for nvidia: **follow-up required** — `nvidia_native.py` now resolves its LLVM companions through the shared resolver; confirm on Super-Bear that a bare pytest still finds `/usr/lib/llvm-23/bin` (it is the first canonical prefix) and that no NVIDIA package test changed from run to skip. No schedule or execution change.


Matched-value comparison (`APPLE-METAL41-20260914`, 2026-09-15): benchmark methodology now checks exact quantized operands against float64 before timing. No nvidia compiler/runtime changes or device evidence; Apple timing and layout results are not transferable.

### Shared TilingPass matmul epilogue — 2026-09-15

Sync `TILING-MATMUL-EPILOGUE-2026-09-15`; owner E2E-REAL-6.
Not applicable to nvidia execution: the tiled inner K step is unchanged, so the scheduled-matmul recognizers see the same nest; a biased matmul now carries an explicit `tessera.broadcast` + `tessera.add` after the nest instead of failing inside tiling. No sibling correctness or performance proof transfers; no CUDA ABI or dtype change.
See [Apple follow-through](../apple/todo.md#shared-tilingpass-matmul-epilogue-preserved--2026-09-15).

### @jit low-precision front door — 2026-09-15

Sync `LOWP-FRONT-DOOR-2026-09-15`; owner E2E-REAL-6.
Shared-frontend change assessed: `to_graph_ir_module` now verifies legality against the compiling target (a `GPUTargetProfile` target keeps the CPU table, so nvidia behaviour is unchanged), and Graph IR spells fp8/fp4 as the MLIR builtins `f8E4M3FN`/`f8E5M2`/`f4E2M1FN`. Not applicable to nvidia execution; no dtype admission, ABI or proof change.
See [Apple follow-through](../apple/todo.md#jit-front-door-for-84-bit-storage-tensors--2026-09-15).

### Matmul epilogue markers and activation order — 2026-09-15

Sync `MATMUL-EPILOGUE-MARKERS-2026-09-15`; owner E2E-REAL-6.
Shared frontend + tiling change assessed: the tiled inner K step is unchanged (markers and `activation` no longer ride on it); a traced biased/activated matmul now carries explicit broadcast + add + activation ops after the nest instead of failing the Graph IR verifier. Not applicable to nvidia execution; no ABI, dtype or proof change.
See [Apple follow-through](../apple/todo.md#matmul-epilogue-markers-and-activation-order--2026-09-15).

## Attention broadcast masks on every axis — 2026-09-15

Sync `ATTN-QK-BROADCAST-2026-09-15`; owner FRONTEND-IR-MEDIUM-1.

Parity validated on owning SM120 (RTX 5070, CUDA 13.4.1, driver 610.88): six B=2 ragged causal/GQA/window attention buckets — batch/head `[1,1,Q,K]`, key-padding `[1,1,1,K]` and per-query `[B,Hq,Q,1]` bias at Q/K = 3/5 and 5/3 — execute without host expansion and match the oracle within 6e-8. The v3 f32 host ABI carries `BiasB`/`BiasH`/`BiasQ`/`BiasK`, validates each against 1 or the logical extent and copies only the physical storage; the kernel reads index 0 with stride 1 on every broadcast axis and the seven logical kernel extents are unchanged. Empty-row refusal judges the logical broadcast view. `check-tessera-nvidia` 61/61 in the rebuilt `build-nvidia-cuda/` tree. Follow-up required: Boolean/padding masks as an operand, f16/bf16 broadcast storage, and any performance admission (none claimed).

Operational note: the NVIDIA native runtime resolves `tessera-nvidia-opt` and `libtessera_nvidia_ptx_launch.so` from `build-nvidia-cuda/`, not `build/`. A `build/`-only rebuild left a two-week-old compiler in place that ignored `bias_shape` and a launcher that rejected the v3 dims (`rc=5`); rebuild `build-nvidia-cuda/` before any sm_120 device claim.

Evidence: [attention broadcast packets](../../../../benchmarks/baselines/attention_broadcast_20260915/README.md). No performance promotion.
## Probed admission and health-checked heap replacement — 2026-09-15

Sync `HEAP-REPLACEMENT-HEALTH-2026-09-15`; owner W4-PRODUCT-1.

Parity validated on owning SM120 (RTX 5070, CUDA 13.4 / driver 610.88): a spawned heap worker is admitted only after its in-process device probe (allocate → one live slot → bitwise pinned readback → empty-graph mark → reclaim) verifies; `replacement()` refuses before confirmed predecessor death and admits a freshly probed worker after it; all nine recorder proofs pass. Prerequisite: the arena replay pipeline is now shared with the packager — on `origin/main` every heap/SSD/ANN/exception/public-result/gradient-sum package was refused on this box as "disagrees with native replay" since 100a2980. Follow-up required: legacy snapshot/import migration; the stall is injected, not a driver hang.

Evidence: [CUDA/HIP replacement packets](../../../../benchmarks/baselines/gated_heap_replacement_20260915/README.md). No performance promotion.

## Fleet reds follow-through — 2026-09-15

Sync `FLEET-REDS-2026-09-15`.

Owner-directed fleet-red follow-through (PR #761), NVIDIA outcome: parity validated on Super-Bear (RTX 5070) — the heap replay tests now select their lane through `tests/_support/environment.native_storage_target()` (ptxas present → `nvidia`/`sm_120`) and resolve the matched LLVM bin per host; 47 passed with no change in what the sm_120 lane packages. The `tessera-jit` configure guard (libffi / MLIRExecutionEngine) does not fire here (both present; `libtessera_jit.so` still builds in `build/` and `build-nvidia-cuda/`). The ROCm diagnostic and Apple slice changes are not NVIDIA paths. No performance promotion.

## Batched native geometric products — 2026-09-16

Sync `GA-NATIVE-BATCHED-2026-09-16`; owner W6.4.

Follow-up required: no sm_120 Clifford lane exists (the proof ladder's GA row has no `nvidia_sm120` column); the GPU package route through the arena pipeline is the next slice and will be proven on Super-Bear. Super-Bear's `build/` now configures the Clifford backend ON (CPU lane parity on its Zen 2 host recorded under x86).

See the [plan log entry](../../compiler/INTEGRATED_COMPILER_LOG.md#2026-09-16--batched-geometric-products-execute-through-the-mlirllvm-backbone).

## Clifford product family behind the JIT — 2026-09-16

Sync `GA-NATIVE-FAMILY-2026-09-16`; owner W6.4.

Follow-up required: CPU-only; the arena-pipeline GPU slice will be proven on Super-Bear (Clifford lit 19/19 there today).

See the [plan log entry](../../compiler/INTEGRATED_COMPILER_LOG.md#2026-09-16--the-clifford-product-family-executes-behind-the-mlirllvm-jit).

## Clifford family through the native GPU route — 2026-09-16

Sync `GA-NATIVE-GPU-2026-09-16`; owner W6.4.

Parity validated on owning sm_120 (RTX 5070, CUDA 13.4 / driver 610.88): ten Clifford ops × three shapes through the native storage route match the GA reference (worst abs error 2.4e-7); grade-2 pruning emits 24 of 64 products in the NVVM kernel; `runtime.launch` row `nvidia_sm120` / `nvidia_clifford_native_compiled` reports native_gpu. This is the first sm_120 GA row in the proof ladder. No performance measured; the route pays per-call host transfers and is correctness evidence only.

See the [plan log entry](../../compiler/INTEGRATED_COMPILER_LOG.md#2026-09-16--the-clifford-family-reaches-rocm-and-sm120-through-the-arena-pipeline) and the [device packets](../../../../benchmarks/baselines/clifford_native_gpu_20260916/README.md).

## sm_120 engineering loops: the device-timed corpus reaches dispatch, the cubin gets its store check, the HVP packet lands — 2026-09-17

Sync `GFX1201-PARITY-2026-09-17` (shared branch with the ROCm slices 3-5); owner COMPILER-DEVEX-1 with W4-PRODUCT-1.

**Four of the six items the plan listed for this backend were already closed
on `main`, and the queue said so further down; the record here corrects the
premises before the work.** The emitted PTX GEMM has had a CUDA-event device
timer since the `columnMajorGrid` fix (`tessera_nvidia_ptx_benchmark`), and
the corpus races all four sm_120 matmul candidates under `timing="device"`;
the Clifford family has an `nvidia_sm120` column in the GA proof ladder and a
packet; the EBM nonlinear energies and sphere integrator run on sm_120 (32/32);
`__nv_fsqrt_rn` is the shipping sqrt on the NVVM row-program route. What was
open, and is now closed:

* **Production dispatch never read the device-timed corpus.** `run_arbitrated`
  consulted `corpus_winner` with its default wall-clock timing only, so the
  rows proving the emitted GEMM 1.5-1.7x faster than the shipped delegate were
  never a selector input and `OP_MATMUL` fell back to tier priority
  (`NVIDIA-TIER-PRIORITY-IS-WRONG-AT-SCALE-2026-08-30`, follow-up #1). It now
  consults the device-timed verdict first and the wall-clock one only when no
  device row exists; both stay hints behind availability and the F4 gate
  (`test_default_dispatch_consults_the_device_timed_row_first`).
* **The cubin-side store check.** `_reject_bodyless_image` covered the AMDGPU
  image only. The NVIDIA twin disassembles the fatbin with the toolkit's
  `cuobjdump --dump-sass` (found via `CUDA_HOME`/`CUDA_PATH`/`/usr/local/cuda`/PATH)
  and refuses a kernel with no `STG`; on a host without the tool it is
  skipped, never faked, and `cuda_store_check_available()` says which. Live on
  Super-Bear (the EBM, Clifford and HVP GPU tests all package through it).
* **Exact sm_120 HVP execution, recorded.** `native_hvp.py` was already
  backend-parametric and its test parametrized for sm_120; what was missing was
  the run and the packet. `benchmarks/record_native_hvp.py` records the
  device-resident gradient and Hessian-vector product against the closed forms
  for two source functions × three shapes: `benchmarks/baselines/native_hvp_20260917/nvidia_sm120.json`
  and `rocm_gfx1201.json`, both exact (worst abs error 0). Correctness only.

**Owed, with the first step named.**
* *NVFP4 emitted kernel as a consumer.* `ptx_emit.emit_nvfp4_block_scale_mma_ptx`
  is still tests-only: its entry is not in the launcher's ABI table, so
  registering it returns rc=5. The first step is an `invokeNvfp4Emitted`
  launcher entry for the fixed m16n8k64 warp tile plus an operand packer that
  lays A/B/scale fragments out in the PTX ISA's per-lane order; the
  exact-operation packet follows from that packer, and an arbiter candidate
  only once the kernel has general-shape dispatch (otherwise it is a
  declaration with a one-tile consumer).
* *Isolated CUDA attention.* `resident_attention.py` (CUDA) and
  `DriverIsolationLease` exist; `isolated_rocm_attention.py` composes them for
  HIP only. The first step is extracting the spawn/health-check body into a
  backend-parametric `IsolatedAttentionTape` and instantiating it over the CUDA
  resident program, proven by a `test_isolated_cuda_attention.py` twin on
  Super-Bear.
* *EBM sphere front door on CUDA.* `geo_sampling.sphere_langevin_step` has one
  device fast path (Apple's fused MSL kernel); on sm_120 the native route
  exists only through `native_langevin` explicitly. The first step is a
  `[rows, features]` row-program module (two projections, the affine step,
  the retract normalisation) through `native_row_program` behind a
  `_try_cuda_gpu_sphere_langevin_step_f32` branch.
* *Toolchain.* Driver 610.88 is unchanged (CUDA 13.3 API, PTX ≤ 9.3); the pin
  test says when a driver update lifts the cap. Nothing to do until then.

Fleet at the head: Super-Bear **16096 passed, 0 failed** (full sweep at
`e461e368`, both trees built, NVIDIA lit and core lit clean; the host-free
files touched afterwards 86 passed at `ecb086c5`); the ROCm and Mac rows are
in the log entry.

## The three owed sm_120 items, worked — 2026-09-18

Sync `GFX1201-PARITY-2026-09-17` (shared branch `claude/gfx1201-sm120-owed-loops`
with the ROCm owed items); owner COMPILER-DEVEX-1 with W4-PRODUCT-1.

* **NVFP4 emitted kernel as a consumer.** The launch bridge dispatches
  `tessera_nvfp4_mma_m16n8k64` (`invokeNvfp4Emitted`: five fixed buffers, one
  warp, no runtime dims) instead of falling off its name chain with rc=5.
  `compiler/nvfp4_fragments.py` lays the logical tile out per lane in the PTX
  ISA m16n8k64 order the on-silicon spike proved — A/B nibble words,
  `scale_vec::4X` scale words on the lower lane pair (A rows `gid`/`gid+8`)
  and lane 0 of each quad (B column `gid`) — folds the accumulator back, and
  carries the exact reference (e2m1 and ue4m3 decoded as the spike decodes
  them, one scale per 16-wide K block). `runtime._nvidia_nvfp4_emitted_mma`
  registers the emitted PTX once and runs the tile. Device: exact in all five
  scale modes (unit, uniform 0.5 and 2.0, the spike's mapped non-uniform,
  random) — `tests/unit/test_nvidia_nvfp4_emitted.py`, packet
  `benchmarks/baselines/nvfp4_emitted_20260918/nvidia_sm120.json`. Still one
  fixed tile per launch: a consumer, not an arbiter candidate.
* **Isolated CUDA attention.** The spawn, lease, health handshake, bounded
  relay and poison/recover/replace body moved from the HIP-only owner into
  `isolated_attention.IsolatedAttentionTape`; the ROCm owner keeps its worker,
  payload and split-reduced oracle as hooks (its tests unchanged).
  `isolated_cuda_attention.IsolatedCUDAAttentionTape` retains the primary
  context in the child, uploads Q/K/V, captures the saved-LSE checkpoint pair
  into `ResidentAttentionTape`, admits itself through zero/nonzero VJP probes
  against the checkpoint contract's numpy reference (GQA-aware, end-aligned
  causal mask), and relays host-array cotangents and gradients. Device: three
  sm_120 rows (causal/non-causal, sq≠sk) with backward at two cotangent scales,
  forced worker death, confirmed recovery and a fresh replacement that passes
  the same oracle — `tests/unit/test_isolated_cuda_attention.py`. Forced
  process termination is not a driver-hang test.
* **EBM sphere front door on CUDA.** `native_row_program.sphere_langevin_step_module`
  is the `[rows, features]` program (two tangent projections, the
  Euler-Maruyama step, the retraction with the host's underflow guard; the
  scalars fold on the host so the kernel carries none), packaged behind
  `energy._try_cuda_gpu_sphere_langevin_step_f32`, tried after the Apple
  lane and before the x86/ROCm affine lanes, with a failed compile remembered
  per feature width. Device: d = 16, 33 and 1024 against the numpy formula
  (rtol 2e-5), a zero update re-normalizing the state, and a 12-step chain
  spied to fire on the device every step and agreeing with the numpy chain
  under the same key — `tests/unit/test_cuda_ebm_geo_langevin_compiled.py`.
* **Toolchain.** Driver 610.88 unchanged (CUDA 13.3 API, PTX ≤ 9.3); the
  emitted NVFP4 packet records the `.version` the driver JIT was handed.

Fleet at the head: Super-Bear **16148 passed, 8 failed** at `a5474704`: four
were the host-side gates every box tripped (fixed at `a9a158e0`, 80 passed
there), four were the sm_120 HVP rows reading `TESSERA_OPT` /
`TESSERA_LLVM_BIN` out of the environment and raising a bare KeyError in a
sweep pinned only by the device gate — they now discover the tools and skip
by name, and pass by discovery (13 passed at `f93dfabe`); both trees built;
the three new files' device rows and the sphere rows pass at the head.

## General-shape NVFP4 dispatch, and the driver JIT ceiling that hid under it — 2026-09-18

Sync `GFX1201-PARITY-2026-09-17` (branch `claude/typed-route-lds-nvfp4-dispatch`,
shared with the ROCm typed-route work); owner COMPILER-DEVEX-1.

**The NVFP4 tile is an arbiter candidate now, because it has a general shape
and a device timer.** The previous loop left the emitted sm_120a NVFP4 kernel
as one fixed m16n8k64 warp tile with host transfers per call — correct, and
ineligible for promotion by construction. It now emits a general-shape GEMM
(`ptx_emit.emit_nvfp4_gemm_ptx`, entry `TESSERA_NVFP4_GEMM_ENTRY`, with a
structural validator), reaches the device through a launcher entry and a
benchmark entry of its own, and registers with the arbiter through
`register_op_kind(OP_NVFP4_MATMUL, ...)` — the additive seam for an op with its
own numpy reference, so NVFP4 is verified against `nvfp4_gemm_reference` rather
than against a dense-matmul oracle that cannot express block scales. Two
candidates are registered: the emitted kernel at `Tier.EMITTED` **with** a
device timer, and the shipped path at `Tier.HAND_TUNED` whose
`measure_device_latency` returns `None`. That asymmetry is deliberate and is
the recorded lesson from the sm_120 PTX GEMM, where a corpus that excluded the
untimed candidate hid the fastest kernel in the registry: a candidate without a
timer must be visible as untimed, never absent.

Its grid convention differs from the f16 emitter on purpose — `ctaid.y` maps M
and `ctaid.x` maps N — and that is stated where the kernel is emitted, because
a silently transposed grid is a wrong-answer bug, not a slow one.

**Root-caused while doing it: every NVRTC-compiled kernel on The-Super-Bear was
dead, and the error message blamed the hardware.** A probe showed NVRTC 13.4
emitting `.version 9.4` while the loaded driver reports API version 13030 —
CUDA 13.3 — whose JIT accepts PTX 9.3 at most. `cuModuleLoadData` returned 222
(`CUDA_ERROR_UNSUPPORTED_PTX_VERSION`) for both `compute_120` and
`compute_120a`, and the runtime surfaced that as "requires an sm_120a-capable
CUDA device/toolchain" — an accurate-sounding sentence about the *device* for a
failure that was entirely about the toolkit/driver skew. This is the same trap
already recorded for the 09-15 pin bump, reaching a second lane: the toolkit's
PTX ISA number is not the driver's JIT capability. `compileKernel` now retries
`cuModuleLoadData` with a progressively lowered `.version`, so a toolkit ahead
of the driver degrades to the highest ISA the driver will actually take instead
of failing opaquely. 47 NVFP4 tests pass on Super-Bear with zero skips.

**Not claimed.** No NVFP4 performance row is promoted. The candidate is
*eligible* — it has a general shape and a device timer — and a promotion still
needs its corpus rows on the owning box, which remains owed. WSL2 timings do
not promote without bare-metal calibration either way.

**Pre-existing red on this box, now correctly a skip.** A full sweep with
`/usr/lib/llvm-23/bin` on `PATH` produced seven failures in
`test_automatic_ad_public_results.py`. They are **not** branch damage: clean
`main` fails the same seven when a compiler is available (measured 2026-09-18).
The cause is a gate that answers the wrong question —
`require_rocm_hsaco_toolkit` mirrored the runtime's detector, which accepts any
root holding an `ld.lld`, and a plain LLVM install has one. So on a CUDA box
with LLVM on `PATH` the gate admitted eight ROCm packaging tests on a host with
no ROCm at all. The ROCDL target links device bitcode as well as calling lld,
so the gate now also requires `<root>/amdgcn/bitcode/ocml.bc`: present under
both ROCm roots on Princess-Luna, absent everywhere on The-Super-Bear. Those
rows are skips again, with a reason.

## The sm_120 Lion stop-sign lane returns rc=3 on main — 2026-09-17

Sync `ROCM-HOST-RED-ZONE-FOLLOWUPS-2026-09-17`; owner COMPILER-DEVEX-1.

**Follow-up required — owed to this queue, pre-existing.**
`test_autodiff_training_series_target_binding.py::test_nvidia_lion_backward_runs_sm120_stop_sign_package`
fails on Super-Bear with `verified SM120 Lion backward launch failed: SM120
descriptor invoke returned rc=3`. Bisected the only valid way: `main` at
`1b042e22` built from its own sources in a fresh worktree on that box, **both**
trees (`build/` and `build-nvidia-cuda/`, the latter being where the PTX
launcher and `tessera-nvidia-opt` load from), and the lane fails identically.
It is the single remaining failure in that box's full sweep (48 → 1 on this
branch). In `tessera_nvidia_ptx_launch.cpp` a return of 3 is always a CUDA
memory-API failure (`cuMemAlloc` / `cuMemcpyDtoH`) or unlocked staging
pointers — a device-runtime condition or a descriptor sizing the launcher cannot
allocate, not a codegen error; the device is healthy in the same shell (the
sm_120 promotion gate passes). Not diagnosed further here. The rc is opaque by
design of the launcher; the first step is to have `tessera_nvidia_ptx_invoke_v2`
report *which* CUDA call failed and its `CUresult`, so the next reader is not
where this one was.

Also found while bisecting, and fixed: `examples/advanced/power_retention/` could
not compile in a fresh tree with `TESSERA_ENABLE_CUDA=ON` (which puts it in
`all`), and the first error hid two more. Its tablegen outputs landed flat in
the binary dir while its sources include them as `tessera/power/*.inc`; with
that fixed, the dialect source hand-wrote a second `PowerDialect` beside the
generated declaration, the ops source included the generated header without
`GET_OP_CLASSES`, and `Passes.cpp` used the two-argument `PassRegistration`
constructor LLVM removed. All four are repaired and the dialect and passes
link in a fresh worktree of main on Super-Bear. Beneath them a fifth: the CUDA
kernel `src/kernels/cuda/power_attention.cu` does not compile at all (an
undefined helper `compute_phi2_q_to_smem_bf16`, an undefined `s`, a malformed
declaration at line 226). That is scaffold code with no kernel behind it, so it
is not "fixed" by inventing one; the kernel libraries and the runtime that links
them are now `EXCLUDE_FROM_ALL`, and a clean `all` build of both trees succeeds
with them out. The example stays a `scaffold` in the surface manifest (its passes
are empty, its Python entry point prints a placeholder), and it is **owed**: either
a real kernel or retirement to `archive/`. Four layers of rot in a target that
was part of `all` on every CUDA-configured tree says no clean CUDA build of this
repository had succeeded in some time — the lived-in trees on Super-Bear had
never built the example library (no `libTesseraPowerDialect.a` in `build/`), and
so never saw it.

**Root-caused and fixed 2026-09-17 (later the same day), verified on
Super-Bear.** The first step above was taken first: every CUDA driver call in
`tessera_nvidia_ptx_launch.cpp` (218 sites) now records its name and `CUresult`
on failure, the JIT log rides along, and `tessera_nvidia_ptx_last_error()`
hands it to Python, which appends it to the rc. The next run said what a week
of `rc=3` had not:

    cuModuleLoadDataEx: CUDA_ERROR_UNSUPPORTED_PTX_VERSION (222) -- JIT log:
    ptxas application ptx input, line 9; fatal: Unsupported .version 9.4;
    current version is '9.3'

**The 2026-09-15 toolchain bump conflated the toolkit with the driver.** nvcc
13.4.59 emits `.version 9.4`, and `gpu_target.py` pinned that as *the* PTX
ISA — but driver 610.88 answers `cuDriverGetVersion` = **13030 (CUDA 13.3)**,
and a driver's ptxas refuses any `.version` newer than its own before reading
an instruction. Every kernel this lane hands to `cuModuleLoadDataEx` is
JIT-compiled by the driver: the Lion VJP's PTX comes from `nvcc --ptx`
(`nvidia_training.py`), so it carried 9.4 and could not load; the hand-emitted
`ptx_emit.py` kernels copied the same pin. The lane was not a memory failure
and not a descriptor-sizing one — the launcher's rc table simply had one number
for "any device op", which is what the instrumentation exists to end.

Fix, in the compiler where the claim belongs: `gpu_target` now pins the driver
separately (`TESSERA_TARGET_CUDA_DRIVER_API = "13.3"`,
`TESSERA_TARGET_DRIVER_JIT_PTX_ISA = "9.3"`, measured) and derives
`driver_jit_ptx_isa()` from the loaded driver when there is one (Decision #30),
capped at the toolkit ISA; `ptx_emit.py` stamps that; and the runtime's one
registration point (`_register_nvidia_ptx`) re-stamps any PTX handed to the
JIT — nvcc's included — down to it (`ptx_for_driver_jit`, the Triton
precedent), recording the version it lowered from. A body that truly needs a
newer instruction still fails, on that instruction's name. The toolkit pins in
`cmake/` and `AdapterVersionPin.h` are untouched (they describe the toolkit).
`test_nvidia_lion_backward_runs_sm120_stop_sign_package` **passes on Super-Bear**;
the training-series file is 47 passed / 14 skipped. CLAUDE.md's toolchain
paragraph carries the correction.

**`power_retention`: retired to `archive/examples/advanced/power_retention/`
(2026-09-17).** The decision the previous paragraph left open. Its op already
lives in the canonical dialect (`tessera.power_attn` / `tessera.retention`,
LA-4, with the Python surface and `test_linear_attn.py`), the CUDA kernel was
a scaffold that never compiled, and `src/extension` was a torch pybind stub
(Decision #23). The manifest row, the CMake subproject, and every active
reference (`examples/README.md`, `examples/advanced/README.md`,
`PROJECT_STRUCTURE.md`, the porting guide, the API spec, three code comments)
now say where it went; `surface_status.{md,csv}` regenerated. Nothing under
`examples/advanced/` is built by CMake any more.

## Duplicate `gpu.kernel` stamp: the Philox generator had it too — 2026-09-17

Sync `ROCM-HOST-RED-ZONE-FOLLOWUPS-2026-09-17`; owner COMPILER-DEVEX-1.

**Parity change, unverified on the owning device.** `GenerateNVIDIAPhiloxKernel.cpp`
stamped `gpu.kernel` by raw attribute name on a `gpu.func` whose `kernel` is an
inherent property in LLVM 23, exactly as 72 ROCm generators did; the ROCm queue
records the mechanism and the assertions-only abort it produced there. The
NVIDIA site is fixed the same way (`setKernelAttr`). Super-Bear runs an NDEBUG
driver, so the duplicate was silent there and the fix changes no observable
result on that box; the only host that could falsify it is an assertions-ON
build with the NVIDIA backend configured, which none currently is. Owed: build
one, or run the Philox lowering through the ROCm-style single-invocation fixture
pattern (`rocm_generated_kernel_stamps_gpu_kernel_once.mlir`) with NVVM.

**Built, and it found three more (2026-09-17, later the same day).** Tajasarus now
holds `build-assertions-nvidia/` — the assertions-ON LLVM 23.1.1 with
`TESSERA_BUILD_NVIDIA_BACKEND=ON`, CUDA off, the same `-fno-rtti -UNDEBUG`
flags as its ROCm assertions tree — the first assertions-enabled NVIDIA driver
in the fleet. Its first `check-tessera-nvidia` run aborted three fixtures:
`GenerateNVIDIAPhiloxKernel` creates `math` ops without declaring the dialect
("Loading a dialect (math) while in a multi-threaded execution context"),
`LowerTileToNVIDIAPass` loads `tile` from inside `runOnOperation`
(`sm120_macro_cta_matmul`), and `philox_distributions.mlir` expected `math.sin`
before `math.cos` while the driver emitted them the other way round: the
generator passed two nested `builder.create` calls as arguments to a third, and
C++ leaves that evaluation order unspecified — the fixture's order held under
the compiler that built Super-Bear's driver and not under the one that built
this tree. That is a determinism defect in the generator, fixed by creating each
operand in its own statement; the same shape may exist in other generators and
should be read for. The driver also registers `convert-gpu-to-nvvm`,
`convert-scf-to-cf`, `reconcile-unrealized-casts` and the ConvertToLLVM
extensions the NVVM lowering promises (arith, cf, func, index, math, memref,
ub, gpu, nvvm), so the single-invocation stamp fixture
(`philox_stamps_gpu_kernel_once.mlir`) runs there: **62/62**, and
`llvm.func @philox_uniform ... attributes {gpu.kernel, nvvm.kernel}` — once.
Super-Bear rebuilt both trees after these edits: `check-tessera-nvidia` 62/62 in
`build/` and `build-nvidia-cuda/`, 552 Philox/PTX/training-series unit tests
passed, `tests/tessera-ir` 493/493 (NDEBUG, so the declarations change nothing
observable there; the operand-order fix changes IR order only).

## Princess-Luna red zone: the shared fixes that touch this backend — 2026-09-17

Sync `ROCM-HOST-RED-ZONE-2026-09-17`; owner COMPILER-DEVEX-1 with W4-PRODUCT-1.

**Follow-up required, one item.** Clearing 23 pre-existing failures on the gfx1151
box changed three things this backend shares. (a) `libtessera_runtime.a` now
publishes its out-of-CMake link requirements beside the archive, and the CUDA half
is symmetric with the HIP half: a build with `TESSERA_ENABLE_CUDA=ON` contributes
`libcudart` to the sidecar, so the runtime C-ABI harnesses link on a CUDA host too.
**Unverified here** — Super-Bear builds the runtime without CUDA, so its sidecar is
empty and the CUDA path of that sidecar has no owning-device proof yet; that is the
follow-up. (b) `arith.select` gained its transpose and index/integer ops are no
longer refused as non-differentiable, which is target-independent and applies to
the sm_120 AD lanes unchanged. (c) Five tests that hard-coded `nvidia`/`sm_120`
now skip where no CUDA toolchain exists instead of failing inside NVVM
serialization — that was the *other* boxes reporting a missing CUDA toolchain as a
broken compiler, and it changes nothing on a host that has one.

## Row-program math admission, rotor sampling, ragged batches, annealing — 2026-09-16

Sync `EBM-GA-GAPCLOSE-2026-09-16`; owner W4-PRODUCT-1 / AD-SOLVER-IFT-1.

**Follow-up required: this backend has an uncovered instance of the defect that drove the slice.** The row-program emitter now refuses any `math.*` op whose accuracy on the device route has not been measured, and `build_native_gpu_storage` refuses an image whose kernel contains no store — the guard that catches a device-library call whose body the binary serialization silently drops (found on gfx1151 with `math.tanh`: the launch succeeded and wrote nothing). **That guard covers the AMDGPU image only.** The NVIDIA cubin needs `nvdisasm`, which is not a matched-LLVM tool, so an equivalent silent body loss on the NVVM route would still ship unnoticed. Owed here: either a cubin-side equivalent of the store check, or a documented argument that the NVVM serializer fails loudly where ROCDL's does not — the second is plausible (libdevice is linked explicitly) and is exactly the kind of plausible claim this stream keeps disproving by measuring.

Parity validated on sm_120 for what landed: the admitted math set at 16384 points per input domain — `sqrt` and `absf` exact, `cos` 1 ulp, `exp` **3** ulp (one looser than both RDNA parts), `log` 3 ulp. `sqrt` at 0 ulp is the `__nv_fsqrt_rn` pin working; the same sweep against the unpinned `__nv_sqrtf` is what produced the original 1-ulp row. The annealed Langevin chain and the Clifford closed forms run here too: 149 unit on Super-Bear, EBM lit 18/18.

See the [plan log entry](../../compiler/INTEGRATED_COMPILER_LOG.md#2026-09-16--the-math-a-kernel-is-allowed-to-contain-and-four-closed-domain-gaps) and the [device packets](../../../../benchmarks/baselines/row_program_math_precision_20260916/README.md).

## EBM bivector integrator and the overhead measurement — 2026-09-16

Sync `EBM-BIVECTOR-OVERHEAD-2026-09-16`; owner W4-PRODUCT-1 / AD-SOLVER-IFT-1.

Parity validated on sm_120 (46/46 device tests). The overhead packet here carries the native route **alone**: sm_120 has no Python-emitted EBM Langevin lane, and pairing the native GPU route with the x86 CPU one would compare two devices and label it a route comparison, so the recorder refuses that. Native cost is flat in K (1.17–1.24 ms across 1–32 steps and 32–16384 elements). Follow-up required: WSL2 wall clock does not promote — bare-metal calibration is owed, as on every NVIDIA perf row.

See the [plan log entry](../../compiler/INTEGRATED_COMPILER_LOG.md#2026-09-16--the-bivector-integrator-and-the-overhead-measurement-close-the-ebm-stream) and the [overhead packets](../../../../benchmarks/baselines/ebm_langevin_overhead_20260916/README.md).

## EBM nonlinear energies and the sphere integrator — 2026-09-16

Sync `EBM-NONLINEAR-MANIFOLD-2026-09-16`; owner W4-PRODUCT-1 / AD-SOLVER-IFT-1.

Parity validated on sm_120 (Super-Bear), 32/32 device tests on both trees. The sphere retraction's f32 `sqrt` goes through the rounding-explicit libdevice call the last slice introduced, so the retracted state matches the host fold; the three energies each lower inside the kernel with no `custom_adjoint_call`. No performance claim.

See the [plan log entry](../../compiler/INTEGRATED_COMPILER_LOG.md#2026-09-16--nonlinear-energies-and-the-sphere-integrator-reach-the-same-kernel).

## EBM Langevin loop as one cooperative kernel — 2026-09-16

Sync `EBM-NATIVE-GPU-2026-09-16`; owner W4-PRODUCT-1 / AD-SOLVER-IFT-1.

Parity validated on sm_120 (Super-Bear): the loop is one launch through the row-program emitter + native storage package, bit-exact with the declared policy in every packet row (worst abs error 0); the reduction program bit-exact with the sequential fold; row `nvidia_sm120` / `nvidia_ebm_langevin_native_compiled`, 11/11 device tests. No performance claim (per-call host transfers; WSL). `build-nvidia-cuda/` — the tree whose driver the runtime resolves — was configured without the EBM/Clifford backends and is now reconfigured with `TESSERA_BUILD_EBM_BACKEND=ON -DTESSERA_BUILD_CLIFFORD_BACKEND=ON`; the fixture passes under both trees there. **Finding, NVIDIA-specific:** on the NVVM arena route `math.sqrt` lowers to libdevice `__nv_sqrtf`, whose precise branch is gated on `__CUDA_PREC_SQRT`, which MLIR's pipeline never sets (LLVM 23 exposes only `nvvm-reflect-ftz`) — the kernel ran `MUFU.SQRT` and one row of the reduction proof was 1 ulp off; `div` was `div.rn`. The row-program emitter now calls libdevice's rounding-explicit `__nv_fsqrt_rn` (`convert-gpu-to-nvvm` marks the LLVM math intrinsics illegal, so `llvm.intr.sqrt` is not an option on this route). Follow-up required: every other libdevice f32 path any NVIDIA arena package takes (`__nv_rsqrtf`, `__nv_expf`, `__nv_tanhf`…) is subject to the same default; measure before calling any such result exact, or set the reflect defaults at the packager once LLVM exposes them.

See the [plan log entry](../../compiler/INTEGRATED_COMPILER_LOG.md#2026-09-16--the-ebm-langevin-loop-runs-as-one-cooperative-kernel-on-gfx1151-gfx1201-and-sm120) and the [device packets](../../../../benchmarks/baselines/ebm_langevin_native_gpu_20260916/README.md).

## EBM quadratic energy loop through the backbone — 2026-09-16

Sync `EBM-NATIVE-QUADRATIC-2026-09-16`; owner W4-PRODUCT-1 / AD-SOLVER-IFT-1.

Follow-up required: CPU-lane only; no sm_120 EBM lane exists and none is claimed. EBM lit 14/14 on Super-Bear.

See the [plan log entry](../../compiler/INTEGRATED_COMPILER_LOG.md#2026-09-16--the-ebm-quadratic-energy-loop-executes-through-the-mlirllvm-backbone).

### Evidence governance gates (2026-09-27)

Owners: X86-EVIDENCE-VOCAB-1 / GOV-ODS-CONSUMER-1 / EVIDENCE-PACKET-1. Sync: `EVIDENCE-GOVERNANCE-GATES-2026-09-27`.

Parity validated host-free (Mac). The device-clock packet's packet-local tags are declared (witness/window/part codes borrowed from the registry by name) and the validator refuses an undeclared tag; every committed sm_120 packet still validates. The ODS gate waives `tessera_nvidia.{mma_fused,mma_attention,fpquant}` (named only by `sm120_differentiation_target_ir.mlir`) and `tessera_nvidia.func` (fixtures only; the bare `FuncOp` in `PipelineOverlapPass.cpp` is `mlir::func::FuncOp`). No device run, measurement or promotion changed. [Log entry](../../compiler/INTEGRATED_COMPILER_LOG.md#2026-09-27--evidence-governance-gates-reason-vocabularies-ods-consumers-corpus-eligibility).

### EVIDENCE-PACKET-1: shared envelope and GA/EBM route receipts (2026-09-27)

Owner: EVIDENCE-PACKET-1. Sync: `EVIDENCE-PACKET-1-2026-09-27`.

Parity validated host-free (Mac) for the envelope. The sm_120 device-clock packets read through `evidence_envelope.read_evidence_packet`, and SSD admission's `%globaltimer` route reads them through it too. All 36 committed packets read and stay promotable. NVIDIA already refused a missing `worktree_dirty`. Follow-up required: record compiler build identity in the device-clock packet. The envelope records it where present and does not yet require it. Follow-up required: make `profiler_cuda_window` (the Nsight activity-window calibration) an envelope family; it is still read by its own rebuild. Route receipts on The-Super-Bear (clean `eed48b9b`): every GA/EBM composition call ran on the NumPy reference. Zen 2 has no x86 AVX-512 lane, and no composition reaches the sm_120 sphere-Langevin row program, which is a separate entry point. No CUDA GPU lane was reached. No measurement or promotion changed. [Log entry](../../compiler/INTEGRATED_COMPILER_LOG.md#2026-09-27--evidence-packet-1-shared-evidence-envelope-ga-and-ebm-route-receipts).

### E2E-REAL-6 native follow-ups — 2026-09-28

Owner E2E-REAL-6; sync `E2E-REAL-6-NATIVE-FOLLOWUPS-2026-09-28`.
Parity validated for the shared paged-KV Schedule change on Super-Bear's RTX
5070 (sm_120): four exact device remap/invalid-table rows and 25 scheduled
packed-state tests pass. The shared MoE Graph direct-gather subtype has no
sm_120 physical consumer; follow-up required before NVIDIA support is claimed.
Attention with LSE/backward and nvfp4/int4/mx matmul migrations remain open.
No paged-KV performance claim follows from these correctness tests.
## RMSNorm to matmul caller-buffer edge -- 2026-09-29

Owner: W1.1 / NVIDIA fragment-producer closure; sync SM120-RMSNORM-MATMUL-EDGE-2026-09-29.

Parity validated on Super-Bear (RTX 5070, sm_120a) for one named Graph->Schedule->Tile package edge: fp16 tessera.rmsnorm -> fp16 tessera.matmul, fp32 output. The package verifies static shapes, dtypes, layouts, alignment, completion order, and non-aliasing. A caller-owned device allocation is passed unchanged between separately compiled packages on the same CUDA stream; the stream is synchronized before the result lease is returned.

The resident slice passed on three shapes: 64x64x64, 512x256x512, and ragged 255x127x129. Max absolute errors were 0/0.00000334, 0.001953125/0.00001526, and 0.00097656/0.00000668 for producer/consumer. At 512x256x512, C++-loop CUDA-event medians were 59.49 us producer and 12.85 us consumer; CV was 0.03% and 2.44%, matching the checked-in packet. Inputs are uploaded once, with no intermediate host copy. The synchronous host wall includes package/runtime overhead and is not kernel time. No selector promotion or fusion claim. See the [case matrix](../../../../benchmarks/baselines/sm120_rmsnorm_matmul_edge_20260929/README.md) and packets.

Resident CUDA inputs must report the same CUDA Array Interface stream as the explicit launch stream; mismatched, missing, and default/sentinel producer streams fail closed rather than launching without a dependency. Follow-up required: widen to additional dtypes/layouts and dynamic-shape contracts, then migrate the remaining NVIDIA tensor-valued fragment producers. This bounded fp16 slice does not close W1.1.

## Matmul output conversion and resident edge review -- 2026-09-30

Owner: E2E-REAL-6 / W1.1; sync `MATMUL-EPILOGUE-RESIDENT-EDGE-2026-09-30`.

Parity validated on Super-Bear (RTX 5070, sm_120) for the existing unfused fp16/bf16 RMSNorm-to-matmul resident contract. Package construction and descriptor validation now reject consumers with bias, residual, or activation because the resident edge ABI carries only the produced tensor and RHS. The shared eager reference retains fp32 through the epilogue before fp16 conversion. Follow-up required: extend the resident ABI before admitting fused consumers; broader producer migration remains open. No benchmark or route promotion.

## Bounded dynamic-M resident RMSNorm to matmul — 2026-09-30

Owner: E2E-REAL-6 / W1.1; sync `SM120-RMSNORM-MATMUL-DYNAMIC-M-2026-09-30`.

Parity validated on Super-Bear (RTX 5070, sm_120) for Graph -> Schedule -> Tile fp16 and bf16 RMSNorm-to-matmul packages. Active M=7 and 16 execute under bound 16 in exact-device tests; the correctness-gated CUDA-event packet also checks active M=64 and 128 under bound 128. Producer and consumer retain the same resident allocation, stream and package images. Fused bias/residual/activation remains rejected until the ABI carries those operands.

The 31-sample timing packet records clean source revision `1053c284`; medians are 11.76/12.74 us (producer/consumer) at M=64 and 16.42/13.58 us at M=128. CV reaches 25.8%, so timing is diagnostic and does not justify promotion. Dynamic M+N, non-row-major layouts, and additional storage formats remain open. The related gfx1201 shared bounded-M transformation was revalidated on Tajasaurus at PR head 6a84094 with a fresh compiler and 18/18 resident tests; this adds no gfx1201 timing claim. [Packet](../../../../benchmarks/baselines/sm120_rmsnorm_matmul_edge_20260930/README.md).

## Bounded dynamic-K resident RMSNorm → matmul — 2026-09-30

Owner: E2E-REAL-6 / W1.1; cross-backend sync `E2E-REAL-6-RESIDENT-DYNAMIC-K-2026-09-30`.

Parity validated on Super-Bear (RTX 5070, sm_120): fp16 and bf16 exact-device tests trace public `from_text` RMSNorm/matmul Graphs, apply the bounded dynamic-K package contract, and execute K=7, 11, and 16 through Graph → Schedule → Tile. A correctness-gated fp16 packet additionally checks M=128, K bound=256, N=256 at active K=128/192/256. Producer and consumer are timed separately on the same resident allocation and stream; image identity is stable. Producer timing is stable, but consumer CUDA-event CV remains 45–54%, so the timing is diagnostic and supports no speedup claim or route promotion. gfx1201 has its own architecture-specific packet and 20/20 owning tests; no physical schedule or timing is transferred. Dynamic M+K/N+K, broader layouts/storage, and remaining W1.1 producer sites remain open.

[SM120 packet](../../../../benchmarks/baselines/sm120_rmsnorm_matmul_edge_20260930/dynamic_k_sm120.json).


## `E2E-REAL-6-RESIDENT-STRIDED-INGRESS-2026-09-30`: padded host-view ingress on the resident edge

Owner E2E-REAL-6; sync `E2E-REAL-6-RESIDENT-STRIDED-INGRESS-2026-09-30`.
The sm_120 resident RMSNorm → matmul route now accepts padded/sliced host views,
packs them to compact row-major producer and column-major RHS storage before
upload, and holds async upload staging memory until successful stream sync.
Exact RTX 5070 tests cover fp16/bf16, bounded K prefixes, numerical parity,
stable package images and resident intermediate reuse. Producer and consumer
device-event timings exclude packing/upload. No performance promotion.


## `E2E-REAL-6-RESIDENT-DYNAMIC-MK-2026-09-30`: paired bounded dynamic M+K resident edge

Owner E2E-REAL-6 / W1.1; sync `E2E-REAL-6-RESIDENT-DYNAMIC-MK-2026-09-30`.
Parity validated on Super-Bear RTX 5070 (sm_120) for fp16 and bf16 exact-device
Graph-traced RMSNorm → matmul at active (M,K)=(5,7),(11,11),(16,16), plus a
padded-ingress benchmark at (64,128),(128,192),(128,256). The shared Graph
projection emits both dynamic axes and the checked shape bounds; producer and
consumer reuse the same resident allocation and package images. Host packing
and upload are excluded from CUDA-event stage timing. High consumer variation
remains diagnostic; no selector or performance promotion.
[Packet](../../../../benchmarks/baselines/resident_dynamic_mk_20260930/README.md).

## E2E-REAL-6-RESIDENT-DYNAMIC-MNK-2026-09-30: paired bounded dynamic M/N/K resident edge — parity validated

Owner E2E-REAL-6 / W1.1; sync E2E-REAL-6-RESIDENT-DYNAMIC-MNK-2026-09-30.
Parity validated on Super-Bear RTX 5070 (sm_120) for public Graph-traced
RMSNorm to matmul with fp16 and bf16 storage, bounded M/N/K, padded host views,
numerical checks, stable images, and intermediate allocation reuse. The 31-sample fp16 and bf16 packets measure active M/N/K of
(256,256,128), (384,384,192), and (512,512,256) under bound
(512,512,256). Stage medians/CVs are recorded separately; they support
attribution only, with no performance promotion. The standalone CUDA
`strided` uploader rejects noncompact 2-D host views before allocation;
the public resident-program path packs accepted padded views to compact
storage. Physical device pitches remain out of scope. [Packet](../../../../benchmarks/baselines/resident_dynamic_mnk_20260930/README.md).


## W1.1-SM120-SOFTMAX-MATMUL-EDGE-2026-10-01: second typed producer edge

Owner: W1.1. Parity validated for one static fp16/bf16 softmax -> matmul
route on Super-Bear RTX 5070 (sm_120). Public from_text Graph packages traverse
Schedule and Tile, preserve one resident intermediate allocation on one CUDA
stream, and match independent softmax/matmul numerical references. The
descriptor-owned scalar contract feeds the resident softmax launch. Typed
fragment tile.view -> tile.mma -> SM120 MMA is asserted. Fifteen-sample
CUDA-event medians are recorded separately; CV is 4.95%/5.34% and no promotion
is claimed. Generic tensor-valued Tile constructors and broader producer
families remain follow-up required. [Packet](../../../../benchmarks/baselines/nvidia_sm120_w1_typed_tensor_edge_20261001/README.md).

### NVIDIA saved-LSE backward exact route and timing - 2026-10-01

Owner: E2E-REAL-6 / NVIDIA-ATTENTION-LSE-BACKWARD; sync
NVIDIA-ATTENTION-LSE-BACKWARD-2026-10-01. The first bridge-owned-buffer timing
packet was diagnostic (1.089 ms device-event median; 48% host E2E CV) and was
superseded by the caller-buffer resident packet below.

The backward resident-stream gap is now closed for the existing saved-LSE
`f32` descriptor ABI. On Super-Bear RTX 5070, native forward generated LSE in
caller-owned buffers and the resident backward dispatcher consumed it on the
same stream; all gradients matched the independent oracle (max error 1.49e-7).
Nine samples of 20 launches measured 1.088 ms device median (0.12% CV) and
2.374 ms host-array E2E median (3.63% CV). This micro-shape packet supports
attribution only. [Packet](../../../../benchmarks/baselines/nvidia_attention_lse_e2e_20261001/backward_resident_20261001.json).



## W1.1-SM120-GRAPH-SCHEDULE-TILE-PIPELINE-2026-10-01

Owner: W1.1 / NVIDIA fragment-producer closure; sync
`W1.1-SM120-GRAPH-SCHEDULE-TILE-PIPELINE-2026-10-01`.

Parity validated for one named SM120 matmul envelope. The registered
`--tessera-nvidia-pipeline-sm120` now runs PM verification, Graph-to-Schedule,
and Schedule-to-Tile before residual generic lowering. Its lit fixture proves
the Graph matmul becomes pointer-backed `tile.view` inputs and typed
`tile.fragment_pack -> tile.mma -> tile.fragment_unpack`; metadata loss is
accounted for at the whole-module boundary. The exact RTX 5070 fp16
RMSNorm-to-matmul package at 64x128x128 matched independent oracles, reused
the caller-owned intermediate, and emitted NVVM MMA plus PTX `mma.sync`.
Five-sample event timing CV is 6.9%/18.0% for separate producer/consumer
calls and 3.7%/13.3% for resident calls, so the packet is attribution only.
Source inspection of addCUDA13PipelineForSM confirms that sm=120 runs PM
verification, GraphToSchedule, and ScheduleToTile before generic Tile
lowering. Those two patterns match pre-schedule tessera.matmul/add forms and
do not execute on the canonical Graph pipeline. They remain unconverted on
direct legacy Tile input. The exact-device producer edge proves the
registered production route, not that legacy path.
[Packet](../../../../benchmarks/baselines/nvidia_sm120_integrated_pipeline_20261001/README.md).

## `E2E-REAL-6-GFX1201-SCHEDULED-MATMUL-IMAGE-CACHE-2026-10-01` sibling assessment

Not applicable to NVIDIA code generation/runtime: this is a gfx1201-only
ROCm generator selection and HIP image-cache test. It changes no shared
Schedule or CUDA ABI, and no NVIDIA evidence is inferred.


## ROCM-NVFP4-INGEST-1 gfx1201 joint requantization sibling assessment

This is a ROCm-only checkpoint-format converter and HIP package. It changes no shared Graph, Schedule, CUDA ABI, or NVIDIA lowering; no NVIDIA execution evidence is inferred.

### 2026-10-01 W1.1 dynamic-K timing recheck

The exact RTX 5070 dynamic-K softmax-to-matmul recheck passed numerical parity
and native execution at active K=7 (bound K=16). Increasing each sample to
1,000 launches did not stabilize event timings: producer/consumer CV remained
36.8%/37.0%. Treat these measurements as attribution only. The softmax producer
edge is functionally validated; generic Tile tensor-valued constructors remain
open. [Packet](../../../../benchmarks/baselines/nvidia_sm120_w1_typed_tensor_edge_20261001/softmax_matmul_dynamic_k16_active7_recheck_20261001.json).


## W1.1 — LayerNorm producer through the resident SM120 edge

Affine-free last-axis LayerNorm is now admitted alongside RMSNorm and
softmax for shape-preserving fp16/bf16 producer -> matmul programs. Public
from_text Graph packages traverse Schedule/Tile, retain the layernorm norm
policy, and execute on one resident allocation on RTX 5070. Exact-device tests
passed fp16 and bf16; the fp16 256x256x256 packet reports 9.77e-4 / 1.53e-5
producer/consumer max errors, 71.64 / 10.06 us resident CUDA-event medians,
zero spills, and no promotion.

The generic LowerMatmulToTileMMA and LowerKReductionAddToTileMMA
pre-schedule Tile constructors are still not migrated. This expansion is
specific to the canonical scheduled producer API and does not close the full
W1.1 census.

[LayerNorm packet](../../../../benchmarks/baselines/nvidia_sm120_w1_typed_tensor_edge_20261001/layernorm_m256k256n256.json).

## NVIDIA-ATTENTION-LSE-ORACLE-PACKET-2026-10-02: saved-LSE per-shape numerical gate

Owner E2E-REAL-6 / NVIDIA attention LSE; sync NVIDIA-ATTENTION-LSE-ORACLE-PACKET-2026-10-02. The exact-device
checkpoint test passed on Super-Bear (RTX 5070, sm_120). The timing recorder
was strengthened to compare saved/recomputed forward output, saved row LSE,
and dq/dk/dv gradients against an independent float64 scalar oracle before
timing each shape. All three shapes, including ragged Sq/Sk=15/17, passed.
Maximum output/LSE/gradient errors were 3.10e-8 / 2.74e-7 / 9.53e-8. The packet
records the GPU UUID and target nvidia_sm120; five-sample timing distributions
are diagnostic and do not establish a saved-LSE speedup. Both saved-route
packages carry complete Graph/Schedule/Tile digests; recompute controls have
partial provenance and are labeled separately. No IR, ABI, or physical
schedule changed.
[Packet](../../../../benchmarks/baselines/nvidia_attention_lse_e2e_20261001/saved_lse_recheck_20261002.json).

## ROCM-SCHEDULED-ATTENTION-CACHE-RECHECK-2026-10-01 sibling assessment

Not applicable to NVIDIA execution: the recheck exercises ROCm scheduled
attention image reuse on gfx1201 and changes no shared Graph/Schedule/ABI
contract or NVIDIA physical schedule. No NVIDIA execution evidence is
inferred.

## NVIDIA-W1.1-TYPED-MATMUL-EDGE-2026-10-02: collected typed-fragment exact-device gate

Owner W1.1; sync NVIDIA-W1.1-TYPED-MATMUL-EDGE-2026-10-02. The exact SM120 typed scheduled-matmul test was
previously uncollected because its function name began with an underscore. It
is now collected and passes five fp16 shapes on Super-Bear's RTX 5070. Every
case compares descriptor Graph/Schedule/ScheduleIR/Tile digests to the
compiler artifacts, checks nvvm.mma.sync in Target IR, and matches the output
to an fp32 oracle (maximum error 4.77e-7). A source-fingerprinted benchmark
records CUDA-event and end-to-end domains separately; event CV is high for
some shapes and no performance promotion is claimed. The generic direct
pre-schedule tensor-valued Tile constructors remain open.
[Packet](../../../../benchmarks/baselines/nvidia_sm120_w1_typed_tensor_edge_20261001/typed_m16n8_20261002.json).


## NVIDIA-ATTENTION-LSE-BACKWARD — larger SM120 correctness and scaling gap

Owner E2E-REAL-6 / NVIDIA attention backward; sync NVIDIA-LSE-BACKWARD-SM120-SCALE-2026-10-02.

Super-Bear RTX 5070 passed saved and recompute forward outputs, saved row-LSE,
and dq/dk/dv against the float64 oracle at ragged 127x131 and regular 256x256
sequence lengths. Maximum output/LSE/gradient errors at 256 were
5.22e-8/1.15e-6/7.06e-7. Full Graph/Schedule/Tile lineage is present for the
saved packages. This closes a numerical shape gap, but surfaces a severe
operational performance gap: at 256x256, saved/recompute backward CUDA-event
medians were 1221.5/1927.4 ms, compared with 2.843/2.826 ms forward. At
127x131, backward medians were 189.3/308.8 ms. End-to-end medians track the
kernel events, so host overhead does not explain this cost. Source inspection
shows scalar D/Dv reductions nested in per-key loops in the SM120 materializer;
a tiled backward schedule is the next engineering action. No selector promotion.

[Exact RTX 5070 packet and source hashes](../../../../benchmarks/baselines/nvidia_attention_lse_e2e_20261001/saved_lse_extended_recheck_20261002.json).

### NVIDIA paged-KV low-sample diagnostic — 2026-10-02

Super-Bear RTX 5070 reran the compiler-owned Tile-direct and staged CUDA gather
for all three boundary/ragged cases with permuted pages. All numerical checks
passed. The intentionally short run (three samples, ten event repetitions, three
end-to-end repetitions) failed the 4% repeatability gate on all six rows. This
is diagnostic only; the existing higher-sample retained foundation result and
selector disposition remain in force.

[Raw packet and method](../../../../benchmarks/baselines/nvidia_sm120_paged_kv_recheck_20261002/README.md).

## `NVIDIA-ATTENTION-LSE-BACKWARD-2026-10-01`: 2026-10-02 forward timing refresh

A fresh Super-Bear RTX 5070 (sm_120) run covers the same eight full/causal,
fp16/fp32, regular/ragged forward rows in two interleaved cohorts. Every output
passed the independent fp64 oracle before timing (max error 5.96e-8). Device
event medians were 25.3–32.4 us and all eight met 3% stability; six of eight
end-to-end rows met 3%, with two non-causal fp16 rows outside it. This confirms
the native route and separates stable device execution from host dispatch noise;
no speedup or selector promotion follows. See
[packet](../../../../benchmarks/baselines/nvidia_attention_lse_e2e_20261001/attention_lse_followup_20261002.json).

## W1.1-SM120-TYPED-PRODUCER-EDGE-2026-10-01: fresh typed route check

On Super-Bear RTX 5070 (sm_120a), five public Graph -> Schedule -> Tile -> PTX
matmul shapes passed numerical comparison with max error 4.77e-7. Their
nine-sample CUDA-event medians were 10.83–15.81 us; timing variation remains
too high for promotion. A fresh RMSNorm -> matmul resident edge also passed at
16x64x8, with four typed fragment accumulator iterations, same-allocation
handoff on one stream, and max errors 0 / 1.43e-6. Separate resident event
medians were 10.42 / 11.45 us at 7.4% / 8.0% CV. This establishes the named
Schedule/Tile producer edge but leaves the two generic tensor-valued
`LowerMatmulToTileMMA` and `LowerKReductionAddToTileMMA` constructors open.
[Typed matmul packet](../../../../benchmarks/baselines/nvidia_sm120_w1_typed_tensor_edge_20261001/typed_m16n8_followup_20261002.json).
[Resident edge packet](../../../../benchmarks/baselines/nvidia_sm120_w1_typed_tensor_edge_20261001/rmsnorm_matmul_followup_20261002.json).


### NVIDIA attention backward profile — 2026-10-02

The exact RTX 5070 `[1,4,2,128,128,64,64]` saved-LSE backward row passed
saved/recompute and independent-oracle checks. Five resident event samples
measured 183.165 ms median (0.012% CV); end-to-end was 185.285 ms (0.029% CV).
Nsight Compute reports 22.7% achieved occupancy, 34.7% SM throughput, and
near-zero DRAM throughput for the 512-block kernel. The profile is diagnostic
and its 208.9 ms replay duration is excluded from benchmark comparisons. This
confirms the open issue is the scalar nested reduction design in the native
materializer; a tiled backward Schedule/Tile implementation is the next
engineering action. No speedup or route promotion is claimed.

[Recheck packet and profile summary](../../../../benchmarks/baselines/nvidia_attention_lse_e2e_20261001/README.md).

The 2026-10-02 dV-only delta-elision follow-up passes saved/recompute/oracle
checks on the same shape but measures 183.179 ms resident and 185.364 ms
end-to-end, statistically unchanged from baseline. This removes semantically
unused dV work without a whole-kernel speedup. dQ/dK row-stat reuse remains
the next implementation target; see the packet section in the evidence README.



### W1.1 canonical SM120 four-panel producer — 2026-10-02

The active `tessera-nvidia-pipeline-sm120` was tested with a 64x64x256 Graph
matmul. Its Tile IR uses Schedule-owned pointer-backed views, typed fragment
packs, a loop-carried typed accumulator, and unpack; no generic async-copy
producer is emitted. The focused compiler regression passed. The same shape
executed natively on Super-Bear RTX 5070 and passed the fp32 oracle (max error
8.34e-7) with complete Graph/Schedule/Tile/Target/image digests. Five-sample
event and E2E CVs were 15.8% and 13.1%; timings are diagnostic only.
`LowerMatmulToTileMMA` and `LowerKReductionAddToTileMMA` remain in the generic
Tile pass for legacy pipelines; the SM120 named route runs Graph->Schedule->Tile
first and does not invoke `tessera-tiling`'s legacy K-reduction construction.
The production canonical SM120 K-panel edge is parity validated, while
arbitrary tensor lifetimes and other legacy routes remain follow-up work.

[Exact-device packet and compiler regression](../../../../benchmarks/baselines/nvidia_sm120_w1_typed_tensor_edge_20261001/README.md).


## NVIDIA-ATTENTION-LSE-BACKWARD saved-output row-delta correction — 2026-10-02

Owner E2E-REAL-6 / NVIDIA attention backward; sync
NVIDIA-ATTENTION-LSE-BACKWARD-2026-10-01. The prior saved-output attempt
reported a numerical mismatch because the host PTX bridge treated its nine
buffers as bias+LSE, allocated only the bias-sized prefix for O, and shifted
the output roles. The bridge now keys this launch on
tessera_tile_attention_backward_lse_output_* and handles saved O, LSE, and
dQ/dK/dV sizes and ordinals explicitly in host, resident, and event paths.

On the RTX 5070, the expanded small and 128x128 grouped-query cases pass saved
versus recompute and independent-oracle checks. At [1,4,2,128,128,64,64],
maximum gradient errors are 5.59e-9/7.45e-9/2.09e-7; device and end-to-end
medians are 2.378/4.079 ms versus the prior 183.179/185.364 ms packet. This
is a correctness-gated SM120 result, not a selector change. Wider physical
shape/dtype coverage and other backend consumers of the shared checkpoint
carrier remain open.

[Shape-16 packet](../../../../benchmarks/baselines/nvidia_attention_lse_e2e_20261001/attention_backward_saved_output_recheck_20261002.json).
[Shape-128 packet](../../../../benchmarks/baselines/nvidia_attention_lse_e2e_20261001/attention_backward_saved_output_shape128_20261002.json).


The post-rebuild small saved-output checkpoint benchmark passes its numerical
oracles on RTX 5070. Ten-buffer pointer capacity is now consistent across host,
resident, and timing launchers. The 0.08015 ms resident median is stable; the
1.419 ms E2E median has 53.3% CV and is diagnostic. The combined saved-O + bias
+ LSE ten-buffer sub-envelope still needs exact-device correctness coverage.
Packet: `benchmarks/baselines/nvidia_attention_lse_e2e_20261001/attention_backward_saved_output_bridge_refresh_20261002.json`.
<!-- entry-fields:end -->


## W1.1-SM120-FOUR-PANEL-BENCHMARK-2026-10-02

Owner W1.1. The typed-matmul benchmark now includes an exact-device 64x256x64
case and reports K panel count. On RTX 5070, all six rows passed fp32 parity;
the new 16-panel case measured 8.30 us resident events and 0.278 ms E2E, with
7.36% and 4.34% CV. This is canonical-route evidence, not a performance
promotion. The two generic `TileIRLoweringPass` tensor-valued MMA constructors
remain open for legacy input paths.

[Packet](../../../../benchmarks/baselines/nvidia_sm120_w1_typed_tensor_edge_20261001/typed_matmul_four_panel_refresh_20261002.json).


## W1.1-SM120-CANONICAL-PRODUCER-GUARD-2026-10-02

The `nvidia_pipeline_alias.mlir` fixture now asserts that the registered SM120
Graph -> Schedule -> Tile route emits pointer-backed `tile.view` inputs to
`tile.fragment_pack`, a typed `tile.mma`, and `tile.fragment_unpack`, with no
residual `tessera.matmul` or generic `tile.async_copy` tensor producer. The
active Super-Bear `tessera-opt` output passed FileCheck under the SM120 prefix.
This is a regression guard for the canonical route; the two generic
`TileIRLoweringPass` constructors remain open for direct legacy Tile inputs.
[Fixture](../../../../tests/tessera-ir/phase3/cuda13/nvidia_pipeline_alias.mlir).

<!-- entry-fields:end -->

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


## `NVIDIA-LSE-FORWARD-BACKWARD-127X131-2026-10-02`: saved state exact-device recheck

Owner E2E-REAL-6; sync `NVIDIA-LSE-FORWARD-BACKWARD-127X131-2026-10-02`. RTX 5070
(sm_120) passed independent fp64 forward output, saved row-LSE, and backward
gradient checks for saved and recompute variants at 1x4x2x127x131x64x64. Saved
forward CUDA-event / E2E medians were 790.67 us / 2.045 ms; recompute 784.92 us /
4.369 ms. Saved backward was 2.387 ms / 6.350 ms; recompute was 307.780 ms /
315.367 ms. This three-sample run is diagnostic; selector remains recompute and
broader dtype, bias, and shape coverage remains open.
[Packet](../../../../benchmarks/baselines/nvidia_attention_lse_e2e_20261001/attention_lse_127x131_full_recheck_20261002.json).


### W1.1 SM120 legacy tensor producer route guard — 2026-10-02

For sm_120+, `TileIRLoweringPass` now skips the two generic tensor-valued MMA patterns and fails explicitly if a residual `tessera.matmul` reaches the direct Tile pass. Earlier SM targets retain the legacy patterns. The canonical Graph -> Schedule -> Tile route remains the supported native SM120 producer path. Its exact RTX 5070 device suite passed 14 cases, and the focused host-free producer/pipeline tests passed. W1.1 remains open for the broader producer census and any supported noncanonical producer that needs typed-fragment migration.


### W1.1 SM120 static ragged-K typed producer — 2026-10-02

The canonical Schedule-to-Tile route now uses the typed fragment producer for positive static K tails when M is 16-aligned and N is 8-aligned. It passes explicit logical bounds to both `tile.view` inputs; fragment materialization zero-fills K lanes beyond the logical extent, while `tile.memory` retains the exact static leading dimension. The envelope remains fp16/bf16, fp32 accumulation/output, no fused epilogue, and aligned M/N. RTX 5070 exact-device NumPy checks passed for fp16 and bf16 at M/K/N=48/67/16. The host-free structural test confirms both bounded views; the exact-device suite passed seven shape/dtype cases.

The correctness-gated benchmark records separate CUDA-event and end-to-end timings. In the current 11-sample/1000-repetition packet, fp16 K=67 measured 15.75 us event median (33.6% CV) and 0.363 ms end-to-end median (3.8% CV). Repeated packet runs varied materially; timing remains diagnostic and does not support promotion. The generic tensor-valued legacy constructors remain a separate W1.1 obligation; this does not close the full producer census. No shared Tile op, ABI, or sibling physical schedule changed; Apple/ROCm/x86 require no parity inference from this SM120-only producer.

[Packet and methodology](../../../../benchmarks/baselines/nvidia_sm120_w1_typed_tensor_edge_20261001/typed_matmul_ragged_k67_stability_20261002.json).



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

[Packet](../../../../benchmarks/baselines/nvidia_sm120_w1_typed_tensor_edge_20261001/typed_matmul_static_mnk_tails_20261002.json).
[Validation and compiler fingerprints](../../../../benchmarks/baselines/nvidia_sm120_w1_typed_tensor_edge_20261001/static_mnk_validation_20261002.json).
[Final test transcript](../../../../benchmarks/baselines/nvidia_sm120_w1_typed_tensor_edge_20261001/static_mnk_validation_20261002.txt).
<!-- entry-fields:end -->

## ROCM-NATIVE-IDENTITY-2026-10-02 sibling assessment

Owner E2E-REAL-6-ROCM-MATMUL-CACHE / E2E-REAL-6-ROCM-CACHE-KEYS.
ROCm-only Target image projection moves from Python to a registered MLIR pass;
ROCm Target portable bias ordering and output storage are now explicit. The
shared native image and launch descriptor schemas, public operations, dtype
catalog and nvidia canonical pipeline are unchanged.
Sibling physical parity is not applicable: nvidia has its own Target
generation and native launch ABI. No RX 9070 XT cache, numerical or timing
evidence transfers to this backend. Its existing open obligations remain open.
<!-- entry-fields:end -->

## ROCM-NATIVE-MODULE-CACHE-2026-10-02 sibling assessment

Owner E2E-REAL-6-ROCM-MATMUL-CACHE / E2E-REAL-6-ROCM-CACHE-KEYS.
The ROCm-only native HIP module/function service adds bounded leases and explicit
context invalidation. The shared runtime edit is confined to ROCm checked
descriptor submission; nvidia target lowering, private module ownership,
kernel ABI IDs and image schema are unchanged. Physical parity is not
applicable: nvidia uses a different native loader/context contract. Shared
checked-ABI host gates ran; no gfx1151/gfx1201 hardware timings or cache
lifecycle proof transfers. Existing nvidia obligations remain open.
<!-- entry-fields:end -->


## NVIDIA-BIAS-SAVED-CHECKPOINT-2026-10-02 — Graph/Schedule/Tile integration

Owner E2E-REAL-6; sync NVIDIA-BIAS-SAVED-CHECKPOINT-2026-10-02.
Explicit exact-shape f32 bias now passes through the registered native checkpoint
Graph -> Schedule -> Tile route and distinct bias launch ABIs. Forward binds
Q/K/V/bias/O/LSE; backward binds dO/Q/K/V/O/bias/LSE/dQ/dK/dV. Native verification
checks the bias extent before schedule hashing, and replay validates physical
binding order. The native resident forward bridge carries all six buffers.

Exact RTX 5070 (sm_120) host and resident results pass independent fp64 oracles
for three regular/ragged/grouped-query shapes, including batch two. Paired
resident capture privately owns Q/K/V/O/LSE and bias, and repeated backward
survives caller input mutation. This also repairs the preexisting resident tape
eight-buffer contract: current saved-output backward requires nine buffers
before adding bias. Bias participates in checkpoint semantic identity; mixed
bias/plain pairs fail before target compilation.

The complete focused device/replay/pair/lifetime/registry/audit lane passed
473 checks with three environment-gated skips, including the mixed-identity negative case. Benchmark results are correctness-gated and
retain separate CUDA-event and E2E samples. No selector promotion. Automatic
bias AD, bias JVP, broader dtype/layout/score policies and generic tensor-valued
producer migration remain open. Sibling queues record their own follow-ups.

[Packet](../../../../benchmarks/baselines/nvidia_attention_lse_e2e_20261001/attention_bias_saved_output_20261002.json).
<!-- entry-fields:end -->


## W1.1-SM120-LEGACY-SCHEDULE-2026-10-02 — positive native migration of legacy matmul entry

Owner W1.1; sync W1.1-SM120-LEGACY-SCHEDULE-2026-10-02. Original frontend function names
now enter tessera-tile-ir-lowering=sm=120 and produce the canonical native
Schedule/Tile result. Native SymbolTable normalization preserves symbol users;
the named Graph/Schedule passes own storage, policy hashing, typed fragments,
and accumulator lineage. The adapter requires explicit nvidia_sm120/sm_120 and
cannot lower a sibling architecture by accident. There is no new Python
kernel emitter, tensor constructor or direct Target bypass.

Exact RTX 5070 checks prove fp16/bf16 plain static regular/ragged/multi-panel
cases and bounded dynamic fused bias/ReLU/residual reuse at two runtime shapes.
Padded input/output and residual views use the checked leading dimensions; output
padding canaries remain intact. Replay comparisons cover static/dynamic fused
fp16/fp32 outputs in both input dtypes. The old SM80/SM90 Tile fixtures still pass
FileCheck; these are artifact regressions, not older-device execution claims.

The final focused lane passed 386 tests. The twelve-shape correctness-gated
packet compiles the delegated Tile product
and retains its original Graph input digest, Schedule lineage, native image,
compiler/runtime hashes, and separate CUDA-event/E2E samples. Package-build time
includes canonical comparison plus delegated packaging. No performance
promotion follows timing variance.

The canonical-K reduction-step constructor remains open, as do arbitrary
tensor lifetime/producer graphs. Older SM targets retain the generic tensor
constructors; this closes neither their execution nor the full W1.1 census.

[Packet](../../../../benchmarks/baselines/nvidia_sm120_w1_typed_tensor_edge_20261001/legacy_scheduled_producer_20261002.json).
<!-- entry-fields:end -->


## W1.1-SM120-NATIVE-K-SCHEDULE-2026-10-02

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

[Packet](../../../../benchmarks/baselines/nvidia_sm120_w1_typed_tensor_edge_20261001/native_k_scheduled_producer_20261002.json).
<!-- entry-fields:end -->

Validation for W1.1-SM120-NATIVE-K-SCHEDULE-2026-10-02: 508 passed,
50 environment/capability skips, zero failures. Generic T16/T32 tiling,
fused epilogue tiling and SM80/SM90 Tile FileCheck fixtures passed. Packet
source, compiler and native-launch-library hashes match the measured checkout.


## W1.1-SM120-RESIDENT-EPILOGUE-2026-10-02

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

<!-- entry-fields:end -->

Boundary for W1.1-SM120-RESIDENT-EPILOGUE-2026-10-02: fused consumers
currently use the native tile.matmul_kernel carrier. Their explicit typed
fragment epilogue migration remains open; resident execution does not close
that producer census.

Validation for W1.1-SM120-RESIDENT-EPILOGUE-2026-10-02: 625 passed,
49 environment/capability skips, zero failures. Eight isolated exact-device
benchmark rows passed before timing and after device replay. Packet source,
compiler and native-launch-library hashes match the measured checkout.


## W1.1-SM120-TYPED-EPILOGUE-2026-10-02

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

<!-- entry-fields:end -->

Validation for W1.1-SM120-TYPED-EPILOGUE-2026-10-02: 688 passed,
49 environment/capability skips, zero failures. Twenty-four correctness-gated
RTX 5070 rows cover fp16/bf16 input, fp16/fp32 output, static/dynamic M, and
bounds M/K/N=16/16/8, 17/67/23 and 256/256/512. The packet binds compiler,
Target compiler, native launch library and source hashes to this checkout.
Generic tiling, SM80/SM90 and the gfx1151 shared-store verifier fixture pass as
compiler evidence only. Timings are diagnostic; there is no selector promotion
or speedup claim. Fused macro-CTA reuse optimization remains a measured follow-up.

## ROCM-K-UNROLL-IMAGE-CACHE-2026-10-02

Owner E2E-REAL-6-ROCM-MATMUL-CACHE; sync ROCM-K-UNROLL-IMAGE-CACHE-2026-10-02.
Not applicable to nvidia: this change preserves the ROCm register-matmul
K-unroll configuration in native image cache keys. No nvidia image-key,
physical schedule or runtime ABI changed. AMD exact-device evidence does not
establish sibling execution. Shared Schedule/package host tests passed on
Super-Bear; no new nvidia exact-device claim is made.

## ROCM-LDS-IMAGE-CACHE-2026-10-02

Owner E2E-REAL-6-ROCM-MATMUL-CACHE; sync ROCM-LDS-IMAGE-CACHE-2026-10-02.
Not applicable to nvidia: the ROCm projected Target directive selects
its existing typed LDS generator and retains wave geometry in its native image
key. No nvidia schedule, ABI, operation, or capability changes. The shared
package/registry gates were run on Super-Bear; no new nvidia exact-device
claim is made. AMD physical schedules and timing do not transfer.

## ROCM-SPLIT-PARTITION-IMAGE-2026-10-02

Owner ROCM-SPLIT-K-1 / E2E-REAL-6-ROCM-MATMUL-CACHE; sync ROCM-SPLIT-PARTITION-IMAGE-2026-10-02.
Not applicable to nvidia: the added k_blocks/problem_k fields belong
to the ROCm Target dialect and its native WMMA generator. No nvidia ABI,
Schedule producer or capability changes. Shared diagnostic and package gates
passed on Super-Bear. AMD split-K physical scheduling and exact-device evidence
do not establish nvidia parity; existing architecture obligations remain open.

## ROCM-W8A8-TARGET-CONSUMER-2026-10-02

Owner ROCM-FP8-BLOCKSCALE-1 / E2E-REAL-6-ROCM-MATMUL-CACHE; sync ROCM-W8A8-TARGET-CONSUMER-2026-10-02.
Not applicable to nvidia: this slice connects the ROCm gfx1201 W8A8
Target directive to its native typed WMMA image consumer. No nvidia
physical schedule, capability or runtime ABI changes. Host package/registry
gates passed; AMD FP8 numerical evidence does not establish sibling parity.
Existing architecture obligations remain open.

[Packet](../../../../benchmarks/baselines/rocm_w8a8_target_consumer_20261002/README.md).

## ROCM-W8A8-REGISTER-IMAGE-CACHE-2026-10-02

Owner ROCM-FP8-BLOCKSCALE-1 / E2E-REAL-6-ROCM-MATMUL-CACHE; sync ROCM-W8A8-REGISTER-IMAGE-CACHE-2026-10-02.
Not applicable to nvidia: runtime_shape and raster fields belong to the
ROCm scaled Target directive and its gfx1201 typed WMMA consumer. No nvidia
physical schedule, Target admission or runtime ABI changes. Host package and
registry gates passed; AMD numerical and timing evidence does not establish
sibling device parity. Existing architecture obligations remain open.

[Packet](../../../../benchmarks/baselines/rocm_w8a8_register_cache_20261002/README.md).

## ROCM-W8A8-LDS-MN-IMAGE-CACHE-2026-10-02

Owner ROCM-FP8-BLOCKSCALE-1 / E2E-REAL-6-ROCM-MATMUL-CACHE; sync ROCM-W8A8-LDS-MN-IMAGE-CACHE-2026-10-02.
Not applicable to nvidia: runtime_mn and edge-class fields belong to the
ROCm scaled Target directive and its gfx1201 LDS typed consumer. Shared
runtime.py changes validate only ROCm W8A8 descriptors; no nvidia physical
schedule, Target admission or launch ABI changes. Host registry/cache/lifetime
checks passed. AMD machine-code/numerical/timing evidence does not establish
sibling exact-device parity; existing architecture obligations remain open.

[Packet](../../../../benchmarks/baselines/rocm_w8a8_lds_mn_cache_20261002/README.md).

## ROCM-W8A8-LDS-RUNTIME-K-2026-10-02

Owner ROCM-FP8-BLOCKSCALE-1 / E2E-REAL-6-ROCM-MATMUL-CACHE; sync ROCM-W8A8-LDS-RUNTIME-K-2026-10-02.
Not applicable to nvidia: runtime_k is a ROCm scaled Target field and
its gfx1201 LDS consumer owns these loop/staging changes. Shared runtime.py
validation is restricted to ROCm W8A8. No sibling physical schedule, launch ABI
or exact-device parity is claimed; existing architecture obligations remain open.

[Packet](../../../../benchmarks/baselines/rocm_w8a8_lds_runtime_k_cache_20261002/README.md).

## ROCM-FOLDED-NATIVE-EPILOGUE-2026-10-02

Owner ROCM-MXFP4-W4A8-1; sync ROCM-FOLDED-NATIVE-EPILOGUE-2026-10-02.
Follow-up required if nvidia admits the new internal
tile.fragment_folded_scale operation: it shares a verified full-K f32/token
f32/E8M0-byte numerical contract, but no nvidia physical consumer,
frontend route or execution capability is added. Existing nvidia routes
are unchanged. Host registry gates passed; gfx1201 proof is not sibling parity.

[Native epilogue proof](../../../../benchmarks/baselines/rocm_folded_native_epilogue_20261002/README.md).


## ROCM-FOLDED-NATIVE-PACKAGE-2026-10-02

Owner ROCM-MXFP4-W4A8-1; sync ROCM-FOLDED-NATIVE-PACKAGE-2026-10-02.
Follow-up required only if nvidia admits the internal folded-scale
operation: this slice adds its gfx1201 LDS producer and explicit physical ABI
presentation. Runtime and benchmark changes are guarded to the ROCm folded
package; no nvidia physical schedule or execution capability is added.
No sibling exact-device parity follows from the gfx1201 packet.

[Native package proof and paired timing](../../../../benchmarks/baselines/rocm_folded_native_package_20261002/README.md).


## ROCM-FOLDED-NATIVE-OPT-2026-10-02

Owner ROCM-MXFP4-W4A8-1; sync ROCM-FOLDED-NATIVE-OPT-2026-10-02.
Not applicable to nvidia physical schedules: the experiments and
stronger package tests concern only the gfx1201 folded producer. No shared
Tile numerical contract or sibling execution capability changed. Follow-up
required only if nvidia admits the internal folded-scale operation;
gfx1201 timing and fallback proof are not sibling parity.

[Attribution experiments](../../../../benchmarks/baselines/rocm_folded_native_fragment_retirement_20261002/README.md).


## ROCM-FOLDED-RUNTIME-MN-2026-10-02

Owner ROCM-MXFP4-W4A8-1 / E2E-REAL-6-ROCM-MATMUL-CACHE; sync ROCM-FOLDED-RUNTIME-MN-2026-10-02.
Not applicable to nvidia physical schedules: folded runtime M/N
projection and image edge classes belong to the gfx1201 Target consumer.
Shared runtime changes are guarded by that native folded ABI. Native pass and
diagnostic registries were updated together; no nvidia Target, dtype,
operation capability or execution parity is added. Existing sibling work
remains open. Follow-up required only if the internal folded op is admitted.

[Native image reuse and paired timing](../../../../benchmarks/baselines/rocm_folded_native_runtime_mn_20261002/README.md).


## ROCM-FOLDED-RUNTIME-K-2026-10-02

Owner ROCM-MXFP4-W4A8-1 / E2E-REAL-6-ROCM-MATMUL-CACHE; sync ROCM-FOLDED-RUNTIME-K-2026-10-02.
Not applicable to nvidia physical schedules: the runtime full-K64 folded
contract is admitted only by the gfx1201 Target producer. Shared launcher
validation now checks buffer capacities before device probing, guarded to the
ROCm MXFP4 ABI. Native pass metadata was updated with the extended identity
option. No nvidia Target, dtype, operation or execution capability is added.
Existing sibling obligations remain open; gfx1201 proof is not sibling parity.

[Native runtime-K image reuse and paired timing](../../../../benchmarks/baselines/rocm_folded_native_runtime_k_20261002/README.md).


## ROCM-FOLDED-LIVENESS-2026-10-02

Owner ROCM-MXFP4-W4A8-1; sync ROCM-FOLDED-LIVENESS-2026-10-02.
Not applicable to nvidia physical schedules: the new recorder observes
gfx1201 native LLVM/ROCDL liveness and compares the selected HSACO instructions.
The measured-negative folded terminal-prefetch peel was removed from active
code. No shared Tile numerical, runtime ABI, dtype/op or sibling execution
contract changed. Existing nvidia obligations remain open; no sibling
parity follows from the ROCm packet.

[Instruction-bound liveness and rejected candidate](../../../../benchmarks/baselines/rocm_folded_terminal_prefetch_20261002/README.md).


## ROCM-FOLDED-COLD-BRANCH-2026-10-02

Owner ROCM-MXFP4-W4A8-1; sync ROCM-FOLDED-COLD-BRANCH-2026-10-02.
Not applicable to nvidia physical schedules: this slice changes the
internal ROCm folded consumer's LLVM branch likelihood, preserving its numeric
and ABI contract, and adds an independently compiled native benchmark reference
arm. No nvidia operation/dtype/Target or execution capability is added.
Follow-up required only if the internal folded operation gains a sibling
consumer; existing nvidia obligations remain open. Gfx1201 timing and
instruction-bound pressure evidence do not establish sibling parity.

[Retained native branch hint](../../../../benchmarks/baselines/rocm_folded_cold_branch_20261002/README.md), [removed K16 group candidate](../../../../benchmarks/baselines/rocm_folded_panel_group_20261002/README.md).


## ROCM-FOLDED-GRAPH-WINDOWS-2026-10-02: checked package timing

Owner ROCM-MXFP4-W4A8-1; sync ROCM-FOLDED-GRAPH-WINDOWS-2026-10-02.
Not applicable: this is a benchmark-only gfx1201 HIP graph adapter using
existing package descriptors. nvidia runtime/IR/ABI are unchanged and have
no exact-device parity claim from RX 9070 XT timings.
Evidence: benchmarks/baselines/rocm_folded_graph_windows_20261002/README.md.


## ROCM-FOLDED-READ-BARRIER-2026-10-02: measured rejection

Owner ROCM-MXFP4-W4A8-1; sync ROCM-FOLDED-READ-BARRIER-2026-10-02.
Not applicable: the candidate changed only the gfx1201 folded native producer
and was removed. No nvidia IR/ABI/runtime change or execution proof follows.
Evidence: benchmarks/baselines/rocm_folded_read_barrier_20261002/README.md.


## NVIDIA-W1.1-CANONICAL-TENSOR-REPLAY-2026-10-02: producer migration

Owner W1.1; sync NVIDIA-W1.1-CANONICAL-TENSOR-REPLAY-2026-10-02.
Parity validated on RTX 5070 sm_120 for replay-equivalent static canonical
tensor M/N/K reductions at fp16/BF16, plain and bias/ReLU/residual. Whole
function replay proves accumulator, padding, bounds and return lineage before
native Schedule/Tile storage and typed fragments. 65 producer tests and twelve
oracle-gated timing rows pass. Follow-up required: arbitrary tensor producers,
noncanonical accumulators, dynamic generic recovery and older-target proof.
Evidence: benchmarks/baselines/nvidia_sm120_canonical_tensor_replay_20261002/README.md.


## NVIDIA-W1.1-REGISTERED-TENSOR-2026-10-02: complete contraction boundary

Owner W1.1; sync NVIDIA-W1.1-REGISTERED-TENSOR-2026-10-02.
Parity validated on RTX 5070 for twelve static fp16/bf16 plain and bias/ReLU/residual canonical tensor reductions through the registered pipeline. Early verified recovery preserves dimensions, epilogue ABI and executable PTX. Arbitrary/dynamic generic tensor reconstruction remains open.
Evidence: benchmarks/baselines/nvidia_sm120_registered_tensor_pipeline_20261002/README.md.


## NVIDIA-JIT-ATTENTION-VJP-2026-10-02: frontend reverse integration

Owner E2E-REAL-6 / AD-RESIDUAL-EVAL-1; sync NVIDIA-JIT-ATTENTION-VJP-2026-10-02.
Parity validated on RTX 5070 for eighteen JIT-generated reverse attention rows: full/causal, regular/ragged, grouped query, batch two and D != Dv; requested Q/K/V order, saved O/LSE generation and repeated backward passed. Compilation/capture/backward wall and forward/backward device windows are separate. Bias derivatives and general composed AD remain open.
Evidence: benchmarks/baselines/nvidia_jit_attention_vjp_20261002/README.md.


## NVIDIA-ATTENTION-BIAS-GRADIENT-2026-10-02: optional checkpoint bias gradient foundation

Owner E2E-REAL-6 / AD-HIGHER-1; sync NVIDIA-ATTENTION-BIAS-GRADIENT-2026-10-02.
Follow-up required: the internal checkpoint backward op now carries an optional
fourth result with exact [B,Hq,Sq,Sk] f32 bias shape. Native Schedule replay
preserves its result binding and emits eleven pointers plus seven dimension
scalars; Tile/NVIDIA lowering assigns each bias element one deterministic
writer and omits the Q.K scale from the bias derivative. The existing three-
gradient contract remains unchanged. Compiler regression validation is recorded
in the engineering log; the explicit checked runtime ABI and exact RTX 5070 package execution now pass
six full/causal GQA and ragged rows with all four gradients checked against an
independent float64 oracle (max absolute error 1.44e-07). NaN-poisoned
host/resident outputs prove complete stores. Separate resident event windows
and host end-to-end samples are recorded; no speedup claim. Bounded public JIT paired AD now executes the exact-shaped f32 bias gradient
on RTX 5070: eighteen rows preserve requested selection/order and private
Q/K/V/bias/O/LSE across caller mutation and changed cotangents (max error
1.62e-07). A Q-only biased request also passes on device. 441 focused tests
cover rollback and launch ranges. Broadcast-bias reduction, bias JVP, wider
dtypes/layouts and general composed AD remain open. This proves the explicit
native reverse compile API, not universal JIT dispatch.

Evidence: [checked bias-gradient packet](../../../../benchmarks/baselines/nvidia_checkpoint_bias_gradient_20261002/README.md).

Evidence: [public JIT bias VJP packet](../../../../benchmarks/baselines/nvidia_jit_attention_bias_vjp_20261002/README.md).


## NVIDIA-ATTENTION-ARGUMENT-ORDER-2026-10-02: native frontend role mapping

Owner E2E-REAL-6 / AD-HIGHER-1; sync NVIDIA-ATTENTION-ARGUMENT-ORDER-2026-10-02.
Parity validated on RTX 5070 for 28 isolated attention VJP rows: canonical
and permuted biased/unbiased frontend arguments, full/causal GQA, batch two,
ragged sequences and independent value width. Native export derives and
Schedule-hashes the argument permutation; capture follows frontend order and
backward maps requested indices to native gradient roles. Maximum gradient
error is 2.56e-07; 452 focused tests verify mutated/invalid mappings and
lifetime/arity checks. Capture and backward wall timings remain separate from
kernel-event evidence. Aliasing/composed graphs, broadcast bias and permuted
JVP/higher derivatives remain open; no universal JIT or speedup claim.

Evidence: [argument-order packet](../../../../benchmarks/baselines/nvidia_attention_argument_order_20261002/README.md).


## ROCM-MXFP8-K64-2026-10-06: native staging candidate

Owner ROCM-FP8-BLOCKSCALE-1; sync ROCM-MXFP8-K64-2026-10-06.
Not applicable to nvidia physical lowering: this gfx1201 bounded selector uses RDNA4 FP8 WMMA and ROCm LDS. Shared semantic scale/ABI interfaces are unchanged; no sibling physical proof is claimed.

Evidence: [K64 candidate packet](../../../../benchmarks/baselines/rocm_mxfp8_k64_20261006/README.md).


## NVIDIA-W11-OPERAND-LINEAGE-2026-10-06

Owner W1.1; sibling FRONTEND-IR-MEDIUM-1.
Parity validated on RTX5070: 12 permuted plain/fused FP16/BF16 rows, full-pipeline PTX parity and 373 producer/registry regressions. General composed producer/attention/AD and dynamic envelopes remain open.

[Evidence](../../../../benchmarks/baselines/nvidia_operand_lineage_20261006/README.md).


## NVIDIA-W11-PUBLIC-PACKAGE-2026-10-06

Owner W1.1 / FRONTEND-IR-MEDIUM-1; sync NVIDIA-W11-PUBLIC-PACKAGE-2026-10-06.
Parity validated: static host-array @jit now executes the canonical native descriptor for permuted FP16/BF16 matmuls, compact C/F RHS, bias/ReLU/residual, and final fp16/fp32 output. Twelve RTX5070 rows pass fp64 comparison, serialized replay parity, cached-call no-eager/no-recompile checks and changed-value rebinding. Dynamic and arbitrary composed Graph/AD routes remain open.

[Evidence](../../../../benchmarks/baselines/nvidia_operand_lineage_20261006/README.md).


## NVIDIA-PREPARED-MATMUL-2026-10-06

Owner W1.1 / FRONTEND-IR-MEDIUM-1; sync NVIDIA-PREPARED-MATMUL-2026-10-06.
Parity validated on RTX5070: 75 device/guard cases, 651 focused regressions (66 skipped), four fresh-process A/B packets with identical images. Native module/context ownership and shared synchronous scratch remove repeat trace/compile/portable restoration for eight warmed signatures. Median per-case prepared/control public wall ratios are 0.405027/0.395491; this measures host wall. General A layouts, composed/dynamic/AD and resident/async consumers remain open.

[Evidence](../../../../benchmarks/baselines/nvidia_prepared_matmul_20261006/README.md).

## NVIDIA-LHS-RHS-LAYOUT-2026-10-06

Owner W1.1 / FRONTEND-IR-MEDIUM-1; sync NVIDIA-LHS-RHS-LAYOUT-2026-10-06.
The frontend records compact RHS storage facts on a copied semantic Graph.
Native Schedule/Tile remains the physical recipe authority. The named static
RMSNorm/LayerNorm/softmax producer edge now preserves row/column RHS bindings,
ABI identity, epilogue and portable replay rather than forcing column storage.
Parity validated on RTX5070: 70 device tests prove C/F RHS ordinary JIT, sealed cache switching, portable replay and pre-allocation ABI/layout guards. Four independent layout packets, 676 focused host regressions (66 skipped) and four fresh compiler-disabled replays pass. Separate producer/consumer event and public wall packets are recorded.
General producer graphs, dynamic row-RHS, A layouts, composed AD and
resident/asynchronous native ownership remain open. No new dtype/op/pass or
physical strategy promotion is claimed.
Evidence: benchmarks/baselines/nvidia_lhs_rhs_layout_20261006/README.md

## NVIDIA-PREPARED-LHS-2026-10-06

Owner W1.1 / FRONTEND-IR-MEDIUM-1; sync NVIDIA-PREPARED-LHS-2026-10-06.
Shared contracts: native two-package ownership, one-time producer attachment,
context-shared pinned/device staging, failure retirement, public close/rebind,
and truthful ordered-product CompileReport fingerprints.
Native Graph/Schedule/Tile/Target/LLVM component images remain authoritative.
Parity validated on RTX5070: 182 owning-device cases in 921 focused checks (67 skipped). Eight identical-image packets show about 89% lower warm public-call overhead, with separate producer/consumer event windows.
General composition/AD, dynamic layouts, asynchronous/resident ownership,
wider formats and common portable native ownership remain open.
No physical kernel/dtype strategy is promoted.
Evidence: benchmarks/baselines/nvidia_prepared_lhs_20261006/README.md

## NVIDIA-PORTABLE-LHS-2026-10-06

Owner W1.1 / FRONTEND-IR-MEDIUM-1; sync NVIDIA-PORTABLE-LHS-2026-10-06.
Shared contracts: type-strict portable admission witnesses, parent Graph/ABI
guards, bounded per-context native replay ownership, serialized invocation/
retirement and pre-lock fork refusal. Native component images remain authoritative.
Parity validated on RTX5070: static named host-array portable replay retains two native modules and bounded context-specific owners. 498 focused checks pass, including 142 device cases. Matched replay wall packets are recorded in the linked evidence.
General composition/AD, dynamic layouts, asynchronous/resident ownership and
wider formats remain open. No kernel or dtype strategy is promoted.
Evidence: benchmarks/baselines/nvidia_portable_lhs_owner_20261006/README.md

## NVIDIA-DYNAMIC-LHS-2026-10-06

Owner W1.1 / FRONTEND-IR-MEDIUM-1; sync NVIDIA-DYNAMIC-LHS-2026-10-06.
Shared contracts: copied full dynamic Graph certificates, explicit traced
capacities, frontend epilogue role mapping, portable identity admission and
pre-allocation bound guards. Native Graph/Schedule/Tile/Target/LLVM packages
remain execution authority. Producer images are capacity specializations;
the consumer retains its existing checked dynamic strided ABI.
Exact RTX5070 parity covers all seven bounded M/N/K axis subsets for three producers, FP16/BF16 and plain/fused epilogues. Full validation and component timing records are linked below.
Ordinary JIT bounded-shape selection, dynamic row-RHS/native prepared owners,
general composed AD, asynchronous ownership and wider formats remain open.
Evidence: benchmarks/baselines/nvidia_dynamic_lhs_frontend_20261006/README.md

## NVIDIA-DYNAMIC-LHS-OWNER-2026-10-06

Owner W1.1 / FRONTEND-IR-MEDIUM-1; sync NVIDIA-DYNAMIC-LHS-OWNER-2026-10-06.
Shared contracts: immutable bounded/exact axes, dynamic native frame projection,
private C-ABI parameter validation, synchronous grow-only scratch leasing and
portable typed/context ownership. Existing native component images and runtime
launch descriptor ABIs remain authoritative.
Exact RTX5070 parity validated: 688 focused checks including 332 device cases, immutable capacities across all seven axis subsets, native parameter ABI checks, concurrent/failure retirement, and portable context-specific reuse. Matched wall/dispatch attribution is linked below.
Ordinary JIT bounded selection, dynamic row-RHS, general composition/AD,
asynchronous/resident owners and wider formats remain open.
Evidence: benchmarks/baselines/nvidia_dynamic_lhs_owner_20261006/README.md

## ROCM-MXFP8-LONG-K-2026-10-06 — native selector trial

Owner ROCM-FP8-BLOCKSCALE-1 / E2E-REAL-6. Native Schedule selection carries existing K32 scale semantics into an LDS K64 recipe within the measured occupancy envelope.

Not applicable to nvidia physical execution: the recipe gate is gfx1201 E4M3/E8M0-only. No sibling schedule or exact-device evidence is transferred. Shared Graph, ABI, dtype and operation contracts are unchanged.

Evidence: benchmarks/baselines/rocm_mxfp8_long_k_20261006/README.md. WSL contract gates: 115 passed; registry/audit gates: 333 passed.

## NVIDIA-NVFP4-SHARED-TRANSPOSE-2026-10-06

Owner W1.1 / FRONTEND-IR-MEDIUM-1 / E2E-REAL-6. Native batch-aware shared-RHS transposed-A integration retains the existing ten-argument NVFP4 ABI, seals policy/dimensions in Schedule, and independently offsets A/scales/output while preserving shared B/scales.

Exact RTX 5070 parity validated: 30 numerical JIT/vmap cases within 133 focused checks; 24 matched orientation benchmark rows with separate event/public timing. General/dynamic/composed batching and AD linear-transpose closure remain open.

Evidence: benchmarks/baselines/compiler_contract_revalidation_20261006/README.md. Registry/audit gates: 333 passed.

## Public independent-prefix primal/JVP integration

Owner FRONTEND-IR-MEDIUM-1 / AD-RESIDUAL-EVAL-1 / ROCM-FP8-BLOCKSCALE-1;
sync INDEPENDENT-SCALE-BATCH-2026-10-07. Public leading maps preserve independent
matrix/scale prefixes. The primal descriptor binds its actual native member
image and geometry. 487 focused WSL host checks pass.
Follow-up required for nvidia-owned independent-prefix scale primal/JVP execution and exact-device parity. The gfx1201 WMMA, HIP owner and event evidence do not prove sibling physical execution.
Transposed A, partial groups, general/dynamic/composed/storage AD, generic closure and publication remain open.
Evidence: benchmarks/baselines/rocm_public_independent_scaled_primal_20261007/README.md.

## Independent-prefix transposed-A — named profile proved

Owner FRONTEND-IR-MEDIUM-1 / AD-RESIDUAL-EVAL-1 / ROCM-FP8-BLOCKSCALE-1;
sync INDEPENDENT-SCALE-BATCH-2026-10-07. Native Schedule/Tile/Target orientation,
column-major A tile.view and serialized orientation now execute through public
leading maps. 855 focused host checks and 80 final native package/corruption
checks pass. The initial owning-device lane-address error was repaired and
the final compiler rerun; failed-build timings are not evidence.
Sibling SM120 NVFP4 orientation/JIT parity validated: 55 cases pass and two unsupported cases skip. This does not prove NVIDIA independent FP8 scale primal/JVP execution.
Partial groups, dynamic/nonleading/composed/storage AD, optimized LDS A storage, generic closure, full-suite and publication remain open.
Evidence: benchmarks/baselines/rocm_independent_transposed_a_20261007/README.md.

## Independent-prefix partial K32 groups — named profile proved

Owner FRONTEND-IR-MEDIUM-1 / AD-RESIDUAL-EVAL-1 / ROCM-FP8-BLOCKSCALE-1;
sync INDEPENDENT-SCALE-BATCH-2026-10-07. Native Schedule/Tile/Target derives
ceiling scale counts and bounds the last WMMA group, preserving isolated
partial accumulation and static plane/image identities. Direct Tile JVP
admission was repaired and the final compiler rerun. 120 projection, 76 native
partial/corruption and 308 final registry/event/audit checks pass.
Aligned native/sibling SM120 NVFP4 parity is validated in the final 135-pass/two-skip lane. Independent FP8 partial-scale execution on NVIDIA still requires its own physical implementation and proof.
Scalar JIT, other group widths, LDS tails, general/dynamic/nonleading/composed/storage AD, generic closure, full-suite and publication remain open.
Evidence: benchmarks/baselines/rocm_independent_partial_groups_20261007/README.md.


## Typed primal single-image — named profiles proved
Owner FRONTEND-IR-MEDIUM-1 / ROCM-FP8-BLOCKSCALE-1;
sync TYPED-PRIMAL-SINGLE-IMAGE-2026-10-07. The descriptor and native owner now
bind the same actual Graph-derived image for scalar and coupled FP8/MXFP8.
388 focused host checks, 365 artifact/registry/audit checks and 309 exact
gfx1201 device regressions pass. Empty-cache packaging uses 7 rather than
11 subprocesses, median control/candidate ratio 1.759. Reused-cache median
ratio 0.901 retains remaining fingerprint/metadata overhead; no universal
performance or execution speedup is claimed. Scalar transposed-A/partial
groups, general AD/batching, full-suite and publication remain open.
Evidence: benchmarks/baselines/rocm_typed_primal_single_image_20261007/README.md.
Not applicable to this backend's image construction; independent-scale physical parity remains follow-up required.


## Scalar typed bounded planes — named profile proved
Owner FRONTEND-IR-MEDIUM-1 / ROCM-FP8-BLOCKSCALE-1;
sync SCALAR-SCALED-PLANE-2026-10-07. Native Schedule/export derives empty-prefix
physical planes while the rank-two semantic Graph remains unmodified.
95 matching native checks and 40 exact gfx1201 primal/JVP cases pass.
All 48 benchmark rows pass independent numerics, compiler-free changed-input
replay and stale-generation refusal; maximum error is 7.2621e-8.
Scalar transposed-A and partial K32 groups are now proved. Other widths/layouts,
dynamic/nonleading/composed/storage AD, generic closure and delivery remain open.
Evidence: benchmarks/baselines/rocm_scalar_scaled_plane_20261007/README.md.
Follow-up required for this backend's independent-scale execution; no gfx1201 physical evidence transfers.


## Scalar scale reverse — named profile proved
Owner FRONTEND-IR-MEDIUM-1 / AD-RESIDUAL-EVAL-1;
sync SCALAR-SCALED-PLANE-2026-10-07. Scalar transposed A/B and ragged K4
scale reverse now have 12 exact gfx1201 role-order/changed-input checks and
four separate public/native timing profiles; maximum error is 3.43592e-8.
Matching existing SM120 NVFP4 regression passes 83 cases with two unsupported
skips. General composed/dynamic/nonleading/storage AD, sibling scale execution,
generic closure, fresh full-suite and publication remain open.
Evidence: benchmarks/baselines/rocm_scalar_scaled_plane_20261007/README.md.
Follow-up required for this backend's scalar scale reverse physical execution; SM120 NVFP4 regression is separate evidence.


## ROCm version-query metadata reuse — gfx1201 measured
Owner FRONTEND-IR-MEDIUM-1 / E2E-REAL-6;
sync ROCM-VERSION-METADATA-2026-10-07. 49 host checks and 40 owning gfx1201
scalar regressions pass. 16 actual package A/B profiles preserve compiler/
toolchain fingerprints, reduce subprocesses from 7 to 5 and measure median
uncached/cached ratio 1.1205. This is package metadata work, not kernel speed.
gfx1151/other-family proof, broader integration, full-suite and delivery remain open.
Evidence: benchmarks/baselines/rocm_version_query_cache_20261007/README.md.
Not applicable to this backend's compiler/driver querying; only ROCm metadata reuse changes.


## Current five-slice native snapshot revalidation
Owner ROCM-NVFP4-INGEST-1 / W1.1 / E2E-REAL-6;
sync FIVE-SLICE-CURRENT-SNAPSHOT-2026-10-07. Matching compiler revalidation
passes 171 combined NVIDIA host/device tensor and saved-LSE attention tests.
gfx1201 ingest/public resident gates pass 47 cases; the missing native-image
library binding is repaired and the fresh-process compiler-free replay passes.
Original failure and repair receipts remain separate. Wider integration,
performance closure, full-suite and reviewable publication remain open.
Evidence: benchmarks/baselines/five_slice_current_snapshot_20261007/README.md.
Owning architecture evidence is limited to the named cases above; sibling physical proof does not transfer.


## SM120 actual-Graph native partition — owning execution proof

Owner W1.1 / FRONTEND-IR-MEDIUM-1 / E2E-REAL-6.
Sync SM120-NATIVE-TENSOR-PROGRAM-2026-10-08.
Native outlining preserves the actual producer-to-matmul SSA edge, reordered
external roles and compiler-authored buffer capacities/read-write lifetimes.
The raw Schedule adapter does not reconstruct Python Graph operations.
32 host/device checks pass on RTX 5070 SM120, including 12 resident numerical
cases with Python Graph constructors forbidden after frontend serialization.
24 benchmark profiles cover RMSNorm/LayerNorm/last-axis softmax, FP16/BF16,
row/column-major RHS and small/larger static shapes. Independent float64
oracles round the producer result to its storage type before matmul.
Separate producer/consumer CUDA-event dispatch windows and end-to-end timings
are recorded; maximum output error is 0.00107674. No speedup or kernel-only claim.
Compiler SHA256: 08165455e4bf6a5babce1680cabf0fdabe006a767ace49ebce6ef12a6b9f620d.
Public JIT wiring, portable native witness validation, dynamic capacities,
broader composition/asynchronous lifetimes and focused PR delivery remain open.
Evidence: benchmarks/baselines/nvidia_native_tensor_partition_20261008/README.md.

Owning SM120 parity validated in the named static raw-adapter envelope.


## SM120 native whole-Graph public integration — owning proof

Owner W1.1 / FRONTEND-IR-MEDIUM-1 / E2E-REAL-6.
Sync SM120-NATIVE-TENSOR-PROGRAM-2026-10-08.
Static public JIT now packages the actual frontend Graph through native member
outlining and raw native Schedule/Tile artifacts. Prepared C++ execution reads
checked descriptor bindings without reconstructing a consumer Graph.
Portable v2 replay binds both member Graph digests and the native program digest
to its packages, validates buffer byte capacities/ownership/read-write lifetimes,
and uses no compiler subprocess or Python Graph constructor in the proved route.
The existing public/device/resident compatibility lane passes 155 checks;
the independent native/portable/audit/registry lane passes 358 checks, including
seven portable Graph/lifetime corruption cases refused before compiler/CUDA.
48 public JIT/replay benchmark profiles cover RMSNorm/LayerNorm/softmax,
FP16/BF16, both RHS storage orders, fused epilogues and two static shapes.
Each profile checks independent numerical results before and after timing;
producer/consumer event-dispatch and cold/warm/portable wall time are separate.
No performance promotion or kernel-only claim is made.
Bounded dynamic projection still uses the prior Python path. Native dynamic
integration, wider composition/asynchronous lifetimes and focused delivery
remain open. The raw-adapter snapshot remains historical evidence.
Evidence: benchmarks/baselines/nvidia_native_tensor_partition_20261008/README.md.

Owning RTX 5070 static public route parity validated; broader W1.1 remains open.


## SM120 native bounded whole-Graph projection — owning checks

Owner W1.1 / FRONTEND-IR-MEDIUM-1 / E2E-REAL-6.
Sync SM120-NATIVE-BOUNDED-TENSOR-PROGRAM-2026-10-08.
C++ MLIR outlining now projects bounded M/N/K from the actual frontend Graph;
the public bounded path no longer constructs replacement Python Graphs.
Original caller IR is preserved. Native v2 witnesses bind active shapes,
capacities, member Graphs and private buffer read/write lifetimes.
All seven nonempty dynamic-axis subsets are covered for RMSNorm/LayerNorm/
softmax and FP16/BF16, including compiler-free portable replay and stable
scratch ownership: 130 focused native/host checks pass on Super-Bear.
The wider device lane reports 291 passes and one legacy-name fixture failure.
After replacing that fixture's reconstructed Graph with actual descriptor
bindings, the entire prepared-owner file passes 100 checks. The static-kernel
dynamic-ABI refusal and numerical checks are retained.
Both RHS layouts complete 48 correctness-gated profiles each (96 total).
Maximum absolute output error is 0.015625. Warm public-call medians span
0.420867–0.837734 ms for column RHS and 0.411667–0.838203 ms for row RHS. Producer/consumer CUDA-event dispatch and
public wall timing are separate; no speedup or kernel-only claim is made.
Compiler SHA256:
65126f8c651e5ecc0473239e311e9e1e81a812b5a59b348b0b75d1c9d982e4b2.
Evidence: benchmarks/baselines/nvidia_native_bounded_tensor_partition_20261008/README.md.
Broader composition/asynchronous ownership, sibling owning-device proof,
broader performance evaluation and focused native publication remain open.

Owning RTX 5070 bounded native envelope validated; broader W1.1 remains open.


## Shared native compiler gfx1201 parity — 2026-10-08

Owner ROCM-NVFP4-INGEST-1 / W1.1 / FRONTEND-IR-MEDIUM-1 / E2E-REAL-6.
Sync SHARED-NATIVE-PROGRAM-PARITY-2026-10-08.
The latest Super-Bear-built LLVM/MLIR 23.1.1 compiler and matching layout
library were transferred with current source into a new Tajasaurus scratch
checkout, preserving the prior proof. Live hardware is RX 9070 XT/gfx1201,
UUID GPU-28d9e7efbf2ef716. Compiler SHA256:
65126f8c651e5ecc0473239e311e9e1e81a812b5a59b348b0b75d1c9d982e4b2.
Native whole-Graph partition plus public NVFP4 resident JIT checks pass:
27 passed, one timeout-plugin configuration warning. The first collection
attempt lacked a shared benchmark helper; the unchanged selection passes
after transferring that source dependency. No test was weakened or skipped.
Six JIT/portable profiles pass independent conversion/storage/folded float64
numerics before and after timing; maximum absolute output error 0.0.
Converter/storage/consumer event windows, graph-dispatch and cold/warm/portable
wall times are separate. No kernel speedup or selector promotion is claimed.
This checks gfx1201 NVFP4 parity after the shared native SM120 dynamic changes;
it does not establish ROCm dynamic producer capacity, wider layouts,
model-quality acceptance, general AD or complete five-slice closure.
Evidence: benchmarks/baselines/gfx1201_shared_native_revalidation_20261008/README.md.

Shared compiler parity validated for the separate gfx1201 NVFP4 route; NVIDIA tensor/attention proof remains separately bound.


## Independent matrix/scale broadcast foundation — evidence reference

Owner FRONTEND-IR-MEDIUM-1 / AD-RESIDUAL-EVAL-1.
Sync INDEPENDENT-SCALE-BATCH-2026-10-07.
benchmarks/baselines/scaled_independent_broadcast_foundation_20261007/README.md
records independent matrix/scale prefix and transpose oracle checks.
This is reference semantic conformance only. Production Graph verification,
four independent Schedule/Tile address maps, scale-gradient reductions and
owning-device ABI/numerical/timing proof remain required; generic batching
and transpose closure states remain unchanged.

Follow-up required: SM120 NVFP4 four-operand maps and native ABI proof.


## Direct broadcast scaled frontend — owning gfx1201 integration

Owner FRONTEND-IR-MEDIUM-1 / AD-RESIDUAL-EVAL-1 / ROCM-FP8-BLOCKSCALE-1.
Sync DIRECT-SCALED-BROADCAST-2026-10-08.
Ordinary traced scaled_matmul with batching="broadcast" now infers its result
from all four right-aligned matrix/scale prefixes. Known incompatible or zero
extents are rejected; named packed physical contracts cannot be reinterpreted.
No replacement Python Graph, Tile constructor or numerical backend is added.
Native existing independent address maps consume the actual frontend Graph.
The named source prefixes (2,1), (3), (), (1,3) join to output (2,3).
FP8/MXFP8 primal plus FP32-scale JVP/VJP cover all four matrix transpose flags
at M=3,N=5,K=35 (ragged K32 groups). Scale adjoints reduce to each scale's own
prefix, including singleton and absent axes. Warm compiler-free calls pass.
Owning RX 9070 XT/gfx1201 lane: 16 passed; focused frontend/dtype/op/diagnostic/
pass gates: 988 passed. Sixteen benchmark profiles check independent float64
primal/JVP and finite-difference VJP before/after timing; maximum output error
1.01959069e-07. Public cold/warm wall and prepared HIP program-event windows
are separate. No isolated kernel speedup or default-selector promotion.
Compiler SHA256: 65126f8c651e5ecc0473239e311e9e1e81a812b5a59b348b0b75d1c9d982e4b2.
All recorder source hashes match the sealed source. Encoded scale byte
annotations retain their existing explicit planned/gated token.
Generic dynamic/composed batching, other group widths/storage types,
matrix derivatives and sibling owning-device proof remain open.
Evidence: benchmarks/baselines/gfx1201_direct_broadcast_scaled_20261008/README.md.
Recorder: benchmarks/rocm/benchmark_direct_broadcast_scaled.py.

Follow-up required: SM120 packed NVFP4 independent prefixes and scale AD need a separate native contract/ABI and exact-device proof.


Native SM120 partition recorder: benchmarks/nvidia/benchmark_native_sm120_tensor_partition.py. It records actual compiler-owned tensor members and separate producer/consumer timings for the named packet; it does not establish generic W1.1 closure.


## Attention owner producer-stream ordering — 2026-10-08

Owner E2E-REAL-6 / AD-RESIDUAL-EVAL-1.
Sync ATTENTION-OWNED-STREAM-2026-10-08.
Saved-O/LSE capture and reverse now use a private CUDA stream with declared
producer event dependencies and private copies; synchronous returns remain.
114 unit/public VJP checks, 2 pending-producer device checks and 28 native JVP
checks pass on RTX 5070. Matched identical-package counterbalanced A/B reports
capture/backward/pair control-over-candidate medians 0.968099/0.938151/0.966780.
This is a wall-time regression, not a performance promotion; allocator/driver
attribution and tuning remain open. Synchronous allocation/free may still
introduce implicit synchronization. Generic dynamic/composed AD and
asynchronous ownership remain open.
Evidence: benchmarks/baselines/nvidia_attention_owned_stream_20261008/README.md.
Recorder: benchmarks/nvidia/benchmark_attention_owned_stream_ab.py.

Owning SM120 producer ordering and JVP parity validated in the named envelope. Performance tuning remains required.

Historical A/B control source: benchmarks/baselines/nvidia_attention_owned_stream_20261008/context_control.py. This preserves the prior context-wide owner for the matched comparison; it is not a production launcher or candidate.


## Grouped attention producer dependencies — 2026-10-08

Owner E2E-REAL-6 / AD-RESIDUAL-EVAL-1.
Sync ATTENTION-GROUPED-STREAM-2026-10-08.
Capture, reverse seed and JVP direction groups now validate every tensor
and order each distinct CUDA producer stream once per call. No wait is
reused across calls. RTX 5070 affected lane: 146 passed, four existing fork
warnings. A biased copy-loop arity issue was repaired before the green rerun.
Matched per-buffer versus grouped-wait A/B is running; no speedup claim.
Evidence: benchmarks/baselines/nvidia_attention_grouped_stream_20261008/README.md.
Recorder: benchmarks/nvidia/benchmark_attention_owned_stream_ab.py.
Historical control: benchmarks/baselines/nvidia_attention_grouped_stream_20261008/per_buffer_control.py.

Named owning SM120 numerical and pending-producer parity validated; performance remains under evaluation.

Historical prior candidate source: benchmarks/baselines/nvidia_attention_owned_stream_20261008/owned_stream_candidate.py. Its hash belongs to that earlier A/B, not the newer grouped-wait change.


### Grouped-wait matched A/B completion

ATTENTION-GROUPED-STREAM-2026-10-08 completed 96 correctness-gated profile
executions and 48 identical-package comparisons on RTX 5070.
Median per-buffer/grouped capture/backward/pair wall ratios:
1.011891 / 1.043463 / 1.026943. Grouping reduces host/runtime dependency work;
no isolated kernel gain or general performance promotion is established.
146 affected tests and 12 audit/recorder checks pass.
The earlier context-wide A/B remains a separate comparison; ratios across
the two runs are not multiplied or treated as proof of baseline recovery.
Evidence: benchmarks/baselines/nvidia_attention_grouped_stream_20261008/README.md.


## Independent matrix/scale batch-axis ownership — 2026-10-08

Owner FRONTEND-IR-MEDIUM-1 / AD-RESIDUAL-EVAL-1 / ROCM-FP8-BLOCKSCALE-1.
Sync SCALE-ONLY-BATCH-2026-10-08.
The ordinary traced native gfx1201 scaled product now has owning-device
proof when any one of A/B/lhs-scale/rhs-scale alone supplies prefix (2,3).
13 frontend/device checks and 12 correctness-gated primal/JVP/VJP benchmark
profiles pass; maximum absolute error 1.15684470559e-07. Shared scale adjoints
reduce unmapped axes and retain their own scale shapes. Warm calls forbid
compiler subprocesses. Public wall and native HIP program-event timings
are separate. Generic batching/transpose statuses remain incomplete.
Evidence: benchmarks/baselines/gfx1201_scale_only_batch_20261008/README.md.
Recorder: benchmarks/rocm/benchmark_scale_only_batch.py.

Follow-up required: SM120 packed contracts need their own four-independent-operand maps and scale AD proof.


## Immutable checked scaled-owner metadata — 2026-10-08

Owner E2E-REAL-6 / FRONTEND-IR-MEDIUM-1 / AD-RESIDUAL-EVAL-1.
Sync SCALED-OWNER-METADATA-2026-10-08.
Prepared HIP execution now reads immutable checked storage metadata rather
than mutable diagnostic program JSON. Diagnostic snapshots remain per-owner
and are parsed lazily; forged shapes/counts/outputs cannot alter execution.
25 host unit checks and 80 exact gfx1201 device checks pass.
Counterbalanced identical-native-package A/B completes 24 matched pairs:
control/candidate warm public wall ratio 0.979850675, approximately 2.1%
slower candidate. No performance promotion or isolated kernel gain.
Evidence: benchmarks/baselines/gfx1201_scaled_owner_metadata_20261008/README.md.
Recorder: benchmarks/rocm/benchmark_scaled_owner_metadata_ab.py.
Historical control: benchmarks/baselines/gfx1201_scaled_owner_metadata_20261008/owner_control.py.

Not applicable to this backend's execution: only the gfx1201 HIP scaled-owner adapter changed. Its own checked owner metadata/lifetime requires separate implementation and proof.


## Cached checked scaled storage dtype — 2026-10-08

Owner E2E-REAL-6 / FRONTEND-IR-MEDIUM-1 / AD-RESIDUAL-EVAL-1.
Sync SCALED-OWNER-DTYPE-2026-10-08.
Canonical NumPy binding dtypes are resolved once in the immutable checked
ABI; each call still checks dtype, shape, layout and byte capacity.
168 host and 80 owning gfx1201 checks pass. Nine-sample counterbalanced
A/B completes 24 matched pairs with identical native program/image/Graph:
control/candidate warm-wall median 1.031385388. This is about 3.1% lower
host-adapter wall time versus the prior metadata version, not kernel gain
or closure of general batching/AD or ROCm performance programs.
Evidence: benchmarks/baselines/gfx1201_scaled_owner_dtype_20261008/README.md.
Recorder: benchmarks/rocm/benchmark_scaled_owner_metadata_ab.py.
Historical control: benchmarks/baselines/gfx1201_scaled_owner_dtype_20261008/dtype_control.py.

Not applicable to physical execution: only the gfx1201 HIP adapter changed; no sibling ABI or schedule promotion.

Preserved prior metadata candidate: benchmarks/baselines/gfx1201_scaled_owner_metadata_20261008/owner_candidate.py. This is the source snapshot for the earlier 0.979850675 ratio, not the current dtype-binding implementation.

## Native short-K dispatch experiment — 2026-10-08

Owner ROCM-FP8-BLOCKSCALE-1 / E2E-REAL-6.
Sync GFX1201-NATIVE-SHORT-K-2026-10-08.
Evidence: benchmarks/baselines/gfx1201_native_short_k_dispatch_20261008/README.md.
New device coverage: tests/device/rocm/test_fp8_blockscale_w8a8.py.
Recorder: benchmarks/rocm/record_gfx1201_interleaved_compiler_formats.py.

Not applicable to nvidia physical lowering: this transformation is restricted to the ROCm K128/fp32-scale projected register generator. Shared package ABI is unchanged. No sibling device or performance closure is claimed.

## Mixed nested scaled map integration — 2026-10-08

Owner FRONTEND-IR-MEDIUM-1 / AD-RESIDUAL-EVAL-1 / E2E-REAL-6.
Sync GFX1201-MIXED-NESTED-SCALED-2026-10-08.
Evidence: benchmarks/baselines/gfx1201_mixed_nested_scaled_20261008/README.md.
Fixtures: tests/unit/test_native_mixed_scaled_maps.py and
tests/device/rocm/test_mixed_nested_scaled_execution.py.
Recorder: benchmarks/rocm/benchmark_mixed_nested_scaled.py.

Shared JIT parity validated on RTX 5070 for attention stream ownership and bounded producer paths within the 151-test regression lane. The mixed-map physical consumer is gfx1201-only; no NVIDIA scaled-map support or performance promotion is claimed.


## Static NVIDIA producer-chain integration — 2026-10-08

Owner W1.1 / FRONTEND-IR-MEDIUM-1 / E2E-REAL-6.
Sync SM120-NATIVE-PRODUCER-CHAIN-2026-10-08.

Native outlining and CUDA ownership for static RMSNorm → softmax → matmul chains are implemented and built. RTX 5070 FP16/BF16 prepared and resident chain numerics pass against an independent float64 oracle with storage rounding at each producer. The 103-test partition/replay lane and 307 audit/diagnostic/pass gates pass. Dynamic chains, broader producer composition and separate stage/program benchmarking remain open.


### Static chain public entry and benchmark evidence

Sync SM120-NATIVE-PRODUCER-CHAIN-2026-10-08.
Evidence: benchmarks/baselines/nvidia_native_producer_chain_20261008/README.md.
Four ordinary public JIT/replay cases and four correctness-gated timed profiles now pass on RTX 5070. Maximum error 7.7983e-6. Separate stage CUDA-event timings and complete prepared host-wall timings are recorded; no performance promotion. Dynamic-chain capacity and broader composition remain open.


### Dynamic-chain capacity investigation

Sync SM120-NATIVE-PRODUCER-CHAIN-2026-10-08.
Fourteen owning SM120 runtime capacity cases pass; compiler dynamic-chain admission remains gated pending compiler-generated manifest and ABI proof. This is NVIDIA runtime evidence only; nvidia target support is unchanged.


### Static three-producer and epilogue proof

Sync SM120-NATIVE-PRODUCER-CHAIN-2026-10-08.
Evidence: benchmarks/baselines/nvidia_native_producer_chain_20261008/README.md.
Owning RTX 5070 static three-producer proof passes 16 cases, including fused epilogues, warm/portable replay and resident execution. Eight timed two/three-producer profiles pass correctness before and after timing. Dynamic compiler admission remains gated.


## Prepared native attention staging — 2026-10-08

Owner E2E-REAL-6 / AD-RESIDUAL-EVAL-1.
Sync PREPARED-ATTENTION-STAGING-2026-10-08.
Evidence: benchmarks/baselines/nvidia_prepared_attention_staging_20261008/README.md.

Owning RTX 5070: 64 prepared JVP/backward checks and both independent-stream isolation tests pass. Control JVP detects the old global wait; native arithmetic and package ABI are unchanged. Matched fresh-process A/B is complete: seven cases, three paired windows each, control/candidate prepared-wall ratios 1.040043–1.608286. Separate device-event samples are retained; no isolated kernel gain or general attention closure is claimed. Public asynchronous ownership is still open.


## Widened gfx1201 short-K evaluation — 2026-10-08

Owner ROCM-FP8-BLOCKSCALE-1 / E2E-REAL-6.
Sync GFX1201-SHORT-K-WIDE-2026-10-08.
Evidence: benchmarks/baselines/gfx1201_short_k_wide_four_format_20261008/README.md.

Not applicable to nvidia physical lowering: this packet evaluates the earlier gfx1201 short-K compiler pair only. No shared ABI or selector change; no sibling performance claim.


## Isolated LDS short-K candidate — 2026-10-08

Owner ROCM-FP8-BLOCKSCALE-1 / E2E-REAL-6.
Sync GFX1201-LDS-SHORT-K-2026-10-08.
Evidence: benchmarks/baselines/gfx1201_lds_short_k_candidate_20261008/README.md.

Not applicable to nvidia physical lowering: this isolated gfx1201-only generator experiment changes no shared ABI, selector or primary source. No sibling performance claim.

Recorder schema adds actual loaded-HIP resource and compiler-source identities. This additive diagnostic evidence does not change this backend's execution ABI or establish physical parity.

K2048-only follow-on remains a gfx1201-only isolated generator experiment; no shared execution ABI or sibling admission changed. Its numerical/timing proof does not establish sibling parity.


## Public NVIDIA softmax alias integration — 2026-10-08

Owner E2E-REAL-6 / FRONTEND-IR-MEDIUM-1.
Sync NVIDIA-PUBLIC-SOFTMAX-ALIAS-2026-10-08.
Evidence: benchmarks/baselines/nvidia_public_softmax_alias_20261008/README.md.

Owning RTX 5070 static last-axis public softmax/safe alias executes through native Schedule/Tile/PTX packages with FP32/FP16/BF16 checked ABIs. Fourteen package and eighteen public/portable/changed-input cases pass; eighteen timed profiles preserve matching alias image bytes and separate resident-event/public-wall scopes. Dynamic/composed/AD and native tensor-chain safe-alias composition remain open.

Public alias shared integration regression: 224 WSL checks pass, ten skips; final matching-runtime RTX 5070 lane passes 32 cases. Generated docs are regenerated. This host lane does not establish sibling physical parity.

## Aggregate validation and isolated native follow-up — 2026-10-08

Owner FRONTEND-IR-MEDIUM-1 / E2E-REAL-6 / W1.1.
Sync NVIDIA-NATIVE-OWNER-FOLLOWUP-2026-10-08.

Fresh aggregate WSL unit sweep completed: 23,208 passes, 7,532 skips,
874 deselections and four failures. Two failures are genuine scaled_matmul
batching/transpose closure obligations. The stale SM120 softmax-safe target
fixture is reconciled with its owning native/public proof; the standalone
dashboard is regenerated with its owning generator. Focused repair validation
passes 234 tests with 179 skips; no aggregate green claim.

An isolated RTX 5070 native staging candidate completes five alternating
fresh-process A/B windows over 18 identical-package profiles. Per-profile
control/candidate public-wall ratios range 2.169–2.353, median 2.270.
Resident event median ratio 0.997 is separate; no arithmetic kernel gain.
An isolated prepared macro-CTA owner candidate passes the original long BF16
norm/matmul numerical gates. Twelve producer/dtype/epilogue cases prove six
plain macro and six fused typed routes with changed-input and retained-output
checks. Broader owner regression: 171 passes and one control-reproduced
diagnostic drift; the diagnostic repair plus norm lane passes 11 tests.
Both native candidates remain outside authoritative source pending integration.
Scratch evidence: /home/angstorms/scratch/nvidia-softmax-staging-evidence-20261008/
and /home/angstorms/scratch/nvidia-macro-producer-candidate-20261008/.

Owning RTX 5070 numerical and host-staging evidence is recorded; authoritative integration, broader dynamic composition and focused native delivery remain open.

### Native owner follow-up integrated into authoritative source

NVIDIA-NATIVE-OWNER-FOLLOWUP-2026-10-08 now integrates the independently
proved native retained staging and static macro-CTA producer owner into the
working branch. A matching CMake runtime build succeeds; runtime SHA256:
ca860af9503bbb4733c64d3a1c51634de7f7f1e09a7e3ad7720c8c6f67c58d60.
A durable twelve-case owning producer test is added. Fresh combined device
and shared contract validation is running. The prior A/B is component-bound
pre-integration evidence, not a new combined-build speedup claim.
Evidence: benchmarks/baselines/nvidia_native_owner_integration_20261008/README.md.
Generic scaled_matmul batching/transpose closure and focused PR delivery remain
open. Dynamic multi-producer and asynchronous ownership admission are unchanged.

### Matching native-owner proof and long-chain characterization

NVIDIA-NATIVE-OWNER-FOLLOWUP-2026-10-08: the combined authoritative runtime
passes 448 owning RTX 5070 device checks and 397 shared contract/registry gates.
The extended producer recorder completes twelve numerical/timing profiles;
four K4096 column-major profiles select native macro consumers for two/three
producers in FP16/BF16. Maximum absolute error across all twelve: 1.48945e-5.
Separate stage CUDA events and prepared host wall are recorded; no A/B kernel
gain or general asynchronous/dynamic composition closure is claimed.
The initial benchmark upload layout mismatch is repaired by consuming the
descriptor layout, with its terminal failure log retained. Evidence:
benchmarks/baselines/nvidia_native_owner_integration_20261008/README.md.
Full generic scaled_matmul batching/transpose closure and PR delivery remain open.

## Isolated cooperative softmax compiler proof — 2026-10-08

Owner W1.1 / E2E-REAL-6.
Sync NVIDIA-COOPERATIVE-SOFTMAX-2026-10-08.

Long-chain stage measurements motivate an opt-in cooperative_128 native
softmax lowering. The isolated matching LLVM/MLIR compiler builds. Three
f16/bf16/f32 Tile fixtures pass verify-each and lower to shared-memory native
reductions; 54 barriers cover the two reductions and protected broadcasts.
Serial target output is byte-identical to the unchanged compiler. Invalid
schedule spelling is refused. This is compiler-only evidence: native launch,
Graph/Schedule policy hashing, Tile verifier/checked geometry registration,
public admission, nonfinite/ragged numerical proof and A/B timing are pending.
No public route or authoritative compiler is replaced by this experiment.
Evidence: benchmarks/baselines/nvidia_native_owner_integration_20261008/cooperative-softmax/README.md.

NVIDIA native lowering proof only; owning RTX 5070 execution remains pending.

### Cooperative softmax native Schedule contract proof

NVIDIA-COOPERATIVE-SOFTMAX-2026-10-08 now has an isolated matching compiler
for explicit Graph policy, native Schedule hashing, checked Tile projection
and cooperative native lowering. FP16/BF16/FP32 policies have distinct serial
and cooperative hashes. Native Schedule replay is byte-identical; forged
Schedule policy and workgroup geometry are rejected. Default serial Graph,
Schedule and Tile boundaries are byte-identical to the unchanged primary.
Six actual ordinary/safe frontend traces yield identical cooperative Tile
without caller-owned Graph mutation. Forty isolated regression tests pass
with the project pytest configuration. The old blanket refusal fixture is
replaced in the candidate by positive hashed-policy and invalid/sibling
refusal checks; primary admission/tests remain unchanged.
Native runtime ABI/geometry, PTX packaging, owning numerical and timing proof,
automatic schedule selection and authoritative integration remain pending.
Evidence: benchmarks/baselines/nvidia_native_owner_integration_20261008/cooperative-softmax/README.md.

### Cooperative softmax full-package owning proof — 2026-10-08

NVIDIA-COOPERATIVE-SOFTMAX-2026-10-08: the isolated matching compiler/runtime
passes 102 RTX 5070 host/resident numerical checks across three storage types,
ragged widths and nonfinite inputs. Five alternating serial/cooperative timing
windows show 81.86–98.74x resident event speedup and 2.25–2.96x separate host
speedup for 128x4096/4097. Small 3x17 rows show no consistent benefit.
This is an explicit candidate package comparison, not full-chain/vendor
performance. Primary source admission is unchanged; producer attachment,
automatic selection, authoritative integration and delivery remain open.
Evidence: benchmarks/baselines/nvidia_native_owner_integration_20261008/cooperative-softmax/README.md.

### Cooperative producer-chain owning execution

NVIDIA-COOPERATIVE-SOFTMAX-2026-10-08: twelve isolated RTX 5070
FP16/BF16 one/two/three-producer profiles pass changed-input numerics
and retained-output lifetime checks. Ragged typed and long macro consumers
are covered. Matched whole-chain timing, selection and authoritative
integration remain open; dynamic multi-producer admission is unchanged.
Evidence: benchmarks/baselines/nvidia_native_owner_integration_20261008/cooperative-softmax/README.md.

NVIDIA-COOPERATIVE-SOFTMAX-2026-10-08 matched whole-chain follow-up:
four long FP16/BF16 two/three-producer profiles show 3.317–3.362x
prepared host speedup, with unchanged consumer images and correctness
before/after five alternating timing windows. This is separate host
copy/launch/completion timing; authoritative integration and selection
remain open. Evidence: benchmarks/baselines/nvidia_native_owner_integration_20261008/cooperative-softmax/README.md.

### Authoritative cooperative softmax explicit-policy integration

NVIDIA-COOPERATIVE-SOFTMAX-2026-10-08: matching compiler/runtime rebuild
succeeds after integration of explicit native Schedule/Tile lowering and
checked package/prepared-owner dispatch. Five native fixtures pass.
Default serial selection and dynamic multi-producer admission are unchanged.
Durable owning regression validation is running; automatic policy selection,
broader closure and focused publication remain open.
Evidence: benchmarks/baselines/nvidia_native_owner_integration_20261008/cooperative-softmax/README.md.

NVIDIA-COOPERATIVE-SOFTMAX-2026-10-08 authoritative regression follow-up:
240 affected device tests pass after preserving the legacy serial descriptor
spelling; five native fixtures and fifteen documentation gates pass.
The initial combined 674-pass/14-failure lane is preserved separately.
Default serial Target IR is byte-identical to preintegration control.
Automatic selection, broader closure and focused publication remain open.

NVIDIA-COOPERATIVE-SOFTMAX-2026-10-08 authoritative timing follow-up:
four matched FP16/BF16 two/three-producer profiles pass correctness and
record 3.280–3.405x prepared host-call median speedup with unchanged
consumer images per pair. Source/binary-bound raw samples are preserved;
no whole-chain kernel gain or automatic selection is claimed.
Evidence: benchmarks/baselines/nvidia_native_owner_integration_20261008/cooperative-softmax/README.md.

## Mapped composed scaled-product native integration — 2026-10-08

Owner FRONTEND-IR-MEDIUM-1 / AD-RESIDUAL-EVAL-1 / E2E-REAL-6.
Sync GFX1201-COMPOSED-SCALED-MAPS-2026-10-08.

72 owning gfx1201 numerical/lifetime cases pass ordinary compiled primal,
JVP and reverse AD with mixed inputs, two leading map axes, shared scales
and K256/K37. Nine complete native-program profiles preserve separate HIP
event and public host timing samples; no speedup claim or default change.
Ordinary primal now selects the registered ROCm family with exact gfx1201
metadata. Generic batching/transpose, nonleading/dynamic/storage derivatives
and arbitrary product-output broadcasting remain open.
Evidence: benchmarks/baselines/gfx1201_composed_scaled_maps_20261008/README.md.

Sibling assessment: Follow-up required: sm_120 needs native scaled-contract execution and owning parity; gfx1201 proof does not transfer.

## LDS loop-only specialization and timing admission — 2026-10-08

Owner ROCM-FP8-BLOCKSCALE-1 / E2E-REAL-6.
Sync GFX1201-LDS-LOOP-ONLY-2026-10-08.

An isolated uniform K2048 group-loop specialization shares prologue/epilogue
but raises M200 registers from 192 to 223. Eight owning gfx1201 tests and
sixteen four-format numerical profiles pass. Duration-admitted paired gains
of 1.6–1.8% are comparable to unchanged-image controls; no promotion.
The recorder now retains/retries entire paired series below its minimum
duration; three new and 21 affected host tests pass. Two final unchanged-image
rows exceed the 5% clock/event agreement band and remain diagnostic.
Evidence: benchmarks/baselines/gfx1201_lds_loop_only_20261008/README.md.

Sibling assessment: Native LDS recipe not applicable to SM120. Follow-up required for duration and independent-clock admission in NVIDIA recorder families; no device parity claim.

## Explicit static asynchronous saved-LSE ownership — 2026-10-08

Owner E2E-REAL-6 / AD-RESIDUAL-EVAL-1.
Sync NVIDIA-ATTENTION-ASYNC-OWNER-2026-10-08.

Compiler-owned VJP capture accepts explicit asynchronous execution on an owned
CUDA stream. Snapshot dependencies, retained sources, stream-ordered seed
release and registered-consumer completion protect saved O/LSE and gradients.
156 affected owning/host tests and eight strengthened serialized replay cases
pass on RTX 5070. Eight numerical timing arms preserve separate submission,
completed host and complete-owner CUDA event costs with identical kernel images.
No counterbalanced speedup claim or default change. Dynamic/composed attention,
automatic external-consumer lifetime tracking and broader AD remain open.
Evidence: benchmarks/baselines/nvidia_attention_async_owner_20261008/README.md.

Sibling assessment: Parity validated on sm_120 in the named static f32 saved-LSE compact/complete envelope. Dynamic/composed/JVP and automatic consumer tracking follow-up required.

## Bounded saved-LSE sequences — 2026-10-08

Owner E2E-REAL-6 / AD-RESIDUAL-EVAL-1.
Sync NVIDIA-ATTENTION-BOUNDED-SEQUENCES-2026-10-08.

Native bounded sequence Graph/Schedule/image work is active; public package/JIT and AD residual integration remain follow-up required.
Native bounds are retained in verified Schedule identity. Raw owning-device image evidence does not establish checked public ABI closure.
Evidence: ../../compiler/FIVE_SLICE_STATUS_20261007.md.

## Checked bounded attention packages — 2026-10-08

Owner E2E-REAL-6 / AD-RESIDUAL-EVAL-1.
Sync NVIDIA-ATTENTION-BOUNDED-PACKAGES-2026-10-08.

Checked bounded package/private residual execution is proved on RTX 5070; public JIT symbolic tracing and automatic AD export require follow-up.
See ../../compiler/FIVE_SLICE_STATUS_20261007.md and the owning evidence packet.

## Bounded public JIT native attention AD — 2026-10-08

Owner E2E-REAL-6 / AD-RESIDUAL-EVAL-1.
Sync NVIDIA-ATTENTION-BOUNDED-JIT-2026-10-08.

Native paired AD now projects a frontend sequence-capacity request into
symbolic Graph/Schedule/Tile products, preserving fixed dimensions, named
argument roles, saved O/LSE and requested adjoints. The matching build,
70 focused host tests and eight owning RTX 5070 compiler-forbidden serialized
replay cases pass; six runtime shapes reuse each image pair. Eight numerical
timing arms record separate completed capture, backward host and whole-owner
CUDA event costs. Dynamic physical broadcast bias, dynamic JVP, arbitrary
composition, aggregate validation and focused publication remain open.
Evidence: benchmarks/baselines/nvidia_bounded_attention_jit_20261008/README.md.

Sibling assessment: Parity validated only on RTX 5070 for no/full bias. Dynamic physical broadcast-bias carrier integration remains required.

NVIDIA-ATTENTION-BOUNDED-JIT-2026-10-08 affected regression follow-up:
652 host WSL attention/ABI/native-AD/registry/lifecycle tests pass.
Six static test doubles now declare empty capacity bounds; saved generation
mismatch assertions remain intact. Initial failures are preserved in the
evidence packet. Aggregate/full-suite and publication gates remain open.

## Bounded physical broadcast-bias execution — 2026-10-08

Owner E2E-REAL-6 / AD-RESIDUAL-EVAL-1.
Sync NVIDIA-ATTENTION-DYNAMIC-BIAS-2026-10-08.

Four actual physical bias extents now pass through native Tile/NVIDIA lowering
for symbolic query/key storage and physical bias-gradient reduction. Package
guards, residual allocations and checked launch scalars preserve the sealed
symbolic policy while resolving actual storage; malformed pitches fail before
CUDA loading. Matching native builds, 668 host WSL gates, sixteen native/
guard cases and twenty owning RTX 5070 public replay cases pass. Six runtime
shapes reuse each image pair with compiler recovery forbidden. Twenty
correctness-gated timing arms preserve separate host and whole-owner event
costs; no isolated kernel speedup or aggregate closure.
Dynamic JVP, arbitrary composition, wider dynamic axes, automatic external
consumer tracking and focused publication remain open.
Evidence: benchmarks/baselines/nvidia_dynamic_attention_bias_20261008/README.md.

Sibling assessment: Parity validated for the named RTX 5070 bounded sequence and traced bias policies. Broader AD and full-suite closure remain required.

NVIDIA-ATTENTION-DYNAMIC-BIAS-2026-10-08 final owning regression follow-up:
92 static tuple-AD/asynchronous/bounded-package RTX 5070 cases and eleven
final audit-document tests pass. Packet source/compiler/runtime fingerprints
match authoritative bytes. Full-suite generic closure and delivery remain open.


## Nested SM120 NVFP4 leading maps — 2026-10-08

Owner E2E-REAL-6. Sync NVIDIA-NVFP4-NESTED-MAPS-2026-10-08.

Native static nested NVFP4 leading maps execute on RTX 5070 (SM120): three coupled policies, all four operand orientations, rank-four/rank-five logical tensors, odd K31/K129. Twenty-four device cases prove one launch, changed-input compiler-free reuse and retained outputs. Full logical prefix guards reject equal-product rebinding. The scalar/single-map regression lane passes 198 tests; 31 nested native-host checks and 490 registry checks pass. Timing characterization is recorded separately in nvidia_nested_nvfp4_20261008. Dynamic/mixed/nonleading maps, encoded-storage AD and generic closure remain open.

## PR895 CI route repair — 2026-10-08

Owner E2E-REAL-6 / FRONTEND-IR-MEDIUM-1 / W1.1.
Sync PR895-CI-ROUTE-REPAIR-2026-10-08.
Structural W1.1 proof belongs to compiler-route; exact SM120 execution remains hardware_nvidia. No device admission change.
Evidence: benchmarks/baselines/pr895_ci_route_repairs_20261008/README.md.
Generic scaled_matmul batching/transpose closure remains open; no coverage flag is promoted.

Matching-source x86-enabled full validation passes: 20,419 passed, 7,387 skipped, 874 deselected. Final drift gates pass 287 checks; generated-document checks pass. Historical build/fixture failures and repairs are recorded in benchmarks/baselines/rocm_linked_tool_identity_pr_20261007/README.md. Owning-device evidence remains architecture-specific.

## Frontend certificate numerical policy — 2026-10-08

Owner FRONTEND-IR-MEDIUM-1 / AD-RESIDUAL-EVAL-1.
Sync FRONTEND-PARITY-NUMERIC-POLICY-2026-10-08.
Frontend differential reuse binds rtol and atol in addition to tensor
signature and permitted effects. Stricter requests must establish their own
structural/numerical certificate. This shared frontend guard changes no
nvidia physical lowering, native image, ABI or execution capability.
Host numerical certificate tests are applicable; device performance is not
applicable to this certification-only correction. Generic scaled-matmul
batching/transpose and wider architecture-owned AD envelopes remain open.
Evidence: benchmarks/baselines/frontend_parity_numeric_policy_20261008/README.md.

## W1.1 current-head proof refresh — 2026-10-08

Owner W1.1 / E2E-REAL-6. Sync NVIDIA-W11-CURRENT-HEAD-2026-10-08.
Shared contracts changed: none; this refresh binds current source and tools.
147 RTX 5070 cases and 24 correctness-gated timing profiles pass.
Owning sm_120 parity validated for the named producer, fused epilogue, legacy migration and partition-lifetime envelopes. Arbitrary/dynamic multi-producer composition and external asynchronous lifetime remain follow-up required.
Evidence: benchmarks/baselines/nvidia_w11_current_head_20261008/README.md.

## ROCm image SDK CI provisioning — 2026-10-08

Owner E2E-REAL-6 / FRONTEND-IR-MEDIUM-1. Sync ROCM-IMAGE-SDK-CI-2026-10-08.
Shared test infrastructure changed: required compiler-route installs verified
ROCm compiler/bitcode tools. No admission, ABI or physical selector changes.
345 host WSL tests pass without skips; the unchanged execution gate passes.
Not applicable to CUDA toolchain/physical schedules: isolated ROCm SDK provisioning leaves SM120 packages and runtime unchanged; existing RTX 5070 proof is retained.
Evidence: benchmarks/baselines/rocm_image_sdk_ci_20261008/README.md.

## Native unary replay reuse — 2026-10-08

Owner E2E-REAL-6. Sync ROCM-NATIVE-REPLAY-CACHE-2026-10-08.
Not applicable: the SM120 package verifier does not use the new ROCm replay cache.
Only successful native MLIR pass output is cached; artifact and descriptor
validation still run per call. Compiler, loaded-library, environment and full
IR and working-directory identity participate, with bounded entry/byte retention.
Evidence: benchmarks/baselines/gfx1151_native_replay_cache_20261008/README.md
and benchmarks/baselines/gfx1201_native_replay_cache_20261008/README.md.

## Matmul native replay reuse — 2026-10-08

Owner E2E-REAL-6. Sync ROCM-MATMUL-REPLAY-CACHE-2026-10-08.
Not applicable: this backend retains its existing native matmul replay path.
Only artifact.target == rocm enters the bounded ROCm cache; sibling physical
schedules and execution capabilities are unchanged.
Evidence: benchmarks/baselines/rocm_matmul_replay_cache_20261008/README.md.

ROCM-MATMUL-REPLAY-CACHE-2026-10-08 synchronized rerun: both gfx1151 and
gfx1201 retain numerical parity, equal images/fingerprints and three-to-one
replay subprocess counts after PR894 and explicit cwd identity. Owning tests
pass 23 gfx1151 / 22 gfx1201 (one unavailable NVIDIA-dialect skip).
Not applicable to nvidia physical execution; no sibling performance claim.
Evidence: benchmarks/baselines/rocm_matmul_replay_cache_20261008/README.md.

## Native math Graph replay reuse — 2026-10-08

Owner E2E-REAL-6 / FRONTEND-IR-MEDIUM-1. Sync ROCM-MATH-REPLAY-2026-10-08.
Shared contract: bounded pure replay additionally admits Graph-to-Schedule;
MathRecipe retains per-call Graph/Schedule/Tile and descriptor validation.
68 host regressions and 19 owning cache tests per architecture pass.
Not applicable: CUDA packaging does not call the ROCm replay adapter. No SM120 ABI, selector or execution state is changed.
Evidence: benchmarks/baselines/rocm_math_replay_cache_20261008/README.md.
