---
last_updated: 2026-10-07
audit_role: reference
---

# Five compiler slices after PR 892: completion evidence

This is an evidence ledger, not a closure declaration. The coordinated source
is the unpublished `codex/next-five-compiler-slices` scratch branch on
Super-Bear. GPU evidence remains architecture-specific. Individual packets
bind their actual dirty source and compiler/runtime binaries; an older packet
is not proof of every later source change.

Required production boundary: Python/textual frontend -> typed semantic Graph
MLIR -> verified differentiation/optimization -> Schedule MLIR -> Tile MLIR
-> backend Target IR -> native MLIR/LLVM lowering where supported -> image
and checked runtime ABI -> execution. Python reference/diagnostic code does
not constitute a production lowering.

| Slice | Implemented evidence | Incomplete or unverified scope |
| --- | --- | --- |
| GFX1201 NVFP4 ingest | Ordinary JIT three-stage resident ingest/storage/scaled-matmul and serialized replay, native Graph/Schedule/Tile/ROCm/LLVM, RX 9070 XT. `benchmarks/baselines/rocm_ingest_jit_program_20261005/README.md`, `rocm_ingest_runtime_20261005/README.md`. | Dynamic/layout/composed/AD envelopes and model-quality acceptance. Synthetic source-decoded comparison does not establish original BF16 checkpoint quality. The synchronized source snapshot passed matching gfx1201 rebuild/revalidation: `benchmarks/baselines/gfx1201_current_source_revalidation_20261007/README.md`. Other affected owning-host synchronization and full-suite gates remain open. |
| NVIDIA W1.1 producer migration | Current census and 12 canonical tensor replay rows on RTX 5070; pointer-backed tile.view plus typed fragments and accumulator recovery. `benchmarks/baselines/nvidia_w11_census_20261006/README.md`. | Arbitrary producer composition, noncanonical accumulator lineage and generic dynamic reconstruction. Historical constructors remain for older SM; SM120 proof does not retire those architectures. |
| NVIDIA attention forward with saved LSE | Public ordinary JIT tuple route; native paired AD; Graph/Schedule/Tile/Target ancestry replay; numerical O/LSE. `benchmarks/baselines/nvidia_ordinary_attention_20261006/README.md`, `nvidia_checkpoint_graph_lineage_20261007/README.md`. | Nested/dynamic tuples and general composed attention. Native explicit LSE cotangents are not covered by forward artifact proof. |
| NVIDIA attention backward | Compiler-generated paired AD, private residual generation, reordered gradient requests, prepared native ownership and replay. Expanded paired timing on RTX 5070: `benchmarks/baselines/nvidia_paired_attention_cost_20261007/README.md`. Explicit static LSE cotangent packages, compiler-free replay and host/resident/event proof: `benchmarks/baselines/nvidia_lse_cotangent_package_20261007/README.md`. Compact/bias-gradient extension: `benchmarks/baselines/nvidia_lse_cotangent_gradient_roles_20261007/README.md`. Native semantic O/LSE differentiation: `benchmarks/baselines/nvidia_multiresult_attention_ad_20261007/README.md`. Public JIT two-result VJP and private residual ownership: `benchmarks/baselines/nvidia_jit_multiresult_owner_20261007/README.md` (72 cases; 600 timing windows). | General dynamic/nested/composed multi-result AD, automatic external-consumer lifetime tracking and broader batching/transpose closure. Explicit static asynchronous capture/backward with registered consumer streams is proved in nvidia_attention_async_owner_20261008. Native paired AD already saves O/LSE; standalone recompute benchmarking is not the current paired policy. |
| ROCm route/performance closure | Native math and movement on gfx1151/gfx1201, image identity reuse for declared packed envelopes, FP8/MXFP8/MXFP4 guarded vector-scale evaluation. `benchmarks/baselines/rocm_native_math_20261006/README.md`, `rocm_packed_image_identity_20261006/README.md`, `gfx1201_packed_vector_scales_20261006/README.md`. | General paged-KV layouts, broader image-key families, W8A8 short/ragged coverage and MXFP4 M256 Radiance attribution. Measured guarded gains do not close these programs. Current-source gfx1201 ingest/movement rerun: `benchmarks/baselines/gfx1201_current_source_revalidation_20261007/README.md`; gfx1151 matching-source math/movement revalidation is recorded below; broader family revalidation remains open. |

## Delivery gates

The aggregate is not ready to claim full completion. Curate source and receipts
into reviewable PR slices; synchronize matching builds for affected owning
hosts; run relevant registry/lifecycle and regression gates; resolve genuine
scaled-matmul batching and transpose closure failures without weakening them.
The earlier full suite is not green; later focused lanes do not establish a
fresh full-suite result. Generated documents and Graphify are refreshed in
sequence; an active refresh is not a completed gate.

All four backend queues carry architecture-specific synchronization entries.
Apple and x86 assessments do not constitute new physical execution proof.

## Current FP8 projected-image candidate

GFX1201-PROJECTION-ATTRIBUTION-2026-10-07 is recorded separately in
`benchmarks/baselines/gfx1201_m200_k1536_attribution_20261007/README.md`.
The native K128 LLVM-assumption candidate improves named ragged measurements,
while K32 FP8/MXFP8 preserve byte-identical reference images. Identical-image
timing controls remain noisy; no selector promotion or program closure is
claimed. The current source and historical compiler packets have separate
fingerprints. Earlier rebuild receipts prove their recorded snapshots.
Focused validation found one stale K64 identity fixture; the corrected
identity/K64 lane passed 88 tests. Full-suite batching/transpose closure and
broader owning-host family revalidation remain open.

## Named NVFP4 shared-LHS batch integration

The SM120 shared-LHS policy now carries one mapped RHS batch through native
Graph/Schedule/Tile/Target and the checked rows/batches ABI. Static scalar,
shared-LHS/shared-RHS and independent operand orientations pass owning RTX
5070 numerical tests and separate timing windows. Source/tool hashes are
verified in benchmarks/baselines/nvidia_shared_lhs_batch_20261007/README.md.
The frontend gate and scalar owner/constraint rules are exercised through
ordinary JIT and public vmap; no Python launch loop or A replication is used.
This extends the named profile. It does not close generic scaled_matmul
batching, dynamic/nested maps, linear-transpose AD or the full unit lane.


## Current-source gfx1151 math and movement revalidation

GFX1151-CURRENT-SOURCE-2026-10-07 synchronizes the compiler/Python and recorder
sources into Princess-Luna scratch, preserves prior bytes, rebuilds matching
LLVM/MLIR 23.1.1 core/ROCm tools and verifies source/driver/runtime hashes.
The required owning-device gate executes: 237 tests pass and one foreign
gfx1201 case skips. Seventy-two math rows cover f32/f16/bf16 JIT/portable
numerics and shape-independent image identities. Five native resident rows
verify 180 bit-exact warm windows and 90 native one-kernel event samples,
including repeated logical mappings across 64 logical/32 physical pages.

The separate paired paged-KV resident launch-window metric fails the 10%
device non-regression gate. Startup outliers and enqueue gaps are retained;
native single-kernel timings are recorded separately and are not a speedup
comparison. No selector promotion, universal layout/AD closure or sibling
physical proof is claimed. General integration and full-suite/publication
remain open. Evidence:
benchmarks/baselines/gfx1151_current_source_revalidation_20261007/README.md.


## Fresh full-unit gate

The current non-slow unit lane finished on Super-Bear WSL with 21,694 passed,
7,503 skipped, 874 deselected and three failures. Two are generic scaled_matmul
batching/transpose closure; the third is the new unit packet citation, now
corrected in tracked backend queues and checked separately. The closure
assertions/states remain unchanged. The isolated paged-KV address candidate
is not part of this frozen-source sweep. Evidence:
benchmarks/baselines/compiler_full_unit_20261007/README.md.

## Native pipeline regression gate and paged-KV repair

Current native compiler lane repair is recorded in
benchmarks/baselines/compiler_native_lane_repair_20261007/README.md:
485 core fixtures pass with 66 unsupported feature cases, all 153 NVIDIA/ROCm
backend fixtures pass, 374 registry/inventory checks pass, and 159 fresh RTX
5070 JIT NVFP4/saved-LSE forward/backward numerical/ownership tests pass.
These gates do not replace the red full-unit closure assertions.

Native paged-KV flat-token indexing is now integrated; both gfx1151/gfx1201
six-case bit-exact A/B and identical-image controls are recorded in
benchmarks/baselines/rocm_paged_kv_flat_index_20261007/README.md.
Focused owning gates pass 339 and 336 cases. The new gain comparison uses
same-allocation resident HIP-event launch windows including Python enqueue
gaps. It does not rewrite the earlier compiler-versus-retained-route
non-regression result or establish isolated kernel/public-call closure.
General layouts, cache families, W8A8 and MXFP4 obligations remain open.

Additional owning W1.1 proof passes 84 bounded row-major cases across seven
M/N/K axis sets, three producers, fp16/BF16 and fused/unfused consumers.
JIT/prepared/portable numerics, warm compiler-free reuse and fixed scratch
are checked on RTX 5070. This remains separate from general composed AD.
Native repair packet delivery gates and Graphify refresh are complete;
generic closure and broader aggregate/publication obligations remain open.

## Native scaled-product AD foundation

Native exact-per-block f32 JVP construction and Graph-to-Schedule scale-seed
derivation pass four focused fixtures. Tile execution remains open because
repeated products create duplicate content-addressed artifacts and the
consumer requires one match. Transposed FP8 Schedule admission and native
sum/program ownership also remain open. No device or timing proof is claimed.
Evidence: benchmarks/baselines/scaled_matmul_native_jvp_20261007/README.md.

The subsequent instance-binding repair now carries the scale-seed JVP through
Tile and ROCm Target while preserving content hashes. Core fixtures pass
489/555, with 66 unsupported features. Native sum, multi-kernel symbol/ABI
ownership and buffer lifetime still need executable integration. Historical
artifact-ambiguity failures remain in their original packet.
Evidence: benchmarks/baselines/scaled_matmul_artifact_binding_20261007/README.md.

Native scaled-product paired program export now outlines actual SSA products
and sum with typed buffer bindings, ownership and read/write lifetimes.
Primal and tangent argument storage metadata survives native differentiation.
This is a native program contract, not a launchable AD package: member image
projection, sum execution, physical allocation and completion ownership still
need integration and owning-device numerical/timing proof.
Evidence: benchmarks/baselines/scaled_matmul_native_program_20261007/README.md.

Native member projection now compiles the three FP8 products and sum to
gfx1201 HSACO images while retaining the complete program witness and
buffer-role mapping. This is cross-target image proof only. Whole-program
native ownership, paired device numerics and timing remain open.
Evidence: benchmarks/baselines/scaled_matmul_native_members_20261007/README.md.

The subsequent owning RX 9070 XT member diagnostic passes all four recorded
images at M17/N19/K256 against a float64 block oracle and a central scale
finite-difference oracle (maximum tangent error 9.52e-7). This validates
individual kernel execution using a diagnostic driver, not production native
whole-program ownership or timing. Receipts are retained in
benchmarks/baselines/scaled_matmul_native_members_20261007/README.md.

The subsequent native HIP owner executes all four recorded members through
one native submission, owns private buffers and completion, checks SSA
lifetimes and returned generations, and passes RX 9070 XT numerics, changed
tangent-input reuse and finite differences. Separate native HIP event
launch-window and host update/invoke/readback samples are recorded.
Automatic compiler/package projection into this owner and ordinary public
JIT AD integration remain open, as do generic closure and sibling execution.
Evidence: benchmarks/baselines/scaled_matmul_native_owner_20261007/README.md.

Compiler-owned member/package ABI projection now removes the diagnostic
hand-authored plan boundary: native SSA export and codegen supply witness,
buffer roles/lifetimes, actual symbols, scalars and resolved geometry.
The immutable package replays on RX 9070 XT with compiler subprocesses
disabled, passes changed-input numerics/finite differences and records
separate device/host windows. Package/registry corruption gates pass 305 tests.
Ordinary public JIT AD capture and general/sibling/full-unit closure remain open.
Evidence: benchmarks/baselines/scaled_matmul_native_package_20261007/README.md.

## Public FP8 scale-JVP integration

Nine public JIT owning gfx1201 cases pass after a matching compiler/runtime rebuild. Independent block numerics, scale finite differences, changed-input reuse and native execution receipts are checked. Benchmarking, generic batching/transpose and sibling physical parity remain open. Evidence: benchmarks/baselines/scaled_matmul_public_jvp_20261007/README.md.

Public FP8 JVP attribution now records compiler-free warm public calls at about 6.7 ms, prepared update/invoke/readback at 0.51–0.72 ms, and native sequence event windows at 0.025–0.030 ms. Native allocation/package validation and frontend/descriptor overhead require further attribution and integration; no isolated-kernel or speedup claim.

## Native scaled-program owner reuse

Bounded native HIP idle reuse reduces named warm public scale-JVP calls from 6.70–6.81 ms to 1.80–2.05 ms. Ten owning gfx1201 tests cover numerics, changed inputs, handles/generations and active allocation isolation. Broader envelopes and generic closure remain open. Evidence: benchmarks/baselines/scaled_matmul_native_reuse_20261007/README.md.

## Wider public FP8 scale-JVP orientation

Thirteen owning gfx1201 cases now cover physical B[N,K], ragged N129/N257, K1536/K2048 and multiple scale blocks. Separate public/native timing exposes transfer/readback overhead at larger shapes. Generic batching/linear-transpose assertions remain open and unchanged. Evidence: benchmarks/baselines/scaled_matmul_public_orientation_20261007/README.md.

## Native pinned staging candidate

Thirteen owning gfx1201 public scale-JVP cases pass with opt-in native pinned upload/readback staging. Wider public calls improve 6.22→2.29 ms and 4.02→2.31 ms; native sequence timing remains essentially unchanged. MXFP8/MXFP4 and sibling staging proof remain required before wider promotion. Evidence: benchmarks/baselines/scaled_matmul_native_pinned_20261007/README.md.

## Three-format native staging attribution

Six owning gfx1201 FP8/MXFP8/folded-MXFP4 diagnostic native ownership rows pass decoded numerical bounds, bitwise existing-launcher parity and changed-scale updates. Pinned staging improves host windows across these rows; wider public compiler-program/JIT integration and opt-in promotion remain open. Evidence: benchmarks/baselines/scaled_program_three_format_staging_20261007/README.md.

## Canonical typed FP8 primal JIT

Four ordinary JIT primal KN/NK cases pass native Graph/Schedule/Tile/Target numerics and compiler-free changed-scale reuse. Seventeen combined owning tests and 337 shared semantic/registry gates pass. MXFP8 primal, generic batching/transpose-left and native primal owner projection remain open. Evidence: benchmarks/baselines/rocm_typed_scaled_primal_20261007/README.md.

## MXFP8 unsigned scale admission follow-up

Native Schedule now admits signless or unsigned E8M0 bytes while rejecting signed storage. Three focused native fixtures pass, including unsigned Graph-to-Target lowering. Matching gfx1201 build succeeds; four existing FP8 primal cases pass. Four new public MXFP8 cases fail at frontend annotation capture because uint8 remains planned/gated. Public dtype metadata/admission and owning-device numerical/timing proof remain open; no MXFP8 ordinary JIT execution claim is made. Raw logs remain in owning-host scratch/scaled-matmul-member-images-20261007.

## Public MXFP8 primal integration

Explicit gated byte Tensor annotations now preserve status through tracer Graph capture. Four MXFP8 KN/NK cases pass on matching gfx1201; combined primal/JVP device lane passes 21 cases and shared frontend/registry gates pass 472. Separate public/native-image timing recorded. The earlier annotation and metadata-loss failures are superseded by this repair. Generic closure, eager E8M0 differential reference and native primal ownership remain open. Evidence: benchmarks/baselines/rocm_mxfp8_public_primal_20261007/README.md.

## Native primal owner integration

Ordinary FP8/MXFP8 now uses compiler-projected one-output SSA packages and native HIP lifetime/completion ownership. 21 owning-device tests, 492 native core fixtures and 483 shared gates pass. General speedup is unproved; large MXFP8 NK member event timing regresses versus the prior shape-projected physical image. Native image projection parity/retuning and generic closure remain open. Evidence: benchmarks/baselines/rocm_native_primal_owner_20261007/README.md.

## Primal physical projection attribution

Native projection now preserves compiler-derived member ABI and reuses image bytes across two same-profile shapes without reusing launch extents. Eight owning paired rows with identical-image controls show MXFP8 NK gains and FP8 KN regressions. Native profile-specific policy and broader three-format/public timing remain required. Evidence: benchmarks/baselines/rocm_primal_image_projection_20261007/README.md.

## Native primal profile policy

Native format/orientation/alignment policy passes sixteen controlled numerical/timing rows and 21 owning tests; native core 492 and shared gates 485 pass. Large ragged FP8 KN and MXFP8 NK gains and aligned NK improvements are recorded. Public calls remain 0.87-4.54 ms; staging/package-validation attribution is next. Six three-format diagnostic staging rows pass, without public MXFP4 or Radiance closure. Evidence: benchmarks/baselines/rocm_primal_profile_policy_20261007/README.md.

## Native public transfer and bounded idle eviction

Automatic moderate-size gfx1201 pinned staging plus bounded completed-owner
LRU eviction passes 24 public primal/JVP tests. Eight paired rows preserve
float64/changed-scale numerics and compiler-free warm execution. Large public
calls measure 1.22-1.74 ms versus pageable 4.17-4.72 ms; small ratios remain
1.000-1.009. These are wall-clock public costs, not isolated kernel gains.
Historical pre-eviction starvation measurements are retained unchanged.
Generic batching/transpose, E8M0 eager differential reference, sibling parity
and full-unit/publication remain open. Evidence: benchmarks/baselines/rocm_primal_transfer_attribution_20261007/README.md.

## E8M0 eager/native differential parity

The typed eager reference now decodes explicit E8M0 scales and preserves
smallest/NaN codes. 474 shared tests and 25 gfx1201 primal/JVP device cases
pass, including four eager/native block-oracle comparisons and one native
reserved-code case. This closes the eager E8M0 primal reference gap only;
encoded-scale derivatives and generic batching/linear transpose remain open.
No production image or timing change is claimed. Evidence: benchmarks/baselines/rocm_e8m0_eager_parity_20261007/README.md.

## Native typed shared-RHS batching and scale JVP

Static E4M3 FP32/E8M0 shared-RHS batches now preserve rank-three Graph/output
storage through native Schedule row flattening, Tile pointers, compiler member
ABI and HIP ownership. 39 owning gfx1201 tests include eight batched primal
and six batched f32 scale-JVP cases with float64/finite-difference checks and
compiler-free changed input/seed reuse. 541 shared checks, 222 target checks
(one skip), and 494 native fixtures (66 unsupported) pass. Twelve rows record
separate public, prepared host and native sequence-event cost; no isolated
kernel speedup or general closure is claimed. Independent/shared-LHS/dynamic
batching, linear transpose and full-unit/PR delivery remain open.
Evidence: benchmarks/baselines/rocm_shared_scaled_batch_20261007/README.md.

## Independent-batch offset foundation

Native gfx1201 unbounded/vector and ragged/scalar fragment paths preserve
dynamic memref offsets through Target-to-ROCDL. This permits native batch
subviews without Python operand replication. Graph/Schedule batch projection,
scale offsets, ABI/geometry, owning-device proof and timings are still open.
Evidence: benchmarks/baselines/rocm_batch_offset_foundation_20261007/README.md.

## Native independent-RHS and shared-LHS integration

Native named static typed FP8/MXFP8 independent-RHS/shared-LHS batches retain
logical matrix/scale storage and per-batch M through Schedule, Tile and Target.
Native grid-z memref views offset matrices, scales and output; shared LHS
storage is reused without Python replication or batch launch loops.
67 gfx1201 tests cover primal/JVP numerics, changed per-batch scales, finite
differences and warm compiler-free execution. The wider LDS shape initially
failed because image projection was incorrectly gated; the repair preserves
checked logical capacity while permitting projected runtime image dimensions.
Twenty-four correctness-gated primal rows separate public, prepared host and
native sequence-event costs. Dynamic/nested batching, transpose/composed AD,
sibling physical parity, generic full-unit closure and PR delivery remain open.
Evidence: benchmarks/baselines/rocm_independent_scaled_batch_20261007/README.md.

## Immutable native scaled-program host binding

Bounded complete-content ABI binding reuse passes 68 gfx1201 default device
tests and 386 shared gates. Six image-identical paired windows show 7-13%
lower FP8/MXFP8 primal and 15-20% lower FP32 scale-JVP public cost.
These are host costs; native checks and device execution ownership remain
unchanged. Folded MXFP4/NVFP4 owners need separate attribution. Generic
integration, full-unit and PR delivery remain open.
Evidence: benchmarks/baselines/rocm_native_plan_binding_20261007/README.md.

## Public typed leading maps and composed scale JVP

Public FP8/MXFP8 vmap now projects static leading shared-RHS, independent-RHS
and shared-LHS intent into native Graph/Schedule/Tile/Target execution.
36 gfx1201 primal cases and 36 timing rows cover small, ragged M200 and wider
LDS envelopes, with one native step per mapped primal. Mapped FP32 scale-JVP
preserves differentiation intent and signature-cached scalar-map/reference
certificates; 72 primal/JVP cases and 12 composed timing rows pass. Native JVP
now checks source bounds before capture/compilation; 59 owning scalar/mapped
JVP cases and 44 frontend tests pass. Source receipts distinguish each change.
Generic dynamic/nested/nonleading maps, storage AD and linear-transpose closure
remain open. Evidence: benchmarks/baselines/rocm_public_typed_vmap_20261007/README.md
and benchmarks/baselines/rocm_public_mapped_jvp_20261007/README.md.

## Diagnostic full-unit integration repairs

A later non-slow run finished with 21,774 passed, 7,502 skipped, 874 deselected
and seven failures while engineering continued. Native tangent forward/proof
registration, recorder consumers and ROCm admission defects are now repaired:
110 focused tests pass with 90 gated skips. The two generic batching/transpose
closure failures remain real and unchanged. The final source needs a fresh
full-suite gate; earlier pass counts do not establish it.
Evidence: benchmarks/baselines/compiler_unit_contract_repairs_20261007/README.md.

Current origin/main remains the merged PR892 base. The unpublished aggregate
has 229 tracked changed files and 686 status entries at this inspection;
curated dependent delivery is still required.

## Independent source-bound delivery

Draft PR #893 (https://github.com/gstoner/tessera/pull/893) isolates native JVP
source-bound enforcement against current origin/main, without the aggregate
dependencies. Commit 0e82c48ec contains one functional line, six positional/
keyword early-rejection tests, and four backend assessments. Host WSL gates
pass 377 tests with two gated skips; 32 generated documents are in sync.
The PR is pushed as a draft. Its publication does not deliver the other
aggregate changes or close generic batching/transpose/AD obligations.

## Native pinned checkpoint conversion evidence

The real q_proj and gate/up recorder now executes the native Graph ingest
package and uses its verified outputs for the native consumer. Exact gfx1201
code/exponent parity and independent block loss checks pass before/after timing;
both consumer outputs match decoded weights exactly. Native conversion event
medians are 3.23/19.21 ms, separately from consumer events and wall time.
BF16-relative weight loss remains 14.99/14.97%; whole-model quality and general
resident model-shape composition remain open. Shared focused gates pass 359
tests with seven skips. Evidence:
benchmarks/baselines/rocm_native_checkpoint_20261007/README.md.

## Native static multidimensional batch integration

Ordinary typed FP8/MXFP8 JIT now retains two leading logical batch dimensions
through native Schedule/Tile/Target and checked program ownership. 102 final
gfx1201 primal/JVP regression cases, 24 separate timing rows, 381 focused shared
checks and 48 target/ABI checks pass. Native fixtures pass 649/715 with 66
unsupported. Nine RTX 5070 NVFP4 regressions establish shared-Schedule parity
for its existing profile. Native and frontend counterexamples reject permuted
equal-product batch prefixes. Generic batching/linear-transpose states remain
open: nested public map projection, dynamic/nonleading axes and wider AD still
need integration. Evidence:
benchmarks/baselines/rocm_multidimensional_scaled_batch_20261007/README.md.

## Nested public typed maps

Two matching public leading maps now preserve scalar semantics, source bounds,
logical rank-four prefixes and independent owners through native compiled
FP8/MXFP8 primal and FP32 scale-JVP execution. 102 gfx1201 device cases,
24 separate primal timing rows, 415 shared tests (16 gated skips) and 11
existing RTX 5070 public map regressions pass. This closes the named two-map
projection gap; mixed policies, dynamic/nonleading/deeper maps and generic
linear-transpose AD remain open. Evidence:
benchmarks/baselines/rocm_nested_typed_vmap_20261007/README.md.

## Typed scale-transpose engineering foundation

Independent every-coordinate finite differences pass 12 ragged group cases.
Existing native gfx1201 nested JVP duality passes 12 cases across shared-RHS,
independent and shared-LHS policies, KN/NK storage, N129/K1536 and warm
compiler/reference-free reuse. Native structured reverse construction is
being integrated; Schedule/Tile program packaging, owning VJP execution and
timing remain open. No generic closure or sibling physical claim.
Evidence: benchmarks/baselines/scaled_product_transpose_foundation_20261007/README.md.

## Native scale-transpose program ownership contract

Paired AD now exports actual generated tensor/SCF regions with captured-input
order, frontend argument permutations, requested gradient order and SSA
byte/lifetime metadata. 55 native export/package regressions, 75 native AD
fixtures and 11 existing RTX 5070 NVFP4 map cases pass. This extends the native
program contract; Schedule/Tile reduction member execution, image/ABI
packaging, gfx1201 physical reverse numerics and timing remain open.
Evidence: benchmarks/baselines/scaled_transpose_program_export_20261007/README.md.

## Native scale-transpose Schedule/Tile integration

Native structured reduction members now carry the actual generated adjoint
body into a sealed Tile GPU region and ROCm Target carrier. Both carriers
provide MLIR symbol-table ownership for the nested GPU module. Schedule-to-Tile
passes after the symbol-trait repair. Serialized GPU function verification
exposed an additional property/attribute mismatch; the verifier now uses typed
GPU function accessors, with the matching rebuild and package lane in progress.
No reverse HSACO execution, owning gfx1201 VJP numerics, timing or public JIT
reverse closure is claimed yet. Original failing logs remain in host scratch.

## Native scale-VJP owning execution baseline

The symbol-table and GPU property repairs now pass 15 native lowering/image
checks and 55 export/program regressions. Twenty-four compiler-owned reverse
packages pass actual gfx1201 float64-oracle numerics, changed-input replay and
stale-generation refusal. Maximum absolute error is 1.107e-5. Separate native
launch-window medians are 0.173–9.789 ms; prepared host medians are
0.517–10.530 ms across different requested gradients and one/two members.
These are initial serial-reduction baselines, not a speedup or isolated ISA
measurement. Public JIT reverse integration, broader AD/maps, sibling physical
proof and generic/full-unit closure remain open. This supersedes the preceding
Schedule/Tile image-validation failure. Evidence:
benchmarks/baselines/rocm_native_scaled_vjp_20261007/README.md.

## Public native scale-VJP integration

Public @jit reverse now uses native Graph paired AD, Schedule/Tile/ROCm/LLVM
images and checked HIP ownership. 72 actual gfx1201 scalar/one-map/two-map
cases pass independent scale-adjoint numerics, gradient order and changed-
cotangent compiler-free replay. 397 focused shared gates and 11 existing RTX
5070 NVFP4 map regressions pass. 48 public warm timing rows record
0.901–11.067 ms wall-time medians including frontend/upload/native/readback.
This supersedes the prior public reverse gap for the named static FP32-scale
profile; general dynamic/composed/storage AD, sibling reverse execution,
serial-reduction performance and generic/full-unit/publication remain open.
Evidence: benchmarks/baselines/rocm_public_scaled_vjp_20261007/README.md.

## Native scale-VJP wave reduction candidate

An explicit native Schedule option partitions the proved additive outer
reduction across 32 lanes while retaining K-group dot products. 372 native
export/image/package/registry gates pass. 24 same-compiler paired gfx1201
rows pass independent numerics and changed-input replay; alternating native
launch-window ratios span 1.752–18.379 (median 7.597). Serial remains default.
Wider regimes, repeated controls, public candidate timing, selector policy
and generic/sibling/full-unit/publication closure remain open.
Evidence: benchmarks/baselines/rocm_scaled_vjp_wave_20261007/README.md.

## Compensated scale-VJP widened regimes

The prior serial/wave schedules each failed six shared-RHS SB-gradient long
cases. Native compensated FP32 carried reductions and innermost contribution
partitioning repair all 72 tiny/ragged/long paired cases at unchanged bounds.
Median serial/wave event-window ratios are 1.140/19.349/14.669; the tiny minimum
is 0.974, so serial remains default. Public candidate timing, general AD,
sibling execution, full-unit green and publication remain open.
Evidence: benchmarks/baselines/rocm_scaled_vjp_compensated_20261007/README.md.

Public compensated serial/wave integration subsequently passes 144 owning
gfx1201 tests and 48 alternating compiler-free public A/B rows. Schedule binds
immutable cache identity and execution receipts. Median paired public ratio
is 3.934; wave medians span 0.740-1.575 ms. These include frontend/ABI/transfers,
not isolated kernel timing. 362 focused native/shared gates pass. Serial stays
default; wider public regimes, general AD, siblings, full-suite and PR delivery
remain open. The compensated evidence packet contains both measurement domains.

## Ragged public reverse admission and wider regimes

The recorder's floor-sized scale storage was repaired, then six frontend
failures exposed a production admission gap. Native scale reverse now validates
logical ceiling-sized groups independently from aligned primal WMMA schedules.
73 frontend tests, 361 shared/native gates and 11 RTX 5070 map regressions pass.
96 gfx1201 ragged/long public A/B rows pass independent numerics and warm replay.
Median paired public ratios are 4.043/10.321; serial remains default.
Generic closure, dynamic/composed/storage AD, siblings and delivery remain open.
Evidence: benchmarks/baselines/rocm_public_scaled_vjp_regimes_20261007/README.md.

## Integration drift repair and fresh full-unit result

The latest Super-Bear WSL non-slow unit run has 21,920 passed and five failures.
Three inventory/evidence drifts are repaired with 780 focused passes.
Two generic scaled-matmul batching/transpose closure failures remain open.
The scale-transpose packet binds gfx1201; generic ROCm alias coverage remains
blocked. Evidence: benchmarks/baselines/compiler_drift_repair_20261007/README.md.

## Packed resident short-M integration

NVFP4-SHORT-M-2026-10-07 extends only the compiler-owned packed folded route
to positive M. 24 host typing/static/runtime-package tests, 16 gfx1201 public
JIT tests and 10 ragged static/runtime owning cases pass. Existing device
regressions pass 35 cases; fresh-process replay passes separately after an
explicit native image library environment repair. 51 host regressions pass.
Twelve shapes pass correctness before/after separate stage and wall timings.
The matching compiler was built on Super-Bear and replayed on Tajasaurus; the
isolated HIP owner was built on Tajasaurus. See exact hashes in the packet.
Unpacked/legacy routes remain M>64. General performance, layout, AD, model
quality, sibling proof and full-unit/publication closure remain open.
Evidence: benchmarks/baselines/rocm_nvfp4_short_m_20261007/README.md.

## Deeper leading maps and reverse source bounds

DEEP-LEADING-MAPS-2026-10-07 removes frontend two-map and native rank-four
scale-transpose ceilings for the static gfx1201 typed profile. Exact prefix
types, capacity checks and matching policies remain mandatory. 60 owning
gfx1201 primal/JVP/VJP cases pass; 18 paired public/native timing rows retain
independent measurement domains. Native regressions pass 89 tests.
Reverse source constraints now precede capture: 15 cross-backend host controls,
72 RTX 5070 attention/VJP cases and 11 owning NVFP4 map cases pass.
Generic batching/transpose states remain partial/planned; dynamic, mixed,
nonleading, storage AD, sibling scale execution and delivery remain open.
Evidence: benchmarks/baselines/rocm_deep_leading_map_20261007/README.md.

## Fresh frozen full-unit result and native resident integration

The latest Super-Bear WSL unit sweep finishes with 22,016 passed, 7,529 skipped,
874 deselected and exactly two failures: generic scaled_matmul batching and
transpose closure. Their assertions remain unchanged. This snapshot precedes
native resident-owner integration and is not its full-suite certificate.

Native resident producer-to-matmul integration retains compiler images and
uses one C++ call for producer/consumer submission and completion. Device
capacity/context/alias checks and read-only input/writable result distinctions
are native/ABI-owned. 57 new exact-device cases and 137 existing native owner
regressions pass. The initial 209-test lane bound the candidate runtime but
original Python because unit conftest reordered sys.path; the guarded candidate
adapter rerun is recorded separately. 48 same-image A/B rows pass numerics.
Receipt profiling identifies repeated descriptor serialization as the initial
regression; preparation now owns those immutable hashes. Median control/native
resident wall ratio is 2.518, with one 2.4% regression. No isolated-kernel or
selector claim is made. General composition, AD, padded resident pitches,
asynchronous lifetime and publication remain open.
Evidence: benchmarks/baselines/nvidia_native_resident_owner_20261007/README.md.

## Native resident padded RHS

Positive row/column RHS pitch and offset views now execute through the shared
native owner. Native physical-span capacity/alias checks precede submission;
the dynamic consumer retains its existing LDB scalar ABI. 249 candidate gates
and 24 changed-value/active-shape replay cases pass on RTX 5070.
48 same-image A/B rows pass independent numerics and separate stage events;
median control/native call ratio is 2.514, with one 2.1% regression.
Matching aggregate runtime validation passes 581 focused tests and 48
canonical numerical timing rows (median control/native wall ratio 2.440;
worst regression 3.62%). Generated documents and Graphify completed.
Padded source/output/
residual, general AD/composition, asynchronous lifetime, generic closure and
publication remain open.
Evidence: benchmarks/baselines/nvidia_padded_resident_owner_20261007/README.md.

## Independent matrix/scale batch semantic foundation

The reference-only oracle supports independent four-operand broadcast prefixes.
All fifteen nonempty mapped/shared combinations, both transpose orientations,
ragged groups and every-coordinate scale duality are covered. Native frontend,
Graph verification, Schedule/Tile addressing and AD must still carry these
independent maps; no production admission or generic coverage state is promoted.
Evidence: benchmarks/baselines/scaled_independent_broadcast_foundation_20261007/README.md.

## Independent batch-prefix native scale reverse

Native Graph verification and scale adjoints now preserve independent prefixes
for matrices/scales, including singleton/shared reduction axes. 131 candidate
native tests and 382 registry/owning SM120 regression gates pass. All 68 serialized
gfx1201 reverse programs pass changed-input replay, stale-generation checks
and float64 gradient comparison (maximum error 7.896e-8). Separate native event
and host update/invoke/read baselines are recorded; no performance promotion.
Matching aggregate CMake rebuild, 589 focused gates and a fresh 68-case
canonical gfx1201 numerical/timing rerun pass. Public projection, primal
addressing, JVP, dynamic/nonleading/composed integration, larger benchmarks,
generic closure and publication remain open.
Evidence: benchmarks/baselines/scaled_independent_batch_native_reverse_20261007/README.md.

## Public independent-scale reverse integration

Public leading maps now preserve independent four-operand prefixes through
native scale transpose, Schedule/Tile, ROCm/LLVM images and checked HIP ownership.
584 integrated host gates and 261 gfx1201 device cases pass. All 180 separate
public/native timing rows pass independent numerics, changed-input compiler-free
replay and stale generation refusal. Maximum gradient error is 9.1122e-8.
Corrected public warm medians are 0.857-1.377 ms; native two-member per-program
launch-window medians are 0.015696-0.208303 ms. These are different measurement domains, not a speedup.
Independent-map primal/JVP addressing, dynamic/nonleading/composed/storage AD,
larger regimes, generic closure, sibling proof and publication remain open.
Evidence: benchmarks/baselines/rocm_public_independent_scale_reverse_20261007/README.md.

## Native independent-prefix primal and scale JVP

Native Schedule/Tile/Target addressing and image/ABI identities now retain all
four independent prefixes for the complete aligned K-group WMMA profile.
465 focused native gates, 82 shared checks and 14 owning SM120 regressions
(two rank-two/no-batch skips) pass. All 69 gfx1201 serialized primal/JVP
cases pass independent numerics, changed values, stale-generation checks and
separate event/host baselines. The final policy-corruption guard passes 74 native tests; all 69 corrected
final-compiler owning replays pass. Public admission, transposed A, partial scale
groups and general/dynamic/composed/storage AD remain open; generic closure,
full-suite and publication are unfinished.
Evidence: benchmarks/baselines/rocm_independent_scaled_primal_20261007/README.md.

## Native program event timing repair

The native HIP ABI returns per-program milliseconds averaged over repeats.
Three new independent-prefix recorders divided again. Their old device event
values are withdrawn; oracles and host wall timings are unaffected. Fresh
gfx1201 corrected runs pass 180 public reverse, 68 native reverse and 69 native
primal/JVP rows. A known-average host regression passes. Corrected receipt,
runtime-source and recorder identities are retained in all three packets.
No speedup or selector decision relies on the withdrawn event values.

## Public independent-prefix primal/JVP integration

Owner FRONTEND-IR-MEDIUM-1 / AD-RESIDUAL-EVAL-1 / ROCM-FP8-BLOCKSCALE-1;
sync INDEPENDENT-SCALE-BATCH-2026-10-07. Public leading maps preserve independent
matrix/scale prefixes. The primal descriptor binds its actual native member
image and geometry. 487 focused WSL host checks pass.
Exact RX 9070 XT proof passes all 90 public device cases and 90 separate timing rows. Changed-input warm calls disable compiler subprocesses. Public admission is now proved for the named aligned profile; transposed A, partial groups, dynamic/nonleading/composed/storage AD, generic closure, full-suite and publication remain open.
Evidence: benchmarks/baselines/rocm_public_independent_scaled_primal_20261007/README.md.

## Independent-prefix transposed-A integration — active

Owner FRONTEND-IR-MEDIUM-1 / AD-RESIDUAL-EVAL-1 / ROCM-FP8-BLOCKSCALE-1;
sync INDEPENDENT-SCALE-BATCH-2026-10-07. Native gfx1201 Schedule orientation,
scaled Tile carrier, column-major A tile.view, serialized program orientation
and public leading-map admission are being integrated. Sixty host frontend
projection cases pass. Native tests found and drove repairs to Schedule
verification and scaled Tile orientation propagation; matching rebuild is
active. This entry does not claim native/device numerical or timing proof.
Partial scale groups, wider layouts/composition and generic closure remain open.

## Independent-prefix transposed-A — named profile proved

Owner FRONTEND-IR-MEDIUM-1 / AD-RESIDUAL-EVAL-1 / ROCM-FP8-BLOCKSCALE-1;
sync INDEPENDENT-SCALE-BATCH-2026-10-07. Native Schedule/Tile/Target orientation,
column-major A tile.view and serialized orientation now execute through public
leading maps. 855 focused host checks and 80 final native package/corruption
checks pass. The initial owning-device lane-address error was repaired and
the final compiler rerun; failed-build timings are not evidence.
The aligned gfx1201 transposed-A profile now has 192 public numerical cases and 180 benchmark rows. Remaining work includes partial scale groups, dynamic/nonleading/composed/storage AD, generic closure, full-suite and publication.
Evidence: benchmarks/baselines/rocm_independent_transposed_a_20261007/README.md.

## Independent-prefix partial scale groups — active integration

Owner FRONTEND-IR-MEDIUM-1 / AD-RESIDUAL-EVAL-1 / ROCM-FP8-BLOCKSCALE-1;
sync INDEPENDENT-SCALE-BATCH-2026-10-07. Native static independent-prefix
register lowering now derives ceiling scale-group counts and bounds trailing
WMMA fragment loads. Aligned recipes and other physical families retain their
existing contracts. 120 host projection cases pass. Matching native build,
image checks, gfx1201 numerical proof and timing are still being validated;
this entry is not an execution or performance claim.

## Independent-prefix partial K32 groups — named profile proved

Owner FRONTEND-IR-MEDIUM-1 / AD-RESIDUAL-EVAL-1 / ROCM-FP8-BLOCKSCALE-1;
sync INDEPENDENT-SCALE-BATCH-2026-10-07. Native Schedule/Tile/Target derives
ceiling scale counts and bounds the last WMMA group, preserving isolated
partial accumulation and static plane/image identities. Direct Tile JVP
admission was repaired and the final compiler rerun. 120 projection, 76 native
partial/corruption and 308 final registry/event/audit checks pass.
Exact gfx1201 proof passes 260 numerical and 180 benchmark rows. The named partial K32 leading-map profile is proved; scalar JIT, other widths/LDS tails, general/dynamic/composed/storage AD, generic closure, full-suite and publication remain open.
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


## Scalar typed bounded planes — active integration
Owner FRONTEND-IR-MEDIUM-1 / ROCM-FP8-BLOCKSCALE-1;
sync SCALAR-SCALED-PLANE-2026-10-07. A shared native helper derives the bounded
plane profile for rank-two typed FP8 transposed-A or partial scale groups.
Schedule and serialized native program export consume the same derivation;
the semantic Graph retains rank-two operands with no batching attribute.
32 frontend projection cases pass. Matching compiler build, native artifact
checks and exact gfx1201 numerical/timing proof remain pending.
Dynamic/nonleading/composed/storage AD, generic closure and delivery remain open.


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


## Scalar scale reverse — named profile proved
Owner FRONTEND-IR-MEDIUM-1 / AD-RESIDUAL-EVAL-1;
sync SCALAR-SCALED-PLANE-2026-10-07. Scalar transposed A/B and ragged K4
scale reverse now have 12 exact gfx1201 role-order/changed-input checks and
four separate public/native timing profiles; maximum error is 3.43592e-8.
Matching existing SM120 NVFP4 regression passes 83 cases with two unsupported
skips. General composed/dynamic/nonleading/storage AD, sibling scale execution,
generic closure, fresh full-suite and publication remain open.
Evidence: benchmarks/baselines/rocm_scalar_scaled_plane_20261007/README.md.


## ROCm version-query metadata reuse — gfx1201 measured
Owner FRONTEND-IR-MEDIUM-1 / E2E-REAL-6;
sync ROCM-VERSION-METADATA-2026-10-07. 49 host checks and 40 owning gfx1201
scalar regressions pass. 16 actual package A/B profiles preserve compiler/
toolchain fingerprints, reduce subprocesses from 7 to 5 and measure median
uncached/cached ratio 1.1205. This is package metadata work, not kernel speed.
gfx1151/other-family proof, broader integration, full-suite and delivery remain open.
Evidence: benchmarks/baselines/rocm_version_query_cache_20261007/README.md.


## Current five-slice native snapshot revalidation
Owner ROCM-NVFP4-INGEST-1 / W1.1 / E2E-REAL-6;
sync FIVE-SLICE-CURRENT-SNAPSHOT-2026-10-07. Matching compiler revalidation
passes 171 combined NVIDIA host/device tensor and saved-LSE attention tests.
gfx1201 ingest/public resident gates pass 47 cases; the missing native-image
library binding is repaired and the fresh-process compiler-free replay passes.
Original failure and repair receipts remain separate. Wider integration,
performance closure, full-suite and reviewable publication remain open.
Evidence: benchmarks/baselines/five_slice_current_snapshot_20261007/README.md.

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

## Focused cache delivery — PR #894 draft

ROCM-LINKED-TOOL-IDENTITY-2026-10-07 is published separately at
https://github.com/gstoner/tessera/pull/894, branch
codex/rocm-linked-tool-identity, head 783890cec.
The main-based PR contains only compiler cache helpers, actual ELF tests,
an existing-route RMSNorm recorder and four-backend assessments. It does
not publish the older host-only converter or the full accumulated branch.
372 focused WSL checks pass with four hardware skips; 103 owning gfx1201
mixed route/dependency checks pass. Twelve package profiles pass numerics.
Cold-image median uncached/cached ratio is 1.2316 (7-to-5 subprocesses);
warm images show parity, median 1.0073 (3-to-3). Neither is a kernel gain.
The broader non-slow suite and isolated graph refresh remain live; keep the
PR draft pending final results. Remaining native five-slice delivery is open.

## NVFP4 native whole-Graph partition — artifact gate

Owner ROCM-NVFP4-INGEST-1 / FRONTEND-IR-MEDIUM-1 / E2E-REAL-6.
Sync NVFP4-NATIVE-PROGRAM-2026-10-08.
Native export now preserves the actual three-operation Graph, argument roles
and eleven buffer lifetimes; native member projection reaches Schedule/Tile/
ROCm Target. Eight partition/adapter checks and 391 shared native/registry
checks pass. Public JIT integration, gfx1201 numerical/lifetime replay and
separate kernel/end-to-end timing remain open; status is artifact_only.
Evidence: benchmarks/baselines/rocm_nvfp4_whole_graph_partition_20261007/README.md.

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

Named gfx1201 public execution is proved; the five-slice objective remains open.


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

## PR894 focused delivery

The isolated ELF/compiler identity repairs are pushed and ready for review at a73212f5820850a3e9842374f6b2fecfc06394ca. Matching-source full WSL suite: 20,419 passed, 7,387 skipped, 874 deselected. Final drift 287 passed; audit docs 11 passed; generated docs and graph refresh passed. GitHub CI is pending. Native five-slice delivery remains separate. https://github.com/gstoner/tessera/pull/894


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

This widens the frontend/AD route; generic primitive batching/transpose statuses remain open and their full-suite closure assertions remain intact.

## Recorder index for accumulated native slice evidence

These are entry points for the existing packets; listing a recorder supplies
navigation, not a new execution or performance claim. Each packet retains its
own architecture, source/compiler and timing-scope qualifications.

| Recorder | Purpose |
| --- | --- |
| benchmarks/baselines/scaled_independent_batch_native_reverse_20261007/record_device.py | Historical packet-local compiler-free reverse runner; retained for reproducing that snapshot. |
| benchmarks/nvidia/benchmark_native_sm120_tensor_partition.py | Actual native SM120 producer/consumer members, numerical checks and separate timing. |
| benchmarks/rocm/record_independent_batch_reverse_device.py | Compiler-free native scale reverse and separate event/wall timings. |
| benchmarks/rocm/record_independent_batch_reverse_packages.py | Native independent-prefix reverse package census. |
| benchmarks/rocm/record_independent_scaled_primal.py | Native independent-prefix primal/JVP package and device recorder. |
| benchmarks/rocm/record_public_independent_scale_reverse.py | Public independent-scale reverse and native program timing. |
| benchmarks/rocm/record_public_independent_scaled_primal.py | Public independent primal/JVP and averaged native event windows. |
| benchmarks/rocm/record_public_partial_scaled_groups.py | Static trailing K32 group numerical/timing checks. |
| benchmarks/rocm/record_rocm_version_query_cache.py | Package wall-time A/B for cold versus reused version metadata. |
| benchmarks/rocm/record_scalar_scale_reverse.py | Scalar scale reverse with public and native timing. |
| benchmarks/rocm/record_scalar_scaled_plane.py | Scalar bounded-plane public/native numerical and timing checks. |
| benchmarks/rocm/record_typed_primal_single_image.py | Cold package work comparison; device execution has a separate gate. |


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

## Native short-K dispatch experiment — 2026-10-08

Owner ROCM-FP8-BLOCKSCALE-1 / E2E-REAL-6.
Sync GFX1201-NATIVE-SHORT-K-2026-10-08.
Evidence: benchmarks/baselines/gfx1201_native_short_k_dispatch_20261008/README.md.
New device coverage: tests/device/rocm/test_fp8_blockscale_w8a8.py.
Recorder: benchmarks/rocm/record_gfx1201_interleaved_compiler_formats.py.

Native short-K fallback/image reuse passes on gfx1201; 423 host gates pass. FP8 K128 register windows improve on named shapes; byte-identical K32 FP8/MXFP8/folded MXFP4 and LDS controls do not establish gains. The aggregate program and full-unit generic batching/transpose closure remain open.

## Mixed nested scaled map integration — 2026-10-08

Owner FRONTEND-IR-MEDIUM-1 / AD-RESIDUAL-EVAL-1 / E2E-REAL-6.
Sync GFX1201-MIXED-NESTED-SCALED-2026-10-08.
Evidence: benchmarks/baselines/gfx1201_mixed_nested_scaled_20261008/README.md.
Fixtures: tests/unit/test_native_mixed_scaled_maps.py and
tests/device/rocm/test_mixed_nested_scaled_execution.py.
Recorder: benchmarks/rocm/benchmark_mixed_nested_scaled.py.

Named mixed leading policies now execute native primals and scale JVP/VJP. 22 gfx1201 device rows, 16 timed profiles and 151 shared/NVIDIA regression tests pass. General generic/dynamic/nonleading/composed batching/transpose closure and native publication remain open.


## Static native SM120 producer chain — 2026-10-08

Owner W1.1 / FRONTEND-IR-MEDIUM-1 / E2E-REAL-6.
Sync SM120-NATIVE-PRODUCER-CHAIN-2026-10-08.

Native Graph outlining carries RMSNorm → softmax → matmul through three
compiler-owned Schedule/Tile members. CUDA ownership uses bounded alternating
scratch and one checked stream, preserving the final producer buffer consumed by
typed matmul. Static FP16/BF16 prepared and resident numerical tests pass on
RTX 5070; portable replay forbids recompilation. The partition/replay lane has
103 passing tests and audit/diagnostic/pass gates have 307 passing tests.
Evidence logs: /home/angstorms/scratch/sm120-chain-focused-tests-20261008.log
and /home/angstorms/scratch/sm120-chain-drift-tests-20261008.log.
Separate producer/consumer/program benchmark packets, ordinary public JIT chain
coverage, dynamic chain capacity, broader composition, fresh full-suite closure
and focused PR delivery remain open. No performance claim or sibling physical
execution claim is made.


### Static chain public frontend and timing completion

SM120-NATIVE-PRODUCER-CHAIN-2026-10-08 now has four passing ordinary JIT/replay cases and four correctness-gated benchmark profiles. Public member receipts and ordered IR fingerprints include every producer. Separate resident stage CUDA events and prepared host-wall timings are saved in benchmarks/baselines/nvidia_native_producer_chain_20261008/README.md. Maximum absolute error 7.7983e-6. No A/B speedup, whole-program device-time or general composition closure is claimed.

Combined owning RTX 5070 public-JIT/partition/portable regression: 167 passed in 96.46 seconds. Evidence: benchmarks/baselines/nvidia_native_producer_chain_20261008/combined-regression.log.


### Static three-producer extension

Sixteen RTX 5070 cases prove LayerNorm → RMSNorm → softmax → matmul, including storage rounding, scratch alternation, plain/fused epilogues, portable replay and resident completion. Eight two/three-producer benchmark profiles pass independent before/after numerical gates. Legacy certificates now explicitly require native member/lifetime certificates for chain metadata. Fresh full WSL unit validation is running at /home/angstorms/scratch/five-slice-current-full-unit-20261008.log. Dynamic admission remains gated; no generic closure is claimed.

Expanded static-chain regression: 184 passing cases; one newly added certificate fixture had misordered positional fields, now corrected with both refusal cases passing in a focused rerun. Initial and repair logs are preserved in the chain evidence packet. Full-suite success remains unproven.


## Prepared attention native staging

PREPARED-ATTENTION-STAGING-2026-10-08 retains pinned staging and submits copies, forward and derivatives on the owned nonblocking stream. Sixty-four exact RTX 5070 checks and two independent-stream isolation tests pass. Control JVP fails the isolation assertion as expected; control backward already passes. Native arithmetic, images and public synchronous ABI are unchanged. Matched A/B remains pending; no performance promotion. Evidence: benchmarks/baselines/nvidia_prepared_attention_staging_20261008/README.md.


## Widened owning gfx1201 evaluation

GFX1201-SHORT-K-WIDE-2026-10-08 records 16 correctness-gated four-format rows over both M=200 shapes, ragged K2048 and fallback K2304. Fourteen image pairs are unchanged. M=200 LDS ratios remain near parity; register ragged/fallback changes require scoped attribution. No AITER or selector closure. Evidence: benchmarks/baselines/gfx1201_short_k_wide_four_format_20261008/README.md.

Prepared attention A/B recorder is ready and its refusal during live pytest validation was verified. Seven immutable artifacts and fresh counterbalanced library processes will be used after validation retires. No timing result exists yet.

## Integration drift repair — 2026-10-08

The recorder naming, legacy chain fixture and two Schedule package tests pass together in a fresh 140-test WSL lane. Schedule replay still owns native image construction; descriptor Graph ancestry remains explicit and changes when its diagnostic source changes. Audit document gates pass eleven tests. The latest aggregate sweep remains red (23,206 passes, six failures); these focused repairs do not replace a fresh full suite, and generic batching/transpose closure is still open.

Prepared attention staging matched A/B completed seven cases, with control/candidate host-wall ratios 1.040043–1.608286; numerical and independent-stream proof remains scoped to RTX 5070. See nvidia_prepared_attention_staging_20261008/README.md.

The isolated LDS short-K candidate completed sixteen gfx1201 four-format rows and is unpromoted due to the M200 N8192 K1024 regression. See gfx1201_lds_short_k_candidate_20261008/README.md.

## Public NVIDIA softmax alias native route — 2026-10-08

NVIDIA-PUBLIC-SOFTMAX-ALIAS-2026-10-08 connects the standalone public alias to native Graph/Schedule/Tile/PTX packaging and checked dtype ABIs, including resident FP32. Fourteen package and eighteen public/portable/changed-input cases pass; eighteen timed profiles record independent numerics and matching safe/ordinary image bytes. This closes the named static last-axis selection/refusal gap, not composed/dynamic/AD or all W1.1 producer chains. Evidence: benchmarks/baselines/nvidia_public_softmax_alias_20261008/README.md.

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

GFX1201-COMPOSED-SCALED-MAPS-2026-10-08 final focused registry/documentation follow-up: 554 host WSL tests pass after owning-generator dashboard repair. Full-suite generic closure remains open.

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

## Native bounded saved-LSE sequence contract — 2026-10-08

Owner E2E-REAL-6 / AD-RESIDUAL-EVAL-1.
Sync NVIDIA-ATTENTION-BOUNDED-SEQUENCES-2026-10-08.

Native checkpoint Graph verification and Graph/Schedule lowering now admit
symbolic Sq/Sk with explicit seven-dimension module capacity bounds. Fixed
batch, head counts and widths remain verified; capacities enter the sealed
Schedule hash and native contract. The first matching build passed 60 native
and static replay tests and twelve RTX 5070 same-image numerical/timing cases.
Final physical-buffer overflow refinement and saved-Graph import regressions
pass: 369 host gates and twelve same-image owning-device cases after rebuilding. This is native image proof only: public JIT/package ABI,
runtime extent guards, AD export, generation-bound residual ownership and
dynamic physical broadcast-bias carriers remain open. No full compiler or
performance closure is claimed.
Evidence: benchmarks/baselines/nvidia_bounded_attention_native_20261008/README.md.

## Checked bounded saved-LSE packages and residual generations — 2026-10-08

Owner E2E-REAL-6 / AD-RESIDUAL-EVAL-1.
Sync NVIDIA-ATTENTION-BOUNDED-PACKAGES-2026-10-08.

Native symbolic sequence dimensions and capacity bounds now survive artifact
decoding, sealed package identity, total min/max shape guards and serialized
runtime replay. Capacity and cross-buffer role failures precede CUDA loading.
Resident capture validates actual Q/K/V dimensions and privately owns that
generation's saved O/LSE and retained gradients. Compact gradients and LSE
cotangents preserve the native policy rather than dropping it.
173 initial package/static host regressions, 72 compact/seeded/pre-driver host
tests and 20 owning RTX 5070 bounded/async tests pass. Eight correctness-gated
checked-API timing arms preserve one image pair per bias policy across runtime
shapes. Final focused drift gates pass 581 host WSL tests; owning replay passes
20 tests. The refreshed eight-arm packet binds the final sources and tools.
Public JIT symbolic tracing and automatic native AD export remain open;
this is explicit scheduled Graph product package integration, not universal AD.
Evidence: benchmarks/baselines/nvidia_bounded_attention_packages_20261008/README.md.

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

NVIDIA-ATTENTION-DYNAMIC-BIAS-2026-10-08 final owning regression follow-up:
92 static tuple-AD/asynchronous/bounded-package RTX 5070 cases and eleven
final audit-document tests pass. Packet source/compiler/runtime fingerprints
match authoritative bytes. Full-suite generic closure and delivery remain open.


## Nested native NVFP4 leading-prefix integration

E2E-REAL-6 / NVIDIA-NVFP4-NESTED-MAPS-2026-10-08 extends the named SM120 contract through public nested maps, typed Graph MLIR, native Schedule flattening, Tile/Target/PTX and the checked existing ABI. Exact logical tuples survive package guards; equal-product prefix and scale permutations are rejected by direct native verification. RTX 5070 proof covers 24 nested cases plus scalar/single-map regressions. Evidence: benchmarks/baselines/nvidia_nested_nvfp4_20261008/README.md. Mixed/dynamic/nonleading batching and generic scaled-matmul transpose/AD closure remain open; no coverage flags are promoted.


## Matching-source gfx1201 publication gate

Current pre-publication source revalidation passes 113 owning gfx1201 composed primal/JVP/VJP and NVFP4 ingest tests after rebuilding core/ROCm tools and native HIP runtimes. Timestamp-preserving source transfer initially retained stale Make objects; the initial 41 failures and corrected terminal result are preserved with 4,931 input hashes and live device/tool fingerprints. Evidence: benchmarks/baselines/gfx1201_matching_source_20261008/README.md. No fresh performance, original-model quality or aggregate closure is claimed.
