# Native compiler contract revalidation — 2026-10-06

Owners: E2E-REAL-6 / W1.1 / FRONTEND-IR-MEDIUM-1.
The initial entries are source/test receipts after native metadata and type-contract
repairs. Subsequent benchmark sections retain their own measurement dates and
timing domains; no default performance promotion is claimed.

| Receipt | Host / envelope | Result |
| --- | --- | --- |
| nvidia_producer_attention.txt | Super-Bear, matching NVIDIA compiler, RTX 5070 sm120 | 107 passed |
| gfx1201_ingest_storage.txt | Tajasarus, matching ROCm compiler, live gfx1201, explicit device proof | 50 passed |
| gfx1201_math_movement.txt | Tajasarus, native math/widening/movement compiler and numerical gates | 106 passed, 2 other-architecture skips |
| gfx1151_math_movement.txt | Princess-Luna, live gfx1151, native math/widening/movement gates | 107 passed, 1 other-architecture skip |
| shared_contracts.txt | WSL host frontend, portable seals, metadata, movement, MXFP8 and dynamic result projection | 125 passed |
| registry_spec_gates.txt | WSL operation/dtype attributes, diagnostics, pass metadata, specification and log routing | 348 passed |
| native_ad_lineage.txt | Matching NVIDIA compiler alias/return-lineage checks plus host-free foreign-claim inventory | 1011 passed |
| cuda_replay_setup.txt | Existing CUDA replay tests with the owning CUDA environment | 11 passed |

The device files include compiler/host checks as well as tests gated by the
live owning GPU. Counts must not be interpreted as that many independent GPU
benchmark rows. gfx1151 does not establish gfx1201 parity, or conversely.
ROCm environment warnings concern missing pytest timeout/marker configuration;
the listed tests completed and returned success.

The earlier portable marker selection finished with 33 failures, 18271 passes
and 9019 skips before these drift repairs. That is a failed integration gate.
Focused repairs do not establish that the entire portable suite is now green.
The remaining scaled-matmul batching/transpose integration, dashboard/evidence
drift and exact-device test placement still require closure.

The full Python lint gate and zero-error mypy ratchet pass (625 source files;
baseline unchanged). The graph refresh could not run because graphify is
absent from the WSL host. Existing benchmark packets retain their own source
hashes and timing domains; these receipts do not re-date those measurements.

## NVIDIA device-test placement follow-through

The six hardware-marked tensor producer/epilogue functions now live in
`tests/device/nvidia/test_tensor_producer_epilogues.py`; host/compiler contract
checks remain in the unit files. Numerical assertions and parameterizations
are preserved. `tensor_producer_node_migrations.json` records their previous
and current paths without altering the historical NVIDIA-TEST-6 migration.
Super-Bear RTX 5070 replay plus the existing location/migration guards:
62 passed (60 owning numerical cases and two placement guards). Receipt:
`nvidia_device_test_relocation.txt`. Focused Ruff passes. No new timing,
kernel, ABI, schedule or performance promotion. General native scaled-matmul
batching/transpose integration and final full-suite/publication gates remain
open. Graphify update was attempted but its CLI is unavailable on this host.

## Scaled transpose shape contract and normalized NVFP4 admission

Generic rank-two frontend shape inference now respects both logical transpose
attributes, including dynamic free/contraction dimensions. MLIR verifies M/N
before optional-scale early exits. This is a semantic verification fix; generic
native batching, linear-transpose/AD products and broader scaled lowering are
still open. Existing packed physical contracts retain their independent gates.

The post-build scheduled NVFP4 benchmark exposed a target-registry mismatch:
normalization creates logical `nvfp4` A/B, while the shared ingest override only
admitted `uint8`. The SM120-only capability now admits that logical dtype at
rank two; native static K16/policy/scale checks still define its envelope.
Sibling target admission is unchanged. Manifest evidence records this named
SM120 descriptor route, without claiming a generic public scaled frontend.

Matching LLVM/MLIR 23.1.1 `tessera-opt` rebuild passed. Focused shape/native
checks: 45 pass; expanded registry/dtype/package contract gates: 540 pass;
capability gates: 34 pass; zero-error mypy. The scheduled package replay runs
three shapes [M,N,K]: [16,8,64], [33,19,129], [7,5,31], each with maximum
absolute error zero against the decoded independent oracle. Four owning native
package tests pass. `nvidia_nvfp4_post_verifier.json` records source/compiler
hashes, actual GPU report, descriptor ancestry and separate CUDA-event and
end-to-end samples (three samples, 20 repetitions, five warmups).
This is a post-build validation measurement with no matched baseline or speedup
claim; no kernel/ABI/schedule/selector change. Additional registry tests and
final generated-doc/full-suite gates remain pending.

Logical NVFP4 shape inference also now distinguishes unpacked Graph A[M,K]/B[K,N] from byte-container dimensions, covering all three ragged benchmark shapes. The expanded shape/frontend/package subset passes 95 checks. Manifest/fixture/audit gates pass 95 checks. Dashboard regeneration caught a duplicate SM120 manifest grain; the bounded evidence row now replaces the existing grain. Native general batching/transpose closure remains open.

## Native shared-RHS NVFP4 batching (W1.1)

`batching = "shared_rhs_rows"` admits a named static SM120 logical Graph:
A[B,M,K], shared B[K,N], A scales [B,M,ceil(K/16)], shared B scales
[ceil(K/16),N], and result [B,M,N]. Graph retains rank three. Native C++
Schedule derivation flattens B*M rows without copying; existing native Tile
NVFP4 carriers, NVIDIA lowering and LLVM/PTX produce one GPU launch. The
five-buffer ABI retains rank-three shape guards on A, A scales and output.
Malformed batch scales, changed logical axes, equal-byte-count reshaping,
row-product overflow and stale flattened M are rejected. A Schedule from
[B=3,M=7] cannot rebind to [B=7,M=3] just because flat rows coincide.

The driver now routes scale-bearing NVFP4 to its scale ABI and reuses the
already produced Schedule. Exact-device tests assert one Graph-to-Schedule
pass; native Schedule-to-Tile replay and source equality remain checked.
The existing rank-two authored scaled route also passes an independent oracle.
Matching native core and NVIDIA target builds passed. 16 host/compiler/device
checks pass (five owning numerical tests, with repeated batch input refresh).
Required registry/package gates: 560 pass. Mypy remains at zero errors.
Receipts: `nvidia_nvfp4_shared_rhs_batch_device.txt` and
`nvidia_nvfp4_shared_rhs_batch_mypy.txt`.

Recorder: `benchmarks/nvidia/benchmark_nvfp4_shared_rhs_batch.py`.
Packet: `nvidia_nvfp4_shared_rhs_batch.json`. Both arms pass independent decoded
FP64 comparison before timing. Three [B,M,N,K] rows [3,7,5,31], [2,17,19,129],
[5,9,11,64] all have max absolute error zero. Five samples, 50 repetitions,
ten warmups; actual RTX 5070/12.0/driver report and source/compiler hashes are
recorded. Native-batch event medians are 0.01049, 0.00933, 0.00970 ms; serial
per-member event sums are 0.02918, 0.01757, 0.04809 ms. Batch wall medians are
0.95083, 1.02271, 1.00575 ms; serial batch wall medians 2.82800, 2.04165,
5.07835 ms. The event sums and wall intervals are separate timing domains.
These small-shape results do not establish large-shape throughput, persistent
kernel selection or FP8/MXFP8/MXFP4 parity. No selector/default changed.

The lowering-rule axis records existence of the proved native consumers;
batching is partial. Generic vmap, independent-RHS batches, dynamic batches,
transpose/AD products, other storage types and sibling physical proof remain
open. This slice does not close those programs or the entire five-slice goal.

Final existing Schedule/tensor/manifest/audit/recorder regression subset: 287 passed, 49 architecture/compiler skips. Regenerated all registered derived views. Closure-gate replay: 18 passed, two failures, both identifying `scaled_matmul` as still open for generic batching and linear transpose. Receipt: `scaled_remaining_closure_gates.txt`. These gates remain enforced; the current branch is not full-suite green or ready for a final closure claim. Full Python Ruff passes. Graphify update was attempted; the CLI is unavailable on Super-Bear.


## Native NVFP4 static dimension arithmetic follow-through

The Graph verifier and native Schedule scale checks now compute ceil(K/16)
as K/16 plus a nonzero-remainder bit, avoiding signed overflow in K+15
for directly authored static dimensions. Native verifier regression cases
cover INT64_MAX K and an overflowing batch-row product. This changes
verification arithmetic only; launch capacity guards and physical schedules
remain unchanged. Compiler rebuild and focused native verification are
in progress; no new device performance evidence is claimed.

A fresh broader WSL integration run is active with output at
/tmp/tessera-five-slices-integration-current.log. It supersedes neither the
previous failed integration receipt nor the enforced batching/transpose gaps
until its terminal result is inspected.

Host argument/audit gates: 19 passed, eight native cases deselected until the matching rebuild completes. Ruff passes. Receipt: `nvfp4_overflow_host_gates.txt`.

Matching core/NVIDIA rebuilds completed. Native verifier/host checks: 16 passed;
RTX 5070 numerical and diagnostic/pass registry replay: 297 passed. Receipts:
`nvfp4_overflow_native_verifier.txt`, `nvfp4_overflow_device_registry.txt`.
No schedule or numerical algorithm changed.

Fresh broad integration result: 7 failed, 18322 passed, 9035 skipped in
288 seconds (`integration_current.txt`). Two enforced scaled-matmul transform
closure failures remain, plus public keyword drift, stale standalone dashboard,
and three CUDA replay failures in the unconfigured broad environment. Public
keyword/dashboard repairs and owning-runtime CUDA replay are being validated;
this remains a failed full integration gate.

The public scaled operation now declares transposeA/transposeB/batching,
matching Graph keyword classification. Its existing packed-folded CPU oracle
validates those options explicitly and rejects transformations outside its
physical profile rather than ignoring them. Generic/native transform closure
remains open. Standalone coverage was regenerated by its owning module.
Frontend keyword/reference/dashboard gates pass; receipt:
`frontend_keyword_repair.txt`. Configured owning CUDA SSM replay: 11 passed
(`ssm_configured_replay.txt`). These focused receipts do not turn the recorded
broad integration run green.


## Typed Python-source NVFP4 batch execution

Typed Python source calling public scaled_matmul now reaches native
Graph/Schedule/Tile/PTX. The source route exposed unsigned annotation drift;
ui8/16/32/64 now round-trip through canonical uint names without changing
target dtype admission. The source/device/dtype subset passes 174 checks.
Ruff passes. The recorder now compiles the typed source and uses checked
descriptor output names in both arms. Nine owning numerical cases preserve
independent FP64 comparison, refreshed inputs and one Graph-to-Schedule call.

Packet: nvidia_nvfp4_python_frontend_batch.json. Three rows are oracle-exact
on RTX 5070. Source/compiler hashes and separate event/wall timing are
recorded. No selector/default changed. Independent/dynamic RHS batches,
large-shape throughput and general frontend/AD transforms remain open.

| B,M,N,K | Native event ms | Serial event sum ms | Native wall ms | Serial wall ms |
|---|---:|---:|---:|---:|
| 3,7,5,31 | 0.00972 | 0.02938 | 1.00332 | 2.99894 |
| 2,17,19,129 | 0.00947 | 0.01859 | 0.98375 | 2.03087 |
| 5,9,11,64 | 0.00943 | 0.04830 | 1.00658 | 4.93497 |


## Independent-RHS NVFP4 native layer (integration in progress)

A distinct ten-argument native Tile ABI adds rows-per-batch and batch count
to packed A/B, scale A/B, D and M/N/K. Grid Y reserves whole M16 tiles for
each batch, so a ragged tail never reads another batch RHS. Native pointer
offsets select per-batch A/B/scales/output. The C++ launcher validates the
flat-row product, byte capacities, scalar bounds and CUDA grid-Y limit before
allocation; event timing uses the same checked geometry. The legacy eight-
argument rank-two/shared-RHS launch path remains separate. No Python batch
loop or kernel construction is introduced.

Matching NVIDIA compiler/runtime builds pass. The new native Tile fixture
passes FileCheck. RTX 5070 tests: 12 passed, including three independent-RHS
shapes with distinct per-batch scales, refreshed inputs, independent FP64
comparison and invalid geometry that leaves output unchanged. Existing
rank-two/shared source/manual packages replay in the same test run.
Receipt: nvfp4_independent_native_device.txt.

This proves native Tile/LLVM/PTX execution only. Graph/Schedule derivation,
serialized package ABI, frontend admission and matched benchmark packets are
still pending. Generic batch/AD closure remains open; no default or selector
changes, and no sibling physical evidence is inferred.


## Independent-RHS Graph/Schedule/package completion

The named static independent_rhs policy now verifies logical A[B,M,K],
B[B,K,N], A scales[B,M,ceil(K/16)], B scales[B,ceil(K/16),N] and
D[B,M,N]. Native C++ Schedule derives per-batch Tile launch semantics and a
batch-sensitive digest. The distinct checked ten-argument ABI carries five
rank-three buffers and M/N/K/BatchRows/BatchCount. Launch and event timing
reject scalar rebinding against the compiled envelope before copying. The
existing shared/rank-two ABI retains its original geometry and entry.

Matching core/NVIDIA builds pass. Typed Python-source Graph to native package
and owning RTX 5070 oracle replay: 43 passed. Shared op/dtype/pass/diagnostic/
ABI regression gates: 561 passed. Mypy has zero errors in four changed source
modules and module lint passes. Receipts: nvfp4_independent_package_device.txt,
nvfp4_independent_registry.txt, nvfp4_independent_mypy.txt.

The recorder supports --batching independent_rhs. Packet:
nvidia_nvfp4_independent_rhs_batch.json. Five samples, 50 repetitions, ten
warmups; both native batch and serial per-member packages pass an independent
FP64 oracle before timing. Actual GPU and compiler/runtime/source hashes are
recorded. Event sums and end-to-end wall intervals remain separate domains.
These small shapes establish batch-launch amortization, not large throughput,
generic vmap/dynamic batches, transpose/AD or sibling physical support.
No selector/default changed.

| B,M,N,K | Native event ms | Serial event sum ms | Native wall ms | Serial wall ms |
|---|---:|---:|---:|---:|
| 3,7,5,31 | 0.01068 | 0.02881 | 0.97934 | 2.91068 |
| 2,17,19,129 | 0.00881 | 0.01731 | 1.02758 | 2.03412 |
| 5,9,11,64 | 0.00915 | 0.04691 | 1.06654 | 5.39459 |

Independent batch manifest/audit/recorder gates: 86 passed. Canonical plan ownership checker passes. General scaled batching/transpose closure remains enforced and open.


## Shared source replay on owning ROCm devices (2026-10-06)

Matching LLVM/MLIR 23.1.1 core compiler rebuilds passed after syncing the
complete programming-model source and its Schedule/Tile schemas. The owning
GFX1201 replay passed 217 tests; the owning GFX1151 replay passed 169 tests.
Receipts: rocm_gfx1201_shared_replay.txt and rocm_gfx1151_shared_replay.txt.
The corresponding shared_build_identity files record live architecture,
compiler version and compiler/source hashes. This revalidates existing ROCm
ingest/math envelopes against the shared changes; it does not admit the
NVIDIA independent-RHS batch ABI on ROCm or establish a new timing result.

Source control-flow comparison projection now updates structured inferred
types as well as the printed result type before emitting an i1 comparison
and the explicit f32 mask conversion. Native source/control-flow, exception
VJP and checkpoint broadcast replay passed 75 tests with two skips. Receipt:
source_comparison_type_repair.txt. The checkpoint physical-mismatch test
accepts the current compact-gradient projection arguments and still proves
that mismatched physical policies are rejected before package compilation.

Configured broad integration is not green: 607 failures, 20396 passed and
6383 skipped. Missing x86/EBM/Clifford/ROCm build components and missing clang
in the previous LLVM_BIN path account for environment failures; genuine
scaled batching/transpose closure and other failures remain under review.
A matching build with x86/EBM/Clifford enabled is underway.


## GFX1151 FP16 softmax capability parity

The f32-only capability override incorrectly excluded FP16 accepted by the
existing native Schedule/Tile package. The bounded last-axis FP16 capability
is restored with f32 accumulation. Owning GFX1151 replay: 67 passed, including
retired/native bitwise output comparison and an independent float64 oracle.
Receipt: rocm_gfx1151_softmax_parity.txt. No BF16 or sibling admission changes.

Packet rocm_gfx1151_softmax_parity.json records nine alternating serial
trials with 100 HIP-event repetitions across eight f32/f16 rows. All numerical
and 10% non-regression gates passed. Retained/compiler device-time speedups
range 0.991 to 1.029; end-to-end wall speedups range 1.730 to 4.194. This is
launch/allocation/copy-inclusive host overhead improvement, not faster kernel
execution. No default selector changed.

Bridge identity tests now explicitly model an unloaded runtime library; a
previous test worker could retain the real loaded CUDA image and bypass the
fixture locator. Actual loaded-image identity semantics are preserved. Shared
identity/audit/diagnostic/pass metadata gates: 342 passed. Source comparison
mypy has zero errors and lint passes. Graphify CLI remains absent on owning
WSL; graph refresh is not claimed.

FP16 capability follow-through: 257 capability/dtype/op/manifest/audit gates passed.
All registered derived documents regenerated. Matching x86/EBM/Clifford/NVIDIA
compiler and CUDA launcher targets rebuilt successfully. W1.1 Latest chronology
was repaired; canonical compiler plan checker passes. Matching-build replay passed 646 tests with 518 skips (including x86 device
rows that lack the runtime image). Receipt: multibackend_configured_replay.txt.
This is no aggregate or skipped-device closure claim.


## Finished multi-backend integration follow-through

The completed LLVM/MLIR 23.1.1 assertions build with NVIDIA/x86/EBM/Clifford
passed 120 fresh RTX 5070 producer, saved-LSE forward/backward, recompute and
independent-RHS NVFP4 tests. Receipt: nvidia_multibackend_device.txt. These are
numerical/compiler/lifetime checks, not newly measured performance rows.

Full integration returned 12 failed, 21044 passed and 6330 skipped. Nine
failures require absent ROCm passes; matching ROCm replay on Tajasaurus
passed 231 host/compiler contracts (193 hardware rows deselected), recorded
in rocm_target_integration_contracts.txt. A remaining GEMM identity fixture
retained a library loaded by another test; it now models its unloaded image
explicitly, with positive loaded-image identity coverage. Identity gates:
60 passed, two skipped, loaded_gemm_identity.txt. Generic scaled batching and
transpose closure remain real open obligations. This is not a green aggregate
result. The next matching compiler build includes ROCm compiler passes but
adds no HIP execution claim on Super-Bear.

Direct native NVFP4 policy reproduction found that three extra numeric/scale
fields were accepted and dropped while the valid positive control compiled.
Receipt nvfp4_named_policy_before.txt is deliberately red. Graph verification
and native Schedule derivation now require the exact named dictionary fields
already required by Python packaging. Matching rebuild and post-fix replay
are pending; no numerical or performance change is claimed yet.


## Native named-policy and complete compiler setup follow-through

The matching assertions-enabled LLVM/MLIR 23.1.1 compiler now includes NVIDIA,
ROCm, x86, EBM and Clifford passes. Super-Bear has a relocated AMD driver,
OCML/OCKL/OCLC bitcode and linker subset in
/home/angstorms/scratch/tessera-validation-sdk/rocm-core. Its llvm symlink
preserves LLVM toolkit lookup. The driver reports AMD clang 23.0.0git at
8f497e0992fb7513f7f78a6f6b6f1056c375e961; it selects device libraries, while
Tessera Graph/Schedule/Tile compilation remains the pinned 23.1.1 assertions
build. No GPU runtime or ROCm execution claim is added on the NVIDIA host.
Host-only ROCm package replay passed ten tests (fourteen hardware rows
deselected). Receipt: rocm_sdk_package_replay.txt.

Native policy rejection and owning NVFP4 execution passed within the earlier
459-test result; that aggregate still had the now-resolved AMD toolkit setup
failure. The final combined regression run passed 460 tests with 16 skips, including
all three named-policy regressions and owning NVFP4 batch execution. Receipt:
alltarget_named_policy_replay.txt. Build/source/tool
identity: alltarget_named_policy_build_identity.txt. Required generated views
regenerated and changed-source lint passed. Generic scaled batching/transpose
closure, complete integration and publication remain open.


## Final integration evidence assembly

The completed named-policy audit/diagnostic/pass-metadata gate passed 303
checks; receipt: named_policy_audit_gates.txt. Owning generators completed;
receipt: named_policy_generated_docs.txt. These receipts establish their
focused gates, not full integration closure.

alltarget_named_policy_tool_versions.txt records the live core compiler,
AMD compiler and linker versions plus OCML/OCKL bitcode hashes, complementing
the binary/source hashes in alltarget_named_policy_build_identity.txt.
Graph refresh was attempted on owning WSL; graphify_refresh_attempt.txt
records that the CLI is absent. No refreshed knowledge-graph claim is made.

The public vmap implementation still scans and stacks in Python, while native
scaled-matmul Schedule admission currently accepts the named static
shared_rhs_rows and independent_rhs NVFP4 profiles. Those native profiles do
not prove generic vmap, dynamic batches or linear-transpose integration.
The enforced zero-open closure gates therefore remain unresolved.


### Public low-precision frontend boundary

The owning NVFP4 batch device tests build typed Python-source Graph modules,
then use canonical native compilation and checked runtime launch. They do not
exercise ordinary JitFn host-array dispatch. Current _try_native_descriptor_call
admits NVIDIA FP16/BF16 matmul and attention, not requests_nvfp4_matmul.
The physical uint8 packed bindings also have different dimensions from logical
NVFP4 Graph tensors. Connecting ordinary JIT requires an explicit logical
storage/physical-buffer binding contract and native shape checks; treating
packed uint8 arrays as ordinary logical NVFP4 tensors would lose that contract.
This frontend integration remains open alongside generic batching/transpose.


## Completed all-target integration diagnosis

The full host WSL suite completed: 21524 passed, 5862 skipped, five failures
in 399.49 seconds. alltarget_full_integration.txt retains the complete result.
Two failures are enforced generic scaled batching/transpose closure gates.
The remaining three were diagnosed and replayed: integration_diagnosed_fixes.txt
records 72 passed and seven skips, including the ROCm lit suite.

The native MXFP8 identity fixture now rejects unsupported K128 rather than the
supported K64 recipe. Added f32/BF16 checks prove distinct K32/K64 identities,
whole-slab runtime-K reuse, and partial-slab rejection. The ROCm lit driver was
built against the matching compiler. Its split-K fixture now supplies required
problem_k for the valid pair, checks the current typed-route-only diagnostic,
and explicitly retains missing-problem-K refusal. The isolated SDK includes
the actual AMD hipcc driver, official clang/clang++ symlinks and HIP version
metadata; hipcc reports HIP 7.15.26333 and AMD clang 23.0.0git. This repairs the
compiler-presence gate without changing production toolchain gates or claiming
ROCm device execution on Super-Bear. No kernel, ABI or selector changed.

These focused fixes do not constitute a green full-suite rerun. Generic
batching/transpose, public logical NVFP4 storage binding and publication remain
open. Existing architecture-specific numerical and timing packets retain their
original source/tool identities and dates.

## Ordinary logical NVFP4 JIT integration

Sync NVIDIA-NVFP4-LOGICAL-JIT-2026-10-06; owner W1.1 /
FRONTEND-IR-MEDIUM-1 / E2E-REAL-6. NVFP4Tensor retains positive logical
rank-two/three dimensions, compact uint8 storage and the packed K axis.
Tracing reads those logical facts without decoding packed values. The native
Graph/Schedule/Tile/Target/PTX route and eight/ten-argument descriptors remain
the computation and physical ABI authorities. Matrices are packed low nibble
first with ceil(K/2) bytes; scales retain their existing UE4M3 K16 operands.

Ordinary JIT now executes named rank-two, shared-RHS and independent-RHS
static profiles. Device/host storage checks: 18 passed; regression gates:
460 passed; manifest/tracing drift: 115 passed, 23 skips. All three affected
Python sources type-check. Wrong matrix packing axes and invalid scale storage
are rejected before compilation. Warm calls change scales without invoking
eager execution or recompilation. Named-profile source snapshots and exact
RTX 5070/compiler/runtime identities accompany six oracle-exact benchmark
rows in nvidia_nvfp4_logical_jit.json. Cold public call, warm public wall and
resident CUDA-event samples are separate domains; no speedup is inferred.

The exact SM120 numerical fixture is registered without changing sibling
capabilities or generic primitive coverage. Subsequent manifest notes identify
this bounded frontend proof. The earlier public-binding gap above is now
resolved for these named static profiles. Generic vmap/dynamic batches,
transpose/AD, sibling physical storage binding, final integration and
publication remain open. No physical selector or kernel strategy changed.

## Public native NVFP4 vmap integration

Sync NVIDIA-NVFP4-NATIVE-VMAP-2026-10-06; W1.1 / FRONTEND-IR-MEDIUM-1 /
E2E-REAL-6. The public autodiff vmap API now projects leading batch intent
for a primal SM120 direct scaled product into existing shared-RHS and
independent-RHS Graph/Schedule/Tile packages. It owns a separate JIT cache,
preserves the scalar owner, validates logical ranks/axes/extents and uses one
native launch per call. No member slicing, stacking or Python launch loop.
Warm calls refresh scale inputs without eager execution or recompilation.

Regression receipt nvfp4_native_vmap_regression.txt: 367 passed, two rank-two
non-batch parameterizations skipped. Packet nvidia_nvfp4_native_vmap.json
records four oracle-exact RTX 5070 rows, seven public-wall and resident CUDA-
event samples per row, with compiler/runtime/source identities. Timing domains
remain separate; no selector or kernel strategy promotion. Existing logical-JIT
packets retain their original snapshots. General/composed/dynamic batching,
transpose/AD, sibling physical consumers and aggregate publication remain open.

Final audit/registry gates: 284 passed. Owning generated documents regenerated; three changed Python modules type-check and lint passes. Receipts: nvfp4_native_vmap_audit.txt, nvfp4_native_vmap_mypy.txt, nvfp4_native_vmap_generated_docs.txt. Final benchmark source hashes match current files. Graphify refresh was attempted but the host CLI is missing (nvfp4_native_vmap_graphify.txt). Full-suite green status and generic closure are not claimed.

## Symbolic native vmap and geometry ownership

Sync NVIDIA-NVFP4-VMAP-SYMBOLIC-2026-10-06. Typed Tensor dimension annotations
previously entered packed-byte shape inference without concrete dtypes; that
shape is now deferred until logical/physical storage is known. The reproduced
constraint bypass (nvfp4_vmap_constraint_before.txt) arose because every
independently mapped annotation kept rank two and the constraint binder skipped
rank-three calls. The mapped owner now lifts those dimension names with a fresh
batch symbol, keeping scalar-owner annotations and constraints unchanged.
Unmapped None/all-None axes retain the original no-map API semantics.

RTX 5070 tests prove B=3/M=7 and B=7/M=3 have distinct Schedule/package
identities despite identical flat rows, and returning to the first geometry
reuses the original package. Typed keyword calls match an independent fp64
oracle; violated symbolic M constraints refuse before compilation. Focused
shape/constraint gates: 81 passed; owning geometry/storage tests: 22 passed,
two non-batch skips. Final combined replay: 177 passed, six skips. Final
registry/audit/diagnostic/pass gate: 303 passed. Type and lint checks pass.

Packet nvidia_nvfp4_typed_vmap.json: four oracle-exact typed frontend rows,
seven public-wall and resident native-event samples with source/tool/GPU
identity. Final source hashes match. This remeasures the same named static
profiles after the frontend repair; no kernel strategy or default changes.
Earlier packets remain historical source snapshots. General native batching,
dynamic/composed producers, transpose/AD, full-suite closure and publication
remain open. No sibling physical support or performance transfer is claimed.

Final typed-vmap benchmark public wall medians are 1.359-1.448 ms; resident native-event medians are 8.893-9.365 us. These are separate timing domains with no speedup claim. Generated views completed successfully (nvfp4_vmap_symbolic_generated_docs.txt); Graphify update still fails because its host CLI is absent (nvfp4_vmap_symbolic_graphify.txt). The packet source hashes remain current.

## Native NVFP4 operand orientation

Sync NVIDIA-NVFP4-NATIVE-ORIENTATION-2026-10-06; W1.1 / FRONTEND-IR-MEDIUM-1 / E2E-REAL-6. Semantic A/B orientation now reaches native Graph verification, sealed Schedule decisions, Tile packed-code/K16-scale accessors and PTX. Standalone Schedule and Tile verification rejects non-boolean flags and enabled flags outside the named NVFP4 contract. No production Python transpose or unpacking.

RTX 5070: 23 numerical JIT/vmap cases pass; final focused native/shape/device replay: 67 passed. Broader regression replay: 149 passed, 26 skipped. Registry/audit gates: 333 passed. Receipts accompany this packet. The rebuilt compiler completed with existing unrelated warnings.

Packet nvidia_nvfp4_native_orientation.json has 20 oracle-exact rows over ragged and larger shapes, with identical quantized values across orientation arms. Final source and tool hashes match. Warm public wall medians: 1.251-2.282 ms; resident native-event medians: 8.662-12.154 us. Separate timing domains, no speedup or strategy promotion.

Shared-RHS transposed batched A, general/dynamic batching, composed producers, linear-transpose AD, full integration and publication remain open. No sibling device support is inferred. Graphify update was attempted; the host CLI is absent.

## Full integration after native orientation

The complete non-slow unit lane on Super-Bear finished with 25 failures, three setup errors, 21567 passes, 7506 skips and 874 deselections (integration_after_orientation.txt). Twenty-two failures and three errors came from using the relocated AMD clang as an unqualified host C/C++ compiler; that SDK lacks host compiler-rt builtins. Host LLVM now precedes the SDK on PATH while TESSERA_ROCM_CLANG and HIP_CLANG_PATH keep AMD generation explicitly pinned. The recorder inventory also lacked names for two new NVFP4 recorders; benchmarks/README.md now names them.

Focused replay of all repaired IEEE/raster/numeric-carrier/recorder lanes: 340 passed (integration_environment_recorder_repair.txt). The two general scaled-matmul batching/linear-transpose closure failures remain genuine; their tests and coverage states are unchanged. No full-suite green status is claimed. Graphify 0.8.27 is restored in an isolated WSL scratch tool environment; refresh has been started after the unit process became terminal.

## Shared-RHS transposed-A native integration

Owner W1.1 / FRONTEND-IR-MEDIUM-1 / E2E-REAL-6; sync NVIDIA-NVFP4-SHARED-TRANSPOSE-2026-10-06.

The frontend and native Graph verifier admit static rank-three transposed NVFP4 A with rank-two shared B. Native Schedule seals shared batch intent and dimensions into the digest; Tile carries that policy into the existing ten-argument rows/batches ABI. Target lowering tiles each batch independently, offsets A/scales/output by batch, and keeps B/scales shared. Packed K and K16 scale axes follow the declared orientation. Existing untransposed shared-row and independent-RHS routes are retained.

Matching LLVM/MLIR 23.1.1 core and NVIDIA builds pass. Focused replay passes **133 tests, including 30 RTX 5070 numerical cases** through ordinary JIT and vmap, warm package reuse, ragged rows/columns, odd K and scale orientation. Batch scalar reinterpretation is rejected before launch. Registry/audit gates pass **333 tests**. The first replay exposed a duplicate frontend shape gate; its six failures are preserved separately and resolved in the final replay.

The new packet has **24 matched orientation rows**, five timing windows each, independent decoded fp64 oracle checks before timing, every public sample and final portable launch. Current source and tool hashes were verified against the packet. All rows have zero measured maximum absolute error for these chosen operands.

| Batch,M,N,K | Transpose B | Resident event median, us | Warm public median, ms |
| --- | --- | ---: | ---: |
| [3, 17, 19, 129] | False | 9.078 | 1.372 |
| [3, 17, 19, 129] | True | 8.916 | 1.345 |
| [3, 128, 128, 256] | False | 12.843 | 1.726 |
| [3, 128, 128, 256] | True | 12.406 | 1.853 |

Resident events measure a 100-launch window including dispatch gaps and exclude upload/readback. Public wall includes validation, allocation, transfers and synchronized launch. These measurements characterize the route; they do not establish a speedup or select a new physical strategy. FP8/MXFP8/MXFP4 strategy evaluation is unchanged.

Remaining: general/dynamic/composed batching and linear-transpose AD closure, higher AD/residual integration, asynchronous ownership, broader producer composition, aggregate validation and publication. No general coverage gate has been weakened.

Sibling regression replay: **481 passed, 24 skipped**. The 24 skips were explicit SM120 proof opt-ins in the legacy emitter suite, not device absence. After the live RTX 5070 probe, the owning gate was enabled and that suite passed **47 tests with no skips**; receipt nvfp4_legacy_emitter_device_replay.txt. This supplements the compiler-owned package proof and does not replace it.
