---
audit_role: plan
plan_state: landing
owner: Compiler frontend and AD
last_updated: 2026-10-14
---

# Public native JVP integration

Discovery: [compiler map](README.md). Global sequencing defers to
[INTEGRATED_COMPILER_PLAN.md](INTEGRATED_COMPILER_PLAN.md).

Synchronization key: PUBLIC-NATIVE-JVP-20261009.
Owning items: FRONTEND-IR-MEDIUM-1 / AD-RESIDUAL-EVAL-1 / E2E-REAL-6.
This is active work toward the five compiler slices; it does not close them.

## Current implementation and proof

The public `tessera.autodiff.jvp` entry projects explicit native JIT requests
into an independent native Graph AD owner before Python tangent tracing.
Caller Graph IR, differentiation intent and shape constraints are retained.
Inactive operands are explicit `None`; the child owner retains the active
input mapping. Warm calls reuse the child; source or constraint changes
invalidate its witness. Native execution receipts are required.

The already traced `flash_attn(..., bias=...)` spelling now reaches the
reference primal and derivative certificates through the interception wrapper.
`bias` and `attn_bias` cannot both select non-null score operands.
This fixes reference certificate intent, not production GPU arithmetic.

Host WSL: 26 focused tests pass, one Apple hardware test skips, and Ruff passes.
The new independent FP64 oracle covers grouped causal attention with dense
and broadcast bias, output and saved LSE, JVP direction products and projected
VJP products. These are reference certificates, not Apple execution proof.

Exact RTX5070 / SM120 / CUDA13.3: 26 public native JVP cases pass,
covering host/resident roots, plain/biased/saved-LSE forward, K=5/129,
value-only and bias-only activity, retained output storage, original graph
invariants, and warm compiler/frontend refusal. Checks reject output-span
overlap, input/output overlap, incorrect extents and package output tampering;
eight independently pending producer streams are ordered before native reads.
The combined public and adjacent resident attention lane passes 84 cases;
All 24 separately opted-in prepared-owner cases also pass using freshly
generated native compiler fixtures.

Six production-compiler contract cases and 106 adjacent JVP unit cases pass.
Another 312 diagnostic/pass metadata and public-transform gates pass.
The immutable parent full unit suite completed: 25,728 passed, 7,565 skipped,
874 deselected and two failures in the existing scaled_matmul batching and
transpose closure tests. No full CI pass is claimed.

Tests: tests/unit/test_attention_bias_alias_certificate.py,
tests/unit/test_public_native_jvp_transform.py,
tests/unit/test_attention_saved_lse_jvp_ir.py,
tests/device/nvidia/test_public_native_jvp_transform.py.

## Native saved-LSE product implementation

The typed Graph tangent interface admits paired O/LSE products. The registered
checkpoint JVP op carries an optional row-LSE tangent of the forward LSE shape.
The hashed native Schedule contract verifies the primal/return mapping and
carries both derivative outputs through Tile and LLVM/PTX lowering.
dLSE is the native probability-weighted score direction; value-only activity
writes exact zero. Production derivative arithmetic remains in native IR.

Portable schema v3 seals saved-LSE output selection. The retained C++ owner
owns distinct O, LSE, dO and dLSE spans, checks extents and overlap, orders
producer streams and retains private generations. Legacy one-result products
retain their previous portable schemas and 9/11-pointer ABIs; paired products
use separate 10/12-pointer contracts.

Two fresh six-profile packets validate all four outputs before alternating
host/resident completed-call timing and separate forward/JVP CUDA events.
Maximum absolute error is 6.167596078299198e-7 against the independent FP64
direction oracle. Resident/host completed-call ratios span 1.187–1.300.
These small profiles establish correctness and overhead characterization;
they do not establish a performance improvement or default-route promotion.

Evidence: benchmarks/baselines/public_saved_lse_jvp_20261009/README.md,
run1.json and run2.json in that directory.
Matching LLVM/MLIR 23.1.1 compiler tools and the fresh C++ provider were built.

## Remaining integration

Public integration on GFX1201 is now proved for 36 named continuous FP32
scaled-product cases, with two fresh 36-profile event/public-call packets.
Evidence: benchmarks/baselines/public_native_scaled_jvp_20261009/README.md.
This is a different program family from the SM120 attention proof; neither
establishes Apple or x86 parity. GFX1201 public preparation/binding overhead,
mixed-axis and broader composition/quantized AD remain open. Dynamic/composed
products, half-storage AD, nested/higher-order transforms and resident-call
overhead remain open. Generic scaled_matmul batching/transpose closure also
remains open; the existing unit gates have not been weakened.

## Wider scope still open

Nested/higher-order native transforms, bounded tensor products and general
frontend compositions remain active. GFX1201 NVFP4 ingest, W1.1 producer
retirement, general attention forward/backward consumers, and ROCm
route/performance closure retain their existing plans and evidence obligations.

Recorder: `benchmarks/nvidia/record_public_saved_lse_jvp.py`; correctness-gated paired output and tangent checks precede alternating host/resident completed-call and native event timing.

GFX1201 recorder: benchmarks/rocm/record_public_native_scaled_jvp.py.

## Retained GFX1201 execution plan

The named public scaled-product route now retains one checked C++ owner per
compiled specialization, with bounded cache, transactional input/update/read
and native guarded cleanup. Cold admission seals parent/child identity;
warm calls preserve the same native IR/images and avoid repeated serialized
validation and preparation. 70 gfx1201 ownership/numerical/FP8 regression cases
and 50 SM120 shared lifecycle cases pass. Four matched 36-profile packets
show 31.9–63.0% lower completed public time, with unchanged kernel images.
Evidence: benchmarks/baselines/retained_public_scaled_jvp_20261009/README.md.
Recorder: benchmarks/rocm/record_retained_public_scaled_jvp.py.
Remaining public overhead and general contracts above remain active.

## Attention JVP schema reconciliation (2026-10-11)

The numeric Graph AD contract is independent of portable program and native
tensor-manifest version numbers:

| Graph AD schema | Meaning |
| --- | --- |
| 1 | Static output-only JVP, no bias |
| 2 | Static output-only JVP, with bias |
| 3 | Bounded output-only JVP, no bias |
| 4 | Bounded output-only JVP, with bias |
| 5 | Bounded saved-LSE JVP; bias is explicit in the contract |
| 6 | Static saved-LSE JVP; bias is explicit in the contract |

Schema 5 requires saved-LSE selection, shape bounds and the
bounded_sequences_v1 policy. Schema 6 requires saved-LSE selection without
bounded shape fields. Output-only schemas 1–4 retain their previous meaning.
The former branch-local saved-LSE schema 3 is rejected rather than interpreted
as a bounded output-only product. No automatic conversion of opaque artifacts
is performed. Native tensor manifests retain schema 1 for static and schema 2
for bounded scalar/grid expressions. Portable paired-LSE programs retain
tessera.native_attention_jvp_program.v3.

The combined ABI orders tensor outputs tangent, dlse before scratch and
query_size/key_size scalars. All four backend queues are assessed under
ATTENTION-JVP-SCHEMA-RECONCILE-20261011. The named combined
bounded saved-LSE SM120 gate is now proved under SM120-BOUNDED-SAVED-LSE-20261011;
broader storage, masks and compositions remain separate gates.

Validation: 624 focused native compiler, portable artifact/runtime, public
JVP, bounded/bias and registry/audit tests pass with a freshly built
LLVM/MLIR 23.1.1 compiler on Tajasaurus. The 24-case schema matrix includes
inactive score/value-only paths and rejection of colliding schema numbers.
Ruff passes. These are compiler/ABI checks, not SM120 numerical evidence
for the newly combined route.


## Public bounded saved-LSE execution (2026-10-11)

The explicit JIT JVP program now accepts sequence_bounds. Python supplies
positive Sq/Sk capacities while retaining its original static traced Graph;
the shared native MLIR projection checks and constructs symbolic sequence
types before either forward or reverse AD. Schema 5 means bounded saved-LSE
specifically. Schema 6 and output-only schemas 1–4 are unchanged.

Fresh matching native compiler checks cover the public request, capacity
rejection, schema selection, dynamic LSE, ordering, artifact replay and prior
reverse projection. RTX5070 numerical execution covers (9,11) and (17,129)
capacities, inactive/value-only/bias-only activity, repeated shapes and old
generations. Generic host/resident forward and backward launches retain all
11 logical/physical scalars for dynamic broadcast bias; seven-scalar full
shape kernels remain valid.

Two fresh 30-profile packets have identical sealed package/image/binding
identities and all 22 source/four binary pins match. Maximum absolute error is
7.188786748102416e-7. Final post-review forward CUDA-event medians span 8.303–56.553 us; JVP
preloaded dispatch medians span 10.967–26.872 us. Completed resident capture+JVP
medians span 5.093–7.285 ms, leaving host preparation/ownership overhead open.
No kernel speedup, promotion, generic AD or sibling device closure is claimed.
All four architecture queues are assessed under the shared synchronization key.

Evidence: benchmarks/baselines/bounded_saved_lse_jvp_20261011/README.md.


## Scratch reconciliation — 2026-10-14

Merged PRs 933–941 are the integration baseline. Historical scratch mirrors
were reconciled and pruned; unique scaled-multiply/prototype overlays remain
in recovery commits rather than active worktrees. The scaled-multiply carrier
remains follow-up required: port it onto the current main use-after-free repair,
then run native compilation, numerical, registry and benchmark gates before
landing. It is not included in bounded saved-LSE closure evidence.

## Named FP32 tensor gate — 2026-10-14

Owner FRONTEND-IR-MEDIUM-1 / AD-RESIDUAL-EVAL-1 / E2E-REAL-6.
Shared contracts: a same-shaped static unencoded FP32 multiplication member
in the native scaled-product SSA program; cotangent linearity, captured
dependencies, checked member bindings and last-read lifetime validation.
Native MLIR and the ROCm backend own Schedule/Tile/LLVM image generation.
The current main use-after-free and bounded attention fixes are retained.
Existing generic capability and manifest execution claims are unchanged.
Evidence: benchmarks/baselines/native_scaled_tensor_gate_20261014/README.md.

Parity validated for eight named RX 9070 XT/gfx1201 primal/JVP/VJP device cases and two fresh correctness-gated timing packets. Follow-up required for gfx1151 execution, encoded FP8/MXFP8/MXFP4, batching, dynamic/general composition and public overhead. No global AD or standalone multiplication closure is claimed.

## Encoded residual coefficients — 2026-10-14

Owner FRONTEND-IR-MEDIUM-1 / AD-RESIDUAL-EVAL-1 / E2E-REAL-6.
Native MLIR retains encoded primal products as residual coefficients when
they do not depend on requested continuous inputs. Requested dependency
analysis is memoized; inactive unsigned slots use existing zeros-like Graph
calls and are removed by requested-gradient program projection. Encoded
members keep their original numeric/storage policy through Schedule/Tile.
Differentiated typed JIT owners now execute the compiled primal too.
Standalone multiplication retains its existing public route; new native
multiplication packaging stays inside the verified scaled program.
Capability/manifest entries link the exact named conformance envelope.
Evidence: benchmarks/baselines/native_encoded_scaled_gate_20261014/README.md.
Recorded by benchmarks/rocm/record_native_encoded_scaled_gate.py.

Parity validated for four named RX 9070 XT/gfx1201 encoded cases:
E4M3 with FP32 scale/gate derivatives, reordered/gate-only roles, and E8M0
gate-only derivatives. Twelve compiler/contract cases include native coded matrix/
exponent derivative refusal. Changed coded values, warm compiler refusal, source
identity and retained outputs are proved. The eight continuous FP32 cases
also pass with the unchanged serial default. Explicit wave scheduling passes
the four encoded cases; broader FP32 residuals retain their separate-admission
guard. No default promotion is made. FP4, transpose/batching/dynamic/general
AD and gfx1151 execution remain follow-up required; gfx1151 has no FP8 WMMA.

## CONTINUOUS-SCALE-WAVE-20261014

Owner E2E-REAL-6 / AD-RESIDUAL-EVAL-1 / FRONTEND-IR-MEDIUM-1.
Native Schedule classifies each program member: explicit wave requests apply
to lhs/rhs scale adjoints; continuous products, matrix gradients and movement
carriers retain serial geometry. Four-FP32 wave members carry matching scale
roles on the Tile wrapper and its sealed GPU body. Shared Tile/Target
verification and package validation preserve those roles and checked geometry.
Automatic selection and the serial default are unchanged.
Evidence: benchmarks/baselines/continuous_scale_wave_20261014/README.md.

Parity validated on RX9070XT/gfx1201: 30 compiler/selector cases, 12 public
FP32/FP8/MXFP8 gate cases and 40 continuous mapped/seed/lifetime cases pass.
Four isolated serial/wave/wave/serial packets retain matched source/binary
pins and identical primal/JVP images. The named continuous reverse program
event window changes from 36.564–36.685 us to 25.009–25.152 us; public latency
is separately 1.812–1.852 ms versus 1.767–1.802 ms. No general public speedup
or default promotion. gfx1151 wave execution is not applicable: the native
structured consumer still requires gfx1201. Wider dynamic/higher AD, MXFP4
residual integration and M256 attribution remain follow-up required.

Focused host/drift delivery gates pass; full CI and review remain pending. This named scheduling
step does not close the five-slice compiler goal.
