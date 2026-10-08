# Prepared native ROCm movement call binding

Owner E2E-REAL-6; sibling FRONTEND-IR-MEDIUM-1 and AD-RESIDUAL-EVAL-1.
Synchronization key ROCM-PREPARED-MOVEMENT-2026-10-05.

## Native ownership and verified compiler route

Each prepared call starts from the canonical compiler result, with adjacent
Graph/Schedule/Tile/Target/backend hashes, a native Schedule producer and an
image-bound checked descriptor. Preparation validates the exact f32/i32 buffer
roles, shapes, scalar order, geometry, workspace and completion policy.
The C++ service copies the compiler image, entry, static dimensions and ABI;
it contains no GPU kernel source or mathematical lowering.

Warm public JIT calls compare the typed Graph against a sealed snapshot and
invoke this prepared native call without textual Graph serialization.
A changed Graph revokes the prior prepared call and re-enters compilation.
Each launch checks native f32/i32 storage, rank, extents, byte strides, alignment,
byte size, output overlap and index bounds. Read-only input aliases are allowed.
Default-stream synchronous completion, existing context-owned staging,
image leases, quarantine and explicit context teardown remain in force.

Prepared handles are monotonically assigned IDs, not dereferenced caller
pointers. Context/device/fork mismatch and closed/unknown handles are refused.
Shared ownership retains the image/ABI through a close during invocation.
Preparation failures do not publish a handle. Explicit native-storage close
releases prepared calls; subsequent JIT execution rebinds the same compiler
package. Matching optional runtime exports select preparation; older/missing
optional libraries retain the existing checked compiled descriptor route.
The independent TESSERA_ROCM_PREPARED_MOVEMENT=0 control is the A/B baseline.
No new semantic operation, dtype, target, pass or stable diagnostic is added.
The three new C ABI exports are recorded by the owning runtime ABI generator.

The JIT clears its last native execution receipt at call entry, so rejected
views or policies cannot expose a prior success. Successful prepared receipts
are emitted only after actual native completion. Runtime profiling reports
host binding wall time, with kernel time left unset.

## Cross-registry reference AD correction

The prior MoE eager shape/gather change exposed stale identity JVP/VJP rules.
Reference tangents now gather token rows; cotangents scatter-add every repeated
slot into its source token. DispatchPlan uses the actual sorted token mapping.
Tape replay retains opaque operands claimed by a rule in nondifferentiable
literal slots, fixing positional DispatchPlan reverse mode. The old swallowed-
transport whitelist entry is removed because transport is now explicitly
handled; non-null transport remains outside the local tensor form.

Independent finite differences, adjoint identities and ordinary eager
forward/reverse APIs cover unequal slot/token counts and DispatchPlan.
The MoE adjoint and canonical-primal chain law pass at their original tolerances.
This is reference/frontend parity, not a native compiled movement AD route:
native differentiation for these movement packages remains open.

## Exact-device tests and performance

AMD Radeon 8060S/gfx1151 runs paged KV and local token-gather MoE.
RX 9070 XT/gfx1201 runs paged KV. Each family has two shapes, reordered
arguments/indices, repeated slots/pages, shape specialization and nonfinite
bit-pattern proof. Tests prohibit warm Graph serialization, reject strided
views and changed policies before launch, and prove explicit close/rebind.

Owning public/prepared tests: gfx1151 20 pass, 4 other-hardware skips;
gfx1201 18 pass, 6 other-hardware skips. Dedicated gfx1151 source has two
pytest mark-registry warnings. Six owning gfx1201 checkpoint JIT regressions
pass. No gfx1151 NVFP4 ingest execution is claimed; its scratch tree lacks that
test module, recorded as unavailable.

Controlled native C++ and shared image/storage/JIT contracts: 119 pass,
23 skips. Reference AD, nested-tape and complete law tests: 209 pass, 10 skips.
Registry, runtime ABI, backend manifest/conformance, diagnostics/pass metadata,
and audit documentation gates: 612 pass. Compiler-plan ownership/link validation,
changed-file Ruff and Git whitespace checks pass. All 32 generated documentation
surfaces are in sync after the owning generators ran.
Initial test-selection and stale-whitelist failures are retained separately.

Ten balanced alternating warm trials compare prepared public JIT, the same
function with preparation disabled, prebound checked descriptor launch, and
retained production helper. Compilation is excluded; first-call wall cost is
recorded separately. All timings are whole host calls including transfers,
dispatch, completion and download. Native counters prove one launch, three
buffer reuses and zero warm staging allocations/frees per compiler call.

| Architecture | Case | Prepared JIT ms | Ordinary binding ms | Descriptor ms | Retained ms |
| --- | --- | ---: | ---: | ---: | ---: |
| gfx1151 | paged small | 0.5143 | 0.7120 | 0.5342 | 0.6332 |
| gfx1151 | paged large | 0.5027 | 0.7035 | 0.5349 | 0.6222 |
| gfx1151 | paged_default small | 0.5071 | 0.7076 | 0.5254 | 0.6227 |
| gfx1151 | paged_default large | 0.5114 | 0.7134 | 0.5403 | 0.6358 |
| gfx1151 | dispatched small | 0.4984 | 0.6931 | 0.5192 | 1.7785 |
| gfx1151 | dispatched large | 1.1558 | 1.3375 | 1.1594 | 3.8554 |
| gfx1201 | paged small | 0.6391 | 0.9003 | 0.6759 | 0.7762 |
| gfx1201 | paged large | 0.5297 | 0.7463 | 0.5775 | 0.6806 |
| gfx1201 | paged_default small | 0.5399 | 0.7401 | 0.5684 | 0.6638 |
| gfx1201 | paged_default large | 0.5821 | 0.7958 | 0.6221 | 0.7408 |

Prepared calls reduce the ordinary JIT binding wall cost by
13.6–29.0% in these envelopes.
All ten rows satisfy the existing retained-route 10% non-regression gate.
All native image digests are unchanged from the preceding public movement
packet: these gains come from host binding/orchestration.
Samples, profiles, device/PCI identity, source/compiler/library fingerprints,
native HSACOs and adjacent IR snapshots are retained. Earlier iterations remain
separate historical packets. No kernel-only or application speedup is claimed.

## Remaining work and sibling assessment

General paged layouts, asynchronous/resident movement, native movement AD,
controlled kernel-only measurements and distributed DispatchPlan transport
retirement remain open. This closes the named synchronous public-call
performance gap, not the full ROCm backend or the five-slice objective.
FP8/MXFP8/MXFP4 quality and performance requirements remain independent gates.

Apple/x86 native physical preparation is not applicable to this HIP service;
their reference AD/Tape changes have shared host tests, with no device claim.
NVIDIA uses the shared Tape/reference AD correction; CUDA call binding and
SM120 physical performance are unchanged. No ROCm timing transfers.
A native persistent compiler session remains an architectural follow-up.
Graphify is unavailable in the authoritative WSL scratch checkout.
