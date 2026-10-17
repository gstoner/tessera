---
last_updated: 2026-10-16
audit_role: reference
---

# Native ROCm SiLU gate-multiply

Owner: ROCM-NVFP4-INGEST-1 / E2E-REAL-6 / FRONTEND-IR-MEDIUM-1.
Synchronization: ROCM-NATIVE-SWIGLU-ACTIVATION-20261016.

Status: owning gfx1201 numerical/replay, benchmark and local validation gates passed; exact-head PR checks remain.
This is an activation dependency for full source-preserving NVFP4 MLP
composition, not full MLP or model-quality acceptance.

## Semantic and storage contract

The original isolated `tessera.silu_mul(a, b)` Graph computes
`(a / (1 + exp(-a))) * b` with FP32 arithmetic. FP16/BF16 inputs widen exactly.
The only narrowing is the final matching-half output store. An explicit Graph
widening cast instead retains FP32 output. Operand order is significant and
must survive original argument permutation; this is not commutative multiply.

| Inputs | Output | Checked ABI identity |
|---|---|---|
| FP32 | FP32 | `tessera.rocm.math.a_b_o_n.f32.v1` |
| FP16 | FP32 | `tessera.rocm.math.a_b_o_n.f16_f32.v1` |
| BF16 | FP32 | `tessera.rocm.math.a_b_o_n.bf16_f32.v1` |
| FP16 | FP16 | `tessera.rocm.math.a_b_o_n.f16_f16.v1` |
| BF16 | BF16 | `tessera.rocm.math.a_b_o_n.bf16_bf16.v1` |

The last two identities are additive. Existing identities do not change their
input/output byte widths. All use the binary pointer/pointer/output/count frame.
The dtype is preserved through Graph, Schedule, typed Tile, Target, native image,
and descriptor output allocation. Portable validation checks exact Target store
storage and descriptor identity against the retained native Tile contract.

Admission requires two distinct frontend argument roles, same-shaped static
compact storage, one returned result, supported exact casts and the default
FP32-compute policy. Invalid numerical attributes or modified Schedule/Tile
roles, source, geometry or storage reject rather than change interpretation.
General producer regions, arbitrary views, broadcasting and dynamic shapes
need their own contracts. Public calls with aliased readonly input arrays do
not authorize aliasing a returned output with either input.

## Compiler ownership

Native Graph-to-Schedule retains the original operation and seals full storage,
shape, argument roles, bindings and numeric policy. Schedule-to-Tile replays the
source before emitting `tile.elementwise_kernel` with family `binary_math`.
The ROCm Target adapter validates that same contract and selects the existing
binary GPU generator with kind `silu_mul`. The body lowers MLIR arithmetic/math
through GPU/ROCDL/LLVM to HSACO. Python decodes native metadata and marshals the
checked frame; it does not construct Tile arithmetic or execute the gate.

The shared Tile narrow-storage exception requires the matching ROCm architecture
and a sealed native SiLU contract. It does not enable narrow `pow` or sibling
Apple, NVIDIA or x86 binary-math storage. Existing sibling FP32 rules remain.

## Validation boundaries

The final focused compiler/registry lane passes 556 cases after the final selector/recorder follow-through. The owning gfx1201 lane passes 45 device cases
plus the separately selected 47 compiler-dependent and 34 pure frontend/capability cases, including matching-half fresh-process replay
with compiler access and eager/legacy arithmetic forbidden. Adjacent native
math/widening and native owner lifetime validation passes 91 cases. These counts
have different scopes and are not independent cumulative coverage totals.

The additive native staging export `tessera_rocm_math_launch_v2` accepts an
explicit output element width. Matching two-byte output requires binary math
with two-byte inputs. Dimensions, exact byte counts, alignment, overlap and
owner identity are checked before resources are acquired. The original
`tessera_rocm_math_launch` remains an FP32-output wrapper; older providers are
still usable for FP32 output. A half-store descriptor requires the v2 export.
The runtime owns storage/launch only; arithmetic remains in the compiled image.
Malformed width/span/alias tests check rejection before GPU synchronization,
and existing allocation/copy/completion/lease fault injection remains green.

Rank-one half widening admission is new only on gfx1201. Its exact-device
fixture passes rank-one widening; gfx1151 retains its previous rank-two/
rank-three widening envelope. Core compiler tests on a named target do not
substitute for numerical execution on that target's owning hardware.

Remaining gates include benchmark interpretation, generated registries and exact
PR checks. Owning tests cover IEEE/signed-zero, ordinary JIT/changed-input/
compiler-free replay and shape-independent images. Full resident gate/up views/lifetimes, activation quantization and
down projection remain part of the full five-slice objective.

Public SiLU selection is gfx1201-only until independent gfx1151 execution proof.
The shared low-level compiler admission remains available for artifact testing;
it does not preempt the sibling public route. Broad host validation found one
missing benchmark citation; that citation and its unchanged gate now pass.

The final broad CPU lane passes 26,067 cases with 5,902 skips; the production
compiler lane passes 1,312 with zero skips. The full shared provider preserves
47 existing C exports and adds only the v2 entry. Its owning SiLU plus
bounded/public/prepared NVFP4 regression lane passes 87 cases.

## Review boundary checks

Shape-only SiLU capture requires matching canonical FP16/BF16/FP32 inputs and
checks actual concrete dtype metadata as well as tracer spelling. Other concrete
storages retain the registered public primitive's eager output dtype. That
wrapper uses FP32 results for integer inputs; raw NumPy expressions can choose
a different promotion and are not the registered wrapper's policy. No new
integer activation execution support is added. Abstract integer or mixed
promotion requires values; supported floating aliases retain shape-only capture.

The gfx1201 public selector, support predicate and portable package validator
all enforce ranks 1–3. Broader low-level C++ artifact recognition does not widen
the public execution certificate.
