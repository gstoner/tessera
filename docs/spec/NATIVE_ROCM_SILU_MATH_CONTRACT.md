---
last_updated: 2026-10-16
audit_role: reference
---

# Native ROCm SiLU gate-multiply

Owner: ROCM-NVFP4-INGEST-1 / E2E-REAL-6 / FRONTEND-IR-MEDIUM-1.
Synchronization: ROCM-NATIVE-SWIGLU-ACTIVATION-20261016.

Status: implementation under validation; owning-device and timing gates remain.
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

The focused compiler/registry lane passes671 cases after the matching-half
extension, including native GPU expression order, unchanged old ABI meanings,
source immutability, argument reversal, mutation refusal and sibling guards.
These are host/compiler tests, not GPU numerical evidence.

Rank-one half widening admission is new only on gfx1201. Its exact-device
fixture must pass before publication; gfx1151 retains its previous rank-two/
rank-three widening envelope. Core compiler tests on a named target do not
substitute for numerical execution on that target's owning hardware.

Remaining gates include owning gfx1201 numerics, IEEE/signed-zero checks,
ordinary JIT/changed-input/compiler-free replay, shape-independent images,
separate kernel and completed-host benchmarks, generated registries and exact
PR checks. Full resident gate/up views/lifetimes, activation quantization and
down projection remain part of the full five-slice objective.
