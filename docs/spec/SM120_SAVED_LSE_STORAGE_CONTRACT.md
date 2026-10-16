---
last_updated: 2026-10-16
audit_role: reference
---

# SM120 saved-LSE storage contract

Synchronization key: SM120-SAVED-LSE-STORAGE-20261016.
Owner: E2E-REAL-6 / FRONTEND-IR-MEDIUM-1 / AD-RESIDUAL-EVAL-1.

## Native route

Python/textual frontend -> typed Tessera Graph MLIR -> native verified
checkpoint projection -> Schedule MLIR -> Tile MLIR -> NVIDIA Target IR ->
LLVM/NVVM/PTX -> native image and checked runtime ABI -> RTX5070 execution.

The compiler owns arithmetic and result storage. Python packages native
artifacts and validates metadata; it does not reconstruct an attention kernel
or convert half roots to FP32 tensors before launching.

## Storage and numerical roles

Q/K/V share compact row-major FP16, BF16 or FP32 storage. O either matches
that storage or is FP32; row-LSE is always FP32. Optional dense or broadcast
bias is FP32. GQA head divisibility and exact tensor shapes are checked.
The native recurrence accumulates in FP32; only the final O store may narrow.
Causal masking retains the existing end-aligned policy.

Production tracing must preserve each result's dtype independently. A
same-storage primary result must not overwrite the independently inferred
FP32 auxiliary. The public half-storage trace previously violated this rule
and silently missed native package selection; tests now require a native owner.

## Additive forward ABI domain

The new identifiers are:

- tessera.nvidia.attention.q_k_v_o_row_lse_dims.typed_f32acc.v1
- tessera.nvidia.attention.q_k_v_bias_o_row_lse_dims.typed_f32acc.v1
- tessera.nvidia.attention.q_k_v_bias_o_row_lse_dims_bias_shape.typed_f32acc.v1

Descriptor buffers carry canonical storage dtypes, ranks, directions and
alignment; provenance and the checkpoint identity include source/output
storage and FP32 accumulation/LSE. Typed entry names seal those roles together
with the native Schedule digest. The checked launch rejects disagreement
before loading the provider. Each buffer's actual byte width controls native
allocation and copies; bias and LSE retain four-byte elements.

The seven logical dimension scalars are B, Hq, Hkv, Sq, Sk, D and Dv.
Broadcast descriptors additionally carry BiasB, BiasH, BiasQ and BiasK.
Static kernels bake physical bias extents into their IR; the extra descriptor
scalars validate host allocation/copy sizes. Bounded kernels consume the
corresponding physical extents in their existing native argument convention.

The retained C++ owner validates resident pointer capacity and CUDA context,
orders producer streams before private snapshots, retains allocation lifetime,
and returns independent completed host O/LSE arrays. Unowned raw resident
half replay is not part of this delivered public owner envelope.

## Compatibility and proof boundaries

Existing FP32 ABI identifiers and raw native images retain their identities.
Saved-LSE paired AD schema 5 still means bounded sequences specifically;
schema 6 remains static. Paired AD remains its existing FP32 storage contract.
Adding half forward storage does not admit half roots into that derivative ABI.

The shared trace correction applies to all frontends. The new physical
checkpoint contract is SM120-specific; Apple, ROCm and x86 need independent
native consumers and owning-device validation. Existing canonical dtype,
operation, target, pass and diagnostic registrations are unchanged.

The exact-device packet covers static compact Q/K/V, same-storage public O,
explicitly authored FP32 O, independent FP32 LSE, dense/broadcast bias and
causal/noncausal checks. Kernel CUDA events and completed public call wall time
are separate quantities. No attention default promotion, half-storage AD,
dynamic resident closure, sibling-device proof or quantized performance claim
follows from this slice.
