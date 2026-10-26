---
last_updated: 2026-10-25
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

Q/K/V share compact row-major FP16, BF16 or FP32 storage, except that the
forward admits a distinct K/V storage (see below). O either matches Q's
storage or is FP32; row-LSE is always FP32. Optional dense or broadcast
bias is FP32. GQA head divisibility and exact tensor shapes are checked.
The native recurrence accumulates in FP32; only the final O store may narrow.
Causal masking retains the existing end-aligned policy.

Production tracing must preserve each result's dtype independently. A
same-storage primary result must not overwrite the independently inferred
FP32 auxiliary. The public half-storage trace previously violated this rule
and silently missed native package selection; tests now require a native owner.

## Mixed Q vs K/V storage (forward)

Synchronization key: SM120-SAVED-LSE-MIXED-STORAGE-20261025.

K and V always share one storage. On the saved-LSE forward that storage may
differ from Q: any pair of FP16, BF16 and FP32 is admitted. O is FP32 or Q's
storage, which is the result type the frontend gives `flash_attn`. Each operand
widens exactly to the declared FP32 accumulation, so no conversion policy is
chosen. Row-LSE remains FP32.

- **Contract:** the native checkpoint contract carries `kv_storage` only when it
  differs from Q. Every homogeneous contract, checkpoint identity, entry name
  and Tile kernel is therefore byte-unchanged.
- **Mixed forward:** the contract and the checkpoint identity both include
  `kv_storage`. The entry is named
  `tessera_tile_attention_lse_<q>_kv_<kv>_[out_<o>_]<digest>`. The Tile
  `attention_kernel` carries `kv_storage`, and its verifier rejects that
  attribute outside the SM120 saved-LSE forward or when it repeats Q's storage.
- **Typed ABI:** a mixed forward uses the same typed ABI identifiers, because
  their buffers already declare per-role storage. The descriptor's K/V
  bindings carry the K/V dtype.
- **Runtime sizing:** both the checked launch and the retained owner size each
  buffer from its own storage. The owner is reached through
  `tessera_nvidia_attention_forward_prepare_kv`, and the original
  `tessera_nvidia_attention_forward_prepare` remains its homogeneous wrapper.

The backward checkpoint, paired AD, original-source reverse and the JVP owners
still require homogeneous Q/K/V. They refuse a mixed forward: the backward
resolver refuses it, the pair checks refuse it, and the sealed entry-name
checks refuse it.

## Unscaled FP8 K/V storage (forward)

Synchronization key: SM120-SAVED-LSE-FP8-KV-20261025.

The shared K/V storage of the saved-LSE forward may also be `fp8_e4m3`
(E4M3FN) or `fp8_e5m2`.

**Unscaled means the stored FP8 values are the operands.** The kernel loads one
byte and decodes it exactly to the declared f32 accumulation:
- **E5M2** is the high byte of an IEEE f16.
- **E4M3FN** is decoded bit for bit: subnormals are m·2⁻⁹, S.1111.111 is NaN,
  and there are no infinities.

Every finite code point of both formats is checked bit-exactly on the device.

| Role | What this slice allows |
|---|---|
| Q | f16, bf16 or f32 |
| O | Q's storage or f32 |
| Backward roles | homogeneous storage only (FP8 is refused) |
| Resident roots | f16, bf16 or f32 only (FP8 roots are refused); FP8 K/V are host roots |
| K/V buffers | one-byte elements, alignment 1 |
| Retained owner | `kv_storage` codes 4 (`fp8_e4m3`) and 5 (`fp8_e5m2`) |

The resident restriction comes from the shared resident-tensor contract.

**A scaled KV cache is a different contract.** K = k_fp8 · k_scale (and the
same for V) needs scale operands and a declared scale layout, which are
semantic keys. It is not admitted by spelling FP8 storage.

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

Existing FP32 ABI identifiers and raw Tile/Target/PTX bytes remain unchanged.
Wrapper image/artifact digests include compiler fingerprints and may change.
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
