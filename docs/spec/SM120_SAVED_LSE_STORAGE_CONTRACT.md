---
last_updated: 2026-10-26
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

The reverse admits the same mixed pairs (next section). Original-source
reverse and the JVP owners still require homogeneous Q/K/V and refuse a mixed
forward.

## Mixed Q vs K/V storage (backward)

Synchronization key: SM120-SAVED-LSE-MIXED-BACKWARD-20261026.

The saved-LSE backward checkpoint and the paired native reverse admit the same
pairs as the forward: K/V share one of FP16, BF16 or FP32, and Q may hold any
other of the three. Each gradient takes its primal's storage — dQ is Q's, dK
and dV are the K/V storage. dO and the saved O follow Q's storage or are FP32;
LSE, the LSE cotangent and bias stay FP32. The recurrence and the
`probability_dot_v_f32_v1` row correction accumulate in FP32; only the final
gradient stores narrow.

- **AD opt-in:** `denseAttentionAD` admits a distinct K/V storage only when its
  caller opts in. `FlashAttnOp::buildAdjoint` and both `AutodiffPairedPass`
  sites do. The JVP, projection and original-source reverse callers do not.
- **Contract and entry:** as on the forward, `kv_storage` appears in the native
  contract and the checkpoint identity only when it differs from Q, so every
  homogeneous backward is byte-unchanged. A mixed backward entry is
  `..._<q>_kv_<kv>_out_<o>_<digest>`. The Tile `attention_backward_kernel`
  carries `kv_storage`, and its verifier requires a distinct FP16/BF16/FP32
  value on an SM120 saved-output-plus-LSE backward.
- **Pair checks:** the forward and backward of a checkpoint pair must agree on
  `kv_storage`. The scheduled and generated pair packagers refuse a mismatch,
  and refuse a scaled K/V forward outright.
- **Typed ABI:** a mixed backward uses the typed backward ABI and the VJP
  program schema v2, even when Q is FP32. Descriptor K/V/dK/dV bindings carry
  the K/V dtype.
- **Retained owner:** `prepareAttentionVjp` sizes each role from its own
  storage and keeps `kvStorage` in its state. The new
  `tessera_nvidia_attention_vjp_prepare_typed_kv` entry reaches it, and
  `tessera_nvidia_attention_vjp_prepare_typed` remains its homogeneous wrapper.
  External-row and source-program owners refuse a mixed K/V.

**Not admitted:**
- FP8 or scaled K/V in any reverse (FP8 is forward-only);
- K and V in different storages, or a dK/dV whose storage differs from K/V's;
- original-source reverse, JVP or projection over a mixed forward;
- any non-SM120 target.

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
| Backward roles | f16, bf16 or f32 K/V, mixed or homogeneous (FP8 is refused) |
| Resident roots | f16, bf16 or f32 only (FP8 roots are refused); FP8 K/V are host roots |
| K/V buffers | one-byte elements, alignment 1 |
| Retained owner | `kv_storage` codes 4 (`fp8_e4m3`) and 5 (`fp8_e5m2`) |

The resident restriction comes from the shared resident-tensor contract.

**A scaled KV cache is a different contract.** K = k_fp8 · k_scale (and the
same for V) needs scale operands and a declared scale layout, which are
semantic keys. It is not admitted by spelling FP8 storage.

## Scaled FP8 K/V (forward)

Synchronization key: SM120-SAVED-LSE-SCALED-FP8-KV-20261025.

**The public form is a composition, not a new attention signature:**

```python
k = ts.ops.dequantize_scaled(k_codes, k_scale)
v = ts.ops.dequantize_scaled(v_codes, v_scale)
o, lse = ts.ops.flash_attn(q, k, v, lse_checkpoint="saved")
```

**`tessera.dequantize_scaled(codes, scale)`** is a Graph op with an ODS
verifier and an eager reference. It returns f32(code) · scale, rounded once to
f32, under these constraints:

| Role | Rule |
|---|---|
| codes | `fp8_e4m3` or `fp8_e5m2` |
| scale | f32, same rank as codes; each axis 1 or the codes' extent |

The broadcast rule covers per-tensor, per-head, per-token, per-channel and
per-element scales.

`tessera.dequantize_fp8` is different: its input is an already-rescaled f32
payload paired with `quantize_fp8`. It keeps that contract unchanged.

**Fusion on SM120.** The checkpoint importer fuses the three-op chain into
`tessera_attn.checkpoint_forward_scaled_kv`. Its operands are Q, K codes,
V codes, K scale, V scale and an optional bias.

Fusion requires:
- both K and V to come from single-use `dequantize_scaled` producers of
  function arguments;
- a forward checkpoint;
- static sequences.

A half-scaled chain is refused by name.

The new checkpoint is a sibling of `checkpoint_forward`. The AD and JVP owners
match `checkpoint_forward` by name, so they never admit scaled K/V.

**What the contract carries.** The native contract and the checkpoint identity
both carry:
- `kv_scale_shapes`: K scale extents, then V scale extents;
- `kv_dequantize = "code_times_f32_scale_round_once_v1"`;
- the existing FP8 `kv_storage`.

**Entry, kernel and ABI.**
- The entry is named `tessera_tile_attention_lse_<q>_kv_<fp8>_scaled_[out_<o>_]<digest>`.
- The entry takes Q, K, V, [bias], K scale, V scale, O, LSE.
- The Tile `attention_kernel` takes the two scale pointers after row-LSE, with
  static `kv_scale_shapes`.
- The kernel decodes each code exactly through the sm_120 f8x2 conversion. It
  then multiplies by its scale, indexed from the element's (b, h, s, d)
  coordinates, so no integer division runs per element.
- ABI ids are new:
  `…q_kcodes_vcodes_kscale_vscale_o_row_lse_dims_scale_shape.scaled_kv_f32acc.v1`
  and its bias variant.
- Scalars are B..Dv, the four bias extents when bias is present, and the eight
  scale extents.
- The checked launcher sizes the scale buffers from those scalars.
- Execution goes through the checked descriptor launch. The homogeneous
  retained owner is not used.

**Not admitted:**
- other `dequantize_scaled` uses on SM120, which execute the reference;
- resident FP8 roots;
- dynamic or bounded sequences;
- AD;
- any non-SM120 target.

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
