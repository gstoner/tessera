---
last_updated: 2026-10-16
audit_role: reference
---

# Native SM120 typed saved-LSE backward contract

Owner: E2E-REAL-6 / FRONTEND-IR-MEDIUM-1 / AD-RESIDUAL-EVAL-1.
Synchronization key: SM120-SAVED-LSE-HALF-BACKWARD-20261016.
Status: static typed public owner candidate; final validation and publication remain open.

## Compiler foundation

Original Python/textual attention -> typed Tessera Graph MLIR -> registered
native reverse AD -> checkpoint Graph -> verified Schedule -> Tile -> NVIDIA
Target IR -> LLVM/NVVM/PTX -> owning RTX5070 kernel execution.
Python does not synthesize derivative arithmetic or promote input matrices.

Q/K/V share FP16, BF16 or FP32 storage. Primary output and dO share input
storage or FP32. Q/K/V gradients use input storage after FP32 accumulation.
LSE, LSE cotangent, bias and bias gradient remain FP32. Mixed Q/K/V storage
is rejected. Half semantic admission requires nvidia_sm120 and sm_120; sibling
physical contracts and the separate tangent interface are unchanged.

## Residual precision

For half inputs, the native contract records storage, output_storage,
gradient_storage, row_lse_storage and accum. It additionally seals
row_correction="probability_dot_v_f32_v1" into the Schedule hash and Tile op.
The Tile verifier and NVIDIA materializer require this named policy and
compatible primary storage before generating executable IR.

The backward row correction is sum(P * dot(dO,V)), accumulated in FP32 from
the saved FP32 LSE and original input values. It does not compute dot(dO,O)
from a rounded public half result. The existing FP32 path retains its saved-O
row correction. An LSE seed subtracts from the correction before the standard
P*(dP-delta) score derivative; only Q/K/bias receive that contribution.
Bias and LSE seeds use four-byte loads; bias gradient uses four-byte stores.

## Typed public owner and serialization

The additive backward descriptor ABI is
`tessera.nvidia.sm120.attention_backward_lse.typed_f32acc.v1`. Buffer roles
seal primary dO, Q/K/V, saved O, optional bias, FP32 LSE/seed and requested
physical gradient outputs independently. Native contract hashes, shape guards,
entry symbols, byte widths, geometry, numeric policy and saved bindings are
checked without invoking compiler tools when restoring the portable program.

Half-storage programs use `tessera.native_attention_vjp_program.v2`.
Existing FP32 programs retain v1, including saved-LSE cotangents. JVP schema5
still means saved-LSE plus bounded sequences; schema6 remains static. No JVP
schema is reinterpreted by this reverse-only extension.

`tessera_nvidia_attention_vjp_prepare_typed` is an additive C registration
entry. Storage codes f32=1/f16=2/bf16=3 and the LSE seed flag must agree with
both exact native entry spellings before module loading or allocation. The
original FP32 prepare signature remains unchanged. Native ownership retains
private roots, O/LSE and gradient scratch, uses per-role byte widths and
logical element counts for grids, orders producer streams before snapshots,
and returns independent completed host gradient arrays.

Ordinary `native_backward` supports static compact host roots or all-CUDA
roots, including requested gradient subsets and tuple primary/LSE seeds.
Data matrices are never promoted to FP32 in Python. Bias/LSE roles stay FP32.
Textually authored half inputs with FP32 primary output also execute through
the same checked owner at 64/128 reverse block sizes. Library/device discovery
precedes the strict subprocess guard; first package preparation and launch
remain inside the guard during cold portable replay.

Explicit GPU-generation `program.capture()` remains the existing FP32 tape
API. Half JVP, dynamic resident frames, pitched/mixed Q/K/V storage, resident
returned gradients and wider composed/mask contracts remain required follow-ups.
Unowned raw half backward submission is not the public composed owner route.

## Proof and delivery boundary

The frozen native stage on 9b2159e655 records 639 focused cases and 167 owning
RTX5070 cases with zero skips/failures and unchanged source/tool witnesses:
benchmarks/baselines/sm120_saved_lse_half_backward_20261016/native-stage/README.md.
Its 96 new direct-kernel cases retain their original source scope.

The public candidate separately passes 96 half host/resident/portable cases,
48 FP32 compatibility cases, twelve native binding rejection/recovery cases,
and sixteen textually authored FP32-primary cases. These are separate test
scopes, not a replacement frozen full-source certificate. Cases cover causal
and noncausal, batching/MHA/GQA, requested subsets, trace-at-one, long keys,
Sq>Sk, bias and LSE seeds. Independent FP64 comparisons use identical stored
inputs; delayed producer writes, held result generations and compiler-free
warm calls/replay are checked. Final source-bound receipts remain active.

`benchmarks/nvidia/record_typed_attention_backward.py` records alternating
host/resident completed public calls and separate forward/backward CUDA event
windows. FP32 controls expand identical half-quantized operand values. Raw
u16 inputs, artifact identities, input hashes, per-role numerical errors,
source/tool/provider pins and fresh-process compiler-free replay support
reproduction. Event windows exclude snapshots/downloads and may include native
dispatch gaps. No default or general speedup promotion follows from a timing
sample. Final isolated measurements, full unit/compiler-route gates and CI/
review are still required before publication. The five-slice goal remains open.
