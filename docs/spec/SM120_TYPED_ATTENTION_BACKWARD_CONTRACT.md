---
last_updated: 2026-10-16
audit_role: reference
---

# Native SM120 typed saved-LSE backward contract

Owner: E2E-REAL-6 / FRONTEND-IR-MEDIUM-1 / AD-RESIDUAL-EVAL-1.
Synchronization key: SM120-SAVED-LSE-HALF-BACKWARD-20261016.
Status: compiler/kernel candidate; public runtime integration remains open.

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

## Proof and delivery boundary

The initial native compiler lane passes 118 cases. The expanded focused
AD/op/dtype/diagnostic/pass lane passes 583. Ninety-six owning RTX5070
kernel cases compare original-Graph compiled forward/backward with an
independent FP64 oracle from identical stored inputs. Guarded two-/four-byte
outputs and poisoned saved output check physical stores and residual use.
These are native kernel tests, not public package/runtime ABI certification.
The matrix covers causal/noncausal, three static shapes, batching/GQA,
trace-at-one, Sk=129, Sq>Sk, bias and optional LSE seeds. Final source/tool
witnesses and delivery receipts remain active.

Public typed backward packaging, retained native ownership, producer-stream
ordering, standalone compiler-free replay, independent returned generations,
benchmarks and full publication gates remain open. Existing JVP schemas5/6
and legacy FP32 AD tape are unchanged. All four backend queues record their
independent physical obligations. The original five-slice goal stays open.
