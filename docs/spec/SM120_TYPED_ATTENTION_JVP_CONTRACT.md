---
last_updated: 2026-10-17
---

# SM120 typed attention JVP contract

Owner: E2E-REAL-6 / FRONTEND-IR-MEDIUM-1 / AD-RESIDUAL-EVAL-1.
Synchronization key: SM120-TYPED-JVP-20261017.

## Semantic and physical roles

The compiler derives the tangent from the authored attention operation through
Graph AD, Schedule, Tile and NVIDIA Target IR. Python binds the resulting
native image and checked manifest; it does not implement the derivative.

Q, K, V and their active tangent operands share one storage type: f16, bf16 or
f32. O and dO use the same storage as Q/K/V or f32. Bias, its tangent, saved
LSE and dLSE remain f32. Inactive tangent roles retain the corresponding data
type; their native loads are eliminated. GQA mapping, causal alignment and
physical broadcast extents follow the forward contract.

For P = softmax(S), dS = scale * (dQ K^T + Q dK^T) + dBias:

- dLSE = sum(P * dS).
- dO = sum(P * dS * V) + sum(P * dV) - sum(P * V) * dLSE.

With a half-precision O store, the correction must use the unrounded FP32
weighted-V moment. Reusing the stored, rounded O changes the derivative.
The half-output score-active kernel accumulates that moment in the existing
key loop from the same V load, using a third FP32 shared reduction vector.
Value-only tangents do not require the score correction. Existing f32-output
kernels retain the stored-f32 O correction. Final output conversion occurs
after FP32 arithmetic.

## Identity, serialization and ABI

Native AD schema numbers retain their existing meaning: schema 5 is saved-LSE
plus bounded sequences specifically; schema 6 is static saved-LSE. Typed
contracts additionally seal storage, output_storage and row_lse_storage.
Role-specific manifest types and the typed checkpoint identity must agree
with the forward package. An existing f32 artifact does not acquire typed
semantics through a descriptor rewrite.

The new native prepare_typed entry takes explicit storage IDs (1=f32, 2=f16,
3=bf16), output storage and a boolean saved-LSE flag. Its typed entry admits
half data; legacy prepare entries retain f32 semantics. Q/K/V and their
tangent buffers use two bytes per element for half data. O/dO follow output
storage; bias and all row quantities retain four bytes. The complete canonical forward symbol and its digest suffix must match the
output storage; a common prefix is insufficient. Grid calculations use
element counts before byte-width projection. Producer stream dependencies,
allocation capacities, snapshots, retained outputs and context ownership
remain checked. Typed forward bias extents are passed explicitly, including
degenerate physical broadcasts.

## Evidence and remaining envelope

Exact-device execution is required on RTX5070 / sm_120. The current static
test envelope covers half/BF16 output, GQA, saved LSE, causal/open attention,
broadcast bias, inactive tangent roles, public compilation, portable resident
replay and ordinary prepared host calls. Raw member tests also cover f32
output with half inputs. These are separate proofs; raw member success does
not establish the complete public route for every authored output policy.

Four exact-device cases also order pending writes from independent producer
streams through the typed prepared ABI. Eight correctness-gated timing rows
characterize tiny and longer static sequences with common quantized-value f32
controls, keeping CUDA event windows separate from completed public calls.
Bounded typed sequence replay, trace-at-one shrink/grow/capacity checks,
fresh-process compiler-absent static/bounded replay and authored FP32 output
now have separate public proofs. The ordinary native prepared host owner is
static; bounded programs use portable resident generations. Wider layouts,
lifetime composition and derivatives remain open before broader closure.
No numerical or performance proof transfers to ROCm, Apple or x86.
