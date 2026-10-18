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
remain checked. Forward bias extents follow the sealed physical policy:
static extents are baked into the image; explicitly dynamic physical extents
are passed as indices. Implicit logical bias follows the runtime score shape.

Source-backed ResidentAttentionTape.prepare_jvp retains validated native data
and output storage from its captured forward generation and supplies them to
the AD materializer. Adapter defaults cannot replace the captured policy.

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
now have separate public proofs. The ordinary native prepared host owner
also admits the bounded sequence contract described below. Wider layouts,
lifetime composition and derivatives remain open before broader closure.
No numerical or performance proof transfers to ROCm, Apple or x86.

## Bounded prepared host ownership

Synchronization key: SM120-BOUNDED-HOST-JVP-20261017.

An explicit fn.native_jvp(..., sequence_bounds=(Sq_capacity, Sk_capacity))
uses the existing native Graph AD, Schedule, Tile, NVIDIA Target and LLVM/PTX
compiler route. Both bounds are positive integers and this physical admission
requires nvidia_sm120. An AST/tracer structural certificate checks topology and
numerical policy without executing the concrete frontend. A Python policy
that changes with the traced sequence length cannot define this envelope.

The prepared C++ owner retains the modules, a capacity-sized device arena and
pinned staging. Runtime sequence lengths must be positive and within bounds;
fixed batch, head, feature and storage dimensions must match. Each invocation
projects current byte spans, grids, shared-memory size and sizing indices
(scratch, query_size, key_size). Warm shape changes reuse the same program and
owner. The bounded cache also keys the sealed native symbolic primal shapes
and dtypes, alongside bounds, activity, source identity and the actual Thread
object. Each fixed batch/head/feature/storage signature has its own owner.
Physical bias contracts participate in matching, including degenerate
broadcasts. Runtime capacity refusal stays with the selected owner and does
not trigger a new compilation. Recent compatible owners are retained.
Producer events order resident-input snapshots. Returned results are
independent completed host arrays. Thread ownership, context/fork guards,
close and draining retain their existing lifetime contracts.

The additive prepare_bounded ABI carries storage/output IDs, saved-LSE policy
and bias_shape_arguments. That flag is zero for static physical or implicit
logical bias, and one for explicitly dynamic physical bias. The forward image
takes 12 parameters without bias, 13 with static/implicit bias, or 17 with
explicitly dynamic bias. The native owner checks the actual CUDA parameter
count, offsets and eight-byte widths before exposing a handle. Tangent launches
carry their pointer roles plus three sizing indices. Legacy static preparation
and invocation exports preserve their contracts and cannot launch a bounded
handle through the static ABI.

Owning RTX5070 evidence includes FP16/BF16/F32, authored FP32 outputs,
query/key/logical bias, trace-at-one grow/shrink/reuse, malformed ABI refusal
with healthy recovery, and pending producer writes. Paired timing covers
FP16/BF16 saved-LSE causal profiles at Sq/Sk 1/1, 7/9, 31/65 and 127/131,
with absent or scalar physical bias. Completed public calls and CUDA event
dispatch windows are reported separately. Wider masks, mixed/quantized
storage, composed AD and resident-result ownership remain open.

Evidence: benchmarks/baselines/sm120_bounded_host_jvp_20261017/README.md.
