---
last_updated: 2026-10-17
audit_role: reference
---

# SM120 prepared attention row-residual owner

Owner: E2E-REAL-6 / AD-RESIDUAL-EVAL-1 / FRONTEND-IR-MEDIUM-1.
Synchronization: SM120-NATIVE-ROW-CORRECTION-20261016.

## Scope

This additive native C ABI owns forward, FP32 row correction and backward images
produced through the original Graph/native AD -> Schedule -> Tile -> NVIDIA/LLVM
route. It initially admits static compact FP16/BF16 QKV, matching half or FP32
dO, optional physical FP32 broadcast bias and optional FP32 dLSE seed. The row
producer computes ordered FP32 probability-weighted dot(dO,V) minus that seed.
The consumer uses the explicit residual and never subtracts the seed twice.

The native owner and portable VJP v3 public program execute through the original
frontend and compiler-free replay. See the public program section below.
Existing VJP v1/v2 and JVP schema 5 bounded / schema 6 static are unchanged.

## Private frame

| Slot | Role | Physical storage |
| --- | --- | --- |
| 0–2 | Q, K, V snapshots | FP16 or BF16 |
| 3 | Forward O | Matching half or FP32 |
| 4 | Saved natural-log LSE | FP32 |
| 5 | dO snapshot | O storage |
| 6 | Optional bias snapshot | FP32 |
| 7–9 | Active dQ, dK, dV | QKV storage |
| 10 | Active physical dBias | FP32 |
| 11 | Optional dLSE snapshot | FP32 |
| 12 | Row correction | FP32, absent for dV-only activity |

One checked aligned device arena and pinned host staging frame remain owned
until close. Root snapshots and all stages share one private nonblocking stream.
Resident roots require producer streams, matching context, alignment and allocation
capacity; private arena aliases are rejected. Native events order pending producer
writes before snapshots. Errors drain queued work before storage is reused or freed.
Output copies complete before return. Fork and context-generation guards remain.

## Additive entry points

`tessera_nvidia_attention_vjp_rows_prepare` accepts three image/size/entry triples,
seven logical dimensions, physical bias extents, frontend permutation, requested
physical gradient order and storage/seed policies. Compiler-generated symbols
encode storage, bias, seed, activity and block geometry. Actual compiled parameter
widths, offsets and exact count are checked using the CUDA driver. The producer
uses exactly 32 threads; direct Tile verification rejects producer 64/128
thread overrides as well. The consumer uses its sealed 64/128-thread geometry
and rejects 32 threads.

For exactly dV-only activity, preparation requires the row image triple to be
absent. No row module, row allocation or row launch is required. Its verified
consumer does not read the row argument. `rows_info` returns logical row bytes
and stage count without exposing private addresses.

`rows_invoke` and `rows_invoke_resident_ordered` require an explicit timing output
capacity. With a nonnull timing pointer the capacity must be exactly four floats:

1. forward event window;
2. row producer event window, zero when elided;
3. backward event window;
4. complete native device interval from forward start through backward stop.

A null pointer with capacity zero suppresses timing output. Wrong capacities and
mixed old/new invoke interfaces refuse before output writes. The legacy invoke
still writes only its original two floats. `last_device_window` can query the
same complete interval for either reverse ABI after a successful completed call;
it refuses before a completed invocation. Thus inline and prepass comparisons
use the same device interval definition without widening legacy timing buffers.

## Evidence and remaining gates

Direct owning RTX5070 proof covers changed inputs, half/F32 dO, bias/seed variants,
requested gradient order, compact geometry, pending producer streams, value-only
elision, timing sentinels and an actual compiled parameter-count mismatch.
Existing public reverse and shared JVP/VJP stream-isolation proof checks provider
compatibility. The diagnostic native-owner A/B uses identical stored inputs and
bit-identical forward PTX, alternating arm order, correctness checks outside the
wall timing interval and separate stage/device/completed-call scopes. This is not
a public @jit latency measurement and does not promote a global default.

All-member portable provenance, original frontend replay, output generations
and public-call benchmarking are now recorded below. Broad gates/CI and
automatic policy selection remain open.

Evidence: benchmarks/baselines/sm120_native_row_correction_20261016/prepared_owner/README.md.


## Portable public program and selection

VJP program schema v3 retains forward, optional correction and backward members.
The row member is absent exactly for dV-only activity; otherwise all three image,
ABI, native-policy, compiler/toolchain and source-pair identities must agree.
The canonical pinned digest covers every executable member. Retained Tile and
Target dictionaries are checked against regenerated descriptors without invoking
a compiler. Full-shaped bias may be implicit in forward metadata; comparison
uses its actual physical extent. Dynamic shapes are not admitted by this rule.

The original synchronous JIT/native backward selects this program with
`TESSERA_NVIDIA_ATTENTION_ROW_CORRECTION=prepass`; `inline` remains the default.
Policy is part of compiler/prototype cache identity. Warm replay, resident ordered
snapshots and fresh-process RuntimeArtifact restoration use the checked native
owner. This does not introduce a row-program capture API or widen dynamic support.
VJP v1/v2 and JVP schema 5 bounded / schema 6 static retain their meanings.

Public proof and separate public/device/stage timings:
benchmarks/baselines/sm120_native_row_correction_20261016/public_program/README.md.


## Graph replay experiment and public dispatch attribution

Synchronization: SM120-NATIVE-ROW-GRAPH-20261016.

An unmerged static native graph replay candidate passed 50 RTX 5070 gates,
including FP16/BF16 host/resident gradients, independent FP64 checks, delayed
producer ordering, fresh root/stream rebinding, retained outputs, invalid-call
recovery and injected driver graph-failure retirement. Two matched eight-profile
timing trials showed no consistent completed-call gain. The candidate APIs and
Python flag were removed from production; the exact source patch and measured
recorder are retained for reproduction. Existing submission and schemas remain.

The public/prepared timing difference is not causal attribution. A separate
instrumented 200-call profile shows repeated Graph IR serialization, certificate
construction and path resolution in public dispatch. Native blocking time is
included in PreparedAttentionVJP.invoke. Those profiles are diagnostic, not
speedup measurements. Further integration must retain mutation checks and
original frontend/Graph authority while avoiding redundant work.

Evidence: benchmarks/baselines/sm120_native_row_graph_20261016/README.md.


## Bounded sequence extension — portable VJP v4

Sync: SM120-BOUNDED-ROW-OWNER-20261017. Only Sq/Sk may be symbolic, with
positive capacity bounds preserving all fixed dimensions. Native Graph, Schedule
and Tile verify this envelope before Target lowering. `row_bytes` in the sealed
native dictionary is the maximum capacity span; a concrete native frame owns
only its actual row extent. Role-specific query/key/bias guards remain mandatory.

VJP v4 means a bounded row program; v3 continues to mean a static row program.
Re-pinning an envelope after relabelling these schemas refuses. All member
images, paired lineage, policies, physical bias extents and bounds contribute to
the immutable program identity. Existing JVP schema 5/6 meanings do not change.

Preparation still takes seven concrete dimensions, after checking them against
the portable envelope. Its thread-local bounded cache may retain a distinct
native frame for each concrete shape while reusing the same images. Native
parameter introspection distinguishes forward seven-scalar versus eleven-scalar
bias ABIs. Row producer/consumer have exactly seven logical scalars independently
of the forward tail; their bias strides derive from runtime logical dimensions.

`compile_native_attention_vjp(..., compact_gradients=True,
row_correction=True, sequence_bounds=(SqCap, SkCap))` exposes the original
frontend route explicitly. Restored programs support public native replay and
resident `capture`; existing typed input, producer-stream, private snapshot,
result retention, asynchronous release and close rules apply. A row buffer is
regenerated for each reverse call. dV-only consumers omit the producer and do
not read their row operand. No automatic schedule promotion is implied.

Evidence: benchmarks/baselines/sm120_bounded_row_owner_20261017/README.md.
Matched bounded inline/prepass A/B now records long-row gains and tiny host
regressions; no default promotion follows. Wider masks/layouts and composed
AD remain open.


## Bounded immutable witness digest reuse

Sync: SM120-AD-WITNESS-DIGEST-20261017. Every member validation still
requires exactly one literal `tessera.attention_ad_pair` attribute, outside
any larger qualified name. Only its immutable encoded-string digest may be
reused. The process-local LRU admits at most 32 strings of at most 256 KiB
UTF8 each; oversized strings decode without caching, and PID changes replace
the cache. Invalid decoding never populates it. No certificate, metadata,
shape, native handle, allocation or stream validation is skipped.

Evidence: benchmarks/baselines/sm120_ad_witness_digest_20261017/README.md.
