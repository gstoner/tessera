---
audit_role: reference
last_updated: 2026-10-17
---

# Native parent-view contract

Owner: ROCM-NVFP4-INGEST-1 / FRONTEND-IR-MEDIUM-1 / LAYOUT-ALG-1 / E2E-REAL-6.
Sync: ROCM-NATIVE-GATE-UP-CONSUMER-20261016.

This is a scoped public compiler profile; general layout/composition remains open.
The explicit gfx1201 profile takes one static row-major f32 parent[M,2I],
two complementary source-preserving slice views, and one SiLU result[M,I].
Gate/up operand order comes from the original Graph SSA. Unknown policy and
layout attributes are refused; each slice uses the shared Python-clipping
normalizer. Schedule seals source, shape, numeric and physical recipe identity.

Tile retains one typed read-only parent and one distinct writable output.
The native scalar body takes parent pointer, output pointer, gid and element
count; logical row count is N/I. Source M is absent from physical code. Every
static package still owns its exact source shape/count/capacity witness.
A byte-identical image across M values does not make either package dynamic.

Row-region schema 1 remains the SM120 two-input/two-output f16/bf16 profile
with its original six-argument ABI and 128-thread schedule. Schema 2 is the
separate gfx1201 f32 parent-view profile with 256 threads. Shared verification
pairs operation name, architecture and schema; a ROCm profile is refused by
NVIDIA lowering before Target projection. Neither schema changes attention/JVP
schema meanings, including schema 5 = bounded saved-LSE specifically.

The ROCm Target consumes actual typed load/store and arithmetic SSA. Its native
wrapper supplies launch coordinates and checks count/shape consistency.
Native export owns buffer bytes/elements, inclusive read/write lifetimes,
returned output retention and flattened rank-one memref-plus-i64 launch facts.
The existing C++ owner independently derives lifetimes, validates gfx identity,
allocates separate storage, uploads and submits on one owned HIP stream, and
waits before close. Source upload remains a host input in this public profile;
no resident matmul producer or arbitrary external-stream contract is proved.

The package respects the native owner's aggregate 2 GiB allocation limit:
parent plus output consume 12*M*I bytes, so M*I must not exceed floor(2^31/12).
The native parent_view family enters from Schedule/Tile on gfx1201 only;
no Graph-to-Target shortcut or Python Tile/kernel emitter is introduced.
The native Target exporter owns the program/member frames and rejects changed
re-exported frames before binary materialization. The checked public descriptor
uses ABI tessera.rocm.parent_view.parent_o_n.f32.v1, original argument/result
names, exact shape guards, N=M*I and the native 256-thread launch geometry.

Portable validation binds Graph, Schedule, Tile and Target bytes to the image
and descriptor. Runtime invocation requires compact f32 parent/output storage,
distinct writable output, the exact scalar count and the owning gfx1201 device.
Completion uses the private native owner stream; arbitrary external streams
and resident input buffers are outside this profile. The host caller Graph is
copied for projection. Serialized replay does not invoke a compiler or eager
SiLU implementation, including in a fresh process with compiler paths absent.

Thirteen RX9070XT/gfx1201 public tests cover literal-static and captured-literal
slice coordinates, both gate/up roles, three shapes, changed roots and retained
results. Captured lists/tuples/dictionaries contain safe literal names only;
executable expressions, subscripts and runtime tensor coordinates are not
evaluated as attributes. Native dynamic_slice parser support is not itself
public execution proof. Native accumulation/storage here
is f32. BF16/FP16 activation stores require an explicit contract and must not
be inferred from this proof or silently replaced by f32 for model-quality work.

Public evidence and separate device/public timings:
benchmarks/baselines/native_parent_view_public_20261017/README.md.
Earlier candidate evidence retains its original source/tool identities:
benchmarks/baselines/native_gate_up_view_20261016/README.md.
