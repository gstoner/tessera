---
audit_role: reference
last_updated: 2026-10-17
---

# Native parent-view contract

Owner: ROCM-NVFP4-INGEST-1 / FRONTEND-IR-MEDIUM-1 / LAYOUT-ALG-1 / E2E-REAL-6.
Sync: ROCM-NATIVE-GATE-UP-CONSUMER-20261016.

This is a scoped public compiler profile; general layout/composition remains open.
The gfx1201 schema-2 profile takes one static row-major f32 parent[M,2I],
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
NVIDIA lowering before Target projection. These row schemas do not change attention/JVP
schema meanings, including schema 5 = bounded saved-LSE specifically.

The ROCm Target consumes actual typed load/store and arithmetic SSA. Its native
wrapper supplies launch coordinates and checks count/shape consistency.
Native export owns buffer bytes/elements, inclusive read/write lifetimes,
returned output retention and flattened rank-one memref-plus-i64 launch facts.
The existing C++ owner independently derives lifetimes, validates gfx identity,
allocates separate storage, uploads and submits on one owned HIP stream, and
waits before close. Source upload remains a host input in this public profile;
the separate static resident composition below owns its matmul parent. An
arbitrary external-stream contract is not admitted.

The package respects the native owner's aggregate 2 GiB allocation limit:
the f32 parent plus output consume 12*M*I bytes, so M*I must not exceed
floor(2^31/12). The BF16 profile conservatively retains this same count bound
while its parent plus output consume 6*M*I bytes.
The native parent_view family enters from Schedule/Tile on gfx1201 only;
no Graph-to-Target shortcut or Python Tile/kernel emitter is introduced.
The native Target exporter owns the program/member frames and rejects changed
re-exported frames before binary materialization. The checked public descriptor
uses ABI tessera.rocm.parent_view.parent_o_n.f32.v1, original argument/result
names, exact shape guards, N=M*I and the native 256-thread launch geometry.

Portable validation binds Graph, Schedule, Tile and Target bytes to the image
and descriptor. Runtime invocation requires compact profile-matching f32 or bf16 parent/output storage,
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
is f32. BF16 execution is admitted only by schema 3 below; an FP16 parent-view
profile is not admitted. Storage must not be silently replaced by f32 for
model-quality work.

Public evidence and separate device/public timings:
benchmarks/baselines/native_parent_view_public_20261017/README.md.
Earlier candidate evidence retains its original source/tool identities:
benchmarks/baselines/native_gate_up_view_20261016/README.md.


## BF16 storage and explicit FP32 computation

Row schema 3 is the named gfx1201 BF16 parent/output profile. The original
Graph has exactly two complementary slices, two explicit BF16-to-FP32 casts,
one FP32 SiLU multiplication, and one FP32-to-BF16 storage cast. The original
operand order and each cast remain in the source witness. Native MLIR loads
BF16, widens in registers, computes in FP32, and stores with BF16 rounding;
there is no physical allocation for either slice or register cast.

Its ABI is tessera.rocm.parent_view.parent_o_n.bf16_f32.v1: bf16 parent,
bf16 output, FP32 computation. The exported profile name is
gfx1201_parent_view_bf16_f32_v1. Buffer byte spans, dtype, shape, scalar count,
image/descriptor identity, source lineage, and private-stream completion are
validated. Generic owner readback transports the declared BF16 bytes directly.
Seven owning gfx1201 public cases include a fresh process with compiler
subprocesses and eager SiLU forbidden. Static captured-Boolean selection in the
Python frontend preserves the chosen call's SSA; tensor truth and executable
conditions are not evaluated.

## Static resident NVFP4 composition

The independently named tessera.native.nvfp4_program.v5 outlines expansion,
a BF16 stored scaled product, and all six activation operations as three native
members. Five root inputs, two expansion outputs, one private BF16 parent and
one returned BF16 activation make nine physical buffers. Native SSA derives
first-write/last-read lifetimes. The same checked owner stream runs all three
members; the private parent is neither uploaded from nor read back to host.

Two owning-device textual Graph cases prove numerical results, import-once
readiness, parent privacy, changed activations, stale-generation rejection and
retained independent outputs. Existing v3/v4 and legacy bounded programs keep
their meanings; v5 refuses a two-member bounded row frame. Three-member bounded
M/count/grid expressions require a separate verified contract.

Public Python authoring of a BF16 scaled-product store, K16 activation
quantization, down projection, original-checkpoint/full-MLP quality and matched
model-sized performance remain open. This static textual Graph proof does not
close them. Evidence and synthetic timing boundaries are in
benchmarks/baselines/native_bf16_parent_view_20261017/README.md.


## Explicit scaled-product store composition

The queued ROCM-NVFP4-BF16-STORE-20261017 extension retains the frontend's
FP32 scaled product and its explicit BF16 cast in the original Graph and
projected consumer witness. Native partitioning groups that single-use cast
with the product. The existing Graph-to-Schedule pass consumes the internal
tessera.native.fp8_store_cast marker after checking the exact gfx1201 E4M3
K16 FP32-scale contract, static compact rank-2 shapes, SSA ownership and cast
policy. FP32 accumulation is unchanged; conversion occurs once at the BF16
store. There is no allocated FP32 parent or host numerical conversion.

Native program v5 retains its static three-member BF16 activation meaning.
Portable source frontend v1 remains two-member expansion/product; v2 carries
all three member images, descriptors, Graph/Schedule witnesses and receipts.
Bounded three-member replay still requires its own runtime-expression contract.
This extension is under validation and does not establish full-model quality
or performance acceptance.
