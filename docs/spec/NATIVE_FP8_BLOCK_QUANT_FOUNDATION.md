---
last_updated: 2026-10-17
audit_role: reference
---

# Typed native FP8 block quantization foundation

Owner: ROCM-NVFP4-INGEST-1 / ROCM-FP8-BLOCKSCALE-1 / FRONTEND-IR-MEDIUM-1.
Sync: ROCM-NATIVE-E4M3-CAST-20261017.

## Implemented storage primitive

The existing Target-to-ROCDL pass legalizes scalar FP32-to-E4M3 RNE casts
inside exact gfx1201 GPU modules. It emits the archived RDNA4 packed
conversion and retains typed FP8 storage. Directed rounding, host functions
and other architectures are unchanged. Existing public fake-quant semantics
and target admission are unchanged.

Owning-device finite encodings and signed zeros match the independent oracle
bitwise; NaNs match semantically. The packet covers 266,560 values and preserves
changed-input/retained-output ownership. This is a GPU storage primitive,
not proof of a public Graph quantizer or complete native MLP.

## Explicit semantic Graph, Schedule, Tile and native Target

The native Graph-to-Schedule pass now validates the complete explicit textual
source, including tensor broadcasts, IEEE scale-floor constants, SSA edges,
output shapes and RNE storage conversion. A content-sealed Schedule artifact
retains the original Graph. Schedule-to-Tile now projects the actual source
SSA into a sealed schema-4 body with three typed non-owning views, one K16
FP32 maximum loop, one scale store and one E4M3 encoding loop. Logical
reshapes and broadcasts allocate no storage. The physical body accepts three
pointers, a group index and a group count; its identity contains no source
M/I constants. Source-specific extents remain in the sealed envelope. Native
Target lowering now materializes BF16/byte/FP32 accesses from that body and
exports a checked three-buffer/two-output frame. The GPU adapter checks flat
memref spans and launches the cloned SSA. Matching LLVM serialization emits
HSACO; three exact gfx1201 envelopes pass the independent oracle through
the diagnostic C++ multi-output owner. That packet establishes textual Graph
input only. The later canonical quantizer and resident-product sections below
record their separate public execution proofs; the diagnostic packet itself
does not establish those boundaries. Schedule profile schema 4 belongs to the
row-region namespace; attention schema numbering is unchanged.

For compact BF16 input X[M,I], with positive static extents and I divisible
by 16, define groups along the last axis. Each group widens exactly to FP32,
then explicitly computes:

1. A = max(abs(X)) over the 16 elements, propagating NaN.
2. S = maximum(A / 448, smallest_positive_normal_FP32).
3. Q = cast_E4M3_RNE(X / S).

Return Q[M,I] in real E4M3 storage and S[M,I/16] in FP32 storage.
These operations must remain explicit semantic Graph SSA, including reshape,
reduction, constants, normalization and storage conversion. Native scalarization
and fusion must consume those operations rather than reconstructing their
meaning from Python flags. The existing quantize_fp8 reference operation
continues to return a dequantized reference payload and is not reinterpreted.

Zero groups retain the minimum-normal scale and signed-zero codes. NaN and
infinity behavior follows the explicit maximum/division/cast expression;
a native implementation must test those cases instead of assuming finite
checkpoint data establishes general behavior. BF16 subnormals and FP32
denormal behavior at the scale floor need exact-device proof.

## Typed Tile verification

Fourteen focused K16 cases cover three source shapes, exact source policy,
changed Schedule metadata and a tampered scalar body. Adjacent parent-view,
FP8 cast, canonical dtype, tensor-attribute, diagnostic, pass-metadata and
audit gates pass 500 combined cases after a clean incremental rebuild with no
source-mtime warning. Physical body hashes are identical across the three
source shapes while source and artifact seals remain distinct.
Those checks establish compiler artifacts; the separate native Target/device
packet below establishes its named diagnostic execution envelope.

## Required integration gates

- Verified Graph shape, dtype, effects, axes, scalar constants and RNE policy.
- Native Schedule/Tile with one read-only input and two typed outputs;
  no allocated reshape, absolute-value or normalization intermediates.
- Checked mixed-width ABI, byte counts, output roles, aliasing and lifetimes.
- Owning gfx1201 numerical checks for zero, signed zero, ties, subnormals,
  scale-floor boundaries, overflow, NaN and infinity.
- Public authored JIT and portable compiler-disabled two-result replay.
- Resident activation input without intermediate host readback; encoded Q
  and S feed the down projection through checked native program ownership.
- Separate quantizer/member/event/completed-host timings and pinned checkpoint
  comparison before full-MLP or model-quality/performance acceptance.
- Four-backend assessment; no gfx1151 or sibling physical proof inheritance.

Evidence for the implemented primitive:
benchmarks/baselines/rocm_native_e4m3_cast_20261017/README.md.

Native Target/device evidence: benchmarks/baselines/rocm_native_k16_quant_20261017/README.md.

## Canonical authored quantizer package

The distinct gfx1201 `block_quant` family now packages the verified typed Tile
body through tessera-rocm-executable. Ordinary Python JIT returns real E4M3
codes plus FP32 scales through the checked C++ two-output owner. Serialized
artifacts replay in a fresh process with no compiler. Immutable FP32 splats
have shared AST/trace value snapshots and canonical upstream arith.constant
SSA; reduction axis/keepdims inference is shared by both frontends. The
upstream operation has explicit coverage and stays in whole-program gates.
Only the exact named source contract is executable; this does not expose
standalone view/cast/constant kernels or alter fake-quant reference semantics.

Exact public and canonical-image numerical/timing evidence:
benchmarks/baselines/rocm_native_k16_quant_public_20261017/README.md.
Resident activation/down projection and full MLP/model-quality gates remain open.


## Native resident quantizer/product program

Sync: ROCM-RESIDENT-QUANT-PRODUCT-20261017.

A module marked `tessera.native.quantized_product` with target
`rocm_gfx1201` and architecture `gfx1201` may contain the original thirteen
quantizer operations followed by one exact scaled product. The native exporter
validates the quantizer through the same `BlockQuantContract.h` matcher as the
standalone Schedule path. It clones the original operations into two members;
Python consumes those projected members and never reconstructs their Graph,
Tile arithmetic or numerical policy.

The three input roles are BF16 X[M,K], E4M3FN B[N,K] and FP32 SB[K/16,N].
The product consumes Q[M,K], B, S[M,K/16] and SB, with `transposeB=true`,
`scale_layout={granularity="block", block=[1,16], format="fp32"}` and
`numeric_policy={accum="fp32", execution_mode="exact_per_block"}`. Unsupported
attributes or changed policies are rejected before source partition. Public
proof covers static FP32 output; BF16 output has separate prepared-route proof.

The permanent SSA manifest schema is `tessera.native.quantized_product.v1`.
Root buffer IDs follow the original function argument order, while role
indices are derived from actual SSA uses. IDs 3 and 4 are the private Q/S
outputs, and ID 5 is the returned product:

| Role | Ownership | First write | Last read |
|---|---|---:|---:|
| X | read-only root | -1 | 0 |
| B, SB | read-only roots | -1 | 1 |
| Q, S | private scratch | 0 | 1 |
| Product | returned output | 1 | 2 (program return) |

The portable package and frontend schemas are respectively
`tessera.rocm.quantized_product_package.v1` and
`tessera.rocm.quantized_product_frontend.v1`. They retain the original Graph,
projected Graph/Schedule/Tile/Target witnesses, member images and checked
member descriptors. The quantizer uses its existing mixed-width two-output
ABI; the product uses its existing four-input/one-output scaled-matmul ABI.
A single checked C++ multi-output owner marshals their flattened rank-one
memrefs, scalar arguments, launch geometry and private stream ordering.
Producer frame shapes, storage and byte extents must match the shared buffers;
individually valid members with incompatible interfaces are refused before
provider access. No private Q/S host readback occurs between members.

Ordinary JIT calls reuse the verified private program with current host roots.
Artifact construction is an inspection/export boundary. Portable launches
validate serialized Graph/argument/stage witnesses and retain normal checked
execution; mutating an inspection artifact cannot change the private JIT
program. The new consumer uses unprojected checked native Target IR rather
than Python shape-free Target rewriting.

Evidence and distinct public, portable-control, captured-member, ordinary
HIP-event and prepared-invoke/read timing domains:
benchmarks/baselines/rocm_resident_quantized_product_public_20261017/README.md.

This names one compiled source envelope. Combining the quantizer with NVFP4
weight expansion and the complete gate/up/activation/down MLP, general layouts,
bounded shapes, encoded AD and model-quality acceptance remain open. No
sibling architecture or attention schema is promoted or renumbered.

## Native quantizer / NVFP4 expansion / product v2

The namespace `tessera.native.quantized_product.v2` adds weight expansion to
the original quantizer/product SSA partition. It does not change v1 or any
attention/NVFP4-program schema numbering. The requested Graph still uses the
`tessera.native.quantized_product` unit marker; the native exporter derives
v1 versus v2 from the verified original SSA and root/member structure.

The v2 Graph has four distinct original roots: BF16 X[M,K], packed NVFP4
byte weights[N,K/2], E4M3 scales[N,K/16], and FP64 projection globals[G].
The existing expansion policy and complete increasing projection row boundaries
remain mandatory. The original thirteen-operation K16 quantizer is validated
by the same native semantic contract as standalone quantization. The expansion
feeds the product's RHS codes and scale plane directly; the quantizer feeds
its LHS codes and scale plane. Conflicting policies and wrong SSA edges are
rejected without numerical normalization or reconstruction.

| Frame | Ownership | First write | Last read |
|---|---|---:|---:|
| Original X root | readonly input | -1 | 0 |
| Three weight roots | readonly input | -1 | 1 |
| Q and SA | private scratch | 0 | 2 |
| Expanded B and SB | private scratch | 1 | 2 |
| Returned product | returned output | 2 | 3 |

The compiler derives original root-role indices, byte extents and all three
member Graphs. Each member is projected through native Schedule/Tile and ROCm
Target IR to LLVM/HSACO. The portable package and frontend use their respective
`quantized_product_package.v2` and `quantized_product_frontend.v2` namespaces;
v1 continues to describe exactly two members. The consumer's checked Target
witness must equal its image-bound Target, and both producer interfaces must
match the shared frame in shape, storage and bytes. Portable replay performs
these checks without requiring a compiler.

The existing multi-output C++ HIP owner receives all three existing member ABIs,
allocates the four private planes, validates producer readiness, and invalidates
private generations after root updates. Python validates current expansion
scales/globals and marshals the compiler-derived bindings; it performs no
intermediate numerical lowering or host readback. Successful current receipts
establish the public execution kind; prior-call receipts cannot describe a
rejected new specialization. Reports retain all three Schedule/Tile/Target
witnesses. The distinct execution path is
`canonical_rocm_quantized_nvfp4_product_program`.

The owning public numerical certificate is the named static FP32-output,
single-projection fixture at M/N/K=1/1/16, 3/17/32, 19/33/64 and
256/128/12288. Native component emission admits additional envelopes; this
certificate does not promote them. Six device tests additionally cover fresh
process replay, unchanged roots, retained outputs, absent producers and root
update invalidation. Source-pinned captured-member, ordinary HIP-event,
prepared invoke/read and public timings remain distinct. Full gate/up/
activation/down MLP, model quality, dynamic/general layouts, wider composed
storage envelopes and encoded AD remain open.


## Four-member quantized gate/up program (v3)

`tessera.native.quantized_product.v3` composes the original BF16 K16
quantizer, NVFP4 expansion, exact FP32 scaled product with an explicit BF16
store, and two parent views feeding FP32 SiLU-multiply with a BF16 result.
Native MLIR validates the original SSA and derives member boundaries; Python
marshals the checked native plan. Native/package/frontend v1 and v2 retain
their existing contracts.

The default frame has four read-only roots: X, packed weights, E4M3 weight
scales and two FP64 projection globals. Private Q/SA (slots 4/5) are written
by member 0 and read by member 2; expanded B/SB (6/7) are written by member 1
and read by member 2. BF16 parent (8) is written by member 2 and read by member
3. BF16 output (9) is returned. Root order is derived from original argument
roles. Logical views allocate no buffers, and the FP32 accumulator does not
escape its product/store member. Readiness and root updates invalidate
private results through the checked native owner.

Certified internal projection attributes
`tessera.native.quantized_product_projection` and
`tessera.native.quantized_product_projection_name` bind each actual SSA
partition to its existing semantic validator; both are stripped before
component lowering. Every member descends through Schedule/Tile/Target into
HSACO. Portable restoration checks all four images, interfaces and shared
frame lifetimes without requiring the compiler.

The public path is `canonical_rocm_quantized_gate_up_program`. Exact-device
proof is scoped to gfx1201 static compact BF16 inputs and the named K16
quantization policy. FP32 product accumulation precedes the explicit BF16
parent store; activation widens that stored parent, preserving its rounding.
Random numerical comparisons use BF16 tolerances; separate analytic cases
prove bitwise parent rounding and both SiLU operand orientations. Full down
projection, wider layouts/dynamic shapes, encoded derivatives and model
quality remain open.


## Seven-member full MLP native partition (v4)

`tessera.native.quantized_product.v4` reuses the validated v3 gate/up prefix
and v2 quantizer/expansion/product suffix over the original Graph SSA. Seven
read-only roots contain the input plus independent gate/up and down packed
weights, scales and projection globals. Seven members derive eighteen
buffers: roots 0..6, first Q/SA 7/8, first B/SB 9/10, BF16 parent 11, BF16
activation 12, second Q/SA 13/14, second B/SB 15/16 and returned output 17.
The activation is written by member 3 and read by member 4; second quantized
planes are read by member 6. The actual product result type controls the
final FP32 or BF16 store under the existing exact product contract.

A common native SSA outliner derives root order, private allocations and
first-write/last-read bounds from certified original operation groups. Both
quantizers and expansions retain their original math and numerical policies.
The native partition was established first; prepared-owner evidence is scoped
below. Public frontend admission, production runtime replay, separate timings
and broader numerical/model acceptance remain required.

### Prepared v4 owner and numerical boundaries

The compiler exports two base64 native component plans in `subprograms`: v3
gate/up and v2 down projection. The v4 package validates those plans, their
seven member graphs and the shared activation interface against the outer
eighteen-buffer frame. It binds all seven steps into one multi-output C++ HIP
owner. Component owners are used only in diagnostic comparisons.

On RX9070XT/gfx1201, three prepared shapes (M/H/K/N = 1/16/16/1,
3/32/48/19 and 256/128/12288/256) pass original-root, compiler-free restored
and retained-owner updated-root checks. Updating roots invalidates downstream
readiness; old returned arrays remain stable. This is prepared execution, not
yet the public authored frontend route or fresh-process full-MLP replay.

Independent checks follow explicit rounding boundaries. FP64 product
references validate the BF16 parent store; the activation is checked against
FP32 SiLU applied to the actual stored parent. The resident result is bitwise
equal to the same native components run separately. The down product is
checked against independent math at its actual activation input. The final
comparison bounds the quantized-input perturbation by the absolute decoded
weight product, with explicit terminal-store rounding and the unchanged
per-stage numerical allowance. A pointwise relative tolerance against an
ideal earlier activation alone is insufficient across discontinuous
requantization and nonlinear SiLU. No numerical policy or physical kernel
was changed to pass these cases.

### Authored public MLP and final BF16 store

The flat public Python function carries both quantizers and both NVFP4
projections in its original Graph. Its final FP32 product is followed by an
explicit BF16 cast: the 38-operation form joins the earlier 37-operation
direct-result form. The down product/cast remain one physical member using
the same pure, single-use FP32-to-BF16 validator as the gate/up store. The
three-member v2 component frame retains its schema and storage semantics.

`canonical_rocm_quantized_mlp_program` binds native/package/frontend v4 through
the existing program executor. Eight exact-gfx1201 tests prove public JIT,
compiler-disabled warm and portable replay, fresh-process replay, current
seven receipts and nested inspection isolation. Three numerical/timing
packets preserve store-boundary and propagation certificates, including flat
FP64-reference disagreements. This is a named static public profile, not
checkpoint/model-quality, dynamic, encoded-AD or sibling physical closure.
Evidence: benchmarks/baselines/rocm_quantized_mlp_20261017/README.md.
