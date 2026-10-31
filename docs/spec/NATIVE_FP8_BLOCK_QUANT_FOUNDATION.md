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

### Bounded-row v5 MLP (GFX1201-NVFP4-BOUNDED-MLP-20261027)

The module attribute `tessera.native.quantized_m_bound = R : i64` asks for
the seven-member MLP at capacity `R` rows. It is admitted only for the
complete MLP; a bound on any smaller quantized program is refused instead of
being dropped. The original activation must be a static rank-2 argument with
`m <= R` rows.

**Capacity retyping.** Before the ordinary static admission, the compiler
retypes the activation's forward slice to capacity. Every leading dimension in
the slice must be a multiple of `m` and scales by `R/m`. That covers flattened
K16 blocks (`m*K/16`), the reshape/expand `shape` and slice `slice_sizes`
attributes, and the row-shaped splat constants. The retyper admits only
row-local operations: reshape, cast, absolute, the axis-1 reduce, div,
maximum, expand, column slices starting at row 0, scaled_matmul and
silu_mul. Anything else is refused before any type changes, as is a constant
shared outside the slice. The retyped program is then certified by the
unchanged v3/v2 gate and down validators. The program at capacity, restricted
to its first `m` rows, is therefore the program at `m` rows.

**Schema v5.** v5 is v4 at capacity plus:

- `active_m` and `m_bound`;
- `original_graph_ir`, the request that replays to the identical plan;
- `row_frame`, with `slots`, `row_strides_bytes` and `row_steps`.

v4 is never repurposed. The row slots are:

- the activation;
- first Q/SA (7/8);
- the BF16 parent (11);
- the activation (12);
- second Q/SA (13/14);
- the output (17).

The row steps are 0, 2, 3, 4 and 6. Both weight expansions (steps 1 and 5) and
their B/SB planes stay invariant. Python re-derives the frame from the buffers
and refuses any difference. Bounded components compile arbitrary-M product
images, and the static checked Target remains the witness, checked as its
shape-free projection. Its portable record is the v5 package (see
"Portable bounded record" below).

**Owner frame.** `tessera_rocm_program_prepare_multi_output_row_frame` takes
capacity values. Each row step scales one scalar to `active_m * per_row` and
one grid axis to `ceil(active_m * grid_per_row / tile)`:

| Member | Scaled scalar | Grid axis |
|---|---|---|
| Quantizer | `Groups` (flat over 256 lanes) | x |
| Scaled product | `M` | y, over macro-tile rows |
| SiLU-mul | element count | x |

The owner never trusts the step list. A step is a row step exactly when it
reads or writes a row slot, and a row step publishes only row storage. Any
other frame is refused before HIP is touched.

A row change is atomic:

- the update must include every row-dependent root;
- it invalidates every private row slot and its dependents;
- it leaves the expanded weights resident.

The original single-consumer `TesseraRocmProgramRows` ABI is one entry of the
same internal frame.

**Evidence on RX9070XT/gfx1201 (Tajasarus, assertions-ON LLVM/MLIR 23.1.1):**

- At three shapes and several active M per capacity, bounded outputs are
  bitwise equal to the static capacity program's first rows. They also pass
  the independent staged chain at the active shape.
- Activation-only updates run the two quantizers, both products and SiLU-mul
  without re-expanding weights.
- Seven tampered frames are refused.

In paired interleaved HIP-event windows at H/K/N = 128/12288/256 with capacity
256, the bounded owner runs at 0.997–1.032× a static owner compiled at each
M, and is bitwise equal to it. An activation update costs 0.19–0.49 ms of host
time, against about 0.9 s to compile each static M. These timings are
diagnostic. Not covered: dynamic K/N and sibling backends. Public JIT
admission, owner retention and the portable record landed separately; see the
following sections.
Evidence: benchmarks/baselines/gfx1201_bounded_quantized_mlp_20261027/README.md.

### Public row-bound compile (GFX1201-PUBLIC-QUANTIZED-ROW-BOUND-20261027)

For the authored seven-member function decorated with
`@jit(target="rocm_gfx1201")`, the public request is
`function.compile_native_quantized_program(*example_inputs, m_bound=B)`. It
mirrors `compile_native_nvfp4_program(..., m_bound=B)`:

- It traces the example call once, sets
  `tessera.native.quantized_m_bound = B : i64` on that Graph and packages the
  v5 program. The frontend does not pre-check the bound; the native export
  owns admission. A bound below the example rows, or a bound on any smaller
  quantized program, comes back as the export's diagnostic.
- A malformed `m_bound` (not a positive `int`; `bool` is refused) is refused
  before tracing.
- The function retains the verified program. Every later ordinary call at
  `1 <= M <= B` binds it without retracing, compiling or running the authored
  Python body. Since GFX1201-PUBLIC-QUANTIZED-CALL-OVERHEAD-20261028 the
  function also retains one prepared capacity owner (see "Retained public
  owner" below); each call updates it, launches the seven members and reads
  `[M, N]`.
- N, K and every weight-root shape stay fixed; a different weight or
  activation width is refused by the existing expansion and storage checks.
  An M above B is refused by the owner's capacity check.
- The frontend record is `tessera.rocm.quantized_product_frontend.v5` and the
  compiler path is `canonical_rocm_quantized_bounded_mlp_program`.
  `compile_report()` names that path. `runtime_artifact()` and `manifest()`
  return the v5 portable record (see "Portable bounded record").
- Without `m_bound`, the call returns the static v4 program and changes no
  dispatch state; ordinary calls keep retracing per M as before.

**Evidence on RX9070XT/gfx1201 (Tajasarus, assertions-ON LLVM/MLIR 23.1.1):**
`tests/device/rocm/test_public_quantized_mlp_row_bound_device.py` covers
H/K/N = 32/48/19 (B = 8; M = 3, 8, 1, 5), 16/16/1 (B = 1) and 128/12288/256
(B = 256; M = 37, 256, 1). Replay runs with `subprocess.run`, the tracer and
the authored body all patched to fail. Every public output is bitwise equal
to the static capacity program's first rows over padded roots, and for
M <= 64 it also passes the independent staged chain at the active shape.

Not covered: dynamic K/N; gfx1151 and sibling backends. Owner retention,
public timing and the portable record landed separately; see the following
sections.

### Retained public owner (GFX1201-PUBLIC-QUANTIZED-CALL-OVERHEAD-20261028)

The explicit bounded compile creates one `RetainedBoundedOwner`
(`rocm_quantized_product_frontend.py`). It closes any owner a previous bounded
compile of the same function left, and a finalizer releases the owner's device
memory when the function is collected. The first public call prepares the
capacity owner and snapshots the bytes, dtype and shape of the six weight
roots. Later calls are serialized by a lock and take one of two paths:

- **Byte-identical weights.** Weight roots equal to the snapshot (equal bytes
  in a fresh array count) replace only the activation rows through
  `update_activations`; both expanded weights stay resident. The weights
  already passed the expansion checks when they were admitted, and the
  activation path keeps its own storage/shape check.
- **Any other weights.** A new or in-place mutated weight set takes the
  owner's full validated `update` of all seven roots, then the snapshot is
  replaced.

Failure policy:

- An admission refusal (`ValueError`, raised before any native write) keeps
  the owner and its snapshot. Examples are an M above capacity, a changed
  weight shape, or a changed activation width.
- A native failure (`RuntimeError` from the HIP status) closes the owner, and
  the next call prepares a fresh one.
- The owner never crosses fork: a child call refuses, and a child `close`
  never touches the parent's handle.

Separately, `graph_ir._loc_path` memoizes the source-location path resolution
every canonical render performs per op. It is keyed by the filename, plus the
working directory for a relative name. This cuts the canonical cache-key
render the static public quantized call does on every call.

**Evidence on RX9070XT/gfx1201 (Tajasarus):**

- **Device test.** `test_public_bound_retains_one_owner_across_calls` makes
  one prepare across these calls:
  - M = 3, 8, 1, 5 on held weights;
  - a capacity refusal;
  - a new weight set;
  - an in-place weight mutation;
  - a return to held weights.

  Each output is bitwise equal to the static capacity rows of the roots it
  was given.
- **Timing.** The #1021 recorder, re-run, puts the public bounded call at
  0.67–1.03 ms against 3.46–3.91 ms before, about 0.13 ms above a held-owner
  control that did not move. The public static call went from 4.75–6.22 ms to
  3.88–4.32 ms. These are diagnostic WSL2 host-wall figures.

Evidence: benchmarks/baselines/gfx1201_public_quantized_call_overhead_20261028/README.md.

Not covered:

- retaining a static program's owner (each static public call still prepares
  one, now its largest cost);
- the remaining per-call canonical render that keys the static program cache;
- dynamic K/N;
- gfx1151 and sibling backends.

### Portable bounded record (GFX1201-PORTABLE-BOUNDED-QUANTIZED-RECORD-20261028)

**Native package.** A bounded v5 program serializes as
`tessera.rocm.quantized_product_package.v5`. Its fields are the same as the
static v4 package: `program_json` (here the v5 plan with `active_m`,
`m_bound`, `row_frame` and `original_graph_ir`), `graph_ir`, `gate`, `down`
and `binding_digest`. The frontend record is
`tessera.rocm.quantized_product_frontend.v5`, and `runtime_artifact()` names
compiler path `canonical_rocm_quantized_bounded_mlp_program`, so the runtime
launches it through the existing `rocm_quantized_product_program` executor.

**Component claim.** Each component record keeps its v3/v2 schema and adds
`"arbitrary_m": true` when its product image is shape-free. The claim proves
nothing by itself:

- **Admission:** it is admitted only inside a v5 MLP record. A standalone
  component record carrying it, a static v4 record carrying it, and any value
  other than `true` are refused ("arbitrary-M claim differs").
  `to_manifest()` on a bounded component alone is refused ("no standalone
  portable record").
- **Digest:** the component's binding digest covers the flag, so dropping it
  is refused ("member binding differs").
- **Image:** even with a recomputed digest, a static claim on a shape-free
  image is refused ("checked Target differs"), because a static component's
  image must be its checked static Target.

**Replay.** Replay binds the package schema to the plan schema in both
directions (v4 ↔ static plan, v5 ↔ bounded plan). It then runs the program's
`validate()`, which:

- re-derives the row frame from the buffers;
- requires both products' arbitrary-M policy to match the plan;
- re-derives each shape-free product image from the recorded checked static
  Target through `tessera-rocm-project-kernel-identity`.

A restated `m_bound` or `active_m` is refused by the row-frame re-derivation.

**Compiler requirement.** The image re-derivation runs the matching
`tessera-opt`. A bounded record therefore replays only where that compiler is
present; a compiler-free process refuses ("tessera-opt is required for ROCm
kernel identity") instead of trusting the recorded projection. The static v4
record has no such requirement, because a static image is checked against its
Target by comparison alone.

**Evidence on RX9070XT/gfx1201 (Tajasarus, assertions-ON LLVM/MLIR 23.1.1):**

- **Fresh-process replay.**
  `test_bounded_artifact_replays_in_a_fresh_process` serializes the public
  bounded artifact, launches it through `rt.launch` in a new process at
  M = 3, 8, 1 and 5, and gets outputs bitwise equal to the public calls.
- **Compiler-free refusal.** The same replay with `TESSERA_OPT` and `PATH`
  pointed at nothing refuses at every M and writes no output.
- **Package round trip.** The capacity package round-trips with an
  unchanged binding digest at all three shapes.
- **Tamper tests.** `tests/unit/test_public_quantized_mlp_row_bound.py`
  covers the round trip, nine tampered records and the static/standalone
  refusals: 22 passed on gfx1201.

### Bounded-contraction v6 MLP (GFX1201-BOUNDED-QUANTIZED-K-20261028)

**Request.** `tessera.native.quantized_k_bound = KB : i64` asks for the
seven-member MLP with the gate contraction at capacity `KB`. It refines the
row frame and never travels alone: the export refuses it without
`tessera.native.quantized_m_bound` ("bounded K requires an M bound"), and on
any smaller quantized program. The traced activation's `K` and `KB` must both
be multiples of 16 with `K <= KB` ("exceeds its checked K16 capacity").
Publicly, `function.compile_native_quantized_program(*example_inputs,
m_bound=MB, k_bound=KB)` sets both attributes. A `k_bound` alone keeps the
traced M as the row capacity, and a malformed `k_bound` is refused before
tracing.

**Contraction retyping.** After the row retype, the compiler retypes the gate
contraction axis to capacity. K reaches exactly:

- the activation `[M, K]`;
- the gate quantizer: flattened K16 blocks `[G, 16]` / `[G, 1]` with
  `G = M*K/16`, and its `[M, K]` codes and `[M, K/16]` scales;
- the gate weights' packed `[N, K/2]` and scale `[N, K/16]` planes;
- the gate expansion's `[N, K]` values and `[K/16, N]` scale plane.

Each such extent scales by `KB/K` exactly, and the reshape/expand `shape`
attributes follow their result types. Any other shape, operation or shared
constant refuses. The gate product contracts K away, so nothing after it
changes. The capacity program is then certified by the unchanged v3/v2
validators.

**Schema v6.** v6 is v5 at capacity K plus:

- `active_k` and `k_bound`;
- `k_frame`, with `slots` (activation, both gate weight planes, Q/SA 7/8,
  expanded B/SB 9/10) and `k_steps` (0, 1, 2).

Python re-derives the contraction frame from the buffers and refuses any
difference ("contraction slot capacity differs", "bounded K differs",
"contraction frame differs").

**Shape-free members.** Under v6 the gate component carries `runtime_k`:

- **Gate expansion.** It compiles the runtime-K image
  `tessera.rocm.nvfp4_expand.five_memrefs.runtime_k.v1`: the kernel-identity
  projection (`family=nvfp4_expand runtime-k=true`) of the checked static
  expansion Target. The projection records `k = 0` and the `runtime_k` unit
  attribute; the verifier refuses a runtime-K directive that still records
  a K. The kernel takes the K16 group count as one i64 scalar on a
  `(groups/256, N)` grid. Grid y is the row, so no lane divides by the
  runtime extent, and every address equals the static kernel's at that K.
  Validation re-derives the checked Target from the Tile through the
  matching `tessera-opt`, and the image must equal its projection.
- **Gate product.** It must be the global-staged shape-free image, which
  reads M, N and K at launch. An LDS-staged capacity schedule refuses rather
  than assuming a runtime-K body.

The down component has no `runtime_k`: its contraction is H.

**Owner shape frame.** `tessera_rocm_program_prepare_multi_output_shape_frame`
generalizes the row frame to two variables, M rows and G = K/16 groups:

- Every frame byte span, scaled scalar and scaled grid field is
  `coef * M^a * G^b` with `a, b` in {0, 1}. Each value is checked equal to
  that monomial at capacity before HIP is touched.
- The frame closure is derived from SSA slot use exactly as for the row
  frame. A step that touches a frame slot must carry an entry and publish
  only frame storage, and no two entries scale the same scalar or grid axis.
- Both row ABIs are now the G-free case of this one internal frame.
- `tessera_rocm_program_update_shape(handle, M, G, ...)` is atomic. A change
  of either variable must include every original slot depending on it, and
  invalidates every private slot depending on it.

The MLP entries are:

| Member | Scaled scalar | Grid axis |
|---|---|---|
| Gate quantizer | `Groups = M*G` | x, over 256 lanes |
| Gate runtime-K expansion | `groups = G` | x, over 256 lanes (y = N rows) |
| Gate product | `M`; `K = 16*G` | y, over macro-tile rows (for M) |
| SiLU-mul, down quantizer, down product | row entries as in v5 | as in v5 |

**Updates.** `update()` takes all seven roots at any `(M, K)` inside the
bounds. `update_activations()` keeps the resident expanded weights, so it
refuses an activation whose K differs from theirs before any native write.
The public retained owner keeps its v5 policy: a K change has new weight
shapes, so it is a full update.

**Portable record.** A v6 program serializes as
`tessera.rocm.quantized_product_package.v6`, with frontend record v6 and
compiler path `canonical_rocm_quantized_k_bounded_mlp_program`. The gate
component adds `"runtime_k": true` beside `"arbitrary_m": true`, under the
same rules: admitted only with `arbitrary_m`, covered by the binding digest,
and never trusted. Replay re-derives the runtime-K expansion and both product
projections, so it requires the matching `tessera-opt`.

**Evidence on RX9070XT/gfx1201 (Tajasarus, assertions-ON LLVM/MLIR 23.1.1):**

- **Public replay.** `tests/device/rocm/test_public_quantized_mlp_k_bound_device.py`
  runs one compile per shape: H/N = 32/19 with bounds (8, 96), and H/N =
  128/256 with bounds (64, 12288). Public calls across six and four (M, K)
  pairs, from K = 16 to the capacity, are bitwise equal to the static v4
  program compiled at each exact (M, K). The small shape also matches the
  independent staged chain. Every K change reuses the one retained owner
  with no prepare, retrace or compile, and refusals keep it.
- **Activation updates.** An activation-only update at the resident K is
  bitwise equal; a K mismatch refuses and leaves the owner usable.
- **Native refusals.** HIP refuses a K change missing a weight slot, G
  beyond capacity, mismatched spans, and four tampered frames (unit bytes, a
  K coefficient, a missing expansion entry, a capacity grid).
- **Fresh-process replay.** The serialized artifact launches at four (M, K)
  in a new process, bitwise equal to the public calls; a compiler-free
  process refuses.
- **Unit tests.** `tests/unit/test_public_quantized_mlp_k_bound.py`: 25
  passed.
- **Device time (diagnostic).** In paired interleaved HIP-event windows at
  H/N = 128/256 with bounds (256, 12288), the K-bounded owner runs at
  1.004–1.032× the static program compiled at each of five (M, K), bitwise
  equal to it. Per-member windows put the cost in the shape-free gate
  product (about +4%, same physical route as the static image); the runtime-K
  expansion is within 1–3%. One bounded compile replaces a 0.6–1.0 s compile
  per (M, K). Evidence:
  `benchmarks/baselines/gfx1201_k_bounded_quantized_mlp_20261028/README.md`.

Not covered: dynamic H, an LDS-staged capacity gate product, AD, model
quality and gfx1151. Dynamic N landed next (§"Bounded-column v7 MLP").

### Bounded-column v7 MLP (GFX1201-BOUNDED-QUANTIZED-N-20261029)

**Request.** `tessera.native.quantized_n_bound = NB : i64` asks for the
seven-member MLP with the down projection's output columns at capacity `NB`.
Like the K bound, it refines the row frame: the export refuses it without
`tessera.native.quantized_m_bound` ("bounded N requires an M bound") and on
any smaller program, and the traced N must satisfy `N <= NB` ("exceeds its
checked column capacity"). It combines with a K bound in one program.
Publicly, `compile_native_quantized_program(..., n_bound=NB)` sets it; an
`n_bound` without `m_bound` keeps the traced M as the row capacity.

**Column retyping.** N reaches only the down tail, and only a
single-projection down expansion (row boundaries `[0, N]`) is admitted.
The retype scales to capacity:

- the down weights' packed `[N, H/2]` and scale `[N, H/16]` planes;
- the expansion's row boundaries `[0, N]`;
- its `[N, H]` values and `[H/16, N]` scale plane;
- the down product's `[M, N]` result and the casts after it.

Anything else refuses. The unchanged v2 validator then certifies the
capacity down projection.

**Schema v7.** v7 is v5 at capacity N plus:

- `active_n` and `n_bound`;
- `n_frame`, with `slots` (both down weight planes, expanded B/SB 15/16 and
  the output 17) and `n_steps` (5, 6).

A K bound adds the v6 contraction keys to the same plan. Python re-derives
both frames from the buffers ("column slot capacity differs", "bounded N
differs", "column frame differs").

**Shape-free members.** Under v7 the down component carries `runtime_n`.

- **Down expansion.** It compiles
  `tessera.rocm.nvfp4_expand.five_memrefs.runtime_n.v1`, the identity
  projection with `family=nvfp4_expand runtime-n=true`. The projection
  records `n = 0` and keeps the checked `[0, N]` boundary. The kernel takes
  N as one i64 scalar on the flat `N*G/256` grid; K stays static, so the
  row/group split divides by a constant.
- **Exclusivity.** runtime-K and runtime-N are exclusive per image. The
  verifier refuses a runtime-N directive that still records an N, or one
  with more than one projection.
- **Down product.** It must be the global-staged shape-free image, as for
  the gate product under K.

**Owner axis frame.** `tessera_rocm_program_prepare_multi_output_axis_frame`
adds N output columns (power bit 2) to the shape frame:

- A variable with capacity 0 is unframed. Its active extent must be 1, and
  any power on it refuses. The two-variable shape frame is now the N-free
  axis frame.
- `tessera_rocm_program_update_axes(handle, M, G, N, ...)` is atomic. An N
  change must include both down weight planes and invalidates every private
  N slot. A zero extent keeps that variable.
- `tessera_rocm_program_axes_info` reports all three extents.

| Member | Scaled scalar | Grid axis |
|---|---|---|
| Down runtime-N expansion | `n = N` | x, over `N * H/16` lanes |
| Down product | `N` (index 1) | x, over macro-tile columns |

The row and contraction entries are unchanged.

**Portable record.** A v7 program serializes as
`tessera.rocm.quantized_product_package.v7`, with frontend record v7 and
compiler path `canonical_rocm_quantized_n_bounded_mlp_program`. The down
component adds `"runtime_n": true` under the same rules as `runtime_k`.

**Evidence on RX9070XT/gfx1201 (Tajasarus, assertions-ON LLVM/MLIR 23.1.1):**

- **Public replay.** `tests/device/rocm/test_public_quantized_mlp_n_bound_device.py`
  runs three bound sets: N only at H = 32, K and N at H = 32, and K and N at
  H = 128 with bounds (64, 12288, 512). Public calls across fourteen
  (M, K, N) shapes, from N = 1 to capacity, are bitwise equal to the static
  v4 program compiled at each exact shape. The H = 32 shapes also match the
  independent staged chain.
  - Every shape reuses the one retained owner, and refusals keep it.
  - HIP refuses an N change missing a weight plane, N beyond capacity, a G
    on an unframed K, and three tampered frames.
  - The serialized K+N artifact replays bitwise in a fresh process; a
    compiler-free process refuses.
- **Unit tests.** `tests/unit/test_public_quantized_mlp_n_bound.py`: 26
  passed on gfx1201.
- **Device time (diagnostic).** At H = 128 with bounds (256, 12288, 512), the
  bounded owner runs at 1.020–1.033× the static program at each of five
  shapes, bitwise equal to it. The cost is the shape-free gate product
  (about +3%); the runtime-N members are within noise. Evidence:
  `benchmarks/baselines/gfx1201_n_bounded_quantized_mlp_20261029/README.md`.

Not covered: dynamic H (landed next, §"Bounded-hidden v8 MLP"), a
multi-projection N, an LDS-staged capacity product, AD, model quality and
gfx1151.

### Bounded-hidden v8 MLP (GFX1201-BOUNDED-QUANTIZED-H-20261029)

**Request.** `tessera.native.quantized_h_bound = HB : i64` asks for the
seven-member MLP with the hidden width H at capacity `HB`, a multiple of 16.
Like the K and N bounds, it refines the row frame: the export refuses it
without `tessera.native.quantized_m_bound` ("bounded H requires an M bound")
and on any smaller program. The traced H, read from the gate/up projection
boundaries `[0, H, 2H]`, must be a multiple of 16 with `H <= HB` ("exceeds
its checked H16 capacity"). It does not combine with a K or N bound ("does
not compose with a K or N bound"): the gate/up expansion would then need
runtime K and runtime N in one image. Publicly,
`compile_native_quantized_program(..., h_bound=HB)` sets it; an `h_bound`
without `m_bound` keeps the traced M as the row capacity, and `h_bound`
with `k_bound` or `n_bound` is refused before tracing.

**Hidden retyping.** H reaches every member except the gate quantizer. The
retype scales to capacity, by `HB/H` exactly:

- the gate/up weights' packed `[2H, K/2]` and scale `[2H, K/16]` planes, the
  equally spaced boundaries `[0, H, 2H]`, and the expansion's `[2H, K]`
  values and `[K/16, 2H]` scale plane;
- the gate product's `[M, 2H]` result and its BF16 parent cast;
- both `[M, H]` halves (the second's origin `H` too), their casts, the
  `silu_mul` and its BF16 cast;
- the whole down quantizer: the flattened `[G, 16]`/`[G, 1]` K16 blocks with
  `G = M*H/16` (including both splat constants), the `[M, H]` codes and the
  `[M, H/16]` scales;
- the down weights' packed `[N, H/2]` and scale `[N, H/16]` planes, and the
  down expansion's `[N, H]` values and `[H/16, N]` scale plane.

The down product contracts H away; its `[M, N]` result does not change.
Anything else refuses, and the unchanged v3/v2 validators then certify the
capacity program.

**Schema v8.** v8 is v5 at capacity H plus:

- `active_h` and `h_bound`;
- `h_frame`, with `slots` (both gate/up and both down weight planes, the
  expanded B/SB 9/10 and 15/16, the parent 11, the SiLU output 12 and the
  down quantizer pair 13/14) and `h_steps` (1–6).

Python re-derives the frame from the buffers ("hidden slot capacity
differs", "bounded H differs", "hidden frame differs").

**Shape-free members.** Under v8 the gate component carries `runtime_n`
(its N is 2H) and the down component carries `runtime_k` (its K is H).

- **Gate/up expansion.** It compiles the runtime-N image
  `tessera.rocm.nvfp4_expand.five_memrefs.runtime_n.v1`, now admitted for
  any number P of **equally spaced** projections. The verifier and the
  identity projection require the checked boundaries to be `p * N/P`; the
  kernel then derives each runtime boundary as `p * (N/P)` from its one N
  scalar. A single projection (P = 1) is unchanged. Unequal boundaries
  refuse ("records N as zero over equally spaced projections").
- **SiLU parent view.** A runtime-N gate/up product's activation member is
  the plan's Graph with the module marker
  `tessera.native.gate_up_runtime_width`. The native Schedule then emits the
  BF16 body with its output width as a fifth i64 argument: the row split,
  the parent width `2W` and the second half's origin all read it. The Tile
  and Target carriers record the declared `runtime_width` UnitAttr (schema 3
  only), and `columns` stays the checked capacity, which the body never
  reads. The materializer guards `0 < W <= columns` and `count % W == 0`.
  The member exports profile `gfx1201_parent_view_bf16_f32_runtime_width_v1`
  with ABI `tessera.rocm.parent_view.parent_o_n_width.bf16_f32.v1` and
  scalars `[M*W, W]` at capacity. It launches only inside its bounded MLP
  owner; the standalone parent-view submit refuses it. The product validator
  binds the profile to the component's `runtime_n` claim.
- **Down expansion.** The existing runtime-K image, with `groups = H/16`.
- **Both products.** Each must be the global-staged shape-free image.

**Admitted-rows panel (every bounded program).** A shape-free
(arbitrary-M) product image runs every M down to 1, so its W8A8 register
panel is chosen at that end of the range, not at capacity. The product
member Graph carries `tessera.rocm.blockscale_min_m = 1 : i64`; the native
selector (`selectFp8W8A8BlockScalePanel`) and its Python oracle
(`blockscale_panel_oracle(..., min_m=)`) then evaluate the 32-row
register-panel rule at the admitted minimum. The selector refuses a value
outside `[1, M]` or not i64 (`ROCM_FP8_BLOCKSCALE_CONTRACT`). The LDS rules
are unchanged, and so is every static program. The product validator
expects the request on exactly the shape-free members, so the v5, v6 and v7
bounded programs carry it too. Their panels change only where the capacity
reaches the 32-row band (M and N multiples of 32 with at least 256 32×32
tiles). A bounded record serialized before this change no longer replays;
recompile it.

Measured on gfx1201: the 32-row panel chosen at capacity 256 × 1024 cost
4.2–4.6× the 16-row panel at M = 16. Across the 32-row band, 16 rows
measured 0.78–1.33× of 32 for static programs, so a capacity in that band
can cost up to about 1.33× at its large end.

**Global staging for runtime-shape products
(GFX1201-BOUNDED-GLOBAL-PANEL-20261030).** A runtime-K/N/H product exists
only as the global-staged shape-free image, so a bounded program whose
capacity reached an LDS gate panel used to refuse ("requires the shape-free
global-staged product"). Its product member now also carries
`tessera.rocm.blockscale_staging = "global"`. The selector then skips every
LDS rule and takes the register panel at the admitted minimum, and the
oracle (`staging="global"`) mirrors it. Any other value is refused. M-only
(v5) bounded products keep their LDS panels. Measured at H-bound capacity
(256, 2048), K/N 4096/256: 1.0–1.12× static at four shapes, but 2.75× at
256 × 2048, where the `[16, 32]` gate product takes 4.2× static's LDS
`[128, 128]`. Evidence:
`benchmarks/baselines/gfx1201_bounded_global_panel_20261030/README.md`.
Per-extent product images (below) replace that trade.

**Per-extent product images (GFX1201-BOUNDED-PER-M-PANEL-20261030).** One
image cannot serve a wide admitted range: an LDS capacity panel cost 2.1–2.9×
at small M, and the small-M register panel cost about 4× at capacity. Every
bounded (shape-free) product now carries a base image plus one LDS
**variant** per box where the static rule takes an LDS panel:

- **The base image** is the register panel for M >= 1 (the global shape-free
  image for a runtime-K/N/H product).
- **A box** is an M range and, when the product varies an extent, a range of
  that extent in frame units. The gate's K varies as G = K/16 and its N = 2H
  as H/16; the down projection's K varies as H/16 and its N as N. Boxes are
  derived by scanning the panel oracle over (M, that extent) and keeping the
  maximal rectangles of one LDS panel. At most eight are allowed.
  Register panels are not split.
- **Each variant** is compiled with its panel evaluated at the box's corner:
  `tessera.rocm.blockscale_min_m` plus `tessera.rocm.blockscale_at_n` or
  `_at_k` (i64 within the extent; K a whole scale group). With them set, the
  native selector evaluates every panel rule at that point, and the oracle
  mirrors it; the selector remains the authority.
- **A variant whose N varies at launch** is projected with the new identity
  option `arbitrary-n` (LDS family only). It records `whole_n = false`, so
  the LDS body masks its column edge. Its provenance records
  `"arbitrary_n": true` only when set, so existing descriptors keep their
  digests. LDS images already take K at launch (`lds_runtime_k`).
- **Validation.** The product validator re-derives the boxes, each variant's
  request Graph, its panel and its arbitrary-N claim. The portable record
  carries each variant's box, checked static Target and stage. Replay
  re-derives every image.

**Owner.** `tessera_rocm_program_prepare_multi_output_axis_frame_variants`
takes the axis frame plus `TesseraRocmProgramStepVariant` rows, each with:

- `step`, `[min_rows, max_rows]`, `axis` (1 G, 2 N, 3 H, 4 none) and
  `[min_axis, max_axis]`;
- the image and entry, and its capacity geometry;
- its frame entries: the step's base entries in order on (scalar, coef,
  powers), with only the grid fields changed.

Admission checks each variant against the frame and requires one shared axis
per step and non-overlapping boxes. The variants key the owner cache.
`applyFrame` selects each stage's image for the active extents, and
`tessera_rocm_program_variant_info` reports the selection. An M-only bounded
program with variants takes the axis frame, as its N-, G- and H-free case.

An LDS variant takes K at launch (`lds_runtime_k`) only where its product's
K varies (the K-bounded gate, the H-bounded down projection). Its validator
refuses any other runtime-K claim, because the runtime-K loop measured about
3% where K is fixed.

Measured on gfx1201 (diagnostic HIP events, behind a kept warm-up owner):
- **M-only program** (capacity 256, H 2048): 0.96–1.01× static from M = 16
  to 256.
- **H-bound program** (capacity (256, 2048)): 0.998–1.08× across six shapes,
  against 2.75× at its corner with the global base alone.

All outputs are bitwise equal. Reading M and N at launch and masking them
cost nothing measurable.

A first run without a warm-up owner showed 10–15% at the capacity corner.
That was mostly an artifact: the first owner a process prepares ran about
17% slower for its whole life, and that owner was the bounded one. Evidence:
`benchmarks/baselines/gfx1201_bounded_per_m_panel_20261030/README.md`.

**Owner axis frame.** The axis frame gains a fourth variable: H16 groups
(power bit 3, `capacity_h`). `prepare_multi_output_axis_frame`,
`update_axes` and `axes_info` take and report four extents (M, G, N, H);
an unframed variable still has capacity 0 and refuses any power on it. An
H change must include all four H-dependent weight planes and invalidates
every private H slot.

| Member | Scaled scalar | Grid axis |
|---|---|---|
| Gate/up runtime-N expansion | `n = 2H` | x, over `2H * K/16` lanes |
| Gate product | `N = 2H` (index 1) | x, over macro-tile columns |
| Runtime-width SiLU | `M*H` and `W = H` | x, over `M*H` lanes |
| Down quantizer | `M*H/16` groups | x |
| Down runtime-K expansion | `groups = H/16` | x |
| Down product | `K = H` (index 2) | none |

The gate quantizer's and both products' row entries are unchanged.

**Portable record.** A v8 program serializes as
`tessera.rocm.quantized_product_package.v8`, with frontend record v8 and
compiler path `canonical_rocm_quantized_h_bounded_mlp_program`. The gate
component carries `"runtime_n": true` and the down component
`"runtime_k": true`, under the existing claim rules.

**Evidence on RX9070XT/gfx1201 (Tajasarus, assertions-ON LLVM/MLIR 23.1.1):**

- **Public replay.** `tests/device/rocm/test_public_quantized_mlp_h_bound_device.py`
  runs two bound sets: H/K/N 32/48/19 with bounds (8, 64), and 128/4096/256
  with bounds (64, 512). Public calls across nine (M, H) shapes, from H = 16
  to capacity, are bitwise equal to the static v4 program compiled at each
  exact shape. The small shapes also match the independent staged chain.
  - Every shape reuses the one retained owner, and refusals keep it.
  - An activation-only update keeps the resident hidden width.
  - HIP refuses an H change missing weight planes, H beyond capacity, an N
    on an unframed axis, and three tampered frames.
  - The serialized artifact replays bitwise in a fresh process.
- **Unit tests.** `tests/unit/test_public_quantized_mlp_h_bound.py`: 36
  passed on gfx1201.
- **Device time (diagnostic).** At K/N 4096/256 with bounds (256, 512), the
  bounded owner runs at 0.98–1.05× static at five shapes, bitwise equal at
  all five. Before the admitted-rows panel it ran at 1.2–3.9× at four of
  them: the capacity schedule had picked a `[32, 32]` gate-product panel
  where static small-M programs pick `[16, 32]`. The H members themselves add
  at most about 1 µs. Evidence:
  `benchmarks/baselines/gfx1201_h_bounded_quantized_mlp_20261029/README.md`.

Not covered: H combined with a K or N bound, selecting among measured
product images per active shape, an LDS-staged capacity product, AD, model
quality and gfx1151.


### gfx1151 (GFX1151-QUANTIZED-MLP-20261030)

**The FP8 program is excluded on gfx1151 by ISA.** RDNA 3.5 has WMMA for
F16, BF16, IU8 and IU4 but no FP8/BF8 form:
`docs/reference/isa/rdna/cross_version/instruction_matrix.md` lists
`V_WMMA_F32_16X16X16_FP8_FP8` and its siblings for RDNA4 only. Both scaled
products of this MLP are FP8 × FP8 WMMA, so no member set above runs there.
Every entry point refuses a non-gfx1201 target or live device by name:
`compile_native_quantized_program` ("requires primal exact gfx1201") and the
prepared owners ("requires owning gfx1201"). No gfx1201 result in this
section transfers to gfx1151, and the generic `rocm` name inherits nothing.

A gfx1151 program would keep the FP8 E4M3 storage contract and decode each
operand exactly to F16 for `V_WMMA_F32_16X16X16_F16` with fp32 accumulation.
That is a separate lowering with its own tolerance-bounded evidence: gfx1151
WMMA rounding already differs from gfx1201 at about 1e-6 (see
`tests/device/rocm/test_bounded_fragment_pack.py`), so bitwise parity with
gfx1201 is not expected. It is open work, not a claim here.

**Slice 1: software E4M3 encode and the scalar members.** RDNA 3.5 also
has no FP8 convert (`V_CVT_PK_FP8_F32` is RDNA4-only). The changes:

- **The encode.** `lower-tessera-target-to-rocdl` lowers a gfx1151 `f32 ->
  f8E4M3FN` cast to an integer-only encode. It has the instruction's
  recorded semantics:
  - round to nearest even;
  - any NaN encodes as `0xFF`;
  - `|x| > 464` and ±inf encode as `sign|0x7F` (no saturation);
  - subnormals land on the 2⁻⁹ grid;
  - f32 denormals become signed zero.

  `e4m3-encode=software` forces it on gfx1201. Every other architecture
  keeps its cast, so lowering is never an admission.
- **Bit-identity.** On gfx1201 hardware, the software image and the
  instruction agree on all 2³² f32 bit patterns
  (`tests/device/rocm/test_e4m3_software_encode.py`).
- **Members admitted.** The K16 quantizer, the NVFP4 expansion and the SiLU
  parent view admit a matching gfx1151 target/arch pair at every compiler
  gate: Graph contract, Schedule identity, Tile and Target carriers,
  verifiers, exported profiles and the TileToROCM row/expansion gates.
  NVFP4 requantization stays gfx1201. gfx1201 identities are unchanged, and
  the pipeline arch must equal the kernel's.
- **Compile-only on gfx1151.** All three members lower without an FP8
  instruction and serialize as gfx1151 HSACOs. No gfx1151 execution is
  claimed. Executable packaging stays gated by device-proven family
  promotion, and nothing submits or consumes a gfx1151 member image.
  Evidence: `tests/unit/test_gfx1151_quantized_scalar_members.py`.

**Slice 2: the decoded-F16 scaled product.** The K16 fp32-block-scaled
E4M3 × E4M3 product compiles for gfx1151 by decoding each fragment exactly
to f16 and running `V_WMMA_F32_16X16X16_F16` with fp32 scale-group
partials.

- **The request.** The WMMA generator names its fragments `e4m3_f16` (E4M3
  storage, F16 registers) only for a gfx1151 scaled E4M3 kernel. The
  fragment layout resolves `e4m3_f16` only on gfx1151, and a plain `e4m3`
  pair stays refused there.
- **The decode.** `materializeFragmentPack` decodes in integer arithmetic:
  a normal value is `s<<15 | (e+8)<<10 | m<<7`, subnormals are
  renormalized, and NaN becomes an f16 quiet NaN. The typed WMMA records
  `input_dtype = "f16"` and `decoded_from = "e4m3"`.
- **The schedule.** Schedule admits the gfx1151 pair only in the
  fp32-scaled form, on the one-wave register panel (there is no LDS body),
  with an explicit `wmma` descriptor. The store-fusion, Schedule-verifier
  and TileToROCM gates admit exactly that form, and the panel oracle
  mirrors it. E5M2, MXFP8 and unscaled FP8 stay refused.
- **Evidence (compile-only).** The gfx1151 HSACO's disassembly holds only
  `v_wmma_f32_16x16x16_f16`, with no FP8 or BF8 instruction
  (`tests/unit/test_gfx1151_decoded_fp8_product.py`). Every E4M3 value is
  exact in f16 and every product exact in f32, so only accumulation order
  can differ from gfx1201. No gfx1151 numerical result is claimed until a
  device run.

## Quantized package certificate and ABI reuse

`TESSERA_ROCM_PROGRAM_BINDING_CACHE=1` (the existing default) also enables
complete-content memoization for quantized v1/v2/v3/v4 package validation and
ABI construction. Each decorator retains at most sixteen entries, with a
4 MiB content-budget limit per entry. Keys recursively include every frozen
dataclass field, raw image bytes, recipe fields and mutable nested descriptor
metadata; bool and int are distinct. No certificate is keyed by an object id
or the declared binding digest alone.

Unsupported, cyclic or oversized snapshots take the original full validation
path. Content changing during certification is refused. Process changes
reset locks and cached host bindings before access. The cache holds host ABI
metadata and image bytes, with no native handles, streams or runtime tensor values.
Runtime input validation/copies, device identity, native readiness and actual
execution still run for each call. Portable restoration retains all existing
checks, and the zero setting supplies the uncached A/B control. No lowering,
image, descriptor, numerical policy or ABI changes follow from a cache hit.

The content budget charges UTF-8 string bytes, mapping keys and integer
magnitude as well as container/field overhead. Large metadata keys cannot
bypass the limit. Keyword `self` validation calls use the same complete
package key as ordinary bound-method calls. Cache inspection/reset also
checks the process before taking any inherited lock.

Floating metadata is keyed by its binary64 bits, preserving signed zero and
NaN bit patterns rather than Python numeric equality. A real native-package
regression verifies that a +0.0-to--0.0 metadata mutation cannot reuse the
earlier valid certificate or ABI binding.
