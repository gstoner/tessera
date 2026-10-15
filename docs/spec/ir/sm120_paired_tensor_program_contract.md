---
last_updated: 2026-10-15
---

# SM120 static paired tensor program contract

Owner W1.1 / FRONTEND-IR-MEDIUM-1 / AD-FWD-CORE-1.
Synchronization key: W11-TENSOR-RHS-JVP-20261015.

## Semantic authority

`--tessera-autodiff-forward=export-scaled-program=true` derives the actual
paired JVP SSA before outlining. For the SM120 RHS-linear product, native
MLIR proves two same-shaped f32 matmul results use the same producer edge,
the original RHS and its requested tangent argument. The requested parent
role, original Graph witness and differentiated Graph witness are retained.
The exporter rejects active-source products rather than relabeling their
additional derivative arithmetic as a coefficient program. Those members
still require native target integration.

The existing native tensor exporter now emits
`tessera.native.sm120_tensor_program.v7` for this paired envelope. This is
independent of attention schema numbering; attention schema 5 remains
saved-LSE plus bounded sequences. Primal tensor v1-v6 semantics are unchanged.

## Manifest and lifetime

Three readonly roots precede topologically ordered SSA result buffers.
`role_indices` maps source/RHS/RHS-tangent to their original argument IDs;
the first two roles may be permuted. `steps` retain actual input/output IDs
and isolated native member Graphs. `outputs` lists primal then tangent;
`output` remains the primal display ID. Both outputs have returned ownership.
The shared producer edge's last read is the second consumer. Every private
buffer remains owned through synchronous completion; this first owner uses
disjoint storage instead of aliasing scratch based on an unproved lifetime.

Each projected member follows native Graph -> Schedule -> Tile -> NVIDIA
Target IR -> LLVM/NVVM/PTX. Physical RHS materialization is row-major; an
explicit conflicting storage request is rejected. Plain static f16/BF16
inputs and f32 matmul outputs define this checked envelope. Fused, dynamic,
active-source and encoded-product envelopes need their own extensions.

## Native owner

`TesseraNvidiaTensorProgramStep` carries member images, entries, actual SSA
IDs and row/matmul mapping tags. `tessera_nvidia_tensor_program_prepare`
seals the complete program in a distinct handle map; existing one-output
matmul or resident calls cannot reinterpret it. Native preparation checks
roles, shape/storage/bytes/strides, topological member IDs, entry/mapping
compatibility, and exact CUDA parameter count/width before publishing a handle.

`invoke` validates three host input views and two separate output views.
Inputs are snapshotted into pinned staging before uploads; the C++ owner
submits every member on its private stream. Both outputs are staged and
committed after successful completion. Output/input aliasing is rejected.
Context identity and process ownership are checked. Failure after submission
drains the stream before scratch can be reused. All byte arithmetic is checked.

The owner shares existing context scratch under the existing native mutex.
A retained frame has a generation/base lease; another owner's invocation
invalidates its profile lease. `profile` records complete and per-member
CUDA graph batches or explicitly labeled enqueue-event batches, with bounded
node/repetition counts and per-call graph lifetime. Host profile overhead,
completed frontend calls and compile/prepare time remain different metrics.

`close` retires the owned stream/modules; returned host arrays stay valid.
Portable v7 replay and borrowed resident input/stream contracts are not
introduced by this host-frame ABI.

## Portable prepared paired replay — 2026-10-15

The distinct tessera.nvidia.tensor_rhs_jvp.v1 container retains the native v7
plan, scheduled member metadata and image/descriptor artifacts in order.
PreparedTensorJVP.manifest() exports the trusted native package;
PreparedTensorJVP.from_manifest() validates and prepares it without a compiler.
No primal schema changes. This explicit prepared API is separate from generic
RuntimeArtifact dispatch. It requires the matching prebuilt native provider.

Validation recomputes root roles, SSA edges, both output identities, private
lifetimes and bytes, checks Graph/Schedule/Tile/Target/image provenance and
existing producer/consumer ABI shape/layout rules. Digests are integrity checks,
not executable authentication. A matching native prepare replaces the cached
lease; older handles must not be invoked or profiled afterward.

Exact-device proof and separate timing domains are recorded in
benchmarks/baselines/sm120_tensor_rhs_jvp_portable_20261015/README.md.
