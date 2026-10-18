---
last_updated: 2026-10-18
audit_role: reference
---

# ROCm native mixed scalar contraction

Owner: W1.1 / FRONTEND-IR-MEDIUM-1 / AD-RESIDUAL-EVAL-1 / E2E-REAL-6.
Synchronization key: ROCM-MIXED-SCALAR-CONTRACTION-20261018.

## Semantic and numerical contract

The named public package accepts one static rank-2 matmul on gfx1151 or
gfx1201 with exactly one FP32 operand and one FP16/BF16 operand. Each operand
has an independent physical element type. Physical row-major operands may be
transposed independently; the output is compact row-major FP32, FP16 or BF16.
Python decorator proof covers explicit FP32/FP16 output policies. Literal Graph
MLIR and checked package proof additionally cover BF16 output; the Python
matmul output-policy allowlist is unchanged.

The compiler widens each half-storage load to FP32, accumulates in increasing
K order with FP32 FMA, and narrows only the final output store. It does not
round the FP32 operand to a WMMA input type. This is a scalar VALU route;
neither architecture has an FP32-input WMMA form. Bias, residual, activation,
encoded storage and dynamic-shape packages require separate contracts.

The checked public envelope requires positive M/N/K, M <= 65535*16 and
N/K <= INT32_MAX. The native scalar Tile operation also represents FP32 by
FP32 contractions; this package's public admission remains the named mixed
pair. General standalone FP32 matmul is not promoted by this work.

## Native compiler boundaries

1. Typed semantic Graph preserves storage, result type and transpose semantics.
2. Native Graph-to-Schedule emits the architecture-specific
   rocm_gfx1151_scalar_fma_v1 or rocm_gfx1201_scalar_fma_v1 contract.
3. Native Schedule-to-Tile emits one registered tile.scalar_matmul_kernel
   with explicit A/B/output types, FP32 accumulator and transpose flags.
4. Tile-to-ROCm consumes its isolated six-argument LLVM wrapper into the
   registered tessera_rocm.scalar_matmul Target operation. It verifies
   architecture, Schedule digest, direct argument binding and 16x8 geometry.
5. The native physical-image projection validates the attribute-only Target
   recipe, removes its launch-owned Schedule digest and derives a stable
   kernel symbol. Architecture, types, orientations and module policy remain
   in image identity. Shapes and frontend function names do not.
6. The registered scalar_matmul executable family materializes the GPU body
   from the same ordered arithmetic emitter used by the NVIDIA scalar route,
   then lowers through ROCDL/LLVM and emits HSACO.

Python projects and checks ABI metadata; it emits no Tile arithmetic, GPU
kernel source, physical image identity or backend launch sequence.

## Checked runtime ABI

The new namespace is
tessera.rocm.scalar_matmul.a_b_o_m_n_k.mixed_fp32_fma.v1.
Other program schemas, including saved-LSE and bounded schemas, keep their
existing meanings.

The descriptor has two readonly rank-2 inputs, one distinct writable result,
six equality shape guards, and three int64 scalar roles M/N/K. Physical
orientation determines input extents. The launch uses grid
(ceil(N/8), ceil(M/16), 1) and workgroup (32,1,1), with no workspace or dynamic
local memory. The device kernel receives three flattened rank-1 memref
descriptors followed by M/N/K: eighteen physical arguments.

The existing native C++ program owner holds images, device buffers, host
snapshots and completion. A/B are original roots; output is first-written by
the sole kernel and remains readable through its generation. Updates refresh
roots; reads return independent host snapshots. The Python adapter marshals
this checked frame and performs host storage copies only.

The descriptor retains the native Schedule and physical Target recipes.
Types and orientation are checked against the Target; dimensions and storage
are checked against the Schedule before native storage acquisition.
Coordinated metadata edits cannot reinterpret the image's output width.
Descriptor replay also requires matching Graph, Schedule, Tile and Target
content identities and the exact live GPU architecture. Inputs and output must be
compact native-endian host ndarrays of the declared shape/type. Output aliasing
an input, nondefault external streams and malformed dimensions are rejected
without output commit. Device-resident producer edges are a separate contract.

## Evidence and timing boundaries

The owning-device suite covers both mixed operand orders, both half types,
all transpose combinations, all three output types, changed roots, serialized
compiler-free replay, output canaries and immutable snapshots. Additional
cases check image reuse across two shapes, FP32 cancellation and finite/
subnormal/nonfinite classification. This does not certify exhaustive rounding,
all shapes, solver derivatives or a composed ROCm source pullback.

The recorder is benchmarks/rocm/record_mixed_scalar_contraction.py. It checks
the live architecture and numerical output before and during timing, and
records source, compiler, provider and image identities. Native HIP-event
windows, prepared update/invoke/read and completed descriptor calls are
separate quantities. Event windows may contain submission gaps. Ordinary
descriptor receipts have no kernel timing; no host wall time is labeled as
kernel time for this ABI.

The original five compiler aggregate slices remain open. The remaining
integration obligation is a native composed AD producer/consumer program
with its own compiler-exported SSA ownership and exact-device proof.

The standalone scalar entry requires two distinct original Graph argument roots
and a direct returned matmul result. Producer or result-user operations require
the native composed SSA program contract; primitive extraction must not drop them.
