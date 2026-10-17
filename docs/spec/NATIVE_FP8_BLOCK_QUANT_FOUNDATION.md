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
the diagnostic C++ multi-output owner. Public packaging/JIT and resident
composition remain required; diagnostic execution is not public closure.
The ordinary Python trace still needs constructor/shape-metadata integration;
the current verified input is textual Graph IR. Schedule profile schema 4
belongs to the row-region namespace; attention schema numbering is unchanged.

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
