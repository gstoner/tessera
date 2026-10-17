---
audit_role: reference
last_updated: 2026-10-16
---

# Native parent-view candidate contract

Owner: ROCM-NVFP4-INGEST-1 / FRONTEND-IR-MEDIUM-1 / LAYOUT-ALG-1 / E2E-REAL-6.
Sync: ROCM-NATIVE-GATE-UP-CONSUMER-20261016.

This is a scoped implementation candidate, not general layout/JIT admission.
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
waits before close. Source upload is still host input in this diagnostic;
no resident matmul producer or arbitrary external-stream contract is proved.

The package must respect the native owner's aggregate allocation limit and
signed launch/byte widths. The full public family/plugin, source authority,
serialization/replay and JIT integration are not yet implemented. Before public
admission, align the explicit source envelope with the generic owner limit,
validate source/physical certificate ancestry, preserve exact storage rounding,
and supply owning hardware proof through the canonical pipeline.

Evidence: benchmarks/baselines/native_gate_up_view_20261016/README.md.
