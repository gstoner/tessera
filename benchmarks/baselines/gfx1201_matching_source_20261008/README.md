# Matching-source gfx1201 compiler and runtime revalidation

Owner E2E-REAL-6 / FRONTEND-IR-MEDIUM-1 / AD-RESIDUAL-EVAL-1 /
ROCM-NVFP4-INGEST-1. Sync FIVE-SLICE-INTEGRATION-2026-10-08.

Tajasaurus: AMD Radeon RX 9070 XT. Fresh HIP query requires gfx1201;
receipt.json records the actual device UUID and four rebuilt tool/runtime hashes.

4,931 compiler/frontend/runtime/test inputs match the authoritative
pre-publication source snapshot. current_source_sha256.json seals those bytes.
Final publication cleanup only normalized trailing blank lines in 23 files;
this is snapshot evidence rather than a claim that source hashes match every
later documentation/formatting edit.

The core and ROCm tools use LLVM/MLIR 23.1.1 (optimized build). Native HIP
image-cache/program-ownership runtimes are rebuilt from matching source.
Missing libz3 visibility was repaired using the existing toolchain dependency
directory. Timestamp-preserving transfer initially reused newer Make objects
for five changed AD inputs; explicit timestamp refresh and recompilation
repaired missing pass options/native NVFP4 projection.

## Exact-device results

Initial log: 72 passed, 41 failed. Retained for provenance.
Corrected log: **113 passed**, no failures, in 95.72 seconds.

The lane covers composed mapped primal/JVP/VJP, scalar composed JVP/VJP,
public resident NVFP4 ingest, compiler-free replay, changed inputs and retained
outputs. Tests execute in host WSL with the matching tools and rebuilt HIP
runtimes. The host lacks pytest-timeout; numerical assertions execute but
timeout enforcement is not claimed.

This is owning gfx1201 regression evidence. No fresh performance comparison,
model-quality acceptance, sibling physical parity, generic AD/batching closure
or aggregate green result is inferred.
