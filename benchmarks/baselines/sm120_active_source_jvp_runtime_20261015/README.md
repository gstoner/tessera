---
last_updated: 2026-10-15
---

# Source-active SM120 JVP and portable replay

Owner: W1.1 / FRONTEND-IR-MEDIUM-1 / AD-FWD-CORE-1.
Synchronization key: W11-ACTIVE-SOURCE-20261015.

## Proved execution

A real Python frontend traces explicit-FP32 RMSNorm -> ReLU -> plain FP32-output
matmul. The native AD transform preserves its actual primal/tangent SSA through
Graph -> Schedule -> typed Tile buffer views -> NVIDIA Target -> LLVM/NVVM ->
PTX. A checked C++ owner stages three original-order roots and publishes two
independent outputs after all three native stages complete. No Python backend
math, Tile construction or per-member launch participates in execution.

The source-active portable schema is distinct from existing RHS-only native v7
and portable contracts. The native outline remains artifact_only. Captured
members bind all compiler boundaries, image digests and the 112-byte native step
ABI. Validation precedes provider access; replay requires no compiler. A live
matching JIT owner is borrowed and a standalone replay closes its own owner.
Digests detect inconsistent witnesses, not authenticity of third-party code.

## Architecture and source scope

Super-Bear WSL, NVIDIA GeForce RTX 5070, compute capability 12.0, driver 610.88.
LLVM/MLIR 23.1.1; CUDA 13.3. All three normal CMake compiler targets
(tessera-opt, tessera-nvidia-opt and tessera-rocm-opt) built successfully.
The CUDA owner library was built separately from the integrated runtime sources.

Latest normal-build benchmark compiler SHA256:
81a98c3b282d27138b7417b03097cac03fceb10813ce2156ddd2d5f305eb266f.
Native owner SHA256:
2f51cf156777a470b803fc9ed8c5f3eb3a9396d0802651d49c7ecc8ec99ee689.
normal-build-timings.json records raw executable hashes separately from the
dependency-aware compiler identity, plus the live GPU UUID and source hashes.
The saved FP16 fixture was recaptured with this compiler. See
docs/spec/ir/source_active_row_jvp.md for the exact envelope.

## Validation

Final focused gate: **754 passed in 224.84 seconds** on host WSL using the
normal-build compiler above. Owning documentation regeneration and Ruff on all
18 changed Python files passed. This is not a full-unit certificate.

- Latest normal-build durable repository tests: 78 passed. FP16/BF16, argument-role
  permutations, minimal/ragged rows, numerical public JVP and standalone replay,
  no-GPU artifact admission, fresh-process replay, new root values, saved-record
  isolation and malformed parent/member/ABI rejection.
- Existing NVIDIA RHS-JVP and prepared ABI regression: 234 passed; one known
  multithreaded fork DeprecationWarning.
- Isolated combined replay/public/native component gate: 320 passed.
- Fresh-process strengthening: four cases passed with PATH, TESSERA_OPT,
  TESSERA_NVIDIA_OPT and LLVM tool paths unavailable before process startup;
  compiler entrypoints, projection/lowering and image file reads are forbidden.
- Direct producer storage tests caught a BF16 negative-zero regression in the
  earlier compiler: two BF16 failures, with two FP16 cases passing. The corrected
  scalar lowering preserves BF16 negative zero and NaN selection; all four direct
  edge cases pass in the 78-test normal-build gate.
- Focused Ruff passed. Earlier normalization full-unit proof retains its
  original source scope and does not certify this new slice.

The checked-in compiler-produced fixture
`tests/fixtures/native_sm120/active_source_jvp_f16.json` enables artifact
validation on hosts without a GPU. Owning-device tests are in
`tests/unit/test_native_sm120_active_source_jvp_runtime.py`; shared frontend
precision tests are in `tests/unit/test_normalization_frontend_policy.py`.

## Matched host-overhead experiment

`normal-build-timings.json` records seven samples, five completed calls per host sample and
alternating mode order. All modes retain the same live native owner and image
identities. The full-certificate control clears only the pure validation cache;
it does not clear a CUDA module/allocation cache. Correctness is checked before
and after timing against an independent FP64 derivative/product oracle. Device
stage events are separately recorded. No competing GPU test/benchmark ran.

| Storage | MNK | Public JVP us | Cached portable replay us | Full certificate replay us |
| --- | --- | ---: | ---: | ---: |
| FP16 | 17,19,35 | 770.709 | 768.545 | 2789.661 |
| BF16 | 17,19,35 | 710.579 | 775.145 | 2778.782 |
| FP16 | 129,33,257 | 882.446 | 889.232 | 2894.633 |
| BF16 | 129,33,257 | 861.126 | 872.963 | 2925.076 |

This measures certificate/host overhead, not a kernel speedup. No physical
schedule or FP8/MXFP8/MXFP4 route is promoted by these results. Larger-row native
execution remains producer dominated.

## Remaining delivery gates

Complete final registry/generated-document gates, synchronize with merged main,
and run the full non-slow unit suite on the frozen synchronized source.
General producer, dynamic/resident and reverse admission remain follow-ups.
Apple, x86 and ROCm need their own physical consumers and architecture-specific
execution certificates; shared frontend correctness does not establish parity.
