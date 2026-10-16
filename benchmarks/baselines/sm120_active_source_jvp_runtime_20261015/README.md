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

Latest synchronized benchmark compiler SHA256:
4933ce714af733dfdf8f4bb7eb1461c65ac979852822e5b41741a9f98a246f4a.
Frozen 3659a2b45 native owner SHA256:
2f51cf156777a470b803fc9ed8c5f3eb3a9396d0802651d49c7ecc8ec99ee689.
publication-timings.json records raw executable hashes separately from the
dependency-aware compiler identity, plus the live GPU UUID and source hashes.
normal-build-timings.json and the saved FP16 fixture retain the earlier
81a98c3b compiler epoch; they are not the synchronized build certificate. See
docs/spec/ir/source_active_row_jvp.md for the exact envelope.

## Validation

Frozen synchronized source **3659a2b457f27835f033bc1c2f6fa735f2194f53**
passes the full non-slow unit gate on host WSL: **26,630 passed, 7,678 skipped,
876 deselected, 11 warnings in 2017.98 seconds**. Source remained clean and
unchanged through the run. full-unit-3659a2b45.json records the exact commands,
tool/runtime hashes and GPU; the compressed raw log is retained. The preceding
audit/routing gate passes 778 checks. Ruff and the zero-error mypy ratchet pass.
The final attention checkpoint controls pass **35 tests in 44.60 seconds**
on this same source and owning GPU; their raw log is retained.

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

`publication-timings.json` records seven samples, five completed calls per host sample and
alternating mode order. All modes retain the same live native owner and image
identities. The full-certificate control clears only the pure validation cache;
it does not clear a CUDA module/allocation cache. Correctness is checked before
and after timing against an independent FP64 derivative/product oracle. Device
stage events are separately recorded. No competing GPU test/benchmark ran.

| Storage | MNK | Public JVP us | Cached portable replay us | Full certificate replay us |
| --- | --- | ---: | ---: | ---: |
| FP16 | 17,19,35 | 847.278 | 854.779 | 2890.900 |
| BF16 | 17,19,35 | 756.597 | 779.410 | 2827.671 |
| FP16 | 129,33,257 | 888.917 | 888.792 | 2930.118 |
| BF16 | 129,33,257 | 851.672 | 889.772 | 2913.185 |

This measures certificate/host overhead, not a kernel speedup. No physical
schedule or FP8/MXFP8/MXFP4 route is promoted by these results. Larger-row native
execution remains producer dominated.

## Remaining delivery gates

Merged main through PR950 is integrated. Frozen full-unit and owning-device
gates above are complete. Publication CI remains a delivery gate.
General producer, dynamic/resident and reverse admission remain follow-ups.
Apple, x86 and ROCm need their own physical consumers and architecture-specific
execution certificates; shared frontend correctness does not establish parity.

## Reproduce the recorder

Recorded by benchmarks/nvidia/benchmark_active_source_jvp.py. Configure matching
TESSERA_OPT, TESSERA_NVIDIA_OPT and TESSERA_NVIDIA_PTX_LAUNCH_LIB paths, then run
from the repository root on the owning SM120 host:

```sh
PYTHONPATH=python:. python benchmarks/nvidia/benchmark_active_source_jvp.py \
  --output ~/scratch/source-active-jvp/timings.json
```

The mandatory output argument keeps generated packets in the selected scratch
location. The recorder queries and checks the actual GPU before measurement.

## Synchronized full-suite audit repair

Frozen 951658503 on Super-Bear WSL completed the full non-slow tests/unit
command: 26,628 passed, 7,678 skipped, 876 deselected and two failed in
2010.33 seconds. Both failures were audit drift: the new recorder was unnamed
and the ODS record-count pin lacked the five new shipped operations.
The recorder now has the reproduction entry above; the ODS scan explicitly
pins all five names and the exact count of 684. All five classify as having
shipped compiler consumers. Both complete audit files pass: 770 checks.

full-unit-951658503.json and the compressed raw log retain the original failed
sweep without relabeling it green. The repair changes documentation and audit
expectations; compiler/runtime/numerical code is unchanged. The fresh frozen
3659a2b45 run above passes both repaired audits and the complete non-slow suite.
Publication CI remains required.


## PR951 review corrections

Both findings reproduce on the original code/provider: 28 new regression cases
fail (four narrower eager accumulator policies and 24 in-place root mutations,
covering FP16/BF16, argument permutations, all three roots and captured/direct
profiling). review-before.txt retains this counterexample run.

Eager normalization now rejects an explicit accumulator with fewer significand
bits than storage, matching native policy admission. Source-active profiling
requires the existing live frame, refreshes all host root contents, and completes
one warmup invocation before opening event/capture windows. Returned outputs
reflect in-place mutation; staging/warmup are excluded from device timing.
The native ABI and compiler images are unchanged.

The integrated native owner was rebuilt from current repository sources.
All 436 affected policy, source/RHS device, diagnostic and pass-metadata checks
pass in 55.76 seconds; Ruff passes. review-after.txt retains the test run.
review-fixed-timings.json records a fresh four-case independent-oracle benchmark
with the rebuilt provider, keeping same-owner host costs separate from captured
stage/program events. Its runtime SHA256 is:
7325c8fd152b817022fa7861591dea269d598254216a04217bad6cb7385c2b51.

The 26,630-pass frozen full-unit certificate above remains scoped to 3659a2b45.
It does not certify these later review corrections; publication CI must pass
on the updated PR head.


## PR951 compiler-route CI correction

The compiler builds in the failed updated-head CI run. The route proof then
fails three BF16 ReLU cases because the test interpreter lets NumPy 1.26 infer
an FP16 dtype for scalar 0.0 inside where; that dtype cannot promote with BF16.
A scratch WSL Python 3.11.17 / NumPy 1.26.4 / ml_dtypes 0.5.4 environment
reproduces precisely the three failures.

The oracle now gives a scalar fill the Graph operand's declared storage dtype.
The independent tangent expectation uses zeros_like plus copyto, preserving a
separate formulation and all existing numerical assertions. No compiler/runtime
code, marker, skip state, dependency pin or CI fan-in is changed.

The complete compiler-route lane passes 822 tests in 176.69 seconds under the
matched CI numerical stack; the JUnit execution gate confirms every selected
proof test executed. Three warnings are expected NaN comparisons. The current
Python 3.14/NumPy 2.5 stack passes 103 affected tests. Ruff passes.
ci-numpy126-validation.json binds the tested files and compiler digest; compressed
failed logs/JUnit and plain passing logs retain the original evidence.
Updated-head publication CI remains required.
