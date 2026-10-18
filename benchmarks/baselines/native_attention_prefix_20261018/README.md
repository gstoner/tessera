# Native source RMSNorm to saved-LSE attention

Owner: W1.1 / FRONTEND-IR-MEDIUM-1 / AD-RESIDUAL-EVAL-1 / E2E-REAL-6.
Synchronization key: NATIVE-COMPOSED-ATTENTION-PREFIX-20261018.

This candidate retains the original static FP16/BF16 source Graph and native
SSA members through Schedule/Tile/NVIDIA/NVVM/PTX. The C++ owner retains a
private normalized-Q allocation and returns independent half O and FP32 LSE.

## Portable fixtures

portable/f16.runtime.json.xz and portable/bf16.runtime.json.xz contain compiled,
source-bound RuntimeArtifact JSON for the named small profile. index.json
records their compressed and expanded hashes. These are trusted native-code
fixtures. Content digests detect corruption, not authenticity.

tests/unit/test_portable_source_attention.py validates them with external
compiler invocation and native-provider acquisition forbidden. It checks
rehashed lifetime, binding, epsilon, guard, architecture, image and member
mismatches, and rejects parent Graph/target/argument/placement mismatches.

The owning numerical and lifetime suite is
tests/device/nvidia/test_source_attention_prefix.py. The reproducible timing
recorder is benchmarks/nvidia/benchmark_source_attention_prefix.py. It checks
reference O/LSE before and after timing and separates native member/composed
dispatch windows, prepared host calls and completed public JIT calls.

## Validation

- CI-equivalent unit selection: 30,734 passed, 3,725 declared skips.
- Production compiler-route: 2,016 passed, zero skips; executed-proof certificate.
- RTX5070: 35 source/public/portable cases plus 70 adjacent forward cases.
- Separate isolated JVP/VJP ownership regression: 34 passed, zero skips.
- Focused compiler/registry/ABI/documentation gates: 1,571 passed.
- Affected spectral regression: 79 passed, five host-capability skips. The four
  initial owning ROCm failures were repaired by building the existing provider
  for live RX9070XT/gfx1201; this is not source-attention parity on ROCm.
- Required Ruff scope and zero-error mypy ratchet pass.

manifest.json binds source/tool hashes and the validation scopes. Compressed
logs retain their complete selections. The initial broad run's documentation
and missing-provider failures were repaired without changing test states.

## Calling-convention review fix

Six genuine red regressions cover mixed keyword-only, all-keyword-only and
positional-only plus keyword-only parameters in FP16/BF16. The corrected
frontend binds tracer proxies to the original Python signature; the source
route retains the caller args/kwargs. The native/portable gate passes 57 tests.
The owning forward suite passes 105 tests; six additional ordinary cold calls
run without manual tracing and reuse their frames with compiler/eager calls
forbidden. The refreshed timing record binds the corrected source hashes.
The earlier 1,571 focused shared results retain their initial-increment scope.

## Timing

sm120.json records four correctness-gated profiles, each with 21 samples.
The shape tuple is B,Hq,Hkv,Sq,Sk,D,Dv. Values below are milliseconds.

| Storage | Shape | Native composed dispatch | Prepared completed host | Public completed host |
|---|---|---:|---:|---:|
| FP16 | 1,2,1,3,5,4,3 | 0.102240 | 0.272554 | 0.347925 |
| FP16 | 1,2,1,17,19,257,13 | 0.234336 | 0.430127 | 0.491278 |
| BF16 | 1,2,1,3,5,4,3 | 0.066304 | 0.224023 | 0.372616 |
| BF16 | 1,2,1,17,19,257,13 | 0.268704 | 0.439137 | 0.468707 |

Separate normalization and attention dispatch windows, sample arrays, numerical
errors, owner initialization and compiler costs remain in the JSON. Event
windows can include enqueue gaps. These are functional characterization
measurements, not pure-kernel or speedup claims.

compiler_guard_controls.json verifies twenty retained stage identities and four
unchanged PTX payloads across the pending-AD guard fix. Full checked image
identities change intentionally because the compiler fingerprint changes.
This does not claim image-key reuse across compiler builds.

## Remaining scope

Current-head GitHub CI/review and delivery remain pending. Composed
original-source reverse, broader layouts/dynamic/precision profiles and all
five aggregate slices remain open. No sibling physical execution or general
performance promotion is claimed.

Contract: docs/spec/NATIVE_SOURCE_ATTENTION_PROGRAM.md.
