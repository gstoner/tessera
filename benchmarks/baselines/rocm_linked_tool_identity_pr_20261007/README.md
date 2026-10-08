# ROCm linked-tool identity and bounded metadata cache

Recorded by benchmarks/rocm/record_native_package_metadata_cache.py.

Owner FRONTEND-IR-MEDIUM-1 / E2E-REAL-6.
Synchronization key ROCM-LINKED-TOOL-IDENTITY-2026-10-07.

This focused PR changes only ROCm compiler identity/metadata helpers, tests,
the reproducible existing-route recorder and backend synchronization notes.
It does not include the accumulated five-slice compiler/runtime changes.

Image-cache identity hashes executable and loaded ELF library contents.
Metadata reuse watches executable/dependency stat signatures, loader
environment, loader cache and RPATH/RUNPATH search directories, including
transitive dependencies. Discovery/digest/version caches are bounded.
Static ELF and non-ELF executable content identities remain supported.
An unchanged warm identity executes no external process.

Host WSL validation: 372 passed, four hardware skips. Real ELF fixtures
exercise replacement, symlink retargeting, higher-priority library addition,
static linking and subprocess-free warm identity. Both existing image/cache
contracts and diagnostic/pass metadata gates are included.

Owning gfx1201 validation: 103 passed. This combines scheduled route host
contracts, opt-in GPU execution and linked-identity tests; it does not claim
103 GPU profiles. One timeout-plugin configuration warning is retained.

All 12 RMSNorm profiles (three shapes, FP16/FP32, two image-cache modes) pass
independent float64 numerics before/after seven alternating metadata A/B
trials. Compiler/toolchain fingerprints and native image bytes agree between
arms. Image caches are explicitly cleared before both arms in cold mode and
primed before both arms in warm mode.

- empty_image_caches: uncached/cached ratios 1.2232–1.2421, median 1.2316; subprocess counts {'uncached_versions': [7], 'cached_versions': [5]}.
- reused_image_caches: uncached/cached ratios 0.9941–1.0236, median 1.0073; subprocess counts {'uncached_versions': [3], 'cached_versions': [3]}.

Version reuse reduces queries when creating an image. Warm image hits
already bypass those queries and show no material metadata speedup.
These are package wall timings, not kernel timings or promotion evidence.
AST-only Graphify refresh passed (117806 nodes, 201253 edges).
The final matching-source non-slow suite is green; see the final validation receipt below.

Source is isolated on main commit 6bfd1a8ae89aaa3f6063f7148530488661f0c7a3.
The actual compiler binary comes from the matching five-slice integration
build and is bound by SHA in benchmark.json. This PR has no C++ lowering
changes; these receipts do not claim a newly rebuilt compiler from this
isolated checkout. gfx1151/other-family performance proof and broader
compiler closure remain open. No sibling physical schedule is promoted.

## Broader validation repair

The first non-slow run reported six failures, 17884 passes and 9916 skips.
One failure was this recorder's missing documentation reference, now fixed.
The other five used nonexistent build paths derived from the isolated
checkout: two compiler-location failures and three CUDA ReplaySSM failures.
All 20 targeted repair cases pass with absolute compiler/runtime paths.
The first run completed before the subsequent native compiler rebuild began.
A subsequent historical full lane used immutable compiler SHA e4e33848...;
do not treat the original failed run or targeted repairs as full-suite closure.

## Matching-source validation repair, 2026-10-08

The isolated-source compiler and backend tools were rebuilt with LLVM/MLIR
23.1.1. Scratch ROCm driver resources and its llvm/bin/ld.lld were supplied for
host-only serialization. Rechecking the prior failed nodes resolved the source
mismatch; after linker repair, 33 checks passed with 10 owning-ROCm execution
skips. A 50-check lane with CUDA libraries preloaded passed after isolating the
emitted-identity fixture's discovery state.

Fresh full non-slow result: 516 failed, 19,900 passed, 7,390 skipped,
874 deselected, 1131.48 seconds. All failures are in test_x86_compile_cache.py
(8) and test_x86_kernel_differential.py (508); the matching-source validation
build had TESSERA_BUILD_X86_BACKEND disabled. That validation build was subsequently rebuilt with
x86 enabled. This historical failed receipt is not a passing full-suite claim.

## Final matching-source validation, 2026-10-08

The x86-enabled matching-source compiler completed the full non-slow suite:
**20,419 passed, 7,387 skipped, 874 deselected**, 10 warnings, 1147.19 seconds.
Compiler SHA256 before and after the run is
95b2f94395680e24347ede9840fa6e40af12a9e61bad5c3e2566946e5cfae0e3.
The x86 cache fixture separates actual compiler actions from cold ELF dependency
discovery while retaining the zero-subprocess repeat-hit assertion. The emitted
identity fixtures isolate loaded CUDA library state before testing discovery.
The affected x86 pair passes 560 checks with 514 environment skips.
Final registry/drift gates pass 287 checks; generated-document checks pass.
Receipts: full-x86-enabled-unit.txt, full-x86-enabled-compiler-before.txt,
full-x86-enabled-compiler-after.txt, final-drift-tests.txt and
final-generated-doc-check.txt. Hardware skips do not establish ROCm or Apple
execution parity; the owning gfx1201 benchmark remains separately bound to its
recorded compiler/image identities.

Published text logs normalize trailing whitespace only; original terminal logs are retained in the host scratch validation archive. Test outcomes and diagnostic text are preserved.
