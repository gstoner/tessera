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
The broader non-slow suite is running; this PR remains draft pending that gate.

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
A fresh full lane is running against immutable compiler SHA e4e33848...;
do not treat the original failed run or targeted repairs as full-suite closure.
