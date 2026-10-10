# Native ROCm math ancestry replay reuse

Owner E2E-REAL-6 / FRONTEND-IR-MEDIUM-1. Sync ROCM-MATH-REPLAY-2026-10-08.

MathRecipe validation now reuses pure native Graph-to-Schedule and Schedule-to-Tile pass outputs. Every invocation still compares original Graph/Schedule/Tile ancestry and validates the native descriptor record. The shared cache retains its 256-entry / 16-MiB limits, full environment, cwd, compiler and loaded-library identity, exact input text and executor identity. Failures are not retained.

68 host WSL tests pass, including warmed-cache Graph/Schedule/Tile/target mutations, changed source text and repeated invalid native Graphs. A checkout timestamp warning is retained: the compiler is a content-hash-bound snapshot, not a claimed fresh all-source build. Each owning host passes 19 cache safety tests. Tajasaurus lacks pytest-timeout; its config warning is recorded.

The existing matmul A/B recorder now has a math family option; no duplicate recorder is introduced. All six operations (sqrt, exp, add, div, cumsum, cummax), f32/f16/bf16 storage, and shapes 3x17 / 64x257 pass independent float64 numerical checks before timing on each exact architecture: 36 profiles per host, 72 total. Portable serialized packages execute with native_gpu receipts. Seven alternating pairs vary only replay retention; images, entries and tool fingerprints are equal in every arm. All profiles remove two ancestry subprocesses: three to one, with zero version queries in either warm arm.

| Owning architecture | Uncached median range, ms | Cached median range, ms | Median per-profile ratio |
| --- | ---: | ---: | ---: |
| gfx1151 | 57.946–63.921 | 19.999–22.444 | 2.887 |
| gfx1201 | 42.060–44.708 | 16.367–17.216 | 2.573 |

These are warm-image package wall costs, not isolated kernel time or end-to-end execution speed. No physical schedule or selector is changed. rocminfo and compiler/source hashes bind each architecture; gfx1151's placeholder UUID is not treated as a unique device identifier. Different hosts are not compared against each other.

Old matmul/unary packets retain their original source hashes and recorder semantics. Broader family/image-key envelopes, generic scaled-matmul transformations, and the five-slice aggregate remain open.
