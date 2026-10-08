# Native ROCm matmul replay cache — 2026-10-08

Owner E2E-REAL-6. Sync ROCM-MATMUL-REPLAY-CACHE-2026-10-08.
Follow-up to the unary-only replay cache in PR896.

The existing exact-source, compiler/library/environment/pass cache now serves
ROCm matmul's native Schedule-to-Tile and canonicalization replay. Every call
still compares Tile text and projects descriptor fields from native MLIR.
Apple, x86 and NVIDIA retain their existing projection paths. No schedule,
native image, numerical policy or runtime ABI changes.

Both live GPUs were probed before execution: Princess-Luna Radeon 8060S
gfx1151 and Tajasaurus RX 9070 XT gfx1201. Four fp16 profiles on each host pass
independent float64 comparison before seven alternating package A/B pairs.
Images, entry symbols and compiler/toolchain fingerprints match every arm.
Warm image state is required; subprocess counts fall from three to one,
and version queries remain zero.

| Architecture | Shape MxKxN | Uncached package ms | Cached package ms |
| --- | --- | ---: | ---: |
| gfx1151 | 16x16x16 | 63.674 | 21.642 |
| gfx1151 | 17x31x19 | 64.341 | 21.063 |
| gfx1151 | 64x256x64 | 64.212 | 20.667 |
| gfx1151 | 200x128x64 | 63.138 | 21.337 |
| gfx1201 | 16x16x16 | 43.045 | 17.017 |
| gfx1201 | 17x31x19 | 42.886 | 16.966 |
| gfx1201 | 64x256x64 | 43.157 | 17.002 |
| gfx1201 | 200x128x64 | 43.149 | 17.081 |

These are package wall times, not kernel speedups. Maximum numerical errors:
4.072e-6 on gfx1151 and 1.450e-6 on gfx1201, within the recorder's existing
fp16 numerical tolerance. No physical default is promoted.

Host WSL gates: 33 cache/audit/native tests; 51 existing ROCm consumer tests
with 4 skips; 11 focused native tests including NVIDIA isolation. Source
Ruff passes. Owning gates: gfx1151 22 passed; gfx1201 21 passed, 1 skipped
because its compiler does not register the NVIDIA dialect. The first
gfx1201 attempt lacked the current production_compiler fixture; its failed
setup log is retained separately, then the current fixture was synced and
the test and benchmark rerun passed. No skipped or failed setup row is proof.
gfx1201 lacks pytest-timeout; timeout enforcement is not claimed.

All four candidate Python source hashes match both packets. Compiler binaries
are the existing recorded LLVM 23.1.1 snapshots, not fresh all-source builds.
Raw logs are retained losslessly as .log.gz files. Generic scaled-matmul
batching/transpose, wider dtype/layout envelopes, shape-independent identity
outside existing proved schedules and compiler discovery overhead remain open.
