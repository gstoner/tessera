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

All four candidate Python source hashes match the initial measured packets
at implementation commit 6333558fd; these initial files predate PR894 sync. Compiler binaries
are the existing recorded LLVM 23.1.1 snapshots, not fresh all-source builds.
Raw logs are retained losslessly as .log.gz files. Generic scaled-matmul
batching/transpose, wider dtype/layout envelopes, shape-independent identity
outside existing proved schedules and compiler discovery overhead remain open.


## Synchronized working-directory guard rerun

Source head 99979a347 includes merged PR894 and the explicit replay cwd key.
Both owning hosts reran the four numerical profiles and seven alternating A/B
pairs. Four candidate Python hashes match both new cwd packets. Images and
fingerprints remain equal in all arms; subprocesses remain three versus one.
No kernel performance or fresh all-source compiler-build claim is made.

| Architecture | Shape MxKxN | Uncached package ms | Cached package ms |
| --- | --- | ---: | ---: |
| gfx1151 | 16x16x16 | 61.95 | 20.822 |
| gfx1151 | 17x31x19 | 63.069 | 21.543 |
| gfx1151 | 64x256x64 | 62.914 | 21.879 |
| gfx1151 | 200x128x64 | 65.727 | 22.507 |
| gfx1201 | 16x16x16 | 43.894 | 16.742 |
| gfx1201 | 17x31x19 | 42.437 | 16.849 |
| gfx1201 | 64x256x64 | 43.608 | 17.02 |
| gfx1201 | 200x128x64 | 43.486 | 16.82 |

Owning tests: gfx1151 23 passed; gfx1201 22 passed, 1 missing NVIDIA-dialect
skip, with the same pytest-timeout warning. Earlier measurements and logs
remain intact under their original names; the cwd rerun has separate files.
