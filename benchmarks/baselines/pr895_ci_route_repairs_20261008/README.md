# PR895 CI route repairs — 2026-10-08

Owner E2E-REAL-6 / FRONTEND-IR-MEDIUM-1 / W1.1.
Sync PR895-CI-ROUTE-REPAIR-2026-10-08.

The first CI aggregate unit job reported 327 failures and 49 setup errors.
Two failures reflect actual generic scaled_matmul batching/transpose gaps.
Other groups came from native compiler/image tests running in the CPU lane,
two ignored portable attention fixtures, an uncited receipt, and stale rank
expectations after named gfx1201 add and nested SM120 NVFP4 integration.

Native parser/pass/ancestry tests now carry compiler_route and run in the
required production compiler lane. HSACO packaging additionally requests the
ROCm image toolchain fixture; absence of AMD clang/device libraries is stated
explicitly. Frontend-only projection tests remain in the CPU lane. SM120
execution tests carry hardware_nvidia; device detection alone no longer runs
them in the CPU lane. Assertions and generic coverage flags are retained.

The two existing attention fixtures are force-added despite artifacts/ being
ignored. Their host-independent pin/corruption/ownership checks pass: 45 tests.
Affected CPU selection without compiler environment: 632 passed, 67 skipped,
417 deselected. Affected lane with native tools: 1042 passed, 29 skipped,
one compiler freshness warning, recorded verbatim. The initial matching-source rebuild and full CPU result are recorded below.
Mypy remains zero against unchanged zero baseline; source Ruff passes.

These are test-routing and fixture repairs, not expanded hardware admission
or generic closure. Required CI still needs a fresh run. The separate ROCm
replay-cache implementation is not included in this repair.


## Fresh core and final migration checks

Core compiler rebuilt successfully. Post-build native route checks: 112 passed,
80 deselected. Post-build exact RTX 5070 tests: 44 passed.
The first full CPU selection: 6 failed, 19445 passed, 9351 skipped.
Two failures are the unchanged generic scaled_matmul closure gaps. The other
four exposed exact-device tests under unit roots and mixed host/CUDA replay
checks. Five SM120 execution functions and two CUDA async replay functions are
now moved into tests/device/nvidia, retaining their numerical/lifetime assertions.
The host factory test explicitly simulates an unavailable CUDA constructor;
a separate device test asserts the real factory binds native state.
Final owning-device migration lane: 78 passed. Focused host/location lane:
13 passed, 22 deselected. The final full CPU run after these repairs reports 2 failed, 19447 passed,
9351 skipped. Both failures are the unchanged generic scaled_matmul batching
and transpose closure assertions. Required remote CI needs a new run.

Source repair commit: 98b715dfa. Ruff passes; mypy errors=0, baseline=0.
Final audit/citation tests: 14 passed. Generated documents: 32 in sync.
Graphify refresh completed: 233247 nodes, 407402 edges, 11576 communities.

Validation logs are retained byte-for-byte as .log.gz files. Decompress with gzip -dc.

## CI eviction fixtures — 2026-10-08

GitHub unit job 113483623479 at head 6186a6595 completed with four failures,
19,405 passes and 9,397 skips. Two failures are the existing scaled_matmul
batching/transpose closure assertions. Two are missing qkv_v_5_0 artifacts
used by JVP/VJP registration-eviction tests: the original recorded files were
locally present but ignored by Git. Both are now tracked without modification.
Their SHA256 values and the original CI log are retained here. The two affected
host WSL replay suites pass all 24 tests after staging. This repair does not
close generic batching/transpose or establish fresh device numerics.
