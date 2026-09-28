# x86 package compile cache: cold vs warm (2026-09-28)

Recorded by `benchmarks/x86/measure_x86_package_cache.py` on Princess-Luna
(Zen 5, AVX-512) under `flock /tmp/tessera-timing.lock`, load average < 2 at
start, from the deduplicated `codex/x86-batch3-dedup` worktree. Cited by the
`E2E-REAL-6-x86-kernel-2026-09-28` log entry and the x86 queue.

**Claim: compile cost only** — the wall time of one Graph-input package call.
It says nothing about kernel speed (the shared object and its entry points are
unchanged; the device differential proves the executions bit-identical).

`princess_luna.json` rows, per case (5 interleaved cold samples, 7 warm). The recorder clears the compiler-run cache and the #875 Schedule, unary-lowering, and verified-package caches before every cold sample:

- `retired_cold_*`: the retired Graph-owned constructor
  (`tests/_support/x86_kernel_baseline.py` / `x86_unary_baseline.py`) from an
  empty cache — one `tessera-opt` Tile -> Target run.
- `compiled_cold_*`: the compiled Graph -> Schedule -> Tile route from an empty
  cache — Graph -> Schedule, Schedule -> Tile, Tile -> Target (plus
  `--canonicalize` for the unary/norm projection); the replays are hits within
  the same call. `cold_compiler_runs` counts them.
- `compiled_warm_*`: an exact repeat — every compiler boundary is a hit
  (`warm_compiler_runs` = 0); descriptors and replay comparisons are rebuilt.
- `process_first_call_ms`: the one-off per process that also SHA-256s the
  ~185 MB `tessera-opt` binary (memoized on its stat signature).

On this run, compiled cold medians were 53.65-82.10 ms and warm medians
0.29-0.95 ms with zero warm compiler runs. Retired cold medians were
25.44-26.86 ms except the retained absolute path (82.37 ms). These are
package wall times, not kernel timings.
