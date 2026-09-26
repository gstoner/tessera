# x86 Zen 5 profiler packet — Princess-Luna, 2026-09-26

`x86_zen5_profiler_packet_20260926_princess_luna.json` is the E2E-REAL-4 matmul
comparison (`benchmarks/x86/benchmark_x86_e2e_real_matmul.py`, shapes 64×128×96
aligned and 127×65×79 ragged) recorded with Zen 5 profiler evidence on the TSC
witness route (sync `WSL-TIMING-ADMISSION-2026-09-26`, `RUNTIME-LIB-OPT-1`).

- **Host:** Princess-Luna (AMD RYZEN AI MAX+ 395, Zen 5), Ubuntu 26.04 under WSL2.
- **Source:** clean tree at `7b3094e9` (branch `claude/amd-x86-alpha-lanes-avx512`),
  in a separate worktree with its own `build/` configured with no build type.
- **Runtime library:** `tessera_x86_elementwise` at
  `O2 (runtime-library default: tree has no build type)` — the `record_for_library`
  stamp is inside the digest-bound `benchmark` record.
- **Command:** `PYTHONPATH=python python benchmarks/x86/record_x86_zen5_profiler_packet.py
  --tprof <tools/profiler build>/tprof --output <outside the repo>` (timing witness on;
  `tprof` built standalone from `tools/profiler`).
- **Result:** `admission_route = tsc_witness`, `verdict = promote`,
  `ineligibility_reasons = []`. Both rows' TSC witnesses were promotion-eligible and
  named the row's own image; TSC and `CLOCK_MONOTONIC_RAW` agreed to within 2e-7.
- **Diagnostic gaps (recorded, not blocking):** `VIRTUALIZED_HOST`,
  `WSL_CLOCK_DOMAIN`, `TIMING_PROOF_INCOMPLETE:perf_event_open,perf_sample_valid`
  (`perf_event_paranoid=2`, `perf_event_open` errno 13 under WSL2),
  `SYMBOL_SAMPLING_MISSING` (no `--sampling` input).

Caveat: under WSL2 the raw clock is itself TSC-derived, so the witness shows a stable
TSC scale across intervals, not agreement with an independent oscillator. The
scheduled/production latencies (~0.75 ms at 64×128×96) are dominated by the public
launch path, not the GEMM; see the x86 queue entry `AVX512-E2E-PACKETS-2026-09-26`.
