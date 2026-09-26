# x86 Zen 5 profiler packet — Princess-Luna, 2026-09-26

`x86_zen5_profiler_packet_20260926_princess_luna.json` is the E2E-REAL-4 matmul
comparison (`benchmarks/x86/benchmark_x86_e2e_real_matmul.py`, shapes 64×128×96
aligned and 127×65×79 ragged) recorded with Zen 5 profiler evidence on the TSC
witness route (sync `WSL-TIMING-ADMISSION-2026-09-26`, `RUNTIME-LIB-OPT-1`).

- **Schema:** `tessera.profiler_x86_packet.v2`. Each row's witness stores its full TSC
  calibration and must bind the per-launch samples; it re-verifies off-host through
  `profiler_x86_clock.verify_witness_sample` (see
  `tests/unit/test_profiler_x86_evidence.py::test_checked_in_princess_luna_packet_validates`).
- **Host:** Princess-Luna (AMD RYZEN AI MAX+ 395, Zen 5), Ubuntu 26.04 under WSL2
  (kernel 6.18.33.1-microsoft-standard-WSL2).
- **Source:** clean tree at `f8022572` (branch `claude/amd-x86-alpha-lanes-avx512`,
  includes the verifier commit `b13beb66`), in a separate worktree with its own
  `build/` configured with no build type. Recorded after the AVX-512 E2E packet on the
  same host, never at the same time.
- **Runtime library:** `tessera_x86_elementwise` at
  `O2 (runtime-library default: tree has no build type)` — the `record_for_library`
  stamp is inside the digest-bound `benchmark` record.
- **Command:** `PYTHONPATH=python python benchmarks/x86/record_x86_zen5_profiler_packet.py
  --tprof <tools/profiler build>/tprof --output <outside the repo>` (timing witness on;
  `tprof` built standalone from `tools/profiler`).
- **Result:** `admission_route = tsc_witness`, `verdict = promote`,
  `ineligibility_reasons = []`. Every row's witness was promotion-eligible and named the
  row's own image.
- **What `promote` means here:** the production and scheduled images are
  **byte-identical** (same image digest) and agree exactly (max abs 0); the
  scheduled/production ratios (1.015×, 1.034×) are noise between two launches of one
  image. So the verdict is a **parity check under WSL2, not a performance promotion**.
- **Diagnostic gaps (recorded, not blocking):** `VIRTUALIZED_HOST`,
  `WSL_CLOCK_DOMAIN`, `TIMING_PROOF_INCOMPLETE:perf_event_open,perf_sample_valid`
  (`perf_event_paranoid=2`, `perf_event_open` errno 13 under WSL2),
  `SYMBOL_SAMPLING_MISSING` (no `--sampling` input).

WSL2 caveat: under Hyper-V the raw clock is itself derived from the TSC through a
hypervisor-supplied scale, so the witness shows a stable TSC scale across intervals,
not agreement with an independent oscillator. The latencies (~0.75 ms at 64×128×96)
are dominated by the public launch path, not the GEMM; see the x86 queue entry
`AVX512-E2E-PACKETS-2026-09-26`.

Replaces the v1 packet recorded at `7b3094e9`, which carried an admission route under
the v1 schema and witnesses without their calibration.
