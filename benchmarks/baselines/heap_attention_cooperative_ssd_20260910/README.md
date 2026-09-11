# Heap IR, raised attention and cooperative SSD

Measured 2026-09-10 in the uncommitted continuation after PR #741. Super-Bear owns CUDA SM120 evidence; Princess-Luna owns ROCm gfx1151 evidence. Embedded compiler/image digests and source-hashes bind the measurements. No promotion is claimed.

## Correctness

`cuda-correctness.json` and `rocm-correctness.json` cover T=5, H=2, N=3, P=2, chunks 1/2/5, padded lanes, immutable inputs, Y, final carry and partial checkpoints. The cooperative block owns a head/value column; state lanes communicate through shared memory and two barriers per time step. The ordered leader reduction preserves state-index summation order.

Two raised dense f32 attention buckets execute on SM120 through the existing status-checked native package. The test forbids GraphIRModule construction during binding and compares the retained loop oracle. It does not admit an arbiter candidate or transfer evidence to ROCm/Apple.

Native heap tests compile production C++ and automatically generated MLIR/LLVM calls. They cover cycles, roots, exhaustion cleanup, repeated instantiation, generated-library close, and actual native source-state exception decoding with a successful subsequent invocation. This is static table materialization on the CPU, not runtime-valued iteration allocation, GPU heap execution or CPython frame reconstruction.

## Measurements

Serial and cooperative JSON packets use T=32, H=2, N=16, P=4, chunk=8. Each has seven event windows of 100 resident launches after a checked correctness call. The event windows exclude transfers and descriptor validation but include host submission gaps; they are not kernel-only timings. Binding and checked-call wall times are recorded separately and are single observations, not statistical performance claims.

The recorded median event windows are about 0.0994 ms serial / 0.0140 ms cooperative on CUDA, and 0.1342 ms / 0.0137 ms on ROCm. These same-device observations favor the candidate but are not calibrated cross-run selector evidence. Earlier exploratory CUDA windows differed substantially because this window includes submission gaps.

Nsight CSVs independently report CUDA kernel, transfer and API timings from instrumented runs. Do not compare their absolute durations to uninstrumented event-window values. Full `.nsys-rep` files remain on Super-Bear under `/tmp/ssd-final-{serial,cooperative}.nsys-rep`.

ROCProfiler was requested to trace kernels, memory copies and HIP calls. On this WSL host it produced only agent information and HIP API records; the checked-in summary explicitly lists missing kernel/copy/counter evidence. Raw traces remain under `/tmp/ssd-rocprof-final/` on Princess-Luna. HIP events establish neither hardware-counter attribution nor complete kernel trace support.

## Validation

Assertions-enabled compiler build; 476 focused registry/runtime/compiler/audit tests passed with three environment-gated skips. A separate owning-CUDA run passed eight attention tests, and Princess-Luna passed three SSD package tests. Ruff and the zero-error mypy ratchet pass. Full unit suite not rerun in this increment.
