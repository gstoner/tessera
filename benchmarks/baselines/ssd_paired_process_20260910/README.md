# Independent paired SSD process measurements

Uncommitted continuation after PR #741. Each backend ran nine prespecified pairs
of fresh processes, alternating serial/cooperative order. Each process checked
all outputs and immutable inputs, then recorded seven windows of 100 resident
launches. The compiler, binding and image identities are stable within variants.
Raw per-process packets are embedded in each JSON file.

| Owning host | Median speedup | Conservative median lower bound |
|---|---:|---:|
| Super-Bear, RTX 5070 SM120 | 6.8788x | 6.7440x |
| Princess-Luna, gfx1151 | 9.7745x | 9.7295x |

The lower bound is the second of nine ordered paired speedups: its one-sided
coverage is 1 - 10/512 (at least 95%), assuming independent representative pairs.
Fresh processes do not eliminate correlated thermal or load conditions.

The objective is resident device-event elapsed time **including host submission
gaps**, excluding transfers and descriptor validation. It is not kernel-only
latency. No device-clock calibration, production selector binding or route
promotion is claimed. ROCm hardware counters remain absent; CUDA profiling from
the prior increment is separate instrumented evidence. Comparisons do not transfer
between architectures. `compare_ssd_variants.summarize` rejects inconsistent
artifact identities and malformed timing values, including booleans.

Reproduce with `benchmarks/compare_ssd_variants.py --backend nvidia|rocm
--compiler /absolute/path/to/tessera-opt --output packet.json` on the owning host.

Validation of this continuation: 489 focused tests passed, four device/toolkit
skips on Super-Bear; separate owning CUDA attention run passed 10 tests (one x86
skip), and owning Zen 5 attention run passed 10 tests (one CUDA skip). Native
exception tests compile the production C++ heap ABI. SSD gradients execute native
CPU MLIR and compare all five input gradients to finite differences. Ruff and the
zero-error mypy ratchet pass. No full unit-suite run is claimed.
