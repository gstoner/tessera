# Source recovery and asynchronous ownership evidence

Recorded 2026-09-08, uncommitted continuation after PR #736. Owning items:
W4-PRODUCT-1, AD-RESIDUAL-EVAL-1, W2.4a, E2E-REAL-6 and MSW-9.
Sync key: `SOURCE-ASYNC-FOUNDATION-2026-09-08`.

## Correctness scope

- `async-public-nvidia.json`: RTX 5070 / SM120, assertions-enabled LLVM23 compiler.
- `async-public-rocm.json`: independent gfx1151 execution on Princess-Luna.
- `source-cfg-native.json`: seven native LLVM CPU executions on Princess-Luna,
  covering nested branches/early returns and single-carry bounded while.

The GPU recorder exercises checked dynamic public shapes, scoped cross-stream
readers, joint-volume matrix temporaries and eight asynchronous capture/backward/
whole-frame retire iterations. Explicit context synchronization is forbidden in
the successful asynchronous path. Capture status flows to backward; generic
reader exposure requires checked completion. Packets contain source/compiler
fingerprints and do not qualify performance promotion. Exceptional cleanup,
arbitrary Python effects/CFG and hard driver-unload latency bounds are excluded.
Module unloading uses at most eight admitted workers; stalled/failed workers
retain their context and resources. This is bounded admission, not cancellation.

## Nonlinear ANN experiment

Nine fresh-process measurements per architecture compare an affine ANN fragment
with a terminal self-product (square). The transformed candidate requests native
elementwise fusion. Inputs and analytic bounds are checked before measurement;
scoped candidates are retired after selection.

| Device | Median speedup | Median order-statistic interval | Selected |
| --- | ---: | --- | --- |
| SM120 | 1.01155x | [0.99243, 1.02168] | Original |
| gfx1151 | 1.01337x | [0.99693, 1.02065] | Original |

Neither lower bound clears the 1.02 threshold. No production promotion occurred.
Timing is **warm package H2D + dispatch + D2H host wall time**, not device kernel
time or hardware-counter attribution. These small 8x4 cases do not establish
performance for broader ANN workloads. Raw reports and selector results are in
`ann-nvidia/` and `ann-rocm/`; report paths are relative within each directory.

## Validation

The host WSL unit run passed 18,528 tests (2,231 skipped; 870 slow deselected).
After final code adjustments, the assertions-enabled focused suite passed 252
tests and the native source/public-frame/x86-attention suite passed 26 tests.
Mypy retained the zero-error baseline. Exact-device proof does not transfer to
Apple, RDNA4 or CDNA targets. No Apple runtime was changed.

Recorded by `benchmarks/record_async_public_frames.py` (`async-public-{nvidia,rocm}.json`)
and `benchmarks/record_native_ann_execution.py` (`ann-{nvidia,rocm}/run-N.json`).
