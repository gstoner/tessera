# Sparse runtime and isolated attention — 2026-09-14

Uncommitted correctness evidence after PR #747, measured on Tajasarus:
Radeon RX 9070 XT gfx1201, ROCm 10.0, LLVM 23.1.1 assertions-enabled,
and its Zen 5 host CPU. No selector-grade timings or performance promotion.

- `focused_tests.txt`: 458 tests, including eight native sparse cases (f16/bf16,
  f32 or matching low-precision accumulators, 16x16x32 and 64x48x128).
  Disassembly must contain the corresponding SWMMAC opcode. Quarter-valued
  inputs are exactly representable; this is not exhaustive rounding proof.
  Device validity is checked before output exposure, including invalid final
  groups followed by valid execution in a new worker.
- `attention_device_tests.txt`: seven isolated/reusable attention tests. A zero
  first cotangent followed by constant nonzero cotangents exposed a dropped
  saved-LSE forward carrier. The corrected direct Tile producer propagates the
  saved policy only with a reciprocal saved backward companion. Saved and
  recompute execute against the oracle; forced worker death requires confirmed
  teardown and a fresh zero-VJP probe before replacement admission.
- `drift_tests.txt`: 83 passing audit/dtype/ownership tests; eight compiler
  fixture cases skipped. The eight new sparse runtime cases compile and execute
  on gfx1201 independently of those fixture skips.
- `route_census.json`: refreshed lexical route inventory, not execution proof.
- `x86_ceil.json`: three native descriptor shapes, bitwise finite/zero/infinity
  checks and NaN classification. Source/image identities are recorded.

Reproduce in owning WSL with `TESSERA_GFX1201_DEVICE_PROOF=1` using
`test_rocm_sparse_runtime.py`, `test_isolated_rocm_attention.py` and
`test_rocm_gfx1201_scheduled.py -k reusable_attention_owner`. Host-only future,
conversion, timeout and failure tests do not establish device overlap.

Sparse binding is an explicit compiler API; automatic Graph/JIT capture,
integer/FP8 sparse packing, arbitrary AD composition and pointer transport across
isolation remain open. Forced process death is not an actual driver-hang test or
GPU reset. WSL profiler restrictions still prevent calibrated kernel attribution.
