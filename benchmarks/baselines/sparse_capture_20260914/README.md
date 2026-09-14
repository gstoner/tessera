# Sparse capture, byte formats and scan — 2026-09-14

Uncommitted correctness work after PR #747. Owning host: Tajasarus, RX 9070 XT
(gfx1201), ROCm 10.0, WSL2, LLVM 23.1.1 with assertions, and its Zen 5 CPU.
No timing, overlap, or performance promotion is claimed.

`focused_tests.txt` records 516 passing checks. `final_tests.txt` records
17 focused checks after the final capture-policy/type fixes (overlapping the
broader suite). Ruff passes and the mypy ratchet remains at zero errors.

Reproduce with the environment in `~/.config/tessera/env.sh`, `PYTHONPATH=python:.`
and `TESSERA_GFX1201_DEVICE_PROOF=1`:

- `test_rocm_sparse_byte_formats.py`: signed i8/i32 and same-format E4M3FN/E5M2
  inputs, two shapes (16x16x32 and 32x48x64), all six sparse pairs and signed
  values. Requires the exact SWMMAC instruction in HSACO disassembly and exact
  integer-valued numerical agreement. This is not exhaustive FP8 rounding proof.
- `test_sparse_capture.py`: explicit `JitFn.compile_sparse_2to4` captures a single
  matmul through the tracer, verifies Graph IR, and emits checked sparse packing
  and output-storage conversion. Rejects unconsumed layout, shard and epilogue
  semantics; invalid 2:4 inputs never expose a result. Ordinary JIT dispatch
  remains unchanged. Native Graph-to-sparse recipe lowering remains open.
- `test_attention_exact_cotangent.py` and public tests in
  `test_rocm_gfx1201_scheduled.py`: capture rejects lossy/nonfinite conversions;
  actual execution checks wider cotangents again before allocation.
- `test_isolated_rocm_attention.py`: zero/nonzero health admission, repeated
  saved/recompute calls, forced worker death, confirmed teardown and replacement.
  Forced process termination is not an actual driver-hang test or a GPU reset.
- `test_scheduled_cumsum.py`: native Schedule replay and tamper rejection,
  descriptor execution for three shapes with finite and exceptional inputs.

Unsigned/INT4/mixed-FP8 sparse formats, automatic sparse dispatch, arbitrary AD
composition, actual driver-hang recovery, remaining cohort/breadth migration and
calibrated performance attribution remain open. Device proof does not transfer
to gfx1151, CUDA, or Metal.
