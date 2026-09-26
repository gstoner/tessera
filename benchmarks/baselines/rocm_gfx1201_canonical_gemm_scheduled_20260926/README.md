# gfx1201 canonical GEMM on the scheduled route — 2026-09-26

The gfx1201 row owed by the Lane B retirement (`docs/audit/backend/rocm/ROCM_LANE_MAP.md`,
§"Decision — Lane B is retired"). Every row is Graph -> Schedule -> Tile -> `tessera_rocm` -> HSACO
through `runtime.build_canonical_gemm_hsaco`, launched from the package descriptor.

- Host: **Tajasarus** (RX 9070 XT, gfx1201), Ubuntu 26.04 under WSL2, ROCm 10.0, assertions-ON
  LLVM/MLIR 23.1.1 (`build/`).
- Source: branch `claude/amd-x86-alpha-lanes` at `541b04ce`, clean detached worktree, full
  `ninja -C build`. Nothing else was running on the box (checked before the run).
- Command: `TESSERA_ROCM_CHIP=gfx1201 TESSERA_GFX1201_DEVICE_PROOF=1 python
  benchmarks/rocm/benchmark_rocm_canonical_gemm_kloop.py` (5 warmup, 9 rounds x 50 launches).
- Same tree, same session: ROCm lit 82/82, IR lit 450 passed / 66 unsupported (assertions and
  Release trees), gfx1201 split-K + scheduled + canonical-GEMM device tests 143 passed.

Result: all six rows correct (f16/bf16 normalized error <= 9e-7; int8 exact). Physical route
`gfx1201_register_wmma_1x1` (macro tile 16x16), `split_k` 1 on every row (these shapes are outside
the split rule), no spills. Every round's clock was `device_event` (HIP event admitted inside the
two-sided band around wall time).

Not compared with anything: gfx1151 runs a different panel (2x4, 32x64) at these shapes
(`../rocm_gfx1151_canonical_gemm_scheduled_20260926/`), and evidence never transfers between the
two parts. Observed, not explained: the f16 aligned row (0.0055 ms) is about 2x faster than the
bf16 and int8 aligned rows. First baseline for this route on gfx1201; no ratchet from one recording.
