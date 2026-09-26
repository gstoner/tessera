# gfx1151 canonical GEMM on the scheduled route — 2026-09-26

First packet from the rebuilt `benchmarks/rocm/benchmark_rocm_canonical_gemm_kloop.py`
after Lane B was retired (`docs/audit/backend/rocm/ROCM_LANE_MAP.md`, §"Decision — Lane B is
retired"). Every row is Graph -> Schedule -> Tile -> `tessera_rocm` -> HSACO through
`runtime.build_canonical_gemm_hsaco`, launched from the package descriptor.

- Host: **Princess-Luna** (Ryzen AI MAX+ 395, gfx1151), Ubuntu 26.04 under WSL2, ROCm 10.0.
- Source: branch `claude/amd-x86-alpha-lanes` at `8880804f`, clean worktree, full `ninja -C build`.
- Command: `TESSERA_ROCM_CHIP=gfx1151 python benchmarks/rocm/benchmark_rocm_canonical_gemm_kloop.py`
  (defaults: 5 warmup, 9 rounds x 50 launches), after `scripts/_rocm_env.sh`.
- Same host, same tree: `check-tessera-rocm` 78/78, `check-tessera-ir` 443 passed / 68 unsupported,
  ROCm GEMM unit tests 2124 passed / 56 skipped.

Result: all six rows correct (f16/bf16 normalized error <= 1.3e-6; int8 exact). Physical route
`gfx1151_register_wmma_2x4`, macro tile 32x64. Every round's clock was `device_event` (a HIP event
admitted inside the two-sided band around wall time).

**Not comparable with `../rocm_gfx1151_canonical_gemm_kloop.json`** (Lane B): that packet timed
launch + `hipDeviceSynchronize` per iteration on the wall clock, which includes the synchronize
round trip; this one reports banded per-launch device time over 50-launch batches. Different
route and different method, so no ratio between them is claimed. This packet is the first baseline
for the scheduled route; no ratchet is set from a single recording.

Noted, not acted on: the f16/bf16 register-body images report 45-48 spills
(`artifact.resources.spill_count`); int8 reports 1.
