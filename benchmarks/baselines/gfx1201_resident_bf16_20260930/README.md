# gfx1201 BF16 resident RMSNorm-to-matmul diagnostic

Exact-device correctness and diagnostic timing for the compiler-owned public
tracer -> Graph IR -> Schedule -> Tile -> native package route on gfx1201.
The intermediate is materialized as BF16; accumulation and output are FP32.

- Shape: M=128, K=256, N=256.
- Latest focused Tajasaurus suite: 128/128 tests passed across resident,
  scheduled gfx1201, and shape-rule registry coverage.
- Separate 100-iteration event medians after 25 warmups: RMSNorm producer
  11.479 us; matmul consumer 12.820 us.
- The independent oracle checked the RMSNorm result and matmul output.
- The packet records `worktree_dirty=true`; these timings are diagnostic and
  do not support performance promotion.

Reproduce on the gfx1201 host with the repository ROCm toolchain and GPU
visible, then run:

```sh
TESSERA_GFX1201_DEVICE_PROOF=1 python -m pytest -q \
  tests/unit/test_rocm_resident_rmsnorm_matmul.py \
  tests/unit/test_rocm_gfx1201_scheduled.py \
  tests/unit/test_shape_rule_registry.py
python benchmarks/rocm/benchmark_gfx1201_resident_norm_matmul.py \
  --dtype bf16 --warmup 25 --iterations 100
python benchmarks/rocm/benchmark_gfx1201_resident_norm_matmul.py \
  --dtype bf16 --dynamic-n --warmup 25 --iterations 100
```

The test and benchmark prove one BF16 resident edge envelope only. A second
correctness-gated packet, `dynamic_n.json`, reused one bounded-N package at
active N=128 and N=256 (bound 256), with identical input, RHS, intermediate,
and output allocations. It recorded producer medians of 8.52/11.36 us and
consumer medians of 10.68/14.04 us respectively. The refreshed packet was
measured after synchronizing the exact source file from the feature branch
to Tajasaurus and rerunning the 128-test gfx1201 suite. Both trials checked
numerical output. The packet was captured from a dirty worktree; timings are diagnostic
and do not establish a performance promotion.

The envelope still has static M/K and one row-major storage contract. These
results do not establish general BF16 matmul support or transfer evidence to
gfx1151, sm_120, Apple, or x86.
