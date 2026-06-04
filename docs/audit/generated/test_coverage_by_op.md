# Test Coverage by Op Family

Generated from `python/tessera/compiler/test_coverage_audit.py`.  Don't edit by hand — regenerate via `python -c "from tessera.compiler.test_coverage_audit import render_dashboard; open('docs/audit/generated/test_coverage_by_op.md', 'w').write(render_dashboard())"`.  Drift gated by `tests/unit/test_test_coverage_audit.py`.

**Honest scope note:** this audit measures *reference counts*, not numerical coverage quality.  A single test that exercises an op across 5 shapes × 3 dtypes counts as one reference but covers more ground than 5 happy-path tests.  Use the thin-coverage list as a starting point for triage, not a hard verdict.

## Headline

- **434** ops in `primitive_coverage` registry.
- **1647** total Python-test references, **469** total lit-fixture references.
- **121** ops have **zero** references in either test surface.
- **229** ops have ≤1 reference ("thinly tested").
- **32** ops have ≥10 references ("well tested").
- **47** ops have at least one associated `pytest.raises` negative test.

## Top 20 most-tested ops

| Op | py refs | lit refs | total | neg | dtypes |
|----|--------:|---------:|------:|----:|--------|
| `matmul` |  205 |  141 |  346 |  10 | `bf16`, `f16`, `f32`, `fp16` … |
| `flash_attn` |   74 |   44 |  118 |   3 | `bf16`, `f32`, `fp16`, `fp32` … |
| `softmax` |   61 |   35 |   96 |   3 | `bf16`, `f16`, `f32`, `fp16` … |
| `relu` |   73 |    3 |   76 |   5 | `bf16`, `f16`, `f32`, `fp32` |
| `gemm` |   69 |    2 |   71 |   6 | `bf16`, `f16`, `f32`, `fp16` … |
| `reduce` |   66 |    0 |   66 |   6 | `f32`, `fp16`, `fp32`, `fp4_e2m1` … |
| `mul` |   57 |    0 |   57 |   6 | `fp16`, `fp32`, `fp4_e2m1`, `fp6_e2m3` … |
| `attn_local_window_2d` |   31 |   25 |   56 |   1 | `fp32` |
| `add` |   44 |    2 |   46 |   4 | `bf16`, `f16`, `f32`, `fp32` |
| `cholesky` |   14 |   30 |   44 |   0 | `bf16`, `f16`, `f32`, `fp16` … |
| `rmsnorm` |   32 |    4 |   36 |   1 | `bf16`, `fp32` |
| `linear_attn` |   24 |    8 |   32 |   1 |  |
| `gelu` |   20 |   10 |   30 |   1 | `bf16`, `f16`, `f32`, `fp16` … |
| `cast` |    4 |   22 |   26 |   0 | `fp32` |
| `rope` |   16 |   10 |   26 |   0 |  |
| `transpose` |   12 |   13 |   25 |   0 | `fp32` |
| `selective_ssm` |   24 |    0 |   24 |   1 |  |
| `layer_norm` |   16 |    7 |   23 |   2 | `bf16`, `f16`, `f32`, `fp16` … |
| `silu` |   21 |    2 |   23 |   0 | `bf16`, `f16`, `f32`, `fp16` … |
| `dropout` |   17 |    5 |   22 |   3 | `f32`, `fp32`, `fp64` |

## Thinly-tested ops (≤1 reference)

These **229** ops have at most one test reference across the whole test surface.  Many will be legitimate — variant aliases, structural ops, or category rollups that inherit coverage from a parent family — but each one is a candidate for explicit per-op test coverage.

| Op | py refs | lit refs | total |
|----|--------:|---------:|------:|
| `abs` |    1 |    0 |    1 |
| `absolute` |    1 |    0 |    1 |
| `acos` |    1 |    0 |    1 |
| `alibi` |    1 |    0 |    1 |
| `aot_export` |    0 |    0 |    0 |
| `aot_load` |    0 |    0 |    0 |
| `asin` |    1 |    0 |    1 |
| `associative_scan` |    0 |    0 |    0 |
| `atan` |    1 |    0 |    1 |
| `atan2` |    1 |    0 |    1 |
| `autocast` |    0 |    0 |    0 |
| `axis_index` |    0 |    0 |    0 |
| `axis_name` |    0 |    0 |    0 |
| `axis_size` |    0 |    0 |    0 |
| `binary_cross_entropy_loss` |    1 |    0 |    1 |
| `bitwise_and` |    1 |    0 |    1 |
| `bitwise_not` |    1 |    0 |    1 |
| `bitwise_or` |    1 |    0 |    1 |
| `bitwise_xor` |    1 |    0 |    1 |
| `broadcast` |    1 |    0 |    1 |
| `calibration_observer` |    0 |    0 |    0 |
| `ceil` |    1 |    0 |    1 |
| `centralize_grad` |    1 |    0 |    1 |
| `chained_schedule` |    1 |    0 |    1 |
| `check_cauchy_riemann` |    0 |    0 |    0 |
| `checkpoint` |    0 |    0 |    0 |
| `chunk` |    1 |    0 |    1 |
| `clamp` |    1 |    0 |    1 |
| `clifford_codiff` |    0 |    0 |    0 |
| `clifford_conjugate` |    0 |    0 |    0 |
| `clifford_exp` |    0 |    0 |    0 |
| `clifford_ext_deriv` |    0 |    0 |    0 |
| `clifford_geometric_product` |    1 |    0 |    1 |
| `clifford_grade_involution` |    0 |    0 |    0 |
| `clifford_grade_projection` |    0 |    0 |    0 |
| `clifford_hodge_star` |    0 |    0 |    0 |
| `clifford_inner` |    0 |    0 |    0 |
| `clifford_integral` |    0 |    0 |    0 |
| `clifford_left_contraction` |    0 |    0 |    0 |
| `clifford_log` |    0 |    0 |    0 |
| `clifford_norm` |    0 |    0 |    0 |
| `clifford_reverse` |    0 |    0 |    0 |
| `clifford_rotor_sandwich` |    1 |    0 |    1 |
| `clifford_vec_deriv` |    0 |    0 |    0 |
| `clifford_wedge` |    0 |    0 |    0 |
| `complex_arg` |    0 |    0 |    0 |
| `complex_div` |    1 |    0 |    1 |
| `complex_log` |    0 |    0 |    0 |
| `complex_pow` |    0 |    0 |    0 |
| `complex_sqrt` |    0 |    0 |    0 |
| `cond` |    0 |    0 |    0 |
| `conformal_jacobian` |    0 |    0 |    0 |
| `contrastive_divergence_loss` |    0 |    0 |    0 |
| `cos` |    1 |    0 |    1 |
| `cosh` |    1 |    0 |    1 |
| `cosine_warmup_lr` |    1 |    0 |    1 |
| `cross_entropy_loss` |    1 |    0 |    1 |
| `cross_ratio` |    0 |    0 |    0 |
| `cummax` |    1 |    0 |    1 |
| `cummin` |    1 |    0 |    1 |

_(169 additional thinly-tested ops omitted; see `collect_op_test_coverage()` for the full list.)_
