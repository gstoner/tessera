# Test Coverage by Op Family

Generated from `python/tessera/compiler/test_coverage_audit.py`.  Don't edit by hand — regenerate via `python -c "from tessera.compiler.test_coverage_audit import render_dashboard; open('docs/audit/generated/test_coverage_by_op.md', 'w').write(render_dashboard())"`.  Drift gated by `tests/unit/test_test_coverage_audit.py`.

**Honest scope note:** this audit measures *reference counts*, not numerical coverage quality.  A single test that exercises an op across 5 shapes × 3 dtypes counts as one reference but covers more ground than 5 happy-path tests.  Use the thin-coverage list as a starting point for triage, not a hard verdict.

## Headline

- **432** ops in `primitive_coverage` registry.
- **1265** total Python-test references, **387** total lit-fixture references.
- **137** ops have **zero** references in either test surface.
- **291** ops have ≤1 reference ("thinly tested").
- **24** ops have ≥10 references ("well tested").
- **42** ops have at least one associated `pytest.raises` negative test.

## Top 20 most-tested ops

| Op | py refs | lit refs | total | neg | dtypes |
|----|--------:|---------:|------:|----:|--------|
| `matmul` |  166 |  136 |  302 |   9 | `bf16`, `f16`, `f32`, `fp16` … |
| `flash_attn` |   57 |   42 |   99 |   2 | `bf16`, `f32`, `fp16`, `fp32` … |
| `softmax` |   46 |   35 |   81 |   2 | `fp16`, `fp32`, `fp4_e2m1`, `fp6_e2m3` … |
| `gemm` |   69 |    2 |   71 |   6 | `bf16`, `f16`, `f32`, `fp16` … |
| `reduce` |   64 |    0 |   64 |   6 | `f32`, `fp16`, `fp32`, `fp4_e2m1` … |
| `attn_local_window_2d` |   31 |   25 |   56 |   1 | `fp32` |
| `mul` |   53 |    0 |   53 |   6 | `fp16`, `fp32`, `fp4_e2m1`, `fp6_e2m3` … |
| `add` |   36 |    0 |   36 |   4 | `fp32` |
| `linear_attn` |   24 |    8 |   32 |   1 |  |
| `relu` |   29 |    2 |   31 |   4 | `f32`, `fp32` |
| `cast` |    4 |   22 |   26 |   0 | `fp32` |
| `gelu` |   16 |   10 |   26 |   1 | `fp32` |
| `transpose` |   12 |   13 |   25 |   0 | `fp32` |
| `rope` |   14 |   10 |   24 |   0 |  |
| `selective_ssm` |   24 |    0 |   24 |   1 |  |
| `dropout` |   17 |    5 |   22 |   3 | `f32`, `fp32`, `fp64` |
| `layer_norm` |   14 |    5 |   19 |   2 | `bf16`, `f16`, `f32`, `fp16` … |
| `latent_kv_compress` |    4 |    9 |   13 |   0 | `fp16`, `fp32` |
| `load_state` |   13 |    0 |   13 |   2 |  |
| `silu_mul` |    5 |    8 |   13 |   0 | `fp32` |

## Thinly-tested ops (≤1 reference)

These **291** ops have at most one test reference across the whole test surface.  Many will be legitimate — variant aliases, structural ops, or category rollups that inherit coverage from a parent family — but each one is a candidate for explicit per-op test coverage.

| Op | py refs | lit refs | total |
|----|--------:|---------:|------:|
| `abs` |    1 |    0 |    1 |
| `absolute` |    1 |    0 |    1 |
| `acos` |    1 |    0 |    1 |
| `adaptive_pool` |    1 |    0 |    1 |
| `alibi` |    1 |    0 |    1 |
| `amax` |    1 |    0 |    1 |
| `amin` |    1 |    0 |    1 |
| `aot_export` |    0 |    0 |    0 |
| `aot_load` |    0 |    0 |    0 |
| `argmax` |    1 |    0 |    1 |
| `argmin` |    1 |    0 |    1 |
| `asin` |    1 |    0 |    1 |
| `associative_scan` |    0 |    0 |    0 |
| `atan` |    1 |    0 |    1 |
| `atan2` |    1 |    0 |    1 |
| `autocast` |    0 |    0 |    0 |
| `avg_pool` |    0 |    0 |    0 |
| `axis_index` |    0 |    0 |    0 |
| `axis_name` |    0 |    0 |    0 |
| `axis_size` |    0 |    0 |    0 |
| `bidirectional_scan` |    1 |    0 |    1 |
| `binary_cross_entropy_loss` |    1 |    0 |    1 |
| `bitwise_and` |    1 |    0 |    1 |
| `bitwise_not` |    1 |    0 |    1 |
| `bitwise_or` |    1 |    0 |    1 |
| `bitwise_xor` |    1 |    0 |    1 |
| `broadcast` |    1 |    0 |    1 |
| `broadcast_to_axis` |    0 |    0 |    0 |
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
| `collective_permute` |    1 |    0 |    1 |
| `complex_abs` |    0 |    0 |    0 |
| `complex_arg` |    0 |    0 |    0 |
| `complex_conjugate` |    0 |    0 |    0 |
| `complex_div` |    0 |    0 |    0 |
| `complex_exp` |    0 |    0 |    0 |
| `complex_log` |    0 |    0 |    0 |

_(231 additional thinly-tested ops omitted; see `collect_op_test_coverage()` for the full list.)_
