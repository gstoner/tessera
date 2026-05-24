# Test Coverage by Op Family

Generated from `python/tessera/compiler/test_coverage_audit.py`.  Don't edit by hand — regenerate via `python -c "from tessera.compiler.test_coverage_audit import render_dashboard; open('docs/audit/generated/test_coverage_by_op.md', 'w').write(render_dashboard())"`.  Drift gated by `tests/unit/test_test_coverage_audit.py`.

**Honest scope note:** this audit measures *reference counts*, not numerical coverage quality.  A single test that exercises an op across 5 shapes × 3 dtypes counts as one reference but covers more ground than 5 happy-path tests.  Use the thin-coverage list as a starting point for triage, not a hard verdict.

## Headline

- **432** ops in `primitive_coverage` registry.
- **1989** total Python-test references, **401** total lit-fixture references.
- **237** ops have **zero** references in either test surface.
- **241** ops have ≤1 reference ("thinly tested").
- **41** ops have ≥10 references ("well tested").
- **37** ops have at least one associated `pytest.raises` negative test.

## Top 20 most-tested ops

| Op | py refs | lit refs | total | neg | dtypes |
|----|--------:|---------:|------:|----:|--------|
| `matmul` |  289 |  136 |  425 |  18 | `bf16`, `f16`, `f32`, `fp16` … |
| `flash_attn` |   99 |   42 |  141 |   4 | `bf16`, `f32`, `fp16`, `fp32` … |
| `gemm` |  129 |    2 |  131 |  16 | `bf16`, `f16`, `f32`, `fp16` … |
| `reduce` |  128 |    0 |  128 |  12 | `f32`, `fp16`, `fp32`, `fp4_e2m1` … |
| `softmax` |   86 |   35 |  121 |   2 | `fp16`, `fp32`, `fp4_e2m1`, `fp6_e2m3` … |
| `mul` |  106 |    0 |  106 |   9 | `fp16`, `fp32`, `fp4_e2m1`, `fp6_e2m3` … |
| `attn_local_window_2d` |   49 |   25 |   74 |   4 | `fp32` |
| `add` |   72 |    0 |   72 |  12 | `fp32` |
| `linear_attn` |   48 |    8 |   56 |   5 |  |
| `relu` |   50 |    2 |   52 |   7 | `f32`, `fp32` |
| `selective_ssm` |   48 |    0 |   48 |   2 |  |
| `dropout` |   33 |    5 |   38 |   6 | `f32`, `fp32`, `fp64` |
| `gelu` |   28 |   10 |   38 |   1 | `fp32` |
| `rope` |   27 |   10 |   37 |   0 |  |
| `transpose` |   22 |   13 |   35 |   0 | `fp32` |
| `layer_norm` |   27 |    5 |   32 |   4 | `bf16`, `f16`, `f32`, `fp16` … |
| `cast` |    6 |   22 |   28 |   0 | `fp32` |
| `quantize_fp8` |   23 |    0 |   23 |   3 | `fp16`, `fp4_e2m1`, `fp6_e2m3`, `fp8_e4m3` … |
| `depthwise_conv1d` |   20 |    0 |   20 |   2 |  |
| `silu_mul` |   10 |    8 |   18 |   0 | `fp32` |

## Thinly-tested ops (≤1 reference)

These **241** ops have at most one test reference across the whole test surface.  Many will be legitimate — variant aliases, structural ops, or category rollups that inherit coverage from a parent family — but each one is a candidate for explicit per-op test coverage.

| Op | py refs | lit refs | total |
|----|--------:|---------:|------:|
| `adafactor` |    0 |    0 |    0 |
| `adamw` |    1 |    0 |    1 |
| `adaptive_pool` |    0 |    0 |    0 |
| `add_decoupled_weight_decay` |    0 |    0 |    0 |
| `alibi` |    0 |    0 |    0 |
| `aot_export` |    0 |    0 |    0 |
| `aot_load` |    0 |    0 |    0 |
| `associative_scan` |    0 |    0 |    0 |
| `autocast` |    0 |    0 |    0 |
| `avg_pool` |    0 |    0 |    0 |
| `axis_index` |    0 |    0 |    0 |
| `axis_name` |    0 |    0 |    0 |
| `axis_size` |    0 |    0 |    0 |
| `bidirectional_scan` |    0 |    0 |    0 |
| `binary_cross_entropy_loss` |    0 |    0 |    0 |
| `broadcast_to_axis` |    0 |    0 |    0 |
| `calibration_observer` |    0 |    0 |    0 |
| `centralize_grad` |    0 |    0 |    0 |
| `chained_schedule` |    0 |    0 |    0 |
| `check_cauchy_riemann` |    0 |    0 |    0 |
| `checkpoint` |    0 |    0 |    0 |
| `cispo_policy_loss` |    0 |    0 |    0 |
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
| `clip_grad_norm` |    0 |    0 |    0 |
| `clip_grad_value` |    0 |    0 |    0 |
| `collective_permute` |    0 |    0 |    0 |
| `compilation_cache` |    0 |    0 |    0 |
| `complex_abs` |    0 |    0 |    0 |
| `complex_arg` |    0 |    0 |    0 |
| `complex_conjugate` |    0 |    0 |    0 |
| `complex_div` |    0 |    0 |    0 |
| `complex_exp` |    0 |    0 |    0 |
| `complex_log` |    0 |    0 |    0 |
| `complex_pow` |    0 |    0 |    0 |
| `complex_sqrt` |    0 |    0 |    0 |
| `cond` |    0 |    0 |    0 |
| `conformal_energy_on_sphere` |    0 |    0 |    0 |
| `conformal_jacobian` |    0 |    0 |    0 |
| `constant_lr` |    0 |    0 |    0 |
| `contrastive_divergence_loss` |    0 |    0 |    0 |
| `contrastive_loss` |    0 |    0 |    0 |
| `conv1d` |    0 |    0 |    0 |
| `conv3d` |    0 |    0 |    0 |
| `conv_transpose` |    0 |    0 |    0 |

_(181 additional thinly-tested ops omitted; see `collect_op_test_coverage()` for the full list.)_
