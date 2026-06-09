# Test Coverage by Op Family

Generated from `python/tessera/compiler/test_coverage_audit.py`.  Don't edit by hand — regenerate via `python -c "from tessera.compiler.test_coverage_audit import render_dashboard; open('docs/audit/generated/test_coverage_by_op.md', 'w').write(render_dashboard())"`.  Drift gated by `tests/unit/test_test_coverage_audit.py`.

**Honest scope note:** this audit measures *reference counts*, not numerical coverage quality.  A single test that exercises an op across 5 shapes × 3 dtypes counts as one reference but covers more ground than 5 happy-path tests.  Use the thin-coverage list as a starting point for triage, not a hard verdict.

## Headline

- **444** ops in `primitive_coverage` registry.
- **1960** total Python-test references, **500** total lit-fixture references.
- **104** ops have **zero** references in either test surface.
- **214** ops have ≤1 reference ("thinly tested").
- **39** ops have ≥10 references ("well tested").
- **52** ops have at least one associated `pytest.raises` negative test.

## Top 20 most-tested ops

| Op | py refs | lit refs | total | neg | dtypes |
|----|--------:|---------:|------:|----:|--------|
| `matmul` |  268 |  137 |  405 |  14 | `bf16`, `f16`, `f32`, `fp16` … |
| `flash_attn` |   78 |   40 |  118 |   4 | `bf16`, `f32`, `fp16`, `fp32` … |
| `softmax` |   66 |   35 |  101 |   3 | `bf16`, `f16`, `f32`, `fp16` … |
| `relu` |   88 |    3 |   91 |   8 | `bf16`, `f16`, `f32`, `fp32` |
| `gemm` |   68 |    2 |   70 |   6 | `bf16`, `f16`, `f32`, `fp16` … |
| `reduce` |   66 |    0 |   66 |   6 | `f32`, `fp16`, `fp32`, `fp4_e2m1` … |
| `silu` |   63 |    2 |   65 |   4 | `bf16`, `f16`, `f32`, `fp16` … |
| `mul` |   62 |    0 |   62 |   6 | `fp16`, `fp32`, `fp4_e2m1`, `fp6_e2m3` … |
| `add` |   57 |    2 |   59 |   7 | `bf16`, `f16`, `f32`, `fp32` |
| `attn_local_window_2d` |   31 |   25 |   56 |   1 | `fp32` |
| `rmsnorm` |   41 |    9 |   50 |   1 | `bf16`, `fp32` |
| `cholesky` |   14 |   30 |   44 |   0 | `bf16`, `f16`, `f32`, `fp16` … |
| `selective_ssm` |   35 |    0 |   35 |   2 |  |
| `gelu` |   22 |   10 |   32 |   1 | `bf16`, `f16`, `f32`, `fp16` … |
| `linear_attn` |   24 |    8 |   32 |   1 |  |
| `cast` |    4 |   22 |   26 |   0 | `fp32` |
| `rope` |   16 |   10 |   26 |   0 |  |
| `transpose` |   12 |   13 |   25 |   0 | `fp32` |
| `dropout` |   18 |    5 |   23 |   3 | `f32`, `fp32`, `fp64` |
| `layer_norm` |   16 |    7 |   23 |   2 | `bf16`, `f16`, `f32`, `fp16` … |

## Thinly-tested ops (≤1 reference)

These **214** ops have at most one test reference across the whole test surface.  Many will be legitimate — variant aliases, structural ops, or category rollups that inherit coverage from a parent family — but each one is a candidate for explicit per-op test coverage.

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
| `clifford_conjugate` |    1 |    0 |    1 |
| `clifford_exp` |    0 |    0 |    0 |
| `clifford_ext_deriv` |    0 |    0 |    0 |
| `clifford_grade_involution` |    1 |    0 |    1 |
| `clifford_hodge_star` |    0 |    0 |    0 |
| `clifford_inner` |    1 |    0 |    1 |
| `clifford_integral` |    0 |    0 |    0 |
| `clifford_left_contraction` |    1 |    0 |    1 |
| `clifford_log` |    0 |    0 |    0 |
| `clifford_norm_squared` |    1 |    0 |    1 |
| `clifford_reverse` |    1 |    0 |    1 |
| `clifford_vec_deriv` |    0 |    0 |    0 |
| `clifford_wedge` |    1 |    0 |    1 |
| `complex_arg` |    0 |    0 |    0 |
| `complex_div` |    1 |    0 |    1 |
| `complex_log` |    0 |    0 |    0 |
| `complex_pow` |    0 |    0 |    0 |
| `complex_sqrt` |    0 |    0 |    0 |
| `conformal_jacobian` |    0 |    0 |    0 |
| `cosh` |    1 |    0 |    1 |
| `cosine_warmup_lr` |    1 |    0 |    1 |
| `cross_entropy_loss` |    1 |    0 |    1 |
| `cross_ratio` |    0 |    0 |    0 |
| `custom_batching` |    0 |    0 |    0 |
| `custom_call` |    0 |    0 |    0 |
| `custom_jvp` |    0 |    0 |    0 |
| `custom_lowering` |    0 |    0 |    0 |
| `custom_primitive` |    0 |    0 |    0 |
| `custom_vjp` |    0 |    0 |    0 |
| `cyclical_lr` |    1 |    0 |    1 |
| `dataset_batch` |    0 |    0 |    0 |
| `dataset_checkpoint` |    0 |    0 |    0 |

_(154 additional thinly-tested ops omitted; see `collect_op_test_coverage()` for the full list.)_

---

## Test Coverage Classification — Thinly-Tested Ops

Generated from `python/tessera/compiler/coverage_classification.py`.  Don't edit by hand — regenerate via `python -c "from tessera.compiler.coverage_classification import write_dashboard; write_dashboard()"`.  Drift gated by `tests/unit/test_coverage_classification.py`.

Companion to `test_coverage_by_op.md`.  That dashboard says **which** ops are thinly tested; this one says **why** and **what to do about it**.

## Headline

**214** ops have ≤1 direct test reference.  They break down as:

| Bucket | Count | Meaning |
|--------|------:|---------|
| `covered_by_family`      |   85 | Tested via a parent op or family wrapper |
| `structural_only`        |  125 | Registry/metadata/wrapper; no direct numerical test meaningful |
| `needs_direct_test`      |    0 | **Actionable test debt** — real primitive without direct test |
| `hardware_gated`         |    4 | Blocked on real device hardware (Phase G/H/I) |
| `deprecated_or_internal` |    0 | Not public test debt |

## Actionable: `needs_direct_test` ops

These **0** ops are real primitives with ≤1 direct test reference.  Each is a candidate for a focused numerical-correctness test.

| Op | py refs | lit refs | reason |
|----|--------:|---------:|--------|

## Hardware-gated ops

These **4** ops need real device hardware (Phase G/H/I).  They cannot be tested with execute-and-compare on this Mac.

| Op | reason |
|----|--------|
| `ebm_bivector_langevin_sample` | manifold Langevin needs real GPU mesh (Phase G) |
| `ebm_bivector_langevin_step` | manifold Langevin needs real GPU mesh (Phase G) |
| `ebm_sphere_langevin_sample` | manifold Langevin needs real GPU mesh (Phase G) |
| `ebm_sphere_langevin_step` | manifold Langevin needs real GPU mesh (Phase G) |

## `covered_by_family` — 85 ops

Tested through a parent op or family wrapper.  Sample (first 30):

| Op | reason |
|----|--------|
| `acos` | category default for 'elementwise' |
| `alibi` | tested via attention_family_support attention paths |
| `asin` | category default for 'elementwise' |
| `atan` | category default for 'elementwise' |
| `binary_cross_entropy_loss` | category default for 'loss' |
| `check_cauchy_riemann` | exercised by complex_jit / CR conformance tests |
| `clifford_codiff` | category default for 'geometric_algebra' |
| `clifford_conjugate` | category default for 'geometric_algebra' |
| `clifford_exp` | category default for 'geometric_algebra' |
| `clifford_ext_deriv` | category default for 'geometric_algebra' |
| `clifford_grade_involution` | category default for 'geometric_algebra' |
| `clifford_hodge_star` | category default for 'geometric_algebra' |
| `clifford_inner` | category default for 'geometric_algebra' |
| `clifford_integral` | category default for 'geometric_algebra' |
| `clifford_left_contraction` | category default for 'geometric_algebra' |
| `clifford_log` | category default for 'geometric_algebra' |
| `clifford_reverse` | category default for 'geometric_algebra' |
| `clifford_vec_deriv` | category default for 'geometric_algebra' |
| `clifford_wedge` | category default for 'geometric_algebra' |
| `complex_arg` | category default for 'elementwise' |
| `complex_div` | category default for 'elementwise' |
| `complex_log` | category default for 'elementwise' |
| `complex_pow` | category default for 'elementwise' |
| `complex_sqrt` | category default for 'elementwise' |
| `conformal_jacobian` | exercised by complex/conformal lane tests |
| `cosh` | category default for 'elementwise' |
| `cross_entropy_loss` | category default for 'loss' |
| `cross_ratio` | category default for 'elementwise' |
| `dbar` | exercised by complex differential tests |
| `ddpm_noise_pred_loss` | category default for 'loss' |

_(55 additional family-covered ops omitted; see `classify_thinly_tested()` for the full list.)_

## `structural_only` — 125 ops

Registry/metadata/wrapper ops; direct numerical tests not meaningful.  Sample (first 30):

| Op | reason |
|----|--------|
| `abs` | unclassified — defaults to structural_only |
| `absolute` | unclassified — defaults to structural_only |
| `aot_export` | category default for 'aot' |
| `aot_load` | category default for 'aot' |
| `associative_scan` | category default for 'control_flow' |
| `autocast` | category default for 'transform' |
| `axis_index` | category default for 'transform' |
| `axis_name` | category default for 'transform' |
| `axis_size` | category default for 'transform' |
| `bitwise_and` | unclassified — defaults to structural_only |
| `bitwise_not` | unclassified — defaults to structural_only |
| `bitwise_or` | unclassified — defaults to structural_only |
| `bitwise_xor` | unclassified — defaults to structural_only |
| `broadcast` | unclassified — defaults to structural_only |
| `calibration_observer` | stateful observer; tested via fake_quantize loop |
| `ceil` | unclassified — defaults to structural_only |
| `centralize_grad` | category default for 'grad_transform' |
| `chained_schedule` | category default for 'schedule' |
| `checkpoint` | category default for 'transform' |
| `chunk` | unclassified — defaults to structural_only |
| `clamp` | unclassified — defaults to structural_only |
| `clifford_norm_squared` | unclassified — defaults to structural_only |
| `cosine_warmup_lr` | category default for 'schedule' |
| `custom_batching` | category default for 'extension' |
| `custom_call` | category default for 'extension' |
| `custom_jvp` | category default for 'extension' |
| `custom_lowering` | category default for 'extension' |
| `custom_primitive` | category default for 'extension' |
| `custom_vjp` | category default for 'extension' |
| `cyclical_lr` | category default for 'schedule' |

_(95 additional structural ops omitted.)_
