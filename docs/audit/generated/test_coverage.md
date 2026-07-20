# Test Coverage by Op Family

Generated from `python/tessera/compiler/test_coverage_audit.py`.  Don't edit by hand — regenerate via `python -m tessera.compiler.generated_docs --write test_coverage`.  Drift gated by `tests/unit/test_generated_docs_registry.py` and `tests/unit/test_test_coverage_audit.py`.

**Honest scope note:** this audit measures *reference counts*, not numerical coverage quality.  A single test that exercises an op across 5 shapes × 3 dtypes counts as one reference but covers more ground than 5 happy-path tests.  Use the thin-coverage list as a starting point for triage, not a hard verdict.

## Headline

- **480** ops in `primitive_coverage` registry.
- **4346** total Python-test references, **1000** total lit-fixture references.
- **81** ops have **zero** references in either test surface.
- **117** ops have ≤1 reference ("thinly tested").
- **146** ops have ≥10 references ("well tested").
- **127** ops have at least one associated `pytest.raises` negative test.

## Top 20 most-tested ops

| Op | py refs | lit refs | total | neg | dtypes |
|----|--------:|---------:|------:|----:|--------|
| `matmul` |  435 |  230 |  665 |  19 | `bf16`, `f16`, `f32`, `f64` … |
| `softmax` |  133 |   38 |  171 |  31 | `bf16`, `f16`, `f32`, `f64` … |
| `flash_attn` |  112 |   57 |  169 |   9 | `bf16`, `f16`, `f32`, `f64` … |
| `add` |  118 |   29 |  147 |  10 | `bf16`, `f16`, `f32`, `f64` … |
| `relu` |  109 |   34 |  143 |   9 | `bf16`, `f16`, `f32`, `f64` … |
| `mul` |   90 |    9 |   99 |   7 | `bf16`, `f16`, `f32`, `f64` … |
| `reduce` |   89 |    2 |   91 |   7 | `bf16`, `f16`, `f32`, `f64` … |
| `silu` |   85 |    6 |   91 |   5 | `bf16`, `f16`, `f32`, `f64` … |
| `rmsnorm` |   76 |   13 |   89 |   4 | `bf16`, `f16`, `f32`, `f64` … |
| `gemm` |   79 |    2 |   81 |   8 | `bf16`, `f16`, `f32`, `fp16` … |
| `selective_ssm` |   71 |   10 |   81 |   3 | `bf16`, `f16`, `fp16`, `fp32` |
| `gelu` |   46 |   19 |   65 |   0 | `bf16`, `f16`, `f32`, `f64` … |
| `transpose` |   27 |   35 |   62 |   0 | `bf16`, `f16`, `fp16`, `fp32` |
| `attn_local_window_2d` |   36 |   25 |   61 |   1 | `fp16`, `fp32` |
| `msa_sparse_attention` |   47 |   11 |   58 |   1 |  |
| `cast` |   16 |   40 |   56 |   1 | `fp16`, `fp32` |
| `grouped_gemm` |   31 |   24 |   55 |   2 | `fp32`, `fp4_e2m1`, `fp8_e4m3`, `fp8_e5m2` … |
| `linear_attn` |   44 |    8 |   52 |   2 |  |
| `cholesky` |   20 |   30 |   50 |   0 | `bf16`, `f16`, `f32`, `fp16` … |
| `layer_norm` |   26 |    9 |   35 |   3 | `bf16`, `f16`, `f32`, `fp16` … |

## Thinly-tested ops (≤1 reference)

These **117** ops have at most one test reference across the whole test surface.  Many will be legitimate — variant aliases, structural ops, or category rollups that inherit coverage from a parent family — but each one is a candidate for explicit per-op test coverage.

| Op | py refs | lit refs | total |
|----|--------:|---------:|------:|
| `aot_export` |    0 |    0 |    0 |
| `aot_load` |    0 |    0 |    0 |
| `associative_scan` |    0 |    0 |    0 |
| `autocast` |    0 |    0 |    0 |
| `axis_index` |    0 |    0 |    0 |
| `axis_name` |    0 |    0 |    0 |
| `axis_size` |    0 |    0 |    0 |
| `calibration_observer` |    0 |    0 |    0 |
| `centralize_grad` |    1 |    0 |    1 |
| `chained_schedule` |    1 |    0 |    1 |
| `check_cauchy_riemann` |    0 |    0 |    0 |
| `checkpoint` |    0 |    0 |    0 |
| `chunk` |    1 |    0 |    1 |
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
| `conformal_jacobian` |    0 |    0 |    0 |
| `cosine_warmup_lr` |    1 |    0 |    1 |
| `custom_batching` |    0 |    0 |    0 |
| `custom_call` |    0 |    0 |    0 |
| `custom_jvp` |    0 |    0 |    0 |
| `custom_lowering` |    0 |    0 |    0 |
| `custom_primitive` |    0 |    0 |    0 |
| `custom_vjp` |    0 |    0 |    0 |
| `cyclical_lr` |    1 |    0 |    1 |
| `dataset_batch` |    0 |    0 |    0 |
| `dataset_checkpoint` |    0 |    0 |    0 |
| `dataset_filter` |    0 |    0 |    0 |
| `dataset_interleave` |    0 |    0 |    0 |
| `dataset_map` |    0 |    0 |    0 |
| `dataset_prefetch` |    0 |    0 |    0 |
| `dataset_repeat` |    0 |    0 |    0 |
| `dataset_shuffle` |    0 |    0 |    0 |
| `dataset_zip` |    0 |    0 |    0 |
| `dbar` |    0 |    0 |    0 |
| `dynamic_slice` |    1 |    0 |    1 |
| `dynamic_update_slice` |    1 |    0 |    1 |
| `dz` |    0 |    0 |    0 |
| `ebm_bivector_langevin_sample` |    0 |    0 |    0 |
| `ebm_bivector_langevin_step` |    0 |    0 |    0 |
| `ebm_decode_init` |    0 |    0 |    0 |
| `ebm_energy` |    0 |    0 |    0 |
| `ebm_langevin_step` |    0 |    0 |    0 |
| `ebm_partition_ais` |    0 |    0 |    0 |
| `ebm_partition_exact` |    0 |    0 |    0 |
| `ebm_partition_monte_carlo` |    0 |    0 |    0 |
| `ebm_sphere_langevin_sample` |    0 |    0 |    0 |
| `ebm_sphere_langevin_step` |    0 |    0 |    0 |
| `ema_update` |    1 |    0 |    1 |

_(57 additional thinly-tested ops omitted; see `collect_op_test_coverage()` for the full list.)_

---

## Test Coverage Classification — Thinly-Tested Ops

Generated from `python/tessera/compiler/coverage_classification.py`.  Don't edit by hand — regenerate via `python -m tessera.compiler.generated_docs --write test_coverage`.  Drift gated by `tests/unit/test_generated_docs_registry.py` and `tests/unit/test_coverage_classification.py`.

Companion section to the by-op coverage table above: that section says **which** ops are thinly tested; this one says **why** and **what to do about it**.

## Headline

**117** ops have ≤1 direct test reference.  They break down as:

| Bucket | Count | Meaning |
|--------|------:|---------|
| `covered_by_family`      |   26 | Tested via a parent op or family wrapper |
| `structural_only`        |   87 | Registry/metadata/wrapper; no direct numerical test meaningful |
| `needs_direct_test`      |    0 | **Actionable test debt** — real primitive without direct test |
| `hardware_gated`         |    4 | Blocked on real device hardware (Phase G/H) |
| `deprecated_or_internal` |    0 | Not public test debt |

## Actionable: `needs_direct_test` ops

These **0** ops are real primitives with ≤1 direct test reference.  Each is a candidate for a focused numerical-correctness test.

| Op | py refs | lit refs | reason |
|----|--------:|---------:|--------|

## Hardware-gated ops

These **4** ops need real device hardware (Phase G/H).  They cannot be tested with execute-and-compare on this Mac.

| Op | reason |
|----|--------|
| `ebm_bivector_langevin_sample` | manifold Langevin needs real GPU mesh (Phase G) |
| `ebm_bivector_langevin_step` | manifold Langevin needs real GPU mesh (Phase G) |
| `ebm_sphere_langevin_sample` | manifold Langevin needs real GPU mesh (Phase G) |
| `ebm_sphere_langevin_step` | manifold Langevin needs real GPU mesh (Phase G) |

## `covered_by_family` — 26 ops

Tested through a parent op or family wrapper.  Sample (first 30):

| Op | reason |
|----|--------|
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
| `conformal_jacobian` | exercised by complex/conformal lane tests |
| `dbar` | exercised by complex differential tests |
| `dz` | exercised by complex differential tests |
| `ebm_decode_init` | scaffold for ebm decode tests |
| `rng_clone` | category default for 'rng' |
| `rng_gibbs_sample` | category default for 'rng' |
| `rng_hmc_sample` | category default for 'rng' |
| `rng_key` | category default for 'rng' |
| `rng_langevin_sample` | category default for 'rng' |
| `rng_mala_sample` | category default for 'rng' |
| `rng_multinomial` | category default for 'rng' |
| `rng_permutation` | category default for 'rng' |

## `structural_only` — 87 ops

Registry/metadata/wrapper ops; direct numerical tests not meaningful.  Sample (first 30):

| Op | reason |
|----|--------|
| `aot_export` | category default for 'aot' |
| `aot_load` | category default for 'aot' |
| `associative_scan` | category default for 'control_flow' |
| `autocast` | category default for 'transform' |
| `axis_index` | category default for 'transform' |
| `axis_name` | category default for 'transform' |
| `axis_size` | category default for 'transform' |
| `calibration_observer` | stateful observer; tested via fake_quantize loop |
| `centralize_grad` | category default for 'grad_transform' |
| `chained_schedule` | category default for 'schedule' |
| `checkpoint` | category default for 'transform' |
| `chunk` | unclassified — defaults to structural_only |
| `clifford_norm_squared` | unclassified — defaults to structural_only |
| `cosine_warmup_lr` | category default for 'schedule' |
| `custom_batching` | category default for 'extension' |
| `custom_call` | category default for 'extension' |
| `custom_jvp` | category default for 'extension' |
| `custom_lowering` | category default for 'extension' |
| `custom_primitive` | category default for 'extension' |
| `custom_vjp` | category default for 'extension' |
| `cyclical_lr` | category default for 'schedule' |
| `dataset_batch` | category default for 'data' |
| `dataset_checkpoint` | category default for 'data' |
| `dataset_filter` | category default for 'data' |
| `dataset_interleave` | category default for 'data' |
| `dataset_map` | category default for 'data' |
| `dataset_prefetch` | category default for 'data' |
| `dataset_repeat` | category default for 'data' |
| `dataset_shuffle` | category default for 'data' |
| `dataset_zip` | category default for 'data' |

_(57 additional structural ops omitted.)_
