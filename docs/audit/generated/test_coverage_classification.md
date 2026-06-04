# Test Coverage Classification — Thinly-Tested Ops

Generated from `python/tessera/compiler/coverage_classification.py`.  Don't edit by hand — regenerate via `python -c "from tessera.compiler.coverage_classification import write_dashboard; write_dashboard()"`.  Drift gated by `tests/unit/test_coverage_classification.py`.

Companion to `test_coverage_by_op.md`.  That dashboard says **which** ops are thinly tested; this one says **why** and **what to do about it**.

## Headline

**230** ops have ≤1 direct test reference.  They break down as:

| Bucket | Count | Meaning |
|--------|------:|---------|
| `covered_by_family`      |   95 | Tested via a parent op or family wrapper |
| `structural_only`        |  131 | Registry/metadata/wrapper; no direct numerical test meaningful |
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

## `covered_by_family` — 95 ops

Tested through a parent op or family wrapper.  Sample (first 30):

| Op | reason |
|----|--------|
| `acos` | category default for 'elementwise' |
| `alibi` | tested via attention_family_support attention paths |
| `asin` | category default for 'elementwise' |
| `atan` | category default for 'elementwise' |
| `atan2` | category default for 'elementwise' |
| `binary_cross_entropy_loss` | category default for 'loss' |
| `check_cauchy_riemann` | exercised by complex_jit / CR conformance tests |
| `clifford_codiff` | category default for 'geometric_algebra' |
| `clifford_conjugate` | category default for 'geometric_algebra' |
| `clifford_exp` | category default for 'geometric_algebra' |
| `clifford_ext_deriv` | category default for 'geometric_algebra' |
| `clifford_geometric_product` | category default for 'geometric_algebra' |
| `clifford_grade_involution` | category default for 'geometric_algebra' |
| `clifford_grade_projection` | category default for 'geometric_algebra' |
| `clifford_hodge_star` | category default for 'geometric_algebra' |
| `clifford_inner` | category default for 'geometric_algebra' |
| `clifford_integral` | category default for 'geometric_algebra' |
| `clifford_left_contraction` | category default for 'geometric_algebra' |
| `clifford_log` | category default for 'geometric_algebra' |
| `clifford_norm` | category default for 'geometric_algebra' |
| `clifford_reverse` | category default for 'geometric_algebra' |
| `clifford_rotor_sandwich` | category default for 'geometric_algebra' |
| `clifford_vec_deriv` | category default for 'geometric_algebra' |
| `clifford_wedge` | category default for 'geometric_algebra' |
| `complex_arg` | category default for 'elementwise' |
| `complex_div` | category default for 'elementwise' |
| `complex_log` | category default for 'elementwise' |
| `complex_pow` | category default for 'elementwise' |
| `complex_sqrt` | category default for 'elementwise' |
| `conformal_jacobian` | exercised by complex/conformal lane tests |

_(65 additional family-covered ops omitted; see `classify_thinly_tested()` for the full list.)_

## `structural_only` — 131 ops

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
| `cond` | category default for 'control_flow' |
| `cosine_warmup_lr` | category default for 'schedule' |
| `cummax` | unclassified — defaults to structural_only |
| `cummin` | unclassified — defaults to structural_only |
| `custom_batching` | category default for 'extension' |
| `custom_call` | category default for 'extension' |
| `custom_jvp` | category default for 'extension' |
| `custom_lowering` | category default for 'extension' |
| `custom_primitive` | category default for 'extension' |

_(101 additional structural ops omitted.)_
