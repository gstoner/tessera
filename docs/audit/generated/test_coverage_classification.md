# Test Coverage Classification — Thinly-Tested Ops

Generated from `python/tessera/compiler/coverage_classification.py`.  Don't edit by hand — regenerate via `python -c "from tessera.compiler.coverage_classification import write_dashboard; write_dashboard()"`.  Drift gated by `tests/unit/test_coverage_classification.py`.

Companion to `test_coverage_by_op.md`.  That dashboard says **which** ops are thinly tested; this one says **why** and **what to do about it**.

## Headline

**291** ops have ≤1 direct test reference.  They break down as:

| Bucket | Count | Meaning |
|--------|------:|---------|
| `covered_by_family`      |   99 | Tested via a parent op or family wrapper |
| `structural_only`        |  140 | Registry/metadata/wrapper; no direct numerical test meaningful |
| `needs_direct_test`      |   48 | **Actionable test debt** — real primitive without direct test |
| `hardware_gated`         |    4 | Blocked on real device hardware (Phase G/H/I) |
| `deprecated_or_internal` |    0 | Not public test debt |

## Actionable: `needs_direct_test` ops

These **48** ops are real primitives with ≤1 direct test reference.  Each is a candidate for a focused numerical-correctness test.

| Op | py refs | lit refs | reason |
|----|--------:|---------:|--------|
| `adaptive_pool` |   1 |   0 | category default for 'pooling' |
| `avg_pool` |   0 |   0 | category default for 'pooling' |
| `bidirectional_scan` |   1 |   0 | category default for 'recurrent' |
| `broadcast_to_axis` |   0 |   0 | category default for 'collective' |
| `collective_permute` |   1 |   0 | category default for 'collective' |
| `conformal_energy_on_sphere` |   0 |   0 | category default for 'stable_reduction' |
| `conv3d` |   1 |   0 | category default for 'stencil' |
| `conv_transpose` |   0 |   0 | category default for 'model_layer' |
| `deepseek_sparse_attention` |   1 |   0 | category default for 'attention' |
| `dequantize_fp4` |   0 |   0 | category default for 'quantize' |
| `dequantize_fp6` |   1 |   0 | category default for 'quantize' |
| `dequantize_int4` |   1 |   0 | category default for 'quantization' |
| `dequantize_int8` |   1 |   0 | category default for 'quantization' |
| `dequantize_nvfp4` |   0 |   0 | category default for 'quantize' |
| `fake_quantize` |   1 |   0 | category default for 'quantization' |
| `gated_attention` |   1 |   0 | category default for 'attention' |
| `grad_scaler_step` |   1 |   0 | category default for 'numerics' |
| `grpo_policy_loss` |   1 |   0 | category default for 'rl_loss' |
| `gru_cell` |   0 |   0 | category default for 'recurrent' |
| `instance_norm` |   0 |   0 | category default for 'normalization' |
| `istft` |   1 |   0 | category default for 'spectral' |
| `laplacian_2d` |   0 |   0 | category default for 'stencil' |
| `log_softmax` |   1 |   0 | category default for 'stable_reduction' |
| `lora_linear` |   1 |   0 | category default for 'model_layer' |
| `max_pool` |   1 |   0 | category default for 'pooling' |
| `memory_write` |   1 |   0 | category default for 'memory' |
| `min_pool` |   0 |   0 | category default for 'pooling' |
| `modified_delta_attention` |   1 |   0 | category default for 'attention' |
| `moe_combine` |   1 |   0 | category default for 'moe_transport' |
| `pack` |   1 |   0 | category default for 'layout_transform' |
| `pmax` |   1 |   0 | category default for 'collective' |
| `pmean` |   1 |   0 | category default for 'collective' |
| `pmin` |   1 |   0 | category default for 'collective' |
| `psum` |   1 |   0 | category default for 'collective' |
| `qr` |   1 |   0 | category default for 'linalg_decomposition' |
| `quantize_int4` |   1 |   0 | category default for 'quantization' |
| `quantize_int8` |   1 |   0 | category default for 'quantization' |
| `rearrange` |   1 |   0 | category default for 'layout_transform' |
| `sigmoid_safe` |   1 |   0 | category default for 'stable_reduction' |
| `simple_rnn_cell` |   1 |   0 | category default for 'recurrent' |
| `spectral_filter` |   1 |   0 | category default for 'spectral' |
| `spectral_norm` |   0 |   0 | category default for 'normalization' |
| `spmm_coo` |   1 |   0 | category default for 'sparse' |
| `stft` |   1 |   0 | category default for 'spectral' |
| `svd` |   1 |   0 | category default for 'linalg_decomposition' |
| `tile_view` |   1 |   0 | category default for 'layout_transform' |
| `unpack` |   1 |   0 | category default for 'layout_transform' |
| `weight_norm` |   1 |   0 | category default for 'normalization' |

## Hardware-gated ops

These **4** ops need real device hardware (Phase G/H/I).  They cannot be tested with execute-and-compare on this Mac.

| Op | reason |
|----|--------|
| `ebm_bivector_langevin_sample` | manifold Langevin needs real GPU mesh (Phase G) |
| `ebm_bivector_langevin_step` | manifold Langevin needs real GPU mesh (Phase G) |
| `ebm_sphere_langevin_sample` | manifold Langevin needs real GPU mesh (Phase G) |
| `ebm_sphere_langevin_step` | manifold Langevin needs real GPU mesh (Phase G) |

## `covered_by_family` — 99 ops

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
| `complex_abs` | category default for 'elementwise' |
| `complex_arg` | category default for 'elementwise' |
| `complex_conjugate` | category default for 'elementwise' |
| `complex_div` | category default for 'elementwise' |
| `complex_exp` | category default for 'elementwise' |
| `complex_log` | category default for 'elementwise' |

_(69 additional family-covered ops omitted; see `classify_thinly_tested()` for the full list.)_

## `structural_only` — 140 ops

Registry/metadata/wrapper ops; direct numerical tests not meaningful.  Sample (first 30):

| Op | reason |
|----|--------|
| `abs` | unclassified — defaults to structural_only |
| `absolute` | unclassified — defaults to structural_only |
| `amax` | unclassified — defaults to structural_only |
| `amin` | unclassified — defaults to structural_only |
| `aot_export` | category default for 'aot' |
| `aot_load` | category default for 'aot' |
| `argmax` | unclassified — defaults to structural_only |
| `argmin` | unclassified — defaults to structural_only |
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
| `cumprod` | unclassified — defaults to structural_only |

_(110 additional structural ops omitted.)_
