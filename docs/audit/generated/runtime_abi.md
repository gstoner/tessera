# Runtime C ABI Surface Audit

Human-readable view. The canonical machine-readable artifact is `runtime_abi.csv` in this directory — that CSV is what the drift gate compares. Don't edit either by hand; run `python -m tessera.compiler.audit runtime_abi --write` (or `scripts/check_generated_docs.sh --write`) to refresh both. Drift gated by `tests/unit/test_runtime_abi_audit.py`.

## Headline

- **330** unique `extern "C" tessera_*` C ABI symbols across all backends.
- **6 / 6** core runtime headers present.
- **134** Apple GPU kernel families with per-dtype variants.

## Core runtime headers

| Header | Status |
|--------|--------|
| `src/runtime/include/tessera/tessera_runtime.h` | ✅ |
| `src/runtime/include/tessera/tsr_kernel.h` | ✅ |
| `src/runtime/include/tessera/tsr_types.h` | ✅ |
| `src/runtime/include/tessera/tsr_status.h` | ✅ |
| `src/runtime/include/tessera/tsr_shape.h` | ✅ |
| `src/runtime/include/tessera/tsr_version.h` | ✅ |

## Symbols per backend

| Backend | Unique tessera_* symbols |
|---------|-------------------------:|
| `apple` | 304 |
| `nvidia` | 4 |
| `rocm` | 10 |
| `x86` | 12 |

## Apple GPU kernel families × dtype matrix

| Op family | dtypes |
|-----------|--------|
| `asymmetric_bce` | `f32` |
| `bmm` | `bf16`, `f16`, `f32` |
| `bmm_dev` | `f32` |
| `cf_scan` | `f32` |
| `cf_serial_draft` | `f32` |
| `cf_while_generate` | `f32` |
| `cholesky` | `f32` |
| `cholesky_batched` | `f32` |
| `clifford_codiff_cl30` | `f32` |
| `clifford_exp_cl30` | `f32` |
| `clifford_ext_deriv_cl30` | `f32` |
| `clifford_geo_product_cl30` | `bf16`, `f16`, `f32` |
| `clifford_geo_product_cl30_value` | `f32` |
| `clifford_grade_projection_cl30` | `f32` |
| `clifford_inner_cl30` | `f32` |
| `clifford_integral_cl30` | `f32` |
| `clifford_left_contraction_cl30` | `f32` |
| `clifford_log_cl30` | `f32` |
| `clifford_norm_cl30` | `f32` |
| `clifford_norm_squared_cl30` | `f32` |
| `clifford_rotor_sandwich_cl30` | `bf16`, `f16`, `f32` |
| `clifford_rotor_sandwich_norm_cl30` | `f32` |
| `clifford_vec_deriv_cl30` | `f32` |
| `clifford_wedge_cl30` | `f32` |
| `complex_exp` | `f32` |
| `complex_mobius` | `f32` |
| `complex_mul` | `f32` |
| `complex_stereographic` | `f32` |
| `conv2d` | `f16`, `f32` |
| `conv3d` | `f16`, `f32` |
| `count_nonzero_lastaxis` | `f32` |
| `dequant_matmul` | `f32` |
| `ebm_decode_init_noise_apply` | `f32` |
| `ebm_dsm` | `f32` |
| `ebm_ebt_tiny_refinement_argmin` | `f32` |
| `ebm_energy_diff_mean` | `f32` |
| `ebm_energy_quadratic` | `f32` |
| `ebm_energy_quadratic_value` | `f32` |
| `ebm_half_mse` | `f32` |
| `ebm_inner_step` | `f32` |
| `ebm_ism` | `f32` |
| `ebm_langevin_step` | `f32` |
| `ebm_langevin_step_philox` | `f32` |
| `ebm_langevin_step_value` | `f32` |
| `ebm_partition_exact` | `f32` |
| `ebm_partition_exact_value` | `f32` |
| `ebm_refinement` | `f32` |
| `ebm_refinement_value` | `f32` |
| `ebm_self_verify_hard_argmin` | `f32` |
| `ebm_sphere_langevin_step` | `f32` |
| `fft` | `f32` |
| `flash_attn` | `bf16`, `f16`, `f32` |
| `flash_attn_bias` | `bf16`, `f16`, `f32` |
| `flash_attn_gqa` | `bf16`, `f16`, `f32` |
| `gated_delta_rule` | `f32` |
| `gated_delta_rule_chunked` | `f32` |
| `gated_delta_rule_decode` | `f16`, `f32` |
| `gated_delta_rule_decode_big` | `f32` |
| `gather_blocks_dev` | `f32` |
| `gelu` | `bf16`, `f16`, `f32` |
| `grouped_gemm` | `f32` |
| `gumbel_argmax` | `f32` |
| `gumbel_argmax_dev` | `f32` |
| `layer_norm` | `bf16`, `f16`, `f32` |
| `linear_attn` | `f32` |
| `load_balance_loss` | `f32` |
| `log_softmax` | `f16`, `f32` |
| `lookahead_sparse_attn` | `f32` |
| `masked_categorical` | `f32` |
| `matmul_softmax_matmul` | `bf16`, `f16`, `f32` |
| `mla_absorb_decode` | `bf16`, `f16`, `f32` |
| `mla_decode` | `bf16`, `f16`, `f32` |
| `mla_decode_rope` | `bf16`, `f16`, `f32` |
| `moe_swiglu` | `f32` |
| `mps_matmul` | `bf16`, `f16`, `f32` |
| `mpsgraph_argreduce` | `f32` |
| `mpsgraph_binary` | `bf16`, `f16`, `f32` |
| `mpsgraph_bsmm` | `f16`, `f32` |
| `mpsgraph_concat` | `f16`, `f32` |
| `mpsgraph_gather` | `f16`, `f32` |
| `mpsgraph_reduce` | `f32` |
| `mpsgraph_scan` | `f32` |
| `mpsgraph_slice` | `f16`, `f32` |
| `mpsgraph_softmax` | `f16`, `f32` |
| `mpsgraph_topk` | `f32` |
| `mpsgraph_transpose` | `f16`, `f32` |
| `mpsgraph_unary` | `bf16`, `f16`, `f32` |
| `msa_block_sparse` | `f16`, `f32` |
| `msa_block_sparse_tiled` | `f16`, `f32` |
| `msa_select_blocks` | `f32` |
| `mtl4_conv2d` | `bf16`, `f16` |
| `mtl4_matmul2d` | `bf16`, `f16` |
| `mtl4_matmul2d_epilogue` | `bf16`, `f16` |
| `mtl4_matmul_sg` | `f32` |
| `mtl4_scan` | `f32` |
| `native_sparse_attn` | `f32` |
| `popcount` | `i32` |
| `ppo_policy_loss` | `f32` |
| `ppo_policy_loss_ex` | `f32` |
| `quantized_matmul_fp4` | `f32` |
| `quantized_matmul_i4` | `f16`, `f32` |
| `quantized_matmul_i4_splitk` | `f32` |
| `quantized_matmul_i4_tiled` | `f32` |
| `random_normal` | `f32` |
| `random_uniform` | `f32` |
| `rmsnorm_gpu` | `bf16`, `f16`, `f32` |
| `rmsnorm_matmul` | `f32` |
| `rope` | `bf16`, `f16`, `f32` |
| `rowop_dev` | `f32` |
| `run_graph_cond` | `f16`, `f32` |
| `run_graph_loop` | `f16`, `f32` |
| `run_graph_scan` | `f32` |
| `run_graph_while` | `f16`, `f32` |
| `softmax` | `bf16`, `f16`, `f32` |
| `solve_cholesky` | `f32` |
| `solve_lu` | `f32` |
| `spike_conv2d_multi_tile` | `f16` |
| `spike_conv2d_single_tile` | `f16` |
| `ssm_block_decode` | `f16`, `f32` |
| `ssm_replay_decode` | `f32` |
| `svd` | `f32` |
| `svd_batched` | `f32` |
| `svd_bl_batched` | `f32` |
| `swiglu` | `bf16`, `f16`, `f32` |
| `synth_attention` | `f16`, `f32` |
| `synth_gated_matmul` | `f16`, `f32` |
| `synth_matmul_epilogue` | `f16`, `f32` |
| `synth_matmul_epilogue_tiled` | `f32` |
| `synth_norm_chain` | `f16`, `f32` |
| `synth_pointwise` | `f16`, `f32` |
| `synth_pointwise_reduce` | `f16`, `f32` |
| `tri_solve` | `f32` |
| `tri_solve_batched` | `f32` |
| `z_loss` | `f32` |

## Toolchain version pins

Pins declared in Python (`gpu_target.py` / `rocm_target.py`) and CMake (`cmake/TesseraToolchainPins.cmake`).  These MUST agree across sources — a mismatch means a sprint left one source behind.

### `cuda_toolkit`

| Source | Declared value |
|--------|----------------|
| `python_gpu_target` | _not found_ |
| `cmake_pins` | `13.3` |

✅ Sources agree.

### `nccl_minimum`

| Source | Declared value |
|--------|----------------|
| `python_gpu_target` | `2.22` |
| `cmake_pins` | _not found_ |

✅ Sources agree.

### `rocm`

| Source | Declared value |
|--------|----------------|
| `python_rocm_target` | `7.2.4` |
| `cmake_pins` | `7.2.4` |

✅ Sources agree.
