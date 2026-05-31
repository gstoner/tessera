# Runtime C ABI Surface Audit

Generated from `python/tessera/compiler/runtime_abi_audit.py`.  Don't edit by hand — regenerate via `python -c "from tessera.compiler.runtime_abi_audit import render_dashboard; open('docs/audit/generated/runtime_abi.md', 'w').write(render_dashboard())"`.  Drift gated by `tests/unit/test_runtime_abi_audit.py`.

## Headline

- **165** unique `extern "C" tessera_*` C ABI symbols across all backends.
- **6 / 6** core runtime headers present.
- **80** Apple GPU kernel families with per-dtype variants.

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
| `apple` | 154 |
| `nvidia` | 3 |
| `x86` | 8 |

## Apple GPU kernel families × dtype matrix

| Op family | dtypes |
|-----------|--------|
| `bmm` | `f16`, `f32` |
| `bmm_dev` | `f32` |
| `cf_scan` | `f32` |
| `cf_serial_draft` | `f32` |
| `cf_while_generate` | `f32` |
| `cholesky` | `f32` |
| `clifford_codiff_cl30` | `f32` |
| `clifford_exp_cl30` | `f32` |
| `clifford_ext_deriv_cl30` | `f32` |
| `clifford_geo_product_cl30` | `bf16`, `f16`, `f32` |
| `clifford_grade_projection_cl30` | `f32` |
| `clifford_inner_cl30` | `f32` |
| `clifford_integral_cl30` | `f32` |
| `clifford_left_contraction_cl30` | `f32` |
| `clifford_log_cl30` | `f32` |
| `clifford_norm_cl30` | `f32` |
| `clifford_norm_squared_cl30` | `f32` |
| `clifford_rotor_sandwich_cl30` | `bf16`, `f16`, `f32` |
| `clifford_vec_deriv_cl30` | `f32` |
| `clifford_wedge_cl30` | `f32` |
| `complex_exp` | `f32` |
| `complex_mobius` | `f32` |
| `complex_mul` | `f32` |
| `complex_stereographic` | `f32` |
| `conv2d` | `f16`, `f32` |
| `conv3d` | `f16`, `f32` |
| `ebm_decode_init_noise_apply` | `f32` |
| `ebm_ebt_tiny_refinement_argmin` | `f32` |
| `ebm_energy_quadratic` | `f32` |
| `ebm_inner_step` | `f32` |
| `ebm_langevin_step` | `f32` |
| `ebm_langevin_step_philox` | `f32` |
| `ebm_partition_exact` | `f32` |
| `ebm_refinement` | `f32` |
| `ebm_self_verify_hard_argmin` | `f32` |
| `ebm_sphere_langevin_step` | `f32` |
| `flash_attn` | `bf16`, `f16`, `f32` |
| `flash_attn_gqa` | `bf16`, `f16`, `f32` |
| `gather_blocks_dev` | `f32` |
| `gelu` | `bf16`, `f16`, `f32` |
| `gumbel_argmax` | `f32` |
| `gumbel_argmax_dev` | `f32` |
| `layer_norm` | `f16`, `f32` |
| `linear_attn` | `f32` |
| `log_softmax` | `f16`, `f32` |
| `matmul_gelu` | `bf16`, `f16`, `f32` |
| `matmul_rmsnorm` | `bf16`, `f16`, `f32` |
| `matmul_softmax` | `bf16`, `f16`, `f32` |
| `matmul_softmax_matmul` | `bf16`, `f16`, `f32` |
| `matmul_softmax_tiled` | `bf16`, `f16`, `f32` |
| `mla_absorb_decode` | `bf16`, `f16`, `f32` |
| `mla_decode` | `bf16`, `f16`, `f32` |
| `mla_decode_rope` | `bf16`, `f16`, `f32` |
| `mps_matmul` | `bf16`, `f16`, `f32` |
| `mpsgraph_argreduce` | `f32` |
| `mpsgraph_binary` | `f16`, `f32` |
| `mpsgraph_bsmm` | `f16`, `f32` |
| `mpsgraph_reduce` | `f32` |
| `mpsgraph_scan` | `f32` |
| `mpsgraph_softmax` | `f16`, `f32` |
| `mpsgraph_unary` | `f16`, `f32` |
| `mtl4_conv2d` | `bf16`, `f16` |
| `mtl4_matmul2d` | `bf16`, `f16` |
| `mtl4_matmul2d_epilogue` | `bf16`, `f16` |
| `mtl4_matmul_sg` | `f32` |
| `mtl4_scan` | `f32` |
| `native_sparse_attn` | `f32` |
| `random_normal` | `f32` |
| `random_uniform` | `f32` |
| `rmsnorm_gpu` | `f16`, `f32` |
| `rope` | `bf16`, `f16`, `f32` |
| `rowop_dev` | `f32` |
| `softmax` | `bf16`, `f16`, `f32` |
| `solve_cholesky` | `f32` |
| `solve_lu` | `f32` |
| `svd` | `f32` |
| `svd_batched` | `f32` |
| `svd_bl_batched` | `f32` |
| `swiglu` | `bf16`, `f16`, `f32` |
| `tri_solve` | `f32` |

## Toolchain version pins

Pins declared in Python (`gpu_target.py` / `rocm_target.py`) and CMake (`cmake/TesseraToolchainPins.cmake`).  These MUST agree across sources — a mismatch means a sprint left one source behind.

### `cuda_toolkit`

| Source | Declared value |
|--------|----------------|
| `python_gpu_target` | _not found_ |
| `cmake_pins` | `13.2` |

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
| `python_rocm_target` | `7.2.3` |
| `cmake_pins` | `7.2.3` |

✅ Sources agree.
