# Autodiff law audit — AD-LAW-1

> **Generated** by `python/tessera/compiler/law_audit.py` from
> `tessera.autodiff.laws.run_law_sweep()`. Do not hand-edit; regenerate
> via `scripts/check_generated_docs.sh --write`. Design authority:
> [`AUTODIFF_NEXTGEN_PLAN.md`](../compiler/AUTODIFF_NEXTGEN_PLAN.md) §4.

Law 3 (adjoint, `⟨Jv,u⟩ = ⟨v,Jᵀu⟩`) is complete for the transpose
relationship between a paired JVP/VJP; it cannot certify the derivative
itself (a matched-wrong pair passes). Law 1 (chain vs finite
differences) is the derivative-correctness complement. `no_spec` rows
are real unswept debt, not exclusions.

## `tensor` registry

| Law | Status | Ops |
|---|---|---:|
| adjoint | no_spec | 125 |
| adjoint | pass | 166 |
| adjoint | vjp_only | 17 |
| chain | not_applicable | 8 |
| chain | pass | 128 |
| kink | pass | 12 |

## `geometric` registry

| Law | Status | Ops |
|---|---|---:|
| adjoint | pass | 16 |

## Checked tensor ops

`abs`, `absolute`, `acos`, `add`, `amax`, `amin`, `asin`, `atan`, `atan2`, `attn_sliding_window`, `batched_gemm`, `binary_cross_entropy_loss`, `broadcast`, `broadcast_to_axis`, `cast`, `cat`, `center_crop`, `cholesky`, `chunk`, `clamp`, `clifford_conjugate`, `clifford_exp`, `clifford_geometric_product`, `clifford_grade_involution`, `clifford_grade_projection`, `clifford_hodge_star`, `clifford_inner`, `clifford_left_contraction`, `clifford_log`, `clifford_norm`, `clifford_norm_squared`, `clifford_reverse`, `clifford_rotor_sandwich`, `clifford_wedge`, `clip`, `cos`, `cosh`, `cross_entropy_loss`, `cummax`, `cummin`, `cumprod`, `cumsum`, `dequantize_nvfp4`, `digamma`, `div`, `dynamic_slice`, `dynamic_update_slice`, `einsum`, `erf`, `erfc`, `exp`, `expand`, `expm1`, `factorized_matmul`, `fft`, `flash_attn`, `flatten`, `flip`, `gather`, `gelu`, `gemm`, `group_norm`, `huber_loss`, `ifft`, `image_normalize`, `index_select`, `index_update`, `instance_norm`, `irfft`, `istft`, `js_divergence`, `kl_divergence`, `label_smoothed_cross_entropy`, `latent_kv_compress`, `latent_kv_expand_k`, `latent_kv_expand_v`, `layer_norm`, `lgamma`, `linear_attn`, `linear_general`, `log`, `log1p`, `log_cosh_loss`, `log_softmax`, `logsumexp`, `lora_linear`, `mae_loss`, `masked_fill`, `masked_scatter`, `matmul`, `max`, `maximum`, `mean`, `min`, `minimum`, `mod`, `mor_scatter`, `mse_loss`, `mul`, `pad`, `patchify`, `permute`, `pixel_shuffle`, `pixel_unshuffle`, `pow`, `power_attn`, `prod`, `qkv_projection`, `qr`, `quantize_fp4`, `quantize_fp6`, `quantize_int4`, `quantize_int8`, `quantize_nvfp4`, `reciprocal`, `relu`, `repeat`, `reshape`, `retention`, `rfft`, `rmsnorm`, `rmsnorm_safe`, `roll`, `rope`, `rope_merge`, `rsqrt`, `scatter`, `scatter_add`, `scatter_reduce`, `segment_reduce`, `select`, `sigmoid`, `sigmoid_safe`, `sign`, `silu`, `silu_mul`, `sin`, `sinh`, `slice`, `smooth_l1_loss`, `softcap`, `softmax`, `softmax_safe`, `softplus`, `spectral_conv`, `spectral_norm`, `split`, `sqrt`, `squeeze`, `stack`, `std`, `stft`, `sub`, `sum`, `svd`, `take`, `tan`, `tanh`, `tile`, `transpose`, `tri_solve`, `unsqueeze`, `var`, `view`, `weight_norm`, `where`, `z_loss`
