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
| adjoint | no_spec | 173 |
| adjoint | pass | 118 |
| adjoint | vjp_only | 17 |
| chain | not_applicable | 1 |
| chain | pass | 87 |
| kink | pass | 12 |

## `geometric` registry

| Law | Status | Ops |
|---|---|---:|
| adjoint | no_spec | 16 |

## Checked tensor ops

`abs`, `absolute`, `acos`, `add`, `amax`, `amin`, `asin`, `atan`, `atan2`, `attn_sliding_window`, `batched_gemm`, `binary_cross_entropy_loss`, `cast`, `clamp`, `clip`, `cos`, `cosh`, `cross_entropy_loss`, `cummax`, `cummin`, `cumprod`, `cumsum`, `dequantize_nvfp4`, `digamma`, `div`, `einsum`, `erf`, `erfc`, `exp`, `expand`, `expm1`, `fft`, `flash_attn`, `flatten`, `flip`, `gather`, `gelu`, `gemm`, `group_norm`, `huber_loss`, `ifft`, `index_select`, `instance_norm`, `irfft`, `istft`, `js_divergence`, `kl_divergence`, `label_smoothed_cross_entropy`, `layer_norm`, `lgamma`, `linear_attn`, `log`, `log1p`, `log_cosh_loss`, `log_softmax`, `logsumexp`, `mae_loss`, `masked_fill`, `matmul`, `max`, `maximum`, `mean`, `min`, `minimum`, `mod`, `mse_loss`, `mul`, `pad`, `permute`, `pow`, `power_attn`, `prod`, `qkv_projection`, `quantize_fp4`, `quantize_fp6`, `quantize_int4`, `quantize_int8`, `quantize_nvfp4`, `reciprocal`, `relu`, `repeat`, `reshape`, `retention`, `rfft`, `rmsnorm`, `rmsnorm_safe`, `roll`, `rope`, `rope_merge`, `rsqrt`, `segment_reduce`, `sigmoid`, `sigmoid_safe`, `sign`, `silu`, `silu_mul`, `sin`, `sinh`, `smooth_l1_loss`, `softcap`, `softmax`, `softmax_safe`, `softplus`, `spectral_conv`, `sqrt`, `squeeze`, `std`, `stft`, `sub`, `sum`, `take`, `tan`, `tanh`, `tile`, `transpose`, `unsqueeze`, `var`, `where`, `z_loss`
