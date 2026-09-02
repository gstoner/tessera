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

Law 7 (`field_calculus` registry, MSW-4) is a different kind of check:
reference-free identities relating the GA field operators to **each
other** — `d∘d = 0`, `δ∘δ = 0`, Stokes, the Leibniz rule — plus one
external-oracle law. It catches what a per-op adjoint cannot: a
wrong-but-self-consistent operator has a perfectly good adjoint. Two of
its rows are order-of-accuracy checks rather than equalities, because a
central-difference product rule is exact only to O(h²); the residual for
those is the coarse-grid relative error, not a tolerance.

## `tensor` registry

| Law | Status | Ops |
|---|---|---:|
| adjoint | no_spec | 8 |
| adjoint | pass | 310 |
| chain | not_applicable | 36 |
| chain | pass | 224 |
| kink | pass | 12 |

## `geometric` registry

| Law | Status | Ops |
|---|---|---:|
| adjoint | pass | 16 |

## `field_calculus` registry

| Law | Status | Ops |
|---|---|---:|
| vector_identity | pass | 5 |

## Checked tensor ops

`abs`, `absolute`, `acos`, `adafactor`, `adam`, `adamw`, `add`, `alibi`, `all_gather`, `all_reduce`, `all_to_all`, `amax`, `amin`, `asin`, `asymmetric_bce`, `atan`, `atan2`, `attn_compressed_blocks`, `attn_local_window_2d`, `attn_sliding_window`, `attn_top_k_blocks`, `avg_pool`, `batched_gemm`, `binary_cross_entropy_loss`, `broadcast`, `broadcast_to_axis`, `bsmm`, `calibration_observer`, `cast`, `cat`, `center_crop`, `cholesky`, `chunk`, `cispo_policy_loss`, `clamp`, `clifford_codiff`, `clifford_conjugate`, `clifford_exp`, `clifford_ext_deriv`, `clifford_geometric_product`, `clifford_grade_involution`, `clifford_grade_projection`, `clifford_hodge_star`, `clifford_inner`, `clifford_left_contraction`, `clifford_log`, `clifford_norm`, `clifford_norm_squared`, `clifford_reverse`, `clifford_rotor_sandwich`, `clifford_vec_deriv`, `clifford_wedge`, `clip`, `collective_permute`, `contrastive_divergence_loss`, `contrastive_loss`, `conv1d`, `conv2d`, `conv3d`, `cos`, `cosh`, `cosine_embedding_loss`, `cross_attention`, `cross_entropy_loss`, `ctc_loss`, `cummax`, `cummin`, `cumprod`, `cumsum`, `dct`, `ddpm_noise_pred_loss`, `deepseek_sparse_attention`, `denoising_score_matching_loss`, `depth_attn`, `depthwise_conv1d`, `depthwise_conv2d`, `dequantize_fp4`, `dequantize_fp6`, `dequantize_fp8`, `dequantize_int4`, `dequantize_int8`, `dequantize_nvfp4`, `det`, `digamma`, `div`, `dropout`, `dynamic_slice`, `dynamic_update_slice`, `ebm_energy_quadratic`, `ebm_inner_step`, `ebm_refinement`, `ebm_self_verify`, `eigh`, `einsum`, `embedding`, `entmax15`, `erf`, `erfc`, `exp`, `expand`, `expm1`, `factorized_matmul`, `factorized_pos_emb`, `fake_quantize`, `fft`, `flash_attn`, `flatten`, `flip`, `floor_div`, `focal_loss`, `fused_epilogue`, `game_boltzmann_value`, `game_coalition_excess`, `game_coalition_marginal`, `game_semivalue`, `game_subset_mobius`, `game_subset_zeta`, `game_superset_mobius`, `game_superset_zeta`, `gated_attention`, `gated_deltanet`, `gather`, `gelu`, `gemm`, `gqa_attention`, `grad_scaler_step`, `group_norm`, `grouped_gemm`, `grpo_policy_loss`, `gru_cell`, `gumbel_softmax`, `huber_loss`, `hybrid_attention`, `ifft`, `image_normalize`, `image_resize`, `implicit_score_matching_loss`, `index_select`, `index_update`, `info_nce_loss`, `instance_norm`, `interpolate`, `inv`, `irfft`, `istft`, `js_divergence`, `kimi_delta_attention`, `kl_divergence`, `kron`, `label_smoothed_cross_entropy`, `latent_kv_compress`, `latent_kv_expand_k`, `latent_kv_expand_v`, `layer_norm`, `lgamma`, `lightning_attention`, `linear_attn`, `linear_attn_state`, `linear_general`, `lion`, `load_balance_loss`, `log`, `log1p`, `log_cosh_loss`, `log_softmax`, `logdet`, `logsumexp`, `lookahead_sparse_attention`, `lora_linear`, `lstm_cell`, `lstm_state_c`, `lstm_state_h`, `mae_loss`, `masked_fill`, `masked_scatter`, `matmul`, `matrix_power`, `max`, `max_pool`, `maximum`, `mean`, `memory_index_score`, `memory_index_select_ste`, `min`, `minimum`, `mla_decode`, `mla_decode_fused`, `mod`, `modified_delta_attention`, `moe`, `moe_combine`, `moe_dispatch`, `momentum`, `mor_partition`, `mor_router`, `mor_scatter`, `mqa_attention`, `mrope_2d`, `msa_index_scores`, `msa_sparse_attention`, `mse_loss`, `mul`, `multi_head_attention`, `nesterov`, `norm`, `normalize_group_advantages`, `nt_xent_loss`, `ntk_rope`, `online_softmax`, `online_softmax_state`, `pad`, `patchify`, `permute`, `persistent_cd_loss`, `perturbed_argmax`, `pixel_shuffle`, `pixel_unshuffle`, `pmax`, `pmean`, `pmin`, `pow`, `power_attn`, `ppo_policy_loss`, `prod`, `psum`, `qkv_projection`, `qr`, `quantize_fp4`, `quantize_fp6`, `quantize_fp8`, `quantize_int4`, `quantize_int8`, `quantize_nvfp4`, `quantized_matmul`, `reciprocal`, `reduce`, `reduce_scatter`, `relu`, `repeat`, `reshape`, `retention`, `rfft`, `rmsnorm`, `rmsnorm_safe`, `roll`, `rope`, `rope_merge`, `rope_split`, `rsqrt`, `scatter`, `scatter_add`, `scatter_reduce`, `score_matching_loss`, `sddmm`, `segment_reduce`, `select`, `selective_ssm`, `seq2seq_loss`, `sgd`, `sigmoid`, `sigmoid_safe`, `sign`, `silu`, `silu_mul`, `simple_rnn_cell`, `sin`, `sinh`, `slice`, `smooth_l1_loss`, `soft_top_k`, `softcap`, `softmax`, `softmax_safe`, `softplus`, `solve`, `sparsemax`, `spectral_conv`, `spectral_filter`, `spectral_norm`, `split`, `spmm_coo`, `spmm_csr`, `sqrt`, `squeeze`, `stack`, `std`, `stft`, `stop_gradient`, `sub`, `sum`, `svd`, `take`, `tan`, `tanh`, `tile`, `top_k_routing`, `trace`, `transpose`, `tri_solve`, `tridiagonal_solve`, `triplet_loss`, `unsqueeze`, `var`, `vec`, `view`, `vlb_loss`, `wasserstein_distance`, `weight_norm`, `where`, `z_loss`
