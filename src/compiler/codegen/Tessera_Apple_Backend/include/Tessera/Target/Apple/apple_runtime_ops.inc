//===- apple_runtime_ops.inc - Apple GPU runtime envelope ------*- C++ -*-===//
//
// Auto-generated from `python/tessera/compiler/apple_gpu_envelope.py`
// (_APPLE_GPU_RUNTIME_OPS).  Do NOT edit by hand — regenerate with:
//
//     python scripts/generate_apple_runtime_ops_table.py
//
// X-macro format: each line is a TESSERA_APPLE_GPU_RUNTIME_OP call the
// includer expands.  Grouped by runtime dispatch lane for readability.
//
// Sync gates: tests/unit/test_apple_runtime_ops_table_in_sync.py (table)
// and tests/unit/test_apple_gpu_tile_pass_status_matches_envelope.py
// (real tessera-opt pass behavior).
//
//===----------------------------------------------------------------------===//

#ifndef TESSERA_APPLE_GPU_RUNTIME_OP
#error "Define TESSERA_APPLE_GPU_RUNTIME_OP(name) before including this file"
#endif

// lane: asymmetric_bce (1 ops)
TESSERA_APPLE_GPU_RUNTIME_OP("tessera.loss.asymmetric_bce")

// lane: attn_wrapper (6 ops)
TESSERA_APPLE_GPU_RUNTIME_OP("tessera.gated_attention")
TESSERA_APPLE_GPU_RUNTIME_OP("tessera.gqa_attention")
TESSERA_APPLE_GPU_RUNTIME_OP("tessera.mla_decode")
TESSERA_APPLE_GPU_RUNTIME_OP("tessera.mla_decode_fused")
TESSERA_APPLE_GPU_RUNTIME_OP("tessera.mqa_attention")
TESSERA_APPLE_GPU_RUNTIME_OP("tessera.multi_head_attention")

// lane: binary (22 ops)
TESSERA_APPLE_GPU_RUNTIME_OP("tessera.add")
TESSERA_APPLE_GPU_RUNTIME_OP("tessera.atan2")
TESSERA_APPLE_GPU_RUNTIME_OP("tessera.bitwise_and")
TESSERA_APPLE_GPU_RUNTIME_OP("tessera.bitwise_or")
TESSERA_APPLE_GPU_RUNTIME_OP("tessera.bitwise_xor")
TESSERA_APPLE_GPU_RUNTIME_OP("tessera.div")
TESSERA_APPLE_GPU_RUNTIME_OP("tessera.eq")
TESSERA_APPLE_GPU_RUNTIME_OP("tessera.floor_div")
TESSERA_APPLE_GPU_RUNTIME_OP("tessera.ge")
TESSERA_APPLE_GPU_RUNTIME_OP("tessera.gt")
TESSERA_APPLE_GPU_RUNTIME_OP("tessera.le")
TESSERA_APPLE_GPU_RUNTIME_OP("tessera.logical_and")
TESSERA_APPLE_GPU_RUNTIME_OP("tessera.logical_or")
TESSERA_APPLE_GPU_RUNTIME_OP("tessera.logical_xor")
TESSERA_APPLE_GPU_RUNTIME_OP("tessera.lt")
TESSERA_APPLE_GPU_RUNTIME_OP("tessera.maximum")
TESSERA_APPLE_GPU_RUNTIME_OP("tessera.minimum")
TESSERA_APPLE_GPU_RUNTIME_OP("tessera.mod")
TESSERA_APPLE_GPU_RUNTIME_OP("tessera.mul")
TESSERA_APPLE_GPU_RUNTIME_OP("tessera.ne")
TESSERA_APPLE_GPU_RUNTIME_OP("tessera.pow")
TESSERA_APPLE_GPU_RUNTIME_OP("tessera.sub")

// lane: clamp (2 ops)
TESSERA_APPLE_GPU_RUNTIME_OP("tessera.clamp")
TESSERA_APPLE_GPU_RUNTIME_OP("tessera.clip")

// lane: clifford (17 ops)
TESSERA_APPLE_GPU_RUNTIME_OP("tessera.clifford_codiff")
TESSERA_APPLE_GPU_RUNTIME_OP("tessera.clifford_conjugate")
TESSERA_APPLE_GPU_RUNTIME_OP("tessera.clifford_exp")
TESSERA_APPLE_GPU_RUNTIME_OP("tessera.clifford_ext_deriv")
TESSERA_APPLE_GPU_RUNTIME_OP("tessera.clifford_geometric_product")
TESSERA_APPLE_GPU_RUNTIME_OP("tessera.clifford_grade_involution")
TESSERA_APPLE_GPU_RUNTIME_OP("tessera.clifford_grade_projection")
TESSERA_APPLE_GPU_RUNTIME_OP("tessera.clifford_hodge_star")
TESSERA_APPLE_GPU_RUNTIME_OP("tessera.clifford_inner")
TESSERA_APPLE_GPU_RUNTIME_OP("tessera.clifford_left_contraction")
TESSERA_APPLE_GPU_RUNTIME_OP("tessera.clifford_log")
TESSERA_APPLE_GPU_RUNTIME_OP("tessera.clifford_norm")
TESSERA_APPLE_GPU_RUNTIME_OP("tessera.clifford_norm_squared")
TESSERA_APPLE_GPU_RUNTIME_OP("tessera.clifford_reverse")
TESSERA_APPLE_GPU_RUNTIME_OP("tessera.clifford_rotor_sandwich")
TESSERA_APPLE_GPU_RUNTIME_OP("tessera.clifford_vec_deriv")
TESSERA_APPLE_GPU_RUNTIME_OP("tessera.clifford_wedge")

// lane: concat (1 ops)
TESSERA_APPLE_GPU_RUNTIME_OP("tessera.cat")

// lane: conv2d (1 ops)
TESSERA_APPLE_GPU_RUNTIME_OP("tessera.conv2d")

// lane: conv3d (1 ops)
TESSERA_APPLE_GPU_RUNTIME_OP("tessera.conv3d")

// lane: count_nonzero (1 ops)
TESSERA_APPLE_GPU_RUNTIME_OP("tessera.count_nonzero")

// lane: delta_attn (3 ops)
TESSERA_APPLE_GPU_RUNTIME_OP("tessera.gated_deltanet")
TESSERA_APPLE_GPU_RUNTIME_OP("tessera.kimi_delta_attention")
TESSERA_APPLE_GPU_RUNTIME_OP("tessera.modified_delta_attention")

// lane: ebm (4 ops)
TESSERA_APPLE_GPU_RUNTIME_OP("tessera.ebm_energy_quadratic")
TESSERA_APPLE_GPU_RUNTIME_OP("tessera.ebm_inner_step")
TESSERA_APPLE_GPU_RUNTIME_OP("tessera.ebm_refinement")
TESSERA_APPLE_GPU_RUNTIME_OP("tessera.ebm_self_verify")

// lane: ebm_loss (5 ops)
TESSERA_APPLE_GPU_RUNTIME_OP("tessera.loss.contrastive_divergence")
TESSERA_APPLE_GPU_RUNTIME_OP("tessera.loss.denoising_score_matching")
TESSERA_APPLE_GPU_RUNTIME_OP("tessera.loss.implicit_score_matching")
TESSERA_APPLE_GPU_RUNTIME_OP("tessera.loss.persistent_cd")
TESSERA_APPLE_GPU_RUNTIME_OP("tessera.loss.score_matching")

// lane: flash_attn (1 ops)
TESSERA_APPLE_GPU_RUNTIME_OP("tessera.flash_attn")

// lane: gather (1 ops)
TESSERA_APPLE_GPU_RUNTIME_OP("tessera.gather")

// lane: gelu (1 ops)
TESSERA_APPLE_GPU_RUNTIME_OP("tessera.gelu")

// lane: grouped_gemm (1 ops)
TESSERA_APPLE_GPU_RUNTIME_OP("tessera.grouped_gemm")

// lane: hybrid_attn (1 ops)
TESSERA_APPLE_GPU_RUNTIME_OP("tessera.hybrid_attention")

// lane: linalg (2 ops)
TESSERA_APPLE_GPU_RUNTIME_OP("tessera.cholesky")
TESSERA_APPLE_GPU_RUNTIME_OP("tessera.tri_solve")

// lane: linear_attn (5 ops)
TESSERA_APPLE_GPU_RUNTIME_OP("tessera.lightning_attention")
TESSERA_APPLE_GPU_RUNTIME_OP("tessera.linear_attn")
TESSERA_APPLE_GPU_RUNTIME_OP("tessera.linear_attn_state")
TESSERA_APPLE_GPU_RUNTIME_OP("tessera.power_attn")
TESSERA_APPLE_GPU_RUNTIME_OP("tessera.retention")

// lane: linear_general (1 ops)
TESSERA_APPLE_GPU_RUNTIME_OP("tessera.linear_general")

// lane: load_balance_loss (1 ops)
TESSERA_APPLE_GPU_RUNTIME_OP("tessera.loss.load_balance_loss")

// lane: loss_compose (11 ops)
TESSERA_APPLE_GPU_RUNTIME_OP("tessera.loss.binary_cross_entropy")
TESSERA_APPLE_GPU_RUNTIME_OP("tessera.loss.cross_entropy")
TESSERA_APPLE_GPU_RUNTIME_OP("tessera.loss.ddpm_noise_pred")
TESSERA_APPLE_GPU_RUNTIME_OP("tessera.loss.huber")
TESSERA_APPLE_GPU_RUNTIME_OP("tessera.loss.js_divergence")
TESSERA_APPLE_GPU_RUNTIME_OP("tessera.loss.kl_divergence")
TESSERA_APPLE_GPU_RUNTIME_OP("tessera.loss.log_cosh")
TESSERA_APPLE_GPU_RUNTIME_OP("tessera.loss.mae")
TESSERA_APPLE_GPU_RUNTIME_OP("tessera.loss.mse")
TESSERA_APPLE_GPU_RUNTIME_OP("tessera.loss.smooth_l1")
TESSERA_APPLE_GPU_RUNTIME_OP("tessera.loss.vlb")

// lane: masked_attn (2 ops)
TESSERA_APPLE_GPU_RUNTIME_OP("tessera.attn_compressed_blocks")
TESSERA_APPLE_GPU_RUNTIME_OP("tessera.attn_sliding_window")

// lane: masked_categorical (1 ops)
TESSERA_APPLE_GPU_RUNTIME_OP("tessera.masked_categorical")

// lane: moe_swiglu_block (1 ops)
TESSERA_APPLE_GPU_RUNTIME_OP("tessera.moe_swiglu_block")

// lane: mps (3 ops)
TESSERA_APPLE_GPU_RUNTIME_OP("tessera.batched_gemm")
TESSERA_APPLE_GPU_RUNTIME_OP("tessera.gemm")
TESSERA_APPLE_GPU_RUNTIME_OP("tessera.matmul")

// lane: norm_compose (3 ops)
TESSERA_APPLE_GPU_RUNTIME_OP("tessera.group_norm")
TESSERA_APPLE_GPU_RUNTIME_OP("tessera.instance_norm")
TESSERA_APPLE_GPU_RUNTIME_OP("tessera.weight_norm")

// lane: popcount (1 ops)
TESSERA_APPLE_GPU_RUNTIME_OP("tessera.popcount")

// lane: qkv_projection (1 ops)
TESSERA_APPLE_GPU_RUNTIME_OP("tessera.qkv_projection")

// lane: quant_matmul (1 ops)
TESSERA_APPLE_GPU_RUNTIME_OP("tessera.quantized_matmul")

// lane: reduce (16 ops)
TESSERA_APPLE_GPU_RUNTIME_OP("tessera.amax")
TESSERA_APPLE_GPU_RUNTIME_OP("tessera.amin")
TESSERA_APPLE_GPU_RUNTIME_OP("tessera.argmax")
TESSERA_APPLE_GPU_RUNTIME_OP("tessera.argmin")
TESSERA_APPLE_GPU_RUNTIME_OP("tessera.cummax")
TESSERA_APPLE_GPU_RUNTIME_OP("tessera.cummin")
TESSERA_APPLE_GPU_RUNTIME_OP("tessera.cumprod")
TESSERA_APPLE_GPU_RUNTIME_OP("tessera.cumsum")
TESSERA_APPLE_GPU_RUNTIME_OP("tessera.logsumexp")
TESSERA_APPLE_GPU_RUNTIME_OP("tessera.max")
TESSERA_APPLE_GPU_RUNTIME_OP("tessera.mean")
TESSERA_APPLE_GPU_RUNTIME_OP("tessera.min")
TESSERA_APPLE_GPU_RUNTIME_OP("tessera.prod")
TESSERA_APPLE_GPU_RUNTIME_OP("tessera.reduce")
TESSERA_APPLE_GPU_RUNTIME_OP("tessera.std")
TESSERA_APPLE_GPU_RUNTIME_OP("tessera.var")

// lane: rope (1 ops)
TESSERA_APPLE_GPU_RUNTIME_OP("tessera.rope")

// lane: rowop (4 ops)
TESSERA_APPLE_GPU_RUNTIME_OP("tessera.layer_norm")
TESSERA_APPLE_GPU_RUNTIME_OP("tessera.log_softmax")
TESSERA_APPLE_GPU_RUNTIME_OP("tessera.rmsnorm")
TESSERA_APPLE_GPU_RUNTIME_OP("tessera.rmsnorm_safe")

// lane: silu_mul (1 ops)
TESSERA_APPLE_GPU_RUNTIME_OP("tessera.silu_mul")

// lane: slice (1 ops)
TESSERA_APPLE_GPU_RUNTIME_OP("tessera.slice")

// lane: softcap (1 ops)
TESSERA_APPLE_GPU_RUNTIME_OP("tessera.softcap")

// lane: softmax (2 ops)
TESSERA_APPLE_GPU_RUNTIME_OP("tessera.softmax")
TESSERA_APPLE_GPU_RUNTIME_OP("tessera.softmax_safe")

// lane: sparse_attn (5 ops)
TESSERA_APPLE_GPU_RUNTIME_OP("tessera.attn_local_window_2d")
TESSERA_APPLE_GPU_RUNTIME_OP("tessera.attn_top_k_blocks")
TESSERA_APPLE_GPU_RUNTIME_OP("tessera.deepseek_sparse_attention")
TESSERA_APPLE_GPU_RUNTIME_OP("tessera.lookahead_sparse_attention")
TESSERA_APPLE_GPU_RUNTIME_OP("tessera.msa_sparse_attention")

// lane: spectral (9 ops)
TESSERA_APPLE_GPU_RUNTIME_OP("tessera.dct")
TESSERA_APPLE_GPU_RUNTIME_OP("tessera.fft")
TESSERA_APPLE_GPU_RUNTIME_OP("tessera.ifft")
TESSERA_APPLE_GPU_RUNTIME_OP("tessera.irfft")
TESSERA_APPLE_GPU_RUNTIME_OP("tessera.istft")
TESSERA_APPLE_GPU_RUNTIME_OP("tessera.rfft")
TESSERA_APPLE_GPU_RUNTIME_OP("tessera.spectral_conv")
TESSERA_APPLE_GPU_RUNTIME_OP("tessera.spectral_filter")
TESSERA_APPLE_GPU_RUNTIME_OP("tessera.stft")

// lane: ssm (1 ops)
TESSERA_APPLE_GPU_RUNTIME_OP("tessera.selective_ssm")

// lane: topk (1 ops)
TESSERA_APPLE_GPU_RUNTIME_OP("tessera.top_k")

// lane: transpose (1 ops)
TESSERA_APPLE_GPU_RUNTIME_OP("tessera.transpose")

// lane: unary (37 ops)
TESSERA_APPLE_GPU_RUNTIME_OP("tessera.abs")
TESSERA_APPLE_GPU_RUNTIME_OP("tessera.absolute")
TESSERA_APPLE_GPU_RUNTIME_OP("tessera.acos")
TESSERA_APPLE_GPU_RUNTIME_OP("tessera.asin")
TESSERA_APPLE_GPU_RUNTIME_OP("tessera.atan")
TESSERA_APPLE_GPU_RUNTIME_OP("tessera.bitwise_not")
TESSERA_APPLE_GPU_RUNTIME_OP("tessera.ceil")
TESSERA_APPLE_GPU_RUNTIME_OP("tessera.cos")
TESSERA_APPLE_GPU_RUNTIME_OP("tessera.cosh")
TESSERA_APPLE_GPU_RUNTIME_OP("tessera.erf")
TESSERA_APPLE_GPU_RUNTIME_OP("tessera.erfc")
TESSERA_APPLE_GPU_RUNTIME_OP("tessera.exp")
TESSERA_APPLE_GPU_RUNTIME_OP("tessera.expm1")
TESSERA_APPLE_GPU_RUNTIME_OP("tessera.floor")
TESSERA_APPLE_GPU_RUNTIME_OP("tessera.isfinite")
TESSERA_APPLE_GPU_RUNTIME_OP("tessera.isinf")
TESSERA_APPLE_GPU_RUNTIME_OP("tessera.isnan")
TESSERA_APPLE_GPU_RUNTIME_OP("tessera.log")
TESSERA_APPLE_GPU_RUNTIME_OP("tessera.log1p")
TESSERA_APPLE_GPU_RUNTIME_OP("tessera.logical_not")
TESSERA_APPLE_GPU_RUNTIME_OP("tessera.neg")
TESSERA_APPLE_GPU_RUNTIME_OP("tessera.negative")
TESSERA_APPLE_GPU_RUNTIME_OP("tessera.reciprocal")
TESSERA_APPLE_GPU_RUNTIME_OP("tessera.relu")
TESSERA_APPLE_GPU_RUNTIME_OP("tessera.round")
TESSERA_APPLE_GPU_RUNTIME_OP("tessera.rsqrt")
TESSERA_APPLE_GPU_RUNTIME_OP("tessera.sigmoid")
TESSERA_APPLE_GPU_RUNTIME_OP("tessera.sigmoid_safe")
TESSERA_APPLE_GPU_RUNTIME_OP("tessera.sign")
TESSERA_APPLE_GPU_RUNTIME_OP("tessera.silu")
TESSERA_APPLE_GPU_RUNTIME_OP("tessera.sin")
TESSERA_APPLE_GPU_RUNTIME_OP("tessera.sinh")
TESSERA_APPLE_GPU_RUNTIME_OP("tessera.softplus")
TESSERA_APPLE_GPU_RUNTIME_OP("tessera.sqrt")
TESSERA_APPLE_GPU_RUNTIME_OP("tessera.tan")
TESSERA_APPLE_GPU_RUNTIME_OP("tessera.tanh")
TESSERA_APPLE_GPU_RUNTIME_OP("tessera.trunc")

// lane: where (1 ops)
TESSERA_APPLE_GPU_RUNTIME_OP("tessera.where")

// lane: z_loss (1 ops)
TESSERA_APPLE_GPU_RUNTIME_OP("tessera.loss.z_loss")

// Total runtime-executable ops: 184
#undef TESSERA_APPLE_GPU_RUNTIME_OP
