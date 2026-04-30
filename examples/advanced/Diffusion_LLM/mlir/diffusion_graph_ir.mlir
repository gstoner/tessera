// diffusion_graph_ir.mlir
//
// Graph-level Tessera IR for the shared DiffusionTransformer backbone.
//
// Represents one transformer block of the small (~117M) MDLM variant:
//   hidden_size=768, num_heads=12, num_kv_heads=12, head_dim=64,
//   intermediate_size=3072, seq_len=512, batch=2
//
// Key differences from an autoregressive model:
//   • flash_attn op has causal=false  (all tokens attend to all tokens)
//   • time conditioning via adaLN (scale+shift from sinusoidal time embed)
//   • no KV-cache — full bidirectional forward at every denoising step

module @diffusion_transformer_block attributes {
  tessera.model = "mdlm_small",
  tessera.variant = "masked_discrete_diffusion"
} {

  // -------------------------------------------------------------------------
  // Sinusoidal time embedding + MLP projection
  // -------------------------------------------------------------------------
  func.func @time_embed(
    %t     : tensor<2xi64>,          // (B,) integer timestep
    %sin_w : tensor<1x384xf32>       // sinusoidal frequency matrix
  ) -> tensor<2x768xf32>
  attributes { tessera.op = "tessera.elementwise.time_embed" }
  {
    // sin/cos embedding → 768-dim via two Linear layers
    %sin = tessera.elementwise.sincos(%t, %sin_w)
      : (tensor<2xi64>, tensor<1x384xf32>) -> tensor<2x768xf32>
    %cond = tessera.matmul(%sin, @time_mlp_w1)
      : (tensor<2x768xf32>, tensor<768x768xf32>) -> tensor<2x768xf32>
    return %cond : tensor<2x768xf32>
  }

  // -------------------------------------------------------------------------
  // adaLN modulation: (scale, shift) from conditioning vector
  // -------------------------------------------------------------------------
  func.func @adaLN_mod(
    %cond  : tensor<2x768xf32>,      // (B, H) time conditioning
    %w_mod : tensor<768x1536xf32>    // Linear → 2×H outputs
  ) -> (tensor<2x1x768xf32>, tensor<2x1x768xf32>)
  attributes { tessera.op = "tessera.elementwise.adaln_mod" }
  {
    %out   = tessera.matmul(%cond, %w_mod)
      : (tensor<2x768xf32>, tensor<768x1536xf32>) -> tensor<2x1536xf32>
    %scale = tessera.slice(%out, [0, 0], [2, 768])
      : tensor<2x1536xf32> -> tensor<2x768xf32>
    %shift = tessera.slice(%out, [0, 768], [2, 1536])
      : tensor<2x1536xf32> -> tensor<2x768xf32>
    // Expand to (B, 1, H) for broadcast with (B, T, H)
    %scale_b = tessera.reshape(%scale, [2, 1, 768]) : tensor<2x768xf32> -> tensor<2x1x768xf32>
    %shift_b = tessera.reshape(%shift, [2, 1, 768]) : tensor<2x768xf32> -> tensor<2x1x768xf32>
    return %scale_b, %shift_b : tensor<2x1x768xf32>, tensor<2x1x768xf32>
  }

  // -------------------------------------------------------------------------
  // Bidirectional multi-head self-attention (non-causal)
  // -------------------------------------------------------------------------
  func.func @diffusion_attn(
    %x     : tensor<2x512x768xf32>,  // (B, T, H) after adaLN
    %wq    : tensor<768x768xf32>,    // Q projection weight
    %wk    : tensor<768x768xf32>,    // K projection weight
    %wv    : tensor<768x768xf32>,    // V projection weight
    %wo    : tensor<768x768xf32>     // output projection weight
  ) -> tensor<2x512x768xf32>
  attributes {
    tessera.op           = "tessera.flash_attn",
    tessera.causal       = false,              // BIDIRECTIONAL — no causal mask
    tessera.num_heads    = 12 : i64,
    tessera.num_kv_heads = 12 : i64,
    tessera.head_dim     = 64 : i64,
    tessera.batch        = 2 : i64,
    tessera.seq_len      = 512 : i64,
    // Tessera sharding: heads → tp axis, batch → dp axis
    tessera.shard        = { q_heads = "tp", kv_heads = "tp", batch = "dp" }
  }
  {
    %q = tessera.matmul(%x, %wq) { shard_k = "tp" }
      : (tensor<2x512x768xf32>, tensor<768x768xf32>) -> tensor<2x512x768xf32>
    %k = tessera.matmul(%x, %wk) { shard_k = "tp" }
      : (tensor<2x512x768xf32>, tensor<768x768xf32>) -> tensor<2x512x768xf32>
    %v = tessera.matmul(%x, %wv) { shard_k = "tp" }
      : (tensor<2x512x768xf32>, tensor<768x768xf32>) -> tensor<2x512x768xf32>

    // Flash attention: causal=false → full NxN attention matrix
    %attn_out = tessera.flash_attn(%q, %k, %v) {
      causal      = false,
      num_heads   = 12 : i64,
      head_dim    = 64 : i64,
      softmax_scale = 0.125 : f32   // 1/sqrt(64)
    } : (tensor<2x512x768xf32>, tensor<2x512x768xf32>, tensor<2x512x768xf32>)
      -> tensor<2x512x768xf32>

    %out = tessera.matmul(%attn_out, %wo)
      : (tensor<2x512x768xf32>, tensor<768x768xf32>) -> tensor<2x512x768xf32>
    return %out : tensor<2x512x768xf32>
  }

  // -------------------------------------------------------------------------
  // SwiGLU MLP
  // -------------------------------------------------------------------------
  func.func @diffusion_mlp(
    %x    : tensor<2x512x768xf32>,
    %wg   : tensor<768x3072xf32>,   // gate projection
    %wu   : tensor<768x3072xf32>,   // up projection
    %wd   : tensor<3072x768xf32>    // down projection
  ) -> tensor<2x512x768xf32>
  attributes {
    tessera.op = "tessera.elementwise.mlp_gate",
    tessera.shard = { intermediate = "tp" }
  }
  {
    %gate = tessera.matmul(%x,  %wg) { shard_n = "tp" }
      : (tensor<2x512x768xf32>, tensor<768x3072xf32>) -> tensor<2x512x3072xf32>
    %up   = tessera.matmul(%x,  %wu) { shard_n = "tp" }
      : (tensor<2x512x768xf32>, tensor<768x3072xf32>) -> tensor<2x512x3072xf32>
    %silu = tessera.elementwise.silu(%gate)
      : tensor<2x512x3072xf32> -> tensor<2x512x3072xf32>
    %gated = tessera.elementwise.mul(%silu, %up)
      : (tensor<2x512x3072xf32>, tensor<2x512x3072xf32>) -> tensor<2x512x3072xf32>
    %out  = tessera.matmul(%gated, %wd) { shard_k = "tp" }
      : (tensor<2x512x3072xf32>, tensor<3072x768xf32>) -> tensor<2x512x768xf32>
    return %out : tensor<2x512x768xf32>
  }

  // -------------------------------------------------------------------------
  // Full transformer block (one MDLM denoising step)
  // -------------------------------------------------------------------------
  func.func @diffusion_block(
    %x    : tensor<2x512x768xf32>,    // (B, T, H) input
    %cond : tensor<2x768xf32>,        // (B, H) time conditioning
    // adaLN weight matrices
    %w_adaln_attn : tensor<768x1536xf32>,
    %w_adaln_mlp  : tensor<768x1536xf32>,
    // RMSNorm weights
    %norm1_w : tensor<768xf32>,
    %norm2_w : tensor<768xf32>,
    // Attention weights
    %wq : tensor<768x768xf32>,
    %wk : tensor<768x768xf32>,
    %wv : tensor<768x768xf32>,
    %wo : tensor<768x768xf32>,
    // MLP weights
    %wg : tensor<768x3072xf32>,
    %wu : tensor<768x3072xf32>,
    %wd : tensor<3072x768xf32>
  ) -> tensor<2x512x768xf32>
  {
    // --- Attention sub-layer ---
    %scale_a, %shift_a = call @adaLN_mod(%cond, %w_adaln_attn)
      : (tensor<2x768xf32>, tensor<768x1536xf32>)
      -> (tensor<2x1x768xf32>, tensor<2x1x768xf32>)

    %h1 = tessera.elementwise.rmsnorm(%x, %norm1_w) {
      eps = 1.0e-6 : f32
    } : (tensor<2x512x768xf32>, tensor<768xf32>) -> tensor<2x512x768xf32>

    // adaLN: h = scale * norm(x) + shift
    %h1_mod = tessera.elementwise.add(
                tessera.elementwise.mul(%scale_a, %h1),
                %shift_a)
      : tensor<2x512x768xf32>

    %attn_out = call @diffusion_attn(%h1_mod, %wq, %wk, %wv, %wo)
      : (tensor<2x512x768xf32>, tensor<768x768xf32>, tensor<768x768xf32>,
         tensor<768x768xf32>, tensor<768x768xf32>)
      -> tensor<2x512x768xf32>

    %x1 = tessera.elementwise.add(%x, %attn_out)
      : (tensor<2x512x768xf32>, tensor<2x512x768xf32>) -> tensor<2x512x768xf32>

    // --- MLP sub-layer ---
    %scale_m, %shift_m = call @adaLN_mod(%cond, %w_adaln_mlp)
      : (tensor<2x768xf32>, tensor<768x1536xf32>)
      -> (tensor<2x1x768xf32>, tensor<2x1x768xf32>)

    %h2 = tessera.elementwise.rmsnorm(%x1, %norm2_w) {
      eps = 1.0e-6 : f32
    } : (tensor<2x512x768xf32>, tensor<768xf32>) -> tensor<2x512x768xf32>

    %h2_mod = tessera.elementwise.add(
                tessera.elementwise.mul(%scale_m, %h2),
                %shift_m)
      : tensor<2x512x768xf32>

    %mlp_out = call @diffusion_mlp(%h2_mod, %wg, %wu, %wd)
      : (tensor<2x512x768xf32>, tensor<768x3072xf32>, tensor<768x3072xf32>,
         tensor<3072x768xf32>)
      -> tensor<2x512x768xf32>

    %x2 = tessera.elementwise.add(%x1, %mlp_out)
      : (tensor<2x512x768xf32>, tensor<2x512x768xf32>) -> tensor<2x512x768xf32>

    return %x2 : tensor<2x512x768xf32>
  }

}
