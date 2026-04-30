// ===- gemma_graph_ir.mlir — Tessera Graph IR for one Gemma 4 decoder layer =====//
//
// Represents a SINGLE Gemma 4 decoder block at the Tessera Graph IR level.
//
// Compiler pipeline to lower this file:
//   tessera-opt gemma_graph_ir.mlir \
//     --alias=graph-to-schedule      \
//     --arch=sm_90 --platform=cuda   \
//     -o gemma_schedule.mlir
//
// Shapes here use symbolic dimensions:
//   B  = batch size
//   T  = sequence length (or 1 for decode step)
//   H  = hidden_size (e.g., 2560 for Gemma4-4B)
//   I  = intermediate_size (e.g., 10240)
//   Hq = num_attention_heads * head_dim (= H for Gemma4)
//   Hkv= num_kv_heads * head_dim
//
// ===--------------------------------------------------------------------------===//

// mesh definition for 2-way tensor-parallel × 2-way data-parallel
// (used by schedule-level sharding annotations below)
// #sdy.mesh<["tp"=2, "dp"=2]>

module @gemma4_decoder_block {

  // -------------------------------------------------------------------------
  // Embedded RMSNorm: x_norm = weight * (x / rms(x, eps))
  // Lowered to tessera.elementwise with a fused rsqrt + mul epilogue.
  // -------------------------------------------------------------------------
  func.func @input_layernorm(
      %x      : tensor<?x?x2560xf16>,   // (B, T, H)
      %weight : tensor<2560xf16>         // (H,)
  ) -> tensor<?x?x2560xf16> {
    %norm = "tessera.elementwise"(%x, %weight) {
      op     = "rmsnorm",
      eps    = 1.0e-6 : f32,
      axis   = -1 : i32,
      // Tessera compiler: lower to fused warp-shuffle reduction kernel
      tile_shape = [1, 64, 2560],
    } : (tensor<?x?x2560xf16>, tensor<2560xf16>) -> tensor<?x?x2560xf16>
    return %norm : tensor<?x?x2560xf16>
  }

  // -------------------------------------------------------------------------
  // Grouped-query attention (Gemma4-4B: H=16, Hkv=8, head_dim=256).
  //
  // Full causal for even layers; sliding-window (W=4096) for odd layers.
  // Represented here as the "full" variant — the schedule pass inserts
  // the window mask annotation based on layer_idx.
  // -------------------------------------------------------------------------
  func.func @self_attention(
      %x       : tensor<?x?x2560xf16>,   // (B, T, hidden)
      %wq      : tensor<4096x2560xf16>,  // (num_heads*head_dim, H) = (16*256, H)
      %wk      : tensor<2048x2560xf16>,  // (num_kv_heads*head_dim, H) = (8*256, H)
      %wv      : tensor<2048x2560xf16>,  //  same
      %wo      : tensor<2560x4096xf16>,  // (H, num_heads*head_dim)
      %rope_cos: tensor<1x?x1x128xf16>,  // (1, T, 1, head_dim//2)
      %rope_sin: tensor<1x?x1x128xf16>
  ) -> tensor<?x?x2560xf16> {

    // Q projection: (B,T,H) × (H, Hq)^T → (B,T,Hq)
    %q_flat = "tessera.matmul"(%x, %wq) {
      transpose_b  = true,
      // shard Q heads across tensor-parallel axis
      tessera.shard = #tessera.shard<kind="block", axes=["tp"], dims=[2]>,
    } : (tensor<?x?x2560xf16>, tensor<4096x2560xf16>) -> tensor<?x?x4096xf16>

    // K projection
    %k_flat = "tessera.matmul"(%x, %wk) {
      transpose_b  = true,
      tessera.shard = #tessera.shard<kind="block", axes=["tp"], dims=[2]>,
    } : (tensor<?x?x2560xf16>, tensor<2048x2560xf16>) -> tensor<?x?x2048xf16>

    // V projection
    %v_flat = "tessera.matmul"(%x, %wv) {
      transpose_b  = true,
      tessera.shard = #tessera.shard<kind="block", axes=["tp"], dims=[2]>,
    } : (tensor<?x?x2560xf16>, tensor<2048x2560xf16>) -> tensor<?x?x2048xf16>

    // Reshape to (B, T, num_heads, head_dim) — shapes resolved by compiler
    // In Graph IR we annotate the head/dim split; the reshape is implicit.

    // Flash attention: (B,T,Hq,256) × (B,T,Hkv,256) → (B,T,Hq,256)
    // head_dim=256 is Gemma4-specific; Tessera will tile along seq & head dims.
    %attn_out = "tessera.flash_attn"(%q_flat, %k_flat, %v_flat) {
      num_heads    = 16 : i32,
      num_kv_heads = 8  : i32,
      head_dim     = 256 : i32,
      causal       = true,
      // sliding_window = 0 means full attention (even layers)
      // sliding_window = 4096 set by schedule pass for odd layers
      sliding_window = 0 : i32,
      scale        = 0.0625 : f32,  // 1/sqrt(256)
      dtype        = f16,
      // Tessera tile config; overridden per-arch by autotune
      tile_m = 64 : i32,
      tile_n = 64 : i32,
      stages = 2  : i32,
    } : (tensor<?x?x4096xf16>, tensor<?x?x2048xf16>, tensor<?x?x2048xf16>)
        -> tensor<?x?x4096xf16>

    // Output projection: (B,T,Hq) × (Hq,H)^T → (B,T,H)
    %out = "tessera.matmul"(%attn_out, %wo) {
      transpose_b = true,
      // All-reduce after shard-local matmul (tensor-parallel)
      tessera.shard = #tessera.shard<kind="replicated", axes=["tp"], dims=[2]>,
    } : (tensor<?x?x4096xf16>, tensor<2560x4096xf16>) -> tensor<?x?x2560xf16>

    return %out : tensor<?x?x2560xf16>
  }

  // -------------------------------------------------------------------------
  // GeGLU MLP (Gemma 4 uses GELU gating, not SiLU)
  //
  //   gate = GELU(gate_proj(x))
  //   up   = up_proj(x)
  //   y    = down_proj(gate * up)
  // -------------------------------------------------------------------------
  func.func @gemma_mlp(
      %x         : tensor<?x?x2560xf16>,   // (B, T, H)
      %wgate     : tensor<10240x2560xf16>, // (I, H)
      %wup       : tensor<10240x2560xf16>, // (I, H)
      %wdown     : tensor<2560x10240xf16>  // (H, I)
  ) -> tensor<?x?x2560xf16> {

    // Gate projection + GELU
    %gate_pre = "tessera.matmul"(%x, %wgate) {
      transpose_b = true,
      tessera.shard = #tessera.shard<kind="block", axes=["tp"], dims=[2]>,
    } : (tensor<?x?x2560xf16>, tensor<10240x2560xf16>) -> tensor<?x?x10240xf16>

    %gate = "tessera.elementwise"(%gate_pre) {
      op   = "gelu_tanh",  // GELU with tanh approximation
      tile_shape = [1, 64, 512],
    } : (tensor<?x?x10240xf16>) -> tensor<?x?x10240xf16>

    // Up projection
    %up = "tessera.matmul"(%x, %wup) {
      transpose_b = true,
      tessera.shard = #tessera.shard<kind="block", axes=["tp"], dims=[2]>,
    } : (tensor<?x?x2560xf16>, tensor<10240x2560xf16>) -> tensor<?x?x10240xf16>

    // Element-wise gate × up
    %hidden = "tessera.elementwise"(%gate, %up) {
      op   = "mul",
      tile_shape = [1, 64, 512],
    } : (tensor<?x?x10240xf16>, tensor<?x?x10240xf16>) -> tensor<?x?x10240xf16>

    // Down projection + all-reduce
    %y = "tessera.matmul"(%hidden, %wdown) {
      transpose_b = true,
      tessera.shard = #tessera.shard<kind="replicated", axes=["tp"], dims=[2]>,
    } : (tensor<?x?x10240xf16>, tensor<2560x10240xf16>) -> tensor<?x?x2560xf16>

    return %y : tensor<?x?x2560xf16>
  }

  // -------------------------------------------------------------------------
  // Full decoder block (pre-norm, residual)
  // -------------------------------------------------------------------------
  func.func @decoder_block(
      %x          : tensor<?x?x2560xf16>,
      %w_norm1    : tensor<2560xf16>,
      %wq, %wk, %wv, %wo : tensor<4096x2560xf16>,  // simplified; wk,wv are 2048x2560
      %w_norm2    : tensor<2560xf16>,
      %wgate, %wup: tensor<10240x2560xf16>,
      %wdown      : tensor<2560x10240xf16>,
      %rope_cos   : tensor<1x?x1x128xf16>,
      %rope_sin   : tensor<1x?x1x128xf16>
  ) -> tensor<?x?x2560xf16> {
    // Pre-norm → attention → residual
    %n1    = func.call @input_layernorm(%x, %w_norm1) : (tensor<?x?x2560xf16>, tensor<2560xf16>) -> tensor<?x?x2560xf16>
    %attn  = func.call @self_attention(%n1, %wq, %wk, %wv, %wo, %rope_cos, %rope_sin)
             : (tensor<?x?x2560xf16>, tensor<4096x2560xf16>, tensor<2048x2560xf16>,
                tensor<2048x2560xf16>, tensor<2560x4096xf16>,
                tensor<1x?x1x128xf16>, tensor<1x?x1x128xf16>) -> tensor<?x?x2560xf16>
    %r1    = "tessera.elementwise"(%x, %attn)  { op = "add" }
             : (tensor<?x?x2560xf16>, tensor<?x?x2560xf16>) -> tensor<?x?x2560xf16>

    // Pre-norm → MLP → residual
    %n2    = func.call @input_layernorm(%r1, %w_norm2) : (tensor<?x?x2560xf16>, tensor<2560xf16>) -> tensor<?x?x2560xf16>
    %mlp   = func.call @gemma_mlp(%n2, %wgate, %wup, %wdown)
             : (tensor<?x?x2560xf16>, tensor<10240x2560xf16>,
                tensor<10240x2560xf16>, tensor<2560x10240xf16>) -> tensor<?x?x2560xf16>
    %r2    = "tessera.elementwise"(%r1, %mlp)  { op = "add" }
             : (tensor<?x?x2560xf16>, tensor<?x?x2560xf16>) -> tensor<?x?x2560xf16>

    return %r2 : tensor<?x?x2560xf16>
  }

} // module @gemma4_decoder_block
