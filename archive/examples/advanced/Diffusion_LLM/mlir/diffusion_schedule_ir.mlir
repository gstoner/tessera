// diffusion_schedule_ir.mlir
//
// Schedule-level Tessera IR for the DiffusionTransformer backbone.
//
// Target: dual-GPU tensor-parallel (tp=2) on H100 SXM (sm_90).
// Batch parallelism dp=1 (single node; extend dp for multi-node).
//
// Schedule decisions:
//   • Bidirectional flash-attention:  BM=128, BN=128, causal=false
//   • MLP gate/up projections:        tile 128×128 WGMMA, split across tp=2
//   • adaLN modulation:               fused elementwise kernel
//   • All-reduce on tp axis:          after attention output + MLP down proj

module @diffusion_schedule attributes {
  tessera.target  = "sm_90",
  tessera.backend = "cuda",
  tessera.mesh    = "sdy.mesh<[\"tp\"=2, \"dp\"=1]>"
} {

  // =========================================================================
  // Mesh & sharding declarations
  // =========================================================================

  sdy.mesh @mesh = <["tp"=2, "dp"=1]>

  // Attention output tensor is sharded along the head dimension on tp
  sdy.sharding @q_sharding   = <@mesh, [{"dp"}, {"seq"}, {"tp"}]>
  sdy.sharding @kv_sharding  = <@mesh, [{"dp"}, {"seq"}, {"tp"}]>
  sdy.sharding @mlp_i_sharding = <@mesh, [{"dp"}, {"seq"}, {"tp"}]>

  // =========================================================================
  // Bidirectional flash-attention schedule
  // =========================================================================

  tessera.schedule @bidir_flash_attn {
    // Block sizes chosen for sm_90 H100 (80 GB HBM, 50 MB L2)
    tile.block_m = 128 : i64,
    tile.block_n = 128 : i64,
    tile.block_k = 64  : i64,

    // Non-causal — compute full NxN blocks; no row/col masking needed
    flash_attn.causal       = false,
    flash_attn.softmax_scale = 0.125 : f32,

    // WGMMA instruction on sm_90
    wgmma.m = 64 : i64,
    wgmma.n = 64 : i64,
    wgmma.k = 16 : i64,

    // Pipeline depth for async SMEM loads (TMA)
    pipeline.depth = 4 : i64,

    // Tensor-parallel: split heads across tp=2
    // Each device handles 6 out of 12 heads
    tp.heads_per_device = 6 : i64,
    tp.all_reduce_after = true
  }

  // =========================================================================
  // SwiGLU MLP schedule
  // =========================================================================

  tessera.schedule @swiglu_mlp {
    // Gate and up projections fused (both read same input)
    tile.block_m   = 128 : i64,
    tile.block_n   = 128 : i64,
    tile.block_k   = 64  : i64,

    // Each tp shard handles 1536 out of 3072 intermediate channels
    tp.intermediate_per_device = 1536 : i64,

    // Fused gate*silu(up) elementwise — no extra SMEM traffic
    fuse.gate_activation = true,

    // Down projection: K=3072 split across tp; need all-reduce at end
    down_proj.all_reduce = true,

    pipeline.depth = 2 : i64
  }

  // =========================================================================
  // adaLN modulation schedule (small fused kernel)
  // =========================================================================

  tessera.schedule @adaLN_fused {
    // Linear (B,H)→(B,2H) followed by broadcast multiply + add with (B,T,H)
    // Small enough to fuse into a single persistent kernel
    fuse.linear_then_broadcast = true,
    tile.threads_per_block     = 256 : i64
  }

  // =========================================================================
  // RMSNorm schedule
  // =========================================================================

  tessera.schedule @rmsnorm {
    tessera.op = "tessera.elementwise.rmsnorm",
    // One warp per row (hidden_size=768 fits in one warp pass with f32 accum)
    tile.rows_per_block  = 32 : i64,
    tile.threads_per_row = 32 : i64,
    compute.accumulate   = "f32"
  }

  // =========================================================================
  // Time embedding + MLP schedule
  // =========================================================================

  tessera.schedule @time_embed_mlp {
    // Sinusoidal embed is tiny (B=2, dim=768); run on one thread block
    tile.threads_per_block = 128 : i64,
    // The MLP (2×Linear) is batched with the conditioning but bandwidth-
    // bound; exploit L1 locality by keeping weights resident
    l1.pin_weights = true
  }

  // =========================================================================
  // LMSE (output logit projection, vocabulary = 50257)
  // lm_head is weight-tied to embed_tokens — same schedule applies
  // =========================================================================

  tessera.schedule @lm_head {
    tile.block_m = 128 : i64,
    tile.block_n = 256 : i64,   // vocab is large; wider N tiles amortise loads
    tile.block_k = 64  : i64,
    pipeline.depth = 2 : i64
  }

}
