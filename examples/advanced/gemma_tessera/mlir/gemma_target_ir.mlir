// ===- gemma_target_ir.mlir — Tessera Target IR for Gemma4 attention =========//
//
// Produced by tessera-opt --alias=schedule-to-tile → tile-to-target passes.
// Targets sm_90 (Hopper) with WGMMA + TMA.
//
// This level shows the tile-level operations that correspond to one CTA's
// work for the flash-attention kernel.  It is illustrative — a real
// lowering would be done by the Tessera tile-to-target pass pipeline.
//
// ===--------------------------------------------------------------------------===//

module @gemma4_target_sm90 {

  // =========================================================================
  // Flash attention tile kernel (one CTA, Hopper WGMMA path)
  // =========================================================================
  "tile.kernel"() {
    name = "gemma4_flash_attn_sm90",
    arch = "sm_90",
    // CTA shape
    threads_per_cta = 128 : i32,   // 4 warps (warpgroup for WGMMA)
    cta_m = 64 : i32,
    cta_n = 64 : i32,
    // Pipeline
    stages = 2 : i32,
    smem_bytes = 98304 : i32,      // 96 KB shared memory (within Hopper limit)
  } ({

    // --- Shared memory descriptors -----------------------------------------
    // Q tile: (cta_m, head_dim) = (64, 256) × 2 bytes = 32 KB per stage × 2 stages
    %smem_q = "tile.smem_layout"() {
      shape      = [64, 256],
      dtype      = f16,
      swizzle    = "128b",           // 128-byte swizzle for bank-conflict avoidance
      n_stages   = 2 : i32,
    } : () -> !tile.smem_desc

    // KV tile: (cta_n, head_dim) = (64, 256) × 2 bytes = 32 KB × 2 stages × 2 (K+V)
    %smem_k = "tile.smem_layout"() {
      shape  = [64, 256], dtype = f16, swizzle = "128b", n_stages = 2 : i32,
    } : () -> !tile.smem_desc
    %smem_v = "tile.smem_layout"() {
      shape  = [64, 256], dtype = f16, swizzle = "128b", n_stages = 2 : i32,
    } : () -> !tile.smem_desc

    // --- Accumulator registers (held in warpgroup RF) -----------------------
    %acc = "tile.warp_config"() {
      kind    = "accumulator",
      shape   = [64, 256],
      dtype   = f32,               // accumulate in f32, convert to f16 for output
      wgmma   = true,
    } : () -> !tile.reg_tile

    // --- TMA descriptors for global-memory loads ---------------------------
    %tma_q = "tile.tma.descriptor"() {
      rank      = 2 : i32,
      box_shape = [64, 256],       // tile footprint in global mem
      dtype     = f16,
      swizzle   = "128b",
    } : () -> !tile.tma_desc

    %tma_k = "tile.tma.descriptor"() {
      rank = 2 : i32, box_shape = [64, 256], dtype = f16, swizzle = "128b",
    } : () -> !tile.tma_desc
    %tma_v = "tile.tma.descriptor"() {
      rank = 2 : i32, box_shape = [64, 256], dtype = f16, swizzle = "128b",
    } : () -> !tile.tma_desc

    // --- Main loop body (over KV tiles) ------------------------------------
    // Pseudo-structure:
    //
    //  for kv_tile in range(0, T, cta_n):
    //    stage = kv_tile % 2                      // double-buffer index
    //    async_copy Q[q_tile, :] → smem_q[stage]  // TMA load (comm stream 0)
    //    async_copy K[kv_tile, :] → smem_k[stage]
    //    async_copy V[kv_tile, :] → smem_v[stage]
    //    mbarrier.arrive_and_wait                  // sync
    //
    //    S  = wgmma(Q_reg, K_smem[stage]) * scale // (cta_m, cta_n) in f32
    //    S  = causal_mask(S, q_tile, kv_tile)      // upper-tri masking
    //    m_new = max(m, row_max(S))
    //    p    = exp(S - m_new)
    //    l    = exp(m - m_new) * l + row_sum(p)
    //    O    = exp(m - m_new) * O + wgmma(p_fp16, V_smem[stage])
    //    m    = m_new
    //
    //  O = O / l                                   // normalize
    //  store O → global via TMA

    "tile.loop_body_placeholder"() {
      comment = "see above pseudo-structure — lowered by tile-to-target pass"
    } : () -> ()

  }) : () -> ()

  // =========================================================================
  // GEMM tile kernel for Q/K/V projections (128×128×64 tiles, sm_90)
  // =========================================================================
  "tile.kernel"() {
    name = "gemma4_gemm_qkv_sm90",
    arch = "sm_90",
    threads_per_cta = 128 : i32,
    cta_m = 128 : i32,
    cta_n = 128 : i32,
    stages = 4 : i32,
    smem_bytes = 65536 : i32,   // 64 KB
  } ({

    %smem_a = "tile.smem_layout"() {
      shape = [64, 128], dtype = f16, swizzle = "128b", n_stages = 4 : i32,
    } : () -> !tile.smem_desc

    %smem_b = "tile.smem_layout"() {
      shape = [128, 64], dtype = f16, swizzle = "128b", n_stages = 4 : i32,
    } : () -> !tile.smem_desc

    %acc_gemm = "tile.warp_config"() {
      kind = "accumulator", shape = [128, 128], dtype = f32, wgmma = true,
    } : () -> !tile.reg_tile

    %tma_a = "tile.tma.descriptor"() {
      rank = 2 : i32, box_shape = [64, 128], dtype = f16, swizzle = "128b",
    } : () -> !tile.tma_desc

    %tma_b = "tile.tma.descriptor"() {
      rank = 2 : i32, box_shape = [128, 64], dtype = f16, swizzle = "128b",
    } : () -> !tile.tma_desc

    "tile.loop_body_placeholder"() {
      comment = "persistent kernel: stream K tiles with 4-stage SMEM pipeline"
    } : () -> ()

  }) : () -> ()

} // module @gemma4_target_sm90
