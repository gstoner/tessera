// ===- gemma_schedule_ir.mlir — Tessera Schedule IR for Gemma4 decoder =====//
//
// Produced by running gemma_graph_ir.mlir through the graph-to-schedule pass:
//
//   tessera-opt gemma_graph_ir.mlir --alias=graph-to-schedule -o gemma_schedule.mlir
//
// This level adds:
//   • Mesh definition and tensor sharding attributes (Shardy)
//   • Pipeline schedule (stages, overlap config)
//   • Layer-type annotations (SWA vs full per layer)
//   • Numeric tile parameters resolved per arch (sm_90 defaults shown)
//
// ===--------------------------------------------------------------------------===//

module @gemma4_schedule
    // Mesh: 2 tensor-parallel × 1 data-parallel (single H100)
    attributes {
      "sdy.mesh" = #sdy.mesh<["tp"=2, "dp"=1]>,
      "schedule.pipeline" = {
        stages = 2 : i32,
        overlap = "comm_compute"
      }
    } {

  // -------------------------------------------------------------------------
  // Flash attention — schedule-level view with tile + stage annotation
  // -------------------------------------------------------------------------
  func.func @flash_attn_scheduled(
      %q  : tensor<?x?x4096xf16>  {sdy.sharding = #sdy.sharding<@mesh, [{"dp"}, {}, {"tp"}]>},
      %k  : tensor<?x?x2048xf16>  {sdy.sharding = #sdy.sharding<@mesh, [{"dp"}, {}, {"tp"}]>},
      %v  : tensor<?x?x2048xf16>  {sdy.sharding = #sdy.sharding<@mesh, [{"dp"}, {}, {"tp"}]>}
  ) -> (tensor<?x?x4096xf16>  {sdy.sharding = #sdy.sharding<@mesh, [{"dp"}, {}, {"tp"}]>})
    attributes {
      // Flash attention schedule parameters (sm_90 / H100 defaults)
      "schedule.attn" = {
        num_heads    = 16 : i32,
        num_kv_heads = 8  : i32,
        head_dim     = 256 : i32,
        tile_q       = 64  : i32,   // output-tile rows (seq dim)
        tile_kv      = 64  : i32,   // key-tile rows
        pipeline_stages = 2 : i32,  // double-buffer SMEM pipeline
        wgmma        = true,
        tma_load     = true,
        cluster_m    = 2 : i32,     // 2-CTA cluster across M
        // sliding_window: set by pass; 0 = full, 4096 = SWA odd layers
        sliding_window = 0 : i32,
      }
    }
  {
    %out = "tessera.flash_attn"(%q, %k, %v) {
      num_heads    = 16 : i32,
      num_kv_heads = 8  : i32,
      head_dim     = 256 : i32,
      causal       = true,
      sliding_window = 0 : i32,
      scale        = 0.0625 : f32,
      tile_m = 64 : i32, tile_n = 64 : i32, stages = 2 : i32,
    } : (tensor<?x?x4096xf16>, tensor<?x?x2048xf16>, tensor<?x?x2048xf16>)
        -> tensor<?x?x4096xf16>
    return %out : tensor<?x?x4096xf16>
  }

  // -------------------------------------------------------------------------
  // GEMM schedule — used for Q/K/V projections and down projection
  // -------------------------------------------------------------------------
  func.func @gemm_scheduled(
      %a : tensor<?x?x2560xf16>  {sdy.sharding = #sdy.sharding<@mesh, [{"dp"}, {}, {}]>},
      %b : tensor<4096x2560xf16> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"tp"}]>}
  ) -> (tensor<?x?x4096xf16>  {sdy.sharding = #sdy.sharding<@mesh, [{"dp"}, {}, {"tp"}]>})
    attributes {
      "schedule.gemm" = {
        tile_m   = 128 : i32,
        tile_n   = 128 : i32,
        tile_k   = 64  : i32,
        stages   = 4   : i32,    // 4-stage SMEM pipeline (Hopper optimal)
        wgmma    = true,
        tma_load = true,
        // Cluster dims for Persistent Kernel (sm_90)
        cluster_m = 1 : i32,
        cluster_n = 1 : i32,
      }
    }
  {
    %out = "tessera.matmul"(%a, %b) {
      transpose_b = true,
      tessera.shard = #tessera.shard<kind="block", axes=["tp"], dims=[2]>,
    } : (tensor<?x?x2560xf16>, tensor<4096x2560xf16>) -> tensor<?x?x4096xf16>
    return %out : tensor<?x?x4096xf16>
  }

  // -------------------------------------------------------------------------
  // Mesh definition op (produced by tessera-opt graph-to-schedule pass)
  // -------------------------------------------------------------------------
  "schedule.mesh.define"() {
    name  = "default_mesh",
    axes  = ["tp", "dp"],
    shape = [2, 1],
  } : () -> ()

} // module @gemma4_schedule
