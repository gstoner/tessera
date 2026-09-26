// RUN: tessera-opt --tessera-graph-to-schedule %s 2>/dev/null | FileCheck %s --check-prefix=SCHED
// RUN: tessera-opt --tessera-graph-to-schedule --tessera-schedule-to-tile %s 2>/dev/null | FileCheck %s --check-prefix=TILE
// RUN: tessera-opt --tessera-graph-to-schedule %s -o /dev/null 2>&1 | FileCheck %s --check-prefix=REMARK
//
// ROCM-SPLIT-K-1: the Graph->Schedule decision. `selectGfx1201SplitK` is the
// one decider (Decision #31; `rocm_tiling.select_split_k` is its oracle): split
// when the problem yields fewer output tiles than gfx1201's 32 WGPs, into a
// power-of-two slice count whose slices are whole macro K blocks of >= 256.
//
//   @router       16x256x2048 f16  -> 16 tiles < 32 -> split_k = 2, ordered
//   @router_fused same, bias+gelu  -> split; the Tile op still states the
//                                     PROGRAM's epilogue (the reduce applies it)
//   @ragged_k     K = 2050         -> occupancy asks, no aligned split exists:
//                                     unsplit, and a ROCM_SPLIT_K_NOT_APPLIED
//                                     remark says so (never silent, #21a)
//   @wide         1024^2 x 2048    -> 256 tiles: not occupancy-short, no split
//
// The last two are the negative fixtures Decision #10a requires.

module attributes {tessera.target = "rocm", tessera.arch = "gfx1201"} {
  func.func @router(%a: tensor<16x2048xf16>, %b: tensor<2048x256xf16>) -> tensor<16x256xf32> {
    %0 = tessera.matmul %a, %b : (tensor<16x2048xf16>, tensor<2048x256xf16>) -> tensor<16x256xf32>
    return %0 : tensor<16x256xf32>
  }
  func.func @router_fused(%a: tensor<16x2048xbf16>, %b: tensor<2048x256xbf16>, %bias: tensor<256xf32>) -> tensor<16x256xf32> {
    %0 = tessera.matmul %a, %b, %bias {bias = "bias", activation = "gelu"} : (tensor<16x2048xbf16>, tensor<2048x256xbf16>, tensor<256xf32>) -> tensor<16x256xf32>
    return %0 : tensor<16x256xf32>
  }
  func.func @ragged_k(%a: tensor<16x2050xf16>, %b: tensor<2050x256xf16>) -> tensor<16x256xf32> {
    %0 = tessera.matmul %a, %b : (tensor<16x2050xf16>, tensor<2050x256xf16>) -> tensor<16x256xf32>
    return %0 : tensor<16x256xf32>
  }
  func.func @wide(%a: tensor<1024x2048xf16>, %b: tensor<2048x1024xf16>) -> tensor<1024x1024xf32> {
    %0 = tessera.matmul %a, %b : (tensor<1024x2048xf16>, tensor<2048x1024xf16>) -> tensor<1024x1024xf32>
    return %0 : tensor<1024x1024xf32>
  }
}

// SCHED-LABEL: func.func @router
// SCHED: schedule.matmul
// SCHED-SAME: block_k = 32 : i64
// SCHED-SAME: split_k = 2 : i64
// SCHED-SAME: split_k_reduction = "ordered"
// SCHED-LABEL: func.func @router_fused
// SCHED: schedule.matmul
// SCHED-SAME: activation = "gelu"
// SCHED-SAME: bias = true
// SCHED-SAME: split_k = 2 : i64
// SCHED-SAME: split_k_reduction = "ordered"
// SCHED-LABEL: func.func @ragged_k
// SCHED: schedule.matmul
// SCHED-NOT: split_k
// SCHED-LABEL: func.func @wide
// SCHED: schedule.matmul
// SCHED-NOT: split_k

// TILE-LABEL: func.func @router
// TILE: tile.matmul_kernel
// TILE-SAME: k_blocks = 2
// TILE-SAME: tessera.split_k = 2 : i64
// TILE-SAME: tessera.split_k_reduction = "ordered"
// TILE-LABEL: func.func @router_fused
// TILE: tile.matmul_kernel
// TILE-SAME: epilogue = #tile.epilogue<bias = true, activation = "gelu", output = "f32">
// TILE-SAME: tessera.split_k = 2 : i64
// TILE-LABEL: func.func @ragged_k
// TILE: tile.matmul_kernel
// TILE-NOT: tessera.split_k
// TILE-LABEL: func.func @wide
// TILE: tile.matmul_kernel
// TILE-NOT: tessera.split_k

// REMARK: remark: ROCM_SPLIT_K_NOT_APPLIED: 16 output tiles on 32 WGPs asks for split-K, but K=2050 has no 2-way split
// REMARK-NOT: ROCM_SPLIT_K_NOT_APPLIED
