// RUN: tessera-opt --tessera-graph-to-schedule %s 2>/dev/null | FileCheck %s --check-prefix=SCHED
// RUN: tessera-opt --tessera-graph-to-schedule --tessera-schedule-to-tile %s 2>/dev/null | FileCheck %s --check-prefix=TILE
// RUN: tessera-opt --tessera-graph-to-schedule %s -o /dev/null 2>&1 | FileCheck %s --check-prefix=WARN
//
// ROCM-SPLIT-K-1: the Graph->Schedule decision. `selectGfx1201SplitK` is the
// one decider (Decision #31; `rocm_tiling.select_split_k` is its oracle): split
// when even a two-way split keeps tiles x S within gfx1201's measured
// 256-workgroup target (2026-09-27 device-clock sweep), into the largest
// power-of-two slice count <= min(32, 256 / tiles) whose slices are whole macro
// K blocks of >= 256.
//
//   @router       16x256x2048 f16  -> 16 tiles, target allows 16, the 256
//                                     slice guard stops at split_k = 8
//   @router_fused same, bias+gelu  -> split; the Tile op still states the
//                                     PROGRAM's epilogue (the reduce applies it)
//   @ragged_k     K = 2050         -> occupancy asks, no aligned split exists:
//                                     unsplit, and a ROCM_SPLIT_K_NOT_APPLIED
//                                     warning says so (never silent, #21a)
//   @decode       16x256x256       -> occupancy-short but K below two
//                                     256-wide slices: outside the rule,
//                                     no split and NO warning
//   @expert       16x768x2048      -> 48 tiles (more than the 32 WGPs, which
//                                     the pre-sweep rule never split): 256/48
//                                     -> split_k = 4, measured 2.4x
//   @past_target  32x1536x4096     -> 192 tiles: 2 x 192 > 256, no split
//                                     (measured neutral at S=2, a loss at 8)
//   @wide         1024^2 x 2048    -> 256 tiles: not occupancy-short, no split
//
// @decode, @past_target and @wide are the negative fixtures Decision #10a
// requires.

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
  func.func @decode(%a: tensor<16x256xf16>, %b: tensor<256x256xf16>) -> tensor<16x256xf32> {
    %0 = tessera.matmul %a, %b : (tensor<16x256xf16>, tensor<256x256xf16>) -> tensor<16x256xf32>
    return %0 : tensor<16x256xf32>
  }
  func.func @expert(%a: tensor<16x2048xbf16>, %b: tensor<2048x768xbf16>) -> tensor<16x768xf32> {
    %0 = tessera.matmul %a, %b : (tensor<16x2048xbf16>, tensor<2048x768xbf16>) -> tensor<16x768xf32>
    return %0 : tensor<16x768xf32>
  }
  func.func @past_target(%a: tensor<32x4096xf16>, %b: tensor<4096x1536xf16>) -> tensor<32x1536xf32> {
    %0 = tessera.matmul %a, %b : (tensor<32x4096xf16>, tensor<4096x1536xf16>) -> tensor<32x1536xf32>
    return %0 : tensor<32x1536xf32>
  }
  func.func @wide(%a: tensor<1024x2048xf16>, %b: tensor<2048x1024xf16>) -> tensor<1024x1024xf32> {
    %0 = tessera.matmul %a, %b : (tensor<1024x2048xf16>, tensor<2048x1024xf16>) -> tensor<1024x1024xf32>
    return %0 : tensor<1024x1024xf32>
  }
}

// SCHED-LABEL: func.func @router
// SCHED: schedule.matmul
// SCHED-SAME: block_k = 32 : i64
// SCHED-SAME: split_k = 8 : i64
// SCHED-SAME: split_k_reduction = "ordered"
// SCHED-LABEL: func.func @router_fused
// SCHED: schedule.matmul
// SCHED-SAME: activation = "gelu"
// SCHED-SAME: bias = true
// SCHED-SAME: split_k = 8 : i64
// SCHED-SAME: split_k_reduction = "ordered"
// SCHED-LABEL: func.func @ragged_k
// SCHED: schedule.matmul
// SCHED-NOT: split_k
// SCHED-LABEL: func.func @decode
// SCHED: schedule.matmul
// SCHED-NOT: split_k
// SCHED-LABEL: func.func @expert
// SCHED: schedule.matmul
// SCHED-SAME: split_k = 4 : i64
// SCHED-SAME: split_k_reduction = "ordered"
// SCHED-LABEL: func.func @past_target
// SCHED: schedule.matmul
// SCHED-NOT: split_k
// SCHED-LABEL: func.func @wide
// SCHED: schedule.matmul
// SCHED-NOT: split_k

// TILE-LABEL: func.func @router
// TILE: tile.matmul_kernel
// TILE-SAME: k_blocks = 2
// TILE-SAME: tessera.split_k = 8 : i64
// TILE-SAME: tessera.split_k_reduction = "ordered"
// TILE-LABEL: func.func @router_fused
// TILE: tile.matmul_kernel
// TILE-SAME: epilogue = #tile.epilogue<bias = true, activation = "gelu", output = "f32">
// TILE-SAME: tessera.split_k = 8 : i64
// TILE-LABEL: func.func @ragged_k
// TILE: tile.matmul_kernel
// TILE-NOT: tessera.split_k
// TILE-LABEL: func.func @expert
// TILE: tile.matmul_kernel
// TILE-SAME: tessera.split_k = 4 : i64
// TILE-LABEL: func.func @past_target
// TILE: tile.matmul_kernel
// TILE-NOT: tessera.split_k
// TILE-LABEL: func.func @wide
// TILE: tile.matmul_kernel
// TILE-NOT: tessera.split_k

// WARN: warning: ROCM_SPLIT_K_NOT_APPLIED: 16 output tiles under the 256-workgroup split-K target ask for split-K, but K=2050 has no 2-way split
// WARN-NOT: ROCM_SPLIT_K_NOT_APPLIED
