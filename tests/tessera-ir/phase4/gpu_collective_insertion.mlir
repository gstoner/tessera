// RUN: tessera-opt --tessera-gpu-collective-insertion %s | FileCheck %s

// Test: GPUCollectiveInsertionPass inserts reduce_scatter after a
// column-parallel linear op (tagged tessera.weight_sharding = "col_parallel")
// and all_gather after a row-parallel op.
//
// 2026-06: un-XFAIL'd.  The matmuls moved to value-semantics tensor form
// (the MLIR-23 TesseraMatmulOp verifier requires one tensor result).  The
// pass inserts registered asynchronous Target operations and rewires downstream
// users through explicit awaits.

module attributes {
  tessera.distributed_plan = {
    mesh = {"dp" = 4, "tp" = 2},
    total_ranks = 8,
    num_stages = 1,
    layers = [
      {name = "fc1", type = "linear", dp = "dp", tp = "tp",
       weight_sharding = "col_parallel"},
      {name = "fc2", type = "linear", dp = "dp", tp = "tp",
       weight_sharding = "row_parallel"}
    ]
  }
} {

  func.func @forward(%x: tensor<128x256xbf16>, %w1: tensor<256x512xbf16>,
                     %w2: tensor<512x256xbf16>) -> tensor<128x256xbf16> {

    // Column-parallel matmul: each TP rank computes a partial column block.
    // The pass should insert tessera_collective.reduce_scatter after this op.
    // CHECK: %[[RS_F:.*]] = tessera_collective.reduce_scatter
    // CHECK-SAME: mesh_axis
    // CHECK-SAME: reduction = "sum"
    // CHECK: %[[RS:.*]] = tessera_collective.await %[[RS_F]]
    %partial = "tessera.matmul"(%x, %w1) {
      tessera.weight_sharding = "col_parallel",
      tessera.tp_axis = "tp"
    } : (tensor<128x256xbf16>, tensor<256x512xbf16>) -> tensor<128x512xbf16>

    // Row-parallel matmul consumes the awaited reduce-scatter value and then
    // inserts an all-gather for its own downstream users.
    // CHECK: "tessera.matmul"(%[[RS]],
    // CHECK: %[[AG_F:.*]] = tessera_collective.all_gather
    // CHECK-SAME: mesh_axis
    // CHECK-SAME: reduction = "none"
    // CHECK: tessera_collective.await %[[AG_F]]
    %partial2 = "tessera.matmul"(%partial, %w2) {
      tessera.weight_sharding = "row_parallel",
      tessera.tp_axis = "tp"
    } : (tensor<128x512xbf16>, tensor<512x256xbf16>) -> tensor<128x256xbf16>

    return %partial2 : tensor<128x256xbf16>
  }
}
