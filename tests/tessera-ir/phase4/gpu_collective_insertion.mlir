// RUN: tessera-opt --tessera-gpu-collective-insertion %s | FileCheck %s

// Test: GPUCollectiveInsertionPass inserts reduce_scatter after a
// column-parallel linear op (tagged tessera.weight_sharding = "col_parallel")
// and all_gather after a row-parallel op.

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

  func.func @forward(%x: memref<128x256xbf16>) -> memref<128x512xbf16> {
    %c0 = arith.constant 0 : index
    %out = memref.alloc() : memref<128x512xbf16>

    // Column-parallel matmul: each TP rank computes a partial column block.
    // The pass should insert tessera.collective.reduce_scatter after this op.
    // CHECK: tessera.collective.reduce_scatter
    // CHECK-SAME: reduce_op = "sum"
    // CHECK-SAME: mesh_axis
    // CHECK-SAME: tessera.future_payload
    %partial = memref.alloc() : memref<128x512xbf16>
    "tessera.matmul"(%x, %partial) {
      tessera.weight_sharding = "col_parallel",
      tessera.tp_axis = "tp"
    } : (memref<128x256xbf16>, memref<128x512xbf16>) -> ()

    // Row-parallel matmul: the pass should insert tessera.collective.all_gather.
    // CHECK: tessera.collective.all_gather
    // CHECK-SAME: mesh_axis
    // CHECK-SAME: tessera.future_payload
    %partial2 = memref.alloc() : memref<128x256xbf16>
    "tessera.matmul"(%partial, %partial2) {
      tessera.weight_sharding = "row_parallel",
      tessera.tp_axis = "tp"
    } : (memref<128x512xbf16>, memref<128x256xbf16>) -> ()

    return %out : memref<128x512xbf16>
  }
}
