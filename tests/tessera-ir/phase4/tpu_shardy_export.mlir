// RUN: tessera-opt --tessera-shardy-export %s | FileCheck %s

// Test: TesseraShardyExportPass converts tessera.shard annotations into
// Shardy (sdy) mesh and tensor_sharding attributes recognised by the
// JAX/XLA Shardy partitioner.

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

  // CHECK-LABEL: func.func @col_parallel_weight
  // Shardy export should attach a mesh declaration and column-parallel
  // sharding on the weight tensor (sharded along tp axis, dim 1).
  // CHECK: sdy.mesh
  // CHECK-SAME: "dp"
  // CHECK-SAME: "tp"
  func.func @col_parallel_weight(
      %x:  tensor<128x256xbf16>
          {tessera.shard = {axes = ["tp"], dim = 1}},
      %w:  tensor<256x512xbf16>
          {tessera.shard = {axes = ["tp"], dim = 1, kind = "col_parallel"}}
  ) -> tensor<128x512xbf16> {

    // CHECK: sdy.tensor_sharding
    // CHECK-SAME: dim_shardings
    // CHECK-SAME: "tp"
    %out = "tessera.matmul"(%x, %w) {
      tessera.layer = {name = "fc1"},
      tessera.weight_sharding = "col_parallel",
      tessera.tp_axis = "tp"
    } : (tensor<128x256xbf16>, tensor<256x512xbf16>) -> tensor<128x512xbf16>

    return %out : tensor<128x512xbf16>
  }

  // CHECK-LABEL: func.func @row_parallel_weight
  // Row-parallel weight is sharded along dim 0; output is replicated and
  // an all_gather is expected (already inserted by collective-insertion pass).
  func.func @row_parallel_weight(
      %x:  tensor<128x512xbf16>
          {tessera.shard = {axes = ["tp"], dim = 1}},
      %w:  tensor<512x256xbf16>
          {tessera.shard = {axes = ["tp"], dim = 0, kind = "row_parallel"}}
  ) -> tensor<128x256xbf16> {

    // CHECK: sdy.tensor_sharding
    // CHECK-SAME: "tp"
    %out = "tessera.matmul"(%x, %w) {
      tessera.layer = {name = "fc2"},
      tessera.weight_sharding = "row_parallel",
      tessera.tp_axis = "tp"
    } : (tensor<128x512xbf16>, tensor<512x256xbf16>) -> tensor<128x256xbf16>

    return %out : tensor<128x256xbf16>
  }

  // CHECK-LABEL: func.func @replicated_norm
  // Replicated layers get full-replication Shardy sharding annotations
  // (empty dim_shardings).
  func.func @replicated_norm(
      %x: tensor<128x256xbf16>
         {tessera.shard = {axes = [], dim = 0, kind = "replicated"}}
  ) -> tensor<128x256xbf16> {

    // CHECK: sdy.tensor_sharding
    // CHECK-NOT: "tp"
    %out = "tessera.layernorm"(%x) {
      tessera.layer = {name = "norm"},
      tessera.weight_sharding = "replicated"
    } : (tensor<128x256xbf16>) -> tensor<128x256xbf16>

    return %out : tensor<128x256xbf16>
  }
}
