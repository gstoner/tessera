// RUN: tessera-opt %s -tessera-export-shardy | FileCheck %s

// ============================================================================
// Test 1: ShardyExportPass — module gets sdy.mesh from schedule.mesh.define
// ============================================================================

// CHECK: module
// CHECK-SAME: sdy.mesh = "#sdy.mesh<[\"dp\"=4, \"tp\"=8]>"
module @test_mesh_export {

  // Feed the pass a schedule.mesh.define so it knows the axis sizes.
  "schedule.mesh.define"() {
      dims       = [4 : i64, 8 : i64],
      axis_names = ["dp", "tp"]
  } : () -> ()

  func.func @identity(%arg0: tensor<128x64xf32>) -> tensor<128x64xf32> {
    return %arg0 : tensor<128x64xf32>
  }
}

// ============================================================================
// Test 2: ShardyExportPass — replicated tessera.shard → open sdy sharding
// ============================================================================

// CHECK-LABEL: func @test_replicated_shard
// CHECK:       "some.op"
// CHECK-SAME:  sdy.tensor_sharding = "#sdy.sharding<@mesh, [{}, {}]>"
// CHECK-NOT:   tessera.shard
module @test_replicated {
  func.func @test_replicated_shard(%arg0: tensor<16x16xf32>) -> tensor<16x16xf32> {
    %out = "some.op"(%arg0) {
        tessera.shard = "replicated"
    } : (tensor<16x16xf32>) -> tensor<16x16xf32>
    return %out : tensor<16x16xf32>
  }
}

// ============================================================================
// Test 3: ShardyExportPass — block shard over dp on dim 0
// ============================================================================

// CHECK-LABEL: func @test_block_shard_dim0
// CHECK:       "some.op"
// CHECK-SAME:  sdy.tensor_sharding = "#sdy.sharding<@mesh, [{\"dp\"}, {}]>"
// CHECK-NOT:   tessera.shard
module @test_block_dim0 {
  func.func @test_block_shard_dim0(%arg0: tensor<128x64xf32>) -> tensor<128x64xf32> {
    %out = "some.op"(%arg0) {
        tessera.shard = {kind = "block", axes = ["dp"], dims = [0 : i64]}
    } : (tensor<128x64xf32>) -> tensor<128x64xf32>
    return %out : tensor<128x64xf32>
  }
}

// ============================================================================
// Test 4: ShardyExportPass — 2D block shard (batch→dp, hidden→tp)
// ============================================================================

// CHECK-LABEL: func @test_2d_block_shard
// CHECK:       "some.op"
// CHECK-SAME:  sdy.tensor_sharding = "#sdy.sharding<@mesh, [{\"dp\"}, {\"tp\"}, {}]>"
module @test_2d_block {
  func.func @test_2d_block_shard(%arg0: tensor<64x32x16xf32>) -> tensor<64x32x16xf32> {
    %out = "some.op"(%arg0) {
        tessera.shard = {
            kind = "block",
            axes = ["dp", "tp"],
            dims = [0 : i64, 1 : i64]
        }
    } : (tensor<64x32x16xf32>) -> tensor<64x32x16xf32>
    return %out : tensor<64x32x16xf32>
  }
}

// ============================================================================
// Test 5: ShardyExportPass — cyclic shard is emitted identically to block
//         (cycling is handled by the scheduler, not the sharding annotation)
// ============================================================================

// CHECK-LABEL: func @test_cyclic_shard
// CHECK:       "some.op"
// CHECK-SAME:  sdy.tensor_sharding = "#sdy.sharding<@mesh, [{\"dp\"}, {}]>"
module @test_cyclic {
  func.func @test_cyclic_shard(%arg0: tensor<128x64xf32>) -> tensor<128x64xf32> {
    %out = "some.op"(%arg0) {
        tessera.shard = {kind = "cyclic", axes = ["dp"], dims = [0 : i64]}
    } : (tensor<128x64xf32>) -> tensor<128x64xf32>
    return %out : tensor<128x64xf32>
  }
}

// ============================================================================
// Test 6: ShardyExportPass — stale placeholder sharding is upgraded
// ============================================================================

// CHECK-LABEL: func @test_placeholder_upgrade
// CHECK:       stablehlo.add
// CHECK-SAME:  sdy.tensor_sharding = "#sdy.sharding<@mesh, [{}, {}]>"
// CHECK-NOT:   "{sharding = replicated}"
module @test_placeholder {
  func.func @test_placeholder_upgrade(
      %arg0: tensor<4x4xf32>, %arg1: tensor<4x4xf32>) -> tensor<4x4xf32> {
    %out = "stablehlo.add"(%arg0, %arg1) {
        sdy.tensor_sharding = "{sharding = replicated}"
    } : (tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<4x4xf32>
    return %out : tensor<4x4xf32>
  }
}

// ============================================================================
// Test 7: ShardyExportPass — function arg/result shardings
// ============================================================================

// CHECK-LABEL: func @test_func_shardings
// CHECK-SAME:  sdy.in_shardings = ["#sdy.sharding<@mesh, [{\"dp\"}, {}]>"]
// CHECK-SAME:  sdy.out_shardings = ["#sdy.sharding<@mesh, [{}, {}]>"]
module @test_func_shardings {
  func.func @test_func_shardings(%arg0: tensor<128x64xf32>) -> tensor<128x64xf32>
      attributes {
        tessera.arg_shardings = [
            {kind = "block", axes = ["dp"], dims = [0 : i64]}
        ],
        tessera.res_shardings = [
            "replicated"
        ]
      } {
    return %arg0 : tensor<128x64xf32>
  }
}
