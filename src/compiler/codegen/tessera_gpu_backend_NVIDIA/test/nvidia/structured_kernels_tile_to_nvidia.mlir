// RUN: %tnv --allow-unregistered-dialect --lower-tile-to-nvidia='sm=90' %s | FileCheck %s

module {
  func.func @conv2d(%input: tensor<1x8x8x4xf32>,
                    %filter: tensor<3x3x4x8xf32>) -> tensor<1x6x6x8xf32> {
    %result = "tile.conv2d"(%input, %filter) {
      tile_h = 6 : i64, tile_w = 6 : i64, tile_c = 8 : i64
    } : (tensor<1x8x8x4xf32>, tensor<3x3x4x8xf32>) -> tensor<1x6x6x8xf32>
    return %result : tensor<1x6x6x8xf32>
  }

  func.func @kv_cache_read() {
    "tile.kv_cache"() {
      source = "tessera.kv_cache.read", result = "K,V",
      storage = "paged", ordinal = 0 : i64
    } : () -> ()
    return
  }
}

// CHECK: tessera_nvidia.cuda_kernel
// CHECK-SAME: arch = "sm_90a"
// CHECK-SAME: source = "tessera.conv2d_nhwc"
// CHECK: tessera_nvidia.cuda_kernel
// CHECK-SAME: arch = "sm_90a"
// CHECK-SAME: source = "tessera.kv_cache.read"
