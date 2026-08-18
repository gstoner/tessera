// RUN: %trop --allow-unregistered-dialect --lower-tile-to-rocm %s | FileCheck %s

module {
  func.func @kernel(%a: tensor<16x16xf32>, %b: tensor<16x16xf32>, %dst: !llvm.ptr, %src: !llvm.ptr, %bytes: i64) {
    %m = "tile.mma"(%a, %b) : (tensor<16x16xf32>, tensor<16x16xf32>) -> tensor<16x16xf32>
    %tok = "tile.async_copy"(%dst, %src, %bytes) : (!llvm.ptr, !llvm.ptr, i64) -> !tessera_rocm.token
    "tile.wait_async"() : () -> ()
    return
  }
}

// CHECK: tessera_rocm.mfma
// CHECK-SAME: arch = "gfx90a"
// CHECK-SAME: shape = "m16n16k16"
// CHECK-SAME: source = "tessera.matmul"
// CHECK: tessera_rocm.async_copy
// CHECK-SAME: dst_space = "lds"
// CHECK-SAME: src_space = "global"
// CHECK: tessera_rocm.wait
