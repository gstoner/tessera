// APPLE-NTBI-2 slice 1: an apple7 matrix-product DAG becomes a native tensor
// program on explicit request. Each member ends at the launch-level Tile
// carrier its Apple scheduled package consumes, the softmax result is a
// program-owned private buffer, and the frame names the Apple target.
// Without the request an Apple program is not claimed (Decision #10a).
//
// RUN: tessera-opt %s --split-input-file --verify-each \
// RUN:   --tessera-graph-to-schedule --tessera-schedule-to-tile | FileCheck %s

// CHECK-LABEL: module @requested
// CHECK-SAME: tessera.native.tensor_program = 1
// CHECK: func.func @scores_values(%arg0: memref<64x32xf16>, %arg1: memref<32x48xf16>) -> memref<64x48xf32>
// CHECK: %[[P:.*]] = memref.alloc() : memref<64x32xf16>
// CHECK: tile.program_invoke %arg0, %[[P]] {callee = @tensor_stage_0::@source, carrier = "tile.softmax_kernel"
// CHECK: %[[O:.*]] = memref.alloc() : memref<64x48xf32>
// CHECK: tile.program_invoke %[[P]], %arg1, %[[O]] {callee = @tensor_stage_1::@source, carrier = "tile.matmul_kernel"
// CHECK: module @tensor_stage_0
// CHECK: tile.softmax_kernel
// CHECK: module @tensor_stage_1
// CHECK: tile.matmul_kernel
module @requested attributes {tessera.target = "apple_gpu", tessera.arch = "apple7",
                              tessera.native.apple_tensor_program_request} {
  func.func @scores_values(%s: tensor<64x32xf16>, %v: tensor<32x48xf16>) -> tensor<64x48xf32> {
    %p = tessera.softmax %s {axis = -1 : i64} : (tensor<64x32xf16>) -> tensor<64x32xf16>
    %o = tessera.matmul %p, %v {output_dtype = "fp32"} : (tensor<64x32xf16>, tensor<32x48xf16>) -> tensor<64x48xf32>
    return %o : tensor<64x48xf32>
  }
}

// -----

// CHECK-LABEL: module @unrequested
// CHECK-NOT: tessera.native.tensor_program
// CHECK-NOT: tile.program_invoke
// CHECK: func.func @scores_values
module @unrequested attributes {tessera.target = "apple_gpu", tessera.arch = "apple7"} {
  func.func @scores_values(%s: tensor<64x32xf16>, %v: tensor<32x48xf16>) -> tensor<64x48xf32> {
    %p = tessera.softmax %s {axis = -1 : i64} : (tensor<64x32xf16>) -> tensor<64x32xf16>
    %o = tessera.matmul %p, %v {output_dtype = "fp32"} : (tensor<64x32xf16>, tensor<32x48xf16>) -> tensor<64x48xf32>
    return %o : tensor<64x48xf32>
  }
}
