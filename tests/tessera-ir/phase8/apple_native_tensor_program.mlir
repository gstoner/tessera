// APPLE-NTBI-2 slice 1: an apple7 matrix-product DAG becomes a native tensor
// program on explicit request. Each member ends at the launch-level Tile
// carrier its Apple scheduled package consumes, the softmax result is a
// program-owned private buffer, and the frame names the Apple target.
// Without the request an Apple program is not claimed (Decision #10a).
//
// REQUIRES: tessera-apple-backend
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

// -----

// APPLE-NTBI-2 slice 5: the tanh GELU activation is an Apple producer member;
// it ends at the shared activation launch carrier the scheduled GELU package
// consumes.
// CHECK-LABEL: module @gelu_requested
// CHECK-SAME: tessera.native.tensor_program = 1
// CHECK: %[[G:.*]] = memref.alloc() : memref<64x32xf16>
// CHECK: tile.program_invoke %arg0, %[[G]] {callee = @tensor_stage_0::@source, carrier = "tile.activation_kernel"
// CHECK: tile.program_invoke %[[G]], %arg1, %{{.*}} {callee = @tensor_stage_1::@source, carrier = "tile.matmul_kernel"
// CHECK: module @tensor_stage_0
// CHECK: tile.activation_kernel
module @gelu_requested attributes {tessera.target = "apple_gpu", tessera.arch = "apple7",
                                   tessera.native.apple_tensor_program_request} {
  func.func @gelu_product(%x: tensor<64x32xf16>, %w: tensor<32x48xf16>) -> tensor<64x48xf32> {
    %g = tessera.gelu %x : (tensor<64x32xf16>) -> tensor<64x32xf16>
    %o = tessera.matmul %g, %w {output_dtype = "fp32"} : (tensor<64x32xf16>, tensor<32x48xf16>) -> tensor<64x48xf32>
    return %o : tensor<64x48xf32>
  }
}

// -----

// APPLE-NTBI-2 slice 6: the f16 ReLU is an Apple producer member; it ends at
// the shared activation launch carrier with the ReLU numerical contract the
// synthesized MSL consumer computes.
// CHECK-LABEL: module @relu_requested
// CHECK-SAME: tessera.native.tensor_program = 1
// CHECK: %[[R:.*]] = memref.alloc() : memref<64x32xf16>
// CHECK: tile.program_invoke %arg0, %[[R]] {callee = @tensor_stage_0::@source, carrier = "tile.activation_kernel"
// CHECK: tile.program_invoke %[[R]], %arg1, %{{.*}} {callee = @tensor_stage_1::@source, carrier = "tile.matmul_kernel"
// CHECK: module @tensor_stage_0
// CHECK: tile.activation_kernel
// CHECK-SAME: kind = "relu"
// CHECK-SAME: zero_mode = "positive"
module @relu_requested attributes {tessera.target = "apple_gpu", tessera.arch = "apple7",
                                   tessera.native.apple_tensor_program_request} {
  func.func @relu_product(%x: tensor<64x32xf16>, %w: tensor<32x48xf16>) -> tensor<64x48xf32> {
    %r = tessera.relu %x : (tensor<64x32xf16>) -> tensor<64x32xf16>
    %o = tessera.matmul %r, %w {output_dtype = "fp32"} : (tensor<64x32xf16>, tensor<32x48xf16>) -> tensor<64x48xf32>
    return %o : tensor<64x48xf32>
  }
}

// -----

// APPLE-NTBI-2 slice 7: the f16 add is an Apple producer member over two
// program inputs, ending at the shared elementwise launch carrier.
// CHECK-LABEL: module @add_requested
// CHECK-SAME: tessera.native.tensor_program = 1
// CHECK: %[[S:.*]] = memref.alloc() : memref<64x32xf16>
// CHECK: tile.program_invoke %arg0, %arg1, %[[S]] {callee = @tensor_stage_0::@source, carrier = "tile.elementwise_kernel"
// CHECK: tile.program_invoke %[[S]], %arg2, %{{.*}} {callee = @tensor_stage_1::@source, carrier = "tile.matmul_kernel"
// CHECK: module @tensor_stage_0
// CHECK: tile.elementwise_kernel
// CHECK-SAME: family = "binary"
// CHECK-SAME: kind = "add"
module @add_requested attributes {tessera.target = "apple_gpu", tessera.arch = "apple7",
                                  tessera.native.apple_tensor_program_request} {
  func.func @add_product(%x: tensor<64x32xf16>, %y: tensor<64x32xf16>, %w: tensor<32x48xf16>) -> tensor<64x48xf32> {
    %s = tessera.add %x, %y : (tensor<64x32xf16>, tensor<64x32xf16>) -> tensor<64x32xf16>
    %o = tessera.matmul %s, %w {output_dtype = "fp32"} : (tensor<64x32xf16>, tensor<32x48xf16>) -> tensor<64x48xf32>
    return %o : tensor<64x48xf32>
  }
}

// -----

// APPLE-NTBI-2 slice 8: a bf16 program -- synthesized ReLU feeding the bf16
// simdgroup product (bf16 storage, f32 accumulation).
// CHECK-LABEL: module @bf16_requested
// CHECK-SAME: tessera.native.tensor_program = 1
// CHECK: tile.program_invoke %arg0, %{{.*}} {callee = @tensor_stage_0::@source, carrier = "tile.activation_kernel"
// CHECK: tile.program_invoke %{{.*}}, %arg1, %{{.*}} {callee = @tensor_stage_1::@source, carrier = "tile.matmul_kernel"
// CHECK: module @tensor_stage_0
// CHECK: tile.activation_kernel
// CHECK-SAME: storage = "bf16"
// CHECK: module @tensor_stage_1
// CHECK: tile.matmul_kernel
// CHECK-SAME: a = "bf16", b = "bf16", acc = "f32"
module @bf16_requested attributes {tessera.target = "apple_gpu", tessera.arch = "apple7",
                                   tessera.native.apple_tensor_program_request} {
  func.func @relu_product(%x: tensor<64x32xbf16>, %w: tensor<32x48xbf16>) -> tensor<64x48xf32> {
    %r = tessera.relu %x : (tensor<64x32xbf16>) -> tensor<64x32xbf16>
    %o = tessera.matmul %r, %w {output_dtype = "fp32"} : (tensor<64x32xbf16>, tensor<32x48xbf16>) -> tensor<64x48xf32>
    return %o : tensor<64x48xf32>
  }
}

// -----

// APPLE-NTBI-2 slice 9: the unweighted f16 layer_norm is an Apple producer
// member ending at the Tile norm launch carrier (compiler-synthesized MSL).
// CHECK-LABEL: module @norm_requested
// CHECK-SAME: tessera.native.tensor_program = 1
// CHECK: %[[N:.*]] = memref.alloc() : memref<64x32xf16>
// CHECK: tile.program_invoke %arg0, %[[N]] {callee = @tensor_stage_0::@source, carrier = "tile.norm_kernel"
// CHECK: tile.program_invoke %[[N]], %arg1, %{{.*}} {callee = @tensor_stage_1::@source, carrier = "tile.matmul_kernel"
// CHECK: module @tensor_stage_0
// CHECK: tile.norm_kernel
// CHECK-SAME: affine = false
// CHECK-SAME: kind = "layernorm"
// CHECK-SAME: storage = "f16"
// CHECK-SAME: tessera.workgroup_size = 1 : i64
module @norm_requested attributes {tessera.target = "apple_gpu", tessera.arch = "apple7",
                                   tessera.native.apple_tensor_program_request} {
  func.func @norm_product(%x: tensor<64x32xf16>, %w: tensor<32x48xf16>) -> tensor<64x48xf32> {
    %n = tessera.layer_norm %x {eps = 1.0e-05 : f64} : (tensor<64x32xf16>) -> tensor<64x32xf16>
    %o = tessera.matmul %n, %w {output_dtype = "fp32"} : (tensor<64x32xf16>, tensor<32x48xf16>) -> tensor<64x48xf32>
    return %o : tensor<64x48xf32>
  }
}
