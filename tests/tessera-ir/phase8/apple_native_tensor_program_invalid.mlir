// APPLE-NTBI-2 slice 1 refusals: a requested Apple program outside the first
// profile fails closed with a stable diagnostic rather than reaching a lane
// that cannot carry it (Decision #21).
//
// REQUIRES: tessera-apple-backend
// RUN: not tessera-opt %s --split-input-file --tessera-graph-to-schedule 2>&1 | FileCheck %s

// relu has no Apple scheduled kernel.
// CHECK: APPLE_TENSOR_PROGRAM_{{UNSUPPORTED}}: an Apple member must be tessera.matmul or tessera.softmax
module attributes {tessera.target = "apple_gpu", tessera.arch = "apple7",
                   tessera.native.apple_tensor_program_request} {
  func.func @relu_product(%a: tensor<64x32xf16>, %b: tensor<32x48xf16>) -> tensor<64x48xf32> {
    %r = tessera.relu %a : (tensor<64x32xf16>) -> tensor<64x32xf16>
    %o = tessera.matmul %r, %b {output_dtype = "fp32"} : (tensor<64x32xf16>, tensor<32x48xf16>) -> tensor<64x48xf32>
    return %o : tensor<64x48xf32>
  }
}

// -----

// The apple7 matrix member is the f16 storage / f32 accumulate product only.
// CHECK: APPLE_TENSOR_PROGRAM_{{UNSUPPORTED}}: an Apple matrix member requires the apple7 f16 storage / f32 accumulate product
module attributes {tessera.target = "apple_gpu", tessera.arch = "apple7",
                   tessera.native.apple_tensor_program_request} {
  func.func @f32_product(%s: tensor<64x32xf32>, %v: tensor<32x48xf32>) -> tensor<64x48xf32> {
    %p = tessera.softmax %s {axis = -1 : i64} : (tensor<64x32xf32>) -> tensor<64x32xf32>
    %o = tessera.matmul %p, %v : (tensor<64x32xf32>, tensor<32x48xf32>) -> tensor<64x48xf32>
    return %o : tensor<64x48xf32>
  }
}

// -----

// A lone product has no program producer to own.
// CHECK: APPLE_TENSOR_PROGRAM_{{UNSUPPORTED}}: the Apple native tensor program needs a producer and a matrix product
module attributes {tessera.target = "apple_gpu", tessera.arch = "apple7",
                   tessera.native.apple_tensor_program_request} {
  func.func @lone_product(%a: tensor<32x32xf16>) -> tensor<32x32xf32> {
    %o = tessera.matmul %a, %a {output_dtype = "fp32"} : (tensor<32x32xf16>, tensor<32x32xf16>) -> tensor<32x32xf32>
    return %o : tensor<32x32xf32>
  }
}

// -----

// A matrix-result consumer belongs to the fused epilogue lowering.
// CHECK: APPLE_TENSOR_PROGRAM_{{UNSUPPORTED}}: the Apple native tensor program must end in its matrix product
module attributes {tessera.target = "apple_gpu", tessera.arch = "apple7",
                   tessera.native.apple_tensor_program_request} {
  func.func @product_softmax(%a: tensor<64x32xf16>, %b: tensor<32x48xf16>) -> tensor<64x48xf32> {
    %o = tessera.matmul %a, %b {output_dtype = "fp32"} : (tensor<64x32xf16>, tensor<32x48xf16>) -> tensor<64x48xf32>
    %p = tessera.softmax %o {axis = -1 : i64} : (tensor<64x48xf32>) -> tensor<64x48xf32>
    return %p : tensor<64x48xf32>
  }
}

// -----

// Two products of program roots: the last product consumes no producer.
// CHECK: APPLE_TENSOR_PROGRAM_{{UNSUPPORTED}}: the Apple matrix product must consume a program producer
module attributes {tessera.target = "apple_gpu", tessera.arch = "apple7",
                   tessera.native.apple_tensor_program_request} {
  func.func @independent(%s: tensor<32x32xf16>, %a: tensor<32x32xf16>) -> tensor<32x32xf32> {
    %p = tessera.softmax %s {axis = -1 : i64} : (tensor<32x32xf16>) -> tensor<32x32xf16>
    %o = tessera.matmul %a, %a {output_dtype = "fp32"} : (tensor<32x32xf16>, tensor<32x32xf16>) -> tensor<32x32xf32>
    return %o : tensor<32x32xf32>
  }
}
