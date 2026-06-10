// RUN: tessera-opt --tessera-canonicalize %s | FileCheck %s
// 2026-06-10: guards for two tessera-canonicalize correctness fixes.
//
// (1) TransposeIntoMatmul flag composition — folding a tessera.transpose
//     into a matmul that ALREADY carries the flag must XOR, not OR:
//     matmul(transpose(A)) with transposeA=true computes (Aᵀ)ᵀ = A, so
//     the rewritten matmul must read A un-transposed (flag false).
// (2) FuseMatmulBiasGELU / FuseConvRelu hasOneUse guards — when the
//     intermediate add/conv (or the producing matmul) has another
//     consumer, fusing would duplicate the matmul/conv work, so the
//     pattern must not fire.

// (1a) Existing transposeA=true + folded transpose ⇒ flag must cancel to
// false (elided or printed false — never true), and the transpose is DCE'd.
// CHECK-LABEL: func.func @transpose_into_matmul_existing_flag
// CHECK-NOT:     tessera.transpose
// CHECK:         tessera.matmul
// CHECK-NOT:     transposeA = true
// CHECK:         return

// (1b) Control: no pre-existing flag ⇒ folding sets transposeA=true.
// CHECK-LABEL: func.func @transpose_into_matmul_sets_flag
// CHECK-NOT:     tessera.transpose
// CHECK:         tessera.matmul
// CHECK-SAME:    transposeA = true

// (2a) add result has a second user (returned) ⇒ must NOT fuse.
// CHECK-LABEL: func.func @no_fuse_gelu_when_add_has_other_user
// CHECK-NOT:     tessera.fused_epilogue
// CHECK:         tessera.matmul
// CHECK:         tessera.add
// CHECK:         tessera.gelu
// CHECK-NOT:     tessera.fused_epilogue

// (2b) matmul result has a second user (returned) ⇒ must NOT fuse.
// CHECK-LABEL: func.func @no_fuse_gelu_when_matmul_has_other_user
// CHECK-NOT:     tessera.fused_epilogue
// CHECK:         tessera.matmul
// CHECK:         tessera.add
// CHECK:         tessera.gelu
// CHECK-NOT:     tessera.fused_epilogue

// (2c) Control: single-use matmul+add+gelu chain still fuses.
// CHECK-LABEL: func.func @fuse_matmul_bias_gelu_single_use
// CHECK:         tessera.fused_epilogue
// CHECK-NOT:     tessera.matmul
// CHECK-NOT:     tessera.gelu

// (2d) conv result has a second user (returned) ⇒ must NOT fuse the relu.
// CHECK-LABEL: func.func @no_fuse_relu_when_conv_has_other_user
// CHECK-NOT:     epilogue
// CHECK:         tessera.conv2d_nhwc
// CHECK:         tessera.relu
// CHECK-NOT:     epilogue

// (2e) Control: single-use conv+relu still fuses into the epilogue.
// CHECK-LABEL: func.func @fuse_conv_relu_single_use
// CHECK:         tessera.conv2d_nhwc
// CHECK-SAME:    epilogue = 1
// CHECK-NOT:     tessera.relu

module attributes {tessera.ir.version = "1.0"} {

  // matmul(transpose(A), B) with transposeA already true:
  // %t = Aᵀ is 128x64; transposeA=true reads it back as A (64x128).
  func.func @transpose_into_matmul_existing_flag(
      %A: tensor<64x128xf32>,
      %B: tensor<128x32xf32>
  ) -> tensor<64x32xf32> {
    %t = "tessera.transpose"(%A) : (tensor<64x128xf32>) -> tensor<128x64xf32>
    %C = "tessera.matmul"(%t, %B) {transposeA = true}
        : (tensor<128x64xf32>, tensor<128x32xf32>) -> tensor<64x32xf32>
    return %C : tensor<64x32xf32>
  }

  func.func @transpose_into_matmul_sets_flag(
      %A: tensor<64x32xf32>,
      %B: tensor<64x16xf32>
  ) -> tensor<32x16xf32> {
    %t = "tessera.transpose"(%A) : (tensor<64x32xf32>) -> tensor<32x64xf32>
    %C = "tessera.matmul"(%t, %B)
        : (tensor<32x64xf32>, tensor<64x16xf32>) -> tensor<32x16xf32>
    return %C : tensor<32x16xf32>
  }

  func.func @no_fuse_gelu_when_add_has_other_user(
      %A: tensor<8x16xf32>,
      %B: tensor<16x8xf32>,
      %bias: tensor<8x8xf32>
  ) -> (tensor<8x8xf32>, tensor<8x8xf32>) {
    %mm = "tessera.matmul"(%A, %B)
        : (tensor<8x16xf32>, tensor<16x8xf32>) -> tensor<8x8xf32>
    %add = "tessera.add"(%mm, %bias)
        : (tensor<8x8xf32>, tensor<8x8xf32>) -> tensor<8x8xf32>
    %g = "tessera.gelu"(%add) : (tensor<8x8xf32>) -> tensor<8x8xf32>
    return %g, %add : tensor<8x8xf32>, tensor<8x8xf32>
  }

  func.func @no_fuse_gelu_when_matmul_has_other_user(
      %A: tensor<8x16xf32>,
      %B: tensor<16x8xf32>,
      %bias: tensor<8x8xf32>
  ) -> (tensor<8x8xf32>, tensor<8x8xf32>) {
    %mm = "tessera.matmul"(%A, %B)
        : (tensor<8x16xf32>, tensor<16x8xf32>) -> tensor<8x8xf32>
    %add = "tessera.add"(%mm, %bias)
        : (tensor<8x8xf32>, tensor<8x8xf32>) -> tensor<8x8xf32>
    %g = "tessera.gelu"(%add) : (tensor<8x8xf32>) -> tensor<8x8xf32>
    return %g, %mm : tensor<8x8xf32>, tensor<8x8xf32>
  }

  func.func @fuse_matmul_bias_gelu_single_use(
      %A: tensor<8x16xf32>,
      %B: tensor<16x8xf32>,
      %bias: tensor<8x8xf32>
  ) -> tensor<8x8xf32> {
    %mm = "tessera.matmul"(%A, %B)
        : (tensor<8x16xf32>, tensor<16x8xf32>) -> tensor<8x8xf32>
    %add = "tessera.add"(%mm, %bias)
        : (tensor<8x8xf32>, tensor<8x8xf32>) -> tensor<8x8xf32>
    %g = "tessera.gelu"(%add) : (tensor<8x8xf32>) -> tensor<8x8xf32>
    return %g : tensor<8x8xf32>
  }

  func.func @no_fuse_relu_when_conv_has_other_user(
      %input: tensor<1x8x8x4xf32>,
      %filter: tensor<3x3x4x8xf32>
  ) -> (tensor<1x8x8x8xf32>, tensor<1x8x8x8xf32>) {
    %c = "tessera.conv2d_nhwc"(%input, %filter)
        {strides = [1, 1], dilations = [1, 1]}
        : (tensor<1x8x8x4xf32>, tensor<3x3x4x8xf32>) -> tensor<1x8x8x8xf32>
    %r = "tessera.relu"(%c) : (tensor<1x8x8x8xf32>) -> tensor<1x8x8x8xf32>
    return %r, %c : tensor<1x8x8x8xf32>, tensor<1x8x8x8xf32>
  }

  func.func @fuse_conv_relu_single_use(
      %input: tensor<1x8x8x4xf32>,
      %filter: tensor<3x3x4x8xf32>
  ) -> tensor<1x8x8x8xf32> {
    %c = "tessera.conv2d_nhwc"(%input, %filter)
        {strides = [1, 1], dilations = [1, 1]}
        : (tensor<1x8x8x4xf32>, tensor<3x3x4x8xf32>) -> tensor<1x8x8x8xf32>
    %r = "tessera.relu"(%c) : (tensor<1x8x8x8xf32>) -> tensor<1x8x8x8xf32>
    return %r : tensor<1x8x8x8xf32>
  }
}
