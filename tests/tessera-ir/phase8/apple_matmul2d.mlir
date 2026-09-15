// REQUIRES: tessera-apple-backend
//
// APPLE-MATMUL2D-1 (2026-09-15): the shared TilingPass turns a Graph-IR matmul
// into the canonical M/N/K contract; the Apple Metal 4 consumer re-forms it as
// two `tensor_view` bindings and one `matmul2d`, the epilogue fusion folds the
// per-column bias add and activation into `matmul2d_epilogue`, and the call
// lowering hands the verified view ABI to the runtime's strided-view symbol.
// Driving all four in one run proves the lane consumes the real shared form.
//
// RUN: tessera-opt %s --tessera-tiling --tessera-apple-canonical-gemm-matmul2d \
// RUN:   --tessera-apple-matmul2d-fuse-epilogue --allow-unregistered-dialect \
// RUN:   | FileCheck %s --check-prefix=TARGET
// RUN: tessera-opt %s --tessera-tiling --tessera-apple-canonical-gemm-matmul2d \
// RUN:   --tessera-apple-matmul2d-fuse-epilogue --tessera-apple-matmul2d-to-call \
// RUN:   --allow-unregistered-dialect | FileCheck %s --check-prefix=CALL
//
// Recognition is not promotion: the incumbent production route is unchanged
// until a paired corpus admits this lane (APPLE-TILE-2 standing rule).

// The nest is consumed; the storage pair, accumulator and layout are stated.
// TARGET-LABEL: func.func @gemm_f16
// TARGET-NOT: scf.for
// TARGET: %[[A:.*]] = tessera_apple.gpu.tensor_view %arg0 {byte_offset = 0 : i64, extents = array<i64: 128, 64>, strides = array<i64: 1, 128>} : tensor<64x128xf16> -> <f16>
// TARGET: %[[B:.*]] = tessera_apple.gpu.tensor_view %arg1 {byte_offset = 0 : i64, extents = array<i64: 96, 128>, strides = array<i64: 1, 96>} : tensor<128x96xf16> -> <f16>
// TARGET: tessera_apple.gpu.matmul2d %[[A]], %[[B]]
// TARGET-SAME: accumulate = "f32"
// TARGET-SAME: simdgroups = 4
// TARGET-SAME: tessera_apple.canonical_k_loop = true
// TARGET-SAME: tile_m = 64
// TARGET-SAME: tile_n = 64
// TARGET-SAME: -> tensor<64x96xf32>
// TARGET-NOT: scf.for
//
// One symbol for every pair; the pair code and the view ABI ride as attributes.
// CALL-LABEL: func.func @gemm_f16
// CALL-NOT: tessera_apple.gpu.matmul2d
// CALL: tessera_apple.gpu.kernel_call %arg0, %arg1
// CALL-SAME: abi = "mtl4_matmul2d_view"
// CALL-SAME: dtype = "f16xf16"
// CALL-SAME: op_kind = "mtl4_matmul2d"
// CALL-SAME: symbol = "tessera_apple_gpu_mtl4_matmul2d_view"
// CALL-SAME: tessera_apple.a_byte_offset = 0
// CALL-SAME: tessera_apple.a_inner = 128
// CALL-SAME: tessera_apple.a_outer = 64
// CALL-SAME: tessera_apple.a_stride = 128
// CALL-SAME: tessera_apple.accumulate = "fp32"
// CALL-SAME: tessera_apple.b_inner = 96
// CALL-SAME: tessera_apple.b_stride = 96
// CALL-SAME: tessera_apple.pair = 10
func.func @gemm_f16(%a: tensor<64x128xf16>, %b: tensor<128x96xf16>) -> tensor<64x96xf32> {
  %0 = "tessera.matmul"(%a, %b) : (tensor<64x128xf16>, tensor<128x96xf16>) -> tensor<64x96xf32>
  return %0 : tensor<64x96xf32>
}

// Ragged M without host padding. The shared tiling pass zero-pads M = 100 to
// 112 and slices the product back; MPP matmul2d edge-checks partial tiles, so
// the view binds the ORIGINAL 100-row operand, the product is the true
// [100, 128], and neither the padding nor the slice survives. The op names
// the runtime contract it relies on (`ragged_tail`). K and N sit on Apple's
// 128-element FP8 quantum, so the packed operands are legal views and the
// call carries pair code 0 (e4m3/e4m3).
// TARGET-LABEL: func.func @gemm_e4m3
// TARGET-NOT: tensor.insert_slice
// TARGET: tessera_apple.gpu.tensor_view %arg0 {byte_offset = 0 : i64, extents = array<i64: 256, 100>, strides = array<i64: 1, 256>} : tensor<100x256xf8E4M3FN> -> <f8E4M3FN>
// TARGET: tessera_apple.gpu.matmul2d
// TARGET-SAME: tessera_apple.ragged_tail = true
// TARGET-SAME: -> tensor<100x128xf32>
// TARGET-NOT: tensor.extract_slice
// CALL-LABEL: func.func @gemm_e4m3
// CALL: tessera_apple.gpu.kernel_call
// CALL-SAME: dtype = "f8E4M3FNxf8E4M3FN"
// CALL-SAME: symbol = "tessera_apple_gpu_mtl4_matmul2d_view"
// CALL-SAME: tessera_apple.a_outer = 100
// CALL-SAME: tessera_apple.pair = 0
// CALL-SAME: tessera_apple.ragged_tail = true
func.func @gemm_e4m3(%a: tensor<100x256xf8E4M3FN>, %b: tensor<256x128xf8E4M3FN>) -> tensor<100x128xf32> {
  %0 = "tessera.matmul"(%a, %b) : (tensor<100x256xf8E4M3FN>, tensor<256x128xf8E4M3FN>) -> tensor<100x128xf32>
  return %0 : tensor<100x128xf32>
}

// The weight-only pair: half activations against an FP4 weight matrix whose
// row stride sits on the 256-element FP4 quantum (pair code 5).
// CALL-LABEL: func.func @gemm_half_e2m1
// CALL: tessera_apple.gpu.kernel_call
// CALL-SAME: dtype = "f16xf4E2M1FN"
// CALL-SAME: tessera_apple.pair = 5
func.func @gemm_half_e2m1(%a: tensor<64x128xf16>, %b: tensor<128x256xf4E2M1FN>) -> tensor<64x256xf32> {
  %0 = "tessera.matmul"(%a, %b) : (tensor<64x128xf16>, tensor<128x256xf4E2M1FN>) -> tensor<64x256xf32>
  return %0 : tensor<64x256xf32>
}

// Sub-block origins. A GEMM over static unit-stride slices of larger matrices
// binds views INTO the parents: nonzero byte offsets ((8 * 512 + 64) * 2 bytes
// for A, 128 * 2 for B), the parents' row stride, nothing copied. The slices
// are dead and dropped; the call passes the parents and the origins.
// TARGET-LABEL: func.func @gemm_sub_block
// TARGET-NOT: tensor.extract_slice
// TARGET: tessera_apple.gpu.tensor_view %arg0 {byte_offset = 8320 : i64, extents = array<i64: 256, 100>, strides = array<i64: 1, 512>} : tensor<200x512xf16> -> <f16>
// TARGET: tessera_apple.gpu.tensor_view %arg1 {byte_offset = 256 : i64, extents = array<i64: 128, 256>, strides = array<i64: 1, 512>} : tensor<512x512xf16> -> <f16>
// TARGET: tessera_apple.gpu.matmul2d
// TARGET-SAME: -> tensor<100x128xf32>
// CALL-LABEL: func.func @gemm_sub_block
// CALL: tessera_apple.gpu.kernel_call %arg0, %arg1
// CALL-SAME: tessera_apple.a_byte_offset = 8320
// CALL-SAME: tessera_apple.a_stride = 512
// CALL-SAME: tessera_apple.b_byte_offset = 256
// CALL-SAME: tessera_apple.b_stride = 512
func.func @gemm_sub_block(%pa: tensor<200x512xf16>, %pb: tensor<512x512xf16>) -> tensor<100x128xf32> {
  %a = tensor.extract_slice %pa[8, 64] [100, 256] [1, 1] : tensor<200x512xf16> to tensor<100x256xf16>
  %b = tensor.extract_slice %pb[0, 128] [256, 128] [1, 1] : tensor<512x512xf16> to tensor<256x128xf16>
  %0 = "tessera.matmul"(%a, %b) : (tensor<100x256xf16>, tensor<256x128xf16>) -> tensor<100x128xf32>
  return %0 : tensor<100x128xf32>
}

// The fused epilogue as an op. The tracer's per-column bias (`tessera.add`
// against a `tessera.broadcast` of a rank-1 [N]) and the following gelu are
// folded into `matmul2d_epilogue`, which states that the bias is added to the
// fp32 accumulator and the activation evaluated in fp32 before one store. The
// call names the epilogue symbol, carries the bias as a third operand, and
// states the activation and the bias presence.
// TARGET-LABEL: func.func @mlp_bias_gelu
// TARGET-NOT: tessera.add
// TARGET-NOT: tessera.gelu
// TARGET: tessera_apple.gpu.matmul2d_epilogue %{{.*}}, %{{.*}} bias %arg2 : tensor<96xf32>
// TARGET-SAME: act = "gelu"
// TARGET-SAME: -> tensor<64x96xf32>
// TARGET-NOT: tessera.broadcast
// CALL-LABEL: func.func @mlp_bias_gelu
// CALL: tessera_apple.gpu.kernel_call %arg0, %arg1, %arg2
// CALL-SAME: op_kind = "mtl4_matmul2d_epilogue"
// CALL-SAME: symbol = "tessera_apple_gpu_mtl4_matmul2d_view_epilogue"
// CALL-SAME: tessera_apple.act = "gelu"
// CALL-SAME: tessera_apple.has_bias = true
// CALL-SAME: (tensor<64x128xf16>, tensor<128x96xf16>, tensor<96xf32>) -> tensor<64x96xf32>
func.func @mlp_bias_gelu(%a: tensor<64x128xf16>, %b: tensor<128x96xf16>, %bias: tensor<96xf32>) -> tensor<64x96xf32> {
  %0 = "tessera.matmul"(%a, %b) : (tensor<64x128xf16>, tensor<128x96xf16>) -> tensor<64x96xf32>
  %bb = "tessera.broadcast"(%bias) {shape = [64, 96]} : (tensor<96xf32>) -> tensor<64x96xf32>
  %1 = "tessera.add"(%0, %bb) : (tensor<64x96xf32>, tensor<64x96xf32>) -> tensor<64x96xf32>
  %2 = "tessera.gelu"(%1) : (tensor<64x96xf32>) -> tensor<64x96xf32>
  return %2 : tensor<64x96xf32>
}

// Activation alone fuses too (no bias operand, has_bias = false).
// TARGET-LABEL: func.func @gemm_relu
// TARGET: tessera_apple.gpu.matmul2d_epilogue %{{.*}}, %{{.*}} {
// TARGET-SAME: act = "relu"
// CALL-LABEL: func.func @gemm_relu
// CALL: tessera_apple.gpu.kernel_call %arg0, %arg1 {
// CALL-SAME: tessera_apple.act = "relu"
// CALL-SAME: tessera_apple.has_bias = false
func.func @gemm_relu(%a: tensor<64x128xbf16>, %b: tensor<128x96xbf16>) -> tensor<64x96xf32> {
  %0 = "tessera.matmul"(%a, %b) : (tensor<64x128xbf16>, tensor<128x96xbf16>) -> tensor<64x96xf32>
  %1 = "tessera.relu"(%0) : (tensor<64x96xf32>) -> tensor<64x96xf32>
  return %1 : tensor<64x96xf32>
}

// A product with two consumers is not fused: the epilogue would be evaluated
// for one and the plain product still needed for the other.
// TARGET-LABEL: func.func @gemm_two_consumers
// TARGET: tessera_apple.gpu.matmul2d %
// TARGET-NOT: matmul2d_epilogue
// TARGET: tessera.gelu
func.func @gemm_two_consumers(%a: tensor<64x128xf16>, %b: tensor<128x96xf16>) -> (tensor<64x96xf32>, tensor<64x96xf32>) {
  %0 = "tessera.matmul"(%a, %b) : (tensor<64x128xf16>, tensor<128x96xf16>) -> tensor<64x96xf32>
  %1 = "tessera.gelu"(%0) : (tensor<64x96xf32>) -> tensor<64x96xf32>
  return %0, %1 : tensor<64x96xf32>, tensor<64x96xf32>
}
