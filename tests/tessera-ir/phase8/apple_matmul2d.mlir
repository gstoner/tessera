// REQUIRES: tessera-apple-backend
//
// APPLE-MATMUL2D-1 (2026-09-15): the shared TilingPass turns a Graph-IR matmul
// into the canonical M/N/K contract; the Apple Metal 4 consumer re-forms it as
// two `tensor_view` bindings and one `matmul2d`, and the call lowering hands
// the verified view ABI to the runtime's matmul2d symbols. Driving all three
// in one run proves the lane consumes the real shared form.
//
// RUN: tessera-opt %s --tessera-tiling --tessera-apple-canonical-gemm-matmul2d \
// RUN:   --allow-unregistered-dialect | FileCheck %s --check-prefix=TARGET
// RUN: tessera-opt %s --tessera-tiling --tessera-apple-canonical-gemm-matmul2d \
// RUN:   --tessera-apple-matmul2d-to-call --allow-unregistered-dialect \
// RUN:   | FileCheck %s --check-prefix=CALL
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
// CALL-LABEL: func.func @gemm_f16
// CALL-NOT: tessera_apple.gpu.matmul2d
// CALL: tessera_apple.gpu.kernel_call %arg0, %arg1
// CALL-SAME: abi = "mtl4_matmul2d_view"
// CALL-SAME: dtype = "f16xf16"
// CALL-SAME: op_kind = "mtl4_matmul2d"
// CALL-SAME: symbol = "tessera_apple_gpu_mtl4_matmul2d_f16"
// CALL-SAME: tessera_apple.a_inner = 128
// CALL-SAME: tessera_apple.a_outer = 64
// CALL-SAME: tessera_apple.a_stride = 128
// CALL-SAME: tessera_apple.accumulate = "fp32"
// CALL-SAME: tessera_apple.b_inner = 96
// CALL-SAME: tessera_apple.b_stride = 96
func.func @gemm_f16(%a: tensor<64x128xf16>, %b: tensor<128x96xf16>) -> tensor<64x96xf32> {
  %0 = "tessera.matmul"(%a, %b) : (tensor<64x128xf16>, tensor<128x96xf16>) -> tensor<64x96xf32>
  return %0 : tensor<64x96xf32>
}

// FP8 on both operands: K and N on Apple's 128-element quantum, so the packed
// operands are legal views and the call names the low-precision symbol with
// its format code (0 = e4m3/e4m3). The ragged M = 100 is zero-padded to the
// shared tiling pass's 112 before the view is taken and sliced back after --
// the IR states the padding, the runtime is never asked to guess the tail.
// TARGET-LABEL: func.func @gemm_e4m3
// TARGET: tensor.insert_slice %arg0 into {{.*}} : tensor<100x256xf8E4M3FN> into tensor<112x256xf8E4M3FN>
// TARGET: tessera_apple.gpu.tensor_view {{.*}} {byte_offset = 0 : i64, extents = array<i64: 256, 112>, strides = array<i64: 1, 256>} : tensor<112x256xf8E4M3FN> -> <f8E4M3FN>
// TARGET: tessera_apple.gpu.matmul2d
// TARGET-SAME: -> tensor<112x128xf32>
// TARGET: tensor.extract_slice {{.*}} : tensor<112x128xf32> to tensor<100x128xf32>
// CALL-LABEL: func.func @gemm_e4m3
// CALL: tessera_apple.gpu.kernel_call
// CALL-SAME: dtype = "f8E4M3FNxf8E4M3FN"
// CALL-SAME: symbol = "tessera_apple_gpu_mtl4_matmul2d_lowp"
// CALL-SAME: tessera_apple.lowp_format = 0
func.func @gemm_e4m3(%a: tensor<100x256xf8E4M3FN>, %b: tensor<256x128xf8E4M3FN>) -> tensor<100x128xf32> {
  %0 = "tessera.matmul"(%a, %b) : (tensor<100x256xf8E4M3FN>, tensor<256x128xf8E4M3FN>) -> tensor<100x128xf32>
  return %0 : tensor<100x128xf32>
}

// The weight-only pair: half activations against an FP4 weight matrix whose
// row stride sits on the 256-element FP4 quantum (format code 5).
// CALL-LABEL: func.func @gemm_half_e2m1
// CALL: tessera_apple.gpu.kernel_call
// CALL-SAME: dtype = "f16xf4E2M1FN"
// CALL-SAME: tessera_apple.lowp_format = 5
func.func @gemm_half_e2m1(%a: tensor<64x128xf16>, %b: tensor<128x256xf4E2M1FN>) -> tensor<64x256xf32> {
  %0 = "tessera.matmul"(%a, %b) : (tensor<64x128xf16>, tensor<128x256xf4E2M1FN>) -> tensor<64x256xf32>
  return %0 : tensor<64x256xf32>
}
