// ODS-WIRE-2: `tessera.istft_jvp` has a producer (ISTFTOp::buildTangent under
// --tessera-autodiff-forward) and, since this fixture, a consumer: the
// GraphToSchedule arm resolves every default, derives tangent activity from
// the IR, and binds the product into one hashed contract plus a matching
// schedule.artifact. The native JVP plugin reads that contract instead of
// re-deriving it from the source op's Python kwargs (Decision #31).
//
// RUN: tessera-opt %s --split-input-file --tessera-autodiff-forward --tessera-graph-to-schedule | FileCheck %s

// Both tangents active on Zen 5; defaults (axis=-1, onesided, uncentered,
// backward) are resolved and carried on the op (Decision #32).
module attributes {tessera.target = "x86", tessera.arch = "zen5-avx512"} {
  func.func @istft_both(%x: tensor<4x5xcomplex<f32>>, %w: tensor<8xf32>)
      -> tensor<20xf32> attributes {tessera.autodiff = "forward"} {
    %y = "tessera.istft"(%x, %w) {hop = 4 : i64, logical_length = 8 : i64}
        : (tensor<4x5xcomplex<f32>>, tensor<8xf32>) -> tensor<20xf32>
    return %y : tensor<20xf32>
  }
}

// CHECK-LABEL: func.func private @istft_both__jvp(
// CHECK: tessera.istft_jvp
// CHECK-SAME: axis = 1 : i64
// CHECK-SAME: output_length = 20 : i64
// CHECK-SAME: schedule.active_tangents = array<i64: 0, 1>
// CHECK-SAME: schedule.arch = "zen5-avx512"
// CHECK-SAME: schedule.artifact_hash = "[[HASH:[0-9a-f]{64}]]"
// CHECK-SAME: schedule.jvp_contract = "schema=tessera.spectral_jvp.v1;kind=tessera.istft;target=x86;arch=zen5-avx512;spectrum=tensor<4x5xcomplex<f32>>;window=tensor<8xf32>;tangent=tensor<20xf32>;axis=1;logical_length=8;hop=4;frames=4;center=0;onesided=1;pad_mode=constant;output_length=20;normalization=backward;
// CHECK-SAME: active_tangents=0,1;
// CHECK-SAME: window_broadcast = "trailing_batch_broadcast_v1"
// CHECK: schedule.artifact {arch = "zen5-avx512", hash = "[[HASH]]", numeric_policy = "jvp;backward", shape_key = "family=spectral_jvp;kind=tessera.istft"

// -----

// Only the spectrum is differentiated: the transform materializes the window
// tangent as a zero splat, and the arm reads that as an inactive tangent
// rather than being told the wrt set (Decision #30).
module attributes {tessera.target = "rocm", tessera.arch = "gfx1201"} {
  func.func @istft_spectrum_only(%x: tensor<2x5x5xcomplex<f32>>, %w: tensor<8xf16>)
      -> tensor<2x12xf16> attributes {
        tessera.autodiff = "forward", tessera.autodiff.wrt_indices = [0]} {
    %y = "tessera.istft"(%x, %w) {
      hop = 4 : i64, logical_length = 8 : i64, center = true,
      normalization = "ortho", output_length = 12 : i64}
        : (tensor<2x5x5xcomplex<f32>>, tensor<8xf16>) -> tensor<2x12xf16>
    return %y : tensor<2x12xf16>
  }
}

// CHECK-LABEL: func.func private @istft_spectrum_only__jvp(
// CHECK: %[[ZW:.*]] = arith.constant dense<0.000000e+00> : tensor<8xf16>
// CHECK: tessera.istft_jvp %{{.*}}, %{{.*}}, %{{.*}}, %[[ZW]]
// CHECK-SAME: axis = 2 : i64
// CHECK-SAME: center = true
// CHECK-SAME: numeric_policy = {accum = "fp32", storage = "fp16"}
// CHECK-SAME: output_length = 12 : i64
// CHECK-SAME: schedule.active_tangents = array<i64: 0>
// CHECK-SAME: schedule.arch = "gfx1201"
// CHECK-SAME: frames=5;center=1;onesided=1;pad_mode=constant;output_length=12;normalization=ortho;
// CHECK-SAME: numeric_storage=fp16;numeric_accum=fp32;active_tangents=0;
// CHECK: schedule.artifact {arch = "gfx1201"
