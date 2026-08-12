// RUN: tessera-opt %s --tessera-autodiff-forward | FileCheck %s

module {
  func.func @filter_product(
      %x: tensor<8xcomplex<f32>>, %filter: tensor<8xcomplex<f32>>)
      -> tensor<8xcomplex<f32>> attributes {tessera.autodiff = "forward"} {
    %y = "tessera.spectral_filter"(%x, %filter) {
      axis = -1 : i64, logical_length = 8 : i64,
      normalization = "backward", spectrum_layout = "full_complex"
    } : (tensor<8xcomplex<f32>>, tensor<8xcomplex<f32>>)
        -> tensor<8xcomplex<f32>>
    return %y : tensor<8xcomplex<f32>>
  }

  func.func @istft_fixed_window(
      %x: tensor<3x5xcomplex<f32>>, %window: tensor<8xf32>)
      -> tensor<16xf32> attributes {
        tessera.autodiff = "forward", tessera.autodiff.wrt_indices = [0]} {
    %y = "tessera.istft"(%x, %window) {
      axis = -1 : i64, logical_length = 8 : i64, hop = 4 : i64,
      normalization = "backward",
      spectrum_layout = "half_spectrum_nyquist_explicit"
    } : (tensor<3x5xcomplex<f32>>, tensor<8xf32>) -> tensor<16xf32>
    return %y : tensor<16xf32>
  }
}

// CHECK-LABEL: func.func private @filter_product__jvp(
// CHECK: %[[P:.*]] = tessera.spectral_filter %{{.*}}, %{{.*}}
// CHECK: %[[DX:.*]] = tessera.spectral_filter %{{.*}}, %{{.*}}
// CHECK: %[[DF:.*]] = tessera.spectral_filter %{{.*}}, %{{.*}}
// CHECK: %[[SUM:.*]] = tessera.add %[[DX]], %[[DF]]
// CHECK: return %[[P]], %[[SUM]]
// CHECK-LABEL: func.func private @istft_fixed_window__jvp(
// CHECK: %[[IP:.*]] = tessera.istft %{{.*}}, %{{.*}}
// CHECK: %[[ZW:.*]] = arith.constant dense<0.000000e+00> : tensor<8xf32>
// CHECK: %[[DI:.*]] = tessera.istft_jvp %{{.*}}, %{{.*}}, %{{.*}}, %[[ZW]]
// CHECK: return %[[IP]], %[[DI]]
