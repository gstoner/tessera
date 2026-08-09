// RUN: tessera-opt --tessera-autodiff %s | FileCheck %s --check-prefix=INPLACE
// RUN: tessera-opt --tessera-autodiff-paired %s | FileCheck %s --check-prefix=PAIRED
//
// AD-CORE-LINEAR-1 — transpose and reshape use the compiler-owned linear
// transposition interface.  Neither operation declares a handwritten
// AdjointInterface implementation.

module {
  func.func @linear(%x: tensor<2x3xf32>) -> tensor<6xf32>
      attributes {tessera.autodiff = "reverse"} {
    %t = "tessera.transpose"(%x) :
        (tensor<2x3xf32>) -> tensor<3x2xf32>
    %y = "tessera.reshape"(%t) :
        (tensor<3x2xf32>) -> tensor<6xf32>
    return %y : tensor<6xf32>
  }
}

// INPLACE-LABEL: func.func @linear
// INPLACE: %[[SEED:.*]] = arith.constant {{.*}}dense<1.000000e+00> : tensor<6xf32>
// INPLACE: %[[RESHAPED:.*]] = tessera.reshape %[[SEED]] {{.*}} : (tensor<6xf32>) -> tensor<3x2xf32>
// INPLACE: %[[TRANSPOSED:.*]] = tessera.transpose %[[RESHAPED]] {{.*}} : (tensor<3x2xf32>) -> tensor<2x3xf32>
// INPLACE: return {{.*}}, %[[TRANSPOSED]] : tensor<6xf32>, tensor<2x3xf32>

// PAIRED-LABEL: func.func @linear__bwd
// PAIRED: %[[RESHAPED:.*]] = tessera.reshape {{.*}} : (tensor<6xf32>) -> tensor<3x2xf32>
// PAIRED: %[[TRANSPOSED:.*]] = tessera.transpose %[[RESHAPED]] : (tensor<3x2xf32>) -> tensor<2x3xf32>
// PAIRED: return %[[TRANSPOSED]] : tensor<2x3xf32>
