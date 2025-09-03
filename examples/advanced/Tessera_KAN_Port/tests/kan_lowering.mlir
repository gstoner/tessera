// RUN: tessera-opt %s -lower-kan-to-tessera | FileCheck %s

func.func @toy(%arg0: tensor<4x8xf32>) -> tensor<4x16xf32> {
  %phi = kan.bspline_eval %arg0 { degree = 3 : i64, grid_min = 0.0 : f32, grid_max = 1.0 : f32, grid_size = 12 : i64 } : (tensor<4x8xf32>) -> tensor<4x8x14xf32>
  %Wphi = tensor.empty() : tensor<112x16xf32>
  %Wbase = tensor.empty() : tensor<8x16xf32>
  %y = kan.linear_mix %phi, %Wphi, %arg0, %Wbase : (tensor<4x8x14xf32>, tensor<112x16xf32>, tensor<4x8xf32>, tensor<8x16xf32>) -> tensor<4x16xf32>
  return %y : tensor<4x16xf32>
}

// CHECK: linalg.matmul