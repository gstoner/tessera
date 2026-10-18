module attributes {tessera.arch = "sm_120", tessera.target = "nvidia_sm120"} {
  func.func private @__native_row_body(%x: tensor<17x35xf16>, %g: tensor<17x35xf16>) -> tensor<17x35xf16> {
    %0 = tessera.rmsnorm %x {eps = 1.000000e-05 : f64, numeric_policy = {accum = "fp32"}, tessera.effect_kind = "pure"} : (tensor<17x35xf16>) -> tensor<17x35xf16>
    %5 = tessera.compare_scalar %0 {predicate = "gt", rhs = 0.000000e+00 : f64} : (tensor<17x35xf16>) -> tensor<17x35xi1>
    %6 = tessera.masked_fill %g, %5 {value = 0.000000e+00 : f64} : (tensor<17x35xf16>, tensor<17x35xi1>) -> tensor<17x35xf16>
    %7 = tessera.cast %x : (tensor<17x35xf16>) -> tensor<17x35xf32>
    %8 = tessera.cast %6 : (tensor<17x35xf16>) -> tensor<17x35xf32>
    %center, %inverse_scale = tessera.normalization_stats %7 {centered = false} : (tensor<17x35xf32>) -> (tensor<17xf32>, tensor<17xf32>)
    %9 = tessera.broadcast_in_dim %inverse_scale, %7 {broadcast_dimensions = [0]} : (tensor<17xf32>, tensor<17x35xf32>) -> tensor<17x35xf32>
    %10 = tessera.mul %7, %9 : (tensor<17x35xf32>, tensor<17x35xf32>) -> tensor<17x35xf32>
    %11 = tessera.mul %10, %9 : (tensor<17x35xf32>, tensor<17x35xf32>) -> tensor<17x35xf32>
    %12 = tessera.mul %7, %11 : (tensor<17x35xf32>, tensor<17x35xf32>) -> tensor<17x35xf32>
    %13 = tessera.reduce %12 {axis = 1 : i64, kind = "mean"} : (tensor<17x35xf32>) -> tensor<17xf32>
    %14 = tessera.broadcast_in_dim %13, %7 {broadcast_dimensions = [0]} : (tensor<17xf32>, tensor<17x35xf32>) -> tensor<17x35xf32>
    %15 = tessera.mul %8, %11 : (tensor<17x35xf32>, tensor<17x35xf32>) -> tensor<17x35xf32>
    %16 = tessera.reduce %15 {axis = 1 : i64, kind = "mean"} : (tensor<17x35xf32>) -> tensor<17xf32>
    %17 = tessera.broadcast_in_dim %16, %7 {broadcast_dimensions = [0]} : (tensor<17xf32>, tensor<17x35xf32>) -> tensor<17x35xf32>
    %18 = tessera.mul %7, %17 : (tensor<17x35xf32>, tensor<17x35xf32>) -> tensor<17x35xf32>
    %19 = tessera.mul %8, %14 : (tensor<17x35xf32>, tensor<17x35xf32>) -> tensor<17x35xf32>
    %20 = tessera.sub %19, %18 : (tensor<17x35xf32>, tensor<17x35xf32>) -> tensor<17x35xf32>
    %cst = arith.constant dense<9.99999974E-6> : tensor<f32>
    %21 = tessera.broadcast_in_dim %cst, %7 {broadcast_dimensions = []} : (tensor<f32>, tensor<17x35xf32>) -> tensor<17x35xf32>
    %22 = tessera.mul %21, %9 : (tensor<17x35xf32>, tensor<17x35xf32>) -> tensor<17x35xf32>
    %23 = tessera.mul %22, %9 : (tensor<17x35xf32>, tensor<17x35xf32>) -> tensor<17x35xf32>
    %24 = tessera.mul %8, %23 : (tensor<17x35xf32>, tensor<17x35xf32>) -> tensor<17x35xf32>
    %25 = tessera.add %20, %24 : (tensor<17x35xf32>, tensor<17x35xf32>) -> tensor<17x35xf32>
    %26 = tessera.mul %25, %9 : (tensor<17x35xf32>, tensor<17x35xf32>) -> tensor<17x35xf32>
    %27 = tessera.cast %26 : (tensor<17x35xf32>) -> tensor<17x35xf16>
    return %27 : tensor<17x35xf16>
  }
}
