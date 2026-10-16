module attributes {tessera.autodiff = "forward", tessera.autodiff.wrt = ["a", "b"], tessera.autodiff.wrt_indices = [0, 1], tessera.frontend.authority = "tracer", tessera.ir.version = "1.0"} {
  func.func @product(%arg0: tensor<7x32xf16>, %arg1: tensor<32x9xf16>) -> tensor<7x9xf32> attributes {tessera.autodiff = "forward", tessera.autodiff.jvp = @product__jvp, tessera.autodiff.wrt = ["a", "b"], tessera.autodiff.wrt_indices = [0, 1], tessera.frontend.authority = "tracer", tessera.structured_cfg.blocks = 2 : i64, tessera.structured_cfg.digest = "bc06e0f096d1b6c2ee50984eb5077c4cd597214e79da7f33196d5541be44c1cd", tessera.structured_cfg.schema = "tessera.structured_cfg.v1"} {
    %0 = tessera.rmsnorm %arg0 {eps = 1.000000e-05 : f64, tessera.effect_kind = "pure"} : (tensor<7x32xf16>) -> tensor<7x32xf16>
    %1 = tessera.relu %0 {tessera.effect_kind = "pure"} : (tensor<7x32xf16>) -> tensor<7x32xf16>
    %2 = tessera.matmul %1, %arg1 {output_dtype = "fp32", tessera.effect_kind = "pure"} : (tensor<7x32xf16>, tensor<32x9xf16>) -> tensor<7x9xf32>
    return %2 : tensor<7x9xf32>
  }
  func.func private @product__jvp(%arg0: tensor<7x32xf16>, %arg1: tensor<32x9xf16>, %arg2: tensor<7x32xf16>, %arg3: tensor<32x9xf16>) -> (tensor<7x9xf32>, tensor<7x9xf32>) attributes {tessera.autodiff.forward = @product, tessera.autodiff.role = "jvp"} {
    %0 = tessera.rmsnorm %arg0 {eps = 1.000000e-05 : f64, tessera.effect_kind = "pure"} : (tensor<7x32xf16>) -> tensor<7x32xf16>
    %center, %inverse_scale = tessera.normalization_stats %arg0 {centered = false} : (tensor<7x32xf16>) -> (tensor<7xf16>, tensor<7xf16>)
    %1 = tessera.broadcast_in_dim %inverse_scale, %arg0 {broadcast_dimensions = [0]} : (tensor<7xf16>, tensor<7x32xf16>) -> tensor<7x32xf16>
    %2 = tessera.mul %arg0, %1 : (tensor<7x32xf16>, tensor<7x32xf16>) -> tensor<7x32xf16>
    %3 = tessera.mul %arg2, %2 : (tensor<7x32xf16>, tensor<7x32xf16>) -> tensor<7x32xf16>
    %4 = tessera.reduce %3 {axis = 1 : i64, kind = "mean"} : (tensor<7x32xf16>) -> tensor<7xf16>
    %5 = tessera.broadcast_in_dim %4, %arg0 {broadcast_dimensions = [0]} : (tensor<7xf16>, tensor<7x32xf16>) -> tensor<7x32xf16>
    %6 = tessera.mul %2, %5 : (tensor<7x32xf16>, tensor<7x32xf16>) -> tensor<7x32xf16>
    %7 = tessera.sub %arg2, %6 : (tensor<7x32xf16>, tensor<7x32xf16>) -> tensor<7x32xf16>
    %8 = tessera.mul %7, %1 : (tensor<7x32xf16>, tensor<7x32xf16>) -> tensor<7x32xf16>
    %9 = tessera.relu %0 {tessera.effect_kind = "pure"} : (tensor<7x32xf16>) -> tensor<7x32xf16>
    %10 = tessera.compare_scalar %0 {predicate = "gt", rhs = 0.000000e+00 : f64} : (tensor<7x32xf16>) -> tensor<7x32xi1>
    %11 = tessera.masked_fill %8, %10 {value = 0.000000e+00 : f64} : (tensor<7x32xf16>, tensor<7x32xi1>) -> tensor<7x32xf16>
    %12 = tessera.matmul %9, %arg1 {output_dtype = "fp32", tessera.effect_kind = "pure"} : (tensor<7x32xf16>, tensor<32x9xf16>) -> tensor<7x9xf32>
    %13 = tessera.matmul %11, %arg1 : (tensor<7x32xf16>, tensor<32x9xf16>) -> tensor<7x9xf32>
    %14 = tessera.matmul %9, %arg3 : (tensor<7x32xf16>, tensor<32x9xf16>) -> tensor<7x9xf32>
    %15 = tessera.add %13, %14 : (tensor<7x9xf32>, tensor<7x9xf32>) -> tensor<7x9xf32>
    return %12, %15 : tensor<7x9xf32>, tensor<7x9xf32>
  }
}

