\
module attributes {torch.debug_module_name = "matmul"} {
  func.func @matmul(%lhs: tensor<128x128xbf16>, %rhs: tensor<128x128xbf16>) -> tensor<128x128xbf16> {
    %0 = stablehlo.dot_general %lhs, %rhs,
      contracting_dims = [1], batch_dims = [],
      rhs_contracting_dims = [0], rhs_batch_dims = []
      : (tensor<128x128xbf16>, tensor<128x128xbf16>) -> tensor<128x128xbf16>
    return %0 : tensor<128x128xbf16>
  }
}
