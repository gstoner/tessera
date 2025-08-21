
// 01_graph.mlir -- High-level Graph IR
tosa.module @m {
  func.func @inference(%A: tensor<4x8xf32>, %B: tensor<8x16xf32>,
                       %mean: tensor<16xf32>, %var: tensor<16xf32>,
                       %gamma: tensor<16xf32>, %beta: tensor<16xf32>) -> tensor<4x16xf32> {
    %C = tgraph.gemm %A, %B {alpha = 1.0 : f32, beta = 0.0 : f32} : tensor<4x16xf32>
    %Y = tgraph.batch_norm %C, %mean, %var, %gamma, %beta {eps = 1.0e-5 : f32} : tensor<4x16xf32>
    return %Y : tensor<4x16xf32>
  }
}
