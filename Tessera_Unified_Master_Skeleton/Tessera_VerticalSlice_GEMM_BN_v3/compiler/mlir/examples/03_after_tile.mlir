
// 03_after_tile.mlir -- After LowerScheduleToTile
tosa.module @m {
  func.func @inference(%A: tensor<4x8xf32>, %B: tensor<8x16xf32>,
                       %mean: tensor<16xf32>, %var: tensor<16xf32>,
                       %gamma: tensor<16xf32>, %beta: tensor<16xf32>) -> tensor<4x16xf32> {
    %C = ttile.gemm_mma %A, %B { bm = 128, bn = 128, bk = 64, epilogue = "none" } : tensor<4x16xf32>
    %Y = ttile.batch_norm_vec %C, %mean, %var, %gamma, %beta { eps = 1.0e-5 : f32, vecWidth = 8 } : tensor<4x16xf32>
    return %Y : tensor<4x16xf32>
  }
}
