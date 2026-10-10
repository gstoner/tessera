module attributes {tessera.target = "nvidia_sm120", tessera.arch = "sm_120",
      tessera.attention_shape_bounds = array<i64: 1, 2, 1, 9, 11, 4, 3>} {
      func.func @attention(%q: tensor<1x2x?x4xf32>, %k: tensor<1x1x?x4xf32>, %v: tensor<1x1x?x3xf32>, %bias: tensor<1x2x?x?xf32>) -> tensor<1x2x?x3xf32>
        attributes {tessera.autodiff = "forward",
                     tessera.autodiff.wrt_indices = [0, 1, 2, 3]} {
        %o = "tessera.flash_attn"(%q, %k, %v, %bias) {causal = false,
          head_dim = 4 : i64, operandSegmentSizes = array<i32: 1, 1, 1, 1>}
          : (tensor<1x2x?x4xf32>, tensor<1x1x?x4xf32>, tensor<1x1x?x3xf32>, tensor<1x2x?x?xf32>) -> tensor<1x2x?x3xf32>
        return %o : tensor<1x2x?x3xf32>
      }
    }