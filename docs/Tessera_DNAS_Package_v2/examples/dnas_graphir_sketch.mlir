// dnas_graphir_sketch.mlir (illustrative, non-executable)

tessera.graph.func @encoder_block(%x: tensor<?x128xf32>) -> tensor<?x128xf32> {
  %tau     = tessera.graph.constant 4.0 : f32
  %alpha   = tessera.graph.arch.parameter {num_candidates = 2} : tensor<2xf32>
  %gate    = tessera.graph.arch.gumbel_softmax(%alpha, %tau) : tensor<2xf32>

  // Candidates
  %y0 = tessera.graph.op.linear(%x) : tensor<?x128xf32>
  %y1 = tessera.graph.op.gmlp(%x)   : tensor<?x128xf32>

  %y  = tessera.graph.arch.weighted_sum %gate, [%y0, %y1]
        : tensor<2xf32>, tensor<?x128xf32> -> tensor<?x128xf32>
  return %y : tensor<?x128xf32>
}

// Attach schedule candidates at a matmul site inside linear/gmlp
%tm = tessera.schedule.choice @tile_m {values=[64,128]} : i64
%tn = tessera.schedule.choice @tile_n {values=[128,256]} : i64
%tk = tessera.schedule.choice @tile_k {values=[32,64]}   : i64
%st = tessera.schedule.choice @stages {values=[2,3,4]}   : i64

tessera.schedule.apply @matmul0 tile(%tm,%tn,%tk) stages(%st)
