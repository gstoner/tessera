
//===- gemm_meta_schedule.mlir --------------------------------------------===//
// Transform dialect script (resolvable placeholders).
// We use nonstandard attributes `tile_sizes_sym` and `stages_sym` that are
// resolved by the ResolveTesseraAttrs pass into concrete constants.
//----------------------------------------------------------------------------
module {
  transform.with_pdl_patterns {
    ^bb0(%mod: !transform.any_op):
      %f_all = transform.match ops{["func.func"]} in %mod
      transform.foreach_match in %f_all : (!transform.any_op) -> !transform.any_op {
      ^bb1(%f: !transform.any_op):
        %mm = transform.match ops{["linalg.matmul","linalg.generic"]} in %f

        // symbolic placeholders (strings) -> replaced by pass with i64 array
        %tiled, %loops = transform.structured.tile %mm 
          { tile_sizes_sym = ["tessera.BLOCK_M", "tessera.BLOCK_N", "tessera.BLOCK_K"] }

        // pipeline stages: string -> integer attr 'stages'
        transform.structured.pipeline %tiled { stages_sym = "tessera.num_stages" }

        transform.structured.vectorize %tiled
        transform.yield %tiled : !transform.any_op
      }
  }
}
