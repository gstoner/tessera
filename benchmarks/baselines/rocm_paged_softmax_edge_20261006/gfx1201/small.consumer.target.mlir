module attributes {tessera.arch = "gfx1201", tessera.frontend.authority = "tracer", tessera.ir.version = "1.0", tessera.pipeline.arch = "gfx1201", tessera.pipeline.backend_codegen = "rocdl_hsaco", tessera.pipeline.family = "softmax", tessera.pipeline.output = "target", tessera.pipeline.schema = "tessera.executable_pipeline.v1", tessera.pipeline.target_ir_consumer = "tessera_rocm", tessera.pipeline.tile_producer = "content_addressed_tile", tessera.target = "rocm"} {
  tessera_rocm.softmax {accum = "f32", arch = "gfx1201", axis = -1 : i64, exp_mode = "accurate", ftz = false, name = "tessera_rocm_softmax_78353eb6b553d974", source = "tile.softmax_kernel"}
}

