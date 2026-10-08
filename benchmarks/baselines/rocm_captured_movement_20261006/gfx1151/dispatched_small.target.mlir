module attributes {tessera.arch = "gfx1151", tessera.frontend.authority = "tracer", tessera.ir.version = "1.0", tessera.pipeline.arch = "gfx1151", tessera.pipeline.backend_codegen = "rocdl_hsaco", tessera.pipeline.family = "moe_dispatch", tessera.pipeline.output = "target", tessera.pipeline.schema = "tessera.executable_pipeline.v1", tessera.pipeline.target_ir_consumer = "tessera_rocm", tessera.pipeline.tile_producer = "content_addressed_tile", tessera.target = "rocm_gfx1151"} {
  tessera_rocm.moe_dispatch {arch = "gfx1151", name = "tessera_rocm_moe_dispatch_e59f0cd4b3176780", source = "tile.moe_dispatch_kernel"}
}

