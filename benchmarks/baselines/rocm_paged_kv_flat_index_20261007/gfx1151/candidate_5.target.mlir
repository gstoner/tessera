module attributes {tessera.arch = "gfx1151", tessera.ir.version = "1.0", tessera.pipeline.arch = "gfx1151", tessera.pipeline.backend_codegen = "rocdl_hsaco", tessera.pipeline.family = "paged_kv", tessera.pipeline.output = "target", tessera.pipeline.schema = "tessera.executable_pipeline.v1", tessera.pipeline.target_ir_consumer = "tessera_rocm", tessera.pipeline.tile_producer = "content_addressed_tile", tessera.target = "rocm_gfx1151"} {
  tessera_rocm.paged_kv_read {arch = "gfx1151", name = "tessera_rocm_paged_kv_2f4d73b62a652530", source = "tile.paged_kv_read_kernel"}
}

