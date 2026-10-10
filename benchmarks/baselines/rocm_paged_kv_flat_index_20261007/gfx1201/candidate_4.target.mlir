module attributes {tessera.arch = "gfx1201", tessera.ir.version = "1.0", tessera.pipeline.arch = "gfx1201", tessera.pipeline.backend_codegen = "rocdl_hsaco", tessera.pipeline.family = "paged_kv", tessera.pipeline.output = "target", tessera.pipeline.schema = "tessera.executable_pipeline.v1", tessera.pipeline.target_ir_consumer = "tessera_rocm", tessera.pipeline.tile_producer = "content_addressed_tile", tessera.target = "rocm_gfx1201"} {
  tessera_rocm.paged_kv_read {arch = "gfx1201", name = "tessera_rocm_paged_kv_f1972d4e63f299f4", source = "tile.paged_kv_read_kernel"}
}

