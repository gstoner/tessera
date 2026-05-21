# Grid-AI core benchmark

Tiny domain-neutral benchmark for regional gridded AI models.  It is a library
workload built on Tessera primitives, not a compiler feature and not a
weather-specific application.

It exercises:

| Piece | Implementation |
| --- | --- |
| tiled field input | `tile_field(..., spatial_axes=(1, 2))` |
| local stencil feature | `local_stencil_feature` 5-point NHWC feature |
| 2D local attention | `tessera.ops.attn_local_window_2d` |
| conv/fused block | `tessera.ops.conv2d` + `tessera.ops.fused_epilogue` |
| deterministic noise | `tessera.rng.RNGKey` + `normal` |
| halo oracle | `periodic_halo_oracle` compared to mock halo transport |

The matching compiler fixture is
`tests/tessera-ir/phase7/grid_ai_core_ir_visible.mlir`.  It keeps the stencil,
2D attention, conv/fused block, RNG, and halo transport path visible in one
`schedule.mesh.region`.
