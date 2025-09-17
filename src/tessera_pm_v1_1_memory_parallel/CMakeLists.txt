# Add in v1.1 ops and pipeline
add_library(TesseraPM11
  ir/tile/TileMemoryOps.td
  ir/cache/CacheOps.td
  ir/schedule/ScheduleMeshPipelineOps.td
  ir/moe/MoeOps.td
  tools/tessera-opt/PassPipelinesPM11.cpp
)

# (In real build, use mlir_tablegen + add_public_tablegen_target etc.)
