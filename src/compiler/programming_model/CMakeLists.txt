# Tessera Programming Model v1.1
# Canonical programming-model component.
# Provides: tile/cache/schedule/moe ODS-generated headers + PassPipelines driver.
#
# Exported targets:
#   TesseraPMTableGen   — tablegen inc targets (headers only, no .cpp)
#   TesseraPM           — PassPipelines compiler driver library

set(PM_TD_INC_DIR ${CMAKE_CURRENT_SOURCE_DIR}/ir)

# ------------------------------------------------------------------
# TableGen — one pair of .h.inc / .cpp.inc per dialect .td file
# ------------------------------------------------------------------
foreach(_td_rel
    tile/TileMemoryOps
    cache/CacheOps
    schedule/ScheduleMeshPipelineOps
    moe/MoeOps
)
    get_filename_component(_td_name ${_td_rel} NAME_WE)
    set(_td_file ${PM_TD_INC_DIR}/${_td_rel}.td)

    mlir_tablegen(${_td_name}.h.inc   -gen-op-decls -I ${PM_TD_INC_DIR} DEPENDS ${_td_file})
    mlir_tablegen(${_td_name}.cpp.inc -gen-op-defs  -I ${PM_TD_INC_DIR} DEPENDS ${_td_file})
endforeach()

add_public_tablegen_target(TesseraPMTableGen)

# ------------------------------------------------------------------
# PassPipelines compiler driver
# ------------------------------------------------------------------
add_library(TesseraPM
    tools/tessera-opt/PassPipelinesPM11.cpp
)

target_include_directories(TesseraPM
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${CMAKE_CURRENT_BINARY_DIR}
)

target_link_libraries(TesseraPM
    PUBLIC
        MLIRIR
        MLIRPass
        MLIRFuncDialect
        MLIRTransforms
)

add_dependencies(TesseraPM TesseraPMTableGen)
