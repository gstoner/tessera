# Tile Optimization + Flash Attention v4
# Canonical component: promoted from Tile_Optimization_FA4/v1_3
# Adds Queue dialect (v1.3) and backward-attention support over v1/v1_1/v1_2.
#
# Exported targets:
#   TesseraAttnDialect      — Attn dialect verifiers + ODS-generated headers
#   TesseraQueueDialect     — Queue dialect verifiers + ODS-generated headers
#   TesseraFA4Conversions   — Schedule-to-Target and Tile-to-PTX lowering passes

# ------------------------------------------------------------------
# Attn dialect
# ------------------------------------------------------------------
set(ATTN_TD ${CMAKE_CURRENT_SOURCE_DIR}/include/tessera/Dialect/Attn/Attn.td)

mlir_tablegen(AttnOps.h.inc        -gen-op-decls     -I ${CMAKE_CURRENT_SOURCE_DIR}/include DEPENDS ${ATTN_TD})
mlir_tablegen(AttnOps.cpp.inc      -gen-op-defs      -I ${CMAKE_CURRENT_SOURCE_DIR}/include DEPENDS ${ATTN_TD})
mlir_tablegen(AttnDialect.h.inc    -gen-dialect-decls -I ${CMAKE_CURRENT_SOURCE_DIR}/include DEPENDS ${ATTN_TD})
mlir_tablegen(AttnDialect.cpp.inc  -gen-dialect-defs  -I ${CMAKE_CURRENT_SOURCE_DIR}/include DEPENDS ${ATTN_TD})
add_public_tablegen_target(TesseraAttnTableGen)

add_mlir_library(TesseraAttnDialect
    lib/Dialect/Attn/AttnVerifiers.cpp
    lib/Dialect/Attn/AttnOps.cpp          # Phase 3: verifiers + ODS includes

    DEPENDS
        TesseraAttnTableGen

    LINK_LIBS PUBLIC
        MLIRIR
        MLIRSupport
)
target_include_directories(TesseraAttnDialect
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${CMAKE_CURRENT_BINARY_DIR}
)

# ------------------------------------------------------------------
# Queue dialect (v1.3 addition)
# ------------------------------------------------------------------
set(QUEUE_TD ${CMAKE_CURRENT_SOURCE_DIR}/include/tessera/Dialect/Queue/Queue.td)

mlir_tablegen(QueueOps.h.inc   -gen-op-decls -I ${CMAKE_CURRENT_SOURCE_DIR}/include DEPENDS ${QUEUE_TD})
mlir_tablegen(QueueOps.cpp.inc -gen-op-defs  -I ${CMAKE_CURRENT_SOURCE_DIR}/include DEPENDS ${QUEUE_TD})
add_public_tablegen_target(TesseraQueueTableGen)

add_mlir_library(TesseraQueueDialect
    lib/Dialect/Queue/QueueVerifiers.cpp

    DEPENDS
        TesseraQueueTableGen

    LINK_LIBS PUBLIC
        MLIRIR
        MLIRSupport
)
target_include_directories(TesseraQueueDialect
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${CMAKE_CURRENT_BINARY_DIR}
)

# ------------------------------------------------------------------
# Conversion passes: Schedule → Target IR, Tile → PTX
# ------------------------------------------------------------------
add_mlir_library(TesseraFA4Conversions
    lib/Conversion/TesseraScheduleToTarget/LowerScheduleToTarget.cpp
    lib/Conversion/TesseraTileToPTX/LowerTileToPTX.cpp

    LINK_LIBS PUBLIC
        TesseraAttnDialect
        TesseraQueueDialect
        MLIRIR
        MLIRPass
        MLIRTransforms
        MLIRConversionPassIncGen
)

# ------------------------------------------------------------------
# Convenience umbrella — depend on this to pull in the whole component
# ------------------------------------------------------------------
add_custom_target(TesseraFA4 ALL
    DEPENDS
        TesseraAttnDialect
        TesseraQueueDialect
        TesseraFA4Conversions
)
