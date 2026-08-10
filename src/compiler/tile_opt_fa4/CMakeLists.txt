# Tile Optimization + Flash Attention v4
# Canonical tile optimization + FA4 component.
# Adds backward-attention support over v1/v1_1/v1_2.
#
# Exported targets:
#   TesseraAttnDialect      — Attn dialect verifiers + ODS-generated headers
#   TesseraFA4Conversions   — Schedule-to-Target and Tile-to-PTX lowering passes
#
# The v1.3 Queue dialect (`tessera.queue`) was deleted 2026-08-10 under
# Decisions #29/#31: zero producers/consumers, unparseable dotted-name type
# syntax, and the warp-specialization boundary's production mechanism is
# `!tile.pipeline_state` + `!tile.async_token`. The queue *vocabulary* lives
# on in the Python tile IR spine (`python/tessera/compiler/tile_ir.py`).

# ------------------------------------------------------------------
# Attn dialect
# ------------------------------------------------------------------
set(ATTN_TD ${CMAKE_CURRENT_SOURCE_DIR}/include/tessera/Dialect/Attn/Attn.td)

set(LLVM_TARGET_DEFINITIONS ${ATTN_TD})
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
        # Sprint V7b (2026-05-22): TesseraIR provides the parent
        # TesseraDialect class that anchors the eager-load extension.
        TesseraIR
)
target_include_directories(TesseraAttnDialect
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${CMAKE_CURRENT_BINARY_DIR}
        # Sprint V7b: visibility of TesseraOps.h for the eager-load
        # extension's parent-dialect anchor.
        ${CMAKE_SOURCE_DIR}/src/compiler/ir/include
        ${CMAKE_BINARY_DIR}/src/compiler/ir
)

# ------------------------------------------------------------------
# Conversion passes: Schedule → Target IR, Tile → PTX
# ------------------------------------------------------------------
add_mlir_library(TesseraFA4Conversions
    lib/Conversion/TesseraScheduleToTarget/LowerScheduleToTarget.cpp
    lib/Conversion/TesseraTileToPTX/LowerTileToPTX.cpp

    LINK_LIBS PUBLIC
        TesseraAttnDialect
        MLIRIR
        MLIRPass
        MLIRTransforms
)
# MLIRConversionPassIncGen is an in-tree-only tablegen aggregate; not exported
# by installed MLIR distributions (Homebrew llvm@23 etc.). Re-attach only when
# present so the build keeps working both ways.
if(TARGET MLIRConversionPassIncGen)
    target_link_libraries(TesseraFA4Conversions PUBLIC MLIRConversionPassIncGen)
endif()

# ------------------------------------------------------------------
# Convenience umbrella — depend on this to pull in the whole component
# ------------------------------------------------------------------
add_custom_target(TesseraFA4 ALL
    DEPENDS
        TesseraAttnDialect
        TesseraFA4Conversions
)
