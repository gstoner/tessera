# Tile Optimization + Flash Attention v4
# Canonical tile optimization + FA4 component.
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
# Queue dialect (v1.3 addition)
# ------------------------------------------------------------------
set(QUEUE_TD ${CMAKE_CURRENT_SOURCE_DIR}/include/tessera/Dialect/Queue/Queue.td)

set(LLVM_TARGET_DEFINITIONS ${QUEUE_TD})
mlir_tablegen(QueueOps.h.inc   -gen-op-decls -I ${CMAKE_CURRENT_SOURCE_DIR}/include DEPENDS ${QUEUE_TD})
mlir_tablegen(QueueOps.cpp.inc -gen-op-defs  -I ${CMAKE_CURRENT_SOURCE_DIR}/include DEPENDS ${QUEUE_TD})
mlir_tablegen(QueueTypes.h.inc   -gen-typedef-decls -typedefs-dialect=tessera.queue -I ${CMAKE_CURRENT_SOURCE_DIR}/include DEPENDS ${QUEUE_TD})
mlir_tablegen(QueueTypes.cpp.inc -gen-typedef-defs  -typedefs-dialect=tessera.queue -I ${CMAKE_CURRENT_SOURCE_DIR}/include DEPENDS ${QUEUE_TD})
mlir_tablegen(QueueDialect.h.inc   -gen-dialect-decls -I ${CMAKE_CURRENT_SOURCE_DIR}/include DEPENDS ${QUEUE_TD})
mlir_tablegen(QueueDialect.cpp.inc -gen-dialect-defs  -I ${CMAKE_CURRENT_SOURCE_DIR}/include DEPENDS ${QUEUE_TD})
add_public_tablegen_target(TesseraQueueTableGen)

add_mlir_library(TesseraQueueDialect
    lib/Dialect/Queue/QueueOps.cpp
    lib/Dialect/Queue/QueueVerifiers.cpp

    DEPENDS
        TesseraQueueTableGen

    LINK_LIBS PUBLIC
        MLIRIR
        MLIRSupport
        # Sprint V8 (2026-05-22): TesseraIR provides the parent
        # TesseraDialect class that anchors the eager-load extension
        # (same pattern V7b used for TesseraAttnDialect).
        TesseraIR
)
# QueueOps.cpp includes `QueueTypes.cpp.inc` (tablegen output) which
# emits `generatedTypeParser` / `generatedTypePrinter` helpers that
# are only referenced when the dialect overrides parseType/printType
# (we don't).  Silence `-Wunused-function` for this source file.
set_source_files_properties(
  lib/Dialect/Queue/QueueOps.cpp
  PROPERTIES COMPILE_OPTIONS "-Wno-unused-function")
target_include_directories(TesseraQueueDialect
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${CMAKE_CURRENT_BINARY_DIR}
        # Sprint V8: visibility of TesseraOps.h for the eager-load
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
        TesseraQueueDialect
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
        TesseraQueueDialect
        TesseraFA4Conversions
)
