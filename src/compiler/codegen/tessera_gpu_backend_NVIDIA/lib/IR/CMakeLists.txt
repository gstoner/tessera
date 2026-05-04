set(LLVM_TARGET_DEFINITIONS
    ${CMAKE_CURRENT_SOURCE_DIR}/../../include/tessera/gpu/IR/TesseraNVIDIADialect.td)
mlir_tablegen(TesseraNVIDIADialect.h.inc -gen-dialect-decls -dialect=tessera_nvidia)
mlir_tablegen(TesseraNVIDIADialect.cpp.inc -gen-dialect-defs -dialect=tessera_nvidia)
add_public_tablegen_target(TesseraNVIDIATableGen)

add_mlir_library(TesseraNVIDIAIR
  TesseraNVIDIADialect.cpp

  DEPENDS
    TesseraNVIDIATableGen

  LINK_LIBS PUBLIC
    MLIRIR
    MLIRSupport
)

target_include_directories(TesseraNVIDIAIR
  PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/../../include
    ${CMAKE_CURRENT_BINARY_DIR}
)
