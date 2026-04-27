
set(LLVM_LINK_COMPONENTS Support)

set(LLVM_TARGET_DEFINITIONS TesseraOps.td)

mlir_tablegen(TesseraOps.h.inc -gen-op-decls)
mlir_tablegen(TesseraOps.cpp.inc -gen-op-defs)
mlir_tablegen(TesseraOpsDialect.h.inc -gen-dialect-decls)
mlir_tablegen(TesseraOpsDialect.cpp.inc -gen-dialect-defs)
mlir_tablegen(TesseraOpsEnums.h.inc -gen-enum-decls)
mlir_tablegen(TesseraOpsEnums.cpp.inc -gen-enum-defs)
add_public_tablegen_target(TesseraIRTableGen)
add_mlir_doc(TesseraOps TesseraIR Dialects/ -gen-dialect-doc)

add_mlir_dialect_library(TesseraIR
  TesseraDialect.cpp
  TesseraOps.cpp
  TesseraTiling.cpp

  DEPENDS
    TesseraIRTableGen

  LINK_LIBS PUBLIC
    MLIRIR
    MLIRTensorDialect
    MLIRArithDialect
    MLIRInferTypeOpInterface
    MLIRSideEffectInterfaces
    MLIRTilingInterface
)

target_include_directories(TesseraIR
  PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_BINARY_DIR}
)

if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/../pdll/CMakeLists.txt)
  add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../pdll ${CMAKE_CURRENT_BINARY_DIR}/pdll)
endif()
