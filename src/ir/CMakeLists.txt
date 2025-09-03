
set(LLVM_LINK_COMPONENTS Support)

set(LLVM_TABLEGEN_EXE ${LLVM_TABLEGEN_EXE} CACHE FILEPATH "llvm-tblgen")
set(MLIR_TABLEGEN_EXE ${MLIR_TABLEGEN_EXE} CACHE FILEPATH "mlir-tblgen")
set(MLIR_TABLEGEN_ENABLED 1)

add_mlir_dialect(TesseraIR tessera)
add_mlir_doc(TesseraIR TesseraIR Dialects/ -gen-dialect-doc)

add_mlir_dialect_library(TesseraIR
  TesseraDialect.cpp
  TesseraOps.cpp
  TesseraTiling.cpp

  DEPENDS
    MLIRTblGen
    MLIRIR

  LINK_LIBS PUBLIC
    MLIRIR
    MLIRTensorDialect
    MLIRArithDialect
    MLIRIR
)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../pdll ${CMAKE_CURRENT_BINARY_DIR}/pdll)
