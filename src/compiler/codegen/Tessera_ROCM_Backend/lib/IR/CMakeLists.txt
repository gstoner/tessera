set(LLVM_TARGET_DEFINITIONS ${CMAKE_CURRENT_SOURCE_DIR}/../../include/TesseraROCM/IR/TesseraROCMOps.td)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../../include/TesseraROCM/IR)
mlir_tablegen(TesseraROCMOps.h.inc -gen-op-decls)
mlir_tablegen(TesseraROCMOps.cpp.inc -gen-op-defs)
mlir_tablegen(TesseraROCMDialect.h.inc -gen-dialect-decls -dialect=tessera_rocm)
mlir_tablegen(TesseraROCMDialect.cpp.inc -gen-dialect-defs -dialect=tessera_rocm)
mlir_tablegen(TesseraROCMTypes.h.inc -gen-typedef-decls -typedefs-dialect=tessera_rocm)
mlir_tablegen(TesseraROCMTypes.cpp.inc -gen-typedef-defs -typedefs-dialect=tessera_rocm)
add_public_tablegen_target(TesseraROCMTableGen)

add_mlir_library(TesseraROCMIR
  TesseraROCMOps.cpp
  MemSpace.cpp
  ABI.cpp
  MFMTables.cpp
  LINK_LIBS PUBLIC
    MLIRIR MLIRFuncDialect MLIRLLVMDialect
  DEPENDS TesseraROCMTableGen)
target_include_directories(TesseraROCMIR PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/../../include
  ${CMAKE_CURRENT_BINARY_DIR})
