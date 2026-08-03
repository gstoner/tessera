set(LLVM_TARGET_DEFINITIONS ${CMAKE_CURRENT_SOURCE_DIR}/../../include/TesseraX86/IR/TesseraX86Ops.td)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../../include/TesseraX86/IR)
mlir_tablegen(TesseraX86Ops.h.inc -gen-op-decls)
mlir_tablegen(TesseraX86Ops.cpp.inc -gen-op-defs)
mlir_tablegen(TesseraX86Dialect.h.inc -gen-dialect-decls -dialect=tessera_x86)
mlir_tablegen(TesseraX86Dialect.cpp.inc -gen-dialect-defs -dialect=tessera_x86)
mlir_tablegen(TesseraX86Types.h.inc -gen-typedef-decls -typedefs-dialect=tessera_x86)
mlir_tablegen(TesseraX86Types.cpp.inc -gen-typedef-defs -typedefs-dialect=tessera_x86)
add_public_tablegen_target(TesseraX86TableGen)

add_mlir_library(TesseraX86IR
  TesseraX86Ops.cpp
  LINK_LIBS PUBLIC
    MLIRIR MLIRFuncDialect MLIRMemRefDialect
  DEPENDS TesseraX86TableGen)
target_include_directories(TesseraX86IR PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/../../include
  ${CMAKE_CURRENT_BINARY_DIR})
