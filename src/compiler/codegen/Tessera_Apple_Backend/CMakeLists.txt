cmake_minimum_required(VERSION 3.20)

option(TESSERA_APPLE_WITH_MLIR "Build Apple Silicon MLIR target dialect" ON)

if(TESSERA_APPLE_WITH_MLIR)
  set(LLVM_TARGET_DEFINITIONS include/Tessera/Target/Apple/TesseraAppleOps.td)
  mlir_tablegen(TesseraAppleOps.h.inc -gen-op-decls)
  mlir_tablegen(TesseraAppleOps.cpp.inc -gen-op-defs)
  mlir_tablegen(TesseraAppleDialect.h.inc -gen-dialect-decls -dialect=tessera_apple)
  mlir_tablegen(TesseraAppleDialect.cpp.inc -gen-dialect-defs -dialect=tessera_apple)
  add_public_tablegen_target(TesseraAppleOpsIncGen)
endif()
