
set(LLVM_LINK_COMPONENTS Support)

# Phase F4 — AdjointInterface (autodiff op trait). Generated headers feed the
# AutodiffPass C++ body in src/transforms/lib/AutodiffPass.cpp and the
# DeclareOpInterfaceMethods<...> mixins on differentiable ops in TesseraOps.td.
# Generated paths land at ${BINARY}/Tessera/AdjointInterface.{h,cpp}.inc so
# that consumers can `#include "Tessera/AdjointInterface.h.inc"` against the
# existing TesseraIR include-dir set.
file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/Tessera)
set(LLVM_TARGET_DEFINITIONS include/Tessera/AdjointInterface.td)
mlir_tablegen(Tessera/AdjointInterface.h.inc -gen-op-interface-decls)
mlir_tablegen(Tessera/AdjointInterface.cpp.inc -gen-op-interface-defs)
add_public_tablegen_target(TesseraAdjointInterfaceTableGen)

set(LLVM_TARGET_DEFINITIONS TesseraOps.td)

# TesseraOps.td now includes "Tessera/AdjointInterface.td" — surface that path
# to the tablegen invocation so it can resolve the include.
include_directories(BEFORE ${CMAKE_CURRENT_SOURCE_DIR}/include)

mlir_tablegen(TesseraOps.h.inc -gen-op-decls)
mlir_tablegen(TesseraOps.cpp.inc -gen-op-defs)
mlir_tablegen(TesseraOpsDialect.h.inc -gen-dialect-decls -dialect=tessera)
mlir_tablegen(TesseraOpsDialect.cpp.inc -gen-dialect-defs -dialect=tessera)
mlir_tablegen(TesseraOpsEnums.h.inc -gen-enum-decls)
mlir_tablegen(TesseraOpsEnums.cpp.inc -gen-enum-defs)
mlir_tablegen(TesseraOpsTypes.h.inc -gen-typedef-decls -typedefs-dialect=tessera)
mlir_tablegen(TesseraOpsTypes.cpp.inc -gen-typedef-defs -typedefs-dialect=tessera)
add_public_tablegen_target(TesseraIRTableGen)
add_mlir_doc(TesseraOps TesseraIR Dialects/ -gen-dialect-doc)

add_mlir_dialect_library(TesseraIR
  TesseraDialect.cpp
  TesseraOps.cpp
  TesseraTiling.cpp
  AdjointInterface.cpp

  DEPENDS
    TesseraIRTableGen
    TesseraAdjointInterfaceTableGen

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
