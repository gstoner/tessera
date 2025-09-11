
cmake_minimum_required(VERSION 3.20)
project(Tessera_RubinCPX_Compiler_v1_1 LANGUAGES C CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(MLIR REQUIRED CONFIG)
message(STATUS "Found MLIR: ${MLIR_DIR}")

list(APPEND CMAKE_MODULE_PATH ${MLIR_CMAKE_DIR})
include(TableGen)
include(AddMLIR)
include(HandleLLVMOptions)

include_directories(${MLIR_INCLUDE_DIRS})
link_directories(${MLIR_LIB_DIR})
add_definitions(${LLVM_DEFINITIONS})

set(CPX_TD include/tessera/Target/NVRubinCPX/NVRubinCPX.td)

mlir_tablegen(NVRubinCPXOps.h.inc -gen-op-decls -dialect=tessera.target.cpx)
mlir_tablegen(NVRubinCPXOps.cpp.inc -gen-op-defs -dialect=tessera.target.cpx)
mlir_tablegen(NVRubinCPXTypes.h.inc -gen-typedef-decls -dialect=tessera.target.cpx)
mlir_tablegen(NVRubinCPXTypes.cpp.inc -gen-typedef-defs -dialect=tessera.target.cpx)
mlir_tablegen(NVRubinCPXAttrs.h.inc -gen-attrdef-decls -dialect=tessera.target.cpx)
mlir_tablegen(NVRubinCPXAttrs.cpp.inc -gen-attrdef-defs -dialect=tessera.target.cpx)
add_public_tablegen_target(TesseraCPXIncGen)

add_subdirectory(include)
add_subdirectory(lib)
add_subdirectory(tools)
enable_testing()
add_subdirectory(test)
