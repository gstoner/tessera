
cmake_minimum_required(VERSION 3.20)
project(Tessera_RubinCPX_Compiler LANGUAGES C CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Options
option(TESSERA_BUILD_CPX_OPT "Build tessera-cpx-opt driver" ON)

# Find LLVM/MLIR (assumes LLVM is already installed and provides MLIRConfig.cmake)
find_package(MLIR REQUIRED CONFIG)
message(STATUS "Found MLIR: ${MLIR_DIR}")

list(APPEND CMAKE_MODULE_PATH ${MLIR_CMAKE_DIR})
include(TableGen)
include(AddMLIR)
include(HandleLLVMOptions)

include_directories(${MLIR_INCLUDE_DIRS})
link_directories(${MLIR_LIB_DIR})
add_definitions(${LLVM_DEFINITIONS})

add_subdirectory(include)
add_subdirectory(lib)
if(TESSERA_BUILD_CPX_OPT)
  add_subdirectory(tools/tessera-cpx-opt)
endif()

enable_testing()
add_subdirectory(test)
