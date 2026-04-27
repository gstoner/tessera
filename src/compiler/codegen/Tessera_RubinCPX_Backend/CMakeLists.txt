
cmake_minimum_required(VERSION 3.20)
project(TesseraRubinCPXBackend LANGUAGES C CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Options
option(TESSERA_BUILD_CPX_OPT "Build tessera-cpx-opt driver" ON)
option(TESSERA_RUBINCPX_ENABLE_TESTS "Enable Rubin CPX lit tests" ON)

# Find LLVM/MLIR (assumes LLVM is already installed and provides MLIRConfig.cmake)
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
find_package(LLVM REQUIRED CONFIG)
find_package(MLIR REQUIRED CONFIG)
message(STATUS "Found MLIR: ${MLIR_DIR}")

list(APPEND CMAKE_MODULE_PATH ${MLIR_CMAKE_DIR})
include(TableGen)
include(AddMLIR)
include(AddLLVM)
include(HandleLLVMOptions)

include_directories(${LLVM_INCLUDE_DIRS})
include_directories(${MLIR_INCLUDE_DIRS})
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
link_directories(${MLIR_LIB_DIR})
add_definitions(${LLVM_DEFINITIONS})

add_subdirectory(include)
add_subdirectory(lib)
if(TESSERA_BUILD_CPX_OPT)
  add_subdirectory(tools/tessera-cpx-opt)
endif()

if(TESSERA_RUBINCPX_ENABLE_TESTS)
  enable_testing()
  add_subdirectory(test)
endif()
