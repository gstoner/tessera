cmake_minimum_required(VERSION 3.20)
project(TesseraCollectivesPlanPass_v1_4 LANGUAGES CXX)

find_package(LLVM REQUIRED CONFIG)
find_package(MLIR REQUIRED CONFIG)

option(TESSERA_ENABLE_NCCL "Enable NCCL adapter" OFF)
option(TESSERA_ENABLE_RCCL "Enable RCCL adapter" OFF)

list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
include(TableGen) ; include(AddLLVM) ; include(AddMLIR) ; include(HandleLLVMOptions)
set(CMAKE_CXX_STANDARD 17) ; set(CMAKE_CXX_STANDARD_REQUIRED YES) ; set(CMAKE_CXX_EXTENSIONS NO)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include ${CMAKE_CURRENT_BINARY_DIR}/include)

add_subdirectory(lib/Dialect/Collective/IR)
add_subdirectory(lib/Dialect/Collective/Runtime)
add_subdirectory(lib/Dialect/Collective/Transforms)
add_subdirectory(tools)
add_subdirectory(test)
