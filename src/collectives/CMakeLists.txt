# Tessera Collectives — collective-communication dialect and runtime adapters
#
# Can be built standalone or as part of the Tessera monorepo.

# ------------------------------------------------------------------
# Standalone-only preamble
# ------------------------------------------------------------------
if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    cmake_minimum_required(VERSION 3.20)
    project(TesseraCollectives LANGUAGES CXX)

    find_package(LLVM REQUIRED CONFIG)
    find_package(MLIR REQUIRED CONFIG)
    list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}" "${LLVM_CMAKE_DIR}")
    include(TableGen)
    include(AddLLVM)
    include(AddMLIR)
    include(HandleLLVMOptions)
    include_directories(${LLVM_INCLUDE_DIRS} ${MLIR_INCLUDE_DIRS})
    add_definitions(${LLVM_DEFINITIONS})
endif()

option(TESSERA_ENABLE_NCCL "Enable NCCL adapter" OFF)
option(TESSERA_ENABLE_RCCL "Enable RCCL adapter" OFF)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED YES)
set(CMAKE_CXX_EXTENSIONS NO)

include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_BINARY_DIR}/include
)

add_subdirectory(lib/Dialect/Collective/IR)
add_subdirectory(lib/Dialect/Collective/Runtime)

# Transforms subdir added when the implementation is present
if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/lib/Dialect/Collective/Transforms/CMakeLists.txt)
    add_subdirectory(lib/Dialect/Collective/Transforms)
endif()

add_subdirectory(tools)
add_subdirectory(test)
