cmake_minimum_required(VERSION 3.20)
project(TesseraROCMBackend LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

option(TESSERA_ROCM_BUILD_TOOLS "Build tessera-rocm tools" ON)
option(TESSERA_ROCM_BUILD_RUNTIME "Build HIP runtime helper" ON)
option(TESSERA_ROCM_ENABLE_CK "Enable Composable Kernels integration" OFF)

find_package(LLVM REQUIRED CONFIG)
find_package(MLIR REQUIRED CONFIG)

list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
include(TableGen)
include(AddMLIR)
include(AddLLVM)

if (MSVC)
  add_compile_options(/W4)
else()
  add_compile_options(-Wall -Wextra -Wno-unused-parameter)
endif()

include_directories(${LLVM_INCLUDE_DIRS})
include_directories(${MLIR_INCLUDE_DIRS})
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
add_definitions(${LLVM_DEFINITIONS})

add_subdirectory(lib)
if (TESSERA_ROCM_BUILD_TOOLS)
  add_subdirectory(tools)
endif()
if (TESSERA_ROCM_BUILD_RUNTIME)
  add_subdirectory(runtime/hip)
endif()

enable_testing()
add_subdirectory(test)

if (TESSERA_ROCM_ENABLE_CK)
  add_subdirectory(external/ck_bridge)
endif()

add_subdirectory(ci)
