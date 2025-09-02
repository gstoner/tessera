
cmake_minimum_required(VERSION 3.20)
project(TesseraAutotunePass LANGUAGES CXX)

find_package(MLIR REQUIRED CONFIG)
message(STATUS "Found MLIR: ${MLIR_DIR}")

set(CMAKE_CXX_STANDARD 17 CACHE STRING "C++ standard required by MLIR" FORCE)
set(CMAKE_BUILD_TYPE Release)

list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
include(TableGen)
include(AddMLIR)

include_directories(${MLIR_INCLUDE_DIRS})
add_definitions(${MLIR_DEFINITIONS})

add_library(TesseraResolveAttrsPass STATIC
  tessera/tools/autotune/lib/ResolveTesseraAttrsPass.cpp
)

target_include_directories(TesseraResolveAttrsPass PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/tessera/tools/autotune/include
)

target_link_libraries(TesseraResolveAttrsPass
  PRIVATE
  MLIRIR
  MLIRPass
)

# Optional: create a 'mlir-opt' with our pass linked in (if desired).
add_executable(tessera-mlir-opt
  ${MLIR_TOOLS_DIR}/mlir-opt.cpp
)
target_link_libraries(tessera-mlir-opt PRIVATE
  TesseraResolveAttrsPass
  MLIRIR
  MLIRPass
  MLIRSupport
)
