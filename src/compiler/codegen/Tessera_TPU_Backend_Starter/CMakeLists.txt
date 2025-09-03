cmake_minimum_required(VERSION 3.20)
project(TesseraTPUBackend LANGUAGES CXX C)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

option(TESSERA_ENABLE_TPU "Build TPU backend starter" ON)

# Find LLVM/MLIR (user supplies MLIR_DIR)
find_package(MLIR REQUIRED CONFIG)
message(STATUS "Found MLIR: ${MLIR_DIR}")
# StableHLO headers (user supplies StableHLO_DIR include path)
if(DEFINED STABLEHLO_DIR)
  include_directories(${STABLEHLO_DIR}/include)
endif()

include_directories(${MLIR_INCLUDE_DIRS})
add_definitions(${MLIR_DEFINITIONS})

add_library(tessera_tpu_passes STATIC
  src/passes/TesseraToStableHLO.cpp
  src/passes/TesseraShardingAnnotator.cpp
  src/passes/RegisterPasses.cpp
)
target_include_directories(tessera_tpu_passes PUBLIC include)
target_link_libraries(tessera_tpu_passes
  PUBLIC
    MLIRIR
    MLIRPass
    MLIRTransformUtils
)

add_executable(tessera-tpu-opt tools/tessera-tpu-opt.cpp)
target_link_libraries(tessera-tpu-opt PRIVATE tessera_tpu_passes MLIRIR MLIRPass)

add_executable(pjrt_runner runtime/pjrt_runner.cc)
# PJRT headers are not included in this starter; you can point to them later.
