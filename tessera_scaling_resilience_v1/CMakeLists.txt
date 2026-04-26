cmake_minimum_required(VERSION 3.20)
project(Tessera_Scaling_Resilience LANGUAGES CXX)

option(TESSERA_WITH_SR "Build Tessera Scaling & Resilience extensions" ON)

# Placeholder MLIR/LLVM discovery (adjust to your tree)
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
find_package(MLIR REQUIRED CONFIG) # or use custom FindMLIR.cmake fallback
message(STATUS "Using MLIR at: ${MLIR_DIR}")

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include_directories(
  ${MLIR_INCLUDE_DIRS}
  ${LLVM_INCLUDE_DIRS}
  ${CMAKE_CURRENT_SOURCE_DIR}/include
)

add_definitions(${LLVM_DEFINITIONS})

add_library(TesseraSRPasses
  lib/passes/InsertRecomputePass.cpp
  lib/passes/OptimizerShardPass.cpp
  lib/passes/ResilienceRestartPass.cpp
  lib/passes/ExportDeploymentManifestPass.cpp
  lib/passes/RegisterPasses.cpp
)

target_link_libraries(TesseraSRPasses
  MLIRIR
  MLIRPass
)

add_executable(tessera-opt-sr
  tools/tessera-opt-sr/main.cpp
)
target_link_libraries(tessera-opt-sr
  TesseraSRPasses
  MLIRIR
  MLIRPass
)

enable_testing()
add_test(NAME sr-checkpoint COMMAND tessera-opt-sr
  -tessera-insert-recompute -verify-diagnostics ${CMAKE_CURRENT_SOURCE_DIR}/tests/sr-checkpoint.mlir)
add_test(NAME sr-optimizer-shard COMMAND tessera-opt-sr
  -tessera-optimizer-shard -verify-diagnostics ${CMAKE_CURRENT_SOURCE_DIR}/tests/sr-optimizer-shard.mlir)
add_test(NAME sr-resilience COMMAND tessera-opt-sr
  -tessera-resilience-restart -verify-diagnostics ${CMAKE_CURRENT_SOURCE_DIR}/tests/sr-resilience.mlir)
add_test(NAME sr-export-manifest COMMAND tessera-opt-sr
  -tessera-export-deployment-manifest ${CMAKE_CURRENT_SOURCE_DIR}/tests/sr-export-manifest.mlir)

message(STATUS "Tessera Scaling & Resilience configured.")