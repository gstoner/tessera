cmake_minimum_required(VERSION 3.20)
project(Tessera_SR LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Expect MLIR/LLVM already set by parent project
# Provide interface targets
add_library(DialectSR
  lib/sr/dialect/SRDialect.cpp
  lib/sr/dialect/SROps.cpp
)
target_include_directories(DialectSR PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(DialectSR PUBLIC MLIRIR MLIRSupport)

add_library(TesseraSR
  lib/sr/passes/InsertRecomputePass.cpp
  lib/sr/passes/OptimizerShardPass.cpp
  lib/sr/passes/ResilienceRestartPass.cpp
  lib/sr/passes/ExportDeploymentManifestPass.cpp
  lib/sr/passes/RegisterPasses.cpp
)
target_include_directories(TesseraSR PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(TesseraSR PUBLIC MLIRIR MLIRPass DialectSR)

# Enable tests in parent tree with llvm-lit