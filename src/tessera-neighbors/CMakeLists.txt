cmake_minimum_required(VERSION 3.20)
project(TesseraNeighbors LANGUAGES CXX)

# Expect LLVM/MLIR to already be configured in the parent project
# add_subdirectory from your main build.

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_library(TesseraNeighbors
  lib/Dialect/Neighbors/IR/TesseraNeighbors.cpp
  lib/Dialect/Neighbors/Transforms/HaloInferPass.cpp
  lib/Dialect/Neighbors/Transforms/StencilLowerPass.cpp
  lib/Dialect/Neighbors/Transforms/PipelineOverlapPass.cpp
  lib/Dialect/Neighbors/Transforms/DynamicTopologyPass.cpp
)

target_include_directories(TesseraNeighbors PUBLIC include)

# If you use tablegen / ODS generation in your build, wire tessera_neighbors.td there.
# For tests, you likely use llvm-lit; see test/Neighbors/*.mlir
