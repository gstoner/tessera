cmake_minimum_required(VERSION 3.20)
project(TesseraLinalgSolvers VERSION 0.1 LANGUAGES CXX)
# This is a scaffold; wire MLIR/LLVM find_package as in your main tree.
add_library(TesseraSolverPasses STATIC
  lib/Passes/MixedPrecision.cpp
  lib/Passes/IterativeRefinement.cpp
)
target_include_directories(TesseraSolverPasses PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
