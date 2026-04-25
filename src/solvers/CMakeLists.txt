# Drop-in CMake for Tessera solvers package
cmake_minimum_required(VERSION 3.20)
project(TesseraSolvers LANGUAGES C CXX)

# Expect caller project to have MLIR and Tessera setup already.
# Provides targets:
#   tessera_solvers_dialects
#   tessera_solvers_passes
#   tessera_solvers (umbrella)
#
# Add this directory from your main CMakeLists:
# add_subdirectory(src/solvers)

add_subdirectory(dialects)
add_subdirectory(passes)

add_custom_target(tessera_solvers DEPENDS tessera_solvers_dialects tessera_solvers_passes)
