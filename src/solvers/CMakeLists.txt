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

add_subdirectory(core/dialects)    # tessera_solvers_dialects target
add_subdirectory(core/passes)      # tessera_solvers_passes target

# Domain-specific solver components (each brings its own CMakeLists).
add_subdirectory(scaling_resilience)
add_subdirectory(tpp)
add_subdirectory(spectral)
# linalg/ remains opt-in until its scaffold APIs align with the canonical build.

add_custom_target(tessera_solvers DEPENDS tessera_solvers_dialects tessera_solvers_passes)
