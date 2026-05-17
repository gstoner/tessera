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
add_subdirectory(linalg)               # parent linalg solver stack

# GA7 — tessera_clifford dialect, behind a build flag (off by default in v1
# so non-GA builds aren't affected).
if(TESSERA_BUILD_CLIFFORD_BACKEND)
  add_subdirectory(clifford)
endif()

# EBM5 — tessera_ebm dialect, behind a build flag.
if(TESSERA_BUILD_EBM_BACKEND)
  add_subdirectory(ebm)
endif()

add_custom_target(tessera_solvers DEPENDS
  tessera_solvers_dialects
  tessera_solvers_passes
  tessera_linalg_solver_passes
)
