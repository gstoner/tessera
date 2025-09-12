set(LLVM_LINK_COMPONENTS Support)
set(SOLVERS_PASSES_SOURCES
  SolversPasses.cpp
  SparseInspector.cpp
  SparsePrecond.cpp
  SparseSolverSpecialize.cpp
  RNGLegalize.cpp
  RNGStreamAssign.cpp
  RNGQMCPlan.cpp
  TrigInit.cpp
  NewtonAutodiff.cpp
  PeriodicHalo.cpp
  ParamBatchPlan.cpp
  ContinuationGuard.cpp
  ImplicitLower.cpp
)

add_library(tessera_solvers_passes STATIC ${SOLVERS_PASSES_SOURCES})
target_include_directories(tessera_solvers_passes PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(tessera_solvers_passes
  PUBLIC
    MLIRIR
    MLIRPass
    MLIRFuncDialect
    MLIRVectorDialect
    MLIRTransformUtils
)

# Optionally build a plugin for tessera-opt (if project uses plugin-style drivers)
add_library(tessera_solvers_opt_plugin MODULE SolversOptPlugin.cpp)
target_link_libraries(tessera_solvers_opt_plugin PRIVATE tessera_solvers_passes)
set_target_properties(tessera_solvers_opt_plugin PROPERTIES
  PREFIX ""
  OUTPUT_NAME "tessera-solver-passes"
)
