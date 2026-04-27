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

mlir_check_all_link_libraries(tessera_solvers_passes)
