set(LLVM_LINK_COMPONENTS Support)

set(LLVM_TARGET_DEFINITIONS include/tessera/Dialect/Solver/Solver.td)
mlir_tablegen(TesseraLinalgSolverOps.h.inc -gen-op-decls)
mlir_tablegen(TesseraLinalgSolverOps.cpp.inc -gen-op-defs)
mlir_tablegen(TesseraLinalgSolverDialect.h.inc -gen-dialect-decls)
mlir_tablegen(TesseraLinalgSolverDialect.cpp.inc -gen-dialect-defs)
mlir_tablegen(TesseraLinalgSolverAttrs.h.inc -gen-attrdef-decls -attrdefs-dialect=tessera.solver)
mlir_tablegen(TesseraLinalgSolverAttrs.cpp.inc -gen-attrdef-defs -attrdefs-dialect=tessera.solver)
add_public_tablegen_target(TesseraLinalgSolverTableGen)

add_library(tessera_linalg_solver_passes STATIC
  lib/Dialect/SolverDialect.cpp
  lib/Passes/MixedPrecision.cpp
  lib/Passes/IterativeRefinement.cpp
)
target_include_directories(tessera_linalg_solver_passes
  PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_BINARY_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/../core/passes
)
target_link_libraries(tessera_linalg_solver_passes
  PUBLIC
    tessera_solvers_passes
    MLIRIR
    MLIRPass
    MLIRFuncDialect
    MLIRLinalgDialect
)
add_dependencies(tessera_linalg_solver_passes TesseraLinalgSolverTableGen)

mlir_check_all_link_libraries(tessera_linalg_solver_passes)
