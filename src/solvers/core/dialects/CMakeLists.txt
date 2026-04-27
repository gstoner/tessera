set(LLVM_LINK_COMPONENTS Support)

set(LLVM_TARGET_DEFINITIONS tessera_sparse.td)
mlir_tablegen(TesseraSparseOps.h.inc -gen-op-decls)
mlir_tablegen(TesseraSparseOps.cpp.inc -gen-op-defs)
mlir_tablegen(TesseraSparseDialect.h.inc -gen-dialect-decls)
mlir_tablegen(TesseraSparseDialect.cpp.inc -gen-dialect-defs)

set(LLVM_TARGET_DEFINITIONS tessera_solver.td)
mlir_tablegen(TesseraSolverOps.h.inc -gen-op-decls)
mlir_tablegen(TesseraSolverOps.cpp.inc -gen-op-defs)
mlir_tablegen(TesseraSolverDialect.h.inc -gen-dialect-decls)
mlir_tablegen(TesseraSolverDialect.cpp.inc -gen-dialect-defs)

set(LLVM_TARGET_DEFINITIONS tessera_rng.td)
mlir_tablegen(TesseraRNGOps.h.inc -gen-op-decls)
mlir_tablegen(TesseraRNGOps.cpp.inc -gen-op-defs)
mlir_tablegen(TesseraRNGDialect.h.inc -gen-dialect-decls)
mlir_tablegen(TesseraRNGDialect.cpp.inc -gen-dialect-defs)

add_public_tablegen_target(TesseraSolversTableGen)

add_library(tessera_solvers_dialects INTERFACE)
add_dependencies(tessera_solvers_dialects TesseraSolversTableGen)
