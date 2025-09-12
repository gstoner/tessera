set(LLVM_LINK_COMPONENTS Support)

# ODS tablegen for dialects
set(TD_FILES
  tessera_sparse.td
  tessera_solver.td
  tessera_rng.td
)

mlir_tablegen(TesseraSolversDialectInterfaces.h.inc -gen-op-interface-decls)
mlir_tablegen(TesseraSolversDialectInterfaces.cpp.inc -gen-op-interface-defs)

foreach(td ${TD_FILES})
  mlir_tablegen(${td}.inc -gen-op-decls -I ${CMAKE_CURRENT_SOURCE_DIR})
  mlir_tablegen(${td}.cpp.inc -gen-op-defs  -I ${CMAKE_CURRENT_SOURCE_DIR})
endforeach()

add_public_tablegen_target(TesseraSolversTableGen)

add_library(tessera_solvers_dialects INTERFACE)
add_dependencies(tessera_solvers_dialects TesseraSolversTableGen)
