add_library(tesseraCerebrasPasses STATIC)
target_include_directories(tesseraCerebrasPasses PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../include)

if(HAVE_MLIR)
  # Cerebras dialect
  set(LLVM_TARGET_DEFINITIONS ${CMAKE_CURRENT_SOURCE_DIR}/../include/tessera/targets/cerebras/Cerebras.td)
  mlir_tablegen(CerebrasDialect.h.inc -gen-dialect-decls -dialect=cerebras)
  mlir_tablegen(CerebrasDialect.cpp.inc -gen-dialect-defs -dialect=cerebras)
  mlir_tablegen(CerebrasOps.h.inc -gen-op-decls -dialect=cerebras)
  mlir_tablegen(CerebrasOps.cpp.inc -gen-op-defs -dialect=cerebras)
  add_public_tablegen_target(CerebrasDialectIncGen)

  # Tessera Target dialect
  set(LLVM_TARGET_DEFINITIONS ${CMAKE_CURRENT_SOURCE_DIR}/../include/tessera/targets/ttarget/TTarget.td)
  mlir_tablegen(TTargetDialect.h.inc -gen-dialect-decls -dialect=ttarget)
  mlir_tablegen(TTargetDialect.cpp.inc -gen-dialect-defs -dialect=ttarget)
  mlir_tablegen(TTargetOps.h.inc -gen-op-decls -dialect=ttarget)
  mlir_tablegen(TTargetOps.cpp.inc -gen-op-defs -dialect=ttarget)
  add_public_tablegen_target(TTargetDialectIncGen)

  add_library(cerebrasDialect STATIC CerebrasDialect.cpp LowerTTargetToCerebras.cpp CSLEmitPass.cpp CerebrasCanonicalize.cpp)
  add_dependencies(cerebrasDialect CerebrasDialectIncGen TTargetDialectIncGen)
  target_link_libraries(cerebrasDialect PUBLIC MLIRIR)
  target_include_directories(cerebrasDialect PUBLIC ${CMAKE_CURRENT_BINARY_DIR})
  target_compile_definitions(cerebrasDialect PUBLIC HAVE_MLIR=1)

  target_link_libraries(tesseraCerebrasPasses PUBLIC cerebrasDialect MLIRIR)
  target_compile_definitions(tesseraCerebrasPasses PUBLIC HAVE_MLIR=1)
else()
  target_compile_definitions(tesseraCerebrasPasses PUBLIC HAVE_MLIR=0)
#endif
