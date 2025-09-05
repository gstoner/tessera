set(LLVM_TARGET_DEFINITIONS CollectiveTypes.td)
mlir_tablegen(CollectiveTypes.h.inc -gen-typedef-decls -dialect=tessera.collective)
mlir_tablegen(CollectiveTypes.cpp.inc -gen-typedef-defs -dialect=tessera.collective)
add_public_tablegen_target(TesseraCollectiveTypesIncGen)

set(LLVM_TARGET_DEFINITIONS CollectiveOps.td)
mlir_tablegen(CollectiveOps.h.inc -gen-op-decls)
mlir_tablegen(CollectiveOps.cpp.inc -gen-op-defs)
add_public_tablegen_target(TesseraCollectiveOpsIncGen)

add_mlir_dialect_library(TesseraCollectiveDialect
  CollectiveOps.cpp
  CollectiveTypes.cpp
  DEPENDS TesseraCollectiveTypesIncGen TesseraCollectiveOpsIncGen
  LINK_LIBS PUBLIC MLIRIR
)
