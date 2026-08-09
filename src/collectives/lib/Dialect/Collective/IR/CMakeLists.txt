include_directories(${CMAKE_CURRENT_BINARY_DIR})

set(LLVM_TARGET_DEFINITIONS
  ${CMAKE_CURRENT_SOURCE_DIR}/../../../../include/tessera/Dialect/Collective/IR/CollectiveTypes.td)
mlir_tablegen(CollectiveTypes.h.inc -gen-typedef-decls -dialect=tessera_collective)
mlir_tablegen(CollectiveTypes.cpp.inc -gen-typedef-defs -dialect=tessera_collective)
add_public_tablegen_target(TesseraCollectiveTypesIncGen)

set(LLVM_TARGET_DEFINITIONS
  ${CMAKE_CURRENT_SOURCE_DIR}/../../../../include/tessera/Dialect/Collective/IR/CollectiveOps.td)
mlir_tablegen(CollectiveOps.h.inc -gen-op-decls)
mlir_tablegen(CollectiveOps.cpp.inc -gen-op-defs)
mlir_tablegen(CollectiveDialect.h.inc -gen-dialect-decls -dialect=tessera_collective)
mlir_tablegen(CollectiveDialect.cpp.inc -gen-dialect-defs -dialect=tessera_collective)
add_public_tablegen_target(TesseraCollectiveOpsIncGen)

add_mlir_dialect_library(TesseraCollectiveDialect
  CollectiveOps.cpp
  CollectiveLowering.cpp
  DEPENDS TesseraCollectiveTypesIncGen TesseraCollectiveOpsIncGen
  LINK_LIBS PUBLIC MLIRIR MLIRPass MLIRSideEffectInterfaces
)

target_include_directories(TesseraCollectiveDialect PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/../../../../include
  ${CMAKE_CURRENT_BINARY_DIR}
)
