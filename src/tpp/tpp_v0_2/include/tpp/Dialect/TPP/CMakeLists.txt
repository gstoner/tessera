
set(LLVM_TARGET_DEFINITIONS TPP.td)
mlir_tablegen(TPPDialect.h.inc -gen-dialect-decls -dialect=tpp)
mlir_tablegen(TPPDialect.cpp.inc -gen-dialect-defs -dialect=tpp)
add_public_tablegen_target(TPPGen)

set(LLVM_TARGET_DEFINITIONS TPPTypes.td)
mlir_tablegen(TPPTypes.h.inc -gen-typedef-decls -dialect=tpp)
mlir_tablegen(TPPTypes.cpp.inc -gen-typedef-defs -dialect=tpp)
add_public_tablegen_target(TPPTypesGen)

set(LLVM_TARGET_DEFINITIONS TPPAttrs.td)
mlir_tablegen(TPPAttrs.h.inc -gen-attrdef-decls)
mlir_tablegen(TPPAttrs.cpp.inc -gen-attrdef-defs)
add_public_tablegen_target(TPPAttrsGen)

set(LLVM_TARGET_DEFINITIONS TPPOps.td)
mlir_tablegen(TPPOps.h.inc -gen-op-decls)
mlir_tablegen(TPPOps.cpp.inc -gen-op-defs)
add_public_tablegen_target(TPPOpsGen)
