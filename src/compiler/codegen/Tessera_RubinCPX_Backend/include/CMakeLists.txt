
# ─── TableGen for Tessera NVRubin CPX dialect ────────────────────────────────
# Generates dialect, type, attribute, and op declarations + definitions from
# the single NVRubinCPX.td source-of-truth.
#
# Generated .h.inc / .cpp.inc files land in the CMake build directory and are
# consumed by the lib/ sources via #include "NVRubinCPX{Dialect,Types,Ops,...}.h.inc"

set(LLVM_TARGET_DEFINITIONS tessera/Target/NVRubinCPX/NVRubinCPX.td)

# Dialect class declaration / definition
mlir_tablegen(NVRubinCPXDialect.h.inc   -gen-dialect-decls   -dialect=tessera.target.cpx)
mlir_tablegen(NVRubinCPXDialect.cpp.inc -gen-dialect-defs    -dialect=tessera.target.cpx)

# Type declarations / definitions  (NVFP4Type, NVFP6Type)
mlir_tablegen(NVRubinCPXTypes.h.inc     -gen-typedef-decls   -typedefs-dialect=tessera.target.cpx)
mlir_tablegen(NVRubinCPXTypes.cpp.inc   -gen-typedef-defs    -typedefs-dialect=tessera.target.cpx)

# Attribute declarations / definitions (CapabilitiesAttr, KVPolicyAttr, …)
mlir_tablegen(NVRubinCPXAttrs.h.inc     -gen-attrdef-decls   -attrdefs-dialect=tessera.target.cpx)
mlir_tablegen(NVRubinCPXAttrs.cpp.inc   -gen-attrdef-defs    -attrdefs-dialect=tessera.target.cpx)

# Op declarations / definitions (KVCacheOp, KVExportOp, … VideoEncodeOp)
mlir_tablegen(NVRubinCPXOps.h.inc       -gen-op-decls)
mlir_tablegen(NVRubinCPXOps.cpp.inc     -gen-op-defs)

# Enum declarations (DeviceKindAttr, MemSpaceAttr, KVRoleAttr, KVLayoutAttr, KVPolicyAttr)
mlir_tablegen(NVRubinCPXEnums.h.inc     -gen-enum-decls)
mlir_tablegen(NVRubinCPXEnums.cpp.inc   -gen-enum-defs)

add_public_tablegen_target(TesseraRubinCPXTableGen)

# Optional: dialect documentation generation
# add_mlir_doc(NVRubinCPX NVRubinCPX docs/Dialects/ -gen-dialect-doc)
