
//===- NVRubinCPX.cpp - Tessera Rubin CPX Dialect Implementation -----------===//
//
// Registers the `tessera.target.cpx` dialect including all ODS-generated
// types, attributes, enums, and operations.
//
// The `initialize()` method is the MLIR-idiomatic hook for post-construction
// dialect setup; it replaces the old constructor-body pattern.
//
//===-----------------------------------------------------------------------===//

#include "tessera/Target/NVRubinCPX/NVRubinCPX.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;

// ── ODS-generated dialect definition ─────────────────────────────────────────
// Provides: NVRubinCPXDialect::NVRubinCPXDialect() constructor body and
//           NVRubinCPXDialect::initialize() default hook.
#include "NVRubinCPXDialect.cpp.inc"

// ── ODS-generated enum definitions ───────────────────────────────────────────
#include "NVRubinCPXEnums.cpp.inc"

// ── ODS-generated attribute definitions ──────────────────────────────────────
#define GET_ATTRDEF_CLASSES
#include "NVRubinCPXAttrs.cpp.inc"

// ── ODS-generated type definitions ───────────────────────────────────────────
#define GET_TYPEDEF_CLASSES
#include "NVRubinCPXTypes.cpp.inc"

// ── ODS-generated op definitions (base layer) ────────────────────────────────
// Custom verifiers live in CPXTargetIROps.cpp; this brings in the scaffolding.
#define GET_OP_CLASSES
#include "NVRubinCPXOps.cpp.inc"

namespace tessera {
namespace target {

//===----------------------------------------------------------------------===//
// NVRubinCPXDialect — initialize()
//===----------------------------------------------------------------------===//
// Called once by MLIR context after the dialect is registered.
// All types, attrs, and ops declared in NVRubinCPX.td are wired in here.
//
void NVRubinCPXDialect::initialize() {
  // Register NVFP4 and NVFP6 scalar storage types
  addTypes<
    NVFP4Type,
    NVFP6Type
  >();

  // Register CPX-specific attributes
  addAttributes<
    CapabilitiesAttr
  >();

  // Register all CPX ops:
  //   KVCacheOp, KVExportOp, KVImportOp, KVPrefetchOp
  //   AttnPrefillFusedOp
  //   VideoDecodeOp, VideoEncodeOp
  addOperations<
#define GET_OP_LIST
#include "NVRubinCPXOps.cpp.inc"
  >();
}

} // namespace target
} // namespace tessera
