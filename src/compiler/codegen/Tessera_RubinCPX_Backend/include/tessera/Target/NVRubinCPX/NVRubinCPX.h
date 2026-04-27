
//===- NVRubinCPX.h - Tessera Target IR: NVIDIA Rubin CPX dialect -*- C++ -*-===//
//
// Umbrella header for the `tessera.target.cpx` dialect.
//
// Includes:
//   - ODS-generated dialect class declaration  (NVRubinCPXDialect.h.inc)
//   - ODS-generated type declarations           (NVRubinCPXTypes.h.inc)
//   - ODS-generated attribute declarations      (NVRubinCPXAttrs.h.inc)
//   - ODS-generated enum declarations           (NVRubinCPXEnums.h.inc)
//   - ODS-generated op declarations             (NVRubinCPXOps.h.inc)
//
// All .h.inc files are produced by mlir-tblgen from NVRubinCPX.td and written
// into the CMake build directory.  They are reachable because
//   target_include_directories(...  $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/../include>)
// is set in lib/CMakeLists.txt.
//
//===--------------------------------------------------------------------------===//

#pragma once

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

// ── ODS-generated dialect declaration ────────────────────────────────────────
// Contains: class NVRubinCPXDialect : public Dialect { ... }
#include "NVRubinCPXDialect.h.inc"

// ── ODS-generated enum helpers ────────────────────────────────────────────────
// Contains: enum class DeviceKind, MemSpace, KVRole, KVLayout, KVPolicy
#include "NVRubinCPXEnums.h.inc"

// ── ODS-generated attribute declarations ─────────────────────────────────────
// Contains: CapabilitiesAttr, DeviceKindAttr, MemSpaceAttr, KVRoleAttr, …
#define GET_ATTRDEF_CLASSES
#include "NVRubinCPXAttrs.h.inc"

// ── ODS-generated type declarations ──────────────────────────────────────────
// Contains: NVFP4Type, NVFP6Type
#define GET_TYPEDEF_CLASSES
#include "NVRubinCPXTypes.h.inc"

// ── ODS-generated op declarations ────────────────────────────────────────────
// Contains: KVCacheOp, KVExportOp, KVImportOp, KVPrefetchOp,
//           AttnPrefillFusedOp, VideoDecodeOp, VideoEncodeOp
#define GET_OP_CLASSES
#include "NVRubinCPXOps.h.inc"
