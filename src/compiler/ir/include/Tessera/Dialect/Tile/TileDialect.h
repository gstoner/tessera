//===- TileDialect.h - Tessera Tile IR dialect ----------------*- C++ -*-===//
//
// Sprint 9: registered Tile IR dialect (value-lane lowering spine). See
// TileOps.td.
//
//===----------------------------------------------------------------------===//

#ifndef TESSERA_DIALECT_TILE_DIALECT_H
#define TESSERA_DIALECT_TILE_DIALECT_H

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

// Generated dialect declarations (cppNamespace = ::tessera::tile).
#include "Tessera/Dialect/Tile/TileOpsDialect.h.inc"

// C1 (2026-06-23) — generated attribute declarations (#tile.layout/#tile.swizzle).
// Must precede the op classes; some ops carry these attributes.
#define GET_ATTRDEF_CLASSES
#include "Tessera/Dialect/Tile/TileAttrs.h.inc"

// Phase B (2026-06-23) — generated type declarations (!tile.async_token). Must
// precede the op classes; ops may carry the token as an operand/result type.
#define GET_TYPEDEF_CLASSES
#include "Tessera/Dialect/Tile/TileTypes.h.inc"

#define GET_OP_CLASSES
#include "Tessera/Dialect/Tile/TileOps.h.inc"

namespace tessera {
namespace tile {

/// Insert the Tile IR dialect into a DialectRegistry. Call from tessera-opt and
/// from any pass (TilingPass) whose getDependentDialects creates tile.* ops.
void registerTileDialect(::mlir::DialectRegistry &registry);

} // namespace tile
} // namespace tessera

#endif // TESSERA_DIALECT_TILE_DIALECT_H
