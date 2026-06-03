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
