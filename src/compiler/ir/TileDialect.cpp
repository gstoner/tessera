//===- TileDialect.cpp - Tessera Tile IR dialect --------------*- C++ -*-===//

#include "Tessera/Dialect/Tile/TileDialect.h"

#include "mlir/IR/DialectImplementation.h"

#include "Tessera/Dialect/Tile/TileOpsDialect.cpp.inc"

#define GET_OP_CLASSES
#include "Tessera/Dialect/Tile/TileOps.cpp.inc"

namespace tessera {
namespace tile {

void TesseraTileDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "Tessera/Dialect/Tile/TileOps.cpp.inc"
      >();
  // Sprint 9: the value-lane contraction + linalg ops above are registered and
  // verified. The artifact lane still emits other transient tile.* ops
  // (tile.mma / tile.async_copy / tile.kv_cache / debug husks) that are not yet
  // ODS-registered — allow them as opaque so registering this dialect does not
  // break the artifact pipeline. The Apple *value* lane produces only the
  // registered ops, so it runs with NO --allow-unregistered-dialect (the win);
  // registering the remaining tile ops is a follow-on.
  allowUnknownOperations(true);
}

void registerTileDialect(::mlir::DialectRegistry &registry) {
  registry.insert<TesseraTileDialect>();
}

} // namespace tile
} // namespace tessera
