//===- TileRelationalLegality.h - shared Tile relation checks -*- C++ -*-===//
#ifndef TESSERA_TRANSFORMS_TILE_RELATIONAL_LEGALITY_H
#define TESSERA_TRANSFORMS_TILE_RELATIONAL_LEGALITY_H

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Support/LogicalResult.h"

namespace tessera {

mlir::LogicalResult verifyTilePipelineRelations(mlir::ModuleOp module);
mlir::LogicalResult verifyTileBarrierReuseRelations(mlir::ModuleOp module);
mlir::LogicalResult verifyWarpSpecializationRelations(mlir::ModuleOp module);

} // namespace tessera

#endif
