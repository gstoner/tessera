// Declared identity edges shared by Tile operation and lifetime verifiers.
// This proves static origins, never dynamic token completion or ownership.
#ifndef TESSERA_DIALECT_TILE_SSA_ORIGINS_H
#define TESSERA_DIALECT_TILE_SSA_ORIGINS_H
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "llvm/ADT/SmallPtrSet.h"

namespace tessera::tile {
struct SSAOrigins {
  llvm::SmallPtrSet<mlir::Value, 4> roots;
  bool complete = true;
};
inline void collectSSAOrigins(mlir::Value value, SSAOrigins &out,
                              llvm::SmallPtrSetImpl<mlir::Value> &seen) {
  if (!seen.insert(value).second) return;
  if (auto arg = mlir::dyn_cast<mlir::BlockArgument>(value)) {
    auto *block = arg.getOwner();
    if (auto loop = mlir::dyn_cast<mlir::scf::ForOp>(block->getParentOp())) {
      if (!arg.getArgNumber()) { out.complete = false; return; }
      unsigned i = arg.getArgNumber() - 1;
      collectSSAOrigins(loop.getInitArgs()[i], out, seen);
      collectSSAOrigins(block->getTerminator()->getOperand(i), out, seen);
      return;
    }
    if (block->hasNoPredecessors()) { out.complete = false; return; }
    for (auto *pred : block->getPredecessors()) {
      auto branch = mlir::dyn_cast<mlir::BranchOpInterface>(pred->getTerminator());
      if (!branch) { out.complete = false; continue; }
      for (unsigned i = 0; i < pred->getNumSuccessors(); ++i) {
        if (pred->getSuccessor(i) != block) continue;
        auto source = branch.getSuccessorOperands(i)[arg.getArgNumber()];
        if (source) collectSSAOrigins(source, out, seen);
        else out.complete = false;
      }
    }
    return;
  }
  if (auto branch = value.getDefiningOp<mlir::scf::IfOp>()) {
    unsigned i = mlir::cast<mlir::OpResult>(value).getResultNumber();
    for (auto &region : branch->getRegions()) {
      if (region.empty()) { out.complete = false; continue; }
      collectSSAOrigins(region.front().getTerminator()->getOperand(i), out, seen);
    }
    return;
  }
  if (auto loop = value.getDefiningOp<mlir::scf::ForOp>()) {
    collectSSAOrigins(loop.getRegionIterArg(mlir::cast<mlir::OpResult>(value).getResultNumber()), out, seen);
    return;
  }
  out.roots.insert(value);
}
inline SSAOrigins resolveSSAOrigins(mlir::Value value) {
  SSAOrigins out;
  llvm::SmallPtrSet<mlir::Value, 16> seen;
  collectSSAOrigins(value, out, seen);
  out.complete &= !out.roots.empty();
  return out;
}
} // namespace tessera::tile
#endif
