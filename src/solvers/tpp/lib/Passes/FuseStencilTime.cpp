//===- FuseStencilTime.cpp - fuse sibling stencils sharing a halo exchange -===//
//
// Previously a no-op.  Now it computes a real fusion plan for stencil-like
// ops within a time step.
//
// The canonical space-time fusion (see the shallow-water example in the TPP
// spec, where `%Hx = grad %h {axis x}` and `%Hy = grad %h {axis y}` both read
// the same height field `%h`): sibling stencils that read the *same* input
// value in the *same* block can share a single halo exchange of the union
// halo, instead of one exchange each.  This pass identifies those groups and
// records the plan; `-tpp-distribute-halo` then materialises one exchange per
// group instead of per op.
//
// Grouping key is (block, operand-0 value), which naturally scopes fusion to a
// single `tpp.time.step` region / basic block and never fuses across a
// producer->consumer chain (those read different values and have additive, not
// shared, halos).
//
// Emitted per member op:
//   tpp.fuse.group   : i64   group id (siblings share it)
//   tpp.fuse.halo    : i64[] union (elementwise max) halo of the group
//   tpp.fuse.members : i64   group size
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Operation.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"

using namespace mlir;

namespace {

struct FuseStencilTime
    : public PassWrapper<FuseStencilTime, OperationPass<ModuleOp>> {
  StringRef getArgument() const final { return "tpp-fuse-stencil-time"; }
  StringRef getDescription() const final {
    return "Group sibling stencils that read the same field so they share one "
           "halo exchange (space-time stencil fusion)";
  }

  void runOnOperation() final {
    ModuleOp m = getOperation();
    OpBuilder b(m.getContext());

    // Group stencil-like, halo-annotated ops by (block, input field value).
    // Preserve first-seen order for deterministic group ids.
    using Key = std::pair<Block *, Value>;
    llvm::DenseMap<Key, SmallVector<Operation *, 4>> groups;
    SmallVector<Key, 8> order;

    m.walk([&](Operation *op) {
      StringRef name = op->getName().getStringRef();
      bool isStencil = name.ends_with("tpp.grad") ||
                       name.ends_with("tpp.stencil.apply") ||
                       name.ends_with("tpp.div");
      if (!isStencil || !op->hasAttr("tpp.halo") || op->getNumOperands() == 0)
        return;
      Key k{op->getBlock(), op->getOperand(0)};
      auto it = groups.find(k);
      if (it == groups.end()) {
        groups[k] = {op};
        order.push_back(k);
      } else {
        it->second.push_back(op);
      }
    });

    int64_t gid = 0;
    for (const Key &k : order) {
      auto &members = groups[k];

      // Union (elementwise max) halo across the group.
      SmallVector<int64_t, 4> fused;
      for (Operation *op : members) {
        auto halo = op->getAttrOfType<ArrayAttr>("tpp.halo");
        if (!halo)
          continue;
        if ((int64_t)fused.size() < (int64_t)halo.size())
          fused.resize(halo.size(), 0);
        for (auto [i, a] : llvm::enumerate(halo))
          if (auto ia = dyn_cast<IntegerAttr>(a))
            fused[i] = std::max(fused[i], ia.getInt());
      }

      ArrayAttr fusedAttr = b.getI64ArrayAttr(fused);
      for (Operation *op : members) {
        op->setAttr("tpp.fuse.group", b.getI64IntegerAttr(gid));
        op->setAttr("tpp.fuse.halo", fusedAttr);
        op->setAttr("tpp.fuse.members",
                    b.getI64IntegerAttr((int64_t)members.size()));
      }
      ++gid;
    }
  }
};

} // namespace

std::unique_ptr<Pass> createFuseStencilTimePass() {
  return std::make_unique<FuseStencilTime>();
}
