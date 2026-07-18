//===- DistributeHalo.cpp - materialise overlapped halo exchanges ---------===//
//
// Turns the halo widths inferred by `-tpp-halo-infer` into *actual* exchange
// operations in the IR (previously this pass was a no-op that only carried a
// description string).
//
// For every stencil-like consumer carrying a `tpp.halo` array attribute, we
// insert a `tpp.halo.exchange` op on its operand-0 field and rewrite the
// consumer to read the exchanged value.  The exchange op carries the plan a
// backend / the LSA prefetcher needs:
//
//   tpp.halo             per-dim ghost width (copied from the consumer)
//   tpp.mesh.axes        mesh axes to exchange over (from the module's
//                        `tessera.mesh.axes`, else empty)
//   tpp.dist.local_only  true when there is no mesh (periodic wrap is local)
//   tpp.dist.overlap     comm-queue token; "comm_q_default" when a mesh is
//                        present so `-tpp-async-prefetch`/codegen can overlap
//                        the transfer with compute, "none" when local-only
//
// Fusion-aware: when `-tpp-fuse-stencil-time` has grouped sibling stencils
// (same input field) under a `tpp.fuse.group`, all members of a group share a
// *single* exchange of the fused (`tpp.fuse.halo`) width instead of one each.
// Without fusion metadata every consumer gets its own exchange, exactly as
// before.
//
// Idempotent: a consumer already fed by a `tpp.halo.exchange`, or already
// marked `tpp.halo.distributed`, is skipped, so running inside the full
// pipeline (or twice) does not stack exchanges.
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

struct DistributeHalo
    : public PassWrapper<DistributeHalo, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(DistributeHalo)

  StringRef getArgument() const final { return "tpp-distribute-halo"; }
  StringRef getDescription() const final {
    return "Materialise tpp.halo.exchange ops (overlapped ghost-cell exchange) "
           "in front of each halo-annotated stencil consumer";
  }

  void runOnOperation() final {
    ModuleOp m = getOperation();
    MLIRContext *ctx = m.getContext();
    OpBuilder b(ctx);

    // Mesh axes (if any) come from the module; their presence decides whether
    // an exchange is a real neighbour comm or a local periodic wrap.
    auto meshAxes = m->getAttrOfType<ArrayAttr>("tessera.mesh.axes");
    bool hasMesh = meshAxes && !meshAxes.empty();

    // Collect first — we mutate operands while walking.
    SmallVector<Operation *, 8> consumers;
    m.walk([&](Operation *op) {
      if (op->hasAttr("tpp.halo") && !op->hasAttr("tpp.halo.distributed"))
        consumers.push_back(op);
    });

    // One exchange per (input field, fusion group).  Members of a fusion
    // group (same field, same tpp.fuse.group) reuse the first exchange; ops
    // without fusion metadata get a unique key so each keeps its own exchange.
    llvm::DenseMap<std::pair<Value, int64_t>, Operation *> exchangeFor;
    int64_t uniq = -1; // distinct negative keys for un-fused ops

    for (Operation *op : consumers) {
      if (op->getNumOperands() == 0)
        continue;
      Value field = op->getOperand(0);

      // Skip if the field is already an exchanged value.
      if (Operation *def = field.getDefiningOp())
        if (def->getName().getStringRef().ends_with("tpp.halo.exchange"))
          continue;

      int64_t groupKey = uniq--;
      if (auto g = op->getAttrOfType<IntegerAttr>("tpp.fuse.group"))
        groupKey = g.getInt();

      std::pair<Value, int64_t> key{field, groupKey};
      Operation *ex = exchangeFor.lookup(key);
      if (!ex) {
        // Prefer the fused (union) halo when fusion ran, else the op's halo.
        auto halo = op->getAttrOfType<ArrayAttr>("tpp.fuse.halo");
        if (!halo)
          halo = op->getAttrOfType<ArrayAttr>("tpp.halo");

        b.setInsertionPoint(op);
        OperationState state(op->getLoc(), "tpp.halo.exchange");
        state.addOperands(field);
        state.addTypes(field.getType());
        state.addAttribute("tpp.halo", halo);
        state.addAttribute("tpp.mesh.axes",
                           hasMesh ? (Attribute)meshAxes
                                   : (Attribute)b.getArrayAttr({}));
        state.addAttribute("tpp.dist.local_only", b.getBoolAttr(!hasMesh));
        state.addAttribute(
            "tpp.dist.overlap",
            b.getStringAttr(hasMesh ? "comm_q_default" : "none"));
        ex = b.create(state);
        exchangeFor[key] = ex;
      }

      op->setOperand(0, ex->getResult(0));
      op->setAttr("tpp.halo.distributed", b.getUnitAttr());
    }
  }
};

} // namespace

std::unique_ptr<Pass> createDistributeHaloPass() {
  return std::make_unique<DistributeHalo>();
}
