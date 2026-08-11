//===- AwaitSinkingPass.cpp - effect-safe async overlap -------------------===//
//
// Sinks registered collective awaits to the first SSA consumer. The pass is
// deliberately fail-closed: it crosses only operations proven memory-effect
// free and never crosses regions, alias/view construction, stochastic work, or
// another ordered collective. This turns future SSA lineage into executable
// overlap without speculating about aliasing or effect order.
//
//===----------------------------------------------------------------------===//

#include "Tessera/Transforms/Passes.h"
#include "Tessera/Transforms/SemanticEffects.h"

#include "tessera/Dialect/Collective/IR/CollectiveDialect.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/SmallVector.h"

using namespace mlir;

namespace {

struct AwaitSinkingPass
    : public PassWrapper<AwaitSinkingPass, OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(AwaitSinkingPass)

  StringRef getArgument() const final { return "tessera-await-sinking"; }
  StringRef getDescription() const final {
    return "Sink collective awaits to their first SSA consumer without "
           "crossing effects, aliases, RNG, regions, or ordered collectives";
  }

  void runOnOperation() override {
    func::FuncOp fn = getOperation();
    SmallVector<Operation *> awaits;
    fn.walk([&](Operation *op) {
      if (isa<tessera::collective::AwaitOp>(op))
        awaits.push_back(op);
    });

    unsigned moved = 0;
    for (Operation *await : awaits) {
      if (await->getNumResults() != 1 || !await->getBlock())
        continue;

      Operation *firstUser = nullptr;
      bool sameBlock = true;
      for (Operation *user : await->getResult(0).getUsers()) {
        if (user->getBlock() != await->getBlock()) {
          sameBlock = false;
          break;
        }
        if (!firstUser || user->isBeforeInBlock(firstUser))
          firstUser = user;
      }
      if (!sameBlock || !firstUser || firstUser == await->getNextNode())
        continue;

      bool safe = true;
      for (Operation *cursor = await->getNextNode(); cursor != firstUser;
           cursor = cursor->getNextNode()) {
        if (!cursor || tessera::isSemanticSchedulingBarrier(cursor)) {
          safe = false;
          break;
        }
      }
      if (!safe)
        continue;

      await->moveBefore(firstUser);
      ++moved;
    }

    fn->setAttr("tessera.await_sinking.moved",
                IntegerAttr::get(IntegerType::get(&getContext(), 64), moved));
  }
};

} // namespace

namespace tessera {
std::unique_ptr<mlir::Pass> createAwaitSinkingPass() {
  return std::make_unique<AwaitSinkingPass>();
}
} // namespace tessera
