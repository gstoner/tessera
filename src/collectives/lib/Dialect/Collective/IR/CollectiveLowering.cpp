#include "tessera/Dialect/Collective/IR/CollectiveDialect.h"
#include "tessera/Dialect/Collective/IR/CollectivePasses.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/SmallVector.h"

using namespace mlir;

namespace tessera::collective {
namespace {

struct LowerTileCollectivesPass
    : PassWrapper<LowerTileCollectivesPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LowerTileCollectivesPass)

  StringRef getArgument() const final {
    return "tessera-lower-tile-collectives";
  }
  StringRef getDescription() const final {
    return "Lower Tile collectives to asynchronous portable Target IR";
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();
    // The input contains only Tile operations, so lazy dialect loading has not
    // yet had a reason to initialize the collective type uniquer. Load the
    // Target dialect before constructing its first FutureType.
    module.getContext()->getOrLoadDialect<TesseraCollectiveDialect>();
    SmallVector<Operation *> work;
    module.walk([&](Operation *op) {
      StringRef name = op->getName().getStringRef();
      if (name == "tile.all_reduce" || name == "tile.reduce_scatter" ||
          name == "tile.all_gather" || name == "tile.all_to_all")
        work.push_back(op);
    });

    Builder moduleBuilder(module.getContext());
    for (Operation *op : work) {
      if (op->getNumOperands() != 1 || op->getNumResults() != 1) {
        op->emitError("collective Target lowering requires one input and one output");
        return signalPassFailure();
      }
      auto meshAxis = op->getAttrOfType<StringAttr>("mesh_axis");
      auto tensorAxis = op->getAttrOfType<IntegerAttr>("tensor_axis");
      auto reduction = op->getAttrOfType<StringAttr>("reduction");
      if (!meshAxis || !tensorAxis || !reduction) {
        op->emitError(
            "collective Target lowering requires mesh_axis, tensor_axis, and reduction");
        return signalPassFailure();
      }
      int64_t normalizedAxis = tensorAxis.getInt();
      if (normalizedAxis < 0) {
        auto shaped = dyn_cast<ShapedType>(op->getOperand(0).getType());
        if (!shaped || !shaped.hasRank() || normalizedAxis < -shaped.getRank()) {
          op->emitError(
              "collective Target lowering cannot normalize tensor_axis");
          return signalPassFailure();
        }
        normalizedAxis += shaped.getRank();
      }

      StringRef tileName = op->getName().getStringRef();
      std::string targetName =
          ("tessera_collective." + tileName.drop_front(5)).str();
      OpBuilder builder(op);
      Type outputType = op->getResult(0).getType();
      OperationState dispatchState(op->getLoc(), targetName);
      dispatchState.addOperands(op->getOperand(0));
      dispatchState.addTypes(FutureType::get(op->getContext(), outputType));
      dispatchState.addAttribute("mesh_axis", meshAxis);
      dispatchState.addAttribute("tensor_axis",
                                 builder.getI64IntegerAttr(normalizedAxis));
      dispatchState.addAttribute("reduction", reduction);
      if (Attribute worldSize = op->getAttr("world_size"))
        dispatchState.addAttribute("world_size", worldSize);
      if (Attribute dtype = op->getAttr("dtype"))
        dispatchState.addAttribute("dtype", dtype);
      if (Attribute chunkBytes = op->getAttr("chunk_bytes"))
        dispatchState.addAttribute("chunk_bytes", chunkBytes);
      dispatchState.addAttribute("tessera.collective.abi",
                                 builder.getStringAttr("v1"));
      dispatchState.addAttribute("tessera.collective.source",
                                 builder.getStringAttr(tileName));
      Operation *dispatch = builder.create(dispatchState);

      OperationState awaitState(op->getLoc(), "tessera_collective.await");
      awaitState.addOperands(dispatch->getResult(0));
      awaitState.addTypes(outputType);
      Operation *await = builder.create(awaitState);
      op->getResult(0).replaceAllUsesWith(await->getResult(0));
      op->erase();
    }

    if (!work.empty()) {
      module->setAttr("tessera.collective.target_abi",
                      moduleBuilder.getStringAttr("tessera.collective.v1"));
      module->setAttr("tessera.collective.transport",
                      moduleBuilder.getStringAttr("runtime_adapter"));
    }
  }
};

} // namespace

std::unique_ptr<Pass> createLowerTileCollectivesPass() {
  return std::make_unique<LowerTileCollectivesPass>();
}

void registerCollectivePasses() {
  PassRegistration<LowerTileCollectivesPass>();
}

} // namespace tessera::collective
