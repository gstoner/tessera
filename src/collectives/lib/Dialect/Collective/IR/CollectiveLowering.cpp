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

  // The input contains only Tile operations, so lazy dialect loading has not
  // yet had a reason to initialize the collective type uniquer.  MLIR forbids
  // loading a dialect from inside `runOnOperation()` -- the pass manager may be
  // multi-threaded there, and the context's registry is not writable under that
  // contract (it is a hard `LLVM ERROR` on an assertions-enabled build and
  // undefined behavior under NDEBUG).  Declare the dependency instead so the
  // context loads it before the pipeline starts.
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<TesseraCollectiveDialect>();
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();
    SmallVector<Operation *> work;
    module.walk([&](Operation *op) {
      StringRef name = op->getName().getStringRef();
      if (name == "tile.all_reduce" || name == "tile.reduce_scatter" ||
          name == "tile.all_gather" || name == "tile.all_to_all" ||
          name == "tile.collective_permute")
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
      bool permute = op->getName().getStringRef() == "tile.collective_permute";
      if (!meshAxis || (!permute && (!tensorAxis || !reduction))) {
        op->emitError(
            "collective Target lowering requires mesh_axis, tensor_axis, and reduction");
        return signalPassFailure();
      }
      int64_t normalizedAxis = permute ? 0 : tensorAxis.getInt();
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
      if (permute) {
        dispatchState.addAttribute("source_peers", op->getAttr("source_peers"));
        dispatchState.addAttribute("target_peers", op->getAttr("target_peers"));
      } else {
        dispatchState.addAttribute("tensor_axis",
                                   builder.getI64IntegerAttr(normalizedAxis));
        dispatchState.addAttribute("reduction", reduction);
      }
      if (Attribute worldSize = op->getAttr("world_size"))
        dispatchState.addAttribute("world_size", worldSize);
      if (Attribute dtype = op->getAttr("dtype"))
        dispatchState.addAttribute("dtype", dtype);
      if (Attribute chunkBytes = op->getAttr("chunk_bytes"))
        dispatchState.addAttribute("chunk_bytes", chunkBytes);
      // These attributes are part of the compiler-owned transport identity,
      // not runtime hints. Preserve them into Target IR so rank-local MPI can
      // admit the exact SSA order/subgroup and the artifact digest changes if
      // any Schedule/reshard identity changes.
      for (StringRef attrName : {
               "ordinal", "subgroup", "reshard_plan_digest", "region_path",
               "matching_rounds", "scatter_axis", "gather_axis"})
        if (Attribute attr = op->getAttr(attrName))
          dispatchState.addAttribute(attrName, attr);
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
