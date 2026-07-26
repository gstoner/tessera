// Materialize Graph layout casts into NVIDIA operand-binding contracts.

#include "Tessera/Transforms/Passes.h"
#include "Tessera/Dialect/Tile/TileDialect.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/ADT/Twine.h"

using namespace mlir;

namespace tessera {
namespace {

static tile::TileLayoutAttr physicalLayoutForCast(OpBuilder &builder,
                                                  Operation *cast,
                                                  StringRef layout) {
  auto type = dyn_cast<RankedTensorType>(cast->getResult(0).getType());
  if (!type || !type.hasStaticShape() || type.getRank() == 0)
    return {};
  SmallVector<int64_t> extents(type.getShape());
  SmallVector<int64_t> strides(extents.size(), 1);
  if (layout == "col_major" && extents.size() == 2) {
    strides[1] = extents[0];
  } else {
    for (int i = static_cast<int>(extents.size()) - 2; i >= 0; --i)
      strides[i] = strides[i + 1] * extents[i + 1];
  }
  SmallVector<StringAttr> axes(extents.size(), builder.getStringAttr("m"));
  return tile::TileLayoutAttr::get(
      builder.getContext(), extents, strides, axes,
      /*replicaCounts=*/{}, /*replicaStrides=*/{}, /*replicaAxes=*/{},
      /*offset=*/0, /*swizzle=*/tile::TileSwizzleAttr());
}

struct NVIDIAGraphLayoutMaterializationPass
    : public PassWrapper<NVIDIAGraphLayoutMaterializationPass,
                         OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(
      NVIDIAGraphLayoutMaterializationPass)

  StringRef getArgument() const override {
    return "tessera-nvidia-materialize-layout-casts";
  }
  StringRef getDescription() const override {
    return "Consume Graph layout casts as structured Tile staging layouts";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<tile::TesseraTileDialect>();
  }

  void runOnOperation() override {
    SmallVector<Operation *> casts;
    getOperation().walk([&](Operation *op) {
      if (op->getName().getStringRef() == "tessera.cast" &&
          op->hasAttr("tessera.layout"))
        casts.push_back(op);
    });

    static const llvm::StringSet<> supported = {
        "row_major", "col_major", "bhsd", "nhwc"};
    for (Operation *cast : casts) {
      auto layout = cast->getAttrOfType<StringAttr>("tessera.layout");
      if (!layout || !supported.contains(layout.getValue())) {
        cast->emitError("NVIDIA Graph layout materializer does not support '")
            << (layout ? layout.getValue() : StringRef("<missing>")) << "'";
        signalPassFailure();
        return;
      }
      if (cast->getNumOperands() != 1 || cast->getNumResults() != 1) {
        cast->emitError("NVIDIA Graph layout materializer requires unary cast");
        signalPassFailure();
        return;
      }
      OpBuilder builder(cast);
      tile::TileLayoutAttr physical =
          physicalLayoutForCast(builder, cast, layout.getValue());
      if (!physical) {
        cast->emitError(
            "NVIDIA Graph layout materializer requires a static ranked tensor "
            "to produce structured #tile.layout");
        signalPassFailure();
        return;
      }
      for (OpOperand &use : llvm::make_early_inc_range(
               cast->getResult(0).getUses())) {
        Operation *consumer = use.getOwner();
        std::string suffix = llvm::Twine(use.getOperandNumber()).str();
        consumer->setAttr("tile.operand_layout_" + suffix, physical);
      }
      cast->getResult(0).replaceAllUsesWith(cast->getOperand(0));
      cast->erase();
    }
  }
};

} // namespace

std::unique_ptr<Pass> createNVIDIAGraphLayoutMaterializationPass() {
  return std::make_unique<NVIDIAGraphLayoutMaterializationPass>();
}

} // namespace tessera
