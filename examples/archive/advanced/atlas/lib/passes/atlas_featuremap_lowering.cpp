
// Lower atlas.feature.map to concrete Target-IR kernels (poly features, random Fourier features, etc.)
// cmd: -tessera-atlas-featuremap-lower
#include "mlir/Pass/Pass.h"
using namespace mlir;
namespace {
struct AtlasFeatureMapLoweringPass : PassWrapper<AtlasFeatureMapLoweringPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(AtlasFeatureMapLoweringPass)
  StringRef getArgument() const final { return "tessera-atlas-featuremap-lower"; }
  StringRef getDescription() const final { return "Lower feature maps to target kernels (tensorcore/MFMA)."; }
  void runOnOperation() override {
    // TODO: Pattern: atlas.feature.map(kind="poly", degree=d) -> tti.poly.kernel(...)
  }
};
}
std::unique_ptr<Pass> createAtlasFeatureMapLoweringPass() { return std::make_unique<AtlasFeatureMapLoweringPass>(); }
