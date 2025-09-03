
#include "Tessera/Transforms/Passes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

using namespace mlir;

namespace {
struct Verify : public PassWrapper<Verify, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(Verify)
  Option<bool> requireIRVersion{*this, "require-ir-version", llvm::cl::init(true)};
  void runOnOperation() override {
    auto m = getOperation();
    if (requireIRVersion && !m->hasAttr("tessera.ir.version")) {
      m.emitError("[TESSERA_VFY_MODULE_VERSION] missing tessera.ir.version");
      signalPassFailure();
    }
  }
};
}

namespace tessera {
std::unique_ptr<Pass> createVerifyTesseraIRPass() { return std::make_unique<Verify>(); }
} // namespace tessera
