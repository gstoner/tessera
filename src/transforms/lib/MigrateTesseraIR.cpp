
#include "Tessera/Transforms/Passes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

using namespace mlir;

namespace {
struct Migrate : public PassWrapper<Migrate, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(Migrate)
  Migrate() = default;
  Migrate(const Migrate &other) : PassWrapper(other) {}
  Option<std::string> toVersion{*this, "to-version", llvm::cl::init("0.9")};
  StringRef getArgument() const override { return "tessera-migrate"; }
  StringRef getDescription() const override { return "Migrate Tessera IR version metadata"; }
  void runOnOperation() override {
    auto m = getOperation();
    m->setAttr("tessera.ir.version", StringAttr::get(m->getContext(), toVersion));
  }
};
}

namespace tessera {
std::unique_ptr<Pass> createMigrateTesseraIRPass() { return std::make_unique<Migrate>(); }
} // namespace tessera
