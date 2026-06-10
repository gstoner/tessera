
#include "Tessera/Transforms/Passes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/StringSet.h"

using namespace mlir;

namespace {
struct Verify : public PassWrapper<Verify, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(Verify)
  Verify() = default;
  Verify(const Verify &other) : PassWrapper(other) {}
  Option<bool> requireIRVersion{*this, "require-ir-version", llvm::cl::init(true)};
  // Audit 2026-06-10 — pipeline-completeness checking. Lowering passes gate
  // per-op via notifyMatchFailure, so a partially-lowered module still
  // "succeeds"; appending `tessera-verify{forbid-ops=tessera.matmul,...}` to a
  // pipeline turns a survivor into a stable named diagnostic (Decision #21).
  ListOption<std::string> forbidOps{
      *this, "forbid-ops",
      llvm::cl::desc("Comma-separated op names that must NOT appear in the "
                     "module (post-lowering completeness check)")};
  StringRef getArgument() const override { return "tessera-verify"; }
  StringRef getDescription() const override {
    return "Verify Tessera IR module invariants (version attribute; optional "
           "post-lowering forbidden-op completeness check)";
  }
  void runOnOperation() override {
    auto m = getOperation();
    if (requireIRVersion && !m->hasAttr("tessera.ir.version")) {
      m.emitError("[TESSERA_VFY_MODULE_VERSION] missing tessera.ir.version");
      signalPassFailure();
    }
    if (!forbidOps.empty()) {
      llvm::StringSet<> forbidden;
      for (const std::string &name : forbidOps)
        forbidden.insert(name);
      m.walk([&](Operation *op) {
        if (forbidden.contains(op->getName().getStringRef())) {
          op->emitError("[TESSERA_VFY_FORBIDDEN_OP] op survived lowering but "
                        "is forbidden at this pipeline stage");
          signalPassFailure();
        }
      });
    }
  }
};
}

namespace tessera {
std::unique_ptr<Pass> createVerifyTesseraIRPass() { return std::make_unique<Verify>(); }
} // namespace tessera
