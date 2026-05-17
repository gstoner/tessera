//===- Canonicalize.cpp ----------------------------------------*- C++ -*-===//
//
// EBMCanonicalizePass: walks every tessera_ebm.* op and normalizes its
// metadata for downstream EBM6 lowering passes:
//
//   - Tags `tessera_ebm.langevin_step` with `tessera.ebm.manifold` (mirrors
//     the op's `manifold` attribute).
//   - Tags every canonical op with `tessera.ebm.canonical`.
//   - Normalizes `tessera_ebm.self_verify` with `beta = 0.0` to a hard
//     argmin form by removing the beta attribute (matches the Python
//     spec: `beta=None` → hard argmin).
//
// Mirrors `LegalizeSpectralPass` in scope and pattern.
//
//===----------------------------------------------------------------------===//

#include "tessera/EBM/EBMPasses.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

using namespace mlir;

namespace tessera {
namespace {

static bool isEBMOp(StringRef name) {
  return name.starts_with("tessera_ebm.");
}

struct EBMCanonicalizePass
    : public PassWrapper<EBMCanonicalizePass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(EBMCanonicalizePass)

  StringRef getArgument() const final { return "tessera-ebm-canonicalize"; }
  StringRef getDescription() const final {
    return "Normalize tessera_ebm.* ops: tag manifold, drop beta=0 on "
           "self_verify, mark canonical.";
  }

  void runOnOperation() override {
    ModuleOp mod = getOperation();
    MLIRContext *ctx = &getContext();
    OpBuilder builder(ctx);

    mod.walk([&](Operation *op) {
      StringRef name = op->getName().getStringRef();
      if (!isEBMOp(name)) return WalkResult::advance();

      // Tag manifold on langevin_step.
      if (name == "tessera_ebm.langevin_step") {
        if (auto manifold = op->getAttrOfType<StringAttr>("manifold")) {
          op->setAttr("tessera.ebm.manifold", manifold);
        } else {
          op->emitWarning("tessera_ebm.langevin_step missing `manifold`; "
                          "defaulting to 'euclidean'");
          op->setAttr("tessera.ebm.manifold",
                      StringAttr::get(ctx, "euclidean"));
        }
      }

      // Normalize self_verify(beta = 0.0) to hard argmin.
      if (name == "tessera_ebm.self_verify") {
        if (auto beta = op->getAttrOfType<FloatAttr>("beta")) {
          if (beta.getValueAsDouble() == 0.0) {
            op->removeAttr("beta");
            op->setAttr("tessera.ebm.hard_argmin", builder.getUnitAttr());
          }
        }
      }

      op->setAttr("tessera.ebm.canonical", builder.getUnitAttr());
      return WalkResult::advance();
    });
  }
};

// EBM6 stub passes — emit a remark per op, no IR rewriting.
struct EBMStubPass
    : public PassWrapper<EBMStubPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(EBMStubPass)
  std::string argName;
  std::string descName;
  std::string remarkTag;
  EBMStubPass(StringRef arg, StringRef desc, StringRef remark)
      : argName(arg.str()), descName(desc.str()), remarkTag(remark.str()) {}
  EBMStubPass(const EBMStubPass &other) = default;
  StringRef getArgument() const final { return argName; }
  StringRef getDescription() const final { return descName; }

  void runOnOperation() override {
    getOperation().walk([&](Operation *op) {
      if (isEBMOp(op->getName().getStringRef())) {
        op->emitRemark()
            << remarkTag << " stub: lowering implementation pending EBM6";
      }
    });
  }
};

}  // namespace

std::unique_ptr<mlir::Pass> createEBMCanonicalizePass() {
  return std::make_unique<EBMCanonicalizePass>();
}

std::unique_ptr<mlir::Pass> createEBMFuseEnergyGradPass() {
  return std::make_unique<EBMStubPass>(
      "tessera-ebm-fuse-energy-grad",
      "[EBM6 stub] Fuse energy + grad_y evaluations across the inner loop.",
      "fuse-energy-grad");
}

std::unique_ptr<mlir::Pass> createEBMCheckpointInnerLoopPass() {
  return std::make_unique<EBMStubPass>(
      "tessera-ebm-checkpoint-inner-loop",
      "[EBM6 stub] Rematerialize the T-step inner-loop trajectory.",
      "checkpoint-inner-loop");
}

std::unique_ptr<mlir::Pass> createEBMPipelineCandidatesPass() {
  return std::make_unique<EBMStubPass>(
      "tessera-ebm-pipeline-candidates",
      "[EBM6 stub] Map K candidates across streams / devices.",
      "pipeline-candidates");
}

}  // namespace tessera
