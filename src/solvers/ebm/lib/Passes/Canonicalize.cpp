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
      //
      // `manifold` is a semantic key (Decision #21a): it selects which
      // integrator runs, so it fails CLOSED. This previously warned and
      // defaulted to "euclidean", which is the worst available behavior --
      // a Euclidean step on spherical or bivector state does not diverge or
      // error, it converges and reports a confidently wrong result. The
      // fail-closed shape mirrors AnnotateAlgebra's handling of `algebra`.
      if (name == "tessera_ebm.langevin_step") {
        auto manifold = op->getAttrOfType<StringAttr>("manifold");
        if (!manifold) {
          op->emitError("tessera_ebm.langevin_step missing required "
                        "`manifold` attribute; expected 'euclidean', "
                        "'sphere', or 'bivector'");
          failed_ = true;
          return WalkResult::interrupt();
        }
        if (!isKnownManifold(manifold.getValue())) {
          op->emitError("tessera_ebm.langevin_step has unrecognized "
                        "`manifold` value '")
              << manifold.getValue()
              << "'; expected 'euclidean', 'sphere', or 'bivector'";
          failed_ = true;
          return WalkResult::interrupt();
        }
        op->setAttr("tessera.ebm.manifold", manifold);
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

    if (failed_) signalPassFailure();
  }

private:
  // Decision #21a: the legal set is declared, and anything outside it is an
  // error. Keep in sync with EBM_ManifoldAttr in EBMOps.td.
  static bool isKnownManifold(StringRef value) {
    return value == "euclidean" || value == "sphere" || value == "bivector";
  }

  bool failed_ = false;
};

}  // namespace

std::unique_ptr<mlir::Pass> createEBMCanonicalizePass() {
  return std::make_unique<EBMCanonicalizePass>();
}

}  // namespace tessera
