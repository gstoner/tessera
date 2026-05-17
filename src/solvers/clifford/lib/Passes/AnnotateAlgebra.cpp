//===- AnnotateAlgebra.cpp ----------------------------------*- C++ -*-===//
//
// CliffordAnnotateAlgebraPass: walks every `tessera_clifford.*` op, reads
// its `algebra` attribute (an I64ArrayAttr `[p, q, r]`), validates the
// signature against the v1 allow-list, and attaches derived metadata:
//
//   tessera.clifford.dim          : 2^(p+q+r) — algebra dimension
//   tessera.clifford.allow_listed : bool — true iff (p,q,r) in
//                                         {(3,0,0), (1,3,0)}
//   tessera.clifford.canonical    : unit attr — present iff the op is
//                                   ready for GA8 lowering
//
// GA8 lowering passes gate on `canonical` and refuse to proceed on
// out-of-allow-list signatures, emitting a precise diagnostic naming the
// op and the unsupported signature.
//
// This is the GA7-load-bearing pass: it walks the IR, doesn't yet emit
// any new ops (annotation-only). Mirrors LegalizeSpectralPass.
//
//===----------------------------------------------------------------------===//

#include "tessera/Clifford/CliffordPasses.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

#include <cstdint>

using namespace mlir;

namespace tessera {
namespace {

// v1 allow-list per docs/audit/ga_scope_lock.md § Q1.
static bool isAllowListedSignature(int64_t p, int64_t q, int64_t r) {
  if (p == 3 && q == 0 && r == 0) return true;  // 3D Euclidean
  if (p == 1 && q == 3 && r == 0) return true;  // Minkowski spacetime
  return false;
}

static bool isCliffordOp(StringRef name) {
  return name.starts_with("tessera_clifford.");
}

struct CliffordAnnotateAlgebraPass
    : public PassWrapper<CliffordAnnotateAlgebraPass,
                         OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(CliffordAnnotateAlgebraPass)

  StringRef getArgument() const final {
    return "tessera-clifford-annotate-algebra";
  }
  StringRef getDescription() const final {
    return "Validate Cl(p,q,r) signatures on tessera_clifford.* ops and "
           "attach derived metadata (dim, allow_listed, canonical).";
  }

  void runOnOperation() override {
    ModuleOp mod = getOperation();
    MLIRContext *ctx = &getContext();
    OpBuilder builder(ctx);
    bool anyError = false;

    mod.walk([&](Operation *op) {
      StringRef name = op->getName().getStringRef();
      if (!isCliffordOp(name)) return WalkResult::advance();

      // Every clifford op must carry an `algebra` attribute.
      auto algebra = op->getAttrOfType<ArrayAttr>("algebra");
      if (!algebra) {
        op->emitError("tessera_clifford op missing required `algebra` attribute");
        anyError = true;
        return WalkResult::interrupt();
      }
      if (algebra.size() != 3) {
        op->emitError(
            "tessera_clifford `algebra` must be [p, q, r]; got size ")
            << algebra.size();
        anyError = true;
        return WalkResult::interrupt();
      }
      int64_t p = -1, q = -1, r = -1;
      auto pAttr = dyn_cast<IntegerAttr>(algebra[0]);
      auto qAttr = dyn_cast<IntegerAttr>(algebra[1]);
      auto rAttr = dyn_cast<IntegerAttr>(algebra[2]);
      if (!pAttr || !qAttr || !rAttr) {
        op->emitError(
            "tessera_clifford `algebra` entries must be IntegerAttr");
        anyError = true;
        return WalkResult::interrupt();
      }
      p = pAttr.getInt();
      q = qAttr.getInt();
      r = rAttr.getInt();
      if (p < 0 || q < 0 || r < 0) {
        op->emitError("tessera_clifford signature must be non-negative; got (")
            << p << ", " << q << ", " << r << ")";
        anyError = true;
        return WalkResult::interrupt();
      }
      int64_t n = p + q + r;
      int64_t dim = 1LL << n;
      bool allowListed = isAllowListedSignature(p, q, r);

      op->setAttr("tessera.clifford.dim", builder.getI64IntegerAttr(dim));
      op->setAttr("tessera.clifford.allow_listed",
                  builder.getBoolAttr(allowListed));
      if (allowListed) {
        op->setAttr("tessera.clifford.canonical", builder.getUnitAttr());
      } else {
        op->emitWarning("tessera_clifford op uses signature Cl(")
            << p << ", " << q << ", " << r << ") which is not in the v1 "
            << "allow-list {Cl(3,0), Cl(1,3)}; GA8 lowering will refuse";
      }

      // Validate the optional `grades` restriction if present.
      if (auto grades = op->getAttrOfType<ArrayAttr>("grades")) {
        for (Attribute g : grades) {
          auto gi = dyn_cast<IntegerAttr>(g);
          if (!gi) continue;
          int64_t k = gi.getInt();
          if (k < 0 || k > n) {
            op->emitError("tessera_clifford grade ")
                << k << " is out of range for Cl(" << p << ", " << q << ", "
                << r << ")";
            anyError = true;
            return WalkResult::interrupt();
          }
        }
      }
      return WalkResult::advance();
    });

    if (anyError) {
      signalPassFailure();
    }
  }
};

// ---------------------------------------------------------------------------
// GA8 stub passes — emit a remark, no IR rewriting.
// ---------------------------------------------------------------------------

struct CliffordStubPass
    : public PassWrapper<CliffordStubPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(CliffordStubPass)
  std::string argName;
  std::string descName;
  std::string remarkTag;
  CliffordStubPass(StringRef arg, StringRef desc, StringRef remark)
      : argName(arg.str()), descName(desc.str()), remarkTag(remark.str()) {}
  CliffordStubPass(const CliffordStubPass &other) = default;
  StringRef getArgument() const final { return argName; }
  StringRef getDescription() const final { return descName; }

  void runOnOperation() override {
    getOperation().walk([&](Operation *op) {
      if (isCliffordOp(op->getName().getStringRef())) {
        op->emitRemark()
            << remarkTag << " stub: lowering implementation pending GA8";
      }
    });
  }
};

}  // namespace

std::unique_ptr<mlir::Pass> createCliffordAnnotateAlgebraPass() {
  return std::make_unique<CliffordAnnotateAlgebraPass>();
}

std::unique_ptr<mlir::Pass> createCliffordExpandProductTablePass() {
  return std::make_unique<CliffordStubPass>(
      "tessera-clifford-expand-product-table",
      "[GA8 stub] Lower clifford.geo_product to a sparse Cayley contraction.",
      "expand-product-table");
}

std::unique_ptr<mlir::Pass> createCliffordGradeFusionPass() {
  return std::make_unique<CliffordStubPass>(
      "tessera-clifford-grade-fusion",
      "[GA8 stub] Fuse grade-projection chains into restricted contractions.",
      "grade-fusion");
}

std::unique_ptr<mlir::Pass> createCliffordRotorSandwichFoldPass() {
  return std::make_unique<CliffordStubPass>(
      "tessera-clifford-rotor-sandwich-fold",
      "[GA8 stub] Fold R x R† into a direct rotor-conjugation kernel.",
      "rotor-sandwich-fold");
}

}  // namespace tessera
