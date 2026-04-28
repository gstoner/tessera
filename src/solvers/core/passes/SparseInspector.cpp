//===- SparseInspector.cpp — tag ops with sparsity hints -----------------*- C++ -*-===//
//
// Walks tensor ops; measures or estimates fill fraction; tags any op with
// fill_fraction < --sparse-threshold as tessera_solver.sparse_hint = true.
//
// Attr contract:
//   Input  (optional): tessera_solver.fill_fraction   <f32 scalar>
//   Output (added):    tessera_solver.sparse_hint      <UnitAttr>
//                      tessera_solver.fill_fraction    (preserved / computed)
//
//===----------------------------------------------------------------------===//

#include "SolversPasses.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/CommandLine.h"

using namespace mlir;

namespace {

static constexpr double kDefaultSparseThreshold = 0.05; // 5 %

struct SparseInspectorPass
    : PassWrapper<SparseInspectorPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(SparseInspectorPass)

  // Pass option: override the sparsity threshold.
  Option<double> sparseThreshold{
      *this, "sparse-threshold",
      llvm::cl::desc("Fill fraction below which an op is tagged sparse"),
      llvm::cl::init(kDefaultSparseThreshold)};

  StringRef getArgument() const final { return "tessera-sparse-inspector"; }
  StringRef getDescription() const final {
    return "Tag tensor ops whose fill-fraction is below the sparse threshold "
           "with tessera_solver.sparse_hint";
  }

  void runOnOperation() override {
    ModuleOp mod = getOperation();
    MLIRContext *ctx = mod.getContext();

    mod.walk([&](Operation *op) {
      // Skip non-tensor ops (e.g. control flow, constants).
      if (op->getNumResults() == 0)
        return;

      // Check for an explicit fill_fraction attr first.
      double fill = -1.0;
      if (auto attr = op->getAttrOfType<FloatAttr>("tessera_solver.fill_fraction")) {
        fill = attr.getValueAsDouble();
      } else {
        // Heuristic: estimate from result type shape.
        // Ops whose name contains "sparse" or "csr" are assumed sparse.
        StringRef opName = op->getName().getStringRef();
        if (opName.contains("sparse") || opName.contains("csr") ||
            opName.contains("coo") || opName.contains("ell")) {
          fill = 0.005; // assume very sparse
        } else if (opName.contains("diag")) {
          fill = 0.01;
        } else {
          // Cannot determine fill without profiling data; skip.
          return;
        }
      }

      if (fill < 0.0 || fill > 1.0)
        return;

      // Record fill fraction (may have been inferred above).
      if (!op->hasAttr("tessera_solver.fill_fraction")) {
        op->setAttr("tessera_solver.fill_fraction",
                    FloatAttr::get(Float32Type::get(ctx),
                                  static_cast<float>(fill)));
      }

      // Tag sparse ops.
      if (fill < sparseThreshold) {
        op->setAttr("tessera_solver.sparse_hint", UnitAttr::get(ctx));
      } else {
        // Remove stale hint if fill has changed.
        op->removeAttr("tessera_solver.sparse_hint");
      }
    });
  }
};

} // namespace

namespace tessera {
namespace passes {
std::unique_ptr<Pass> createSparseInspectorPass() {
  return std::make_unique<SparseInspectorPass>();
}
} // namespace passes
} // namespace tessera
