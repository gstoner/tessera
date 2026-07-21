//===- AutodiffPass.cpp - Reverse-mode autodiff at Graph IR ----*- C++ -*-===//
//
// Phase F4 of docs/audit/roadmap/ROADMAP_AUDIT.md. Consumes the
// `Tessera_AdjointInterface` op trait (see
// `src/compiler/ir/include/Tessera/AdjointInterface.td`) to emit backward
// computation for any ``func.func`` annotated with the
// ``tessera.autodiff = "reverse"`` attribute.
//
// Pass shape (four-step reverse walk):
//
//   1. Identify funcs to differentiate (annotation-driven).
//   2. Walk the forward region top-down; record op order.
//   3. Walk in reverse program order. For each op:
//        - Look up cotangents for its results in the cotangent map.
//        - If `op` implements `AdjointInterface` and is differentiable,
//          dispatch to `buildAdjoint`.
//        - If `op->customAdjointName()` is non-empty, the implementation
//          is responsible for emitting a `tessera.custom_adjoint_call`
//          placeholder that the runtime VJP registry resolves.
//        - Otherwise emit a diagnostic per Architecture Decision #21.
//      Accumulate returned cotangents into the input slots' map entries.
//   4. The cotangents at function arguments become the new function's
//      additional outputs (or — for `Module.parameters()` — are routed via
//      a side-channel that becomes `param.grad` in the Python wrapper).
//
// Effect-aware adjoint collective insertion (Phase F5) runs **after** this
// pass and inserts `reduce_scatter` / `all_gather` for adjoints of
// distributed parameters.
//
// Cross-references:
//   * AdjointInterface.td — the ODS interface this pass dispatches via
//   * AdjointInterface.cpp — per-op `buildAdjoint` impls
//   * docs/spec/AUTODIFF_SPEC.md §Phase F4
//   * python/tessera/autodiff/ — the v1 numpy-tape impl that this pass
//     replaces internally while keeping the same public surface
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"

// TesseraOps includes the generated AdjointInterface declaration and provides
// CustomAdjointCallOp for the dynamic cotangent seed below.
#include "Tessera/IR/TesseraOps.h"

namespace tessera {

namespace {

/// Marker attribute on `func.func` to opt into autodiff transformation.
constexpr const char *kAutodiffMarker = "tessera.autodiff";

/// Track per-Value cotangents. Map keys are forward Values; map values are
/// the Value of the cotangent emitted into the backward IR.
using CotangentMap = llvm::DenseMap<mlir::Value, mlir::Value>;

/// Accumulate `g` into `cotan[v]`. First contribution stores directly; later
/// contributions add (float → arith.addf, integer → arith.addi) so an integer
/// cotangent path doesn't feed addf an int type and trip the op verifier.
void accumulateCotangent(mlir::OpBuilder &builder,
                          CotangentMap &cotan,
                          mlir::Value v,
                          mlir::Value g) {
  if (!g)
    return;
  auto it = cotan.find(v);
  if (it == cotan.end()) {
    cotan[v] = g;
    return;
  }
  auto loc = g.getLoc();
  mlir::Type elemTy = mlir::getElementTypeOrSelf(g.getType());
  mlir::Value sum;
  if (llvm::isa<mlir::FloatType>(elemTy))
    sum = builder.create<mlir::arith::AddFOp>(loc, it->second, g).getResult();
  else
    sum = builder.create<mlir::arith::AddIOp>(loc, it->second, g).getResult();
  cotan[v] = sum;
}

class AutodiffPass : public mlir::PassWrapper<
                         AutodiffPass,
                         mlir::OperationPass<mlir::func::FuncOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(AutodiffPass)

  llvm::StringRef getArgument() const final { return "tessera-autodiff"; }

  llvm::StringRef getDescription() const final {
    return "Reverse-mode autodiff via the Tessera AdjointInterface op trait. "
           "Phase F4 of docs/audit/roadmap/ROADMAP_AUDIT.md.";
  }

  void getDependentDialects(mlir::DialectRegistry &registry) const override {
    registry.insert<mlir::arith::ArithDialect, mlir::func::FuncDialect>();
  }

  void runOnOperation() override {
    auto func = getOperation();
    auto markerAttr = func->getAttrOfType<mlir::StringAttr>(kAutodiffMarker);
    if (!markerAttr || markerAttr.getValue() != "reverse")
      return;

    // Step 1: collect ops in forward program order.
    // NOTE: only top-level body ops are collected. Reverse-iterating a flat
    // walk that descended into nested regions (scf.for / scf.if) would
    // interleave parent/child adjoints out of structured order, so we restrict
    // to the function body's top level and reject nested control-flow regions
    // until structured reverse-mode lands.
    llvm::SmallVector<mlir::Operation *> forwardOps;
    for (mlir::Operation &opRef : func.getBody().front()) {
      mlir::Operation *op = &opRef;
      if (mlir::isa<mlir::func::ReturnOp>(op))
        continue;
      if (op->getNumRegions() != 0) {
        op->emitError() << "[AUTODIFF_NESTED_REGION] reverse-mode autodiff does "
                           "not yet support ops with nested regions ('"
                        << op->getName().getStringRef() << "')";
        signalPassFailure();
        return;
      }
      forwardOps.push_back(op);
    }

    // Step 2: identify scalar terminator (the loss seed) — convention is
    // that the function's single return is the seed for backward.
    auto returnOp = mlir::dyn_cast<mlir::func::ReturnOp>(
        func.getBody().front().getTerminator());
    if (!returnOp || returnOp.getNumOperands() != 1) {
      func.emitError() << "tessera-autodiff: function must have a single "
                          "scalar return (the loss seed).";
      return signalPassFailure();
    }
    mlir::Value lossValue = returnOp.getOperand(0);

    // Step 3: build cotangent map. Seed with cotangent=1.0 at the loss.
    mlir::OpBuilder builder(&getContext());
    builder.setInsertionPoint(returnOp);

    CotangentMap cotan;
    // Seed cotangent at the loss with all-ones matching the output type.
    // Equivalent to "loss = sum(output)" — gives unit cotangent at every
    // element of the loss tensor (or scalar). For arbitrary loss shapes
    // the user can wrap the call in a sum reduction at the Python boundary.
    mlir::Value seed;
    if (auto shapedType = mlir::dyn_cast<mlir::ShapedType>(lossValue.getType());
        shapedType.hasStaticShape()) {
      auto elemType = shapedType.getElementType();
      mlir::Attribute oneAttr;
      if (mlir::isa<mlir::FloatType>(elemType)) {
        oneAttr = mlir::FloatAttr::get(elemType, 1.0);
      } else {
        oneAttr = mlir::IntegerAttr::get(elemType, 1);
      }
      auto splatAttr = mlir::DenseElementsAttr::get(shapedType, oneAttr);
      seed = builder.create<mlir::arith::ConstantOp>(
          lossValue.getLoc(), splatAttr);
    } else if (!mlir::isa<mlir::ShapedType>(lossValue.getType())) {
      auto seedAttr = builder.getF32FloatAttr(1.0f);
      seed = builder.create<mlir::arith::ConstantOp>(
          lossValue.getLoc(), seedAttr);
    } else {
      // DenseElementsAttr cannot represent a dynamic-shaped splat. Use the
      // existing runtime-resolved custom-adjoint bridge to construct the
      // all-ones cotangent without claiming a static shape.
      auto seedOp = builder.create<CustomAdjointCallOp>(
          lossValue.getLoc(),
          llvm::SmallVector<mlir::Type>{lossValue.getType()},
          builder.getStringAttr("ones_like"), mlir::ValueRange{lossValue});
      seed = seedOp.getResult(0);
    }
    cotan[lossValue] = seed;

    // Step 4: reverse walk. Dispatch to `buildAdjoint` for each op that
    // implements AdjointInterface and lies on the gradient path.
    for (auto it = forwardOps.rbegin(); it != forwardOps.rend(); ++it) {
      mlir::Operation *op = *it;

      // Gather cotangents for this op's results.
      llvm::SmallVector<mlir::Value> outCotans;
      bool anyOutCotan = false;
      for (mlir::Value result : op->getResults()) {
        auto entry = cotan.lookup(result);
        outCotans.push_back(entry);
        if (entry) anyOutCotan = true;
      }
      if (!anyOutCotan)
        continue;  // Op is not on the gradient path.

      auto adjointOp = mlir::dyn_cast<AdjointInterface>(op);
      if (!adjointOp) {
        // Decision #21 (documented in the pass header but previously not
        // implemented): an op ON the gradient path that cannot propagate
        // cotangents must fail loudly — silently skipping it drops the
        // operand gradients and produces wrong results downstream.
        // Zero-operand ops (constants) terminate the chain naturally.
        if (op->getNumOperands() > 0) {
          op->emitError()
              << "[AUTODIFF_OP_NOT_DIFFERENTIABLE] op " << op->getName()
              << " is on the gradient path but does not implement "
                 "AdjointInterface; its operand cotangents would be "
                 "silently dropped";
          return signalPassFailure();
        }
        continue;
      }
      if (!adjointOp.isDifferentiable()) {
        op->emitError() << "tessera-autodiff: op " << op->getName()
                        << " declares AdjointInterface but isDifferentiable() "
                           "returned false";
        return signalPassFailure();
      }

      // Position the builder right before the return — keeps the seed (which
      // we inserted there) in scope for every adjoint, and avoids dominance
      // errors when later-walked ops produce cotangents consumed by
      // earlier-walked adjoints.
      builder.setInsertionPoint(returnOp);

      llvm::SmallVector<mlir::Value> inCotans =
          adjointOp.buildAdjoint(builder, outCotans);

      if (inCotans.size() != op->getNumOperands()) {
        op->emitError()
            << "tessera-autodiff: buildAdjoint returned "
            << inCotans.size() << " cotangents, expected "
            << op->getNumOperands() << " (one per operand)";
        return signalPassFailure();
      }

      // Accumulate into operand cotangent slots.
      for (auto [operand, g] :
           llvm::zip(op->getOperands(), inCotans)) {
        accumulateCotangent(builder, cotan, operand, g);
      }
    }

    // Step 5 — multi-output rewrite. Expose argument cotangents as
    // additional function outputs so downstream consumers (Phase F5
    // AdjointCollectiveInsertionPass; Python tape integration) have a
    // first-class SSA handle to each gradient.
    //
    // Rewrites the function:
    //   func @f(args...) -> (orig_outputs...) → @f(args...) -> (orig_outputs..., arg_cotans...)
    //
    // Args without a cotangent (not on the gradient path) are recorded as
    // empty entries in `tessera.autodiff.arg_cotangents` and skipped from
    // the output list — keeps the rewrite minimal and the IR readable.
    llvm::SmallVector<mlir::Value> argCotangentValues;
    llvm::SmallVector<mlir::Attribute> argCotanNames;
    for (auto arg : func.getArguments()) {
      auto entry = cotan.lookup(arg);
      if (!entry) {
        argCotanNames.push_back(builder.getStringAttr(""));
        continue;
      }
      argCotangentValues.push_back(entry);
      argCotanNames.push_back(builder.getStringAttr(
          "%cotan_arg_" + llvm::Twine(arg.getArgNumber()).str()));
    }
    func->setAttr("tessera.autodiff.arg_cotangents",
                   builder.getArrayAttr(argCotanNames));

    if (!argCotangentValues.empty()) {
      // Rewrite the return op to yield (original_results..., cotangents...).
      llvm::SmallVector<mlir::Value> newReturnOperands(returnOp.getOperands());
      for (auto cotanV : argCotangentValues) {
        newReturnOperands.push_back(cotanV);
      }
      builder.setInsertionPoint(returnOp);
      builder.create<mlir::func::ReturnOp>(returnOp.getLoc(), newReturnOperands);
      returnOp.erase();

      // Update the function's type signature to include the cotangent return types.
      llvm::SmallVector<mlir::Type> newResultTypes(
          func.getFunctionType().getResults().begin(),
          func.getFunctionType().getResults().end());
      for (auto cotanV : argCotangentValues) {
        newResultTypes.push_back(cotanV.getType());
      }
      auto newFnType = builder.getFunctionType(
          func.getFunctionType().getInputs(), newResultTypes);
      func.setType(newFnType);
    }
  }
};

}  // namespace

std::unique_ptr<mlir::Pass> createAutodiffPass() {
  return std::make_unique<AutodiffPass>();
}

}  // namespace tessera
