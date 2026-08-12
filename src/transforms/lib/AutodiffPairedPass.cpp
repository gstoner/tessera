//===- AutodiffPairedPass.cpp - Paired forward/backward autodiff --*- C++ -*-===//
//
// Phase 2 of docs/audit/compiler/AUTODIFF_UNIFICATION_PLAN.md. Where the
// in-place `--tessera-autodiff` pass fuses the backward into the forward
// function's return (a bootstrap), this pass emits the **paired-program model**:
//
//   forward(inputs)                       -> primals            (unchanged)
//   @f__bwd(inputs, out_cotangents...)    -> input_cotangents   (new function)
//
// This is the deterministic forward/backward/residual ABI the rest of the plan
// (runtime binding in Phase 4, per-op-family expansion in Phase 5, distributed +
// accelerator promotion in Phase 6) keys off. It is verifiable independently of
// Python tape state — a lit fixture checks the backward signature + body.
//
// Residual policy — RECOMPUTE_ALL (first cut). The backward function takes the
// forward *inputs* as arguments and recomputes any forward intermediates it
// needs by cloning the forward ops into the backward body (CSE later collapses
// redundant recompute). This is not a toy choice: the shipped ROCm gfx1151
// flash-attention backward lane (`_execute_rocm_compiled_flash_attn_bwd`) takes
// `(dO, Q, K, V)` and likewise *recomputes* the softmax rather than saving the
// logsumexp. A future SAVE policy (return selected forward values as explicit
// residual outputs of the forward, e.g. flash-attn's `L`) is an optimization the
// same ABI already accommodates via `tessera.autodiff.residual_policy`.
//
// The paired backward is an **ABI, not an implementation**: a hand-emitted
// backward kernel (ROCm WMMA flash-attn bwd) satisfies the same
// `@f__bwd(inputs, out_cotangents) -> input_cotangents` contract and is a
// first-class arbiter candidate (Decision #28). This pass is the compiler-
// generated implementation of that contract.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"

#include "Tessera/AdjointInterface.h.inc"
#include "Tessera/LinearTransposeInterface.h.inc"
#include "Tessera/Transforms/GraphDataflow.h"
#include "Tessera/Transforms/RegionAdjointInterface.h"
#include "Tessera/Transforms/SemanticEffects.h"

namespace tessera {

namespace {

constexpr const char *kAutodiffMarker = "tessera.autodiff";

using CotangentMap = llvm::DenseMap<mlir::Value, mlir::Value>;
bool hasStochasticEffect(mlir::Operation *op) {
  return getRegisteredSemanticEffect(op) == SemanticEffectLevel::Random;
}

void eraseStopGradientBarriers(mlir::func::FuncOp func) {
  llvm::SmallVector<mlir::Operation *> barriers;
  func.walk([&](mlir::Operation *op) {
    if (op->getName().getStringRef() == "tessera.stop_gradient")
      barriers.push_back(op);
  });
  for (mlir::Operation *op : barriers) {
    if (op->getNumOperands() == 1 && op->getNumResults() == 1) {
      op->getResult(0).replaceAllUsesWith(op->getOperand(0));
      op->erase();
    }
  }
}

/// Accumulate `g` into `cotan[v]` (float → addf, integer → addi). Shared shape
/// with AutodiffPass.cpp; kept local so the two passes stay independent.
void accumulateCotangent(mlir::OpBuilder &builder, CotangentMap &cotan,
                         mlir::Value v, mlir::Value g) {
  if (!g)
    return;
  auto it = cotan.find(v);
  if (it == cotan.end()) {
    cotan[v] = g;
    return;
  }
  auto loc = g.getLoc();
  mlir::Type elemTy = mlir::getElementTypeOrSelf(g.getType());
  mlir::Value sum =
      llvm::isa<mlir::FloatType>(elemTy)
          ? builder.create<mlir::arith::AddFOp>(loc, it->second, g).getResult()
          : builder.create<mlir::arith::AddIOp>(loc, it->second, g).getResult();
  cotan[v] = sum;
}

class AutodiffPairedPass
    : public mlir::PassWrapper<AutodiffPairedPass,
                               mlir::OperationPass<mlir::ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(AutodiffPairedPass)

  llvm::StringRef getArgument() const final {
    return "tessera-autodiff-paired";
  }
  llvm::StringRef getDescription() const final {
    return "Paired forward/backward autodiff — emits @f__bwd(inputs, "
           "out_cotangents) -> input_cotangents (recompute-all residual "
           "policy). Phase 2 of AUTODIFF_UNIFICATION_PLAN.md.";
  }
  void getDependentDialects(mlir::DialectRegistry &registry) const override {
    registry.insert<mlir::arith::ArithDialect, mlir::func::FuncDialect,
                    mlir::scf::SCFDialect, mlir::tensor::TensorDialect>();
  }

  void runOnOperation() override {
    auto module = getOperation();
    llvm::SmallVector<mlir::func::FuncOp> targets;
    module.walk([&](mlir::func::FuncOp fn) {
      auto marker = fn->getAttrOfType<mlir::StringAttr>(kAutodiffMarker);
      // Skip functions we already produced (role=backward) or that aren't marked.
      if (marker && marker.getValue() == "reverse" &&
          !fn->hasAttr("tessera.autodiff.role"))
        targets.push_back(fn);
    });
    for (auto fn : targets)
      if (failed(buildBackward(fn)))
        return signalPassFailure();
  }

private:
  mlir::LogicalResult buildBackward(mlir::func::FuncOp fwd) {
    auto module = fwd->getParentOfType<mlir::ModuleOp>();
    mlir::MLIRContext *ctx = &getContext();

    if (fwd.getBody().empty()) {
      fwd.emitError() << "[AUTODIFF_PAIRED] cannot differentiate a declaration";
      return mlir::failure();
    }
    mlir::Block &fwdBlock = fwd.getBody().front();

    auto returnOp = mlir::dyn_cast<mlir::func::ReturnOp>(fwdBlock.getTerminator());
    if (!returnOp) {
      fwd.emitError() << "[AUTODIFF_PAIRED] forward has no return terminator";
      return mlir::failure();
    }
    GraphDataflowAnalysis dataflow(fwd);
    if (mlir::failed(dataflow.run())) {
      fwd.emitError()
          << "[AUTODIFF_PAIRED] Graph IR dataflow analysis failed";
      return mlir::failure();
    }
    GraphDataflowAnalysis::ActiveOpSet activeOps =
        dataflow.computeActivity(returnOp.getOperands());

    // Collect only the backward-reachable forward cone. Inactive side
    // computations and inactive nested regions are neither cloned nor rejected.
    llvm::SmallVector<mlir::Operation *> forwardOps;
    for (mlir::Operation &opRef : fwdBlock) {
      mlir::Operation *op = &opRef;
      if (mlir::isa<mlir::func::ReturnOp>(op))
        continue;
      op->setAttr("tessera.autodiff.activity",
                  mlir::StringAttr::get(ctx, activeOps.contains(op)
                                                ? "active"
                                                : "inactive"));
      if (!activeOps.contains(op))
        continue;
      if (op->getNumRegions() != 0 &&
          !RegionAdjointInterface::supports(op)) {
        op->emitError() << "[AUTODIFF_NESTED_REGION] active paired reverse-mode "
                           "path contains unsupported nested-region op ('"
                        << op->getName().getStringRef() << "')";
        return mlir::failure();
      }
      if (hasStochasticEffect(op)) {
        op->emitError()
            << "AUTODIFF_STOCHASTIC_EFFECT: active stochastic op "
            << op->getName()
            << " requires an explicit pathwise or score-function adjoint";
        return mlir::failure();
      }
      forwardOps.push_back(op);
    }

    // Backward signature: (forward inputs..., out_cotangents...) ->
    // (input_cotangents...). One out-cotangent per forward result; the input
    // cotangent types mirror the forward argument types.
    llvm::SmallVector<mlir::Type> fwdInTypes(fwd.getArgumentTypes().begin(),
                                             fwd.getArgumentTypes().end());
    llvm::SmallVector<mlir::Type> fwdResTypes(
        fwd.getResultTypes().begin(), fwd.getResultTypes().end());

    llvm::SmallVector<mlir::Type> bwdInTypes(fwdInTypes);
    for (mlir::Type rt : fwdResTypes)
      bwdInTypes.push_back(rt);
    // Input cotangents mirror the input types (one per forward argument).
    llvm::SmallVector<mlir::Type> bwdResTypes(fwdInTypes);

    mlir::OpBuilder builder(ctx);
    builder.setInsertionPointToEnd(module.getBody());
    auto bwdName = (fwd.getName() + "__bwd").str();
    auto bwdType = builder.getFunctionType(bwdInTypes, bwdResTypes);
    auto bwd = builder.create<mlir::func::FuncOp>(fwd.getLoc(), bwdName, bwdType);
    bwd->setAttr("tessera.autodiff.role", builder.getStringAttr("backward"));
    bwd->setAttr("tessera.autodiff.forward",
                 mlir::FlatSymbolRefAttr::get(ctx, fwd.getName()));
    bwd->setAttr("tessera.autodiff.residual_policy",
                 builder.getStringAttr("recompute_all"));

    mlir::Block *bwdBlock = bwd.addEntryBlock();
    builder.setInsertionPointToStart(bwdBlock);

    unsigned nIn = fwd.getNumArguments();
    unsigned nRes = fwdResTypes.size();

    // Map forward SSA values into the backward function: forward argument i →
    // backward argument i (the recompute-all residual = the forward inputs).
    mlir::IRMapping map;
    for (unsigned i = 0; i < nIn; ++i)
      map.map(fwd.getArgument(i), bwdBlock->getArgument(i));

    // Recompute the forward ops inside the backward body (clones), so each
    // adjoint's `getX()` resolves to a value that lives in this function.
    llvm::SmallVector<mlir::Operation *> clones;
    for (mlir::Operation *op : forwardOps) {
      // Recompute-all can preserve a stopped primal only when its operand is
      // already a backward argument.  Recomputing an inactive producer cone
      // here would be wrong for stateful or stochastic producers; a future
      // saved-residual policy can lift this restriction explicitly.
      if (op->getName().getStringRef() == "tessera.stop_gradient" &&
          !map.contains(op->getOperand(0))) {
        op->emitError()
            << "AUTODIFF_STOP_GRADIENT_RESIDUAL_REQUIRED: paired "
               "recompute-all cannot preserve a stopped intermediate; save "
               "the stopped primal as an explicit residual";
        return mlir::failure();
      }
      mlir::Operation *clone = builder.clone(*op, map);
      clones.push_back(clone);
    }

    // Seed cotangents: forward result j ↦ backward out-cotangent argument
    // (nIn + j). The clone of the op producing that result carries the seed.
    CotangentMap cotan;
    for (unsigned j = 0; j < nRes; ++j) {
      mlir::Value fwdRes = returnOp.getOperand(j);
      mlir::Value cloneRes = map.lookupOrNull(fwdRes);
      if (!cloneRes) {
        // A forward result that is itself an argument (identity return): route
        // the cotangent straight to that input slot.
        if (auto ba = llvm::dyn_cast<mlir::BlockArgument>(fwdRes))
          accumulateCotangent(builder, cotan, bwdBlock->getArgument(ba.getArgNumber()),
                              bwdBlock->getArgument(nIn + j));
        continue;
      }
      accumulateCotangent(builder, cotan, cloneRes, bwdBlock->getArgument(nIn + j));
    }

    // Reverse walk the clones.
    for (auto it = clones.rbegin(); it != clones.rend(); ++it) {
      mlir::Operation *op = *it;
      llvm::SmallVector<mlir::Value> outCotans;
      bool any = false;
      for (mlir::Value r : op->getResults()) {
        mlir::Value c = cotan.lookup(r);
        outCotans.push_back(c);
        if (c)
          any = true;
      }
      if (!any)
        continue;

      if (failed(differentiateOperation(op, outCotans, builder, cotan)))
        return mlir::failure();
    }

    // Return input cotangents (zero-splat for inputs off the gradient path so
    // the signature is total and the buffer binding in Phase 4 is uniform).
    llvm::SmallVector<mlir::Value> results;
    for (unsigned i = 0; i < nIn; ++i) {
      mlir::Value g = cotan.lookup(bwdBlock->getArgument(i));
      if (!g) {
        mlir::Type ty = fwdInTypes[i];
        mlir::Value zero;
        if (auto shaped = llvm::dyn_cast<mlir::ShapedType>(ty)) {
          auto elem = shaped.getElementType();
          mlir::Attribute z = llvm::isa<mlir::FloatType>(elem)
                                  ? (mlir::Attribute)mlir::FloatAttr::get(elem, 0.0)
                                  : (mlir::Attribute)mlir::IntegerAttr::get(elem, 0);
          zero = builder.create<mlir::arith::ConstantOp>(
              fwd.getLoc(), mlir::DenseElementsAttr::get(shaped, z));
        } else {
          zero = builder.create<mlir::arith::ConstantOp>(
              fwd.getLoc(), builder.getZeroAttr(ty));
        }
        g = zero;
      }
      results.push_back(g);
    }
    builder.create<mlir::func::ReturnOp>(fwd.getLoc(), results);

    // Link the forward to its paired backward (residuals empty under
    // recompute-all — the forward stays primals-only).
    fwd->setAttr("tessera.autodiff.paired",
                 mlir::FlatSymbolRefAttr::get(ctx, bwdName));
    fwd->setAttr("tessera.autodiff.residual_policy",
                 builder.getStringAttr("recompute_all"));
    eraseStopGradientBarriers(bwd);
    eraseStopGradientBarriers(fwd);
    return mlir::success();
  }

  mlir::Value buildZeroLike(mlir::OpBuilder &builder, mlir::Value primal) {
    mlir::Type type = primal.getType();
    mlir::Type elementType = mlir::getElementTypeOrSelf(type);
    mlir::TypedAttr zero = llvm::isa<mlir::FloatType>(elementType)
                               ? mlir::TypedAttr(
                                     mlir::FloatAttr::get(elementType, 0.0))
                               : mlir::TypedAttr(
                                     mlir::IntegerAttr::get(elementType, 0));
    if (auto shaped = llvm::dyn_cast<mlir::ShapedType>(type);
        shaped && shaped.hasStaticShape()) {
      return builder
          .create<mlir::arith::ConstantOp>(
              primal.getLoc(), mlir::DenseElementsAttr::get(shaped, zero))
          .getResult();
    }
    if (!llvm::isa<mlir::ShapedType>(type))
      return builder
          .create<mlir::arith::ConstantOp>(primal.getLoc(), zero)
          .getResult();

    mlir::OperationState state(primal.getLoc(),
                               "tessera.custom_adjoint_call");
    state.addOperands(primal);
    state.addTypes(type);
    state.addAttribute("name", builder.getStringAttr("zeros_like"));
    return builder.create(state)->getResult(0);
  }

  mlir::LogicalResult buildRegionPullback(
      mlir::Region &region, mlir::ValueRange outputCotangents,
      mlir::ValueRange blockArgumentValues,
      llvm::ArrayRef<mlir::Value> captures, mlir::OpBuilder &builder,
      llvm::SmallVectorImpl<mlir::Value> &blockArgumentCotangents,
      llvm::SmallVectorImpl<mlir::Value> &captureCotangents) {
    if (!region.hasOneBlock() ||
        region.front().getNumArguments() != blockArgumentValues.size())
      return mlir::failure();
    auto yield = mlir::dyn_cast<mlir::scf::YieldOp>(
        region.front().getTerminator());
    if (!yield || yield.getNumOperands() != outputCotangents.size())
      return mlir::failure();

    mlir::IRMapping mapping;
    for (auto [argument, value] : llvm::zip_equal(
             region.front().getArguments(), blockArgumentValues))
      mapping.map(argument, value);
    for (mlir::Value capture : captures)
      mapping.map(capture, capture);
    llvm::SmallVector<mlir::Operation *> clones;
    for (mlir::Operation &source : region.front().without_terminator())
      clones.push_back(builder.clone(source, mapping));

    CotangentMap cotangents;
    for (auto [yielded, seed] :
         llvm::zip_equal(yield.getOperands(), outputCotangents))
      if (seed)
        accumulateCotangent(builder, cotangents,
                            mapping.lookupOrDefault(yielded), seed);

    for (mlir::Operation *clone : llvm::reverse(clones)) {
      llvm::SmallVector<mlir::Value> resultCotangents;
      bool active = false;
      for (mlir::Value result : clone->getResults()) {
        mlir::Value cotangent = cotangents.lookup(result);
        resultCotangents.push_back(cotangent);
        active |= static_cast<bool>(cotangent);
      }
      if (active && failed(differentiateOperation(
                        clone, resultCotangents, builder, cotangents)))
        return mlir::failure();
    }

    for (auto [argument, value] : llvm::zip_equal(
             region.front().getArguments(), blockArgumentValues)) {
      if (!llvm::isa<mlir::FloatType>(
              mlir::getElementTypeOrSelf(argument.getType()))) {
        blockArgumentCotangents.push_back({});
        continue;
      }
      mlir::Value cotangent = cotangents.lookup(value);
      blockArgumentCotangents.push_back(
          cotangent ? cotangent : buildZeroLike(builder, value));
    }
    for (mlir::Value capture : captures) {
      mlir::Value cotangent = cotangents.lookup(capture);
      captureCotangents.push_back(cotangent ? cotangent
                                            : buildZeroLike(builder, capture));
    }
    return mlir::success();
  }

  mlir::LogicalResult differentiateOperation(
      mlir::Operation *op, mlir::ValueRange outputCotangents,
      mlir::OpBuilder &builder, CotangentMap &cotangents) {
    if (op->getNumRegions() != 0) {
      if (!RegionAdjointInterface::supports(op)) {
        op->emitError() << "[AUTODIFF_NESTED_REGION] no registered "
                           "RegionAdjointInterface model for "
                        << op->getName();
        return mlir::failure();
      }
      llvm::SmallVector<RegionCotangent> regionCotangents;
      auto callback = [&](mlir::Region &region, mlir::ValueRange seeds,
                          mlir::ValueRange blockArgumentValues,
                          llvm::ArrayRef<mlir::Value> captures,
                          mlir::OpBuilder &nestedBuilder,
                          llvm::SmallVectorImpl<mlir::Value> &blockResults,
                          llvm::SmallVectorImpl<mlir::Value> &results) {
        return buildRegionPullback(region, seeds, blockArgumentValues,
                                   captures, nestedBuilder, blockResults,
                                   results);
      };
      if (failed(RegionAdjointInterface::buildAdjoint(
              op, builder, outputCotangents, callback, regionCotangents))) {
        op->emitError() << "[AUTODIFF_REGION_ADJOINT] structured pullback "
                           "construction failed";
        return mlir::failure();
      }
      for (const RegionCotangent &entry : regionCotangents)
        accumulateCotangent(builder, cotangents, entry.primal,
                            entry.cotangent);
      return mlir::success();
    }

    // These tensor slice operations are introduced by control_scan lowering.
    // Keep their linear transpose here instead of teaching the Tessera
    // operation interface about foreign-dialect operations.  extract_slice^T
    // scatters into a zero source; insert_slice^T gathers the overwritten
    // source region and masks that region out of the destination cotangent.
    if (auto extract = mlir::dyn_cast<mlir::tensor::ExtractSliceOp>(op)) {
      if (outputCotangents.size() != 1 || !outputCotangents.front())
        return mlir::failure();
      mlir::Value sourceZero = buildZeroLike(builder, extract.getSource());
      mlir::Value sourceCotangent =
          mlir::tensor::InsertSliceOp::create(
              builder, op->getLoc(), outputCotangents.front(), sourceZero,
              extract.getMixedOffsets(), extract.getMixedSizes(),
              extract.getMixedStrides());
      accumulateCotangent(builder, cotangents, extract.getSource(),
                          sourceCotangent);
      return mlir::success();
    }
    if (auto insert = mlir::dyn_cast<mlir::tensor::InsertSliceOp>(op)) {
      if (outputCotangents.size() != 1 || !outputCotangents.front())
        return mlir::failure();
      mlir::Value sourceCotangent =
          mlir::tensor::ExtractSliceOp::create(
              builder, op->getLoc(), insert.getSourceType(),
              outputCotangents.front(), insert.getMixedOffsets(),
              insert.getMixedSizes(), insert.getMixedStrides());
      mlir::Value sourceZero = buildZeroLike(builder, insert.getSource());
      mlir::Value destinationCotangent =
          mlir::tensor::InsertSliceOp::create(
              builder, op->getLoc(), sourceZero, outputCotangents.front(),
              insert.getMixedOffsets(), insert.getMixedSizes(),
              insert.getMixedStrides());
      accumulateCotangent(builder, cotangents, insert.getSource(),
                          sourceCotangent);
      accumulateCotangent(builder, cotangents, insert.getDest(),
                          destinationCotangent);
      return mlir::success();
    }

    llvm::SmallVector<mlir::Value> inputCotangents;
    if (auto adjoint = mlir::dyn_cast<AdjointInterface>(op)) {
      if (!adjoint.isDifferentiable()) {
        op->emitError() << "[AUTODIFF_PAIRED] op " << op->getName()
                        << " declares AdjointInterface but isDifferentiable() "
                           "is false";
        return mlir::failure();
      }
      inputCotangents = adjoint.buildAdjoint(builder, outputCotangents);
    } else if (auto linear =
                   mlir::dyn_cast<LinearTransposeInterface>(op)) {
      inputCotangents =
          linear.buildLinearTranspose(builder, outputCotangents);
      for (auto [index, cotangent] : llvm::enumerate(inputCotangents)) {
        if (cotangent && !linear.isLinearInOperand(index)) {
          op->emitError()
              << "[AUTODIFF_PAIRED] LinearTransposeInterface produced a "
                 "cotangent for non-linear operand "
              << index;
          return mlir::failure();
        }
      }
    } else {
      if (op->getNumOperands() > 0) {
        op->emitError() << "[AUTODIFF_OP_NOT_DIFFERENTIABLE] op "
                        << op->getName()
                        << " is on the gradient path but implements neither "
                           "AdjointInterface nor LinearTransposeInterface";
        return mlir::failure();
      }
      return mlir::success();
    }
    if (inputCotangents.size() != op->getNumOperands()) {
      op->emitError() << "[AUTODIFF_PAIRED] derivative interface returned "
                      << inputCotangents.size() << " cotangents, expected "
                      << op->getNumOperands();
      return mlir::failure();
    }
    for (auto [operand, cotangent] :
         llvm::zip_equal(op->getOperands(), inputCotangents))
      accumulateCotangent(builder, cotangents, operand, cotangent);
    return mlir::success();
  }
};

}  // namespace

std::unique_ptr<mlir::Pass> createAutodiffPairedPass() {
  return std::make_unique<AutodiffPairedPass>();
}

}  // namespace tessera
