//===- AdjointInterface.cpp - buildAdjoint impls per op --------*- C++ -*-===//
//
// Phase F4 of docs/audit/roadmap/ROADMAP_AUDIT.md. The generated header
// `Tessera/AdjointInterface.cpp.inc` (from `AdjointInterface.td` via
// `TesseraAdjointInterfaceTableGen`) provides the interface plumbing; this
// file implements `buildAdjoint` for each op that opts in via
// `DeclareOpInterfaceMethods<Tessera_AdjointInterface>` in TesseraOps.td.
//
// Each impl mirrors the Python VJP in `python/tessera/autodiff/vjp.py` —
// the IR-level pass and the numpy tape share semantics so any op covered
// here will produce identical gradients regardless of which path executed.
//
// Every Tessera matmul / unary op that declares the interface in
// `TesseraOps.td` has its `buildAdjoint` here; arith.* ops on the
// cotangent path are handled by the AutodiffPass via fresh `arith.addf`
// emission (see `accumulateCotangent` in `AutodiffPass.cpp`).
//
//===----------------------------------------------------------------------===//

#include "Tessera/IR/TesseraOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "llvm/ADT/SmallVector.h"

// The interface impl mixin — generated from AdjointInterface.td.
namespace tessera {
#include "Tessera/AdjointInterface.cpp.inc"
}  // namespace tessera (re-opened below for the buildAdjoint defs)

namespace tessera {

// ─────────────────────────────────────────────────────────────────────────────
// F5 collectives. These are the IR counterparts of the existing DDP/FSDP VJPs:
// all-reduce(sum) is self-dual; all-gather and reduce-scatter are transposes.
// The paired pass only wires the existing collective semantics — it does not
// invent placement or communication insertion policy.
// ─────────────────────────────────────────────────────────────────────────────

llvm::SmallVector<mlir::Value> AllReduceOp::buildAdjoint(
    mlir::OpBuilder &builder, mlir::ValueRange outputCotangents) {
  if (outputCotangents.size() != 1 || !outputCotangents[0])
    return {mlir::Value()};
  auto grad = builder.create<AllReduceOp>(
      getLoc(), getX().getType(), outputCotangents[0], getAxisAttr(), getOpAttr());
  return {grad.getY()};
}

llvm::SmallVector<mlir::Value> AllGatherOp::buildAdjoint(
    mlir::OpBuilder &builder, mlir::ValueRange outputCotangents) {
  if (outputCotangents.size() != 1 || !outputCotangents[0])
    return {mlir::Value()};
  auto grad = builder.create<ReduceScatterOp>(
      getLoc(), getX().getType(), outputCotangents[0], getAxisAttr(),
      builder.getStringAttr("sum"));
  return {grad.getY()};
}

llvm::SmallVector<mlir::Value> ReduceScatterOp::buildAdjoint(
    mlir::OpBuilder &builder, mlir::ValueRange outputCotangents) {
  if (outputCotangents.size() != 1 || !outputCotangents[0])
    return {mlir::Value()};
  auto grad = builder.create<AllGatherOp>(
      getLoc(), getX().getType(), outputCotangents[0], getAxisAttr(), getOpAttr());
  return {grad.getY()};
}

// ─────────────────────────────────────────────────────────────────────────────
// MatmulOp
//
// Forward:  C = A @ B
// Adjoints: dA = dout @ B^T
//           dB = A^T @ dout
//
// Mirrors python/tessera/autodiff/vjp.py::vjp_gemm.
// ─────────────────────────────────────────────────────────────────────────────

llvm::SmallVector<mlir::Value> MatmulOp::buildAdjoint(
    mlir::OpBuilder &builder, mlir::ValueRange outputCotangents) {
  if (outputCotangents.size() != 1 || !outputCotangents[0])
    return {mlir::Value(), mlir::Value()};
  mlir::Value dOut = outputCotangents[0];
  auto loc = getLoc();

  // dA = dout @ B^T
  auto dA = builder.create<MatmulOp>(
      loc, getLhs().getType(), dOut, getRhs(),
      /*tile_k=*/mlir::IntegerAttr(),
      /*numeric_policy=*/nullptr,
      /*transposeA=*/builder.getBoolAttr(false),
      /*transposeB=*/builder.getBoolAttr(true));

  // dB = A^T @ dout
  auto dB = builder.create<MatmulOp>(
      loc, getRhs().getType(), getLhs(), dOut,
      /*tile_k=*/mlir::IntegerAttr(),
      /*numeric_policy=*/nullptr,
      /*transposeA=*/builder.getBoolAttr(true),
      /*transposeB=*/builder.getBoolAttr(false));

  return {dA.getResult(), dB.getResult()};
}

static llvm::SmallVector<mlir::Value>
placeholderAdjoint(mlir::OpBuilder &builder, mlir::Location loc, mlir::Type ty,
                   llvm::StringRef key, mlir::Value dy, mlir::Value x);
static mlir::Value splatConst(mlir::OpBuilder &builder, mlir::Location loc,
                              mlir::Type ty, double value);

static mlir::RankedTensorType
lastAxisReducedType(mlir::RankedTensorType inputType) {
  llvm::SmallVector<int64_t> shape(inputType.getShape());
  shape.pop_back();
  return mlir::RankedTensorType::get(shape, inputType.getElementType(),
                                     inputType.getEncoding());
}

static std::pair<mlir::Value, mlir::Value> buildNormalizationStats(
    mlir::OpBuilder &builder, mlir::Location loc, mlir::Value x,
    mlir::RankedTensorType inputType, mlir::FloatAttr eps, bool centered) {
  mlir::RankedTensorType statsType = lastAxisReducedType(inputType);
  mlir::OperationState state(loc, "tessera.normalization_stats");
  state.addOperands(x);
  state.addTypes({statsType, statsType});
  state.addAttribute("axis", builder.getI64IntegerAttr(-1));
  state.addAttribute("eps", eps ? eps : builder.getF64FloatAttr(1.0e-5));
  state.addAttribute("centered", builder.getBoolAttr(centered));
  mlir::Operation *stats = builder.create(state);
  return {stats->getResult(0), stats->getResult(1)};
}

static mlir::Value broadcastLastAxisStat(mlir::OpBuilder &builder,
                                         mlir::Location loc,
                                         mlir::RankedTensorType inputType,
                                         mlir::Value stat,
                                         mlir::Value shapeLike) {
  llvm::SmallVector<mlir::Attribute> dimensions;
  dimensions.reserve(inputType.getRank() - 1);
  for (int64_t dim = 0; dim + 1 < inputType.getRank(); ++dim)
    dimensions.push_back(builder.getI64IntegerAttr(dim));
  return builder
      .create<BroadcastInDimOp>(loc, inputType, stat, shapeLike,
                                builder.getArrayAttr(dimensions))
      .getY();
}

static mlir::Value reduceMeanLastAxis(mlir::OpBuilder &builder,
                                      mlir::Location loc,
                                      mlir::RankedTensorType inputType,
                                      mlir::Value value) {
  return builder
      .create<ReduceOp>(loc, lastAxisReducedType(inputType), value,
                        builder.getStringAttr("mean"),
                        builder.getI64IntegerAttr(inputType.getRank() - 1))
      .getResult();
}

static mlir::Value broadcastChannelVector(mlir::OpBuilder &builder,
                                          mlir::Location loc,
                                          mlir::RankedTensorType inputType,
                                          mlir::Value channel,
                                          mlir::Value shapeLike) {
  auto mapping = builder.getArrayAttr(
      {builder.getI64IntegerAttr(inputType.getRank() - 1)});
  return builder
      .create<BroadcastInDimOp>(loc, inputType, channel, shapeLike, mapping)
      .getY();
}

// Reduce every leading dimension, leaving the final channel dimension. Since
// each reduction removes axis zero, repeating axis zero is valid for all ranks.
static mlir::Value reduceLeadingSum(mlir::OpBuilder &builder,
                                    mlir::Location loc,
                                    mlir::RankedTensorType inputType,
                                    mlir::Value value) {
  mlir::RankedTensorType currentType = inputType;
  for (int64_t remaining = inputType.getRank() - 1; remaining > 0; --remaining) {
    llvm::SmallVector<int64_t> shape(currentType.getShape());
    shape.erase(shape.begin());
    auto reducedType = mlir::RankedTensorType::get(
        shape, currentType.getElementType(), currentType.getEncoding());
    value = builder
                .create<ReduceOp>(loc, reducedType, value,
                                  builder.getStringAttr("sum"),
                                  builder.getI64IntegerAttr(0))
                .getResult();
    currentType = reducedType;
  }
  return value;
}

// These tables are both implementation policy and the generated ledger's
// source for kind-aware `tessera.reduce` classification. Keep their spelling
// stable unless autodiff_ledger.py is updated in the same change.
static constexpr llvm::StringLiteral kReduceNativeAdjointKinds[] = {
    "sum", "mean", "max", "min"};
static constexpr llvm::StringLiteral kReducePlaceholderAdjointKinds[] = {};

static bool containsKind(llvm::ArrayRef<llvm::StringLiteral> kinds,
                         llvm::StringRef kind) {
  for (llvm::StringLiteral candidate : kinds)
    if (candidate == kind)
      return true;
  return false;
}

// Elementwise tensor algebra. Add is self-linear; multiply applies the product
// rule. These formulas contain no shape-dependent constants, so they are valid
// for both static and dynamic same-shape tensors.
llvm::SmallVector<mlir::Value> AddOp::buildAdjoint(
    mlir::OpBuilder &builder, mlir::ValueRange outputCotangents) {
  if (outputCotangents.size() != 1 || !outputCotangents[0])
    return {mlir::Value(), mlir::Value()};
  return {outputCotangents[0], outputCotangents[0]};
}

llvm::SmallVector<mlir::Value> MulOp::buildAdjoint(
    mlir::OpBuilder &builder, mlir::ValueRange outputCotangents) {
  if (outputCotangents.size() != 1 || !outputCotangents[0])
    return {mlir::Value(), mlir::Value()};
  mlir::Value dy = outputCotangents[0];
  auto dLhs = builder.create<MulOp>(getLoc(), getLhs().getType(), dy, getRhs());
  auto dRhs = builder.create<MulOp>(getLoc(), getRhs().getType(), dy, getLhs());
  return {dLhs.getResult(), dRhs.getResult()};
}

// MSE keeps reduction semantics in a dedicated compiler carrier so dynamic
// element counts remain runtime values instead of becoming dense constants.
// d(pred) = 2 * (pred - target) * dy / N for mean reduction; d(target) is the
// exact negative. The shared Linalg lowering owns N and scalar broadcasting.
llvm::SmallVector<mlir::Value> MSELossOp::buildAdjoint(
    mlir::OpBuilder &builder, mlir::ValueRange outputCotangents) {
  if (outputCotangents.size() != 1 || !outputCotangents[0])
    return {mlir::Value(), mlir::Value()};
  mlir::OperationState state(getLoc(), "tessera.loss.mse_backward");
  state.addOperands({getPrediction(), getTarget(), outputCotangents[0]});
  state.addTypes({getPrediction().getType(), getTarget().getType()});
  state.addAttribute("reduction", getReductionAttr());
  mlir::Operation *backward = builder.create(state);
  return {backward->getResult(0), backward->getResult(1)};
}

llvm::SmallVector<mlir::Value> BinaryCrossEntropyLossOp::buildAdjoint(
    mlir::OpBuilder &builder, mlir::ValueRange outputCotangents) {
  if (outputCotangents.size() != 1 || !outputCotangents[0])
    return {mlir::Value(), mlir::Value()};
  mlir::OperationState state(
      getLoc(), "tessera.loss.binary_cross_entropy_backward");
  state.addOperands({getLogits(), getTarget(), outputCotangents[0]});
  state.addTypes({getLogits().getType(), getTarget().getType()});
  state.addAttribute("reduction", getReductionAttr());
  mlir::Operation *backward = builder.create(state);
  return {backward->getResult(0), backward->getResult(1)};
}

llvm::SmallVector<mlir::Value> CrossEntropyLossOp::buildAdjoint(
    mlir::OpBuilder &builder, mlir::ValueRange outputCotangents) {
  if (outputCotangents.size() != 1 || !outputCotangents[0])
    return {mlir::Value(), mlir::Value()};
  mlir::OperationState state(getLoc(), "tessera.loss.cross_entropy_backward");
  state.addOperands({getLogits(), getTarget(), outputCotangents[0]});
  state.addTypes(getLogits().getType());
  state.addAttribute("axis", getAxisAttr());
  state.addAttribute("ignore_index", getIgnoreIndexAttr());
  state.addAttribute("label_smoothing", getLabelSmoothingAttr());
  state.addAttribute("reduction", getReductionAttr());
  mlir::Operation *backward = builder.create(state);
  return {backward->getResult(0), mlir::Value()};
}

static llvm::SmallVector<mlir::Value> buildDistributionLossAdjoint(
    mlir::Operation *op, mlir::OpBuilder &builder,
    mlir::ValueRange outputCotangents, llvm::StringRef kind,
    mlir::IntegerAttr axis, mlir::FloatAttr epsilon,
    mlir::StringAttr reduction) {
  if (outputCotangents.size() != 1 || !outputCotangents[0])
    return {mlir::Value(), mlir::Value()};
  mlir::OperationState state(op->getLoc(),
                             "tessera.loss.distribution_backward");
  state.addOperands(
      {op->getOperand(0), op->getOperand(1), outputCotangents[0]});
  state.addTypes({op->getOperand(0).getType(), op->getOperand(1).getType()});
  state.addAttribute("kind", builder.getStringAttr(kind));
  state.addAttribute("axis", axis);
  state.addAttribute("epsilon", epsilon);
  state.addAttribute("reduction", reduction);
  mlir::Operation *backward = builder.create(state);
  return {backward->getResult(0), backward->getResult(1)};
}

llvm::SmallVector<mlir::Value> KLDivergenceLossOp::buildAdjoint(
    mlir::OpBuilder &builder, mlir::ValueRange outputCotangents) {
  return buildDistributionLossAdjoint(
      getOperation(), builder, outputCotangents, "kl", getAxisAttr(),
      getEpsilonAttr(), getReductionAttr());
}

llvm::SmallVector<mlir::Value> JSDivergenceLossOp::buildAdjoint(
    mlir::OpBuilder &builder, mlir::ValueRange outputCotangents) {
  return buildDistributionLossAdjoint(
      getOperation(), builder, outputCotangents, "js", getAxisAttr(),
      getEpsilonAttr(), getReductionAttr());
}

static llvm::SmallVector<mlir::Value> buildRegressionLossAdjoint(
    mlir::Operation *op, mlir::OpBuilder &builder,
    mlir::ValueRange outputCotangents, llvm::StringRef kind,
    mlir::FloatAttr parameter, mlir::StringAttr reduction) {
  if (outputCotangents.size() != 1 || !outputCotangents[0])
    return {mlir::Value(), mlir::Value()};
  mlir::OperationState state(op->getLoc(), "tessera.loss.regression_backward");
  state.addOperands(
      {op->getOperand(0), op->getOperand(1), outputCotangents[0]});
  state.addTypes({op->getOperand(0).getType(), op->getOperand(1).getType()});
  state.addAttribute("kind", builder.getStringAttr(kind));
  state.addAttribute("parameter",
                     parameter ? parameter : builder.getF64FloatAttr(1.0));
  state.addAttribute("reduction", reduction);
  mlir::Operation *backward = builder.create(state);
  return {backward->getResult(0), backward->getResult(1)};
}

llvm::SmallVector<mlir::Value> MAELossOp::buildAdjoint(
    mlir::OpBuilder &builder, mlir::ValueRange outputCotangents) {
  return buildRegressionLossAdjoint(getOperation(), builder, outputCotangents,
                                    "mae", mlir::FloatAttr(),
                                    getReductionAttr());
}

llvm::SmallVector<mlir::Value> HuberLossOp::buildAdjoint(
    mlir::OpBuilder &builder, mlir::ValueRange outputCotangents) {
  return buildRegressionLossAdjoint(getOperation(), builder, outputCotangents,
                                    "huber", getDeltaAttr(),
                                    getReductionAttr());
}

llvm::SmallVector<mlir::Value> SmoothL1LossOp::buildAdjoint(
    mlir::OpBuilder &builder, mlir::ValueRange outputCotangents) {
  return buildRegressionLossAdjoint(getOperation(), builder, outputCotangents,
                                    "smooth_l1", getBetaAttr(),
                                    getReductionAttr());
}

llvm::SmallVector<mlir::Value> SGDOp::buildAdjoint(
    mlir::OpBuilder &builder, mlir::ValueRange outputCotangents) {
  if (outputCotangents.size() != 1 || !outputCotangents[0])
    return {mlir::Value(), mlir::Value()};
  mlir::OperationState state(getLoc(), "tessera.sgd_backward");
  state.addOperands(outputCotangents[0]);
  state.addTypes({getParam().getType(), getGrad().getType()});
  state.addAttribute("lr", getLrAttr());
  mlir::Operation *backward = builder.create(state);
  return {backward->getResult(0), backward->getResult(1)};
}

static llvm::SmallVector<mlir::Value> buildMomentumAdjoint(
    mlir::Operation *op, mlir::OpBuilder &builder,
    mlir::ValueRange outputCotangents, mlir::FloatAttr lr,
    mlir::FloatAttr momentum, bool nesterov) {
  if (outputCotangents.size() != 2 || !outputCotangents[0] ||
      !outputCotangents[1])
    return {mlir::Value(), mlir::Value(), mlir::Value()};
  mlir::OperationState state(op->getLoc(), "tessera.momentum_backward");
  state.addOperands({outputCotangents[0], outputCotangents[1]});
  state.addTypes(
      {op->getOperand(0).getType(), op->getOperand(1).getType(),
       op->getOperand(2).getType()});
  state.addAttribute("lr", lr);
  state.addAttribute("momentum", momentum);
  state.addAttribute("nesterov", builder.getBoolAttr(nesterov));
  mlir::Operation *backward = builder.create(state);
  return {backward->getResult(0), backward->getResult(1),
          backward->getResult(2)};
}

llvm::SmallVector<mlir::Value> MomentumOp::buildAdjoint(
    mlir::OpBuilder &builder, mlir::ValueRange outputCotangents) {
  return buildMomentumAdjoint(getOperation(), builder, outputCotangents,
                              getLrAttr(), getMomentumAttr(), false);
}

llvm::SmallVector<mlir::Value> NesterovOp::buildAdjoint(
    mlir::OpBuilder &builder, mlir::ValueRange outputCotangents) {
  return buildMomentumAdjoint(getOperation(), builder, outputCotangents,
                              getLrAttr(), getMomentumAttr(), true);
}

static llvm::SmallVector<mlir::Value> buildAdamAdjoint(
    mlir::Operation *op, mlir::OpBuilder &builder,
    mlir::ValueRange outputCotangents, bool adamw) {
  if (outputCotangents.size() != 3 || !outputCotangents[0] ||
      !outputCotangents[1] || !outputCotangents[2])
    return {mlir::Value(), mlir::Value(), mlir::Value(), mlir::Value()};
  auto f64 = [&](llvm::StringRef name, double fallback) -> mlir::FloatAttr {
    if (auto attr = op->getAttrOfType<mlir::FloatAttr>(name))
      return attr;
    return builder.getF64FloatAttr(fallback);
  };
  auto i64 = [&](llvm::StringRef name, int64_t fallback) -> mlir::IntegerAttr {
    if (auto attr = op->getAttrOfType<mlir::IntegerAttr>(name))
      return attr;
    return builder.getI64IntegerAttr(fallback);
  };
  mlir::OperationState state(op->getLoc(), "tessera.adam_backward");
  state.addOperands(
      {op->getOperand(0), op->getOperand(1), op->getOperand(2),
       op->getOperand(3), outputCotangents[0], outputCotangents[1],
       outputCotangents[2]});
  state.addTypes(
      {op->getOperand(0).getType(), op->getOperand(1).getType(),
       op->getOperand(2).getType(), op->getOperand(3).getType()});
  state.addAttribute("lr", f64("lr", 1.0e-3));
  state.addAttribute("beta1", f64("beta1", 0.9));
  state.addAttribute("beta2", f64("beta2", 0.999));
  state.addAttribute("eps", f64("eps", 1.0e-8));
  state.addAttribute("weight_decay", f64("weight_decay", 0.0));
  state.addAttribute("step", i64("step", 1));
  state.addAttribute("adamw", builder.getBoolAttr(adamw));
  mlir::Operation *backward = builder.create(state);
  return {backward->getResult(0), backward->getResult(1),
          backward->getResult(2), backward->getResult(3)};
}

llvm::SmallVector<mlir::Value> AdamOp::buildAdjoint(
    mlir::OpBuilder &builder, mlir::ValueRange outputCotangents) {
  return buildAdamAdjoint(
      getOperation(), builder, outputCotangents, false);
}

llvm::SmallVector<mlir::Value> AdamWOp::buildAdjoint(
    mlir::OpBuilder &builder, mlir::ValueRange outputCotangents) {
  return buildAdamAdjoint(
      getOperation(), builder, outputCotangents, true);
}

// Broadcast is inverted by summing every expanded dimension. Reduce in
// descending axis order so each remaining axis keeps its original index, then
// reshape to restore the input's explicit singleton dimensions. Leading axes
// and statically-singleton aligned axes remain known under dynamic shapes.
// An aligned `? -> static non-unit` dimension is genuinely ambiguous (runtime
// equality versus runtime singleton expansion), so only that case retains the
// reference fallback.
llvm::SmallVector<mlir::Value> BroadcastOp::buildAdjoint(
    mlir::OpBuilder &builder, mlir::ValueRange outputCotangents) {
  if (outputCotangents.size() != 1 || !outputCotangents[0])
    return {mlir::Value()};

  auto inTy = mlir::dyn_cast<mlir::RankedTensorType>(getX().getType());
  auto outTy = mlir::dyn_cast<mlir::RankedTensorType>(getY().getType());
  mlir::Value dy = outputCotangents[0];
  if (!inTy || !outTy)
    return placeholderAdjoint(builder, getLoc(), getX().getType(), "broadcast",
                              dy, getX());

  llvm::SmallVector<int64_t> reduceAxes;
  int64_t offset = outTy.getRank() - inTy.getRank();
  for (int64_t axis = 0; axis < offset; ++axis)
    reduceAxes.push_back(axis);
  for (int64_t axis = 0; axis < inTy.getRank(); ++axis) {
    int64_t outAxis = axis + offset;
    int64_t inputExtent = inTy.getDimSize(axis);
    int64_t outputExtent = outTy.getDimSize(outAxis);
    if (mlir::ShapedType::isDynamic(inputExtent) &&
        !mlir::ShapedType::isDynamic(outputExtent) && outputExtent != 1)
      return placeholderAdjoint(builder, getLoc(), getX().getType(),
                                "broadcast", dy, getX());
    if (inputExtent == 1 && outputExtent != 1)
      reduceAxes.push_back(outAxis);
  }

  mlir::Value grad = dy;
  for (int64_t i = static_cast<int64_t>(reduceAxes.size()) - 1; i >= 0; --i) {
    int64_t axis = reduceAxes[i];
    auto currentTy = mlir::cast<mlir::RankedTensorType>(grad.getType());
    llvm::SmallVector<int64_t> shape(currentTy.getShape());
    shape.erase(shape.begin() + axis);
    auto reducedTy = mlir::RankedTensorType::get(
        shape, currentTy.getElementType(), currentTy.getEncoding());
    grad = builder
               .create<ReduceOp>(getLoc(), reducedTy, grad,
                                 builder.getStringAttr("sum"),
                                 builder.getI64IntegerAttr(axis))
               .getResult();
  }
  if (grad.getType() != inTy)
    grad = builder.create<ReshapeOp>(getLoc(), inTy, grad).getY();
  return {grad};
}

// Single-axis reductions. Sum and static-extent mean retain the small
// decomposition into unsqueeze/broadcast/scale. Dynamic mean and max/min use a
// compiler-internal carrier whose lowering owns runtime extent and explicit
// equal-share tie semantics.
llvm::SmallVector<mlir::Value> ReduceOp::buildAdjoint(
    mlir::OpBuilder &builder, mlir::ValueRange outputCotangents) {
  if (outputCotangents.size() != 1 || !outputCotangents[0])
    return {mlir::Value()};

  auto inTy = mlir::dyn_cast<mlir::RankedTensorType>(getInput().getType());
  mlir::Value dy = outputCotangents[0];
  llvm::StringRef kind = getKind();
  if (!inTy || !containsKind(kReduceNativeAdjointKinds, kind))
    return placeholderAdjoint(builder, getLoc(), getInput().getType(), kind,
                              dy, getInput());

  int64_t axis = getAxisAttr().getInt();
  if (axis < 0)
    axis += inTy.getRank();
  if (axis < 0 || axis >= inTy.getRank())
    return {mlir::Value()}; // The op verifier diagnoses this before autodiff.

  if (kind == "max" || kind == "min" ||
      (kind == "mean" &&
       mlir::ShapedType::isDynamic(inTy.getDimSize(axis)))) {
    auto backward = builder.create<ReduceBackwardOp>(
        getLoc(), inTy, getInput(), getResult(), dy, getKindAttr(),
        builder.getI64IntegerAttr(axis), builder.getStringAttr("equal"));
    return {backward.getGrad()};
  }

  llvm::SmallVector<int64_t> expandedShape(inTy.getShape());
  expandedShape[axis] = 1;
  auto expandedTy = mlir::RankedTensorType::get(
      expandedShape, inTy.getElementType(), inTy.getEncoding());
  auto expanded = builder.create<UnsqueezeOp>(getLoc(), expandedTy, dy);
  expanded->setAttr(
      "axes", builder.getArrayAttr({builder.getI64IntegerAttr(axis)}));
  mlir::Value grad =
      builder.create<BroadcastOp>(getLoc(), inTy, expanded.getY()).getY();

  if (kind == "mean") {
    double reciprocal = 1.0 / static_cast<double>(inTy.getDimSize(axis));
    mlir::Value scale = splatConst(builder, getLoc(), inTy, reciprocal);
    grad = builder.create<MulOp>(getLoc(), inTy, grad, scale).getResult();
  }
  return {grad};
}

// ─────────────────────────────────────────────────────────────────────────────
// Unary ops — adjoints follow the standard chain rule on a scalar function.
//
// Each impl recomputes the forward intermediates that show up in the adjoint
// formula via fresh ops; downstream canonicalization can DCE / CSE them.
// ─────────────────────────────────────────────────────────────────────────────

// LayerNormOp — dx = (1/n) * inv_std * (n*dout - sum(dout) - y * sum(dout * y))
// where y = (x - mean) / sqrt(var + eps).
// (See Python `vjp_layer_norm` for the matching closed form.)
llvm::SmallVector<mlir::Value> LayerNormOp::buildAdjoint(
    mlir::OpBuilder &builder, mlir::ValueRange outputCotangents) {
  if (outputCotangents.size() != 1 || !outputCotangents[0])
    return {mlir::Value()};
  auto inputType = mlir::dyn_cast<mlir::RankedTensorType>(getX().getType());
  if (!inputType || inputType.getRank() < 1)
    return placeholderAdjoint(builder, getLoc(), getX().getType(),
                              "layer_norm", outputCotangents[0], getX());
  auto loc = getLoc();
  mlir::Value dy = outputCotangents[0];
  auto [center, inverseScale] = buildNormalizationStats(
      builder, loc, getX(), inputType,
      (*this)->getAttrOfType<mlir::FloatAttr>("eps"), /*centered=*/true);
  mlir::Value centerBroadcast =
      broadcastLastAxisStat(builder, loc, inputType, center, getX());
  mlir::Value inverseBroadcast =
      broadcastLastAxisStat(builder, loc, inputType, inverseScale, getX());
  mlir::Value centered =
      builder.create<SubOp>(loc, inputType, getX(), centerBroadcast).getResult();
  mlir::Value normalized =
      builder.create<MulOp>(loc, inputType, centered, inverseBroadcast).getResult();
  mlir::Value projectedDy = dy;
  if (!getAffine().empty()) {
    mlir::Value gamma = getAffine().front();
    projectedDy = builder
                      .create<MulOp>(loc, inputType, dy,
                                     broadcastChannelVector(builder, loc,
                                                            inputType, gamma,
                                                            getX()))
                      .getResult();
  }
  mlir::Value meanDy =
      reduceMeanLastAxis(builder, loc, inputType, projectedDy);
  mlir::Value meanDyBroadcast =
      broadcastLastAxisStat(builder, loc, inputType, meanDy, getX());
  mlir::Value dyNormalized =
      builder.create<MulOp>(loc, inputType, projectedDy, normalized).getResult();
  mlir::Value meanDyNormalized =
      reduceMeanLastAxis(builder, loc, inputType, dyNormalized);
  mlir::Value meanDyNormalizedBroadcast =
      broadcastLastAxisStat(builder, loc, inputType, meanDyNormalized, getX());
  mlir::Value centeredDy =
      builder.create<SubOp>(loc, inputType, projectedDy, meanDyBroadcast).getResult();
  mlir::Value correction = builder
                               .create<MulOp>(loc, inputType, normalized,
                                              meanDyNormalizedBroadcast)
                               .getResult();
  mlir::Value projected =
      builder.create<SubOp>(loc, inputType, centeredDy, correction).getResult();
  llvm::SmallVector<mlir::Value> gradients;
  gradients.push_back(
      builder.create<MulOp>(loc, inputType, projected, inverseBroadcast)
          .getResult());
  if (!getAffine().empty()) {
    mlir::Value gammaProduct =
        builder.create<MulOp>(loc, inputType, dy, normalized).getResult();
    gradients.push_back(
        reduceLeadingSum(builder, loc, inputType, gammaProduct));
  }
  if (getAffine().size() > 1)
    gradients.push_back(reduceLeadingSum(builder, loc, inputType, dy));
  return gradients;
}

llvm::SmallVector<mlir::Value> SoftmaxOp::buildAdjoint(
    mlir::OpBuilder &builder, mlir::ValueRange outputCotangents) {
  if (outputCotangents.size() != 1 || !outputCotangents[0])
    return {mlir::Value()};
  auto inTy = mlir::dyn_cast<mlir::RankedTensorType>(getX().getType());
  if (!inTy || inTy.getRank() < 1)
    return placeholderAdjoint(builder, getLoc(), getX().getType(), "softmax",
                              outputCotangents[0], getX());

  int64_t axis = -1;
  if (auto axisAttr = getAxisAttr())
    axis = axisAttr.getInt();
  if (axis < 0)
    axis += inTy.getRank();
  if (axis < 0 || axis >= inTy.getRank())
    return {mlir::Value()}; // Diagnosed by SoftmaxOp::verify.

  auto loc = getLoc();
  mlir::Value dy = outputCotangents[0];
  // Use the saved forward probabilities in the normal latency-oriented path.
  // The production rematerialization pass can choose to recompute this op when
  // a function memory budget makes the activation too expensive to retain.
  mlir::Value s = getY();
  mlir::Value weighted =
      builder.create<MulOp>(loc, inTy, dy, s).getResult();
  llvm::SmallVector<int64_t> reducedShape(inTy.getShape());
  reducedShape.erase(reducedShape.begin() + axis);
  auto reducedTy = mlir::RankedTensorType::get(
      reducedShape, inTy.getElementType(), inTy.getEncoding());
  mlir::Value dot =
      builder
          .create<ReduceOp>(loc, reducedTy, weighted,
                            builder.getStringAttr("sum"),
                            builder.getI64IntegerAttr(axis))
          .getResult();
  llvm::SmallVector<int64_t> expandedShape(inTy.getShape());
  expandedShape[axis] = 1;
  auto expandedTy = mlir::RankedTensorType::get(
      expandedShape, inTy.getElementType(), inTy.getEncoding());
  auto expanded = builder.create<UnsqueezeOp>(loc, expandedTy, dot);
  expanded->setAttr(
      "axes", builder.getArrayAttr({builder.getI64IntegerAttr(axis)}));
  mlir::Value dotBroadcast =
      builder.create<BroadcastOp>(loc, inTy, expanded.getY()).getY();
  mlir::Value centered =
      builder.create<SubOp>(loc, inTy, dy, dotBroadcast).getResult();
  return {builder.create<MulOp>(loc, inTy, centered, s).getResult()};
}

// Pointwise activations (relu, gelu, sigmoid, sin) — each has a closed-form
// derivative. Easiest path: emit a `tessera.custom_adjoint_call` placeholder
// the runtime resolves via the Python VJP registry. The buildAdjoint impl
// carries the key as a StringAttr on the placeholder; ops don't need to
// override `customAdjointName()` (its default-empty return is correct since
// the dispatch goes through the placeholder, not the interface method).
#define POINTWISE_BUILD_ADJOINT(OPNAME, KEY)                                  \
  llvm::SmallVector<mlir::Value> OPNAME::buildAdjoint(                        \
      mlir::OpBuilder &builder, mlir::ValueRange outputCotangents) {          \
    if (outputCotangents.size() != 1 || !outputCotangents[0])                 \
      return {mlir::Value()};                                                 \
    auto loc = getLoc();                                                      \
    auto callOp = builder.create<CustomAdjointCallOp>(                        \
        loc, llvm::SmallVector<mlir::Type>{getX().getType()},                 \
        builder.getStringAttr(KEY),                                           \
        mlir::ValueRange{outputCotangents[0], getX()});                       \
    return {callOp.getResult(0)};                                             \
  }

POINTWISE_BUILD_ADJOINT(SinOp, "sin")
// Tier-1 MPSGraph-lane ops (2026-05-29). Each has a Python VJP the runtime
// resolves via the custom_adjoint_call placeholder keyed by name.
POINTWISE_BUILD_ADJOINT(SoftplusOp, "softplus")
POINTWISE_BUILD_ADJOINT(LogSoftmaxOp, "log_softmax")

#undef POINTWISE_BUILD_ADJOINT

// relu: dx = x > 0 ? dy : 0.  compare_scalar produces a genuine i1 tensor
// mask even for dynamic shapes; masked_fill supplies the scalar-zero branch
// without requiring a statically-shaped dense constant.
llvm::SmallVector<mlir::Value> ReluOp::buildAdjoint(
    mlir::OpBuilder &builder, mlir::ValueRange outputCotangents) {
  if (outputCotangents.size() != 1 || !outputCotangents[0])
    return {mlir::Value()};
  auto inputType = mlir::dyn_cast<mlir::RankedTensorType>(getX().getType());
  if (!inputType)
    return placeholderAdjoint(builder, getLoc(), getX().getType(), "relu",
                              outputCotangents[0], getX());
  auto maskType = mlir::RankedTensorType::get(
      inputType.getShape(), builder.getI1Type(), inputType.getEncoding());
  auto mask = builder.create<CompareScalarOp>(
      getLoc(), maskType, getX(), builder.getF64FloatAttr(0.0),
      builder.getStringAttr("gt"));
  auto dx = builder.create<MaskedFillOp>(
      getLoc(), inputType, outputCotangents[0], mask.getMask(),
      builder.getF64FloatAttr(0.0));
  return {dx.getResult()};
}

// rmsnorm: r = rsqrt(mean(x^2)+eps)
//          dx = dy*r - x*r^3*mean(dy*x)
llvm::SmallVector<mlir::Value> RmsNormOp::buildAdjoint(
    mlir::OpBuilder &builder, mlir::ValueRange outputCotangents) {
  if (outputCotangents.size() != 1 || !outputCotangents[0])
    return {mlir::Value()};
  auto inputType = mlir::dyn_cast<mlir::RankedTensorType>(getX().getType());
  if (!inputType || inputType.getRank() < 1)
    return placeholderAdjoint(builder, getLoc(), getX().getType(), "rmsnorm",
                              outputCotangents[0], getX());
  auto loc = getLoc();
  mlir::Value dy = outputCotangents[0];
  auto [unusedCenter, inverseScale] = buildNormalizationStats(
      builder, loc, getX(), inputType,
      (*this)->getAttrOfType<mlir::FloatAttr>("eps"), /*centered=*/false);
  (void)unusedCenter;
  mlir::Value inverseBroadcast =
      broadcastLastAxisStat(builder, loc, inputType, inverseScale, getX());
  mlir::Value projectedDy = dy;
  if (!getAffine().empty()) {
    projectedDy = builder
                      .create<MulOp>(
                          loc, inputType, dy,
                          broadcastChannelVector(builder, loc, inputType,
                                                 getAffine().front(), getX()))
                      .getResult();
  }
  mlir::Value leading = builder
                            .create<MulOp>(loc, inputType, projectedDy,
                                           inverseBroadcast)
                            .getResult();
  mlir::Value dyX =
      builder.create<MulOp>(loc, inputType, projectedDy, getX()).getResult();
  mlir::Value meanDyX = reduceMeanLastAxis(builder, loc, inputType, dyX);
  mlir::Value meanDyXBroadcast =
      broadcastLastAxisStat(builder, loc, inputType, meanDyX, getX());
  mlir::Value inverseSquared =
      builder.create<MulOp>(loc, inputType, inverseBroadcast, inverseBroadcast)
          .getResult();
  mlir::Value inverseCubed =
      builder.create<MulOp>(loc, inputType, inverseSquared, inverseBroadcast)
          .getResult();
  mlir::Value scaledX =
      builder.create<MulOp>(loc, inputType, getX(), inverseCubed).getResult();
  mlir::Value correction =
      builder.create<MulOp>(loc, inputType, scaledX, meanDyXBroadcast)
          .getResult();
  llvm::SmallVector<mlir::Value> gradients;
  gradients.push_back(
      builder.create<SubOp>(loc, inputType, leading, correction).getResult());
  if (!getAffine().empty()) {
    mlir::Value normalized =
        builder.create<MulOp>(loc, inputType, getX(), inverseBroadcast).getResult();
    gradients.push_back(reduceLeadingSum(
        builder, loc, inputType,
        builder.create<MulOp>(loc, inputType, dy, normalized).getResult()));
  }
  return gradients;
}

// ─────────────────────────────────────────────────────────────────────────────
// Native (compiler-visible) pointwise adjoints (W5).
//
// tanh and sigmoid have compare-free closed-form derivatives expressible in
// pure Tessera Graph IR (their own forward op + mul/sub + a constant), so their
// backward is emitted NATIVELY instead of a `tessera.custom_adjoint_call`
// placeholder that round-trips to the Python VJP. This makes the backward graph
// first-class compiler IR — it can be canonicalized, CSE'd (the recomputed
// forward intermediate dedups with the forward), and fused, and it drops the
// host VJP call. The recompute of the forward activation is intentional (the
// docstring convention): downstream CSE collapses it back to the forward value.
// ─────────────────────────────────────────────────────────────────────────────

// The native form materializes a dense splat `1` sized to the result type, which
// MLIR only allows for a statically-shaped ranked tensor. A dynamic activation
// type (e.g. before a later shape refinement) has no such constant, so the native
// path is gated on a static shape and falls back to the placeholder otherwise.
static bool isStaticShaped(mlir::Type ty) {
  auto rt = mlir::dyn_cast<mlir::RankedTensorType>(ty);
  return rt && rt.hasStaticShape();
}

// A dense splat constant of `value` matching a static shape-preserving tensor.
static mlir::Value splatConst(mlir::OpBuilder &builder, mlir::Location loc,
                              mlir::Type ty, double value) {
  auto shaped = mlir::cast<mlir::ShapedType>(ty);
  auto attr = mlir::DenseElementsAttr::get(
      shaped, builder.getFloatAttr(shaped.getElementType(), value));
  return builder.create<mlir::arith::ConstantOp>(loc, attr).getResult();
}

// Dynamic-shape (or unranked) fallback: the opaque placeholder the runtime VJP
// registry resolves — the pre-W5 behavior for these ops, kept shape-safe.
static llvm::SmallVector<mlir::Value>
placeholderAdjoint(mlir::OpBuilder &builder, mlir::Location loc, mlir::Type ty,
                   llvm::StringRef key, mlir::Value dy, mlir::Value x) {
  auto callOp = builder.create<CustomAdjointCallOp>(
      loc, llvm::SmallVector<mlir::Type>{ty}, builder.getStringAttr(key),
      mlir::ValueRange{dy, x});
  return {callOp.getResult(0)};
}

// tanh:  dx = dy · (1 − tanh(x)²)
llvm::SmallVector<mlir::Value> TanhOp::buildAdjoint(
    mlir::OpBuilder &builder, mlir::ValueRange outputCotangents) {
  if (outputCotangents.size() != 1 || !outputCotangents[0])
    return {mlir::Value()};
  auto loc = getLoc();
  mlir::Type ty = getResult().getType();
  mlir::Value dy = outputCotangents[0];
  if (!isStaticShaped(ty))
    return placeholderAdjoint(builder, loc, ty, "tanh", dy, getX());
  // Save the forward activation on the latency path; function-budgeted
  // rematerialization may replace backward uses with recomputation.
  mlir::Value t = getResult();
  mlir::Value t2 = builder.create<MulOp>(loc, ty, t, t).getResult();
  mlir::Value one = splatConst(builder, loc, ty, 1.0);
  mlir::Value d = builder.create<SubOp>(loc, ty, one, t2).getResult();
  mlir::Value dx = builder.create<MulOp>(loc, ty, dy, d).getResult();
  return {dx};
}

// sigmoid:  dx = dy · s · (1 − s),  s = sigmoid(x)
llvm::SmallVector<mlir::Value> SigmoidOp::buildAdjoint(
    mlir::OpBuilder &builder, mlir::ValueRange outputCotangents) {
  if (outputCotangents.size() != 1 || !outputCotangents[0])
    return {mlir::Value()};
  auto loc = getLoc();
  mlir::Type ty = getResult().getType();
  mlir::Value dy = outputCotangents[0];
  if (!isStaticShaped(ty))
    return placeholderAdjoint(builder, loc, ty, "sigmoid", dy, getX());
  // Save the forward activation on the latency path; function-budgeted
  // rematerialization may replace backward uses with recomputation.
  mlir::Value s = getResult();
  mlir::Value one = splatConst(builder, loc, ty, 1.0);
  mlir::Value oneMinusS = builder.create<SubOp>(loc, ty, one, s).getResult();
  mlir::Value sPrime = builder.create<MulOp>(loc, ty, s, oneMinusS).getResult();
  mlir::Value dx = builder.create<MulOp>(loc, ty, dy, sPrime).getResult();
  return {dx};
}

// silu: dx = dy · (s + x·s·(1−s)), s = sigmoid(x)
llvm::SmallVector<mlir::Value> SiluOp::buildAdjoint(
    mlir::OpBuilder &builder, mlir::ValueRange outputCotangents) {
  if (outputCotangents.size() != 1 || !outputCotangents[0])
    return {mlir::Value()};
  auto loc = getLoc();
  mlir::Type ty = getResult().getType();
  mlir::Value dy = outputCotangents[0];
  if (!isStaticShaped(ty))
    return placeholderAdjoint(builder, loc, ty, "silu", dy, getX());
  mlir::Value s = builder.create<SigmoidOp>(loc, ty, getX()).getResult();
  mlir::Value one = splatConst(builder, loc, ty, 1.0);
  mlir::Value oneMinusS = builder.create<SubOp>(loc, ty, one, s).getResult();
  mlir::Value xs = builder.create<MulOp>(loc, ty, getX(), s).getResult();
  mlir::Value correction =
      builder.create<MulOp>(loc, ty, xs, oneMinusS).getResult();
  mlir::Value derivative =
      builder.create<AddOp>(loc, ty, s, correction).getResult();
  return {builder.create<MulOp>(loc, ty, dy, derivative).getResult()};
}

// tanh-approx GELU, matching python/tessera/autodiff/vjp.py::vjp_gelu.
llvm::SmallVector<mlir::Value> GeluOp::buildAdjoint(
    mlir::OpBuilder &builder, mlir::ValueRange outputCotangents) {
  if (outputCotangents.size() != 1 || !outputCotangents[0])
    return {mlir::Value()};
  auto loc = getLoc();
  mlir::Type ty = getResult().getType();
  mlir::Value dy = outputCotangents[0];
  if (!isStaticShaped(ty))
    return placeholderAdjoint(builder, loc, ty, "gelu", dy, getX());

  constexpr double k = 0.7978845608028654; // sqrt(2 / pi)
  mlir::Value one = splatConst(builder, loc, ty, 1.0);
  mlir::Value half = splatConst(builder, loc, ty, 0.5);
  mlir::Value cubicCoeff = splatConst(builder, loc, ty, 0.044715);
  mlir::Value threeCubicCoeff = splatConst(builder, loc, ty, 0.134145);
  mlir::Value kConst = splatConst(builder, loc, ty, k);

  mlir::Value x2 = builder.create<MulOp>(loc, ty, getX(), getX()).getResult();
  mlir::Value x3 = builder.create<MulOp>(loc, ty, x2, getX()).getResult();
  mlir::Value cubic =
      builder.create<MulOp>(loc, ty, cubicCoeff, x3).getResult();
  mlir::Value innerBase =
      builder.create<AddOp>(loc, ty, getX(), cubic).getResult();
  mlir::Value inner =
      builder.create<MulOp>(loc, ty, kConst, innerBase).getResult();
  mlir::Value t = builder.create<TanhOp>(loc, ty, inner).getResult();

  mlir::Value onePlusT = builder.create<AddOp>(loc, ty, one, t).getResult();
  mlir::Value first =
      builder.create<MulOp>(loc, ty, half, onePlusT).getResult();
  mlir::Value t2 = builder.create<MulOp>(loc, ty, t, t).getResult();
  mlir::Value oneMinusT2 =
      builder.create<SubOp>(loc, ty, one, t2).getResult();
  mlir::Value scaledX2 =
      builder.create<MulOp>(loc, ty, threeCubicCoeff, x2).getResult();
  mlir::Value slopeBase =
      builder.create<AddOp>(loc, ty, one, scaledX2).getResult();
  mlir::Value innerSlope =
      builder.create<MulOp>(loc, ty, kConst, slopeBase).getResult();
  mlir::Value halfX =
      builder.create<MulOp>(loc, ty, half, getX()).getResult();
  mlir::Value gatedX =
      builder.create<MulOp>(loc, ty, halfX, oneMinusT2).getResult();
  mlir::Value second =
      builder.create<MulOp>(loc, ty, gatedX, innerSlope).getResult();
  mlir::Value derivative =
      builder.create<AddOp>(loc, ty, first, second).getResult();
  return {builder.create<MulOp>(loc, ty, dy, derivative).getResult()};
}

}  // namespace tessera
