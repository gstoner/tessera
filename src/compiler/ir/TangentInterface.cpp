//===- TangentInterface.cpp - Graph IR JVP implementations -----*- C++ -*-===//

#include "Tessera/IR/TesseraOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/Builders.h"
#include "llvm/ADT/SmallVector.h"

namespace tessera {

static mlir::Value buildZeroLike(mlir::OpBuilder &builder, mlir::Location loc,
                                 mlir::Type type) {
  auto shaped = mlir::dyn_cast<mlir::ShapedType>(type);
  if (!shaped || !shaped.hasStaticShape())
    return {};
  mlir::Attribute zero;
  if (auto element = mlir::dyn_cast<mlir::FloatType>(shaped.getElementType()))
    zero = builder.getFloatAttr(element, 0.0);
  else if (auto element =
               mlir::dyn_cast<mlir::IntegerType>(shaped.getElementType()))
    zero = builder.getIntegerAttr(element, 0);
  else
    return {};
  return builder
      .create<mlir::arith::ConstantOp>(
          loc, mlir::DenseElementsAttr::get(shaped, zero))
      .getResult();
}

static mlir::Value buildFloatSplat(mlir::OpBuilder &builder, mlir::Location loc,
                                   mlir::Type type, double value) {
  auto shaped = mlir::dyn_cast<mlir::ShapedType>(type);
  if (!shaped || !shaped.hasStaticShape())
    return {};
  auto element = mlir::dyn_cast<mlir::FloatType>(shaped.getElementType());
  if (!element)
    return {};
  return builder
      .create<mlir::arith::ConstantOp>(
          loc, mlir::DenseElementsAttr::get(
                   shaped, builder.getFloatAttr(element, value)))
      .getResult();
}

static mlir::Value cloneUnaryTangent(mlir::Operation *source,
                                     mlir::Value tangent,
                                     mlir::OpBuilder &builder) {
  mlir::OperationState state(source->getLoc(), source->getName());
  state.addOperands(tangent);
  state.addTypes(source->getResultTypes());
  state.addAttributes(source->getAttrs());
  return builder.create(state)->getResult(0);
}

static mlir::Value cloneTangentWithOperands(
    mlir::Operation *source, mlir::ValueRange operands,
    mlir::OpBuilder &builder) {
  mlir::OperationState state(source->getLoc(), source->getName());
  state.addOperands(operands);
  state.addTypes(source->getResultTypes());
  state.addAttributes(source->getAttrs());
  return builder.create(state)->getResult(0);
}

static mlir::RankedTensorType lastAxisReducedType(
    mlir::RankedTensorType inputType) {
  llvm::SmallVector<int64_t> shape(inputType.getShape());
  shape.pop_back();
  return mlir::RankedTensorType::get(shape, inputType.getElementType(),
                                     inputType.getEncoding());
}

static std::pair<mlir::Value, mlir::Value> buildNormalizationStats(
    mlir::OpBuilder &builder, mlir::Location loc, mlir::Value x,
    mlir::RankedTensorType inputType, mlir::FloatAttr eps, bool centered) {
  auto statsType = lastAxisReducedType(inputType);
  auto stats = builder.create<NormalizationStatsOp>(
      loc, mlir::TypeRange{statsType, statsType}, x,
      builder.getI64IntegerAttr(-1),
      eps ? eps : builder.getF64FloatAttr(1.0e-5),
      builder.getBoolAttr(centered));
  return {stats.getCenter(), stats.getInverseScale()};
}

static mlir::Value broadcastLastAxis(mlir::OpBuilder &builder,
                                     mlir::Location loc,
                                     mlir::RankedTensorType inputType,
                                     mlir::Value value,
                                     mlir::Value shapeLike) {
  llvm::SmallVector<mlir::Attribute> dimensions;
  for (int64_t dim = 0; dim + 1 < inputType.getRank(); ++dim)
    dimensions.push_back(builder.getI64IntegerAttr(dim));
  return builder
      .create<BroadcastInDimOp>(loc, inputType, value, shapeLike,
                                builder.getArrayAttr(dimensions))
      .getY();
}

static mlir::Value broadcastChannel(mlir::OpBuilder &builder,
                                    mlir::Location loc,
                                    mlir::RankedTensorType inputType,
                                    mlir::Value value,
                                    mlir::Value shapeLike) {
  return builder
      .create<BroadcastInDimOp>(
          loc, inputType, value, shapeLike,
          builder.getArrayAttr(
              {builder.getI64IntegerAttr(inputType.getRank() - 1)}))
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

static mlir::Value addIfPresent(mlir::OpBuilder &builder, mlir::Location loc,
                                mlir::Type type, mlir::Value lhs,
                                mlir::Value rhs) {
  if (!lhs)
    return rhs;
  if (!rhs)
    return lhs;
  return builder.create<AddOp>(loc, type, lhs, rhs).getResult();
}

llvm::SmallVector<mlir::Value> AddOp::buildTangent(
    mlir::OpBuilder &builder, mlir::ValueRange tangents) {
  if (tangents.size() != 2 || (!tangents[0] && !tangents[1]))
    return {};
  if (!tangents[0])
    return {tangents[1]};
  if (!tangents[1])
    return {tangents[0]};
  return {builder.create<AddOp>(getLoc(), getResult().getType(), tangents[0],
                                tangents[1]).getResult()};
}

llvm::SmallVector<mlir::Value> SubOp::buildTangent(
    mlir::OpBuilder &builder, mlir::ValueRange tangents) {
  if (tangents.size() != 2 || (!tangents[0] && !tangents[1]))
    return {};
  if (!tangents[1])
    return {tangents[0]};
  if (!tangents[0]) {
    mlir::Value zero = buildZeroLike(builder, getLoc(), getResult().getType());
    if (!zero)
      return {};
    return {builder
                .create<SubOp>(getLoc(), getResult().getType(), zero,
                               tangents[1])
                .getResult()};
  }
  return {builder.create<SubOp>(getLoc(), getResult().getType(), tangents[0],
                                tangents[1]).getResult()};
}

llvm::SmallVector<mlir::Value> MulOp::buildTangent(
    mlir::OpBuilder &builder, mlir::ValueRange tangents) {
  if (tangents.size() != 2 || (!tangents[0] && !tangents[1]))
    return {};
  mlir::Type type = getResult().getType();
  mlir::Value lhsTerm = tangents[0]
                            ? builder
                                  .create<MulOp>(getLoc(), type, tangents[0],
                                                 getRhs())
                                  .getResult()
                            : mlir::Value{};
  mlir::Value rhsTerm = tangents[1]
                            ? builder
                                  .create<MulOp>(getLoc(), type, getLhs(),
                                                 tangents[1])
                                  .getResult()
                            : mlir::Value{};
  if (!lhsTerm)
    return {rhsTerm};
  if (!rhsTerm)
    return {lhsTerm};
  return {builder.create<AddOp>(getLoc(), type, lhsTerm, rhsTerm).getResult()};
}

llvm::SmallVector<mlir::Value> MatmulOp::buildTangent(
    mlir::OpBuilder &builder, mlir::ValueRange tangents) {
  if (tangents.size() != 2 || (!tangents[0] && !tangents[1]))
    return {};
  mlir::Type type = getResult().getType();
  auto build = [&](mlir::Value lhs, mlir::Value rhs) {
    return builder
        .create<MatmulOp>(getLoc(), type, lhs, rhs, getTileKAttr(),
                          getNumericPolicyAttr(), getTransposeAAttr(),
                          getTransposeBAttr())
        .getResult();
  };
  mlir::Value lhsTerm =
      tangents[0] ? build(tangents[0], getRhs()) : mlir::Value{};
  mlir::Value rhsTerm =
      tangents[1] ? build(getLhs(), tangents[1]) : mlir::Value{};
  if (!lhsTerm)
    return {rhsTerm};
  if (!rhsTerm)
    return {lhsTerm};
  return {builder.create<AddOp>(getLoc(), type, lhsTerm, rhsTerm).getResult()};
}

llvm::SmallVector<mlir::Value> ESLowRankCorrectionOp::buildTangent(
    mlir::OpBuilder &builder, mlir::ValueRange tangents) {
  if (tangents.size() != 3 || !tangents[0] || tangents[1] || tangents[2])
    return {};
  return {cloneTangentWithOperands(
      getOperation(), {tangents[0], getMemberIds(), getKey()}, builder)};
}

llvm::SmallVector<mlir::Value> ReduceOp::buildTangent(
    mlir::OpBuilder &builder, mlir::ValueRange tangents) {
  if (tangents.size() != 1 || !tangents[0])
    return {};
  // max/min have no single Frechet derivative at ties. They remain fail-closed
  // until a separately named directional/subgradient policy is selected.
  if (getKind() != "sum" && getKind() != "mean")
    return {};
  return {cloneTangentWithOperands(getOperation(), {tangents[0]}, builder)};
}

llvm::SmallVector<mlir::Value> SoftmaxOp::buildTangent(
    mlir::OpBuilder &builder, mlir::ValueRange tangents) {
  if (tangents.size() != 1 || !tangents[0])
    return {};
  auto inputType = mlir::dyn_cast<mlir::RankedTensorType>(getX().getType());
  if (!inputType || inputType.getRank() < 1)
    return {};
  int64_t axis = getAxisAttr() ? getAxisAttr().getInt() : -1;
  if (axis < 0)
    axis += inputType.getRank();
  if (axis < 0 || axis >= inputType.getRank())
    return {};
  mlir::Location loc = getLoc();
  mlir::Value weighted =
      builder.create<MulOp>(loc, inputType, getY(), tangents[0]).getResult();
  llvm::SmallVector<int64_t> reducedShape(inputType.getShape());
  reducedShape.erase(reducedShape.begin() + axis);
  auto reducedType = mlir::RankedTensorType::get(
      reducedShape, inputType.getElementType(), inputType.getEncoding());
  mlir::Value dot =
      builder
          .create<ReduceOp>(loc, reducedType, weighted,
                            builder.getStringAttr("sum"),
                            builder.getI64IntegerAttr(axis))
          .getResult();
  llvm::SmallVector<int64_t> expandedShape(inputType.getShape());
  expandedShape[axis] = 1;
  auto expandedType = mlir::RankedTensorType::get(
      expandedShape, inputType.getElementType(), inputType.getEncoding());
  auto expanded = builder.create<UnsqueezeOp>(loc, expandedType, dot);
  expanded->setAttr(
      "axes", builder.getArrayAttr({builder.getI64IntegerAttr(axis)}));
  mlir::Value dotBroadcast =
      builder.create<BroadcastOp>(loc, inputType, expanded.getY()).getY();
  mlir::Value centered =
      builder.create<SubOp>(loc, inputType, tangents[0], dotBroadcast)
          .getResult();
  return {builder.create<MulOp>(loc, inputType, getY(), centered).getResult()};
}

llvm::SmallVector<mlir::Value> LayerNormOp::buildTangent(
    mlir::OpBuilder &builder, mlir::ValueRange tangents) {
  if (tangents.empty() || !tangents[0])
    return {};
  auto type = mlir::dyn_cast<mlir::RankedTensorType>(getX().getType());
  if (!type || type.getRank() < 1)
    return {};
  auto loc = getLoc();
  auto [center, inverse] = buildNormalizationStats(
      builder, loc, getX(), type, getEpsAttr(), /*centered=*/true);
  mlir::Value centerB = broadcastLastAxis(builder, loc, type, center, getX());
  mlir::Value inverseB = broadcastLastAxis(builder, loc, type, inverse, getX());
  mlir::Value normalized = builder.create<MulOp>(
      loc, type,
      builder.create<SubOp>(loc, type, getX(), centerB).getResult(),
      inverseB).getResult();
  mlir::Value meanDx = reduceMeanLastAxis(builder, loc, type, tangents[0]);
  mlir::Value dxCentered = builder.create<SubOp>(
      loc, type, tangents[0],
      broadcastLastAxis(builder, loc, type, meanDx, getX())).getResult();
  mlir::Value meanProjected = reduceMeanLastAxis(
      builder, loc, type,
      builder.create<MulOp>(loc, type, tangents[0], normalized).getResult());
  mlir::Value correction = builder.create<MulOp>(
      loc, type, normalized,
      broadcastLastAxis(builder, loc, type, meanProjected, getX())).getResult();
  mlir::Value result = builder.create<MulOp>(
      loc, type,
      builder.create<SubOp>(loc, type, dxCentered, correction).getResult(),
      inverseB).getResult();
  if (!getAffine().empty())
    result = builder.create<MulOp>(
        loc, type, result,
        broadcastChannel(builder, loc, type, getAffine().front(), getX()))
                 .getResult();
  if (tangents.size() > 1 && tangents[1])
    result = addIfPresent(
        builder, loc, type, result,
        builder.create<MulOp>(
            loc, type, normalized,
            broadcastChannel(builder, loc, type, tangents[1], getX()))
            .getResult());
  if (tangents.size() > 2 && tangents[2])
    result = addIfPresent(
        builder, loc, type, result,
        broadcastChannel(builder, loc, type, tangents[2], getX()));
  return {result};
}

llvm::SmallVector<mlir::Value> RmsNormOp::buildTangent(
    mlir::OpBuilder &builder, mlir::ValueRange tangents) {
  if (tangents.empty() || !tangents[0])
    return {};
  auto type = mlir::dyn_cast<mlir::RankedTensorType>(getX().getType());
  if (!type || type.getRank() < 1)
    return {};
  auto loc = getLoc();
  auto [unused, inverse] = buildNormalizationStats(
      builder, loc, getX(), type, getEpsAttr(), /*centered=*/false);
  (void)unused;
  mlir::Value inverseB = broadcastLastAxis(builder, loc, type, inverse, getX());
  mlir::Value normalized =
      builder.create<MulOp>(loc, type, getX(), inverseB).getResult();
  mlir::Value meanProjected = reduceMeanLastAxis(
      builder, loc, type,
      builder.create<MulOp>(loc, type, tangents[0], normalized).getResult());
  mlir::Value correction = builder.create<MulOp>(
      loc, type, normalized,
      broadcastLastAxis(builder, loc, type, meanProjected, getX())).getResult();
  mlir::Value result = builder.create<MulOp>(
      loc, type,
      builder.create<SubOp>(loc, type, tangents[0], correction).getResult(),
      inverseB).getResult();
  if (!getAffine().empty())
    result = builder.create<MulOp>(
        loc, type, result,
        broadcastChannel(builder, loc, type, getAffine().front(), getX()))
                 .getResult();
  if (tangents.size() > 1 && tangents[1])
    result = addIfPresent(
        builder, loc, type, result,
        builder.create<MulOp>(
            loc, type, normalized,
            broadcastChannel(builder, loc, type, tangents[1], getX()))
            .getResult());
  return {result};
}

llvm::SmallVector<mlir::Value> DropoutOp::buildTangent(
    mlir::OpBuilder &builder, mlir::ValueRange tangents) {
  if (tangents.size() != 1 || !tangents[0])
    return {};
  bool training = getTraining();
  double probability = getP() ? getP()->convertToDouble() : 0.5;
  if (training && probability > 0.0 && !getSeedAttr())
    return {};
  return {cloneTangentWithOperands(getOperation(), {tangents[0]}, builder)};
}

#define TESSERA_COLLECTIVE_TANGENT(OPNAME)                                  \
  llvm::SmallVector<mlir::Value> OPNAME::buildTangent(                      \
      mlir::OpBuilder &builder, mlir::ValueRange tangents) {                \
    if (tangents.size() != 1 || !tangents[0])                               \
      return {};                                                             \
    if (auto reduction = getOpAttr()) {                                      \
      if (reduction.getValue() != "sum" && reduction.getValue() != "mean") \
        return {};                                                           \
    }                                                                        \
    return {cloneTangentWithOperands(getOperation(), {tangents[0]}, builder)}; \
  }

TESSERA_COLLECTIVE_TANGENT(AllReduceOp)
TESSERA_COLLECTIVE_TANGENT(ReduceScatterOp)
TESSERA_COLLECTIVE_TANGENT(AllGatherOp)
TESSERA_COLLECTIVE_TANGENT(AllToAllOp)

#undef TESSERA_COLLECTIVE_TANGENT

llvm::SmallVector<mlir::Value> StopGradientOp::buildTangent(
    mlir::OpBuilder &builder, mlir::ValueRange tangents) {
  if (tangents.size() != 1)
    return {};
  mlir::Value zero = buildZeroLike(builder, getLoc(), getResult().getType());
  return zero ? llvm::SmallVector<mlir::Value>{zero}
              : llvm::SmallVector<mlir::Value>{};
}

llvm::SmallVector<mlir::Value> TanhOp::buildTangent(
    mlir::OpBuilder &builder, mlir::ValueRange tangents) {
  if (tangents.size() != 1 || !tangents[0])
    return {};
  mlir::Type type = getResult().getType();
  mlir::Value one = buildFloatSplat(builder, getLoc(), type, 1.0);
  if (!one)
    return {};
  mlir::Value squared =
      builder.create<MulOp>(getLoc(), type, getY(), getY()).getResult();
  mlir::Value slope =
      builder.create<SubOp>(getLoc(), type, one, squared).getResult();
  return {builder.create<MulOp>(getLoc(), type, tangents[0], slope).getResult()};
}

llvm::SmallVector<mlir::Value> SigmoidOp::buildTangent(
    mlir::OpBuilder &builder, mlir::ValueRange tangents) {
  if (tangents.size() != 1 || !tangents[0])
    return {};
  mlir::Type type = getResult().getType();
  mlir::Value one = buildFloatSplat(builder, getLoc(), type, 1.0);
  if (!one)
    return {};
  mlir::Value complement =
      builder.create<SubOp>(getLoc(), type, one, getY()).getResult();
  mlir::Value slope =
      builder.create<MulOp>(getLoc(), type, getY(), complement).getResult();
  return {builder.create<MulOp>(getLoc(), type, tangents[0], slope).getResult()};
}

#define TESSERA_UNARY_LINEAR_TANGENT(OPNAME)                                \
  llvm::SmallVector<mlir::Value> OPNAME::buildTangent(                      \
      mlir::OpBuilder &builder, mlir::ValueRange tangents) {                \
    if (tangents.size() != 1 || !tangents[0])                               \
      return {};                                                             \
    return {cloneUnaryTangent(getOperation(), tangents[0], builder)};        \
  }

TESSERA_UNARY_LINEAR_TANGENT(TransposeOp)
TESSERA_UNARY_LINEAR_TANGENT(ReshapeOp)
TESSERA_UNARY_LINEAR_TANGENT(SqueezeOp)
TESSERA_UNARY_LINEAR_TANGENT(UnsqueezeOp)
TESSERA_UNARY_LINEAR_TANGENT(ExpandOp)
TESSERA_UNARY_LINEAR_TANGENT(BroadcastOp)
TESSERA_UNARY_LINEAR_TANGENT(PermuteOp)
TESSERA_UNARY_LINEAR_TANGENT(FlattenOp)
TESSERA_UNARY_LINEAR_TANGENT(ViewOp)
TESSERA_UNARY_LINEAR_TANGENT(FFTOp)
TESSERA_UNARY_LINEAR_TANGENT(IFFTOp)
TESSERA_UNARY_LINEAR_TANGENT(RFFTOp)
TESSERA_UNARY_LINEAR_TANGENT(IRFFTOp)
TESSERA_UNARY_LINEAR_TANGENT(DCTOp)

#undef TESSERA_UNARY_LINEAR_TANGENT

#include "Tessera/TangentInterface.cpp.inc"
}  // namespace tessera
