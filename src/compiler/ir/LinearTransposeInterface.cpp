//===- LinearTransposeInterface.cpp - Graph IR linear transpose -*- C++ -*-===//

#include "Tessera/IR/TesseraOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"

namespace tessera {

static mlir::StringAttr transposeFFTNormalization(mlir::OpBuilder &builder,
                                                  mlir::Operation *op) {
  auto normalization = op->getAttrOfType<mlir::StringAttr>("normalization");
  llvm::StringRef value = normalization ? normalization.getValue() : "backward";
  if (value == "backward")
    value = "forward";
  else if (value == "forward")
    value = "backward";
  return builder.getStringAttr(value);
}

static mlir::Value buildSpectralTranspose(
    mlir::Operation *source, llvm::StringRef targetName,
    mlir::Type resultType, mlir::Value cotangent, mlir::OpBuilder &builder,
    llvm::StringRef hermitianWeight) {
  mlir::OperationState state(source->getLoc(), targetName);
  state.addOperands(cotangent);
  state.addTypes(resultType);
  for (llvm::StringRef name : {"axis", "logical_length", "spectrum_layout"})
    if (mlir::Attribute attr = source->getAttr(name))
      state.addAttribute(name, attr);
  state.addAttribute("normalization", transposeFFTNormalization(builder, source));
  state.addAttribute("hermitian_weight",
                     builder.getStringAttr(hermitianWeight));
  return builder.create(state)->getResult(0);
}

static llvm::SmallVector<mlir::Value> transposeImplicitBroadcast(
    mlir::Operation *op, mlir::Value input, mlir::Value output,
    mlir::OpBuilder &builder, mlir::ValueRange outputCotangents,
    llvm::StringRef fallbackKey) {
  if (outputCotangents.size() != 1 || !outputCotangents[0])
    return {mlir::Value()};

  auto inTy = mlir::dyn_cast<mlir::RankedTensorType>(input.getType());
  auto outTy = mlir::dyn_cast<mlir::RankedTensorType>(output.getType());
  mlir::Value dy = outputCotangents[0];
  if (!inTy || !outTy)
    return {mlir::Value()};

  llvm::SmallVector<int64_t> reduceAxes;
  int64_t offset = outTy.getRank() - inTy.getRank();
  for (int64_t axis = 0; axis < offset; ++axis)
    reduceAxes.push_back(axis);
  for (int64_t axis = 0; axis < inTy.getRank(); ++axis) {
    int64_t outAxis = axis + offset;
    int64_t inputExtent = inTy.getDimSize(axis);
    int64_t outputExtent = outTy.getDimSize(outAxis);
    if (mlir::ShapedType::isDynamic(inputExtent) &&
        !mlir::ShapedType::isDynamic(outputExtent) && outputExtent != 1) {
      auto fallback = builder.create<CustomAdjointCallOp>(
          op->getLoc(), llvm::SmallVector<mlir::Type>{input.getType()},
          builder.getStringAttr(fallbackKey), mlir::ValueRange{dy, input});
      return {fallback.getResult(0)};
    }
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
               .create<ReduceOp>(op->getLoc(), reducedTy, grad,
                                 builder.getStringAttr("sum"),
                                 builder.getI64IntegerAttr(axis))
               .getResult();
  }
  if (grad.getType() != inTy)
    grad = builder.create<ReshapeOp>(op->getLoc(), inTy, grad).getY();
  return {grad};
}

llvm::SmallVector<mlir::Value> MatmulOp::buildLinearTranspose(
    mlir::OpBuilder &builder, mlir::ValueRange outputCotangents) {
  if (outputCotangents.size() != 1 || !outputCotangents[0])
    return {mlir::Value(), mlir::Value()};

  mlir::Value dy = outputCotangents[0];
  mlir::Value dLhs;
  mlir::Value dRhs;
  if (!getTransposeA()) {
    dLhs = builder
               .create<MatmulOp>(
                   getLoc(), getLhs().getType(), dy, getRhs(),
                   mlir::ValueRange{}, mlir::IntegerAttr(), nullptr,
                   builder.getBoolAttr(false),
                   builder.getBoolAttr(!getTransposeB()))
               .getResult();
  } else {
    dLhs = builder
               .create<MatmulOp>(
                   getLoc(), getLhs().getType(), getRhs(), dy,
                   mlir::ValueRange{}, mlir::IntegerAttr(), nullptr,
                   builder.getBoolAttr(getTransposeB()),
                   builder.getBoolAttr(true))
               .getResult();
  }

  if (!getTransposeB()) {
    dRhs = builder
               .create<MatmulOp>(
                   getLoc(), getRhs().getType(), getLhs(), dy,
                   mlir::ValueRange{}, mlir::IntegerAttr(), nullptr,
                   builder.getBoolAttr(!getTransposeA()),
                   builder.getBoolAttr(false))
               .getResult();
  } else {
    dRhs = builder
               .create<MatmulOp>(
                   getLoc(), getRhs().getType(), dy, getLhs(),
                   mlir::ValueRange{}, mlir::IntegerAttr(), nullptr,
                   builder.getBoolAttr(true),
                   builder.getBoolAttr(getTransposeA()))
               .getResult();
  }
  return {dLhs, dRhs};
}
#include "Tessera/LinearTransposeInterface.cpp.inc"
}  // namespace tessera

namespace tessera {

llvm::SmallVector<mlir::Value> TransposeOp::buildLinearTranspose(
    mlir::OpBuilder &builder, mlir::ValueRange outputCotangents) {
  if (outputCotangents.size() != 1 || !outputCotangents[0])
    return {mlir::Value()};

  // Graph IR transpose currently means reverse all dimensions, so it is
  // self-adjoint.  An explicit permutation attribute must carry and invert
  // that permutation here when the Graph contract grows one.
  auto grad = builder.create<TransposeOp>(
      getLoc(), getX().getType(), outputCotangents[0]);
  return {grad.getY()};
}

llvm::SmallVector<mlir::Value> ReshapeOp::buildLinearTranspose(
    mlir::OpBuilder &builder, mlir::ValueRange outputCotangents) {
  if (outputCotangents.size() != 1 || !outputCotangents[0])
    return {mlir::Value()};

  auto grad = builder.create<ReshapeOp>(
      getLoc(), getX().getType(), outputCotangents[0]);
  return {grad.getY()};
}

llvm::SmallVector<mlir::Value> SqueezeOp::buildLinearTranspose(
    mlir::OpBuilder &builder, mlir::ValueRange outputCotangents) {
  if (outputCotangents.size() != 1 || !outputCotangents[0])
    return {mlir::Value()};
  auto grad = builder.create<UnsqueezeOp>(
      getLoc(), getX().getType(), outputCotangents[0]);
  if (auto axes = (*this)->getAttr("axes"))
    grad->setAttr("axes", axes);
  return {grad.getY()};
}

llvm::SmallVector<mlir::Value> UnsqueezeOp::buildLinearTranspose(
    mlir::OpBuilder &builder, mlir::ValueRange outputCotangents) {
  if (outputCotangents.size() != 1 || !outputCotangents[0])
    return {mlir::Value()};
  auto grad = builder.create<SqueezeOp>(
      getLoc(), getX().getType(), outputCotangents[0]);
  if (auto axes = (*this)->getAttr("axes"))
    grad->setAttr("axes", axes);
  return {grad.getY()};
}

llvm::SmallVector<mlir::Value> FlattenOp::buildLinearTranspose(
    mlir::OpBuilder &builder, mlir::ValueRange outputCotangents) {
  if (outputCotangents.size() != 1 || !outputCotangents[0])
    return {mlir::Value()};
  return {builder
              .create<ReshapeOp>(getLoc(), getX().getType(),
                                 outputCotangents[0])
              .getY()};
}

llvm::SmallVector<mlir::Value> ViewOp::buildLinearTranspose(
    mlir::OpBuilder &builder, mlir::ValueRange outputCotangents) {
  if (outputCotangents.size() != 1 || !outputCotangents[0])
    return {mlir::Value()};
  return {builder
              .create<ReshapeOp>(getLoc(), getX().getType(),
                                 outputCotangents[0])
              .getY()};
}

llvm::SmallVector<mlir::Value> PermuteOp::buildLinearTranspose(
    mlir::OpBuilder &builder, mlir::ValueRange outputCotangents) {
  if (outputCotangents.size() != 1 || !outputCotangents[0])
    return {mlir::Value()};
  auto perm = (*this)->getAttrOfType<mlir::ArrayAttr>("perm");
  if (!perm)
    return {mlir::Value()};
  llvm::SmallVector<mlir::Attribute> inverse(perm.size());
  for (auto [inputAxis, outputAxisAttr] : llvm::enumerate(perm)) {
    auto outputAxis = mlir::cast<mlir::IntegerAttr>(outputAxisAttr).getInt();
    inverse[outputAxis] = builder.getI64IntegerAttr(inputAxis);
  }
  auto grad = builder.create<PermuteOp>(
      getLoc(), getX().getType(), outputCotangents[0]);
  grad->setAttr("perm", builder.getArrayAttr(inverse));
  return {grad.getY()};
}

llvm::SmallVector<mlir::Value> ExpandOp::buildLinearTranspose(
    mlir::OpBuilder &builder, mlir::ValueRange outputCotangents) {
  return transposeImplicitBroadcast(getOperation(), getX(), getY(), builder,
                                    outputCotangents, "expand");
}

llvm::SmallVector<mlir::Value> BroadcastOp::buildLinearTranspose(
    mlir::OpBuilder &builder, mlir::ValueRange outputCotangents) {
  return transposeImplicitBroadcast(getOperation(), getX(), getY(), builder,
                                    outputCotangents, "broadcast");
}

llvm::SmallVector<mlir::Value> FFTOp::buildLinearTranspose(
    mlir::OpBuilder &builder, mlir::ValueRange outputCotangents) {
  if (outputCotangents.size() != 1 || !outputCotangents[0])
    return {mlir::Value()};
  auto inputType = mlir::dyn_cast<mlir::ShapedType>(getX().getType());
  if (!inputType ||
      !mlir::isa<mlir::ComplexType>(inputType.getElementType())) {
    emitError("real-input full FFT has no complex-linear transpose; use rfft so Hermitian weighting is explicit");
    return {};
  }
  return {buildSpectralTranspose(getOperation(), "tessera.ifft",
                                 getX().getType(), outputCotangents[0], builder,
                                 "none")};
}

llvm::SmallVector<mlir::Value> IFFTOp::buildLinearTranspose(
    mlir::OpBuilder &builder, mlir::ValueRange outputCotangents) {
  if (outputCotangents.size() != 1 || !outputCotangents[0])
    return {mlir::Value()};
  return {buildSpectralTranspose(getOperation(), "tessera.fft",
                                 getX().getType(), outputCotangents[0], builder,
                                 "none")};
}

llvm::SmallVector<mlir::Value> RFFTOp::buildLinearTranspose(
    mlir::OpBuilder &builder, mlir::ValueRange outputCotangents) {
  if (outputCotangents.size() != 1 || !outputCotangents[0])
    return {mlir::Value()};
  // The real-domain transpose uses the explicit half spectrum. DC and an
  // even-length Nyquist bin retain unit weight; each interior complex bin is
  // halved before c2r because Hermitian reconstruction contributes it twice.
  return {buildSpectralTranspose(getOperation(), "tessera.irfft",
                                 getX().getType(), outputCotangents[0], builder,
                                 "half_interior")};
}

llvm::SmallVector<mlir::Value> IRFFTOp::buildLinearTranspose(
    mlir::OpBuilder &builder, mlir::ValueRange outputCotangents) {
  if (outputCotangents.size() != 1 || !outputCotangents[0])
    return {mlir::Value()};
  // Conversely c2r's transpose doubles the interior r2c bins while leaving
  // DC and the even-length Nyquist endpoint unchanged.
  return {buildSpectralTranspose(getOperation(), "tessera.rfft",
                                 getX().getType(), outputCotangents[0], builder,
                                 "double_interior")};
}

llvm::SmallVector<mlir::Value> DCTOp::buildLinearTranspose(
    mlir::OpBuilder &builder, mlir::ValueRange outputCotangents) {
  if (outputCotangents.size() != 1 || !outputCotangents[0])
    return {mlir::Value()};
  mlir::OperationState state(getLoc(), "tessera.dct");
  state.addOperands(outputCotangents[0]);
  state.addTypes(getX().getType());
  for (llvm::StringRef name : {"axis", "logical_length", "normalization",
                               "spectrum_layout", "hermitian_weight", "type"})
    if (mlir::Attribute attr = getOperation()->getAttr(name))
      state.addAttribute(name, attr);
  // DCT-I/II/III are not generally symmetric under Tessera's unnormalised
  // convention. Keep basis transposition explicit in the Graph contract;
  // DCT-IV naturally takes the same path and physical implementations may
  // fold the flag away.
  state.addAttribute("transpose_basis", builder.getBoolAttr(!getTransposeBasis()));
  return {builder.create(state)->getResult(0)};
}

}  // namespace tessera
