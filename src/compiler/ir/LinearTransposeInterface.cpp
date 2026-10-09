#include "Tessera/IR/ScaledBatchContract.h"
#include "Tessera/IR/TransposeUtils.h"
//===- LinearTransposeInterface.cpp - Graph IR linear transpose -*- C++ -*-===//

#include "Tessera/IR/TesseraOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include <functional>
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


bool ScaledMatmulOp::isLinearInOperand(unsigned index) {
  // Typed low-precision matrix storage has no implicit straight-through rule.
  return index == 2 || index == 3;
}

llvm::SmallVector<mlir::Value> ScaledMatmulOp::buildLinearTranspose(
    mlir::OpBuilder &builder, mlir::ValueRange outputCotangents) {
  using namespace mlir;
  auto result = dyn_cast<RankedTensorType>(getResult().getType());
  auto a = dyn_cast<RankedTensorType>(getLhs().getType());
  auto b = dyn_cast<RankedTensorType>(getRhs().getType());
  auto sa = dyn_cast<RankedTensorType>(getLhsScale().getType());
  auto sb = dyn_cast<RankedTensorType>(getRhsScale().getType());
  auto layout = getScaleLayoutAttr();
  auto block = layout ? layout.getAs<ArrayAttr>("block") : ArrayAttr{};
  auto format = layout ? layout.getAs<StringAttr>("format") : StringAttr{};
  auto granularity = layout ? layout.getAs<StringAttr>("granularity") : StringAttr{};
  auto policy = getNumericPolicyAttr();
  auto mode = policy ? policy.getAs<StringAttr>("execution_mode") : StringAttr{};
  auto accum = policy ? policy.getAs<StringAttr>("accum") : StringAttr{};
  auto batch = getOperation()->getAttrOfType<StringAttr>("batching");
  StringRef batching = batch ? batch.getValue() : "";
  bool broadcast = batching == "broadcast";
  bool mappedA = batching == "shared_rhs_rows" || batching == "independent_rhs";
  bool mappedB = batching == "shared_lhs" || batching == "independent_rhs";
  if (outputCotangents.size() != 1 || !outputCotangents[0] ||
      !result || !a || !b || !sa || !sb ||
      !result.hasStaticShape() || !a.hasStaticShape() ||
      !b.hasStaticShape() || !sa.hasStaticShape() || !sb.hasStaticShape() ||
      result.getRank() < 2 ||
      !result.getElementType().isF32() || !sa.getElementType().isF32() ||
      !sb.getElementType().isF32() ||
      !isa<Float8E4M3FNType>(a.getElementType()) ||
      !isa<Float8E4M3FNType>(b.getElementType()) ||
      getOperation()->hasAttr("physical_contract") ||
      !format || format.getValue() != "fp32" ||
      !granularity || granularity.getValue() != "block" ||
      !mode || mode.getValue() != "exact_per_block" ||
      !accum || accum.getValue() != "fp32" ||
      !block || block.size() != 2 ||
      !isa<IntegerAttr>(block[0]) || !isa<IntegerAttr>(block[1]) ||
      (!batching.empty() && !mappedA && !mappedB && !broadcast) ||
      outputCotangents[0].getType() != result)
    return {};
  int64_t sn = cast<IntegerAttr>(block[0]).getInt();
  int64_t sk = cast<IntegerAttr>(block[1]).getInt();
  int64_t prefix = result.getRank() - 2;
  int64_t m = result.getDimSize(prefix), n = result.getDimSize(prefix + 1);
  int64_t k = a.getDimSize(a.getRank() - (getTransposeA() ? 2 : 1));
  if (sn <= 0 || sk <= 0 || m <= 0 || n <= 0 || k <= 0 ||
      (broadcast ? !hasExactScaledBroadcastPrefix({a, b, sa, sb}, result)
                 : (a.getRank() != (mappedA ? prefix : 0) + 2 ||
                    b.getRank() != (mappedB ? prefix : 0) + 2 ||
                    sa.getRank() != a.getRank() || sb.getRank() != b.getRank())) ||
      (prefix != 0 && batching.empty()))
    return {};
  auto matches = [&](RankedTensorType type, bool mapped, int64_t x, int64_t y) {
    if (type.getDimSize(type.getRank()-2) != x ||
        type.getDimSize(type.getRank()-1) != y) return false;
    for (int64_t i = 0; !broadcast && mapped && i < prefix; ++i)
      if (type.getDimSize(i) != result.getDimSize(i)) return false;
    return true;
  };
  if (!matches(a, mappedA, getTransposeA() ? k : m, getTransposeA() ? m : k) ||
      !matches(b, mappedB, getTransposeB() ? n : k, getTransposeB() ? k : n) ||
      !matches(sa, mappedA, m, (k+sk-1)/sk) ||
      !matches(sb, mappedB, (k+sk-1)/sk, (n+sn-1)/sn))
    return {};

  // Each scale gradient is its own structured reduction. Shared operands
  // reduce every logical batch axis; mapped operands keep the exact prefix.
  // The ragged final K/N groups stop at the original logical extent.
  auto loc = getLoc();
  auto makeGradient = [&](bool lhs) -> Value {
    auto type = lhs ? sa : sb;
    // Scale storage, not matrix mapping, owns gradient coordinates.
    int64_t scalePrefix = type.getRank() - 2;
    SmallVector<int64_t> reductionAxes;
    for (int64_t axis = 0; axis < prefix; ++axis) {
      int64_t ownAxis = axis - (prefix - scalePrefix);
      if (ownAxis < 0 || type.getDimSize(ownAxis) == 1)
        reductionAxes.push_back(axis);
    }
    auto generated = tensor::GenerateOp::create(
        builder, loc, type, ValueRange{},
        [&](OpBuilder &g, Location l, ValueRange indices) {
          auto constant = [&](int64_t v) -> Value {
            return arith::ConstantIndexOp::create(g, l, v);
          };
          Value zero = arith::ConstantOp::create(g, l, g.getF32FloatAttr(0));
          Value c0 = constant(0), c1 = constant(1);
          SmallVector<Value> batchIndices(prefix, c0);
          for (int64_t axis = 0; axis < scalePrefix; ++axis)
            if (type.getDimSize(axis) != 1)
              batchIndices[prefix - scalePrefix + axis] = indices[axis];
          auto operandPrefix = [&](RankedTensorType operand) {
            SmallVector<Value> coordinates;
            int64_t ownPrefix = operand.getRank() - 2;
            for (int64_t axis = 0; axis < ownPrefix; ++axis)
              coordinates.push_back(operand.getDimSize(axis) == 1 ? c0
                  : batchIndices[prefix - ownPrefix + axis]);
            return coordinates;
          };
          int64_t offset = scalePrefix;
          Value group = indices[offset + (lhs ? 1 : 0)];
          Value row = lhs ? indices[offset] : Value{};
          Value column;
          Value klo = arith::MulIOp::create(g, l, group, constant(sk));
          Value khi = arith::MinUIOp::create(g, l, constant(k),
              arith::AddIOp::create(g, l, klo, constant(sk)));
          Value nlo = lhs ? c0 : arith::MulIOp::create(
              g, l, indices[offset+1], constant(sn));
          Value nhi = lhs ? constant(n) : arith::MinUIOp::create(g, l, constant(n),
              arith::AddIOp::create(g, l, nlo, constant(sn)));
          std::function<Value(OpBuilder &, int64_t, Value)> reduce;
          reduce = [&](OpBuilder &r, int64_t axis, Value seed) -> Value {
            // Shared batch axes, then M (RHS gradient), N, and group-local K.
            int64_t batchLoops = reductionAxes.size();
            int64_t rowLoops = lhs ? 0 : 1;
            int64_t nAxis = batchLoops + rowLoops;
            int64_t kAxis = nAxis + 1;
            Value lower = c0, upper;
            if (axis < batchLoops) {
              upper = arith::ConstantIndexOp::create(r, l, result.getDimSize(reductionAxes[axis]));
            } else if (!lhs && axis == batchLoops) {
              upper = arith::ConstantIndexOp::create(r, l, m);
            } else if (axis == nAxis) {
              lower = nlo; upper = nhi;
            } else {
              lower = klo; upper = khi;
            }
            auto loop = scf::ForOp::create(r, l, lower, upper, c1,
                ValueRange{seed}, [&](OpBuilder &body, Location at,
                                      Value iv, ValueRange carried) {
                  if (axis < batchLoops) batchIndices[reductionAxes[axis]] = iv;
                  else if (!lhs && axis == batchLoops) row = iv;
                  else if (axis == nAxis) column = iv;
                  if (axis < nAxis) {
                    scf::YieldOp::create(body, at, reduce(body, axis+1, carried[0]));
                    return;
                  }
                  if (axis == nAxis) {
                    // Preserve the primal's group-local dot before either
                    // scale/cotangent multiplication and spatial/batch join.
                    Value partial = reduce(body, kAxis, zero);
                    SmallVector<Value> si = operandPrefix(sa);
                    SmallVector<Value> ti = operandPrefix(sb);
                    SmallVector<Value> oi(batchIndices);
                    si.append({row, group});
                    Value blockColumn = arith::DivUIOp::create(body, at, column,
                        arith::ConstantIndexOp::create(body, at, sn));
                    ti.append({group, blockColumn});
                    oi.append({row, column});
                    Value coefficient = tensor::ExtractOp::create(body, at,
                        lhs ? getRhsScale() : getLhsScale(), lhs ? ti : si);
                    Value dy = tensor::ExtractOp::create(body, at, outputCotangents[0], oi);
                    Value term = arith::MulFOp::create(body, at, partial, coefficient);
                    term = arith::MulFOp::create(body, at, term, dy);
                    scf::YieldOp::create(body, at,
                        arith::AddFOp::create(body, at, carried[0], term).getResult());
                    return;
                  }
                  SmallVector<Value> ai = operandPrefix(a);
                  SmallVector<Value> bi = operandPrefix(b);
                  ai.push_back(getTransposeA() ? iv : row);
                  ai.push_back(getTransposeA() ? row : iv);
                  bi.push_back(getTransposeB() ? column : iv);
                  bi.push_back(getTransposeB() ? iv : column);
                  Value av = tensor::ExtractOp::create(body, at, getLhs(), ai);
                  Value bv = tensor::ExtractOp::create(body, at, getRhs(), bi);
                  av = arith::ExtFOp::create(body, at, body.getF32Type(), av);
                  bv = arith::ExtFOp::create(body, at, body.getF32Type(), bv);
                  Value term = arith::MulFOp::create(body, at, av, bv);
                  scf::YieldOp::create(body, at,
                      arith::AddFOp::create(body, at, carried[0], term).getResult());
                });
            return loop.getResult(0);
          };
          tensor::YieldOp::create(g, l, reduce(g, 0, zero));
        });
    generated->setAttr("tessera.autodiff.scale_adjoint",
                       builder.getStringAttr(lhs ? "lhs_scale" : "rhs_scale"));
    return generated.getResult();
  };
  Value dsa = makeGradient(true), dsb = makeGradient(false);
  return {Value{}, Value{}, dsa, dsb};
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

  auto permutation = transposePermutation(getOperation());
  // An unranked default reverse remains self-adjoint. Explicit or ranked
  // axes must have a verified inverse before constructing the adjoint.
  if (!permutation && (mlir::isa<mlir::RankedTensorType>(getX().getType()) ||
                       (*this)->hasAttr("permutation")))
    return {mlir::Value()};
  auto grad = builder.create<TransposeOp>(
      getLoc(), getX().getType(), outputCotangents[0]);
  // Carry layout/policy obligations rather than erasing them in AD.
  grad->setAttrs((*this)->getAttrs());
  // The adjoint consumes primal output axes and restores primal input axes.
  for (auto pair : {std::pair{"tessera.dim_names_in", "tessera.dim_names_out"},
                    std::pair{"tessera.dim_names_out", "tessera.dim_names_in"}}) {
    grad->removeAttr(pair.first);
    if (auto names = (*this)->getAttr(pair.second))
      grad->setAttr(pair.first, names);
  }
  if (permutation) {
    llvm::SmallVector<int64_t> inverse(permutation->size());
    for (size_t outputAxis = 0; outputAxis < permutation->size(); ++outputAxis)
      inverse[(*permutation)[outputAxis]] = outputAxis;
    grad->setAttr("permutation", builder.getDenseI64ArrayAttr(inverse));
  }
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
