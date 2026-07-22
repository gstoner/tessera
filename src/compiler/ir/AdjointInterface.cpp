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

// These tables are both implementation policy and the generated ledger's
// source for kind-aware `tessera.reduce` classification. Keep their spelling
// stable unless autodiff_ledger.py is updated in the same change.
static constexpr llvm::StringLiteral kReduceNativeAdjointKinds[] = {
    "sum", "mean"};
static constexpr llvm::StringLiteral kReducePlaceholderAdjointKinds[] = {
    "max", "min"};

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

// Broadcast is inverted by summing every expanded dimension. Reduce in
// descending axis order so each remaining axis keeps its original index, then
// reshape to restore the input's explicit singleton dimensions. Dynamic shapes
// cannot prove which dimensions expanded at compile time and retain the Python
// VJP fallback.
llvm::SmallVector<mlir::Value> BroadcastOp::buildAdjoint(
    mlir::OpBuilder &builder, mlir::ValueRange outputCotangents) {
  if (outputCotangents.size() != 1 || !outputCotangents[0])
    return {mlir::Value()};

  auto inTy = mlir::dyn_cast<mlir::RankedTensorType>(getX().getType());
  auto outTy = mlir::dyn_cast<mlir::RankedTensorType>(getY().getType());
  mlir::Value dy = outputCotangents[0];
  if (!inTy || !outTy || !inTy.hasStaticShape() || !outTy.hasStaticShape())
    return placeholderAdjoint(builder, getLoc(), getX().getType(), "broadcast",
                              dy, getX());

  llvm::SmallVector<int64_t> reduceAxes;
  int64_t offset = outTy.getRank() - inTy.getRank();
  for (int64_t axis = 0; axis < offset; ++axis)
    reduceAxes.push_back(axis);
  for (int64_t axis = 0; axis < inTy.getRank(); ++axis) {
    int64_t outAxis = axis + offset;
    if (inTy.getDimSize(axis) == 1 && outTy.getDimSize(outAxis) != 1)
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

// Single-axis sum/mean reduction. The output cotangent first regains the
// removed singleton axis and then broadcasts to the input shape. Mean adds the
// reciprocal static extent. Dynamic mean retains the reference fallback until
// Graph IR owns a runtime scalar-from-dim primitive; sum is shape-polymorphic.
// Max/min retain the Python VJP because their equal-tie distribution policy
// requires comparisons and a second reduction.
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

  if (kind == "mean" &&
      (!inTy.hasStaticShape() ||
       mlir::ShapedType::isDynamic(inTy.getDimSize(axis))))
    return placeholderAdjoint(builder, getLoc(), getInput().getType(), kind,
                              dy, getInput());

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
  // Emit a custom_adjoint_call placeholder — the closed-form layernorm
  // adjoint requires several reductions that are awkward in pure ODS.
  // Routing through the runtime's Python VJP keeps semantics identical to
  // the numpy tape and avoids a second source of truth.
  auto loc = getLoc();
  auto callOp = builder.create<CustomAdjointCallOp>(
      loc, llvm::SmallVector<mlir::Type>{getX().getType()},
      builder.getStringAttr("layer_norm"),
      mlir::ValueRange{outputCotangents[0], getX()});
  return {callOp.getResult(0)};
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
  mlir::Value s = builder
                      .create<SoftmaxOp>(loc, inTy, getX(), getAxisAttr(),
                                         getNumericPolicyAttr())
                      .getY();
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

POINTWISE_BUILD_ADJOINT(ReluOp, "relu")
POINTWISE_BUILD_ADJOINT(SinOp, "sin")
// Tier-1 MPSGraph-lane ops (2026-05-29). Each has a Python VJP the runtime
// resolves via the custom_adjoint_call placeholder keyed by name.
POINTWISE_BUILD_ADJOINT(SoftplusOp, "softplus")
POINTWISE_BUILD_ADJOINT(RmsNormOp, "rmsnorm")
POINTWISE_BUILD_ADJOINT(LogSoftmaxOp, "log_softmax")

#undef POINTWISE_BUILD_ADJOINT

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
  mlir::Value t = builder.create<TanhOp>(loc, ty, getX()).getResult();
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
  mlir::Value s = builder.create<SigmoidOp>(loc, ty, getX()).getResult();
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
