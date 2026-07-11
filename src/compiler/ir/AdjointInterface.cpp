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
  auto loc = getLoc();
  auto callOp = builder.create<CustomAdjointCallOp>(
      loc, llvm::SmallVector<mlir::Type>{getX().getType()},
      builder.getStringAttr("softmax"),
      mlir::ValueRange{outputCotangents[0], getX()});
  return {callOp.getResult(0)};
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

POINTWISE_BUILD_ADJOINT(GeluOp, "gelu")
POINTWISE_BUILD_ADJOINT(ReluOp, "relu")
POINTWISE_BUILD_ADJOINT(SinOp, "sin")
// Tier-1 MPSGraph-lane ops (2026-05-29). Each has a Python VJP the runtime
// resolves via the custom_adjoint_call placeholder keyed by name.
POINTWISE_BUILD_ADJOINT(SiluOp, "silu")
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

// A dense splat constant of `value` matching a shape-preserving tensor type.
static mlir::Value splatConst(mlir::OpBuilder &builder, mlir::Location loc,
                              mlir::Type ty, double value) {
  auto shaped = mlir::cast<mlir::ShapedType>(ty);
  auto attr = mlir::DenseElementsAttr::get(
      shaped, builder.getFloatAttr(shaped.getElementType(), value));
  return builder.create<mlir::arith::ConstantOp>(loc, attr).getResult();
}

// tanh:  dx = dy · (1 − tanh(x)²)
llvm::SmallVector<mlir::Value> TanhOp::buildAdjoint(
    mlir::OpBuilder &builder, mlir::ValueRange outputCotangents) {
  if (outputCotangents.size() != 1 || !outputCotangents[0])
    return {mlir::Value()};
  auto loc = getLoc();
  mlir::Type ty = getResult().getType();
  mlir::Value dy = outputCotangents[0];
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
  mlir::Value s = builder.create<SigmoidOp>(loc, ty, getX()).getResult();
  mlir::Value one = splatConst(builder, loc, ty, 1.0);
  mlir::Value oneMinusS = builder.create<SubOp>(loc, ty, one, s).getResult();
  mlir::Value sPrime = builder.create<MulOp>(loc, ty, s, oneMinusS).getResult();
  mlir::Value dx = builder.create<MulOp>(loc, ty, dy, sPrime).getResult();
  return {dx};
}

}  // namespace tessera
