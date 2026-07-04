//===- EpilogueFusion.h - Shared matmul-epilogue fusion body ---*- C++ -*-===//
//
// Workstream A2c (COMPILER_REFACTOR_PLAN §3, Workstream A). After A2a (shared
// emit) and A2b (shared chain walk), the three matmul->epilogue fusion passes
// (matmul->softmax / ->gelu / ->rmsnorm) have a near-identical match+emit body:
// rank-2 + dtype gate, walk to the matmul, validate its rank/dtype/static
// shape/K/N<=256, stamp the `synth_matmul_epilogue` descriptor, and emit one
// runtime call. Collapse it into one data-driven body; each pass becomes a thin
// RewritePattern that supplies its `EpilogueFusionSpec`.
//
//===----------------------------------------------------------------------===//

#ifndef TESSERA_TARGET_APPLE_EPILOGUEFUSION_H
#define TESSERA_TARGET_APPLE_EPILOGUEFUSION_H

#include "Tessera/Common/Lowering.h"
#include "Tessera/Target/Apple/FusionChainUtils.h"
#include "mlir/IR/BuiltinTypes.h"

namespace tessera {
namespace apple {

/// The per-epilogue data the shared body varies on.  The runtime kernel is the
/// uniform synthesized epilogue (`synth_matmul_epilogue`, (A,B,O,M,N,K) ABI) for
/// all three; only the epilogue label + dtype/axis/eps policy differ.
struct EpilogueFusionSpec {
  ::mlir::StringRef epilogueLabel;   // "softmax" / "gelu" / "rmsnorm"
  ::mlir::StringRef intentKernel;    // "matmul_softmax" / "matmul_gelu" / ...
  ::mlir::StringRef synthSymbol;     // tessera_apple_gpu_synth_matmul_epilogue_f32
  bool allowHalfPrecision = false;   // softmax also accepts f16/bf16
  bool requireAxisMinusOne = false;  // softmax: reduction axis must be -1
  bool hasEps = false;               // rmsnorm carries an eps
  float defaultEps = 0.0f;           // rmsnorm variant default (op attr overrides)
};

/// Fuse a `matmul -> <epilogue>` chain (rewriting at the epilogue `op`) into one
/// Apple GPU synthesized-epilogue runtime call.  Returns success after the
/// fusion, or a match-failure when the chain/constraints don't hold.  Shared by
/// the softmax / gelu / rmsnorm passes; emit order is byte-identical to the
/// previously hand-written bodies via `common::emitFusionCall`.
inline ::mlir::LogicalResult lowerMatmulEpilogueFusion(
    ::mlir::PatternRewriter &rewriter, ::mlir::Operation *op,
    const EpilogueFusionSpec &spec) {
  using ::mlir::dyn_cast;
  using ::mlir::failed;
  using ::mlir::failure;
  using ::mlir::FloatAttr;
  using ::mlir::IntegerAttr;
  using ::mlir::Location;
  using ::mlir::MemRefType;
  using ::mlir::ModuleOp;
  using ::mlir::NamedAttribute;
  using ::mlir::Operation;
  using ::mlir::RankedTensorType;
  using ::mlir::success;
  using ::mlir::Type;
  using ::mlir::Value;

  if (op->getNumOperands() < 1)
    return failure();
  Value in = op->getOperand(0);
  bool descriptorDriven = fusionDescriptorDriven(op, spec.intentKernel);

  auto ty = dyn_cast<RankedTensorType>(in.getType());
  if (!ty || ty.getRank() != 2)
    return rewriter.notifyMatchFailure(op, "epilogue fusion: rank-2 only");
  Type elem = ty.getElementType();
  bool dtypeOk =
      elem.isF32() ||
      (spec.allowHalfPrecision && (elem.isF16() || elem.isBF16()));
  if (!dtypeOk)
    return rewriter.notifyMatchFailure(op, "epilogue fusion: dtype not supported");
  if (spec.requireAxisMinusOne) {
    int64_t axis = -1;
    if (auto attr = op->getAttrOfType<IntegerAttr>("axis"))
      axis = attr.getInt();
    if (axis != -1 && axis != ty.getRank() - 1)
      return rewriter.notifyMatchFailure(op, "epilogue fusion: axis must be -1");
  }

  // The epilogue input must be the result of a matmul, with no other uses.
  auto matmulOr =
      walkChainProducer(rewriter, op, in, "tessera.matmul", descriptorDriven);
  if (failed(matmulOr))
    return failure();
  Operation *matmulOp = *matmulOr;
  if (matmulOp->getNumOperands() < 2)
    return failure();
  Value lhs = matmulOp->getOperand(0);
  Value rhs = matmulOp->getOperand(1);

  auto lhsTy = dyn_cast<RankedTensorType>(lhs.getType());
  auto rhsTy = dyn_cast<RankedTensorType>(rhs.getType());
  if (!lhsTy || !rhsTy || lhsTy.getRank() != 2 || rhsTy.getRank() != 2)
    return rewriter.notifyMatchFailure(op, "epilogue fusion: matmul inputs not rank-2");
  if (lhsTy.getElementType() != elem || rhsTy.getElementType() != elem)
    return rewriter.notifyMatchFailure(op, "epilogue fusion: matmul dtype must match");
  if (lhsTy.isDynamicDim(0) || lhsTy.isDynamicDim(1) || rhsTy.isDynamicDim(0) ||
      rhsTy.isDynamicDim(1))
    return rewriter.notifyMatchFailure(op, "epilogue fusion: requires static shapes");

  int64_t M = lhsTy.getDimSize(0);
  int64_t K = lhsTy.getDimSize(1);
  int64_t N = rhsTy.getDimSize(1);
  if (rhsTy.getDimSize(0) != K)
    return rewriter.notifyMatchFailure(op, "epilogue fusion: matmul K mismatch");
  if (N > 256)
    return rewriter.notifyMatchFailure(op, "epilogue fusion: GPU kernel limited to N <= 256");

  Location loc = op->getLoc();
  ModuleOp mod = op->getParentOfType<ModuleOp>();
  auto aMemTy = MemRefType::get({M, K}, elem);
  auto bMemTy = MemRefType::get({K, N}, elem);
  auto oMemTy = MemRefType::get({M, N}, elem);
  auto outTensorTy = RankedTensorType::get({M, N}, elem);

  ::llvm::SmallVector<NamedAttribute> desc;
  desc.push_back(rewriter.getNamedAttr(
      "tessera.fusion.kernel",
      rewriter.getStringAttr("synth_matmul_epilogue")));
  desc.push_back(rewriter.getNamedAttr(
      "tessera.fusion.epilogue", rewriter.getStringAttr(spec.epilogueLabel)));
  if (spec.hasEps) {
    float eps = spec.defaultEps;
    if (auto attr = op->getAttrOfType<FloatAttr>("eps"))
      eps = static_cast<float>(attr.getValueAsDouble());
    desc.push_back(rewriter.getNamedAttr("tessera.fusion.eps",
                                         rewriter.getF32FloatAttr(eps)));
  }
  desc.push_back(rewriter.getNamedAttr(
      "tessera.fusion.source",
      rewriter.getStringAttr(descriptorDriven ? "descriptor" : "rediscovered")));

  rewriter.setInsertionPoint(matmulOp);
  Value result = tessera::common::emitFusionCall(
      rewriter, loc, mod, spec.synthSymbol, {{lhs, aMemTy}, {rhs, bMemTy}},
      oMemTy, outTensorTy, {M, N, K}, desc);

  rewriter.replaceOp(op, result);
  rewriter.eraseOp(matmulOp);
  return success();
}

} // namespace apple
} // namespace tessera

#endif // TESSERA_TARGET_APPLE_EPILOGUEFUSION_H
