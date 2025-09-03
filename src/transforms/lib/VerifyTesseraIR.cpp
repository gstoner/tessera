
#include "Tessera/Transforms/Passes.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LogicalResult.h"

using namespace mlir;

namespace {

static bool isOp(Operation *op, StringRef name) {
  return op && op->getName().getStringRef() == name;
}

static LogicalResult verifyMatmul(Operation *op) {
  if (op->getNumOperands() != 2 || op->getNumResults() != 1)
    return op->emitError("[TESSERA_VFY_MATMUL_ARITY] matmul expects 2 operands and 1 result");

  auto aTy = dyn_cast<RankedTensorType>(op->getOperand(0).getType());
  auto bTy = dyn_cast<RankedTensorType>(op->getOperand(1).getType());
  auto rTy = dyn_cast<RankedTensorType>(op->getResult(0).getType());
  if (!aTy || !bTy || !rTy) return success();

  if (aTy.getElementType() != bTy.getElementType() ||
      aTy.getElementType() != rTy.getElementType())
    return op->emitError("[TESSERA_VFY_MATMUL_ELEM_TYPE_MISMATCH] element types must match");

  if (aTy.getRank() == 2 && bTy.getRank() == 2 && rTy.getRank() == 2) {
    int64_t M = aTy.getDimSize(0);
    int64_t K1 = aTy.getDimSize(1);
    int64_t K2 = bTy.getDimSize(0);
    int64_t N = bTy.getDimSize(1);
    int64_t MR = rTy.getDimSize(0);
    int64_t NR = rTy.getDimSize(1);
    if (K1 != ShapedType::kDynamic && K2 != ShapedType::kDynamic && K1 != K2)
      return op->emitError("[TESSERA_VFY_MATMUL_K] inner K dims must match");
    if (MR != ShapedType::kDynamic && M != ShapedType::kDynamic && MR != M)
      return op->emitError("[TESSERA_VFY_MATMUL_SHAPE] result M must equal lhs M");
    if (NR != ShapedType::kDynamic && N != ShapedType::kDynamic && NR != N)
      return op->emitError("[TESSERA_VFY_MATMUL_SHAPE] result N must equal rhs N");
  }

  if (auto tileK = op->getAttrOfType<IntegerAttr>("tile_k")) {
    if (tileK.getInt() <= 0)
      return op->emitError("[TESSERA_VFY_MATMUL_TILEK] tile_k must be > 0");
  }
  return success();
}

static LogicalResult verifyConv(Operation *op) {
  auto checkPos = [&](StringRef name) -> LogicalResult {
    if (auto arr = op->getAttrOfType<ArrayAttr>(name)) {
      for (Attribute a : arr) {
        auto ia = dyn_cast<IntegerAttr>(a);
        if (!ia || ia.getInt() <= 0)
          return op->emitError("[TESSERA_VFY_CONV_ATTR] ") << name << " must be positive integers";
      }
    }
    return success();
  };
  if (failed(checkPos("strides")) || failed(checkPos("dilations")))
    return failure();
  return success();
}

static LogicalResult verifyFlashAttn(Operation *op) {
  if (auto hd = op->getAttrOfType<IntegerAttr>("head_dim")) {
    if (hd.getInt() <= 0 || hd.getInt() > 256)
      return op->emitError("[TESSERA_VFY_ATTN_HEADDIM] head_dim should be in (0, 256]");
  }
  if (auto p = op->getAttrOfType<FloatAttr>("dropout_p")) {
    double v = p.getValueAsDouble();
    if (v < 0.0 || v >= 1.0)
      return op->emitError("[TESSERA_VFY_ATTN_DROPOUT] dropout_p must be in [0,1)");
  }
  return success();
}

static LogicalResult verifyFusedEpilogue(Operation *op) {
  if (auto ep = op->getAttrOfType<StringAttr>("epilogue")) {
    auto s = ep.getValue();
    if (!(s == "none" || s == "relu" || s == "gelu" || s == "silu"))
      return op->emitError("[TESSERA_VFY_EPILOGUE_KIND] unsupported epilogue kind: ") << s;
  }
  return success();
}

struct VerifyTesseraIRPass
    : public PassWrapper<VerifyTesseraIRPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(VerifyTesseraIRPass)
  void runOnOperation() override {
    ModuleOp m = getOperation();
    LogicalResult anyFailure = success();
    m.walk([&](Operation *op) {
      if (isOp(op, "tessera.matmul"))
        if (failed(verifyMatmul(op))) anyFailure = failure();
      if (isOp(op, "tessera.conv2d_nhwc"))
        if (failed(verifyConv(op))) anyFailure = failure();
      if (isOp(op, "tessera.flash_attn"))
        if (failed(verifyFlashAttn(op))) anyFailure = failure();
      if (isOp(op, "tessera.fused_epilogue"))
        if (failed(verifyFusedEpilogue(op))) anyFailure = failure();
    });
    if (failed(anyFailure)) signalPassFailure();
  }
};

} // namespace

namespace tessera {
std::unique_ptr<Pass> createVerifyTesseraIRPass() {
  return std::make_unique<VerifyTesseraIRPass>();
}
} // namespace tessera
