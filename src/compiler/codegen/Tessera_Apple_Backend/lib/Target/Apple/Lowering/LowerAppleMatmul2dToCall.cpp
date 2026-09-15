//===- LowerAppleMatmul2dToCall.cpp - matmul2d → runtime dispatch --------===//
//
// APPLE-MATMUL2D-1. Lowers `tessera_apple.gpu.matmul2d` (over two
// `tessera_apple.gpu.tensor_view` operands) to a value-producing
// `tessera_apple.gpu.kernel_call` on the runtime's Metal 4 matmul2d symbols:
//   f16/f16   -> tessera_apple_gpu_mtl4_matmul2d_f16
//   bf16/bf16 -> tessera_apple_gpu_mtl4_matmul2d_bf16
//   low-precision pairs -> tessera_apple_gpu_mtl4_matmul2d_lowp (format code)
//
// Under Decision #31 as reconciled with #28 this is ONE lowering path with
// Tier-3 implementations behind it: the op is declared, verified and carries
// its contracts; the hand-written runtime kernel is the delegate. The call
// carries every view parameter as scalar attributes so the Python
// materializer projects the ABI from IR and never re-derives layout.
//===----------------------------------------------------------------------===//
#include "Tessera/Target/Apple/Passes.h"
#include "Tessera/Target/Apple/TesseraAppleDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

using namespace mlir;

namespace tessera::apple {
namespace {

std::string elementName(Type t) {
  if (t.isF16()) return "f16";
  if (t.isBF16()) return "bf16";
  if (isa<Float8E4M3FNType>(t)) return "f8E4M3FN";
  if (isa<Float8E5M2Type>(t)) return "f8E5M2";
  if (isa<Float4E2M1FNType>(t)) return "f4E2M1FN";
  return "unsupported";
}

struct LowerAppleMatmul2dToCallPass
    : public PassWrapper<LowerAppleMatmul2dToCallPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LowerAppleMatmul2dToCallPass)

  StringRef getArgument() const override { return "tessera-apple-matmul2d-to-call"; }
  StringRef getDescription() const override {
    return "APPLE-MATMUL2D-1 — lower tessera_apple.gpu.matmul2d to the runtime's "
           "Metal 4 matmul2d dispatch (kernel_call) with the view ABI in attributes.";
  }
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<TesseraAppleDialect>();
  }

  void runOnOperation() override {
    SmallVector<Matmul2dOp> ops;
    getOperation().walk([&](Matmul2dOp op) { ops.push_back(op); });
    for (Matmul2dOp op : ops) {
      auto aView = op.getA().getDefiningOp<TensorViewOp>();
      auto bView = op.getB().getDefiningOp<TensorViewOp>();
      if (!aView || !bView) {
        op.emitOpError("APPLE_MATMUL2D_OPERANDS: operands must be tensor_view results");
        signalPassFailure();
        return;
      }
      Type aElem = cast<TensorViewType>(op.getA().getType()).getElementType();
      Type bElem = cast<TensorViewType>(op.getB().getType()).getElementType();
      const int code = appleMatmul2dPairCode(aElem, bElem);
      StringRef symbol;
      if (code == 10) symbol = "tessera_apple_gpu_mtl4_matmul2d_f16";
      else if (code == 11) symbol = "tessera_apple_gpu_mtl4_matmul2d_bf16";
      else if (code >= 0) symbol = "tessera_apple_gpu_mtl4_matmul2d_lowp";
      else {
        op.emitOpError("APPLE_MATMUL2D_PAIR_UNSUPPORTED: no runtime symbol for this operand pair");
        signalPassFailure();
        return;
      }
      OpBuilder builder(op);
      OperationState state(op.getLoc(), "tessera_apple.gpu.kernel_call");
      state.addOperands({aView.getBuffer(), bView.getBuffer()});
      state.addTypes({op.getResult().getType()});
      state.addAttribute("op_kind", builder.getStringAttr("mtl4_matmul2d"));
      state.addAttribute("symbol", builder.getStringAttr(symbol));
      state.addAttribute("abi", builder.getStringAttr("mtl4_matmul2d_view"));
      state.addAttribute("status", builder.getStringAttr("executable"));
      state.addAttribute("framework", builder.getStringAttr("Metal"));
      state.addAttribute("dtype", builder.getStringAttr(elementName(aElem) + "x" + elementName(bElem)));
      state.addAttribute("tessera_apple.accumulate", builder.getStringAttr("fp32"));
      if (code < 10)
        state.addAttribute("tessera_apple.lowp_format", builder.getI64IntegerAttr(code));
      auto viewAttrs = [&](StringRef prefix, TensorViewOp v) {
        state.addAttribute((prefix + "_inner").str(), builder.getI64IntegerAttr(v.getExtents()[0]));
        state.addAttribute((prefix + "_outer").str(), builder.getI64IntegerAttr(v.getExtents()[1]));
        state.addAttribute((prefix + "_stride").str(), builder.getI64IntegerAttr(v.getStrides()[1]));
        state.addAttribute((prefix + "_byte_offset").str(), builder.getI64IntegerAttr(v.getByteOffset()));
      };
      viewAttrs("tessera_apple.a", aView);
      viewAttrs("tessera_apple.b", bView);
      state.addAttribute("tessera_apple.tile_m", op.getTileMAttr());
      state.addAttribute("tessera_apple.tile_n", op.getTileNAttr());
      state.addAttribute("tessera_apple.simdgroups", op.getSimdgroupsAttr());
      if (op->hasAttr("tessera_apple.canonical_k_loop"))
        state.addAttribute("tessera_apple.canonical_k_loop", builder.getBoolAttr(true));
      if (op->hasAttr("tessera_apple.ragged_zero_pad"))
        state.addAttribute("tessera_apple.ragged_zero_pad", builder.getBoolAttr(true));
      Operation *call = builder.create(state);
      op.getResult().replaceAllUsesWith(call->getResult(0));
      op.erase();
      for (TensorViewOp v : {aView, bView})
        if (v.use_empty()) v.erase();
    }
  }
};

} // namespace

std::unique_ptr<Pass> createLowerAppleMatmul2dToCallPass() {
  return std::make_unique<LowerAppleMatmul2dToCallPass>();
}

} // namespace tessera::apple
