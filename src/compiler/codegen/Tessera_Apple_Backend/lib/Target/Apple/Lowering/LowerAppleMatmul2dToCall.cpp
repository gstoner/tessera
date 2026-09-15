//===- LowerAppleMatmul2dToCall.cpp - matmul2d → runtime dispatch --------===//
//
// APPLE-MATMUL2D-1. Lowers `tessera_apple.gpu.matmul2d` and
// `tessera_apple.gpu.matmul2d_epilogue` (over two
// `tessera_apple.gpu.tensor_view` operands) to a value-producing
// `tessera_apple.gpu.kernel_call` on the runtime's strided-view Metal 4
// matmul2d entry:
//   gpu.matmul2d          -> tessera_apple_gpu_mtl4_matmul2d_view
//   gpu.matmul2d_epilogue -> tessera_apple_gpu_mtl4_matmul2d_view_epilogue
// One symbol for every operand pair; the pair code (`tessera_apple.pair`,
// TesseraAppleDialect.h) selects the kernel, and every view parameter --
// inner/outer extent, row stride, byte offset, per operand -- rides as a
// scalar attribute, so the Python materializer projects the ABI from IR and
// never re-derives layout. Nonzero origins and padded strides therefore reach
// the dispatcher exactly as the verifier admitted them.
//
// Under Decision #31 as reconciled with #28 this is ONE lowering path with
// Tier-3 implementations behind it: the op is declared, verified and carries
// its contracts; the hand-written runtime kernel is the delegate.
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
    return "APPLE-MATMUL2D-1 — lower tessera_apple.gpu.matmul2d[_epilogue] to the "
           "runtime's strided-view Metal 4 matmul2d dispatch (kernel_call) with the "
           "view ABI in attributes.";
  }
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<TesseraAppleDialect>();
  }

  // Lower one op. `bias` is null for the plain product; `act` is "none" then.
  bool lower(Operation *op, Value a, Value b, Value bias, StringRef act, IntegerAttr tileM,
             IntegerAttr tileN, IntegerAttr simdgroups) {
    auto aView = a.getDefiningOp<TensorViewOp>();
    auto bView = b.getDefiningOp<TensorViewOp>();
    if (!aView || !bView) {
      op->emitOpError("APPLE_MATMUL2D_OPERANDS: operands must be tensor_view results");
      return false;
    }
    Type aElem = cast<TensorViewType>(a.getType()).getElementType();
    Type bElem = cast<TensorViewType>(b.getType()).getElementType();
    const int code = appleMatmul2dPairCode(aElem, bElem);
    if (code < 0) {
      op->emitOpError("APPLE_MATMUL2D_PAIR_UNSUPPORTED: no runtime symbol for this operand pair");
      return false;
    }
    const int actCode = appleMatmul2dActCode(act);
    if (actCode < 0) {
      op->emitOpError("APPLE_MATMUL2D_EPILOGUE_ACT: no runtime activation code for ") << act;
      return false;
    }
    const bool fused = bias || actCode != 0;
    OpBuilder builder(op);
    OperationState state(op->getLoc(), "tessera_apple.gpu.kernel_call");
    state.addOperands({aView.getBuffer(), bView.getBuffer()});
    if (bias) state.addOperands({bias});
    state.addTypes({op->getResult(0).getType()});
    state.addAttribute("op_kind", builder.getStringAttr(fused ? "mtl4_matmul2d_epilogue"
                                                              : "mtl4_matmul2d"));
    state.addAttribute("symbol", builder.getStringAttr(
                                     fused ? "tessera_apple_gpu_mtl4_matmul2d_view_epilogue"
                                           : "tessera_apple_gpu_mtl4_matmul2d_view"));
    state.addAttribute("abi", builder.getStringAttr("mtl4_matmul2d_view"));
    state.addAttribute("status", builder.getStringAttr("executable"));
    state.addAttribute("framework", builder.getStringAttr("Metal"));
    state.addAttribute("dtype", builder.getStringAttr(elementName(aElem) + "x" + elementName(bElem)));
    state.addAttribute("tessera_apple.accumulate", builder.getStringAttr("fp32"));
    state.addAttribute("tessera_apple.pair", builder.getI64IntegerAttr(code));
    if (code < 10)  // kept for readers of the earlier slice; `pair` is the contract
      state.addAttribute("tessera_apple.lowp_format", builder.getI64IntegerAttr(code));
    if (fused) {
      state.addAttribute("tessera_apple.act", builder.getStringAttr(act));
      state.addAttribute("tessera_apple.has_bias", builder.getBoolAttr(bool(bias)));
    }
    auto viewAttrs = [&](StringRef prefix, TensorViewOp v) {
      state.addAttribute((prefix + "_inner").str(), builder.getI64IntegerAttr(v.getExtents()[0]));
      state.addAttribute((prefix + "_outer").str(), builder.getI64IntegerAttr(v.getExtents()[1]));
      state.addAttribute((prefix + "_stride").str(), builder.getI64IntegerAttr(v.getStrides()[1]));
      state.addAttribute((prefix + "_byte_offset").str(), builder.getI64IntegerAttr(v.getByteOffset()));
    };
    viewAttrs("tessera_apple.a", aView);
    viewAttrs("tessera_apple.b", bView);
    state.addAttribute("tessera_apple.tile_m", tileM);
    state.addAttribute("tessera_apple.tile_n", tileN);
    state.addAttribute("tessera_apple.simdgroups", simdgroups);
    for (StringRef prov : {"tessera_apple.canonical_k_loop", "tessera_apple.ragged_zero_pad",
                           "tessera_apple.ragged_tail"})
      if (Attribute attr = op->getAttr(prov)) state.addAttribute(prov, attr);
    Operation *call = builder.create(state);
    op->getResult(0).replaceAllUsesWith(call->getResult(0));
    op->erase();
    for (TensorViewOp v : {aView, bView})
      if (v.use_empty()) v.erase();
    return true;
  }

  void runOnOperation() override {
    SmallVector<Operation *> ops;
    getOperation().walk([&](Operation *op) {
      if (isa<Matmul2dOp, Matmul2dEpilogueOp>(op)) ops.push_back(op);
    });
    for (Operation *op : ops) {
      bool ok;
      if (auto mm = dyn_cast<Matmul2dOp>(op))
        ok = lower(op, mm.getA(), mm.getB(), nullptr, "none", mm.getTileMAttr(),
                   mm.getTileNAttr(), mm.getSimdgroupsAttr());
      else {
        auto epi = cast<Matmul2dEpilogueOp>(op);
        ok = lower(op, epi.getA(), epi.getB(), epi.getBias(), epi.getAct(), epi.getTileMAttr(),
                   epi.getTileNAttr(), epi.getSimdgroupsAttr());
      }
      if (!ok) {
        signalPassFailure();
        return;
      }
    }
  }
};

} // namespace

std::unique_ptr<Pass> createLowerAppleMatmul2dToCallPass() {
  return std::make_unique<LowerAppleMatmul2dToCallPass>();
}

} // namespace tessera::apple
