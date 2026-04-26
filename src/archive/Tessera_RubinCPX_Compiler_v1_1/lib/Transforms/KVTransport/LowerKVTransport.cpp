
#include "tessera/Transforms/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Async/IR/Async.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"

using namespace mlir;

static func::FuncOp getOrInsert(ModuleOp m, StringRef name,
                                FunctionType ty, OpBuilder &b) {
  if (auto f = m.lookupSymbol<func::FuncOp>(name)) return f;
  OpBuilder::InsertionGuard g(b);
  b.setInsertionPointToStart(m.getBody());
  auto f = b.create<func::FuncOp>(m.getLoc(), name, ty);
  f.setPrivate();
  return f;
}

namespace {
struct LowerKVExport : OpRewritePattern<Operation> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(Operation *op, PatternRewriter &rewriter) const override {
    if (op->getName().getStringRef() != "tessera.target.cpx.kv.export")
      return failure();

    auto m = op->getParentOfType<ModuleOp>();
    // Parse policy attr
    StringAttr policyAttr = op->getAttrOfType<StringAttr>("policy");
    StringRef policy = policyAttr ? policyAttr.getValue() : "pcie+cx9";
    StringRef fn = (policy == "nvlink") ? "tessera_kv_send_nvlink" : "tessera_kv_send_pcie_cx9";

    // Build function type: (i8*, i64) -> i32
    auto i64Ty = rewriter.getI64Type();
    auto i8Ptr = LLVM::LLVMPointerType::get(rewriter.getContext());
    auto i32Ty = rewriter.getI32Type();
    auto fnTy  = rewriter.getFunctionType({i8Ptr, i64Ty}, {i32Ty});
    auto callee = getOrInsert(m, fn, fnTy, rewriter);

    // Token result
    auto tokTy = async::TokenType::get(rewriter.getContext());
    // Replace with call + async.create to fabricate a token
    Value zeroPtr = rewriter.create<LLVM::NullOp>(op->getLoc(), i8Ptr);
    Value sz      = rewriter.create<arith::ConstantOp>(op->getLoc(), i64Ty,
                      rewriter.getI64IntegerAttr(0));
    rewriter.create<func::CallOp>(op->getLoc(), callee, ValueRange{zeroPtr, sz});
    Value token = rewriter.create<async::CreateOp>(op->getLoc(), tokTy);
    rewriter.replaceOp(op, token);
    return success();
  }
};

struct LowerKVImport : OpRewritePattern<Operation> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(Operation *op, PatternRewriter &rewriter) const override {
    if (op->getName().getStringRef() != "tessera.target.cpx.kv.import")
      return failure();
    // Expect first operand is !async.token; insert async.await and erase op.
    Value tok = op->getOperand(0);
    rewriter.create<async::AwaitOp>(op->getLoc(), tok);
    rewriter.eraseOp(op);
    return success();
  }
};

struct LowerKVTransportPass
  : public PassWrapper<LowerKVTransportPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LowerKVTransportPass)
  StringRef getArgument() const override { return "tessera-lower-kv-transport"; }
  StringRef getDescription() const override { return "Lower kv export/import with policy and async token flow"; }
  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    patterns.add<LowerKVExport, LowerKVImport>(&getContext());
    (void)applyPatternsAndFoldGreedily(getOperation(), std::move(patterns));
  }
};
} // namespace

namespace tessera {
std::unique_ptr<mlir::Pass> createLowerKVTransportPass() {
  return std::make_unique<LowerKVTransportPass>();
}
} // namespace tessera
