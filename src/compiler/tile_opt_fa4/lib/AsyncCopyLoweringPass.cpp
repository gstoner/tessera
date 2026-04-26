//===- AsyncCopyLoweringPass.cpp — Phase 3 ───────────────────────────────===//
//
// Lowers tile.async_copy + tile.wait_async to TMA descriptor ops for SM_90+,
// or to cp.async row-copy fallback for SM < 90.
//
// SM_90+ TMA path:
//   tile.async_copy(%src) {tile_rows, tile_cols}
//   →
//   tessera.tma.descriptor { src_ptr, tile_rows, tile_cols, dtype }
//   tessera.tma.copy_async  { descriptor, dst_smem_offset, mbarrier_slot }
//
// SM < 90 fallback (cp.async):
//   tile.async_copy(%src)
//   →
//   tessera.cp_async.copy_row { src_ptr, dst_smem_offset, num_bytes }
//   tessera.cp_async.commit_group
//
// tile.wait_async → tessera.mbarrier.wait (SM_90) or
//                   tessera.cp_async.wait_all (SM < 90)
//
// Registration: --tessera-async-copy-lowering
//   Options:
//     --sm   target SM version (int). Defaults to 90.
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/StringRef.h"

using namespace mlir;

namespace tessera {

namespace {

// ─────────────────────────────────────────────────────────────────────────────
// Helper: emit a TMA descriptor op (SM_90 path)
// ─────────────────────────────────────────────────────────────────────────────
static Operation *emitTMADescriptor(OpBuilder &b, Location loc, Value src,
                                     int64_t tileRows, int64_t tileCols) {
  OperationState st(loc, "tessera.tma.descriptor");
  st.addOperands({src});
  st.addAttribute("tile_rows", b.getI64IntegerAttr(tileRows));
  st.addAttribute("tile_cols", b.getI64IntegerAttr(tileCols));
  // Result: opaque descriptor handle (i64 pointer in the final PTX).
  st.addTypes(b.getIntegerType(64));
  return b.create(st);
}

static Operation *emitTMACopyAsync(OpBuilder &b, Location loc,
                                    Value descriptor, int64_t mbarrierSlot) {
  OperationState st(loc, "tessera.tma.copy_async");
  st.addOperands({descriptor});
  st.addAttribute("mbarrier_slot", b.getI64IntegerAttr(mbarrierSlot));
  return b.create(st);
}

// ─────────────────────────────────────────────────────────────────────────────
// Helper: emit cp.async fallback (SM < 90)
// ─────────────────────────────────────────────────────────────────────────────
static void emitCpAsyncFallback(OpBuilder &b, Location loc, Value src,
                                 int64_t tileRows, int64_t tileCols) {
  int64_t numBytes = tileRows * tileCols * 2; // BF16 = 2 bytes

  OperationState cpSt(loc, "tessera.cp_async.copy_row");
  cpSt.addOperands({src});
  cpSt.addAttribute("num_bytes", b.getI64IntegerAttr(numBytes));
  b.create(cpSt);

  OperationState commitSt(loc, "tessera.cp_async.commit_group");
  b.create(commitSt);
}

// ─────────────────────────────────────────────────────────────────────────────
// Patterns
// ─────────────────────────────────────────────────────────────────────────────

struct LowerAsyncCopyTMA : public RewritePattern {
  int smVersion;

  LowerAsyncCopyTMA(MLIRContext *ctx, int sm)
      : RewritePattern("tile.async_copy", /*benefit=*/2, ctx),
        smVersion(sm) {}

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    if (op->getNumOperands() < 1)
      return failure();
    Value src = op->getOperand(0);
    Location loc = op->getLoc();

    int64_t tileRows = 64, tileCols = 64;
    if (auto a = op->getAttrOfType<IntegerAttr>("tile_rows"))
      tileRows = a.getInt();
    if (auto a = op->getAttrOfType<IntegerAttr>("tile_cols")) {
      int64_t v = a.getInt();
      if (v > 0) tileCols = v;
    }

    if (smVersion >= 90) {
      // SM_90+ → TMA path
      Operation *desc = emitTMADescriptor(rewriter, loc, src, tileRows, tileCols);
      // mbarrier_slot is 0 for the first async copy; NVTMADescriptorPass
      // will hoist the descriptor and assign unique slot indices.
      Operation *copyOp = emitTMACopyAsync(rewriter, loc,
                                            desc->getResult(0), /*slot=*/0);
      rewriter.replaceOp(op, copyOp->getResults());
    } else {
      // SM < 90 → cp.async fallback
      emitCpAsyncFallback(rewriter, loc, src, tileRows, tileCols);
      // No results — consumers will use src directly.
      if (!op->getResults().empty()) {
        // Forward original src as the tile (not yet in shared mem — lowered
        // further by memory allocation passes).
        rewriter.replaceOp(op, src);
      } else {
        rewriter.eraseOp(op);
      }
    }
    return success();
  }
};

struct LowerWaitAsync : public RewritePattern {
  int smVersion;

  LowerWaitAsync(MLIRContext *ctx, int sm)
      : RewritePattern("tile.wait_async", /*benefit=*/2, ctx),
        smVersion(sm) {}

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    Location loc = op->getLoc();
    if (smVersion >= 90) {
      OperationState st(loc, "tessera.mbarrier.wait");
      st.addAttribute("slot", rewriter.getI64IntegerAttr(0));
      rewriter.create(st);
    } else {
      OperationState st(loc, "tessera.cp_async.wait_all");
      rewriter.create(st);
    }
    rewriter.eraseOp(op);
    return success();
  }
};

// ─────────────────────────────────────────────────────────────────────────────
// Pass
// ─────────────────────────────────────────────────────────────────────────────

struct AsyncCopyLoweringPass
    : public PassWrapper<AsyncCopyLoweringPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(AsyncCopyLoweringPass)

  Option<int> smVersion{*this, "sm",
                        llvm::cl::desc("Target SM version (e.g. 90 for Hopper)"),
                        llvm::cl::init(90)};

  StringRef getArgument() const override {
    return "tessera-async-copy-lowering";
  }
  StringRef getDescription() const override {
    return "Lower tile.async_copy/wait_async to TMA (SM≥90) or cp.async (SM<90)";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<arith::ArithDialect>();
    registry.insert<func::FuncDialect>();
  }

  void runOnOperation() override {
    MLIRContext *ctx = &getContext();
    RewritePatternSet patterns(ctx);
    patterns.add<LowerAsyncCopyTMA>(ctx, smVersion);
    patterns.add<LowerWaitAsync>(ctx, smVersion);

    if (failed(applyPatternsAndFoldGreedily(getOperation(),
                                            std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

} // namespace

std::unique_ptr<mlir::Pass> createAsyncCopyLoweringPass() {
  return std::make_unique<AsyncCopyLoweringPass>();
}

} // namespace tessera
