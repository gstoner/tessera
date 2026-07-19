//===- AsyncCopyLoweringPass.cpp — Phase 3 ───────────────────────────────===//
//
// Lowers tile.async_copy + tile.wait_async to TMA descriptor ops for SM_90+,
// or to cp.async row-copy fallback for SM < 90.
//
// SM_90+ TMA path:
//   tile.async_copy(%src) {tile_rows, tile_cols}
//   →
//   tile.tma.descriptor { src_ptr, tile_rows, tile_cols, dtype }
//   tile.tma.copy_async  { descriptor, dst_smem_offset, mbarrier_slot }
//
// SM < 90 fallback (cp.async):
//   tile.async_copy(%src)
//   →
//   tile.cp_async.copy_row { src_ptr, dst_smem_offset, num_bytes }
//   tile.cp_async.commit_group
//
// tile.wait_async → tile.mbarrier.wait (SM_90) or
//                   tile.cp_async.wait_all (SM < 90)
//
// Registration: --tessera-async-copy-lowering
//   Options:
//     --sm   target SM version (int). Defaults to 90.
//===----------------------------------------------------------------------===//

#include "Tessera/Dialect/Tile/TileDialect.h"

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
  OperationState st(loc, "tile.tma.descriptor");
  st.addOperands({src});
  st.addAttribute("tile_rows", b.getI64IntegerAttr(tileRows));
  st.addAttribute("tile_cols", b.getI64IntegerAttr(tileCols));
  // Result: opaque descriptor handle (i64 pointer in the final PTX).
  st.addTypes(b.getIntegerType(64));
  return b.create(st);
}

static Operation *emitTMACopyAsync(OpBuilder &b, Location loc,
                                    Value descriptor, int64_t mbarrierSlot,
                                    Type resultType, Type tokenType = Type()) {
  OperationState st(loc, "tile.tma.copy_async");
  st.addOperands({descriptor});
  st.addAttribute("mbarrier_slot", b.getI64IntegerAttr(mbarrierSlot));
  // Produce the loaded tile so it can replace the original tile.async_copy
  // result 1:1 (the copy lands the tile in shared memory).  Without a result
  // the replaceOp below would be a result-count mismatch that corrupts the IR.
  if (resultType)
    st.addTypes(resultType);
  // Carry the planner/warpspec !tile.async_token completion edge through TMA
  // lowering so a consuming wait/mma operand (and any warp-region yield of it)
  // stays a valid SSA def-use after the copy is lowered — the NV analogue of the
  // ROCm token edge (Phase C-NV). The mbarrier still carries the byte count.
  if (tokenType)
    st.addTypes(tokenType);
  return b.create(st);
}

// ─────────────────────────────────────────────────────────────────────────────
// Helper: emit cp.async fallback (SM < 90)
// ─────────────────────────────────────────────────────────────────────────────
static void emitCpAsyncFallback(OpBuilder &b, Location loc, Value src,
                                 int64_t tileRows, int64_t tileCols) {
  int64_t numBytes = tileRows * tileCols * 2; // BF16 = 2 bytes

  OperationState cpSt(loc, "tile.cp_async.copy_row");
  cpSt.addOperands({src});
  cpSt.addAttribute("num_bytes", b.getI64IntegerAttr(numBytes));
  b.create(cpSt);

  OperationState commitSt(loc, "tile.cp_async.commit_group");
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

    // A trailing !tile.async_token result is the planner/warpspec completion
    // edge (Phase C-NV). It is the last result; the leading result (if any) is
    // the staged tile. Carry the token through lowering so the replacement is
    // 1:1 and the consuming wait/mma/yield operands stay valid SSA.
    unsigned numResults = op->getNumResults();
    bool hasToken =
        numResults >= 1 &&
        isa<tessera::tile::AsyncTokenType>(
            op->getResult(numResults - 1).getType());

    if (smVersion >= 90) {
      // SM_90+ → TMA path
      Operation *desc = emitTMADescriptor(rewriter, loc, src, tileRows, tileCols);
      // mbarrier_slot is 0 for the first async copy; NVTMADescriptorPass
      // will hoist the descriptor and assign unique slot indices.  The copy
      // carries the original tile result type so the replacement is 1:1.
      Type tokenTy = hasToken ? op->getResult(numResults - 1).getType() : Type();
      Type tileTy;
      if (hasToken && numResults >= 2)
        tileTy = op->getResult(0).getType();
      else if (!hasToken && numResults >= 1)
        tileTy = op->getResult(0).getType();
      Operation *copyOp = emitTMACopyAsync(rewriter, loc, desc->getResult(0),
                                           /*slot=*/0, tileTy, tokenTy);
      if (op->getNumResults())
        rewriter.replaceOp(op, copyOp->getResults());
      else
        rewriter.eraseOp(op);
    } else {
      // SM < 90 → cp.async fallback. The cp.async path has no SSA completion
      // token; carrying one here would silently drop the dependency edge, so
      // refuse it with a clear diagnostic rather than corrupt the IR.
      if (hasToken) {
        op->emitError(
            "ASYNC_COPY_TOKEN_NO_CP_ASYNC_PATH: tile.async_copy carries a "
            "!tile.async_token but the SM<90 cp.async fallback has no token "
            "completion path; thread async tokens only on the SM>=90 TMA path.");
        return failure();
      }
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
      OperationState st(loc, "tile.mbarrier.wait");
      st.addOperands(op->getOperands());
      st.addAttribute("slot", rewriter.getI64IntegerAttr(0));
      rewriter.create(st);
    } else {
      OperationState st(loc, "tile.cp_async.wait_all");
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

  AsyncCopyLoweringPass() = default;
  explicit AsyncCopyLoweringPass(int sm) { smVersion = sm; }
  AsyncCopyLoweringPass(const AsyncCopyLoweringPass &other)
      : PassWrapper(other) {}

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

    FrozenRewritePatternSet frozenPatterns(std::move(patterns));
    if (failed(applyPatternsGreedily(getOperation(), frozenPatterns))) {
      signalPassFailure();
    }
  }
};

} // namespace

std::unique_ptr<mlir::Pass> createAsyncCopyLoweringPass(int sm) {
  return std::make_unique<AsyncCopyLoweringPass>(sm);
}

} // namespace tessera
