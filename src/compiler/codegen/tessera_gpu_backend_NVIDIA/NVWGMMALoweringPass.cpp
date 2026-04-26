//===- NVWGMMALoweringPass.cpp — Phase 3 ─────────────────────────────────===//
//
// Lowers tile.mma ops to NVIDIA WGMMA inline PTX (SM_90+) or WMMA fallback
// (SM < 90).
//
// SM_90+ WGMMA path:
//   tile.mma(%A, %B) {sm=90}
//   →
//   // 128-thread warpgroup (4 warps), tile 64×64×16 BF16
//   tessera.nvgpu.wgmma.mma_async {
//     shape = "m64n64k16", dtype_ab = "bf16", dtype_c = "f32",
//     ptx = "wgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 ..."
//   }
//   tessera.nvgpu.wgmma.commit_group
//   tessera.nvgpu.wgmma.wait_group {pending = 0}
//
// SM < 90 WMMA fallback:
//   tile.mma(%A, %B) {sm=80}
//   →
//   tessera.nvgpu.mma.sync { shape = "m16n16k16", dtype = "bf16" }
//
// Registration: --tessera-nvwgmma-lowering
//   Options:
//     --sm  target SM version (int, default 90)
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
// WGMMA shape constants for SM_90 BF16
// ─────────────────────────────────────────────────────────────────────────────

// Canonical wgmma.mma_async PTX for BF16 64×64×16 accumulate into f32.
// Emitted verbatim into the tessera.nvgpu.wgmma op's ptx attribute.
// The actual register binding is resolved by NVFlashAttnKernelEmitter.
static constexpr llvm::StringLiteral kWGMMAPTX_BF16_m64n64k16 =
    "wgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 "
    "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15},"
    " [%16], [%17], 1, 1, 1, 0, 0;";

static constexpr llvm::StringLiteral kWGMMAPTX_F16_m64n64k16 =
    "wgmma.mma_async.sync.aligned.m64n64k16.f32.f16.f16 "
    "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15},"
    " [%16], [%17], 1, 1, 1, 0, 0;";

// ─────────────────────────────────────────────────────────────────────────────
// Pattern: tile.mma → WGMMA (SM≥90) or WMMA (SM<90)
// ─────────────────────────────────────────────────────────────────────────────

struct LowerTileMMA : public RewritePattern {
  int smVersion;

  LowerTileMMA(MLIRContext *ctx, int sm)
      : RewritePattern("tile.mma", /*benefit=*/2, ctx),
        smVersion(sm) {}

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    if (op->getNumOperands() < 2)
      return failure();

    Value A = op->getOperand(0);
    Value B = op->getOperand(1);
    Location loc = op->getLoc();
    Type resType = op->getResults().empty() ? A.getType()
                                             : op->getResult(0).getType();

    // Read sm from op attr if set by TileIRLoweringPass, else use pass option.
    int sm = smVersion;
    if (auto a = op->getAttrOfType<IntegerAttr>("sm"))
      sm = (int)a.getInt();

    if (sm >= 90) {
      // ── WGMMA path ────────────────────────────────────────────────────────
      // Detect dtype from operand type to select the right PTX template.
      llvm::StringRef ptx = kWGMMAPTX_BF16_m64n64k16;
      StringRef dtypeAB = "bf16";
      if (auto tType = mlir::dyn_cast<ShapedType>(A.getType())) {
        if (tType.getElementType().isF16()) {
          ptx = kWGMMAPTX_F16_m64n64k16;
          dtypeAB = "f16";
        }
      }

      OperationState wgmmaSt(loc, "tessera.nvgpu.wgmma.mma_async");
      wgmmaSt.addOperands({A, B});
      wgmmaSt.addTypes(resType);
      wgmmaSt.addAttribute("shape",   rewriter.getStringAttr("m64n64k16"));
      wgmmaSt.addAttribute("dtype_ab", rewriter.getStringAttr(dtypeAB));
      wgmmaSt.addAttribute("dtype_c",  rewriter.getStringAttr("f32"));
      wgmmaSt.addAttribute("ptx",      rewriter.getStringAttr(ptx));
      Operation *wgmma = rewriter.create(wgmmaSt);

      // Commit the wgmma group.
      OperationState commitSt(loc, "tessera.nvgpu.wgmma.commit_group");
      rewriter.create(commitSt);

      // Wait for all pending wgmma to complete before consuming results.
      OperationState waitSt(loc, "tessera.nvgpu.wgmma.wait_group");
      waitSt.addAttribute("pending", rewriter.getI32IntegerAttr(0));
      rewriter.create(waitSt);

      rewriter.replaceOp(op, wgmma->getResults());
    } else {
      // ── WMMA fallback (SM 70–89) ─────────────────────────────────────────
      // nvgpu.mma.sync handles the details; we just emit the op with shape.
      OperationState wmmaSt(loc, "tessera.nvgpu.mma.sync");
      wmmaSt.addOperands({A, B});
      wmmaSt.addTypes(resType);
      wmmaSt.addAttribute("shape", rewriter.getStringAttr("m16n16k16"));
      wmmaSt.addAttribute("dtype", rewriter.getStringAttr("bf16"));
      Operation *wmma = rewriter.create(wmmaSt);
      rewriter.replaceOp(op, wmma->getResults());
    }

    return success();
  }
};

// ─────────────────────────────────────────────────────────────────────────────
// Pass
// ─────────────────────────────────────────────────────────────────────────────

struct NVWGMMALoweringPass
    : public PassWrapper<NVWGMMALoweringPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(NVWGMMALoweringPass)

  Option<int> smVersion{*this, "sm",
                        llvm::cl::desc("Target SM version (90 = Hopper WGMMA)"),
                        llvm::cl::init(90)};

  StringRef getArgument() const override { return "tessera-nvwgmma-lowering"; }
  StringRef getDescription() const override {
    return "Lower tile.mma to wgmma.mma_async PTX (SM≥90) or nvgpu.mma.sync fallback";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<arith::ArithDialect>();
    registry.insert<func::FuncDialect>();
  }

  void runOnOperation() override {
    MLIRContext *ctx = &getContext();
    RewritePatternSet patterns(ctx);
    patterns.add<LowerTileMMA>(ctx, smVersion);
    if (failed(applyPatternsAndFoldGreedily(getOperation(),
                                            std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

} // namespace

std::unique_ptr<mlir::Pass> createNVWGMMALoweringPass() {
  return std::make_unique<NVWGMMALoweringPass>();
}

} // namespace tessera
