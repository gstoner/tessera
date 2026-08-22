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

#include "Tessera/Dialect/Tile/TileDialect.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/StringRef.h"

using namespace mlir;

namespace tessera {

namespace {

static func::FuncOp getOrDeclareBackendCall(ModuleOp module, StringRef name,
                                            Type resultType, ValueRange operands,
                                            PatternRewriter &rewriter) {
  if (auto fn = module.lookupSymbol<func::FuncOp>(name))
    return fn;

  SmallVector<Type> operandTypes;
  for (Value operand : operands)
    operandTypes.push_back(operand.getType());

  OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPointToStart(module.getBody());
  auto fnType = rewriter.getFunctionType(operandTypes, TypeRange{resultType});
  auto fn = rewriter.create<func::FuncOp>(module.getLoc(), name, fnType);
  fn.setPrivate();
  return fn;
}

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

    SmallVector<Value> data = tessera::tile::dataOperands(op);
    if (data.size() < 2 || data.size() > 3)
      return failure();
    Value A = data[0];
    ValueRange callOperands(data);
    Location loc = op->getLoc();
    Type resType = op->getResults().empty() ? A.getType()
                                             : op->getResult(0).getType();

    // Read sm from op attr if set by TileIRLoweringPass, else use pass option.
    int sm = smVersion;
    if (auto a = op->getAttrOfType<IntegerAttr>("sm"))
      sm = (int)a.getInt();

    if (sm >= 90) {
      // ── WGMMA path ────────────────────────────────────────────────────────
      // Keep this as a legal func.call boundary until a real NVVM/NVGPU WGMMA
      // op model is registered. This prevents unregistered pseudo target ops
      // from escaping the backend pipeline.
      StringRef callee = "tessera_nvidia_wgmma_mma_async_bf16_m64n64k16";
      StringRef dtypeAB = "bf16";
      if (auto tType = mlir::dyn_cast<ShapedType>(A.getType())) {
        if (tType.getElementType().isF16()) {
          callee = "tessera_nvidia_wgmma_mma_async_f16_m64n64k16";
          dtypeAB = "f16";
        }
      }

      auto module = op->getParentOfType<ModuleOp>();
      auto fn = getOrDeclareBackendCall(module, callee, resType, callOperands,
                                        rewriter);
      auto call = rewriter.create<func::CallOp>(loc, fn, callOperands);
      call->setAttr("tessera.nvidia.shape", rewriter.getStringAttr("m64n64k16"));
      call->setAttr("tessera.nvidia.dtype_ab", rewriter.getStringAttr(dtypeAB));
      call->setAttr("tessera.nvidia.dtype_c", rewriter.getStringAttr("f32"));
      rewriter.replaceOp(op, call.getResults());
    } else {
      // ── WMMA fallback (SM 70–89) ─────────────────────────────────────────
      auto module = op->getParentOfType<ModuleOp>();
      auto fn = getOrDeclareBackendCall(module, "tessera_nvidia_wmma_mma_sync_bf16_m16n16k16",
                                        resType, callOperands, rewriter);
      auto call = rewriter.create<func::CallOp>(loc, fn, callOperands);
      call->setAttr("tessera.nvidia.shape", rewriter.getStringAttr("m16n16k16"));
      call->setAttr("tessera.nvidia.dtype_ab", rewriter.getStringAttr("bf16"));
      rewriter.replaceOp(op, call.getResults());
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

  NVWGMMALoweringPass() = default;
  explicit NVWGMMALoweringPass(int sm) { smVersion = sm; }
  NVWGMMALoweringPass(const NVWGMMALoweringPass &other)
      : PassWrapper(other) {}

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

    // Decision #21 — refuse an mma this compatibility pass cannot represent.
    //
    // A/B/C are now threaded through the opaque backend-call boundary. More
    // than three data operands belongs to an architecture-specific extended
    // MMA contract (for example block scales), which this legacy pass cannot
    // encode. Control tokens are filtered by dataOperands and remain owned by
    // the synchronization pipeline.
    //
    // Measured on merged main, this is NOT specific to the typed fragment form:
    // a legacy bare `tile.mma(A, B, C)` -- precisely what
    // `LowerKReductionAddToTileMMA` emits for the canonical K-step -- was
    // dropped the same way, and no fixture in the tree covered either case.
    // So the condition is "has an accumulator", not "is typed".
    //
    // This runs in the PASS BODY, not in the pattern. A pattern that emits an
    // error and returns `failure()` only declines to match: the diagnostic
    // prints, the pass still succeeds, and the pipeline continues with an
    // unlowered `tile.mma` -- an error that does not fail the compilation is a
    // warning in disguise, which is the same fail-open shape as the bug itself.
    // ROCm calls `signalPassFailure()` at this seam; this is NVIDIA catching up.
    //
    // Counting DATA operands via the shared helper keeps warp-spec tokens from
    // being mistaken for an accumulator (the #491 rule).
    //
    // Temporary: W1.1 step 2b threads the accumulator for real, and then this
    // is replaced by lowering, not relaxed.
    bool refused = false;
    getOperation()->walk([&](Operation *op) {
      if (op->getName().getStringRef() != "tile.mma")
        return;
      if (tessera::tile::dataOperands(op).size() <= 3)
        return;
      op->emitError(
          "NVWGMMA_ACCUMULATOR_DROPPED: this legacy pass supports A/B and an "
          "optional accumulator, but cannot represent additional data "
          "operands. Refusing instead of dropping an extended MMA operand.");
      refused = true;
    });
    if (refused)
      return signalPassFailure();

    RewritePatternSet patterns(ctx);
    patterns.add<LowerTileMMA>(ctx, smVersion);
    FrozenRewritePatternSet frozenPatterns(std::move(patterns));
    if (failed(applyPatternsGreedily(getOperation(), frozenPatterns))) {
      signalPassFailure();
    }
  }
};

} // namespace

std::unique_ptr<mlir::Pass> createNVWGMMALoweringPass(int sm) {
  return std::make_unique<NVWGMMALoweringPass>(sm);
}

std::unique_ptr<mlir::Pass> createNVTMADescriptorPass();
std::unique_ptr<mlir::Pass> createNVFlashAttnKernelEmitterPass(int sm);

void buildTesseraNVIDIALegacyBackendPipeline(OpPassManager &pm) {
  pm.addPass(createNVWGMMALoweringPass(90));
  pm.addNestedPass<func::FuncOp>(createNVTMADescriptorPass());
  pm.addPass(createNVFlashAttnKernelEmitterPass(90));
}

void registerTesseraNVIDIALegacyBackendPasses() {
  PassPipelineRegistration<> pipeline(
      "tessera-nvidia-backend",
      "Lower Tessera Tile IR to NVIDIA backend calls and kernel metadata",
      [](OpPassManager &pm) { buildTesseraNVIDIALegacyBackendPipeline(pm); });
}

} // namespace tessera
