#include "TesseraROCM/Passes.h"
#include "TesseraROCM/IR/TesseraROCMDialect.h.inc"
#include "TesseraROCM/IR/TesseraROCMOps.h.inc"
#include "TesseraROCM/MFMTables.h"

#include "mlir/Conversion/LLVMCommon/ConversionTarget.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/ROCDL/IR/ROCDLDialect.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/DialectConversion.h"

using namespace mlir;

static LLVM::LLVMFuncOp declareIntrinsic(ModuleOp module, StringRef name,
                                         Type retType, ArrayRef<Type> argTypes) {
  if (auto fn = module.lookupSymbol<LLVM::LLVMFuncOp>(name))
    return fn;
  OpBuilder b(module.getBodyRegion());
  auto fty = LLVM::LLVMFunctionType::get(retType, argTypes, false);
  return OpBuilder::atBlockBegin(module.getBody()).create<LLVM::LLVMFuncOp>(module.getLoc(), name, fty);
}

namespace {

struct MFMAOpLowering : OpConversionPattern<tessera_rocm::ROCM_MFMAOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(tessera_rocm::ROCM_MFMAOp op, OpAdaptor a,
                                ConversionPatternRewriter &rewriter) const override {
    auto module = op->getParentOfType<ModuleOp>();
    auto ctx = rewriter.getContext();
    // For the starter, assume scalar f32 returns and f16/bf16/f32 inputs
    auto name = mlir::tessera_rocm::chooseMFMAIntrinsic(op.getA().getType(), op.getB().getType(), op.getAcc().getType(), "gfx90a");
    auto f32 = LLVM::LLVMFloatType::get(ctx);
    auto fn = declareIntrinsic(module, name, f32, {f32,f32,f32});
    auto call = rewriter.create<LLVM::CallOp>(op.getLoc(), f32, SymbolRefAttr::get(fn), ValueRange{a.getA(), a.getB(), a.getAcc()});
    // Simple epilogue fuse sketch: if op has "gelu" attr, apply approx GELU
    if (op->hasAttr("gelu")) {
      // x * 0.5 * (1 + erf(x / sqrt(2))) approx with tanh-based
      auto c0 = rewriter.create<LLVM::ConstantOp>(op.getLoc(), f32, rewriter.getF32FloatAttr(0.79788456f)); // sqrt(2/pi)
      auto c05 = rewriter.create<LLVM::ConstantOp>(op.getLoc(), f32, rewriter.getF32FloatAttr(0.5f));
      auto c0447 = rewriter.create<LLVM::ConstantOp>(op.getLoc(), f32, rewriter.getF32FloatAttr(0.044715f));
      auto x = call.getResult();
      auto x3 = rewriter.create<LLVM::FMulOp>(op.getLoc(), f32, x, rewriter.create<LLVM::FMulOp>(op.getLoc(), f32, x, x));
      auto t = rewriter.create<LLVM::FMulOp>(op.getLoc(), f32, c0, rewriter.create<LLVM::FAddOp>(op.getLoc(), f32, x, rewriter.create<LLVM::FMulOp>(op.getLoc(), f32, c0447, x3)));
      auto th = rewriter.create<LLVM::TanhOp>(op.getLoc(), f32, t);
      auto y = rewriter.create<LLVM::FMulOp>(op.getLoc(), f32, x, rewriter.create<LLVM::FAddOp>(op.getLoc(), f32, c05, rewriter.create<LLVM::FMulOp>(op.getLoc(), f32, c05, rewriter.create<LLVM::FAddOp>(op.getLoc(), f32, th, rewriter.create<LLVM::ConstantOp>(op.getLoc(), f32, rewriter.getF32FloatAttr(1.0f))))));
      rewriter.replaceOp(op, y.getResult());
      return success();
    }
    rewriter.replaceOp(op, call.getResult());
    return success();
  }
};

// Lower async_copy to raw.buffer.load.* + ds.write.b128 and insert s_barrier for group sync.
struct AsyncCopyLowering : OpConversionPattern<tessera_rocm::ROCM_AsyncCopyOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(tessera_rocm::ROCM_AsyncCopyOp op, OpAdaptor a,
                                ConversionPatternRewriter &rewriter) const override {
    auto module = op->getParentOfType<ModuleOp>();
    auto ctx = rewriter.getContext();
    auto i32 = IntegerType::get(ctx, 32);
    auto i64 = IntegerType::get(ctx, 64);
    auto i8Ptr = LLVM::LLVMPointerType::get(IntegerType::get(ctx,8));
    auto f32 = LLVM::LLVMFloatType::get(ctx);

    // Declare intrinsics we need
    auto loadName = "llvm.amdgcn.raw.buffer.load.v4f32"; // vectorized 16B
    auto dsWriteName = "llvm.amdgcn.ds.write.b128";
    auto barrierName = "llvm.amdgcn.s.barrier";

    auto v4f32 = LLVM::LLVMVectorType::get(f32, 4);
    auto loadFn = declareIntrinsic(module, loadName, v4f32, {i8Ptr, i32, i32, i32, i32});
    auto dsFn = declareIntrinsic(module, dsWriteName, LLVM::LLVMVoidType::get(ctx), {v4f32, i32, i32, i32});
    auto barFn = declareIntrinsic(module, barrierName, LLVM::LLVMVoidType::get(ctx), {});

    // Pointers (treat as i8* for raw ops)
    Value srcPtr = rewriter.create<LLVM::BitcastOp>(op.getLoc(), i8Ptr, a.getSrc());
    Value dstIndex = rewriter.create<LLVM::UndefOp>(op.getLoc(), i32); // LDS offset placeholder

    // Offsets zero for demo; real path would compute from memref + indices
    Value zero32 = rewriter.create<LLVM::ConstantOp>(op.getLoc(), i32, rewriter.getI32IntegerAttr(0));
    Value soff = zero32, bo = zero32, so = zero32, flags = zero32;

    // Load 16B and write to LDS
    auto vec = rewriter.create<LLVM::CallOp>(op.getLoc(), v4f32, SymbolRefAttr::get(loadFn), ValueRange{srcPtr, soff, bo, so, flags});
    rewriter.create<LLVM::CallOp>(op.getLoc(), TypeRange(), SymbolRefAttr::get(dsFn), ValueRange{vec.getResult(), dstIndex, zero32, zero32});

    // s_barrier for group completion (simple model)
    rewriter.create<LLVM::CallOp>(op.getLoc(), TypeRange(), SymbolRefAttr::get(barFn), ValueRange{});

    // Return a null token
    auto nullTok = rewriter.create<LLVM::NullOp>(op.getLoc(), LLVM::LLVMPointerType::get(IntegerType::get(ctx,8)));
    rewriter.replaceOp(op, ValueRange{nullTok});
    return success();
  }
};

struct WaitLowering : OpConversionPattern<tessera_rocm::ROCM_WaitTokenOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(tessera_rocm::ROCM_WaitTokenOp op, OpAdaptor a,
                                ConversionPatternRewriter &rewriter) const override {
    // Already inserted barrier in AsyncCopyLowering; here we erase wait.
    rewriter.eraseOp(op);
    return success();
  }
};

struct LoweringPass : PassWrapper<LoweringPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LoweringPass)
  void runOnOperation() override {
    MLIRContext *ctx = &getContext();
    LLVMTypeConverter tc(ctx);

    ConversionTarget target(*ctx);
    target.addLegalDialect<LLVM::LLVMDialect, ROCDL::ROCDLDialect, func::FuncDialect>();
    target.addIllegalOp<tessera_rocm::ROCM_MFMAOp, tessera_rocm::ROCM_AsyncCopyOp, tessera_rocm::ROCM_WaitTokenOp>();

    RewritePatternSet patterns(ctx);
    patterns.add<MFMAOpLowering, AsyncCopyLowering, WaitLowering>(tc, ctx);

    if (failed(applyPartialConversion(getOperation(), target, std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace

std::unique_ptr<mlir::Pass> createLowerTesseraToROCDLImpl() { return std::make_unique<LoweringPass>(); }
