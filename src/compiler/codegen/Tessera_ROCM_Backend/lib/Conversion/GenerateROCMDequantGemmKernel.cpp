//===- GenerateROCMDequantGemmKernel.cpp - packed dequant GEMM kernel -----===//
//
// Expands a `tessera_rocm.dequant_gemm` directive into a fused f32 kernel:
//
//   O[m,n] = sum_k X[m,k] * code(k,n) * scales[k/group_size,n]
//
// `mode=4` consumes int4 codes packed two per byte as `(N, K/2)`. `mode=8`
// consumes int8 codes as `(K, N)`.  The full fp32 weight is never materialized.
// This is the first native ROCm proof for DK4; later tuning can replace this
// scalar-per-output kernel with a cooperative tiled implementation without
// changing the runtime ABI.
//===----------------------------------------------------------------------===//

#include "TesseraROCM/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Pass/Pass.h"

using namespace mlir;

namespace {

static constexpr int64_t BD = 256;

void emitDequantGemmBody(OpBuilder &b, Location loc, gpu::GPUFuncOp f) {
  Type f32 = b.getF32Type();
  Type i32 = b.getI32Type();
  auto slt = arith::CmpIPredicate::slt;
  auto eq = arith::CmpIPredicate::eq;
  auto sge = arith::CmpIPredicate::sge;

  b.setInsertionPointToStart(&f.getBody().front());
  Value X = f.getArgument(0);
  Value Codes = f.getArgument(1);
  Value Scales = f.getArgument(2);
  Value O = f.getArgument(3);
  Value M = f.getArgument(4);
  Value K = f.getArgument(5);
  Value N = f.getArgument(6);
  Value G = f.getArgument(7);
  Value mode = f.getArgument(8);

  auto ci = [&](int64_t v) { return b.create<arith::ConstantIndexOp>(loc, v); };
  auto ci32 = [&](int32_t v) { return b.create<arith::ConstantIntOp>(loc, i32, v); };
  Value c0 = ci(0), c1 = ci(1), c2 = ci(2), cBD = ci(BD);
  Value f0 = b.create<arith::ConstantOp>(loc, f32, b.getF32FloatAttr(0.0f));
  Value mode4 = ci(4);

  Value bid = b.create<gpu::BlockIdOp>(loc, gpu::Dimension::x);
  Value tid = b.create<gpu::ThreadIdOp>(loc, gpu::Dimension::x);
  Value gid = b.create<arith::AddIOp>(loc, b.create<arith::MulIOp>(loc, bid, cBD),
                                      tid);
  Value total = b.create<arith::MulIOp>(loc, M, N);
  Value inb = b.create<arith::CmpIOp>(loc, slt, gid, total);
  auto rowIf = b.create<scf::IfOp>(loc, inb, /*withElse=*/false);
  b.setInsertionPointToStart(rowIf.thenBlock());

  Value m = b.create<arith::DivUIOp>(loc, gid, N);
  Value n = b.create<arith::RemUIOp>(loc, gid, N);
  Value isInt4 = b.create<arith::CmpIOp>(loc, eq, mode, mode4);

  auto flat2 = [&](Value i, Value j, Value J) -> Value {
    return b.create<arith::AddIOp>(loc, b.create<arith::MulIOp>(loc, i, J), j);
  };

  auto loadCodeF32 = [&](Value k) -> Value {
    auto ifop = b.create<scf::IfOp>(loc, i32, isInt4, /*withElse=*/true);
    {
      OpBuilder::InsertionGuard g(b);
      b.setInsertionPointToStart(ifop.thenBlock());
      Value halfK = b.create<arith::DivUIOp>(loc, K, c2);
      Value byteIdx = flat2(n, b.create<arith::DivUIOp>(loc, k, c2), halfK);
      Value raw8 = b.create<memref::LoadOp>(loc, Codes, ValueRange{byteIdx});
      Value raw = b.create<arith::ExtUIOp>(loc, i32, raw8);
      Value odd = b.create<arith::CmpIOp>(
          loc, eq, b.create<arith::RemUIOp>(loc, k, c2), c1);
      Value hi = b.create<arith::ShRUIOp>(loc, raw, ci32(4));
      Value pick = b.create<arith::SelectOp>(loc, odd, hi, raw);
      Value nib = b.create<arith::AndIOp>(loc, pick, ci32(15));
      Value neg = b.create<arith::SubIOp>(loc, nib, ci32(16));
      Value signedNib = b.create<arith::SelectOp>(
          loc, b.create<arith::CmpIOp>(loc, sge, nib, ci32(8)), neg, nib);
      b.create<scf::YieldOp>(loc, ValueRange{signedNib});

      b.setInsertionPointToStart(ifop.elseBlock());
      Value codeIdx = flat2(k, n, N);
      Value rawI8 = b.create<memref::LoadOp>(loc, Codes, ValueRange{codeIdx});
      Value signedI32 = b.create<arith::ExtSIOp>(loc, i32, rawI8);
      b.create<scf::YieldOp>(loc, ValueRange{signedI32});
    }
    return b.create<arith::SIToFPOp>(loc, f32, ifop.getResult(0));
  };

  auto accLoop = b.create<scf::ForOp>(loc, c0, K, c1, ValueRange{f0});
  {
    OpBuilder::InsertionGuard g(b);
    b.setInsertionPointToStart(accLoop.getBody());
    Value k = accLoop.getInductionVar();
    Value acc = accLoop.getRegionIterArgs()[0];
    Value x = b.create<memref::LoadOp>(loc, X, ValueRange{flat2(m, k, K)});
    Value code = loadCodeF32(k);
    Value scaleIdx = flat2(b.create<arith::DivUIOp>(loc, k, G), n, N);
    Value scale = b.create<memref::LoadOp>(loc, Scales, ValueRange{scaleIdx});
    Value dq = b.create<arith::MulFOp>(loc, code, scale);
    Value prod = b.create<arith::MulFOp>(loc, x, dq);
    Value next = b.create<arith::AddFOp>(loc, acc, prod);
    b.create<scf::YieldOp>(loc, ValueRange{next});
  }

  b.create<memref::StoreOp>(loc, accLoop.getResult(0), O, ValueRange{gid});
  b.setInsertionPointToEnd(&f.getBody().front());
  b.create<gpu::ReturnOp>(loc);
}

struct GenerateROCMDequantGemmKernelPass
    : PassWrapper<GenerateROCMDequantGemmKernelPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(GenerateROCMDequantGemmKernelPass)

  StringRef getArgument() const final {
    return "generate-rocm-dequant-gemm-kernel";
  }
  StringRef getDescription() const final {
    return "Expand a tessera_rocm.dequant_gemm directive into a fused packed "
           "dequantize-into-GEMM gpu kernel (compiler-generated)";
  }
  void getDependentDialects(DialectRegistry &registry) const final {
    registry.insert<gpu::GPUDialect, scf::SCFDialect, arith::ArithDialect,
                    memref::MemRefDialect>();
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();
    SmallVector<Operation *> directives;
    module.walk([&](Operation *op) {
      if (op->getName().getStringRef() == "tessera_rocm.dequant_gemm")
        directives.push_back(op);
    });
    for (Operation *op : directives) {
      auto nameAttr = op->getAttrOfType<StringAttr>("name");
      if (!nameAttr) {
        op->emitError("tessera_rocm.dequant_gemm missing name");
        return signalPassFailure();
      }
      OpBuilder b(module.getBodyRegion());
      b.setInsertionPointToEnd(module.getBody());
      Location loc = op->getLoc();
      std::string kname = nameAttr.getValue().str();

      auto gpuMod = b.create<gpu::GPUModuleOp>(loc, kname + "_mod");
      b.setInsertionPointToStart(&gpuMod.getBodyRegion().front());
      Type idxTy = b.getIndexType();
      auto fmem = MemRefType::get({ShapedType::kDynamic}, b.getF32Type());
      auto imem = MemRefType::get({ShapedType::kDynamic}, b.getI8Type());
      auto fnTy = b.getFunctionType(
          {fmem, imem, fmem, fmem, idxTy, idxTy, idxTy, idxTy, idxTy}, {});
      auto gpuFunc = b.create<gpu::GPUFuncOp>(loc, kname, fnTy);
      gpuFunc->setAttr(gpu::GPUDialect::getKernelFuncAttrName(),
                       b.getUnitAttr());
      OpBuilder body(gpuFunc.getContext());
      emitDequantGemmBody(body, loc, gpuFunc);
      op->erase();
    }
  }
};

} // namespace

std::unique_ptr<mlir::Pass>
mlir::tessera_rocm::createGenerateROCMDequantGemmKernelPass() {
  return std::make_unique<GenerateROCMDequantGemmKernelPass>();
}
