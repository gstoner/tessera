//===- GenerateROCMAlibiKernel.cpp - ALiBi positional-bias kernel ---------===//
//
// Expands a `tessera_rocm.alibi` directive into a flat elementwise gpu kernel
// generating the ALiBi positional-bias tensor of shape [H, S, S] (one thread
// per element, strided grid):
//
//   bias[h, i, j] = slope[h] * (j - i)
//
// `slope` is a per-head scalar supplied as a length-H f32 buffer (the runtime
// fills the default 2^(-8(k+1)/H) ramp when the caller passes none). Matches the
// `nn.functional.alibi` reference (distance = positions[j] - positions[i]).
// Output computes in f32 and stores f16/bf16/f32. H/S are runtime index args.
// Validated vs the numpy reference on gfx1151.
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

void emitAlibiBody(OpBuilder &b, Location loc, gpu::GPUFuncOp f, Type storeTy) {
  Type f32 = b.getF32Type();
  Type i64 = b.getIntegerType(64);
  bool isF32 = storeTy.isF32();
  auto slt = arith::CmpIPredicate::slt;

  b.setInsertionPointToStart(&f.getBody().front());
  // (Slopes : memref<?xf32>, O : memref<?xstore>, H : index, S : index)
  Value Slopes = f.getArgument(0), O = f.getArgument(1), H = f.getArgument(2),
        S = f.getArgument(3);

  Value bid = b.create<gpu::BlockIdOp>(loc, gpu::Dimension::x);
  Value tid = b.create<gpu::ThreadIdOp>(loc, gpu::Dimension::x);
  Value cBD = b.create<arith::ConstantIndexOp>(loc, BD);
  Value gid = b.create<arith::AddIOp>(loc, b.create<arith::MulIOp>(loc, bid, cBD),
                                      tid);
  // N = H * S * S
  Value SS = b.create<arith::MulIOp>(loc, S, S);
  Value N = b.create<arith::MulIOp>(loc, H, SS);
  Value inb = b.create<arith::CmpIOp>(loc, slt, gid, N);
  auto ifo = b.create<scf::IfOp>(loc, inb, /*withElse=*/false);
  b.setInsertionPointToStart(ifo.thenBlock());

  // decode gid -> (h, i, j): j = gid % S; t = gid / S; i = t % S; h = t / S.
  Value j = b.create<arith::RemUIOp>(loc, gid, S);
  Value t = b.create<arith::DivUIOp>(loc, gid, S);
  Value i = b.create<arith::RemUIOp>(loc, t, S);
  Value h = b.create<arith::DivUIOp>(loc, t, S);

  Value slope = b.create<memref::LoadOp>(loc, Slopes, ValueRange{h});  // f32
  // (j - i) in f32 via index -> i64 -> f32.
  auto toF32 = [&](Value idx) -> Value {
    Value asI64 = b.create<arith::IndexCastOp>(loc, i64, idx);
    return b.create<arith::SIToFPOp>(loc, f32, asI64);
  };
  Value dist = b.create<arith::SubFOp>(loc, toF32(j), toF32(i));
  Value y = b.create<arith::MulFOp>(loc, slope, dist);

  Value sv = isF32 ? y : b.create<arith::TruncFOp>(loc, storeTy, y);
  b.create<memref::StoreOp>(loc, sv, O, ValueRange{gid});

  b.setInsertionPointToEnd(&f.getBody().front());
  b.create<gpu::ReturnOp>(loc);
}

struct GenerateROCMAlibiKernelPass
    : PassWrapper<GenerateROCMAlibiKernelPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(GenerateROCMAlibiKernelPass)

  StringRef getArgument() const final { return "generate-rocm-alibi-kernel"; }
  StringRef getDescription() const final {
    return "Expand a tessera_rocm.alibi directive into a flat elementwise ALiBi "
           "positional-bias gpu kernel (compiler-generated)";
  }
  void getDependentDialects(DialectRegistry &registry) const final {
    registry.insert<gpu::GPUDialect, scf::SCFDialect, arith::ArithDialect,
                    memref::MemRefDialect>();
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();
    SmallVector<Operation *> directives;
    module.walk([&](Operation *op) {
      if (op->getName().getStringRef() == "tessera_rocm.alibi")
        directives.push_back(op);
    });
    for (Operation *op : directives) {
      auto nameAttr = op->getAttrOfType<StringAttr>("name");
      if (!nameAttr) {
        op->emitError("tessera_rocm.alibi missing name");
        return signalPassFailure();
      }
      OpBuilder b(module.getBodyRegion());
      b.setInsertionPointToEnd(module.getBody());
      Location loc = op->getLoc();
      std::string kname = nameAttr.getValue().str();

      Type storeTy = b.getF32Type();
      if (auto a = op->getAttrOfType<StringAttr>("dtype")) {
        StringRef dt = a.getValue();
        if (dt == "f16" || dt == "float16")
          storeTy = b.getF16Type();
        else if (dt == "bf16" || dt == "bfloat16")
          storeTy = b.getBF16Type();
        else if (dt != "f32" && dt != "float32") {
          op->emitError("generate-rocm-alibi-kernel: dtype must be f32, f16, or "
                        "bf16 (got '")
              << dt << "')";
          return signalPassFailure();
        }
      }

      auto gpuMod = b.create<gpu::GPUModuleOp>(loc, kname + "_mod");
      b.setInsertionPointToStart(&gpuMod.getBodyRegion().front());
      Type idxTy = b.getIndexType();
      auto slopeTy = MemRefType::get({ShapedType::kDynamic}, b.getF32Type());
      auto outTy = MemRefType::get({ShapedType::kDynamic}, storeTy);
      // (Slopes : memref<?xf32>, O : memref<?xstore>, H : index, S : index)
      auto fnTy = b.getFunctionType({slopeTy, outTy, idxTy, idxTy}, {});
      auto gpuFunc = b.create<gpu::GPUFuncOp>(loc, kname, fnTy);
      gpuFunc->setAttr(gpu::GPUDialect::getKernelFuncAttrName(),
                       b.getUnitAttr());
      OpBuilder body(gpuFunc.getContext());
      emitAlibiBody(body, loc, gpuFunc, storeTy);
      op->erase();
    }
  }
};

} // namespace

std::unique_ptr<mlir::Pass>
mlir::tessera_rocm::createGenerateROCMAlibiKernelPass() {
  return std::make_unique<GenerateROCMAlibiKernelPass>();
}
