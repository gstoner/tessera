//===- GenerateROCMBinaryLossKernel.cpp - BCE / asymmetric BCE ------------===//
//
// Expands `tessera_rocm.binary_loss` into a flat 2-operand elementwise per-
// element binary-cross-entropy-with-logits kernel over (z=logits, t=target):
//
//   kind 0 = bce            max(z,0) − z·t + log1p(exp(−|z|))
//   kind 1 = asymmetric_bce pw·t·softplus(−z) + nw·(1−t)·softplus(z)
//             softplus(±z) = max(±z,0) + log1p(exp(−|z|));  pw=nw=1 ⇒ bce.
//
// `kind`/`pw`/`nw` are compile-time attrs. The runtime reduces (none/mean/sum).
// f32 compute; log1p/exp lower through math→ROCDL. CPU analog: avx512_loss_f32.
//===----------------------------------------------------------------------===//

#include "TesseraROCM/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Pass/Pass.h"

using namespace mlir;

namespace {

static constexpr int64_t BD = 256;

void emitBinaryLossBody(OpBuilder &b, Location loc, gpu::GPUFuncOp f,
                        Type storeTy, int64_t kind, double pw, double nw) {
  Type f32 = b.getF32Type();
  bool isF32 = storeTy.isF32();
  auto slt = arith::CmpIPredicate::slt;
  b.setInsertionPointToStart(&f.getBody().front());
  Value Z = f.getArgument(0), T = f.getArgument(1), O = f.getArgument(2),
        N = f.getArgument(3);
  Value bid = b.create<gpu::BlockIdOp>(loc, gpu::Dimension::x);
  Value tid = b.create<gpu::ThreadIdOp>(loc, gpu::Dimension::x);
  Value cBD = b.create<arith::ConstantIndexOp>(loc, BD);
  Value gid = b.create<arith::AddIOp>(loc, b.create<arith::MulIOp>(loc, bid, cBD),
                                      tid);
  Value inb = b.create<arith::CmpIOp>(loc, slt, gid, N);
  auto ifo = b.create<scf::IfOp>(loc, inb, /*withElse=*/false);
  b.setInsertionPointToStart(ifo.thenBlock());

  Value rz = b.create<memref::LoadOp>(loc, Z, ValueRange{gid});
  Value rt = b.create<memref::LoadOp>(loc, T, ValueRange{gid});
  Value z = isF32 ? rz : b.create<arith::ExtFOp>(loc, f32, rz);
  Value t = isF32 ? rt : b.create<arith::ExtFOp>(loc, f32, rt);
  auto cst = [&](double v) {
    return b.create<arith::ConstantOp>(loc, f32, b.getF32FloatAttr(v))
        .getResult();
  };
  Value zero = cst(0.0), one = cst(1.0);
  Value az = b.create<math::AbsFOp>(loc, z);
  Value L = b.create<math::Log1pOp>(
      loc, b.create<math::ExpOp>(loc, b.create<arith::NegFOp>(loc, az)));
  Value sp_pos = b.create<arith::AddFOp>(loc, b.create<arith::MaximumFOp>(loc, z, zero), L);
  Value y;
  if (kind == 0) {  // bce: sp_pos - z*t
    y = b.create<arith::SubFOp>(loc, sp_pos, b.create<arith::MulFOp>(loc, z, t));
  } else {          // asym
    Value sp_neg = b.create<arith::AddFOp>(
        loc, b.create<arith::MaximumFOp>(loc, b.create<arith::NegFOp>(loc, z), zero), L);
    Value t1 = b.create<arith::MulFOp>(loc, b.create<arith::MulFOp>(loc, cst(pw), t), sp_neg);
    Value t2 = b.create<arith::MulFOp>(
        loc, b.create<arith::MulFOp>(loc, cst(nw), b.create<arith::SubFOp>(loc, one, t)),
        sp_pos);
    y = b.create<arith::AddFOp>(loc, t1, t2);
  }
  Value sv = isF32 ? y : b.create<arith::TruncFOp>(loc, storeTy, y);
  b.create<memref::StoreOp>(loc, sv, O, ValueRange{gid});
  b.setInsertionPointToEnd(&f.getBody().front());
  b.create<gpu::ReturnOp>(loc);
}

struct GenerateROCMBinaryLossKernelPass
    : PassWrapper<GenerateROCMBinaryLossKernelPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(GenerateROCMBinaryLossKernelPass)

  StringRef getArgument() const final {
    return "generate-rocm-binary-loss-kernel";
  }
  StringRef getDescription() const final {
    return "Expand a tessera_rocm.binary_loss directive into a flat 2-operand "
           "elementwise BCE / asymmetric-BCE gpu kernel";
  }
  void getDependentDialects(DialectRegistry &registry) const final {
    registry.insert<gpu::GPUDialect, scf::SCFDialect, arith::ArithDialect,
                    math::MathDialect, memref::MemRefDialect>();
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();
    SmallVector<Operation *> directives;
    module.walk([&](Operation *op) {
      if (op->getName().getStringRef() == "tessera_rocm.binary_loss")
        directives.push_back(op);
    });
    for (Operation *op : directives) {
      auto nameAttr = op->getAttrOfType<StringAttr>("name");
      if (!nameAttr) {
        op->emitError("tessera_rocm.binary_loss missing name");
        return signalPassFailure();
      }
      int64_t kind = 0;
      if (auto k = op->getAttrOfType<IntegerAttr>("kind")) kind = k.getInt();
      if (kind < 0 || kind > 1) {
        op->emitError("generate-rocm-binary-loss-kernel: kind must be 0 or 1");
        return signalPassFailure();
      }
      double pw = 1.0, nw = 1.0;
      if (auto a = op->getAttrOfType<FloatAttr>("pos_weight"))
        pw = a.getValueAsDouble();
      if (auto a = op->getAttrOfType<FloatAttr>("neg_weight"))
        nw = a.getValueAsDouble();
      OpBuilder b(module.getBodyRegion());
      b.setInsertionPointToEnd(module.getBody());
      Location loc = op->getLoc();
      std::string kname = nameAttr.getValue().str();
      Type storeTy = b.getF32Type();
      if (auto a = op->getAttrOfType<StringAttr>("dtype")) {
        StringRef dt = a.getValue();
        if (dt == "f16" || dt == "float16") storeTy = b.getF16Type();
        else if (dt == "bf16" || dt == "bfloat16") storeTy = b.getBF16Type();
        else if (dt != "f32" && dt != "float32") {
          op->emitError("generate-rocm-binary-loss-kernel: dtype must be f32, "
                        "f16, or bf16 (got '") << dt << "')";
          return signalPassFailure();
        }
      }
      auto gpuMod = b.create<gpu::GPUModuleOp>(loc, kname + "_mod");
      b.setInsertionPointToStart(&gpuMod.getBodyRegion().front());
      Type idxTy = b.getIndexType();
      auto memTy = MemRefType::get({ShapedType::kDynamic}, storeTy);
      auto fnTy = b.getFunctionType({memTy, memTy, memTy, idxTy}, {});
      auto gpuFunc = b.create<gpu::GPUFuncOp>(loc, kname, fnTy);
      gpuFunc->setAttr(gpu::GPUDialect::getKernelFuncAttrName(), b.getUnitAttr());
      OpBuilder body(gpuFunc.getContext());
      emitBinaryLossBody(body, loc, gpuFunc, storeTy, kind, pw, nw);
      op->erase();
    }
  }
};

} // namespace

std::unique_ptr<mlir::Pass>
mlir::tessera_rocm::createGenerateROCMBinaryLossKernelPass() {
  return std::make_unique<GenerateROCMBinaryLossKernelPass>();
}
