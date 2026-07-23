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

#include <limits>

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

void emitBinaryLossBackwardBody(OpBuilder &b, Location loc,
                                gpu::GPUFuncOp f, Type storeTy,
                                bool tensorCotangent) {
  Type f32 = b.getF32Type();
  bool isF32 = storeTy.isF32();
  b.setInsertionPointToStart(&f.getBody().front());
  Value logits = f.getArgument(0), target = f.getArgument(1),
        cotangent = f.getArgument(2), logitsGrad = f.getArgument(3),
        targetGrad = f.getArgument(4), n = f.getArgument(5),
        scale = f.getArgument(6);
  Value bid = b.create<gpu::BlockIdOp>(loc, gpu::Dimension::x);
  Value tid = b.create<gpu::ThreadIdOp>(loc, gpu::Dimension::x);
  Value block = b.create<arith::ConstantIndexOp>(loc, BD);
  Value gid = b.create<arith::AddIOp>(
      loc, b.create<arith::MulIOp>(loc, bid, block), tid);
  Value inBounds =
      b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::slt, gid, n);
  auto ifOp = b.create<scf::IfOp>(loc, inBounds, false);
  b.setInsertionPointToStart(ifOp.thenBlock());
  auto loadF32 = [&](Value memref, Value index) {
    Value raw = b.create<memref::LoadOp>(loc, memref, ValueRange{index});
    return isF32 ? raw : b.create<arith::ExtFOp>(loc, f32, raw).getResult();
  };
  Value z = loadF32(logits, gid);
  Value t = loadF32(target, gid);
  Value dyIndex = tensorCotangent
                      ? gid
                      : b.create<arith::ConstantIndexOp>(loc, 0).getResult();
  Value dy = loadF32(cotangent, dyIndex);
  auto cst = [&](double value) {
    return b.create<arith::ConstantOp>(
        loc, f32, b.getF32FloatAttr(value)).getResult();
  };
  Value zero = cst(0.0), one = cst(1.0);
  Value positive = b.create<arith::CmpFOp>(
      loc, arith::CmpFPredicate::OGE, z, zero);
  Value expNegative =
      b.create<math::ExpOp>(loc, b.create<arith::NegFOp>(loc, z));
  Value expPositive = b.create<math::ExpOp>(loc, z);
  Value sigmoidPositive = b.create<arith::DivFOp>(
      loc, one, b.create<arith::AddFOp>(loc, one, expNegative));
  Value sigmoidNegative = b.create<arith::DivFOp>(
      loc, expPositive, b.create<arith::AddFOp>(loc, one, expPositive));
  Value sigmoid = b.create<arith::SelectOp>(
      loc, positive, sigmoidPositive, sigmoidNegative);
  Value combined =
      b.create<arith::MulFOp>(loc, dy, scale);
  Value dz = b.create<arith::MulFOp>(
      loc, b.create<arith::SubFOp>(loc, sigmoid, t), combined);
  Value dt = b.create<arith::MulFOp>(
      loc, b.create<arith::NegFOp>(loc, z), combined);
  Value storedDz =
      isF32 ? dz : b.create<arith::TruncFOp>(loc, storeTy, dz).getResult();
  Value storedDt =
      isF32 ? dt : b.create<arith::TruncFOp>(loc, storeTy, dt).getResult();
  b.create<memref::StoreOp>(loc, storedDz, logitsGrad, ValueRange{gid});
  b.create<memref::StoreOp>(loc, storedDt, targetGrad, ValueRange{gid});
  b.setInsertionPointToEnd(&f.getBody().front());
  b.create<gpu::ReturnOp>(loc);
}

void emitClassLossBackwardBody(OpBuilder &b, Location loc,
                               gpu::GPUFuncOp f, Type storeTy,
                               double smoothing, int64_t ignoreIndex,
                               bool tensorCotangent) {
  Type f32 = b.getF32Type();
  bool isF32 = storeTy.isF32();
  b.setInsertionPointToStart(&f.getBody().front());
  Value logits = f.getArgument(0), targets = f.getArgument(1),
        cotangent = f.getArgument(2), gradient = f.getArgument(3),
        rows = f.getArgument(4), classes = f.getArgument(5),
        scale = f.getArgument(6);
  Value row = b.create<arith::AddIOp>(
      loc,
      b.create<arith::MulIOp>(
          loc, b.create<gpu::BlockIdOp>(loc, gpu::Dimension::x),
          b.create<arith::ConstantIndexOp>(loc, BD)),
      b.create<gpu::ThreadIdOp>(loc, gpu::Dimension::x));
  Value inBounds =
      b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::slt, row, rows);
  auto rowIf = b.create<scf::IfOp>(loc, inBounds, false);
  b.setInsertionPointToStart(rowIf.thenBlock());
  Value target = b.create<memref::LoadOp>(loc, targets, ValueRange{row});
  Value ignored = b.create<arith::CmpIOp>(
      loc, arith::CmpIPredicate::eq, target,
      b.create<arith::ConstantIntOp>(loc, ignoreIndex, 64));
  Value c0 = b.create<arith::ConstantIndexOp>(loc, 0);
  Value c1 = b.create<arith::ConstantIndexOp>(loc, 1);
  auto flatIndex = [&](Value klass) {
    return b.create<arith::AddIOp>(
        loc, b.create<arith::MulIOp>(loc, row, classes), klass).getResult();
  };
  auto loadLogit = [&](Value klass) {
    Value raw =
        b.create<memref::LoadOp>(loc, logits, ValueRange{flatIndex(klass)});
    return isF32 ? raw : b.create<arith::ExtFOp>(loc, f32, raw).getResult();
  };
  Value negInf = b.create<arith::ConstantOp>(
      loc, f32, b.getF32FloatAttr(-std::numeric_limits<float>::infinity()));
  auto maxLoop =
      b.create<scf::ForOp>(loc, c0, classes, c1, ValueRange{negInf});
  {
    OpBuilder::InsertionGuard guard(b);
    b.setInsertionPointToStart(maxLoop.getBody());
    Value value = loadLogit(maxLoop.getInductionVar());
    b.create<scf::YieldOp>(
        loc, b.create<arith::MaximumFOp>(
                 loc, maxLoop.getRegionIterArgs()[0], value)
                 .getResult());
  }
  Value maximum = maxLoop.getResult(0);
  Value zero = b.create<arith::ConstantOp>(
      loc, f32, b.getF32FloatAttr(0.0));
  auto sumLoop =
      b.create<scf::ForOp>(loc, c0, classes, c1, ValueRange{zero});
  {
    OpBuilder::InsertionGuard guard(b);
    b.setInsertionPointToStart(sumLoop.getBody());
    Value shifted = b.create<arith::SubFOp>(
        loc, loadLogit(sumLoop.getInductionVar()), maximum);
    Value next = b.create<arith::AddFOp>(
        loc, sumLoop.getRegionIterArgs()[0],
        b.create<math::ExpOp>(loc, shifted));
    b.create<scf::YieldOp>(loc, next);
  }
  Value denominator = sumLoop.getResult(0);
  Value dyIndex = tensorCotangent
                      ? row
                      : b.create<arith::ConstantIndexOp>(loc, 0).getResult();
  Value rawDy =
      b.create<memref::LoadOp>(loc, cotangent, ValueRange{dyIndex});
  Value dy = isF32 ? rawDy
                   : b.create<arith::ExtFOp>(loc, f32, rawDy).getResult();
  Value weighted = b.create<arith::MulFOp>(loc, dy, scale);
  Value targetIndex =
      b.create<arith::IndexCastOp>(loc, b.getIndexType(), target);
  Value one = b.create<arith::ConstantOp>(
      loc, f32, b.getF32FloatAttr(1.0));
  Value smooth = b.create<arith::ConstantOp>(
      loc, f32, b.getF32FloatAttr(smoothing));
  Value offTarget = zero;
  if (smoothing != 0.0) {
    Value classesF = b.create<arith::UIToFPOp>(
        loc, f32, b.create<arith::IndexCastOp>(
                      loc, b.getI64Type(), classes));
    offTarget = b.create<arith::DivFOp>(
        loc, smooth, b.create<arith::SubFOp>(loc, classesF, one));
  }
  Value onTarget = b.create<arith::SubFOp>(loc, one, smooth);
  auto storeLoop = b.create<scf::ForOp>(loc, c0, classes, c1);
  {
    OpBuilder::InsertionGuard guard(b);
    b.setInsertionPointToStart(storeLoop.getBody());
    Value klass = storeLoop.getInductionVar();
    Value probability = b.create<arith::DivFOp>(
        loc,
        b.create<math::ExpOp>(
            loc, b.create<arith::SubFOp>(loc, loadLogit(klass), maximum)),
        denominator);
    Value isTarget = b.create<arith::CmpIOp>(
        loc, arith::CmpIPredicate::eq, klass, targetIndex);
    Value distribution = b.create<arith::SelectOp>(
        loc, isTarget, onTarget, offTarget);
    Value local = b.create<arith::MulFOp>(
        loc, b.create<arith::SubFOp>(loc, probability, distribution),
        weighted);
    local = b.create<arith::SelectOp>(loc, ignored, zero, local);
    Value stored = isF32
                       ? local
                       : b.create<arith::TruncFOp>(loc, storeTy, local)
                             .getResult();
    b.create<memref::StoreOp>(
        loc, stored, gradient, ValueRange{flatIndex(klass)});
  }
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
      if (op->getName().getStringRef() == "tessera_rocm.binary_loss" ||
          op->getName().getStringRef() ==
              "tessera_rocm.class_loss_backward")
        directives.push_back(op);
    });
    for (Operation *op : directives) {
      auto nameAttr = op->getAttrOfType<StringAttr>("name");
      if (!nameAttr) {
        op->emitError("tessera_rocm.binary_loss missing name");
        return signalPassFailure();
      }
      bool classBackward =
          op->getName().getStringRef() ==
          "tessera_rocm.class_loss_backward";
      int64_t kind = 0;
      if (auto k = op->getAttrOfType<IntegerAttr>("kind")) kind = k.getInt();
      if (!classBackward && (kind < 0 || kind > 1)) {
        op->emitError("generate-rocm-binary-loss-kernel: kind must be 0 or 1");
        return signalPassFailure();
      }
      double pw = 1.0, nw = 1.0;
      if (auto a = op->getAttrOfType<FloatAttr>("pos_weight"))
        pw = a.getValueAsDouble();
      if (auto a = op->getAttrOfType<FloatAttr>("neg_weight"))
        nw = a.getValueAsDouble();
      bool backward = false;
      if (auto a = op->getAttrOfType<BoolAttr>("backward"))
        backward = a.getValue();
      StringRef reduction = "mean";
      if (auto a = op->getAttrOfType<StringAttr>("reduction"))
        reduction = a.getValue();
      if (classBackward)
        backward = true;
      if (!classBackward && backward && kind != 0) {
        op->emitError("generate-rocm-binary-loss-kernel: backward currently "
                      "supports BCE kind 0 only");
        return signalPassFailure();
      }
      if (reduction != "none" && reduction != "sum" &&
          reduction != "mean") {
        op->emitError("generate-rocm-binary-loss-kernel: backward reduction "
                      "must be none, sum, or mean");
        return signalPassFailure();
      }
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
      SmallVector<Type> inputs;
      if (classBackward) {
        auto targetTy =
            MemRefType::get({ShapedType::kDynamic}, b.getI64Type());
        inputs = {memTy, targetTy, memTy, memTy, idxTy, idxTy,
                  b.getF32Type()};
      } else if (backward)
        inputs = {memTy, memTy, memTy, memTy, memTy, idxTy, b.getF32Type()};
      else
        inputs = {memTy, memTy, memTy, idxTy};
      auto fnTy = b.getFunctionType(inputs, {});
      auto gpuFunc = b.create<gpu::GPUFuncOp>(loc, kname, fnTy);
      gpuFunc->setAttr(gpu::GPUDialect::getKernelFuncAttrName(), b.getUnitAttr());
      OpBuilder body(gpuFunc.getContext());
      if (classBackward) {
        double smoothing = 0.0;
        int64_t ignoreIndex = -100;
        if (auto attr = op->getAttrOfType<FloatAttr>("label_smoothing"))
          smoothing = attr.getValueAsDouble();
        if (auto attr = op->getAttrOfType<IntegerAttr>("ignore_index"))
          ignoreIndex = attr.getInt();
        emitClassLossBackwardBody(body, loc, gpuFunc, storeTy, smoothing,
                                  ignoreIndex, reduction == "none");
      } else if (backward)
        emitBinaryLossBackwardBody(body, loc, gpuFunc, storeTy,
                                   reduction == "none");
      else
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
