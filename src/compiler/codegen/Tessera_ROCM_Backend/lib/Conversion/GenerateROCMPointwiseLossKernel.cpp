//===- GenerateROCMPointwiseLossKernel.cpp - pointwise regression loss ----===//
//
// Expands a `tessera_rocm.pointwise_loss` directive into a flat 2-operand
// elementwise gpu kernel computing the PER-ELEMENT S11 regression loss over N
// elements (one thread per element). The runtime applies the reduction
// (none/mean/sum) via the rocm reduce lane.
//
//   kind 0 = mse        (p−t)²
//   kind 1 = mae        |p−t|
//   kind 2 = huber(δ)   a=|p−t|; a≤δ ? ½a² : δ(a−½δ)
//   kind 3 = smooth_l1(β) a=|p−t|; a<β ? ½a²/β : a−½β
//   kind 4 = log_cosh   e + log1p(exp(−2e)) − log2
//
// `kind` and `param` (δ/β) are compile-time attrs (no per-thread branch).
// Computes in f32 regardless of storage dtype; exp/log1p lower through
// convert-math-to-rocdl. The CPU analog is avx512_loss_f32.cpp. Validated vs the
// tessera.losses reference on gfx1151.
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

void emitLossBody(OpBuilder &b, Location loc, gpu::GPUFuncOp f, Type storeTy,
                  int64_t kind, double param) {
  Type f32 = b.getF32Type();
  bool isF32 = storeTy.isF32();
  auto slt = arith::CmpIPredicate::slt;

  b.setInsertionPointToStart(&f.getBody().front());
  Value A = f.getArgument(0), B = f.getArgument(1), O = f.getArgument(2),
        N = f.getArgument(3);

  Value bid = b.create<gpu::BlockIdOp>(loc, gpu::Dimension::x);
  Value tid = b.create<gpu::ThreadIdOp>(loc, gpu::Dimension::x);
  Value cBD = b.create<arith::ConstantIndexOp>(loc, BD);
  Value gid = b.create<arith::AddIOp>(loc, b.create<arith::MulIOp>(loc, bid, cBD),
                                      tid);
  Value inb = b.create<arith::CmpIOp>(loc, slt, gid, N);
  auto ifo = b.create<scf::IfOp>(loc, inb, /*withElse=*/false);
  b.setInsertionPointToStart(ifo.thenBlock());

  Value rp = b.create<memref::LoadOp>(loc, A, ValueRange{gid});
  Value rt = b.create<memref::LoadOp>(loc, B, ValueRange{gid});
  Value p = isF32 ? rp : b.create<arith::ExtFOp>(loc, f32, rp);
  Value t = isF32 ? rt : b.create<arith::ExtFOp>(loc, f32, rt);
  auto cst = [&](double v) {
    return b.create<arith::ConstantOp>(loc, f32, b.getF32FloatAttr(v))
        .getResult();
  };
  Value e = b.create<arith::SubFOp>(loc, p, t);
  Value ax = b.create<math::AbsFOp>(loc, e);
  Value half = cst(0.5);
  Value y;
  switch (kind) {
  case 0:  // mse
    y = b.create<arith::MulFOp>(loc, e, e);
    break;
  case 1:  // mae
    y = ax;
    break;
  case 2: {  // huber(delta)
    Value d = cst(param);
    Value le = b.create<arith::CmpFOp>(loc, arith::CmpFPredicate::OLE, ax, d);
    Value q = b.create<arith::MulFOp>(loc, half,
                                      b.create<arith::MulFOp>(loc, ax, ax));
    Value lin = b.create<arith::MulFOp>(
        loc, d, b.create<arith::SubFOp>(loc, ax,
                                        b.create<arith::MulFOp>(loc, half, d)));
    y = b.create<arith::SelectOp>(loc, le, q, lin);
    break;
  }
  case 3: {  // smooth_l1(beta)
    Value beta = cst(param);
    Value lt = b.create<arith::CmpFOp>(loc, arith::CmpFPredicate::OLT, ax, beta);
    Value q = b.create<arith::DivFOp>(
        loc, b.create<arith::MulFOp>(loc, half,
                                     b.create<arith::MulFOp>(loc, ax, ax)),
        beta);
    Value lin = b.create<arith::SubFOp>(
        loc, ax, b.create<arith::MulFOp>(loc, half, beta));
    y = b.create<arith::SelectOp>(loc, lt, q, lin);
    break;
  }
  default: {  // log_cosh: e + log1p(exp(-2e)) - log2
    Value two = cst(2.0);
    Value nege2 = b.create<arith::NegFOp>(loc, b.create<arith::MulFOp>(loc, two, e));
    Value l = b.create<math::Log1pOp>(loc, b.create<math::ExpOp>(loc, nege2));
    y = b.create<arith::SubFOp>(loc, b.create<arith::AddFOp>(loc, e, l),
                                cst(0.6931471805599453));
    break;
  }
  }

  Value sv = isF32 ? y : b.create<arith::TruncFOp>(loc, storeTy, y);
  b.create<memref::StoreOp>(loc, sv, O, ValueRange{gid});

  b.setInsertionPointToEnd(&f.getBody().front());
  b.create<gpu::ReturnOp>(loc);
}

void emitLossBackwardBody(OpBuilder &b, Location loc, gpu::GPUFuncOp f,
                          Type storeTy, int64_t kind, double param,
                          bool tensorCotangent, bool trainingSGD,
                          bool trainingAdamW) {
  Type f32 = b.getF32Type();
  bool isF32 = storeTy.isF32();
  b.setInsertionPointToStart(&f.getBody().front());
  Value prediction = f.getArgument(0), target = f.getArgument(1),
        cotangent = f.getArgument(2);
  Value parameterBuffer, moment1Buffer, moment2Buffer, predictionGrad,
      newParameter, newMoment1, newMoment2, targetGrad, n, scale,
      learningRate, beta1, beta2, epsilon, weightDecay, correction1,
      correction2;
  if (trainingAdamW) {
    parameterBuffer = f.getArgument(3);
    moment1Buffer = f.getArgument(4);
    moment2Buffer = f.getArgument(5);
    newParameter = f.getArgument(6);
    newMoment1 = f.getArgument(7);
    newMoment2 = f.getArgument(8);
    targetGrad = f.getArgument(9);
    n = f.getArgument(10);
    scale = f.getArgument(11);
    learningRate = f.getArgument(12);
    beta1 = f.getArgument(13);
    beta2 = f.getArgument(14);
    epsilon = f.getArgument(15);
    weightDecay = f.getArgument(16);
    correction1 = f.getArgument(17);
    correction2 = f.getArgument(18);
  } else if (trainingSGD) {
    parameterBuffer = f.getArgument(3);
    newParameter = f.getArgument(4);
    targetGrad = f.getArgument(5);
    n = f.getArgument(6);
    scale = f.getArgument(7);
    learningRate = f.getArgument(8);
  } else {
    predictionGrad = f.getArgument(3);
    targetGrad = f.getArgument(4);
    n = f.getArgument(5);
    scale = f.getArgument(6);
  }

  Value bid = b.create<gpu::BlockIdOp>(loc, gpu::Dimension::x);
  Value tid = b.create<gpu::ThreadIdOp>(loc, gpu::Dimension::x);
  Value block = b.create<arith::ConstantIndexOp>(loc, BD);
  Value gid = b.create<arith::AddIOp>(
      loc, b.create<arith::MulIOp>(loc, bid, block), tid);
  Value inBounds =
      b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::slt, gid, n);
  auto ifOp = b.create<scf::IfOp>(loc, inBounds, /*withElse=*/false);
  b.setInsertionPointToStart(ifOp.thenBlock());

  auto loadF32 = [&](Value memref, Value index) {
    Value raw = b.create<memref::LoadOp>(loc, memref, ValueRange{index});
    return isF32 ? raw : b.create<arith::ExtFOp>(loc, f32, raw).getResult();
  };
  Value p = loadF32(prediction, gid);
  Value t = loadF32(target, gid);
  Value dyIndex = tensorCotangent
                      ? gid
                      : b.create<arith::ConstantIndexOp>(loc, 0).getResult();
  Value dy = loadF32(cotangent, dyIndex);
  Value error = b.create<arith::SubFOp>(loc, p, t);
  auto cst = [&](double value) {
    return b.create<arith::ConstantOp>(
        loc, f32, b.getF32FloatAttr(value)).getResult();
  };
  Value zero = cst(0.0);
  Value one = cst(1.0);
  Value negativeOne = cst(-1.0);
  Value positive = b.create<arith::CmpFOp>(
      loc, arith::CmpFPredicate::OGT, error, zero);
  Value negative = b.create<arith::CmpFOp>(
      loc, arith::CmpFPredicate::OLT, error, zero);
  Value sign = b.create<arith::SelectOp>(
      loc, positive, one,
      b.create<arith::SelectOp>(loc, negative, negativeOne, zero));
  Value localGradient;
  switch (kind) {
  case 0:
    localGradient = b.create<arith::MulFOp>(loc, error, cst(2.0));
    break;
  case 1:
    localGradient = sign;
    break;
  case 2: {
    Value transition = cst(param);
    Value absError = b.create<math::AbsFOp>(loc, error);
    Value inside = b.create<arith::CmpFOp>(
        loc, arith::CmpFPredicate::OLE, absError, transition);
    Value outside =
        b.create<arith::MulFOp>(loc, transition, sign);
    localGradient =
        b.create<arith::SelectOp>(loc, inside, error, outside);
    break;
  }
  case 3: {
    Value transition = cst(param);
    Value absError = b.create<math::AbsFOp>(loc, error);
    Value inside = b.create<arith::CmpFOp>(
        loc, arith::CmpFPredicate::OLT, absError, transition);
    Value insideGradient =
        b.create<arith::DivFOp>(loc, error, transition);
    localGradient =
        b.create<arith::SelectOp>(loc, inside, insideGradient, sign);
    break;
  }
  case 4: {
    // Training-only kind 4 is BCE-with-logits. The forward pointwise-loss
    // directive retains kind 4 as log-cosh.
    Value nonnegative = b.create<arith::CmpFOp>(
        loc, arith::CmpFPredicate::OGE, p, zero);
    Value expNegative =
        b.create<math::ExpOp>(loc, b.create<arith::NegFOp>(loc, p));
    Value expPositive = b.create<math::ExpOp>(loc, p);
    Value sigmoidPositive = b.create<arith::DivFOp>(
        loc, one, b.create<arith::AddFOp>(loc, one, expNegative));
    Value sigmoidNegative = b.create<arith::DivFOp>(
        loc, expPositive, b.create<arith::AddFOp>(loc, one, expPositive));
    Value sigmoid = b.create<arith::SelectOp>(
        loc, nonnegative, sigmoidPositive, sigmoidNegative);
    localGradient = b.create<arith::SubFOp>(loc, sigmoid, t);
    break;
  }
  default:
    localGradient = zero;
    break;
  }
  Value grad = b.create<arith::MulFOp>(
      loc, b.create<arith::MulFOp>(loc, localGradient, dy), scale);
  Value targetValue;
  if ((trainingSGD || trainingAdamW) && kind == 4) {
    targetValue = b.create<arith::MulFOp>(
        loc, b.create<arith::NegFOp>(loc, p),
        b.create<arith::MulFOp>(loc, dy, scale));
  } else {
    targetValue = b.create<arith::NegFOp>(loc, grad);
  }
  Value storedGrad =
      isF32 ? grad : b.create<arith::TruncFOp>(loc, storeTy, grad).getResult();
  Value storedTarget =
      isF32 ? targetValue
            : b.create<arith::TruncFOp>(loc, storeTy, targetValue).getResult();
  if (trainingAdamW) {
    Value parameterValue = loadF32(parameterBuffer, gid);
    Value moment1Value = loadF32(moment1Buffer, gid);
    Value moment2Value = loadF32(moment2Buffer, gid);
    Value oneMinusBeta1 = b.create<arith::SubFOp>(loc, one, beta1);
    Value oneMinusBeta2 = b.create<arith::SubFOp>(loc, one, beta2);
    Value updatedMoment1 = b.create<arith::AddFOp>(
        loc, b.create<arith::MulFOp>(loc, beta1, moment1Value),
        b.create<arith::MulFOp>(loc, oneMinusBeta1, grad));
    Value updatedMoment2 = b.create<arith::AddFOp>(
        loc, b.create<arith::MulFOp>(loc, beta2, moment2Value),
        b.create<arith::MulFOp>(
            loc, oneMinusBeta2, b.create<arith::MulFOp>(loc, grad, grad)));
    Value correctedMoment1 =
        b.create<arith::DivFOp>(loc, updatedMoment1, correction1);
    Value correctedMoment2 =
        b.create<arith::DivFOp>(loc, updatedMoment2, correction2);
    Value denominator = b.create<arith::AddFOp>(
        loc, b.create<math::SqrtOp>(loc, correctedMoment2), epsilon);
    Value update = b.create<arith::AddFOp>(
        loc, b.create<arith::DivFOp>(loc, correctedMoment1, denominator),
        b.create<arith::MulFOp>(loc, weightDecay, parameterValue));
    Value updatedParameter = b.create<arith::SubFOp>(
        loc, parameterValue,
        b.create<arith::MulFOp>(loc, learningRate, update));
    auto store = [&](Value value, Value buffer) {
      Value stored =
          isF32 ? value
                : b.create<arith::TruncFOp>(loc, storeTy, value).getResult();
      b.create<memref::StoreOp>(loc, stored, buffer, ValueRange{gid});
    };
    store(updatedParameter, newParameter);
    store(updatedMoment1, newMoment1);
    store(updatedMoment2, newMoment2);
  } else if (trainingSGD) {
    Value parameterValue = loadF32(parameterBuffer, gid);
    Value updated = b.create<arith::SubFOp>(
        loc, parameterValue,
        b.create<arith::MulFOp>(loc, learningRate, grad));
    Value storedParameter =
        isF32
            ? updated
            : b.create<arith::TruncFOp>(loc, storeTy, updated).getResult();
    b.create<memref::StoreOp>(
        loc, storedParameter, newParameter, ValueRange{gid});
  } else {
    b.create<memref::StoreOp>(
        loc, storedGrad, predictionGrad, ValueRange{gid});
  }
  b.create<memref::StoreOp>(loc, storedTarget, targetGrad, ValueRange{gid});

  b.setInsertionPointToEnd(&f.getBody().front());
  b.create<gpu::ReturnOp>(loc);
}

struct GenerateROCMPointwiseLossKernelPass
    : PassWrapper<GenerateROCMPointwiseLossKernelPass,
                  OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(
      GenerateROCMPointwiseLossKernelPass)

  StringRef getArgument() const final {
    return "generate-rocm-pointwise-loss-kernel";
  }
  StringRef getDescription() const final {
    return "Expand a tessera_rocm.pointwise_loss directive into a flat "
           "2-operand elementwise per-element regression-loss gpu kernel";
  }
  void getDependentDialects(DialectRegistry &registry) const final {
    registry.insert<gpu::GPUDialect, scf::SCFDialect, arith::ArithDialect,
                    math::MathDialect, memref::MemRefDialect>();
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();
    SmallVector<Operation *> directives;
    module.walk([&](Operation *op) {
      if (op->getName().getStringRef() == "tessera_rocm.pointwise_loss")
        directives.push_back(op);
    });
    for (Operation *op : directives) {
      auto nameAttr = op->getAttrOfType<StringAttr>("name");
      if (!nameAttr) {
        op->emitError("tessera_rocm.pointwise_loss missing name");
        return signalPassFailure();
      }
      int64_t kind = 0;
      if (auto k = op->getAttrOfType<IntegerAttr>("kind"))
        kind = k.getInt();
      if (kind < 0 || kind > 4) {
        op->emitError("generate-rocm-pointwise-loss-kernel: kind must be 0..4");
        return signalPassFailure();
      }
      double param = 1.0;
      if (auto pp = op->getAttrOfType<FloatAttr>("param"))
        param = pp.getValueAsDouble();
      bool backward = false;
      if (auto attr = op->getAttrOfType<BoolAttr>("backward"))
        backward = attr.getValue();
      bool trainingSGD = false;
      if (auto attr = op->getAttrOfType<BoolAttr>("training_sgd"))
        trainingSGD = attr.getValue();
      bool trainingAdamW = false;
      if (auto attr = op->getAttrOfType<BoolAttr>("training_adamw"))
        trainingAdamW = attr.getValue();
      if (trainingSGD && trainingAdamW) {
        op->emitError("generate-rocm-pointwise-loss-kernel: training_sgd and "
                      "training_adamw are mutually exclusive");
        return signalPassFailure();
      }
      backward = backward || trainingSGD || trainingAdamW;
      StringRef reduction = "mean";
      if (auto attr = op->getAttrOfType<StringAttr>("reduction"))
        reduction = attr.getValue();
      if (backward && reduction != "none" && reduction != "sum" &&
          reduction != "mean") {
        op->emitError("generate-rocm-pointwise-loss-kernel: backward reduction "
                      "must be none, sum, or mean");
        return signalPassFailure();
      }
      if (backward && kind > 3 && !trainingSGD && !trainingAdamW) {
        op->emitError("generate-rocm-pointwise-loss-kernel: compiled backward "
                      "supports MSE/MAE/Huber/Smooth-L1 kinds 0..3");
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
          op->emitError("generate-rocm-pointwise-loss-kernel: dtype must be "
                        "f32, f16, or bf16 (got '") << dt << "')";
          return signalPassFailure();
        }
      }

      auto gpuMod = b.create<gpu::GPUModuleOp>(loc, kname + "_mod");
      b.setInsertionPointToStart(&gpuMod.getBodyRegion().front());
      Type idxTy = b.getIndexType();
      auto memTy = MemRefType::get({ShapedType::kDynamic}, storeTy);
      SmallVector<Type> inputs;
      if (trainingAdamW)
        inputs = {memTy, memTy, memTy, memTy, memTy, memTy, memTy, memTy,
                  memTy, memTy, idxTy, b.getF32Type(), b.getF32Type(),
                  b.getF32Type(), b.getF32Type(), b.getF32Type(),
                  b.getF32Type(), b.getF32Type(), b.getF32Type()};
      else if (trainingSGD)
        inputs = {memTy, memTy, memTy, memTy, memTy, memTy, idxTy,
                  b.getF32Type(), b.getF32Type()};
      else if (backward)
        inputs = {memTy, memTy, memTy, memTy, memTy, idxTy, b.getF32Type()};
      else
        inputs = {memTy, memTy, memTy, idxTy};
      auto fnTy = b.getFunctionType(inputs, {});
      auto gpuFunc = b.create<gpu::GPUFuncOp>(loc, kname, fnTy);
      gpuFunc->setAttr(gpu::GPUDialect::getKernelFuncAttrName(),
                       b.getUnitAttr());
      OpBuilder body(gpuFunc.getContext());
      if (backward)
        emitLossBackwardBody(body, loc, gpuFunc, storeTy, kind, param,
                             reduction == "none", trainingSGD, trainingAdamW);
      else
        emitLossBody(body, loc, gpuFunc, storeTy, kind, param);
      op->erase();
    }
  }
};

} // namespace

std::unique_ptr<mlir::Pass>
mlir::tessera_rocm::createGenerateROCMPointwiseLossKernelPass() {
  return std::make_unique<GenerateROCMPointwiseLossKernelPass>();
}
