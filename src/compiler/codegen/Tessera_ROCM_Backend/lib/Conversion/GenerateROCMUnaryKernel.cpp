//===- GenerateROCMUnaryKernel.cpp - elementwise unary math kernel -------===//
//
// Expands a `tessera_rocm.unary` directive into a flat elementwise gpu kernel
// applying a pointwise unary math function over N elements (one thread per
// element, strided grid) — the standalone S2 scalar-math / stability family,
// the unary sibling of the activation lane:
//
//   algebraic:   sqrt, rsqrt, reciprocal, abs, neg, sign
//   transcend.:  exp, log, erf, tanh, sigmoid
//   stability:   log1p, expm1, softplus   (softplus stable: log1p(exp(-|x|))+max(x,0))
//
// Computes in f32 regardless of storage dtype; the transcendentals lower
// through convert-math-to-rocdl. N is a runtime index arg. Validated vs a
// numpy reference on gfx1151.
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
#include "llvm/ADT/StringSet.h"
#include "llvm/ADT/StringSwitch.h"

using namespace mlir;

namespace {

static constexpr int64_t BD = 256;

enum class Un {
  Exp, Log, Sqrt, Rsqrt, Reciprocal, Abs, Neg, Sign,
  Erf, Tanh, Sigmoid, Log1p, Expm1, Softplus
};

static Value cst(OpBuilder &b, Location loc, Type f32, float v) {
  return b.create<arith::ConstantOp>(loc, f32, b.getF32FloatAttr(v));
}

void emitUnaryBody(OpBuilder &b, Location loc, gpu::GPUFuncOp f, Type storeTy,
                   Un un) {
  Type f32 = b.getF32Type();
  bool isF32 = storeTy.isF32();
  auto slt = arith::CmpIPredicate::slt;
  auto ogt = arith::CmpFPredicate::OGT;
  auto olt = arith::CmpFPredicate::OLT;

  b.setInsertionPointToStart(&f.getBody().front());
  Value X = f.getArgument(0), O = f.getArgument(1), N = f.getArgument(2);

  Value bid = b.create<gpu::BlockIdOp>(loc, gpu::Dimension::x);
  Value tid = b.create<gpu::ThreadIdOp>(loc, gpu::Dimension::x);
  Value cBD = b.create<arith::ConstantIndexOp>(loc, BD);
  Value gid = b.create<arith::AddIOp>(loc, b.create<arith::MulIOp>(loc, bid, cBD),
                                      tid);
  Value inb = b.create<arith::CmpIOp>(loc, slt, gid, N);
  auto ifo = b.create<scf::IfOp>(loc, inb, /*withElse=*/false);
  b.setInsertionPointToStart(ifo.thenBlock());

  Value raw = b.create<memref::LoadOp>(loc, X, ValueRange{gid});
  Value x = isF32 ? raw : b.create<arith::ExtFOp>(loc, f32, raw);
  Value one = cst(b, loc, f32, 1.0f);
  Value zero = cst(b, loc, f32, 0.0f);
  Value y;
  switch (un) {
  case Un::Exp:
    y = b.create<math::ExpOp>(loc, x);
    break;
  case Un::Log:
    y = b.create<math::LogOp>(loc, x);
    break;
  case Un::Sqrt:
    y = b.create<math::SqrtOp>(loc, x);
    break;
  case Un::Rsqrt:
    y = b.create<math::RsqrtOp>(loc, x);
    break;
  case Un::Reciprocal:
    y = b.create<arith::DivFOp>(loc, one, x);
    break;
  case Un::Abs:
    y = b.create<math::AbsFOp>(loc, x);
    break;
  case Un::Neg:
    y = b.create<arith::NegFOp>(loc, x);
    break;
  case Un::Sign: {
    Value pos = b.create<arith::CmpFOp>(loc, ogt, x, zero);
    Value neg = b.create<arith::CmpFOp>(loc, olt, x, zero);
    Value negOne = cst(b, loc, f32, -1.0f);
    Value s = b.create<arith::SelectOp>(loc, neg, negOne, zero);
    y = b.create<arith::SelectOp>(loc, pos, one, s);
    break;
  }
  case Un::Erf:
    y = b.create<math::ErfOp>(loc, x);
    break;
  case Un::Tanh:
    y = b.create<math::TanhOp>(loc, x);
    break;
  case Un::Sigmoid: {
    Value e = b.create<math::ExpOp>(loc, b.create<arith::NegFOp>(loc, x));
    y = b.create<arith::DivFOp>(loc, one,
                                b.create<arith::AddFOp>(loc, one, e));
    break;
  }
  case Un::Log1p:
    y = b.create<math::Log1pOp>(loc, x);
    break;
  case Un::Expm1:
    y = b.create<math::ExpM1Op>(loc, x);
    break;
  case Un::Softplus: {
    // Stable: log1p(exp(-|x|)) + max(x, 0)
    Value ax = b.create<math::AbsFOp>(loc, x);
    Value e = b.create<math::ExpOp>(loc, b.create<arith::NegFOp>(loc, ax));
    Value lp = b.create<math::Log1pOp>(loc, e);
    Value mx = b.create<arith::MaximumFOp>(loc, x, zero);
    y = b.create<arith::AddFOp>(loc, lp, mx);
    break;
  }
  }
  Value sv = isF32 ? y : b.create<arith::TruncFOp>(loc, storeTy, y);
  b.create<memref::StoreOp>(loc, sv, O, ValueRange{gid});

  b.setInsertionPointToEnd(&f.getBody().front());
  b.create<gpu::ReturnOp>(loc);
}

struct GenerateROCMUnaryKernelPass
    : PassWrapper<GenerateROCMUnaryKernelPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(GenerateROCMUnaryKernelPass)

  StringRef getArgument() const final { return "generate-rocm-unary-kernel"; }
  StringRef getDescription() const final {
    return "Expand a tessera_rocm.unary directive into a flat elementwise unary "
           "math gpu kernel (compiler-generated)";
  }
  void getDependentDialects(DialectRegistry &registry) const final {
    registry.insert<gpu::GPUDialect, scf::SCFDialect, arith::ArithDialect,
                    math::MathDialect, memref::MemRefDialect>();
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();
    SmallVector<Operation *> directives;
    module.walk([&](Operation *op) {
      if (op->getName().getStringRef() == "tessera_rocm.unary")
        directives.push_back(op);
    });
    for (Operation *op : directives) {
      auto nameAttr = op->getAttrOfType<StringAttr>("name");
      if (!nameAttr) {
        op->emitError("tessera_rocm.unary missing name");
        return signalPassFailure();
      }
      StringRef kindStr = "exp";
      if (auto a = op->getAttrOfType<StringAttr>("kind"))
        kindStr = a.getValue();
      Un un = llvm::StringSwitch<Un>(kindStr)
                  .Case("exp", Un::Exp)
                  .Case("log", Un::Log)
                  .Case("sqrt", Un::Sqrt)
                  .Case("rsqrt", Un::Rsqrt)
                  .Case("reciprocal", Un::Reciprocal)
                  .Case("abs", Un::Abs)
                  .Case("neg", Un::Neg)
                  .Case("sign", Un::Sign)
                  .Case("erf", Un::Erf)
                  .Case("tanh", Un::Tanh)
                  .Case("sigmoid", Un::Sigmoid)
                  .Case("log1p", Un::Log1p)
                  .Case("expm1", Un::Expm1)
                  .Case("softplus", Un::Softplus)
                  .Default(Un::Exp);
      static const llvm::StringSet<> kValid = {
          "exp",  "log",     "sqrt",  "rsqrt", "reciprocal", "abs",  "neg",
          "sign", "erf",     "tanh",  "sigmoid", "log1p",    "expm1",
          "softplus"};
      if (!kValid.contains(kindStr)) {
        op->emitError("generate-rocm-unary-kernel: unknown kind '")
            << kindStr << "' (exp/log/sqrt/rsqrt/reciprocal/abs/neg/sign/erf/"
                          "tanh/sigmoid/log1p/expm1/softplus)";
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
          op->emitError("generate-rocm-unary-kernel: dtype must be f32, f16, "
                        "or bf16 (got '")
              << dt << "')";
          return signalPassFailure();
        }
      }

      auto gpuMod = b.create<gpu::GPUModuleOp>(loc, kname + "_mod");
      b.setInsertionPointToStart(&gpuMod.getBodyRegion().front());
      Type idxTy = b.getIndexType();
      auto memTy = MemRefType::get({ShapedType::kDynamic}, storeTy);
      // (X, O : memref<?xstore>, N : index)
      auto fnTy = b.getFunctionType({memTy, memTy, idxTy}, {});
      auto gpuFunc = b.create<gpu::GPUFuncOp>(loc, kname, fnTy);
      gpuFunc->setAttr(gpu::GPUDialect::getKernelFuncAttrName(),
                       b.getUnitAttr());
      OpBuilder body(gpuFunc.getContext());
      emitUnaryBody(body, loc, gpuFunc, storeTy, un);
      op->erase();
    }
  }
};

} // namespace

std::unique_ptr<mlir::Pass>
mlir::tessera_rocm::createGenerateROCMUnaryKernelPass() {
  return std::make_unique<GenerateROCMUnaryKernelPass>();
}
