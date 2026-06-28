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

#include <limits>
#include "llvm/ADT/StringSet.h"
#include "llvm/ADT/StringSwitch.h"

using namespace mlir;

namespace {

static constexpr int64_t BD = 256;

enum class Un {
  Exp, Log, Sqrt, Rsqrt, Reciprocal, Abs, Neg, Sign,
  Erf, Tanh, Sigmoid, Log1p, Expm1, Softplus,
  // tail: trig / special / rounding (2026-06-26)
  Cos, Tan, Sinh, Cosh, Asin, Acos, Atan, Erfc,
  Floor, Ceil, Round, Trunc,
  Sin, Lgamma, Digamma
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
  auto ole = arith::CmpFPredicate::OLE;

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
  case Un::Sin:
    y = b.create<math::SinOp>(loc, x);
    break;
  case Un::Lgamma: {
    // ln Γ(x): Numerical Recipes `gammln` — Lanczos g=5 (matches the x86
    // lgamma512 SIMD core and math.lgamma). Valid for x > 0; reflection for
    // x < 0.5 via ln Γ(x) = ln(π) - ln|sin(πx)| - ln Γ(1-x).
    static const float kCof[6] = {
        76.18009172947146f,    -86.50532032941677f,    24.01409824083091f,
        -1.231739572450155f,    0.1208650973866179e-2f, -0.5395239384953e-5f};
    Value half = cst(b, loc, f32, 0.5f);
    Value isRefl = b.create<arith::CmpFOp>(loc, olt, x, half);
    // xc = x>=0.5 ? x : 1-x   (the Lanczos arg, always >= 0.5)
    Value oneMinusX = b.create<arith::SubFOp>(loc, one, x);
    Value xc = b.create<arith::SelectOp>(loc, isRefl, oneMinusX, x);
    // ser = 1.000000000190015 + Σ cof[j]/(xc+1+j)
    Value ser = cst(b, loc, f32, 1.000000000190015f);
    Value yk = xc;
    for (int j = 0; j < 6; ++j) {
      yk = b.create<arith::AddFOp>(loc, yk, one);
      Value term = b.create<arith::DivFOp>(loc, cst(b, loc, f32, kCof[j]), yk);
      ser = b.create<arith::AddFOp>(loc, ser, term);
    }
    // tmp = (xc+5.5) - (xc+0.5)*log(xc+5.5)
    Value tmp = b.create<arith::AddFOp>(loc, xc, cst(b, loc, f32, 5.5f));
    Value lt = b.create<math::LogOp>(loc, tmp);
    Value xcHalf = b.create<arith::AddFOp>(loc, xc, half);
    tmp = b.create<arith::SubFOp>(loc, tmp,
                                  b.create<arith::MulFOp>(loc, xcHalf, lt));
    // core = -tmp + log(√(2π) * ser / xc)   == lnΓ(xc)
    Value sp = cst(b, loc, f32, 2.5066282746310005f);
    Value arg = b.create<arith::DivFOp>(
        loc, b.create<arith::MulFOp>(loc, sp, ser), xc);
    Value core = b.create<arith::AddFOp>(
        loc, b.create<arith::NegFOp>(loc, tmp), b.create<math::LogOp>(loc, arg));
    // reflected = ln(π) - ln|sin(πx)| - core
    Value pi = cst(b, loc, f32, 3.14159265358979324f);
    Value sinPiX = b.create<math::SinOp>(loc, b.create<arith::MulFOp>(loc, pi, x));
    Value lnAbsSin = b.create<math::LogOp>(loc, b.create<math::AbsFOp>(loc, sinPiX));
    Value reflected = b.create<arith::SubFOp>(
        loc,
        b.create<arith::SubFOp>(loc, b.create<math::LogOp>(loc, pi), lnAbsSin),
        core);
    y = b.create<arith::SelectOp>(loc, isRefl, reflected, core);
    break;
  }
  case Un::Digamma: {
    // ψ(x): recurrence to xx>=8 then the asymptotic series (matches the x86
    // digamma512 core and tessera.ops.digamma). For x<=0, reflect via
    // ψ(x) = ψ(1-x) - π/tan(πx); non-positive integers (poles) return NaN.
    Value zero = cst(b, loc, f32, 0.0f);
    Value eight = cst(b, loc, f32, 8.0f);
    Value pi = cst(b, loc, f32, 3.14159265358979324f);
    Value isRefl = b.create<arith::CmpFOp>(loc, ole, x, zero);
    Value oneMinusX = b.create<arith::SubFOp>(loc, one, x);
    Value xw = b.create<arith::SelectOp>(loc, isRefl, oneMinusX, x);  // > 0
    // recurrence: 8 unrolled masked steps reach xx>=8 for any xw>0.
    Value result = zero;
    Value xx = xw;
    for (int k = 0; k < 8; ++k) {
      Value m = b.create<arith::CmpFOp>(loc, olt, xx, eight);
      Value recip = b.create<arith::DivFOp>(loc, one, xx);
      Value rsub = b.create<arith::SubFOp>(loc, result, recip);
      result = b.create<arith::SelectOp>(loc, m, rsub, result);
      Value xinc = b.create<arith::AddFOp>(loc, xx, one);
      xx = b.create<arith::SelectOp>(loc, m, xinc, xx);
    }
    Value inv = b.create<arith::DivFOp>(loc, one, xx);
    Value inv2 = b.create<arith::MulFOp>(loc, inv, inv);
    Value inv4 = b.create<arith::MulFOp>(loc, inv2, inv2);
    Value inv6 = b.create<arith::MulFOp>(loc, inv4, inv2);
    Value inv8 = b.create<arith::MulFOp>(loc, inv4, inv4);
    // log(xx) - inv/2 - inv2/12 + inv4/120 - inv6/252 + inv8/240
    Value core = b.create<arith::AddFOp>(loc, result, b.create<math::LogOp>(loc, xx));
    auto fma = [&](Value acc, float c, Value t, bool add) {
      Value term = b.create<arith::MulFOp>(loc, cst(b, loc, f32, c), t);
      return add ? b.create<arith::AddFOp>(loc, acc, term).getResult()
                 : b.create<arith::SubFOp>(loc, acc, term).getResult();
    };
    core = fma(core, 0.5f, inv, false);
    core = fma(core, 1.0f / 12.0f, inv2, false);
    core = fma(core, 1.0f / 120.0f, inv4, true);
    core = fma(core, 1.0f / 252.0f, inv6, false);
    core = fma(core, 1.0f / 240.0f, inv8, true);
    // reflection: ψ(x) = core(1-x) - π/tan(πx)
    Value tanPiX = b.create<math::TanOp>(loc, b.create<arith::MulFOp>(loc, pi, x));
    Value reflected = b.create<arith::SubFOp>(
        loc, core, b.create<arith::DivFOp>(loc, pi, tanPiX));
    Value yv = b.create<arith::SelectOp>(loc, isRefl, reflected, core);
    // pole at non-positive integer -> NaN
    Value rnd = b.create<math::RoundOp>(loc, x);
    Value frac = b.create<math::AbsFOp>(loc, b.create<arith::SubFOp>(loc, x, rnd));
    Value isInt = b.create<arith::CmpFOp>(loc, olt, frac, cst(b, loc, f32, 1e-4f));
    Value isPole = b.create<arith::AndIOp>(loc, isRefl, isInt);
    Value nan = cst(b, loc, f32, std::numeric_limits<float>::quiet_NaN());
    y = b.create<arith::SelectOp>(loc, isPole, nan, yv);
    break;
  }
  case Un::Cos:
    y = b.create<math::CosOp>(loc, x);
    break;
  case Un::Tan:
    y = b.create<math::TanOp>(loc, x);
    break;
  case Un::Sinh:
    y = b.create<math::SinhOp>(loc, x);
    break;
  case Un::Cosh:
    y = b.create<math::CoshOp>(loc, x);
    break;
  case Un::Asin:
    y = b.create<math::AsinOp>(loc, x);
    break;
  case Un::Acos:
    y = b.create<math::AcosOp>(loc, x);
    break;
  case Un::Atan:
    y = b.create<math::AtanOp>(loc, x);
    break;
  case Un::Erfc:
    y = b.create<math::ErfcOp>(loc, x);
    break;
  case Un::Floor:
    y = b.create<math::FloorOp>(loc, x);
    break;
  case Un::Ceil:
    y = b.create<math::CeilOp>(loc, x);
    break;
  case Un::Round:
    // numpy round is round-half-to-even (banker's rounding)
    y = b.create<math::RoundEvenOp>(loc, x);
    break;
  case Un::Trunc:
    y = b.create<math::TruncOp>(loc, x);
    break;
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
                  .Case("sin", Un::Sin)
                  .Case("cos", Un::Cos)
                  .Case("tan", Un::Tan)
                  .Case("sinh", Un::Sinh)
                  .Case("cosh", Un::Cosh)
                  .Case("asin", Un::Asin)
                  .Case("acos", Un::Acos)
                  .Case("atan", Un::Atan)
                  .Case("erfc", Un::Erfc)
                  .Case("floor", Un::Floor)
                  .Case("ceil", Un::Ceil)
                  .Case("round", Un::Round)
                  .Case("trunc", Un::Trunc)
                  .Case("lgamma", Un::Lgamma)
                  .Case("digamma", Un::Digamma)
                  .Default(Un::Exp);
      static const llvm::StringSet<> kValid = {
          "exp",  "log",     "sqrt",  "rsqrt", "reciprocal", "abs",  "neg",
          "sign", "erf",     "tanh",  "sigmoid", "log1p",    "expm1",
          "softplus", "sin", "cos", "tan",   "sinh",  "cosh", "asin", "acos",
          "atan", "erfc",    "floor", "ceil",  "round", "trunc", "lgamma",
          "digamma"};
      if (!kValid.contains(kindStr)) {
        op->emitError("generate-rocm-unary-kernel: unknown kind '")
            << kindStr << "' (exp/log/sqrt/rsqrt/reciprocal/abs/neg/sign/erf/"
                          "tanh/sigmoid/log1p/expm1/softplus/cos/tan/sinh/cosh/"
                          "asin/acos/atan/erfc/floor/ceil/round/trunc)";
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
