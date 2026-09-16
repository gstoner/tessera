//===- LowerLangevin.cpp -----------------------------------------*- C++ -*-===//
//
// EBMLowerLangevinPass (2026-09-16, W4-PRODUCT-1 / AD-SOLVER-IFT-1): the first
// *lowering* in the EBM dialect. Every pass before it annotated; this one
// turns the energy program into arithmetic the shared backbone executes:
//
//   %e = tessera_ebm.energy(%x, %y) {energy_fn = @E}
//        -> func.call @E(%y, %x)                       (E is E(state, captures...))
//   %y1 = tessera_ebm.inner_step(%y, %g) {eta}
//        -> %y - eta * %g
//   %y1, %k1 = tessera_ebm.langevin_step(%y, %key, captures...)
//              {energy_fn = @E, eta, temperature, manifold = "euclidean"}
//        -> %g = call @E__bwd(%y, captures..., ones)#0   (the compiler's own
//                 reverse-mode gradient of E w.r.t. the state; the paired
//                 autodiff pass must have produced @E__bwd -- this pass never
//                 differentiates by hand and refuses when the symbol is absent)
//           %z = Philox-4x32-10 / Box-Muller standard normals, one per
//                 element, generated inside a linalg.generic on the device
//                 side of whatever backend consumes the loop (no host RNG)
//           %y1 = %y - eta * %g + sqrt(2 * eta * T) * %z
//           %k1 = %key + [0, 1]
//
// Declared RNG policy (mirrored bit-for-bit by
// python/tessera/ebm/native_langevin.py::reference_langevin_loop):
//   philox key     = (lo32(key[0]), hi32(key[0]))
//   philox counter = (flat element index, lo32(key[1]), hi32(key[1]), 0)
//   z = sqrt(-2 ln u0) * cos(2 pi u1),  u = (word + 0.5) * 2^-32 on words 0, 1
//   next key       = (key[0], key[1] + 1)          -- one stream per step
// Temperature 0 emits no noise at all (pure gradient descent), so a T = 0 loop
// is exactly y - eta * g per step.
//
// Envelope: static ranked f32 state, key `tensor<2xi64>`, manifold
// "euclidean" only ("sphere" / "bivector" fail closed with a diagnostic).
//
//===----------------------------------------------------------------------===//
#include "tessera/EBM/EBMPasses.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include <cmath>

using namespace mlir;

namespace tessera {
namespace {

constexpr StringRef kEnergyOp = "tessera_ebm.energy";
constexpr StringRef kInnerStepOp = "tessera_ebm.inner_step";
constexpr StringRef kLangevinOp = "tessera_ebm.langevin_step";

static RankedTensorType staticF32(Value v) {
  auto ty = dyn_cast<RankedTensorType>(v.getType());
  if (!ty || !ty.hasStaticShape() || !ty.getElementType().isF32()) return nullptr;
  return ty;
}

static Value splat(OpBuilder &b, Location loc, RankedTensorType ty, double value) {
  auto attr = DenseElementsAttr::get(ty, b.getFloatAttr(ty.getElementType(), value));
  return b.create<arith::ConstantOp>(loc, attr);
}

// energy(x, y) {energy_fn = @E} -> call @E(y, x)
struct LowerEnergy : public RewritePattern {
  LowerEnergy(MLIRContext *ctx) : RewritePattern(kEnergyOp, 1, ctx) {}
  LogicalResult matchAndRewrite(Operation *op, PatternRewriter &rewriter) const override {
    auto fn = op->getAttrOfType<FlatSymbolRefAttr>("energy_fn");
    if (!fn) return failure();
    auto module = op->getParentOfType<ModuleOp>();
    auto callee = module.lookupSymbol<func::FuncOp>(fn.getValue());
    if (!callee || callee.isExternal()) {
      op->emitError("EBM lowering: energy_fn @") << fn.getValue() << " must be defined in this module";
      return failure();
    }
    if (callee.getNumArguments() != 2 || callee.getNumResults() != 1 ||
        callee.getArgument(0).getType() != op->getOperand(1).getType() ||
        callee.getArgument(1).getType() != op->getOperand(0).getType() ||
        callee.getResultTypes()[0] != op->getResult(0).getType()) {
      op->emitError("EBM lowering: energy_fn must be E(state, context) -> energies matching the op types");
      return failure();
    }
    rewriter.replaceOpWithNewOp<func::CallOp>(op, callee, ValueRange{op->getOperand(1), op->getOperand(0)});
    return success();
  }
};

// inner_step(y, g) {eta} -> y - eta * g
struct LowerInnerStep : public RewritePattern {
  LowerInnerStep(MLIRContext *ctx) : RewritePattern(kInnerStepOp, 1, ctx) {}
  LogicalResult matchAndRewrite(Operation *op, PatternRewriter &rewriter) const override {
    auto eta = op->getAttrOfType<FloatAttr>("eta");
    auto ty = staticF32(op->getOperand(0));
    if (!eta || !ty || op->getOperand(1).getType() != ty) {
      op->emitError("EBM lowering: inner_step requires static f32 state/grad of one type and eta");
      return failure();
    }
    Location loc = op->getLoc();
    Value scaled = rewriter.create<arith::MulFOp>(loc, op->getOperand(1), splat(rewriter, loc, ty, eta.getValueAsDouble()));
    rewriter.replaceOpWithNewOp<arith::SubFOp>(op, op->getOperand(0), scaled);
    return success();
  }
};

// One Philox-4x32-10 evaluation on scalar i32 words; returns the four output words.
static SmallVector<Value, 4> philox4x32(OpBuilder &b, Location loc, ArrayRef<Value> ctr, ArrayRef<Value> key) {
  Type i32 = b.getI32Type();
  auto c = [&](uint32_t v) { return b.create<arith::ConstantOp>(loc, b.getIntegerAttr(i32, (int64_t)v)).getResult(); };
  Value m0 = c(0xD2511F53u), m1 = c(0xCD9E8D57u), w0 = c(0x9E3779B9u), w1 = c(0xBB67AE85u);
  Value c0 = ctr[0], c1 = ctr[1], c2 = ctr[2], c3 = ctr[3], k0 = key[0], k1 = key[1];
  for (int round = 0; round < 10; ++round) {
    auto p0 = b.create<arith::MulUIExtendedOp>(loc, m0, c0);
    auto p1 = b.create<arith::MulUIExtendedOp>(loc, m1, c2);
    Value n0 = b.create<arith::XOrIOp>(loc, b.create<arith::XOrIOp>(loc, p1.getHigh(), c1), k0);
    Value n1 = p1.getLow();
    Value n2 = b.create<arith::XOrIOp>(loc, b.create<arith::XOrIOp>(loc, p0.getHigh(), c3), k1);
    Value n3 = p0.getLow();
    c0 = n0; c1 = n1; c2 = n2; c3 = n3;
    k0 = b.create<arith::AddIOp>(loc, k0, w0);
    k1 = b.create<arith::AddIOp>(loc, k1, w1);
  }
  return {c0, c1, c2, c3};
}

// word (u32) -> (word + 0.5) * 2^-32 in f64, as the reference does.
static Value uniform(OpBuilder &b, Location loc, Value word) {
  Type f64 = b.getF64Type();
  Value asF = b.create<arith::UIToFPOp>(loc, f64, word);
  Value half = b.create<arith::ConstantOp>(loc, b.getF64FloatAttr(0.5));
  Value scale = b.create<arith::ConstantOp>(loc, b.getF64FloatAttr(std::ldexp(1.0, -32)));
  return b.create<arith::MulFOp>(loc, b.create<arith::AddFOp>(loc, asF, half), scale);
}

// langevin_step(y, key, captures...) -> (y - eta*grad + scale*noise, key + [0, 1])
struct LowerLangevin : public RewritePattern {
  LowerLangevin(MLIRContext *ctx) : RewritePattern(kLangevinOp, 1, ctx) {}
  LogicalResult matchAndRewrite(Operation *op, PatternRewriter &rewriter) const override {
    auto fn = op->getAttrOfType<FlatSymbolRefAttr>("energy_fn");
    auto eta = op->getAttrOfType<FloatAttr>("eta");
    auto temperature = op->getAttrOfType<FloatAttr>("temperature");
    auto manifold = op->getAttrOfType<StringAttr>("manifold");
    if (!fn || !eta || !temperature || !manifold) return failure();
    if (manifold.getValue() != "euclidean") {
      op->emitError("EBM lowering: manifold \"") << manifold.getValue()
          << "\" has no native integrator yet; only \"euclidean\" lowers";
      return failure();
    }
    if (eta.getValueAsDouble() <= 0.0 || temperature.getValueAsDouble() < 0.0) {
      op->emitError("EBM lowering: langevin_step requires eta > 0 and temperature >= 0");
      return failure();
    }
    Value state = op->getOperand(0), key = op->getOperand(1);
    auto stateTy = staticF32(state);
    auto keyTy = dyn_cast<RankedTensorType>(key.getType());
    if (!stateTy || !keyTy || keyTy.getShape() != ArrayRef<int64_t>{2} || !keyTy.getElementType().isInteger(64)) {
      op->emitError("EBM lowering: langevin_step requires a static f32 state and a tensor<2xi64> key");
      return failure();
    }
    if (op->getResult(0).getType() != stateTy || op->getResult(1).getType() != keyTy) {
      op->emitError("EBM lowering: langevin_step results must match its state and key types");
      return failure();
    }
    auto module = op->getParentOfType<ModuleOp>();
    auto backward = module.lookupSymbol<func::FuncOp>((fn.getValue() + "__bwd").str());
    if (!backward || backward.isExternal()) {
      op->emitError("EBM lowering: the compiler-derived gradient @") << fn.getValue()
          << "__bwd is absent; run tessera-autodiff-paired on an energy_fn marked "
             "tessera.autodiff = \"reverse\" before this pass";
      return failure();
    }
    // @E__bwd(state, captures..., cotangent) -> (dstate, dcaptures...)
    SmallVector<Value> captures(op->getOperands().begin() + 2, op->getOperands().end());
    if (backward.getNumArguments() != captures.size() + 2 || backward.getNumResults() < 1 ||
        backward.getArgument(0).getType() != stateTy || backward.getResultTypes()[0] != stateTy) {
      op->emitError("EBM lowering: @") << fn.getValue() << "__bwd must be (state, captures..., cotangent) -> (dstate, ...)";
      return failure();
    }
    for (auto [i, capture] : llvm::enumerate(captures))
      if (backward.getArgument(i + 1).getType() != capture.getType()) {
        op->emitError("EBM lowering: capture ") << i << " type disagrees with @" << fn.getValue() << "__bwd";
        return failure();
      }
    auto cotTy = dyn_cast<RankedTensorType>(backward.getArgument(captures.size() + 1).getType());
    if (!cotTy || !cotTy.hasStaticShape() || !cotTy.getElementType().isF32()) {
      op->emitError("EBM lowering: the energy cotangent must be a static f32 tensor");
      return failure();
    }
    Location loc = op->getLoc();
    SmallVector<Value> args(captures.size() + 2);
    args[0] = state;
    for (auto [i, capture] : llvm::enumerate(captures)) args[i + 1] = capture;
    args[captures.size() + 1] = splat(rewriter, loc, cotTy, 1.0);  // dE/dE = 1 per energy
    Value grad = rewriter.create<func::CallOp>(loc, backward, args).getResult(0);
    Value step = rewriter.create<arith::MulFOp>(loc, grad, splat(rewriter, loc, stateTy, eta.getValueAsDouble()));
    Value next = rewriter.create<arith::SubFOp>(loc, state, step);
    const double noiseScale = std::sqrt(2.0 * eta.getValueAsDouble() * temperature.getValueAsDouble());
    if (noiseScale > 0.0) {
      // Standard normals per element from Philox on (flat index, key[1]), key[0].
      Type i32 = rewriter.getI32Type(), i64 = rewriter.getI64Type();
      Value c0i = rewriter.create<arith::ConstantIndexOp>(loc, 0);
      Value c1i = rewriter.create<arith::ConstantIndexOp>(loc, 1);
      Value key0 = rewriter.create<tensor::ExtractOp>(loc, key, ValueRange{c0i});
      Value key1 = rewriter.create<tensor::ExtractOp>(loc, key, ValueRange{c1i});
      Value c32 = rewriter.create<arith::ConstantOp>(loc, rewriter.getIntegerAttr(i64, 32));
      Value k0 = rewriter.create<arith::TruncIOp>(loc, i32, key0);
      Value k1 = rewriter.create<arith::TruncIOp>(loc, i32, rewriter.create<arith::ShRUIOp>(loc, key0, c32));
      Value s0 = rewriter.create<arith::TruncIOp>(loc, i32, key1);
      Value s1 = rewriter.create<arith::TruncIOp>(loc, i32, rewriter.create<arith::ShRUIOp>(loc, key1, c32));
      Value zero32 = rewriter.create<arith::ConstantOp>(loc, rewriter.getIntegerAttr(i32, 0));
      Value init = rewriter.create<tensor::EmptyOp>(loc, stateTy.getShape(), stateTy.getElementType());
      const int64_t rank = stateTy.getRank();
      SmallVector<AffineMap> maps{rewriter.getMultiDimIdentityMap(rank)};
      SmallVector<utils::IteratorType> iterators(rank, utils::IteratorType::parallel);
      auto generic = rewriter.create<linalg::GenericOp>(
          loc, TypeRange{stateTy}, ValueRange{}, ValueRange{init}, maps, iterators,
          [&](OpBuilder &b, Location l, ValueRange) {
            // flat index = sum(idx[d] * stride[d]) in row-major order.
            Value flat = b.create<arith::ConstantIndexOp>(l, 0);
            int64_t stride = 1;
            for (int64_t d = rank - 1; d >= 0; --d) {
              Value idx = b.create<linalg::IndexOp>(l, d);
              Value s = b.create<arith::ConstantIndexOp>(l, stride);
              flat = b.create<arith::AddIOp>(l, flat, b.create<arith::MulIOp>(l, idx, s));
              stride *= stateTy.getShape()[d];
            }
            Value flat32 = b.create<arith::IndexCastOp>(l, i32, flat);
            auto words = philox4x32(b, l, {flat32, s0, s1, zero32}, {k0, k1});
            Value u0 = uniform(b, l, words[0]), u1 = uniform(b, l, words[1]);
            Value minusTwo = b.create<arith::ConstantOp>(l, b.getF64FloatAttr(-2.0));
            Value twoPi = b.create<arith::ConstantOp>(l, b.getF64FloatAttr(2.0 * M_PI));
            Value r = b.create<math::SqrtOp>(l, b.create<arith::MulFOp>(l, minusTwo, b.create<math::LogOp>(l, u0)));
            Value z = b.create<arith::MulFOp>(l, r, b.create<math::CosOp>(l, b.create<arith::MulFOp>(l, twoPi, u1)));
            b.create<linalg::YieldOp>(l, b.create<arith::TruncFOp>(l, b.getF32Type(), z).getResult());
          });
      Value noise = rewriter.create<arith::MulFOp>(loc, generic.getResult(0), splat(rewriter, loc, stateTy, noiseScale));
      next = rewriter.create<arith::AddFOp>(loc, next, noise);
    }
    auto bump = DenseElementsAttr::get(keyTy, ArrayRef<int64_t>{0, 1});
    Value nextKey = rewriter.create<arith::AddIOp>(loc, key, rewriter.create<arith::ConstantOp>(loc, bump));
    rewriter.replaceOp(op, {next, nextKey});
    return success();
  }
};

struct EBMLowerLangevinPass
    : public PassWrapper<EBMLowerLangevinPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(EBMLowerLangevinPass)
  StringRef getArgument() const final { return "tessera-ebm-lower-langevin"; }
  StringRef getDescription() const final {
    return "Lower tessera_ebm.energy / inner_step / langevin_step (euclidean) to "
           "arith/linalg over the compiler-derived gradient (@E__bwd) with "
           "on-device Philox noise.";
  }
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<arith::ArithDialect, func::FuncDialect, linalg::LinalgDialect,
                    math::MathDialect, tensor::TensorDialect>();
  }
  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    patterns.add<LowerEnergy, LowerInnerStep, LowerLangevin>(&getContext());
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns))))
      signalPassFailure();
  }
};

}  // namespace

std::unique_ptr<mlir::Pass> createEBMLowerLangevinPass() {
  return std::make_unique<EBMLowerLangevinPass>();
}

}  // namespace tessera
