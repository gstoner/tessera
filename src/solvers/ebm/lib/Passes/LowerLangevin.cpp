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
// "euclidean" (any rank), "sphere" ([rows, features]) or "bivector"
// ([rows, 2^n] Clifford coefficients); the two manifold integrators report a
// per-row status word (2026-09-16).
//
//===----------------------------------------------------------------------===//
#include "tessera/EBM/EBMDialect.h"
#include "tessera/EBM/EBMPasses.h"
#ifdef TESSERA_EBM_HAVE_CLIFFORD
#include "tessera/Clifford/CliffordDialect.h"
#endif

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
// Standard normals of `stateTy`'s shape from Philox-4x32-10 / Box-Muller on
// (flat index, key[1]) with key[0] as the Philox key (the declared policy).
static Value standardNormals(PatternRewriter &rewriter, Location loc, RankedTensorType stateTy, Value key) {
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
  return generic.getResult(0);
}

// Per-row sum over the feature axis of a [rows, features] tensor (sequential
// order over features: the declared reduction order, matched by the
// reference's f32 accumulate and by the row-program emitter's ordered fold).
static Value rowSum(PatternRewriter &rewriter, Location loc, Value lanes) {
  auto ty = cast<RankedTensorType>(lanes.getType());
  auto rowTy = RankedTensorType::get({ty.getDimSize(0)}, ty.getElementType());
  Value zero = rewriter.create<arith::ConstantOp>(loc, rewriter.getF32FloatAttr(0.0));
  Value init = rewriter.create<linalg::FillOp>(
      loc, ValueRange{zero}, ValueRange{rewriter.create<tensor::EmptyOp>(loc, rowTy.getShape(), rowTy.getElementType())}).getResult(0);
  auto reduce = rewriter.create<linalg::ReduceOp>(
      loc, ValueRange{lanes}, ValueRange{init}, ArrayRef<int64_t>{1},
      [](OpBuilder &b, Location l, ValueRange args) {
        b.create<linalg::YieldOp>(l, b.create<arith::AddFOp>(l, args[0], args[1]).getResult());
      });
  return reduce.getResult(0);
}

// Broadcast a [rows] value across the feature axis: [rows] -> [rows, features].
static Value rowBroadcast(PatternRewriter &rewriter, Location loc, Value row, RankedTensorType lanesTy) {
  auto rowTy = cast<RankedTensorType>(row.getType());
  auto expandedTy = RankedTensorType::get({rowTy.getDimSize(0), 1}, rowTy.getElementType());
  Value expanded = rewriter.create<tensor::ExpandShapeOp>(loc, expandedTy, row, ArrayRef<ReassociationIndices>{{0, 1}});
  Value init = rewriter.create<tensor::EmptyOp>(loc, lanesTy.getShape(), lanesTy.getElementType());
  MLIRContext *ctx = rewriter.getContext();
  AffineMap rowMap = AffineMap::get(2, 0, {getAffineDimExpr(0, ctx), getAffineConstantExpr(0, ctx)}, ctx);
  SmallVector<AffineMap> maps{rowMap, rewriter.getMultiDimIdentityMap(2)};
  SmallVector<utils::IteratorType> iterators(2, utils::IteratorType::parallel);
  auto generic = rewriter.create<linalg::GenericOp>(
      loc, TypeRange{lanesTy}, ValueRange{expanded}, ValueRange{init}, maps, iterators,
      [](OpBuilder &b, Location l, ValueRange args) { b.create<linalg::YieldOp>(l, args[0]); });
  return generic.getResult(0);
}

// Project a batched multivector onto `grades` with the Clifford dialect's own
// op — never a second grade projection here (Decision #31). The op is expanded
// downstream by `-tessera-clifford-expand-product-table`, which turns the
// compile-time keep-mask into scalar arithmetic; the EBM lowering only names
// the projection.
static Value gradeProject(PatternRewriter &rewriter, Location loc, Value value,
                          ArrayAttr grades, ArrayAttr algebra) {
#ifdef TESSERA_EBM_HAVE_CLIFFORD
  return rewriter.create<tessera::clifford::GradeProjectionOp>(
      loc, value.getType(), value, grades, algebra, rewriter.getStringAttr("f32"));
#else
  (void)rewriter; (void)loc; (void)grades; (void)algebra;
  return value;  // unreachable: the caller refuses "bivector" without Clifford
#endif
}

struct LowerLangevin : public RewritePattern {
  LowerLangevin(MLIRContext *ctx) : RewritePattern(kLangevinOp, 1, ctx) {}
  LogicalResult matchAndRewrite(Operation *op, PatternRewriter &rewriter) const override {
    auto fn = op->getAttrOfType<FlatSymbolRefAttr>("energy_fn");
    auto eta = op->getAttrOfType<FloatAttr>("eta");
    auto temperature = op->getAttrOfType<FloatAttr>("temperature");
    auto manifold = op->getAttrOfType<StringAttr>("manifold");
    // The temperature is either a constant attribute or a runtime scalar operand
    // (an annealing schedule). The op verifier has already established that
    // exactly one is present.
    auto stepOp = cast<::tessera::ebm::LangevinStepOp>(op);
    Value temperatureValue = stepOp.getTemperatureValue();
    if (!fn || !eta || !manifold || (!temperature && !temperatureValue)) return failure();
    const bool sphere = manifold.getValue() == "sphere";
    const bool bivector = manifold.getValue() == "bivector";
    if (manifold.getValue() != "euclidean" && !sphere && !bivector) {
      op->emitError("EBM lowering: manifold \"") << manifold.getValue()
          << "\" has no native integrator yet; \"euclidean\", \"sphere\" and \"bivector\" lower";
      return failure();
    }
#ifndef TESSERA_EBM_HAVE_CLIFFORD
    if (bivector) {
      op->emitError("EBM lowering: the bivector integrator needs the Clifford dialect's grade "
                    "projection; rebuild with -DTESSERA_BUILD_CLIFFORD_BACKEND=ON");
      return failure();
    }
#endif
    // `grade` and `algebra` are SEMANTIC keys for this integrator (Decision
    // #21a): a wrong grade or signature converges to a different distribution
    // rather than failing, so neither may be defaulted.
    ArrayAttr gradesAttr, algebraAttr;
    if (bivector) {
      auto grade = op->getAttrOfType<IntegerAttr>("grade");
      algebraAttr = op->getAttrOfType<ArrayAttr>("algebra");
      if (!grade || !algebraAttr) {
        op->emitError("EBM lowering: the bivector integrator requires `grade` and `algebra` "
                      "(the Clifford signature [p, q, r]); neither is defaulted");
        return failure();
      }
      int64_t p = 0, q = 0, r = 0, idx = 0;
      for (Attribute a : algebraAttr) {
        auto ai = dyn_cast<IntegerAttr>(a);
        if (!ai || ai.getInt() < 0) { op->emitError("EBM lowering: `algebra` must be [p, q, r]"); return failure(); }
        (idx == 0 ? p : idx == 1 ? q : r) = ai.getInt();
        ++idx;
      }
      if (idx != 3) { op->emitError("EBM lowering: `algebra` must be [p, q, r]"); return failure(); }
      const int64_t n = p + q + r;
      if (n < 1 || n > 12) { op->emitError("EBM lowering: the algebra must have 1..12 generators"); return failure(); }
      if (grade.getInt() < 0 || grade.getInt() > n) {
        op->emitError("EBM lowering: grade ") << grade.getInt() << " is out of range for a "
            << n << "-generator algebra";
        return failure();
      }
      gradesAttr = rewriter.getI64ArrayAttr({grade.getInt()});
    }
    if (eta.getValueAsDouble() <= 0.0 || (temperature && temperature.getValueAsDouble() < 0.0)) {
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
    RankedTensorType statusTy;
    if (bivector) {
      // State is [rows, 2^n] coefficients — the batched Clifford layout.
      int64_t n = 0;
      for (Attribute a : algebraAttr) n += cast<IntegerAttr>(a).getInt();
      const int64_t dim = int64_t{1} << n;
      if (stateTy.getRank() != 2 || stateTy.getDimSize(1) != dim) {
        op->emitError("EBM lowering: the bivector integrator takes a [rows, 2^n] state; the "
                      "algebra has ") << n << " generators, so the feature axis must be " << dim;
        return failure();
      }
      statusTy = RankedTensorType::get({stateTy.getDimSize(0)}, rewriter.getI32Type());
      if (op->getNumResults() != 3 || op->getResult(2).getType() != statusTy) {
        op->emitError("EBM lowering: the bivector integrator reports a per-row status word; "
                      "declare a third result of type ") << statusTy;
        return failure();
      }
    } else if (sphere) {
      if (stateTy.getRank() != 2) {
        op->emitError("EBM lowering: the sphere integrator takes a [rows, features] state (one unit vector per row)");
        return failure();
      }
      statusTy = RankedTensorType::get({stateTy.getDimSize(0)}, rewriter.getI32Type());
      if (op->getNumResults() != 3 || op->getResult(2).getType() != statusTy) {
        op->emitError("EBM lowering: the sphere integrator reports a per-row status word; declare a third result of type ")
            << statusTy;
        return failure();
      }
    } else if (op->getNumResults() != 2) {
      op->emitError("EBM lowering: the euclidean integrator has no status result");
      return failure();
    }
    const bool manifoldStatus = sphere || bivector;
    auto module = op->getParentOfType<ModuleOp>();
    auto backward = module.lookupSymbol<func::FuncOp>((fn.getValue() + "__bwd").str());
    if (!backward || backward.isExternal()) {
      op->emitError("EBM lowering: the compiler-derived gradient @") << fn.getValue()
          << "__bwd is absent; run tessera-autodiff-paired on an energy_fn marked "
             "tessera.autodiff = \"reverse\" before this pass";
      return failure();
    }
    // @E__bwd(state, captures..., cotangent) -> (dstate, dcaptures...)
    SmallVector<Value> captures(stepOp.getCaptures().begin(), stepOp.getCaptures().end());
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
    // sqrt(2 * eta * T). With the attribute this folds at compile time, and a
    // zero temperature emits no noise op at all; with a runtime schedule the
    // scale is computed per step and the noise is always emitted, since the pass
    // cannot know the schedule ever reaches zero. At T = 0 the two agree exactly:
    // the scale is +0.0 and the draws are finite.
    const double noiseScale =
        temperature ? std::sqrt(2.0 * eta.getValueAsDouble() * temperature.getValueAsDouble()) : 1.0;
    // A constant T = 0 is the one case where no noise op is emitted at all.
    const bool emitNoise = !temperature || noiseScale > 0.0;
    auto noiseScaleFor = [&](RankedTensorType ty) -> Value {
      if (temperature) return splat(rewriter, loc, ty, noiseScale);
      Value t = temperatureValue;
      if (!t.getType().isF32()) {
        if (t.getType().isF64())
          t = rewriter.create<arith::TruncFOp>(loc, rewriter.getF32Type(), t);
        else
          t = rewriter.create<arith::ExtFOp>(loc, rewriter.getF32Type(), t);
      }
      Value scaled = rewriter.create<arith::MulFOp>(
          loc, t, rewriter.create<arith::ConstantOp>(
                      loc, rewriter.getF32FloatAttr(float(2.0 * eta.getValueAsDouble()))));
      // Negative temperatures have no meaning; clamp to zero before the sqrt so a
      // bad schedule yields no noise rather than NaN state that propagates.
      Value clamped = rewriter.create<arith::MaximumFOp>(
          loc, scaled, rewriter.create<arith::ConstantOp>(loc, rewriter.getF32FloatAttr(0.0f)));
      Value root = rewriter.create<math::SqrtOp>(loc, clamped);
      // linalg.fill, not linalg.broadcast: the scale is a scalar, and fill is the
      // destination-passing form the JIT's bufferization already handles.
      Value empty = rewriter.create<tensor::EmptyOp>(loc, ty.getShape(), ty.getElementType());
      return rewriter.create<linalg::FillOp>(loc, ValueRange{root}, ValueRange{empty}).getResult(0);
    };
    auto bump = DenseElementsAttr::get(keyTy, ArrayRef<int64_t>{0, 1});
    auto bumpKey = [&]() -> Value {
      return rewriter.create<arith::AddIOp>(loc, key, rewriter.create<arith::ConstantOp>(loc, bump));
    };
    if (bivector) {
      // geo_sampling.bivector_langevin_step: both the gradient and the noise
      // are grade-projected, the Euclidean affine step applies, and a final
      // projection cleans up float leakage outside the subspace. The state
      // must already be grade-k on entry; that precondition is REPORTED per
      // row (status bit 0), never repaired (Decision #21a).
      auto rowTy = RankedTensorType::get({stateTy.getDimSize(0)}, stateTy.getElementType());
      Value kept = gradeProject(rewriter, loc, state, gradesAttr, algebraAttr);
      Value leak = rewriter.create<arith::SubFOp>(loc, state, kept);
      Value leak2 = rowSum(rewriter, loc, rewriter.create<arith::MulFOp>(loc, leak, leak));
      Value entryBad = rewriter.create<arith::CmpFOp>(loc, arith::CmpFPredicate::OGT, leak2,
                                                      splat(rewriter, loc, rowTy, 1.0e-12));
      Value gk = gradeProject(rewriter, loc, grad, gradesAttr, algebraAttr);
      Value next = rewriter.create<arith::SubFOp>(
          loc, state, rewriter.create<arith::MulFOp>(loc, gk, splat(rewriter, loc, stateTy, eta.getValueAsDouble())));
      if (emitNoise) {
        Value zk = gradeProject(rewriter, loc, standardNormals(rewriter, loc, stateTy, key),
                                gradesAttr, algebraAttr);
        next = rewriter.create<arith::AddFOp>(
            loc, next, rewriter.create<arith::MulFOp>(loc, zk, noiseScaleFor(stateTy)));
      }
      next = gradeProject(rewriter, loc, next, gradesAttr, algebraAttr);
      auto statusRow = [&](int64_t v) {
        return rewriter.create<arith::ConstantOp>(loc, DenseElementsAttr::get(statusTy, rewriter.getI32IntegerAttr(v))).getResult();
      };
      Value status = rewriter.create<arith::SelectOp>(loc, entryBad, statusRow(1), statusRow(0));
      rewriter.replaceOp(op, {next, bumpKey(), status});
      return success();
    }
    if (!sphere) {
      Value step = rewriter.create<arith::MulFOp>(loc, grad, splat(rewriter, loc, stateTy, eta.getValueAsDouble()));
      Value next = rewriter.create<arith::SubFOp>(loc, state, step);
      if (emitNoise) {
        Value noise = standardNormals(rewriter, loc, stateTy, key);
        next = rewriter.create<arith::AddFOp>(loc, next, rewriter.create<arith::MulFOp>(loc, noise, noiseScaleFor(stateTy)));
      }
      rewriter.replaceOp(op, {next, bumpKey()});
      return success();
    }
    Value noise;  // standard normals, drawn once per step when T > 0
    if (emitNoise) noise = standardNormals(rewriter, loc, stateTy, key);
    Value nextKey = bumpKey();
    // Sphere (geo_sampling.sphere_langevin_step, per row):
    //   g_t = g - <g, x> x ; xi_t = xi - <xi, x> x
    //   y   = x - eta g_t + sqrt(2 eta T) xi_t ; x' = y / |y|
    // Entry precondition |x|^2 in [1 - 2e-3, 1 + 2e-3] per row (the reference's
    // | |x| - 1 | <= 1e-3, stated on the squared norm) -> status bit 0; a
    // retraction underflow |y|^2 < 1e-12 keeps x and sets status bit 1. The
    // dot products and norms are sequential f32 row sums (rowSum).
    auto rowTy = RankedTensorType::get({stateTy.getDimSize(0)}, stateTy.getElementType());
    auto rowI1Ty = RankedTensorType::get({stateTy.getDimSize(0)}, rewriter.getI1Type());
    auto rowSplat = [&](double v) { return splat(rewriter, loc, rowTy, v); };
    auto project = [&](Value v) {
      Value dot = rowSum(rewriter, loc, rewriter.create<arith::MulFOp>(loc, v, state));
      return rewriter.create<arith::SubFOp>(loc, v, rewriter.create<arith::MulFOp>(loc, rowBroadcast(rewriter, loc, dot, stateTy), state)).getResult();
    };
    Value n0 = rowSum(rewriter, loc, rewriter.create<arith::MulFOp>(loc, state, state));
    Value dev = rewriter.create<math::AbsFOp>(loc, rewriter.create<arith::SubFOp>(loc, n0, rowSplat(1.0)));
    Value entryBad = rewriter.create<arith::CmpFOp>(loc, arith::CmpFPredicate::OGT, dev, rowSplat(2.0e-3));
    Value gt = project(grad);
    Value y = rewriter.create<arith::SubFOp>(loc, state, rewriter.create<arith::MulFOp>(loc, gt, splat(rewriter, loc, stateTy, eta.getValueAsDouble())));
    if (noise) {
      Value xt = project(noise);
      y = rewriter.create<arith::AddFOp>(loc, y, rewriter.create<arith::MulFOp>(loc, xt, noiseScaleFor(stateTy)));
    }
    Value n2 = rowSum(rewriter, loc, rewriter.create<arith::MulFOp>(loc, y, y));
    Value under = rewriter.create<arith::CmpFOp>(loc, arith::CmpFPredicate::OLT, n2, rowSplat(1.0e-12));
    Value safeN2 = rewriter.create<arith::SelectOp>(loc, under, rowSplat(1.0), n2);
    Value norm = rewriter.create<math::SqrtOp>(loc, safeN2);
    Value normalized = rewriter.create<arith::DivFOp>(loc, y, rowBroadcast(rewriter, loc, norm, stateTy));
    // Blend by a {0,1} row weight instead of broadcasting an i1 row.
    Value keep = rewriter.create<arith::SelectOp>(loc, under, rowSplat(1.0), rowSplat(0.0));
    Value keepB = rowBroadcast(rewriter, loc, keep, stateTy);
    Value one = splat(rewriter, loc, stateTy, 1.0);
    Value next = rewriter.create<arith::AddFOp>(
        loc, rewriter.create<arith::MulFOp>(loc, keepB, state),
        rewriter.create<arith::MulFOp>(loc, rewriter.create<arith::SubFOp>(loc, one, keepB), normalized));
    auto i32Row = [&](int64_t v) {
      return rewriter.create<arith::ConstantOp>(loc, DenseElementsAttr::get(statusTy, rewriter.getI32IntegerAttr(v))).getResult();
    };
    Value status = rewriter.create<arith::OrIOp>(
        loc, rewriter.create<arith::SelectOp>(loc, entryBad, i32Row(1), i32Row(0)),
        rewriter.create<arith::SelectOp>(loc, under, i32Row(2), i32Row(0)));
    (void)rowI1Ty;
    rewriter.replaceOp(op, {next, nextKey, status});
    return success();
  }
};

struct EBMLowerLangevinPass
    : public PassWrapper<EBMLowerLangevinPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(EBMLowerLangevinPass)
  StringRef getArgument() const final { return "tessera-ebm-lower-langevin"; }
  StringRef getDescription() const final {
    return "Lower tessera_ebm.energy / inner_step / langevin_step (euclidean, sphere, bivector) to "
           "arith/linalg over the compiler-derived gradient (@E__bwd) with "
           "on-device Philox noise.";
  }
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<arith::ArithDialect, func::FuncDialect, linalg::LinalgDialect,
                    math::MathDialect, tensor::TensorDialect>();
#ifdef TESSERA_EBM_HAVE_CLIFFORD
    // The bivector integrator emits the Clifford dialect's own grade op.
    registry.insert<tessera::clifford::CliffordDialect>();
#endif
  }
  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    patterns.add<LowerEnergy, LowerInnerStep, LowerLangevin>(&getContext());
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns))))
      return signalPassFailure();
    // A pattern that refuses an op emits a diagnostic and declines to rewrite;
    // the greedy driver still reports success, so the pass used to EXIT 0 with
    // an unlowered `tessera_ebm.*` op and an error already printed. Downstream
    // refuses that op, but the exit status must say so here (2026-09-16).
    bool survived = false;
    getOperation().walk([&](Operation *op) {
      if (op->getName().getDialectNamespace() == "tessera_ebm") {
        op->emitError("EBM lowering: this op was not lowered; see the diagnostic above");
        survived = true;
      }
    });
    if (survived) signalPassFailure();
  }
};

}  // namespace

std::unique_ptr<mlir::Pass> createEBMLowerLangevinPass() {
  return std::make_unique<EBMLowerLangevinPass>();
}

}  // namespace tessera
