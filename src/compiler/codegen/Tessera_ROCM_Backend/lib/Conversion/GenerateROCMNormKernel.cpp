//===- GenerateROCMNormKernel.cpp - compiler-generated row normalization --===//
//
// Expands a `tessera_rocm.norm` directive into a row-reduction gpu kernel for
// the **unweighted** row normalizations over the last axis (the sibling of the
// softmax reduction kernel):
//
//   kind = "rmsnorm"    : O[m,:] = X[m,:] / sqrt(mean_k X[m,k]² + eps)
//   kind = "layer_norm" : O[m,:] = (X[m,:] − μ) / sqrt(var + eps),
//                         μ = mean_k X,  var = mean_k (X − μ)²
//
//   One workgroup per row (blockIdx.x = m), blockDim = 256, CUB/rocPRIM-style
//   warp-shuffle reduction (gpu.shuffle xor within a 32-lane subgroup → 8 LDS
//   partials → combine; no 256-wide LDS tree).
//   rmsnorm is one reduction (Σx²). layer_norm is TWO reductions — Σx for the
//   mean, then Σ(x−μ)² for the variance (the stable squared-deviation form,
//   NOT E[x²]−E[x]², which cancels for large-offset/small-variance rows). Then
//   a write pass applies the per-row normalize plus optional channel-affine
//   gamma/beta vectors. The affine-presence flags are uniform runtime values,
//   so one shape-independent HSACO serves unary and affine Graph contracts.
//   Reductions run in f32 regardless of storage dtype; `sqrt` lowers through
//   convert-math-to-rocdl. eps is a trailing f32 runtime arg; M/K are runtime
//   index args. Validated vs the numpy reference (`_apple_gpu_rowop_numpy`).
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
static constexpr int64_t SG = 32;           // shuffle subgroup width
static constexpr int64_t NGROUPS = BD / SG; // per-subgroup partials (= 8)

void emitNormBody(OpBuilder &b, Location loc, gpu::GPUFuncOp f, Type storeTy,
                  bool isLayerNorm, StringRef epilogue) {
  MLIRContext *ctx = b.getContext();
  Type f32 = b.getF32Type();
  bool isF32 = storeTy.isF32();
  auto slt = arith::CmpIPredicate::slt;

  auto ws = gpu::AddressSpaceAttr::get(ctx, gpu::AddressSpace::Workgroup);
  // LDS holds only the per-subgroup partials (NGROUPS = 8), reused per pass.
  auto ldsT = MemRefType::get({NGROUPS}, f32, MemRefLayoutAttrInterface(), ws);
  Value red = f.addWorkgroupAttribution(ldsT, loc);

  b.setInsertionPointToStart(&f.getBody().front());
  Value X = f.getArgument(0), gamma = f.getArgument(1);
  Value beta = f.getArgument(2), O = f.getArgument(3);
  Value M = f.getArgument(4), K = f.getArgument(5), eps = f.getArgument(6);
  Value hasGamma = f.getArgument(7), hasBeta = f.getArgument(8);

  auto ci = [&](int64_t v) { return b.create<arith::ConstantIndexOp>(loc, v); };
  Value c0 = ci(0), cBD = ci(BD);
  Value zerof = b.create<arith::ConstantOp>(loc, f32, b.getF32FloatAttr(0.0));

  Value m = b.create<gpu::BlockIdOp>(loc, gpu::Dimension::x);
  Value tid = b.create<gpu::ThreadIdOp>(loc, gpu::Dimension::x);
  Type i32 = b.getI32Type();
  Value wSG = b.create<arith::ConstantIntOp>(loc, i32, SG);
  Value cSG = ci(SG);
  Value group = b.create<arith::DivUIOp>(loc, tid, cSG);     // subgroup id
  Value laneInSg = b.create<arith::RemUIOp>(loc, tid, cSG);   // lane within it
  Value isLeader =
      b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq, laneInSg, c0);
  Value rowInb = b.create<arith::CmpIOp>(loc, slt, m, M);
  auto rowIf = b.create<scf::IfOp>(loc, rowInb, /*withElse=*/false);
  b.setInsertionPointToStart(rowIf.thenBlock());
  Value base = b.create<arith::MulIOp>(loc, m, K);
  // Kf = (f32) K  (for the means).
  Value Kf = b.create<arith::SIToFPOp>(
      loc, f32, b.create<arith::IndexCastOp>(loc, b.getI64Type(), K));

  auto loadF32 = [&](Value idx) -> Value {
    Value v = b.create<memref::LoadOp>(loc, X, ValueRange{idx});
    return isF32 ? v : b.create<arith::ExtFOp>(loc, f32, v);
  };
  auto storeFromF32 = [&](Value val, Value idx) {
    Value sv = isF32 ? val : b.create<arith::TruncFOp>(loc, storeTy, val);
    b.create<memref::StoreOp>(loc, sv, O, ValueRange{idx});
  };
  auto loadAffineF32 = [&](Value buffer, Value channel) -> Value {
    Value v = b.create<memref::LoadOp>(loc, buffer, ValueRange{channel});
    return isF32 ? v : b.create<arith::ExtFOp>(loc, f32, v);
  };
  // One full row reduction (sum), CUB/rocPRIM warp-shuffle style: accumulate a
  // per-element f32 value over the strided cols in-register, butterfly-reduce
  // within a 32-lane subgroup via `gpu.shuffle xor` (no LDS), stage the 8
  // per-subgroup partials to `red`, then every thread sums the partials to get
  // the broadcast row total. `red` is reused across passes, so barriers bracket
  // its read. (FP add reorders → matches numpy within tolerance.)
  auto reduceRow =
      [&](function_ref<Value(Value /*idx*/)> localOf) -> Value {
    auto lp = b.create<scf::ForOp>(loc, tid, K, cBD, ValueRange{zerof});
    {
      OpBuilder::InsertionGuard g(b);
      b.setInsertionPointToStart(lp.getBody());
      Value c = lp.getInductionVar();
      Value acc = lp.getRegionIterArgs()[0];
      Value v = localOf(b.create<arith::AddIOp>(loc, base, c));
      b.create<scf::YieldOp>(loc,
                             ValueRange{b.create<arith::AddFOp>(loc, acc, v)});
    }
    Value acc = lp.getResult(0);
    for (int64_t off = SG / 2; off > 0; off >>= 1) {
      Value offC = b.create<arith::ConstantIntOp>(loc, i32, off);
      auto sh = b.create<gpu::ShuffleOp>(loc, acc, offC, wSG,
                                         gpu::ShuffleMode::XOR);
      acc = b.create<arith::AddFOp>(loc, acc, sh.getShuffleResult());
    }
    auto ldIf = b.create<scf::IfOp>(loc, isLeader, /*withElse=*/false);
    {
      OpBuilder::InsertionGuard g(b);
      b.setInsertionPointToStart(ldIf.thenBlock());
      b.create<memref::StoreOp>(loc, acc, red, ValueRange{group});
    }
    b.create<gpu::BarrierOp>(loc);
    Value total = b.create<memref::LoadOp>(loc, red, ValueRange{c0});
    for (int64_t gi = 1; gi < NGROUPS; ++gi)
      total = b.create<arith::AddFOp>(
          loc, total, b.create<memref::LoadOp>(loc, red, ValueRange{ci(gi)}));
    b.create<gpu::BarrierOp>(loc);  // all read red before the next pass reuses it
    return total;
  };

  // denom = sqrt(stat + eps), matching numpy's x / √(...). For layer_norm the
  // variance is computed as a SECOND reduction of the squared deviations
  // (mean((x−μ)²)) rather than E[x²]−E[x]² — the latter cancels catastrophically
  // for rows with a large common offset and small variance (PR#123 review).
  Value mean, denom;
  if (isLayerNorm) {
    mean = b.create<arith::DivFOp>(loc, reduceRow(loadF32), Kf);
    Value vsum = reduceRow([&](Value idx) {
      Value d = b.create<arith::SubFOp>(loc, loadF32(idx), mean);
      return b.create<arith::MulFOp>(loc, d, d).getResult();
    });
    Value var = b.create<arith::DivFOp>(loc, vsum, Kf);
    denom = b.create<math::SqrtOp>(loc, b.create<arith::AddFOp>(loc, var, eps));
  } else {  // rmsnorm: denom = sqrt(mean(x²) + eps)
    Value sumsq = reduceRow([&](Value idx) {
      Value v = loadF32(idx);
      return b.create<arith::MulFOp>(loc, v, v).getResult();
    });
    Value ms = b.create<arith::DivFOp>(loc, sumsq, Kf);
    denom = b.create<math::SqrtOp>(loc, b.create<arith::AddFOp>(loc, ms, eps));
  }

  // write pass — O = ((x [− μ]) / denom) * gamma + beta. Missing affine
  // vectors are represented by uniform flags and never dereferenced.
  {
    auto lp = b.create<scf::ForOp>(loc, tid, K, cBD);
    OpBuilder::InsertionGuard g(b);
    b.setInsertionPointToStart(lp.getBody());
    Value c = lp.getInductionVar();
    Value idx = b.create<arith::AddIOp>(loc, base, c);
    Value v = loadF32(idx);
    if (isLayerNorm)
      v = b.create<arith::SubFOp>(loc, v, mean);
    v = b.create<arith::DivFOp>(loc, v, denom);
    auto gammaIf = b.create<scf::IfOp>(loc, TypeRange{f32}, hasGamma,
                                       /*withElse=*/true);
    {
      OpBuilder::InsertionGuard affineGuard(b);
      b.setInsertionPointToStart(gammaIf.thenBlock());
      Value scaled =
          b.create<arith::MulFOp>(loc, v, loadAffineF32(gamma, c));
      b.create<scf::YieldOp>(loc, ValueRange{scaled});
      b.setInsertionPointToStart(gammaIf.elseBlock());
      b.create<scf::YieldOp>(loc, ValueRange{v});
    }
    v = gammaIf.getResult(0);
    auto betaIf = b.create<scf::IfOp>(loc, TypeRange{f32}, hasBeta,
                                      /*withElse=*/true);
    {
      OpBuilder::InsertionGuard affineGuard(b);
      b.setInsertionPointToStart(betaIf.thenBlock());
      Value shifted =
          b.create<arith::AddFOp>(loc, v, loadAffineF32(beta, c));
      b.create<scf::YieldOp>(loc, ValueRange{shifted});
      b.setInsertionPointToStart(betaIf.elseBlock());
      b.create<scf::YieldOp>(loc, ValueRange{v});
    }
    v = betaIf.getResult(0);
    if (epilogue == "relu") {
      Value zero =
          b.create<arith::ConstantOp>(loc, f32, b.getF32FloatAttr(0.0));
      Value positive = b.create<arith::CmpFOp>(
          loc, arith::CmpFPredicate::OGT, v, zero);
      v = b.create<arith::SelectOp>(loc, positive, v, zero);
    } else if (epilogue == "silu") {
      Value one =
          b.create<arith::ConstantOp>(loc, f32, b.getF32FloatAttr(1.0));
      Value neg = b.create<arith::NegFOp>(loc, v);
      Value denom =
          b.create<arith::AddFOp>(loc, one, b.create<math::ExpOp>(loc, neg));
      v = b.create<arith::DivFOp>(loc, v, denom);
    }
    storeFromF32(v, idx);
  }

  b.setInsertionPointToEnd(&f.getBody().front());
  b.create<gpu::ReturnOp>(loc);
}

void emitNormBackwardBody(OpBuilder &b, Location loc, gpu::GPUFuncOp f,
                          Type storeTy, bool isLayerNorm) {
  MLIRContext *ctx = b.getContext();
  Type f32 = b.getF32Type();
  bool isF32 = storeTy.isF32();
  auto ws = gpu::AddressSpaceAttr::get(ctx, gpu::AddressSpace::Workgroup);
  auto ldsT = MemRefType::get({NGROUPS}, f32, MemRefLayoutAttrInterface(), ws);
  Value red = f.addWorkgroupAttribution(ldsT, loc);

  b.setInsertionPointToStart(&f.getBody().front());
  Value X = f.getArgument(0), gamma = f.getArgument(1);
  Value dY = f.getArgument(2), dX = f.getArgument(3);
  // dGamma contributions are written row-major into private slots. A second
  // kernel folds those slots—and dY directly for dBeta—in ascending row order,
  // so no cross-workgroup atomic ordering can affect either affine gradient.
  Value dGammaPartials = f.getArgument(4);
  Value M = f.getArgument(5), K = f.getArgument(6), eps = f.getArgument(7);
  Value hasGamma = f.getArgument(8);

  auto ci = [&](int64_t v) { return b.create<arith::ConstantIndexOp>(loc, v); };
  Value c0 = ci(0), cBD = ci(BD);
  Value zerof = b.create<arith::ConstantOp>(loc, f32, b.getF32FloatAttr(0.0));
  Value m = b.create<gpu::BlockIdOp>(loc, gpu::Dimension::x);
  Value tid = b.create<gpu::ThreadIdOp>(loc, gpu::Dimension::x);
  Type i32 = b.getI32Type();
  Value wSG = b.create<arith::ConstantIntOp>(loc, i32, SG);
  Value cSG = ci(SG);
  Value group = b.create<arith::DivUIOp>(loc, tid, cSG);
  Value laneInSg = b.create<arith::RemUIOp>(loc, tid, cSG);
  Value isLeader =
      b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq, laneInSg, c0);
  Value rowInb =
      b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::slt, m, M);
  auto rowIf = b.create<scf::IfOp>(loc, rowInb, /*withElse=*/false);
  b.setInsertionPointToStart(rowIf.thenBlock());
  Value base = b.create<arith::MulIOp>(loc, m, K);
  Value Kf = b.create<arith::SIToFPOp>(
      loc, f32, b.create<arith::IndexCastOp>(loc, b.getI64Type(), K));

  auto loadF32 = [&](Value buffer, Value idx) -> Value {
    Value v = b.create<memref::LoadOp>(loc, buffer, ValueRange{idx});
    return isF32 ? v : b.create<arith::ExtFOp>(loc, f32, v);
  };
  auto storeDX = [&](Value value, Value idx) {
    Value stored = isF32 ? value : b.create<arith::TruncFOp>(loc, storeTy, value);
    b.create<memref::StoreOp>(loc, stored, dX, ValueRange{idx});
  };
  auto loadGamma = [&](Value channel) -> Value {
    return loadF32(gamma, channel);
  };
  auto reduceRow = [&](function_ref<Value(Value)> localOf) -> Value {
    auto lp = b.create<scf::ForOp>(loc, tid, K, cBD, ValueRange{zerof});
    {
      OpBuilder::InsertionGuard guard(b);
      b.setInsertionPointToStart(lp.getBody());
      Value idx = b.create<arith::AddIOp>(loc, base, lp.getInductionVar());
      Value value = localOf(idx);
      Value accum = b.create<arith::AddFOp>(
          loc, lp.getRegionIterArgs()[0], value);
      b.create<scf::YieldOp>(loc, ValueRange{accum});
    }
    Value accum = lp.getResult(0);
    for (int64_t off = SG / 2; off > 0; off >>= 1) {
      Value offset = b.create<arith::ConstantIntOp>(loc, i32, off);
      auto shuffled = b.create<gpu::ShuffleOp>(
          loc, accum, offset, wSG, gpu::ShuffleMode::XOR);
      accum = b.create<arith::AddFOp>(loc, accum,
                                      shuffled.getShuffleResult());
    }
    auto leaderIf = b.create<scf::IfOp>(loc, isLeader, /*withElse=*/false);
    {
      OpBuilder::InsertionGuard guard(b);
      b.setInsertionPointToStart(leaderIf.thenBlock());
      b.create<memref::StoreOp>(loc, accum, red, ValueRange{group});
    }
    b.create<gpu::BarrierOp>(loc);
    Value total = b.create<memref::LoadOp>(loc, red, ValueRange{c0});
    for (int64_t gi = 1; gi < NGROUPS; ++gi)
      total = b.create<arith::AddFOp>(
          loc, total, b.create<memref::LoadOp>(loc, red, ValueRange{ci(gi)}));
    b.create<gpu::BarrierOp>(loc);
    return total;
  };

  Value mean = zerof;
  Value inverse;
  if (isLayerNorm) {
    mean = b.create<arith::DivFOp>(
        loc, reduceRow([&](Value idx) { return loadF32(X, idx); }), Kf);
    Value varianceSum = reduceRow([&](Value idx) {
      Value centered = b.create<arith::SubFOp>(loc, loadF32(X, idx), mean);
      return b.create<arith::MulFOp>(loc, centered, centered).getResult();
    });
    Value variance = b.create<arith::DivFOp>(loc, varianceSum, Kf);
    Value denom = b.create<math::SqrtOp>(
        loc, b.create<arith::AddFOp>(loc, variance, eps));
    inverse = b.create<arith::DivFOp>(
        loc, b.create<arith::ConstantOp>(loc, f32, b.getF32FloatAttr(1.0)),
        denom);
  } else {
    Value squareSum = reduceRow([&](Value idx) {
      Value x = loadF32(X, idx);
      return b.create<arith::MulFOp>(loc, x, x).getResult();
    });
    Value meanSquare = b.create<arith::DivFOp>(loc, squareSum, Kf);
    Value denom = b.create<math::SqrtOp>(
        loc, b.create<arith::AddFOp>(loc, meanSquare, eps));
    inverse = b.create<arith::DivFOp>(
        loc, b.create<arith::ConstantOp>(loc, f32, b.getF32FloatAttr(1.0)),
        denom);
  }

  auto normalizedAt = [&](Value idx) -> Value {
    Value x = loadF32(X, idx);
    if (isLayerNorm)
      x = b.create<arith::SubFOp>(loc, x, mean);
    return b.create<arith::MulFOp>(loc, x, inverse);
  };
  auto dzAt = [&](Value idx) -> Value {
    Value dy = loadF32(dY, idx);
    Value channel = b.create<arith::RemUIOp>(loc, idx, K);
    auto gammaIf = b.create<scf::IfOp>(loc, TypeRange{f32}, hasGamma,
                                       /*withElse=*/true);
    {
      OpBuilder::InsertionGuard guard(b);
      b.setInsertionPointToStart(gammaIf.thenBlock());
      Value scaled = b.create<arith::MulFOp>(loc, dy, loadGamma(channel));
      b.create<scf::YieldOp>(loc, ValueRange{scaled});
      b.setInsertionPointToStart(gammaIf.elseBlock());
      b.create<scf::YieldOp>(loc, ValueRange{dy});
    }
    return gammaIf.getResult(0);
  };

  Value meanDz = zerof;
  if (isLayerNorm)
    meanDz = b.create<arith::DivFOp>(loc, reduceRow(dzAt), Kf);
  Value meanDzZ = b.create<arith::DivFOp>(
      loc,
      reduceRow([&](Value idx) {
        return b.create<arith::MulFOp>(loc, dzAt(idx), normalizedAt(idx))
            .getResult();
      }),
      Kf);

  auto write = b.create<scf::ForOp>(loc, tid, K, cBD);
  {
    OpBuilder::InsertionGuard guard(b);
    b.setInsertionPointToStart(write.getBody());
    Value channel = write.getInductionVar();
    Value idx = b.create<arith::AddIOp>(loc, base, channel);
    Value dy = loadF32(dY, idx);
    Value z = normalizedAt(idx);
    Value inner = b.create<arith::SubFOp>(loc, dzAt(idx), meanDz);
    inner = b.create<arith::SubFOp>(
        loc, inner, b.create<arith::MulFOp>(loc, z, meanDzZ));
    storeDX(b.create<arith::MulFOp>(loc, inverse, inner), idx);

    auto gammaIf = b.create<scf::IfOp>(loc, hasGamma, /*withElse=*/false);
    {
      OpBuilder::InsertionGuard affineGuard(b);
      b.setInsertionPointToStart(gammaIf.thenBlock());
      Value contribution = b.create<arith::MulFOp>(loc, dy, z);
      b.create<memref::StoreOp>(loc, contribution, dGammaPartials,
                                ValueRange{idx});
    }
  }

  b.setInsertionPointToEnd(&f.getBody().front());
  b.create<gpu::ReturnOp>(loc);
}

void emitNormBackwardFinalizeBody(OpBuilder &b, Location loc,
                                  gpu::GPUFuncOp f, Type storeTy) {
  Type f32 = b.getF32Type();
  b.setInsertionPointToStart(&f.getBody().front());
  Value dGammaPartials = f.getArgument(0);
  Value dY = f.getArgument(1);
  Value dGamma = f.getArgument(2), dBeta = f.getArgument(3);
  Value M = f.getArgument(4), K = f.getArgument(5);
  Value hasGamma = f.getArgument(6), hasBeta = f.getArgument(7);

  auto ci = [&](int64_t value) {
    return b.create<arith::ConstantIndexOp>(loc, value);
  };
  Value c0 = ci(0), c1 = ci(1), cBD = ci(BD);
  Value block = b.create<gpu::BlockIdOp>(loc, gpu::Dimension::x);
  Value tid = b.create<gpu::ThreadIdOp>(loc, gpu::Dimension::x);
  Value channel = b.create<arith::AddIOp>(
      loc, b.create<arith::MulIOp>(loc, block, cBD), tid);
  Value inBounds =
      b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::slt, channel, K);
  auto channelIf = b.create<scf::IfOp>(loc, inBounds, /*withElse=*/false);
  b.setInsertionPointToStart(channelIf.thenBlock());
  Value zero = b.create<arith::ConstantOp>(loc, f32, b.getF32FloatAttr(0.0));

  auto foldRows = [&](Value partials) -> Value {
    auto loop = b.create<scf::ForOp>(loc, c0, M, c1, ValueRange{zero});
    {
      OpBuilder::InsertionGuard guard(b);
      b.setInsertionPointToStart(loop.getBody());
      Value index = b.create<arith::AddIOp>(
          loc, b.create<arith::MulIOp>(loc, loop.getInductionVar(), K),
          channel);
      Value contribution =
          b.create<memref::LoadOp>(loc, partials, ValueRange{index});
      Value sum = b.create<arith::AddFOp>(
          loc, loop.getRegionIterArgs()[0], contribution);
      b.create<scf::YieldOp>(loc, ValueRange{sum});
    }
    return loop.getResult(0);
  };
  auto foldDY = [&]() -> Value {
    auto loop = b.create<scf::ForOp>(loc, c0, M, c1, ValueRange{zero});
    {
      OpBuilder::InsertionGuard guard(b);
      b.setInsertionPointToStart(loop.getBody());
      Value index = b.create<arith::AddIOp>(
          loc, b.create<arith::MulIOp>(loc, loop.getInductionVar(), K),
          channel);
      Value contribution =
          b.create<memref::LoadOp>(loc, dY, ValueRange{index});
      if (!storeTy.isF32())
        contribution = b.create<arith::ExtFOp>(loc, f32, contribution);
      Value sum = b.create<arith::AddFOp>(
          loc, loop.getRegionIterArgs()[0], contribution);
      b.create<scf::YieldOp>(loc, ValueRange{sum});
    }
    return loop.getResult(0);
  };

  auto gammaIf = b.create<scf::IfOp>(loc, hasGamma, /*withElse=*/false);
  {
    OpBuilder::InsertionGuard guard(b);
    b.setInsertionPointToStart(gammaIf.thenBlock());
    b.create<memref::StoreOp>(loc, foldRows(dGammaPartials), dGamma,
                              ValueRange{channel});
  }
  auto betaIf = b.create<scf::IfOp>(loc, hasBeta, /*withElse=*/false);
  {
    OpBuilder::InsertionGuard guard(b);
    b.setInsertionPointToStart(betaIf.thenBlock());
    b.create<memref::StoreOp>(loc, foldDY(), dBeta,
                              ValueRange{channel});
  }

  b.setInsertionPointToEnd(&f.getBody().front());
  b.create<gpu::ReturnOp>(loc);
}

struct GenerateROCMNormKernelPass
    : PassWrapper<GenerateROCMNormKernelPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(GenerateROCMNormKernelPass)

  StringRef getArgument() const final { return "generate-rocm-norm-kernel"; }
  StringRef getDescription() const final {
    return "Expand a tessera_rocm.norm directive into a row-reduction "
           "(rmsnorm / layer_norm over the last axis) gpu kernel "
           "(compiler-generated)";
  }
  void getDependentDialects(DialectRegistry &registry) const final {
    registry.insert<gpu::GPUDialect, scf::SCFDialect, arith::ArithDialect,
                    math::MathDialect, memref::MemRefDialect>();
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();
    SmallVector<Operation *> directives;
    module.walk([&](Operation *op) {
      if (op->getName().getStringRef() == "tessera_rocm.norm")
        directives.push_back(op);
    });
    for (Operation *op : directives) {
      auto nameAttr = op->getAttrOfType<StringAttr>("name");
      if (!nameAttr) {
        op->emitError("tessera_rocm.norm missing name");
        return signalPassFailure();
      }
      StringRef kind = "rmsnorm";
      if (auto a = op->getAttrOfType<StringAttr>("kind"))
        kind = a.getValue();
      if (kind != "rmsnorm" && kind != "layer_norm") {
        op->emitError("generate-rocm-norm-kernel: kind must be rmsnorm or "
                      "layer_norm (got '")
            << kind << "')";
        return signalPassFailure();
      }
      OpBuilder b(module.getBodyRegion());
      b.setInsertionPointToEnd(module.getBody());
      Location loc = op->getLoc();
      std::string kname = nameAttr.getValue().str();
      bool backward = false;
      if (auto a = op->getAttrOfType<BoolAttr>("backward"))
        backward = a.getValue();
      StringRef epilogue = "none";
      if (auto a = op->getAttrOfType<StringAttr>("epilogue"))
        epilogue = a.getValue();
      if (epilogue != "none" && epilogue != "relu" && epilogue != "silu") {
        op->emitError("generate-rocm-norm-kernel: epilogue must be none, relu, "
                      "or silu (got '")
            << epilogue << "')";
        return signalPassFailure();
      }
      if (backward && epilogue != "none") {
        op->emitError("generate-rocm-norm-kernel: backward normalization does "
                      "not accept a forward epilogue");
        return signalPassFailure();
      }

      Type storeTy = b.getF32Type();
      if (auto a = op->getAttrOfType<StringAttr>("dtype")) {
        StringRef dt = a.getValue();
        if (dt == "f16" || dt == "float16")
          storeTy = b.getF16Type();
        else if (dt == "bf16" || dt == "bfloat16")
          storeTy = b.getBF16Type();
        else if (dt != "f32" && dt != "float32") {
          op->emitError("generate-rocm-norm-kernel: dtype must be f32, f16, or "
                        "bf16 (got '")
              << dt << "')";
          return signalPassFailure();
        }
      }

      auto gpuMod = b.create<gpu::GPUModuleOp>(loc, kname + "_mod");
      b.setInsertionPointToStart(&gpuMod.getBodyRegion().front());
      Type f32 = b.getF32Type();
      Type idxTy = b.getIndexType();
      auto memTy = MemRefType::get({ShapedType::kDynamic}, storeTy);
      FunctionType fnTy;
      if (backward) {
        auto accumTy = MemRefType::get({ShapedType::kDynamic}, f32);
        // X, gamma, dY, dX use storage dtype. The row kernel writes f32
        // dGamma partials; dBeta is folded directly from dY by the finalize
        // kernel. Beta values are absent because its derivative is independent
        // of beta.
        fnTy = b.getFunctionType(
            {memTy, memTy, memTy, memTy, accumTy, idxTy, idxTy, f32,
             b.getI1Type()},
            {});
      } else {
        // X, gamma, beta, O plus runtime shape/epsilon and uniform flags.
        fnTy = b.getFunctionType(
            {memTy, memTy, memTy, memTy, idxTy, idxTy, f32, b.getI1Type(),
             b.getI1Type()},
            {});
      }
      auto gpuFunc = b.create<gpu::GPUFuncOp>(loc, kname, fnTy);
      gpuFunc->setAttr(gpu::GPUDialect::getKernelFuncAttrName(),
                       b.getUnitAttr());
      OpBuilder body(gpuFunc.getContext());
      if (backward)
        emitNormBackwardBody(body, loc, gpuFunc, storeTy,
                             kind == "layer_norm");
      else
        emitNormBody(body, loc, gpuFunc, storeTy, kind == "layer_norm",
                     epilogue);
      if (backward) {
        auto accumTy = MemRefType::get({ShapedType::kDynamic}, f32);
        auto finalizeTy = b.getFunctionType(
            {accumTy, memTy, accumTy, accumTy, idxTy, idxTy,
             b.getI1Type(), b.getI1Type()},
            {});
        auto finalize =
            b.create<gpu::GPUFuncOp>(loc, kname + "_reduce", finalizeTy);
        finalize->setAttr(gpu::GPUDialect::getKernelFuncAttrName(),
                          b.getUnitAttr());
        OpBuilder finalizeBody(finalize.getContext());
        emitNormBackwardFinalizeBody(finalizeBody, loc, finalize, storeTy);
      }
      op->erase();
    }
  }
};

} // namespace

std::unique_ptr<mlir::Pass>
mlir::tessera_rocm::createGenerateROCMNormKernelPass() {
  return std::make_unique<GenerateROCMNormKernelPass>();
}
