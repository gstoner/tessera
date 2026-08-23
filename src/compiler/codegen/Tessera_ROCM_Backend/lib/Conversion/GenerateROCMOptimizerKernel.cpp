//===- GenerateROCMOptimizerKernel.cpp - fused optimizer-step gpu kernel --===//
//
// Expands `tessera_rocm.optimizer` into a fused per-parameter optimizer update,
// one thread per parameter element. The `kind` StrAttr selects the rule at
// codegen (one cached kernel per optimizer); the bias-correction scalars
// (1-β^t) are computed on the host and passed in:
//
//   sgd      : p -= lr·g
//   momentum : v = β1·v + g ; p -= lr·v
//   nesterov : v = β1·v + g ; p -= lr·(g + β1·v)   (look-ahead momentum)
//   adam     : m=β1·m+(1-β1)g ; v=β2·v+(1-β2)g² ; p -= lr·(m/b1c)/(√(v/b2c)+eps)
//   adamw    : p *= (1-lr·wd) ; then adam (decoupled decay)
//   lion     : u=β1·m+(1-β1)g ; m=β2·m+(1-β2)g ; p *= (1-lr·wd) ; p -= lr·sign(u)
//
// Buffers p/g/m/v in, p_out/m_out/v_out out. Scalars (lr,β1,β2,eps,wd,b1c,b2c)
// are f32 kernel args. √ via math→ROCDL. All f32. CPU analog:
// avx512_optimizer_f32.
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

using namespace mlir;

namespace {

static constexpr int64_t BD = 256;

static Value flatGid(OpBuilder &b, Location loc) {
  Value bid = b.create<gpu::BlockIdOp>(loc, gpu::Dimension::x);
  Value tid = b.create<gpu::ThreadIdOp>(loc, gpu::Dimension::x);
  Value block = b.create<arith::ConstantIndexOp>(loc, BD);
  return b.create<arith::AddIOp>(
      loc, b.create<arith::MulIOp>(loc, bid, block), tid);
}

static void emitAdafactorMomentBody(OpBuilder &b, Location loc,
                                    gpu::GPUFuncOp f, bool rows) {
  b.setInsertionPointToStart(&f.getBody().front());
  Value gradient = f.getArgument(0), oldMoment = f.getArgument(1);
  Value newMoment = f.getArgument(2), m = f.getArgument(3);
  Value n = f.getArgument(4), beta2 = f.getArgument(5);
  Value gid = flatGid(b, loc);
  Value extent = rows ? m : n;
  Value inBounds = b.create<arith::CmpIOp>(
      loc, arith::CmpIPredicate::slt, gid, extent);
  auto guard = b.create<scf::IfOp>(loc, inBounds, false);
  b.setInsertionPointToStart(guard.thenBlock());
  Type f32 = b.getF32Type();
  Value zero = b.create<arith::ConstantOp>(
      loc, f32, b.getF32FloatAttr(0.0f));
  Value one = b.create<arith::ConstantOp>(
      loc, f32, b.getF32FloatAttr(1.0f));
  Value lower = b.create<arith::ConstantIndexOp>(loc, 0);
  Value step = b.create<arith::ConstantIndexOp>(loc, 1);
  Value reductionExtent = rows ? n : m;
  auto loop = b.create<scf::ForOp>(loc, lower, reductionExtent, step,
                                   ValueRange{zero});
  b.setInsertionPointToStart(loop.getBody());
  Value k = loop.getInductionVar();
  Value linear = rows
                     ? b.create<arith::AddIOp>(
                           loc, b.create<arith::MulIOp>(loc, gid, n), k)
                     : b.create<arith::AddIOp>(
                           loc, b.create<arith::MulIOp>(loc, k, n), gid);
  Value g = b.create<memref::LoadOp>(
      loc, gradient, ValueRange{linear});
  Value square = b.create<arith::MulFOp>(loc, g, g);
  Value next = b.create<arith::AddFOp>(
      loc, loop.getRegionIterArg(0), square);
  b.create<scf::YieldOp>(loc, next);
  b.setInsertionPointAfter(loop);
  Value divisorInt = b.create<arith::IndexCastUIOp>(
      loc, b.getI64Type(), reductionExtent);
  Value divisor =
      b.create<arith::UIToFPOp>(loc, f32, divisorInt);
  Value average =
      b.create<arith::DivFOp>(loc, loop.getResult(0), divisor);
  Value old = b.create<memref::LoadOp>(
      loc, oldMoment, ValueRange{gid});
  Value updated = b.create<arith::AddFOp>(
      loc, b.create<arith::MulFOp>(loc, beta2, old),
      b.create<arith::MulFOp>(
          loc, b.create<arith::SubFOp>(loc, one, beta2), average));
  b.create<memref::StoreOp>(
      loc, updated, newMoment, ValueRange{gid});
  b.setInsertionPointToEnd(&f.getBody().front());
  b.create<gpu::ReturnOp>(loc);
}

static void emitAdafactorMeanBody(OpBuilder &b, Location loc,
                                  gpu::GPUFuncOp f) {
  b.setInsertionPointToStart(&f.getBody().front());
  Value row = f.getArgument(0), mean = f.getArgument(1);
  Value m = f.getArgument(2);
  Value gid = flatGid(b, loc);
  Value zeroIndex = b.create<arith::ConstantIndexOp>(loc, 0);
  Value isOwner = b.create<arith::CmpIOp>(
      loc, arith::CmpIPredicate::eq, gid, zeroIndex);
  auto guard = b.create<scf::IfOp>(loc, isOwner, false);
  b.setInsertionPointToStart(guard.thenBlock());
  Type f32 = b.getF32Type();
  Value zero = b.create<arith::ConstantOp>(
      loc, f32, b.getF32FloatAttr(0.0f));
  Value oneIndex = b.create<arith::ConstantIndexOp>(loc, 1);
  auto loop =
      b.create<scf::ForOp>(loc, zeroIndex, m, oneIndex, ValueRange{zero});
  b.setInsertionPointToStart(loop.getBody());
  Value value = b.create<memref::LoadOp>(
      loc, row, ValueRange{loop.getInductionVar()});
  Value next =
      b.create<arith::AddFOp>(loc, loop.getRegionIterArg(0), value);
  b.create<scf::YieldOp>(loc, ValueRange{next});
  b.setInsertionPointAfter(loop);
  Value divisorInt =
      b.create<arith::IndexCastUIOp>(loc, b.getI64Type(), m);
  Value divisor =
      b.create<arith::UIToFPOp>(loc, f32, divisorInt);
  Value average =
      b.create<arith::DivFOp>(loc, loop.getResult(0), divisor);
  b.create<memref::StoreOp>(
      loc, average, mean, ValueRange{zeroIndex});
  b.setInsertionPointToEnd(&f.getBody().front());
  b.create<gpu::ReturnOp>(loc);
}

static void emitAdafactorUpdateBody(OpBuilder &b, Location loc,
                                    gpu::GPUFuncOp f) {
  b.setInsertionPointToStart(&f.getBody().front());
  Value parameter = f.getArgument(0), gradient = f.getArgument(1);
  Value row = f.getArgument(2), col = f.getArgument(3);
  Value mean = f.getArgument(4), output = f.getArgument(5);
  Value m = f.getArgument(6), n = f.getArgument(7);
  Value lr = f.getArgument(8), eps = f.getArgument(9);
  Value gid = flatGid(b, loc);
  Value count = b.create<arith::MulIOp>(loc, m, n);
  Value inBounds = b.create<arith::CmpIOp>(
      loc, arith::CmpIPredicate::slt, gid, count);
  auto guard = b.create<scf::IfOp>(loc, inBounds, false);
  b.setInsertionPointToStart(guard.thenBlock());
  Value rowIndex = b.create<arith::DivUIOp>(loc, gid, n);
  Value colIndex = b.create<arith::RemUIOp>(loc, gid, n);
  Value rowValue =
      b.create<memref::LoadOp>(loc, row, ValueRange{rowIndex});
  Value colValue =
      b.create<memref::LoadOp>(loc, col, ValueRange{colIndex});
  Value zeroIndex = b.create<arith::ConstantIndexOp>(loc, 0);
  Value meanValue =
      b.create<memref::LoadOp>(loc, mean, ValueRange{zeroIndex});
  // maximumf (NaN-propagating), not maxnumf: the reference floors are
  // np.maximum (optim.py _adafactor_update_from_state), so a NaN statistic
  // must surface as NaN in every update it feeds — maxnumf silently
  // laundered it into eps, giving finite-but-wrong updates to every other
  // parameter sharing the poisoned row/col (JIT-MATH-AUDIT-2026-08-23).
  // Same rule for every eps/tiny floor in this file.
  Value safeRow = b.create<arith::MaximumFOp>(loc, rowValue, eps);
  Value safeCol = b.create<arith::MaximumFOp>(loc, colValue, eps);
  Value safeMean = b.create<arith::MaximumFOp>(loc, meanValue, eps);
  Value scale = b.create<arith::DivFOp>(
      loc, b.create<arith::MulFOp>(loc, safeRow, safeCol), safeMean);
  Value denominator = b.create<arith::AddFOp>(
      loc, b.create<math::SqrtOp>(loc, scale), eps);
  Value p =
      b.create<memref::LoadOp>(loc, parameter, ValueRange{gid});
  Value g =
      b.create<memref::LoadOp>(loc, gradient, ValueRange{gid});
  Value update = b.create<arith::DivFOp>(loc, g, denominator);
  b.create<memref::StoreOp>(
      loc,
      b.create<arith::SubFOp>(
          loc, p, b.create<arith::MulFOp>(loc, lr, update)),
      output, ValueRange{gid});
  b.setInsertionPointToEnd(&f.getBody().front());
  b.create<gpu::ReturnOp>(loc);
}

static void emitAdafactorFullBody(OpBuilder &b, Location loc,
                                  gpu::GPUFuncOp f) {
  b.setInsertionPointToStart(&f.getBody().front());
  Value parameter = f.getArgument(0), gradient = f.getArgument(1);
  Value oldMoment = f.getArgument(2), output = f.getArgument(3);
  Value newMoment = f.getArgument(4), count = f.getArgument(5);
  Value lr = f.getArgument(6), beta2 = f.getArgument(7);
  Value eps = f.getArgument(8);
  Value gid = flatGid(b, loc);
  Value inBounds = b.create<arith::CmpIOp>(
      loc, arith::CmpIPredicate::slt, gid, count);
  auto guard = b.create<scf::IfOp>(loc, inBounds, false);
  b.setInsertionPointToStart(guard.thenBlock());
  Type f32 = b.getF32Type();
  Value one = b.create<arith::ConstantOp>(
      loc, f32, b.getF32FloatAttr(1.0f));
  Value p = b.create<memref::LoadOp>(
      loc, parameter, ValueRange{gid});
  Value g = b.create<memref::LoadOp>(
      loc, gradient, ValueRange{gid});
  Value old = b.create<memref::LoadOp>(
      loc, oldMoment, ValueRange{gid});
  Value moment = b.create<arith::AddFOp>(
      loc, b.create<arith::MulFOp>(loc, beta2, old),
      b.create<arith::MulFOp>(
          loc, b.create<arith::SubFOp>(loc, one, beta2),
          b.create<arith::MulFOp>(loc, g, g)));
  Value safeMoment = b.create<arith::MaximumFOp>(loc, moment, eps);
  Value denominator = b.create<arith::AddFOp>(
      loc, b.create<math::SqrtOp>(loc, safeMoment), eps);
  Value updated = b.create<arith::SubFOp>(
      loc, p,
      b.create<arith::MulFOp>(
          loc, lr, b.create<arith::DivFOp>(loc, g, denominator)));
  b.create<memref::StoreOp>(
      loc, updated, output, ValueRange{gid});
  b.create<memref::StoreOp>(
      loc, moment, newMoment, ValueRange{gid});
  b.setInsertionPointToEnd(&f.getBody().front());
  b.create<gpu::ReturnOp>(loc);
}

static Value emitAdafactorDScale(OpBuilder &b, Location loc, Value gradient,
                                 Value cotangent, Value row, Value col,
                                 Value mean, Value lr, Value eps) {
  Type f32 = b.getF32Type();
  Value tiny = b.create<arith::ConstantOp>(
      loc, f32, b.getF32FloatAttr(1.0e-30f));
  Value two = b.create<arith::ConstantOp>(
      loc, f32, b.getF32FloatAttr(2.0f));
  Value safeRow = b.create<arith::MaximumFOp>(loc, row, eps);
  Value safeCol = b.create<arith::MaximumFOp>(loc, col, eps);
  Value safeMean = b.create<arith::MaximumFOp>(loc, mean, eps);
  Value scale = b.create<arith::DivFOp>(
      loc, b.create<arith::MulFOp>(loc, safeRow, safeCol), safeMean);
  Value root = b.create<math::SqrtOp>(loc, scale);
  Value safeRoot = b.create<arith::MaximumFOp>(loc, root, tiny);
  Value denominator = b.create<arith::AddFOp>(loc, root, eps);
  Value denominatorSquared =
      b.create<arith::MulFOp>(loc, denominator, denominator);
  Value divisor = b.create<arith::MulFOp>(
      loc, two,
      b.create<arith::MulFOp>(loc, safeRoot, denominatorSquared));
  return b.create<arith::DivFOp>(
      loc,
      b.create<arith::MulFOp>(
          loc, lr,
          b.create<arith::MulFOp>(loc, cotangent, gradient)),
      divisor);
}

static void emitAdafactorBackwardMeanBody(OpBuilder &b, Location loc,
                                          gpu::GPUFuncOp f) {
  b.setInsertionPointToStart(&f.getBody().front());
  Value gradient = f.getArgument(0), cotangent = f.getArgument(1);
  Value row = f.getArgument(2), col = f.getArgument(3);
  Value mean = f.getArgument(4), dmean = f.getArgument(5);
  Value m = f.getArgument(6), n = f.getArgument(7);
  Value lr = f.getArgument(8), eps = f.getArgument(9);
  Value gid = flatGid(b, loc);
  Value zeroIndex = b.create<arith::ConstantIndexOp>(loc, 0);
  Value isOwner = b.create<arith::CmpIOp>(
      loc, arith::CmpIPredicate::eq, gid, zeroIndex);
  auto guard = b.create<scf::IfOp>(loc, isOwner, false);
  b.setInsertionPointToStart(guard.thenBlock());
  Type f32 = b.getF32Type();
  Value zero = b.create<arith::ConstantOp>(
      loc, f32, b.getF32FloatAttr(0.0f));
  Value oneIndex = b.create<arith::ConstantIndexOp>(loc, 1);
  Value count = b.create<arith::MulIOp>(loc, m, n);
  auto loop =
      b.create<scf::ForOp>(loc, zeroIndex, count, oneIndex, ValueRange{zero});
  b.setInsertionPointToStart(loop.getBody());
  Value linear = loop.getInductionVar();
  Value rowIndex = b.create<arith::DivUIOp>(loc, linear, n);
  Value colIndex = b.create<arith::RemUIOp>(loc, linear, n);
  Value g = b.create<memref::LoadOp>(loc, gradient, ValueRange{linear});
  Value dy = b.create<memref::LoadOp>(loc, cotangent, ValueRange{linear});
  Value rv = b.create<memref::LoadOp>(loc, row, ValueRange{rowIndex});
  Value cv = b.create<memref::LoadOp>(loc, col, ValueRange{colIndex});
  Value mv = b.create<memref::LoadOp>(loc, mean, ValueRange{zeroIndex});
  Value ds = emitAdafactorDScale(b, loc, g, dy, rv, cv, mv, lr, eps);
  Value safeRow = b.create<arith::MaximumFOp>(loc, rv, eps);
  Value safeCol = b.create<arith::MaximumFOp>(loc, cv, eps);
  Value safeMean = b.create<arith::MaximumFOp>(loc, mv, eps);
  Value numerator = b.create<arith::MulFOp>(
      loc, ds, b.create<arith::MulFOp>(loc, safeRow, safeCol));
  Value contribution = b.create<arith::DivFOp>(
      loc, numerator,
      b.create<arith::MulFOp>(loc, safeMean, safeMean));
  Value next = b.create<arith::AddFOp>(
      loc, loop.getRegionIterArg(0), contribution);
  b.create<scf::YieldOp>(loc, ValueRange{next});
  b.setInsertionPointAfter(loop);
  b.create<memref::StoreOp>(
      loc, b.create<arith::SubFOp>(loc, zero, loop.getResult(0)), dmean,
      ValueRange{zeroIndex});
  b.setInsertionPointToEnd(&f.getBody().front());
  b.create<gpu::ReturnOp>(loc);
}

static void emitAdafactorBackwardMomentBody(OpBuilder &b, Location loc,
                                            gpu::GPUFuncOp f, bool rows) {
  b.setInsertionPointToStart(&f.getBody().front());
  Value gradient = f.getArgument(0), cotangent = f.getArgument(1);
  Value row = f.getArgument(2), col = f.getArgument(3);
  Value mean = f.getArgument(4), dmean = f.getArgument(5);
  Value raw = f.getArgument(6), oldStateGrad = f.getArgument(7);
  Value m = f.getArgument(8), n = f.getArgument(9);
  Value lr = f.getArgument(10), beta2 = f.getArgument(11);
  Value eps = f.getArgument(12);
  Value gid = flatGid(b, loc);
  Value extent = rows ? m : n;
  Value inBounds = b.create<arith::CmpIOp>(
      loc, arith::CmpIPredicate::slt, gid, extent);
  auto guard = b.create<scf::IfOp>(loc, inBounds, false);
  b.setInsertionPointToStart(guard.thenBlock());
  Type f32 = b.getF32Type();
  Value zero = b.create<arith::ConstantOp>(
      loc, f32, b.getF32FloatAttr(0.0f));
  Value one = b.create<arith::ConstantOp>(
      loc, f32, b.getF32FloatAttr(1.0f));
  Value zeroIndex = b.create<arith::ConstantIndexOp>(loc, 0);
  Value oneIndex = b.create<arith::ConstantIndexOp>(loc, 1);
  Value reductionExtent = rows ? n : m;
  auto loop = b.create<scf::ForOp>(
      loc, zeroIndex, reductionExtent, oneIndex, ValueRange{zero});
  b.setInsertionPointToStart(loop.getBody());
  Value k = loop.getInductionVar();
  Value linear = rows
                     ? b.create<arith::AddIOp>(
                           loc, b.create<arith::MulIOp>(loc, gid, n), k)
                     : b.create<arith::AddIOp>(
                           loc, b.create<arith::MulIOp>(loc, k, n), gid);
  Value rowIndex = rows ? gid : k;
  Value colIndex = rows ? k : gid;
  Value g = b.create<memref::LoadOp>(loc, gradient, ValueRange{linear});
  Value dy = b.create<memref::LoadOp>(loc, cotangent, ValueRange{linear});
  Value rv = b.create<memref::LoadOp>(loc, row, ValueRange{rowIndex});
  Value cv = b.create<memref::LoadOp>(loc, col, ValueRange{colIndex});
  Value mv = b.create<memref::LoadOp>(loc, mean, ValueRange{zeroIndex});
  Value ds = emitAdafactorDScale(b, loc, g, dy, rv, cv, mv, lr, eps);
  Value safeOther = b.create<arith::MaximumFOp>(
      loc, rows ? cv : rv, eps);
  Value safeMean = b.create<arith::MaximumFOp>(loc, mv, eps);
  Value contribution = b.create<arith::DivFOp>(
      loc, b.create<arith::MulFOp>(loc, ds, safeOther), safeMean);
  Value next = b.create<arith::AddFOp>(
      loc, loop.getRegionIterArg(0), contribution);
  b.create<scf::YieldOp>(loc, ValueRange{next});
  b.setInsertionPointAfter(loop);
  Value current =
      b.create<memref::LoadOp>(loc, rows ? row : col, ValueRange{gid});
  Value active = b.create<arith::CmpFOp>(
      loc, arith::CmpFPredicate::OGT, current, eps);
  Value value = b.create<arith::SelectOp>(
      loc, active, loop.getResult(0), zero);
  if (rows) {
    Value mv = b.create<memref::LoadOp>(loc, mean, ValueRange{zeroIndex});
    Value meanActive = b.create<arith::CmpFOp>(
        loc, arith::CmpFPredicate::OGT, mv, eps);
    Value dmv = b.create<memref::LoadOp>(loc, dmean, ValueRange{zeroIndex});
    Value divisorInt =
        b.create<arith::IndexCastUIOp>(loc, b.getI64Type(), m);
    Value divisor =
        b.create<arith::UIToFPOp>(loc, f32, divisorInt);
    Value adjustment = b.create<arith::SelectOp>(
        loc, meanActive, b.create<arith::DivFOp>(loc, dmv, divisor), zero);
    value = b.create<arith::AddFOp>(loc, value, adjustment);
  }
  b.create<memref::StoreOp>(loc, value, raw, ValueRange{gid});
  b.create<memref::StoreOp>(
      loc, b.create<arith::MulFOp>(loc, beta2, value), oldStateGrad,
      ValueRange{gid});
  b.setInsertionPointToEnd(&f.getBody().front());
  b.create<gpu::ReturnOp>(loc);
}

static void emitAdafactorBackwardFinalizeBody(OpBuilder &b, Location loc,
                                              gpu::GPUFuncOp f) {
  b.setInsertionPointToStart(&f.getBody().front());
  Value gradient = f.getArgument(0), cotangent = f.getArgument(1);
  Value row = f.getArgument(2), col = f.getArgument(3);
  Value mean = f.getArgument(4), drow = f.getArgument(5);
  Value dcol = f.getArgument(6), dparam = f.getArgument(7);
  Value dgrad = f.getArgument(8), m = f.getArgument(9);
  Value n = f.getArgument(10), lr = f.getArgument(11);
  Value beta2 = f.getArgument(12), eps = f.getArgument(13);
  Value gid = flatGid(b, loc);
  Value count = b.create<arith::MulIOp>(loc, m, n);
  Value inBounds = b.create<arith::CmpIOp>(
      loc, arith::CmpIPredicate::slt, gid, count);
  auto guard = b.create<scf::IfOp>(loc, inBounds, false);
  b.setInsertionPointToStart(guard.thenBlock());
  Type f32 = b.getF32Type();
  Value one = b.create<arith::ConstantOp>(
      loc, f32, b.getF32FloatAttr(1.0f));
  Value zero = b.create<arith::ConstantOp>(
      loc, f32, b.getF32FloatAttr(0.0f));
  Value rowIndex = b.create<arith::DivUIOp>(loc, gid, n);
  Value colIndex = b.create<arith::RemUIOp>(loc, gid, n);
  Value g = b.create<memref::LoadOp>(loc, gradient, ValueRange{gid});
  Value dy = b.create<memref::LoadOp>(loc, cotangent, ValueRange{gid});
  Value rv = b.create<memref::LoadOp>(loc, row, ValueRange{rowIndex});
  Value cv = b.create<memref::LoadOp>(loc, col, ValueRange{colIndex});
  Value zeroIndex = b.create<arith::ConstantIndexOp>(loc, 0);
  Value mv = b.create<memref::LoadOp>(loc, mean, ValueRange{zeroIndex});
  Value safeRow = b.create<arith::MaximumFOp>(loc, rv, eps);
  Value safeCol = b.create<arith::MaximumFOp>(loc, cv, eps);
  Value safeMean = b.create<arith::MaximumFOp>(loc, mv, eps);
  Value root = b.create<math::SqrtOp>(
      loc, b.create<arith::DivFOp>(
               loc, b.create<arith::MulFOp>(loc, safeRow, safeCol), safeMean));
  Value denominator = b.create<arith::AddFOp>(loc, root, eps);
  Value direct = b.create<arith::DivFOp>(
      loc, b.create<arith::MulFOp>(
               loc, b.create<arith::SubFOp>(loc, zero, lr), dy),
      denominator);
  Value dr = b.create<memref::LoadOp>(loc, drow, ValueRange{rowIndex});
  Value dc = b.create<memref::LoadOp>(loc, dcol, ValueRange{colIndex});
  Value mInt = b.create<arith::IndexCastUIOp>(loc, b.getI64Type(), m);
  Value nInt = b.create<arith::IndexCastUIOp>(loc, b.getI64Type(), n);
  Value mf = b.create<arith::UIToFPOp>(loc, f32, mInt);
  Value nf = b.create<arith::UIToFPOp>(loc, f32, nInt);
  Value stateContribution = b.create<arith::AddFOp>(
      loc, b.create<arith::DivFOp>(loc, dr, nf),
      b.create<arith::DivFOp>(loc, dc, mf));
  Value momentContribution = b.create<arith::MulFOp>(
      loc,
      b.create<arith::MulFOp>(
          loc, b.create<arith::ConstantOp>(
                   loc, f32, b.getF32FloatAttr(2.0f)),
          b.create<arith::SubFOp>(loc, one, beta2)),
      b.create<arith::MulFOp>(loc, g, stateContribution));
  b.create<memref::StoreOp>(loc, dy, dparam, ValueRange{gid});
  b.create<memref::StoreOp>(
      loc, b.create<arith::AddFOp>(loc, direct, momentContribution), dgrad,
      ValueRange{gid});
  b.setInsertionPointToEnd(&f.getBody().front());
  b.create<gpu::ReturnOp>(loc);
}

static void emitAdafactorFullBackwardBody(OpBuilder &b, Location loc,
                                          gpu::GPUFuncOp f) {
  b.setInsertionPointToStart(&f.getBody().front());
  Value gradient = f.getArgument(0), oldMoment = f.getArgument(1);
  Value cotangent = f.getArgument(2), dparam = f.getArgument(3);
  Value dgrad = f.getArgument(4), dmoment = f.getArgument(5);
  Value count = f.getArgument(6), lr = f.getArgument(7);
  Value beta2 = f.getArgument(8), eps = f.getArgument(9);
  Value gid = flatGid(b, loc);
  Value inBounds = b.create<arith::CmpIOp>(
      loc, arith::CmpIPredicate::slt, gid, count);
  auto guard = b.create<scf::IfOp>(loc, inBounds, false);
  b.setInsertionPointToStart(guard.thenBlock());
  Type f32 = b.getF32Type();
  Value one = b.create<arith::ConstantOp>(
      loc, f32, b.getF32FloatAttr(1.0f));
  Value zero = b.create<arith::ConstantOp>(
      loc, f32, b.getF32FloatAttr(0.0f));
  Value tiny = b.create<arith::ConstantOp>(
      loc, f32, b.getF32FloatAttr(1.0e-30f));
  Value g = b.create<memref::LoadOp>(loc, gradient, ValueRange{gid});
  Value old = b.create<memref::LoadOp>(loc, oldMoment, ValueRange{gid});
  Value dy = b.create<memref::LoadOp>(loc, cotangent, ValueRange{gid});
  Value moment = b.create<arith::AddFOp>(
      loc, b.create<arith::MulFOp>(loc, beta2, old),
      b.create<arith::MulFOp>(
          loc, b.create<arith::SubFOp>(loc, one, beta2),
          b.create<arith::MulFOp>(loc, g, g)));
  Value safeMoment = b.create<arith::MaximumFOp>(loc, moment, eps);
  Value root = b.create<math::SqrtOp>(loc, safeMoment);
  Value denominator = b.create<arith::AddFOp>(loc, root, eps);
  Value direct = b.create<arith::DivFOp>(
      loc, b.create<arith::MulFOp>(
               loc, b.create<arith::SubFOp>(loc, zero, lr), dy),
      denominator);
  Value active = b.create<arith::CmpFOp>(
      loc, arith::CmpFPredicate::OGT, moment, eps);
  Value denominatorSquared =
      b.create<arith::MulFOp>(loc, denominator, denominator);
  Value derivative = b.create<arith::DivFOp>(
      loc,
      b.create<arith::MulFOp>(
          loc, lr, b.create<arith::MulFOp>(loc, dy, g)),
      b.create<arith::MulFOp>(
          loc, b.create<arith::ConstantOp>(
                   loc, f32, b.getF32FloatAttr(2.0f)),
          b.create<arith::MulFOp>(
              loc, b.create<arith::MaximumFOp>(loc, root, tiny),
              denominatorSquared)));
  Value dm = b.create<arith::SelectOp>(loc, active, derivative, zero);
  Value dg = b.create<arith::AddFOp>(
      loc, direct,
      b.create<arith::MulFOp>(
          loc,
          b.create<arith::MulFOp>(
              loc, b.create<arith::ConstantOp>(
                       loc, f32, b.getF32FloatAttr(2.0f)),
              b.create<arith::SubFOp>(loc, one, beta2)),
          b.create<arith::MulFOp>(loc, g, dm)));
  b.create<memref::StoreOp>(loc, dy, dparam, ValueRange{gid});
  b.create<memref::StoreOp>(loc, dg, dgrad, ValueRange{gid});
  b.create<memref::StoreOp>(
      loc, b.create<arith::MulFOp>(loc, beta2, dm), dmoment,
      ValueRange{gid});
  b.setInsertionPointToEnd(&f.getBody().front());
  b.create<gpu::ReturnOp>(loc);
}

// Emit the per-element update for `kind`. gid indexes the flat parameter array;
// p/g/m/v are loaded, the new param/state stored.
void emitOptBody(OpBuilder &b, Location loc, gpu::GPUFuncOp f, StringRef kind) {
  Type f32 = b.getF32Type();
  auto slt = arith::CmpIPredicate::slt;
  b.setInsertionPointToStart(&f.getBody().front());
  Value P = f.getArgument(0), G = f.getArgument(1), M = f.getArgument(2);
  Value V = f.getArgument(3), POUT = f.getArgument(4), MOUT = f.getArgument(5);
  Value VOUT = f.getArgument(6);
  Value N = f.getArgument(7);
  Value lr = f.getArgument(8), b1 = f.getArgument(9), b2 = f.getArgument(10);
  Value eps = f.getArgument(11), wd = f.getArgument(12);
  Value b1c = f.getArgument(13), b2c = f.getArgument(14);

  Value bid = b.create<gpu::BlockIdOp>(loc, gpu::Dimension::x);
  Value tid = b.create<gpu::ThreadIdOp>(loc, gpu::Dimension::x);
  Value cBD = b.create<arith::ConstantIndexOp>(loc, BD);
  Value gid = b.create<arith::AddIOp>(loc, b.create<arith::MulIOp>(loc, bid, cBD),
                                      tid);
  Value inb = b.create<arith::CmpIOp>(loc, slt, gid, N);
  auto guard = b.create<scf::IfOp>(loc, inb, /*withElse=*/false);
  b.setInsertionPointToStart(guard.thenBlock());

  Value one = b.create<arith::ConstantOp>(loc, f32, b.getF32FloatAttr(1.0f));
  Value zero = b.create<arith::ConstantOp>(loc, f32, b.getF32FloatAttr(0.0f));
  auto ld = [&](Value buf) {
    return b.create<memref::LoadOp>(loc, buf, ValueRange{gid}).getResult();
  };
  Value pi = ld(P), gi = ld(G);
  auto st = [&](Value buf, Value val) {
    b.create<memref::StoreOp>(loc, val, buf, ValueRange{gid});
  };

  if (kind == "sgd") {
    // p - lr*g
    st(POUT, b.create<arith::SubFOp>(loc, pi,
                                     b.create<arith::MulFOp>(loc, lr, gi)));
  } else if (kind == "momentum") {
    Value vv = b.create<arith::AddFOp>(loc, b.create<arith::MulFOp>(loc, b1, ld(V)),
                                       gi);
    st(VOUT, vv);
    st(POUT, b.create<arith::SubFOp>(loc, pi,
                                     b.create<arith::MulFOp>(loc, lr, vv)));
  } else if (kind == "nesterov") {
    // v = β1·v + g ; p -= lr·(g + β1·v)  (look-ahead momentum)
    Value vv = b.create<arith::AddFOp>(loc, b.create<arith::MulFOp>(loc, b1, ld(V)),
                                       gi);
    st(VOUT, vv);
    Value upd = b.create<arith::AddFOp>(loc, gi,
                                        b.create<arith::MulFOp>(loc, b1, vv));
    st(POUT, b.create<arith::SubFOp>(loc, pi,
                                     b.create<arith::MulFOp>(loc, lr, upd)));
  } else if (kind == "lion") {
    Value mi = ld(M);
    Value om1 = b.create<arith::SubFOp>(loc, one, b1);
    Value om2 = b.create<arith::SubFOp>(loc, one, b2);
    Value u = b.create<arith::AddFOp>(loc, b.create<arith::MulFOp>(loc, b1, mi),
                                      b.create<arith::MulFOp>(loc, om1, gi));
    st(MOUT, b.create<arith::AddFOp>(loc, b.create<arith::MulFOp>(loc, b2, mi),
                                     b.create<arith::MulFOp>(loc, om2, gi)));
    // sign(u) = (u>0) ? 1 : (u<0 ? -1 : 0)
    Value pos = b.create<arith::CmpFOp>(loc, arith::CmpFPredicate::OGT, u, zero);
    Value neg = b.create<arith::CmpFOp>(loc, arith::CmpFPredicate::OLT, u, zero);
    Value negone = b.create<arith::SubFOp>(loc, zero, one);
    Value sgn = b.create<arith::SelectOp>(
        loc, pos, one, b.create<arith::SelectOp>(loc, neg, negone, zero));
    // p *= (1-lr*wd)
    Value pd = b.create<arith::MulFOp>(
        loc, pi, b.create<arith::SubFOp>(
                     loc, one, b.create<arith::MulFOp>(loc, lr, wd)));
    st(POUT, b.create<arith::SubFOp>(loc, pd,
                                     b.create<arith::MulFOp>(loc, lr, sgn)));
  } else {  // adam / adamw
    Value mi = ld(M), vi = ld(V);
    Value om1 = b.create<arith::SubFOp>(loc, one, b1);
    Value om2 = b.create<arith::SubFOp>(loc, one, b2);
    Value mm = b.create<arith::AddFOp>(loc, b.create<arith::MulFOp>(loc, b1, mi),
                                       b.create<arith::MulFOp>(loc, om1, gi));
    Value vv = b.create<arith::AddFOp>(
        loc, b.create<arith::MulFOp>(loc, b2, vi),
        b.create<arith::MulFOp>(loc, om2, b.create<arith::MulFOp>(loc, gi, gi)));
    st(MOUT, mm);
    st(VOUT, vv);
    Value pbase = pi;
    if (kind == "adamw")
      pbase = b.create<arith::MulFOp>(
          loc, pi, b.create<arith::SubFOp>(
                       loc, one, b.create<arith::MulFOp>(loc, lr, wd)));
    Value denom = b.create<arith::AddFOp>(
        loc, b.create<math::SqrtOp>(loc, b.create<arith::DivFOp>(loc, vv, b2c)),
        eps);
    Value upd = b.create<arith::DivFOp>(
        loc, b.create<arith::DivFOp>(loc, mm, b1c), denom);
    st(POUT, b.create<arith::SubFOp>(loc, pbase,
                                     b.create<arith::MulFOp>(loc, lr, upd)));
  }
  b.setInsertionPointToEnd(&f.getBody().front());
  b.create<gpu::ReturnOp>(loc);
}

void emitSgdBackwardBody(OpBuilder &b, Location loc, gpu::GPUFuncOp f) {
  b.setInsertionPointToStart(&f.getBody().front());
  Value dy = f.getArgument(0), dParam = f.getArgument(1);
  Value dGrad = f.getArgument(2), n = f.getArgument(3);
  Value lr = f.getArgument(4);
  Value bid = b.create<gpu::BlockIdOp>(loc, gpu::Dimension::x);
  Value tid = b.create<gpu::ThreadIdOp>(loc, gpu::Dimension::x);
  Value block = b.create<arith::ConstantIndexOp>(loc, BD);
  Value gid = b.create<arith::AddIOp>(
      loc, b.create<arith::MulIOp>(loc, bid, block), tid);
  Value inBounds =
      b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::slt, gid, n);
  auto guard = b.create<scf::IfOp>(loc, inBounds, /*withElse=*/false);
  b.setInsertionPointToStart(guard.thenBlock());
  Value incoming = b.create<memref::LoadOp>(loc, dy, ValueRange{gid});
  b.create<memref::StoreOp>(loc, incoming, dParam, ValueRange{gid});
  Value zero = b.create<arith::ConstantOp>(
      loc, b.getF32Type(), b.getF32FloatAttr(0.0f));
  Value scaled = b.create<arith::MulFOp>(loc, lr, incoming);
  b.create<memref::StoreOp>(
      loc, b.create<arith::SubFOp>(loc, zero, scaled), dGrad,
      ValueRange{gid});
  b.setInsertionPointToEnd(&f.getBody().front());
  b.create<gpu::ReturnOp>(loc);
}

void emitMomentumBackwardBody(OpBuilder &b, Location loc, gpu::GPUFuncOp f,
                              bool nesterov) {
  b.setInsertionPointToStart(&f.getBody().front());
  Value dParamOut = f.getArgument(0), dVelocityOut = f.getArgument(1);
  Value dParam = f.getArgument(2), dGrad = f.getArgument(3);
  Value dVelocity = f.getArgument(4), n = f.getArgument(5);
  Value lr = f.getArgument(6), mu = f.getArgument(7);
  Value bid = b.create<gpu::BlockIdOp>(loc, gpu::Dimension::x);
  Value tid = b.create<gpu::ThreadIdOp>(loc, gpu::Dimension::x);
  Value block = b.create<arith::ConstantIndexOp>(loc, BD);
  Value gid = b.create<arith::AddIOp>(
      loc, b.create<arith::MulIOp>(loc, bid, block), tid);
  Value inBounds =
      b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::slt, gid, n);
  auto guard = b.create<scf::IfOp>(loc, inBounds, /*withElse=*/false);
  b.setInsertionPointToStart(guard.thenBlock());
  Value dp =
      b.create<memref::LoadOp>(loc, dParamOut, ValueRange{gid});
  Value dv =
      b.create<memref::LoadOp>(loc, dVelocityOut, ValueRange{gid});
  Value zero = b.create<arith::ConstantOp>(
      loc, b.getF32Type(), b.getF32FloatAttr(0.0f));
  Value one = b.create<arith::ConstantOp>(
      loc, b.getF32Type(), b.getF32FloatAttr(1.0f));
  Value fromParam = b.create<arith::MulFOp>(
      loc, b.create<arith::SubFOp>(loc, zero, lr), dp);
  Value gradFactor =
      nesterov ? b.create<arith::AddFOp>(loc, one, mu).getResult() : one;
  Value dg = b.create<arith::AddFOp>(
      loc, b.create<arith::MulFOp>(loc, gradFactor, fromParam), dv);
  Value velocityFactor = nesterov ? mu : one;
  Value velocityBase = b.create<arith::AddFOp>(
      loc, b.create<arith::MulFOp>(loc, velocityFactor, fromParam), dv);
  Value oldVelocityGrad =
      b.create<arith::MulFOp>(loc, mu, velocityBase);
  b.create<memref::StoreOp>(loc, dp, dParam, ValueRange{gid});
  b.create<memref::StoreOp>(loc, dg, dGrad, ValueRange{gid});
  b.create<memref::StoreOp>(
      loc, oldVelocityGrad, dVelocity, ValueRange{gid});
  b.setInsertionPointToEnd(&f.getBody().front());
  b.create<gpu::ReturnOp>(loc);
}

void emitLionBackwardBody(OpBuilder &b, Location loc, gpu::GPUFuncOp f) {
  // Lion treats sign(beta1*m + (1-beta1)*g) as stop-gradient.  Consequently
  // the parameter output contributes only through decoupled weight decay,
  // while the carried moment output contributes the affine beta2 update.
  b.setInsertionPointToStart(&f.getBody().front());
  Value dParamOut = f.getArgument(0), dMomentOut = f.getArgument(1);
  Value dParam = f.getArgument(2), dGrad = f.getArgument(3);
  Value dMoment = f.getArgument(4), n = f.getArgument(5);
  Value lr = f.getArgument(6), beta2 = f.getArgument(7);
  Value weightDecay = f.getArgument(8);
  Value bid = b.create<gpu::BlockIdOp>(loc, gpu::Dimension::x);
  Value tid = b.create<gpu::ThreadIdOp>(loc, gpu::Dimension::x);
  Value block = b.create<arith::ConstantIndexOp>(loc, BD);
  Value gid = b.create<arith::AddIOp>(
      loc, b.create<arith::MulIOp>(loc, bid, block), tid);
  Value inBounds =
      b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::slt, gid, n);
  auto guard = b.create<scf::IfOp>(loc, inBounds, /*withElse=*/false);
  b.setInsertionPointToStart(guard.thenBlock());
  Type f32 = b.getF32Type();
  Value one = b.create<arith::ConstantOp>(
      loc, f32, b.getF32FloatAttr(1.0f));
  Value dpOut =
      b.create<memref::LoadOp>(loc, dParamOut, ValueRange{gid});
  Value dmOut =
      b.create<memref::LoadOp>(loc, dMomentOut, ValueRange{gid});
  Value parameterFactor = b.create<arith::SubFOp>(
      loc, one, b.create<arith::MulFOp>(loc, lr, weightDecay));
  Value gradientFactor = b.create<arith::SubFOp>(loc, one, beta2);
  b.create<memref::StoreOp>(
      loc, b.create<arith::MulFOp>(loc, parameterFactor, dpOut), dParam,
      ValueRange{gid});
  b.create<memref::StoreOp>(
      loc, b.create<arith::MulFOp>(loc, gradientFactor, dmOut), dGrad,
      ValueRange{gid});
  b.create<memref::StoreOp>(
      loc, b.create<arith::MulFOp>(loc, beta2, dmOut), dMoment,
      ValueRange{gid});
  b.setInsertionPointToEnd(&f.getBody().front());
  b.create<gpu::ReturnOp>(loc);
}

void emitAdamBackwardBody(OpBuilder &b, Location loc, gpu::GPUFuncOp f,
                          bool adamw) {
  b.setInsertionPointToStart(&f.getBody().front());
  Value grad = f.getArgument(1), moment1 = f.getArgument(2);
  Value moment2 = f.getArgument(3), dParamOut = f.getArgument(4);
  Value dMoment1Out = f.getArgument(5), dMoment2Out = f.getArgument(6);
  Value dParam = f.getArgument(7), dGrad = f.getArgument(8);
  Value dMoment1 = f.getArgument(9), dMoment2 = f.getArgument(10);
  Value n = f.getArgument(11), lr = f.getArgument(12);
  Value beta1 = f.getArgument(13), beta2 = f.getArgument(14);
  Value eps = f.getArgument(15), weightDecay = f.getArgument(16);
  Value beta1Correction = f.getArgument(17);
  Value beta2Correction = f.getArgument(18);

  Value bid = b.create<gpu::BlockIdOp>(loc, gpu::Dimension::x);
  Value tid = b.create<gpu::ThreadIdOp>(loc, gpu::Dimension::x);
  Value block = b.create<arith::ConstantIndexOp>(loc, BD);
  Value gid = b.create<arith::AddIOp>(
      loc, b.create<arith::MulIOp>(loc, bid, block), tid);
  Value inBounds =
      b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::slt, gid, n);
  auto guard = b.create<scf::IfOp>(loc, inBounds, /*withElse=*/false);
  b.setInsertionPointToStart(guard.thenBlock());

  Type f32 = b.getF32Type();
  auto constant = [&](float value) {
    return b.create<arith::ConstantOp>(loc, f32, b.getF32FloatAttr(value))
        .getResult();
  };
  auto load = [&](Value buffer) {
    return b.create<memref::LoadOp>(loc, buffer, ValueRange{gid}).getResult();
  };
  auto store = [&](Value value, Value buffer) {
    b.create<memref::StoreOp>(loc, value, buffer, ValueRange{gid});
  };
  Value zero = constant(0.0f), one = constant(1.0f);
  Value g = load(grad), m = load(moment1), v = load(moment2);
  Value dpOut = load(dParamOut), dmOut = load(dMoment1Out);
  Value dvOut = load(dMoment2Out);
  Value oneMinusBeta1 = b.create<arith::SubFOp>(loc, one, beta1);
  Value oneMinusBeta2 = b.create<arith::SubFOp>(loc, one, beta2);
  Value mNew = b.create<arith::AddFOp>(
      loc, b.create<arith::MulFOp>(loc, beta1, m),
      b.create<arith::MulFOp>(loc, oneMinusBeta1, g));
  Value vNew = b.create<arith::AddFOp>(
      loc, b.create<arith::MulFOp>(loc, beta2, v),
      b.create<arith::MulFOp>(
          loc, oneMinusBeta2, b.create<arith::MulFOp>(loc, g, g)));
  Value normalizedV =
      b.create<arith::DivFOp>(loc, vNew, beta2Correction);
  Value root = b.create<math::SqrtOp>(loc, normalizedV);
  Value denom = b.create<arith::AddFOp>(loc, root, eps);
  Value negativeLr =
      b.create<arith::SubFOp>(loc, zero, lr);
  Value dMFromParam = b.create<arith::DivFOp>(
      loc,
      b.create<arith::MulFOp>(
          loc, dpOut,
          b.create<arith::DivFOp>(loc, negativeLr, beta1Correction)),
      denom);
  Value dMNew = b.create<arith::AddFOp>(loc, dmOut, dMFromParam);
  Value numerator =
      b.create<arith::DivFOp>(loc, mNew, beta1Correction);
  Value positive = b.create<arith::CmpFOp>(
      loc, arith::CmpFPredicate::OGT, normalizedV, zero);
  Value dRoot = b.create<arith::SelectOp>(
      loc, positive,
      b.create<arith::DivFOp>(
          loc, constant(0.5f),
          b.create<arith::MulFOp>(loc, beta2Correction, root)),
      zero);
  Value denomSquared = b.create<arith::MulFOp>(loc, denom, denom);
  Value dVFromParam = b.create<arith::MulFOp>(
      loc, dpOut,
      b.create<arith::MulFOp>(
          loc, lr,
          b.create<arith::DivFOp>(
              loc, b.create<arith::MulFOp>(loc, numerator, dRoot),
              denomSquared)));
  Value dVNew = b.create<arith::AddFOp>(loc, dvOut, dVFromParam);
  Value paramFactor = one;
  if (adamw)
    paramFactor = b.create<arith::SubFOp>(
        loc, one, b.create<arith::MulFOp>(loc, lr, weightDecay));
  Value gradFromMoment1 =
      b.create<arith::MulFOp>(loc, oneMinusBeta1, dMNew);
  Value gradFromMoment2 = b.create<arith::MulFOp>(
      loc, constant(2.0f),
      b.create<arith::MulFOp>(
          loc, oneMinusBeta2, b.create<arith::MulFOp>(loc, g, dVNew)));
  store(b.create<arith::MulFOp>(loc, dpOut, paramFactor), dParam);
  store(b.create<arith::AddFOp>(loc, gradFromMoment1, gradFromMoment2),
        dGrad);
  store(b.create<arith::MulFOp>(loc, beta1, dMNew), dMoment1);
  store(b.create<arith::MulFOp>(loc, beta2, dVNew), dMoment2);
  b.setInsertionPointToEnd(&f.getBody().front());
  b.create<gpu::ReturnOp>(loc);
}

struct GenerateROCMOptimizerKernelPass
    : PassWrapper<GenerateROCMOptimizerKernelPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(GenerateROCMOptimizerKernelPass)

  StringRef getArgument() const final { return "generate-rocm-optimizer-kernel"; }
  StringRef getDescription() const final {
    return "Expand a tessera_rocm.optimizer directive into a fused per-parameter "
           "optimizer-step gpu kernel (kind StrAttr selects the rule)";
  }
  void getDependentDialects(DialectRegistry &registry) const final {
    registry.insert<gpu::GPUDialect, scf::SCFDialect, arith::ArithDialect,
                    math::MathDialect, memref::MemRefDialect>();
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();
    SmallVector<Operation *> directives;
    module.walk([&](Operation *op) {
      if (op->getName().getStringRef() == "tessera_rocm.optimizer")
        directives.push_back(op);
    });
    for (Operation *op : directives) {
      auto nameAttr = op->getAttrOfType<StringAttr>("name");
      auto kindAttr = op->getAttrOfType<StringAttr>("kind");
      if (!nameAttr || !kindAttr) {
        op->emitError("tessera_rocm.optimizer missing name/kind");
        return signalPassFailure();
      }
      OpBuilder b(module.getBodyRegion());
      b.setInsertionPointToEnd(module.getBody());
      Location loc = op->getLoc();
      std::string kname = nameAttr.getValue().str();
      Type f32 = b.getF32Type();
      Type idxTy = b.getIndexType();
      auto memF32 = MemRefType::get({ShapedType::kDynamic}, f32);
      if (kindAttr.getValue() == "adafactor") {
        auto gpuMod = b.create<gpu::GPUModuleOp>(loc, kname + "_mod");
        b.setInsertionPointToStart(&gpuMod.getBodyRegion().front());
        auto momentTy = b.getFunctionType(
            {memF32, memF32, memF32, idxTy, idxTy, f32}, {});
        auto meanTy =
            b.getFunctionType({memF32, memF32, idxTy}, {});
        auto updateTy = b.getFunctionType(
            {memF32, memF32, memF32, memF32, memF32, memF32,
             idxTy, idxTy, f32, f32},
            {});
        auto fullTy = b.getFunctionType(
            {memF32, memF32, memF32, memF32, memF32, idxTy,
             f32, f32, f32},
            {});
        auto backwardMeanTy = b.getFunctionType(
            {memF32, memF32, memF32, memF32, memF32, memF32,
             idxTy, idxTy, f32, f32},
            {});
        auto backwardMomentTy = b.getFunctionType(
            {memF32, memF32, memF32, memF32, memF32, memF32, memF32,
             memF32, idxTy, idxTy, f32, f32, f32},
            {});
        auto backwardFinalizeTy = b.getFunctionType(
            {memF32, memF32, memF32, memF32, memF32, memF32, memF32,
             memF32, memF32, idxTy, idxTy, f32, f32, f32},
            {});
        auto fullBackwardTy = b.getFunctionType(
            {memF32, memF32, memF32, memF32, memF32, memF32, idxTy,
             f32, f32, f32},
            {});
        auto makeKernel = [&](StringRef suffix, FunctionType type) {
          auto fn =
              b.create<gpu::GPUFuncOp>(loc, kname + suffix.str(), type);
          fn->setAttr(gpu::GPUDialect::getKernelFuncAttrName(),
                      b.getUnitAttr());
          return fn;
        };
        auto row = makeKernel("_row", momentTy);
        auto col = makeKernel("_col", momentTy);
        auto mean = makeKernel("_mean", meanTy);
        auto update = makeKernel("_update", updateTy);
        auto full = makeKernel("_full", fullTy);
        auto backwardMean = makeKernel("_bwd_mean", backwardMeanTy);
        auto backwardRow = makeKernel("_bwd_row", backwardMomentTy);
        auto backwardCol = makeKernel("_bwd_col", backwardMomentTy);
        auto backwardFinalize =
            makeKernel("_bwd_finalize", backwardFinalizeTy);
        auto fullBackward = makeKernel("_full_bwd", fullBackwardTy);
        OpBuilder body(module.getContext());
        emitAdafactorMomentBody(body, loc, row, true);
        emitAdafactorMomentBody(body, loc, col, false);
        emitAdafactorMeanBody(body, loc, mean);
        emitAdafactorUpdateBody(body, loc, update);
        emitAdafactorFullBody(body, loc, full);
        emitAdafactorBackwardMeanBody(body, loc, backwardMean);
        emitAdafactorBackwardMomentBody(body, loc, backwardRow, true);
        emitAdafactorBackwardMomentBody(body, loc, backwardCol, false);
        emitAdafactorBackwardFinalizeBody(body, loc, backwardFinalize);
        emitAdafactorFullBackwardBody(body, loc, fullBackward);
        op->erase();
        continue;
      }
      bool backward = false;
      if (auto attr = op->getAttrOfType<BoolAttr>("backward"))
        backward = attr.getValue();
      bool momentumBackward =
          backward && (kindAttr.getValue() == "momentum" ||
                       kindAttr.getValue() == "nesterov");
      bool adamBackward =
          backward && (kindAttr.getValue() == "adam" ||
                       kindAttr.getValue() == "adamw");
      bool lionBackward = backward && kindAttr.getValue() == "lion";
      auto fnTy = backward
                      ? (adamBackward
                             ? b.getFunctionType(
                                   {memF32, memF32, memF32, memF32, memF32,
                                    memF32, memF32, memF32, memF32, memF32,
                                    memF32, idxTy, f32, f32, f32, f32, f32,
                                   f32, f32},
                                   {})
                             : lionBackward
                             ? b.getFunctionType(
                                   {memF32, memF32, memF32, memF32, memF32,
                                    idxTy, f32, f32, f32},
                                   {})
                             : momentumBackward
                             ? b.getFunctionType(
                                   {memF32, memF32, memF32, memF32, memF32,
                                    idxTy, f32, f32},
                                   {})
                             : b.getFunctionType(
                                   {memF32, memF32, memF32, idxTy, f32}, {}))
                      : b.getFunctionType(
                            {memF32, memF32, memF32, memF32, memF32, memF32,
                             memF32, idxTy, f32, f32, f32, f32, f32, f32,
                             f32},
                            {});
      auto gpuMod = b.create<gpu::GPUModuleOp>(loc, kname + "_mod");
      b.setInsertionPointToStart(&gpuMod.getBodyRegion().front());
      auto gpuFunc = b.create<gpu::GPUFuncOp>(loc, kname, fnTy);
      gpuFunc->setAttr(gpu::GPUDialect::getKernelFuncAttrName(), b.getUnitAttr());
      OpBuilder body(gpuFunc.getContext());
      if (backward) {
        if (adamBackward)
          emitAdamBackwardBody(body, loc, gpuFunc,
                               kindAttr.getValue() == "adamw");
        else if (lionBackward)
          emitLionBackwardBody(body, loc, gpuFunc);
        else if (momentumBackward)
          emitMomentumBackwardBody(body, loc, gpuFunc,
                                   kindAttr.getValue() == "nesterov");
        else if (kindAttr.getValue() == "sgd")
          emitSgdBackwardBody(body, loc, gpuFunc);
        else {
          op->emitError(
              "optimizer backward supports sgd, momentum, nesterov, adam, "
              "adamw, lion");
          return signalPassFailure();
        }
      } else {
        emitOptBody(body, loc, gpuFunc, kindAttr.getValue());
      }
      op->erase();
    }
  }
};

} // namespace

std::unique_ptr<mlir::Pass>
mlir::tessera_rocm::createGenerateROCMOptimizerKernelPass() {
  return std::make_unique<GenerateROCMOptimizerKernelPass>();
}
