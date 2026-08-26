//===- GenerateROCMSpectralBackwardKernel.cpp ----------------------------===//
// Native gfx1151 consumers for the bounded compound-spectral VJP contract.
//===----------------------------------------------------------------------===//

#include "TesseraROCM/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

#include <cmath>
#include <utility>

using namespace mlir;

namespace {

static constexpr int64_t kBlockSize = 256;

static Value globalIndex(OpBuilder &b, Location loc) {
  Value block = b.create<gpu::BlockIdOp>(loc, gpu::Dimension::x);
  Value thread = b.create<gpu::ThreadIdOp>(loc, gpu::Dimension::x);
  Value width = b.create<arith::ConstantIndexOp>(loc, kBlockSize);
  return b.create<arith::AddIOp>(
      loc, b.create<arith::MulIOp>(loc, block, width), thread);
}

static Value indexToF32(OpBuilder &b, Location loc, Value value) {
  Value integer = b.create<arith::IndexCastOp>(loc, b.getI64Type(), value);
  return b.create<arith::SIToFPOp>(loc, b.getF32Type(), integer);
}

static Value f32(OpBuilder &b, Location loc, double value) {
  return b.create<arith::ConstantOp>(loc, b.getF32Type(),
                                     b.getF32FloatAttr(value));
}

static Value loadAsF32(OpBuilder &b, Location loc, Value buffer, Value index) {
  Value value = b.create<memref::LoadOp>(loc, buffer, ValueRange{index});
  if (!value.getType().isF32())
    value = b.create<arith::ExtFOp>(loc, b.getF32Type(), value);
  return value;
}

static void storeFromF32(OpBuilder &b, Location loc, Value value, Value buffer,
                         Value index) {
  auto memref = cast<MemRefType>(buffer.getType());
  Type element = memref.getElementType();
  if (!element.isF32()) value = b.create<arith::TruncFOp>(loc, element, value);
  b.create<memref::StoreOp>(loc, value, buffer, ValueRange{index});
}

static Value logicalAnd(OpBuilder &b, Location loc, Value lhs, Value rhs) {
  return b.create<arith::AndIOp>(loc, lhs, rhs);
}

static Value physicalBatchAxisIndex(OpBuilder &b, Location loc,
                                    Value batchIndex, Value axisIndex,
                                    int64_t axisExtent, int64_t inner) {
  Value innerValue = b.create<arith::ConstantIndexOp>(loc, inner);
  Value outerIndex = b.create<arith::DivUIOp>(loc, batchIndex, innerValue);
  Value innerIndex = b.create<arith::RemUIOp>(loc, batchIndex, innerValue);
  Value outerAxis = b.create<arith::AddIOp>(
      loc, b.create<arith::MulIOp>(
               loc, outerIndex,
               b.create<arith::ConstantIndexOp>(loc, axisExtent)),
      axisIndex);
  return b.create<arith::AddIOp>(
      loc, b.create<arith::MulIOp>(loc, outerAxis, innerValue), innerIndex);
}

static Value physicalSpectrumIndex(OpBuilder &b, Location loc, Value row,
                                   Value frequency, int64_t frames,
                                   int64_t bins, int64_t inner) {
  Value frameValue = b.create<arith::ConstantIndexOp>(loc, frames);
  Value batchIndex = b.create<arith::DivUIOp>(loc, row, frameValue);
  Value frame = b.create<arith::RemUIOp>(loc, row, frameValue);
  Value frameBin = b.create<arith::AddIOp>(
      loc, b.create<arith::MulIOp>(
               loc, frame,
               b.create<arith::ConstantIndexOp>(loc, bins)),
      frequency);
  return physicalBatchAxisIndex(b, loc, batchIndex, frameBin, frames * bins,
                                inner);
}

// Direct evaluation is deliberate for the bounded n=16/n=18 package: it is
// independent of both the x86 packed-C2R implementation and the ROCm forward
// FFT, so the device packet is a genuine stored-bin adjoint proof.
static Value emitStoredBinAdjoint(OpBuilder &b, Location loc, Value dy,
                                  Value row, Value local, int64_t frames,
                                  int64_t bins, int64_t n, int64_t inner,
                                  float scale) {
  Value zero = f32(b, loc, 0.0);
  Value lb = b.create<arith::ConstantIndexOp>(loc, 0);
  Value ub = b.create<arith::ConstantIndexOp>(loc, bins);
  Value step = b.create<arith::ConstantIndexOp>(loc, 1);
  auto loop = b.create<scf::ForOp>(loc, lb, ub, step, ValueRange{zero});
  b.setInsertionPointToStart(loop.getBody());
  Value frequency = loop.getInductionVar();
  Value complexIndex = physicalSpectrumIndex(
      b, loc, row, frequency, frames, bins, inner);
  Value realIndex = b.create<arith::MulIOp>(
      loc, complexIndex, b.create<arith::ConstantIndexOp>(loc, 2));
  Value imagIndex = b.create<arith::AddIOp>(
      loc, realIndex, b.create<arith::ConstantIndexOp>(loc, 1));
  Value real = loadAsF32(b, loc, dy, realIndex);
  Value imag = loadAsF32(b, loc, dy, imagIndex);
  Value angle = b.create<arith::MulFOp>(
      loc, f32(b, loc, 2.0 * M_PI / double(n)),
      b.create<arith::MulFOp>(loc, indexToF32(b, loc, frequency),
                              indexToF32(b, loc, local)));
  Value cosine = b.create<math::CosOp>(loc, angle);
  Value sine = b.create<math::SinOp>(loc, angle);
  Value term = b.create<arith::SubFOp>(
      loc, b.create<arith::MulFOp>(loc, real, cosine),
      b.create<arith::MulFOp>(loc, imag, sine));
  Value sum = b.create<arith::AddFOp>(loc, loop.getRegionIterArgs()[0], term);
  b.create<scf::YieldOp>(loc, sum);
  b.setInsertionPointAfter(loop);
  return b.create<arith::MulFOp>(loc, loop.getResult(0), f32(b, loc, scale));
}

static Value emitInverseStoredFrame(OpBuilder &b, Location loc, Value spectrum,
                                    Value row, Value local, int64_t bins,
                                    int64_t n, int64_t frames,
                                    int64_t inner) {
  Value zero = f32(b, loc, 0.0);
  Value lb = b.create<arith::ConstantIndexOp>(loc, 0);
  Value ub = b.create<arith::ConstantIndexOp>(loc, bins);
  Value step = b.create<arith::ConstantIndexOp>(loc, 1);
  auto loop = b.create<scf::ForOp>(loc, lb, ub, step, ValueRange{zero});
  b.setInsertionPointToStart(loop.getBody());
  Value frequency = loop.getInductionVar();
  Value complexIndex = physicalSpectrumIndex(
      b, loc, row, frequency, frames, bins, inner);
  Value realIndex = b.create<arith::MulIOp>(
      loc, complexIndex, b.create<arith::ConstantIndexOp>(loc, 2));
  Value imagIndex = b.create<arith::AddIOp>(
      loc, realIndex, b.create<arith::ConstantIndexOp>(loc, 1));
  Value real = loadAsF32(b, loc, spectrum, realIndex);
  Value imag = loadAsF32(b, loc, spectrum, imagIndex);
  Value angle = b.create<arith::MulFOp>(
      loc, f32(b, loc, 2.0 * M_PI / double(n)),
      b.create<arith::MulFOp>(loc, indexToF32(b, loc, frequency),
                              indexToF32(b, loc, local)));
  Value term = b.create<arith::SubFOp>(
      loc, b.create<arith::MulFOp>(loc, real,
                                   b.create<math::CosOp>(loc, angle)),
      b.create<arith::MulFOp>(loc, imag,
                                   b.create<math::SinOp>(loc, angle)));
  Value isDC = b.create<arith::CmpIOp>(
      loc, arith::CmpIPredicate::eq, frequency, lb);
  Value isNyquist = b.create<arith::CmpIOp>(
      loc, arith::CmpIPredicate::eq, frequency,
      b.create<arith::ConstantIndexOp>(loc, n / 2));
  Value endpoint = b.create<arith::OrIOp>(loc, isDC, isNyquist);
  Value multiplicity = b.create<arith::SelectOp>(
      loc, endpoint, f32(b, loc, 1.0), f32(b, loc, 2.0));
  term = b.create<arith::MulFOp>(loc, term, multiplicity);
  Value sum = b.create<arith::AddFOp>(loc, loop.getRegionIterArgs()[0], term);
  b.create<scf::YieldOp>(loc, sum);
  b.setInsertionPointAfter(loop);
  return loop.getResult(0);
}

static std::pair<Value, Value>
emitOverlapStats(OpBuilder &b, Location loc, Value spectrum, Value window,
                 Value batchIndex, Value sample, int64_t frames, int64_t bins,
                 int64_t n, int64_t hop, int64_t inner) {
  Value zero = f32(b, loc, 0.0);
  Value lb = b.create<arith::ConstantIndexOp>(loc, 0);
  Value ub = b.create<arith::ConstantIndexOp>(loc, frames);
  Value step = b.create<arith::ConstantIndexOp>(loc, 1);
  auto loop = b.create<scf::ForOp>(loc, lb, ub, step,
                                   ValueRange{zero, zero});
  b.setInsertionPointToStart(loop.getBody());
  Value frame = loop.getInductionVar();
  Value local = b.create<arith::SubIOp>(
      loc, sample,
      b.create<arith::MulIOp>(
          loc, frame, b.create<arith::ConstantIndexOp>(loc, hop)));
  Value geZero = b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::sge,
                                         local, lb);
  Value ltN = b.create<arith::CmpIOp>(
      loc, arith::CmpIPredicate::slt, local,
      b.create<arith::ConstantIndexOp>(loc, n));
  auto contribution = b.create<scf::IfOp>(
      loc, TypeRange{b.getF32Type(), b.getF32Type()},
      logicalAnd(b, loc, geZero, ltN), true);
  b.setInsertionPointToStart(contribution.thenBlock());
  Value row = b.create<arith::AddIOp>(
      loc, b.create<arith::MulIOp>(
               loc, batchIndex,
               b.create<arith::ConstantIndexOp>(loc, frames)),
      frame);
  Value frameValue = emitInverseStoredFrame(b, loc, spectrum, row, local,
                                             bins, n, frames, inner);
  Value w = loadAsF32(b, loc, window, local);
  b.create<scf::YieldOp>(
      loc, ValueRange{b.create<arith::MulFOp>(loc, frameValue, w),
                      b.create<arith::MulFOp>(loc, w, w)});
  b.setInsertionPointToStart(contribution.elseBlock());
  b.create<scf::YieldOp>(loc, ValueRange{zero, zero});
  b.setInsertionPointAfter(contribution);
  Value numerator = b.create<arith::AddFOp>(
      loc, loop.getRegionIterArgs()[0], contribution.getResult(0));
  Value weight = b.create<arith::AddFOp>(
      loc, loop.getRegionIterArgs()[1], contribution.getResult(1));
  b.create<scf::YieldOp>(loc, ValueRange{numerator, weight});
  b.setInsertionPointAfter(loop);
  return {loop.getResult(0), loop.getResult(1)};
}

static void emitFilterBody(OpBuilder &b, Location loc, gpu::GPUFuncOp fn,
                           int64_t elements) {
  b.setInsertionPointToStart(&fn.getBody().front());
  Value dy = fn.getArgument(0), input = fn.getArgument(1);
  Value filter = fn.getArgument(2), dx = fn.getArgument(3);
  Value dfilter = fn.getArgument(4);
  Value gid = globalIndex(b, loc);
  Value limit = b.create<arith::ConstantIndexOp>(loc, elements);
  Value inBounds = b.create<arith::CmpIOp>(
      loc, arith::CmpIPredicate::slt, gid, limit);
  auto ifOp = b.create<scf::IfOp>(loc, inBounds, false);
  b.setInsertionPointToStart(ifOp.thenBlock());
  Value two = b.create<arith::ConstantIndexOp>(loc, 2);
  Value one = b.create<arith::ConstantIndexOp>(loc, 1);
  Value realIndex = b.create<arith::MulIOp>(loc, gid, two);
  Value imagIndex = b.create<arith::AddIOp>(loc, realIndex, one);
  auto load = [&](Value buffer, Value index) {
    return b.create<memref::LoadOp>(loc, buffer, ValueRange{index}).getResult();
  };
  Value dyr = load(dy, realIndex), dyi = load(dy, imagIndex);
  Value xr = load(input, realIndex), xi = load(input, imagIndex);
  Value fr = load(filter, realIndex), fi = load(filter, imagIndex);
  Value dxr = b.create<arith::AddFOp>(
      loc, b.create<arith::MulFOp>(loc, dyr, fr),
      b.create<arith::MulFOp>(loc, dyi, fi));
  Value dxi = b.create<arith::SubFOp>(
      loc, b.create<arith::MulFOp>(loc, dyi, fr),
      b.create<arith::MulFOp>(loc, dyr, fi));
  Value dfr = b.create<arith::AddFOp>(
      loc, b.create<arith::MulFOp>(loc, dyr, xr),
      b.create<arith::MulFOp>(loc, dyi, xi));
  Value dfi = b.create<arith::SubFOp>(
      loc, b.create<arith::MulFOp>(loc, dyi, xr),
      b.create<arith::MulFOp>(loc, dyr, xi));
  b.create<memref::StoreOp>(loc, dxr, dx, ValueRange{realIndex});
  b.create<memref::StoreOp>(loc, dxi, dx, ValueRange{imagIndex});
  b.create<memref::StoreOp>(loc, dfr, dfilter, ValueRange{realIndex});
  b.create<memref::StoreOp>(loc, dfi, dfilter, ValueRange{imagIndex});
  b.setInsertionPointAfter(ifOp);
  b.create<gpu::ReturnOp>(loc);
}

static void emitConvBody(OpBuilder &b, Location loc, gpu::GPUFuncOp fn,
                         int64_t batch, int64_t outputLength,
                         int64_t inputLength, int64_t kernelLength,
                         float scale) {
  b.setInsertionPointToStart(&fn.getBody().front());
  Value dy = fn.getArgument(0), input = fn.getArgument(1);
  Value kernel = fn.getArgument(2), dx = fn.getArgument(3);
  Value dkernel = fn.getArgument(4);
  Value gid = globalIndex(b, loc);
  Value rowWidth = b.create<arith::ConstantIndexOp>(
      loc, inputLength + kernelLength);
  Value limit = b.create<arith::ConstantIndexOp>(
      loc, batch * (inputLength + kernelLength));
  Value inBounds = b.create<arith::CmpIOp>(
      loc, arith::CmpIPredicate::slt, gid, limit);
  auto ifOp = b.create<scf::IfOp>(loc, inBounds, false);
  b.setInsertionPointToStart(ifOp.thenBlock());
  Value row = b.create<arith::DivUIOp>(loc, gid, rowWidth);
  Value local = b.create<arith::RemUIOp>(loc, gid, rowWidth);
  Value inputLimit = b.create<arith::ConstantIndexOp>(loc, inputLength);
  Value isInput = b.create<arith::CmpIOp>(
      loc, arith::CmpIPredicate::slt, local, inputLimit);
  Value zero = b.create<arith::ConstantOp>(
      loc, b.getF32Type(), b.getF32FloatAttr(0.0f));
  Value lb = b.create<arith::ConstantIndexOp>(loc, 0);
  Value kernelLimit = b.create<arith::ConstantIndexOp>(loc, kernelLength);
  Value step = b.create<arith::ConstantIndexOp>(loc, 1);
  auto branch = b.create<scf::IfOp>(loc, TypeRange{b.getF32Type()}, isInput,
                                    true);
  b.setInsertionPointToStart(branch.thenBlock());
  auto dxLoop = b.create<scf::ForOp>(loc, lb, kernelLimit, step,
                                     ValueRange{zero});
  b.setInsertionPointToStart(dxLoop.getBody());
  Value j = dxLoop.getInductionVar();
  Value dyBase = b.create<arith::MulIOp>(
      loc, row, b.create<arith::ConstantIndexOp>(loc, outputLength));
  Value dyIndex = b.create<arith::AddIOp>(
      loc, dyBase, b.create<arith::AddIOp>(loc, local, j));
  Value kernelBase = b.create<arith::MulIOp>(loc, row, kernelLimit);
  Value kernelIndex = b.create<arith::AddIOp>(loc, kernelBase, j);
  Value product = b.create<arith::MulFOp>(
      loc, b.create<memref::LoadOp>(loc, dy, ValueRange{dyIndex}),
      b.create<memref::LoadOp>(loc, kernel, ValueRange{kernelIndex}));
  Value sum = b.create<arith::AddFOp>(loc, dxLoop.getRegionIterArgs()[0],
                                      product);
  b.create<scf::YieldOp>(loc, sum);
  b.setInsertionPointToEnd(branch.thenBlock());
  b.create<scf::YieldOp>(loc, dxLoop.getResult(0));

  b.setInsertionPointToStart(branch.elseBlock());
  Value kernelLocal = b.create<arith::SubIOp>(loc, local, inputLimit);
  auto dkLoop = b.create<scf::ForOp>(loc, lb, inputLimit, step,
                                     ValueRange{zero});
  b.setInsertionPointToStart(dkLoop.getBody());
  Value i = dkLoop.getInductionVar();
  Value dyBase2 = b.create<arith::MulIOp>(
      loc, row, b.create<arith::ConstantIndexOp>(loc, outputLength));
  Value dyIndex2 = b.create<arith::AddIOp>(
      loc, dyBase2, b.create<arith::AddIOp>(loc, i, kernelLocal));
  Value inputBase = b.create<arith::MulIOp>(loc, row, inputLimit);
  Value inputIndex = b.create<arith::AddIOp>(loc, inputBase, i);
  Value product2 = b.create<arith::MulFOp>(
      loc, b.create<memref::LoadOp>(loc, dy, ValueRange{dyIndex2}),
      b.create<memref::LoadOp>(loc, input, ValueRange{inputIndex}));
  Value sum2 = b.create<arith::AddFOp>(loc, dkLoop.getRegionIterArgs()[0],
                                       product2);
  b.create<scf::YieldOp>(loc, sum2);
  b.setInsertionPointToEnd(branch.elseBlock());
  b.create<scf::YieldOp>(loc, dkLoop.getResult(0));

  b.setInsertionPointAfter(branch);
  Value scaleValue = b.create<arith::ConstantOp>(
      loc, b.getF32Type(), b.getF32FloatAttr(scale));
  Value result = b.create<arith::MulFOp>(loc, branch.getResult(0), scaleValue);
  auto storeBranch = b.create<scf::IfOp>(loc, isInput, false);
  b.setInsertionPointToStart(storeBranch.thenBlock());
  Value dxIndex = b.create<arith::AddIOp>(
      loc, b.create<arith::MulIOp>(loc, row, inputLimit), local);
  b.create<memref::StoreOp>(loc, result, dx, ValueRange{dxIndex});
  b.setInsertionPointAfter(storeBranch);
  Value isKernel = b.create<arith::XOrIOp>(
      loc, isInput,
      b.create<arith::ConstantIntOp>(loc, 1, 1));
  auto kernelStore = b.create<scf::IfOp>(loc, isKernel, false);
  b.setInsertionPointToStart(kernelStore.thenBlock());
  Value kernelLocal2 = b.create<arith::SubIOp>(loc, local, inputLimit);
  Value dkIndex = b.create<arith::AddIOp>(
      loc, b.create<arith::MulIOp>(loc, row,
                                   b.create<arith::ConstantIndexOp>(loc, kernelLength)),
      kernelLocal2);
  b.create<memref::StoreOp>(loc, result, dkernel, ValueRange{dkIndex});
  b.setInsertionPointAfter(kernelStore);
  b.setInsertionPointAfter(ifOp);
  b.create<gpu::ReturnOp>(loc);
}

static void emitSTFTBody(OpBuilder &b, Location loc, gpu::GPUFuncOp fn,
                         int64_t batch, int64_t samples, int64_t frames,
                         int64_t bins, int64_t n, int64_t hop, int64_t inner,
                         bool center, bool reflect, float scale) {
  b.setInsertionPointToStart(&fn.getBody().front());
  Value dy = fn.getArgument(0), input = fn.getArgument(1);
  Value window = fn.getArgument(2), dx = fn.getArgument(3);
  Value dwindow = fn.getArgument(4);
  Value gid = globalIndex(b, loc);
  Value dxCount = b.create<arith::ConstantIndexOp>(loc, batch * samples);
  Value total = b.create<arith::ConstantIndexOp>(loc, batch * samples + n);
  Value zero = f32(b, loc, 0.0);
  Value lb = b.create<arith::ConstantIndexOp>(loc, 0);
  Value step = b.create<arith::ConstantIndexOp>(loc, 1);
  Value sampleLimit = b.create<arith::ConstantIndexOp>(loc, samples);
  Value pad = b.create<arith::ConstantIndexOp>(loc, center ? n / 2 : 0);
  Value isDx = b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::slt,
                                       gid, dxCount);
  auto dxBranch = b.create<scf::IfOp>(loc, isDx, false);
  b.setInsertionPointToStart(dxBranch.thenBlock());
  Value batchIndex = b.create<arith::DivUIOp>(
      loc, gid, b.create<arith::ConstantIndexOp>(loc, samples));
  Value sample = b.create<arith::RemUIOp>(
      loc, gid, b.create<arith::ConstantIndexOp>(loc, samples));
  Value frameLimit = b.create<arith::ConstantIndexOp>(loc, frames);
  Value localLimit = b.create<arith::ConstantIndexOp>(loc, n);
  auto frameLoop = b.create<scf::ForOp>(loc, lb, frameLimit, step,
                                        ValueRange{zero});
  b.setInsertionPointToStart(frameLoop.getBody());
  Value frame = frameLoop.getInductionVar();
  auto localLoop = b.create<scf::ForOp>(loc, lb, localLimit, step,
                                        ValueRange{zero});
  b.setInsertionPointToStart(localLoop.getBody());
  Value local = localLoop.getInductionVar();
  Value paddedIndex = b.create<arith::AddIOp>(
      loc, b.create<arith::MulIOp>(
               loc, frame, b.create<arith::ConstantIndexOp>(loc, hop)),
      local);
  Value rawSource = b.create<arith::SubIOp>(loc, paddedIndex, pad);
  Value left = b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::slt,
                                       rawSource, lb);
  Value right = b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::sge,
                                        rawSource, sampleLimit);
  Value leftSource = b.create<arith::SubIOp>(loc, lb, rawSource);
  Value rightSource = b.create<arith::SubIOp>(
      loc, b.create<arith::ConstantIndexOp>(loc, 2 * samples - 2), rawSource);
  Value mapped = b.create<arith::SelectOp>(
      loc, left, leftSource,
      b.create<arith::SelectOp>(loc, right, rightSource, rawSource));
  Value inRange = logicalAnd(
      b, loc,
      b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::sge, rawSource, lb),
      b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::slt, rawSource,
                              sampleLimit));
  Value present = reflect ? b.create<arith::ConstantIntOp>(loc, 1, 1)
                          : inRange;
  Value matches = logicalAnd(
      b, loc, present,
      b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq, mapped, sample));
  auto active = b.create<scf::IfOp>(loc, TypeRange{b.getF32Type()},
                                    matches, true);
  b.setInsertionPointToStart(active.thenBlock());
  Value row = b.create<arith::AddIOp>(
      loc, b.create<arith::MulIOp>(
               loc, batchIndex,
               b.create<arith::ConstantIndexOp>(loc, frames)),
      frame);
  Value transformed = emitStoredBinAdjoint(b, loc, dy, row, local, frames,
                                            bins, n, inner, scale);
  Value contribution = b.create<arith::MulFOp>(
      loc, transformed, loadAsF32(b, loc, window, local));
  b.create<scf::YieldOp>(loc, contribution);
  b.setInsertionPointToStart(active.elseBlock());
  b.create<scf::YieldOp>(loc, zero);
  b.setInsertionPointAfter(active);
  Value localSum = b.create<arith::AddFOp>(
      loc, localLoop.getRegionIterArgs()[0], active.getResult(0));
  b.create<scf::YieldOp>(loc, localSum);
  b.setInsertionPointAfter(localLoop);
  Value frameSum = b.create<arith::AddFOp>(
      loc, frameLoop.getRegionIterArgs()[0], localLoop.getResult(0));
  b.create<scf::YieldOp>(loc, frameSum);
  b.setInsertionPointAfter(frameLoop);
  Value physicalDx = physicalBatchAxisIndex(
      b, loc, batchIndex, sample, samples, inner);
  storeFromF32(b, loc, frameLoop.getResult(0), dx, physicalDx);
  b.setInsertionPointAfter(dxBranch);

  Value atOrAfterDx = b.create<arith::CmpIOp>(
      loc, arith::CmpIPredicate::sge, gid, dxCount);
  Value beforeTotal = b.create<arith::CmpIOp>(
      loc, arith::CmpIPredicate::slt, gid, total);
  auto dwBranch = b.create<scf::IfOp>(
      loc, logicalAnd(b, loc, atOrAfterDx, beforeTotal), false);
  b.setInsertionPointToStart(dwBranch.thenBlock());
  Value windowIndex = b.create<arith::SubIOp>(loc, gid, dxCount);
  Value rowLimit = b.create<arith::ConstantIndexOp>(loc, batch * frames);
  auto rowLoop = b.create<scf::ForOp>(loc, lb, rowLimit, step,
                                      ValueRange{zero});
  b.setInsertionPointToStart(rowLoop.getBody());
  Value rowIndex = rowLoop.getInductionVar();
  Value rowBatch = b.create<arith::DivUIOp>(
      loc, rowIndex, b.create<arith::ConstantIndexOp>(loc, frames));
  Value rowFrame = b.create<arith::RemUIOp>(
      loc, rowIndex, b.create<arith::ConstantIndexOp>(loc, frames));
  Value source = b.create<arith::AddIOp>(
      loc, b.create<arith::MulIOp>(
               loc, rowFrame,
               b.create<arith::ConstantIndexOp>(loc, hop)),
      windowIndex);
  Value rawWindowSource = b.create<arith::SubIOp>(loc, source, pad);
  Value windowLeft = b.create<arith::CmpIOp>(
      loc, arith::CmpIPredicate::slt, rawWindowSource, lb);
  Value windowRight = b.create<arith::CmpIOp>(
      loc, arith::CmpIPredicate::sge, rawWindowSource, sampleLimit);
  Value mappedWindowSource = b.create<arith::SelectOp>(
      loc, windowLeft, b.create<arith::SubIOp>(loc, lb, rawWindowSource),
      b.create<arith::SelectOp>(
          loc, windowRight,
          b.create<arith::SubIOp>(
              loc, b.create<arith::ConstantIndexOp>(loc, 2 * samples - 2),
              rawWindowSource),
          rawWindowSource));
  Value windowInRange = logicalAnd(
      b, loc,
      b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::sge,
                              rawWindowSource, lb),
      b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::slt,
                              rawWindowSource, sampleLimit));
  Value windowPresent = reflect ? b.create<arith::ConstantIntOp>(loc, 1, 1)
                                : windowInRange;
  Value inputIndex = physicalBatchAxisIndex(
      b, loc, rowBatch, mappedWindowSource, samples, inner);
  Value transformedWindow = emitStoredBinAdjoint(
      b, loc, dy, rowIndex, windowIndex, frames, bins, n, inner, scale);
  auto sourceBranch = b.create<scf::IfOp>(
      loc, TypeRange{b.getF32Type()}, windowPresent, true);
  b.setInsertionPointToStart(sourceBranch.thenBlock());
  Value windowContribution = b.create<arith::MulFOp>(
      loc, loadAsF32(b, loc, input, inputIndex), transformedWindow);
  b.create<scf::YieldOp>(loc, ValueRange{windowContribution});
  b.setInsertionPointToStart(sourceBranch.elseBlock());
  b.create<scf::YieldOp>(loc, zero);
  b.setInsertionPointAfter(sourceBranch);
  Value windowSum = b.create<arith::AddFOp>(
      loc, rowLoop.getRegionIterArgs()[0], sourceBranch.getResult(0));
  b.create<scf::YieldOp>(loc, windowSum);
  b.setInsertionPointAfter(rowLoop);
  storeFromF32(b, loc, rowLoop.getResult(0), dwindow, windowIndex);
  b.setInsertionPointAfter(dwBranch);
  b.create<gpu::ReturnOp>(loc);
}

static void emitISTFTBody(OpBuilder &b, Location loc, gpu::GPUFuncOp fn,
                          int64_t batch, int64_t outputSamples,
                          int64_t rawSamples, int64_t frames, int64_t bins,
                          int64_t n, int64_t hop, int64_t inner, bool center,
                          float scale) {
  (void)rawSamples;
  b.setInsertionPointToStart(&fn.getBody().front());
  Value dy = fn.getArgument(0), spectrum = fn.getArgument(1);
  Value window = fn.getArgument(2), dspectrum = fn.getArgument(3);
  Value dwindow = fn.getArgument(4);
  Value gid = globalIndex(b, loc);
  int64_t spectrumScalars = 2 * batch * frames * bins;
  Value spectrumLimit =
      b.create<arith::ConstantIndexOp>(loc, spectrumScalars);
  Value total = b.create<arith::ConstantIndexOp>(loc, spectrumScalars + n);
  Value zeroIndex = b.create<arith::ConstantIndexOp>(loc, 0);
  Value oneIndex = b.create<arith::ConstantIndexOp>(loc, 1);
  Value trim = b.create<arith::ConstantIndexOp>(loc, center ? n / 2 : 0);
  Value outputLimit = b.create<arith::ConstantIndexOp>(loc, outputSamples);
  Value zero = f32(b, loc, 0.0);
  Value isSpectrum = b.create<arith::CmpIOp>(
      loc, arith::CmpIPredicate::slt, gid, spectrumLimit);
  auto spectrumBranch = b.create<scf::IfOp>(loc, isSpectrum, false);
  b.setInsertionPointToStart(spectrumBranch.thenBlock());
  Value complexIndex = b.create<arith::DivUIOp>(
      loc, gid, b.create<arith::ConstantIndexOp>(loc, 2));
  Value component = b.create<arith::RemUIOp>(
      loc, gid, b.create<arith::ConstantIndexOp>(loc, 2));
  Value bin = b.create<arith::RemUIOp>(
      loc, complexIndex, b.create<arith::ConstantIndexOp>(loc, bins));
  Value row = b.create<arith::DivUIOp>(
      loc, complexIndex, b.create<arith::ConstantIndexOp>(loc, bins));
  Value frame = b.create<arith::RemUIOp>(
      loc, row, b.create<arith::ConstantIndexOp>(loc, frames));
  Value batchIndex = b.create<arith::DivUIOp>(
      loc, row, b.create<arith::ConstantIndexOp>(loc, frames));
  Value nLimit = b.create<arith::ConstantIndexOp>(loc, n);
  auto nLoop = b.create<scf::ForOp>(loc, zeroIndex, nLimit, oneIndex,
                                    ValueRange{zero});
  b.setInsertionPointToStart(nLoop.getBody());
  Value local = nLoop.getInductionVar();
  Value sample = b.create<arith::AddIOp>(
      loc, b.create<arith::MulIOp>(
               loc, frame, b.create<arith::ConstantIndexOp>(loc, hop)),
      local);
  auto [numerator, weight] = emitOverlapStats(
      b, loc, spectrum, window, batchIndex, sample, frames, bins, n, hop,
      inner);
  Value positive = b.create<arith::CmpFOp>(
      loc, arith::CmpFPredicate::OGT, weight, f32(b, loc, 1.0e-12));
  Value denominator = b.create<arith::SelectOp>(
      loc, positive, weight, f32(b, loc, 1.0e-12));
  Value outputIndex = b.create<arith::SubIOp>(loc, sample, trim);
  Value inOutput = logicalAnd(
      b, loc,
      b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::sge, outputIndex,
                              zeroIndex),
      b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::slt, outputIndex,
                              outputLimit));
  Value dyIndex = physicalBatchAxisIndex(
      b, loc, batchIndex, outputIndex, outputSamples, inner);
  auto dyBranch = b.create<scf::IfOp>(loc, TypeRange{b.getF32Type()},
                                      inOutput, true);
  b.setInsertionPointToStart(dyBranch.thenBlock());
  b.create<scf::YieldOp>(loc, loadAsF32(b, loc, dy, dyIndex));
  b.setInsertionPointToStart(dyBranch.elseBlock());
  b.create<scf::YieldOp>(loc, zero);
  b.setInsertionPointAfter(dyBranch);
  Value draw = b.create<arith::DivFOp>(
      loc, b.create<arith::MulFOp>(loc, dyBranch.getResult(0),
                                   f32(b, loc, scale)),
      denominator);
  Value dframe = b.create<arith::MulFOp>(
      loc, draw, loadAsF32(b, loc, window, local));
  Value angle = b.create<arith::MulFOp>(
      loc, f32(b, loc, 2.0 * M_PI / double(n)),
      b.create<arith::MulFOp>(loc, indexToF32(b, loc, bin),
                              indexToF32(b, loc, local)));
  Value realComponent = b.create<arith::CmpIOp>(
      loc, arith::CmpIPredicate::eq, component, zeroIndex);
  Value coefficient = b.create<arith::SelectOp>(
      loc, realComponent, b.create<math::CosOp>(loc, angle),
      b.create<arith::NegFOp>(loc, b.create<math::SinOp>(loc, angle)));
  Value isDC = b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq, bin,
                                       zeroIndex);
  Value isNyquist = b.create<arith::CmpIOp>(
      loc, arith::CmpIPredicate::eq, bin,
      b.create<arith::ConstantIndexOp>(loc, n / 2));
  Value endpoint = b.create<arith::OrIOp>(loc, isDC, isNyquist);
  Value endpointImag = logicalAnd(
      b, loc, endpoint,
      b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq, component,
                              oneIndex));
  Value multiplicity = b.create<arith::SelectOp>(
      loc, endpoint, f32(b, loc, 1.0), f32(b, loc, 2.0));
  coefficient = b.create<arith::MulFOp>(loc, coefficient, multiplicity);
  coefficient = b.create<arith::SelectOp>(loc, endpointImag, zero,
                                           coefficient);
  Value contribution = b.create<arith::MulFOp>(loc, dframe, coefficient);
  Value accumulated = b.create<arith::AddFOp>(
      loc, nLoop.getRegionIterArgs()[0], contribution);
  b.create<scf::YieldOp>(loc, accumulated);
  b.setInsertionPointAfter(nLoop);
  Value physicalComplex = physicalSpectrumIndex(
      b, loc, row, bin, frames, bins, inner);
  Value physicalScalar = b.create<arith::AddIOp>(
      loc, b.create<arith::MulIOp>(
               loc, physicalComplex,
               b.create<arith::ConstantIndexOp>(loc, 2)),
      component);
  b.create<memref::StoreOp>(loc, nLoop.getResult(0), dspectrum,
                            ValueRange{physicalScalar});
  b.setInsertionPointAfter(spectrumBranch);

  Value afterSpectrum = b.create<arith::CmpIOp>(
      loc, arith::CmpIPredicate::sge, gid, spectrumLimit);
  Value beforeTotal = b.create<arith::CmpIOp>(
      loc, arith::CmpIPredicate::slt, gid, total);
  auto windowBranch = b.create<scf::IfOp>(
      loc, logicalAnd(b, loc, afterSpectrum, beforeTotal), false);
  b.setInsertionPointToStart(windowBranch.thenBlock());
  Value windowIndex = b.create<arith::SubIOp>(loc, gid, spectrumLimit);
  Value rowLimit = b.create<arith::ConstantIndexOp>(loc, batch * frames);
  auto rowLoop = b.create<scf::ForOp>(loc, zeroIndex, rowLimit, oneIndex,
                                      ValueRange{zero});
  b.setInsertionPointToStart(rowLoop.getBody());
  Value rowIndex = rowLoop.getInductionVar();
  Value rowFrame = b.create<arith::RemUIOp>(
      loc, rowIndex, b.create<arith::ConstantIndexOp>(loc, frames));
  Value rowBatch = b.create<arith::DivUIOp>(
      loc, rowIndex, b.create<arith::ConstantIndexOp>(loc, frames));
  Value rowSample = b.create<arith::AddIOp>(
      loc, b.create<arith::MulIOp>(
               loc, rowFrame,
               b.create<arith::ConstantIndexOp>(loc, hop)),
      windowIndex);
  auto [rowNumerator, rowWeight] = emitOverlapStats(
      b, loc, spectrum, window, rowBatch, rowSample, frames, bins, n, hop,
      inner);
  Value rowPositive = b.create<arith::CmpFOp>(
      loc, arith::CmpFPredicate::OGT, rowWeight, f32(b, loc, 1.0e-12));
  Value rowDenominator = b.create<arith::SelectOp>(
      loc, rowPositive, rowWeight, f32(b, loc, 1.0e-12));
  Value rowOutputIndex = b.create<arith::SubIOp>(loc, rowSample, trim);
  Value rowInOutput = logicalAnd(
      b, loc,
      b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::sge, rowOutputIndex,
                              zeroIndex),
      b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::slt, rowOutputIndex,
                              outputLimit));
  Value rowDyIndex = physicalBatchAxisIndex(
      b, loc, rowBatch, rowOutputIndex, outputSamples, inner);
  auto rowDyBranch = b.create<scf::IfOp>(loc, TypeRange{b.getF32Type()},
                                         rowInOutput, true);
  b.setInsertionPointToStart(rowDyBranch.thenBlock());
  b.create<scf::YieldOp>(loc, loadAsF32(b, loc, dy, rowDyIndex));
  b.setInsertionPointToStart(rowDyBranch.elseBlock());
  b.create<scf::YieldOp>(loc, zero);
  b.setInsertionPointAfter(rowDyBranch);
  Value scaledDy = b.create<arith::MulFOp>(
      loc, rowDyBranch.getResult(0), f32(b, loc, scale));
  Value rowDraw = b.create<arith::DivFOp>(loc, scaledDy, rowDenominator);
  Value denominatorSquared = b.create<arith::MulFOp>(
      loc, rowDenominator, rowDenominator);
  Value dweight = b.create<arith::NegFOp>(
      loc, b.create<arith::DivFOp>(
               loc, b.create<arith::MulFOp>(loc, scaledDy, rowNumerator),
               denominatorSquared));
  Value frameValue = emitInverseStoredFrame(
      b, loc, spectrum, rowIndex, windowIndex, bins, n, frames, inner);
  Value windowValue = loadAsF32(b, loc, window, windowIndex);
  Value windowContribution = b.create<arith::AddFOp>(
      loc, b.create<arith::MulFOp>(loc, rowDraw, frameValue),
      b.create<arith::MulFOp>(
          loc, f32(b, loc, 2.0),
          b.create<arith::MulFOp>(loc, dweight, windowValue)));
  Value windowAccumulated = b.create<arith::AddFOp>(
      loc, rowLoop.getRegionIterArgs()[0], windowContribution);
  b.create<scf::YieldOp>(loc, windowAccumulated);
  b.setInsertionPointAfter(rowLoop);
  storeFromF32(b, loc, rowLoop.getResult(0), dwindow, windowIndex);
  b.setInsertionPointAfter(windowBranch);
  b.create<gpu::ReturnOp>(loc);
}

struct GenerateROCMSpectralBackwardKernelPass
    : PassWrapper<GenerateROCMSpectralBackwardKernelPass,
                  OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(
      GenerateROCMSpectralBackwardKernelPass)
  StringRef getArgument() const final {
    return "generate-rocm-spectral-backward-kernel";
  }
  StringRef getDescription() const final {
    return "Generate bounded native gfx1151 compound spectral adjoints";
  }
  void getDependentDialects(DialectRegistry &registry) const final {
    registry.insert<gpu::GPUDialect, scf::SCFDialect, arith::ArithDialect,
                    math::MathDialect, memref::MemRefDialect>();
  }
  void runOnOperation() override {
    SmallVector<Operation *> directives;
    getOperation().walk([&](Operation *op) {
      if (op->getName().getStringRef() == "tessera_rocm.spectral_backward")
        directives.push_back(op);
    });
    for (Operation *op : directives) {
      auto name = op->getAttrOfType<StringAttr>("name");
      auto hash = op->getAttrOfType<StringAttr>("artifact_hash");
      auto kind = op->getAttrOfType<StringAttr>("kind");
      if (!name || !hash || hash.getValue().size() != 64 || !kind) {
        op->emitError("native ROCm spectral adjoint contract is incomplete");
        return signalPassFailure();
      }
      int64_t elements = op->getAttrOfType<IntegerAttr>("elements").getInt();
      int64_t batch = op->getAttrOfType<IntegerAttr>("batch").getInt();
      int64_t outer = op->getAttrOfType<IntegerAttr>("outer").getInt();
      int64_t inner = op->getAttrOfType<IntegerAttr>("inner").getInt();
      int64_t outputLength =
          op->getAttrOfType<IntegerAttr>("cotangent_length").getInt();
      int64_t rawLength =
          op->getAttrOfType<IntegerAttr>("raw_length").getInt();
      int64_t inputLength =
          op->getAttrOfType<IntegerAttr>("input_length").getInt();
      int64_t kernelLength =
          op->getAttrOfType<IntegerAttr>("parameter_length").getInt();
      int64_t logicalLength =
          op->getAttrOfType<IntegerAttr>("logical_length").getInt();
      int64_t frames = op->getAttrOfType<IntegerAttr>("frames").getInt();
      int64_t bins = op->getAttrOfType<IntegerAttr>("bins").getInt();
      float scale = op->getAttrOfType<FloatAttr>("normalization_scale")
                        .getValueAsDouble();
      auto storage = op->getAttrOfType<StringAttr>("storage");
      auto center = op->getAttrOfType<BoolAttr>("center");
      auto onesided = op->getAttrOfType<BoolAttr>("onesided");
      auto padMode = op->getAttrOfType<StringAttr>("pad_mode");
      auto numericPolicy =
          op->getAttrOfType<DictionaryAttr>("numeric_policy");
      auto numericStorage = numericPolicy
                                ? numericPolicy.getAs<StringAttr>("storage")
                                : StringAttr();
      auto numericAccum = numericPolicy
                              ? numericPolicy.getAs<StringAttr>("accum")
                              : StringAttr();
      bool filter = kind.getValue() == "tessera.spectral_filter";
      bool conv = kind.getValue() == "tessera.spectral_conv";
      bool stft = kind.getValue() == "tessera.stft";
      bool istft = kind.getValue() == "tessera.istft";
      StringRef expectedNumericStorage =
          storage && storage.getValue() == "f16"
              ? "fp16"
              : storage && storage.getValue() == "bf16" ? "bf16" : "fp32";
      if ((!filter && !conv && !stft && !istft) || !storage ||
          (storage.getValue() != "f32" && storage.getValue() != "f16" &&
           storage.getValue() != "bf16") ||
          !numericStorage || !numericAccum ||
          numericStorage.getValue() != expectedNumericStorage ||
          numericAccum.getValue() != "fp32" ||
          (filter && elements <= 0) ||
          (conv && (batch <= 0 || outputLength <= 0 || inputLength <= 0 ||
                    kernelLength <= 0)) ||
          !center || !padMode ||
          (padMode.getValue() != "constant" && padMode.getValue() != "reflect") ||
          (stft && (batch <= 0 || outer <= 0 || inner <= 0 ||
                    batch != outer * inner || inputLength <= 0 || kernelLength <= 0 ||
                    logicalLength < kernelLength || frames <= 0 || !onesided ||
                    bins != (onesided.getValue() ? logicalLength / 2 + 1
                                                 : logicalLength))) ||
          (istft && (batch <= 0 || outer <= 0 || inner <= 0 ||
                     batch != outer * inner || outputLength <= 0 || kernelLength <= 0 ||
                     logicalLength < kernelLength || rawLength <= 0 ||
                     frames <= 0 || !onesided ||
                     bins != (onesided.getValue() ? logicalLength / 2 + 1
                                                  : logicalLength)))) {
        op->emitError("compound spectral adjoint kind has no gfx1151 native package");
        return signalPassFailure();
      }
      OpBuilder b(getOperation().getBodyRegion());
      b.setInsertionPointToEnd(getOperation().getBody());
      Location loc = op->getLoc();
      auto gpuModule =
          b.create<gpu::GPUModuleOp>(loc, name.getValue().str() + "_mod");
      b.setInsertionPointToStart(&gpuModule.getBodyRegion().front());
      Type realElement = storage.getValue() == "f16"
                             ? Type(b.getF16Type())
                             : storage.getValue() == "bf16"
                                   ? Type(b.getBF16Type())
                                   : Type(b.getF32Type());
      Type f32Buffer =
          MemRefType::get({ShapedType::kDynamic}, b.getF32Type());
      Type realBuffer =
          MemRefType::get({ShapedType::kDynamic}, realElement);
      SmallVector<Type> buffers(5, f32Buffer);
      if (stft)
        buffers = {f32Buffer, realBuffer, realBuffer, realBuffer, realBuffer};
      else if (istft)
        buffers = {realBuffer, f32Buffer, realBuffer, f32Buffer, realBuffer};
      auto fn = b.create<gpu::GPUFuncOp>(
          loc, name.getValue(),
          b.getFunctionType(buffers, {}));
      fn->setAttr(gpu::GPUDialect::getKernelFuncAttrName(), b.getUnitAttr());
      OpBuilder body(fn.getContext());
      if (filter)
        emitFilterBody(body, loc, fn, elements);
      else if (conv)
        emitConvBody(body, loc, fn, batch, outputLength, inputLength,
                     kernelLength, scale);
      else if (stft)
        emitSTFTBody(body, loc, fn, batch, inputLength, frames, bins,
                     kernelLength, op->getAttrOfType<IntegerAttr>("hop").getInt(),
                     inner, center.getValue(),
                     padMode.getValue() == "reflect", scale);
      else
        emitISTFTBody(body, loc, fn, batch, outputLength, rawLength, frames,
                      bins, kernelLength,
                      op->getAttrOfType<IntegerAttr>("hop").getInt(),
                      inner, center.getValue(), scale);
      op->erase();
    }
  }
};

} // namespace

std::unique_ptr<mlir::Pass>
mlir::tessera_rocm::createGenerateROCMSpectralBackwardKernelPass() {
  return std::make_unique<GenerateROCMSpectralBackwardKernelPass>();
}
