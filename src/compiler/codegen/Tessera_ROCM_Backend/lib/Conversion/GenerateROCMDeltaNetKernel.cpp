//===- GenerateROCMDeltaNetKernel.cpp - gated/delta linear-attention scan -===//
//
// Expands a `tessera_rocm.deltanet` directive into a causal sequential-scan gpu
// kernel for the gated/delta linear-attention recurrence (the compiled lane for
// gated_deltanet / kimi_delta_attention / modified_delta_attention). One
// workgroup per (b,h); one thread per value-column e (blockDim = D_v). Thread e
// owns the state column Ŝ[:,e] (length D_qk) in LDS, so the per-step matvecs are
// independent per-thread d-loops with no cross-thread state hazard — the only
// barriers are the cooperative k/q load and the modified-delta ‖target‖
// reduction.
//
// Per timestep t (matching `_delta_attention_impl`, causal path):
//   target = V[t]
//   if erase:   target -= α_t · (Ŝᵀ k_t)         // read OLD state
//   if decay:   Ŝ *= α_t
//   delta = outer(k_t, target)  [/ (1 + ‖k_t‖·‖target‖) if modified]
//   Ŝ += β_t · delta
//   O[t] = Q_t @ Ŝ                                // read NEW state
//   if gate: O[t] *= sigmoid(gate[t])
//
// D_qk/D_v are compile-time (static loop bounds + LDS sizing); the flags
// (erase / modified / has_gate / has_beta / has_decay) are emitted as
// straight-line code. Compute is f32; storage is f16/bf16/f32. Validated vs the
// numpy reference on gfx1151. State starts at zero (state=None); no state out.
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

struct DeltaNetFlags {
  bool erase = false;
  bool modified = false;
  bool hasGate = false;
  bool hasBeta = false;
  bool hasDecay = false;
};

static void emitDeltaNetCheckpointBody(OpBuilder &b, Location loc,
                                       gpu::GPUFuncOp f, int64_t Dqk,
                                       int64_t Dv, DeltaNetFlags fl) {
  Type f32 = b.getF32Type();
  b.setInsertionPointToStart(&f.getBody().front());
  Value K = f.getArgument(0), V = f.getArgument(1);
  Value beta = f.getArgument(2), decay = f.getArgument(3);
  Value hist = f.getArgument(4), S = f.getArgument(5);
  Value bh = b.create<gpu::BlockIdOp>(loc, gpu::Dimension::x);
  auto ci = [&](int64_t v) { return b.create<arith::ConstantIndexOp>(loc, v); };
  auto cf = [&](double v) {
    return b.create<arith::ConstantOp>(loc, f32, b.getF32FloatAttr(v));
  };
  auto addi = [&](Value x, Value y) {
    return b.create<arith::AddIOp>(loc, x, y).getResult();
  };
  auto muli = [&](Value x, Value y) {
    return b.create<arith::MulIOp>(loc, x, y).getResult();
  };
  auto ld = [&](Value m, Value i) {
    return b.create<memref::LoadOp>(loc, m, ValueRange{i}).getResult();
  };
  Value c0 = ci(0), c1 = ci(1), cDk = ci(Dqk), cDv = ci(Dv);
  Value zero = cf(0.0), one = cf(1.0);
  Value stateSize = muli(cDk, cDv);
  Value histStride = muli(addi(S, c1), stateSize);
  Value histBase = muli(bh, histStride);
  Value qkBase = muli(muli(bh, S), cDk);
  Value vBase = muli(muli(bh, S), cDv);
  Value scalarBase = muli(bh, S);

  auto init = b.create<scf::ForOp>(loc, c0, stateSize, c1);
  {
    OpBuilder::InsertionGuard g(b);
    b.setInsertionPointToStart(init.getBody());
    b.create<memref::StoreOp>(loc, zero, hist,
                              ValueRange{addi(histBase, init.getInductionVar())});
  }
  auto tokens = b.create<scf::ForOp>(loc, c0, S, c1);
  {
    OpBuilder::InsertionGuard g(b);
    b.setInsertionPointToStart(tokens.getBody());
    Value t = tokens.getInductionVar();
    Value oldBase = addi(histBase, muli(t, stateSize));
    Value newBase = addi(oldBase, stateSize);
    Value kRow = addi(qkBase, muli(t, cDk));
    Value vRow = addi(vBase, muli(t, cDv));
    Value scalar = addi(scalarBase, t);
    Value a = fl.hasDecay ? ld(decay, scalar) : one;
    Value bw = fl.hasBeta ? ld(beta, scalar) : one;
    Value updateScale = one;
    if (fl.modified) {
      auto norm2 =
          b.create<scf::ForOp>(loc, c0, stateSize, c1, ValueRange{zero});
      {
        OpBuilder::InsertionGuard gn(b);
        b.setInsertionPointToStart(norm2.getBody());
        Value de = norm2.getInductionVar();
        Value d = b.create<arith::DivUIOp>(loc, de, cDv);
        Value e = b.create<arith::RemUIOp>(loc, de, cDv);
        Value target = ld(V, addi(vRow, e));
        if (fl.erase) {
          auto vh =
              b.create<scf::ForOp>(loc, c0, cDk, c1, ValueRange{zero});
          {
            OpBuilder::InsertionGuard gv(b);
            b.setInsertionPointToStart(vh.getBody());
            Value rd = vh.getInductionVar();
            Value term = b.create<arith::MulFOp>(
                loc, ld(K, addi(kRow, rd)),
                ld(hist, addi(oldBase, addi(muli(rd, cDv), e))));
            b.create<scf::YieldOp>(
                loc, ValueRange{b.create<arith::AddFOp>(
                         loc, vh.getRegionIterArgs()[0], term)});
          }
          target = b.create<arith::SubFOp>(
              loc, target,
              b.create<arith::MulFOp>(loc, a, vh.getResult(0)));
        }
        Value update =
            b.create<arith::MulFOp>(loc, ld(K, addi(kRow, d)), target);
        b.create<scf::YieldOp>(
            loc, ValueRange{b.create<arith::AddFOp>(
                     loc, norm2.getRegionIterArgs()[0],
                     b.create<arith::MulFOp>(loc, update, update))});
      }
      Value norm = b.create<math::SqrtOp>(loc, norm2.getResult(0));
      updateScale = b.create<arith::DivFOp>(
          loc, one, b.create<arith::AddFOp>(loc, one, norm));
    }
    auto dl = b.create<scf::ForOp>(loc, c0, cDk, c1);
    {
      OpBuilder::InsertionGuard g2(b);
      b.setInsertionPointToStart(dl.getBody());
      Value d = dl.getInductionVar();
      Value kd = ld(K, addi(kRow, d));
      auto el = b.create<scf::ForOp>(loc, c0, cDv, c1);
      OpBuilder::InsertionGuard g3(b);
      b.setInsertionPointToStart(el.getBody());
      Value e = el.getInductionVar();
      Value de = addi(muli(d, cDv), e);
      Value target = ld(V, addi(vRow, e));
      if (fl.erase) {
        auto vh = b.create<scf::ForOp>(loc, c0, cDk, c1, ValueRange{zero});
        {
          OpBuilder::InsertionGuard g4(b);
          b.setInsertionPointToStart(vh.getBody());
          Value rd = vh.getInductionVar();
          Value term = b.create<arith::MulFOp>(
              loc, ld(K, addi(kRow, rd)),
              ld(hist, addi(oldBase, addi(muli(rd, cDv), e))));
          b.create<scf::YieldOp>(
              loc, ValueRange{b.create<arith::AddFOp>(
                       loc, vh.getRegionIterArgs()[0], term)});
        }
        target = b.create<arith::SubFOp>(
            loc, target, b.create<arith::MulFOp>(loc, a, vh.getResult(0)));
      }
      Value old = ld(hist, addi(oldBase, de));
      Value update = b.create<arith::MulFOp>(
          loc, b.create<arith::MulFOp>(loc, bw, updateScale),
          b.create<arith::MulFOp>(loc, kd, target));
      b.create<memref::StoreOp>(
          loc, b.create<arith::AddFOp>(
                   loc, b.create<arith::MulFOp>(loc, a, old), update),
          hist, ValueRange{addi(newBase, de)});
    }
  }
  b.setInsertionPointToEnd(&f.getBody().front());
  b.create<gpu::ReturnOp>(loc);
}

static void emitDeltaNetChunkSummaryBody(OpBuilder &b, Location loc,
                                         gpu::GPUFuncOp f, int64_t Dqk,
                                         int64_t Dv, int64_t chunkSize,
                                         DeltaNetFlags fl) {
  Type f32 = b.getF32Type();
  b.setInsertionPointToStart(&f.getBody().front());
  Value K = f.getArgument(0), V = f.getArgument(1);
  Value beta = f.getArgument(2), decay = f.getArgument(3);
  Value scales = f.getArgument(4), updates = f.getArgument(5);
  Value S = f.getArgument(6), chunks = f.getArgument(7);
  auto ci = [&](int64_t v) { return b.create<arith::ConstantIndexOp>(loc, v); };
  auto cf = [&](double v) {
    return b.create<arith::ConstantOp>(loc, f32, b.getF32FloatAttr(v));
  };
  auto addi = [&](Value x, Value y) {
    return b.create<arith::AddIOp>(loc, x, y).getResult();
  };
  auto muli = [&](Value x, Value y) {
    return b.create<arith::MulIOp>(loc, x, y).getResult();
  };
  auto ld = [&](Value m, Value i) {
    return b.create<memref::LoadOp>(loc, m, ValueRange{i}).getResult();
  };
  Value c0 = ci(0), c1 = ci(1), cDv = ci(Dv), cDk = ci(Dqk);
  Value cChunk = ci(chunkSize), zero = cf(0.0), one = cf(1.0);
  Value stateSize = muli(cDk, cDv);
  Value task = b.create<gpu::BlockIdOp>(loc, gpu::Dimension::x);
  Value bh = b.create<arith::DivUIOp>(loc, task, chunks);
  Value chunk = b.create<arith::RemUIOp>(loc, task, chunks);
  Value begin = muli(chunk, cChunk);
  Value candidateEnd = addi(begin, cChunk);
  Value end = b.create<arith::SelectOp>(
      loc, b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::slt,
                                   candidateEnd, S),
      candidateEnd, S);
  Value updateBase = muli(task, stateSize);
  auto init = b.create<scf::ForOp>(loc, c0, stateSize, c1);
  {
    OpBuilder::InsertionGuard g(b);
    b.setInsertionPointToStart(init.getBody());
    b.create<memref::StoreOp>(
        loc, zero, updates,
        ValueRange{addi(updateBase, init.getInductionVar())});
  }
  auto tokens =
      b.create<scf::ForOp>(loc, begin, end, c1, ValueRange{one});
  {
    OpBuilder::InsertionGuard gt(b);
    b.setInsertionPointToStart(tokens.getBody());
    Value t = tokens.getInductionVar();
    Value qkRow = muli(addi(muli(bh, S), t), cDk);
    Value vRow = muli(addi(muli(bh, S), t), cDv);
    Value scalar = addi(muli(bh, S), t);
    Value a = fl.hasDecay ? ld(decay, scalar) : one;
    Value weight = fl.hasBeta ? ld(beta, scalar) : one;
    Value inv = one;
    if (fl.modified) {
      auto norm2 =
          b.create<scf::ForOp>(loc, c0, stateSize, c1, ValueRange{zero});
      {
        OpBuilder::InsertionGuard gn(b);
        b.setInsertionPointToStart(norm2.getBody());
        Value de = norm2.getInductionVar();
        Value d = b.create<arith::DivUIOp>(loc, de, cDv);
        Value e = b.create<arith::RemUIOp>(loc, de, cDv);
        Value update = b.create<arith::MulFOp>(
            loc, ld(K, addi(qkRow, d)), ld(V, addi(vRow, e)));
        b.create<scf::YieldOp>(
            loc, ValueRange{b.create<arith::AddFOp>(
                     loc, norm2.getRegionIterArgs()[0],
                     b.create<arith::MulFOp>(loc, update, update))});
      }
      inv = b.create<arith::DivFOp>(
          loc, one,
          b.create<arith::AddFOp>(
              loc, one, b.create<math::SqrtOp>(loc, norm2.getResult(0))));
    }
    auto state = b.create<scf::ForOp>(loc, c0, stateSize, c1);
    {
      OpBuilder::InsertionGuard gs(b);
      b.setInsertionPointToStart(state.getBody());
      Value de = state.getInductionVar();
      Value d = b.create<arith::DivUIOp>(loc, de, cDv);
      Value e = b.create<arith::RemUIOp>(loc, de, cDv);
      Value old = ld(updates, addi(updateBase, de));
      Value update = b.create<arith::MulFOp>(
          loc, b.create<arith::MulFOp>(loc, weight, inv),
          b.create<arith::MulFOp>(
              loc, ld(K, addi(qkRow, d)), ld(V, addi(vRow, e))));
      b.create<memref::StoreOp>(
          loc, b.create<arith::AddFOp>(
                   loc, b.create<arith::MulFOp>(loc, a, old), update),
          updates, ValueRange{addi(updateBase, de)});
    }
    b.create<scf::YieldOp>(
        loc, ValueRange{b.create<arith::MulFOp>(
                 loc, tokens.getRegionIterArgs()[0], a)});
  }
  b.create<memref::StoreOp>(loc, tokens.getResult(0), scales,
                            ValueRange{task});
  b.setInsertionPointToEnd(&f.getBody().front());
  b.create<gpu::ReturnOp>(loc);
}

static void emitDeltaNetChunkPrefixBody(OpBuilder &b, Location loc,
                                        gpu::GPUFuncOp f, int64_t Dqk,
                                        int64_t Dv, int64_t chunkSize) {
  b.setInsertionPointToStart(&f.getBody().front());
  Value scales = f.getArgument(0), updates = f.getArgument(1);
  Value hist = f.getArgument(2), S = f.getArgument(3);
  Value chunks = f.getArgument(4);
  auto ci = [&](int64_t v) { return b.create<arith::ConstantIndexOp>(loc, v); };
  auto addi = [&](Value x, Value y) {
    return b.create<arith::AddIOp>(loc, x, y).getResult();
  };
  auto muli = [&](Value x, Value y) {
    return b.create<arith::MulIOp>(loc, x, y).getResult();
  };
  auto ld = [&](Value m, Value i) {
    return b.create<memref::LoadOp>(loc, m, ValueRange{i}).getResult();
  };
  Value c0 = ci(0), c1 = ci(1), cDv = ci(Dv), cDk = ci(Dqk);
  Value cChunk = ci(chunkSize);
  Value stateSize = muli(cDk, cDv);
  Value bh = b.create<gpu::BlockIdOp>(loc, gpu::Dimension::x);
  Value histBase = muli(bh, muli(addi(S, c1), stateSize));
  auto init = b.create<scf::ForOp>(loc, c0, stateSize, c1);
  {
    OpBuilder::InsertionGuard g(b);
    b.setInsertionPointToStart(init.getBody());
    b.create<memref::StoreOp>(
        loc, b.create<arith::ConstantOp>(
                 loc, b.getF32Type(), b.getF32FloatAttr(0.0)),
        hist, ValueRange{addi(histBase, init.getInductionVar())});
  }
  auto prefix = b.create<scf::ForOp>(loc, c0, chunks, c1);
  {
    OpBuilder::InsertionGuard gp(b);
    b.setInsertionPointToStart(prefix.getBody());
    Value chunk = prefix.getInductionVar();
    Value task = addi(muli(bh, chunks), chunk);
    Value begin = muli(chunk, cChunk);
    Value candidateEnd = addi(begin, cChunk);
    Value end = b.create<arith::SelectOp>(
        loc, b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::slt,
                                     candidateEnd, S),
        candidateEnd, S);
    Value oldBase = addi(histBase, muli(begin, stateSize));
    Value nextBase = addi(histBase, muli(end, stateSize));
    Value updateBase = muli(task, stateSize);
    Value scale = ld(scales, task);
    auto state = b.create<scf::ForOp>(loc, c0, stateSize, c1);
    OpBuilder::InsertionGuard gs(b);
    b.setInsertionPointToStart(state.getBody());
    Value de = state.getInductionVar();
    b.create<memref::StoreOp>(
        loc, b.create<arith::AddFOp>(
                 loc, b.create<arith::MulFOp>(
                          loc, scale, ld(hist, addi(oldBase, de))),
                 ld(updates, addi(updateBase, de))),
        hist, ValueRange{addi(nextBase, de)});
  }
  b.setInsertionPointToEnd(&f.getBody().front());
  b.create<gpu::ReturnOp>(loc);
}

static void emitDeltaNetChunkFillBody(OpBuilder &b, Location loc,
                                      gpu::GPUFuncOp f, int64_t Dqk,
                                      int64_t Dv, int64_t chunkSize,
                                      DeltaNetFlags fl) {
  Type f32 = b.getF32Type();
  b.setInsertionPointToStart(&f.getBody().front());
  Value K = f.getArgument(0), V = f.getArgument(1);
  Value beta = f.getArgument(2), decay = f.getArgument(3);
  Value hist = f.getArgument(4), scratch = f.getArgument(5);
  Value S = f.getArgument(6), chunks = f.getArgument(7);
  auto ci = [&](int64_t v) { return b.create<arith::ConstantIndexOp>(loc, v); };
  auto cf = [&](double v) {
    return b.create<arith::ConstantOp>(loc, f32, b.getF32FloatAttr(v));
  };
  auto addi = [&](Value x, Value y) {
    return b.create<arith::AddIOp>(loc, x, y).getResult();
  };
  auto muli = [&](Value x, Value y) {
    return b.create<arith::MulIOp>(loc, x, y).getResult();
  };
  auto ld = [&](Value m, Value i) {
    return b.create<memref::LoadOp>(loc, m, ValueRange{i}).getResult();
  };
  Value c0 = ci(0), c1 = ci(1), cDv = ci(Dv), cDk = ci(Dqk);
  Value cChunk = ci(chunkSize), zero = cf(0.0), one = cf(1.0);
  Value stateSize = muli(cDk, cDv);
  Value task = b.create<gpu::BlockIdOp>(loc, gpu::Dimension::x);
  Value bh = b.create<arith::DivUIOp>(loc, task, chunks);
  Value chunk = b.create<arith::RemUIOp>(loc, task, chunks);
  Value begin = muli(chunk, cChunk);
  Value candidateEnd = addi(begin, cChunk);
  Value end = b.create<arith::SelectOp>(
      loc, b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::slt,
                                   candidateEnd, S),
      candidateEnd, S);
  Value histBase = muli(bh, muli(addi(S, c1), stateSize));
  Value boundary = addi(histBase, muli(begin, stateSize));
  Value scratchBase = muli(task, stateSize);
  auto copy = b.create<scf::ForOp>(loc, c0, stateSize, c1);
  {
    OpBuilder::InsertionGuard gc(b);
    b.setInsertionPointToStart(copy.getBody());
    Value de = copy.getInductionVar();
    b.create<memref::StoreOp>(loc, ld(hist, addi(boundary, de)), scratch,
                              ValueRange{addi(scratchBase, de)});
  }
  auto tokens = b.create<scf::ForOp>(loc, begin, end, c1);
  {
    OpBuilder::InsertionGuard gt(b);
    b.setInsertionPointToStart(tokens.getBody());
    Value t = tokens.getInductionVar();
    Value qkRow = muli(addi(muli(bh, S), t), cDk);
    Value vRow = muli(addi(muli(bh, S), t), cDv);
    Value scalar = addi(muli(bh, S), t);
    Value a = fl.hasDecay ? ld(decay, scalar) : one;
    Value weight = fl.hasBeta ? ld(beta, scalar) : one;
    Value inv = one;
    if (fl.modified) {
      auto norm2 =
          b.create<scf::ForOp>(loc, c0, stateSize, c1, ValueRange{zero});
      {
        OpBuilder::InsertionGuard gn(b);
        b.setInsertionPointToStart(norm2.getBody());
        Value de = norm2.getInductionVar();
        Value d = b.create<arith::DivUIOp>(loc, de, cDv);
        Value e = b.create<arith::RemUIOp>(loc, de, cDv);
        Value update = b.create<arith::MulFOp>(
            loc, ld(K, addi(qkRow, d)), ld(V, addi(vRow, e)));
        b.create<scf::YieldOp>(
            loc, ValueRange{b.create<arith::AddFOp>(
                     loc, norm2.getRegionIterArgs()[0],
                     b.create<arith::MulFOp>(loc, update, update))});
      }
      inv = b.create<arith::DivFOp>(
          loc, one,
          b.create<arith::AddFOp>(
              loc, one, b.create<math::SqrtOp>(loc, norm2.getResult(0))));
    }
    auto state = b.create<scf::ForOp>(loc, c0, stateSize, c1);
    {
      OpBuilder::InsertionGuard gs(b);
      b.setInsertionPointToStart(state.getBody());
      Value de = state.getInductionVar();
      Value d = b.create<arith::DivUIOp>(loc, de, cDv);
      Value e = b.create<arith::RemUIOp>(loc, de, cDv);
      Value old = ld(scratch, addi(scratchBase, de));
      Value next = b.create<arith::AddFOp>(
          loc, b.create<arith::MulFOp>(loc, a, old),
          b.create<arith::MulFOp>(
              loc, b.create<arith::MulFOp>(loc, weight, inv),
              b.create<arith::MulFOp>(
                  loc, ld(K, addi(qkRow, d)), ld(V, addi(vRow, e)))));
      b.create<memref::StoreOp>(loc, next, scratch,
                                ValueRange{addi(scratchBase, de)});
      Value notBoundary = b.create<arith::CmpIOp>(
          loc, arith::CmpIPredicate::slt, addi(t, c1), end);
      auto store = b.create<scf::IfOp>(loc, notBoundary, false);
      OpBuilder::InsertionGuard gi(b);
      b.setInsertionPointToStart(store.thenBlock());
      Value nextBase = addi(histBase, muli(addi(t, c1), stateSize));
      b.create<memref::StoreOp>(loc, next, hist,
                                ValueRange{addi(nextBase, de)});
    }
  }
  b.setInsertionPointToEnd(&f.getBody().front());
  b.create<gpu::ReturnOp>(loc);
}

static void emitDeltaNetReverseBody(OpBuilder &b, Location loc,
                                    gpu::GPUFuncOp f, int64_t Dqk, int64_t Dv,
                                    int64_t chunkSize, DeltaNetFlags fl) {
  Type f32 = b.getF32Type();
  b.setInsertionPointToStart(&f.getBody().front());
  Value Q = f.getArgument(0), K = f.getArgument(1), V = f.getArgument(2);
  Value gate = f.getArgument(3), beta = f.getArgument(4);
  Value decay = f.getArgument(5), DY = f.getArgument(6);
  Value hist = f.getArgument(7), DS = f.getArgument(8);
  Value DN = f.getArgument(9), DU = f.getArgument(10);
  Value DQ = f.getArgument(11), DK = f.getArgument(12);
  Value DV = f.getArgument(13), DG = f.getArgument(14);
  Value DB = f.getArgument(15), DA = f.getArgument(16);
  Value S = f.getArgument(17);
  Value bh = b.create<gpu::BlockIdOp>(loc, gpu::Dimension::x);
  auto ci = [&](int64_t v) { return b.create<arith::ConstantIndexOp>(loc, v); };
  auto cf = [&](double v) {
    return b.create<arith::ConstantOp>(loc, f32, b.getF32FloatAttr(v));
  };
  auto addi = [&](Value x, Value y) {
    return b.create<arith::AddIOp>(loc, x, y).getResult();
  };
  auto muli = [&](Value x, Value y) {
    return b.create<arith::MulIOp>(loc, x, y).getResult();
  };
  auto ld = [&](Value m, Value i) {
    return b.create<memref::LoadOp>(loc, m, ValueRange{i}).getResult();
  };
  auto addf = [&](Value x, Value y) {
    return b.create<arith::AddFOp>(loc, x, y).getResult();
  };
  auto mulf = [&](Value x, Value y) {
    return b.create<arith::MulFOp>(loc, x, y).getResult();
  };
  Value c0 = ci(0), c1 = ci(1), cDk = ci(Dqk), cDv = ci(Dv);
  Value cChunk = ci(chunkSize), zero = cf(0.0), one = cf(1.0);
  Value stateSize = muli(cDk, cDv);
  Value histBase = muli(bh, muli(addi(S, c1), stateSize));
  Value dsBase = muli(bh, stateSize);
  Value qkBase = muli(muli(bh, S), cDk);
  Value vBase = muli(muli(bh, S), cDv);
  Value scalarBase = muli(bh, S);
  auto init = b.create<scf::ForOp>(loc, c0, stateSize, c1);
  {
    OpBuilder::InsertionGuard g(b);
    b.setInsertionPointToStart(init.getBody());
    b.create<memref::StoreOp>(loc, zero, DS,
                              ValueRange{addi(dsBase, init.getInductionVar())});
  }
  Value chunks = b.create<arith::DivUIOp>(
      loc, addi(S, ci(chunkSize - 1)), cChunk);
  auto chunksRev = b.create<scf::ForOp>(loc, c0, chunks, c1);
  {
    OpBuilder::InsertionGuard gc(b);
    b.setInsertionPointToStart(chunksRev.getBody());
    Value chunk = b.create<arith::SubIOp>(
        loc, b.create<arith::SubIOp>(loc, chunks, c1),
        chunksRev.getInductionVar());
    Value start = muli(chunk, cChunk);
    Value candidateEnd = addi(start, cChunk);
    Value end = b.create<arith::SelectOp>(
        loc, b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::slt,
                                     candidateEnd, S),
        candidateEnd, S);
    Value span = b.create<arith::SubIOp>(loc, end, start);
    auto tokens = b.create<scf::ForOp>(loc, c0, span, c1);
    {
      OpBuilder::InsertionGuard gt(b);
      b.setInsertionPointToStart(tokens.getBody());
      Value t = b.create<arith::SubIOp>(
          loc, b.create<arith::SubIOp>(loc, end, c1),
          tokens.getInductionVar());
      Value oldBase = addi(histBase, muli(t, stateSize));
      Value newBase = addi(oldBase, stateSize);
      Value qkRow = addi(qkBase, muli(t, cDk));
      Value vRow = addi(vBase, muli(t, cDv));
      Value scalar = addi(scalarBase, t);
      Value a = fl.hasDecay ? ld(decay, scalar) : one;
      Value bw = fl.hasBeta ? ld(beta, scalar) : one;
      Value norm = zero, denom = one;
      if (fl.modified) {
        auto norm2 =
            b.create<scf::ForOp>(loc, c0, stateSize, c1, ValueRange{zero});
        {
          OpBuilder::InsertionGuard gn(b);
          b.setInsertionPointToStart(norm2.getBody());
          Value de = norm2.getInductionVar();
          Value d = b.create<arith::DivUIOp>(loc, de, cDv);
          Value e = b.create<arith::RemUIOp>(loc, de, cDv);
          Value target = ld(V, addi(vRow, e));
          if (fl.erase) {
            auto vh =
                b.create<scf::ForOp>(loc, c0, cDk, c1, ValueRange{zero});
            {
              OpBuilder::InsertionGuard gv(b);
              b.setInsertionPointToStart(vh.getBody());
              Value rd = vh.getInductionVar();
              Value term = mulf(
                  ld(K, addi(qkRow, rd)),
                  ld(hist, addi(oldBase, addi(muli(rd, cDv), e))));
              b.create<scf::YieldOp>(
                  loc, ValueRange{addf(vh.getRegionIterArgs()[0], term)});
            }
            target = b.create<arith::SubFOp>(
                loc, target, mulf(a, vh.getResult(0)));
          }
          Value update = mulf(ld(K, addi(qkRow, d)), target);
          b.create<scf::YieldOp>(
              loc, ValueRange{addf(norm2.getRegionIterArgs()[0],
                                   mulf(update, update))});
        }
        norm = b.create<math::SqrtOp>(loc, norm2.getResult(0));
        denom = addf(one, norm);
      }
      // Inject output cotangents into the carried-state adjoint.
      auto el = b.create<scf::ForOp>(loc, c0, cDv, c1);
      {
        OpBuilder::InsertionGuard ge(b);
        b.setInsertionPointToStart(el.getBody());
        Value e = el.getInductionVar();
        Value ve = addi(vRow, e);
        auto raw = b.create<scf::ForOp>(loc, c0, cDk, c1, ValueRange{zero});
        {
          OpBuilder::InsertionGuard gr(b);
          b.setInsertionPointToStart(raw.getBody());
          Value d = raw.getInductionVar();
          Value term = mulf(ld(Q, addi(qkRow, d)),
                            ld(hist, addi(newBase, addi(muli(d, cDv), e))));
          b.create<scf::YieldOp>(
              loc, ValueRange{addf(raw.getRegionIterArgs()[0], term)});
        }
        Value dy = ld(DY, ve);
        if (fl.hasGate) {
          Value gv = ld(gate, ve);
          Value sig = b.create<arith::DivFOp>(
              loc, one, addf(one, b.create<math::ExpOp>(
                                      loc, b.create<arith::NegFOp>(loc, gv))));
          b.create<memref::StoreOp>(
              loc, mulf(mulf(dy, raw.getResult(0)),
                        mulf(sig, b.create<arith::SubFOp>(loc, one, sig))),
              DG, ValueRange{ve});
          dy = mulf(dy, sig);
        }
        auto dl = b.create<scf::ForOp>(loc, c0, cDk, c1);
        {
          OpBuilder::InsertionGuard gd(b);
          b.setInsertionPointToStart(dl.getBody());
          Value d = dl.getInductionVar();
          Value de = addi(muli(d, cDv), e);
          Value dsi = addi(dsBase, de);
          Value dnew = addf(ld(DS, dsi), mulf(ld(Q, addi(qkRow, d)), dy));
          b.create<memref::StoreOp>(loc, dnew, DS, ValueRange{dsi});
          b.create<memref::StoreOp>(loc, dnew, DN, ValueRange{dsi});
        }
      }

      // Differentiate delta = update / (1 + ||update||). DU owns d(update).
      // The safe-norm branch intentionally mirrors the shared analytic VJP.
      Value projection = zero;
      if (fl.modified) {
        auto projectionLoop =
            b.create<scf::ForOp>(loc, c0, stateSize, c1, ValueRange{zero});
        {
          OpBuilder::InsertionGuard gp(b);
          b.setInsertionPointToStart(projectionLoop.getBody());
          Value de = projectionLoop.getInductionVar();
          Value d = b.create<arith::DivUIOp>(loc, de, cDv);
          Value e = b.create<arith::RemUIOp>(loc, de, cDv);
          Value target = ld(V, addi(vRow, e));
          if (fl.erase) {
            auto vh =
                b.create<scf::ForOp>(loc, c0, cDk, c1, ValueRange{zero});
            {
              OpBuilder::InsertionGuard gv(b);
              b.setInsertionPointToStart(vh.getBody());
              Value rd = vh.getInductionVar();
              Value term = mulf(
                  ld(K, addi(qkRow, rd)),
                  ld(hist, addi(oldBase, addi(muli(rd, cDv), e))));
              b.create<scf::YieldOp>(
                  loc, ValueRange{addf(vh.getRegionIterArgs()[0], term)});
            }
            target = b.create<arith::SubFOp>(
                loc, target, mulf(a, vh.getResult(0)));
          }
          Value update = mulf(ld(K, addi(qkRow, d)), target);
          Value ddelta = mulf(bw, ld(DN, addi(dsBase, de)));
          b.create<scf::YieldOp>(
              loc, ValueRange{addf(projectionLoop.getRegionIterArgs()[0],
                                   mulf(ddelta, update))});
        }
        projection = projectionLoop.getResult(0);
      }
      auto updateGrad =
          b.create<scf::ForOp>(loc, c0, stateSize, c1);
      {
        OpBuilder::InsertionGuard gu(b);
        b.setInsertionPointToStart(updateGrad.getBody());
        Value de = updateGrad.getInductionVar();
        Value d = b.create<arith::DivUIOp>(loc, de, cDv);
        Value e = b.create<arith::RemUIOp>(loc, de, cDv);
        Value dupdate = mulf(bw, ld(DN, addi(dsBase, de)));
        if (fl.modified) {
          Value target = ld(V, addi(vRow, e));
          if (fl.erase) {
            auto vh =
                b.create<scf::ForOp>(loc, c0, cDk, c1, ValueRange{zero});
            {
              OpBuilder::InsertionGuard gv(b);
              b.setInsertionPointToStart(vh.getBody());
              Value rd = vh.getInductionVar();
              Value term = mulf(
                  ld(K, addi(qkRow, rd)),
                  ld(hist, addi(oldBase, addi(muli(rd, cDv), e))));
              b.create<scf::YieldOp>(
                  loc, ValueRange{addf(vh.getRegionIterArgs()[0], term)});
            }
            target = b.create<arith::SubFOp>(
                loc, target, mulf(a, vh.getResult(0)));
          }
          Value update = mulf(ld(K, addi(qkRow, d)), target);
          // Divide by `norm`, not by max(norm, 1). See the derivation in
          // `avx512_deltanet_f32.cpp`: clamping understated the bound's
          // correction by a factor of n for every n < 1, which is the ordinary
          // case with L2-normalised keys. The numerator is O(n^2), so the
          // quotient vanishes with n and the `norm > 0` select below covers
          // n == 0 exactly.
          Value correction = b.create<arith::DivFOp>(
              loc, mulf(update, projection),
              mulf(norm, mulf(denom, denom)));
          correction = b.create<arith::SelectOp>(
              loc, b.create<arith::CmpFOp>(
                       loc, arith::CmpFPredicate::OGT, norm, zero),
              correction, zero);
          dupdate =
              b.create<arith::SubFOp>(
                  loc, b.create<arith::DivFOp>(loc, dupdate, denom),
                  correction);
        }
        b.create<memref::StoreOp>(loc, dupdate, DU,
                                  ValueRange{addi(dsBase, de)});
      }

      // dtarget[e] = sum_d beta * dstate[d,e] * k[d]. DV is both
      // the unique-owner output and token-local reverse scratch.
      auto targetLoop = b.create<scf::ForOp>(loc, c0, cDv, c1);
      {
        OpBuilder::InsertionGuard ge(b);
        b.setInsertionPointToStart(targetLoop.getBody());
        Value e = targetLoop.getInductionVar();
        auto reduction =
            b.create<scf::ForOp>(loc, c0, cDk, c1, ValueRange{zero});
        {
          OpBuilder::InsertionGuard gd(b);
          b.setInsertionPointToStart(reduction.getBody());
          Value d = reduction.getInductionVar();
          Value term = mulf(
              ld(DU, addi(dsBase, addi(muli(d, cDv), e))),
              ld(K, addi(qkRow, d)));
          b.create<scf::YieldOp>(
              loc, ValueRange{addf(reduction.getRegionIterArgs()[0], term)});
        }
        b.create<memref::StoreOp>(loc, reduction.getResult(0), DV,
                                  ValueRange{addi(vRow, e)});
      }

      // Unique d owner produces dQ/dK and advances the state adjoint.
      auto dl = b.create<scf::ForOp>(loc, c0, cDk, c1);
      {
        OpBuilder::InsertionGuard gd(b);
        b.setInsertionPointToStart(dl.getBody());
        Value d = dl.getInductionVar();
        Value kd = ld(K, addi(qkRow, d));
        Value dq = zero, dk = zero;
        auto el2 = b.create<scf::ForOp>(
            loc, c0, cDv, c1, ValueRange{dq, dk});
        {
          OpBuilder::InsertionGuard ge(b);
          b.setInsertionPointToStart(el2.getBody());
          Value e = el2.getInductionVar();
          Value de = addi(muli(d, cDv), e);
          Value dnew = ld(DN, addi(dsBase, de));
          Value old = ld(hist, addi(oldBase, de));
          Value target = ld(V, addi(vRow, e));
          Value vhat = zero;
          if (fl.erase) {
            auto vh = b.create<scf::ForOp>(
                loc, c0, cDk, c1, ValueRange{zero});
            {
              OpBuilder::InsertionGuard gv(b);
              b.setInsertionPointToStart(vh.getBody());
              Value rd = vh.getInductionVar();
              Value term = mulf(
                  ld(K, addi(qkRow, rd)),
                  ld(hist, addi(oldBase, addi(muli(rd, cDv), e))));
              b.create<scf::YieldOp>(
                  loc, ValueRange{addf(vh.getRegionIterArgs()[0], term)});
            }
            vhat = vh.getResult(0);
            target = b.create<arith::SubFOp>(loc, target, mulf(a, vhat));
          }
          Value dtarget = ld(DV, addi(vRow, e));
          Value ndk = addf(
              el2.getRegionIterArgs()[1],
              mulf(ld(DU, addi(dsBase, de)), target));
          Value prev = mulf(a, dnew);
          if (fl.erase) {
            Value dvhat = b.create<arith::NegFOp>(loc, mulf(a, dtarget));
            ndk = addf(ndk, mulf(old, dvhat));
            prev = addf(prev, mulf(kd, dvhat));
          }
          b.create<memref::StoreOp>(loc, prev, DS,
                                    ValueRange{addi(dsBase, de)});
          Value dy = ld(DY, addi(vRow, e));
          if (fl.hasGate) {
            Value gv = ld(gate, addi(vRow, e));
            Value sig = b.create<arith::DivFOp>(
                loc, one,
                addf(one, b.create<math::ExpOp>(
                              loc, b.create<arith::NegFOp>(loc, gv))));
            dy = mulf(dy, sig);
          }
          b.create<scf::YieldOp>(
              loc, ValueRange{
                       addf(el2.getRegionIterArgs()[0],
                            mulf(ld(hist, addi(newBase, de)), dy)),
                       ndk});
        }
        b.create<memref::StoreOp>(loc, el2.getResult(0), DQ,
                                  ValueRange{addi(qkRow, d)});
        b.create<memref::StoreOp>(loc, el2.getResult(1), DK,
                                  ValueRange{addi(qkRow, d)});
      }

      // Scalar beta/decay gradients have one (b,h,t) owner.
      auto scalarReduction = b.create<scf::ForOp>(
          loc, c0, stateSize, c1, ValueRange{zero, zero});
      {
        OpBuilder::InsertionGuard gs(b);
        b.setInsertionPointToStart(scalarReduction.getBody());
        Value de = scalarReduction.getInductionVar();
        Value d = b.create<arith::DivUIOp>(loc, de, cDv);
        Value e = b.create<arith::RemUIOp>(loc, de, cDv);
        Value old = ld(hist, addi(oldBase, de));
        Value dnew = ld(DN, addi(dsBase, de));
        Value dtarget = ld(DV, addi(vRow, e));
        Value target = ld(V, addi(vRow, e));
        Value vhat = zero;
        if (fl.erase) {
          auto vh =
              b.create<scf::ForOp>(loc, c0, cDk, c1, ValueRange{zero});
          {
            OpBuilder::InsertionGuard gv(b);
            b.setInsertionPointToStart(vh.getBody());
            Value rd = vh.getInductionVar();
            Value term = mulf(
                ld(K, addi(qkRow, rd)),
                ld(hist, addi(oldBase, addi(muli(rd, cDv), e))));
            b.create<scf::YieldOp>(
                loc, ValueRange{addf(vh.getRegionIterArgs()[0], term)});
          }
          vhat = vh.getResult(0);
          target = b.create<arith::SubFOp>(loc, target, mulf(a, vhat));
        }
        Value nextDb = addf(
            scalarReduction.getRegionIterArgs()[0],
            b.create<arith::DivFOp>(
                loc,
                mulf(dnew, mulf(ld(K, addi(qkRow, d)), target)),
                denom));
        Value nextDa = addf(scalarReduction.getRegionIterArgs()[1],
                            mulf(dnew, old));
        b.create<scf::YieldOp>(loc, ValueRange{nextDb, nextDa});
      }
      Value da = scalarReduction.getResult(1);
      if (fl.erase) {
        auto eraseReduction =
            b.create<scf::ForOp>(loc, c0, cDv, c1, ValueRange{zero});
        {
          OpBuilder::InsertionGuard ge(b);
          b.setInsertionPointToStart(eraseReduction.getBody());
          Value e = eraseReduction.getInductionVar();
          auto vh =
              b.create<scf::ForOp>(loc, c0, cDk, c1, ValueRange{zero});
          {
            OpBuilder::InsertionGuard gv(b);
            b.setInsertionPointToStart(vh.getBody());
            Value rd = vh.getInductionVar();
            Value term = mulf(
                ld(K, addi(qkRow, rd)),
                ld(hist, addi(oldBase, addi(muli(rd, cDv), e))));
            b.create<scf::YieldOp>(
                loc, ValueRange{addf(vh.getRegionIterArgs()[0], term)});
          }
          b.create<scf::YieldOp>(
              loc, ValueRange{addf(
                       eraseReduction.getRegionIterArgs()[0],
                       mulf(ld(DV, addi(vRow, e)), vh.getResult(0)))});
        }
        da = b.create<arith::SubFOp>(loc, da, eraseReduction.getResult(0));
      }
      b.create<memref::StoreOp>(loc, scalarReduction.getResult(0), DB,
                                ValueRange{scalar});
      b.create<memref::StoreOp>(loc, da, DA, ValueRange{scalar});
    }
  }
  b.setInsertionPointToEnd(&f.getBody().front());
  b.create<gpu::ReturnOp>(loc);
}

void emitDeltaNetBody(OpBuilder &b, Location loc, gpu::GPUFuncOp f, Type storeTy,
                      int64_t Dqk, int64_t Dv, DeltaNetFlags fl) {
  MLIRContext *ctx = b.getContext();
  Type f32 = b.getF32Type();
  Type i64 = b.getIntegerType(64);
  bool isF32 = storeTy.isF32();
  auto slt = arith::CmpIPredicate::slt;
  auto eqp = arith::CmpIPredicate::eq;

  auto ws = gpu::AddressSpaceAttr::get(ctx, gpu::AddressSpace::Workgroup);
  auto lds = [&](int64_t n) {
    return f.addWorkgroupAttribution(
        MemRefType::get({n}, f32, MemRefLayoutAttrInterface(), ws), loc);
  };
  Value state = lds(Dv * Dqk);   // Ŝ[e*Dqk + d] — thread e owns column e
  Value kSh = lds(Dqk);          // current k_t
  Value qSh = lds(Dqk);          // current q_t
  Value red = lds(Dv);           // ‖target‖ reduction scratch (modified only)

  b.setInsertionPointToStart(&f.getBody().front());
  // (Q, K : memref<?xstore>[BH*S*Dqk], V, O : [BH*S*Dv],
  //  gate : [BH*S*Dv], beta, decay : f32[BH*S], S : index)
  Value Q = f.getArgument(0), K = f.getArgument(1), V = f.getArgument(2),
        O = f.getArgument(3), gate = f.getArgument(4), beta = f.getArgument(5),
        decay = f.getArgument(6), S = f.getArgument(7);

  auto ci = [&](int64_t v) { return b.create<arith::ConstantIndexOp>(loc, v); };
  auto cf = [&](double v) {
    return b.create<arith::ConstantOp>(loc, f32, b.getF32FloatAttr(v));
  };
  Value c0 = ci(0), c1 = ci(1), cDqk = ci(Dqk), cDv = ci(Dv);
  Value zero = cf(0.0), one = cf(1.0);

  Value bid = b.create<gpu::BlockIdOp>(loc, gpu::Dimension::x);
  Value e = b.create<gpu::ThreadIdOp>(loc, gpu::Dimension::x);  // value-col, tid

  // base offsets: qk_base = bid*S*Dqk ; v_base = bid*S*Dv ; sc_base = bid*S
  Value SDqk = b.create<arith::MulIOp>(loc, S, cDqk);
  Value SDv = b.create<arith::MulIOp>(loc, S, cDv);
  Value qkBase = b.create<arith::MulIOp>(loc, bid, SDqk);
  Value vBase = b.create<arith::MulIOp>(loc, bid, SDv);
  Value scBase = b.create<arith::MulIOp>(loc, bid, S);

  auto extF = [&](Value mr, Value idx) -> Value {
    Value v = b.create<memref::LoadOp>(loc, mr, ValueRange{idx});
    return isF32 ? v : b.create<arith::ExtFOp>(loc, f32, v);
  };
  auto stCol = [&](Value d) {  // state index for this thread's column e
    return b.create<arith::AddIOp>(
        loc, b.create<arith::MulIOp>(loc, e, cDqk), d);
  };

  // init Ŝ[e,:] = 0
  {
    auto lp = b.create<scf::ForOp>(loc, c0, cDqk, c1);
    OpBuilder::InsertionGuard g(b);
    b.setInsertionPointToStart(lp.getBody());
    b.create<memref::StoreOp>(loc, zero, state,
                              ValueRange{stCol(lp.getInductionVar())});
  }
  b.create<gpu::BarrierOp>(loc);

  // sequential scan over t in [0, S)
  auto tloop = b.create<scf::ForOp>(loc, c0, S, c1);
  b.setInsertionPointToStart(tloop.getBody());
  Value t = tloop.getInductionVar();
  Value tDqk = b.create<arith::MulIOp>(loc, t, cDqk);
  Value tDv = b.create<arith::MulIOp>(loc, t, cDv);
  Value qkRow = b.create<arith::AddIOp>(loc, qkBase, tDqk);  // Q/K row start
  Value vRow = b.create<arith::AddIOp>(loc, vBase, tDv);     // V/O row start
  Value scIdx = b.create<arith::AddIOp>(loc, scBase, t);     // beta/decay index

  // cooperative load k_sh/q_sh (length Dqk), strided by blockDim=Dv
  {
    auto lp = b.create<scf::ForOp>(loc, e, cDqk, cDv);
    OpBuilder::InsertionGuard g(b);
    b.setInsertionPointToStart(lp.getBody());
    Value d = lp.getInductionVar();
    Value gidx = b.create<arith::AddIOp>(loc, qkRow, d);
    b.create<memref::StoreOp>(loc, extF(K, gidx), kSh, ValueRange{d});
    b.create<memref::StoreOp>(loc, extF(Q, gidx), qSh, ValueRange{d});
  }
  b.create<gpu::BarrierOp>(loc);

  // ---- thread e (value column) ----
  Value alpha = fl.hasDecay
                    ? b.create<memref::LoadOp>(loc, decay, ValueRange{scIdx})
                    : one;
  Value betaW = fl.hasBeta
                    ? b.create<memref::LoadOp>(loc, beta, ValueRange{scIdx})
                    : one;
  Value vIdx = b.create<arith::AddIOp>(loc, vRow, e);
  Value targetE = extF(V, vIdx);

  // d-loop reducer over this thread's state column: acc += f(d)
  auto colReduce = [&](function_ref<Value(Value /*d*/)> term) -> Value {
    auto lp = b.create<scf::ForOp>(loc, c0, cDqk, c1, ValueRange{zero});
    OpBuilder::InsertionGuard g(b);
    b.setInsertionPointToStart(lp.getBody());
    Value acc = lp.getRegionIterArgs()[0];
    Value v = term(lp.getInductionVar());
    b.create<scf::YieldOp>(loc,
                           ValueRange{b.create<arith::AddFOp>(loc, acc, v)});
    return lp.getResult(0);
  };

  if (fl.erase) {
    // v̂_e = Σ_d k_sh[d]·Ŝ[e,d]  (OLD state); target -= α·v̂
    Value vhat = colReduce([&](Value d) {
      Value kv = b.create<memref::LoadOp>(loc, kSh, ValueRange{d});
      Value sv = b.create<memref::LoadOp>(loc, state, ValueRange{stCol(d)});
      return b.create<arith::MulFOp>(loc, kv, sv).getResult();
    });
    targetE = b.create<arith::SubFOp>(
        loc, targetE, b.create<arith::MulFOp>(loc, alpha, vhat));
  }

  if (fl.hasDecay) {  // Ŝ[e,:] *= α
    auto lp = b.create<scf::ForOp>(loc, c0, cDqk, c1);
    OpBuilder::InsertionGuard g(b);
    b.setInsertionPointToStart(lp.getBody());
    Value idx = stCol(lp.getInductionVar());
    Value sv = b.create<memref::LoadOp>(loc, state, ValueRange{idx});
    b.create<memref::StoreOp>(loc, b.create<arith::MulFOp>(loc, sv, alpha),
                              state, ValueRange{idx});
  }

  Value scale = one;
  if (fl.modified) {
    // scale = 1 / (1 + ‖k‖·‖target‖);  ‖delta‖_F = ‖k‖₂·‖target‖₂
    Value knorm2 = colReduce([&](Value d) {  // reuse d-loop shape over Dqk
      Value kv = b.create<memref::LoadOp>(loc, kSh, ValueRange{d});
      return b.create<arith::MulFOp>(loc, kv, kv).getResult();
    });
    Value knorm = b.create<math::SqrtOp>(loc, knorm2);
    // ‖target‖: cross-thread reduction over the Dv columns via LDS.
    b.create<memref::StoreOp>(
        loc, b.create<arith::MulFOp>(loc, targetE, targetE), red,
        ValueRange{e});
    b.create<gpu::BarrierOp>(loc);
    Value isT0 = b.create<arith::CmpIOp>(loc, eqp, e, c0);
    auto t0if = b.create<scf::IfOp>(loc, isT0, /*withElse=*/false);
    {
      OpBuilder::InsertionGuard g(b);
      b.setInsertionPointToStart(t0if.thenBlock());
      auto lp = b.create<scf::ForOp>(loc, c0, cDv, c1, ValueRange{zero});
      b.setInsertionPointToStart(lp.getBody());
      Value acc = lp.getRegionIterArgs()[0];
      Value rv = b.create<memref::LoadOp>(loc, red,
                                          ValueRange{lp.getInductionVar()});
      b.create<scf::YieldOp>(loc,
                             ValueRange{b.create<arith::AddFOp>(loc, acc, rv)});
      b.setInsertionPointAfter(lp);
      b.create<memref::StoreOp>(loc, lp.getResult(0), red, ValueRange{c0});
    }
    b.create<gpu::BarrierOp>(loc);
    Value tnorm =
        b.create<math::SqrtOp>(loc, b.create<memref::LoadOp>(loc, red,
                                                             ValueRange{c0}));
    Value denom = b.create<arith::AddFOp>(
        loc, one, b.create<arith::MulFOp>(loc, knorm, tnorm));
    scale = b.create<arith::DivFOp>(loc, one, denom);
    b.create<gpu::BarrierOp>(loc);  // all read red[0] before next-t red reuse
  }

  // Ŝ[e,d] += β · k[d] · target · scale
  Value bt = b.create<arith::MulFOp>(
      loc, betaW, b.create<arith::MulFOp>(loc, targetE, scale));
  {
    auto lp = b.create<scf::ForOp>(loc, c0, cDqk, c1);
    OpBuilder::InsertionGuard g(b);
    b.setInsertionPointToStart(lp.getBody());
    Value d = lp.getInductionVar();
    Value idx = stCol(d);
    Value kv = b.create<memref::LoadOp>(loc, kSh, ValueRange{d});
    Value add = b.create<arith::MulFOp>(loc, bt, kv);
    Value sv = b.create<memref::LoadOp>(loc, state, ValueRange{idx});
    b.create<memref::StoreOp>(loc, b.create<arith::AddFOp>(loc, sv, add), state,
                              ValueRange{idx});
  }

  // O_e = Σ_d q[d]·Ŝ[e,d]  (NEW state)
  Value oE = colReduce([&](Value d) {
    Value qv = b.create<memref::LoadOp>(loc, qSh, ValueRange{d});
    Value sv = b.create<memref::LoadOp>(loc, state, ValueRange{stCol(d)});
    return b.create<arith::MulFOp>(loc, qv, sv).getResult();
  });
  if (fl.hasGate) {  // O *= sigmoid(gate)
    Value g = extF(gate, vIdx);
    Value sig = b.create<arith::DivFOp>(
        loc, one,
        b.create<arith::AddFOp>(
            loc, one, b.create<math::ExpOp>(loc, b.create<arith::NegFOp>(loc, g))));
    oE = b.create<arith::MulFOp>(loc, oE, sig);
  }
  Value sv = isF32 ? oE : b.create<arith::TruncFOp>(loc, storeTy, oE);
  b.create<memref::StoreOp>(loc, sv, O, ValueRange{vIdx});

  b.create<gpu::BarrierOp>(loc);  // before next-t cooperative load reuses k/q
  // (end t-loop body)

  b.setInsertionPointToEnd(&f.getBody().front());
  b.create<gpu::ReturnOp>(loc);
}

struct GenerateROCMDeltaNetKernelPass
    : PassWrapper<GenerateROCMDeltaNetKernelPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(GenerateROCMDeltaNetKernelPass)

  StringRef getArgument() const final { return "generate-rocm-deltanet-kernel"; }
  StringRef getDescription() const final {
    return "Expand a tessera_rocm.deltanet directive into a causal "
           "sequential-scan gated/delta linear-attention gpu kernel "
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
      if (op->getName().getStringRef() == "tessera_rocm.deltanet")
        directives.push_back(op);
    });
    for (Operation *op : directives) {
      auto nameAttr = op->getAttrOfType<StringAttr>("name");
      if (!nameAttr) {
        op->emitError("tessera_rocm.deltanet missing name");
        return signalPassFailure();
      }
      auto dimAttr = [&](StringRef n) -> int64_t {
        if (auto a = op->getAttrOfType<IntegerAttr>(n)) return a.getInt();
        return 0;
      };
      int64_t Dqk = dimAttr("d_qk"), Dv = dimAttr("d_v");
      if (Dqk <= 0 || Dv <= 0) {
        op->emitError("generate-rocm-deltanet-kernel: d_qk and d_v must be "
                      "positive");
        return signalPassFailure();
      }
      auto flag = [&](StringRef n) -> bool {
        if (auto a = op->getAttrOfType<BoolAttr>(n)) return a.getValue();
        return false;
      };
      DeltaNetFlags fl;
      fl.erase = flag("erase");
      fl.modified = flag("modified");
      fl.hasGate = flag("has_gate");
      fl.hasBeta = flag("has_beta");
      fl.hasDecay = flag("has_decay");
      bool backward = flag("backward");
      int64_t chunkSize = dimAttr("chunk_size");
      if (backward && chunkSize <= 0) chunkSize = 64;

      Type storeTy = b_storeType(op);
      if (!storeTy) return signalPassFailure();

      OpBuilder b(module.getBodyRegion());
      b.setInsertionPointToEnd(module.getBody());
      Location loc = op->getLoc();
      std::string kname = nameAttr.getValue().str();
      auto gpuMod = b.create<gpu::GPUModuleOp>(loc, kname + "_mod");
      b.setInsertionPointToStart(&gpuMod.getBodyRegion().front());
      Type idxTy = b.getIndexType();
      auto store = MemRefType::get({ShapedType::kDynamic}, storeTy);
      auto f32mr = MemRefType::get({ShapedType::kDynamic}, b.getF32Type());
      if (backward) {
        if (!storeTy.isF32()) {
          op->emitError("generate-rocm-deltanet-kernel: backward packaging "
                        "currently requires f32 storage");
          return signalPassFailure();
        }
        auto checkpointTy =
            b.getFunctionType({f32mr, f32mr, f32mr, f32mr, f32mr, idxTy}, {});
        auto chunkSummaryTy = b.getFunctionType(
            {f32mr, f32mr, f32mr, f32mr, f32mr, f32mr, idxTy, idxTy}, {});
        auto chunkPrefixTy = b.getFunctionType(
            {f32mr, f32mr, f32mr, idxTy, idxTy}, {});
        auto chunkFillTy = b.getFunctionType(
            {f32mr, f32mr, f32mr, f32mr, f32mr, f32mr, idxTy, idxTy}, {});
        SmallVector<Type> reverseArgs(17, f32mr);
        reverseArgs.push_back(idxTy);
        auto reverseTy = b.getFunctionType(reverseArgs, {});
        auto checkpoint = b.create<gpu::GPUFuncOp>(
            loc, kname + "_checkpoint", checkpointTy);
        checkpoint->setAttr(gpu::GPUDialect::getKernelFuncAttrName(),
                            b.getUnitAttr());
        auto chunkSummary = b.create<gpu::GPUFuncOp>(
            loc, kname + "_chunk_summary", chunkSummaryTy);
        chunkSummary->setAttr(gpu::GPUDialect::getKernelFuncAttrName(),
                              b.getUnitAttr());
        auto chunkPrefix = b.create<gpu::GPUFuncOp>(
            loc, kname + "_chunk_prefix", chunkPrefixTy);
        chunkPrefix->setAttr(gpu::GPUDialect::getKernelFuncAttrName(),
                             b.getUnitAttr());
        auto chunkFill = b.create<gpu::GPUFuncOp>(
            loc, kname + "_chunk_fill", chunkFillTy);
        chunkFill->setAttr(gpu::GPUDialect::getKernelFuncAttrName(),
                           b.getUnitAttr());
        auto reverse =
            b.create<gpu::GPUFuncOp>(loc, kname + "_reverse", reverseTy);
        reverse->setAttr(gpu::GPUDialect::getKernelFuncAttrName(),
                         b.getUnitAttr());
        OpBuilder checkpointBody(checkpoint.getContext());
        emitDeltaNetCheckpointBody(checkpointBody, loc, checkpoint, Dqk, Dv, fl);
        OpBuilder chunkSummaryBody(chunkSummary.getContext());
        emitDeltaNetChunkSummaryBody(chunkSummaryBody, loc, chunkSummary, Dqk,
                                     Dv, chunkSize, fl);
        OpBuilder chunkPrefixBody(chunkPrefix.getContext());
        emitDeltaNetChunkPrefixBody(chunkPrefixBody, loc, chunkPrefix, Dqk, Dv,
                                    chunkSize);
        OpBuilder chunkFillBody(chunkFill.getContext());
        emitDeltaNetChunkFillBody(chunkFillBody, loc, chunkFill, Dqk, Dv,
                                  chunkSize, fl);
        OpBuilder reverseBody(reverse.getContext());
        emitDeltaNetReverseBody(reverseBody, loc, reverse, Dqk, Dv, chunkSize,
                                fl);
        op->erase();
        continue;
      }
      // (Q,K,V,O,gate : store, beta,decay : f32, S : index)
      auto fnTy = b.getFunctionType(
          {store, store, store, store, store, f32mr, f32mr, idxTy}, {});
      auto gpuFunc = b.create<gpu::GPUFuncOp>(loc, kname, fnTy);
      gpuFunc->setAttr(gpu::GPUDialect::getKernelFuncAttrName(),
                       b.getUnitAttr());
      OpBuilder body(gpuFunc.getContext());
      emitDeltaNetBody(body, loc, gpuFunc, storeTy, Dqk, Dv, fl);
      op->erase();
    }
  }

  // dtype attr -> storage type (f32/f16/bf16); emits an error + returns null.
  Type b_storeType(Operation *op) {
    OpBuilder b(op->getContext());
    if (auto a = op->getAttrOfType<StringAttr>("dtype")) {
      StringRef dt = a.getValue();
      if (dt == "f16" || dt == "float16") return b.getF16Type();
      if (dt == "bf16" || dt == "bfloat16") return b.getBF16Type();
      if (dt != "f32" && dt != "float32") {
        op->emitError("generate-rocm-deltanet-kernel: dtype must be f32, f16, "
                      "or bf16 (got '")
            << dt << "')";
        return nullptr;
      }
    }
    return b.getF32Type();
  }
};

} // namespace

std::unique_ptr<mlir::Pass>
mlir::tessera_rocm::createGenerateROCMDeltaNetKernelPass() {
  return std::make_unique<GenerateROCMDeltaNetKernelPass>();
}
