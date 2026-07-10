//===- GenerateROCMRecurrentCellKernel.cpp - RNN/GRU cell gpu kernel -----===//
//
// Expands `tessera_rocm.recurrent_cell` into a fused single-step recurrent cell
// kernel, one thread per output element (gid over B*H):
//
//   simple_rnn : h' = act(x·W_ih + h·W_hh + bias)        act = tanh | relu
//                W_ih [In,H], W_hh [H,H], bias [H]        (out [B,H])
//   gru        : gates_x = x·W_ih + b_ih, gates_h = h·W_hh + b_hh  (gate order
//                z,r,n; W_ih [In,3H], W_hh [H,3H], biases [3H])
//                z = σ(x_z+h_z), r = σ(x_r+h_r), n = tanh(x_n + r·h_n)
//                h' = (1−z)·n + z·h                       (out [B,H])
//
// The native device analog of the structured-compute host lane (nn.functional
// simple_rnn_cell / gru_cell): the two gate GEMMs and the elementwise gate math
// are fused into one kernel, so `native_gpu` provenance is genuine. Storage is
// f16/bf16/f32 with an f32 accumulator (loads ext to f32, stores trunc); the
// gate transcendentals lower through convert-math-to-rocdl. Bias buffers are
// always present in the ABI (the runtime passes zeros when a bias is absent),
// keeping one signature per (cell, dtype). B/In/H are runtime index args.
// Validated vs the numpy reference on gfx1151.
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

enum class Cell { SimpleRNN, GRU };
enum class Act { Tanh, Relu };

// f32 helpers -----------------------------------------------------------------
static Value extLoad(OpBuilder &b, Location loc, Value mem, Value idx,
                     bool isF32, Type f32) {
  Value v = b.create<memref::LoadOp>(loc, mem, ValueRange{idx});
  return isF32 ? v : b.create<arith::ExtFOp>(loc, f32, v);
}
static void truncStore(OpBuilder &b, Location loc, Value val, Value mem,
                       Value idx, bool isF32, Type storeTy) {
  Value sv = isF32 ? val : b.create<arith::TruncFOp>(loc, storeTy, val);
  b.create<memref::StoreOp>(loc, sv, mem, ValueRange{idx});
}
static Value sigmoid(OpBuilder &b, Location loc, Value x, Type f32) {
  Value one = b.create<arith::ConstantOp>(loc, f32, b.getF32FloatAttr(1.0f));
  Value e = b.create<math::ExpOp>(loc, b.create<arith::NegFOp>(loc, x));
  return b.create<arith::DivFOp>(loc, one, b.create<arith::AddFOp>(loc, one, e));
}

// out[b,j] = act( Σ_i x[b,i]·Wih[i,H,j] + Σ_k h[b,k]·Whh[k,H,j] + bias[j] )
void emitSimpleRNN(OpBuilder &b, Location loc, gpu::GPUFuncOp f, Type storeTy,
                   Act act) {
  Type f32 = b.getF32Type();
  bool isF32 = storeTy.isF32();
  auto slt = arith::CmpIPredicate::slt;
  b.setInsertionPointToStart(&f.getBody().front());
  Value X = f.getArgument(0), Hin = f.getArgument(1), Wih = f.getArgument(2),
        Whh = f.getArgument(3), Bias = f.getArgument(4), O = f.getArgument(5);
  Value B = f.getArgument(6), In = f.getArgument(7), H = f.getArgument(8);
  auto ci = [&](int64_t v) { return b.create<arith::ConstantIndexOp>(loc, v); };
  Value c0 = ci(0), c1 = ci(1);
  Value bid = b.create<gpu::BlockIdOp>(loc, gpu::Dimension::x);
  Value tid = b.create<gpu::ThreadIdOp>(loc, gpu::Dimension::x);
  Value gid = b.create<arith::AddIOp>(
      loc, b.create<arith::MulIOp>(loc, bid, ci(BD)), tid);
  Value total = b.create<arith::MulIOp>(loc, B, H);
  auto guard = b.create<scf::IfOp>(
      loc, b.create<arith::CmpIOp>(loc, slt, gid, total), /*withElse=*/false);
  b.setInsertionPointToStart(guard.thenBlock());

  Value bi = b.create<arith::DivUIOp>(loc, gid, H);   // batch row
  Value jj = b.create<arith::RemUIOp>(loc, gid, H);   // hidden col
  Value xRow = b.create<arith::MulIOp>(loc, bi, In);
  Value hRow = b.create<arith::MulIOp>(loc, bi, H);
  Value acc0 = extLoad(b, loc, Bias, jj, isF32, f32);

  // Σ_i x[b,i]·Wih[i*H + j]
  auto lx = b.create<scf::ForOp>(loc, c0, In, c1, ValueRange{acc0});
  {
    OpBuilder::InsertionGuard g(b);
    b.setInsertionPointToStart(lx.getBody());
    Value i = lx.getInductionVar();
    Value acc = lx.getRegionIterArgs()[0];
    Value xv = extLoad(b, loc, X, b.create<arith::AddIOp>(loc, xRow, i), isF32,
                       f32);
    Value wv = extLoad(b, loc, Wih,
                       b.create<arith::AddIOp>(
                           loc, b.create<arith::MulIOp>(loc, i, H), jj),
                       isF32, f32);
    b.create<scf::YieldOp>(
        loc, ValueRange{b.create<arith::AddFOp>(
                 loc, acc, b.create<arith::MulFOp>(loc, xv, wv))});
  }
  // + Σ_k h[b,k]·Whh[k*H + j]
  auto lh = b.create<scf::ForOp>(loc, c0, H, c1, ValueRange{lx.getResult(0)});
  {
    OpBuilder::InsertionGuard g(b);
    b.setInsertionPointToStart(lh.getBody());
    Value k = lh.getInductionVar();
    Value acc = lh.getRegionIterArgs()[0];
    Value hv = extLoad(b, loc, Hin, b.create<arith::AddIOp>(loc, hRow, k), isF32,
                       f32);
    Value wv = extLoad(b, loc, Whh,
                       b.create<arith::AddIOp>(
                           loc, b.create<arith::MulIOp>(loc, k, H), jj),
                       isF32, f32);
    b.create<scf::YieldOp>(
        loc, ValueRange{b.create<arith::AddFOp>(
                 loc, acc, b.create<arith::MulFOp>(loc, hv, wv))});
  }
  Value pre = lh.getResult(0);
  Value out;
  if (act == Act::Relu) {
    Value z = b.create<arith::ConstantOp>(loc, f32, b.getF32FloatAttr(0.0f));
    out = b.create<arith::MaximumFOp>(loc, pre, z);
  } else {
    out = b.create<math::TanhOp>(loc, pre);
  }
  truncStore(b, loc, out, O, gid, isF32, storeTy);
  b.setInsertionPointToEnd(&f.getBody().front());
  b.create<gpu::ReturnOp>(loc);
}

// GRU, gate order z,r,n. W_ih [In,3H], W_hh [H,3H]; column of gate g = g*H + j.
void emitGRU(OpBuilder &b, Location loc, gpu::GPUFuncOp f, Type storeTy) {
  Type f32 = b.getF32Type();
  bool isF32 = storeTy.isF32();
  auto slt = arith::CmpIPredicate::slt;
  b.setInsertionPointToStart(&f.getBody().front());
  Value X = f.getArgument(0), Hin = f.getArgument(1), Wih = f.getArgument(2),
        Whh = f.getArgument(3), Bih = f.getArgument(4), Bhh = f.getArgument(5),
        O = f.getArgument(6);
  Value B = f.getArgument(7), In = f.getArgument(8), H = f.getArgument(9);
  auto ci = [&](int64_t v) { return b.create<arith::ConstantIndexOp>(loc, v); };
  Value c0 = ci(0), c1 = ci(1);
  Value H3 = b.create<arith::MulIOp>(loc, H, ci(3));
  Value bid = b.create<gpu::BlockIdOp>(loc, gpu::Dimension::x);
  Value tid = b.create<gpu::ThreadIdOp>(loc, gpu::Dimension::x);
  Value gid = b.create<arith::AddIOp>(
      loc, b.create<arith::MulIOp>(loc, bid, ci(BD)), tid);
  Value total = b.create<arith::MulIOp>(loc, B, H);
  auto guard = b.create<scf::IfOp>(
      loc, b.create<arith::CmpIOp>(loc, slt, gid, total), /*withElse=*/false);
  b.setInsertionPointToStart(guard.thenBlock());

  Value bi = b.create<arith::DivUIOp>(loc, gid, H);
  Value jj = b.create<arith::RemUIOp>(loc, gid, H);
  Value xRow = b.create<arith::MulIOp>(loc, bi, In);
  Value hRow = b.create<arith::MulIOp>(loc, bi, H);
  // Gate column offsets within the 3H-wide weight rows.
  Value colZ = jj;
  Value colR = b.create<arith::AddIOp>(loc, H, jj);
  Value colN = b.create<arith::AddIOp>(loc, b.create<arith::MulIOp>(loc, ci(2), H),
                                       jj);
  Value bz = extLoad(b, loc, Bih, colZ, isF32, f32);
  Value br = extLoad(b, loc, Bih, colR, isF32, f32);
  Value bn = extLoad(b, loc, Bih, colN, isF32, f32);

  // x-side: one pass over In accumulating all 3 gates (reuse each x load).
  auto lx = b.create<scf::ForOp>(loc, c0, In, c1, ValueRange{bz, br, bn});
  {
    OpBuilder::InsertionGuard g(b);
    b.setInsertionPointToStart(lx.getBody());
    Value i = lx.getInductionVar();
    Value az = lx.getRegionIterArgs()[0], ar = lx.getRegionIterArgs()[1],
          an = lx.getRegionIterArgs()[2];
    Value xv = extLoad(b, loc, X, b.create<arith::AddIOp>(loc, xRow, i), isF32,
                       f32);
    Value base = b.create<arith::MulIOp>(loc, i, H3);   // row i of [In,3H]
    auto wg = [&](Value col) {
      return extLoad(b, loc, Wih, b.create<arith::AddIOp>(loc, base, col), isF32,
                     f32);
    };
    Value nz = b.create<arith::AddFOp>(loc, az,
                                       b.create<arith::MulFOp>(loc, xv, wg(colZ)));
    Value nr = b.create<arith::AddFOp>(loc, ar,
                                       b.create<arith::MulFOp>(loc, xv, wg(colR)));
    Value nn = b.create<arith::AddFOp>(loc, an,
                                       b.create<arith::MulFOp>(loc, xv, wg(colN)));
    b.create<scf::YieldOp>(loc, ValueRange{nz, nr, nn});
  }
  Value xz = lx.getResult(0), xr = lx.getResult(1), xn = lx.getResult(2);

  // h-side: one pass over H accumulating all 3 gates (start from b_hh).
  Value hz0 = extLoad(b, loc, Bhh, colZ, isF32, f32);
  Value hr0 = extLoad(b, loc, Bhh, colR, isF32, f32);
  Value hn0 = extLoad(b, loc, Bhh, colN, isF32, f32);
  auto lh = b.create<scf::ForOp>(loc, c0, H, c1, ValueRange{hz0, hr0, hn0});
  {
    OpBuilder::InsertionGuard g(b);
    b.setInsertionPointToStart(lh.getBody());
    Value k = lh.getInductionVar();
    Value az = lh.getRegionIterArgs()[0], ar = lh.getRegionIterArgs()[1],
          an = lh.getRegionIterArgs()[2];
    Value hv = extLoad(b, loc, Hin, b.create<arith::AddIOp>(loc, hRow, k), isF32,
                       f32);
    Value base = b.create<arith::MulIOp>(loc, k, H3);   // row k of [H,3H]
    auto wg = [&](Value col) {
      return extLoad(b, loc, Whh, b.create<arith::AddIOp>(loc, base, col), isF32,
                     f32);
    };
    Value nz = b.create<arith::AddFOp>(loc, az,
                                       b.create<arith::MulFOp>(loc, hv, wg(colZ)));
    Value nr = b.create<arith::AddFOp>(loc, ar,
                                       b.create<arith::MulFOp>(loc, hv, wg(colR)));
    Value nn = b.create<arith::AddFOp>(loc, an,
                                       b.create<arith::MulFOp>(loc, hv, wg(colN)));
    b.create<scf::YieldOp>(loc, ValueRange{nz, nr, nn});
  }
  Value hz = lh.getResult(0), hr = lh.getResult(1), hn = lh.getResult(2);

  // z = σ(xz+hz); r = σ(xr+hr); n = tanh(xn + r·hn); h' = (1−z)·n + z·hprev
  Value one = b.create<arith::ConstantOp>(loc, f32, b.getF32FloatAttr(1.0f));
  Value z = sigmoid(b, loc, b.create<arith::AddFOp>(loc, xz, hz), f32);
  Value r = sigmoid(b, loc, b.create<arith::AddFOp>(loc, xr, hr), f32);
  Value nPre = b.create<arith::AddFOp>(loc, xn, b.create<arith::MulFOp>(loc, r, hn));
  Value n = b.create<math::TanhOp>(loc, nPre);
  Value hprev = extLoad(b, loc, Hin, b.create<arith::AddIOp>(loc, hRow, jj), isF32,
                        f32);
  Value oneMinusZ = b.create<arith::SubFOp>(loc, one, z);
  Value out = b.create<arith::AddFOp>(
      loc, b.create<arith::MulFOp>(loc, oneMinusZ, n),
      b.create<arith::MulFOp>(loc, z, hprev));
  truncStore(b, loc, out, O, gid, isF32, storeTy);
  b.setInsertionPointToEnd(&f.getBody().front());
  b.create<gpu::ReturnOp>(loc);
}

struct GenerateROCMRecurrentCellKernelPass
    : PassWrapper<GenerateROCMRecurrentCellKernelPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(
      GenerateROCMRecurrentCellKernelPass)

  StringRef getArgument() const final {
    return "generate-rocm-recurrent-cell-kernel";
  }
  StringRef getDescription() const final {
    return "Expand a tessera_rocm.recurrent_cell directive into a fused "
           "single-step simple_rnn / gru gpu kernel (compiler-generated)";
  }
  void getDependentDialects(DialectRegistry &registry) const final {
    registry.insert<gpu::GPUDialect, scf::SCFDialect, arith::ArithDialect,
                    math::MathDialect, memref::MemRefDialect>();
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();
    SmallVector<Operation *> directives;
    module.walk([&](Operation *op) {
      if (op->getName().getStringRef() == "tessera_rocm.recurrent_cell")
        directives.push_back(op);
    });
    for (Operation *op : directives) {
      auto nameAttr = op->getAttrOfType<StringAttr>("name");
      if (!nameAttr) {
        op->emitError("tessera_rocm.recurrent_cell missing name");
        return signalPassFailure();
      }
      Cell cell = Cell::SimpleRNN;
      if (auto a = op->getAttrOfType<StringAttr>("cell")) {
        if (a.getValue() == "gru")
          cell = Cell::GRU;
        else if (a.getValue() != "simple_rnn") {
          op->emitError("generate-rocm-recurrent-cell-kernel: cell must be "
                        "simple_rnn or gru (got '")
              << a.getValue() << "')";
          return signalPassFailure();
        }
      }
      Act act = Act::Tanh;
      if (auto a = op->getAttrOfType<StringAttr>("act")) {
        if (a.getValue() == "relu")
          act = Act::Relu;
        else if (a.getValue() != "tanh") {
          op->emitError("generate-rocm-recurrent-cell-kernel: act must be tanh "
                        "or relu (got '")
              << a.getValue() << "')";
          return signalPassFailure();
        }
      }
      Type storeTy = b_typeFromAttr(op);
      if (!storeTy) return signalPassFailure();

      OpBuilder b(module.getBodyRegion());
      b.setInsertionPointToEnd(module.getBody());
      Location loc = op->getLoc();
      std::string kname = nameAttr.getValue().str();
      Type idxTy = b.getIndexType();
      auto mem = MemRefType::get({ShapedType::kDynamic}, storeTy);
      // simple_rnn: (X,Hin,Wih,Whh,Bias,O, B,In,H) = 6 memrefs + 3 index.
      // gru:        (X,Hin,Wih,Whh,Bih,Bhh,O, B,In,H) = 7 memrefs + 3 index.
      SmallVector<Type> argTys;
      int64_t nMem = (cell == Cell::GRU) ? 7 : 6;
      for (int64_t i = 0; i < nMem; ++i) argTys.push_back(mem);
      argTys.push_back(idxTy);
      argTys.push_back(idxTy);
      argTys.push_back(idxTy);
      auto fnTy = b.getFunctionType(argTys, {});
      auto gpuMod = b.create<gpu::GPUModuleOp>(loc, kname + "_mod");
      b.setInsertionPointToStart(&gpuMod.getBodyRegion().front());
      auto gpuFunc = b.create<gpu::GPUFuncOp>(loc, kname, fnTy);
      gpuFunc->setAttr(gpu::GPUDialect::getKernelFuncAttrName(), b.getUnitAttr());
      OpBuilder body(gpuFunc.getContext());
      if (cell == Cell::GRU)
        emitGRU(body, loc, gpuFunc, storeTy);
      else
        emitSimpleRNN(body, loc, gpuFunc, storeTy, act);
      op->erase();
    }
  }

  Type b_typeFromAttr(Operation *op) {
    OpBuilder b(op->getContext());
    if (auto a = op->getAttrOfType<StringAttr>("dtype")) {
      StringRef dt = a.getValue();
      if (dt == "f16" || dt == "float16") return b.getF16Type();
      if (dt == "bf16" || dt == "bfloat16") return b.getBF16Type();
      if (dt == "f32" || dt == "float32") return b.getF32Type();
      op->emitError("generate-rocm-recurrent-cell-kernel: dtype must be f32, "
                    "f16, or bf16 (got '")
          << dt << "')";
      return Type();
    }
    return b.getF32Type();
  }
};

} // namespace

std::unique_ptr<mlir::Pass>
mlir::tessera_rocm::createGenerateROCMRecurrentCellKernelPass() {
  return std::make_unique<GenerateROCMRecurrentCellKernelPass>();
}
