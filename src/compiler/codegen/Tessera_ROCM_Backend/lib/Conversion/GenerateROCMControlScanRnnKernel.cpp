//===- GenerateROCMControlScanRnnKernel.cpp — nonlinear RNN-cell scan -----===//
//
// CF4e-3 of the control-flow track (ROCm-led, gfx1151). The full nonlinear RNN
// cell as a scan — two captures, a bias, and an activation:
//
//     h_t = tanh(h_{t-1} @ W + x_t @ U + b) ,   y_t = h_t
//
// over a 1×K carry h, two K×K loop-invariant captures W (recurrent) and U (input
// projection), a 1×K bias b, and the per-step input x_t (the scan's xs). Two
// GEMVs (h@W over the LDS carry, x@U over the per-step input) + bias + tanh — the
// canonical Elman/GRU-style recurrent cell. Builds directly on the CF4e-2 capture
// substrate (control_scan already carries variadic captures); the new piece is the
// SECOND GEMV (over x_t) fused into the per-step reduction, plus the bias + tanh.
//
//   gpu.func @ctrl_scan_rnn(%INIT,%XS,%W,%U,%B,%COUT,%YS: memref<?xf32>, %K) {
//     %lds = workgroup memref<BDxf32>                      // h in LDS
//     if tid<K { lds[tid] = INIT[tid] } ; barrier
//     scf.for %t = 0 to TRIP {
//       if tid<K {
//         acc = Σ_k ( lds[k]·W[k*K+tid] + XS[t*K+k]·U[k*K+tid] )   // h@W + x@U
//         h'  = tanh(acc + B[tid])                                 // + b, tanh
//       }
//       barrier ; if tid<K { lds[tid]=h' ; YS[t*K+tid]=h' } ; barrier
//     }
//     if tid<K { COUT[tid]=lds[tid] }
//   }
//
// Single-tile (K ≤ BD). TRIP baked. xs/ys keep the body's per-step shape (1×K) →
// (trip,1,K). Bodies outside the exact two-matmul + bias + tanh form are left for
// the guard / SCF.

#include "TesseraROCM/Passes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/SymbolTable.h"
#include "llvm/ADT/SmallVector.h"

using namespace mlir;

namespace {

constexpr int64_t BD = 256;

static bool is1xK(Type t, int64_t K) {
  auto r = dyn_cast<RankedTensorType>(t);
  return r && r.getRank() == 2 && r.getDimSize(0) == 1 &&
         r.getDimSize(1) == K && r.getElementType().isF32();
}
static bool is2D(Type t, int64_t d0, int64_t d1) {
  auto r = dyn_cast<RankedTensorType>(t);
  return r && r.getRank() == 2 && r.getDimSize(0) == d0 &&
         r.getDimSize(1) == d1 && r.getElementType().isF32();
}
static bool is3D(Type t, int64_t d0, int64_t d1, int64_t d2) {
  auto r = dyn_cast<RankedTensorType>(t);
  return r && r.getRank() == 3 && r.getDimSize(0) == d0 &&
         r.getDimSize(1) == d1 && r.getDimSize(2) == d2 &&
         r.getElementType().isF32();
}
static bool noTranspose(Operation *m) {
  auto ta = m->getAttrOfType<BoolAttr>("transposeA");
  auto tb = m->getAttrOfType<BoolAttr>("transposeB");
  return !((ta && ta.getValue()) || (tb && tb.getValue()));
}

// control_scan with two captures (W, U) + a bias (b) whose body is the RNN cell
// `(h, x, W, U, b) -> (tanh(h@W + x@U + b), same)`. operands = [init(1×K),
// xs(trip×1×K), W(K×K), U(K×K), b(1×K)]; results = [carry(1×K), ys(trip×1×K)].
// Fills K, trip. Returns the body, or null.
static func::FuncOp validateScanRnn(Operation *op, SymbolTable &symTab,
                                    int64_t &K, int64_t &trip) {
  auto bodySym = op->getAttrOfType<FlatSymbolRefAttr>("body");
  auto tripA = op->getAttrOfType<IntegerAttr>("trip");
  auto carryA = op->getAttrOfType<IntegerAttr>("carry_arg_index");
  if (!bodySym || !tripA || !carryA || carryA.getInt() != 0 ||
      tripA.getInt() <= 0 || op->getNumOperands() != 5 ||
      op->getNumResults() != 2)
    return nullptr;  // init + xs + (W, U, b)
  trip = tripA.getInt();

  auto initT = dyn_cast<RankedTensorType>(op->getOperand(0).getType());
  if (!initT || initT.getRank() != 2 || initT.getDimSize(0) != 1 ||
      !initT.getElementType().isF32())
    return nullptr;
  K = initT.getDimSize(1);
  if (K <= 0 || K > BD)
    return nullptr;
  if (!is3D(op->getOperand(1).getType(), trip, 1, K) ||  // xs: trip×1×K
      !is2D(op->getOperand(2).getType(), K, K) ||         // W: K×K
      !is2D(op->getOperand(3).getType(), K, K) ||         // U: K×K
      !is1xK(op->getOperand(4).getType(), K) ||           // b: 1×K
      !is1xK(op->getResult(0).getType(), K) ||            // carry: 1×K
      !is3D(op->getResult(1).getType(), trip, 1, K))      // ys: trip×1×K
    return nullptr;

  auto fn = dyn_cast_or_null<func::FuncOp>(
      symTab.lookupNearestSymbolFrom(op, bodySym.getAttr()));
  if (!fn || fn.isExternal())
    return nullptr;
  FunctionType ft = fn.getFunctionType();
  if (ft.getNumInputs() != 5 || ft.getNumResults() != 2 ||
      !is1xK(ft.getInput(0), K) || !is1xK(ft.getInput(1), K) ||
      !is2D(ft.getInput(2), K, K) || !is2D(ft.getInput(3), K, K) ||
      !is1xK(ft.getInput(4), K) || !is1xK(ft.getResult(0), K) ||
      !is1xK(ft.getResult(1), K))
    return nullptr;
  // body must be exactly:
  //   %m1 = matmul(h, W) ; %m2 = matmul(x, U) ; %s1 = add(%m1,%m2) ;
  //   %s2 = add(%s1, b) ; %t = tanh(%s2) ; return %t, %t
  Block &blk = fn.getBody().front();
  Value h = blk.getArgument(0), x = blk.getArgument(1), W = blk.getArgument(2),
        U = blk.getArgument(3), b = blk.getArgument(4);
  auto it = blk.begin();
  auto isMatmul = [&](Operation *o, Value a0, Value a1) {
    return o && o->getName().getStringRef() == "tessera.matmul" &&
           o->getNumOperands() == 2 && o->getOperand(0) == a0 &&
           o->getOperand(1) == a1 && o->getNumResults() == 1 && noTranspose(o);
  };
  auto isAdd = [&](Operation *o, Value p, Value q) {
    return o && o->getName().getStringRef() == "tessera.add" &&
           o->getNumOperands() == 2 && o->getNumResults() == 1 &&
           ((o->getOperand(0) == p && o->getOperand(1) == q) ||
            (o->getOperand(0) == q && o->getOperand(1) == p));
  };
  Operation *m1 = (it == blk.end()) ? nullptr : &*it;
  if (!isMatmul(m1, h, W))
    return nullptr;
  Operation *m2 = m1->getNextNode();
  if (!isMatmul(m2, x, U))
    return nullptr;
  Operation *s1 = m2->getNextNode();
  if (!isAdd(s1, m1->getResult(0), m2->getResult(0)))
    return nullptr;
  Operation *s2 = s1->getNextNode();
  if (!isAdd(s2, s1->getResult(0), b))
    return nullptr;
  Operation *tn = s2->getNextNode();
  if (!tn || tn->getName().getStringRef() != "tessera.tanh" ||
      tn->getNumOperands() != 1 || tn->getOperand(0) != s2->getResult(0) ||
      tn->getNumResults() != 1)
    return nullptr;
  auto ret = dyn_cast_or_null<func::ReturnOp>(tn->getNextNode());
  if (!ret || ret.getNumOperands() != 2 ||
      ret.getOperand(0) != tn->getResult(0) ||
      ret.getOperand(1) != tn->getResult(0))
    return nullptr;
  return fn;
}

struct GenerateROCMControlScanRnnKernelPass
    : public PassWrapper<GenerateROCMControlScanRnnKernelPass,
                         OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(
      GenerateROCMControlScanRnnKernelPass)

  StringRef getArgument() const final {
    return "generate-rocm-control-scan-rnn-kernel";
  }
  StringRef getDescription() const final {
    return "CF4e-3: lower a nonlinear RNN-cell scan (h_t = tanh(h@W + x@U + b); "
           "two GEMV captures + bias + tanh + per-step xs) to one "
           "cooperative-workgroup gpu.func for gfx1151.";
  }
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<gpu::GPUDialect, scf::SCFDialect, arith::ArithDialect,
                    math::MathDialect, memref::MemRefDialect, func::FuncDialect>();
  }

  void emitKernel(Operation *op, int64_t K, int64_t trip, ModuleOp module,
                  unsigned idx) {
    MLIRContext *ctx = module.getContext();
    OpBuilder b(module.getBodyRegion());
    b.setInsertionPointToEnd(module.getBody());
    Location loc = op->getLoc();
    std::string kname = ("tessera_control_scan_rnn_" + Twine(idx)).str();

    Type f32 = b.getF32Type(), idxTy = b.getIndexType();
    auto memTy = MemRefType::get({ShapedType::kDynamic}, f32);

    auto gpuMod = gpu::GPUModuleOp::create(b, loc, kname + "_mod");
    b.setInsertionPointToStart(&gpuMod.getBodyRegion().front());
    // (INIT, XS, W, U, B, COUT, YS : memref<?xf32>, K : index)
    auto fnTy = b.getFunctionType(
        {memTy, memTy, memTy, memTy, memTy, memTy, memTy, idxTy}, {});
    auto f = gpu::GPUFuncOp::create(b, loc, kname, fnTy);
    f->setAttr(gpu::GPUDialect::getKernelFuncAttrName(), b.getUnitAttr());

    auto ws = gpu::AddressSpaceAttr::get(ctx, gpu::AddressSpace::Workgroup);
    auto ldsT = MemRefType::get({BD}, f32, MemRefLayoutAttrInterface(), ws);
    Value lds = f.addWorkgroupAttribution(ldsT, loc);

    OpBuilder kb(f.getContext());
    kb.setInsertionPointToStart(&f.getBody().front());
    Value INIT = f.getArgument(0), XS = f.getArgument(1), W = f.getArgument(2),
          U = f.getArgument(3), B = f.getArgument(4), COUT = f.getArgument(5),
          YS = f.getArgument(6), Kv = f.getArgument(7);
    auto ci = [&](int64_t v) { return arith::ConstantIndexOp::create(kb, loc, v); };
    Value c0 = ci(0), c1 = ci(1), cK = ci(K),
          zerof = arith::ConstantOp::create(kb, loc, f32, kb.getF32FloatAttr(0));
    Value tid = gpu::ThreadIdOp::create(kb, loc, gpu::Dimension::x);
    Value tidLtK =
        arith::CmpIOp::create(kb, loc, arith::CmpIPredicate::slt, tid, Kv);

    // Load init h → LDS ; barrier.
    {
      auto g = scf::IfOp::create(kb, loc, tidLtK, /*withElse=*/false);
      OpBuilder::InsertionGuard ig(kb);
      kb.setInsertionPointToStart(g.thenBlock());
      Value v = memref::LoadOp::create(kb, loc, INIT, ValueRange{tid});
      memref::StoreOp::create(kb, loc, v, lds, ValueRange{tid});
    }
    gpu::BarrierOp::create(kb, loc);

    auto loop = scf::ForOp::create(kb, loc, c0, ci(trip), c1);
    {
      OpBuilder::InsertionGuard ig(kb);
      kb.setInsertionPointToStart(loop.getBody());
      Value t = loop.getInductionVar();
      Value tK = arith::MulIOp::create(kb, loc, t, Kv);  // t*K (xs row base)
      // h'[tid] = tanh( Σ_k (lds[k]·W[k*K+tid] + XS[t*K+k]·U[k*K+tid]) + B[tid] )
      auto comp = scf::IfOp::create(kb, loc, TypeRange{f32}, tidLtK,
                                    /*withElseRegion=*/true);
      {
        OpBuilder::InsertionGuard ig2(kb);
        kb.setInsertionPointToStart(comp.thenBlock());
        auto red = scf::ForOp::create(kb, loc, c0, cK, c1, ValueRange{zerof});
        {
          OpBuilder::InsertionGuard ig3(kb);
          kb.setInsertionPointToStart(red.getBody());
          Value k = red.getInductionVar(), acc = red.getRegionIterArg(0);
          Value kK = arith::MulIOp::create(kb, loc, k, Kv);  // k*K
          Value off = arith::AddIOp::create(kb, loc, kK, tid);  // k*K + tid
          // lds[k] · W[k*K+tid]
          Value hk = memref::LoadOp::create(kb, loc, lds, ValueRange{k});
          Value wv = memref::LoadOp::create(kb, loc, W, ValueRange{off});
          Value t1 = arith::MulFOp::create(kb, loc, hk, wv);
          // XS[t*K+k] · U[k*K+tid]
          Value xo = arith::AddIOp::create(kb, loc, tK, k);
          Value xk = memref::LoadOp::create(kb, loc, XS, ValueRange{xo});
          Value uv = memref::LoadOp::create(kb, loc, U, ValueRange{off});
          Value t2 = arith::MulFOp::create(kb, loc, xk, uv);
          Value step = arith::AddFOp::create(
              kb, loc, acc, arith::AddFOp::create(kb, loc, t1, t2));
          scf::YieldOp::create(kb, loc, ValueRange{step});
        }
        Value bias = memref::LoadOp::create(kb, loc, B, ValueRange{tid});
        Value pre = arith::AddFOp::create(kb, loc, red.getResult(0), bias);
        Value hp = math::TanhOp::create(kb, loc, pre);
        scf::YieldOp::create(kb, loc, ValueRange{hp});
      }
      {
        OpBuilder::InsertionGuard ig2(kb);
        kb.setInsertionPointToStart(comp.elseBlock());
        scf::YieldOp::create(kb, loc, ValueRange{zerof});
      }
      gpu::BarrierOp::create(kb, loc);  // all reads of the old carry done
      {
        auto g = scf::IfOp::create(kb, loc, tidLtK, /*withElse=*/false);
        OpBuilder::InsertionGuard ig2(kb);
        kb.setInsertionPointToStart(g.thenBlock());
        Value hp = comp.getResult(0);
        memref::StoreOp::create(kb, loc, hp, lds, ValueRange{tid});
        Value yo = arith::AddIOp::create(kb, loc, tK, tid);
        memref::StoreOp::create(kb, loc, hp, YS, ValueRange{yo});
      }
      gpu::BarrierOp::create(kb, loc);  // new carry visible before next step
    }

    {
      auto g = scf::IfOp::create(kb, loc, tidLtK, /*withElse=*/false);
      OpBuilder::InsertionGuard ig(kb);
      kb.setInsertionPointToStart(g.thenBlock());
      Value v = memref::LoadOp::create(kb, loc, lds, ValueRange{tid});
      memref::StoreOp::create(kb, loc, v, COUT, ValueRange{tid});
    }

    kb.setInsertionPointToEnd(&f.getBody().front());
    gpu::ReturnOp::create(kb, loc);
    op->setAttr("tessera.rocm_kernel", b.getStringAttr(kname));
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();
    SymbolTable symTab(module);
    SmallVector<Operation *> scans;
    module.walk([&](Operation *op) {
      if (op->getName().getStringRef() == "tessera.control_scan")
        scans.push_back(op);
    });
    unsigned idx = 0;
    for (Operation *op : scans) {
      int64_t K = 0, trip = 0;
      if (func::FuncOp body = validateScanRnn(op, symTab, K, trip)) {
        (void)body;
        emitKernel(op, K, trip, module, idx++);
      }
    }
  }
};

}  // namespace

std::unique_ptr<Pass>
mlir::tessera_rocm::createGenerateROCMControlScanRnnKernelPass() {
  return std::make_unique<GenerateROCMControlScanRnnKernelPass>();
}
