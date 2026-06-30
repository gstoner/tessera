//===- GenerateROCMControlScanGemvKernel.cpp — linear SSM scan kernel -----===//
//
// CF4e-2 of the control-flow track (ROCm-led, gfx1151). The first scan with a
// CROSS-ELEMENT body AND a CAPTURE: the canonical linear state-space / linear-
// attention state update
//
//     h_t = h_{t-1} @ W + x_t ,   y_t = h_t
//
// over a 1×K carry h, a K×K loop-invariant capture W, and a per-step input slice
// x_t (the scan's xs). The `h @ W` is a GEMV — a reduction over the whole carry —
// so it can't be the per-thread elementwise scan (CF4e-1); it needs the CF4d-1
// cooperative-workgroup GEMV substrate (h in LDS, barrier per step), combined
// with the CF4e-1 per-step xs-in / stacked-ys-out streaming.
//
//   gpu.func @ctrl_scan_gemv(%INIT,%XS,%W,%COUT,%YS: memref<?xf32>, %K) kernel {
//     %lds = workgroup memref<BDxf32>
//     if tid<K { lds[tid] = INIT[tid] } ; barrier
//     scf.for %t = 0 to TRIP {
//       if tid<K {                                  // GEMV + per-step input:
//         acc = Σ_k lds[k]·W[k*K+tid]                //   h @ W
//         h'  = acc + XS[t*K+tid]                    //   + x_t
//       }
//       barrier                                     // all GEMV reads of lds done
//       if tid<K { lds[tid]=h' ; YS[t*K+tid]=h' }   // new carry + stacked output
//       barrier
//     }
//     if tid<K { COUT[tid]=lds[tid] }               // final carry
//   }
//
// Single-tile (K ≤ BD). TRIP baked from the op's `trip` attr. Bodies/shapes
// outside this exact `matmul(h,W)+x` form are left for the CF0 guard / SCF.

#include "TesseraROCM/Passes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
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
// The scan stream / stacked output keep the body's per-step element shape: with
// a 1×K carry/y, `xs`/`ys` are `(trip, 1, K)` — i.e. `(trip, *y.shape)` per the
// control_scan contract, NOT a flattened `(trip, K)`. The kernel's row-major
// flattening (`t*K + tid`) is identical since the middle dim is 1.
static bool is3D(Type t, int64_t d0, int64_t d1, int64_t d2) {
  auto r = dyn_cast<RankedTensorType>(t);
  return r && r.getRank() == 3 && r.getDimSize(0) == d0 &&
         r.getDimSize(1) == d1 && r.getDimSize(2) == d2 &&
         r.getElementType().isF32();
}

// control_scan with one capture W whose body is the linear SSM step
// `(h, x, W) -> (h@W + x, same)`. operands = [init(1×K), xs(trip×1×K), W(K×K)];
// results = [carry(1×K), ys(trip×1×K)]. Fills K, trip. Returns the body, or null.
static func::FuncOp validateScanGemv(Operation *op, SymbolTable &symTab,
                                     int64_t &K, int64_t &trip) {
  auto bodySym = op->getAttrOfType<FlatSymbolRefAttr>("body");
  auto tripA = op->getAttrOfType<IntegerAttr>("trip");
  auto carryA = op->getAttrOfType<IntegerAttr>("carry_arg_index");
  if (!bodySym || !tripA || !carryA || carryA.getInt() != 0 ||
      tripA.getInt() <= 0 || op->getNumOperands() != 3 ||
      op->getNumResults() != 2)
    return nullptr;  // exactly init + xs + one capture
  trip = tripA.getInt();

  // K from the init carry (a 1×K f32 tensor).
  auto initT = dyn_cast<RankedTensorType>(op->getOperand(0).getType());
  if (!initT || initT.getRank() != 2 || initT.getDimSize(0) != 1 ||
      !initT.getElementType().isF32())
    return nullptr;
  K = initT.getDimSize(1);
  if (!is3D(op->getOperand(1).getType(), trip, 1, K) ||  // xs: trip×1×K
      !is2D(op->getOperand(2).getType(), K, K) ||         // W: K×K
      !is1xK(op->getResult(0).getType(), K) ||            // carry: 1×K
      !is3D(op->getResult(1).getType(), trip, 1, K))      // ys: trip×1×K
    return nullptr;
  if (K <= 0 || K > BD)
    return nullptr;

  auto fn = dyn_cast_or_null<func::FuncOp>(
      symTab.lookupNearestSymbolFrom(op, bodySym.getAttr()));
  if (!fn || fn.isExternal())
    return nullptr;
  FunctionType ft = fn.getFunctionType();
  if (ft.getNumInputs() != 3 || ft.getNumResults() != 2 ||
      !is1xK(ft.getInput(0), K) || !is1xK(ft.getInput(1), K) ||
      !is2D(ft.getInput(2), K, K) || !is1xK(ft.getResult(0), K) ||
      !is1xK(ft.getResult(1), K))
    return nullptr;
  // body must be exactly: %m = matmul(h, W) ; %s = add(%m, x) ; return %s, %s
  Block &blk = fn.getBody().front();
  Value h = blk.getArgument(0), x = blk.getArgument(1), W = blk.getArgument(2);
  auto it = blk.begin();
  if (it == blk.end() || it->getName().getStringRef() != "tessera.matmul" ||
      it->getNumOperands() != 2 || it->getOperand(0) != h ||
      it->getOperand(1) != W || it->getNumResults() != 1)
    return nullptr;
  auto ta = it->getAttrOfType<BoolAttr>("transposeA");
  auto tb = it->getAttrOfType<BoolAttr>("transposeB");
  if ((ta && ta.getValue()) || (tb && tb.getValue()))
    return nullptr;
  Value mm = it->getResult(0);
  auto add = std::next(it);
  if (add == blk.end() || add->getName().getStringRef() != "tessera.add" ||
      add->getNumOperands() != 2 || add->getNumResults() != 1 ||
      !((add->getOperand(0) == mm && add->getOperand(1) == x) ||
        (add->getOperand(0) == x && add->getOperand(1) == mm)))
    return nullptr;
  Value s = add->getResult(0);
  auto ret = dyn_cast<func::ReturnOp>(std::next(add));
  if (!ret || ret.getNumOperands() != 2 || ret.getOperand(0) != s ||
      ret.getOperand(1) != s)
    return nullptr;
  return fn;
}

struct GenerateROCMControlScanGemvKernelPass
    : public PassWrapper<GenerateROCMControlScanGemvKernelPass,
                         OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(
      GenerateROCMControlScanGemvKernelPass)

  StringRef getArgument() const final {
    return "generate-rocm-control-scan-gemv-kernel";
  }
  StringRef getDescription() const final {
    return "CF4e-2: lower a linear state-space scan (h_t = h_{t-1} @ W + x_t, a "
           "GEMV body + a W capture + per-step xs) to one cooperative-workgroup "
           "gpu.func (h in LDS; barrier per step; stacked ys) for gfx1151.";
  }
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<gpu::GPUDialect, scf::SCFDialect, arith::ArithDialect,
                    memref::MemRefDialect, func::FuncDialect>();
  }

  void emitKernel(Operation *op, int64_t K, int64_t trip, ModuleOp module,
                  unsigned idx) {
    MLIRContext *ctx = module.getContext();
    OpBuilder b(module.getBodyRegion());
    b.setInsertionPointToEnd(module.getBody());
    Location loc = op->getLoc();
    std::string kname = ("tessera_control_scan_gemv_" + Twine(idx)).str();

    Type f32 = b.getF32Type(), idxTy = b.getIndexType();
    auto memTy = MemRefType::get({ShapedType::kDynamic}, f32);

    auto gpuMod = gpu::GPUModuleOp::create(b, loc, kname + "_mod");
    b.setInsertionPointToStart(&gpuMod.getBodyRegion().front());
    // (INIT, XS, W, COUT, YS : memref<?xf32>, K : index)
    auto fnTy =
        b.getFunctionType({memTy, memTy, memTy, memTy, memTy, idxTy}, {});
    auto f = gpu::GPUFuncOp::create(b, loc, kname, fnTy);
    f->setAttr(gpu::GPUDialect::getKernelFuncAttrName(), b.getUnitAttr());

    auto ws = gpu::AddressSpaceAttr::get(ctx, gpu::AddressSpace::Workgroup);
    auto ldsT = MemRefType::get({BD}, f32, MemRefLayoutAttrInterface(), ws);
    Value lds = f.addWorkgroupAttribution(ldsT, loc);

    OpBuilder kb(f.getContext());
    kb.setInsertionPointToStart(&f.getBody().front());
    Value INIT = f.getArgument(0), XS = f.getArgument(1), W = f.getArgument(2),
          COUT = f.getArgument(3), YS = f.getArgument(4), Kv = f.getArgument(5);
    auto ci = [&](int64_t v) { return arith::ConstantIndexOp::create(kb, loc, v); };
    Value c0 = ci(0), c1 = ci(1), cK = ci(K), zerof = arith::ConstantOp::create(
                                                  kb, loc, f32,
                                                  kb.getF32FloatAttr(0.0f));
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
      // h'[tid] = Σ_k lds[k]·W[k*K+tid] + XS[t*K+tid]   (tid < K).
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
          Value lk = memref::LoadOp::create(kb, loc, lds, ValueRange{k});
          // W[k*K + tid]
          Value wo = arith::AddIOp::create(
              kb, loc, arith::MulIOp::create(kb, loc, k, Kv), tid);
          Value wv = memref::LoadOp::create(kb, loc, W, ValueRange{wo});
          Value prod = arith::MulFOp::create(kb, loc, lk, wv);
          scf::YieldOp::create(
              kb, loc, ValueRange{arith::AddFOp::create(kb, loc, acc, prod)});
        }
        // + XS[t*K + tid]
        Value xo = arith::AddIOp::create(
            kb, loc, arith::MulIOp::create(kb, loc, t, Kv), tid);
        Value xt = memref::LoadOp::create(kb, loc, XS, ValueRange{xo});
        Value hp = arith::AddFOp::create(kb, loc, red.getResult(0), xt);
        scf::YieldOp::create(kb, loc, ValueRange{hp});
      }
      {
        OpBuilder::InsertionGuard ig2(kb);
        kb.setInsertionPointToStart(comp.elseBlock());
        scf::YieldOp::create(kb, loc, ValueRange{zerof});
      }
      gpu::BarrierOp::create(kb, loc);  // all GEMV reads of lds done
      {
        auto g = scf::IfOp::create(kb, loc, tidLtK, /*withElse=*/false);
        OpBuilder::InsertionGuard ig2(kb);
        kb.setInsertionPointToStart(g.thenBlock());
        Value hp = comp.getResult(0);
        memref::StoreOp::create(kb, loc, hp, lds, ValueRange{tid});  // new carry
        Value yo = arith::AddIOp::create(
            kb, loc, arith::MulIOp::create(kb, loc, t, Kv), tid);
        memref::StoreOp::create(kb, loc, hp, YS, ValueRange{yo});  // stacked ys
      }
      gpu::BarrierOp::create(kb, loc);  // new carry visible before next GEMV
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
      if (func::FuncOp body = validateScanGemv(op, symTab, K, trip)) {
        (void)body;
        emitKernel(op, K, trip, module, idx++);
      }
    }
  }
};

}  // namespace

std::unique_ptr<Pass>
mlir::tessera_rocm::createGenerateROCMControlScanGemvKernelPass() {
  return std::make_unique<GenerateROCMControlScanGemvKernelPass>();
}
