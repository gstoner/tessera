//===- GenerateROCMControlWhileGemvKernel.cpp — cross-element while -------===//
//
// CF4f of the control-flow track (ROCm-led, gfx1151). The first CROSS-ELEMENT
// control_while: a bounded power-iteration / fixed-point loop
//
//     h = h @ W   while  Σ h > eps   (up to max_iters)
//
// over a 1×K carry h and a K×K loop-invariant capture W. Both the body (a GEMV, a
// reduction over the whole carry) AND the continuation cond (Σ h, a reduction)
// are cross-element, so this can't be the per-thread elementwise control_while
// (CF4c-cont). The two structural blockers that deferred it are resolved HERE by
// generating the kernel directly (not via CF2's SCF lowering):
//
//   1. UNIFORM continuation. A matmul body couples every element each step, so
//      the loop can't freeze elements independently — it needs one workgroup-wide
//      decision. Every thread computes the SAME reduction Σ_k lds[k] over the
//      shared carry and the SAME predicate, so the whole workgroup loops the same
//      number of times. That uniformity is what makes the per-iteration
//      gpu.barriers safe (a divergent cond would deadlock the barrier).
//   2. CAPTURE threading. This pass reads the control_while op directly and wires
//      W as a kernel arg (like CF4d-1), so the W capture the elementwise while
//      path lacks is threaded here.
//
//   gpu.func @ctrl_while_gemv(%H,%W,%OUT: memref<?xf32>, %K: index) kernel {
//     %lds = workgroup memref<BDxf32>
//     if tid<K { lds[tid] = H[tid] } ; barrier
//     %i = scf.while (%i = 0) {                       // before:
//       %s   = Σ_k lds[k]                             //   uniform reduction
//       %p   = (%i < max) AND (%s > eps)              //   uniform predicate
//       scf.condition(%p) %i
//     } do { ^bb(%i):                                 // after (GEMV step):
//       %o = (tid<K) ? Σ_k lds[k]·W[k*K+tid] : 0      //   h @ W
//       barrier ; if tid<K { lds[tid]=%o } ; barrier  //   publish new carry
//       scf.yield %i+1
//     }
//     if tid<K { OUT[tid] = lds[tid] }
//   }
//
// Body must be matmul(h,W) (no transpose); cond must be reduce(h){kind="sum"};
// eps is the discardable `tessera.while_cond_eps` attr (default 0). Single-tile
// (K ≤ BD). Anything else is left for the CF0 guard / SCF.

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

// control_while whose body is matmul(h, W) (GEMV, W capture) and whose cond is
// reduce(h){kind="sum"} (the uniform continuation reduction). operands =
// [h(1×K), W(K×K)]; carry_arg_index 0; max_iters > 0; one result == h. Fills
// K, maxIters, eps. Returns true.
static bool validateWhileGemv(Operation *op, SymbolTable &symTab, int64_t &K,
                              int64_t &maxIters, float &eps) {
  auto bodySym = op->getAttrOfType<FlatSymbolRefAttr>("body");
  auto condSym = op->getAttrOfType<FlatSymbolRefAttr>("cond");
  auto carryA = op->getAttrOfType<IntegerAttr>("carry_arg_index");
  auto maxA = op->getAttrOfType<IntegerAttr>("max_iters");
  if (!bodySym || !condSym || !carryA || !maxA || carryA.getInt() != 0 ||
      maxA.getInt() <= 0 || op->getNumOperands() != 2 ||
      op->getNumResults() != 1)
    return false;
  maxIters = maxA.getInt();
  eps = 0.0f;
  if (auto e = op->getAttrOfType<FloatAttr>("tessera.while_cond_eps"))
    eps = static_cast<float>(e.getValueAsDouble());

  auto hT = dyn_cast<RankedTensorType>(op->getOperand(0).getType());
  if (!hT || hT.getRank() != 2 || hT.getDimSize(0) != 1 ||
      !hT.getElementType().isF32())
    return false;
  K = hT.getDimSize(1);
  if (K <= 0 || K > BD || !is1xK(op->getResult(0).getType(), K))
    return false;
  auto wT = dyn_cast<RankedTensorType>(op->getOperand(1).getType());
  if (!wT || wT.getRank() != 2 || wT.getDimSize(0) != K ||
      wT.getDimSize(1) != K || !wT.getElementType().isF32())
    return false;

  // body @b: (h, W) -> matmul(h, W) no-transpose ; return.
  auto bodyF = dyn_cast_or_null<func::FuncOp>(
      symTab.lookupNearestSymbolFrom(op, bodySym.getAttr()));
  auto condF = dyn_cast_or_null<func::FuncOp>(
      symTab.lookupNearestSymbolFrom(op, condSym.getAttr()));
  if (!bodyF || !condF || bodyF == condF || bodyF.isExternal() ||
      condF.isExternal())
    return false;
  FunctionType bf = bodyF.getFunctionType();
  if (bf.getNumInputs() != 2 || bf.getNumResults() != 1 ||
      !is1xK(bf.getInput(0), K) || !is1xK(bf.getResult(0), K))
    return false;
  {
    Block &blk = bodyF.getBody().front();
    auto it = blk.begin();
    if (it == blk.end() || it->getName().getStringRef() != "tessera.matmul" ||
        it->getNumOperands() != 2 || it->getOperand(0) != blk.getArgument(0) ||
        it->getOperand(1) != blk.getArgument(1) || it->getNumResults() != 1)
      return false;
    auto ta = it->getAttrOfType<BoolAttr>("transposeA");
    auto tb = it->getAttrOfType<BoolAttr>("transposeB");
    if ((ta && ta.getValue()) || (tb && tb.getValue()))
      return false;
    auto ret = dyn_cast<func::ReturnOp>(std::next(it));
    if (!ret || ret.getNumOperands() != 1 || ret.getOperand(0) != it->getResult(0))
      return false;
  }
  // cond @c: (h) -> reduce(h){kind="sum"} ; return.
  FunctionType cf = condF.getFunctionType();
  if (cf.getNumInputs() != 1 || cf.getNumResults() != 1 || !is1xK(cf.getInput(0), K))
    return false;
  {
    Block &blk = condF.getBody().front();
    auto it = blk.begin();
    if (it == blk.end() || it->getName().getStringRef() != "tessera.reduce" ||
        it->getNumOperands() != 1 || it->getOperand(0) != blk.getArgument(0) ||
        it->getNumResults() != 1)
      return false;
    auto kind = it->getAttrOfType<StringAttr>("kind");
    if (!kind || kind.getValue() != "sum")
      return false;
    auto ret = dyn_cast<func::ReturnOp>(std::next(it));
    if (!ret || ret.getNumOperands() != 1 || ret.getOperand(0) != it->getResult(0))
      return false;
  }
  return true;
}

struct GenerateROCMControlWhileGemvKernelPass
    : public PassWrapper<GenerateROCMControlWhileGemvKernelPass,
                         OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(
      GenerateROCMControlWhileGemvKernelPass)

  StringRef getArgument() const final {
    return "generate-rocm-control-while-gemv-kernel";
  }
  StringRef getDescription() const final {
    return "CF4f: lower a cross-element control_while (power iteration h=h@W "
           "while Σh > eps, GEMV body + W capture + uniform reduction cond) to "
           "one cooperative-workgroup gpu.func for gfx1151.";
  }
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<gpu::GPUDialect, scf::SCFDialect, arith::ArithDialect,
                    memref::MemRefDialect, func::FuncDialect>();
  }

  void emitKernel(Operation *op, int64_t K, int64_t maxIters, float eps,
                  ModuleOp module, unsigned idx) {
    MLIRContext *ctx = module.getContext();
    OpBuilder b(module.getBodyRegion());
    b.setInsertionPointToEnd(module.getBody());
    Location loc = op->getLoc();
    std::string kname = ("tessera_control_while_gemv_" + Twine(idx)).str();

    Type f32 = b.getF32Type(), idxTy = b.getIndexType();
    auto memTy = MemRefType::get({ShapedType::kDynamic}, f32);

    auto gpuMod = gpu::GPUModuleOp::create(b, loc, kname + "_mod");
    b.setInsertionPointToStart(&gpuMod.getBodyRegion().front());
    auto fnTy = b.getFunctionType({memTy, memTy, memTy, idxTy}, {});  // H,W,OUT,K
    auto f = gpu::GPUFuncOp::create(b, loc, kname, fnTy);
    f->setAttr(gpu::GPUDialect::getKernelFuncAttrName(), b.getUnitAttr());

    auto ws = gpu::AddressSpaceAttr::get(ctx, gpu::AddressSpace::Workgroup);
    auto ldsT = MemRefType::get({BD}, f32, MemRefLayoutAttrInterface(), ws);
    Value lds = f.addWorkgroupAttribution(ldsT, loc);

    OpBuilder kb(f.getContext());
    kb.setInsertionPointToStart(&f.getBody().front());
    Value H = f.getArgument(0), W = f.getArgument(1), OUT = f.getArgument(2),
          Kv = f.getArgument(3);
    auto ci = [&](int64_t v) { return arith::ConstantIndexOp::create(kb, loc, v); };
    auto cf = [&](float v) {
      return arith::ConstantOp::create(kb, loc, f32, kb.getF32FloatAttr(v));
    };
    Value c0 = ci(0), c1 = ci(1), cK = ci(K), cMax = ci(maxIters);
    Value epsC = cf(eps), zerof = cf(0.0f);
    Value tid = gpu::ThreadIdOp::create(kb, loc, gpu::Dimension::x);
    Value tidLtK =
        arith::CmpIOp::create(kb, loc, arith::CmpIPredicate::slt, tid, Kv);

    // Load h → LDS ; barrier.
    {
      auto g = scf::IfOp::create(kb, loc, tidLtK, /*withElse=*/false);
      OpBuilder::InsertionGuard ig(kb);
      kb.setInsertionPointToStart(g.thenBlock());
      Value v = memref::LoadOp::create(kb, loc, H, ValueRange{tid});
      memref::StoreOp::create(kb, loc, v, lds, ValueRange{tid});
    }
    gpu::BarrierOp::create(kb, loc);

    // Σ_k lds[k] — a serial reduction every thread computes identically (uniform).
    auto sumLds = [&](OpBuilder &rb) -> Value {
      auto red = scf::ForOp::create(rb, loc, c0, cK, c1, ValueRange{zerof});
      OpBuilder::InsertionGuard ig(rb);
      rb.setInsertionPointToStart(red.getBody());
      Value k = red.getInductionVar(), acc = red.getRegionIterArg(0);
      Value lk = memref::LoadOp::create(rb, loc, lds, ValueRange{k});
      scf::YieldOp::create(rb, loc,
                           ValueRange{arith::AddFOp::create(rb, loc, acc, lk)});
      return red.getResult(0);
    };

    auto whileOp = scf::WhileOp::create(kb, loc, TypeRange{idxTy}, ValueRange{c0});
    // before: pred = (i < max) AND (Σ lds > eps).
    {
      Block *before = kb.createBlock(&whileOp.getBefore(), {}, {idxTy}, {loc});
      OpBuilder::InsertionGuard ig(kb);
      kb.setInsertionPointToStart(before);
      Value i = before->getArgument(0);
      Value ilt = arith::CmpIOp::create(kb, loc, arith::CmpIPredicate::slt, i, cMax);
      Value s = sumLds(kb);
      Value sgt = arith::CmpFOp::create(kb, loc, arith::CmpFPredicate::OGT, s, epsC);
      Value pred = arith::AndIOp::create(kb, loc, ilt, sgt);
      scf::ConditionOp::create(kb, loc, pred, ValueRange{i});
    }
    // after: GEMV h = h @ W ; barrier handoff ; i++.
    {
      Block *after = kb.createBlock(&whileOp.getAfter(), {}, {idxTy}, {loc});
      OpBuilder::InsertionGuard ig(kb);
      kb.setInsertionPointToStart(after);
      Value i = after->getArgument(0);
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
          Value wo = arith::AddIOp::create(
              kb, loc, arith::MulIOp::create(kb, loc, k, Kv), tid);
          Value wv = memref::LoadOp::create(kb, loc, W, ValueRange{wo});
          Value prod = arith::MulFOp::create(kb, loc, lk, wv);
          scf::YieldOp::create(
              kb, loc, ValueRange{arith::AddFOp::create(kb, loc, acc, prod)});
        }
        scf::YieldOp::create(kb, loc, ValueRange{red.getResult(0)});
      }
      {
        OpBuilder::InsertionGuard ig2(kb);
        kb.setInsertionPointToStart(comp.elseBlock());
        scf::YieldOp::create(kb, loc, ValueRange{zerof});
      }
      gpu::BarrierOp::create(kb, loc);  // all GEMV reads of the old carry done
      {
        auto g = scf::IfOp::create(kb, loc, tidLtK, /*withElse=*/false);
        OpBuilder::InsertionGuard ig2(kb);
        kb.setInsertionPointToStart(g.thenBlock());
        memref::StoreOp::create(kb, loc, comp.getResult(0), lds, ValueRange{tid});
      }
      gpu::BarrierOp::create(kb, loc);  // new carry visible before next reduction
      Value i1 = arith::AddIOp::create(kb, loc, i, c1);
      scf::YieldOp::create(kb, loc, ValueRange{i1});
    }

    kb.setInsertionPointAfter(whileOp);
    {
      auto g = scf::IfOp::create(kb, loc, tidLtK, /*withElse=*/false);
      OpBuilder::InsertionGuard ig(kb);
      kb.setInsertionPointToStart(g.thenBlock());
      Value v = memref::LoadOp::create(kb, loc, lds, ValueRange{tid});
      memref::StoreOp::create(kb, loc, v, OUT, ValueRange{tid});
    }

    kb.setInsertionPointToEnd(&f.getBody().front());
    gpu::ReturnOp::create(kb, loc);
    op->setAttr("tessera.rocm_kernel", b.getStringAttr(kname));
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();
    SymbolTable symTab(module);
    SmallVector<Operation *> whiles;
    module.walk([&](Operation *op) {
      if (op->getName().getStringRef() == "tessera.control_while")
        whiles.push_back(op);
    });
    unsigned idx = 0;
    for (Operation *op : whiles) {
      int64_t K = 0, maxIters = 0;
      float eps = 0.0f;
      if (validateWhileGemv(op, symTab, K, maxIters, eps))
        emitKernel(op, K, maxIters, eps, module, idx++);
    }
  }
};

}  // namespace

std::unique_ptr<Pass>
mlir::tessera_rocm::createGenerateROCMControlWhileGemvKernelPass() {
  return std::make_unique<GenerateROCMControlWhileGemvKernelPass>();
}
