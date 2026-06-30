//===- GenerateROCMSpecAcceptTreeSampleKernel.cpp — tree Leviathan accept --===//
//
// SD1 (tree multi-path rejection sampling). The device form of
// tessera.speculative.batch_verify: over a draft tree of P paths each D deep, the
// draft token at (p,i) is accepted iff
//   accept_u[p,i] <= exp(target_log_probs[p,i] - draft_log_probs[p,i])
// (the division-free Leviathan rule). Each path's accepted prefix is the run of
// leading accepts; the op picks the longest-prefix path (first wins ties) and
// writes [accepted_path_idx, accepted_prefix_length].
//
// One workgroup, one thread per path (P ≤ BD). Structurally this mirrors the
// greedy spec_accept kernel — the only change is the per-position predicate
// (exp-accept against an explicit uniform, not a token match). Deterministic
// (accept_u is an explicit operand).
//
//   gpu.func @spec_accept_tree_sample(%TLP,%DLP,%AU,%OUT) kernel {  // 1 wg
//     %lds = workgroup memref<BDxi32>
//     if tid<P {                              // path tid's accepted length:
//       (len,_) = scf.for i=0..D iter(acc=0, matching=1) {
//         a = exp(TLP[tid*D+i] - DLP[tid*D+i]) ; ok = AU[tid*D+i] <= a
//         m' = matching & ok ; acc' = acc + (m'?1:0) ; yield acc', m'
//       }
//       lds[tid] = len
//     }
//     barrier
//     if tid==0 { argmax len over P (first wins) → OUT[0]=p, OUT[1]=len }
//   }

#include "TesseraROCM/Passes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "llvm/ADT/SmallVector.h"

using namespace mlir;

namespace {

constexpr int64_t BD = 256;

// target_log_probs / draft_log_probs / accept_u all P×D f32 ; result tensor<2xi32>.
static bool validateTreeSample(Operation *op, int64_t &P, int64_t &D) {
  if (op->getNumOperands() != 3 || op->getNumResults() != 1)
    return false;
  auto t = dyn_cast<RankedTensorType>(op->getOperand(0).getType());
  auto d = dyn_cast<RankedTensorType>(op->getOperand(1).getType());
  auto u = dyn_cast<RankedTensorType>(op->getOperand(2).getType());
  auto r = dyn_cast<RankedTensorType>(op->getResult(0).getType());
  if (!t || !d || !u || !r || t.getRank() != 2 || !t.getElementType().isF32())
    return false;
  P = t.getDimSize(0);
  D = t.getDimSize(1);
  auto same = [&](RankedTensorType x) {
    return x.getRank() == 2 && x.getElementType().isF32() &&
           x.getDimSize(0) == P && x.getDimSize(1) == D;
  };
  if (P <= 0 || D <= 0 || P > BD || !same(d) || !same(u) || r.getRank() != 1 ||
      !r.getElementType().isInteger(32) || r.getDimSize(0) != 2)
    return false;
  return true;
}

struct GenerateROCMSpecAcceptTreeSampleKernelPass
    : public PassWrapper<GenerateROCMSpecAcceptTreeSampleKernelPass,
                         OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(
      GenerateROCMSpecAcceptTreeSampleKernelPass)

  StringRef getArgument() const final {
    return "generate-rocm-spec-accept-tree-sample-kernel";
  }
  StringRef getDescription() const final {
    return "SD1: lower a tessera.spec_accept_tree_sample (tree Leviathan "
           "rejection acceptance: longest accepted prefix over P paths) to one "
           "cooperative-workgroup gpu.func for gfx1151.";
  }
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<gpu::GPUDialect, scf::SCFDialect, arith::ArithDialect,
                    math::MathDialect, memref::MemRefDialect>();
  }

  void emitKernel(Operation *op, int64_t P, int64_t D, ModuleOp module,
                  unsigned idx) {
    MLIRContext *ctx = module.getContext();
    OpBuilder b(module.getBodyRegion());
    b.setInsertionPointToEnd(module.getBody());
    Location loc = op->getLoc();
    std::string kname = ("tessera_spec_accept_tree_sample_" + Twine(idx)).str();

    Type i32 = b.getI32Type(), f32 = b.getF32Type(), i1 = b.getI1Type();
    auto iMem = MemRefType::get({ShapedType::kDynamic}, i32);
    auto fMem = MemRefType::get({ShapedType::kDynamic}, f32);

    auto gpuMod = gpu::GPUModuleOp::create(b, loc, kname + "_mod");
    b.setInsertionPointToStart(&gpuMod.getBodyRegion().front());
    auto fnTy = b.getFunctionType({fMem, fMem, fMem, iMem}, {});  // TLP,DLP,AU,OUT
    auto f = gpu::GPUFuncOp::create(b, loc, kname, fnTy);
    f->setAttr(gpu::GPUDialect::getKernelFuncAttrName(), b.getUnitAttr());

    auto ws = gpu::AddressSpaceAttr::get(ctx, gpu::AddressSpace::Workgroup);
    auto ldsT = MemRefType::get({BD}, i32, MemRefLayoutAttrInterface(), ws);
    Value lds = f.addWorkgroupAttribution(ldsT, loc);

    OpBuilder kb(f.getContext());
    kb.setInsertionPointToStart(&f.getBody().front());
    Value TLP = f.getArgument(0), DLP = f.getArgument(1), AU = f.getArgument(2),
          OUT = f.getArgument(3);
    auto ci = [&](int64_t v) { return arith::ConstantIndexOp::create(kb, loc, v); };
    auto cI = [&](int32_t v) {
      return arith::ConstantOp::create(kb, loc, i32, kb.getI32IntegerAttr(v));
    };
    Value c0 = ci(0), c1 = ci(1), cD = ci(D), cP = ci(P);
    Value zeroI = cI(0), oneI = cI(1);
    Value tid = gpu::ThreadIdOp::create(kb, loc, gpu::Dimension::x);
    Value tidLtP =
        arith::CmpIOp::create(kb, loc, arith::CmpIPredicate::slt, tid, cP);
    Value tidIs0 =
        arith::CmpIOp::create(kb, loc, arith::CmpIPredicate::eq, tid, c0);
    Value tT = arith::ConstantOp::create(kb, loc, i1, kb.getBoolAttr(true));

    // thread p<P: lds[p] = run of leading Leviathan-accepts.
    {
      auto g = scf::IfOp::create(kb, loc, tidLtP, /*withElse=*/false);
      OpBuilder::InsertionGuard ig(kb);
      kb.setInsertionPointToStart(g.thenBlock());
      Value row = arith::MulIOp::create(kb, loc, tid, cD);  // tid*D
      auto lp = scf::ForOp::create(kb, loc, c0, cD, c1, ValueRange{zeroI, tT});
      {
        OpBuilder::InsertionGuard ig2(kb);
        kb.setInsertionPointToStart(lp.getBody());
        Value i = lp.getInductionVar();
        Value acc = lp.getRegionIterArg(0), matching = lp.getRegionIterArg(1);
        Value off = arith::AddIOp::create(kb, loc, row, i);
        Value tl = memref::LoadOp::create(kb, loc, TLP, ValueRange{off});
        Value dl = memref::LoadOp::create(kb, loc, DLP, ValueRange{off});
        Value au = memref::LoadOp::create(kb, loc, AU, ValueRange{off});
        // accept = au <= exp(target_lp - draft_lp).
        Value diff = arith::SubFOp::create(kb, loc, tl, dl);
        Value ap = math::ExpOp::create(kb, loc, diff);
        Value ok =
            arith::CmpFOp::create(kb, loc, arith::CmpFPredicate::OLE, au, ap);
        Value m2 = arith::AndIOp::create(kb, loc, matching, ok);
        Value inc = arith::SelectOp::create(kb, loc, m2, oneI, zeroI);
        Value acc2 = arith::AddIOp::create(kb, loc, acc, inc);
        scf::YieldOp::create(kb, loc, ValueRange{acc2, m2});
      }
      memref::StoreOp::create(kb, loc, lp.getResult(0), lds, ValueRange{tid});
    }
    gpu::BarrierOp::create(kb, loc);

    // thread 0: argmax over P (first wins ties), write OUT[0..1].
    {
      auto g = scf::IfOp::create(kb, loc, tidIs0, /*withElse=*/false);
      OpBuilder::InsertionGuard ig(kb);
      kb.setInsertionPointToStart(g.thenBlock());
      Value negOne = cI(-1);
      auto am = scf::ForOp::create(kb, loc, c0, cP, c1, ValueRange{zeroI, negOne});
      {
        OpBuilder::InsertionGuard ig2(kb);
        kb.setInsertionPointToStart(am.getBody());
        Value p = am.getInductionVar();
        Value bestP = am.getRegionIterArg(0), bestLen = am.getRegionIterArg(1);
        Value lp = memref::LoadOp::create(kb, loc, lds, ValueRange{p});
        Value gt =
            arith::CmpIOp::create(kb, loc, arith::CmpIPredicate::sgt, lp, bestLen);
        Value pI32 = arith::IndexCastOp::create(kb, loc, i32, p);
        Value bp2 = arith::SelectOp::create(kb, loc, gt, pI32, bestP);
        Value bl2 = arith::SelectOp::create(kb, loc, gt, lp, bestLen);
        scf::YieldOp::create(kb, loc, ValueRange{bp2, bl2});
      }
      memref::StoreOp::create(kb, loc, am.getResult(0), OUT, ValueRange{c0});
      memref::StoreOp::create(kb, loc, am.getResult(1), OUT, ValueRange{c1});
    }

    kb.setInsertionPointToEnd(&f.getBody().front());
    gpu::ReturnOp::create(kb, loc);
    op->setAttr("tessera.rocm_kernel", b.getStringAttr(kname));
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();
    SmallVector<Operation *> ops;
    module.walk([&](Operation *op) {
      if (op->getName().getStringRef() == "tessera.spec_accept_tree_sample")
        ops.push_back(op);
    });
    unsigned idx = 0;
    for (Operation *op : ops) {
      int64_t P = 0, D = 0;
      if (validateTreeSample(op, P, D))
        emitKernel(op, P, D, module, idx++);
    }
  }
};

}  // namespace

std::unique_ptr<Pass>
mlir::tessera_rocm::createGenerateROCMSpecAcceptTreeSampleKernelPass() {
  return std::make_unique<GenerateROCMSpecAcceptTreeSampleKernelPass>();
}
