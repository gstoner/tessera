//===- GenerateROCMSpecAcceptKernel.cpp — speculative-decode accept -------===//
//
// SD1 of CONTROL_FLOW_AND_DEEPSEEK_ACCELERATION_PLAN (ROCm-led, gfx1151). The
// first native ROCm proof of `tessera.spec_accept` — greedy (argmax-match)
// speculative-decode acceptance over a tree of draft paths, run as ONE device
// kernel. For each path p the accepted length is the run of leading positions
// where draft[p,i] == target[p,i]; the op selects the longest-prefix path (first
// wins ties) and writes [accepted_path_idx, accepted_prefix_length, bonus_token]
// (bonus = target[path, length]; target is P×(D+1) so the bonus is always
// defined). Mirrors the tessera.speculative greedy reference (_ref_spec_accept).
//
//   gpu.func @spec_accept(%DRAFT,%TARGET,%OUT: memref<?xi32>) kernel {  // 1 wg
//     %lds = workgroup memref<BDxi32>
//     // thread p<P: lds[p] = run of leading draft[p,i]==target[p,i]
//     if tid<P {
//       (len, _) = scf.for i=0..D iter(acc=0, matching=1) {
//         eq = draft[tid*D+i] == target[tid*(D+1)+i]
//         m' = matching & eq ; acc' = acc + (m' ? 1 : 0) ; yield acc', m'
//       }
//       lds[tid] = len
//     }
//     barrier
//     // thread 0: argmax over P (first wins ties), bonus, write OUT.
//     if tid==0 {
//       (bp, bl) = scf.for p=0..P iter(bp=0, bl=-1) {
//         lp=lds[p] ; gt = lp>bl ; yield (gt?p:bp), (gt?lp:bl)
//       }
//       OUT[0]=bp ; OUT[1]=bl ; OUT[2]=target[bp*(D+1)+bl]
//     }
//   }
//
// One workgroup, P ≤ BD threads. Deterministic (no RNG). The Leviathan /
// rejection-sampling form is SD1-2. Shapes outside P×D draft / P×(D+1) target /
// tensor<3xi32> result are left for the guard.

#include "TesseraROCM/Passes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "llvm/ADT/SmallVector.h"

using namespace mlir;

namespace {

constexpr int64_t BD = 256;

// tessera.spec_accept with draft P×D i32, target P×(D+1) i32, result tensor<3xi32>
// (the op verifier already guarantees this). Fills P, D; requires P ≤ BD.
static bool validateSpecAccept(Operation *op, int64_t &P, int64_t &D) {
  if (op->getNumOperands() != 2 || op->getNumResults() != 1)
    return false;
  auto draftT = dyn_cast<RankedTensorType>(op->getOperand(0).getType());
  auto targetT = dyn_cast<RankedTensorType>(op->getOperand(1).getType());
  auto resT = dyn_cast<RankedTensorType>(op->getResult(0).getType());
  if (!draftT || !targetT || !resT || draftT.getRank() != 2 ||
      targetT.getRank() != 2 || resT.getRank() != 1 ||
      !draftT.getElementType().isInteger(32) ||
      !targetT.getElementType().isInteger(32) ||
      !resT.getElementType().isInteger(32))
    return false;
  P = draftT.getDimSize(0);
  D = draftT.getDimSize(1);
  if (P <= 0 || D <= 0 || P > BD || targetT.getDimSize(0) != P ||
      targetT.getDimSize(1) != D + 1 || resT.getDimSize(0) != 3)
    return false;
  return true;
}

struct GenerateROCMSpecAcceptKernelPass
    : public PassWrapper<GenerateROCMSpecAcceptKernelPass,
                         OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(GenerateROCMSpecAcceptKernelPass)

  StringRef getArgument() const final {
    return "generate-rocm-spec-accept-kernel";
  }
  StringRef getDescription() const final {
    return "SD1: lower a tessera.spec_accept (greedy longest-prefix path + bonus) "
           "to one cooperative-workgroup gpu.func (thread/path match length + "
           "argmax) for gfx1151.";
  }
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<gpu::GPUDialect, scf::SCFDialect, arith::ArithDialect,
                    memref::MemRefDialect>();
  }

  void emitKernel(Operation *op, int64_t P, int64_t D, ModuleOp module,
                  unsigned idx) {
    MLIRContext *ctx = module.getContext();
    OpBuilder b(module.getBodyRegion());
    b.setInsertionPointToEnd(module.getBody());
    Location loc = op->getLoc();
    std::string kname = ("tessera_spec_accept_" + Twine(idx)).str();

    Type i32 = b.getI32Type(), idxTy = b.getIndexType();
    auto memTy = MemRefType::get({ShapedType::kDynamic}, i32);

    auto gpuMod = gpu::GPUModuleOp::create(b, loc, kname + "_mod");
    b.setInsertionPointToStart(&gpuMod.getBodyRegion().front());
    auto fnTy = b.getFunctionType({memTy, memTy, memTy}, {});  // DRAFT, TARGET, OUT
    auto f = gpu::GPUFuncOp::create(b, loc, kname, fnTy);
    f->setAttr(gpu::GPUDialect::getKernelFuncAttrName(), b.getUnitAttr());

    auto ws = gpu::AddressSpaceAttr::get(ctx, gpu::AddressSpace::Workgroup);
    auto ldsT = MemRefType::get({BD}, i32, MemRefLayoutAttrInterface(), ws);
    Value lds = f.addWorkgroupAttribution(ldsT, loc);

    OpBuilder kb(f.getContext());
    kb.setInsertionPointToStart(&f.getBody().front());
    Value DRAFT = f.getArgument(0), TARGET = f.getArgument(1),
          OUT = f.getArgument(2);
    auto ci = [&](int64_t v) { return arith::ConstantIndexOp::create(kb, loc, v); };
    auto cI32 = [&](int32_t v) {
      return arith::ConstantOp::create(kb, loc, i32, kb.getI32IntegerAttr(v));
    };
    Value c0 = ci(0), c1 = ci(1), cP = ci(P), cD = ci(D), cD1 = ci(D + 1);
    Value zeroI = cI32(0), oneI = cI32(1);
    Value tid = gpu::ThreadIdOp::create(kb, loc, gpu::Dimension::x);
    Value tidLtP =
        arith::CmpIOp::create(kb, loc, arith::CmpIPredicate::slt, tid, cP);
    Value tidIs0 =
        arith::CmpIOp::create(kb, loc, arith::CmpIPredicate::eq, tid, c0);
    Value tF = arith::ConstantOp::create(kb, loc, kb.getI1Type(),
                                         kb.getBoolAttr(true));

    // thread p<P: lds[p] = run of leading matches draft[p,i]==target[p,i].
    {
      auto g = scf::IfOp::create(kb, loc, tidLtP, /*withElse=*/false);
      OpBuilder::InsertionGuard ig(kb);
      kb.setInsertionPointToStart(g.thenBlock());
      Value rowD = arith::MulIOp::create(kb, loc, tid, cD);       // tid*D
      Value rowT = arith::MulIOp::create(kb, loc, tid, cD1);      // tid*(D+1)
      auto lp = scf::ForOp::create(kb, loc, c0, cD, c1,
                                   ValueRange{zeroI, tF});  // (acc, matching)
      {
        OpBuilder::InsertionGuard ig2(kb);
        kb.setInsertionPointToStart(lp.getBody());
        Value i = lp.getInductionVar();
        Value acc = lp.getRegionIterArg(0), matching = lp.getRegionIterArg(1);
        Value dv = memref::LoadOp::create(
            kb, loc, DRAFT, ValueRange{arith::AddIOp::create(kb, loc, rowD, i)});
        Value tv = memref::LoadOp::create(
            kb, loc, TARGET, ValueRange{arith::AddIOp::create(kb, loc, rowT, i)});
        Value eq = arith::CmpIOp::create(kb, loc, arith::CmpIPredicate::eq, dv, tv);
        Value m2 = arith::AndIOp::create(kb, loc, matching, eq);  // still matching
        Value inc = arith::SelectOp::create(kb, loc, m2, oneI, zeroI);
        Value acc2 = arith::AddIOp::create(kb, loc, acc, inc);
        scf::YieldOp::create(kb, loc, ValueRange{acc2, m2});
      }
      memref::StoreOp::create(kb, loc, lp.getResult(0), lds, ValueRange{tid});
    }
    gpu::BarrierOp::create(kb, loc);

    // thread 0: argmax over P (first wins ties), bonus, write OUT[0..2].
    {
      auto g = scf::IfOp::create(kb, loc, tidIs0, /*withElse=*/false);
      OpBuilder::InsertionGuard ig(kb);
      kb.setInsertionPointToStart(g.thenBlock());
      Value negOne = cI32(-1);
      auto am = scf::ForOp::create(kb, loc, c0, cP, c1,
                                   ValueRange{zeroI, negOne});  // (bestP, bestLen)
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
      Value bestP = am.getResult(0), bestLen = am.getResult(1);
      // bonus = target[bestP*(D+1) + bestLen].
      Value bpIdx = arith::IndexCastOp::create(kb, loc, idxTy, bestP);
      Value blIdx = arith::IndexCastOp::create(kb, loc, idxTy, bestLen);
      Value off = arith::AddIOp::create(
          kb, loc, arith::MulIOp::create(kb, loc, bpIdx, cD1), blIdx);
      Value bonus = memref::LoadOp::create(kb, loc, TARGET, ValueRange{off});
      memref::StoreOp::create(kb, loc, bestP, OUT, ValueRange{c0});
      memref::StoreOp::create(kb, loc, bestLen, OUT, ValueRange{c1});
      memref::StoreOp::create(kb, loc, bonus, OUT, ValueRange{ci(2)});
    }

    kb.setInsertionPointToEnd(&f.getBody().front());
    gpu::ReturnOp::create(kb, loc);
    op->setAttr("tessera.rocm_kernel", b.getStringAttr(kname));
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();
    SmallVector<Operation *> ops;
    module.walk([&](Operation *op) {
      if (op->getName().getStringRef() == "tessera.spec_accept")
        ops.push_back(op);
    });
    unsigned idx = 0;
    for (Operation *op : ops) {
      int64_t P = 0, D = 0;
      if (validateSpecAccept(op, P, D))
        emitKernel(op, P, D, module, idx++);
    }
  }
};

}  // namespace

std::unique_ptr<Pass>
mlir::tessera_rocm::createGenerateROCMSpecAcceptKernelPass() {
  return std::make_unique<GenerateROCMSpecAcceptKernelPass>();
}
