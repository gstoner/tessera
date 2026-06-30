//===- GenerateROCMControlForGemvKernel.cpp — cross-element control loop --===//
//
// CF4d-1 of docs/audit/roadmap/CF_CROSS_ELEMENT_PLAN.md (ROCm-led, gfx1151).
// The first CROSS-ELEMENT device control-flow kernel: a control_for whose body
// is a GEMV recurrence `carry = carry @ W` (carry a 1xK vector, W a KxK matrix).
//
// The shipped elementwise kernels are per-thread (one thread owns one carry
// element); a GEMV can't be — each output o[j] = Σ_k carry[k]·W[k][j] needs the
// WHOLE carry. So this is a cooperative-workgroup kernel: one workgroup holds
// the carry in LDS, thread j computes o[j] by a serial dot product over the
// LDS-resident carry, and a gpu.barrier separates loop iterations. The whole
// bounded loop is one dispatch → carry @ W^max_iters.
//
//   gpu.func @ctrl_for_gemv(%CARRY, %W, %OUT: memref<?xf32>, %K: index) kernel {
//     %lds = workgroup memref<BDxf32>
//     if tid < K { lds[tid] = CARRY[tid] }   ; barrier
//     scf.for %i = 0 to max_iters {
//       %acc = scf.if tid<K { Σ_k lds[k]*W[k*K+tid] } else { 0 }
//       barrier ; if tid<K { lds[tid] = %acc } ; barrier
//     }
//     if tid < K { OUT[tid] = lds[tid] }
//   }
//
// Single-tile: K ≤ BD (one workgroup, no cross-workgroup sync). A larger carry
// (multi-tile) needs a grid-wide barrier — CF4d-4. Bodies/shapes outside this
// exact form are left untouched for the CF0 guard / SCF.

#include "TesseraROCM/Passes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/SmallVector.h"

using namespace mlir;

namespace {

constexpr int64_t BD = 256;  // threads per workgroup (one per carry element)

// The body must be exactly one tessera.matmul(%carry, %W) with no transpose,
// returning the carry. Returns K (the carry width) on success, or -1.
static int64_t validateGemvBody(Operation *op, SymbolTable &symTab) {
  auto bodySym = op->getAttrOfType<FlatSymbolRefAttr>("body");
  auto carryA = op->getAttrOfType<IntegerAttr>("carry_arg_index");
  // operands = [carry (idx 0), W (loop-invariant capture)]; one result.
  if (!bodySym || !carryA || carryA.getInt() != 0 ||
      op->getNumOperands() != 2 || op->getNumResults() != 1)
    return -1;
  auto fn = dyn_cast_or_null<func::FuncOp>(
      symTab.lookupNearestSymbolFrom(op, bodySym.getAttr()));
  if (!fn || fn.isExternal())
    return -1;
  FunctionType ft = fn.getFunctionType();
  if (ft.getNumInputs() != 2 || ft.getNumResults() != 1)
    return -1;

  auto carryT = dyn_cast<RankedTensorType>(ft.getInput(0));
  auto wT = dyn_cast<RankedTensorType>(ft.getInput(1));
  auto resT = dyn_cast<RankedTensorType>(ft.getResult(0));
  if (!carryT || !wT || !resT || !carryT.getElementType().isF32() ||
      !wT.getElementType().isF32())
    return -1;
  // carry = (1, K) ; W = (K, K) ; result == carry ; K ≤ BD (single tile).
  if (carryT.getRank() != 2 || carryT.getDimSize(0) != 1)
    return -1;
  int64_t K = carryT.getDimSize(1);
  if (K <= 0 || K > BD || wT.getRank() != 2 || wT.getDimSize(0) != K ||
      wT.getDimSize(1) != K || resT != carryT)
    return -1;
  // The op's operand types must match the @body signature.
  if (op->getOperand(0).getType() != carryT ||
      op->getOperand(1).getType() != wT ||
      op->getResult(0).getType() != carryT)
    return -1;

  // Body is exactly: %0 = matmul(%arg0, %arg1) [no transpose] ; return %0.
  Block &blk = fn.getBody().front();
  auto it = blk.begin();
  if (it == blk.end() || it->getName().getStringRef() != "tessera.matmul")
    return -1;
  Operation &mm = *it;
  if (mm.getNumOperands() != 2 || mm.getOperand(0) != blk.getArgument(0) ||
      mm.getOperand(1) != blk.getArgument(1) || mm.getNumResults() != 1)
    return -1;
  auto ta = mm.getAttrOfType<BoolAttr>("transposeA");
  auto tb = mm.getAttrOfType<BoolAttr>("transposeB");
  if ((ta && ta.getValue()) || (tb && tb.getValue()))
    return -1;  // transposed GEMV is a later variant
  auto ret = dyn_cast<func::ReturnOp>(std::next(it));
  if (!ret || ret.getNumOperands() != 1 || ret.getOperand(0) != mm.getResult(0))
    return -1;
  return K;
}

struct GenerateROCMControlForGemvKernelPass
    : public PassWrapper<GenerateROCMControlForGemvKernelPass,
                         OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(
      GenerateROCMControlForGemvKernelPass)

  StringRef getArgument() const final {
    return "generate-rocm-control-for-gemv-kernel";
  }
  StringRef getDescription() const final {
    return "CF4d-1: lower a GEMV-recurrence control_for (carry @ W, carry 1xK / "
           "W KxK) to one cooperative-workgroup gpu.func (carry in LDS; "
           "per-thread dot product; barrier per iteration) for gfx1151.";
  }
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<gpu::GPUDialect, scf::SCFDialect, arith::ArithDialect,
                    memref::MemRefDialect, func::FuncDialect>();
  }

  void emitKernel(Operation *op, int64_t K, ModuleOp module, unsigned idx) {
    auto startA = op->getAttrOfType<IntegerAttr>("start");
    auto stopA = op->getAttrOfType<IntegerAttr>("stop");
    auto stepA = op->getAttrOfType<IntegerAttr>("step");
    if (!startA || !stopA || !stepA)
      return;

    MLIRContext *ctx = module.getContext();
    OpBuilder b(module.getBodyRegion());
    b.setInsertionPointToEnd(module.getBody());
    Location loc = op->getLoc();
    std::string kname = ("tessera_control_for_gemv_" + Twine(idx)).str();

    Type f32 = b.getF32Type();
    Type idxTy = b.getIndexType();
    auto memTy = MemRefType::get({ShapedType::kDynamic}, f32);

    auto gpuMod = gpu::GPUModuleOp::create(b, loc, kname + "_mod");
    b.setInsertionPointToStart(&gpuMod.getBodyRegion().front());
    // (CARRY, W, OUT : memref<?xf32>, K : index)
    auto fnTy = b.getFunctionType({memTy, memTy, memTy, idxTy}, {});
    auto f = gpu::GPUFuncOp::create(b, loc, kname, fnTy);
    f->setAttr(gpu::GPUDialect::getKernelFuncAttrName(), b.getUnitAttr());

    // LDS carry buffer (BD elements; K ≤ BD active).
    auto ws = gpu::AddressSpaceAttr::get(ctx, gpu::AddressSpace::Workgroup);
    auto ldsT = MemRefType::get({BD}, f32, MemRefLayoutAttrInterface(), ws);
    Value lds = f.addWorkgroupAttribution(ldsT, loc);

    OpBuilder kb(f.getContext());
    kb.setInsertionPointToStart(&f.getBody().front());
    Value CARRY = f.getArgument(0), W = f.getArgument(1), OUT = f.getArgument(2),
          Kv = f.getArgument(3);
    auto ci = [&](int64_t v) { return arith::ConstantIndexOp::create(kb, loc, v); };
    Value c0 = ci(0), c1 = ci(1), cK = ci(K);
    Value tid = gpu::ThreadIdOp::create(kb, loc, gpu::Dimension::x);
    Value zerof = arith::ConstantOp::create(kb, loc, f32, kb.getF32FloatAttr(0.0f));
    Value tidLtK =
        arith::CmpIOp::create(kb, loc, arith::CmpIPredicate::slt, tid, Kv);

    // Load carry → LDS, then barrier.
    {
      auto g = scf::IfOp::create(kb, loc, tidLtK, /*withElse=*/false);
      OpBuilder::InsertionGuard ig(kb);
      kb.setInsertionPointToStart(g.thenBlock());
      Value v = memref::LoadOp::create(kb, loc, CARRY, ValueRange{tid});
      memref::StoreOp::create(kb, loc, v, lds, ValueRange{tid});
    }
    gpu::BarrierOp::create(kb, loc);

    Value lb = ci(startA.getInt()), ub = ci(stopA.getInt()),
          st = ci(stepA.getInt());
    auto loop = scf::ForOp::create(kb, loc, lb, ub, st);
    {
      OpBuilder::InsertionGuard ig(kb);
      kb.setInsertionPointToStart(loop.getBody());
      // acc = Σ_k lds[k] * W[k*K + tid]  (only for tid < K; else 0).
      auto comp = scf::IfOp::create(kb, loc, TypeRange{f32}, tidLtK,
                                    /*withElseRegion=*/true);
      {
        OpBuilder::InsertionGuard ig2(kb);
        kb.setInsertionPointToStart(comp.thenBlock());
        auto red = scf::ForOp::create(kb, loc, c0, cK, c1, ValueRange{zerof});
        {
          OpBuilder::InsertionGuard ig3(kb);
          kb.setInsertionPointToStart(red.getBody());
          Value k = red.getInductionVar();
          Value acc = red.getRegionIterArg(0);
          Value ck = memref::LoadOp::create(kb, loc, lds, ValueRange{k});
          // W[k*K + tid]
          Value off = arith::AddIOp::create(
              kb, loc, arith::MulIOp::create(kb, loc, k, cK), tid);
          Value wv = memref::LoadOp::create(kb, loc, W, ValueRange{off});
          Value prod = arith::MulFOp::create(kb, loc, ck, wv);
          scf::YieldOp::create(kb, loc,
                               ValueRange{arith::AddFOp::create(kb, loc, acc,
                                                                prod)});
        }
        scf::YieldOp::create(kb, loc, ValueRange{red.getResult(0)});
      }
      {
        OpBuilder::InsertionGuard ig2(kb);
        kb.setInsertionPointToStart(comp.elseBlock());
        scf::YieldOp::create(kb, loc, ValueRange{zerof});
      }
      gpu::BarrierOp::create(kb, loc);  // all reads of lds done
      {
        auto g = scf::IfOp::create(kb, loc, tidLtK, /*withElse=*/false);
        OpBuilder::InsertionGuard ig2(kb);
        kb.setInsertionPointToStart(g.thenBlock());
        memref::StoreOp::create(kb, loc, comp.getResult(0), lds,
                                ValueRange{tid});
      }
      gpu::BarrierOp::create(kb, loc);  // new carry visible before next iter
    }

    // OUT[tid] = lds[tid]  (tid < K).
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
    SmallVector<Operation *> fors;
    module.walk([&](Operation *op) {
      if (op->getName().getStringRef() == "tessera.control_for")
        fors.push_back(op);
    });
    unsigned idx = 0;
    for (Operation *op : fors) {
      int64_t K = validateGemvBody(op, symTab);
      if (K > 0)
        emitKernel(op, K, module, idx++);
    }
  }
};

}  // namespace

std::unique_ptr<Pass>
mlir::tessera_rocm::createGenerateROCMControlForGemvKernelPass() {
  return std::make_unique<GenerateROCMControlForGemvKernelPass>();
}
