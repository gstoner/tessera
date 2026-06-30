//===- GenerateROCMControlForNormKernel.cpp — norm-in-loop control kernel -===//
//
// CF4d-2 of docs/audit/roadmap/CF_CROSS_ELEMENT_PLAN.md (ROCm-led, gfx1151).
// The second CROSS-ELEMENT device control-flow body: a control_for whose body is
// a single `rmsnorm(carry)` or `layer_norm(carry)` over a 1xK carry. Like the
// CF4d-1 GEMV kernel, a norm can't be per-thread — the normalization statistic
// (mean of squares / mean+var) is a reduction over the WHOLE carry.
//
// Reuses the CF4d-1 cooperative-workgroup substrate: the carry lives in LDS, and
// a gpu.barrier separates loop iterations. Because K ≤ BD (one element per
// thread), each thread reads the K LDS values and computes the statistic
// in-register (K small), then normalizes its own element — no inter-thread
// reduction op needed, just the per-iteration barrier handoff.
//
//   gpu.func @ctrl_for_norm(%CARRY, %OUT: memref<?xf32>, %K: index) kernel {
//     %lds = workgroup memref<BDxf32>
//     if tid<K { lds[tid] = CARRY[tid] } ; barrier
//     scf.for %i = 0 to max_iters {
//       %v = scf.if tid<K {                       // rmsnorm:
//              ss = Σ_k lds[k]² ; m = ss/K         //   mean of squares
//              lds[tid] / sqrt(m + eps)            //   normalize self
//            } else { 0 }                          // layer_norm: mean+var, then
//       barrier ; if tid<K { lds[tid]=%v } ; barrier   //   (x-μ)/√(var+eps)
//     }
//     if tid<K { OUT[tid] = lds[tid] }
//   }
//
// eps is baked from the op's `eps` attr. Single-tile (K ≤ BD). Bodies/shapes
// outside this exact form are left for the CF0 guard / SCF.

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

struct NormInfo {
  int64_t K;
  bool layerNorm;  // false → rmsnorm
  float eps;
};

// Body must be exactly one tessera.rmsnorm(%carry) / tessera.layer_norm(%carry)
// returning the carry; carry = 1xK f32, K ≤ BD, op has 1 operand + 1 result.
static bool validateNormBody(Operation *op, SymbolTable &symTab, NormInfo &out) {
  auto bodySym = op->getAttrOfType<FlatSymbolRefAttr>("body");
  auto carryA = op->getAttrOfType<IntegerAttr>("carry_arg_index");
  if (!bodySym || !carryA || carryA.getInt() != 0 ||
      op->getNumOperands() != 1 || op->getNumResults() != 1)
    return false;
  auto fn = dyn_cast_or_null<func::FuncOp>(
      symTab.lookupNearestSymbolFrom(op, bodySym.getAttr()));
  if (!fn || fn.isExternal())
    return false;
  FunctionType ft = fn.getFunctionType();
  if (ft.getNumInputs() != 1 || ft.getNumResults() != 1)
    return false;
  auto carryT = dyn_cast<RankedTensorType>(ft.getInput(0));
  if (!carryT || carryT.getRank() != 2 || carryT.getDimSize(0) != 1 ||
      !carryT.getElementType().isF32())
    return false;
  int64_t K = carryT.getDimSize(1);
  if (K <= 0 || K > BD || ft.getResult(0) != carryT ||
      op->getOperand(0).getType() != carryT ||
      op->getResult(0).getType() != carryT)
    return false;

  Block &blk = fn.getBody().front();
  auto it = blk.begin();
  if (it == blk.end())
    return false;
  StringRef nm = it->getName().getStringRef();
  bool layer = nm == "tessera.layer_norm";
  if (!layer && nm != "tessera.rmsnorm")
    return false;
  Operation &n = *it;
  if (n.getNumOperands() != 1 || n.getOperand(0) != blk.getArgument(0) ||
      n.getNumResults() != 1)
    return false;
  auto ret = dyn_cast<func::ReturnOp>(std::next(it));
  if (!ret || ret.getNumOperands() != 1 || ret.getOperand(0) != n.getResult(0))
    return false;

  float eps = 1e-5f;
  if (auto e = n.getAttrOfType<FloatAttr>("eps"))
    eps = static_cast<float>(e.getValueAsDouble());
  out = {K, layer, eps};
  return true;
}

struct GenerateROCMControlForNormKernelPass
    : public PassWrapper<GenerateROCMControlForNormKernelPass,
                         OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(
      GenerateROCMControlForNormKernelPass)

  StringRef getArgument() const final {
    return "generate-rocm-control-for-norm-kernel";
  }
  StringRef getDescription() const final {
    return "CF4d-2: lower a norm-in-loop control_for (rmsnorm/layer_norm over a "
           "1xK carry) to one cooperative-workgroup gpu.func (carry in LDS; "
           "per-thread statistic + normalize; barrier per iteration) for gfx1151.";
  }
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<gpu::GPUDialect, scf::SCFDialect, arith::ArithDialect,
                    math::MathDialect, memref::MemRefDialect, func::FuncDialect>();
  }

  void emitKernel(Operation *op, const NormInfo &ni, ModuleOp module,
                  unsigned idx) {
    auto startA = op->getAttrOfType<IntegerAttr>("start");
    auto stopA = op->getAttrOfType<IntegerAttr>("stop");
    auto stepA = op->getAttrOfType<IntegerAttr>("step");
    if (!startA || !stopA || !stepA)
      return;
    int64_t K = ni.K;

    MLIRContext *ctx = module.getContext();
    OpBuilder b(module.getBodyRegion());
    b.setInsertionPointToEnd(module.getBody());
    Location loc = op->getLoc();
    std::string kname = ("tessera_control_for_norm_" + Twine(idx)).str();

    Type f32 = b.getF32Type();
    Type idxTy = b.getIndexType();
    auto memTy = MemRefType::get({ShapedType::kDynamic}, f32);

    auto gpuMod = gpu::GPUModuleOp::create(b, loc, kname + "_mod");
    b.setInsertionPointToStart(&gpuMod.getBodyRegion().front());
    auto fnTy = b.getFunctionType({memTy, memTy, idxTy}, {});  // (CARRY, OUT, K)
    auto f = gpu::GPUFuncOp::create(b, loc, kname, fnTy);
    f->setAttr(gpu::GPUDialect::getKernelFuncAttrName(), b.getUnitAttr());

    auto ws = gpu::AddressSpaceAttr::get(ctx, gpu::AddressSpace::Workgroup);
    auto ldsT = MemRefType::get({BD}, f32, MemRefLayoutAttrInterface(), ws);
    Value lds = f.addWorkgroupAttribution(ldsT, loc);

    OpBuilder kb(f.getContext());
    kb.setInsertionPointToStart(&f.getBody().front());
    Value CARRY = f.getArgument(0), OUT = f.getArgument(1), Kv = f.getArgument(2);
    auto ci = [&](int64_t v) { return arith::ConstantIndexOp::create(kb, loc, v); };
    auto cf = [&](float v) {
      return arith::ConstantOp::create(kb, loc, f32, kb.getF32FloatAttr(v));
    };
    Value c0 = ci(0), c1 = ci(1), cK = ci(K);
    Value Kf = cf(static_cast<float>(K)), epsC = cf(ni.eps), zerof = cf(0.0f);
    Value tid = gpu::ThreadIdOp::create(kb, loc, gpu::Dimension::x);
    Value tidLtK =
        arith::CmpIOp::create(kb, loc, arith::CmpIPredicate::slt, tid, Kv);

    // Load carry → LDS ; barrier.
    {
      auto g = scf::IfOp::create(kb, loc, tidLtK, /*withElse=*/false);
      OpBuilder::InsertionGuard ig(kb);
      kb.setInsertionPointToStart(g.thenBlock());
      Value v = memref::LoadOp::create(kb, loc, CARRY, ValueRange{tid});
      memref::StoreOp::create(kb, loc, v, lds, ValueRange{tid});
    }
    gpu::BarrierOp::create(kb, loc);

    // Σ over k of f(lds[k]) — a serial in-register reduction (K ≤ BD small).
    auto reduce = [&](OpBuilder &rb, llvm::function_ref<Value(Value)> term) {
      auto red = scf::ForOp::create(rb, loc, c0, cK, c1, ValueRange{zerof});
      OpBuilder::InsertionGuard ig(rb);
      rb.setInsertionPointToStart(red.getBody());
      Value k = red.getInductionVar(), acc = red.getRegionIterArg(0);
      Value lk = memref::LoadOp::create(rb, loc, lds, ValueRange{k});
      scf::YieldOp::create(rb, loc,
                           ValueRange{arith::AddFOp::create(rb, loc, acc,
                                                            term(lk))});
      return red.getResult(0);
    };

    Value lb = ci(startA.getInt()), ub = ci(stopA.getInt()),
          st = ci(stepA.getInt());
    auto loop = scf::ForOp::create(kb, loc, lb, ub, st);
    {
      OpBuilder::InsertionGuard ig(kb);
      kb.setInsertionPointToStart(loop.getBody());
      auto comp = scf::IfOp::create(kb, loc, TypeRange{f32}, tidLtK,
                                    /*withElseRegion=*/true);
      {
        OpBuilder::InsertionGuard ig2(kb);
        kb.setInsertionPointToStart(comp.thenBlock());
        Value xj = memref::LoadOp::create(kb, loc, lds, ValueRange{tid});
        Value res;
        if (!ni.layerNorm) {
          // rmsnorm: x / sqrt(mean(x²) + eps)
          Value ss = reduce(kb, [&](Value v) {
            return arith::MulFOp::create(kb, loc, v, v);
          });
          Value mean = arith::DivFOp::create(kb, loc, ss, Kf);
          Value denom = math::SqrtOp::create(
              kb, loc, arith::AddFOp::create(kb, loc, mean, epsC));
          res = arith::DivFOp::create(kb, loc, xj, denom);
        } else {
          // layer_norm: (x - μ) / sqrt(mean((x-μ)²) + eps)
          Value sum = reduce(kb, [&](Value v) { return v; });
          Value mu = arith::DivFOp::create(kb, loc, sum, Kf);
          Value var = reduce(kb, [&](Value v) {
            Value d = arith::SubFOp::create(kb, loc, v, mu);
            return arith::MulFOp::create(kb, loc, d, d);
          });
          Value vmean = arith::DivFOp::create(kb, loc, var, Kf);
          Value denom = math::SqrtOp::create(
              kb, loc, arith::AddFOp::create(kb, loc, vmean, epsC));
          res = arith::DivFOp::create(
              kb, loc, arith::SubFOp::create(kb, loc, xj, mu), denom);
        }
        scf::YieldOp::create(kb, loc, ValueRange{res});
      }
      {
        OpBuilder::InsertionGuard ig2(kb);
        kb.setInsertionPointToStart(comp.elseBlock());
        scf::YieldOp::create(kb, loc, ValueRange{zerof});
      }
      gpu::BarrierOp::create(kb, loc);
      {
        auto g = scf::IfOp::create(kb, loc, tidLtK, /*withElse=*/false);
        OpBuilder::InsertionGuard ig2(kb);
        kb.setInsertionPointToStart(g.thenBlock());
        memref::StoreOp::create(kb, loc, comp.getResult(0), lds,
                                ValueRange{tid});
      }
      gpu::BarrierOp::create(kb, loc);
    }

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
      NormInfo ni;
      if (validateNormBody(op, symTab, ni))
        emitKernel(op, ni, module, idx++);
    }
  }
};

}  // namespace

std::unique_ptr<Pass>
mlir::tessera_rocm::createGenerateROCMControlForNormKernelPass() {
  return std::make_unique<GenerateROCMControlForNormKernelPass>();
}
