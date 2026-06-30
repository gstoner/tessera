//===- GenerateROCMControlIfNormKernel.cpp — cross-element control_if -----===//
//
// CF4d-if of the control-flow track (ROCm-led, gfx1151). The first CROSS-ELEMENT
// control_if device kernel: `O = flag>0 ? rmsnorm(x) : layer_norm(x)` over a 1xK
// carry. Each branch is a NORM — a reduction over the whole carry — so it can't
// be the per-thread elementwise control_if (CF4c). The flag is a shape-(1) tensor
// read once and UNIFORM across the workgroup, so the divergent branch is a single
// uniform `scf.if` selecting which cooperative norm every thread computes; only
// the taken branch's statistic runs (divergent, not a data-parallel blend).
//
// Reuses the CF4d-2 cooperative-norm substrate: x lives in LDS, every thread
// reads the K LDS values and computes its element's normalized result in-register
// (K ≤ BD). No loop, no LDS write-back — read the LDS carry, write O.
//
//   gpu.func @ctrl_if_norm(%X,%FLAG,%O: memref<?xf32>, %K: index) kernel {
//     %lds = workgroup memref<BDxf32>
//     if tid<K { lds[tid] = X[tid] } ; barrier
//     %pred = FLAG[0] > 0
//     if tid<K {
//       %r = scf.if %pred { <then-norm stat over lds @ tid> }
//                   else  { <else-norm stat over lds @ tid> }
//       O[tid] = %r
//     }
//   }
//
// Both branches must be a single rmsnorm/layer_norm over the carry (any combo);
// eps is baked per branch. Branches outside this form / a non-(1) flag are left
// for the CF0 guard / SCF.

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
  bool layerNorm;  // false → rmsnorm
  float eps;
};

// A norm branch func: (1xK f32) -> (1xK f32) whose single body op is
// rmsnorm/layer_norm(arg0). Fills `out`. Returns false otherwise.
static bool parseNormFunc(func::FuncOp fn, int64_t K, NormInfo &out) {
  if (!fn || fn.isExternal())
    return false;
  FunctionType ft = fn.getFunctionType();
  if (ft.getNumInputs() != 1 || ft.getNumResults() != 1)
    return false;
  auto carryT = dyn_cast<RankedTensorType>(ft.getInput(0));
  if (!carryT || carryT.getRank() != 2 || carryT.getDimSize(0) != 1 ||
      carryT.getDimSize(1) != K || !carryT.getElementType().isF32() ||
      ft.getResult(0) != Type(carryT))
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
  out = {layer, eps};
  return true;
}

// control_if whose then/else are both norm branches over a 1xK f32 carry, flag a
// shape-(1) f32 tensor. Exactly flag + one data operand, one result == data type.
// Fills K + per-branch NormInfo. Returns true.
static bool validateCrossElementIf(Operation *op, SymbolTable &symTab, int64_t &K,
                                   NormInfo &thenI, NormInfo &elseI) {
  auto thenSym = op->getAttrOfType<FlatSymbolRefAttr>("then_branch");
  auto elseSym = op->getAttrOfType<FlatSymbolRefAttr>("else_branch");
  auto flagA = op->getAttrOfType<IntegerAttr>("flag_arg_index");
  if (!thenSym || !elseSym || !flagA)
    return false;
  int64_t n = static_cast<int64_t>(op->getNumOperands());
  int64_t flag = flagA.getInt();
  if (n != 2 || flag < 0 || flag >= n || op->getNumResults() != 1)
    return false;
  // flag: a ranked f32 tensor with exactly one element.
  auto flagT = dyn_cast<RankedTensorType>(op->getOperand(flag).getType());
  if (!flagT || !flagT.getElementType().isF32() || !flagT.hasStaticShape() ||
      flagT.getNumElements() != 1)
    return false;
  // data: 1xK f32 ; result == data.
  Type dataTy = op->getOperand(flag == 0 ? 1 : 0).getType();
  auto dataT = dyn_cast<RankedTensorType>(dataTy);
  if (!dataT || dataT.getRank() != 2 || dataT.getDimSize(0) != 1 ||
      !dataT.getElementType().isF32() || op->getResult(0).getType() != dataTy)
    return false;
  K = dataT.getDimSize(1);
  if (K <= 0 || K > BD)
    return false;
  auto t = dyn_cast_or_null<func::FuncOp>(
      symTab.lookupNearestSymbolFrom(op, thenSym.getAttr()));
  auto e = dyn_cast_or_null<func::FuncOp>(
      symTab.lookupNearestSymbolFrom(op, elseSym.getAttr()));
  if (!t || !e || t == e)  // shared stub → leave for guard
    return false;
  return parseNormFunc(t, K, thenI) && parseNormFunc(e, K, elseI);
}

struct GenerateROCMControlIfNormKernelPass
    : public PassWrapper<GenerateROCMControlIfNormKernelPass,
                         OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(
      GenerateROCMControlIfNormKernelPass)

  StringRef getArgument() const final {
    return "generate-rocm-control-if-norm-kernel";
  }
  StringRef getDescription() const final {
    return "CF4d-if: lower a cross-element control_if whose branches are "
           "rmsnorm/layer_norm over a 1xK carry to one cooperative-workgroup "
           "gpu.func (uniform flag selects the norm) for gfx1151.";
  }
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<gpu::GPUDialect, scf::SCFDialect, arith::ArithDialect,
                    math::MathDialect, memref::MemRefDialect, func::FuncDialect>();
  }

  void emitKernel(Operation *op, int64_t K, const NormInfo &thenI,
                  const NormInfo &elseI, ModuleOp module, unsigned idx) {
    MLIRContext *ctx = module.getContext();
    OpBuilder b(module.getBodyRegion());
    b.setInsertionPointToEnd(module.getBody());
    Location loc = op->getLoc();
    std::string kname = ("tessera_control_if_norm_" + Twine(idx)).str();

    Type f32 = b.getF32Type(), idxTy = b.getIndexType();
    auto memTy = MemRefType::get({ShapedType::kDynamic}, f32);

    auto gpuMod = gpu::GPUModuleOp::create(b, loc, kname + "_mod");
    b.setInsertionPointToStart(&gpuMod.getBodyRegion().front());
    auto fnTy = b.getFunctionType({memTy, memTy, memTy, idxTy}, {});  // X,FLAG,O,K
    auto f = gpu::GPUFuncOp::create(b, loc, kname, fnTy);
    f->setAttr(gpu::GPUDialect::getKernelFuncAttrName(), b.getUnitAttr());

    auto ws = gpu::AddressSpaceAttr::get(ctx, gpu::AddressSpace::Workgroup);
    auto ldsT = MemRefType::get({BD}, f32, MemRefLayoutAttrInterface(), ws);
    Value lds = f.addWorkgroupAttribution(ldsT, loc);

    OpBuilder kb(f.getContext());
    kb.setInsertionPointToStart(&f.getBody().front());
    Value X = f.getArgument(0), FLAG = f.getArgument(1), OUT = f.getArgument(2),
          Kv = f.getArgument(3);
    auto ci = [&](int64_t v) { return arith::ConstantIndexOp::create(kb, loc, v); };
    auto cf = [&](float v) {
      return arith::ConstantOp::create(kb, loc, f32, kb.getF32FloatAttr(v));
    };
    Value c0 = ci(0), c1 = ci(1), cK = ci(K);
    Value Kf = cf(static_cast<float>(K)), zerof = cf(0.0f);
    Value tid = gpu::ThreadIdOp::create(kb, loc, gpu::Dimension::x);
    Value tidLtK =
        arith::CmpIOp::create(kb, loc, arith::CmpIPredicate::slt, tid, Kv);

    // Load x → LDS ; barrier.
    {
      auto g = scf::IfOp::create(kb, loc, tidLtK, /*withElse=*/false);
      OpBuilder::InsertionGuard ig(kb);
      kb.setInsertionPointToStart(g.thenBlock());
      Value v = memref::LoadOp::create(kb, loc, X, ValueRange{tid});
      memref::StoreOp::create(kb, loc, v, lds, ValueRange{tid});
    }
    gpu::BarrierOp::create(kb, loc);

    // Uniform predicate: flag[0] > 0 (every thread reads the same scalar).
    Value flag0 = memref::LoadOp::create(kb, loc, FLAG, ValueRange{c0});
    Value pred = arith::CmpFOp::create(kb, loc, arith::CmpFPredicate::OGT, flag0,
                                       zerof);

    // Σ over k of f(lds[k]) — serial in-register reduction (K ≤ BD).
    auto reduce = [&](OpBuilder &rb, llvm::function_ref<Value(Value)> term) {
      auto red = scf::ForOp::create(rb, loc, c0, cK, c1, ValueRange{zerof});
      OpBuilder::InsertionGuard ig(rb);
      rb.setInsertionPointToStart(red.getBody());
      Value k = red.getInductionVar(), acc = red.getRegionIterArg(0);
      Value lk = memref::LoadOp::create(rb, loc, lds, ValueRange{k});
      scf::YieldOp::create(
          rb, loc, ValueRange{arith::AddFOp::create(rb, loc, acc, term(lk))});
      return red.getResult(0);
    };
    // Normalized value of element tid under one NormInfo (reads lds, K, eps).
    auto normStat = [&](OpBuilder &nb, const NormInfo &ni) -> Value {
      Value xj = memref::LoadOp::create(nb, loc, lds, ValueRange{tid});
      Value epsC = cf(ni.eps);
      if (!ni.layerNorm) {
        Value ss =
            reduce(nb, [&](Value v) { return arith::MulFOp::create(nb, loc, v, v); });
        Value mean = arith::DivFOp::create(nb, loc, ss, Kf);
        Value denom = math::SqrtOp::create(
            nb, loc, arith::AddFOp::create(nb, loc, mean, epsC));
        return arith::DivFOp::create(nb, loc, xj, denom);
      }
      Value sum = reduce(nb, [&](Value v) { return v; });
      Value mu = arith::DivFOp::create(nb, loc, sum, Kf);
      Value var = reduce(nb, [&](Value v) {
        Value d = arith::SubFOp::create(nb, loc, v, mu);
        return arith::MulFOp::create(nb, loc, d, d);
      });
      Value vmean = arith::DivFOp::create(nb, loc, var, Kf);
      Value denom = math::SqrtOp::create(
          nb, loc, arith::AddFOp::create(nb, loc, vmean, epsC));
      return arith::DivFOp::create(
          nb, loc, arith::SubFOp::create(nb, loc, xj, mu), denom);
    };

    {
      auto g = scf::IfOp::create(kb, loc, tidLtK, /*withElse=*/false);
      OpBuilder::InsertionGuard ig(kb);
      kb.setInsertionPointToStart(g.thenBlock());
      // Uniform divergent branch: only the taken norm's statistic runs.
      auto sel = scf::IfOp::create(kb, loc, TypeRange{f32}, pred,
                                   /*withElseRegion=*/true);
      {
        OpBuilder::InsertionGuard ig2(kb);
        kb.setInsertionPointToStart(sel.thenBlock());
        scf::YieldOp::create(kb, loc, ValueRange{normStat(kb, thenI)});
      }
      {
        OpBuilder::InsertionGuard ig2(kb);
        kb.setInsertionPointToStart(sel.elseBlock());
        scf::YieldOp::create(kb, loc, ValueRange{normStat(kb, elseI)});
      }
      memref::StoreOp::create(kb, loc, sel.getResult(0), OUT, ValueRange{tid});
    }

    kb.setInsertionPointToEnd(&f.getBody().front());
    gpu::ReturnOp::create(kb, loc);
    op->setAttr("tessera.rocm_kernel", b.getStringAttr(kname));
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();
    SymbolTable symTab(module);
    SmallVector<Operation *> ifs;
    module.walk([&](Operation *op) {
      if (op->getName().getStringRef() == "tessera.control_if")
        ifs.push_back(op);
    });
    unsigned idx = 0;
    for (Operation *op : ifs) {
      int64_t K = 0;
      NormInfo thenI, elseI;
      if (validateCrossElementIf(op, symTab, K, thenI, elseI))
        emitKernel(op, K, thenI, elseI, module, idx++);
    }
  }
};

}  // namespace

std::unique_ptr<Pass>
mlir::tessera_rocm::createGenerateROCMControlIfNormKernelPass() {
  return std::make_unique<GenerateROCMControlIfNormKernelPass>();
}
