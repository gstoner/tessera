//===- GenerateROCMControlScanKernel.cpp — device scan kernel -------------===//
//
// CF4e-1 of the control-flow track (ROCm-led, gfx1151). The first EXECUTABLE
// proof that a Tessera `tessera.control_scan` — the 4th control primitive
// (scan/for/while/cond), the only one with no ROCm device lowering — runs as ONE
// device kernel: a bounded recurrence that CONSUMES a per-step input slice
// `xs[t]` and PRODUCES a stacked per-step output `ys[t]`, not just a final carry.
// That xs-in / ys-out streaming is exactly what scan adds over control_for.
//
// Input: a `tessera.control_scan` whose body is an ELEMENTWISE chain of tessera.*
// ops `(carry, xt) -> (carry', y)` over a rank-1 f32 carry, with NO loop-invariant
// captures. carry/xt are tensor<Kxf32>; xs/ys are tensor<TxKxf32> (row t at
// offset t*K). Output: a gpu.module + a kernel `gpu.func` that grids over the K
// carry elements and, per thread, runs the scan's trip-count loop locally:
//
//   gpu.func @<name>(%INIT,%XS,%YS,%COUT: memref<?xf32>, %N: index) kernel {
//     %g = blockIdx*BD + threadIdx
//     scf.if %g < %N {
//       %c0 = load %INIT[%g]
//       %cf = scf.for %t = 0 to TRIP step 1 iter_args(%c = %c0) -> f32 {
//               %off = %t*%N + %g
//               %xt  = load %XS[%off]
//               (%cn, %y) = <body as scalar ops on (%c, %xt)>
//               store %y, %YS[%off]              // stacked per-step output
//               scf.yield %cn
//             }
//       store %cf, %COUT[%g]                     // final carry
//     }
//   }
//
// One dispatch for the whole trip. TRIP is baked from the op's `trip` attr; N
// (carry width) is a kernel arg so one kernel serves any width. Cross-element
// bodies (matmul/norm in the scan body — true linear-attention / SSM scan) and
// captures are NOT elementwise; the pass leaves those control_scan ops untouched
// for a future cooperative-workgroup scan kernel / the guard.

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
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"

using namespace mlir;

namespace {

constexpr int64_t BD = 256;  // block dim (threads per block, x)

static Value cstF32(OpBuilder &b, Location loc, float v) {
  return arith::ConstantOp::create(b, loc, b.getF32Type(),
                                   b.getF32FloatAttr(v));
}

// Emit the scalar equivalent of one elementwise tessera.* body op, reading
// operand scalars from `smap`. Returns null for a non-elementwise op (caller then
// declines to generate the kernel). Mirrors the CF4b scalar translator.
static Value scalarOp(OpBuilder &b, Location loc, Operation *op,
                      const llvm::DenseMap<Value, Value> &smap) {
  StringRef name = op->getName().getStringRef();
  auto in = [&](unsigned i) -> Value { return smap.lookup(op->getOperand(i)); };
  Value a = op->getNumOperands() > 0 ? in(0) : Value();
  if (!a)
    return nullptr;
  if (name == "tessera.add")
    return arith::AddFOp::create(b, loc, a, in(1));
  if (name == "tessera.sub")
    return arith::SubFOp::create(b, loc, a, in(1));
  if (name == "tessera.mul")
    return arith::MulFOp::create(b, loc, a, in(1));
  if (name == "tessera.div")
    return arith::DivFOp::create(b, loc, a, in(1));
  if (name == "tessera.relu")
    return arith::MaximumFOp::create(b, loc, a, cstF32(b, loc, 0.0f));
  if (name == "tessera.tanh")
    return math::TanhOp::create(b, loc, a);
  if (name == "tessera.sigmoid") {
    Value one = cstF32(b, loc, 1.0f);
    Value e = math::ExpOp::create(b, loc, arith::NegFOp::create(b, loc, a));
    return arith::DivFOp::create(b, loc, one,
                                 arith::AddFOp::create(b, loc, one, e));
  }
  if (name == "tessera.silu") {
    Value one = cstF32(b, loc, 1.0f);
    Value e = math::ExpOp::create(b, loc, arith::NegFOp::create(b, loc, a));
    Value sig = arith::DivFOp::create(b, loc, one,
                                      arith::AddFOp::create(b, loc, one, e));
    return arith::MulFOp::create(b, loc, a, sig);
  }
  if (name == "tessera.gelu") {
    Value half = cstF32(b, loc, 0.5f);
    Value one = cstF32(b, loc, 1.0f);
    Value invs2 = cstF32(b, loc, 0.70710678f);
    Value e =
        math::ErfOp::create(b, loc, arith::MulFOp::create(b, loc, a, invs2));
    Value t = arith::AddFOp::create(b, loc, one, e);
    return arith::MulFOp::create(b, loc, arith::MulFOp::create(b, loc, half, a),
                                 t);
  }
  return nullptr;  // matmul / softmax / norm / unknown → not elementwise
}

static bool isElementwiseName(StringRef name) {
  return name == "tessera.add" || name == "tessera.sub" ||
         name == "tessera.mul" || name == "tessera.div" ||
         name == "tessera.relu" || name == "tessera.tanh" ||
         name == "tessera.sigmoid" || name == "tessera.silu" ||
         name == "tessera.gelu";
}

static bool isRank1F32(Type t, int64_t &dim) {
  auto r = dyn_cast<RankedTensorType>(t);
  if (!r || r.getRank() != 1 || !r.getElementType().isF32())
    return false;
  dim = r.getDimSize(0);
  return true;
}

// @body of a control_scan: (carry, xt) -> (carry', y), both args + both results
// rank-1 f32 of the carry width K, every op elementwise (scalarOp-translatable).
// The flat (INIT, XS, YS, COUT, N) kernel ABI realizes only the no-capture form
// (carry_arg_index 0, exactly init + xs operands). Validates the op shape:
// init (Kxf32), xs (TxKxf32, T == trip), result carry (Kxf32), result ys
// (TxKxf32). Returns @body, or null (left for a future cross-element scan / guard).
static func::FuncOp validateElementwiseScan(Operation *op, SymbolTable &symTab,
                                            int64_t &trip, int64_t &K) {
  auto bodySym = op->getAttrOfType<FlatSymbolRefAttr>("body");
  auto tripA = op->getAttrOfType<IntegerAttr>("trip");
  auto carryA = op->getAttrOfType<IntegerAttr>("carry_arg_index");
  if (!bodySym || !tripA || !carryA || carryA.getInt() != 0 ||
      tripA.getInt() <= 0 || op->getNumOperands() != 2 ||
      op->getNumResults() != 2)
    return nullptr;
  trip = tripA.getInt();

  int64_t ck = 0, xt0 = 0, xt1 = 0, rk = 0, yt0 = 0, yt1 = 0;
  // init: Kxf32.
  if (!isRank1F32(op->getOperand(0).getType(), ck))
    return nullptr;
  K = ck;
  // xs: TxKxf32 with T == trip.
  auto xsT = dyn_cast<RankedTensorType>(op->getOperand(1).getType());
  if (!xsT || xsT.getRank() != 2 || !xsT.getElementType().isF32() ||
      xsT.getDimSize(0) != trip || xsT.getDimSize(1) != K)
    return nullptr;
  // result 0 (final carry): Kxf32 ; result 1 (stacked ys): TxKxf32.
  if (!isRank1F32(op->getResult(0).getType(), rk) || rk != K)
    return nullptr;
  auto ysT = dyn_cast<RankedTensorType>(op->getResult(1).getType());
  if (!ysT || ysT.getRank() != 2 || !ysT.getElementType().isF32() ||
      ysT.getDimSize(0) != trip || ysT.getDimSize(1) != K)
    return nullptr;

  auto fn = dyn_cast_or_null<func::FuncOp>(
      symTab.lookupNearestSymbolFrom(op, bodySym.getAttr()));
  if (!fn || fn.isExternal())
    return nullptr;
  FunctionType ft = fn.getFunctionType();
  if (ft.getNumInputs() != 2 || ft.getNumResults() != 2)
    return nullptr;
  // body args (carry, xt) and results (carry', y) are all Kxf32.
  if (!isRank1F32(ft.getInput(0), xt0) || xt0 != K ||
      !isRank1F32(ft.getInput(1), xt1) || xt1 != K ||
      !isRank1F32(ft.getResult(0), yt0) || yt0 != K ||
      !isRank1F32(ft.getResult(1), yt1) || yt1 != K)
    return nullptr;
  // every body op must be elementwise (translatable).
  for (Operation &o : fn.getBody().front()) {
    if (isa<func::ReturnOp>(o))
      continue;
    if (!isElementwiseName(o.getName().getStringRef()))
      return nullptr;
  }
  return fn;
}

struct GenerateROCMControlScanKernelPass
    : public PassWrapper<GenerateROCMControlScanKernelPass,
                         OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(
      GenerateROCMControlScanKernelPass)

  StringRef getArgument() const final {
    return "generate-rocm-control-scan-kernel";
  }
  StringRef getDescription() const final {
    return "CF4e-1: lower an elementwise-body tessera.control_scan (per-step xs "
           "in, stacked ys out) to one gpu.func device kernel for gfx1151.";
  }
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<gpu::GPUDialect, scf::SCFDialect, arith::ArithDialect,
                    memref::MemRefDialect, math::MathDialect, func::FuncDialect>();
  }

  // Map body args to `inputs`, emit each op via scalarOp, return the two
  // return-operand scalars (carry', y). {null, null} if any op can't translate.
  std::pair<Value, Value> emitScalarScanBody(OpBuilder &kb, Location loc,
                                             func::FuncOp body,
                                             ArrayRef<Value> inputs) {
    llvm::DenseMap<Value, Value> smap;
    for (unsigned i = 0; i < inputs.size(); ++i)
      smap[body.getArgument(i)] = inputs[i];
    for (Operation &o : body.getBody().front()) {
      if (auto ret = dyn_cast<func::ReturnOp>(o))
        return {smap.lookup(ret.getOperand(0)), smap.lookup(ret.getOperand(1))};
      Value s = scalarOp(kb, loc, &o, smap);
      if (!s)
        return {Value(), Value()};
      smap[o.getResult(0)] = s;
    }
    return {Value(), Value()};
  }

  bool emitKernel(Operation *scanOp, func::FuncOp body, ModuleOp module,
                  unsigned idx, int64_t trip) {
    OpBuilder b(module.getBodyRegion());
    b.setInsertionPointToEnd(module.getBody());
    Location loc = scanOp->getLoc();
    std::string kname = ("tessera_control_scan_" + Twine(idx)).str();

    Type f32 = b.getF32Type(), idxTy = b.getIndexType();
    auto memTy = MemRefType::get({ShapedType::kDynamic}, f32);

    auto gpuMod = gpu::GPUModuleOp::create(b, loc, kname + "_mod");
    b.setInsertionPointToStart(&gpuMod.getBodyRegion().front());
    // (INIT, XS, YS, COUT : memref<?xf32>, N : index)
    auto fnTy = b.getFunctionType({memTy, memTy, memTy, memTy, idxTy}, {});
    auto gpuFunc = gpu::GPUFuncOp::create(b, loc, kname, fnTy);
    gpuFunc->setAttr(gpu::GPUDialect::getKernelFuncAttrName(), b.getUnitAttr());

    OpBuilder kb(gpuFunc.getContext());
    kb.setInsertionPointToStart(&gpuFunc.getBody().front());
    Value INIT = gpuFunc.getArgument(0), XS = gpuFunc.getArgument(1),
          YS = gpuFunc.getArgument(2), COUT = gpuFunc.getArgument(3),
          N = gpuFunc.getArgument(4);
    Value bid = gpu::BlockIdOp::create(kb, loc, gpu::Dimension::x);
    Value tid = gpu::ThreadIdOp::create(kb, loc, gpu::Dimension::x);
    Value bd = arith::ConstantIndexOp::create(kb, loc, BD);
    Value gid = arith::AddIOp::create(
        kb, loc, arith::MulIOp::create(kb, loc, bid, bd), tid);
    Value inb = arith::CmpIOp::create(kb, loc, arith::CmpIPredicate::slt, gid, N);
    auto ifo = scf::IfOp::create(kb, loc, inb, /*withElse=*/false);
    kb.setInsertionPointToStart(ifo.thenBlock());

    Value c0 = memref::LoadOp::create(kb, loc, INIT, ValueRange{gid});
    Value lb = arith::ConstantIndexOp::create(kb, loc, 0);
    Value ub = arith::ConstantIndexOp::create(kb, loc, trip);
    Value st = arith::ConstantIndexOp::create(kb, loc, 1);
    auto forT = scf::ForOp::create(kb, loc, lb, ub, st, ValueRange{c0});
    {
      OpBuilder::InsertionGuard g(kb);
      kb.setInsertionPointToStart(forT.getBody());
      Value t = forT.getInductionVar();
      Value c = forT.getRegionIterArg(0);
      // off = t*N + gid  (row-major TxK, flattened).
      Value off = arith::AddIOp::create(
          kb, loc, arith::MulIOp::create(kb, loc, t, N), gid);
      Value xt = memref::LoadOp::create(kb, loc, XS, ValueRange{off});
      auto cy = emitScalarScanBody(kb, loc, body, {c, xt});
      if (!cy.first || !cy.second)
        return false;  // shouldn't happen post-validation; bail safely
      memref::StoreOp::create(kb, loc, cy.second, YS, ValueRange{off});
      scf::YieldOp::create(kb, loc, ValueRange{cy.first});
    }
    memref::StoreOp::create(kb, loc, forT.getResult(0), COUT, ValueRange{gid});

    // scf.if (withElse=false) already carries its scf.yield terminator; ops above
    // were inserted before it. Close the kernel after the if.
    kb.setInsertionPointToEnd(&gpuFunc.getBody().front());
    gpu::ReturnOp::create(kb, loc);
    scanOp->setAttr("tessera.rocm_kernel", b.getStringAttr(kname));
    return true;
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
      int64_t trip = 0, K = 0;
      if (func::FuncOp body = validateElementwiseScan(op, symTab, trip, K))
        if (emitKernel(op, body, module, idx, trip))
          ++idx;
    }
  }
};

}  // namespace

std::unique_ptr<Pass>
mlir::tessera_rocm::createGenerateROCMControlScanKernelPass() {
  return std::make_unique<GenerateROCMControlScanKernelPass>();
}
