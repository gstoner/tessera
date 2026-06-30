//===- GenerateROCMControlForKernel.cpp — device control-flow kernel ------===//
//
// CF4b of CONTROL_FLOW_AND_DEEPSEEK_ACCELERATION_PLAN (ROCm-led, gfx1151). The
// first EXECUTABLE proof that a Tessera bounded control loop runs as ONE device
// kernel — not one launch per iteration.
//
// Input: a `tessera.control_for` whose body (materialized by CF4a's
// MaterializeControlPayloadPass) is an ELEMENTWISE chain of tessera.* ops over a
// 1-D f32 carry with NO loop-invariant captures. Output: a gpu.module + a kernel
// `gpu.func` that grids over the carry's elements and, per thread, runs the
// loop's `scf.for` (K iterations) applying the body as scalar arith/math — so a
// thread computes the full recurrence locally and the whole loop is a single
// dispatch. Lowers through convert-scf-to-cf → convert-gpu-to-rocdl → hsaco.
//
//   gpu.func @<name>(%X: memref<?xf32>, %O: memref<?xf32>, %N: index) kernel {
//     %g = blockIdx*BD + threadIdx
//     scf.if %g < %N {
//       %x = load %X[%g]
//       %r = scf.for %i = 0 to K step 1 iter_args(%c = %x) -> f32 {
//              <body as scalar ops on %c> ; scf.yield %out
//            }
//       store %r, %O[%g]
//     }
//   }
//
// Bodies with matmul / softmax / norm (cross-element) or captures are NOT
// elementwise; the pass leaves those control_for ops untouched (the CF0 guard /
// a future cross-element kernel-gen handles them). matmul/norm bodies + the
// CUDA mirror are CF4c/CF3.

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"

using namespace mlir;

namespace mlir {
namespace tessera_rocm {
std::unique_ptr<Pass> createGenerateROCMControlForKernelPass();
}  // namespace tessera_rocm
}  // namespace mlir

namespace {

constexpr int64_t BD = 256;  // block dim (threads per block, x)

static Value cstF32(OpBuilder &b, Location loc, float v) {
  return arith::ConstantOp::create(b, loc, b.getF32Type(),
                                   b.getF32FloatAttr(v));
}

// Emit the scalar equivalent of one elementwise tessera.* body op, reading
// operand scalars from `smap`. Returns null for a non-elementwise / unsupported
// op (caller then declines to generate the kernel).
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
    // 0.5 * x * (1 + erf(x / sqrt2))
    Value half = cstF32(b, loc, 0.5f);
    Value one = cstF32(b, loc, 1.0f);
    Value invs2 = cstF32(b, loc, 0.70710678f);
    Value e = math::ErfOp::create(b, loc, arith::MulFOp::create(b, loc, a, invs2));
    Value t = arith::AddFOp::create(b, loc, one, e);
    return arith::MulFOp::create(b, loc, arith::MulFOp::create(b, loc, half, a),
                                 t);
  }
  return nullptr;  // matmul / softmax / norm / unknown → not elementwise
}

// True iff `name` is an elementwise tessera op scalarOp() can translate.
static bool isElementwiseName(StringRef name) {
  return name == "tessera.add" || name == "tessera.sub" ||
         name == "tessera.mul" || name == "tessera.div" ||
         name == "tessera.relu" || name == "tessera.tanh" ||
         name == "tessera.sigmoid" || name == "tessera.silu" ||
         name == "tessera.gelu";
}

// A defined func is a "scalarizable elementwise body" iff it takes exactly
// `nInputs` rank-1 f32 tensors, returns one rank-1 f32 tensor, and every op is
// elementwise (translatable by scalarOp). The emitted kernel ABI is a flat
// memref<?xf32>, hence the rank-1 requirement (a rank>1 carry would not match
// the flat descriptor — left for the guard / a future multi-dim lowering).
static bool isElementwiseFunc(func::FuncOp fn, unsigned nInputs) {
  if (!fn || fn.isExternal())
    return false;
  FunctionType ft = fn.getFunctionType();
  if (ft.getNumInputs() != nInputs || ft.getNumResults() != 1)
    return false;
  auto rank1F32 = [](Type t) {
    auto r = dyn_cast<RankedTensorType>(t);
    return r && r.getRank() == 1 && r.getElementType().isF32();
  };
  for (Type t : ft.getInputs())
    if (!rank1F32(t))
      return false;
  if (!rank1F32(ft.getResult(0)))
    return false;
  for (Operation &o : fn.getBody().front()) {
    if (isa<func::ReturnOp>(o))
      continue;
    if (!isElementwiseName(o.getName().getStringRef()))
      return false;  // matmul / softmax / norm / unknown → not elementwise
  }
  return true;
}

// @body of a control_for: a single-arg (carry-only) elementwise rank-1 f32 func.
static func::FuncOp validateElementwiseBody(Operation *forOp,
                                            SymbolTable &symTab) {
  auto bodySym = forOp->getAttrOfType<FlatSymbolRefAttr>("body");
  if (!bodySym)
    return nullptr;
  auto fn = dyn_cast_or_null<func::FuncOp>(
      symTab.lookupNearestSymbolFrom(forOp, bodySym.getAttr()));
  return isElementwiseFunc(fn, /*nInputs=*/1) ? fn : nullptr;
}

// @then / @else of a control_if, validated against the (X, FLAG, O, N) kernel
// ABI this pass emits. Requires: distinct symbols; both single-arg elementwise
// rank-1 f32 funcs; the OP itself has exactly the flag + ONE non-flag data
// operand and a single result; and the data-operand / result types match the
// branch signature. Anything else (extra operands, a result the branches don't
// produce) is left for the SCF lowering / guard, since the flat kernel could
// not realize it. Returns {then, else}, or {null, null}.
static std::pair<func::FuncOp, func::FuncOp>
validateElementwiseIf(Operation *op, SymbolTable &symTab) {
  auto thenSym = op->getAttrOfType<FlatSymbolRefAttr>("then_branch");
  auto elseSym = op->getAttrOfType<FlatSymbolRefAttr>("else_branch");
  auto flagA = op->getAttrOfType<IntegerAttr>("flag_arg_index");
  if (!thenSym || !elseSym || !flagA)
    return {};
  int64_t n = static_cast<int64_t>(op->getNumOperands());
  int64_t flag = flagA.getInt();
  // Exactly the flag + one data operand, one result.
  if (n != 2 || flag < 0 || flag >= n || op->getNumResults() != 1)
    return {};
  auto t = dyn_cast_or_null<func::FuncOp>(
      symTab.lookupNearestSymbolFrom(op, thenSym.getAttr()));
  auto e = dyn_cast_or_null<func::FuncOp>(
      symTab.lookupNearestSymbolFrom(op, elseSym.getAttr()));
  if (!t || !e || t == e)  // shared stub → leave for the materialize/guard
    return {};
  if (!isElementwiseFunc(t, /*nInputs=*/1) || !isElementwiseFunc(e, 1))
    return {};
  // The branch signature must realize the op: the non-flag data operand feeds
  // the branch input, and the op result is the branch result.
  Type dataTy = op->getOperand(flag == 0 ? 1 : 0).getType();
  Type resTy = op->getResult(0).getType();
  FunctionType tf = t.getFunctionType(), ef = e.getFunctionType();
  if (tf.getInput(0) != dataTy || ef.getInput(0) != dataTy ||
      tf.getResult(0) != resTy || ef.getResult(0) != resTy)
    return {};
  return {t, e};
}

struct GenerateROCMControlForKernelPass
    : public PassWrapper<GenerateROCMControlForKernelPass,
                         OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(GenerateROCMControlForKernelPass)

  StringRef getArgument() const final {
    return "generate-rocm-control-for-kernel";
  }
  StringRef getDescription() const final {
    return "CF4b/CF4c: lower an elementwise-body tessera.control_for / "
           "control_if to a single gpu.func device kernel (grid over the data "
           "elements; per-thread scf.for / scf.if) for the ROCm gfx1151 "
           "control-flow proof.";
  }
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<gpu::GPUDialect, scf::SCFDialect, arith::ArithDialect,
                    math::MathDialect, memref::MemRefDialect, func::FuncDialect>();
  }

  // Translate a single-result elementwise body func to scalar at the current
  // insertion point: map body args to `inputs`, emit each op via scalarOp,
  // return the return-operand scalar (null if any op can't translate).
  Value emitScalarBody(OpBuilder &kb, Location loc, func::FuncOp body,
                       ArrayRef<Value> inputs) {
    llvm::DenseMap<Value, Value> smap;
    for (unsigned i = 0; i < inputs.size(); ++i)
      smap[body.getArgument(i)] = inputs[i];
    for (Operation &o : body.getBody().front()) {
      if (auto ret = dyn_cast<func::ReturnOp>(o))
        return smap.lookup(ret.getOperand(0));
      Value s = scalarOp(kb, loc, &o, smap);
      if (!s)
        return Value();
      smap[o.getResult(0)] = s;
    }
    return Value();
  }

  // Re-validate by actually walking + translating (the probe above is a name
  // filter; here we emit). Returns false (no kernel) if any op can't translate.
  bool emitKernel(Operation *forOp, func::FuncOp body, ModuleOp module,
                  unsigned idx) {
    auto startA = forOp->getAttrOfType<IntegerAttr>("start");
    auto stopA = forOp->getAttrOfType<IntegerAttr>("stop");
    auto stepA = forOp->getAttrOfType<IntegerAttr>("step");
    if (!startA || !stopA || !stepA)
      return false;

    OpBuilder b(module.getBodyRegion());
    b.setInsertionPointToEnd(module.getBody());
    Location loc = forOp->getLoc();
    std::string kname = ("tessera_control_for_" + Twine(idx)).str();

    Type f32 = b.getF32Type();
    Type idxTy = b.getIndexType();
    auto memTy = MemRefType::get({ShapedType::kDynamic}, f32);

    auto gpuMod = gpu::GPUModuleOp::create(b, loc, kname + "_mod");
    b.setInsertionPointToStart(&gpuMod.getBodyRegion().front());
    auto fnTy = b.getFunctionType({memTy, memTy, idxTy}, {});
    auto gpuFunc = gpu::GPUFuncOp::create(b, loc, kname, fnTy);
    gpuFunc->setAttr(gpu::GPUDialect::getKernelFuncAttrName(), b.getUnitAttr());

    OpBuilder kb(gpuFunc.getContext());
    kb.setInsertionPointToStart(&gpuFunc.getBody().front());
    Value X = gpuFunc.getArgument(0), O = gpuFunc.getArgument(1),
          N = gpuFunc.getArgument(2);
    Value bid = gpu::BlockIdOp::create(kb, loc, gpu::Dimension::x);
    Value tid = gpu::ThreadIdOp::create(kb, loc, gpu::Dimension::x);
    Value bd = arith::ConstantIndexOp::create(kb, loc, BD);
    Value gid = arith::AddIOp::create(
        kb, loc, arith::MulIOp::create(kb, loc, bid, bd), tid);
    Value inb = arith::CmpIOp::create(kb, loc, arith::CmpIPredicate::slt, gid, N);
    auto ifo = scf::IfOp::create(kb, loc, inb, /*withElse=*/false);
    kb.setInsertionPointToStart(ifo.thenBlock());

    Value x0 = memref::LoadOp::create(kb, loc, X, ValueRange{gid});
    Value lb = arith::ConstantIndexOp::create(kb, loc, startA.getInt());
    Value ub = arith::ConstantIndexOp::create(kb, loc, stopA.getInt());
    Value st = arith::ConstantIndexOp::create(kb, loc, stepA.getInt());
    auto forK = scf::ForOp::create(kb, loc, lb, ub, st, ValueRange{x0});
    {
      OpBuilder::InsertionGuard g(kb);
      kb.setInsertionPointToStart(forK.getBody());
      // Per-iteration: translate @body's ops to scalar over the carry.
      Value out = emitScalarBody(kb, loc, body, {forK.getRegionIterArg(0)});
      if (!out)
        return false;  // shouldn't happen post-validation; bail safely
      scf::YieldOp::create(kb, loc, ValueRange{out});
    }
    memref::StoreOp::create(kb, loc, forK.getResult(0), O, ValueRange{gid});

    kb.setInsertionPointToEnd(&gpuFunc.getBody().front());
    gpu::ReturnOp::create(kb, loc);

    // Tag the original loop so the host knows which kernel realizes it, then
    // leave it (a later pass / runtime consumes the gpu.module).
    forOp->setAttr("tessera.rocm_kernel", b.getStringAttr(kname));
    return true;
  }

  // control_if → one gpu.func: grid over the data elements; per thread,
  //   x = X[gid]; r = (FLAG[0] > 0) ? then_scalar(x) : else_scalar(x); O[gid]=r.
  // The flag is a shape-(1) selector for all threads. (X, FLAG, O, N) ABI.
  bool emitIfKernel(Operation *ifOp, func::FuncOp thenB, func::FuncOp elseB,
                    ModuleOp module, unsigned idx) {
    OpBuilder b(module.getBodyRegion());
    b.setInsertionPointToEnd(module.getBody());
    Location loc = ifOp->getLoc();
    std::string kname = ("tessera_control_if_" + Twine(idx)).str();

    Type f32 = b.getF32Type();
    Type idxTy = b.getIndexType();
    auto memTy = MemRefType::get({ShapedType::kDynamic}, f32);

    auto gpuMod = gpu::GPUModuleOp::create(b, loc, kname + "_mod");
    b.setInsertionPointToStart(&gpuMod.getBodyRegion().front());
    auto fnTy = b.getFunctionType({memTy, memTy, memTy, idxTy}, {});
    auto gpuFunc = gpu::GPUFuncOp::create(b, loc, kname, fnTy);
    gpuFunc->setAttr(gpu::GPUDialect::getKernelFuncAttrName(), b.getUnitAttr());

    OpBuilder kb(gpuFunc.getContext());
    kb.setInsertionPointToStart(&gpuFunc.getBody().front());
    Value X = gpuFunc.getArgument(0), FLAG = gpuFunc.getArgument(1),
          O = gpuFunc.getArgument(2), N = gpuFunc.getArgument(3);
    Value bid = gpu::BlockIdOp::create(kb, loc, gpu::Dimension::x);
    Value tid = gpu::ThreadIdOp::create(kb, loc, gpu::Dimension::x);
    Value bd = arith::ConstantIndexOp::create(kb, loc, BD);
    Value gid = arith::AddIOp::create(
        kb, loc, arith::MulIOp::create(kb, loc, bid, bd), tid);
    Value inb = arith::CmpIOp::create(kb, loc, arith::CmpIPredicate::slt, gid, N);
    auto bounds = scf::IfOp::create(kb, loc, inb, /*withElse=*/false);
    kb.setInsertionPointToStart(bounds.thenBlock());

    Value x = memref::LoadOp::create(kb, loc, X, ValueRange{gid});
    Value z0 = arith::ConstantIndexOp::create(kb, loc, 0);
    Value f = memref::LoadOp::create(kb, loc, FLAG, ValueRange{z0});
    Value sel = arith::CmpFOp::create(kb, loc, arith::CmpFPredicate::OGT, f,
                                      cstF32(kb, loc, 0.0f));
    auto pick = scf::IfOp::create(kb, loc, TypeRange{f32}, sel,
                                  /*withElseRegion=*/true);
    {
      OpBuilder::InsertionGuard g(kb);
      kb.setInsertionPointToStart(pick.thenBlock());
      Value r = emitScalarBody(kb, loc, thenB, {x});
      if (!r)
        return false;
      scf::YieldOp::create(kb, loc, ValueRange{r});
    }
    {
      OpBuilder::InsertionGuard g(kb);
      kb.setInsertionPointToStart(pick.elseBlock());
      Value r = emitScalarBody(kb, loc, elseB, {x});
      if (!r)
        return false;
      scf::YieldOp::create(kb, loc, ValueRange{r});
    }
    memref::StoreOp::create(kb, loc, pick.getResult(0), O, ValueRange{gid});

    kb.setInsertionPointToEnd(&gpuFunc.getBody().front());
    gpu::ReturnOp::create(kb, loc);
    ifOp->setAttr("tessera.rocm_kernel", b.getStringAttr(kname));
    return true;
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();
    SymbolTable symTab(module);
    SmallVector<Operation *> ctrl;
    module.walk([&](Operation *op) {
      StringRef nm = op->getName().getStringRef();
      if (nm == "tessera.control_for" || nm == "tessera.control_if")
        ctrl.push_back(op);
    });
    unsigned idx = 0;
    for (Operation *op : ctrl) {
      if (op->getName().getStringRef() == "tessera.control_for") {
        if (func::FuncOp body = validateElementwiseBody(op, symTab))
          (void)emitKernel(op, body, module, idx++);
      } else {  // tessera.control_if
        auto [t, e] = validateElementwiseIf(op, symTab);
        if (t && e)
          (void)emitIfKernel(op, t, e, module, idx++);
      }
    }
  }
};

}  // namespace

std::unique_ptr<Pass>
mlir::tessera_rocm::createGenerateROCMControlForKernelPass() {
  return std::make_unique<GenerateROCMControlForKernelPass>();
}
