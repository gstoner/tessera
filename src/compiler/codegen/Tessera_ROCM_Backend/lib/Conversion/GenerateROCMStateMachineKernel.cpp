//===- GenerateROCMStateMachineKernel.cpp — W4 device state machine -------===//
//
// W4-PRODUCT-1's exact-device slice (INTEGRATED_COMPILER_PLAN queue order 2):
// the bounded arbitrary-CFG program-counter state machine that
// `--tessera-autodiff-paired` structurizes out of an irreducible source CFG
// EXECUTES on gfx1151 as one device kernel — forward and generated backward
// alike — instead of remaining a host-only canonical-SCF artifact.
//
// Input: a `func.func` whose body contains at least one `scf.for` carrying
// `tessera.structured_cfg.execution = "bounded_state_machine_v1"` (the paired
// pass's structurized form; the backward function additionally carries a
// reverse sweep with a nested replay machine). All tensor traffic in that
// form is per-element independent: PC/step slots are `index`/`i1` scalars and
// the data slots are rank-1 f32 tensors touched only by elementwise ops.
//
// Output: one gpu.module + gpu.func per matched function. The kernel grids
// over the tensor elements and, per thread, runs the ENTIRE function body
// scalarized — nested `scf.for`/`scf.if` cloned with tensor types rewritten
// to f32, `tessera.*` elementwise ops translated to scalar arith/math, and
// every other arith/math op cloned generically with converted types. Each
// thread owns its own program counter, so per-element data-dependent control
// flow is expressed as ordinary SIMT divergence.
//
//   gpu.func @tessera_state_machine_<fn>(
//       %FLAGS: memref<?xf32>,   // one f32 per i1 argument; FLAGS[k] > 0
//       %T0..:  memref<?xf32>,   // one per tensor argument
//       %O0..:  memref<?xf32>,   // one per tensor result
//       %STATUS: memref<?xf32>,  // STATUS[gid] = 1.0 iff every cf.assert
//                                //   condition held on this thread's path
//       %N: index) kernel
//
// `cf.assert` cannot trap on device; its condition joins a per-thread
// conjunction written to STATUS, and the HOST enforces the bound (all
// STATUS > 0, else the launch result is rejected) — the max_steps
// exhaustion check stays observable rather than silently dropped. Scalar
// (i1) results are inactive cotangent placeholders in the paired backward
// and are not realized by the kernel ABI.
//
// The structured-CFG digest and residual policy are stamped onto the emitted
// gpu.func, so the execution row binds the exact CFG identity it claims
// (W4-PRODUCT-1 acceptance: "native rows must bind the exact CFG and
// residual digests"). The kernel executes the WHOLE function, so its bound
// identity is every machine inside it: a machine without a digest fails
// closed, one distinct digest is stamped as a string, and several distinct
// digests are stamped as the ordered `tessera.structured_cfg.digests` array
// — never a silently chosen first one.
//
// Function arguments/results are i1 scalars or rank-1 static f32 tensors of
// one common size (the flat memref ABI). INTERIOR values may additionally be
// rank-1 tensors of i1 / signless integers over the same size — e.g. an
// `arith.cmpf` over the data slots feeding `arith.select` (per-element
// data-dependent selection) — and scalarize to their element type. Anything
// outside that vocabulary — non-elementwise tessera ops, tensors off the
// common shape, `cf.assert` below the function's top level, non-splat dense
// constants — declines with a remark naming the reason (Decision #21) and
// leaves the function untouched.
//
// The tessera→scalar translation table intentionally mirrors
// GenerateROCMControlForKernel's (CF4b); this pass needs the generic
// converted-type region cloner around it, which the op-list CF passes do not.

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"

using namespace mlir;

namespace mlir {
namespace tessera_rocm {
std::unique_ptr<Pass> createGenerateROCMStateMachineKernelPass();
std::unique_ptr<Pass> createGenerateROCMStateMachineKernelPass(bool strict);
}  // namespace tessera_rocm
}  // namespace mlir

namespace {

constexpr int64_t BD = 256;  // block dim (threads per block, x)
constexpr llvm::StringLiteral kExecAttr = "tessera.structured_cfg.execution";
constexpr llvm::StringLiteral kExecForm = "bounded_state_machine_v1";
constexpr llvm::StringLiteral kDigestAttr = "tessera.structured_cfg.digest";
constexpr llvm::StringLiteral kResidualAttr = "tessera.autodiff.residual_policy";

static Value cstF32(OpBuilder &b, Location loc, float v) {
  return arith::ConstantOp::create(b, loc, b.getF32Type(),
                                   b.getF32FloatAttr(v));
}

static bool isRank1F32(Type t) {
  auto r = dyn_cast<RankedTensorType>(t);
  return r && r.getRank() == 1 && r.hasStaticShape() &&
         r.getElementType().isF32();
}

// Interior values may also be rank-1 tensors of i1 or signless integers —
// e.g. an `arith.cmpf` over the data slots yielding tensor<Nxi1> consumed by
// `arith.select` (per-element data-dependent selection). Only the FUNCTION
// boundary is restricted to f32 (the flat memref ABI).
static bool isScalarizableRank1(Type t) {
  auto r = dyn_cast<RankedTensorType>(t);
  if (!r || r.getRank() != 1 || !r.hasStaticShape())
    return false;
  Type e = r.getElementType();
  return e.isF32() || e.isSignlessInteger();
}

// Scalar equivalent of one elementwise tessera.* op (same vocabulary as the
// CF4b table — see the header comment for why it is restated here).
static Value scalarTesseraOp(OpBuilder &b, Location loc, Operation *op,
                             ArrayRef<Value> in) {
  StringRef name = op->getName().getStringRef();
  Value a = in.empty() ? Value() : in[0];
  if (!a)
    return nullptr;
  if (name == "tessera.add")
    return arith::AddFOp::create(b, loc, a, in[1]);
  if (name == "tessera.sub")
    return arith::SubFOp::create(b, loc, a, in[1]);
  if (name == "tessera.mul")
    return arith::MulFOp::create(b, loc, a, in[1]);
  if (name == "tessera.div")
    return arith::DivFOp::create(b, loc, a, in[1]);
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
  return nullptr;
}

static bool isElementwiseTesseraName(StringRef name) {
  return name == "tessera.add" || name == "tessera.sub" ||
         name == "tessera.mul" || name == "tessera.div" ||
         name == "tessera.relu" || name == "tessera.tanh" ||
         name == "tessera.sigmoid" || name == "tessera.silu" ||
         name == "tessera.gelu";
}

// ── Validation ──────────────────────────────────────────────────────────────

// One decline reason (for the Decision #21 remark), or empty = admissible.
static std::string validateFunc(func::FuncOp fn, int64_t &numElems) {
  if (fn.isExternal() || fn.getBody().getBlocks().size() != 1)
    return "function body is not a single block";
  numElems = -1;
  auto checkCommon = [&](Type t) -> bool {
    int64_t n = cast<RankedTensorType>(t).getDimSize(0);
    if (numElems == -1)
      numElems = n;
    return n == numElems;
  };
  auto checkTensor = [&](Type t) -> bool {
    return isRank1F32(t) && checkCommon(t);
  };
  auto checkInterior = [&](Type t) -> bool {
    return isScalarizableRank1(t) && checkCommon(t);
  };
  for (Type t : fn.getFunctionType().getInputs()) {
    if (t.isInteger(1))
      continue;
    if (!checkTensor(t))
      return "argument types must be i1 or one common rank-1 static f32 tensor";
  }
  bool anyTensorResult = false;
  for (Type t : fn.getFunctionType().getResults()) {
    if (t.isInteger(1))
      continue;
    if (!checkTensor(t))
      return "result types must be i1 or one common rank-1 static f32 tensor";
    anyTensorResult = true;
  }
  if (!anyTensorResult || numElems <= 0)
    return "no rank-1 static f32 tensor result to realize";

  std::string reason;
  fn.getBody().walk([&](Operation *op) {
    if (!reason.empty())
      return WalkResult::interrupt();
    if (isa<scf::ForOp, scf::IfOp, scf::YieldOp, func::ReturnOp>(op))
      return WalkResult::advance();
    if (auto assertOp = dyn_cast<cf::AssertOp>(op)) {
      if (assertOp->getParentOp() != fn) {
        reason = "cf.assert below the function top level";
        return WalkResult::interrupt();
      }
      return WalkResult::advance();
    }
    if (auto cst = dyn_cast<arith::ConstantOp>(op)) {
      if (auto dense = dyn_cast<DenseElementsAttr>(cst.getValue())) {
        if (!dense.isSplat() || !checkInterior(cst.getType())) {
          reason = "non-splat or non-common-rank-1 dense constant";
          return WalkResult::interrupt();
        }
      }
      return WalkResult::advance();
    }
    StringRef dialect = op->getName().getDialectNamespace();
    StringRef name = op->getName().getStringRef();
    if (dialect == "arith" || dialect == "math") {
      // Both operands AND results must scalarize — an admitted tensor cmpf
      // yields tensor<Nxi1>, which must convert to i1, not survive as a
      // tensor result on an op with scalar operands (PR #605 review, P2).
      for (Type t :
           llvm::concat<const Type>(op->getOperandTypes(),
                                    op->getResultTypes()))
        if (isa<RankedTensorType>(t) && !checkInterior(t)) {
          reason = ("tensor-typed " + name + " outside the scalarizable "
                    "common shape").str();
          return WalkResult::interrupt();
        }
      return WalkResult::advance();
    }
    if (dialect == "tessera") {
      if (!isElementwiseTesseraName(name)) {
        reason = ("non-elementwise op " + name).str();
        return WalkResult::interrupt();
      }
      for (Type t : op->getOperandTypes())
        if (!checkTensor(t)) {
          reason = (name + " over a non-common tensor shape").str();
          return WalkResult::interrupt();
        }
      return WalkResult::advance();
    }
    reason = ("unsupported op " + name).str();
    return WalkResult::interrupt();
  });
  return reason;
}

// ── Scalarizing region cloner ───────────────────────────────────────────────

struct Scalarizer {
  OpBuilder &b;
  llvm::DenseMap<Value, Value> map;
  Value assertAcc;  // running conjunction of cf.assert conditions
  bool failed = false;

  Scalarizer(OpBuilder &b) : b(b) {}

  Type convertType(Type t) {
    if (isScalarizableRank1(t))
      return cast<RankedTensorType>(t).getElementType();
    return t;  // index, i1, f32 pass through
  }

  Value lookup(Value v) { return map.lookup(v); }

  // Returns the mapped `func.return` operands once reached (empty until then).
  SmallVector<Value> cloneBlock(Block &block, Location loc) {
    SmallVector<Value> results;
    for (Operation &op : block) {
      if (failed)
        return {};
      if (auto ret = dyn_cast<func::ReturnOp>(op)) {
        for (Value v : ret.getOperands())
          results.push_back(lookup(v));
        return results;
      }
      cloneOp(&op, loc);
    }
    return results;
  }

  void cloneOp(Operation *op, Location loc) {
    if (auto cst = dyn_cast<arith::ConstantOp>(op)) {
      if (auto dense = dyn_cast<DenseElementsAttr>(cst.getValue())) {
        // Splat over the common shape → one scalar constant of the element
        // type (f32 or a signless integer — validation admitted only those).
        auto elem = cast<TypedAttr>(dense.getSplatValue<Attribute>());
        map[cst.getResult()] =
            arith::ConstantOp::create(b, loc, elem.getType(), elem);
      } else {
        Operation *cl = b.clone(*op);
        map[cst.getResult()] = cl->getResult(0);
      }
      return;
    }
    if (auto assertOp = dyn_cast<cf::AssertOp>(op)) {
      Value c = lookup(assertOp.getArg());
      assertAcc = assertAcc ? Value(arith::AndIOp::create(b, loc, assertAcc, c))
                            : c;
      return;
    }
    if (auto forOp = dyn_cast<scf::ForOp>(op)) {
      SmallVector<Value> inits;
      for (Value v : forOp.getInitArgs())
        inits.push_back(lookup(v));
      auto newFor = scf::ForOp::create(
          b, loc, lookup(forOp.getLowerBound()), lookup(forOp.getUpperBound()),
          lookup(forOp.getStep()), inits);
      {
        OpBuilder::InsertionGuard g(b);
        // `create` with init args builds an entry block with iv + iter args
        // and no terminator.
        Block *body = newFor.getBody();
        if (!body->empty())
          body->clear();
        b.setInsertionPointToStart(body);
        map[forOp.getInductionVar()] = newFor.getInductionVar();
        for (auto [oldA, newA] :
             llvm::zip(forOp.getRegionIterArgs(), newFor.getRegionIterArgs()))
          map[oldA] = newA;
        for (Operation &inner : *forOp.getBody()) {
          if (failed)
            return;
          if (auto yield = dyn_cast<scf::YieldOp>(inner)) {
            SmallVector<Value> ys;
            for (Value v : yield.getOperands())
              ys.push_back(lookup(v));
            scf::YieldOp::create(b, loc, ys);
            break;
          }
          cloneOp(&inner, loc);
        }
      }
      for (auto [oldR, newR] :
           llvm::zip(forOp.getResults(), newFor.getResults()))
        map[oldR] = newR;
      return;
    }
    if (auto ifOp = dyn_cast<scf::IfOp>(op)) {
      SmallVector<Type> resTys;
      for (Type t : ifOp.getResultTypes())
        resTys.push_back(convertType(t));
      auto newIf = scf::IfOp::create(b, loc, resTys, lookup(ifOp.getCondition()),
                                     /*withElseRegion=*/ifOp.elseBlock() !=
                                         nullptr);
      auto cloneRegionInto = [&](Block *src, Block *dst) {
        OpBuilder::InsertionGuard g(b);
        if (!dst->empty())
          dst->clear();
        b.setInsertionPointToStart(dst);
        for (Operation &inner : *src) {
          if (failed)
            return;
          if (auto yield = dyn_cast<scf::YieldOp>(inner)) {
            SmallVector<Value> ys;
            for (Value v : yield.getOperands())
              ys.push_back(lookup(v));
            scf::YieldOp::create(b, loc, ys);
            return;
          }
          cloneOp(&inner, loc);
        }
      };
      cloneRegionInto(ifOp.thenBlock(), newIf.thenBlock());
      if (ifOp.elseBlock())
        cloneRegionInto(ifOp.elseBlock(), newIf.elseBlock());
      for (auto [oldR, newR] : llvm::zip(ifOp.getResults(), newIf.getResults()))
        map[oldR] = newR;
      return;
    }
    StringRef dialect = op->getName().getDialectNamespace();
    if (dialect == "tessera") {
      SmallVector<Value> ins;
      for (Value v : op->getOperands())
        ins.push_back(lookup(v));
      Value s = scalarTesseraOp(b, op->getLoc(), op, ins);
      if (!s) {
        failed = true;
        return;
      }
      map[op->getResult(0)] = s;
      return;
    }
    // Generic arith/math clone with converted operand/result types.
    if (dialect == "arith" || dialect == "math") {
      SmallVector<Value> ins;
      for (Value v : op->getOperands())
        ins.push_back(lookup(v));
      SmallVector<Type> resTys;
      for (Type t : op->getResultTypes())
        resTys.push_back(convertType(t));
      OperationState state(op->getLoc(), op->getName().getStringRef(), ins,
                           resTys, op->getAttrs());
      Operation *cl = b.create(state);
      for (auto [oldR, newR] : llvm::zip(op->getResults(), cl->getResults()))
        map[oldR] = newR;
      return;
    }
    failed = true;  // validation should have caught this
  }
};

// ── The pass ────────────────────────────────────────────────────────────────

struct GenerateROCMStateMachineKernelPass
    : public PassWrapper<GenerateROCMStateMachineKernelPass,
                         OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(
      GenerateROCMStateMachineKernelPass)

  GenerateROCMStateMachineKernelPass() = default;
  explicit GenerateROCMStateMachineKernelPass(bool strict) : strict(strict) {}
  GenerateROCMStateMachineKernelPass(
      const GenerateROCMStateMachineKernelPass &other)
      : PassWrapper(other), strict(other.strict) {}

  // Standalone CLI use keeps the CF-family decline-with-remark convention
  // (the guard/backstop owns what a generator leaves). The CANONICAL
  // executable pipeline constructs the pass with strict=true: the caller
  // REQUESTED family=control_state_machine, so a module with no machine or
  // a machine the vocabulary rejects must FAIL the pipeline rather than
  // sail through gpu-module-to-binary with no gpu.binary and report
  // success (PR #606 review, P1).
  bool strict = false;

  StringRef getArgument() const final {
    return "generate-rocm-state-machine-kernel";
  }
  StringRef getDescription() const final {
    return "W4-PRODUCT-1: lower a paired-pass bounded_state_machine_v1 "
           "function (forward or generated backward) to one per-thread "
           "gpu.func device kernel for the gfx1151 exact-device row.";
  }
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<gpu::GPUDialect, scf::SCFDialect, arith::ArithDialect,
                    math::MathDialect, memref::MemRefDialect, func::FuncDialect,
                    cf::ControlFlowDialect>();
  }

  bool emitKernel(func::FuncOp fn, ModuleOp module) {
    int64_t numElems = 0;
    std::string reason = validateFunc(fn, numElems);
    if (!reason.empty()) {
      fn.emitRemark() << "not lowered to a ROCm state-machine kernel: "
                      << reason;
      return false;
    }

    // The kernel executes the ENTIRE function, so its bound identity is the
    // identity of EVERY machine inside it (PR #605 review, P2): a machine
    // without a digest fails closed, and multiple distinct digests are
    // stamped as an ordered composite rather than silently picking one.
    SmallVector<StringAttr> digests;
    bool missingDigest = false;
    fn.getBody().walk([&](scf::ForOp forOp) {
      auto exec = forOp->getAttrOfType<StringAttr>(kExecAttr);
      if (!exec || exec.getValue() != kExecForm)
        return;
      auto d = forOp->getAttrOfType<StringAttr>(kDigestAttr);
      if (!d) {
        missingDigest = true;
        return;
      }
      if (!llvm::is_contained(digests, d))
        digests.push_back(d);
    });
    if (missingDigest || digests.empty()) {
      fn.emitRemark() << "not lowered to a ROCm state-machine kernel: a "
                         "bounded state machine carries no structured-CFG "
                         "digest — the execution row could not bind the "
                         "exact CFG identity";
      return false;
    }

    OpBuilder b(module.getBodyRegion());
    b.setInsertionPointToEnd(module.getBody());
    Location loc = fn.getLoc();
    std::string kname = ("tessera_state_machine_" + fn.getName()).str();

    Type f32 = b.getF32Type();
    Type idxTy = b.getIndexType();
    auto memTy = MemRefType::get({ShapedType::kDynamic}, f32);

    // ABI: FLAGS, tensor inputs, tensor outputs, STATUS, N.
    unsigned nFlagArgs = 0, nTensorArgs = 0, nTensorResults = 0;
    for (Type t : fn.getFunctionType().getInputs())
      (t.isInteger(1) ? nFlagArgs : nTensorArgs) += 1;
    for (Type t : fn.getFunctionType().getResults())
      if (!t.isInteger(1))
        nTensorResults += 1;

    SmallVector<Type> abi;
    abi.push_back(memTy);  // FLAGS (present even when nFlagArgs == 0)
    for (unsigned i = 0; i < nTensorArgs + nTensorResults; ++i)
      abi.push_back(memTy);
    abi.push_back(memTy);   // STATUS
    abi.push_back(idxTy);   // N

    auto gpuMod = gpu::GPUModuleOp::create(b, loc, kname + "_mod");
    b.setInsertionPointToStart(&gpuMod.getBodyRegion().front());
    auto gpuFunc =
        gpu::GPUFuncOp::create(b, loc, kname, b.getFunctionType(abi, {}));
    gpuFunc->setAttr(gpu::GPUDialect::getKernelFuncAttrName(), b.getUnitAttr());
    if (digests.size() == 1) {
      gpuFunc->setAttr(kDigestAttr, digests.front());
    } else {
      SmallVector<Attribute> all(digests.begin(), digests.end());
      gpuFunc->setAttr("tessera.structured_cfg.digests", b.getArrayAttr(all));
    }
    if (auto residual = fn->getAttrOfType<StringAttr>(kResidualAttr))
      gpuFunc->setAttr(kResidualAttr, residual);

    OpBuilder kb(gpuFunc.getContext());
    kb.setInsertionPointToStart(&gpuFunc.getBody().front());
    unsigned a = 0;
    Value FLAGS = gpuFunc.getArgument(a++);
    SmallVector<Value> TIN, TOUT;
    for (unsigned i = 0; i < nTensorArgs; ++i)
      TIN.push_back(gpuFunc.getArgument(a++));
    for (unsigned i = 0; i < nTensorResults; ++i)
      TOUT.push_back(gpuFunc.getArgument(a++));
    Value STATUS = gpuFunc.getArgument(a++);
    Value N = gpuFunc.getArgument(a++);

    Value bid = gpu::BlockIdOp::create(kb, loc, gpu::Dimension::x);
    Value tid = gpu::ThreadIdOp::create(kb, loc, gpu::Dimension::x);
    Value bd = arith::ConstantIndexOp::create(kb, loc, BD);
    Value gid = arith::AddIOp::create(
        kb, loc, arith::MulIOp::create(kb, loc, bid, bd), tid);
    Value inb =
        arith::CmpIOp::create(kb, loc, arith::CmpIPredicate::slt, gid, N);
    auto bounds = scf::IfOp::create(kb, loc, inb, /*withElse=*/false);
    kb.setInsertionPointToStart(bounds.thenBlock());

    Scalarizer sc(kb);
    unsigned flagIdx = 0, tensorIdx = 0;
    for (BlockArgument arg : fn.getArguments()) {
      if (arg.getType().isInteger(1)) {
        Value k = arith::ConstantIndexOp::create(kb, loc, flagIdx++);
        Value f = memref::LoadOp::create(kb, loc, FLAGS, ValueRange{k});
        sc.map[arg] = arith::CmpFOp::create(kb, loc, arith::CmpFPredicate::OGT,
                                            f, cstF32(kb, loc, 0.0f));
      } else {
        sc.map[arg] = memref::LoadOp::create(kb, loc, TIN[tensorIdx++],
                                             ValueRange{gid});
      }
    }

    SmallVector<Value> results = sc.cloneBlock(fn.getBody().front(), loc);
    if (sc.failed) {
      fn.emitRemark()
          << "not lowered to a ROCm state-machine kernel: scalarization failed";
      gpuMod.erase();
      return false;
    }

    unsigned outIdx = 0;
    for (auto [ty, v] :
         llvm::zip(fn.getFunctionType().getResults(), results)) {
      if (ty.isInteger(1))
        continue;  // inactive scalar cotangent placeholder — not realized
      memref::StoreOp::create(kb, loc, v, TOUT[outIdx++], ValueRange{gid});
    }
    Value ok = sc.assertAcc
                   ? Value(arith::SelectOp::create(kb, loc, sc.assertAcc,
                                                   cstF32(kb, loc, 1.0f),
                                                   cstF32(kb, loc, 0.0f)))
                   : cstF32(kb, loc, 1.0f);
    memref::StoreOp::create(kb, loc, ok, STATUS, ValueRange{gid});

    kb.setInsertionPointToEnd(&gpuFunc.getBody().front());
    gpu::ReturnOp::create(kb, loc);

    fn->setAttr("tessera.rocm_kernel", b.getStringAttr(kname));
    return true;
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();
    SmallVector<func::FuncOp> matched;
    module.walk([&](func::FuncOp fn) {
      bool hasMachine = false;
      fn.getBody().walk([&](scf::ForOp forOp) {
        if (auto exec = forOp->getAttrOfType<StringAttr>(kExecAttr))
          if (exec.getValue() == kExecForm)
            hasMachine = true;
      });
      if (hasMachine)
        matched.push_back(fn);
    });
    if (strict && matched.empty()) {
      module.emitError(
          "family=control_state_machine requested but the module contains "
          "no bounded_state_machine_v1 function");
      return signalPassFailure();
    }
    bool allEmitted = true;
    for (func::FuncOp fn : matched)
      allEmitted &= emitKernel(fn, module);
    if (strict && !allEmitted) {
      module.emitError(
          "family=control_state_machine could not realize every bounded "
          "state machine as a device kernel (see the per-function remarks); "
          "refusing to emit a binary that silently omits requested kernels");
      return signalPassFailure();
    }
  }
};

}  // namespace

std::unique_ptr<Pass>
mlir::tessera_rocm::createGenerateROCMStateMachineKernelPass() {
  return std::make_unique<GenerateROCMStateMachineKernelPass>();
}

std::unique_ptr<Pass>
mlir::tessera_rocm::createGenerateROCMStateMachineKernelPass(bool strict) {
  return std::make_unique<GenerateROCMStateMachineKernelPass>(strict);
}
