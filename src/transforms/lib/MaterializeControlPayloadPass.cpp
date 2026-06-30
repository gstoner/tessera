// MaterializeControlPayloadPass.cpp — CF4a: decode the control-flow op-list
// payload into a real @body func.func of tessera.* ops.
//
// CF4 of CONTROL_FLOW_AND_DEEPSEEK_ACCELERATION_PLAN.md. The frontend emits
// tessera.control_for in the "executable-payload" form: @loop_body is a
// carry-only STUB, and the real loop body is serialized into the
// body_opcodes/body_in0/body_in1/body_iattr/body_fattr/body_out_id attributes
// (the run_graph op-list ABI). CF2's LowerControlFlowToSCFPass SKIPS that form
// (it can't build a correct func.call from a stub). This pass closes the gap:
// it DECODES the op-list back into real ops inside a materialized @body, so the
// loop becomes an ordinary @body-call control_for that CF2 then lowers to
// scf.for — the prerequisite for an executable device body (CF4b: ROCm).
//
// Op-list ABI (python/tessera/apple_mlpkg.py GRAPH_OP + _jit_boundary.py
// _serialize_loop_spec). Tensor-id space for a control_for with n operands:
//   * id 0..n-1      → the n operands (one, at carry_arg_index, is the carry
//                      init; the rest are loop-invariant captures);
//   * id n           → the LIVE carry inside the body;
//   * id n+1+j       → body op j's result.
// body_out_id selects the next-carry value.
//
// We materialize @body with signature (operand types) -> (carry type). Inside,
// id k<n maps to arg k, and id n (the live carry) ALSO maps to arg
// carry_arg_index — consistent because LowerControlFlowToSCFPass passes the
// loop iter_arg in that operand slot. Opcode → tessera op:
//   0 matmul · 1 add · 2 sub · 3 mul · 4 div · 10 softmax · 11 rmsnorm ·
//   12 layer_norm · 20 relu · 21 sigmoid · 22 tanh · 23 silu · 24 gelu.
// iattr packs matmul transpose flags (bit0=transpose_a, bit1=transpose_b);
// fattr carries the rmsnorm/layer_norm eps. Result types are inferred
// (elementwise/unary/norm = in0 type; matmul = M×N from the operand shapes).
// control_if / control_while payloads are the CF4a follow-up.

#include "Tessera/Transforms/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"

using namespace mlir;

namespace {

// Opcode → (tessera op name, is-binary). Mirrors apple_mlpkg.GRAPH_OP.
struct OpInfo {
  StringRef name;
  bool binary;  // true → consumes in1; false → unary (in1 == -1)
};

static bool opForCode(int32_t code, OpInfo &out) {
  switch (code) {
  case 0: out = {"tessera.matmul", true}; return true;
  case 1: out = {"tessera.add", true}; return true;
  case 2: out = {"tessera.sub", true}; return true;
  case 3: out = {"tessera.mul", true}; return true;
  case 4: out = {"tessera.div", true}; return true;
  case 10: out = {"tessera.softmax", false}; return true;
  case 11: out = {"tessera.rmsnorm", false}; return true;
  case 12: out = {"tessera.layer_norm", false}; return true;
  case 20: out = {"tessera.relu", false}; return true;
  case 21: out = {"tessera.sigmoid", false}; return true;
  case 22: out = {"tessera.tanh", false}; return true;
  case 23: out = {"tessera.silu", false}; return true;
  case 24: out = {"tessera.gelu", false}; return true;
  default: return false;
  }
}

// Infer a body op's result type from its operand TYPES (no Values needed, so it
// runs in the pre-mutation validation pass). Elementwise/unary/norm preserve
// the in0 type; matmul is M×N over the (transpose-adjusted) operand shapes.
// Returns null on a shape we can't infer (caller leaves the payload for the
// guard).
static Type inferResultType(StringRef name, Type in0Ty, Type in1Ty,
                            int32_t iattr) {
  if (name != "tessera.matmul")
    return in0Ty;
  auto a = dyn_cast<RankedTensorType>(in0Ty);
  auto b = dyn_cast_or_null<RankedTensorType>(in1Ty);
  if (!a || !b || a.getRank() != 2 || b.getRank() != 2)
    return nullptr;
  bool ta = iattr & 1, tb = iattr & 2;
  int64_t m = ta ? a.getDimSize(1) : a.getDimSize(0);
  int64_t nDim = tb ? b.getDimSize(0) : b.getDimSize(1);
  return RankedTensorType::get({m, nDim}, a.getElementType());
}

// One decoded body op, planned in the validation pass and replayed in the emit
// pass. `a`/`b` are tensor ids into the value map; `b` < 0 for unary ops.
struct PlannedOp {
  StringRef name;
  int32_t a, b, iattr;
  float eps;
  Type resTy;
  int32_t resultId;
};

struct MaterializeControlPayload
    : public PassWrapper<MaterializeControlPayload, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(MaterializeControlPayload)

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<func::FuncDialect>();
  }

  StringRef getArgument() const override {
    return "tessera-materialize-control-payload";
  }
  StringRef getDescription() const override {
    return "CF4a: decode the tessera.control_{for,if,while} op-list payload "
           "(body_opcodes/then_opcodes/cond_opcodes/...) into real @body / "
           "@then / @else / @cond func.funcs of tessera.* ops, so "
           "LowerControlFlowToSCF can lower the op.";
  }

  // ── Two-phase op-list decode, shared by every control construct. ──
  //
  // Phase 1 (validate + infer): walk the op-list against `seedTypes` (tensor-id
  // → Type for the pre-bound ids: operands / carry), where body op j binds id
  // `baseId + j`. Fails (no mutation) on an unknown opcode or an unresolvable
  // operand id. `resultTy` non-null → the out-id type must equal it; null →
  // infer-only (the @cond branch, whose predicate type is whatever the op-list
  // produces). The out-id's type is returned via `outTyOut`. Returns the plan.
  bool validateOpList(ArrayRef<int32_t> codes, ArrayRef<int32_t> in0,
                      ArrayRef<int32_t> in1, ArrayRef<int32_t> iattr,
                      ArrayRef<float> fattr, int32_t outId,
                      const llvm::DenseMap<int32_t, Type> &seedTypes,
                      int32_t baseId, Type resultTy,
                      SmallVectorImpl<PlannedOp> &plan, Type &outTyOut) {
    llvm::DenseMap<int32_t, Type> typeMap = seedTypes;
    auto typeOf = [&](int32_t id) -> Type {
      auto it = typeMap.find(id);
      return it == typeMap.end() ? Type() : it->second;
    };
    for (size_t j = 0; j < codes.size(); ++j) {
      OpInfo info;
      if (!opForCode(codes[j], info))
        return false;
      int32_t aId = in0[j];
      Type aTy = typeOf(aId);
      if (!aTy)
        return false;
      int32_t bId = -1;
      Type bTy;
      if (info.binary) {
        bId = j < in1.size() ? in1[j] : -1;
        bTy = typeOf(bId);
        if (!bTy)
          return false;
      }
      int32_t ia = j < iattr.size() ? iattr[j] : 0;
      Type resTy = inferResultType(info.name, aTy, bTy, ia);
      if (!resTy)
        return false;
      float eps = j < fattr.size() ? fattr[j] : 1e-5f;
      int32_t resultId = baseId + static_cast<int32_t>(j);
      typeMap[resultId] = resTy;
      plan.push_back({info.name, aId, bId, ia, eps, resTy, resultId});
    }
    Type outTy = typeMap.lookup(outId);
    if (!outTy || (resultTy && outTy != resultTy))
      return false;
    outTyOut = outTy;
    return true;
  }

  // Phase 2: materialize `stub` as (argTys) -> resultTy and emit the validated
  // plan. `idToArg` maps a pre-bound tensor-id to its materialized arg index.
  void emitOpList(func::FuncOp stub, ArrayRef<Type> argTys,
                  const llvm::DenseMap<int32_t, int> &idToArg,
                  ArrayRef<PlannedOp> plan, int32_t outId, Type resultTy) {
    OpBuilder b(stub);
    stub.setType(b.getFunctionType(argTys, {resultTy}));
    stub.getBody().getBlocks().clear();
    Block *entry = b.createBlock(&stub.getBody());
    SmallVector<Location> argLocs(argTys.size(), stub.getLoc());
    entry->addArguments(argTys, argLocs);
    b.setInsertionPointToStart(entry);
    Location loc = stub.getLoc();

    llvm::DenseMap<int32_t, Value> valueMap;
    for (auto &kv : idToArg)
      valueMap[kv.first] = entry->getArgument(kv.second);
    for (const PlannedOp &p : plan) {
      OperationState st(loc, p.name);
      st.addOperands(valueMap.lookup(p.a));
      if (p.b >= 0)
        st.addOperands(valueMap.lookup(p.b));
      st.addTypes(p.resTy);
      if (p.name == "tessera.matmul") {
        st.addAttribute("transpose_a", b.getBoolAttr(p.iattr & 1));
        st.addAttribute("transpose_b", b.getBoolAttr(p.iattr & 2));
      } else if (p.name == "tessera.rmsnorm" ||
                 p.name == "tessera.layer_norm") {
        st.addAttribute("eps", b.getF64FloatAttr(p.eps));
      }
      valueMap[p.resultId] = b.create(st)->getResult(0);
    }
    func::ReturnOp::create(b, loc, valueMap.lookup(outId));
  }

  static ArrayRef<int32_t> i32(DenseI32ArrayAttr a) {
    return a ? a.asArrayRef() : ArrayRef<int32_t>();
  }
  static ArrayRef<float> f32(DenseF32ArrayAttr a) {
    return a ? a.asArrayRef() : ArrayRef<float>();
  }

  // control_for: ids 0..n-1 = operands, id n = live carry, body op j = id n+1+j.
  // @body takes ALL operands (id k<n → arg k; live carry id n → arg
  // carry_arg_index — consistent because CF2 passes the iter_arg there).
  bool materializeControlFor(Operation *op, SymbolTable &symTab) {
    auto opcodesA = op->getAttrOfType<DenseI32ArrayAttr>("body_opcodes");
    auto bodySym = op->getAttrOfType<FlatSymbolRefAttr>("body");
    auto carryIdxA = op->getAttrOfType<IntegerAttr>("carry_arg_index");
    auto outIdA = op->getAttrOfType<IntegerAttr>("body_out_id");
    auto in0A = op->getAttrOfType<DenseI32ArrayAttr>("body_in0");
    if (!opcodesA || !bodySym || !carryIdxA || !outIdA || !in0A)
      return false;
    auto stub = dyn_cast_or_null<func::FuncOp>(
        symTab.lookupNearestSymbolFrom(op, bodySym.getAttr()));
    if (!stub)
      return false;
    int64_t n = static_cast<int64_t>(op->getNumOperands());
    int64_t carryIdx = carryIdxA.getInt();
    if (carryIdx < 0 || carryIdx >= n)
      return false;
    Type carryTy = op->getOperand(carryIdx).getType();
    int32_t outId = static_cast<int32_t>(outIdA.getInt());

    llvm::DenseMap<int32_t, Type> seedTypes;
    llvm::DenseMap<int32_t, int> idToArg;
    for (int64_t k = 0; k < n; ++k) {
      seedTypes[k] = op->getOperand(k).getType();
      idToArg[k] = static_cast<int>(k);
    }
    seedTypes[n] = carryTy;
    idToArg[n] = static_cast<int>(carryIdx);

    SmallVector<PlannedOp> plan;
    Type outTy;
    if (!validateOpList(opcodesA.asArrayRef(), in0A.asArrayRef(),
                        i32(op->getAttrOfType<DenseI32ArrayAttr>("body_in1")),
                        i32(op->getAttrOfType<DenseI32ArrayAttr>("body_iattr")),
                        f32(op->getAttrOfType<DenseF32ArrayAttr>("body_fattr")),
                        outId, seedTypes, static_cast<int32_t>(n + 1), carryTy,
                        plan, outTy))
      return false;
    SmallVector<Type> argTys(op->getOperandTypes().begin(),
                             op->getOperandTypes().end());
    emitOpList(stub, argTys, idToArg, plan, outId, carryTy);
    for (StringRef k : {"body_opcodes", "body_in0", "body_in1", "body_iattr",
                        "body_fattr", "body_out_id"})
      op->removeAttr(k);
    return true;
  }

  // control_if: ids 0..n-1 = operands, each branch op j = id n+j (no carry).
  // CF2's lowerControlIf calls the branches with the NON-flag operands, so the
  // materialized @then/@else take the non-flag operands and a payload id k maps
  // to arg (k < flag ? k : k-1). A branch that references the flag id is left
  // for the guard. Both branches are validated before EITHER is materialized.
  bool materializeControlIf(Operation *op, SymbolTable &symTab) {
    auto thenCodes = op->getAttrOfType<DenseI32ArrayAttr>("then_opcodes");
    auto elseCodes = op->getAttrOfType<DenseI32ArrayAttr>("else_opcodes");
    auto thenSym = op->getAttrOfType<FlatSymbolRefAttr>("then_branch");
    auto elseSym = op->getAttrOfType<FlatSymbolRefAttr>("else_branch");
    auto flagA = op->getAttrOfType<IntegerAttr>("flag_arg_index");
    auto thenOut = op->getAttrOfType<IntegerAttr>("then_out_id");
    auto elseOut = op->getAttrOfType<IntegerAttr>("else_out_id");
    auto thenIn0 = op->getAttrOfType<DenseI32ArrayAttr>("then_in0");
    auto elseIn0 = op->getAttrOfType<DenseI32ArrayAttr>("else_in0");
    if (!thenCodes || !elseCodes || !thenSym || !elseSym || !flagA ||
        !thenOut || !elseOut || !thenIn0 || !elseIn0 || op->getNumResults() != 1)
      return false;
    auto thenStub = dyn_cast_or_null<func::FuncOp>(
        symTab.lookupNearestSymbolFrom(op, thenSym.getAttr()));
    auto elseStub = dyn_cast_or_null<func::FuncOp>(
        symTab.lookupNearestSymbolFrom(op, elseSym.getAttr()));
    if (!thenStub || !elseStub)
      return false;
    // If both arms resolve to the SAME stub, materializing the second branch
    // would overwrite the first's body — both arms would then call one body,
    // silently changing the branch semantics. Leave such a payload for the
    // guard rather than emit a wrong lowering.
    if (thenStub == elseStub)
      return false;
    int64_t n = static_cast<int64_t>(op->getNumOperands());
    int64_t flag = flagA.getInt();
    if (flag < 0 || flag >= n)
      return false;
    Type resultTy = op->getResult(0).getType();

    // Seed every NON-flag operand id; a reference to the flag id fails to
    // resolve (intentional — the flag isn't a branch arg).
    llvm::DenseMap<int32_t, Type> seedTypes;
    llvm::DenseMap<int32_t, int> idToArg;
    SmallVector<Type> argTys;
    for (int64_t k = 0; k < n; ++k) {
      if (k == flag)
        continue;
      seedTypes[k] = op->getOperand(k).getType();
      idToArg[k] = static_cast<int>(argTys.size());
      argTys.push_back(op->getOperand(k).getType());
    }

    int32_t tOut = static_cast<int32_t>(thenOut.getInt());
    int32_t eOut = static_cast<int32_t>(elseOut.getInt());
    SmallVector<PlannedOp> thenPlan, elsePlan;
    Type tOutTy, eOutTy;
    if (!validateOpList(thenCodes.asArrayRef(), thenIn0.asArrayRef(),
                        i32(op->getAttrOfType<DenseI32ArrayAttr>("then_in1")),
                        i32(op->getAttrOfType<DenseI32ArrayAttr>("then_iattr")),
                        f32(op->getAttrOfType<DenseF32ArrayAttr>("then_fattr")),
                        tOut, seedTypes, static_cast<int32_t>(n), resultTy,
                        thenPlan, tOutTy))
      return false;
    if (!validateOpList(elseCodes.asArrayRef(), elseIn0.asArrayRef(),
                        i32(op->getAttrOfType<DenseI32ArrayAttr>("else_in1")),
                        i32(op->getAttrOfType<DenseI32ArrayAttr>("else_iattr")),
                        f32(op->getAttrOfType<DenseF32ArrayAttr>("else_fattr")),
                        eOut, seedTypes, static_cast<int32_t>(n), resultTy,
                        elsePlan, eOutTy))
      return false;

    emitOpList(thenStub, argTys, idToArg, thenPlan, tOut, resultTy);
    emitOpList(elseStub, argTys, idToArg, elsePlan, eOut, resultTy);
    for (StringRef k : {"then_opcodes", "then_in0", "then_in1", "then_iattr",
                        "then_fattr", "then_out_id", "else_opcodes", "else_in0",
                        "else_in1", "else_iattr", "else_fattr", "else_out_id"})
      op->removeAttr(k);
    return true;
  }

  // control_while: ids 0..n-1 = operands, id n = live carry, body & cond op j =
  // id n+1+j. CF2's lowerControlWhile calls @body / @cond with only the carry,
  // so the materialized funcs take the single carry (id n → arg 0); the body is
  // (carry)->carry and the cond is (carry)->pred (the predicate type is whatever
  // the cond op-list produces — inferred). @body and @cond must be distinct.
  bool materializeControlWhile(Operation *op, SymbolTable &symTab) {
    auto bodyCodes = op->getAttrOfType<DenseI32ArrayAttr>("body_opcodes");
    auto condCodes = op->getAttrOfType<DenseI32ArrayAttr>("cond_opcodes");
    auto bodySym = op->getAttrOfType<FlatSymbolRefAttr>("body");
    auto condSym = op->getAttrOfType<FlatSymbolRefAttr>("cond");
    auto carryIdxA = op->getAttrOfType<IntegerAttr>("carry_arg_index");
    auto bodyOutA = op->getAttrOfType<IntegerAttr>("body_out_id");
    auto condOutA = op->getAttrOfType<IntegerAttr>("cond_out_id");
    auto bodyIn0 = op->getAttrOfType<DenseI32ArrayAttr>("body_in0");
    auto condIn0 = op->getAttrOfType<DenseI32ArrayAttr>("cond_in0");
    if (!bodyCodes || !condCodes || !bodySym || !condSym || !carryIdxA ||
        !bodyOutA || !condOutA || !bodyIn0 || !condIn0 ||
        op->getNumResults() != 1)
      return false;
    int64_t n = static_cast<int64_t>(op->getNumOperands());
    // CF2 passes only the carry — require a single carry operand (no captures).
    if (n != 1 || carryIdxA.getInt() != 0)
      return false;
    auto bodyStub = dyn_cast_or_null<func::FuncOp>(
        symTab.lookupNearestSymbolFrom(op, bodySym.getAttr()));
    auto condStub = dyn_cast_or_null<func::FuncOp>(
        symTab.lookupNearestSymbolFrom(op, condSym.getAttr()));
    if (!bodyStub || !condStub || bodyStub == condStub)
      return false;
    Type carryTy = op->getOperand(0).getType();
    if (op->getResult(0).getType() != carryTy)
      return false;

    llvm::DenseMap<int32_t, Type> seedTypes;
    llvm::DenseMap<int32_t, int> idToArg;
    seedTypes[static_cast<int32_t>(n)] = carryTy;  // live carry
    idToArg[static_cast<int32_t>(n)] = 0;
    SmallVector<Type> argTys{carryTy};
    int32_t base = static_cast<int32_t>(n + 1);
    int32_t bOut = static_cast<int32_t>(bodyOutA.getInt());
    int32_t cOut = static_cast<int32_t>(condOutA.getInt());

    SmallVector<PlannedOp> bodyPlan, condPlan;
    Type bodyOutTy, condOutTy;
    if (!validateOpList(bodyCodes.asArrayRef(), bodyIn0.asArrayRef(),
                        i32(op->getAttrOfType<DenseI32ArrayAttr>("body_in1")),
                        i32(op->getAttrOfType<DenseI32ArrayAttr>("body_iattr")),
                        f32(op->getAttrOfType<DenseF32ArrayAttr>("body_fattr")),
                        bOut, seedTypes, base, carryTy, bodyPlan, bodyOutTy))
      return false;
    // @cond's predicate type is inferred (resultTy = null).
    if (!validateOpList(condCodes.asArrayRef(), condIn0.asArrayRef(),
                        i32(op->getAttrOfType<DenseI32ArrayAttr>("cond_in1")),
                        i32(op->getAttrOfType<DenseI32ArrayAttr>("cond_iattr")),
                        f32(op->getAttrOfType<DenseF32ArrayAttr>("cond_fattr")),
                        cOut, seedTypes, base, Type(), condPlan, condOutTy))
      return false;

    emitOpList(bodyStub, argTys, idToArg, bodyPlan, bOut, carryTy);
    emitOpList(condStub, argTys, idToArg, condPlan, cOut, condOutTy);
    for (StringRef k : {"body_opcodes", "body_in0", "body_in1", "body_iattr",
                        "body_fattr", "body_out_id", "cond_opcodes", "cond_in0",
                        "cond_in1", "cond_iattr", "cond_fattr", "cond_out_id"})
      op->removeAttr(k);
    return true;
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();
    SymbolTable symTab(module);
    SmallVector<Operation *> ctrl;
    module.walk([&](Operation *op) {
      StringRef nm = op->getName().getStringRef();
      if (nm == "tessera.control_for" || nm == "tessera.control_if" ||
          nm == "tessera.control_while")
        ctrl.push_back(op);
    });
    for (Operation *op : ctrl) {
      StringRef nm = op->getName().getStringRef();
      if (nm == "tessera.control_for")
        (void)materializeControlFor(op, symTab);
      else if (nm == "tessera.control_if")
        (void)materializeControlIf(op, symTab);
      else
        (void)materializeControlWhile(op, symTab);
    }
  }
};

}  // namespace

namespace tessera {
std::unique_ptr<Pass> createMaterializeControlPayloadPass() {
  return std::make_unique<MaterializeControlPayload>();
}
}  // namespace tessera
