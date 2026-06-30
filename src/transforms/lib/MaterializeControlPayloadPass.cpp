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
    return "CF4a: decode the tessera.control_for op-list payload "
           "(body_opcodes/...) into a real @body func.func of tessera.* ops, so "
           "the LowerControlFlowToSCF pass can lower the loop. control_if/while "
           "payloads → CF4a follow-up.";
  }

  // Returns true on a successful decode (op rewritten to @body-call form), false
  // if the form is left untouched (no payload, or an unsupported op/shape).
  bool materializeControlFor(Operation *op, SymbolTable &symTab) {
    auto opcodesA = op->getAttrOfType<DenseI32ArrayAttr>("body_opcodes");
    if (!opcodesA)
      return false;  // not a payload form
    auto bodySym = op->getAttrOfType<FlatSymbolRefAttr>("body");
    auto carryIdxA = op->getAttrOfType<IntegerAttr>("carry_arg_index");
    if (!bodySym || !carryIdxA)
      return false;

    auto in0A = op->getAttrOfType<DenseI32ArrayAttr>("body_in0");
    auto in1A = op->getAttrOfType<DenseI32ArrayAttr>("body_in1");
    auto iattrA = op->getAttrOfType<DenseI32ArrayAttr>("body_iattr");
    auto fattrA = op->getAttrOfType<DenseF32ArrayAttr>("body_fattr");
    auto outIdA = op->getAttrOfType<IntegerAttr>("body_out_id");
    if (!in0A || !outIdA)
      return false;

    ArrayRef<int32_t> codes = opcodesA.asArrayRef();
    ArrayRef<int32_t> in0 = in0A.asArrayRef();
    ArrayRef<int32_t> in1 = in1A ? in1A.asArrayRef() : ArrayRef<int32_t>();
    ArrayRef<int32_t> iattr = iattrA ? iattrA.asArrayRef() : ArrayRef<int32_t>();
    ArrayRef<float> fattr = fattrA ? fattrA.asArrayRef() : ArrayRef<float>();

    auto stub = dyn_cast_or_null<func::FuncOp>(
        symTab.lookupNearestSymbolFrom(op, bodySym.getAttr()));
    if (!stub)
      return false;

    int64_t n = static_cast<int64_t>(op->getNumOperands());
    int64_t carryIdx = carryIdxA.getInt();
    if (carryIdx < 0 || carryIdx >= n)
      return false;
    Type carryTy = op->getOperand(carryIdx).getType();

    // ── Phase 1: validate + infer types WITHOUT mutating anything. Only after
    // the whole body decodes cleanly do we touch the stub (so a bail on an
    // unknown opcode / unresolvable shape never leaves a half-built func). ──
    llvm::DenseMap<int32_t, Type> typeMap;
    for (int64_t k = 0; k < n; ++k)
      typeMap[static_cast<int32_t>(k)] = op->getOperand(k).getType();
    typeMap[static_cast<int32_t>(n)] = carryTy;  // live carry

    SmallVector<PlannedOp> plan;
    plan.reserve(codes.size());
    for (size_t j = 0; j < codes.size(); ++j) {
      OpInfo info;
      if (!opForCode(codes[j], info))
        return false;  // unknown opcode → leave the payload for the guard
      auto typeOf = [&](int32_t id) -> Type {
        auto it = typeMap.find(id);
        return it == typeMap.end() ? Type() : it->second;
      };
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
      int32_t resultId = static_cast<int32_t>(n + 1 + j);
      typeMap[resultId] = resTy;
      plan.push_back({info.name, aId, bId, ia, eps, resTy, resultId});
    }
    int32_t outId = static_cast<int32_t>(outIdA.getInt());
    Type outTy = typeMap.lookup(outId);
    if (!outTy || outTy != carryTy)
      return false;  // body output must be the next carry

    // ── Phase 2: validated — materialize the stub and emit the body. ──
    SmallVector<Type> argTys(op->getOperandTypes().begin(),
                             op->getOperandTypes().end());
    OpBuilder b(stub);
    stub.setType(b.getFunctionType(argTys, {carryTy}));
    stub.getBody().getBlocks().clear();
    Block *entry = b.createBlock(&stub.getBody());
    SmallVector<Location> argLocs(argTys.size(), stub.getLoc());
    entry->addArguments(argTys, argLocs);
    b.setInsertionPointToStart(entry);
    Location loc = stub.getLoc();

    llvm::DenseMap<int32_t, Value> valueMap;
    for (int64_t k = 0; k < n; ++k)
      valueMap[static_cast<int32_t>(k)] = entry->getArgument(k);
    valueMap[static_cast<int32_t>(n)] = entry->getArgument(carryIdx);

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

    // The op is now an ordinary @body-call control_for: drop the payload attrs.
    for (StringRef k : {"body_opcodes", "body_in0", "body_in1", "body_iattr",
                        "body_fattr", "body_out_id"})
      op->removeAttr(k);
    return true;
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();
    SymbolTable symTab(module);
    SmallVector<Operation *> fors;
    module.walk([&](Operation *op) {
      if (op->getName().getStringRef() == "tessera.control_for")
        fors.push_back(op);
    });
    for (Operation *op : fors)
      (void)materializeControlFor(op, symTab);
  }
};

}  // namespace

namespace tessera {
std::unique_ptr<Pass> createMaterializeControlPayloadPass() {
  return std::make_unique<MaterializeControlPayload>();
}
}  // namespace tessera
