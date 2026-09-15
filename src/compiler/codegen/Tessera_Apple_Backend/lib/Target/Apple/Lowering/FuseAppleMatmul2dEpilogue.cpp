//===- FuseAppleMatmul2dEpilogue.cpp - matmul2d + bias/act → one op ------===//
//
// APPLE-MATMUL2D-1 follow-through (2026-09-15): the fused epilogue as an op.
// A `tessera_apple.gpu.matmul2d` whose only consumer is the per-column bias
// add the tracer emits (`tessera.add` against `tessera.broadcast` of a rank-1
// f32 [N]) and/or a `tessera.gelu` / `tessera.relu` / `tessera.silu` becomes
// one `gpu.matmul2d_epilogue`, so the contract the runtime's fused kernels
// implement -- bias added to the fp32 accumulator, activation evaluated in
// fp32, one store -- is stated in IR instead of being reconstructed at dispatch.
//
// Recognition is not promotion: the fused kernel measured no faster than the
// decomposed pair (2026-09-14), so this pass is standalone and the paired
// corpus decides admission. It fuses only what the runtime evaluates
// identically: Tessera's gelu IS the tanh form (runtime.py reference), so no
// semantic key is silently changed (Decision #21a).
//===----------------------------------------------------------------------===//
#include "Tessera/Target/Apple/Passes.h"
#include "Tessera/Target/Apple/TesseraAppleDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Pass/Pass.h"

using namespace mlir;

namespace tessera::apple {
namespace {

// The per-column bias form that survives the shared tiling pass:
//   %bb = tessera.broadcast %bias {shape = [M, N]} : (tensor<Nxf32>) -> tensor<MxNxf32>
//   %s  = tessera.add %mm, %bb   (either operand order)
// Returns the rank-1 bias value, or null when `add` is not that shape.
Value perColumnBias(Operation *add, Value product) {
  if (add->getName().getStringRef() != "tessera.add" || add->getNumOperands() != 2 ||
      add->getNumResults() != 1)
    return nullptr;
  Value other = add->getOperand(0) == product ? add->getOperand(1) : add->getOperand(0);
  if (other == product) return nullptr;
  Operation *bc = other.getDefiningOp();
  if (!bc || bc->getName().getStringRef() != "tessera.broadcast" || bc->getNumOperands() != 1)
    return nullptr;
  auto productType = dyn_cast<RankedTensorType>(product.getType());
  auto biasType = dyn_cast<RankedTensorType>(bc->getOperand(0).getType());
  if (!productType || !biasType || biasType.getRank() != 1 ||
      !biasType.getElementType().isF32() || !biasType.hasStaticShape() ||
      other.getType() != product.getType() || add->getResult(0).getType() != product.getType())
    return nullptr;
  // A rank-1 [N] broadcast to [M, N] adds bias[n] to every row: the runtime's
  // per-column contract. Any other extent is not this epilogue.
  if (biasType.getDimSize(0) != productType.getDimSize(1)) return nullptr;
  return bc->getOperand(0);
}

StringRef activationOf(Operation *op) {
  if (op->getNumOperands() != 1 || op->getNumResults() != 1 ||
      op->getResult(0).getType() != op->getOperand(0).getType())
    return "";
  StringRef name = op->getName().getStringRef();
  if (name == "tessera.gelu") return "gelu";
  if (name == "tessera.relu") return "relu";
  if (name == "tessera.silu") return "silu";
  return "";
}

struct FuseAppleMatmul2dEpiloguePass
    : public PassWrapper<FuseAppleMatmul2dEpiloguePass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(FuseAppleMatmul2dEpiloguePass)

  StringRef getArgument() const override { return "tessera-apple-matmul2d-fuse-epilogue"; }
  StringRef getDescription() const override {
    return "APPLE-MATMUL2D-1 — fuse the per-column bias add and gelu/relu/silu that "
           "consume a gpu.matmul2d into gpu.matmul2d_epilogue.";
  }
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<TesseraAppleDialect>();
  }

  void runOnOperation() override {
    SmallVector<Matmul2dOp> ops;
    getOperation().walk([&](Matmul2dOp op) { ops.push_back(op); });
    for (Matmul2dOp mm : ops) {
      Value cur = mm.getResult();
      Value bias;
      Operation *addOp = nullptr, *actOp = nullptr;
      StringRef act = "none";
      if (!cur.hasOneUse()) continue;
      Operation *user = *cur.getUsers().begin();
      if (Value b = perColumnBias(user, cur)) {
        bias = b;
        addOp = user;
        cur = addOp->getResult(0);
      }
      if (cur.hasOneUse()) {
        Operation *next = *cur.getUsers().begin();
        StringRef a = activationOf(next);
        if (!a.empty()) {
          act = a;
          actOp = next;
          cur = actOp->getResult(0);
        }
      }
      if (!bias && act == "none") continue;

      // Insert where the fused chain ended: every operand (views, bias) is
      // defined above the matmul or above the add that consumed the bias.
      Operation *tail = actOp ? actOp : addOp;
      OpBuilder builder(tail);
      OperationState state(mm.getLoc(), "tessera_apple.gpu.matmul2d_epilogue");
      state.addOperands({mm.getA(), mm.getB()});
      if (bias) state.addOperands({bias});
      state.addTypes({cur.getType()});
      state.addAttribute("tile_m", mm.getTileMAttr());
      state.addAttribute("tile_n", mm.getTileNAttr());
      state.addAttribute("simdgroups", mm.getSimdgroupsAttr());
      state.addAttribute("accumulate", mm.getAccumulateAttr());
      state.addAttribute("act", builder.getStringAttr(act));
      for (StringRef prov : {"tessera_apple.canonical_k_loop", "tessera_apple.ragged_zero_pad",
                             "tessera_apple.ragged_tail"})
        if (Attribute attr = mm->getAttr(prov)) state.addAttribute(prov, attr);
      Operation *fused = builder.create(state);
      cur.replaceAllUsesWith(fused->getResult(0));
      if (actOp) actOp->erase();
      Operation *broadcast = addOp ? (addOp->getOperand(0) == mm.getResult()
                                          ? addOp->getOperand(1).getDefiningOp()
                                          : addOp->getOperand(0).getDefiningOp())
                                   : nullptr;
      if (addOp) addOp->erase();
      if (broadcast && broadcast->use_empty()) broadcast->erase();
      mm.erase();
    }
  }
};

} // namespace

std::unique_ptr<Pass> createFuseAppleMatmul2dEpiloguePass() {
  return std::make_unique<FuseAppleMatmul2dEpiloguePass>();
}

} // namespace tessera::apple
