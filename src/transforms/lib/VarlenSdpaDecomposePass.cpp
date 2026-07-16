//===- VarlenSdpaDecomposePass.cpp - varlen → per-block flash_attn -*- C++ -*-===//
//
// Lowers `tessera.varlen_sdpa` (packed-sequence SDPA — the NVIDIA Cosmos 3
// "two-way flat attention" contract) onto the existing per-block `tessera.flash_attn`
// lane:
//
//   * When `cu_seqlens_q` / `cu_seqlens_k` are compile-time constants, slice each
//     sample's `(query_block, kv_block)` out of the packed streams
//     (`tensor.extract_slice`), run `tessera.flash_attn` per block, and reassemble
//     the result with `tensor.insert_slice`. This is the static-packing case
//     (fixed per-step sequence packing) and is the verifiable "compiler consumes
//     the contract" path.
//   * When cu_seqlens are dynamic (data-dependent packing), or a causal block is
//     rectangular (`Lq != Lk`, where flash_attn's triangular-causal would not
//     match the varlen bottom-right rule), the op is preserved and annotated
//     `tessera.varlen_lowering = "runtime_per_block_flash_attn"` — the honest
//     marker that the runtime (tessera.nn.varlen) carries the decomposition, with
//     a fused FA-3 / NATTEN varlen kernel deferred to the Phase G/H backend gate.
//
//===----------------------------------------------------------------------===//

#include "Tessera/Transforms/Passes.h"

#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/SmallVector.h"

using namespace mlir;

namespace {

// Read a constant rank-1 integer cu_seqlens operand into `out`. Returns false
// when the value is not a compile-time integer constant (the dynamic path).
bool readConstCuSeqlens(Value v, SmallVectorImpl<int64_t> &out) {
  DenseIntElementsAttr attr;
  if (!matchPattern(v, m_Constant(&attr)))
    return false;
  for (const APInt &a : attr.getValues<APInt>())
    out.push_back(a.getSExtValue());
  return true;
}

// Try the static decomposition. Returns failure (op left intact) when cu_seqlens
// are dynamic or a causal block is rectangular.
LogicalResult tryDecompose(Operation *op, OpBuilder &b) {
  if (op->getNumOperands() != 5 || op->getNumResults() != 1)
    return failure();

  Value q = op->getOperand(0), k = op->getOperand(1), v = op->getOperand(2);
  auto qT = dyn_cast<RankedTensorType>(q.getType());
  auto kT = dyn_cast<RankedTensorType>(k.getType());
  auto vT = dyn_cast<RankedTensorType>(v.getType());
  auto oT = dyn_cast<RankedTensorType>(op->getResult(0).getType());
  if (!qT || !kT || !vT || !oT)
    return failure();
  if (qT.getRank() != 3 || !qT.hasStaticShape() || !kT.hasStaticShape() ||
      !vT.hasStaticShape() || !oT.hasStaticShape())
    return failure();

  SmallVector<int64_t> cuQ, cuK;
  if (!readConstCuSeqlens(op->getOperand(3), cuQ) ||
      !readConstCuSeqlens(op->getOperand(4), cuK))
    return failure();
  if (cuQ.size() != cuK.size() || cuQ.size() < 2)
    return failure();

  bool causal = false;
  if (auto c = op->getAttrOfType<BoolAttr>("causal"))
    causal = c.getValue();
  int64_t headDim = qT.getDimSize(2);
  if (auto hd = op->getAttrOfType<IntegerAttr>("head_dim"))
    headDim = hd.getInt();

  const int64_t H = qT.getDimSize(0);
  const int64_t Dh = qT.getDimSize(2);
  Type elem = qT.getElementType();

  // Causal blocks must be square for flash_attn's triangular causal to match the
  // varlen bottom-right rule; otherwise fall back to the runtime annotation.
  for (size_t i = 0; i + 1 < cuQ.size(); ++i)
    if (causal && (cuQ[i + 1] - cuQ[i]) != (cuK[i + 1] - cuK[i]))
      return failure();

  b.setInsertionPoint(op);
  Location loc = op->getLoc();

  Value acc = b.create<tensor::EmptyOp>(loc, oT.getShape(), elem);
  auto idx = [&](int64_t n) { return OpFoldResult(b.getIndexAttr(n)); };
  SmallVector<OpFoldResult> strides(3, b.getIndexAttr(1));

  for (size_t i = 0; i + 1 < cuQ.size(); ++i) {
    int64_t q0 = cuQ[i], Lq = cuQ[i + 1] - cuQ[i];
    int64_t k0 = cuK[i], Lk = cuK[i + 1] - cuK[i];
    if (Lq == 0)
      continue;

    SmallVector<OpFoldResult> qOff{idx(0), idx(q0), idx(0)};
    SmallVector<OpFoldResult> qSz{idx(H), idx(Lq), idx(Dh)};
    SmallVector<OpFoldResult> kOff{idx(0), idx(k0), idx(0)};
    SmallVector<OpFoldResult> kSz{idx(H), idx(Lk), idx(Dh)};
    auto qbT = RankedTensorType::get({H, Lq, Dh}, elem);
    auto kbT = RankedTensorType::get({H, Lk, Dh}, elem);

    Value qb = b.create<tensor::ExtractSliceOp>(loc, qbT, q, qOff, qSz, strides);
    Value kb = b.create<tensor::ExtractSliceOp>(loc, kbT, k, kOff, kSz, strides);
    Value vb = b.create<tensor::ExtractSliceOp>(loc, kbT, v, kOff, kSz, strides);

    // Build tessera.flash_attn generically (the dialect stores
    // operandSegmentSizes as a normal attribute).
    OperationState fa(loc, "tessera.flash_attn");
    fa.addOperands({qb, kb, vb});
    fa.addTypes({qbT});
    fa.addAttribute("operandSegmentSizes", b.getDenseI32ArrayAttr({1, 1, 1, 0}));
    fa.addAttribute("head_dim", b.getI64IntegerAttr(headDim));
    fa.addAttribute("causal", b.getBoolAttr(causal));
    Operation *faOp = b.create(fa);

    acc = b.create<tensor::InsertSliceOp>(loc, faOp->getResult(0), acc,
                                          qOff, qSz, strides);
  }

  op->getResult(0).replaceAllUsesWith(acc);
  op->erase();
  return success();
}

class VarlenSdpaDecomposePass
    : public PassWrapper<VarlenSdpaDecomposePass, OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(VarlenSdpaDecomposePass)

  StringRef getArgument() const override {
    return "tessera-decompose-varlen-sdpa";
  }
  StringRef getDescription() const override {
    return "Decompose tessera.varlen_sdpa into per-block tessera.flash_attn when "
           "cu_seqlens are compile-time constant; annotate + preserve the runtime "
           "per-block lowering otherwise (Cosmos-3 two-way flat attention)";
  }
  void getDependentDialects(DialectRegistry &reg) const override {
    reg.insert<tensor::TensorDialect>();
  }

  void runOnOperation() override {
    SmallVector<Operation *> targets;
    getOperation().walk([&](Operation *op) {
      if (op->getName().getStringRef() == "tessera.varlen_sdpa")
        targets.push_back(op);
    });
    OpBuilder b(&getContext());
    for (Operation *op : targets) {
      if (failed(tryDecompose(op, b)))
        op->setAttr("tessera.varlen_lowering",
                    StringAttr::get(&getContext(),
                                    "runtime_per_block_flash_attn"));
    }
  }
};

} // namespace

namespace tessera {
std::unique_ptr<mlir::Pass> createVarlenSdpaDecomposePass() {
  return std::make_unique<VarlenSdpaDecomposePass>();
}
} // namespace tessera
