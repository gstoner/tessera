// LayoutAssignmentPass.cpp — the *assignment* half of the layout contract
// (2026-06-17), paired with LayoutLegalityPass as its verifier.
//
// Three phases, matching the front-to-back closure plan's Phase-1 description
// ("seed kernel layouts → propagate through pointwise → insert cast{layout}"):
//
//   1. SEED      — stamp `tessera.layout` on kernel-producer ops with a natural
//                  output layout (matmul/batched_gemm → row_major, flash_attn →
//                  bhsd, conv2d_nhwc → nhwc) when they don't already carry one.
//   2. PROPAGATE — flow a compatible producer layout and physical-storage
//                  contract through single-result pointwise ops, transpose
//                  row/column-major rank-2 values explicitly, then carry a
//                  row-major result through a last-axis reduction. Conflicting
//                  binary layouts/storage contracts are left unassigned rather
//                  than choosing the first operand.
//   3. INSERT    — at a consumer op with a known accept-set (matmul/conv2d_nhwc/
//                  flash_attn), if an operand's producer carries a layout outside
//                  that accept-set, splice in a `tessera.cast {tessera.layout=...}`
//                  marker requesting an accepted layout (same dtype). The
//                  CastOp-fold / EraseIdentityCast guards (2026-06-17) keep these
//                  same-type markers from being canonicalized away.
//
// Scope honesty: the assignments are Graph IR metadata. Generic x86 emitter
// bindings consume row-major layout contracts, while Graph cast materializers
// remain target-owned follow-up for Apple/NVIDIA. The pass therefore stays
// opt-in in named Graph pipelines until each inserted cast has a real consumer.

#include "Tessera/Transforms/Passes.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/StringSet.h"

using namespace mlir;

namespace {

constexpr StringRef kLayoutAttr = "tessera.layout";

// Kernel producers → the layout they naturally emit.
static StringRef producerLayout(StringRef opName) {
  if (opName == "tessera.matmul" || opName == "tessera.batched_gemm")
    return "row_major";
  if (opName == "tessera.flash_attn")
    return "bhsd";
  if (opName == "tessera.conv2d_nhwc")
    return "nhwc";
  return {};
}

// Consumer ops whose operands carry a layout contract → accept-set (first
// element is the canonical/preferred layout the inserted cast requests). Mirrors
// LayoutLegalityPass::checkTensorOpLayouts; the operand indices that carry the
// contract are encoded in `contractOperands`.
static ArrayRef<StringRef> consumerAcceptSet(StringRef opName) {
  static const StringRef matmul[] = {"row_major", "col_major"};
  static const StringRef rowMajor[] = {"row_major"};
  static const StringRef nhwc[] = {"nhwc"};
  static const StringRef bhsd[] = {"bhsd"};
  if (opName == "tessera.matmul" || opName == "tessera.batched_gemm")
    return matmul;
  if (opName == "tessera.conv2d_nhwc")
    return nhwc;
  if (opName == "tessera.flash_attn")
    return bhsd;
  if (opName == "tessera.reduce")
    return rowMajor;
  return {};
}

// Operand indices that carry the layout contract for a consumer op.
static SmallVector<unsigned> contractOperands(StringRef opName, unsigned n) {
  if (opName == "tessera.matmul" || opName == "tessera.batched_gemm")
    return {0u, 1u};
  if (opName == "tessera.conv2d_nhwc")
    return {0u};  // data operand; filter is a separate weight layout
  if (opName == "tessera.flash_attn") {
    SmallVector<unsigned> qkv;
    for (unsigned i = 0; i < std::min(n, 3u); ++i) qkv.push_back(i);
    return qkv;
  }
  if (opName == "tessera.reduce")
    return {0u};
  return {};
}

static bool isPointwise(StringRef opName) {
  static const llvm::StringSet<> kSet = {
      "tessera.add",  "tessera.sub",     "tessera.mul",   "tessera.div",
      "tessera.relu", "tessera.gelu",    "tessera.silu",  "tessera.sigmoid",
      "tessera.tanh", "tessera.exp",     "tessera.log",   "tessera.neg",
      "tessera.abs",  "tessera.sqrt",    "tessera.rsqrt", "tessera.softplus",
  };
  return kSet.contains(opName);
}

static ArrayRef<StringRef> physicalStorageAttrs() {
  static const StringRef attrs[] = {
      "tessera.storage_packed",
      "tessera.storage_container",
      "tessera.storage_pack",
  };
  return attrs;
}

// Preserve a physical storage contract only when every layout-bearing operand
// agrees. Untagged scalar/bias operands do not veto propagation, but two tagged
// operands with different packing contracts do.
static void propagatePhysicalStorage(Operation *op) {
  for (StringRef name : physicalStorageAttrs()) {
    Attribute resolved;
    bool sawLayoutOperand = false;
    bool conflict = false;
    for (Value operand : op->getOperands()) {
      Operation *def = operand.getDefiningOp();
      if (!def || !def->hasAttr(kLayoutAttr))
        continue;
      Attribute candidate = def->getAttr(name);
      if (!sawLayoutOperand) {
        resolved = candidate;
        sawLayoutOperand = true;
      } else if (resolved != candidate) {
        conflict = true;
        break;
      }
    }
    if (!conflict && resolved)
      op->setAttr(name, resolved);
  }
}

static void copyPhysicalStorage(Operation *from, Operation *to) {
  if (!from)
    return;
  for (StringRef name : physicalStorageAttrs())
    if (Attribute attr = from->getAttr(name))
      to->setAttr(name, attr);
}

// The layout a value's defining op advertises (its `tessera.layout` attr), or
// empty for a block argument / untagged producer.
static StringRef layoutOf(Value v) {
  Operation *def = v.getDefiningOp();
  if (!def)
    return {};
  if (auto a = def->getAttrOfType<StringAttr>(kLayoutAttr))
    return a.getValue();
  return {};
}

static StringRef propagatedTransposeLayout(Operation *op) {
  if (op->getName().getStringRef() != "tessera.transpose" ||
      op->getNumOperands() != 1 || op->getNumResults() != 1)
    return {};
  auto inputTy = dyn_cast<RankedTensorType>(op->getOperand(0).getType());
  auto outputTy = dyn_cast<RankedTensorType>(op->getResult(0).getType());
  if (!inputTy || !outputTy || inputTy.getRank() != 2 ||
      outputTy.getRank() != 2)
    return {};
  bool swapsAxes = true; // Attribute-free transpose reverses dimensions.
  if (auto permutation =
          op->getAttrOfType<DenseI64ArrayAttr>("permutation")) {
    ArrayRef<int64_t> values = permutation.asArrayRef();
    if (values.size() == 2 && values[0] == 0 && values[1] == 1)
      swapsAxes = false;
    else if (values.size() != 2 || values[0] != 1 || values[1] != 0)
      return {};
  }
  StringRef input = layoutOf(op->getOperand(0));
  if (!swapsAxes)
    return input;
  if (input == "row_major")
    return "col_major";
  if (input == "col_major")
    return "row_major";
  return {};
}

static bool isLastAxisReduction(Operation *op) {
  if (op->getName().getStringRef() != "tessera.reduce" ||
      op->getNumOperands() != 1 || op->getNumResults() != 1)
    return false;
  auto inputTy = dyn_cast<RankedTensorType>(op->getOperand(0).getType());
  auto axis = op->getAttrOfType<IntegerAttr>("axis");
  if (!inputTy || !axis || inputTy.getRank() < 1)
    return false;
  int64_t normalized = axis.getInt();
  if (normalized < 0)
    normalized += inputTy.getRank();
  return normalized == inputTy.getRank() - 1;
}

static StringRef propagatedPointwiseLayout(Operation *op) {
  StringRef resolved;
  for (Value operand : op->getOperands()) {
    StringRef candidate = layoutOf(operand);
    if (candidate.empty())
      continue;
    if (resolved.empty()) {
      resolved = candidate;
      continue;
    }
    if (resolved != candidate)
      return {};
  }
  return resolved;
}

struct LayoutAssignment
    : public PassWrapper<LayoutAssignment, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LayoutAssignment)

  StringRef getArgument() const override { return "tessera-layout-assignment"; }
  StringRef getDescription() const override {
    return "Layout assignment pass — seed kernel-producer layouts "
           "(matmul→row_major, flash_attn→bhsd, conv2d_nhwc→nhwc), propagate "
           "through pointwise ops, and insert tessera.cast{layout} markers at "
           "consumer accept-set boundaries. The assignment half of the layout "
           "contract; LayoutLegalityPass is its verifier.";
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();
    MLIRContext *ctx = &getContext();
    auto stamp = [&](Operation *op, StringRef layout) {
      op->setAttr(kLayoutAttr, StringAttr::get(ctx, layout));
    };

    // ── Phase 1: seed kernel producers. ──────────────────────────────────
    module.walk([&](Operation *op) {
      if (op->hasAttr(kLayoutAttr))
        return;
      StringRef l = producerLayout(op->getName().getStringRef());
      if (!l.empty())
        stamp(op, l);
    });

    // ── Phase 2: propagate through transpose/pointwise to a fixpoint. ────
    bool changed = true;
    while (changed) {
      changed = false;
      module.walk([&](Operation *op) {
        if (op->getNumResults() != 1 || op->hasAttr(kLayoutAttr))
          return;
        StringRef l;
        if (op->getName().getStringRef() == "tessera.transpose")
          l = propagatedTransposeLayout(op);
        else if (isPointwise(op->getName().getStringRef()))
          l = propagatedPointwiseLayout(op);
        else
          return;
        if (!l.empty()) {
          stamp(op, l);
          propagatePhysicalStorage(op);
          changed = true;
        }
      });
    }

    // A last-axis reduction consumes contiguous rows and emits a contiguous
    // lower-rank row-major tensor. This closes the first complete propagation
    // chain: matmul -> pointwise epilogue -> reduction.
    module.walk([&](Operation *op) {
      if (!op->hasAttr(kLayoutAttr) && isLastAxisReduction(op) &&
          !layoutOf(op->getOperand(0)).empty())
        stamp(op, "row_major");
    });

    // ── Phase 3: insert cast{layout} at consumer accept-set boundaries. ──
    // Collect first (don't mutate operands mid-walk).
    struct Fix {
      Operation *consumer;
      unsigned operandIdx;
      StringRef wanted;
    };
    SmallVector<Fix> fixes;
    module.walk([&](Operation *op) {
      ArrayRef<StringRef> accept = consumerAcceptSet(op->getName().getStringRef());
      if (accept.empty())
        return;
      if (op->getName().getStringRef() == "tessera.reduce" &&
          !isLastAxisReduction(op))
        return;
      llvm::StringSet<> acceptSet;
      for (StringRef a : accept)
        acceptSet.insert(a);
      for (unsigned i : contractOperands(op->getName().getStringRef(),
                                         op->getNumOperands())) {
        if (i >= op->getNumOperands())
          continue;
        StringRef l = layoutOf(op->getOperand(i));
        if (!l.empty() && !acceptSet.contains(l))
          fixes.push_back({op, i, accept.front()});
      }
    });

    OpBuilder builder(ctx);
    for (const Fix &f : fixes) {
      Value operand = f.consumer->getOperand(f.operandIdx);
      builder.setInsertionPoint(f.consumer);
      // Same-type tessera.cast carrying the requested layout — a layout-change
      // marker the CastOp-fold / EraseIdentityCast guards preserve. Built
      // generically (tessera.cast is a registered tessera-dialect op).
      OperationState st(f.consumer->getLoc(), "tessera.cast");
      st.addOperands(operand);
      st.addTypes(operand.getType());
      st.addAttribute(kLayoutAttr, StringAttr::get(ctx, f.wanted));
      StringRef source = layoutOf(operand);
      if (!source.empty())
        st.addAttribute("tessera.source_layout", StringAttr::get(ctx, source));
      Operation *marker = builder.create(st);
      copyPhysicalStorage(operand.getDefiningOp(), marker);
      f.consumer->setOperand(f.operandIdx, marker->getResult(0));
    }
  }
};

}  // namespace

namespace tessera {
std::unique_ptr<Pass> createLayoutAssignmentPass() {
  return std::make_unique<LayoutAssignment>();
}
}  // namespace tessera
