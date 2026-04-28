//===- ShapeInferencePass.cpp — Forward shape propagation (Phase 6) --------===//
//
// Infers static tensor shapes for Tessera ops through forward dataflow.
// After inference each op is annotated with:
//
//   tessera.inferred_shape = [d0, d1, …]   (ArrayAttr of IntegerAttr i64)
//
// If an op already carries a "tessera.expected_shape" annotation (set by the
// Python ShapeInferenceEngine) the inferred shape is compared to it.  A
// mismatch sets "tessera.actual_shape" so the downstream ErrorReporterPass
// can surface the error with a nice diagnostic.
//
// Supported op families
// ---------------------
//   tessera.matmul          (M×K) × (K×N) → (M×N); batched variant too
//   tessera.elementwise_*   all operands must share shape → same shape
//   tessera.flash_attention  (B,H,S,D) layout; output = Q shape
//   tessera.reduce_*        shape[axis] collapsed
//   tessera.reshape         result shape taken from tessera.target_shape attr
//   tessera.transpose       permutation from tessera.perm attr
//   tessera.concat          concatenated along tessera.axis attr
//   tessera.slice           shape from tessera.sizes attr
//
// Pass options
// ------------
//   --fail-on-unknown   Signal pass failure for ops with no inference rule
//                       (default: false — emit a note and continue).
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Visitors.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/raw_ostream.h"

#include <cstdint>
#include <optional>
#include <vector>

namespace tessera {
namespace diagnostics {

// ---------------------------------------------------------------------------
// Shape utilities
// ---------------------------------------------------------------------------

using Shape = llvm::SmallVector<int64_t, 6>;

/// Build an ArrayAttr of i64 IntegerAttrs from a Shape.
static mlir::ArrayAttr shapeToAttr(mlir::MLIRContext* ctx,
                                   const Shape& shape) {
  llvm::SmallVector<mlir::Attribute> attrs;
  for (int64_t d : shape)
    attrs.push_back(mlir::IntegerAttr::get(
        mlir::IntegerType::get(ctx, 64), d));
  return mlir::ArrayAttr::get(ctx, attrs);
}

/// Extract a Shape from an ArrayAttr of i64 IntegerAttrs.
static std::optional<Shape> attrToShape(mlir::ArrayAttr arr) {
  if (!arr) return std::nullopt;
  Shape s;
  for (mlir::Attribute a : arr) {
    auto ia = mlir::dyn_cast<mlir::IntegerAttr>(a);
    if (!ia) return std::nullopt;
    s.push_back(ia.getInt());
  }
  return s;
}

/// Get the static shape of a Value's type (if ranked tensor).
static std::optional<Shape> valueShape(mlir::Value v) {
  auto rtt = mlir::dyn_cast<mlir::RankedTensorType>(v.getType());
  if (!rtt) return std::nullopt;
  Shape s;
  for (int64_t d : rtt.getShape()) s.push_back(d);
  return s;
}

/// Look up an op-result shape from the symbol table populated during the walk.
static std::optional<Shape> lookupResult(
    const llvm::DenseMap<mlir::Value, Shape>& table, mlir::Value v) {
  auto it = table.find(v);
  if (it != table.end()) return it->second;
  // Fall back to the type.
  return valueShape(v);
}

// ---------------------------------------------------------------------------
// Per-op inference rules
// ---------------------------------------------------------------------------

static std::optional<Shape> inferMatmul(
    mlir::Operation* op,
    const llvm::DenseMap<mlir::Value, Shape>& table) {
  if (op->getNumOperands() < 2) return std::nullopt;
  auto lhs = lookupResult(table, op->getOperand(0));
  auto rhs = lookupResult(table, op->getOperand(1));
  if (!lhs || !rhs) return std::nullopt;

  // Standard 2-D matmul: (M,K) × (K,N) → (M,N)
  if (lhs->size() == 2 && rhs->size() == 2) {
    // K must match
    if ((*lhs)[1] != (*rhs)[0] && (*lhs)[1] != -1 && (*rhs)[0] != -1)
      return std::nullopt; // shape error — let ErrorReporter handle it
    return Shape{(*lhs)[0], (*rhs)[1]};
  }

  // Batched matmul: (..., M, K) × (..., K, N) → (..., M, N)
  if (lhs->size() >= 3 && rhs->size() >= 3 &&
      lhs->size() == rhs->size()) {
    Shape out;
    for (size_t i = 0; i + 2 < lhs->size(); ++i)
      out.push_back((*lhs)[i]);
    out.push_back((*lhs)[lhs->size() - 2]);
    out.push_back((*rhs)[rhs->size() - 1]);
    return out;
  }
  return std::nullopt;
}

static std::optional<Shape> inferElementwise(
    mlir::Operation* op,
    const llvm::DenseMap<mlir::Value, Shape>& table) {
  if (op->getNumOperands() == 0) return std::nullopt;
  auto ref = lookupResult(table, op->getOperand(0));
  if (!ref) return std::nullopt;
  for (unsigned i = 1; i < op->getNumOperands(); ++i) {
    auto s = lookupResult(table, op->getOperand(i));
    if (!s || s->size() != ref->size()) return std::nullopt;
  }
  return ref;
}

static std::optional<Shape> inferFlashAttn(
    mlir::Operation* op,
    const llvm::DenseMap<mlir::Value, Shape>& table) {
  // Expects operands: Q, K, V  each (B, H, S, D)
  if (op->getNumOperands() < 3) return std::nullopt;
  auto Q = lookupResult(table, op->getOperand(0));
  if (!Q || Q->size() != 4) return std::nullopt;
  // Output shape = Q shape
  return Q;
}

static std::optional<Shape> inferReshape(mlir::Operation* op) {
  auto target = op->getAttrOfType<mlir::ArrayAttr>("tessera.target_shape");
  return attrToShape(target);
}

static std::optional<Shape> inferTranspose(
    mlir::Operation* op,
    const llvm::DenseMap<mlir::Value, Shape>& table) {
  if (op->getNumOperands() < 1) return std::nullopt;
  auto src = lookupResult(table, op->getOperand(0));
  if (!src) return std::nullopt;
  auto perm_attr = op->getAttrOfType<mlir::ArrayAttr>("tessera.perm");
  auto perm = attrToShape(perm_attr);
  if (!perm || perm->size() != src->size()) return std::nullopt;
  Shape out(src->size());
  for (size_t i = 0; i < perm->size(); ++i)
    out[i] = (*src)[(*perm)[i]];
  return out;
}

static std::optional<Shape> inferConcat(
    mlir::Operation* op,
    const llvm::DenseMap<mlir::Value, Shape>& table) {
  if (op->getNumOperands() < 2) return std::nullopt;
  auto axis_attr = op->getAttrOfType<mlir::IntegerAttr>("tessera.axis");
  if (!axis_attr) return std::nullopt;
  int64_t axis = axis_attr.getInt();

  auto base = lookupResult(table, op->getOperand(0));
  if (!base) return std::nullopt;
  if (axis < 0 || static_cast<size_t>(axis) >= base->size()) return std::nullopt;

  Shape out = *base;
  for (unsigned i = 1; i < op->getNumOperands(); ++i) {
    auto s = lookupResult(table, op->getOperand(i));
    if (!s || s->size() != base->size()) return std::nullopt;
    out[axis] += (*s)[axis];
  }
  return out;
}

static std::optional<Shape> inferSlice(mlir::Operation* op) {
  auto sizes_attr = op->getAttrOfType<mlir::ArrayAttr>("tessera.sizes");
  return attrToShape(sizes_attr);
}

static std::optional<Shape> inferReduce(
    mlir::Operation* op,
    const llvm::DenseMap<mlir::Value, Shape>& table) {
  if (op->getNumOperands() < 1) return std::nullopt;
  auto src = lookupResult(table, op->getOperand(0));
  if (!src) return std::nullopt;
  auto axis_attr = op->getAttrOfType<mlir::IntegerAttr>("tessera.axis");
  if (!axis_attr) return std::nullopt;
  int64_t axis = axis_attr.getInt();
  if (axis < 0 || static_cast<size_t>(axis) >= src->size()) return std::nullopt;

  // keepdims check
  bool keepdims = false;
  if (auto kd = op->getAttrOfType<mlir::BoolAttr>("tessera.keepdims"))
    keepdims = kd.getValue();

  Shape out;
  for (size_t i = 0; i < src->size(); ++i) {
    if (static_cast<int64_t>(i) == axis) {
      if (keepdims) out.push_back(1);
    } else {
      out.push_back((*src)[i]);
    }
  }
  return out;
}

/// Dispatch to the right inference rule.
static std::optional<Shape> inferShape(
    mlir::Operation* op,
    const llvm::DenseMap<mlir::Value, Shape>& table) {
  llvm::StringRef name = op->getName().getStringRef();

  if (name == "tessera.matmul")          return inferMatmul(op, table);
  if (name.starts_with("tessera.elementwise"))
                                          return inferElementwise(op, table);
  if (name == "tessera.flash_attention") return inferFlashAttn(op, table);
  if (name == "tessera.reshape")         return inferReshape(op);
  if (name == "tessera.transpose")       return inferTranspose(op, table);
  if (name == "tessera.concat")          return inferConcat(op, table);
  if (name == "tessera.slice")           return inferSlice(op);
  if (name.starts_with("tessera.reduce"))
                                          return inferReduce(op, table);

  // For any op with a single ranked-tensor result, propagate its type.
  if (op->getNumResults() == 1)
    return valueShape(op->getResult(0));

  return std::nullopt;
}

// ---------------------------------------------------------------------------
// Pass definition
// ---------------------------------------------------------------------------

struct ShapeInferencePass
    : public mlir::PassWrapper<ShapeInferencePass,
                               mlir::OperationPass<mlir::ModuleOp>> {

  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ShapeInferencePass)

  mlir::Pass::Option<bool> failOnUnknown{
      *this, "fail-on-unknown",
      llvm::cl::desc("Fail the pass when no inference rule is found"),
      llvm::cl::init(false)};

  llvm::StringRef getArgument() const override {
    return "tessera-shape-inference";
  }
  llvm::StringRef getDescription() const override {
    return "Forward-propagate static tensor shapes and annotate ops with "
           "tessera.inferred_shape; compare against tessera.expected_shape "
           "when present";
  }

  void runOnOperation() override {
    mlir::ModuleOp mod = getOperation();
    mlir::MLIRContext* ctx = mod.getContext();
    llvm::DenseMap<mlir::Value, Shape> table;

    mod.walk([&](mlir::Operation* op) -> mlir::WalkResult {
      auto inferred = inferShape(op, table);

      if (!inferred) {
        if (failOnUnknown) {
          op->emitNote()
              << "[tessera-shape-inference] no inference rule for '"
              << op->getName() << "'";
          signalPassFailure();
        }
        return mlir::WalkResult::advance();
      }

      // Annotate the op with the inferred shape.
      op->setAttr("tessera.inferred_shape",
                  shapeToAttr(ctx, *inferred));

      // Populate the symbol table for downstream ops.
      if (op->getNumResults() == 1)
        table[op->getResult(0)] = *inferred;

      // Compare with expected shape if present.
      if (auto exp_attr = op->getAttrOfType<mlir::ArrayAttr>(
              "tessera.expected_shape")) {
        auto expected = attrToShape(exp_attr);
        if (expected && *expected != *inferred) {
          // Tag with actual shape so ErrorReporterPass can surface the diff.
          op->setAttr("tessera.actual_shape",
                      shapeToAttr(ctx, *inferred));
        }
      }

      return mlir::WalkResult::advance();
    });
  }
};

// ---------------------------------------------------------------------------
// Pass registration
// ---------------------------------------------------------------------------

void registerShapeInferencePass() {
  mlir::registerPass([]() -> std::unique_ptr<mlir::Pass> {
    return std::make_unique<ShapeInferencePass>();
  });
}

/// Add the pass to a pass manager.
void addShapeInferencePass(mlir::PassManager& pm,
                           bool failOnUnknown = false) {
  auto pass = std::make_unique<ShapeInferencePass>();
  pass->failOnUnknown = failOnUnknown;
  pm.addPass(std::move(pass));
}

} // namespace diagnostics
} // namespace tessera
