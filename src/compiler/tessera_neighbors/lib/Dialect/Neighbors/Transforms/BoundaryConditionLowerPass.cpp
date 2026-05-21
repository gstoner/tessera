//===- BoundaryConditionLowerPass.cpp — Lower stencil BC (Gap 2) ----------===//
//
// StencilLowerPass annotates each ``stencil.apply`` op with the textual
// boundary condition declared on its ``stencil.define`` (``stencil.bc``).
// That string however is opaque to downstream passes: the halo packer and
// the loop materialiser need to know *what to write into the ghost cells*
// at the global boundary, per axis.
//
// This pass parses ``stencil.bc`` and emits per-axis, per-side structured
// annotations that the halo-exchange + tile-lowering passes can consume
// directly.  It does NOT emit loops — it materialises a contract for the
// loop emitter, the same shape as ``StencilLowerPass``.
//
// Supported boundary conditions (Gap 2 ships periodic + reflect first;
// dirichlet/neumann go through the same code path with attribute-carried
// values).
//
//   periodic       — wrap: ghost(i) = field((i + N) mod N).
//   reflect        — mirror: ghost(-1) = field(0), ghost(-2) = field(1), ...
//   dirichlet(v)   — fixed: ghost = v   (defaults to 0 if no value).
//   neumann(v)     — derivative: ghost = field(0) + v   (defaults to 0).
//
// Form: a comma-separated list of per-axis specs.  A single token
// (``"periodic"``) is broadcast to every axis.  Examples:
//
//   "periodic"
//   "periodic,reflect"
//   "dirichlet(0.0),neumann(1.0)"
//
// Attributes written on ``stencil.apply``:
//
//   "stencil.bc.lowered"          : BoolAttr true   (sentinel)
//   "stencil.bc.modes"            : ArrayAttr<StringAttr>
//                                     one per axis: "periodic"|"reflect"|
//                                     "dirichlet"|"neumann".
//   "stencil.bc.values"           : ArrayAttr<FloatAttr>
//                                     one per axis: the BC value
//                                     (0.0 for periodic/reflect, parsed
//                                     constant for dirichlet/neumann).
//   "stencil.bc.has_value"        : ArrayAttr<BoolAttr>
//                                     one per axis: true ⇔ the BC mode
//                                     carries a meaningful scalar value.
//
// Pass is idempotent (sentinel-checked).  An unrecognised BC is left
// alone and a diagnostic is emitted; the pass returns success because the
// stencil.lowered attribute path stays valid (downstream still has the
// raw string), per Architecture Decision #21 (named diagnostic, not a
// silent no-op).
//===----------------------------------------------------------------------===//

#include "tessera/Dialect/Neighbors/Transforms/Passes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Visitors.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/raw_ostream.h"

#include <cstdlib>

using namespace mlir;

namespace tessera {
namespace neighbors {

// ---------------------------------------------------------------------------
// BC token parsing
// ---------------------------------------------------------------------------

namespace {

struct AxisBC {
  StringRef mode;       // "periodic" | "reflect" | "dirichlet" | "neumann"
  bool      hasValue;   // mode carries a meaningful scalar
  double    value;      // 0.0 unless explicitly parsed
  bool      recognised; // false ⇒ pass-through (downstream gets raw string)
};

static AxisBC parseAxisToken(StringRef token) {
  AxisBC out;
  out.hasValue   = false;
  out.value      = 0.0;
  out.recognised = false;
  token = token.trim();

  StringRef name = token;
  StringRef payload;
  // Split "mode(value)" into ("mode", "value")
  auto lParen = token.find('(');
  if (lParen != StringRef::npos) {
    name = token.substr(0, lParen).trim();
    StringRef rest = token.substr(lParen + 1);
    auto rParen = rest.find(')');
    if (rParen != StringRef::npos)
      payload = rest.substr(0, rParen).trim();
  }

  if (name == "periodic") {
    out.mode = "periodic";
    out.recognised = true;
    return out;
  }
  if (name == "reflect") {
    out.mode = "reflect";
    out.recognised = true;
    return out;
  }
  if (name == "dirichlet") {
    out.mode = "dirichlet";
    out.recognised = true;
    if (!payload.empty()) {
      // strtod is forgiving — accept "0", "1.5", "-2.0", "1e-3".
      std::string s = payload.str();
      char *endp = nullptr;
      double v = std::strtod(s.c_str(), &endp);
      if (endp != s.c_str()) {
        out.value    = v;
        out.hasValue = true;
      }
    } else {
      // Bare "dirichlet" → ghost = 0.
      out.hasValue = true;
      out.value    = 0.0;
    }
    return out;
  }
  if (name == "neumann") {
    out.mode = "neumann";
    out.recognised = true;
    if (!payload.empty()) {
      std::string s = payload.str();
      char *endp = nullptr;
      double v = std::strtod(s.c_str(), &endp);
      if (endp != s.c_str()) {
        out.value    = v;
        out.hasValue = true;
      }
    } else {
      out.hasValue = true;
      out.value    = 0.0;
    }
    return out;
  }
  // Unknown — leave recognised=false; caller emits diagnostic.
  out.mode = token;
  return out;
}

static llvm::SmallVector<AxisBC, 4> parseBcString(StringRef bc) {
  llvm::SmallVector<AxisBC, 4> axes;
  if (bc.empty()) return axes;
  // Split on commas (ignoring those inside parens — payloads here are
  // single scalars so no nested commas, but we still respect parens).
  size_t start = 0;
  int depth = 0;
  for (size_t i = 0; i < bc.size(); ++i) {
    char c = bc[i];
    if (c == '(') ++depth;
    else if (c == ')' && depth > 0) --depth;
    else if (c == ',' && depth == 0) {
      axes.push_back(parseAxisToken(bc.substr(start, i - start)));
      start = i + 1;
    }
  }
  axes.push_back(parseAxisToken(bc.substr(start)));
  return axes;
}

// Determine rank for broadcasting a single-token BC.  Prefer
// ``stencil.halo_width`` (one entry per axis) if present, else default 1.
static int64_t inferRank(Operation *applyOp) {
  if (auto hw = applyOp->getAttrOfType<ArrayAttr>("stencil.halo_width"))
    return static_cast<int64_t>(hw.size());
  if (auto hw = applyOp->getAttrOfType<ArrayAttr>("halo.width"))
    return static_cast<int64_t>(hw.size());
  return 1;
}

} // anonymous namespace

// ---------------------------------------------------------------------------
// Pass
// ---------------------------------------------------------------------------

struct BoundaryConditionLowerPass
    : public PassWrapper<BoundaryConditionLowerPass,
                         OperationPass<ModuleOp>> {

  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(BoundaryConditionLowerPass)

  StringRef getArgument() const final {
    return "tessera-boundary-condition-lower";
  }
  StringRef getDescription() const final {
    return "Lower stencil.bc string into per-axis structured BC annotations "
           "(periodic/reflect/dirichlet(v)/neumann(v))";
  }

  void runOnOperation() override {
    ModuleOp mod = getOperation();
    MLIRContext *ctx = mod.getContext();
    OpBuilder builder(ctx);

    mod.walk([&](Operation *op) -> WalkResult {
      if (op->getName().getStringRef() !=
          "tessera.neighbors.stencil.apply")
        return WalkResult::advance();

      // Sentinel — idempotent.
      if (op->hasAttr("stencil.bc.lowered"))
        return WalkResult::advance();

      auto bcAttr = op->getAttrOfType<StringAttr>("stencil.bc");
      if (!bcAttr) {
        // No BC declared — default to periodic on every axis; this matches
        // what the halo packer would do today.  Mark as lowered so the
        // pass stays idempotent.
        int64_t rank = inferRank(op);
        llvm::SmallVector<Attribute, 4> modes(
            rank, builder.getStringAttr("periodic"));
        llvm::SmallVector<Attribute, 4> values(
            rank, builder.getF64FloatAttr(0.0));
        llvm::SmallVector<Attribute, 4> hasVal(
            rank, builder.getBoolAttr(false));
        op->setAttr("stencil.bc.modes",     builder.getArrayAttr(modes));
        op->setAttr("stencil.bc.values",    builder.getArrayAttr(values));
        op->setAttr("stencil.bc.has_value", builder.getArrayAttr(hasVal));
        op->setAttr("stencil.bc.lowered",   builder.getBoolAttr(true));
        return WalkResult::advance();
      }

      auto parsed = parseBcString(bcAttr.getValue());
      int64_t rank = inferRank(op);

      // Broadcast a single-token BC across every axis.
      if (parsed.size() == 1 && rank > 1) {
        AxisBC bc = parsed.front();
        parsed.assign(rank, bc);
      }

      // If rank disagrees, accept what we have — emit a diagnostic but
      // do not block; the structured attribute simply matches the parsed
      // token count.  Downstream consumers can warn again on rank match.
      if (rank > 0 && static_cast<int64_t>(parsed.size()) != rank) {
        op->emitWarning()
            << "stencil.bc declares " << parsed.size()
            << " axes but stencil/halo width has rank " << rank
            << "; downstream may treat trailing axes as 'periodic'";
      }

      llvm::SmallVector<Attribute, 4> modes;
      llvm::SmallVector<Attribute, 4> values;
      llvm::SmallVector<Attribute, 4> hasVal;
      modes.reserve(parsed.size());
      values.reserve(parsed.size());
      hasVal.reserve(parsed.size());

      for (const AxisBC &bc : parsed) {
        if (!bc.recognised) {
          op->emitWarning()
              << "unrecognised stencil boundary condition '" << bc.mode
              << "' — leaving raw string for downstream";
          // Fall through to "periodic"-like sentinel so callers see a
          // populated attribute.
          modes.push_back(builder.getStringAttr(bc.mode));
          values.push_back(builder.getF64FloatAttr(0.0));
          hasVal.push_back(builder.getBoolAttr(false));
          continue;
        }
        modes.push_back(builder.getStringAttr(bc.mode));
        values.push_back(builder.getF64FloatAttr(bc.value));
        hasVal.push_back(builder.getBoolAttr(bc.hasValue));
      }

      op->setAttr("stencil.bc.modes",     builder.getArrayAttr(modes));
      op->setAttr("stencil.bc.values",    builder.getArrayAttr(values));
      op->setAttr("stencil.bc.has_value", builder.getArrayAttr(hasVal));
      op->setAttr("stencil.bc.lowered",   builder.getBoolAttr(true));

      return WalkResult::advance();
    });
  }
};

void registerBoundaryConditionLowerPass() {
  PassRegistration<BoundaryConditionLowerPass>();
}

} // namespace neighbors
} // namespace tessera
