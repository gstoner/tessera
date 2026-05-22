// SymbolicDimEqualityPass.cpp — Sprint V5 (2026-05-22)
//
// Closes the **4th and final** MLIR-verifier gap in SHAPE_SYSTEM.md §11.2:
// "No MLIR-level pass that re-checks symbolic dim equality after lowering."
//
// V1 scope (honestly minimal — V2 work is documented at the bottom):
//
//   1. Function-level binding consistency.  Each `func.func` may carry
//      `tessera.dim_bindings` (ArrayAttr<StringAttr>) of equations like
//      "D = H * Dh" and `tessera.dim_sizes` (DictionaryAttr) of
//      symbol → i64 concrete sizes.  When both LHS and RHS symbols of
//      a binding are bound, the verifier evaluates the product and
//      checks the equation.  Mismatch ⇒ `SYMDIM_BINDING_VIOLATION`.
//
//   2. Per-op dim-name contracts (best effort).  When `tessera.reshape`,
//      `tessera.transpose`, or `tessera.matmul` ops carry dim-name
//      attributes the pass checks the local contract:
//
//        transpose : in_dim_names ↔ out_dim_names must be a permutation
//                    (`SYMDIM_TRANSPOSE_VIOLATION` on mismatch).
//
//        reshape   : product of in_dim_names sizes (resolved via
//                    dim_sizes + bindings) == product of out_dim_names.
//                    (`SYMDIM_RESHAPE_VIOLATION` on mismatch.)
//
//        matmul    : K dim name on lhs (last) must equal K dim name on
//                    rhs (first, modulo transposeA/B).
//                    (`SYMDIM_MATMUL_CONTRACT_VIOLATION` on mismatch.)
//
//      Ops *without* the dim-name attributes are skipped — the pass is
//      a best-effort verifier, not a hard requirement.  This is what
//      makes it composable with downstream passes: existing IR
//      without dim-name annotations passes the verifier silently.
//
// V2 followups (NOT in this sprint — comments only):
//
//   * Automatic SSA-value flow propagation: read `tessera.dim_names`
//     on each SSA value, propagate through ops without explicit per-op
//     annotations.
//   * Cross-function symbol-table tracking (inter-procedural).
//   * Affine / Presburger reasoning beyond simple products (handle
//     `D = H * Dh + K`-style constraints).
//   * Integration into the named lowering pipelines so the check runs
//     automatically after DistributionLoweringPass.
//
// Stable diagnostic codes (for SHAPE_SYSTEM.md §11 + lit fixture
// `expected-error` matching):
//
//   SYMDIM_BINDING_VIOLATION
//   SYMDIM_RESHAPE_VIOLATION
//   SYMDIM_TRANSPOSE_VIOLATION
//   SYMDIM_MATMUL_CONTRACT_VIOLATION

#include "Tessera/Transforms/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"

#include <optional>

using namespace mlir;

namespace {

// ─────────────────────────────────────────────────────────────────────────
// Binding parser
// ─────────────────────────────────────────────────────────────────────────

struct Binding {
  std::string lhs;
  SmallVector<std::string, 4> rhs;
};

// Trim ASCII whitespace from both ends.
static StringRef trim(StringRef s) {
  while (!s.empty() && (s.front() == ' ' || s.front() == '\t'))
    s = s.drop_front();
  while (!s.empty() && (s.back() == ' ' || s.back() == '\t'))
    s = s.drop_back();
  return s;
}

// Parse "D = H * Dh" → {lhs="D", rhs={"H", "Dh"}}.
// Returns std::nullopt for malformed inputs.
static std::optional<Binding> parseBinding(StringRef raw) {
  auto eq = raw.find('=');
  if (eq == StringRef::npos) return std::nullopt;
  StringRef lhs = trim(raw.substr(0, eq));
  StringRef rhs = trim(raw.substr(eq + 1));
  if (lhs.empty() || rhs.empty()) return std::nullopt;
  Binding b;
  b.lhs = lhs.str();
  // Split RHS on '*' tokens.
  size_t pos = 0;
  while (pos < rhs.size()) {
    auto star = rhs.find('*', pos);
    StringRef tok = (star == StringRef::npos)
                        ? rhs.substr(pos)
                        : rhs.substr(pos, star - pos);
    tok = trim(tok);
    if (tok.empty()) return std::nullopt;
    b.rhs.push_back(tok.str());
    if (star == StringRef::npos) break;
    pos = star + 1;
  }
  return b;
}

// ─────────────────────────────────────────────────────────────────────────
// Symbol resolution
// ─────────────────────────────────────────────────────────────────────────

// Build a symbol → size table from a func.func's `tessera.dim_sizes`
// DictionaryAttr.  Missing or malformed attrs yield an empty map.
static llvm::DenseMap<StringRef, int64_t> readDimSizes(func::FuncOp fn) {
  llvm::DenseMap<StringRef, int64_t> out;
  auto dict = fn->getAttrOfType<DictionaryAttr>("tessera.dim_sizes");
  if (!dict) return out;
  for (NamedAttribute named : dict.getValue()) {
    if (auto i = dyn_cast<IntegerAttr>(named.getValue()))
      out[named.getName().strref()] = i.getInt();
  }
  return out;
}

// Read a function-level `tessera.dim_bindings` ArrayAttr<StringAttr>.
// Returns an empty list when missing.
static SmallVector<Binding, 4> readBindings(func::FuncOp fn) {
  SmallVector<Binding, 4> out;
  auto arr = fn->getAttrOfType<ArrayAttr>("tessera.dim_bindings");
  if (!arr) return out;
  for (Attribute a : arr) {
    if (auto s = dyn_cast<StringAttr>(a))
      if (auto parsed = parseBinding(s.getValue()))
        out.push_back(*parsed);
  }
  return out;
}

// Read an op-level dim-name list attribute.  Returns nullopt when
// absent — the verifier silently skips ops without annotations.
static std::optional<SmallVector<std::string, 4>>
readDimNames(Operation *op, StringRef attrName) {
  auto arr = op->getAttrOfType<ArrayAttr>(attrName);
  if (!arr) return std::nullopt;
  SmallVector<std::string, 4> names;
  for (Attribute a : arr) {
    if (auto s = dyn_cast<StringAttr>(a))
      names.push_back(s.getValue().str());
    else
      return std::nullopt;  // malformed
  }
  return names;
}

// Resolve a symbol via dim_sizes; if not directly in dim_sizes try
// resolving via one of the bindings (single level — no recursion in
// V1 to keep the algorithm trivially terminating).
static std::optional<int64_t>
resolveSymbol(StringRef sym,
              const llvm::DenseMap<StringRef, int64_t> &sizes,
              ArrayRef<Binding> bindings) {
  auto it = sizes.find(sym);
  if (it != sizes.end()) return it->second;
  for (const auto &b : bindings) {
    if (b.lhs != sym) continue;
    int64_t prod = 1;
    for (const auto &r : b.rhs) {
      auto rIt = sizes.find(r);
      if (rIt == sizes.end()) return std::nullopt;  // unresolved
      prod *= rIt->second;
    }
    return prod;
  }
  return std::nullopt;
}

// ─────────────────────────────────────────────────────────────────────────
// The pass
// ─────────────────────────────────────────────────────────────────────────

struct SymbolicDimEquality
    : public PassWrapper<SymbolicDimEquality, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(SymbolicDimEquality)

  StringRef getArgument() const override {
    return "tessera-symdim-equality";
  }
  StringRef getDescription() const override {
    return "Sprint V5 — verify function-level dim bindings + per-op "
           "dim-name contracts (reshape, transpose, matmul).";
  }

  // ── Function-level binding evaluation ─────────────────────────────────
  static LogicalResult checkBindings(
      func::FuncOp fn,
      ArrayRef<Binding> bindings,
      const llvm::DenseMap<StringRef, int64_t> &sizes) {
    bool failed = false;
    for (const auto &b : bindings) {
      auto lhsIt = sizes.find(b.lhs);
      if (lhsIt == sizes.end()) continue;  // LHS unbound; nothing to check
      int64_t prod = 1;
      bool rhsBound = true;
      for (const auto &r : b.rhs) {
        auto rIt = sizes.find(r);
        if (rIt == sizes.end()) { rhsBound = false; break; }
        prod *= rIt->second;
      }
      if (!rhsBound) continue;
      if (prod != lhsIt->second) {
        // Build the equation text in one string so the diagnostic is a
        // single coherent line (lit `expected-error` matches one msg).
        std::string rhsText;
        for (size_t i = 0; i < b.rhs.size(); ++i) {
          if (i) rhsText += " * ";
          rhsText += b.rhs[i];
        }
        fn.emitOpError("SYMDIM_BINDING_VIOLATION: binding '")
            << b.lhs << " = " << rhsText << "' violated: "
            << b.lhs << " = " << lhsIt->second
            << " but product of RHS = " << prod;
        failed = true;
      }
    }
    return failed ? failure() : success();
  }

  // ── Per-op contracts ──────────────────────────────────────────────────
  static LogicalResult checkTranspose(
      Operation *op,
      const llvm::DenseMap<StringRef, int64_t> & /*sizes*/) {
    auto in = readDimNames(op, "tessera.dim_names_in");
    auto out = readDimNames(op, "tessera.dim_names_out");
    if (!in || !out) return success();
    if (in->size() != out->size()) {
      op->emitOpError(
          "SYMDIM_TRANSPOSE_VIOLATION: dim_names_in has ")
          << in->size() << " names but dim_names_out has "
          << out->size();
      return failure();
    }
    // Multiset equality via sorting copies.
    SmallVector<std::string> a(in->begin(), in->end());
    SmallVector<std::string> b(out->begin(), out->end());
    std::sort(a.begin(), a.end());
    std::sort(b.begin(), b.end());
    if (a != b) {
      op->emitOpError(
          "SYMDIM_TRANSPOSE_VIOLATION: dim_names_in and "
          "dim_names_out are not a permutation");
      return failure();
    }
    return success();
  }

  static LogicalResult checkReshape(
      Operation *op,
      const llvm::DenseMap<StringRef, int64_t> &sizes,
      ArrayRef<Binding> bindings) {
    auto in = readDimNames(op, "tessera.dim_names_in");
    auto out = readDimNames(op, "tessera.dim_names_out");
    if (!in || !out) return success();
    auto product = [&](ArrayRef<std::string> names)
                       -> std::optional<int64_t> {
      int64_t p = 1;
      for (const auto &n : names) {
        auto v = resolveSymbol(n, sizes, bindings);
        if (!v) return std::nullopt;
        p *= *v;
      }
      return p;
    };
    auto inProd = product(*in);
    auto outProd = product(*out);
    if (!inProd || !outProd) return success();  // can't evaluate
    if (*inProd != *outProd) {
      op->emitOpError(
          "SYMDIM_RESHAPE_VIOLATION: dim_names_in product = ")
          << *inProd
          << " but dim_names_out product = " << *outProd
          << "; the equation cannot hold given the function's "
             "tessera.dim_sizes + tessera.dim_bindings";
      return failure();
    }
    return success();
  }

  static LogicalResult checkMatmul(
      Operation *op,
      const llvm::DenseMap<StringRef, int64_t> & /*sizes*/) {
    auto lhs = readDimNames(op, "tessera.dim_names_lhs");
    auto rhs = readDimNames(op, "tessera.dim_names_rhs");
    if (!lhs || !rhs) return success();
    if (lhs->empty() || rhs->empty()) return success();
    // For non-transposed matmul, the contracting dim is lhs.back()
    // and rhs.front().  Transposed variants (`transposeA`/`transposeB`)
    // shift these positions; V1 reads the canonical orientation.
    const std::string &kL = lhs->back();
    const std::string &kR = rhs->front();
    if (kL != kR) {
      op->emitOpError(
          "SYMDIM_MATMUL_CONTRACT_VIOLATION: lhs contracts on '")
          << kL << "' but rhs contracts on '" << kR
          << "' — symbol names must match";
      return failure();
    }
    return success();
  }

  // ── Sprint V2-flow (2026-05-22) — SSA-value dim-name propagation ─────
  //
  // Per-function map: SSA value → list of symbolic dim names.  Built
  // by seeding from `tessera.arg_dim_names` (one inner ArrayAttr per
  // function argument) and walking ops in linear program order
  // (Graph IR is largely straight-line at this stage; future SSA-CFG
  // pass-through can extend the walker to dominate-handle scf.for /
  // scf.if bodies).  Inferred per-op contracts feed the V1 verifier:
  // a mismatch between an inferred dim-name list and an explicit
  // `tessera.dim_names_in` annotation emits SYMDIM_FLOW_INCONSISTENCY.
  //
  // Diagnostic codes added:
  //   SYMDIM_FLOW_INCONSISTENCY — propagated dim-names disagree with
  //                                explicit per-op annotation.
  //
  // Propagation rules (V2 minimal):
  //   transpose: out_names = in_names (multiset preserved; positional
  //              info isn't tracked in V2 since the op carries no
  //              `perm` attribute).
  //   matmul:    out_names = lhs_names[:-1] + rhs_names[-1:]
  //              (canonical matmul shape; transposeA/B not handled
  //              in V2 — those declare it via explicit per-op attrs).
  //   reshape:   out_names = explicit `tessera.dim_names_out` if
  //              present; otherwise unknown (V2 doesn't infer factor
  //              splits without explicit declaration).
  //   any other op: propagation stops at the boundary (unknown).
  //
  // Flow tracking is OPT-IN: when neither tessera.arg_dim_names nor
  // any explicit per-op dim_names attrs exist, V2 falls through to
  // V1 behaviour (silent skip).

  using DimNameList = SmallVector<std::string, 4>;
  using ValueDimMap = llvm::DenseMap<Value, DimNameList>;

  static std::optional<DimNameList>
  readArgDimNames(func::FuncOp fn, unsigned argIdx) {
    auto arr = fn->getAttrOfType<ArrayAttr>("tessera.arg_dim_names");
    if (!arr) return std::nullopt;
    if (argIdx >= arr.size()) return std::nullopt;
    auto inner = dyn_cast<ArrayAttr>(arr[argIdx]);
    if (!inner) return std::nullopt;
    DimNameList names;
    for (Attribute a : inner) {
      if (auto s = dyn_cast<StringAttr>(a))
        names.push_back(s.getValue().str());
      else
        return std::nullopt;
    }
    return names;
  }

  static LogicalResult crossCheck(
      Operation *op, StringRef attrName,
      ArrayRef<std::string> propagated) {
    auto declared = readDimNames(op, attrName);
    if (!declared) return success();  // user didn't declare; nothing to compare
    if (declared->size() != propagated.size()
        || !std::equal(declared->begin(), declared->end(),
                       propagated.begin())) {
      op->emitOpError(
          "SYMDIM_FLOW_INCONSISTENCY: propagated dim-names disagree "
          "with explicit '")
          << attrName << "' annotation";
      return failure();
    }
    return success();
  }

  static LogicalResult propagateThroughFunction(
      func::FuncOp fn,
      const llvm::DenseMap<StringRef, int64_t> &sizes,
      ArrayRef<Binding> bindings) {
    (void)sizes;
    (void)bindings;
    ValueDimMap valueDims;
    // Seed function arguments from tessera.arg_dim_names.
    for (unsigned i = 0, e = fn.getNumArguments(); i < e; ++i) {
      auto names = readArgDimNames(fn, i);
      if (names) valueDims[fn.getArgument(i)] = *names;
    }
    if (valueDims.empty()) {
      // No flow seeds → fall through to V1 behaviour.  This keeps the
      // pass backward-compatible with functions that don't carry the
      // `tessera.arg_dim_names` attribute yet.
      return success();
    }
    bool failed = false;
    // Walk ops in program order (Graph IR straight-line block bodies).
    for (Block &block : fn.getBody()) {
      for (Operation &opRef : block) {
        Operation *op = &opRef;
        StringRef name = op->getName().getStringRef();

        if (name == "tessera.transpose") {
          if (op->getNumOperands() < 1 || op->getNumResults() < 1) continue;
          auto it = valueDims.find(op->getOperand(0));
          if (it == valueDims.end()) continue;
          // V2: propagate the multiset (per-position order isn't
          // recoverable without an explicit `perm` attr).  Cross-check
          // against explicit `tessera.dim_names_in` if present.
          if (mlir::failed(
                  crossCheck(op, "tessera.dim_names_in", it->second)))
            failed = true;
          // Propagate to result.  If user declared
          // tessera.dim_names_out explicitly we trust it; otherwise
          // mark the result with the multiset (positions unknown).
          auto declared = readDimNames(op, "tessera.dim_names_out");
          valueDims[op->getResult(0)] =
              declared ? *declared : it->second;

        } else if (name == "tessera.matmul") {
          if (op->getNumOperands() < 2 || op->getNumResults() < 1) continue;
          auto lhsIt = valueDims.find(op->getOperand(0));
          auto rhsIt = valueDims.find(op->getOperand(1));
          if (lhsIt == valueDims.end() || rhsIt == valueDims.end()) continue;
          if (mlir::failed(crossCheck(
                  op, "tessera.dim_names_lhs", lhsIt->second)))
            failed = true;
          if (mlir::failed(crossCheck(
                  op, "tessera.dim_names_rhs", rhsIt->second)))
            failed = true;
          // Infer result: lhs[:-1] + rhs[-1:].
          DimNameList out;
          for (size_t i = 0; i + 1 < lhsIt->second.size(); ++i)
            out.push_back(lhsIt->second[i]);
          if (!rhsIt->second.empty())
            out.push_back(rhsIt->second.back());
          valueDims[op->getResult(0)] = out;

        } else if (name == "tessera.reshape") {
          if (op->getNumResults() < 1) continue;
          // V2 doesn't infer reshape splits without explicit
          // declaration.  Use the user's explicit out names if
          // present; otherwise leave the result unbound.
          auto declared = readDimNames(op, "tessera.dim_names_out");
          if (declared) valueDims[op->getResult(0)] = *declared;
          // Cross-check input against propagated value.
          if (op->getNumOperands() >= 1) {
            auto it = valueDims.find(op->getOperand(0));
            if (it != valueDims.end()) {
              if (mlir::failed(crossCheck(
                      op, "tessera.dim_names_in", it->second)))
                failed = true;
            }
          }

        } else {
          // Unknown op: propagation stops at the boundary.  Result
          // SSA values are left unbound in the map.
        }
      }
    }
    return failed ? mlir::failure() : mlir::success();
  }

  void runOnOperation() override {
    bool anyFailure = false;
    getOperation().walk([&](func::FuncOp fn) {
      auto sizes = readDimSizes(fn);
      auto bindings = readBindings(fn);
      // Function-level binding equation check.
      if (failed(checkBindings(fn, bindings, sizes)))
        anyFailure = true;
      // Sprint V2-flow: SSA-value dim-name propagation + cross-checks.
      // Runs BEFORE the per-op contract walk so propagated names can
      // trigger SYMDIM_FLOW_INCONSISTENCY before the per-op walker
      // catches them as standalone violations.
      if (failed(propagateThroughFunction(fn, sizes, bindings)))
        anyFailure = true;
      // Per-op contracts.
      fn.walk([&](Operation *op) {
        StringRef name = op->getName().getStringRef();
        if (name == "tessera.transpose") {
          if (failed(checkTranspose(op, sizes))) anyFailure = true;
        } else if (name == "tessera.reshape") {
          if (failed(checkReshape(op, sizes, bindings))) anyFailure = true;
        } else if (name == "tessera.matmul") {
          if (failed(checkMatmul(op, sizes))) anyFailure = true;
        }
      });
    });
    if (anyFailure) signalPassFailure();
  }
};

}  // namespace

namespace tessera {
std::unique_ptr<Pass> createSymbolicDimEqualityPass() {
  return std::make_unique<SymbolicDimEquality>();
}
}  // namespace tessera
