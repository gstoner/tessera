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
//   * Typed affine / Presburger reasoning. `D = H * Dh + K` is nonlinear
//     when both factors are symbolic and remains concrete-witness-only.
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
//   SYMDIM_BINDING_MALFORMED
//   SYMDIM_DIM_SIZES_MALFORMED

#include "Tessera/Transforms/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Analysis/Presburger/IntegerRelation.h"
#include "mlir/Analysis/Presburger/PresburgerSpace.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/ADT/STLExtras.h"

#include <optional>

using namespace mlir;

namespace {

// ─────────────────────────────────────────────────────────────────────────
// Binding parser
// ─────────────────────────────────────────────────────────────────────────

// Sprint V3a (2026-05-22): widened Binding shape from a single
// product to a sum-of-products concrete-witness form.
//
// V1 (V5) only accepted bindings of the form "D = H * Dh" — a
// single product term.  V3a extends this to
// "D = H * Dh + K" (sum of products).  Each term is still a
// product of symbols (no division, no constants), but multiple
// terms can be combined with `+`.  When evaluating the RHS the
// pass sums the products of resolved symbols.
//
// Examples accepted by V3a (and rejected by V5):
//   "D = H * Dh + K"             — flash attention block + residual
//   "S = A * B + C * D"          — sum of two contractions
//   "N = M * K"                  — V5-compatible (single term)
//   "Total = Group_a + Group_b"  — sum of singletons (each is a
//                                    1-symbol product)
//
// Out of scope for V3a:
//   constants ("D = H * Dh + 4") — would need integer literal
//                                   parsing; tracked as V3a.1
//   subtraction / negation       — V3a.2
//   parenthesized groups         — V3a.3
struct Binding {
  std::string lhs;
  // Each inner vector is one PRODUCT term.  V3a accepts multiple
  // terms; their sum is the RHS value.  V5 produced a single-term
  // sum-of-products (rhs = {{a,b,c}}), so V3a is fully backward
  // compatible.
  SmallVector<SmallVector<std::string, 4>, 2> terms;
};

// Trim ASCII whitespace from both ends.
static StringRef trim(StringRef s) {
  while (!s.empty() && (s.front() == ' ' || s.front() == '\t'))
    s = s.drop_front();
  while (!s.empty() && (s.back() == ' ' || s.back() == '\t'))
    s = s.drop_back();
  return s;
}

// Split a string on a single character at the top level (no nested
// parens, no escapes — V3a operates on a flat product/sum grammar).
static SmallVector<StringRef, 4>
splitOn(StringRef s, char delim) {
  SmallVector<StringRef, 4> parts;
  size_t pos = 0;
  while (pos < s.size()) {
    auto next = s.find(delim, pos);
    StringRef tok = (next == StringRef::npos)
                        ? s.substr(pos)
                        : s.substr(pos, next - pos);
    parts.push_back(tok);
    if (next == StringRef::npos) break;
    pos = next + 1;
  }
  return parts;
}

// Parse a single product term: "H * Dh * X" → {"H", "Dh", "X"}.
// Returns std::nullopt for malformed input (empty token).
static std::optional<SmallVector<std::string, 4>>
parseProductTerm(StringRef raw) {
  SmallVector<std::string, 4> syms;
  for (StringRef tok : splitOn(raw, '*')) {
    tok = trim(tok);
    if (tok.empty()) return std::nullopt;
    syms.push_back(tok.str());
  }
  return syms;
}

// Sprint V3a: parse `lhs = term + term + ...` where each term is a
// product of symbols.  Sprint V5's "lhs = sym * sym * sym" is the
// single-term case.
static std::optional<Binding> parseBinding(StringRef raw) {
  auto eq = raw.find('=');
  if (eq == StringRef::npos) return std::nullopt;
  StringRef lhs = trim(raw.substr(0, eq));
  StringRef rhs = trim(raw.substr(eq + 1));
  if (lhs.empty() || rhs.empty()) return std::nullopt;
  Binding b;
  b.lhs = lhs.str();
  // V3a: split RHS into product terms on '+', then parse each.
  for (StringRef termStr : splitOn(rhs, '+')) {
    auto term = parseProductTerm(trim(termStr));
    if (!term) return std::nullopt;
    b.terms.push_back(*term);
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

static bool transposedAttr(Operation *op, StringRef name) {
  auto attr = op->getAttrOfType<BoolAttr>(name);
  return attr && attr.getValue();
}

// readDimSizes and readBindings are best-effort readers: they skip what they
// cannot interpret. That is the wrong default for a verifier — a typo in a
// binding string disables exactly the equation it was written to check while
// the pass still reports success — so the well-formedness of both attributes is
// decided here, once, and fails closed. Out-of-scope grammar (constants,
// subtraction, parentheses) still parses; it resolves to unknown symbols and is
// skipped later, so only genuinely malformed entries reach these errors.
static LogicalResult checkDimAttrsWellFormed(func::FuncOp fn) {
  bool sawMalformed = false;
  if (auto dict = fn->getAttrOfType<DictionaryAttr>("tessera.dim_sizes")) {
    for (NamedAttribute named : dict.getValue()) {
      if (!isa<IntegerAttr>(named.getValue())) {
        fn.emitOpError("SYMDIM_DIM_SIZES_MALFORMED: symbol '")
            << named.getName().strref()
            << "' must map to an integer attribute";
        sawMalformed = true;
      }
    }
  }
  if (auto arr = fn->getAttrOfType<ArrayAttr>("tessera.dim_bindings")) {
    for (Attribute a : arr) {
      auto s = dyn_cast<StringAttr>(a);
      if (!s) {
        fn.emitOpError(
            "SYMDIM_BINDING_MALFORMED: every tessera.dim_bindings entry must "
            "be a string");
        sawMalformed = true;
        continue;
      }
      if (!parseBinding(s.getValue())) {
        fn.emitOpError("SYMDIM_BINDING_MALFORMED: cannot parse binding '")
            << s.getValue() << "'";
        sawMalformed = true;
      }
    }
  }
  return failure(sawMalformed);
}

// W4.2 (2026-08-12): consume a typed coefficient-vector carrier instead of
// reparsing expression strings.  A row represents
//   sum(coefficients[i] * symbols[i]) + constant {==,>=} 0.
// The legacy tessera.dim_bindings parser remains a compatibility verifier
// while producers migrate; it is not used to construct this Presburger set.
static LogicalResult checkTypedPresburger(func::FuncOp fn) {
  auto carrier =
      fn->getAttrOfType<DictionaryAttr>("tessera.presburger_constraints");
  if (!carrier)
    return success();

  auto malformed = [&](Twine detail) {
    fn.emitOpError("SYMDIM_PRESBURGER_MALFORMED: ") << detail;
    return failure();
  };
  auto version = carrier.getAs<IntegerAttr>("version");
  auto symbolsAttr = carrier.getAs<ArrayAttr>("symbols");
  auto constraintsAttr = carrier.getAs<ArrayAttr>("constraints");
  if (!version || version.getInt() != 1 || !symbolsAttr ||
      symbolsAttr.empty() || !constraintsAttr || constraintsAttr.empty())
    return malformed("expected version 1 with non-empty symbols and constraints");

  SmallVector<std::string> symbols;
  llvm::StringSet<> uniqueSymbols;
  for (Attribute attr : symbolsAttr) {
    auto symbol = dyn_cast<StringAttr>(attr);
    if (!symbol || symbol.getValue().empty() ||
        !uniqueSymbols.insert(symbol.getValue()).second)
      return malformed("symbols must be unique, non-empty strings");
    symbols.push_back(symbol.str());
  }

  unsigned modularRows = 0;
  for (Attribute attr : constraintsAttr) {
    if (auto constraint = dyn_cast<DictionaryAttr>(attr))
      if (auto relation = constraint.getAs<StringAttr>("relation"))
        modularRows += relation == "mod";
  }
  presburger::IntegerPolyhedron set(presburger::PresburgerSpace::getSetSpace(
      symbols.size(), /*numSymbols=*/0, modularRows));
  unsigned modularIndex = 0;
  for (Attribute attr : constraintsAttr) {
    auto constraint = dyn_cast<DictionaryAttr>(attr);
    if (!constraint)
      return malformed("every constraint must be a dictionary");
    auto relation = constraint.getAs<StringAttr>("relation");
    auto coefficients = constraint.getAs<DenseI64ArrayAttr>("coefficients");
    auto constant = constraint.getAs<IntegerAttr>("constant");
    if (!relation || !coefficients || !constant ||
        coefficients.size() != static_cast<int64_t>(symbols.size()))
      return malformed("constraint rows require relation, total coefficients, and constant");
    SmallVector<int64_t> row(set.getNumVars() + 1, 0);
    llvm::copy(coefficients.asArrayRef(), row.begin());
    row.back() = constant.getInt();
    if (relation == "eq")
      set.addEquality(row);
    else if (relation == "ge")
      set.addInequality(row);
    else if (relation == "mod") {
      auto modulus = constraint.getAs<IntegerAttr>("modulus");
      auto remainder = constraint.getAs<IntegerAttr>("remainder");
      if (!modulus || modulus.getInt() < 2 || !remainder ||
          remainder.getInt() < 0 || remainder.getInt() >= modulus.getInt())
        return malformed(
            "mod rows require modulus >= 2 and 0 <= remainder < modulus");
      // expr == remainder (mod modulus) iff there exists integer q such that
      // expr - remainder - modulus*q == 0.
      row[symbols.size() + modularIndex++] = -modulus.getInt();
      row.back() -= remainder.getInt();
      set.addEquality(row);
    }
    else
      return malformed("constraint relation must be 'eq', 'ge', or 'mod'");
  }

  // Concrete witnesses restrict the same integer set rather than selecting a
  // separate ad-hoc evaluation path.
  auto sizes = readDimSizes(fn);
  for (auto [index, symbol] : llvm::enumerate(symbols)) {
    auto witness = sizes.find(symbol);
    if (witness == sizes.end())
      continue;
    SmallVector<int64_t> equality(set.getNumVars() + 1, 0);
    equality[index] = 1;
    equality.back() = -witness->second;
    set.addEquality(equality);
  }
  if (set.isIntegerEmpty()) {
    fn.emitOpError(
        "SYMDIM_PRESBURGER_UNSATISFIABLE: typed affine constraints have no "
        "integer solution under the available dimension witnesses");
    return failure();
  }
  return success();
}

// Polynomial shape relations are not Presburger facts.  They are exact
// specialization guards and therefore require a complete concrete witness;
// accepting a partial witness would let downstream affine analyses assume a
// relation this pass never proved.
static LogicalResult checkNonlinearShapeGuards(func::FuncOp fn) {
  auto carrier =
      fn->getAttrOfType<DictionaryAttr>("tessera.nonlinear_shape_guards");
  if (!carrier)
    return success();
  auto malformed = [&](Twine detail) {
    fn.emitOpError("SYMDIM_NONLINEAR_GUARD_MALFORMED: ") << detail;
    return failure();
  };
  auto version = carrier.getAs<IntegerAttr>("version");
  auto symbolsAttr = carrier.getAs<ArrayAttr>("symbols");
  auto constraintsAttr = carrier.getAs<ArrayAttr>("constraints");
  if (!version || version.getInt() != 1 || !symbolsAttr ||
      symbolsAttr.empty() || !constraintsAttr || constraintsAttr.empty())
    return malformed("expected version 1 with non-empty symbols and constraints");

  SmallVector<std::string> symbols;
  llvm::StringSet<> uniqueSymbols;
  auto sizes = readDimSizes(fn);
  SmallVector<int64_t> witnesses;
  for (Attribute attr : symbolsAttr) {
    auto symbol = dyn_cast<StringAttr>(attr);
    if (!symbol || symbol.getValue().empty() ||
        !uniqueSymbols.insert(symbol.getValue()).second)
      return malformed("symbols must be unique, non-empty strings");
    auto witness = sizes.find(symbol.getValue());
    if (witness == sizes.end()) {
      fn.emitOpError("SYMDIM_NONLINEAR_GUARD_INCOMPLETE: symbol '")
          << symbol.getValue()
          << "' has no concrete tessera.dim_sizes witness";
      return failure();
    }
    if (witness->second < 0)
      return malformed("dimension witnesses must be nonnegative");
    symbols.push_back(symbol.str());
    witnesses.push_back(witness->second);
  }

  auto checkedAdd = [](int64_t lhs, int64_t rhs,
                       int64_t &result) -> bool {
    return !__builtin_add_overflow(lhs, rhs, &result);
  };
  auto checkedMul = [](int64_t lhs, int64_t rhs,
                       int64_t &result) -> bool {
    return !__builtin_mul_overflow(lhs, rhs, &result);
  };
  for (Attribute attr : constraintsAttr) {
    auto row = dyn_cast<DictionaryAttr>(attr);
    auto relation = row ? row.getAs<StringAttr>("relation") : nullptr;
    auto constant = row ? row.getAs<IntegerAttr>("constant") : nullptr;
    auto terms = row ? row.getAs<ArrayAttr>("terms") : nullptr;
    if (!row || !relation || !constant || !terms || terms.empty() ||
        (relation != "eq" && relation != "ge"))
      return malformed("rows require eq/ge relation, constant, and terms");
    int64_t value = constant.getInt();
    for (Attribute termAttr : terms) {
      auto term = dyn_cast<DictionaryAttr>(termAttr);
      auto coefficient = term ? term.getAs<IntegerAttr>("coefficient") : nullptr;
      auto powers = term ? term.getAs<DenseI64ArrayAttr>("powers") : nullptr;
      if (!term || !coefficient || !powers ||
          powers.size() != static_cast<int64_t>(symbols.size()))
        return malformed("every term requires a coefficient and total power row");
      int64_t monomial = coefficient.getInt();
      bool hasVariable = false;
      for (auto [witness, power] : llvm::zip_equal(witnesses, powers.asArrayRef())) {
        if (power < 0 || power > 16)
          return malformed("term powers must lie in [0, 16]");
        hasVariable |= power != 0;
        for (int64_t exponent = 0; exponent < power; ++exponent)
          if (!checkedMul(monomial, witness, monomial))
            return malformed("polynomial witness evaluation overflowed i64");
      }
      if (!hasVariable)
        return malformed("constant terms belong in the row constant");
      if (!checkedAdd(value, monomial, value))
        return malformed("polynomial witness evaluation overflowed i64");
    }
    const bool satisfied = relation == "eq" ? value == 0 : value >= 0;
    if (!satisfied) {
      fn.emitOpError("SYMDIM_NONLINEAR_GUARD_VIOLATION: polynomial ")
          << relation.getValue() << " row evaluated to " << value;
      return failure();
    }
  }
  return success();
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

// Sprint V3a (2026-05-22): evaluate the sum-of-products RHS of a
// binding using the dim_sizes table.  All symbols in every term
// must be bound for the binding to evaluate; otherwise std::nullopt.
static std::optional<int64_t>
evaluateBindingRHS(const Binding &b,
                   const llvm::DenseMap<StringRef, int64_t> &sizes) {
  int64_t sum = 0;
  for (const auto &term : b.terms) {
    int64_t prod = 1;
    for (const auto &r : term) {
      auto it = sizes.find(r);
      if (it == sizes.end()) return std::nullopt;  // unresolved
      prod *= it->second;
    }
    sum += prod;
  }
  return sum;
}

// Resolve a symbol via dim_sizes; if not directly in dim_sizes try
// resolving via one of the bindings (single level — no recursion to
// keep the algorithm trivially terminating).  Sprint V3a: bindings
// are now sum-of-products instead of single products.
static std::optional<int64_t>
resolveSymbol(StringRef sym,
              const llvm::DenseMap<StringRef, int64_t> &sizes,
              ArrayRef<Binding> bindings) {
  auto it = sizes.find(sym);
  if (it != sizes.end()) return it->second;
  for (const auto &b : bindings) {
    if (b.lhs != sym) continue;
    return evaluateBindingRHS(b, sizes);
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
      auto rhsValue = evaluateBindingRHS(b, sizes);
      if (!rhsValue) continue;  // RHS has unresolved symbols
      if (*rhsValue != lhsIt->second) {
        // Build the equation text in one string so the diagnostic is a
        // single coherent line (lit `expected-error` matches one msg).
        // Sprint V3a (2026-05-22): renders sum-of-products with ` + `
        // separating terms and ` * ` separating symbols within each
        // term.  Single-term bindings render unchanged from V5.
        std::string rhsText;
        for (size_t i = 0; i < b.terms.size(); ++i) {
          if (i) rhsText += " + ";
          for (size_t j = 0; j < b.terms[i].size(); ++j) {
            if (j) rhsText += " * ";
            rhsText += b.terms[i][j];
          }
        }
        // Pick the V5 wording when there's a single term, the V3a
        // sum-of-products wording when multi-term.  This keeps V5
        // lit fixtures matching `product of RHS` while V3a lit
        // fixtures match `value of RHS` for sums.
        if (b.terms.size() == 1) {
          fn.emitOpError("SYMDIM_BINDING_VIOLATION: binding '")
              << b.lhs << " = " << rhsText << "' violated: "
              << b.lhs << " = " << lhsIt->second
              << " but product of RHS = " << *rhsValue;
        } else {
          fn.emitOpError("SYMDIM_BINDING_VIOLATION: binding '")
              << b.lhs << " = " << rhsText << "' violated: "
              << b.lhs << " = " << lhsIt->second
              << " but value of RHS (sum of products) = "
              << *rhsValue;
        }
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
    // transposeA stores A as KxM and transposeB stores B as NxK, moving the
    // contracting position to the other end of that operand's name list.
    const std::string &kL = transposedAttr(op, "transposeA") ? lhs->front()
                                                             : lhs->back();
    const std::string &kR = transposedAttr(op, "transposeB") ? rhs->back()
                                                             : rhs->front();
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
    // The Python frontend emits standard func argument attributes. Keep the
    // older aggregate carrier for textual producers, with the argument-local
    // carrier as the authoritative source when present.
    auto inner = fn.getArgAttrOfType<ArrayAttr>(argIdx, "tessera.dim_names");
    if (!inner) {
      auto arr = fn->getAttrOfType<ArrayAttr>("tessera.arg_dim_names");
      if (!arr || argIdx >= arr.size()) return std::nullopt;
      inner = dyn_cast<ArrayAttr>(arr[argIdx]);
    }
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

  // ── Sprint V3b (2026-05-22) — interprocedural via func.call ──────────
  //
  // When a caller invokes a callee whose `tessera.arg_dim_names` is
  // declared, V3b cross-checks that the propagated dim-names at the
  // call-site operands match the callee's declared arg names
  // position-by-position.  Mismatch ⇒ SYMDIM_CALL_ARG_MISMATCH.
  //
  // Additionally, V3b reads `tessera.ret_dim_names` on the callee
  // (ArrayAttr-of-ArrayAttr, one inner list per result) and seeds the
  // caller's `valueDims` for each `func.call` result, so dim-names
  // flow through the call boundary.
  //
  // V3b only handles the *direct* call form (`func.call @callee(...)`)
  // — indirect calls via `func.call_indirect` are out of scope (they
  // need a symbol-table-free analysis).
  static std::optional<DimNameList>
  readRetDimNames(func::FuncOp fn, unsigned resIdx) {
    auto arr = fn->getAttrOfType<ArrayAttr>("tessera.ret_dim_names");
    if (!arr) return std::nullopt;
    if (resIdx >= arr.size()) return std::nullopt;
    auto inner = dyn_cast<ArrayAttr>(arr[resIdx]);
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

  // ── Structured-region symbolic-name propagation ───────────────────────
  //
  // scf.for body block:  args = [induction_var, iter_args...]
  //   - iter_args[i] inherits the caller-side init operand's dim-names
  //   - the body's `scf.yield` operands must match iter_args' dim-names
  //     (loop-invariant naming) ⇒ SYMDIM_LOOP_YIELD_MISMATCH on conflict
  //   - the scf.for result values inherit the iter_args' dim-names
  //
  // scf.if:
  //   - both regions terminate in `scf.yield`
  //   - the two branches must yield values with matching dim-names
  //     (so the result's dim-names are well-defined regardless of
  //     branch taken) ⇒ SYMDIM_IF_BRANCH_MISMATCH on conflict
  //   - the scf.if result values inherit the (matching) yield dim-names

  // Forward declaration: walking a block can recurse into scf.for /
  // scf.if regions, which themselves walk inner blocks.
  static bool propagateThroughBlock(Block &block,
                                    ValueDimMap &valueDims,
                                    SymbolTable *symtab);

  // Handle a single op in the program-order walker.  Returns true if
  // any cross-check or yield-match diagnostic fired.
  static bool propagateThroughOp(Operation *op,
                                 ValueDimMap &valueDims,
                                 SymbolTable *symtab) {
    bool failed = false;
    StringRef name = op->getName().getStringRef();

    if (name == "tessera.transpose") {
      if (op->getNumOperands() < 1 || op->getNumResults() < 1) return false;
      auto it = valueDims.find(op->getOperand(0));
      if (it == valueDims.end()) return false;
      // Copy before inserting the result: DenseMap may rehash on insertion,
      // invalidating `it` under LLVM 23's checked iterators.
      DimNameList inputNames = it->second;
      if (mlir::failed(crossCheck(op, "tessera.dim_names_in", inputNames)))
        failed = true;
      auto declared = readDimNames(op, "tessera.dim_names_out");
      valueDims[op->getResult(0)] = declared ? *declared : inputNames;

    } else if (name == "tessera.matmul") {
      if (op->getNumOperands() < 2 || op->getNumResults() < 1) return false;
      auto lhsIt = valueDims.find(op->getOperand(0));
      auto rhsIt = valueDims.find(op->getOperand(1));
      if (lhsIt == valueDims.end() || rhsIt == valueDims.end()) return false;
      if (mlir::failed(crossCheck(op, "tessera.dim_names_lhs", lhsIt->second)))
        failed = true;
      if (mlir::failed(crossCheck(op, "tessera.dim_names_rhs", rhsIt->second)))
        failed = true;
      // The result keeps every lhs name except the contracting one, then the
      // rhs's non-contracting name. Which end each of those sits on depends on
      // transposeA/transposeB, so reading a fixed position mislabels the
      // result and cascades into a downstream flow mismatch.
      bool transA = transposedAttr(op, "transposeA");
      bool transB = transposedAttr(op, "transposeB");
      DimNameList out;
      const DimNameList &lhsNames = lhsIt->second;
      for (size_t i = 0; i < lhsNames.size(); ++i)
        if (i != (transA ? 0 : lhsNames.size() - 1))
          out.push_back(lhsNames[i]);
      if (!rhsIt->second.empty())
        out.push_back(transB ? rhsIt->second.front() : rhsIt->second.back());
      valueDims[op->getResult(0)] = out;

    } else if (name == "tessera.reshape") {
      if (op->getNumResults() < 1) return false;
      auto declared = readDimNames(op, "tessera.dim_names_out");
      if (declared) valueDims[op->getResult(0)] = *declared;
      if (op->getNumOperands() >= 1) {
        auto it = valueDims.find(op->getOperand(0));
        if (it != valueDims.end()) {
          if (mlir::failed(crossCheck(op, "tessera.dim_names_in", it->second)))
            failed = true;
        }
      }

    } else if (auto call = dyn_cast<func::CallOp>(op)) {
      // Sprint V3b — interprocedural cross-check + return propagation.
      if (!symtab) return false;
      auto callee =
          dyn_cast_or_null<func::FuncOp>(symtab->lookup(call.getCallee()));
      if (!callee) return false;  // unresolved (extern) — skip
      // 1. Cross-check each call operand against the callee's
      //    declared arg_dim_names[i] when both are present.
      for (unsigned i = 0, e = call.getNumOperands(); i < e; ++i) {
        auto calleeArgNames = readArgDimNames(callee, i);
        if (!calleeArgNames) continue;
        auto it = valueDims.find(call.getOperand(i));
        if (it == valueDims.end()) continue;
        if (it->second.size() != calleeArgNames->size()
            || !std::equal(it->second.begin(), it->second.end(),
                           calleeArgNames->begin())) {
          op->emitOpError(
              "SYMDIM_CALL_ARG_MISMATCH: call to '@")
              << call.getCallee() << "' arg " << i
              << " propagated dim-names disagree with callee's "
              << "tessera.arg_dim_names";
          failed = true;
        }
      }
      // 2. Seed call result values from callee's ret_dim_names.
      for (unsigned i = 0, e = call.getNumResults(); i < e; ++i) {
        auto retNames = readRetDimNames(callee, i);
        if (retNames) valueDims[call.getResult(i)] = *retNames;
      }

    } else if (auto forOp = dyn_cast<scf::ForOp>(op)) {
      // Sprint V3c — scf.for iter_args propagation + yield invariance.
      // Seed the body's block args[1:] (skip induction var) from the
      // call-site init operands' propagated dim-names.
      Block &body = forOp.getRegion().front();
      auto iterArgs = body.getArguments().drop_front(1);  // skip ind var
      auto initOperands = forOp.getInitArgs();
      assert(iterArgs.size() == initOperands.size());
      // Save expected names per iter_arg so we can cross-check yield.
      SmallVector<std::optional<DimNameList>, 4> expectedNames;
      for (auto [iterArg, init] :
           llvm::zip_equal(iterArgs, initOperands)) {
        auto it = valueDims.find(init);
        if (it != valueDims.end()) {
          // Save the map value before inserting `iterArg`: insertion may
          // rehash DenseMap and invalidate `it` under LLVM 23.
          DimNameList initNames = it->second;
          valueDims[iterArg] = initNames;
          expectedNames.push_back(initNames);
        } else {
          expectedNames.push_back(std::nullopt);
        }
      }
      // Recurse into body block.
      if (propagateThroughBlock(body, valueDims, symtab))
        failed = true;
      // Find the scf.yield terminator and check each yielded value's
      // propagated dim-names match the corresponding iter_arg's
      // expected names.  Mismatch ⇒ SYMDIM_LOOP_YIELD_MISMATCH.
      Operation *term = body.getTerminator();
      if (auto yieldOp = dyn_cast<scf::YieldOp>(term)) {
        for (unsigned i = 0, e = yieldOp.getNumOperands(); i < e; ++i) {
          if (i >= expectedNames.size() || !expectedNames[i]) continue;
          auto yi = valueDims.find(yieldOp.getOperand(i));
          if (yi == valueDims.end()) continue;
          if (yi->second.size() != expectedNames[i]->size()
              || !std::equal(yi->second.begin(), yi->second.end(),
                             expectedNames[i]->begin())) {
            yieldOp->emitOpError(
                "SYMDIM_LOOP_YIELD_MISMATCH: scf.for yield operand ")
                << i << " dim-names disagree with the corresponding "
                << "iter_arg's dim-names (loop must be name-invariant)";
            failed = true;
          }
        }
      }
      // Seed scf.for results from iter_args' expected names.
      for (unsigned i = 0, e = forOp.getNumResults(); i < e; ++i) {
        if (i < expectedNames.size() && expectedNames[i])
          valueDims[forOp.getResult(i)] = *expectedNames[i];
      }

    } else if (auto whileOp = dyn_cast<scf::WhileOp>(op)) {
      // A while state must preserve its symbolic identity around both region
      // boundaries.  Seed before args from the initial state, carry those
      // facts through scf.condition into the after args, and require the
      // after yield to return the same names. Unknown names stay unknown;
      // known disagreement fails closed.
      SmallVector<std::optional<DimNameList>, 4> expectedNames;
      Block &before = whileOp.getBefore().front();
      for (auto [argument, init] :
           llvm::zip_equal(before.getArguments(), whileOp.getInits())) {
        auto it = valueDims.find(init);
        if (it == valueDims.end()) {
          expectedNames.push_back(std::nullopt);
          continue;
        }
        DimNameList names = it->second;
        valueDims[argument] = names;
        expectedNames.push_back(std::move(names));
      }
      if (propagateThroughBlock(before, valueDims, symtab))
        failed = true;

      auto condition = dyn_cast<scf::ConditionOp>(before.getTerminator());
      Block &after = whileOp.getAfter().front();
      if (!condition || condition.getArgs().size() != after.getNumArguments())
        return true;
      for (auto [index, pair] :
           llvm::enumerate(llvm::zip_equal(condition.getArgs(),
                                           after.getArguments()))) {
        Value forwarded = std::get<0>(pair);
        BlockArgument argument = std::get<1>(pair);
        // Copy the forwarded names out before inserting `argument`: the
        // insertion may rehash and invalidate `it`, and LLVM 23's checked
        // DenseMap iterators abort on any later use of the stale handle
        // (silently reading freed storage under NDEBUG).
        auto it = valueDims.find(forwarded);
        std::optional<DimNameList> forwardedNames;
        if (it != valueDims.end())
          forwardedNames = it->second;
        if (forwardedNames)
          valueDims[argument] = *forwardedNames;
        // Result `index` IS the value the condition forwards at `index` — that
        // is what an scf.while returns. Seeding it from the init/yield position
        // instead was only correct when the condition forwards the carried
        // state positionally; for `scf.condition(%c) %b` out of inits
        // (%x, %y), result 0 is %b while position 0 names %x.
        if (forwardedNames && index < whileOp.getNumResults())
          valueDims[whileOp.getResult(index)] = *forwardedNames;
        // For the same reason, only compare against the carried state's names
        // when this forward really is positional: `condition.getArgs()[index]`
        // must be the before-block argument at that index. Otherwise the two
        // sequences describe different values and the comparison would reject
        // a valid loop.
        bool positionalForward =
            index < before.getNumArguments() &&
            forwarded == before.getArgument(static_cast<unsigned>(index));
        if (positionalForward && index < expectedNames.size() &&
            expectedNames[index] && forwardedNames &&
            *forwardedNames != *expectedNames[index]) {
          condition.emitOpError(
              "SYMDIM_LOOP_YIELD_MISMATCH: scf.while condition operand ")
              << index << " changes the carried state's dim-names";
          failed = true;
        }
      }
      if (propagateThroughBlock(after, valueDims, symtab))
        failed = true;
      auto yield = dyn_cast<scf::YieldOp>(after.getTerminator());
      if (!yield || yield.getNumOperands() != expectedNames.size())
        return true;
      for (unsigned i = 0, e = yield.getNumOperands(); i < e; ++i) {
        auto it = valueDims.find(yield.getOperand(i));
        if (expectedNames[i] && it != valueDims.end() &&
            it->second != *expectedNames[i]) {
          yield.emitOpError(
              "SYMDIM_LOOP_YIELD_MISMATCH: scf.while yield operand ")
              << i << " dim-names disagree with the corresponding carried "
                      "state's dim-names";
          failed = true;
        }
        // Results are seeded from the CONDITION's forwarded values above, not
        // from here: yield operands feed the next iteration's carried state,
        // which is a different sequence with a different length.
      }

    } else if (auto ifOp = dyn_cast<scf::IfOp>(op)) {
      // Sprint V3c — scf.if: both branches must yield matching names.
      // Walk both regions; collect per-result dim-names from each
      // branch's terminator; cross-check; seed scf.if results.
      auto walkBranch = [&](Region &region)
          -> SmallVector<std::optional<DimNameList>, 4> {
        SmallVector<std::optional<DimNameList>, 4> out;
        if (region.empty()) return out;
        Block &block = region.front();
        if (propagateThroughBlock(block, valueDims, symtab))
          failed = true;
        Operation *term = block.getTerminator();
        if (auto yieldOp = dyn_cast<scf::YieldOp>(term)) {
          for (Value v : yieldOp.getOperands()) {
            auto it = valueDims.find(v);
            if (it != valueDims.end()) out.push_back(it->second);
            else out.push_back(std::nullopt);
          }
        }
        return out;
      };
      auto thenNames = walkBranch(ifOp.getThenRegion());
      auto elseNames = walkBranch(ifOp.getElseRegion());
      // Compare per-result.  Only when both branches have names do we
      // require agreement.  If `else` region is empty (no-else if),
      // skip the comparison — there's nothing to disagree with.
      bool hasElse = !ifOp.getElseRegion().empty();
      for (unsigned i = 0, e = ifOp.getNumResults(); i < e; ++i) {
        std::optional<DimNameList> chosen;
        if (i < thenNames.size() && thenNames[i]) chosen = thenNames[i];
        if (hasElse && i < elseNames.size() && elseNames[i]) {
          if (chosen) {
            if (chosen->size() != elseNames[i]->size()
                || !std::equal(chosen->begin(), chosen->end(),
                               elseNames[i]->begin())) {
              ifOp->emitOpError(
                  "SYMDIM_IF_BRANCH_MISMATCH: scf.if result ")
                  << i << " has different dim-names in then-branch "
                  << "vs else-branch";
              failed = true;
              continue;
            }
          } else {
            chosen = elseNames[i];
          }
        }
        if (chosen) valueDims[ifOp.getResult(i)] = *chosen;
      }
    }
    // Unknown op: propagation stops at the boundary.
    return failed;
  }

  static LogicalResult propagateThroughFunction(
      func::FuncOp fn,
      const llvm::DenseMap<StringRef, int64_t> &sizes,
      ArrayRef<Binding> bindings,
      SymbolTable *symtab) {
    (void)sizes;
    (void)bindings;
    ValueDimMap valueDims;
    // Seed function arguments from tessera.arg_dim_names.
    for (unsigned i = 0, e = fn.getNumArguments(); i < e; ++i) {
      auto names = readArgDimNames(fn, i);
      if (names) valueDims[fn.getArgument(i)] = *names;
    }
    if (valueDims.empty()) {
      // No flow seeds → fall through to V1 behaviour.  Backward-compat
      // path: existing V5/V6a/V6b functions keep working unchanged.
      return success();
    }
    bool failed = false;
    for (Block &block : fn.getBody()) {
      if (propagateThroughBlock(block, valueDims, symtab))
        failed = true;
    }
    return failed ? mlir::failure() : mlir::success();
  }

  void runOnOperation() override {
    bool anyFailure = false;
    // Sprint V3b — module-level symbol table for func.call resolution.
    SymbolTable symtab(getOperation());
    getOperation().walk([&](func::FuncOp fn) {
      auto sizes = readDimSizes(fn);
      auto bindings = readBindings(fn);
      if (failed(checkDimAttrsWellFormed(fn)))
        anyFailure = true;
      if (failed(checkTypedPresburger(fn)))
        anyFailure = true;
      if (failed(checkNonlinearShapeGuards(fn)))
        anyFailure = true;
      // Function-level binding equation check.
      if (failed(checkBindings(fn, bindings, sizes)))
        anyFailure = true;
      // Sprint V2-flow + V3b + V3c: SSA-value dim-name propagation
      // with interprocedural call-site and scf.for/scf.if region
      // support.  Runs BEFORE the per-op contract walk so propagated
      // names trigger SYMDIM_FLOW_INCONSISTENCY before per-op walker
      // catches them.
      if (failed(propagateThroughFunction(fn, sizes, bindings, &symtab)))
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

// Out-of-class definition: mutually recursive with propagateThroughOp.
bool SymbolicDimEquality::propagateThroughBlock(
    Block &block,
    ValueDimMap &valueDims,
    SymbolTable *symtab) {
  bool failed = false;
  for (Operation &opRef : block) {
    if (propagateThroughOp(&opRef, valueDims, symtab))
      failed = true;
  }
  return failed;
}

}  // namespace

namespace tessera {
std::unique_ptr<Pass> createSymbolicDimEqualityPass() {
  return std::make_unique<SymbolicDimEquality>();
}
}  // namespace tessera
