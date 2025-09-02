
//===- ResolveTesseraAttrsPass.cpp ----------------------------------------===//
// A tiny MLIR pass that converts symbolic placeholders on transform ops into
// concrete constants.
//
// Supported patterns (generic form):
//   - transform.structured.tile op with `tile_sizes_sym` : ArrayAttr<StringAttr>
//       -> sets DenseI64ArrayAttr("tile_sizes")
//   - transform.structured.pipeline op with `stages_sym` : StringAttr
//       -> sets IntegerAttr("stages")
//
// Resolution Sources (in order):
//   1) Command-line k/v: --tessera-resolve="BLOCK_M=128" (prefixed with "tessera." internally)
//   2) Module attributes: e.g., module attributes {tessera.BLOCK_M = 128 : i64}
//   3) Function attributes (on the matched func.func)
//
// This pass is intentionally conservative and best-effort; unresolvable symbols
// are left intact so downstream errors point to missing values.
//----------------------------------------------------------------------------//
#include "tessera/ResolveTesseraAttrsPass.h"

#include "mlir/IR/AsmState.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Operation.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"

using namespace mlir;

static llvm::cl::list<std::string> ClResolvePairs(
    "tessera-resolve",
    llvm::cl::desc("Key=Value pairs to resolve (e.g., BLOCK_M=128)"),
    llvm::cl::ZeroOrMore);

namespace {
struct ResolveTesseraAttrsPass
    : public PassWrapper<ResolveTesseraAttrsPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ResolveTesseraAttrsPass)

  StringRef getArgument() const final { return "tessera-resolve-attrs"; }
  StringRef getDescription() const final {
    return "Resolve Tessera transform placeholders into constants";
  }

  void getDependentDialects(DialectRegistry &registry) const override {}

  void runOnOperation() final {
    ModuleOp mod = getOperation();
    // Build a resolution map from CLI and module attrs
    llvm::StringMap<int64_t> kv;
    for (auto &s : ClResolvePairs) {
      auto eq = s.find('=');
      if (eq == std::string::npos) continue;
      std::string k = s.substr(0, eq);
      std::string v = s.substr(eq + 1);
      int64_t vi = 0;
      (void)llvm::to_integer(v, vi, 10);
      kv.try_emplace(("tessera." + k), vi);
    }
    // Module attrs
    for (auto named : mod->getAttrs()) {
      if (named.getName().str().starts_with("tessera.")) {
        if (auto ia = dyn_cast<IntegerAttr>(named.getValue())) {
          kv.try_emplace(named.getName().str(), ia.getInt());
        }
      }
    }

    // Walk all ops; look for transform.structured.* by name to avoid deps.
    mod.walk([&](Operation *op) {
      StringRef name = op->getName().getStringRef();
      if (name == "transform.structured.tile") {
        if (auto symAttr = op->getAttrOfType<ArrayAttr>("tile_sizes_sym")) {
          SmallVector<int64_t, 4> sizes;
          sizes.reserve(symAttr.size());
          bool ok = true;
          for (Attribute a : symAttr) {
            auto sa = dyn_cast<StringAttr>(a);
            if (!sa) { ok = false; break; }
            auto it = kv.find(sa.getValue());
            if (it == kv.end()) { ok = false; break; }
            sizes.push_back(it->second);
          }
          if (ok) {
            auto dense = DenseI64ArrayAttr::get(op->getContext(), sizes);
            op->setAttr("tile_sizes", dense);
            op->removeAttr("tile_sizes_sym");
          }
        }
      } else if (name == "transform.structured.pipeline") {
        if (auto sa = op->getAttrOfType<StringAttr>("stages_sym")) {
          auto it = kv.find(sa.getValue());
          if (it != kv.end()) {
            auto ia = IntegerAttr::get(IntegerType::get(op->getContext(), 64), it->second);
            op->setAttr("stages", ia);
            op->removeAttr("stages_sym");
          }
        }
      } else if (name == "func.func") {
        // Optionally record function-level tessera.* attrs into map (lower priority than module).
        for (auto named : op->getAttrs()) {
          if (named.getName().str().starts_with("tessera.")) {
            if (auto ia = dyn_cast<IntegerAttr>(named.getValue())) {
              kv.try_emplace(named.getName().str(), ia.getInt());
            }
          }
        }
      }
    });
  }
};
} // end anonymous namespace

std::unique_ptr<Pass> mlir::createResolveTesseraAttrsPass() {
  return std::make_unique<ResolveTesseraAttrsPass>();
}
