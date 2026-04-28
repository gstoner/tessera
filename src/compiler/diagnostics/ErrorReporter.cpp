//===- ErrorReporter.cpp — Tessera structured diagnostics (Phase 6) --------===//
//
// Walks the MLIR location chain attached to each op and enriches shape-error
// diagnostics with the originating Python file + line number.
//
// The pass registers as a function-level pass so it operates after all IR
// transformations that may introduce shape errors.  It does NOT modify the
// IR; it only emits diagnostics via the MLIR DiagnosticEngine.
//
// Pass options
// ------------
//   --error-limit=N   Stop after N errors (0 = unlimited, default 20).
//   --warn-shape      Demote shape-mismatch errors to warnings.
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Visitors.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/raw_ostream.h"

#include <string>
#include <vector>
#include <optional>

namespace tessera {
namespace diagnostics {

// ---------------------------------------------------------------------------
// Helpers — extract Python source location from an MLIR Location
// ---------------------------------------------------------------------------

struct PyLoc {
  std::string file;
  unsigned line = 0;
  unsigned col  = 0;
};

/// Walk a (possibly fused / call-site) location chain looking for a
/// FileLineColLoc whose filename ends in ".py".
static std::optional<PyLoc> findPythonLoc(mlir::Location loc) {
  // FileLineColLoc — check directly
  if (auto fllc = mlir::dyn_cast<mlir::FileLineColLoc>(loc)) {
    llvm::StringRef fn = fllc.getFilename().strref();
    if (fn.ends_with(".py") || fn.ends_with(".pyi")) {
      return PyLoc{fn.str(), fllc.getLine(), fllc.getColumn()};
    }
  }
  // FusedLoc — recurse into children
  if (auto fused = mlir::dyn_cast<mlir::FusedLoc>(loc)) {
    for (mlir::Location child : fused.getLocations()) {
      if (auto found = findPythonLoc(child))
        return found;
    }
  }
  // CallSiteLoc — check caller
  if (auto cs = mlir::dyn_cast<mlir::CallSiteLoc>(loc)) {
    if (auto found = findPythonLoc(cs.getCaller()))
      return found;
    return findPythonLoc(cs.getCallee());
  }
  return std::nullopt;
}

// ---------------------------------------------------------------------------
// Shape-mismatch detection
// ---------------------------------------------------------------------------

struct ShapeMismatch {
  mlir::Operation* op;
  std::string message;          // human-readable description
  std::optional<PyLoc> py_loc; // Python origin (if found in loc chain)
};

/// Inspect an op for shape annotations left by the Python-side
/// ShapeInferenceEngine.  We look for:
///   - "tessera.expected_shape" attribute  (RankedTensorType)
///   - "tessera.actual_shape"   attribute  (RankedTensorType)
/// If both are present and they differ, record a mismatch.
static std::vector<ShapeMismatch> collectMismatches(mlir::ModuleOp mod) {
  std::vector<ShapeMismatch> out;

  mod.walk([&](mlir::Operation* op) {
    auto expected_attr = op->getAttr("tessera.expected_shape");
    auto actual_attr   = op->getAttr("tessera.actual_shape");
    if (!expected_attr || !actual_attr) return mlir::WalkResult::advance();

    // Both attrs must be ArrayAttr of IntegerAttr for comparison.
    auto exp_arr = mlir::dyn_cast<mlir::ArrayAttr>(expected_attr);
    auto act_arr = mlir::dyn_cast<mlir::ArrayAttr>(actual_attr);
    if (!exp_arr || !act_arr) return mlir::WalkResult::advance();

    if (exp_arr != act_arr) {
      std::string msg;
      llvm::raw_string_ostream os(msg);
      os << "shape mismatch on op '" << op->getName() << "': expected [";
      llvm::interleaveComma(exp_arr, os, [&](mlir::Attribute a) {
        if (auto ia = mlir::dyn_cast<mlir::IntegerAttr>(a))
          os << ia.getInt();
        else
          os << "?";
      });
      os << "] but got [";
      llvm::interleaveComma(act_arr, os, [&](mlir::Attribute a) {
        if (auto ia = mlir::dyn_cast<mlir::IntegerAttr>(a))
          os << ia.getInt();
        else
          os << "?";
      });
      os << "]";

      out.push_back(ShapeMismatch{op, os.str(), findPythonLoc(op->getLoc())});
    }
    return mlir::WalkResult::advance();
  });
  return out;
}

// ---------------------------------------------------------------------------
// Pass definition
// ---------------------------------------------------------------------------

struct ErrorReporterPass
    : public mlir::PassWrapper<ErrorReporterPass,
                               mlir::OperationPass<mlir::ModuleOp>> {

  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ErrorReporterPass)

  // Pass options
  mlir::Pass::Option<unsigned> errorLimit{
      *this, "error-limit",
      llvm::cl::desc("Maximum number of errors to emit (0 = unlimited)"),
      llvm::cl::init(20)};
  mlir::Pass::Option<bool> warnShape{
      *this, "warn-shape",
      llvm::cl::desc("Demote shape-mismatch errors to warnings"),
      llvm::cl::init(false)};

  llvm::StringRef getArgument() const override {
    return "tessera-error-reporter";
  }
  llvm::StringRef getDescription() const override {
    return "Emit structured diagnostics for shape mismatches with Python "
           "source locations";
  }

  void runOnOperation() override {
    mlir::ModuleOp mod = getOperation();
    auto mismatches = collectMismatches(mod);

    unsigned emitted = 0;
    for (auto& mm : mismatches) {
      if (errorLimit > 0 && emitted >= errorLimit) {
        mm.op->emitRemark()
            << "[tessera-error-reporter] error limit reached ("
            << errorLimit << "); suppressing further diagnostics";
        break;
      }

      // Build the primary diagnostic
      mlir::InFlightDiagnostic diag =
          warnShape
          ? mm.op->emitWarning(mm.message)
          : mm.op->emitError(mm.message);

      // Attach Python source location as a note if we found one
      if (mm.py_loc) {
        diag.attachNote()
            << "originated at Python " << mm.py_loc->file
            << ":" << mm.py_loc->line
            << ":" << mm.py_loc->col;
      }

      // Attach any tessera.diag_notes attribute as additional notes
      if (auto notes = mm.op->getAttrOfType<mlir::ArrayAttr>(
              "tessera.diag_notes")) {
        for (mlir::Attribute note : notes) {
          if (auto sa = mlir::dyn_cast<mlir::StringAttr>(note))
            diag.attachNote() << sa.getValue();
        }
      }

      ++emitted;
      if (!warnShape)
        signalPassFailure();
    }
  }
};

// ---------------------------------------------------------------------------
// Pass registration
// ---------------------------------------------------------------------------

void registerErrorReporterPass() {
  mlir::registerPass([]() -> std::unique_ptr<mlir::Pass> {
    return std::make_unique<ErrorReporterPass>();
  });
}

/// Add the pass to a pass manager with default options.
void addErrorReporterPass(mlir::PassManager& pm,
                          unsigned errorLimit = 20,
                          bool warnShape = false) {
  auto pass = std::make_unique<ErrorReporterPass>();
  pass->errorLimit = errorLimit;
  pass->warnShape  = warnShape;
  pm.addPass(std::move(pass));
}

} // namespace diagnostics
} // namespace tessera
