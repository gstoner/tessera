//===- AutodiffPass.cpp - Reverse-mode autodiff at Graph IR -----*- C++ -*-===//
//
// Phase F4 of docs/audit/execution_roadmap.md. Consumes the
// `Tessera_AdjointInterface` op trait (see
// `src/compiler/ir/include/Tessera/AdjointInterface.td`) to emit backward
// computation for any ``func.func`` annotated with the
// ``tessera.autodiff = "reverse"`` attribute.
//
// **Status (2026-05-09):** ODS interface scaffolded; this file documents the
// pass shape and registers a stub. The numpy-tape autodiff
// (`python/tessera/autodiff/`) remains the production path. F4 build
// integration is a follow-up requiring an MLIR 21 build tree wired against
// `Tessera_AdjointInterface`.
//
// Pass outline (when fully wired):
//
// 1. Identify funcs to differentiate (annotation-driven).
// 2. Walk the forward region top-down; for each op, capture (op, results,
//    operands) into a per-func tape vector.
// 3. Walk in reverse program order. For each op:
//    - Look up cotangents for its results in the cotangent map.
//    - If `op` implements `AdjointInterface`, dispatch to `buildAdjoint`.
//    - If `op->customAdjointName()` is non-empty, look up the Python-side
//      VJP via the autodiff registry bridge.
//    - Otherwise emit a diagnostic per Architecture Decision #21.
// 4. The cotangents at function arguments become the new function's
//    additional outputs (or, for `Module.parameters()`, are routed via a
//    side-channel that becomes `param.grad` in the Python wrapper).
//
// Effect-aware adjoint collective insertion (Phase F5) is a follow-on pass
// that runs after this one and inserts `reduce_scatter` / `all_gather` for
// adjoints of distributed parameters.
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir::tessera {

// Forward-declared in AdjointInterface.h.inc once ODS is wired into the build.
// class AdjointInterface;

namespace {

class AutodiffPass : public PassWrapper<AutodiffPass, OperationPass<func::FuncOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(AutodiffPass)

  StringRef getArgument() const final { return "tessera-autodiff"; }

  StringRef getDescription() const final {
    return "Reverse-mode autodiff via the Tessera AdjointInterface op trait. "
           "Phase F4 of docs/audit/execution_roadmap.md.";
  }

  void runOnOperation() override {
    // Stub: real implementation lands once the AdjointInterface ODS is
    // generated into the build (requires MLIR 21 tablegen on the include
    // path). Until then, this pass is a no-op so that pipelines that opt
    // into `--tessera-autodiff` don't break — they just don't get IR-level
    // adjoints, and continue to use the numpy-tape path from
    // `python/tessera/autodiff/`.
    //
    // When wiring the body in, see the four-step outline at the top of
    // this file.
  }
};

}  // namespace

std::unique_ptr<Pass> createAutodiffPass() {
  return std::make_unique<AutodiffPass>();
}

}  // namespace mlir::tessera
