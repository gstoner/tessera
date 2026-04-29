//===- TesseraScheduleIR.cpp — Schedule IR dialect registration ------------===//
//
// Registers the Tessera Schedule IR dialect ops and provides helpers used by
// ScheduleToTilePass.
//
// Schedule IR op namespace: schedule.*
// Key ops (defined in ScheduleMeshPipelineOps.td):
//   schedule.mesh.define       — declare the device mesh (axes + sizes)
//   schedule.mesh.region       — scope a region to a mesh axis
//   schedule.pipeline.region   — declare pipeline schedule (1f1b, gpipe …)
//   schedule.stage             — one pipeline stage with device assignment
//   schedule.prefetch          — data prefetch with memory-space + overlap hint
//   schedule.async_copy        — async DMA copy between memory spaces
//   schedule.await_movement    — wait for an async copy token
//   schedule.artifact          — compiled kernel artifact (hash + arch)
//   schedule.knob              — auto-tuning knob (choices + optional logits)
//
// These ops use generic MLIR op registration (no ODS-generated tables).
// Attribute invariants are enforced by ScheduleOps.cpp verifiers.
//===----------------------------------------------------------------------===//

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Operation.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/StringRef.h"

using namespace mlir;

namespace tessera {
namespace schedule {

// ---------------------------------------------------------------------------
// Emit a schedule.mesh.define op in the given builder context.
//
// This helper is called by translation passes (e.g. from sdy.mesh attrs)
// when materialising a mesh descriptor in Schedule IR.
// ---------------------------------------------------------------------------
void emitMeshDefine(
    OpBuilder &builder,
    Location loc,
    llvm::ArrayRef<int64_t> dims,
    llvm::ArrayRef<llvm::StringRef> axisNames) {

  SmallVector<Attribute> dimAttrs;
  for (int64_t d : dims)
    dimAttrs.push_back(builder.getI64IntegerAttr(d));

  SmallVector<Attribute> nameAttrs;
  for (auto n : axisNames)
    nameAttrs.push_back(builder.getStringAttr(n));

  OperationState state(loc, "schedule.mesh.define");
  state.addAttribute("dims",       ArrayAttr::get(builder.getContext(), dimAttrs));
  state.addAttribute("axis_names", ArrayAttr::get(builder.getContext(), nameAttrs));
  builder.create(state);
}

// ---------------------------------------------------------------------------
// Collect all schedule.mesh.define ops in a module and return a map of
// axis_name → size.  Used by the Shardy export pass to build the mesh.
// ---------------------------------------------------------------------------
llvm::StringMap<int64_t> collectScheduleMeshAxes(ModuleOp mod) {
  llvm::StringMap<int64_t> result;
  mod.walk([&](Operation *op) -> WalkResult {
    if (op->getName().getStringRef() != "schedule.mesh.define")
      return WalkResult::advance();
    auto dims  = op->getAttrOfType<ArrayAttr>("dims");
    auto names = op->getAttrOfType<ArrayAttr>("axis_names");
    if (!dims || !names || dims.size() != names.size())
      return WalkResult::advance();
    for (size_t i = 0; i < dims.size(); ++i) {
      auto nameAttr = names[i].dyn_cast<StringAttr>();
      auto sizeAttr = dims[i].dyn_cast<IntegerAttr>();
      if (nameAttr && sizeAttr)
        result[nameAttr.getValue()] = sizeAttr.getInt();
    }
    return WalkResult::advance();
  });
  return result;
}

} // namespace schedule
} // namespace tessera
