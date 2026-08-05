//===- ScheduleDialect.cpp — Schedule IR registration and verification ---===//

#include "tessera/ProgrammingModel/ScheduleDialect.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringExtras.h"

using namespace mlir;
using namespace tessera::schedule;

#include "ScheduleDialect.cpp.inc"

#define GET_OP_CLASSES
#include "ScheduleMeshPipelineOps.cpp.inc"

void ScheduleDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "ScheduleMeshPipelineOps.cpp.inc"
      >();
}

LogicalResult MeshDefineOp::verify() {
  if (getDims().empty())
    return emitOpError("requires at least one mesh dimension");
  if (getDims().size() != getAxisNames().size())
    return emitOpError("requires one axis name per mesh dimension");
  for (Attribute attr : getDims()) {
    auto value = dyn_cast<IntegerAttr>(attr);
    if (!value || value.getInt() <= 0)
      return emitOpError("mesh dimensions must be positive integers");
  }
  for (Attribute attr : getAxisNames())
    if (!isa<StringAttr>(attr))
      return emitOpError("mesh axis names must be strings");
  return success();
}

LogicalResult MeshRegionOp::verify() {
  if (getAxis().empty())
    return emitOpError("requires a non-empty mesh axis");
  if (getBody().empty())
    return emitOpError("requires a non-empty body");
  if (!llvm::hasSingleElement(getBody()))
    return emitOpError("requires exactly one body block");
  Operation *terminator = getBody().front().getTerminator();
  if (!terminator || terminator->getName().getStringRef() != "schedule.yield")
    return emitOpError("body must terminate with schedule.yield");
  // Legacy marker-only regions may yield an informational value while exposing
  // no SSA result. Preserve those until their owning passes migrate; once the
  // region declares results, enforce the real value contract exactly.
  if (getNumResults() == 0)
    return success();
  if (terminator->getNumOperands() != getNumResults())
    return emitOpError("yield operand count must match region result count");
  for (auto [yielded, result] :
       llvm::zip_equal(terminator->getOperands(), getResults()))
    if (yielded.getType() != result.getType())
      return emitOpError("yield operand types must match region result types");
  return success();
}

LogicalResult PipelineRegionOp::verify() {
  if (getSchedule().empty())
    return emitOpError("requires a non-empty pipeline schedule");
  if (getMicroBatches() < 1)
    return emitOpError("requires micro_batches >= 1");
  if (getBody().empty())
    return emitOpError("requires a non-empty body");
  return success();
}

LogicalResult StageOp::verify() {
  if (getDevices().empty())
    return emitOpError("requires at least one device");
  if (getBody().empty())
    return emitOpError("requires a non-empty body");
  return success();
}

LogicalResult TileOp::verify() {
  if (getSource().empty() || getResult().empty())
    return emitOpError("requires non-empty source and result identities");
  if (getOrdinalAttr().getInt() < 0)
    return emitOpError("requires ordinal >= 0");
  for (StringRef name : {"tile_m", "tile_n", "tile_k", "tile_h", "tile_w",
                         "tile_c"}) {
    if (auto value = (*this)->getAttrOfType<IntegerAttr>(name);
        value && value.getInt() <= 0)
      return emitOpError("optional tile dimensions must be positive");
  }
  return success();
}

LogicalResult MatmulOp::verify() {
  if (getSubject().getType() != getScheduled().getType())
    return emitOpError("must preserve the scheduled Graph value type");
  if (getArtifactHash().size() != 64 ||
      !llvm::all_of(getArtifactHash(), [](char c) {
        return (c >= '0' && c <= '9') || (c >= 'a' && c <= 'f');
      }))
    return emitOpError("requires a lowercase SHA-256 artifact_hash");
  if (getArch().empty())
    return emitOpError("requires a non-empty architecture");
  if (getTileM() <= 0 || getTileN() <= 0 || getTileK() <= 0)
    return emitOpError("tile dimensions must be positive");
  if (getWarps() != 1 && getWarps() != 4)
    return emitOpError("warps must be 1 or 4");
  if (getPipelineDepth() <= 0)
    return emitOpError("pipeline_depth must be positive");
  if (getStorage().empty() || getAccum().empty())
    return emitOpError("requires explicit storage and accumulation types");
  if (getALayout() != "row_major" || getBLayout() != "col_major")
    return emitOpError("initial matmul contract requires row/col layouts");
  if (getRasterOrder() != "row_major")
    return emitOpError("initial matmul contract requires row-major raster order");
  if (getRasterGroup() <= 0)
    return emitOpError("raster_group must be positive");
  return success();
}

LogicalResult WarpOp::verify() {
  if (getRole().empty())
    return emitOpError("requires a non-empty warp role");
  if (auto count = getCountAttr(); count && count.getInt() <= 0)
    return emitOpError("warp count must be positive");
  if (getBody().empty())
    return emitOpError("requires a non-empty body");
  return success();
}

LogicalResult OptimizerShardOp::verify() {
  if (getAxis().empty())
    return emitOpError("requires a non-empty shard axis");
  if (getSubject().getType() != getSharded().getType())
    return emitOpError("must preserve the subject type");
  if (auto partitions = getPartitionsAttr();
      partitions && partitions.getInt() <= 0)
    return emitOpError("partitions must be positive");
  return success();
}

LogicalResult PrefetchOp::verify() {
  if (getSource().getType() != getStaged().getType())
    return emitOpError("must preserve the source type");
  if (getInto().empty())
    return emitOpError("requires a destination memory space");
  StringRef overlap = getOverlap();
  if (overlap != "none" && overlap != "compute" && overlap != "collective")
    return emitOpError("overlap must be none, compute, or collective");
  return success();
}

LogicalResult AsyncCopyOp::verify() {
  if (getSrcSpace().empty() || getDstSpace().empty())
    return emitOpError("requires source and destination memory spaces");
  if (getSrcSpace() == getDstSpace())
    return emitOpError("source and destination memory spaces must differ");
  if (getStageAttr().getInt() < 0)
    return emitOpError("requires stage >= 0");
  StringRef overlap = getOverlap();
  if (overlap != "none" && overlap != "compute" && overlap != "collective")
    return emitOpError("overlap must be none, compute, or collective");
  return success();
}

LogicalResult AwaitMovementOp::verify() { return success(); }

LogicalResult ArtifactOp::verify() {
  if (getHash().empty() || getArch().empty() || getShapeKey().empty())
    return emitOpError("requires non-empty hash, arch, and shape_key");
  return success();
}

LogicalResult KnobOp::verify() {
  if (getName().empty())
    return emitOpError("requires a non-empty knob name");
  if (getChoices().empty())
    return emitOpError("requires at least one choice");
  if (auto logits = getLogitsAttr();
      logits && logits.size() != getChoices().size())
    return emitOpError("logits and choices must have equal length");
  if (getSubject().getType() != getSelected().getType())
    return emitOpError("must preserve the subject type");
  return success();
}
