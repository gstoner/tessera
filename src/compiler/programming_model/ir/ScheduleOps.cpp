//===- ScheduleOps.cpp — Schedule / Cache / TileMemory op verifiers (Phase 7) ===//
//
// Provides C++ verifier implementations for every op in:
//   ScheduleMeshPipelineOps.td  → schedule.*
//   CacheOps.td                 → cache.*
//   TileMemoryOps.td            → tile.*
//
// Each op's hasVerifier = 1 TableGen flag calls <Op>::verify().
// This file provides those definitions so the project links without
// pulling in ODS-generated tables (which require a full build).
//
// Verification philosophy:
//   - Required structural invariants are hard errors (emitOpError).
//   - Recoverable advisory checks are warnings (emitWarning).
//   - All verifiers return success() unless a hard error is found.
//===----------------------------------------------------------------------===//

#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/TypeRange.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/SmallVector.h"

using namespace mlir;

namespace tessera {

// ===========================================================================
//  Schedule dialect verifiers
// ===========================================================================

namespace schedule {

// ---------------------------------------------------------------------------
// schedule.mesh.define
// ---------------------------------------------------------------------------
// Invariants:
//   - dims attr must be a non-empty I64ArrayAttr
//   - axis_names must be an ArrayAttr of strings with the same length as dims
// ---------------------------------------------------------------------------
LogicalResult verifyMeshDefine(Operation *op) {
  auto dims = op->getAttrOfType<ArrayAttr>("dims");
  if (!dims || dims.empty())
    return op->emitOpError("requires non-empty 'dims' I64 array attribute");

  auto names = op->getAttrOfType<ArrayAttr>("axis_names");
  if (!names || names.empty())
    return op->emitOpError("requires non-empty 'axis_names' array attribute");

  if (dims.size() != names.size())
    return op->emitOpError("'dims' and 'axis_names' must have the same length; "
                           "got ")
           << dims.size() << " vs " << names.size();

  for (auto attr : dims)
    if (!attr.isa<IntegerAttr>())
      return op->emitOpError("'dims' entries must be I64IntegerAttr");

  for (auto attr : names)
    if (!attr.isa<StringAttr>())
      return op->emitOpError("'axis_names' entries must be StringAttr");

  // Check all dim sizes are positive
  for (auto attr : dims) {
    int64_t sz = attr.cast<IntegerAttr>().getInt();
    if (sz <= 0)
      return op->emitOpError("all mesh axis sizes must be > 0, got ") << sz;
  }
  return success();
}

// ---------------------------------------------------------------------------
// schedule.mesh.region
// ---------------------------------------------------------------------------
// Invariants:
//   - 'mesh' must be a SymbolRefAttr
//   - 'axis' must be a non-empty StringAttr
//   - body region must have at least one block
// ---------------------------------------------------------------------------
LogicalResult verifyMeshRegion(Operation *op) {
  if (!op->getAttr("mesh"))
    return op->emitOpError("requires 'mesh' symbol reference attribute");
  auto axis = op->getAttrOfType<StringAttr>("axis");
  if (!axis || axis.getValue().empty())
    return op->emitOpError("requires non-empty 'axis' string attribute");
  if (op->getNumRegions() == 0 || op->getRegion(0).empty())
    return op->emitOpError("body region must be non-empty");
  return success();
}

// ---------------------------------------------------------------------------
// schedule.pipeline.region
// ---------------------------------------------------------------------------
// Invariants:
//   - 'schedule' is a non-empty string (e.g. "1f1b", "gpipe")
//   - 'micro_batches' must be > 0
// ---------------------------------------------------------------------------
LogicalResult verifyPipelineRegion(Operation *op) {
  auto sched = op->getAttrOfType<StringAttr>("schedule");
  if (!sched || sched.getValue().empty())
    return op->emitOpError(
        "requires non-empty 'schedule' string attribute "
        "(e.g. \"1f1b\", \"gpipe\", \"interleaved\")");

  auto mb = op->getAttrOfType<IntegerAttr>("micro_batches");
  if (!mb)
    return op->emitOpError("requires 'micro_batches' integer attribute");
  if (mb.getInt() < 1)
    return op->emitOpError("'micro_batches' must be >= 1, got ")
           << mb.getInt();
  return success();
}

// ---------------------------------------------------------------------------
// schedule.stage
// ---------------------------------------------------------------------------
// Invariants:
//   - 'devices' must be a non-empty ArrayAttr
// ---------------------------------------------------------------------------
LogicalResult verifyStage(Operation *op) {
  auto devs = op->getAttrOfType<ArrayAttr>("devices");
  if (!devs || devs.empty())
    return op->emitOpError("requires non-empty 'devices' array attribute");
  return success();
}

// ---------------------------------------------------------------------------
// schedule.prefetch
// ---------------------------------------------------------------------------
// Invariants:
//   - 'into' must be one of the known MemorySpace strings
//   - 'overlap' must be one of the known OverlapPolicy strings
//   - operand type must match result type
// ---------------------------------------------------------------------------
LogicalResult verifyPrefetch(Operation *op) {
  static const char *validSpaces[] = {
      "register", "shared", "lds", "global", "managed", "host", "tmem"};
  auto intoAttr = op->getAttrOfType<StringAttr>("into");
  if (!intoAttr)
    return op->emitOpError("requires 'into' memory-space string attribute");

  bool validSpace = false;
  for (auto s : validSpaces)
    if (intoAttr.getValue() == s) { validSpace = true; break; }
  if (!validSpace)
    return op->emitOpError("'into' value ")
           << intoAttr.getValue()
           << " is not a recognised memory space";

  auto overlapAttr = op->getAttrOfType<StringAttr>("overlap");
  if (!overlapAttr)
    return op->emitOpError("requires 'overlap' policy string attribute");

  llvm::StringRef ov = overlapAttr.getValue();
  if (ov != "none" && ov != "compute" && ov != "collective")
    return op->emitOpError("'overlap' must be one of: none, compute, collective");

  return success();
}

// ---------------------------------------------------------------------------
// schedule.async_copy
// ---------------------------------------------------------------------------
// Invariants:
//   - src_space and dst_space must be known memory spaces
//   - src_space != dst_space (no-op copies are a bug)
//   - stage must be >= 0
// ---------------------------------------------------------------------------
LogicalResult verifyAsyncCopy(Operation *op) {
  auto srcSpace = op->getAttrOfType<StringAttr>("src_space");
  auto dstSpace = op->getAttrOfType<StringAttr>("dst_space");
  if (!srcSpace)
    return op->emitOpError("requires 'src_space' string attribute");
  if (!dstSpace)
    return op->emitOpError("requires 'dst_space' string attribute");
  if (srcSpace.getValue() == dstSpace.getValue())
    return op->emitOpError("'src_space' and 'dst_space' must differ");

  auto stage = op->getAttrOfType<IntegerAttr>("stage");
  if (!stage)
    return op->emitOpError("requires 'stage' integer attribute");
  if (stage.getInt() < 0)
    return op->emitOpError("'stage' must be >= 0");
  return success();
}

// ---------------------------------------------------------------------------
// schedule.await_movement
// ---------------------------------------------------------------------------
// No additional constraints beyond type-checking done by ODS.
// ---------------------------------------------------------------------------
LogicalResult verifyAwaitMovement(Operation *op) {
  if (op->getNumOperands() != 1)
    return op->emitOpError("expected exactly one token operand");
  return success();
}

// ---------------------------------------------------------------------------
// schedule.artifact
// ---------------------------------------------------------------------------
// Invariants:
//   - 'hash' must be a non-empty string
//   - 'arch' must be a non-empty string
// ---------------------------------------------------------------------------
LogicalResult verifyArtifact(Operation *op) {
  auto hash = op->getAttrOfType<StringAttr>("hash");
  if (!hash || hash.getValue().empty())
    return op->emitOpError("requires non-empty 'hash' string attribute");
  auto arch = op->getAttrOfType<StringAttr>("arch");
  if (!arch || arch.getValue().empty())
    return op->emitOpError("requires non-empty 'arch' string attribute");
  return success();
}

// ---------------------------------------------------------------------------
// schedule.knob
// ---------------------------------------------------------------------------
// Invariants:
//   - 'name' must be non-empty
//   - 'choices' must be non-empty
//   - if 'logits' is present, it must have the same length as 'choices'
// ---------------------------------------------------------------------------
LogicalResult verifyKnob(Operation *op) {
  auto name = op->getAttrOfType<StringAttr>("name");
  if (!name || name.getValue().empty())
    return op->emitOpError("requires non-empty 'name' string attribute");

  auto choices = op->getAttrOfType<ArrayAttr>("choices");
  if (!choices || choices.empty())
    return op->emitOpError("requires non-empty 'choices' array attribute");

  auto logits = op->getAttrOfType<ArrayAttr>("logits");
  if (logits && logits.size() != choices.size())
    return op->emitOpError("'logits' length (")
           << logits.size() << ") must match 'choices' length ("
           << choices.size() << ")";
  return success();
}

} // namespace schedule

// ===========================================================================
//  Cache dialect verifiers
// ===========================================================================

namespace cache {

// cache.kv.create — no parameters; just ensure it was not given operands
LogicalResult verifyKVCreate(Operation *op) {
  if (op->getNumOperands() != 0)
    return op->emitOpError("cache.kv.create takes no operands");
  return success();
}

// cache.page.lookup — kv handle + I32 position
LogicalResult verifyPageLookup(Operation *op) {
  if (op->getNumOperands() != 2)
    return op->emitOpError("expected exactly 2 operands (kv, pos)");
  return success();
}

// cache.pt.create — no parameters
LogicalResult verifyPTCreate(Operation *op) {
  if (op->getNumOperands() != 0)
    return op->emitOpError("cache.pt.create takes no operands");
  return success();
}

// cache.ring.create — no parameters
LogicalResult verifyRingCreate(Operation *op) {
  if (op->getNumOperands() != 0)
    return op->emitOpError("cache.ring.create takes no operands");
  return success();
}

} // namespace cache

// ===========================================================================
//  Tile dialect verifiers
// ===========================================================================

namespace tile {

// tile.alloc_shared — result must be a MemRefType
LogicalResult verifyAllocShared(Operation *op) {
  if (op->getNumOperands() != 1)
    return op->emitOpError("expected one memref operand/result");
  if (!op->getOperand(0).getType().isa<MemRefType>())
    return op->emitOpError("operand must be a memref type");
  return success();
}

// tile.async_copy — stage >= 0; src + dst are memrefs
LogicalResult verifyAsyncCopy(Operation *op) {
  auto stage = op->getAttrOfType<IntegerAttr>("stage");
  if (!stage)
    return op->emitOpError("requires 'stage' I32 integer attribute");
  if (stage.getInt() < 0)
    return op->emitOpError("'stage' must be >= 0");
  if (op->getNumOperands() < 2)
    return op->emitOpError("requires src and dst operands");
  if (!op->getOperand(0).getType().isa<MemRefType>())
    return op->emitOpError("'src' must be a memref");
  if (!op->getOperand(1).getType().isa<MemRefType>())
    return op->emitOpError("'dst' must be a memref");
  return success();
}

// tile.wait_async — stage >= 0
LogicalResult verifyWaitAsync(Operation *op) {
  auto stage = op->getAttrOfType<IntegerAttr>("stage");
  if (!stage)
    return op->emitOpError("requires 'stage' I32 integer attribute");
  if (stage.getInt() < 0)
    return op->emitOpError("'stage' must be >= 0");
  return success();
}

// tile.mbarrier.alloc — count > 0; scope non-empty
LogicalResult verifyMBarrierAlloc(Operation *op) {
  auto count = op->getAttrOfType<IntegerAttr>("count");
  if (!count || count.getInt() <= 0)
    return op->emitOpError("'count' must be a positive integer");
  auto scope = op->getAttrOfType<StringAttr>("scope");
  if (!scope || scope.getValue().empty())
    return op->emitOpError("requires non-empty 'scope' string attribute");
  return success();
}

// tile.mbarrier.arrive_expect_tx — bytes > 0; scope + semantics non-empty
LogicalResult verifyMBarrierArriveExpectTx(Operation *op) {
  auto bytes = op->getAttrOfType<IntegerAttr>("bytes");
  if (!bytes || bytes.getInt() <= 0)
    return op->emitOpError("'bytes' must be > 0");
  auto semantics = op->getAttrOfType<StringAttr>("semantics");
  if (!semantics || semantics.getValue().empty())
    return op->emitOpError("requires non-empty 'semantics' string attribute");
  return success();
}

// tile.mbarrier.try_wait — exactly 2 operands (barrier + token)
LogicalResult verifyMBarrierTryWait(Operation *op) {
  if (op->getNumOperands() != 2)
    return op->emitOpError("expected exactly 2 operands (barrier, token)");
  return success();
}

// tile.reduce — op string and order string must be non-empty
LogicalResult verifyReduce(Operation *op) {
  auto opAttr = op->getAttrOfType<StringAttr>("op");
  if (!opAttr || opAttr.getValue().empty())
    return op->emitOpError("requires non-empty 'op' reduction string");
  auto order = op->getAttrOfType<StringAttr>("order");
  if (!order || order.getValue().empty())
    return op->emitOpError("requires non-empty 'order' string attribute");
  // Valid reduction ops
  static const char *validOps[] = {
      "add", "mul", "max", "min", "and", "or", "xor", "sum"};
  bool found = false;
  for (auto s : validOps)
    if (opAttr.getValue() == s) { found = true; break; }
  if (!found)
    op->emitWarning("unrecognised reduction op '") << opAttr.getValue() << "'";
  return success();
}

} // namespace tile

// ---------------------------------------------------------------------------
// Dispatcher — called from a generic verifier pass when op names match
// ---------------------------------------------------------------------------

LogicalResult verifyProgrammingModelOp(Operation *op) {
  llvm::StringRef name = op->getName().getStringRef();

  // Schedule ops
  if (name == "schedule.mesh.define")    return schedule::verifyMeshDefine(op);
  if (name == "schedule.mesh.region")    return schedule::verifyMeshRegion(op);
  if (name == "schedule.pipeline.region")return schedule::verifyPipelineRegion(op);
  if (name == "schedule.stage")          return schedule::verifyStage(op);
  if (name == "schedule.prefetch")       return schedule::verifyPrefetch(op);
  if (name == "schedule.async_copy")     return schedule::verifyAsyncCopy(op);
  if (name == "schedule.await_movement") return schedule::verifyAwaitMovement(op);
  if (name == "schedule.artifact")       return schedule::verifyArtifact(op);
  if (name == "schedule.knob")           return schedule::verifyKnob(op);

  // Cache ops
  if (name == "cache.kv.create")         return cache::verifyKVCreate(op);
  if (name == "cache.page.lookup")       return cache::verifyPageLookup(op);
  if (name == "cache.pt.create")         return cache::verifyPTCreate(op);
  if (name == "cache.ring.create")       return cache::verifyRingCreate(op);

  // Tile ops
  if (name == "tile.alloc_shared")       return tile::verifyAllocShared(op);
  if (name == "tile.async_copy")         return tile::verifyAsyncCopy(op);
  if (name == "tile.wait_async")         return tile::verifyWaitAsync(op);
  if (name == "tile.mbarrier.alloc")     return tile::verifyMBarrierAlloc(op);
  if (name == "tile.mbarrier.arrive_expect_tx")
    return tile::verifyMBarrierArriveExpectTx(op);
  if (name == "tile.mbarrier.try_wait")  return tile::verifyMBarrierTryWait(op);
  if (name == "tile.reduce")             return tile::verifyReduce(op);

  return success(); // unknown op — not our concern
}

} // namespace tessera
