//===- QueueVerifiers.cpp — tessera.queue.{create,push,pop} verifiers ────===//
//
// Sprint V8 (2026-05-22) — replaced 3 orphan `return success();` stubs with
// real per-op verifiers wired to the ODS-generated `hasVerifier = 1` slots
// in Queue.td.  Before this sprint these symbols (`verifyCreate` /
// `verifyPush` / `verifyPop`) were free functions that no code path ever
// called — pure dead-code stubs.
//
// Standalone lit testing note: the verifiers are exercised today
// through the FA-4 lowering passes (`tessera-lower-schedule` etc.)
// because MLIR's default type-syntax parser cannot round-trip types
// from a dotted-name dialect (`tessera.queue`).  The parser splits
// `!tessera.queue.tile_queue` on the first dot, routes it into the
// `tessera` Graph IR dialect, and rejects.  Tessera's Attn dialect
// avoids this by defining no TypeDefs; Queue exposes
// `tile_queue` and `token` types and so cannot be exercised through
// raw lit IR until either (a) the dialect name moves to a single
// segment (e.g. `tessera_queue`) or (b) the TD declares custom
// type assembly format that uses a different escape.  Both options
// are tracked as follow-ups; this sprint's contribution is the
// real verifier bodies + the public `registerQueueDialect()` so
// programmatic loads (the lowering passes) can rely on them.
//
// Stable diagnostic codes (for SHAPE_SYSTEM.md cross-linking + lit
// `expected-error` matching):
//
//   QUEUE_CREATE_OPERAND_COUNT   — create op must have zero operands.
//   QUEUE_PUSH_QUEUE_PROVENANCE  — push's queue operand must be defined
//                                  by `tessera.queue.create`.
//   QUEUE_PUSH_TILE_TYPE         — push's tile operand must be a ranked
//                                  tensor or memref.
//   QUEUE_POP_QUEUE_PROVENANCE   — pop's queue operand must be defined
//                                  by `tessera.queue.create`.
//   QUEUE_POP_TOKEN_PROVENANCE   — pop's dep token must be defined by
//                                  `tessera.queue.push`.
//   QUEUE_POP_TILE_TYPE          — pop's result tile must be a ranked
//                                  tensor or memref.
//
//===----------------------------------------------------------------------===//

#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/StringRef.h"

// ODS-generated declarations for the queue dialect.  Including these
// gives us the concrete op classes (`CreateOp`, `PushOp`, `PopOp`)
// whose `verify()` methods we define below.
#include "QueueDialect.h.inc"
#define GET_TYPEDEF_CLASSES
#include "QueueTypes.h.inc"
#define GET_OP_CLASSES
#include "QueueOps.h.inc"

using namespace mlir;

namespace tessera {
namespace queue {

namespace {

// Reusable helper: a "tile-shaped" value is a ranked tensor or a memref.
// Scalars / unranked tensors / token-like opaque types are rejected —
// the FA-4 pipeline only enqueues real tile values.
static bool isTileShapedType(Type ty) {
  return mlir::isa<RankedTensorType, MemRefType>(ty);
}

// Returns the defining op of ``v`` if it has one; nullptr otherwise.
// Pure helper — block arguments and external values have no defining
// op, so this returns nullptr for them.
static Operation *definingOp(Value v) { return v.getDefiningOp(); }

// Match a defining op against a stable op-name string.  We do not
// require the concrete C++ type because that creates an awkward
// circular include (this file already pulls in QueueOps.h.inc; the
// op-name string suffices for provenance checks).
static bool definedBy(Value v, StringRef opName) {
  if (auto *op = definingOp(v))
    return op->getName().getStringRef() == opName;
  return false;
}

}  // namespace

// ── CreateOp ─────────────────────────────────────────────────────────────
//
// The result type is constrained by ODS to !tessera.queue.tile_queue;
// the assembly format also enforces zero operands.  This verifier guards
// against a future TD revision that accidentally adds an operand —
// catching it at the IR layer rather than the lowering layer.
LogicalResult CreateOp::verify() {
  if (this->getOperation()->getNumOperands() != 0) {
    return emitOpError(
        "QUEUE_CREATE_OPERAND_COUNT: tessera.queue.create must have zero "
        "operands; got ") << this->getOperation()->getNumOperands();
  }
  return success();
}

// ── PushOp ───────────────────────────────────────────────────────────────
//
// Data-flow well-formedness on the producer end of the FA-4 pipeline:
//   - the queue handle must come from a `tessera.queue.create`
//   - the tile must be a ranked tensor or memref (not a scalar or
//     an opaque token)
LogicalResult PushOp::verify() {
  if (!definedBy(getQ(), "tessera.queue.create")) {
    return emitOpError(
        "QUEUE_PUSH_QUEUE_PROVENANCE: queue handle must be defined by "
        "`tessera.queue.create`");
  }
  if (!isTileShapedType(getTile().getType())) {
    return emitOpError(
        "QUEUE_PUSH_TILE_TYPE: tile operand must be a ranked tensor or "
        "memref; got ") << getTile().getType();
  }
  return success();
}

// ── PopOp ────────────────────────────────────────────────────────────────
//
// Data-flow well-formedness on the consumer end:
//   - the queue handle must come from a `tessera.queue.create`
//   - the dep token must come from a `tessera.queue.push`
//   - the result must be a ranked tensor or memref
LogicalResult PopOp::verify() {
  if (!definedBy(getQ(), "tessera.queue.create")) {
    return emitOpError(
        "QUEUE_POP_QUEUE_PROVENANCE: queue handle must be defined by "
        "`tessera.queue.create`");
  }
  if (!definedBy(getDep(), "tessera.queue.push")) {
    return emitOpError(
        "QUEUE_POP_TOKEN_PROVENANCE: dep token must be defined by "
        "`tessera.queue.push`");
  }
  if (!isTileShapedType(getTile().getType())) {
    return emitOpError(
        "QUEUE_POP_TILE_TYPE: result tile must be a ranked tensor or "
        "memref; got ") << getTile().getType();
  }
  return success();
}

}  // namespace queue
}  // namespace tessera
