
//===- CPXTargetIROps.cpp - Custom op verifiers for CPX dialect ops --------===//
//
// Provides custom `verify()` implementations for ops that need constraints
// beyond what ODS can express declaratively.
//
// KVExportOp  — chunk_bytes must be > 0 and a power of two
// KVImportOp  — dst must be a writable memref
// AttnPrefillFusedOp — seq_len must be > 0; kv_cache must be declared
//
//===-----------------------------------------------------------------------===//

#include "tessera/Target/NVRubinCPX/NVRubinCPX.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Operation.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/APInt.h"

using namespace mlir;
using namespace tessera::target;

//===----------------------------------------------------------------------===//
// KVExportOp
//===----------------------------------------------------------------------===//

LogicalResult KVExportOp::verify() {
  int64_t chunkBytes = getChunkBytes();
  if (chunkBytes <= 0)
    return emitOpError("chunk_bytes must be a positive integer (got ")
           << chunkBytes << ")";

  // chunk_bytes should be a power of two (transport alignment requirement)
  if (chunkBytes & (chunkBytes - 1))
    return emitOpError("chunk_bytes must be a power of two for PCIe/CX9 "
                       "alignment (got ")
           << chunkBytes << ")";

  return success();
}

//===----------------------------------------------------------------------===//
// KVImportOp
//===----------------------------------------------------------------------===//

LogicalResult KVImportOp::verify() {
  // dst must be a MemRefType (not a ranked tensor)
  if (!getDst().getType().isa<MemRefType>())
    return emitOpError("dst must be a memref (writable target buffer)");

  return success();
}

//===----------------------------------------------------------------------===//
// AttnPrefillFusedOp
//===----------------------------------------------------------------------===//

LogicalResult AttnPrefillFusedOp::verify() {
  int64_t seqLen = getSeqLen();
  if (seqLen <= 0)
    return emitOpError("seq_len must be positive (got ") << seqLen << ")";

  // kv_cache must be a memref — it's the pre-allocated GDDR7 slab
  if (!getKvCache().getType().isa<MemRefType>())
    return emitOpError("kv_cache operand must be a memref (GDDR7-resident "
                       "KV slab)");

  return success();
}

//===----------------------------------------------------------------------===//
// KVCacheOp
//===----------------------------------------------------------------------===//

LogicalResult KVCacheOp::verify() {
  // buffer and out must have the same element type
  auto bufTy = getBuffer().getType().cast<MemRefType>();
  auto outTy = getOut().getType().cast<MemRefType>();
  if (bufTy.getElementType() != outTy.getElementType())
    return emitOpError("buffer and result element types must match");

  return success();
}
