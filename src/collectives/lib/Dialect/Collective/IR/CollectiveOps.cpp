#include "tessera/Dialect/Collective/IR/CollectiveDialect.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;

#define GET_TYPEDEF_CLASSES
#include "CollectiveTypes.cpp.inc"

namespace tessera::collective {

void TesseraCollectiveDialect::initialize() {
  addTypes<
#define GET_TYPEDEF_LIST
#include "CollectiveTypes.cpp.inc"
      >();
  addOperations<
#define GET_OP_LIST
#include "CollectiveOps.cpp.inc"
      >();
}

void registerCollectiveDialect(DialectRegistry &registry) {
  registry.insert<TesseraCollectiveDialect>();
}

static ShapedType shapedPayload(Type type) {
  if (auto future = dyn_cast<FutureType>(type))
    type = future.getValueType();
  return dyn_cast<ShapedType>(type);
}

static LogicalResult verifyCollective(Operation *op, Value input, Value result,
                                      bool scalesAxis, bool multiply,
                                      bool requiresReduction) {
  auto meshAxis = op->getAttrOfType<StringAttr>("mesh_axis");
  auto tensorAxis = op->getAttrOfType<IntegerAttr>("tensor_axis");
  auto reduction = op->getAttrOfType<StringAttr>("reduction");
  if (!meshAxis || meshAxis.getValue().empty())
    return op->emitOpError("requires a non-empty mesh_axis");
  if (!tensorAxis || tensorAxis.getInt() < 0)
    return op->emitOpError("requires a non-negative tensor_axis");
  if (!reduction)
    return op->emitOpError("requires a reduction contract");
  if (auto dtype = op->getAttrOfType<StringAttr>("dtype");
      dtype && dtype.getValue().empty())
    return op->emitOpError("dtype must be non-empty when present");
  if (auto chunkBytes = op->getAttrOfType<IntegerAttr>("chunk_bytes");
      chunkBytes && chunkBytes.getInt() <= 0)
    return op->emitOpError("chunk_bytes must be positive when present");
  StringRef reductionName = reduction.getValue();
  if (requiresReduction) {
    if (reductionName != "sum" && reductionName != "max" &&
        reductionName != "min" && reductionName != "prod" &&
        reductionName != "mean")
      return op->emitOpError(
          "reduction must be sum|mean|max|min|prod for a reducing collective");
  } else if (reductionName != "none") {
    return op->emitOpError("non-reducing collective requires reduction=none");
  }

  uint64_t worldSize = 0;
  if (auto size = op->getAttrOfType<IntegerAttr>("world_size")) {
    if (size.getInt() <= 0)
      return op->emitOpError("world_size must be positive when present");
    worldSize = static_cast<uint64_t>(size.getInt());
  }

  ShapedType inputType = shapedPayload(input.getType());
  ShapedType outputType = shapedPayload(result.getType());
  if (!inputType || !outputType)
    return success();
  if (inputType.getElementType() != outputType.getElementType() ||
      inputType.getRank() != outputType.getRank())
    return op->emitOpError("input and awaited output must preserve rank and element type");
  int64_t axis = tensorAxis.getInt();
  if (axis >= inputType.getRank())
    return op->emitOpError("tensor_axis is outside the payload rank");
  for (int64_t dim = 0; dim < inputType.getRank(); ++dim) {
    int64_t in = inputType.getDimSize(dim);
    int64_t out = outputType.getDimSize(dim);
    if (ShapedType::isDynamic(in) || ShapedType::isDynamic(out))
      continue;
    if (dim != axis || !scalesAxis) {
      if (in != out)
        return op->emitOpError("non-transport dimensions must be preserved");
      continue;
    }
    if (!worldSize)
      continue;
    int64_t expected = multiply ? in * static_cast<int64_t>(worldSize)
                                : in / static_cast<int64_t>(worldSize);
    if (!multiply && in % static_cast<int64_t>(worldSize) != 0)
      return op->emitOpError(
          "scatter axis extent must divide evenly by world_size");
    if (out != expected)
      return op->emitOpError("awaited output shape disagrees with world_size");
  }
  return success();
}

LogicalResult AllReduceOp::verify() {
  return verifyCollective(getOperation(), getInput(), getResult(), false,
                          false, true);
}
LogicalResult ReduceScatterOp::verify() {
  return verifyCollective(getOperation(), getInput(), getResult(), true,
                          false, true);
}
LogicalResult AllGatherOp::verify() {
  return verifyCollective(getOperation(), getInput(), getResult(), true, true,
                          false);
}
LogicalResult AllToAllOp::verify() {
  if (failed(verifyCollective(getOperation(), getInput(), getResult(), false,
                              false, false)))
    return failure();
  auto worldSize = getOperation()->getAttrOfType<IntegerAttr>("world_size");
  auto inputType = shapedPayload(getInput().getType());
  if (!worldSize || !inputType)
    return success();
  int64_t extent = inputType.getDimSize(getTensorAxis());
  if (!ShapedType::isDynamic(extent) && extent % worldSize.getInt() != 0)
    return emitOpError(
        "exchange axis extent must divide evenly by world_size");
  return success();
}

static FailureOr<int64_t> oneSidedDTypeBytes(Operation *op, StringRef dtype,
                                             Type elementType) {
  int64_t bytes = 0;
  bool matches = false;
  if (dtype == "f32") {
    bytes = 4;
    matches = elementType.isF32();
  } else if (dtype == "f16") {
    bytes = 2;
    matches = elementType.isF16();
  } else if (dtype == "bf16") {
    bytes = 2;
    matches = elementType.isBF16();
  } else if (dtype == "i8") {
    bytes = 1;
    matches = elementType.isInteger(8);
  } else {
    op->emitOpError("one-sided dtype must be f32|f16|bf16|i8");
    return failure();
  }
  if (!matches) {
    op->emitOpError("one-sided dtype disagrees with the source element type");
    return failure();
  }
  return bytes;
}

LogicalResult WindowRegisterOp::verify() {
  if (getWindowId().empty())
    return emitOpError("requires a non-empty window_id");
  int64_t bytes = (*this)->getAttrOfType<IntegerAttr>("bytes").getInt();
  if (bytes <= 0)
    return emitOpError("requires a positive byte extent");
  if (getWindow().getType().getValueType() != getBuffer().getType())
    return emitOpError("window payload type must match the registered buffer");
  auto memref = dyn_cast<MemRefType>(getBuffer().getType());
  if (memref && memref.hasStaticShape()) {
    int64_t elementBits = memref.getElementTypeBitWidth();
    if (elementBits <= 0 || elementBits % 8 != 0)
      return emitOpError("registered buffer requires byte-addressable elements");
    int64_t available = memref.getNumElements() * (elementBits / 8);
    if (bytes > available)
      return emitOpError("byte extent exceeds the registered buffer");
  }
  return success();
}

LogicalResult PutSignalOp::verify() {
  int64_t count = (*this)->getAttrOfType<IntegerAttr>("count").getInt();
  int64_t peer = (*this)->getAttrOfType<IntegerAttr>("peer").getInt();
  int64_t offset =
      (*this)->getAttrOfType<IntegerAttr>("peer_window_offset").getInt();
  int64_t signalIndex =
      (*this)->getAttrOfType<IntegerAttr>("signal_index").getInt();
  int64_t context =
      (*this)->getAttrOfType<IntegerAttr>("rma_context").getInt();
  if (count <= 0)
    return emitOpError("requires a positive element count");
  if (peer < 0 || offset < 0 || signalIndex < 0 || context < 0)
    return emitOpError(
        "peer, window offset, signal index, and context must be non-negative");
  auto source = dyn_cast<MemRefType>(getSource().getType());
  if (!source)
    return emitOpError("source must be a memref");
  FailureOr<int64_t> elementBytes = oneSidedDTypeBytes(
      getOperation(), getDtype(), source.getElementType());
  if (failed(elementBytes))
    return failure();
  if (offset % *elementBytes != 0)
    return emitOpError("peer window offset must align to dtype width");
  if (source.hasStaticShape() &&
      count > source.getNumElements())
    return emitOpError("element count exceeds the source buffer");
  auto peerWindow = getPeerWindow().getType().getValueType();
  auto peerMemref = dyn_cast<MemRefType>(peerWindow);
  if (peerMemref && peerMemref.hasStaticShape()) {
    int64_t peerBits = peerMemref.getElementTypeBitWidth();
    int64_t peerBytes = peerMemref.getNumElements() * (peerBits / 8);
    int64_t transferEnd = offset + count * *elementBytes;
    if (transferEnd > peerBytes)
      return emitOpError("put extent exceeds the peer window");
  }
  return success();
}

static LogicalResult verifySignalFields(Operation *op, int64_t peer,
                                        int64_t signalIndex,
                                        int64_t context) {
  if (peer < 0 || signalIndex < 0 || context < 0)
    return op->emitOpError(
        "peer, signal index, and context must be non-negative");
  return success();
}

LogicalResult SignalOp::verify() {
  return verifySignalFields(
      getOperation(), (*this)->getAttrOfType<IntegerAttr>("peer").getInt(),
      (*this)->getAttrOfType<IntegerAttr>("signal_index").getInt(),
      (*this)->getAttrOfType<IntegerAttr>("rma_context").getInt());
}

LogicalResult WaitSignalOp::verify() {
  int64_t operationCount =
      (*this)->getAttrOfType<IntegerAttr>("operation_count").getInt();
  if (operationCount <= 0)
    return emitOpError("operation_count must be positive");
  return verifySignalFields(
      getOperation(), (*this)->getAttrOfType<IntegerAttr>("peer").getInt(),
      (*this)->getAttrOfType<IntegerAttr>("signal_index").getInt(),
      (*this)->getAttrOfType<IntegerAttr>("rma_context").getInt());
}

LogicalResult AwaitOp::verify() {
  if (getFuture().getType().getValueType() != getValue().getType())
    return emitOpError("result type must match the future payload type");
  return success();
}

LogicalResult ShardViewOp::verify() {
  if (getMeshAxis().empty())
    return emitOpError("requires a non-empty mesh_axis");
  if (auto shaped = dyn_cast<ShapedType>(getInput().getType()))
    if (getDim() >= static_cast<uint64_t>(shaped.getRank()))
      return emitOpError("dim is outside the input rank");
  if (getShard().getType().getValueType() != getInput().getType())
    return emitOpError("shard payload type must match the input type");
  return success();
}

LogicalResult MaterializeShardOp::verify() {
  if (getShard().getType().getValueType() != getValue().getType())
    return emitOpError("materialized value type must match the shard payload");
  return success();
}

LogicalResult QoSLimitOp::verify() {
  if (getMaxInflight() <= 0)
    return emitOpError("max_inflight must be positive");
  return success();
}

} // namespace tessera::collective

#define GET_OP_CLASSES
#include "CollectiveOps.cpp.inc"
#include "CollectiveDialect.cpp.inc"
