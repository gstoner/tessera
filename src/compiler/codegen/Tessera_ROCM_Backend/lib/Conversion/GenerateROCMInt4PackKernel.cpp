//===- GenerateROCMInt4PackKernel.cpp - signed packed INT4 storage --------===//
//
// Expands tessera_rocm.int4_pack into a runtime-sized pack/unpack kernel or
// one of three physical packed-storage consumers:
//   relu          nibblewise signed ReLU, packed bytes in/out;
//   sparse_gather indexed logical i8 output loaded directly from packed codes;
//   cache_append  packed-byte append into a packed cache without repacking.
// Logical signed int4 values use int8 at the Graph/runtime boundary and two
// two's-complement nibbles per byte in terminal storage (low logical index in
// the low nibble). This is the generic compiled consumer of
// tessera.storage_pack={logical="int4", container="int8", factor=2,
// signedness="signed_twos_complement"}; WMMA remains a separate matrix consumer.

#include "TesseraROCM/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/STLExtras.h"

using namespace mlir;

namespace {

constexpr int64_t kBlockSize = 256;

void emitPackBody(OpBuilder &b, Location loc, gpu::GPUFuncOp function,
                  bool unpack) {
  b.setInsertionPointToStart(&function.getBody().front());
  Value input = function.getArgument(0);
  Value output = function.getArgument(1);
  Value logicalElements = function.getArgument(2);
  Value block = gpu::BlockIdOp::create(b, loc, gpu::Dimension::x);
  Value thread = gpu::ThreadIdOp::create(b, loc, gpu::Dimension::x);
  Value blockSize = arith::ConstantIndexOp::create(b, loc, kBlockSize);
  Value gid = arith::AddIOp::create(
      b, loc, arith::MulIOp::create(b, loc, block, blockSize), thread);
  Value zeroIndex = arith::ConstantIndexOp::create(b, loc, 0);
  Value oneIndex = arith::ConstantIndexOp::create(b, loc, 1);
  Value twoIndex = arith::ConstantIndexOp::create(b, loc, 2);
  Value zero = arith::ConstantIntOp::create(b, loc, 0, 8);
  Value mask = arith::ConstantIntOp::create(b, loc, 15, 8);
  Value sign = arith::ConstantIntOp::create(b, loc, 8, 8);
  Value shift = arith::ConstantIntOp::create(b, loc, 4, 8);

  if (!unpack) {
    Value packedElements = arith::DivUIOp::create(
        b, loc, arith::AddIOp::create(b, loc, logicalElements, oneIndex),
        twoIndex);
    Value inBounds = arith::CmpIOp::create(
        b, loc, arith::CmpIPredicate::ult, gid, packedElements);
    auto guarded = scf::IfOp::create(b, loc, inBounds, false);
    b.setInsertionPointToStart(guarded.thenBlock());
    Value lowIndex = arith::MulIOp::create(b, loc, gid, twoIndex);
    Value highIndex = arith::AddIOp::create(b, loc, lowIndex, oneIndex);
    Value low = memref::LoadOp::create(b, loc, input, ValueRange{lowIndex});
    low = arith::AndIOp::create(b, loc, low, mask);
    Value hasHigh = arith::CmpIOp::create(
        b, loc, arith::CmpIPredicate::ult, highIndex, logicalElements);
    auto highIf = scf::IfOp::create(b, loc, TypeRange{b.getI8Type()},
                                    hasHigh, true);
    {
      OpBuilder::InsertionGuard guard(b);
      b.setInsertionPointToStart(highIf.thenBlock());
      Value high =
          memref::LoadOp::create(b, loc, input, ValueRange{highIndex});
      high = arith::AndIOp::create(b, loc, high, mask);
      high = arith::ShLIOp::create(b, loc, high, shift);
      scf::YieldOp::create(b, loc, high);
      b.setInsertionPointToStart(highIf.elseBlock());
      scf::YieldOp::create(b, loc, zero);
    }
    Value packed = arith::OrIOp::create(b, loc, low, highIf.getResult(0));
    memref::StoreOp::create(b, loc, packed, output, ValueRange{gid});
  } else {
    Value inBounds = arith::CmpIOp::create(
        b, loc, arith::CmpIPredicate::ult, gid, logicalElements);
    auto guarded = scf::IfOp::create(b, loc, inBounds, false);
    b.setInsertionPointToStart(guarded.thenBlock());
    Value packedIndex = arith::DivUIOp::create(b, loc, gid, twoIndex);
    Value packed =
        memref::LoadOp::create(b, loc, input, ValueRange{packedIndex});
    Value odd = arith::CmpIOp::create(
        b, loc, arith::CmpIPredicate::ne,
        arith::RemUIOp::create(b, loc, gid, twoIndex), zeroIndex);
    Value high = arith::ShRUIOp::create(b, loc, packed, shift);
    Value nibble = arith::SelectOp::create(b, loc, odd, high, packed);
    nibble = arith::AndIOp::create(b, loc, nibble, mask);
    // Sign-extend four bits without relying on a target-specific i4 type.
    Value logical = arith::SubIOp::create(
        b, loc, arith::XOrIOp::create(b, loc, nibble, sign), sign);
    memref::StoreOp::create(b, loc, logical, output, ValueRange{gid});
  }

  b.setInsertionPointToEnd(&function.getBody().front());
  gpu::ReturnOp::create(b, loc);
}

Value emitSignedNibble(OpBuilder &b, Location loc, Value packed, Value logical,
                       Value twoIndex) {
  Value zeroIndex = arith::ConstantIndexOp::create(b, loc, 0);
  Value mask = arith::ConstantIntOp::create(b, loc, 15, 8);
  Value sign = arith::ConstantIntOp::create(b, loc, 8, 8);
  Value shift = arith::ConstantIntOp::create(b, loc, 4, 8);
  Value odd = arith::CmpIOp::create(
      b, loc, arith::CmpIPredicate::ne,
      arith::RemUIOp::create(b, loc, logical, twoIndex), zeroIndex);
  Value high = arith::ShRUIOp::create(b, loc, packed, shift);
  Value nibble = arith::SelectOp::create(b, loc, odd, high, packed);
  nibble = arith::AndIOp::create(b, loc, nibble, mask);
  return arith::SubIOp::create(
      b, loc, arith::XOrIOp::create(b, loc, nibble, sign), sign);
}

void emitPackedReluBody(OpBuilder &b, Location loc, gpu::GPUFuncOp function) {
  b.setInsertionPointToStart(&function.getBody().front());
  Value input = function.getArgument(0);
  Value output = function.getArgument(1);
  Value logicalElements = function.getArgument(2);
  Value block = gpu::BlockIdOp::create(b, loc, gpu::Dimension::x);
  Value thread = gpu::ThreadIdOp::create(b, loc, gpu::Dimension::x);
  Value blockSize = arith::ConstantIndexOp::create(b, loc, kBlockSize);
  Value gid = arith::AddIOp::create(
      b, loc, arith::MulIOp::create(b, loc, block, blockSize), thread);
  Value one = arith::ConstantIndexOp::create(b, loc, 1);
  Value two = arith::ConstantIndexOp::create(b, loc, 2);
  Value packedElements = arith::DivUIOp::create(
      b, loc, arith::AddIOp::create(b, loc, logicalElements, one), two);
  Value inBounds = arith::CmpIOp::create(
      b, loc, arith::CmpIPredicate::ult, gid, packedElements);
  auto guarded = scf::IfOp::create(b, loc, inBounds, false);
  b.setInsertionPointToStart(guarded.thenBlock());
  Value packed = memref::LoadOp::create(b, loc, input, ValueRange{gid});
  Value lowIndex = arith::MulIOp::create(b, loc, gid, two);
  Value highIndex = arith::AddIOp::create(b, loc, lowIndex, one);
  Value zero = arith::ConstantIntOp::create(b, loc, 0, 8);
  Value shift = arith::ConstantIntOp::create(b, loc, 4, 8);
  Value low = emitSignedNibble(b, loc, packed, lowIndex, two);
  Value lowPositive = arith::CmpIOp::create(
      b, loc, arith::CmpIPredicate::sgt, low, zero);
  low = arith::SelectOp::create(b, loc, lowPositive, low, zero);
  Value high = emitSignedNibble(b, loc, packed, highIndex, two);
  Value highPositive = arith::CmpIOp::create(
      b, loc, arith::CmpIPredicate::sgt, high, zero);
  high = arith::SelectOp::create(b, loc, highPositive, high, zero);
  Value hasHigh = arith::CmpIOp::create(
      b, loc, arith::CmpIPredicate::ult, highIndex, logicalElements);
  high = arith::SelectOp::create(b, loc, hasHigh, high, zero);
  Value repacked = arith::OrIOp::create(
      b, loc, low, arith::ShLIOp::create(b, loc, high, shift));
  memref::StoreOp::create(b, loc, repacked, output, ValueRange{gid});
  b.setInsertionPointToEnd(&function.getBody().front());
  gpu::ReturnOp::create(b, loc);
}

void emitSparseGatherBody(OpBuilder &b, Location loc,
                          gpu::GPUFuncOp function) {
  b.setInsertionPointToStart(&function.getBody().front());
  Value input = function.getArgument(0);
  Value indices = function.getArgument(1);
  Value output = function.getArgument(2);
  Value logicalElements = function.getArgument(3);
  Value outputElements = function.getArgument(4);
  Value block = gpu::BlockIdOp::create(b, loc, gpu::Dimension::x);
  Value thread = gpu::ThreadIdOp::create(b, loc, gpu::Dimension::x);
  Value blockSize = arith::ConstantIndexOp::create(b, loc, kBlockSize);
  Value gid = arith::AddIOp::create(
      b, loc, arith::MulIOp::create(b, loc, block, blockSize), thread);
  Value inBounds = arith::CmpIOp::create(
      b, loc, arith::CmpIPredicate::ult, gid, outputElements);
  auto guarded = scf::IfOp::create(b, loc, inBounds, false);
  b.setInsertionPointToStart(guarded.thenBlock());
  Value logicalI64 =
      memref::LoadOp::create(b, loc, indices, ValueRange{gid});
  Value logical =
      arith::IndexCastOp::create(b, loc, b.getIndexType(), logicalI64);
  Value valid = arith::CmpIOp::create(
      b, loc, arith::CmpIPredicate::ult, logical, logicalElements);
  auto validIf = scf::IfOp::create(b, loc, valid, false);
  b.setInsertionPointToStart(validIf.thenBlock());
  Value two = arith::ConstantIndexOp::create(b, loc, 2);
  Value packedIndex = arith::DivUIOp::create(b, loc, logical, two);
  Value packed =
      memref::LoadOp::create(b, loc, input, ValueRange{packedIndex});
  Value value = emitSignedNibble(b, loc, packed, logical, two);
  memref::StoreOp::create(b, loc, value, output, ValueRange{gid});
  b.setInsertionPointToEnd(&function.getBody().front());
  gpu::ReturnOp::create(b, loc);
}

void emitCacheAppendBody(OpBuilder &b, Location loc,
                         gpu::GPUFuncOp function) {
  b.setInsertionPointToStart(&function.getBody().front());
  Value update = function.getArgument(0);
  Value cache = function.getArgument(1);
  Value byteOffset = function.getArgument(2);
  Value byteCount = function.getArgument(3);
  Value block = gpu::BlockIdOp::create(b, loc, gpu::Dimension::x);
  Value thread = gpu::ThreadIdOp::create(b, loc, gpu::Dimension::x);
  Value blockSize = arith::ConstantIndexOp::create(b, loc, kBlockSize);
  Value gid = arith::AddIOp::create(
      b, loc, arith::MulIOp::create(b, loc, block, blockSize), thread);
  Value inBounds = arith::CmpIOp::create(
      b, loc, arith::CmpIPredicate::ult, gid, byteCount);
  auto guarded = scf::IfOp::create(b, loc, inBounds, false);
  b.setInsertionPointToStart(guarded.thenBlock());
  Value value = memref::LoadOp::create(b, loc, update, ValueRange{gid});
  Value target = arith::AddIOp::create(b, loc, byteOffset, gid);
  memref::StoreOp::create(b, loc, value, cache, ValueRange{target});
  b.setInsertionPointToEnd(&function.getBody().front());
  gpu::ReturnOp::create(b, loc);
}

struct GenerateROCMInt4PackKernel
    : PassWrapper<GenerateROCMInt4PackKernel, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(GenerateROCMInt4PackKernel)

  StringRef getArgument() const final {
    return "generate-rocm-int4-pack-kernel";
  }
  StringRef getDescription() const final {
    return "Generate signed INT4 pack/unpack and physical consumer kernels";
  }
  void getDependentDialects(DialectRegistry &registry) const final {
    registry.insert<gpu::GPUDialect, scf::SCFDialect, arith::ArithDialect,
                    memref::MemRefDialect>();
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();
    SmallVector<Operation *> directives;
    module.walk([&](Operation *op) {
      if (op->getName().getStringRef() == "tessera_rocm.int4_pack")
        directives.push_back(op);
    });
    for (Operation *op : directives) {
      auto name = op->getAttrOfType<StringAttr>("name");
      auto kind = op->getAttrOfType<StringAttr>("kind");
      if (!name || !kind ||
          !llvm::is_contained(
              ArrayRef<StringRef>{"pack", "unpack", "relu", "sparse_gather",
                                  "cache_append"},
              kind.getValue())) {
        op->emitError("tessera_rocm.int4_pack requires name and "
                      "kind=pack|unpack|relu|sparse_gather|cache_append");
        return signalPassFailure();
      }
      OpBuilder b(module.getBodyRegion());
      b.setInsertionPointToEnd(module.getBody());
      Location loc = op->getLoc();
      auto gpuModule =
          gpu::GPUModuleOp::create(b, loc, name.getValue().str() + "_mod");
      b.setInsertionPointToStart(&gpuModule.getBodyRegion().front());
      auto bytes = MemRefType::get({ShapedType::kDynamic}, b.getI8Type());
      FunctionType type;
      if (kind.getValue() == "sparse_gather") {
        auto indices =
            MemRefType::get({ShapedType::kDynamic}, b.getI64Type());
        type = b.getFunctionType(
            {bytes, indices, bytes, b.getIndexType(), b.getIndexType()},
            TypeRange{});
      } else if (kind.getValue() == "cache_append") {
        type = b.getFunctionType(
            {bytes, bytes, b.getIndexType(), b.getIndexType()}, TypeRange{});
      } else {
        type =
            b.getFunctionType({bytes, bytes, b.getIndexType()}, TypeRange{});
      }
      auto function =
          gpu::GPUFuncOp::create(b, loc, name.getValue(), type);
      function->setAttr(gpu::GPUDialect::getKernelFuncAttrName(),
                        b.getUnitAttr());
      OpBuilder body(function.getContext());
      if (kind.getValue() == "relu")
        emitPackedReluBody(body, loc, function);
      else if (kind.getValue() == "sparse_gather")
        emitSparseGatherBody(body, loc, function);
      else if (kind.getValue() == "cache_append")
        emitCacheAppendBody(body, loc, function);
      else
        emitPackBody(body, loc, function, kind.getValue() == "unpack");
      op->erase();
    }
  }
};

} // namespace

std::unique_ptr<mlir::Pass>
mlir::tessera_rocm::createGenerateROCMInt4PackKernelPass() {
  return std::make_unique<GenerateROCMInt4PackKernel>();
}
