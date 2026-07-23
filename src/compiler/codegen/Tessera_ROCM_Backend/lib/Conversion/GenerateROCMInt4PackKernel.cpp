//===- GenerateROCMInt4PackKernel.cpp - signed packed INT4 storage --------===//
//
// Expands tessera_rocm.int4_pack into a runtime-sized pack or unpack kernel.
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

struct GenerateROCMInt4PackKernel
    : PassWrapper<GenerateROCMInt4PackKernel, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(GenerateROCMInt4PackKernel)

  StringRef getArgument() const final {
    return "generate-rocm-int4-pack-kernel";
  }
  StringRef getDescription() const final {
    return "Generate signed two's-complement INT4 terminal pack/unpack kernels";
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
          (kind.getValue() != "pack" && kind.getValue() != "unpack")) {
        op->emitError("tessera_rocm.int4_pack requires name and kind=pack|unpack");
        return signalPassFailure();
      }
      OpBuilder b(module.getBodyRegion());
      b.setInsertionPointToEnd(module.getBody());
      Location loc = op->getLoc();
      auto gpuModule =
          gpu::GPUModuleOp::create(b, loc, name.getValue().str() + "_mod");
      b.setInsertionPointToStart(&gpuModule.getBodyRegion().front());
      auto bytes = MemRefType::get({ShapedType::kDynamic}, b.getI8Type());
      auto type =
          b.getFunctionType({bytes, bytes, b.getIndexType()}, TypeRange{});
      auto function =
          gpu::GPUFuncOp::create(b, loc, name.getValue(), type);
      function->setAttr(gpu::GPUDialect::getKernelFuncAttrName(),
                        b.getUnitAttr());
      OpBuilder body(function.getContext());
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
