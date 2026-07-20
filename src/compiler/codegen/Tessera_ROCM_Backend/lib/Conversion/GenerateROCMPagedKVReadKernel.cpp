//===- GenerateROCMPagedKVReadKernel.cpp - direct paged-KV gather ----------===//

#include "TesseraROCM/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Pass/Pass.h"

using namespace mlir;

namespace {

static constexpr int64_t BlockSize = 256;

void emitBody(OpBuilder &b, Location loc, gpu::GPUFuncOp function) {
  b.setInsertionPointToStart(&function.getBody().front());
  Value pages = function.getArgument(0);
  Value table = function.getArgument(1);
  Value output = function.getArgument(2);
  Value pageSize = function.getArgument(5);
  Value heads = function.getArgument(6);
  Value dim = function.getArgument(7);
  Value start = function.getArgument(8);
  Value tokens = function.getArgument(9);

  auto ci = [&](int64_t value) {
    return b.create<arith::ConstantIndexOp>(loc, value);
  };
  auto mul = [&](Value lhs, Value rhs) {
    return b.create<arith::MulIOp>(loc, lhs, rhs);
  };
  auto add = [&](Value lhs, Value rhs) {
    return b.create<arith::AddIOp>(loc, lhs, rhs);
  };

  Value block = b.create<gpu::BlockIdOp>(loc, gpu::Dimension::x);
  Value thread = b.create<gpu::ThreadIdOp>(loc, gpu::Dimension::x);
  Value linear = add(mul(block, ci(BlockSize)), thread);
  Value total = mul(mul(tokens, heads), dim);
  Value inBounds = b.create<arith::CmpIOp>(
      loc, arith::CmpIPredicate::slt, linear, total);
  auto guarded = b.create<scf::IfOp>(loc, inBounds, false);
  b.setInsertionPointToStart(guarded.thenBlock());

  Value d = b.create<arith::RemUIOp>(loc, linear, dim);
  Value tokenHead = b.create<arith::DivUIOp>(loc, linear, dim);
  Value head = b.create<arith::RemUIOp>(loc, tokenHead, heads);
  Value token = b.create<arith::DivUIOp>(loc, tokenHead, heads);
  Value logical = add(start, token);
  Value logicalPage = b.create<arith::DivUIOp>(loc, logical, pageSize);
  Value pageOffset = b.create<arith::RemUIOp>(loc, logical, pageSize);
  Value physical32 = b.create<memref::LoadOp>(loc, table,
                                              ValueRange{logicalPage});
  Value physical = b.create<arith::IndexCastUIOp>(
      loc, b.getIndexType(), physical32);
  Value pageIndex = add(
      mul(add(mul(add(mul(physical, pageSize), pageOffset), heads), head), dim),
      d);
  Value value = b.create<memref::LoadOp>(loc, pages, ValueRange{pageIndex});
  b.create<memref::StoreOp>(loc, value, output, ValueRange{linear});

  b.setInsertionPointToEnd(&function.getBody().front());
  b.create<gpu::ReturnOp>(loc);
}

struct GenerateROCMPagedKVReadKernelPass
    : PassWrapper<GenerateROCMPagedKVReadKernelPass,
                  OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(
      GenerateROCMPagedKVReadKernelPass)

  StringRef getArgument() const final {
    return "generate-rocm-paged-kv-read-kernel";
  }
  StringRef getDescription() const final {
    return "Expand tessera_rocm.paged_kv_read into a direct f32 gather kernel";
  }
  void getDependentDialects(DialectRegistry &registry) const final {
    registry.insert<gpu::GPUDialect, scf::SCFDialect, arith::ArithDialect,
                    memref::MemRefDialect>();
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();
    SmallVector<Operation *> directives;
    module.walk([&](Operation *op) {
      if (op->getName().getStringRef() == "tessera_rocm.paged_kv_read")
        directives.push_back(op);
    });
    for (Operation *op : directives) {
      auto name = op->getAttrOfType<StringAttr>("name");
      auto storage = op->getAttrOfType<StringAttr>("storage");
      auto tableStorage = op->getAttrOfType<StringAttr>("table_storage");
      auto route = op->getAttrOfType<StringAttr>("route");
      if (!name || !storage || storage.getValue() != "f32" ||
          !tableStorage || tableStorage.getValue() != "i32" || !route ||
          route.getValue() != "direct") {
        op->emitError("paged-KV generator requires name, f32 storage, i32 "
                      "table_storage, and route=direct");
        return signalPassFailure();
      }
      OpBuilder b(module.getBodyRegion());
      b.setInsertionPointToEnd(module.getBody());
      Location loc = op->getLoc();
      std::string kernelName = name.getValue().str();
      auto gpuModule =
          b.create<gpu::GPUModuleOp>(loc, kernelName + "_mod");
      b.setInsertionPointToStart(&gpuModule.getBodyRegion().front());
      Type index = b.getIndexType();
      auto pages = MemRefType::get({ShapedType::kDynamic}, b.getF32Type());
      auto table = MemRefType::get({ShapedType::kDynamic}, b.getI32Type());
      SmallVector<Type> arguments{pages, table, pages, index, index, index,
                                  index, index, index, index};
      auto gpuFunction = b.create<gpu::GPUFuncOp>(
          loc, kernelName, b.getFunctionType(arguments, {}));
      gpuFunction->setAttr(gpu::GPUDialect::getKernelFuncAttrName(),
                           b.getUnitAttr());
      OpBuilder body(gpuFunction.getContext());
      emitBody(body, loc, gpuFunction);
      op->erase();
    }
  }
};

} // namespace

std::unique_ptr<mlir::Pass>
mlir::tessera_rocm::createGenerateROCMPagedKVReadKernelPass() {
  return std::make_unique<GenerateROCMPagedKVReadKernelPass>();
}
