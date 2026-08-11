//===- GenerateROCMSolverIFTKernel.cpp - implicit-function VJP -----------===//

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

static constexpr int64_t kBlockSize = 256;

static void emitBody(OpBuilder &builder, Location loc, gpu::GPUFuncOp function) {
  builder.setInsertionPointToStart(&function.getBody().front());
  Value theta = function.getArgument(0);
  Value solution = function.getArgument(1);
  Value cotangent = function.getArgument(2);
  Value residual = function.getArgument(3);
  Value linearSolution = function.getArgument(4);
  Value parameterCotangent = function.getArgument(5);
  Value count = function.getArgument(6);

  Value block = builder.create<gpu::BlockIdOp>(loc, gpu::Dimension::x);
  Value thread = builder.create<gpu::ThreadIdOp>(loc, gpu::Dimension::x);
  Value width = builder.create<arith::ConstantIndexOp>(loc, kBlockSize);
  Value index = builder.create<arith::AddIOp>(
      loc, builder.create<arith::MulIOp>(loc, block, width), thread);
  Value inBounds = builder.create<arith::CmpIOp>(
      loc, arith::CmpIPredicate::slt, index, count);
  auto ifOp = builder.create<scf::IfOp>(loc, inBounds, /*withElse=*/false);
  builder.setInsertionPointToStart(ifOp.thenBlock());

  Value t = builder.create<memref::LoadOp>(loc, theta, ValueRange{index});
  Value x = builder.create<memref::LoadOp>(loc, solution, ValueRange{index});
  Value c = builder.create<memref::LoadOp>(loc, cotangent, ValueRange{index});
  Value r = builder.create<arith::SubFOp>(
      loc, builder.create<arith::MulFOp>(loc, x, x), t);
  Value two = builder.create<arith::ConstantOp>(
      loc, builder.getF32Type(), builder.getF32FloatAttr(2.0f));
  Value lambda = builder.create<arith::DivFOp>(
      loc, c, builder.create<arith::MulFOp>(loc, two, x));
  builder.create<memref::StoreOp>(loc, r, residual, ValueRange{index});
  builder.create<memref::StoreOp>(loc, lambda, linearSolution,
                                  ValueRange{index});
  builder.create<memref::StoreOp>(loc, lambda, parameterCotangent,
                                  ValueRange{index});
  builder.setInsertionPointAfter(ifOp);
  builder.create<gpu::ReturnOp>(loc);
}

struct GenerateROCMSolverIFTKernelPass
    : PassWrapper<GenerateROCMSolverIFTKernelPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(
      GenerateROCMSolverIFTKernelPass)

  StringRef getArgument() const final {
    return "generate-rocm-solver-ift-kernel";
  }
  StringRef getDescription() const final {
    return "Generate the promoted gfx1151 matrix-free implicit-function VJP";
  }
  void getDependentDialects(DialectRegistry &registry) const final {
    registry.insert<gpu::GPUDialect, scf::SCFDialect, arith::ArithDialect,
                    memref::MemRefDialect>();
  }

  void runOnOperation() override {
    SmallVector<Operation *> directives;
    getOperation().walk([&](Operation *op) {
      if (op->getName().getStringRef() == "tessera_rocm.solver_ift")
        directives.push_back(op);
    });
    for (Operation *op : directives) {
      auto name = op->getAttrOfType<StringAttr>("name");
      auto model = op->getAttrOfType<StringAttr>("residual_model");
      auto solver = op->getAttrOfType<StringAttr>("linear_solver");
      auto productMode = op->getAttrOfType<StringAttr>("product_mode");
      auto dtype = op->getAttrOfType<StringAttr>("dtype");
      auto hash = op->getAttrOfType<StringAttr>("artifact_hash");
      auto residualDigest = op->getAttrOfType<StringAttr>("residual_digest");
      if (!name || !model || model.getValue() != "diagonal_sqrt_v1" ||
          !solver || solver.getValue() != "diagonal_matrix_free_v1" ||
          (productMode && productMode.getValue() != "jvp" &&
                          productMode.getValue() != "vjp") ||
          !dtype || dtype.getValue() != "f32" || !hash ||
          hash.getValue().size() != 64 || !residualDigest ||
          residualDigest.getValue().size() != 64) {
        op->emitError("generate-rocm-solver-ift-kernel: unsupported or incomplete contract");
        return signalPassFailure();
      }

      OpBuilder builder(getOperation().getBodyRegion());
      builder.setInsertionPointToEnd(getOperation().getBody());
      Location loc = op->getLoc();
      auto gpuModule =
          builder.create<gpu::GPUModuleOp>(loc, name.getValue().str() + "_mod");
      builder.setInsertionPointToStart(&gpuModule.getBodyRegion().front());
      Type f32 = builder.getF32Type();
      Type index = builder.getIndexType();
      Type buffer = MemRefType::get({ShapedType::kDynamic}, f32);
      auto functionType = builder.getFunctionType(
          {buffer, buffer, buffer, buffer, buffer, buffer, index}, {});
      auto function = builder.create<gpu::GPUFuncOp>(
          loc, name.getValue(), functionType);
      function->setAttr(gpu::GPUDialect::getKernelFuncAttrName(),
                        builder.getUnitAttr());
      OpBuilder body(function.getContext());
      emitBody(body, loc, function);
      op->erase();
    }
  }
};

} // namespace

std::unique_ptr<mlir::Pass>
mlir::tessera_rocm::createGenerateROCMSolverIFTKernelPass() {
  return std::make_unique<GenerateROCMSolverIFTKernelPass>();
}
