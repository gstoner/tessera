//===- GenerateROCMTridiagonalKernel.cpp - cooperative LDS PCR ----------===//

#include "TesseraROCM/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "llvm/Support/MathExtras.h"

using namespace mlir;

namespace {

static Value ldsBuffer(gpu::GPUFuncOp function, Location loc, int64_t n,
                       Type element) {
  auto space = gpu::AddressSpaceAttr::get(function.getContext(),
                                          gpu::AddressSpace::Workgroup);
  return function.addWorkgroupAttribution(
      MemRefType::get({n}, element, MemRefLayoutAttrInterface(), space), loc);
}

static void emitPCR(OpBuilder &builder, Location loc, gpu::GPUFuncOp function,
                    int64_t n) {
  Type f32 = builder.getF32Type();
  Type f64 = builder.getF64Type();
  builder.setInsertionPointToStart(&function.getBody().front());
  Value dl = function.getArgument(0), diagonal = function.getArgument(1);
  Value du = function.getArgument(2), rhs = function.getArgument(3);
  Value output = function.getArgument(4);
  Value status = function.getArgument(5);
  Value lane = builder.create<gpu::ThreadIdOp>(loc, gpu::Dimension::x);
  Value system = builder.create<gpu::BlockIdOp>(loc, gpu::Dimension::x);
  Value cN = builder.create<arith::ConstantIndexOp>(loc, n);
  Value global = builder.create<arith::AddIOp>(
      loc, builder.create<arith::MulIOp>(loc, system, cN), lane);
  Value zero = builder.create<arith::ConstantOp>(
      loc, f64, builder.getF64FloatAttr(0.0));
  Value one = builder.create<arith::ConstantOp>(
      loc, f64, builder.getF64FloatAttr(1.0));
  Value falseValue = builder.create<arith::ConstantIntOp>(loc, 0, 1);
  Value trueValue = builder.create<arith::ConstantIntOp>(loc, 1, 1);
  Value bad = falseValue;
  auto validPivot = [&](Value pivot) -> Value {
    // finite x has x-x == 0; NaN and either infinity produce unordered NaN.
    Value delta = builder.create<arith::SubFOp>(loc, pivot, pivot);
    Value finite = builder.create<arith::CmpFOp>(
        loc, arith::CmpFPredicate::OEQ, delta, zero);
    Value nonzero = builder.create<arith::CmpFOp>(
        loc, arith::CmpFPredicate::ONE, pivot, zero);
    return builder.create<arith::AndIOp>(loc, finite, nonzero);
  };

  Value a[2] = {ldsBuffer(function, loc, n, f64),
                ldsBuffer(function, loc, n, f64)};
  Value b[2] = {ldsBuffer(function, loc, n, f64),
                ldsBuffer(function, loc, n, f64)};
  Value c[2] = {ldsBuffer(function, loc, n, f64),
                ldsBuffer(function, loc, n, f64)};
  Value r[2] = {ldsBuffer(function, loc, n, f64),
                ldsBuffer(function, loc, n, f64)};
  auto loadWide = [&](Value source, Value index) -> Value {
    Value loaded = builder.create<memref::LoadOp>(loc, source, index);
    return builder.create<arith::ExtFOp>(loc, f64, loaded);
  };
  builder.create<memref::StoreOp>(loc, loadWide(dl, lane), a[0], lane);
  builder.create<memref::StoreOp>(
      loc, loadWide(diagonal, lane), b[0], lane);
  builder.create<memref::StoreOp>(loc, loadWide(du, lane), c[0], lane);
  builder.create<memref::StoreOp>(loc, loadWide(rhs, global), r[0], lane);
  builder.create<gpu::BarrierOp>(loc);

  unsigned source = 0;
  for (int64_t stride = 1; stride < n; stride <<= 1) {
    unsigned target = 1 - source;
    Value cStride = builder.create<arith::ConstantIndexOp>(loc, stride);
    Value cLast = builder.create<arith::ConstantIndexOp>(loc, n - 1);
    Value hasLeft = builder.create<arith::CmpIOp>(
        loc, arith::CmpIPredicate::uge, lane, cStride);
    Value rawLeft = builder.create<arith::SubIOp>(loc, lane, cStride);
    Value left = builder.create<arith::SelectOp>(
        loc, hasLeft, rawLeft,
        builder.create<arith::ConstantIndexOp>(loc, 0));
    Value rawRight = builder.create<arith::AddIOp>(loc, lane, cStride);
    Value hasRight = builder.create<arith::CmpIOp>(
        loc, arith::CmpIPredicate::ult, rawRight, cN);
    Value right =
        builder.create<arith::SelectOp>(loc, hasRight, rawRight, cLast);

    Value ai = builder.create<memref::LoadOp>(loc, a[source], lane);
    Value bi = builder.create<memref::LoadOp>(loc, b[source], lane);
    Value ci = builder.create<memref::LoadOp>(loc, c[source], lane);
    Value ri = builder.create<memref::LoadOp>(loc, r[source], lane);
    Value leftB = builder.create<memref::LoadOp>(loc, b[source], left);
    Value rightB = builder.create<memref::LoadOp>(loc, b[source], right);
    Value leftValid = validPivot(leftB);
    Value rightValid = validPivot(rightB);
    Value badLeft = builder.create<arith::AndIOp>(
        loc, hasLeft,
        builder.create<arith::XOrIOp>(loc, leftValid, trueValue));
    Value badRight = builder.create<arith::AndIOp>(
        loc, hasRight,
        builder.create<arith::XOrIOp>(loc, rightValid, trueValue));
    bad = builder.create<arith::OrIOp>(
        loc, bad, builder.create<arith::OrIOp>(loc, badLeft, badRight));
    Value safeLeft = builder.create<arith::SelectOp>(
        loc, builder.create<arith::AndIOp>(loc, hasLeft, leftValid), leftB,
        one);
    Value safeRight = builder.create<arith::SelectOp>(
        loc, builder.create<arith::AndIOp>(loc, hasRight, rightValid), rightB,
        one);
    Value alphaRaw = builder.create<arith::NegFOp>(
        loc, builder.create<arith::DivFOp>(loc, ai, safeLeft));
    Value gammaRaw = builder.create<arith::NegFOp>(
        loc, builder.create<arith::DivFOp>(loc, ci, safeRight));
    Value alpha =
        builder.create<arith::SelectOp>(loc, hasLeft, alphaRaw, zero);
    Value gamma =
        builder.create<arith::SelectOp>(loc, hasRight, gammaRaw, zero);
    auto fused = [&](Value center, Value leftValue,
                     Value rightValue) -> Value {
      return builder.create<arith::AddFOp>(
          loc,
          builder.create<arith::AddFOp>(
              loc, center,
              builder.create<arith::MulFOp>(loc, alpha, leftValue)),
          builder.create<arith::MulFOp>(loc, gamma, rightValue));
    };
    Value nextA = builder.create<arith::MulFOp>(
        loc, alpha, builder.create<memref::LoadOp>(loc, a[source], left));
    Value nextB = fused(
        bi, builder.create<memref::LoadOp>(loc, c[source], left),
        builder.create<memref::LoadOp>(loc, a[source], right));
    Value nextC = builder.create<arith::MulFOp>(
        loc, gamma, builder.create<memref::LoadOp>(loc, c[source], right));
    Value nextR = fused(
        ri, builder.create<memref::LoadOp>(loc, r[source], left),
        builder.create<memref::LoadOp>(loc, r[source], right));
    builder.create<memref::StoreOp>(loc, nextA, a[target], lane);
    builder.create<memref::StoreOp>(loc, nextB, b[target], lane);
    builder.create<memref::StoreOp>(loc, nextC, c[target], lane);
    builder.create<memref::StoreOp>(loc, nextR, r[target], lane);
    builder.create<gpu::BarrierOp>(loc);
    source = target;
  }
  Value finalB = builder.create<memref::LoadOp>(loc, b[source], lane);
  Value finalValid = validPivot(finalB);
  bad = builder.create<arith::OrIOp>(
      loc, bad, builder.create<arith::XOrIOp>(loc, finalValid, trueValue));
  Value safeFinal =
      builder.create<arith::SelectOp>(loc, finalValid, finalB, one);
  Value solution = builder.create<arith::DivFOp>(
      loc, builder.create<memref::LoadOp>(loc, r[source], lane), safeFinal);
  Value stored = builder.create<arith::SelectOp>(
      loc, bad,
      builder.create<arith::ConstantOp>(loc, f32,
                                        builder.getF32FloatAttr(0.0)),
      builder.create<arith::TruncFOp>(loc, f32, solution));
  builder.create<memref::StoreOp>(loc, stored, output, global);
  Value statusCode = builder.create<arith::SelectOp>(
      loc, bad, builder.create<arith::ConstantIntOp>(loc, 4, 32),
      builder.create<arith::ConstantIntOp>(loc, 0, 32));
  builder.create<memref::StoreOp>(loc, statusCode, status, global);
  builder.create<gpu::ReturnOp>(loc);
}

struct GenerateROCMTridiagonalKernelPass
    : PassWrapper<GenerateROCMTridiagonalKernelPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(
      GenerateROCMTridiagonalKernelPass)
  StringRef getArgument() const final {
    return "generate-rocm-tridiagonal-kernel";
  }
  StringRef getDescription() const final {
    return "Expand a gfx1151 tridiagonal Target record into cooperative LDS PCR";
  }
  void getDependentDialects(DialectRegistry &registry) const final {
    registry.insert<gpu::GPUDialect, arith::ArithDialect,
                    memref::MemRefDialect>();
  }
  void runOnOperation() override {
    ModuleOp module = getOperation();
    SmallVector<Operation *> directives;
    module.walk([&](Operation *op) {
      if (op->getName().getStringRef() == "tessera_rocm.tridiagonal_solve")
        directives.push_back(op);
    });
    for (Operation *op : directives) {
      auto name = op->getAttrOfType<StringAttr>("name");
      auto arch = op->getAttrOfType<StringAttr>("arch");
      auto hash = op->getAttrOfType<StringAttr>("artifact_hash");
      auto batch = op->getAttrOfType<IntegerAttr>("batch");
      auto systemSize = op->getAttrOfType<IntegerAttr>("system_size");
      auto workgroup = op->getAttrOfType<IntegerAttr>("workgroup_size");
      auto algorithm = op->getAttrOfType<StringAttr>("algorithm");
      auto dtype = op->getAttrOfType<StringAttr>("dtype");
      auto accum = op->getAttrOfType<StringAttr>("accum");
      auto statusAbi = op->getAttrOfType<StringAttr>("status_abi");
      if (!name || !arch || arch.getValue() != "gfx1151" || !hash ||
          hash.getValue().size() != 64 || !batch || batch.getInt() <= 0 ||
          !systemSize || systemSize.getInt() <= 0 ||
          systemSize.getInt() > 256 ||
          !llvm::isPowerOf2_64(systemSize.getInt()) || !workgroup ||
          workgroup.getInt() != systemSize.getInt() || !algorithm ||
          algorithm.getValue() != "cooperative_lds_pcr_v1" || !dtype ||
          dtype.getValue() != "f32" || !accum || accum.getValue() != "f64" ||
          !statusAbi || statusAbi.getValue() != "per_equation_i32_v1") {
        op->emitError("invalid gfx1151 cooperative tridiagonal contract");
        return signalPassFailure();
      }
      OpBuilder builder(module.getBodyRegion());
      builder.setInsertionPointToEnd(module.getBody());
      Location loc = op->getLoc();
      auto gpuModule = builder.create<gpu::GPUModuleOp>(
          loc, name.getValue().str() + "_mod");
      builder.setInsertionPointToStart(&gpuModule.getBodyRegion().front());
      auto memref = MemRefType::get({ShapedType::kDynamic}, builder.getF32Type());
      auto statusMemref =
          MemRefType::get({ShapedType::kDynamic}, builder.getI32Type());
      auto type = builder.getFunctionType(
          {memref, memref, memref, memref, memref, statusMemref}, {});
      auto function =
          builder.create<gpu::GPUFuncOp>(loc, name.getValue(), type);
      function->setAttr(gpu::GPUDialect::getKernelFuncAttrName(),
                        builder.getUnitAttr());
      function->setAttr("tessera.schedule_hash", hash);
      function->setAttr("tessera.block_size", workgroup);
      function->setAttr("tessera.grid_size", batch);
      OpBuilder body(function.getContext());
      emitPCR(body, loc, function, systemSize.getInt());
      op->erase();
    }
  }
};

} // namespace

std::unique_ptr<mlir::Pass>
mlir::tessera_rocm::createGenerateROCMTridiagonalKernelPass() {
  return std::make_unique<GenerateROCMTridiagonalKernelPass>();
}
