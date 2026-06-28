//===- GenerateROCMPredicateKernel.cpp - unary predicate gpu kernel ------===//
//
// Expands `tessera_rocm.predicate` into a flat per-element unary predicate over
// f32 input with i8 (0/1) bool output; the `kind` StrAttr selects the test:
//
//   isnan    : x != x                  (unordered)
//   isinf    : |x| == +inf
//   isfinite : ordered AND |x| < +inf
//
// One thread per element. CPU analog: avx512_predicate_f32. All f32 in / i8 out.
//===----------------------------------------------------------------------===//

#include "TesseraROCM/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"

#include <limits>

using namespace mlir;

namespace {

static constexpr int64_t BD = 256;

void emitPredBody(OpBuilder &b, Location loc, gpu::GPUFuncOp f, StringRef kind) {
  Type f32 = b.getF32Type();
  Type i8 = b.getIntegerType(8);
  auto slt = arith::CmpIPredicate::slt;
  b.setInsertionPointToStart(&f.getBody().front());
  Value X = f.getArgument(0), O = f.getArgument(1), N = f.getArgument(2);

  Value bid = b.create<gpu::BlockIdOp>(loc, gpu::Dimension::x);
  Value tid = b.create<gpu::ThreadIdOp>(loc, gpu::Dimension::x);
  Value cBD = b.create<arith::ConstantIndexOp>(loc, BD);
  Value gid = b.create<arith::AddIOp>(loc, b.create<arith::MulIOp>(loc, bid, cBD),
                                      tid);
  Value inb = b.create<arith::CmpIOp>(loc, slt, gid, N);
  auto ifo = b.create<scf::IfOp>(loc, inb, /*withElse=*/false);
  b.setInsertionPointToStart(ifo.thenBlock());

  Value x = b.create<memref::LoadOp>(loc, X, ValueRange{gid});
  Value inf = b.create<arith::ConstantOp>(
      loc, f32, b.getF32FloatAttr(std::numeric_limits<float>::infinity()));
  Value ax = b.create<math::AbsFOp>(loc, x);
  Value pred;  // i1
  if (kind == "isnan") {
    pred = b.create<arith::CmpFOp>(loc, arith::CmpFPredicate::UNO, x, x);
  } else if (kind == "isinf") {
    pred = b.create<arith::CmpFOp>(loc, arith::CmpFPredicate::OEQ, ax, inf);
  } else {  // isfinite
    Value ord = b.create<arith::CmpFOp>(loc, arith::CmpFPredicate::ORD, x, x);
    Value lt = b.create<arith::CmpFOp>(loc, arith::CmpFPredicate::OLT, ax, inf);
    pred = b.create<arith::AndIOp>(loc, ord, lt);
  }
  Value byte = b.create<arith::ExtUIOp>(loc, i8, pred);
  b.create<memref::StoreOp>(loc, byte, O, ValueRange{gid});
  b.setInsertionPointToEnd(&f.getBody().front());
  b.create<gpu::ReturnOp>(loc);
}

struct GenerateROCMPredicateKernelPass
    : PassWrapper<GenerateROCMPredicateKernelPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(GenerateROCMPredicateKernelPass)

  StringRef getArgument() const final { return "generate-rocm-predicate-kernel"; }
  StringRef getDescription() const final {
    return "Expand a tessera_rocm.predicate directive into a flat unary "
           "predicate gpu kernel (isnan/isinf/isfinite, f32 in / i8 bool out)";
  }
  void getDependentDialects(DialectRegistry &registry) const final {
    registry.insert<gpu::GPUDialect, scf::SCFDialect, arith::ArithDialect,
                    math::MathDialect, memref::MemRefDialect>();
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();
    SmallVector<Operation *> directives;
    module.walk([&](Operation *op) {
      if (op->getName().getStringRef() == "tessera_rocm.predicate")
        directives.push_back(op);
    });
    for (Operation *op : directives) {
      auto nameAttr = op->getAttrOfType<StringAttr>("name");
      auto kindAttr = op->getAttrOfType<StringAttr>("kind");
      if (!nameAttr || !kindAttr) {
        op->emitError("tessera_rocm.predicate missing name/kind");
        return signalPassFailure();
      }
      OpBuilder b(module.getBodyRegion());
      b.setInsertionPointToEnd(module.getBody());
      Location loc = op->getLoc();
      std::string kname = nameAttr.getValue().str();
      Type f32 = b.getF32Type();
      Type i8 = b.getIntegerType(8);
      Type idxTy = b.getIndexType();
      auto memF32 = MemRefType::get({ShapedType::kDynamic}, f32);
      auto memI8 = MemRefType::get({ShapedType::kDynamic}, i8);
      auto fnTy = b.getFunctionType({memF32, memI8, idxTy}, {});
      auto gpuMod = b.create<gpu::GPUModuleOp>(loc, kname + "_mod");
      b.setInsertionPointToStart(&gpuMod.getBodyRegion().front());
      auto gpuFunc = b.create<gpu::GPUFuncOp>(loc, kname, fnTy);
      gpuFunc->setAttr(gpu::GPUDialect::getKernelFuncAttrName(), b.getUnitAttr());
      OpBuilder body(gpuFunc.getContext());
      emitPredBody(body, loc, gpuFunc, kindAttr.getValue());
      op->erase();
    }
  }
};

} // namespace

std::unique_ptr<mlir::Pass>
mlir::tessera_rocm::createGenerateROCMPredicateKernelPass() {
  return std::make_unique<GenerateROCMPredicateKernelPass>();
}
