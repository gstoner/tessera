//===- GenerateROCMSiluMulKernel.cpp - SwiGLU silu_mul kernel -------------===//
//
// Expands a `tessera_rocm.silu_mul` directive into a flat 2-operand elementwise
// gpu kernel computing the SwiGLU gate-multiply over N elements (one thread per
// element, strided grid):
//
//   silu_mul(a, b) = silu(a) * b = (a / (1 + exp(-a))) * b
//
// The standalone analog of the gate-multiply the fused SwiGLU MLP applies
// in-register. Computes in f32 regardless of storage dtype; the transcendental
// lowers through convert-math-to-rocdl. N is a runtime index arg. Validated vs
// the numpy reference on gfx1151.
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
#include "mlir/Pass/Pass.h"

using namespace mlir;

namespace {

static constexpr int64_t BD = 256;

void emitSiluMulBody(OpBuilder &b, Location loc, gpu::GPUFuncOp f,
                     Type storeTy) {
  Type f32 = b.getF32Type();
  bool isF32 = storeTy.isF32();
  auto slt = arith::CmpIPredicate::slt;

  b.setInsertionPointToStart(&f.getBody().front());
  Value A = f.getArgument(0), B = f.getArgument(1), O = f.getArgument(2),
        N = f.getArgument(3);

  Value bid = b.create<gpu::BlockIdOp>(loc, gpu::Dimension::x);
  Value tid = b.create<gpu::ThreadIdOp>(loc, gpu::Dimension::x);
  Value cBD = b.create<arith::ConstantIndexOp>(loc, BD);
  Value gid = b.create<arith::AddIOp>(loc, b.create<arith::MulIOp>(loc, bid, cBD),
                                      tid);
  Value inb = b.create<arith::CmpIOp>(loc, slt, gid, N);
  auto ifo = b.create<scf::IfOp>(loc, inb, /*withElse=*/false);
  b.setInsertionPointToStart(ifo.thenBlock());

  Value ra = b.create<memref::LoadOp>(loc, A, ValueRange{gid});
  Value rb = b.create<memref::LoadOp>(loc, B, ValueRange{gid});
  Value a = isF32 ? ra : b.create<arith::ExtFOp>(loc, f32, ra);
  Value bb = isF32 ? rb : b.create<arith::ExtFOp>(loc, f32, rb);
  // silu(a) = a / (1 + exp(-a)); then * b.
  Value one = b.create<arith::ConstantOp>(loc, f32, b.getF32FloatAttr(1.0));
  Value e = b.create<math::ExpOp>(loc, b.create<arith::NegFOp>(loc, a));
  Value silu = b.create<arith::DivFOp>(loc, a,
                                       b.create<arith::AddFOp>(loc, one, e));
  Value y = b.create<arith::MulFOp>(loc, silu, bb);

  Value sv = isF32 ? y : b.create<arith::TruncFOp>(loc, storeTy, y);
  b.create<memref::StoreOp>(loc, sv, O, ValueRange{gid});

  b.setInsertionPointToEnd(&f.getBody().front());
  b.create<gpu::ReturnOp>(loc);
}

struct GenerateROCMSiluMulKernelPass
    : PassWrapper<GenerateROCMSiluMulKernelPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(GenerateROCMSiluMulKernelPass)

  StringRef getArgument() const final { return "generate-rocm-silu-mul-kernel"; }
  StringRef getDescription() const final {
    return "Expand a tessera_rocm.silu_mul directive into a flat 2-operand "
           "elementwise SwiGLU gate-multiply gpu kernel (compiler-generated)";
  }
  void getDependentDialects(DialectRegistry &registry) const final {
    registry.insert<gpu::GPUDialect, scf::SCFDialect, arith::ArithDialect,
                    math::MathDialect, memref::MemRefDialect>();
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();
    SmallVector<Operation *> directives;
    module.walk([&](Operation *op) {
      if (op->getName().getStringRef() == "tessera_rocm.silu_mul")
        directives.push_back(op);
    });
    for (Operation *op : directives) {
      auto nameAttr = op->getAttrOfType<StringAttr>("name");
      if (!nameAttr) {
        op->emitError("tessera_rocm.silu_mul missing name");
        return signalPassFailure();
      }
      OpBuilder b(module.getBodyRegion());
      b.setInsertionPointToEnd(module.getBody());
      Location loc = op->getLoc();
      std::string kname = nameAttr.getValue().str();

      Type storeTy = b.getF32Type();
      if (auto a = op->getAttrOfType<StringAttr>("dtype")) {
        StringRef dt = a.getValue();
        if (dt == "f16" || dt == "float16")
          storeTy = b.getF16Type();
        else if (dt == "bf16" || dt == "bfloat16")
          storeTy = b.getBF16Type();
        else if (dt != "f32" && dt != "float32") {
          op->emitError("generate-rocm-silu-mul-kernel: dtype must be f32, "
                        "f16, or bf16 (got '")
              << dt << "')";
          return signalPassFailure();
        }
      }

      auto gpuMod = b.create<gpu::GPUModuleOp>(loc, kname + "_mod");
      b.setInsertionPointToStart(&gpuMod.getBodyRegion().front());
      Type idxTy = b.getIndexType();
      auto memTy = MemRefType::get({ShapedType::kDynamic}, storeTy);
      // (A, B, O : memref<?xstore>, N : index)
      auto fnTy = b.getFunctionType({memTy, memTy, memTy, idxTy}, {});
      auto gpuFunc = b.create<gpu::GPUFuncOp>(loc, kname, fnTy);
      gpuFunc->setAttr(gpu::GPUDialect::getKernelFuncAttrName(),
                       b.getUnitAttr());
      OpBuilder body(gpuFunc.getContext());
      emitSiluMulBody(body, loc, gpuFunc, storeTy);
      op->erase();
    }
  }
};

} // namespace

std::unique_ptr<mlir::Pass>
mlir::tessera_rocm::createGenerateROCMSiluMulKernelPass() {
  return std::make_unique<GenerateROCMSiluMulKernelPass>();
}
