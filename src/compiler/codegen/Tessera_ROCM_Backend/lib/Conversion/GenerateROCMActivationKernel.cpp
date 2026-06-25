//===- GenerateROCMActivationKernel.cpp - elementwise activation kernel ---===//
//
// Expands a `tessera_rocm.activation` directive into a flat elementwise gpu
// kernel applying a pointwise activation over N elements (one thread per
// element, strided grid):
//
//   gelu : 0.5·x·(1 + tanh(√(2/π)·(x + 0.044715·x³)))   (tanh approximation)
//   silu : x·σ(x) = x / (1 + exp(−x))
//   relu : max(x, 0)
//
// The standalone analog of the activations the GEMM fused epilogue applies
// in-register. Computes in f32 regardless of storage dtype; the transcendentals
// lower through convert-math-to-rocdl. N is a runtime index arg. Validated vs a
// numpy reference on gfx1151.
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

enum class Act { Gelu, Silu, Relu };

void emitActivationBody(OpBuilder &b, Location loc, gpu::GPUFuncOp f,
                        Type storeTy, Act act) {
  Type f32 = b.getF32Type();
  bool isF32 = storeTy.isF32();
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

  Value raw = b.create<memref::LoadOp>(loc, X, ValueRange{gid});
  Value x = isF32 ? raw : b.create<arith::ExtFOp>(loc, f32, raw);
  Value y;
  if (act == Act::Relu) {
    Value z = b.create<arith::ConstantOp>(loc, f32, b.getF32FloatAttr(0.0));
    y = b.create<arith::MaximumFOp>(loc, x, z);
  } else if (act == Act::Silu) {
    Value one = b.create<arith::ConstantOp>(loc, f32, b.getF32FloatAttr(1.0));
    Value e = b.create<math::ExpOp>(loc, b.create<arith::NegFOp>(loc, x));
    y = b.create<arith::DivFOp>(loc, x,
                                b.create<arith::AddFOp>(loc, one, e));
  } else {  // gelu (tanh approximation)
    Value half = b.create<arith::ConstantOp>(loc, f32, b.getF32FloatAttr(0.5f));
    Value one = b.create<arith::ConstantOp>(loc, f32, b.getF32FloatAttr(1.0f));
    Value c044 =
        b.create<arith::ConstantOp>(loc, f32, b.getF32FloatAttr(0.044715f));
    Value csqrt = b.create<arith::ConstantOp>(
        loc, f32, b.getF32FloatAttr(0.7978845608028654f));
    Value x2 = b.create<arith::MulFOp>(loc, x, x);
    Value x3 = b.create<arith::MulFOp>(loc, x2, x);
    Value inner = b.create<arith::AddFOp>(
        loc, x, b.create<arith::MulFOp>(loc, c044, x3));
    inner = b.create<arith::MulFOp>(loc, csqrt, inner);
    Value t = b.create<math::TanhOp>(loc, inner);
    Value oneP = b.create<arith::AddFOp>(loc, one, t);
    y = b.create<arith::MulFOp>(loc, b.create<arith::MulFOp>(loc, half, x), oneP);
  }
  Value sv = isF32 ? y : b.create<arith::TruncFOp>(loc, storeTy, y);
  b.create<memref::StoreOp>(loc, sv, O, ValueRange{gid});

  b.setInsertionPointToEnd(&f.getBody().front());
  b.create<gpu::ReturnOp>(loc);
}

struct GenerateROCMActivationKernelPass
    : PassWrapper<GenerateROCMActivationKernelPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(GenerateROCMActivationKernelPass)

  StringRef getArgument() const final {
    return "generate-rocm-activation-kernel";
  }
  StringRef getDescription() const final {
    return "Expand a tessera_rocm.activation directive into a flat elementwise "
           "gelu/silu/relu gpu kernel (compiler-generated)";
  }
  void getDependentDialects(DialectRegistry &registry) const final {
    registry.insert<gpu::GPUDialect, scf::SCFDialect, arith::ArithDialect,
                    math::MathDialect, memref::MemRefDialect>();
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();
    SmallVector<Operation *> directives;
    module.walk([&](Operation *op) {
      if (op->getName().getStringRef() == "tessera_rocm.activation")
        directives.push_back(op);
    });
    for (Operation *op : directives) {
      auto nameAttr = op->getAttrOfType<StringAttr>("name");
      if (!nameAttr) {
        op->emitError("tessera_rocm.activation missing name");
        return signalPassFailure();
      }
      StringRef kindStr = "gelu";
      if (auto a = op->getAttrOfType<StringAttr>("kind"))
        kindStr = a.getValue();
      Act act;
      if (kindStr == "gelu")
        act = Act::Gelu;
      else if (kindStr == "silu")
        act = Act::Silu;
      else if (kindStr == "relu")
        act = Act::Relu;
      else {
        op->emitError("generate-rocm-activation-kernel: kind must be gelu, "
                      "silu, or relu (got '")
            << kindStr << "')";
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
          op->emitError("generate-rocm-activation-kernel: dtype must be f32, "
                        "f16, or bf16 (got '")
              << dt << "')";
          return signalPassFailure();
        }
      }

      auto gpuMod = b.create<gpu::GPUModuleOp>(loc, kname + "_mod");
      b.setInsertionPointToStart(&gpuMod.getBodyRegion().front());
      Type idxTy = b.getIndexType();
      auto memTy = MemRefType::get({ShapedType::kDynamic}, storeTy);
      // (X, O : memref<?xstore>, N : index)
      auto fnTy = b.getFunctionType({memTy, memTy, idxTy}, {});
      auto gpuFunc = b.create<gpu::GPUFuncOp>(loc, kname, fnTy);
      gpuFunc->setAttr(gpu::GPUDialect::getKernelFuncAttrName(),
                       b.getUnitAttr());
      OpBuilder body(gpuFunc.getContext());
      emitActivationBody(body, loc, gpuFunc, storeTy, act);
      op->erase();
    }
  }
};

} // namespace

std::unique_ptr<mlir::Pass>
mlir::tessera_rocm::createGenerateROCMActivationKernelPass() {
  return std::make_unique<GenerateROCMActivationKernelPass>();
}
